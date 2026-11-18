from __future__ import annotations

import json
import re
from time import monotonic, sleep
from typing import Any

from src.agent.proxy_client import ProxyClient

# LLM seats: the actor, the one-time request reader, item and rank checks, the decline audit, shopper lines, question plans
ACTOR = ["anthropic/claude-haiku-5.5", "openai/gpt-6-luna", "anthropic/claude-haiku-5.5"]
READER = ["anthropic/claude-sonnet-5.5", "openai/gpt-6-luna"]
CHECKER = ["anthropic/claude-haiku-5.5", "anthropic/claude-sonnet-5.5"]
AUDITOR = ["anthropic/claude-haiku-5.5", "anthropic/claude-sonnet-5.5"]
LISTENER = ["anthropic/claude-haiku-5.5", "anthropic/claude-sonnet-5.5"]
PLANNER = ["anthropic/claude-haiku-5.5", "anthropic/claude-sonnet-5.5"]
# our ask engine: the one ranking call per episode, and the switch back to the planner above
ASK_SEAT = ["anthropic/claude-haiku-5.5", "google/gemini-3.8-flash"]
OUR_ASK = True
ASK_FIRST_PICKS = 2
ASK_SURE = 0.1
ASK_MENU = 24
ASK_SAMPLES = 3
ASK_RANKINGS = 2
ASK_CHARS = 70

# time (seconds)
DEADLINE = 265.0
CLOSING = 70.0
LAST_CALL = 28.0
CLOCK_STEP = 30
CH_SYNONYM_MIN = 4
# observation sizes shown to the actor
VIEW_CHARS = 16000
SEARCH_TEXT = 240
RECORD_TEXT = 700
# LLM spend in USD
SOFT_SPEND = 0.03
SPEND_CAP = 0.045
TURN_LIMIT = 80
# request pacing and retries
PACE_GAP = 0.15
SLOW_GAP = 1.0
SLOW_FOR = 30.0
DOWN_AFTER = 45.0
MODEL_TRIES = 2
WORLD_TRIES = 5
# what the code may do on its own
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
REVEAL_ROUNDS = 3
REVEAL_PAGES = 8
REVEAL_PROBES = 5
REVEAL_READS = 12
REVEAL_READS_MANY = 6
REVEAL_CHECKS = 3
LINK_WORDS = {"and", "or", "the", "for", "with", "of", "a", "an", "in", "on", "to", "by", "from", "at", "as", "is", "are", "be",
              "this", "that", "it", "its", "your", "our", "you", "all", "any", "not", "no"}
ASK_AFTER_SETS = 2
ASK_GRACE_STEPS = 3
ASK_AFTER_STEPS = 5

ACT_RULES = """Goal: as one shopper's assistant in a simulated store, acting only through the supplied tools, bring the errand to a finish correct in every respect. Scoring is 1 or 0, and 1 requires all of this: the item ordered satisfies each requirement the shopper wrote, each hidden need the shopper reveals on being asked, and each change announced along the way; quantity and budget are correct; the cheapest (or best) item is chosen whenever the request calls for it; and every procedural rule of the request is respected. When nothing in the store satisfies everything, the correct finish is a decline performed exactly the way the request prescribes.

1. Know how the store behaves.
- Each tool call spends one step, even a failed or invalid one, and steps are limited. Thinking is free.
- Several tool calls may share a reply when none needs another's result (opening a few listings' details, say); they are run in sequence. Orders and declines always travel alone.
- A details tool that accepts a LIST of listing ids shows all of their full records for ONE step (within its item limit): use it whenever two or more listings are opened.
- Tool, argument and result names change from store to store. Read each tool's description and JSON schema and use precisely the names and permitted values found there. Learn from the first results which tool reveals which facts (price, stock, detailed properties).
- Early on, discover which category labels this store attaches to listings of the requested kind. Among neighbouring labels, the correct one is the label matching the request's wording as a whole; search and buy within it.
- Stores search differently; infer this one's habits from the first results. Rows may be sorted best match first or by a fixed id; a keyword search may insist on all words (an extra word shrinks the total) or settle for any word (an extra word grows the total, with listings matching more words ranked higher); maybe only a handful of rows or pages can be read; a filter may work in a single mode only (keyword mode ignoring the category or price filter, filter mode ignoring the words). Weigh a reported total against the rows in view: a larger one means the fitting or cheapest listing could be on a later page or beyond the visible ones. Where all words are needed, narrow with more precise words (or with the category and price filters under a lowered price limit) till the total fits the visible pages. Queries stay short (one to three words) and open with the most distinctive words: certain stores look at only the first word or two and discard the rest, so a long query might really search only its opening words (compare the total and rows of a longer query with a shorter one that begins identically; the notes spell out what that shows). Where all words are needed, a required value followed by the item kind ("blue chair", value leading) brings up precisely the listings holding that value. Where any word matches, common words (the item kind, a unit) catch most of the shelf and bury everything else: the distinctive word or figure of a required value by itself ("herringbone", "30m", "0.4") brings up only the listings holding it, so search it by itself and in the alternative spellings listings use ("30 m", "3 atm"), read each page of that search when its total fits them, and pair the item kind with values only to lift the nearest matches to the top. Keyword searches may also match category names and property values. Notes after each search sum up what the totals reveal about this store's search.

2. Learn the shopper's hidden needs.
- The shopper holds unwritten needs and discloses one only when a message asks about exactly that property. Whatever a message asks FIRST is answered most dependably, so ask about one property per message, two at most with the likelier first: a brief, plain, friendly question that names the property (one or two sample values are fine). No instructions, lists, labels, positions or rules.
- Replies like "don't mind", "no preference" or "anything is fine" leave that property open. A preference or limit the shopper states is a hard requirement from then on, worded exactly as said, numbers included.
- Planned questions are sent on your behalf once the details of two or three listings of the requested kind are open, so open them early. Replies arrive in the conversation with their meaning attached; a revealed need binds as firmly as a stated one: search again with it alongside the stated values.
- When asking yourself: first open the details of two or three listings, then ask about what the story points to (whom it is for, how it should look, where or how it is used), then about listing properties that differ between candidates. Never ask about what the request already decides, and never go past the message budget.

3. Keep the rules of procedure.
- The request may prescribe how to work: how many listings to view or compare, what to verify, what to tell the shopper, what a remark must contain, how to decline. Such rules are graded like requirements; carry out each one literally and in full. "Look at" or "open" N listings asks for N listings that fit completely, every one of them opened through the details tool.
- Questions the shopper puts to you (confirming a change, for example) are answered for you while messages remain; the change itself is binding.
- Mere wishes ("ideally", "would be nice", "if possible") are not requirements: they only break ties between items already meeting everything and never override a cheapest rule.
- When the request leaves a point only the shopper can settle (no price limit, or two wishes that might not both be met and the shopper must rank them), ask before you choose.

4. Follow changes as they come.
- The shopper may alter things: a new budget (higher or lower), a requirement dropped, swapped or added, a price increase, an item selling out. One line may hold several changes, and such lines can appear in any tool result (as an extra field holding sentences) or come back with an order attempt. From then on each change is binding: re-examine the pick and the backups, and when the cheapest or best is wanted, choose again among the new acceptable set.
- A REFUSED order does not end the episode; the cart is untouched. Read the refusal, apply every change in it, take out of the cart whatever no longer fits (a sold-out item, say), add the next verified fit, look at its stock and price again, and send the order once more.
- Stock and prices also change silently, so re-check both just before each order attempt; an item whose current price times quantity is above the current budget no longer fits.

5. Handle several products.
- Some requests want several different products, each with its own quantity, requirements and hidden needs. Handle every one: search each category, verify a fit per product, and buy them in one order with one line for each product at its quantity. When a cart tool takes a list of lines, it may swap the entire cart for that list, so each such call has to include all products' lines. Cheapest or best is decided per product.
- A requirement may give its value by pointing at another listing or page ("the colour of the jumpsuit sold by X"): find that listing first (search for what identifies it, such as its shop or brand plus its kind), open its record, copy the value out of the named field, and then search the requested product with that value.

6. Mind the cart.
- Some cart tools ADD units to a line already held rather than setting them: putting in 1 more of a held line makes 2. Never put in a line the cart already holds; to change units, remove or add the difference. Each cart call is followed by a note on what the code counts in the cart.

7. Judge each listing (this decides the score).
- The grader looks up each requirement in the listing field that carries its property, comparing the value's own words as this store spells them there. Any field whose name expresses that property qualifies regardless of spelling: separators, a prefix, a shortened or lengthened name, or a familiar alternative name ("neckline" for a collar type, "color family" for a colour, "fit" for a pants fit type, "dial size" for a watch dial size). If there is a field bearing the property's exact name, that field settles it, and so does one bearing that name with a single extra word in front (pants length for length, clothing material for material): the requirement holds as soon as either of them writes the value. Field names that differ only by separators (size class, size_class) are one and the same field: the requirement holds when either one writes the value. Without such fields the alternative field settles it; where not a single field names the property, the value may come from the options list or the title. Descriptions are never evidence.
- Values are compared word by word as the store writes them: case, separators, singular or plural and spelling variants make no difference; a narrower value that contains the required one satisfies it ("rtx 3060 12gb" for rtx 3060, "80% cotton" for cotton, "forest green" for green); one entry of a list is sufficient, even where the shopper insists on that value alone: a field offering several variants that include it meets it. Synonyms, abbreviations, related words and shades ("solid" for plain, "win11" for windows 11, ivory or cream for off-white, "tech wood" for technical wood, "embroidered" for embroidery) satisfy a value only while no listing in the category writes the value itself in that field: if other listings say "matte" for their finish, a "satin" finish does not satisfy matte. A value combined with another by a plus ("wool + nylon") does not satisfy either part when the field elsewhere holds the other part alone ("nylon"). FIELD EVIDENCE lines, where included, are derived from this store's records and tell which situation applies. A different value of the same property fails (quartz is not mechanical, rosewood is not maple), and so does a word that merely resembles it. A value that pretends to cover every value of the property ("universal", "all seasons", "suitable for all occasions") names none and fails a required value; a list that names the required value as well still satisfies it. A field naming another value does not satisfy the required one just because the two are often paired or one hints at the other ("slip-on" is not "lace-up", a "pullover" style is not "zip-up"); only a shade normally sold under the required colour, or a narrower value containing it, qualifies.
- A field describing some other property is not the requirement's property (a strap length is not a strap type, a lining material is not the material, a length is not a style): it can never stand in where the category's listings carry the property's own field.
- Nothing counts except what the shopper stated or revealed. Never rule out a listing over something the shopper never asked for: an age group, gender, size, occasion, condition, quality or use guessed from the description or the story. Treat the story as context; when it seems relevant, put a question about that property to the shopper rather than guess. Requirements come from the shopper's replies, never from your own questions.
- Numbers are compared as amounts once units are converted (4.5 cm is 45 mm); a limit includes its edge; a range passes only if the whole range meets the limit. If the title is the evidence because no field names the property, an exact required value counts as met when the title gives its words and figures in that order, even followed by more detail (9' x 13' does not show 9' x 12', while 9' x 12' 4" does).
- Title figures ("55 x 94 Inch", "117 x 228 cm") do not reveal which dimension each is: never reject a listing, or leave it unopened, because of them; check the fields shown in its details.
- Closed lists ("A or B and nothing else") and exclusions ("leave out anything with X") must be respected.
- The listing's category alone settles the requested kind of product: for the grader, a listing placed in the requested category IS that kind even when its title or description mentions some other object (a case, a container, a set, a refill, an accessory), so never pass over, reject or downgrade it for what it looks like; weigh only the requirements, price and stock. No property has to restate the category name. Where the store labels its categories, the item must belong to the one whose label suits the request's wording taken whole (a request for "A or B" suits a category "A & B", not the narrower neighbour "B"). Where the request wants goods on hand, it has to show stock, and price times quantity must not exceed the budget (equal passes). A price is in the budget's unit unless the record names a different one (45 against a budget of 1000 US cents is 45 cents); rescale neither of them.
- Declining is rarely correct. If no listing satisfies the strict wording yet one satisfies every requirement under these rules (a narrower value, a field going by another name, or a synonym or shade that no category listing contradicts by writing the value itself), that listing is the fit; decline only if the search genuinely turned up nothing satisfying every requirement under these rules.

8. Pick the cheapest or the best when asked.
- For the cheapest, what gets ordered must be the lowest-priced item of ALL items that satisfy every requirement; one fitting item does not settle it. Where the keyword search needs all words and matches property values, the value-and-kind search returns the candidates holding the value: go through all its pages and open them in price order; a category listing under a price limit reveals the prices across the whole category but not their values: use it to learn which cheaper rows exist (drop the limit until the total fits), then examine those rows' properties. Search each required value together with the item kind as a brief query, value leading ("blue chair", "30m watch"), and read all its pages if its total fits the readable pages (if not, add a second required value or a narrower kind word); do likewise with other names of the kind ("stool", "bench") and with the value by itself; where a value cannot be searched (a figure the store strips), also search the product line of every fit or near-fit found, its maker plus one distinctive title word as two words, all pages: other sizes and colours of a line are separate listings sharing its property values and may be cheaper; sort the visible candidates by price yourself and open their details starting from the cheapest; the first one whose property fields satisfy everything, every cheaper one having been opened and rejected, is the answer. Hold the next fitting ones as backups.
- For the best by a measure (most, largest, longest, highest...), compare ALL fitting items on that measure likewise and take the extreme. An item giving no value for the measure cannot win.

9. Finish cleanly.
- Maintain a ranked list of verified fits with a backup at least: the first pick may sell out at checkout.
- Before each order attempt: the cart contains precisely one line for each product asked for (a single line for a single product), each being the chosen item at the quantity required, with nothing more; since your final cart change you have read the chosen item's stock and price again, finding it available and inside the budget.
- Decline only after a thorough search finds nothing satisfying every requirement, obeying the request's decline rule to the letter. If substitutes may not be offered, the decline remark names, describes or points to no listing: no ids, titles or prices.
- Budget the steps: send the first order attempt while seven or more steps remain, leaving room to handle a refusal. Once 7 steps are left, only the cart, stock-check and finishing tools stay usable. Decide once; no flipping the cart back and forth.

Return: every reply carries a tool call or more, with brief text stating the requirements now, the verified fits with price and stock, the next move."""

READ_REQUEST = """Goal: read the request and tool list below and reply with exactly ONE compact JSON object (no indentation), nothing else.
1. Use these keys:

"product": the sort of product wanted, in the request's wording (several products: all of them, linked by "and").
"items": an entry per DIFFERENT product to buy, in order: {"product": its kind as the request phrases it, "quantity": how many units of a single listing of it (same reading as "quantity" below), "requirements": the firm requirements on it (same reading as "requirements" below)}. One product means one entry.
"quantity": integer units of a single listing to buy; 1 unless the request wants more. Goods sold as one piece (a pair of shoes, a set) count as 1. Only an explicit wish for several units raises it ("I need three of them", "order 2 packs"); the number of people who will use or share it, or how many old ones there used to be, belongs to the story, so the count stays 1.
"requirements": a list of strings, a string for each firm requirement on the product (with several products: those of every product, each in the form "property (product): constraint"), each in the form "property: constraint" carrying exactly the value, limit, exclusion or closed list the shopper gives. Price, stock, quantity, budgets and plain wishes stay out, as does a repeat of the product kind or category: words that are part of the category's name (a season, occasion or audience inside it) are covered by the category, not a requirement of their own. A requirement pairs a product property with the value the request insists on ("at least 4 drawers", "the pattern has to be herringbone", "the occasion has to be party", "with a mechanical movement"). The story is never a requirement: the person it is for, an event where it gets worn or used ("to wear at his wedding", "for the staff party"), the place, time or way of use, how many people share it, or that an old one is replaced ("a new watch" means a replacement, not a condition). An occasion, use or audience counts as a requirement only where the request turns it into a property of the item itself ("the occasion has to be business", "a blazer made for business"). The value of a requirement consists of words or figures the request itself puts down as the item's value; any quality, reason or wish coming from the story (slim enough for a case, natural and breathable, covers the whole drop, works with what I already have) is merely a hint.
"price_cap": the largest total price, as the very number written in the request and in its own unit (do not convert: "3050 US cents" yields 3050), or null if the request names none.
"currency": the unit the request prices in, or null.
"in_stock_only": true if only items on hand or in stock will do.
"priority": "cheapest" only if the request clearly wants the cheapest or lowest-priced fitting item; "best" only if it clearly wants the best by some measure; "none" in every other case. Remarks that money is short or every penny counts do not ask for the cheapest; they just account for the cap.
"priority_detail": the measure when "priority" is "best"; otherwise "".
"message_budget": messages the shopper allows you, an integer, or null if unstated.
"step_budget": moves or steps the request permits, an integer, or null if unstated.
"figures": {"price_cap": the request's words that set the price limit, "message_budget": the words that set the message limit, "step_budget": the words that set the step limit}, each copied exactly, "" where the request sets none.
"fit_with": each product the request says has to work with, fit or suit another product picked for the same order, as {"product": the product that has to fit, worded as in "items", "with": the product it has to fit, worded as in "items", "quote": that sentence of the request, copied exactly}. Empty if none.
"procedures": each step the request makes the assistant take with the shopper that needs a message of its own and is not a question about the products (a go-ahead on the basket before ordering, a report before acting), as {"quote": that sentence of the request, copied exactly, "needs_yes": true when the shopper's agreement has to come before the assistant acts, "final_cart": true when it is about the finished basket or cart}. Asking about the products, the people they are for or anything else worth knowing is never such a step, even where the request invites questions; nor are rules on how to choose, what to buy or how to deliver. Empty if none.
"rules": every procedural rule the request gives the assistant (how many listings to look through, what to verify or report, how to decline, what to avoid, what to include), copied word for word. What the shopper will do personally is story.
"soft_wants": things the shopper would like but does not strictly need (ideally, preferably, would be nice, if possible). They must not appear in "requirements".
"ask_first": what only the shopper can answer, to be asked before choosing: the budget when no limit is set, which of two wishes weighs more when the request admits both may not be possible (both wishes then go in soft_wants and not in requirements), or a wish the request says to check with the shopper before choosing (a go-ahead on the basket or the order belongs under "procedures", not here). Empty if none.
"hints": story details pointing at some product property without giving its value (what it is used for or used with, where, by whom, what it has to fit, match or withstand), each as "property (product): the detail in a few words" ("property: detail" when only one product is wanted), the property named the way listings would most likely name it. If a detail could concern several properties, give the likeliest first and the runner-up after a slash ("colour / style (chair): must suit a room with one painted wall"): anything the item has to suit, match or sit well with (an outfit, a wall, a room, things used together with it) and anything about how it looks points first at its colour, then at its pattern or style. Leave out a detail that merely justifies a requirement already stated (why the stated material or size is needed). A job, the weather, travel, a schedule, or a place where nobody wears or uses the item is mere background and never a hint; the same goes for a purpose every listing of that kind serves (storage for a dresser, warmth for a coat). Empty if none.
No sentence of the request is left unplaced: each becomes a requirement, the priority, a soft want, a rule, a budget, a hint, or plain story. If you cannot tell whether a sentence is a rule, file it under "rules".
"references": requirements whose value the request supplies only by pointing to ANOTHER listing or page ("a colour other than the one on Z's page", "the colour of the jumpsuit X sells", "within 5 of the figure Y's page gives"), each as "property (product): as shown on <how the request names that listing>" (or "property (product): different from the one shown on <how the request names that listing>" if the value has to differ from that listing's); also add each of them to "requirements" in that same form. Empty if none.
"reference_targets": one object for each entry in "references", describing the listing pointed to: {"property": the property concerned, "product": the product concerned, "seller": the shop or brand the request names for that listing, copied exactly, "kind": what kind of product that listing is, "details": [any further value the request gives for that listing, brief and worded as the request words it ("4gb", "blue", "net weight 167.262186 g")], "relation": "same", "differ" (NOT that listing's value: other than, different from, set apart from), "counterpart" (compatible with, goes with or pairs with that listing's value), "within", "at most" or "at least" (how this value compares with that listing's), "tolerance": the figure for "within", otherwise null, "unit": the unit, otherwise ""}. Empty if none.
"decline_no_names": true if the request's decline rule bans offering, naming or suggesting substitutes or alternatives; false if it wants you to name something; null if it has no such rule.
"roles": an object mapping each role to an exact tool name from the list, or null where none suits: "search", "details", "view", "message", "cart", "cart_view", "order", "decline". "message" writes to the shopper; "cart" puts things into or takes them out of the cart; "order" buys or checks out; "decline" closes the errand with no purchase.
Return: just that JSON object.
"""

GO_AHEAD = """Goal: a shopping assistant put the final basket to the shopper before ordering it. Read the shopper's reply and return exactly ONE JSON object and nothing else: {"answer": "change"} when the reply asks for something in the basket to be different (another product, quantity or property, a new price limit); {"answer": "no"} when the reply plainly refuses the order or says not to buy; {"answer": "yes"} for anything else, since agreement, thanks, an acknowledgement or a remark that asks for no change all let the basket stand."""

STEP_CHECK = """Goal: a shopper's request gave a shopping assistant the rules below. Mark the rules that make the assistant send the shopper a message of its own that is not a question about the products, such as getting the shopper's go-ahead on the basket before ordering. Reply with exactly ONE JSON object and nothing else: {"steps": [the numbers of those rules]}, an empty list when none does."""

ASK_RANK = """Goal: guess what a shopper has left unsaid. The shopper wrote the request below and is keeping back one or two must-haves, each of them one property of a product. A must-have comes out only when a question touches that property or one close to it, and just a few one-line questions can be sent, so the order you give decides the outcome.

You get the request, what is settled so far, notes from a first reading, any questions already sent with their replies, and for each product (P1, P2 ...) the properties found on listings of that product, each with up to three sample values.

Work through these points for each product:
1. Read the story, not only the demands. Who is it for, what will they do with it, where will it sit, what let them down before, what must it go with or look like? Such a detail cares about some property even while it names no value, so that property is the likeliest place for a must-have.
2. Drop every property the request already settles: one it gives a value or limit for, and one the product's own name decides. Drop what was already asked too.
3. Drop logistics and paperwork: shipping, delivery, returns, warranty, condition, stock, seller, price, listing codes. Drop catch-all fields such as features or options too: the shopper takes them as several questions at once and refuses.
4. When properties overlap in meaning, keep the broader everyday word a person would say aloud. A question on the broad word also draws out the narrow one.
5. Only when the story gives nothing for a product, prefer what a buyer of that product usually insists on over technical detail.

Name up to eight properties per product, likeliest first, spelled as listed, and give each a likelihood from 0 to 1 that it holds a must-have.

Reply with ONE JSON object and nothing else: {"P1": [{"property": "name", "likelihood": 0.4}, ...], "P2": [...]}"""

HEAR = """Goal: work out what the shopper's latest words mean for a shopping assistant.
1. Produce ONE JSON object and no other text, with these keys:
"meaning": a single short sentence giving the shopper's intent.
"needs": firm requirements on the product that are new, each written "property: constraint" with exactly the value, limit, exclusion or closed list stated. Following a comparison phrase (falls short of, nothing less than, at least), a value made of words instead of a figure is itself the value and is copied as the shopper words it: "falls short of not available" becomes "property: not available" (a field may really hold "not available" or "none").
"no_preference": the properties the shopper declared indifference to.
"dropped": earlier requirements the shopper has let go of.
"new_budget": the fresh highest total price, a number in the unit of the request's budget (take the line's figure when it is in that unit: "13075 cents" gives 13075 for a budget kept in cents), or null.
"unavailable": true if the shopper reports that an item cannot be bought any more (sold out, gone, out of stock, button disabled); false otherwise.
"meta_only": true if the line merely asks the assistant to ask fewer things at a time (one question at a time, less in one go, the first one now and the others later) without giving any value: nothing is answered yet, so needs, no_preference and dropped get nothing from it, and its figurative wording names no value; false otherwise. A line that just says the shopper did not follow or understand the question, or wants it asked again more briefly or more simply, is not meta_only.
"requirements_now": the whole current list of firm requirements on the product, each "property: constraint": take REQUIREMENTS SO FAR as the base, keep every entry the shopper has not dropped or replaced, keep only the newest value of a replaced one, and append each new need the shopper's lines express. Add nothing from the request's story that REQUIREMENTS SO FAR lacks; requirements come only from the shopper's lines. Leave out price, stock, quantity and product category.
"reply": if the shopper asks the assistant a question or wants something confirmed, a short plain answer to send back that states the change in substance (such as confirming the new value); otherwise "".
"references": for every need in "needs" whose value the shopper supplies only by pointing at another listing or page, the listing pointed to: {"property": the property concerned, "seller": the shop or brand the line names for that listing, copied exactly, "kind": what kind of product that listing is, "details": [any further value the line gives for that listing, brief and worded as the line words it], "relation": "same", "differ" (NOT that listing's value: other than, different from, set apart from), "counterpart" (compatible with, goes with or pairs with that listing's value), "within", "at most" or "at least", "tolerance": the figure for "within", otherwise null, "unit": the unit, otherwise ""}; [] if there is none.
2. A single line may hold several changes; list all of them. A need about some other property, even a neighbouring one (seat height versus overall height, strap width versus width), is an extra requirement and replaces nothing: an older requirement is dropped or replaced only when the line says so for that very property. A replacement shows up twice: the old one under dropped and the new one under needs. If the shopper says which of two wishes matters more, the favoured one becomes a need. A budget the shopper states goes in new_budget. Plain wishes ("ideally", "would be nice") are not needs.
3. Take the shopper at their word and in good faith: homely, figurative or indirect phrasing that names a value means the shopper firmly wants it; a value counts as excluded only if the shopper plainly says to avoid it. Words that merely underline how firmly a value is held (strictly, only, locked to, nothing but, sticking with, stays, holding out for, nothing short of) do not belong to the value: write the value words alone ("material: jute", "color: white"), never "<value> only"; an exclusion is added only when the shopper also rules out blends or other values in so many words. A figure plus a comparison (under, over, no more than, at least, at most) forms a limit and keeps its unit. A new budget overrides the old one, up or down, even when called good news.
4. If the shopper merely says they don't mind, have no preference, or anything will do, the property that was asked about goes in no_preference (if the message asked about two properties, only the FIRST). When the reply only says the question went over the shopper's head ("you've lost me", "could you put it another way", "that was rather dense"), this shopper is expressing no preference when the question named the product the property belongs to: put that property in no_preference too; if several products are wanted and the question did not say which, the property is still open (no_preference stays empty).
5. With several products, each need is written "property (product): constraint", with the product taken from the line (a line on "toe style of the women's boots" yields "toe style (women's boots): ...").
6. The assistant's own question never becomes a requirement; only the shopper's words do, and sample values mentioned in the question are not needs. A "don't mind" frees only the property the assistant's message asked about; it never removes or alters a need stated earlier, and if the message asked nothing (it only reported something), such a reply changes nothing. For a need given by pointing at another listing, the value is the line's own pointing wording ("finish: unlike the one shown on Y's kettle page"); values the line gives for that other listing describe that listing, not the product, so they are neither needs, drops nor freed properties.
Return: the JSON object alone."""

READ_RULES = """1. Look up each requirement in the listing field that carries its property, comparing the value's own words as this store spells them there. Any field whose name expresses that property qualifies regardless of spelling (separators, a prefix, a shortened or lengthened name, or a familiar alternative name, e.g. "neckline" for a collar type, "color family" for a colour, "fit" for a pants fit type). If there is a field bearing the property's exact name, that field settles it, and so does one bearing that name with a single extra word in front (pants length for length, clothing material for material): the requirement holds as soon as either of them writes the value. Field names that differ only by separators (size class, size_class) are one and the same field: the requirement holds when either one writes the value. Otherwise the alternative field settles it; only when no field names the property may the options list or title show the value. Descriptions never count.
2. Compare values word by word as the store writes them: case, separators, singular, plural and spelling variants do not matter; a narrower value that contains the required one satisfies it ("rtx 3060 12gb" for rtx 3060, "80% cotton" for cotton, "forest green" for green); one entry of a list is sufficient, even where the shopper insists on that value alone: a field offering several variants that include it meets it. Synonyms, abbreviations, related words and shades ("solid" for plain, "win11" for windows 11, ivory or cream for off-white, "tech wood" for technical wood, "embroidered" for embroidery) satisfy a value only while no listing in the category writes the value itself in that field: if other listings say "matte" for their finish, a "satin" finish does not satisfy matte. A value combined with another by a plus ("wool + nylon") does not satisfy either part when the field elsewhere holds the other part alone ("nylon"). FIELD EVIDENCE lines, where included, are derived from this store's records and tell which situation applies. A different value of the same property fails (quartz is not mechanical, rosewood is not maple), and so does a mere look-alike word. A value that pretends to cover every value of the property ("universal", "all seasons", "suitable for all occasions") names none and fails a required value; a list that names the required value as well still satisfies it. A field naming another value does not satisfy the required one just because the two are often paired or one hints at the other ("slip-on" is not "lace-up", a "pullover" style is not "zip-up"); only a shade normally sold under the required colour, or a narrower value containing it, qualifies.
3. A field about a different property never counts (a strap length is not a strap type, a lining material is not the material, a length is not a style): it cannot replace a property whose own field the category's listings carry.
4. Numbers are compared as amounts once units are converted (4.5 cm is 45 mm); a limit includes its edge; a range passes only if the whole range meets the limit. If the title is the evidence because no field names the property, an exact required value counts as met when the title gives its words and figures in that order, even followed by more detail (9' x 13' does not show 9' x 12', while 9' x 12' 4" does).
5. A price is in the unit of the request's budget unless the record names some other currency or unit (45 set against a budget of 1000 US cents means 45 cents, comfortably inside it); rescale neither the price nor the budget.
6. Only the listed requirements matter: never fail a listing on an age group, gender, size, occasion, condition, season, quality or use that the requirements leave unnamed.
7. The category label alone settles the requested kind: a listing in the requested category is of that kind even when its title names something else (a case, a container, a set, a refill, an accessory); judge it purely on the requirements, price and stock."""

CHECK_ITEM = """Goal: act as the final check before a shopping assistant commits to an item; the errand scores 1 only if the item meets every requirement below. Decide ONLY from the records shown, following these steps:
""" + READ_RULES + """
8. Words that are part of the category's own name are met by the category.
9. Exclusions and closed lists hold.
10. The item must be of the requested kind; where listings carry a category label, the right label is the one fitting the request's wording as a whole, not a narrower or neighbouring one.
11. Price times quantity has to stay within the budget (equal is fine). If the records show it out of stock, it fails.
12. A mere wish ("ideally", "would be nice") marked as not required is no requirement.
13. When records conflict (on price or stock, say), the newest one wins.
14. Set fits=false only for a listed requirement the item's fields clearly fail under these rules (a FIELD EVIDENCE line saying the item does NOT meet a requirement is such a failure), or one that neither a field nor (when no field for it exists) the options list or the title states; never fail an item because a stock or price record is missing.
Return: ONE JSON object and nothing else: {"item": the item's id, "fits": true or false, "problems": [short strings, each naming a requirement not met or not shown]}."""

RANK_CHECK = """Goal: a shopping assistant is about to order an item for a request that wants the PRIORITY; find any listing that could beat it. The requirements, the chosen item and the listing records seen follow below.
1. Apply the reading rules below (price times quantity within the budget; not shown out of stock).
2. Name the listings that may meet every requirement AND outdo the chosen item on the priority (a lower price for the cheapest; a better value of the measure for the best).
3. Drop listings the records show failing a requirement, out of stock or over budget. Figures in a ROW's title (like 55 x 94) do not say which dimension each one is, so they never rule a ROW out.
READING RULES:
""" + READ_RULES + """
Return: ONE JSON object and nothing else: {"better": [ids, best first, at most 4], "reason": "one sentence"}."""

ASK_PLAN = """Goal: choose the questions a shopping assistant puts to a shopper. The shopper hides needs, each a value of one product property, given away only when a message asks about precisely that property; whatever a message asks first gets the most dependable reply, while questions about other things just get a polite "don't mind". Select at most QUESTION_COUNT distinct properties, the most promising first:
1. First take whatever appears below under MUST BE LEARNED FROM THE SHOPPER and has not yet been asked, and ask it in plain words. For the budget, ask the most they are willing to spend. For a choice between two wishes, mention both values together with their properties, e.g. "If one listing can't have both, would you rather have the <value> <property> or the <value> <property>?".
2. Then properties that really occur in the listings' details below. Study the story first (STORY HINTS below, if present, list its details and the property each relates to; they lead): the hidden need is typically the property that the story's concrete details relate to while never naming its value. What the user will touch or press, look at closely or in photos, hear, measure it against or fit it to, read on its label, or how it was produced, points at that part or aspect of the item and at its property: tags, labels and the fine print suggest the material, fabric or ingredients; photos and appearance suggest the colour, pattern or the visible part; a space or body part it has to fit suggests that part's shape or size; a gift for someone else, or for someone barely known, suggests whom it is made for (gender, age group); how it should feel against the hand, skin or foot suggests the material or fabric first; where, when or during which activity it is worn or used (outdoors, the office, a gym class) suggests its style or type, i.e. the property whose values name such uses; something fastened or adjusted again and again suggests its closure or mechanism; whatever does the work inside it suggests its core component. Weather, a job, a schedule, or a place other than where the item is worn or used is background. Ask first about the property the story's details point to, calling it by the listings' name for it. Whom the item is made for competes like any other candidate: ask about it if the story points at another person, or if nothing else is suggested. Then rank properties describing the product itself (material, pattern, closure, shape, length, type or style variant, function) above logistics (warranty, condition, pack size, variant or size lists, years).
3. Skip properties the request already decides (its requirements, the price, stock, quantity), any already asked, near-duplicates competing for two questions (two sorts of style, two spellings of one property), and anything on price, stock, delivery, brand, maker or a product or model name. Only ask about a property that shows up as a key in the listings' details below, using the listings' name for it; a STORY HINTS property that matches no key below may be asked under its plain name, except when the hint carries [no key] (enough of that product's listings were read without such a field): then ask the key below whose values vary as the detail describes, or drop it.
4. Each step-2 question is one short, plain, friendly sentence of at most 15 words, asking which <property> they would like for the <item kind> and naming the property the way the listings do (sample values on their own never reveal a need), optionally ending with one or two sample values. No introduction, no progress report, no instructions or lists, and never a catch-all key like options, variants, a description, a title or a name. A question targets ONE property of ONE product (the step-1 choice between two wishes excepted): never two properties in one question nor properties of two products, since a reply reveals one need at most and a two-subject question may go unanswered. Under "second", give the SAME product's runner-up property (never just a second name for the first one), or leave it empty: the assistant may append it to its final message.
5. If the request wants SEVERAL products, the hidden needs may belong to any of them: share out the questions so every product gets one before any gets two (a product with no property left aside), take every product's properties from the list below for its own category, put the property down as "property (product)", and name the product inside the question ("Which toe style would you like for the boots?").
Return: ONE JSON object and nothing more: {"questions": [{"property": "...", "second": "... or empty", "question": "..."}, ...]}."""

NEXT_ONE = """
6. Only ONE question goes out this time; the shopper's reply is read before the next one is picked. REMAINING question(s) are left altogether and each should be used: list the single most promising property first (one or two spares may follow); after a revealed need, the next question moves to another property and story detail."""

BEFORE_DECLINE = """Goal: a shopping assistant is about to tell the shopper that nothing in the store fits, which is seldom the right finish; check that claim. Hold the listing records below against the requirements with these reading rules (the price must stay inside the budget and the item must not be shown out of stock):
""" + READ_RULES + """
8. Some listings below are known only from search rows (no details read, so their property fields are unknown). Such a row in the requested category (whatever object its title names) whose title, price and stock do not show it failing a requirement is still unchecked (figures in a title, such as 55 x 94, do not say which dimension each one is, so they never show a row failing): put it under "unopened", the likeliest first (its title naming a required value, or the requested kind with nothing against it).
Return: ONE JSON object and nothing else: {"possible_fits": [ids of listings that seem to meet every requirement, best first, at most 5], "unopened": [ids of rows-only listings to open before declining, at most 6], "reason": "one sentence"}."""

REF_HOP = """Goal: a shopper's request borrows the PROPERTY of its product from another store listing, described as: DESCRIPTION. The records below come from listings that could be it.
1. Reply with ONE JSON object and nothing besides, holding:
"target": the id of whichever record IS the described listing: offered by that seller, of that kind, and carrying every named detail, judged by meaning (a figure stated in a different unit counts if the amount agrees); null if no record is.
"missing": the details absent from the target ([] if it has them all).
"go_to": if the target's own text points the reader to a DIFFERENT listing for the PROPERTY (take it from, use, refer to, look at or go by another listing), give that listing as {"seller": its shop or brand name as written, "kind": its kind of product, "details": [every value the text names for it, short and as written: a colour, a size with its unit, a material]}; otherwise null.
"value": if there is no go_to, the target's own value of the PROPERTY, copied as its record has it.
"goes_with": values the target's text says go with, pair with or equal that value ([] if none).
2. When the text mentions a listing only as what the target is not ("not the green one from X"), or as a version, size or colour available elsewhere, that is no go_to; nor is a listing mentioned for some other property. A record lacking the described details is never the target, whatever its text claims.
Return: the JSON object."""

REF_READ = """Goal: a line of text from a simulated store may give the value of a product property only by pointing at another store listing. Read the line as a whole and reply with ONE JSON object and nothing else: {"pointers": [...]}, one entry per listing pointed at, [] when it points at none.
1. Each entry holds:
"property": the product property whose value depends on that listing, named as the line or the QUESTION names it.
"seller": the shop or brand named for that listing, copied exactly.
"kind": what kind of product that listing is, copied from the line, "" if unsaid.
"details": [every value the line gives to identify that listing, copied word for word together with its field name when the line gives one].
"relation": how the wanted value relates to that listing's value: "same" (that value itself), "differ" (must not be that value), "counterpart" (must be compatible with, go with or pair with that value rather than be it), "at least", "at most", "within".
"tolerance": the figure for "within", otherwise null.
"unit": its unit, otherwise "".
"quote": the words of the line from where the pointing starts to the end of the listing's description, copied exactly.
2. Negations cancel out: a shopper who will not take anything other than what a listing gives wants that value itself, so its relation is "same".
3. Details describe the listing and nothing else: no verb, and no words after the description (how firmly it is meant, how the value may be found, pleasantries).
4. A listing named only as what an item is not, or as a version sold elsewhere, is no pointer.
Return: the JSON object."""

DECLINE_WORDS = r"refus|declin|cancel|abandon|reject|give.?up|abstain|withdraw|no.?(?:purchase|order|buy)|nothing|without"
ORDER_WORDS = r"commit|buy|order|purchas|check.?out|pay|confirm|complete|place|accept|proceed"
SORRY = "Sorry, nothing in the store meets all of your requirements, so I am not ordering anything."
ID_LIKE = r"[A-Za-z0-9][A-Za-z0-9_\-:.]{3,63}"
SPELLED = ("one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve")
CODE_LIKE = r"[A-Za-z0-9_\-]+"


# ---------------------------------------------------------------------------------------------------------------- text helpers
BOUND_WORDS = r"\b(?:no|not)\s+(?:more|less|fewer|over|under|above|below|beyond|exceeding)\b"
CH_BOUND = r"\b(?:at least|no more than|at most|no less than|more than|less than|up to|or more|or less|or larger|or smaller|or higher|" \
           r"or lower|minimum|maximum|between|within|under|over|above|below|than)\b"
CJ_COMPARE = (r"\b(?:falls?\s+short\s+of|short\s+of|nothing\s+(?:less|other)\s+than|nothing\s+short\s+of|nothing\s+but|"
              r"no\s+(?:less|worse)\s+than|at\s+least|(?:less|more|better|worse)\s+than|exactly|only|just)\b")
CH_SKIP = {"and", "the", "a", "an", "of"}
# word values a shopper gave after a comparison; emptied when a new run starts
LITERAL_VALUES: set[str] = set()
EXCLUDE_WORDS = r"\b(?:no(?!-)|not|without|nothing|never|neither|nor|except|excluding|avoid\w*|anything but|other than|rather than|instead of)\b"
VALUE_LINKS = r"at least|no more than|at most|no less than|more than|less than|up to|or more|or less|or larger|or smaller|" \
              r"or higher|or lower|minimum|maximum|between|within|under|over|above|below|exactly|only|either|neither|nor|than|plus"
REF_WORDS = r"\bas shown\b|\bshown on\b|\bsame as\b|\bpage\b|\bsold by\b|\bredirect|\brefer(?:s|red|ring)?\b|\blisting points\b|\bpoints you\b"
REF_STEPS = 0.4
REF_CALLS = 12
REF_HOPS = 8
DIFFER_WORDS = (r"\bset apart from\b|\bapart from\b|\bdistinct from\b|\bdifferent (?:from|to|than)\b|\bdiffer(?:s|ent|ing)?\s+from\b"
                r"|\bother than\b|\bunlike\b|\bnot the same as\b|\banything but\b")
POINT_WORDS = r"\bwithin\b|\bat (?:most|least)\b|\bno (?:more|less) than\b|\bsame as\b|\bas shown\b|\bthe value\b|\bthe one\b"
REF_DIFFER_STEPS = 0.3
REF_DIFFER_BLIND = 5
REF_READS = 8
REF_OWN = 2
DECLINE_SEARCHES = 2
DECLINE_READS = 3
WALK_AWAY = (r"\b(?:nothing|none|no (?:item|listing|product)\w*)\b[^.!?]*\b(?:fits?|suits?|matche?s?|works?)\b|\bwalk away\b|\bdecline\b"
             r"|\b(?:leave|go) empty.handed\b")
SELLER_KEYS = r"shop|seller|store|vendor|brand|make|manufactur|merchant|retailer|supplier|sold.?by"


def as_text(piece: Any, ceiling: int = VIEW_CHARS) -> str:
    try:
        shown = json.dumps(piece, ensure_ascii=False, separators=(",", ":"))
    except (TypeError, ValueError):
        shown = str(piece)
    if len(shown) > ceiling:
        return shown[:ceiling] + "...(cut)"
    return shown


def clipped(piece: Any, chars: int, lvl: int = 0) -> Any:
    if lvl > 8:
        return None
    deeper = lvl + 1
    if isinstance(piece, dict):
        return {name: clipped(inner, chars, deeper) for name, inner in piece.items()}
    if isinstance(piece, list):
        return [clipped(inner, chars, deeper) for inner in piece]
    if isinstance(piece, str) and len(piece) > chars:
        return piece[:chars] + "..."
    return piece


def json_object(text_body: Any) -> dict[str, Any] | None:
    if not isinstance(text_body, str):
        return None
    body = text_body.strip()
    if body.startswith("```"):
        body = re.sub(r"^```[A-Za-z]*", "", body).strip()
        if body.endswith("```"):
            body = body[:-3].strip()
    attempts = [body]
    if "{" in body:
        attempts.append(body[body.find("{"): body.rfind("}") + 1])
    for chunk in attempts:
        if not chunk:
            continue
        try:
            parsed = json.loads(chunk)
        except ValueError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def is_number(piece: Any) -> bool:
    if isinstance(piece, bool):
        return False
    return isinstance(piece, (int, float))


def number_words() -> dict[str, int]:
    return {word: spot for spot, word in enumerate("one two three four five six seven eight nine ten".split(), 1)}


def id_marks(text_body: Any) -> set[str]:
    # identifiers written as a capitalised name and a number ("Name 1234")
    return set(re.findall(r"\b[A-Z][A-Za-z]+ \d{3,5}\b", str(text_body or "")))


def ask_name(key: Any) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", str(key).replace("_", " ").lower()))


def ask_menu(listings: list[dict[str, str]], settled: set[str], basket: bool) -> list[tuple[str, int, list[str]]]:
    seen: dict[str, dict[str, int]] = {}
    for facts in listings:
        for key, value in facts.items():
            seen.setdefault(key, {})
            seen[key][value] = seen[key].get(value, 0) + 1
    ranked = []
    for key, values in seen.items():
        name = ask_name(key)
        if not name or name in settled:
            continue
        carriers = sum(values.values())
        distinct = len(values)
        if all(re.fullmatch(r"[a-z]{0,3}[-_]?[0-9a-f]{8,}", value.lower()) for value in values):
            continue
        if distinct == carriers and sum(len(value.split()) for value in values) >= 4 * distinct:
            continue
        # a variant label joins several parts into one value ("part / part") and holds no wish of its own
        if sum(count for value, count in values.items() if " / " in value) * 2 > carriers:
            continue
        # a basket product gets one or two questions: the most carried property that varies goes first
        weight = (distinct > 1, carriers) if basket else (True, carriers * min(distinct, 4))
        samples = [value for value, _ in sorted(values.items(), key=lambda pair: -pair[1])][:ASK_SAMPLES]
        ranked.append((weight, key, carriers, samples))
    ranked.sort(key=lambda entry: entry[0], reverse=True)
    unique: list[tuple[str, int, list[str]]] = []
    for _, key, carriers, samples in ranked:
        twin = next((at for at, kept in enumerate(unique) if ask_twins(listings, key, kept[0])), None)
        if twin is None:
            unique.append((key, carriers, samples))
            continue
        kept = unique[twin]
        # the twin's place stays, and the broader of the two names speaks for both
        if ask_breadth(key, carriers) < ask_breadth(kept[0], kept[1]):
            unique[twin] = (key, max(carriers, kept[1]), samples)
    return unique


def ask_breadth(key: str, carriers: int) -> tuple[int, int, int]:
    # the name most listings use: replay lost 5 points of exact hits keeping the shorter word, for 1 point of near hits
    return -carriers, len(prop_words(key)), len(ask_name(key))


def ask_twins(listings: list[dict[str, str]], one: str, other: str) -> bool:
    if ask_name(one).replace(" ", "") == ask_name(other).replace(" ", ""):
        return True
    left, right = set(prop_words(one)), set(prop_words(other))
    near = bool(left and right) and (left <= right or right <= left or same_prop(one, other))
    shared = [(ask_name(facts[one]), ask_name(facts[other])) for facts in listings if one in facts and other in facts]
    if shared:
        # a unit may trail on one side only; unrelated names need two different values, since yes and yes prove nothing
        agree = all(a == b or a.startswith(b + " ") or b.startswith(a + " ") for a, b in shared)
        return agree and (near or len({a for a, _ in shared}) >= 2)
    if not near:
        return False
    mine = {ask_name(facts[one]) for facts in listings if one in facts}
    theirs = {ask_name(facts[other]) for facts in listings if other in facts}
    return len(mine & theirs) * 2 >= min(len(mine), len(theirs))


def ask_picks(reply: Any) -> list[tuple[str, float]]:
    # one product's share of the ranking reply in the model's order; an unreadable likelihood does not sink a pick
    picks = []
    for entry in reply if isinstance(reply, list) else []:
        key = entry.get("property") if isinstance(entry, dict) else entry
        like = entry.get("likelihood") if isinstance(entry, dict) else None
        if key:
            picks.append((str(key), like / 100 if is_number(like) and like > 1 else like if is_number(like) else 1.0))
    return picks


def ask_merge(picks: list[tuple[str, float]], spread: list[str]) -> list[str]:
    # the ranking leads; once its first picks are placed it keeps a slot only while it still believes in the key
    found: dict[str, float] = {}
    for raw, like in picks:
        key = next((key for key in spread if ask_name(key) == ask_name(raw)), None)
        key = key or next((key for key in spread if strict_prop(key, raw)), None)
        if key:
            found.setdefault(key, like)
    lead = list(found)[:ASK_FIRST_PICKS]
    sure = [key for key in list(found)[ASK_FIRST_PICKS:] if found[key] >= ASK_SURE]
    return list(dict.fromkeys(lead + sure + spread))


def ask_opaque(name: str) -> bool:
    return all(len(word) < 3 or not re.search(r"[aeiouy]", word) for word in name.split())


def written_figures(text_body: Any) -> list[float]:
    figures = [float(raw.replace(",", "")) for raw in re.findall(r"\d[\d,]*(?:\.\d+)?", str(text_body or ""))]
    # a limit can be spelled out ("five messages"), and a small limit is the one a request is likely to spell
    return figures + [float(SPELLED.index(word) + 1) for word in re.findall(r"[a-z]+", str(text_body or "").lower()) if word in SPELLED]


def id_field(key: Any) -> bool:
    # listing ids sit under names like id or product_id, but a store may call them a key
    low = str(key).lower()
    return "id" in low or low.endswith("key")


def plain_words(text_body: Any) -> list[str]:
    return re.findall(r"[a-z0-9]+", str(text_body or "").lower())


def request_sentences(text_body: str) -> list[str]:
    return [part.strip() for part in re.split(r"(?<=[.!?;])\s+", text_body or "") if part.strip()]


def matching_sentence(quote: str, text_body: str, share: float = 0.7) -> str:
    wanted = plain_words(quote)
    if not wanted:
        return ""
    best, best_share = "", 0.0
    for sentence in request_sentences(text_body):
        have = set(plain_words(sentence))
        found = sum(1 for word in wanted if word in have) / len(wanted)
        if found > best_share:
            best, best_share = sentence, found
    return best if best_share >= share else ""


def long_strings(piece: Any, collected: list[str], lvl: int = 0) -> None:
    if lvl <= 4:
        if isinstance(piece, str):
            if len(piece.split()) > 3:
                collected.append(piece)
        elif isinstance(piece, list):
            nested = lvl + 1
            for inner in piece[:10]:
                long_strings(inner, collected, nested)


def walk_dicts(piece: Any, visit, lvl: int = 0, ceiling: int = 6) -> None:
    if lvl > ceiling:
        return
    if isinstance(piece, list):
        for inner in piece:
            walk_dicts(inner, visit, lvl + 1, ceiling)
        return
    if not isinstance(piece, dict):
        return
    visit(piece)
    for inner in piece.values():
        if isinstance(inner, (dict, list)):
            walk_dicts(inner, visit, lvl + 1, ceiling)


def _name_head(name: Any) -> str:
    # the name before any "(product)" tag, camelCase opened up, lower case
    head = re.split(r"\s*\(", str(name))[0]
    return re.sub(r"([a-z])([A-Z])", r"\1 \2", head).lower()


def _singular(word: str) -> str:
    if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


def junk_key(key: Any) -> bool:
    words = " ".join(re.findall(r"[a-z0-9]+", _name_head(key)))
    if re.search(r"\b(?:brand|shop|seller|store|vendor|manufacturer|category|categories|department|title|headline|"
                 r"heading|listing|description|overview|price|cost|amount|availability|sku|url)\b", words):
        return True
    return bool(re.fullmatch(r"(?:name|notes?|stock|in stock|on hand|images?|links?|make|maker|shelf|ids?|id)", words))


def prop_words(name: Any) -> list[str]:
    return [_singular(word) for word in re.findall(r"[a-z]+", _name_head(name))]


def prop_key(name: Any) -> str:
    return "".join(prop_words(name))


def edits_within(a: str, b: str, most: int) -> bool:
    if abs(len(a) - len(b)) > most:
        return False
    costs = list(range(len(b) + 1))
    for row in range(1, len(a) + 1):
        diagonal, costs[0] = costs[0], row
        for col in range(1, len(b) + 1):
            above = costs[col]
            costs[col] = min(above + 1, costs[col - 1] + 1, diagonal + (a[row - 1] != b[col - 1]))
            diagonal = above
        if min(costs) > most:
            return False
    return costs[-1] <= most


def same_prop(a: Any, b: Any) -> bool:
    first, other = prop_key(a), prop_key(b)
    if not (first and other):
        return False
    if first == other:
        return True
    shorter = min(len(first), len(other))
    if shorter >= 4 and first[:3] == other[:3]:
        if edits_within(first, other, 2 if shorter >= 6 else 1):
            return True
    if shorter < 5:
        return False
    return first.startswith(other) or other.startswith(first)


def _soft_letter(word: str, at: int) -> bool:
    if word[at] in "aeiouy":
        return True
    if at > 0 and word[at - 1] == word[at]:
        return True
    return at + 1 < len(word) and word[at + 1] == word[at]


def _without_soft(word: str):
    for at in range(len(word)):
        if _soft_letter(word, at):
            yield word[:at] + word[at + 1:]


def spelled_alike(lo: str, hi: str) -> bool:
    extra = len(hi) - len(lo)
    if extra == 0:
        spots = [at for at in range(len(lo)) if lo[at] != hi[at]]
        if len(spots) != 2:
            return False
        one, two = spots
        return two == one + 1 and lo[one] == hi[two] and lo[two] == hi[one]
    if extra == 1:
        return any(shorter == lo for shorter in _without_soft(hi))
    if extra == 2:
        return any(twice == lo for shorter in _without_soft(hi) for twice in _without_soft(shorter))
    return False


def strict_prop(a: Any, b: Any) -> bool:
    first, other = prop_key(a), prop_key(b)
    if not (first and other):
        return False
    if first == other:
        return True
    lo, hi = sorted((first, other), key=len)
    if len(lo) >= 4 and lo[:3] == hi[:3] and spelled_alike(lo, hi):
        return True
    short, longer = sorted((prop_words(a), prop_words(b)), key=len)
    if len(longer) != len(short) + 1 or longer[:len(short)] != short:
        return False
    return len(longer[-1]) <= 3 and len("".join(short)) >= 5


def ch_words(text: Any) -> list[str]:
    found = re.findall(r"[a-z0-9]+(?:\.[0-9]+)?", str(text or "").lower())
    return [word for word in found if word not in CH_SKIP]


def ch_word_in(word: str, pool: list[str]) -> bool:
    if word in pool:
        return True
    if not word.isalpha():
        return False
    for other in pool:
        if not other.isalpha():
            continue
        if same_kind(word, other):
            return True
        if len(word) >= 6 and len(other) >= 6 and word[:2] == other[:2] and edits_within(word, other, 1):
            return True
    return False


def ch_alts(value: Any) -> list[list[str]]:
    marked = str(value or "").lower()
    bare = re.sub(r"\((?:stated|revealed[^)]*)\)", " ", marked).replace('"', " ")
    bare = re.sub(r"\b(?:only|just|exactly|strictly)\b", " ", bare)
    split = (ch_words(part) for part in re.split(r"\bor\b", bare))
    return list(filter(None, split))


def ch_writes(value: Any, text: Any) -> bool:
    options = ch_alts(value)
    return ch_match(options, ch_words(text))


def _joined_run(pool: list[str], target: str) -> bool:
    # some run of neighbouring words, glued together, spells the target
    for start in range(len(pool)):
        glued = ""
        for word in pool[start:]:
            glued += word
            if glued == target:
                return True
            if len(glued) >= len(target):
                break
    return False


def ch_match(alts: list[list[str]], pool: list[str]) -> bool:
    if not pool:
        return False
    digits_in_pool = None
    for option in alts:
        if all(ch_word_in(word, pool) for word in option):
            return True
        target = "".join(option)
        if len(option) < 2 and not re.search(r"\d", target):
            if digits_in_pool is None:
                digits_in_pool = any(re.search(r"\d", word) for word in pool)
            if not digits_in_pool:
                continue
        if _joined_run(pool, target):
            return True
    return False


def ch_touches(value: Any, text: Any) -> bool:
    pool = ch_words(text)
    low = str(text or "").lower()

    def touches(word: str) -> bool:
        if ch_word_in(word, pool):
            return True
        if re.fullmatch(r"\d+(?:\.\d+)?", word) and re.search(r"(?<![\d.])" + re.escape(word) + r"(?!\d|\.\d)", low):
            return True
        if len(word) < 3:
            return False
        return any(len(other) >= 3 and (other.startswith(word[:3]) or word.startswith(other[:3])) for other in pool)

    return any(touches(word) for option in ch_alts(value) for word in option)


def ch_parts(text: Any) -> list[str]:
    low = str(text or "").strip().lower()
    pieces = [bit.strip() for bit in re.split(r"[,;|]", low)]
    kept = [bit for bit in pieces if bit]
    if kept:
        return kept
    return [low] if low else []


def ch_sides(part: Any) -> list[str]:
    sides = [side.strip() for side in re.split(r"\s*\+\s*", str(part))]
    sides = [side for side in sides if side]
    return sides if len(sides) > 1 else []


def figure_written(fig: str, said: str) -> bool:
    decimal = "." in fig
    for hit in re.finditer(r"(?<![\d.])" + re.escape(fig) + r"(?!\d|\.\d)", said):
        if decimal:
            return True
        unit = re.match(r"\s*([a-z]+)", said[hit.end():])
        if unit is None or len(unit.group(1)) <= 3:
            return True
    return False


def _content_words(text: str) -> list[str]:
    return [word for word in re.findall(r"[a-z]{3,}", text) if word not in LINK_WORDS]


def positive_value(value: Any, name: Any = "") -> bool:
    low = str(value).lower().strip()
    if re.fullmatch(r"\W*(?:yes|no|true|false|none|required|included|needed)\W*", low):
        return True
    if LITERAL_VALUES:
        plain = " ".join(re.findall(r"[a-z0-9]+", re.sub(r"\((?:stated|revealed[^)]*)\)", " ", low)))
        if plain in LITERAL_VALUES:
            return True
    # bounds such as "no more than" or "no thicker than" are limits, not exclusions
    low = re.sub(BOUND_WORDS, " ", low)
    low = re.sub(r"\b(?:no|not)\s+[a-z]+er\b(?=\s+than\b)", " ", low)
    if re.search(r"\d", low):
        low = re.sub(r"\b(?:no|not)\s+[a-z]+er\b", " ", low)
    body = re.sub(r"\b(?:" + VALUE_LINKS + r")\b", " ", low)
    clauses = re.split(r",|;|/|&|\(|\)|\bor\b|\band\b", body)
    if any(len(_content_words(clause)) >= 4 for clause in clauses):
        return False
    hit = re.search(EXCLUDE_WORDS, low)
    if hit is None or _content_words(low[:hit.start()]):
        return True
    after = _content_words(low[hit.end():])
    own = set(prop_words(name)) | set(re.findall(r"[a-z]+", str(name).lower()))
    if not after or not own:
        return False
    return all(word in own or word.rstrip("s") in own for word in after)


def _figure_runs(body: str) -> list[list[str]]:
    runs = re.findall(r"\d+(?:\.\d+)?(?:\s*[:x/]\s*\d+(?:\.\d+)?)*", body)
    return [re.findall(r"\d+(?:\.\d+)?", run) for run in runs]


def value_seen(value: Any, samples: Any) -> bool:
    pool = str(samples or "").lower()
    if not pool.strip():
        return False
    body = re.sub(r"\b(?:" + VALUE_LINKS + r")\b", " ", str(value).lower())
    for word in re.findall(r"[a-z]{3,}", body):
        if word in LINK_WORDS:
            continue
        stem = word[:-1] if len(word) > 4 else word
        if re.search(r"\b" + re.escape(stem), pool):
            return True
    for figs in _figure_runs(body):
        if re.search(r"(?<![\d.])" + r"\D{0,4}".join(re.escape(f) for f in figs) + r"(?![\d.])", pool):
            return True
    return False


def kind_words(text: Any) -> set[str]:
    stems = set()
    for word in _content_words(str(text).lower()):
        stems.add(word.rstrip("s"))
    return stems


def kind_tokens(text: Any) -> list[str]:
    low = re.sub(r"'s\b", "", str(text).lower().replace("’", "'"))
    return _content_words(low)


# (ending, letters cut, ending added) pairs for singular and plural spellings
_PLURAL_SWAPS = (("y", 1, "ies"), ("fe", 2, "ves"), ("ies", 3, "y"), ("ves", 3, "f"), ("ves", 3, "fe"), ("es", 2, ""), ("s", 1, ""))


def plural_forms(word: Any) -> set[str]:
    base = str(word)
    forms = {base, base + "s", base + "es"}
    for ending, cut, added in _PLURAL_SWAPS:
        if base.endswith(ending):
            forms.add(base[:-cut] + added)
    if base.endswith("f") and not base.endswith("fe"):
        forms.add(base[:-1] + "ves")
    return set(form for form in forms if len(form) > 2)


def same_kind(a: Any, b: Any) -> bool:
    if a == b:
        return True
    return str(b) in plural_forms(a) or str(a) in plural_forms(b)


def kind_in(word: Any, pool: Any) -> bool:
    for other in pool:
        if same_kind(word, other):
            return True
    return False


def kinds_within(words: Any, pool: Any) -> bool:
    for word in words:
        if not kind_in(word, pool):
            return False
    return True


def kind_score(own: Any, words: Any, shared: Any) -> float:
    total = 0
    for word in own:
        if kind_in(word, words):
            total += 0.5 if kind_in(word, shared) else 1.0
    return total


def _longest_first(form: str) -> tuple:
    return -len(form), form


def name_pattern(name: Any) -> str:
    def word_pattern(token: str) -> str:
        choices = "|".join(re.escape(form) for form in sorted(plural_forms(token), key=_longest_first))
        return r"(?:" + choices + r")(?:'s\b|'|\b)"
    return r"[\s'-]+".join(word_pattern(token) for token in kind_tokens(name))


def one_prop_hint(hint: str) -> str:
    try:
        head, colon, detail = hint.partition(":")
        if not colon or "/" not in head:
            return hint
        tag = re.search(r"\(([^()]*)\)\s*$", head)
        listed = head[: tag.start()] if tag else head
        names = [bit.strip() for bit in listed.split("/") if bit.strip()]
        if len(names) < 2:
            return hint
        suffix = f" ({tag.group(1).strip()})" if tag else ""
        return f"{names[0]}{suffix}:{detail} (or {', '.join(names[1:])})"
    except Exception:
        return hint


def lead_value(detail: Any, request: str, product: str) -> str:
    try:
        opening = re.match(r"\s*([a-z][a-z-]*[a-z])(?:\s*[,;.()]|\s*$|\s+(?:" + "|".join(sorted(LINK_WORDS)) + r")\b)",
                           str(detail).lower())
        if not opening:
            return ""
        word = opening.group(1)
        kinds = _content_words(str(product).lower())
        if len(word) < 3 or word in LINK_WORDS or not kinds:
            return ""
        if word.rstrip("s") in {kind.rstrip("s") for kind in kinds}:
            return ""
        head_kind = kinds[0].rstrip("s")
        tokens = re.findall(r"[a-z][a-z'-]*", str(request).lower())
        for here, after in zip(tokens, tokens[1:]):
            if here == word and after.split("'")[0].rstrip("s") == head_kind:
                return word
    except Exception:
        return ""
    return ""


def _word_said(word: str, said_words: set[str], said_stems: set[str]) -> bool:
    if word.endswith("es") and len(word) > 4:
        stem = word[:-2]
    elif word.endswith("s") and len(word) > 3:
        stem = word[:-1]
    else:
        stem = word
    for said in said_words:
        if said == word or (said.startswith(stem) and len(said) <= len(stem) + 3):
            return True
        if stem.startswith(said) and len(said) >= max(4, len(stem) - 2):
            return True
        if len(word) >= 4 and said.startswith(word):
            return True
        if len(word) >= 4 and len(said) >= 4 and said[0] == word[0] and edits_within(said, word, 1):
            return True
    return any(len(said) >= 3 and stem.startswith(said) and len(said) >= len(stem) - 3 for said in said_stems)


def _value_body(req: str) -> str:
    head, _, value = req.partition(":")
    low = value.strip().lower()
    if re.fullmatch(r"\W*(?:yes|no|true|false|required|included|needed)\W*", low):
        low = re.split(r"\s*\(", head)[0].lower()
    return re.sub(r"\b(?:" + VALUE_LINKS + r")\b", " ", low)


def _unsaid_runs(body: str, said: str) -> list[str]:
    runs = []
    for run in re.findall(r"\d+(?:\.\d+)?(?:\s*[:x/]\s*\d+(?:\.\d+)?)*", body):
        figs = re.findall(r"\d+(?:\.\d+)?", run)
        if len(figs) > 1 and re.search(r"(?<![\d.])" + r"\D{0,4}".join(re.escape(f) for f in figs) + r"(?!\d)", said) is None:
            runs.append(run)
    return runs


def story_guess(req: Any, request: str, refs: list[str]) -> list[str]:
    req = str(req)
    value = req.partition(":")[2].strip().lower()
    if not value or req in refs or re.search(r"as shown|same as|the page|listing|sold by|redirect|refer", value):
        return []
    body = _value_body(req)
    said = str(request).lower()
    figures = re.findall(r"\d+(?:\.\d+)?", body)
    if figures and all(figure_written(fig, said) for fig in figures):
        return []
    said_words = set(re.findall(r"[a-z]+", said))
    said_stems = {word[:-1] if word.endswith("s") and len(word) > 3 else word for word in said_words}
    content = _content_words(body)
    unsaid = []
    for word in content:
        if not _word_said(word, said_words, said_stems):
            unsaid.append(word)
    short_and_written = len(content) <= 2 and any(len(word) > 3 and word not in unsaid for word in content)
    return [] if short_and_written else unsaid + _unsaid_runs(body, said)


def ref_text(value: Any) -> bool:
    return re.search(REF_WORDS, str(value or "").lower()) is not None


def _drop_plural(token: str) -> str:
    if re.search(r"(?:ch|sh|x|z)es$", token):
        return token[:-2]
    return token[:-1]


def ref_toks(text: Any) -> list[str]:
    def glue(hit):
        if hit.group(2) in LINK_WORDS:
            return hit.group(0)
        return hit.group(1) + hit.group(2)

    low = re.sub(r"(\d)(?:\s+|\s*-\s*)([a-z]{1,8})\b", glue, str(text or "").lower())
    tokens = []
    for token in re.findall(r"\d+(?:[.,:/]\d+)*[a-z]*|[a-z][a-z0-9]*", low):
        if token in LINK_WORDS:
            continue
        if len(token) == 1 and not token.isdigit():
            continue
        if len(token) > 3 and token.endswith("s") and not token.endswith("ss") and not token[-2].isdigit():
            token = _drop_plural(token)
        tokens.append(token)
    return tokens


def ref_figs(text: Any) -> set[tuple[float, str]]:
    pairs: set[tuple[float, str]] = set()
    try:
        low = str(text or "").lower().replace(",", "")
        for hit in re.finditer(r"(\d+(?:\.\d+)?(?:\s*/\s*\d+(?:\.\d+)?)*)\s*-?\s*([a-z]{1,8})\b", low):
            unit = hit.group(2)
            if len(unit) >= 3 and unit.endswith("s") and not unit.endswith("ss"):
                unit = _drop_plural(unit)
            if unit in LINK_WORDS:
                continue
            pairs.update((float(fig), unit) for fig in re.split(r"\s*/\s*", hit.group(1)))
    except Exception:
        return pairs
    return pairs


def ref_name(text: Any) -> str:
    low = re.sub(r"'s\b", "", str(text or "").lower())
    return re.sub(r"[^a-z0-9]", "", low)


_BOUND_RELATIONS = ((r"\b(?:up to|at most|or less|no more than|or lower|or smaller|no (?:higher|larger|greater) than)\b", "at most"),
                    (r"\b(?:at least|no less than|or more|or higher|or larger|no (?:lower|smaller) than)\b", "at least"))
COUNTER_WORDS = r"\bcompatible\b|\bcounterparts?\b|\bgoes? (?:together )?with\b|\bpairs? (?:up )?with\b"
# what may follow a listing's description: verbs closing the clause, and clauses about something else
REF_VERBS = r"\s+(?:says?|shows?|states?|lists?|gives?|reads?|displays?|has|have|had|is|are|does|do|did|carries|offers)"
REF_TAIL = r"(?:" + REF_VERBS + r")+\s*$"
REF_CLAUSE_END = r"\b(?:whether|regardless|no matter|either way|however|please|thanks|thank you|because|since)\b"
REF_NAMES = ("same", "differ", "counterpart", "at most", "at least", "within")


def _set_apart(lead: str) -> bool:
    # each "other than" flips the meaning, and so does a negation in front of the first one
    hits = list(re.finditer(DIFFER_WORDS, lead))
    if not hits:
        return False
    before = re.split(r"[,;:.!?]", lead[:hits[0].start()])[-1]
    negated = re.search(r"\b(?:not|never|nothing|none|no)\b|n't\b", before) is not None
    return (len(hits) + negated) % 2 == 1


def ref_relation(value: Any, seller: Any = "") -> tuple[str, float | None, str]:
    low = str(value or "").lower()
    # words after the seller describe that listing, so only what comes before can set apart or pair
    cut = low.find(str(seller).lower()) if str(seller or "").strip() else -1
    lead = low[:cut] if cut > 0 else low
    if _set_apart(lead):
        return "differ", None, ""
    if re.search(COUNTER_WORDS, lead):
        return "counterpart", None, ""
    within = re.search(r"\bwithin\s+(\d+(?:\.\d+)?)\s*([a-z%]+)?", low)
    if within is not None:
        unit = within.group(2) or ""
        return "within", float(within.group(1)), unit.strip()
    for pattern, relation in _BOUND_RELATIONS:
        if re.search(pattern, low):
            return relation, None, ""
    return "same", None, ""


def ref_relation_name(word: Any) -> str:
    low = str(word or "").lower().strip()
    if low in REF_NAMES:
        return low
    if re.search(r"compat|counterpart|pair|goes", low):
        return "counterpart"
    for pattern, relation in _BOUND_RELATIONS:
        if re.search(pattern, low):
            return relation
    return "differ" if re.search(r"differ|not\b|other|exclu|apart|unlike", low) else "same"


def _seller_hit(body: str, ok: Any):
    for hit in re.finditer(r"(?<![A-Za-z'])([A-Z][a-z]{3,})(?:'s)?(?![A-Za-z])", body):
        # the line's own first word is capitalised whoever it names
        if re.search(r"[A-Za-z]", body[:hit.start()]) and (ok is None or ok(hit.group(1))):
            return hit
    return None


def ref_detail(text: Any) -> str:
    kept = re.sub(r"^\W*(?:(?:and|with|in|at|of|the|a|an|that|which)\s+)*", "", str(text or "")).strip(" ,;.!?\"'")
    if len(kept.split()) >= 2:
        kept = re.sub(REF_TAIL, "", kept, flags=re.I).strip(" ,;.!?")
    return kept


def ref_parse(text: Any, ok: Any = None) -> dict[str, Any] | None:
    # the code's own reading, used when the model's fails: it stops where the listing's description stops
    try:
        body = re.split(r"\(\s*or\b", str(text or ""))[0]
        hit = _seller_hit(body, ok)
        if hit is None:
            return None
        before, rest = body[:hit.start()], body[hit.end():]
        rest = re.split(r"[.!?;](?:\s|$)|" + REF_CLAUSE_END + "|" + REF_VERBS + r"(?=\s*[,.;!?])", rest, maxsplit=1, flags=re.I)[0]
        kind = ""
        sold = re.search(r".*(?:\bthe|\bfor|\ba|\ban)\s+(.*?)\s+(?:sold|made|offered|listed)\s+by\s*$", before, re.I | re.S)
        if sold is not None:
            kind = sold.group(1)
        else:
            lead = re.match(r"\s*([^,;(]*?)(?=\s+(?:with|that|which|in|at|having|from|of|for|sold)\b|[,;]|$)", rest, re.I)
            if lead is not None:
                kind, rest = lead.group(1), rest[lead.end():]
        details = []
        for chunk in re.split(r",|;|\band\b|\bwith\b|\bthat\s+(?:is|are|has|have|comes?(?:\s+as)?)\b|\bwhich\b|\bin\b|\bat\b", rest, flags=re.I):
            kept = ref_detail(" ".join(word for word in re.findall(r"[A-Za-z0-9~][A-Za-z0-9.%:/&'~-]*", chunk) if word.lower() not in LINK_WORDS))
            if kept and len(kept.split()) <= 6:
                details.append(kept)
        relation, tolerance, unit = ref_relation(before + hit.group(0) + rest, hit.group(1))
        return {"seller": hit.group(1), "kind": kind.strip()[:80], "details": details[:6], "property": "", "relation": relation,
                "tolerance": tolerance, "unit": unit, "quote": "", "by": "code"}
    except Exception:
        return None


def _said_in(detail: str, said: set[str]) -> bool:
    tokens = ref_toks(detail)
    # a bare figure may stand for one the line writes with its unit
    return bool(tokens) and all(tok in said or (tok[0].isdigit() and any(word.startswith(tok) for word in said)) for tok in tokens)


def ref_checked(raw: Any, line: Any, ok: Any = None) -> dict[str, Any] | None:
    # a structured reading is kept only as far as the line itself bears it out
    if not isinstance(raw, dict):
        return None
    text = str(line or "")
    seller = re.sub(r"'s$", "", str(raw.get("seller") or "").strip())
    written = re.search(r"(?<![a-z0-9])" + re.escape(seller.lower()) + r"(?:'s)?(?![a-z0-9])", text.lower()) if seller else None
    if written is None or (ok is not None and not ok(seller)):
        return None
    said = set(ref_toks(text))
    name = set(ref_toks(seller))
    listed = raw.get("details") if isinstance(raw.get("details"), list) else []
    details = []
    for piece in listed:
        kept = ref_detail(piece) if isinstance(piece, (str, int, float)) and not isinstance(piece, bool) else ""
        if kept and _said_in(kept, said) and not set(ref_toks(kept)) <= name and kept not in details:
            details.append(kept)
    kind = str(raw.get("kind") or "").strip()
    relation = ref_relation_name(raw.get("relation"))
    if relation == "same" and ref_relation(text[:written.start()])[0] == "counterpart":
        relation = "counterpart"
    tolerance = raw.get("tolerance")
    tolerance = float(tolerance) if is_number(tolerance) and relation == "within" else None
    if relation == "within" and (tolerance is None or not figure_written(f"{tolerance:g}", text.lower())):
        relation, tolerance = "same", None
    return {"seller": seller, "kind": (kind if _said_in(kind, said) else "")[:80], "details": details[:6],
            "property": re.split(r"\s+(?:of|for)\s+(?:the|this|these|those|my|your)\b", str(raw.get("property") or "").strip())[0][:60],
            "relation": relation, "tolerance": tolerance,
            "unit": str(raw.get("unit") or "").strip()[:12] if relation == "within" else "", "quote": str(raw.get("quote") or "").strip(),
            "by": "read"}


def ref_partners(value: Any, texts: Any, own: bool = True, prop: Any = "") -> list[str]:
    """The values a store's rule pairs with this one, both ways. On the page that holds the value (own) a rule may leave the value
    unnamed, so it has to name the property instead; a rule on any other page counts only when it names the value."""
    def plain(text: Any) -> str:
        return re.sub(r"[^a-z0-9.]", "", str(text).lower())

    def bare(text: str) -> str:
        return re.sub(r"^(?:that|this|the|our|a|an)\s+|\s+here$", "", text.rpartition(":")[2].strip().lower())

    name = str(value or "").strip().lower()
    want = plain(name)

    def holds(side: str) -> bool:
        # a rule may repeat the property around the value
        return plain(side) == want or side.startswith(name + " ") or side.endswith(" " + name)

    found: list[str] = []
    if not want:
        return found
    for text in texts or []:
        if own:
            for hit in re.finditer(r"\bits [a-z0-9 &~.-]+? goes? (?:together )?with (?:an? )?([^.,;]+)", str(text), re.I):
                if not prop or prop_named(prop, hit.group(0)):
                    found.append(hit.group(1).strip().lower())
        for clause in re.split(r",\s*(?:and\s+)?|;\s*|\.\s+|\s+and\s+(?=[\w ~.-]+?\s+(?:goes|go|pairs?|matches?)\b)", str(text)):
            hit = re.search(r"^\s*(.+?)\s+(?:goes|go|pairs?|matches?)\s+(?:together\s+)?with\s+(.+?)\s*\.?$", clause, re.I)
            if hit is None:
                continue
            left, right = bare(hit.group(1)), bare(hit.group(2))
            if holds(left):
                found.append(right)
            elif holds(right):
                found.append(left)
    return [side for side in dict.fromkeys(found) if plain(side) != want]


def prop_named(prop: Any, text: Any, exact: bool = False) -> bool:
    head = re.sub(r"([a-z])([A-Z])", r"\1 \2", re.split(r"\s*\(", str(prop or ""))[0])
    wanted = re.findall(r"[a-z0-9]+", head.lower())
    tokens = re.findall(r"[a-z0-9]+", str(text or "").lower())
    if not wanted or len(" ".join(wanted)) < 3:
        return False

    def alike(seen: str, word: str) -> bool:
        if seen == word:
            return True
        if exact:
            return False
        if len(word) >= 3 and seen in (word + "s", word + "es"):
            return True
        return len(seen) >= 3 and word in (seen + "s", seen + "es")

    width = len(wanted)
    for start in range(len(tokens) - width + 1):
        if all(alike(seen, word) for seen, word in zip(tokens[start: start + width], wanted)):
            return True
    return False


def ref_span(prop: Any, source: Any) -> str:
    try:
        words = [word for word in re.findall(r"[a-z0-9]+", re.split(r"\s*\(", str(prop))[0].lower()) if word not in LINK_WORDS]
        if not words:
            return ""
        hit = re.search(r"\b" + r"\W+".join(map(re.escape, words)) + r"\b", str(source).lower())
        if not hit:
            return ""
        tail = str(source)[hit.end(): hit.end() + 500]
        return re.split(r"\(\s*or\b|(?<=[.!?])\s", tail)[0]
    except Exception:
        return ""


def _single(names: dict[str, Any], record: dict[str, Any], accept) -> Any:
    # the one field key that fits, or None when zero or several fit
    fitting = [key for low, key in names.items() if accept(low, record[key])]
    return fitting[0] if len(fitting) == 1 else None


def _add_plain_names(record: dict[str, Any]) -> None:
    if not any(isinstance(val, str) and id_field(key) and re.fullmatch(ID_LIKE, val) for key, val in record.items()):
        return
    names = {str(key).lower(): key for key in record}
    if not any(re.search(r"price|cost", low) and is_number(record[key]) for low, key in names.items()):
        key = _single(names, record, lambda low, val: is_number(val) and re.search(r"amount|fee|charge|pay|rate|value", low)
                      and not re.search(r"total|count|qty|quant|unit|due|sum|rating|star|score|page|number|num|size|weight|width|height|length|found|match", low))
        if key is not None:
            record["price"] = record[key]
    if not any(re.search(r"stock|avail", low) for low in names):
        key = _single(names, record, lambda low, val: isinstance(val, bool)
                      and re.search(r"hand|orderable|buyable|sellable|ready|ship|instock|purchas|in_store", low))
        if key is not None:
            record["in_stock"] = record[key]
    if not any(re.search(r"title|heading", low) for low in names):
        key = _single(names, record, lambda low, val: isinstance(val, str) and re.search(r"headline|name|label|caption", low)
                      and not re.search(r"shop|seller|store|vendor|brand|make|categor|shelf|department|id", low))
        if key is not None:
            record["title"] = record[key]


def lensed(piece: Any) -> Any:
    try:
        piece = json.loads(json.dumps(piece))
    except (TypeError, ValueError):
        return piece
    walk_dicts(piece, _add_plain_names)
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
    def kinds_of(cls, schema: dict[str, Any]) -> list[str]:
        declared = schema.get("type")
        if isinstance(declared, str):
            return [declared]
        return list(declared or [])

    @classmethod
    def fits_kind(cls, value: Any, kind: str) -> bool:
        probe = cls.TYPES.get(kind)
        if probe is None:
            return True
        return probe(value)

    @classmethod
    def fits_any(cls, value: Any, kinds: list[str]) -> bool:
        return any(cls.fits_kind(value, kind) for kind in kinds)

    @staticmethod
    def whole_float(value: Any) -> bool:
        return isinstance(value, float) and value.is_integer()

    @classmethod
    def coerce(cls, value: Any, kinds: list[str]) -> Any:
        if not kinds:
            return value
        if cls.fits_any(value, kinds):
            return int(value) if "integer" in kinds and cls.whole_float(value) else value
        for kind in kinds:
            changed = cls.coerce_one(value, kind)
            if changed is not cls.TYPES:
                return changed
        return value

    @classmethod
    def coerce_one(cls, value: Any, kind: str) -> Any:
        if kind == "integer":
            if cls.whole_float(value):
                return int(value)
            if isinstance(value, str) and re.fullmatch(r"\s*-?\d+(\.0+)?\s*", value):
                return int(float(value))
        elif kind == "number":
            if isinstance(value, str) and re.fullmatch(r"\s*-?\d+(\.\d+)?\s*", value):
                figure = float(value)
                return figure if not figure.is_integer() else int(figure)
        elif kind == "string":
            if is_number(value):
                return str(value)
        elif kind == "boolean":
            if isinstance(value, str) and value.lower() in ("true", "false"):
                return value.lower() == "true"
        return cls.TYPES

    @classmethod
    def execute(cls, piece: Any, jschema: Any, where: str = "args") -> tuple[Any, list[str]]:
        if not isinstance(jschema, dict):
            return piece, []
        kinds = cls.kinds_of(jschema)
        value = cls.coerce(piece, kinds)
        if kinds and not cls.fits_any(value, kinds):
            return value, [f"{where}: expected {'/'.join(kinds)}"]
        problems: list[str] = []
        for keyword, apply_rule in cls.RULES:
            if keyword not in jschema:
                continue
            value, found = apply_rule(value, jschema, where)
            problems += found
        return value, problems

    @staticmethod
    def rule_enum(value, schema, where):
        choices = schema["enum"]
        if value in choices:
            return value, []
        if isinstance(value, str):
            wanted = value.strip().lower()
            hit = next((choice for choice in choices if isinstance(choice, str) and choice.lower() == wanted), None)
            if hit is not None:
                return hit, []
        return value, [f"{where}: must be one of {choices}"]

    @staticmethod
    def rule_const(value, schema, where):
        if value == schema["const"]:
            return value, []
        return value, [f"{where}: must equal {schema['const']!r}"]

    @staticmethod
    def rule_short(value, schema, where):
        least = schema["minLength"]
        if not isinstance(value, str) or len(value) >= least:
            return value, []
        return value, [f"{where}: shorter than {least} characters"]

    @staticmethod
    def rule_trim(value, schema, where):
        if isinstance(value, str):
            value = value[: schema["maxLength"]]
        return value, []

    @staticmethod
    def limit_rule(keyword, breaks, wording):
        def check(value, schema, where):
            edge = schema[keyword]
            broken = is_number(value) and is_number(edge) and breaks(value, edge)
            return value, ([f"{where}: {wording} {edge}"] if broken else [])
        return check

    @staticmethod
    def rule_few(value, schema, where):
        least = schema["minItems"]
        if isinstance(value, list) and len(value) < least:
            return value, [f"{where}: needs at least {least} items"]
        return value, []

    @staticmethod
    def rule_many(value, schema, where):
        most = schema["maxItems"]
        if isinstance(value, list) and len(value) > most:
            return value, [f"{where}: allows at most {most} items"]
        return value, []

    @classmethod
    def rule_items(cls, value, schema, where):
        inner = schema["items"]
        if not (isinstance(value, list) and isinstance(inner, dict)):
            return value, []
        fixed, problems = [], []
        for pos, element in enumerate(value):
            element, found = cls.execute(element, inner, f"{where}[{pos}]")
            fixed.append(element)
            problems.extend(found)
        return fixed, problems

    @classmethod
    def rule_fields(cls, value, schema, where):
        if not isinstance(value, dict):
            return value, []
        fields = schema.get("properties") or {}
        problems = [f"{where}.{name}: required" for name in schema.get("required") or [] if name not in value]
        if schema.get("additionalProperties") is False:
            problems.extend(f"{where}.{name}: not an allowed argument" for name in value if name not in fields)
        fixed = {}
        for name, element in value.items():
            if name in fields:
                element, found = cls.execute(element, fields[name], f"{where}.{name}")
                problems.extend(found)
            fixed[name] = element
        return fixed, problems

    @classmethod
    def rule_object(cls, value, schema, where):
        if isinstance(value, dict):
            return cls.rule_fields(value, schema, where)
        return value, []


ArgCheck.RULES = [
    ("enum", ArgCheck.rule_enum),
    ("const", ArgCheck.rule_const),
    ("minLength", ArgCheck.rule_short),
    ("maxLength", ArgCheck.rule_trim),
    ("minimum", ArgCheck.limit_rule("minimum", lambda got, edge: got < edge, "below minimum")),
    ("maximum", ArgCheck.limit_rule("maximum", lambda got, edge: got > edge, "above maximum")),
    ("exclusiveMinimum", ArgCheck.limit_rule("exclusiveMinimum", lambda got, edge: got <= edge, "must be above")),
    ("exclusiveMaximum", ArgCheck.limit_rule("exclusiveMaximum", lambda got, edge: got >= edge, "must be below")),
    ("minItems", ArgCheck.rule_few),
    ("maxItems", ArgCheck.rule_many),
    ("items", ArgCheck.rule_items),
    ("type", ArgCheck.rule_object),
]


class Wire:
    def __init__(self, bind: dict[str, Any]):
        self.t0 = monotonic()
        self.bind = dict(bind)
        self.model_list = ProxyClient(timeout=70, max_retries=1, retry_delay=1)
        self.world = ProxyClient(timeout=95, max_retries=1, retry_delay=1)
        self.spend = 0.0
        self.scale = 1.0
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
        return self.spend >= SPEND_CAP * self.scale

    def send(self, client: Any, path: str, body: dict[str, Any]) -> Any:
        spacing = PACE_GAP if monotonic() >= self.slow_until else SLOW_GAP
        hold = self.last_send + spacing - monotonic()
        if "/inference/" in path:
            hold = max(hold, self.model_resume - monotonic())
        hold = min(hold, self.spare() - 1)
        if hold > 0:
            sleep(hold)
        self.last_send = monotonic()
        reply = client.post_verbose(path, body)
        if reply.data is None:
            if (reply.error or {}).get("status") in (429, 503):
                self.slow_until = monotonic() + SLOW_FOR
        return reply

    def pause(self, seed: str, tries: int, status: Any) -> float:
        spread = (len(seed) * 7 + sum(ord(ch) for ch in seed[-64:]) + 31 * tries + 13 * self.tick) % 101 / 100.0
        first = 2.0 if status == 429 else 1.0
        return min(8.0, first * 2 ** tries) * (0.75 + 0.5 * spread)

    def down(self, at: float | None = None) -> bool:
        if self.no_model:
            return True
        if self.misses <= 0:
            return False
        moment = monotonic() if at is None else at
        return moment - self.last_reply > DOWN_AFTER

    def usable(self, model_list: list[str], tools: bool = False) -> bool:
        if self.no_model or self.mute:
            return False
        for name in list(model_list) + self.spare_models:
            if name in self.banned or (tools and name in self.tool_banned):
                continue
            return True
        return False

    @staticmethod
    def plain(msgs: list[dict[str, Any]]) -> list[dict[str, Any]]:
        flat = []
        for original in msgs:
            copy = dict(original)
            parts = copy.get("content")
            if isinstance(parts, list):
                copy["content"] = "\n\n".join(str(part.get("text") or "") for part in parts if isinstance(part, dict))
            flat.append(copy)
        return flat

    def lone_model(self, ladder: list[str]) -> bool:
        return len({name for name in ladder if name not in self.banned}) <= 1

    def sonnet_too_dear(self, pos: int, model_list: list[str], msgs: list[dict[str, Any]], token_cap: int) -> bool:
        cheaper_next = pos + 1 < len(model_list) and "sonnet" not in model_list[pos + 1]
        guess = sum(len(json.dumps(m, default=str)) for m in msgs) / 2.3 * 2e-6 + min(token_cap, 1500) * 1e-5
        scale = self.scale
        if self.spent_out():
            return True
        return bool(cheaper_next and (self.spend >= SOFT_SPEND * scale or self.spend + guess > SPEND_CAP * scale))

    def query_model(self, msgs: list[dict[str, Any]], model_list: list[str], token_cap: int,
            toolset: list[dict[str, Any]] | None = None) -> dict[str, Any] | None:
        self.call_fail = ""
        self.call_models = set()
        if self.mute or self.no_model:
            return None
        ladder = list(model_list) + [name for name in self.spare_models if name not in model_list]
        trial = {"seed": "", "kinds": set(), "refused": set(), "ladder": ladder, "lone": self.lone_model(ladder)}
        for pos, mdl in enumerate(ladder):
            if self.no_model:
                return self.failed(trial["kinds"])
            if mdl in self.banned or mdl in trial["refused"] or (toolset and mdl in self.tool_banned):
                continue
            if "sonnet" in mdl and self.sonnet_too_dear(pos, model_list, msgs, token_cap):
                continue
            settled, outcome = self.try_model(mdl, msgs, model_list, token_cap, toolset, trial)
            if settled:
                return outcome
        return self.failed(trial["kinds"])

    def take_reply(self, data: dict[str, Any], mdl: str, kinds: set[str]) -> dict[str, Any] | None:
        self.last_reply, self.misses = monotonic(), 0
        usage = data.get("usage") or {}
        if is_number(usage.get("cost")):
            self.spend += float(usage["cost"])
        message = data["choices"][0].get("message") or {}
        if message.get("content") or message.get("tool_calls"):
            return message
        kinds.add("empty")
        self.log.append({"kind": "inference_failure", "model": mdl, "error": "empty"})
        return None

    def drop_model(self, mdl: str, status: Any, raw: str, toolset: Any, model_list: list[str], trial: dict[str, Any]) -> None:
        lowered = raw.lower()
        trial["kinds"].add("hard")
        trial["refused"].add(mdl)
        if status == 404 and toolset and "tool" in lowered:
            self.tool_banned.add(mdl)
        elif not (status == 403 and re.search(r"moderat|flagged", lowered)):
            self.banned.add(mdl)
        allowed = re.search(r'allowed_models"?\s*:\s*\[([^\]]*)', raw) if status == 403 else None
        if not allowed or self.spare_models:
            return
        self.spare_models = [name for name in re.findall(r'"([^"\s]{3,100})"', allowed.group(1))
                             if name not in self.banned and name not in model_list][:2]
        trial["ladder"] += [name for name in self.spare_models if name not in trial["ladder"]]
        trial["lone"] = self.lone_model(trial["ladder"])

    def try_model(self, mdl: str, msgs: list[dict[str, Any]], model_list: list[str], token_cap: int, toolset: Any,
                  trial: dict[str, Any]) -> tuple[bool, Any]:
        kinds = trial["kinds"]
        flatten = False
        for _ in range(1 if self.down() else MODEL_TRIES):
            spare = self.spare()
            if spare < LAST_CALL - 10:
                return True, self.failed(kinds)
            self.model_list.timeout = int(max(8, min(30 if self.down() else 75, spare - LAST_CALL + 15)))
            envelope: dict[str, Any] = {"model": mdl, "messages": self.plain(msgs) if flatten else self.cached(msgs, mdl),
                                        "temperature": 0, "max_tokens": token_cap}
            if toolset:
                envelope["tools"] = toolset
                envelope["tool_choice"] = "auto"
            self.call_models.add(mdl)
            started = monotonic()
            answer = self.send(self.model_list, "/inference/chat/completions", envelope)
            data = answer.data
            if isinstance(data, dict) and data.get("choices"):
                message = self.take_reply(data, mdl, kinds)
                return (True, message) if message is not None else (False, None)
            self.misses += 1
            self.log.append({"kind": "inference_failure", "model": mdl,
                             "error": str(answer.error)[:300] if answer.error else "empty"})
            error = answer.error or {}
            status, raw = error.get("status"), str(error.get("body") or "")
            if status == 402 or (status == 403 and re.search(r"key limit|insufficient|credit", raw.lower())):
                self.no_model = True
                return True, self.failed(kinds)
            if status in (401, 403, 404):
                self.drop_model(mdl, status, raw, toolset, model_list, trial)
                return False, None
            if status in (400, 413, 422):
                kinds.add("hard")
                if flatten:
                    trial["refused"].add(mdl)
                    return False, None
                flatten = True
                continue
            if isinstance(status, int) and 400 <= status < 500 and status not in (408, 425, 429):
                kinds.add("hard")
                trial["refused"].add(mdl)
                return False, None
            kinds.add("soft")
            if monotonic() - started > 20:
                if self.down() and (trial["lone"] or len(self.call_models) >= 2):
                    return True, self.failed(kinds)
                return False, None
            if not trial["seed"]:
                trial["seed"] = json.dumps(msgs[-1:], default=str)[-600:]
            wait = self.pause(trial["seed"] + mdl, min(self.misses - 1, 6), status)
            if self.spare() - wait < LAST_CALL - 10:
                return True, self.failed(kinds)
            self.model_resume = monotonic() + wait
            if self.down(self.model_resume) and (trial["lone"] or len(self.call_models) >= 2):
                return True, self.failed(kinds)
        return False, None

    def failed(self, kinds: set[str]) -> None:
        if "empty" in kinds:
            self.call_fail = "empty"
        elif kinds == {"hard"}:
            self.call_fail = "refused"
        else:
            self.call_fail = ""
        return None

    @staticmethod
    def cached(msgs: list[dict[str, Any]], mdl: str) -> list[dict[str, Any]]:
        if len(msgs) <= 2 or not mdl.startswith("anthropic/"):
            return msgs
        marked = [dict(m) for m in msgs]
        for spot in (0, len(marked) - 1):
            text = marked[spot].get("content")
            if text and isinstance(text, str):
                marked[spot]["content"] = [{"type": "text", "text": text, "cache_control": {"type": "ephemeral"}}]
        return marked

    def consult(self, ask_text: str, token_cap: int, model_list: list[str]) -> dict[str, Any] | None:
        reply = self.query_model([{"role": "user", "content": ask_text}], model_list, token_cap)
        return json_object((reply or {}).get("content") or "")

    def landed(self, data: dict[str, Any], count: int) -> dict[str, Any]:
        step = data.get("step")
        self.tick = step if isinstance(step, int) else self.tick + count
        if data.get("finished"):
            self.done = True
        return data

    def step_clash(self, raw: str, env_body: dict[str, Any], reused: int) -> int | None:
        if "finished" in raw:
            self.done = True
            return None
        named = re.search(r"expected (\d+)", raw)
        if named:
            self.tick = int(named.group(1)) - 1
            env_body["step"] = self.tick + 1
            return reused
        if "reused" in raw and reused < 2:
            env_body["step"] += 1
            return reused + 1
        return None

    def act(self, acts: list[dict[str, Any]]) -> dict[str, Any] | None:
        env_body: dict[str, Any] = dict(self.bind)
        env_body["step"] = self.tick + 1
        if len(acts) != 1:
            env_body["actions"] = acts
        else:
            env_body["action"] = acts[0]
        tries = reused = 0
        for _ in range(3 + WORLD_TRIES):
            spare = self.spare()
            if spare < 2:
                return None
            self.world.timeout = int(max(3, min(95, spare - 1)))
            answer = self.send(self.world, "/environment/call", env_body)
            if isinstance(answer.data, dict):
                return self.landed(answer.data, len(acts))
            error = answer.error or {}
            code, raw = error.get("status"), str(error.get("body") or "")
            self.log.append({"kind": "env_failure", "status": code, "body": raw[:300]})
            if code == 402:
                self.broke = True
                return None
            if code == 409:
                reused = self.step_clash(raw, env_body, reused)
                if reused is None:
                    return None
                continue
            if code in (403, 422) or tries >= WORLD_TRIES - 1 or (code in (400, 401, 404) and tries >= 1):
                return None
            wait = self.pause(json.dumps(env_body, default=str)[-600:], tries, code)
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

    @staticmethod
    def own_label(d: dict[str, Any]) -> str:
        for key, val in d.items():
            if not isinstance(val, str) or not 0 < len(val.strip()) <= 60:
                continue
            low = str(key).lower()
            if re.search(r"categor|shelf|department|aisle", low) and "id" not in low:
                return val.strip()
        return ""

    @staticmethod
    def own_id(d: dict[str, Any]) -> str | None:
        for key, val in d.items():
            if not isinstance(val, str):
                continue
            low = str(key).lower()
            if low in ("id", "_id") or (low.endswith(("id", "key")) and re.fullmatch(ID_LIKE, val)):
                return val
        return None

    @staticmethod
    def price_field(d: dict[str, Any]) -> float | None:
        for key, val in d.items():
            if is_number(val) and not isinstance(val, bool) and re.search(r"price|cost", str(key).lower()):
                return float(val)
        return None

    def read_id(self, d: dict[str, Any]) -> str | None:
        return next((val for key, val in d.items() if isinstance(val, str) and id_field(key) and val in self.read), None)

    def note_record(self, d: dict[str, Any]) -> None:
        for key, val in d.items():
            if isinstance(val, str) and id_field(key) and re.fullmatch(ID_LIKE, val):
                self.ids.add(val)
        ident = self.own_id(d)
        if not ident:
            return
        if any(isinstance(inner, dict) and sum(1 for v in inner.values() if isinstance(v, (str, int, float))) >= 2
               for inner in d.values()):
            self.read.add(ident)
        label = self.own_label(d)
        if label:
            self.label_of.setdefault(ident, label)

    def add(self, obs: Any) -> None:
        self.items.append(obs)
        walk_dicts(obs, self.note_record)

    def price(self, thing: str) -> float | None:
        seen: list[float] = []

        def look(d: dict[str, Any]) -> None:
            if any(isinstance(val, str) and val == thing and id_field(key) for key, val in d.items()):
                figure = self.price_field(d)
                if figure is not None:
                    seen.append(figure)
        for record in self.items:
            walk_dicts(record, look)
        return seen[-1] if seen else None

    def gather_about(self, piece: Any, thing: str, found: list[str], depth: int) -> None:
        if depth > 6 or len(found) >= 8:
            return
        if isinstance(piece, dict):
            if any(isinstance(val, str) and val == thing for val in piece.values()):
                text = as_text(clipped(piece, 400))[:2500]
                if text not in found:
                    found.append(text)
                return
            children = list(piece.values())
        elif isinstance(piece, list):
            children = piece
        else:
            return
        for child in children:
            self.gather_about(child, thing, found, depth + 1)

    def about(self, thing: str, hold: int = 4) -> list[str]:
        found: list[str] = []
        for record in self.items:
            self.gather_about(record, thing, found, 0)
        return found[-hold:]

    def merge_cards(self) -> tuple[dict[str, dict[str, Any]], list[str]]:
        merged: dict[str, dict[str, Any]] = {}
        order: list[str] = []

        def fold(d: dict[str, Any]) -> None:
            ident = next((val for key, val in d.items() if isinstance(val, str) and id_field(key)
                          and re.fullmatch(ID_LIKE, val) and val in self.ids), None)
            if not ident:
                return
            mapped = any(isinstance(val, dict) and len(val) >= 2 for val in d.values())
            if len(d) < 4 and not mapped:
                return
            card = merged.setdefault(ident, {})
            if ident not in order:
                order.append(ident)
            card.update({key: val for key, val in d.items()})
            if mapped:
                self.read.add(ident)
        for record in self.items:
            walk_dicts(record, fold)
        return merged, order

    def cards(self, char_cap: int = 30000, cheaper_than: str = "", skip: set[str] | None = None, first: Any = ()) -> list[str]:
        merged, order = self.merge_cards()

        def cost(ident: str) -> float | None:
            return self.price_field(merged.get(ident) or {})

        ceiling = cost(cheaper_than) if cheaper_than else None
        if skip:
            order = [i for i in order if i == cheaper_than or i not in skip]
        if ceiling is not None:
            order = [i for i in order if i == cheaper_than or cost(i) is None or cost(i) < ceiling]
        lead = set(first or ())
        ranked = sorted(order, key=lambda i: (i not in lead, i not in self.read, order.index(i)))
        texts = {}
        for ident in ranked:
            texts[ident] = ("DETAILS " if ident in self.read else "ROW ") + as_text(clipped(merged[ident], 160))
        if sum(len(t) for t in texts.values()) > char_cap:
            def sold_out(ident: str) -> bool:
                return any(Observer.key_stock(key, val) is False for key, val in merged[ident].items())
            ranked = sorted(ranked, key=lambda i: (i not in lead, i not in self.read, sold_out(i), order.index(i)))
        out, used = [], 0
        for ident in ranked:
            text = texts[ident]
            if used + len(text) <= char_cap:
                out.append(text)
                used += len(text)
        return out

    def label_for(self, d: dict[str, Any]) -> str:
        label = self.own_label(d)
        if label:
            return label
        known = self.label_of or {}
        for key, val in d.items():
            if isinstance(val, str) and val in known and id_field(key):
                return known[val]
        return ""

    def rich_reads(self, shelf: str = "") -> int:
        rich: set[str] = set()
        wanted = shelf.strip().lower()

        def judge(d: dict[str, Any]) -> None:
            ident = self.read_id(d)
            if not ident:
                return
            if shelf and self.label_for(d).strip().lower() != wanted:
                return
            maps = [val for val in d.values() if isinstance(val, dict)]
            if maps:
                if any(len(val) >= 3 for val in maps):
                    rich.add(ident)
            elif sum(1 for val in d.values() if isinstance(val, str) and 0 < len(val) <= 70) >= 5:
                rich.add(ident)
        for record in self.items:
            walk_dicts(record, judge)
        return len(rich)

    def shelves_read(self) -> list[str]:
        counts: dict[str, int] = {}

        def tally(d: dict[str, Any]) -> None:
            if not self.read_id(d):
                return
            label = self.label_for(d)
            if label and any(isinstance(val, dict) for val in d.values()):
                counts[label] = counts.get(label, 0) + 1
        for record in self.items:
            walk_dicts(record, tally)
        return sorted(counts, key=lambda label: -counts[label])

    def menu_scan(self, piece: Any, shelf: str, inside: bool, depth: int, sink) -> None:
        if depth > 6:
            return
        if isinstance(piece, list):
            for child in piece:
                self.menu_scan(child, shelf, inside, depth + 1, sink)
            return
        if not isinstance(piece, dict):
            return
        if not inside:
            inside = not shelf or self.on_shelf(piece, shelf)
        if inside:
            sink(piece)
        for child in piece.values():
            if isinstance(child, (dict, list)):
                self.menu_scan(child, shelf, inside, depth + 1, sink)

    def on_shelf(self, d: dict[str, Any], shelf: str) -> bool:
        for key, val in d.items():
            if isinstance(val, str) and val.strip().lower() == shelf.lower() and re.search(r"categor|shelf|department|aisle", str(key).lower()):
                return True
        return self.label_for(d).strip().lower() == shelf.strip().lower()

    def fact_sets(self, shelf: str = "") -> list[dict[str, str]]:
        sets: list[dict[str, str]] = []

        def take(d: dict[str, Any]) -> None:
            short = {str(key): val for key, val in d.items() if isinstance(val, str) and 0 < len(val) <= 70 and not junk_key(key)}
            if len(short) >= 5 and short not in sets:
                sets.append(short)

        for record in self.items:
            self.menu_scan(record, shelf, False, 0, take)
        return sets

    def prop_menu(self, shelf: str = "") -> tuple[str, int]:
        totals: dict[str, int] = {}
        row_totals: dict[str, int] = {}
        samples: dict[str, list[str]] = {}
        tally = {"sets": 0, "mapped": 0}

        def take(d: dict[str, Any]) -> None:
            short = [(key, val) for key, val in d.items() if isinstance(val, str) and 0 < len(val) <= 70 and not junk_key(key)]
            if len(short) < 5:
                flat_row = shelf and len(short) >= 2 and not any(isinstance(v, dict) for v in d.values())
                if not flat_row:
                    return
            tally["sets"] += 1
            row_like = len(short) < 5 and any(isinstance(val, str) and id_field(key) for key, val in d.items())
            if not row_like:
                tally["mapped"] += 1
            for key, val in short:
                if re.search(r"\d", val) and re.fullmatch(CODE_LIKE, val) and ("-" in val or "_" in val):
                    continue
                name = re.sub(r"[_.]+", " ", str(key)).strip().lower()
                totals[name] = totals.get(name, 0) + 1
                if row_like:
                    row_totals[name] = row_totals.get(name, 0) + 1
                bucket = samples.setdefault(name, [])
                if len(bucket) < 5 and val.lower() not in bucket:
                    bucket.append(val.lower())

        for record in self.items:
            self.menu_scan(record, shelf, False, 0, take)
        if tally["mapped"]:
            for name in [name for name in totals if row_totals.get(name, 0) >= totals[name]]:
                del totals[name]
        best = sorted(totals, key=lambda name: -totals[name])[:28]
        listing = "\n".join(f"- {name} (in {totals[name]} listings): " + " | ".join(samples.get(name, [])) for name in best)
        return listing, tally["sets"]

    def label_lines(self) -> str:
        seen: dict[str, dict[str, int]] = {}

        def count(d: dict[str, Any]) -> None:
            if len(d) < 4:
                return
            for key, val in d.items():
                if not isinstance(val, str) or not 0 < len(val) <= 40:
                    continue
                if re.search(r"\d", val) and re.fullmatch(CODE_LIKE, val):
                    continue
                per_value = seen.setdefault(str(key), {})
                per_value[val] = per_value.get(val, 0) + 1
        for record in self.items:
            walk_dicts(record, count)
        out = []
        for key, per_value in seen.items():
            if sum(per_value.values()) < 4 or not 2 <= len(per_value) <= 8 or max(per_value.values()) < 2:
                continue
            ordered = sorted(per_value.items(), key=lambda pair: -pair[1])
            out.append(f"- {key}: " + " | ".join(f"{val} (x{times})" for val, times in ordered))
        return "\n".join(out[:12])


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
        self.pending: list[Any] | None = None
        self.pending_at = 0.0
        self.pinned_heard = 0
        self.meta_last: str | None = None
        self.fresh_needs: list[str] | None = None
        self.field_hook = None
        self.ref_hook = None

    @staticmethod
    def whole_count(value: Any) -> bool:
        return isinstance(value, int) and not isinstance(value, bool) and value >= 1

    def items(self) -> list[dict[str, Any]]:
        products = []
        for entry in self.reading.get("items") or []:
            if not isinstance(entry, dict):
                continue
            name = str(entry.get("product") or "").strip()
            if not name:
                continue
            asked = entry.get("quantity")
            listed = entry.get("requirements")
            listed = listed if isinstance(listed, list) else []
            products.append({"product": str(entry["product"]).strip(),
                             "quantity": asked if self.whole_count(asked) else 1,
                             "requirements": [str(req) for req in listed if req and ":" in str(req)]})
        return products if len(products) > 1 else []

    def quantity(self) -> int:
        if self.items():
            return 1
        asked = self.reading.get("quantity")
        return asked if self.whole_count(asked) else 1

    def counts(self) -> list[int]:
        per_product = [entry["quantity"] for entry in self.items()]
        return per_product if per_product else [self.quantity()]

    def items_text(self) -> str:
        products = self.items()
        if not products:
            return ""
        numbered = " | ".join(f"{pos}) {entry['product']} x{entry['quantity']}" for pos, entry in enumerate(products, 1))
        return ("THE REQUEST ASKS FOR SEVERAL PRODUCTS; the order carries one line per product at that product's quantity and "
                "nothing more; a listing is measured only against the requirements of its own product (its category settles "
                "which), and the cheapest or best rule works inside each product: " + numbered)

    def priority(self) -> str:
        rule = self.reading.get("priority")
        return rule if rule in ("cheapest", "best") else "none"

    def props_left(self) -> list[str]:
        if self.now_reqs is None:
            pool = [req for req in self.written + self.uncovered if req not in self.let_go]
        else:
            pool = self.now_reqs
        return [req for req in pool if req not in self.let_go
                and not re.search(r"\bprice\b|budget|quantity|\bstock\b|\bcents?\b", req.lower())]

    def text_body(self) -> str:
        if self.now_reqs is None:
            shown = [f"{req} (stated)" for req in self.written if req not in self.let_go]
            shown += [f"{req} (revealed by the shopper)" for req in self.uncovered if req not in self.let_go]
        else:
            shown = list(self.now_reqs)
        if shown:
            text = "; ".join(shown)
        elif self.let_go:
            text = "no property requirement remains: any in-stock listing of the requested category fits"
        else:
            text = "see the request"
        if self.spend_cap is not None:
            text += f"; total price at most {self.spend_cap} {self.reading.get('currency') or ''}".rstrip()
        text += "; " + self.items_text() if self.items() else f"; quantity {self.quantity()}"
        if self.let_go:
            text += "; no longer required: " + "; ".join(self.let_go)
        if self.indifferent:
            text += "; shopper does not care about: " + ", ".join(self.indifferent)
        return text

    def changes(self) -> str:
        if not self.let_go:
            return ""
        return ("\n\nCHANGES SINCE THE REQUEST (these win over its wording): the shopper DROPPED " + "; ".join(self.let_go)
                + ". Treat it as gone: no listing fails, ranks or is preferred because of it.")

    def overview(self, step_count: int, msgs: int | None) -> str:
        got = self.reading
        if not got:
            return ""
        out = ["Initial reading of the request (verify it against the request text):"]
        if got.get("product"):
            out.append(f"- product: {got.get('product')}")
        products = self.items()
        if not products:
            out.append(f"- quantity: {self.quantity()}")
        else:
            for pos, entry in enumerate(self.items(), 1):
                out.append(f"- product {pos}: {entry['product']}, quantity {entry['quantity']} (one cart line)")
            out.append("- the order consists of these lines only, one for each product; whenever the cart tool accepts a list of lines, every cart call sends all of them")
        out.extend(f"- stated requirement: {req}" for req in got.get("requirements") or [])
        out.extend(f"- value set by pointing at another listing (open it first, then search using its value): {req}" for req in got.get("references") or [])
        if got.get("price_cap") is not None:
            out.append(f"- price cap (total): {got.get('price_cap')} {got.get('currency') or ''}".rstrip())
        if got.get("in_stock_only"):
            out.append("- only items in stock / on hand")
        if self.priority() == "none":
            out.append("- the request sets no cheapest/best rule: an item meeting all requirements inside the budget is correct, so no steps go into chasing the lowest price (unless the request itself says otherwise)")
        else:
            detail = f" ({got.get('priority_detail')})" if got.get("priority_detail") else ""
            out.append(f"- among all items that fit, pick the {self.priority()}{detail}")
        out.extend(f"- a wish, NOT a requirement (it only breaks ties): {want}" for want in got.get("soft_wants") or [])
        out.extend(f"- to learn from the shopper before choosing: {item}" for item in got.get("ask_first") or [])
        out.extend(f"- rule: {req}" for req in got.get("rules") or [])
        if msgs is not None:
            out.append(f"- messages to the shopper allowed: {msgs}")
        out.append(f"- steps allowed: {step_count}")
        return "\n".join(out)

    @staticmethod
    def line_names(needs: list[Any], lns: list[str]) -> dict[str, str]:
        tokens = re.findall(r"[a-z]+", " ".join(lns).lower())
        spoken = " ".join(tokens)
        renamed: dict[str, str] = {}
        for need in needs:
            if not (isinstance(need, str) and ":" in need):
                continue
            head, _, value = need.partition(":")
            tag = head[head.find("("):].strip() if "(" in head else ""
            stems = re.findall(r"[a-z]+", re.split(r"\s*\(", head)[0].lower())
            if not stems or min(len(stem) for stem in stems) < 3:
                continue
            if re.search(r"\b" + r"\s+".join(stems) + r"\b", spoken):
                continue
            width = len(stems)
            for start in range(len(tokens) - width + 1):
                window = tokens[start: start + width]
                if window == stems or not all(word.startswith(stem) for stem, word in zip(stems, window)):
                    continue
                renamed[need] = " ".join(window) + (" " + tag if tag else "") + ":" + value
                break
        return renamed

    @staticmethod
    def tag_list(text: str) -> list[str]:
        return re.findall(r"\(([^)]*)\)", text)

    def drop_named(self, old_need: str, lns: list[str]) -> bool:
        try:
            return self.drop_allowed(old_need, lns)
        except Exception:
            return True

    def drop_allowed(self, old_need: str, lns: list[str]) -> bool:
        name = str(old_need).split(":", 1)[0]
        tag = self.tag_list(name)

        def same_slot(other: str) -> bool:
            head = other.split(":", 1)[0]
            return same_prop(head, name) and (not tag or self.tag_list(head) in ([], tag))

        matching = [req for req in self.written if same_slot(req)]
        if not matching:
            fresh = [n for n in self.fresh_needs or [] if isinstance(n, str) and ":" in n]
            matching = [req for req in self.uncovered if same_slot(req) and req not in self.written and req not in fresh]
            if not matching or any(same_slot(n) and n != old_need for n in fresh):
                return True
        tokens = re.findall(r"[a-z]+", " ".join(lns).lower())
        phrases = set()
        for size in (1, 2, 3):
            for start in range(len(tokens) - size + 1):
                phrases.add(" ".join(tokens[start: start + size]))
        phrases = {p for p in phrases if len(p) >= 3 and p not in LINK_WORDS}
        if any(same_prop(req.split(":", 1)[0], p) for req in matching for p in phrases):
            return True
        return bool(self.field_hook) and self.dropped_by_field(matching, name, phrases)

    def dropped_by_field(self, matching: list[str], name: str, phrases: set[str]) -> bool:
        for req in matching:
            head, _, raw_value = req.partition(":")
            cleaned = re.sub(r"\((?:stated|revealed[^)]*)\)", " ", raw_value)
            if ref_text(cleaned) or not ch_alts(cleaned):
                continue
            fields = self.field_hook(head) or {}
            rivals = [r.split(":", 1)[0] for r in self.written + self.uncovered
                      if isinstance(r, str) and ":" in r and not same_prop(r.split(":", 1)[0], name)]
            phrase = self.phrase_naming_field(phrases, rivals, fields, cleaned)
            if phrase is not None:
                self.wire.log.append({"kind": "drop_by_field", "need": req[:200], "phrase": phrase})
                return True
        return False

    @staticmethod
    def phrase_naming_field(phrases: set[str], rivals: list[str], fields: dict[str, Any], value: str) -> str | None:
        for phrase in phrases:
            if any(strict_prop(rival, phrase) for rival in rivals):
                continue
            for key, vals in fields.items():
                if strict_prop(key, phrase) and value_seen(value, " | ".join(vals)):
                    return phrase
        return None

    @staticmethod
    def plain_words(text: str) -> str:
        return " ".join(re.findall(r"[a-z0-9]+", text))

    def literal_values(self, answer: dict[str, Any]) -> None:
        if not self.field_hook:
            return
        swaps: dict[str, str] = {}
        for need in answer.get("needs") or []:
            if not (isinstance(need, str) and ":" in need):
                continue
            head, _, value = need.partition(":")
            if re.search(r"\d", value) or ref_text(value):
                continue
            bare = self.plain_words(re.sub(CJ_COMPARE, " ", value.lower()))
            if not bare:
                continue
            if bare == self.plain_words(value.lower()) and positive_value(value, head):
                continue
            prop = re.split(r"\s*\(", head)[0].strip()
            for key, vals in (self.field_hook(head) or {}).items():
                if not strict_prop(key, prop):
                    continue
                match = next((v for v in vals if self.plain_words(str(v).lower()) == bare), None)
                if match is None:
                    continue
                LITERAL_VALUES.add(bare)
                swaps[need] = f"{head.strip()}: {str(match).strip().lower()}"
                break
        if swaps:
            self.swap_needs(answer, swaps)
            self.wire.log.append({"kind": "literal_value", "needs": list(swaps.values())[:4]})

    @staticmethod
    def swap_needs(answer: dict[str, Any], swaps: dict[str, str]) -> None:
        answer["needs"] = [swaps.get(n, n) if isinstance(n, str) else n for n in answer.get("needs") or []]
        if isinstance(answer.get("requirements_now"), list):
            answer["requirements_now"] = [swaps.get(r, r) if isinstance(r, str) else r for r in answer["requirements_now"]]

    def pinned(self, prop_name: str, qtext: str | None) -> bool:
        try:
            listed = self.reading.get("references")
            refs = [str(r) for r in listed or []] if isinstance(listed, list) else []
            refs += list(self.resolved or [])
            tag = self.tag_list(str(prop_name))
            asked_key = prop_key(prop_name)
            for req in self.written:
                if req in self.let_go or ":" not in req:
                    continue
                head, _, value = req.partition(":")
                own_tag = self.tag_list(head)
                if tag and own_tag and not (kind_words(tag[-1]) & kind_words(own_tag[-1])):
                    continue
                if not positive_value(value, head) or story_guess(req, self.request, refs):
                    continue
                own_key = prop_key(head)
                if same_prop(head, prop_name) or (len(own_key) >= 4 and asked_key.endswith(own_key)):
                    return True
        except Exception:
            return False
        return False

    @staticmethod
    def valued(needs: Any, lns: list[str]) -> bool:
        if not isinstance(needs, list):
            return False
        spoken = " ".join(str(line) for line in lns).lower()
        for need in needs:
            head, _, value = str(need).partition(":")
            prop_bits = [bit for bit in prop_words(head) if len(bit) >= 3]
            value_bits = [bit for bit in re.findall(r"[a-z0-9]+", value.lower()) if bit not in LINK_WORDS]
            if not prop_bits or not value_bits:
                continue
            if all(bit in spoken for bit in prop_bits + value_bits):
                return True
        return False

    def listener_prompt(self, fresh: list[str], qtext: str | None, so_far: str) -> str:
        parts = [HEAR, "\n\nTHE SHOPPER'S REQUEST (for context):\n", self.request, "\n\nREQUIREMENTS SO FAR:\n", so_far]
        if self.exchanges:
            parts += ["\n\nEARLIER EXCHANGES WITH THE SHOPPER (oldest first):\n", "\n".join(self.exchanges[-12:])]
        if qtext:
            parts += ["\n\nTHE ASSISTANT ASKED:\n", qtext]
        parts += ["\n\nTHE SHOPPER NOW SAID:\n", "\n".join(fresh)]
        return "".join(parts)

    def hold_unheard(self, fresh: list[str], qtext: str | None) -> None:
        self.caught.difference_update(fresh)
        self.exchanges.pop()
        self.pending = list(self.pending or []) + [(list(fresh), qtext)]
        self.pending_at = monotonic()

    def only_meta(self, answer: dict[str, Any], fresh: list[str], qtext: str | None) -> dict[str, Any]:
        try:
            if answer.get("meta_only") is True and qtext and not self.valued(answer.get("needs"), fresh):
                answer = dict(answer, needs=[], no_preference=[], dropped=[], references=[], requirements_now=None,
                              new_budget=None, unavailable=False, reply="")
                self.meta_last = qtext
                self.wire.log.append({"kind": "meta_only", "line": " / ".join(fresh)[:200]})
        except Exception:
            pass
        return answer

    def keep_line_names(self, answer: dict[str, Any], fresh: list[str]) -> None:
        try:
            renames = self.line_names(answer.get("needs") or [], fresh)
        except Exception:
            renames = {}
        if not renames:
            return
        self.swap_needs(answer, renames)
        self.wire.log.append({"kind": "need_renamed", "names": list(renames.values())[:4]})

    def masked_lines(self, answer: dict[str, Any], fresh: list[str], qtext: str | None) -> list[str]:
        if not self.ref_hook:
            return fresh
        try:
            return self.ref_hook(answer, fresh, qtext) or fresh
        except Exception as err:
            self.wire.log.append({"kind": "ref_line_error", "error": f"{type(err).__name__}: {err}"[:300]})
            return fresh

    def note_pointed(self, answer: dict[str, Any], fresh: list[str]) -> None:
        try:
            given = answer.get("references")
            infos = [r for r in given or [] if isinstance(r, dict)] if isinstance(given, list) else []
            for need in answer.get("needs") or []:
                if not isinstance(need, str) or need not in self.uncovered:
                    continue
                if ref_text(need.partition(":")[2]) or need in (answer.get("ref_needs") or []):
                    self.ref_news = list(self.ref_news or []) + [{"need": need, "infos": infos, "line": " ".join(fresh)}]
        except Exception:
            pass

    def free_props(self, answer: dict[str, Any], qtext: str | None) -> None:
        source = self.written + self.uncovered if self.now_reqs is None else self.now_reqs
        held = {r.split(":", 1)[0].strip().lower() for r in source if isinstance(r, str) and ":" in r and r not in self.let_go}
        for prop_name in answer.get("no_preference") or []:
            if not (isinstance(prop_name, str) and prop_name) or prop_name in self.indifferent:
                continue
            if prop_name.strip().lower() in held:
                continue
            if self.pinned(prop_name, qtext):
                self.pinned_heard = self.pinned_heard + 1
                continue
            self.indifferent.append(prop_name)

    def take_drops(self, answer: dict[str, Any], masked: list[str], remarks: list[str]) -> bool:
        widened = False
        new_budget = is_number(answer.get("new_budget"))
        for old_need in answer.get("dropped") or []:
            if not (isinstance(old_need, str) and old_need) or old_need in self.let_go:
                continue
            if new_budget and re.search(r"\b(?:prices?|budget|costs?|spend\w*|cents?|dollars?)\b", old_need.lower()):
                continue
            if not self.drop_named(old_need, masked):
                self.wire.log.append({"kind": "drop_kept", "need": old_need[:200]})
                continue
            self.let_go.append(old_need)
            remarks.append(f"dropped: {old_need}")
            widened = widened or self.priority() != "none"
        return widened

    def restore_lost(self, masked: list[str]) -> None:
        for need in self.uncovered:
            prop_name = need.split(":", 1)[0].strip().lower()
            if not prop_name:
                continue
            if any(prop_name in req.lower() for req in self.now_reqs) or any(prop_name in old.lower() for old in self.let_go):
                continue
            self.now_reqs.append(need)
        for need in self.written:
            try:
                self.restore_stated(need, masked)
            except Exception:
                continue

    def restore_stated(self, need: str, masked: list[str]) -> None:
        if need in self.let_go or need in self.now_reqs or self.drop_named(need, masked):
            return
        name = need.split(":", 1)[0]
        tag = self.tag_list(name)

        def same_slot(other: str) -> bool:
            head = other.split(":", 1)[0]
            return same_prop(name, head) and self.tag_list(head) == tag

        if any(same_slot(old) for old in self.let_go):
            return
        if not any(same_slot(req) for req in self.now_reqs if isinstance(req, str)):
            self.now_reqs.append(need)

    def widened_remark(self) -> str:
        if self.props_left():
            return ("a requirement the shopper let go OPENS the field: rows that failed on that one point now pass, and rows "
                    "no search has shown yet may undercut the current pick; search the kind once more without it, using short "
                    "queries of each remaining required value with the kind word, value first, reading "
                    "every page, and choose the cheapest over everything acceptable now")
        if self.priority() == "cheapest" and self.sweepable:
            return ("nothing about the properties is required any more: any in-stock row of the requested category passes, "
                    "of any sort (accessories, parts and sets on that shelf too), and the lowest price on the full shelf "
                    "takes it. Rows arrive in relevance or id order, never by price, so the next shelf slices are fetched for "
                    "you (the listing pages, the label's words, the object words in titles); if no "
                    "'[shelf sweep done]' list follows, cover the shelf yourself with short queries, the first page of "
                    "each - the category label, each kind word of the label alone, each item name that recurs in the "
                    "titles seen - and the category listing's pages. Then make the lowest-priced in-stock row on the "
                    "requested category within the budget the cart's only line, re-check its stock and price, and order it")
        return ("nothing about the properties is required any more: any in-stock row of the requested category passes, "
                "of any sort (accessories, parts and sets on that shelf too), and the lowest price on the full shelf "
                "takes it. Rows arrive in relevance or id order, never by price: cover the shelf with short queries, the first page "
                "of each - the category label, each kind word of the label alone, each item name that recurs in the "
                "titles seen - and the category listing's pages; then order the lowest in-stock price")

    def hear(self, lns: list[str], qtext: str | None) -> str:
        fresh = [line for line in lns if line not in self.caught]
        if not fresh:
            return ""
        self.caught.update(fresh)
        if self.wire.spare() < CLOSING:
            return ""
        so_far = self.text_body()
        prompt = self.listener_prompt(fresh, qtext, so_far)
        self.exchanges.append((f"asked: {qtext} -> " if qtext else "") + "shopper: " + " / ".join(fresh))
        answer = self.wire.consult(prompt, 2500, LISTENER)
        if not answer:
            if self.wire.mute or self.wire.misses > 0:
                self.hold_unheard(fresh, qtext)
            return ""
        answer = self.only_meta(answer, fresh, qtext)
        self.keep_line_names(answer, fresh)
        masked = self.masked_lines(answer, fresh, qtext)
        heard = (so_far + " " + " ".join(self.exchanges) + " " + (qtext or "")).lower()

        def grounded(req: str) -> bool:
            words = [w for w in re.findall(r"[a-z]+", req.lower()) if len(w) >= 3]
            return not words or any(w in heard for w in words)

        try:
            self.literal_values(answer)
        except Exception as err:
            self.wire.log.append({"kind": "literal_value_error", "error": f"{type(err).__name__}: {err}"[:300]})
        self.fresh_needs = [n for n in answer.get("needs") or [] if isinstance(n, str)]
        stated_now = answer.get("requirements_now")
        if isinstance(stated_now, list) and all(isinstance(req, str) for req in stated_now):
            self.now_reqs = [req for req in stated_now if req.strip() and grounded(req)]
        remarks = []
        for need in answer.get("needs") or []:
            if isinstance(need, str) and need and need not in self.uncovered and grounded(need):
                self.uncovered.append(need)
                remarks.append(f"new requirement: {need}")
        self.note_pointed(answer, fresh)
        self.free_props(answer, qtext)
        widened = self.take_drops(answer, masked, remarks)
        budget = answer.get("new_budget")
        if is_number(budget) and budget > 0 and budget != self.spend_cap:
            self.spend_cap = budget
            remarks.append(f"new budget: {budget}")
        if answer.get("unavailable"):
            remarks.append("an item became unavailable; take it out of the cart and do not order it")
        if remarks:
            self.version += 1
            if widened:
                self.widened_at = self.version
                self.widened_tick = self.wire.tick
        if self.now_reqs is not None:
            self.restore_lost(masked)
        if widened:
            remarks.append(self.widened_remark())
        self.wire.log.append({"kind": "heard", "question": qtext is not None, "needs": len(answer.get("needs") or []),
                              "free": len(answer.get("no_preference") or []), "dropped": len(answer.get("dropped") or []),
                              "budget": is_number(budget)})
        asked_back = qtext is None and any("?" in line for line in fresh) and not answer.get("no_preference")
        self.reply_due = str(answer.get("reply") or "").strip() if asked_back else ""
        tail = " (" + "; ".join(remarks) + ")" if remarks else ""
        return "Meaning: " + str(answer.get("meaning") or "") + tail


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
        chosen = {}
        for key, val in call_args.items():
            spec = self.prop_specs.get(key)
            if isinstance(spec, dict) and "enum" in spec:
                chosen[key] = val
        return as_text(chosen)

    def line_spec(self) -> tuple[str, str, str] | None:
        for key, spec in self.prop_specs.items():
            if not (isinstance(spec, dict) and spec.get("type") == "array"):
                continue
            inner = spec.get("items")
            fields = inner.get("properties") if isinstance(inner, dict) else None
            if not isinstance(fields, dict):
                continue
            texts = [name for name, sub in fields.items() if isinstance(sub, dict) and sub.get("type") == "string" and "enum" not in sub]
            id_fields = [name for name in texts if id_field(name)] or texts
            numbers = [name for name, sub in fields.items() if isinstance(sub, dict) and sub.get("type") in ("integer", "number")]
            if id_fields and numbers:
                return key, id_fields[0], numbers[0]
        return None

    def listed_lines(self, call_args: dict[str, Any]) -> list[tuple[str, float | None]] | None:
        spec = self.line_spec()
        if not spec or not isinstance(call_args.get(spec[0]), list):
            return None
        array_key, id_key, count_key = spec
        found = []
        for row in call_args[array_key]:
            if not isinstance(row, dict):
                continue
            ident = row.get(id_key)
            if isinstance(ident, str) and ident.strip():
                found.append((ident.strip(), float(row[count_key]) if is_number(row.get(count_key)) else None))
        return found

    def lines(self, call_args: dict[str, Any]) -> list[tuple[str, float | None]]:
        listed = self.listed_lines(call_args)
        if listed is not None:
            return listed
        thing = self.aim_flat(call_args)
        if not thing:
            return []
        return [(thing, self.count_arg(call_args))]

    @staticmethod
    def first_put(found: list[tuple[str, float | None]]) -> tuple[str, float | None] | None:
        if not found:
            return None
        putting = [line for line in found if line[1] is None or line[1] > 0]
        return (putting or found)[0]

    def aim(self, call_args: dict[str, Any]) -> str | None:
        line = self.first_put(self.lines(call_args))
        return line[0] if line else None

    def aim_flat(self, call_args: dict[str, Any]) -> str | None:
        for key, piece in call_args.items():
            spec = self.prop_specs.get(key)
            if not isinstance(spec, dict):
                spec = {}
            if not isinstance(piece, str) or "enum" in spec:
                continue
            bare = piece.strip()
            if bare and " " not in bare:
                return piece
        return None

    def count_arg(self, call_args: dict[str, Any]) -> float | None:
        if self.line_spec() and isinstance(call_args.get(self.line_spec()[0]), list):
            line = self.first_put(self.lines(call_args))
            return line[1] if line else None
        for key, spec in self.prop_specs.items():
            if not isinstance(spec, dict) or "enum" in spec or spec.get("type") not in ("integer", "number"):
                continue
            piece = call_args.get(key, spec.get("default"))
            if is_number(piece):
                return float(piece)
        return None

    def is_put(self, call_args: dict[str, Any]) -> bool:
        return self.put_form is None or self.put_form == self.form(call_args)

    def held(self, thing: str) -> float:
        return self.units.get(thing, 0.0)

    def settle_view(self, obs: Any, whole: bool = True) -> bool:
        seen = self.view_units(obs)
        if seen is None:
            return False
        # an edit reply naming one line says nothing about the other lines (stores answer per line)
        if not whole and len(seen) == 1:
            kept = {thing: qty for thing, qty in self.units.items() if thing not in seen}
            self.learn(seen)
            self.units.update(kept)
            return True
        self.learn(seen)
        return True

    def book_list(self, call_args: dict[str, Any], obs: Any) -> str:
        self.adds = False
        self.units = {}
        for thing, count in self.lines(call_args):
            if count is not None and count <= 0:
                continue
            self.units[thing] = self.units.get(thing, 0.0) + (1.0 if count is None else count)
        self.settle_view(obs)
        return self.summary()

    def booked(self, call_args: dict[str, Any], obs: Any) -> str:
        self.used = True
        if isinstance(obs, dict) and obs.get("error"):
            return ""
        if self.put_form is None:
            self.put_form = self.form(call_args)
        if self.line_spec() and isinstance(call_args.get(self.line_spec()[0]), list):
            return self.book_list(call_args, obs)
        thing = self.aim(call_args)
        if not thing:
            return ""
        count = self.count_arg(call_args)
        if count is None:
            count = 1.0
        before = self.held(thing)
        putting = self.is_put(call_args)
        if not putting:
            self.units[thing] = 0.0 if count <= 0 else max(0.0, before - count)
        elif before > 0 and self.adds is not False:
            self.units[thing] = before + count
            self.pending = {thing: count}
        else:
            self.units[thing] = 0.0 if count <= 0 else count
        if self.units.get(thing, 0) <= 0:
            self.units.pop(thing, None)
        if not self.settle_view(obs, whole=False):
            if (count <= 0 or not self.is_put(call_args)) and self.shows_empty(obs, thing):
                self.units.pop(thing, None)
                self.pending.pop(thing, None)
        return self.summary()

    def shows_empty(self, obs: Any, thing: str) -> bool:
        if not isinstance(obs, dict) or thing not in self.units:
            return False
        lists = [val for val in obs.values() if isinstance(val, list)]
        if not lists or any(lists):
            return False
        figures = [val for val in obs.values() if isinstance(val, (int, float)) and not isinstance(val, bool)]
        if not figures or any(val != 0 for val in figures):
            return False
        text = as_text(obs)
        return not any(other in text for other in self.units if other != thing)

    def view_units(self, obs: Any) -> dict[str, float] | None:
        found: dict[str, float] = {}
        known = set(self.units) | set(self.pending)

        def read_row(d: dict[str, Any]) -> None:
            idents = [val for val in d.values()
                      if isinstance(val, str) and (val in known or re.fullmatch(ID_LIKE, val)) and " " not in val]
            figures = [(key, val) for key, val in d.items()
                       if isinstance(val, int) and not isinstance(val, bool) and 0 < val < 1000 and str(key) != "price"]
            if len(figures) > 1:
                counted = [(key, val) for key, val in figures
                           if re.search(r"qty|quant|count|units|number|total|held|ordered", str(key).lower())
                           and not re.search(r"price|cost|due|sum|fee|charge", str(key).lower())]
                if len(counted) == 1:
                    figures = counted
            if len(figures) != 1 or not idents:
                return
            ident = next((i for i in idents if i in known), idents[0])
            found[ident] = found.get(ident, 0) + figures[0][1]
        walk_dicts(obs, read_row)
        if not found:
            return None
        if not set(found) & (known or set(found)):
            return None
        return found

    def learn(self, seen: dict[str, float]) -> None:
        for thing, if_set in self.pending.items():
            if thing not in seen:
                continue
            shown = seen[thing]
            if abs(shown - if_set) < 1e-9 and abs(self.units.get(thing, 0) - if_set) > 1e-9:
                self.adds = False
            elif shown > if_set:
                self.adds = True
        self.pending = {}
        self.units = {thing: qty for thing, qty in seen.items() if qty > 0}

    def viewed(self, obs: Any, tick: int) -> None:
        self.last_view = obs
        self.viewed_at = tick
        if self.settle_view(obs):
            return
        if isinstance(obs, (dict, list)) and not self.ids_in(obs):
            self.units = {}

    def ids_in(self, obs: Any) -> bool:
        text = as_text(obs)
        return any(thing in text for thing in self.units)

    @staticmethod
    def unit_text(qty: float) -> Any:
        return int(qty) if float(qty).is_integer() else qty

    def summary(self) -> str:
        if not self.units:
            return "[cart as counted] empty."
        held = ", ".join(f"{thing} x{self.unit_text(qty)}" for thing, qty in self.units.items())
        return f"[cart as counted] {held}."

    def faults(self, need_n: int, counts: list[int] | None = None) -> list[str]:
        if not self.units:
            return []
        if counts and len(counts) > 1:
            return self.basket_faults(counts)
        found = []
        if len(self.units) > 1:
            listing = ", ".join(f"{i} x{int(qty)}" for i, qty in self.units.items())
            found.append("the cart holds more than one line (" + listing + "); take out every line but the chosen one")
            return found
        for thing, qty in self.units.items():
            if abs(qty - need_n) <= 1e-9:
                continue
            change = "take out" if qty > need_n else "put in"
            found.append(f"the line {thing} holds {int(qty)} units but the request needs {need_n}; {change} {int(abs(qty - need_n))} with the cart tool")
        return found

    def basket_faults(self, counts: list[int]) -> list[str]:
        held = sorted(int(q) for q in self.units.values())
        if len(self.units) != len(counts):
            listing = ", ".join(f"{i} x{int(q)}" for i, q in self.units.items())
            return [f"the cart holds {len(self.units)} line(s) ({listing}) but "
                    f"the request asks for {len(counts)} products, one line each; every product needs its line and nothing else"]
        if held != sorted(counts):
            return [f"the lines hold {held} units but the products need {sorted(counts)} (one line per product at its quantity)"]
        return []


class Judge:
    def __init__(self, run):
        self.run = run
        self.audit_fits = None
        self.audit_ids = None
        self.audit_reads = None
        self.audits = 0
        self.ch_rows_memo = None
        self.ch_view_memo = {}
        self.last_reads = None
        self.named = set()
        self.named_list = []
        self.named_pages = set()
        self.rank_better = None

    def field_names(self) -> str:
        menu, _ = self.run.archive.prop_menu()
        if menu:
            return ("\n\nFIELD NAMES SEEN ON THIS STORE'S LISTINGS (names that vary only in separators count as one; each "
                    "number tells on how many read listings it appears):\n" + menu)
        return ""

    def _not_met(self, thing: str, why: str) -> str:
        return f"An independent check says {thing} does NOT meet every requirement: {why}. Choose another listing."

    def _pointed_records(self, refs: list[str], thing: str) -> str:
        archive = self.run.archive
        words = set()
        for ref in refs:
            words.update(re.findall(r"[a-z]{4,}", ref.lower().split(":", 1)[-1]))
        hits = []
        for ident in sorted(archive.read):
            if ident == thing:
                continue
            blob = " ".join(archive.about(ident, 1)).lower()
            if any(w in blob for w in words):
                hits.append(ident)
            if len(hits) == 3:
                break
        if not hits:
            return ""
        return ("\n\nRECORDS OF READ LISTINGS THAT THE REQUEST POINTS TO (when a requirement refers elsewhere, its value comes "
                "from the named field of the listing meant):\n" + "\n".join(line for ident in hits for line in archive.about(ident, 1)))

    def _own_notes_text(self, own) -> str:
        pieces = []
        for req, prop, sent in own:
            pieces.append(f"a line in its notes might redirect {prop} elsewhere: \"{sent[:240]}\". Should it do that (asking the reader to "
                          f"take {prop} from, refer to, go by or look to some other listing), grading uses {prop} from "
                          f"there, and this item's own {prop} field no longer decides '{req}': it holds only if the "
                          "other listing's record shows it. Mentioning a listing merely as a version sold elsewhere, or as what "
                          "this item is not, changes nothing.")
        return "\n\nNOTE (from the code's reading of this item's notes): " + " ".join(pieces)

    def _ruling_key(self, stamp: str, refs, tracked, second: str, own) -> str:
        parts = [stamp]
        if refs or any(r.get("status") != "resolved" for r in tracked):
            parts.append(str(len(self.run.archive.read)))
        if second:
            parts.append("second")
        if own:
            parts.append(f"own{len(own)}.{self.run.refdesk.own_stamp(stamp.split('|')[0])}")
        return "|".join(parts)

    def _rule_out(self, memo: str, thing: str, verdict: str, entry: dict[str, Any]) -> str:
        self.run.rulings[memo] = verdict
        self.run.log.append({"kind": "verify", "item": thing, "result": entry})
        return verdict

    def _check_prompt(self, thing, aim_text, second, own, refs, tracked, history, evidence) -> str:
        run = self.run
        needs = run.needs
        sections = [CHECK_ITEM, "\n\nTHE SHOPPER'S REQUEST:\n", run.request, needs.changes(),
                    "\n\nREQUIREMENTS NOW (stated, revealed and changed):\n", needs.text_body()]
        if needs.items():
            held = run.trolley.held(thing)
            cart_note = f" It is in the cart with {int(run.trolley.held(thing))} units." if held > 0 else ""
            sections.append("\n\nThis item is one of several requested products, and its category shows which one: test it solely "
                            "against that product's own requirements and quantity, ignoring what the other products require."
                            + cart_note)
        sections.append(run.refdesk.ref_note(thing) or "")
        refs = run.refdesk.ref_untracked(refs, tracked)
        if refs:
            sections.append(self._pointed_records(refs, thing))
        sections.append(second)
        if own:
            sections.append(self._own_notes_text(own))
        sections.append(f"\n\nTHE ITEM TO {aim_text}: {thing}\nRECORDS OF THIS ITEM SEEN IN THE STORE (oldest first):\n"
                        + "\n".join(history))
        if run.trolley.last_view is not None:
            sections.append("\n\nLATEST CART VIEW:\n" + as_text(run.trolley.last_view)[:3000])
        sections.append(self.field_names())
        if evidence["lines"]:
            sections.append("\n\nFIELD EVIDENCE (derived by code from the records of this item's category: the way the shop "
                            "spells every required value, and what this item displays):\n" + "\n".join(evidence["lines"]))
        labels = run.archive.label_lines()
        if labels:
            sections.append("\n\nLABELS RECURRING ON THIS STORE'S LISTINGS (counted; the category label of the item has to fit "
                            "the whole wording of the request, never a nearby label):\n" + labels)
        return "".join(sections)

    def check_item(self, thing: str, aim_text: str, second: str = "") -> str:
        run = self.run
        if run.wire.spare() < CLOSING + 20:
            return ""
        stamp = f"{thing}|{run.needs.version}"
        if stamp in run.shelf.near_ok:
            return ""
        listed = run.needs.reading.get("references")
        refs = [str(r) for r in listed or [] if r] if isinstance(listed, list) else []
        tracked = list(run.refs or [])
        own = run.refdesk.own_redirects(thing)
        run.refdesk.chase_own(thing, own)
        memo = self._ruling_key(stamp, refs, tracked, second, own)
        if memo in run.rulings:
            return run.rulings[memo]
        pointer = run.refdesk.ref_pointer_note(thing)
        if pointer:
            return self._rule_out(memo, thing, self._not_met(thing, pointer),
                                  {"item": thing, "fits": False, "problems": ["reference chain"]})
        borrowed = run.refdesk.own_clash(thing)
        if borrowed:
            return self._rule_out(memo, thing, self._not_met(thing, borrowed),
                                  {"item": thing, "fits": False, "problems": [borrowed], "by": "own redirect"})
        history = run.archive.about(thing)
        if not history:
            return ""
        evidence = self.field_view(thing)
        if evidence["fails"]:
            shown = evidence["fails"][:3]
            return self._rule_out(memo, thing, self._not_met(thing, "; ".join(shown)),
                                  {"item": thing, "fits": False, "problems": shown, "by": "field evidence"})
        prompt = self._check_prompt(thing, aim_text, second, own, refs, tracked, history, evidence)
        answer = run.wire.consult(prompt, 3000, CHECKER)
        run.log.append({"kind": "verify", "item": thing, "result": answer})
        if answer and answer.get("fits") is False:
            problems = [str(p) for p in answer.get("problems") or []][:5]
            verdict = (f"An independent check says {thing} does NOT meet every requirement: " + "; ".join(problems)
                       + ". Unless you are certain the check is wrong, choose another fully verified fit.")
            run.rulings[memo] = verdict
            return verdict
        run.rulings[memo] = ""
        if answer:
            run.passed.add(stamp)
            run.pass_price[thing] = run.archive.price(thing)
        return ""

    def ch_rows(self) -> dict[str, dict[str, Any]]:
        count = len(self.run.archive.items)
        cached = self.ch_rows_memo
        if cached and cached[0] == count:
            return cached[1]
        rows = self.run.closer.code_rows()
        self.ch_rows_memo = (len(self.run.archive.items), rows)
        return rows

    def store_fields(self, label: Any = "") -> dict[str, list[str]]:
        run = self.run
        try:
            rows = self.ch_rows()
            homes = run.shelf.home_shelves()
            if run.needs.items():
                pos = run.talk.product_of(str(label))
                wanted = set(homes[pos]) if pos is not None and pos < len(homes) else set()
            else:
                mine = run.shelf.own_shelf()
                if mine:
                    wanted = {mine}
                else:
                    wanted = set(homes[0]) if homes else set()
            found: dict[str, list[str]] = {}
            for row in rows.values():
                if not row["fields"]:
                    continue
                if wanted and row["shelf"] not in wanted:
                    continue
                for name, val in row["fields"].items():
                    if not junk_key(name) and str(val).strip():
                        found.setdefault(str(name), []).append(str(val))
            return found
        except Exception:
            return {}

    def ch_redirected(self, thing: str, prop: str) -> bool:
        try:
            for got in self.run.refdesk.own_redirects(thing) or []:
                heads = (got if isinstance(got, (list, tuple)) else [got])[:2]
                if any(same_prop(str(h).partition(":")[0], prop) for h in heads):
                    return True
            return False
        except Exception:
            return True

    def field_view(self, thing: str) -> dict[str, list[str]]:
        run = self.run
        needs = run.needs
        key = (thing, len(run.archive.items), needs.version, "|".join(needs.props_left()))
        memo = self.ch_view_memo
        if key not in memo:
            try:
                memo[key] = self.field_facts(thing)
            except Exception as err:
                run.log.append({"kind": "field_facts_error", "error": f"{type(err).__name__}: {err}"[:300]})
                memo[key] = {"lines": [], "rules": [], "fails": []}
        confirmed = (run.shelf.near_said or {}).get(f"{thing}|{needs.version}")
        if not confirmed:
            return memo[key]
        try:
            return self._confirmed_view(memo[key], confirmed)
        except Exception:
            return memo[key]

    @staticmethod
    def _confirmed_view(view: dict[str, list[str]], confirmed) -> dict[str, list[str]]:
        names = [p for p, _, _ in confirmed]

        def matches(field: str, prop: str) -> bool:
            return same_prop(field, prop) or strict_prop(field, prop)
        lines = []
        for line in view["lines"]:
            head = re.match(r"- (.+?) = '", line)
            hit = None
            if head:
                hit = next(((p, v, s) for p, v, s in confirmed if matches(head.group(1), p)), None)
            if hit:
                line += (f" A meaning check confirmed its '{hit[2][:80]}' as '{hit[1][:60]}' (the same value in other words): it "
                         "meets it.")
            lines.append(line)
        fails = []
        for fail in view["fails"]:
            field = fail.split(":", 1)[0].strip()
            if not any(matches(field, p) for p in names):
                fails.append(fail)
        return {"lines": lines, "rules": list(view["rules"]), "fails": fails}

    @staticmethod
    def _clip(text: Any, n: int = 70) -> str:
        text = " ".join(str(text).split())
        if len(text) <= n:
            return text
        return text[: n - 3] + "..."

    def field_facts(self, thing: str) -> dict[str, list[str]]:
        result: dict[str, list[str]] = {"lines": [], "rules": [], "fails": []}
        rows = self.ch_rows()
        mine = rows.get(thing)
        if not mine or not mine["fields"]:
            return result
        several = bool(self.run.needs.items())
        shelf = mine["shelf"]
        if several and not shelf:
            return result
        slot = None
        if several:
            slot = next((n for n, home in enumerate(self.run.shelf.home_shelves()) if shelf in home), None)
        peers = {i: r for i, r in rows.items() if r["fields"] and (not shelf or r["shelf"] == shelf)}
        if len(peers) < 3:
            return result
        carry: dict[str, int] = {}
        table: dict[str, list[tuple[str, str, list[str]]]] = {}
        for ident, row in peers.items():
            for name, val in row["fields"].items():
                if junk_key(name) or not str(val).strip():
                    continue
                carry[name] = carry.get(name, 0) + 1
                pieces = ch_parts(val)
                if len(pieces) > 1:
                    pieces += [str(val).strip().lower()]
                table.setdefault(name, []).extend((ident, piece, ch_words(piece)) for piece in pieces)
        context = (rows, mine, shelf, several, slot, peers, carry, table)
        for req in self.run.needs.props_left()[:12]:
            self._fact_for(thing, req, context, result)
        return result

    def _bound_fact(self, prop, value, head, context, result) -> None:
        rows, mine, shelf, several, slot, peers, carry, table = context
        clip = self._clip
        own_slot = self.run.talk.product_of(head) if several else None
        kin = {w for w in set(prop_words(prop)) if len(w) >= 3 and w not in LINK_WORDS}
        named = [name for name in carry if carry[name] >= 2 and strict_prop(name, prop)]
        if not (prop and kin and named):
            return
        if several and own_slot is not None and own_slot != slot:
            return
        for name, v in mine["fields"].items():
            if str(v).strip() and (strict_prop(name, prop) or set(prop_words(name)) & kin):
                return
        listed = " / ".join(f"'{x}'" for x in named[:3])
        holding = sum(1 for r in peers.values() if any(str(r['fields'].get(x, '')).strip() for x in named))
        result["lines"].append(f"- {prop} = '{clip(value)}': the shop records it under {listed}, found on {holding} of "
                               f"{len(peers)} read listings in this category. That field is absent here, hence the limit is unproven "
                               "and the item fails it: it does NOT meet it (another field's figure does not stand in).")
        result["fails"].append(f"{prop}: it has no {named[0]} field, which this store's listings of the category carry, so "
                               f"'{clip(value)}' is not shown met")

    @staticmethod
    def _parse_requirement(req):
        head, _, raw = str(req).partition(":")
        value = " ".join(re.sub(r"\((?:stated|revealed[^)]*)\)", " ", raw).split())
        prop = re.split(r"\s*\(", head)[0].strip()
        if not prop or not value or ref_text(value):
            return None
        if re.search(r"as shown|same as|the page|sold by|redirect|refer", value.lower()):
            return None
        return head, prop, value

    def _field_names_for(self, fact, context):
        carry, table = context[6], context[7]
        prop, alts = fact["prop"], fact["alts"]

        def heaviest_first(names):
            return sorted(names, key=lambda x: -carry[x])
        named = heaviest_first(n for n in carry if strict_prop(n, prop))
        fact["by_name"] = bool(named)
        if named and not fact["positive"]:
            flat = " ".join(re.findall(r"[a-z0-9]+", fact["value"].lower()))
            if any(" ".join(words) == flat for n in named for _, _, words in table.get(n, [])):
                fact["positive"] = True
                LITERAL_VALUES.add(flat)
        if named:
            return named
        if not fact["positive"] or (context[3] and fact["slot"] is None):
            return None
        writing = [n for n in carry if carry[n] >= 2 and any(ch_match(alts, words) for _, _, words in table.get(n, []))]
        if not writing:
            return None
        top = max(writing, key=lambda x: carry[x])
        return heaviest_first(n for n in carry if strict_prop(n, top))

    def _rule_text(self, fact) -> str:
        clip = self._clip
        lead, value = fact["lead"], fact["value"]
        if not fact["positive"]:
            return lead
        seen_vals = " | ".join(fact["samples"])
        if fact["writers"]:
            extra = "" if fact["by_name"] else (" Since no field carries the property's name, a title that spells out the value is "
                                                "enough as well, unless a field of its own that shares the property's name says otherwise.")
            return (lead + " Only a value spelling those very words passes here (containment and one list element "
                    "count; a synonym or related word does not)." + extra)
        if fact["naming"]:
            return (lead + " Values met so far: " + seen_vals + f"; '{clip(value)}' turns up in the title or text of "
                    f"listings in this category ({', '.join(fact['naming'])}), meaning the shop uses that very word: other "
                    "wording counts only once such a listing's record shows this field writes it otherwise.")
        if fact["on"] < CH_SYNONYM_MIN:
            return (lead + " Values met so far: " + seen_vals + f"; not one spells it out, but this field appears "
                    f"on too few read listings to know if the shop would write '{clip(value)}' literally: a value of the same "
                    "meaning is not shown to meet it yet.")
        return lead + " Values met so far: " + seen_vals + "; not one spells it out, so a value of the same meaning may meet it."

    def _fact_for(self, thing, req, context, result) -> None:
        rows, mine, shelf, several, slot, peers, carry, table = context
        clip = self._clip
        parsed = self._parse_requirement(req)
        if parsed is None:
            return
        head, prop, value = parsed
        if re.search(r"\d", value) and re.search(CH_BOUND, value.lower()):
            self._bound_fact(prop, value, head, context, result)
            return
        if re.fullmatch(r"\s*\d+(?:\.\d+)?\s+[a-z]+\s*", value.lower()):
            return
        fact = {"prop": prop, "value": value, "slot": self.run.talk.product_of(head) if several else None}
        if several and fact["slot"] is not None and fact["slot"] != slot:
            return
        fact["positive"] = positive_value(value, head)
        if not fact["positive"] and not re.search(EXCLUDE_WORDS, value.lower()):
            return
        fact["alts"] = alts = ch_alts(value)
        if not alts:
            return
        named = self._field_names_for(fact, context)
        if named is None:
            return
        fact["named"] = named
        fact["on"] = sum(1 for r in peers.values() if any(str(r["fields"].get(n, "")).strip() for n in named))
        entries = [(ident, words) for n in named for ident, _, words in table.get(n, [])]
        others = [(ident, words) for ident, words in entries if ident != thing]
        fact["writers"] = sorted({ident for ident, words in others if ch_match(alts, words)})
        fact["alone"] = {" ".join(words) for _, words in others}
        fact["samples"] = list(dict.fromkeys(clip(r["fields"][n], 40) for r in peers.values() for n in named
                                             if str(r["fields"].get(n, "")).strip()))[:6]
        tail = f"{len(fact['writers'])} other listing(s) write '{clip(value)}' itself in it."
        if not fact["positive"]:
            bare = " ".join(re.sub(EXCLUDE_WORDS, " ", value.lower()).split())
            bare_alts = ch_alts(bare)
            held = {ident for ident, words in others if bare_alts and ch_match(bare_alts, words)}
            if bare_alts:
                tail = (f"{len(held)} other listing(s) put '{clip(bare)}' there, which is the excluded value (a listing that writes "
                        "it does not meet it).")
        lead = f"- {prop} = '{clip(value)}': the store's field for it is " + " / ".join(f"'{x}'" for x in named[:3])
        if not fact["by_name"]:
            lead += f" (nothing is called '{prop}'; the values of this one carry the value)"
        fact["lead"] = lead + f", on {fact['on']} of {len(peers)} read listings of this category; " + tail
        kinds = kind_words(value)
        fact["naming"] = []
        if kinds and not fact["writers"]:
            fact["naming"] = [i for i, r in rows.items() if i != thing and (not shelf or r["shelf"] == shelf)
                              and kinds <= r["words"]][:3]
        result["rules"].append(self._rule_text(fact))
        self._fact_line(thing, prop, value, fact["positive"], fact["by_name"], named, fact["writers"], fact["alone"],
                        fact["samples"], fact["naming"], fact["on"], fact["lead"], context, result)

    def _qualified_fields(self, fields, named, prop, value):
        base_words = prop_words(prop)
        if not base_words:
            return []
        found = []
        for name, v in fields.items():
            if name in named or not str(v).strip() or junk_key(name):
                continue
            words = prop_words(name)
            if len(words) == len(base_words) + 1 and words[1:] == base_words:
                found.append((name, str(v)))
        return found

    def _missing_field_line(self, lead, prop, value, fields, named, qual_note, title_says) -> str:
        clip = self._clip
        if qual_note:
            return lead + " This listing has no such field." + qual_note
        if title_says:
            return (lead + f" No such field here, and nothing carries the property's name, yet its title "
                    f"writes '{clip(value)}': it meets it.")
        short = {w for w in prop_words(prop) if len(w) >= 3}
        related = [(name, v) for name, v in fields.items() if name not in named and str(v).strip() and not junk_key(name)
                   and (set(prop_words(name)) & short or ch_writes(value, v))]
        related = sorted(related, key=lambda pair: not ch_writes(value, pair[1]))[:4]
        listed = ""
        if related:
            listed = ": candidate fields for the property: " + "; ".join(f"{name}: '{clip(v, 60)}'" for name, v in related)
        return (lead + " No such field here" + listed + ". A different label for this SAME property is acceptable; any "
                "field of another property (a height for a type, a frame material for the material) does not stand in.")

    def _fact_line(self, thing, prop, value, positive, by_name, named, writers, alone, samples, naming, on, lead, context,
                   result) -> None:
        mine = context[1]
        clip = self._clip
        fields = mine["fields"]
        own = [(name, str(fields[name])) for name in named if str(fields.get(name, "")).strip()]
        shown = "; ".join(f"{name}: '{clip(v, 90)}'" for name, v in own)
        qualified = self._qualified_fields(fields, named, prop, value)
        qual_writes = [pair for pair in qualified if ch_writes(value, pair[1])]
        qual_near = bool(qual_writes) or any(ch_touches(value, v) for _, v in qualified)
        qual_note = ""
        if qual_writes:
            first_name, first_value = qual_writes[0]
            qual_note = (f" Its {first_name} '{clip(first_value, 60)}' carries it (a fuller key for the very same "
                         "property: one extra word in front of the property's name still names that property, only under a longer "
                         "name), so it meets it.")
        lines = result["lines"]
        if self.ch_redirected(thing, prop):
            lines.append(lead + " Notes on this listing might point to a different listing for this property, so its own "
                         "field may not decide.")
            return
        if not positive:
            lines.append(lead + " Here: " + (shown or "no such field") + " (this is an exclusion, so the listing must not "
                         "carry the excluded value).")
            return
        kin_words = {w for w in prop_words(prop) if len(w) >= 3 and w not in LINK_WORDS}
        kin = [name for name, v in fields.items() if str(v).strip() and not junk_key(name) and set(prop_words(name)) & kin_words
               and not ch_writes(value, v)]
        title_says = not by_name and not kin and ch_writes(value, mine.get("title") or "")
        if not own:
            lines.append(self._missing_field_line(lead, prop, value, fields, named, qual_note, title_says))
            return
        alts = ch_alts(value)
        pieces = [p for _, v in own for p in ch_parts(v)]
        pieces += [v.strip().lower() for _, v in own if len(ch_parts(v)) > 1]

        def knocked(part: str) -> list[str]:
            sides = ch_sides(part)
            if not sides or not any(ch_match(alts, ch_words(side)) for side in sides):
                return []
            return [side for side in sides if not ch_match(alts, ch_words(side)) and " ".join(ch_words(side)) in alone]
        matching = [p for p in pieces if ch_match(alts, ch_words(p))]
        joins = [(p, knocked(p)) for p in matching]
        clean = [p for p, extra in joins if not extra]
        joined = [(p, extra) for p, extra in joins if extra]
        line, fail = self._own_field_verdict(lead, shown, prop, value, named, own, writers, samples, naming, on, by_name,
                                             clean, joined, qual_note, qual_near, title_says)
        lines.append(line)
        if fail:
            result["fails"].append(fail)

    def _own_field_verdict(self, lead, shown, prop, value, named, own, writers, samples, naming, on, by_name, clean, joined,
                           qual_note, qual_near, title_says):
        clip = self._clip
        seen_vals = " | ".join(samples)
        if clean:
            return lead + f" Here: {shown} - it writes '{clip(value)}': it meets it.", ""
        if qual_note:
            return lead + f" Here: {shown}." + qual_note, ""
        if title_says:
            return (lead + f" Here: {shown} - since nothing carries the property's name, the title decides, and its title writes "
                    f"'{clip(value)}': it meets it."), ""
        if joined:
            part, others = joined[0]
            line = (lead + f" Here: {shown} - '{clip(part)}' combines '{clip(value)}' with '{others[0]}', a value that "
                    "other listings carry alone in this field: it does NOT meet it.")
            fail = ""
            if by_name and not qual_near:
                fail = (f"{prop}: its {named[0]} is '{clip(part)}', which joins {clip(value)} with {others[0]} (a "
                        f"value this store also writes alone there), so it is not {clip(value)}")
            return line, fail
        if writers:
            line = (lead + f" Here: {shown} - '{clip(value)}' is missing, though fellow listings in "
                    "this category write it in this field: so it fails (other wording, a synonym or related word does not "
                    "count where the store writes the value itself).")
            fail = ""
            if by_name and not qual_near and not any(ch_touches(value, v) for _, v in own):
                fail = (f"{prop}: its {named[0]} is '{clip(own[0][1], 60)}', while this store writes "
                        f"'{clip(value)}' itself in that field on other listings")
            return line, fail
        if naming:
            return (lead + f" Here: {shown} - '{clip(value)}' is not written. So far no read listing puts it in "
                    "this field (values: " + seen_vals + f"), yet titles or texts of this category's listings use "
                    f"'{clip(value)}' ({', '.join(naming)}), so the shop has its own word: "
                    "so this listing's other value is not shown to meet it."), ""
        if on < CH_SYNONYM_MIN:
            return (lead + f" Here: {shown}. Among read listings of this category none puts '{clip(value)}' in "
                    "this field (values: " + seen_vals + f"); with only {on} read listing(s) having it, nobody can "
                    f"tell yet if '{clip(value)}' would be written literally, so a value of the same meaning is "
                    "not shown to meet it yet."), ""
        return (lead + f" Here: {shown}. Among read listings of this category none puts '{clip(value)}' in "
                "this field (values: " + seen_vals + "), so a value of the same meaning may meet it."), ""

    def rank_check(self, thing: str) -> str:
        run = self.run
        needs = run.needs
        priority = needs.priority()
        stamp = f"{thing}|{needs.version}|{run.refusal_n}"
        if priority == "none" or stamp in run.rank_checked:
            return ""
        if run.wire.spare() < CLOSING + 30 or run.moves_left() < 4:
            return ""
        run.rank_checked.add(stamp)
        detail = needs.reading.get("priority_detail") or ""
        if priority == "cheapest":
            what = "cheapest (lowest price)"
        else:
            what = f"best ({detail})" if detail else "best"
        prompt = RANK_CHECK.replace("PRIORITY", what) + "\n\nTHE SHOPPER'S REQUEST:\n" + run.request + needs.changes()
        prompt += "\n\nREQUIREMENTS NOW:\n" + needs.text_body()
        prompt += run.refdesk.ref_note()
        if needs.items():
            prompt += ("\n\nSeveral products are requested, so the chosen item only competes with listings of the SAME product "
                       "(category); a listing for some other product can never beat it.")
        cheapest = priority == "cheapest"
        cards = run.archive.cards(20000, thing if cheapest else "", run.refdesk.ref_foreign(), {thing})
        prompt += (f"\n\nTHE CHOSEN ITEM: {thing}\n\nLISTINGS SEEN (a card each; DETAILS = full record opened, ROW = known only "
                   "from search rows, fields unknown: a ROW in the requested category (no matter what object the title names) "
                   "that beats the chosen item and is not ruled out by title, price or stock MAY fit and is named"
                   + ("; only listings priced below the chosen one are shown" if cheapest else "") + "):\n" + "\n".join(cards))
        prompt += self.field_names()
        rules = self.field_view(thing)["rules"]
        if rules:
            prompt += ("\n\nFIELD EVIDENCE (derived by code from the records of the chosen item's category: the way the shop "
                       "spells every required value):\n" + "\n".join(rules))
        answer = run.wire.consult(prompt, 2500, CHECKER)
        run.log.append({"kind": "rank", "result": answer})
        gone = run.shelf.dead_lines()
        better = []
        for x in (answer or {}).get("better") or []:
            if not isinstance(x, (str, int)):
                continue
            x = str(x)
            if x != thing and x not in run.refused_lines and x not in gone:
                better.append(x)
        better = [x for x in better[:4] if not self.field_view(x)["fails"]]
        self.rank_better = list(better)
        self.named.update(better)
        if not better:
            return ""
        reason = f" ({answer.get('reason')})" if answer.get("reason") else ""
        return (f"A ranker believes these listings could also fit and beat {thing} on the {what}: " + ", ".join(better) + reason
                + ". Check them (details, stock, price) before ordering; if each fails, order the current pick.")

    def _unread_hits(self, cap, qty, shelves, hit_shelf) -> list[str]:
        run = self.run
        scores = run.cover.hit_score
        keep = []
        for ident in sorted(run.cover.all_hits, key=lambda i: -scores.get(i, 0)):
            if ident in run.archive.read:
                continue
            label = hit_shelf.get(ident)
            if shelves and label and label not in shelves:
                continue
            if is_number(cap) and is_number(run.archive.price(ident)) and run.archive.price(ident) * qty > cap:
                continue
            keep.append(ident)
        return keep

    def _audit_prompt(self, cards, fresh) -> str:
        run = self.run
        sections = [BEFORE_DECLINE, "\n\nTHE SHOPPER'S REQUEST:\n", run.request, run.needs.changes(),
                    "\n\nREQUIREMENTS NOW:\n", run.needs.text_body(), run.refdesk.ref_note(),
                    "\n\nLISTINGS SEEN (a card each; DETAILS = full record opened, ROW = known only from search rows):\n",
                    "\n".join(cards)]
        if fresh:
            sections.append("\n\nOPENED SINCE THE PREVIOUS CHECK (the DETAILS cards above for these are fresh: assess them "
                            "before the rest, field by field, using the reading rules): " + ", ".join(fresh))
        sections.append(self.field_names())
        return "".join(sections)

    def _audit_followups(self, unopened: list[str]):
        run = self.run
        archive = run.archive
        shelves, cap, qty, hit_shelf = run.watch.shelf_labels(), run.needs.spend_cap, run.needs.quantity(), run.watch.hit_shelf
        hits = self._unread_hits(cap, qty, shelves, hit_shelf)
        try:
            slack = [i for i in run.shelf.slack_rows() if i not in archive.read]
        except Exception:
            slack = []
        unopened = list(dict.fromkeys(slack + hits[:4] + [i for i in unopened if i not in hits]))[:6]
        pages = run.watch.open_pages(2) if run.moves_left() > 5 else []
        if pages:
            unopened = unopened[:4]
            self.named_pages.update(json.dumps(p, sort_keys=True, default=str) for p in pages)
        try:
            combos = run.walk.combined_queries() if run.moves_left() > 5 else []
        except Exception as err:
            run.log.append({"kind": "combo_error", "error": f"{type(err).__name__}: {err}"[:300]})
            combos = []
        if combos:
            self.named_pages.update(json.dumps(p, sort_keys=True, default=str) for p in combos)
        return unopened, pages, combos

    def _audit_note(self, answer, fits, unopened, pages, combos) -> str:
        search = self.run.role('search')

        def calls(found) -> str:
            return "; ".join(f"{search} {json.dumps(p, ensure_ascii=False)}" for p in found)
        parts = []
        if fits:
            reason = f" ({answer.get('reason')})" if answer.get("reason") else ""
            parts.append("A separate review suggests these listings could satisfy every requirement: " + ", ".join(fits) + reason + ".")
        if pages:
            parts.append("Some searches with required values returned more listings than have been read; matches could be waiting on "
                         "pages not yet read: " + calls(pages) + ". Go through those pages first (each step brings a full page of "
                         "rows) and open the in-budget rows of the requested kind they show.")
        if combos:
            parts.append("Searches joining the required values have not been tried yet (where the store needs every word, each returns "
                         "just the listings holding all of them): " + calls(combos) + ". Try them (they remain allowed on the search "
                         "tool) and open the in-budget rows of the requested kind they show.")
        if unopened:
            parts.append("Listings of the requested kind known only from search rows, never opened, so nobody has checked their "
                         "fields: " + ", ".join(unopened) + ". Open their details first (several in one reply).")
        return " ".join(parts) + " Verify them (details, stock, price) and buy the right one; decline only if each one truly fails."

    def review_decline(self, early: bool = False) -> str:
        run = self.run
        archive = run.archive
        limit = 10 ** 9 if self.audit_reads is None else self.audit_reads
        if run.audit_done and not (len(archive.read) > limit and self.audits < 2):
            return ""
        if run.wire.spare() < CLOSING + 30:
            return ""
        run.audit_done = True
        if not early:
            self.audits += 1
        before = self.audit_ids
        skip = () if before is None else set(archive.read) - set(before or ())
        cards = archive.cards(16000, "", run.refdesk.ref_foreign(), skip)
        self.audit_reads = len(archive.read)
        if not cards:
            return ""
        fresh = [] if before is None else sorted(archive.read - before)
        self.audit_ids = set(archive.read)
        answer = run.wire.consult(self._audit_prompt(cards, fresh), 6000, AUDITOR)
        run.log.append({"kind": "audit", "result": answer})
        reply = answer or {}
        fits = [str(x) for x in reply.get("possible_fits") or [] if isinstance(x, (str, int))][:5]
        fits = [x for x in fits if not self.field_view(x)["fails"]]
        self.audit_fits = fits
        suggested = [str(x) for x in reply.get("unopened") or [] if isinstance(x, (str, int))
                     and str(x) not in archive.read and str(x) in archive.ids][:6]
        unopened, pages, combos = self._audit_followups(suggested)
        self.named.update(fits + unopened)
        self.named_list += [i for i in fits + unopened if i not in self.named_list]
        if not (fits or unopened or pages or combos):
            return ""
        return self._audit_note(answer, fits, unopened, pages, combos)

    def named_unread(self) -> list[str]:
        run = self.run
        archive = run.archive
        cap, qty = run.needs.spend_cap, run.needs.quantity()
        shelves, hit_shelf = run.watch.shelf_labels(), run.watch.hit_shelf
        scores = run.cover.hit_score
        ranked = sorted(run.cover.all_hits, key=lambda i: -scores.get(i, 0))
        picked = []
        for ident in dict.fromkeys(ranked + list(self.named_list)):
            price = archive.price(ident)
            if ident in archive.read or ident in run.refused_lines or ident not in archive.ids:
                continue
            if is_number(cap) and is_number(price) and price * qty > cap:
                continue
            if shelves and hit_shelf.get(ident) and hit_shelf[ident] not in shelves:
                continue
            picked.append(ident)
        return picked

    def details_call(self, ident: str) -> dict[str, Any] | None:
        run = self.run
        tool = run.role("details")
        known = run.archive.ids
        for entry in reversed(run.log):
            if entry.get("kind") != "env":
                continue
            for act in entry.get("actions") or []:
                if act.get("name") != tool:
                    continue
                args = act.get("args")
                if isinstance(args, dict) and any(isinstance(v, str) and v in known for v in args.values()):
                    return {k: (ident if isinstance(v, str) and v in known else v) for k, v in args.items()}
        schema = run.shapes.get(tool or "") or {}
        props = schema.get("properties") if isinstance(schema.get("properties"), dict) else {}
        texts = [k for k, p in props.items() if isinstance(p, dict) and p.get("type") == "string" and not p.get("enum")]
        needed = schema.get("required") or []
        slot = next((k for k in texts if k in needed), None)
        if not slot and len(texts) == 1:
            slot = texts[0]
        return {slot: ident} if slot else None

    def open_reads(self, queue: list[str], say: str, batch: tuple[str, str, int] | None, reader: str | None = None) -> list[tuple[str, str]]:
        run = self.run
        if batch:
            tool, arg, most = batch
            calls = [{arg: queue[at: at + most]} for at in range(0, len(queue), most)]
        else:
            tool = reader or run.role("details")
            calls = []
            if tool:
                for ident in queue:
                    built = run.shelf.lookup_call(ident, tool)
                    if built:
                        calls.append(built)
        tick = run.wire.tick
        proposals = [{"id": f"open_{n}_{tick}", "type": "function", "function": {"name": tool, "arguments": json.dumps(args)}}
                     for n, args in enumerate(calls)]
        if not proposals:
            return []
        run.chat.append({"role": "assistant", "content": say, "tool_calls": proposals})
        start = len(run.archive.items)
        run.covering = True
        try:
            replies = run.dispatch(proposals, scripted=True)
        finally:
            run.covering = False
        self.last_reads = run.archive.items[start:]
        return replies

    def _take_queue(self):
        run = self.run
        reads = run.reads
        queue, reads.open_queue = reads.open_queue, None
        from_put = False
        if not queue and reads.put_queue and not run.wire.done and not run.wire.broke:
            queue, reads.put_queue = reads.put_queue, None
            from_put = True
            tool, reads.put_tool = reads.put_tool, None
            reads.open_recheck, reads.open_purpose, reads.open_tool = True, "cart", tool
            reads.open_say = "Checking stock on the line I just added, since an item may sell out the moment it lands in the cart."
        return queue, from_put

    def _reserve_trim(self, queue: list[str], purpose: str) -> list[str]:
        run = self.run
        try:
            keep = run.intake.order_due() if run.orders_tried == 0 else FIRST_ORDER_BY
            room = run.moves_left() - keep - 1
            listed = run.shelf.batch_tool() and len(queue) >= 2
            if room < (1 if listed else len(queue)) and (run.trolley.units or not run.intake.unverified()):
                cut = [] if (run.shelf.batch_tool() and len(queue) >= 2) else queue[: max(0, room)]
                run.log.append({"kind": "reserve_skip", "purpose": purpose, "ids": queue[:6], "kept": cut})
                return cut
        except Exception as err:
            run.log.append({"kind": "reserve_skip_error", "error": f"{type(err).__name__}: {err}"[:300]})
        return queue

    def open_named(self) -> bool:
        run = self.run
        wire = run.wire
        queue, from_put = self._take_queue()
        if not queue or wire.done or wire.broke:
            return False
        queue = list(dict.fromkeys(str(i) for i in queue))
        reads = run.reads
        recheck, reads.open_recheck = reads.open_recheck, False
        purpose = str(reads.open_purpose or "")
        reads.open_purpose = ""
        floor, reads.open_floor = reads.open_floor, None
        reader, reads.open_tool = reads.open_tool, None
        say, reads.open_say = reads.open_say, ""
        if not say:
            say = "One more look at the cart lines before the order." if recheck else "The checks named some listings; opening them before any decline."
        if purpose in ("menu", "quota", "walk") and run.role("order") and run.role("cart") and not wire.mute:
            queue = self._reserve_trim(queue, purpose)
            if not queue:
                return False
        if purpose == "cart":
            stale = set(run.shelf.stale_lines())
            queue = [i for i in queue if i in stale or (not from_put and i not in run.trolley.units)]
            if not queue:
                return False
        batch = run.shelf.batch_tool() if len(queue) >= 2 else None
        if batch and purpose == "cart" and not run.batch_proven:
            batch = None
        if batch and reader and not ((run.shelf.tool_caps or {}).get(batch[0]) or {}).get("stock"):
            batch = None
        replies = self.open_reads(queue, say, batch, reader)
        if batch and replies and not run.batch_proven:
            if run.shelf.batch_ok(self.last_reads or [], queue):
                run.batch_proven = True
            else:
                run.batch_spec = None
                run.batch_spec_known = True
                run.log.append({"kind": "batch_dropped", "tool": batch[0]})
                most = {"menu": 0, "quota": 2, "walk": 2}.get(purpose, 3)
                if is_number(floor):
                    keep = int(floor)
                else:
                    keep = run.intake.order_due() if purpose in ("menu", "quota") else 3
                count = min(most, len(queue), run.moves_left() - keep - 1)
                if count > 0 and not wire.done and (reader or run.role("details")):
                    run.talk.post_replies(replies)
                    replies = self.open_reads(queue[:count], say, None, reader)
                    if not replies:
                        return True
        if not replies:
            return False
        note = ""
        if wire.spare() > CLOSING + 30 and not recheck:
            note = self.review_decline()
        if note:
            cid, body = replies[-1]
            replies[-1] = (cid, body + "\n[check after opening] " + note)
        run.talk.post_replies(replies)
        return True

    @staticmethod
    def dicts_in(pieces) -> list[dict[str, Any]]:
        found: list[dict[str, Any]] = []
        for piece in pieces:
            walk_dicts(piece, found.append)
        return found

    def last_stock(self, ident: str) -> bool | None:
        watch = self.run.watch
        state = None
        for d in self.dicts_in(self.run.archive.items):
            if watch.row_mark(d) == ident:
                got = watch.stock_state(d)
                state = state if got is None else got
        return state

    def _rank_candidate_ok(self, x: str, version: str, cap) -> bool:
        run = self.run
        if x not in run.archive.read or f"{x}|{version}" in run.passed or x in run.refused_lines:
            return False
        if x in run.shelf.dead_lines() or self.last_stock(x) is False or self.field_view(x)["fails"]:
            return False
        items = run.needs.items()
        pos = run.walk.product_index(x) if items else None
        if pos is not None and pos < len(run.needs.items()):
            qty = run.needs.items()[pos]["quantity"]
        else:
            qty = run.needs.quantity()
        price = run.archive.price(x)
        if not is_number(price):
            return False
        return not (is_number(cap) and is_number(qty) and price * qty > cap)

    def rank_verify(self, thing: str) -> str:
        run = self.run
        try:
            if run.needs.priority() != "cheapest":
                return ""
            version = str(run.needs.version)
            cap = run.needs.spend_cap
            checked = 0
            for x in self.rank_better or []:
                if checked >= 2 or run.wire.spare() < CLOSING + 30:
                    break
                if self._rank_candidate_ok(x, version, cap):
                    checked += 1
                    self.check_item(x, "ORDER")
            return self.cheaper_fit(thing)
        except Exception as err:
            run.log.append({"kind": "rank_verify_error", "error": f"{type(err).__name__}: {err}"[:300]})
            return ""

    def cheaper_fit(self, thing: str) -> str:
        run = self.run
        if run.needs.priority() != "cheapest" or not thing:
            return ""
        ceiling = run.archive.price(thing)
        if ceiling is None:
            return ""
        label = self.shelf_known(thing)
        best_id, best_cost = "", None
        gone = run.shelf.dead_lines()
        version = str(run.needs.version)
        for key in run.passed:
            ident, _, ver = key.rpartition("|")
            if ver != version or ident == thing or ident in run.refused_lines or ident in gone:
                continue
            if run.needs.items() and (not label or self.shelf_known(ident) != label):
                continue
            cost = run.archive.price(ident)
            if cost is None or cost >= ceiling or (best_cost is not None and cost >= best_cost):
                continue
            if self.field_view(ident)["fails"]:
                continue
            best_id, best_cost = ident, cost
        return best_id

    def shelf_known(self, thing: str) -> str:
        watch = self.run.watch
        label = ""
        for d in self.dicts_in(self.run.archive.items):
            for key, val in d.items():
                if isinstance(val, str) and val == thing and id_field(key):
                    label = watch.shelf_of(d) or label
                    break
        if label:
            return label
        return str((self.run.archive.label_of or {}).get(thing) or "").strip().lower()

    def chosen(self) -> str | None:
        units = self.run.trolley.units
        return list(units)[-1] if units else None


class Catalogue:
    def __init__(self, run):
        self.run = run
        self.cart_shown = -1
        self.cart_ver = None
        self.dead = set()
        self.dead_by = {}
        self.ed_memo = {}
        self.first_puts = set()
        self.home_memo = None
        self.joined_seen = {}
        self.near_cart = None
        self.near_done = None
        self.near_ok = set()
        self.near_said = {}
        self.note_keys = {}
        self.pre_picked = None
        self.pre_read_done = None
        self.pre_rounds = 0
        self.read_with = {}
        self.repair_tried = set()
        self.role_guess = None
        self.row_keys = set()
        self.row_text = {}
        self.rows_lag = None
        self.rows_live = None
        self.slack_total = None
        self.stock_by = {}
        self.stock_seen = None
        self.swap_page = None
        self.swap_unread = None
        self.text_keys = {}
        self.tool_caps = {}
        self.tool_keys = {}

    def _labels_seen(self) -> set[str]:
        archive = self.run.archive
        watch = self.run.watch
        labels: set[str] = set()
        for record in Judge.dicts_in(archive.items):
            label = watch.shelf_of(record)
            if label and any(isinstance(v, str) and v in archive.ids and id_field(key) for key, v in record.items()):
                labels.add(label)
        return labels

    @staticmethod
    def _best_labels(kinds: list[set], labels: set[str]) -> list[set[str]]:
        homes = []
        for pos, own in enumerate(kinds):
            shared = set()
            for other, words in enumerate(kinds):
                if other != pos:
                    shared |= words
            winners, top = set(), 0.0
            for label in sorted(labels):
                score = kind_score(own, set(kind_tokens(label)), shared)
                if score > top:
                    winners, top = {label}, score
                elif score > 0 and score == top:
                    winners.add(label)
            homes.append(winners)
        return homes

    def home_shelves(self) -> list[set[str]]:
        run = self.run
        count = len(run.archive.items)
        if self.home_memo and self.home_memo[0] == count:
            return [set(h) for h in self.home_memo[1]]
        try:
            products = run.needs.items() or [{"product": str(run.needs.reading.get("product") or "")}]
            labels = self._labels_seen()
            homes = self._best_labels([set(kind_tokens(row["product"])) for row in products], labels)
        except Exception:
            homes = [set() for _ in (run.needs.items() or [0])]
        self.home_memo = (len(run.archive.items), [set(h) for h in homes])
        return homes

    def own_shelf(self) -> str:
        try:
            if self.run.needs.items():
                return ""
            home = self.home_shelves()[0]
            labels = [s.strip().lower() for s in self.run.archive.shelves_read()]
            if not home or all(s in home for s in labels):
                return ""
            for label in labels:
                if label in home:
                    return label
            return sorted(home)[0]
        except Exception:
            return ""

    @staticmethod
    def _label_in_order(text: str, label: str):
        halves = [p for p in re.split(JOINERS, str(label).lower()) if p.strip()]
        if len(halves) != 2:
            return None
        left, right = name_pattern(halves[0]), name_pattern(halves[1])
        if not left or not right:
            return None
        return re.search(r"\b" + left + r"\s*(?:\band\b|&|\bor\b|/|\+)\s*" + right, text)

    @staticmethod
    def _loose_join(text: str, x: str, y: str, a: str, b: str):
        pat_x, pat_y = name_pattern(x), name_pattern(y)
        if not pat_x or not pat_y:
            return None
        for hit in re.finditer(r"\b(?:" + pat_x + r")(?P<mid>[^.;:!?]{0,40}?)(?:" + pat_y + r")", text):
            gap = hit.group("mid")
            short_gap = len(re.findall(r"[a-z0-9]+", gap)) <= 3
            if not short_gap or not re.search(r"\band\b|&|\bor\b|,|/|\+", gap) or re.search(r"\d|\bx\b", gap):
                continue
            outside = text[:hit.start()] + " | " + text[hit.end():]
            if not all(re.search(r"\b" + name_pattern(z), outside) for z in (a, b)):
                return hit
        return None

    def joined_phrase(self, a: str, b: str, label: str) -> tuple[str, bool] | None:
        text = re.sub(r"\s+", " ", self.run.request.lower().replace("’", "'"))
        whole = self._label_in_order(text, label)
        if whole:
            return whole.group(0).strip(), True
        kinds_a, kinds_b = kind_tokens(a), kind_tokens(b)
        if not kinds_a or not kinds_b or same_kind(kinds_a[-1], kinds_b[-1]):
            return None
        for x, y in ((a, b), (b, a)):
            hit = self._loose_join(text, x, y, a, b)
            if hit:
                return hit.group(0).strip(), False
        return None

    def product_question(self, line: str, rows: list[dict[str, Any]]) -> bool:
        parsed = re.match(r'-\s*(.*?): asked "(.*?)" -> ', str(line), re.S)
        if parsed:
            label, question = parsed.group(1), parsed.group(2)
        else:
            label = question = str(line)
        kinds = [kind_tokens(r["product"]) for r in rows]
        for tag in re.findall(r"\(([^()]*)\)", label):
            words = kind_tokens(tag)
            if words and any(kind_in(w, k) for w in words for k in kinds):
                return True
        text = question.lower().replace("’", "'")
        for row in rows:
            pattern = name_pattern(row["product"])
            if pattern and re.search(r"\b" + name_pattern(row["product"]), text):
                return True
        return False

    @staticmethod
    def joined_clash(a: dict[str, Any], b: dict[str, Any]) -> bool:
        def pairs(row: dict[str, Any]) -> list[tuple[str, str]]:
            found = []
            for req in row.get("requirements") or []:
                head, _, value = str(req).partition(":")
                found.append((re.split(r"\s*\(", head)[0].strip(), " ".join(re.findall(r"[a-z0-9]+", value.lower()))))
            return found
        right = pairs(b)
        for prop_a, val_a in pairs(a):
            for prop_b, val_b in right:
                if same_prop(prop_a, prop_b) and val_a != val_b:
                    return True
        return False

    def _note_joined_labels(self, obs: Any) -> None:
        seen = self.joined_seen
        for d in Judge.dicts_in([obs]):
            if not any(isinstance(v, str) and id_field(k) and re.fullmatch(ID_LIKE, v) for k, v in d.items()):
                continue
            label = ""
            for key, val in d.items():
                if not isinstance(val, str) or not 0 < len(val.strip()) <= 60:
                    continue
                name = str(key).lower()
                if re.search(r"categor|shelf|department|aisle", name) and "id" not in name:
                    label = val.strip()
                    break
            if label and len([p for p in re.split(JOINERS, label.lower()) if p.strip()]) == 2:
                seen.setdefault(label.lower(), label)

    @staticmethod
    def _sides_split(ti, tj, sides) -> bool:
        only_i = [w for w in ti if not kind_in(w, tj)]
        only_j = [w for w in tj if not kind_in(w, ti)]
        for x, y in ((sides[0], sides[1]), (sides[1], sides[0])):
            if not all(kind_in(w, x) and not kind_in(w, y) for w in only_i):
                continue
            if not all(kind_in(w, y) and not kind_in(w, x) for w in only_j):
                continue
            if any(kind_in(w, x) for w in ti) and any(kind_in(w, y) for w in tj):
                return True
        return False

    def _joined_pair(self, rows):
        seen = self.joined_seen
        for low in sorted(seen):
            sides = [kind_tokens(p) for p in re.split(JOINERS, low) if p.strip()]
            if len(sides) != 2 or not sides[0] or not sides[1]:
                continue
            whole = sides[0] + sides[1]
            for i in range(len(rows)):
                for j in range(i + 1, len(rows)):
                    ti, tj = kind_tokens(rows[i]["product"]), kind_tokens(rows[j]["product"])
                    if not ti or not tj:
                        continue
                    if kinds_within(ti, tj) and kinds_within(tj, ti):
                        continue
                    if not kinds_within(ti + tj, whole) or rows[i]["quantity"] != rows[j]["quantity"]:
                        continue
                    if not self._sides_split(ti, tj, sides) or self.joined_clash(rows[i], rows[j]):
                        continue
                    got = self.joined_phrase(rows[i]["product"], rows[j]["product"], seen[low])
                    if got:
                        return i, j, seen[low], got[0], got[1]
        return None

    def merge_joined(self, obs: Any) -> str:
        run = self.run
        self._note_joined_labels(obs)
        remarks: list[str] = []
        for _ in range(2):
            rows = run.needs.items()
            if len(rows) < 2 or not self.joined_seen or run.trolley.used or run.trolley.units:
                break
            if run.needs.uncovered or run.needs.indifferent:
                break
            if any(self.product_question(line, rows) for line in run.questions_log):
                break
            found = self._joined_pair(rows)
            if not found:
                break
            remarks.append(self.merge_items(*found))
        return " ".join(r for r in remarks if r)

    def _merged_requirements(self, first: dict[str, Any], second: dict[str, Any]) -> list[str]:
        merged: list[str] = []
        lowered: set[str] = set()
        for req in list(first["requirements"]) + list(second["requirements"]):
            head, _, tail = str(req).partition(":")
            bare = re.sub(r"\s*\([^()]*\)", "", head).strip() + ":" + tail
            if bare.lower() in lowered:
                continue
            lowered.add(bare.lower())
            merged.append(bare)
        return merged

    def _drop_product_caches(self) -> None:
        run = self.run
        self.home_memo = None
        run.intake.menu_memo = {}
        run.talk.quota_reads = {}
        run.walk.walked = set()

    def merge_items(self, i: int, j: int, label: str, phrase: str, in_order: bool = True) -> str:
        run = self.run
        needs = run.needs
        book = needs.reading
        products = needs.items()
        listed = [r for r in book.get("items") or [] if isinstance(r, dict) and str(r.get("product") or "").strip()]
        if len(listed) != len(products) or not (0 <= i < j < len(products)):
            return ""
        pair_kinds = kind_tokens(products[i]["product"]) + kind_tokens(products[j]["product"])
        label_kinds = kind_tokens(label)
        rest_kinds = [kind_tokens(row["product"]) for spot, row in enumerate(products) if spot not in (i, j)]
        remaining = len(products) - 1
        new_tag = phrase if remaining >= 2 else ""

        def belongs(tag: Any) -> bool:
            tokens = kind_tokens(tag)
            if not tokens or not kinds_within(tokens, label_kinds):
                return False
            if not any(kind_in(t, pair_kinds) for t in tokens):
                return False
            return all(not (kinds_within(tokens, o) and kinds_within(o, tokens)) for o in rest_kinds)

        def swap(hit: Any) -> str:
            if belongs(hit.group(1)):
                return f" ({new_tag})" if new_tag else ""
            return hit.group(0)

        def retag(text: Any) -> Any:
            if isinstance(text, str):
                head, colon, tail = text.partition(":")
                return re.sub(r"\s*\(([^()]*)\)", swap, head) + colon + tail
            return text

        def retag_list(values: list[Any]) -> list[Any]:
            fresh: list[Any] = []
            for value in map(retag, values):
                if value not in fresh:
                    fresh.append(value)
            return fresh
        joined = {"product": phrase, "quantity": products[i]["quantity"],
                  "requirements": self._merged_requirements(products[i], products[j])}
        survivors = []
        for spot, row in enumerate(listed):
            if spot != j:
                survivors.append(joined if spot == i else row)
        needs.written = retag_list(needs.written)
        needs.uncovered = retag_list(needs.uncovered)
        needs.let_go = retag_list(needs.let_go)
        needs.indifferent = retag_list(needs.indifferent)
        if needs.now_reqs is not None:
            needs.now_reqs = retag_list(needs.now_reqs)
        needs.resolved = list(map(retag, needs.resolved or []))
        for field in ("requirements", "hints", "references", "soft_wants"):
            if isinstance(book.get(field), list):
                book[field] = retag_list(book[field])
        targets = book.get("reference_targets")
        if isinstance(targets, list):
            for target in targets:
                if isinstance(target, dict) and belongs(str(target.get("product") or "")):
                    target["product"] = phrase
        for ref in run.refs or []:
            for field in ("req", "head", "now"):
                if isinstance(ref.get(field), str):
                    ref[field] = retag(ref[field])
            if belongs(str(ref.get("tag") or "")):
                ref["tag"] = new_tag
        talk = run.talk
        if isinstance(talk.pending_seconds, list):
            talk.pending_seconds = list(map(retag, talk.pending_seconds))
        if isinstance(talk.open_pair, list):
            talk.open_pair = list(map(retag, talk.open_pair))
        if remaining < 2:
            book.pop("items", None)
            book["product"] = phrase
            book["quantity"] = products[i]["quantity"]
        else:
            book["items"] = survivors
            book["product"] = " and ".join(str(row.get("product")) for row in survivors)
        needs.version += 1
        self._drop_product_caches()
        run.wire.scale = 1.0 + 0.3 * min(2, max(0, len(needs.items()) - 1))
        run.log.append({"kind": "merge_joined", "label": label, "products": [products[i]["product"], products[j]["product"]],
                        "into": phrase, "label_words": in_order, "products_left": max(1, len(needs.items()))})
        tail = "holds one line for it" if remaining >= 2 else "holds ONE line"
        return (f"'{label}' is a SINGLE category in this store, and the request wants one product from it ('{phrase}'), so the order "
                + tail + ", not one line for each kind its name joins. Requirements now: " + needs.text_body())

    def lookup_key(self, tool: str | None) -> str | None:
        schema = self.run.shapes.get(tool or "") or {}
        props = schema.get("properties") if isinstance(schema.get("properties"), dict) else {}
        required = schema.get("required")
        needed = [k for k in required or [] if k in props] if isinstance(required, list) else []
        texts = [k for k, p in props.items() if isinstance(p, dict) and p.get("type") == "string" and not p.get("enum")]
        slot = next((k for k in texts if k in needed), None)
        if not slot and len(texts) == 1:
            slot = texts[0]
        if not slot or any(k != slot for k in needed):
            return None
        return slot

    def lookup_tools(self) -> list[str]:
        run = self.run
        busy = {run.role(r) for r in ("search", "message", "cart", "cart_view", "order", "decline")} - {None}
        return [name for name in run.shapes if name not in busy and self.lookup_key(name)]

    def lookup_call(self, ident: str, tool: str | None) -> dict[str, Any] | None:
        run = self.run
        if not tool:
            return None
        if tool == run.role("details"):
            return run.judge.details_call(ident)
        known = run.archive.ids
        for entry in reversed(run.log):
            if entry.get("kind") != "env":
                continue
            for act in entry.get("actions") or []:
                args = act.get("args") if act.get("name") == tool else None
                if isinstance(args, dict) and any(isinstance(v, str) and v in known for v in args.values()):
                    return {k: (ident if isinstance(v, str) and v in known else v) for k, v in args.items()}
        slot = self.lookup_key(tool)
        return {slot: ident} if slot else None

    @staticmethod
    def rec_id(d: dict[str, Any]) -> str | None:
        for exact in (True, False):
            for key, val in d.items():
                name = str(key).lower()
                named = name == "id" if exact else name.endswith(("id", "key"))
                if named and isinstance(val, str) and re.fullmatch(ID_LIKE, val):
                    return val
        return None

    @staticmethod
    def asked_ids(call_args: Any) -> set[str]:
        args = call_args if isinstance(call_args, dict) else {}
        found = set()
        for v in args.values():
            if isinstance(v, str) and re.fullmatch(ID_LIKE, v):
                found.add(v)
        for v in args.values():
            if isinstance(v, list):
                found.update(x for x in v if isinstance(x, str) and re.fullmatch(ID_LIKE, x))
        return found

    def _book_stock(self, ident: str, state: bool, source: str, epoch: int, newly: list[str]) -> None:
        self.stock_seen = True
        history = self.stock_by.setdefault(ident, {})
        opposite = history.get("row" if source == "lookup" else "lookup")
        if opposite and opposite[0] == epoch and opposite[1] != state:
            self.rows_lag = True
        earlier_row = history.get("row")
        if source == "row" and earlier_row and earlier_row[1] != state:
            self.rows_live = True
        history[source] = (epoch, state)
        if state is not False:
            if source == "lookup" or self.dead_by.get(ident) != "lookup":
                self.dead.discard(ident)
            return
        if ident not in self.dead and ident in self.run.trolley.units and ident not in newly:
            newly.append(ident)
        self.dead.add(ident)
        self.dead_by[ident] = source

    @staticmethod
    def _has_property_map(record: dict[str, Any]) -> bool:
        for v in record.values():
            if isinstance(v, dict) and sum(1 for x in v.values() if isinstance(x, (str, int, float))) >= 2:
                return True
        return False

    def learn_caps(self, call_name: str, call_args: Any, obs: Any) -> list[str]:
        run = self.run
        named_ids = self.asked_ids(call_args)
        tool = str(call_name)
        tally = self.tool_caps.setdefault(tool, {"calls": 0, "id": 0, "stock": 0, "props": 0, "notes": 0, "price": 0})
        tally["calls"] += 1
        seen = dict.fromkeys(("id", "stock", "props", "notes", "price"), False)
        tool_keys = self.tool_keys.setdefault(tool, set())
        note_keys = self.note_keys.setdefault(tool, set())
        cart_tools = {run.role("cart"), run.role("cart_view")} - {None}
        epoch = run.cart_seq
        newly: list[str] = []
        watch = run.watch
        failed = isinstance(obs, dict) and obs.get("error")
        for record in [] if failed else Judge.dicts_in([obs]):
            ident = self.rec_id(record)
            if not ident:
                continue
            state = watch.stock_state(record)
            looked_up = ident in named_ids or call_name in cart_tools
            if state is not None:
                self._book_stock(ident, state, "lookup" if looked_up else "row", epoch, newly)
            if ident not in named_ids:
                if call_name not in cart_tools:
                    self.row_keys.update(str(k) for k in record)
                    self.row_text.setdefault(ident, {}).update({str(k): v for k, v in record.items() if isinstance(v, str)})
                continue
            seen["id"] = True
            self.read_with.setdefault(tool, set()).add(ident)
            self.text_keys.setdefault(tool, set()).update(str(k) for k, v in record.items() if v is None or isinstance(v, str))
            if state is not None:
                seen["stock"] = True
            if not seen["props"]:
                seen["props"] = self._has_property_map(record)
            if not seen["price"]:
                seen["price"] = any(is_number(v) and re.search(r"price|cost", str(k).lower()) for k, v in record.items())
            tool_keys.update(str(k) for k in record)
            row = self.row_text.get(ident)
            for k, v in record.items():
                if not isinstance(v, str) or len(v.split()) < 8:
                    continue
                if row is None:
                    fresh_note = str(k) not in self.row_keys
                else:
                    fresh_note = row.get(str(k)) != v
                if fresh_note:
                    note_keys.add(str(k))
                    seen["notes"] = True
        for key, flag in seen.items():
            tally[key] += int(flag)
        return newly

    def stock_tool(self) -> str | None:
        run = self.run
        caps = self.tool_caps or {}
        tools = self.lookup_tools()
        showing = [t for t in tools if (caps.get(t) or {}).get("stock", 0) > 0]
        if showing:
            return max(showing, key=lambda t: (caps[t]["stock"], t == run.role("view"), t == run.role("details"), t))
        guess = self.role_guess
        if guess is None:
            try:
                guess = run.closer.code_roles()
            except Exception:
                guess = {}
            self.role_guess = guess
        for tool in (run.role("view"), guess.get("view"), run.role("details")):
            got = caps.get(tool or "") or {}
            if not tool or tool not in tools:
                continue
            if got.get("id", 0) >= 2 and got.get("stock", 0) == 0:
                continue
            return tool
        busy = {guess.get(r) for r in ("cart", "cart_view", "order", "decline", "message")} - {None}
        for tool in tools:
            if tool not in busy and not (caps.get(tool) or {}).get("id", 0):
                return tool
        return None

    def notes_tool(self) -> str | None:
        run = self.run
        caps = self.tool_caps or {}
        details = run.role("details")
        detail_keys = (self.tool_keys or {}).get(details or "", set())
        notes = self.note_keys or {}
        found = [t for t in self.lookup_tools() if t != details and set(notes.get(t) or set()) - set(detail_keys)]
        if not found:
            return None
        return max(found, key=lambda t: ((caps.get(t) or {}).get("notes", 0), t))

    def dead_lines(self) -> set[str]:
        return set(self.dead or set())

    def stale_lines(self) -> list[str]:
        run = self.run
        seqs = run.line_seq or {}
        mark = run.cart_seq
        return [i for i in run.trolley.units if seqs.get(i, -1) < mark]

    def _price_or_inf(self, ident: str) -> float:
        price = self.run.archive.price(ident)
        return float(price) if is_number(price) else float("inf")

    def _verified_swap(self, line, gone, version, label, several, cap, qty, rest):
        run = self.run
        judge = run.judge
        cost = self._price_or_inf
        blocked = set(gone) | set(run.refused_lines) | set(run.trolley.units) | {line}
        found = []
        for stamp in run.passed:
            ident, _, ver = stamp.rpartition("|")
            if ver != version or ident in blocked:
                continue
            if judge.field_view(ident)["fails"]:
                continue
            other_label = judge.shelf_known(ident)
            if label and (other_label != label if several else other_label not in ("", label)):
                continue
            price = cost(ident)
            if is_number(cap) and price != float("inf") and price * qty + rest > cap:
                continue
            found.append(ident)
        return found

    def _swap_rows(self, line, gone, version, label, slot, cap, qty, rest):
        run = self.run
        closer = run.closer
        cost = self._price_or_inf
        plain_values = [(prop, plain) for prop, plain in closer.code_values(slot) if self.ed_plain(plain)]
        failed = {stamp.split("|")[0] for stamp, verdict in run.rulings.items() if verdict and stamp.split("|")[1:2] == [version]}
        pointers = run.refdesk.ref_pointers()
        blocked = set(gone) | set(run.refused_lines) | set(run.trolley.units) | {line}
        ranked = []
        for ident, row in closer.code_rows().items():
            price = row["price"] if is_number(row["price"]) else cost(ident)
            if ident in blocked or row["stock"] is not True:
                continue
            if (label and row["shelf"] != label) or price == float("inf"):
                continue
            if is_number(cap) and price * qty + rest > cap:
                continue
            if ident in failed or ident in pointers or run.refdesk.ref_excluded(row["fields"], slot):
                continue
            opened = ident in run.archive.read
            if opened:
                if run.judge.field_view(ident)["fails"]:
                    continue
                if any(closer.code_clash(row["fields"], prop, plain) for prop, plain in plain_values):
                    continue
                if not all(closer.code_shows(row, prop, plain, set(), True) for prop, plain in plain_values):
                    continue
            ranked.append((0 if opened else 1, price, ident))
        ranked.sort()
        return ranked

    def dead_swap(self, line: str) -> tuple[str, str]:
        run = self.run
        gone, version = self.dead_lines(), str(run.needs.version)
        label = run.judge.shelf_known(line)
        several = len(run.needs.items()) >= 2
        cap, qty = run.needs.spend_cap, max(1.0, run.trolley.held(line))
        cost = self._price_or_inf
        rest = 0
        for other in run.trolley.units:
            if other != line and cost(other) != float("inf"):
                rest += cost(other) * run.trolley.held(other)
        verified = self._verified_swap(line, gone, version, label, several, cap, qty, rest)
        if verified:
            return min(verified, key=lambda i: (cost(i), i)), ""
        if several and not label:
            return "", ""
        slot = None
        if several:
            slot = next((n for n, home in enumerate(self.home_shelves()) if label and label in home), None)
        ranked = self._swap_rows(line, gone, version, label, slot, cap, qty, rest)
        unread = [x for x in ranked if x[0] == 1]
        lowest = min(unread, key=lambda x: (x[1], x[2])) if unread else None
        if lowest and run.needs.priority() == "cheapest" and all(lowest[1] < p for r, p, _ in ranked if r == 0):
            ranked.remove(lowest)
            ranked.insert(0, lowest)
        top = ranked[:3]
        self.swap_unread = [ident for r, _, ident in top if r == 1]
        self.swap_page = None
        if all(r != 0 for r, _, _ in ranked):
            try:
                self.swap_page = self.dead_page(slot)
            except Exception:
                self.swap_page = None
        shown = []
        for r, p, ident in top:
            figure = int(p) if float(p).is_integer() else p
            shown.append(f"{ident} ({figure}{', read: its record shows every stated value' if r == 0 else ''})")
        return "", ", ".join(shown)

    def dead_page(self, k: int | None) -> dict[str, Any] | None:
        run = self.run
        found = run.watch.page_arg()
        if not found:
            return None
        walker, watch = run.walk, run.watch
        open_walks = [w for w in run.page_book.values() if walker.walk_carries(w["args"], k) and watch.next_page(w) is not None]
        if not open_walks:
            return None
        best = max(open_walks, key=lambda w: (len(walker.walk_carries(w["args"], k)), -len(w["read"])))
        return {**best["args"], found[0]: watch.next_page(best)}

    def swap_open(self) -> str:
        run = self.run
        if run.intake.swap_done:
            return ""
        gone = self.dead_lines()
        for ident in self.swap_unread or []:
            if ident not in run.archive.read and ident not in gone and ident not in run.trolley.units:
                return ident
        return ""

    def dead_note(self, lines: list[str]) -> str:
        run = self.run
        parts = []
        run.event_seen = True
        for line in lines:
            self.swap_page, self.swap_unread = None, []
            pick, rows = self.dead_swap(line)
            head = f"{line} is SOLD OUT (the latest record says so), and an order holding it fails. Take it out of the cart"
            if pick:
                least = ", the cheapest such listing" if run.needs.priority() == "cheapest" else ""
                parts.append(f"{head}, then add {pick} (the item check found it meets every requirement now in force{least}), "
                             "re-read its stock and price, then order.")
            elif rows:
                parts.append(f"{head}; there is no other verified match for that product, so look at its category's cheapest "
                             f"in-stock rows inside the budget ({rows}; rows already read whose record misses or breaks a stated "
                             "requirement were dropped, and an unread one must have its details opened first), put the first that fits in, re-read "
                             "its stock and price, then order.")
            else:
                parts.append(f"{head} and find another listing that fits.")
            if self.swap_page and not pick:
                parts.append("Further rows of the same kind sit on the next unread page of the search that leads with the stated value, "
                             + json.dumps(self.swap_page, default=str) + ".")
        return " ".join(parts)

    def refusal_reread(self, obs: Any = None) -> str:
        heard = ""
        try:
            heard, gone = self.refusal_gone(obs)
        except Exception as err:
            gone = ""
            self.run.log.append({"kind": "refusal_gone_error", "error": f"{type(err).__name__}: {err}"[:300]})
        return heard + (gone if gone else self.refusal_queue())

    def refusal_gone(self, obs: Any) -> tuple[str, str]:
        run = self.run
        trolley = run.trolley
        held = [i for i in trolley.units if trolley.held(i) > 0]
        if obs is None or len(held) != 1 or run.wire.mute or run.covering or run.last_kind != "order":
            return "", ""
        texts = run.cover.shopper_lines(run.role("order") or "", obs, True)
        if not texts:
            return "", ""
        needs = run.needs

        def snapshot():
            return (needs.version, len(needs.let_go), len(needs.uncovered), str(needs.spend_cap), "|".join(needs.props_left()))
        before = snapshot()
        note = needs.hear(texts, None)
        after = snapshot()
        heard = (" " + note) if note else ""
        if not note or after[0] == before[0] or after[1:] != before[1:]:
            return heard, ""
        line = held[0]
        self.dead.add(line)
        self.dead_by[line] = "lookup"
        run.log.append({"kind": "refusal_dead", "line": line})
        return heard, (" The refusal reports an item that is no longer available, and only one line was in the cart, so the code treats " + line
                       + " as sold out (no re-read): " + self.dead_note([line]))

    def refusal_queue(self) -> str:
        run = self.run
        tool = self.stock_tool()
        lines = list(run.trolley.units)[:4]
        if not lines or not tool or run.wire.mute or run.covering or run.reads.open_queue:
            return ""
        if run.moves_left() < len(lines) + 3 or run.wire.spare() < CLOSING:
            return ""
        reads = run.reads
        reads.open_queue = lines
        reads.open_recheck, reads.open_purpose, reads.open_tool = True, "cart", tool
        reads.open_say = "Order refused, so the cart lines get a fresh stock check (one of them may be gone)."
        return " The cart lines' stock is read for you now (" + tool + ")."

    def code_judges_all(self) -> bool:
        run = self.run
        items = run.needs.items()
        slots = range(len(items)) if items else [None]
        values = [plain for k in slots for _, plain in run.closer.code_values(k)]
        plain = [v for v in values if self.ed_plain(v)]
        resolved = self.differ_done(None)
        judged = 0
        for req in run.needs.props_left():
            if any(run.refdesk.ref_match(ref, req) for ref in resolved):
                judged += 1
        return len(plain) + judged >= len(list(run.needs.props_left()))

    def differ_done(self, k: int | None) -> list[dict[str, Any]]:
        run = self.run
        desk = run.refdesk
        rows, left, found = run.needs.items(), run.needs.props_left(), []
        for ref in run.refs or []:
            if ref.get("relation") != "differ" or ref.get("status") != "resolved":
                continue
            if not str(ref.get("value") or "").strip() or not desk.ref_live(ref):
                continue
            if not any(desk.ref_match(ref, req) for req in left):
                continue
            if rows and k is not None and ref.get("tag") and run.talk.product_of(f"x ({ref['tag']})") not in (k, None):
                continue
            found.append(ref)
        return found

    def _repair_fits(self, picks, own: bool, version: str) -> bool:
        run = self.run
        fit = run.closer.code_fit or {}
        if not picks:
            return False
        return all(f"{i}|{version}" in run.passed or (own and i in run.archive.read and fit.get(i)) for i, _ in picks)

    def _repair_picks(self, several: bool, gone: set, own: bool, version: str):
        run = self.run
        closer = run.closer
        avoid = set(gone)
        pool: dict[str, dict[str, Any]] = {}
        if several:
            pool = {e["id"]: e for k in range(len(run.needs.items())) for e in self.ed_pool(k)}
            avoid |= {i for i, e in pool.items() if e["clash"]}
        picks = closer.code_picks(avoid)
        checks = rounds = 0
        while several and picks and checks < 3 and rounds < 30 and run.wire.spare() > CLOSING + 30:
            unchecked = [i for i, _ in picks if not self._repair_fits([(i, 1)], own, version)]
            if not unchecked:
                break
            bare = [i for i in unchecked if not pool.get(i) or not pool[i]["revealed_ok"] or pool[i]["misses"]]
            if not bare:
                checks += 1
                run.judge.check_item(unchecked[0], "ORDER")
                picks = closer.code_picks(avoid)
                continue
            avoid.update(bare)
            avoid.update(i for i, e in pool.items() if not e["read"] and f"{i}|{version}" not in run.passed
                         and (not e["revealed_ok"] or e["misses"]))
            rounds += 1
            picks = closer.code_picks(avoid)
        return picks

    def _repair_window(self):
        run = self.run
        wire = run.wire
        several = len(run.needs.items()) >= 2
        early = several and run.refusal_n == 0 and run.moves_left() > run.intake.order_due()
        if run.refusal_n <= 0 and run.moves_left() > run.intake.order_due() and not early:
            return None
        blocked = (not run.role("cart") or not run.role("order") or run.moves_left() < 3 or wire.spare() <= CLOSING
                   or wire.mute or run.covering or wire.done)
        if blocked:
            return None
        mark: Any = ("early", run.refusal_n) if early else run.refusal_n
        if mark in self.repair_tried:
            return None
        self.repair_tried.add(mark)
        return several

    def _proof_lets_decline(self) -> bool:
        run = self.run
        try:
            proof = self.decline_proof()
        except Exception as err:
            proof = ""
            run.log.append({"kind": "decline_proof_error", "error": f"{type(err).__name__}: {err}"[:300]})
        if proof:
            run.log.append({"kind": "decline_proof", "why": proof})
        return bool(proof)

    def decline_repair(self) -> str:
        run = self.run
        several = self._repair_window()
        if several is None:
            return ""
        if several and self._proof_lets_decline():
            return ""
        gone = self.dead_lines()
        stale = set(self.stale_lines())
        alive = {line for line in run.trolley.units if line not in gone and line not in stale}
        saved = set(run.refused_lines)
        version = str(run.needs.version)
        own = self.code_judges_all()
        done = False
        try:
            run.refused_lines.difference_update(alive)
            picks = self._repair_picks(several, gone, own, version)
            if not self._repair_fits(picks, own, version):
                run.log.append({"kind": "decline_repair", "picks": picks, "done": False})
            elif run.moves_left() < len(picks) + 2:
                run.log.append({"kind": "decline_repair", "picks": picks, "done": False, "why": "steps"})
            else:
                done = self._repair_put(picks)
                run.log.append({"kind": "decline_repair", "picks": picks, "done": done, "cart": dict(run.trolley.units)})
        except Exception as err:
            done = False
            run.log.append({"kind": "decline_repair_error", "error": f"{type(err).__name__}: {err}"[:300]})
        finally:
            if not done:
                run.refused_lines.clear()
                run.refused_lines.update(saved)
        if not done:
            return ""
        several = bool(run.needs.items())
        return ("Not run yet (no step used). Declining fits only when no listing can work, yet the records show a suitable candidate "
                + ("for every product asked for" if several else "") + " (passed by the item check, or opened with each stated "
                "value present in its record), so the code placed " + ("them" if several else "it") + " in the cart and checked stock: "
                + run.trolley.summary() + " Order this cart now; decline only if a record you read shows one of these lines "
                "failing a requirement now in force.")

    def _repair_put(self, picks) -> bool:
        run = self.run
        wire = run.wire
        was = wire.mute
        wire.mute = True
        try:
            bad = run.closer.code_put(picks)
            done = bad == [] and all(i in run.trolley.units for i, _ in picks)
            stale_now = [i for i in self.stale_lines() if any(i == p for p, _ in picks)]
            if done and stale_now and run.moves_left() >= len(stale_now) + 2:
                _, sold = run.closer.code_recheck(stale_now)
                done = not sold
        finally:
            wire.mute = was
        return done and bool(run.trolley.units) and not set(run.trolley.units) & self.dead_lines()

    @staticmethod
    def ed_plain(value: str) -> bool:
        return not re.search(r"\b(?:" + VALUE_LINKS + r")\b|\d\s*(?:-|to)\s*\d|[+<>]|\b(?:about|around|approx\w*|roughly)\b",
                             str(value).lower())

    @staticmethod
    def ed_word(value: str) -> bool:
        text = str(value).lower()
        if not re.search(r"[a-z]{3,}", text) or re.search(r"\d", text):
            return False
        return Catalogue.ed_plain(text) and not re.search(EXCLUDE_WORDS, text) and not ref_text(text)

    def ed_reqs(self, k: int | None) -> list[tuple[str, str, bool]]:
        run = self.run
        needs = run.needs
        several = needs.items()
        found = []
        for req in needs.props_left():
            head, _, value = str(req).partition(":")
            if several and run.talk.product_of(head) not in (k, None):
                continue
            plain = re.sub(r"\((?:stated|revealed[^)]*)\)", "", value).strip()
            found.append((re.split(r"\s*\(", head)[0].strip(), plain, req not in needs.written))
        return found

    def _pool_entry(self, ident, row, ctx) -> dict[str, Any]:
        run = self.run
        closer = run.closer
        plain_values, joins, revealed, pointers, gone_set, failed, live, budget, units, slot = ctx
        opened = ident in run.archive.read
        quoted = row["price"] if row["price"] is not None else run.archive.price(ident)
        price = float(quoted) if is_number(quoted) else None
        gone = ident in gone_set or row["stock"] is False or ident in run.refused_lines
        over = budget is not None and price is not None and price * units > budget + 1e-9
        pointer = ident in pointers or run.refdesk.ref_excluded(row["fields"], slot)
        fields = row["fields"]

        def same_field_values(prop):
            return [val for key, val in fields.items() if same_prop(key, prop)]
        clash = [prop for prop, wanted in plain_values if closer.code_clash(fields, prop, wanted)
                 and not any(closer.code_seen(wanted, val) for val in same_field_values(prop))]
        for prop, parts in joins:
            present = same_field_values(prop)
            if present and not all(any(closer.code_seen(part, val) for val in present) for part in parts):
                clash.append(prop)
        misses = [(prop, wanted) for prop, wanted in plain_values if not closer.code_shows(row, prop, wanted, set(), True)]
        ruled = ident in failed or bool(opened and run.judge.field_view(ident)["fails"])
        verified = ident in live and not gone
        blocked = ruled or pointer or gone or over
        fit = not blocked and (verified or (opened and not misses and not clash))
        entry = {"id": ident, "price": price, "read": opened, "verified": verified, "fit": fit, "gone": gone,
                 "over": over, "pointer": pointer, "ruled": ruled, "clash": clash, "misses": misses, "row": row}
        entry["out"] = bool(gone or over or pointer or ruled or clash)
        entry["revealed_ok"] = all(closer.code_shows(row, prop, wanted, set(), True) for prop, wanted in revealed)
        return entry

    def _pool_home(self, k, several) -> set:
        homes = self.home_shelves()
        if not several:
            return set(self.run.watch.shelf_labels()) or (set(homes[0]) if homes else set())
        if k is not None and k < len(homes):
            return set(homes[k])
        return set()

    def ed_pool(self, k: int | None) -> list[dict[str, Any]]:
        run = self.run
        needs = run.needs
        several = needs.items()
        version = str(needs.version)
        mark = (k, len(run.archive.items), len(run.archive.read), version, "|".join(needs.props_left()), len(run.passed),
                sum(1 for v in run.rulings.values() if v), tuple(sorted(self.dead_lines())), tuple(sorted(run.refused_lines)),
                str(needs.spend_cap), len(run.refs or []))
        if mark in self.ed_memo:
            return self.ed_memo[mark]
        if len(self.ed_memo) > 64:
            self.ed_memo.clear()
        home = self._pool_home(k, several)
        entries: list[dict[str, Any]] = []
        if home:
            counts = needs.counts()
            if several and k is not None and k < len(counts):
                units = counts[k]
            else:
                units = counts[0] if counts else 1
            failed = {stamp.split("|")[0] for stamp, verdict in run.rulings.items() if verdict and stamp.split("|")[1:2] == [version]}
            live = {stamp.rpartition("|")[0] for stamp in run.passed if stamp.rpartition("|")[2] == version}
            budget = float(needs.spend_cap) if is_number(needs.spend_cap) else None
            slot = k if several else None
            closer = run.closer
            plain_values = [(p, v) for p, v in closer.code_values(slot) if self.ed_plain(v)]
            joins = []
            for p, v in closer.code_values(slot):
                if "+" in v:
                    parts = [x.strip() for x in v.split("+") if x.strip()]
                    if len(parts) >= 2 and all(re.search(r"[a-z]{2,}", x.lower()) for x in parts):
                        joins.append((p, parts))
            revealed = [(p, v) for p, v, shown in self.ed_reqs(slot) if shown]
            ctx = (plain_values, joins, revealed, run.refdesk.ref_pointers(), self.dead_lines(), failed, live, budget, units, slot)
            entries = [self._pool_entry(ident, row, ctx) for ident, row in closer.code_rows().items() if row["shelf"] in home]
        self.ed_memo[mark] = entries
        return entries

    def searches_run(self) -> int:
        run = self.run
        search = run.role("search")
        seen = set()
        for entry in run.log:
            if isinstance(entry, dict) and entry.get("kind") == "env":
                seen.update(as_text(act.get("args")) for act in entry.get("actions") or [] if act.get("name") == search)
        return len(seen)

    def shelf_paged(self, k: int | None) -> bool:
        run = self.run
        if not run.watch.page_arg():
            return True
        home = self._pool_home(k, run.needs.items())
        for walk in run.page_book.values():
            if home and not walk["shelves"] & home:
                continue
            if len(walk["read"]) >= 2 or run.watch.next_page(walk) is None:
                return True
        return False

    def walk_away_asked(self) -> bool:
        reading = self.run.needs.reading
        rules = [str(r) for r in reading.get("rules") or []] if isinstance(reading.get("rules"), list) else []
        return reading.get("decline_no_names") is not None or any(re.search(WALK_AWAY, rule.lower()) for rule in rules)

    def _need_unmet(self, k: int | None, live: list[dict[str, Any]], every: bool) -> str:
        # one need no in-budget listing shows: each one read lacks it, and no unread row's title writes it
        closer = self.run.closer
        read = [e for e in live if e["read"]]
        unread = [e for e in live if not e["read"]]
        for prop, value, revealed in self.ed_reqs(k):
            if not value or ref_text(value) or not (revealed or every):
                continue
            # a limit or an exclusion has no word to look for
            if not self.ed_plain(value) or re.search(EXCLUDE_WORDS, value.lower()):
                continue
            if any(closer.code_shows(e["row"], prop, value, set(), True) for e in read):
                continue
            if not any(closer.code_seen(value, e["row"].get("title") or "") for e in unread):
                return f"{prop}: {value}"
        return ""

    def decline_proof(self) -> str:
        """Why a decline stands, or "" when it does not. A wrong order scores what a decline scores and real decline tasks are rare,
        so only these stand: the kind has nothing in stock within budget, or a need is shown unmet across the kind after real paging."""
        run = self.run
        needs = run.needs
        if self.searches_run() < DECLINE_SEARCHES:
            return ""
        wanted = needs.items()
        counts = needs.counts()
        budget = float(needs.spend_cap) if is_number(needs.spend_cap) else None
        walk_away = self.walk_away_asked()
        floors = []
        for n, k in enumerate(range(len(wanted)) if wanted else [None]):
            name = wanted[n]["product"] if wanted else "the product"
            entries = self.ed_pool(k)
            live = [e for e in entries if not e["gone"] and not e["over"]]
            paged = self.shelf_paged(k)
            if not live:
                if entries and paged:
                    return f"no listing of {name} in stock within the budget"
                continue
            priced = [e["price"] for e in live if e["price"] is not None]
            if priced:
                floors.append(min(priced) * (counts[n] if n < len(counts) else 1))
            if any(e["fit"] for e in live) or not paged:
                continue
            reads = sum(1 for e in live if e["read"])
            unmet = self._need_unmet(k, live, walk_away) if reads >= DECLINE_READS else ""
            if unmet:
                return f"no in-budget listing of {name} shows {unmet}"
            if walk_away and reads > DECLINE_READS:
                return f"the request says to walk away and nothing of {name} fits"
        if budget is not None and wanted and len(floors) == len(wanted) and sum(floors) > budget + 1e-9:
            return "the cheapest in-stock listings of the products together go over the budget"
        return ""

    def _closeness(self, entry: dict[str, Any], revealed: list[tuple[str, str]]):
        closer = self.run.closer
        lacking = sum(1 for prop, value in revealed if not closer.code_shows(entry["row"], prop, value, set(), True))
        price = entry["price"] if entry["price"] is not None else float("inf")
        return (lacking, len(entry["clash"]), entry["ruled"], len(entry["misses"]), not entry["read"], price, entry["id"])

    def _within_budget(self, ranked: list[list[dict[str, Any]]], units: list[int], budget: float | None):
        # closest first; over the budget, the line that saves most steps down to a cheaper listing of its product
        chosen = [row[0] for row in ranked]
        for _ in range(16):
            total = sum((e["price"] or 0.0) * n for e, n in zip(chosen, units))
            if budget is None or total <= budget + 1e-9:
                return chosen
            swaps = []
            for spot, row in enumerate(ranked):
                now = chosen[spot]["price"] or 0.0
                swaps += [((now - (e["price"] or 0.0)) * units[spot], spot, e["id"], e) for e in row
                          if e["price"] is not None and e["price"] < now]
            if not swaps:
                return None
            _, spot, _, entry = max(swaps, key=lambda swap: swap[:3])
            chosen[spot] = entry
        return None

    def closest_picks(self, avoid: set[str] | None = None, keep: set[str] | None = None) -> list[tuple[str, int]]:
        run = self.run
        needs = run.needs
        avoid = set(avoid or ()) | self.dead_lines()
        picks = run.closer.code_picks(avoid, keep)
        if picks:
            return picks
        wanted = needs.items()
        slots = list(range(len(wanted))) if wanted else [None]
        counts = needs.counts()
        units = [counts[n] if n < len(counts) else 1 for n in range(len(slots))]
        ranked = []
        for k in slots:
            revealed = [(prop, value) for prop, value, shown in self.ed_reqs(k) if shown]
            pool = self.ed_pool(k)
            usable = [e for e in pool if not e["gone"] and not e["over"] and not e["pointer"] and e["id"] not in avoid]
            # a value still pointing elsewhere is unknown, not failed, and a line left empty sinks every other line
            usable = usable or [e for e in pool if not e["gone"] and not e["over"] and e["id"] not in avoid]
            usable = usable or self.any_listing(k, avoid, units[slots.index(k)])
            if not usable:
                return []
            ranked.append(sorted(usable, key=lambda e: self._closeness(e, revealed)))
        budget = float(needs.spend_cap) if is_number(needs.spend_cap) else None
        chosen = self._within_budget(ranked, units, budget)
        if not chosen or len({e["id"] for e in chosen}) < len(chosen):
            return []
        return [(e["id"], n) for e, n in zip(chosen, units)]

    def any_listing(self, k: int | None, avoid: set[str], units: int) -> list[dict[str, Any]]:
        """Last resort for an order: anything in stock within the budget whose row names the product, off its shelf or not."""
        run = self.run
        needs = run.needs
        wanted = needs.items()
        name = str(wanted[k]["product"]) if wanted and k is not None else str(needs.reading.get("product") or "")
        words = {w.rstrip("s") for w in re.findall(r"[a-z]{3,}", name.lower())}
        budget = float(needs.spend_cap) if is_number(needs.spend_cap) else None
        found = []
        for ident, row in run.closer.code_rows().items():
            price = row["price"] if row["price"] is not None else run.archive.price(ident)
            price = float(price) if is_number(price) else None
            text = " ".join(str(row.get(part) or "") for part in ("title", "shelf")).lower()
            if ident in avoid or ident in self.dead_lines() or row["stock"] is False or ident in run.refused_lines:
                continue
            if budget is not None and price is not None and price * units > budget + 1e-9:
                continue
            if words & {w.rstrip("s") for w in re.findall(r"[a-z]{3,}", text)}:
                found.append({"id": ident, "price": price, "read": ident in run.archive.read, "gone": False, "over": False, "pointer": False,
                              "ruled": False, "clash": [], "misses": [], "row": row, "fit": False, "verified": False, "out": False, "revealed_ok": False})
        return found

    def closest_basket(self) -> str:
        run = self.run
        wire, needs = run.wire, run.needs
        if not run.role("cart") or not run.role("order") or run.moves_left() < 2 or wire.done or wire.broke:
            return ""
        held = set(run.trolley.units) - self.dead_lines()
        picks = self.closest_picks(None, held)
        if not picks or not self._repair_put(picks):
            run.log.append({"kind": "closest_basket", "picks": picks, "done": False})
            return ""
        version = str(needs.version)
        lines = [ident for ident, _ in picks]
        for ident in lines:
            stamp = f"{ident}|{version}"
            for memo in [m for m in run.rulings if m == stamp or m.startswith(stamp + "|")]:
                run.rulings[memo] = ""
            run.passed.add(stamp)
            self.near_ok.add(stamp)
            run.pass_price[ident] = run.archive.price(ident)
        self.near_cart = {"ver": needs.version, "refusals": run.refusal_n, "lines": lines}
        run.log.append({"kind": "closest_basket", "picks": picks, "done": True, "cart": dict(run.trolley.units)})
        return ("Not run yet (no step used). A wrong order costs no more than a decline, and the store does hold listings of what was "
                "asked for in stock within the budget, so the code loaded the closest basket into the cart and checked its stock: "
                + run.trolley.summary() + " Order this cart now, or first swap a line for a listing that meets more of the requirements.")

    def last_basket(self, avoid: set[str]) -> bool:
        run = self.run
        try:
            if not run.role("cart") or run.moves_left() < 2 or self.decline_proof():
                return False
            picks = self.closest_picks(avoid, set(run.trolley.units) - set(avoid) - self.dead_lines())
            done = bool(picks) and self._repair_put(picks)
            run.log.append({"kind": "last_basket", "picks": picks, "done": done})
            return done
        except Exception as err:
            run.log.append({"kind": "last_basket_error", "error": f"{type(err).__name__}: {err}"[:300]})
            return False

    def _lead_scores(self, k: int, hits: set, book: list) -> dict[str, int]:
        run = self.run
        wanted_words = []
        for _, value, _ in self.ed_reqs(k):
            wanted_words.append({t for t in re.findall(r"[a-z0-9]{2,}", value.lower()) if t not in LINK_WORDS})
        scores: dict[str, int] = dict.fromkeys(hits, 1)
        for search in book:
            leading = []
            for text in run.watch.free_text(search["args"]):
                leading.extend(re.findall(r"[a-z0-9]{2,}", text.lower()))
            if run.lead_words:
                leading = leading[: run.lead_words]
            leading_set = set(leading)
            carried = sum(1 for group in wanted_words if group & leading_set)
            if not carried:
                continue
            for ident in (search.get("row_shelf") or {}):
                scores[str(ident)] = max(scores.get(str(ident), 0), carried)
        return scores

    def _cheapest_floors(self, count: int, per_unit):
        floors = []
        for k in range(count):
            prices = [e["price"] for e in self.ed_pool(k) if e["fit"] and e["price"] is not None]
            if not prices:
                return None
            floors.append(min(prices) * per_unit(k))
        return floors

    def slack_rows(self) -> list[str]:
        run = self.run
        wanted = run.needs.items()
        budget = run.needs.spend_cap
        if len(wanted) < 2 or not is_number(budget):
            return []
        try:
            counts = run.needs.counts()

            def per_unit(k: int):
                return counts[k] if k < len(counts) else 1
            floors = self._cheapest_floors(len(wanted), per_unit)
            if floors is None or sum(floors) <= float(budget) + 1e-9:
                return []
            book = list(run.page_book.values())
            hits = set(run.cover.all_hits or [])
            queues = []
            for k in range(len(wanted)):
                room = (float(budget) - (sum(floors) - floors[k])) / max(1, per_unit(k))
                scores = self._lead_scores(k, hits, book)
                ranked = []
                for e in self.ed_pool(k):
                    if e["read"] or e["gone"] or e["pointer"] or e["ruled"] or e["clash"] or e["price"] is None:
                        continue
                    if 0 < room and e["price"] <= room + 1e-9:
                        ranked.append((-scores.get(e["id"], 0), e["price"], e["id"]))
                queues.append([ident for _, _, ident in sorted(ranked)])
            chosen: list[str] = []
            while len(chosen) < 3 and any(queues):
                for queue in queues:
                    if queue and len(chosen) < 3:
                        chosen.append(queue.pop(0))
            if chosen:
                self.slack_total = sum(floors)
            return chosen
        except Exception as err:
            run.log.append({"kind": "slack_error", "error": f"{type(err).__name__}: {err}"[:300]})
            return []

    def verify_problems(self, ident: str) -> list[str] | None:
        for entry in reversed(self.run.log):
            if not isinstance(entry, dict) or entry.get("kind") != "verify" or entry.get("item") != ident:
                continue
            res = entry.get("result")
            if isinstance(res, dict) and res.get("fits") is False:
                return [str(p) for p in res.get("problems") or [] if str(p).strip()]
            return None
        return None

    def ed_news(self, k: int | None) -> bool:
        run = self.run
        if run.refusal_n > 0:
            return True
        rows = run.needs.items()
        for req in list(run.needs.uncovered) + list(run.needs.let_go):
            if not rows or run.talk.product_of(str(req).partition(":")[0]) in (k, None):
                return True
        return False

    def near_hold(self) -> str:
        run = self.run
        try:
            info = self.near_cart
            if not info or info.get("ver") != run.needs.version or info.get("refusals") != run.refusal_n or not run.role("order"):
                return ""
            lines = list(info.get("lines") or [])
            if not lines:
                return ""
            for ident in lines:
                if ident not in run.trolley.units or ident in self.dead_lines() or ident in run.refused_lines:
                    return ""
            wire = run.wire
            if run.moves_left() < 1 or wire.done or wire.broke or wire.spare() < 2:
                return ""
            self.near_cart = None
            run.log.append({"kind": "near_order", "lines": lines})
            was = wire.mute
            try:
                run.closer.finish_now()
            finally:
                wire.mute = was
            return ("Not run (no step used). A wording review already found that the cart's lines (" + ", ".join(lines) + ") express the "
                    "requested values differently, and nothing changed since, so the code ordered the cart instead of declining.")
        except Exception as err:
            run.log.append({"kind": "near_hold_error", "error": f"{type(err).__name__}: {err}"[:300]})
            return ""

    def near_check(self) -> str:
        run = self.run
        wire = run.wire
        if self.near_done or not run.role("cart") or not run.role("order") or run.moves_left() < 4:
            return ""
        if wire.spare() <= CLOSING + 30 or wire.mute or run.covering or wire.done or wire.spent_out() or wire.no_model:
            return ""
        try:
            return self.near_basket()
        except Exception as err:
            run.log.append({"kind": "near_error", "error": f"{type(err).__name__}: {err}"[:300]})
            return ""

    def _worded_lines(self, entry, reqs):
        closer = self.run.closer
        lines = []
        for prop, value, revealed in reqs:
            if closer.code_shows(entry["row"], prop, value, set(), True):
                continue
            written = [str(v).strip() for key, v in entry["row"]["fields"].items() if same_prop(key, prop) and str(v).strip()]
            if revealed or not written or not self.ed_word(value):
                return None
            if any(re.search(r"\d", text) for text in written):
                return None
            lines.append((prop, value, "; ".join(dict.fromkeys(written))[:120]))
        return lines

    def _near_candidates(self, pool, reqs):
        judge = self.run.judge
        found = []
        for entry in pool:
            if not entry["read"] or entry["gone"] or entry["over"] or entry["pointer"] or entry["price"] is None:
                continue
            lines = self._worded_lines(entry, reqs)
            usable = lines is not None
            lines = lines or []
            fields_failed, problems = set(), []
            if entry["ruled"]:
                fields_failed = {str(x).split(":", 1)[0].strip() for x in judge.field_view(entry["id"])["fails"]}
                problems = self.verify_problems(entry["id"]) or []
            marks = set()
            for prop, value, _ in lines:
                for token in prop_words(prop) + re.findall(r"[a-z0-9]+", value.lower()):
                    if len(token) >= 3 and token not in LINK_WORDS:
                        marks.add(token)
            if not usable or not lines:
                continue
            props = [prop for prop, _, _ in lines]
            if any(not any(same_prop(f, prop) for prop in props) for f in fields_failed):
                continue
            if all(any(re.search(r"\b" + re.escape(t), problem.lower()) for t in marks) for problem in problems):
                found.append((len(lines), entry["price"], entry["id"], lines))
        return found

    def _near_prompt(self, flat) -> str:
        listing = "\n".join(f"{i + 1}. {ident}: {prop} = '{value[:60]}' | the listing's field says: {says}"
                            for i, (_, ident, prop, value, says) in enumerate(flat))
        return ("Goal: tell whether listings that miss a requested value word for word, which would make the shopping "
                "assistant decline, write it in other words. THE SHOPPER'S REQUEST (context):\n"
                + self.run.request[:1500] + "\n\nLines: a listing, a requested property with its requested value, and what the "
                "listing's own field says for it.\n" + listing + "\n\n1. Go line by line.\n2. Mark it true only when the "
                "field expresses what was requested using different words (synonym, alternate spelling or phrasing, or a more "
                "precise version of the same thing); any other value of the property is false.\n3. When in doubt, answer false.\n"
                "Return: ONE JSON object, nothing else: {\"lines\": [{\"n\": 1, \"same\": true}]}")

    def _near_pools(self, slots, units, basket):
        pools: dict[int, list] = {}
        for n, k in enumerate(slots):
            entries = self.ed_pool(k)
            priced = sorted((e["price"], e["id"]) for e in entries if e["fit"] and e["price"] is not None)
            if priced:
                cheapest_price, cheapest_id = priced[0]
                basket[n] = (cheapest_id, cheapest_price, units(n))
                continue
            if not self.ed_news(k):
                return None
            options = self._near_candidates(entries, self.ed_reqs(k))
            if not options:
                return None
            pools[n] = sorted(options)[: 6 if len(slots) == 1 else 3]
        return pools

    def _near_agreed(self, answer) -> set[int]:
        agreed: set[int] = set()
        if not isinstance(answer, dict) or not isinstance(answer.get("lines"), list):
            return agreed
        for verdict in answer["lines"]:
            if isinstance(verdict, dict) and verdict.get("same") is True and is_number(verdict.get("n")):
                agreed.add(int(verdict["n"]) - 1)
        return agreed

    def near_basket(self) -> str:
        run = self.run
        needs = run.needs
        wanted = needs.items()
        slots: list[int | None] = list(range(len(wanted))) if wanted else [None]
        counts = needs.counts()

        def units(n: int):
            return counts[n] if n < len(counts) else 1
        budget = float(needs.spend_cap) if is_number(needs.spend_cap) else None
        basket: dict[int, tuple[str, float, int]] = {}
        pools = self._near_pools(slots, units, basket)
        if not pools:
            return ""

        def total(extra: float = 0.0) -> float:
            return sum(price * many for _, price, many in basket.values()) + extra
        floor = total(sum(min(option[1] for option in pools[n]) * units(n) for n in pools))
        if budget is not None and floor > budget + 1e-9:
            return ""
        flat = []
        for n in sorted(pools):
            for _, _, ident, lines in pools[n]:
                flat.extend((n, ident, prop, value, says) for prop, value, says in lines)
        flat = flat[:12]
        self.near_done = True
        answer = run.wire.consult(self._near_prompt(flat), 600, CHECKER)
        agreed = self._near_agreed(answer)
        run.log.append({"kind": "near_check", "lines": [f"{i}:{p}={v}|{s}"[:120] for _, i, p, v, s in flat], "same": sorted(agreed)})
        for n in sorted(pools):
            for _, price, ident, lines in pools[n]:
                spots = [at for at, row in enumerate(flat) if row[0] == n and row[1] == ident]
                if len(spots) == len(lines) and all(at in agreed for at in spots):
                    basket[n] = (ident, price, units(n))
                    break
            if n not in basket:
                return ""
        if budget is not None and total() > budget + 1e-9:
            return ""
        picks = [(basket[n][0], basket[n][2]) for n in range(len(slots))]
        if run.moves_left() < len(picks) + 2:
            return ""
        version = str(needs.version)
        for n in pools:
            chosen_id = basket[n][0]
            stamp = f"{chosen_id}|{version}"
            for memo in [m for m in run.rulings if m == stamp or m.startswith(stamp + "|")]:
                run.rulings[memo] = ""
            run.passed.add(stamp)
            self.near_ok.add(stamp)
            run.pass_price[chosen_id] = run.archive.price(chosen_id)
        done = self._repair_put(picks)
        run.log.append({"kind": "near_basket", "picks": picks, "done": done, "cart": dict(run.trolley.units)})
        if not done:
            return ""
        self.near_cart = {"ver": needs.version, "refusals": run.refusal_n, "lines": [ident for ident, _ in picks]}
        kept = [(ident, prop, value, says) for n, ident, prop, value, says in flat if ident == basket.get(n, ("",))[0]]
        for ident, prop, value, says in kept:
            self.near_said.setdefault(f"{ident}|{version}", []).append((prop, value, says))
        said = "; ".join(f"{ident} writes {prop} as '{says}' for '{value[:60]}'" for ident, prop, value, says in kept)
        return ("Not run yet (no step used). A wording review judged that " + said + " - one value, said differently - and every "
                "listing involved satisfies each need the shopper revealed, has stock and fits the budget, so the code loaded the basket "
                "into the cart and checked its stock: " + run.trolley.summary() + " Order this cart now; decline only if a record "
                "you read shows one of these lines failing a requirement now in force.")

    def cart_moved(self, before: dict[str, float]) -> None:
        run = self.run
        self.cart_ver = (0 if self.cart_ver is None else self.cart_ver) + 1
        run.since_cart = 0
        now = run.trolley.units
        if any(line not in before for line in now):
            run.cart_seq = run.absorb_n
        for line, units in now.items():
            if line in before and units > before[line] + 1e-9:
                run.line_seq.pop(line, None)

    def after_put(self, call_args: dict[str, Any], obs: Any) -> None:
        run = self.run
        trolley = run.trolley
        if trolley.view_units(obs) is not None:
            self.cart_shown = 0 if self.cart_ver is None else self.cart_ver
        new = [line for line, count in trolley.lines(call_args) if (count is None or count > 0) and line in trolley.units
               and line not in self.first_puts]
        if not new or not trolley.is_put(call_args) or run.closer.code_fault(obs):
            return
        self.first_puts.update(new)
        stale = set(self.stale_lines())
        new = [line for line in new if line in stale]
        tool = self.stock_tool()
        if not new or not tool or run.wire.mute or run.covering:
            return
        if run.moves_left() < 4 or run.wire.spare() < CLOSING:
            return
        reads = run.reads
        queue = list(reads.put_queue or [])
        reads.put_queue = (queue + [line for line in new if line not in queue])[:4]
        reads.put_tool = tool
        run.log.append({"kind": "after_put_read", "lines": list(reads.put_queue), "tool": tool})

    def mark_fresh(self, obs: Any) -> None:
        run = self.run
        if not self.stock_seen:
            return
        seqs = run.line_seq
        watch = run.watch
        for d in Judge.dicts_in([obs]):
            ident = self.rec_id(d)
            if ident in run.trolley.units and watch.stock_state(d) is not None:
                seqs[ident] = run.absorb_n

    def cart_gone(self, lines: list[str]) -> set[str]:
        run = self.run
        gone = set(lines) & self.dead_lines()
        stale = [i for i in self.stale_lines() if i in lines and i not in gone]
        if stale and self.stock_tool() and run.moves_left() >= len(stale) + 2:
            _, sold = run.closer.code_recheck(stale)
            gone |= set(sold) | (set(lines) & self.dead_lines())
        return gone

    def _changing_tools(self) -> set:
        run = self.run
        busy = {run.role(r) for r in ("search", "message", "cart", "cart_view", "order", "decline")} - {None}
        spoken = {}
        for tdef in run.toolset or []:
            fn = tdef.get("function") if isinstance(tdef, dict) else None
            if not isinstance(fn, dict) or not isinstance(fn.get("name"), str):
                continue
            words = re.sub(r"([a-z])([A-Z])", r"\1 \2", fn["name"]).replace("_", " ")
            spoken[fn["name"]] = words + " " + str(fn.get("description") or "")
        pattern = r"\b(?:remov|delet|cancel|clear|discard|releas|refund|purchas|checkout|wish|basket|cart|bag)"
        for name in list(run.shapes):
            if re.search(pattern, spoken.get(name, name).lower()):
                busy.add(name)
        return busy

    @staticmethod
    def _list_argument(schema):
        props = schema.get("properties") if isinstance(schema, dict) else None
        if not isinstance(props, dict):
            props = {}
        arrays = []
        for key, spec in props.items():
            if not isinstance(spec, dict) or spec.get("type") != "array":
                continue
            inner = spec.get("items")
            if isinstance(inner, dict) and inner.get("type") == "string" and not inner.get("enum"):
                arrays.append(key)
        if len(arrays) != 1:
            return None
        required = schema.get("required") if isinstance(schema, dict) and isinstance(schema.get("required"), list) else []
        if any(r not in arrays for r in required):
            return None
        most = props[arrays[0]].get("maxItems")
        if not is_number(most) or most < 2:
            return None
        return arrays[0], int(min(most, 10))

    def _find_batch_spec(self):
        busy = self._changing_tools()
        for name, schema in self.run.shapes.items():
            if name in busy:
                continue
            found = self._list_argument(schema)
            if found:
                return name, found[0], found[1]
        return None

    def batch_tool(self) -> tuple[str, str, int] | None:
        run = self.run
        if not run.batch_spec_known:
            try:
                spec = self._find_batch_spec()
            except Exception:
                spec = None
            run.batch_spec = spec
            run.batch_spec_known = True
        return run.batch_spec

    @staticmethod
    def batch_ok(found: list[Any], ids: list[str]) -> bool:
        wanted = set(ids)
        got: set[str] = set()
        try:
            for d in Judge.dicts_in(found):
                values = list(d.values())
                if any(isinstance(v, dict) for v in values):
                    rich = any(isinstance(v, dict) and sum(1 for x in v.values() if isinstance(x, (str, int, float))) >= 2
                               for v in values)
                else:
                    rich = sum(1 for v in values if isinstance(v, str) and 0 < len(v) <= 70) >= 5
                if rich:
                    got.update(v for v in values if isinstance(v, str) and v in wanted)
        except Exception:
            return False
        return bool(got) and 2 * len(got) >= len(wanted)

    @staticmethod
    def title_of(row: dict[str, Any]) -> str:
        if isinstance(row.get("title"), str):
            return row["title"]
        for key, v in row.items():
            if isinstance(v, str) and re.search(r"title|heading|headline", str(key).lower()):
                return v
        return ""

    def _led_first(self, home, most, words, known) -> list[str]:
        run = self.run
        needs = run.needs
        several = bool(needs.items())
        slot = next((n for n, h in enumerate(self.home_shelves()) if h == home), None) if several else None
        values = []
        for req in needs.props_left():
            if several and run.talk.product_of(req.partition(":")[0]) not in (slot, None):
                continue
            values.append(re.sub(r"\((?:stated|revealed[^)]*)\)", "", req.partition(":")[2]))
        values = [v for v in values if v.strip() and positive_value(v)
                  and not re.search(r"as shown|same as|the page|sold by|redirect|refer", v.lower())]
        facts: dict[str, list[str]] = {}
        watch = run.watch
        usable = values and most > 0 and (not several or slot is not None)
        for d in Judge.dicts_in(run.archive.items if usable else []):
            ident = watch.row_mark(d)
            if ident in run.archive.read and watch.shelf_of(d) in home:
                for val in d.values():
                    if isinstance(val, dict):
                        facts.setdefault(ident, []).extend(str(x) for x in val.values() if isinstance(x, (str, int, float)))
        fit_read = any(all(value_seen(v, " | ".join(texts)) for v in values) for texts in facts.values())
        if not usable or fit_read:
            return []
        walker = run.walk
        led = sorted((w for w in run.page_book.values() if walker.walk_carries(w["args"], slot)),
                     key=lambda w: -len(walker.walk_carries(w["args"], slot)))
        for walk in led:
            first = next((i for i in (walk.get("row_shelf") or {}) if i in words), None)
            if first:
                known.append(words[first])
                return [first]
        return []

    def menu_rows(self, home: set[str], most: int, taken: list[str], qty: int = 1) -> list[str]:
        run = self.run
        archive = run.archive
        watch = run.watch
        budget = run.needs.spend_cap
        unread_order: list[str] = []
        title_by: dict[str, str] = {}
        sold: set[str] = set()
        seen_words: list[set[str]] = []
        for record in Judge.dicts_in(archive.items):
            ident = watch.row_mark(record)
            if ident not in archive.ids or watch.shelf_of(record) not in home:
                continue
            if ident in archive.read:
                seen_words.append(set(re.findall(r"[a-z0-9]{3,}", self.title_of(record).lower())))
                continue
            if watch.sold_out(record):
                sold.add(ident)
            if ident not in unread_order:
                unread_order.append(ident)
                title_by[ident] = self.title_of(record)
        scores = run.cover.hit_score
        ordered_hits = sorted(run.cover.all_hits, key=lambda i: -scores.get(i, 0))
        pool_order = [i for i in ordered_hits if i in title_by] + [i for i in unread_order if i not in ordered_hits]

        def allowed(ident: str) -> bool:
            if ident in sold or ident in taken or ident in run.refused_lines:
                return False
            return not (is_number(budget) and is_number(archive.price(ident)) and archive.price(ident) * qty > budget)
        pool_order = [i for i in pool_order if allowed(i)]
        title_words = {i: set(re.findall(r"[a-z0-9]{3,}", title_by.get(i, "").lower())) for i in pool_order}
        lead = run.cover.row_score
        try:
            chosen = self._led_first(home, most, title_words, seen_words)
        except Exception:
            chosen = []

        def rank(ident: str):
            mine = title_words[ident]
            overlap = max((len(mine & other) / max(1, len(mine | other)) for other in seen_words), default=0.0)
            tier = 0 if ident in ordered_hits else 1 if lead.get(ident) else 2
            return tier, overlap, pool_order.index(ident)
        while len(chosen) < most:
            window = [i for i in pool_order if i not in chosen][:16]
            if not window:
                break
            best = min(window, key=rank)
            chosen.append(best)
            seen_words.append(title_words[best])
        return chosen

    def _queue_menu(self, picks: list[str], say: str) -> None:
        reads = self.run.reads
        reads.open_queue = picks
        reads.open_recheck = True
        reads.open_say = say
        reads.open_purpose = "menu"

    def _rich_count(self, home) -> int:
        if not home:
            return 4
        return sum(self.run.archive.rich_reads(s) for s in home)

    def _second_round(self, homes, batch, qty_for) -> bool:
        self.pre_rounds = 2
        queues: list[list[str]] = []
        claimed: list[str] = []
        for k, home in enumerate(homes):
            have = self._rich_count(home)
            if have < 3:
                more = self.menu_rows(home, 4 - have, claimed + list(self.pre_picked or []), qty_for(k))
                claimed += more
                queues.append(more)
        order: list[str] = []
        while any(queues):
            for queue in queues:
                if queue:
                    order.append(queue.pop(0))
        order = order[: batch[2]]
        if not order:
            return False
        self._queue_menu(order, "A few more listings first: the earlier ones of some products carried only a handful of fields.")
        return True

    def _first_round(self, homes, batch, qty_for) -> bool:
        run = self.run
        bare: list[str] = []
        thin: list[tuple[int, int]] = []
        for k, home in enumerate(homes):
            have = self._rich_count(home)
            if have == 0:
                bare += self.menu_rows(home, 2 if run.needs.items() else 4, bare, qty_for(k))
            elif have < 4:
                thin.append((k, 4 - have))
        order = list(bare)
        for k, missing in thin:
            order += self.menu_rows(homes[k], missing, order, qty_for(k))
        rounds = 2 if len(bare) > batch[2] and run.moves_left() > 25 else 1
        order = order[: batch[2] * rounds]
        if not order:
            return False
        self.pre_read_done = True
        self.pre_rounds = 1
        self.pre_picked = list(order)
        self._queue_menu(order, "Opening a handful of listings per requested product so their fields are known before any question goes out.")
        return True

    def pre_read(self) -> bool:
        run = self.run
        try:
            batch = self.batch_tool()
            again = self.pre_rounds == 1
            if not batch or (self.pre_read_done and not again) or len(run.questions_log) > 1:
                return False
            if run.moves_left() <= (25 if again else 20) or run.wire.spare() < CLOSING + 60:
                return False
            homes = self.home_shelves()
            quantities = [row["quantity"] for row in run.needs.items()] or [run.needs.quantity()]

            def qty_for(k: int):
                return quantities[min(k, len(quantities) - 1)]
            if again:
                return self._second_round(homes, batch, qty_for)
            return self._first_round(homes, batch, qty_for)
        except Exception as err:
            run.log.append({"kind": "pre_read_error", "error": f"{type(err).__name__}: {err}"[:300]})
            return False

    def _covered_labels(self) -> set[str]:
        run = self.run
        version = str(run.needs.version)
        verified = set()
        for stamp in run.passed:
            ident, _, ver = stamp.rpartition("|")
            if ver == version:
                verified.add(ident)
        verified = verified - run.refused_lines - self.dead_lines()
        labels = {run.judge.shelf_known(i) for i in verified | set(run.judge.audit_fits or [])}
        labels.discard("")
        return labels

    def lacking_rows(self) -> list[str]:
        run = self.run
        archive = run.archive
        try:
            wanted = run.needs.items()
            if not wanted:
                return []
            homes = self.home_shelves()
            covered = self._covered_labels()
            short = [k for k, home in enumerate(homes) if home and not home & covered]
            if not short:
                return []
            batch = self.batch_tool()
            limit = batch[2] if batch else 3
            budget = run.needs.spend_cap
            sightings: list[tuple[str, str]] = []
            sold: set[str] = set()
            watch = run.watch
            for record in Judge.dicts_in(reversed(archive.items)):
                ident, label = watch.row_mark(record), watch.shelf_of(record)
                if ident in archive.ids and label:
                    sightings.append((ident, label))
                    if watch.sold_out(record):
                        sold.add(ident)
            label_of: dict[str, str] = {}
            for ident, label in sightings:
                label_of.setdefault(ident, label)
            scores = run.cover.hit_score
            ordered_hits = sorted(run.cover.all_hits, key=lambda i: -scores.get(i, 0))
            queues: dict[int, list[str]] = {k: [] for k in short}
            candidates = [i for i in ordered_hits if i in label_of] + [i for i, _ in sightings]
            for ident in dict.fromkeys(candidates):
                if ident in archive.read or ident in run.refused_lines or ident in sold:
                    continue
                home_slot = next((k for k in short if label_of.get(ident) in homes[k]), None)
                if home_slot is None:
                    continue
                price = archive.price(ident)
                too_dear = is_number(budget) and is_number(price) and price * wanted[home_slot]["quantity"] > budget
                if not too_dear:
                    queues[home_slot].append(ident)
            chosen: list[str] = []
            while len(chosen) < limit and any(queues.values()):
                for k in short:
                    if queues[k] and len(chosen) < limit:
                        chosen.append(queues[k].pop(0))
            return chosen
        except Exception as err:
            run.log.append({"kind": "lacking_error", "error": f"{type(err).__name__}: {err}"[:300]})
            return []

class Dialogue:
    def __init__(self, run):
        self.run = run
        self.ask_facts = {}
        self.asked_at_sets = None
        self.asked_at_tick = 0
        self.asking_label = None
        self.empty_plans = 0
        self.no_pairs = None
        self.open_pair = []
        self.pending_seconds = []
        self.pending_text = {}
        self.plan_wait = None
        self.product_plans = 0
        self.quiet_asks = []
        self.quota_reads = {}
        self.quota_wait = False
        self.reasked = set()
        self.spent_asks = set()
        self.wanted_asks = None

    def text_arg(self, the_tool: str) -> str | None:
        schema = self.run.shapes.get(the_tool or "") or {}
        props = schema.get("properties") or {}
        required = list(schema.get("required") or [])
        texts = []
        for name, spec in props.items():
            if not isinstance(spec, dict) or "enum" in spec:
                continue
            if spec.get("type") == "string" or "string" in (spec.get("type") or []):
                texts.append(name)
        slot = next((name for name in required if name in texts), None)
        if not slot and len(texts) == 1:
            slot = texts[0]
        if not slot or any(name != slot for name in required):
            return None
        return slot

    def message_call(self, text_body: str, cid: str) -> dict[str, Any] | None:
        run = self.run
        tool = run.role("message")
        slot = self.text_arg(tool) if tool else None
        if not slot:
            return None
        args, faults = ArgCheck.execute({slot: text_body}, run.shapes[tool])
        if faults:
            return None
        return {"id": cid, "type": "function", "function": {"name": tool, "arguments": json.dumps(args)}}

    def product_of(self, text: str) -> int | None:
        rows = self.run.needs.items()
        if not rows:
            return None
        kinds = [set(kind_tokens(row["product"])) for row in rows]
        pieces = re.findall(r"\(([^)]*)\)", str(text))[:1] + [str(text)]
        for piece in pieces:
            words = set(kind_tokens(piece))
            scores = []
            for pos, own in enumerate(kinds):
                shared = set().union(*(kinds[other] for other in range(len(kinds)) if other != pos))
                scores.append(kind_score(own, words, shared))
            top = max(scores)
            if top > 0 and scores.count(top) == 1:
                return scores.index(top)
        return None

    def plan_product(self, qitem: dict[str, Any]) -> int | None:
        slot = self.product_of(str(qitem.get("property") or ""))
        if slot is None:
            slot = self.product_of(str(qitem.get("question") or ""))
        return slot

    def _tally_asked(self, count: int):
        answered = [0] * count
        sent = [0] * count
        for line in self.run.questions_log:
            head = line.split(":", 1)[0]
            slot = self.product_of(head)
            if slot is None:
                quoted = re.search(r'asked "(.*?)" -> ', line)
                slot = self.product_of(quoted.group(1)) if quoted else None
            if slot is None:
                continue
            sent[slot] += 1
            if "[not answered]" not in head:
                answered[slot] += 1
        return answered, sent

    def _hinted_products(self, count: int, answered: list[int]) -> set:
        intake = self.run.intake
        try:
            intake.open_hints()
        except Exception:
            pass
        try:
            named, keyless = set(), [0] * count
            for hint in intake.open_hints():
                slot = self.product_of(str(hint).split(":", 1)[0])
                if slot is None:
                    continue
                if intake.plain_hint(slot, str(hint).split(":", 1)[0].strip(), hinted=True):
                    keyless[slot] += 1
                else:
                    named.add(slot)
            return {j for j in range(count) if j in named or keyless[j] > answered[j]}
        except Exception:
            return set()

    def _second_alone(self, scripted, due, rows):
        for qitem in scripted:
            second = str(qitem.get("second") or "").strip()
            slot = self.product_of(second) if second else None
            if slot not in due:
                continue
            tied = re.search(r"\(([^)]+)\)", second)
            product = tied.group(1).strip() if tied else rows[slot]["product"]
            return {"property": second, "question": f"Which {second.split('(')[0].strip()} would you like for the {product}?"}
        return None

    def _levels(self, count, answered, sent, revealed, hinted, alt_left) -> dict[int, tuple]:
        never_answered = [j for j in range(count) if not answered[j]]
        table = {}
        for k in range(count):
            waits = 1 if revealed[k] and any(j != k for j in never_answered) else 0
            silent = 0 if k in alt_left else sent[k] - answered[k]
            table[k] = (waits, 0 if k in hinted else 1, answered[k] + revealed[k], silent, revealed[k])
        return table

    def _quota_for(self, due, scripted, hinted, answered, ok):
        intake = self.run.intake
        for k in sorted(due, key=lambda k: (k not in hinted, k)):
            try:
                pick = self.quota_pick(k, answered[k] == 0, ok)
            except Exception:
                pick = None
            if pick and ok is not None:
                if not ok(pick.get("property")) or intake.asked_for(k, pick.get("property")):
                    pick = None
            if pick:
                return [pick] + scripted
            if self.quota_wait:
                return scripted
        return None

    def quota_first(self, scripted: list[dict[str, Any]], ok: Any = None) -> list[dict[str, Any]]:
        run = self.run
        rows = run.needs.items()
        if not rows:
            return scripted
        count = len(rows)
        answered, sent = self._tally_asked(count)
        if max(answered) == 0 and scripted and self.plan_product(scripted[0]) is None:
            return scripted
        revealed = [0] * count
        for need in run.needs.uncovered:
            slot = self.product_of(str(need).split(":", 1)[0])
            if slot is not None:
                revealed[slot] += 1
        hinted = self._hinted_products(count, answered)
        alt_left = {j for j in range(count) if sent[j] > answered[j] and run.intake.alt_untried(j)}
        table = self._levels(count, answered, sent, revealed, hinted, alt_left)
        for level in sorted(set(table.values())):
            due = [k for k in range(count) if table[k] == level]
            for n, qitem in enumerate(scripted):
                if self.plan_product(qitem) in due:
                    return [qitem] + scripted[:n] + scripted[n + 1:]
            alone = self._second_alone(scripted, due, rows)
            if alone:
                return [alone] + scripted
            ordered = self._quota_for(due, scripted, hinted, answered, ok)
            if ordered is not None:
                return ordered
        return scripted

    def _quota_open_rows(self, k: int, home, product: str) -> None:
        run = self.run
        shelf = run.shelf
        batch = shelf.batch_tool()
        unread = shelf.menu_rows(home, min(4, batch[2]) if batch else 2, [], run.needs.items()[k]["quantity"]) if home else []
        if not unread or not (run.role("details") or batch) or self.quota_reads.get(k) or run.moves_left() <= 12:
            return
        self.quota_reads[k] = True
        reads = run.reads
        reads.open_queue = unread
        reads.open_recheck = True
        reads.open_say = f"Opening some {product} listings to see their fields; the question about it comes after."
        reads.open_purpose = "quota"
        self.quota_wait = True

    def _read_home(self, k: int, home) -> list[str]:
        archive = self.run.archive
        labels = [label for label in archive.shelves_read() if label.strip().lower() in home]
        if labels or not home or not self.quota_reads.get(k):
            return labels
        return [label for label in sorted(home) if archive.prop_menu(label)[0]][:1]

    def _menu_question(self, k, product, menu, fallback, good):
        intake = self.run.intake
        try:
            hints = intake.open_hints(k)
        except Exception:
            hints = []
        for hint in hints:
            prop = re.split(r"\s*\(", hint.split(":", 1)[0])[0].strip()
            if not prop or junk_key(prop) or intake.plain_hint(k, prop, hinted=True):
                continue
            if good(f"{prop} ({product})"):
                return prop
        if not fallback:
            return None
        for weigh in (False, True):
            for entry in menu.splitlines():
                prop = entry.strip("- ").split(" (in ")[0].strip()
                if prop and self.open_prop(k, prop, entry, by_samples=weigh) and good(f"{prop} ({product})"):
                    return prop
        return None

    def quota_pick(self, k: int, fallback: bool = True, ok: Any = None) -> dict[str, Any] | None:
        run = self.run
        archive = run.archive
        intake = run.intake
        product = run.needs.items()[k]["product"]
        homes = run.shelf.home_shelves()
        home = homes[k] if k < len(homes) else set()
        labels = self._read_home(k, home)
        if not labels:
            self._quota_open_rows(k, home, product)
            return None
        menu = intake.menu_trim(archive.prop_menu(labels[0])[0], archive.rich_reads(labels[0]))

        def good(label: Any) -> bool:
            return ok is None or (bool(ok(label)) and not intake.asked_for(k, label))
        planned = self.product_plan(k, menu)
        if planned and good(planned.get("property")):
            return planned
        prop = self._menu_question(k, product, menu, fallback, good)
        if prop is None:
            return None
        return {"property": f"{prop} ({product})", "question": f"Which {prop} would you like for the {product}?"}

    def product_plan(self, k: int, menu: str) -> dict[str, Any] | None:
        try:
            return self.product_plan_call(k, menu)
        except Exception:
            return None

    def product_plan_call(self, k: int, menu: str) -> dict[str, Any] | None:
        run = self.run
        intake = run.intake
        if not menu or self.product_plans >= 1 or run.wire.spare() < CLOSING + 40:
            return None
        self.product_plans += 1
        product = run.needs.items()[k]["product"]
        remaining = max(1, run.ask_limit() - run.msgs_sent)
        prompt = ASK_PLAN.replace("QUESTION_COUNT", str(min(3, remaining))) + "\n\nTHE SHOPPER'S REQUEST:\n" + run.request
        prompt += "\n\nREQUIREMENTS SO FAR:\n" + run.needs.text_body()
        if run.questions_log:
            prompt += "\n\nALREADY ASKED (with the shopper's reply):\n" + "\n".join(run.questions_log)
        prompt += (f"\n\nTHIS ROUND EVERY QUESTION CONCERNS THE {product.upper()}: all the other products were asked about "
                   f"already, this one never. Choose only {product} properties, taken from its list below, "
                   "beginning with whichever the story's details suggest.")
        try:
            hints = intake.open_hints(k)
        except Exception:
            hints = []
        if hints:
            prompt += ("\n\nSTORY HINTS (bits of the request's story, each with the property it probably points at, most likely "
                       "first; none of them is a requirement):\n" + "\n".join(f"- {intake.hint_mark(h)}" for h in hints[:6]))
        prompt += self.plan_notes(k)
        prompt += (f"\n\nPROPERTIES IN THE DETAILS OF THE {product.upper()} LISTINGS SEEN SO FAR (name, number of listings, "
                   "example values):\n" + menu)
        answer = run.wire.consult(prompt, 1500, PLANNER[:1])
        run.log.append({"kind": "product_plan", "product": product, "result": answer})
        return self._plan_choice(k, product, answer)

    def _plan_choice(self, k, product, answer):
        intake = self.run.intake
        words = kind_words(product)
        for qitem in (answer or {}).get("questions") or []:
            if not isinstance(qitem, dict):
                continue
            first = str(qitem.get("property") or "").split("(")[0].strip()
            second = str(qitem.get("second") or "").split("(")[0].strip()
            if re.search(r"none|^-$|n/a", second.lower()):
                second = ""
            question = str(qitem.get("question") or "").strip()
            if intake.plain_hint(k, first):
                first = ""
            if second and intake.plain_hint(k, second):
                second = ""
            if first and intake.plan_open(k, first):
                named = bool(question) and bool(words & kind_words(question))
                pair = ""
                if named and second and not strict_prop(first, second) and intake.plan_open(k, second):
                    pair = second
                return {"property": f"{first} ({product})", "second": f"{pair} ({product})" if pair else "",
                        "question": question if named else f"Which {first} would you like for the {product}?"}
            if second and intake.plan_open(k, second):
                return {"property": f"{second} ({product})", "question": f"Which {second} would you like for the {product}?"}
        return None

    def _settled_by(self, req: str, name: str, slot, samples: str, by_samples: bool) -> bool:
        intake = self.run.intake
        head, _, value = str(req).partition(":")
        if strict_prop(head, name):
            return bool(intake.settles(str(req), samples if by_samples else ""))
        if same_prop(head, name):
            if not positive_value(value, head):
                return False
            return bool(value_seen(value, samples or " | ".join(intake.field_values(slot, name))))
        tokens = [t for t in re.findall(r"[a-z]{3,}", value.lower()) if t not in LINK_WORDS]
        if not samples or not tokens or re.search(r"\d", value) or not positive_value(value, head):
            return False
        return all(re.search(r"\b" + re.escape(t), samples) for t in tokens)

    def open_prop(self, k: int | None, prop: str, line: str = "", by_samples: bool = False) -> bool:
        run = self.run
        intake = run.intake
        try:
            name = str(prop).split("(")[0].strip()
            if not name or junk_key(name):
                return False
            several = bool(run.needs.items())

            def concerns(text: str, tagged_only: bool = False) -> bool:
                if k is None or not several:
                    return True
                slot = self.product_of(text)
                if slot == k:
                    return True
                return slot is None and not tagged_only
            for logged in run.questions_log:
                head = logged.split(":", 1)[0].strip("- ")
                if "[not answered]" in head:
                    continue
                for part in head.split(";"):
                    if concerns(part) and intake.same_field(k, part, name):
                        return False
            samples = line.split("):", 1)[1].lower() if "):" in line else ""
            for req in run.needs.props_left():
                if concerns(str(req).partition(":")[0]) and self._settled_by(req, name, k, samples, by_samples):
                    return False
            return not any(concerns(str(p), tagged_only=True) and intake.same_field(k, str(p), name)
                           for p in run.needs.indifferent)
        except Exception:
            return True

    def plan_notes(self, k: int | None) -> str:
        run = self.run
        try:
            several = bool(run.needs.items())
            asked = {p.strip().lower() for line in run.questions_log for p in line.split(":", 1)[0].strip("- ").split(";")}
            opens = []
            for prop in self.open_pair or []:
                slot = self.product_of(prop) if several else None
                if k is not None and several and slot not in (k, None):
                    continue
                if prop.lower() in asked or not run.intake.plan_open(slot, prop):
                    continue
                opens.append(prop)
            wanted = [w for w in self.wanted_asks or [] if k is None or not several or self.product_of(w) in (k, None)]
            text = ""
            if opens:
                text += ("\n\nOPEN FROM LAST PAIR (second halves of planned pairs that got no answer, either because the shopper "
                         "replied to the first only or only the first went out; they remain open, and each was the planner's "
                         "next choice):\n" + "\n".join(f"- {p}" for p in opens[-3:]))
            if wanted:
                text += ("\n\nQUESTIONS THE ASSISTANT WANTED TO ASK (written after reading listing records; when one names a property "
                         "still open that is also a key below, that key goes first, asked by its own name):\n"
                         + "\n".join(f"- {w}" for w in wanted[-3:]))
            return text
        except Exception:
            return ""

    def want_ask(self, raw_args: Any) -> None:
        try:
            if isinstance(raw_args, dict):
                args = raw_args
            elif isinstance(raw_args, str) and raw_args.strip():
                args = json_object(raw_args)
            else:
                args = {}
            text = " ".join(str(v).strip() for v in (args or {}).values() if isinstance(v, str) and v.strip())[:300]
            if not text:
                return
            kept = [w for w in self.wanted_asks or [] if w.lower() != text.lower()]
            self.wanted_asks = (kept + [text])[-3:]
        except Exception:
            pass

    def drop_wanted(self, label: str) -> None:
        stem = self.run.watch.stem
        try:
            props = [re.split(r"\s*\(", p)[0] for p in re.split(r";|\band\b", str(label).replace("[not answered]", ""))]
            kept = []
            for text in self.wanted_asks or []:
                words = {stem(w) for w in re.findall(r"[a-z]+", text.lower())}
                covered = any(prop_words(p) and all(w in words or stem(w) in words for w in prop_words(p)) for p in props)
                if not covered:
                    kept.append(text)
            self.wanted_asks = kept
        except Exception:
            pass

    @staticmethod
    def _singular(word: str) -> str:
        return word[:-1] if len(word) > 3 and word.endswith("s") else word

    def _own_words(self, prop: str) -> set[str]:
        needs = self.run.needs
        mine = set(prop_words(prop)) | {"which", "would", "you", "like"}
        for row in needs.items() or [{"product": str(needs.reading.get("product") or "")}]:
            for token in re.findall(r"[a-z0-9]+", str(row.get("product") or "").lower()):
                mine.add(token)
                mine.add(self._singular(token))
        return mine

    def names_other(self, question: str, prop: str, menu: str, second: str = "") -> bool:
        try:
            spoken = " ".join(re.findall(r"[a-z0-9]+", str(question).lower()))
            heard = set(spoken.split())
            for token in list(heard):
                if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
                    heard.add(token[:-1])
            mine = self._own_words(prop)

            def mentions(field: str) -> bool:
                core = [t for t in prop_words(field) if len(t) >= 3 and t not in LINK_WORDS]
                if not core or not all(t in heard for t in core):
                    return False
                return any(t not in mine for t in core)
            if second and not strict_prop(second, prop) and mentions(second):
                return True
            for entry in str(menu or "").splitlines():
                field = entry.strip().lstrip("-").strip().split(" (in ")[0].strip()
                if not field or field.endswith(":") or strict_prop(field, prop) or junk_key(field):
                    continue
                if mentions(field):
                    return True
                examples = entry.split("):", 1)[1] if "):" in entry else ""
                for example in examples.split("|"):
                    tokens = re.findall(r"[a-z0-9]+", example.lower())
                    foreign = [t for t in tokens if t not in mine and t not in LINK_WORDS]
                    if not foreign or len("".join(tokens)) < 3:
                        continue
                    pattern = r"(?<![a-z0-9])" + re.escape(" ".join(tokens)) + r"(?![a-z0-9])"
                    if re.search(pattern, spoken):
                        return True
            return False
        except Exception:
            return False

    def queue_second(self, prop: str) -> None:
        queue = self.pending_seconds
        if prop and prop.lower() not in [p.lower() for p in queue]:
            queue.append(prop)

    def _product_tag(self, label: str, fallback: Any = None) -> str:
        tied = re.search(r"\(([^)]+)\)", label)
        if tied:
            return tied.group(1).strip()
        if fallback is not None:
            return fallback
        return str(self.run.needs.reading.get("product") or "item")

    @staticmethod
    def _plain_ask(prop: str, product: str) -> str:
        return f"Which {prop.split('(')[0].strip()} would you like for the {product}?"

    def _too_early(self, sets: int, pressing: bool, urge: bool, ceiling: int) -> bool | None:
        run = self.run
        tick = run.wire.tick
        thin = sets < ASK_AFTER_SETS and not pressing and not urge
        if thin and (tick < ASK_AFTER_STEPS or (tick < ASK_AFTER_STEPS + ASK_GRACE_STEPS and run.moves_left() > 20)):
            return True
        if run.moves_left() <= max(10, run.intake.order_due()) or run.wire.spare() < CLOSING + 40:
            run.asking = False
            return True
        if self.plan_wait is not None and sets <= self.plan_wait and not urge:
            return True
        last = self.asked_at_sets
        if last is None or sets > last or urge or pressing:
            return False
        if run.moves_left() <= run.intake.order_due() + 2 * max(0, ceiling - run.msgs_sent) + 14:
            return False
        return run.wire.tick - self.asked_at_tick < 4 and not run.needs.items()

    def _category_menus(self, prop_list: str, shown: str):
        run = self.run
        archive = run.archive
        intake = run.intake
        labels = archive.shelves_read()
        homes = run.shelf.home_shelves()
        all_home = set().union(*homes)
        mine = [label for label in labels if label.strip().lower() in all_home]
        order = (mine or labels) if all(homes) else mine + [label for label in labels if label not in mine]
        menus = [(label, archive.prop_menu(label)[0]) for label in order[:4]]
        listed = [(label, menu) for label, menu in menus if menu]
        whole = "\n".join(f"CATEGORY {label}:\n{menu}" for label, menu in listed) or prop_list
        trimmed = "\n".join(f"CATEGORY {label}:\n{intake.menu_trim(menu, archive.rich_reads(label))}"
                             for label, menu in listed) or shown
        return whole, trimmed

    def _plan_prompt(self, remaining: int, prop_list: str, sets: int):
        run = self.run
        intake = run.intake
        needs = run.needs
        parts = [ASK_PLAN.replace("QUESTION_COUNT", str(remaining)), NEXT_ONE.replace("REMAINING", str(remaining)),
                 "\n\nTHE SHOPPER'S REQUEST:\n" + run.request, "\n\nREQUIREMENTS SO FAR:\n" + needs.text_body()]
        if run.questions_log:
            parts.append("\n\nALREADY ASKED (with the shopper's reply):\n" + "\n".join(run.questions_log))
        must = needs.reading.get("ask_first")
        if must:
            parts.append("\n\nMUST BE LEARNED FROM THE SHOPPER: " + "; ".join(str(h) for h in must or []))
        parts.append(self.plan_notes(None))
        try:
            hints = intake.open_hints()
        except Exception:
            hints = []
        if hints:
            parts.append("\n\nSTORY HINTS (bits of the request's story, each with the property it probably points at; none is a "
                         "requirement, yet the hidden need tends to be one of them):\n"
                         + "\n".join(f"- {intake.hint_mark(h)}" for h in hints[:8]))
        shown = intake.menu_trim(prop_list, sets)
        if needs.items():
            parts.append("\n\nTHE PRODUCTS ASKED FOR: " + " | ".join(f"{n + 1}) {row['product']} x{row['quantity']}"
                                                                    for n, row in enumerate(needs.items())))
            prop_list, shown = self._category_menus(prop_list, shown)
        parts.append("\n\nPROPERTIES IN THE LISTINGS' DETAILS SEEN SO FAR (name, number of listings, example values):\n"
                     + (shown or "(none seen yet)"))
        return "".join(parts), prop_list

    def _vet_plan(self, answer, fresh) -> list[dict[str, Any]]:
        run = self.run
        several = bool(run.needs.items())
        kept = []
        for qitem in (answer or {}).get("questions") or []:
            if not isinstance(qitem, dict) or not str(qitem.get("question") or "").strip():
                continue
            second = str(qitem.get("second") or "").strip()
            if re.search(r"none|^-$|n/a", second.lower()):
                second = ""
            if fresh(qitem.get("property")):
                kept.append(dict(qitem, second=second if second and fresh(second) else ""))
                if second and run.intake.plain_hint(self.product_of(second) if several else None, second):
                    first = str(qitem.get("property") or "").strip()
                    kept[-1]["question"] = self._plain_ask(first, self._product_tag(first))
            elif second and fresh(second):
                kept.append({"property": second, "question": self._plain_ask(second, self._product_tag(second))})
        return kept

    def _named_product_question(self, thing):
        prop = str(thing.get("property") or "")
        product = self._product_tag(prop, "")
        words = re.findall(r"[a-z]{3,}", product.lower())
        if not product or not words:
            return thing
        asked = str(thing.get("question") or "").lower()
        if any(w.rstrip("s") in asked for w in words):
            return thing
        return dict(thing, question=self._plain_ask(prop, product))

    def _with_second(self, label, second):
        # one property to a message: a pair halves the second reveal and is often refused, so the runner-up stays open
        run = self.run
        single = self._plain_ask(label[0], self._product_tag(label[0]))
        return self.message_call(single, f"ask_{run.msgs_sent + 1}_{run.wire.tick}"), (label[0], single), second

    def _propose(self, scripted, pending, prop_list):
        run = self.run
        needs = run.needs
        proposal, label, split_second = None, None, ""
        strict_start = needs.reading.get("ask_first") and not run.questions_log
        for thing in scripted[:2]:
            if needs.items():
                thing = self._named_product_question(thing)
            worded = str(thing.get("question") or "")
            if not pending and not strict_start and (
                    not (needs.reading.get("ask_first") or worded.lower().startswith("which "))
                    or self.names_other(worded, str(thing.get("property") or ""), prop_list, str(thing.get("second") or ""))):
                prop = str(thing.get("property") or "")
                thing = dict(thing, question=self._plain_ask(prop, self._product_tag(prop)))
            proposal = self.message_call(str(thing.get("question")).strip(), f"ask_{run.msgs_sent + 1}_{run.wire.tick}")
            if not proposal:
                continue
            label = (str(thing.get("property")), str(thing.get("question")).strip())
            second = str(thing.get("second") or "").strip()
            if not second or second.lower() == label[0].lower() or re.search(r"none|^-$|n/a", second.lower()):
                break
            proposal, candidate, split = self._with_second(label, second)
            if not proposal:
                continue
            label, split_second = candidate, split
            break
        return proposal, label, split_second

    def _menu_basis(self):
        archive = self.run.archive
        whole_menu, whole_sets = archive.prop_menu()[0], archive.rich_reads()
        own = self.run.shelf.own_shelf() or self._lead_shelf()
        if not own:
            return whole_menu, whole_sets
        return archive.prop_menu(own)[0], archive.rich_reads(own)

    def _planned_answer(self, prompt, pending, sets):
        if pending:
            text = (self.pending_text or {}).pop(pending.lower(), "")
            question = text or self._plain_ask(pending, self._product_tag(pending))
            return {"questions": [{"property": pending, "question": question}]}
        if OUR_ASK:
            ours = self.our_plan()
            if ours is False:
                return False
            if ours.get("questions"):
                return ours
        wire = self.run.wire
        answer = wire.consult(prompt, 2500, PLANNER)
        if answer is None and wire.misses > 0:
            self.plan_wait = sets
            return False
        return answer

    def our_plan(self):
        run = self.run
        needs = run.needs
        rows = needs.items()
        names = [str(row["product"]) for row in rows] or [str(needs.reading.get("product") or "item")]
        basket = len(names) > 1
        room = run.moves_left() > run.intake.order_due() + 12
        menus = []
        for slot, name in enumerate(names):
            facts = self._own_facts(slot, basket, room)
            if not basket and not facts and room:
                # no listing of the product's own kind read yet: questions built from other shelves name no wish of it
                run.log.append({"kind": "our_plan", "wait": "no home shelf yet"})
                return False
            self.ask_facts[slot if basket else None] = facts
            settled = set()
            for line in needs.written:
                head = line.partition(":")[0]
                if not basket or self.product_of(head) == slot:
                    settled.add(ask_name(head.split("(")[0]))
            menus.append(ask_menu(facts, settled, basket)[:ASK_MENU])
        if basket and not all(menus):
            # a product with nothing read yet: wait for its listings while steps allow, else the planner takes the turn
            empty = [names[slot] for slot, menu in enumerate(menus) if not menu]
            run.log.append({"kind": "our_plan", "empty": empty, "wait": room})
            return False if room else {}
        ranks = self.ask_ranks(names, menus)
        queues = []
        for slot, menu in enumerate(menus):
            queues.append(ask_merge(ask_picks(ranks.get(f"P{slot + 1}")), [key for key, _, _ in menu]))
        # the product asked least so far comes first in every round, whether or not its questions were answered
        sent = self._tally_asked(len(names))[1] if basket else [0]
        turns = sorted(range(len(queues)), key=lambda slot: (sent[slot] if slot < len(sent) else 0, slot))
        questions = []
        for spot in range(max((len(queue) for queue in queues), default=0)):
            for slot in turns:
                queue = queues[slot]
                if spot >= len(queue):
                    continue
                key = queue[spot]
                samples = next((found for listed, _, found in menus[slot] if listed == key), [])
                label = f"{ask_name(key)} ({names[slot]})" if basket else ask_name(key)
                questions.append({"property": label, "question": self.our_question(ask_name(key), names[slot], samples)})
        run.log.append({"kind": "our_plan", "menus": [[key for key, _, _ in menu][:8] for menu in menus], "first": [q["property"] for q in questions[:4]]})
        return {"questions": questions}

    def _own_facts(self, slot, basket, room):
        run = self.run
        homes = run.shelf.home_shelves()
        labels = sorted(homes[slot]) if slot < len(homes) else []
        if not labels and not basket:
            lead = self._lead_shelf(settle=not room)
            labels = [lead] if lead else []
        return [found for label in labels for found in run.archive.fact_sets(label)]

    def _lead_shelf(self, settle: bool = False) -> str:
        # no label answers to the requested noun: the shelf most rows sit on stands in, once it holds a majority or time runs out
        run = self.run
        archive = run.archive
        try:
            if run.needs.items() or run.shelf.home_shelves()[0]:
                return ""
            counts: dict[str, int] = {}
            for record in Judge.dicts_in(archive.items):
                label = run.watch.shelf_of(record)
                if label and any(isinstance(v, str) and v in archive.ids and id_field(key) for key, v in record.items()):
                    counts[label] = counts.get(label, 0) + 1
            top = max(sorted(counts), key=counts.get, default="")
            return top if top and (settle or counts[top] * 2 > sum(counts.values())) else ""
        except Exception:
            return ""

    def _rank_prompt(self, names, menus):
        needs = self.run.needs
        blocks = []
        for slot, (name, menu) in enumerate(zip(names, menus)):
            # alphabetical and without counts: the order of the list must not pass for an opinion
            lines = [f"- {ask_name(key)}: {' | '.join(samples)}" for key, _, samples in sorted(menu, key=lambda row: ask_name(row[0]))]
            blocks.append(f"## P{slot + 1}: {name}\n" + "\n".join(lines))
        parts = [ASK_RANK, "# Request\n" + self.run.request]
        known = needs.written + needs.uncovered
        if known:
            parts.append("# Settled so far\n" + "\n".join(f"- {line}" for line in known))
        hints = needs.reading.get("hints") or []
        if hints:
            parts.append("# Notes from a first reading of the story\n" + "\n".join(f"- {hint}" for hint in hints[:8]))
        if self.run.questions_log:
            parts.append("# Already asked, with the replies\n" + "\n".join(self.run.questions_log))
        parts.append("# Products\n" + "\n\n".join(blocks))
        return "\n\n".join(parts)

    def ask_ranks(self, names, menus):
        mark = tuple(tuple(sorted(key for key, _, _ in menu[:12])) for menu in menus)
        memo = getattr(self, "rank_memo", None)
        if memo and (memo[0] == mark or memo[2] >= ASK_RANKINGS) or not any(menus):
            return memo[1] if memo else {}
        answer = self.run.wire.consult(self._rank_prompt(names, menus), 1500, ASK_SEAT) or {}
        # a failed call keeps no mark, so the next turn may try again while the allowance lasts
        self.rank_memo = (mark if answer else None, answer or (memo[1] if memo else {}), (memo[2] if memo else 0) + 1)
        return self.rank_memo[1]

    @staticmethod
    def our_question(prop: str, product: str, samples: list[str]) -> str:
        bare = Dialogue._plain_ask(prop, product)
        if not ask_opaque(prop):
            return bare
        # a name with no real word in it says nothing to a shopper: only then do two values show what is meant
        shown = [value for value in samples if len(value) <= 24][:2]
        worded = f"{bare[:-1]}, such as {' or '.join(shown)}?"
        return worded if shown and len(worded) <= ASK_CHARS else bare

    def quiet_twin(self, slot, prop) -> bool:
        intake = self.run.intake
        try:
            name = re.split(r"\s*\(", str(prop))[0].strip()
            if intake.quiet_on(slot, name):
                return True
            facts = self.ask_facts.get(slot) or []
            keys = {ask_name(key): key for found in facts for key in found}
            own = keys.get(ask_name(name))
            for asked in self.quiet_asks:
                other = keys.get(ask_name(re.split(r"\s*\(", asked)[0]))
                if slot is not None and self.product_of(asked) not in (slot, None):
                    continue
                if own and other and ask_twins(facts, own, other):
                    return True
            return False
        except Exception:
            return False

    def _no_proposal(self, sets: int, urge: bool) -> bool:
        run = self.run
        if sets < ASK_AFTER_SETS and run.moves_left() > 12 and not urge:
            self.plan_wait = sets
            return False
        self.empty_plans += 1
        if run.moves_left() > 12 and self.empty_plans <= 1:
            self.plan_wait = sets
        else:
            run.asking = False
        return False

    def _send_question(self, proposal, label):
        run = self.run
        needs = run.needs
        run.chat.append({"role": "assistant", "content": f"Question for the shopper next: {label[0]}.", "tool_calls": [proposal]})
        counts = len(needs.uncovered), len(needs.indifferent) + needs.pinned_heard
        self.asking_label = label[0]
        try:
            replies = run.dispatch([proposal], scripted=True)
        finally:
            self.asking_label = ""
        self.post_replies(replies)
        return replies, counts

    def pose_planned(self, urge: bool = False) -> bool:
        run = self.run
        needs = run.needs
        intake = run.intake
        ceiling = run.ask_limit()
        if not run.role("message") or run.msgs_sent >= ceiling:
            run.asking = False
            return False
        prop_list, sets = self._menu_basis()
        # a question that names no listed property draws no hidden need; only the budget is worth asking before any read
        pressing = bool(needs.reading.get("ask_first")) and not run.questions_log and not is_number(needs.spend_cap)
        if self._too_early(sets, pressing, urge, ceiling):
            return False
        if run.shelf.pre_read():
            return True
        nothing_asked = not run.questions_log and not run.msgs_sent
        if nothing_asked and not pressing and not urge and intake.first_reads():
            return True
        remaining = ceiling - run.msgs_sent
        prompt, prop_list = self._plan_prompt(remaining, prop_list, sets)
        asked_props = set()
        for logged in run.questions_log:
            asked_props.update(part.strip().lower() for part in logged.split(":", 1)[0].strip("- ").split(";"))

        def fresh(prop: Any) -> bool:
            prop = str(prop or "").strip()
            lowered = prop.lower()
            if not prop or lowered in asked_props or junk_key(prop) or lowered in self.spent_asks:
                return False
            slot = self.product_of(prop) if needs.items() else None
            return not intake.plain_hint(slot, prop) and intake.plan_open(slot, prop) and not self.quiet_twin(slot, prop)
        self.pending_seconds = [p for p in self.pending_seconds or [] if fresh(p)]
        pending = self.pending_seconds[0] if self.pending_seconds else ""
        answer = self._planned_answer(prompt, pending, sets)
        if answer is False:
            return False
        scripted = self._vet_plan(answer, fresh)
        if needs.items() and not pending:
            scripted = self.quota_first(scripted, fresh)
            waited, self.quota_wait = self.quota_wait, False
            if waited:
                return True
        proposal, label, split_second = self._propose(scripted, pending, prop_list)
        if not proposal:
            return self._no_proposal(sets, urge)
        self.asked_at_sets, self.asked_at_tick = sets, run.wire.tick
        replies, (found_before, freed_before) = self._send_question(proposal, label)
        asked_label = label[0]
        if pending and asked_label.split(";")[0].strip().lower() == pending.lower():
            self.pending_seconds = [p for p in self.pending_seconds or [] if p.lower() != pending.lower()]
        heard_meta, needs.meta_last = needs.meta_last, None
        label, note = self._read_reply(label, heard_meta, found_before, freed_before, split_second)
        if heard_meta is None and len(needs.uncovered) == found_before:
            self.quiet_asks.append(asked_label.split(";")[0].strip())
        if replies:
            body = replies[0][1]
            run.questions_log.append(f"- {label[0]}: asked \"{label[1]}\" -> {body.split(chr(10) + '[status]')[0][:400]}{note}")
        self.drop_wanted(asked_label)
        run.asking = run.msgs_sent < ceiling
        return True

    def _put_first(self, prop: str) -> None:
        self.pending_seconds = [prop] + [p for p in self.pending_seconds or [] if p.lower() != prop.lower()]

    def _keep_open(self, prop: str) -> None:
        if prop.lower() not in [p.lower() for p in self.open_pair]:
            self.open_pair.append(prop)

    def _one_at_a_time(self, label, split_second):
        intake = self.run.intake

        def plain(text: Any) -> str:
            return " ".join(re.findall(r"[a-z0-9]+", str(text).lower()))
        self.no_pairs = True
        parts = [p.strip() for p in label[0].split(";") if p.strip()] or [label[0]]
        first = parts[0]
        plain_q = self._plain_ask(first, self._product_tag(first))
        alone = len(parts) == 1
        alt = intake.collide_alt(first) if alone else ""
        if alt:
            self.reasked.add(first.lower())
            self.spent_asks.add(first.lower())
            self._put_first(alt)
            note = f" [{first}: not answered, its story hint's alternative {alt} asked next]"
        elif first.lower() not in self.reasked and plain(plain_q) != plain(label[1]) and not (alone and intake.holds_plain(label[1], plain_q)):
            self.reasked.add(first.lower())
            self._put_first(first)
            note = f" [{first}: not answered, asked again alone next]"
        else:
            redo = intake.distinct_ask(first) if first.lower() not in self.reasked else None
            if redo:
                self.reasked.update({first.lower(), redo[0].lower()})
                self.pending_text[redo[0].lower()] = redo[1]
                self._put_first(redo[0])
                itself = re.split(r"\s*\(", redo[0])[0].strip().lower() == re.split(r"\s*\(", first)[0].strip().lower()
                if not itself:
                    self.spent_asks.add(first.lower())
                note = f" [{first}: not answered, asked again " + ("by its values next]" if itself else f"as {redo[0]} next]")
            else:
                self.spent_asks.add(first.lower())
                note = f" [{first}: not answered, not asked again]"
        for second in parts[1:] + ([split_second] if split_second else []):
            self._keep_open(second)
        return (" and ".join(parts) + " [not answered] [one at a time]", label[1]), note

    def _read_reply(self, label, heard_meta, found_before, freed_before, split_second):
        needs = self.run.needs

        def plain(text: Any) -> str:
            return " ".join(re.findall(r"[a-z0-9]+", str(text).lower()))
        if heard_meta is not None and plain(label[1]) and plain(label[1]) in plain(heard_meta):
            return self._one_at_a_time(label, split_second)
        note = ""
        paired = ";" in label[0]
        if paired and len(needs.uncovered) > found_before:
            second = label[0].split(";", 1)[1].strip()
            found_text = " ".join(needs.uncovered[found_before:]).lower()
            if not any(w in found_text for w in re.findall(r"[a-z]{4,}", second.split("(")[0].lower())):
                self.queue_second(second)
                note = f" [{second}: not answered, still open]"
            return (label[0].split(";")[0].strip(), label[1]), note
        freed_now = len(needs.indifferent) + needs.pinned_heard if paired else None
        if paired and freed_now > freed_before:
            second = label[0].split(";", 1)[1].strip()
            label = (label[0].split(";")[0].strip(), label[1])
            plain_second = second.split("(")[0].strip().lower()
            needs.indifferent = [p for p in needs.indifferent if p.strip().lower() not in (second.lower(), plain_second)]
            self._keep_open(second)
            return label, f" [{second}: not answered, still open]"
        if paired and freed_now == freed_before:
            self.no_pairs = True
            for part in (p.strip() for p in label[0].split(";")):
                if part:
                    self._keep_open(part)
            return (label[0].replace(";", " and") + " [not answered]", label[1]), note
        if split_second:
            found_text = " ".join(needs.uncovered[found_before:]).lower()
            if not any(w in found_text for w in re.findall(r"[a-z]{4,}", split_second.split("(")[0].lower())):
                self._keep_open(split_second)
        return label, note

    def reply_back(self, text_body: str) -> bool:
        run = self.run
        wire = run.wire
        if not run.role("message") or run.msgs_sent >= run.ask_limit() or run.moves_left() <= 4 or wire.spare() < CLOSING:
            return False
        if run.asking and run.msgs_sent >= run.ask_limit() - 1 and run.moves_left() > 10 and wire.spare() >= CLOSING + 40:
            return False
        proposal = self.message_call(text_body, f"reply_{wire.tick}")
        if not proposal:
            return False
        run.chat.append({"role": "assistant", "content": "Replying to what the shopper asked.", "tool_calls": [proposal]})
        self.post_replies(run.dispatch([proposal], scripted=True))
        return True

    def post_replies(self, responses: list[tuple[str, str]]) -> None:
        run = self.run
        last = len(responses) - 1
        for idx, (cid, body) in enumerate(responses):
            if idx == last:
                body += "\n" + run.status_line()
            run.chat.append({"role": "tool", "tool_call_id": cid, "content": body})


class CodeCloser:
    def __init__(self, run):
        self.run = run
        self.code_fit = {}
        self.code_hits = {}
        self.code_more = None
        self.code_searched = None
        self.news_read = set()
        self.reread = None

    # finishing arguments
    def schema_args(self, tool: str, text: str, kind: str = "") -> dict[str, Any]:
        schema = self.run.shapes.get(tool) or {}
        fields = schema.get("properties") or {}
        wording = ""
        if kind == "decline":
            wording = DECLINE_WORDS
        elif kind == "order":
            wording = ORDER_WORDS
        filled: dict[str, Any] = {}
        for field in schema.get("required") or []:
            spec = fields.get(field) or {}
            ftype = spec.get("type")
            if spec.get("enum"):
                matching = [choice for choice in spec["enum"]
                            if wording and isinstance(choice, str) and re.search(wording, choice.lower())]
                filled[field] = (matching or spec["enum"])[0]
            elif ftype == "string" or "string" in (ftype or []):
                filled[field] = text[: spec.get("maxLength", len(text))]
            elif ftype == "boolean":
                filled[field] = True
            elif ftype in ("integer", "number"):
                filled[field] = max(1, spec.get("minimum", 1))
        if len(fields) != 1:
            return filled
        field, spec = next(iter(fields.items()))
        lone_text = isinstance(spec, dict) and spec.get("type") == "string" and "enum" not in spec
        if field not in filled and lone_text:
            filled[field] = text[: spec.get("maxLength", len(text))]
        return filled

    def scrub_decline(self, call_args: dict[str, Any], call_name: str) -> dict[str, Any]:
        rule = self.run.needs.reading.get("decline_no_names")
        if rule is False:
            return call_args
        known = self.run.archive.ids
        mentions_id = False
        for value in call_args.values():
            if not isinstance(value, str):
                continue
            if any(piece and piece in known for piece in re.split(r"[\s.,;:!?()\[\]{}<>\"'`]+", value)):
                mentions_id = True
                break
        if not rule and not mentions_id:
            return call_args
        fields = (self.run.shapes.get(call_name) or {}).get("properties") or {}
        cleaned = {}
        for field, value in call_args.items():
            spec = fields.get(field) or {}
            if isinstance(value, str) and "enum" not in spec:
                cleaned[field] = SORRY[: spec.get("maxLength", len(SORRY))]
            else:
                cleaned[field] = value
        return cleaned

    # a plan made by code when no LLM could read the request
    def code_reading(self) -> dict[str, Any]:
        text = self.run.request
        lowered = text.lower()
        plan: dict[str, Any] = {"code_reading": True, "requirements": [], "items": [], "quantity": 1, "rules": [],
                                "soft_wants": [], "ask_first": [], "hints": [], "references": [], "priority_detail": ""}
        plan["price_cap"], plan["currency"] = self.budget_in(text)
        stock_cue = re.search(r"\bin[- ]stock\b|\bon hand\b|\bon the shelf\b|\bsold[- ]out\b|\bback[- ]?order|"
                              r"\bout of stock\b|\bavailable now\b", lowered)
        plan["in_stock_only"] = bool(stock_cue)
        cheap_cue = re.search(r"\bcheapest\b|\blowest[- ](?:price|priced|cost)\b|\bleast expensive\b|"
                              r"\blowest\b[^.]{0,20}\bprice\b|\bsmallest (?:total|price)\b", lowered)
        plan["priority"] = "cheapest" if cheap_cue else "none"
        message_limit = self.limit_in(lowered, r"messages?|questions?|notes?|repl(?:y|ies)|texts?")
        if message_limit is not None:
            plan["message_budget"] = message_limit
        step_limit = self.limit_in(lowered, r"steps?|moves?|actions?|(?:tool )?calls?|turns?")
        if step_limit is not None and step_limit >= FIRST_ORDER_BY + 3:
            plan["step_budget"] = step_limit
        roles = self.code_roles()
        plan["roles"] = roles
        products = self.counted_products(roles.get("search"))
        if len(products) < 2:
            if products:
                plan["product"], plan["quantity"] = products[0]["label"], products[0]["quantity"]
            else:
                plan["product"] = self.closest_label(roles.get("search"), kind_words(text))
                plan["quantity"] = self.units_in(lowered)
            plan["requirements"] = [f"{name}: {value}" for name, value in self.stated_pairs(lowered)]
            return plan
        plan["items"] = [{field: entry[field] for field in ("product", "quantity", "requirements")} for entry in products]
        plan["product"] = " and ".join(entry["product"] for entry in products)
        plan["requirements"] = [need for entry in products for need in entry["requirements"]]
        return plan

    @staticmethod
    def budget_in(text: str) -> tuple[Any, Any]:
        top: tuple[int, float] = (0, 0.0)
        answer: tuple[Any, Any] = (None, None)
        for sentence in re.split(r"(?<=[.!?;])\s+", str(text).lower()):
            base_score = 0
            if re.search(r"\b(?:order|total|whole|lot|sum|bill|altogether|overall|everything)\b", sentence):
                base_score = 1
            for hit in re.finditer(r"([$€£]\s*)?(\d[\d,]*(?:\.\d+)?)", sentence):
                tail = sentence[hit.end():]
                if re.match(r"\s*(?:x\b|×|%|:\s*\d|-\s*\d|/\s*\d|(?:[a-z-]+\s+){0,2}?(?:messages?|questions?|notes?|repl\w*|texts?|"
                            r"steps?|moves?|actions?|(?:tool )?calls?|turns?|units?|items?|pieces?|packs?|pairs?|lots?|sets?|of\b|times?)\b)", tail):
                    continue
                sign = hit.group(1)
                if not sign and re.search(r"[\d.,:/-]\s*$|\bx\s*$", sentence[:hit.start()]):
                    continue
                lead = sentence[max(0, hit.start() - 60): hit.start()]
                score = base_score
                if sign or re.search(r"\b(?:spend\w*|budget\w*|afford\w*|pay\w*|price[ds]?|costs?|cash|money)\b(?:\W+[\w']+){0,4}\W*$", lead):
                    score += 3
                if re.search(r"\b(?:under|below|no more than|not more than|up to|within|at most|max\w*|less than|cheaper than|"
                             r"not (?:more|over|above)|limit\w*|reach\w*|go to|to|at|cap\w*|past|exceed\w*|beyond|go over)\W+(?:[\w']+\W+)?$", lead):
                    score += 2
                if re.match(r"\s*(?:[a-z.()]+\s*){0,3}?(?:or (?:lower|less|under|below|cheaper)|at (?:the (?:very )?)?most|max\w*|tops\b|"
                            r",?\s*(?:and )?not (?:one|a) \w+ (?:over|more|above))", tail):
                    score += 2
                amount = float(hit.group(2).replace(",", ""))
                if score < 2 or (score, amount) <= top:
                    continue
                top = (score, amount)
                unit_match = re.match(r"\s*((?:[a-z][a-z.]*\s*){0,2})", tail)
                unit_words = []
                for word in (unit_match.group(1).split() if unit_match else []):
                    bare = word.strip(".")
                    if bare in LINK_WORDS or re.fullmatch(r"or|and|not|tops|max\w*|only|just|is|was", bare):
                        continue
                    unit_words.append(bare)
                symbol = (sign or "").strip()
                answer = (int(amount) if amount.is_integer() else amount, " ".join(unit_words) or symbol or None)
        return answer

    @staticmethod
    def limit_in(lowered: str, nouns: str) -> int | None:
        numbers = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
                   "twelve": 12, "fifteen": 15, "twenty": 20}
        pattern = (r"\b(\d+|one|two|three|four|five|six|seven|eight|nine|ten|twelve|fifteen|twenty)\s+(?:[a-z-]+\s+){0,2}?(?:"
                   + nouns + r")\b")
        for hit in re.finditer(pattern, lowered):
            lead = lowered[max(0, hit.start() - 60): hit.start()]
            tail = lowered[hit.end(): hit.end() + 40]
            cue_before = re.search(r"\b(?:no more than|not more than|up to|at most|only|max\w*|use|using|take|send|allow\w*|receive|get|total of|"
                                   r"limit\w*|inside|within|under|fewer than|less than|you've|you have|have|keep\w*|stick to|go past|past|"
                                   r"exceed\w*|beyond)\W+(?:[\w']+\W+){0,3}$", lead)
            if not cue_before and not re.match(r"\W*(?:[\w']+\W+){0,3}?(?:in total|total|no more|at most|max\w*|tops|or fewer|or less|allowed|limit)\b", tail):
                continue
            word = hit.group(1)
            return int(word) if word.isdigit() else numbers[word]
        return None

    @staticmethod
    def units_in(lowered: str) -> int:
        numbers = number_words()
        hit = re.search(r"\b(\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s+(?:units?|of (?:them|those|these)|copies)\b", lowered)
        if hit is None:
            return 1
        word = hit.group(1)
        count = int(word) if word.isdigit() else numbers[word]
        return count if 1 <= count <= 99 else 1

    @staticmethod
    def stated_pairs(stretch: str) -> list[tuple[str, str]]:
        verbs = r"(?:is|are|has to be|have to be|must be|should be|needs to be|need to be|can be)"
        stop = r"(?=\s*(?:[,.;:!?()]|$)|\s+(?:so|since|as|because|for|to|which|that|nothing|otherwise|but|when|while|if|then|though|please|and (?:the|its|their|a|an|one|two|three|\d))\b)"
        pattern = (r"(?:\b(?:where|whose)\s+(?:the\s+|its\s+|their\s+)?|(?:\b(?:and|with|but)\s+|[,;:]\s*|^|[.!?]\s+)(?:the|its|their)\s+)"
                   r"([a-z][a-z'&/ -]{1,32}?)\s+" + verbs + r"\s+(?:either\s+|just\s+)?([^,.;:!?()]{1,60}?)" + stop)
        pairs: list[tuple[str, str]] = []
        for hit in re.finditer(pattern, stretch):
            name, value = hit.group(1).strip(), hit.group(2).strip()
            after = stretch[hit.end(): hit.end() + 60]
            if ref_text(value):
                continue
            if ref_text(after) and re.search(r"\b(?:page|listing|shown|sold by)\b", after):
                continue
            if not value or len(value.split()) > 4 or len(name.split()) > 4:
                continue
            if re.search(r"\b(?:i|you|we|it|he|she|they|my|your|our|there|this|that|which|what|who|me|us|them|one|ones|price|budget|cost|total|order|stock)\b", name):
                continue
            if re.match(r"(?:a|an|the|my|your|our|his|her|their|me|us|for|to|in|on|at|of|see|go|get|be|have|only|too|very|so|not|no|what|how|it|all|"
                        r"mostly|always|never|just|really|still|also|from|about|like|one|ones)\b", value):
                continue
            pairs.append((name, value))
        return pairs

    def search_labels(self, tool: str | None) -> list:
        fields = (self.run.shapes.get(tool or "") or {}).get("properties")
        choice_lists = []
        for spec in (fields.values() if isinstance(fields, dict) else []):
            if isinstance(spec, dict) and isinstance(spec.get("enum"), list) and len(spec["enum"]) >= 4:
                choice_lists.append(spec["enum"])
        return max(choice_lists, key=len) if choice_lists else []

    def counted_products(self, tool: str | None) -> list[dict[str, Any]]:
        labels = [str(label) for label in self.search_labels(tool)]
        lowered = self.run.request.lower()
        tokens = [(hit.group(0).rstrip("s"), hit.start(), hit.end())
                  for hit in re.finditer(r"[a-z]{3,}", lowered) if hit.group(0) not in LINK_WORDS]
        stems = [token[0] for token in tokens]
        numbers = number_words()
        found: dict[str, tuple[int, int, int]] = {}
        for label in labels:
            label_stems = [w.rstrip("s") for w in re.findall(r"[a-z]{3,}", label.lower()) if w not in LINK_WORDS]
            width = len(label_stems)
            if not width:
                continue
            for at in range(len(stems) - width + 1):
                if stems[at: at + width] != label_stems:
                    continue
                begin, finish = tokens[at][1], tokens[at + width - 1][2]
                if len(re.findall(r"[a-z0-9]+", lowered[begin:finish])) > 2 * width:
                    continue
                counter = re.search(r"\b(\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s*(?:x\s+|×\s*)?(?:(?:of|[a-z'&-]{3,})\s+){0,4}$",
                                    lowered[max(0, begin - 56): begin])
                if not counter or label in found:
                    continue
                word = counter.group(1)
                count = int(word) if word.isdigit() else numbers[word]
                if 1 <= count <= 99:
                    found[label] = (begin, finish, count)
                break
        in_order = sorted(found.items(), key=lambda pair: pair[1][0])
        products = []
        for index, (label, (begin, finish, count)) in enumerate(in_order):
            last = index + 1 == len(in_order)
            stretch = lowered[finish: len(lowered) if last else in_order[index + 1][1][0]]
            if last:
                stretch = re.split(r"(?<=[.!?])\s", stretch)[0]
            name = label.lower()
            products.append({"product": name, "label": label, "quantity": count,
                             "requirements": [f"{prop} ({name}): {value}" for prop, value in self.stated_pairs(stretch)]})
        return products

    def code_roles(self) -> dict[str, str]:
        described: dict[str, str] = {}
        for tdef in self.run.toolset:
            fn = tdef["function"]
            spaced = re.sub(r"([a-z])([A-Z])", r"\1 \2", fn["name"]).replace("_", " ").lower()
            described[fn["name"]] = spaced + " | " + str(fn.get("description") or "").lower()
        shapes = self.run.shapes

        def fields_of(tool: str) -> dict[str, Any]:
            found = (shapes.get(tool) or {}).get("properties")
            return found if isinstance(found, dict) else {}

        def required_of(tool: str) -> list[str]:
            found = (shapes.get(tool) or {}).get("required")
            if not isinstance(found, list):
                return []
            return [field for field in found if field in fields_of(tool)]

        def free_texts(tool: str) -> list[str]:
            return [field for field, spec in fields_of(tool).items()
                    if isinstance(spec, dict) and spec.get("type") == "string" and not spec.get("enum")]

        roles: dict[str, str] = {}
        for tool in described:
            for field in required_of(tool):
                spec = fields_of(tool)[field]
                choices = [str(c).lower() for c in (spec.get("enum") or [])] if isinstance(spec, dict) else []
                both = any(re.search(ORDER_WORDS, c) for c in choices) and any(re.search(DECLINE_WORDS, c) for c in choices)
                if both:
                    roles.setdefault("order", tool)
                    roles.setdefault("decline", tool)
        for tool in described:
            if tool in roles.values():
                continue
            title = described[tool].split(" | ")[0]
            if "decline" not in roles and re.search(DECLINE_WORDS, title):
                roles["decline"] = tool
            elif "order" not in roles and re.search(ORDER_WORDS, title):
                roles["order"] = tool
        for tool in described:
            if tool in roles.values() or required_of(tool):
                continue
            words = described[tool]
            final = re.search(r"\bfinal\b", words)
            if "decline" not in roles and final and re.search(DECLINE_WORDS, words):
                roles["decline"] = tool
            elif "order" not in roles and final and re.search(ORDER_WORDS, words):
                roles["order"] = tool
        for tool in described:
            if tool in roles.values() or not re.search(r"\b(?:cart|basket|bag|trolley)\b", described[tool]):
                continue
            takes_list = any(isinstance(spec, dict) and spec.get("type") == "array" for spec in fields_of(tool).values())
            if required_of(tool) or takes_list:
                roles.setdefault("cart", tool)
            else:
                roles.setdefault("cart_view", tool)
        person = r"\b(?:shoppers?|customers?|buyers?|clients?|users?)\b"
        speak = r"\b(?:send\w*|repl\w*|notes?|tell\w*|messag\w*|chat\w*|writ\w*|talk\w*)\b"
        cues = [("search", r"\b(?:search\w*|find\w*|look\w*|locat\w*|hunt\w*|brows\w*|quer\w*)\b", False),
                ("message", speak, True),
                ("details", r"\b(?:details?|records?|spec\w*|propert\w*|attribut\w*|facts?|full)\b", True),
                ("view", r"\b(?:offer|price|stock|availab\w*)\b", True)]
        for role, cue, single_text in cues:
            for tool in described:
                words = described[tool]
                if role in roles or tool in roles.values() or not re.search(cue, words):
                    continue
                talks_to_shopper = re.search(person, words) and re.search(speak, words)
                if role == "search" and talks_to_shopper:
                    continue
                if role == "message" and not re.search(person, words) and not re.search(r"\b(?:messag\w*|chat\w*)\b", words):
                    continue
                if single_text:
                    needed = required_of(tool)
                    if len(needed) != 1 or needed[0] not in free_texts(tool):
                        continue
                elif not free_texts(tool) and not any(isinstance(spec, dict) and spec.get("enum") for spec in fields_of(tool).values()):
                    continue
                roles[role] = tool
        return roles

    def closest_label(self, tool: str | None, wanted: set[str]) -> str:
        chosen, score = "", (0, 0.0)
        for label in self.search_labels(tool):
            own = kind_words(label)
            shared = len(own & wanted)
            if not shared:
                continue
            rank = (shared, shared / max(1, len(own)))
            if rank > score:
                chosen, score = str(label), rank
        return chosen

    def code_values(self, k: int | None) -> list[tuple[str, str]]:
        needs = self.run.needs
        several = needs.items()
        pairs = []
        for need in needs.props_left():
            head, _, value = str(need).partition(":")
            if several and self.run.talk.product_of(head) not in (k, None):
                continue
            bare = re.sub(r"\((?:stated|revealed[^)]*)\)", "", value).strip()
            brief = len(re.findall(r"[a-z0-9]+", bare.lower())) <= 5 and not re.search(EXCLUDE_WORDS, bare.lower())
            if not bare or ref_text(bare):
                continue
            if positive_value(value, head) or brief:
                pairs.append((re.split(r"\s*\(", head)[0].strip(), bare))
        return pairs

    def code_queries(self, k: int | None) -> list[dict[str, Any]]:
        run = self.run
        needs = run.needs
        tool = run.role("search")
        schema = run.shapes.get(tool or "") or {}
        fields = schema.get("properties") if isinstance(schema.get("properties"), dict) else {}
        must = [field for field in schema.get("required") or [] if field in fields]
        several = needs.items()
        if several and k is not None and k < len(several):
            product = str(several[k]["product"])
        else:
            product = str(needs.reading.get("product") or "")
        label = self.closest_label(tool, kind_words(product) or kind_words(run.request))
        blurb = ""
        for tdef in run.toolset:
            if tdef["function"]["name"] == tool:
                blurb = str(tdef["function"].get("description") or "")
                break
        clauses = [clause.lower() for clause in re.split(r"[.;]", blurb)]
        paging = run.watch.token_key()
        free = [field for field, spec in fields.items()
                if isinstance(spec, dict) and spec.get("type") == "string" and not spec.get("enum") and field != paging]
        text_field = next((field for field in free if re.search(r"^q$|query|text|keyword|needle|term|word|phrase|string|search", field.lower())),
                          free[0] if free else None)
        choices = {field: spec["enum"] for field, spec in fields.items()
                   if isinstance(spec, dict) and isinstance(spec.get("enum"), list) and spec["enum"]}
        shelf_field = max((field for field in choices if len(choices[field]) >= 4), key=lambda field: len(choices[field]), default=None)
        mode_fields = [field for field in choices if field != shelf_field and len(choices[field]) <= 4]
        mode_field = next((field for field in mode_fields if field in must), mode_fields[0] if mode_fields else None)

        def mode_value(arg: str, use_last: bool) -> Any:
            options = choices.get(mode_field or "", [])
            for option in options:
                for clause in clauses:
                    if re.search(r"\b" + re.escape(str(option).lower()) + r"\b", clause) and re.search(r"\b" + re.escape(arg.lower()) + r"\b", clause):
                        return option
            return options[-1] if use_last else options[0]

        def about_price(field: str) -> bool:
            if re.search(r"ceil|max|price|cap$|budget|upper|most", field.lower()):
                return True
            return any(field.lower() in clause and re.search(r"\bpric|\bcost", clause) for clause in clauses)

        ceiling_field = None
        for field, spec in fields.items():
            if not isinstance(spec, dict) or spec.get("type") not in ("integer", "number") or spec.get("enum"):
                continue
            if about_price(field) and not re.search(r"page|first|offset|skip|count|size|num", field.lower()):
                ceiling_field = field
                break
        budget = needs.spend_cap
        counts = needs.counts()
        units = (counts[k] if k is not None and k < len(counts) else needs.quantity()) or 1
        limit_arg = {}
        if ceiling_field and is_number(budget) and budget > 0:
            per_unit = int(budget // units) if fields[ceiling_field].get("type") == "integer" else round(budget / units, 2)
            limit_arg = {ceiling_field: per_unit}
        longest = None
        if text_field and isinstance(fields.get(text_field), dict):
            longest = fields.get(text_field, {}).get("maxLength")
        product_words = [w for w in re.findall(r"[a-z0-9]+", product.lower()) if len(w) > 2 and w not in LINK_WORDS]
        kind = product_words[-1] if product_words else ""
        firm = []
        for _, value in self.code_values(k):
            if not re.search(r"\bor\b|/|,|\b(?:" + VALUE_LINKS + r")\b", value.lower()):
                firm.append(value)
        phrases: list[str] = []
        if firm:
            value_words = [w for w in re.findall(r"[a-z0-9]+", firm[0].lower()) if w not in LINK_WORDS]
            if value_words:
                phrases.append(" ".join(dict.fromkeys(value_words + [kind])))
                if len(value_words) > 1:
                    phrases.append(" ".join(dict.fromkeys([max(value_words, key=len), kind])))
        if not phrases:
            if product_words:
                phrases.append(" ".join(dict.fromkeys(product_words[:1] + product_words[-1:])))
            else:
                phrases.append(" ".join(sorted(kind_words(label))[:2]))
        if is_number(longest):
            trimmed = []
            for phrase in phrases:
                while " " in phrase and len(phrase) > longest:
                    phrase = phrase.rsplit(" ", 1)[0]
                trimmed.append(phrase[: int(longest)])
            phrases = trimmed

        def with_mode(field: str, use_last: bool) -> dict[str, Any]:
            return {mode_field: mode_value(field, use_last)} if mode_field else {}

        drafts: list[dict[str, Any]] = []
        if text_field and phrases[0]:
            drafts.append({text_field: phrases[0], **with_mode(text_field, False), **limit_arg})
        if label and shelf_field:
            drafts.append({shelf_field: label, **with_mode(shelf_field, True), **limit_arg})
        if text_field and len(phrases) > 1:
            drafts.append({text_field: phrases[1], **with_mode(text_field, False), **limit_arg})
        ready: list[dict[str, Any]] = []
        for draft in drafts:
            for field in must:
                if field in draft:
                    continue
                spec = fields[field] if isinstance(fields[field], dict) else {}
                if spec.get("enum"):
                    draft[field] = spec["enum"][0]
                elif spec.get("type") in ("integer", "number"):
                    draft[field] = spec.get("default", spec.get("minimum", 1))
                elif spec.get("type") == "string":
                    draft[field] = phrases[0] or label
            fixed, faults = ArgCheck.execute(draft, schema)
            if faults and limit_arg:
                fixed, faults = ArgCheck.execute({f: v for f, v in draft.items() if f not in limit_arg}, schema)
            if not faults and fixed not in ready:
                ready.append(fixed)
        return ready

    def code_rows(self) -> dict[str, dict[str, Any]]:
        watch = self.run.watch
        known = self.run.archive.ids
        table: dict[str, dict[str, Any]] = {}

        def note(record: dict[str, Any]) -> None:
            ident = watch.row_mark(record)
            if ident not in known:
                return
            entry = table.setdefault(ident, {"shelf": "", "title": "", "price": None, "stock": None, "words": set(), "fields": {}})
            entry["shelf"] = watch.shelf_of(record) or entry["shelf"]
            title = ""
            for key, value in record.items():
                lowered = str(key).lower()
                if isinstance(value, str) and re.search(r"title|name", lowered) and "id" not in lowered:
                    title = str(value)
                    break
            entry["title"] = title or entry["title"]
            for key, value in record.items():
                lowered = str(key).lower()
                if not is_number(value) or isinstance(value, bool) or not re.search(r"price|cost|amount", lowered):
                    continue
                if re.search(r"count|qty|quant|total|units|line|due|payable|sum", lowered):
                    continue
                entry["price"] = float(value)
                break
            for key, value in record.items():
                state = watch.key_stock(key, value)
                if state is not None:
                    entry["stock"] = state
            texts = [str(v) for v in record.values() if isinstance(v, (str, int, float)) and not isinstance(v, bool)]
            for value in record.values():
                if isinstance(value, dict):
                    texts.extend(str(x) for x in value.values() if isinstance(x, (str, int, float)))
            entry["words"] |= kind_words(" ".join(texts))
            for value in record.values():
                if not isinstance(value, dict):
                    continue
                entry["fields"].update({str(key): str(x) for key, x in value.items()
                                        if isinstance(x, (str, int, float)) and not isinstance(x, bool)})

        for record in self.run.archive.items:
            walk_dicts(record, note)
        return table

    @staticmethod
    def code_shows(row: dict[str, Any], prop: str, value: str, hit_words: set[str], read: bool) -> bool:
        own_field = [text for key, text in row["fields"].items() if same_prop(key, prop)]
        if own_field:
            return any(CodeCloser.code_seen(value, text) for text in own_field)
        if CodeCloser.code_seen(value, row.get("title") or ""):
            return True
        if read:
            return False
        value_words = kind_words(value)
        return bool(value_words) and value_words <= hit_words

    @staticmethod
    def code_seen(value: str, text: str) -> bool:
        haystack = str(text or "").lower()
        if not haystack.strip():
            return False
        stripped = re.sub(r"\b(?:" + VALUE_LINKS + r")\b", " ", str(value).lower())
        for option in re.split(r"\s*(?:\bor\b|/|,|;)\s*", stripped):
            words = [w for w in re.findall(r"[a-z]{2,}", option) if w not in LINK_WORDS]
            numbers = re.findall(r"\d+(?:\.\d+)?", option)
            if not words and not numbers:
                continue
            words_ok = all(re.search(r"\b" + re.escape(w[:-1] if len(w) > 4 else w), haystack) for w in words)
            if words_ok and all(re.search(r"(?<![\d.])" + re.escape(n) + r"(?![\d.])", haystack) for n in numbers):
                return True
        return False

    @staticmethod
    def evidence_rank(row: dict[str, Any], prop: str, value: str) -> tuple[int, int]:
        wanted = set()
        for option in re.split(r"\s*(?:\bor\b|/)\s*", value.lower()):
            if ch_words(option):
                wanted.add(" ".join(ch_words(option)))
        rank = (2, 3)
        for key, text in row["fields"].items():
            if not same_prop(key, prop) or not CodeCloser.code_seen(value, text):
                continue
            if " ".join(ch_words(text)) in wanted:
                match = 0
            elif any(" ".join(ch_words(part)) in wanted for part in ch_parts(text)):
                match = 1
            else:
                match = 3
            rank = min(rank, (0 if prop_key(key) == prop_key(prop) else 1, match))
        if rank == (2, 3) and CodeCloser.code_seen(value, row.get("title") or ""):
            rank = (2, 2)
        return rank

    def audit_silence(self, live: set[str]) -> set[str]:
        run = self.run
        judge = run.judge
        seen = judge.audit_ids
        if run.audit_done:
            limit = -1 if judge.audit_reads is None else judge.audit_reads
            if seen is None or len(run.archive.read) <= limit:
                seen = set(run.archive.read)
        answered = False
        for entry in reversed(run.log):
            if isinstance(entry, dict) and entry.get("kind") == "audit":
                answered = isinstance(entry.get("result"), dict)
                break
        if not answered:
            return set()
        return set(seen or set()) - set(judge.audit_fits or []) - live

    def code_picks(self, avoid: set[str] | None = None, keep: set[str] | None = None) -> list[tuple[str, int]]:
        run = self.run
        needs, archive = run.needs, run.archive
        several = needs.items()
        counts = needs.counts()
        if several:
            homes = run.shelf.home_shelves()
        else:
            homes = [set(run.watch.shelf_labels()) or (run.shelf.home_shelves() or [set()])[0]]
        table = self.code_rows()
        version = str(needs.version)
        failed = set()
        for key, verdict in run.rulings.items():
            if verdict and key.split("|")[1:2] == [version]:
                failed.add(key.split("|")[0])
        live = {key.rpartition("|")[0] for key in run.passed if key.rpartition("|")[2] == version}
        audited = self.audit_silence(live)
        trust_own = bool(audited) and run.shelf.code_judges_all()
        cap = float(needs.spend_cap) if is_number(needs.spend_cap) else None
        request_words = kind_words(run.request)
        hits = self.code_hits or {}
        cheapest = needs.priority() == "cheapest"
        avoid, keep = set(avoid or ()), set(keep or ())
        avoid |= run.shelf.dead_lines()
        avoid |= run.refdesk.ref_pointers()
        pools: list[list[tuple]] = []
        for k, home in enumerate(homes):
            units = counts[k] if k < len(counts) else 1
            slot = k if several else None
            kind = kind_words(several[k]["product"] if several else needs.reading.get("product") or "")
            value_words: set[str] = set()
            judged: list[tuple[str, str]] = []
            for prop, bare in self.code_values(slot):
                value_words |= kind_words(bare)
                if not re.search(r"\b(?:" + VALUE_LINKS + r")\b|\d\s*(?:-|to)\s*\d|[+<>]|\b(?:about|around|approx\w*|roughly)\b", bare.lower()):
                    judged.append((prop, bare))
            set_apart = []
            for ref in run.shelf.differ_done(slot):
                excluded = [str(v) for v in [ref.get("value")] + list(ref.get("goes_with") or []) if str(v or "").strip()]
                set_apart.append((ref["prop"], excluded))
            pool = []
            for ident, row in table.items():
                if ident in failed or ident in run.refused_lines or ident in avoid or row["stock"] is False:
                    continue
                if home:
                    if row["shelf"] not in home:
                        continue
                elif not kind & row["words"]:
                    continue
                if any(self.code_clash(row["fields"], prop, value) for prop, value in judged):
                    continue
                if run.refdesk.ref_excluded(row["fields"], slot):
                    continue
                cost = row["price"] if row["price"] is not None else archive.price(ident)
                if cap is not None and is_number(cost) and cost * units > cap:
                    continue
                was_read = ident in archive.read
                fits = all(self.code_shows(row, prop, value, hits.get(ident) or set(), was_read) for prop, value in judged)
                if fits:
                    fits = all(any(same_prop(key, prop) and str(text).strip() and not any(ch_writes(v, text) for v in bad)
                                   for key, text in row["fields"].items()) for prop, bad in set_apart)
                ranks = [self.evidence_rank(row, prop, value) for prop, value in judged] if was_read else []
                if ident in audited:
                    cleared = trust_own and fits and (judged or set_apart) and all(max(r) <= 1 for r in ranks)
                    if not cleared:
                        continue
                words = row["words"]
                score = (3 * len(value_words & words) + 2 * len(kind & words) + len(request_words & words)
                         + 4 * len(value_words & (hits.get(ident) or set())))
                price = cost if is_number(cost) else float("inf")
                quality = (sum(r[0] for r in ranks), sum(r[1] for r in ranks)) if ranks else (9, 9)
                pool.append((0 if ident in keep else 1, 0 if ident in live else 1, not fits, cap is not None and not is_number(cost),
                             price if cheapest else 0.0, 0 if was_read else 1, row["stock"] is not True,
                             (quality, -score), price, ident))
            if not pool:
                return []
            pools.append(sorted(pool))
        orderings = ((lambda e: e), (lambda e: (e[2], e[3], e[8], e[-1])))
        for ordering in orderings:
            picks: list[tuple[str, int]] = []
            chosen = []
            total = 0.0
            for k, pool in enumerate(pools):
                taken = None
                for entry in sorted(pool, key=ordering):
                    if any(entry[-1] == ident for ident, _ in picks):
                        continue
                    if entry[-1] in archive.read and run.judge.field_view(entry[-1])["fails"]:
                        continue
                    taken = entry
                    break
                if taken is None:
                    return []
                units = counts[k] if k < len(counts) else 1
                picks.append((taken[-1], units))
                chosen.append(taken)
                total += (0.0 if taken[8] == float("inf") else taken[8]) * units
            if cap is None or total <= cap:
                self.code_fit = {entry[-1]: not entry[2] for entry in chosen}
                return picks
        return []

    @staticmethod
    def code_clash(fields: dict[str, str], prop: str, value: str) -> bool:
        own_field = [text for key, text in fields.items() if same_prop(key, prop)]
        if not own_field:
            return False
        return not any(value_seen(value, text) for text in own_field)

    @staticmethod
    def choice_words(value: Any) -> set[str]:
        return set(re.findall(r"[a-z]+", re.sub(r"([a-z])([A-Z])", r"\1 \2", str(value)).lower()))

    def cart_form(self) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], str | None, list[str]]:
        run = self.run
        schema = run.shapes.get(run.role("cart") or "") or {}
        fields = schema.get("properties") if isinstance(schema.get("properties"), dict) else {}
        first_form = json_object(run.trolley.put_form) if run.trolley.put_form else None
        put_values: dict[str, Any] = {}
        drop_values: dict[str, Any] = {}
        for field in schema.get("required") or []:
            spec = fields.get(field) if isinstance(fields.get(field), dict) else {}
            options = spec.get("enum") if isinstance(spec.get("enum"), list) else []
            if not options:
                continue
            adders = [o for o in options if self.choice_words(o) & PUT_WORDS and not self.choice_words(o) & DROP_WORDS]
            removers = [o for o in options if self.choice_words(o) & DROP_WORDS]
            if isinstance(first_form, dict) and first_form.get(field) in options:
                chosen = first_form[field]
            else:
                chosen = adders[0] if adders else options[0]
            put_values[field] = chosen
            rest = [o for o in options if o != chosen]
            drop_values[field] = next((o for o in removers if o != chosen), rest[0] if len(options) == 2 else None)
        free = [field for field, spec in fields.items() if isinstance(spec, dict) and spec.get("type") == "string" and not spec.get("enum")]
        id_field = next((field for field in free if re.search(r"id|item|product|listing|sku|target|which|number", field.lower())),
                        free[0] if free else None)
        unit_fields = [field for field, spec in fields.items()
                       if isinstance(spec, dict) and spec.get("type") in ("integer", "number") and not spec.get("enum")]
        return schema, put_values, drop_values, id_field, unit_fields

    def cart_args(self, picks: list[tuple[str, int]]) -> list[dict[str, Any]]:
        schema, put_values, _, id_field, unit_fields = self.cart_form()
        line_spec = self.run.trolley.line_spec()
        drafts: list[dict[str, Any]] = []
        if line_spec:
            array_field, line_id, line_units = line_spec
            drafts.append({**put_values, array_field: [{line_id: ident, line_units: units} for ident, units in picks]})
        elif not id_field:
            return []
        else:
            for ident, units in picks:
                draft = {**put_values, id_field: ident}
                if unit_fields:
                    draft[unit_fields[0]] = units
                drafts.append(draft)
        checked = []
        for draft in drafts:
            fixed, faults = ArgCheck.execute(draft, schema)
            if faults:
                return []
            checked.append(fixed)
        return checked

    def cart_drops(self, lines: list[str]) -> list[dict[str, Any]] | None:
        trolley = self.run.trolley
        if not lines or trolley.line_spec():
            return []
        schema, put_values, drop_values, id_field, unit_fields = self.cart_form()
        fields = schema.get("properties") if isinstance(schema.get("properties"), dict) else {}
        if not id_field:
            return None
        checked = []
        for ident in lines:
            if put_values:
                if any(drop_values.get(field) is None for field in put_values):
                    return None
                draft = {**drop_values, id_field: ident}
                if unit_fields:
                    draft[unit_fields[0]] = max(1, int(trolley.held(ident) or 1))
            else:
                if not unit_fields:
                    return None
                floor = (fields.get(unit_fields[0]) or {}).get("minimum")
                if is_number(floor) and fields[unit_fields[0]]["minimum"] > 0:
                    return None
                draft = {id_field: ident, unit_fields[0]: 0}
            fixed, faults = ArgCheck.execute(draft, schema)
            if faults:
                return None
            checked.append(fixed)
        return checked

    def latest_replies(self, count: int) -> list[Any]:
        newest = None
        for entry in reversed(self.run.log):
            if isinstance(entry, dict) and entry.get("kind") == "env":
                newest = entry
                break
        reply = (newest or {}).get("reply")
        if not isinstance(reply, dict):
            return []
        if isinstance(reply.get("observations"), list):
            seen = reply.get("observations")
        else:
            seen = [reply.get("observation")]
        return list(seen)[:count]

    @staticmethod
    def code_fault(obs: Any) -> bool:
        flagged = []

        def look(record: dict[str, Any]) -> None:
            for key, value in record.items():
                if re.search(FAULT_KEYS, str(key).lower()) and value not in (None, False, "", [], {}):
                    flagged.append(record)
                    return
        walk_dicts(obs, look, 0, 2)
        return bool(flagged)

    def book_hits(self, calls: list[dict[str, Any]], replies: list[Any], text_fields: list[str]) -> None:
        run = self.run
        hits = self.code_hits
        for call_args, obs in zip(calls, replies):
            typed = " ".join(str(call_args.get(field) or "") for field in text_fields)
            if not typed.strip():
                continue
            words = kind_words(typed)

            def mark(record: dict[str, Any]) -> None:
                ident = run.watch.row_mark(record)
                if ident in run.archive.ids:
                    hits.setdefault(ident, set()).update(words)
            walk_dicts(obs, mark)

    def code_search(self, products: int, more: bool = False) -> None:
        run = self.run
        tool = run.role("search")
        fields = (run.shapes.get(tool or "") or {}).get("properties") or {}
        text_fields = [field for field, spec in fields.items()
                       if isinstance(spec, dict) and spec.get("type") == "string" and not spec.get("enum")]
        reserve = 2 + 2 * products
        if more:
            leading = list(self.code_more or [])
            calls = leading[: max(0, run.moves_left() - reserve - 2 * products)]
            rounds = 1
        else:
            per_product = [self.code_queries(k if run.needs.items() else None) for k in range(products)]
            calls = []
            depth = max((len(q) for q in per_product), default=0)
            for rank in range(depth):
                for queries in per_product:
                    if rank < len(queries) and queries[rank] not in calls:
                        calls.append(queries[rank])
            calls = calls[: max(0, run.moves_left() - reserve)]
            leading = [queries[0] for queries in per_product if queries]
            rounds = 1 + CODE_PAGES
            if run.needs.priority() == "cheapest":
                rounds += 2
        if not calls:
            return
        self.code_searched = True
        for round_no in range(rounds):
            run.send_chunk([(f"code_search_{n}_{run.wire.tick}", tool, call_args) for n, call_args in enumerate(calls)])
            replies = self.latest_replies(len(calls))
            if run.word_kind != "any":
                self.book_hits(calls, replies, text_fields)
            onward = []
            for call_args, obs in zip(calls, replies):
                if call_args not in leading:
                    continue
                follow = self.next_page_call(call_args, obs)
                if follow:
                    onward.append(follow)
            self.code_more = onward
            if round_no == rounds - 1:
                break
            picks = self.code_picks()
            fitting = self.code_fit or {}
            if run.needs.priority() != "cheapest" and picks and all(fitting.get(ident) for ident, _ in picks):
                break
            leading = onward
            calls = onward[: max(0, run.moves_left() - reserve - 3 * products)]
            if not calls:
                break

    def next_page_call(self, call_args: dict[str, Any], obs: Any) -> dict[str, Any] | None:
        paging = self.run.watch.page_arg()
        if not paging:
            return None
        field, per_page, lowest, highest = paging
        pages: list[dict[str, Any]] = []

        def gather(record: dict[str, Any]) -> None:
            for value in record.values():
                if isinstance(value, list) and value and all(isinstance(x, dict) for x in value):
                    pages.append(record)
                    return
        walk_dicts(obs, gather, 0, 3)
        if not pages:
            return None
        page = pages[0]
        shown = max(len(v) for v in page.values() if isinstance(v, list))
        markers = [v for k, v in page.items() if re.search(r"next|onward|cursor|bookmark|token|continu|more", str(k).lower())]
        matched = next((v for k, v in page.items() if is_number(v) and not isinstance(v, bool)
                        and re.search(r"count|total|matched|found|hits", str(k).lower())), None)
        if markers and all(m in (None, False, "", 0) for m in markers):
            return None
        if per_page == "token":
            token = next((m for m in markers if isinstance(m, str) and m.strip()), None)
            if not token:
                return None
            following = {**call_args, field: token}
        else:
            current = call_args.get(field, lowest)
            if not is_number(current):
                current = lowest
            upcoming = current + 1 if per_page else current + shown
            if highest is not None and upcoming > highest:
                return None
            if not markers:
                reached = (current - lowest + 1) * shown if per_page else current + shown
                if matched is None or reached >= matched:
                    return None
            following = {**call_args, field: int(upcoming) if float(upcoming).is_integer() else upcoming}
        fixed, faults = ArgCheck.execute(following, self.run.shapes.get(self.run.role("search") or "") or {})
        return None if faults else fixed

    def read_picks(self, picks: list[tuple[str, int]], avoid: set[str], keep: set[str] | None) -> list[tuple[str, int]]:
        run = self.run
        for _ in range(CODE_READS):
            unread = [ident for ident, _ in picks if ident not in run.archive.read]
            batch = run.shelf.batch_tool()
            as_batch = bool(batch) and len(unread) >= 2
            needed = 1 if as_batch else len(unread)
            if not unread or not (batch or run.role("details")):
                break
            if run.moves_left() < needed + 2 * len(picks) + 3 or run.wire.spare() <= 8:
                break
            if as_batch:
                slab = [(f"code_read_{run.wire.tick}", batch[0], {batch[1]: unread[: batch[2]]})]
            else:
                details = run.role("details")
                slab = []
                for n, call_args in enumerate(run.judge.details_call(ident) for ident in unread):
                    if call_args:
                        slab.append((f"code_read_{n}_{run.wire.tick}", details, call_args))
            if not slab:
                break
            run.send_chunk(slab)
            again = self.code_picks(avoid, keep)
            if again == picks:
                break
            picks = again
        return picks

    def code_put(self, picks: list[tuple[str, int]]) -> list[str] | None:
        run = self.run
        trolley = run.trolley
        tool = run.role("cart")
        wanted = {ident for ident, _ in picks}
        line_spec = trolley.line_spec()
        surplus = [line for line in trolley.units if line not in wanted]
        if line_spec:
            to_put = list(picks)
        else:
            to_put = [(ident, units) for ident, units in picks if abs(trolley.held(ident) - units) > 1e-9]
        removals = self.cart_drops(surplus)
        additions = self.cart_args(to_put) if to_put else []
        if removals is None or (to_put and not additions):
            return None
        calls = removals + additions
        if not calls:
            return []
        if run.moves_left() < len(calls) + 1:
            return None
        run.send_chunk([(f"code_cart_{n}_{run.wire.tick}", tool, call_args) for n, call_args in enumerate(calls)])
        replies = self.latest_replies(len(calls))
        missing: set[str] = set()
        held_view = None
        for call_args, obs in zip(calls, replies):
            units_seen = trolley.view_units(obs)
            if units_seen is not None:
                held_view = units_seen
            if not self.code_fault(obs):
                continue
            named = [ident for ident in wanted if ident in as_text(obs)]
            touched = [line for line, _ in trolley.lines(call_args) if line in wanted]
            missing.update(named or touched or (list(wanted) if line_spec else []))
        if len(replies) < len(calls):
            return None
        viewer = run.role("cart_view")
        if held_view is None and not missing and viewer and run.moves_left() >= 3:
            run.send_chunk([(f"code_view_{run.wire.tick}", run.role("cart_view"), {})])
            view = self.latest_replies(1)
            held_view = trolley.view_units(view[0]) if view else None
            if view and held_view is None:
                missing.update(ident for ident in wanted if ident not in trolley.units)
        if held_view is not None:
            missing.update(ident for ident in wanted if ident not in held_view)
        for ident in missing:
            trolley.units.pop(ident, None)
        return sorted(missing)

    def code_recheck(self, lines: list[str]) -> tuple[bool, set[str]]:
        run = self.run
        tool = run.shelf.stock_tool() or run.role("view") or run.role("details")
        if not lines or not tool:
            return False, set()
        if run.moves_left() < len(lines) + 2 or run.wire.spare() < 8:
            return False, set()
        slab = []
        for n, ident in enumerate(lines):
            call_args = run.shelf.lookup_call(ident, tool)
            if call_args:
                slab.append((f"code_check_{n}_{run.wire.tick}", tool, call_args))
        if not slab:
            return False, set()
        run.send_chunk(slab)
        table = self.code_rows()
        return True, {ident for ident in lines if (table.get(ident) or {}).get("stock") is False}

    def read_news(self, lines: list[str]) -> None:
        run = self.run
        queued = [line for group, _ in run.needs.pending or [] for line in group]
        for line in list(lines) + queued:
            if line in self.news_read:
                continue
            self.news_read.add(line)
            amount, _ = self.budget_in(line)
            if not is_number(amount) or amount <= 0:
                continue
            run.needs.spend_cap = amount
            run.log.append({"kind": "code_news", "budget": amount})

    def code_cart(self, avoid: set[str] | None = None, keep: set[str] | None = None) -> bool:
        run = self.run
        wire = run.wire
        tool, finder = run.role("cart"), run.role("search")
        if not tool or wire.done or wire.broke or run.moves_left() < 2 or wire.spare() < 4:
            return False
        wire.mute = True
        try:
            return self.shop_rounds(set(avoid or ()), keep, finder)
        except Exception as err:
            run.log.append({"kind": "code_finish_error", "error": f"{type(err).__name__}: {err}"[:300]})
            return False

    def shop_rounds(self, avoid: set[str], keep: set[str] | None, finder: str | None) -> bool:
        run = self.run
        products = max(1, len(run.needs.items()))
        for _ in range(4):
            picks = self.code_picks(avoid, keep)
            unfit = not picks or not all(self.code_fit.get(ident) for ident, _ in picks)
            if finder and not self.code_searched and run.moves_left() >= 4 and unfit:
                self.code_search(products)
                picks = self.code_picks(avoid, keep)
            elif finder and self.code_more and run.moves_left() >= 6 + 2 * products and (
                    unfit or run.needs.priority() == "cheapest"):
                self.code_search(products, True)
                picks = self.code_picks(avoid, keep)
            picks = self.read_picks(picks, avoid, keep)
            if not picks:
                run.log.append({"kind": "code_finish", "picks": [], "result": "nothing to order"})
                return False
            faulty = self.code_put(picks)
            if faulty is None:
                run.log.append({"kind": "code_finish", "picks": picks, "result": "no cart call"})
                return False
            if not faulty:
                if run.moves_left() >= len(picks) + 3:
                    _, sold = self.code_recheck([ident for ident, _ in picks])
                else:
                    sold = set()
                faulty = sorted(sold)
            if not faulty:
                run.log.append({"kind": "code_finish", "picks": picks, "cart": dict(run.trolley.units)})
                return bool(run.trolley.units)
            avoid |= set(faulty)
            run.log.append({"kind": "code_swap", "lines": faulty})
        return False

    def order_move(self, tool: str) -> dict[str, Any]:
        return {"name": tool, "args": self.schema_args(tool, "Order the cart.", "order")}

    def decline_move(self, tool: str) -> dict[str, Any]:
        return {"name": tool, "args": self.schema_args(tool, SORRY, "decline")}

    def stop_finishing(self) -> bool:
        wire = self.run.wire
        return wire.done or wire.broke or self.run.moves_left() <= 0 or wire.spare() < 2

    def finish_now(self) -> None:
        if self.stop_finishing():
            return
        run = self.run
        wire, trolley = run.wire, run.trolley
        order_tool, decline_tool = run.role("order"), run.role("decline")
        avoid: set[str] = set()
        keep: set[str] | None = None
        move: dict[str, Any] | None = None
        kind = None
        for turn in range(CODE_FINISHES):
            if self.stop_finishing():
                return
            if move is None:
                sold_lines: set[str] = set()
                if order_tool and turn == 0 and trolley.units:
                    try:
                        wire.mute = True
                        held_before = list(trolley.units)
                        sold_lines = run.shelf.cart_gone(held_before)
                        if sold_lines:
                            avoid |= sold_lines
                            keep = set(held_before) - sold_lines
                            run.log.append({"kind": "code_dead", "lines": sorted(sold_lines)})
                    except Exception as err:
                        run.log.append({"kind": "cart_gone_error", "error": f"{type(err).__name__}: {err}"[:300]})
                        sold_lines = set()
                cart_ready = (order_tool and turn == 0 and not sold_lines and trolley.used
                              and (trolley.units or trolley.view_units(trolley.last_view) is None))
                if cart_ready and run.basket_short():
                    # a partial basket is not an order: the code fills the missing lines around the kept ones, or declines
                    cart_ready = False
                    keep = set(trolley.units) if keep is None else keep
                    run.log.append({"kind": "code_partial", "faults": run.basket_short()})
                if cart_ready or (order_tool and self.code_cart(avoid, keep)):
                    move = self.order_move(order_tool)
                elif order_tool and turn > 0 and self.again_ok(avoid):
                    move = self.order_move(order_tool)
                elif order_tool and run.shelf.last_basket(avoid):
                    move = self.order_move(order_tool)
                elif decline_tool:
                    move = self.decline_move(decline_tool)
                else:
                    return
            if run.call_kind(move["name"], move["args"]) == "order":
                # out of time: the go-ahead still goes first when a message and a step are left
                run.send_go_ahead(room=2)
            reply = wire.act([move])
            run.log.append({"kind": "forced_finish", "action": move, "reply": reply})
            kind = run.call_kind(move["name"], move["args"])
            if not reply or reply.get("finished") or wire.done or turn == CODE_FINISHES - 1 or run.moves_left() <= 1:
                break
            wire.mute = True
            lines = list(trolley.units)
            obs = reply.get("observation")
            run.absorb(move["name"], move["args"], obs, reply, True)
            self.read_news(run.cover.shopper_lines(move["name"], obs, True))
            move = None
            if kind != "order" or not lines:
                continue
            reread, sold = self.code_recheck(lines)
            if not reread:
                continue
            run.refused_lines.difference_update(set(lines) - sold)
            table = self.code_rows()
            total = sum(((table.get(line) or {}).get("price") or 0.0) * trolley.held(line) for line in lines)
            budget = run.needs.spend_cap
            over = is_number(budget) and total > budget
            if not sold and not over:
                move = self.order_move(order_tool)
            elif over:
                avoid, keep = set(sold), None
            else:
                avoid, keep = set(sold), set(lines) - sold
        if wire.done or not reply or reply.get("finished") or not decline_tool or kind == "decline" or run.moves_left() <= 0:
            return
        again = False
        try:
            again = bool(order_tool) and kind == "order" and self.again_ok(avoid)
        except Exception as err:
            run.log.append({"kind": "again_error", "error": f"{type(err).__name__}: {err}"[:300]})
        move = self.order_move(order_tool) if again else self.decline_move(decline_tool)
        run.log.append({"kind": "forced_finish", "action": move, "reply": wire.act([move])})

    def again_ok(self, avoid: set[str]) -> bool:
        held = set(self.run.trolley.units)
        ruled_out = set(avoid) | self.run.shelf.dead_lines()
        if not held or held & ruled_out:
            return False
        return bool(self.code_picks(ruled_out))

    def catch_up(self) -> None:
        run = self.run
        wire, needs = run.wire, run.needs
        try:
            if wire.mute or wire.spare() < CLOSING:
                return
            notes = []
            read_at = run.intake.read_at
            if (needs.reading.get("code_reading") and not self.reread and read_at is not None
                    and wire.last_reply > read_at and wire.spare() > CLOSING + 60):
                self.reread = True
                answer = wire.query_model([{"role": "user", "content": run.intake.read_content}], READER, 3000)
                fresh = json_object((answer or {}).get("content") or "") or {}
                usable = isinstance(fresh.get("roles"), dict)
                run.log.append({"kind": "re_read", "ok": usable})
                if usable:
                    run.intake.use_reading(fresh, True)
                    notes.append("[plan updated] LLMs reply once more and re-read the request; this new reading replaces the first:\n"
                                 + needs.overview(run.step_cap, run.msg_cap))
            queued = needs.pending or []
            if queued and wire.last_reply > needs.pending_at and wire.spare() >= CLOSING:
                needs.pending = []
                for group, question in queued[-3:]:
                    heard = needs.hear(group, question)
                    if heard:
                        notes.append("[the shopper's earlier lines, read now] " + heard)
            if not notes:
                return
            text = "\n".join(notes) + "\n" + run.status_line()
            newest = run.chat[-1] if run.chat else None
            if newest and newest.get("role") in ("user", "tool") and isinstance(newest.get("content"), str):
                newest["content"] += "\n" + text
            else:
                run.chat.append({"role": "user", "content": text})
        except Exception as err:
            run.log.append({"kind": "catch_up_error", "error": f"{type(err).__name__}: {err}"[:300]})


class Observer:
    def __init__(self, run):
        self.run = run
        self.hit_shelf = {}
        self.pointed_memo = None
        self.refusal_texts = None
        self.sweep_tried = set()
        self.sweep_turn = 0
        self.swept = 0

    def refused(self, call_name: str, obs: Any, resp: dict[str, Any]) -> bool:
        if isinstance(obs, dict) and "refusal" in obs:
            return True
        if call_name != self.run.role("order") or resp.get("finished"):
            return False
        return not (isinstance(obs, dict) and obs.get("error") in ("invalid_args", "unknown_tool"))

    @staticmethod
    def wordy_texts(obs: Any) -> list[str]:
        found: list[str] = []

        def collect(piece: Any, depth: int = 0) -> None:
            if depth > 6:
                return
            if isinstance(piece, dict):
                for value in piece.values():
                    collect(value, depth + 1)
            elif isinstance(piece, list):
                for value in piece[:20]:
                    collect(value, depth + 1)
            elif isinstance(piece, str) and len(piece.split()) >= 4:
                found.append(piece)
        collect(obs)
        return found

    def order_refusal_note(self, obs: Any) -> str:
        run = self.run
        run.refusal_n += 1
        run.order_vetted = False
        run.event_seen = True
        hint = ""
        try:
            hint = run.shelf.refusal_reread(obs)
        except Exception as err:
            run.log.append({"kind": "refusal_reread_error", "error": f"{type(err).__name__}: {err}"[:300]})
        run.refused_lines.update(run.trolley.units)
        self.refusal_texts = {" ".join(str(s).lower().split()) for s in self.wordy_texts(obs)}
        return ("ORDER REFUSED; the episode goes on and the cart stays as it was. Carry out every change the refusal names, "
                "repair the cart (drop any line that stopped being acceptable, such as a sold-out item), re-check stock and "
                "price of the new pick, and order again." + hint)

    def shopper_push_note(self, obs: dict[str, Any]) -> str | None:
        pushed: list[str] = []
        for value in obs.values():
            if isinstance(value, list) and value and all(isinstance(x, str) for x in value):
                long_strings(value, pushed)
        if not pushed:
            return None
        earlier = self.refusal_texts or set()
        self.refusal_texts = None
        if earlier and all(any(" ".join(str(s).lower().split()) in text for text in earlier) for s in pushed):
            return ("This shopper text matches what the refused finish already carried (handled above); it tells "
                    "nothing about this result.")
        self.run.event_seen = True
        return "A shopper message (a change) is included with this result. Apply it: " + " | ".join(pushed[:4])

    def notes_for(self, call_name: str, obs: Any, refused: bool) -> str:
        run = self.run
        remarks = []
        if isinstance(obs, dict) and obs.get("error") == "budget_exceeded":
            if call_name == run.role("message") or run.msg_cap is not None:
                run.msg_cap = run.msgs_sent if run.msg_cap is None else min(run.msg_cap, run.msgs_sent)
            remarks.append("That budget is used up; do not call this tool again.")
        if refused and call_name == run.role("decline") and run.last_kind == "decline":
            run.audit_done = False
            run.decline_vetted = False
            run.event_seen = True
            remarks.append("DECLINE REFUSED; the episode goes on because the store believes that something can still be bought. "
                           "Do what the refusal says, look once more at the nearest candidates (their own properties, stock "
                           "and price against the current requirements), and order the one that fits; decline again only if "
                           "each one truly fails.")
        elif refused:
            remarks.append(self.order_refusal_note(obs))
        elif isinstance(obs, dict) and len(obs) >= 2 and call_name != run.role("message"):
            pushed = self.shopper_push_note(obs)
            if pushed:
                remarks.append(pushed)
        return " ".join(remarks)

    @staticmethod
    def shelf_of(row: dict[str, Any]) -> str:
        for key, value in row.items():
            if not isinstance(value, str) or not 0 < len(value.strip()) <= 60:
                continue
            name = str(key).lower()
            if re.search(r"categor|shelf|department|aisle", name) and "id" not in name:
                return value.strip().lower()
        return ""

    @staticmethod
    def key_stock(key: Any, val: Any) -> bool | None:
        name = str(key).lower()
        if val is None or not re.search(r"stock|avail|on.?hand|on.?shelf", name):
            return None
        out = False
        if re.search(r"stock|avail", name):
            if re.search(r"out|sold", name):
                out = val is True
            elif val is False or (is_number(val) and val <= 0):
                out = True
            elif isinstance(val, str):
                out = bool(re.search(r"\b(?:out|sold|none|unavailable|not|no|false)\b", val.lower()))
        if re.search(r"on.?hand|on.?shelf", name) and (val is False or (is_number(val) and val <= 0)):
            out = True
        if isinstance(val, str) and re.search(r"back.?order|pre.?order|discontinu", val.lower()):
            out = True
        return not out

    @staticmethod
    def stock_state(row: dict[str, Any]) -> bool | None:
        says = [Observer.key_stock(key, val) for key, val in row.items()]
        says = [s for s in says if s is not None]
        if not says:
            return None
        return all(says)

    @staticmethod
    def sold_out(row: dict[str, Any]) -> bool:
        for key, val in row.items():
            if Observer.key_stock(key, val) is False:
                return True
        return False

    def read_label_tally(self) -> dict[str, int]:
        archive = self.run.archive
        tally: dict[str, int] = {}
        counted: set[str] = set()
        foreign = self.run.refdesk.ref_foreign()

        def count(record: dict[str, Any]) -> None:
            ident = next((value for key, value in record.items()
                          if isinstance(value, str) and id_field(key) and value in archive.read), None)
            if not ident or ident in counted or ident in foreign:
                return
            label = self.shelf_of(record)
            if label:
                counted.add(ident)
                tally[label] = tally.get(label, 0) + 1
        for record in archive.items:
            walk_dicts(record, count)
        return tally

    def shelf_labels(self) -> set[str]:
        tally = self.read_label_tally()
        most = max(tally.values(), default=0)
        leaders = {label for label, n in tally.items() if n == most}
        if not tally or not self.run.needs.items():
            return leaders
        try:
            homes = self.run.shelf.home_shelves()
            chosen: set[str] = set()
            for home in homes:
                known = [label for label in home if label in tally]
                if not known:
                    chosen |= set(home)
                    continue
                top = max(tally[label] for label in known)
                chosen |= {label for label in known if tally[label] == top}
            for label in sorted(tally, key=lambda s: -tally[s]):
                if all(homes) or len(chosen) >= len(homes):
                    break
                chosen.add(label)
            return chosen or leaders
        except Exception:
            return leaders

    @staticmethod
    def row_mark(row: dict[str, Any]) -> str:
        for whole_name in (True, False):
            for key, val in row.items():
                name = str(key).lower()
                matches = name == "id" if whole_name else name.endswith(("id", "key"))
                if matches and isinstance(val, (str, int)) and not isinstance(val, bool):
                    return str(val)
        return json.dumps(row, sort_keys=True, default=str)

    def search_keys(self, call_args: dict[str, Any]) -> tuple[list[tuple], tuple, bool, set]:
        run = self.run
        schema = run.shapes.get(run.role("search") or "")
        fields = schema.get("properties") if isinstance(schema, dict) else None
        if not isinstance(fields, dict):
            fields = {}

        def spec_of(field: str) -> dict[str, Any]:
            found = fields.get(field)
            return found if isinstance(found, dict) else {}
        texts, modes, paging = [], [], set()
        first_page = True
        token = self.token_key()
        for field, value in call_args.items():
            name = str(field).lower()
            if field == token and isinstance(value, str):
                paging.add(field)
                first_page = first_page and not value.strip()
                continue
            if isinstance(value, str):
                if spec_of(field).get("enum"):
                    modes.append((field, value))
                    continue
                stripped = (piece.strip("\"'()[]{},.;:!?|/") for piece in value.lower().split())
                words = tuple(dict.fromkeys(w for w in stripped if re.search(r"[a-z0-9]", w)))
                if words and all(re.fullmatch(r"[a-z0-9]{2,}", w) for w in words):
                    texts.append((field, words))
                continue
            if not is_number(value) or not re.search(r"page|offset|first|start|skip|from|cursor", name):
                continue
            if re.search(r"size|per_?page|limit|price|cost", name):
                continue
            paging.add(field)
            floor = spec_of(field).get("minimum", spec_of(field).get("default"))
            if not is_number(floor):
                floor = 1 if "page" in name else 0
            first_page = first_page and value <= floor
        return texts, tuple(sorted(modes)), first_page, paging

    @staticmethod
    def result_total(obs: dict[str, Any], rows: list) -> int:
        return max([v for v in obs.values() if isinstance(v, int) and not isinstance(v, bool)] + [len(rows)])

    def word_rule(self, call_args: dict[str, Any], obs: dict[str, Any], rows: list) -> str:
        run = self.run
        texts, mode, first_page, paging = self.search_keys(call_args)
        if not first_page:
            return run.word_kind
        total = self.result_total(obs, rows)
        for field, words in texts:
            context = (field, mode)
            rest = tuple(sorted((k, str(v)) for k, v in call_args.items() if k != field and k not in paging))
            subset_hint = False
            for old_context, old_rest, old_words, old_total in run.word_seen:
                if old_context != context or old_rest != rest or old_total == total:
                    continue
                if len(words) > len(old_words) and words[: len(old_words)] == old_words:
                    run.word_kind = "any" if total > old_total else "all"
                elif len(old_words) > len(words) and old_words[: len(words)] == words:
                    run.word_kind = "any" if total < old_total else "all"
                else:
                    depth = run.lead_words or None
                    ours, theirs = set(words[:depth]), set(old_words[:depth])
                    if (ours > theirs and total < old_total) or (theirs > ours and old_total < total):
                        subset_hint = True
            if subset_hint and not run.word_kind:
                run.word_kind = "all"
            run.word_seen.append((context, rest, words, total))
        return run.word_kind

    def lead_rule(self, call_args: dict[str, Any], obs: dict[str, Any], rows: list) -> int:
        run = self.run
        texts, mode, first_page, paging = self.search_keys(call_args)
        if not first_page:
            return run.lead_words
        total = self.result_total(obs, rows)
        marks = tuple(self.row_mark(r) for r in rows)
        for field, words in texts:
            context = (field, mode)
            rest = tuple(sorted((k, str(v)) for k, v in call_args.items() if k != field and k not in paging))
            bounds = run.lead_bounds.setdefault(context, [-1, set()])
            for old_context, old_rest, old_words, old_total, old_marks in run.lead_seen:
                if old_context != context or old_rest != rest or old_words == words:
                    continue
                shorter = min(len(words), len(old_words))
                shared = 0
                while shared < shorter and words[shared] == old_words[shared]:
                    shared += 1
                if total != old_total or marks != old_marks:
                    bounds[0] = max(bounds[0], shared)
                elif shared and shared == shorter and len(set(marks)) >= 3:
                    bounds[1].add(shared)
            run.lead_seen.append((context, rest, words, total, marks))
        claims = []
        for context, (proven, unread) in run.lead_bounds.items():
            above = [k for k in unread if k > proven]
            if above:
                claims.append((min(above), context))
        claims.sort()
        claims = [(k, context) for k, context in claims if k == run.lead_bounds[context][0] + 1]
        run.lead_words, run.lead_ctx = claims[0] if claims else (0, None)
        return run.lead_words

    def search_fields(self) -> dict[str, Any]:
        found = (self.run.shapes.get(self.run.role("search") or "") or {}).get("properties")
        return found if isinstance(found, dict) else {}

    def token_key(self) -> str | None:
        for field, spec in self.search_fields().items():
            if not isinstance(spec, dict) or spec.get("type") != "string" or spec.get("enum"):
                continue
            if re.search(r"bookmark|cursor|token|resume|continu|after|next|page|offset|start", str(field).lower()):
                return field
        return None

    def page_arg(self) -> tuple[str, Any, Any, float | None] | None:
        for field, spec in self.search_fields().items():
            name = str(field).lower()
            if not isinstance(spec, dict) or spec.get("type") not in ("integer", "number"):
                continue
            if not re.search(r"page|offset|first|start|skip|from|cursor", name) or re.search(r"size|per_?page|limit|price|cost", name):
                continue
            whole_pages = "page" in name and not re.search(r"offset|skip|first|start|from", name)
            first = spec.get("minimum", spec.get("default"))
            if not is_number(first):
                first = 1 if whole_pages else 0
            ceiling = spec.get("maximum") if is_number(spec.get("maximum")) else None
            return field, whole_pages, first, ceiling
        token = self.token_key()
        if not token:
            return None
        return token, "token", "", None

    @staticmethod
    def stem(word: str) -> str:
        if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
            return word[:-1]
        return word

    def shelf_rows(self) -> dict[str, tuple[float, str, str]]:
        run = self.run
        archive = run.archive
        labels, budget, units = self.shelf_labels(), run.needs.spend_cap, run.needs.quantity()
        rows: dict[str, tuple[float, str, str]] = {}
        if not labels:
            return rows

        def take(record: dict[str, Any]) -> None:
            ident = next((value for key, value in record.items()
                          if isinstance(value, str) and id_field(key) and value in archive.ids), None)
            if not ident or ident in run.refused_lines or self.shelf_of(record) not in labels:
                return
            if self.sold_out(record):
                rows.pop(ident, None)
                return
            price = next((value for key, value in record.items()
                          if is_number(value) and not isinstance(value, bool) and re.search(r"price|cost", str(key).lower())), None)
            title = record.get("title")
            if not (isinstance(title, str) and 0 < len(title) <= 300):
                title = next((value for key, value in record.items()
                              if isinstance(value, str) and 0 < len(value) <= 300
                              and re.search(r"title|heading|headline|name", str(key).lower())
                              and not re.search(r"categor|shelf|brand|make|manufactur|vendor|shop|seller|store|id", str(key).lower())), "")
            maker = " ".join(value for key, value in record.items()
                             if isinstance(value, str) and re.search(r"brand|make|manufactur|vendor", str(key).lower()))
            if price is not None:
                rows[ident] = (float(price), title, maker)
        for record in archive.items:
            walk_dicts(record, take)
        for ident, (price, title, maker) in list(rows.items()):
            latest = archive.price(ident)
            if latest is not None:
                price = latest
            if is_number(budget) and price * units > budget:
                rows.pop(ident)
                continue
            rows[ident] = (float(price), title, maker)
        return rows

    def label_queries(self) -> list[str]:
        queries: list[str] = []
        for label in sorted(self.shelf_labels()):
            groups, current = [], []
            for raw in label.lower().split():
                word = raw.strip(".,;:!?'\"()[]")
                if word not in LINK_WORDS and re.fullmatch(r"[a-z0-9][a-z0-9'-]*", word):
                    current.append(word)
                    continue
                if current:
                    groups.append(current)
                current = []
            if current:
                groups.append(current)
            flat = [w for group in groups for w in group]
            heads = [group[-1] for group in groups]
            if len(flat) >= 2:
                queries.append(" ".join(flat[:2]))
            queries.extend(heads)
            queries.extend(w for w in flat if w not in heads)
        return list(dict.fromkeys(queries))

    def kind_queries(self, pairs: bool) -> list[str]:
        label_stems = {self.stem(w) for label in self.shelf_labels() for w in re.findall(r"[a-z0-9]+", label)}
        titles: dict[str, tuple[float, str]] = {}
        for price, title, maker in self.shelf_rows().values():
            key = title.lower().strip()
            if not key:
                continue
            titles[key] = (min(price, titles.get(key, (price, ""))[0]), maker.lower())
        seen_in: dict[str, int] = {}
        lowest: dict[str, float] = {}
        written: dict[str, str] = {}
        for title, (price, maker) in titles.items():
            maker_words = set(re.findall(r"[a-z]+", maker))
            words = re.findall(r"[a-z]+", title)
            usable = [len(w) >= 3 and w not in LINK_WORDS and w not in maker_words for w in words]
            if pairs:
                grams = [(words[i], words[i + 1]) for i in range(len(words) - 1) if usable[i] and usable[i + 1]]
            else:
                grams = [(words[i],) for i in range(len(words)) if usable[i]]
            counted: set[str] = set()
            for gram in grams:
                joined = " ".join(self.stem(w) for w in gram)
                if joined in counted or all(self.stem(w) in label_stems for w in gram):
                    continue
                counted.add(joined)
                seen_in[joined] = seen_in.get(joined, 0) + 1
                lowest[joined] = min(lowest.get(joined, price), price)
                written.setdefault(joined, " ".join(gram))
        common = [g for g in seen_in if seen_in[g] >= 2]
        common.sort(key=lambda g: (lowest[g], -seen_in[g], g))
        return [written[g] for g in common]

    def sweep_next(self, turn: int) -> tuple[dict[str, Any] | None, bool]:
        run = self.run
        paging = self.page_arg()
        labels = self.shelf_labels()
        book = run.page_book
        if not paging or not labels or not book:
            return None, False
        page_field, _, first, _ = paging
        tried = self.sweep_tried
        listings = [w for w in book.values()
                    if w.get("id_order") and w["shelves"] and w["shelves"] <= labels and self.next_page(w) is not None]
        listing = max(listings, key=lambda w: w["total"], default=None)
        ranked = [w for w in book.values() if w.get("id_order") is False and self.free_text(w["args"])]
        if not ranked:
            ranked = [w for w in book.values() if w.get("id_order") is None and self.free_text(w["args"])]
        template, text_field = None, None
        if ranked:
            fields = self.search_fields() or {}
            model_args = ranked[-1]["args"]

            def kept(k: str, v: Any) -> bool:
                if not isinstance(v, str) or (fields.get(k) or {}).get("enum"):
                    return True
                return v.strip().lower() in labels or k == page_field
            free = [k for k, v in model_args.items() if not kept(k, v)]
            if run.lead_ctx and run.lead_ctx[0] in free:
                text_field = run.lead_ctx[0]
            else:
                text_field = max(free, key=lambda k: len(model_args[k].split()), default=None)
            if text_field:
                template = {k: v for k, v in model_args.items() if k != text_field and kept(k, v)}
        covered = set()
        for w in book.values():
            if first in w["read"] and w.get("id_order") is not True:
                lead = re.findall(r"[a-z0-9]{2,}", " ".join(self.free_text(w["args"])).lower())[:2]
                covered.add(tuple(self.stem(x) for x in lead))

        def as_call(text: str) -> dict[str, Any] | None:
            signature = tuple(self.stem(w) for w in re.findall(r"[a-z0-9]{2,}", text.lower())[:2])
            if template is None or not signature or signature in covered:
                return None
            call_args = {**template, text_field: text}
            if json.dumps(call_args, sort_keys=True, default=str) in tried:
                return None
            return call_args

        def from_texts(texts: list[str]) -> list[dict[str, Any]]:
            return [c for c in (as_call(t) for t in texts) if c]
        sources = [
            lambda: [{**listing["args"], page_field: self.next_page(listing)}] if listing else [],
            lambda: from_texts(self.kind_queries(True)[:6]),
            lambda: from_texts(self.kind_queries(False)[:6]),
            lambda: from_texts(self.label_queries()),
        ]
        ceiling_call, finished = run.cover.ceiling_sweep(tried)
        if ceiling_call is not None:
            return ceiling_call, True
        if finished:
            return None, False
        for source in [0] + [1 + (turn + shift) % 3 for shift in range(3)]:
            fresh = [c for c in sources[source]() if json.dumps(c, sort_keys=True, default=str) not in tried]
            if fresh:
                return fresh[0], source == 0
        return None, False

    def sweep_room(self, reserve: int) -> bool:
        run = self.run
        return run.moves_left() > reserve and run.wire.spare() >= CLOSING + 40

    def sweep_step(self) -> bool:
        run = self.run
        needs = run.needs
        if needs.priority() != "cheapest" or not needs.let_go or needs.props_left() or not run.role("search") or run.wire.done:
            return False
        reserve = 3 + (1 if run.trolley.units else 0)
        if run.orders_tried == 0:
            reserve = max(reserve + 3, FIRST_ORDER_BY)
        if self.swept >= SWEEP_READS or not self.sweep_room(reserve):
            return False
        call_args, from_listing = self.sweep_next(self.sweep_turn)
        if call_args is None:
            return False
        self.swept += 1
        if not from_listing:
            self.sweep_turn += 1
        self.sweep_tried.add(json.dumps(call_args, sort_keys=True, default=str))
        proposal = {"id": f"sweep_{self.swept}_{run.wire.tick}", "type": "function",
                    "function": {"name": run.role("search"), "arguments": json.dumps(call_args)}}
        run.chat.append({"role": "assistant", "content": "With no property requirement left, the cheapest in-stock listing on the "
                         "shelf wins: I read one more slice of the shelf before choosing.", "tool_calls": [proposal]})
        run.covering = True
        try:
            replies = run.dispatch([proposal], scripted=True)
        finally:
            run.covering = False
        more = (self.swept < SWEEP_READS and self.sweep_room(reserve)
                and self.sweep_next(self.sweep_turn)[0] is not None)
        if not more and replies:
            cheapest = sorted(self.shelf_rows().items(), key=lambda kv: (kv[1][0], kv[0]))[:6]
            if cheapest:
                cid, text = replies[-1]
                listed = "; ".join(f"{i} '{t[:80]}' {c:g}" for i, (c, t, _) in cheapest)
                replies[-1] = (cid, text + "\n[shelf sweep done] Lowest-priced in-stock rows of the requested category under "
                               "the budget, gathered over all the pages seen: " + listed + ". Take the cheapest one unless a change rules "
                               "it out: make it the cart's only line, re-check its stock and price, and order it.")
        run.talk.post_replies(replies)
        return True

    def free_text(self, call_args: dict[str, Any]) -> list[str]:
        fields = self.search_fields()
        token = self.token_key()
        return [value for key, value in call_args.items()
                if isinstance(value, str) and not (fields.get(key) or {}).get("enum") and key != token]

    @staticmethod
    def next_token(obs: Any) -> str | None:
        if not isinstance(obs, dict):
            return None
        for key, value in obs.items():
            if not isinstance(value, str) or not value.strip():
                continue
            if re.search(r"next|onward|cursor|bookmark|token|continu|more", str(key).lower()):
                return value
        return None

    @staticmethod
    def natural_key(mark: str) -> list:
        return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", mark)]

    def book_page(self, call_args: dict[str, Any], rows: list, total: int, obs: Any = None) -> None:
        paging = self.page_arg()
        if not paging or not rows:
            return
        field, mode, first, _ = paging
        position = call_args.get(field, first)
        if mode == "token":
            position = position.strip() if isinstance(position, str) else ""
        elif not is_number(position):
            return
        rest = {k: v for k, v in call_args.items() if k != field}
        walk = self.run.page_book.setdefault(json.dumps(rest, sort_keys=True, default=str),
                                             {"args": rest, "read": set(), "size": 0, "total": 0, "shelves": set()})
        walk["read"].add(position)
        if mode == "token":
            walk.setdefault("tokens", {})[position] = self.next_token(obs)
        records = [r for r in rows if isinstance(r, dict)]
        walk["size"] = max(walk["size"], len(rows))
        walk["total"] = max(walk["total"], total)
        walk["shelves"] |= {self.shelf_of(r) for r in records} - {""}
        row_shelf = walk.setdefault("row_shelf", {})
        for record in records:
            row_shelf.setdefault(self.row_mark(record), self.shelf_of(record))
        if position == first:
            walk["first"] = tuple(self.row_mark(r) for r in records)
            if total <= len(rows):
                walk["every"] = records
        marks = [self.row_mark(r) for r in records]
        if len(marks) < 3:      # some stores show only 4 rows a page
            return
        ascending = marks == sorted(marks) or marks == sorted(marks, key=self.natural_key)
        walk["id_order"] = walk.get("id_order", True) is not False and ascending

    def next_page(self, walk: dict[str, Any]) -> Any:
        field, whole_pages, first, highest = self.page_arg()
        if whole_pages == "token":
            for token in (walk.get("tokens") or {}).values():
                if isinstance(token, str) and token not in walk["read"]:
                    return token
            return None
        start = min([first] + [v for v in walk["read"] if is_number(v)])
        size = max(1, walk["size"])
        stride = 1 if whole_pages else size
        for n in range(1, 400):
            candidate = start + n * stride
            if highest is not None and candidate > highest:
                return None
            if n * size >= walk["total"]:
                return None
            if candidate not in walk["read"]:
                return candidate
        return None

    @staticmethod
    def close_words(a: str, b: str) -> bool:
        return min(len(a), len(b)) >= 3 and (a.startswith(b) or b.startswith(a))

    def value_score(self, call_args: dict[str, Any]) -> int:
        run = self.run
        value_sets = []
        for need in run.needs.props_left():
            value = need.partition(":")[2]
            if not ref_text(value):
                value_sets.append(re.findall(r"[a-z0-9]{3,}", value.lower()))
        typed = [w for text in self.free_text(call_args) for w in re.findall(r"[a-z0-9]{2,}", text.lower())]
        if run.lead_words:
            typed = typed[: run.lead_words]
        return sum(1 for words in value_sets if any(self.close_words(t, v) for t in typed for v in words))

    def row_hits(self, rows: list, words: list[str] | None = None) -> list[str]:
        run = self.run
        archive = run.archive
        budget, units = run.needs.spend_cap, run.needs.quantity()
        labels = self.shelf_labels()
        found = []
        named = set()
        if run.needs.reading.get("references"):
            named = {w for t in words or [] for w in re.findall(r"[a-z0-9]+", str(t).lower())}
        pointers = run.ref_hits
        for row in rows:
            if not isinstance(row, dict):
                continue
            ident = self.row_mark(row)
            makers = []
            for key, value in row.items():
                lowered = str(key).lower()
                if isinstance(value, str) and value.strip() and re.search(r"shop|seller|store|vendor|brand|make|manufactur", lowered) \
                        and "id" not in lowered:
                    makers.append(set(re.findall(r"[a-z0-9]+", value.lower())))
            unread = ident in archive.ids and ident not in archive.read
            if named and unread and any(m and m <= named and m <= self.pointed_words() for m in makers):
                if ident not in pointers:
                    pointers.append(ident)
            price = next((value for key, value in row.items() if is_number(value) and re.search(r"price|cost", str(key).lower())), None)
            label = self.shelf_of(row)
            if not unread:
                continue
            if is_number(budget) and is_number(price) and price * units > budget:
                continue
            if self.sold_out(row) or (labels and label and label not in labels):
                continue
            found.append(ident)
            self.hit_shelf[ident] = label
        return found

    def pointed_words(self) -> set[str]:
        run = self.run
        archive = run.archive
        memo = self.pointed_memo
        if memo and memo[0] == len(archive.items):
            return memo[1]
        words: set[str] = set()
        try:
            for ref in run.needs.reading.get("references") or []:
                words |= set(re.findall(r"[a-z0-9]+", str(ref).lower()))

            def gather(record: dict[str, Any]) -> None:
                if self.row_mark(record) not in archive.read:
                    return
                seller_words: set[str] = set()
                bodies: list[str] = []
                for key, value in record.items():
                    if not isinstance(value, str):
                        continue
                    lowered = str(key).lower()
                    names_seller = (re.search(r"shop|seller|store|vendor|brand|make|manufactur", lowered) and "id" not in lowered
                                    and len(value) <= 40 and not re.search(r"note|text|desc|comment|detail|info|message|section", lowered))
                    if names_seller:
                        seller_words |= set(re.findall(r"[a-z0-9]+", value.lower()))
                    elif not re.search(r"title|heading|headline|name|label|caption", lowered):
                        bodies.append(value.lower())
                for body in bodies:
                    words.update(set(re.findall(r"[a-z0-9]+", body)) - seller_words)
            for record in archive.items:
                walk_dicts(record, gather)
        except Exception:
            pass
        self.pointed_memo = (len(archive.items), words)
        return words

    def open_pages(self, most: int = 2) -> list[dict[str, Any]]:
        paging = self.page_arg()
        if not paging:
            return []
        labels = self.shelf_labels()
        ranked = []
        for walk in self.run.page_book.values():
            following = self.next_page(walk)
            if following is None:
                continue
            if labels and walk["shelves"] and not walk["shelves"] & labels:
                continue
            score = self.value_score(walk["args"])
            if score:
                ranked.append((-score, walk["total"], {**walk["args"], paging[0]: following}))
        ranked.sort(key=lambda entry: (entry[0], entry[1]))
        return [call_args for _, _, call_args in ranked[:most]]

    @staticmethod
    def probe_near(a: str, b: str) -> bool:
        if any(c.isdigit() for c in a + b):
            if a.lstrip("0") == b.lstrip("0"):
                return True
            return a.isdigit() and b.startswith(a) and not b[len(a):len(a) + 1].isdigit()
        return min(len(a), len(b)) >= 3 and (a.startswith(b) or b.startswith(a))

    def value_probe(self, raw: list[str]) -> tuple[str, str]:
        run = self.run
        if run.word_kind == "any":
            return "", ""
        depth = run.lead_words or 2
        lead = [w for w in re.findall(r"[a-z0-9]+", " ".join(raw).lower()) if len(w) >= 2][:depth]
        product = " ".join(re.findall(r"[a-z0-9]+", str(run.needs.reading.get("product") or "").lower())[:3])
        wanted = []
        for need in run.needs.props_left():
            name, colon, value = need.partition(":")
            if not colon or ref_text(value):
                continue
            plain = re.split(r"[,;]", re.sub(r"\((?:stated|revealed[^)]*)\)", "", value))[0].strip().lower()
            wanted.append((name.strip(), plain))
        known = [w for _, value in wanted for w in re.findall(r"[a-z0-9]{2,}", value)] + product.split()
        if not lead or any(self.probe_near(w, k) for w in lead for k in known):
            return "", ""
        for name, value in wanted:
            if not value or not any(self.probe_near(w, n) for w in lead for n in re.findall(r"[a-z0-9]{2,}", name.lower())):
                continue
            digit = re.search(r"\d", value)
            core = value[digit.start():] if digit else value
            probe = f"{core.strip(' .;')} {product}".strip()
            head = [w for w in re.findall(r"[a-z0-9]+", probe.lower()) if len(w) >= 2][:depth]
            for walk in run.page_book.values():
                ran = [w for t in self.free_text(walk["args"]) for w in re.findall(r"[a-z0-9]+", t.lower()) if len(w) >= 2][:depth]
                if len(ran) == len(head) and all(self.probe_near(a, b) for a, b in zip(head, ran)):
                    return "", ""
            return name, probe
        return "", ""


class CheapWalk:
    def __init__(self, run):
        self.run = run
        self.decline_queue = None
        self.decline_read_done = None
        self.fit_held = None
        self.walked = set()

    def ran_query(self, text: str) -> bool:
        run = self.run
        watch = run.watch
        width = run.lead_words or 3
        wanted = sorted(watch.stem(w) for w in re.findall(r"[a-z0-9]+", str(text).lower())[:width])
        for walk in run.page_book.values():
            typed = " ".join(watch.free_text(walk["args"])).lower()
            if sorted(watch.stem(w) for w in re.findall(r"[a-z0-9]+", typed)[:width]) == wanted:
                return True
        return False

    def search_template(self) -> tuple[dict[str, Any], str] | None:
        run = self.run
        for walk in reversed(list(run.page_book.values())):
            call_args = walk["args"]
            if run.cover.cg_listing(call_args):
                continue
            texts = run.watch.free_text(call_args)
            text_fields = [k for k, v in call_args.items() if isinstance(v, str) and v in texts and v.strip()]
            if len(texts) != 1 or len(text_fields) != 1:
                continue
            return {k: v for k, v in call_args.items() if k != text_fields[0]}, text_fields[0]
        return None

    @staticmethod
    def trap_near(a: str, b: str) -> bool:
        if any(c.isdigit() for c in a + b):
            return a.lstrip("0") == b.lstrip("0")
        return min(len(a), len(b)) >= 3 and (a.startswith(b) or b.startswith(a))

    @staticmethod
    def token_parts(token: str) -> list[str]:
        return [p for p in re.findall(r"[a-z0-9]+", token) if len(p) >= 2]

    def word_trap(self, raw: list[str]) -> tuple[str, str]:
        run = self.run
        try:
            return self.find_trap(raw)
        except Exception as err:
            run.log.append({"kind": "trap_error", "error": f"{type(err).__name__}: {err}"[:300]})
            return "", ""

    def find_trap(self, raw: list[str]) -> tuple[str, str]:
        run = self.run
        needs = run.needs
        near, parts = self.trap_near, self.token_parts
        if run.word_kind == "any":
            return "", ""
        tokens = [t.strip("\"'()[]{},.;!?|").lower() for t in " ".join(raw).split()]
        tokens = [t for t in tokens if re.search(r"[a-z0-9]", t)]
        if run.lead_words:
            width = run.lead_words
        elif run.word_kind == "all":
            width = max(3, len(tokens))
        else:
            width = 3
        tokens = tokens[:width]
        needs_now = []
        for need in needs.props_left():
            head, colon, value = need.partition(":")
            if not colon or ref_text(value):
                continue
            needs_now.append((re.split(r"\s*\(", head)[0].lower(), head, re.sub(r"\((?:stated|revealed[^)]*)\)", "", value).lower()))
        value_words = {w for _, _, v in needs_now for w in re.findall(r"[a-z0-9]{2,}", v)}
        products = [entry["product"] for entry in needs.items()] or [str(needs.reading.get("product") or "")]
        kinds = set(re.findall(r"[a-z0-9]{3,}", " ".join(products + sorted(run.watch.shelf_labels())).lower())) - LINK_WORDS
        prop_names = {w for n, _, _ in needs_now for w in re.findall(r"[a-z]{3,}", n)} - LINK_WORDS
        dropped = {w for old in needs.let_go for w in re.findall(r"[a-z]{3,}", old.partition(":")[2].lower())} - LINK_WORDS
        flagged, reason = [], ""
        safe = value_words | kinds
        for token in tokens:
            pieces = parts(token)
            if not pieces or any(near(p, k) for p in pieces for k in safe):
                continue
            if any(near(p, n) for p in pieces for n in prop_names):
                flagged.append(token)
                reason = reason or "name"
            elif any(near(p, d) for p in pieces for d in dropped):
                flagged.append(token)
                reason = reason or "dropped"
        if not flagged:
            return "", ""
        kept = [t for t in tokens if t not in flagged]
        carried = [(tag, v) for _, tag, v in needs_now
                   if any(near(p, w) for t in kept for p in parts(t) for w in re.findall(r"[a-z0-9]{2,}", v))]
        if not carried:
            return "", ""
        if not any(near(p, k) for t in kept for p in parts(t) for k in kinds):
            slot = run.talk.product_of(carried[0][0]) if needs.items() else None
            name = products[slot] if slot is not None and slot < len(products) else products[0]
            kept += [w for w in re.findall(r"[a-z]{3,}", name.lower()) if w not in LINK_WORDS][-1:]
        query = " ".join(kept[:width])
        if not query or self.ran_query(query):
            return "", ""
        word = flagged[0]
        if reason == "name":
            note = (f"The word '{word}' in this query comes from a required property's NAME, not from a required value. As "
                    "the store needs every word, it keeps only listings whose fields or title repeat that name and misses "
                    "those whose field holds just the value.")
        else:
            note = (f"Query word '{word}' belongs only to a requirement the shopper DROPPED, so it will still keep only the "
                    "listings carrying it, though they no longer need to.")
        return note + f" Also search '{query}' and read every page of it.", query

    def combined_queries(self) -> list[dict[str, Any]]:
        run = self.run
        needs = run.needs
        template = self.search_template()
        if not template:
            return []
        several = needs.items()
        width = run.lead_words or 3
        version = str(needs.version)
        live = {k.rpartition("|")[0] for k in run.passed if k.rpartition("|")[2] == version} - run.refused_lines
        covered = {run.judge.shelf_known(i) for i in live} - {""}
        homes = run.shelf.home_shelves() if several else []
        built: list[dict[str, Any]] = []
        for k in (range(len(several)) if several else [None]):
            if several:
                if k >= len(homes) or not homes[k] or homes[k] & covered:
                    continue
            elif not run.intake.unverified():
                continue
            groups: list[list[str]] = []
            for need in needs.props_left():
                head, _, value = need.partition(":")
                if several and run.talk.product_of(head) not in (k, None):
                    continue
                value = re.sub(r"\((?:stated|revealed[^)]*)\)", "", value).strip().lower()
                if not value or re.search(r"\d|as shown|same as|page|redirect|refer|\bnot\b|\bno\b|without|except|anything|other than", value):
                    continue
                lead_value = re.split(r"[,;/]|\bor\b|\band\b", value)[0]
                words = [w for w in re.findall(r"[a-z]{2,}", lead_value) if w not in LINK_WORDS]
                if 1 <= len(words) <= 2:
                    groups.append(words)
            if len(groups) < 2:
                continue
            query: list[str] = []
            for words in groups:
                if len(query) + len(words) <= width:
                    query.extend(words)
            product = several[k]["product"] if several else str(needs.reading.get("product") or "")
            kind = [w for w in re.findall(r"[a-z]{3,}", product.lower()) if w not in LINK_WORDS][-1:]
            if kind and len(query) < width and kind[0] not in query:
                query.extend(kind)
            text = " ".join(query)
            if len(query) >= 2 and not self.ran_query(text):
                built.append({**template[0], template[1]: text})
        return built[:2]

    def arbitrate(self, thing: str, ruling: str) -> str:
        run = self.run
        try:
            reason = ""
            for entry in reversed(run.log):
                if isinstance(entry, dict) and entry.get("kind") == "audit" and isinstance(entry.get("result"), dict):
                    reason = str((entry.get("result") or {}).get("reason") or "")
                    break
            extra = ("\n\nCONFLICTING READINGS OF THIS ITEM. Goal: settle which reading of the same records is right. 1. The "
                     "decline audit judged that it meets every requirement" + (f" ({reason[:400]})" if reason else "")
                     + "; an earlier check reported: " + ruling[:500]
                     + "\n2. Decide from the records: a field giving the required value in other words, or a shade, qualifier or "
                     "spelling of the same named value, meets the requirement; a field giving another value fails it.")
            verdict = run.judge.check_item(thing, "ORDER", second=extra)
            if verdict:
                return verdict
            key = f"{thing}|{run.needs.version}"
            for memo in [k for k in run.rulings if k == key or k.startswith(key + "|")]:
                run.rulings[memo] = ""
            return ""
        except Exception as err:
            run.log.append({"kind": "arbitrate_error", "error": f"{type(err).__name__}: {err}"[:300]})
            return ruling

    def unrefuted(self) -> list[str]:
        run = self.run
        try:
            return self.decline_candidates()
        except Exception as err:
            run.log.append({"kind": "unrefuted_error", "error": f"{type(err).__name__}: {err}"[:300]})
            return []

    def decline_candidates(self) -> list[str]:
        run = self.run
        needs, archive, judge = run.needs, run.archive, run.judge
        several = needs.items()
        homes = run.shelf.home_shelves() if several else [set(run.watch.shelf_labels())]
        if not homes or not all(homes):
            return []
        version = str(needs.version)
        failed = {k.split("|")[0] for k, v in run.rulings.items() if v and k.split("|")[1:2] == [version]}
        live = {k.rpartition("|")[0] for k in run.passed if k.rpartition("|")[2] == version}
        reviewed = judge.audit_ids
        if run.audit_done:
            limit = -1 if judge.audit_reads is None else judge.audit_reads
            if reviewed is None or len(archive.read) <= limit:
                reviewed = set(archive.read)
        audited = set(reviewed or set()) - set(judge.audit_fits or [])
        sold: dict[str, bool] = {}

        def note_stock(record: dict[str, Any]) -> None:
            ident = run.watch.row_mark(record)
            if ident in archive.ids and any(re.search(r"stock|avail", str(key).lower()) for key in record):
                sold[ident] = run.watch.sold_out(record)
        for record in archive.items:
            walk_dicts(record, note_stock)
        budget = needs.spend_cap
        chosen: list[str] = []
        total = 0.0
        pointers = run.refdesk.ref_pointers()
        for k, home in enumerate(homes):
            units = several[k]["quantity"] if several else needs.quantity()
            pool = []
            for ident in sorted(archive.read):
                if ident in failed or ident in audited or ident in run.refused_lines or sold.get(ident):
                    continue
                if ident in chosen or ident in pointers or judge.shelf_known(ident) not in home:
                    continue
                price = archive.price(ident)
                if not is_number(price) or (is_number(budget) and price * units > budget):
                    continue
                pool.append((ident not in live, price, ident))
            if not pool:
                return []
            _, price, ident = min(pool)
            chosen.append(ident)
            total += price * units
        if is_number(budget) and total > budget:
            return []
        return chosen

    def fit_hold(self) -> str:
        run = self.run
        try:
            return self.hold_for_fit()
        except Exception as err:
            run.log.append({"kind": "fit_hold_error", "error": f"{type(err).__name__}: {err}"[:300]})
            return ""

    def hold_for_fit(self) -> str:
        run = self.run
        needs, archive, wire = run.needs, run.archive, run.wire
        if self.fit_held or run.trolley.units or run.moves_left() <= 4 or wire.spare() < CLOSING + 30:
            return ""
        candidates = [i for i in run.judge.audit_fits or [] if i in archive.ids and i not in run.refused_lines]
        if not candidates:
            return ""
        self.fit_held = True
        good = []
        for thing in candidates[:2]:
            if thing not in archive.read:
                continue
            verdict = run.judge.check_item(thing, "ORDER")
            if verdict and wire.spare() > CLOSING + 30:
                verdict = self.arbitrate(thing, verdict)
            if verdict or f"{thing}|{needs.version}" not in run.passed:
                continue
            good.append(thing)
            if not needs.items():
                break
        if not good:
            return ""
        if needs.items():
            live = {k.rpartition("|")[0] for k in run.passed if k.rpartition("|")[2] == str(needs.version)} - run.refused_lines
            covered = {run.judge.shelf_known(i) for i in live} - {""}
            if any(home and not home & covered for home in run.shelf.home_shelves()):
                return ""
        one = len(good) == 1
        target = "a verified line of every product" if needs.items() else good[0]
        return ("The decline audit named " + ", ".join(good) + " as a possible fit, and the independent item check read "
                + ("its" if one else "their") + " full record and found " + ("it meets" if one else "they meet")
                + " every requirement now in force. Examine " + ("it" if one else "them") + " before declining: if it fits, put "
                + target + " in the cart, re-read its stock and price, and order; decline if you find that it fails a "
                "requirement in force.")

    def id_store(self) -> bool:
        flags = [walk.get("id_order") for walk in self.run.page_book.values() if "id_order" in walk]
        return False not in flags and flags.count(True) >= 2

    def product_index(self, thing: str) -> int | None:
        run = self.run
        if not run.needs.items():
            return None
        label = run.judge.shelf_known(thing)
        if not label:
            return None
        for k, home in enumerate(run.shelf.home_shelves()):
            if label in home:
                return k
        return None

    def walk_home(self, k: int | None) -> set[str]:
        homes = self.run.shelf.home_shelves()
        slot = k if k is not None else 0
        home = set(homes[slot]) if homes and (k or 0) < len(homes) else set()
        if home:
            return home
        return set(self.run.watch.shelf_labels()) if k is None else set()

    @staticmethod
    def fig_amount(text: Any) -> tuple[str, str] | None:
        lowered = str(text).lower()
        if re.search(CH_BOUND, lowered):
            return None
        hit = re.match(r"\s*(\d+)\s+([a-z]+)", lowered)
        if hit is None:
            return None
        return hit.group(1), hit.group(2)

    @staticmethod
    def amount_part(text: Any) -> bool:
        lowered = str(text).lower()
        if not re.search(r"\d", lowered):
            return False
        return bool(re.search(CH_BOUND, lowered) or re.match(r"\s*\d+(?:[.,]\d+)?\s+[a-z]", lowered))

    def leading_words(self, call_args: dict[str, Any], pattern: str) -> list[str]:
        run = self.run
        found = [w for text in run.watch.free_text(call_args) for w in re.findall(pattern, text.lower())]
        return found[: run.lead_words] if run.lead_words else found

    def walk_carries(self, args: dict[str, Any], k: int | None, alts: bool = False) -> set[Any]:
        run = self.run
        near = run.watch.close_words
        per_product = bool(run.needs.items()) and k is not None
        words = self.leading_words(args, r"[a-z0-9]{2,}")
        tokens = self.leading_words(args, r"[a-z0-9]+")
        carried: set[Any] = set()
        for n, need in enumerate(run.needs.props_left()):
            if per_product and run.talk.product_of(need.partition(":")[0]) not in (k, None):
                continue
            value = re.sub(r"\((?:stated|revealed[^)]*)\)", "", need.partition(":")[2]).lower()
            for a, part in enumerate(re.split(r"[,;/]|\bor\b|\beither\b|\bnor\b", value)):
                mark = (n, a) if alts else n
                amount = self.fig_amount(part)
                if amount:
                    if amount[0] in tokens and any(t == amount[1] or near(t, amount[1]) for t in tokens):
                        carried.add(mark)
                    continue
                if self.amount_part(part):
                    continue
                if any(near(w, v) for w in words for v in re.findall(r"[a-z0-9]{3,}", part)):
                    carried.add(mark)
        return carried

    def walk_score(self, args: dict[str, Any], k: int | None) -> int:
        return len(self.walk_carries(args, k))

    def walk_options(self, k: int | None) -> list[tuple[int, str, dict[str, Any]]]:
        run = self.run
        watch = run.watch
        home = self.walk_home(k)
        if not home:
            return []
        depth = run.lead_words or None

        def lead(walk: dict[str, Any]) -> set[str]:
            typed = " ".join(watch.free_text(walk["args"])).lower()
            return {watch.stem(w) for w in re.findall(r"[a-z0-9]+", typed)[:depth]}
        label_stems = {watch.stem(w) for s in home for w in re.findall(r"[a-z0-9]{2,}", s)}
        per_product = bool(run.needs.items()) and k is not None
        value_stems = set()
        for need in run.needs.props_left():
            if per_product and run.talk.product_of(need.partition(":")[0]) not in (k, None):
                continue
            value_stems |= {watch.stem(w) for w in re.findall(r"[a-z0-9]{2,}", need.partition(":")[2].lower())}
        allowed = value_stems | label_stems
        book = run.page_book
        finished = []
        for walk in book.values():
            if walk.get("id_order") is not True or not walk.get("first") or watch.next_page(walk) is not None:
                continue
            if not lead(walk) or len(walk.get("row_shelf") or {}) < walk["total"]:
                continue
            finished.append((lead(walk), self.walk_carries(walk["args"], k, alts=True), run.cover.vl_mode(walk["args"])))
        best: dict[tuple, tuple[int, str, dict[str, Any]]] = {}
        for key, walk in book.items():
            if walk.get("id_order") is not True or not walk.get("first") or watch.next_page(walk) is None:
                continue
            row_labels = walk.get("row_shelf") or {}
            carries = self.walk_carries(walk["args"], k)
            if not carries or not any(s in home for s in row_labels.values()):
                continue
            words = lead(walk)
            pairs = self.walk_carries(walk["args"], k, alts=True)
            mode = run.cover.vl_mode(walk["args"])
            shadowed = False
            for seen_words, seen_pairs, seen_mode in finished:
                if seen_mode == mode and (seen_words <= words or (seen_words <= allowed and pairs <= seen_pairs)):
                    shadowed = True
                    break
            if shadowed:
                continue
            pages = -(-walk["total"] // max(1, walk["size"])) - len(walk["read"])
            twin = (walk["total"], walk["first"])
            if not 0 < pages <= 2 * WALK_PAGES:
                continue
            if twin not in best or pages < best[twin][0]:
                best[twin] = (pages, key, walk)
        options = list(best.values())
        if run.cover.all_words():
            def nested(p: tuple, o: tuple) -> bool:
                mine, other = lead(p[2]), lead(o[2])
                if mine < other and not (other - mine) <= allowed and p[0] <= WALK_PAGES:
                    return True
                return other < mine and (mine - other) <= allowed
            options = [o for o in options if not any(nested(p, o) for p in options if p is not o)]
        return sorted(options, key=lambda p: (p[0], -self.walk_score(p[2]["args"], k), p[1]))

    def walk_plan(self, budget: int, only: set | None = None) -> list[tuple[int | None, str, int]]:
        run = self.run
        if budget <= 0 or run.needs.priority() != "cheapest" or not run.role("search"):
            return []
        if not run.watch.page_arg() or not self.id_store():
            return []
        wanted = []
        for k in (list(range(len(run.needs.items()))) or [None]):
            if k in self.walked or (only is not None and k not in only):
                continue
            options = self.walk_options(k)
            if options:
                wanted.append((options[0][0], -1 if k is None else k, k, options[0][1]))
        wanted.sort(key=lambda w: (w[0], w[1]))
        plan, room = [], budget
        for pages, _, k, key in wanted:
            take = min(pages, WALK_PAGES, room)
            if take <= 0:
                break
            plan.append((k, key, take))
            room -= take
        return plan

    def walk_pick(self, k: int | None) -> tuple[str, float] | None:
        run = self.run
        sold = run.shelf.dead_lines()

        def mine(ident: str) -> bool:
            return k is None or self.product_index(ident) == k
        options = [i for i in run.trolley.units if mine(i) and i not in sold]
        if not options:
            version = str(run.needs.version)
            options = [key.rpartition("|")[0] for key in run.passed if key.rpartition("|")[2] == version]
            options = [i for i in options if i not in run.refused_lines and i not in sold and mine(i)]
        priced = []
        for ident in options:
            price = run.archive.price(ident)
            if is_number(price):
                priced.append((ident, float(price)))
        if not priced:
            return None
        return min(priced, key=lambda pair: (pair[1], pair[0]))

    def line_query(self, k: int | None) -> dict[str, Any] | None:
        run = self.run
        needs = run.needs
        pick = self.walk_pick(k)
        template = self.search_template()
        if not pick or not template:
            return None
        record: dict[str, Any] = {}

        def find(candidate: dict[str, Any]) -> None:
            nonlocal record
            if run.watch.row_mark(candidate) == pick[0] and run.shelf.title_of(candidate):
                record = candidate
        for item in run.archive.items:
            walk_dicts(item, find)
        maker = ""
        for key, value in record.items():
            lowered = str(key).lower()
            if isinstance(value, str) and value.strip() and re.search(r"brand|make|manufactur", lowered) and "id" not in lowered:
                maker = value
                break
        maker_words = re.findall(r"[a-z0-9]+", maker.lower())
        if not maker_words:
            return run.cover.title_line(k, record, template)
        skip = set(maker_words) | LINK_WORDS | {w for s in self.walk_home(k) for w in re.findall(r"[a-z]+", s)}
        skip |= {w for need in needs.props_left() for w in re.findall(r"[a-z]+", need.lower())}
        products = needs.items() or [{"product": str(needs.reading.get("product") or "")}]
        skip |= {w for entry in products for w in re.findall(r"[a-z]+", entry["product"].lower())}
        title_words = re.findall(r"[a-z][a-z'-]{3,}", run.shelf.title_of(record).lower())
        if maker_words[0] in title_words:
            title_words = title_words[title_words.index(maker_words[0]) + 1:]
        line_words = [w for w in title_words if w.strip("'-") not in skip and "'" not in w]
        if not line_words:
            return None
        text = f"{maker_words[0]} {line_words[0]}"
        if self.ran_query(text):
            return None
        return {**template[0], template[1]: text}

    def walk_rows(self, k: int | None, keys: list[str], most: int) -> list[str]:
        run = self.run
        needs, archive = run.needs, run.archive
        home = self.walk_home(k)
        book = run.page_book
        top = max([len(self.walk_carries(w["args"], k)) for w in book.values() if w.get("id_order") is True] or [0])
        shown: set[str] = set()
        for key, walk in book.items():
            carried = self.walk_carries(walk["args"], k) if walk.get("id_order") is True else set()
            if (carried and len(carried) == top) or key in keys:
                shown |= {i for i, s in (walk.get("row_shelf") or {}).items() if s in home}
        pick = self.walk_pick(k)
        if not pick:
            read_prices = sorted((float(archive.price(i)), i) for i in shown if i in archive.read and is_number(archive.price(i)))
            pick = (read_prices[0][1], read_prices[0][0]) if read_prices else None
        several = needs.items()
        units = several[k]["quantity"] if k is not None and k < len(several) else needs.quantity()
        budget = needs.spend_cap
        sold: dict[str, bool] = {}

        def note_stock(record: dict[str, Any]) -> None:
            if any(isinstance(v, str) and v in shown and id_field(key) for key, v in record.items()):
                sold[run.watch.row_mark(record)] = run.watch.sold_out(record)
        for item in archive.items:
            walk_dicts(item, note_stock)
        cheaper = []
        for ident in shown:
            price = archive.price(ident)
            if ident in archive.read or ident in run.refused_lines or sold.get(ident) or not is_number(price):
                continue
            if is_number(budget) and price * units > budget:
                continue
            if pick and price >= pick[1]:
                continue
            cheaper.append((price, ident))
        return [ident for _, ident in sorted(cheaper)[:most]]

    def page_digest(self, obs: Any) -> str:
        run = self.run
        watch = run.watch
        rows: list[dict[str, Any]] = []

        def gather(record: dict[str, Any]) -> None:
            if len(record) < 4:
                return
            if any(isinstance(v, str) and v in run.archive.ids and id_field(key) for key, v in record.items()):
                rows.append(record)
        walk_dicts(obs, gather, ceiling=3)
        lines = []
        for row in rows[:12]:
            price = next((v for key, v in row.items() if is_number(v) and re.search(r"price|cost", str(key).lower())), None)
            shown_price = price if price is not None else "?"
            sold = " SOLD OUT" if watch.sold_out(row) else ""
            lines.append(f"{watch.row_mark(row)} {shown_price}{sold} '{run.shelf.title_of(row)[:70]}' [{watch.shelf_of(row)}]")
        return "Page rows (id, price, title, category): " + ("; ".join(lines) if lines else "none")

    def walk_call(self, args: dict[str, Any], say: str) -> bool:
        run = self.run
        archive = run.archive
        proposal = {"id": f"walk_{len(run.chat)}_{run.wire.tick}", "type": "function",
                    "function": {"name": run.role("search"), "arguments": json.dumps(args)}}
        run.chat.append({"role": "assistant", "content": say, "tool_calls": [proposal]})
        records_before = len(archive.items)
        run.covering = True
        try:
            replies = run.dispatch([proposal], scripted=True)
        finally:
            run.covering = False
        if not replies:
            run.chat.append({"role": "tool", "tool_call_id": proposal["id"], "content": "Not run."})
            return False
        cid, text = replies[0]
        grew = len(archive.items) > records_before
        body = self.page_digest(archive.items[-1]) if grew else text
        note = text.split("\n[note] ", 1)[1] if "\n[note] " in text else ""
        if grew and note and re.search(r"shopper|Meaning:|REFUSED", note):
            body += "\n[note] " + note
        run.chat.append({"role": "tool", "tool_call_id": cid, "content": body})
        return grew

    def walk_open(self) -> bool:
        wire = self.run.wire
        return not wire.done and wire.spare() >= CLOSING + 30

    @staticmethod
    def interleave(pools: list[list[str]], most: int) -> list[str]:
        queue: list[str] = []
        while len(queue) < most and any(pools):
            for pool in pools:
                while pool and pool[0] in queue:
                    pool.pop(0)
                if pool and len(queue) < most:
                    queue.append(pool.pop(0))
        return queue

    def trim_queue(self, queue: list[str], batch: Any, floor: int) -> None:
        while queue and (1 if batch and len(queue) >= 2 else len(queue)) >= self.run.moves_left() - floor:
            queue.pop()

    def read_pages(self, plan: list[tuple[int | None, str, int]], field: str, floor: int, report: list[str]) -> bool:
        run = self.run
        wire = run.wire
        book = run.page_book
        progressed = False
        for k, walk_key, count in plan:
            self.walked.add(k)
            walk = book.get(walk_key)
            if not walk:
                continue
            query = " ".join(run.watch.free_text(walk["args"]))
            read = 0
            for _ in range(count):
                following = run.watch.next_page(walk)
                if following is None or run.moves_left() <= floor + 1 or wire.done or wire.broke or wire.spare() < CLOSING + 30:
                    break
                said = ("The cheapest is wanted and this store sorts search results by a fixed id, not by price, so I read the "
                        f"next page of '{query}'.")
                if not self.walk_call({**walk["args"], field: following}, said):
                    break
                read += 1
                progressed = True
            report.append(f"'{query}': {read} more page(s) read, {len(walk.get('row_shelf') or {})} of its {walk['total']} matches seen")
        return progressed

    def read_lines(self, plan: list[tuple[int | None, str, int]], field: str, floor: int, report: list[str],
                   lines: dict[Any, str]) -> bool:
        run = self.run
        wire = run.wire
        progressed = False
        for k, _, _ in plan:
            if run.moves_left() <= floor + 2 or not self.walk_open():
                break
            call_args = self.line_query(k)
            if not call_args:
                continue
            said = ("I also search the pick's product line (its maker plus the word that names the line): other sizes and "
                    "colours of a line are separate listings that may cost less.")
            if not self.walk_call(call_args, said):
                continue
            progressed = True
            lines[k] = json.dumps({a: v for a, v in call_args.items() if a != field}, sort_keys=True, default=str)
            line = run.page_book.get(lines[k])
            extra = 0
            while (line and extra < 2 and line["total"] <= 3 * max(1, line["size"]) and run.watch.next_page(line) is not None
                   and run.moves_left() > floor + 3 and not wire.done and wire.spare() > CLOSING + 30):
                if not self.walk_call({**line["args"], field: run.watch.next_page(line)}, "Next page of the line search."):
                    break
                extra += 1
            report.append(f"the pick's line '{' '.join(run.watch.free_text(call_args))}'")
        return progressed

    def run_walk(self, plan: list[tuple[int | None, str, int]], floor: int, ceil: list | None = None, repair: bool = False) -> bool:
        run = self.run
        paging = run.watch.page_arg()
        if not (plan or ceil) or not paging or not run.role("search"):
            return False
        field = paging[0]
        report: list[str] = []
        lines: dict[Any, str] = {}
        ran = self.read_pages(plan, field, floor, report)
        ran = self.read_lines(plan, field, floor, report, lines) or ran
        ceiling_rows: dict[Any, list[str]] = {}
        for k in dict.fromkeys(ceil or []):
            keep = floor if repair else max(floor, run.intake.order_due() + 2)
            if run.moves_left() <= keep + 1 or not self.walk_open():
                break
            outcome = run.cover.ceiling_walk(k, floor, repair)
            if outcome:
                ran = True
                report.append(outcome[0])
                ceiling_rows[k] = list(outcome[1])
        if not ran:
            return False
        batch = run.shelf.batch_tool()
        most = batch[2] if batch else (CEIL_OPENS if ceiling_rows else 2)
        pools = [self.walk_rows(k, [lines.get(k, "")], most) for k, _, _ in plan] + list(ceiling_rows.values())
        queue = self.interleave(pools, most)
        self.trim_queue(queue, batch, floor)
        homes = set().union(*(self.walk_home(k) for k in [k for k, _, _ in plan] + list(ceiling_rows)))
        run.rank_checked = {mark for mark in run.rank_checked if run.judge.shelf_known(mark.split("|")[0]) not in homes}
        run.order_vetted = False
        if queue:
            listed = ", ".join(f"{i} ({run.archive.price(i):g})" for i in queue)
            middle = ("Next, I open the cheapest unread in-stock rows within the budget that these searches showed below each "
                      "pick's price: " + listed + ". ")
        else:
            middle = "Not one unread in-stock row in the budget below the pick's price was seen. "
        summary = ("[cheapest walk] " + "; ".join(report) + ". " + middle
                   + "For every product, select the lowest-priced listing whose own fields satisfy every requirement in "
                   "force; the rank check runs again on the cart call and the order.")
        if queue:
            reads = run.reads
            reads.open_queue = queue
            reads.open_recheck = True
            reads.open_say = "The cheapest is wanted: I open the cheapest unread rows the walk showed."
            reads.open_purpose, reads.open_floor = "walk", floor
            run.judge.open_named()
        if run.chat and run.chat[-1].get("role") == "tool":
            tail = "" if queue else "\n" + run.status_line()
            run.chat[-1]["content"] = str(run.chat[-1]["content"]) + "\n" + summary + tail
        run.log.append({"kind": "walk", "plan": [[k, n] for k, _, n in plan], "ceiling": list(ceiling_rows), "opened": queue})
        return True

    def decline_fits(self) -> set[str]:
        run = self.run
        version = str(run.needs.version)
        live = {k.rpartition("|")[0] for k in run.passed if k.rpartition("|")[2] == version}
        failed = {k.split("|")[0] for k, v in run.rulings.items() if v and k.split("|")[1:2] == [version]}
        named = set(run.judge.audit_fits or []) - failed
        return (live | named) - run.refused_lines

    def decline_lacking(self) -> list[int | None]:
        run = self.run
        covered = {run.judge.shelf_known(i) for i in self.decline_fits()} - {""}
        if run.needs.items():
            return [k for k, home in enumerate(run.shelf.home_shelves()) if home and not home & covered]
        if set(run.watch.shelf_labels()) & covered:
            return []
        return [None]

    def decline_walks(self) -> list[str]:
        needs_now = self.run.needs.props_left()

        def worded(call_args: dict[str, Any], k: int | None) -> int:
            count = 0
            for n in self.walk_carries(call_args, k):
                if n < len(needs_now) and not re.search(r"as shown|same as|the page|sold by|redirect|refer", needs_now[n].lower()):
                    count += 1
            return count
        chosen: list[str] = []
        for k in self.decline_lacking():
            options = [o for o in self.walk_options(k) if worded(o[2]["args"], k)]
            options.sort(key=lambda p: (-worded(p[2]["args"], k), -self.walk_score(p[2]["args"], k), p[0], p[1]))
            for _, key, _ in options:
                if key not in chosen:
                    chosen.append(key)
                    break
        return chosen[:2]

    def row_pools(self, homes: list[tuple[set[str], int]], ids: set[str] | None, below: list[float] | None = None) -> list[list[str]]:
        run = self.run
        archive, watch = run.archive, run.watch
        budget = run.needs.spend_cap
        labels: dict[str, str] = {}
        sold: dict[str, bool] = {}

        def note(record: dict[str, Any]) -> None:
            ident = watch.row_mark(record)
            if ident not in archive.ids or not watch.shelf_of(record):
                return
            labels[ident] = watch.shelf_of(record)
            if any(re.search(r"stock|avail", str(key).lower()) for key in record):
                sold[ident] = watch.sold_out(record)
        for record in archive.items:
            walk_dicts(record, note)
        pools = []
        for n, (home, units) in enumerate(homes):
            ranked = []
            for ident, label in labels.items():
                price = archive.price(ident)
                if label not in home or ident in archive.read or ident in run.refused_lines or sold.get(ident):
                    continue
                if (ids is not None and ident not in ids) or not is_number(price):
                    continue
                if (is_number(budget) and price * units > budget) or (below is not None and not price < below[n]):
                    continue
                ranked.append((float(price), ident))
            pools.append([ident for _, ident in sorted(ranked)])
        return pools

    def over_budget_rows(self) -> list[list[str]]:
        run = self.run
        try:
            several, budget = run.needs.items(), run.needs.spend_cap
            if not several or run.needs.priority() == "cheapest" or not is_number(budget):
                return []
            homes = run.shelf.home_shelves()
            fits = self.decline_fits()
            floors: list[float] = []
            shares: list[float] = []
            total = 0.0
            for k, home in enumerate(homes):
                prices = [float(run.archive.price(i)) for i in fits
                          if run.judge.shelf_known(i) in home and is_number(run.archive.price(i))]
                if not home or not prices:
                    return []
                lowest = min(prices)
                floors.append(lowest)
                shares.append(lowest * several[k]["quantity"])
                total += lowest * several[k]["quantity"]
            if total <= budget:
                return []
            pools = self.row_pools([(home, several[k]["quantity"]) for k, home in enumerate(homes)], None, floors)
            order = sorted(range(len(pools)), key=lambda k: -shares[k])
            pools = [pools[k] for k in order]
            return pools if any(pools) else []
        except Exception as err:
            run.log.append({"kind": "over_budget_error", "error": f"{type(err).__name__}: {err}"[:300]})
            return []

    def decline_reads(self) -> str:
        run = self.run
        try:
            return self.queue_decline_reads()
        except Exception as err:
            run.log.append({"kind": "decline_reads_error", "error": f"{type(err).__name__}: {err}"[:300]})
            return ""

    def queue_decline_reads(self) -> str:
        run = self.run
        wire = run.wire
        batch = run.shelf.batch_tool()
        if self.decline_read_done or wire.spare() < CLOSING + 60 or not (run.role("details") or batch) or run.trolley.units:
            return ""
        opens = 1 if batch else 2
        room = min(3, run.moves_left() - opens - 6)
        keys = []
        if room > 0 and run.watch.page_arg() and run.role("search") and self.id_store():
            keys = self.decline_walks()
        cheaper = self.over_budget_rows() if run.moves_left() > opens + 6 else []
        if not keys and not cheaper:
            return ""
        self.decline_read_done = True
        self.decline_queue = (keys, room, 6)
        parts = []
        if keys:
            named = "; ".join(f"'{' '.join(run.watch.free_text(run.page_book[key]['args']))}'" for key in keys)
            parts.append("searches holding required values matched more listings than were read (" + named + "), and this "
                         "store returns results in a fixed id order rather than by fit or price, so fits are just as likely on "
                         f"unread pages as on read ones: the code now reads up to {room} of those pages for you and opens the "
                         "cheapest in-stock rows within the budget they show")
        if cheaper:
            parts.append("taken together, the listings found for every product go over the budget: the cheapest unread "
                         "in-stock rows priced below each product's pick are opened for you too")
        target = "one line per product, at its quantity" if run.needs.items() else "the item"
        return ("Before a decline, " + "; ".join(parts) + ". Weigh those records against the requirements now in force by "
                "meaning; buy what fits (" + target + "), and decline only if each one fails.")

    def run_decline_reads(self) -> bool:
        run = self.run
        wire = run.wire
        keys, most_pages, floor = self.decline_queue
        self.decline_queue = None
        paging = run.watch.page_arg()
        book = run.page_book
        batch = run.shelf.batch_tool()
        opens = 1 if batch else 2
        read = 0
        walked: list[str] = []
        for key in (keys if paging else []):
            walk = book.get(key)
            if not walk:
                continue
            walked.append(key)
            query = " ".join(run.watch.free_text(walk["args"]))
            while (read < most_pages and run.moves_left() > floor + opens and not wire.done and not wire.broke
                   and wire.spare() > CLOSING + 30):
                following = run.watch.next_page(walk)
                if following is None:
                    break
                said = (f"Before declining, I read the next page of '{query}': results here come in a fixed id order, "
                        "so fits may sit on unread pages.")
                if not self.walk_call({**walk["args"], paging[0]: following}, said):
                    break
                read += 1
        seen: set[str] = set()
        for key in walked:
            seen |= set((book.get(key) or {}).get("row_shelf") or {})
        several = run.needs.items()
        homes = []
        for k in self.decline_lacking():
            units = several[k]["quantity"] if k is not None and k < len(several) else run.needs.quantity()
            homes.append((self.walk_home(k), units))
        pools = (self.row_pools(homes, seen) if walked else []) + self.over_budget_rows()
        most = batch[2] if batch else 2
        queue = self.interleave(pools, most)
        self.trim_queue(queue, batch, floor)
        run.log.append({"kind": "decline_reads", "pages": read, "opened": queue})
        if queue and not wire.done:
            reads = run.reads
            reads.open_queue = queue
            reads.open_say = "Before declining, I open the cheapest unread in-budget rows those pages and picks show."
            reads.open_purpose, reads.open_floor = "decline", floor
            run.judge.open_named()
        elif run.chat and run.chat[-1].get("role") == "tool":
            run.chat[-1]["content"] = (str(run.chat[-1]["content"]) + "\nThese pages showed no unread in-stock row in the "
                                       "budget.\n" + run.status_line())
        return read > 0 or bool(queue)


class ShopperReplyStep:
    def perform(self, run) -> bool:
        pending = run.needs.reply_due
        if not pending:
            return False
        run.needs.reply_due = ""
        return run.talk.reply_back(pending)


class QueuedReadsStep:
    def perform(self, run) -> bool:
        return run.judge.open_named()


class ShelfSweepStep:
    def perform(self, run) -> bool:
        if run.asking and not run.talk.empty_plans:
            return False
        return run.watch.sweep_step()


class ValuePagesStep:
    def perform(self, run) -> bool:
        return run.cover.value_pages()


class PlannedQuestionStep:
    def perform(self, run) -> bool:
        return bool(run.asking) and run.talk.pose_planned()


class ReferenceHopStep:
    def perform(self, run) -> bool:
        return run.refdesk.ref_step()


class RevealSweepStep:
    def perform(self, run) -> bool:
        return run.cover.reveal_sweep()


class CheapWalkStep:
    def perform(self, run) -> bool:
        return run.cover.walk_step()


class ActorTurnStep:
    def __init__(self):
        self.idle = 0
        self.failures = 0
        self.seq_no = 0
        self.last_ok = monotonic()
        self.missed = 0

    def perform(self, run) -> bool:
        wire = run.wire
        models = ACTOR if run.stuck < STUCK_LIMIT else ACTOR[1:] + ACTOR[:1]
        msg = wire.query_model(run.chat, models, 2500, run.toolset)
        if msg is None:
            return self.after_miss(run, models)
        self.last_ok, self.missed = monotonic(), 0
        proposals, self.seq_no = run.tool_calls(msg, self.seq_no)
        content = msg.get("content") or ""
        if not isinstance(content, str):
            content = as_text(content)
        binding = {run.role("cart"), run.role("order"), run.role("decline")} - {None}
        if run.asking and any(c["function"]["name"] in binding for c in proposals) and run.talk.pose_planned(urge=True):
            return True
        if not proposals:
            self.idle += 1
            run.chat.append({"role": "assistant", "content": content or "(no tool call)"})
            run.chat.append({"role": "user", "content": "Every reply has to call at least one of the tools offered. Proceed.\n"
                             + run.status_line()})
            return self.idle < IDLE_LIMIT
        self.idle = 0
        run.log.append({"kind": "model", "content": content[:2000], "calls": [c["function"] for c in proposals]})
        run.chat.append({"role": "assistant", "content": content or None, "tool_calls": proposals})
        step_before = wire.tick
        run.talk.post_replies(run.dispatch(proposals))
        run.stuck = run.stuck + 1 if wire.tick == step_before else 0
        return True

    def after_miss(self, run, models: list[str]) -> bool:
        wire = run.wire
        self.missed += 1
        if not wire.usable(models, True):
            return False
        if monotonic() - self.last_ok > DOWN_AFTER and (self.missed >= 2 or (wire.down() and len(wire.call_models) >= 2)):
            return False
        if wire.call_fail in ("empty", "refused"):
            self.failures += 1
            if self.failures >= 2:
                return False
        return True


class PointingLine:
    def __init__(self, text, phrase, start, end, parsed):
        self.text = text
        self.phrase = phrase
        self.parsed = parsed
        self.outside = text[:start] + " " + text[end:]
        self.inside = text[start:end]

    def inside_only(self, prop):
        if not prop:
            return False
        return prop_named(prop, self.inside) and not prop_named(prop, self.outside)

    def describes_other(self, req, name):
        if self.inside_only(name(req)):
            return True
        words = set(ref_toks(str(req).partition(":")[2]))
        if not words:
            return False
        return words <= set(ref_toks(self.inside)) and not words & set(ref_toks(self.outside))


class ReferenceDesk:
    def __init__(self, run):
        self.run = run
        self.redirect_told = set()
        self.hop_calls = 0
        self.candidate_ids = set()
        self.index_memo = None
        self.index_seen_at = None
        self.steps_spent = 0
        self.pending_tells = None
        self.wait_logged = None
        self.seller_field_memo = None
        self.field_names_memo = None
        self.line_memo = {}
        self.line_calls = 0
        self.own_tried = set()
        self.own_values = {}

    # -- shared shortcuts
    @property
    def needs(self):
        return self.run.needs

    @property
    def archive(self):
        return self.run.archive

    def open_refs(self):
        return [entry for entry in self.run.refs or [] if entry.get("status") == "open"]

    @staticmethod
    def head_prop(head):
        return re.split(r"\s*\(", str(head))[0]

    def product_kinds(self):
        kinds = set()
        for entry in self.needs.items() or [{"product": str(self.needs.reading.get("product") or "")}]:
            kinds |= kind_words(entry["product"])
        return kinds

    # -- opening references
    def ref_register(self):
        reading = self.needs.reading
        raw_targets = reading.get("reference_targets")
        infos = [t for t in raw_targets if isinstance(t, dict)] if isinstance(raw_targets, list) else []
        raw_refs = reading.get("references")
        listed = {str(r) for r in raw_refs or []} if isinstance(raw_refs, list) else set()
        for written in list(self.needs.written):
            value = str(written).partition(":")[2]
            pointing = ref_text(value)
            if not pointing and not (str(written) in listed and re.search(DIFFER_WORDS, value.lower())):
                continue
            self.ref_add(str(written), infos, self.run.request, "request")

    def ref_untracked(self, refs, tracked):
        def followed(req):
            for entry in tracked:
                if self.ref_match(entry, req) and (entry.get("status") == "resolved" or entry.get("chain")):
                    return True
            return False
        try:
            return [req for req in refs if not followed(req)]
        except Exception:
            return refs

    def ref_take(self):
        try:
            for entry in self.run.refs:
                if entry["status"] != "resolved":
                    continue
                if any(self.ref_match(entry, req) and req != entry["now"] for req in self.needs.props_left()):
                    self.swap_in(entry, entry["now"])
            fresh = list(self.needs.ref_news or [])
            self.needs.ref_news = []
            for item in fresh:
                opened = self.ref_add(str(item.get("need") or ""), list(item.get("infos") or []), str(item.get("line") or ""), "reply")
                if opened:
                    self.list_reference(opened["req"])
        except Exception as err:
            self.run.log.append({"kind": "ref_take_error", "error": f"{type(err).__name__}: {err}"[:300]})

    def list_reference(self, req):
        reading = self.needs.reading
        known = reading.get("references")
        if not isinstance(known, list):
            known = reading["references"] = []
        if req not in known:
            known.append(req)

    def info_for(self, infos, prop, tag):
        for cand in infos:
            if not same_prop(str(cand.get("property") or ""), prop):
                continue
            if not tag or not cand.get("product") or not self.needs.items() or kind_words(cand.get("product")) & kind_words(tag):
                return cand
        return None

    def already_open(self, req, prop, tag):
        for known in self.run.refs:
            if known.get("for_item"):
                continue
            if known["req"] == req:
                return True
            if same_prop(known["prop"], prop) and kind_words(known["tag"]) == kind_words(tag):
                return True
        return False

    def relation_of(self, value, info, prop, source, target):
        stated = isinstance(info, dict)
        if stated and info.get("checked"):
            return info.get("relation") or "same", info.get("tolerance"), str(info.get("unit") or "")
        seller = str(target.get("seller") or "").strip()
        relation, tolerance, unit = ref_relation(value, seller)
        told = ref_relation_name(info.get("relation")) if stated else "same"
        if relation == "same" and told == "within" and is_number(info.get("tolerance")):
            return "within", float(info["tolerance"]), str(info.get("unit") or "")
        if relation != "same":
            return relation, tolerance, unit
        # only words placed before the seller name (after the property) count
        span = ref_span(prop, source)
        if seller and seller.lower() in span.lower():
            spanned = ref_relation(span, seller)
            if spanned[0] in ("differ", "counterpart"):
                return spanned
        return (told, None, "") if told != "within" else ("same", None, "")

    def ref_add(self, req, infos, source, origin):
        head, _, value = str(req).partition(":")
        prop = self.head_prop(head).strip()
        bracketed = re.findall(r"\(([^)]*)\)", head)
        tag = bracketed[-1].strip() if bracketed else ""
        if not prop or junk_key(prop) or self.already_open(req, prop, tag):
            return None
        info = self.info_for(infos, prop, tag)
        target = self.ref_target(info, value, source, prop)
        if not target:
            self.run.log.append({"kind": "ref_skip", "req": str(req)[:200]})
            return None
        relation, tolerance, unit = self.relation_of(value, info, prop, source, target)
        record = dict(req=str(req), head=head.strip(), prop=prop, tag=tag, relation=relation, tol=tolerance, unit=unit,
                      target=target, first=dict(target), chain=[], seen={self.target_key(target)}, judged=set(), queries=[],
                      pages=0, hops=0, tries=0, fails=0, status="open", value="", goes_with=[], now="", origin=origin,
                      unindexed=False)
        self.run.refs.append(record)
        self.run.log.append({"kind": "ref_open", "req": str(req)[:200], "target": target, "relation": relation, "origin": origin})
        return record

    def ref_target(self, info, value, source, prop):
        def seller_ok(name):
            return self.ref_seller_ok(name, source)
        checked = ref_checked(info, source, seller_ok)
        parsed = None
        for text in (ref_span(prop, source), value):
            got = ref_parse(text, seller_ok)
            if got and seller_ok(got["seller"]):
                parsed = got
                break
        chosen = checked or parsed
        if not chosen:
            return None
        details = list(chosen["details"])
        if not details and parsed and ref_name(parsed["seller"]) == ref_name(chosen["seller"]):
            details = list(parsed["details"])
        kind = chosen["kind"] or (parsed["kind"] if parsed else "")
        return {"seller": chosen["seller"], "kind": kind[:80], "details": details}

    def ref_seller_ok(self, seller, source):
        name = str(seller or "").strip()
        if len(re.sub(r"[^A-Za-z]", "", name)) < 3:
            return False
        pattern = r"(?<![a-z0-9])" + re.escape(name.lower()) + r"(?:'s)?(?![a-z0-9])"
        if not re.search(pattern, str(source or "").lower()):
            return False
        kinds = self.product_kinds()
        search_shape = self.run.shapes.get(self.run.role("search") or "") or {}
        for spec in (search_shape.get("properties") or {}).values():
            if isinstance(spec, dict) and isinstance(spec.get("enum"), list):
                for option in spec["enum"]:
                    kinds.update(kind_words(option))
        own = kind_words(name)
        return bool(own) and not own <= kinds

    @staticmethod
    def target_key(target):
        tokens = set()
        for detail in target.get("details") or []:
            tokens.update(ref_toks(detail))
        return ref_name(target.get("seller") or ""), tuple(sorted(tokens))

    @staticmethod
    def describe_target(target):
        text = f"the {target.get('kind') or 'listing'} sold by {target.get('seller')}"
        if target.get("details"):
            text += " (" + "; ".join(target.get("details") or []) + ")"
        return text

    def describe(self, ref):
        return self.describe_target(ref["target"])

    # -- the listings seen
    def listing_index(self):
        items = self.archive.items
        if self.index_memo and self.index_memo[0] == len(items):
            return self.index_memo[1]
        merged, newest = {}, {}
        mark_of = self.run.watch.row_mark
        known = self.archive.ids
        for position, record in enumerate(items):
            def take(fields, position=position):
                ident = mark_of(fields)
                if ident not in known:
                    return
                # small records still count when they hold a property map
                if len(fields) < 4 and not any(isinstance(v, dict) and len(v) >= 2 for v in fields.values()):
                    return
                merged.setdefault(ident, {}).update(fields)
                newest[ident] = position
            walk_dicts(record, take)
        self.index_memo = (len(items), merged)
        self.index_seen_at = newest
        return merged

    def sellers_of(self, row):
        learned = self.seller_fields()
        names = set()
        for key, val in row.items():
            if not isinstance(val, str) or not val.strip() or len(val) > 60:
                continue
            low = str(key).lower()
            if not (re.search(SELLER_KEYS, low) or str(key) in learned) or id_field(low):
                continue
            if re.search(r"note|text|desc|comment|detail|info|message|section", low):
                continue
            names.add(ref_name(val))
        return names - {""}

    def capitalised_names(self, rows):
        found = set()
        for row in rows.values():
            for sentence in re.split(r"(?<=[.!?;])\s+", self.note_text(row, desc=False)):
                streak = []
                for word in re.findall(r"[A-Za-z][A-Za-z'-]*", sentence)[1:] + [""]:
                    if word[:1].isupper() and re.search(r"[a-z]", word):
                        streak.append(word)
                        continue
                    for start in range(len(streak)):
                        for stop in range(start + 1, min(len(streak), start + 3) + 1):
                            found.add(ref_name(" ".join(streak[start:stop])))
                    streak = []
        return found

    def chased_sellers(self):
        names = set()
        for entry in self.run.refs or []:
            for side in (entry.get("target") or {}, entry.get("first") or {}):
                if isinstance(side, dict):
                    names.add(ref_name(side.get("seller")))
        return names - {""}

    def seller_fields(self):
        try:
            names = self.chased_sellers()
            stamp = (len(self.archive.items), tuple(sorted(names)))
            if self.seller_field_memo and self.seller_field_memo[0] == stamp:
                return self.seller_field_memo[1]
            rows = self.listing_index()
            vocabulary = set()
            for row in rows.values():
                for val in row.values():
                    if isinstance(val, dict):
                        vocabulary.update(ref_name(x) for x in val.values() if isinstance(x, str))
            noted = {w for w in self.capitalised_names(rows) if len(w) >= 4} - vocabulary
            skip = (r"title|heading|name|label|caption|categor|shelf|department|aisle|note|text|desc|comment|detail|info|"
                    r"message|section|url|image|summary|overview")
            fields = set()
            for row in rows.values():
                for key, val in row.items():
                    low = str(key).lower()
                    if not isinstance(val, str) or not 0 < len(val) <= 60 or id_field(low) or re.search(skip, low):
                        continue
                    if ref_name(val) in names or (val.strip()[:1].isupper() and len(val.split()) <= 3 and ref_name(val) in noted):
                        fields.add(str(key))
            self.seller_field_memo = (stamp, fields)
            return fields
        except Exception:
            return set()

    @staticmethod
    def note_text(row, desc=True):
        notes, overview = [], []
        for key, val in row.items():
            low = str(key).lower()
            if not isinstance(val, str) or len(val.split()) < 3 or low == "id" or low.endswith(("id", "key")):
                continue
            if re.search(r"title|heading|headline|name|label|caption|categor|shelf|department|url|image", low):
                continue
            if re.search(r"desc|overview|summary|about", low):
                overview.append(val)
            else:
                notes.append(val)
        if notes:
            return "\n".join(notes)
        return "\n".join(overview if desc else [])

    def token_parts(self, row):
        notes = re.sub(r"\b(?:not|no|never|without|nor)\b(?:\W+[A-Za-z0-9.:/&'-]+){1,3}", " ", self.note_text(row, desc=False), flags=re.I)
        names, values = [], [self.run.shelf.title_of(row), notes]
        maps = [v for v in row.values() if isinstance(v, dict)]
        if not maps:
            flat = {}
            for key, val in row.items():
                if isinstance(val, (str, int, float)) and not isinstance(val, bool) \
                        and not re.search(r"id$|title|shop|seller|brand|price|amount|desc|overview|note", str(key).lower()):
                    flat[key] = val
            maps = [flat]
        for fields in maps:
            for key, val in fields.items():
                names.append(re.sub(r"([a-z])([A-Z])", r"\1 \2", str(key)).replace("_", " "))
                if isinstance(val, (str, int, float)) and not isinstance(val, bool):
                    values.append(str(val))
        return set(ref_toks(" ".join(names))), set(ref_toks(" | ".join(values)))

    def own_figures_text(self, row):
        pieces = [self.run.shelf.title_of(row)]
        for val in row.values():
            if isinstance(val, dict):
                pieces += [str(x) for x in val.values() if isinstance(x, (str, int, float))]
        return " | ".join(pieces)

    def may_show(self, row, detail, parts=None):
        names, values = parts or self.token_parts(row)
        wanted = {t for t in ref_toks(detail) if len(t) >= 3 and not re.search(r"\d", t)} - names
        if wanted and not wanted & values:
            return False
        figures = ref_figs(detail)
        if not figures:
            return True
        written = ref_figs(self.own_figures_text(row))
        for figure, unit in figures:
            same_unit = [f for f, u in written if u == unit]
            if same_unit and not any(abs(f - figure) < 1e-9 for f in same_unit):
                return False
        return True

    @staticmethod
    def field_value(row, prop):
        for matcher in (strict_prop, same_prop):
            for val in row.values():
                if not isinstance(val, dict):
                    continue
                for key, x in val.items():
                    if isinstance(x, (str, int, float)) and not isinstance(x, bool) and str(x).strip() and matcher(key, prop):
                        return str(x).strip()
        return ""

    def remember_ids(self, ids):
        self.candidate_ids = set(self.candidate_ids or set()) | set(ids)

    def ref_foreign(self):
        try:
            ids = self.candidate_ids or set()
            if not ids:
                return set()
            shelves = self.run.shelf.home_shelves()
            if not shelves or not all(shelves):
                return set()
            homes = set().union(*shelves)
            rows = self.listing_index()
            in_cart = set(self.run.trolley.units)
            shelf_of = self.run.watch.shelf_of
            foreign = set()
            for ident in ids:
                if ident in in_cart:
                    continue
                label = shelf_of(rows.get(ident) or {})
                if label and shelf_of(rows.get(ident) or {}) not in homes:
                    foreign.add(ident)
            return foreign
        except Exception:
            return set()

    def candidates(self, ref):
        target = ref["target"]
        wanted = ref_name(target["seller"])
        kinds = kind_words(target["kind"])
        rows = self.listing_index()
        seen_at = self.index_seen_at or {}
        title_of, shelf_of = self.run.shelf.title_of, self.run.watch.shelf_of
        scored = []
        for ident, row in rows.items():
            if wanted not in self.sellers_of(row):
                continue
            blurb = " ".join(str(v) for k, v in row.items() if isinstance(v, str) and re.search(r"desc|overview|summary", str(k).lower()))
            hay = set(ref_toks(title_of(row) + " " + blurb))
            shown = 0
            for detail in target["details"]:
                toks = set(ref_toks(detail))
                if toks and toks <= hay:
                    shown += 1
            on_kind = 2 if kinds & kind_words(shelf_of(row)) else 0
            scored.append((-on_kind - shown, -seen_at.get(ident, 0), ident, shown))
        scored.sort()
        ref["shows"] = {ident: shown for _, _, ident, shown in scored}
        return [entry[2] for entry in scored]

    # -- how the code searches for a target
    @staticmethod
    def ignores_words(seen):
        for i, (words_a, rows_a) in enumerate(seen):
            for words_b, rows_b in seen[i + 1:]:
                if len(rows_a) >= 2 and rows_a == rows_b and not (words_a & words_b):
                    return True
        return False

    def search_modes(self, needed, labels):
        watch = self.run.watch
        groups = {}
        for order, entry in enumerate(self.run.page_book.values()):
            args = entry.get("args") if isinstance(entry.get("args"), dict) else {}
            texts = watch.free_text(args)
            text_keys = [k for k, v in args.items() if isinstance(v, str) and v in texts and v.strip()]
            if len(texts) != 1 or len(text_keys) != 1:
                continue
            others = {}
            for k, v in args.items():
                if k == text_keys[0]:
                    continue
                if k in needed or not (isinstance(v, str) and v.strip().lower() in labels):
                    others[k] = v
            slot = json.dumps(others, sort_keys=True, default=str)
            group = groups.setdefault(slot, {"tpl": (others, text_keys[0]), "seen": [], "at": order})
            stems = {watch.stem(w) for w in re.findall(r"[a-z0-9]+", texts[0].lower())}
            group["seen"].append((stems, tuple(entry.get("first") or ())))
        ranked, blind = [], set()
        for slot, group in groups.items():
            if self.ignores_words(group["seen"]):
                blind.add(slot)
            else:
                variety = len({tuple(sorted(words)) for words, _ in group["seen"]})
                ranked.append((-variety, group["at"], group["tpl"]))
        ranked.sort(key=lambda x: (x[0], x[1]))
        return [entry[2] for entry in ranked], blind

    def logged_template(self, needed, labels, blind, paging):
        search = self.run.role("search")
        free_text = self.run.watch.free_text
        for entry in reversed(self.run.log):
            if entry.get("kind") != "env":
                continue
            for act in reversed(entry.get("actions") or []):
                args = act.get("args") if act.get("name") == search else None
                if not isinstance(args, dict):
                    continue
                texts = [k for k, v in args.items() if isinstance(v, str) and v.strip() and v in free_text(args)]
                if len(texts) != 1:
                    continue
                others = {k: v for k, v in args.items() if k != texts[0] and not (paging and k == paging[0])}
                kept = {k: v for k, v in others.items() if k in needed or not (isinstance(v, str) and v.strip().lower() in labels)}
                if json.dumps(kept, sort_keys=True, default=str) in blind:
                    continue
                return others, texts[0]
        return None

    def search_template(self):
        shape = self.run.shapes.get(self.run.role("search") or "") or {}
        needed = set(shape.get("required") or []) if isinstance(shape.get("required"), list) else set()
        shelf_of = self.run.watch.shelf_of
        labels = {shelf_of(r) for r in self.listing_index().values()} - {""}
        modes, blind = self.search_modes(needed, labels)
        template = modes[0] if modes else None
        paging = self.run.watch.page_arg()
        if not template:
            template = self.logged_template(needed, labels, blind, paging)
        if template:
            others, key = template
            return {k: v for k, v in others.items() if k in needed or not (isinstance(v, str) and v.strip().lower() in labels)}, key
        props = shape.get("properties") if isinstance(shape.get("properties"), dict) else {}
        token = self.run.watch.token_key()
        texts = [k for k, p in props.items() if isinstance(p, dict) and p.get("type") == "string" and not p.get("enum") and k != token]
        if len(texts) != 1 or needed - {texts[0]}:
            return None
        return {}, texts[0]

    def known_words(self):
        known = set()
        for row in self.listing_index().values():
            for val in row.values():
                if isinstance(val, dict):
                    for key in val:
                        known.update(re.findall(r"[a-z]+", re.sub(r"([a-z])([A-Z])", r"\1 \2", str(key)).lower()))
        for req in self.needs.written + self.needs.uncovered:
            known.update(re.findall(r"[a-z]+", self.head_prop(str(req).partition(":")[0]).lower()))
        return known

    def detail_words(self, ref):
        known = self.known_words()
        details = ref["target"]["details"]
        seller_words = set(re.findall(r"[a-z0-9]+", str(ref["target"]["seller"]).lower()))
        mixed, numbers, plain = [], [], []
        for detail in details:
            previous = ""
            for whole in str(detail).lower().split():
                after_figure = bool(re.fullmatch(r"\d+(?:[.,]\d+)?", previous))
                previous = whole
                # skip unit words that follow a number
                for word in re.findall(r"[a-z0-9]+", whole):
                    if word in LINK_WORDS or word in seller_words or (after_figure and not re.search(r"\d", word)):
                        continue
                    if re.search(r"\d", word) and re.search(r"[a-z]", word):
                        mixed.append(word)
                    elif word.isdigit():
                        if len(word) >= 2:
                            numbers.append(word)
                    elif len(word) >= 3 and word == whole.strip(".,;:!?()[]{}\"'") and word not in known and word.rstrip("s") not in known:
                        plain.append(word)
        plain.sort(key=len, reverse=True)
        leading = self.value_words_first(ref, details, plain)
        plain = [w for w in leading if w in plain] + [w for w in plain if w not in leading]
        return list(dict.fromkeys(mixed + numbers + plain))

    def value_words_first(self, ref, details, plain):
        # a detail written as "<field> <value>": its value word goes first
        field_names = self.field_names()
        split = [[w for w in re.findall(r"[a-z0-9]+", str(d).lower()) if w not in LINK_WORDS] for d in details]
        all_pairs = ref.get("hops", 0) == 0 and len(split) >= 1 and all(len(parts) >= 2 for parts in split)
        leading = []
        for parts in split:
            if len(parts) < 2 or parts[-1] not in plain:
                continue
            if all_pairs or any(" ".join(parts[:j]) in field_names for j in range(1, len(parts))):
                leading.append(parts[-1])
        return leading

    def field_names(self):
        stamp = len(self.archive.items)
        if self.field_names_memo and self.field_names_memo[0] == stamp:
            return self.field_names_memo[1]
        names = set()

        def collect(fields):
            for val in fields.values():
                if not isinstance(val, dict) or sum(1 for x in val.values() if isinstance(x, (str, int, float))) < 2:
                    continue
                for key in val:
                    spaced = " ".join(re.findall(r"[a-z0-9]+", re.sub(r"([a-z])([A-Z])", r"\1 \2", str(key)).lower()))
                    if spaced:
                        names.add(spaced)
        for record in self.archive.items:
            walk_dicts(record, collect)
        self.field_names_memo = (stamp, names)
        return names

    def planned_queries(self, ref):
        width = max(2, self.run.lead_words or 3)
        target = ref["target"]
        seller = " ".join(re.findall(r"[A-Za-z0-9]+", re.sub(r"'s\b", "", str(target["seller"]))))
        words = self.detail_words(ref)
        kind = [w for w in re.findall(r"[a-z]{3,}", str(target["kind"]).lower()) if w not in LINK_WORDS][-1:]
        if ref.get("unindexed"):
            plans = [" ".join((words[:1] + kind)[:width])] if (words or kind) else []
        else:
            plans = [seller]
            if words and len(seller.split()) < width:
                plans.insert(0, seller + " " + words[0])
        return [p for p in plans if p.strip() and p.lower() not in ref["queries"]]

    def next_seller_page(self, ref):
        watch = self.run.watch
        paging = watch.page_arg()
        if not paging or ref["pages"] >= 3:
            return None
        wanted = ref_name(ref["target"]["seller"])
        for entry in reversed(list(self.run.page_book.values())):
            if not wanted or wanted not in ref_name(" ".join(watch.free_text(entry["args"]))):
                continue
            following = watch.next_page(entry)
            if following is not None:
                return {**entry["args"], paging[0]: following}
        return None

    # -- the step budget of the chase
    def ref_room(self):
        pending = self.open_refs()
        if pending and all(r.get("relation") == "differ" and not r.get("chain") for r in pending) \
                and self.steps_spent >= REF_DIFFER_BLIND:
            return False
        step_cap = self.run.step_cap
        limit = max(8, int(REF_STEPS * step_cap))
        floor = self.step_floor()
        pending = self.open_refs()
        if pending and all(r.get("relation") == "differ" for r in pending):
            limit = max(6, int(REF_DIFFER_STEPS * step_cap))
        wire = self.run.wire
        if self.steps_spent >= limit or wire.spare() <= CLOSING + 60 or wire.spent_out() or wire.done or wire.broke:
            return False
        if self.waits_for_others(floor):
            return False
        return self.run.moves_left() > floor or self.late_hop_room()

    def step_floor(self):
        reserve = max(FIRST_ORDER_BY, min(FIRST_ORDER_DUE, self.run.step_cap // 4 + 1)) + 10
        return max(reserve, self.run.intake.order_due() + 7)

    def product_for(self, ref):
        if not self.needs.items() or not ref.get("tag"):
            return None
        return self.run.talk.product_of(f"x ({ref['tag']})")

    def waits_for_others(self, floor=None):
        rows = self.needs.items()
        pending = self.open_refs()
        if not rows or not pending or any(r.get("relation") != "differ" or not r.get("chain") for r in pending):
            return False
        if floor is None:
            floor = self.step_floor()
        if self.run.moves_left() <= floor + 4:
            return False
        try:
            owners = {self.product_for(r) for r in pending}
            lacking = [k for k in range(len(rows)) if not any(e["fit"] for e in self.run.shelf.ed_pool(k))]
        except Exception as err:
            self.run.log.append({"kind": "ref_wait_error", "error": f"{type(err).__name__}: {err}"[:300]})
            return False
        others = [k for k in lacking if k not in owners - {None}]
        waiting = len(others) >= 2 if None in owners else bool(others)
        if waiting and self.wait_logged != tuple(others):
            self.wait_logged = tuple(others)
            self.run.log.append({"kind": "ref_wait", "lacking": others})
        return waiting

    def late_hop_room(self):
        if self.run.moves_left() <= self.run.intake.order_due() + 2:
            return False
        spent = self.steps_spent
        for ref in self.run.refs or []:
            target = ref.get("target") or {}
            named = str(target.get("seller") or "").strip() and str(target.get("kind") or "").strip() and target.get("details")
            if ref.get("status") != "open" or ref.get("hops", 0) < 1 or not named:
                continue
            began = ref.get("late_from")
            if began is not None:
                if spent - began < 3:
                    return True
                continue
            if ref.get("queries") or not self.others_verified(ref):
                continue
            ref["late_from"] = spent
            self.run.log.append({"kind": "ref_late", "prop": ref.get("prop"), "to": str(target.get("seller"))[:80]})
            return True
        return False

    def others_verified(self, ref):
        rows = self.needs.items()
        if not rows:
            return True
        try:
            bare = [k for k in range(len(rows)) if not any(e["verified"] and e["fit"] for e in self.run.shelf.ed_pool(k))]
        except Exception:
            return False
        own = self.product_for(ref)
        if own is None:
            return len(bare) <= 1
        return all(k == own for k in bare)

    def hop_budget_ok(self):
        wire = self.run.wire
        if self.hop_calls >= REF_CALLS or wire.spare() <= CLOSING + 40 or wire.spent_out():
            return False
        # hops stop at 80% of the spend cap
        return wire.spend < 0.8 * SPEND_CAP * wire.scale

    def ref_match(self, ref, req):
        # a chase for one listing's own property stands for no requirement
        if ref.get("for_item"):
            return False
        if str(req) in (ref["req"], ref.get("now")):
            return True
        head, _, value = str(req).partition(":")
        if ref_text(value):
            pointing = True
        elif ref.get("relation") == "counterpart":
            pointing = re.search(COUNTER_WORDS, value.lower())
        else:
            pointing = ref.get("relation") == "differ" and re.search(DIFFER_WORDS, value.lower())
        if not pointing or not same_prop(self.head_prop(head), ref["prop"]):
            return False
        bracketed = re.findall(r"\(([^)]*)\)", head)
        if not bracketed or not ref["tag"]:
            return True
        return bool(kind_words(bracketed[-1]) & kind_words(ref["tag"]))

    def ref_live(self, ref):
        def same(req):
            return same_prop(self.head_prop(str(req).partition(":")[0]), ref["prop"])
        if any(self.ref_match(ref, req) or same(req) for req in self.needs.props_left()):
            return True
        return not any(same(req) for req in self.needs.let_go)

    # -- one resolver turn
    def ref_step(self):
        wire = self.run.wire
        if not self.run.refs or wire.done or wire.broke:
            return False
        try:
            return self.resolver_turn()
        except Exception as err:
            self.run.log.append({"kind": "ref_error", "error": f"{type(err).__name__}: {err}"[:300]})
            for ref in self.run.refs:
                if ref["status"] == "open":
                    ref["status"], ref["why"] = "stuck", "the code's resolver failed"
            return False

    def live_open(self):
        still = []
        for ref in self.run.refs:
            if ref["status"] == "open" and not self.ref_live(ref):
                ref["status"] = "dropped"
                self.run.log.append({"kind": "ref_dropped", "req": ref["req"][:200]})
            if ref["status"] == "open":
                still.append(ref)
        return still

    @staticmethod
    def read_queue(pools, most):
        queue = []
        while len(queue) < most and any(pools):
            for pool in pools:
                while pool and pool[0] in queue:
                    pool.pop(0)
                if pool and len(queue) < most:
                    queue.append(pool.pop(0))
        return queue

    def read_pools(self, pending, most):
        read = self.archive.read
        pools = []
        for ref in pending:
            pool = [i for i in self.candidates(ref) if i not in read and i not in ref["judged"]]
            shows = ref.get("shows") or {}
            # candidates get opened after a detail search, if few, or if the leading title matches
            ready = bool(ref["queries"]) or not self.planned_queries(ref) or len(pool) <= most or any(shows.get(i) for i in pool[:1])
            pools.append(pool if ready else [])
        return pools

    def search_next(self, needy, pending, template, acted):
        for ref in needy or sorted(pending, key=lambda r: (r["hops"], len(r["queries"]))):
            for plan in self.planned_queries(ref):
                ref["queries"].append(plan.lower())
                if self.run.walk.ran_query(plan):
                    return True
                return self.search_target(ref, {**template[0], template[1]: plan}) or acted
            more = self.next_seller_page(ref)
            if more:
                ref["pages"] += 1
                return self.search_target(ref, more) or acted
        for ref in pending:
            if [i for i in self.candidates(ref) if i not in ref["judged"]] or self.planned_queries(ref) or self.next_seller_page(ref):
                continue
            self.give_up(ref, "the listing it points at was not found in the store's search")
            acted = True
        return acted

    def resolver_turn(self):
        pending = self.live_open()
        if not pending:
            return False
        acted = False
        start_tick = self.run.wire.tick
        for ref in sorted(pending, key=lambda r: r["hops"]):
            acted = self.judge_read(ref) or acted
        if self.run.wire.tick > start_tick:
            return True
        pending = [r for r in pending if r["status"] == "open"]
        if not pending or not self.ref_room():
            return acted
        batch = self.run.shelf.batch_tool()
        most = batch[2] if batch else 2
        pools = self.read_pools(pending, most)
        template = self.search_template()
        empty = [r for r, pool in zip(pending, pools) if not pool and self.planned_queries(r)] if template else []
        queue = self.read_queue(pools, most)
        needy = empty if queue and len(queue) < most else []
        if queue and not needy:
            return self.read_candidates(queue, pending) or acted
        if not template:
            return acted
        return self.search_next(needy, pending, template, acted)

    def read_candidates(self, queue, pending):
        run = self.run
        batch = run.shelf.batch_tool() if len(queue) >= 2 else None
        start = run.wire.tick
        heads = ", ".join(dict.fromkeys(r["head"] for r in pending))
        replies = run.judge.open_reads(queue, f"The request copies {heads} from a different listing: I open the ones that may be it.", batch)
        dropped = False
        if batch and replies and not run.batch_proven:
            if run.shelf.batch_ok(run.judge.last_reads or [], queue):
                run.batch_proven = True
            else:
                run.batch_spec, run.batch_spec_known, dropped = None, True, True
                run.log.append({"kind": "batch_dropped", "tool": batch[0]})
        if replies:
            run.talk.post_replies(replies)
        self.steps_spent += run.wire.tick - start
        self.remember_ids(queue)
        run.log.append({"kind": "ref_read", "ids": queue})
        if not dropped:
            missed = [i for i in queue if i not in self.archive.read]
            for ref in pending:
                ref["judged"].update(missed)
        return run.wire.tick > start

    def search_target(self, ref, args):
        run = self.run
        start = run.wire.tick
        got = run.walk.walk_call(args, f"The request copies {ref['head']} from a different listing ({self.describe(ref)}): searching for it.")
        self.steps_spent += run.wire.tick - start
        rows = []
        if got:
            mark_of, known = run.watch.row_mark, self.archive.ids

            def keep(fields):
                if mark_of(fields) in known and len(fields) >= 4:
                    rows.append(fields)
            walk_dicts(self.archive.items[-1], keep, ceiling=3)
        wanted = ref_name(ref["target"]["seller"])
        text = " ".join(run.watch.free_text(args))
        if rows and not any(wanted in self.sellers_of(r) for r in rows) and ref_name(text) == wanted:
            ref["unindexed"] = True
        run.log.append({"kind": "ref_search", "prop": ref["prop"], "query": text[:120], "rows": len(rows)})
        return run.wire.tick > start

    # -- judging the records read
    def judge_read(self, ref):
        target = ref["target"]
        read = self.archive.read
        cands = [i for i in self.candidates(ref) if i in read and i not in ref["judged"]]
        if not cands:
            return False
        rows = self.listing_index()
        by_seller = [i for i in self.candidates(ref) if i in read]
        parts = {i: self.token_parts(rows.get(i) or {}) for i in set(cands) | set(by_seller)}
        loose = ref.setdefault("loose", set())
        details = target["details"]

        def shows(ident, detail):
            return self.may_show(rows.get(ident) or {}, detail, parts[ident])
        if len(by_seller) >= 4:
            # a worded detail none of 4+ records shows cannot be tested by code
            loose |= {d for d in details if not any(shows(i, d) for i in by_seller)}

        def words_ok(ident):
            return all(shows(ident, d) for d in details if d not in loose)
        full = {}
        for ident in cands:
            names, values = parts[ident]
            full[ident] = sum(1 for d in details if set(ref_toks(d)) and set(ref_toks(d)) <= names | values)
        ranked = sorted(cands, key=lambda i: (not words_ok(i), -full[i], cands.index(i)))
        if not words_ok(ranked[0]) and self.more_to_read(ref):
            return False
        pick = [i for i in ranked if words_ok(i)][:4] or ranked[:4]
        exact = [i for i in pick if details and full[i] == len(details)]
        asked = exact if len(exact) == 1 else ([i for i in pick if words_ok(i)][:2] or pick[:1])
        outcome = self.settle_by_notes(ref, pick, self.notes_unread(ref, asked))
        if outcome is not None:
            return outcome
        if len(exact) == 1 and self.plain_record(ref, exact[0]):
            ref["judged"].update(pick)
            self.remember_ids(pick)
            self.resolve(ref, exact[0], self.field_value(rows.get(exact[0]) or {}, ref["prop"]), [])
            return True
        if not self.hop_budget_ok():
            ref["judged"].update(pick)
            self.give_up(ref, "no hop checks are left")
            return True
        self.apply_hop(ref, pick, self.ask_hop(ref, pick, ""))
        if ref["status"] == "open" and ref["target"] is target:
            ref["judged"].update(i for i in cands if not words_ok(i))
        return True

    def more_to_read(self, ref):
        if not (self.ref_room() or self.waits_for_others()):
            return False
        read = self.archive.read
        if [i for i in self.candidates(ref) if i not in read and i not in ref["judged"]]:
            return True
        return bool(self.planned_queries(ref) or self.next_seller_page(ref))

    def settle_by_notes(self, ref, pick, missing):
        if not missing:
            return None
        if self.ref_room() and self.read_notes(ref, missing):
            return True
        if self.waits_for_others():
            return False
        if not self.ref_room() and ref.get("relation") == "differ":
            # unread notes may point elsewhere: an exclusion stays unresolved
            ref["judged"].update(pick)
            self.give_up(ref, "the notes of the listing it points at could not be read in the steps left")
            return True
        return None

    def field_tokens(self, row, names):
        pieces = [self.run.shelf.title_of(row), self.run.watch.shelf_of(row)]
        for val in row.values():
            if isinstance(val, dict):
                pieces += [str(x) for x in val.values() if isinstance(x, (str, int, float))]
        return names | set(ref_toks(" | ".join(pieces)))

    def plain_record(self, ref, ident):
        row = self.listing_index().get(ident) or {}
        value = self.field_value(row, ref["prop"])
        notes = self.note_text(row, desc=False)
        if not value or self.names_property(ref["prop"], notes) or set(ref_toks(value)) & set(ref_toks(notes)):
            return False
        own = self.sellers_of(row)
        elsewhere = {s for other in self.listing_index().values() for s in self.sellers_of(other)} - own
        names, _ = self.token_parts(row)
        fields = self.field_tokens(row, names)
        for sentence in re.split(r"(?<=[.!?;])\s+", notes):
            for word in re.findall(r"[A-Za-z][A-Za-z'-]*", sentence)[1:]:
                low = ref_name(word)
                if low in elsewhere:
                    return False
                if word[:1].isupper() and len(low) >= 4 and low not in own and low not in fields:
                    return False
        return True

    def notes_reader(self):
        shelf = self.run.shelf
        learned = shelf.notes_tool()
        if learned:
            return learned
        details = self.run.role("details")
        if (shelf.note_keys or {}).get(details or ""):
            return None
        tool_keys = shelf.tool_keys or {}
        text_keys = shelf.text_keys or {}
        detail_keys = set(tool_keys.get(details or "") or set())
        row_keys = set(shelf.row_keys or set())
        caps = shelf.tool_caps or {}
        others = [t for t in shelf.lookup_tools() if t != details]
        extra = []
        for tool in others:
            if (set(tool_keys.get(tool) or set()) - detail_keys - row_keys) & set(text_keys.get(tool) or set()):
                extra.append(tool)
        if extra:
            return max(extra, key=lambda t: ((caps.get(t) or {}).get("calls", 0), t))
        for tool in (self.run.role("view"), shelf.stock_tool()):
            if tool in others and not (caps.get(tool) or {}).get("id", 0):
                return tool
        return None

    def notes_unread(self, ref, ids):
        tool = self.notes_reader()
        if not tool or not ids:
            return []
        seen = set((self.run.shelf.read_with or {}).get(tool) or set())
        tried = ref.setdefault("noted", set())
        return [i for i in dict.fromkeys(ids) if i not in seen and i not in tried][:2]

    def read_notes(self, ref, need):
        tool = self.notes_reader()
        if not tool or not need:
            return False
        run = self.run
        ref.setdefault("noted", set()).update(need)
        start = run.wire.tick
        replies = run.judge.open_reads(need, f"The request copies {ref['head']} from a different listing: I read the notes of the one "
                                       f"that may be it ({tool}), as a note may send the reader on.", None, tool)
        if replies:
            run.talk.post_replies(replies)
        self.steps_spent += run.wire.tick - start
        self.remember_ids(need)
        run.log.append({"kind": "ref_notes", "ids": need, "tool": tool})
        return run.wire.tick > start

    # -- differ references against listings
    def ref_pointers(self):
        pointed = set()
        for ref in self.run.refs or []:
            if ref.get("relation") != "differ" or ref.get("status") not in ("open", "stuck", "resolved") or not self.ref_live(ref):
                continue
            pointed.update(i for i in ref.get("chain") or [] if self.same_product(ref, i))
        return pointed

    def same_product(self, ref, ident):
        if not self.needs.items():
            return True
        shelf = self.run.judge.shelf_known(ident)
        product_of = self.run.talk.product_of
        k = product_of(f"x ({shelf})") if shelf else None
        if k is None or not ref.get("tag"):
            return True
        return k == product_of(f"x ({ref['tag']})")

    def ref_pointer_note(self, thing):
        for ref in self.run.refs or []:
            if ref.get("relation") != "differ" or thing not in (ref.get("chain") or []):
                continue
            if not self.ref_live(ref) or not self.same_product(ref, thing):
                continue
            what = f" ({ref['value']})" if ref.get("status") == "resolved" and ref.get("value") else ""
            return (f"{thing} is where the shopper's reference for {ref['head']} points, or a listing its notes pass "
                    f"through, so its {ref['prop']} is the very value{what} the shopper's {ref['prop']} must differ from: it cannot "
                    "meet that requirement")
        return ""

    def ref_excluded(self, fields, k):
        several = self.needs.items()
        for ref in self.run.refs or []:
            if ref.get("relation") != "differ" or ref.get("status") != "resolved" or not self.ref_live(ref):
                continue
            if several and ref.get("tag") and self.run.talk.product_of(f"x ({ref['tag']})") not in (k, None):
                continue
            excluded = set()
            for val in [ref.get("value")] + list(ref.get("goes_with") or []):
                if str(val or "").strip():
                    excluded.add(" ".join(ref_toks(val)))
            if any(same_prop(key, ref["prop"]) and " ".join(ref_toks(val)) in excluded for key, val in (fields or {}).items()):
                return True
        return False

    def redirect_in(self, sentence, prop, own, elsewhere, fields):
        words = list(re.finditer(r"[A-Za-z][A-Za-z'-]*", sentence))
        # blank the kind word in phrases like 'the lamp by Name'
        blanked = re.sub(r"[A-Za-z0-9]+(?=\s+by\s+[A-Z])", lambda h: " " * len(h.group(0)), sentence)
        first = next((m.start() for m in words if prop_named(prop, blanked[:m.end()])), None)
        if first is None:
            return False
        for n, m in enumerate(words[1:], 1):
            word = m.group(0)
            low = ref_name(word)
            if m.start() <= first or not low:
                continue
            other = low in elsewhere or (word[:1].isupper() and len(low) >= 4 and low not in own
                                         and not set(ref_toks(word)) & fields and low not in fields)
            negated = any(w.group(0).lower() in ("not", "nor", "never", "no", "neither") for w in words[max(0, n - 5): n])
            if other and not negated:
                return True
        return False

    def own_redirects(self, thing):
        try:
            rows = self.listing_index()
            row = rows.get(thing) or {}
            notes = self.note_text(row, desc=False)
            if not notes:
                return []
            own = self.sellers_of(row)
            elsewhere = {s for other in rows.values() for s in self.sellers_of(other)} - own
            names, _ = self.token_parts(row)
            fields = self.field_tokens(row, names)
            several = self.needs.items()
            shelf_of, product_of = self.run.watch.shelf_of, self.run.talk.product_of
            k = product_of(f"x ({shelf_of(row)})") if several and shelf_of(row) else None
            found = []
            for req in self.needs.props_left():
                head = str(req).partition(":")[0]
                if several and k is not None and product_of(head) not in (k, None):
                    continue
                prop = self.head_prop(head).strip()
                if not prop:
                    continue
                sentence = next((s for s in re.split(r"(?<=[.!?;])\s+", notes) if self.redirect_in(s, prop, own, elsewhere, fields)), None)
                if sentence is not None:
                    found.append((str(req), prop, sentence.strip()))
            return found
        except Exception:
            return []

    def chase_own(self, thing, found):
        # notes that send a required property elsewhere: the chain runs like a shopper's reference, its end counting for this listing
        for req, prop, sentence in found or []:
            key = (thing, prop.lower())
            if key in self.own_tried or len(self.own_tried) >= REF_OWN or not self.ref_room():
                continue
            self.own_tried.add(key)
            read = self.read_line(sentence, f"which {prop} does this listing have?", None, True)
            if not read:
                continue
            target = {"seller": read["seller"], "kind": read["kind"], "details": list(read["details"])}
            self.run.refs.append(dict(req=str(req), head=f"{prop} of {thing}", prop=prop, tag="", relation="same", tol=None, unit="",
                                      target=target, first=dict(target), chain=[thing], seen={self.target_key(target)}, judged=set(),
                                      queries=[], pages=0, hops=0, tries=0, fails=0, status="open", value="", goes_with=[], now="",
                                      origin="own", unindexed=False, for_item=thing))
            self.run.log.append({"kind": "own_chase", "id": thing, "prop": prop, "target": target})

    def own_stamp(self, thing):
        return str(len(self.own_values.get(thing) or {}))

    @staticmethod
    def own_meets(wanted, head, value):
        bare = re.sub(r"\((?:stated|revealed[^)]*)\)", " ", str(wanted)).strip()
        if not bare or ref_text(bare) or re.search(CH_BOUND, bare.lower()):
            return None
        if positive_value(bare, head):
            return CodeCloser.code_seen(bare, value)
        return not ch_writes(" ".join(re.sub(EXCLUDE_WORDS, " ", bare.lower()).split()), value)

    def own_clash(self, thing):
        values = self.own_values.get(thing) or {}
        if not values:
            return ""
        run = self.run
        label = run.watch.shelf_of(self.listing_index().get(thing) or {})
        k = run.talk.product_of(f"x ({label})") if self.needs.items() and label else None
        for req in self.needs.props_left():
            head, _, wanted = str(req).partition(":")
            if self.needs.items() and k is not None and run.talk.product_of(head) not in (k, None):
                continue
            for prop, value in values.items():
                if same_prop(self.head_prop(head), prop) and self.own_meets(wanted, head, value) is False:
                    return f"its {prop} is taken from another listing, which shows {value}, and that does not meet '{req}'"
        return ""

    def settle_own(self, ref, target, value):
        thing, prop = ref["for_item"], ref["prop"]
        run = self.run
        ref["chain"].append(target)
        ref.update(status="resolved", value=str(value).strip(), now=f"{prop} of {thing}: {str(value).strip()}")
        self.own_values.setdefault(thing, {})[prop] = str(value).strip()
        stamp = f"{thing}|{self.needs.version}"
        for memo in [m for m in run.rulings if m == stamp or m.startswith(stamp + "|")]:
            del run.rulings[memo]
        clash = self.own_clash(thing)
        if clash:
            run.passed.discard(stamp)
        run.log.append({"kind": "own_settled", "id": thing, "prop": prop, "value": str(value)[:120], "chain": list(ref["chain"]), "fails": bool(clash)})
        ending = ": it does not fit, so take it out of the cart if it is there and pick another listing" if clash else (
            f": that value counts as {thing}'s {prop}, whatever its own field says")
        self.tell_actor(f"{thing}'s notes send {prop} to another listing; the code followed them to {target}, which shows {value}{ending}.")

    def notes_remark(self, call_name, call_args):
        run = self.run
        quiet = (run.role("search"), run.role("cart"), run.role("cart_view"), run.role("message"))
        if call_name in quiet or call_name in run.finishers():
            return ""
        version = str(self.needs.version)
        remarks = []
        for ident in sorted(run.shelf.asked_ids(call_args)):
            key = f"{ident}|{version}"
            if ident not in run.trolley.units and key not in run.passed:
                continue
            found = self.own_redirects(ident) if key not in self.redirect_told else []
            if not found:
                continue
            self.chase_own(ident, found)
            self.redirect_told.add(key)
            run.passed.discard(key)
            run.log.append({"kind": "own_redirect", "id": ident, "props": [p for _, p, _ in found]})
            what = "; ".join(f"{p} (\"{s[:160]}\")" for _, p, s in found)
            quoted = "; ".join(f"'{r}'" for r, _, _ in found)
            tail = ". If they do, take it out of the cart and put another verified fit in." if ident in run.trolley.units else "."
            remarks.append(f"{ident}'s notes may point {what} at a different listing: then its own field does not count for "
                           + quoted + " (the grader uses the value of that other listing), so it meets "
                           + ("them" if len(found) > 1 else "it") + " only when that listing's record shows it; the item check runs again "
                           "before it is ordered" + tail)
        return " ".join(remarks)

    # -- the shopper's pointing lines
    def ask_line(self, text, qtext):
        wire = self.run.wire
        if self.line_calls >= REF_READS or wire.spent_out() or wire.spare() <= CLOSING:
            return None
        self.line_calls += 1
        prompt = REF_READ + "\n\nQUESTION: " + str(qtext or "none") + "\n\nLINE:\n" + text[:1200]
        answer = wire.consult(prompt, 900, LISTENER)
        self.run.log.append({"kind": "ref_read_line", "line": text[:200], "answer": answer})
        return answer

    def may_point(self, text, hinted, ok):
        if ref_text(text) or any(ref_checked(raw, text, ok) for raw in hinted or []):
            return True
        return re.search(DIFFER_WORDS + "|" + COUNTER_WORDS, text.lower()) is not None and _seller_hit(text, ok) is not None

    def read_line(self, line, qtext="", hinted=None, strict=False):
        text = str(line or "")
        if text in self.line_memo:
            return self.line_memo[text]

        def seller_ok(name):
            return self.ref_seller_ok(name, text)
        got = None
        answer = self.ask_line(text, qtext) if strict or self.may_point(text, hinted, seller_ok) else None
        pointers = answer.get("pointers") if isinstance(answer, dict) else None
        for raw in pointers if isinstance(pointers, list) else []:
            got = got or ref_checked(raw, text, seller_ok)
        if not got and not strict and self.may_point(text, hinted, seller_ok):
            for raw in hinted or []:
                got = got or ref_checked(raw, text, seller_ok)
            # the listener only half reads a pointer: its "same" is no finding
            if got and got["relation"] == "same":
                got.update(zip(("relation", "tolerance", "unit"), ref_relation(text, got["seller"])))
            parsed = None if got else ref_parse(text, seller_ok)
            got = got or (parsed if parsed and seller_ok(parsed["seller"]) else None)
        self.line_memo[text] = got
        return got

    def line_span(self, text, read):
        hit = re.search(r"(?<![A-Za-z'])" + re.escape(read["seller"]) + r"(?:'s)?(?![A-Za-z])", text, re.I)
        if not hit:
            return None
        quote = read.get("quote") or ""
        spot = text.lower().find(quote.lower()) if quote else -1
        before = text[:hit.start()]
        stops = [before.rfind(c) for c in (",", ";", "!", "?")] + [m.start() for m in re.finditer(r"[.](?:\s|$)", before)]
        cut = max(stops + [-1])
        clause = before[cut + 1:].lower()
        marks = [m.start() for pattern in (DIFFER_WORDS, COUNTER_WORDS, REF_WORDS, POINT_WORDS) for m in re.finditer(pattern, clause)]
        if not marks and read.get("by") != "read":
            return None
        start = cut + 1 + (min(marks) if marks else 0)
        if 0 <= spot <= hit.start() < spot + len(quote):
            start = spot
        after = text[hit.end():]
        aside = re.search(r"\(\s*or\b[^)]*\)", after)
        stop = re.search(r"[.!?;](?:\s|$)", after)
        if aside and (not stop or aside.start() < stop.start()):
            end = hit.end() + aside.end()
        elif stop:
            end = hit.end() + stop.start()
        else:
            end = hit.end() + len(after)
        return start, end

    def pointing_phrase(self, line, hinted=None, qtext=None):
        text = str(line or "")
        parsed = self.read_line(text, qtext, hinted)
        if not parsed:
            return None
        span = self.line_span(text, parsed)
        if not span:
            return None
        start, end = span
        phrase = text[start:end].strip(" ,;")
        return (phrase, start, end, parsed) if phrase else None

    def question_words(self, qtext, outside):
        if not qtext:
            return []
        kinds = set()
        for entry in self.needs.items() or [{"product": str(self.needs.reading.get("product") or "")}]:
            kinds.update(re.findall(r"[a-z]+", str(entry["product"]).lower()))
        tokens = re.findall(r"[a-z0-9]+", str(qtext).lower())
        runs = []
        for size in (3, 2, 1):
            for at in range(len(tokens) - size + 1):
                piece = tokens[at: at + size]
                if any(w in LINK_WORDS or len(w) < 3 or w in kinds or w.rstrip("s") in kinds for w in piece):
                    continue
                if prop_named(" ".join(piece), outside):
                    runs.append(" ".join(piece))
        return runs

    def ref_fix(self, answer, lines, qtext):
        masked = []
        given = answer.get("references")
        hinted = [r for r in given if isinstance(r, dict)] if isinstance(given, list) else []
        for line in lines:
            got = self.pointing_phrase(line, hinted, qtext)
            if not got:
                masked.append(line)
                continue
            phrase, start, end, parsed = got
            piece = PointingLine(line, phrase, start, end, parsed)
            masked.append(piece.outside)
            self.place_pointing_need(answer, piece, qtext)
        return masked

    def need_name(self, req):
        return self.head_prop(str(req).partition(":")[0]).strip()

    def line_property(self, piece, needs, refs, qtext):
        name = self.need_name
        outside = piece.outside
        asked = [name(a) for a in str(self.run.talk.asking_label or "").split(";") if name(a)]
        pointing = [name(n) for n in needs if ref_text(n.partition(":")[2]) or re.search(DIFFER_WORDS, n.partition(":")[2].lower())]
        tiers = [[str(r.get("property") or "") for r in refs], pointing, [name(n) for n in needs], asked,
                 sorted(self.field_names()) + [name(r) for r in self.needs.written], self.question_words(qtext, outside)]
        read = str(piece.parsed.get("property") or "")
        if read and (prop_named(read, piece.text) or prop_named(read, qtext or "")):
            return next((p for tier in tiers[:4] for p in tier if p and same_prop(p, read)), read)
        for tier in tiers:
            named = [p for p in tier if p and prop_named(p, outside)]
            if named:
                return max(named, key=lambda p: (prop_named(p, outside, exact=True), len(re.findall(r"[a-z0-9]+", p.lower())), len(p)))
        fallback = tiers[0] + tiers[1] + (asked if len(asked) == 1 else [])
        return next((p for p in fallback if p and not piece.inside_only(p)), "")

    def line_tag(self, piece, needs, prop, qtext, several):
        product_of = self.run.talk.product_of
        k = product_of(piece.outside)
        if k is None:
            tags = [t for n in needs if same_prop(self.need_name(n), prop) for t in re.findall(r"\(([^)]*)\)", n.partition(":")[0])]
            k = product_of(f"x ({tags[-1]})") if tags else None
        if k is None and qtext:
            k = product_of(qtext)
        return several[k]["product"] if k is not None else ""

    def kept_pointing(self, mine, relation, seller):
        for n in mine:
            value = n.partition(":")[2]
            if not (ref_text(value) or re.search(DIFFER_WORDS, value.lower())):
                continue
            if ref_relation(value, seller)[0] == relation and seller in ref_name(value):
                return n
        return None

    def place_pointing_need(self, answer, piece, qtext):
        raw_needs = answer.get("needs")
        needs = [n for n in raw_needs or [] if isinstance(n, str) and ":" in n] if isinstance(raw_needs, list) else []
        raw_refs = answer.get("references")
        refs = [r for r in raw_refs or [] if isinstance(r, dict)] if isinstance(raw_refs, list) else []
        name = self.need_name
        prop = self.line_property(piece, needs, refs, qtext)
        if not prop:
            self.run.log.append({"kind": "ref_line_unplaced", "line": piece.text[:200]})
            return
        several = self.needs.items()
        tag = self.line_tag(piece, needs, prop, qtext, several) if several else ""

        def same_item(req):
            tags = re.findall(r"\(([^)]*)\)", str(req).partition(":")[0])
            return not several or not tag or not tags or bool(kind_words(tags[-1]) & kind_words(tag))
        mine = [n for n in needs if same_prop(name(n), prop) and same_item(n)]
        relation, tolerance, unit = piece.parsed["relation"], piece.parsed["tolerance"], piece.parsed["unit"]
        seller = ref_name(piece.parsed["seller"])
        keep = self.kept_pointing(mine, relation, seller)
        head = mine[0].partition(":")[0].strip() if mine else (f"{prop} ({tag})" if tag else prop)
        if several and tag and not re.findall(r"\(([^)]*)\)", head):
            head = f"{head} ({tag})"
        # the need has to read as a pointer until it resolves, or its words would be taken for the value
        need = keep or f"{head}: {piece.phrase}" + ("" if ref_text(piece.phrase) else " (as shown on that page)")
        details = [n for n in needs if n not in mine and piece.describes_other(n, name)]
        spot = needs.index(mine[0]) if mine else len(needs)
        rest = [n for n in needs if n not in mine and n not in details]
        answer["needs"] = rest[:spot] + [need] + rest[spot:]
        self.trim_answer(answer, piece, prop, need, details, same_item)
        self.merge_entry(refs, prop, piece, relation, tolerance, unit, seller)
        answer["references"] = refs
        answer["ref_needs"] = list(answer.get("ref_needs") or []) + [need]
        self.run.log.append({"kind": "ref_line", "prop": prop, "tag": tag, "relation": relation, "need": need[:200], "kept": bool(keep),
                             "details_out": [d[:80] for d in details]})

    def trim_answer(self, answer, piece, prop, need, details, same_item):
        name = self.need_name
        current = answer.get("requirements_now")
        if isinstance(current, list):
            written = self.needs.written
            kept = []
            for req in current:
                if not isinstance(req, str):
                    continue
                if req not in written and (req in details or piece.describes_other(req, name)):
                    continue
                if same_prop(name(req), prop) and same_item(req) and req not in written:
                    continue
                kept.append(req)
            if need not in kept:
                kept.append(need)
            answer["requirements_now"] = kept
        if isinstance(answer.get("dropped"), list):
            answer["dropped"] = [d for d in answer["dropped"]
                                 if not (isinstance(d, str) and (piece.inside_only(name(d)) or piece.describes_other(d, name)))]
        if isinstance(answer.get("no_preference"), list):
            answer["no_preference"] = [p for p in answer["no_preference"]
                                       if not (isinstance(p, str) and (piece.inside_only(name(p)) or same_prop(name(p), prop)))]

    def merge_entry(self, refs, prop, piece, relation, tolerance, unit, seller):
        parsed = piece.parsed
        checked = parsed.get("by") == "read"
        entry = next((r for r in refs if same_prop(str(r.get("property") or ""), prop)), None)
        if entry is None:
            entry = {"property": prop}
            refs.append(entry)
        if checked or not self.ref_seller_ok(entry.get("seller"), piece.text) or ref_name(entry.get("seller")) != seller:
            entry.update(seller=parsed["seller"], kind=parsed["kind"], details=list(parsed["details"]))
        elif not entry.get("details"):
            entry["details"] = list(parsed["details"])
        entry.setdefault("kind", parsed["kind"])
        if checked or relation != "same" or not entry.get("relation"):
            entry.update(relation=relation, tolerance=tolerance, unit=unit, checked=checked)

    @staticmethod
    def names_property(prop, text):
        words = [w for w in prop_words(prop) if len(w) >= 3]
        if not words:
            return True
        tokens = set(re.findall(r"[a-z]+", str(text).lower()))
        for tok in tokens:
            for word in words:
                if tok == word or (len(word) >= 4 and len(tok) >= 4 and tok[:4] == word[:4]):
                    return True
        return False

    # -- the hop call
    def record_card(self, ident):
        row = self.listing_index().get(ident) or {}
        facts = {}
        for val in row.values():
            if not isinstance(val, dict):
                continue
            for key, x in list(val.items())[:40]:
                if isinstance(x, (str, int, float)) and not isinstance(x, bool):
                    facts[str(key)] = str(x)[:90]
        learned = self.seller_fields()
        sellers = set()
        for key, val in row.items():
            low = str(key).lower()
            if not isinstance(val, str) or not val.strip() or len(val) > 60 or id_field(low):
                continue
            if (re.search(SELLER_KEYS, low) or str(key) in learned) and not re.search(r"note|text|desc|comment|detail|info|message|section", low):
                sellers.add(str(val))
        card = {"id": ident, "sold by": " / ".join(sorted(sellers)), "category": self.run.watch.shelf_of(row),
                "title": self.run.shelf.title_of(row)[:160], "facts": facts, "notes": self.note_text(row)[:900]}
        return as_text(card, 5000)

    def ask_hop(self, ref, pick, reason):
        if not self.hop_budget_ok():
            return None
        self.hop_calls += 1
        prompt = REF_HOP.replace("PROPERTY", ref["prop"]).replace("DESCRIPTION", self.describe(ref))
        prompt += "\n\nRECORDS:\n" + "\n".join(self.record_card(i) for i in pick)
        if reason:
            prompt += "\n\nTHE PREVIOUS ANSWER WAS NOT USABLE: " + reason + " Answer again from the records."
        answer = self.run.wire.consult(prompt, 3000, CHECKER[:1])
        self.run.log.append({"kind": "ref_hop", "prop": ref["prop"], "ids": pick, "answer": answer})
        return answer

    def apply_hop(self, ref, pick, answer, again=True):
        if not isinstance(answer, dict):
            ref["fails"] += 1
            if ref["fails"] >= 2:
                ref["judged"].update(pick)
                self.give_up(ref, "the hop check gave no answer")
            return
        raw = answer.get("target")
        target = str(raw).strip() if isinstance(raw, (str, int)) and not isinstance(raw, bool) else ""
        raw_missing = answer.get("missing")
        missing = [str(x) for x in raw_missing or [] if str(x).strip()] if isinstance(raw_missing, list) else []
        if target not in pick or missing:
            ref["judged"].update(pick)
            self.remember_ids(pick)
            if target in pick:
                self.run.log.append({"kind": "ref_partial", "prop": ref["prop"], "target": target, "missing": missing[:4]})
            return
        where, problem, go, value, pairs = self.check_hop(ref, target, answer, pick)
        if problem and again and self.hop_budget_ok():
            self.apply_hop(ref, pick, self.ask_hop(ref, pick, problem), again=False)
            return
        self.remember_ids(pick)
        if problem:
            self.run.log.append({"kind": "ref_invalid", "prop": ref["prop"], "target": target, "problem": problem[:300]})
            if where == "target":
                ref["judged"].add(target)
            else:
                ref["judged"].update(pick)
                ref["chain"].append(target)
                self.give_up(ref, problem)
            return
        ref["judged"].update(pick)
        if target in ref["chain"]:
            self.give_up(ref, "its notes lead back to a listing already followed")
        elif go:
            self.follow(ref, target, go)
        else:
            self.resolve(ref, target, value, pairs)

    @staticmethod
    def clean_strings(raw):
        items = raw if isinstance(raw, list) else []
        return [str(x).strip() for x in items if isinstance(x, (str, int, float)) and not isinstance(x, bool) and str(x).strip()]

    def check_go_to(self, ref, target, go, row, notes, note_tokens):
        seller = str(go["seller"]).strip()
        pattern = r"(?<![a-z0-9])" + re.escape(seller.lower()) + r"(?:'s)?(?![a-z0-9])"
        if not re.search(pattern, notes.lower()):
            return "answer", f"the go_to seller '{seller}' is not written in {target}'s notes.", None, "", []
        sentences = re.split(r"(?<=[.!?])\s+", notes)
        if not any(re.search(pattern, s.lower()) and self.names_property(ref["prop"], s) for s in sentences):
            return "answer", f"the note naming {seller} is about another property, not {ref['prop']}.", None, "", []
        details = []
        for d in self.clean_strings(go.get("details")):
            toks = set(ref_toks(d))
            if toks and 2 * len(toks & note_tokens) >= len(toks):
                details.append(d)
        following = {"seller": seller, "kind": str(go.get("kind") or "").strip()[:80], "details": details[:6]}
        if ref_name(seller) in self.sellers_of(row) and self.target_key(following) == self.target_key(ref["target"]):
            return "answer", f"the go_to is {target} itself.", None, "", []
        return "", "", following, "", []

    def details_shown(self, ref, ident):
        names, values = self.token_parts(self.listing_index().get(ident) or {})
        return sum(1 for d in ref["target"]["details"] if set(ref_toks(d)) and set(ref_toks(d)) <= names | values)

    def check_hop(self, ref, target, answer, pick=None):
        row = self.listing_index().get(target) or {}
        names, values = self.token_parts(row)
        for d in ref["target"]["details"]:
            if d in (ref.get("loose") or set()):
                continue
            if not self.may_show(row, d, (names, values)):
                return "target", f"{target} does not show '{d}', so it is not the listing described.", None, "", []
        best = max(pick or [target], key=lambda ident: self.details_shown(ref, ident))
        if self.details_shown(ref, best) > self.details_shown(ref, target):
            return "target", f"{best} shows more of the details named than {target}.", None, "", []
        notes = self.note_text(row)
        note_tokens = set(ref_toks(notes))
        go = answer.get("go_to") if isinstance(answer.get("go_to"), dict) else None
        if go and str(go.get("seller") or "").strip():
            return self.check_go_to(ref, target, go, row, notes, note_tokens)
        value = str(answer.get("value") or "").strip()
        value_tokens = set(ref_toks(value))
        if not value or not value_tokens or not value_tokens <= values | names:
            return "answer", f"the value '{value}' is not written in {target}'s record.", None, "", []
        pairs = []
        for p in dict.fromkeys(self.clean_strings(answer.get("goes_with"))):
            toks = set(ref_toks(p))
            if toks and toks <= note_tokens and toks != value_tokens:
                pairs.append(p)
        return "", "", None, value, pairs[:3]

    # -- outcomes
    def follow(self, ref, target, go):
        ref["chain"].append(target)
        key = self.target_key(go)
        if key in ref["seen"]:
            self.give_up(ref, "its notes lead back to a listing already followed")
            return
        if ref["hops"] + 1 >= REF_HOPS:
            self.give_up(ref, f"more than {REF_HOPS} listings in a row send the reader elsewhere")
            return
        ref["seen"].add(key)
        ref["hops"] += 1
        ref.update(target=go, judged=set(), queries=[], pages=0, tries=0, fails=0, unindexed=False, loose=set())
        self.run.log.append({"kind": "ref_go", "prop": ref["prop"], "from": target, "to": go})
        self.tell_actor(f"{target} is the listing given for {ref['head']}; its notes pass the reader on to {self.describe(ref)}: the "
                        "code follows them.")

    @staticmethod
    def resolved_value(ref):
        value, pairs = str(ref["value"]), list(ref.get("goes_with") or [])
        relation = ref["relation"]
        if relation == "within" and is_number(ref.get("tol")):
            hit = re.search(r"(-?\d+(?:\.\d+)?)\s*([a-z%]*)", value.lower().replace(",", ""))
            if hit:
                middle, spread = float(hit.group(1)), float(ref["tol"])
                unit = hit.group(2) or str(ref.get("unit") or "")
                span = f"{round(middle - spread, 6):g} to {round(middle + spread, 6):g} {unit}".strip()
                return span + f" (within {spread:g} of {value})"
        if relation in ("at most", "at least"):
            return f"{relation} {value}"
        others = [p for p in pairs if p.lower() != value.lower()]
        if relation == "counterpart":
            return " or ".join(others) if others else value
        if relation == "differ":
            return " and ".join(f"not {v}" for v in [value] + others)
        return " or ".join([value] + others)

    def resolve(self, ref, target, value, pairs):
        if not str(value or "").strip():
            self.give_up(ref, f"{target} shows no value for it")
            return
        if ref.get("for_item"):
            self.settle_own(ref, target, value)
            return
        if ref["relation"] == "counterpart" and not pairs:
            pairs = self.store_pairs(target, str(value).strip(), ref["prop"])
        ref["chain"].append(target)
        ref.update(status="resolved", value=str(value).strip(), goes_with=list(pairs))
        new_req = f"{ref['head']}: {self.resolved_value(ref)}"
        ref["now"] = new_req
        self.swap_in(ref, new_req)
        self.needs.resolved.append(new_req)
        self.needs.version += 1
        self.run.log.append({"kind": "ref_resolved", "req": ref["req"][:200], "now": new_req, "chain": list(ref["chain"])})
        chain = ref["chain"]
        via = f", reached from {chain[0]} through the listings' notes" if len(chain) > 1 else ""
        joined = " and ".join(pairs)
        if ref["relation"] == "differ":
            pair = (f"; its notes pair {joined} with it" if pairs else "") + (
                ". The shopper's value MUST DIFFER from that one: any other value meets it, that value does not")
        elif ref["relation"] == "counterpart":
            pair = (f"; the store pairs it with {joined}, and the shopper wants what goes with it, so {ref['value']} itself does not qualify"
                    if pairs else "; no record states a pairing for it, so that value itself is the requirement")
        else:
            pair = f"; its notes say {ref['value']} goes together with {joined}, so each of them meets it" if pairs else ""
        self.tell_actor(f"RESOLVED: another listing gives {ref['head']}, and {target} shows {ref['value']}{via}{pair}. The "
                        f"requirement is now: {new_req}.")

    def store_pairs(self, target, value, prop):
        # the page holding the value first; another page's rule counts when it names the value
        rows = self.listing_index()
        found = ref_partners(value, [self.note_text(rows.get(target) or {})], True, prop)
        return found or ref_partners(value, [self.note_text(row, desc=False) for ident, row in rows.items() if ident != target], False)

    def swap_in(self, ref, new_req):
        needs = self.needs
        seller = ref_name(ref["first"]["seller"])

        def mine(req):
            if self.ref_match(ref, req):
                return True
            head, _, value = str(req).partition(":")
            bracketed = re.findall(r"\(([^)]*)\)", head)
            if bracketed and ref["tag"] and not kind_words(bracketed[-1]) & kind_words(ref["tag"]):
                return False
            return bool(seller) and seller in ref_name(value) and same_prop(self.head_prop(head), ref["prop"])
        changed = False
        for attr in ("written", "uncovered"):
            before = list(getattr(needs, attr))
            after = [new_req if mine(r) else r for r in before]
            if after != before:
                changed = True
            setattr(needs, attr, list(dict.fromkeys(after)))
        if needs.now_reqs is not None:
            replaced = [new_req if mine(r) else r for r in needs.now_reqs]
            if new_req not in replaced:
                replaced.append(new_req)
            needs.now_reqs = list(dict.fromkeys(replaced))
        if not changed:
            needs.uncovered.append(new_req)
        if isinstance(needs.reading.get("requirements"), list):
            needs.reading["requirements"] = list(needs.written)
        value = new_req.partition(":")[2].strip()
        entries = [e for e in needs.reading.get("items") or [] if isinstance(e, dict) and isinstance(e.get("requirements"), list)]
        for entry in entries:
            if len(entries) >= 2 and ref["tag"] and not (kind_words(entry.get("product") or "") & kind_words(ref["tag"])):
                continue
            entry["requirements"] = [f"{str(r).partition(':')[0].strip()}: {value}" if mine(r) else r for r in entry["requirements"]]

    def give_up(self, ref, why):
        if ref["status"] != "open":
            return
        ref["status"], ref["why"] = "stuck", why
        self.run.log.append({"kind": "ref_stuck", "req": ref["req"][:200], "why": why[:300], "chain": list(ref["chain"])})
        self.tell_actor(f"NOT RESOLVED: {ref['head']} ({why}). The value is still unknown: if a listing's notes you read show where it "
                        "ends, use that value; otherwise never decline or reject a listing over it.")

    def tell_actor(self, text):
        chat = self.run.chat
        if chat and chat[-1].get("role") in ("tool", "user") and isinstance(chat[-1].get("content"), str) and len(chat) > 2:
            chat[-1]["content"] += "\n[reference] " + text
        else:
            self.pending_tells = list(self.pending_tells or []) + [text]

    def ref_line(self):
        try:
            parts = []
            for ref in self.run.refs or []:
                status = ref["status"]
                if status == "resolved":
                    differ = ", must differ from it" if ref.get("relation") == "differ" else ""
                    parts.append(f"{ref['now']} (read on {ref['chain'][-1]}" + differ + ")")
                elif status == "open":
                    parts.append(f"{ref['head']}: the code is following it (now: {self.describe(ref)})")
                elif status == "stuck":
                    parts.append(f"{ref['head']}: not resolved, value unknown (never decline or reject a listing over it)")
            told, self.pending_tells = self.pending_tells, None
            text = "\n[references] " + "; ".join(parts) if parts else ""
            if told:
                text += "".join("\n[reference] " + t for t in told)
            return text
        except Exception:
            return ""

    @staticmethod
    def resolved_line(ref):
        chain = ref["chain"]
        via = f" (reached from {chain[0]} through the listings' notes)" if len(chain) > 1 else ""
        joined = " and ".join(ref["goes_with"])
        if ref.get("relation") == "counterpart":
            pair = f"; the store pairs it with {joined}, so {ref['value']} itself does not qualify" if ref["goes_with"] else ""
            return f"- {ref['now']}: what goes with the value shown on {chain[-1]} ({ref['value']}){via}{pair}."
        if ref.get("relation") != "differ":
            pair = f"; its notes say {ref['value']} goes together with {joined}, so each of them meets it" if ref["goes_with"] else ""
            return f"- {ref['now']}: the value shown on {chain[-1]}{via}{pair}."
        pair = f"; its notes pair {joined} with it" if ref["goes_with"] else ""
        return (f"- {ref['now']}: the shopper's {ref['prop']} has to DIFFER from what {chain[-1]} shows "
                f"({ref['value']}){via}{pair}: any other value meets it, that value does not; the listings of its "
                f"chain ({', '.join(chain)}) carry it.")

    def ref_note(self, thing=""):
        try:
            lines, ends = [], []
            for ref in self.run.refs or []:
                status = ref["status"]
                if ref.get("for_item") and ref["for_item"] != thing:
                    continue
                if status == "resolved":
                    lines.append(self.resolved_line(ref))
                elif status in ("open", "stuck"):
                    chain = ref["chain"]
                    where = "its chain so far: " + " -> ".join(chain) if chain else "the listing it points at was not found"
                    lines.append(f"- {ref['head']}: the code could NOT RESOLVE it (" + where + "). If a record below is that listing, or "
                                 "the last one its chain reached, and gives its own value for it without pointing the reader on, that "
                                 "value is the requirement; otherwise its value is unknown, so never fail an item on it alone.")
                    ends += [i for i in chain[-2:] if i != thing]
            if not lines:
                return ""
            text = ("\n\nREQUIREMENTS SET BY POINTING AT ANOTHER LISTING (the code traced that listing's notes through the store's "
                    "records):\n" + "\n".join(lines))
            records = [r for i in dict.fromkeys(reversed(ends)) for r in self.archive.about(i, 1)]
            if records:
                text += "\nRECORDS OF THE LAST LISTINGS ITS CHAIN REACHED:\n" + "\n".join(records[:2])
            return text
        except Exception:
            return ""


class Coverage:
    def __init__(self, run):
        self.run = run
        self.all_hits = []
        self.hit_score = {}
        self.row_score = {}
        self.aimed_at = {}
        self.put_aims = {}
        self.broken_modes = set()
        self.listing_keys = set()
        self.price_mode = None
        self.seed_keys = set()
        self.seeded_products = set()
        self.totals_seen = {}
        self.walked_products = set()
        self.back_hits_done = None
        self.settled_values = {}
        self.label_indexed = None
        self.composed_sets = {}
        self.label_attempts = {}
        self.name_hints = 0
        self.depth_cap = None
        self.page_signatures = {}
        self.short_hint_given = None
        self.trap_hints = 0
        self.valued_walks = set()
        self.value_pages_read = {}
        self.valued_walk_keys = {}
        self.opens_by_product = {}
        self.waiting_flush = {}
        self.guard_marks = set()
        self.queued_walk = None
        self.sweep_marks = {}
        self.sweep_rounds = {}
        self.sweep_off = False

    @property
    def needs(self):
        return self.run.needs

    @property
    def archive(self):
        return self.run.archive

    def cheapest_rule(self):
        return self.needs.priority() == "cheapest"

    def tell_tool(self, text):
        chat = self.run.chat
        chat[-1]["content"] = str(chat[-1]["content"]) + "\n" + text

    def last_is_tool(self):
        chat = self.run.chat
        return bool(chat) and chat[-1].get("role") == "tool"

    @staticmethod
    def on_home(entry, home):
        return [i for i, s in (entry.get("row_shelf") or {}).items() if s in home]

    def priced_list(self, ids):
        return ", ".join(f"{i} ({self.archive.price(i):g})" for i in ids)

    def may_page(self, floor):
        run = self.run
        wire = run.wire
        return run.moves_left() > floor + 1 and not wire.done and not wire.broke and wire.spare() >= CLOSING + 30

    # -- learning the price filter and the reachable depth
    def note_depth_cap(self, call_args, paging, dicts, total, obs):
        at = call_args.get(paging[0], paging[2])
        nexts = [v for key, v in obs.items() if re.search(r"next|more|onward|continu|cursor", str(key).lower()) and not isinstance(v, (list, dict))]
        size = max(len(dicts), self.widest_page())
        if not is_number(at) or not nexts or len(dicts) < size:
            return
        if not all(v is None or v is False or v == "" for v in nexts):
            return
        reach = int((at - paging[2] + 1) * size)
        if total > reach:
            self.depth_cap = min(self.depth_cap or reach, reach)

    @staticmethod
    def limit_argument(call_args, page_key, paging_keys):
        found = []
        for k, v in call_args.items():
            if not is_number(v) or isinstance(v, bool) or k == page_key or k in paging_keys or not v > 1:
                continue
            if not re.search(r"page|size|count|result|min|low|from|start|offset", str(k).lower()):
                found.append(k)
        return found[0] if len(found) == 1 else None

    @staticmethod
    def maker_free(base, dicts, bound_key, shelf_key, fixed):
        def carried_by_all(val):
            low = val.strip().lower()
            for row in dicts:
                if not any(isinstance(x, str) and x.strip().lower() == low for kk, x in row.items()
                           if not re.search(r"categor|shelf|department|aisle", str(kk).lower())):
                    return False
            return True
        template = {}
        for k, v in base.items():
            if k in (bound_key, shelf_key):
                continue
            if isinstance(v, str) and v.strip() and k not in fixed and carried_by_all(v):
                continue
            template[k] = v
        return template

    def learn_filter(self, call_args, rows, total, obs):
        watch = self.run.watch
        paging = watch.page_arg()
        page_key = paging[0] if paging else None
        base = {k: v for k, v in call_args.items() if k != page_key}
        self.totals_seen[json.dumps(base, sort_keys=True, default=str)] = int(total)
        dicts = [r for r in rows if isinstance(r, dict)]
        if paging and paging[1] is True and dicts and isinstance(obs, dict):
            self.note_depth_cap(call_args, paging, dicts, total, obs)
        _, mode, _, paging_keys = watch.search_keys(call_args)
        bound_key = self.limit_argument(call_args, page_key, paging_keys)
        if bound_key is None or not dicts:
            return
        limit = float(call_args[bound_key])
        labels = {watch.shelf_of(r) for r in dicts}
        shelf_keys = [k for k, v in call_args.items() if isinstance(v, str) and k != page_key and v.strip() and v.strip().lower() in labels]
        shelf_key = shelf_keys[0] if len(shelf_keys) == 1 else None
        fixed = {k: v for k, v in mode if k != shelf_key}
        sig = json.dumps([sorted(fixed.items()), bound_key], default=str)
        prices = [self.row_price(r) for r in dicts]
        if any(p is not None and p > limit + 1e-9 for p in prices):
            self.broken_modes.add(sig)
            return
        if sig in self.broken_modes or shelf_key is None or len(labels) != 1 or "" in labels or None in prices:
            return
        template = self.maker_free(base, dicts, bound_key, shelf_key, fixed)
        words = sum(len(v.split()) for k, v in template.items() if isinstance(v, str) and k not in fixed)
        props = (self.run.shapes.get(self.run.role("search") or "") or {}).get("properties") or {}
        current = self.price_mode
        if current and current["sig"] not in self.broken_modes and current["words"] < words:
            return
        bound_spec = (props.get(bound_key) or {}) if isinstance(props, dict) else {}
        self.price_mode = dict(sig=sig, tmpl=template, mode=fixed, shelf=shelf_key, bound=bound_key, words=words,
                               int=bound_spec.get("type") == "integer")

    @staticmethod
    def row_price(row):
        for key, val in row.items():
            if is_number(val) and not isinstance(val, bool) and re.search(r"price|cost", str(key).lower()):
                return float(val)
        return None

    def price_spec(self):
        spec = self.price_mode
        if not spec or spec["sig"] in self.broken_modes:
            return None
        return spec

    def cg_listing(self, args):
        spec = self.price_spec()
        if not spec or not isinstance(args, dict):
            return False
        if not spec["mode"]:
            return spec["bound"] in args
        return all(args.get(k) == v for k, v in spec["mode"].items())

    def widest_page(self):
        sizes = [entry.get("size", 0) for entry in self.run.page_book.values()]
        return max(sizes or [0])

    def reachable_depth(self):
        paging = self.run.watch.page_arg()
        size = self.widest_page()
        if not paging or size <= 0 or paging[1] == "token" or paging[3] is None:
            return self.depth_cap
        last, first = paging[3], paging[2]
        if paging[1]:
            return int((last - first + 1) * size)
        return int(last - first + size)

    def shelf_spelling(self, labels):
        if not labels:
            return ""
        counts = {}
        known = self.archive.ids

        def tally(fields):
            if not any(isinstance(v, str) and v in known and id_field(key) for key, v in fields.items()):
                return
            spelled = ""
            for key, val in fields.items():
                if isinstance(val, str) and 0 < len(val.strip()) <= 60 and re.search(r"categor|shelf|department|aisle", str(key).lower()) \
                        and "id" not in str(key).lower():
                    spelled = val.strip()
                    break
            if spelled and spelled.lower() in labels:
                counts[spelled] = counts.get(spelled, 0) + 1
        for record in self.archive.items:
            walk_dicts(record, tally)
        return max(sorted(counts), key=lambda s: counts[s], default="")

    def just_below(self, price):
        if not is_number(price):
            return None
        spec = self.price_spec() or {}
        val = float(price)
        if not (spec.get("int") or val.is_integer()):
            lower = round(val - 0.01, 2)
        elif val.is_integer():
            lower = int(val) - 1
        else:
            lower = int(val)
        return lower if lower > 0 else None

    def midpoint(self, a, b):
        spec = self.price_spec() or {}
        whole = spec.get("int") or (float(a).is_integer() and float(b).is_integer())
        return (int(a) + int(b)) // 2 if whole else round((float(a) + float(b)) / 2, 2)

    @staticmethod
    def filter_args(spec, label, bound):
        limit = int(bound) if spec.get("int") or float(bound).is_integer() else bound
        return {**spec["tmpl"], spec["shelf"]: label, spec["bound"]: limit}

    def cg_products(self):
        return list(range(len(self.needs.items()))) or [None]

    def can_list_below(self, k):
        if self.needs.items() and k is None:
            return False
        if not self.cheapest_rule() or k in self.walked_products or self.price_spec() is None:
            return False
        return self.run.watch.page_arg() is not None and bool(self.shelf_spelling(self.run.walk.walk_home(k)))

    def read_on_home(self, home):
        run, archive = self.run, self.archive
        judge, shelf = run.judge, run.shelf
        prices = []
        for ident in archive.read:
            if judge.shelf_known(ident) not in home or not is_number(archive.price(ident)):
                continue
            if ident in shelf.dead_lines() or ident in run.refused_lines or judge.last_stock(ident) is False:
                continue
            prices.append(float(archive.price(ident)))
        return prices

    def aim_price(self, k):
        walk = self.run.walk
        pick = walk.walk_pick(k)
        held = self.put_aims.get(k)
        costs = ([pick[1]] if pick else []) + ([float(held)] if is_number(held) else [])
        if not costs and self.has_amount_bound(k):
            home = walk.walk_home(k)
            seen = [c for c, _ in self.cheapest_unread(self.rows_on(home), k)] + self.read_on_home(home)
            costs = [min(seen)] if seen else []
        return min(costs) if costs else None

    def has_amount_bound(self, k):
        if not self.needs.items() or k is None:
            return False
        product_of = self.run.talk.product_of
        for req in self.needs.props_left():
            head, _, value = str(req).partition(":")
            if product_of(head) == k and re.search(r"\d", value) and re.search(CH_BOUND, value.lower()):
                return True
        return False

    def rows_on(self, home):
        watch, known = self.run.watch, self.archive.ids
        found = set()

        def collect(fields):
            ident = watch.row_mark(fields)
            if ident in known and watch.shelf_of(fields) in home:
                found.add(ident)
        for record in self.archive.items:
            walk_dicts(record, collect)
        return found

    def cheaper_in_stock(self, home, bound):
        watch, known = self.run.watch, self.archive.ids
        hit = False

        def check(fields):
            nonlocal hit
            if hit or watch.shelf_of(fields) not in home:
                return
            if not any(isinstance(v, str) and v in known and id_field(key) for key, v in fields.items()):
                return
            price = self.row_price(fields)
            hit = price is not None and price <= bound and watch.stock_state(fields) is True
        for record in self.archive.items:
            walk_dicts(record, check)
        return hit

    def sold_out_marks(self, ids):
        watch = self.run.watch
        sold_out = {}

        def note_stock(fields):
            if not any(isinstance(v, str) and v in ids and id_field(key) for key, v in fields.items()):
                return
            state = watch.stock_state(fields)
            if state is not None:
                sold_out[watch.row_mark(fields)] = not state
        for record in self.archive.items:
            walk_dicts(record, note_stock)
        return sold_out

    def cheapest_unread(self, ids, k, below=None):
        ids = set(ids)
        if not ids:
            return []
        rows = self.needs.items()
        qty = rows[k]["quantity"] if k is not None and k < len(self.needs.items()) else self.needs.quantity()
        budget = self.needs.spend_cap
        sold_out = self.sold_out_marks(ids)
        found = []
        for ident in ids:
            price = self.archive.price(ident)
            if ident in self.archive.read or ident in self.run.refused_lines or sold_out.get(ident) or not is_number(price):
                continue
            if is_number(budget) and is_number(qty) and price * qty > budget:
                continue
            if below is not None and price >= below:
                continue
            found.append((float(price), ident))
        return sorted(found)

    def all_words(self):
        kind = self.run.word_kind
        return kind == "all" or (bool(self.run.lead_words) and kind != "any")

    def value_led_rows(self, k):
        if not self.all_words():
            return set()
        walk = self.run.walk
        home = walk.walk_home(k)
        rows = set()
        for entry in self.run.page_book.values():
            if entry.get("id_order") is True and not self.cg_listing(entry["args"]) and walk.walk_carries(entry["args"], k):
                rows.update(self.on_home(entry, home))
        return rows

    def own_requirements(self, k):
        several = bool(self.needs.items()) and k is not None
        product_of = self.run.talk.product_of
        mine = []
        for n, req in enumerate(self.needs.props_left()):
            if several and product_of(req.partition(":")[0]) not in (k, None):
                continue
            mine.append((n, req))
        return mine

    @staticmethod
    def plain_value(req):
        return re.sub(r"\((?:stated|revealed[^)]*)\)", "", req.partition(":")[2]).lower()

    def finished_value_walk(self, entry, home, k, single):
        if entry.get("id_order") is not True or self.cg_listing(entry["args"]) or not entry.get("first"):
            return False
        if self.run.watch.next_page(entry) is not None or len(entry.get("row_shelf") or {}) < entry["total"]:
            return False
        return bool(self.on_home(entry, home) and self.run.walk.walk_carries(entry["args"], k) & single)

    def value_search_complete(self, k):
        try:
            if not self.all_words():
                return False
            home = self.run.walk.walk_home(k)
            single = set()
            for n, req in self.own_requirements(k):
                options = [p for p in re.split(r"[,;/]|\bor\b|\beither\b|\bnor\b", self.plain_value(req)) if p.strip()]
                if len(options) == 1:
                    single.add(n)
            return any(self.finished_value_walk(entry, home, k, single) for entry in self.run.page_book.values())
        except Exception:
            return False

    def wanted_words(self, k):
        walk, stem = self.run.walk, self.run.watch.stem
        wanted, figures = [], []
        for _, req in self.own_requirements(k):
            value = self.plain_value(req)
            if ref_text(value):
                continue
            for part in re.split(r"[,;/]|\bor\b|\beither\b|\bnor\b", value):
                figure = walk.fig_amount(part)
                if figure:
                    figures.append(figure)
                    continue
                words = [] if walk.amount_part(part) else [stem(w) for w in re.findall(r"[a-z0-9]{3,}", part)]
                if words:
                    wanted.append(words)
        return wanted, figures

    def row_writes_value(self, ident, k):
        stem = self.run.watch.stem
        wanted, figures = self.wanted_words(k)
        if not wanted and not figures:
            return False
        texts = []

        def gather(fields):
            if any(isinstance(v, str) and v == ident and id_field(key) for key, v in fields.items()):
                texts.extend(str(v).lower() for v in fields.values() if isinstance(v, str))
        for record in self.archive.items:
            walk_dicts(record, gather)
        have = {stem(w) for w in re.findall(r"[a-z0-9]{3,}", " ".join(texts))}
        body = " ".join(texts)
        for figure, unit in figures:
            if re.search(r"(?<![\d.,])" + re.escape(figure) + r"\s*-?\s*" + re.escape(stem(unit)), body):
                return True
        return any(all(w in have for w in words) for words in wanted)

    # -- composed value + label searches
    def burst_room(self):
        run = self.run
        left = max(0, run.ask_limit() - run.msgs_sent) if run.asking and run.role("message") else 0
        return left, run.moves_left() - run.intake.order_due() - 2 * left

    def option_words(self, piece, own_name):
        figure = self.run.walk.fig_amount(piece)
        if re.search(EXCLUDE_WORDS, piece) or (self.run.walk.amount_part(piece) and not figure):
            return None
        piece = re.sub(r"\b(?:" + VALUE_LINKS + r")\b", " ", re.sub(r"'s\b", "", piece))
        # keep a leading whole number together with its unit
        words = []
        for w in re.findall(r"\d+(?:[.,]\d+)+|[a-z0-9]+", piece):
            long_enough = len(w) >= 2 or (figure and w == figure[0])
            if not long_enough or re.search(r"[.,]", w) or w in LINK_WORDS:
                continue
            if w in own_name and not (figure and w == figure[1]):
                continue
            words.append(w)
        return words

    def value_options(self, head, value):
        low = re.sub(r"\([^)]*\)", " ", str(value).lower().replace("’", "'")).strip()
        if not low or ref_text(low):
            return []
        if re.search(r"\bor\b|\beither\b|/", low):
            pieces = [p for p in re.split(r"[,;/]|\bor\b|\beither\b", low) if p.strip()][:3]
        else:
            pieces = [re.split(r"[,;&+]|\band\b", low)[0]]
        own_name = set(re.findall(r"[a-z0-9]+", str(head).lower()))
        options = []
        for piece in pieces:
            words = self.option_words(piece, own_name)
            if words and words not in options:
                options.append(words)
        return options

    def firm_values(self, k):
        values = []
        for head, value in self.run.closer.code_values(k):
            options = self.value_options(head, value)
            if options and options not in values:
                values.append(options)
        return values

    def title_stems_on(self, k):
        watch = self.run.watch
        home = self.run.walk.walk_home(k)
        titles = []

        def collect(fields):
            if watch.shelf_of(fields) in home and watch.row_mark(fields) in self.archive.ids:
                titles.append({watch.stem(w) for w in re.findall(r"[a-z0-9]+", self.run.shelf.title_of(fields).lower())})
        for record in self.archive.items:
            walk_dicts(record, collect)
        return titles

    def label_words(self, k, label):
        cleaned = re.sub(r"'s\b", "", label.lower().replace("’", "'"))
        words = list(dict.fromkeys(w for w in re.findall(r"[a-z0-9]+", cleaned) if len(w) >= 2 and w not in LINK_WORDS))
        if self.label_indexed is not False:
            return words
        stem = self.run.watch.stem
        titles = self.title_stems_on(k)
        common = [w for w in words if titles and 2 * sum(1 for t in titles if stem(w) in t) >= len(titles)]
        if common:
            return common
        rows = self.needs.items()
        if rows and k is not None and k < len(rows):
            product = str(rows[k]["product"])
        else:
            product = str(self.needs.reading.get("product") or "")
        return [w for w in re.findall(r"[a-z0-9]+", product.lower()) if len(w) > 2 and w not in LINK_WORDS][-1:]

    def query_text(self, parts, label_terms, cap_len):
        stem = self.run.watch.stem
        lead = self.run.lead_words
        stems, picked = set(), []
        for w in [w for p in parts for w in p] + label_terms:
            if stem(w) in stems:
                continue
            stems.add(stem(w))
            picked.append(w)
        text = " ".join(picked[:lead] if lead else picked)
        if is_number(cap_len):
            while len(text) > cap_len and " " in text:
                text = text.rsplit(" ", 1)[0]
            text = text[: int(cap_len)]
        return text

    def composed_queries(self, k, label, cap_len):
        values = self.firm_values(k)
        if not values:
            return [], []
        label_terms = self.label_words(k, label)
        every_option = self.cheapest_rule()

        def options(v):
            return v if every_option else v[:1]

        def unique(forms):
            return list(dict.fromkeys(f for f in forms if f))
        single = [self.query_text([a], label_terms, cap_len) for a in options(values[0])]
        if len(values) < 2:
            return unique(single), []
        spread = next((n for n, v in enumerate(values) if len(options(v)) > 1), 0)
        combined = []
        for a in options(values[spread]):
            parts = [a if n == spread else v[0] for n, v in enumerate(values)]
            combined.append(self.query_text(parts, label_terms, cap_len))
        return unique(combined), unique(single)

    def next_form(self, first, second, text_key, mode):
        for forms in (first, second):
            if not forms:
                continue
            totals = [self.sent_total(t, text_key, mode) for t in forms]
            if None in totals:
                return forms[totals.index(None)]
            if any(totals):
                return ""
        if first or second:
            return None
        return ""

    def vl_mode(self, args):
        props = (self.run.shapes.get(self.run.role("search") or "") or {}).get("properties")
        if not isinstance(props, dict):
            props = {}
        chosen = []
        for key, val in args.items():
            spec = props.get(key)
            if isinstance(val, str) and isinstance(spec, dict) and isinstance(spec.get("enum"), list) and len(spec["enum"]) <= 4:
                chosen.append((key, val))
        return tuple(sorted(chosen))

    def word_set(self, text):
        stem = self.run.watch.stem
        return frozenset(stem(w) for w in re.findall(r"[a-z0-9]+", str(text).lower())[: self.run.lead_words or None])

    def sent_total(self, text, text_key, mode):
        wanted = self.word_set(text)
        for key, total in (self.totals_seen or {}).items():
            try:
                args = json.loads(key)
            except Exception:
                continue
            if not isinstance(args, dict) or not isinstance(args.get(text_key), str):
                continue
            if all(args.get(m) == v for m, v in mode.items()) and self.word_set(args[text_key]) == wanted:
                return int(total)
        return None

    def note_label_indexing(self, label):
        if self.label_indexed is not None or not self.archive.items:
            return
        watch = self.run.watch
        label_stems = {watch.stem(w) for w in re.findall(r"[a-z0-9]+", re.sub(r"'s\b", "", label.lower())) if len(w) >= 2 and w not in LINK_WORDS}
        hits = 0

        def count(fields):
            nonlocal hits
            if watch.row_mark(fields) not in self.archive.ids or watch.shelf_of(fields) != label.strip().lower():
                return
            elsewhere = set()
            for key, v in fields.items():
                if isinstance(v, str) and not re.search(r"categor|shelf|department|aisle", str(key).lower()):
                    elsewhere.update(watch.stem(w) for w in re.findall(r"[a-z0-9]+", v.lower()))
            hits += bool(label_stems - elsewhere)
        walk_dicts(self.archive.items[-1], count, ceiling=3)
        if hits >= 2:
            self.label_indexed = True

    def trim_to_steps(self, queue, batch, floor):
        while queue and (1 if batch and len(queue) >= 2 else len(queue)) >= self.run.moves_left() - floor:
            queue.pop()
        return queue

    def cheap_queue(self, k, ids, floor, most):
        cheapest = self.cheapest_rule()
        pick = self.run.walk.walk_pick(k) if cheapest else None
        batch = self.run.shelf.batch_tool()
        opened = self.opens_by_product
        room = CEIL_OPENS - opened.get(k, 0) if cheapest else most
        take = max(0, min(batch[2] if batch else most, room))
        queue = self.trim_to_steps([i for _, i in self.cheapest_unread(ids, k, pick[1] if pick else None)][:take], batch, floor)
        if cheapest:
            opened[k] = opened.get(k, 0) + len(queue)
        return queue

    def launch_reads(self, queue, homes, floor, say):
        run = self.run
        shelf_known = run.judge.shelf_known
        run.rank_checked = {mark for mark in run.rank_checked if shelf_known(mark.split("|")[0]) not in homes}
        run.order_vetted = False
        reads = run.reads
        reads.open_queue = queue
        reads.open_recheck = True
        reads.open_say = say
        reads.open_purpose, reads.open_floor = "walk", floor
        run.judge.open_named()

    def label_template(self, k):
        free_text = self.run.watch.free_text
        for call in self.run.closer.code_queries(k):
            keys = [key for key, val in call.items() if isinstance(val, str) and val.strip() and val in free_text(call)]
            if len(keys) == 1 and not self.cg_listing(call):
                return call, keys[0]
        return None, ""

    def bare_search(self, k):
        shape = self.run.shapes.get(self.run.role("search") or "") or {}
        props = shape.get("properties") if isinstance(shape.get("properties"), dict) else {}
        template, text_key = self.label_template(k)
        if not template:
            return None, "", {}, None
        mode = {key: val for key, val in template.items() if isinstance(props.get(key), dict) and props[key].get("enum")}
        # a bare keyword search, comparable with the actor's own
        required = shape.get("required") or []
        template = {key: val for key, val in template.items() if key == text_key or key in mode or key in required}
        cap_len = props[text_key].get("maxLength") if isinstance(props.get(text_key), dict) else None
        return template, text_key, mode, cap_len

    def pending_form(self, k, label, home, cap_len, text_key, mode):
        first, second = self.composed_queries(k, label, cap_len)
        composed = self.composed_sets.setdefault(str(k), set())
        composed.update(self.word_set(q) for q in first + second)
        todo = self.next_form(first, second, text_key, mode)
        if todo is not None:
            return todo, first, second
        walk = self.run.walk
        carried = any(walk.walk_carries(w["args"], k) and self.on_home(w, home) for w in self.run.page_book.values())
        if not carried or self.label_indexed is not None:
            return "", first, second
        # the label is not indexed: its words in the titles (or the kind word) stand in
        self.label_indexed = False
        first, second = self.composed_queries(k, label, cap_len)
        composed.update(self.word_set(q) for q in first + second)
        return self.next_form(first, second, text_key, mode) or "", first, second

    def value_label_search(self):
        _, room = self.burst_room()
        if room <= 4:
            return False
        run = self.run
        walk = run.walk
        shape = run.shapes.get(run.role("search") or "") or {}
        paging = run.watch.page_arg()
        products = self.cg_products()
        for k in products:
            if self.label_attempts.get(k, 0) >= LABEL_TRIES:
                continue
            sig = json.dumps(self.firm_values(k))
            settled = self.settled_values.setdefault(str(k), [])
            if sig in settled:
                continue
            home = walk.walk_home(k)
            label = self.shelf_spelling(home)
            if not label:
                continue
            template, text_key, mode, cap_len = self.bare_search(k)
            if not template:
                settled.append(sig)
                continue
            todo, first, second = self.pending_form(k, label, home, cap_len, text_key, mode)
            args, faults = ArgCheck.execute({**template, text_key: todo}, shape) if todo else (None, True)
            if faults or not isinstance(args, dict):
                settled.append(sig)
                continue
            self.label_attempts[k] = self.label_attempts.get(k, 0) + 1
            plural = "s" if len(self.firm_values(k)) > 1 else ""
            walk.walk_call(args, f"Searching the required value{plural} with the category name as this store's rows spell it "
                                 f"('{todo}'): keyword search matches listing values, titles and categories, so this "
                                 "lists the category's listings carrying the value.")
            self.after_label_search(k, args, todo, text_key, mode, label, home, first or second, paging, len(products) > 1)
            return True
        return False

    def after_label_search(self, k, args, todo, text_key, mode, label, home, forms, paging, several):
        run = self.run
        total = self.sent_total(todo, text_key, mode)
        self.note_label_indexing(label)
        key = json.dumps({a: v for a, v in args.items() if not paging or a != paging[0]}, sort_keys=True, default=str)
        entry = run.page_book.get(key)
        every = bool(entry) and len(entry.get("row_shelf") or {}) >= entry["total"] and bool(self.on_home(entry, home))
        floor = run.intake.order_due() + 2
        queue = []
        if total is None:
            note = ""
        elif not total:
            note = f"[value search] '{todo}': no listing matched."
        else:
            note = f"[value search] '{todo}': {total} match(es)" + (" - every one is shown here." if every else ".")
        cheapest = self.cheapest_rule()
        if every and cheapest and paging and entry.get("id_order") is not False:
            self.valued_walks.add(key)
            self.valued_walk_keys.setdefault(str(k), []).append(key)
            if several:
                self.waiting_flush[k] = True
            else:
                queue = self.cheap_queue(k, self.on_home(entry, home), floor, CEIL_OPENS)
        elif every and not cheapest and todo in forms:
            queue = self.cheap_queue(k, self.on_home(entry, home), floor, LABEL_OPENS)
        if queue:
            below = " below the pick" if run.walk.walk_pick(k) and self.cheapest_rule() else ""
            note += (" Its unread in-stock rows within the budget on the requested category" + below + ", cheapest first, are opened next: "
                     + self.priced_list(queue) + ".")
        if note and self.last_is_tool():
            self.tell_tool(note + ("" if queue else "\n" + run.status_line()))
        run.log.append({"kind": "label_search", "product": k, "query": todo, "total": total, "every": every, "opened": queue})
        if queue:
            self.launch_reads(queue, home, floor, "Opening the cheapest unread rows of that search: every listing of the category "
                                                  "that carries the required values is among its matches.")

    def flush_pools(self, waiting, per):
        walk = self.run.walk
        book = self.run.page_book
        keys = self.valued_walk_keys or {}
        pools = []
        for k in waiting:
            home = walk.walk_home(k)
            ids = set()
            for key in keys.get(str(k), []):
                if key in book and walk.walk_carries(book[key]["args"], k):
                    ids.update(self.on_home(book[key], home))
            pick = walk.walk_pick(k)
            room = max(0, min(per, CEIL_OPENS - self.opens_by_product.get(k, 0)))
            pools.append([i for _, i in self.cheapest_unread(ids, k, pick[1] if pick else None)][:room])
        return pools

    @staticmethod
    def take_turns(waiting, pools, most):
        owner, queue = {}, []
        while len(queue) < most and any(pools):
            for k, pool in zip(waiting, pools):
                while pool and pool[0] in queue:
                    pool.pop(0)
                if pool and len(queue) < most:
                    owner[pool[0]] = k
                    queue.append(pool.pop(0))
        return owner, queue

    def flush_waiting(self, floor):
        waiting = [k for k, on in self.waiting_flush.items() if on]
        self.waiting_flush.clear()
        if not waiting:
            return False
        run = self.run
        batch = run.shelf.batch_tool()
        per = 2 if run.moves_left() - floor - 1 >= 2 * len(waiting) else 1
        pools = self.flush_pools(waiting, per)
        owner, queue = self.take_turns(waiting, pools, batch[2] if batch else per * len(waiting))
        if not self.trim_to_steps(queue, batch, floor):
            return False
        opened = self.opens_by_product
        for ident in queue:
            opened[owner[ident]] = opened.get(owner[ident], 0) + 1
        homes = set().union(*(run.walk.walk_home(k) for k in waiting))
        self.launch_reads(queue, homes, floor, "Cheapest wanted: opening the cheapest unread rows that each product's value search listed.")
        summary = ("[value searches read] Opened, below each pick, the cheapest unread in-stock rows within the budget that each "
                   "product's value search listed: " + self.priced_list(queue) + ". For each product, "
                   "choose the lowest-priced listing whose own fields meet every requirement now in force; a fit that costs more needs "
                   "every cheaper candidate opened and rejected first.")
        if self.last_is_tool():
            self.tell_tool(summary)
        run.log.append({"kind": "value_flush", "products": waiting, "opened": queue})
        return True

    def product_steps(self):
        try:
            rows = self.needs.items()
            if len(rows) < 2 or not self.cheapest_rule():
                return ""
            read = self.value_pages_read or {}
            parts, missing = [], ""
            for k, row in enumerate(rows):
                pick = self.run.walk.walk_pick(k)
                parts.append(f"{row['product']}: {read.get(k, 0)} value page(s) read, pick {'yes' if pick else 'no'}")
                if not missing and not pick:
                    missing = str(row["product"])
            return "\n[product steps] " + "; ".join(parts) + (f"; next: {missing}" if missing else "") + "."
        except Exception:
            return ""

    # -- reading value-led searches to their end
    def finished_searches(self, k):
        watch, walk = self.run.watch, self.run.walk
        finished = []
        for w in self.run.page_book.values():
            if w.get("id_order") is not True or not w.get("first") or watch.next_page(w) is not None:
                continue
            if self.cg_listing(w["args"]) or len(w.get("row_shelf") or {}) < w["total"]:
                continue
            finished.append((self.word_set(" ".join(watch.free_text(w["args"]))), walk.walk_carries(w["args"], k, alts=True), self.vl_mode(w["args"])))
        return finished

    def product_terms(self, k, home):
        stem = self.run.watch.stem
        several_k = bool(self.needs.items()) and k is not None
        product_of = self.run.talk.product_of
        values = set()
        for req in self.needs.props_left():
            if several_k and product_of(req.partition(":")[0]) not in (k, None):
                continue
            values.update(stem(w) for w in re.findall(r"[a-z0-9]{2,}", req.partition(":")[2].lower()))
        label = {stem(w) for s in home for w in re.findall(r"[a-z0-9]{2,}", s)}
        return values | label

    def open_value_walk(self, key, entry, k, home):
        if key in self.valued_walks or entry.get("id_order") is not True or not entry.get("first"):
            return False
        if self.run.watch.next_page(entry) is None or self.cg_listing(entry["args"]):
            return False
        return bool(self.run.walk.walk_carries(entry["args"], k)) and bool(self.on_home(entry, home))

    def page_candidates(self, k, n, home, plan):
        run = self.run
        watch, walk = run.watch, run.walk
        used, opened = self.value_pages_read, self.opens_by_product
        known, depth = plan["known"], plan["depth"]
        mine = (self.composed_sets or {}).get(str(k)) or set()
        terms = self.product_terms(k, home)
        finished = self.finished_searches(k)
        cands = []
        for key, entry in run.page_book.items():
            if not self.open_value_walk(key, entry, k, home):
                continue
            words = self.word_set(" ".join(watch.free_text(entry["args"])))
            composed = words in mine
            too_deep = depth is not None and entry["total"] > depth
            if not known and (depth is None or too_deep):
                continue
            if known and too_deep and not composed:
                continue
            pairs = walk.walk_carries(entry["args"], k, alts=True)
            how = self.vl_mode(entry["args"])
            # a finished search that showed every match already holds what this one can show
            if known and any(mode == how and (seen <= words or (seen <= terms and pairs <= got)) for seen, got, mode in finished):
                continue
            reach = entry["total"] if depth is None else min(entry["total"], depth)
            pages = -(-reach // max(1, entry["size"])) - len(entry["read"])
            allowed = min(plan["per"] - used.get(k, 0), plan["cap"] - sum(used.values()))
            if plan["several"]:
                allowed = min(allowed, plan["share"] - used.get(k, 0) - opened.get(k, 0) - (0 if opened.get(k) else plan["opens"]))
            take = pages if pages <= allowed else (allowed if composed else 0)
            if take <= 0 or run.moves_left() - take - plan["opens"] <= plan["floor"]:
                continue
            if (take > 2 or plan["several"]) and plan["left"] and plan["room"] > take + plan["opens"] + 4:
                plan["pending"] = True
                continue
            cands.append((take, -walk.walk_score(entry["args"], k), key, words))
        if known:
            # of two nested candidates keep the one whose extra words are not values or label words
            cands = [c for c in cands if not any((o[3] < c[3] and not (c[3] - o[3]) <= terms)
                                                 or (c[3] < o[3] and (o[3] - c[3]) <= terms) for o in cands)]
        return [(used.get(k, 0), take, score, n, key, k) for take, score, key, _ in cands]

    def paging_plan(self, products, several):
        run = self.run
        depth = self.reachable_depth()
        size = max(1, self.widest_page())
        left, room = self.burst_room()
        walk = run.walk
        unread_homes = len([k for k in products if walk.walk_home(k) and not self.value_pages_read.get(k)])
        return {"depth": depth, "per": max(VALUE_PAGES, -(-depth // size) - 1) if depth else VALUE_PAGES,
                "several": several, "cap": max(VALUE_TOTAL, VALUE_TOTAL_PER * len(products)),
                "opens": 1 if run.shelf.batch_tool() else (2 if several else CEIL_OPENS),
                "floor": run.intake.order_due() + 2, "known": self.all_words(), "left": left, "room": room,
                "share": max(4, room // max(1, unread_homes)), "pending": False}

    def value_pages(self):
        run = self.run
        try:
            if any(r.get("status") == "open" for r in run.refs or []) and run.refdesk.ref_room():
                return False
            wire = run.wire
            if run.orders_tried or run.refusal_n or wire.done or wire.broke or wire.spare() < CLOSING + 60 or not run.role("search"):
                return False
            if self.value_label_search():
                return True
            if not self.cheapest_rule() or run.word_kind == "any" or not run.watch.page_arg():
                return False
            products = self.cg_products()
            several = len(products) > 1
            plan = self.paging_plan(products, several)
            options = []
            for n, k in enumerate(products):
                home = run.walk.walk_home(k)
                if home:
                    options += self.page_candidates(k, n, home, plan)
            if options:
                _, take, _, _, key, k = min(options, key=lambda o: o[:5])
                self.valued_walks.add(key)
                return self.read_value_walk(k, key, plan["floor"], take, several)
            waiting = self.waiting_flush
            if not waiting or plan["pending"] or (plan["left"] and plan["room"] > plan["opens"] * len(waiting) + 4):
                return False
            return self.flush_waiting(plan["floor"])
        except Exception as err:
            run.log.append({"kind": "value_pages_error", "error": f"{type(err).__name__}: {err}"[:300]})
            return False

    def read_pages(self, k, entry, paging, most, floor, say):
        watch = self.run.watch
        got, tried = 0, False
        while watch.next_page(entry) is not None and got < most and self.may_page(floor):
            tried = True
            self.value_pages_read[k] = self.value_pages_read.get(k, 0) + 1
            following = watch.next_page(entry)
            if not self.run.walk.walk_call({**entry["args"], paging[0]: following}, say):
                break
            got += 1
            if watch.next_page(entry) == following:
                break
        return got, tried

    def read_value_walk(self, k, key, floor, most=VALUE_PAGES, defer=False):
        run = self.run
        watch = run.watch
        paging = watch.page_arg()
        entry = run.page_book.get(key)
        if not paging or not entry:
            return False
        query = " ".join(watch.free_text(entry["args"]))
        known = self.all_words()
        if known:
            say = ("Cheapest wanted, and this store's keyword search needs every word while it lists results in a fixed id order, "
                   f"not by price: reading the next page of '{query}' so every listing carrying these words gets seen.")
        else:
            say = ("Cheapest wanted, and this store lists search results in a fixed id order, not by price: reading the next page "
                   f"of '{query}' so every listing it matched gets seen.")
        got, tried = self.read_pages(k, entry, paging, most, floor, say)
        if not got:
            return tried
        self.valued_walk_keys.setdefault(str(k), []).append(key)
        seen = len(entry.get("row_shelf") or {})
        complete = watch.next_page(entry) is None and seen >= entry["total"]
        home = run.walk.walk_home(k)
        if complete:
            reach = " - every listing carrying these words is among them" if known else " - every listing it matched is among them"
        elif watch.next_page(entry) is None:
            reach = " - the rest cannot be reached, so a cheaper fit may lie past the reachable rows"
        else:
            reach = ""
        head = f"[value search read] '{query}': {seen} of its {entry['total']} matches seen" + reach + ". "
        closing = ("Choose the lowest-priced listing whose own fields meet every requirement now in force; a fit that costs more "
                   "needs every cheaper candidate opened and rejected first.")
        queue = []
        if defer:
            self.waiting_flush[k] = True
            middle = "Once every product's value search is read, its cheapest unread in-stock rows within the budget are opened. "
        else:
            pick = run.walk.walk_pick(k)
            queue = self.cheap_queue(k, self.on_home(entry, home), floor, CEIL_OPENS)
            if queue:
                middle = ("Its cheapest unread in-stock rows within the budget on the requested category" + (" below the pick" if pick else "")
                          + " are opened next: " + self.priced_list(queue) + ". ")
            else:
                middle = "No unread in-stock row within the budget on the requested category is left among them. "
        summary = head + middle + closing
        if queue:
            self.launch_reads(queue, home, floor, "Cheapest wanted: opening the cheapest unread rows of that search.")
        if self.last_is_tool():
            self.tell_tool(summary + ("" if queue else "\n" + run.status_line()))
        run.log.append({"kind": "value_pages", "product": k, "query": query, "pages": got, "complete": complete, "opened": queue,
                        "defer": defer})
        return True

    # -- after a reveal: a narrow search read to its end, a bulk read of what it found, then the judge
    @staticmethod
    def value_key(text):
        return tuple(re.findall(r"[a-z0-9]+", str(text).lower()))

    def later_values(self, k):
        # values learned after the request, as the current requirement list spells them
        needs = self.needs
        later = {self.value_key(self.plain_value(req)) for req in list(needs.uncovered) + list(needs.resolved or [])}
        later |= {self.value_key(self.plain_value(req)) for _, req in self.own_requirements(k) if "(revealed" in req.lower()}
        later.discard(())
        return later

    def sweep_sig(self, k):
        own = sorted(req for _, req in self.own_requirements(k))
        later = self.later_values(k)
        if k not in self.sweep_rounds and not any(self.value_key(self.plain_value(req)) in later for req in own):
            return None
        return json.dumps(own)

    def sweep_settled(self, k):
        # nothing more to gain: the product's cart line passed the judge on today's requirements and no rank is asked for
        if self.needs.priority() != "none":
            return False
        pick = self.run.walk.walk_pick(k)
        return bool(pick) and pick[0] in self.run.trolley.units and f"{pick[0]}|{self.needs.version}" in self.run.passed

    def sweep_due(self):
        run = self.run
        wire = run.wire
        if self.sweep_off or wire.done or wire.broke or wire.mute or wire.spare() < CLOSING + 60 or not run.role("search"):
            return []
        if self.needs.reply_due or (any(r.get("status") == "open" for r in run.refs or []) and run.refdesk.ref_room()):
            return []
        due = []
        for k in self.cg_products():
            sig = self.sweep_sig(k)
            if sig is None or self.sweep_marks.get(k) == sig or self.sweep_rounds.get(k, 0) >= REVEAL_ROUNDS:
                continue
            if self.sweep_settled(k):
                self.sweep_marks[k] = sig
                continue
            due.append(k)
        return due

    def sweep_values(self, k):
        later = self.later_values(k)
        revealed, stated, shown, plain = [], [], [], []
        for head, value in self.run.closer.code_values(k):
            fresh = self.value_key(value) in later
            (shown if fresh else plain).append(value)
            options = self.value_options(head, value)
            if options:
                (revealed if fresh else stated).append(options)
        # a figure is spelled many ways and an every-word search misses the other spellings, so words narrow first
        stated.sort(key=lambda o: (any(c.isdigit() for w in o[0] for c in w), len(o[0])))
        return revealed, stated, shown, plain

    def sweep_noun(self, k, home):
        rows = self.needs.items()
        if rows and k is not None and k < len(rows):
            product = str(rows[k]["product"])
        else:
            product = str(self.needs.reading.get("product") or "")
        words = [w for w in re.findall(r"[a-z0-9]+", product.lower()) if len(w) > 2 and w not in LINK_WORDS][-1:]
        stem = self.run.watch.stem
        titles = self.title_stems_on(k)
        if words and titles and 2 * sum(1 for t in titles if stem(words[0]) in t) >= len(titles):
            return words
        label = self.shelf_spelling(home)
        return (self.label_words(k, label) if label else []) or words

    def sweep_forms(self, lead, stated, noun, cap_len):
        # broadest first: the value alone, then with the noun, then one stated value more each; noun before stated so a length cap cuts the latter
        core = list(lead) or [v[0] for v in stated[:1]]
        rest = [v[0] for v in (stated if lead else stated[1:])]
        if not core:
            return [], 0
        forms, sets = [], []
        for parts in [core] + [core + [noun] + rest[:n] for n in range(len(rest) + 1)]:
            text = self.query_text(parts, [], cap_len)
            if text and self.word_set(text) not in sets:
                forms.append(text)
                sets.append(self.word_set(text))
        aim = self.word_set(self.query_text(core + [noun] + rest[:1], [], cap_len))
        return forms, sets.index(aim) if aim in sets else len(forms) - 1

    def sweep_hunts(self, revealed, stated, noun, cap_len):
        picks = [[v[0] for v in revealed]]
        spread = next((n for n, v in enumerate(revealed) if len(v) > 1), None)
        if spread is not None:
            for alt in revealed[spread][1:]:
                picks.append([alt if n == spread else v[0] for n, v in enumerate(revealed)])
        hunts = [self.sweep_forms(lead, stated, noun, cap_len) for lead in picks]
        # a first stated value spelled otherwise in this store would end the ladder at nothing
        if len(stated) > 1:
            hunts.append(self.sweep_forms(picks[0], stated[1:], noun, cap_len))
        return hunts

    def ledger_search(self, args, text_key):
        # the same words under the same filters were already sent: their total is known and costs no step
        paging = self.run.watch.page_arg()
        args = {a: v for a, v in args.items() if not paging or a != paging[0]}
        wanted = self.word_set(args[text_key])
        rest = {a: v for a, v in args.items() if a != text_key}
        for key, total in self.totals_seen.items():
            try:
                held = json.loads(key)
            except Exception:
                continue
            if isinstance(held, dict) and set(held) == set(args) and all(held[a] == v for a, v in rest.items()) \
                    and self.word_set(held[text_key]) == wanted:
                return key, int(total)
        return None, None

    def sweep_probe(self, text, ctx):
        run = self.run
        args, faults = ArgCheck.execute({**ctx["template"], ctx["text_key"]: text}, run.shapes.get(run.role("search") or "") or {})
        if faults or not isinstance(args, dict):
            return None, None
        key, total = self.ledger_search(args, ctx["text_key"])
        if key is None:
            if ctx["pages"] <= 0 or ctx["probes"] <= 0 or not self.may_page(ctx["floor"]):
                return None, None
            ctx["pages"] -= 1
            ctx["probes"] -= 1
            if not run.walk.walk_call(args, f"Searching what is known now, the newest need first, with the item kind: '{text}'."):
                return None, None
            key, total = self.ledger_search(args, ctx["text_key"])
        return key, total

    def pages_short(self, entry):
        # pages this search still needs; None when its total lies beyond what the store lets anyone reach
        watch = self.run.watch
        if not watch.page_arg() or watch.next_page(entry) is None:
            return 0
        depth = self.reachable_depth()
        if depth is not None and entry["total"] > depth:
            return None
        return max(1, -(-entry["total"] // max(1, entry["size"])) - len(entry["read"]))

    def sweep_pages(self, k, key, most, ctx):
        run = self.run
        entry = run.page_book.get(key)
        paging = run.watch.page_arg()
        if not entry:
            return
        self.valued_walks.add(key)
        self.valued_walk_keys.setdefault(str(k), []).append(key)
        if not paging or most <= 0:
            return
        query = " ".join(run.watch.free_text(entry["args"]))
        got, _ = self.read_pages(k, entry, paging, most, ctx["floor"],
                                 f"This store lists matches in a fixed order, so the fit can sit on any page: reading on through '{query}'.")
        ctx["pages"] -= got

    def sweep_hunt(self, k, forms, start, ctx):
        book = self.run.page_book
        at, last, covered, tried = start, None, False, set()
        if self.run.word_kind == "any":
            # more words widen such a store and rank the closest first: the fullest form, first pages only
            at = len(forms) - 1
        while 0 <= at < len(forms) and at not in tried:
            tried.add(at)
            key, total = self.sweep_probe(forms[at], ctx)
            if key is None or self.sweep_sig(k) != ctx["sig"]:
                break
            if not total:
                at -= 1
                continue
            entry = book.get(key)
            if not entry:
                break
            # one more word that does not cut the total: this store is not narrowing, so the fullest form stands
            flat = last is not None and total >= last
            if flat or not covered:
                ctx["wide"] = (key, total) if flat or not ctx["wide"] or total < ctx["wide"][1] else ctx["wide"]
            if flat:
                break
            need = self.pages_short(entry)
            if need is not None and need <= ctx["pages"]:
                self.sweep_pages(k, key, need, ctx)
                ctx["covered"] = covered = True
                if not self.sweep_short(k, ctx):
                    return
                at -= 1
                continue
            if covered:
                return
            last = total
            at += 1

    def sweep_short(self, k, ctx):
        # a rank needs every option seen; otherwise a fuller pool than can be read adds nothing
        return self.cheapest_rule() or len(self.sweep_queue(k, ctx)) < ctx["reads"]

    def row_texts(self, ids):
        watch = self.run.watch
        texts = {}

        def gather(fields):
            ident = watch.row_mark(fields)
            if ident in ids:
                texts.setdefault(ident, []).extend(str(v) for v in fields.values() if isinstance(v, (str, int, float)))
        for record in self.archive.items:
            walk_dicts(record, gather)
        return {ident: " | ".join(parts) for ident, parts in texts.items()}

    def whole_rows(self, k):
        # every row of an all-words search read to its end holds the words searched, shown in its title or not
        run = self.run
        watch = run.watch
        rows = set()
        if run.word_kind == "any" or not watch.page_arg():
            return rows
        for entry in run.page_book.values():
            if watch.next_page(entry) is None and len(entry.get("row_shelf") or {}) >= entry["total"] and run.walk.walk_carries(entry["args"], k):
                rows.update(entry.get("row_shelf") or {})
        return rows

    def sweep_queue(self, k, ctx):
        run = self.run
        home, book = ctx["home"], run.page_book
        ids = self.rows_on(home) if home else set()
        for key in dict.fromkeys(self.valued_walk_keys.get(str(k)) or []):
            entry = book.get(key) or {}
            ids.update(self.on_home(entry, home) if home else entry.get("row_shelf") or {})
        pick = run.walk.walk_pick(k) if self.cheapest_rule() else None
        gone = run.shelf.dead_lines() | run.refdesk.ref_pointers()
        order = [i for _, i in self.cheapest_unread(ids, k, pick[1] if pick else None) if i not in gone]
        texts = self.row_texts(set(order))
        whole = self.whole_rows(k)
        seen = run.closer.code_seen
        shown, plain = ctx["shown"], ctx["plain"]

        def rank(ident):
            text = texts.get(ident, "")
            hits = sum(1 for value in shown if seen(value, text))
            return 0 if hits else 1 if ident in whole else 2, -hits, -sum(1 for value in plain if seen(value, text)), order.index(ident)
        return sorted(order, key=rank)[: ctx["reads"]]

    def sweep_share(self, spare, several):
        # steps go to reading first: a page shows rows, only an opened listing can pass
        run = self.run
        most = REVEAL_READS_MANY if several else REVEAL_READS
        batch = run.shelf.batch_tool()
        whole_read = -(-most // batch[2]) if batch else most
        calls = min(whole_read, max(1, spare // 2)) if spare > 0 else 0
        if run.wire.spend >= SOFT_SPEND * run.wire.scale:
            calls = min(calls, 1)
        reads = min(most, calls * batch[2] if batch else calls)
        pages = min(REVEAL_PAGES // 2 if several else REVEAL_PAGES, spare - calls)
        return max(0, pages), calls, reads

    def evidence_order(self, ids, k):
        closer = self.run.closer
        table = closer.code_rows()
        values = closer.code_values(k)
        hits = closer.code_hits or {}
        cheapest = self.cheapest_rule()

        def rank(ident):
            row = table.get(ident)
            missing = sum(1 for prop, value in values if not row or not closer.code_shows(row, prop, value, hits.get(ident) or set(), True))
            price = self.archive.price(ident)
            return missing, price if cheapest and is_number(price) else 0.0, ident
        return sorted(ids, key=rank)

    def sweep_judge(self, k, home, several):
        run = self.run
        wire, judge, needs = run.wire, run.judge, run.needs
        version, cap = str(needs.version), needs.spend_cap
        mine = [i for i in sorted(self.archive.read) if not home or judge.shelf_known(i) in home]
        refuted = {stamp.split("|")[0] for stamp, verdict in run.rulings.items() if verdict and stamp.split("|")[1:2] == [version]}
        ruled = sorted(refuted & set(mine) | {i for i in mine if judge.field_view(i)["fails"]})
        passed = [i for i in mine if f"{i}|{version}" in run.passed and i not in run.refused_lines]
        waiting = [i for i in mine if i not in ruled and judge._rank_candidate_ok(i, version, cap)]
        wanted = 2 if needs.priority() == "best" else 1
        if self.cheapest_rule() and passed:
            # only a lower price can change a cheapest choice
            floor = min(self.archive.price(i) for i in passed if is_number(self.archive.price(i)))
            waiting = [i for i in waiting if self.archive.price(i) < floor]
            wanted = len(passed) + 1
        waiting = self.evidence_order(waiting, k)
        turns = REVEAL_CHECKS - 1 if several else REVEAL_CHECKS
        for ident in waiting[:turns]:
            # each check goes through consult, so it counts as spend; it stops at the soft line and the actor's own checks carry on
            if len(passed) >= wanted or wire.spend >= SOFT_SPEND * wire.scale or wire.spare() < CLOSING + 60:
                break
            stamp = f"{ident}|{version}"
            if judge.check_item(ident, "ORDER"):
                ruled.append(ident)
            elif stamp in run.passed:
                passed.append(ident)
            elif run.rulings.get(stamp) == "":
                # no answer is no ruling: the check at the cart may still be made
                del run.rulings[stamp]
        left = [i for i in waiting if i not in passed and i not in ruled]
        return passed, left, ruled

    def sweep_note(self, k, passed, left, ruled, opened):
        rows = self.needs.items()
        name = rows[k]["product"] if rows and k is not None and k < len(rows) else self.needs.reading.get("product") or "the item"
        parts = [f"[search after the reveal] {name}: {len(opened)} more listing(s) read."]
        if passed:
            parts.append("Passed the independent check on every requirement now in force: " + self.priced_list(passed[:4]) + ".")
        if left:
            parts.append("Read, nothing against them yet: " + self.priced_list(left[:6]) + ".")
        if ruled:
            parts.append("Ruled out: " + ", ".join(ruled[:6]) + ".")
        parts.append("Choose among all the listings read, not the first that came up.")
        return " ".join(parts)

    def sweep_tell(self, text):
        chat = self.run.chat
        if chat and chat[-1].get("role") in ("tool", "user") and isinstance(chat[-1].get("content"), str):
            chat[-1]["content"] += "\n" + text
        else:
            chat.append({"role": "user", "content": text})

    def sweep_product(self, k, spare, several):
        run = self.run
        sig = self.sweep_sig(k)
        self.sweep_marks[k] = sig
        pages, calls, reads = self.sweep_share(spare, several)
        if not calls:
            return
        tick = run.wire.tick
        home = run.walk.walk_home(k)
        revealed, stated, shown, plain = self.sweep_values(k)
        left, _ = self.burst_room()
        floor = run.intake.order_due() + 2
        template, text_key, _, cap_len = self.bare_search(k)
        ctx = dict(sig=sig, home=home, shown=shown, plain=plain, reads=reads, pages=pages, probes=REVEAL_PROBES,
                   template=template, text_key=text_key, floor=floor + 2 * left + calls, wide=None, covered=False)
        if template and run.role("search"):
            for forms, start in self.sweep_hunts(revealed, stated, self.sweep_noun(k, home), cap_len):
                if forms:
                    self.sweep_hunt(k, forms, start, ctx)
                if ctx["pages"] <= 0 or self.sweep_sig(k) != sig or not self.sweep_short(k, ctx):
                    break
            if ctx["wide"] and not ctx["covered"]:
                # no form fits the budget: the narrowest gets a few pages, for it is the best of what is left
                self.sweep_pages(k, ctx["wide"][0], min(ctx["pages"], 3), ctx)
        queue = self.sweep_queue(k, ctx)
        batch = run.shelf.batch_tool()
        queue = queue[: max(0, run.moves_left() - floor - 2 * left) * (batch[2] if batch else 1)]
        if queue and run.wire.spare() >= CLOSING + 30:
            self.launch_reads(queue, home or {run.judge.shelf_known(queue[0])}, floor,
                              "Reading the candidates of the requirements known now together, so the choice is made among all of them.")
        opened = [i for i in queue if i in self.archive.read]
        passed, others, ruled = self.sweep_judge(k, home, several)
        if run.wire.tick != tick or opened:
            self.sweep_rounds[k] = self.sweep_rounds.get(k, 0) + 1
            self.sweep_tell(self.sweep_note(k, passed, others, ruled, opened) + ("" if self.last_is_tool() and opened else "\n" + run.status_line()))
        else:
            self.sweep_rounds.setdefault(k, 0)
        run.log.append({"kind": "reveal_sweep", "product": k, "pages_left": ctx["pages"], "queue": queue, "opened": opened,
                        "passed": passed, "ruled": ruled, "steps": run.wire.tick - tick})

    def reveal_sweep(self):
        run = self.run
        tick = run.wire.tick
        try:
            due = self.sweep_due()
            for n, k in enumerate(due):
                _, room = self.burst_room()
                self.sweep_product(k, (room - 2) // (len(due) - n), len(self.cg_products()) > 1)
                if run.wire.done or self.needs.reply_due:
                    break
        except Exception as err:
            self.sweep_off = True
            run.log.append({"kind": "reveal_sweep_error", "error": f"{type(err).__name__}: {err}"[:300]})
            tail = run.chat[-1] if run.chat else {}
            for call in tail.get("tool_calls") or []:
                run.chat.append({"role": "tool", "tool_call_id": call.get("id"), "content": "Not run."})
        return run.wire.tick != tick

    # -- the price-filter walk below a pick
    def walk_floor(self, floor, repair):
        if repair:
            return floor + 2
        if not self.run.orders_tried:
            return max(floor, self.run.intake.order_due() + 2)
        return floor

    def finish_listing(self, entry, args, paging, calls, floor):
        watch = self.run.watch
        while watch.next_page(entry) is not None and calls < CEIL_CALLS and self.may_page(floor):
            calls += 1
            following = watch.next_page(entry)
            if not self.run.walk.walk_call({**args, paging[0]: following}, "Reading the next page of that price-filter listing.") \
                    or watch.next_page(entry) == following:
                break
        return calls, watch.next_page(entry) is None

    def lower_limit(self, entry, bound, total, calls):
        prices = [float(c) for c in (self.archive.price(i) for i in (entry.get("row_shelf") or {})) if is_number(c)]
        low = min(prices) if prices else 0.0
        # halfway down, or lower where the total spread over the prices fits the calls left
        fill = max(1, CEIL_CALLS - calls) * max(1, entry["size"])
        return min(self.midpoint(low, bound), self.midpoint(low, low + (bound - low) * 1.8 * fill / max(total, 1)))

    def ceiling_walk(self, k, floor, repair=False):
        run = self.run
        watch, walk = run.watch, run.walk
        spec, paging = self.price_spec(), watch.page_arg()
        home = walk.walk_home(k)
        label = self.shelf_spelling(home)
        aim = self.aim_price(k)
        if not spec or not paging or not label or aim is None:
            return None
        self.walked_products.add(k)
        pick_now = walk.walk_pick(k)
        self.aimed_at[k] = (float(aim), pick_now[0] if pick_now else "")
        bound = self.just_below(aim)
        if bound is None:
            return None
        floor = self.walk_floor(floor, repair)
        depth = self.reachable_depth()
        book = run.page_book
        keys = []
        calls, top, complete, ran, last = 0, None, False, False, bound
        what = "the pick" if pick_now or is_number((self.put_aims or {}).get(k)) else "the cheapest row seen"
        say = f"Cheapest wanted: listing '{label}' priced under {what} ({aim:g}) through the store's price filter."
        while calls < CEIL_CALLS and self.may_page(floor):
            args = self.filter_args(spec, label, bound)
            key = json.dumps(args, sort_keys=True, default=str)
            self.listing_keys.add(key)
            calls += 1
            reuse = key in book and bool(book[key].get("first")) and key in (self.totals_seen or {})
            if not reuse and not walk.walk_call(args, say):
                break
            ran, last = True, bound
            keys.append(key)
            entry = book.get(key)
            if entry and any(is_number(self.archive.price(i)) and self.archive.price(i) > bound for i in (entry.get("row_shelf") or {})):
                self.broken_modes.add(spec["sig"])
            total = self.totals_seen.get(key) if self.price_spec() is not None else None
            if self.price_spec() is None or total is None:
                break
            if total and entry:
                pages = -(-total // max(1, entry["size"]))
                if (depth is None or total <= depth) and pages - len(entry["read"]) <= CEIL_CALLS - calls:
                    calls, complete = self.finish_listing(entry, args, paging, calls, floor)
                    break
                top = bound
                lowered = self.lower_limit(entry, bound, total, calls)
                if not lowered < bound:
                    break
                bound = lowered
                continue
            if self.cheaper_in_stock(home, bound):
                self.broken_modes.add(spec["sig"])
                break
            if top is None:
                complete = True
                break
            lowered = self.midpoint(bound, top)
            if not bound < lowered < top:
                break
            bound = lowered
        if not ran:
            return None
        return self.walk_report(k, keys, home, aim, label, last, complete, what, calls)

    def walk_report(self, k, keys, home, aim, label, last, complete, what, calls):
        book = self.run.page_book
        ids = set()
        for key in keys:
            ids.update(self.on_home(book.get(key, {}), home))
        cands = [i for _, i in self.cheapest_unread(ids, k, aim)]
        valued = self.value_led_rows(k)
        lead = [i for i in cands if i in valued]
        rest = [i for i in cands if i not in valued]
        texty = [i for i in rest if self.row_writes_value(i, k)]
        bare = [] if self.value_search_complete(k) else [i for i in rest if i not in texty]
        ranked = lead + texty + bare
        if complete and not ids:
            report = f"the price filter on '{label}' below {aim:g}: no listing of it costs less than {what}"
        elif complete:
            report = (f"the price filter on '{label}': every listing priced up to {last:g} seen ({len(ids)}, {len(cands)} of them unread, "
                      "in stock and within the budget)")
        else:
            report = f"the price filter on '{label}' below {aim:g}: {len(ids)} of the cheaper listings seen ({len(cands)} unread, in stock and within the budget)"
        self.run.log.append({"kind": "ceiling_walk", "product": k, "label": label, "aim": aim, "calls": calls, "complete": complete,
                             "rows": len(ids), "ranked": ranked[:CEIL_OPENS]})
        return report, ranked[:CEIL_OPENS]

    def ceiling_sweep(self, tried):
        watch = self.run.watch
        spec, paging = self.price_spec(), watch.page_arg()
        if not spec or not paging or self.needs.items():
            return None, False
        rows = watch.shelf_rows()
        shelves = watch.shelf_labels()
        label = self.shelf_spelling(shelves)
        if not rows or not label:
            return None, False
        bound = self.just_below(min(c for c, _, _ in rows.values()))
        if bound is None:
            return None, True
        args = self.filter_args(spec, label, bound)
        key = json.dumps(args, sort_keys=True, default=str)
        if key not in tried:
            self.listing_keys.add(key)
            return args, False
        entry = self.run.page_book.get(key)
        if not entry:
            return None, self.totals_seen.get(key) == 0 and not self.cheaper_in_stock(set(shelves), bound)
        following = watch.next_page(entry)
        if following is None:
            return None, len(entry.get("row_shelf") or {}) >= entry["total"]
        more = {**args, paging[0]: following}
        if json.dumps(more, sort_keys=True, default=str) in tried:
            return None, False
        return more, False

    def guarded_listing(self, call_args):
        run = self.run
        named = [v for v in call_args.values() if isinstance(v, str) and v in self.archive.ids]
        if len(named) != 1 or named[0] in self.archive.read or named[0] in run.trolley.units:
            return None
        ident = named[0]
        if ident in run.intake.named_open() or run.refs or self.needs.reading.get("references"):
            return None
        return ident

    def value_guard(self, call_name, call_args):
        run = self.run
        try:
            if call_name != run.role("details") or not self.cheapest_rule() or not self.all_words() or run.intake.in_reserve():
                return ""
            ident = self.guarded_listing(call_args)
            if ident is None:
                return ""
            k = run.walk.product_index(ident)
            if self.needs.items() and k is None:
                return ""
            mark = (k, self.needs.version)
            if mark in self.guard_marks:
                return ""
            valued = self.value_led_rows(k)
            if not valued or ident in valued:
                return ""
            pick = run.walk.walk_pick(k)
            own_price = self.archive.price(ident)
            costs = ([pick[1]] if pick else []) + ([float(own_price)] if is_number(own_price) else [])
            rows = self.cheapest_unread(valued, k, min(costs))[:6] if costs else []
            if not rows:
                return ""
            self.guard_marks.add(mark)
            return ("Not run (no step used). The cheapest is asked for, and this store's keyword search needs every word, so the rows "
                    f"its searches led by a required value showed carry those words somewhere in their records; {ident} is not among "
                    "them, while these rows of theirs cost less" + (" than it and the pick" if pick else " than it") + " and are unread: "
                    + ", ".join(f"{i} ({c:g})" for c, i in rows) + ". Open them first, cheapest first; send this call again if you "
                    "still want it.")
        except Exception as err:
            run.log.append({"kind": "value_guard_error", "error": f"{type(err).__name__}: {err}"[:300]})
            return ""

    def title_line(self, k, rec, template):
        stem = self.run.watch.stem
        avoid = set(LINK_WORDS)
        for label in self.run.walk.walk_home(k):
            avoid.update(re.findall(r"[a-z]+", label))
        for req in self.needs.props_left():
            avoid.update(re.findall(r"[a-z0-9]+", req.lower()))
        for entry in self.needs.items() or [{"product": str(self.needs.reading.get("product") or "")}]:
            avoid.update(re.findall(r"[a-z]+", str(entry.get("product") or "").lower()))
        stems = {stem(w) for w in avoid}
        tokens = re.findall(r"[a-z0-9]+", self.run.shelf.title_of(rec).lower())

        def free(word):
            return word not in avoid and stem(word) not in stems
        for n, tok in enumerate(tokens):
            if not re.fullmatch(r"[a-z]{3,}", tok) or not free(tok):
                continue
            following = tokens[n + 1] if n + 1 < len(tokens) else ""
            text = tok + (f" {following}" if following and free(following) else "")
            if self.run.walk.ran_query(text):
                return None
            return {**template[0], template[1]: text}
        return None

    def rewalk_dead_aim(self):
        run = self.run
        if not self.cheapest_rule() or run.wire.done or run.wire.spare() < CLOSING + 60 or not run.role("search"):
            return None
        aimed = self.aimed_at or {}
        gone = run.shelf.dead_lines() | run.refused_lines
        opens = 1 if run.shelf.batch_tool() else 2
        for k, (aim, ident) in list(aimed.items()):
            if not ident or ident not in gone or self.price_spec() is None or run.watch.page_arg() is None:
                continue
            pick = run.walk.walk_pick(k)
            if not pick or pick[1] <= aim + 1e-9 or run.moves_left() < FIRST_ORDER_BY + 3 + opens:
                continue
            aimed.pop(k, None)
            (self.put_aims or {}).pop(k, None)
            run.log.append({"kind": "ceil_rearm", "product": k, "dead": ident, "aim": pick[1]})
            return run.walk.run_walk([], FIRST_ORDER_BY, [k], repair=True)
        return None

    def reserve_walk(self, spare, opens):
        walk = self.run.walk
        due = self.run.intake.order_due
        plan = walk.walk_plan(WALK_TOTAL)
        if not plan:
            return False
        pages = sum(n for _, _, n in plan) + sum(1 for k, _, _ in plan if walk.walk_pick(k))
        if spare > due() + 2 + pages + opens:
            return False
        plan = walk.walk_plan(min(WALK_TOTAL, spare - due() - 1 - opens))
        if not plan:
            return False
        return walk.run_walk(plan, due())

    def walk_step(self):
        run = self.run
        walk = run.walk
        try:
            if walk.decline_queue:
                return walk.run_decline_reads()
            queued, self.queued_walk = self.queued_walk, None
            if queued:
                return walk.run_walk(*queued)
            again = self.rewalk_dead_aim()
            if again is not None:
                return again
            if run.orders_tried or not self.cheapest_rule() or run.wire.spare() < CLOSING + 60 or run.wire.done:
                return False
            if self.product_unfound():
                return False
            opens = 1 if run.shelf.batch_tool() else 2
            spare = run.moves_left()
            due = run.intake.order_due
            ready = [k for k in self.cg_products() if (walk.walk_pick(k) or self.has_amount_bound(k)) and self.can_list_below(k)
                     and self.aim_price(k) is not None]
            roomy = spare > due() + 2 + CEIL_CALLS + opens
            if ready and roomy:
                # the product with no pick, then the fewest value pages read, goes first
                read = self.value_pages_read or {}
                ready.sort(key=lambda k: (bool(walk.walk_pick(k)), read.get(k, 0), -1 if k is None else k))
                return walk.run_walk([], due() + 2, ready[:1])
            if not ready and roomy and self.seed_price_filter():
                return True
            return self.reserve_walk(spare, opens)
        except Exception as err:
            run.log.append({"kind": "walk_error", "error": f"{type(err).__name__}: {err}"[:300]})
            return False

    @staticmethod
    def shelf_argument(props, label, skip):
        for key, spec in props.items():
            if not isinstance(spec, dict) or key in skip or not re.search(r"categor|shelf|department|aisle", str(key).lower()):
                continue
            if isinstance(spec.get("enum"), list):
                hit = next((v for v in spec["enum"] if isinstance(v, str) and v.strip().lower() == label.strip().lower()), None)
                if hit is not None:
                    return key, hit
            elif spec.get("type") == "string":
                return key, label
        return None, None

    @staticmethod
    def price_argument(props, page_key):
        for key, spec in props.items():
            if not isinstance(spec, dict) or spec.get("type") not in ("integer", "number") or spec.get("enum") or key == page_key:
                continue
            if re.search(r"ceil|max|price|cap$|budget|upper|most", str(key).lower()) \
                    and not re.search(r"page|first|offset|skip|count|size|num", str(key).lower()):
                return key
        return None

    @staticmethod
    def paired_modes(props, shelf_key, sentences):
        chosen = {}
        for key, spec in props.items():
            if key == shelf_key or not isinstance(spec, dict) or not isinstance(spec.get("enum"), list) or not 2 <= len(spec["enum"]) <= 4:
                continue
            paired = None
            for option in spec["enum"]:
                if any(re.search(r"\b" + re.escape(str(option).lower()) + r"\b", part)
                       and re.search(r"\b" + re.escape(shelf_key.lower()) + r"\b", part) for part in sentences):
                    paired = option
                    break
            if paired is None:
                return None
            chosen[key] = paired
        return chosen

    def seed_listing_args(self, label, bound):
        run = self.run
        tool = run.role("search")
        shape = run.shapes.get(tool or "") or {}
        props = shape.get("properties") if isinstance(shape.get("properties"), dict) else {}
        paging = run.watch.page_arg()
        if not props or not paging or not is_number(bound):
            return None
        described = next((str(tdef["function"].get("description") or "") for tdef in run.toolset if tdef["function"]["name"] == tool), "")
        sentences = [part.lower() for part in re.split(r"[.;]", described)]
        token = run.watch.token_key()
        shelf_key, shelf_val = self.shelf_argument(props, label, (token, paging[0]))
        bound_key = self.price_argument(props, paging[0])
        if not shelf_key or not bound_key:
            return None
        whole = props[bound_key].get("type") == "integer" or float(bound).is_integer()
        args = {shelf_key: shelf_val, bound_key: int(bound) if whole else bound}
        modes = self.paired_modes(props, shelf_key, sentences)
        if modes is None:
            return None
        args.update(modes)
        if any(key not in args for key in shape.get("required") or []):
            return None
        fixed, faults = ArgCheck.execute(args, shape)
        return fixed if not faults and isinstance(fixed, dict) else None

    def seed_signature(self, args, label, paging):
        mode = dict(self.run.watch.search_keys(args)[1])
        page_key = paging[0] if paging else None
        fixed_modes = sorted((m, v) for m, v in mode.items() if not (isinstance(args.get(m), str) and args[m] == label))
        number_key = next((a for a in args if is_number(args[a]) and a != page_key), "")
        return json.dumps([fixed_modes, number_key], default=str)

    def seed_price_filter(self):
        run = self.run
        try:
            if self.price_spec() is not None or self.price_mode or not self.cheapest_rule():
                return False
            paging = run.watch.page_arg()
            walk = run.walk
            for k in self.cg_products():
                if k in self.seeded_products or (self.needs.items() and k is None):
                    continue
                pick = walk.walk_pick(k)
                label = self.shelf_spelling(walk.walk_home(k))
                if not pick or not label:
                    continue
                self.seeded_products.add(k)
                args = self.seed_listing_args(label, self.just_below(pick[1]))
                if not args:
                    return False
                sig = self.seed_signature(args, label, paging)
                key = json.dumps({a: v for a, v in args.items() if not paging or a != paging[0]}, sort_keys=True, default=str)
                if sig in self.broken_modes or key in (self.totals_seen or {}):
                    continue
                self.seed_keys.add(key)
                walk.walk_call(args, f"Cheapest wanted: listing '{label}' priced under the pick ({pick[1]:g}) to test whether this "
                                     "store's price filter holds.")
                run.log.append({"kind": "ceil_seed", "product": k, "args": args, "learned": self.price_spec() is not None})
                if self.last_is_tool() and self.price_spec() is None:
                    self.tell_tool(run.status_line())
                return True
            return False
        except Exception as err:
            run.log.append({"kind": "ceil_seed_error", "error": f"{type(err).__name__}: {err}"[:300]})
            return False

    def product_unfound(self):
        if not self.needs.items():
            return False
        try:
            read = {s.strip().lower() for s in self.archive.shelves_read()}
            for home in self.run.shelf.home_shelves():
                if not home or not home & read:
                    return True
            version = str(self.needs.version)
            live = [mark for mark in self.run.passed if mark.rpartition("|")[2] == version]
            return bool(live) and self.run.intake.unverified()
        except Exception:
            return False

    def hold_aims(self, ids, only, room):
        walk = self.run.walk
        ceil = []
        for k in sorted(only, key=lambda x: -1 if x is None else x):
            if room < 2 or not self.can_list_below(k):
                continue
            ceil.append(k)
            room -= CEIL_CALLS
            costs = [float(p) for p in (self.archive.price(i) for i in ids if i and walk.product_index(i) == k) if is_number(p)]
            if costs:
                held = self.put_aims.get(k)
                self.put_aims[k] = min(costs + ([float(held)] if is_number(held) else []))
        return ceil, room

    def walk_hold(self, call_args):
        run = self.run
        try:
            if not self.cheapest_rule() or self.queued_walk or run.wire.spare() < CLOSING + 60 \
                    or (run.refusal_n and run.moves_left() <= 15):
                return ""
            walk = run.walk
            if call_args is None:
                ids = list(run.trolley.units)
            else:
                ids = [i for i, c in run.trolley.lines(call_args) if c is None or c > 0]
            only = {walk.product_index(i) for i in ids if i}
            floor = FIRST_ORDER_BY + 1 if run.orders_tried == 0 else 4
            # pages, line search and opens share half of the spare steps
            half = (run.moves_left() - floor - 1) // 2
            ceil, room = self.hold_aims(ids, only, half - (1 if run.shelf.batch_tool() else 2))
            plan = walk.walk_plan(min(WALK_TOTAL, room), only)
            if not plan and not ceil:
                return ""
            self.queued_walk = (plan, max(floor, run.moves_left() - half - 1), ceil)
            book = run.page_book
            free_text = run.watch.free_text
            queries = [" ".join(free_text(book[key]["args"])) for _, key, _ in plan if key in book]
            parts = []
            if queries:
                parts.append("reads the remaining pages of " + ", ".join(f"'{q}'" for q in queries))
            if ceil:
                parts.append("lists the category's listings priced below the pick with the store's price filter")
            return ("Not run yet (no step used). The cheapest is asked for, and this store lists search results in a fixed id order, "
                    "not by price, so a cheaper fit may sit on pages never read: the code " + " and ".join(parts)
                    + " for you now and opens the cheapest unread rows below the pick. Then send the call again, or the same call "
                    "with a cheaper fit.")
        except Exception as err:
            run.log.append({"kind": "walk_hold_error", "error": f"{type(err).__name__}: {err}"[:300]})
            return ""

    # -- remarks on a search result
    @staticmethod
    def result_rows(piece):
        lists = [val for val in piece.values() if isinstance(val, list) and all(isinstance(r, dict) for r in val)]
        return next((val for val in lists if val), lists[0] if lists else None)

    def unwrap_rows(self, obs):
        rows = self.result_rows(obs)
        if rows is not None:
            return obs, rows
        # results may sit inside a nested dict beside a message
        inner = next((val for val in obs.values() if isinstance(val, dict) and self.result_rows(val) is not None), None)
        if inner is None:
            return None, None
        return inner, self.result_rows(inner) or []

    def score_rows(self, call_args, rows):
        watch = self.run.watch
        score = watch.value_score(call_args) if rows else 0
        if not score:
            return
        for row in rows:
            if isinstance(row, dict):
                mark = watch.row_mark(row)
                self.row_score[mark] = max(self.row_score.get(mark, 0), score)

    def search_note(self, call_args, obs):
        if not isinstance(obs, dict):
            return ""
        obs, rows = self.unwrap_rows(obs)
        if obs is None:
            return ""
        run = self.run
        watch = run.watch
        totals = [val for val in obs.values() if isinstance(val, int) and not isinstance(val, bool) and val > len(rows)]
        watch.book_page(call_args, rows, max(totals or [len(rows)]), obs)
        try:
            self.learn_filter(call_args, rows, max(totals or [len(rows)]), obs)
        except Exception as err:
            run.log.append({"kind": "ceil_learn_error", "error": f"{type(err).__name__}: {err}"[:300]})
        self.score_rows(call_args, rows)
        remarks = []
        kind = watch.word_rule(call_args, obs, rows)
        lead = watch.lead_rule(call_args, obs, rows)
        if lead:
            remarks.append(self.lead_remark(call_args, lead))
        texts, mode, first, paging_keys = watch.search_keys(call_args)
        raw = [val for key, val in call_args.items() if isinstance(val, str) and key not in dict(mode) and key not in paging_keys]
        if first and raw:
            self.query_remarks(call_args, rows, raw, totals, mode, paging_keys, texts, kind, remarks)
        if not rows:
            return " ".join(remarks)
        if totals:
            remarks.append(self.unseen_remark(kind, totals, rows))
        over_note = self.over_limit_remark(call_args, rows)
        if over_note:
            remarks.append(over_note)
        return " ".join(remarks)

    def over_limit_remark(self, call_args, rows):
        paging = self.run.watch.search_keys(call_args)[3]
        limits = [val for key, val in call_args.items() if is_number(val) and val > 1 and key not in paging
                  and not re.search(r"page|size|count|result|min|low|from|start", str(key).lower())]
        if not limits:
            return ""
        cap = max(limits)
        over = [r for r in rows if any(is_number(val) and val > cap and "price" in str(key).lower() for key, val in r.items())]
        if not over:
            return ""
        return (f"{len(over)} of these rows are priced above the limit {cap} this call set: the search mode ignored that filter "
                "(and may ignore others, such as the category). Use the mode whose filters hold, or check every row yourself.")

    def lead_remark(self, call_args, lead):
        context = self.run.lead_ctx
        texts, mode, _, _ = self.run.watch.search_keys(call_args)
        words = next((w for key, w in texts if key == context[0]), ()) if mode == context[1] else ()
        counted = f" (this query counted only: {' '.join(words[:lead])})" if len(words) > lead else ""
        return (f"This store's keyword search ({context[0]}) counts just the FIRST {lead} word(s) of a query and drops "
                f"the rest{counted}: put the {lead} most distinctive words first (a required value and the item "
                "kind, the value first), and search other values in separate short queries.")

    def hint_remarks(self, rows, raw, remarks):
        run = self.run
        # single-letter query pieces find nothing in some stores
        pieces = [p for t in raw for p in re.findall(r"[a-z0-9]+", t.lower())]
        if pieces and all(len(p) == 1 for p in pieces) and not rows and not self.short_hint_given:
            self.short_hint_given = True
            remarks.append("Nothing was found: one-character words are skipped by this store's search, so a figure like 0.4 or "
                           "5 x 7 searches nothing (or only its other words). Search a word value, or the unit word with the item "
                           "kind, and read every page.")
        prop, probe = run.watch.value_probe(raw)
        hints = self.name_hints
        if probe and hints < 2 and run.moves_left() > run.intake.order_due() + 3:
            self.name_hints = hints + 1
            if hints:
                opening = f"Again led by a property's name ({prop})."
            else:
                opening = (f"The query opens with a property's NAME ({prop}) and carries no required value: keyword search matches "
                           "listing values, titles and categories, and a field name only where a value or title repeats it, so "
                           f"listings whose {prop} field holds just the value are missed.")
            remarks.append(opening + f" Search the value first, then the item kind: '{probe}' (a figure the store cannot read drops out, "
                           "leaving its unit word), and read every page of it.")
        elif not probe and self.trap_hints < 2 and run.moves_left() > run.intake.order_due() + 3:
            trap, _ = run.walk.word_trap(raw)
            if trap:
                self.trap_hints += 1
                remarks.append(trap)

    def query_remarks(self, call_args, rows, raw, totals, mode, paging_keys, texts, kind, remarks):
        run = self.run
        self.hint_remarks(rows, raw, remarks)
        marks = tuple(run.watch.row_mark(r) for r in rows)
        others = tuple(sorted((k2, str(v2)) for k2, v2 in call_args.items()
                              if k2 not in paging_keys and not (isinstance(v2, str) and k2 not in dict(mode))))
        sig = (mode, others, max(totals or [len(rows)]), marks)
        query = " ".join(raw)
        earlier = self.page_signatures.get(sig)
        if len(set(marks)) >= 3 and sig in self.page_signatures and earlier != query:
            remarks.append(f"Your earlier search '{earlier}' gave the same total and rows: the store read both alike, so re-spelling "
                           "it shows nothing new. Read its next page, or change its leading words.")
        self.page_signatures.setdefault(sig, query)
        # fully shown result of an all-words store: every row holds the query words
        hits = []
        told = bool((kind == "all" or run.lead_words) and not totals and rows)
        if told or (kind != "any" and not totals and rows):
            hits = self.book_hits(call_args, rows, raw, told, texts, remarks)
        if (kind == "all" or run.lead_words) and not self.back_hits_done:
            self.back_hits_done = True
            self.earlier_hits(call_args, paging_keys, hits, remarks)

    def book_hits(self, call_args, rows, raw, told, texts, remarks):
        run = self.run
        watch = run.watch
        held = None if told else list(run.ref_hits or [])
        run.ref_hits = []
        hits = watch.row_hits(rows, raw)
        score = watch.value_score(call_args)
        self.hit_score.update({i: max(self.hit_score.get(i, 0), score) for i in hits})
        self.all_hits += [i for i in hits if i not in self.all_hits]
        if held is not None:
            run.ref_hits = held
        if told and hits and len(texts) and len(rows) <= 12:
            remarks.append("All words are required and all matches are shown here: every row holds the words the store read "
                           "somewhere in its record, often in fields its title does not show. Open the in-budget ones before "
                           "searching on: " + ", ".join(hits[:6]) + ".")
        pointers = []
        if told:
            popped, run.ref_hits = run.ref_hits, []
            pointers = [i for i in popped if i not in hits][:6]
        if pointers:
            remarks.append("A value in the request points at another listing, and the shop or brand this search names sells these "
                           "rows: open them whatever their price, stock or category (a listing that gives a value "
                           "is read, not bought), and follow what its notes say: " + ", ".join(pointers) + ".")
        return hits

    def earlier_hits(self, call_args, paging_keys, hits, remarks):
        watch = self.run.watch
        here = json.dumps({k: v for k, v in call_args.items() if k not in paging_keys}, sort_keys=True, default=str)
        back = []
        for walk_key, entry in self.run.page_book.items():
            if walk_key == here or not entry.get("every") or not watch.value_score(entry["args"]):
                continue
            fresh = [i for i in watch.row_hits(entry["every"]) if i not in back and i not in hits]
            self.hit_score.update({i: max(self.hit_score.get(i, 0), watch.value_score(entry["args"])) for i in fresh})
            back += fresh
        self.all_hits += [i for i in back if i not in self.all_hits]
        if back:
            remarks.append("Every word is needed by this store's keyword search, so earlier fully shown searches listed only "
                           "rows holding all of their words, often in fields the titles do not show. Open the in-budget ones not "
                           "read yet before searching on: " + ", ".join(back[:6]) + ".")

    @staticmethod
    def unseen_remark(kind, totals, rows):
        if kind == "any":
            how = ("this store's keyword search accepts ANY of the words and ranks listings that match more of them higher, so "
                   "more words widen it: a required value's own distinctive word or figure alone lists only the listings "
                   "carrying it (often few enough to read every page and compare prices); the item kind with the values ranks "
                   "the closest matches first; use the category mode (or a price filter if one holds) for the rest.")
        elif kind == "all":
            how = ("this store's keyword search wants EVERY word, so extra exact words narrow it, as do the category and price "
                   "filters with a lower price limit.")
        else:
            how = ("then narrow it for the rest: when every word is needed, add exact words; when any word matches (a longer "
                   "query matching more listings shows that), put the required values in the query so fitting listings rank first; "
                   "a price filter, when one holds, also helps.")
        return (f"{max(totals)} listings matched this search, yet this page shows only {len(rows)}, and just a few pages can "
                "be read: the listings that matter may be among the unseen ones. Read the further pages of this same search "
                "first (as many as can be read); if the total is larger than those pages, " + how)

    def shopper_lines(self, call_name, obs, refused):
        lines = []
        if call_name == self.run.role("message"):
            if isinstance(obs, str):
                return [obs]
            if isinstance(obs, dict) and "error" not in obs:
                long_strings([val for val in obs.values() if isinstance(val, (str, list))], lines)
            return lines
        if not isinstance(obs, dict):
            return lines
        for piece in obs.values():
            if isinstance(piece, list) and piece and all(isinstance(val, str) for val in piece):
                long_strings(piece, lines)
            elif isinstance(piece, str) and len(piece.split()) >= 4 and refused:
                lines.append(piece)
        return lines


class ReadQueue:
    # listings queued for the code to open, and how the opening is announced
    def __init__(self):
        self.open_queue = None
        self.open_say = ""
        self.open_recheck = False
        self.open_purpose = ""
        self.open_floor = None
        self.open_tool = None
        self.put_queue = None
        self.put_tool = None


class Intake:
    def __init__(self, run):
        self.run = run
        self.read_content = None
        self.read_at = None
        self.steps_given = None
        self.menu_memo = {}
        self.hint_alts = {}
        self.twin_asks = set()
        self.first_reads_done = None
        self.late_q = 0
        self.late_ids = set()
        self.backup_n = 0
        self.swap_done = None

    # reading the request

    def read_request(self):
        run = self.run
        listing = []
        for entry in run.toolset:
            spec = entry["function"]
            listing.append(f"- {spec['name']}: {spec.get('description', '')} | arguments: {as_text(spec.get('parameters') or {})}")
        tail = "REQUEST:\n" + run.request
        opening = run.pview.get("opening")
        if opening is not None:
            tail = tail + "\n\nOPENING OBSERVATION:\n" + as_text(opening)
        parts = [
            {"type": "text", "text": READ_REQUEST + "\n\nTOOLS:\n" + "\n".join(listing), "cache_control": {"type": "ephemeral"}},
            {"type": "text", "text": tail},
        ]
        self.read_content = parts
        reading = self.ask_readers(parts)
        if not isinstance(reading.get("roles"), dict):
            reading = self.reading_by_code(reading)
        self.guarded(lambda: self.check_reading(reading), "check_reading_error")
        self.use_reading(reading)

    def check_reading(self, reading):
        run = self.run
        text = run.request
        figures = reading.get("figures") if isinstance(reading.get("figures"), dict) else {}
        for key in ("price_cap", "message_budget", "step_budget"):
            value = reading.get(key)
            if not is_number(value):
                continue
            # a limit stands only on a figure the request writes; the quoted words may carry the right one instead
            if float(value) in written_figures(text):
                continue
            quote = str(figures.get(key) or "")
            found = [number for number in written_figures(quote) if number in written_figures(text)] if matching_sentence(quote, text) else []
            reading[key] = (int(found[0]) if found[0] == int(found[0]) else found[0]) if len(found) == 1 else None
            run.log.append({"kind": "read_check", "field": key, "read": value, "kept": reading[key]})
        steps = []
        for step in reading.get("procedures") or []:
            if isinstance(step, str):
                step = {"quote": step}
            if not isinstance(step, dict):
                continue
            sentence = matching_sentence(str(step.get("quote") or ""), text)
            if not sentence:
                run.log.append({"kind": "read_check", "field": "procedure", "dropped": str(step.get("quote"))[:200]})
                continue
            steps.append({"quote": sentence, "needs_yes": step.get("needs_yes") is not False, "final_cart": step.get("final_cart") is not False})
        rules = [str(rule) for rule in reading.get("rules") or [] if rule]
        if not steps and rules:
            steps = self.steps_in_rules(rules)
        quotes = [set(plain_words(step["quote"])) for step in steps]
        # a go-ahead read twice, as a step and as something to ask first, is asked only as the step, at its time
        reading["ask_first"] = [item for item in reading.get("ask_first") or [] if not any(
            words and len(words & set(plain_words(item))) >= 0.6 * max(1, len(set(plain_words(item)))) for words in quotes)]
        reading["procedures"] = steps
        run.procedures = steps
        names = [" ".join(plain_words(row.get("product"))) for row in reading.get("items") or [] if isinstance(row, dict)]

        def slot_of(name):
            words = " ".join(plain_words(name))
            hits = [spot for spot, item in enumerate(names) if item and (item == words or item in words or words in item)]
            return hits[0] if len(hits) == 1 else None
        fits = []
        for pair in reading.get("fit_with") or []:
            if not isinstance(pair, dict) or not matching_sentence(str(pair.get("quote") or ""), text):
                continue
            dep, anchor = slot_of(pair.get("product")), slot_of(pair.get("with"))
            if dep is not None and anchor is not None and dep != anchor and (dep, anchor) not in fits:
                fits.append((dep, anchor))
        run.pairs = fits
        run.log.append({"kind": "pairs", "pairs": fits})
        run.log.append({"kind": "procedures", "steps": steps})

    def steps_in_rules(self, rules):
        run = self.run
        listed = "\n".join(f"{spot}. {rule}" for spot, rule in enumerate(rules))
        prompt = (STEP_CHECK + "\n\nRULES:\n" + listed)
        answer = run.wire.query_model([{"role": "user", "content": prompt}], LISTENER, 300)
        picked = (json_object((answer or {}).get("content") or "") or {}).get("steps") or []
        steps = []
        for spot in picked:
            if isinstance(spot, int) and 0 <= spot < len(rules):
                sentence = matching_sentence(rules[spot], run.request)
                if sentence:
                    steps.append({"quote": sentence, "needs_yes": True, "final_cart": True})
        run.log.append({"kind": "rule_steps", "rules": len(rules), "steps": steps})
        return steps

    def ask_readers(self, parts):
        wire = self.run.wire
        ladder = READER + LISTENER[:1]
        reading = {}
        for turn, model in enumerate(ladder * 3):
            second_round = turn >= len(ladder)
            if second_round and self.readers_gone(ladder):
                break
            answer = wire.query_model([{"role": "user", "content": parts}], [model], 3000)
            reading = json_object((answer or {}).get("content") or "") or {}
            if isinstance(reading.get("roles"), dict):
                break
            wire.log.append({"kind": "setup_retry", "model": model})
        return reading

    def readers_gone(self, ladder):
        wire = self.run.wire
        return (wire.misses == 0 or wire.call_fail == "refused" or wire.down()
                or not wire.usable(ladder) or wire.spare() < CLOSING + 60)

    def reading_by_code(self, partial):
        wire = self.run.wire
        try:
            kept = {key: val for key, val in partial.items() if key != "roles"}
            reading = {**self.run.closer.code_reading(), **kept}
        except Exception as err:
            wire.log.append({"kind": "code_reading_error", "error": f"{type(err).__name__}: {err}"[:300]})
            reading = {}
        self.read_at = monotonic()
        wire.log.append({"kind": "code_reading", "roles": reading.get("roles"), "product": reading.get("product"),
                         "price_cap": reading.get("price_cap"), "items": len(reading.get("items") or [])})
        return reading

    def use_reading(self, reading, again=False):
        run = self.run
        needs, wire = run.needs, run.wire
        if not again:
            self.steps_given = run.step_cap
        cart_before = run.role("cart")
        raw_roles = reading.get("roles")
        if not isinstance(raw_roles, dict):
            raw_roles = {}
        run.role_map = {name: tool for name, tool in raw_roles.items() if isinstance(tool, str) and tool in run.shapes}
        keep_cart = again and run.trolley.used
        if keep_cart and cart_before:
            run.role_map["cart"] = cart_before
        if not keep_cart:
            run.trolley = CartBook(run.role("cart"), run.shapes.get(run.role("cart") or "", {}))
        needs.sweepable = bool(run.role("search")) and run.watch.page_arg() is not None
        budget = reading.get("message_budget")
        if is_number(budget) and budget >= 0:
            run.msg_cap = int(budget)
        needs.reading = reading
        try:
            extra_products = max(0, len(needs.items()) - 1)
            wire.scale = 1.0 + 0.3 * min(2, extra_products)
        except Exception:
            wire.scale = 1.0
        self.split_requirements(reading)
        self.guarded(lambda: self.story_values(reading), "story_values_error")
        self.guarded(run.refdesk.ref_register, "ref_register_error")
        if not again or needs.version <= 0:
            needs.spend_cap = reading.get("price_cap")
        if reading and needs.spend_cap is None and run.role("message"):
            ask_more = [str(a) for a in reading.get("ask_first") or [] if a]
            reading["ask_first"] = ["the most the shopper wants to spend, since the request states no price limit (skip if the same thing is listed again here)"] + ask_more
        if again:
            run.step_cap = run.step_cap if self.steps_given is None else self.steps_given
        allowed = reading.get("step_budget")
        if is_number(allowed) and 0 < allowed < run.step_cap:
            run.step_cap = int(allowed)
        if again:
            needs.version += 1

    def split_requirements(self, reading):
        needs = self.run.needs
        named = [str(req) for req in reading.get("requirements") or [] if req]
        proper = []
        for req in named:
            head, colon, value = req.partition(":")
            if colon and head.strip() and value.strip():
                proper.append(req)
        needs.written = proper
        stray = [req for req in named if req not in needs.written]
        if not stray:
            return
        wishes = [str(w) for w in reading.get("soft_wants") or [] if w]
        reading["soft_wants"] = wishes + stray
        reading["requirements"] = list(needs.written)

    def guarded(self, action, kind):
        try:
            action()
        except Exception as err:
            self.run.wire.log.append({"kind": kind, "error": f"{type(err).__name__}: {err}"[:300]})

    @staticmethod
    def hint_list(raw):
        if isinstance(raw, dict):
            flat = []
            for key, vals in raw.items():
                for val in vals if isinstance(vals, list) else [vals]:
                    if val:
                        flat.append(f"{val}" if ":" in str(val) else f"{key}: {val}")
            raw = flat
        if not isinstance(raw, (list, tuple)):
            return []
        return [str(h).strip() for h in raw or [] if isinstance(h, (str, int, float)) and str(h).strip()]

    @staticmethod
    def req_core(req):
        head, _, value = str(req).partition(":")
        return re.split(r"\s*\(", head)[0].strip().lower(), value.strip().lower()

    def story_values(self, reading):
        run = self.run
        needs = run.needs
        hints = [one_prop_hint(h) for h in self.hint_list(reading.get("hints"))]
        reader_hints = list(hints)
        given = reading.get("references")
        refs = [str(r) for r in given or []] if isinstance(given, list) else []
        guessed = [req for req in needs.written if story_guess(req, run.request, refs)]
        if guessed and len(needs.written) >= 2 and len(guessed) * 2 <= len(needs.written):
            needs.written = [req for req in needs.written if req not in guessed]
            reading["requirements"] = list(needs.written)
            dropped = {self.req_core(req) for req in guessed}
            for row in reading.get("items") or []:
                if not isinstance(row, dict) or not isinstance(row.get("requirements"), list):
                    continue
                row["requirements"] = [r for r in row["requirements"] if self.req_core(r) not in dropped]
            for req in guessed:
                prop, _, value = req.partition(":")
                hints.append(f"{prop.strip()}: the story suggests {value.strip()} (not stated as a value; ask)")
            run.wire.log.append({"kind": "story_values", "moved": guessed})
        try:
            promoted = self.written_hints(reading, reader_hints)
            hints = [h for h in hints if h not in promoted]
        except Exception as err:
            run.wire.log.append({"kind": "written_hint_error", "error": f"{type(err).__name__}: {err}"[:300]})
        reading["hints"] = hints[:12]

    def hint_word(self, detail, names):
        word, name = "", ""
        for name in names:
            word = lead_value(detail, self.run.request, name)
            if word:
                break
        return word, name

    def clashes(self, prop, word, mine):
        for req in self.run.needs.written:
            rhead, _, rvalue = req.partition(":")
            rtag = re.findall(r"\(([^)]*)\)", rhead)
            other_product = rtag and mine and not (kind_words(rtag[-1]) & mine)
            if other_product:
                continue
            if same_prop(rhead, prop) or re.search(r"\b" + re.escape(word) + r"\b", rvalue.lower()):
                return True
        return False

    def written_hints(self, reading, hints):
        run = self.run
        needs = run.needs
        rows = [r for r in reading.get("items") or [] if isinstance(r, dict)]
        promoted = []
        for hint in hints:
            head, sep, detail = str(hint).partition(":")
            prop = re.split(r"\s*\(", head)[0].strip()
            if not (sep and prop) or junk_key(prop):
                continue
            tags = re.findall(r"\(([^)]*)\)", head)
            if tags:
                names = [tags[-1]]
            else:
                names = [str(r.get("product") or "") for r in rows] or [str(reading.get("product") or "")]
            word, name = self.hint_word(detail, names)
            if not word:
                continue
            mine = kind_words(name)
            if self.clashes(prop, word, mine):
                continue
            req = f"{head.strip()}: {word}"
            needs.written.append(req)
            reading["requirements"] = list(needs.written)
            target = next((row for row in rows if isinstance(row.get("requirements"), list)
                           and (len(rows) == 1 or kind_words(row.get("product") or "") & mine)), None)
            if target is not None:
                target["requirements"].append(req)
            promoted.append(hint)
            run.wire.log.append({"kind": "written_hint", "hint": str(hint)[:200], "requirement": req})
        return promoted

    # story hints and the question menu

    def several(self):
        return bool(self.run.needs.items())

    def slot_of(self, text):
        return self.run.talk.product_of(text) if self.several() else None

    def open_hints(self, k=None):
        talk = self.run.talk
        fresh, reopened = [], []
        for hint in self.run.needs.reading.get("hints") or []:
            prop = str(hint).split(":", 1)[0].strip()
            if not prop:
                continue
            slot = self.slot_of(prop)
            if k is not None and slot not in (k, None):
                continue
            own = slot if self.several() else None
            still_open = talk.open_prop(own, prop, self.prop_line(slot, prop), by_samples=True)
            if still_open and not self.asked_inside(own, prop):
                fresh.append(str(hint))
                continue
            alt = self.hint_alt(own, str(hint), prop)
            if alt:
                reopened.append(alt)
        return reopened + fresh

    def build_menu(self, k):
        run = self.run
        archive, shelf = run.archive, run.shelf
        if not self.several():
            own = shelf.own_shelf()
            return archive.prop_menu(own)[0] if own else archive.prop_menu()[0]
        homes = shelf.home_shelves()
        home = homes[k] if k is not None and k < len(homes) else set()
        shelves = [s for s in archive.shelves_read() if s.strip().lower() in home]
        return archive.prop_menu(shelves[0])[0] if shelves else ""

    def prop_line(self, k, prop):
        try:
            stamp = len(self.run.archive.items)
            cached = self.menu_memo.get(k)
            if cached and cached[0] == stamp:
                menu = cached[1]
            else:
                menu = self.build_menu(k)
                self.menu_memo[k] = (len(self.run.archive.items), menu)
            for line in menu.splitlines():
                if strict_prop(self.menu_key(line), prop):
                    return line
        except Exception:
            return ""
        return ""

    @staticmethod
    def menu_key(line):
        return line.strip("- ").split(" (in ")[0].strip()

    def cached_menu(self, k):
        return (self.menu_memo.get(k) or (0, ""))[1]

    @staticmethod
    def plain_words(value):
        return " ".join(re.findall(r"[a-z0-9]+", value.lower()))

    def field_values(self, k, prop):
        line = self.prop_line(k, prop)
        if "):" not in line:
            return set()
        return {self.plain_words(v) for v in line.split("):", 1)[1].split("|")} - {""}

    def same_field(self, k, a, b):
        if strict_prop(a, b):
            return True
        try:
            if not same_prop(a, b):
                return False
            return bool(self.field_values(k, a) & self.field_values(k, b))
        except Exception:
            return False

    def hinted_prop(self, k, prop):
        name = str(prop).split("(")[0].strip()
        for hint in self.run.needs.reading.get("hints") or []:
            head = str(hint).split(":", 1)[0]
            if not strict_prop(head, name):
                continue
            if k is None or not self.several() or self.run.talk.product_of(head) in (k, None):
                return True
        return False

    def product_reads(self, k):
        run = self.run
        try:
            if not self.several():
                own = run.shelf.own_shelf()
                return run.archive.rich_reads(own) if own else run.archive.rich_reads()
            homes = run.shelf.home_shelves()
            home = homes[k] if k is not None and k < len(homes) else set()
            if not home:
                return 0
            return sum(run.archive.rich_reads(s) for s in home)
        except Exception:
            return 0

    def plain_hint(self, k, prop, hinted=False):
        try:
            name = re.split(r"\s*\(", str(prop))[0].strip()
            if not name:
                return False
            if not hinted and not self.hinted_prop(k, name):
                return False
            if self.prop_line(k, name) or self.product_reads(k) < 2:
                return False
            return all(not same_prop(self.menu_key(line), name) for line in self.cached_menu(k).splitlines())
        except Exception:
            return False

    def hint_mark(self, hint):
        try:
            prop = str(hint).split(":", 1)[0].strip()
            marked = self.plain_hint(self.slot_of(prop), prop, hinted=True)
            return f"{hint} [no key]" if marked else str(hint)
        except Exception:
            return str(hint)

    @staticmethod
    def hint_alts_of(detail):
        found = []
        for group in re.findall(r"\(or ([^()]+)\)", str(detail)):
            found.extend(piece.strip() for piece in re.split(r",|\bor\b|/", group) if piece.strip())
        return found

    @staticmethod
    def long_words(name):
        return {w for w in prop_words(name) if len(w) >= 3 and w not in LINK_WORDS}

    @staticmethod
    def name_inside(a, b):
        left, right = Intake.long_words(a), Intake.long_words(b)
        if not left or not right:
            return False
        return left <= right or right <= left

    @staticmethod
    def asked_text(line):
        return re.search(r'asked "(.*?)" -> ', line)

    def asked_inside(self, k, head):
        talk = self.run.talk
        try:
            several = self.several()
            for line in self.run.questions_log:
                label = line.split(":", 1)[0].strip("- ")
                if "[not answered]" in label:
                    continue
                said = self.asked_text(line)
                for part in label.split(";"):
                    slot = talk.product_of(part) if several else None
                    if several and slot is None and said:
                        slot = talk.product_of(said.group(1))
                    fits_product = not several or k is None or slot in (k, None)
                    if fits_product and self.name_inside(part, head):
                        return True
        except Exception:
            return False
        return False

    def quiet_on(self, k, head):
        several = self.several()
        for asked in self.run.talk.quiet_asks or []:
            if not (self.same_field(k, asked, head) or self.name_inside(asked, head)):
                continue
            if not several or self.run.talk.product_of(asked) in (k, None):
                return True
        return False

    def hint_alt(self, k, hint, head):
        try:
            memo = self.hint_alts
            if not memo.get(hint):
                if not self.quiet_on(k, head):
                    return ""
                for alt in self.hint_alts_of(hint.partition(":")[2]):
                    line = "" if junk_key(alt) or strict_prop(alt, head) else self.prop_line(k, alt)
                    if line:
                        memo[hint] = self.menu_key(line)
                        break
            key = memo.get(hint) or ""
            if not key or key.lower() in self.run.talk.spent_asks or not self.plan_open(k, key):
                return ""
            front = hint.partition(":")
            tag = re.search(r"\(([^()]*)\)\s*$", front[0])
            detail = re.sub(r"\s*\(or [^()]+\)", "", front[2]).strip()
            label = key + (" (" + tag.group(1).strip() + ")" if tag else "")
            return f"{label}: {detail} (its first property, {head}, drew no preference)"
        except Exception:
            return ""

    def hint_mapped(self, k, prop):
        try:
            name = re.split(r"\s*\(", str(prop))[0].strip()
            words = [w for w in prop_words(name) if len(w) >= 3 and w not in LINK_WORDS]
            several = self.several()
            for hint in self.run.needs.reading.get("hints") or []:
                head, _, detail = str(hint).partition(":")
                if several and k is not None and self.run.talk.product_of(head) not in (k, None):
                    continue
                names = [re.split(r"\s*\(", head)[0].strip()] + self.hint_alts_of(detail)
                if any(n and strict_prop(n, name) for n in names):
                    return True
                text = str(hint).lower()
                if words and all(re.search(r"\b" + re.escape(w), text) for w in words):
                    return True
        except Exception:
            return False
        return False

    @staticmethod
    def name_twin(a, b):
        if strict_prop(a, b):
            return False

        def split(name):
            every = [w for w in prop_words(name) if len(w) >= 3 and w not in LINK_WORDS]
            leading = every[:-1] if len(every) > 1 else every
            return every, leading

        def close(x, y):
            if x == y:
                return True
            return min(len(x), len(y)) >= 4 and (x.startswith(y) or y.startswith(x))
        all_a, lead_a = split(a)
        all_b, lead_b = split(b)
        for x in lead_a:
            if any(close(x, y) for y in all_b):
                return True
        return any(close(x, y) for x in all_a for y in lead_b)

    def twin_of(self, k, a, b):
        try:
            if strict_prop(a, b):
                return False
            return self.name_twin(a, b) or self.same_field(k, a, b)
        except Exception:
            return False

    def story_text(self, k):
        run = self.run
        several = self.several()
        pieces = [run.request.lower()]
        for hint in run.needs.reading.get("hints") or []:
            if not several or run.talk.product_of(str(hint).split(":", 1)[0]) in (k, None):
                pieces.append(str(hint).lower())
        return " ".join(pieces)

    def product_name(self, k, fallback=""):
        needs = self.run.needs
        if self.several() and k is not None:
            return needs.items()[k]["product"]
        return str(needs.reading.get("product") or fallback)

    def twin_key(self, k, prop):
        try:
            self.prop_line(k, prop)
            menu = self.cached_menu(k)
            own = kind_tokens(self.product_name(k))
            story = self.story_text(k)
            spent = self.run.talk.spent_asks
            mine = set(prop_words(prop))
            for line in menu.splitlines():
                key = self.menu_key(line)
                if not key or junk_key(key) or key.lower() in spent:
                    continue
                if not self.twin_of(k, key, prop) or not self.plan_open(k, key) or self.name_inside(key, prop):
                    continue
                fresh = [w for w in prop_words(key)
                         if len(w) >= 3 and w not in LINK_WORDS and w not in mine and not kind_in(w, own)]
                if any(re.search(r"\b" + re.escape(w), story) for w in fresh) or self.hint_mapped(k, key):
                    return key
        except Exception:
            return ""
        return ""

    def values_form(self, k, prop, product):
        try:
            line = self.prop_line(k, prop)
            if "):" not in line:
                return ""
            others = set()
            for other in self.cached_menu(k).splitlines():
                if "):" in other and other.strip() != line.strip():
                    others.update(self.plain_words(v) for v in other.split("):", 1)[1].split("|"))
            picks = []
            for value in line.split("):", 1)[1].split("|"):
                value = value.strip()
                if not value or len(value) > 30 or not re.search(r"[a-z]", value.lower()) or re.search(r"[,;/]", value):
                    continue
                plain = self.plain_words(value)
                if plain in others or plain in [self.plain_words(p) for p in picks]:
                    continue
                picks.append(value)
            picks = picks[:3]
            if len(picks) < 2:
                return ""
            return f"For the {product}, would you like {', '.join(picks[:-1])} or {picks[-1]}?"
        except Exception:
            return ""

    def twin_target(self, prop):
        multi = self.several()
        slot = self.run.talk.product_of(prop) if multi else None
        if multi and slot is None:
            return None
        mark = slot if multi else -1
        if mark in self.twin_asks:
            return None
        return slot, mark

    def labelled(self, key, slot):
        if not self.several():
            return key
        return f"{key} ({self.run.needs.items()[slot]['product']})"

    def distinct_ask(self, prop):
        try:
            target_slot = self.twin_target(prop)
            if target_slot is None:
                return None
            slot, mark = target_slot
            if self.several():
                product = self.product_name(slot, "item")
            else:
                product = str(self.run.needs.reading.get("product") or "item")
            field = re.split(r"\s*\(", str(prop))[0].strip()
            text = self.values_form(slot, field, product)
            if not text:
                twin = self.twin_key(slot, prop)
                if twin:
                    field = twin
                    text = self.values_form(slot, twin, product) or f"Which {twin} would you like for the {product}?"
            if not text:
                return None
            self.twin_asks.add(mark)
            return (f"{field} ({product})" if self.several() else field), text
        except Exception:
            return None

    def asked_for(self, k, prop):
        talk = self.run.talk
        try:
            several = self.several()
            name = re.split(r"\s*\(", str(prop or ""))[0].strip()
            if not name:
                return False
            for line in self.run.questions_log:
                label = re.sub(r"\s*\[[^\]]*\]", "", line.split(":", 1)[0].strip("- "))
                said = self.asked_text(line)
                parts = [p.strip() for p in re.split(r";|\band\b", label) if p.strip()]
                for part in parts:
                    slot = talk.product_of(part) if several else None
                    if several and slot is None and said:
                        slot = talk.product_of(said.group(1))
                    if several and k is not None and slot != k:
                        continue
                    if strict_prop(re.split(r"\s*\(", part)[0].strip(), name):
                        return True
        except Exception:
            return False
        return False

    def alt_usable(self, k, alt, head, name, spent):
        if junk_key(alt) or strict_prop(alt, head) or strict_prop(alt, name) or self.name_inside(alt, name):
            return ""
        line = self.prop_line(k, alt)
        key = self.menu_key(line) if line else ""
        if not key or strict_prop(key, name) or key.lower() in spent:
            return ""
        if self.asked_for(k, key) or not self.plan_open(k, key):
            return ""
        return key

    def alt_key(self, k, prop):
        try:
            name = re.split(r"\s*\(", str(prop))[0].strip()
            if not name:
                return ""
            several = self.several()
            spent = self.run.talk.spent_asks
            for hint in self.run.needs.reading.get("hints") or []:
                full_head, _, detail = str(hint).partition(":")
                head = re.split(r"\s*\(", full_head)[0].strip()
                if not head:
                    continue
                if several and self.run.talk.product_of(full_head) not in (k, None):
                    continue
                if not self.same_field(k, name, head) and not self.name_inside(name, head):
                    continue
                for alt in self.hint_alts_of(detail):
                    key = self.alt_usable(k, alt, head, name, spent)
                    if key:
                        return key
        except Exception:
            return ""
        return ""

    def collide_alt(self, prop):
        try:
            target_slot = self.twin_target(prop)
            if target_slot is None:
                return ""
            slot, mark = target_slot
            key = self.alt_key(slot, prop)
            if key:
                self.twin_asks.add(mark)
                return self.labelled(key, slot)
            return ""
        except Exception:
            return ""

    def holds_plain(self, text, plain_q):
        stem = self.run.watch.stem
        filler = {"which", "would", "you", "like", "for", "the"}
        present = {stem(w) for w in re.findall(r"[a-z0-9]+", str(text).lower())}
        wanted = [stem(w) for w in re.findall(r"[a-z0-9]+", str(plain_q).lower()) if w not in filler and w not in LINK_WORDS]
        return bool(wanted) and all(w in present for w in wanted)

    def alt_untried(self, k):
        talk = self.run.talk
        try:
            hints = self.run.needs.reading.get("hints") or []
            for full_head in (str(h).split(":", 1)[0] for h in hints):
                if talk.product_of(full_head) in (k, None):
                    head = re.split(r"\s*\(", full_head)[0].strip()
                    if head and self.asked_for(k, head) and self.alt_key(k, head):
                        return True
            return False
        except Exception:
            return False

    def others_unasked(self, k):
        talk = self.run.talk
        log = self.run.questions_log
        reached = {talk.product_of(line.split(":", 1)[0]) for line in log}
        for line in log:
            said = self.asked_text(line)
            if said:
                reached.add(talk.product_of(said.group(1)))
        return any(j != k and j not in reached for j in range(len(self.run.needs.items())))

    def quiet_reask(self, prop):
        try:
            target_slot = self.twin_target(prop)
            if target_slot is None:
                return ""
            slot, mark = target_slot
            if not self.hint_mapped(slot, prop):
                return ""
            reopened = [h for h in self.open_hints(slot) if " drew no preference)" in h]
            if reopened or (self.several() and self.others_unasked(slot)):
                return ""
            twin = self.twin_key(slot, prop)
            if not twin:
                return ""
            self.twin_asks.add(mark)
            return self.labelled(twin, slot)
        except Exception:
            return ""

    @staticmethod
    def menu_trim(menu, reads):
        if reads < 4 or not menu:
            return menu
        kept = [line for line in menu.splitlines() if not re.search(r"\(in [01] listings?\)", line)]
        return "\n".join(kept) if kept else menu

    def first_reads(self):
        run = self.run
        try:
            if self.first_reads_done or run.shelf.batch_tool() or not run.role("details"):
                return False
            self.first_reads_done = True
            rows = run.needs.items()
            homes = run.shelf.home_shelves()
            counts = [row["quantity"] for row in rows] or [run.needs.quantity()]
            room = run.moves_left() - self.order_due() - 2 * max(0, run.ask_limit() - run.msgs_sent) - 8
            wants, taken = [], []
            for k in range(len(rows) or 1):
                home = homes[k] if k < len(homes) else set()
                have = self.product_reads(k if rows else None)
                if not home or have >= 2 or room <= 0:
                    continue
                picks = run.shelf.menu_rows(home, 2 - have, taken, counts[min(k, len(counts) - 1)])
                taken += picks
                if picks:
                    wants.append((have, k, picks))
            ordered = sorted(wants)
            queue = []
            for n in range(2):
                queue.extend(picks[n] for _, _, picks in ordered if n < len(picks))
            queue = queue[: max(0, room)]
            if not queue:
                return False
            reads = run.reads
            reads.open_queue = queue
            reads.open_recheck = True
            reads.open_say = "To learn their fields, I first open a few listings of each product requested before any question."
            reads.open_purpose = "menu"
            run.log.append({"kind": "first_reads", "ids": queue})
            return True
        except Exception as err:
            run.log.append({"kind": "first_reads_error", "error": f"{type(err).__name__}: {err}"[:300]})
            return False

    def plan_open(self, k, prop):
        talk = self.run.talk
        try:
            hinted = self.hinted_prop(k, prop)
            line = self.prop_line(k, prop) if hinted else ""
            return talk.open_prop(k, prop, line, by_samples=hinted)
        except Exception:
            return talk.open_prop(k, prop)

    def settles(self, req, samples=""):
        run = self.run
        value = str(req).partition(":")[2]
        low = value.lower()
        if re.search(r"as shown|same as|the page|sold by|redirect|refer", low) or ref_text(low):
            return True
        if not positive_value(value, str(req).partition(":")[0]):
            return False
        given = run.needs.reading.get("references")
        refs = [str(r) for r in given or []] if isinstance(given, list) else []
        refs += list(run.needs.resolved or [])
        if story_guess(req, run.request + " " + " ".join(run.needs.exchanges), refs):
            return False
        if not samples or re.search(r"\d", value):
            return True
        words = [w for w in re.findall(r"[a-z]{3,}", low) if w not in LINK_WORDS]
        if not words:
            return True
        pool = set(re.findall(r"[a-z]+", samples))
        for word in words:
            if re.search(r"\b" + re.escape(word[:-1] if len(word) > 4 else word), samples):
                return True
            if len(word) >= 4 and any(len(s) >= 4 and edits_within(s, word, 1) for s in pool):
                return True
        return False

    # which tools the actor may use

    def eventful(self):
        run = self.run
        if run.event_seen:
            return True
        dead = run.shelf.dead
        return bool(dead and set(run.trolley.units) & dead)

    def order_due(self):
        cap = self.run.step_cap
        floor = max(FIRST_ORDER_BY, min(FIRST_ORDER_DUE, cap // 4 + 1))
        due = max(floor, min(16, cap // 3))
        if self.eventful():
            try:
                due = max(due, 8 + max(1, len(self.run.needs.items())))
            except Exception:
                due = max(due, 9)
        return max(floor, min(due, cap // 2))

    def narrowing(self):
        return bool(self.run.role("order") and self.run.role("cart"))

    def reserve_names(self, spare):
        run = self.run
        role = run.role
        order, decline, cart = role("order"), role("decline"), role("cart")
        if spare <= 1:
            return {order, decline}
        if spare <= 3:
            return {order, decline, cart}
        late_window = (spare <= self.order_due() and run.orders_tried == 0) or (spare <= FIRST_ORDER_BY and run.refusal_n > 0)
        if not late_window:
            return None
        names = {order, decline, cart, role("cart_view"), role("view") or role("details"), run.shelf.stock_tool()}
        if spare > 3 and run.shelf.swap_open():
            names.add(role("details"))
        few_late = run.late_n < 3
        if few_late:
            if (self.unverified() and spare > 5) or (self.named_open() and spare >= min(FIRST_ORDER_BY + 2, self.order_due())):
                names.add(role("details"))
            if (self.widened_late() or run.judge.named_pages or (self.unverified() and spare > 5)) and spare > 4:
                names.add(role("search"))
        if spare >= 7 and self.backup_rows():
            names.add(role("details"))
        return names

    def offered(self):
        run = self.run
        names = self.reserve_names(run.moves_left())
        tools = run.toolset
        if names is not None and run.role("order") and run.role("cart"):
            tools = [entry for entry in run.toolset if entry["function"]["name"] in names] or tools
        messenger = run.role("message")
        if run.asking and messenger:
            tools = [entry for entry in tools if entry["function"]["name"] != messenger]
        return tools

    def in_reserve(self):
        run = self.run
        spare = run.moves_left()
        if not (run.role("order") and run.role("cart")) or spare <= 3:
            return False
        if spare <= self.order_due() and run.orders_tried == 0:
            return True
        return spare <= FIRST_ORDER_BY and run.refusal_n > 0

    def unverified(self):
        run = self.run
        dead = run.shelf.dead_lines()
        version = str(run.needs.version)
        live = []
        for key in run.passed:
            thing, _, stamp = key.rpartition("|")
            if stamp == version and thing not in run.refused_lines and thing not in dead:
                live.append(thing)
        if not live:
            return True
        if not self.several():
            return False
        try:
            known = {run.judge.shelf_known(i) for i in live} - {""}
            if not known:
                return False
            return any(home and not home & known for home in run.shelf.home_shelves())
        except Exception:
            return False

    def named_open(self):
        run = self.run
        seen = {i for i in run.judge.named if i in run.archive.ids}
        return seen - run.archive.read - run.refused_lines

    def widened_late(self):
        run = self.run
        needs = run.needs
        return needs.widened_at == needs.version and run.wire.tick - needs.widened_tick <= self.order_due()

    def begin_batch(self):
        self.late_q = 0
        self.late_ids = set()

    def spare_backups(self):
        left = 2 - self.backup_n
        return [i for i in self.backup_rows() if i not in self.late_ids][:left]

    def late_allowed(self, call_name, call_args, named, spare):
        run = self.run
        if call_name == run.role("details"):
            if self.unverified() and spare > 5:
                return True
            return bool(named & self.named_open()) and spare >= min(FIRST_ORDER_BY + 2, self.order_due())
        widened = self.widened_late() or json.dumps(call_args, sort_keys=True, default=str) in run.judge.named_pages
        return (widened or (self.unverified() and spare > 5)) and spare > 4

    def late_hold(self, call_name, call_args):
        run = self.run
        details, search = run.role("details"), run.role("search")
        if call_name not in (details, search) and call_name in run.shelf.lookup_tools() and details:
            return self.late_hold(details, call_args)
        if call_name not in (details, search) or not self.in_reserve():
            return ""
        named = {val for val in call_args.values() if isinstance(val, str)}
        if call_name == details:
            mine = set(run.trolley.units) | {key.rpartition("|")[0] for key in run.passed}
            if named & mine:
                return ""
        queued = self.late_q
        spare = run.moves_left() - queued
        allowed = run.late_n + queued < 3 and self.late_allowed(call_name, call_args, named, spare)
        if not allowed and call_name == details and spare - 1 >= 6 and self.backup_n < 2:
            if named & set(self.spare_backups()):
                allowed = True
                self.backup_n = self.backup_n + 1
        if not allowed and call_name == details:
            opened = run.shelf.swap_open()
            if opened and opened in named and spare > 2:
                allowed = True
                self.swap_done = True
        if allowed:
            self.late_q = queued + 1
            self.late_ids.update(named)
            return ""
        extra = []
        if call_name == details and spare - 1 >= 6 and self.backup_n < 2:
            extra = self.spare_backups()
        note = ("Not run (no step used): what is left stays reserved for the order. Details are read only for the cart's line, a "
                "listing a check named, or (while nothing is verified) up to 3 listings in all: put your best verified fit in the "
                "cart, re-check its stock and price, and order.")
        if extra:
            note += " Cheaper unread rows that may back up the pick can still be read: " + ", ".join(extra) + "."
        return note

    def backup_rows(self):
        run = self.run
        if run.needs.priority() != "cheapest" or self.backup_n >= 2:
            return []
        if not (run.event_seen or run.shelf.dead_lines() or self.named_open()):
            return []
        found = []
        try:
            for k in run.cover.cg_products():
                pool = run.shelf.ed_pool(k)
                fits = [e["price"] for e in pool if e["verified"] and e["fit"] and e["price"] is not None]
                if not fits:
                    continue
                ceiling = min(fits)
                for entry in pool:
                    blocked = (entry["read"] or entry["gone"] or entry["over"] or entry["pointer"]
                               or entry["ruled"] or entry["clash"])
                    if not blocked and entry["price"] is not None and entry["price"] < ceiling:
                        found.append((entry["price"], entry["id"]))
        except Exception as err:
            run.log.append({"kind": "backup_error", "error": f"{type(err).__name__}: {err}"[:300]})
            return []
        return [i for _, i in sorted(found)][:6]


class ShoppingRun:
    def __init__(self, pdata):
        environment = pdata["environment"]
        self.wire = Wire(environment["binding"])
        self.pview = environment["policy_view"]
        self.request = str(self.pview.get("query") or "")
        self.toolset = [entry for entry in self.pview.get("tools") or []
                        if isinstance(entry, dict) and isinstance(entry.get("function"), dict)]
        self.shapes = {}
        for entry in self.toolset:
            spec = entry["function"]
            self.shapes[spec["name"]] = spec.get("parameters") or {"type": "object", "properties": {}}
        self.step_cap = int(self.pview.get("max_steps") or 30)
        self.wire.done = bool(self.pview.get("finished"))
        self.archive = Records()
        self.needs = Needs(self.wire, self.request)
        self.role_map = {}
        self.trolley = CartBook(None, {})
        self.open_ledgers()
        LITERAL_VALUES.clear()
        self.reads = ReadQueue()
        self.intake = Intake(self)
        self.judge = Judge(self)
        self.shelf = Catalogue(self)
        self.talk = Dialogue(self)
        self.closer = CodeCloser(self)
        self.watch = Observer(self)
        self.walk = CheapWalk(self)
        self.refdesk = ReferenceDesk(self)
        self.cover = Coverage(self)
        self.needs.ref_hook = self.refdesk.ref_fix
        self.needs.field_hook = self.judge.store_fields

    def open_ledgers(self):
        # counters and flags
        self.msgs_sent = 0
        self.msg_cap = None
        self.procedures: list[dict[str, Any]] = []
        self.confirms = 0
        self.confirmed_cart: Any = None
        self.confirming = False
        self.go_verdict = ""
        self.go_hold_mark: Any = None
        self.pairs: list[tuple[int, int]] = []
        self.pair_told: dict[Any, int] = {}
        self.orders_tried = 0
        self.refusal_n = 0
        self.stuck = 0
        self.absorb_n = 0
        self.cart_seq = 0
        self.late_n = 0
        self.since_cart = None
        self.asking = True
        self.order_vetted = False
        self.decline_vetted = False
        self.audit_done = False
        self.event_seen = None
        self.covering = None
        self.last_kind = None
        self.best_held = None
        self.early_audit = None
        self.recheck_mark = None
        self.ref_decline_held = None
        self.slack_done = None
        self.decline_fill = None
        self.decline_kept = None
        self.decline_opens = None
        self.decline_searched = None
        self.batch_cart = None
        self.batch_proven = None
        self.batch_spec = None
        self.batch_spec_known = False
        # what was checked, refused and told
        self.rank_checked = set()
        self.checked_adds = set()
        self.passed = set()
        self.refused_lines = set()
        self.warned = set()
        self.view_told = set()
        self.rulings = {}
        self.refusals_seen = {}
        self.pass_price = {}
        self.page_book = {}
        self.cheaper_told = {}
        self.dead_told = {}
        self.pointer_told = {}
        self.line_read = {}
        self.line_seq = {}
        self.puts = {}
        self.questions_log = []
        self.last_calls = []
        self.batch_puts = []
        self.ref_hits = []
        self.chat = []
        self.refs = []
        # how the store's keyword search reads a query
        self.word_seen = []
        self.word_kind = ""
        self.lead_seen = []
        self.lead_bounds = {}
        self.lead_words = 0
        self.lead_ctx = None

    @property
    def log(self):
        return self.wire.log

    def moves_left(self):
        return self.step_cap - self.wire.tick

    def role(self, call_name):
        return self.role_map.get(call_name)

    def ask_limit(self):
        return max(0, (ASK_DEFAULT if self.msg_cap is None else self.msg_cap) - self.held_back())

    def held_back(self):
        # one message per stated step stays out of asking until the step is done
        if self.confirms or not self.procedures:
            return 0
        return min(len(self.procedures), ASK_DEFAULT if self.msg_cap is None else self.msg_cap)

    def cart_mark(self):
        return tuple(sorted((line, units) for line, units in self.trolley.units.items() if units))

    def basket_note(self):
        table = self.closer.code_rows()
        lines, total = [], 0.0
        for line, units in sorted(self.trolley.units.items()):
            if not units:
                continue
            row = table.get(line) or {}
            price = row.get("price")
            total += (price or 0.0) * units
            title = str(row.get("title") or line)[:90]
            lines.append(f"{units:g} x {title} ({line})" + (f" at {price:g} each" if is_number(price) else ""))
        limit = self.needs.spend_cap
        tail = f" Total {total:g}" + (f", within your limit of {limit:g}" if is_number(limit) and total <= limit else "") + "." if total else ""
        return "Final basket: " + "; ".join(lines) + "." + tail + " May I place this order? Please answer yes or no."

    def go_ahead(self, note, said):
        prompt = GO_AHEAD + "\n\nTHE ASSISTANT WROTE:\n" + note + "\n\nTHE SHOPPER REPLIED:\n" + said
        answer = self.wire.query_model([{"role": "user", "content": prompt}], LISTENER, 200)
        verdict = str((json_object((answer or {}).get("content") or "") or {}).get("answer") or "").lower()
        return verdict if verdict in ("yes", "no", "change") else "yes"

    def send_go_ahead(self, room=3):
        if not self.procedures or not self.role("message") or not self.cart_mark():
            return None
        mark = self.cart_mark()
        if mark == self.confirmed_cart or (self.msg_cap is not None and self.msgs_sent >= self.msg_cap):
            return None
        if self.moves_left() < room or self.wire.spare() < LAST_CALL:
            return None
        note = self.basket_note()
        proposal = self.talk.message_call(note, f"goahead_{self.wire.tick}")
        if not proposal:
            return None
        before = (self.needs.text_body(), self.needs.spend_cap)
        self.confirming = True
        try:
            replies = self.dispatch([proposal], scripted=True)
        finally:
            self.confirming = False
        self.confirms += 1
        self.confirmed_cart = mark
        said = replies[0][1].split("\n[status]")[0] if replies else ""
        verdict = self.go_ahead(note, said)
        if (self.needs.text_body(), self.needs.spend_cap) != before:
            verdict = "change"
        self.go_verdict = verdict
        self.log.append({"kind": "go_ahead", "note": note, "reply": said[:400], "verdict": verdict})
        return note, said, verdict

    def pair_cards(self):
        merged, _ = self.archive.merge_cards()
        return {ident: as_text(card) for ident, card in merged.items()}

    def pair_slot(self, line):
        label = str(self.archive.label_of.get(line) or "").strip().lower()
        homes = self.shelf.home_shelves()
        slots = [spot for spot, home in enumerate(homes) if label and label in home]
        return slots[0] if len(slots) == 1 else None

    def pair_marks(self, line, texts):
        # the anchor's own identifiers (its model and the names it is also sold as): marks on its page that are not common
        # to its shelf (a mark on half the shelf's pages or more, once four are known, names no single item)
        label = self.archive.label_of.get(line)
        shelf = [body for ident, body in texts.items() if self.archive.label_of.get(ident) == label]
        return {mark for mark in id_marks(texts.get(line, ""))
                if len(shelf) < 4 or 2 * sum(1 for body in shelf if mark in body) < len(shelf)}

    def pair_gate(self):
        if not self.pairs or self.moves_left() <= 3 or self.wire.spare() <= CLOSING:
            return ""
        lines = [line for line, units in self.trolley.units.items() if units]
        by_slot: dict[int, list[str]] = {}
        for line in lines:
            spot = self.pair_slot(line)
            if spot is not None:
                by_slot.setdefault(spot, []).append(line)
        texts = self.pair_cards()
        products = [str(row["product"]) for row in self.needs.items()]
        dead = self.shelf.dead_lines()
        for dep, anchor in self.pairs:
            for base in by_slot.get(anchor, []):
                marks = self.pair_marks(base, texts)
                if not marks:
                    continue
                for line in by_slot.get(dep, []):
                    if any(mark in texts.get(line, "") for mark in marks):
                        continue
                    told = (line, base, self.cart_seq)
                    if self.pair_told.get(told, 0) >= 1:
                        continue
                    self.pair_told[told] = 1
                    label = self.archive.label_of.get(line)
                    named = sorted((ident for ident, body in texts.items() if ident not in dead and ident != line
                                    and self.archive.label_of.get(ident) == label and any(mark in body for mark in marks)),
                                   key=lambda ident: self.archive.price(ident) if is_number(self.archive.price(ident)) else 1e18)
                    dep_name = products[dep] if dep < len(products) else "item"
                    anchor_name = products[anchor] if anchor < len(products) else "item"
                    shown = ", ".join(sorted(marks)[:2])
                    self.log.append({"kind": "pair_hold", "line": line, "anchor": base, "marks": sorted(marks), "named": named[:6]})
                    return (f"Not run yet (no step used). The {dep_name} has to work with the {anchor_name} picked: {base}'s own page "
                            f"carries {shown}, and the page of {line} does not name it. "
                            + (f"Read listings of the {dep_name} that name it: {', '.join(named[:6])}. Swap {line} for the cheapest of "
                               "them that meets every other requirement, then order." if named else
                               f"Search the store for '{sorted(marks)[0]}' (search matches listing text), read the {dep_name} rows it "
                               f"shows, and swap {line} for one whose page names it."))
        return ""

    def basket_short(self):
        counts = self.needs.counts()
        if len(counts) < 2:
            return []
        return self.trolley.basket_faults(counts) if self.trolley.units else ["the cart is empty"]

    def basket_gate(self):
        # every product needs its own line: a basket missing one is never right, so this holds on every order attempt
        faults = self.basket_short()
        if not faults:
            return ""
        products = [str(row["product"]) for row in self.needs.items()]
        slots = {self.pair_slot(line) for line, units in self.trolley.units.items() if units}
        missing = [name for spot, name in enumerate(products) if spot not in slots]
        named = (" No line yet for: " + ", ".join(missing) + ".") if missing and len(missing) < len(products) else ""
        self.log.append({"kind": "basket_hold", "faults": faults, "missing": missing})
        if self.moves_left() <= 3:
            return ("Not run (no step used). An order missing a product's line fails every time: " + "; ".join(faults) + "." + named
                    + " With the steps left, put the missing line in now if a fitting listing is known; otherwise decline.")
        return ("Not run yet (no step used). The cart is not the full basket: " + "; ".join(faults) + "." + named
                + " Find and add a fitting listing for each product missing (one line each, at its quantity), then order; decline "
                "only if some product has no listing that can meet its requirements.")

    def pair_text(self):
        if not self.pairs:
            return ""
        products = [str(row["product"]) for row in self.needs.items()]
        pairs = "; ".join(f"the {products[dep]} has to work with the {products[anchor]}" for dep, anchor in self.pairs
                          if dep < len(products) and anchor < len(products))
        return ("\n[fit] " + pairs + ". Pick the " + "/".join(dict.fromkeys(products[a] for _, a in self.pairs if a < len(products)))
                + " first: its page carries its own model name and number, and a fitting item's page lists that model among those it "
                "works with (a search for the model finds them).") if pairs else ""

    def go_ahead_text(self):
        # the actor never sends the go-ahead itself: it is told so in every status line while a stated step is due
        if not self.procedures or not self.role("message"):
            return ""
        if self.go_verdict == "yes" and self.confirmed_cart == self.cart_mark():
            return ("\n[go-ahead] The shopper has seen this basket and let it stand: that is the go-ahead the request asks for. "
                    "Call the order tool now; do not change the cart unless a line stopped fitting.")
        return ("\n[go-ahead] The shopper must approve the finished cart before it is bought. The code handles that step: "
                "once the cart holds the final basket, call the order tool, and the code first sends the shopper the basket in the "
                "message kept for it, reads the reply and then places the order. Do not ask for the yes yourself, and never decline "
                "or empty the cart for want of it.")

    def go_ahead_decline_gate(self):
        mark = (self.cart_mark(), self.needs.version)
        if not self.procedures or self.go_verdict == "no" or not mark[0] or self.go_hold_mark == mark or self.moves_left() <= 1:
            return ""
        self.go_hold_mark = mark
        return ("Not run yet (no step used). The shopper's approval is not missing: the code shows the cart to the shopper "
                "when you call the order tool, using the message kept for it, and orders only if they let it stand. The cart holds "
                + ", ".join(f"{line} x{units:g}" for line, units in mark[0]) + ". Call the order tool with it; decline only if a line "
                "plainly fails a requirement now in force and nothing in the records can replace it.")

    def go_ahead_gate(self):
        sent = self.send_go_ahead()
        if not sent or sent[2] == "yes":
            return ""
        note, said, _ = sent
        return ("Not run yet (one step used to message the shopper). The request wants the shopper's go-ahead on the final basket, so it was "
                f"put to them: \"{note}\" They replied: \"{said}\" Act on what they said; a changed basket is put to them again before the order.")

    def call_kind(self, call_name, call_args):
        order, decline = self.role("order"), self.role("decline")
        if not call_name or call_name not in (order, decline):
            return None
        if order != decline:
            return "order" if call_name == order else "decline"
        for piece in (call_args or {}).values():
            if not isinstance(piece, str) or len(piece) > 40:
                continue
            low = piece.lower()
            if re.search(DECLINE_WORDS, low):
                return "decline"
            if re.search(ORDER_WORDS, low):
                return "order"
        return "order"

    def finishers(self):
        return {self.role("order"), self.role("decline")} - {None}

    def clock_text(self):
        left = max(0.0, self.wire.spare() - LAST_CALL)
        if left > (DEADLINE - LAST_CALL) / 2:
            return ""
        rounded = int(left // CLOCK_STEP) * CLOCK_STEP
        if rounded <= 0:
            return f"under {CLOCK_STEP} s left"
        return f"about {rounded} s left"

    def status_head(self, spare):
        sent = f"messages sent {self.msgs_sent}"
        if self.msg_cap is not None:
            sent += f" of {self.msg_cap}"
        pieces = [f"[status] steps used {self.wire.tick} of {self.step_cap}, {spare} left", sent]
        clock = self.clock_text()
        if clock:
            pieces.append(clock)
        return "; ".join(pieces) + ".\n[requirements now] " + self.needs.text_body() + self.refdesk.ref_line() + self.cover.product_steps()

    def status_tail(self, spare):
        several = bool(self.needs.items())
        if spare <= 2:
            target = "every product asked for" if several else "the item"
            return (" LAST STEPS: if the cart already holds a candidate for " + target + ", order it"
                    "; else put the best candidate in (a verified fit, or failing one, the cheapest "
                    "in-stock listing within budget with no failing requirement) and order. Decline only when the records "
                    "show that nothing can fit.")
        if spare <= 5 or self.wire.spare() < CLOSING:
            lines = "one line per product asked for" if several else "one line"
            return (" FINISH NOW: set the cart to " + lines
                    + " (a verified fit, or failing one, the cheapest in-stock listing within budget with no failing "
                    "requirement; a sold-out line out), re-check its stock, then order. Decline only when the records show that "
                    "nothing can fit.")
        if self.wire.spent_out():
            return (" MODEL BUDGET USED: no more exploring; in the next few steps, put the best candidate of "
                    + ("every product asked for (one line each, at its quantity)" if several else "the item asked for")
                    + " in the cart (a verified fit if there is one, else the nearest fit in the records), check its "
                    "stock and price again, and order. Declining is right only when the records show that "
                    + ("some product has" if several else "the store has") + " no listing that can meet its requirements.")
        if self.orders_tried != 0 or not self.narrowing_now():
            return ""
        due = self.intake.order_due()
        if due < spare <= due + 7:
            return (f" Few steps remain: in the next couple of steps, place your best verified fit in the cart and order; searches and detail reads now end when {due} steps remain, and a refused order needs about 4 steps to repair.")
        if spare <= due:
            return " Searches are closed (details only for the cart's line, listings a check named, or, while some product has no verified fit, up to 3 more reads or searches in all): put your best verified fit in the cart, re-check its stock and price, and order; a refused order still needs about 4 steps to repair."
        return ""

    def narrowing_now(self):
        return self.intake.narrowing()

    def status_line(self):
        spare = self.moves_left()
        text = self.status_head(spare)
        usable = [entry["function"]["name"] for entry in self.intake.offered()]
        if len(usable) < len(self.toolset):
            text += "\n[tools usable now] " + ", ".join(usable) + " (any other tool call is refused without using a step" + (
                "; questions to the shopper are planned and sent for you" if self.asking else "") + ")."
        if self.trolley.used:
            text += "\n" + self.trolley.summary()
        return text + self.pair_text() + self.go_ahead_text() + self.status_tail(spare)

    # one batch of proposed calls
    def screen(self, proposal: dict[str, Any], permitted: set[str], scripted: bool, batch: int, queued_messages: int) -> tuple[str | None, str, dict[str, Any]]:
        fn = proposal.get("function") or {}
        tool, raw = fn.get("name"), fn.get("arguments")
        if tool not in self.shapes:
            return f"Not run (no step used): '{tool}' isn't a tool here. Use one of these: {', '.join(sorted(permitted))}.", tool, {}
        if not scripted and tool not in permitted:
            if tool == self.role("message") and self.asking:
                self.talk.want_ask(raw)
                return ("Not run (no step used): the code plans and sends questions to the shopper for you once you have read "
                        "some listings' details."), tool, {}
            return ("Not run (no step used): too few steps are left for it now; the tools still usable now are: "
                    + ", ".join(sorted(permitted)) + "."), tool, {}
        if isinstance(raw, dict):
            call_args = raw
        else:
            call_args = json_object(raw) if (raw or "").strip() else {}
        if call_args is None:
            return "Not run (no step used): the arguments did not form one JSON object.", tool, {}
        call_args, faults = ArgCheck.execute(call_args, self.shapes[tool])
        if faults:
            return "Not run (no step used): arguments invalid: " + "; ".join(faults[:6]), tool, call_args
        refusal = self.batch_rules(tool, call_args, scripted, batch, queued_messages)
        if refusal:
            return refusal, tool, call_args
        signature = tool + as_text(call_args)
        if tool not in self.finishers() and tool not in (self.role("message"), self.role("cart")) and signature in self.last_calls:
            return ("Not run (no step used): this exact call was just made and nothing changed since then; its result is "
                    "above. Use other arguments or do something else."), tool, call_args
        held = "" if scripted else self.intake.late_hold(tool, call_args)
        if not held and not scripted:
            held = self.cover.value_guard(tool, call_args)
        if held:
            return held, tool, call_args
        if tool == self.role("cart"):
            refusal = self.screen_cart(call_args, signature)
            if refusal:
                return refusal, tool, call_args
        return None, tool, call_args

    def batch_rules(self, tool: str, call_args: dict[str, Any], scripted: bool, batch: int, queued_messages: int) -> str:
        if tool in self.finishers() and batch > 1:
            return ("Not run (no step used): send an order or a decline on its own, only after you have seen what the other "
                    "calls returned.")
        if tool == self.role("message") and self.msg_cap is not None and self.msgs_sent + queued_messages >= self.msg_cap - (0 if self.confirming else self.held_back()):
            return "Not run (no step used): no message is left in the budget. Do not text the shopper again."
        if scripted or tool != self.role("cart_view") or tool == self.role("cart") or self.batch_cart:
            return ""
        if not self.trolley.units:
            return ""
        shelf = self.shelf
        version = -2 if shelf.cart_ver is None else shelf.cart_ver
        since = 1 if self.since_cart is None else self.since_cart
        if shelf.cart_shown != version or since:
            return ""
        if shelf.cart_shown in self.view_told:
            return ""
        self.view_told.add(shelf.cart_shown)
        return ("Not run (no step used): the newest cart result already lists the basket - " + self.trolley.summary()
                + " A cart view adds nothing at all; to re-check a line's stock, open its record with "
                + (shelf.stock_tool() or "the tool that shows stock") + ".")

    def screen_cart(self, call_args: dict[str, Any], sig: str) -> str:
        trolley, needs = self.trolley, self.needs
        thing = trolley.aim(call_args)
        putting = trolley.is_put(call_args)
        point = self.refdesk.ref_pointer_note(thing) if thing and putting else ""
        if point and self.pointer_told.get(thing, 0) < 2:
            self.pointer_told[thing] = self.pointer_told.get(thing, 0) + 1
            return f"Not run (no step used): {point}. Select another listing."
        held_note = self.passed_line_hold(call_args, thing, sig)
        if held_note:
            return held_note
        if not putting or not thing:
            return ""
        if trolley.held(thing) > 0 and trolley.adds is not False and sig not in self.warned:
            self.warned.add(sig)
            return (f"Not run (no step used): {thing} x{int(trolley.held(thing))} is already in the cart, and this cart tool adds "
                    "units to held lines, so this call would raise the count. To order it, leave the cart alone; to change the "
                    "units, remove or add just the difference. If adding is meant, send the same call again.")
        walk = "" if point else self.cover.walk_hold(call_args)
        if walk:
            return walk
        verdict = self.check_new_lines(call_args, thing)
        if verdict:
            return verdict
        verdict = self.rank_lines(call_args)
        if verdict:
            return verdict
        return self.unit_rules(call_args, thing, sig)

    def passed_line_hold(self, call_args: dict[str, Any], thing: str | None, sig: str) -> str:
        trolley, needs = self.trolley, self.needs
        mark = f"{sig}#{self.puts.get(thing, 0)}"
        own_price = self.archive.price(thing) if thing else None
        cheaper_put = False
        for other in self.batch_puts:
            if other != thing and is_number(own_price) and is_number(self.archive.price(other)) and self.archive.price(other) < own_price:
                cheaper_put = True
                break
        swap = needs.priority() == "cheapest" and (cheaper_put or bool(self.judge.cheaper_fit(thing)))
        moved = thing in self.pass_price and self.pass_price[thing] != own_price
        if not thing or trolley.is_put(call_args) or f"{thing}|{needs.version}" not in self.passed or swap or moved:
            return ""
        count = trolley.count_arg(call_args)
        if count is not None and 0 < count < trolley.held(thing):
            return ""
        if self.refusal_n != self.refusals_seen.get(thing, self.refusal_n) or mark in self.warned:
            return ""
        self.warned.add(mark)
        least = ""
        if needs.priority() == "cheapest" and not self.judge.cheaper_fit(thing):
            least = " It is the cheapest verified fit, too."
        return (f"Not run (no step used): an independent check judged that {thing} satisfies every requirement in force, taking "
                "each property by the value rules (where the store puts the value itself in a field, only those words count); "
                "only the category label settles what kind of item it is, whatever the object its title or description "
                f"names.{least} Nothing changed since then. If a change or refusal excludes it, if one of its fields plainly "
                "fails a requirement, or if you swap it for a cheaper listing, repeat the same call.")

    def check_new_lines(self, call_args: dict[str, Any], thing: str) -> str:
        version = self.needs.version
        others = [other for other, count in self.trolley.lines(call_args)[1:] if count is None or count > 0]
        for other in others + [thing]:
            mark = f"{other}|{version}"
            if self.moves_left() <= 3 or mark in self.checked_adds:
                continue
            self.checked_adds.add(mark)
            verdict = self.judge.check_item(other, "ADD TO THE CART")
            if verdict:
                return "Not run (no step used). " + verdict + " If you're certain that it fits, send this call again."
        return ""

    def rank_lines(self, call_args: dict[str, Any]) -> str:
        lines = self.trolley.lines(call_args)
        if self.moves_left() > 3:
            for line_id, count in lines:
                if count is not None and count <= 0:
                    continue
                better = self.judge.rank_check(line_id)
                if not better:
                    continue
                cheaper = self.judge.rank_verify(line_id)
                text = better.replace("before ordering", "before putting it in the cart").replace(
                    "order the current pick", "send the same call again")
                if cheaper:
                    text += (f" An independent item check found that {cheaper} fits every requirement now in force and costs "
                             f"less than {line_id}: put {cheaper} in its place.")
                return "Not run (no step used). " + text
        if self.needs.priority() != "cheapest":
            return ""
        for line_id, count in lines:
            if count is not None and count <= 0:
                continue
            cheaper = self.judge.cheaper_fit(line_id)
            told = self.cheaper_told
            if cheaper and cheaper not in self.trolley.units and not told.get(("cart", line_id, cheaper)):
                told[("cart", line_id, cheaper)] = 1
                return (f"Not run (no step used). The independent item check found that {cheaper} meets every requirement now in "
                        f"force, and costs less than {line_id}. The cheapest fit is requested and the check's reading stands over "
                        f"your own: put {cheaper} in instead. If a record shows {cheaper} failing a requirement now in force, send "
                        "the same call again.")
        return ""

    def unit_rules(self, call_args: dict[str, Any], thing: str, sig: str) -> str:
        trolley, needs = self.trolley, self.needs
        wanted = needs.quantity()
        asked = trolley.count_arg(call_args)
        aim = wanted - trolley.held(thing) if trolley.adds is not False else wanted
        if needs.items():
            odd = [(i, int(c)) for i, c in trolley.lines(call_args) if c is not None and c > 0 and int(c) not in needs.counts()]
            if not odd or sig in self.warned:
                return ""
            self.warned.add(sig)
            return (f"Not run (no step used): the products asked for need {needs.counts()} units (one line each), yet this call "
                    f"asks for {odd}. If you are sure, send the same call again.")
        if asked is None or abs(asked - aim) <= 1e-9 or sig in self.warned:
            return ""
        self.warned.add(sig)
        return (f"Not run (no step used): the request wants a quantity of {wanted} of one listing, but this call asks for "
                f"{int(asked)}. If you are sure, send the same call again.")

    def screen_finish(self, cid: str, call_name: str, call_args: dict[str, Any]) -> tuple[str | None, dict[str, Any]]:
        kind = self.call_kind(call_name, call_args)
        if kind == "order":
            for gate in (self.sold_line_gate, self.basket_gate, self.pair_gate, self.cheaper_pick_gate, self.walk_gate, self.stale_gate, self.vet_gate, self.go_ahead_gate):
                held = gate()
                if held:
                    return held, call_args
            return None, call_args
        if kind != "decline":
            return None, call_args
        gates = (self.go_ahead_decline_gate, self.shelf.near_hold, self.kept_line_gate, self.slack_gate, self.fill_gate, self.named_gate,
                 self.pages_gate, self.audit_again_gate, self.first_audit_gate, self.fit_gate, self.best_gate,
                 self.soft_ref_gate, self.shelf.decline_repair, self.shelf.near_check, self.no_decline_gate)
        for gate in gates:
            held = gate()
            if held:
                return held, call_args
        return None, self.closer.scrub_decline(call_args, call_name)

    def sold_line_gate(self) -> str:
        if self.moves_left() <= 2:
            return ""
        shelf = self.shelf
        dead = sorted(set(self.trolley.units) & shelf.dead_lines())
        mark = (tuple(dead), self.cart_seq)
        told = self.dead_told
        if not dead:
            return ""
        if told.get(mark, 0) < 2:
            told[mark] = told.get(mark, 0) + 1
            return "Not run yet (no step used). " + shelf.dead_note(dead)
        if told.get(mark, 0) != 2 or not self.role("cart") or self.moves_left() < 6 or self.wire.spare() <= CLOSING:
            return ""
        told[mark] = 3
        muted_before = self.wire.mute
        try:
            swapped = self.closer.code_cart(set(dead) | shelf.dead_lines(), set(self.trolley.units) - set(dead))
        except Exception as err:
            swapped = False
            self.log.append({"kind": "dead_swap_error", "error": f"{type(err).__name__}: {err}"[:300]})
        finally:
            self.wire.mute = muted_before
        self.log.append({"kind": "dead_swap", "lines": dead, "done": swapped, "cart": dict(self.trolley.units)})
        if not swapped or not self.trolley.units or set(self.trolley.units) & shelf.dead_lines():
            return ""
        return ("Not run yet (no step used). The cart still held " + ", ".join(dead) + ", which is sold out, so the code "
                "removed it and put a candidate in its place, kept the other lines, and read the lines' stock: "
                + self.trolley.summary() + " Compare this cart against the request, then order it.")

    def cheaper_pick_gate(self) -> str:
        if self.moves_left() <= 3:
            return ""
        judge = self.judge
        thing = judge.chosen()
        cheaper = judge.cheaper_fit(thing) if thing else ""
        if not cheaper:
            for line in [i for i in self.trolley.units if i != thing]:
                cheaper = judge.cheaper_fit(line)
                if cheaper:
                    thing = line
                    break
        told = self.cheaper_told
        if not cheaper or told.get((thing, cheaper), 0) >= 2:
            return ""
        told[(thing, cheaper)] = told.get((thing, cheaper), 0) + 1
        return (f"Not run yet (no step used). The independent item check found that {cheaper} meets every requirement and it "
                f"costs less than {thing}. The cheapest fit is requested and the check's reading outranks your own: take "
                f"{thing} out of your cart, put {cheaper} in, re-read its stock and price, and order it.")

    def walk_gate(self) -> str:
        if not self.trolley.units:
            return ""
        return self.cover.walk_hold(None)

    def stale_gate(self) -> str:
        reader = self.shelf.stock_tool()
        if not reader or self.moves_left() <= len(self.trolley.units) + 2 or self.wire.spare() <= CLOSING:
            return ""
        stale = self.shelf.stale_lines()
        mark = (tuple(sorted(self.trolley.units)), self.cart_seq)
        if not stale or self.recheck_mark == mark:
            return ""
        self.recheck_mark = mark
        reads = self.reads
        reads.open_queue = stale[:4]
        reads.open_recheck = True
        reads.open_purpose = "cart"
        reads.open_tool = reader
        return ("Not run yet (no step used). Stock and prices may change without notice, and these cart lines went unread since "
                "the last cart change: " + ", ".join(stale[:4]) + f". The code reads their stock for you now ({reader}, the tool "
                "that shows stock); then order if each one is still in stock and the total fits the budget, else replace the line.")

    def vet_gate(self) -> str:
        if self.order_vetted or self.moves_left() <= 2 or self.wire.spare() <= CLOSING:
            return ""
        self.order_vetted = True
        judge, needs = self.judge, self.needs
        faults = self.trolley.faults(needs.quantity(), needs.counts())
        if faults and self.moves_left() > 3:
            return "Not run yet (no step used). As counted, the cart isn't ready: " + "; ".join(faults) + ". Fix it, then order."
        if needs.items():
            lines = list(self.trolley.units)
        else:
            lines = [judge.chosen()] if judge.chosen() else []
        for thing in lines:
            verdict = judge.check_item(thing, "ORDER")
            if verdict:
                return "Not run yet (no step used). " + verdict
        for thing in lines:
            better = judge.rank_check(thing)
            if not better:
                continue
            cheaper = judge.rank_verify(thing)
            extra = ""
            if cheaper:
                extra = (f" An independent item check found that {cheaper} fits every requirement now in force and costs less "
                         f"than {thing}: take {thing} out, put {cheaper} in, read its stock and price anew, and order.")
            return "Not run yet (no step used). " + better + extra
        scope = ("per product asked for, each at its quantity" if needs.items()
                 else "the chosen item at the required quantity")
        return ("Not run yet (no step used). Confirm before you order: (1) the cart contains exactly one line " + scope + ", and "
                "no other lines; (2) after the latest cart change you reread this item's stock and price with a tool that reports "
                "them, it is available, and price times quantity stays inside the current budget; (3) the item's own properties "
                "satisfy every stated requirement, every need the shopper disclosed and every announced change; (4) when the "
                "cheapest or best was requested, nothing you found fits better; (5) all of the request's procedural rules were "
                "obeyed" + (" (the code asks the shopper to approve the cart when you call the order tool)" if self.procedures else "")
                + ". Check anything still unchecked now; if everything holds, call the order tool again.")

    def kept_line_gate(self) -> str:
        dead = self.shelf.dead_lines
        version = self.needs.version
        verified = [i for i in self.trolley.units
                    if f"{i}|{version}" in self.passed and i not in self.refused_lines and i not in dead()]
        if not verified or self.moves_left() <= 1 or self.decline_kept:
            return ""
        self.decline_kept = True
        every = " with each product's line in your cart" if self.needs.items() else ""
        return ("Not run yet (no step used). The cart holds " + ", ".join(verified) + ", which the independent check found to "
                "meet every requirement now in force. A decline would waste it: re-read its stock and price, and order it" + every
                + ". Decline only if a change since then clearly rules it out now.")

    @staticmethod
    def read_calls(rows: list[str], batch: Any) -> int:
        if batch and len(rows) >= 2:
            return -(-len(rows) // batch[2])
        return len(rows)

    def slack_gate(self) -> str:
        shelf = self.shelf
        slack = []
        if not self.slack_done and (self.role("details") or shelf.batch_tool()):
            slack = shelf.slack_rows()
        spread = shelf.batch_tool()
        if not slack or self.moves_left() <= self.read_calls(slack, spread) + 4 or self.wire.spare() <= CLOSING + 40:
            return ""
        self.slack_done = True
        self.reads.open_queue = slack
        self.reads.open_say = "Before declining, I open cheaper rows which may let all products fit the budget together."
        total = shelf.slack_total
        shown = f"{total:g}" if is_number(total) else "more than the budget"
        return ("Not run yet (no step used). Each product has a fitting listing, yet the cheapest fits together cost " + shown
                + f", above the budget of {self.needs.spend_cap}. Rows priced within what the budget leaves each product after the "
                "other products' cheapest fits were never opened, so the code reads their records now: " + ", ".join(slack)
                + ". Test each one against the requirements of its own product by the value rules (where the store puts the value "
                "itself in a field, only those words count); whenever fits within the budget cover all products together, put "
                "each product's line in the cart and order; decline only if no such set exists.")

    def fill_gate(self) -> str:
        shelf = self.shelf
        batch = shelf.batch_tool()
        fill = shelf.lacking_rows() if not self.decline_fill and (self.role("details") or batch) else []
        if not fill or self.moves_left() <= self.read_calls(fill, batch) + 3 or self.wire.spare() <= CLOSING + 40:
            return ""
        self.decline_fill = True
        self.reads.open_queue = fill
        self.reads.open_say = "Before declining, I open unread listings of the products that have no verified fit yet."
        return ("Not run yet (no step used). Some products asked for still lack any verified fit, and listings of their "
                "categories within the budget were never opened, so the code reads their records now, before any decline: "
                + ", ".join(fill) + ". Test each one against the requirements of its own product by the value rules (where the "
                "store puts the value itself in a field, only those words count); once every product has a fitting listing, "
                "put each product's line in the cart and order; decline only if a product truly has none.")

    def named_gate(self) -> str:
        batch = self.shelf.batch_tool()
        queue = self.judge.named_unread()[: batch[2] if batch else 3]
        if not queue or self.decline_opens or not (self.role("details") or batch):
            return ""
        if self.moves_left() <= self.read_calls(queue, batch) + 3 or self.wire.spare() <= CLOSING + 40:
            return ""
        self.decline_opens = True
        self.reads.open_queue = queue
        return ("Not run yet (no step used). Nobody has opened the listings the checks named, so the code fetches their records "
                "now, before any decline: " + ", ".join(queue) + ". Weigh each one against the requirements by the value rules "
                "(the field named after the property decides, and where the store puts the value itself there, only those words "
                "count; with no such field, another field, the options list or the title may carry the value), buy one that "
                "fits, and decline only when every one fails.")

    def pages_gate(self) -> str:
        note = self.walk.decline_reads()
        return "Not run yet (no step used). " + note if note else ""

    def fit_gate(self) -> str:
        note = self.walk.fit_hold()
        return "Not run yet (no step used). " + note if note else ""

    def audit_again_gate(self) -> str:
        judge = self.judge
        limit = 10 ** 9 if judge.audit_reads is None else judge.audit_reads
        fresh_reads = len(self.archive.read) > limit and judge.audits < 2
        if not fresh_reads or not self.decline_vetted or self.moves_left() <= 3 or self.wire.spare() <= CLOSING + 30:
            return ""
        note = judge.review_decline()
        return "Not run yet (no step used). " + note if note else ""

    def first_audit_gate(self) -> str:
        if self.decline_vetted or self.moves_left() <= 3 or self.wire.spare() <= CLOSING + 30:
            return ""
        self.decline_vetted = True
        note = self.judge.review_decline()
        if note:
            return "Not run yet (no step used). " + note
        return ("Not run yet (no step used). Be certain before declining: your searches varied (other words, filters, pages), "
                "you inspected each candidate's own properties, and each candidate fails a stated requirement, a revealed need, "
                "the budget or stock. A property's field counts under any key spelling. When something fits, buy it instead; "
                "when nothing fits, call the decline tool again.")

    def best_gate(self) -> str:
        walk, needs = self.walk, self.needs
        best = walk.unrefuted() if not self.best_held and self.moves_left() > 4 else []
        for _ in range(3):
            unchecked = [i for i in best if f"{i}|{needs.version}" not in self.passed]
            if not unchecked or self.wire.spare() < CLOSING + 30:
                break
            self.judge.check_item(unchecked[0], "ORDER")
            best = walk.unrefuted()
        if not best or self.moves_left() <= len(best) + 3 or self.wire.spare() <= CLOSING + 20:
            return ""
        self.best_held = True
        several = bool(needs.items())
        return ("Not run yet (no step used). A decline is proper only when no listing can meet the requirements, and the "
                "records hold " + ("a candidate for every product asked for" if several else "a candidate") + " showing "
                "stock, inside the budget and ruled out by no check: " + ", ".join(best) + ". Unless a record shows one of "
                "them failing a requirement now in force, put " + ("one line of each product, at its quantity," if several
                else "it") + " in the cart, re-read stock and price, and order; decline only if each candidate plainly fails.")

    def soft_ref_gate(self) -> str:
        soft = [r for r in self.refs or [] if r.get("relation") == "differ" and r.get("status") in ("open", "stuck")
                and self.refdesk.ref_live(r)]
        if not soft or self.moves_left() <= 3 or self.ref_decline_held:
            return ""
        self.ref_decline_held = True
        heads = ", ".join(dict.fromkeys(r["head"] for r in soft))
        per_product = ", one line per product" if self.needs.items() else ""
        return ("Not run yet (no step used). " + heads + " is set by another listing the code could not pin down, so its value "
                "is unknown: it can neither exclude a listing nor justify a decline. Put the listing that satisfies all other "
                "requirements (in stock, within the budget) in the cart" + per_product + ", read its stock and price again, and "
                "order; decline only if no listing satisfies the other requirements.")

    def no_decline_gate(self) -> str:
        shelf = self.shelf
        try:
            proof = shelf.decline_proof()
            if proof:
                self.log.append({"kind": "decline_allowed", "why": proof})
                return ""
            note = shelf.closest_basket()
        except Exception as err:
            self.log.append({"kind": "no_decline_error", "error": f"{type(err).__name__}: {err}"[:300]})
            return ""
        if note or self.decline_searched or self.moves_left() <= 4 or self.wire.spare() <= CLOSING:
            return note
        self.decline_searched = True
        return ("Not run yet (no step used). No decline stands yet: the code has no listing of the requested kind to put in the cart, "
                "and the searches so far prove nothing. Search the requested kind with other short wordings and read the pages they "
                "show, then put the closest in-stock listing within the budget in the cart and order it.")

    def note_batch_puts(self, proposals: list[dict[str, Any]]) -> None:
        cart = self.role("cart")
        self.batch_puts = []
        self.batch_cart = any((p.get("function") or {}).get("name") == cart for p in proposals if isinstance(p, dict))
        for proposal in proposals:
            fn = proposal.get("function") or {}
            if fn.get("name") != cart:
                continue
            raw = fn.get("arguments")
            if isinstance(raw, dict):
                put_args = raw
            else:
                put_args = json_object(raw) if isinstance(raw, str) and raw.strip() else {}
            if isinstance(put_args, dict) and self.trolley.is_put(put_args) and self.trolley.aim(put_args):
                self.batch_puts.append(self.trolley.aim(put_args))

    def dispatch(self, proposals: list[dict[str, Any]], scripted: bool = False) -> list[tuple[str, str]]:
        answers: list[tuple[str, str]] = []
        self.intake.begin_batch()
        permitted = {tdef["function"]["name"] for tdef in self.intake.offered()}
        outgoing: list[tuple[str, str, dict[str, Any]]] = []
        messages_queued = 0
        cart_blocked = False
        self.note_batch_puts(proposals)
        for proposal in proposals:
            if cart_blocked and proposal["function"]["name"] == self.role("cart"):
                answers.append((proposal["id"], "Not run (no step used): a cart change earlier in this reply was not sent; "
                                                "read why, then send the cart changes again."))
                continue
            refusal, tool, call_args = self.screen(proposal, permitted, scripted, len(proposals), messages_queued)
            if refusal:
                if not cart_blocked:
                    cart_blocked = (tool == self.role("cart") and isinstance(call_args, dict) and bool(call_args)
                                    and not self.trolley.is_put(call_args))
                answers.append((proposal["id"], refusal))
                continue
            if tool == self.role("message"):
                messages_queued += 1
            outgoing.append((proposal["id"], tool, call_args))
        if outgoing and outgoing[0][1] in self.finishers():
            cid, tool, call_args = outgoing[0]
            held, call_args = self.screen_finish(cid, tool, call_args)
            if held:
                answers.append((cid, held))
                return answers
            outgoing[0] = (cid, tool, call_args)
        if outgoing and outgoing[0][1] not in self.finishers() and self.finishers():
            room = max(0, self.moves_left() - 1)
            for cid, _, _ in outgoing[room:]:
                answers.append((cid, "Not run (no step used): just one step is left, and it must be the order (or else the decline)."))
            outgoing = outgoing[:room]
        for start in range(0, len(outgoing), 16):
            slab = outgoing[start: start + 16]
            if self.wire.done or self.wire.broke or self.moves_left() <= 0:
                answers.extend((cid, "Not run: the episode has ended or no step is left.") for cid, _, _ in slab)
                continue
            answers.extend(self.send_chunk(slab[: self.moves_left()]))
        position = {c["id"]: i for i, c in enumerate(proposals)}
        answers.sort(key=lambda answer: position.get(answer[0], 0))
        return answers

    def send_chunk(self, slab: list[tuple[str, str, dict[str, Any]]]) -> list[tuple[str, str]]:
        actions = [{"name": tool, "args": call_args} for _, tool, call_args in slab]
        step_before = self.wire.tick
        reply = self.wire.act(actions)
        self.log.append({"kind": "env", "step": step_before + 1, "actions": actions, "reply": reply})
        if reply is None:
            return [(cid, "The call failed (network/server). Try it again.") for cid, _, _ in slab]
        if isinstance(reply.get("observations"), list):
            observations = reply.get("observations")
        else:
            observations = [reply.get("observation")]
        texts = []
        for index, (cid, tool, call_args) in enumerate(slab):
            if index >= len(observations):
                texts.append((cid, "Not run: the episode was over before this call."))
                continue
            texts.append((cid, self.absorb(tool, call_args, observations[index], reply, index == len(observations) - 1)))
        return texts

    def book_call(self, tool: str, call_args: dict[str, Any], obs: Any) -> None:
        message_tool, cart_tool = self.role("message"), self.role("cart")
        if tool == message_tool:
            self.msgs_sent += 1
        self.last_kind = self.call_kind(tool, call_args)
        if self.last_kind == "order":
            self.orders_tried += 1
        if tool in (cart_tool, message_tool) or tool in self.finishers():
            self.last_calls = []
        else:
            self.last_calls = (self.last_calls + [tool + as_text(call_args)])[-2:]
        self.archive.add(obs)
        self.absorb_n += 1
        if tool != cart_tool and not self.covering:
            self.since_cart = (0 if self.since_cart is None else self.since_cart) + 1

    def note_cart_reads(self, tool: str, call_args: dict[str, Any], obs: Any) -> None:
        shelf = self.shelf
        held = self.trolley.units
        stock_rule = bool(shelf.stock_seen)
        asked = shelf.asked_ids(call_args)
        rows_trusted = bool(shelf.rows_live) and not shelf.rows_lag
        lookups_only = not rows_trusted and tool not in (self.role("cart"), self.role("cart_view"))
        fresh: set[str] = set()

        def spot(record: dict[str, Any]) -> None:
            named = {v for k, v in record.items() if isinstance(v, str) and v in held and id_field(k)}
            if not named:
                return
            if stock_rule:
                if self.watch.stock_state(record) is None or (lookups_only and not named & asked):
                    return
            elif not any(isinstance(v, dict) for v in record.values()):
                return
            fresh.update(v for k, v in record.items() if isinstance(v, str) and v in held)
        walk_dicts(obs, spot)
        for line_id in fresh:
            self.line_read[line_id] = self.wire.tick
            self.line_seq[line_id] = self.absorb_n

    def count_late_read(self, tool: str, call_args: dict[str, Any], obs: Any) -> None:
        details, search = self.role("details"), self.role("search")
        if tool == details and not (isinstance(obs, dict) and obs.get("error")):
            self.archive.read.update(v for v in call_args.values() if isinstance(v, str) and v in self.archive.ids)
        looked = bool(details) and tool not in (details, search) and tool in self.shelf.lookup_tools()
        named = {v for v in call_args.values() if isinstance(v, str)}
        if self.moves_left() >= self.intake.order_due():
            return
        if tool not in (details, search) and not looked:
            return
        if self.covering:
            return
        held = set(self.trolley.units)
        if tool == details and named & held:
            return
        if looked and named & (held | {k.rpartition("|")[0] for k in self.passed}):
            return
        self.late_n += 1

    def cart_remarks(self, call_args: dict[str, Any], obs: Any, remarks: list[str]) -> None:
        trolley, shelf = self.trolley, self.shelf
        self.order_vetted = False
        trolley.changed_at = self.wire.tick
        held_before = dict(trolley.units)
        remarks.append(trolley.booked(call_args, obs))
        try:
            shelf.cart_moved(held_before)
            shelf.mark_fresh(obs)
            shelf.after_put(call_args, obs)
        except Exception as err:
            self.log.append({"kind": "after_put_error", "error": f"{type(err).__name__}: {err}"[:300]})
        item = trolley.aim(call_args)
        putting = bool(item) and trolley.is_put(call_args)
        if putting:
            self.refusals_seen[item] = self.refusal_n
            self.puts[item] = self.puts.get(item, 0) + 1
        units = trolley.count_arg(call_args)
        if not putting or f"{item}|{self.needs.version}" not in self.passed:
            return
        if (units is None or units > 0) and trolley.held(item) > 0 and item not in shelf.dead_lines():
            remarks.append(f"An independent check judged {item} satisfies every requirement now in force, reading each property "
                           "by the value rules (where this store puts the value itself in a field, only those words count); keep "
                           "it unless a later change rules it out.")

    def absorb(self, call_name: str, call_args: dict[str, Any], obs: Any, resp: dict[str, Any], last: bool) -> str:
        shown, obs = obs, lensed(obs)
        cart_tool, message_tool = self.role("cart"), self.role("message")
        self.book_call(call_name, call_args, obs)
        try:
            sold_now = self.shelf.learn_caps(call_name, call_args, obs)
        except Exception as err:
            sold_now = []
            self.log.append({"kind": "caps_error", "error": f"{type(err).__name__}: {err}"[:300]})
        if self.trolley.units:
            self.note_cart_reads(call_name, call_args, obs)
        self.count_late_read(call_name, call_args, obs)
        remarks = []
        if sold_now:
            remarks.append("[stock] " + self.shelf.dead_note(sold_now))
        try:
            redirected = self.refdesk.notes_remark(call_name, call_args)
            if redirected:
                remarks.append("[notes] " + redirected)
        except Exception as err:
            self.log.append({"kind": "notes_remark_error", "error": f"{type(err).__name__}: {err}"[:300]})
        if call_name == cart_tool:
            self.cart_remarks(call_args, obs, remarks)
        if call_name == self.role("cart_view"):
            self.trolley.viewed(obs, self.wire.tick)
            remarks.append(self.trolley.summary())
        refused = self.watch.refused(call_name, obs, resp if last else {"finished": False})
        if refused and self.last_kind == "order":
            self.cart_seq = self.absorb_n
        remarks.append(self.watch.notes_for(call_name, obs, refused))
        if call_name == self.role("search"):
            remarks.append(self.cover.search_note(call_args, obs))
            self.judge.named_pages.discard(json.dumps(call_args, sort_keys=True, default=str))
            try:
                remarks.append(self.shelf.merge_joined(obs))
            except Exception as err:
                self.log.append({"kind": "merge_joined_error", "error": f"{type(err).__name__}: {err}"[:300]})
        if last and self.early_audit_due(call_name):
            self.early_audit = True
            ahead = self.judge.review_decline(early=True)
            if ahead:
                remarks.append("No verified fit found yet. " + ahead)
        question = None
        if call_name == message_tool:
            question = " ".join(str(v) for v in call_args.values() if isinstance(v, str))
        caught = self.cover.shopper_lines(call_name, obs, refused)
        if caught:
            remarks.append(self.needs.hear(caught, question))
            self.refdesk.ref_take()
        text = as_text(clipped(shown, SEARCH_TEXT if call_name == self.role("search") else RECORD_TEXT))
        if last and resp.get("finished"):
            text += " [the episode is finished]"
        joined = " ".join(r for r in remarks if r)
        return text + ("\n[note] " + joined if joined else "")

    def early_audit_due(self, tool: str) -> bool:
        if self.early_audit or tool in (self.role("cart"), self.role("message")) or tool in self.finishers():
            return False
        if not FIRST_ORDER_BY < self.moves_left() <= self.step_cap // 3 + 2 or self.trolley.units:
            return False
        version = str(self.needs.version)
        return not any(key.rpartition("|")[2] == version for key in self.passed)

    # the episode loop
    def tool_calls(self, msg: dict[str, Any], seq_no: int) -> tuple[list[dict[str, Any]], int]:
        proposals = []
        for call in msg.get("tool_calls") or []:
            if not isinstance(call, dict):
                continue
            seq_no += 1
            fn = call.get("function") or {}
            raw = fn.get("arguments")
            arguments = raw if isinstance(raw, str) else json.dumps(raw or {})
            proposals.append({"id": call.get("id") or f"call_{seq_no}", "type": "function",
                              "function": {"name": fn.get("name"), "arguments": arguments}})
        return proposals, seq_no

    def open_chat(self) -> None:
        self.intake.read_request()
        self.log.append({"kind": "plan", "plan": self.needs.reading, "roles": self.role_map})
        opening = self.pview.get("opening")
        first_message = "SHOPPER REQUEST:\n" + self.request
        if opening is not None:
            first_message += "\n\nOPENING OBSERVATION:\n" + as_text(opening)
            self.archive.add(opening)
        overview = self.needs.overview(self.step_cap, self.msg_cap)
        if overview:
            first_message += "\n\n" + overview
        first_message += "\n\n" + self.status_line()
        self.chat = [{"role": "system", "content": ACT_RULES}, {"role": "user", "content": first_message}]

    def turn_blocked(self) -> bool:
        wire = self.wire
        if wire.done or wire.broke or self.moves_left() <= 0:
            return True
        return wire.spare() < LAST_CALL

    def execute(self) -> list[dict[str, Any]]:
        if self.wire.done:
            return self.log
        self.open_chat()
        scripted = (ShopperReplyStep(), QueuedReadsStep(), ShelfSweepStep(), ValuePagesStep(),
                    PlannedQuestionStep(), ReferenceHopStep(), RevealSweepStep(), CheapWalkStep())
        actor = ActorTurnStep()
        for _ in range(TURN_LIMIT):
            if self.turn_blocked():
                break
            self.closer.catch_up()
            if any(step.perform(self) for step in scripted):
                continue
            if not actor.perform(self):
                break
        if not self.wire.done and not self.wire.broke:
            self.closer.finish_now()
        self.log.append({"kind": "end", "finished": self.wire.done, "steps": self.wire.tick,
                         "cost": round(self.wire.spend, 5), "seconds": round(monotonic() - self.wire.t0, 1)})
        return self.log


def agent_main(pdata: dict[str, Any]) -> list[dict[str, Any]]:
    run = ShoppingRun(pdata)
    try:
        return run.execute()
    except Exception as err:
        run.log.append({"kind": "crash", "error": f"{type(err).__name__}: {err}"[:500]})
        try:
            if not run.wire.done and not run.wire.broke:
                run.closer.finish_now()
        except Exception as later:
            run.log.append({"kind": "crash", "error": f"{type(later).__name__}: {later}"[:500]})
        return run.log


__all__ = ["agent_main"]
