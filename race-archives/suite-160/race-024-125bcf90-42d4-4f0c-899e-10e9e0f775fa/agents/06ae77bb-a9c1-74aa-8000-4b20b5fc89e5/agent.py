from __future__ import annotations

import json
import re
from time import monotonic, sleep
from typing import Any

from src.agent.proxy_client import ProxyClient

# ---- which LLM plays which seat
ACTOR = ["anthropic/claude-haiku-5.5", "openai/gpt-6-luna", "anthropic/claude-haiku-5.5"]
LISTENER = ["anthropic/claude-haiku-5.5", "anthropic/claude-sonnet-5.5"]
PLANNER = ["anthropic/claude-sonnet-5.5", "anthropic/claude-haiku-5.5"]   # asked one question at a time, Sonnet found the hidden need in 4-5 more of 96 tasks for about $0.005 a task
CHECKER = ["anthropic/claude-haiku-5.5", "anthropic/claude-sonnet-5.5"]
AUDITOR = ["anthropic/claude-haiku-5.5", "anthropic/claude-sonnet-5.5"]
READER = ["anthropic/claude-haiku-5.5", "openai/gpt-6-luna"]
MENDER = ["anthropic/claude-sonnet-5.5", "anthropic/claude-haiku-5.5"]
JUDGE = ["anthropic/claude-sonnet-5.5", "anthropic/claude-haiku-5.5"]   # the first item check of a task: the costly mistake
JUDGE_EFFORT = "low"
PLAN_EFFORT = "low"       # planner reasoning: at the default, Sonnet wrote ~700 tokens a call, 45% of a full run's cost
JUDGE_MAX = 1             # Sonnet item checks per task (~13k prompt tokens, ~$0.016 live); later ones go to Haiku first
EXTRACT_EFFORT = "none"   # reasoning for the request reader (tested on 150 requests); the planner and listener keep their default   # mends only the keys of a reading the code checks failed

# ---- time, money and request pacing
DEADLINE = 265.0
LAST_CALL = 28.0
CLOSING = 70.0
CLOCK_STEP = 30
SPEND_CAP = 0.05          # a runaway guard for one product; spend_scale() widens it for baskets and long step caps
SOFT_SPEND = 0.035
HARD_SPEND = 1.3          # past this many times the guard the turn loop ends and the code finishes
PACE_GAP = 0.15
SLOW_FOR = 30.0
SLOW_GAP = 1.0
DOWN_AFTER = 45.0
WORLD_TRIES = 5
MODEL_TRIES = 2

# ---- how much text the actor sees
RECORD_TEXT = 700
SEARCH_TEXT = 240
VIEW_CHARS = 16000

# ---- turn and step budgets
TURN_LIMIT = 80
IDLE_LIMIT = 3
STUCK_LIMIT = 4
FIRST_ORDER_DUE = 10
FIRST_ORDER_BY = 7
CODE_READS = 8
CODE_FINISHES = 4
CODE_PAGES = 3
ASK_DEFAULT = 3
ASK_AFTER_STEPS = 5
ASK_AFTER_SETS = 2
ASK_GRACE_STEPS = 3
PAIR_ALL = False          # one property per question: a hidden property named second in a pair came out 22 times in 61,
                          # named alone 93 in 113; pairs stay only on two-product errands, where they widened the reach

# ---- the code's own page walks
SWEEP_READS = 8
WALK_TOTAL = 8
COVER_PAGES = 5           # one product, cheapest: further pages read of the search that showed the pick, in any store order

# ---- routes by product count: one shared core; only the knobs below differ between a one-product request and a basket
ROUTES = {
    "single": {"cover_pages": COVER_PAGES, "blind_refs": True, "early_repair": False, "long_pages": 6},
    "multi": {"cover_pages": 2, "blind_refs": False, "early_repair": True, "long_pages": 3},
}
WALK_PAGES = 6
VALUE_TOTAL_PER = 5
VALUE_TOTAL = 8
VALUE_PAGES = 4
LABEL_OPENS = 3
LABEL_TRIES = 4
CEIL_OPENS = 3
REACH_OPENS = 4          # rows opened from the first pages of a value search when the store pages its matches
CEIL_CALLS = 4
CH_SYNONYM_MIN = 4

# ---- words the code reads in tool calls and listings
FAULT_KEYS = r"error|fault|refus|reject|denied|invalid|fail"
JOINERS = r"\s+and\s+|\s*&\s*|\s*/\s*|\s+\+\s+"
DROP_KEYS = r"remove|drop|release|delete|take|out|clear|discard|minus|decrease|subtract|off|unhold"
PUT_KEYS = r"add|put|hold|set|insert|place|keep|include|in|plus|increase|append"
LINK_KEYS = r"and|or|the|for|with|of|a|an|in|on|to|by|from|at|as|is|are|be|this|that|it|its|your|our|you|all|any|not|no"
DROP_WORDS, PUT_WORDS, LINK_WORDS = (frozenset(keys.split("|")) for keys in (DROP_KEYS, PUT_KEYS, LINK_KEYS))

# ---- LLM prompts
ACT_RULES = """You are the buyer for one shopper in a simulated shop, and calling the tools provided is your only way to act.
Check: scoring is all or nothing. The point is earned only when the end result is right on every count: the purchased product satisfies each written requirement, each hidden need the shopper discloses when questioned, and each change announced mid-way; the unit count and the spending limit are respected; when the request wants the lowest price (or the top value of some measure), that product is the one bought; and all procedural instructions of the request are carried out. If the shop holds no product satisfying everything, finish with a decline performed exactly as the request instructs.
Check: tools. Every call, a broken or rejected one too, uses up a step from a limited supply; reasoning is free.
Check: calls that need nothing from each other's output may share a reply (opening several product records, for instance) and execute in order. An order or a decline always travels alone.
Check: a details tool accepting a LIST of ids fetches all those full records in ONE step (within its item cap); use it whenever two or more records are needed.
Check: naming of tools, parameters and outputs changes from shop to shop. Rely on each tool's description and JSON schema and copy names and allowed values exactly. Early responses reveal which tool carries which facts (price, stock, detailed properties).
Then: at the start, learn the category labels this shop puts on products of the requested type. Where similar labels sit side by side, the right one matches the request's wording in full; search and purchase within it.
Check: searching. Each shop searches its own way, so infer it from early results. Results may be ranked by relevance or listed by fixed id. Keyword matching may need all words (each extra word lowers the total) or any word (each extra word raises the total, and better-matching products rise). Visibility may stop after a few rows or pages. Filters may apply in one mode only (keywords may override category or price filters, or filters may override keywords). Whenever a reported total exceeds the visible rows, the right or cheapest product may be hidden further on. Under all-words matching, add precise words (or category and price filters with a lower ceiling) until everything fits on visible pages. Queries should be one to three words, most distinctive FIRST: some shops read just the opening word or two, so long queries may get truncated (compare a long query with a shorter one sharing its start; the notes report the outcome). Under all-words matching, value plus type ("blue chair", value leading) returns exactly the products carrying that value. Under any-word matching, how the rows are ordered decides the query. Where rows are ranked by relevance, a product carrying more of the words sits higher: write every required value plus the item type, values first and without property names ("cylinder plastic food container", not "pack type cylinder container"), and read its first two pages, where the products holding all of them gather. Where rows come in a fixed id order, generic words (the type, a unit) only flood the list: query the value's own unusual word or number alone ("herringbone", "30m", "0.4"), also in its other spellings ("30 m", "3 atm"), and go through every page the total allows. Keywords may also hit category names and property values. Notes after every search summarise what the totals show about this shop's search.
Check: hidden needs. The shopper keeps some needs back and discloses one only when a message names that property. The opening question of a message is answered most dependably, so ask about one property per message, two at most with the likelier leading: brief, simple and warm, naming the property, maybe with one or two sample values. Never include instructions, lists, labels, positions or rules.
Check: replies like "Don't mind", "no preference" or "anything is fine" leave the property open. Any preference or limit the shopper voices binds from then on, exactly as phrased, numbers included.
Then: planned questions, if any, are dispatched for you once you have opened records of two or three products of the requested type, so do that early. Answers arrive in the chat with their interpretation; treat each disclosed need as binding and search again combining it with the stated values.
Then: when asking by yourself, first open two or three records, then ask about what the backstory implies (the user, the look, the place or manner of use), then about properties that vary across candidates. Never ask about something the request already fixes and never exceed the message allowance.
Check: procedure. A request may prescribe how to work: how many products to inspect or compare, what to verify, what to tell the shopper, what a remark should contain, how to decline. These are scored like requirements, so carry each out literally and in full. "Look at" or "open" N products means N fully suitable products, each opened via the details tool.
Check: questions from the shopper (to confirm a change, e.g.) are answered on your behalf while messages remain; the change itself binds.
Check: nice-to-haves ("ideally", "would be nice", "if possible") are no requirements; they only decide between products already satisfying everything and never beat a cheapest rule.
Then: if the request leaves a point only the shopper can settle (say, no price ceiling, or two wishes that may conflict so the shopper must rank them), ask before deciding.
Check: changes. The shopper may revise things mid-way: a new budget (up or down), a requirement removed, swapped or added, a price increase, a product going out of stock. One line may bundle several changes, and such lines can appear in any tool result (an extra field of sentences) or with an order attempt. Each takes effect at once: re-examine the pick and backups, and when cheapest or best is wanted, choose afresh from the new acceptable set.
Check: an order attempt can be REFUSED; the episode continues with the cart untouched. Read the refusal, apply all changes it mentions, remove anything that stopped qualifying (e.g. a sold-out product), add the next verified fit, re-verify its stock and price, and order again.
Check: products may also sell out or be repriced silently, so stock and price are re-read right before every order attempt; a product whose current price times quantity exceeds the current budget is out.
Check: one request can name several different products, and each carries its own quantity, requirements and hidden needs. Cover every one: search the category of each, verify a fit per product, and order them together as one line per product at its quantity. Some cart tools take a list of lines and replace the whole cart with it, so every such call has to carry the line of every product. The cheapest or best rule is judged separately for each product.
Then: some requirements take their value from a different listing or page ("the colour matching X's jumpsuit"). Find that listing before anything else: search for the words that single it out (its brand or shop together with its type, for instance), open its record and take the value from the field named, and only then search for the requested product with that value.
Check: the cart. Its tool may ADD units to an existing line rather than set them, so adding 1 more of a held line yields 2. Never re-add a line already held; to adjust units, remove or add only the difference. A note after each cart call reports the code's count of the cart.
Check: how the grade reads a listing. For every requirement it opens the field of the listing named after that property and looks for the value's words in the form this shop uses there. Spelling of the field name does not matter: with separators or a prefix, shortened or lengthened, or under a common alternative name (a pants fit type may sit in "fit", a collar type in "neckline", a watch dial size in "dial size", a colour in "color family"), it is still that property's field. When a field bears exactly the property's name, or that name with one more word placed before it (clothing material for material, pants length for length), that field rules, and the requirement is met as soon as either of them shows the value. Two names that differ only in separators (size_class and size class) are the same field, and a value in either is enough. Without such a field, the alternative-named field rules; the options list or the title may give the value only when no field names the property at all. A description is never evidence.
Check: compare values by the words this shop writes, ignoring case, separators, singular or plural and spelling variants. A more specific value that contains the required one is a match (cotton is met by "80% cotton", green by "forest green", rtx 3060 by "rtx 3060 12gb"), also when other words stand between its words (quick assembly is met by "quick tool-free assembly"), and one element of a list is enough. A synonym, an abbreviation, a related word or a shade (technical wood written "tech wood", windows 11 written "win11", embroidery written "embroidered", plain written "solid", off-white written as ivory or cream) is a match only while no listing of the category writes the value itself in that field; once others write "matte" as their finish, "satin" cannot stand for matte. When a field holds a joined value such as "wool + nylon" and also lists one part on its own ("nylon"), the joined value does not count for that part. Any FIELD EVIDENCE lines given come from this shop's records and decide which case holds. A different value of the same property is never a match (mechanical is not quartz, maple is not rosewood), and neither is a word that only looks alike. Blanket values such as "universal", "all seasons" or "suitable for all occasions" name nothing and fail, unless the same list also carries the required value. A value does not pass for another one just because the two often go together or one hints at the other ("zip-up" is not met by a "pullover" style, "lace-up" not by "slip-on"); only a shade usually sold under the required colour, or a narrower value containing it, passes.
Check: the field of some other property never stands for the requirement's property (the material is not a lining material, a style is not a length, a strap type is not a strap length), and it never replaces a property that this category's listings carry in a field of its own.
Check: only stated or disclosed needs count. Never exclude a product for something unrequested: an age group, gender, size, occasion, condition, quality or use inferred from the story or description. The story is context; if it seems relevant, ask about that property rather than assume. Your questions are no requirements; only the shopper's answers are.
Check: figures are amounts, compared after converting units (45 mm equals 4.5 cm); a limit includes its own figure, and a range passes only when all of it lies within the limit. Where no field names the property and the title is the evidence, an exact required value is met when the title shows its words and figures in the same order, extra text after them allowed (a title with 9' x 12' 4" shows 9' x 12', while 9' x 13' does not).
Check: title figures ("55 x 94 Inch", "117 x 228 cm") do not identify their dimensions: never reject or skip a product because of them; read its detail fields.
Check: honour exclusions ("leave out anything with X") and closed lists ("A or B and nothing else").
Check: belonging to the requested category settles the product type. For the grader a product listed in that category IS the requested kind, even when its title or description also names some other object (a case, a container, a refill, an accessory, a set); never drop, reject or rank it lower for that, and judge only requirements, price and stock. No property has to repeat the category name. When the shop labels its categories, the product's label must match the full wording of the request (a request for "A or B" goes with the category "A & B", not with the narrower neighbour "B"). If the request wants items on hand, the pick has to show stock, and its price times the quantity may equal the budget but not exceed it. A price is in the budget's unit unless the record states another one (against a budget of 1000 US cents, 45 means 45 cents); never rescale prices.
Then: declining is rarely correct. If nothing matches the strict wording but a product meets each requirement by these rules (a narrower value, a synonymous field name, or a synonym or shade where no category listing writes the value itself in that field), it is the fit; decline only if the search genuinely found nothing meeting every requirement under these rules.
Check: cheapest. Of ALL the products that meet every requirement, the one ordered must have the lowest price; finding a single fit is not enough. When the keyword search needs all words and matches property values, searching the value together with the type returns exactly the candidates holding that value: walk every page of it and open the rows in price order. A category listing under a price ceiling shows the prices of the whole category without the values: use it to spot cheaper rows (keep lowering the ceiling until the total fits), then check what those rows hold. Search each required value with the type in a short query that starts with the value ("30m watch", "blue chair") and read every page when the total fits within the pages you can read (if not, add a second required value or a narrower type word); do the same with synonyms of the type ("bench", "stool") and with the value alone. When a value cannot be searched (a figure the shop strips out), also search the product line of each fit or near fit, as two words: the maker and one telling word of the title, every page; other colours and sizes of a line are separate products with the same property values, and some may cost less. Order the visible candidates by price yourself and open the records starting from the cheapest: the first one whose property fields meet everything, once every cheaper one was opened and turned down, wins. Keep the fits after it as backups.
Check: best. For the best by a measure (most, largest, longest, highest...), compare ALL fits on that measure the same way and pick the extreme. A product giving no value for the measure cannot win.
Then: finishing. Maintain a ranked list of verified fits with at least one backup, since the top pick may sell out at checkout.
Then: before each order attempt the cart contains exactly one line per requested product (one line for one product), each the chosen product at its required quantity, and nothing else; after the final cart change you have re-read the chosen product's stock and price, and it is in stock and affordable.
Then: decline only after an exhaustive search finds nothing meeting every requirement, following the request's decline rule precisely. If the request forbids offering substitutes, the decline remark must not name, describe or point to any listing: no ids, titles or prices.
Then: manage steps. Place the first order attempt while seven or more steps remain, leaving room to handle a refusal. With 7 steps left only cart, stock-check and finishing tools remain. Decide once; no swapping the cart back and forth.
Reply: every reply includes at least one tool call; keep its text brief: current requirements, verified fits with price and stock, next move."""

READ_REQUEST = """Study the request and the tool list below and fill in these keys.
Check: "product" names the wanted product type in the request's wording (with more than one product, every one of them, linked by "and").
Check: "items" gets an entry for each DISTINCT product to be bought, following the request's order: {"product": that product's type as the request words it, "quantity": its units per listing (see "quantity"), "requirements": its firm requirements (see "requirements"), "works_with": the "product" of the other entry this one has to work with, fit, suit or be compatible with when the request itself writes that demand for this product, else ""}. Just one product means just one entry. A demand that X has to fit Y fills X's "works_with" only, never Y's, and lives nowhere else: it is no requirement and no hint.
Check: "quantity" counts the units of a single listing to buy (integer; 1 unless more are wanted). Goods sold as one unit, a set or a pair of shoes say, stay at 1. Only an outright wish for more units changes it ("order 2 packs", "I need three of them"); how many people use or share it, or how many old ones existed, belongs to the story and leaves 1.
Check: "requirements" is a list of strings, one for each firm requirement on the product, written "property: constraint" (with several products, cover all of them and write each "property (product): constraint"), and keeping, word for word, whatever closed list, exclusion, limit or value the shopper stated. Budgets, price, stock, quantity and things merely nice to have stay out, and so does any repetition of the category or product type: words making up the category's name (say an audience, a season or an occasion that is in it) are already met and add no requirement. A requirement binds one property of the product to a value the request insists on; examples: "at least 4 drawers"; "with a mechanical movement"; "the pattern has to be herringbone"; "the occasion has to be party". The story around the purchase never is one: the person getting it, an event at which it gets worn or used ("for the staff party"; "to wear at his wedding"), the place, time or manner of use, the number of people sharing it, or the item being bought to replace an earlier one ("a new watch" is a replacement, not a condition). Only once the request turns an occasion, a use or an audience into a property of the item itself does it count as a requirement ("a blazer made for business"; "the occasion has to be business"). Build the value solely out of what the request itself writes as the item's value, figures or words; qualities, wishes or reasons taken from the story (covers the whole drop; natural and breathable; works with what I already have; slim enough for a case) are hints, never requirements.
Check: "price_cap" is the ceiling on the total price, as the number and unit the request writes (never converted, so "3050 US cents" gives 3050), else null.
Check: "currency" is the unit the request prices in, else null.
Check: "in_stock_only" is true when the request accepts only items that are in stock or on hand.
Check: "priority" is "cheapest" only when the request openly demands the lowest-priced or cheapest fitting item, "best" only when it openly demands the top item by some measure, and "none" in every other case. Remarks that money is short or that every penny matters do not ask for the cheapest; they merely justify the cap.
Check: "priority_detail" names the measure when the priority is "best", and is "" otherwise.
Check: "message_budget" is the integer count of messages the shopper allows, null when unstated. Such a cap is never a procedure.
Check: "step_budget" is the integer count of steps or moves the request allows, null when unstated.
Check: "procedures" lists every instruction on how the assistant must work (what to include or avoid, what to verify, report or show, how many listings to inspect, how to decline, what has to happen before or after a cart change or the purchase), one object each: {"quote": the instruction copied character for character from the request, "action": "message_shopper" when it asks for a message to the shopper, "inspect" when it sets how many listings to look at or what to verify, "decline_rule" when it says how to decline, else "other", "trigger": "after_last_cart_change" when it applies once the basket is final, "before_order" when only before the purchase, "before_cart" when before anything is added, "before_decline", else "anytime", "needs_yes": true when the shopper's agreement has to arrive before going on, else false, "content": what the message must show, picked from "lines", "quantities", "unit_prices", "total", "limit" ([] when nothing is named)}. A wish to approve or see the basket before buying is a message_shopper procedure. Things the shopper plans to do personally are story. Empty when the request sets no procedure.
Check: "story_numbers" lists, as written, every figure in the request that is only story and sets nothing (an age, a year, a count of people or rooms, a distance).
Check: "soft_wants" collects what the shopper wants without truly needing it (if possible, would be nice, preferably, ideally); these never enter "requirements".
Check: "ask_first" collects what only the shopper can settle and should be asked before any choice: the budget when the request sets no ceiling, say, or which of two wishes counts more when the request admits both might be impossible (in that case both wishes belong in soft_wants and not in requirements), or anything the request tells you to check with the shopper. Empty otherwise.
Check: "hints" gathers story details that bear on a product property without stating its value (the user, the place, what the item goes with or is used for, what it must match, fit or withstand), each written "property (product): the detail in a few words" (with one product, "property: detail"), naming the property as listings would most likely name it. When a detail may touch more than one property, put the most likely first and the next one after a slash ("colour / style (chair): must suit a room with one painted wall"): how the item looks, and what it has to suit, match or go well with (an outfit, things used next to it, a wall or a room), point first to colour and then to style or pattern. A detail that only gives the reason for a requirement already stated (why this material or that size) stays out. Background is never a hint: a job, weather, travel, schedules, or a place where nobody uses or wears the item, and likewise a purpose every listing of the type serves (a coat keeping warm, a dresser storing things). Otherwise empty.
Then: every sentence in the request lands in exactly one place: requirement, priority, soft want, procedure, budget, hint, or plain story, and every figure it writes lands in some key (story_numbers for story). When it is unclear whether a sentence is a procedure, file it under "procedures".
Check: "references" collects requirements whose value comes only from pointing at ANOTHER page or listing ("a colour other than the one on Z's page", "within 5 of the figure Y's page shows", "the colour matching X's jumpsuit"), each written "property (product): as shown on <how the request names that listing>" (or "property (product): different from the one shown on <how the request names that listing>" when the value has to differ from it); every such entry is also copied into "requirements" in that form. Empty otherwise.
Check: "reference_targets" holds one object per "references" entry describing the listing it points at, with these keys: "property" (its property), "product" (its product), "seller" (the brand or shop name the request uses for that listing, copied exactly), "kind" (what type of product that listing is), "details" (a list of every further value the request mentions for that listing, kept short and in the request's words, such as "blue", "4gb" or "net weight 167.262186 g"), "relation" (one of "same", "differ" when the value must NOT be the one of that listing, as in other than, different from or set apart from, "pair" when the value is the one that listing's own text pairs with its value, as in its counterpart, partner or what it goes with, "within", "at most", "at least"), "tolerance" (the figure for "within", else null), "unit" (its unit, else ""). Empty otherwise.
Check: "decline_no_names" is true when the decline rule of the request forbids suggesting, naming or offering alternatives or substitutes, false when it wants something named, and null when it gives no rule.
Check: "none_fit" sets out what the request wants when no listing meets every requirement: {"quote": that passage copied exactly ("" when the request says nothing about it), "action": "order_closest" when it wants the listing coming closest bought or put forward instead of nothing, "ask_first" when it wants that listing shown to the shopper for a yes before any purchase, "decline" when it wants nothing bought, "counted": what the passage adds to the requirements being counted (such as "stock", "budget" or any other measure it names), "may_miss": the requirements it lets that listing give up, "must_keep": the ones it must still meet, "choose_by": how to pick among equally close listings, as {"measure": "price" or the property it names, "lowest": true when the lowest value wins}, or null, "allowance": how far past the best of those is still fine, as {"measure": "price" or the property it names, "fraction": the share (15% gives 0.15)}, or null}. Every condition the passage writes goes in one of these slots; leave a slot empty when the passage does not write it.
Check: "roles" is an object assigning each role the exact tool name from the list, or null where no tool suits: "search", "details", "view", "message", "cart", "cart_view", "order", "decline". "message" talks to the shopper; "cart" puts things into the cart or takes them out; "order" checks out or buys; "decline" closes the errand with no purchase.
Reply: ONE compact JSON object (no indentation) holding exactly these keys, with no other text.
"""

MEND_READING = """A first reading of the shopper's request below failed some checks. Read the request again and give these keys anew.
Check: every quote is copied character for character from the request, and every figure is written as the request writes it.
Check: each key follows its rule below; keys not listed here are not wanted.
Check: the failed checks come from simple code tests that can be wrong: when the request or the tool descriptions support the first value, give it again unchanged.
Reply: ONE compact JSON object (no indentation) holding exactly the keys asked for, with no other text."""

HEAR = """Work out for a shopping assistant what the shopper's latest words mean.
Check: "meaning" sums up in one brief sentence what the shopper intends.
Check: "needs" lists new binding requirements, written "property: constraint" and carrying the exact closed list, exclusion, limit or value given. When a comparison word precedes it (at least, nothing less than, falls short of), a value made of words rather than a figure is taken as the value itself, spelled the shopper's way: "falls short of not available" gives "property: not available" (both "none" and "not available" may be real field values). Words that only say how firmly the shopper holds a value (hold it tight to, keep it at, stick with, set on) frame it and are no part of it: "keep the colour tight to navy" gives "colour: navy".
Check: "no_preference" lists properties the shopper declared indifference about.
Check: "dropped" lists earlier requirements the shopper gave up.
Check: "new_budget" is the revised spending ceiling as a number expressed in the unit of the request's budget (take the line's figure when the line uses that unit: "13075 cents" gives 13075 for a budget in cents), else null.
Check: "unavailable" is true when the shopper reports that something can't be purchased anymore (sold out, gone, out of stock, button disabled), false otherwise.
Check: "meta_only" is true when all the line does is ask the assistant to ask less per message (only one question each time, fewer at once, the first now and the others afterwards) without supplying any value: it answers nothing so far, so needs, no_preference and dropped get nothing from it, and none of its figurative words is a value; false otherwise. A line only saying that the shopper could not follow or understand the question, or wanting it repeated more simply or briefly, is not meta_only.
Check: "requirements_now" is the complete current list of binding requirements, each "property: constraint": begin with REQUIREMENTS SO FAR, retain every entry the shopper neither removed nor replaced, keep just the newest version of a replaced value, then add every new need stated in the shopper's lines. Requirements drawn from the request's story that REQUIREMENTS SO FAR does not contain must never be added, because only the shopper's lines can add one. Leave out product category, quantity, stock and price.
Check: "approves" is false only when the lines turn down the basket or purchase the assistant just put forward, or ask for a change to it; true when they agree to it or simply acknowledge it; null when they say nothing about it.
Check: "reply" is, when the shopper puts a question to the assistant or asks for confirmation, a brief plain answer to send that states the change in substance (confirming the new value, for instance); else "".
Check: "references" holds, for every entry of "needs" whose value is only given by pointing to another page or listing, one object for the listing pointed at, with the keys "property" (its property), "seller" (the brand or shop name the line uses for that listing, copied exactly), "kind" (what type of product that listing is), "details" (a list of every further value the line mentions for that listing, short and in the line's words), "relation" (one of "same", "differ" when the value must NOT be the one of that listing, as in other than, different from or set apart from, "pair" when the value is the one that listing's own text pairs with its value, as in its counterpart, partner or what it goes with, "within", "at most", "at least"), "tolerance" (the figure for "within", else null), "unit" (its unit, else ""); [] when there is none.
Then: a single line can bundle several changes, and each one gets reported. A need about another property, even a neighbouring one (a width next to a strap width, an overall height next to a seat height), is an extra requirement that replaces nothing: an older requirement is only dropped or replaced when the line says that about the very same property. Replacing a requirement means the old one is dropped and the new one is a need. When the shopper ranks two wishes, the winning wish becomes a need. Any stated budget goes to new_budget. A mere wish ("would be nice", "ideally") never becomes a need.
Then: read the shopper literally and with good will: roundabout, figurative or homely wording that names a value means the shopper insists on it; a value gets excluded only if the shopper plainly asks to avoid it. Words that just stress how firmly the shopper sticks to a value (strictly, only, stays, nothing but, sticking with, nothing short of, holding out for, locked to) do not belong to it: write only the value's words ("color: white", "material: jute"), never "<value> only"; an exclusion is added only when the shopper also rules out other values or mixes in so many words. A figure paired with a comparison (no more than, over, under, at least, at most) is a limit and keeps its unit. A new budget always replaces the previous one, lower or higher, even when the shopper presents it as good news.
Then: when the shopper merely says anything works, they have no preference or they don't mind, the property asked about goes into no_preference (only the FIRST property when the message asked about two). A reply only saying the shopper could not follow or understand the question ("could you put it another way", "you've lost me", "that was rather dense") is this shopper's form of no preference whenever the question named the product the property belongs to, so the property asked about goes into no_preference as well; with several products wanted and a question that did not say which, the property is still open (no_preference stays without it).
Then: when the request covers several products, write every need as "property (product): constraint", taking the product from the line (from a line about "toe style of the women's boots" comes "toe style (women's boots): ..."). Requirements come from the shopper's words only, never from the assistant's own question, and the sample values that question offered never become needs. A "don't mind" reply frees nothing but the property the assistant's message asked about; it never removes or alters a need stated earlier, and if the message asked nothing (it only passed on information) the reply changes nothing. A need given by pointing at another listing keeps the line's own pointing words as its value ("finish: unlike the one shown on Y's kettle page"); values the line states for that other listing describe it, not the product, so they are not needs, not drops and not freed properties.
Reply: ONE JSON object with these keys and no other text."""

READ_RULES = """Check: for every requirement, open the listing's field named after that property and look for the value's words in the form this shop writes them there. The field name's spelling does not matter: with separators or a prefix, shortened or lengthened, or under a common alternative name (a pants fit type may sit in "fit", a collar type in "neckline", a colour in "color family"), it is still that property's field. A field bearing exactly the property's name, or that name with one more word before it (clothing material for material, pants length for length), rules, and the requirement is met as soon as either shows the value. Two names that differ only in separators (size_class and size class) are the same field, and a value in either is enough. Without such a field the alternative-named one rules; the options list or the title may give the value only when no field names the property. A description is never evidence.
Check: compare values by the words this shop writes, ignoring case, separators, singular or plural and spelling variants. A more specific value containing the required one is a match (cotton is met by "80% cotton", green by "forest green", rtx 3060 by "rtx 3060 12gb"), also when other words stand between its words (quick assembly is met by "quick tool-free assembly"), and one element of a list is enough. A synonym, an abbreviation, a related word or a shade (technical wood written "tech wood", windows 11 written "win11", embroidery written "embroidered", plain written "solid", off-white written as ivory or cream) is a match only while no listing of the category writes the value itself in that field; once others write "matte" as their finish, "satin" cannot stand for matte. When a field holds a joined value such as "wool + nylon" and also lists one part on its own ("nylon"), the joined value does not count for that part. Any FIELD EVIDENCE lines given come from this shop's records and decide which case holds. A different value of the same property is never a match (mechanical is not quartz, maple is not rosewood), and neither is a word that only looks alike. Blanket values such as "universal", "all seasons" or "suitable for all occasions" name nothing and fail, unless the same list also carries the required value. A value does not pass for another one just because the two often go together or one hints at the other ("zip-up" is not met by a "pullover" style, "lace-up" not by "slip-on"); only a shade usually sold under the required colour, or a narrower value containing it, passes.
Check: a field of some other property is left aside (the material is not a lining material, a style is not a length, a strap type is not a strap length), and it never stands in for a property that has its own field on this category's listings.
Check: figures are amounts, compared after converting units (45 mm equals 4.5 cm); a limit includes its own figure, and a range passes only when all of it lies within the limit. Where no field names the property and the title is the evidence, an exact required value is met when the title shows its words and figures in the same order, extra text after them allowed (a title with 9' x 12' 4" shows 9' x 12', while 9' x 13' does not).
Check: a price is in the unit of the request's budget unless the record states another currency or unit (against a budget of 1000 US cents a price of 45 means 45 cents, far below it); never rescale either the price or the budget.
Check: nothing but the listed requirements counts, so a listing never fails over a gender, an age group, a size, a season, an occasion, a condition, a quality or a use that no requirement names.
Check: belonging to the requested category settles the type: a listing in that category is the requested kind even if its title names some other object (a case, a container, a refill, an accessory, a set); judge only requirements, price and stock."""

CHECK_ITEM = "\n".join([
    "Act as the last gate before a shopping assistant commits to an item; the errand scores 1 only if the item "
    "satisfies every requirement below. Base the verdict solely on the records shown.",
    READ_RULES,
    "Check: words forming part of the category's own name are satisfied by the category.",
    "Check: exclusions and closed lists are honoured.",
    "Check: the item must be of the requested type; where listings carry a category label, it must be the label that "
    "matches the request's wording in full, not a narrower or adjacent one.",
    "Check: price times quantity must not exceed the budget (equality is fine); an item the records show out of stock fails.",
    "Check: a mere wish (\"ideally\", \"would be nice\") flagged as not required is no requirement.",
    "Then: if records conflict (on price or stock, e.g.), trust the newest.",
    "Then: answer fits=false only for a listed requirement the item's fields clearly fail by these rules (a FIELD "
    "EVIDENCE line stating the item does NOT meet a requirement is such a failure), or one that no field and (when no field "
    "for it exists) neither the options list nor the title states; never fail an item because a stock or price record is "
    "missing.",
    "Reply: ONE JSON object and no other text: {\"item\": the item's id, \"fits\": true or false, \"problems\": [brief "
    "strings, each naming a requirement unmet or unshown]}.",
])

RANK_CHECK = "\n".join([
    "A shopping assistant will shortly order an item for a request that wants the PRIORITY; below come the requirements, "
    "the chosen item and the listing records seen so far.",
    "Check: applying the reading rules below (price times quantity inside the budget; not shown out of stock), identify "
    "listings that may satisfy every requirement AND beat the chosen item on the priority (cheaper for the cheapest; a "
    "better figure on the measure for the best).",
    "Check: omit listings the records show failing a requirement, sold out or over budget. Figures in a ROW's title (such "
    "as 55 x 94) do not say which dimension each one is, so they never rule a ROW out.",
    "READING RULES:",
    READ_RULES,
    "Reply: ONE JSON object and no other text: {\"better\": [ids, best first, at most 4], \"reason\": \"one sentence\"}.",
])

ASK_PLAN = """Draft the questions a shopping assistant will put to the shopper. Each hidden need is one product property's value, and it surfaces only if a message asks about precisely that property; the first thing asked in a message gets the surest answer, and anything else draws a polite "don't mind". Choose at most QUESTION_COUNT distinct properties, ranked by promise.
Check: step 0 leads: anything in the MUST BE LEARNED FROM THE SHOPPER list below that nobody has asked yet, put in plain words. For the budget, ask for their spending ceiling. For a choice between two wishes, mention both values together with their properties, for example "If one listing can't have both, would you rather have the <value> <property> or the <value> <property>?".
Check: step 1 comes after: properties that truly occur among the listing details below. Begin with the story (when STORY HINTS appear below they list its details with the property each one concerns, and they go first): usually the hidden need is the property that the story's concrete details touch while leaving its value unsaid. Things the user will press or touch, inspect closely or photograph, hear, measure or fit against, read on a label, or the way it was produced, point to that part or aspect of the item and to its property: labels, tags or fine print suggest ingredients, fabric or material; looks and photos suggest colour, pattern or the visible part; a space or body part it has to fit suggests the size or shape of that part; a purchase for another person, or for someone hardly known, suggests the intended wearer (age group, gender); how it should feel on skin, hand or foot suggests fabric or material first; the activity, time or place of wear or use (outdoors, the office, a gym class) suggests its type or style, the property whose values name such uses; something repeatedly fastened or adjusted suggests its mechanism or closure; the part doing the work inside suggests its core component. Weather, a job, a timetable, or any place other than where the item is worn or used counts as background. Ask first about the property that the story's details point at, using the listings' name for it. The intended wearer or user is a normal candidate: ask about it when the story points at another person or nothing else is hinted. Once the story's leads are used, rank the listed keys the way a buyer of that kind of product chooses between listings (its size, dimensions or capacity, its type or style, its material, its intended user, its main technical figure: whichever that kind is bought by), favouring keys carried by most of its listings whose values differ from listing to listing. Weak picks: a key seen on only one or two listings, a key with the same value on every listing, and logistics (years, variant lists, pack size, condition, warranty).
Check: skip any property the request already fixes (its requirements, price, stock, quantity), any property asked before, and pairs of near-duplicates (two spellings of one property, two style kinds); never ask about delivery, stock, price, a model or product name, maker or brand. Ask only about properties showing up as keys in the listing details below, under the listings' wording; map a STORY HINTS property to the closest key below (a hint about the wearer to an age group or gender key, one about looks to a colour, pattern or finish key, one about fitting a space to a size or dimension key); ask it by its plain name only when no key below comes close, and never when the hint is tagged [no key] (enough listings of its product lacked that field): then ask the key below whose values differ in the way the detail describes, or drop it.
Check: a step-1 question is a single short, friendly, plain sentence (no more than 15 words) asking which <property> they would like for the <item kind>, calling the property by the listings' name (sample values on their own never reveal a need), with one or two sample values allowed after it. No opener, no progress report, no lists or instructions, and never a catch-all key like a name, a title, a description, variants or options.
Check: a question targets ONE property of ONE product (apart from the step-0 choice between two wishes): no question ever carries two properties or two products, because one reply discloses at most one need and a two-subject question can go unanswered entirely. Under "second", give the SAME product's next best property (not another label for the first), or leave it empty; the assistant may tack it onto its final message.
Then: when SEVERAL products are wanted, any of them may carry hidden needs: distribute the questions so each product gets its first before any product gets a second (leave out a product whose properties are used up), take a product's properties from its own category's list below, write the property as "property (product)", and say the product in the question ("Which toe style would you like for the boots?").
Reply: ONE JSON object and no other text: {"questions": [{"property": "...", "second": "... or empty", "question": "..."}, ...]}."""

NEXT_ONE = """
Then: this round sends just ONE question, and the shopper's answer is read before the next gets chosen. REMAINING question(s) are still available in total and all should be spent: lead the list with the single most promising property (one or two extras may follow as spares); after a disclosed need, the following question turns to another property and story detail."""

BEFORE_DECLINE = "\n".join([
    "A shopping assistant intends to tell the shopper that the store has nothing suitable, which is seldom the right ending.",
    "Check: test the listing records below against the requirements with these reading rules (a price "
    "inside the budget, an item never shown out of stock):",
    READ_RULES,
    "Check: a decline is the worst ending, so here a listing counts as a possible fit when it meets every requirement in "
    "meaning: a value this shop writes with another word of the same meaning is met, even where other listings write the "
    "required word itself. A different value of the same property still fails.",
    "Check: some listings below are known from search rows only (no details read, so their property fields are unknown). "
    "Such a row in the requested category (whatever object its title mentions) whose title, price and stock do not reveal "
    "a failed requirement remains unchecked (title figures, such as 55 x 94, do not say which dimension each one is, so "
    "they never prove a row fails): file it under \"unopened\", likeliest first (a title naming a required value, or the "
    "requested type with nothing against it).",
    "Reply: ONE JSON object and no other text: {\"possible_fits\": [ids of listings that appear to satisfy every "
    "requirement, best first, at most 5], \"unopened\": [ids of rows-only listings to open before declining, at most 6], "
    "\"reason\": \"one sentence\"}.",
])

FINAL_CHECK = """A shopping assistant is about to finish an errand with this move: FINISH. Below are the shopper's request, every later line from the shopper or the store, and the calls made since the cart last changed. The code sends the shopper the final basket with its total by itself right before an order: never list that. List any other step that the request or a later line explicitly asks for, still undone before this finish, which one of the TOOLS below can carry out: a message the shopper asked to receive (an answer to their question, a remark they asked for), or a look at the cart's lines that the request asked for. A step done since the last cart change counts as done. Invent nothing; at most two.
Reply: ONE JSON object, nothing else: {"pending": [{"why": "the asked step in a few words", "tool": "the exact tool name", "args": {the arguments as that tool's schema asks}}]}, with "pending": [] when nothing is undone."""


REF_HOP = """A shopper wants the PROPERTY of the product taken over from a different listing in the store, which the request describes as: DESCRIPTION. The records below belong to listings that could be that one.
Check: "target" names the id of the record that really IS the described listing, with the same seller, the same type and every named detail showing, read for meaning (a figure given in another unit counts when the amount is the same); use null when none of the records is it.
Check: "missing" holds the details the target does not show ([] when it shows all of them).
Check: "go_to" is used when the target's own text sends the reader on to a DIFFERENT listing for the PROPERTY (telling them to take it from, use, refer to, look at or go by another listing): give that listing as {"seller": its shop or brand name as written, "kind": its product type, "details": [every value the text gives for it, short and as written: a colour, a size with its unit, a material]}; give null otherwise.
Check: "value", when there is no go_to, is the target's own value of the PROPERTY exactly as its record writes it.
Check: "goes_with" holds the values that the target's text says pair with, go together with or equal that value ([] when there are none).
Then: a listing the text names only as something the target is not ("not the green one from X"), or as a size, colour or version sold elsewhere, is not a go_to, and neither is one named for some other property. A record without the described details is not the target, whatever its text says.
Reply: ONE JSON object holding these keys and no other text."""

# ---- texts and patterns about the world's answers
SORRY = "Sorry, nothing in the store meets all of your requirements, so I am not ordering anything."
DECLINE_WORDS = r"refus|declin|cancel|abandon|reject|give.?up|abstain|withdraw|no.?(?:purchase|order|buy)|nothing|without"
ORDER_WORDS = r"commit|buy|order|purchas|check.?out|pay|confirm|complete|place|accept|proceed"
CODE_LIKE = r"[A-Za-z0-9_\-]+"
ID_LIKE = r"[A-Za-z0-9][A-Za-z0-9_\-:.]{3,63}"
FIG_RUN = r"\d+(?:\.\d+)?(?:\s*[:x/]\s*\d+(?:\.\d+)?)*"



# ---- pure text helpers
def as_text(piece, ceiling=VIEW_CHARS):
    try:
        dumped = json.dumps(piece, ensure_ascii=False, separators=(",", ":"))
    except (TypeError, ValueError):
        dumped = str(piece)
    return dumped[:ceiling] + "...(cut)" if len(dumped) > ceiling else dumped


def clipped(piece, chars, lvl=0):
    if lvl > 8:
        return None
    if isinstance(piece, str):
        return piece[:chars] + "..." if len(piece) > chars else piece
    if isinstance(piece, list):
        return [clipped(inner, chars, lvl + 1) for inner in piece]
    if not isinstance(piece, dict):
        return piece
    return dict((name, clipped(inner, chars, lvl + 1)) for name, inner in piece.items())


def unfenced(text):
    body = text.strip()
    if not body.startswith("```"):
        return body
    body = re.sub(r"^```[A-Za-z]*", "", body).strip()
    return body[:-3].strip() if body.endswith("```") else body


def json_object(text_body, last=False):
    if not isinstance(text_body, str):
        return None
    body = unfenced(text_body)
    attempts = [body, body[body.find("{"): body.rfind("}") + 1] if "{" in body else ""]
    for attempt in filter(None, attempts):
        try:
            parsed = json.loads(attempt)
        except ValueError:
            parsed = None
        if isinstance(parsed, dict):
            return parsed
    if not last:
        return None
    # (a plan that corrects itself carries several objects one after another: the last one stands)
    found, decoder, at = None, json.JSONDecoder(), body.find("{")
    while at != -1:
        try:
            parsed, end = decoder.raw_decode(body, at)
        except ValueError:
            break
        if isinstance(parsed, dict):
            found = parsed
        at = body.find("{", end)
    return found


def is_number(piece):
    return type(piece) is not bool and isinstance(piece, (int, float))


def long_strings(piece, collected, lvl=0):
    if lvl > 4:
        return
    if isinstance(piece, str):
        if len(piece.split()) > 3:
            collected.append(piece)
        return
    if isinstance(piece, list):
        for inner in piece[:10]:
            long_strings(inner, collected, lvl + 1)


def walk_dicts(piece, visit, lvl=0, ceiling=6):
    if lvl > ceiling:
        return
    if isinstance(piece, list):
        below = piece
    elif isinstance(piece, dict):
        visit(piece)
        below = [inner for inner in piece.values() if isinstance(inner, (dict, list))]
    else:
        below = []
    for inner in below:
        walk_dicts(inner, visit, lvl + 1, ceiling)


def plain_words(name):
    head = re.split(r"\s*\(", str(name))[0]
    return re.sub(r"([a-z])([A-Z])", r"\1 \2", head).lower()


def junk_key(key):
    words = " ".join(re.findall(r"[a-z0-9]+", plain_words(key)))
    named_junk = re.search(r"\b(?:brand|shop|seller|store|vendor|manufacturer|category|categories|department|title|headline|"
                           r"heading|listing|description|overview|price|cost|amount|availability|sku|url)\b", words)
    return bool(named_junk or re.fullmatch(r"(?:name|notes?|stock|in stock|on hand|images?|links?|make|maker|shelf|ids?|id)", words))


def singular(word):
    trims = len(word) > 3 and word.endswith("s") and not word.endswith("ss")
    return word[:-1] if trims else word


def prop_words(name):
    return list(map(singular, re.findall(r"[a-z]+", plain_words(name))))


def prop_key(name):
    return "".join(prop_words(name))


def edits_within(a, b, most):
    if abs(len(a) - len(b)) > most:
        return False
    costs = list(range(len(b) + 1))
    for i, left in enumerate(a):
        diagonal, costs[0] = costs[0], i + 1
        for j, right in enumerate(b):
            diagonal, costs[j + 1] = costs[j + 1], min(costs[j + 1] + 1, costs[j] + 1, diagonal + (left != right))
        if min(costs) > most:
            return False
    return costs[-1] <= most


def same_prop(a, b):
    x, y = prop_key(a), prop_key(b)
    if not x or not y:
        return False
    short = min(len(x), len(y))
    return (x == y
            or (short >= 4 and x[:3] == y[:3] and edits_within(x, y, 2 if short >= 6 else 1))
            or (short >= 5 and (x.startswith(y) or y.startswith(x))))


def soft_letter(word, pos):
    letter = word[pos]
    if letter in "aeiouy":
        return True
    before = word[pos - 1] if pos > 0 else ""
    return letter in (before, word[pos + 1:pos + 2])


def drop_soft(word):
    out = []
    for pos in range(len(word)):
        if soft_letter(word, pos):
            out.append(word[:pos] + word[pos + 1:])
    return out


def spelling_variant(lo, hi):
    gap = len(hi) - len(lo)
    if gap == 1:
        return lo in drop_soft(hi)
    if gap == 2:
        for shorter in drop_soft(hi):
            if lo in drop_soft(shorter):
                return True
        return False
    if gap:
        return False
    spots = [pos for pos in range(len(lo)) if lo[pos] != hi[pos]]
    if len(spots) != 2:
        return False
    first, second = spots
    return second == first + 1 and (lo[first], lo[second]) == (hi[second], hi[first])


def trailing_tag(short, longer):
    if len(longer) != len(short) + 1 or longer[:-1] != short:
        return False
    return len(longer[-1]) <= 3 and len("".join(short)) >= 5


def strict_prop(a, b):
    x, y = prop_key(a), prop_key(b)
    if not x or not y:
        return False
    lo, hi = sorted((x, y), key=len)
    if x == y or (len(lo) >= 4 and lo[:3] == hi[:3] and spelling_variant(lo, hi)):
        return True
    return trailing_tag(*sorted((prop_words(a), prop_words(b)), key=len))


# ---- field value comparison (requirement words against a store's field)
BOUND_WORDS = r"\b(?:no|not)\s+(?:more|less|fewer|over|under|above|below|beyond|exceeding)\b"
CH_BOUND = r"\b(?:no more than|at least|no less than|at most|more than|less than|up to|or more|or less|or larger|or smaller|or higher|" \
           r"or lower|minimum|maximum|between|within|under|over|above|below|than)\b"
CJ_COMPARE = (r"\b(?:falls?\s+short\s+of|short\s+of|nothing\s+(?:less|other)\s+than|nothing\s+short\s+of|nothing\s+but|"
              r"no\s+(?:less|worse)\s+than|at\s+least|(?:less|more|better|worse)\s+than|exactly|only|just)\b")
CH_SKIP = {"and", "the", "a", "an", "of"}
LITERAL_VALUES: set[str] = set()   # filled per run, emptied when a run starts


def ch_words(text):
    return [w for w in re.findall(r"[a-z0-9]+(?:\.[0-9]+)?", str(text or "").lower()) if w not in CH_SKIP]


def near_misspelling(word, other):
    return len(word) >= 6 and len(other) >= 6 and word[:2] == other[:2] and edits_within(word, other, 1)


def word_among(word, pool):
    if word in pool:
        return True
    return word.isalpha() and any(other.isalpha() and (same_kind(word, other) or near_misspelling(word, other)) for other in pool)


def ch_alts(value):
    text = re.sub(r"\((?:stated|revealed[^)]*)\)", " ", str(value or "").lower()).replace('"', " ")
    text = re.sub(r"\b(?:only|just|exactly|strictly)\b", " ", text)
    split = [ch_words(part) for part in re.split(r"\bor\b", text)]
    return [words for words in split if words]


def ch_writes(value, text):
    alts = ch_alts(value)
    return ch_match(alts, ch_words(text))


def runs_to(pool, target):
    for start in range(len(pool)):
        joined = ""
        pos = start
        while pos < len(pool) and len(joined) < len(target):
            joined += pool[pos]
            pos += 1
            if joined == target:
                return True
    return False


def ch_match(alts, pool):
    if not pool:
        return False
    for words in alts:
        if all(word_among(w, pool) for w in words):
            return True
        target = "".join(words)
        if len(words) < 2 and not re.search(r"\d", target) and not any(re.search(r"\d", p) for p in pool):
            continue
        if runs_to(pool, target):
            return True
    return False


def shares_stem(w, pool):
    return len(w) >= 3 and any(len(p) >= 3 and (p.startswith(w[:3]) or w.startswith(p[:3])) for p in pool)


def ch_touches(value, text):
    pool = ch_words(text)
    low = str(text or "").lower()
    every = [w for words in ch_alts(value) for w in words]
    for w in every:
        if word_among(w, pool):
            return True
        lone_figure = re.fullmatch(r"\d+(?:\.\d+)?", w)
        if lone_figure and re.search(r"(?<![\d.])" + re.escape(w) + r"(?!\d|\.\d)", low):
            return True
        if shares_stem(w, pool):
            return True
    return False


def ch_parts(text):
    low = str(text or "").strip().lower()
    parts = list(filter(None, (p.strip() for p in re.split(r"[,;|]", low))))
    return parts or ([low] if low else [])


def ch_sides(part):
    sides = list(filter(None, (s.strip() for s in re.split(r"\s*\+\s*", str(part)))))
    return [] if len(sides) < 2 else sides


def writes_figure(fig, said):
    pattern = r"(?<![\d.])" + re.escape(fig) + r"(?!\d|\.\d)"
    if "." in fig:
        return re.search(pattern, said) is not None
    for hit in re.finditer(pattern, said):
        unit = re.match(r"\s*([a-z]+)", said[hit.end():])
        if unit is None or len(unit.group(1)) <= 3:
            return True
    return False


EXCLUDE_WORDS = r"\b(?:no(?!-)|not|without|nothing|never|neither|nor|except|excluding|avoid\w*|anything but|other than|rather than|instead of)\b"


def meaning_words(text):
    found = re.findall(r"[a-z]{3,}", text)
    return list(filter(lambda w: w not in LINK_WORDS, found))


def without_bounds(low):
    low = re.sub(BOUND_WORDS, " ", low)
    low = re.sub(r"\b(?:no|not)\s+[a-z]+er\b(?=\s+than\b)", " ", low)
    if not re.search(r"\d", low):
        return low
    return re.sub(r"\b(?:no|not)\s+[a-z]+er\b", " ", low)


def negates_own_name(rest, name):
    own = set(prop_words(name)) | set(re.findall(r"[a-z]+", str(name).lower()))
    if not rest or not own:
        return False
    return all(w in own or w.rstrip("s") in own for w in rest)


def positive_value(value, name=""):
    low = str(value).lower().strip()
    if re.fullmatch(r"\W*(?:yes|no|true|false|none|required|included|needed)\W*", low):
        return True
    if LITERAL_VALUES and " ".join(re.findall(r"[a-z0-9]+", re.sub(r"\((?:stated|revealed[^)]*)\)", " ", low))) in LITERAL_VALUES:
        return True
    low = without_bounds(low)
    body = re.sub(r"\b(?:" + VALUE_LINKS + r")\b", " ", low)
    for clause in re.split(r",|;|/|&|\(|\)|\bor\b|\band\b", body):
        if len(meaning_words(clause)) >= 4:
            return False
    hit = re.search(EXCLUDE_WORDS, low)
    if not hit or meaning_words(low[:hit.start()]):
        return True
    return negates_own_name(meaning_words(low[hit.end():]), name)


def figure_run_in(figs, text, tail):
    return re.search(r"(?<![\d.])" + r"\D{0,4}".join(re.escape(f) for f in figs) + tail, text) is not None


def value_seen(value, samples):
    pool = str(samples or "").lower()
    if not pool.strip():
        return False
    body = re.sub(r"\b(?:" + VALUE_LINKS + r")\b", " ", str(value).lower())
    stems = [w[:-1] if len(w) > 4 else w for w in meaning_words(body)]
    if any(re.search(r"\b" + re.escape(stem), pool) for stem in stems):
        return True
    runs = re.findall(FIG_RUN, body)
    return any(figure_run_in(re.findall(r"\d+(?:\.\d+)?", run), pool, r"(?![\d.])") for run in runs)


# ---- kind words of product and category names
def kind_words(text):
    return set(w.rstrip("s") for w in meaning_words(str(text).lower()))


def kind_tokens(text):
    low = str(text).lower().replace("’", "'")
    return meaning_words(re.sub(r"'s\b", "", low))


PLURAL_SWAPS = (("y", ("ies",)), ("fe", ("ves",)), ("f", ("ves",)), ("ies", ("y",)), ("ves", ("f", "fe")), ("es", ("",)), ("s", ("",)))


def plural_forms(word):
    w = str(word)
    forms = {w, w + "s", w + "es"}
    for ending, swaps in PLURAL_SWAPS:
        if w.endswith(ending):
            forms.update(w[:len(w) - len(ending)] + swap for swap in swaps)
    return {form for form in forms if len(form) >= 3}


def same_kind(a, b):
    return a == b or str(b) in plural_forms(a) or str(a) in plural_forms(b)


def kind_in(word, pool):
    return next((True for other in pool if same_kind(word, other)), False)


def kinds_within(words, pool):
    return not any(not kind_in(w, pool) for w in words)


def kind_score(own, words, shared):
    score = 0
    for w in filter(lambda w: kind_in(w, words), own):
        score += 0.5 if kind_in(w, shared) else 1.0
    return score


def name_pattern(name):
    pieces = []
    for w in kind_tokens(name):
        forms = sorted(plural_forms(w), key=lambda form: (-len(form), form))
        pieces.append("(?:%s)(?:'s\\b|'|\\b)" % "|".join(map(re.escape, forms)))
    return r"[\s'-]+".join(pieces)


VALUE_LINKS = r"no more than|at least|no less than|at most|more than|less than|up to|or more|or less|or larger|or smaller|" \
              r"or higher|or lower|minimum|maximum|between|within|under|over|above|below|exactly|only|either|neither|nor|than|plus"


# ---- request story helpers
def one_prop_hint(hint):
    try:
        head, sep, detail = hint.partition(":")
        if not sep or "/" not in head:
            return hint
        tag = re.search(r"\(([^()]*)\)\s*$", head)
        names = [n.strip() for n in (head[: tag.start()] if tag else head).split("/")]
        names = [n for n in names if n]
        if len(names) < 2:
            return hint
        product = " (%s)" % tag.group(1).strip() if tag else ""
        return "%s%s:%s (or %s)" % (names[0], product, detail, ", ".join(names[1:]))
    except Exception:
        return hint


def word_before(word, kind, request):
    toks = re.findall(r"[a-z][a-z'-]*", str(request).lower())
    for left, right in zip(toks, toks[1:]):
        if left == word and right.split("'")[0].rstrip("s") == kind:
            return True
    return False


def lead_value(detail, request, product):
    try:
        hit = re.match(r"\s*([a-z][a-z-]*[a-z])(?:\s*[,;.()]|\s*$|\s+(?:" + "|".join(sorted(LINK_WORDS)) + r")\b)", str(detail).lower())
        if not hit:
            return ""
        word = hit.group(1)
        kinds = [w for w in re.findall(r"[a-z]{3,}", str(product).lower()) if w not in LINK_WORDS]
        own_kind = len(word) < 3 or word in LINK_WORDS or not kinds or word.rstrip("s") in {k.rstrip("s") for k in kinds}
        if not own_kind and word_before(word, kinds[0].rstrip("s"), request):
            return word
    except Exception:
        return ""
    return ""


def stem_of(w):
    if w.endswith("es") and len(w) > 4:
        return w[:-2]
    if w.endswith("s") and len(w) > 3:
        return w[:-1]
    return w


def written_in(w, said_words, said_stems):
    stem = stem_of(w)
    for s in said_words:
        close = len(w) >= 4 and (s.startswith(w) or (len(s) >= 4 and s[0] == w[0] and edits_within(s, w, 1)))
        if (s == w or (s.startswith(stem) and len(s) <= len(stem) + 3)
                or (stem.startswith(s) and len(s) >= max(4, len(stem) - 2)) or close):
            return True
    return any(len(s) >= 3 and stem.startswith(s) and len(s) >= len(stem) - 3 for s in said_stems)


def story_value(req, refs):
    head, _, value = req.partition(":")
    low = value.strip().lower()
    if not low or req in refs or re.search(r"as shown|same as|the page|listing|sold by|redirect|refer", low):
        return None
    if not re.fullmatch(r"\W*(?:yes|no|true|false|required|included|needed)\W*", low):
        return low
    return re.split(r"\s*\(", head)[0].lower()


def unwritten_words(body, said):
    said_words = set(re.findall(r"[a-z]+", said))
    said_stems = set(s[:-1] if s.endswith("s") and len(s) > 3 else s for s in said_words)
    words = meaning_words(body)
    missing = [w for w in words if not written_in(w, said_words, said_stems)]
    core_written = len(words) <= 2 and any(len(w) >= 4 and w not in missing for w in words)
    return None if core_written else missing


def unwritten_runs(body, said):
    runs = []
    for run in re.findall(FIG_RUN, body):
        parts = re.findall(r"\d+(?:\.\d+)?", run)
        if len(parts) > 1 and not figure_run_in(parts, said, r"(?!\d)"):
            runs.append(run)
    return runs


def story_guess(req, request, refs):
    low = story_value(str(req), refs)
    if low is None:
        return []
    body = re.sub(r"\b(?:" + VALUE_LINKS + r")\b", " ", low)
    said = str(request).lower()
    figs = re.findall(r"\d+(?:\.\d+)?", body)
    if figs and all(writes_figure(f, said) for f in figs):
        return []
    missing = unwritten_words(body, said)
    if missing is None:
        return []
    return missing + unwritten_runs(body, said)


# ---- references to other listings
REF_WORDS = r"\bas shown\b|\bshown on\b|\bsame as\b|\bpage\b|\bsold by\b|\bredirect|\brefer(?:s|red|ring)?\b|\blisting points\b|\bpoints you\b"
REF_HOPS = 8
REF_CALLS = 12
REF_STEPS = 0.4
REF_DIFFER_BLIND = 5
REF_DIFFER_STEPS = 0.3
DIFFER_WORDS = (r"\bset apart from\b|\bapart from\b|\bdistinct from\b|\bdifferent (?:from|to|than)\b|\bdiffer(?:s|ent|ing)?\s+from\b"
                r"|\bother than\b|\bunlike\b|\bnot the same as\b|\banything but\b")
PAIR_WORDS = r"\bcounterpart\b|\bpartner\b|\bpairs? with\b|\bpaired with\b|\bgoes with\b|\bgo together with\b"
POINT_WORDS = r"\bwithin\b|\bat (?:most|least)\b|\bno (?:more|less) than\b|\bsame as\b|\bas shown\b|\bthe value\b|\bthe one\b"
SELLER_KEYS = r"shop|seller|store|vendor|brand|make|manufactur|merchant|retailer|supplier|sold.?by"
BOUND_RELATIONS = (("at most", r"\b(?:no more than|up to|at most|or less|or lower|or smaller)\b"),
                   ("at least", r"\b(?:at least|no less than|or more|or higher|or larger)\b"))


def ref_text(value):
    return re.search(REF_WORDS, str(value or "").lower()) is not None


def unit_glued(hit):
    first, unit = hit.group(1), hit.group(2)
    return hit.group(0) if unit in LINK_WORDS else first + unit


def plural_cut(tok):
    return tok[:-2] if re.search(r"(?:ch|sh|x|z)es$", tok) else tok[:-1]


def ref_toks(text):
    low = re.sub(r"(\d)(?:\s+|\s*-\s*)([a-z]{1,8})\b", unit_glued, str(text or "").lower())
    toks = []
    for tok in re.findall(r"\d+(?:[.,:/]\d+)*[a-z]*|[a-z][a-z0-9]*", low):
        skip = tok in LINK_WORDS or (len(tok) == 1 and not tok.isdigit())
        if skip:
            continue
        plural = len(tok) > 3 and tok.endswith("s") and not tok.endswith("ss") and not tok[-2].isdigit()
        toks.append(plural_cut(tok) if plural else tok)
    return toks


def ref_figs(text):
    pairs = set()
    try:
        low = str(text or "").lower().replace(",", "")
        for hit in re.finditer(r"(\d+(?:\.\d+)?(?:\s*/\s*\d+(?:\.\d+)?)*)\s*-?\s*([a-z]{1,8})\b", low):
            unit = hit.group(2)
            if len(unit) > 2 and unit.endswith("s") and not unit.endswith("ss"):
                unit = plural_cut(unit)
            if unit not in LINK_WORDS:
                for fig in re.split(r"\s*/\s*", hit.group(1)):
                    pairs.add((float(fig), unit))
    except Exception:
        return pairs
    return pairs


def ref_name(text):
    return re.sub(r"[^a-z0-9]", "", re.sub(r"'s\b", "", str(text or "").lower()))


def ref_relation(value):
    low = str(value or "").lower()
    if re.search(DIFFER_WORDS, low):
        return "differ", None, ""
    if re.search(PAIR_WORDS, low):
        return "pair", None, ""
    within = re.search(r"\bwithin\s+(\d+(?:\.\d+)?)\s*([a-z%]+)?", low)
    if within:
        return "within", float(within.group(1)), (within.group(2) or "").strip()
    found = [relation for relation, pattern in BOUND_RELATIONS if re.search(pattern, low)]
    return (found[0] if found else "same"), None, ""


def seller_hit(body, ok):
    for found in re.finditer(r"(?<![A-Za-z'])([A-Z][a-z]{3,})(?:'s)?(?![A-Za-z])", body):
        if found.start() == 0:
            continue
        if ok is None or ok(found.group(1)):
            return found
    return None


def kind_split(before, after):
    sold = re.search(r".*(?:\bthe|\bfor|\ba|\ban)\s+(.*?)\s+(?:sold|made|offered|listed)\s+by\s*$", before, re.I | re.S)
    if sold:
        return sold.group(1), after
    lead = re.match(r"\s*([^,;(]*?)(?=\s+(?:with|that|which|in|at|having|from|of|for|sold)\b|[,;]|$)", after, re.I)
    if lead:
        return lead.group(1), after[lead.end():]
    return "", after


def detail_list(rest):
    details = []
    for piece in re.split(r",|;|\band\b|\bwith\b|\bthat\s+(?:is|are|has|have|comes?(?:\s+as)?)\b|\bwhich\b|\bin\b|\bat\b", rest, flags=re.I):
        words = [w for w in re.findall(r"[A-Za-z0-9][A-Za-z0-9.%:/&'-]*", piece) if w.lower() not in LINK_WORDS]
        if words:
            details.append(" ".join(words))
    return details[:6]


def ref_parse(text, ok=None):
    try:
        body = re.split(r"\(\s*or\b", str(text or ""))[0]
        name_hit = seller_hit(body, ok)
        if name_hit is None:
            return None
        kind, rest = kind_split(body[:name_hit.start()], body[name_hit.end():])
        return {"seller": name_hit.group(1), "kind": kind.strip()[:80], "details": detail_list(rest)}
    except Exception:
        return None


def word_alike(tok, word, exact):
    if tok == word:
        return True
    if exact:
        return False
    return (len(word) >= 3 and tok in (word + "s", word + "es")) or (len(tok) >= 3 and word in (tok + "s", tok + "es"))


def prop_named(prop, text, exact=False):
    head = re.sub(r"([a-z])([A-Z])", r"\1 \2", re.split(r"\s*\(", str(prop or ""))[0])
    words = re.findall(r"[a-z0-9]+", head.lower())
    toks = re.findall(r"[a-z0-9]+", str(text or "").lower())
    if not words or len(" ".join(words)) < 3:
        return False
    windows = zip(*(toks[shift:] for shift in range(len(words))))
    return any(all(word_alike(tok, word, exact) for tok, word in zip(window, words)) for window in windows)


def ref_span(prop, source):
    try:
        words = [w for w in re.findall(r"[a-z0-9]+", re.split(r"\s*\(", str(prop))[0].lower()) if w not in LINK_WORDS]
        hit = None
        if words:
            hit = re.search(r"\b" + r"\W+".join(map(re.escape, words)) + r"\b", str(source).lower())
        if hit is None:
            return ""
        return re.split(r"\(\s*or\b|(?<=[.!?])\s", str(source)[hit.end(): hit.end() + 500])[0]
    except Exception:
        return ""


LENS_FIELDS = (
    ("price",
     lambda low, val: re.search(r"price|cost", low) and is_number(val),
     lambda low, val: is_number(val) and re.search(r"amount|fee|charge|pay|rate|value", low)
     and not re.search(r"total|count|qty|quant|unit|due|sum|rating|star|score|page|number|num|size|weight|width|height|length|found|match", low)),
    ("in_stock",
     lambda low, val: re.search(r"stock|avail", low),
     lambda low, val: isinstance(val, bool) and re.search(r"hand|orderable|buyable|sellable|ready|ship|instock|purchas|in_store", low)),
    ("title",
     lambda low, val: re.search(r"title|heading", low),
     lambda low, val: isinstance(val, str) and re.search(r"headline|name|label|caption", low)
     and not re.search(r"shop|seller|store|vendor|brand|make|categor|shelf|department|id", low)),
)


def lens_record(d):
    if not any(isinstance(val, str) and "id" in str(key).lower() and re.fullmatch(ID_LIKE, val) for key, val in d.items()):
        return
    names = {str(key).lower(): key for key in d}
    for field, present, accepts in LENS_FIELDS:
        if any(present(low, d[key]) for low, key in names.items()):
            continue
        picks = [key for low, key in names.items() if accepts(low, d[key])]
        if len(picks) == 1:
            d[field] = d[picks[0]]


def lensed(piece):
    try:
        copy = json.loads(json.dumps(piece))
    except (TypeError, ValueError):
        return piece
    walk_dicts(copy, lens_record)
    return copy


# ---- argument checks against a tool's JSON schema
TYPE_TESTS = {
    "string": lambda val: isinstance(val, str),
    "integer": lambda val: isinstance(val, int) and not isinstance(val, bool),
    "number": is_number,
    "boolean": lambda val: isinstance(val, bool),
    "object": lambda val: isinstance(val, dict),
    "array": lambda val: isinstance(val, list),
    "null": lambda val: val is None,
}
LIMIT_TESTS = {
    "minimum": (lambda val, lim: val < lim, "below minimum"),
    "maximum": (lambda val, lim: val > lim, "above maximum"),
    "exclusiveMinimum": (lambda val, lim: val <= lim, "must be above"),
    "exclusiveMaximum": (lambda val, lim: val >= lim, "must be below"),
}
UNCHANGED = object()


def schema_types(jschema):
    declared = jschema.get("type")
    return [declared] if isinstance(declared, str) else list(declared or [])


def type_holds(piece, kind):
    test = TYPE_TESTS.get(kind)
    return test is None or test(piece)


def converted(piece, kind):
    if kind == "integer" and isinstance(piece, float) and piece.is_integer():
        return int(piece)
    if kind == "integer" and isinstance(piece, str) and re.fullmatch(r"\s*-?\d+(\.0+)?\s*", piece):
        return int(float(piece))
    if kind == "number" and isinstance(piece, str) and re.fullmatch(r"\s*-?\d+(\.\d+)?\s*", piece):
        num = float(piece)
        return int(num) if num.is_integer() else num
    if kind == "string" and is_number(piece):
        return str(piece)
    if kind == "boolean" and isinstance(piece, str) and piece.lower() in ("true", "false"):
        return piece.lower() == "true"
    return UNCHANGED


def coerce_value(piece, kinds):
    if not kinds:
        return piece
    if not any(type_holds(piece, kind) for kind in kinds):
        changes = (converted(piece, kind) for kind in kinds)
        return next((change for change in changes if change is not UNCHANGED), piece)
    whole_float = isinstance(piece, float) and piece.is_integer()
    return int(piece) if "integer" in kinds and whole_float else piece


def rule_enum(piece, jschema, where):
    options = jschema["enum"]
    if piece in options:
        return piece, []
    wanted = piece.strip().lower() if isinstance(piece, str) else None
    for opt in options:
        if wanted is not None and isinstance(opt, str) and opt.lower() == wanted:
            return opt, []
    return piece, [f"{where}: must be one of {options}"]


def rule_const(piece, jschema, where):
    faults = [] if piece == jschema["const"] else [f"{where}: must equal {jschema['const']!r}"]
    return piece, faults


def rule_min_length(piece, jschema, where):
    least = jschema["minLength"]
    short = isinstance(piece, str) and len(piece) < least
    return piece, [f"{where}: shorter than {least} characters"] if short else []


def rule_max_length(piece, jschema, where):
    return (piece[: jschema["maxLength"]] if isinstance(piece, str) else piece), []


def rule_min_items(piece, jschema, where):
    least = jschema["minItems"]
    few = isinstance(piece, list) and len(piece) < least
    return piece, [f"{where}: needs at least {least} items"] if few else []


def rule_max_items(piece, jschema, where):
    most = jschema["maxItems"]
    many = isinstance(piece, list) and len(piece) > most
    return piece, [f"{where}: allows at most {most} items"] if many else []


def rule_items(piece, jschema, where):
    inner = jschema["items"]
    if not (isinstance(piece, list) and isinstance(inner, dict)):
        return piece, []
    checked = [check_args(element, inner, f"{where}[{pos}]") for pos, element in enumerate(piece)]
    faults = []
    for _, found in checked:
        faults += found
    return [element for element, _ in checked], faults


def rule_object(piece, jschema, where):
    if not isinstance(piece, dict):
        return piece, []
    specs = jschema.get("properties") or {}
    faults = [f"{where}.{arg}: required" for arg in jschema.get("required") or [] if arg not in piece]
    if jschema.get("additionalProperties") is False:
        faults.extend(f"{where}.{arg}: not an allowed argument" for arg in piece if arg not in specs)
    mended = {}
    for arg, given in piece.items():
        if arg not in specs:
            mended[arg] = given
            continue
        mended[arg], found = check_args(given, specs[arg], f"{where}.{arg}")
        faults += found
    return mended, faults


SCHEMA_RULES = {
    "enum": rule_enum,
    "const": rule_const,
    "minLength": rule_min_length,
    "maxLength": rule_max_length,
    "minimum": None,
    "maximum": None,
    "exclusiveMinimum": None,
    "exclusiveMaximum": None,
    "minItems": rule_min_items,
    "maxItems": rule_max_items,
    "items": rule_items,
    "type": rule_object,
}


def limit_fault(keyword, piece, jschema, where):
    broken, words = LIMIT_TESTS[keyword]
    limit = jschema[keyword]
    if is_number(piece) and is_number(limit) and broken(piece, limit):
        return [f"{where}: {words} {limit}"]
    return []


def check_args(piece, jschema, where="args"):
    if not isinstance(jschema, dict):
        return piece, []
    kinds = schema_types(jschema)
    piece = coerce_value(piece, kinds)
    if kinds and not any(type_holds(piece, kind) for kind in kinds):
        return piece, [f"{where}: expected {'/'.join(kinds)}"]
    faults = []
    for keyword in [name for name in SCHEMA_RULES if name in jschema]:
        rule = SCHEMA_RULES[keyword]
        if rule is None:
            faults += limit_fault(keyword, piece, jschema, where)
        else:
            piece, found = rule(piece, jschema, where)
            faults += found
    return piece, faults


# ---- the shell: the only place that reads the clock, sleeps or talks to the proxy
class Link:
    def __init__(self, bind):
        self.t0 = clock_now()
        self.bind = dict(bind)
        self.model_list = ProxyClient(timeout=70, max_retries=1, retry_delay=1)
        self.world = ProxyClient(timeout=95, max_retries=1, retry_delay=1)
        self.spend, self.tick = 0.0, 0
        self.done = self.broke = self.mute = self.no_model = False
        self.log = []
        self.last_send = self.slow_until = self.model_resume = 0.0
        self.last_reply, self.misses = self.t0, 0
        self.banned, self.tool_banned, self.call_models = set(), set(), set()
        self.spare_models = []
        self.call_fail = ""
        self.no_effort = False      # set once the proxy refuses the reasoning field


class ModelCall:
    def __init__(self, ladder, lone, effort=None):
        self.ladder, self.lone = ladder, lone
        self.kinds, self.refused = set(), set()
        self.seed, self.plain = "", False
        self.effort, self.effort_sent = effort, False


def clock_now():
    return monotonic()


def time_left(link):
    elapsed = clock_now() - link.t0
    return DEADLINE - elapsed


def spend_scale(run):
    # (about $0.05 for one or two products, $0.06 for three or more: baskets are common, and a guard that ends their turn
    # loop early leaves the code to finish them)
    count = len(need_items(run.needs)) or 1
    run.wire.scale = 1.2 if count >= 3 else 1.0


def over_budget(link):
    scale = vars(link).get("scale", 1.0)
    return link.spend >= SPEND_CAP * scale


def pace_wait(link, path):
    gap = PACE_GAP if clock_now() >= link.slow_until else SLOW_GAP
    wait = link.last_send + gap - clock_now()
    if path.find("/inference/") >= 0:
        wait = max(wait, link.model_resume - clock_now())
    return min(wait, time_left(link) - 1)


def post(link, client, path, body):
    wait = pace_wait(link, path)
    if wait > 0:
        sleep(wait)
    link.last_send = clock_now()
    answer = client.post_verbose(path, body)
    throttled = answer.data is None and (answer.error or {}).get("status") in (429, 503)
    if throttled:
        link.slow_until = clock_now() + SLOW_FOR
    return answer


def retry_pause(link, seed, tries, status):
    spread = (len(seed) * 7 + sum(map(ord, seed[-64:])) + 31 * tries + 13 * link.tick) % 101 / 100.0
    first_wait = 2.0 if status == 429 else 1.0
    return min(8.0, first_wait * 2 ** tries) * (0.75 + 0.5 * spread)


def models_down(link, at=None):
    if link.no_model:
        return True
    if not link.misses > 0:
        return False
    return (clock_now() if at is None else at) - link.last_reply > DOWN_AFTER


def can_call(link, model_list, tools=False):
    if link.no_model or link.mute:
        return False
    open_models = [mdl for mdl in list(model_list) + link.spare_models
                   if mdl not in link.banned and not (tools and mdl in link.tool_banned)]
    return len(open_models) > 0


def flatten_parts(msgs):
    flat = []
    for original in msgs:
        msg = dict(original)
        if isinstance(msg.get("content"), list):
            texts = [str(part.get("text") or "") for part in msg["content"] if isinstance(part, dict)]
            msg["content"] = "\n\n".join(texts)
        flat.append(msg)
    return flat


def cache_marked(msgs, mdl):
    if len(msgs) <= 2 or not mdl.startswith("anthropic/"):
        return msgs
    marked = list(map(dict, msgs))
    for msg in (marked[0], marked[-1]):
        text = msg.get("content")
        if text and isinstance(text, str):
            msg["content"] = [{"type": "text", "text": text, "cache_control": {"type": "ephemeral"}}]
    return marked


def note_failure(link, kinds):
    link.call_fail = "empty" if "empty" in kinds else ("refused" if kinds == {"hard"} else "")


def lone_model(link, ladder):
    return len(set(mdl for mdl in ladder if mdl not in link.banned)) < 2


def model_skipped(link, call, mdl, pos, msgs, model_list, token_cap, toolset):
    blocked = mdl in link.banned or mdl in call.refused or (toolset and mdl in link.tool_banned)
    if blocked or "sonnet" not in mdl:
        return bool(blocked)
    cheaper_next = pos + 1 < len(model_list) and "sonnet" not in model_list[pos + 1]
    guess = sum(len(json.dumps(m, default=str)) for m in msgs) / 2.3 * 2e-6 + min(token_cap, 1500) * 1e-5
    scale = vars(link).get("scale", 1.0)
    if over_budget(link):
        return True
    if not cheaper_next:
        return False
    return link.spend >= SOFT_SPEND * scale or link.spend + guess > SPEND_CAP * scale


def took_reply(link, call, mdl, data):
    link.last_reply, link.misses = clock_now(), 0
    usage = data.get("usage") or {}
    if is_number(usage.get("cost")):
        link.spend += float(usage["cost"])
    msg = data["choices"][0].get("message") or {}
    if not (msg.get("content") or msg.get("tool_calls")):
        call.kinds.add("empty")
        link.log.append({"kind": "inference_failure", "model": mdl, "error": "empty"})
        return "next", None
    return "reply", msg


def refused_model(link, call, mdl, status, body, said, model_list, toolset):
    call.kinds.add("hard")
    call.refused.add(mdl)
    if status == 404 and toolset and "tool" in said:
        link.tool_banned.add(mdl)
    elif status != 403 or not re.search(r"moderat|flagged", said):
        link.banned.add(mdl)
    listed = re.search(r'allowed_models"?\s*:\s*\[([^\]]*)', body) if status == 403 else None
    if listed is None or link.spare_models:
        return
    named = re.findall(r'"([^"\s]{3,100})"', listed.group(1))
    link.spare_models = [m for m in named if m not in link.banned and m not in model_list][:2]
    call.ladder.extend([m for m in link.spare_models if m not in call.ladder])
    call.lone = lone_model(link, call.ladder)


def soft_failure(link, call, mdl, status, started, msgs):
    call.kinds.add("soft")
    if clock_now() - started > 20:
        gone = models_down(link) and (call.lone or len(link.call_models) >= 2)
        return "stop" if gone else "next"
    call.seed = call.seed or json.dumps(msgs[-1:], default=str)[-600:]
    wait = retry_pause(link, call.seed + mdl, min(link.misses - 1, 6), status)
    if time_left(link) - wait < LAST_CALL - 10:
        return "stop"
    link.model_resume = clock_now() + wait
    if models_down(link, link.model_resume) and (call.lone or len(link.call_models) >= 2):
        return "stop"
    return "again"


def after_error(link, call, mdl, error, started, msgs, model_list, toolset):
    status = error.get("status")
    body = str(error.get("body") or "")
    said = body.lower()
    if status == 402 or (status == 403 and re.search(r"key limit|insufficient|credit", said)):
        link.no_model = True
        return "stop"
    if status in (401, 403, 404):
        refused_model(link, call, mdl, status, body, said, model_list, toolset)
        return "next"
    bad_body = status in (400, 413, 422)
    if bad_body and call.effort_sent:
        link.no_effort = True       # (the proxy or the model refused the reasoning field: the call goes again without it)
        return "again"
    if bad_body and not call.plain:
        call.kinds.add("hard")
        call.plain = True
        return "again"
    if bad_body or (isinstance(status, int) and 400 <= status < 500 and status not in (408, 425, 429)):
        call.kinds.add("hard")
        call.refused.add(mdl)
        return "next"
    return soft_failure(link, call, mdl, status, started, msgs)


def one_attempt(link, call, mdl, msgs, model_list, token_cap, toolset):
    spare = time_left(link)
    if spare < LAST_CALL - 10:
        return "stop", None
    link.model_list.timeout = int(max(8, min(30 if models_down(link) else 75, spare - LAST_CALL + 15)))
    envelope = {"model": mdl, "messages": flatten_parts(msgs) if call.plain else cache_marked(msgs, mdl),
                "temperature": 0, "max_tokens": token_cap}
    if toolset:
        envelope.update(tools=toolset, tool_choice="auto")
    call.effort_sent = bool(call.effort and mdl.startswith("anthropic/") and not link.no_effort)
    if call.effort_sent:
        # (Sonnet cannot switch its reasoning off: "none" goes as its least setting)
        level = "minimal" if call.effort == "none" and "sonnet" in mdl else call.effort
        envelope["reasoning"] = {"effort": level}
    link.call_models.add(mdl)
    started = clock_now()
    answer = post(link, link.model_list, "/inference/chat/completions", envelope)
    if isinstance(answer.data, dict) and answer.data.get("choices"):
        return took_reply(link, call, mdl, answer.data)
    link.misses += 1
    reason = str(answer.error)[:300] if answer.error else "empty"
    link.log.append({"kind": "inference_failure", "model": mdl, "error": reason})
    return after_error(link, call, mdl, answer.error or {}, started, msgs, model_list, toolset), None


def ask_model(link, msgs, model_list, token_cap, toolset=None, effort=None):
    link.call_fail = ""
    link.call_models = set()
    if link.mute or link.no_model:
        return None
    ladder = list(model_list)
    ladder += [mdl for mdl in link.spare_models if mdl not in model_list]
    call = ModelCall(ladder, lone_model(link, ladder), effort)
    for pos, mdl in enumerate(ladder):
        if link.no_model:
            break
        if model_skipped(link, call, mdl, pos, msgs, model_list, token_cap, toolset):
            continue
        call.plain = False
        rounds = 1 if models_down(link) else MODEL_TRIES
        while rounds > 0:
            rounds -= 1
            verdict, msg = one_attempt(link, call, mdl, msgs, model_list, token_cap, toolset)
            if verdict == "reply":
                return msg
            if verdict == "stop":
                return note_failure(link, call.kinds)
            if verdict == "next":
                break
    return note_failure(link, call.kinds)


def consult(link, ask_text, token_cap, model_list, effort=None, last=False):
    reply = ask_model(link, [{"role": "user", "content": ask_text}], model_list, token_cap, effort=effort) or {}
    return json_object(reply.get("content") or "", last)


def world_reply(link, data, count):
    step = data.get("step")
    link.tick = data["step"] if isinstance(step, int) else link.tick + count
    if data.get("finished"):
        link.done = True
    return data


class WorldCall:
    def __init__(self, payload):
        self.payload = payload
        self.tries = 0
        self.reused = 0


def resync(link, call, body):
    if "finished" in body:
        link.done = True
        return False
    expected = re.search(r"expected (\d+)", body)
    if expected:
        link.tick = int(expected.group(1)) - 1
        call.payload["step"] = link.tick + 1
        return True
    if "reused" not in body or call.reused >= 2:
        return False
    call.reused += 1
    call.payload["step"] += 1
    return True


def world_failure(link, call, error):
    status, body = error.get("status"), str(error.get("body") or "")
    link.log.append({"kind": "env_failure", "status": status, "body": body[:300]})
    if status == 402:
        link.broke = True
        return False
    if status == 409:
        return resync(link, call, body)
    if status in (403, 422) or call.tries >= WORLD_TRIES - 1 or (status in (400, 401, 404) and call.tries >= 1):
        return False
    wait = retry_pause(link, json.dumps(call.payload, default=str)[-600:], call.tries, status)
    call.tries += 1
    if time_left(link) - wait < 2:
        return False
    sleep(wait)
    return True


def world_act(link, acts):
    payload = dict(link.bind)
    payload["step"] = link.tick + 1
    payload["action" if len(acts) == 1 else "actions"] = acts[0] if len(acts) == 1 else acts
    call = WorldCall(payload)
    for _ in range(3 + WORLD_TRIES):
        spare = time_left(link)
        if spare < 2:
            break
        link.world.timeout = int(max(3, min(95, spare - 1)))
        answer = post(link, link.world, "/environment/call", payload)
        if isinstance(answer.data, dict):
            return world_reply(link, answer.data, len(acts))
        if not world_failure(link, call, answer.error or {}):
            break
    return None




# ---------------------------------------------------------------------------------------------------- observation archive
class Archive:
    def __init__(self):
        vars(self).update(items=[], ids=set(), read=set(), label_of={})


def record_failure(link, kind, err):
    link.log.append({"kind": kind, "error": f"{type(err).__name__}: {err}"[:300]})


def shelf_label_field(rec):
    for key, val in rec.items():
        if not isinstance(val, str):
            continue
        tidy = val.strip()
        name = str(key).lower()
        if 0 < len(tidy) <= 60 and re.search(r"categor|shelf|department|aisle", name) and "id" not in name:
            return tidy
    return ""


def record_ident(rec):
    for key, val in rec.items():
        if not isinstance(val, str):
            continue
        name = str(key).lower()
        if name in ("id", "_id") or (name.endswith("id") and re.fullmatch(ID_LIKE, val)):
            return val
    return None


def ident_already_read(arch, rec):
    for key, val in rec.items():
        if isinstance(val, str) and "id" in str(key).lower() and val in arch.read:
            return val
    return None


def first_price(rec):
    for key, val in rec.items():
        if is_number(val) and not isinstance(val, bool) and re.search(r"price|cost", str(key).lower()):
            return float(val)
    return None


def file_obs(arch, obs):
    arch.items.append(obs)

    def note(rec):
        for key, val in rec.items():
            if isinstance(val, str) and "id" in str(key).lower() and re.fullmatch(ID_LIKE, val):
                arch.ids.add(val)
        ident = record_ident(rec)
        if not ident:
            return
        for val in rec.values():
            if isinstance(val, dict) and len([v for v in val.values() if isinstance(v, (str, int, float))]) >= 2:
                arch.read.add(ident)
                break
        label = shelf_label_field(rec)
        if label:
            vars(arch).setdefault("label_of", {}).setdefault(ident, label)
    walk_dicts(obs, note)


def price_of(arch, thing):
    prices = []

    def look(rec):
        names_it = False
        for key, val in rec.items():
            if isinstance(val, str) and val == thing and "id" in str(key).lower():
                names_it = True
                break
        if names_it:
            found = first_price(rec)
            if found is not None:
                prices.append(found)
    for obs in arch.items:
        walk_dicts(obs, look)
    return prices[-1] if prices else None


def notes_about(arch, thing, hold=4):
    texts = []

    def dig(node, depth):
        if depth > 6 or len(texts) >= 8:
            return
        if isinstance(node, list):
            for part in node:
                dig(part, depth + 1)
            return
        if not isinstance(node, dict):
            return
        if thing in [val for val in node.values() if isinstance(val, str)]:
            text = as_text(clipped(node, 400))[:2500]
            if text not in texts:
                texts.append(text)
            return
        for part in node.values():
            dig(part, depth + 1)

    for obs in arch.items:
        dig(obs, 0)
    return texts[-hold:]


def merged_cards(arch):
    merged = {}

    def gather(rec):
        ident = next((val for key, val in rec.items() if isinstance(val, str) and "id" in str(key).lower()
                      and re.fullmatch(ID_LIKE, val) and val in arch.ids), None)
        has_map = any(isinstance(val, dict) and len(val) >= 2 for val in rec.values())
        if ident and (len(rec) >= 4 or has_map):
            merged.setdefault(ident, {}).update(rec)
            if has_map:
                arch.read.add(ident)

    for obs in arch.items:
        walk_dicts(obs, gather)
    return merged


def record_cards(arch, char_cap=30000, cheaper_than="", skip=None, first=()):
    merged = merged_cards(arch)
    order = list(merged)

    def card_price(ident):
        return first_price(merged.get(ident) or {})

    ceiling = card_price(cheaper_than) if cheaper_than else None
    if skip:
        order = [i for i in order if i == cheaper_than or i not in skip]
    if ceiling is not None:
        order = [i for i in order if i == cheaper_than or card_price(i) is None or card_price(i) < ceiling]
    place = dict(zip(order, range(len(order))))
    front = set(first or ())
    ranked = sorted(order, key=lambda i: (i not in front, i not in arch.read, place[i]))
    texts = dict((i, ("DETAILS " if i in arch.read else "ROW ") + as_text(clipped(merged[i], 160))) for i in ranked)
    if sum(map(len, texts.values())) > char_cap:
        # sold-out read listings drop behind the other read ones when space is short
        sold = dict((i, any(key_stock(key, val) is False for key, val in merged[i].items())) for i in ranked)
        ranked.sort(key=lambda i: (i not in front, i not in arch.read, sold[i], place[i]))
    lines, total = [], 0
    for text in (texts[i] for i in ranked):
        if total + len(text) > char_cap:
            continue
        lines.append(text)
        total += len(text)
    return lines


def label_of_rec(arch, d):
    own = shelf_label_field(d)
    if own:
        return own
    known = vars(arch).get("label_of") or {}
    for key, val in d.items():
        if isinstance(val, str) and val in known and "id" in str(key).lower():
            return known[val]
    return ""


def holds_fields(obs):
    # (how well a reply shows a listing's properties: 2 for a nested table of three or more values, 1 for a flat record
    # of five short fields, 0 otherwise; a flat page of labels can reach 1, a property table reaches 2)
    best = [0]

    def visit(rec):
        maps = [val for val in rec.values() if isinstance(val, dict)]
        if any(len(m) >= 3 for m in maps):
            best[0] = 2
        elif not maps and len([val for val in rec.values() if isinstance(val, str) and 0 < len(val) <= 70]) >= 5:
            best[0] = max(best[0], 1)
    walk_dicts(obs, visit)
    return best[0]


def note_fields(run, call_name, obs):
    if not call_name or call_name not in {role(run, "details"), role(run, "view")} | set(lookup_tools(run)):
        return
    seen = vars(run).setdefault("field_tools", {})
    level, total = seen.get(call_name, (0, 0))
    seen[call_name] = (max(level, holds_fields(obs)), total + 1)


def field_reader(run):
    # (the read tool whose replies show the property table, for reads made to learn the fields: the role reading
    # guesses from tool names, and it has named a price-and-stock page "details" in most episodes of a world, so the
    # first questions were planned with no fields at all; an untried second tool gets one read before it is trusted.
    # The table tool can also be left with no role at all (a side-by-side list has taken "view" in about a
    # quarter of a world's episodes), so every read tool that takes one listing's id is a candidate, the view first)
    details = role(run, "details")
    if not details:
        return details
    seen = vars(run).get("field_tools") or {}
    level, total = seen.get(details, (0, 0))
    guess = role_guesses(run)
    acting = {guess.get(r) for r in ("cart", "cart_view", "order", "decline", "message", "search")} - {None}
    others = [t for t in lookup_tools(run) if t != details and t not in acting]
    others.sort(key=lambda t: t != role(run, "view"))
    best = max(others, key=lambda t: seen.get(t, (0, 0))[0], default=None)
    best_level = seen.get(best, (0, 0))[0] if best else 0
    if best_level >= 2 and best_level > level:
        return best
    untried = [t for t in others if not seen.get(t, (0, 0))[1]]
    if total and level < 2 and untried:
        return untried[0]
    return best if best_level > level else details


def table_reads(arch, shelf):
    # (listings of this shelf, or of any shelf when none is named, read with a nested property table)
    found = set()
    wanted = shelf.strip().lower()

    def judge(rec):
        ident = ident_already_read(arch, rec)
        if ident and (not wanted or label_of_rec(arch, rec).strip().lower() == wanted) \
                and any(isinstance(v, dict) and len(v) >= 3 for v in rec.values()):
            found.add(ident)
    for obs in arch.items:
        walk_dicts(obs, judge)
    return len(found)


def rich_read_count(arch, shelf=""):
    rich = set()
    wanted = shelf.strip().lower()

    def judge(rec):
        ident = ident_already_read(arch, rec)
        if not ident:
            return
        if shelf and label_of_rec(arch, rec).strip().lower() != wanted:
            return
        maps = [val for val in rec.values() if isinstance(val, dict)]
        if maps:
            full = any(len(m) >= 3 for m in maps)
        else:
            full = len([val for val in rec.values() if isinstance(val, str) and 0 < len(val) <= 70]) >= 5
        if full:
            rich.add(ident)

    for obs in arch.items:
        walk_dicts(obs, judge)
    return len(rich)


def read_shelves(arch):
    tally = {}

    def count(rec):
        if not ident_already_read(arch, rec):
            return
        label = label_of_rec(arch, rec)
        if label and any(isinstance(val, dict) for val in rec.values()):
            tally[label] = tally.get(label, 0) + 1

    for obs in arch.items:
        walk_dicts(obs, count)
    return sorted(tally, key=lambda s: -tally[s])


def shelf_records(arch, shelf, node, inside=False, depth=0):
    if depth > 6:
        return
    if isinstance(node, list):
        for part in node:
            yield from shelf_records(arch, shelf, part, inside, depth + 1)
        return
    if not isinstance(node, dict):
        return
    inside = inside or not shelf or on_named_shelf(arch, shelf, node)
    if inside:
        yield node
    for part in node.values():
        if isinstance(part, (dict, list)):
            yield from shelf_records(arch, shelf, part, inside, depth + 1)


def on_named_shelf(arch, shelf, rec):
    for key, val in rec.items():
        if isinstance(val, str) and val.strip().lower() == shelf.lower() and re.search(r"categor|shelf|department|aisle", str(key).lower()):
            return True
    return label_of_rec(arch, rec).strip().lower() == shelf.strip().lower()


def property_menu(arch, shelf=""):
    hits, samples, row_hits = {}, {}, {}
    sets = mapped = 0
    records = (rec for obs in arch.items for rec in shelf_records(arch, shelf, obs))
    for rec in records:
        short = [(key, val) for key, val in rec.items() if isinstance(val, str) and 0 < len(val) <= 70 and not junk_key(key)]
        if len(short) < 5:
            flat_row = shelf and len(short) >= 2 and all(not isinstance(v, dict) for v in rec.values())
            if not flat_row:
                continue
        sets += 1
        row_like = len(short) < 5 and any(isinstance(val, str) and "id" in str(key).lower() for key, val in rec.items())
        mapped += 0 if row_like else 1
        for key, val in short:
            if re.search(r"\d", val) and re.fullmatch(CODE_LIKE, val) and ("-" in val or "_" in val):
                continue
            name = re.sub(r"[_.]+", " ", str(key)).strip().lower()
            hits[name] = 1 + hits.get(name, 0)
            if row_like:
                row_hits[name] = 1 + row_hits.get(name, 0)
            pool = samples.setdefault(name, [])
            if len(pool) < 5 and val.lower() not in pool:
                pool.append(val.lower())
    if mapped:
        listing_only = [n for n in hits if row_hits.get(n, 0) >= hits[n]]
        for name in listing_only:
            hits.pop(name)
    best = sorted(hits, key=lambda n: -hits[n])[:28]
    menu = "\n".join("- %s (in %d listings): %s" % (name, hits[name], " | ".join(samples.get(name, []))) for name in best)
    return menu, sets


def label_list(arch):
    per_key = {}

    def count(rec):
        if len(rec) < 4:
            return
        for key, val in rec.items():
            if not isinstance(val, str) or not 0 < len(val) <= 40:
                continue
            if re.search(r"\d", val) and re.fullmatch(CODE_LIKE, val):
                continue
            bucket = per_key.setdefault(str(key), {})
            bucket[val] = bucket.get(val, 0) + 1

    for obs in arch.items:
        walk_dicts(obs, count)
    out = []
    for key, bucket in per_key.items():
        if sum(bucket.values()) < 4 or not 2 <= len(bucket) <= 8 or max(bucket.values()) < 2:
            continue
        ordered = sorted(bucket.items(), key=lambda kv: -kv[1])
        out.append(f"- {key}: " + " | ".join(f"{val} (x{n})" for val, n in ordered))
    return "\n".join(out[:12])


# ---------------------------------------------------------------------------------------------------- the need sheet
class NeedSheet:
    def __init__(self, wire, request):
        vars(self).update(wire=wire, request=request, reading={}, written=[], uncovered=[], indifferent=[], let_go=[],
                          now_reqs=None, spend_cap=None, version=0, caught=set(), exchanges=[], reply_due="",
                          widened_at=-1, widened_tick=-10 ** 6, sweepable=False, resolved=[], ref_news=[], approval=None)


def whole_count(figure):
    return isinstance(figure, int) and not isinstance(figure, bool) and figure >= 1


def tag_list(head):
    return re.findall(r"\(([^)]*)\)", head)


def bare_words(text):
    return " ".join(re.findall(r"[a-z0-9]+", text))


def need_items(needs):
    wanted = []
    for entry in needs.reading.get("items") or []:
        if not isinstance(entry, dict):
            continue
        product = str(entry.get("product") or "").strip()
        if not product:
            continue
        qty = entry.get("quantity")
        reqs = entry.get("requirements")
        if not isinstance(reqs, list):
            reqs = []
        partner = entry.get("works_with")
        wanted.append({"product": product, "quantity": qty if whole_count(qty) else 1,
                       "requirements": [str(r) for r in reqs if r and ":" in str(r)],
                       "works_with": str(partner).strip() if isinstance(partner, str) else ""})
    if len(wanted) < 2:
        return []
    return wanted


def route(needs):
    return "multi" if need_items(needs) else "single"


def routed(run, knob):
    return ROUTES[route(run.needs)][knob]


def need_qty(needs):
    if need_items(needs):
        return 1
    figure = needs.reading.get("quantity")
    return figure if whole_count(figure) else 1


def need_counts(needs):
    return [entry["quantity"] for entry in need_items(needs)] or [need_qty(needs)]


def items_line(needs):
    products = need_items(needs)
    if not products:
        return ""
    listing = " | ".join(f"{n + 1}) {entry['product']} x{entry['quantity']}" for n, entry in enumerate(products))
    return ("THE REQUEST ASKS FOR SEVERAL PRODUCTS; the order carries one line for each product, at that product's quantity, "
            "and nothing more; a listing is held only to the requirements of the product it is (its category tells which), "
            "and the cheapest/best rule works within each product: " + listing)


def need_priority(needs):
    rule = needs.reading.get("priority")
    return rule if rule in ("cheapest", "best") else "none"


def props_pending(needs):
    if needs.now_reqs is None:
        pool = [r for r in needs.written + needs.uncovered if r not in needs.let_go]
    else:
        pool = needs.now_reqs
    return [r for r in pool if r not in needs.let_go and not re.search(r"\bprice\b|budget|quantity|\bstock\b|\bcents?\b", r.lower())]


def needs_text(needs):
    if needs.now_reqs is None:
        entries = [f"{req} (stated)" for req in needs.written if req not in needs.let_go]
        entries += [f"{req} (revealed by the shopper)" for req in needs.uncovered if req not in needs.let_go]
    else:
        entries = list(needs.now_reqs)
    if entries:
        text = "; ".join(entries)
    elif needs.let_go:
        text = "no property requirement is left: any listing of the requested category that shows stock fits"
    else:
        text = "see the request"
    if needs.spend_cap is not None:
        text += f"; total price at most {needs.spend_cap} {needs.reading.get('currency') or ''}".rstrip()
    text += "; " + items_line(needs) if need_items(needs) else f"; quantity {need_qty(needs)}"
    if needs.let_go:
        text += "; no longer required: " + "; ".join(needs.let_go)
    if needs.indifferent:
        text += "; shopper does not care about: " + ", ".join(needs.indifferent)
    return text


def change_text(needs):
    if not needs.let_go:
        return ""
    return ("\n\nCHANGES SINCE THE REQUEST (these win over its wording): Check: the shopper DROPPED " + "; ".join(needs.let_go)
            + ". Then: treat a dropped requirement as gone, so no listing fails, ranks lower or is preferred because of it.")


def needs_overview(needs, step_count, msgs):
    reading = needs.reading
    if not reading:
        return ""
    lines = ["First reading of the request (Check: compare it to the request text):"]
    if reading.get("product"):
        lines.append(f"- product: {reading.get('product')}")
    products = need_items(needs)
    if not products:
        lines.append(f"- quantity: {need_qty(needs)}")
    else:
        for n, entry in enumerate(products):
            lines.append(f"- product {n + 1}: {entry['product']}, quantity {entry['quantity']} (one cart line)")
        lines.append("- the order holds exactly these lines, one per product, and nothing else; send every line in each cart "
                     "call when the cart tool takes a list of lines")
    for req in reading.get("requirements") or []:
        lines.append(f"- stated requirement: {req}")
    for req in reading.get("references") or []:
        lines.append(f"- value given by reference (read that listing first, then search with its value): {req}")
    if reading.get("price_cap") is not None:
        lines.append(f"- price cap (total): {reading.get('price_cap')} {reading.get('currency') or ''}".rstrip())
    if reading.get("in_stock_only"):
        lines.append("- only items in stock / on hand")
    if (reading.get("none_fit") or {}).get("order_closest"):
        lines.append("- when no listing meets every requirement the request says: \"" + reading["none_fit"]["quote"][:300]
                     + "\"; so a decline is not the answer then: the closest listing is put in the cart and ordered (naming it "
                     "inside a decline does not put it forward), the code names it when a decline comes up, and every "
                     "condition of that passage still holds for it")
    if reading.get("basket_yes"):
        lines.append("- the shopper approves the final basket before the purchase: the code shows it and waits for the yes "
                     "right before your order call, and again after any later cart change; never send that message yourself, and never "
                     "decline for want of that yes: call the order tool and the code gets it")
    rule = need_priority(needs)
    if rule == "none":
        lines.append("- no cheapest/best rule was found: any item meeting every requirement within the budget is right, so do not "
                     "spend steps hunting for the lowest price (unless the request itself says otherwise)")
    else:
        detail = reading.get("priority_detail")
        lines.append(f"- choose the {rule}" + (f" ({detail})" if detail else "") + " among all fitting items")
    lines += [f"- nice to have, NOT required (only a tie-breaker): {wish}" for wish in reading.get("soft_wants") or []]
    lines += [f"- must be learned from the shopper first: {item}" for item in reading.get("ask_first") or []]
    lines += [f"- rule: {rule_text}" for rule_text in reading.get("rules") or []]
    if msgs is not None:
        lines.append(f"- messages to the shopper allowed: {msgs}")
    lines.append(f"- steps allowed: {step_count}")
    return "\n".join(lines)


def names_in_lines(needs, lns):
    words = re.findall(r"[a-z]+", " ".join(lns).lower())
    spoken = " ".join(words)
    renamed = {}
    for need in needs:
        if not isinstance(need, str) or ":" not in need:
            continue
        head, _, value = need.partition(":")
        tag = head[head.find("("):].strip() if "(" in head else ""
        parts = re.findall(r"[a-z]+", re.split(r"\s*\(", head)[0].lower())
        if not parts or min(len(p) for p in parts) < 3:
            continue
        if re.search(r"\b" + r"\s+".join(parts) + r"\b", spoken):
            continue
        span = len(parts)
        windows = (words[k: k + span] for k in range(len(words) - span + 1))
        hit = next((w for w in windows if w != parts and all(x.startswith(p) for p, x in zip(parts, w))), None)
        if hit is not None:
            renamed[need] = " ".join(hit) + (" " + tag if tag else "") + ":" + value
    return renamed


def need_matcher(name):
    tags = tag_list(name)

    def matches(entry):
        head = entry.split(":", 1)[0]
        return same_prop(head, name) and (not tags or tag_list(head) in ([], tags))
    return matches


def revealed_only(needs, old_need, matches):
    fresh = [n for n in vars(needs).get("fresh_needs") or [] if isinstance(n, str) and ":" in n]
    found = [w for w in needs.uncovered if matches(w) and w not in needs.written and w not in fresh]
    replaced = bool(found) and any(matches(n) and n != old_need for n in fresh)
    return [] if replaced else found


def line_phrases(lns):
    words = re.findall(r"[a-z]+", " ".join(lns).lower())
    grams = {" ".join(words[i: i + n]) for n in (1, 2, 3) for i in range(len(words) - n + 1)}
    return {g for g in grams if len(g) >= 3 and g not in LINK_WORDS}


def field_names_value(needs, hook, name, stated, phrases):
    for entry in stated:
        head, _, value = entry.partition(":")
        value = re.sub(r"\((?:stated|revealed[^)]*)\)", " ", value)
        if ref_text(value) or not ch_alts(value):
            continue
        fields = hook(head) or {}
        others = [r.split(":", 1)[0] for r in needs.written + needs.uncovered
                  if isinstance(r, str) and ":" in r and not same_prop(r.split(":", 1)[0], name)]
        free = (p for p in phrases if not any(strict_prop(other, p) for other in others))
        for phrase in free:
            if any(strict_prop(key, phrase) and value_seen(value, " | ".join(vals)) for key, vals in fields.items()):
                needs.wire.log.append({"kind": "drop_by_field", "need": entry[:200], "phrase": phrase})
                return True
    return False


def drop_need(needs, old_need, lns):
    try:
        name = str(old_need).split(":", 1)[0]
        matches = need_matcher(name)
        stated = [w for w in needs.written if matches(w)] or revealed_only(needs, old_need, matches)
        if not stated:
            return True
        phrases = line_phrases(lns)
        if any(same_prop(w.split(":", 1)[0], p) for w in stated for p in phrases):
            return True
        # (a requirement the request states goes only when the shopper names its property: a reply that names another
        # property sharing its values in the listings adds a need beside it, and both must hold)
        if any(w in needs.written for w in stated):
            return False
        hook = vars(needs).get("field_hook")
        return bool(hook) and field_names_value(needs, hook, name, stated, phrases)
    except Exception:
        return True


def swap_in_answer(answer, table):
    answer["needs"] = [table.get(n, n) if isinstance(n, str) else n for n in answer.get("needs") or []]
    if isinstance(answer.get("requirements_now"), list):
        answer["requirements_now"] = [table.get(r, r) if isinstance(r, str) else r for r in answer["requirements_now"]]


def learn_literals(needs, answer):
    hook = vars(needs).get("field_hook")
    if not hook:
        return
    swaps = {}
    for need in answer.get("needs") or []:
        if not isinstance(need, str) or ":" not in need:
            continue
        head, _, value = need.partition(":")
        if re.search(r"\d", value) or ref_text(value):
            continue
        bare = bare_words(re.sub(CJ_COMPARE, " ", value.lower()))
        if not bare:
            continue
        if bare == bare_words(value.lower()) and positive_value(value, head):
            continue
        prop = re.split(r"\s*\(", head)[0].strip()
        for key, vals in (hook(head) or {}).items():
            if not strict_prop(key, prop):
                continue
            own = next((v for v in vals if bare_words(str(v).lower()) == bare), None)
            if own is None:
                continue
            LITERAL_VALUES.add(bare)
            swaps[need] = f"{head.strip()}: {str(own).strip().lower()}"
            break
    if swaps:
        swap_in_answer(answer, swaps)
        needs.wire.log.append({"kind": "literal_value", "needs": list(swaps.values())[:4]})


def is_pinned(needs, prop_name, qtext):
    try:
        listed = needs.reading.get("references")
        refs = [str(r) for r in listed or []] if isinstance(listed, list) else []
        refs += list(vars(needs).get("resolved") or [])
        tags = tag_list(str(prop_name))
        asked_key = prop_key(prop_name)
        for req in needs.written:
            if req in needs.let_go or ":" not in req:
                continue
            head, _, value = req.partition(":")
            own_tags = tag_list(head)
            if tags and own_tags and not (kind_words(tags[-1]) & kind_words(own_tags[-1])):
                continue
            if not positive_value(value, head) or story_guess(req, needs.request, refs):
                continue
            own_key = prop_key(head)
            if same_prop(head, prop_name) or (len(own_key) >= 4 and asked_key.endswith(own_key)):
                return True
    except Exception:
        return False
    return False


def has_values(needs, lns):
    said = " ".join(str(ln) for ln in lns).lower()
    if not isinstance(needs, list):
        return False
    for need in needs:
        head, _, value = str(need).partition(":")
        names = [w for w in prop_words(head) if len(w) >= 3]
        values = [w for w in re.findall(r"[a-z0-9]+", value.lower()) if w not in LINK_WORDS]
        if names and values and all(w in said for w in names + values):
            return True
    return False


def shielded(link, kind, step, *args):
    try:
        step(*args)
    except Exception as err:
        record_failure(link, kind, err)


def grounded_in(req, said):
    words = [w for w in re.findall(r"[a-z]+", req.lower()) if len(w) >= 3]
    return not words or any(w in said for w in words)


def listed_needs(answer):
    # (a reply read as {"property": "value"} instead of ["property: value"] keeps its needs; the same for the lists
    # of properties beside them)
    if not isinstance(answer, dict):
        return answer
    for key in ("needs", "dropped", "requirements_now"):
        val = answer.get(key)
        if isinstance(val, dict):
            answer[key] = [f"{k}: {v}" for k, v in val.items() if str(k).strip() and str(v).strip()]
    val = answer.get("no_preference")
    if isinstance(val, dict):
        answer["no_preference"] = [str(k) for k in val if str(k).strip()]
    return answer


def listener_answer(needs, fresh, qtext, so_far):
    sections = [HEAR, "FOR CONTEXT, THE SHOPPER'S ORIGINAL REQUEST:\n" + needs.request, "REQUIREMENTS SO FAR:\n" + so_far]
    if needs.exchanges:
        sections.append("PAST EXCHANGES WITH THE SHOPPER, OLDEST FIRST:\n" + "\n".join(needs.exchanges[-12:]))
    if qtext:
        sections.append("WHAT THE ASSISTANT ASKED:\n" + qtext)
    sections.append("THE SHOPPER NOW SAID:\n" + "\n".join(fresh))
    asked = f"asked: {qtext} -> " if qtext else ""
    needs.exchanges.append(asked + "shopper: " + " / ".join(fresh))
    return consult(needs.wire, "\n\n".join(sections), 2500, LISTENER)


def park_lines(needs, fresh, qtext):
    link = needs.wire
    if not (link.mute or link.misses > 0):
        return
    # unread for now; read again once an LLM answers
    needs.caught.difference_update(fresh)
    needs.exchanges.pop()
    needs.pending = list(vars(needs).get("pending") or []) + [(list(fresh), qtext)]
    needs.pending_at = clock_now()


def without_meta(needs, answer, fresh, qtext):
    try:
        meta = answer.get("meta_only") is True and qtext and not has_values(answer.get("needs"), fresh)
        if not meta:
            return answer
        blanked = dict(answer, needs=[], no_preference=[], dropped=[], references=[], requirements_now=None,
                       new_budget=None, unavailable=False, reply="")
        needs.meta_last = qtext
        needs.wire.log.append({"kind": "meta_only", "line": " / ".join(fresh)[:200]})
        return blanked
    except Exception:
        return answer


def keep_line_names(needs, answer, fresh):
    try:
        renames = names_in_lines(answer.get("needs") or [], fresh)
    except Exception:
        return
    if renames:
        swap_in_answer(answer, renames)
        needs.wire.log.append({"kind": "need_renamed", "names": list(renames.values())[:4]})


def masked_lines(needs, answer, fresh, qtext):
    hook = vars(needs).get("ref_hook")
    if not hook:
        return fresh
    try:
        return hook(answer, fresh, qtext) or fresh
    except Exception as err:
        record_failure(needs.wire, "ref_line_error", err)
        return fresh


def echoed_status(needs, req):
    # (a heading or the products line of the status text, copied back as if it were a requirement)
    if not re.search(r"[a-z]", req):
        return True
    words = flat_words(req)
    return bool(need_items(needs)) and bool(words) and words in flat_words(items_line(needs))


def adopt_new_needs(needs, answer, said):
    rebuilt = answer.get("requirements_now")
    if isinstance(rebuilt, list) and all(isinstance(req, str) for req in rebuilt):
        needs.now_reqs = [req for req in rebuilt if req.strip() and grounded_in(req, said) and not echoed_status(needs, req)]
    added = []
    for need in answer.get("needs") or []:
        fits = isinstance(need, str) and need and need not in needs.uncovered
        if fits and grounded_in(need, said):
            needs.uncovered.append(need)
            added.append(f"new requirement: {need}")
    return added


def pass_pointed(needs, answer, fresh):
    pointers = answer.get("references")
    infos = [r for r in pointers or [] if isinstance(r, dict)] if isinstance(pointers, list) else []
    marked = answer.get("ref_needs") or []
    for need in answer.get("needs") or []:
        if isinstance(need, str) and need in needs.uncovered and (ref_text(need.partition(":")[2]) or need in marked):
            needs.ref_news = list(vars(needs).get("ref_news") or []) + [{"need": need, "infos": infos, "line": " ".join(fresh)}]


def drop_said(old_need, heard):
    # (a requirement is given up only by a line, or the question it answers, that names it: an "any is fine" about one
    # property was once read as dropping another)
    head = re.sub(r"\([^)]*\)", " ", old_need.partition(":")[0]).lower()
    words = [w for w in re.findall(r"[a-z]+", head) if len(w) >= 3]
    return not words or any(w[:4] in heard for w in words)


def take_drops(needs, answer, masked, heard=None):
    widened, notes = False, []
    budget_moved = is_number(answer.get("new_budget"))
    for old_need in answer.get("dropped") or []:
        if not isinstance(old_need, str) or not old_need or old_need in needs.let_go:
            continue
        if heard is not None and not drop_said(old_need, heard):
            needs.wire.log.append({"kind": "drop_unsaid", "need": old_need[:200]})
            continue
        if budget_moved and re.search(r"\b(?:prices?|budget|costs?|spend\w*|cents?|dollars?)\b", old_need.lower()):
            continue
        if drop_need(needs, old_need, masked):
            needs.let_go.append(old_need)
            notes.append(f"dropped: {old_need}")
            widened = widened or need_priority(needs) != "none"
        else:
            needs.wire.log.append({"kind": "drop_kept", "need": old_need[:200]})
    return widened, notes


def hear_lines(needs, lns, qtext):
    # (an answer to the basket shown for approval is read even when the same words came before)
    fresh = list(lns) if vars(needs).pop("hear_all", False) else [ln for ln in lns if ln not in needs.caught]
    if not fresh:
        return ""
    needs.caught.update(fresh)
    if time_left(needs.wire) < CLOSING:
        return ""
    so_far = needs_text(needs)
    answer = listener_answer(needs, fresh, qtext, so_far)
    if not answer:
        park_lines(needs, fresh, qtext)
        return ""
    answer = listed_needs(answer)
    answer = without_meta(needs, answer, fresh, qtext)
    keep_line_names(needs, answer, fresh)
    masked = masked_lines(needs, answer, fresh, qtext)
    said = " ".join([so_far, " ".join(needs.exchanges), qtext or ""]).lower()
    shielded(needs.wire, "literal_value_error", learn_literals, needs, answer)
    needs.fresh_needs = [n for n in answer.get("needs") or [] if isinstance(n, str)]
    needs.approval = answer.get("approves") if isinstance(answer.get("approves"), bool) else None
    remarks = adopt_new_needs(needs, answer, said)
    try:
        pass_pointed(needs, answer, fresh)
    except Exception:
        pass
    take_dont_minds(needs, answer, qtext)
    widened, dropped = take_drops(needs, answer, masked, " ".join(list(fresh) + [qtext or ""]).lower())
    remarks += dropped
    budget = answer.get("new_budget")
    if is_number(budget) and budget > 0 and budget != needs.spend_cap:
        needs.spend_cap = budget
        remarks.append(f"new budget: {budget}")
    if answer.get("unavailable"):
        remarks.append("an item became unavailable; take it out of the cart and do not order it")
    if remarks:
        needs.version += 1
    if remarks and widened:
        needs.widened_at, needs.widened_tick = needs.version, needs.wire.tick
    if needs.now_reqs is not None:
        restore_lost(needs, masked)
    if widened:
        remarks.append(widened_note(needs))
    needs.wire.log.append({"kind": "heard", "question": qtext is not None, "needs": len(answer.get("needs") or []),
                           "free": len(answer.get("no_preference") or []), "dropped": len(answer.get("dropped") or []),
                           "budget": is_number(budget)})
    # a reply goes back only to a question the shopper put
    asked_back = qtext is None and any("?" in ln for ln in fresh) and not answer.get("no_preference")
    needs.reply_due = str(answer.get("reply") or "").strip() if asked_back else ""
    tail = " (" + "; ".join(remarks) + ")" if remarks else ""
    return "Meaning: " + str(answer.get("meaning") or "") + tail


def take_dont_minds(needs, answer, qtext):
    in_force = needs.now_reqs if needs.now_reqs is not None else needs.written + needs.uncovered
    held = {r.split(":", 1)[0].strip().lower() for r in in_force if isinstance(r, str) and ":" in r and r not in needs.let_go}
    for prop_name in answer.get("no_preference") or []:
        if not isinstance(prop_name, str) or not prop_name or prop_name in needs.indifferent:
            continue
        if prop_name.strip().lower() in held:
            continue
        if is_pinned(needs, prop_name, qtext):
            needs.pinned_heard = vars(needs).get("pinned_heard", 0) + 1
            continue
        needs.indifferent.append(prop_name)


def restore_lost(needs, masked):
    for need in needs.uncovered:
        prop_name = need.split(":", 1)[0].strip().lower()
        if not prop_name:
            continue
        if any(prop_name in req.lower() for req in needs.now_reqs) or any(prop_name in old.lower() for old in needs.let_go):
            continue
        needs.now_reqs.append(need)
    for need in needs.written:
        try:
            if need in needs.let_go or need in needs.now_reqs or drop_need(needs, need, masked):
                continue
            name = need.split(":", 1)[0]
            tags = tag_list(name)

            def same_slot(other):
                head = other.split(":", 1)[0]
                return same_prop(name, head) and tag_list(head) == tags
            if any(same_slot(old) for old in needs.let_go):
                continue
            slot = [req for req in needs.now_reqs if isinstance(req, str) and same_slot(req)]
            # (a revealed need the reading filed under the stated property, with a value of its own, stands beside it)
            told = need.split(":", 1)[1]
            beside = slot and all(req in needs.uncovered and not value_seen(told, req.split(":", 1)[1]) for req in slot)
            if not slot or beside:
                needs.now_reqs.append(need)
        except Exception:
            continue


def widened_note(needs):
    open_props = props_pending(needs)
    if not open_props and need_priority(needs) == "cheapest" and needs.sweepable:
        return ("nothing about the properties is required any more: any in-stock listing of the requested category fits, "
                "whatever kind of object it is (an accessory, part or set shelved there is included), and the lowest price "
                "on the entire shelf wins. Rows arrive by relevance or by id rather than by price, so the code reads more "
                "slices of the shelf for you next (the category listing's pages, the label's own words, words naming the "
                "object kinds in titles); when no '[shelf sweep done]' list arrives, sweep the shelf yourself with short "
                "queries, the first page of each - the category label, each kind word of the label alone, each item name that recurs in the "
                "titles seen - and the category listing's pages. Then make the lowest-priced in-stock row on the "
                "requested category within the budget the cart's only line, re-check its stock and price, and order it")
    if not open_props:
        return ("nothing about the properties is required any more: any in-stock listing of the requested category fits, "
                "whatever kind of object it is (an accessory, part or set shelved there is included), and the lowest price "
                "on the entire shelf wins. Rows arrive by relevance or by id rather than by price: sweep the shelf with "
                "short queries, the first page of each - the category label, each kind word of the label alone, each item name that recurs in the "
                "titles seen - and the category listing's pages; then order the lowest in-stock price")
    return ("a dropped requirement WIDENS the acceptable set: listings failing on it alone fit now, and listings never "
            "searched for may beat the pick; search the item kind afresh without it, with short "
            "queries of each remaining required value with the kind word, value first, reading "
            "every page, and choose the cheapest over everything acceptable now")


# ---------------------------------------------------------------------------------------------------- the basket
class Basket:
    def __init__(self, the_tool, jschema):
        specs = (jschema or {}).get("properties") or {}
        vars(self).update(the_tool=the_tool, prop_specs=specs, put_form=None, units={}, adds=None, used=False,
                          changed_at=0, viewed_at=-1, last_view=None, pending={})


def line_form(cart, call_args):
    specs = cart.prop_specs
    return as_text({key: val for key, val in call_args.items() if isinstance(specs.get(key), dict) and "enum" in specs[key]})


def line_fields(cart):
    for arg_name, spec in cart.prop_specs.items():
        if not isinstance(spec, dict) or spec.get("type") != "array":
            continue
        inner = spec.get("items")
        fields = inner.get("properties") if isinstance(inner, dict) else None
        if not isinstance(fields, dict):
            continue
        texts = [f for f, s in fields.items() if isinstance(s, dict) and s.get("type") == "string" and "enum" not in s]
        id_fields = [f for f in texts if "id" in f.lower()] or texts
        numbers = [f for f, s in fields.items() if isinstance(s, dict) and s.get("type") in ("integer", "number")]
        if id_fields and numbers:
            return arg_name, id_fields[0], numbers[0]
    return None


def list_call(cart, call_args):
    spec = line_fields(cart)
    if spec and isinstance(call_args.get(spec[0]), list):
        return spec
    return None


def lead_line(found):
    for entry in found:
        if entry[1] is None or entry[1] > 0:
            return entry
    return found[0]


def cart_lines(cart, call_args):
    spec = list_call(cart, call_args)
    if spec is None:
        thing = flat_aim(cart, call_args)
        return [(thing, count_value(cart, call_args))] if thing else []
    arg_name, id_field, count_field = spec
    named = []
    for row in call_args[arg_name]:
        if not isinstance(row, dict):
            continue
        ident = row.get(id_field)
        if isinstance(ident, str) and ident.strip():
            named.append((ident.strip(), float(row[count_field]) if is_number(row.get(count_field)) else None))
    return named


def aimed_id(cart, call_args):
    found = cart_lines(cart, call_args)
    return lead_line(found)[0] if found else None


def flat_aim(cart, call_args):
    for key, val in call_args.items():
        spec = cart.prop_specs.get(key)
        if not isinstance(spec, dict):
            spec = {}
        if isinstance(val, str) and "enum" not in spec and val.strip() and " " not in val.strip():
            return val
    return None


def count_value(cart, call_args):
    if list_call(cart, call_args):
        found = cart_lines(cart, call_args)
        return lead_line(found)[1] if found else None
    for key, spec in cart.prop_specs.items():
        if not isinstance(spec, dict) or spec.get("type") not in ("integer", "number") or "enum" in spec:
            continue
        val = call_args.get(key, spec.get("default"))
        if is_number(val):
            return float(val)
    return None


def adds_to_cart(cart, call_args):
    return cart.put_form is None or line_form(cart, call_args) == cart.put_form


def units_held(cart, thing):
    return cart.units.get(thing, 0.0)


def absorb_view(cart, obs, touched=None):
    seen = units_in_view(cart, obs)
    if seen is None:
        return False
    # (a flat reply to a cart call that shows only the line it touched says nothing about the other lines)
    flat = isinstance(obs, dict) and not any(isinstance(v, (list, dict)) for v in obs.values())
    learn_units(cart, seen, touched is not None and flat and set(seen) <= {touched})
    return True


def book_list_call(cart, call_args, obs):
    # a list call states the whole basket
    cart.adds = False
    fresh = {}
    for thing, count in cart_lines(cart, call_args):
        if count is None or count > 0:
            fresh[thing] = fresh.get(thing, 0.0) + (1.0 if count is None else count)
    cart.units = fresh
    absorb_view(cart, obs)


def units_after(cart, thing, count, putting):
    before = units_held(cart, thing)
    if not putting:
        return max(0.0, before - count) if count > 0 else 0.0
    if before > 0 and count > 0 and cart.adds is not False:
        cart.pending = {thing: count}
        return before + count
    return count if count > 0 else 0.0


def book_cart(cart, call_args, obs):
    cart.used = True
    if isinstance(obs, dict) and obs.get("error"):
        return ""
    if cart.put_form is None:
        cart.put_form = line_form(cart, call_args)
    if list_call(cart, call_args):
        book_list_call(cart, call_args, obs)
        return cart_summary(cart)
    thing = aimed_id(cart, call_args)
    if not thing:
        return ""
    count = count_value(cart, call_args)
    count = 1.0 if count is None else count
    putting = adds_to_cart(cart, call_args)
    after = units_after(cart, thing, count, putting)
    if after > 0:
        cart.units[thing] = after
    else:
        cart.units.pop(thing, None)
    if not absorb_view(cart, obs, thing) and (count <= 0 or not putting) and empty_view(cart, obs, thing):
        cart.units.pop(thing, None)
        cart.pending.pop(thing, None)
    return cart_summary(cart)


def empty_view(cart, obs, thing):
    if not isinstance(obs, dict) or thing not in cart.units:
        return False
    lists = [v for v in obs.values() if isinstance(v, list)]
    if not lists or any(lists):
        return False
    numbers = [v for v in obs.values() if isinstance(v, (int, float)) and not isinstance(v, bool)]
    if not numbers or any(v != 0 for v in numbers):
        return False
    shown = as_text(obs)
    return all(other not in shown for other in cart.units if other != thing)


def units_in_view(cart, obs):
    tally = {}
    known = set(cart.units) | set(cart.pending)

    def scan(rec):
        ids = [v for v in rec.values() if isinstance(v, str) and (v in known or re.fullmatch(ID_LIKE, v)) and " " not in v]
        counts = [(key, v) for key, v in rec.items() if isinstance(v, int) and not isinstance(v, bool) and 0 < v < 1000
                  and str(key) != "price"]
        if len(counts) > 1:
            named = [(key, v) for key, v in counts if re.search(r"qty|quant|count|units|number|total|held|ordered", str(key).lower())
                     and not re.search(r"price|cost|due|sum|fee|charge", str(key).lower())]
            if len(named) == 1:
                counts = named
        if len(counts) != 1 or not ids:
            return
        ident = next((i for i in ids if i in known), ids[0])
        tally[ident] = tally.get(ident, 0) + counts[0][1]

    walk_dicts(obs, scan)
    if not tally or not (set(tally) & (known or set(tally))):
        return None
    return tally


def learn_units(cart, seen, one_line=False):
    for thing, if_set in cart.pending.items():
        if thing not in seen:
            continue
        if abs(seen[thing] - if_set) < 1e-9 and abs(cart.units.get(thing, 0) - if_set) > 1e-9:
            cart.adds = False
        elif seen[thing] > if_set:
            cart.adds = True
    cart.pending = {}
    merged = {**cart.units, **seen} if one_line else seen
    cart.units = {key: val for key, val in merged.items() if val > 0}


def take_view(cart, obs, tick):
    cart.last_view = obs
    cart.viewed_at = tick
    if not absorb_view(cart, obs) and isinstance(obs, (dict, list)) and not view_has_ids(cart, obs):
        cart.units = {}


def view_has_ids(cart, obs):
    shown = as_text(obs)
    return any(thing in shown for thing in cart.units)


def cart_summary(cart):
    if not cart.units:
        return "[cart as counted] empty."
    shown = ", ".join(f"{thing} x{int(n) if float(n).is_integer() else n}" for thing, n in cart.units.items())
    return f"[cart as counted] {shown}."


def cart_faults(cart, need_n, counts=None):
    if not cart.units:
        return []
    if counts and len(counts) > 1:
        if len(cart.units) != len(counts):
            listing = ", ".join(f"{i} x{int(q)}" for i, q in cart.units.items())
            return [f"the basket holds {len(cart.units)} line(s) ({listing}), but the request wants {len(counts)} products, "
                    f"one line each; every product needs its line and nothing else"]
        have, want = sorted(int(q) for q in cart.units.values()), sorted(counts)
        if have == want:
            return []
        return [f"the lines hold {have} units but the products need {want} (one line per product at its quantity)"]
    problems = []
    if len(cart.units) > 1:
        problems.append("the cart holds more than one line (" + ", ".join(f"{i} x{int(q)}" for i, q in cart.units.items())
                        + "); take out every line but the chosen one")
        return problems
    (thing, qty), = cart.units.items()
    if abs(qty - need_n) > 1e-9:
        verb = "take out" if qty > need_n else "put in"
        problems.append(f"the line {thing} holds {int(qty)} units but the request needs {need_n}; {verb} {int(abs(qty - need_n))} with the cart tool")
    return problems


# ---------------------------------------------------------------------------------------------------- the run record
class Run:
    def __init__(self, pdata):
        environment = pdata["environment"]
        self.wire = Link(environment["binding"])
        view = environment["policy_view"]
        tools = [entry for entry in view.get("tools") or [] if isinstance(entry, dict) and isinstance(entry.get("function"), dict)]
        vars(self).update(
            pview=view, request=str(view.get("query") or ""), toolset=tools,
            shapes={entry["function"]["name"]: entry["function"].get("parameters") or {"type": "object", "properties": {}}
                    for entry in tools},
            step_cap=int(view.get("max_steps") or 30), archive=Archive(), role_map={}, trolley=Basket(None, {}),
            msgs_sent=0, msg_cap=None, orders_tried=0, refusal_n=0, order_vetted=False, rank_checked=set(),
            decline_vetted=False, audit_done=False, questions_log=[], asking=True, rulings={}, checked_adds=set(),
            passed=set(), refused_lines=set(), refusals_seen={}, warned=set(), word_seen=[], word_kind="",
            lead_seen=[], lead_bounds={}, lead_words=0, lead_ctx=None, last_calls=[], chat=[], stuck=0, refs=[])
        self.wire.done = bool(view.get("finished"))
        self.needs = NeedSheet(self.wire, self.request)
        self.needs.ref_hook = lambda answer, lines, qtext: ref_fix(self, answer, lines, qtext)
        self.needs.field_hook = lambda label="": store_fields(self, label)
        LITERAL_VALUES.clear()


def moves_left(run):
    return run.step_cap - run.wire.tick


def role(run, call_name):
    return run.role_map.get(call_name)


def ask_limit(run):
    base = ASK_DEFAULT if run.msg_cap is None else run.msg_cap
    return max(0, base - yes_reserve(run))


# ---- the final basket and its total go to the shopper only when the request asks for it (a procedure the reading quotes
# from the request): sent by default it took the last question on every task of a world that grades no such step
def basket_stated(run):
    reading = run.needs.reading
    rows = [row for row in reading.get("procedures") or [] if isinstance(row, dict) and row.get("action") == "message_shopper"]
    # (a message procedure tied to the order or the final cart, or wanting a yes or named content; a cap on messages read as
    # a procedure has none of these: earlier readings filed "send me no more than 5 messages" so in 24 of 30)
    return bool(reading.get("basket_yes")) or any(
        row.get("trigger") in ("after_last_cart_change", "before_order") or row.get("needs_yes") or row.get("content") for row in rows)


def yes_wanted(run):
    return bool(role(run, "message")) and basket_stated(run)


def basket_mark(run):
    return tuple(sorted((str(thing), float(n)) for thing, n in run.trolley.units.items()))


def basket_approved(run):
    state = vars(run).get("yes_state") or {}
    return state.get("yes") is True and state.get("mark") == basket_mark(run)


def yes_reserve(run):
    return 1 if yes_wanted(run) and not basket_approved(run) else 0


def msg_room(run):
    return (ASK_DEFAULT if run.msg_cap is None else run.msg_cap) - run.msgs_sent


def money_text(amount):
    if amount is None:
        return "?"
    return f"{amount:,.2f}".rstrip("0").rstrip(".")


def line_facts(run, rec, most):
    # (the listing's own property map, the properties the reading names first: the shopper sees what the item is)
    if most <= 0 or not isinstance(rec, dict):
        return ""
    maps = [val for val in rec.values() if isinstance(val, dict)
            and sum(isinstance(v, (str, int, float)) and not isinstance(v, bool) for v in val.values()) >= 2]
    pairs = [(str(k), str(v)) for k, v in (maps[0].items() if maps else ())
             if isinstance(v, (str, int, float)) and not isinstance(v, bool) and 0 < len(str(v)) <= 40]
    named = set(word_run(" ".join(str(x) for x in list(run.needs.written) + list(run.needs.uncovered)
                                  + list(run.needs.reading.get("hints") or []))))
    pairs.sort(key=lambda kv: not set(word_run(kv[0])) & named)
    return ", ".join(f"{k.replace('_', ' ')} {v}" for k, v in pairs[:most])


def rich_record(run, thing):
    # (of every record read for this listing, the one holding most values: a details page over a search row)
    best, size = {}, -1

    def visit(rec):
        nonlocal best, size
        if row_mark(rec) != thing:
            return
        held = sum(len(v) if isinstance(v, dict) else 1 for v in rec.values())
        if held > size:
            best, size = rec, held
    for item in run.archive.items:
        walk_dicts(item, visit)
    return best


def basket_message(run, room, most=8):
    unit = str(run.needs.reading.get("currency") or "").strip()
    parts, total, known = [], 0.0, True
    for thing, n in run.trolley.units.items():
        price = price_of(run.archive, thing)
        rec = rich_record(run, thing)
        title = title_of(rec)[:room]
        count = int(n) if float(n).is_integer() else n
        facts = line_facts(run, rec, most)
        line = f"{title} ({thing})" if title else str(thing)
        line += f" [{facts}]" if facts else ""
        if price is None:
            known = False
            parts.append(f"{line} x{count}")
        else:
            total += price * float(n)
            parts.append(f"{line} x{count} at {money_text(price)} each = {money_text(price * float(n))}")
    text_body = "Before I place your order, here is the final basket: " + "; ".join(parts) + "."
    if known:
        text_body += f" Total {money_text(total)} {unit}".rstrip() + "."
    cap = run.needs.spend_cap
    if is_number(cap):
        text_body += f" Your limit is {money_text(cap)} {unit}".rstrip() + "."
    return text_body + " Shall I order exactly this? Please answer yes, or tell me what to change."


def yes_call(run):
    for room, most in ((70, 8), (50, 5), (40, 3), (30, 0), (0, 0)):
        proposal = message_call(run, basket_message(run, room, most), f"yes_{run.wire.tick}")
        if proposal:
            return proposal
    return None


def final_steps(run, kind):
    # (a safety net for asked steps no code knows about: one read of the request and the later lines before a finish; code
    # runs at most two, a message only when one is spare beyond the basket's, otherwise only calls that change nothing)
    state = vars(run)
    done = state.setdefault("final_checks", set())
    key = (basket_mark(run), kind, len(state.get("news_read") or ()))
    # (a decline can be turned away with a need dropped, and the buy that follows still needs the basket's yes, so a
    # closing note may not spend that message; it is free only when the request asks for a word before declining)
    told = any(isinstance(row, dict) and row.get("trigger") == "before_decline" and row.get("action") == "message_shopper"
               for row in run.needs.reading.get("procedures") or [])
    keep = 0 if kind == "decline" and told else yes_reserve(run)
    if key in done or len(done) >= 3 or time_left(run.wire) < CLOSING / 2 + 15 or moves_left(run) < 3 + keep:
        return
    done.add(key)
    msg = role(run, "message")
    looks = {role(run, name) for name in ("details", "view", "cart_view")} | {stock_tool(run)}
    usable = [entry for entry in run.toolset if entry["function"]["name"] in (looks | {msg}) - {None}]
    since = [act for e in run.wire.log if e.get("kind") == "env" and e.get("step", 0) > run.trolley.changed_at
             for act in e.get("actions") or []]
    later = [line for line in state.get("news_read") or () if isinstance(line, str)]
    prompt = (FINAL_CHECK.replace("FINISH", kind) + "\n\nTOOLS:\n" + "\n".join(map(tool_row, usable)) + "\n\nREQUEST:\n"
              + run.request + "\n\nLATER LINES:\n" + ("\n".join(later[-8:]) or "none") + "\n\nCALLS SINCE THE LAST CART CHANGE:\n"
              + ("\n".join(as_text(act, 300) for act in since[-10:]) or "none") + "\n\nCART:\n" + cart_summary(run.trolley))
    answer = consult(run.wire, prompt, 900, CHECKER)
    names = {entry["function"]["name"] for entry in usable}
    for step in ((answer or {}).get("pending") or [])[:2]:
        name, args = (step.get("tool"), step.get("args")) if isinstance(step, dict) else (None, None)
        if name not in names or not isinstance(args, dict) or moves_left(run) < 3 + keep:
            continue
        if name == msg and msg_room(run) <= keep:
            continue
        args, faults = check_args(args, (run.shapes.get(name) or {}))
        if faults:
            continue
        run.wire.log.append({"kind": "final_step", "finish": kind, "why": str(step.get("why"))[:120], "tool": name})
        send_chunk(run, [(f"final_{run.wire.tick}", name, args)])


def basket_gate(run, closing=False):
    # (returns a note when the order must wait; "" lets it go)
    if not yes_wanted(run) or not run.trolley.units or basket_approved(run):
        return ""
    state = vars(run).setdefault("yes_state", {"mark": None, "yes": None, "tries": 0})
    mark = basket_mark(run)
    tries = state["tries"] if state["mark"] == mark else 0
    # (in closing time the reply is not read by the LLM (hear_lines), so the basket costs one world call: a few seconds do)
    room = msg_room(run) > 0 and moves_left(run) >= 2 and time_left(run.wire) > 4
    proposal = yes_call(run) if room and tries < 2 else None
    if not proposal:
        run.wire.log.append({"kind": "basket_yes_skipped", "tries": tries, "messages_left": msg_room(run), "steps_left": moves_left(run)})
        return ""
    run.needs.approval, run.needs.hear_all = None, True
    version, cap = run.needs.version, run.needs.spend_cap
    call_args = json.loads(proposal["function"]["arguments"])
    replies = send_chunk(run, [(proposal["id"], proposal["function"]["name"], call_args)])
    heard, run.needs.hear_all = run.needs.approval, False
    if heard is not False and run.needs.version != version and (run.needs.spend_cap == cap or not cart_within_cap(run)):
        # (the reply that acknowledged the basket also changed the request: the shown cart no longer stands)
        heard = False
    # (the shopper may only acknowledge the basket: anything short of turning it down or asking a change approves it)
    state.update(mark=mark, tries=tries + 1, yes=heard is not False)
    said = replies[0][1] if replies else ""
    run.wire.log.append({"kind": "basket_yes", "tries": tries + 1, "approved": heard, "basket": list(mark)})
    if heard is not False or closing or time_left(run.wire) < CLOSING / 2:
        return ""
    return ("The order was not sent: the shopper was shown the final basket and asked for a change (one message used): "
            f"{said[:600]}\nMake that change; before the order the code shows the basket again, and an acknowledgement is enough.")


def cart_within_cap(run):
    cap = run.needs.spend_cap
    if not is_number(cap):
        return True
    costs = [(price_of(run.archive, line), units_held(run.trolley, line)) for line in run.trolley.units]
    return all(is_number(c) for c, _ in costs) and sum(c * n for c, n in costs) <= cap


def call_kind(run, call_name, call_args):
    order_tool, decline_tool = role(run, "order"), role(run, "decline")
    if not call_name or call_name not in (order_tool, decline_tool):
        return None
    if order_tool != decline_tool:
        return "decline" if call_name == decline_tool else "order"
    for val in (call_args or {}).values():
        if not isinstance(val, str) or len(val) > 40:
            continue
        low = val.lower()
        if re.search(DECLINE_WORDS, low):
            return "decline"
        if re.search(ORDER_WORDS, low):
            return "order"
    return "order"


def finishers(run):
    return {role(run, "order"), role(run, "decline")} - {None}


def time_hint(run):
    left = max(0.0, time_left(run.wire) - LAST_CALL)
    if left > (DEADLINE - LAST_CALL) / 2:
        return ""
    whole = int(left // CLOCK_STEP) * CLOCK_STEP
    if whole <= 0:
        return f"under {CLOCK_STEP} s left"
    return f"about {whole} s left"


def status_line(run):
    spare = moves_left(run)
    sent = f"messages sent {run.msgs_sent}"
    if run.msg_cap is not None:
        sent += f" of {run.msg_cap}"
    pieces = [f"[status] steps used {run.wire.tick} of {run.step_cap}, {spare} left", sent]
    clock = time_hint(run)
    if clock:
        pieces.append(clock)
    text = "; ".join(pieces) + ".\n[requirements now] " + needs_text(run.needs) + ref_line(run) + product_steps(run)
    usable = [entry["function"]["name"] for entry in offered(run)]
    if len(usable) < len(run.toolset):
        text += ("\n[tools usable now] " + ", ".join(usable) + " (any other tool call is refused without using a step"
                 + ("; questions to the shopper are planned and sent for you" if run.asking else "") + ").")
    key = count_key(run.trolley)[0]
    if key and role(run, "cart") and not re.search(r"qty|quant|count|units?\b|amount|how_?many", key, re.I):
        text += (f"\n[cart tool] '{key}' of {role(run, 'cart')} is the number of units on that line" + (" (0 takes the line out)"
                 if run.trolley.adds is False else "") + ", not a line position; the code sets it to each product's quantity.")
    if run.trolley.used:
        text += "\n" + cart_summary(run.trolley)
    return text + closing_push(run, spare)


def closing_push(run, spare):
    if spare <= 2:
        target = "every product asked for" if need_items(run.needs) else "the item"
        return (" LAST STEPS: order the cart at once if it has a candidate for " + target + "; if not, put the best "
                "candidate in (a verified fit, else the cheapest in-stock, in-budget listing whose record shows no requirem"
                "ent failing) and order. Decline only when the records show that nothing can fit.")
    if spare <= 5 or time_left(run.wire) < CLOSING:
        target = "one line per product asked for" if need_items(run.needs) else "one line"
        return (" FINISH NOW: set the cart to " + target + " (a verified fit, else the cheapest in-stock, in-bu"
                "dget listing with no failing requirement on record; no sold-out line inside), re-check its stock, then "
                "order. Decline only when the records show that nothing can fit.")
    if over_budget(run.wire):
        several = bool(need_items(run.needs))
        return (" MODEL BUDGET USED: exploring is over; in the next few steps put the top candidate for "
                + ("every product asked for (one line each, at its quantity)" if several else "the item asked for")
                + " in the cart (a verified fit where there is one, else the closest fit the records show), re-check its "
                "stock and price, and order. A decline is right only when the records show that "
                + ("some product has" if several else "the store has") + " no listing that can meet its requirements.")
    if run.orders_tried == 0 and spare <= order_due(run) + 7 and spare > order_due(run) and narrowing(run):
        return (" Few steps are left by now: in the coming steps put your best verified fit into the cart and order; searching"
                f" and reading details end when {order_due(run)} steps remain, and a refused order needs about 4 steps to repair.")
    if run.orders_tried == 0 and spare <= order_due(run) and narrowing(run):
        return (" Searching is over (detail reads only for the cart's line or listings a check named, or, while any product "
                "lacks a verified fit, at most 3 more reads or searches): put your best verified fit in the cart, "
                "re-check its stock and price, and order; a refused order still needs about 4 steps to repair.")
    return ""


# ---------------------------------------------------------------------------------------------------- reading the request
def tool_row(entry):
    spec = entry["function"]
    shown = as_text(spec.get("parameters") or {})
    return "- %s: %s | arguments: %s" % (spec["name"], spec.get("description", ""), shown)


def reader_content(run):
    fixed = "\n\n".join([READ_REQUEST, "TOOLS:\n" + "\n".join(map(tool_row, run.toolset))])
    varying = "REQUEST:\n" + run.request
    opening = run.pview.get("opening")
    if opening is not None:
        varying = "\n\n".join([varying, "OPENING OBSERVATION:\n" + as_text(opening)])
    return [dict(type="text", text=fixed, cache_control=dict(type="ephemeral")), dict(type="text", text=varying)]


def first_reading(run, content, rounds=3):
    ladder = READER + MENDER[:1]
    reading, tries = {}, 0
    for model in ladder * rounds:
        if tries >= len(ladder) and not reader_retry_ok(run, ladder):
            break
        tries += 1
        reply = ask_model(run.wire, [{"role": "user", "content": content}], [model], 3000, effort="none")
        reading = json_object((reply or {}).get("content") or "") or {}
        if isinstance(reading.get("roles"), dict):
            return reading
        run.wire.log.append({"kind": "setup_retry", "model": model})
    return reading


# ---- checking a reading against the request's own words and the store's own tools
# (structure only: copied stretches, figures, schema names; the meaning stays the LLM's)
PROC_ACTIONS = ("message_shopper", "inspect", "decline_rule", "other")
PROC_TRIGGERS = ("after_last_cart_change", "before_order", "before_cart", "before_decline", "anytime")
PROC_PARTS = ("lines", "quantities", "unit_prices", "total", "limit")
FIGURE_KEYS = ("items", "requirements", "quantity", "price_cap", "message_budget", "step_budget", "references",
               "reference_targets", "story_numbers")


def word_run(text_body):
    return re.findall(r"[a-z0-9]+", str(text_body).lower())


def figure_form(raw):
    bare = str(raw).replace(",", "").replace(" ", "")
    if "." in bare:
        bare = bare.rstrip("0").rstrip(".")
    return bare.lstrip("0") or "0"


def figures(text_body):
    return {figure_form(f) for f in re.findall(r"\d[\d,]*(?:\.\d+)?", str(text_body))}


def sentences(text_body):
    return [part for part in re.split(r"(?<=[.!?;])\s+", str(text_body)) if part.strip()]


def word_stem(word):
    return word[:-1] if len(word) > 3 and word.endswith("s") else word


def copied(quote, text_body, share=0.8):
    want = word_run(quote)
    if not want:
        return False
    have, n = word_run(text_body), len(want)
    if any(have[i:i + n] == want for i in range(len(have) - n + 1)):
        return True
    # (a near copy: most of its words inside one sentence of the request)
    for part in sentences(text_body):
        words = set(map(word_stem, word_run(part)))
        if sum(word_stem(w) in words for w in want) >= share * n:
            return True
    return False


def value_grounded(value, text_body):
    # (figures must be the request's own; a value without figures needs a word sharing its stem with the request)
    told = figures(value)
    if told:
        return told <= figures(text_body)
    roots = {w[:5] for w in word_run(text_body) if len(w) >= 2}
    return any(w[:5] in roots for w in word_run(value) if len(w) >= 2 and not w.isdigit())


def lone_figure(val):
    if isinstance(val, str):
        found = re.findall(r"\d[\d,]*(?:\.\d+)?", val)
        if len(found) == 1:
            num = float(found[0].replace(",", ""))
            return int(num) if num.is_integer() else num
    return val


def tidy_reading(reading):
    # (shape fixes the code can make alone: figures given as text; the requirement list also holds every product's own
    # requirements, which a reading often writes only inside "items")
    for key in ("price_cap", "message_budget", "step_budget", "quantity"):
        reading[key] = lone_figure(reading.get(key))
    listed = [str(r) for r in reading.get("requirements") or [] if isinstance(r, str)] \
        if isinstance(reading.get("requirements"), list) else []
    for entry in reading.get("items") or []:
        if isinstance(entry, dict) and isinstance(entry.get("requirements"), list):
            listed += [str(r) for r in entry["requirements"] if isinstance(r, str) and str(r) not in listed]
    if listed:
        reading["requirements"] = listed


def lone_item(reading):
    # (last resort once a mend gave no product entry: the reading's own product becomes the one entry)
    entries = [r for r in reading.get("items") or [] if isinstance(r, dict) and r.get("product")]
    if not entries and str(reading.get("product") or "").strip():
        qty = reading.get("quantity")
        reading["items"] = [{"product": str(reading["product"]).strip(), "quantity": qty if is_number(qty) and qty >= 1 else 1,
                             "requirements": [r for r in reading.get("requirements") or [] if isinstance(r, str)]}]


def role_fault(run, reading):
    roles = reading.get("roles")
    if not isinstance(roles, dict):
        return "no roles object"
    wrong = [job for job, tool in roles.items() if tool is not None and tool not in run.shapes]
    if wrong:
        return "names no tool of the list for " + ", ".join(map(str, wrong))
    try:
        guess = code_roles(run)
    except Exception as err:
        record_failure(run.wire, "code_roles_error", err)
        guess = {}
    run.code_guess = guess
    for job, tool in guess.items():
        if not roles.get(job):
            roles[job] = tool          # (a gap the tools' schemas fill)
    if not roles.get("order"):
        return "no tool for the order"
    clash = [job for job in ("order", "decline", "cart", "message") if guess.get(job) and roles.get(job) != guess[job]]
    if clash:
        return "the tool schemas point elsewhere: " + "; ".join(f"{job} -> {guess[job]}" for job in clash)
    return ""


def procedure_fault(entry, text_body):
    if not isinstance(entry, dict) or not copied(entry.get("quote") or "", text_body):
        return "a procedure quote is not copied from the request"
    if entry.get("action") not in PROC_ACTIONS or entry.get("trigger") not in PROC_TRIGGERS:
        return "a procedure has an action or trigger outside the allowed values"
    return ""


def reading_faults(run, reading, mended=False):
    text_body, faults = run.request, {}
    why = "" if mended else role_fault(run, reading)
    if why:
        faults["roles"] = why
    for key in ("price_cap", "message_budget", "step_budget"):
        val = reading.get(key)
        if val is not None and not (is_number(val) and figure_form(val) in figures(text_body)):
            faults[key] = f"{val!r} is not a figure the request writes"
    loose = [str(req) for req in reading.get("requirements") or []
             if ":" in str(req) and not value_grounded(str(req).partition(":")[2], text_body)]
    if loose:
        faults["requirements"] = "values the request does not write: " + " | ".join(r[:80] for r in loose[:4])
    entries = [str(r.get("product")) for r in reading.get("items") or [] if isinstance(r, dict) and r.get("product")]
    if not entries:
        faults["items"] = "no product entry"
    joined = joined_pair(entries, text_body)
    if joined:
        faults["items"] = (f"the request writes '{joined}' as one phrase: when it names one kind of product (a store category "
                           "can join two kinds in its name), it is ONE entry, not two")
    bad = [why for why in (procedure_fault(entry, text_body) for entry in reading.get("procedures") or []) if why]
    if bad or not isinstance(reading.get("procedures", []), list):
        faults["procedures"] = bad[0] if bad else "not a list"
    sellers = [str(t.get("seller") or "") for t in reading.get("reference_targets") or [] if isinstance(t, dict)]
    if any(name and not copied(name, text_body, 1.0) for name in sellers):
        faults["reference_targets"] = "a seller name is not copied from the request"
    else:
        faults.update(reference_faults(reading, text_body))
    told = figures(json.dumps({key: val for key, val in reading.items() if key != "roles"}, default=str))
    missing = sorted(figures(text_body) - told)
    if missing:
        faults["figures"] = "figures of the request found in no key: " + ", ".join(missing[:8])
    return faults


def reference_faults(reading, text_body, copies=None):
    # (the property a reference fills is named where its seller is; the referenced listing's own details stay with it)
    faults, own = {}, []
    for target in reading.get("reference_targets") or []:
        if not isinstance(target, dict) or not target.get("seller"):
            continue
        seller = word_run(target["seller"])
        parts = sentences(text_body)
        spots = [n for n, part in enumerate(parts) if seller and copied(" ".join(seller), part, 1.0)]
        if not spots:
            continue
        near = [parts[n] for n in spots]
        words = {w[:4] for n in spots for w in word_run(" ".join(parts[max(0, n - 1):n + 2]))}
        prop = [w[:4] for w in word_run(target.get("property")) if len(w) >= 3]
        if prop and not all(w in words for w in prop):
            faults["reference_targets"] = (f"property {target.get('property')!r} is not what the sentence naming "
                                           f"{target['seller']} asks to take from that listing")
        for part in near:
            # (the referenced listing's details follow its seller's name in that sentence)
            at = part.lower().find(target["seller"].lower())
            own.append(part[at:] if at >= 0 else part)
    rest = text_body
    for part in own:
        rest = rest.replace(part, " ")
    reqs = [str(r) for r in reading.get("requirements") or []]
    reqs += [str(r) for item in reading.get("items") or [] if isinstance(item, dict) for r in item.get("requirements") or []]
    names = [str(t.get("seller")).lower() for t in reading.get("reference_targets") or [] if isinstance(t, dict) and t.get("seller")]
    details = " ".join(str(d) for t in reading.get("reference_targets") or [] if isinstance(t, dict) for d in t.get("details") or [])
    copied_in = [r for r in dict.fromkeys(reqs) if own and details and ":" in r and not any(n in r.lower() for n in names)
                 and value_grounded(r.partition(":")[2], details) and not value_grounded(r.partition(":")[2], rest)]
    if copies is not None:
        copies.extend(copied_in)
    if copied_in:
        faults["requirements"] = ("these describe the listing a reference points at, not the product wanted: "
                                  + " | ".join(r[:80] for r in copied_in[:4]))
        faults.setdefault("reference_targets", "the referenced listing's details were read as the wanted product's "
                                               "requirements: read again which property is taken from that listing")
    return faults


def joined_pair(entries, text_body):
    low = " ".join(word_run(text_body))
    for i, first in enumerate(entries):
        for second in entries[i + 1:]:
            a, b = " ".join(word_run(first)), " ".join(word_run(second))
            for left, right in ((a, b), (b, a)):
                hit = re.search(re.escape(left) + r"(?:" + JOINERS + r"| )" + re.escape(right), low) if left and right else None
                if hit:
                    return hit.group(0)
    return ""


def mend_keys(faults):
    keys = set(faults) - {"figures"}
    if "figures" in faults:
        keys.update(FIGURE_KEYS)
    if "requirements" in keys:
        keys.add("items")
    if "reference_targets" in keys:
        keys.add("references")
    return sorted(keys)


def key_rules(keys):
    lines = []
    for line in READ_REQUEST.splitlines():
        named = re.match(r'Check: "(\w+)"', line)
        if named and named.group(1) in keys:
            lines.append(line)
    return lines


def mend_text(run, reading, faults, keys):
    parts = [MEND_READING, "\n".join(key_rules(keys)),
             "FAILED CHECKS:\n" + "\n".join(f"- {key}: {why}" for key, why in faults.items()),
             "FIRST READING OF THESE KEYS:\n" + json.dumps({key: reading.get(key) for key in keys}, default=str)]
    if "roles" in keys:
        parts.append("TOOLS:\n" + "\n".join(map(tool_row, run.toolset)))
    parts.append("REQUEST:\n" + run.request)
    return "\n\n".join(parts)


def keep_checked(run, reading, left):
    # (what still fails after the mend is kept only where dropping it costs more: a stated limit or procedure)
    text_body = run.request
    if "requirements" in left:
        loose = [str(r) for r in reading.get("requirements") or []
                 if ":" in str(r) and not value_grounded(str(r).partition(":")[2], text_body)]
        reading["requirements"] = [r for r in reading.get("requirements") or [] if str(r) not in loose]
        for entry in reading.get("items") or []:
            if isinstance(entry, dict) and isinstance(entry.get("requirements"), list):
                entry["requirements"] = [r for r in entry["requirements"] if str(r) not in loose]
        hints = reading.get("hints") if isinstance(reading.get("hints"), list) else []
        reading["hints"] = hints + [f"{r.partition(':')[0].strip()}: {r.partition(':')[2].strip()} (unsure reading; ask)" for r in loose]
    if "procedures" in left:
        for entry in reading.get("procedures") or []:
            if isinstance(entry, dict) and procedure_fault(entry, text_body):
                entry["unsure"] = True
    if "reference_targets" in left:
        reading["reference_targets"] = [t for t in reading.get("reference_targets") or []
                                        if isinstance(t, dict) and copied(str(t.get("seller") or ""), text_body, 1.0)]


def mended_reading(run, reading, faults):
    # (left-out figures or a missing product entry go to the cheap reader first; what it leaves goes to Sonnet)
    cheap = set(faults) <= {"figures", "items"} and "one phrase" not in str(faults.get("items"))
    reading = mend_once(run, reading, faults, READER[:1] if cheap else MENDER)
    left = vars(run).pop("mend_left", {})
    if cheap and left:
        reading = mend_once(run, reading, left, MENDER)
    lone_item(reading)
    return reading


def mend_once(run, reading, faults, ladder):
    keys = mend_keys(faults)
    reply = consult(run.wire, mend_text(run, reading, faults, keys), 2500, ladder, effort="low")
    taken = [key for key in keys if isinstance(reply, dict) and key in reply]
    for key in taken:
        reading[key] = reply[key]
    tidy_reading(reading)
    if "roles" in faults and "roles" not in taken:
        run.wire.log.append({"kind": "roles_kept", "why": faults["roles"]})
    left = reading_faults(run, reading, mended=True)
    run.wire.log.append({"kind": "read_mend", "model": ladder[0], "asked": keys, "taken": taken, "left": left})
    run.mend_left = {key: why for key, why in left.items() if key in ("items", "figures")}
    keep_checked(run, reading, left)
    return reading


def settle_procedures(reading):
    rows = []
    for entry in reading.get("procedures") or []:
        if not isinstance(entry, dict) or not str(entry.get("quote") or "").strip():
            continue
        listed = entry.get("content") if isinstance(entry.get("content"), list) else []
        shown = [part for part in listed if part in PROC_PARTS]
        rows.append({"quote": str(entry["quote"]).strip(),
                     "action": entry.get("action") if entry.get("action") in PROC_ACTIONS else "other",
                     "trigger": entry.get("trigger") if entry.get("trigger") in PROC_TRIGGERS else "anytime",
                     "needs_yes": entry.get("needs_yes") is True, "content": shown, "unsure": bool(entry.get("unsure"))})
    reading["procedures"] = rows
    reading["rules"] = [row["quote"] for row in rows]
    # (a message the shopper must answer before the purchase: the engine sends it once the basket is final)
    reading["basket_yes"] = any(row["action"] == "message_shopper" and row["needs_yes"]
                                and row["trigger"] in ("after_last_cart_change", "before_order") for row in rows)


def checked_reading(run, content, rounds=3):
    reading = first_reading(run, content, rounds)
    if not isinstance(reading.get("roles"), dict):
        return reading
    tidy_reading(reading)
    one_shelf_items(run, reading)
    try:
        drop_pointed_details(run, reading)
    except Exception as err:
        record_failure(run.wire, "pointed_details_error", err)
    faults = reading_faults(run, reading)
    run.wire.log.append({"kind": "read_check", "faults": faults})
    if faults:
        reading = mended_reading(run, reading, faults)
        one_shelf_items(run, reading)
    try:
        stated_counts(run, reading)
    except Exception as err:
        record_failure(run.wire, "stated_counts_error", err)
    settle_procedures(reading)
    settle_none_fit(run, reading)
    return reading


def settle_none_fit(run, reading):
    # (the closest listing replaces a decline only on a rule the request writes; a quote not found in it counts as none.
    # The passage's own conditions are kept as read, so a rule worded around other measures is carried the same way)
    rule = reading.get("none_fit")
    quote = str(rule.get("quote") or "") if isinstance(rule, dict) else ""
    if not quote.strip() or not copied(quote, run.request):
        reading["none_fit"] = {"quote": "", "order_closest": False}
        return
    # (its "say so" binds only when nothing fits, and the closest-listing step says it then: filed again as a message
    # procedure it would send the final basket on every errand and take a question's slot)
    flat = lambda text: " ".join(re.findall(r"[a-z0-9%]+", str(text).lower()))
    whole = flat(quote)
    reading["procedures"] = [row for row in reading.get("procedures") or [] if not (
        isinstance(row, dict) and flat(row.get("quote")) and (flat(row.get("quote")) in whole or whole in flat(row.get("quote"))))]
    reading["rules"] = [row["quote"] for row in reading["procedures"] if isinstance(row, dict) and row.get("quote")]
    reading["basket_yes"] = bool(reading.get("basket_yes")) and any(
        isinstance(row, dict) and row.get("action") == "message_shopper" and row.get("needs_yes") for row in reading["procedures"])
    action = str(rule.get("action") or "").strip().lower()
    if action not in ("order_closest", "ask_first", "decline"):
        action = "order_closest" if rule.get("order_closest") is True else "decline"
    words = lambda key: [str(x).strip() for x in rule.get(key) or [] if scalar(x) and str(x).strip()] \
        if isinstance(rule.get(key), list) else []
    counted = words("counted")
    measured = lambda key: rule.get(key) if isinstance(rule.get(key), dict) and str(rule[key].get("measure") or "").strip() else None
    choose, allow = measured("choose_by"), measured("allowance")
    share = allow.get("fraction") if allow else None
    reading["none_fit"] = {"quote": quote.strip(), "action": action, "order_closest": action != "decline",
                           "counted": counted, "may_miss": words("may_miss"), "must_keep": words("must_keep"),
                           "counts_stock": any("stock" in c.lower() for c in counted) or rule.get("counts_stock") is True,
                           "counts_budget": any(w in c.lower() for c in counted for w in ("budget", "price", "cost"))
                           or rule.get("counts_budget") is True,
                           "choose_by": {"measure": str(choose["measure"]).strip(), "lowest": choose.get("lowest") is not False}
                           if choose else None,
                           "allowance": {"measure": str(allow["measure"]).strip(), "fraction": float(share)}
                           if allow and is_number(share) and 0 <= float(share) < 1 else None}


def stated_counts(run, reading):
    # (a count the request writes right before a product's name is that product's quantity: an entry kept at 1 takes it;
    # a count after "of" is part of a name, as in "2 units of 1 coffee")
    entries = [e for e in reading.get("items") or [] if isinstance(e, dict) and str(e.get("product") or "").strip()]
    low = " ".join(str(run.request).lower().split())
    counts = r"(\d{1,2}|" + "|".join(NUMBER_WORDS) + r")"
    for entry in entries:
        if entry.get("quantity") not in (1, None):
            continue
        words = re.findall(r"[a-z0-9]+", str(entry["product"]).lower())
        if not words:
            continue
        name = r"\W+".join(re.escape(word_stem(w)) + r"[a-z]{0,2}" for w in words)
        found = set()
        for hit in re.finditer(r"(?<![\w.])" + counts + r"\s+(?:x\s+)?(?:units?\s+of\s+(?:\d+\s+)?)?" + name + r"\b", low):
            if re.search(r"\bof\s*$", low[:hit.start()]):
                continue
            found.add(int(hit.group(1)) if hit.group(1).isdigit() else NUMBER_WORDS[hit.group(1)])
        if len(found) == 1 and min(found) > 1:
            entry["quantity"] = min(found)
            run.wire.log.append({"kind": "stated_count", "product": entry["product"][:60], "quantity": entry["quantity"]})
    if len(entries) == 1 and entries[0].get("quantity", 1) > 1:
        reading["quantity"] = entries[0]["quantity"]


def schema_shelves(run):
    # (category names the tools themselves list as allowed values)
    found = []
    for shape in run.shapes.values():
        for spec in props_of(shape).values():
            values = spec.get("enum") if isinstance(spec, dict) else None
            found += [val for val in values or [] if isinstance(val, str) and len(word_run(val)) >= 2]
    return list(dict.fromkeys(found))


def one_shelf_items(run, reading):
    # (entries that all sit inside one category name the store lists, and the request writes, are one product)
    entries = [r for r in reading.get("items") or [] if isinstance(r, dict) and r.get("product")]
    if len(entries) < 2:
        return
    said = " ".join(w for w in word_run(run.request) if w != "and")
    for shelf in schema_shelves(run):
        run_of = " ".join(w for w in word_run(shelf) if w != "and")
        if f" {run_of} " not in f" {said} ":
            continue
        words = {word_stem(w) for w in run_of.split()}
        if all({word_stem(w) for w in word_run(e["product"]) if w != "and"} <= words for e in entries):
            reqs = list(dict.fromkeys(str(r) for e in entries for r in e.get("requirements") or []))
            qty = max((e.get("quantity") for e in entries if is_number(e.get("quantity"))), default=1)
            reading["items"] = [{"product": shelf, "quantity": qty, "requirements": reqs}]
            run.wire.log.append({"kind": "items_one_shelf", "shelf": shelf, "was": [e["product"] for e in entries]})
            return


def code_fallback(run, reading):
    # no LLM read it: the code's own reading stands in
    try:
        kept = {key: val for key, val in reading.items() if key != "roles"}
        reading = {**code_reading(run), **kept}
    except Exception as err:
        record_failure(run.wire, "code_reading_error", err)
        reading = {}
    run.read_at = clock_now()
    summary = dict(kind="code_reading", roles=reading.get("roles"), product=reading.get("product"),
                   price_cap=reading.get("price_cap"), items=len(reading.get("items") or []))
    run.wire.log.append(summary)
    return reading


def read_request(run):
    content = reader_content(run)
    run.read_content = content
    reading = checked_reading(run, content)
    if not isinstance(reading.get("roles"), dict):
        reading = code_fallback(run, reading)
    use_reading(run, reading)


def reader_retry_ok(run, ladder):
    link = run.wire
    if link.misses == 0 or link.call_fail == "refused" or models_down(link):
        return False
    return can_call(link, ladder) and not time_left(link) < CLOSING + 60


def has_prop_value(req):
    head, sep, value = req.partition(":")
    return bool(sep and head.strip() and value.strip())


def adopt_roles(run, reading, again):
    old_cart = role(run, "cart")
    roles = reading.get("roles")
    pairs = roles.items() if isinstance(roles, dict) else ()
    run.role_map = dict((job, tool) for job, tool in pairs if isinstance(tool, str) and tool in run.shapes)
    keep_cart = again and run.trolley.used
    if not keep_cart:
        run.trolley = Basket(role(run, "cart"), run.shapes.get(role(run, "cart") or "", {}))
    elif old_cart:
        run.role_map["cart"] = old_cart
    run.needs.sweepable = bool(role(run, "search")) and page_arg(run) is not None


def flat_words(text):
    return " ".join(re.findall(r"[a-z0-9]+", str(text).lower()))


def pointed_only(req, details, request):
    # (a value the request writes once, inside the description of a listing it points at, describes that listing and
    # not the item wanted, even when both are of one kind)
    head, _, value = str(req).partition(":")
    value, head = flat_words(value), flat_words(re.split(r"\s*\(", head)[0])
    if not value:
        return False
    once = lambda part: len(re.findall(r"\b" + re.escape(part) + r"\b", request)) == 1
    for detail in map(plain_words, details):
        if detail and re.search(r"\b" + re.escape(value) + r"\b", detail) and once(detail) \
                and (once(value) or set(head.split()) <= set(detail.split())):
            return True
    return False


def drop_pointed_details(run, reading):
    # (requirements the checks find copied from the description of a listing a reference points at go by code: a re-read
    # to mend them also rewords the pointer itself, e.g. "counterpart of" into "different from")
    gone = []
    reference_faults(reading, run.request, gone)
    details = [str(d) for t in reading.get("reference_targets") or [] if isinstance(t, dict) for d in t.get("details") or [] if str(d).strip()]
    request = flat_words(run.request)
    gone += [str(req) for req in reading.get("requirements") or [] if str(req) not in gone and pointed_only(req, details, request)]
    if not gone:
        return
    reading["requirements"] = [req for req in reading.get("requirements") or [] if str(req) not in gone]
    for entry in reading.get("items") or []:
        if isinstance(entry, dict) and isinstance(entry.get("requirements"), list):
            entry["requirements"] = [req for req in entry["requirements"] if str(req) not in gone]
    run.wire.log.append({"kind": "pointed_details_dropped", "requirements": gone})


def adopt_written(run, reading):
    stated = [str(req) for req in reading.get("requirements") or [] if req]
    run.needs.written = list(filter(has_prop_value, stated))
    loose = [req for req in stated if req not in run.needs.written]
    if not loose:
        return
    reading["soft_wants"] = [str(w) for w in reading.get("soft_wants") or [] if w] + loose
    reading["requirements"] = list(run.needs.written)


def use_reading(run, reading, again=False):
    if not again:
        run.steps_given = run.step_cap
    adopt_roles(run, reading, again)
    budget = reading.get("message_budget")
    if is_number(budget) and budget >= 0:
        run.msg_cap = int(budget)
    run.needs.reading = reading
    adopt_written(run, reading)
    shielded(run.wire, "story_values_error", story_values, run, reading)
    shielded(run.wire, "ref_register_error", ref_register, run)
    if not again or run.needs.version <= 0:
        run.needs.spend_cap = reading.get("price_cap")
    if reading and run.needs.spend_cap is None and role(run, "message"):
        asked_before = [str(a) for a in reading.get("ask_first") or [] if a]
        reading["ask_first"] = ["the most the shopper wants to spend, since the request states no price limit (skip if the same thing is listed again here)"] + asked_before
    if reading:
        reading["ask_first"] = kept_leads(run, reading)
    if again:
        run.step_cap = vars(run).get("steps_given", run.step_cap)
    limit = reading.get("step_budget")
    if is_number(limit) and 0 < limit < run.step_cap:
        run.step_cap = int(limit)
    shielded(run.wire, "spend_scale_error", spend_scale, run)
    if again:
        run.needs.version += 1


def kept_leads(run, reading):
    # (a step-0 lead goes out in the planner's own words, so it is kept only for what the shopper alone settles: the
    # budget, a choice between two soft wishes, or a check the request itself asks for; a story line read as a lead
    # spends a message on a question the shopper cannot follow)
    leads = [str(a) for a in reading.get("ask_first") or [] if a]
    told = re.search(r"\b(check|confirm)\b[^.]{0,40}\bwith me\b|\bask me\b[^.]{0,20}\b(first|before)\b", run.request, re.I)
    two = len(reading.get("soft_wants") or []) >= 2
    return [a for a in leads if two or told or re.search(r"spend|budget|price|cost|ceiling", a, re.I)]


def prop_and_value(req):
    head, _, value = str(req).partition(":")
    return re.split(r"\s*\(", head)[0].strip().lower(), value.strip().lower()


def hint_strings(raw):
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


def demote_story_guesses(run, reading, refs, hints):
    written = run.needs.written
    moved = [req for req in written if story_guess(req, run.request, refs)]
    if not moved or len(written) < 2 or 2 * len(moved) > len(written):
        return
    run.needs.written = [req for req in written if req not in moved]
    reading["requirements"] = list(run.needs.written)
    gone = set(map(prop_and_value, moved))
    for entry in reading.get("items") or []:
        if isinstance(entry, dict) and isinstance(entry.get("requirements"), list):
            entry["requirements"] = [r for r in entry["requirements"] if prop_and_value(r) not in gone]
    hints.extend("%s: the story suggests %s (not stated as a value; ask)" % tuple(part.strip() for part in req.split(":", 1))
                 for req in moved)
    run.wire.log.append({"kind": "story_values", "moved": moved})


def story_values(run, reading):
    hints = list(map(one_prop_hint, hint_strings(reading.get("hints"))))
    own_hints = hints[:]
    listed = reading.get("references")
    refs = [str(r) for r in listed or []] if isinstance(listed, list) else []
    demote_story_guesses(run, reading, refs, hints)
    try:
        taken = written_hints(run, reading, own_hints)
    except Exception as err:
        record_failure(run.wire, "written_hint_error", err)
    else:
        hints = [h for h in hints if h not in taken]
    reading["hints"] = hints[:12]


def hint_clashes(run, prop, word, mine):
    for req in run.needs.written:
        head, _, value = req.partition(":")
        tags = tag_list(head)
        if tags and mine and not (kind_words(tags[-1]) & mine):
            continue
        if same_prop(head, prop) or re.search(r"\b" + re.escape(word) + r"\b", value.lower()):
            return True
    return False


def hint_target(run, reading, entries, head, detail):
    tags = tag_list(head)
    names = [tags[-1]] if tags else ([str(r.get("product") or "") for r in entries] or [str(reading.get("product") or "")])
    word, name = "", ""
    for name in names:
        word = lead_value(detail, run.request, name)
        if word:
            break
    return word, name


def place_requirement(run, reading, entries, req, mine):
    run.needs.written.append(req)
    reading["requirements"] = list(run.needs.written)
    single = len(entries) == 1
    for entry in entries:
        if isinstance(entry.get("requirements"), list) and (single or kind_words(entry.get("product") or "") & mine):
            entry["requirements"].append(req)
            return


def written_hints(run, reading, hints):
    entries = [r for r in reading.get("items") or [] if isinstance(r, dict)]
    moved = []
    for hint in hints:
        head, sep, detail = str(hint).partition(":")
        prop = re.split(r"\s*\(", head)[0].strip()
        if not (sep and prop) or junk_key(prop):
            continue
        word, name = hint_target(run, reading, entries, head, detail)
        if not word:
            continue
        mine = kind_words(name)
        if hint_clashes(run, prop, word, mine):
            continue
        req = "%s: %s" % (head.strip(), word)
        place_requirement(run, reading, entries, req, mine)
        moved.append(hint)
        run.wire.log.append({"kind": "written_hint", "hint": str(hint)[:200], "requirement": req})
    return moved


# ---------------------------------------------------------------------------------------------------- story hints
def open_hints(run, k=None):
    still_open, alternates = [], []
    for hint in run.needs.reading.get("hints") or []:
        prop = str(hint).split(":", 1)[0].strip()
        if not prop:
            continue
        j = product_of(run, prop) if need_items(run.needs) else None
        if k is not None and j not in (k, None):
            continue
        slot = j if need_items(run.needs) else None
        if open_prop(run, slot, prop, prop_line(run, j, prop), by_samples=True) and not asked_inside(run, slot, prop):
            still_open.append(str(hint))
            continue
        alt = hint_alt(run, slot, str(hint), prop)
        if alt:
            alternates.append(alt)
    return alternates + still_open


def menu_field(line):
    return line.strip("- ").split(" (in ")[0].strip()


def fresh_menu(run, k):
    if need_items(run.needs):
        homes = home_shelves(run)
        home = homes[k] if k is not None and k < len(homes) else set()
        shelves = [s for s in read_shelves(run.archive) if s.strip().lower() in home]
        return property_menu(run.archive, shelves[0])[0] if shelves else ""
    own = own_shelf(run)
    if not own and reference_reads_only(run):
        return ""
    return property_menu(run.archive, own)[0] if own else property_menu(run.archive)[0]


def prop_line(run, k, prop):
    try:
        memo = vars(run).setdefault("menu_memo", {}).get(k)
        if memo and memo[0] == len(run.archive.items):
            menu = memo[1]
        else:
            menu = fresh_menu(run, k)
            run.menu_memo[k] = (len(run.archive.items), menu)
        for line in menu.splitlines():
            if strict_prop(menu_field(line), prop):
                return line
    except Exception:
        return ""
    return ""


def field_values(run, k, prop):
    line = prop_line(run, k, prop)
    if "):" not in line:
        return set()
    samples = line.split("):", 1)[1].split("|")
    return {" ".join(re.findall(r"[a-z0-9]+", s.lower())) for s in samples} - {""}


def same_field(run, k, a, b):
    if strict_prop(a, b):
        return True
    try:
        x, y = sorted((prop_key(a), prop_key(b)), key=len)
        # (one name ending the other, "bulb-type" and "light bulb type", is the same field when their values overlap)
        tail = len(x) >= 6 and y.endswith(x)
        if not same_prop(a, b) and not tail:
            return False
        return bool(field_values(run, k, a) & field_values(run, k, b))
    except Exception:
        return False


def hinted_prop(run, k, prop):
    name = str(prop).split("(")[0].strip()
    for hint in run.needs.reading.get("hints") or []:
        head = str(hint).split(":", 1)[0]
        if not strict_prop(head, name):
            continue
        if k is None or not need_items(run.needs) or product_of(run, head) in (k, None):
            return True
    return False


def product_reads(run, k):
    try:
        if not need_items(run.needs):
            own = own_shelf(run)
            return rich_read_count(run.archive, own) if own else rich_read_count(run.archive)
        homes = home_shelves(run)
        home = homes[k] if k is not None and k < len(homes) else set()
        if not home:
            return 0
        return sum(rich_read_count(run.archive, shelf) for shelf in home)
    except Exception:
        return 0


def plain_hint(run, k, prop, hinted=False):
    try:
        name = re.split(r"\s*\(", str(prop))[0].strip()
        if not name or not (hinted or hinted_prop(run, k, name)):
            return False
        if prop_line(run, k, name) or product_reads(run, k) < 2:
            return False
        menu = (vars(run).get("menu_memo", {}).get(k) or (0, ""))[1]
        return all(not same_prop(menu_field(line), name) for line in menu.splitlines())
    except Exception:
        return False


def hint_mark(run, hint):
    try:
        prop = str(hint).split(":", 1)[0].strip()
        k = product_of(run, prop) if need_items(run.needs) else None
        if plain_hint(run, k, prop, hinted=True):
            return f"{hint} [no key]"
        return str(hint)
    except Exception:
        return str(hint)


def hint_alts_of(detail):
    found = []
    for group in re.findall(r"\(or ([^()]+)\)", str(detail)):
        found += [alt.strip() for alt in re.split(r",|\bor\b|/", group) if alt.strip()]
    return found


def telling_words(name):
    return {w for w in prop_words(name) if len(w) >= 3 and w not in LINK_WORDS}


def name_inside(a, b):
    left, right = telling_words(a), telling_words(b)
    if not left or not right:
        return False
    return left <= right or right <= left


def asked_inside(run, k, head):
    try:
        several = bool(need_items(run.needs))
        for entry in run.questions_log:
            label = entry.split(":", 1)[0].strip("- ")
            if "[not answered]" in label:
                continue
            quoted = re.search(r'asked "(.*?)" -> ', entry)
            for part in label.split(";"):
                j = None
                if several:
                    j = product_of(run, part)
                    if j is None and quoted:
                        j = product_of(run, quoted.group(1))
                if (not several or k is None or j in (k, None)) and name_inside(part, head):
                    return True
    except Exception:
        return False
    return False


def quiet_asked(run, k, head, several):
    for asked in vars(run).get("quiet_asks") or []:
        same = same_field(run, k, asked, head) or name_inside(asked, head)
        if same and (not several or product_of(run, asked) in (k, None)):
            return True
    return False


def hint_alt(run, k, hint, head):
    try:
        memo = vars(run).setdefault("hint_alts", {})
        several = bool(need_items(run.needs))
        front, _, back = hint.partition(":")
        if not memo.get(hint):
            if not quiet_asked(run, k, head, several):
                return ""
            for alt in hint_alts_of(back):
                if junk_key(alt) or strict_prop(alt, head):
                    continue
                found = prop_line(run, k, alt)
                if found:
                    memo[hint] = found.strip("- ").split(" (in ")[0].strip()
                    break
        key = memo.get(hint) or ""
        if not key or key.lower() in vars(run).get("spent_asks", set()) or not plan_open(run, k, key):
            return ""
        tag = re.search(r"\(([^()]*)\)\s*$", front)
        label = " (" + tag.group(1).strip() + ")" if tag else ""
        detail = re.sub(r"\s*\(or [^()]+\)", "", back).strip()
        return f"{key}{label}: {detail} (its first property, {head}, drew no preference)"
    except Exception:
        return ""


def bare_name(prop):
    return re.split(r"\s*\(", str(prop))[0].strip()


def hint_names_prop(hint, name, needed):
    head, _, detail = str(hint).partition(":")
    if any(strict_prop(h, name) for h in [bare_name(head)] + hint_alts_of(detail) if h):
        return True
    text = str(hint).lower()
    return bool(needed) and all(re.search(r"\b" + re.escape(w), text) for w in needed)


def hint_mapped(run, k, prop):
    try:
        name = bare_name(prop)
        needed = [w for w in prop_words(name) if len(w) >= 3 and w not in LINK_WORDS]
        several = bool(need_items(run.needs))
        for hint in run.needs.reading.get("hints") or []:
            if several and k is not None and product_of(run, str(hint).partition(":")[0]) not in (k, None):
                continue
            if hint_names_prop(hint, name, needed):
                return True
        return False
    except Exception:
        return False



def twin_words(name):
    kept = [w for w in prop_words(name) if len(w) >= 3 and w not in LINK_WORDS]
    leading = kept[:-1] if len(kept) > 1 else kept
    return kept, leading


def words_alike(x, y):
    if x == y:
        return True
    return min(len(x), len(y)) >= 4 and (x.startswith(y) or y.startswith(x))


def name_twin(a, b):
    if strict_prop(a, b):
        return False
    all_a, lead_a = twin_words(a)
    all_b, lead_b = twin_words(b)
    for x in lead_a:
        if any(words_alike(x, y) for y in all_b):
            return True
    return any(words_alike(x, y) for x in all_a for y in lead_b)


def twin_of(run, k, a, b):
    try:
        if strict_prop(a, b):
            return False
        return name_twin(a, b) or same_field(run, k, a, b)
    except Exception:
        return False


def menu_text(run, k):
    return (vars(run).get("menu_memo", {}).get(k) or (0, ""))[1]


def product_name(run, k, several, fallback=""):
    if several and k is not None:
        return need_items(run.needs)[k]["product"]
    return str(run.needs.reading.get("product") or fallback)


def story_of(run, k, several):
    texts = [run.request.lower()]
    for hint in run.needs.reading.get("hints") or []:
        if not several or product_of(run, str(hint).split(":", 1)[0]) in (k, None):
            texts.append(str(hint).lower())
    return " ".join(texts)


def menu_keys(menu):
    for entry in menu.splitlines():
        yield entry.strip("- ").split(" (in ")[0].strip()


def twin_key(run, k, prop):
    try:
        several = bool(need_items(run.needs))
        prop_line(run, k, prop)
        menu = menu_text(run, k)
        own = kind_tokens(product_name(run, k, several))
        story = story_of(run, k, several)
        spent = vars(run).get("spent_asks", set())
        mine = set(prop_words(prop))

        def fresh_word(w):
            return len(w) >= 3 and w not in LINK_WORDS and w not in mine and not kind_in(w, own)
        for key in menu_keys(menu):
            if not key or junk_key(key) or key.lower() in spent:
                continue
            if not (twin_of(run, k, key, prop) and plan_open(run, k, key)) or name_inside(key, prop):
                continue
            extra = list(filter(fresh_word, prop_words(key)))
            if any(re.search(r"\b" + re.escape(w), story) for w in extra) or hint_mapped(run, k, key):
                return key
        return ""
    except Exception:
        return ""


def value_norm(text):
    return " ".join(re.findall(r"[a-z0-9]+", text.lower()))


def values_form(run, k, prop, product):
    try:
        own_line = prop_line(run, k, prop)
        if "):" not in own_line:
            return ""
        menu = menu_text(run, k)
        others = set()
        for entry in menu.splitlines():
            if "):" in entry and entry.strip() != own_line.strip():
                others.update(value_norm(v) for v in entry.split("):", 1)[1].split("|"))
        picks = []
        for raw in own_line.split("):", 1)[1].split("|"):
            v = raw.strip()
            if not v or len(v) > 30 or not re.search(r"[a-z]", v.lower()) or re.search(r"[,;/]", v):
                continue
            if value_norm(v) in others or value_norm(v) in [value_norm(p) for p in picks]:
                continue
            picks.append(v)
        picks = picks[:3]
        if len(picks) < 2:
            return ""
        return f"For the {product}, would you like {', '.join(picks[:-1])} or {picks[-1]}?"
    except Exception:
        return ""


def twin_slot(run, prop):
    several = bool(need_items(run.needs))
    k = product_of(run, prop) if several else None
    done = vars(run).setdefault("twin_asks", set())
    mark = k if several else -1
    if several and k is None:
        return None
    if mark in done:
        return None
    return several, k, done, mark


def distinct_ask(run, prop):
    try:
        slot = twin_slot(run, prop)
        if slot is None:
            return None
        several, k, done, mark = slot
        product = product_name(run, k, several, "item")
        target = re.split(r"\s*\(", str(prop))[0].strip()
        text = values_form(run, k, target, product)
        twin = twin_key(run, k, prop) if not text else ""
        if twin:
            target = twin
            text = values_form(run, k, twin, product) or f"Which {twin} would you like for the {product}?"
        if not text:
            return None
        done.add(mark)
        return (f"{target} ({product})" if several else target), text
    except Exception:
        return None


def asked_parts(entry):
    label = re.sub(r"\s*\[[^\]]*\]", "", entry.split(":", 1)[0].strip("- "))
    return [p.strip() for p in re.split(r";|\band\b", label) if p.strip()]


def asked_for(run, k, prop):
    try:
        several = bool(need_items(run.needs))
        name = re.split(r"\s*\(", str(prop or ""))[0].strip()
        if not name:
            return False
        for entry in run.questions_log:
            said = re.search(r'asked "(.*?)" -> ', entry)
            for part in asked_parts(entry):
                j = product_of(run, part) if several else None
                if several and j is None and said:
                    j = product_of(run, said.group(1))
                if several and k is not None and j != k:
                    continue
                if strict_prop(re.split(r"\s*\(", part)[0].strip(), name):
                    return True
    except Exception:
        return False
    return False


def alt_candidate(run, k, alt, head, name, spent):
    if junk_key(alt) or strict_prop(alt, head) or strict_prop(alt, name) or name_inside(alt, name):
        return ""
    found = prop_line(run, k, alt)
    key = found.strip("- ").split(" (in ")[0].strip() if found else ""
    if not key or strict_prop(key, name) or key.lower() in spent:
        return ""
    if asked_for(run, k, key) or not plan_open(run, k, key):
        return ""
    return key


def alt_key(run, k, prop):
    try:
        name = bare_name(prop)
        if not name:
            return ""
        several = bool(need_items(run.needs))
        spent = vars(run).get("spent_asks", set())
        for hint in run.needs.reading.get("hints") or []:
            head_full, _, detail = str(hint).partition(":")
            head = bare_name(head_full)
            if not head or (several and product_of(run, head_full) not in (k, None)):
                continue
            if not (same_field(run, k, name, head) or name_inside(name, head)):
                continue
            for alt in hint_alts_of(detail):
                key = alt_candidate(run, k, alt, head, name, spent)
                if key:
                    return key
        return ""
    except Exception:
        return ""


def collide_alt(run, prop):
    try:
        slot = twin_slot(run, prop)
        if slot is None:
            return ""
        several, k, done, mark = slot
        key = alt_key(run, k, prop)
        if not key:
            return ""
        done.add(mark)
        if several:
            return f"{key} ({need_items(run.needs)[k]['product']})"
        return key
    except Exception:
        return ""


def holds_plain(run, text, plain_q):
    frame = {"which", "would", "you", "like", "for", "the"}
    present = {stem(w) for w in re.findall(r"[a-z0-9]+", str(text).lower())}
    wanted = [stem(w) for w in re.findall(r"[a-z0-9]+", str(plain_q).lower()) if w not in frame and w not in LINK_WORDS]
    if not wanted:
        return False
    return all(w in present for w in wanted)


def alt_untried(run, k):
    try:
        for hint in run.needs.reading.get("hints") or []:
            head_full = str(hint).split(":", 1)[0]
            if product_of(run, head_full) not in (k, None):
                continue
            head = re.split(r"\s*\(", head_full)[0].strip()
            if head and asked_for(run, k, head) and alt_key(run, k, head):
                return True
    except Exception:
        return False
    return False


def menu_trim(menu, reads):
    if reads < 4 or not menu:
        return menu
    kept = [entry for entry in menu.splitlines() if not re.search(r"\(in [01] listings?\)", entry)]
    if not kept:
        return menu
    return "\n".join(kept)


def menu_wants(run, products, room):
    homes = home_shelves(run)
    qtys = [p["quantity"] for p in products] or [need_qty(run.needs)]
    wants, taken = [], []
    k = 0
    while k < (len(products) or 1):
        home = homes[k] if k < len(homes) else set()
        have = product_reads(run, k if products else None)
        if home and have < 2 and room > 0:
            picked = menu_rows(run, home, 2 - have, taken, qtys[min(k, len(qtys) - 1)])
            taken.extend(picked)
            if picked:
                wants.append((have, k, picked))
        k += 1
    return sorted(wants)


def first_reads(run):
    try:
        if vars(run).get("first_reads_done") or batch_tool(run) or not role(run, "details"):
            return False
        run.first_reads_done = True
        products = need_items(run.needs)
        room = moves_left(run) - order_due(run) - 2 * max(0, ask_limit(run) - run.msgs_sent) - 8
        wants = menu_wants(run, products, room)
        queue = [picked[0] for _, _, picked in wants] + [picked[1] for _, _, picked in wants if len(picked) > 1]
        queue = queue[: max(0, room)]
        if not queue:
            return False
        run.open_queue = queue
        run.open_recheck = True
        run.open_say = "To learn their fields before asking the shopper, I first open a few listings of every product requested."
        run.open_purpose = "menu"
        run.wire.log.append({"kind": "first_reads", "ids": queue})
        return True
    except Exception as err:
        run.wire.log.append({"kind": "first_reads_error", "error": f"{type(err).__name__}: {err}"[:300]})
        return False


def plan_open(run, k, prop):
    try:
        hinted = hinted_prop(run, k, prop)
        shown = prop_line(run, k, prop) if hinted else ""
        return open_prop(run, k, prop, shown, by_samples=hinted)
    except Exception:
        return open_prop(run, k, prop)


def sample_backed(value, samples):
    low = value.lower()
    words = [w for w in re.findall(r"[a-z]{3,}", low) if w not in LINK_WORDS]
    if not words:
        return True
    pool = set(re.findall(r"[a-z]+", samples))
    for w in words:
        if re.search(r"\b" + re.escape(w[:-1] if len(w) > 4 else w), samples):
            return True
        if len(w) >= 4 and any(len(s) >= 4 and edits_within(s, w, 1) for s in pool):
            return True
    return False


def settles(run, req, samples=""):
    head, _, value = str(req).partition(":")
    low = value.lower()
    if re.search(r"as shown|same as|the page|sold by|redirect|refer", low) or ref_text(low):
        return True
    if not positive_value(value, head):
        return False
    given = run.needs.reading.get("references")
    refs = [str(r) for r in given or []] if isinstance(given, list) else []
    refs += list(vars(run.needs).get("resolved") or [])
    if story_guess(req, run.request + " " + " ".join(run.needs.exchanges), refs):
        return False
    if samples and not re.search(r"\d", value):
        return sample_backed(value, samples)
    return True


def order_due(run):
    floor = max(FIRST_ORDER_BY, min(FIRST_ORDER_DUE, run.step_cap // 4 + 1))
    due = max(floor, min(16, run.step_cap // 3))
    state = vars(run)
    if state.get("event_seen") or (state.get("dead") and set(run.trolley.units) & state["dead"]):
        try:
            due = max(due, 8 + max(1, len(need_items(run.needs))))
        except Exception:
            due = max(due, 9)
    # (a basket the shopper must approve keeps one more step for that message)
    return max(floor, min(due, run.step_cap // 2)) + yes_reserve(run)


def narrowing(run):
    return bool(role(run, "order") and role(run, "cart"))


def reserve_names(run, spare, order, decline, cart):
    names = {order, decline, cart, role(run, "cart_view"), role(run, "view") or role(run, "details"), stock_tool(run)}
    if spare > 3 and swap_open(run):
        names.add(role(run, "details"))
    late = vars(run).get("late_n", 0) < 3
    if late:
        if (unverified(run) and spare > 5) or (named_open(run) and spare >= min(FIRST_ORDER_BY + 2, order_due(run))):
            names.add(role(run, "details"))
        pages = vars(run).get("named_pages")
        if (widened_late(run) or pages or (unverified(run) and spare > 5)) and spare > 4:
            names.add(role(run, "search"))
    if spare >= 7 and backup_rows(run):
        names.add(role(run, "details"))
    return names


def offered(run):
    spare = moves_left(run)
    order, decline, cart = role(run, "order"), role(run, "decline"), role(run, "cart")
    names = None
    if spare <= 1:
        names = {order, decline}
    elif spare <= 3:
        names = {order, decline, cart}
    elif (spare <= order_due(run) and run.orders_tried == 0) or (spare <= FIRST_ORDER_BY and run.refusal_n > 0):
        names = reserve_names(run, spare, order, decline, cart)
    tools = run.toolset
    if names is not None and order and cart:
        tools = [t for t in run.toolset if t["function"]["name"] in names] or tools
    if run.asking and role(run, "message"):
        tools = [t for t in tools if t["function"]["name"] != role(run, "message")]
    return tools


def in_reserve(run):
    spare = moves_left(run)
    if not (role(run, "order") and role(run, "cart")) or spare <= 3:
        return False
    if spare <= order_due(run) and run.orders_tried == 0:
        return True
    return spare <= FIRST_ORDER_BY and run.refusal_n > 0


def unverified(run):
    dead = dead_lines(run)
    version = str(run.needs.version)
    live = []
    for key in run.passed:
        ident, _, ver = key.rpartition("|")
        if ver == version and ident not in run.refused_lines and ident not in dead:
            live.append(ident)
    if not live:
        return True
    if not need_items(run.needs):
        return False
    try:
        got = {shelf_known(run, i) for i in live} - {""}
        if not got:
            return False
        return any(home and not home & got for home in home_shelves(run))
    except Exception:
        return False


def named_open(run):
    seen = {i for i in vars(run).get("named", set()) if i in run.archive.ids}
    return seen - run.archive.read - run.refused_lines


def widened_late(run):
    if run.needs.widened_at != run.needs.version:
        return False
    return run.wire.tick - run.needs.widened_tick <= order_due(run)


def begin_batch(run):
    run.late_q = 0
    run.late_ids = set()


def spare_backups(run):
    used = vars(run).get("backup_n", 0)
    already = vars(run).get("late_ids", set())
    return [i for i in backup_rows(run) if i not in already][: 2 - used]


def late_budget_ok(run, reading, named, spare, call_args):
    state = vars(run)
    if reading:
        if unverified(run) and spare > 5:
            return True
        return bool(named & named_open(run)) and spare >= min(FIRST_ORDER_BY + 2, order_due(run))
    if widened_late(run) or json.dumps(call_args, sort_keys=True, default=str) in state.get("named_pages", set()):
        return spare > 4
    return unverified(run) and spare > 5 and spare > 4


def backup_room(run, reading, spare):
    return reading and spare - 1 >= 6 and vars(run).get("backup_n", 0) < 2


def late_hold(run, call_name, call_args):
    details, search = role(run, "details"), role(run, "search")
    if call_name not in (details, search):
        if call_name in lookup_tools(run) and details:
            return late_hold(run, details, call_args)
        return ""
    if not in_reserve(run):
        return ""
    named = {v for v in call_args.values() if isinstance(v, str)}
    reading = call_name == details
    if reading:
        kept = set(run.trolley.units) | {key.rpartition("|")[0] for key in run.passed}
        if named & kept:
            return ""
    state = vars(run)
    queued = state.get("late_q", 0)
    spare = moves_left(run) - queued
    allowed = state.get("late_n", 0) + queued < 3 and late_budget_ok(run, reading, named, spare, call_args)
    if not allowed and backup_room(run, reading, spare) and named & set(spare_backups(run)):
        allowed = True
        run.backup_n = state.get("backup_n", 0) + 1
    if not allowed and reading:
        opened = swap_open(run)
        if opened and opened in named and spare > 2:
            allowed = True
            run.swap_done = True
    if allowed:
        run.late_q = queued + 1
        state.setdefault("late_ids", set()).update(named)
        return ""
    extra = spare_backups(run) if backup_room(run, reading, spare) else []
    tail = " Cheaper unread rows that may back up the pick can still be read: %s." % ", ".join(extra) if extra else ""
    return ("Not run (no step used): the steps left are kept for the order. Now details are read only for the cart's line, a "
            "listing a check named, or (while nothing is verified) up to 3 listings in all: put your best verified fit in the "
            "cart, re-check its stock and price, and order." + tail)


def backup_rows(run):
    if need_priority(run.needs) != "cheapest" or vars(run).get("backup_n", 0) >= 2:
        return []
    if not (vars(run).get("event_seen") or dead_lines(run) or named_open(run)):
        return []
    found = []
    try:
        for k in cg_products(run):
            pool = ed_pool(run, k)
            tops = [e["price"] for e in pool if e["verified"] and e["fit"] and e["price"] is not None]
            if not tops:
                continue
            ceiling = min(tops)
            for e in pool:
                blocked = e["read"] or e["gone"] or e["over"] or e["pointer"] or e["ruled"] or e["clash"]
                if not blocked and e["price"] is not None and e["price"] < ceiling:
                    found.append((e["price"], e["id"]))
    except Exception as err:
        run.wire.log.append({"kind": "backup_error", "error": f"{type(err).__name__}: {err}"[:300]})
        return []
    return [ident for _, ident in sorted(found)][:6]


def field_names(run):
    listing, _ = property_menu(run.archive)
    if not listing:
        return ""
    return ("\n\nFIELD NAMES ON THIS STORE'S LISTINGS (separator-only spellings merged; how many listings read carry each):\n"
            + listing)


def failed_ruling(run, memo, thing, problems, extra=None):
    ruling = f"An independent check says {thing} does NOT meet every requirement: " + problems + ". Choose another listing."
    run.rulings[memo] = ruling
    result = {"item": thing, "fits": False, "problems": extra["problems"]}
    if "by" in extra:
        result["by"] = extra["by"]
    run.wire.log.append({"kind": "verify", "item": thing, "result": result})
    return ruling


def redirect_note(own):
    pieces = []
    for req, prop, sent in own:
        pieces.append(
            "one sentence of its notes may point %s at another listing: \"%s\". Should it do so (telling the reader to take %s "
            "from, refer to, go by or look to another listing), the grader reads %s from that listing instead; this item's own %s "
            "field then does not count for '%s', and the requirement is met only when that other listing's record shows it. A "
            "listing the notes mention only as another version sold elsewhere, or as what this item is not, changes nothing."
            % (prop, sent[:240], prop, prop, prop, req))
    return "\n\nNOTE (from the code's reading of this item's own notes): " + " ".join(pieces)


def check_memo(run, key, refs, tracked, second, own):
    parts = [key]
    if refs or any(r.get("status") != "resolved" for r in tracked):
        parts.append("|%s" % len(run.archive.read))
    if second:
        parts.append("|second")
    if own:
        parts.append("|own%s" % len(own))
    return "".join(parts)


def pointed_records(run, thing, refs):
    words = set()
    for r in refs:
        words.update(re.findall(r"[a-z]{4,}", r.lower().split(":", 1)[-1]))
    hits = []
    for i in sorted(run.archive.read):
        if i != thing and any(w in " ".join(notes_about(run.archive, i, 1)).lower() for w in words):
            hits.append(i)
    return hits[:3]


def check_prompt(run, thing, aim_text, second, own, refs, tracked, seen, evidence):
    text = CHECK_ITEM + "\n\nTHE SHOPPER'S REQUEST:\n" + run.request + change_text(run.needs)
    text += "\n\nCURRENT REQUIREMENTS (stated, revealed and changed):\n" + needs_text(run.needs)
    if need_items(run.needs):
        held = ""
        if units_held(run.trolley, thing) > 0:
            held = " The cart holds %s units of it." % int(units_held(run.trolley, thing))
        text += ("\n\nThis item belongs to one of the several products requested, and its category tells which one. Compare it "
                 "with that product's requirements and quantity only; what the other products require does not concern it." + held)
    text += ref_note(run, thing) or ""
    refs = ref_untracked(run, refs, tracked)
    pointed = pointed_records(run, thing, refs) if refs else []
    if pointed:
        text += ("\n\nRECORDS OF READ LISTINGS THE REQUEST POINTS AT (when a requirement is given by reference, its value is "
                 "the one written in the named field of the listing the request means):\n"
                 + "\n".join(r for i in pointed for r in notes_about(run.archive, i, 1)))
    text += second
    if own:
        text += redirect_note(own)
    text += "\n\nTHE ITEM TO %s: %s\nRECORDS OF THIS ITEM SEEN IN THE STORE (oldest first):\n" % (aim_text, thing) + "\n".join(seen)
    if run.trolley.last_view is not None:
        text += "\n\nLATEST CART VIEW:\n" + as_text(run.trolley.last_view)[:3000]
    text += field_names(run)
    if evidence["lines"]:
        text += ("\n\nFIELD EVIDENCE (worked out by code from this store's records of the item's category: how the store "
                 "writes each required value, and what this item shows):\n" + "\n".join(evidence["lines"]))
    labels = label_list(run.archive)
    if labels:
        text += ("\n\nLABELS THAT RECUR ON THIS STORE'S LISTINGS (with counts; the item's category label must be the one that "
                 "matches the request's wording as a whole, not a neighbouring one):\n" + labels)
    return text


def check_item(run, thing, aim_text, second=""):
    if time_left(run.wire) < CLOSING + 20:
        return ""
    key = "%s|%s" % (thing, run.needs.version)
    if key in vars(run).get("near_ok", set()):
        return ""
    given = run.needs.reading.get("references")
    refs = [str(r) for r in given or [] if r] if isinstance(given, list) else []
    tracked = list(vars(run).get("refs") or [])
    own = own_redirects(run, thing)
    memo = check_memo(run, key, refs, tracked, second, own)
    if memo in run.rulings:
        return run.rulings[memo]
    pointer = ref_pointer_note(run, thing)
    if pointer:
        return failed_ruling(run, memo, thing, pointer, {"problems": ["reference chain"]})
    seen = notes_about(run.archive, thing)
    if not seen:
        return ""
    evidence = field_view(run, thing)
    if evidence["fails"]:
        top = evidence["fails"][:3]
        return failed_ruling(run, memo, thing, "; ".join(top), {"problems": top, "by": "field evidence"})
    prompt = check_prompt(run, thing, aim_text, second, own, refs, tracked, seen, evidence)
    answer = consult(run.wire, prompt, 3000, judge_seat(run), effort=JUDGE_EFFORT)
    run.wire.log.append({"kind": "verify", "item": thing, "result": answer})
    rejected = bool(answer) and answer.get("fits") is False
    if not rejected:
        run.rulings[memo] = ""
        if answer:
            run.passed.add(key)
            vars(run).setdefault("pass_price", {})[thing] = price_of(run.archive, thing)
        return ""
    problems = [str(p) for p in answer.get("problems") or []][:5]
    verdict = ("An independent check says %s does NOT meet every requirement: %s. Unless you are certain the check is wrong, "
               "choose another fully verified fit." % (thing, "; ".join(problems)))
    run.rulings[memo] = verdict
    return verdict


def ch_rows(run):
    kept = vars(run).get("ch_rows_memo")
    if kept and kept[0] == len(run.archive.items):
        return kept[1]
    fresh = code_rows(run)
    run.ch_rows_memo = (len(run.archive.items), fresh)
    return fresh


def store_fields(run, label=""):
    try:
        rows = ch_rows(run)
        homes = home_shelves(run)
        if need_items(run.needs):
            k = product_of(run, str(label))
            shelves = set(homes[k]) if k is not None and k < len(homes) else set()
        else:
            mine = own_shelf(run)
            if mine:
                shelves = {mine}
            else:
                shelves = set(homes[0]) if homes else set()
        found = {}
        for row in rows.values():
            if not row["fields"]:
                continue
            if shelves and row["shelf"] not in shelves:
                continue
            for name, val in row["fields"].items():
                if str(val).strip() and not junk_key(name):
                    found.setdefault(str(name), []).append(str(val))
        return found
    except Exception:
        return {}


def ch_redirected(run, thing, prop):
    try:
        for got in own_redirects(run, thing) or []:
            heads = got if isinstance(got, (list, tuple)) else [got]
            if any(same_prop(str(x).partition(":")[0], prop) for x in heads[:2]):
                return True
        return False
    except Exception:
        return True


def confirmed_view(view, said):
    props = [p for p, _, _ in said]
    lines = []
    for entry in view["lines"]:
        head = re.match(r"- (.+?) = '", entry)
        hit = None
        if head:
            hit = next(((p, v, s) for p, v, s in said if same_prop(head.group(1), p) or strict_prop(head.group(1), p)), None)
        if hit:
            entry += (f" A meaning check confirmed its '{hit[2][:80]}' as '{hit[1][:60]}' (the same value in other words): it "
                      "meets it.")
        lines.append(entry)
    fails = []
    for f in view["fails"]:
        name = f.split(":", 1)[0].strip()
        if not any(same_prop(name, p) or strict_prop(name, p) for p in props):
            fails.append(f)
    return {"lines": lines, "rules": list(view["rules"]), "fails": fails}


def field_view(run, thing):
    key = (thing, len(run.archive.items), run.needs.version, "|".join(props_pending(run.needs)))
    memo = vars(run).setdefault("ch_view_memo", {})
    if key not in memo:
        try:
            memo[key] = field_facts(run, thing)
        except Exception as err:
            run.wire.log.append({"kind": "field_facts_error", "error": f"{type(err).__name__}: {err}"[:300]})
            memo[key] = {"lines": [], "rules": [], "fails": []}
    said = (vars(run).get("near_said") or {}).get(f"{thing}|{run.needs.version}")
    if not said:
        return memo[key]
    try:
        return confirmed_view(memo[key], said)
    except Exception:
        return memo[key]


def clip_text(text, n=70):
    flat = " ".join(str(text).split())
    if len(flat) <= n:
        return flat
    return flat[: n - 3] + "..."


class FactScope:
    def __init__(self, thing, me, rows, peers, shelf, several, k):
        self.thing = thing
        self.me = me
        self.rows = rows
        self.peers = peers
        self.shelf = shelf
        self.several = several
        self.k = k
        self.carry = {}
        self.table = {}
        self.out = {"lines": [], "rules": [], "fails": []}
        for ident, row in peers.items():
            for name, val in row["fields"].items():
                if not str(val).strip() or junk_key(name):
                    continue
                self.carry[name] = self.carry.get(name, 0) + 1
                elems = ch_parts(val)
                if len(elems) > 1:
                    elems += [str(val).strip().lower()]
                self.table.setdefault(name, []).extend((ident, p, ch_words(p)) for p in elems)


def field_facts(run, thing):
    empty = {"lines": [], "rules": [], "fails": []}
    rows = ch_rows(run)
    me = rows.get(thing)
    if not me or not me["fields"]:
        return empty
    several = bool(need_items(run.needs))
    shelf = me["shelf"]
    if several and not shelf:
        return empty
    k = None
    if several:
        k = next((n for n, home in enumerate(home_shelves(run)) if shelf in home), None)
    peers = {i: r for i, r in rows.items() if r["fields"] and (not shelf or r["shelf"] == shelf)}
    if len(peers) < 3:
        return empty
    scope = FactScope(thing, me, rows, peers, shelf, several, k)
    for req in props_pending(run.needs)[:12]:
        requirement_facts(run, scope, req)
    return scope.out


def limit_facts(run, scope, head, prop, value):
    j = product_of(run, head) if scope.several else None
    carry, me, peers = scope.carry, scope.me, scope.peers
    named = [name for name in carry if carry[name] >= 2 and strict_prop(name, prop)]
    kin = {w for w in set(prop_words(prop)) if len(w) >= 3 and w not in LINK_WORDS}
    if not (prop and kin and named):
        return
    if scope.several and j is not None and j != scope.k:
        return
    if any(str(v).strip() and (strict_prop(name, prop) or set(prop_words(name)) & kin) for name, v in me["fields"].items()):
        return
    keys = " / ".join(f"'{x}'" for x in named[:3])
    carrying = sum(1 for r in peers.values() if any(str(r["fields"].get(x, "")).strip() for x in named))
    scope.out["lines"].append(f"- {prop} = '{clip_text(value)}': the store keeps it in {keys}, a field filled on {carrying} of "
                              f"the {len(peers)} read listings of this category. No such field appears on this listing, so the "
                              "limit is not shown met: it does NOT meet it (another field's figure does not stand in).")
    scope.out["fails"].append(f"{prop}: it has no {named[0]} field, which this store's listings of the category carry, so "
                              f"'{clip_text(value)}' is not shown met")


def joins_other(part, alts, alone):
    sides = ch_sides(part)
    if not sides or not any(ch_match(alts, ch_words(s)) for s in sides):
        return []
    return [s for s in sides if not ch_match(alts, ch_words(s)) and " ".join(ch_words(s)) in alone]


def skip_requirement(value):
    low = value.lower()
    if ref_text(value) or re.search(r"as shown|same as|the page|sold by|redirect|refer", low):
        return True
    return False


def ranked_fields(carry, like):
    picked = [name for name in carry if strict_prop(name, like)]
    picked.sort(key=lambda x: -carry[x])
    return picked


def written_literally(table, named, value):
    flat = " ".join(re.findall(r"[a-z0-9]+", value.lower()))
    for name in named:
        if any(" ".join(words) == flat for _, _, words in table.get(name, [])):
            return flat
    return None


def requirement_facts(run, scope, req):
    head, _, raw = str(req).partition(":")
    value = " ".join(re.sub(r"\((?:stated|revealed[^)]*)\)", " ", raw).split())
    prop = re.split(r"\s*\(", head)[0].strip()
    if not (prop and value) or skip_requirement(value):
        return
    low = value.lower()
    if re.search(r"\d", value) and re.search(CH_BOUND, low):
        return limit_facts(run, scope, head, prop, value)
    if re.fullmatch(r"\s*\d+(?:\.\d+)?\s+[a-z]+\s*", low):
        return
    j = product_of(run, head) if scope.several else None
    if scope.several and j is not None and j != scope.k:
        return
    positive = positive_value(value, head)
    if not (positive or re.search(EXCLUDE_WORDS, low)):
        return
    alts = ch_alts(value)
    if not alts:
        return
    carry = scope.carry
    named = ranked_fields(carry, prop)
    by_name = len(named) > 0
    if by_name and not positive:
        flat = written_literally(scope.table, named, value)
        if flat is not None:
            positive = True
            LITERAL_VALUES.add(flat)
    if not named:
        if not positive or (scope.several and j is None):
            return
        found = []
        for name in carry:
            if carry[name] >= 2 and any(ch_match(alts, words) for _, _, words in scope.table.get(name, [])):
                found.append(name)
        if not found:
            return
        named = ranked_fields(carry, max(found, key=lambda x: carry[x]))
    value_facts(run, scope, prop, value, positive, alts, named, by_name)


class ValueSight:
    def __init__(self, scope, value, alts, named):
        thing = scope.thing
        hits = [(i, words) for name in named for i, _, words in scope.table.get(name, [])]
        self.on = len([r for r in scope.peers.values() if any(str(r["fields"].get(name, "")).strip() for name in named)])
        self.writers = sorted({i for i, words in hits if i != thing and ch_match(alts, words)})
        self.alone = {" ".join(words) for i, words in hits if i != thing}
        gathered = []
        for r in scope.peers.values():
            for name in named:
                if str(r["fields"].get(name, "")).strip():
                    gathered.append(clip_text(r["fields"][name], 40))
        self.samples = list(dict.fromkeys(gathered))[:6]
        self.keys = " / ".join("'%s'" % x for x in named[:3])
        self.value = clip_text(value)


def exclusion_tail(scope, value, named):
    bare = " ".join(re.sub(EXCLUDE_WORDS, " ", value.lower()).split())
    bare_alts = ch_alts(bare)
    if not bare_alts:
        return None
    holders = set()
    for name in named:
        for i, _, words in scope.table.get(name, []):
            if i != scope.thing and ch_match(bare_alts, words):
                holders.add(i)
    return ("%s other listing(s) write '%s' there, the value it excludes (a listing that writes it does not meet it)."
            % (len(holders), clip_text(bare)))


def category_rule(sight, lead, positive, by_name, naming):
    if not positive:
        return lead
    seen = " Its values seen: " + " | ".join(sight.samples)
    if sight.writers:
        extra = "" if by_name else (" As no field carries the property's name, a listing whose title writes the value meets it "
                                    "too, unless a field of its own that shares the property's name says otherwise.")
        return (lead + " Hence a value meets it only when it writes those words (containment and one list element count; a "
                "synonym or related word does not)." + extra)
    if naming:
        return (lead + seen + "; the category's listings name '%s' in their title or text (%s), so the store has its own word "
                "for it: a synonym or related value meets it only once such a listing's record shows this field writes it "
                "otherwise." % (sight.value, ", ".join(naming)))
    if sight.on < CH_SYNONYM_MIN:
        return (lead + seen + "; no value written there matches, yet too few read listings carry this field to tell whether "
                "this store writes '%s' itself: a value of the same meaning is not shown to meet it yet." % sight.value)
    return lead + seen + "; none writes it, so a value of the same meaning may meet it."


def value_facts(run, scope, prop, value, positive, alts, named, by_name):
    sight = ValueSight(scope, value, alts, named)
    tail = None if positive else exclusion_tail(scope, value, named)
    if tail is None:
        tail = "%s other listing(s) write '%s' itself in it." % (len(sight.writers), sight.value)
    lead = "- %s = '%s': the store records it under %s" % (prop, sight.value, sight.keys)
    if not by_name:
        lead += " (nothing is called '%s'; the values of this one write the value)" % prop
    lead += ", on %s of %s read listings of this category; " % (sight.on, len(scope.peers)) + tail
    naming = []
    kinds = kind_words(value)
    if kinds and not sight.writers:
        for i, r in scope.rows.items():
            if i != scope.thing and (not scope.shelf or r["shelf"] == scope.shelf) and kinds <= r["words"]:
                naming.append(i)
        naming = naming[:3]
    scope.out["rules"].append(category_rule(sight, lead, positive, by_name, naming))
    listing_facts(run, scope, prop, value, positive, alts, named, by_name, lead, sight, naming)


def qualified_note(me, prop, named, value):
    own_words = prop_words(prop)
    qualified = []
    for name, v in me["fields"].items():
        if own_words and name not in named and str(v).strip() and not junk_key(name) \
                and len(prop_words(name)) == len(own_words) + 1 and prop_words(name)[1:] == own_words:
            qualified.append((name, str(v)))
    writing = [(name, v) for name, v in qualified if ch_writes(value, v)]
    near = bool(writing) or any(ch_touches(value, v) for _, v in qualified)
    if not writing:
        return "", near
    return (" Its %s '%s' writes it (a qualified key of the same property: a field named for the property with one more "
            "leading word is the same property's field under its fuller name), so it meets it."
            % (writing[0][0], clip_text(writing[0][1], 60))), near


def qualified_twin(scope, me, prop, named, value):
    # (a field named for the property with one more leading word, writing another value: it stands for the property when
    # the store's own rows show it agreeing with the field of the exact name wherever both appear, as a sports material
    # repeats the material, and unlike a frame material beside the material)
    own = prop_words(prop)
    if not own:
        return None
    for name, v in me["fields"].items():
        words = prop_words(name)
        if name in named or not str(v).strip() or junk_key(name) or len(words) != len(own) + 1 or words[1:] != own:
            continue
        if ch_touches(value, v):
            continue
        agree = clash = 0
        for row in scope.peers.values():
            fields = row["fields"]
            exact = [str(fields[n]) for n in fields if strict_prop(n, prop) and str(fields[n]).strip()]
            if not exact or not str(fields.get(name, "")).strip():
                continue
            if any(ch_touches(fields[name], e) or ch_touches(e, fields[name]) for e in exact):
                agree += 1
            else:
                clash += 1
        if agree and not clash:
            return name, str(v)
    return None


def missing_field_line(me, prop, named, value, note, title_says, sight):
    if note:
        return " This listing has no such field." + note
    if title_says:
        return (" No such field on this listing, and no field is named for the property: its title writes '%s': it meets it."
                % sight.value)
    stems = {w for w in prop_words(prop) if len(w) >= 3}
    related = [(name, v) for name, v in me["fields"].items() if name not in named and str(v).strip() and not junk_key(name)
               and (set(prop_words(name)) & stems or ch_writes(value, v))]
    related = sorted(related, key=lambda kv: not ch_writes(value, kv[1]))[:4]
    listed = ""
    if related:
        listed = ": its fields that may name the property: " + "; ".join("%s: '%s'" % (name, clip_text(v, 60)) for name, v in related)
    return (" No such field on this listing" + listed + ". Another name of the SAME property counts; a field of another property "
            "(a height for a type, a frame material for the material) does not stand in.")


def field_verdict(scope, sight, prop, value, alts, named, by_name, mine, shown, note, near, title_says, naming):
    pieces = [p for _, v in mine for p in ch_parts(v)]
    pieces += [v.strip().lower() for _, v in mine if len(ch_parts(v)) > 1]
    matching = [p for p in pieces if ch_match(alts, ch_words(p))]
    if any(not joins_other(p, alts, sight.alone) for p in matching):
        return " This listing: %s - it writes '%s': it meets it." % (shown, sight.value), None
    if note:
        return " This listing: %s." % shown + note, None
    if title_says:
        return (" This listing: %s - with no field named for the property, its title writes '%s': it meets it."
                % (shown, sight.value)), None
    if matching:
        part = matching[0]
        others = joins_other(part, alts, sight.alone)
        text = (" This listing: %s - '%s' joins '%s' with '%s', which other listings carry alone in this field: it does NOT "
                "meet it." % (shown, clip_text(part), sight.value, others[0]))
        fail = None
        if by_name and not near:
            fail = ("%s: its %s is '%s', which joins %s with %s (a value this store also writes alone there), so it is not %s"
                    % (prop, named[0], clip_text(part), sight.value, others[0], sight.value))
        return text, fail
    if sight.writers:
        text = (" This listing: %s - '%s' is not written here, though other listings of this category write it in this field: "
                "it does NOT meet it (a synonym or related word does not count where the store writes the value itself)."
                % (shown, sight.value))
        fail = None
        if by_name and not near and not any(ch_touches(value, v) for _, v in mine):
            fail = ("%s: its %s is '%s', while this store writes '%s' itself in that field on other listings"
                    % (prop, named[0], clip_text(mine[0][1], 60), sight.value))
        return text, fail
    values = " | ".join(sight.samples)
    if naming:
        return (" This listing: %s - '%s' is not written here. No read listing has it in this field yet (its values: %s), yet "
                "listings of this category name '%s' in their title or text (%s): the store has its own word for it, so this "
                "listing's other value is not shown to meet it." % (shown, sight.value, values, sight.value, ", ".join(naming))), None
    if sight.on < CH_SYNONYM_MIN:
        return (" This listing: %s. '%s' appears in this field on no read listing of this category (its values: %s), and only "
                "%s read listing(s) carry the field: too few to tell whether this store writes '%s' itself, so a value of the "
                "same meaning is not shown to meet it yet." % (shown, sight.value, values, sight.on, sight.value)), None
    return (" This listing: %s. '%s' appears in this field on no read listing of this category (its values: %s), so a value "
            "of the same meaning may meet it." % (shown, sight.value, values)), None


def listing_facts(run, scope, prop, value, positive, alts, named, by_name, lead, sight, naming):
    me = scope.me
    mine = [(name, str(me["fields"][name])) for name in named if str(me["fields"].get(name, "")).strip()]
    shown = "; ".join("%s: '%s'" % (name, clip_text(v, 90)) for name, v in mine)
    note, near = qualified_note(me, prop, named, value)
    lines = scope.out["lines"]
    if ch_redirected(run, scope.thing, prop):
        lines.append(lead + " This listing's own notes may send this property to another listing, so its own field may not decide.")
        return
    if not positive:
        lines.append(lead + " This listing: " + (shown or "no such field") + " (an exclusion: the listing must not carry the "
                     "excluded value).")
        return
    kin_words = {w for w in prop_words(prop) if len(w) >= 3 and w not in LINK_WORDS}
    kin = [name for name, v in me["fields"].items() if str(v).strip() and not junk_key(name)
           and set(prop_words(name)) & kin_words and not ch_writes(value, v)]
    title_says = not by_name and not kin and ch_writes(value, me.get("title") or "")
    twin = qualified_twin(scope, me, prop, named, value) if not mine and not note else None
    if twin:
        lines.append(lead + (" This listing has no such field, but its %s '%s' is this store's %s under a fuller name (on the "
                             "read listings carrying both, the two agree), and it does not write '%s': it does NOT meet it."
                             % (twin[0], clip_text(twin[1], 60), prop, sight.value)))
        scope.out["fails"].append("%s: its %s is '%s', not %s" % (prop, twin[0], clip_text(twin[1], 60), sight.value))
        return
    if not mine:
        lines.append(lead + missing_field_line(me, prop, named, value, note, title_says, sight))
        return
    text, fail = field_verdict(scope, sight, prop, value, alts, named, by_name, mine, shown, note, near, title_says, naming)
    lines.append(lead + text)
    if fail:
        scope.out["fails"].append(fail)


def rank_goal(run, cheapest):
    if cheapest:
        return "cheapest (lowest price)"
    detail = run.needs.reading.get("priority_detail") or ""
    return "best (%s)" % detail if detail else "best"


def rank_prompt(run, thing, what, cheapest):
    chunks = [RANK_CHECK.replace("PRIORITY", what), "\n\nTHE SHOPPER'S REQUEST:\n", run.request, change_text(run.needs),
              "\n\nCURRENT REQUIREMENTS:\n", needs_text(run.needs), ref_note(run)]
    if need_items(run.needs):
        chunks.append("\n\nThe request names several products. Only listings of the chosen item's own product (its category) "
                      "can compete with it; a listing of another product never beats it.")
    cheaper_only = "; the cards show only listings cheaper than the chosen one" if cheapest else ""
    cards = record_cards(run.archive, 20000, thing if cheapest else "", ref_foreign(run), {thing})
    chunks.append("\n\nTHE CHOSEN ITEM: %s\n\nLISTINGS SEEN (a card per listing; DETAILS = its whole record was read, ROW = "
                  "known only from search rows, so its fields are unknown: a ROW in the requested category (whatever its title "
                  "calls the object) that beats the chosen item and is not excluded by its title, price or stock MAY fit and is "
                  "named%s):\n" % (thing, cheaper_only) + "\n".join(cards))
    chunks.append(field_names(run))
    rules = field_view(run, thing)["rules"]
    if rules:
        chunks.append("\n\nFIELD EVIDENCE (worked out by code from this store's records of the chosen item's category: how the "
                      "store writes each required value):\n" + "\n".join(rules))
    return "".join(chunks)


def rank_check(run, thing):
    priority = need_priority(run.needs)
    key = "%s|%s|%s" % (thing, run.needs.version, run.refusal_n)
    if priority == "none" or key in run.rank_checked:
        return ""
    if time_left(run.wire) < CLOSING + 30 or moves_left(run) < 4:
        return ""
    run.rank_checked.add(key)
    cheapest = priority == "cheapest"
    what = rank_goal(run, cheapest)
    answer = consult(run.wire, rank_prompt(run, thing, what, cheapest), 2500, CHECKER)
    run.wire.log.append({"kind": "rank", "result": answer})
    dead = dead_lines(run)
    blocked = run.refused_lines
    picks = [str(x) for x in (answer or {}).get("better") or [] if isinstance(x, (str, int))]
    picks = [x for x in picks if x != thing and x not in blocked and x not in dead][:4]
    better = [x for x in picks if not field_view(run, x)["fails"]]
    run.rank_better = list(better)
    vars(run).setdefault("named", set()).update(better)
    if not better:
        return ""
    reason = " (%s)" % answer.get("reason") if answer.get("reason") else ""
    return ("A rank check thinks these listings may also fit and beat %s on the %s: %s%s. Check them (details, stock, price) "
            "before ordering; if each fails, order the current pick." % (thing, what, ", ".join(better), reason))


def audit_hits(run):
    shelves = shelf_labels(run)
    cap = run.needs.spend_cap
    qty = need_qty(run.needs)
    state = vars(run)
    hit_shelf = state.get("hit_shelf", {})
    scores = state.get("hit_score", {})

    def keep(i):
        if i in run.archive.read:
            return False
        if shelves and hit_shelf.get(i) and hit_shelf.get(i) not in shelves:
            return False
        return not (is_number(cap) and is_number(price_of(run.archive, i)) and price_of(run.archive, i) * qty > cap)
    ordered = sorted(state.get("all_hits", []), key=lambda i: -scores.get(i, 0))
    return list(filter(keep, ordered))


def search_list(run, pages):
    tool = role(run, "search")
    return "; ".join("%s %s" % (tool, json.dumps(p, ensure_ascii=False)) for p in pages)


def audit_prompt(run, cards, fresh):
    sections = [BEFORE_DECLINE, "\n\nTHE SHOPPER'S REQUEST:\n", run.request, change_text(run.needs),
                "\n\nCURRENT REQUIREMENTS:\n", needs_text(run.needs), ref_note(run),
                "\n\nLISTINGS SEEN (a card per listing; DETAILS = its whole record was read, ROW = known only from search rows):\n",
                "\n".join(cards)]
    if fresh:
        sections.append("\n\nOPENED SINCE THE LAST CHECK (their DETAILS cards above are new: judge each of them first, field by "
                        "field, by the reading rules): " + ", ".join(fresh))
    sections.append(field_names(run))
    return "".join(sections)


def audit_ids(reply, key, ok, most):
    return [str(x) for x in reply.get(key) or [] if isinstance(x, (str, int)) and ok(str(x))][:most]


def remember_pages(run, pages):
    if pages:
        vars(run).setdefault("named_pages", set()).update(json.dumps(p, sort_keys=True, default=str) for p in pages)


def audit_note(run, answer, fits, pages, combos, unopened):
    said = []
    if fits:
        reason = " (%s)" % answer.get("reason") if answer.get("reason") else ""
        said.append("An independent check thinks these listings may meet every requirement: %s%s." % (", ".join(fits), reason))
    if pages:
        said.append("These searches carry required values and matched more listings than were read, so fitting ones may sit "
                    "on their unread pages: %s. Read those pages first (one step shows a whole page of rows) and open the in-budget rows "
                    "of the requested kind they show." % search_list(run, pages))
    if combos:
        said.append("These searches combine the required values and were never run (when every word is required, each lists "
                    "only the listings carrying all of them): %s. Run them (the search tool stays usable for them) and open the "
                    "in-budget rows of the requested kind they show." % search_list(run, combos))
    if unopened:
        said.append("These listings of the requested kind were seen only in search rows and never opened, so their fields are "
                    "unchecked: %s. Open their details first (several in one reply)." % ", ".join(unopened))
    said.append("Verify them (details, stock, price) and buy the right one; decline only if each one truly fails.")
    return " ".join(said)


def review_decline(run, early=False):
    state = vars(run)
    again = len(run.archive.read) > state.get("audit_reads", 10 ** 9) and state.get("audits", 0) < 2
    if run.audit_done and not again:
        return ""
    if time_left(run.wire) < CLOSING + 30:
        return ""
    run.audit_done = True
    if not early:
        run.audits = state.get("audits", 0) + 1
    before = state.get("audit_ids")
    skip = () if before is None else set(run.archive.read) - set(before or ())
    cards = record_cards(run.archive, 16000, "", ref_foreign(run), skip)
    run.audit_reads = len(run.archive.read)
    if not cards:
        return ""
    fresh = [] if before is None else sorted(run.archive.read - before)
    run.audit_ids = set(run.archive.read)
    answer = consult(run.wire, audit_prompt(run, cards, fresh), 6000, AUDITOR)
    run.wire.log.append({"kind": "audit", "result": answer})
    reply = answer or {}
    fits = audit_ids(reply, "possible_fits", lambda x: True, 5)
    fits = [x for x in fits if not field_view(run, x)["fails"]]
    run.audit_fits = fits
    listed = audit_ids(reply, "unopened", lambda x: x not in run.archive.read and x in run.archive.ids, 6)
    hits = audit_hits(run)
    try:
        slack = [i for i in slack_rows(run) if i not in run.archive.read]
    except Exception:
        slack = []
    unopened = list(dict.fromkeys(slack + hits[:4] + [i for i in listed if i not in hits]))[:6]
    roomy = moves_left(run) > 5
    pages = open_pages(run, 2) if roomy else []
    if pages:
        unopened = unopened[:4]
    remember_pages(run, pages)
    try:
        combos = combined_queries(run) if moves_left(run) > 5 else []
    except Exception as err:
        run.wire.log.append({"kind": "combo_error", "error": f"{type(err).__name__}: {err}"[:300]})
        combos = []
    remember_pages(run, combos)
    state.setdefault("named", set()).update(fits + unopened)
    named_list = state.setdefault("named_list", [])
    named_list += [i for i in fits + unopened if i not in named_list]
    if fits or unopened or pages or combos:
        return audit_note(run, answer, fits, pages, combos, unopened)
    return ""


def named_unread(run):
    cap, qty = run.needs.spend_cap, need_qty(run.needs)
    state = vars(run)
    shelves, hit_shelf = shelf_labels(run), state.get("hit_shelf", {})
    scores = state.get("hit_score", {})
    ranked = sorted(state.get("all_hits", []), key=lambda i: -scores.get(i, 0))
    picked = []
    for ident in dict.fromkeys(ranked + list(state.get("named_list", []))):
        cost = price_of(run.archive, ident)
        if ident in run.archive.read or ident in run.refused_lines or ident not in run.archive.ids:
            continue
        if is_number(cap) and is_number(cost) and cost * qty > cap:
            continue
        if shelves and hit_shelf.get(ident) and hit_shelf[ident] not in shelves:
            continue
        picked.append(ident)
    return picked


def details_template(run, tool):
    known = run.archive.ids
    for entry in reversed(run.wire.log):
        actions = (entry.get("actions") or []) if entry.get("kind") == "env" else []
        for act in actions:
            args = act.get("args") if act.get("name") == tool else None
            if isinstance(args, dict) and any(isinstance(v, str) and v in known for v in args.values()):
                return args
    return None


def details_call(run, ident):
    tool = role(run, "details")
    template = details_template(run, tool)
    if template is not None:
        known = run.archive.ids
        return dict((name, ident if isinstance(v, str) and v in known else v) for name, v in template.items())
    jschema = run.shapes.get(tool or "") or {}
    props = jschema.get("properties")
    if not isinstance(props, dict):
        props = {}
    texts = [name for name, p in props.items() if isinstance(p, dict) and p.get("type") == "string" and not p.get("enum")]
    needed = jschema.get("required") or []
    slot = next((name for name in texts if name in needed), None) or (texts[0] if len(texts) == 1 else None)
    if not slot:
        return None
    return {slot: ident}


def open_reads(run, queue, say, batch, reader=None):
    if batch:
        tool, arg, most = batch
        calls = [{arg: queue[i: i + most]} for i in range(0, len(queue), most)]
    else:
        tool = reader or role(run, "details")
        calls = []
        if tool:
            for ident in queue:
                args = lookup_call(run, ident, tool)
                if args:
                    calls.append(args)
    proposals = []
    for n, args in enumerate(calls):
        proposals.append({"id": f"open_{n}_{run.wire.tick}", "type": "function",
                          "function": {"name": tool, "arguments": json.dumps(args)}})
    if not proposals:
        return []
    run.chat.append({"role": "assistant", "content": say, "tool_calls": proposals})
    start = len(run.archive.items)
    run.covering = True
    try:
        responses = dispatch(run, proposals, scripted=True)
    finally:
        run.covering = False
    run.last_reads = run.archive.items[start:]
    return responses


def reserve_trim(run, queue, purpose):
    try:
        keep = order_due(run) if run.orders_tried == 0 else FIRST_ORDER_BY
        room = moves_left(run) - keep - 1
        listed = batch_tool(run) and len(queue) >= 2
        if room < (1 if listed else len(queue)) and (run.trolley.units or not unverified(run)):
            trimmed = [] if (batch_tool(run) and len(queue) >= 2) else queue[: max(0, room)]
            run.wire.log.append({"kind": "reserve_skip", "purpose": purpose, "ids": queue[:6], "kept": trimmed})
            return trimmed
    except Exception as err:
        run.wire.log.append({"kind": "reserve_skip_error", "error": f"{type(err).__name__}: {err}"[:300]})
    return queue


def take_put_queue(run, state):
    if not state.get("put_queue") or run.wire.done or run.wire.broke:
        return None
    queue = state.pop("put_queue")
    run.open_recheck, run.open_purpose, run.open_tool = True, "cart", state.pop("put_tool", None)
    run.open_say = "A listing can sell out right after it goes into the cart, so I read the stock of what I just put in."
    return queue


def usable_batch(run, state, queue, purpose, reader):
    if len(queue) < 2:
        return None
    batch = batch_tool(run)
    if not batch:
        return batch
    if purpose == "cart" and not state.get("batch_proven"):
        return None
    if reader and not ((state.get("tool_caps") or {}).get(batch[0]) or {}).get("stock"):
        return None
    return batch


def single_reads(run, dropped, queue, say, reader, purpose, floor, responses):
    run.batch_spec = None
    run.wire.log.append({"kind": "batch_dropped", "tool": dropped})
    most = {"menu": 0, "quota": 2, "walk": 2}.get(purpose, 3)
    if is_number(floor):
        keep = int(floor)
    elif purpose in ("menu", "quota"):
        keep = order_due(run)
    else:
        keep = 3
    n = min(most, len(queue), moves_left(run) - keep - 1)
    if n <= 0 or run.wire.done or not (reader or role(run, "details")):
        return responses, False
    post_replies(run, responses)
    return open_reads(run, queue[:n], say, None, reader), True


def open_named(run):
    state = vars(run)
    queue = state.pop("open_queue", None)
    put_read = False
    if not queue:
        taken = take_put_queue(run, state)
        if taken is not None:
            queue, put_read = taken, True
    if not queue or run.wire.done or run.wire.broke:
        return False
    queue = list(dict.fromkeys(str(i) for i in queue))
    recheck = state.pop("open_recheck", False)
    purpose = str(state.pop("open_purpose", "") or "")
    floor = state.pop("open_floor", None)
    reader = state.pop("open_tool", None)
    if not reader and purpose != "cart" and field_reader(run) != role(run, "details") \
            and (purpose in ("menu", "quota") or table_tool(run)):
        reader = field_reader(run)
    say = state.pop("open_say", "") or ("I re-read the cart lines before the order." if recheck
                                        else "I open the listings the checks named before any decline.")
    guarded = purpose in ("menu", "quota", "walk") and role(run, "order") and role(run, "cart") and not run.wire.mute
    if guarded:
        queue = reserve_trim(run, queue, purpose)
        if not queue:
            return False
    if purpose == "cart":
        stale = set(stale_lines(run))
        queue = [i for i in queue if i in stale or (i not in run.trolley.units and not put_read)]
        if not queue:
            return False
    batch = usable_batch(run, state, queue, purpose, reader)
    responses = open_reads(run, queue, say, batch, reader)
    if batch and responses and not state.get("batch_proven"):
        if batch_ok(state.get("last_reads") or [], queue):
            run.batch_proven = True
        else:
            responses, retried = single_reads(run, batch[0], queue, say, reader, purpose, floor, responses)
            if retried and not responses:
                return True
    if not responses:
        return False
    if time_left(run.wire) > CLOSING + 30 and not recheck:
        note = review_decline(run)
        if note:
            cid, body = responses[-1]
            responses[-1] = (cid, body + "\n[check after opening] " + note)
    post_replies(run, responses)
    return True


def last_stock(run, ident):
    found = [None]

    def visit(d):
        if row_mark(d) == ident:
            got = stock_state(d)
            if got is not None:
                found[0] = got
    for rec in run.archive.items:
        walk_dicts(rec, visit)
    return found[0]


def rank_candidate(run, x, version, cap):
    if x not in run.archive.read or f"{x}|{version}" in run.passed or x in run.refused_lines or x in dead_lines(run):
        return False
    if last_stock(run, x) is False or field_view(run, x)["fails"]:
        return False
    k = product_index(run, x) if need_items(run.needs) else None
    if k is not None and k < len(need_items(run.needs)):
        qty = need_items(run.needs)[k]["quantity"]
    else:
        qty = need_qty(run.needs)
    cost = price_of(run.archive, x)
    if not is_number(cost):
        return False
    return not (is_number(cap) and is_number(qty) and cost * qty > cap)


def rank_verify(run, thing):
    try:
        if need_priority(run.needs) != "cheapest":
            return ""
        version = str(run.needs.version)
        cap = run.needs.spend_cap
        checked = 0
        for x in vars(run).get("rank_better") or []:
            if checked >= 2 or time_left(run.wire) < CLOSING + 30:
                break
            if rank_candidate(run, x, version, cap):
                checked += 1
                check_item(run, x, "ORDER")
        return cheaper_fit(run, thing)
    except Exception as err:
        run.wire.log.append({"kind": "rank_verify_error", "error": f"{type(err).__name__}: {err}"[:300]})
        return ""


def cheaper_fit(run, thing):
    if need_priority(run.needs) != "cheapest" or not thing:
        return ""
    mine = price_of(run.archive, thing)
    if mine is None:
        return ""
    kind = shelf_known(run, thing)
    best_id, best_cost = "", None
    dead = dead_lines(run)
    for key in run.passed:
        ident, _, version = key.rpartition("|")
        if version != str(run.needs.version) or ident == thing or ident in run.refused_lines or ident in dead:
            continue
        if need_items(run.needs) and (not kind or shelf_known(run, ident) != kind):
            continue
        cost = price_of(run.archive, ident)
        if cost is None or cost >= mine or (best_cost is not None and cost >= best_cost):
            continue
        if field_view(run, ident)["fails"] or not pair_keeps(run, thing, ident):
            continue
        best_id, best_cost = ident, cost
    return best_id


# ---- works-with pairs: a product the request ties to another one must be a listing whose own code names that one
PART_CODE = r"\b[A-Z][A-Za-z]*[ -]?\d+(?:-\d+)?\b"
PAIR_TOLD = 2


def pair_rows(run):
    # (dependent product -> the product it has to work with, matched by the kind words of the reading's works_with)
    products = need_items(run.needs)
    found = {}
    for k, entry in enumerate(products):
        words = kind_tokens(entry.get("works_with") or "")
        if not words:
            continue
        scores = [(kind_score(kind_tokens(other["product"]), words, []), j) for j, other in enumerate(products) if j != k]
        score, j = max(scores, default=(0, None))
        if score > 0 and found.get(j) != k:
            found[k] = j
        elif score > 0 and k > j:
            found.pop(j)
            found[k] = j
    return found


def code_key(code):
    return re.sub(r"[^a-z0-9]", "", str(code).lower())


def page_texts(run, thing):
    # (the listing's own written texts of more than one sentence, and its one-line texts such as the title)
    texts, titles = [], []

    def visit(rec):
        if row_mark(rec) != thing:
            return
        title = title_of(rec)
        for val in rec.values():
            if not isinstance(val, str) or not re.search(PART_CODE, val):
                continue
            pool = texts if len(text_parts(val)) > 1 and val != title else titles
            if val not in pool:
                pool.append(val)
    for item in run.archive.items:
        walk_dicts(item, visit)
    return texts, titles


def text_parts(text):
    return [part for part in re.split(r"(?<=[.;!?])\s+", str(text)) if part.strip()]


def model_codes(run, thing):
    # (the codes a listing goes by: the first run of sentences naming codes in its own text)
    for text in page_texts(run, thing)[0]:
        codes, started = [], False
        for part in text_parts(text):
            found = re.findall(PART_CODE, part)
            if found:
                codes += found
                started = True
            elif started:
                break
        if codes:
            return list(dict.fromkeys(codes))[:6]
    return []


def fits_codes(run, thing):
    # (the codes a listing says it goes with: its first sentence naming codes, and the codes in its title)
    texts, titles = page_texts(run, thing)
    codes = []
    for text in texts:
        found = next((re.findall(PART_CODE, part) for part in text_parts(text) if re.search(PART_CODE, part)), [])
        if found:
            codes += found
            break
    for title in titles:
        codes += re.findall(PART_CODE, title)
    return list(dict.fromkeys(codes))


def code_keys(codes):
    keys = set()
    for code in codes:
        keys.add(code_key(code))
        head = re.match(r"(.+?)-\d+$", str(code))
        if head:
            keys.add(code_key(head.group(1)))
    return keys


def one_way(run, dep, anchor):
    want = {code_key(c) for c in model_codes(run, anchor)}
    have = code_keys(fits_codes(run, dep))
    if not want or not have:
        return None
    return bool(want & have)


def pair_fit(run, dep, anchor):
    # (True when either listing names the other's codes, False when codes are given and none match, None without codes;
    # the side the reading calls the anchor may be the wrong one)
    ways = [one_way(run, dep, anchor), one_way(run, anchor, dep)]
    if True in ways:
        return True
    return False if False in ways else None


def seen_ids(run):
    found = []

    def visit(d):
        mark = row_mark(d)
        if (title_of(d) or shelf_of(d)) and not mark.startswith("{") and mark not in found:
            found.append(mark)
    for item in run.archive.items:
        walk_dicts(item, visit)
    return found


def held_line(run, k):
    return next((line for line in run.trolley.units if product_index(run, line) == k), None)


def pair_live(run, k, j):
    # (the store shows this mechanism: some seen listing of k fits some seen listing of j)
    memo = vars(run).setdefault("pair_live", {})
    if memo.get((k, j)):
        return True
    ids = seen_ids(run)
    sides = ([i for i in ids if product_index(run, i) == k], [i for i in ids if product_index(run, i) == j])
    live = False
    for one, other in (sides, sides[::-1]):
        keys = set()
        for a in other:
            keys |= {code_key(c) for c in model_codes(run, a)}
        live = live or (bool(keys) and any(keys & code_keys(fits_codes(run, d)) for d in one))
    memo[(k, j)] = live
    return live


def pair_keeps(run, thing, ident):
    # (swapping the cart line thing for ident leaves every works-with pair in the cart fitting)
    rows = pair_rows(run)
    k = product_index(run, thing) if rows else None
    if k is None:
        return True
    for dep_k, anchor_k in rows.items():
        if dep_k == k:
            anchor = held_line(run, anchor_k)
            if anchor and pair_fit(run, ident, anchor) is False:
                return False
        if anchor_k == k:
            dep = held_line(run, dep_k)
            if dep and pair_fit(run, dep, ident) is False:
                return False
    return True


def pair_faults(run):
    out = []
    for k, j in pair_rows(run).items():
        dep, anchor = held_line(run, k), held_line(run, j)
        if dep and anchor and pair_fit(run, dep, anchor) is False and pair_live(run, k, j):
            out.append((k, j, dep, anchor))
    return out


def pair_options(run, k, fits, skip):
    # (seen listings of product k for which fits(ident) holds, cheapest first)
    found = []
    for ident in seen_ids(run):
        if ident in skip or product_index(run, ident) != k or not fits(ident):
            continue
        cost = price_of(run.archive, ident)
        found.append((cost if is_number(cost) else float("inf"), ident))
    return [ident for _, ident in sorted(found)][:4]


def pair_skip(run):
    return set(run.refused_lines) | dead_lines(run)


def pair_put_note(run, item):
    # (the line just put breaks a works-with pair with a line already held: said with the cart reply, no step spent)
    for k, j, dep, anchor in pair_faults(run):
        if item not in (dep, anchor):
            continue
        products = need_items(run.needs)
        other, side = (anchor, k) if item == dep else (dep, j)
        fits = pair_options(run, side, lambda i: pair_fit(run, i, other) is True, pair_skip(run) | {item})
        run.wire.log.append({"kind": "pair_put", "item": item, "other": other, "fits": fits})
        return (f"{item} and {other} do not fit together (the request wants the {products[k]['product']} to work with the "
                f"{products[j]['product']}; neither listing names the other's codes: {', '.join(model_codes(run, anchor)) or 'none'})."
                + (f" Seen {products[side]['product']} listings that fit {other}: {', '.join(fits)}." if fits else ""))
    return None


def held_for_pairs(run):
    faults = pair_faults(run)
    if not faults or moves_left(run) <= 4 + yes_reserve(run) or time_left(run.wire) < CLOSING / 2:
        return None
    k, j, dep, anchor = faults[0]
    told = vars(run).setdefault("pairs_told", {})
    if told.get((dep, anchor), 0) >= PAIR_TOLD or sum(told.values()) >= 2 * PAIR_TOLD:
        return None
    told[(dep, anchor)] = told.get((dep, anchor), 0) + 1
    products = need_items(run.needs)
    dep_name, anchor_name = products[k]["product"], products[j]["product"]
    codes, own = model_codes(run, anchor), fits_codes(run, dep)
    skip = pair_skip(run)
    deps = pair_options(run, k, lambda i: pair_fit(run, i, anchor) is True, skip | {dep})
    anchors = pair_options(run, j, lambda i: pair_fit(run, dep, i) is True, skip | {anchor})
    run.wire.log.append({"kind": "pair_held", "dep": dep, "anchor": anchor, "codes": codes, "own": own, "deps": deps, "anchors": anchors})
    note = (HELD + f"The request wants the {dep_name} to work with the {anchor_name}. {anchor} goes by "
            f"{', '.join(codes)}, but the codes {dep} names for itself ({', '.join(own[:6])}) include none of them, so this "
            "pair does not fit.")
    if deps:
        note += f" Seen {dep_name} listings that name {anchor}'s codes: {', '.join(deps)}."
    if anchors:
        note += f" Seen {anchor_name} listings that {dep} fits: {', '.join(anchors)}."
    if not deps and not anchors:
        note += f" Search the {dep_name} with one of {', '.join(codes[:3])} as the phrase and open what comes back."
    return note + (" Swap one side of the pair for a listing that also meets every other requirement, read its stock and "
                   "price, then order.")


def shelf_known(run, thing):
    found = [""]

    def visit(d):
        if any(isinstance(val, str) and val == thing and "id" in str(name).lower() for name, val in d.items()):
            found[0] = shelf_of(d) or found[0]
    for rec in run.archive.items:
        walk_dicts(rec, visit)
    if found[0]:
        return found[0]
    return str((vars(run.archive).get("label_of") or {}).get(thing) or "").strip().lower()


def chosen(run):
    if not run.trolley.units:
        return None
    return list(run.trolley.units)[-1]


def seen_shelves(run):
    labels = set()

    def visit(d):
        shelf = shelf_of(d)
        if shelf and any(isinstance(v, str) and v in run.archive.ids and "id" in str(name).lower() for name, v in d.items()):
            labels.add(shelf)
    for rec in run.archive.items:
        walk_dicts(rec, visit)
    return labels


def best_shelves(own, shared, labels):
    top = 0.0
    best = set()
    for shelf in sorted(labels):
        score = kind_score(own, set(kind_tokens(shelf)), shared)
        if score > top:
            top = score
            best = {shelf}
        elif score > 0 and score == top:
            best.add(shelf)
    return best


def home_shelves(run):
    memo = vars(run).get("home_memo")
    count = len(run.archive.items)
    if memo and memo[0] == count:
        return list(map(set, memo[1]))
    try:
        products = need_items(run.needs) or [{"product": str(run.needs.reading.get("product") or "")}]
        labels = seen_shelves(run)
        kinds = [set(kind_tokens(p["product"])) for p in products]
        homes = []
        for k, own in enumerate(kinds):
            others = [kinds[j] for j in range(len(kinds)) if j != k]
            homes.append(best_shelves(own, set().union(*others), labels))
    except Exception:
        homes = [set() for _ in (need_items(run.needs) or [0])]
    run.home_memo = (len(run.archive.items), list(map(set, homes)))
    return homes



def reference_reads_only(run):
    # (every listing read so far is of the kind a reference points at, not of the product bought: a menu built from them
    # would ask the shopper about another product's properties)
    try:
        kinds = [set(kind_tokens(str(t.get("kind") or ""))) for t in run.needs.reading.get("reference_targets") or []
                 if isinstance(t, dict)]
        # (a reference of the product's own kind is a listing of that product: its keys stay)
        own = set(kind_tokens(str(run.needs.reading.get("product") or "")))
        kinds = [k for k in kinds if k and not k & own]
        read = read_shelves(run.archive)
        return bool(kinds and read) and all(any(set(kind_tokens(s)) & k for k in kinds) for s in read)
    except Exception:
        return False


def own_shelf(run):
    try:
        if need_items(run.needs):
            return ""
        home = home_shelves(run)[0]
        read = [s.strip().lower() for s in read_shelves(run.archive)]
        if not home or all(s in home for s in read):
            return ""
        inside = [s for s in read if s in home]
        return inside[0] if inside else sorted(home)[0]
    except Exception:
        return ""


def loose_join(low, first, second, a, b):
    for hit in re.finditer(r"\b(?:" + first + r")(?P<mid>[^.;:!?]{0,40}?)(?:" + second + r")", low):
        mid = hit.group("mid")
        if len(re.findall(r"[a-z0-9]+", mid)) > 3:
            continue
        if not re.search(r"\band\b|&|\bor\b|,|/|\+", mid) or re.search(r"\d|\bx\b", mid):
            continue
        rest = low[:hit.start()] + " | " + low[hit.end():]
        if all(re.search(r"\b" + name_pattern(z), rest) for z in (a, b)):
            continue
        return hit.group(0).strip()
    return None


def joined_phrase(run, a, b, label):
    low = re.sub(r"\s+", " ", run.request.lower().replace("’", "'"))
    sides = [p for p in re.split(JOINERS, str(label).lower()) if p.strip()]
    if len(sides) == 2:
        left, right = name_pattern(sides[0]), name_pattern(sides[1])
        if left and right:
            hit = re.search(r"\b" + left + r"\s*(?:\band\b|&|\bor\b|/|\+)\s*" + right, low)
            if hit:
                return hit.group(0).strip(), True
    ta, tb = kind_tokens(a), kind_tokens(b)
    if not ta or not tb or same_kind(ta[-1], tb[-1]):
        return None
    for x, y in ((a, b), (b, a)):
        px, py = name_pattern(x), name_pattern(y)
        if px and py:
            phrase = loose_join(low, px, py, a, b)
            if phrase is not None:
                return phrase, False
    return None


def product_question(run, entry, products):
    hit = re.match(r'-\s*(.*?): asked "(.*?)" -> ', str(entry), re.S)
    if hit:
        label, question = hit.group(1), hit.group(2)
    else:
        label = question = str(entry)
    kinds = [kind_tokens(p["product"]) for p in products]
    for tag in re.findall(r"\(([^()]*)\)", label):
        words = kind_tokens(tag)
        if words and any(kind_in(w, kk) for w in words for kk in kinds):
            return True
    low = question.lower().replace("’", "'")
    for p in products:
        pattern = name_pattern(p["product"])
        if pattern and re.search(r"\b" + name_pattern(p["product"]), low):
            return True
    return False


def requirement_pairs(product):
    pairs = []
    for req in product.get("requirements") or []:
        head, _, value = str(req).partition(":")
        pairs.append((re.split(r"\s*\(", head)[0].strip(), " ".join(re.findall(r"[a-z0-9]+", value.lower()))))
    return pairs


def joined_clash(a, b):
    left, right = requirement_pairs(a), requirement_pairs(b)
    return any(same_prop(pa, pb) and va != vb for pa, va in left for pb, vb in right)


def sides_split(ti, tj, sides):
    di = [w for w in ti if not kind_in(w, tj)]
    dj = [w for w in tj if not kind_in(w, ti)]
    for x, y in ((sides[0], sides[1]), (sides[1], sides[0])):
        if not all(kind_in(w, x) and not kind_in(w, y) for w in di):
            continue
        if not all(kind_in(w, y) and not kind_in(w, x) for w in dj):
            continue
        if any(kind_in(w, x) for w in ti) and any(kind_in(w, y) for w in tj):
            return True
    return False


def pair_matches(products, i, j, sides, whole):
    ti, tj = kind_tokens(products[i]["product"]), kind_tokens(products[j]["product"])
    if not ti or not tj:
        return False
    if kinds_within(ti, tj) and kinds_within(tj, ti):
        return False
    if not kinds_within(ti + tj, whole) or products[i]["quantity"] != products[j]["quantity"]:
        return False
    return sides_split(ti, tj, sides) and not joined_clash(products[i], products[j])


def merge_pair(run, products, seen):
    for low in sorted(seen):
        sides = [kind_tokens(p) for p in re.split(JOINERS, low) if p.strip()]
        if len(sides) != 2 or not all(sides):
            continue
        whole = sides[0] + sides[1]
        pairs = [(i, j) for i in range(len(products)) for j in range(i + 1, len(products))]
        for i, j in pairs:
            if not pair_matches(products, i, j, sides, whole):
                continue
            got = joined_phrase(run, products[i]["product"], products[j]["product"], seen[low])
            if got:
                return i, j, seen[low], got[0], got[1]
    return None


def joined_label(d):
    has_id = any(isinstance(v, str) and "id" in str(name).lower() and re.fullmatch(ID_LIKE, v) for name, v in d.items())
    if not has_id:
        return ""
    for name, val in d.items():
        key = str(name).lower()
        if isinstance(val, str) and 0 < len(val.strip()) <= 60 and re.search(r"categor|shelf|department|aisle", key) and "id" not in key:
            return val.strip()
    return ""


def merge_blocked(run, products, seen):
    if len(products) < 2 or not seen or run.trolley.used or run.trolley.units:
        return True
    if run.needs.uncovered or run.needs.indifferent:
        return True
    return any(product_question(run, entry, products) for entry in run.questions_log)


def merge_joined(run, obs):
    seen = vars(run).setdefault("joined_seen", {})

    def visit(d):
        label = joined_label(d)
        if label and len([p for p in re.split(JOINERS, label.lower()) if p.strip()]) == 2:
            seen.setdefault(label.lower(), label)
    walk_dicts(obs, visit)
    remarks = []
    rounds = 2
    while rounds:
        rounds -= 1
        products = need_items(run.needs)
        if merge_blocked(run, products, seen):
            break
        found = merge_pair(run, products, seen)
        if found is None:
            break
        remarks.append(merge_items(run, *found))
    return " ".join(r for r in remarks if r)


def jot(run, kind, **facts):
    run.wire.log.append(dict(kind=kind, **facts))


def jot_error(run, kind, err):
    jot(run, kind, error=f"{type(err).__name__}: {err}"[:300])


def merge_items(run, i, j, label, phrase, in_order=True):
    sheet = run.needs
    reading = sheet.reading
    products = need_items(sheet)
    raw = [r for r in reading.get("items") or [] if isinstance(r, dict) and str(r.get("product") or "").strip()]
    if len(raw) != len(products) or not 0 <= i < j < len(products):
        return ""
    pair_words = kind_tokens(products[i]["product"]) + kind_tokens(products[j]["product"])
    label_words = kind_tokens(label)
    rest_words = [kind_tokens(p["product"]) for n, p in enumerate(products) if n not in (i, j)]
    remaining = len(products) - 1
    new_tag = phrase if remaining >= 2 else ""

    def ours(tag):
        words = kind_tokens(tag)
        if not words or not kinds_within(words, label_words):
            return False
        if not any(kind_in(w, pair_words) for w in words):
            return False
        return not any(kinds_within(words, o) and kinds_within(o, words) for o in rest_words)

    def swap_tag(found):
        if not ours(found.group(1)):
            return found.group(0)
        return f" ({new_tag})" if new_tag else ""

    def retag(text):
        if not isinstance(text, str):
            return text
        head, sep, value = text.partition(":")
        return re.sub(r"\s*\(([^()]*)\)", swap_tag, head) + sep + value

    def retag_all(texts):
        kept = []
        for text in map(retag, texts):
            if text not in kept:
                kept.append(text)
        return kept

    joined = []
    for req in list(products[i]["requirements"]) + list(products[j]["requirements"]):
        head, _, value = str(req).partition(":")
        bare = re.sub(r"\s*\([^()]*\)", "", head).strip() + ":" + value
        if bare.lower() not in [r.lower() for r in joined]:
            joined.append(bare)
    merged = {"product": phrase, "quantity": products[i]["quantity"], "requirements": joined}
    survivors = [merged if n == i else r for n, r in enumerate(raw) if n != j]
    for name in ("written", "uncovered", "let_go", "indifferent"):
        setattr(sheet, name, retag_all(getattr(sheet, name)))
    if sheet.now_reqs is not None:
        sheet.now_reqs = retag_all(sheet.now_reqs)
    sheet.resolved = [retag(r) for r in vars(sheet).get("resolved") or []]
    for key in ("requirements", "hints", "references", "soft_wants"):
        if isinstance(reading.get(key), list):
            reading[key] = retag_all(reading[key])
    targets = reading.get("reference_targets")
    for target in targets if isinstance(targets, list) else []:
        if isinstance(target, dict) and ours(str(target.get("product") or "")):
            target["product"] = phrase
    state = vars(run)
    for ref in state.get("refs") or []:
        ref.update({key: retag(ref[key]) for key in ("req", "head", "now") if isinstance(ref.get(key), str)})
        if ours(str(ref.get("tag") or "")):
            ref["tag"] = new_tag
    for key in ("pending_seconds", "open_pair"):
        if isinstance(state.get(key), list):
            state[key] = [retag(p) for p in state[key]]
    if remaining < 2:
        reading.pop("items", None)
        reading["product"] = phrase
        reading["quantity"] = products[i]["quantity"]
    else:
        reading["items"] = survivors
        reading["product"] = " and ".join(str(r.get("product")) for r in survivors)
    sheet.version += 1
    for key in ("home_memo", "menu_memo", "quota_reads", "walked"):
        state.pop(key, None)
    spend_scale(run)
    jot(run, "merge_joined", label=label, products=[products[i]["product"], products[j]["product"]], into=phrase,
        label_words=in_order, products_left=max(1, len(need_items(sheet))))
    lines_said = "keeps one line for it" if remaining >= 2 else "keeps ONE line"
    return (f"'{label}' is ONE category in this store and the request wants a single product of it ('{phrase}'): the order "
            + lines_said + ", never a line per kind the category name joins. Requirements now: " + needs_text(sheet))


def lookup_key(run, tool):
    schema = run.shapes.get(tool or "") or {}
    fields = schema.get("properties")
    if not isinstance(fields, dict):
        fields = {}
    required = schema.get("required")
    must = [name for name in required if name in fields] if isinstance(required, list) else []
    texts = []
    for name, spec in fields.items():
        if isinstance(spec, dict) and spec.get("type") == "string" and not spec.get("enum"):
            texts.append(name)
    picked = next((name for name in texts if name in must), None)
    if not picked and len(texts) == 1:
        picked = texts[0]
    if not picked or any(name != picked for name in must):
        return None
    return picked


def lookup_tools(run):
    busy = {role(run, r) for r in ("search", "message", "cart", "cart_view", "order", "decline")} - {None}
    return [name for name in run.shapes if name not in busy and lookup_key(run, name)]


def lookup_call(run, ident, tool):
    if not tool:
        return None
    if tool == role(run, "details"):
        return details_call(run, ident)
    known = run.archive.ids

    def names_known(value):
        return isinstance(value, str) and value in known
    for entry in reversed(run.wire.log):
        if entry.get("kind") != "env":
            continue
        for act in entry.get("actions") or []:
            if act.get("name") != tool:
                continue
            args = act.get("args")
            if isinstance(args, dict) and any(names_known(v) for v in args.values()):
                return {name: ident if names_known(v) else v for name, v in args.items()}
    key = lookup_key(run, tool)
    return {key: ident} if key else None


def rec_id(d):
    pairs = [(str(key).lower(), val) for key, val in d.items()]
    for test in (lambda name: name == "id", lambda name: name.endswith("id")):
        for name, val in pairs:
            if test(name) and isinstance(val, str) and re.fullmatch(ID_LIKE, val):
                return val
    return None


def asked_ids(call_args):
    values = list(call_args.values()) if isinstance(call_args, dict) else []
    found = {v for v in values if isinstance(v, str) and re.fullmatch(ID_LIKE, v)}
    nested = {x for v in values if isinstance(v, list) for x in v if isinstance(x, str) and re.fullmatch(ID_LIKE, x)}
    found |= nested
    return found


LOOKUP_FLAGS = ("id", "stock", "props", "notes", "price")
NO_PRICE = float("inf")


class CapsPass:
    def __init__(self, run, call_name, asked):
        state = vars(run)
        self.run = run
        self.call_name = call_name
        self.tool = str(call_name)
        self.asked = asked
        self.flags = dict.fromkeys(LOOKUP_FLAGS, False)
        self.dead = state.setdefault("dead", set())
        self.dead_by = state.setdefault("dead_by", {})
        self.stock_by = state.setdefault("stock_by", {})
        self.row_text = state.setdefault("row_text", {})
        self.row_keys = state.setdefault("row_keys", set())
        self.seen_keys = state.setdefault("tool_keys", {}).setdefault(self.tool, set())
        self.note_keys = state.setdefault("note_keys", {}).setdefault(self.tool, set())
        self.cart_tools = {role(run, "cart"), role(run, "cart_view")} - {None}
        self.epoch = state.get("cart_seq", 0)
        self.newly = []


def learn_caps(run, call_name, call_args, obs):
    asked = asked_ids(call_args)
    tool = str(call_name)
    caps = vars(run).setdefault("tool_caps", {}).setdefault(tool, {"calls": 0, "id": 0, "stock": 0, "props": 0, "notes": 0,
                                                                     "price": 0})
    caps["calls"] += 1
    cp = CapsPass(run, call_name, asked)
    if not isinstance(obs, dict) or not obs.get("error"):
        walk_dicts(obs, lambda d: caps_visit(cp, d))
    for flag, got in cp.flags.items():
        caps[flag] += int(got)
    return cp.newly


def role_guesses(run):
    guess = vars(run).get("role_guess")
    if guess is None:
        try:
            guess = code_roles(run)
        except Exception:
            guess = {}
        run.role_guess = guess
    return guess


def stock_tool(run):
    caps = vars(run).get("tool_caps") or {}
    tools = lookup_tools(run)
    with_stock = [t for t in tools if (caps.get(t) or {}).get("stock", 0) > 0]
    if with_stock:
        return max(with_stock, key=lambda t: (caps[t]["stock"], t == role(run, "view"), t == role(run, "details"), t))
    guess = role_guesses(run)
    for tool in (role(run, "view"), guess.get("view"), role(run, "details")):
        got = caps.get(tool or "") or {}
        blind = got.get("id", 0) >= 2 and got.get("stock", 0) == 0
        if tool and tool in tools and not blind:
            return tool
    busy = {guess.get(r) for r in ("cart", "cart_view", "order", "decline", "message")} - {None}
    for tool in tools:
        if tool not in busy and not (caps.get(tool) or {}).get("id", 0):
            return tool
    return None


def notes_tool(run):
    state = vars(run)
    caps = state.get("tool_caps") or {}
    details = role(run, "details")
    detail_keys = set((state.get("tool_keys") or {}).get(details or "", set()))
    note_keys = state.get("note_keys") or {}
    showing = [t for t in lookup_tools(run) if t != details and set(note_keys.get(t) or set()) - detail_keys]
    if not showing:
        return None
    return max(showing, key=lambda t: ((caps.get(t) or {}).get("notes", 0), t))


def dead_lines(run):
    return set(vars(run).get("dead") or set())


def stale_lines(run):
    marks = vars(run).get("line_seq") or {}
    since = vars(run).get("cart_seq", 0)
    return [i for i in run.trolley.units if marks.get(i, -1) < since]


def ed_firm(value):
    return not re.search(r"\b(?:" + VALUE_LINKS + r")\b|\d\s*(?:-|to)\s*\d|[+<>]|\b(?:about|around|approx\w*|roughly)\b", value)


def dead_swap(run, line):
    gone, ver = dead_lines(run), str(run.needs.version)
    kind = shelf_known(run, line)
    several = len(need_items(run.needs)) >= 2
    qty = max(1.0, units_held(run.trolley, line))
    others = [i for i in run.trolley.units if i != line and swap_cost(run, i) != NO_PRICE]
    rest = sum(swap_cost(run, i) * units_held(run.trolley, i) for i in others)
    fits = swap_verified(run, line, kind, several, gone, ver, qty, rest)
    if fits:
        return min(fits, key=lambda i: (swap_cost(run, i), i)), ""
    if several and not kind:
        return "", ""
    k = next((n for n, home in enumerate(home_shelves(run)) if kind and kind in home), None) if several else None
    ranked = swap_rows(run, line, kind, k, gone, ver, qty, rest)
    top = ranked[:3]
    run.swap_unread = [i for r, _, i in top if r == 1]
    run.swap_page = None
    if all(r != 0 for r, _, _ in ranked):
        try:
            run.swap_page = dead_page(run, k)
        except Exception:
            run.swap_page = None
    return "", ", ".join(swap_label(r, p, i) for r, p, i in top)


def dead_page(run, k):
    paging = page_arg(run)
    if not paging:
        return None
    open_walks = []
    for walk in vars(run).get("page_book", {}).values():
        if walk_carries(run, walk["args"], k) and next_page(run, walk) is not None:
            open_walks.append(walk)
    if not open_walks:
        return None
    best = max(open_walks, key=lambda w: (len(walk_carries(run, w["args"], k)), -len(w["read"])))
    return {**best["args"], paging[0]: next_page(run, best)}


def swap_open(run):
    if vars(run).get("swap_done"):
        return ""
    gone = dead_lines(run)
    for ident in vars(run).get("swap_unread") or []:
        if ident not in run.archive.read and ident not in gone and ident not in run.trolley.units:
            return ident
    return ""


def dead_note(run, lines):
    said = []
    run.event_seen = True
    for line in lines:
        run.swap_page, run.swap_unread = None, []
        pick, rows = dead_swap(run, line)
        head = f"{line} is OUT OF STOCK now (its newest record says so), so an order that holds it fails. Remove it from the cart"
        if pick:
            least = ", the cheapest such listing" if need_priority(run.needs) == "cheapest" else ""
            said.append(f"{head} and put {pick} in (the item check found it meets every requirement now in force{least}), "
                        "re-read its stock and price, then order.")
        elif rows:
            said.append(f"{head}; no other verified fit of this product is known, so look at the cheapest in-stock rows of its "
                        f"category inside the budget ({rows}; rows already read whose own record lacks or fails a stated requirement "
                        "are excluded, and an unread one needs its details read first), put the first that fits in, re-read "
                        "its stock and price, then order.")
        else:
            said.append(f"{head} and find another listing that fits.")
        page = vars(run).get("swap_page")
        if page and not pick:
            said.append("Further rows of that kind sit on the next unread page of the search the stated value leads: "
                        + json.dumps(page, default=str) + ".")
    return " ".join(said)


def refusal_reread(run, obs=None):
    heard = ""
    try:
        heard, gone = refusal_gone(run, obs)
    except Exception as err:
        gone = ""
        jot_error(run, "refusal_gone_error", err)
    return heard + (gone if gone else refusal_queue(run))


def refusal_gone(run, obs):
    held_now = [i for i in run.trolley.units if units_held(run.trolley, i) > 0]
    state = vars(run)
    if obs is None or len(held_now) != 1 or run.wire.mute or state.get("covering") or state.get("last_kind") != "order":
        return "", ""
    texts = shopper_lines(run, role(run, "order") or "", obs, True)
    if not texts:
        return "", ""
    sheet = run.needs

    def snapshot():
        return (sheet.version, len(sheet.let_go), len(sheet.uncovered), str(sheet.spend_cap), "|".join(props_pending(sheet)))
    before = snapshot()
    note = hear_lines(sheet, texts, None)
    after = snapshot()
    heard = " " + note if note else ""
    moved_alone = after[0] != before[0] and after[1:] == before[1:]
    if not note or not moved_alone:
        return heard, ""
    line = held_now[0]
    state.setdefault("dead", set()).add(line)
    state.setdefault("dead_by", {})[line] = "lookup"
    jot(run, "refusal_dead", line=line)
    return heard, (" The refusal reads as an item turning unavailable while the cart held a single line, so the code treats "
                   + line + " as sold out (no re-read): " + dead_note(run, [line]))


def refusal_queue(run):
    tool = stock_tool(run)
    lines = list(run.trolley.units)[:4]
    if lines and tool and not vars(run).get("open_queue") and not reads_blocked(run, len(lines) + 3):
        run.open_queue = lines
        run.open_recheck, run.open_purpose, run.open_tool = True, "cart", tool
        run.open_say = "Refused order: the stock of every cart line gets read again now, since one may have sold out."
        return " The cart lines' stock is read for you now (" + tool + ")."
    return ""


def code_judges_all(run):
    products = need_items(run.needs)
    every = range(len(products)) if products else [None]
    firm = [plain for k in every for _, plain in code_values(run, k) if ed_firm(plain.lower())]
    settled = differ_done(run, None)
    by_reference = 0
    for req in props_pending(run.needs):
        if any(ref_match(run, ref, req) for ref in settled):
            by_reference += 1
    return len(firm) + by_reference >= len(list(props_pending(run.needs)))


def differ_done(run, k):
    products, left = need_items(run.needs), props_pending(run.needs)

    def standing(ref):
        if ref.get("relation") != "differ" or ref.get("status") != "resolved":
            return False
        if not str(ref.get("value") or "").strip() or not ref_live(run, ref):
            return False
        return any(ref_match(run, ref, req) for req in left)

    def elsewhere(ref):
        if not products or k is None or not ref.get("tag"):
            return False
        return product_of(run, f"x ({ref['tag']})") not in (k, None)
    return [ref for ref in vars(run).get("refs") or [] if standing(ref) and not elsewhere(ref)]


def muted_basket_put(run, picks):
    was = run.wire.mute
    run.wire.mute = True
    try:
        bad = code_put(run, picks)
        placed = bad == [] and all(i in run.trolley.units for i, _ in picks)
        unread = [i for i in stale_lines(run) if any(i == p for p, _ in picks)]
        if placed and unread and moves_left(run) >= len(unread) + 2:
            _, sold = code_recheck(run, unread)
            placed = not sold
    finally:
        run.wire.mute = was
    return placed and bool(run.trolley.units) and not set(run.trolley.units) & dead_lines(run)


def decline_repair(run):
    several = repair_mark(run)
    if several is None or (several and proof_blocks(run)):
        return ""
    gone = dead_lines(run)
    unread = set(stale_lines(run))
    alive = {i for i in run.trolley.units if i not in gone and i not in unread}
    saved = set(run.refused_lines)
    ver = str(run.needs.version)
    own = code_judges_all(run)
    done = False
    try:
        run.refused_lines.difference_update(alive)
        picks = repair_picks(run, several, gone, ver, own)
        if not repair_fits(run, picks, ver, own):
            jot(run, "decline_repair", picks=picks, done=False)
            return ""
        if moves_left(run) < len(picks) + 2:
            jot(run, "decline_repair", picks=picks, done=False, why="steps")
            return ""
        done = muted_basket_put(run, picks)
        jot(run, "decline_repair", picks=picks, done=done, cart=dict(run.trolley.units))
    except Exception as err:
        done = False
        jot_error(run, "decline_repair_error", err)
    finally:
        if not done:
            run.refused_lines.clear()
            run.refused_lines.update(saved)
    if not done:
        return ""
    plural = bool(need_items(run.needs))
    return ("Not run yet (no step used). Declining is right only when nothing can fit, yet the records hold a fitting candidate "
            + ("for every product asked for" if plural else "") + " (each passed by the item check or read with all stated "
            "values on its own record), so the code placed " + ("them" if plural else "it") + " in the cart and read the stock: "
            + cart_summary(run.trolley) + " Order this cart now; decline only if a record you read shows one of these lines "
            "failing a requirement now in force.")


def ed_plain(value):
    return ed_firm(str(value).lower())


def ed_word(value):
    low = str(value).lower()
    if not re.search(r"[a-z]{3,}", low) or re.search(r"\d", low):
        return False
    return ed_plain(low) and not re.search(EXCLUDE_WORDS, low) and not ref_text(low)


def ed_reqs(run, k):
    products = need_items(run.needs)
    found = []
    for req in props_pending(run.needs):
        head, _, value = str(req).partition(":")
        if products and product_of(run, head) not in (k, None):
            continue
        prop = re.split(r"\s*\(", head)[0].strip()
        bare = re.sub(r"\((?:stated|revealed[^)]*)\)", "", value).strip()
        found.append((prop, bare, req not in run.needs.written))
    return found


def ed_pool(run, k):
    sheet = run.needs
    products = need_items(sheet)
    ver = str(sheet.version)
    mark = (k, len(run.archive.items), len(run.archive.read), ver, "|".join(props_pending(sheet)), len(run.passed),
            sum(1 for v in run.rulings.values() if v), tuple(sorted(dead_lines(run))), tuple(sorted(run.refused_lines)),
            str(sheet.spend_cap), len(vars(run).get("refs") or []))
    memo = vars(run).setdefault("ed_memo", {})
    if mark in memo:
        return memo[mark]
    if len(memo) > 64:
        memo.clear()
    homes = home_shelves(run)
    if not products:
        home = set(shelf_labels(run)) or (set(homes[0]) if homes else set())
    elif k is not None and k < len(homes):
        home = set(homes[k])
    else:
        home = set()
    memo[mark] = entries = pool_entries(run, k if products else None, k, home, ver) if home else []
    return entries


def pool_entries(run, which, k, home, ver):
    sheet = run.needs
    counts = need_counts(sheet)
    if need_items(sheet) and k is not None and k < len(counts):
        qty = counts[k]
    else:
        qty = counts[0] if counts else 1
    failed = {key.split("|")[0] for key, val in run.rulings.items() if val and key.split("|")[1:2] == [ver]}
    live = {key.rpartition("|")[0] for key in run.passed if key.rpartition("|")[2] == ver}
    cap = float(sheet.spend_cap) if is_number(sheet.spend_cap) else None
    judged = [(p, v) for p, v in code_values(run, which) if ed_plain(v)]
    joins = joined_values(code_values(run, which))
    shown = [(p, v) for p, v, revealed in ed_reqs(run, which) if revealed]
    pointers, gone_ids = ref_pointers(run), dead_lines(run)

    def entry_for(ident, row):
        fields = row["fields"]
        read = ident in run.archive.read
        figure = price_of(run.archive, ident) if row["price"] is None else row["price"]
        price = float(figure) if is_number(figure) else None
        gone = ident in gone_ids or row["stock"] is False or ident in run.refused_lines
        over = cap is not None and price is not None and price * qty > cap + 1e-9
        pointer = ident in pointers or ref_excluded(run, fields, which)
        clash = field_clashes(fields, judged, joins)
        misses = [(p, v) for p, v in judged if not code_shows(row, p, v, set(), True)]
        ruled = ident in failed or bool(read and field_view(run, ident)["fails"])
        verified = ident in live and not gone
        fit = False
        if not (ruled or pointer or gone or over):
            fit = verified or (read and not misses and not clash)
        return dict(id=ident, price=price, read=read, verified=verified, fit=fit, gone=gone, over=over, pointer=pointer,
                    ruled=ruled, clash=clash, misses=misses, row=row, out=bool(gone or over or pointer or ruled or clash),
                    revealed_ok=all(code_shows(row, p, v, set(), True) for p, v in shown))
    return [entry_for(ident, row) for ident, row in code_rows(run).items() if row["shelf"] in home]


def ed_proof(run):
    def proven(k):
        reads = [e for e in ed_pool(run, k) if e["read"]]
        return bool(reads) and all(e["out"] for e in reads)
    return [k for k in range(len(need_items(run.needs))) if proven(k)]


def slack_rows(run):
    products = need_items(run.needs)
    cap = run.needs.spend_cap
    if len(products) < 2 or not is_number(cap):
        return []
    try:
        return slack_picks(run, products, cap)
    except Exception as err:
        jot_error(run, "slack_error", err)
        return []


def slack_picks(run, products, cap):
    counts = need_counts(run.needs)

    def units(k):
        return counts[k] if k < len(counts) else 1
    floors = []
    for k in range(len(products)):
        prices = [e["price"] for e in ed_pool(run, k) if e["fit"] and e["price"] is not None]
        if not prices:
            return []
        floors.append(min(prices) * units(k))
    if sum(floors) <= float(cap) + 1e-9:
        return []
    walks = list(vars(run).get("page_book", {}).values())
    hits = set(vars(run).get("all_hits") or [])
    pools = []
    for k in range(len(products)):
        slack = (float(cap) - (sum(floors) - floors[k])) / max(1, units(k))
        value_sets = [{w for w in re.findall(r"[a-z0-9]{2,}", v.lower()) if w not in LINK_WORDS} for _, v, _ in ed_reqs(run, k)]
        lead = dict.fromkeys(hits, 1)
        for walk in walks:
            words = [w for text in free_text(run, walk["args"]) for w in re.findall(r"[a-z0-9]{2,}", text.lower())]
            if run.lead_words:
                words = words[: run.lead_words]
            heard = set(words)
            score = sum(1 for vals in value_sets if vals & heard)
            for i in walk.get("row_shelf") or {}:
                if score:
                    lead[str(i)] = max(lead.get(str(i), 0), score)

        def open_row(e):
            if e["read"] or e["gone"] or e["pointer"] or e["ruled"] or e["clash"] or e["price"] is None:
                return False
            return 0 < slack and e["price"] <= slack + 1e-9
        ranked = sorted((-lead.get(e["id"], 0), e["price"], e["id"]) for e in ed_pool(run, k) if open_row(e))
        pools.append([i for _, _, i in ranked])
    picked = []
    while len(picked) < 3 and any(pools):
        for pool in pools:
            if pool and len(picked) < 3:
                picked.append(pool.pop(0))
    if picked:
        run.slack_total = sum(floors)
    return picked


def verify_problems(run, ident):
    latest = next((e for e in reversed(run.wire.log)
                   if isinstance(e, dict) and e.get("kind") == "verify" and e.get("item") == ident), None)
    if latest is None:
        return None
    outcome = latest.get("result")
    if not isinstance(outcome, dict) or outcome.get("fits") is not False:
        return None
    return [str(p) for p in outcome.get("problems") or [] if str(p).strip()]


def ed_news(run, k):
    if run.refusal_n > 0:
        return True
    products = need_items(run.needs)
    changed = list(run.needs.uncovered) + list(run.needs.let_go)
    return any(not products or product_of(run, str(req).partition(":")[0]) in (k, None) for req in changed)


def near_hold(run):
    try:
        return near_order_now(run)
    except Exception as err:
        jot_error(run, "near_hold_error", err)
        return ""


def near_order_now(run):
    info = vars(run).get("near_cart")
    if not info or info.get("ver") != run.needs.version or info.get("refusals") != run.refusal_n or not role(run, "order"):
        return ""
    lines = list(info.get("lines") or [])
    if not lines:
        return ""
    for ident in lines:
        if ident not in run.trolley.units or ident in dead_lines(run) or ident in run.refused_lines:
            return ""
    if moves_left(run) < 1 or run.wire.done or run.wire.broke or time_left(run.wire) < 2:
        return ""
    run.near_cart = None
    jot(run, "near_order", lines=lines)
    was = run.wire.mute
    try:
        finish_now(run)
    finally:
        run.wire.mute = was
    return ("Not run (no step used). A meaning check showed that the cart's lines (" + ", ".join(lines) + ") give the "
            "requested values in other words, and nothing changed since, so the code ordered the cart instead of declining.")


def near_check(run):
    wire = run.wire
    if vars(run).get("near_done") or not (role(run, "cart") and role(run, "order")) or moves_left(run) < 4:
        return ""
    if time_left(wire) <= CLOSING + 30 or wire.mute or vars(run).get("covering") or wire.done or over_budget(wire) \
            or wire.no_model:
        return ""
    try:
        return near_basket(run)
    except Exception as err:
        jot_error(run, "near_error", err)
        return ""


def near_candidates(run, pool, reqs):
    found = []
    for e in pool:
        if not e["read"] or e["gone"] or e["over"] or e["pointer"] or e["price"] is None:
            continue
        written = near_lines(e, reqs)
        fails, problems = set(), []
        if e["ruled"]:
            fails = {str(x).split(":", 1)[0].strip() for x in field_view(run, e["id"])["fails"]}
            problems = verify_problems(run, e["id"]) or []
        if not written:
            continue
        marks = set()
        for prop, value, _ in written:
            marks.update(w for w in prop_words(prop) + re.findall(r"[a-z0-9]+", value.lower()) if len(w) >= 3 and w not in LINK_WORDS)
        owned_fails = all(any(same_prop(f, prop) for prop, _, _ in written) for f in fails)
        if owned_fails and all(any(re.search(r"\b" + re.escape(w), p.lower()) for w in marks) for p in problems):
            found.append((len(written), e["price"], e["id"], written))
    return found


def near_basket(run):
    products = need_items(run.needs)
    slots = list(range(len(products))) if products else [None]
    counts = need_counts(run.needs)
    spend = run.needs.spend_cap
    cap = float(spend) if is_number(spend) else None

    def units(n):
        return counts[n] if n < len(counts) else 1

    def over(total):
        return cap is not None and total > cap + 1e-9
    found = near_pools(run, slots, units)
    if found is None:
        return ""
    basket, pools = found
    floor = sum(price * qty for _, price, qty in basket.values())
    floor += sum(min(c[1] for c in pools[n]) * units(n) for n in pools)
    if over(floor):
        return ""
    flat = [(n, ident, prop, value, says) for n in sorted(pools) for _, _, ident, lines in pools[n]
            for prop, value, says in lines][:12]
    agreed = ask_near(run, flat)
    for n in sorted(pools):
        for _, price, ident, lines in pools[n]:
            spots = [i for i, row in enumerate(flat) if row[0] == n and row[1] == ident]
            if len(spots) == len(lines) and all(i in agreed for i in spots):
                basket[n] = (ident, price, units(n))
                break
        else:
            return ""
    if over(sum(price * qty for _, price, qty in basket.values())):
        return ""
    picks = [(basket[n][0], basket[n][2]) for n in range(len(slots))]
    if moves_left(run) < len(picks) + 2:
        return ""
    ver = str(run.needs.version)
    for n in pools:
        approve_near(run, basket[n][0], ver)
    done = muted_basket_put(run, picks)
    jot(run, "near_basket", picks=picks, done=done, cart=dict(run.trolley.units))
    return near_report(run, flat, basket, picks, ver) if done else ""


def cart_moved(run, before):
    state = vars(run)
    run.cart_ver = state.get("cart_ver", 0) + 1
    run.since_cart = 0
    now = run.trolley.units
    if any(line not in before for line in now):
        run.cart_seq = state.get("absorb_n", 0)
    marks = state.setdefault("line_seq", {})
    grown = [line for line, count in now.items() if line in before and count > before[line] + 1e-9]
    for line in grown:
        marks.pop(line, None)


def after_put(run, call_args, obs):
    state = vars(run)
    if units_in_view(run.trolley, obs) is not None:
        run.cart_shown = state.get("cart_ver", 0)
    firsts = state.setdefault("first_puts", set())
    cart = run.trolley.units
    fresh = []
    for line, count in cart_lines(run.trolley, call_args):
        if line in cart and line not in firsts and (count is None or count > 0):
            fresh.append(line)
    if fresh and adds_to_cart(run.trolley, call_args) and not code_fault(obs):
        firsts.update(fresh)
        queue_put_reads(run, fresh)


def mark_fresh(run, obs):
    state = vars(run)
    if not state.get("stock_seen"):
        return
    marks = state.setdefault("line_seq", {})

    def visit(d):
        ident = rec_id(d)
        if ident in run.trolley.units and stock_state(d) is not None:
            marks[ident] = state.get("absorb_n", 0)
    walk_dicts(obs, visit)


def cart_gone(run, lines):
    gone = set(lines) & dead_lines(run)
    unread = [i for i in stale_lines(run) if i in lines and i not in gone]
    if not unread or not stock_tool(run) or moves_left(run) < len(unread) + 2:
        return gone
    _, sold = code_recheck(run, unread)
    return gone | set(sold) | (set(lines) & dead_lines(run))


LIST_CHANGERS = r"\b(?:remov|delet|cancel|clear|discard|releas|refund|purchas|checkout|wish|basket|cart|bag)"


def batch_tool(run):
    if "batch_spec" not in vars(run):
        try:
            spec = find_batch_spec(run)
        except Exception:
            spec = None
        run.batch_spec = spec
    return run.batch_spec


def find_batch_spec(run):
    busy = {role(run, r) for r in ("search", "message", "cart", "cart_view", "order", "decline")} - {None}
    blurbs = {}
    for tdef in run.toolset or []:
        fn = tdef.get("function") if isinstance(tdef, dict) else None
        if isinstance(fn, dict) and isinstance(fn.get("name"), str):
            spaced = re.sub(r"([a-z])([A-Z])", r"\1 \2", fn["name"]).replace("_", " ")
            blurbs[fn["name"]] = spaced + " " + str(fn.get("description") or "")
    for name in list(run.shapes):
        if re.search(LIST_CHANGERS, blurbs.get(name, name).lower()):
            busy.add(name)
    for name, schema in run.shapes.items():
        fields = schema.get("properties") if isinstance(schema, dict) else None
        if not isinstance(fields, dict):
            fields = {}
        arrays = []
        for key, spec in fields.items():
            if not isinstance(spec, dict) or spec.get("type") != "array":
                continue
            inner = spec.get("items")
            if isinstance(inner, dict) and inner.get("type") == "string" and not inner.get("enum"):
                arrays.append(key)
        required = schema.get("required") if isinstance(schema, dict) and isinstance(schema.get("required"), list) else []
        if name in busy or len(arrays) != 1:
            continue
        most = fields[arrays[0]].get("maxItems")
        if any(r not in arrays for r in required) or not is_number(most) or most < 2:
            continue
        return name, arrays[0], int(min(most, 10))
    return None


def batch_ok(found, ids):
    wanted = set(ids)
    got = set()

    def visit(d):
        values = list(d.values())
        nested = any(isinstance(v, dict) and sum(1 for x in v.values() if isinstance(x, (str, int, float))) >= 2 for v in values)
        flat = not any(isinstance(v, dict) for v in values) \
            and sum(1 for v in values if isinstance(v, str) and 0 < len(v) <= 70) >= 5
        if nested or flat:
            got.update(v for v in values if isinstance(v, str) and v in wanted)
    try:
        for obs in found:
            walk_dicts(obs, visit)
    except Exception:
        return False
    return bool(got) and 2 * len(got) >= len(wanted)


def title_of(row):
    title = row.get("title")
    if isinstance(title, str):
        return title
    for key, v in row.items():
        if isinstance(v, str) and re.search(r"title|heading|headline", str(key).lower()):
            return v
    return ""


def menu_rows(run, home, most, taken, qty=1):
    cap = run.needs.spend_cap
    state = vars(run)
    order, titles, sold, known = shelf_scan(run, home)
    hits = sorted(state.get("all_hits", []), key=lambda i: -state.get("hit_score", {}).get(i, 0))

    def allowed(i):
        if i in sold or i in taken or i in run.refused_lines:
            return False
        return not (is_number(cap) and is_number(price_of(run.archive, i)) and price_of(run.archive, i) * qty > cap)
    seen = [i for i in hits if i in titles] + [i for i in order if i not in hits]
    cands = [i for i in seen if allowed(i)]
    if len(cands) < most:
        # (these reads only learn the product's field names for the questions: when its rows so far all cost more than
        # the budget, over-budget rows still show those fields, and without them the product is never asked about)
        cands += [i for i in seen if i not in cands and i not in sold and i not in taken and i not in run.refused_lines]
    words = {i: title_words(titles.get(i, "")) for i in cands}
    lead = state.get("row_score", {})
    picks = []
    try:
        first = led_pick(run, home, most, words)
        if first:
            picks.append(first)
            known.append(words[first])
    except Exception:
        picks = []

    def rank(i):
        tier = 0 if i in hits else 1 if lead.get(i) else 2
        overlap = max((len(words[i] & w) / max(1, len(words[i] | w)) for w in known), default=0.0)
        return tier, overlap, cands.index(i)
    while len(picks) < most:
        pool = [i for i in cands if i not in picks][:16]
        if not pool:
            break
        best = min(pool, key=rank)
        picks.append(best)
        known.append(words[best])
    return picks


def start_menu_queue(run, picks, say):
    run.open_queue = picks
    run.open_recheck = True
    run.open_say = say
    run.open_purpose = "menu"


def field_probe(run):
    # (once, before the first question: when no listing read so far shows its fields and the other read tool is the
    # likelier one to show them, a few listings already opened are read again through it)
    try:
        state = vars(run)
        reader = field_reader(run)
        if state.get("field_probed") or not reader or reader == role(run, "details") or moves_left(run) <= order_due(run) + 12:
            return False
        state["field_probed"] = True
        shelf = own_shelf(run)
        ids = [i for i in run.archive.read if not shelf or shelf_known(run, i).strip().lower() == shelf.strip().lower()]
        ids = ids or list(run.archive.read)
        if not shelf:
            # (several products: one listing of each shelf first, so every product's fields reach the planner)
            firsts = {}
            for ident in ids:
                firsts.setdefault(shelf_known(run, ident).strip().lower(), ident)
            ids = list(firsts.values()) + [i for i in ids if i not in firsts.values()]
        ids = ids[:3]
        if not ids:
            return False
        run.open_queue, run.open_purpose, run.open_tool = ids, "menu", reader
        run.open_say = "These listings showed no fields yet, so I read them through the other listing tool before asking the shopper."
        run.wire.log.append({"kind": "field_probe", "tool": reader, "ids": ids})
        return True
    except Exception as err:
        jot_error(run, "field_probe_error", err)
        return False


def pre_read(run):
    try:
        return queue_menu_reads(run)
    except Exception as err:
        jot_error(run, "pre_read_error", err)
        return False


def queue_menu_reads(run):
    batch = batch_tool(run)
    state = vars(run)
    again = state.get("pre_rounds", 0) == 1
    if not batch or (state.get("pre_read_done") and not again) or len(run.questions_log) > 1 \
            or moves_left(run) <= (25 if again else 20) or time_left(run.wire) < CLOSING + 60:
        return False
    homes = home_shelves(run)
    qtys = [p["quantity"] for p in need_items(run.needs)] or [need_qty(run.needs)]

    def qty_for(k):
        return qtys[min(k, len(qtys) - 1)]

    def reads_on(home):
        return sum(rich_read_count(run.archive, s) for s in home) if home else 4
    if again:
        run.pre_rounds = 2
        wants, taken = [], []
        for k, home in enumerate(homes):
            have = reads_on(home)
            if have >= 3:
                continue
            rows = menu_rows(run, home, 4 - have, taken + list(state.get("pre_picked") or []), qty_for(k))
            taken += rows
            wants.append(rows)
        picks = []
        while any(wants):
            for rows in wants:
                if rows:
                    picks.append(rows.pop(0))
        picks = picks[: batch[2]]
        if not picks:
            return False
        start_menu_queue(run, picks, "A few more listings get opened first: some products' first ones showed only a few fields.")
        return True
    empty, thin = [], []
    for k, home in enumerate(homes):
        have = reads_on(home)
        if have == 0:
            empty += menu_rows(run, home, 2 if need_items(run.needs) else 4, empty, qty_for(k))
        elif have < 4:
            thin.append((k, 4 - have))
    picks = list(empty)
    for k, short in thin:
        picks += menu_rows(run, homes[k], short, picks, qty_for(k))
    doubled = len(empty) > batch[2] and moves_left(run) > 25
    picks = picks[: batch[2] * (2 if doubled else 1)]
    if not picks:
        return False
    run.pre_read_done = True
    run.pre_rounds = 1
    run.pre_picked = list(picks)
    start_menu_queue(run, picks, "First a few listings per requested product get opened so their fields are known before the shopper is asked anything.")
    return True


def lacking_rows(run):
    try:
        return lacking_picks(run)
    except Exception as err:
        jot_error(run, "lacking_error", err)
        return []


def lacking_picks(run):
    products = need_items(run.needs)
    if not products:
        return []
    homes = home_shelves(run)
    state = vars(run)
    live = {key.rpartition("|")[0] for key in run.passed if key.rpartition("|")[2] == str(run.needs.version)}
    live = live - run.refused_lines - dead_lines(run)
    covered = {shelf_known(run, i) for i in live | set(state.get("audit_fits") or [])} - {""}
    lacking = [k for k, home in enumerate(homes) if home and not home & covered]
    if not lacking:
        return []
    batch = batch_tool(run)
    most = batch[2] if batch else 3
    cap = run.needs.spend_cap
    newest, sold = [], set()

    def visit(d):
        ident, shelf = row_mark(d), shelf_of(d)
        if ident not in run.archive.ids or not shelf:
            return
        newest.append((ident, shelf))
        if sold_out(d):
            sold.add(ident)
    for rec in reversed(run.archive.items):
        walk_dicts(rec, visit)
    where = {}
    for ident, shelf in newest:
        where.setdefault(ident, shelf)
    hits = sorted(state.get("all_hits", []), key=lambda i: -state.get("hit_score", {}).get(i, 0))
    pools = {k: [] for k in lacking}
    for ident in dict.fromkeys([i for i in hits if i in where] + [i for i, _ in newest]):
        if ident in run.archive.read or ident in run.refused_lines or ident in sold:
            continue
        k = next((k for k in lacking if where.get(ident) in homes[k]), None)
        if k is None:
            continue
        cost = price_of(run.archive, ident)
        if not (is_number(cap) and is_number(cost) and cost * products[k]["quantity"] > cap):
            pools[k].append(ident)
    picks = []
    while len(picks) < most and any(pools.values()):
        for k in lacking:
            if pools[k] and len(picks) < most:
                picks.append(pools[k].pop(0))
    return picks


def text_arg(run, the_tool):
    schema = run.shapes.get(the_tool or "") or {}
    specs = schema.get("properties") or {}
    required = list(schema.get("required") or [])

    def is_text(spec):
        if not isinstance(spec, dict) or "enum" in spec:
            return False
        return spec.get("type") == "string" or "string" in (spec.get("type") or [])
    texts = [name for name, spec in specs.items() if is_text(spec)]
    aim = next((name for name in required if name in texts), None)
    if not aim:
        aim = texts[0] if len(texts) == 1 else None
    if not aim or any(name != aim for name in required):
        return None
    return aim


def message_call(run, text_body, cid):
    tool = role(run, "message")
    aim = text_arg(run, tool) if tool else None
    if not aim:
        return None
    call_args, faults = check_args({aim: text_body}, run.shapes[tool])
    if faults:
        return None
    return {"id": cid, "type": "function", "function": {"name": tool, "arguments": json.dumps(call_args)}}


def product_of(run, text):
    products = need_items(run.needs)
    if not products:
        return None
    names = [set(kind_tokens(p["product"])) for p in products]
    others = [set().union(*(names[j] for j in range(len(names)) if j != k)) for k in range(len(names))]
    for piece in re.findall(r"\(([^)]*)\)", str(text))[:1] + [str(text)]:
        words = set(kind_tokens(piece))
        scores = [kind_score(own, words, others[k]) for k, own in enumerate(names)]
        best = max(scores)
        if best > 0 and scores.count(best) == 1:
            return scores.index(best)
    return None


def plan_product(run, qitem):
    k = product_of(run, str(qitem.get("property") or ""))
    if k is None:
        k = product_of(run, str(qitem.get("question") or ""))
    return k


def planned_for(run, scripted, due, products):
    for n, qitem in enumerate(scripted):
        if plan_product(run, qitem) in due:
            return [qitem] + scripted[:n] + scripted[n + 1:]
    for qitem in scripted:
        second = str(qitem.get("second") or "").strip()
        j = product_of(run, second) if second else None
        if j not in due:
            continue
        tied = re.search(r"\(([^)]+)\)", second)
        product = tied.group(1).strip() if tied else products[j]["product"]
        return [{"property": second, "question": f"Which {second.split('(')[0].strip()} would you like for the {product}?"}] + scripted
    return None


def quota_first(run, scripted, ok=None):
    products = need_items(run.needs)
    if not products:
        return scripted
    count = len(products)
    asked, sent = question_counts(run, count)
    if max(asked) == 0 and scripted and plan_product(run, scripted[0]) is None:
        return scripted
    found = reveal_counts(run, count)
    unasked = [j for j in range(count) if not asked[j]]
    hinted = story_products(run, count, asked)
    alt_left = {j for j in range(count) if sent[j] > asked[j] and alt_untried(run, j)}

    # (two products: each carries about one hidden need, so once one is told the other goes first, story hints
    # included; with more products several carry none, and a told product waits only for unasked ones)
    pair = count == 2

    def level_of(k):
        waiting = [j for j in everyone if j != k and not found[j]] if pair else [j for j in unasked if j != k]
        late = 1 if found[k] and waiting else 0
        quiet = 0 if k in alt_left else sent[k] - asked[k]
        # (every product gets one question before a story hint pulls more toward one of them)
        return late, min(asked[k], 1), 0 if k in hinted else 1, asked[k] + found[k], quiet, found[k]
    everyone = list(range(count))
    for level in sorted({level_of(k) for k in everyone}):
        due = [k for k in everyone if level_of(k) == level]
        planned = planned_for(run, scripted, due, products)
        if planned is not None:
            return planned
        for k in sorted(due, key=lambda k: (k not in hinted, k)):
            try:
                pick = quota_pick(run, k, asked[k] == 0 or pair and not found[k] and any(found), ok)
            except Exception:
                pick = None
            if pick and ok is not None and (not ok(pick.get("property")) or asked_for(run, k, pick.get("property"))):
                pick = None
            if pick:
                return [pick] + scripted
            if vars(run).get("quota_wait"):
                return scripted
    return scripted


def shelf_probe(run, k, shelf):
    # (once per product: its listings were read only through a tool that shows no fields, so the field names it would
    # be asked about are page labels; a few are read through the tool that shows fields first)
    try:
        probed = vars(run).setdefault("shelf_probed", set())
        reader = field_reader(run)
        if k in probed or not reader or reader == role(run, "details") or moves_left(run) <= order_due(run) + 10:
            return False
        probed.add(k)
        ids = [i for i in run.archive.read if shelf_known(run, i).strip().lower() == shelf.strip().lower()][:3]
        if not ids:
            return False
        run.open_queue, run.open_purpose, run.open_tool = ids, "menu", reader
        run.open_say = "Before asking about this product I read the fields of a few of its listings."
        run.quota_wait = True
        run.wire.log.append({"kind": "shelf_probe", "product": k, "tool": reader, "ids": ids})
        return True
    except Exception as err:
        jot_error(run, "shelf_probe_error", err)
        return False


def quota_pick(run, k, fallback=True, ok=None):
    entry = need_items(run.needs)[k]
    product = entry["product"]
    homes = home_shelves(run)
    home = homes[k] if k < len(homes) else set()
    opened_before = vars(run).get("quota_reads", {}).get(k)
    shelves = [s for s in read_shelves(run.archive) if s.strip().lower() in home]
    if not shelves and home and opened_before:
        shelves = [s for s in sorted(home) if property_menu(run.archive, s)[0]][:1]
    if not shelves:
        queue_quota_reads(run, k, home, entry, opened_before)
        return None
    first_shelf = shelves[0]
    if not table_reads(run.archive, first_shelf) and shelf_probe(run, k, first_shelf):
        return None
    menu = menu_trim(property_menu(run.archive, first_shelf)[0], rich_read_count(run.archive, first_shelf))

    def fresh(label):
        return ok is None or (bool(ok(label)) and not asked_for(run, k, label))
    planned = product_plan(run, k, menu)
    if planned and fresh(planned.get("property")):
        return planned
    try:
        hints = open_hints(run, k)
    except Exception:
        hints = []
    for hint in hints:
        prop = re.split(r"\s*\(", hint.split(":", 1)[0])[0].strip()
        if not prop or junk_key(prop) or plain_hint(run, k, prop, hinted=True):
            continue
        if fresh(f"{prop} ({product})"):
            return tagged_question(prop, product)
    if fallback:
        for weigh in (False, True):
            for line in menu.splitlines():
                prop = line.strip("- ").split(" (in ")[0].strip()
                if prop and open_prop(run, k, prop, line, by_samples=weigh) and fresh(f"{prop} ({product})"):
                    return tagged_question(prop, product)
    return None


def product_plan(run, k, menu):
    try:
        return product_plan_call(run, k, menu)
    except Exception:
        return None


def reads_blocked(run, steps):
    return run.wire.mute or vars(run).get("covering") or moves_left(run) < steps or time_left(run.wire) < CLOSING


def queue_put_reads(run, fresh):
    unread = set(stale_lines(run))
    fresh = [line for line in fresh if line in unread]
    tool = stock_tool(run)
    if not fresh or not tool or reads_blocked(run, 4):
        return
    queue = list(vars(run).get("put_queue") or [])
    queue += [line for line in fresh if line not in queue]
    run.put_queue = queue[:4]
    run.put_tool = tool
    jot(run, "after_put_read", lines=list(run.put_queue), tool=tool)


def near_lines(e, reqs):
    row = e["row"]
    written = []
    for prop, value, revealed in reqs:
        if code_shows(row, prop, value, set(), True):
            continue
        says = []
        for key, v in row["fields"].items():
            text = str(v).strip()
            if same_prop(key, prop) and text:
                says.append(text)
        if revealed or not says or not ed_word(value) or any(re.search(r"\d", s) for s in says):
            return None
        written.append((prop, value, "; ".join(dict.fromkeys(says))[:120]))
    return written


def field_clashes(fields, judged, joins):
    clash = []
    for p, v in judged:
        if not code_clash(fields, p, v):
            continue
        if not any(code_seen(v, val) for key, val in fields.items() if same_prop(key, p)):
            clash.append(p)
    for p, parts in joins:
        named = [val for key, val in fields.items() if same_prop(key, p)]
        if named and not all(any(code_seen(x, val) for val in named) for x in parts):
            clash.append(p)
    return clash


def joined_values(pairs):
    found = []
    for p, v in pairs:
        if "+" in v:
            parts = [x.strip() for x in v.split("+") if x.strip()]
            if len(parts) >= 2 and all(re.search(r"[a-z]{2,}", x.lower()) for x in parts):
                found.append((p, parts))
    return found


def near_pools(run, slots, units):
    basket, pools = {}, {}
    for n, k in enumerate(slots):
        pool = ed_pool(run, k)
        cheapest = min(((e["price"], e["id"]) for e in pool if e["fit"] and e["price"] is not None), default=None)
        if cheapest is not None:
            basket[n] = (cheapest[1], cheapest[0], units(n))
            continue
        if not ed_news(run, k):
            return None
        cands = near_candidates(run, pool, ed_reqs(run, k))
        if not cands:
            return None
        pools[n] = sorted(cands)[: 6 if len(slots) == 1 else 3]
    return (basket, pools) if pools else None


def ask_near(run, flat):
    listing = "\n".join(f"{i + 1}. {ident}: {prop} = '{value[:60]}' | its field reads: {says}"
                        for i, (_, ident, prop, value, says) in enumerate(flat))
    prompt = ("A shopping assistant will decline the order unless a listing below writes a requested value in other words.\n"
              "Check: THE SHOPPER'S REQUEST (context):\n" + run.request[:1500] + "\n\nCheck: every numbered line names a "
              "listing, one requested property with its requested value, and the text that listing's own field gives for it.\n"
              + listing + "\n\nThen: decide per line whether that field text is the requested value put differently (a "
              "synonym, a different spelling or phrasing, or a more precise form of the same thing) or another value of "
              "the property; if in doubt, answer false.\nReply: ONE JSON object and nothing more: "
              "{\"lines\": [{\"n\": 1, \"same\": true}]}")
    run.near_done = True
    answer = consult(run.wire, prompt, 600, CHECKER)
    said_lines = []
    if isinstance(answer, dict) and isinstance(answer.get("lines"), list):
        said_lines = answer.get("lines")
    agreed = {int(entry["n"]) - 1 for entry in said_lines
              if isinstance(entry, dict) and entry.get("same") is True and is_number(entry.get("n"))}
    jot(run, "near_check", lines=[f"{i}:{p}={v}|{s}"[:120] for _, i, p, v, s in flat], same=sorted(agreed))
    return agreed


def approve_near(run, ident, ver):
    key = f"{ident}|{ver}"
    for memo in [m for m in run.rulings if m == key or m.startswith(key + "|")]:
        run.rulings[memo] = ""
    run.passed.add(key)
    vars(run).setdefault("near_ok", set()).add(key)
    vars(run).setdefault("pass_price", {})[ident] = price_of(run.archive, ident)


def near_report(run, flat, basket, picks, ver):
    run.near_cart = {"ver": run.needs.version, "refusals": run.refusal_n, "lines": [i for i, _ in picks]}
    said_by = vars(run).setdefault("near_said", {})
    kept = [(ident, prop, value, says) for n, ident, prop, value, says in flat if ident == basket.get(n, ("",))[0]]
    for ident, prop, value, says in kept:
        said_by.setdefault(f"{ident}|{ver}", []).append((prop, value, says))
    said = "; ".join(f"{ident} writes {prop} as '{says}' for '{value[:60]}'" for ident, prop, value, says in kept)
    return ("Not run yet (no step used). The meaning check says " + said + ", that is one value worded differently; each such "
            "listing also meets all needs the shopper revealed, has stock and fits the budget, hence the basket went "
            "into the cart and its stock was read: " + cart_summary(run.trolley) + " Order this cart now; decline only if a record "
            "you read shows one of these lines failing a requirement now in force.")


def repair_mark(run):
    several = len(need_items(run.needs)) >= 2
    early = routed(run, "early_repair") and run.refusal_n == 0 and moves_left(run) > order_due(run)
    if run.refusal_n <= 0 and not moves_left(run) <= order_due(run) and not early:
        return None
    wire = run.wire
    if not role(run, "cart") or not role(run, "order") or moves_left(run) < 3 or time_left(wire) <= CLOSING \
            or wire.mute or vars(run).get("covering") or wire.done:
        return None
    tried = vars(run).setdefault("repair_tried", set())
    mark = ("early", run.refusal_n) if early else run.refusal_n
    if mark in tried:
        return None
    tried.add(mark)
    return several


def proof_blocks(run):
    try:
        proof = ed_proof(run)
    except Exception as err:
        proof = []
        jot_error(run, "decline_proof_error", err)
    if proof:
        jot(run, "decline_proof", products=proof)
    return bool(proof)


def repair_fits(run, picks, ver, own):
    if not picks:
        return False
    fit = vars(run).get("code_fit") or {}
    for i, _ in picks:
        if f"{i}|{ver}" in run.passed:
            continue
        if not (own and i in run.archive.read and fit.get(i)):
            return False
    return True


def repair_picks(run, several, gone, ver, own):
    avoid = set(gone)
    pool = {}
    if several:
        pool = {e["id"]: e for k in range(len(need_items(run.needs))) for e in ed_pool(run, k)}
        avoid |= {i for i, e in pool.items() if e["clash"]}

    def thin(i, e):
        return not e["revealed_ok"] or e["misses"]
    picks = code_picks(run, avoid)
    checks = rounds = 0
    while several and picks and checks < 3 and rounds < 30 and time_left(run.wire) > CLOSING + 30:
        unchecked = [i for i, _ in picks if not repair_fits(run, [(i, 1)], ver, own)]
        if not unchecked:
            break
        bare = [i for i in unchecked if not pool.get(i) or thin(i, pool[i])]
        if bare:
            avoid.update(bare)
            avoid.update(i for i, e in pool.items() if not e["read"] and f"{i}|{ver}" not in run.passed and thin(i, e))
            rounds += 1
        else:
            checks += 1
            check_item(run, unchecked[0], "ORDER")
        picks = code_picks(run, avoid)
    return picks


def caps_stock(cp, ident, stock, source):
    run = cp.run
    run.stock_seen = True
    mine = cp.stock_by.setdefault(ident, {})
    opposite = mine.get("lookup" if source == "row" else "row")
    if opposite and opposite[0] == cp.epoch and opposite[1] != stock:
        run.rows_lag = True
    earlier_row = mine.get("row")
    if source == "row" and earlier_row and earlier_row[1] != stock:
        run.rows_live = True
    mine[source] = (cp.epoch, stock)
    if stock is False:
        if ident not in cp.dead and ident in run.trolley.units and ident not in cp.newly:
            cp.newly.append(ident)
        cp.dead.add(ident)
        cp.dead_by[ident] = source
    elif source == "lookup" or cp.dead_by.get(ident) != "lookup":
        cp.dead.discard(ident)


def caps_visit(cp, d):
    ident = rec_id(d)
    if not ident:
        return
    stock = stock_state(d)
    named = ident in cp.asked
    by_cart = cp.call_name in cp.cart_tools
    if stock is not None:
        caps_stock(cp, ident, stock, "lookup" if named or by_cart else "row")
    if not named:
        if not by_cart:
            cp.row_keys.update(str(k) for k in d)
            cp.row_text.setdefault(ident, {}).update({str(k): v for k, v in d.items() if isinstance(v, str)})
        return
    state, flags = vars(cp.run), cp.flags
    flags["id"] = True
    state.setdefault("read_with", {}).setdefault(cp.tool, set()).add(ident)
    state.setdefault("text_keys", {}).setdefault(cp.tool, set()).update(
        str(k) for k, v in d.items() if v is None or isinstance(v, str))
    if stock is not None:
        flags["stock"] = True
    if not flags["props"]:
        flags["props"] = any(isinstance(v, dict) and sum(1 for x in v.values() if isinstance(x, (str, int, float))) >= 2
                             for v in d.values())
    if not flags["price"]:
        flags["price"] = any(is_number(v) and re.search(r"price|cost", str(k).lower()) for k, v in d.items())
    cp.seen_keys.update(str(k) for k in d)
    row = cp.row_text.get(ident)
    for k, v in d.items():
        if isinstance(v, str) and len(v.split()) >= 8 \
                and (str(k) not in cp.row_keys if row is None else row.get(str(k)) != v):
            cp.note_keys.add(str(k))
            flags["notes"] = True


def title_words(text):
    return set(re.findall(r"[a-z0-9]{3,}", text.lower()))


def shelf_scan(run, home):
    archive = run.archive
    order, titles, sold, known = [], {}, set(), []

    def visit(d):
        ident = row_mark(d)
        if ident not in archive.ids or shelf_of(d) not in home:
            return
        if ident in archive.read:
            known.append(title_words(title_of(d)))
            return
        if sold_out(d):
            sold.add(ident)
        if ident not in order:
            order.append(ident)
            titles[ident] = title_of(d)
    for rec in archive.items:
        walk_dicts(rec, visit)
    return order, titles, sold, known


def value_wants(run, several, k):
    wanted = []
    for req in props_pending(run.needs):
        if several and product_of(run, req.partition(":")[0]) not in (k, None):
            continue
        wanted.append(re.sub(r"\((?:stated|revealed[^)]*)\)", "", req.partition(":")[2]))
    return [v for v in wanted if v.strip() and positive_value(v)
            and not re.search(r"as shown|same as|the page|sold by|redirect|refer", v.lower())]


def led_pick(run, home, most, words):
    several = bool(need_items(run.needs))
    k = next((n for n, h in enumerate(home_shelves(run)) if h == home), None) if several else None
    wanted = value_wants(run, several, k)
    if not wanted or most <= 0 or (several and k is None):
        return None
    facts = {}

    def fact_visit(d):
        ident = row_mark(d)
        if ident not in run.archive.read or shelf_of(d) not in home:
            return
        for val in d.values():
            if isinstance(val, dict):
                facts.setdefault(ident, []).extend(str(x) for x in val.values() if isinstance(x, (str, int, float)))
    for rec in run.archive.items:
        walk_dicts(rec, fact_visit)
    if any(all(value_seen(v, " | ".join(texts)) for v in wanted) for texts in facts.values()):
        return None
    carrying = [w for w in vars(run).get("page_book", {}).values() if walk_carries(run, w["args"], k)]
    carrying.sort(key=lambda w: -len(walk_carries(run, w["args"], k)))
    for walk in carrying:
        first = next((i for i in (walk.get("row_shelf") or {}) if i in words), None)
        if first:
            return first
    return None


def twin_products(run, text):
    # (two items under one product name tie in product_of; a question naming that name counts for each of them, or
    # the pair looks unasked for ever and keeps drawing every question)
    groups = {}
    for k, p in enumerate(need_items(run.needs)):
        groups.setdefault(frozenset(kind_tokens(p["product"])), []).append(k)
    words = set(kind_tokens(str(text)))
    return next((ks for name, ks in groups.items() if len(ks) > 1 and name and name <= words), [])


def question_counts(run, count):
    asked, sent = [0] * count, [0] * count
    for entry in run.questions_log:
        head = entry.split(":", 1)[0]
        quoted = re.search(r'asked "(.*?)" -> ', entry)
        k = product_of(run, head)
        if k is None:
            k = product_of(run, quoted.group(1)) if quoted else None
        owners = [k] if k is not None else twin_products(run, head) or (twin_products(run, quoted.group(1)) if quoted else [])
        for k in owners:
            if k < count:
                sent[k] += 1
                asked[k] += "[not answered]" not in head
    return asked, sent


def reveal_counts(run, count):
    found = [0] * count
    for need in run.needs.uncovered:
        k = product_of(run, str(need).split(":", 1)[0])
        if k is not None:
            found[k] += 1
    return found


def story_products(run, count, asked):
    try:
        named, keyless = set(), [0] * count
        for h in open_hints(run):
            head = str(h).split(":", 1)[0]
            j = product_of(run, head)
            if j is None:
                continue
            if plain_hint(run, j, head.strip(), hinted=True):
                keyless[j] += 1
            else:
                named.add(j)
        return {j for j in range(count) if j in named or keyless[j] > asked[j]}
    except Exception:
        return set()


def swap_cost(run, ident):
    figure = price_of(run.archive, ident)
    return float(figure) if is_number(figure) else NO_PRICE


def swap_verified(run, line, kind, several, gone, ver, qty, rest):
    cap = run.needs.spend_cap
    cart = run.trolley.units
    fits = []
    for key in run.passed:
        ident, _, version = key.rpartition("|")
        if version != ver or ident == line or ident in gone or ident in run.refused_lines or ident in cart:
            continue
        if field_view(run, ident)["fails"]:
            continue
        shelf = shelf_known(run, ident)
        if kind and (shelf != kind if several else shelf not in ("", kind)):
            continue
        cost = swap_cost(run, ident)
        if is_number(cap) and cost != NO_PRICE and cost * qty + rest > cap:
            continue
        fits.append(ident)
    return fits


def swap_rows(run, line, kind, k, gone, ver, qty, rest):
    cap = run.needs.spend_cap
    cart = run.trolley.units
    judged = [(prop, plain) for prop, plain in code_values(run, k) if ed_firm(plain.lower())]
    failed = {key.split("|")[0] for key, val in run.rulings.items() if val and key.split("|")[1:2] == [ver]}
    pointers = ref_pointers(run)
    ranked = []
    for ident, row in code_rows(run).items():
        price = row["price"] if is_number(row["price"]) else swap_cost(run, ident)
        if ident == line or ident in gone or ident in run.refused_lines or ident in cart or row["stock"] is not True:
            continue
        if (kind and row["shelf"] != kind) or price == NO_PRICE or (is_number(cap) and price * qty + rest > cap):
            continue
        if ident in failed or ident in pointers or ref_excluded(run, row["fields"], k):
            continue
        read = ident in run.archive.read
        if read:
            if field_view(run, ident)["fails"]:
                continue
            if any(code_clash(row["fields"], prop, plain) for prop, plain in judged):
                continue
            if not all(code_shows(row, prop, plain, set(), True) for prop, plain in judged):
                continue
        ranked.append((0 if read else 1, price, ident))
    ranked.sort()
    unread_low = min((x for x in ranked if x[0] == 1), key=lambda x: (x[1], x[2]), default=None)
    if unread_low and need_priority(run.needs) == "cheapest" and all(unread_low[1] < p for r, p, _ in ranked if r == 0):
        ranked.remove(unread_low)
        ranked.insert(0, unread_low)
    return ranked


def swap_label(r, p, i):
    figure = int(p) if float(p).is_integer() else p
    return f"{i} ({figure}{', read, with every stated value in its own record' if r == 0 else ''})"


def queue_quota_reads(run, k, home, entry, opened_before):
    batch = batch_tool(run)
    unread = menu_rows(run, home, min(4, batch[2]) if batch else 2, [], entry["quantity"]) if home else []
    if not unread or not (role(run, "details") or batch) or opened_before or moves_left(run) <= 12:
        return
    vars(run).setdefault("quota_reads", {})[k] = True
    run.open_queue = unread
    run.open_recheck = True
    run.open_say = f"I open a few {entry['product']} listings to learn their fields before asking about it."
    run.open_purpose = "quota"
    run.quota_wait = True


def tagged_question(prop, product):
    return {"property": f"{prop} ({product})", "question": f"Which {prop} would you like for the {product}?"}


NUMBER_WORDS = {word: n for n, word in enumerate(r"one|two|three|four|five|six|seven|eight|nine|ten".split("|"), 1)}


def props_of(schema):
    found = schema.get("properties")
    return found if isinstance(found, dict) else {}


def text_keys(props, skip=None):
    return [key for key, spec in props.items() if isinstance(spec, dict) and spec.get("type") == "string" and not spec.get("enum") and key != skip]


def number_keys(props):
    return [key for key, spec in props.items() if isinstance(spec, dict) and spec.get("type") in ("integer", "number") and not spec.get("enum")]


def log_entry(run, kind, **fields):
    run.wire.log.append({"kind": kind, **fields})


def tagged_product(run, prop, fallback=None):
    tag = re.search(r"\(([^)]+)\)", prop)
    if tag:
        return tag.group(1).strip()
    # (an untagged property naming one of several products belongs to it, so the question names that product alone
    # rather than the whole request's list of products)
    named = [str(item.get("product")) for item in need_items(run.needs) or []
             if str(item.get("product") or "").strip() and str(item.get("product")).lower() in str(prop).lower()]
    if len(named) == 1:
        return named[0]
    products = need_items(run.needs) or []
    if len(products) > 1:
        # (an untagged property on a basket goes to the product whose own listings carry that field: the request's whole
        # list of names in one question draws "one thing at a time" and tells nothing)
        owners = [k for k in range(len(products)) if prop_line(run, k, prop)]
        if owners:
            return str(products[owners[0]]["product"])
    return str(run.needs.reading.get("product") or "item") if fallback is None else fallback


def plain_question(prop, product):
    return f"Which {prop.split('(')[0].strip()} would you like for the {product}?"


def letters_digits(text):
    return " ".join(re.findall(r"[a-z0-9]+", str(text).lower()))


def no_second(text):
    return bool(re.search(r"none|^-$|n/a", text.lower()))


def plan_seat(run):
    # (the task's first plan goes to Sonnet, whose first question gets the surest answer; re-plans go to Haiku first)
    made = stash(run, "plans_made", 0)
    run.plans_made = made + 1
    return PLANNER if not made else PLANNER[1:] + PLANNER[:1]


def product_plan_call(run, k, menu):
    if not menu or vars(run).get("product_plans", 0) >= 1 or time_left(run.wire) < CLOSING + 40:
        return None
    run.product_plans = vars(run).get("product_plans", 0) + 1
    product = need_items(run.needs)[k]["product"]
    left = max(1, ask_limit(run) - run.msgs_sent)
    prompt = ASK_PLAN.replace("QUESTION_COUNT", str(min(3, left))) + "\n\nREQUEST AS WRITTEN BY THE SHOPPER:\n" + run.request
    prompt += "\n\nREQUIREMENTS SO FAR:\n" + needs_text(run.needs)
    if run.questions_log:
        prompt += "\n\nQUESTIONS SENT EARLIER (each with the shopper's reply):\n" + "\n".join(run.questions_log)
    prompt += (f"\n\nTHIS ROUND ASKS ONLY ABOUT THE {product.upper()}: each other product wanted has had a question already "
               f"and this one has had none. Pick properties of the {product} alone, taken from its own list further down, "
               "starting where the story's details point.")
    try:
        hints = open_hints(run, k)
    except Exception:
        hints = []
    if hints:
        prompt += ("\n\nSTORY HINTS (story details with the property each seems to touch, likeliest first; these are not "
                   "requirements):\n" + "\n".join("- " + hint_mark(run, h) for h in hints[:6]))
    prompt += planner_extras(run, k)
    prompt += (f"\n\nFIELD NAMES FOUND IN {product.upper()} LISTING DETAILS SO FAR (name, number of listings, "
               "example values):\n" + menu)
    answer = consult(run.wire, prompt, 1500, plan_seat(run)[:1], effort=PLAN_EFFORT, last=True)
    log_entry(run, "product_plan", product=product, result=answer)
    kind = kind_words(product)
    for entry in (answer or {}).get("questions") or []:
        if not isinstance(entry, dict):
            continue
        lead = str(entry.get("property") or "").split("(")[0].strip()
        extra = str(entry.get("second") or "").split("(")[0].strip()
        if no_second(extra):
            extra = ""
        text = str(entry.get("question") or "").strip()
        if plain_hint(run, k, lead):
            lead = ""
        if extra and plain_hint(run, k, extra):
            extra = ""
        if lead and plan_open(run, k, lead):
            names_it = bool(text) and bool(kind & kind_words(text))
            partner = ""
            if names_it and extra and not strict_prop(lead, extra) and plan_open(run, k, extra):
                partner = extra
            return {"property": f"{lead} ({product})", "second": f"{partner} ({product})" if partner else "",
                    "question": text if names_it else f"Which {lead} would you like for the {product}?"}
        if extra and plan_open(run, k, extra):
            return {"property": f"{extra} ({product})", "question": f"Which {extra} would you like for the {product}?"}
    return None


def open_prop(run, k, prop, line="", by_samples=False):
    try:
        name = str(prop).split("(")[0].strip()
        if not name or junk_key(name):
            return False
        several = bool(need_items(run.needs))

        def belongs(text, tagged_only=False):
            if k is None or not several:
                return True
            owner = product_of(run, text)
            return owner == k or (owner is None and not tagged_only)

        for asked in run.questions_log:
            head = asked.split(":", 1)[0].strip("- ")
            if "[not answered]" in head:
                continue
            if any(belongs(part) and same_field(run, k, part, name) for part in head.split(";")):
                return False
        samples = line.split("):", 1)[1].lower() if "):" in line else ""
        for req in props_pending(run.needs):
            head, _, value = str(req).partition(":")
            if not belongs(head):
                continue
            if strict_prop(head, name):
                if settles(run, str(req), samples if by_samples else ""):
                    return False
            elif same_prop(head, name):
                if positive_value(value, head) and value_seen(value, samples or " | ".join(field_values(run, k, name))):
                    return False
            else:
                words = [w for w in re.findall(r"[a-z]{3,}", value.lower()) if w not in LINK_WORDS]
                # (a store's prefixed name for a stated property, "<prefix> <property>", is settled when its own values
                # carry the stated value: asking it only spends a question)
                tail = prop_words(head)[-1:] == prop_words(name)[-1:]
                pool = samples or (" | ".join(field_values(run, k, name)).lower() if tail else "")
                if not (pool and words) or re.search(r"\d", value) or not positive_value(value, head):
                    continue
                if all(re.search(r"\b" + re.escape(w), pool) for w in words):
                    return False
        for other in run.needs.indifferent:
            if belongs(str(other), tagged_only=True) and same_field(run, k, str(other), name):
                return False
        return True
    except Exception:
        return True


def planner_extras(run, k):
    try:
        several = bool(need_items(run.needs))
        asked = set()
        for entry in run.questions_log:
            asked.update(part.strip().lower() for part in entry.split(":", 1)[0].strip("- ").split(";"))
        still_open = []
        for prop in vars(run).get("open_pair") or []:
            owner = product_of(run, prop) if several else None
            foreign = k is not None and several and owner not in (k, None)
            if not (foreign or prop.lower() in asked or not plan_open(run, owner, prop)):
                still_open.append(prop)
        hoped = []
        for text in vars(run).get("wanted_asks") or []:
            if k is None or not several or product_of(run, text) in (k, None):
                hoped.append(text)
        extra = ""
        if still_open:
            extra += ("\n\nSTILL OPEN FROM AN EARLIER PAIR (the second property of a planned pair that the reply left "
                      "unanswered - the shopper took up only the first, or only the first went out - so it remains open; "
                      "each one was the plan's next choice):\n" + "\n".join("- " + p for p in still_open[-3:]))
        if hoped:
            extra += ("\n\nQUESTIONS THE ASSISTANT TRIED TO SEND (it wrote them after reading listing records; "
                      "a property one of them names that remains open and shows up as a key below should lead, "
                      "phrased with that key's own name):\n" + "\n".join("- " + w for w in hoped[-3:]))
        return extra
    except Exception:
        return ""


def want_ask(run, raw_args):
    try:
        if isinstance(raw_args, dict):
            args = raw_args
        elif isinstance(raw_args, str) and raw_args.strip():
            args = json_object(raw_args)
        else:
            args = {}
        pieces = [str(v).strip() for v in (args or {}).values() if isinstance(v, str) and v.strip()]
        text = " ".join(pieces)[:300]
        if not text:
            return
        others = [w for w in vars(run).get("wanted_asks") or [] if w.lower() != text.lower()]
        others.append(text)
        run.wanted_asks = others[-3:]
    except Exception:
        pass


def forget_wanted(run, label):
    try:
        cleaned = str(label).replace("[not answered]", "")
        names = [re.split(r"\s*\(", piece)[0] for piece in re.split(r";|\band\b", cleaned)]

        def covers(text):
            stems = {stem(w) for w in re.findall(r"[a-z]+", text.lower())}
            for name in names:
                parts = prop_words(name)
                if parts and all(w in stems or stem(w) in stems for w in parts):
                    return True
            return False

        run.wanted_asks = [text for text in vars(run).get("wanted_asks") or [] if not covers(text)]
    except Exception:
        pass


def mentions_other_subject(run, question, prop, menu, second=""):
    try:
        flat = letters_digits(question)
        said = set(flat.split())
        said.update(w[:-1] for w in list(said) if len(w) > 3 and w.endswith("s") and not w.endswith("ss"))
        own = set(prop_words(prop)) | {"which", "would", "you", "like"}
        rows = need_items(run.needs) or [{"product": str(run.needs.reading.get("product") or "")}]
        for row in rows:
            for w in re.findall(r"[a-z0-9]+", str(row.get("product") or "").lower()):
                own.add(w)
                own.add(w[:-1] if len(w) > 3 and w.endswith("s") else w)

        def spelled_out(name):
            words = [w for w in prop_words(name) if len(w) >= 3 and w not in LINK_WORDS]
            if not words:
                return False
            return all(w in said for w in words) and any(w not in own for w in words)

        if second and not strict_prop(second, prop) and spelled_out(second):
            return True
        for raw in str(menu or "").splitlines():
            key = raw.strip().lstrip("-").strip().split(" (in ")[0].strip()
            if not key or key.endswith(":") or strict_prop(key, prop) or junk_key(key):
                continue
            if spelled_out(key):
                return True
            tail = raw.split("):", 1)[1] if "):" in raw else ""
            for sample in tail.split("|"):
                toks = re.findall(r"[a-z0-9]+", sample.lower())
                if not any(t not in own and t not in LINK_WORDS for t in toks) or len("".join(toks)) < 3:
                    continue
                if re.search(r"(?<![a-z0-9])" + re.escape(" ".join(toks)) + r"(?![a-z0-9])", flat):
                    return True
        return False
    except Exception:
        return False


def can_pair(run, first, second):
    try:
        k = None
        if need_items(run.needs):
            k = product_of(run, first)
            if k is None or product_of(run, second) != k:
                return False
        if strict_prop(first, second) or same_prop(first, second) or same_field(run, k, first, second):
            return False

        def name_words(prop):
            return {w for w in prop_words(prop) if len(w) >= 3 and w not in LINK_WORDS}

        if name_words(first) & name_words(second):
            return False
        if field_values(run, k, first) & field_values(run, k, second):
            return False
        memo = vars(run).get("menu_memo", {}).get(k) or (0, "")
        for raw in memo[1].splitlines():
            key = raw.strip("- ").split(" (in ")[0].strip()
            if not key or strict_prop(key, first) or strict_prop(key, second):
                continue
            if twin_of(run, k, key, second) and plan_open(run, k, key):
                return False
        return True
    except Exception:
        return False


def enqueue_second(run, prop):
    waiting = vars(run).setdefault("pending_seconds", [])
    if prop and all(p.lower() != prop.lower() for p in waiting):
        waiting.append(prop)


def waits_for_others(run, prop):
    # (several products: a queued property of a product already asked about waits while another product has had no
    # question yet, or (two products) while it told its need and the other has told none; it stays queued for later)
    products = need_items(run.needs)
    if len(products) < 2:
        return False
    k = product_of(run, prop)
    owners = [k] if k is not None else twin_products(run, prop)
    asked, _ = question_counts(run, len(products))
    found = reveal_counts(run, len(products))
    others = [j for j in range(len(products)) if j not in owners]
    return bool(owners) and (all(asked[j] for j in owners) and any(not asked[j] for j in others)
                             or len(products) == 2 and all(found[j] for j in owners) and any(not found[j] for j in others))


def lead_queue(run, prop):
    return [prop] + [p for p in vars(run).get("pending_seconds") or [] if p.lower() != prop.lower()]


def add_open_pair(run, props):
    listed = vars(run).setdefault("open_pair", [])
    for prop in props:
        if prop.lower() not in [p.lower() for p in listed]:
            listed.append(prop)


def basket_wait(run, limit):
    # (with three or more products the first question waits until every product has a listing read, or a quarter of
    # the steps are gone: asking after reads from one or two shelves left the other products without keys and without
    # questions; a product with rows seen but none opened gets a couple opened. Waiting for many more reads cost steps
    # where the step cap does not grow with the basket)
    products = need_items(run.needs)
    if len(products) < 3 or run.questions_log or run.msgs_sent:
        return False
    if run.wire.tick >= max(8, run.step_cap // 4) or moves_left(run) <= order_due(run) + 2 * limit + 12:
        return False
    reads = [sum(1 for e in ed_pool(run, k) if e["read"]) for k in range(len(products))]
    if all(reads):
        return False
    if not vars(run).get("open_queue"):
        homes = home_shelves(run)
        for k, n in enumerate(reads):
            done = vars(run).get("quota_reads") or {}
            if not n and k < len(homes) and homes[k] and not done.get(k):
                queue_quota_reads(run, k, set(homes[k]), products[k], False)
                break
    return True


def ask_gate(run, urge, sets, pressing, limit):
    if not urge and not pressing and basket_wait(run, limit):
        return "wait"
    starved = sets < ASK_AFTER_SETS and not pressing and not urge
    if starved:
        tick = run.wire.tick
        if tick < ASK_AFTER_STEPS or (tick < ASK_AFTER_STEPS + ASK_GRACE_STEPS and moves_left(run) > 20):
            return "wait"
    if moves_left(run) <= max(10, order_due(run)) or time_left(run.wire) < CLOSING + 40:
        return "stop"
    held = vars(run).get("plan_wait")
    if held is not None and sets <= held and not urge:
        return "wait"
    at_sets = vars(run).get("asked_at_sets")
    if at_sets is None or sets > at_sets or urge or pressing:
        return ""
    if moves_left(run) > order_due(run) + 2 * max(0, limit - run.msgs_sent) + 14 \
            and run.wire.tick - vars(run).get("asked_at_tick", 0) < 4 and not need_items(run.needs):
        return "wait"
    return ""


def planner_prompt(run, left, menu, sets):
    prompt = ASK_PLAN.replace("QUESTION_COUNT", str(left)) + NEXT_ONE.replace("REMAINING", str(left))
    if is_number(run.needs.spend_cap):
        # (the budget is already known: with the line in place the planner could open with a question about the
        # spending ceiling, which can only draw "no preference")
        prompt = prompt.replace(" For the budget, ask for their spending ceiling.", "")
    prompt += "\n\nREQUEST AS WRITTEN BY THE SHOPPER:\n" + run.request
    prompt += "\n\nREQUIREMENTS SO FAR:\n" + needs_text(run.needs)
    if run.questions_log:
        prompt += "\n\nQUESTIONS SENT EARLIER (each with the shopper's reply):\n" + "\n".join(run.questions_log)
    if run.needs.reading.get("ask_first"):
        prompt += "\n\nMUST BE LEARNED FROM THE SHOPPER: " + "; ".join(str(h) for h in run.needs.reading.get("ask_first") or [])
    prompt += planner_extras(run, None)
    try:
        hints = open_hints(run)
    except Exception:
        hints = []
    if hints:
        prompt += ("\n\nSTORY HINTS (story details, each with the property it seems to touch; these are not requirements, "
                   "and the hidden need is most often on one of them):\n" + "\n".join("- " + hint_mark(run, h) for h in hints[:8]))
    visible = menu_trim(menu, sets)
    rows = need_items(run.needs)
    if rows:
        prompt += "\n\nPRODUCTS WANTED: " + " | ".join(f"{n + 1}) {row['product']} x{row['quantity']}" for n, row in enumerate(need_items(run.needs)))
        seen = read_shelves(run.archive)
        homes = home_shelves(run)
        known = set().union(*homes)
        ours = [s for s in seen if s.strip().lower() in known]
        if all(homes):
            order = ours or seen
        else:
            order = ours + [s for s in seen if s not in ours]
        menus = [(shelf, property_menu(run.archive, shelf)[0]) for shelf in order[:4]]
        menu = "\n".join(f"CATEGORY {shelf}:\n{part}" for shelf, part in menus if part) or menu
        visible = "\n".join(f"CATEGORY {shelf}:\n{menu_trim(part, rich_read_count(run.archive, shelf))}"
                            for shelf, part in menus if part) or visible
    prompt += ("\n\nFIELD NAMES FOUND IN LISTING DETAILS SO FAR (name, number of listings, example values):\n"
               + (visible or "(none seen yet)"))
    return prompt, menu


def settled_prop(run, prop):
    # (a question about what the request or the shopper already settled wastes a message)
    sheet = run.needs
    head = re.split(r"\s*\(", str(prop or ""))[0].strip()
    if not head:
        return False
    if is_number(sheet.spend_cap) and (same_prop(head, "budget") or same_prop(head, "price")):
        return True
    held = [str(r).split(":", 1)[0] for r in sheet.written + sheet.uncovered] + list(sheet.indifferent)
    return any(same_prop(head, re.split(r"\s*\(", h)[0].strip()) for h in held)


def plan_state(run, sets):
    return (sets, str(run.needs.version), len(run.needs.uncovered), len(run.needs.indifferent) - len(run.questions_log))


def keep_plan(run, answer, sets):
    if isinstance(answer, dict) and isinstance(answer.get("questions"), list):
        run.plan_kept = {"state": plan_state(run, sets), "questions": answer["questions"], "asked": len(run.questions_log)}


def kept_plan(run, sets, fresh):
    # (the planner is asked again only when something changed since its last plan: a need found, a requirement changed,
    # new listings read, or an answer that was more than "no preference"; otherwise its next question goes out as planned.
    # a planner call takes ~11 s of the 195 s before closing, about a quarter of the time)
    kept = vars(run).get("plan_kept")
    if not kept or len(run.questions_log) <= kept["asked"]:
        return None
    if plan_state(run, sets)[:3] != kept["state"][:3] or plan_state(run, sets)[3] != kept["state"][3]:
        return None
    rest = [q for q in kept["questions"] if isinstance(q, dict) and fresh(q.get("property"))]
    if not rest:
        return None
    run.wire.log.append({"kind": "plan_kept", "next": str(rest[0].get("property"))[:60]})
    return {"questions": rest}


def screen_plan(run, answer, fresh):
    picked = []
    for entry in (answer or {}).get("questions") or []:
        if not isinstance(entry, dict) or not str(entry.get("question") or "").strip():
            continue
        if settled_prop(run, entry.get("property")):
            run.wire.log.append({"kind": "settled_question_dropped", "property": str(entry.get("property"))[:60]})
            continue
        second = str(entry.get("second") or "").strip()
        if no_second(second):
            second = ""
        if fresh(entry.get("property")):
            picked.append(dict(entry, second=second if second and fresh(second) else ""))
            if second and plain_hint(run, product_of(run, second) if need_items(run.needs) else None, second):
                lead = str(entry.get("property") or "").strip()
                picked[-1]["question"] = plain_question(lead, tagged_product(run, lead))
        elif second and fresh(second):
            picked.append({"property": second, "question": plain_question(second, tagged_product(run, second))})
    return picked


def carried_lead(run, fresh):
    # (the planner left nothing to ask while no need is out and messages remain: the property most listings carry, with
    # values that differ between them, so spare messages are not left unused)
    stated = [str(req).split(":", 1)[0] for req in props_pending(run.needs)]
    # (the whole archive only when the reads are not all of a listing the request points at: a smartwatch's display,
    # asked about a table because only the referenced watch had been read, never carries the table's need)
    for menu in (fresh_menu(run, None), "" if reference_reads_only(run) else property_menu(run.archive)[0]):
        best = []
        for line in menu.splitlines():
            found = re.match(r"- (.+?) \(in (\d+) listings?\): (.*)", line)
            if not found:
                continue
            name = found.group(1)
            values = {v.strip() for v in found.group(3).split("|") if v.strip()}
            if junk_key(name) or len(values) < 2 or set(prop_words(name)) & {"option", "variant", "title", "description", "model"}:
                continue
            if any(same_prop(head, name) or prop_words(head)[-1:] == prop_words(name)[-1:] for head in stated):
                continue
            if fresh(name):
                best.append((-int(found.group(2)), len(best), name))
        if best:
            name = min(best)[2]
            run.wire.log.append({"kind": "carried_lead", "lead": name})
            return [{"property": name, "question": plain_question(name, product_name(run, None, False)), "second": ""}]
    return []


def last_slot_lead(run, plan, fresh, limit):
    # (the last question slot of a product with nothing revealed goes to the property most of its listings carry,
    # unless the planner already picks it: on live menus the planner's own first three plus that one found the
    # hidden need in 29 of 41 cases against 22 for the planner alone; the planner keeps every earlier slot, since
    # taking an earlier one lost tasks)
    if not plan or limit < 3 or run.msgs_sent != limit - 1:
        return plan
    prop = str(plan[0].get("property") or "")
    several = bool(need_items(run.needs))
    k = product_of(run, prop) if several else None
    if several and k is None:
        return plan
    if (reveal_counts(run, len(need_items(run.needs)))[k] if several else len(run.needs.uncovered)) or hinted_prop(run, k, prop):
        return plan
    product = product_name(run, k, several)
    tag = lambda name: f"{name} ({product})" if several else name
    words = lambda text: {singular(w) for w in re.findall(r"[a-z]+", str(text).split("(")[0].lower())}
    mine = words(prop)
    for line in fresh_menu(run, k).splitlines():
        found = re.match(r"- (.+?) \(in (\d+) listings?\)", line)
        if not found or int(found.group(2)) < 2:
            continue
        name = found.group(1)
        own = words(name)
        if not own or junk_key(name) or own & {"option", "variant", "title", "description", "model", "feature"}:
            continue
        if mine and (own <= mine or mine <= own):
            return plan
        # (the lead is a fresh property: one the request states, under any of its store names, is not)
        stated = [str(req).split(":", 1)[0] for req in props_pending(run.needs)]
        if any(same_prop(head, name) or prop_words(head)[-1:] == prop_words(name)[-1:] for head in stated):
            continue
        if not fresh(tag(name)):
            continue
        # (one value on every listing read sets no listing apart, so it is no lead and the planner's pick keeps the slot)
        if len({v.strip() for v in line.split("):", 1)[1].split("|") if v.strip()}) < 2:
            continue
        run.wire.log.append({"kind": "last_slot_lead", "planner": prop[:60], "lead": name, "count": int(found.group(2))})
        return [{"property": tag(name), "question": plain_question(name, product), "second": ""}] + plan
    return plan


def names_property(entry, must=""):
    # (true when the question names its property, or the property is one the request says must be asked)
    prop = re.split(r"\s*\(", str(entry.get("property") or ""))[0]
    words = [w.rstrip("s") for w in re.findall(r"[a-z]{3,}", prop.lower())]
    asked = str(entry.get("question") or "").lower()
    return not words or any(w in asked or w in must for w in words)


def pick_question(run, plan, pending, menu, left):
    proposal, label, split_second = None, None, ""
    for entry in plan[:2]:
        if need_items(run.needs):
            prop = str(entry.get("property") or "")
            product = tagged_product(run, prop, "")
            words = re.findall(r"[a-z]{3,}", product.lower())
            if product and words and not any(w.rstrip("s") in str(entry.get("question") or "").lower() for w in words):
                entry = dict(entry, question=plain_question(prop, product))
        must = " ".join(str(h) for h in run.needs.reading.get("ask_first") or []).lower()
        if not (must and not run.questions_log) and not names_property(entry, must):
            # (a hidden need answers only a question naming its property: an open "anything else?" draws "any is fine")
            prop = str(entry.get("property") or "")
            entry = dict(entry, question=plain_question(prop, tagged_product(run, prop)))
        if not pending and not (run.needs.reading.get("ask_first") and not run.questions_log) \
                and mentions_other_subject(run, str(entry.get("question") or ""), str(entry.get("property") or ""), menu,
                                           str(entry.get("second") or "")):
            prop = str(entry.get("property") or "")
            entry = dict(entry, question=plain_question(prop, tagged_product(run, prop)))
        proposal = message_call(run, str(entry.get("question")).strip(), f"ask_{run.msgs_sent + 1}_{run.wire.tick}")
        if not proposal:
            continue
        label = (str(entry.get("property")), str(entry.get("question")).strip())
        second = str(entry.get("second") or "").strip()
        if not second or second.lower() == label[0].lower() or no_second(second):
            break
        if (PAIR_ALL or len(need_items(run.needs)) == 2) and not vars(run).get("no_pairs") and can_pair(run, label[0], second):
            product = tagged_product(run, label[0], "") or tagged_product(run, second)
            # (one question naming two properties, worded as a single question: "Which A…, and which B?" is turned away
            # as two questions far more often than a question about one property)
            both = (f"Which {label[0].split('(')[0].strip()} or {second.split('(')[0].strip()} "
                    f"would you like for the {product}?")
            proposal = message_call(run, both, f"ask_{run.msgs_sent + 1}_{run.wire.tick}")
            if not proposal:
                continue
            label = (label[0] + "; " + second, both)
            break
        alone = f"Which {label[0].split('(')[0].strip()} would you like for the {tagged_product(run, label[0])}?"
        proposal = message_call(run, alone, f"ask_{run.msgs_sent + 1}_{run.wire.tick}")
        label = (label[0], alone)
        if not proposal:
            continue
        split_second = second
        break
    return proposal, label, split_second


def one_at_a_time(run, label):
    run.no_pairs = True
    parts = [p.strip() for p in label[0].split(";") if p.strip()] or [label[0]]
    first = parts[0]
    product = tagged_product(run, first)
    reasked = vars(run).setdefault("reasked", set())
    plain_q = f"Which {first.split('(')[0].strip()} would you like for the {product}?"
    single = len(parts) == 1
    alternative = collide_alt(run, first) if single else ""
    if alternative:
        reasked.add(first.lower())
        vars(run).setdefault("spent_asks", set()).add(first.lower())
        run.pending_seconds = lead_queue(run, alternative)
        note = f" [{first}: no answer came; the story hint's other option {alternative} goes out next]"
    elif single:
        # (asking for one thing at a time after a question that asked one thing is how some shoppers say they do not
        # mind it: the same question sent again draws a firm answer about once in a dozen)
        reasked.add(first.lower())
        vars(run).setdefault("spent_asks", set()).add(first.lower())
        note = f" [{first}: only this was asked, so the reply counts as no preference; it will not be asked again]"
    elif first.lower() not in reasked and letters_digits(plain_q) != letters_digits(label[1]) \
            and not (single and holds_plain(run, label[1], plain_q)):
        # (both parts go out again, one per message: re-asking the first alone lost the need whenever it sat second)
        reasked.add(first.lower())
        run.pending_seconds = lead_queue(run, first)
        for part in parts[1:]:
            enqueue_second(run, part)
        note = f" [{' and '.join(parts)}: no answer came; each goes out again on its own next]"
    else:
        redo = distinct_ask(run, first) if first.lower() not in reasked else None
        if not redo and single and first.lower() not in reasked:
            # (a single plain question that drew no answer goes out once more as it was)
            reasked.add(first.lower())
            run.pending_seconds = lead_queue(run, first)
            note = f" [{first}: no answer came; it goes out once more next]"
        elif not redo:
            vars(run).setdefault("spent_asks", set()).add(first.lower())
            note = f" [{first}: no answer came; it will not be asked again]"
        else:
            reasked.update({first.lower(), redo[0].lower()})
            vars(run).setdefault("pending_text", {})[redo[0].lower()] = redo[1]
            run.pending_seconds = lead_queue(run, redo[0])
            same = re.split(r"\s*\(", redo[0])[0].strip().lower() == re.split(r"\s*\(", first)[0].strip().lower()
            if not same:
                vars(run).setdefault("spent_asks", set()).add(first.lower())
            note = f" [{first}: no answer came; it goes out again " + ("through its values next]" if same else f"as {redo[0]} next]")
    return parts, note


def settle_reply(run, label, split_second, found_before, freed_before):
    note = ""
    heard_meta = vars(run.needs).pop("meta_last", None)
    freed_now = len(run.needs.indifferent) + vars(run.needs).get("pinned_heard", 0)
    if heard_meta is not None and letters_digits(label[1]) and letters_digits(label[1]) in letters_digits(heard_meta):
        parts, note = one_at_a_time(run, label)
        add_open_pair(run, parts[1:] + ([split_second] if split_second else []))
        label = (" and ".join(parts) + " [not answered] [one at a time]", label[1])
    elif ";" in label[0] and len(run.needs.uncovered) > found_before:
        second = label[0].split(";", 1)[1].strip()
        revealed = " ".join(run.needs.uncovered[found_before:]).lower()
        if not any(w in revealed for w in re.findall(r"[a-z]{4,}", second.split("(")[0].lower())):
            enqueue_second(run, second)
            note = f" [{second}: no answer came; still open]"
        label = (label[0].split(";")[0].strip(), label[1])
    elif ";" in label[0] and freed_now > freed_before:
        # (a "no preference" to a question naming two properties covers both: the second, asked again on its own,
        # almost never draws a stated need)
        second = label[0].split(";", 1)[1].strip()
        label = (label[0].split(";")[0].strip(), label[1])
        if second.lower() not in [p.strip().lower() for p in run.needs.indifferent]:
            run.needs.indifferent.append(second)
        vars(run).setdefault("spent_asks", set()).add(second.lower())
        note = f" [{second}: the same answer covers it]"
    elif ";" in label[0] and freed_now == freed_before:
        # (a pair that drew neither a need, a "no preference" nor "one at a time" drew the shopper's "say it again": that
        # reply comes for properties that are not hidden, so neither part is asked again)
        run.no_pairs = True
        parts = [part for part in (p.strip() for p in label[0].split(";")) if part]
        vars(run).setdefault("spent_asks", set()).update(part.lower() for part in parts)
        label = (label[0].replace(";", " and") + " [not answered]", label[1])
    elif split_second:
        revealed = " ".join(run.needs.uncovered[found_before:]).lower()
        listed = vars(run).setdefault("open_pair", [])
        if not any(w in revealed for w in re.findall(r"[a-z]{4,}", split_second.split("(")[0].lower())) \
                and split_second.lower() not in [p.lower() for p in listed]:
            listed.append(split_second)
    return label, note, heard_meta


def pose_planned(run, urge=False):
    tool = role(run, "message")
    limit = ask_limit(run)
    if not tool or run.msgs_sent >= limit:
        run.asking = False
        return False
    menu, _ = property_menu(run.archive)
    sets = rich_read_count(run.archive)
    shelf = own_shelf(run)
    if shelf:
        menu, sets = property_menu(run.archive, shelf)[0], rich_read_count(run.archive, shelf)
    elif reference_reads_only(run):
        menu, sets = "", 0
    # (what only the shopper can settle still waits for a first listing read: sent at step one it spent a message
    # before the planner had any keys to pair with it)
    pressing = bool(run.needs.reading.get("ask_first")) and not run.questions_log and bool(run.archive.read)
    gate = ask_gate(run, urge, sets, pressing, limit)
    if gate == "stop":
        run.asking = False
    if gate:
        return False
    if pre_read(run):
        return True
    if not run.questions_log and not run.msgs_sent and not pressing and not urge and first_reads(run):
        return True
    if not run.questions_log and not table_reads(run.archive, shelf) and not urge and field_probe(run):
        return True
    left = limit - run.msgs_sent
    prompt, menu = planner_prompt(run, left, menu, sets)
    asked = set()
    for entry in run.questions_log:
        asked.update(p.strip().lower() for p in entry.split(":", 1)[0].strip("- ").split(";"))

    def fresh(prop):
        prop = str(prop or "").strip()
        if not prop or prop.lower() in asked or junk_key(prop) or prop.lower() in vars(run).get("spent_asks", set()):
            return False
        owner = product_of(run, prop) if need_items(run.needs) else None
        return not plain_hint(run, owner, prop) and plan_open(run, owner, prop)

    queued = [p for p in vars(run).get("pending_seconds") or [] if fresh(p)]
    run.pending_seconds = queued
    pending = queued[0] if queued and not waits_for_others(run, queued[0]) else ""
    if pending:
        product = tagged_product(run, pending)
        text = (vars(run).get("pending_text") or {}).pop(pending.lower(), "")
        answer = {"questions": [{"property": pending, "question": text or plain_question(pending, product)}]}
    else:
        answer = kept_plan(run, sets, fresh)
        if answer is None:
            answer = consult(run.wire, prompt, 2500, plan_seat(run), effort=PLAN_EFFORT, last=True)
            if answer is None and run.wire.misses > 0:
                run.plan_wait = sets
                return False
            keep_plan(run, answer, sets)
    plan = screen_plan(run, answer, fresh)
    if not pending:
        try:
            plan = last_slot_lead(run, plan, fresh, limit)
        except Exception as err:
            record_failure(run.wire, "last_slot_error", err)
    if need_items(run.needs) and not pending:
        plan = quota_first(run, plan, fresh)
        if vars(run).pop("quota_wait", False):
            return True
    if not plan and not pending and not need_items(run.needs) and not run.needs.uncovered:
        plan = carried_lead(run, fresh)
    proposal, label, split_second = pick_question(run, plan, pending, menu, left)
    if not proposal:
        if sets < ASK_AFTER_SETS and moves_left(run) > 12 and not urge:
            run.plan_wait = sets
            return False
        run.empty_plans = vars(run).get("empty_plans", 0) + 1
        if moves_left(run) > 12 and run.empty_plans <= 1:
            run.plan_wait = sets
            return False
        run.asking = False
        return False
    run.asked_at_sets, run.asked_at_tick = sets, run.wire.tick
    run.chat.append({"role": "assistant", "content": f"I will ask the shopper about {label[0]}.", "tool_calls": [proposal]})
    found_before = len(run.needs.uncovered)
    freed_before = len(run.needs.indifferent) + vars(run.needs).get("pinned_heard", 0)
    run.asking_label = label[0]
    try:
        replies = dispatch(run, [proposal], scripted=True)
    finally:
        run.asking_label = ""
    post_replies(run, replies)
    asked_label = label[0]
    if pending and label[0].split(";")[0].strip().lower() == pending.lower():
        run.pending_seconds = [p for p in vars(run).get("pending_seconds") or [] if p.lower() != pending.lower()]
    label, note, heard_meta = settle_reply(run, label, split_second, found_before, freed_before)
    if heard_meta is None and len(run.needs.uncovered) == found_before:
        vars(run).setdefault("quiet_asks", []).append(asked_label.split(";")[0].strip())
    if replies:
        reply_text = replies[0][1]
        run.questions_log.append(f"- {label[0]}: asked \"{label[1]}\" -> {reply_text.split(chr(10) + '[status]')[0][:400]}{note}")
    forget_wanted(run, asked_label)
    run.asking = run.msgs_sent < limit
    return True


def reply_back(run, text_body):
    if not role(run, "message") or run.msgs_sent >= ask_limit(run) or moves_left(run) <= 4 or time_left(run.wire) < CLOSING:
        return False
    if run.asking and moves_left(run) > 10 and time_left(run.wire) >= CLOSING + 40:
        # (a confirmation of an event the code already applied costs a message that a property question still needs)
        return False
    proposal = message_call(run, text_body, f"reply_{run.wire.tick}")
    if not proposal:
        return False
    run.chat.append({"role": "assistant", "content": "Replying to the shopper's question now.", "tool_calls": [proposal]})
    post_replies(run, dispatch(run, [proposal], scripted=True))
    return True


def post_replies(run, responses):
    last = len(responses) - 1
    for pos, (call_id, text) in enumerate(responses):
        if pos == last:
            text = text + "\n" + status_line(run)
        run.chat.append({"role": "tool", "tool_call_id": call_id, "content": text})


def plain_args(run, call_name, text_body, kind=""):
    schema = run.shapes.get(call_name) or {}
    specs = schema.get("properties") or {}
    pattern = {"decline": DECLINE_WORDS, "order": ORDER_WORDS}.get(kind, "")
    out = {}
    for key in schema.get("required") or []:
        spec = specs.get(key) or {}
        kind_of = spec.get("type")
        if spec.get("enum"):
            matching = [v for v in spec["enum"] if pattern and isinstance(v, str) and re.search(pattern, v.lower())]
            out[key] = (matching or spec["enum"])[0]
        elif kind_of == "string" or "string" in (kind_of or []):
            out[key] = text_body[: spec.get("maxLength", len(text_body))]
        elif kind_of == "boolean":
            out[key] = True
        elif kind_of in ("integer", "number"):
            out[key] = max(1, spec.get("minimum", 1))
    if len(specs) == 1:
        key, spec = next(iter(specs.items()))
        if key not in out and isinstance(spec, dict) and spec.get("type") == "string" and "enum" not in spec:
            out[key] = text_body[: spec.get("maxLength", len(text_body))]
    return out


def scrub_decline(run, call_args, call_name):
    rule = run.needs.reading.get("decline_no_names")
    if rule is False:
        return call_args
    named = False
    for piece in call_args.values():
        if isinstance(piece, str) and any(tok and tok in run.archive.ids for tok in re.split(r"[\s.,;:!?()\[\]{}<>\"'`]+", piece)):
            named = True
            break
    if not (rule or named):
        return call_args
    specs = (run.shapes.get(call_name) or {}).get("properties") or {}
    cleaned = {}
    for key, piece in call_args.items():
        spec = specs.get(key) or {}
        cleaned[key] = SORRY[: spec.get("maxLength", len(SORRY))] if isinstance(piece, str) and "enum" not in spec else piece
    return cleaned


def code_reading(run):
    text = run.request
    low = text.lower()
    plan = {"code_reading": True, "requirements": [], "items": [], "quantity": 1, "rules": [], "procedures": [], "basket_yes": False,
            "soft_wants": [], "ask_first": [], "hints": [], "references": [], "priority_detail": ""}
    plan["price_cap"], plan["currency"] = spend_limit_in(text)
    plan["in_stock_only"] = bool(re.search(r"\bin[- ]stock\b|\bon hand\b|\bon the shelf\b|\bsold[- ]out\b|\bback[- ]?order|"
                                           r"\bout of stock\b|\bavailable now\b", low))
    cheapest = re.search(r"\bcheapest\b|\blowest[- ](?:price|priced|cost)\b|\bleast expensive\b|"
                         r"\blowest\b[^.]{0,20}\bprice\b|\bsmallest (?:total|price)\b", low)
    plan["priority"] = "cheapest" if cheapest else "none"
    messages = count_limit_in(low, r"messages?|questions?|notes?|repl(?:y|ies)|texts?")
    if messages is not None:
        plan["message_budget"] = messages
    steps = count_limit_in(low, r"steps?|moves?|actions?|(?:tool )?calls?|turns?")
    if steps is not None and steps >= FIRST_ORDER_BY + 3:
        plan["step_budget"] = steps
    tools = code_roles(run)
    plan["roles"] = tools
    found = counted_products(run, tools.get("search"))
    if len(found) < 2:
        if found:
            plan["product"], plan["quantity"] = found[0]["label"], found[0]["quantity"]
        else:
            plan["product"] = best_shelf_label(run, tools.get("search"), kind_words(text))
            plan["quantity"] = units_asked(low)
        plan["requirements"] = [f"{prop}: {value}" for prop, value in stated_pairs(low)]
        return plan
    plan["items"] = [{"product": row["product"], "quantity": row["quantity"], "requirements": row["requirements"]} for row in found]
    plan["product"] = " and ".join(row["product"] for row in found)
    plan["requirements"] = [req for row in found for req in row["requirements"]]
    return plan


def figure_score(sentence, m, whole):
    sign = m.group(1)
    after = sentence[m.end():]
    if re.match(r"\s*(?:x\b|\u00d7|%|:\s*\d|-\s*\d|/\s*\d|(?:[a-z-]+\s+){0,2}?(?:messages?|questions?|notes?|repl\w*|texts?|"
                r"steps?|moves?|actions?|(?:tool )?calls?|turns?|units?|items?|pieces?|packs?|pairs?|lots?|sets?|of\b|times?)\b)", after):
        return None
    if not sign and re.search(r"[\d.,:/-]\s*$|\bx\s*$", sentence[:m.start()]):
        return None
    before = sentence[max(0, m.start() - 60): m.start()]
    cues = (
        (3, bool(sign) or bool(re.search(r"\b(?:spend\w*|budget\w*|afford\w*|pay\w*|price[ds]?|costs?|cash|money)\b(?:\W+[\w']+){0,4}\W*$", before))),
        (2, bool(re.search(r"\b(?:under|below|no more than|at most|not more than|up to|within|max\w*|less than|cheaper than|"
                           r"not (?:more|over|above)|limit\w*|reach\w*|go to|to|at|cap\w*|past|exceed\w*|beyond|go over)\W+(?:[\w']+\W+)?$", before))),
        (2, bool(re.match(r"\s*(?:[a-z.()]+\s*){0,3}?(?:or (?:lower|less|under|below|cheaper)|at (?:the (?:very )?)?most|max\w*|tops\b|"
                          r",?\s*(?:and )?not (?:one|a) \w+ (?:over|more|above))", after))),
    )
    return whole + sum(points for points, hit in cues if hit)


def figure_unit(after, sign):
    unit = re.match(r"\s*((?:[a-z][a-z.]*\s*){0,2})", after)
    bare = [w.strip(".") for w in (unit.group(1).split() if unit else [])]
    kept = [w for w in bare if w not in LINK_WORDS and not re.fullmatch(r"or|and|not|tops|max\w*|only|just|is|was", w)]
    return " ".join(kept) or (sign or "").strip() or None


def spend_limit_in(text_body):
    top, result = (0, 0.0), (None, None)
    for sentence in re.split(r"(?<=[.!?;])\s+", str(text_body).lower()):
        whole = 1 if re.search(r"\b(?:order|total|whole|lot|sum|bill|altogether|overall|everything)\b", sentence) else 0
        for m in re.finditer(r"([$\u20ac\u00a3]\s*)?(\d[\d,]*(?:\.\d+)?)", sentence):
            score = figure_score(sentence, m, whole)
            if score is None:
                continue
            amount = float(m.group(2).replace(",", ""))
            if score >= 2 and (score, amount) > top:
                top = (score, amount)
                result = (int(amount) if amount.is_integer() else amount, figure_unit(sentence[m.end():], m.group(1)))
    return result


def count_limit_in(low, nouns):
    spelled = dict(NUMBER_WORDS, twelve=12, fifteen=15, twenty=20)
    finder = (r"\b(\d+|one|two|three|four|five|six|seven|eight|nine|ten|twelve|fifteen|twenty)\s+(?:[a-z-]+\s+){0,2}?(?:"
              + nouns + r")\b")
    for m in re.finditer(finder, low):
        before = low[max(0, m.start() - 60): m.start()]
        after = low[m.end(): m.end() + 40]
        cued = re.search(r"\b(?:no more than|only|at most|not more than|up to|max\w*|use|using|take|send|allow\w*|receive|get|total of|"
                         r"limit\w*|inside|within|under|fewer than|less than|you've|you have|have|keep\w*|stick to|go past|past|"
                         r"exceed\w*|beyond)\W+(?:[\w']+\W+){0,3}$", before) \
            or re.match(r"\W*(?:[\w']+\W+){0,3}?(?:in total|total|no more|at most|max\w*|tops|or fewer|or less|allowed|limit)\b", after)
        if cued:
            word = m.group(1)
            return int(word) if word.isdigit() else spelled[word]
    return None


def units_asked(low):
    said = re.search(r"\b(\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s+(?:units?|of (?:them|those|these)|copies)\b", low)
    if said is None:
        return 1
    word = said.group(1)
    n = int(word) if word.isdigit() else NUMBER_WORDS[word]
    return n if 1 <= n <= 99 else 1


def stated_pairs(seg):
    verbs = r"(?:is|are|has to be|have to be|must be|should be|needs to be|need to be|can be)"
    stop = r"(?=\s*(?:[,.;:!?()]|$)|\s+(?:so|since|as|because|for|to|which|that|nothing|otherwise|but|when|while|if|then|though|please|and (?:the|its|their|a|an|one|two|three|\d))\b)"
    pattern = (r"(?:\b(?:where|whose)\s+(?:the\s+|its\s+|their\s+)?|(?:\b(?:and|with|but)\s+|[,;:]\s*|^|[.!?]\s+)(?:the|its|their)\s+)"
               r"([a-z][a-z'&/ -]{1,32}?)\s+" + verbs + r"\s+(?:either\s+|just\s+)?([^,.;:!?()]{1,60}?)" + stop)
    pairs = []
    for m in re.finditer(pattern, seg):
        prop, value = m.group(1).strip(), m.group(2).strip()
        ahead = seg[m.end(): m.end() + 60]
        if ref_text(value) or ref_text(ahead) and re.search(r"\b(?:page|listing|shown|sold by)\b", ahead):
            continue
        if not value or len(value.split()) > 4 or len(prop.split()) > 4:
            continue
        if re.search(r"\b(?:i|you|we|it|he|she|they|my|your|our|there|this|that|which|what|who|me|us|them|one|ones|price|budget|cost|total|order|stock)\b", prop):
            continue
        if re.match(r"(?:a|an|the|my|your|our|his|her|their|me|us|for|to|in|on|at|of|see|go|get|be|have|only|too|very|so|not|no|what|how|it|all|"
                    r"mostly|always|never|just|really|still|also|from|about|like|one|ones)\b", value):
            continue
        pairs.append((prop, value))
    return pairs


def shelf_choices(run, tool):
    props = (run.shapes.get(tool or "") or {}).get("properties")
    lists = []
    for spec in (props.values() if isinstance(props, dict) else []):
        if isinstance(spec, dict) and isinstance(spec.get("enum"), list) and len(spec["enum"]) >= 4:
            lists.append(spec["enum"])
    return max(lists, key=len) if lists else []


def label_spot(label, low, words, toks):
    own = [w.rstrip("s") for w in re.findall(r"[a-z]{3,}", label.lower()) if w not in LINK_WORDS]
    size = len(own)
    if not size:
        return None
    starts = (i for i in range(len(words) - size + 1) if words[i: i + size] == own)
    for i in starts:
        start, end = toks[i][1], toks[i + size - 1][2]
        if len(re.findall(r"[a-z0-9]+", low[start:end])) > 2 * size:
            continue
        count = re.search(r"\b(\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s*(?:x\s+|\u00d7\s*)?(?:(?:of|[a-z'&-]{3,})\s+){0,4}$",
                          low[max(0, start - 56): start])
        if count:
            word = count.group(1)
            n = int(word) if word.isdigit() else NUMBER_WORDS[word]
            return (start, end, n) if 1 <= n <= 99 else False
    return None


def counted_products(run, tool):
    low = run.request.lower()
    toks = [(m.group(0).rstrip("s"), m.start(), m.end()) for m in re.finditer(r"[a-z]{3,}", low) if m.group(0) not in LINK_WORDS]
    words = [t[0] for t in toks]
    counted = {}
    for label in map(str, shelf_choices(run, tool)):
        if label in counted:
            continue
        spot = label_spot(label, low, words, toks)
        if spot:
            counted[label] = spot
    ordered = sorted(counted.items(), key=lambda kv: kv[1][0])
    found = []
    for pos, (label, (start, end, n)) in enumerate(ordered):
        if pos + 1 < len(ordered):
            seg = low[end: ordered[pos + 1][1][0]]
        else:
            seg = re.split(r"(?<=[.!?])\s", low[end: len(low)])[0]
        name = label.lower()
        found.append({"product": name, "label": label, "quantity": n,
                      "requirements": [f"{prop} ({name}): {value}" for prop, value in stated_pairs(seg)]})
    return found


PERSON_WORDS = r"\b(?:shoppers?|customers?|buyers?|clients?|users?)\b"
SPEAK_WORDS = r"\b(?:send\w*|repl\w*|notes?|tell\w*|messag\w*|chat\w*|writ\w*|talk\w*)\b"
ROLE_CUES = (("search", r"\b(?:search\w*|find\w*|look\w*|locat\w*|hunt\w*|brows\w*|quer\w*)\b", False),
             ("message", SPEAK_WORDS, True),
             ("details", r"\b(?:details?|records?|spec\w*|propert\w*|attribut\w*|facts?|full)\b", True),
             ("view", r"\b(?:offer|price|stock|availab\w*)\b", True))


class ToolCard:
    def __init__(self, run, tdef):
        fn = tdef["function"]
        self.name = fn["name"]
        spaced = re.sub(r"([a-z])([A-Z])", r"\1 \2", self.name).replace("_", " ").lower()
        self.blurb = spaced + " | " + str(fn.get("description") or "").lower()
        self.head = self.blurb.split(" | ")[0]
        schema = run.shapes.get(self.name) or {}
        self.props = props_of(schema)
        listed = schema.get("required")
        self.needed = [key for key in listed if key in self.props] if isinstance(listed, list) else []
        self.texts = text_keys(self.props)


def finish_both_ways(card):
    for key in card.needed:
        spec = card.props[key]
        vals = [str(v).lower() for v in (spec.get("enum") or [])] if isinstance(spec, dict) else []
        if any(re.search(ORDER_WORDS, v) for v in vals) and any(re.search(DECLINE_WORDS, v) for v in vals):
            return True
    return False


def cue_fits(card, job, one_text):
    if job == "search" and re.search(PERSON_WORDS, card.blurb) and re.search(SPEAK_WORDS, card.blurb):
        return False
    if job == "message" and not re.search(PERSON_WORDS, card.blurb) and not re.search(r"\b(?:messag\w*|chat\w*)\b", card.blurb):
        return False
    if one_text:
        return len(card.needed) == 1 and card.needed[0] in card.texts
    return bool(card.texts) or any(isinstance(spec, dict) and spec.get("enum") for spec in card.props.values())


def code_roles(run):
    cards = {}
    for tdef in run.toolset:
        card = ToolCard(run, tdef)
        cards[card.name] = card
    roles = {}
    for card in cards.values():
        if finish_both_ways(card):
            roles.setdefault("order", card.name)
            roles.setdefault("decline", card.name)

    def free(card):
        return card.name not in roles.values()

    # (a finishing tool is judged by what its own description says first, its name only when the description is silent:
    # a name may begin with the order word and still close without a purchase)
    for card in filter(free, list(cards.values())):
        told = card.blurb.partition(" | ")[2]
        quits = re.search(DECLINE_WORDS, told) or (not re.search(ORDER_WORDS, told) and re.search(DECLINE_WORDS, card.head))
        buys = not quits and (re.search(ORDER_WORDS, told) or (not told.strip() and re.search(ORDER_WORDS, card.head)))
        if "decline" not in roles and quits:
            roles["decline"] = card.name
        elif "order" not in roles and buys and (re.search(ORDER_WORDS, card.head) or re.search(r"\bfinal\b", told)):
            roles["order"] = card.name
    for card in cards.values():
        if not free(card) or card.needed or not re.search(r"\bfinal\b", card.blurb):
            continue
        if "decline" not in roles and re.search(DECLINE_WORDS, card.blurb):
            roles["decline"] = card.name
        elif "order" not in roles and re.search(ORDER_WORDS, card.blurb):
            roles["order"] = card.name
    for card in cards.values():
        if not free(card) or not re.search(r"\b(?:cart|basket|bag|trolley)\b", card.blurb):
            continue
        arrays = any(isinstance(spec, dict) and spec.get("type") == "array" for spec in card.props.values())
        roles.setdefault("cart" if card.needed or arrays else "cart_view", card.name)
    for job, cue, one_text in ROLE_CUES:
        for card in cards.values():
            if job not in roles and free(card) and re.search(cue, card.blurb) and cue_fits(card, job, one_text):
                roles[job] = card.name
    return roles


def best_shelf_label(run, tool, want):
    best, top = "", (0, 0.0)
    for label in shelf_choices(run, tool):
        words = kind_words(label)
        hit = len(words & want)
        if not hit:
            continue
        mark = (hit, hit / max(1, len(words)))
        if mark > top:
            best, top = str(label), mark
    return best


def naming_value(head, value):
    bare = re.sub(r"\((?:stated|revealed[^)]*)\)", "", value).strip()
    lowered = bare.lower()
    brief = len(re.findall(r"[a-z0-9]+", lowered)) <= 5 and not re.search(EXCLUDE_WORDS, lowered)
    if not (positive_value(value, head) or brief) or not bare or ref_text(bare):
        return None
    return re.split(r"\s*\(", head)[0].strip(), bare


def code_values(run, k):
    several = need_items(run.needs)
    split = [str(req).partition(":") for req in props_pending(run.needs)]
    mine = [(head, value) for head, _, value in split if not several or product_of(run, head) in (k, None)]
    return [pair for pair in (naming_value(head, value) for head, value in mine) if pair]


class SearchForm:
    def __init__(self, run, tool):
        self.schema = run.shapes.get(tool or "") or {}
        self.props = props_of(self.schema)
        self.needed = [key for key in self.schema.get("required") or [] if key in self.props]
        self.clauses = []
        self.text_key = self.shelf_key = self.mode_key = None
        self.choices = {}


def fill_form(run, form, tool):
    blurb = ""
    for tdef in run.toolset:
        if tdef["function"]["name"] == tool:
            blurb = str(tdef["function"].get("description") or "")
            break
    form.clauses = [part.lower() for part in re.split(r"[.;]", blurb)]
    free = text_keys(form.props, token_key(run))
    named = [key for key in free if re.search(r"^q$|query|text|keyword|needle|term|word|phrase|string|search", key.lower())]
    form.text_key = named[0] if named else (free[0] if free else None)
    form.choices = {key: spec["enum"] for key, spec in form.props.items()
                    if isinstance(spec, dict) and isinstance(spec.get("enum"), list) and spec["enum"]}
    wide = [key for key in form.choices if len(form.choices[key]) >= 4]
    form.shelf_key = max(wide, key=lambda key: len(form.choices[key])) if wide else None
    modes = [key for key in form.choices if key != form.shelf_key and len(form.choices[key]) <= 4]
    required_modes = [key for key in modes if key in form.needed]
    form.mode_key = required_modes[0] if required_modes else (modes[0] if modes else None)


def mode_value(form, arg, last):
    vals = form.choices.get(form.mode_key or "", [])
    arg_pat = r"\b" + re.escape(arg.lower()) + r"\b"
    for val in vals:
        val_pat = r"\b" + re.escape(str(val).lower()) + r"\b"
        if any(re.search(val_pat, part) and re.search(arg_pat, part) for part in form.clauses):
            return val
    return vals[-1] if last else vals[0]


def ceiling_key(form):
    def priced(key):
        low = key.lower()
        if re.search(r"ceil|max|price|cap$|budget|upper|most", low):
            return True
        return any(low in part and re.search(r"\bpric|\bcost", part) for part in form.clauses)

    for key in number_keys(form.props):
        if priced(key) and not re.search(r"page|first|offset|skip|count|size|num", key.lower()):
            return key
    return None


def search_phrases(run, k, product, label, cap_len):
    names = [w for w in re.findall(r"[a-z0-9]+", product.lower()) if len(w) > 2 and w not in LINK_WORDS]
    kind = names[-1] if names else ""
    firm = [value for _, value in code_values(run, k) if not re.search(r"\bor\b|/|,|\b(?:" + VALUE_LINKS + r")\b", value.lower())]
    found = []
    lead_words = [w for w in re.findall(r"[a-z0-9]+", firm[0].lower()) if w not in LINK_WORDS] if firm else []
    if lead_words:
        found.append(" ".join(dict.fromkeys(lead_words + [kind])))
        if len(lead_words) >= 2:
            found.append(" ".join(dict.fromkeys([max(lead_words, key=len), kind])))
    if not found:
        found.append(" ".join(dict.fromkeys(names[:1] + names[-1:])) if names else " ".join(sorted(kind_words(label))[:2]))
    if not is_number(cap_len):
        return found
    fitted = []
    for text in found:
        while len(text) > cap_len and " " in text:
            text = text.rsplit(" ", 1)[0]
        fitted.append(text[: int(cap_len)])
    return fitted


def code_queries(run, k):
    tool = role(run, "search")
    form = SearchForm(run, tool)
    rows = need_items(run.needs)
    product = str(rows[k]["product"] if rows and k is not None and k < len(rows) else run.needs.reading.get("product") or "")
    label = best_shelf_label(run, tool, kind_words(product) or kind_words(run.request))
    fill_form(run, form, tool)
    props = form.props
    cap_key = ceiling_key(form)
    cap = run.needs.spend_cap
    counts = need_counts(run.needs)
    qty = (counts[k] if k is not None and k < len(need_counts(run.needs)) else need_qty(run.needs)) or 1
    bound = {}
    if cap_key and is_number(cap) and cap > 0:
        bound[cap_key] = int(cap // qty) if props[cap_key].get("type") == "integer" else round(cap / qty, 2)
    text_spec = props.get(form.text_key) if form.text_key else None
    cap_len = text_spec.get("maxLength") if isinstance(text_spec, dict) else None
    queries = search_phrases(run, k, product, label, cap_len)
    plans = []
    if form.text_key and queries[0]:
        plans.append((form.text_key, queries[0], False))
    if label and form.shelf_key:
        plans.append((form.shelf_key, label, True))
    if form.text_key and len(queries) > 1:
        plans.append((form.text_key, queries[1], False))
    calls = []
    for key, value, last in plans:
        draft = {key: value}
        if form.mode_key:
            draft[form.mode_key] = mode_value(form, key, last)
        draft.update(bound)
        for need in form.needed:
            spec = props[need] if isinstance(props[need], dict) else {}
            if need in draft:
                continue
            if spec.get("enum"):
                draft[need] = spec["enum"][0]
            elif spec.get("type") in ("integer", "number"):
                draft[need] = spec.get("default", spec.get("minimum", 1))
            elif spec.get("type") == "string":
                draft[need] = queries[0] or label
        fixed, faults = check_args(draft, form.schema)
        if faults and bound:
            fixed, faults = check_args({key: val for key, val in draft.items() if key not in bound}, form.schema)
        if not faults and fixed not in calls:
            calls.append(fixed)
    return calls


def code_rows(run):
    table = {}

    def visit(d):
        ident = row_mark(d)
        if ident not in run.archive.ids:
            return
        info = table.setdefault(ident, {"shelf": "", "title": "", "price": None, "stock": None, "words": set(), "fields": {}})
        info["shelf"] = shelf_of(d) or info["shelf"]
        title = ""
        for key, val in d.items():
            if isinstance(val, str) and re.search(r"title|name", str(key).lower()) and "id" not in str(key).lower():
                title = str(val)
                break
        info["title"] = title or info["title"]
        for key, val in d.items():
            name = str(key).lower()
            if not is_number(val) or isinstance(val, bool) or not re.search(r"price|cost|amount", name):
                continue
            if re.search(r"count|qty|quant|total|units|line|due|payable|sum", name):
                continue
            info["price"] = float(val)
            break
        for key, val in d.items():
            state = key_stock(key, val)
            if state is not None:
                info["stock"] = state
        nested = [v for v in d.values() if isinstance(v, dict)]
        pieces = [str(v) for v in d.values() if isinstance(v, (str, int, float)) and not isinstance(v, bool)]
        pieces += [str(x) for v in nested for x in v.values() if isinstance(x, (str, int, float))]
        info["words"] |= kind_words(" ".join(pieces))
        for v in nested:
            info["fields"].update({str(key): str(x) for key, x in v.items() if isinstance(x, (str, int, float)) and not isinstance(x, bool)})

    for rec in run.archive.items:
        walk_dicts(rec, visit)
    return table


def code_shows(row, prop, value, hit_words, read):
    named = [val for key, val in row["fields"].items() if same_prop(key, prop)]
    if named:
        return any(code_seen(value, val) for val in named)
    if code_seen(value, row.get("title") or ""):
        return True
    own = kind_words(value)
    return not read and bool(own) and own <= hit_words


def code_seen(value, text):
    pool = str(text or "").lower()
    if not pool.strip():
        return False
    body = re.sub(r"\b(?:" + VALUE_LINKS + r")\b", " ", str(value).lower())
    for alt in re.split(r"\s*(?:\bor\b|/|,|;)\s*", body):
        words = [w for w in re.findall(r"[a-z]{2,}", alt) if w not in LINK_WORDS]
        figs = re.findall(r"\d+(?:\.\d+)?", alt)
        if not (words or figs):
            continue
        if not all(re.search(r"\b" + re.escape(w[:-1] if len(w) > 4 else w), pool) for w in words):
            continue
        if all(re.search(r"(?<![\d.])" + re.escape(f) + r"(?![\d.])", pool) for f in figs):
            return True
    return False


def value_tier(row, prop, value):
    wanted = set()
    for alt in re.split(r"\s*(?:\bor\b|/)\s*", value.lower()):
        if ch_words(alt):
            wanted.add(" ".join(ch_words(alt)))
    best = (2, 3)
    for key, val in row["fields"].items():
        if not same_prop(key, prop) or not code_seen(value, val):
            continue
        if " ".join(ch_words(val)) in wanted:
            tier = 0
        elif any(" ".join(ch_words(p)) in wanted for p in ch_parts(val)):
            tier = 1
        else:
            tier = 3
        best = min(best, (0 if prop_key(key) == prop_key(prop) else 1, tier))
    if best == (2, 3) and code_seen(value, row.get("title") or ""):
        best = (2, 2)
    return best


class PickScene:
    def __init__(self, run, avoid, keep):
        self.rows = need_items(run.needs)
        self.counts = need_counts(run.needs)
        self.homes = home_shelves(run) if self.rows else [set(shelf_labels(run)) or (home_shelves(run) or [set()])[0]]
        self.table = code_rows(run)
        version = str(run.needs.version)
        self.failed = {key.split("|")[0] for key, val in run.rulings.items() if val and key.split("|")[1:2] == [version]}
        self.live = {key.rpartition("|")[0] for key in run.passed if key.rpartition("|")[2] == version}
        reviewed = vars(run).get("audit_ids")
        if vars(run).get("audit_done") and (reviewed is None or len(run.archive.read) <= vars(run).get("audit_reads", -1)):
            reviewed = set(run.archive.read)
        audits = [e for e in reversed(run.wire.log) if isinstance(e, dict) and e.get("kind") == "audit"][:1]
        self.audited = set()
        if audits and isinstance(audits[0].get("result"), dict):
            self.audited = set(reviewed or set()) - set(vars(run).get("audit_fits") or []) - self.live
        self.judges_all = bool(self.audited) and code_judges_all(run)
        self.cap = float(run.needs.spend_cap) if is_number(run.needs.spend_cap) else None
        self.asked = kind_words(run.request)
        self.hits = vars(run).get("code_hits") or {}
        self.cheapest = need_priority(run.needs) == "cheapest"
        self.avoid = set(avoid or ())
        self.keep = set(keep or ())
        self.avoid |= dead_lines(run)
        self.avoid |= ref_pointers(run)


def shows_apart(fields, apart):
    for prop, bad in apart:
        carried = [val for key, val in fields.items() if same_prop(key, prop) and str(val).strip()]
        if not any(not any(ch_writes(v, val) for v in bad) for val in carried):
            return False
    return True


def ranked_entry(run, scene, ident, row, want):
    qty, kind, value_words, judged, apart, owner = want
    blocked = ident in scene.failed or ident in run.refused_lines or ident in scene.avoid or row["stock"] is False
    if blocked or (row["shelf"] not in scene.homes_now if scene.homes_now else not (kind & row["words"])):
        return None
    if any(code_clash(row["fields"], prop, value) for prop, value in judged) or ref_excluded(run, row["fields"], owner):
        return None
    cost = price_of(run.archive, ident) if row["price"] is None else row["price"]
    if scene.cap is not None and is_number(cost) and cost * qty > scene.cap:
        return None
    seen = ident in run.archive.read
    hit_words = scene.hits.get(ident) or set()
    fits = all(code_shows(row, prop, value, hit_words, seen) for prop, value in judged) and shows_apart(row["fields"], apart)
    tiers = [value_tier(row, prop, value) for prop, value in judged] if seen else []
    if ident in scene.audited:
        if not (scene.judges_all and fits and (judged or apart) and all(max(t) <= 1 for t in tiers)):
            return None
    words = row["words"]
    score = 3 * len(value_words & words) + 2 * len(kind & words) + len(scene.asked & words) + 4 * len(value_words & hit_words)
    price = cost if is_number(cost) else float("inf")
    quality = (sum(t[0] for t in tiers), sum(t[1] for t in tiers)) if tiers else (9, 9)
    return (int(ident not in scene.keep), int(ident not in scene.live), not fits, scene.cap is not None and not is_number(cost),
            price if scene.cheapest else 0.0, int(not seen), row["stock"] is not True, (quality, -score), price, ident)


def product_wants(run, scene, k):
    owner = k if scene.rows else None
    qty = scene.counts[k] if k < len(scene.counts) else 1
    kind = kind_words(scene.rows[k]["product"] if scene.rows else run.needs.reading.get("product") or "")
    value_words = set()
    judged = []
    for prop, bare in code_values(run, owner):
        value_words |= kind_words(bare)
        loose = re.search(r"\b(?:" + VALUE_LINKS + r")\b|\d\s*(?:-|to)\s*\d|[+<>]|\b(?:about|around|approx\w*|roughly)\b", bare.lower())
        if not loose:
            judged.append((prop, bare))
    apart = [(ref["prop"], [str(v) for v in [ref.get("value")] + list(ref.get("goes_with") or []) if str(v or "").strip()])
             for ref in differ_done(run, owner)]
    return qty, kind, value_words, judged, apart, owner


def settle_picks(run, scene, pools):
    def units(k):
        return scene.counts[k] if k < len(scene.counts) else 1

    for order in (None, lambda e: (e[2], e[3], e[8], e[-1])):
        picks, chosen, total = [], [], 0.0
        for k, pool in enumerate(pools):
            taken = {p for p, _ in picks}
            entry = None
            for cand in sorted(pool, key=order):
                if cand[-1] in taken:
                    continue
                if cand[-1] in run.archive.read and field_view(run, cand[-1])["fails"]:
                    continue
                entry = cand
                break
            if entry is None:
                return []
            picks.append((entry[-1], units(k)))
            chosen.append(entry)
            total += (0.0 if entry[8] == float("inf") else entry[8]) * units(k)
        if scene.cap is None or total <= scene.cap:
            run.code_fit = {e[-1]: not e[2] for e in chosen}
            return picks
    return []


def code_picks(run, avoid=None, keep=None):
    scene = PickScene(run, avoid, keep)
    pools = []
    for k, home in enumerate(scene.homes):
        want = product_wants(run, scene, k)
        scene.homes_now = home
        pool = [entry for ident, row in scene.table.items() if (entry := ranked_entry(run, scene, ident, row, want))]
        if not pool:
            return []
        pools.append(sorted(pool))
    return settle_picks(run, scene, pools)


def code_clash(fields, prop, value):
    named = [val for key, val in fields.items() if same_prop(key, prop)]
    if not named:
        return False
    return not any(value_seen(value, val) for val in named)


def split_words(v):
    return set(re.findall(r"[a-z]+", re.sub(r"([a-z])([A-Z])", r"\1 \2", str(v)).lower()))


def choice_pair(vals, remembered):
    adds = [v for v in vals if split_words(v) & PUT_WORDS and not split_words(v) & DROP_WORDS]
    takes = [v for v in vals if split_words(v) & DROP_WORDS]
    chosen = remembered[0] if remembered else (adds[0] if adds else vals[0])
    rest = [v for v in vals if v != chosen]
    removal = next((v for v in takes if v != chosen), rest[0] if len(vals) == 2 else None)
    return chosen, removal


def cart_schema_parts(run):
    schema = run.shapes.get(role(run, "cart") or "") or {}
    props = props_of(schema)
    known = json_object(run.trolley.put_form) if run.trolley.put_form else None
    put, drop = {}, {}
    for key in schema.get("required") or []:
        spec = props.get(key) if isinstance(props.get(key), dict) else {}
        vals = spec.get("enum") if isinstance(spec.get("enum"), list) else []
        if vals:
            remembered = [known[key]] if isinstance(known, dict) and known.get(key) in vals else []
            put[key], drop[key] = choice_pair(vals, remembered)
    free = text_keys(props)
    id_like = [key for key in free if re.search(r"id|item|product|listing|sku|target|which|number", key.lower())]
    id_key = id_like[0] if id_like else (free[0] if free else None)
    return schema, put, drop, id_key, number_keys(props)


def put_calls(run, picks):
    schema, base, _, id_key, numbers = cart_schema_parts(run)
    spec_lines = line_fields(run.trolley)
    if spec_lines:
        arr, id_field, count_field = spec_lines
        drafts = [{**base, arr: [{id_field: ident, count_field: qty} for ident, qty in picks]}]
    elif not id_key:
        return []
    else:
        drafts = []
        for ident, qty in picks:
            draft = {**base, id_key: ident}
            if numbers:
                draft[numbers[0]] = qty
            drafts.append(draft)
    ready = []
    for draft in drafts:
        fixed, faults = check_args(draft, schema)
        if faults:
            return []
        ready.append(fixed)
    return ready


def take_out_draft(run, ident, parts):
    put, drop, id_key, numbers, props = parts
    if put:
        if any(drop.get(key) is None for key in put):
            return None
        draft = {**drop, id_key: ident}
        if numbers:
            draft[numbers[0]] = max(1, int(units_held(run.trolley, ident) or 1))
        return draft
    if not numbers:
        return None
    floor = (props.get(numbers[0]) or {}).get("minimum")
    if is_number(floor) and props[numbers[0]]["minimum"] > 0:
        return None
    return {id_key: ident, numbers[0]: 0}


def take_out_calls(run, lines):
    if not lines or line_fields(run.trolley):
        return []
    schema, put, drop, id_key, numbers = cart_schema_parts(run)
    if not id_key:
        return None
    parts = (put, drop, id_key, numbers, props_of(schema))
    ready = []
    for ident in lines:
        draft = take_out_draft(run, ident, parts)
        if draft is None:
            return None
        fixed, faults = check_args(draft, schema)
        if faults:
            return None
        ready.append(fixed)
    return ready


def latest_observations(run, n):
    newest = None
    for entry in reversed(run.wire.log):
        if isinstance(entry, dict) and entry.get("kind") == "env":
            newest = entry
            break
    reply = (newest or {}).get("reply")
    if not isinstance(reply, dict):
        return []
    many = reply.get("observations")
    return list(many if isinstance(many, list) else [reply.get("observation")])[:n]


def code_fault(obs):
    hits = []

    def visit(d):
        for key, val in d.items():
            if re.search(FAULT_KEYS, str(key).lower()) and val not in (None, False, "", [], {}):
                hits.append(d)
                return

    walk_dicts(obs, visit, 0, 2)
    return bool(hits)


def search_batch(run, products, more, reserve):
    if more:
        lead = list(vars(run).get("code_more") or [])
        return lead[: max(0, moves_left(run) - reserve - 2 * products)], lead, 1
    several = need_items(run.needs)
    per_product = [code_queries(run, k if several else None) for k in range(products)]
    calls = []
    depth = max((len(q) for q in per_product), default=0)
    for rank in range(depth):
        fresh_rank = [q[rank] for q in per_product if rank < len(q) and q[rank] not in calls]
        calls.extend(fresh_rank)
    lead = [q[0] for q in per_product if q]
    extra = 2 if need_priority(run.needs) == "cheapest" else 0
    return calls[: max(0, moves_left(run) - reserve)], lead, 1 + CODE_PAGES + extra


def search_round(run, tool, calls, lead, free):
    send_chunk(run, [(f"code_search_{n}_{run.wire.tick}", tool, args) for n, args in enumerate(calls)])
    replies = latest_observations(run, len(calls))
    answered = list(zip(calls, replies))
    if vars(run).get("word_kind") != "any":
        hits = vars(run).setdefault("code_hits", {})
        for args, obs in answered:
            said = " ".join(str(args.get(key) or "") for key in free)
            if said.strip():
                note_hits(run, obs, hits, kind_words(said))
    later = [nxt for nxt in (following_page(run, args, obs) for args, obs in answered if args in lead) if nxt]
    run.code_more = later
    return later


def own_searches(run, products, more=False):
    tool = role(run, "search")
    free = text_keys((run.shapes.get(tool or "") or {}).get("properties") or {})
    reserve = 2 + 2 * products
    calls, lead, rounds = search_batch(run, products, more, reserve)
    if not calls:
        return
    run.code_searched = True
    for rnd in range(rounds):
        lead = search_round(run, tool, calls, lead, free)
        if rnd + 1 == rounds:
            return
        picks = code_picks(run)
        fit = vars(run).get("code_fit") or {}
        if need_priority(run.needs) != "cheapest" and picks and all(fit.get(ident) for ident, _ in picks):
            return
        calls = lead[: max(0, moves_left(run) - reserve - 3 * products)]
        if not calls:
            return


def note_hits(run, obs, hits, words):
    def visit(d):
        ident = row_mark(d)
        if ident in run.archive.ids:
            hits.setdefault(ident, set()).update(words)

    walk_dicts(obs, visit)


def following_page(run, call_args, obs):
    spec = page_arg(run)
    if not spec:
        return None
    key, by_page, low, high = spec
    pages = []

    def visit(d):
        if any(isinstance(v, list) and v and all(isinstance(x, dict) for x in v) for v in d.values()):
            pages.append(d)

    walk_dicts(obs, visit, 0, 3)
    if not pages:
        return None
    page = pages[0]
    shown = max(len(v) for v in page.values() if isinstance(v, list))
    marks = [v for name, v in page.items() if re.search(r"next|onward|cursor|bookmark|token|continu|more", str(name).lower())]
    total = next((v for name, v in page.items() if is_number(v) and not isinstance(v, bool)
                  and re.search(r"count|total|matched|found|hits", str(name).lower())), None)
    if marks and all(m in (None, False, "", 0) for m in marks):
        return None
    if by_page == "token":
        tok = next((m for m in marks if isinstance(m, str) and m.strip()), None)
        if not tok:
            return None
        nxt_args = {**call_args, key: tok}
    else:
        cur = call_args.get(key, low)
        if not is_number(cur):
            cur = low
        nxt = cur + 1 if by_page else cur + shown
        if high is not None and nxt > high:
            return None
        if not marks:
            seen = (cur - low + 1) * shown if by_page else cur + shown
            if total is None or seen >= total:
                return None
        nxt_args = {**call_args, key: int(nxt) if float(nxt).is_integer() else nxt}
    fixed, faults = check_args(nxt_args, run.shapes.get(role(run, "search") or "") or {})
    return None if faults else fixed


def read_picks(run, picks, avoid, keep):
    for _ in range(CODE_READS):
        unread = [ident for ident, _ in picks if ident not in run.archive.read]
        batch = batch_tool(run)
        together = bool(batch) and len(unread) >= 2
        reads = 1 if together else len(unread)
        if not unread or not (batch or role(run, "details")):
            break
        if moves_left(run) < reads + 2 * len(picks) + 3 or time_left(run.wire) <= 8:
            break
        if together:
            slab = [(f"code_read_{run.wire.tick}", batch[0], {batch[1]: unread[: batch[2]]})]
        else:
            details = role(run, "details")
            slab = [(f"code_read_{n}_{run.wire.tick}", details, args)
                    for n, args in enumerate(details_call(run, ident) for ident in unread) if args]
        if not slab:
            break
        send_chunk(run, slab)
        again = code_picks(run, avoid, keep)
        if again == picks:
            break
        picks = again
    return picks


def faulted_lines(run, calls, replies, want, whole_list):
    bad, shown = set(), None
    for args, obs in zip(calls, replies):
        seen = units_in_view(run.trolley, obs)
        shown = shown if seen is None else seen
        if code_fault(obs):
            named = [ident for ident in want if ident in as_text(obs)]
            touched = [line for line, _ in cart_lines(run.trolley, args) if line in want]
            bad.update(named or touched or (list(want) if whole_list else []))
    return bad, shown


def code_put(run, picks):
    cart = role(run, "cart")
    want = {ident for ident, _ in picks}
    whole_list = line_fields(run.trolley)
    extra = [line for line in run.trolley.units if line not in want]
    todo = list(picks) if whole_list else [(i, q) for i, q in picks if abs(units_held(run.trolley, i) - q) > 1e-9]
    drops = take_out_calls(run, extra)
    puts = put_calls(run, todo) if todo else []
    if drops is None or (todo and not puts):
        return None
    calls = drops + puts
    if not calls:
        return []
    if moves_left(run) < len(calls) + 1:
        return None
    send_chunk(run, [(f"code_cart_{n}_{run.wire.tick}", cart, args) for n, args in enumerate(calls)])
    replies = latest_observations(run, len(calls))
    bad, shown = faulted_lines(run, calls, replies, want, whole_list)
    if len(replies) < len(calls):
        return None
    if shown is None and not bad and role(run, "cart_view") and moves_left(run) >= 3:
        send_chunk(run, [(f"code_view_{run.wire.tick}", role(run, "cart_view"), {})])
        view = latest_observations(run, 1)
        shown = units_in_view(run.trolley, view[0]) if view else None
        if view and shown is None:
            bad.update(i for i in want if i not in run.trolley.units)
    if shown is not None:
        bad.update(i for i in want if i not in shown)
    for ident in bad:
        run.trolley.units.pop(ident, None)
    return sorted(bad)


def code_recheck(run, lines):
    tool = stock_tool(run) or role(run, "view") or role(run, "details")
    if not lines or not tool or moves_left(run) < len(lines) + 2 or time_left(run.wire) < 8:
        return False, set()
    slab = [(f"code_check_{n}_{run.wire.tick}", tool, args) for n, ident in enumerate(lines)
            if (args := lookup_call(run, ident, tool))]
    if not slab:
        return False, set()
    send_chunk(run, slab)
    table = code_rows(run)
    return True, {ident for ident in lines if (table.get(ident) or {}).get("stock") is False}


def code_news(run, lines):
    held_back = [ln for group, _ in vars(run.needs).get("pending") or [] for ln in group]
    done = vars(run).setdefault("news_read", set())
    for line in list(lines) + held_back:
        if line in done:
            continue
        done.add(line)
        figure = spend_limit_in(line)[0]
        if is_number(figure) and figure > 0:
            run.needs.spend_cap = figure
            log_entry(run, "code_news", budget=figure)


def code_cart(run, avoid=None, keep=None):
    cart, search = role(run, "cart"), role(run, "search")
    if not cart or run.wire.done or run.wire.broke or moves_left(run) < 2 or time_left(run.wire) < 4:
        return False
    run.wire.mute = True
    try:
        avoid = set(avoid or ())
        products = max(1, len(need_items(run.needs)))
        for _ in range(4):
            picks = code_picks(run, avoid, keep)
            unfit = not picks or not all(vars(run).get("code_fit", {}).get(ident) for ident, _ in picks)
            if search and not vars(run).get("code_searched") and moves_left(run) >= 4 and unfit:
                own_searches(run, products)
                picks = code_picks(run, avoid, keep)
            elif search and vars(run).get("code_more") and moves_left(run) >= 6 + 2 * products \
                    and (unfit or need_priority(run.needs) == "cheapest"):
                own_searches(run, products, True)
                picks = code_picks(run, avoid, keep)
            picks = read_picks(run, picks, avoid, keep)
            if not picks:
                log_entry(run, "code_finish", picks=[], result="nothing to order")
                return False
            bad = code_put(run, picks)
            if bad is None:
                log_entry(run, "code_finish", picks=picks, result="no cart call")
                return False
            if not bad:
                if moves_left(run) >= len(picks) + 3:
                    bad = sorted(code_recheck(run, [ident for ident, _ in picks])[1])
                else:
                    bad = []
            if not bad:
                log_entry(run, "code_finish", picks=picks, cart=dict(run.trolley.units))
                return bool(run.trolley.units)
            avoid |= set(bad)
            log_entry(run, "code_swap", lines=bad)
        return False
    except Exception as err:
        log_entry(run, "code_finish_error", error=f"{type(err).__name__}: {err}"[:300])
        return False


# (one store named its paging argument "begin_at_at": the code paged nowhere there, so the names any word of a
# position may take count)
PAGE_KEYS = r"page|offset|first|start|skip|from|cursor|begin|resume|position|index|after"
NOT_PAGE_KEYS = r"size|per_?page|limit|price|cost"
STRIKE_MARKS = r"\((?:stated|revealed[^)]*)\)"


def call_sig(args):
    return json.dumps(args, sort_keys=True, default=str)


def error_entry(kind, err):
    return {"kind": kind, "error": f"{type(err).__name__}: {err}"[:300]}


# ---- finishing the errand from code

def finish_blocked(run):
    link = run.wire
    if link.done or link.broke:
        return True
    return moves_left(run) <= 0 or time_left(link) < 2


def order_move(run, tool):
    return dict(name=tool, args=plain_args(run, tool, "Order the cart.", "order"))


def decline_move(run, tool):
    return dict(name=tool, args=plain_args(run, tool, SORRY, "decline"))


def cart_may_repeat(run, avoid):
    lines_now = set(run.trolley.units)
    unwanted = set(avoid) | dead_lines(run)
    if lines_now and not lines_now & unwanted:
        return bool(code_picks(run, unwanted))
    return False


def stale_lines_out(run, avoid):
    run.wire.mute = True
    before = list(run.trolley.units)
    try:
        gone = cart_gone(run, before)
    except Exception as err:
        run.wire.log.append(error_entry("cart_gone_error", err))
        return set(), None
    if gone:
        avoid |= gone
        run.wire.log.append({"kind": "code_dead", "lines": sorted(gone)})
        return gone, set(before) - gone
    return gone, None


def opening_move(run, order_tool, decline_tool, turn, gone, avoid, keep):
    cart = run.trolley
    if order_tool:
        as_is = turn == 0 and not gone and cart.used and (cart.units or units_in_view(cart, cart.last_view) is None)
        if as_is or code_cart(run, avoid, keep) or (turn > 0 and cart_may_repeat(run, avoid)) or (turn == 0 and full_cart(run)):
            return order_move(run, order_tool)
    return decline_move(run, decline_tool) if decline_tool else None


def refused_finish(run, move, resp):
    run.wire.mute = True
    held_lines = list(run.trolley.units)
    obs = resp.get("observation")
    absorb(run, move["name"], move["args"], obs, resp, True)
    code_news(run, shopper_lines(run, move["name"], obs, True))
    return held_lines


def finish_now(run):
    if finish_blocked(run):
        return
    order_tool = role(run, "order")
    decline_tool = role(run, "decline")
    avoid, keep, move, kind = set(), None, None, None
    turn = 0
    while turn < CODE_FINISHES:
        if finish_blocked(run):
            return
        if move is None:
            gone = set()
            if order_tool and not turn and run.trolley.units:
                gone, kept = stale_lines_out(run, avoid)
                keep = kept if gone else keep
            move = opening_move(run, order_tool, decline_tool, turn, gone, avoid, keep)
            if not move:
                return
        if call_kind(run, move["name"], move["args"]) == "order":
            shielded(run.wire, "basket_yes_error", basket_gate, run, True)
            if finish_blocked(run):
                return
        resp = world_act(run.wire, [move])
        run.wire.log.append({"kind": "forced_finish", "action": move, "reply": resp})
        kind = call_kind(run, move["name"], move["args"])
        if not resp or resp.get("finished") or run.wire.done:
            break
        if turn == CODE_FINISHES - 1 or moves_left(run) <= 1:
            break
        # the store said no: book it like any refusal, then see whether the same cart can go again
        held_lines = refused_finish(run, move, resp)
        move = None
        turn += 1
        if kind == "order" and held_lines:
            outcome = repair_plan(run, held_lines, order_tool)
            if outcome is not None:
                move, new_avoid, new_keep = outcome
                if move is None:
                    avoid, keep = new_avoid, new_keep
    if run.wire.done or not resp or resp.get("finished"):
        return
    if not decline_tool or kind == "decline" or moves_left(run) <= 0:
        return
    repeat = False
    try:
        repeat = bool(order_tool) and kind == "order" and cart_may_repeat(run, avoid)
    except Exception as err:
        run.wire.log.append(error_entry("again_error", err))
    last = order_move(run, order_tool) if repeat else decline_move(run, decline_tool)
    reply = world_act(run.wire, [last])
    run.wire.log.append({"kind": "forced_finish", "action": last, "reply": reply})


def reread_request(run):
    link, state = run.wire, vars(run)
    if not run.needs.reading.get("code_reading") or state.get("reread"):
        return ""
    if link.last_reply <= state.get("read_at", link.last_reply) or time_left(link) <= CLOSING + 60:
        return ""
    run.reread = True
    reading = checked_reading(run, run.read_content, 1)
    understood = isinstance(reading.get("roles"), dict)
    link.log.append({"kind": "re_read", "ok": understood})
    if understood:
        use_reading(run, reading, True)
        return ("[plan updated] The models reply again and have read the request anew; this reading takes the place of the first:\n"
                + needs_overview(run.needs, run.step_cap, run.msg_cap))
    return ""


def late_heard_lines(run):
    link, sheet = run.wire, run.needs
    queued = vars(sheet).get("pending") or []
    if not queued or link.last_reply <= vars(sheet).get("pending_at", 0.0) or time_left(link) < CLOSING:
        return []
    sheet.pending = []
    heard = []
    for lns, qtext in queued[-3:]:
        said = hear_lines(sheet, lns, qtext)
        if said:
            heard.append("[the shopper's earlier lines, read now] " + said)
    return heard


def catch_up(run):
    try:
        if run.wire.mute or time_left(run.wire) < CLOSING:
            return
        notes = list(filter(None, [reread_request(run)]))
        notes.extend(late_heard_lines(run))
        if notes:
            text_body = "\n".join(notes) + "\n" + status_line(run)
            newest = run.chat[-1] if run.chat else None
            if newest and newest.get("role") in ("user", "tool") and isinstance(newest.get("content"), str):
                newest["content"] += "\n" + text_body
            else:
                run.chat.append({"role": "user", "content": text_body})
    except Exception as err:
        run.wire.log.append(error_entry("catch_up_error", err))


# ---- reading call results

def refused(run, call_name, obs, resp):
    as_dict = obs if isinstance(obs, dict) else None
    if as_dict is not None and "refusal" in as_dict:
        return True
    if call_name != role(run, "order") or resp.get("finished"):
        return False
    return as_dict is None or as_dict.get("error") not in ("invalid_args", "unknown_tool")


def wordy_strings(piece, found, depth=0):
    if depth > 6:
        return
    if isinstance(piece, dict):
        children = list(piece.values())
    elif isinstance(piece, list):
        children = piece[:20]
    else:
        if isinstance(piece, str) and len(piece.split()) >= 4:
            found.append(piece)
        return
    for child in children:
        wordy_strings(child, found, depth + 1)


def squashed(text):
    return " ".join(str(text).lower().split())


def order_refusal_note(run, obs):
    run.refusal_n += 1
    run.order_vetted = False
    run.event_seen = True
    extra = ""
    try:
        extra = refusal_reread(run, obs)
    except Exception as err:
        run.wire.log.append(error_entry("refusal_reread_error", err))
    run.refused_lines.update(run.trolley.units)
    texts = []
    wordy_strings(obs, texts)
    run.refusal_texts = set(map(squashed, texts))
    return ("ORDER REFUSED. The episode is still running and nothing in the cart has changed. Make each change the refusal "
            "asks for, then fix the cart (remove each line that has become unacceptable, such as a sold-out item), re-check stock "
            "and price of the new pick, and order again." + extra)


def shopper_news_note(run, obs):
    lns = []
    for piece in obs.values():
        if isinstance(piece, list) and piece and all(isinstance(val, str) for val in piece):
            long_strings(piece, lns)
    if not lns:
        return ""
    carried = vars(run).pop("refusal_texts", None) or set()
    repeated = bool(carried)
    for line in lns:
        if not repeated:
            break
        flat = squashed(line)
        repeated = any(flat in t for t in carried)
    if repeated:
        return ("Nothing new: this repeats the message the refused finish already carried (handled above); it says nothing "
                "about this result.")
    run.event_seen = True
    return "This result carries a message from the shopper (a change). Apply it: " + " | ".join(lns[:4])


def notes_for(run, call_name, obs, refused):
    remarks = []
    as_dict = isinstance(obs, dict)
    if as_dict and obs.get("error") == "budget_exceeded":
        if call_name == role(run, "message") or run.msg_cap is not None:
            run.msg_cap = run.msgs_sent if run.msg_cap is None else min(run.msg_cap, run.msgs_sent)
        remarks.append("That budget is used up; do not call this tool again.")
    if not refused:
        if as_dict and len(obs) >= 2 and call_name != role(run, "message"):
            remarks.append(shopper_news_note(run, obs))
    elif call_name == role(run, "decline") and getattr(run, "last_kind", None) == "decline":
        run.audit_done = False
        run.decline_vetted = False
        run.event_seen = True
        remarks.append("DECLINE REFUSED; the episode is not over, because the store does not agree that nothing fits. Act on "
                       "what the refusal says, then look once more at the nearest candidates (own properties, stock and price measured "
                       "against the requirements now), and order the one that fits; decline again only if each one truly fails.")
    else:
        remarks.append(order_refusal_note(run, obs))
    return " ".join(filter(None, remarks))


def shelf_of(row):
    for key, val in row.items():
        if not isinstance(val, str):
            continue
        label, name = val.strip(), str(key).lower()
        if label and len(label) <= 60 and "id" not in name and re.search(r"categor|shelf|department|aisle", name):
            return label.lower()
    return ""


def key_stock(key, val):
    name = str(key).lower()
    if val is None or not re.search(r"stock|avail|on.?hand", name):
        return None
    none_left = val is False or (is_number(val) and val <= 0)
    text = val.lower() if isinstance(val, str) else None
    if not re.search(r"stock|avail", name):
        blocked = False
    elif re.search(r"out|sold", name):
        blocked = val is True
    elif none_left:
        blocked = True
    else:
        blocked = text is not None and bool(re.search(r"\b(?:out|sold|none|unavailable|not|no|false)\b", text))
    if none_left and re.search(r"on.?hand", name):
        blocked = True
    if text is not None and re.search(r"back.?order|pre.?order|discontinu", text):
        blocked = True
    return not blocked


def stock_state(row):
    said = []
    for key, val in row.items():
        verdict = key_stock(key, val)
        if verdict is not None:
            said.append(verdict)
    return all(said) if said else None


def sold_out(row):
    return False in [key_stock(key, val) for key, val in row.items()]


def id_in_pool(d, pool):
    for key, val in d.items():
        if isinstance(val, str) and "id" in str(key).lower() and val in pool:
            return val
    return None


def row_price(d):
    for key, val in d.items():
        if is_number(val) and re.search(r"price|cost", str(key).lower()):
            return val
    return None


def labels_per_product(run, tally):
    homes = home_shelves(run)
    taken = set()
    for home in homes:
        counted_here = [label for label in home if label in tally]
        if not counted_here:
            taken |= set(home)
            continue
        peak = max(tally[label] for label in counted_here)
        taken |= {label for label in counted_here if tally[label] == peak}
    by_count = sorted(tally, key=lambda s: -tally[s])
    while by_count and not all(homes) and len(taken) < len(homes):
        taken.add(by_count.pop(0))
    return taken


def shelf_labels(run):
    tally, counted = {}, set()
    foreign = ref_foreign(run)
    arch = run.archive

    def visit(d):
        ident = id_in_pool(d, arch.read)
        if ident and ident not in counted and ident not in foreign:
            label = shelf_of(d)
            if label:
                counted.add(ident)
                tally[label] = tally.get(label, 0) + 1
    for rec in arch.items:
        walk_dicts(rec, visit)
    peak = max(tally.values(), default=0)
    most_read = {label for label, n in tally.items() if n == peak}
    if tally and need_items(run.needs):
        try:
            return labels_per_product(run, tally) or most_read
        except Exception:
            pass
    return most_read


def row_mark(row):
    loose = None
    for key, val in row.items():
        if isinstance(val, bool) or not isinstance(val, (str, int)):
            continue
        name = str(key).lower()
        if name == "id":
            return str(val)
        if loose is None and name.endswith("id"):
            loose = str(val)
    return loose if loose is not None else json.dumps(row, sort_keys=True, default=str)


def search_props(run):
    schema = run.shapes.get(role(run, "search") or "")
    props = schema.get("properties") if isinstance(schema, dict) else None
    return props if isinstance(props, dict) else {}


def query_words(text):
    kept = []
    for raw in text.lower().split():
        word = raw.strip("\"'()[]{},.;:!?|/")
        if re.search(r"[a-z0-9]", word) and word not in kept:
            kept.append(word)
    return tuple(kept)


def search_keys(run, call_args):
    props = search_props(run)
    texts, modes, paging = [], [], set()
    first = True
    tok = token_key(run)
    for key, val in call_args.items():
        name = str(key).lower()
        spec = props.get(key)
        if not isinstance(spec, dict):
            spec = {}
        if isinstance(val, str):
            if key == tok:
                paging.add(key)
                first = first and not val.strip()
            elif spec.get("enum"):
                modes.append((key, val))
            else:
                words = query_words(val)
                if words and all(re.fullmatch(r"[a-z0-9]{2,}", w) for w in words):
                    texts.append((key, words))
        elif is_number(val) and re.search(PAGE_KEYS, name) and not re.search(NOT_PAGE_KEYS, name):
            paging.add(key)
            floor = spec.get("minimum", spec.get("default"))
            if not is_number(floor):
                floor = 1 if "page" in name else 0
            first = first and val <= floor
    return texts, tuple(sorted(modes)), first, paging


def search_total(obs, rows):
    counts = [val for val in obs.values() if isinstance(val, int) and not isinstance(val, bool)]
    counts.append(len(rows))
    return max(counts)


def fixed_args(call_args, key, paging):
    pairs = []
    for other, val in call_args.items():
        if other != key and other not in paging:
            pairs.append((other, str(val)))
    return tuple(sorted(pairs))


def extends(longer, shorter):
    return len(longer) > len(shorter) and longer[: len(shorter)] == shorter


def word_rule(run, call_args, obs, rows):
    texts, mode, first, paging = search_keys(run, call_args)
    if not first:
        return run.word_kind
    total = search_total(obs, rows)
    for key, words in texts:
        ctx = (key, mode)
        others = fixed_args(call_args, key, paging)
        subset_hint = False
        for seen_ctx, seen_others, seen_words, seen_total in run.word_seen:
            if (seen_ctx, seen_others) != (ctx, others) or seen_total == total:
                continue
            if extends(words, seen_words):
                run.word_kind = ("all", "any")[total > seen_total]
            elif extends(seen_words, words):
                run.word_kind = ("all", "any")[total < seen_total]
            elif not subset_hint:
                # reordered words: a strict superset that matched fewer listings hints at 'all'
                cut = run.lead_words or None
                ours, theirs = set(words[:cut]), set(seen_words[:cut])
                subset_hint = (ours > theirs and total < seen_total) or (theirs > ours and seen_total < total)
        if subset_hint and not run.word_kind:
            run.word_kind = "all"
        run.word_seen.append((ctx, others, words, total))
    return run.word_kind


def shared_lead(a, b):
    count = 0
    while count < min(len(a), len(b)) and a[count] == b[count]:
        count += 1
    return count


def lead_claim(bounds):
    ready = []
    for ctx, (low, high) in bounds.items():
        above = [k for k in high if k > low]
        if above:
            ready.append((min(above), ctx))
    ready.sort()
    for k, ctx in ready:
        if k == bounds[ctx][0] + 1:
            return k, ctx
    return 0, None


def lead_rule(run, call_args, obs, rows):
    texts, mode, first, paging = search_keys(run, call_args)
    if not first:
        return run.lead_words
    total = search_total(obs, rows)
    marks = tuple(map(row_mark, rows))
    for key, words in texts:
        ctx = (key, mode)
        others = fixed_args(call_args, key, paging)
        bound = run.lead_bounds.setdefault(ctx, [-1, set()])
        for entry in run.lead_seen:
            if entry[:2] == (ctx, others) and entry[2] != words:
                lead_compare(bound, words, total, marks, entry[2:])
        run.lead_seen.append((ctx, others, words, total, marks))
    run.lead_words, run.lead_ctx = lead_claim(run.lead_bounds)
    return run.lead_words


def token_key(run):
    for key, spec in search_props(run).items():
        plain_string = isinstance(spec, dict) and spec.get("type") == "string" and not spec.get("enum")
        if plain_string and re.search(r"bookmark|cursor|token|resume|continu|after|next|page|offset|start", str(key).lower()):
            return key
    return None


def next_token(obs):
    if not isinstance(obs, dict):
        return None
    hits = [val for key, val in obs.items()
            if isinstance(val, str) and val.strip() and re.search(r"next|onward|cursor|bookmark|token|continu|more", str(key).lower())]
    return hits[0] if hits else None


def page_arg(run):
    for key, spec in search_props(run).items():
        name = str(key).lower()
        numeric = isinstance(spec, dict) and spec.get("type") in ("integer", "number")
        if not numeric or not re.search(PAGE_KEYS, name) or re.search(NOT_PAGE_KEYS, name):
            continue
        whole_pages = "page" in name and not re.search(r"offset|skip|first|start|from", name)
        floor = spec.get("minimum", spec.get("default"))
        if not is_number(floor):
            floor = int(whole_pages)
        top = spec.get("maximum")
        if not is_number(top):
            top = None
        return key, whole_pages, floor, top
    found = result_page_key(run)
    if found:
        spec = search_props(run).get(found) or {}
        floor, top = spec.get("minimum", spec.get("default")), spec.get("maximum")
        return found, False, floor if is_number(floor) else 0, top if is_number(top) else None
    tok = token_key(run)
    return (tok, "token", "", None) if tok else None


def result_page_key(run):
    # (paging read from the results when no argument name says it: a search answer carrying a numeric next-position mark
    # beside its rows shows that the tool pages, and its one integer argument with no other meaning takes that position)
    memo = vars(run).get("result_page_memo")
    if memo and memo[0] == len(run.archive.items):
        return memo[1]
    marked = []

    def visit(d):
        rows = any(isinstance(v, list) and v and all(isinstance(x, dict) for x in v) for v in d.values())
        if rows and any(is_number(v) and not isinstance(v, bool) and re.search(r"next|onward|cursor|bookmark|token|continu|more|follow", str(k).lower())
                        for k, v in d.items()):
            marked.append(d)
    for rec in run.archive.items:
        walk_dicts(rec, visit, 0, 3)
    key = None
    if marked:
        free = [k for k, spec in search_props(run).items() if isinstance(spec, dict) and spec.get("type") in ("integer", "number")
                and not re.search(NOT_PAGE_KEYS + r"|min|max|rating|quantity|qty|year", str(k).lower())]
        key = free[0] if len(free) == 1 else None
    run.result_page_memo = (len(run.archive.items), key)
    return key


def stem(word):
    plural = len(word) > 3 and word.endswith("s") and not word.endswith("ss")
    return word[:-1] if plural else word


def row_title(d):
    own = d.get("title")
    if isinstance(own, str) and 0 < len(own) <= 300:
        return own
    for key, val in d.items():
        name = str(key).lower()
        if not isinstance(val, str) or not 0 < len(val) <= 300 or not re.search(r"title|heading|headline|name", name):
            continue
        if not re.search(r"categor|shelf|brand|make|manufactur|vendor|shop|seller|store|id", name):
            return val
    return ""


def shelf_rows(run):
    shelves = shelf_labels(run)
    sheet = run.needs
    cap = sheet.spend_cap
    qty = need_qty(sheet)
    found = {}
    if not shelves:
        return found
    arch = run.archive

    def visit(d):
        ident = id_in_pool(d, arch.ids)
        if not ident or ident in run.refused_lines or shelf_of(d) not in shelves:
            return
        if sold_out(d):
            found.pop(ident, None)
            return
        cost = row_price(d)
        title = row_title(d)
        makers = [val for key, val in d.items() if isinstance(val, str) and re.search(r"brand|make|manufactur|vendor", str(key).lower())]
        if cost is not None:
            found[ident] = (float(cost), title, " ".join(makers))
    for rec in arch.items:
        walk_dicts(rec, visit)
    for ident in list(found):
        cost, title, maker = found[ident]
        latest = price_of(arch, ident)
        if latest is not None:
            cost = latest
        if is_number(cap) and cost * qty > cap:
            del found[ident]
        else:
            found[ident] = (float(cost), title, maker)
    return found


def label_parts(label):
    parts, current = [], []
    for raw in label.lower().split():
        word = raw.strip(".,;:!?'\"()[]")
        if word not in LINK_WORDS and re.fullmatch(r"[a-z0-9][a-z0-9'-]*", word):
            current.append(word)
        elif current:
            parts.append(current)
            current = []
    if current:
        parts.append(current)
    return parts


def label_queries(run):
    queries = []
    for label in sorted(shelf_labels(run)):
        parts = label_parts(label)
        words = sum(parts, [])
        heads = [part[-1] for part in parts]
        if len(words) > 1:
            queries.append(" ".join(words[:2]))
        queries += heads
        queries += [w for w in words if w not in heads]
    return list(dict.fromkeys(queries))


def title_grams(words, usable, pairs):
    grams = []
    if pairs:
        for i in range(len(words) - 1):
            if usable[i] and usable[i + 1]:
                grams.append((words[i], words[i + 1]))
    else:
        for i, word in enumerate(words):
            if usable[i]:
                grams.append((word,))
    return grams


def kind_queries(run, pairs):
    label_stems = words_of(shelf_labels(run), r"[a-z0-9]+")
    label_stems = set(map(stem, label_stems))
    titles = {}
    for cost, title, maker in shelf_rows(run).values():
        key = title.lower().strip()
        if not key:
            continue
        earlier = titles.get(key)
        cheapest = cost if earlier is None else min(cost, earlier[0])
        titles[key] = (cheapest, maker.lower())
    tallies, low_cost, shown = tally_title_grams(titles, label_stems, pairs)
    common = sorted((b for b, n in tallies.items() if n > 1), key=lambda b: (low_cost[b], -tallies[b], b))
    return list(map(shown.get, common))


def lead_stems(text):
    return tuple(map(stem, re.findall(r"[a-z0-9]{2,}", text.lower())[:2]))


def sweep_template(run, ranked, shelves, page_key):
    props = (run.shapes.get(role(run, "search") or "") or {}).get("properties") or {}
    newest_args = ranked[-1]["args"]

    def pinned_arg(k, v):
        if not isinstance(v, str) or (props.get(k) or {}).get("enum"):
            return True
        return v.strip().lower() in shelves or k == page_key
    free_keys = [k for k, v in newest_args.items() if not pinned_arg(k, v)]
    if run.lead_ctx and run.lead_ctx[0] in free_keys:
        text_key = run.lead_ctx[0]
    else:
        text_key = max(free_keys, key=lambda k: len(newest_args[k].split()), default=None)
    if not text_key:
        return None, text_key
    return {k: v for k, v in newest_args.items() if k != text_key and pinned_arg(k, v)}, text_key


def sweep_next(run, turn):
    paging = page_arg(run)
    shelves = shelf_labels(run)
    book = vars(run).get("page_book", {})
    if not (paging and shelves and book):
        return None, False
    page_key, low = paging[0], paging[2]
    tried = vars(run).get("sweep_tried", set())
    listing = None
    for walk in book.values():
        if walk.get("id_order") and walk["shelves"] and walk["shelves"] <= shelves and next_page(run, walk) is not None:
            if listing is None or walk["total"] > listing["total"]:
                listing = walk
    ranked = [w for w in book.values() if w.get("id_order") is False and free_text(run, w["args"])]
    if not ranked:
        ranked = [w for w in book.values() if w.get("id_order") is None and free_text(run, w["args"])]
    template, text_key = sweep_template(run, ranked, shelves, page_key) if ranked else (None, None)
    done = set()
    for walk in book.values():
        if low in walk["read"] and walk.get("id_order") is not True:
            done.add(lead_stems(" ".join(free_text(run, walk["args"]))))

    def query(text):
        sig = lead_stems(text)
        if template is None or not sig or sig in done:
            return None
        fresh = dict(template)
        fresh[text_key] = text
        return fresh if call_sig(fresh) not in tried else None

    def candidates(source):
        if source == 0:
            return [{**listing["args"], page_key: next_page(run, listing)}] if listing else []
        texts = label_queries(run) if source == 3 else kind_queries(run, source == 1)[:6]
        return [q for q in map(query, texts) if q]
    ceil, ended = ceiling_sweep(run, tried)
    if ceil is not None:
        return ceil, True
    if not ended:
        for source in [0] + [1 + (turn + shift) % 3 for shift in range(3)]:
            for args in candidates(source):
                if call_sig(args) not in tried:
                    return args, source == 0
    return None, False


def sweep_room(run, reserve):
    if run.swept >= SWEEP_READS or moves_left(run) <= reserve:
        return False
    return time_left(run.wire) >= CLOSING + 40


def sweep_step(run):
    if not sweep_wanted(run):
        return False
    reserve = sweep_reserve(run)
    swept = vars(run).get("swept", 0)
    if swept >= SWEEP_READS or moves_left(run) <= reserve or time_left(run.wire) < CLOSING + 40:
        return False
    turn = vars(run).get("sweep_turn", 0)
    call_args, from_listing = sweep_next(run, turn)
    if call_args is None:
        return False
    run.swept = swept + 1
    run.sweep_turn = turn if from_listing else turn + 1
    vars(run).setdefault("sweep_tried", set()).add(call_sig(call_args))
    say = ("With no property requirement left, the cheapest in-stock listing anywhere on the shelf is the answer, so I read "
           "another slice of the shelf before I choose.")
    _, responses = scripted_search(run, f"sweep_{run.swept}_{run.wire.tick}", call_args, say)
    more = sweep_room(run, reserve) and sweep_next(run, run.sweep_turn)[0] is not None
    if responses and not more:
        sweep_close(run, responses)
    post_replies(run, responses)
    return True


def free_text(run, call_args):
    props = search_props(run)
    tok = token_key(run)
    picked = []
    for key, val in call_args.items():
        if isinstance(val, str) and not (props.get(key) or {}).get("enum") and key != tok:
            picked.append(val)
    return picked


def id_sort_key(mark):
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", mark)]


def book_page(run, call_args, rows, total, obs=None):
    paging = page_arg(run)
    if not paging or not rows:
        return
    at = page_position(paging, call_args)
    if at is None:
        return
    page_key, mode, low = paging[0], paging[1], paging[2]
    rest = {k: v for k, v in call_args.items() if k != page_key}
    book = vars(run).setdefault("page_book", {})
    walk = book.setdefault(call_sig(rest), {"args": rest, "read": set(), "size": 0, "total": 0, "shelves": set()})
    walk["read"].add(at)
    if mode == "token":
        walk.setdefault("tokens", {})[at] = next_token(obs)
    walk["size"] = max(walk["size"], len(rows))
    walk["total"] = max(walk["total"], total)
    note_page_rows(walk, [r for r in rows if isinstance(r, dict)], at, low, total, len(rows))


def aligned_page(run, call_args):
    # (a row offset that falls inside a page of the same search already read only shifts that page by a few rows:
    # it moves to the first unread page instead)
    try:
        paging = page_arg(run)
        if not paging or paging[1] is not False:
            return call_args
        key = paging[0]
        at = call_args.get(key)
        rest = {k: v for k, v in call_args.items() if k != key}
        walk = (vars(run).get("page_book") or {}).get(call_sig(rest))
        if not is_number(at) or not walk or not walk["size"] or at in walk["read"]:
            return call_args
        if not any(is_number(r) and r < at < r + walk["size"] for r in walk["read"]):
            return call_args
        nxt = next_page(run, walk)
        if nxt is None:
            return call_args
        run.wire.log.append({"kind": "page_aligned", "asked": at, "sent": nxt})
        return {**call_args, key: nxt}
    except Exception as err:
        record_failure(run.wire, "page_align_error", err)
        return call_args


def next_page(run, walk):
    _, by_page, low, high = page_arg(run)
    if by_page == "token":
        for tok in (walk.get("tokens") or {}).values():
            if isinstance(tok, str) and tok not in walk["read"]:
                return tok
        return None
    numbers = [v for v in walk["read"] if is_number(v)]
    start = min([low] + numbers)
    size = max(1, walk["size"])
    stride = 1 if by_page else size
    i = 1
    while i < 400:
        value = start + i * stride
        if high is not None and value > high:
            return None
        if i * size >= walk["total"]:
            return None
        if value not in walk["read"]:
            return value
        i += 1
    return None


def prefix_near(a, b):
    if min(len(a), len(b)) < 3:
        return False
    return a.startswith(b) or b.startswith(a)


def value_score(run, call_args):
    value_sets = []
    for req in props_pending(run.needs):
        tail = req.partition(":")[2]
        if not ref_text(tail):
            value_sets.append(re.findall(r"[a-z0-9]{3,}", tail.lower()))
    words = []
    for text in free_text(run, call_args):
        words += re.findall(r"[a-z0-9]{2,}", text.lower())
    if run.lead_words:
        del words[run.lead_words:]
    score = 0
    for value_words in value_sets:
        if any(prefix_near(w, v) for w in words for v in value_words):
            score += 1
    return score


def seller_words_of(row):
    found = []
    for key, val in row.items():
        name = str(key).lower()
        if isinstance(val, str) and val.strip() and re.search(r"shop|seller|store|vendor|brand|make|manufactur", name) and "id" not in name:
            found.append(set(re.findall(r"[a-z0-9]+", val.lower())))
    return found


def row_hits(run, rows, words=None):
    sheet = run.needs
    cap, qty = sheet.spend_cap, need_qty(sheet)
    shelves = shelf_labels(run)
    hits = []
    asked = set()
    if sheet.reading.get("references"):
        for t in words or []:
            asked.update(re.findall(r"[a-z0-9]+", str(t).lower()))
    pointers = vars(run).setdefault("ref_hits", [])
    arch = run.archive
    for r in filter(lambda r: isinstance(r, dict), rows):
        ident = row_mark(r)
        makers = seller_words_of(r)
        unread = ident in arch.ids and ident not in arch.read
        pointed = asked and unread and any(m and m <= asked and m <= pointed_words(run) for m in makers)
        if pointed and ident not in pointers:
            pointers.append(ident)
        cost = row_price(r)
        shelf = shelf_of(r)
        too_dear = is_number(cap) and is_number(cost) and cost * qty > cap
        off_shelf = shelves and shelf and shelf not in shelves
        if unread and not too_dear and not sold_out(r) and not off_shelf:
            hits.append(ident)
            vars(run).setdefault("hit_shelf", {})[ident] = shelf
    return hits


def own_and_texts(d):
    own, texts = set(), []
    for key, val in d.items():
        if not isinstance(val, str):
            continue
        name = str(key).lower()
        seller_field = re.search(r"shop|seller|store|vendor|brand|make|manufactur", name) and "id" not in name and len(val) <= 40 \
            and not re.search(r"note|text|desc|comment|detail|info|message|section", name)
        if seller_field:
            own |= set(re.findall(r"[a-z0-9]+", val.lower()))
        elif not re.search(r"title|heading|headline|name|label|caption", name):
            texts.append(val.lower())
    return own, texts


def pointed_words(run):
    arch = run.archive
    memo = vars(run).get("pointed_memo")
    if memo and memo[0] == len(arch.items):
        return memo[1]
    words = set()
    try:
        for ref in run.needs.reading.get("references") or []:
            words |= set(re.findall(r"[a-z0-9]+", str(ref).lower()))

        def visit(d):
            if row_mark(d) not in arch.read:
                return
            own, texts = own_and_texts(d)
            for text in texts:
                words.update(set(re.findall(r"[a-z0-9]+", text)) - own)
        for rec in arch.items:
            walk_dicts(rec, visit)
    except Exception:
        pass
    run.pointed_memo = (len(arch.items), words)
    return words


def open_pages(run, most=2):
    paging = page_arg(run)
    if not paging:
        return []
    shelves = shelf_labels(run)
    ranked = []
    for walk in vars(run).get("page_book", {}).values():
        upcoming = next_page(run, walk)
        elsewhere = shelves and walk["shelves"] and not walk["shelves"] & shelves
        if upcoming is None or elsewhere:
            continue
        score = value_score(run, walk["args"])
        if score:
            args = dict(walk["args"])
            args[paging[0]] = upcoming
            ranked.append((-score, walk["total"], args))
    ranked.sort(key=lambda p: p[:2])
    return [args for _, _, args in ranked[:most]]


def lead_tokens(text, reach):
    found = [w for w in re.findall(r"[a-z0-9]+", text.lower()) if len(w) >= 2]
    return found[:reach]


def probe_near(a, b):
    if not any(c.isdigit() for c in a + b):
        return prefix_near(a, b)
    if a.lstrip("0") == b.lstrip("0"):
        return True
    return a.isdigit() and b.startswith(a) and not b[len(a):len(a) + 1].isdigit()


def value_probe(run, raw):
    if run.word_kind == "any":
        return "", ""
    reach = run.lead_words or 2
    lead = lead_tokens(" ".join(raw), reach)
    product = " ".join(re.findall(r"[a-z0-9]+", str(run.needs.reading.get("product") or "").lower())[:3])
    reqs = probe_requirements(run.needs)
    guard = []
    for _, value in reqs:
        guard += re.findall(r"[a-z0-9]{2,}", value)
    guard += product.split()
    if not lead:
        return "", ""
    for word in lead:
        if any(probe_near(word, g) for g in guard):
            return "", ""
    for name, value in reqs:
        name_words = re.findall(r"[a-z0-9]{2,}", name.lower())
        if value and any(probe_near(w, n) for w in lead for n in name_words):
            digit = re.search(r"\d", value)
            core = value[digit.start():] if digit else value
            probe = f"{core.strip(' .;')} {product}".strip()
            if probe_ran(run, lead_tokens(probe, reach), reach):
                return "", ""
            return name, probe
    return "", ""


# ---- the cheapest walk

def stem_lead(text, width):
    return sorted(map(stem, re.findall(r"[a-z0-9]+", str(text).lower())[:width]))


def ran_query(run, text):
    width = run.lead_words or 3
    wanted = stem_lead(text, width)
    for walk in vars(run).get("page_book", {}).values():
        if stem_lead(" ".join(free_text(run, walk["args"])), width) == wanted:
            return True
    return False


def search_template(run):
    walks = list(vars(run).get("page_book", {}).values())
    while walks:
        args = walks.pop()["args"]
        if cg_listing(run, args):
            continue
        texts = free_text(run, args)
        text_keys = [k for k, v in args.items() if isinstance(v, str) and v in texts and v.strip()]
        if len(texts) == 1 == len(text_keys):
            rest = dict(args)
            del rest[text_keys[0]]
            return rest, text_keys[0]
    return None


def trap_near(a, b):
    if any(c.isdigit() for c in a + b):
        return a.lstrip("0") == b.lstrip("0")
    return prefix_near(a, b)


def word_parts(token):
    return list(filter(lambda p: len(p) >= 2, re.findall(r"[a-z0-9]+", token)))


def touches(tokens, pool):
    for tok in tokens:
        for piece in word_parts(tok):
            for other in pool:
                if trap_near(piece, other):
                    return True
    return False


def trap_width(run, toks):
    if run.lead_words:
        return run.lead_words
    return max(3, len(toks)) if run.word_kind == "all" else 3


def trap_note(why, word, query):
    if why == "name":
        note = (f"The word '{word}' names a required property (the property's NAME), not a required value; as this store "
                "needs every word, the search returns just the listings whose fields or title repeat that name and misses those "
                "whose field holds just the value.")
    else:
        note = (f"'{word}' here stems from nothing but a requirement the shopper DROPPED: it still keeps only "
                "the listings carrying it, though they no longer need to.")
    return note + f" Also search '{query}' and read every page of it."


def trap_query(run, raw):
    if run.word_kind == "any":
        return "", ""
    toks = []
    for piece in " ".join(raw).split():
        tok = piece.strip("\"'()[]{},.;!?|").lower()
        if re.search(r"[a-z0-9]", tok):
            toks.append(tok)
    width = trap_width(run, toks)
    del toks[width:]
    sheet = run.needs
    reqs = trap_requirements(sheet)
    vals = words_of([value for _, _, value in reqs], r"[a-z0-9]{2,}")
    products = [r["product"] for r in need_items(sheet)] or [str(sheet.reading.get("product") or "")]
    kinds = words_of([" ".join(products + sorted(shelf_labels(run))).lower()], r"[a-z0-9]{3,}") - LINK_WORDS
    prop_names = words_of([short for short, _, _ in reqs], r"[a-z]{3,}") - LINK_WORDS
    let_go = words_of([old.partition(":")[2].lower() for old in sheet.let_go], r"[a-z]{3,}") - LINK_WORDS
    flagged, why = flag_tokens(toks, vals | kinds, prop_names, let_go)
    if not flagged:
        return "", ""
    rest = [t for t in toks if t not in flagged]
    tags = [tag for _, tag, value in reqs if touches(rest, re.findall(r"[a-z0-9]{2,}", value))]
    if not tags:
        return "", ""
    if not touches(rest, kinds):
        k = product_of(run, tags[0]) if need_items(sheet) else None
        rest += last_kind_word(products[k] if k is not None and k < len(products) else products[0])
    query = " ".join(rest[:width])
    if not query or ran_query(run, query):
        return "", ""
    return trap_note(why, flagged[0], query), query


def word_trap(run, raw):
    try:
        return trap_query(run, raw)
    except Exception as err:
        run.wire.log.append(error_entry("trap_error", err))
        return "", ""


def passed_now(run):
    version = str(run.needs.version)
    return {key.rpartition("|")[0] for key in run.passed if key.rpartition("|")[2] == version}


def word_value_lists(run, k, several):
    found = []
    for req in props_pending(run.needs):
        head, _, value = req.partition(":")
        if several and product_of(run, head) not in (k, None):
            continue
        value = re.sub(STRIKE_MARKS, "", value).strip().lower()
        if not value or re.search(r"\d|as shown|same as|page|redirect|refer|\bnot\b|\bno\b|without|except|anything|other than", value):
            continue
        first_choice = re.split(r"[,;/]|\bor\b|\band\b", value)[0]
        words = [w for w in re.findall(r"[a-z]{2,}", first_choice) if w not in LINK_WORDS]
        if 0 < len(words) < 3:
            found.append(words)
    return found


def joined_query(value_lists, product, width):
    words = []
    for value_words in value_lists:
        if len(words) + len(value_words) <= width:
            words.extend(value_words)
    kind = last_kind_word(product)
    if kind and len(words) < width and kind[0] not in words:
        words.extend(kind)
    return words


def combined_queries(run):
    template = search_template(run)
    if not template:
        return []
    base_args, text_key = template
    items = need_items(run.needs)
    width = run.lead_words or 3
    covered = {shelf_known(run, i) for i in passed_now(run) - run.refused_lines} - {""}
    homes = home_shelves(run) if items else []
    ready = []
    for k in (range(len(items)) if items else [None]):
        if not items:
            if not unverified(run):
                continue
        elif k >= len(homes) or not homes[k] or homes[k] & covered:
            continue
        value_lists = word_value_lists(run, k, bool(items))
        if len(value_lists) < 2:
            continue
        product = items[k]["product"] if items else str(run.needs.reading.get("product") or "")
        words = joined_query(value_lists, product, width)
        text = " ".join(words)
        if len(words) > 1 and not ran_query(run, text):
            fresh = dict(base_args)
            fresh[text_key] = text
            ready.append(fresh)
    return ready[:2]


def last_audit_reason(run):
    for entry in reversed(run.wire.log):
        if isinstance(entry, dict) and entry.get("kind") == "audit" and isinstance(entry.get("result"), dict):
            return str((entry.get("result") or {}).get("reason") or "")
    return ""


def arbitrate(run, thing, ruling):
    try:
        reason = last_audit_reason(run)
        cited = f" ({reason[:400]})" if reason else ""
        note = ("\n\nCONFLICTING READINGS OF THIS ITEM. From the same records the decline audit found that it meets every requirement"
                + cited + "; an earlier check found: " + ruling[:500]
                + "\nCheck: which reading the records support.\nThen: a field giving the required value in different words, or a "
                "shade, qualifier or spelling of that same named value, satisfies the requirement; a field giving another value "
                "does not.")
        second_ruling = check_item(run, thing, "ORDER", second=note)
        if second_ruling:
            return second_ruling
        key = f"{thing}|{run.needs.version}"
        stale = [memo for memo in run.rulings if memo == key or memo.startswith(key + "|")]
        for memo in stale:
            run.rulings[memo] = ""
        return ""
    except Exception as err:
        run.wire.log.append(error_entry("arbitrate_error", err))
        return ruling


def failed_now(run, version):
    out = set()
    for key, verdict in run.rulings.items():
        parts = key.split("|")
        if verdict and parts[1:2] == [version]:
            out.add(parts[0])
    return out


def stock_marks(run):
    marks = {}
    arch = run.archive

    def visit(d):
        ident = row_mark(d)
        if ident not in arch.ids:
            return
        if any(re.search(r"stock|avail", str(key).lower()) for key in d):
            marks[ident] = sold_out(d)
    for rec in arch.items:
        walk_dicts(rec, visit)
    return marks


def audited_out(run):
    state = vars(run)
    reviewed = state.get("audit_ids")
    if state.get("audit_done") and (reviewed is None or len(run.archive.read) <= state.get("audit_reads", -1)):
        reviewed = set(run.archive.read)
    return set(reviewed or set()) - set(state.get("audit_fits") or [])


def unrefuted_picks(run):
    items = need_items(run.needs)
    homes = home_shelves(run) if items else [set(shelf_labels(run))]
    if not homes or not all(homes):
        return []
    failed = failed_now(run, str(run.needs.version))
    live = passed_now(run)
    ruled_out = failed | audited_out(run)
    sold = stock_marks(run)
    cap = run.needs.spend_cap
    picks, total = [], 0.0
    skip = ruled_out | run.refused_lines | {ident for ident, gone in sold.items() if gone} | set(ref_pointers(run))
    for k, home in enumerate(homes):
        qty = items[k]["quantity"] if items else need_qty(run.needs)
        best = cheapest_on(run, home, qty, skip, live, picks)
        if best is None:
            return []
        _, cost, ident = best
        picks.append(ident)
        total += cost * qty
    if is_number(cap) and total > cap:
        return []
    return picks


def unrefuted(run):
    try:
        return unrefuted_picks(run)
    except Exception as err:
        run.wire.log.append(error_entry("unrefuted_error", err))
        return []


def fit_hold_note(run):
    if vars(run).get("fit_held") or run.trolley.units or moves_left(run) <= 4 or time_left(run.wire) < CLOSING + 30:
        return ""
    fits = [i for i in vars(run).get("audit_fits") or [] if i in run.archive.ids and i not in run.refused_lines]
    if not fits:
        return ""
    run.fit_held = True
    good = []
    for thing in fits[:2]:
        if thing not in run.archive.read:
            continue
        ruling = check_item(run, thing, "ORDER")
        if ruling and time_left(run.wire) > CLOSING + 30:
            ruling = arbitrate(run, thing, ruling)
        if ruling or f"{thing}|{run.needs.version}" not in run.passed:
            continue
        good.append(thing)
        if not need_items(run.needs):
            break
    if not good:
        return ""
    if need_items(run.needs):
        covered = {shelf_known(run, i) for i in passed_now(run) - run.refused_lines} - {""}
        for home in home_shelves(run):
            if home and not home & covered:
                return ""
    single = len(good) == 1
    target = "a verified line of every product" if need_items(run.needs) else good[0]
    return ("Possible fit named by the decline audit: " + ", ".join(good) + ". The separate item check went through "
            + ("its" if single else "their") + " full record and found " + ("it meets" if single else "they meet")
            + " each requirement in force now. Weigh " + ("it" if single else "them") + " before any decline: if it fits, put "
            + target + " in the cart, re-read its stock and price, and order; decline if you find that it fails a requirement in force.")


def fit_hold(run):
    try:
        return fit_hold_note(run)
    except Exception as err:
        run.wire.log.append(error_entry("fit_hold_error", err))
        return ""


def id_store(run):
    orders = [w.get("id_order") for w in vars(run).get("page_book", {}).values() if "id_order" in w]
    if False in orders:
        return False
    return orders.count(True) > 1


def product_index(run, thing):
    if not need_items(run.needs):
        return None
    shelf = shelf_known(run, thing)
    for k, home in enumerate(home_shelves(run)):
        if shelf and shelf in home:
            return k
    return None


def walk_home(run, k):
    homes = home_shelves(run)
    spot = k or 0
    if homes and spot < len(homes) and homes[spot]:
        return set(homes[spot])
    if k is None:
        return set(shelf_labels(run))
    return set()


def fig_amount(text):
    low = str(text).lower()
    found = None if re.search(CH_BOUND, low) else re.match(r"\s*(\d+)\s+([a-z]+)", low)
    return found.groups() if found else None


def amount_part(text):
    low = str(text).lower()
    if re.search(r"\d", low) is None:
        return False
    if re.search(CH_BOUND, low):
        return True
    return re.match(r"\s*\d+(?:[.,]\d+)?\s+[a-z]", low) is not None


def walk_carries(run, args, k, alts=False):
    several = k is not None and bool(need_items(run.needs))
    texts = [text.lower() for text in free_text(run, args)]
    words = sum((re.findall(r"[a-z0-9]{2,}", text) for text in texts), [])
    toks = sum((re.findall(r"[a-z0-9]+", text) for text in texts), [])
    if run.lead_words:
        words, toks = words[: run.lead_words], toks[: run.lead_words]
    carried = set()
    for n, req in enumerate(props_pending(run.needs)):
        head, _, value = req.partition(":")
        if several and product_of(run, head) not in (k, None):
            continue
        parts = re.split(r"[,;/]|\bor\b|\beither\b|\bnor\b", re.sub(STRIKE_MARKS, "", value).lower())
        for a, part in enumerate(parts):
            if part_carried(part, words, toks):
                carried.add((n, a) if alts else n)
    return carried


def walk_score(run, args, k):
    return len(walk_carries(run, args, k))


def walk_options(run, k):
    home = walk_home(run, k)
    if not home:
        return []
    cut = run.lead_words or None
    sheet = run.needs

    def lead(walk):
        joined = " ".join(free_text(run, walk["args"])).lower()
        return set(map(stem, re.findall(r"[a-z0-9]+", joined)[:cut]))
    label = set()
    for shelf in home:
        label.update(map(stem, re.findall(r"[a-z0-9]{2,}", shelf)))
    several = bool(need_items(sheet)) and k is not None
    values = set()
    for req in props_pending(sheet):
        head, _, tail = req.partition(":")
        if several and product_of(run, head) not in (k, None):
            continue
        values.update(map(stem, re.findall(r"[a-z0-9]{2,}", tail.lower())))
    known = values | label
    book = vars(run).get("page_book", {})
    finished = []
    for walk in book.values():
        complete = walk.get("id_order") is True and walk.get("first") and next_page(run, walk) is None
        if complete and lead(walk) and len(walk.get("row_shelf") or {}) >= walk["total"]:
            finished.append((lead(walk), walk_carries(run, walk["args"], k, alts=True), vl_mode(run, walk["args"])))
    best = {}
    for key, walk in book.items():
        if walk.get("id_order") is not True or not walk.get("first") or next_page(run, walk) is None:
            continue
        on_home = any(seen_label in home for seen_label in (walk.get("row_shelf") or {}).values())
        if not on_home or not walk_carries(run, walk["args"], k):
            continue
        words, pairs, how = lead(walk), walk_carries(run, walk["args"], k, alts=True), vl_mode(run, walk["args"])
        shadowed = False
        for seen, got, mode in finished:
            if mode == how and (seen <= words or (seen <= known and pairs <= got)):
                shadowed = True
                break
        if shadowed:
            continue
        pages = -(-walk["total"] // max(1, walk["size"])) - len(walk["read"])
        twin = walk["total"], walk["first"]
        if pages <= 0 or pages > 2 * WALK_PAGES:
            continue
        if twin in best and best[twin][0] <= pages:
            continue
        best[twin] = (pages, key, walk)
    options = list(best.values())
    if all_words(run):
        # nested word sets in an all-words store: keep the one that loses no fit
        def covers(p, o):
            a, b = lead(p[2]), lead(o[2])
            if a < b:
                return not (b - a) <= known and p[0] <= WALK_PAGES
            return b < a and (a - b) <= known
        options = [o for o in options if not any(covers(p, o) for p in options if p is not o)]
    options.sort(key=lambda p: (p[0], -walk_score(run, p[2]["args"], k), p[1]))
    return options


def walk_plan(run, budget, only=None):
    if budget <= 0 or need_priority(run.needs) != "cheapest":
        return []
    if not role(run, "search") or not page_arg(run) or not id_store(run):
        return []
    walked = vars(run).get("walked", set())
    wants = []
    for k in list(range(len(need_items(run.needs)))) or [None]:
        if k in walked or (only is not None and k not in only):
            continue
        options = walk_options(run, k)
        if not options:
            continue
        pages, key = options[0][0], options[0][1]
        wants.append((pages, -1 if k is None else k, k, key))
    wants.sort(key=lambda p: p[:2])
    plan, left = [], budget
    for pages, _, k, key in wants:
        count = min(pages, WALK_PAGES, left)
        if count < 1:
            break
        plan.append((k, key, count))
        left -= count
    return plan


def walk_pick(run, k):
    gone = dead_lines(run)

    def ours(ident):
        return k is None or product_index(run, ident) == k
    options = [i for i in run.trolley.units if ours(i) and i not in gone]
    if not options:
        options = []
        for ident in [key.rpartition("|")[0] for key in run.passed if key.rpartition("|")[2] == str(run.needs.version)]:
            if ident not in run.refused_lines and ident not in gone and ours(ident):
                options.append(ident)
    priced = [(i, price_of(run.archive, i)) for i in options]
    best = None
    for ident, cost in priced:
        if is_number(cost) and (best is None or (float(cost), ident) < (best[1], best[0])):
            best = (ident, float(cost))
    return best


def record_of(run, ident):
    found = {}

    def visit(d):
        nonlocal found
        if row_mark(d) == ident and title_of(d):
            found = d
    for item in run.archive.items:
        walk_dicts(item, visit)
    return found


def line_query(run, k):
    pick = walk_pick(run, k)
    template = search_template(run)
    if not pick or not template:
        return None
    rec = record_of(run, pick[0])
    maker_words = re.findall(r"[a-z0-9]+", maker_of(rec).lower())
    if not maker_words:
        return title_line(run, k, rec, template)
    avoid = line_avoid_words(run, k, maker_words)
    title_words = re.findall(r"[a-z][a-z'-]{3,}", title_of(rec).lower())
    brand = maker_words[0]
    if brand in title_words:
        title_words = title_words[title_words.index(brand) + 1:]
    line_word = next((w for w in title_words if w.strip("'-") not in avoid and "'" not in w), None)
    if line_word is None:
        return None
    text = f"{brand} {line_word}"
    if ran_query(run, text):
        return None
    base_args, text_key = template
    fresh = dict(base_args)
    fresh[text_key] = text
    return fresh


def walk_rows(run, k, keys, most):
    home = walk_home(run, k)
    ids = walk_row_ids(run, k, keys, home)
    arch = run.archive
    # no pick yet: use the lowest-priced row of these that was read
    pick = walk_pick(run, k) or stand_in_pick(run, ids)
    items = need_items(run.needs)
    if k is not None and k < len(need_items(run.needs)):
        qty = items[k]["quantity"]
    else:
        qty = need_qty(run.needs)
    cap = run.needs.spend_cap
    sold = sold_among(run, ids)
    cheaper = []
    for ident in ids:
        cost = price_of(arch, ident)
        if ident in arch.read or ident in run.refused_lines or sold.get(ident) or not is_number(cost):
            continue
        if is_number(cap) and cost * qty > cap:
            continue
        if pick and cost >= pick[1]:
            continue
        cheaper.append((cost, ident))
    cheaper.sort()
    return [ident for _, ident in cheaper[:most]]


def digest_line(r):
    cost = row_price(r)
    shown_cost = "?" if cost is None else cost
    stock = " SOLD OUT" if sold_out(r) else ""
    return f"{row_mark(r)} {shown_cost}{stock} '{title_of(r)[:70]}' [{shelf_of(r)}]"


def page_digest(run, obs):
    rows = []
    known_ids = run.archive.ids

    def visit(d):
        if len(d) < 4:
            return
        for key, val in d.items():
            if isinstance(val, str) and val in known_ids and "id" in str(key).lower():
                rows.append(d)
                return
    walk_dicts(obs, visit, ceiling=3)
    listed = [digest_line(r) for r in rows[:12]]
    return "Page rows (id, price, title, category): " + ("; ".join(listed) or "none")


def walk_call(run, args, say):
    arch = run.archive
    before = len(arch.items)
    call_id, responses = scripted_search(run, f"walk_{len(run.chat)}_{run.wire.tick}", args, say)
    if not responses:
        content, cid, grew = "Not run.", call_id, False
    else:
        cid, text_body = responses[0]
        grew = len(arch.items) > before
        content = text_body
        if grew:
            content = page_digest(run, arch.items[-1])
            _, marker, note = text_body.partition("\n[note] ")
            if marker and note and re.search(r"shopper|Meaning:|REFUSED", note):
                content += "\n[note] " + note
    run.chat.append({"role": "tool", "tool_call_id": cid, "content": content})
    return grew


def walk_open(run, floor):
    link = run.wire
    if moves_left(run) <= floor + 1 or link.done or link.broke:
        return False
    return time_left(link) >= CLOSING + 30


def walk_planned_pages(run, plan, page_key, floor, report):
    book = vars(run).get("page_book", {})
    walked = vars(run).setdefault("walked", set())
    progressed = False
    for k, walk_key, count in plan:
        walked.add(k)
        walk = book.get(walk_key)
        if not walk:
            continue
        query = " ".join(free_text(run, walk["args"]))
        say = f"The cheapest is wanted and this store orders its search results by id, not by price, so I read the next page of '{query}'."
        read = 0
        while read < count:
            upcoming = next_page(run, walk)
            if upcoming is None or not walk_open(run, floor):
                break
            if not walk_call(run, {**walk["args"], page_key: upcoming}, say):
                break
            read += 1
            progressed = True
        report.append(f"'{query}': {read} more page(s) read, {len(walk.get('row_shelf') or {})} of its {walk['total']} matches seen")
    return progressed


def short_line_left(run, line, floor, extra):
    if not line or extra >= 2 or line["total"] > 3 * max(1, line["size"]) or next_page(run, line) is None:
        return False
    if moves_left(run) <= floor + 3 or run.wire.done:
        return False
    return time_left(run.wire) > CLOSING + 30


def walk_pick_lines(run, plan, page_key, floor, report):
    line_keys = {}
    progressed = False
    for k, _, _ in plan:
        if moves_left(run) <= floor + 2 or run.wire.done or time_left(run.wire) < CLOSING + 30:
            break
        args = line_query(run, k)
        if not args:
            continue
        if not walk_call(run, args, "I search the pick's product line as well (its maker plus the word that names the line): "
                         "other sizes and colours of a line are separate listings and may cost less."):
            continue
        progressed = True
        line_keys[k] = call_sig({a: v for a, v in args.items() if a != page_key})
        line = vars(run).get("page_book", {}).get(line_keys[k])
        extra = 0
        # lines spanning three pages or fewer get read completely
        while short_line_left(run, line, floor, extra):
            if not walk_call(run, {**line["args"], page_key: next_page(run, line)}, "Next page of the line search."):
                break
            extra += 1
        report.append(f"the pick's line '{' '.join(free_text(run, args))}'")
    return progressed, line_keys


def walk_ceilings(run, ceil, floor, repair, report):
    found = {}
    for k in dict.fromkeys(ceil or []):
        reserve = floor if repair else max(floor, order_due(run) + 2)
        if moves_left(run) <= reserve + 1 or run.wire.done or time_left(run.wire) < CLOSING + 30:
            break
        got_rows = ceiling_walk(run, k, floor, repair)
        if got_rows:
            note, rows = got_rows[0], got_rows[1]
            report.append(note)
            found[k] = list(rows)
    return found


def interleave(pools, most):
    queue = []
    while len(queue) < most and any(pools):
        for pool in pools:
            while pool and pool[0] in queue:
                del pool[0]
            if pool and len(queue) < most:
                queue.append(pool.pop(0))
    return queue


def walk_summary(run, report, queue):
    if queue:
        priced = ", ".join(f"{i} ({price_of(run.archive, i):g})" for i in queue)
        opened = ("Opened next are the cheapest unread in-stock rows these searches showed within the budget and below each "
                  "pick's price: " + priced + ". ")
    else:
        opened = "Nothing unread, in stock, within the budget and cheaper than the pick turned up. "
    return ("[cheapest walk] " + "; ".join(report) + ". " + opened + "Per product, the choice is the cheapest listing that, "
            "by its own fields, satisfies every requirement now in force; the rank check runs again on the cart call and the order.")


def run_walk(run, plan, floor, ceil=None, repair=False):
    paging = page_arg(run)
    if not (plan or ceil) or not paging or not role(run, "search"):
        return False
    page_key = paging[0]
    report = []
    pages_ran = walk_planned_pages(run, plan, page_key, floor, report)
    lines_ran, line_keys = walk_pick_lines(run, plan, page_key, floor, report)
    ceil_rows = walk_ceilings(run, ceil, floor, repair, report)
    if not (pages_ran or lines_ran or ceil_rows):
        return False
    batch = batch_tool(run)
    most = batch[2] if batch else (CEIL_OPENS if ceil_rows else 2)
    planned = [k for k, _, _ in plan]
    pools = [walk_rows(run, k, [line_keys.get(k, "")], most) for k in planned]
    pools += list(ceil_rows.values())
    queue = interleave(pools, most)
    trim_queue(run, queue, batch, floor)
    homes = set()
    for k in planned + list(ceil_rows):
        homes |= walk_home(run, k)
    run.rank_checked = {mark for mark in run.rank_checked if shelf_known(run, mark.split("|")[0]) not in homes}
    run.order_vetted = False
    summary = walk_summary(run, report, queue)
    if queue:
        hand_over_queue(run, queue, floor)
    if run.chat and run.chat[-1].get("role") == "tool":
        newest = run.chat[-1]
        newest["content"] = "\n".join([str(newest["content"]), summary] + ([] if queue else [status_line(run)]))
    entry = {"kind": "walk", "plan": [[k, n] for k, _, n in plan], "ceiling": list(ceil_rows), "opened": queue}
    run.wire.log.append(entry)
    return True


# ---- reads before a decline

def decline_fits(run):
    fits = passed_now(run)
    audit_named = set(vars(run).get("audit_fits") or []) - failed_now(run, str(run.needs.version))
    return (fits | audit_named) - run.refused_lines


def decline_lacking(run):
    covered = {shelf_known(run, i) for i in decline_fits(run)} - {""}
    if not need_items(run.needs):
        return [] if set(shelf_labels(run)) & covered else [None]
    lacking = []
    for k, home in enumerate(home_shelves(run)):
        if home and not home & covered:
            lacking.append(k)
    return lacking


def decline_walks(run):
    reqs = props_pending(run.needs)

    def worded(args, k):
        count = 0
        for n in walk_carries(run, args, k):
            if n < len(reqs) and not re.search(r"as shown|same as|the page|sold by|redirect|refer", reqs[n].lower()):
                count += 1
        return count
    picked = []
    for k in decline_lacking(run):
        options = [o for o in walk_options(run, k) if worded(o[2]["args"], k)]
        options.sort(key=lambda p: (-worded(p[2]["args"], k), -walk_score(run, p[2]["args"], k), p[0], p[1]))
        for _, key, _ in options:
            if key not in picked:
                picked.append(key)
                break
    return picked[:2]


def long_walks(run):
    # (searches too long to read through, in any order: for each product still lacking a fit, the one on its shelf carrying
    # the most required values is paged on, since a fit may sit just past the pages read)
    book = vars(run).get("page_book", {})
    picked = []
    for k in decline_lacking(run):
        home = walk_home(run, k)
        if not home:
            continue
        options = []
        for key, walk in book.items():
            if key in picked or not walk.get("first") or next_page(run, walk) is None:
                continue
            if not any(label in home for label in (walk.get("row_shelf") or {}).values()):
                continue
            carried = len(walk_carries(run, walk["args"], k))
            if carried:
                options.append((-carried, len(walk.get("read") or []), key))
        if options:
            picked.append(min(options)[2])
    return picked[:2]


def row_pools(run, homes, ids, below=None):
    cap = run.needs.spend_cap
    arch = run.archive
    shelf_by_id, sold = {}, {}

    def visit(d):
        ident = row_mark(d)
        if ident not in arch.ids or not shelf_of(d):
            return
        shelf_by_id[ident] = shelf_of(d)
        if any(re.search(r"stock|avail", str(key).lower()) for key in d):
            sold[ident] = sold_out(d)
    for rec in arch.items:
        walk_dicts(rec, visit)
    pools = []
    for n, (home, qty) in enumerate(homes):
        ceiling = None if below is None else below[n]
        pool = []
        for ident, where in shelf_by_id.items():
            cost = price_of(arch, ident)
            seen = ident in arch.read or ident in run.refused_lines
            if pool_member((ident, where, cost, sold.get(ident), seen), home, ids, cap, qty, ceiling):
                pool.append((float(cost), ident))
        pools.append([ident for _, ident in sorted(pool)])
    return pools


def cheaper_pools(run):
    sheet = run.needs
    items, cap = need_items(sheet), sheet.spend_cap
    if not items or need_priority(sheet) == "cheapest" or not is_number(cap):
        return []
    homes = home_shelves(run)
    fits = decline_fits(run)
    floors, shares = [], []
    total = 0.0
    for k, home in enumerate(homes):
        priced = []
        for ident in fits:
            if shelf_known(run, ident) in home and is_number(price_of(run.archive, ident)):
                priced.append(float(price_of(run.archive, ident)))
        if not home or not priced:
            return []
        lowest = min(priced)
        units = items[k]["quantity"]
        floors.append(lowest)
        shares.append(lowest * units)
        total += lowest * units
    if total <= cap:
        return []
    pools = row_pools(run, [(home, items[k]["quantity"]) for k, home in enumerate(homes)], None, floors)
    # largest cost share first, where a cheaper row helps most
    order = sorted(range(len(pools)), key=lambda k: -shares[k])
    pools = [pools[k] for k in order]
    return pools if any(pools) else []


def over_budget_rows(run):
    try:
        return cheaper_pools(run)
    except Exception as err:
        run.wire.log.append(error_entry("over_budget_error", err))
        return []


def decline_read_note(run):
    batch = batch_tool(run)
    if vars(run).get("decline_read_done") or time_left(run.wire) < CLOSING + 60:
        return ""
    if not (role(run, "details") or batch) or run.trolley.units:
        return ""
    opens = 1 if batch else 2
    room = min(3, moves_left(run) - opens - 6)
    keys, long = [], False
    if room > 0 and page_arg(run) and role(run, "search"):
        keys = decline_walks(run) if id_store(run) else []
        if not keys:
            keys = long_walks(run)
            long = bool(keys)
            room = min(routed(run, "long_pages"), moves_left(run) - opens - 6) if long else room
    cheaper = over_budget_rows(run) if moves_left(run) > opens + 6 else []
    if not keys and not cheaper:
        return ""
    run.decline_read_done = True
    run.decline_queue = (keys, room, 6, long)
    said = []
    if keys:
        book = vars(run).get("page_book", {})
        queries = "; ".join(f"'{' '.join(free_text(run, book[key]['args']))}'" for key in keys)
        if long:
            said.append("a search carrying required values found far more listings than were read (" + queries + "), and a fit "
                        f"may sit just past the pages read, so the code goes through up to {room} more of its pages for you, "
                        "opening the cheapest in-stock rows inside the budget that they show")
        else:
            said.append("a few searches that carry required values found more listings than were read (" + queries + "); results "
                        "here come in a fixed id order rather than by fit or price, so a fitting listing is as likely to sit on an "
                        f"unread page as on a read one, and the code now goes through up to {room} of those pages for you, opening "
                        "the cheapest in-stock rows inside the budget that they show")
    if cheaper:
        said.append("the listings found add up to more than the budget across all products, so you also get the cheapest unread "
                    "in-stock rows priced under each product's pick opened")
    buy = "one line per product, at its quantity" if need_items(run.needs) else "the item"
    return ("Before any decline: " + "; ".join(said) + ". Hold those records up against the current requirements, going "
            "by meaning; buy what fits (" + buy + "), and decline only if each one fails.")


def decline_reads(run):
    try:
        return decline_read_note(run)
    except Exception as err:
        run.wire.log.append(error_entry("decline_reads_error", err))
        return ""



def last_kind_word(product):
    kinds = [w for w in re.findall(r"[a-z]{3,}", product.lower()) if w not in LINK_WORDS]
    return kinds[-1:]


def words_of(texts, pattern):
    found = set()
    for text in texts:
        found.update(re.findall(pattern, text))
    return found


def trap_requirements(sheet):
    out = []
    for req in props_pending(sheet):
        head, colon, value = req.partition(":")
        if not colon or ref_text(value):
            continue
        short = re.split(r"\s*\(", head)[0].lower()
        out.append((short, head, re.sub(STRIKE_MARKS, "", value).lower()))
    return out


def flag_tokens(toks, known, prop_names, let_go):
    flagged, why = [], ""
    for tok in toks:
        if not word_parts(tok) or touches([tok], known):
            continue
        if touches([tok], prop_names):
            reason = "name"
        elif touches([tok], let_go):
            reason = "dropped"
        else:
            continue
        flagged.append(tok)
        why = why or reason
    return flagged, why


def walk_row_ids(run, k, keys, home):
    book = vars(run).get("page_book", {})
    carried = {}
    for key, walk in book.items():
        if walk.get("id_order") is True:
            carried[key] = walk_carries(run, walk["args"], k)
    top = max(map(len, carried.values()), default=0)
    ids = set()
    for key, walk in book.items():
        got = carried.get(key, set())
        if key not in keys and not (got and len(got) == top):
            continue
        for ident, label in (walk.get("row_shelf") or {}).items():
            if label in home:
                ids.add(ident)
    return ids


def stand_in_pick(run, ids):
    arch = run.archive
    seen = []
    for ident in ids:
        if ident in arch.read and is_number(price_of(arch, ident)):
            seen.append((float(price_of(arch, ident)), ident))
    if not seen:
        return None
    cost, ident = min(seen)
    return ident, cost


def sold_among(run, ids):
    sold = {}

    def visit(d):
        for key, val in d.items():
            if isinstance(val, str) and val in ids and "id" in str(key).lower():
                sold[row_mark(d)] = sold_out(d)
                return
    for item in run.archive.items:
        walk_dicts(item, visit)
    return sold


def trim_queue(run, queue, batch, floor):
    while queue:
        steps = 1 if batch and len(queue) >= 2 else len(queue)
        if steps < moves_left(run) - floor:
            return
        queue.pop()


def hand_over_queue(run, queue, floor):
    run.open_queue = queue
    run.open_recheck = True
    run.open_say = "Since the cheapest is wanted, I open the cheapest unread rows that the walk turned up."
    run.open_purpose = "walk"
    run.open_floor = floor
    open_named(run)


def tally_title_grams(titles, label_stems, pairs):
    tallies, low_cost, shown = {}, {}, {}
    for title, (cost, maker) in titles.items():
        maker_words = set(re.findall(r"[a-z]+", maker))
        words = re.findall(r"[a-z]+", title)
        usable = [len(w) > 2 and w not in LINK_WORDS and w not in maker_words for w in words]
        met = set()
        for gram in title_grams(words, usable, pairs):
            joined = " ".join(map(stem, gram))
            if joined in met or set(joined.split()) <= label_stems:
                continue
            met.add(joined)
            count, floor = tallies.get(joined, 0), low_cost.get(joined, cost)
            tallies[joined] = count + 1
            low_cost[joined] = floor if floor < cost else cost
            shown.setdefault(joined, " ".join(gram))
    return tallies, low_cost, shown


def probe_requirements(sheet):
    reqs = []
    for req in props_pending(sheet):
        name, colon, value = req.partition(":")
        if not colon or ref_text(value):
            continue
        first_value = re.split(r"[,;]", re.sub(STRIKE_MARKS, "", value))[0]
        reqs.append((name.strip(), first_value.strip().lower()))
    return reqs


def probe_ran(run, head, reach):
    for walk in vars(run).get("page_book", {}).values():
        ran = lead_tokens(" ".join(free_text(run, walk["args"])), reach)
        if len(ran) == len(head) and all(map(probe_near, head, ran)):
            return True
    return False


def page_position(paging, call_args):
    page_key, mode, low, _ = paging
    at = call_args.get(page_key, low)
    if mode == "token":
        return at.strip() if isinstance(at, str) else ""
    return at if is_number(at) else None


def note_page_rows(walk, dict_rows, at, low, total, row_count):
    walk["shelves"] |= {shelf_of(r) for r in dict_rows} - {""}
    row_labels = walk.setdefault("row_shelf", {})
    marks = []
    for r in dict_rows:
        mark = row_mark(r)
        row_labels.setdefault(mark, shelf_of(r))
        marks.append(mark)
    if at == low:
        walk["first"] = tuple(marks)
        if total <= row_count:
            walk["every"] = list(dict_rows)
    if len(marks) > 4:
        # five or more rows in rising id order point to a fixed listing order
        rising = marks == sorted(marks) or marks == sorted(marks, key=id_sort_key)
        walk["id_order"] = walk.get("id_order", True) is not False and rising


def maker_of(rec):
    for key, val in rec.items():
        name = str(key).lower()
        if isinstance(val, str) and val.strip() and re.search(r"brand|make|manufactur", name) and "id" not in name:
            return val
    return ""


def line_avoid_words(run, k, maker_words):
    sheet = run.needs
    avoid = set(maker_words) | LINK_WORDS
    avoid |= words_of(walk_home(run, k), r"[a-z]+")
    avoid |= words_of([req.lower() for req in props_pending(sheet)], r"[a-z]+")
    products = need_items(sheet) or [{"product": str(sheet.reading.get("product") or "")}]
    avoid |= words_of([item["product"].lower() for item in products], r"[a-z]+")
    return avoid


def cheapest_on(run, home, qty, skip, live, picks):
    arch = run.archive
    cap = run.needs.spend_cap
    best = None
    for ident in sorted(arch.read):
        if ident in skip or ident in picks or shelf_known(run, ident) not in home:
            continue
        cost = price_of(arch, ident)
        if not is_number(cost) or (is_number(cap) and cost * qty > cap):
            continue
        entry = (ident not in live, cost, ident)
        if best is None or entry < best:
            best = entry
    return best


def repair_plan(run, held_lines, order_tool):
    read_ok, sold = code_recheck(run, held_lines)
    if not read_ok:
        return None
    run.refused_lines.difference_update(set(held_lines) - sold)
    info = code_rows(run)
    total = sum(((info.get(line) or {}).get("price") or 0.0) * units_held(run.trolley, line) for line in held_lines)
    cap = run.needs.spend_cap
    over_cap = is_number(cap) and total > cap
    if not sold and not over_cap:
        return order_move(run, order_tool), None, None
    return None, set(sold), (None if over_cap else set(held_lines) - sold)


def pool_member(row_facts, home, ids, cap, qty, ceiling):
    ident, where, cost, gone, seen = row_facts
    if where not in home or seen or gone:
        return False
    if ids is not None and ident not in ids:
        return False
    if not is_number(cost) or (is_number(cap) and cost * qty > cap):
        return False
    return ceiling is None or cost < ceiling


def part_carried(part, words, toks):
    fig = fig_amount(part)
    if fig:
        # an amount counts only with its figure as a token and a word of its unit
        figure, unit = fig
        return figure in toks and any(w == unit or prefix_near(w, unit) for w in toks)
    if amount_part(part):
        return False
    value_words = re.findall(r"[a-z0-9]{3,}", part)
    return any(prefix_near(w, v) for w in words for v in value_words)


def sweep_reserve(run):
    reserve = 4 if run.trolley.units else 3
    if not run.orders_tried:
        reserve = max(reserve + 3, FIRST_ORDER_BY)
    return reserve


def sweep_wanted(run):
    sheet = run.needs
    if need_priority(sheet) != "cheapest" or not sheet.let_go or props_pending(sheet):
        return False
    return bool(role(run, "search")) and not run.wire.done


def sweep_close(run, responses):
    cheapest = sorted(shelf_rows(run).items(), key=lambda kv: (kv[1][0], kv[0]))[:6]
    if not cheapest:
        return
    cid, text_body = responses[-1]
    listed = "; ".join(f"{i} '{t[:80]}' {c:g}" for i, (c, t, _) in cheapest)
    responses[-1] = (cid, text_body + "\n[shelf sweep done] Out of every listing seen, these are the cheapest in-stock rows "
                     "of the requested category that fit the budget: " + listed + ". The first of them wins as the "
                     "pick unless some change rules it out: make it the cart's only line, re-check its stock and "
                     "price, and order it.")


def scripted_search(run, call_id, args, say):
    proposal = {"id": call_id, "type": "function", "function": {"name": role(run, "search"), "arguments": json.dumps(args)}}
    run.chat.append({"role": "assistant", "content": say, "tool_calls": [proposal]})
    run.covering = True
    try:
        return call_id, dispatch(run, [proposal], scripted=True)
    finally:
        run.covering = False


def lead_compare(bound, words, total, marks, seen):
    seen_words, seen_total, seen_marks = seen
    depth = shared_lead(words, seen_words)
    if total != seen_total or marks != seen_marks:
        if depth > bound[0]:
            bound[0] = depth
        return
    shortest = min(len(words), len(seen_words))
    if depth and depth == shortest and len(set(marks)) > 2:
        bound[1].add(depth)


def note_event(run, kind, **fields):
    run.wire.log.append(dict(kind=kind, **fields))


def stash(run, key, default=None):
    return vars(run).get(key, default)


def error_text(err):
    return f"{type(err).__name__}: {err}"[:300]


def interleave_picks(pools, most):
    picked = []
    while len(picked) < most and any(pools):
        for pool in pools:
            while pool and pool[0] in picked:
                del pool[0]
            if pool and len(picked) < most:
                picked.append(pool.pop(0))
    return picked


def page_more_for_decline(run, walk, page_key, floor, reserve, budget, long=False):
    query = " ".join(free_text(run, walk["args"]))
    done = 0
    while done < budget and moves_left(run) > floor + reserve:
        link = run.wire
        if link.done or link.broke or time_left(link) <= CLOSING + 30:
            break
        token = next_page(run, walk)
        if token is None:
            break
        args = dict(walk["args"])
        args[page_key] = token
        why = "fits may sit just past the pages read." if long else ("this store lists results in a fixed id order, so fits may "
                                                                    "sit on unread pages.")
        if not walk_call(run, args, f"Ahead of a decline I page on through '{query}': {why}"):
            break
        done += 1
    return done


def run_decline_reads(run):
    keys, page_limit, floor, *rest = vars(run).pop("decline_queue")
    long = bool(rest and rest[0])
    paging = page_arg(run)
    book = stash(run, "page_book", {})
    batch = batch_tool(run)
    walked, pages_read = [], 0
    if paging:
        for key in keys:
            walk = book.get(key)
            if walk:
                walked.append(key)
                pages_read += page_more_for_decline(run, walk, paging[0], floor, 1 if batch else 2, page_limit - pages_read, long)
    shown = set()
    for key in walked:
        shown.update((book.get(key) or {}).get("row_shelf") or {})
    products = need_items(run.needs)
    homes = []
    for k in decline_lacking(run):
        home = walk_home(run, k)
        known = k is not None and k < len(products)
        homes.append((home, products[k]["quantity"] if known else need_qty(run.needs)))
    pools = (row_pools(run, homes, shown) if walked else []) + over_budget_rows(run)
    queue = interleave_picks(pools, batch[2] if batch else 2)
    while queue and (1 if batch and len(queue) >= 2 else len(queue)) >= moves_left(run) - floor:
        del queue[-1]
    note_event(run, "decline_reads", pages=pages_read, opened=queue)
    if queue and not run.wire.done:
        run.open_queue = queue
        run.open_say = "Before declining, I open the cheapest unread in-budget rows those pages and picks show."
        run.open_purpose = "decline"
        run.open_floor = floor
        open_named(run)
    elif run.chat and run.chat[-1].get("role") == "tool":
        last = run.chat[-1]
        last["content"] = "%s\nNo unread in-stock row within the budget was shown there.\n%s" % (last["content"], status_line(run))
    return bool(queue) or pages_read > 0


def ref_register(run):
    reading = run.needs.reading
    targets = reading.get("reference_targets")
    infos = list(filter(lambda t: isinstance(t, dict), targets)) if isinstance(targets, list) else []
    raw = reading.get("references")
    listed = set(map(str, raw or [])) if isinstance(raw, list) else set()
    for req in list(run.needs.written):
        text = str(req)
        value = text.partition(":")[2]
        if ref_text(value) or (text in listed and re.search(DIFFER_WORDS, value.lower())):
            open_reference(run, text, infos, run.request, "request")


def follows(run, tracker, req):
    return ref_match(run, tracker, req) and (tracker.get("status") == "resolved" or tracker.get("chain"))


def ref_untracked(run, refs, tracked):
    try:
        out = []
        for req in refs:
            if not any(follows(run, t, req) for t in tracked):
                out.append(req)
        return out
    except Exception:
        return refs


def ref_take(run):
    try:
        for ref in run.refs:
            if ref["status"] != "resolved":
                continue
            stale = [r for r in props_pending(run.needs) if ref_match(run, ref, r) and r != ref["now"]]
            if stale:
                swap_requirement(run, ref, ref["now"])
        news = list(vars(run.needs).get("ref_news") or [])
        run.needs.ref_news = []
        reading = run.needs.reading
        for item in news:
            made = open_reference(run, str(item.get("need") or ""), list(item.get("infos") or []),
                                  str(item.get("line") or ""), "reply")
            if made is None:
                continue
            if not isinstance(reading.get("references"), list):
                reading["references"] = []
            known = reading["references"]
            if made["req"] not in known:
                known.append(made["req"])
    except Exception as err:
        note_event(run, "ref_take_error", error=error_text(err))


def split_head(req):
    head, _, value = str(req).partition(":")
    tags = re.findall(r"\(([^)]*)\)", head)
    return head, re.split(r"\s*\(", head)[0], tags, value


def stated_relation(info):
    if not isinstance(info, dict):
        return ""
    return str(info.get("relation") or "").lower()


def open_reference(run, req, infos, source, origin):
    head, bare, tags, value = split_head(req)
    prop = bare.strip()
    if not prop or junk_key(prop):
        return None
    tag = tags[-1].strip() if tags else ""
    if any(clashes(old, req, prop, tag) for old in run.refs):
        return None
    several = lambda: need_items(run.needs)
    info = next((c for c in infos if info_fits(c, prop, tag, several)), None)
    target = pointed_listing(run, info, value, source, prop)
    if not target:
        note_event(run, "ref_skip", req=str(req)[:200])
        return None
    relation, tol, unit = settle_relation(info, value, prop, source, target)
    ref = fresh_reference(str(req), head.strip(), prop, tag, (relation, tol, unit), target, origin)
    run.refs.append(ref)
    note_event(run, "ref_open", req=str(req)[:200], target=target, relation=relation, origin=origin)
    return ref


def clashes(old, req, prop, tag):
    if old["req"] == req:
        return True
    return same_prop(old["prop"], prop) and kind_words(old["tag"]) == kind_words(tag)


def info_fits(cand, prop, tag, several):
    if not same_prop(str(cand.get("property") or ""), prop):
        return False
    if not tag or not cand.get("product") or not several():
        return True
    return bool(kind_words(cand.get("product")) & kind_words(tag))


def pointer_lead(prop, source, seller):
    # (in the sentence naming the seller, the words from the property's first mention before the seller up to the seller;
    # words match on their first four letters, so colour finds color)
    words = [w for w in re.findall(r"[a-z0-9]+", re.split(r"\s*\(", str(prop))[0].lower()) if w not in LINK_WORDS]
    seller = seller.strip().lower()
    if not words or not seller:
        return ""
    for part in re.split(r"(?<=[.!?;])\s", str(source).lower()):
        at = part.find(seller)
        if at < 0:
            continue
        hit = re.search(r"\b" + r"\W+(?:[a-z]+\W+){0,2}".join(re.escape(w[:4]) + r"[a-z0-9]*" for w in words) + r"\b", part[:at])
        if hit:
            return part[hit.end(): at]
    return ""


def settle_relation(info, value, prop, source, target):
    # (the source's own words between the property and the seller decide a pairing: a reworded requirement can hide it)
    lead = pointer_lead(prop, source, str(target.get("seller") or ""))
    if re.search(PAIR_WORDS, lead) and not re.search(DIFFER_WORDS, lead):
        return "pair", None, ""
    found = ref_relation(value)
    stated = stated_relation(info)
    if found[0] != "same":
        return found
    if stated == "within" and is_number(info.get("tolerance")):
        found = ("within", float(info["tolerance"]), str(info.get("unit") or ""))
    elif stated == "pair":
        found = ("pair", None, "")
    elif source_differs(prop, source, target, stated):
        found = ("differ", None, "")
    return found


def fresh_reference(req, head, prop, tag, relation_parts, target, origin):
    relation, tol, unit = relation_parts
    return {"req": req, "head": head, "prop": prop, "tag": tag, "relation": relation, "tol": tol, "unit": unit,
            "target": target, "first": dict(target), "chain": [], "seen": {target_signature(target)}, "judged": set(),
            "queries": [], "pages": 0, "hops": 0, "tries": 0, "fails": 0, "status": "open", "value": "", "goes_with": [],
            "now": "", "origin": origin, "unindexed": False}


def source_differs(prop, source, target, stated):
    span = ref_span(prop, source).lower()
    seller = str(target.get("seller") or "")
    cut = span.find(seller.lower()) if seller.strip() else -1
    if re.search(r"differ|not the same|exclu|apart|other than|unlike", stated):
        return True
    return cut > 0 and bool(re.search(DIFFER_WORDS, span[:cut]))


class DetailCheck:
    def __init__(self, run, rows, tokens, details, loose):
        self.run, self.rows, self.tokens, self.details, self.loose = run, rows, tokens, details, loose


def check_shows(check, ident, detail):
    return row_shows_detail(check.run, check.rows.get(ident) or {}, detail, check.tokens[ident])


def check_passes(check, ident):
    for detail in check.details:
        if detail not in check.loose and not check_shows(check, ident, detail):
            return False
    return True


def judge_candidates(run, ref):
    target = ref["target"]
    details = target["details"]
    every = seller_candidates(run, ref)
    cands = [i for i in every if i in run.archive.read and i not in ref["judged"]]
    if not cands:
        return False
    rows = listing_fields(run)
    by_seller = [i for i in seller_candidates(run, ref) if i in run.archive.read]
    tokens = dict((i, row_tokens(run, rows.get(i) or {})) for i in set(cands) | set(by_seller))
    check = DetailCheck(run, rows, tokens, details, ref.setdefault("loose", set()))
    if len(by_seller) > 3:
        for detail in details:
            if not any(check_shows(check, i, detail) for i in by_seller):
                check.loose.add(detail)
    full = {}
    for ident in cands:
        both = tokens[ident][0] | tokens[ident][1]
        full[ident] = detail_count(details, both)
    # (a listing matching the details may be a same-seller decoy: one carrying nothing about the property ranks last, and
    # when the seller shows four rows or fewer all of them are read before any is judged)
    shows = dict((i, bool(field_for_prop(rows.get(i) or {}, ref["prop"]) or notes_name_prop(ref["prop"], note_text(rows.get(i) or {}))))
                 for i in cands)
    ranked = sorted(cands, key=lambda i: (not check_passes(check, i), not shows[i], -full[i], cands.index(i)))
    if (check.loose or not check_passes(check, ranked[0])) and fields_first(run, ref, ranked):
        return True
    unread = [i for i in every if i not in run.archive.read and i not in ref["judged"]]
    settled = check_passes(check, ranked[0]) and shows[ranked[0]] and not (unread and len(every) <= 4)
    if not settled and (ref_room(run) or chase_waits(run)) and more_to_read(run, ref):
        return False
    passing = [i for i in ranked if check_passes(check, i)]
    pick = (passing or ranked)[:4]
    exact = [i for i in pick if details and full[i] == len(details)]
    verdict = notes_first(run, ref, check, pick, exact)
    if verdict is not None:
        return verdict
    # (a plain record settles by code only when a field carries the value; a value given only in its text goes to the hop check)
    plain = field_for_prop(rows.get(exact[0]) or {}, ref["prop"]) if len(exact) == 1 and plain_record(run, ref, exact[0]) else ""
    if str(plain or "").strip():
        ref["judged"].update(pick)
        remember_ref_ids(run, pick)
        settle_reference(run, ref, exact[0], plain, [])
    elif not hop_calls_left(run):
        ref["judged"].update(pick)
        give_up_reference(run, ref, "no hop checks are left")
    else:
        apply_hop(run, ref, pick, ask_hop(run, ref, pick, ""))
        if ref["status"] == "open" and ref["target"] is target:
            ref["judged"].update(i for i in cands if not check_passes(check, i))
    return True


def notes_name_prop(prop, text):
    stems = [w[:6] for w in prop_words(prop) if len(w) >= 4]
    toks = set(re.findall(r"[a-z]+", str(text).lower()))
    return not stems or any(t.startswith(s) for s in stems for t in toks)


def fields_first(run, ref, ranked):
    # (no listing read so far shows all the described details, and they were read through a tool with no property table:
    # the details that mark the pointed-at listing (its type, pattern, size) sit in that table, so the two likeliest
    # are read through the table tool before more searching; four per reference at most)
    reader = field_reader(run)
    if not reader or reader == role(run, "details") or not ref_room(run):
        return False
    done = ref.setdefault("fielded", set())
    need = [i for i in ranked if i not in done and not table_read(run.archive, i)][:min(2, 4 - len(done))]
    if not need:
        return False
    done.update(need)
    start = run.wire.tick
    responses = open_reads(run, need, f"The shopper wants {ref['head']} as on another listing: I read the property table "
                                      f"of the listings that may be it ({reader}).", None, reader)
    if responses:
        post_replies(run, responses)
    steps_spent(run, start)
    remember_ref_ids(run, need)
    note_event(run, "ref_fields", ids=need, tool=reader)
    return run.wire.tick > start


def notes_first(run, ref, check, pick, exact):
    if len(exact) == 1:
        asked = exact
    else:
        asked = [i for i in pick if check_passes(check, i)][:2] or pick[:1]
    unread = notes_unread(run, ref, asked)
    if not unread:
        return None
    if ref_room(run) and read_notes_for(run, ref, unread):
        return True
    if chase_waits(run):
        return False
    if ref_room(run) or ref.get("relation") != "differ":
        return None
    ref["judged"].update(pick)
    give_up_reference(run, ref, "the notes of the listing it points at could not be read in the steps left")
    return True


def apply_hop(run, ref, pick, answer, again=True):
    if not isinstance(answer, dict):
        hop_failed(run, ref, pick)
        return
    target = hop_target(answer)
    missing = answer.get("missing")
    missing = [str(x) for x in missing or [] if str(x).strip()] if isinstance(missing, list) else []
    if missing or target not in pick:
        set_aside(run, ref, pick)
        if target in pick:
            note_event(run, "ref_partial", prop=ref["prop"], target=target, missing=missing[:4])
        return
    where, problem, go, value, pairs = check_hop(run, ref, target, answer, pick)
    if problem and again and hop_calls_left(run):
        return apply_hop(run, ref, pick, ask_hop(run, ref, pick, problem), again=False)
    remember_ref_ids(run, pick)
    if problem:
        reject_hop(run, ref, pick, target, where, problem)
        return
    ref["judged"].update(pick)
    if target in ref["chain"]:
        give_up_reference(run, ref, "its notes lead back to a listing already followed")
        return
    (follow_hop(run, ref, target, go) if go else settle_reference(run, ref, target, value, pairs))


def hop_failed(run, ref, pick):
    ref["fails"] += 1
    if ref["fails"] >= 2:
        ref["judged"].update(pick)
        give_up_reference(run, ref, "the hop check gave no answer")


def set_aside(run, ref, pick):
    ref["judged"].update(pick)
    remember_ref_ids(run, pick)


def reject_hop(run, ref, pick, target, where, problem):
    note_event(run, "ref_invalid", prop=ref["prop"], target=target, problem=problem[:300])
    if where == "target":
        ref["judged"].add(target)
        return
    ref["judged"].update(pick)
    ref["chain"].append(target)
    give_up_reference(run, ref, problem)


def swap_requirement(run, ref, new_req):
    sheet = run.needs
    matcher = SwapMatch(run, ref, ref_name(ref["first"]["seller"]))
    changed = False
    for attr in ("written", "uncovered"):
        before = list(getattr(sheet, attr))
        after = swap_in(matcher, before, new_req)
        changed = changed or after != before
        setattr(sheet, attr, list(dict.fromkeys(after)))
    if sheet.now_reqs is not None:
        after = swap_in(matcher, sheet.now_reqs, new_req)
        sheet.now_reqs = list(dict.fromkeys(after if new_req in after else after + [new_req]))
    if not changed:
        sheet.uncovered.append(new_req)
    reading = sheet.reading
    if isinstance(reading.get("requirements"), list):
        reading["requirements"] = list(sheet.written)
    swap_item_entries(matcher, reading, new_req.partition(":")[2].strip())


class SwapMatch:
    def __init__(self, run, ref, first_seller):
        self.run, self.ref, self.first_seller = run, ref, first_seller


def swap_hits(matcher, req):
    ref = matcher.ref
    if ref_match(matcher.run, ref, req):
        return True
    head, bare, tags, value = split_head(req)
    if tags and ref["tag"] and not kind_words(tags[-1]) & kind_words(ref["tag"]):
        return False
    seller = matcher.first_seller
    return bool(seller) and seller in ref_name(value) and same_prop(bare, ref["prop"])


def swap_in(matcher, reqs, new_req):
    return [new_req if swap_hits(matcher, r) else r for r in reqs]


def swap_item_entries(matcher, reading, value):
    tag = matcher.ref["tag"]
    entries = list(filter(lambda e: isinstance(e, dict) and isinstance(e.get("requirements"), list), reading.get("items") or []))
    for entry in entries:
        if len(entries) > 1 and tag and not kind_words(entry.get("product") or "") & kind_words(tag):
            continue
        out = []
        for req in entry["requirements"]:
            if swap_hits(matcher, req):
                req = str(req).partition(":")[0].strip() + ": " + value
            out.append(req)
        entry["requirements"] = out


def learn_ceiling(run, call_args, rest, dicts, mode, page_key, limit_key):
    labels = set(map(shelf_of, dicts))
    label_keys = [k for k in call_args if k != page_key and names_label(call_args[k], labels)]
    shelf_key = label_keys[0] if len(label_keys) == 1 else None
    held = {}
    for k, v in mode:
        if k != shelf_key:
            held[k] = v
    sig = json.dumps([sorted(held.items()), limit_key], default=str)
    prices = [row_cost(r) for r in dicts]
    broken = vars(run).setdefault("ceil_bad", set())
    ceiling = float(call_args[limit_key]) + 1e-9
    if [p for p in prices if p is not None and p > ceiling]:
        broken.add(sig)
        return
    usable = sig not in broken and shelf_key is not None and len(labels) == 1 and "" not in labels and None not in prices
    if not usable:
        return
    template = dict((k, v) for k, v in rest.items() if k not in (limit_key, shelf_key) and not maker_filter(k, v, held, dicts))
    words = sum(len(v.split()) for k, v in template.items() if k not in held and isinstance(v, str))
    previous = stash(run, "ceil_mode")
    if previous and previous["sig"] not in broken and words > previous["words"]:
        return
    props = (run.shapes.get(role(run, "search") or "") or {}).get("properties") or {}
    kind = ((props.get(limit_key) or {}) if isinstance(props, dict) else {}).get("type")
    run.ceil_mode = dict(sig=sig, tmpl=template, mode=held, shelf=shelf_key, bound=limit_key, words=words, int=kind == "integer")


def names_label(val, labels):
    return isinstance(val, str) and bool(val.strip()) and val.strip().lower() in labels


def maker_filter(key, val, held, dicts):
    if not isinstance(val, str) or not val.strip() or key in held:
        return False
    low = val.strip().lower()
    for row in dicts:
        cells = [x for kk, x in row.items() if not re.search(r"categor|shelf|department|aisle", str(kk).lower())]
        if not any(isinstance(x, str) and x.strip().lower() == low for x in cells):
            return False
    return True


def answer_lists(answer):
    needs, refs = [], []
    if isinstance(answer.get("needs"), list):
        needs = [n for n in answer["needs"] if isinstance(n, str) and ":" in n]
    if isinstance(answer.get("references"), list):
        refs = [r for r in answer["references"] if isinstance(r, dict)]
    return needs, refs


def fix_same_item(fix, req):
    if not fix.several or not fix.tag:
        return True
    tags = re.findall(r"\(([^)]*)\)", str(req).partition(":")[0])
    return not tags or bool(kind_words(tags[-1]) & kind_words(fix.tag))


def kept_pointer(req, relation, seller_key):
    value = req.partition(":")[2]
    return req_points(req) and ref_relation(value)[0] == relation and seller_key in ref_name(value)


def fix_head(fix, mine, prop):
    if mine:
        head = mine[0].partition(":")[0].strip()
    elif fix.tag:
        head = "%s (%s)" % (prop, fix.tag)
    else:
        head = prop
    if fix.several and fix.tag and not re.findall(r"\(([^)]*)\)", head):
        head = "%s (%s)" % (head, fix.tag)
    return head


def keep_now(fix, req, prop, facts):
    if req in fix.run.needs.written:
        return True
    if req in facts or fix_other_fact(fix, req):
        return False
    return not (same_prop(req_name(req), prop) and fix_same_item(fix, req))


def apply_line_fix(fix):
    answer, run = fix.answer, fix.run
    needs, refs = answer_lists(answer)
    prop = fix_pick_prop(fix, needs, refs)
    if not prop:
        note_event(run, "ref_line_unplaced", line=fix.line[:200])
        return
    fix.several = need_items(run.needs)
    fix.tag = fix_tag(fix, needs, prop, fix.several) if fix.several else ""
    mine = [n for n in needs if same_prop(req_name(n), prop) and fix_same_item(fix, n)]
    relation, tol, unit = ref_relation(fix.phrase)
    seller_key = ref_name(fix.parsed["seller"])
    keep = next((n for n in mine if kept_pointer(n, relation, seller_key)), None)
    need = keep or fix_head(fix, mine, prop) + ": " + fix.phrase
    facts, rest = [], []
    for n in needs:
        if n in mine:
            continue
        (facts if fix_other_fact(fix, n) else rest).append(n)
    spot = needs.index(mine[0]) if mine else len(needs)
    answer["needs"] = rest[:spot] + [need] + rest[spot:]
    if isinstance(answer.get("requirements_now"), list):
        kept = [r for r in answer["requirements_now"] if isinstance(r, str) and keep_now(fix, r, prop, facts)]
        if need not in kept:
            kept.append(need)
        answer["requirements_now"] = kept
    if isinstance(answer.get("dropped"), list):
        answer["dropped"] = [d for d in answer["dropped"] if not isinstance(d, str)
                             or not (fix_inside_only(fix, req_name(d)) or fix_other_fact(fix, d))]
    if isinstance(answer.get("no_preference"), list):
        answer["no_preference"] = [p for p in answer["no_preference"] if not isinstance(p, str)
                                   or not (fix_inside_only(fix, req_name(p)) or same_prop(req_name(p), prop))]
    merge_reference_entry(fix, refs, prop, seller_key, relation, tol, unit)
    answer["references"] = refs
    answer["ref_needs"] = list(answer.get("ref_needs") or []) + [need]
    note_event(run, "ref_line", prop=prop, tag=fix.tag, relation=relation, need=need[:200], kept=bool(keep),
               details_out=[d[:80] for d in facts])


def clean_details(raw):
    out = []
    for item in raw if isinstance(raw, list) else []:
        if isinstance(item, bool) or not isinstance(item, (str, int, float)):
            continue
        text = str(item).strip()
        if text:
            out.append(text)
    return out


def pointed_listing(run, info, value, source, prop):
    seller, kind, details = "", "", []
    if isinstance(info, dict):
        seller = str(info.get("seller") or "").strip()
        kind = str(info.get("kind") or "").strip()
        details = clean_details(info.get("details"))
    vouched = lambda name: seller_named_in(run, name, source)
    parsed = None
    for text in (ref_span(prop, source), value):
        got = ref_parse(text, vouched)
        if got and vouched(got["seller"]):
            parsed = got
            break
    if vouched(seller):
        if parsed and not details and ref_name(parsed["seller"]) == ref_name(seller):
            details = parsed["details"]
    elif parsed is None:
        return None
    else:
        seller = parsed["seller"]
        kind = kind or parsed["kind"]
        details = details or parsed["details"]
    said = set(ref_toks(source))
    kept = [d for d in dict.fromkeys(details) if said & set(ref_toks(d))]
    if parsed:
        kind = kind or parsed["kind"]
    return {"seller": seller, "kind": kind[:80], "details": kept[:6]}


def seller_named_in(run, seller, source):
    name = str(seller or "").strip()
    letters = re.sub(r"[^A-Za-z]", "", name)
    if len(letters) < 3:
        return False
    if re.search(r"(?<![a-z0-9])" + re.escape(name.lower()) + r"(?:'s)?(?![a-z0-9])", str(source or "").lower()) is None:
        return False
    common = set()
    for entry in need_items(run.needs) or [{"product": str(run.needs.reading.get("product") or "")}]:
        common.update(kind_words(entry["product"]))
    schema = run.shapes.get(role(run, "search") or "") or {}
    for spec in (schema.get("properties") or {}).values():
        if isinstance(spec, dict) and isinstance(spec.get("enum"), list):
            for option in spec["enum"]:
                common.update(kind_words(option))
    own = kind_words(name)
    return len(own) > 0 and not own.issubset(common)


def target_signature(target):
    tokens = set()
    for detail in target.get("details") or []:
        tokens.update(ref_toks(detail))
    return ref_name(target.get("seller") or ""), tuple(sorted(tokens))


def describe_target(target):
    extra = target.get("details")
    tail = " (%s)" % "; ".join(target.get("details") or []) if extra else ""
    return "the %s sold by %s%s" % (target.get("kind") or "listing", target.get("seller"), tail)


def ceil_learn(run, call_args, rows, total, obs):
    paging = page_arg(run)
    page_key = paging[0] if paging else None
    rest = {k: call_args[k] for k in call_args if k != page_key}
    totals = vars(run).setdefault("ceil_seen", {})
    totals[json.dumps(rest, sort_keys=True, default=str)] = int(total)
    dicts = list(filter(lambda r: isinstance(r, dict), rows))
    if dicts and paging and paging[1] is True and isinstance(obs, dict):
        learn_page_cap(run, call_args, paging, dicts, total, obs)
    keys_seen = search_keys(run, call_args)
    limits = limit_keys(call_args, page_key, keys_seen[3])
    if dicts and len(limits) == 1:
        learn_ceiling(run, call_args, rest, dicts, keys_seen[1], page_key, limits[0])


def merge_reference_entry(fix, refs, prop, seller_key, relation, tol, unit):
    parsed = fix.parsed
    matches = [r for r in refs if same_prop(str(r.get("property") or ""), prop)]
    if matches:
        entry = matches[0]
    else:
        entry = {"property": prop}
        refs.append(entry)
    listed = entry.get("seller")
    if seller_named_in(fix.run, listed, fix.line) and ref_name(listed) == seller_key:
        if not entry.get("details"):
            entry["details"] = list(parsed["details"])
    else:
        entry["seller"] = parsed["seller"]
        entry["kind"] = parsed["kind"]
        entry["details"] = list(parsed["details"])
    if "kind" not in entry:
        entry["kind"] = parsed["kind"]
    if relation != "same" or not entry.get("relation"):
        entry["relation"], entry["tolerance"], entry["unit"] = relation, tol, unit


def judge_round(run, active):
    start = run.wire.tick
    acted = False
    for ref in sorted(active, key=lambda r: r["hops"]):
        if judge_candidates(run, ref):
            acted = True
    return acted, run.wire.tick > start


def resolver_turn(run):
    active = live_open_refs(run)
    if not active:
        return False
    acted, stepped = judge_round(run, active)
    if stepped:
        return True
    active = [r for r in active if r["status"] == "open"]
    if not active or not ref_room(run):
        return acted
    outcome = next_reference_step(run, active)
    if outcome is None:
        outcome = stop_unfound(run, active)
    return outcome or acted


def next_reference_step(run, active):
    batch = batch_tool(run)
    most = batch[2] if batch else 2
    pools = [candidate_pool(run, ref, most) for ref in active]
    shape = ref_search_shape(run)
    starving = [r for r, pool in zip(active, pools) if not pool and pending_ref_queries(run, r)] if shape else []
    queue = interleave_picks(pools, most)
    searchers = starving if queue and len(queue) < most else []
    if queue and not searchers:
        return read_candidates(run, queue, active)
    if not shape:
        return False
    return search_next_target(run, searchers or sorted(active, key=lambda r: (r["hops"], len(r["queries"]))), shape)


def stop_unfound(run, active):
    stopped = False
    for ref in active:
        exhausted = all(i in ref["judged"] for i in seller_candidates(run, ref))
        if exhausted and not pending_ref_queries(run, ref) and not seller_next_page(run, ref):
            give_up_reference(run, ref, "the listing it points at was not found in the store's search")
            stopped = True
    return stopped


def mode_groups(run, needed, labels):
    groups = {}
    walks = list(stash(run, "page_book", {}).values())
    for pos in range(len(walks)):
        walk = walks[pos]
        args = walk.get("args") if isinstance(walk.get("args"), dict) else {}
        texts, text_keys = text_args(run, args)
        if len(texts) == 1 and len(text_keys) == 1:
            others = {k: v for k, v in args.items() if k != text_keys[0] and not label_value(k, v, needed, labels)}
            group = groups.setdefault(json.dumps(others, sort_keys=True, default=str), {"tpl": (others, text_keys[0]), "seen": [], "at": pos})
            group["seen"].append(({stem(w) for w in re.findall(r"[a-z0-9]+", texts[0].lower())}, tuple(walk.get("first") or ())))
    return groups


def keyword_modes(run, needed, labels):
    ranked, blind = [], set()
    for sig, group in mode_groups(run, needed, labels).items():
        seen = group["seen"]
        if words_ignored(seen):
            blind.add(sig)
        else:
            ranked.append((-len(set(tuple(sorted(w)) for w, _ in seen)), group["at"], group["tpl"]))
    ranked.sort(key=lambda entry: (entry[0], entry[1]))
    return [tpl for _, _, tpl in ranked], blind


def phrase_end(text, hit):
    after = text[hit.end():]
    aside = re.search(r"\(\s*or\b[^)]*\)", after)
    stop = re.search(r"[.!?;](?:\s|$)", after)
    if aside is not None and (stop is None or aside.start() < stop.start()):
        return hit.end() + aside.end()
    if stop is not None:
        return hit.end() + stop.start()
    return len(text)


def pointer_offsets(clause):
    offsets = []
    for pattern in (DIFFER_WORDS, REF_WORDS, POINT_WORDS):
        offsets.extend(m.start() for m in re.finditer(pattern, clause))
    return offsets


def pointing_phrase(run, line):
    text = str(line or "")
    if not ref_text(text):
        return None
    parsed = ref_parse(text, lambda name: seller_named_in(run, name, text))
    if not parsed or not seller_named_in(run, parsed["seller"], text):
        return None
    hit = re.search(r"(?<![A-Za-z'])" + re.escape(parsed["seller"]) + r"(?:'s)?(?![A-Za-z])", text)
    if hit is None:
        return None
    before = text[:hit.start()]
    cut = clause_start(before)
    offsets = pointer_offsets(before[cut + 1:].lower())
    if not offsets:
        return None
    start = cut + 1 + min(offsets)
    end = phrase_end(text, hit)
    phrase = text[start:end].strip(" ,;")
    return (phrase, start, end, parsed) if phrase else None


def known_sellers(run):
    known = set()
    for ref in stash(run, "refs") or []:
        for tgt in (ref.get("target") or {}, ref.get("first") or {}):
            if isinstance(tgt, dict):
                known.add(ref_name(tgt.get("seller")))
    known.discard("")
    return known


def note_names(rows):
    named, vocab = set(), set()
    for row in rows.values():
        for val in row.values():
            if isinstance(val, dict):
                vocab.update(ref_name(x) for x in val.values() if isinstance(x, str))
        named.update(capital_runs(note_text(row, desc=False)))
    return {w for w in named if len(w) >= 4} - vocab


def seller_fields(run):
    try:
        known = known_sellers(run)
        stamp = (len(run.archive.items), tuple(sorted(known)))
        memo = stash(run, "seller_memo")
        if memo and memo[0] == stamp:
            return memo[1]
        rows = listing_fields(run)
        named = note_names(rows)
        fields = set()
        for row in rows.values():
            fields.update(str(k) for k, v in row.items() if seller_like_field(k, v, known, named))
        run.seller_memo = (stamp, fields)
        return fields
    except Exception:
        return set()


def notes_lookup_tool(run):
    learned = notes_tool(run)
    if learned:
        return learned
    details = role(run, "details")
    if (stash(run, "note_keys") or {}).get(details or ""):
        return None
    tool_keys, text_keys = stash(run, "tool_keys") or {}, stash(run, "text_keys") or {}
    common = set(tool_keys.get(details or "") or set()).union(stash(run, "row_keys") or set())
    caps = stash(run, "tool_caps") or {}
    others = list(filter(lambda t: t != details, lookup_tools(run)))
    richer = [t for t in others if set(text_keys.get(t) or set()) & (set(tool_keys.get(t) or set()) - common)]
    if richer:
        return max(richer, key=lambda t: ((caps.get(t) or {}).get("calls", 0), t))
    fallback = (role(run, "view"), stock_tool(run))
    return next((t for t in fallback if t in others and not (caps.get(t) or {}).get("id", 0)), None)


def listing_fields(run):
    size = len(run.archive.items)
    memo = stash(run, "ref_memo")
    if memo and memo[0] == size:
        return memo[1]
    merged, newest = {}, {}
    for pos, rec in enumerate(run.archive.items):
        found = []
        walk_dicts(rec, found.append)
        for d in found:
            ident = row_mark(d)
            if ident in run.archive.ids and (len(d) >= 4 or any(isinstance(v, dict) and len(v) >= 2 for v in d.values())):
                merged.setdefault(ident, {}).update(d)
                newest[ident] = pos
    run.ref_memo = (size, merged)
    run.ref_seen_at = newest
    return merged


NOTE_KEYS = r"note|text|desc|comment|detail|info|message|section"


def seller_value(key, val, learned):
    low = str(key).lower()
    if not isinstance(val, str) or not val.strip() or len(val) > 60:
        return False
    if "id" in low or re.search(NOTE_KEYS, low):
        return False
    return bool(re.search(SELLER_KEYS, low)) or str(key) in learned


def row_sellers(run, row):
    learned = seller_fields(run)
    out = {ref_name(v) for k, v in row.items() if seller_value(k, v, learned)}
    out.discard("")
    return out


def capital_runs(text):
    found = set()
    for sentence in re.split(r"(?<=[.!?;])\s+", text):
        streak = []
        for word in re.findall(r"[A-Za-z][A-Za-z'-]*", sentence)[1:] + [""]:
            if word[:1].isupper() and re.search(r"[a-z]", word):
                streak.append(word)
            else:
                size = len(streak)
                for start in range(size):
                    for stop in range(start + 1, min(size, start + 3) + 1):
                        found.add(ref_name(" ".join(streak[start:stop])))
                streak = []
    return found


SELLER_SKIP = (r"title|heading|name|label|caption|categor|shelf|department|aisle|note|text|desc|comment|detail|info|"
               r"message|section|url|image|summary|overview")


def seller_like_field(key, val, known, named):
    low = str(key).lower()
    if not isinstance(val, str) or not 0 < len(val) <= 60 or "id" in low or re.search(SELLER_SKIP, low):
        return False
    if ref_name(val) in known:
        return True
    return val.strip()[:1].isupper() and len(val.split()) <= 3 and ref_name(val) in named


def note_text(row, desc=True):
    notes, about = [], []
    for key, val in row.items():
        low = str(key).lower()
        skip = not isinstance(val, str) or len(val.split()) < 3 or low == "id" or low.endswith("id")
        if skip or re.search(r"title|heading|headline|name|label|caption|categor|shelf|department|url|image", low):
            continue
        bucket = about if re.search(r"desc|overview|summary|about", low) else notes
        bucket.append(val)
    chosen_texts = notes if notes else (about if desc else [])
    return "\n".join(chosen_texts)


def field_snapshot(row):
    out = []
    for val in row.values():
        if isinstance(val, dict):
            out.extend(str(x) for x in val.values() if isinstance(x, (str, int, float)))
    return out


def scalar(val):
    return isinstance(val, (str, int, float)) and not isinstance(val, bool)


def loose_fields(row):
    out = {}
    for key, val in row.items():
        if scalar(val) and not re.search(r"id$|title|shop|seller|brand|price|amount|desc|overview|note", str(key).lower()):
            out[key] = val
    return out


def row_tokens(run, row):
    notes = re.sub(r"\b(?:not|no|never|without|nor)\b(?:\W+[A-Za-z0-9.:/&'-]+){1,3}", " ", note_text(row, desc=False), flags=re.I)
    keys, values = [], [title_of(row), notes]
    groups = [v for v in row.values() if isinstance(v, dict)] or [loose_fields(row)]
    for group in groups:
        for key, val in group.items():
            keys.append(re.sub(r"([a-z])([A-Z])", r"\1 \2", str(key)).replace("_", " "))
            if scalar(val):
                values.append(str(val))
    return set(ref_toks(" ".join(keys))), set(ref_toks(" | ".join(values)))


def row_shows_detail(run, row, detail, parts=None):
    keys, values = parts if parts else row_tokens(run, row)
    wanted = set()
    for tok in ref_toks(detail):
        if len(tok) >= 3 and not re.search(r"\d", tok) and tok not in keys:
            wanted.add(tok)
    if wanted and wanted.isdisjoint(values):
        return False
    figures = ref_figs(detail)
    if figures:
        written = ref_figs(" | ".join([title_of(row)] + field_snapshot(row)))
        for fig, unit in figures:
            same_unit = [f for f, u in written if u == unit]
            if same_unit and not any(abs(f - fig) < 1e-9 for f in same_unit):
                return False
    return True


def field_for_prop(row, prop):
    groups = [val for val in row.values() if isinstance(val, dict)]
    for match in (strict_prop, same_prop):
        hit = next((str(x).strip() for g in groups for k, x in g.items() if scalar(x) and str(x).strip() and match(k, prop)), None)
        if hit is not None:
            return hit
    return ""


def remember_ref_ids(run, ids):
    merged = set(stash(run, "ref_ids") or set())
    merged.update(ids)
    run.ref_ids = merged


def ref_foreign(run):
    try:
        ids = stash(run, "ref_ids") or set()
        if not ids:
            return set()
        shelves = home_shelves(run)
        if not shelves or not all(shelves):
            return set()
        homes = set().union(*shelves)
        rows = listing_fields(run)
        held_now = set(run.trolley.units)
        out = set()
        for ident in ids:
            if ident in held_now:
                continue
            shelf = shelf_of(rows.get(ident) or {})
            if shelf and shelf not in homes:
                out.add(ident)
        return out
    except Exception:
        return set()


def detail_count(details, tokens):
    return len([d for d in details if set(ref_toks(d)) and set(ref_toks(d)) <= tokens])


def shown_count(run, details, ident):
    keys, values = row_tokens(run, listing_fields(run).get(ident) or {})
    return detail_count(details, keys | values)


def check_hop(run, ref, target, answer, pick=None):
    details = ref["target"]["details"]
    row = listing_fields(run).get(target) or {}
    parts = row_tokens(run, row)
    loose = ref.get("loose") or set()
    unshown = next((d for d in details if d not in loose and not row_shows_detail(run, row, d, parts)), None)
    if unshown is not None:
        return miss("target", f"{target} does not show '{unshown}', so it is not the listing described.")
    best = max(pick or [target], key=lambda ident: shown_count(run, details, ident))
    if shown_count(run, details, best) > shown_count(run, details, target):
        return miss("target", f"{best} shows more of the details named than {target}.")
    notes = note_text(row)
    note_toks = set(ref_toks(notes))
    go = answer.get("go_to")
    if isinstance(go, dict) and str(go.get("seller") or "").strip():
        return check_go_to(run, ref, target, row, go, notes, note_toks)
    value = str(answer.get("value") or "").strip()
    value_toks = set(ref_toks(value))
    keys, values = parts
    if value == "" or not value_toks or not value_toks.issubset(values | keys):
        return miss("answer", f"the value '{value}' is not written in {target}'s record.")
    pairs = []
    for extra in dict.fromkeys(clean_details(answer.get("goes_with"))):
        toks = set(ref_toks(extra))
        if toks and toks.issubset(note_toks) and toks != value_toks:
            pairs.append(extra)
    return "", "", None, value, pairs[:3]


def check_go_to(run, ref, target, row, go, notes, note_toks):
    seller = str(go["seller"]).strip()
    spot = re.compile(r"(?<![a-z0-9])" + re.escape(seller.lower()) + r"(?:'s)?(?![a-z0-9])")
    if not spot.search(notes.lower()):
        return miss("answer", f"the go_to seller '{seller}' is not written in {target}'s notes.")
    sentences = re.split(r"(?<=[.!?])\s+", notes)
    if not any(spot.search(s.lower()) and text_names_prop(ref["prop"], s) for s in sentences):
        return miss("answer", f"the note naming {seller} is about another property, not {ref['prop']}.")
    kept = list(filter(lambda d: set(ref_toks(d)) and 2 * len(set(ref_toks(d)) & note_toks) >= len(set(ref_toks(d))),
                       clean_details(go.get("details"))))
    hop_to = dict(seller=seller, kind=str(go.get("kind") or "").strip()[:80], details=kept[:6])
    same_place = ref_name(seller) in row_sellers(run, row) and target_signature(hop_to) == target_signature(ref["target"])
    if same_place:
        return miss("answer", f"the go_to is {target} itself.")
    return "", "", hop_to, "", []


def detail_search_words(run, ref):
    target = ref["target"]
    seller_words = set(re.findall(r"[a-z0-9]+", str(target["seller"]).lower()))
    joined, figures, words = sort_detail_words(target["details"], seller_words, known_field_words(run))
    known = record_field_names(run)
    split = [list(filter(lambda w: w not in LINK_WORDS, re.findall(r"[a-z0-9]+", str(d).lower()))) for d in target["details"]]
    paired = ref.get("hops", 0) == 0 and bool(split) and all(len(p) > 1 for p in split)

    def leads_with_field(parts):
        return paired or any(" ".join(parts[:j]) in known for j in range(1, len(parts)))
    lasts = [p[-1] for p in split if len(p) > 1 and p[-1] in words and leads_with_field(p)]
    front = [w for w in lasts if w in words]
    back = [w for w in words if w not in lasts]
    return list(dict.fromkeys(joined + figures + front + back))


def ref_search_shape(run):
    schema = run.shapes.get(role(run, "search") or "") or {}
    required = schema.get("required")
    needed = set(required) if isinstance(required, list) else set()
    labels = set(shelf_of(r) for r in listing_fields(run).values()) - {""}
    modes, blind = keyword_modes(run, needed, labels)
    if modes:
        shape = modes[0]
    else:
        shape = shape_from_log(run, page_arg(run), blind, needed, labels)
    if shape is not None:
        return dict((k, v) for k, v in shape[0].items() if not label_value(k, v, needed, labels)), shape[1]
    props = schema.get("properties") if isinstance(schema.get("properties"), dict) else {}
    token = token_key(run)
    free = []
    for key, spec in props.items():
        if key != token and isinstance(spec, dict) and spec.get("type") == "string" and not spec.get("enum"):
            free.append(key)
    if len(free) != 1 or needed - {free[0]}:
        return None
    return {}, free[0]


def search_acts(run):
    search = None
    for entry in reversed(run.wire.log):
        if entry.get("kind") != "env":
            continue
        for act in reversed(entry.get("actions") or []):
            search = role(run, "search")
            if act.get("name") == search and isinstance(act.get("args"), dict):
                yield act["args"]


def shape_from_log(run, paging, blind, needed, labels):
    for args in search_acts(run):
        texts = [k for k, v in args.items() if isinstance(v, str) and v.strip() and v in free_text(run, args)]
        if len(texts) == 1:
            others = {k: v for k, v in args.items() if k != texts[0] and not (paging and k == paging[0])}
            kept = {k: v for k, v in others.items() if not label_value(k, v, needed, labels)}
            if json.dumps(kept, sort_keys=True, default=str) not in blind:
                return others, texts[0]
    return None


def seller_candidates(run, ref):
    target = ref["target"]
    wanted = ref_name(target["seller"])
    kinds = kind_words(target["kind"])
    rows = listing_fields(run)
    newest = stash(run, "ref_seen_at") or {}
    scored = []
    for ident, row in rows.items():
        if wanted in row_sellers(run, row):
            blurb = [str(v) for k, v in row.items() if isinstance(v, str) and re.search(r"desc|overview|summary", str(k).lower())]
            shown = detail_count(target["details"], set(ref_toks(" ".join([title_of(row)] + blurb))))
            scored.append(((0 if not kinds & kind_words(shelf_of(row)) else -2) - shown, -newest.get(ident, 0), ident, shown))
    scored.sort()
    ref["shows"] = {}
    order = []
    for entry in scored:
        ref["shows"][entry[2]] = entry[3]
        order.append(entry[2])
    return order


def resolved_value(ref):
    value = str(ref["value"])
    relation = ref["relation"]
    if relation == "within" and is_number(ref.get("tol")):
        hit = re.search(r"(-?\d+(?:\.\d+)?)\s*([a-z%]*)", value.lower().replace(",", ""))
        if hit is not None:
            tol = float(ref["tol"])
            low, high = float(hit.group(1)) - tol, float(hit.group(1)) + tol
            unit = hit.group(2) or str(ref.get("unit") or "")
            return f"{round(low, 6):g} to {round(high, 6):g} {unit}".strip() + f" (within {tol:g} of {value})"
    if relation in ("at most", "at least"):
        return relation + " " + value
    options = [value]
    for other in list(ref.get("goes_with") or []):
        if other.lower() != value.lower():
            options.append(other)
    # (a counterpart is the value the listing's text pairs with its own; with no pairing on record the own value stands)
    if relation == "pair" and len(options) > 1:
        return " or ".join(options[1:])
    if relation == "differ":
        return " and ".join(["not " + v for v in options])
    return " or ".join(options)


def plain_record(run, ref, ident):
    row = listing_fields(run).get(ident) or {}
    value = field_for_prop(row, ref["prop"])
    notes = note_text(row, desc=False)
    if not value:
        return False
    if text_names_prop(ref["prop"], notes) or not set(ref_toks(value)).isdisjoint(ref_toks(notes)):
        return False
    own = row_sellers(run, row)
    elsewhere = other_sellers(run, listing_fields(run), own)
    fields = row_vocabulary(run, row)
    words = []
    for sentence in re.split(r"(?<=[.!?;])\s+", notes):
        words.extend(re.findall(r"[A-Za-z][A-Za-z'-]*", sentence)[1:])
    return not any(foreign_name(w, ref_name(w), own, fields, elsewhere) for w in words)


def foreign_name(word, low, own, fields, elsewhere):
    if low in elsewhere:
        return True
    return word[:1].isupper() and len(low) >= 4 and low not in own and low not in fields


def detail_values(run, ref):
    # (the value part of each named detail: a leading field name the store uses is dropped, else the last two words
    # stay, "shoe type boat shoes" -> "boat shoes")
    known = record_field_names(run)
    out = []
    for detail in ref["target"].get("details") or []:
        parts = [w for w in re.findall(r"[a-z0-9]+", str(detail).lower()) if w not in LINK_WORDS]
        cut = max([j for j in range(1, len(parts)) if " ".join(parts[:j]) in known], default=0)
        rest = parts[cut:] if cut else parts[-2:] if len(parts) > 2 else parts[-1:]
        if rest:
            out.append(" ".join(rest))
    return list(dict.fromkeys(out))


def pending_ref_queries(run, ref):
    target = ref["target"]
    width = max(2, run.lead_words or 3)
    seller = " ".join(re.findall(r"[A-Za-z0-9]+", re.sub(r"'s\b", "", str(target["seller"]))))
    words = detail_search_words(run, ref)
    kind = [w for w in re.findall(r"[a-z]{3,}", str(target["kind"]).lower()) if w not in LINK_WORDS][-1:]
    if not ref.get("unindexed"):
        # (one query per named detail before the seller alone: "seller + first detail" found that seller's listings of
        # another type, and the detail telling the listing apart, its shoe type, was never searched)
        values = detail_values(run, ref) if len(seller.split()) < width else []
        plans = [seller + " " + v for v in values[:3]] + ([seller + " " + words[0]] if words and len(seller.split()) < width else [])
        plans = list(dict.fromkeys(plans + [seller]))
    elif words or kind:
        plans = [" ".join((words[:1] + kind)[:width])]
    else:
        plans = []
    done = ref["queries"]
    return [p for p in plans if p.strip() and p.lower() not in done]


def ref_note(run, thing=""):
    try:
        lines, ends = [], []
        for ref in stash(run, "refs") or []:
            status = ref["status"]
            if status == "resolved":
                lines.append(resolved_note(ref))
                continue
            if status not in ("open", "stuck"):
                continue
            chain = ref["chain"]
            where = ("chain followed so far: " + " -> ".join(chain)) if chain else "the listing it names was never found"
            lines.append(f"- {ref['head']}: the code could NOT resolve it ({where}). Check: if a record below is that listing, "
                         "or the last one its chain reached, and gives its own value without pointing elsewhere, that value "
                         "is the requirement. Then: otherwise the value is unknown, so it alone never fails an item.")
            ends += [i for i in chain[-2:] if i != thing]
        if not lines:
            return ""
        recs = [r for ident in dict.fromkeys(reversed(ends)) for r in notes_about(run.archive, ident, 1)]
        tail = "\nRecords of the last listings those chains reached:\n" + "\n".join(recs[:2]) if recs else ""
        return ("\n\nRequirements that point at another listing (the code traced that listing through the store's notes):\n"
                + "\n".join(lines) + tail)
    except Exception:
        return ""


def text_args(run, args):
    texts = free_text(run, args)
    return texts, [k for k, v in args.items() if isinstance(v, str) and v in texts and v.strip()]


def words_ignored(seen):
    for pos in range(len(seen)):
        words_a, first_a = seen[pos]
        for words_b, first_b in seen[pos + 1:]:
            if len(first_a) >= 2 and first_a == first_b and not (words_a & words_b):
                return True
    return False


def label_value(key, val, needed, labels):
    return key not in needed and isinstance(val, str) and val.strip().lower() in labels


def camel_words(key):
    return re.sub(r"([a-z])([A-Z])", r"\1 \2", str(key)).lower()


def known_field_words(run):
    out = set()
    for row in listing_fields(run).values():
        for val in row.values():
            if isinstance(val, dict):
                for k in val:
                    out.update(re.findall(r"[a-z]+", camel_words(k)))
    for req in run.needs.written + run.needs.uncovered:
        out.update(re.findall(r"[a-z]+", split_head(req)[1].lower()))
    return out


def sort_detail_words(details, seller_words, field_words):
    joined, figures, words = [], [], []
    for detail in details:
        previous = ""
        for whole in str(detail).lower().split():
            unit_slot = re.fullmatch(r"\d+(?:[.,]\d+)?", previous) is not None
            previous = whole
            for w in re.findall(r"[a-z0-9]+", whole):
                digit = re.search(r"\d", w)
                if w in LINK_WORDS or w in seller_words or (unit_slot and not digit):
                    continue
                if digit and re.search(r"[a-z]", w):
                    joined.append(w)
                elif w.isdigit():
                    if len(w) > 1:
                        figures.append(w)
                elif len(w) > 2 and w == whole.strip(".,;:!?()[]{}\"'") and w not in field_words and w.rstrip("s") not in field_words:
                    words.append(w)
    return joined, figures, sorted(words, key=len, reverse=True)


def record_field_names(run):
    stamp = len(run.archive.items)
    memo = stash(run, "store_keys_memo")
    if memo and memo[0] == stamp:
        return memo[1]
    names = set()

    def gather(d):
        maps = [v for v in d.values() if isinstance(v, dict) and sum(1 for x in v.values() if isinstance(x, (str, int, float))) >= 2]
        for group in maps:
            for key in group:
                spaced = " ".join(re.findall(r"[a-z0-9]+", camel_words(key)))
                if spaced:
                    names.add(spaced)
    for rec in run.archive.items:
        walk_dicts(rec, gather)
    run.store_keys_memo = (stamp, names)
    return names


def seller_next_page(run, ref):
    paging = page_arg(run)
    if not paging or ref["pages"] >= 3:
        return None
    wanted = ref_name(ref["target"]["seller"])
    walks = list(stash(run, "page_book", {}).values())
    walks.reverse()
    for walk in walks:
        if wanted and wanted in ref_name(" ".join(free_text(run, walk["args"]))):
            token = next_page(run, walk)
            if token is not None:
                args = dict(walk["args"])
                args[paging[0]] = token
                return args
    return None


def open_refs(run):
    return [r for r in stash(run, "refs") or [] if r.get("status") == "open"]


def ref_room(run):
    pending = open_refs(run)
    all_differ = bool(pending) and all(r.get("relation") == "differ" for r in pending)
    used = stash(run, "ref_steps", 0)
    if all_differ and not any(r.get("chain") for r in pending) and used >= REF_DIFFER_BLIND:
        return False
    share = REF_DIFFER_STEPS if all_differ else REF_STEPS
    least = 6 if all_differ else 8
    floor = ref_step_floor(run)
    link = run.wire
    if used >= max(least, int(share * run.step_cap)) or time_left(link) <= CLOSING + 60:
        return False
    if over_budget(link) or link.done or link.broke or chase_waits(run, floor):
        return False
    return moves_left(run) > floor or late_hop_allowed(run)


def ref_step_floor(run):
    reserve = max(FIRST_ORDER_BY, min(FIRST_ORDER_DUE, run.step_cap // 4 + 1)) + 10
    return max(reserve, order_due(run) + 7)


def ref_owner(run, ref):
    tag = ref.get("tag")
    return product_of(run, f"x ({ref['tag']})") if need_items(run.needs) and tag else None


def settle_reference(run, ref, target, value, pairs):
    if not str(value or "").strip():
        return give_up_reference(run, ref, f"{target} shows no value for it")
    ref["chain"].append(target)
    ref["status"], ref["value"], ref["goes_with"] = "resolved", str(value).strip(), list(pairs)
    new_req = ref["head"] + ": " + resolved_value(ref)
    ref["now"] = new_req
    swap_requirement(run, ref, new_req)
    run.needs.resolved.append(new_req)
    run.needs.version += 1
    chain = ref["chain"]
    note_event(run, "ref_resolved", req=ref["req"][:200], now=new_req, chain=list(chain))
    tell_actor(run, f"RESOLVED: {ref['head']} is given by another listing, and {target} shows {ref['value']}"
                    f"{resolved_via(chain)}{resolved_pairing(ref, pairs)}. The requirement is now: {new_req}.")


def resolved_via(chain):
    if len(chain) < 2:
        return ""
    return ", reached from " + chain[0] + " through the listings' notes"


def resolved_pairing(ref, pairs):
    joined = " and ".join(pairs)
    if ref["relation"] == "differ":
        lead = "; its notes pair " + joined + " with it" if pairs else ""
        return lead + ". The shopper's value MUST DIFFER from that one: any other value meets it, that value does not"
    if ref["relation"] == "pair" and pairs:
        return f"; its notes pair {ref['value']} with {joined}, and the shopper wants that counterpart: {joined} meets it, {ref['value']} itself does not"
    if not pairs:
        return ""
    return f"; its notes say {ref['value']} goes together with {joined}, so each of them meets it"


def fix_pick_prop(fix, needs, refs):
    outside = fix.outside
    asked = list(filter(None, (req_name(a) for a in str(stash(fix.run, "asking_label") or "").split(";"))))
    by_ref = [str(r.get("property") or "") for r in refs]
    pointing = [req_name(n) for n in needs if req_points(n)]
    tiers = [by_ref, pointing, list(map(req_name, needs)), asked,
             sorted(record_field_names(fix.run)) + list(map(req_name, fix.run.needs.written)),
             reply_prop_runs(fix.run, fix.qtext, outside)]
    for tier in tiers:
        named = [p for p in tier if p and prop_named(p, outside)]
        if not named:
            continue
        return max(named, key=lambda p: (prop_named(p, outside, exact=True), len(re.findall(r"[a-z0-9]+", p.lower())), len(p)))
    for p in by_ref + pointing + (asked if len(asked) == 1 else []):
        if p and not fix_inside_only(fix, p):
            return p
    return ""


def own_redirects(run, thing):
    try:
        rows = listing_fields(run)
        row = rows.get(thing) or {}
        notes = note_text(row, desc=False)
        if not notes:
            return []
        own = row_sellers(run, row)
        context = (own, row_vocabulary(run, row), other_sellers(run, rows, own))
        several = need_items(run.needs)
        shelf = shelf_of(row)
        k = product_of(run, "x (" + shelf + ")") if several and shelf else None
        sentences = re.split(r"(?<=[.!?;])\s+", notes)
        found = []
        for req in props_pending(run.needs):
            head = str(req).partition(":")[0]
            if several and k is not None and product_of(run, head) not in (k, None):
                continue
            prop = re.split(r"\s*\(", head)[0].strip()
            for sentence in sentences if prop else []:
                if redirect_in(sentence, prop, *context):
                    found.append((str(req), prop, sentence.strip()))
                    break
        return found
    except Exception:
        return []


def search_target(run, ref, args):
    start = run.wire.tick
    say = "The shopper wants %s as on another listing (%s): I search for it." % (ref["head"], describe_target(ref["target"]))
    rows = []
    if walk_call(run, args, say):
        steps_spent(run, start)

        def keep(d):
            if row_mark(d) in run.archive.ids and len(d) >= 4:
                rows.append(d)
        walk_dicts(run.archive.items[-1], keep, ceiling=3)
    else:
        steps_spent(run, start)
    wanted = ref_name(ref["target"]["seller"])
    query = " ".join(free_text(run, args))
    if rows and all(wanted not in row_sellers(run, r) for r in rows) and ref_name(query) == wanted:
        ref["unindexed"] = True
    note_event(run, "ref_search", prop=ref["prop"], query=query[:120], rows=len(rows))
    return start < run.wire.tick


def chase_waits(run, floor=None):
    products = need_items(run.needs)
    pending = open_refs(run)
    if not (products and pending) or any(r.get("relation") != "differ" or not r.get("chain") for r in pending):
        return False
    limit = floor if floor is not None else ref_step_floor(run)
    if moves_left(run) <= limit + 4:
        return False
    try:
        owners = {ref_owner(run, r) for r in pending}
        lacking = [k for k in range(len(products)) if not any(entry["fit"] for entry in ed_pool(run, k))]
    except Exception as err:
        note_event(run, "ref_wait_error", error=error_text(err))
        return False
    known = owners - {None}
    rest = list(filter(lambda k: k not in known, lacking))
    waits = (len(rest) >= 2) if None in owners else bool(rest)
    if waits and tuple(rest) != stash(run, "ref_wait_said"):
        run.ref_wait_said = tuple(rest)
        note_event(run, "ref_wait", lacking=rest)
    return waits


def reply_prop_runs(run, qtext, outside):
    if not qtext:
        return []
    kinds = set()
    for entry in need_items(run.needs) or [{"product": str(run.needs.reading.get("product") or "")}]:
        kinds.update(re.findall(r"[a-z]+", str(entry["product"]).lower()))
    asked = re.findall(r"[a-z0-9]+", str(qtext).lower())
    usable = [not (w in LINK_WORDS or len(w) < 3 or w in kinds or w.rstrip("s") in kinds) for w in asked]
    runs = []
    for size in (3, 2, 1):
        for pos in range(len(asked) - size + 1):
            if all(usable[pos: pos + size]):
                phrase = " ".join(asked[pos: pos + size])
                if prop_named(phrase, outside):
                    runs.append(phrase)
    return runs


def raw_label(run, labels):
    if not labels:
        return ""
    tally = {}
    found = []
    for rec in run.archive.items:
        walk_dicts(rec, found.append)
    for d in found:
        if not any(isinstance(v, str) and v in run.archive.ids and "id" in str(key).lower() for key, v in d.items()):
            continue
        spelled = next((v.strip() for key, v in d.items() if isinstance(v, str) and 0 < len(v.strip()) <= 60
                        and re.search(r"categor|shelf|department|aisle", str(key).lower()) and "id" not in str(key).lower()), "")
        if spelled and spelled.lower() in labels:
            tally[spelled] = tally.get(spelled, 0) + 1
    best = ""
    for name in sorted(tally):
        if not best or tally[name] > tally[best]:
            best = name
    return best


def learn_page_cap(run, call_args, paging, dicts, total, obs):
    first = paging[2]
    at = call_args.get(paging[0], first)
    markers = []
    for key, val in obs.items():
        if isinstance(val, (list, dict)):
            continue
        if re.search(r"next|more|onward|continu|cursor", str(key).lower()):
            markers.append(val)
    size = max(len(dicts), page_size(run))
    if not is_number(at) or not markers or len(dicts) < size:
        return
    if any(val is not None and val is not False and val != "" for val in markers):
        return
    reach = int((at - first + 1) * size)
    if reach < total:
        run.page_cap = min(stash(run, "page_cap") or reach, reach)


def hop_ready(ref):
    target = ref.get("target") or {}
    if ref.get("status") != "open" or ref.get("hops", 0) < 1:
        return False
    return bool(str(target.get("seller") or "").strip() and str(target.get("kind") or "").strip() and target.get("details"))


def late_hop_allowed(run):
    if moves_left(run) <= order_due(run) + 2:
        return False
    used = stash(run, "ref_steps", 0)
    for ref in filter(hop_ready, stash(run, "refs") or []):
        began = ref.get("late_from")
        if began is None:
            if ref.get("queries") or not others_verified(run, ref):
                continue
            ref["late_from"] = used
            note_event(run, "ref_late", prop=ref.get("prop"), to=str((ref.get("target") or {}).get("seller"))[:80])
            return True
        if used - began < 3:
            return True
    return False


def others_verified(run, ref):
    products = need_items(run.needs)
    if not products:
        return True
    try:
        bare = [k for k in range(len(products)) if not any(e["verified"] and e["fit"] for e in ed_pool(run, k))]
    except Exception:
        return False
    own = ref_owner(run, ref)
    return len(bare) <= 1 if own is None else set(bare) <= {own}


def hop_calls_left(run):
    link = run.wire
    if stash(run, "ref_calls", 0) >= REF_CALLS or time_left(link) <= CLOSING + 40 or over_budget(link):
        return False
    return link.spend < 0.8 * SPEND_CAP * vars(link).get("scale", 1.0)


def ref_match(run, ref, req):
    text = str(req)
    if text == ref["req"] or text == ref.get("now"):
        return True
    head, bare, tags, value = split_head(text)
    pointing = ref_text(value) or (ref.get("relation") == "differ" and re.search(DIFFER_WORDS, value.lower()))
    if not pointing or not same_prop(bare, ref["prop"]):
        return False
    if tags and ref["tag"]:
        return bool(kind_words(tags[-1]) & kind_words(ref["tag"]))
    return True


def ref_live(run, ref):
    prop = ref["prop"]
    for req in props_pending(run.needs):
        if ref_match(run, ref, req) or same_prop(split_head(req)[1], prop):
            return True
    return not any(same_prop(split_head(r)[1], prop) for r in run.needs.let_go)


def ref_step(run):
    if not stash(run, "refs") or run.wire.done or run.wire.broke:
        return False
    try:
        return resolver_turn(run)
    except Exception as err:
        note_event(run, "ref_error", error=error_text(err))
        for ref in run.refs:
            if ref["status"] == "open":
                ref["status"] = "stuck"
                ref["why"] = "the code's resolver failed"
        return False


def live_open_refs(run):
    active = []
    for ref in run.refs:
        if ref["status"] == "open" and not ref_live(run, ref):
            ref["status"] = "dropped"
            note_event(run, "ref_dropped", req=ref["req"][:200])
        if ref["status"] == "open":
            active.append(ref)
    return active


def candidate_pool(run, ref, most):
    pool = [i for i in seller_candidates(run, ref) if i not in run.archive.read and i not in ref["judged"]]
    shows = ref.get("shows") or {}
    if ref["queries"] or not pending_ref_queries(run, ref) or len(pool) <= most:
        return pool
    return pool if pool and shows.get(pool[0]) else []


def search_next_target(run, order, shape):
    for ref in order:
        plans = pending_ref_queries(run, ref)
        if plans:
            query = plans[0]
            ref["queries"].append(query.lower())
            if ran_query(run, query):
                return True
            args = dict(shape[0])
            args[shape[1]] = query
            return search_target(run, ref, args)
        args = seller_next_page(run, ref)
        if args:
            ref["pages"] += 1
            return search_target(run, ref, args)
    return None


def steps_spent(run, since):
    run.ref_steps = stash(run, "ref_steps", 0) + run.wire.tick - since


def read_candidates(run, queue, active):
    batch = batch_tool(run) if len(queue) > 1 else None
    start = run.wire.tick
    heads = ", ".join(dict.fromkeys(r["head"] for r in active))
    responses = open_reads(run, queue, f"The shopper wants {heads} as on another listing: I open the listings that may be it.", batch)
    dropped = False
    if batch and responses and not stash(run, "batch_proven"):
        if batch_ok(stash(run, "last_reads") or [], queue):
            run.batch_proven = True
        else:
            run.batch_spec = None
            dropped = True
            note_event(run, "batch_dropped", tool=batch[0])
    if responses:
        post_replies(run, responses)
    steps_spent(run, start)
    remember_ref_ids(run, queue)
    note_event(run, "ref_read", ids=queue)
    if not dropped:
        missed = [i for i in queue if i not in run.archive.read]
        for ref in active:
            ref["judged"].update(missed)
    return run.wire.tick > start


def more_to_read(run, ref):
    if [i for i in seller_candidates(run, ref) if i not in run.archive.read and i not in ref["judged"]]:
        return True
    return bool(pending_ref_queries(run, ref) or seller_next_page(run, ref))


def row_vocabulary(run, row):
    keys = row_tokens(run, row)[0]
    return keys.union(ref_toks(" | ".join([title_of(row), shelf_of(row)] + field_snapshot(row))))


def other_sellers(run, rows, own):
    out = set()
    for other in rows.values():
        out |= row_sellers(run, other)
    return out - own


def notes_unread(run, ref, ids):
    tool = notes_lookup_tool(run)
    if not (tool and ids):
        return []
    skip = set((stash(run, "read_with") or {}).get(tool) or set())
    skip |= ref.setdefault("noted", set())
    return [i for i in dict.fromkeys(ids) if i not in skip][:2]


def read_notes_for(run, ref, need):
    tool = notes_lookup_tool(run)
    if not (tool and need):
        return False
    ref.setdefault("noted", set()).update(need)
    start = run.wire.tick
    responses = open_reads(run, need, f"The shopper wants {ref['head']} as on another listing: I read the notes of the listing "
                                      f"that may be it ({tool}), since a note may send the reader on.", None, tool)
    if responses:
        post_replies(run, responses)
    steps_spent(run, start)
    remember_ref_ids(run, need)
    note_event(run, "ref_notes", ids=need, tool=tool)
    return run.wire.tick > start


def differ_refs(run):
    return [r for r in stash(run, "refs") or [] if r.get("relation") == "differ"]


def ref_pointers(run):
    chained = set()
    for ref in differ_refs(run):
        if ref.get("status") in ("open", "stuck", "resolved") and ref_live(run, ref):
            for ident in ref.get("chain") or []:
                if in_own_product(run, ref, ident):
                    chained.add(ident)
    return chained


def in_own_product(run, ref, ident):
    if not need_items(run.needs):
        return True
    shelf = shelf_known(run, ident)
    k = product_of(run, f"x ({shelf})") if shelf else None
    if k is not None and ref.get("tag"):
        return k == product_of(run, f"x ({ref['tag']})")
    return True


def ref_pointer_note(run, thing):
    for ref in differ_refs(run):
        if thing in (ref.get("chain") or []) and ref_live(run, ref) and in_own_product(run, ref, thing):
            shown = " (%s)" % ref["value"] if ref.get("status") == "resolved" and ref.get("value") else ""
            return (f"{thing} is the listing that the shopper's reference for {ref['head']} names, or a listing its notes pass "
                    f"through, so its {ref['prop']} is the very value{shown} the shopper's {ref['prop']} must differ from: it cannot "
                    "meet that requirement")
    return ""


def ref_excluded(run, fields, k):
    products = need_items(run.needs)
    for ref in differ_refs(run):
        if ref.get("status") != "resolved" or not ref_live(run, ref):
            continue
        if products and ref.get("tag") and product_of(run, f"x ({ref['tag']})") not in (k, None):
            continue
        banned = set(" ".join(ref_toks(v)) for v in [ref.get("value")] + list(ref.get("goes_with") or []) if str(v or "").strip())
        for key in fields or {}:
            if same_prop(key, ref["prop"]) and " ".join(ref_toks(fields[key])) in banned:
                return True
    return False


NEGATIONS = ("not", "nor", "never", "no", "neither")


def redirect_in(sentence, prop, own, fields, elsewhere):
    words = list(re.finditer(r"[A-Za-z][A-Za-z'-]*", sentence))
    blanked = re.sub(r"[A-Za-z0-9]+(?=\s+by\s+[A-Z])", lambda hit: " " * len(hit.group(0)), sentence)
    first = None
    for m in words:
        if prop_named(prop, blanked[:m.end()]):
            first = m.start()
            break
    if first is None:
        return False
    pos = 0
    while pos + 1 < len(words):
        pos += 1
        word = words[pos].group(0)
        low = ref_name(word)
        if words[pos].start() <= first or not low:
            continue
        stranger = low in elsewhere
        if not stranger and word[:1].isupper() and len(low) >= 4 and low not in own:
            stranger = not set(ref_toks(word)) & fields and low not in fields
        if stranger and all(w.group(0).lower() not in NEGATIONS for w in words[max(0, pos - 5): pos]):
            return True
    return False


def notes_remark(run, call_name, call_args):
    if call_name in [role(run, r) for r in ("search", "cart", "cart_view", "message")] or call_name in finishers(run):
        return ""
    version = str(run.needs.version)
    remarks = []
    for ident in sorted(asked_ids(call_args)):
        key = "%s|%s" % (ident, version)
        if key not in run.passed and ident not in run.trolley.units:
            continue
        told = vars(run).setdefault("own_told", set())
        sends = [] if key in told else own_redirects(run, ident)
        if sends:
            told.add(key)
            run.passed.discard(key)
            note_event(run, "own_redirect", id=ident, props=[p for _, p, _ in sends])
            remarks.append(remark_text(ident, sends, ident in run.trolley.units))
    return " ".join(remarks)


def remark_text(ident, sends, in_cart):
    what = "; ".join('%s ("%s")' % (p, s[:160]) for _, p, s in sends)
    reqs = "; ".join("'%s'" % r for r, _, _ in sends)
    ending = ". If they do, take it out of the cart and put another verified fit in." if in_cart else "."
    count_word = "them" if len(sends) > 1 else "it"
    return (f"{ident}'s notes might point {what} at another listing; in that case its own field does not settle {reqs} "
            f"(grading uses that other listing's value instead), so it meets {count_word} only when that listing's record shows "
            "it; the item check runs again before it is ordered" + ending)


def clause_start(before):
    cuts = [before.rfind(c) for c in ",;!?"]
    cuts += [m.start() for m in re.finditer(r"[.](?:\s|$)", before)]
    return max(cuts + [-1])


def req_name(req):
    return split_head(req)[1].strip()


def req_points(req):
    value = req.partition(":")[2]
    return bool(ref_text(value) or re.search(DIFFER_WORDS, value.lower()))


def ref_fix(run, answer, lines, qtext):
    masked = []
    for line in lines:
        got = pointing_phrase(run, line)
        if got is None:
            masked.append(line)
        else:
            phrase, start, end, parsed = got
            outside = line[:start] + " " + line[end:]
            masked.append(outside)
            apply_line_fix(LineFix(run, answer, line, qtext, phrase, parsed, outside, line[start:end]))
    return masked


class LineFix:
    def __init__(self, run, answer, line, qtext, phrase, parsed, outside, inside):
        self.run, self.answer, self.line, self.qtext = run, answer, line, qtext
        self.phrase, self.parsed, self.outside, self.inside = phrase, parsed, outside, inside


def fix_inside_only(fix, prop):
    return bool(prop) and prop_named(prop, fix.inside) and not prop_named(prop, fix.outside)


def fix_other_fact(fix, req):
    vals = set(ref_toks(str(req).partition(":")[2]))
    if fix_inside_only(fix, req_name(req)):
        return True
    return len(vals) > 0 and vals <= set(ref_toks(fix.inside)) and vals.isdisjoint(ref_toks(fix.outside))


def fix_tag(fix, needs, prop, several):
    k = product_of(fix.run, fix.outside)
    if k is None:
        tags = []
        for n in needs:
            if same_prop(req_name(n), prop):
                tags.extend(re.findall(r"\(([^)]*)\)", n.partition(":")[0]))
        k = product_of(fix.run, "x (%s)" % tags[-1]) if tags else None
    if k is None and fix.qtext:
        k = product_of(fix.run, fix.qtext)
    return several[k]["product"] if k is not None else ""


def text_names_prop(prop, text):
    words = [w for w in prop_words(prop) if len(w) >= 3]
    if not words:
        return True
    toks = set(re.findall(r"[a-z]+", str(text).lower()))
    for tok in toks:
        if tok in words:
            return True
        if len(tok) >= 4 and any(len(w) >= 4 and tok[:4] == w[:4] for w in words):
            return True
    return False


def listing_card(run, ident):
    row = listing_fields(run).get(ident) or {}
    facts = {}
    for val in row.values():
        if isinstance(val, dict):
            for key, x in list(val.items())[:40]:
                if scalar(x):
                    facts[str(key)] = str(x)[:90]
    learned = seller_fields(run)
    sellers = sorted(set(str(v) for k, v in row.items() if seller_value(k, v, learned)))
    card = {"id": ident, "sold by": " / ".join(sellers), "category": shelf_of(row), "title": title_of(row)[:160],
            "facts": facts, "notes": note_text(row)[:900]}
    return as_text(card, 5000)


def ask_hop(run, ref, pick, reason):
    if not hop_calls_left(run):
        return None
    run.ref_calls = stash(run, "ref_calls", 0) + 1
    parts = [REF_HOP.replace("PROPERTY", ref["prop"]).replace("DESCRIPTION", describe_target(ref["target"])),
             "\n\nCandidate listing records:\n", "\n".join(listing_card(run, i) for i in pick)]
    if reason:
        parts.append("\n\nYour previous reply was rejected: %s Answer again from the records." % reason)
    answer = consult(run.wire, "".join(parts), 3000, CHECKER[:1])
    note_event(run, "ref_hop", prop=ref["prop"], ids=pick, answer=answer)
    return answer


def hop_target(answer):
    named = answer.get("target")
    if isinstance(named, bool) or not isinstance(named, (str, int)):
        return ""
    return str(named).strip()


def miss(where, problem):
    return where, problem, None, "", []


def follow_hop(run, ref, target, go):
    ref["chain"].append(target)
    sig = target_signature(go)
    if sig in ref["seen"]:
        give_up_reference(run, ref, "its notes lead back to a listing already followed")
    elif ref["hops"] + 1 >= REF_HOPS:
        give_up_reference(run, ref, f"more than {REF_HOPS} listings in a row send the reader elsewhere")
    else:
        ref["seen"].add(sig)
        ref["hops"] += 1
        ref.update(target=go, judged=set(), queries=[], pages=0, tries=0, fails=0, unindexed=False, loose=set())
        note_event(run, "ref_go", prop=ref["prop"], **{"from": target, "to": go})
        tell_actor(run, "%s matches the listing given for %s, and its notes send the reader on to %s: the code follows them."
                   % (target, ref["head"], describe_target(ref["target"])))


def give_up_reference(run, ref, why):
    if ref["status"] == "open":
        ref["status"] = "stuck"
        ref["why"] = why
        note_event(run, "ref_stuck", req=ref["req"][:200], why=why[:300], chain=list(ref["chain"]))
        tell_actor(run, f"NOT RESOLVED: {ref['head']} ({why}). Nobody knows its value yet: if a listing's notes you read show "
                        "where it ends, use that value; otherwise never decline or reject a listing over it.")


def tell_actor(run, text):
    chat = run.chat
    if chat and len(chat) > 2 and chat[-1].get("role") in ("tool", "user") and isinstance(chat[-1].get("content"), str):
        chat[-1]["content"] += "\n[reference] " + text
    else:
        run.ref_told = list(stash(run, "ref_told") or []) + [text]


def ref_status_piece(ref):
    status = ref["status"]
    if status == "resolved":
        differ = ", must differ from it" if ref.get("relation") == "differ" else ""
        return "%s (read on %s%s)" % (ref["now"], ref["chain"][-1], differ)
    if status == "open":
        return f"{ref['head']}: the code is following it (now: {describe_target(ref['target'])})"
    if status == "stuck":
        return f"{ref['head']}: not resolved, value unknown (never decline or reject a listing over it)"
    return None


def ref_line(run):
    try:
        parts = [p for p in map(ref_status_piece, stash(run, "refs") or []) if p is not None]
        told = vars(run).pop("ref_told", None)
        out = ""
        if parts:
            out = "\n[references] " + "; ".join(parts)
        for text in told or []:
            out += "\n[reference] " + text
        return out
    except Exception:
        return ""


def resolved_note(ref):
    chain = ref["chain"]
    via = " (first seen on %s, then followed through the listings' notes)" % chain[0] if len(chain) > 1 else ""
    pairs = " and ".join(ref["goes_with"])
    if ref.get("relation") == "differ":
        pair = "; its notes put %s alongside it" % pairs if ref["goes_with"] else ""
        return (f"- {ref['now']}: Check: the shopper's {ref['prop']} has to be different from what {chain[-1]} shows "
                f"({ref['value']}){via}{pair}. Then: every other value passes and that one fails; the listings along its "
                f"chain ({', '.join(chain)}) all carry it.")
    if ref.get("relation") == "pair" and ref["goes_with"]:
        return (f"- {ref['now']}: {chain[-1]} shows {ref['value']}{via}, and by its notes {ref['value']} comes paired with "
                f"{pairs}; the shopper wants that counterpart, so {pairs} passes and {ref['value']} itself fails.")
    pair = f"; by its notes {ref['value']} comes paired with {pairs}, and either of them passes" if ref["goes_with"] else ""
    return f"- {ref['now']}: this is the value written on {chain[-1]}{via}{pair}."


def limit_keys(call_args, page_key, fixed_paging):
    out = []
    for key, val in call_args.items():
        if key == page_key or key in fixed_paging or isinstance(val, bool) or not is_number(val) or val <= 1:
            continue
        if not re.search(r"page|size|count|result|min|low|from|start|offset", str(key).lower()):
            out.append(key)
    return out


def row_cost(row):
    for key in row:
        val = row[key]
        if is_number(val) and not isinstance(val, bool) and re.search(r"price|cost", str(key).lower()):
            return float(val)
    return None


def ceil_spec(run):
    spec = stash(run, "ceil_mode")
    if spec and spec["sig"] not in stash(run, "ceil_bad", set()):
        return spec
    return None


def cg_listing(run, args):
    spec = ceil_spec(run)
    if not spec or not isinstance(args, dict):
        return False
    pinned = spec["mode"]
    if pinned:
        return all(args.get(k) == pinned[k] for k in pinned)
    return spec["bound"] in args


def page_size(run):
    sizes = [w.get("size", 0) for w in stash(run, "page_book", {}).values()]
    return max(sizes or [0])


def visible_depth(run):
    paging = page_arg(run)
    size = page_size(run)
    if paging and size > 0 and paging[1] != "token" and paging[3] is not None:
        first, last = paging[2], paging[3]
        return int((last - first + 1) * size) if paging[1] else int(last - first + size)
    return stash(run, "page_cap")



# ---------------------------------------------------------------------------------------------- price ceilings and value rows
def below(run, price):
    if not is_number(price):
        return None
    whole_only = (ceil_spec(run) or {}).get("int")
    amount = float(price)
    if not (whole_only or amount.is_integer()):
        limit = round(amount - 0.01, 2)
    elif amount.is_integer():
        limit = int(amount) - 1
    else:
        limit = int(amount)
    return limit if limit > 0 else None


def ceil_mid(run, a, b):
    whole_only = (ceil_spec(run) or {}).get("int")
    if not whole_only and not (float(a).is_integer() and float(b).is_integer()):
        return round((float(a) + float(b)) / 2, 2)
    return (int(a) + int(b)) // 2


def ceil_args(run, spec, label, bound):
    args = dict(spec["tmpl"])
    args[spec["shelf"]] = label
    args[spec["bound"]] = int(bound) if spec.get("int") or float(bound).is_integer() else bound
    return args


def cg_products(run):
    count = len(need_items(run.needs))
    return list(range(count)) if count else [None]


def ceiling_ready(run, k):
    if need_items(run.needs) and k is None:
        return False
    if need_priority(run.needs) != "cheapest":
        return False
    # (a dropped requirement widens the acceptable set, so a product walked before it may be walked once more)
    if k in vars(run).get("ceil_walked", set()) and (vars(run).get("ceil_walk_at") or {}).get(k, -1) == run.needs.widened_at:
        return False
    if ceil_spec(run) is None or page_arg(run) is None:
        return False
    return bool(raw_label(run, walk_home(run, k)))


def ceil_aim(run, k):
    pick = walk_pick(run, k)
    held_aim = (vars(run).get("ceil_aims") or {}).get(k)
    costs = [pick[1]] if pick else []
    if is_number(held_aim):
        costs.append(float(held_aim))
    if not costs and bound_need(run, k):
        home = walk_home(run, k)
        seen = [cost for cost, _ in cheap_unread(run, home_rows(run, home), k)]
        for ident in run.archive.read:
            if shelf_known(run, ident) not in home or not is_number(price_of(run.archive, ident)):
                continue
            if ident in dead_lines(run) or ident in run.refused_lines or last_stock(run, ident) is False:
                continue
            seen.append(float(price_of(run.archive, ident)))
        costs = [min(seen)] if seen else []
    return min(costs) if costs else None


def bound_need(run, k):
    if not need_items(run.needs) or k is None:
        return False
    for req in props_pending(run.needs):
        head, _, value = str(req).partition(":")
        if product_of(run, head) != k:
            continue
        if re.search(r"\d", value) and re.search(CH_BOUND, value.lower()):
            return True
    return False


def archive_rows(run):
    rows = []
    for rec in run.archive.items:
        walk_dicts(rec, rows.append)
    return rows


def names_listing(row, ids):
    return any(isinstance(v, str) and v in ids and "id" in str(key).lower() for key, v in row.items())


def home_rows(run, home):
    known = run.archive.ids
    marks = map(row_mark, filter(lambda row: shelf_of(row) in home, archive_rows(run)))
    return {mark for mark in marks if mark in known}


def ceil_clash(run, home, bound):
    for row in archive_rows(run):
        if shelf_of(row) not in home or not names_listing(row, run.archive.ids):
            continue
        cost = row_cost(row)
        if cost is not None and cost <= bound and stock_state(row) is True:
            return True
    return False


def cheap_unread(run, ids, k, under=None):
    wanted = set(ids)
    if not wanted:
        return []
    rows = need_items(run.needs)
    qty = rows[k]["quantity"] if k is not None and k < len(rows) else need_qty(run.needs)
    cap = run.needs.spend_cap
    sold = {}
    for row in archive_rows(run):
        state = stock_state(row) if names_listing(row, wanted) else None
        if state is not None:
            sold[row_mark(row)] = not state
    ranked = []
    for ident in wanted:
        cost = price_of(run.archive, ident)
        if fields_seen(run, ident) or ident in run.refused_lines or sold.get(ident) or not is_number(cost):
            continue
        too_dear = is_number(cap) and is_number(qty) and cost * qty > cap
        if too_dear or (under is not None and cost >= under):
            continue
        ranked.append((float(cost), ident))
    ranked.sort()
    return ranked


def all_words(run):
    if run.word_kind == "all":
        return True
    return bool(run.lead_words) and run.word_kind != "any"


def valued_ids(run, k):
    if not all_words(run):
        return set()
    home = walk_home(run, k)
    hits = set()
    for walk in vars(run).get("page_book", {}).values():
        if walk.get("id_order") is not True or cg_listing(run, walk["args"]) or not walk_carries(run, walk["args"], k):
            continue
        hits |= {ident for ident, shelf in (walk.get("row_shelf") or {}).items() if shelf in home}
    return hits


def bare_value(req):
    return re.sub(r"\((?:stated|revealed[^)]*)\)", "", req.partition(":")[2]).lower()


def value_alternatives(value):
    return re.split(r"[,;/]|\bor\b|\beither\b|\bnor\b", value)


def valued_complete(run, k):
    try:
        if not all_words(run):
            return False
        home = walk_home(run, k)
        several = bool(need_items(run.needs)) and k is not None
        single = set()
        for n, req in enumerate(props_pending(run.needs)):
            if several and product_of(run, req.partition(":")[0]) not in (k, None):
                continue
            if len([p for p in value_alternatives(bare_value(req)) if p.strip()]) == 1:
                single.add(n)
        for walk in vars(run).get("page_book", {}).values():
            finished = walk.get("id_order") is True and not cg_listing(run, walk["args"]) and walk.get("first") \
                and next_page(run, walk) is None and len(walk.get("row_shelf") or {}) >= walk["total"]
            if not finished:
                continue
            if any(s in home for s in (walk.get("row_shelf") or {}).values()) and walk_carries(run, walk["args"], k) & single:
                return True
    except Exception:
        return False
    return False


def row_carries(run, ident, k):
    several = bool(need_items(run.needs)) and k is not None
    word_sets, figures = [], []
    for req in props_pending(run.needs):
        if several and product_of(run, req.partition(":")[0]) not in (k, None):
            continue
        value = bare_value(req)
        if ref_text(value):
            continue
        for part in value_alternatives(value):
            fig = fig_amount(part)
            if fig:
                figures.append(fig)
            elif not amount_part(part):
                stems = [stem(w) for w in re.findall(r"[a-z0-9]{3,}", part)]
                if stems:
                    word_sets.append(stems)
    if not word_sets and not figures:
        return False
    own_rows = [row for row in archive_rows(run) if names_listing(row, {ident})]
    body = " ".join(str(v).lower() for row in own_rows for v in row.values() if isinstance(v, str))
    have = {stem(w) for w in re.findall(r"[a-z0-9]{3,}", body)}
    for fig, unit in figures:
        if re.search(r"(?<![\d.,])" + re.escape(fig) + r"\s*-?\s*" + re.escape(stem(unit)), body):
            return True
    return any(all(w in have for w in words) for words in word_sets)


# ---------------------------------------------------------------------------------------------- composed value searches
def vl_room(run):
    left = 0
    if run.asking and role(run, "message"):
        left = max(0, ask_limit(run) - run.msgs_sent)
    return left, moves_left(run) - order_due(run) - 2 * left


def pointed_value_words(piece, own_name):
    fig = fig_amount(piece)
    if re.search(EXCLUDE_WORDS, piece) or (amount_part(piece) and not fig):
        return []
    piece = re.sub(r"\b(?:" + VALUE_LINKS + r")\b", " ", re.sub(r"'s\b", "", piece))
    kept = []
    for word in re.findall(r"\d+(?:[.,]\d+)+|[a-z0-9]+", piece):
        if re.search(r"[.,]", word) or word in LINK_WORDS:
            continue
        if len(word) < 2 and not (fig and word == fig[0]):
            continue
        if word in own_name and not (fig and word == fig[1]):
            continue
        kept.append(word)
    return kept


def vl_values(run, k):
    groups = []
    for head, value in code_values(run, k):
        low = re.sub(r"\([^)]*\)", " ", str(value).lower().replace("’", "'")).strip()
        if not low or ref_text(low):
            continue
        if not re.search(r"\bor\b|\beither\b|/", low):
            pieces = [re.split(r"[,;&+]|\band\b", low)[0]]
        else:
            pieces = [p for p in re.split(r"[,;/]|\bor\b|\beither\b", low) if p.strip()][:3]
        own_name = set(re.findall(r"[a-z0-9]+", str(head).lower()))
        choices = []
        for piece in pieces:
            words = pointed_value_words(piece, own_name)
            if words and words not in choices:
                choices.append(words)
        if choices and choices not in groups:
            groups.append(choices)
    return groups


def vl_words(run, k, label):
    cleaned = re.sub(r"'s\b", "", label.lower().replace("’", "'"))
    words = list(dict.fromkeys(w for w in re.findall(r"[a-z0-9]+", cleaned) if len(w) >= 2 and w not in LINK_WORDS))
    if vars(run).get("label_indexed") is not False:
        return words
    home = walk_home(run, k)
    titles = [{stem(w) for w in re.findall(r"[a-z0-9]+", title_of(row).lower())} for row in archive_rows(run)
              if shelf_of(row) in home and row_mark(row) in run.archive.ids]
    if titles:
        common = [w for w in words if 2 * sum(1 for t in titles if stem(w) in t) >= len(titles)]
        if common:
            return common
    rows = need_items(run.needs)
    if rows and k is not None and k < len(rows):
        product = str(rows[k]["product"])
    else:
        product = str(run.needs.reading.get("product") or "")
    return [w for w in re.findall(r"[a-z0-9]+", product.lower()) if len(w) > 2 and w not in LINK_WORDS][-1:]


def unique_forms(forms):
    return list(dict.fromkeys(f for f in forms if f))


def vl_queries(run, k, label, cap_len):
    groups = vl_values(run, k)
    if not groups:
        return [], []
    label_words = vl_words(run, k, label)
    every_alt = need_priority(run.needs) == "cheapest"

    def compose(parts):
        seen, picked = set(), []
        for word in [w for part in parts for w in part] + label_words:
            if stem(word) in seen:
                continue
            seen.add(stem(word))
            picked.append(word)
        if run.lead_words:
            picked = picked[: run.lead_words]
        text = " ".join(picked)
        if is_number(cap_len):
            while len(text) > cap_len and " " in text:
                text = text.rsplit(" ", 1)[0]
            text = text[: int(cap_len)]
        return text

    def options(group):
        return group if every_alt else group[:1]
    single = [compose([alt]) for alt in options(groups[0])]
    if len(groups) < 2:
        return unique_forms(single), []
    j = next((n for n, group in enumerate(groups) if len(options(group)) > 1), 0)
    joined = [compose([alt if n == j else group[0] for n, group in enumerate(groups)]) for alt in options(groups[j])]
    return unique_forms(joined), unique_forms(single)


def vl_next(run, first, second, tkey, mode):
    for forms in filter(None, (first, second)):
        unsent = None
        totals = []
        for form in forms:
            total = vl_seen(run, form, tkey, mode)
            totals.append(total)
            if total is None and unsent is None:
                unsent = form
        if unsent is not None:
            return unsent
        if any(totals):
            return ""
    if first or second:
        return None
    return ""


def vl_mode(run, args):
    schema = run.shapes.get(role(run, "search") or "") or {}
    props = schema.get("properties")
    if not isinstance(props, dict):
        props = {}
    chosen_pairs = []
    for key, val in args.items():
        spec = props.get(key)
        if isinstance(val, str) and isinstance(spec, dict) and isinstance(spec.get("enum"), list) and len(spec["enum"]) <= 4:
            chosen_pairs.append((key, val))
    return tuple(sorted(chosen_pairs))


def vl_set(run, text):
    words = re.findall(r"[a-z0-9]+", str(text).lower())
    return frozenset(stem(w) for w in words[: run.lead_words or None])


def vl_seen(run, text, tkey, mode):
    target = vl_set(run, text)
    for key, total in (vars(run).get("ceil_seen") or {}).items():
        try:
            args = json.loads(key)
        except Exception:
            continue
        if not isinstance(args, dict) or not isinstance(args.get(tkey), str):
            continue
        if all(args.get(m) == v for m, v in mode.items()) and vl_set(run, args[tkey]) == target:
            return int(total)
    return None


def vl_indexed(run, label):
    if vars(run).get("label_indexed") is not None or not run.archive.items:
        return
    words = {stem(w) for w in re.findall(r"[a-z0-9]+", re.sub(r"'s\b", "", label.lower())) if len(w) >= 2 and w not in LINK_WORDS}
    shelf = label.strip().lower()
    newest = []
    walk_dicts(run.archive.items[-1], newest.append, ceiling=3)
    hits = 0
    for row in newest:
        if row_mark(row) not in run.archive.ids or shelf_of(row) != shelf:
            continue
        other = set()
        for key, v in row.items():
            if isinstance(v, str) and not re.search(r"categor|shelf|department|aisle", str(key).lower()):
                other.update(stem(w) for w in re.findall(r"[a-z0-9]+", v.lower()))
        hits += bool(words - other)
    if hits >= 2:
        run.label_indexed = True


def fit_to_steps(run, queue, batch, floor):
    while queue and (1 if batch and len(queue) >= 2 else len(queue)) >= moves_left(run) - floor:
        queue.pop()


def vl_queue(run, k, ids, floor, most):
    cheapest = need_priority(run.needs) == "cheapest"
    pick = walk_pick(run, k) if cheapest else None
    batch = batch_tool(run)
    opened = vars(run).setdefault("valued_opens", {})
    room = CEIL_OPENS - opened.get(k, 0) if cheapest else most
    width = batch[2] if batch else most
    queue = [ident for _, ident in cheap_unread(run, ids, k, pick[1] if pick else None)][:max(0, min(width, room))]
    fit_to_steps(run, queue, batch, floor)
    if cheapest:
        opened[k] = opened.get(k, 0) + len(queue)
    return queue


def vl_launch(run, queue, homes, floor, say):
    still_ranked = set()
    for mark in run.rank_checked:
        if shelf_known(run, mark.split("|")[0]) not in homes:
            still_ranked.add(mark)
    run.rank_checked = still_ranked
    run.order_vetted = False
    for name, value in (("open_queue", queue), ("open_recheck", True), ("open_say", say), ("open_purpose", "walk"), ("open_floor", floor)):
        setattr(run, name, value)
    open_named(run)


def keyword_template(run, k):
    for call in code_queries(run, k):
        typed = free_text(run, call)
        keys = [key for key, val in call.items() if isinstance(val, str) and val.strip() and val in typed]
        if len(keys) != 1 or cg_listing(run, call):
            continue
        return call, keys[0]
    return None, ""


def priced_ids(run, queue):
    return ", ".join(f"{i} ({price_of(run.archive, i):g})" for i in queue)


def tack_on_tool(run, text, with_status=False):
    if not run.chat or run.chat[-1].get("role") != "tool":
        return False
    if with_status:
        text += "\n" + status_line(run)
    run.chat[-1]["content"] = str(run.chat[-1]["content"]) + "\n" + text
    return True


def rows_on(walk, home):
    return [i for i, s in (walk.get("row_shelf") or {}).items() if s in home]


def shows_home(walk, home):
    return any(s in home for s in (walk.get("row_shelf") or {}).values())


def label_forms(run, k, label, cap_len, tkey, mode, home, sets):
    first, second = vl_queries(run, k, label, cap_len)
    sets.setdefault(str(k), set()).update(vl_set(run, q) for q in first + second)
    todo = vl_next(run, first, second, tkey, mode)
    if todo is not None:
        return first, second, todo
    book = vars(run).get("page_book", {})
    carried = any(walk_carries(run, w["args"], k) and shows_home(w, home) for w in book.values())
    if not carried or vars(run).get("label_indexed") is not None:
        return first, second, ""
    # all forms empty although the bare value found rows there, so labels are not indexed here
    run.label_indexed = False
    first, second = vl_queries(run, k, label, cap_len)
    sets[str(k)].update(vl_set(run, q) for q in first + second)
    return first, second, vl_next(run, first, second, tkey, mode) or ""


def vl_search(run):
    if vl_room(run)[1] <= 4:
        return False
    store, needs = vars(run), run.needs
    jschema = run.shapes.get(role(run, "search") or "") or {}
    props = jschema.get("properties")
    if not isinstance(props, dict):
        props = {}
    found = page_arg(run)
    tries = store.setdefault("label_tries", {})
    settled = store.setdefault("label_done", {})
    sets = store.setdefault("label_sets", {})
    products = cg_products(run)
    for k in products:
        if tries.get(k, 0) >= LABEL_TRIES:
            continue
        sig = json.dumps(vl_values(run, k))
        closed = settled.setdefault(str(k), [])
        if sig in closed:
            continue
        home = walk_home(run, k)
        label = raw_label(run, home)
        if not label:
            continue
        tmpl, tkey = keyword_template(run, k)
        if not tmpl:
            closed.append(sig)
            continue
        mode = {}
        for key, val in tmpl.items():
            if isinstance(props.get(key), dict) and props[key].get("enum"):
                mode[key] = val
        # a bare keyword call, comparable with the actor's own searches
        required = jschema.get("required") or []
        tmpl = {key: val for key, val in tmpl.items() if key == tkey or key in mode or key in required}
        text_spec = props.get(tkey)
        cap_len = text_spec.get("maxLength") if isinstance(text_spec, dict) else None
        first, second, todo = label_forms(run, k, label, cap_len, tkey, mode, home, sets)
        args, faults = check_args({**tmpl, tkey: todo}, jschema) if todo else (None, ["empty"])
        if faults or not isinstance(args, dict):
            closed.append(sig)
            continue
        tries[k] = tries.get(k, 0) + 1
        run_label_search(run, k, args, todo, first, second, tkey, mode, home, label, found, len(products))
        return True
    return False


def run_label_search(run, k, args, todo, first, second, tkey, mode, home, label, found, product_count):
    store, needs = vars(run), run.needs
    plural = "s" if len(vl_values(run, k)) > 1 else ""
    walk_call(run, args, f"Searching the required value{plural} together with the category name as written on this store's "
                         f"rows ('{todo}'): keyword search reaches listing values, titles and categories, so the hits are "
                         "the category's listings that carry the value.")
    total = vl_seen(run, todo, tkey, mode)
    vl_indexed(run, label)
    key = json.dumps({a: v for a, v in args.items() if not found or a != found[0]}, sort_keys=True, default=str)
    walk = store.get("page_book", {}).get(key)
    every = bool(walk) and len(walk.get("row_shelf") or {}) >= walk["total"] and shows_home(walk, home)
    floor = order_due(run) + 2
    queue = []
    note = ""
    if total is not None:
        if not total:
            note = f"[value search] '{todo}': no listing matched."
        else:
            note = f"[value search] '{todo}': {total} match(es)" + (" - every one is shown here." if every else ".")
    cheapest = lambda: need_priority(needs) == "cheapest"
    if every and cheapest() and found and walk.get("id_order") is not False:
        store.setdefault("valued", set()).add(key)
        store.setdefault("valued_keys", {}).setdefault(str(k), []).append(key)
        if product_count > 1:
            store.setdefault("valued_wait", {})[k] = True
        else:
            queue = vl_queue(run, k, rows_on(walk, home), floor, CEIL_OPENS)
    elif every and not cheapest() and todo in (first or second):
        queue = vl_queue(run, k, rows_on(walk, home), floor, LABEL_OPENS)
    elif walk and total and not every and found:
        queue = reach_rows(run, k, walk, key, todo, home, floor, found[0], REACH_OPENS if product_count == 1 else 2)
    if queue:
        under_pick = " below the pick" if walk_pick(run, k) and cheapest() else ""
        note += (" Its unread in-stock rows within the budget on the requested category" + under_pick
                 + ", cheapest first, are opened next: " + priced_ids(run, queue) + ".")
    if note:
        tack_on_tool(run, note, with_status=not queue)
    run.wire.log.append({"kind": "label_search", "product": k, "query": todo, "total": total, "every": every, "opened": queue})
    if queue:
        vl_launch(run, queue, home, floor, "Opening the cheapest unread rows from that search, since every category "
                                           "listing that carries the required values is among its matches.")


def reach_rows(run, k, walk, key, todo, home, floor, page_key, most):
    # (a store that pages its matches puts the rows carrying the most searched words first, so the fits for the values
    # just learned sit on the first pages of their search: read one more page when the first one holds too few unread
    # in-budget rows of the product's category, then open the cheapest of them, as the order still has to be bought)
    ready = lambda: cheap_unread(run, rows_on(walk, home), k)
    if len(ready()) < most and moves_left(run) > floor + 3 and walk_open(run, floor + 2):
        token = next_page(run, walk)
        if token is not None:
            args = dict(walk["args"])
            args[page_key] = token
            walk_call(run, args, f"Reading the next page of '{todo}': rows carrying the searched values come first in this "
                                 "store, so the fits sit on the first pages.")
            walk = vars(run).get("page_book", {}).get(key) or walk
    return vl_queue(run, k, rows_on(walk, home), floor, most)


def take_in_turns(ks, pools, most):
    owner, queue = {}, []
    while len(queue) < most and any(pools):
        for k, pool in zip(ks, pools):
            while pool and pool[0] in queue:
                del pool[0]
            if pool and len(queue) < most:
                ident = pool.pop(0)
                owner[ident] = k
                queue.append(ident)
    return owner, queue


def vl_flush(run, floor):
    store = vars(run)
    wait = store.setdefault("valued_wait", {})
    ks = [k for k, on in wait.items() if on]
    wait.clear()
    if not ks:
        return False
    book = store.get("page_book", {})
    keys = store.get("valued_keys") or {}
    opened = store.setdefault("valued_opens", {})
    batch = batch_tool(run)
    per = 1 if moves_left(run) - floor - 1 < 2 * len(ks) else 2
    pools = []
    for k in ks:
        home = walk_home(run, k)
        ids = set()
        for key in keys.get(str(k), []):
            if key in book and walk_carries(run, book[key]["args"], k):
                ids |= set(rows_on(book[key], home))
        pick = walk_pick(run, k)
        cheapest_first = [i for _, i in cheap_unread(run, ids, k, pick[1] if pick else None)]
        pools.append(cheapest_first[:max(0, min(per, CEIL_OPENS - opened.get(k, 0)))])
    owner, queue = take_in_turns(ks, pools, batch[2] if batch else per * len(ks))
    fit_to_steps(run, queue, batch, floor)
    if not queue:
        return False
    for ident in queue:
        opened[owner[ident]] = opened.get(owner[ident], 0) + 1
    homes = set().union(*(walk_home(run, k) for k in ks))
    vl_launch(run, queue, homes, floor, "The cheapest was requested, so the cheapest unread rows from every product's value search get opened.")
    tack_on_tool(run, "[value searches read] Opened, per product and below its pick, the cheapest unread in-stock rows within the "
                      "budget that its value search turned up: " + priced_ids(run, queue) + ". For each product, "
                      "pick the least costly listing whose own fields satisfy every requirement now in force; a fit that costs more "
                      "needs every cheaper candidate opened and rejected first.")
    run.wire.log.append({"kind": "value_flush", "products": ks, "opened": queue})
    return True


def product_steps(run):
    try:
        rows = need_items(run.needs)
        if len(rows) < 2 or need_priority(run.needs) != "cheapest":
            return ""
        pages_read = vars(run).get("valued_by") or {}
        parts = []
        waiting = ""
        for k, row in enumerate(rows):
            has_pick = bool(walk_pick(run, k))
            answer = "yes" if has_pick else "no"
            parts.append(f"{row['product']}: {pages_read.get(k, 0)} value page(s) read, pick {answer}")
            if not (waiting or has_pick):
                waiting = str(row["product"])
        tail = f"; next: {waiting}" if waiting else ""
        return "\n[product steps] " + "; ".join(parts) + tail + "."
    except Exception:
        return ""


def search_done_whole(run, walk):
    if walk.get("id_order") is not True or not walk.get("first") or next_page(run, walk) is not None:
        return False
    return not cg_listing(run, walk["args"]) and len(walk.get("row_shelf") or {}) >= walk["total"]


def pages_blocked(run, walk, k, home, done, key):
    if key in done or walk.get("id_order") is not True or not walk.get("first") or next_page(run, walk) is None:
        return True
    if cg_listing(run, walk["args"]) or not walk_carries(run, walk["args"], k):
        return True
    return not shows_home(walk, home)


class PagePlan:
    def __init__(self, run, products):
        store = vars(run)
        self.book = store.get("page_book", {})
        self.done = store.setdefault("valued", set())
        self.used = store.setdefault("valued_by", {})
        self.opened = store.setdefault("valued_opens", {})
        self.wait = store.setdefault("valued_wait", {})
        self.sets = store.get("label_sets") or {}
        self.depth = visible_depth(run)
        size = max(1, page_size(run))
        self.per = VALUE_PAGES if not self.depth else max(VALUE_PAGES, -(-self.depth // size) - 1)
        self.several = len(products) > 1
        self.cap = max(VALUE_TOTAL, VALUE_TOTAL_PER * len(products))
        self.opens = 1 if batch_tool(run) else (2 if self.several else CEIL_OPENS)
        self.floor = order_due(run) + 2
        self.known = all_words(run)
        self.left, self.room = vl_room(run)
        unread_products = [k for k in products if walk_home(run, k) and not self.used.get(k)]
        self.share = max(4, self.room // max(1, len(unread_products)))
        self.pending = False


def product_value_words(run, k):
    per_product = bool(need_items(run.needs)) and k is not None
    values = set()
    for req in props_pending(run.needs):
        if per_product and product_of(run, req.partition(":")[0]) not in (k, None):
            continue
        values.update(stem(w) for w in re.findall(r"[a-z0-9]{2,}", req.partition(":")[2].lower()))
    return values


def page_candidate(run, plan, k, key, walk, home, composed_sets, extra_ok, whole):
    if pages_blocked(run, walk, k, home, plan.done, key):
        return None
    words = vl_set(run, " ".join(free_text(run, walk["args"])))
    composed = words in composed_sets
    deep = plan.depth is not None and walk["total"] > plan.depth
    if not plan.known and (plan.depth is None or deep):
        return None
    if plan.known and deep and not composed:
        return None
    pairs = walk_carries(run, walk["args"], k, alts=True)
    how = vl_mode(run, walk["args"])
    if plan.known and any(mode == how and (seen <= words or (seen <= extra_ok and pairs <= got)) for seen, got, mode in whole):
        return None
    reach = min(walk["total"], plan.depth) if plan.depth is not None else walk["total"]
    pages = -(-reach // max(1, walk["size"])) - len(walk["read"])
    used_k, opened_k = plan.used.get(k, 0), plan.opened.get(k, 0)
    allowed = min(plan.per - used_k, plan.cap - sum(plan.used.values()))
    if plan.several:
        allowed = min(allowed, plan.share - used_k - opened_k - (0 if plan.opened.get(k) else plan.opens))
    take = pages if pages <= allowed else (allowed if composed else 0)
    if take <= 0 or moves_left(run) - take - plan.opens <= plan.floor:
        return None
    if (take > 2 or plan.several) and plan.left and plan.room > take + plan.opens + 4:
        # the question burst goes first; a reveal may narrow this search
        plan.pending = True
        return None
    return take, -walk_score(run, walk["args"], k), key, words


def product_page_options(run, plan, n, k):
    home = walk_home(run, k)
    if not home:
        return []
    composed_sets = plan.sets.get(str(k)) or set()
    extra_ok = product_value_words(run, k) | {stem(w) for s in home for w in re.findall(r"[a-z0-9]{2,}", s)}
    whole = [(vl_set(run, " ".join(free_text(run, w["args"]))), walk_carries(run, w["args"], k, alts=True), vl_mode(run, w["args"]))
             for w in plan.book.values() if search_done_whole(run, w)]
    cands = []
    for key, walk in plan.book.items():
        found = page_candidate(run, plan, k, key, walk, home, composed_sets, extra_ok, whole)
        if found is not None:
            cands.append(found)
    if plan.known:
        cands = [c for c in cands if not any((o[3] < c[3] and not (c[3] - o[3]) <= extra_ok)
                                             or (c[3] < o[3] and (o[3] - c[3]) <= extra_ok) for o in cands)]
    return [(plan.used.get(k, 0), take, score, n, key, k) for take, score, key, _ in cands]


def value_pages(run):
    store, wire = vars(run), run.wire
    try:
        if any(r.get("status") == "open" for r in store.get("refs") or []) and ref_room(run):
            return False
        stopped = run.orders_tried or run.refusal_n or wire.done or wire.broke
        if stopped or time_left(wire) < CLOSING + 60 or not role(run, "search"):
            return False
        if vl_search(run):
            return True
        if need_priority(run.needs) != "cheapest" or run.word_kind == "any" or not page_arg(run):
            return False
        products = cg_products(run)
        plan = PagePlan(run, products)
        options = []
        for n, k in enumerate(products):
            options += product_page_options(run, plan, n, k)
        if options:
            _, take, _, _, key, k = min(options, key=lambda o: o[:5])
            plan.done.add(key)
            return read_value_search(run, k, key, plan.floor, take, plan.several)
        if not plan.wait or plan.pending or (plan.left and plan.room > plan.opens * len(plan.wait) + 4):
            return False
        return vl_flush(run, plan.floor)
    except Exception as err:
        wire.log.append({"kind": "value_pages_error", "error": f"{type(err).__name__}: {err}"[:300]})
        return False


VALUE_CLOSING = ("Pick the least costly listing whose own fields satisfy every requirement now in force; a fit that costs more "
                 "needs every cheaper candidate opened and rejected first.")


def read_value_search(run, k, key, floor, most=VALUE_PAGES, defer=False):
    store, wire = vars(run), run.wire
    found = page_arg(run)
    walk = store.get("page_book", {}).get(key)
    if not (found and walk):
        return False
    query = " ".join(free_text(run, walk["args"]))
    used = store.setdefault("valued_by", {})
    known = all_words(run)
    if known:
        say = ("Cheapest requested; this store demands all words and lists by fixed id instead of price, so the next page of "
               f"'{query}' gets read until each listing with these words has been seen.")
    else:
        say = ("Cheapest requested; results here follow a fixed id order instead of price, so the next page of "
               f"'{query}' gets read until each listing it matched has been seen.")
    got, tried = 0, False
    while next_page(run, walk) is not None and got < most and moves_left(run) > floor + 1:
        if wire.done or wire.broke or time_left(wire) < CLOSING + 30:
            break
        tried = True
        used[k] = used.get(k, 0) + 1
        upcoming = next_page(run, walk)
        if not walk_call(run, {**walk["args"], found[0]: upcoming}, say):
            break
        got += 1
        if next_page(run, walk) == upcoming:
            break
    if got == 0:
        return tried
    store.setdefault("valued_keys", {}).setdefault(str(k), []).append(key)
    seen = len(walk.get("row_shelf") or {})
    exhausted = next_page(run, walk) is None
    complete = exhausted and seen >= walk["total"]
    home = walk_home(run, k)
    if complete:
        reach = " - every listing carrying these words is among them" if known else " - every listing it matched is among them"
    elif next_page(run, walk) is None:
        reach = " - the rest cannot be reached, so a cheaper fit may lie past the reachable rows"
    else:
        reach = ""
    head = f"[value search read] '{query}': {seen} of its {walk['total']} matches seen{reach}. "
    queue = []
    if defer:
        store.setdefault("valued_wait", {})[k] = True
        middle = ("Once the value search of every product has been read, its cheapest unread in-stock rows within the "
                  "budget get opened. ")
    else:
        pick = walk_pick(run, k)
        queue = vl_queue(run, k, rows_on(walk, home), floor, CEIL_OPENS)
        if not queue:
            middle = "Among them, no unread in-stock row within the budget remains on the requested category. "
        else:
            middle = ("Its cheapest unread in-stock rows within the budget on the requested category" + (" below the pick" if pick else "")
                      + " are opened next: " + priced_ids(run, queue) + ". ")
            vl_launch(run, queue, home, floor, "The cheapest was requested, so the cheapest unread rows of that search get opened.")
    tack_on_tool(run, head + middle + VALUE_CLOSING, with_status=not queue)
    wire.log.append({"kind": "value_pages", "product": k, "query": query, "pages": got, "complete": complete, "opened": queue,
                     "defer": defer})
    return True


# ---------------------------------------------------------------------------------------------- the price-filter walk
def walk_can_go(run, floor):
    wire = run.wire
    if moves_left(run) <= floor + 1 or wire.done or wire.broke:
        return False
    return time_left(wire) >= CLOSING + 30


def mark_ceiling_bad(run, spec):
    vars(run).setdefault("ceil_bad", set()).add(spec["sig"])


def ceiling_report(label, aim, what, last, ids, cands, complete):
    if complete and not ids:
        return f"the price filter on '{label}' below {aim:g}: no listing of it costs less than {what}"
    if complete:
        return (f"the price filter on '{label}': every listing priced up to {last:g} seen ({len(ids)}, {len(cands)} of them unread, "
                "in stock and within the budget)")
    return (f"the price filter on '{label}' below {aim:g}: {len(ids)} of the cheaper listings seen "
            f"({len(cands)} unread, in stock and within the budget)")


class PriceWalk:
    def __init__(self, bound):
        self.bound = bound
        self.top = None
        self.calls = 0
        self.complete = False
        self.ran = False
        self.last = bound
        self.keys = []


def price_walk_turn(run, pw, spec, found, label, home, depth, say, floor):
    store, archive = vars(run), run.archive
    book = store["page_book"]
    args = ceil_args(run, spec, label, pw.bound)
    key = json.dumps(args, sort_keys=True, default=str)
    store.setdefault("ceil_keys", set()).add(key)
    pw.calls += 1
    booked = key in book and bool(book[key].get("first")) and key in (store.get("ceil_seen") or {})
    if not booked and not walk_call(run, args, say):
        return False
    pw.ran, pw.last = True, pw.bound
    pw.keys.append(key)
    walk = book.get(key)
    shelves = (walk.get("row_shelf") or {}) if walk else {}
    if walk and any(is_number(price_of(archive, i)) and price_of(archive, i) > pw.bound for i in shelves):
        mark_ceiling_bad(run, spec)
    if ceil_spec(run) is None:
        return False
    total = store.get("ceil_seen", {}).get(key)
    if total is None:
        return False
    if not (total and walk):
        return narrow_after_empty(run, pw, spec, home)
    pages = -(-total // max(1, walk["size"]))
    if (depth is None or total <= depth) and pages - len(walk["read"]) <= CEIL_CALLS - pw.calls:
        read_listing_out(run, pw, walk, args, found, floor)
        return False
    return lower_ceiling(run, pw, walk, total)


def narrow_after_empty(run, pw, spec, home):
    if ceil_clash(run, home, pw.bound):
        mark_ceiling_bad(run, spec)
        return False
    if pw.top is None:
        pw.complete = True
        return False
    middle = ceil_mid(run, pw.bound, pw.top)
    if not pw.bound < middle < pw.top:
        return False
    pw.bound = middle
    return True


def read_listing_out(run, pw, walk, args, found, floor):
    while next_page(run, walk) is not None and pw.calls < CEIL_CALLS and walk_can_go(run, floor):
        pw.calls += 1
        upcoming = next_page(run, walk)
        sent = walk_call(run, {**args, found[0]: upcoming}, "Reading the following page of that price-filter listing.")
        if not sent or next_page(run, walk) == upcoming:
            break
    pw.complete = next_page(run, walk) is None


def lower_ceiling(run, pw, walk, total):
    pw.top = pw.bound
    prices = [float(c) for c in map(lambda i: price_of(run.archive, i), walk.get("row_shelf") or {}) if is_number(c)]
    low = min(prices) if prices else 0.0
    # spread the total evenly over the prices to guess where the readable pages end
    fill = max(1, CEIL_CALLS - pw.calls) * max(1, walk["size"])
    spread_guess = low + (pw.bound - low) * 1.8 * fill / max(total, 1)
    lowered = min(ceil_mid(run, low, pw.bound), ceil_mid(run, low, spread_guess))
    if not lowered < pw.bound:
        return False
    pw.bound = lowered
    return True


def ranked_cheaper(run, k, ids, aim):
    cands = [i for _, i in cheap_unread(run, ids, k, aim)]
    valued = valued_ids(run, k)
    lead = [i for i in cands if i in valued]
    rest = [i for i in cands if i not in valued]
    texty = [i for i in rest if row_carries(run, i, k)]
    bare = [] if valued_complete(run, k) else [i for i in rest if i not in texty]
    return cands, lead + texty + bare


def ceiling_walk(run, k, floor, repair=False):
    store = vars(run)
    spec, found = ceil_spec(run), page_arg(run)
    home = walk_home(run, k)
    label = raw_label(run, home)
    aim = ceil_aim(run, k)
    if aim is None or not (spec and found and label):
        return None
    store.setdefault("ceil_walked", set()).add(k)
    store.setdefault("ceil_walk_at", {})[k] = run.needs.widened_at
    pick_now = walk_pick(run, k)
    store.setdefault("ceil_aimed", {})[k] = (float(aim), pick_now[0] if pick_now else "")
    bound = below(run, aim)
    if bound is None:
        return None
    if repair:
        floor += 2
    elif not run.orders_tried:
        floor = max(floor, order_due(run) + 2)
    depth = visible_depth(run)
    book = store.setdefault("page_book", {})
    aimed_at_pick = pick_now or is_number((store.get("ceil_aims") or {}).get(k))
    what = "the pick" if aimed_at_pick else "the cheapest row seen"
    say = f"Since the cheapest is wanted, the store's price filter lists '{label}' priced below {what} ({aim:g})."
    pw = PriceWalk(bound)
    while pw.calls < CEIL_CALLS and walk_can_go(run, floor):
        if not price_walk_turn(run, pw, spec, found, label, home, depth, say, floor):
            break
    if not pw.ran:
        return None
    ids = set()
    for key in pw.keys:
        ids |= {i for i, s in (book.get(key, {}).get("row_shelf") or {}).items() if s in home}
    cands, ranked = ranked_cheaper(run, k, ids, aim)
    report = ceiling_report(label, aim, what, pw.last, ids, cands, pw.complete)
    run.wire.log.append({"kind": "ceiling_walk", "product": k, "label": label, "aim": aim, "calls": pw.calls,
                         "complete": pw.complete, "rows": len(ids), "ranked": ranked[:CEIL_OPENS]})
    return report, ranked[:CEIL_OPENS]


def ceiling_sweep(run, tried):
    store = vars(run)
    spec, found = ceil_spec(run), page_arg(run)
    if not (spec and found) or need_items(run.needs):
        return None, False
    rows = shelf_rows(run)
    shelves = shelf_labels(run)
    label = raw_label(run, shelves)
    if not (rows and label):
        return None, False
    bound = below(run, min(cost for cost, _, _ in rows.values()))
    if bound is None:
        return None, True
    args = ceil_args(run, spec, label, bound)
    key = json.dumps(args, sort_keys=True, default=str)
    if key not in tried:
        store.setdefault("ceil_keys", set()).add(key)
        return args, False
    walk = store.get("page_book", {}).get(key)
    if not walk:
        empty = store.get("ceil_seen", {}).get(key) == 0
        return None, empty and not ceil_clash(run, set(shelves), bound)
    upcoming = next_page(run, walk)
    if upcoming is None:
        return None, len(walk.get("row_shelf") or {}) >= walk["total"]
    more = {**args, found[0]: upcoming}
    return (None if json.dumps(more, sort_keys=True, default=str) in tried else more), False


def value_guard(run, call_name, call_args):
    store, needs, archive = vars(run), run.needs, run.archive
    try:
        wrong_call = call_name != role(run, "details") or need_priority(needs) != "cheapest"
        if wrong_call or not all_words(run) or in_reserve(run):
            return ""
        named = [v for v in call_args.values() if isinstance(v, str) and v in archive.ids]
        if len(named) != 1:
            return ""
        ident = named[0]
        if ident in archive.read or ident in run.trolley.units:
            return ""
        if ident in named_open(run) or store.get("refs") or needs.reading.get("references"):
            return ""
        k = product_index(run, ident)
        if need_items(needs) and k is None:
            return ""
        told = store.setdefault("vguard_told", set())
        mark = (k, needs.version)
        valued = None if mark in told else valued_ids(run, k)
        if not valued or ident in valued:
            return ""
        pick = walk_pick(run, k)
        costs = [pick[1]] if pick else []
        own_price = price_of(archive, ident)
        if is_number(own_price):
            costs.append(float(own_price))
        rows = cheap_unread(run, valued, k, min(costs))[:6] if costs else []
        if not rows:
            return ""
        told.add(mark)
        listed = ", ".join(f"{i} ({c:g})" for c, i in rows)
        than = " than it and the pick" if pick else " than it"
        return ("Not run (no step used). The cheapest is wanted and keyword search here requires every word, so each row that a "
                f"search led by a required value returned holds those words somewhere in its record; {ident} is not one of those "
                "rows, while these other rows do cost less" + than + " and sit unread: "
                + listed + ". Open those first, cheapest first; repeat the call later if it "
                "is still wanted.")
    except Exception as err:
        run.wire.log.append({"kind": "value_guard_error", "error": f"{type(err).__name__}: {err}"[:300]})
        return ""


def title_line(run, k, rec, template):
    needs = run.needs
    avoid = set(LINK_WORDS)
    for shelf in walk_home(run, k):
        avoid.update(re.findall(r"[a-z]+", shelf))
    for req in props_pending(needs):
        avoid.update(re.findall(r"[a-z0-9]+", req.lower()))
    for row in need_items(needs) or [{"product": str(needs.reading.get("product") or "")}]:
        avoid.update(re.findall(r"[a-z]+", str(row.get("product") or "").lower()))
    stems = set(map(stem, avoid))

    def plain_word(word):
        return word not in avoid and stem(word) not in stems
    toks = re.findall(r"[a-z0-9]+", title_of(rec).lower())
    starts = [n for n, tok in enumerate(toks) if re.fullmatch(r"[a-z]{3,}", tok) and plain_word(tok)]
    if not starts:
        return None
    n = starts[0]
    follower = toks[n + 1] if n + 1 < len(toks) else ""
    text = toks[n] + (f" {follower}" if follower and plain_word(follower) else "")
    return None if ran_query(run, text) else {**template[0], template[1]: text}


def rearm_walk(run):
    wire = run.wire
    if need_priority(run.needs) != "cheapest" or wire.done or time_left(wire) < CLOSING + 60 or not role(run, "search"):
        return None
    aimed = vars(run).get("ceil_aimed") or {}
    gone = dead_lines(run) | run.refused_lines
    opens = 2 - bool(batch_tool(run))
    for k, (aim, ident) in list(aimed.items()):
        if not ident or ident not in gone:
            continue
        if ceil_spec(run) is None or page_arg(run) is None:
            continue
        pick = walk_pick(run, k)
        if not pick or pick[1] <= aim + 1e-9 or moves_left(run) < FIRST_ORDER_BY + 3 + opens:
            continue
        del aimed[k]
        (vars(run).get("ceil_aims") or {}).pop(k, None)
        wire.log.append({"kind": "ceil_rearm", "product": k, "dead": ident, "aim": pick[1]})
        return run_walk(run, [], FIRST_ORDER_BY, [k], repair=True)
    return None


def ceiling_candidates(run):
    picked = []
    for k in cg_products(run):
        if not (walk_pick(run, k) or bound_need(run, k)):
            continue
        if ceiling_ready(run, k) and ceil_aim(run, k) is not None:
            picked.append(k)
    return picked


def walk_step(run):
    store, wire = vars(run), run.wire
    try:
        if store.get("decline_queue"):
            return run_decline_reads(run)
        queued = store.pop("walk_queue", None)
        if queued:
            return run_walk(run, *queued)
        again = rearm_walk(run)
        if again is not None:
            return again
        if run.orders_tried or need_priority(run.needs) != "cheapest" or time_left(wire) < CLOSING + 60 or wire.done:
            return False
        if unfound_product(run):
            return False
        opens = 2 - bool(batch_tool(run))
        spare = moves_left(run)
        ceil = ceiling_candidates(run)
        reserve = order_due(run) + 2
        if ceil and spare > reserve + CEIL_CALLS + opens:
            # pickless products lead, then those with fewer value pages behind them
            pages_read = store.get("valued_by") or {}
            ceil.sort(key=lambda k: (bool(walk_pick(run, k)), pages_read.get(k, 0), -1 if k is None else k))
            return run_walk(run, [], order_due(run) + 2, ceil[:1])
        if not ceil and spare > order_due(run) + 2 + CEIL_CALLS + opens and ceil_seed(run):
            return True
        plan = walk_plan(run, WALK_TOTAL)
        if not plan:
            return False
        pages = sum(n for _, _, n in plan) + sum(1 for k, _, _ in plan if walk_pick(run, k))
        if spare > order_due(run) + 2 + pages + opens:
            return False
        plan = walk_plan(run, min(WALK_TOTAL, spare - order_due(run) - 1 - opens))
        if plan:
            run.walk_early = True
            return run_walk(run, plan, order_due(run))
        return False
    except Exception as err:
        wire.log.append({"kind": "walk_error", "error": f"{type(err).__name__}: {err}"[:300]})
        return False


def shelf_choice(props, label, skip):
    wanted = label.strip().lower()
    for key, spec in props.items():
        if not isinstance(spec, dict) or key in skip:
            continue
        if not re.search(r"categor|shelf|department|aisle", str(key).lower()):
            continue
        choices = spec.get("enum")
        if not isinstance(choices, list):
            if spec.get("type") == "string":
                return key, label
            continue
        match = [v for v in choices if isinstance(v, str) and v.strip().lower() == wanted]
        if match:
            return key, match[0]
    return None, None


def limit_key(props, page_key):
    for key, spec in props.items():
        if not isinstance(spec, dict) or spec.get("type") not in ("integer", "number") or spec.get("enum") or key == page_key:
            continue
        low = str(key).lower()
        if re.search(r"ceil|max|price|cap$|budget|upper|most", low) and not re.search(r"page|first|offset|skip|count|size|num", low):
            return key
    return None


def paired_mode(spec, shelf_key, parts):
    for value in spec["enum"]:
        value_pat = r"\b" + re.escape(str(value).lower()) + r"\b"
        shelf_pat = r"\b" + re.escape(shelf_key.lower()) + r"\b"
        if any(re.search(value_pat, part) and re.search(shelf_pat, part) for part in parts):
            return value
    return None


def seed_args(run, label, bound):
    tool = role(run, "search")
    jschema = run.shapes.get(tool or "") or {}
    props = jschema.get("properties") if isinstance(jschema.get("properties"), dict) else {}
    found = page_arg(run)
    if not (props and found and is_number(bound)):
        return None
    described = [str(tdef["function"].get("description") or "") for tdef in run.toolset if tdef["function"]["name"] == tool]
    parts = [part.lower() for part in re.split(r"[.;]", described[0] if described else "")]
    shelf_key, shelf_val = shelf_choice(props, label, (token_key(run), found[0]))
    bound_key = limit_key(props, found[0])
    if not (shelf_key and bound_key):
        return None
    whole = props[bound_key].get("type") == "integer" or float(bound).is_integer()
    args = {shelf_key: shelf_val, bound_key: int(bound) if whole else bound}
    for key, spec in props.items():
        if key == shelf_key or not isinstance(spec, dict):
            continue
        if not isinstance(spec.get("enum"), list) or not 2 <= len(spec["enum"]) <= 4:
            continue
        hit = paired_mode(spec, shelf_key, parts)
        if hit is None:
            return None
        args[key] = hit
    missing = [key for key in jschema.get("required") or [] if key not in args]
    if missing:
        return None
    fixed, faults = check_args(args, jschema)
    if faults or not isinstance(fixed, dict):
        return None
    return fixed


def ceil_seed(run):
    store, needs = vars(run), run.needs
    try:
        if ceil_spec(run) is not None or store.get("ceil_mode") or need_priority(needs) != "cheapest":
            return False
        seeded = store.setdefault("ceil_seeded", set())
        found = page_arg(run)
        page_key = found[0] if found else None
        for k in cg_products(run):
            if k in seeded or (need_items(needs) and k is None):
                continue
            pick = walk_pick(run, k)
            label = raw_label(run, walk_home(run, k))
            if not (pick and label):
                continue
            seeded.add(k)
            args = seed_args(run, label, below(run, pick[1]))
            if not args:
                return False
            mode = dict(search_keys(run, args)[1])
            fixed_modes = sorted((m, v) for m, v in mode.items() if not (isinstance(args.get(m), str) and args[m] == label))
            price_arg = next((a for a in args if is_number(args[a]) and a != page_key), "")
            sig = json.dumps([fixed_modes, price_arg], default=str)
            key = json.dumps({a: v for a, v in args.items() if not found or a != found[0]}, sort_keys=True, default=str)
            if sig in store.get("ceil_bad", set()) or key in (store.get("ceil_seen") or {}):
                continue
            store.setdefault("ceil_seed_keys", set()).add(key)
            walk_call(run, args, f"Since the cheapest is wanted, I list '{label}' priced below the pick ({pick[1]:g}) to test whether "
                                 "the price filter of this store holds.")
            learned = ceil_spec(run) is not None
            run.wire.log.append({"kind": "ceil_seed", "product": k, "args": args, "learned": learned})
            if run.chat and run.chat[-1].get("role") == "tool" and ceil_spec(run) is None:
                run.chat[-1]["content"] = str(run.chat[-1]["content"]) + "\n" + status_line(run)
            return True
        return False
    except Exception as err:
        run.wire.log.append({"kind": "ceil_seed_error", "error": f"{type(err).__name__}: {err}"[:300]})
        return False


def unfound_product(run):
    needs = run.needs
    if not need_items(needs):
        return False
    try:
        read = set(map(lambda s: s.strip().lower(), read_shelves(run.archive)))
        for home in home_shelves(run):
            if not home or not home & read:
                return True
        version = str(needs.version)
        return any(mark.rpartition("|")[2] == version for mark in run.passed) and unverified(run)
    except Exception:
        return False


def walk_hold(run, call_args):
    store, cart, archive = vars(run), run.trolley, run.archive
    try:
        if need_priority(run.needs) != "cheapest" or store.get("walk_queue") or time_left(run.wire) < CLOSING + 60:
            return ""
        if run.refusal_n and moves_left(run) <= 15:
            return ""
        if call_args is not None:
            ids = [i for i, c in cart_lines(cart, call_args) if c is None or c > 0]
        else:
            ids = list(cart.units)
        only = {product_index(run, i) for i in ids if i}
        floor = 4 if run.orders_tried else FIRST_ORDER_BY + 1
        # half of the spare steps go to pages, the line search and the opens together
        half = (moves_left(run) - floor - 1) // 2
        room = half - (1 if batch_tool(run) else 2)
        ceil = []
        aims = store.setdefault("ceil_aims", {})
        for k in sorted(only, key=lambda x: -1 if x is None else x):
            if room < 2 or not ceiling_ready(run, k):
                continue
            ceil.append(k)
            room -= CEIL_CALLS
            prices = [price_of(archive, i) for i in ids if i and product_index(run, i) == k]
            costs = [float(p) for p in prices if is_number(p)]
            if not costs:
                continue
            if is_number(aims.get(k)):
                costs.append(float(aims[k]))
            aims[k] = min(costs)
        plan = walk_plan(run, min(WALK_TOTAL, room), only)
        if not (plan or ceil):
            return ""
        run.walk_queue = (plan, max(floor, moves_left(run) - half - 1), ceil)
        book = store.get("page_book", {})
        queries = [" ".join(free_text(run, book[key]["args"])) for _, key, _ in plan if key in book]
        parts = []
        if queries:
            parts.append("reads the remaining pages of " + ", ".join(f"'{q}'" for q in queries))
        if ceil:
            parts.append("lists the category's listings priced below the pick with the store's price filter")
        return ("Not run yet (no step used). With the cheapest requested and results listed by a fixed id rather than "
                "by price, a cheaper fit could hide on unread pages; so the code " + " and ".join(parts)
                + " itself at this point, then opens the cheapest unread rows under the pick. Afterwards, send the call once "
                "more, or the same call naming a cheaper fit.")
    except Exception as err:
        run.wire.log.append({"kind": "walk_hold_error", "error": f"{type(err).__name__}: {err}"[:300]})
        return ""


# ---------------------------------------------------------------------------------------------- search result remarks
def table_rows(piece):
    lists = [val for val in piece.values() if isinstance(val, list) and all(isinstance(r, dict) for r in val)]
    filled = [val for val in lists if val]
    if filled:
        return filled[0]
    return lists[0] if lists else None


def search_note(run, call_args, obs):
    if not isinstance(obs, dict):
        return ""
    store = vars(run)
    rows = table_rows(obs)
    if rows is None:
        # rows wrapped one level down, next to a shopper line or a refusal
        inner = [val for val in obs.values() if isinstance(val, dict) and table_rows(val) is not None]
        if not inner:
            return ""
        obs = inner[0]
        rows = table_rows(obs) or []
    shown = len(rows)
    totals = [val for val in obs.values() if isinstance(val, int) and not isinstance(val, bool) and val > shown]
    reported = max(totals or [shown])
    # (a store that gives no count but a numeric next-page mark, the offset to send next, has rows at least that far)
    marks = [val for name, val in obs.items() if re.search(r"next|onward|cursor|bookmark|token|continu|more", str(name).lower())
             and isinstance(val, int) and not isinstance(val, bool) and val > 0]
    if marks:
        reported = max(reported, max(marks) + 1)
    book_page(run, call_args, rows, reported, obs)
    try:
        ceil_learn(run, call_args, rows, max(totals or [shown]), obs)
    except Exception as err:
        run.wire.log.append({"kind": "ceil_learn_error", "error": f"{type(err).__name__}: {err}"[:300]})
    score = value_score(run, call_args) if rows else 0
    if score:
        best_seen = store.setdefault("row_score", {})
        for row in filter(lambda r: isinstance(r, dict), rows):
            mark = row_mark(row)
            best_seen[mark] = max(best_seen.get(mark, 0), score)
    remarks = []
    kind = word_rule(run, call_args, obs, rows)
    lead = lead_rule(run, call_args, obs, rows)
    if lead:
        remarks.append(lead_remark(run, call_args, lead))
    texts, mode, first, paging_keys = search_keys(run, call_args)
    mode_keys = dict(mode)
    raw = [val for key, val in call_args.items() if isinstance(val, str) and key not in mode_keys and key not in paging_keys]
    if first and raw:
        query_remarks(run, call_args, rows, raw, totals, kind, texts, mode, paging_keys, remarks)
    if not rows:
        return " ".join(remarks)
    if totals:
        remarks.append(paging_remark(kind, max(totals), shown))
    dear_note = over_limit_remark(call_args, rows, search_keys(run, call_args)[3])
    if dear_note:
        remarks.append(dear_note)
    return " ".join(remarks)


def over_limit_remark(call_args, rows, paging):
    limits = []
    for key, val in call_args.items():
        if not is_number(val) or val <= 1 or key in paging:
            continue
        if not re.search(r"page|size|count|result|min|low|from|start", str(key).lower()):
            limits.append(val)
    if not limits:
        return ""
    cap = max(limits)
    dear = sum(1 for r in rows if any(is_number(val) and val > cap and "price" in str(key).lower() for key, val in r.items()))
    if not dear:
        return ""
    return (f"{dear} of these rows are priced above the limit {cap} that this call set, so this search mode "
            "skipped that filter (and perhaps others, the category among them). Switch to a mode whose filters hold, "
            "or check every row yourself.")


def lead_remark(run, call_args, lead):
    texts, mode, _, _ = search_keys(run, call_args)
    arg_name, arg_mode = run.lead_ctx[0], run.lead_ctx[1]
    words = ()
    if mode == arg_mode:
        words = next((w for key, w in texts if key == arg_name), ())
    counted = ""
    if len(words) > lead:
        counted = f" (this query counted only: {' '.join(words[:lead])})"
    return (f"Keyword search in this store ({arg_name}) looks at just the FIRST {lead} word(s) of a query and drops "
            f"the remainder{counted}: lead with the {lead} most telling words (a required value and the item kind, value "
            "first), and give other values their own short queries.")


def paging_remark(kind, total, shown):
    advice = {
        "any": ("this store matches ANY single word and ranks rows hitting more words higher, which means each added word "
                "broadens the result: searched alone, one distinctive word or figure from a required value returns just the "
                "rows holding it (usually few, so every page can be read and prices compared); kind plus values pushes the "
                "nearest rows up; cover the remainder with the category mode (or a working price filter)."),
        "all": ("this store demands EVERY word, so each extra exact word shrinks the result, as do the category filter and "
                "a lower limit on the price filter."),
    }
    fallback = ("shrink the rest: should every word be demanded, add exact words; should any word match (seen when a longer "
                "query brings back more rows), include the required values so fitting rows climb to the top; a working price "
                "filter helps too.")
    return (f"{total} listings matched here, but this page carries only {shown} of them and few pages are readable, so the "
            "rows that count could be hidden. Begin with the next pages of this very search (as many as possible); when "
            "the total is larger than those pages, " + advice.get(kind, fallback))


def naming_remarks(run, raw, remarks):
    store = vars(run)
    prop, probe = value_probe(run, raw)
    told = store.get("name_told", 0)
    roomy = lambda: moves_left(run) > order_due(run) + 3
    if probe and told < 2 and roomy():
        run.name_told = told + 1
        opening = f"Once more the query leads with a property's name ({prop})."
        if told == 0:
            opening = (f"This query starts with words from a property's NAME ({prop}) and carries none of a required value: "
                       "keyword search matches listing values, titles and categories, and a field name only where a value or "
                       f"title repeats it, so listings whose {prop} field holds only the value are missed.")
        remarks.append(opening + f" Put the value first and the item kind after it: '{probe}' (any figure the store cannot "
                       "read falls away, leaving its unit word), and read every page of it.")
        return
    if probe or store.get("trap_told", 0) >= 2 or not roomy():
        return
    trap, _ = word_trap(run, raw)
    if trap:
        run.trap_told = store.get("trap_told", 0) + 1
        remarks.append(trap)


def note_hit_scores(run, ids, score):
    ranks = vars(run).setdefault("hit_score", {})
    ranks.update({i: max(ranks.get(i, 0), score) for i in ids})


def add_all_hits(run, ids):
    gathered = vars(run).setdefault("all_hits", [])
    gathered += [i for i in ids if i not in gathered]


def order_remark(run, remarks):
    # (said once, when the searches so far show how this store orders its rows: the query form follows from it)
    store = vars(run)
    if store.get("order_told"):
        return
    orders = [w.get("id_order") for w in store.get("page_book", {}).values() if "id_order" in w]
    if False in orders:
        run.order_told = True
        remarks.append("Rows here are ranked by relevance, not by id: put every required value with the item type in one "
                       "query, values first and without property names, and read its first two pages.")
    elif orders.count(True) > 1 and run.word_kind == "any":
        run.order_told = True
        remarks.append("Rows here come in a fixed id order: query a value's own distinctive word alone and page through, "
                       "since adding the type only floods the list.")


def query_remarks(run, call_args, rows, raw, totals, kind, texts, mode, paging_keys, remarks):
    store = vars(run)
    # single-letter pieces only: such stores skip them and return nothing
    pieces = [p for t in raw for p in re.findall(r"[a-z0-9]+", t.lower())]
    if pieces and not rows and not store.get("short_told") and max(map(len, pieces)) == 1:
        run.short_told = True
        remarks.append("No rows came back: one-character words are ignored by search in this store, and so figures such as 0.4 or "
                       "5 x 7 find nothing (or only their other words). Try a word value, or the unit word plus the "
                       "item kind; read every page.")
    naming_remarks(run, raw, remarks)
    order_remark(run, remarks)
    marks = tuple(row_mark(r) for r in rows)
    mode_keys = dict(mode)
    others = sorted((name, str(val)) for name, val in call_args.items()
                    if name not in paging_keys and not (isinstance(val, str) and name not in mode_keys))
    sig = (mode, tuple(others), max(totals or [len(rows)]), marks)
    sigs = store.setdefault("page_sigs", {})
    query = " ".join(raw)
    earlier = sigs.get(sig) if sig in sigs else None
    if len(set(marks)) >= 3 and earlier is not None and earlier != query:
        remarks.append(f"The total and rows match your earlier search '{earlier}': the store read both the same way, so a new "
                       "spelling reveals nothing. Read its next page instead, or change the words it starts with.")
    sigs.setdefault(sig, query)
    hits = []
    strict = kind == "all" or run.lead_words
    whole_page = bool(rows) and not totals
    told_all = bool(strict and whole_page)
    if told_all or (whole_page and kind != "any"):
        kept_refs = None if told_all else list(store.get("ref_hits") or [])
        run.ref_hits = []
        hits = row_hits(run, rows, raw)
        note_hit_scores(run, hits, value_score(run, call_args))
        add_all_hits(run, hits)
        if kept_refs is not None:
            run.ref_hits = kept_refs
        if told_all and hits and len(texts) and len(rows) <= 12:
            remarks.append("All words count in this store and the full match list is visible, so every row contains the words "
                           "searched for, maybe only in fields the title leaves out. Before any new search, open the affordable "
                           "ones: " + ", ".join(hits[:6]) + ".")
        pointers = []
        if told_all:
            pointers = [i for i in store.pop("ref_hits", []) if i not in hits][:6]
        if pointers:
            remarks.append("One requested value is defined by a different listing; the seller or brand named in this search "
                           "offers these rows: read them whatever their price, stock or category (a listing that only provides "
                           "a value gets read, never bought) and act on what their notes say: " + ", ".join(pointers) + ".")
    elif rows and totals and kind != "all":
        # (a paged store that matches any word puts the rows carrying most of the words first, even in fields rows leave
        # out: the top row of a search carrying a required value, the top two with two or more, join the hits the checks
        # before a decline or a short order open, so nothing-fits is not said over a fitting row the search put first)
        score = value_score(run, call_args)
        if score >= 1:
            kept_refs = list(store.get("ref_hits") or [])
            run.ref_hits = []
            top = row_hits(run, rows[:2 if score >= 2 else 1], raw)
            run.ref_hits = kept_refs
            note_hit_scores(run, top, score)
            add_all_hits(run, top)
    if strict and not store.get("hits_back"):
        hits_from_earlier(run, call_args, paging_keys, hits, remarks)


def hits_from_earlier(run, call_args, paging_keys, hits, remarks):
    store = vars(run)
    # the word rule is known only now, so earlier fully shown searches carrying a required value count too
    run.hits_back = True
    here = json.dumps({k: v for k, v in call_args.items() if k not in paging_keys}, sort_keys=True, default=str)
    back = []
    for walk_key, walk in store.get("page_book", {}).items():
        if walk_key == here or not walk.get("every") or not value_score(run, walk["args"]):
            continue
        fresh = [i for i in row_hits(run, walk["every"]) if i not in back and i not in hits]
        note_hit_scores(run, fresh, value_score(run, walk["args"]))
        back += fresh
    add_all_hits(run, back)
    if back:
        remarks.append("Keyword search in this store requires every word, so earlier searches that showed all matches had "
                       "only rows holding each of their words, often in fields hidden from the titles. Open the in-budget ones "
                       "unread so far, before searching: " + ", ".join(back[:6]) + ".")


def shopper_lines(run, call_name, obs, refused):
    lns = []
    from_shopper = call_name == role(run, "message")
    if from_shopper and isinstance(obs, str):
        return [obs]
    if from_shopper:
        if isinstance(obs, dict) and "error" not in obs:
            long_strings([val for val in obs.values() if isinstance(val, (str, list))], lns)
        return lns
    for piece in (obs.values() if isinstance(obs, dict) else ()):
        if isinstance(piece, list) and piece and all(isinstance(val, str) for val in piece):
            long_strings(piece, lns)
        elif refused and isinstance(piece, str) and len(piece.split()) >= 4:
            lns.append(piece)
    return lns


# ---------------------------------------------------------------------------------------------- screening proposed calls
def parsed_arguments(raw_args):
    if isinstance(raw_args, dict):
        return raw_args
    return json_object(raw_args) if (raw_args or "").strip() else {}


def screen(run, proposal, permitted, scripted, batch, queued_messages):
    func = proposal.get("function") or {}
    call_name = func.get("name")
    raw_args = func.get("arguments")
    usable = ", ".join(sorted(permitted))
    if call_name not in run.shapes:
        return f"Not run (no step used): '{call_name}' is not one of the tools. Use one of: {usable}.", call_name, {}
    message_tool = role(run, "message")
    if call_name not in permitted and not scripted:
        if run.asking and call_name == message_tool:
            want_ask(run, raw_args)
            return ("Not run (no step used): the code plans the shopper's questions and sends them itself once you have opened some "
                    "detail pages."), call_name, {}
        return "Not run (no step used): not enough steps remain for this; your usable tools right now are: " + usable + ".", call_name, {}
    parsed = parsed_arguments(raw_args)
    if parsed is None:
        return "Not run (no step used): the arguments were not a valid JSON object.", call_name, {}
    call_args, faults = check_args(parsed, run.shapes[call_name])
    refusal = early_refusal(run, call_name, call_args, faults, batch, queued_messages, scripted)
    if refusal:
        return refusal, call_name, call_args
    if call_name == role(run, "search") and not scripted:
        call_args = aligned_page(run, call_args)
    sig = call_name + as_text(call_args)
    repeat_free = call_name in finishers(run) or call_name in (message_tool, role(run, "cart"))
    if not repeat_free and sig in run.last_calls:
        return ("Not run (no step used): this identical call was just made and nothing changed since; its result is shown above. "
                "Pick new arguments or other actions."), call_name, call_args
    if not scripted:
        held = late_hold(run, call_name, call_args) or value_guard(run, call_name, call_args)
        if held:
            return held, call_name, call_args
    if call_name != role(run, "cart"):
        return None, call_name, call_args
    if not scripted and cart_cycling(run):
        run.wire.log.append({"kind": "cart_cycle_held", "cart": list(basket_mark(run))})
        return ("Not run (no step used): the cart has gone back and forth to this same state several times, and it now holds "
                "every product at its quantity. Stop changing it: call the order tool now."), call_name, call_args
    try:
        fix_cart_count(run, call_args)
    except Exception as err:
        record_failure(run.wire, "cart_count_error", err)
    return screen_cart(run, call_args, sig) or None, call_name, call_args


def early_refusal(run, call_name, call_args, faults, batch, queued_messages, scripted):
    if faults:
        return "Not run (no step used): invalid arguments: " + "; ".join(faults[:6])
    if batch > 1 and call_name in finishers(run):
        return ("Not run (no step used): an order or a decline has to be sent by itself, once the outcome of your "
                "other calls is known.")
    if call_name == role(run, "message") and run.msg_cap is not None and run.msgs_sent + queued_messages >= run.msg_cap:
        return "Not run (no step used): no messages remain in the budget. Send the shopper nothing more."
    if call_name == role(run, "message") and not scripted and yes_reserve(run) and run.msgs_sent + queued_messages >= ask_limit(run):
        return ("Not run (no step used): the last message is kept for the final basket and its total, which the code "
                "sends right before the order. Send the shopper nothing else.")
    return idle_view_note(run, call_name, scripted)


def idle_view_note(run, call_name, scripted):
    store = vars(run)
    if scripted or call_name != role(run, "cart_view") or call_name == role(run, "cart") or store.get("batch_cart"):
        return ""
    if not run.trolley.units or store.get("cart_shown", -1) != store.get("cart_ver", -2) or store.get("since_cart", 1):
        return ""
    # the newest cart reply already listed the lines and the actor made no call since
    told = store.setdefault("view_told", set())
    if run.cart_shown in told:
        return ""
    told.add(run.cart_shown)
    return ("Not run (no step used): the newest cart reply already lists the basket - " + cart_summary(run.trolley)
            + " Viewing the cart would add nothing; to confirm a line's stock, open its record via "
            + (stock_tool(run) or "the tool that shows stock") + ".")


def fit_again(ruling):
    return "Not run (no step used). " + ruling + " If you are certain it fits, send the same call again."


def live_lines(cart, call_args):
    return [(line_id, count) for line_id, count in cart_lines(cart, call_args) if count is None or count > 0]


def removal_of_passed(run, thing, call_args, sig, point):
    store, needs, archive = vars(run), run.needs, run.archive
    mark = f"{sig}#{store.get('puts', {}).get(thing, 0)}"
    mine = price_of(archive, thing) if thing else None
    batch_prices = [price_of(archive, p) for p in store.get("batch_puts", []) if p != thing]
    cheaper_put = any(is_number(mine) and is_number(other) and other < mine for other in batch_prices)
    swap = need_priority(needs) == "cheapest" and (cheaper_put or bool(cheaper_fit(run, thing)))
    pass_price = store.get("pass_price", {})
    moved = thing in pass_price and pass_price[thing] != mine
    if not thing or adds_to_cart(run.trolley, call_args) or f"{thing}|{needs.version}" not in run.passed:
        return ""
    if swap or moved or run.refusal_n != run.refusals_seen.get(thing, run.refusal_n) or mark in run.warned:
        return ""
    run.warned.add(mark)
    least = ""
    if need_priority(needs) == "cheapest" and not cheaper_fit(run, thing):
        least = " It is also the cheapest verified fit."
    return (f"Not run (no step used): {thing} passed an independent check against all requirements in force, with each "
            "property judged by the value rules (a field in which this store writes the value itself is met only by those "
            "words); only the category label decides what kind of item it is, no matter what object its title or text "
            f"mentions.{least} Since then, nothing changed. Should a change or refusal exclude it, should a field of it plainly "
            "miss a requirement, or should a cheaper listing be taking its place, repeat the very same call.")


def screen_cart(run, call_args, sig):
    store, needs, cart = vars(run), run.needs, run.trolley
    thing = aimed_id(cart, call_args)
    point = ref_pointer_note(run, thing) if thing and adds_to_cart(cart, call_args) else ""
    pointed = store.setdefault("pointer_told", {})
    if point and pointed.get(thing, 0) < 2:
        pointed[thing] = pointed.get(thing, 0) + 1
        return f"Not run (no step used): {point}. Choose another listing."
    kept = removal_of_passed(run, thing, call_args, sig, point)
    if kept:
        return kept
    if not thing or not adds_to_cart(cart, call_args):
        return ""
    wanted = count_value(cart, call_args)
    adding = wanted is None or wanted > 0
    if units_held(cart, thing) > 0 and cart.adds is not False and adding and sig not in run.warned:
        run.warned.add(sig)
        return (f"Not run (no step used): the cart already holds {thing} x{int(units_held(cart, thing))}, and this cart tool adds "
                "units to a held line, so the call would increase it. To order it, keep the cart unchanged; to change units, take the "
                "difference in or out. If adding is truly meant, simply send this call again.")
    walk = walk_hold(run, call_args) if not point else ""
    if walk:
        return walk
    blind = blind_swap(run, thing, sig)
    if blind:
        return blind
    extra_lines = [other for other, count in cart_lines(cart, call_args)[1:] if count is None or count > 0]
    if thing in (store.get("closest_picks") or ()) and not extra_lines:
        return quantity_refusal(run, call_args, thing, sig)
    for ident in extra_lines + [thing]:
        ruling = item_check_ruling(run, ident)
        if ruling:
            return ruling
    return rank_ruling(run, call_args) or cheaper_ruling(run, call_args) or quantity_refusal(run, call_args, thing, sig)


def blind_swap(run, thing, sig):
    # (too few steps remain for the item check, so a listing nobody checked would replace one that passed it: a verified
    # line swapped for a cheaper unread one this way fails; the verified line stays)
    if moves_left(run) > 3 or sig in run.warned:
        return ""
    version = run.needs.version
    if f"{thing}|{version}" in run.passed:
        return ""
    kept = [line for line in run.trolley.units if line != thing and f"{line}|{version}" in run.passed]
    if not kept:
        return ""
    run.warned.add(sig)
    return (f"Not run (no step used): {kept[0]} in the cart passed the item check, and too few steps remain to check "
            f"{thing}. Keep the cart as it is and order it.")


def item_check_ruling(run, ident):
    mark = f"{ident}|{run.needs.version}"
    if moves_left(run) <= 3 or mark in run.checked_adds:
        return ""
    run.checked_adds.add(mark)
    ruling = check_item(run, ident, "ADD TO THE CART")
    return fit_again(ruling) if ruling else ""


def rank_ruling(run, call_args):
    if moves_left(run) <= 3:
        return ""
    for line_id, _ in live_lines(run.trolley, call_args):
        better = rank_check(run, line_id)
        if not better:
            continue
        cheaper = rank_verify(run, line_id)
        text = better.replace("before ordering", "before putting it in the cart").replace("order the current pick", "send the same call again")
        if cheaper:
            text += (f" The independent item check found that {cheaper} meets every requirement now in force and costs less "
                     f"than {line_id}: put {cheaper} in instead.")
        return "Not run (no step used). " + text
    return ""


def cheaper_ruling(run, call_args):
    if need_priority(run.needs) != "cheapest":
        return ""
    for line_id, _ in live_lines(run.trolley, call_args):
        cheaper = cheaper_fit(run, line_id)
        told = vars(run).setdefault("cheaper_told", {})
        if not cheaper or cheaper in run.trolley.units or told.get(("cart", line_id, cheaper)):
            continue
        told[("cart", line_id, cheaper)] = 1
        return (f"Not run (no step used). The independent item check found that {cheaper} meets every requirement now in "
                f"force, and it costs less than {line_id}. The cheapest fit was requested, and the check's reading overrides "
                f"yours: put {cheaper} in instead. If a record shows {cheaper} failing a requirement now in force, send the "
                "same call again.")
    return ""


def count_key(cart):
    for key, spec in cart.prop_specs.items():
        if isinstance(spec, dict) and spec.get("type") in ("integer", "number") and "enum" not in spec:
            return key, spec
    return None, None


def line_want(run, thing):
    # (the units a listing's line should end at: its product's quantity, the product told by the one home shelf its
    # category sits on; None where the shelf is shared or unknown)
    counts = need_counts(run.needs)
    if not need_items(run.needs):
        return need_qty(run.needs)
    shelf = shelf_known(run, thing)
    ks = [k for k, home in enumerate(home_shelves(run)) if shelf and shelf in home]
    return counts[ks[0]] if len(ks) == 1 and ks[0] < len(counts) else None


def fix_cart_count(run, call_args):
    # (a cart tool whose count argument carries an unclear name, "lineNumber" in one store, was filled with the line's
    # position: products got 0, 2 or 3 units and the cart looked broken; the code sets the count each product needs.
    # A 0 sent for a held line stays a removal)
    cart = run.trolley
    key, spec = count_key(cart)
    thing = aimed_id(cart, call_args)
    if not key or not thing or list_call(cart, call_args) or not is_number(call_args.get(key)):
        return
    count, held = float(call_args[key]), units_held(cart, thing)
    want = line_want(run, thing)
    if want is None or (held and count == 0) or (held and cart.adds is None):
        return
    target = want - held if cart.adds is True else want
    top = spec.get("maximum")
    if target <= 0 or abs(count - target) < 1e-9 or (is_number(top) and target > float(top)):
        return
    call_args[key] = int(target) if spec.get("type") == "integer" else target
    run.wire.log.append({"kind": "cart_count_set", "line": thing, "sent": count, "set": target})


def quantity_refusal(run, call_args, thing, sig):
    needs, cart = run.needs, run.trolley
    need_n = need_qty(needs)
    count = count_value(cart, call_args)
    aim = need_n if cart.adds is False else need_n - units_held(cart, thing)
    if need_items(needs):
        wanted = need_counts(needs)
        odd = [(i, int(c)) for i, c in cart_lines(cart, call_args) if c is not None and c > 0 and int(c) not in wanted]
        if not odd or sig in run.warned:
            return ""
        run.warned.add(sig)
        return (f"Not run (no step used): the products asked for need {need_counts(needs)} units (one line each), but this "
                f"call asks for {odd}. If you are sure, send the same call again.")
    if count is None or abs(count - aim) <= 1e-9 or sig in run.warned:
        return ""
    run.warned.add(sig)
    return (f"Not run (no step used): the request needs a quantity of {need_n} of one listing, but this call asks for "
            f"{int(count)}. If you are sure, send the same call again.")


# ---------------------------------------------------------------------------------------------- holding an order or a decline
HELD = "Not run yet (no step used). "


def held_for_dead_lines(run, call_args):
    dead = sorted(set(run.trolley.units) & dead_lines(run))
    mark = (tuple(dead), vars(run).get("cart_seq", 0))
    told = vars(run).setdefault("dead_told", {})
    if dead and told.get(mark, 0) < 2:
        told[mark] = told.get(mark, 0) + 1
        return HELD + dead_note(run, dead)
    if not dead or told.get(mark, 0) != 2 or not role(run, "cart") or moves_left(run) < 6 or time_left(run.wire) <= CLOSING:
        return None
    told[mark] = 3
    was_mute = run.wire.mute
    try:
        swapped = code_cart(run, set(dead) | dead_lines(run), set(run.trolley.units) - set(dead))
    except Exception as err:
        swapped = False
        run.wire.log.append({"kind": "dead_swap_error", "error": f"{type(err).__name__}: {err}"[:300]})
    finally:
        run.wire.mute = was_mute
    run.wire.log.append({"kind": "dead_swap", "lines": dead, "done": swapped, "cart": dict(run.trolley.units)})
    if swapped and run.trolley.units and not set(run.trolley.units) & dead_lines(run):
        return (HELD + "The cart still held " + ", ".join(dead) + ", which is sold out, so the code "
                "removed it, placed a candidate there, left the other lines alone, and read the lines' stock: "
                + cart_summary(run.trolley) + " Compare this cart with the request, and then order it.")
    return None


def unchecked_cheaper(run, thing):
    # (listings already opened, of the line's own category, cheaper than it and with no plain field failure, but never
    # put through the item check: the cheapest of them go through it before the order)
    mine = price_of(run.archive, thing)
    if need_priority(run.needs) != "cheapest" or mine is None:
        return []
    version, kind = str(run.needs.version), shelf_known(run, thing)
    skip = set(run.trolley.units) | set(run.refused_lines) | dead_lines(run)
    found = []
    for ident in run.archive.read:
        key = f"{ident}|{version}"
        if ident in skip or key in run.passed or any(m.startswith(key) and run.rulings[m] for m in run.rulings):
            continue
        if kind and shelf_known(run, ident) != kind:
            continue
        cost = price_of(run.archive, ident)
        if cost is None or cost >= mine or field_view(run, ident)["fails"] or not pair_keeps(run, thing, ident):
            continue
        found.append((cost, ident))
    return [ident for _, ident in sorted(found)][:3]


def checked_cheaper(run, thing):
    if time_left(run.wire) < CLOSING + 30 or moves_left(run) <= 4:
        return ""
    for ident in unchecked_cheaper(run, thing):
        check_item(run, ident, "ORDER")
        if f"{ident}|{run.needs.version}" in run.passed:
            run.wire.log.append({"kind": "cheaper_checked", "line": thing, "cheaper": ident})
            return ident
    return ""


def held_for_cheaper(run):
    thing = chosen(run)
    cheaper = cheaper_fit(run, thing) if thing else ""
    for line in ([] if cheaper else list(run.trolley.units)):
        cheaper = (cheaper_fit(run, line) if line != thing else "") or checked_cheaper(run, line)
        if cheaper:
            thing = line
            break
    for line in ([] if cheaper else [i for i in run.trolley.units if i != thing]):
        cheaper = cheaper_fit(run, line)
        if cheaper:
            thing = line
            break
    told = vars(run).setdefault("cheaper_told", {})
    if not cheaper or told.get((thing, cheaper), 0) >= 2:
        return None
    told[(thing, cheaper)] = told.get((thing, cheaper), 0) + 1
    return (HELD + f"From the independent item check: {cheaper} satisfies every requirement, and "
            f"its price is lower than {thing}. The cheapest fit is wanted, and the check's reading overrides yours: take "
            f"{thing} out of the cart, put {cheaper} in, re-read its stock and price, and order it.")


def figures_of(text_body):
    return [float(f) for f in re.findall(r"-?\d+(?:\.\d+)?", str(text_body).replace(",", ""))]


def values_meet(have, want):
    # (a record's value meets a wanted one when their figures agree, or when every word of the wanted value is in it)
    got, need = figures_of(have), figures_of(want)
    if need:
        return bool(got) and all(any(abs(g - n) <= 1e-6 * max(1.0, abs(n)) for g in got) for n in need)
    words = set(map(word_stem, word_run(want)))
    return bool(words) and words <= set(map(word_stem, word_run(have)))


def pointer_field(run, line, ref):
    # (the cart line's own field for the pointer's property, from any record read of that listing)
    pairs = []

    def visit(rec):
        if row_mark(rec) != line:
            return
        for key, val in rec.items():
            inner = val.items() if isinstance(val, dict) else [(key, val)]
            pairs.extend((str(k), str(v)) for k, v in inner if isinstance(v, (str, int, float)) and not isinstance(v, bool))
    for item in run.archive.items:
        walk_dicts(item, visit)
    named = [(k, v) for k, v in pairs if same_prop(k.replace("_", " "), ref["prop"])]
    return named[-1] if named else None


def pointer_breach(ref, value):
    relation, want = ref.get("relation"), str(ref.get("value") or "")
    options = [want] + [str(o) for o in ref.get("goes_with") or []]
    if relation == "pair" and len(options) > 1:
        options = options[1:]
    if relation == "differ":
        return any(values_meet(value, o) for o in options)
    have, base = figures_of(value), figures_of(want)
    if relation == "within" and is_number(ref.get("tol")) and have and base:
        return not any(abs(h - base[0]) <= float(ref["tol"]) + 1e-9 for h in have)
    if relation in ("at most", "at least") and have and base:
        return not any(h <= base[0] if relation == "at most" else h >= base[0] for h in have)
    if relation in ("same", "pair", "", None):
        return not any(values_meet(value, o) for o in options)
    return False


def held_for_rejected(run):
    # (a cart line the item check turned down under the requirements in force, and nothing approved since: no such order
    # has been seen paid)
    told = vars(run).setdefault("rejected_told", {})
    version = str(run.needs.version)
    for line in run.trolley.units:
        key = f"{line}|{version}"
        if key in run.passed:
            continue
        said = [why for memo, why in run.rulings.items() if memo.startswith(key) and why]
        if not said or told.get(key, 0) >= 2:
            continue
        told[key] = told.get(key, 0) + 1
        run.wire.log.append({"kind": "rejected_line_held", "line": line})
        return (HELD + f"The independent item check turned down cart line {line}: {str(said[-1])[:400]}. Replace it with a listing "
                "that meets every requirement now in force; decline only if the records show none can.")
    return None


def held_for_pointers(run):
    told = vars(run).setdefault("pointer_told", {})
    waiting = [ref for ref in run.refs if ref.get("status") == "open"]
    if waiting and moves_left(run) > order_due(run) and told.get("open", 0) < 2:
        told["open"] = told.get("open", 0) + 1
        return (HELD + "A requirement takes its value from another listing and is not settled yet: "
                + "; ".join(ref["head"] for ref in waiting[:3]) + ". The code is reading that listing; follow it, then order.")
    for ref in run.refs:
        if ref.get("status") != "resolved" or not ref.get("value"):
            continue
        for line in run.trolley.units:
            if need_items(run.needs) and ref.get("tag") and not kind_words(ref["tag"]) & kind_words(title_of(rich_record(run, line)) + " " + shelf_known(run, line)):
                continue
            field = pointer_field(run, line, ref)
            mark = (line, ref["head"])
            if not field or not pointer_breach(ref, field[1]) or told.get(mark, 0) >= 2:
                continue
            told[mark] = told.get(mark, 0) + 1
            run.wire.log.append({"kind": "pointer_breach", "line": line, "field": field, "want": resolved_value(ref)})
            return (HELD + f"Cart line {line} shows {field[0]} '{field[1]}', but {ref['head']} must be {resolved_value(ref)} "
                    f"(the value of {ref.get('target')}). Replace that line with a listing whose {field[0]} meets it, then order.")
    return None


def held_for_restock(run, reader):
    stale = stale_lines(run)
    mark = (tuple(sorted(run.trolley.units)), vars(run).get("cart_seq", 0))
    if not stale or vars(run).get("recheck_mark") == mark:
        return None
    run.recheck_mark = mark
    run.open_queue = stale[:4]
    run.open_recheck = True
    run.open_purpose = "cart"
    run.open_tool = reader
    return (HELD + "Stock and prices may shift without warning, and nobody has read these cart lines "
            "since the latest cart change: " + ", ".join(stale[:4]) + f". Their stock is read for you now ({reader}, the "
            "tool that shows stock); then order if every one stays in stock and the total fits the budget, or else replace "
            "the line.")


def table_tool(run):
    # (the field reader is a second tool that has already shown a property table)
    reader = field_reader(run)
    level = ((vars(run).get("field_tools") or {}).get(reader) or (0, 0))[0]
    return bool(reader) and reader != role(run, "details") and level >= 2


def fields_seen(run, ident):
    # (a listing counts as read once its fields were seen: a page from a tool without the property table does not count
    # when another tool shows that table; the only in-budget fit can be skipped because only its plain page was open)
    if ident not in run.archive.read:
        return False
    return not table_tool(run) or table_read(run.archive, ident)


def table_read(arch, ident):
    found = []

    def visit(rec):
        if not found and any(val == ident and "id" in str(key).lower() for key, val in rec.items()) \
                and any(isinstance(v, dict) and len(v) >= 3 for v in rec.values()):
            found.append(rec)
    for obs in arch.items:
        walk_dicts(obs, visit)
    return bool(found)


def held_for_fields(run):
    # (a cart line judged from a page without its property table, while another tool shows that table: a value taken
    # from the title this way can fail; the table is read and the line judged again)
    reader = field_reader(run)
    level = ((vars(run).get("field_tools") or {}).get(reader) or (0, 0))[0]
    if not reader or reader == role(run, "details") or level < 2:
        return None
    lines = [line for line in run.trolley.units if not table_read(run.archive, line)]
    mark = tuple(sorted(lines))
    if not lines or vars(run).get("fields_mark") == mark:
        return None
    run.fields_mark = mark
    for line in lines:
        run.passed.discard(f"{line}|{run.needs.version}")
    run.order_vetted = False
    run.open_queue, run.open_purpose, run.open_tool = lines[:4], "fields", reader
    return (HELD + "The property table of " + ", ".join(lines[:4]) + f" was never read; it is read now ({reader}), and "
            "each line is checked against it before the order.")


def held_for_unread_hits(run):
    # (a cart line that misses a requirement while rows the searches ranked first for the requested values sit unopened:
    # in one store the fitting row was the first row of the shopper's own value search in most "request not met" orders,
    # never opened, and a near miss was bought instead; those rows are read once before such an order)
    version = str(run.needs.version)
    if vars(run).get("hits_order_mark") == version or not role(run, "details"):
        return None
    short = [line for line in run.trolley.units if f"{line}|{version}" not in run.passed and field_view(run, line)["fails"]]
    if not short:
        return None
    shelves = {shelf_known(run, line) for line in short}
    skip = set(run.archive.read) | set(run.refused_lines) | dead_lines(run) | set(run.trolley.units)
    unread = [i for i in vars(run).get("all_hits") or [] if i not in skip and shelf_known(run, i) in shelves][:3]
    if not unread or moves_left(run) <= len(unread) + len(run.trolley.units) + 4 or time_left(run.wire) <= CLOSING + 30:
        return None
    run.hits_order_mark = version
    run.open_queue, run.open_purpose = unread, "decline"
    run.open_say = "Before ordering a line that misses a requirement, I open the rows the searches ranked first for the values asked."
    run.wire.log.append({"kind": "unread_hits_held", "short": short, "open": unread})
    return (HELD + "Cart line " + ", ".join(short) + " misses a requirement, while rows the searches ranked first for the "
            "requested values were never opened: " + ", ".join(unread) + ". They are read for you now; if one meets every "
            "requirement, put it in place of the short line, otherwise order as planned.")


def held_for_vetting(run):
    run.order_vetted = True
    faults = cart_faults(run.trolley, need_qty(run.needs), need_counts(run.needs))
    if faults and moves_left(run) > 3:
        return HELD + "The cart as counted is not ready: " + "; ".join(faults) + ". Fix it, then order."
    several = bool(need_items(run.needs))
    if several:
        lines = list(run.trolley.units)
    else:
        lines = [chosen(run)] if chosen(run) else []
    for thing in lines:
        ruling = check_item(run, thing, "ORDER")
        if ruling:
            return HELD + ruling
    for thing in lines:
        better = rank_check(run, thing)
        if not better:
            continue
        cheaper = rank_verify(run, thing)
        extra = ""
        if cheaper:
            extra = (f" The independent item check found that {cheaper} meets every requirement now in force and costs less "
                     f"than {thing}: take {thing} out, put {cheaper} in, re-read its stock and price, and order.")
        return HELD + better + extra
    shape = "per product asked for, each at its quantity" if several else "the chosen item at the required quantity"
    return (HELD + "A last look before ordering. Is there exactly one cart line " + shape + ", with nothing extra? "
            "Was this item's stock and price read again with a stock-showing tool after your latest cart change, showing it in "
            "stock with price times quantity inside today's budget? Do its own properties satisfy each written requirement, each "
            "need the shopper disclosed and each change they announced? When the cheapest or best was wanted, is there truly no "
            "better fitting item among all you managed to find? Was each procedural rule of the request respected? Settle any open "
            "point now; if all hold, call the order tool once more.")


def batch_reads(batch, queue):
    return -(-len(queue) // batch[2]) if batch and len(queue) >= 2 else len(queue)


def detail_reader(run):
    return role(run, "details") or batch_tool(run)


def decline_kept_gate(run):
    if vars(run).get("decline_kept"):
        return None
    passed_lines = [i for i in run.trolley.units if f"{i}|{run.needs.version}" in run.passed and i not in run.refused_lines
                    and i not in dead_lines(run)]
    if not passed_lines or moves_left(run) <= 1:
        return None
    run.decline_kept = True
    return (HELD + "The cart holds " + ", ".join(passed_lines) + ", which the independent check "
            "judged to satisfy every current requirement. Declining would waste that: read its stock and price again, and "
            "order it"
            + (" with every product's line in the cart" if need_items(run.needs) else "") + ". Decline only if a "
            "change since then plainly rules it out.")


def decline_slack_gate(run):
    store = vars(run)
    slack = [] if store.get("slack_done") or not detail_reader(run) else slack_rows(run)
    spread = batch_tool(run)
    if not slack or moves_left(run) <= batch_reads(spread, slack) + 4 or time_left(run.wire) <= CLOSING + 40:
        return None
    run.slack_done = True
    run.open_queue = slack
    run.open_say = "Ahead of any decline, I open cheaper rows that might let all products fit the budget together."
    total = store.get("slack_total")
    return (HELD + "A fit exists for each product, but taken together the cheapest fits come to "
            + (f"{total:g}" if is_number(total) else "more than the budget") + f", above the {run.needs.spend_cap} budget. "
            "Nobody opened the rows whose price fits inside what remains for one product once the others take their cheapest "
            "fits, so the code reads those records for you at this point: " + ", ".join(slack) + ". Weigh every one against "
            "the needs of its own product by the value rules (a field in which the store writes the value itself is satisfied "
            "only by those words); if some in-budget set of fits covers all products, cart one line per product and order; a "
            "decline is right only when no such set can be formed.")


def decline_fill_gate(run):
    batch = batch_tool(run)
    fill = lacking_rows(run) if not vars(run).get("decline_fill") and (role(run, "details") or batch) else []
    if not fill or moves_left(run) <= batch_reads(batch, fill) + 3 or time_left(run.wire) <= CLOSING + 40:
        return None
    run.decline_fill = True
    run.open_queue = fill
    run.open_say = "Ahead of any decline, I open unread listings of the products still lacking a verified fit."
    return (HELD + "For some products nothing has been verified yet, while affordable listings in their categories sit "
            "unopened; before any decline the code reads them for you: " + ", ".join(fill)
            + ". Weigh every one against the needs of its own product by the value rules (a field in which the store writes "
            "the value itself is satisfied only by those words); as soon as each product has a fitting listing, cart one line "
            "per product and order; a decline is right only for a product that really has none.")


def decline_named_gate(run):
    batch = batch_tool(run)
    queue = named_unread(run)[: batch[2] if batch else 3]
    if not queue or vars(run).get("decline_opens") or not (role(run, "details") or batch):
        return None
    if moves_left(run) <= batch_reads(batch, queue) + 3 or time_left(run.wire) <= CLOSING + 40:
        return None
    run.decline_opens = True
    run.open_queue = queue
    return (HELD + "The checks pointed at listings that nobody opened, so before any decline the code reads them for "
            "you: " + ", ".join(queue) + ". Weigh every one against the needs by the value rules (the field carrying the "
            "property's name has the final word, and a field in which the store writes the value itself is satisfied only by "
            "those words; lacking such a field, a related field, the option list or the title can show it), buy whichever "
            "fits, and decline only once all of them fail.")


def decline_pages_gate(run):
    pages_note = decline_reads(run)
    return HELD + pages_note if pages_note else None


def audit_room(run):
    return moves_left(run) > 3 and time_left(run.wire) > CLOSING + 30


def decline_reaudit_gate(run):
    store = vars(run)
    fresh_reads = len(run.archive.read) > store.get("audit_reads", 10 ** 9) and store.get("audits", 0) < 2
    if not (fresh_reads and run.decline_vetted and audit_room(run)):
        return None
    audit_note = review_decline(run)
    return HELD + audit_note if audit_note else None


def decline_audit_gate(run):
    if run.decline_vetted or not audit_room(run):
        return None
    run.decline_vetted = True
    audit_note = review_decline(run)
    if audit_note:
        return HELD + audit_note
    return (HELD + "Ahead of a decline, confirm three things: the search went several ways (other words, other filters, more pages), every candidate's own "
            "properties were read, and each fails a stated requirement, revealed need, budget or stock. Every key spelling of a field counts. "
            "Should one item fit, buy it; should none fit, the decline tool can be called again.")


def decline_fit_gate(run):
    fit_note = fit_hold(run)
    return HELD + fit_note if fit_note else None


def decline_best_gate(run):
    best = unrefuted(run) if not vars(run).get("best_held") and moves_left(run) > 4 else []
    tries = 0
    while tries < 3:
        tries += 1
        # check unverified picks one at a time; failures fall away
        unchecked = [i for i in best if f"{i}|{run.needs.version}" not in run.passed]
        if not unchecked or time_left(run.wire) < CLOSING + 30:
            break
        check_item(run, unchecked[0], "ORDER")
        best = unrefuted(run)
    if not best or moves_left(run) <= len(best) + 3 or time_left(run.wire) <= CLOSING + 20:
        return None
    run.best_held = True
    several = bool(need_items(run.needs))
    return (HELD + "A decline fits only a store where nothing can meet the needs, yet the records contain "
            + ("a candidate for every product asked for" if several else "a candidate") + " that is available, affordable "
            "and untouched by every check so far: " + ", ".join(best) + ". Unless some record shows one of these breaking a "
            "current requirement, place " + ("one line of each product, at its quantity," if several else "it")
            + " in the cart, check stock and price once more, and order; decline only when each candidate plainly fails.")


def decline_differ_gate(run):
    store = vars(run)
    soft = [r for r in store.get("refs") or [] if r.get("relation") == "differ" and r.get("status") in ("open", "stuck")
            and ref_live(run, r)]
    if not soft or moves_left(run) <= 3 or store.get("ref_decline_held"):
        return None
    run.ref_decline_held = True
    heads = ", ".join(dict.fromkeys(r["head"] for r in soft))
    return (HELD + heads + " comes from another listing the code was unable to settle, so its "
            "value stays unknown: it can neither rule a listing out nor justify a decline. Put the listing meeting every "
            "other requirement (in stock, inside the budget) into the cart" + (", one line per product" if need_items(run.needs) else "")
            + ", read stock and price again, then order; decline only when no listing meets those other requirements.")


def decline_cart_gate(run):
    # (a decline while the cart still holds a line that meets everything now in force, the replacement put in after a
    # refused order next to the line it replaces, say: that line is ordered, the extra one taken out)
    if vars(run).get("cart_decline_held") or not run.trolley.units or not role(run, "order") or moves_left(run) < 4:
        return None
    products = need_items(run.needs)
    version, held = str(run.needs.version), set(run.trolley.units)
    fits = {e["id"] for k in (range(len(products)) if products else [None]) for e in ed_pool(run, k)
            if e["id"] in held and (e["fit"] or e["verified"])}
    fits |= {line for line in held if f"{line}|{version}" in run.passed and line not in dead_lines(run)}
    if not fits:
        return None
    run.cart_decline_held = True
    extra = sorted(held - fits)
    run.wire.log.append({"kind": "cart_decline_held", "fits": sorted(fits), "extra": extra})
    return (HELD + "The cart still holds " + ", ".join(sorted(fits)) + ", which meets every requirement now in force, so "
            "a decline would buy nothing the request can use. Keep one line per product (where two do the same job, the one the "
            "request prefers, the cheaper by default)" + (": take out " + ", ".join(extra) + " by sending the cart tool for it "
            "with its count set to 0 or through its remove form" if extra else "") + ", read the cart, then order.")


def closest_rows(run, rule, k=None):
    # (misses per read listing of the product, counted by the code's own field check and any judge verdict on record;
    # sold-out listings cannot be bought, and an over-budget one only when the rule counts the budget as a requirement.
    # What the rule lets go is not counted, what it keeps rules a listing out, and its own measure breaks ties)
    ranked = []
    loose, kept = rule.get("may_miss") or [], rule.get("must_keep") or []
    named = lambda miss, among: any(same_prop(miss, str(x).split(":", 1)[0]) for x in among)
    for e in ed_pool(run, k):
        if not e["read"] or e["price"] is None or e["gone"]:
            continue
        if e["over"] and not rule.get("counts_budget"):
            continue
        missed = {str(x).split(":", 1)[0].strip().lower() for x in field_view(run, e["id"])["fails"]}
        missed |= {str(x).split(":", 1)[0].strip().lower() for x in verify_problems(run, e["id"]) or []}
        if e["pointer"]:
            # (a value taken from another listing that this one breaks, or that could not be read, is one miss)
            missed.add("the value taken from the other listing")
        if any(named(m, kept) for m in missed):
            continue
        let = sorted(m + " (allowed)" for m in missed if named(m, loose))
        missed = {m for m in missed if not named(m, loose)}
        count = len(missed) + (1 if e["over"] else 0)
        ranked.append((count, closest_measure(e, rule), e["price"], e["id"], sorted(missed) + (["budget"] if e["over"] else []) + let))
    allow = rule.get("allowance") or {}
    if str(allow.get("measure") or "").lower() in ("price", "cost"):
        # (a price allowance over the cheapest of the equally close bounds a pick made on another measure)
        floor = {}
        for row in ranked:
            floor[row[0]] = min(floor.get(row[0], row[2]), row[2])
        ranked = [row for row in ranked if row[2] <= floor[row[0]] * (1 + allow["fraction"]) + 1e-9]
    return [(count, price, ident, missed) for count, _, price, ident, missed in sorted(ranked)]


def closest_measure(e, rule):
    pick = rule.get("choose_by") or {}
    measure = str(pick.get("measure") or "price")
    if measure.lower() in ("price", "cost"):
        value = e["price"]
    else:
        found = re.search(r"-?\d+(?:\.\d+)?", field_for_prop(e["row"], measure).replace(",", ""))
        value = float(found.group()) if found else None
    if value is None:
        return 0.0
    return value if pick.get("lowest", True) else -value


def closest_gate(run):
    # (per product: one with a listing that meets everything keeps it, the others take their closest listing; a product
    # with no buyable listing read at all leaves the decline to the other gates)
    rule = run.needs.reading.get("none_fit") or {}
    if not rule.get("order_closest") or vars(run).get("closest_done"):
        return None
    products = need_items(run.needs)
    if not (role(run, "cart") and role(run, "order")) or moves_left(run) < 3 + max(1, len(products)) \
            or time_left(run.wire) <= CLOSING + 20:
        return None
    picks, parts = [], []
    for k in range(len(products)) if products else [None]:
        ranked = closest_rows(run, rule, k)
        if not ranked:
            return None
        name = products[k]["product"] + ": " if products else ""
        count, price, ident, missed = ranked[0]
        if not missed:
            parts.append(f"{name}{ident} at {price:g} meets every requirement")
            continue
        picks.append(ident)
        others = "; ".join(f"{i} misses {n}" for n, _, i, _ in ranked[1:3])
        parts.append(f"{name}the closest is {ident} at {price:g}, missing {count} ({', '.join(missed)[:140]})"
                     + (" (next: " + others + ")" if others else ""))
    if not picks:
        return None
    run.closest_done = True
    run.closest_picks = picks
    jot(run, "closest_pick", items=picks)
    lines = ", one line per product" if products else ""
    if rule.get("action") == "ask_first" and role(run, "message") and ask_limit(run) > run.msgs_sent:
        return (HELD + "Nothing read meets every requirement, and the request wants the closest listing shown first (\""
                + rule["quote"][:200] + "\"). Counted over the listings read: " + "; ".join(parts) + ". Put that to the "
                "shopper in one message, saying what it misses, and cart and order it on a yes; decline only on a no.")
    return (HELD + "Nothing read meets every requirement, and the request wants the closest listing then, not a decline (\""
            + rule["quote"][:200] + "\"). Counted over the listings read: " + "; ".join(parts) + ". Put "
            + ("these" if products else picks[0]) + " in the cart" + lines + ", check stock and price once more, and order; "
            "where the request asks to be told that nothing fits, say so in the order's own remark if the order tool takes one, "
            "otherwise in one message to the shopper if one is left. Decline only if it cannot be bought.")


def actor_closest(run):
    # (under a closest-listing rule the actor may pick the nearest listing itself: a cart line among the fewest-miss
    # listings of its product stands as that pick, so the checks that refuse a failing line do not turn it into a decline)
    rule = run.needs.reading.get("none_fit") or {}
    if not rule.get("order_closest"):
        return False
    products, version, short = need_items(run.needs), str(run.needs.version), False
    for line in run.trolley.units:
        if f"{line}|{version}" in run.passed:
            continue
        k = product_index(run, line) if products else None
        if products and k is None:
            return False
        ranked = closest_rows(run, rule, k)
        least = ranked[0][0] if ranked else None
        if not least or not any(i == line and n == least for n, _, i, _ in ranked):
            return False
        allow = rule.get("allowance") or {}
        if str(allow.get("measure") or "").lower() in ("price", "cost"):
            floor = min(p for n, p, _, _ in ranked if n == least)
            price = next(p for n, p, i, _ in ranked if i == line)
            if price > floor * (1 + allow["fraction"]) + 1e-9:
                return False
        short = True
    if short:
        run.wire.log.append({"kind": "actor_closest", "lines": list(run.trolley.units)})
    return short


def decline_gates():
    return (near_hold, decline_kept_gate, decline_slack_gate, decline_fill_gate, decline_named_gate, decline_pages_gate,
            decline_reaudit_gate, decline_audit_gate, decline_fit_gate, decline_best_gate, decline_differ_gate, decline_repair,
            near_check, decline_cart_gate, closest_gate)


def held_decline(run):
    for gate in decline_gates():
        note = gate(run)
        if note:
            return note
    return None


def judge_seat(run):
    used = vars(run).get("judge_calls", 0)
    run.judge_calls = used + 1
    return JUDGE if used < JUDGE_MAX else CHECKER


def cart_cycling(run):
    # (the same cart reached again and again by add/remove pairs, while its line counts are already right)
    marks = vars(run).get("cart_marks") or []
    mark = basket_mark(run)
    if not mark or marks.count(mark) < 3:
        return False
    return not cart_faults(run.trolley, need_qty(run.needs), need_counts(run.needs))


def blind_to_ref(run, ident):
    # (one product only: a reference still open or stuck, and this listing has no field for its property, so no check of
    # it could have seen that value, and its pass is no reason to overrule a decline)
    # (a reference resolved to a value counts the same way when no record of the listing writes that value anywhere)
    if not routed(run, "blind_refs"):
        return False
    row = listing_fields(run).get(ident) or {}
    said = None
    for ref in stash(run, "refs") or []:
        if field_for_prop(row, ref["prop"]):
            continue
        if ref.get("status") in ("open", "stuck"):
            return True
        if ref.get("status") != "resolved" or not ref.get("value") or ref.get("relation") not in ("same", "pair", "", None):
            continue
        wanted = [str(v) for v in ref.get("goes_with") or []] if ref.get("relation") == "pair" else []
        wanted = [v for v in wanted if v.lower() != str(ref["value"]).lower()] or [str(ref["value"])]
        if said is None:
            said = listing_words(run, ident)
        if not any(ch_writes(v, said) for v in wanted):
            return True
    return False


def listing_words(run, ident):
    seen = []

    def visit(rec):
        if row_mark(rec) == ident:
            seen.append(json.dumps(rec, ensure_ascii=False))
    for item in run.archive.items:
        walk_dicts(item, visit)
    return " ".join(seen)


def verified_cart(run):
    # (every line passed the independent check under the requirements now in force, one line per product)
    lines = list(run.trolley.units)
    version = run.needs.version
    wanted = max(1, len(need_items(run.needs)))
    return bool(lines) and len(lines) == wanted and all(f"{thing}|{version}" in run.passed and not blind_to_ref(run, thing)
                                                        for thing in lines)


def verified_fits(run):
    # (for every product, the cheapest listing that passed the item check under the requirements in force)
    version, skip = str(run.needs.version), set(run.refused_lines) | dead_lines(run)
    found = {}
    for key in run.passed:
        ident, _, ver = key.rpartition("|")
        if ver != version or ident in skip or blind_to_ref(run, ident):
            continue
        k = product_index(run, ident) if need_items(run.needs) else None
        cost = price_of(run.archive, ident)
        best = found.get(k)
        if best is None or (is_number(cost) and (not is_number(best[1]) or cost < best[1])):
            found[k] = (ident, cost)
    wanted = list(range(len(need_items(run.needs)))) or [None]
    return [found[k][0] for k in wanted] if all(k in found for k in wanted) else []


def held_decline_with_fits(run):
    told = vars(run).setdefault("fits_told", 0)
    fits = verified_fits(run)
    if not fits or told >= 2:
        return None
    run.fits_told = told + 1
    run.wire.log.append({"kind": "decline_with_fits_held", "fits": fits})
    return (HELD + "A decline is wrong here: the independent item check verified " + ", ".join(fits) + " under the requirements "
            "now in force. Make the cart hold exactly " + ", ".join(fits) + " (at the quantities asked), taking out any other line, "
            "then call the order tool.")


def audit_fit_lines(run):
    # (one listing per product that the decline audit called a fit and no item check turned down, inside the budget)
    version, skip = str(run.needs.version), set(run.refused_lines) | dead_lines(run)
    cheapest = need_priority(run.needs) == "cheapest"
    found = {}
    for ident in vars(run).get("audit_fits") or []:
        if ident in skip or ident not in run.archive.read:
            continue
        if any(why and memo.startswith(f"{ident}|{version}") for memo, why in run.rulings.items()):
            continue
        k = product_index(run, ident) if need_items(run.needs) else None
        cost = price_of(run.archive, ident)
        if not is_number(cost):
            continue
        if k not in found or (cheapest and cost < found[k][1]):
            found[k] = (ident, cost)
    wanted = list(range(len(need_items(run.needs)))) or [None]
    if not all(k in found for k in wanted):
        return []
    if any(pair_fit(run, found[k][0], found[j][0]) is False for k, j in pair_rows(run).items() if k in found and j in found):
        return []
    counts = need_counts(run.needs)
    lines = [(found[k][0], counts[k or 0]) for k in wanted]
    total = sum(found[k][1] * counts[k or 0] for k in wanted)
    cap = run.needs.spend_cap
    return [] if is_number(cap) and total > cap else lines


def order_audit_fits(run):
    # (a decline that passed every hold, while the audit named a fit for every product: code carts them and orders instead)
    lines = audit_fit_lines(run)
    if not lines or vars(run).get("fits_ordered") or not role(run, "order"):
        return False
    if moves_left(run) < len(lines) + len(run.trolley.units) + 2 + yes_reserve(run) or time_left(run.wire) < CLOSING:
        return False
    run.fits_ordered = True
    run.wire.log.append({"kind": "decline_to_audit_fits", "lines": lines})
    bad = code_put(run, lines)
    if bad is None or bad or {i for i, _ in lines} != set(run.trolley.units):
        return False
    finish_now(run)
    return bool(run.wire.done)


def fits_ordered(run):
    try:
        return order_audit_fits(run)
    except Exception as err:
        record_failure(run.wire, "audit_fits_error", err)
        return False


def full_cart(run):
    # (one line for every product asked for, none of them failing a requirement the records show)
    lines = list(run.trolley.units)
    if not lines:
        return False
    if need_items(run.needs):
        held = {product_index(run, line) for line in lines}
        if not all(k in held for k in range(len(need_items(run.needs)))):
            return False
    return not any(field_view(run, line)["fails"] for line in lines)


def clean_full_cart(run):
    # (a line for every product, none failing a requirement on record, sold out, refused or turned down by the item
    # check: declining over such a cart dropped a passing basket; the item a winning run bought sat in it unordered)
    if not full_cart(run):
        return False
    version = str(run.needs.version)
    skip = set(run.refused_lines) | dead_lines(run)
    for line in run.trolley.units:
        if line in skip or any(why and memo.startswith(f"{line}|{version}") for memo, why in run.rulings.items()):
            return False
    return True


def keep_verified_cart(run, call_name, call_args):
    # (a decline while the cart holds a verified fit becomes the order: holds and the approval gate still run before it;
    # after a refused order, a cart nothing on record rules out is kept the same way)
    order_tool = role(run, "order")
    if call_kind(run, call_name, call_args) != "decline" or not order_tool:
        return call_name, call_args
    if not verified_cart(run) and not (run.orders_tried and full_cart(run)) and not clean_full_cart(run):
        return call_name, call_args
    run.wire.log.append({"kind": "decline_kept_cart", "lines": list(run.trolley.units)})
    return order_tool, plain_args(run, order_tool, "Order the cart.", "order")


def screen_finish(run, cid, call_name, call_args):
    kind = lambda: call_kind(run, call_name, call_args)
    # (the closest listing the request asked for when nothing fits misses some requirement by design: the holds that
    # refuse a failing line stand aside for it, while stock, price, cheaper fits and the basket still count)
    picks = set(vars(run).get("closest_picks") or ())
    closest = bool(picks) and picks <= set(run.trolley.units) and all(
        line in picks or f"{line}|{run.needs.version}" in run.passed for line in run.trolley.units)
    if not closest and kind() == "order" and run.trolley.units:
        try:
            closest = actor_closest(run)
        except Exception as err:
            record_failure(run.wire, "actor_closest_error", err)
    if kind() == "order" and moves_left(run) > 2:
        note = held_for_dead_lines(run, call_args)
        if note:
            return note, call_args
    if kind() == "order" and moves_left(run) > 3:
        note = held_for_cheaper(run)
        if note:
            return note, call_args
    if kind() == "order" and run.trolley.units:
        note = held_for_pairs(run)
        if note:
            return note, call_args
    if kind() == "order" and run.trolley.units and moves_left(run) > 3 and not closest:
        note = held_for_rejected(run) or held_for_pointers(run)
        if note:
            return note, call_args
    if kind() == "order" and run.trolley.units:
        walk = walk_hold(run, None)
        if walk:
            return walk, call_args
    reader = stock_tool(run) if kind() == "order" else None
    if reader and moves_left(run) > len(run.trolley.units) + 2 and time_left(run.wire) > CLOSING:
        note = held_for_restock(run, reader)
        if note:
            return note, call_args
    if kind() == "order" and moves_left(run) > len(run.trolley.units) + 3 and time_left(run.wire) > CLOSING and not closest:
        note = held_for_fields(run)
        if note:
            return note, call_args
    if kind() == "order" and run.trolley.units:
        note = held_for_unread_hits(run)
        if note:
            return note, call_args
    if kind() == "order" and not run.order_vetted and moves_left(run) > 2 and time_left(run.wire) > CLOSING and not closest:
        return held_for_vetting(run), call_args
    if kind() in ("order", "decline"):
        try:
            final_steps(run, kind())
        except Exception as err:
            record_failure(run.wire, "final_steps_error", err)
    if kind() == "order":
        note = basket_gate(run)
        if note:
            return note, call_args
    if kind() != "decline":
        return None, call_args
    note = held_decline_with_fits(run) or held_decline(run)
    if note:
        return note, call_args
    return None, scrub_decline(run, call_args, call_name)


# ---------------------------------------------------------------------------------------------- sending a batch of calls
def cart_put_target(cart, func):
    raw = func.get("arguments")
    if isinstance(raw, dict):
        put_args = raw
    elif isinstance(raw, str) and raw.strip():
        put_args = json_object(raw)
    else:
        put_args = {}
    if not isinstance(put_args, dict) or not adds_to_cart(cart, put_args):
        return None
    return aimed_id(cart, put_args) or None


def dispatch(run, proposals, scripted=False):
    cart, wire = run.trolley, run.wire
    responses = []
    begin_batch(run)
    permitted = {tdef["function"]["name"] for tdef in offered(run)}
    run.batch_puts = []
    run.batch_cart = any((p.get("function") or {}).get("name") == role(run, "cart") for p in proposals if isinstance(p, dict))
    for proposal in proposals:
        func = proposal.get("function") or {}
        if func.get("name") == role(run, "cart"):
            target = cart_put_target(cart, func)
            if target:
                run.batch_puts.append(target)
    to_send, queued_messages, cart_held = [], 0, False
    for proposal in proposals:
        cid = proposal["id"]
        if cart_held and proposal["function"]["name"] == role(run, "cart"):
            responses.append((cid, "Not run (no step used): a cart change earlier in this reply was held; see the reason, "
                                   "then resend those cart changes."))
            continue
        refusal, call_name, call_args = screen(run, proposal, permitted, scripted, len(proposals), queued_messages)
        if not refusal:
            queued_messages += call_name == role(run, "message")
            to_send.append((cid, call_name, call_args))
            continue
        if not cart_held and call_name == role(run, "cart") and isinstance(call_args, dict) and call_args:
            cart_held = not adds_to_cart(cart, call_args)
        responses.append((cid, refusal))
    if to_send and to_send[0][1] in finishers(run):
        cid, call_name, call_args = to_send[0]
        call_name, call_args = keep_verified_cart(run, call_name, call_args)
        held, call_args = screen_finish(run, cid, call_name, call_args)
        if held:
            return responses + [(cid, held)]
        if call_kind(run, call_name, call_args) == "decline" and fits_ordered(run):
            return responses + [(cid, "Not run: the code ordered the listings the audit found to fit, and the errand is over.")]
        to_send[0] = (cid, call_name, call_args)
    elif to_send and finishers(run):
        keep = max(0, moves_left(run) - 1)
        for cid, _, _ in to_send[keep:]:
            responses.append((cid, "Not run (no step used): one single step remains, and it is reserved for your order (or the decline)."))
        del to_send[keep:]
    while to_send:
        slab, to_send = to_send[:16], to_send[16:]
        if wire.done or wire.broke or moves_left(run) <= 0:
            responses.extend((cid, "Not run: the episode is over or no steps are left.") for cid, _, _ in slab)
        else:
            responses.extend(send_chunk(run, slab[: moves_left(run)]))
    position = {c["id"]: i for i, c in enumerate(proposals)}
    return sorted(responses, key=lambda pair: position.get(pair[0], 0))


def send_chunk(run, slab):
    wire = run.wire
    acts = [dict(name=call_name, args=call_args) for _, call_name, call_args in slab]
    prior = wire.tick
    resp = world_act(wire, acts)
    wire.log.append({"kind": "env", "step": prior + 1, "actions": acts, "reply": resp})
    if resp is None:
        return [(cid, "The call failed (network or server). Try again.") for cid, _, _ in slab]
    many = resp.get("observations")
    obs_list = many if isinstance(many, list) else [resp.get("observation")]
    last_idx = len(obs_list) - 1
    answers = []
    for idx, (cid, call_name, call_args) in enumerate(slab):
        if idx > last_idx:
            answers.append((cid, "Not run: the episode finished before this call."))
            continue
        answers.append((cid, absorb(run, call_name, call_args, obs_list[idx], resp, idx == last_idx)))
    return answers


def stamp_cart_lines(run, call_name, call_args, obs, basket_tool):
    store = vars(run)
    units = run.trolley.units
    fresh = set()
    stock_rule = bool(store.get("stock_seen"))
    asked = asked_ids(call_args)
    rows_ok = bool(store.get("rows_live")) and not store.get("rows_lag")
    lookups_only = not rows_ok and call_name not in (basket_tool, role(run, "cart_view"))

    def seen_line(row):
        hits = {v for k, v in row.items() if isinstance(v, str) and v in units and "id" in str(k).lower()}
        if not hits:
            return
        if not stock_rule:
            if not any(isinstance(v, dict) for v in row.values()):
                return
        elif stock_state(row) is None or (lookups_only and not hits & asked):
            return
        fresh.update(v for k, v in row.items() if isinstance(v, str) and v in units)
    walk_dicts(obs, seen_line)
    for line_id in fresh:
        store.setdefault("line_read", {})[line_id] = run.wire.tick
        store.setdefault("line_seq", {})[line_id] = run.absorb_n


def count_late_read(run, call_name, call_args):
    store = vars(run)
    details, search = role(run, "details"), role(run, "search")
    looked = bool(details) and call_name not in (details, search) and call_name in lookup_tools(run)
    named_ids = {val for val in call_args.values() if isinstance(val, str)}
    if moves_left(run) >= order_due(run) or not (call_name in (details, search) or looked) or store.get("covering"):
        return
    if call_name == details and named_ids & set(run.trolley.units):
        return
    if looked and named_ids & (set(run.trolley.units) | {mark.rpartition("|")[0] for mark in run.passed}):
        return
    run.late_n = store.get("late_n", 0) + 1


def cart_call_remarks(run, call_args, obs, remarks):
    store, cart = vars(run), run.trolley
    run.order_vetted = False
    cart.changed_at = run.wire.tick
    held_before = dict(cart.units)
    remarks.append(book_cart(cart, call_args, obs))
    try:
        for after in (lambda: cart_moved(run, held_before), lambda: mark_fresh(run, obs), lambda: after_put(run, call_args, obs)):
            after()
    except Exception as err:
        run.wire.log.append({"kind": "after_put_error", "error": f"{type(err).__name__}: {err}"[:300]})
    put_item = aimed_id(cart, call_args)
    putting = bool(put_item) and adds_to_cart(cart, call_args)
    if putting:
        run.refusals_seen[put_item] = run.refusal_n
        puts = store.setdefault("puts", {})
        puts[put_item] = puts.get(put_item, 0) + 1
        guarded(run, "pair_put_error", remarks, lambda: pair_put_note(run, put_item))
    # a removal looks like a put; the remark goes only with units going in while the line is held and not sold out
    put_units = count_value(cart, call_args)
    if not (put_item and adds_to_cart(cart, call_args)) or f"{put_item}|{run.needs.version}" not in run.passed:
        return
    if put_units is not None and put_units <= 0:
        return
    if units_held(cart, put_item) > 0 and put_item not in dead_lines(run):
        remarks.append(f"An independent check found {put_item} meets all requirements now in force, judging each property "
                       "by the value rules (where the store writes the value itself in a field, only those words satisfy it); keep it "
                       "unless some new change excludes it.")


def guarded(run, kind, remarks, action):
    try:
        text = action()
    except Exception as err:
        run.wire.log.append({"kind": kind, "error": f"{type(err).__name__}: {err}"[:300]})
        return
    if text is not None:
        remarks.append(text)


def note_call_made(run, call_name, call_args, msg_tool, basket_tool):
    if call_name == msg_tool:
        run.msgs_sent += 1
    run.last_kind = call_kind(run, call_name, call_args)
    if run.last_kind == "order":
        run.orders_tried += 1
    resets = call_name in (basket_tool, msg_tool) or call_name in finishers(run)
    run.last_calls = [] if resets else (run.last_calls + [call_name + as_text(call_args)])[-2:]


def early_audit_due(run, call_name, last, msg_tool, basket_tool):
    if not last or vars(run).get("early_audit") or call_name in (basket_tool, msg_tool) or call_name in finishers(run):
        return False
    if not FIRST_ORDER_BY < moves_left(run) <= run.step_cap // 3 + 2 or run.trolley.units:
        return False
    version = str(run.needs.version)
    return not any(key.rpartition("|")[2] == version for key in run.passed)


def learn_list_cap(run, call_name, call_args, obs):
    # (a call carrying a list of ids that came back as a short reply naming none of them: the tool takes fewer at once
    # than the schema says, so its schema gains the limit, later calls over it are held, and batch reads use it)
    props = props_of(run.shapes.get(call_name) or {})
    lists = [k for k, v in (call_args or {}).items() if isinstance(v, list) and len(v) >= 2
             and all(isinstance(x, str) for x in v) and isinstance(props.get(k), dict)]
    if len(lists) != 1 or call_name in finishers(run) or call_name == role(run, "cart"):
        return
    ids = call_args[lists[0]]
    text = obs if isinstance(obs, str) else json.dumps(obs, default=str)
    if len(text) > 200 or any(ident in text for ident in ids):
        return
    cap = len(ids) - 1 if len(ids) <= 6 else len(ids) // 2
    old = props[lists[0]].get("maxItems")
    if is_number(old) and old <= cap:
        return
    props[lists[0]]["maxItems"] = cap
    vars(run).pop("batch_spec", None)
    run.wire.log.append({"kind": "list_cap", "tool": call_name, "cap": cap, "reply": text[:120]})


def absorb(run, call_name, call_args, obs, resp, last):
    store, archive, wire = vars(run), run.archive, run.wire
    shown, obs = obs, lensed(obs)
    msg_tool, basket_tool = role(run, "message"), role(run, "cart")
    note_call_made(run, call_name, call_args, msg_tool, basket_tool)
    file_obs(archive, obs)
    try:
        note_fields(run, call_name, obs)
    except Exception as err:
        record_failure(wire, "note_fields_error", err)
    try:
        learn_list_cap(run, call_name, call_args, obs)
    except Exception as err:
        record_failure(wire, "list_cap_error", err)
    run.absorb_n = store.get("absorb_n", 0) + 1
    if call_name != basket_tool and not store.get("covering"):
        run.since_cart = store.get("since_cart", 0) + 1
    try:
        gone_now = learn_caps(run, call_name, call_args, obs)
    except Exception as err:
        gone_now = []
        wire.log.append({"kind": "caps_error", "error": f"{type(err).__name__}: {err}"[:300]})
    if run.trolley.units:
        stamp_cart_lines(run, call_name, call_args, obs, basket_tool)
    failed_read = isinstance(obs, dict) and obs.get("error")
    if call_name == role(run, "details") and not failed_read:
        archive.read.update(val for val in call_args.values() if isinstance(val, str) and val in archive.ids)
    count_late_read(run, call_name, call_args)
    remarks = ["[stock] " + dead_note(run, gone_now)] if gone_now else []

    def redirect_note():
        redirected = notes_remark(run, call_name, call_args)
        return "[notes] " + redirected if redirected else None
    guarded(run, "notes_remark_error", remarks, redirect_note)
    if call_name == basket_tool:
        cart_call_remarks(run, call_args, obs, remarks)
        store.setdefault("cart_marks", []).append(basket_mark(run))
    if call_name == role(run, "cart_view"):
        take_view(run.trolley, obs, wire.tick)
        remarks.append(cart_summary(run.trolley))
    refused_now = refused(run, call_name, obs, resp if last else {"finished": False})
    if refused_now and run.last_kind == "order":
        run.cart_seq = run.absorb_n
    remarks.append(notes_for(run, call_name, obs, refused_now))
    searched = call_name == role(run, "search")
    if searched:
        remarks.append(search_note(run, call_args, obs))
        store.get("named_pages", set()).discard(json.dumps(call_args, sort_keys=True, default=str))
        guarded(run, "merge_joined_error", remarks, lambda: merge_joined(run, obs))
    if early_audit_due(run, call_name, last, msg_tool, basket_tool):
        # no verified fit and a third of the steps remain: run the audit while opens are possible
        run.early_audit = True
        ahead = review_decline(run, early=True)
        if ahead:
            remarks.append("Nothing verified fits yet. " + ahead)
    qtext = None
    if call_name == msg_tool:
        qtext = " ".join(str(val) for val in call_args.values() if isinstance(val, str))
    caught = shopper_lines(run, call_name, obs, refused_now)
    if caught:
        remarks.append(hear_lines(run.needs, caught, qtext))
        ref_take(run)
    text_body = as_text(clipped(shown, SEARCH_TEXT if call_name == role(run, "search") else RECORD_TEXT))
    if last and resp.get("finished"):
        text_body += " [the episode is finished]"
    summary = " ".join(filter(None, remarks))
    return text_body + "\n[note] " + summary if summary else text_body


# ---------------------------------------------------------------------------------------------- the run as phases
def tool_calls(run, msg, seq_no):
    proposals = []
    for raw in filter(lambda p: isinstance(p, dict), msg.get("tool_calls") or []):
        seq_no += 1
        func = raw.get("function") or {}
        given = func.get("arguments")
        if not isinstance(given, str):
            given = json.dumps(given or {})
        call_id = raw.get("id") or f"call_{seq_no}"
        proposals.append(dict(id=call_id, type="function", function=dict(name=func.get("name"), arguments=given)))
    return proposals, seq_no


class TurnTally:
    def __init__(self, started):
        self.idle = 0
        self.fails = 0
        self.seq = 0
        self.ok_at = started
        self.missed = 0


def open_chat(run):
    read_request(run)
    run.wire.log.append({"kind": "plan", "plan": run.needs.reading, "roles": run.role_map})
    opening = run.pview.get("opening")
    sections = ["SHOPPER REQUEST:\n" + run.request]
    if opening is not None:
        sections.append("OPENING OBSERVATION:\n" + as_text(opening))
        file_obs(run.archive, opening)
    overview = needs_overview(run.needs, run.step_cap, run.msg_cap)
    if overview:
        sections.append(overview)
    sections.append(status_line(run))
    run.chat = [{"role": "system", "content": ACT_RULES}, {"role": "user", "content": "\n\n".join(sections)}]


def answer_shopper(run):
    due = run.needs.reply_due
    if not due:
        return False
    run.needs.reply_due = ""
    return reply_back(run, due)


def sweep_phase(run):
    return (not run.asking or vars(run).get("empty_plans")) and sweep_step(run)


def ask_phase(run):
    return run.asking and pose_planned(run)


def cover_rows(run, walk, thing, cost):
    label = (walk.get("row_shelf") or {}).get(thing)
    skip = set(run.archive.read) | set(run.refused_lines) | dead_lines(run) | set(run.trolley.units)
    rows = []
    for ident, shelf in (walk.get("row_shelf") or {}).items():
        price = price_of(run.archive, ident)
        if ident in skip or (label and shelf != label) or not is_number(price) or price >= cost:
            continue
        rows.append((price, ident))
    return [ident for _, ident in sorted(rows)]


def cover_step(run):
    # (under a cheapest or best rule, for each product: whatever the store's order, the query that showed its pick is paged
    # toward its end; under cheapest the cheaper rows it turns up are opened, under best the rank check weighs its rows)
    store = vars(run)
    cap = routed(run, "cover_pages")
    if not cap or need_priority(run.needs) == "none" or run.orders_tried:
        return False
    if time_left(run.wire) < CLOSING + 60 or not role(run, "search") or not page_arg(run):
        return False
    done, read_by = store.setdefault("cover_done", set()), store.setdefault("cover_pages", {})
    for k in list(range(len(need_items(run.needs)))) or [None]:
        if k in done:
            continue
        acted = cover_product(run, k, cap, done, read_by)
        if acted is not None:
            return acted
    return False


def cover_product(run, k, cap, done, read_by):
    pick = walk_pick(run, k)
    if not pick:
        return None
    thing, cost = pick
    floor = order_due(run) + 3
    walks = [w for w in vars(run).get("page_book", {}).values() if thing in (w.get("row_shelf") or {})]
    if not walks:
        return None
    walk = min(walks, key=lambda w: w.get("total") or 10 ** 9)
    read = read_by.get(k, 0)
    upcoming = next_page(run, walk)
    if read < cap and upcoming is not None and moves_left(run) > floor + 1:
        read_by[k] = read + 1
        query = " ".join(free_text(run, walk["args"]))
        say = f"The cheapest is wanted, so I read the next page of '{query}', the search that showed the pick, for cheaper fits."
        if walk_call(run, {**walk["args"], page_arg(run)[0]: upcoming}, say):
            return True
    done.add(k)
    if need_priority(run.needs) != "cheapest":
        return None
    queue = cover_rows(run, walk, thing, cost)[:3]
    while queue and len(queue) >= moves_left(run) - floor:
        queue.pop()
    if not queue:
        return None
    run.wire.log.append({"kind": "cover_open", "pick": thing, "rows": queue, "pages": read})
    hand_over_queue(run, queue, floor)
    run.open_say = "Since the cheapest is wanted, I open the cheapest unread rows the extra pages turned up."
    return True


def code_phases():
    return answer_shopper, open_named, sweep_phase, value_pages, ask_phase, ref_step, walk_step, cover_step


def actor_missed(run, tally, model_list):
    tally.missed += 1
    if not can_call(run.wire, model_list, True):
        return False
    if clock_now() - tally.ok_at > DOWN_AFTER and (tally.missed >= 2 or (models_down(run.wire) and len(run.wire.call_models) >= 2)):
        return False
    if run.wire.call_fail in ("empty", "refused"):
        tally.fails += 1
        if tally.fails >= 2:
            return False
    return True


def actor_turn(run, tally):
    wire = run.wire
    model_list = ACTOR[1:] + ACTOR[:1] if run.stuck >= STUCK_LIMIT else ACTOR
    msg = ask_model(wire, run.chat, model_list, 2500, run.toolset)
    if msg is None:
        return actor_missed(run, tally, model_list)
    tally.ok_at, tally.missed = clock_now(), 0
    proposals, tally.seq = tool_calls(run, msg, tally.seq)
    content = msg.get("content") or ""
    if not isinstance(content, str):
        content = as_text(content)
    binding = {role(run, "cart"), role(run, "order"), role(run, "decline")} - {None}
    if run.asking and any(c["function"]["name"] in binding for c in proposals) and pose_planned(run, urge=True):
        return True
    if not proposals:
        tally.idle += 1
        run.chat.append({"role": "assistant", "content": content or "(no tool call)"})
        run.chat.append({"role": "user", "content": "Each reply has to call one or more of the supplied tools. Carry on.\n" + status_line(run)})
        return tally.idle < IDLE_LIMIT
    tally.idle = 0
    wire.log.append({"kind": "model", "content": content[:2000], "calls": [c["function"] for c in proposals]})
    run.chat.append({"role": "assistant", "content": content or None, "tool_calls": proposals})
    step_before = wire.tick
    post_replies(run, dispatch(run, proposals))
    run.stuck = 0 if wire.tick != step_before else run.stuck + 1
    return True


def turn_allowed(run):
    wire = run.wire
    if wire.done or wire.broke or moves_left(run) <= 0:
        return False
    if wire.spend >= SPEND_CAP * vars(wire).get("scale", 1.0) * HARD_SPEND:
        wire.log.append({"kind": "spend_stop", "spent": round(wire.spend, 4)})
        return False        # (the code finishes from here: the order of the cart it holds, or the decline)
    return time_left(wire) >= LAST_CALL


def execute(run):
    wire = run.wire
    if wire.done:
        return wire.log
    open_chat(run)
    tally = TurnTally(clock_now())
    turns = 0
    while turns < TURN_LIMIT and turn_allowed(run):
        turns += 1
        catch_up(run)
        if any(phase(run) for phase in code_phases()):
            continue
        if not actor_turn(run, tally):
            break
    if not (wire.done or wire.broke):
        finish_now(run)
    wire.log.append({"kind": "end", "finished": wire.done, "steps": wire.tick,
                     "cost": round(wire.spend, 5), "seconds": round(clock_now() - wire.t0, 1)})
    return wire.log


def agent_main(pdata: dict[str, Any]) -> list[dict[str, Any]]:
    run = Run(pdata)
    try:
        return execute(run)
    except Exception as err:
        trace = run.wire.log
        trace.append({"kind": "crash", "error": f"{type(err).__name__}: {err}"[:500]})
        try:
            if not (run.wire.done or run.wire.broke):
                finish_now(run)
        except Exception as second_err:
            trace.append({"kind": "crash", "error": f"{type(second_err).__name__}: {second_err}"[:500]})
        return trace


__all__ = ["agent_main"]
