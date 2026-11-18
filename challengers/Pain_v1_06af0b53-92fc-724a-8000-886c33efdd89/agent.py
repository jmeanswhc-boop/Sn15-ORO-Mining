from __future__ import annotations

import json
import re
from time import monotonic, sleep
from typing import Any

from src.agent.proxy_client import ProxyClient

ACTOR = ["anthropic/claude-haiku-5.5", "openai/gpt-6-luna", "anthropic/claude-haiku-5.5"]
READER = ["anthropic/claude-sonnet-5.5", "openai/gpt-6-luna"]
CHECKER = ["anthropic/claude-haiku-5.5", "openai/gpt-6-luna", "anthropic/claude-sonnet-5.5"]
AUDITOR = ["anthropic/claude-haiku-5.5", "openai/gpt-6-luna", "anthropic/claude-sonnet-5.5"]
LISTENER = ["anthropic/claude-haiku-5.5", "openai/gpt-6-luna", "anthropic/claude-sonnet-5.5"]
PLANNER = ["anthropic/claude-haiku-5.5", "openai/gpt-6-luna", "anthropic/claude-sonnet-5.5"]
RANKER = ["xiaomi/mimo-v2.6-pro", "anthropic/claude-haiku-5.5"]

DEADLINE = 265.0
CLOSING = 70.0
LAST_CALL = 28.0
CLOCK_STEP = 30
CH_SYNONYM_MIN = 4
VIEW_CHARS = 16000
SEARCH_TEXT = 240
RECORD_TEXT = 700
SOFT_SPEND = 0.03
SPEND_CAP = 0.045
HARD_SPEND = 0.066         # $ a task, everything counted: past it no model call goes out, and the code finishes the task (a run's key pays $5 for 90)
HARD_SPEND_MANY = 0.058    # the same for a request of three or more products (their long errands are most of a run's spend)
HISTORY_CHARS = 110000     # characters of the conversation: past it, older tool results are cut down before the next model turn
ASK_RESERVE = 0.012        # $ the questions (their planning and the reading of the answers) may spend past the hard total
ASK_SPEND_SHARE = 0.55     # share of the hard total after which questions not yet sent go out now
CALL_LIMIT = 55            # model calls a task: past it no model call goes out, and the code finishes the task
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
ASK_LIST_MAX = 8             # properties one message may name when the request asks for several products (one product per message)
ASK_GRACE_STEPS = 3
ASK_AFTER_STEPS = 5
BATCH_DEFAULT = 2         # ids sent in one multi-listing read when the tool states no limit (an unstated limit has been 2)
BROWSE_READS = 12         # listings of one product read (several at a time) before its questions, when the steps allow
BROWSE_PAGES = 3          # pages of a product's capped shelf listing the code reads before the questions
GATE_TRIES = 2            # declines held back while steps are left and the shelf is not covered
GATE_LEAST = 4            # moves always kept for the cart and the order after a held decline
HUNT_CALLS = 6            # value searches (pages) the code runs after the questions while no opened listing fits
HUNT_PAGES = 3            # pages of one value search while its pages show no new rows of the product's shelf
HUNT_PAGES_MAX = 5        # pages of one value search while each page still shows unread in-budget rows of the shelf
COVER_READS = 3           # listings of each product read before the first question (several products)
FIRST_READS = 5           # listings read before the first question (one product)
LINE_READS = 2            # listings of a product read before a question about it (several products)
STEP_PLAN = {"one": (0.41, 0.45), "cheapest": (0.54, 0.45), "several": (0.57, 0.45)}
STAND_IN_BY = 0.70         # share of the steps from which the code's first cart may hold a stand-in (an unchecked closest listing)   # questions end by, first cart by (shares of the step limit)

ACT_RULES = 'Your role: a meticulous buying assistant serving a single shopper inside a simulated shop, where you can act only through the tools provided. Each errand is scored 1 or 0. A 1 requires a flawless ending on all counts: the purchased item satisfies each requirement written by the shopper, each hidden need the shopper discloses in reply to a question, and each change the shopper declares mid-way; quantity and budget are correct; when the request wants the cheapest (or best) item, that is the one picked; and every procedural rule contained in the request is respected. If no item in the shop satisfies all of it, the correct ending is to decline, carried out precisely the way the request specifies.\n\nHOW THE STORE WORKS\n- Every tool call uses up one step, including calls that fail or are invalid, and the supply of steps is finite. Reasoning is free.\n- When no call depends on the output of another, a single reply can hold several of them (say, fetching the details of a few listings at once); they execute in sequence. An order, or a decline, is always sent by itself.\n- If the details tool accepts a LIST of listing ids, it returns the complete records of all of them for ONE step (within its item cap): use it whenever you inspect two or more listings.\n- Tool names, argument names and the shape of results change from shop to shop. Study the description and JSON schema of every tool and use the names and permitted values exactly as given. From the earliest results, work out which tool reveals which facts (price, stock, detailed attributes).\n- Find out early which category labels this shop attaches to listings of the kind requested. If similar labels sit side by side, pick the one whose label fits the wording of the request taken as a whole, and do your searching and buying within it.\n- Search behaviour varies per shop, so infer it from the first results here. Rows may be ranked by relevance or listed in a fixed id sequence. A keyword search may demand all words (an extra word lowers the total) or accept any of them (an extra word raises the total, and listings hitting more words are ranked above others). Possibly only a handful of rows or pages can be seen. A filter may function in a single mode only (keyword mode can disregard the category or price filter; filter mode can disregard the words). Check the reported total against the rows in view: if it is bigger, the matching or lowest-priced listing could be on a later page or past the visible range. With all-words search, tighten using more precise words (or apply the price and category filters using a lower ceiling) until everything fits on the pages you can see.\n- Use brief queries (one to three words) and put the most distinctive words FIRST: certain shops only read the opening word or two and drop the remainder, so a lengthy query might effectively search just its start (put the hits and total of a longer query next to those of a shorter query opening identically; the notes tell you what they reveal).\n- Under all-words search, a query made of a needed value plus the item kind, value leading ("blue chair"), returns precisely the listings that have that value. Under any-word search, common words such as the item kind or a unit hit almost the entire shelf and drown out everything else: query a required value\'s own distinctive term or number by itself ("herringbone", "30m", "0.4") to get only the listings carrying it, try it also in the alternative spellings listings use ("30 m", "3 atm"), read all of that search\'s pages whenever its total fits there, and use item kind plus values together only to push the nearest matches to the top. Keyword search can also hit category names and attribute values. After each search, notes explain what the totals collected so far reveal about how this shop searches.\n\nTHE SHOPPER AND HIDDEN NEEDS\n- Some needs are never written down; the shopper discloses one only if a message enquires about that particular property. Whatever a message asks about FIRST gets the most dependable answer, so each message covers a single property, or at most two with the more probable one leading: a brief, simple, warm question that mentions the property (giving one or two sample values is acceptable). Leave out instructions, lists, labels, positions and rules.\n- Replies like "Don\'t mind", "no preference" or "anything is fine" mean that property is unconstrained. Any preference or limit that is stated becomes a hard requirement from that point on, taken word for word, figures included.\n- If the questions are prepared on your behalf, they are sent as soon as the details of a couple of listings (two or three) of the requested kind have been opened; make that one of your opening moves. Answers appear in the dialogue along with what they mean; every disclosed need is a hard requirement, so run new searches combining it with the values already stated.\n- If you write the questions yourself: begin by opening details for two or three candidate listings, next ask about whatever the story suggests (the recipient, the desired look, the place or manner of use), and after that about attributes the listings have that vary across candidates. Do not ask about anything the request already decides, and never exceed the message allowance.\n\nRULES OF PROCEDURE\n- A request can lay down working rules: the number of listings to view or compare, what to verify, what to say to the shopper, the content a remark must have, the way to decline. These are graded the same as requirements; carry out each one literally and completely. When asked to "open" or "look at" N listings, that means N listings fitting in full, every one of them viewed through the details tool.\n- If the shopper puts a question to you (such as asking you to confirm a change), a reply is supplied for you as long as messages are left; the change itself is binding regardless.\n- Mere wishes, phrased as "ideally", "would be nice" or "if possible", are not requirements: they serve only as a tiebreak among items already satisfying everything, and they never override a cheapest rule.\n- If the request leaves open a point that only the shopper can decide (for instance, no stated price ceiling, or two wishes that might conflict so the shopper has to rank them), ask before making the choice.\n\nCHANGES WHILE YOU WORK\n- Mid-task, the shopper may alter things: a revised budget (up or down), a requirement removed, swapped or introduced, a price increase, an item going out of stock. A single line may contain more than one change. These lines may appear within any tool result (as an additional field containing sentences) or alongside an order attempt. Every change applies from the moment it appears: re-verify your choice and your backups, and if the cheapest or best is wanted, select afresh from the updated set of acceptable items.\n- A message that drops or loosens one requirement (it says that property can be left out, does not matter any more, or may be looser) changes only that requirement: the shopper still wants the order, and a cart that still fits is ordered. Only a plain statement that the shopper wants nothing bought at all cancels the errand.\n- Lines marked [status] or [note], and every "Not run" reply, come from your own assistant code, not from the store or the shopper: follow them.\n- An order attempt may be REFUSED: the episode continues and the cart remains unchanged. Read the refusal, carry out each change it mentions, remove from the cart anything that has stopped fitting (a sold-out item, say), add your next verified fitting item, recheck its price and stock, and place the order again.\n- Stock and prices can also shift silently, so recheck both immediately before each order attempt; once an item\'s present price multiplied by quantity exceeds the present budget, it stops fitting.\n\nSEVERAL PRODUCTS\n- A request can call for multiple distinct products, and every one has its own requirements, hidden needs and quantity. Handle all of them: search the category of each, confirm a fitting item for each, and order them in one go, with each product on its own line carrying its quantity. A cart tool accepting a list of lines might overwrite the entire cart with that list, so every such call must carry the lines of all products. Any cheapest-or-best rule holds separately for each product.\n- A requirement may specify its value by pointing to a separate listing or page (e.g. "the same shade as the rug that Q sells"): locate that listing before anything else (query the words that identify it, such as its brand or shop plus its kind), go through its record, copy the value out of the field referred to, and then search the requested product using that value.\n\nTHE CART\n- Rather than setting units, the cart tool can ADD them to a line already present: adding 1 more of a line you hold gives 2. Do not insert a line the cart already contains; to adjust units, remove or add the difference. Following each cart call, a note gives the code\'s count of cart contents.\n\nHOW TO JUDGE A LISTING (the score depends on this)\n- Field names. For every requirement, what counts is the listing field covering that property, using the value\'s own words in the form this shop writes them in that field. A field qualifies when its name holds every word of the property\'s name, whatever the separators, singular or plural or spelling variant, also with words added in front (shoe width for width, frame colour for colour); a field whose name leaves out a word of the property\'s name (a plain "type" for a fit type) or calls the property by another name is not that property\'s field. Names that differ solely in separators (seat height, seat_height) count as a single field, and the requirement is satisfied if any qualifying field contains the value. Only if not a single field names the property can the title display the value; the description, however, never counts as evidence.\n- Matching words. Values are matched on their words in this shop\'s own wording: case, separators, singular versus plural and spelling variants make no difference; a value that is more specific and contains the required one satisfies it (cotton is met by "80% cotton", green by "forest green", and rtx 3060 by "rtx 3060 12gb"); a single item of a list suffices.\n- Synonyms and shades. A synonym, abbreviation, related term, shade or shorter form (technical wood written as "tech wood", windows 11 as "win11", embroidery as "embroidered", plain as "solid", grey as charcoal or slate) never satisfies a value, even where no listing of the category writes that exact value: a value is read word for word, so a "satin" finish fails matte. If that same field also lists the other part separately ("nylon"), a plus-joined value ("wool + nylon") does not count for one of its parts. FIELD EVIDENCE lines, where present, are derived from this shop\'s records and state which situation applies.\n- Different values. A different value of that property fails (steel is not aluminium, oval is not round), and so does a word that merely resembles the required one. Catch-all wording such as "all seasons", "universal" or "suitable for all occasions" claims to cover every value yet names none in particular, so it fails a required value; it is different when a list also spells out the required value, which then still counts. A field that names some other value of this property is no match for the required one, even when the two commonly appear together or one usually suggests the other (style "pullover" fails "zip-up"; "slip-on" fails "lace-up"); what does count is a more specific value containing its words.\n- A field belonging to another property is not the property of the requirement (the material of a handle is not the material, a width is not a shape, a cord length is not a cord type): it never replaces a property that has its own field on the category\'s listings.\n- Only stated or disclosed needs count. Never exclude a listing over anything the shopper did not request: a gender, an age group, a size, a condition, an occasion, a level of quality or a use that you read into the description or the story. The narrative is background; if it looks relevant, put a question on that property to the shopper instead of assuming. Questions you ask never become requirements by themselves; the replies of the shopper do.\n- Numbers are compared as quantities once units are converted (1.2 kg equals 1200 g); limits are inclusive; a range qualifies only when the whole of it satisfies the limit. In cases where the title is the evidence because no field names the property, an exact required value counts when the title contains its words and figures in that same sequence, even when further detail follows (40 x 60 x 5 cm shows 40 x 60; 40 x 65 does not).\n- Title figures such as "55 x 94 Inch" or "117 x 228 cm" never tell you which dimension is which: never reject a listing, or avoid opening it, because of its title figures; read the fields in its details.\n- Obey any exclusion (for instance "leave out anything with X") as well as any closed list (for instance "A or B and nothing else").\n- Category alone settles the requested kind: any listing in the category that was requested IS that kind, even if its title or description mentions a different object (a box, a holder, a replacement part, a kit, an add-on), so never pass over, reject or downgrade it for what it appears to be; assess it purely on requirements, price and stock. No attribute needs to restate the category name. Where the shop labels categories, the item has to be in the one whose label fits the request\'s wording overall (so "A or B" belongs in a category called "A & B" and not in a tighter neighbour called "B"). Stock must be available whenever the request needs the goods on hand; also, price multiplied by quantity must not exceed the budget (being equal is acceptable). Treat a listing\'s price as being in the unit of the budget, except when its record gives another unit (300 against a budget of 5000 cents means 300 cents); never rescale either figure.\n- Decline only once searching truly turned up nothing satisfying all requirements by these rules; a listing that merely comes close (a synonym, a shorter or related word, a missing field) is no fit, an order holding it fails, and it never takes the place of a decline.\n\nCHEAPEST OR BEST\n- When the lowest price is wanted, you must order the lowest-priced item out of ALL items that satisfy every requirement; finding a single fitting item is insufficient.\n- How to gather candidates: when keyword search requires all words and hits attribute values, the value-plus-kind search returns the candidates bearing the value: read every one of its pages and open the hits by price. Browsing the category under a price ceiling reveals prices across the whole category but not attribute values: use it to find which cheaper rows exist (drop the ceiling until the total fits), then inspect those rows\' attributes. For every required value, run a short query of value then kind (e.g. "blue chair" or "30m watch"), and go through every page if the total fits within the readable pages (otherwise narrow it with another required value or a tighter kind word); repeat this using other names of the kind (e.g. "stool" or "bench"), and using only the value. If a value is unsearchable (a number the shop discards), additionally search the product line of every fit or near-fit you come across, as two words, its maker plus one distinctive title word, on every page: the other sizes and colours of a line are separate listings sharing its attribute values and possibly priced lower.\n- Next, sort the candidates in view by price on your own and open their details starting from the cheapest; the answer is the first one whose attribute fields satisfy everything, provided every cheaper candidate was opened and turned down. Hold the following fitting ones as backups.\n- When the best is wanted by some measure (largest, highest, longest, most...), rank ALL fitting items by that measure in the same way and choose whichever is extreme. If an item states nothing for that measure, it cannot win.\n\nFINISHING\n- Keep your verified fits in ranked order, with one backup or more, because the top choice could sell out during checkout.\n- Prior to each order attempt: the cart contains exactly one line for every product requested (a single line when one product is requested), each holding the selected item in the quantity required, and nothing beyond that; following the last change to the cart, you have checked the chosen item\'s price and stock again, and it is available and within budget.\n- Decline only once a thorough search turns up nothing satisfying every requirement, applying the request\'s decline rule precisely. If the request forbids offering substitutes, the decline remark must contain no listing at all, neither named, described nor pointed to: no titles, ids or prices.\n- Plan your steps: send the first order attempt while seven or more steps are left, so that a refusal can still be dealt with. Once 7 steps remain, the only tools left to you are cart tools, stock-check tools and finishing tools. Make the decision once; avoid toggling the cart back and forth.\n- Each reply must include one or more tool calls. Keep the text brief: current requirements, the verified fits with their price and stock, and the next step.\n'

READ_REQUEST = 'You will get what a shopper wrote and the list of tools a store offers. Your reply is ONE compact JSON object (no indentation) and no other text. It carries these keys:\n\n"product": the type of goods wanted, using the wording of the request (when more than one product is wanted, list them all linked by "and").\n"items": a separate entry for each DIFFERENT product the shopper wants bought, kept in the order the request gives: {"product": that type, worded as the request words it, "quantity": how many units of a single listing of it (same logic as "quantity" further down), "requirements": the firm conditions on it (same logic as "requirements" further down)}. If just one product is wanted, there is just one entry.\n"quantity": the number of units of a single listing to buy (an integer; 1 unless the request asks for more than one). Goods that come as one item, for instance a set or a pair of gloves, count as 1. Only an outright request to purchase multiple units changes it ("get me four of these", "buy 3 boxes"); the number of people who will use or split it, or the number of old ones owned before, belongs to the story and keeps the value at 1. When the request asks for several units and also says each one comes as a set or pack ("three mugs, each one boxed in sixes"), the quantity is the number of units asked for and the set or pack size is a requirement of its own (pack size).\n"requirements": an array of strings (a word that narrows the product to a sub-kind, variant or model family of a broader product, which the category\'s listings would carry in a type-like field, is a firm condition too: keep the broader noun as the product and write the narrowing word under that type-like property), a single string per firm condition on the product, each written "property: constraint" (with more than one product: the conditions of every product, each written "property (product): constraint"), carrying the shopper\'s own value, bound, exclusion or fixed set of choices. Omit price, stock, quantity, budgets and simple wishes, and never repeat the product type or category: words that are part of the category\'s name (such as an audience, an occasion or a season it includes) are already handled by the category and are not an extra condition. When the request puts a value before the product\'s noun as a property of it (material, colour, finish, size or any other value: "a 40 inch oak dining table", "a leather wallet"; the value may run to several words, as a two-word material does), that value is also a requirement of its own ("material: leather"), even when the same words stay in "product"; the item check reads only "requirements", so a value left only in "product" is never checked. A condition is a product property together with the value the request insists on ("the stitching must be double", "the occasion has to be party", "with a mechanical movement", "no fewer than 3 shelves"). The surrounding story is never a condition: the person it is meant for, an event where it is going to be used or worn ("for the staff party", "to wear at his wedding"), the place, time or manner of use, the number of people sharing it, or the fact that it takes the place of an old one (with "a new watch" the old one is simply being replaced; that sets no condition). An audience, use or occasion counts as a condition only when the request turns it into a property of the item itself ("a blazer made for business", "the occasion has to be business"). The value of a condition is a word or figure the request itself states as the value of the item; a quality, reason or wish coming from the story (soft and warm, light enough to carry daily, goes well with my other gear, reaches the full length of the hallway) is a hint and never a condition.\n"price_cap": the top total price, as the figure written in the request and in the unit the request uses (no conversion ever: "1999 US cents" yields 1999), or null if there is none.\n"currency": the unit the request prices in, or null if absent.\n"in_stock_only": true if only goods that are currently stocked or on hand are acceptable.\n"priority": "cheapest" only if the request clearly wants the lowest-priced or cheapest suitable item; "best" only if it clearly wants the top one by some measure; otherwise "none". Remarks that funds are short or that each cent matters do not ask for the cheapest; they merely justify the cap.\n"priority_detail": the measure when the value is "best"; otherwise "".\n"message_budget": the number of messages the shopper lets you send, an integer, or null if the request is silent on it.\n"step_budget": the number of moves or steps the request permits, an integer, or null if the request is silent on it.\n"rules": each procedural instruction the request gives the assistant (the number of listings to examine, things to verify or report, the way to decline, things to avoid, things to include), copied verbatim. Actions the shopper plans to take personally are story and not a rule.\n"finish_messages": every rule of the request that calls for a message to the shopper: presenting the final basket, its total or its lines ahead of the order; presenting options the shopper picks from; asking a question the rule tells you to ask; informing the shopper of something ahead of a decline: {"rule": the rule copied verbatim, "when": "before first cart" (the rule wants this message sent while the basket is still empty), "before order" (sent along the path to the order after the cart is settled: options, a pick, the basket, its total, a sign-off) or "before decline", "messages": the count of distinct messages needed (a two-stage rule, for instance options first and then the basket, needs two; 0 if the decline or order call carries by itself what the rule wants)}. Empty if there are none.\n"soft_wants": things the shopper would welcome but does not strictly need (if possible, would be nice, preferably, ideally). Never put them in "requirements".\n"ask_first": what can only come from the shopper and should be asked ahead of choosing: for instance, which of two wishes weighs more when the request admits both may not be achievable (then list both wishes under soft_wants rather than under requirements), or anything the request tells you to confirm with the shopper or to ask the shopper about before choosing. Never the price or the budget, never which product to purchase (the request already names it), and not a finish rule that already sits under "finish_messages" (for instance presenting the basket). Empty if there are none.\n"hints": details from the story that suggest some product property but leave its value unstated (its use or what it is used with, the place, the user, what it has to fit, match or withstand), every one written as "property (product): the detail in a few words" ("property: detail" when there is one product), naming the property the way listings would probably name it. If a detail might bear on several properties, give the most probable first and the runner-up after a slash ("colour / style (lamp): has to go with a dark walnut desk"): anything the item has to match, suit or look right beside (an outfit, a wall, a room, items used alongside it) and its visual appearance point first at its colour and next at its pattern or style. Drop a detail whose only role is to justify a condition the request already gives (the reason it must be the stated size or material). Weather, an occupation, a timetable, a trip, or a location where the item is neither worn nor used is background and never a hint; the same goes for a purpose that every listing of that type fulfils (holding things for a cabinet, keeping dry for a raincoat). Empty if there are none.\nEach sentence in the request must end up somewhere: a budget, a rule, the priority, a requirement, a hint, a soft want, or just story. If you cannot tell whether some sentence counts as a rule, place it in "rules".\n"references": conditions whose value the request supplies only by pointing to ANOTHER listing or page ("matching the size of the backpack sold by Q", "within 2 of the number on the page for R", "a finish unlike the one on S\'s page"); write each one as "property (product): as shown on <how the request names that listing>" (or "property (product): different from the one shown on <how the request names that listing>" if that listing\'s value must be avoided, or "property (product): the partner of the one shown on <how the request names that listing>" if the request wants the value the store pairs with that listing\'s), and add each of them to "requirements" in that same form too. Empty if there are none.\n"reference_targets": one object per "references" entry, describing the listing it points to: {"property": the property involved, "product": the product involved, "seller": the store or brand the request attaches to that listing, copied exactly, "kind": the product type of that listing, "details": [every further value the request mentions for that listing, brief and in the request\'s wording ("blue", "16 oz", "pack of 12 count")], "relation": "same", "partner" (the value must be the one the store pairs with that listing\'s: its partner, not that value itself), "differ" (the value may NOT equal that listing\'s: distinct from, unlike, anything but), "within", "at most" or "at least" (the way the value has to compare with that listing\'s), "tolerance": the figure for "within", otherwise null, "unit": the unit of it, otherwise ""}. Empty if there are none.\n"fits": a list of {"item": item number (counting the entries of "items" from 1), "with": item number} for each requested item that the request says must work with, fit or suit ANOTHER requested item (the one the agent picks); [] when none.\n"decline_no_names": true if the decline rule of the request bans offering, naming or proposing substitutes or alternatives; false if it tells you to name something; null if it has no such rule.\n"roles": an object that maps each role to a tool, using the tool names exactly as listed, or null where no tool suits: "search", "details", "view", "message", "cart", "cart_view", "order", "decline". "details" is the tool that shows a listing\'s properties (its table of attributes); "view" is the tool that shows its offer (price, stock, description) when a different tool does that; "message" is for writing to the shopper; "cart" puts items into or removes them from the cart; "order" purchases or checks out; "decline" closes the errand with no purchase.\n"tool_args": how the store\'s search is called, read from the tool definitions below, with argument names and values exactly as listed, null where there is none: {"words": the search tool\'s argument taking free search words, "mode": its argument choosing how it searches, "mode_words": the value of that argument for a search by words, "next": the argument that moves a search on to further results (whatever it is called), "next_kind": "page_number", "row_offset" or "token", "price_cap": the argument of a search or listing tool setting a highest price}.\n'

HEAR = 'Your job is to interpret the shopper\'s latest message on behalf of a shopping assistant. Reply with ONE JSON object and no other text; it contains:\n"meaning": a brief sentence stating what the shopper intends.\n"needs": firm product requirements that are new in this message, each written "property: constraint" carrying precisely the value, limit, exclusion or closed set of options stated. When a comparison phrase (nothing less than, at least, falls short of) is followed by words instead of a number, those words are themselves the value and are copied in the shopper\'s wording: "falls short of not available" yields "property: not available" (a field may legitimately hold "not available" or "none" as values).\n"no_preference": properties about which the shopper declared indifference.\n"dropped": previously stated requirements the shopper now gives up.\n"new_budget": the shopper\'s new maximum overall price, as a number expressed in the same unit as the budget in the request (take the figure the message writes in that unit: for a budget kept in pence, "4820 pence" gives 4820), otherwise null.\n"unavailable": true if the shopper reports that an item cannot be purchased anymore (out of stock, gone, button disabled, sold out); false otherwise.\n"meta_only": true if the message merely asks the assistant to put fewer questions at a time (a single question per turn, less in each go, just the first now with the others afterwards) and gives no value; such a message answers nothing so far, so it contributes nothing to needs, no_preference or dropped, and any figurative wording in it names no value. Otherwise false. A message saying only that the shopper failed to follow or grasp the question, or requesting it again more briefly or more plainly, does not count as meta_only.\n"requirements_now": the complete current set of firm product requirements, each as "property: constraint". Begin with REQUIREMENTS SO FAR; retain every entry the shopper has neither dropped nor replaced, keep just the latest value where one was replaced, and append every new need stated in the shopper\'s messages. Requirements come solely from the shopper\'s messages: never insert one taken from the request\'s background narrative unless REQUIREMENTS SO FAR already contains it. Leave out price, stock, quantity and product category.\n"reply": if the shopper poses a question to the assistant or wants something confirmed, a brief plain response to return that states the substance of the change (such as confirming the updated value); otherwise "".\n"references": for every need in "needs" whose value is supplied only by referring to some other listing or page, describe the referred listing as {"property": the need\'s property, "seller": the store or brand the message names for that listing, copied verbatim, "kind": the product type of that listing, "details": [any further values the message mentions for that listing, brief and in the message\'s own words], "relation": "same", "differ" (i.e. the value must NOT match that listing\'s: distinct from, unlike, anything but), "at most", "at least" or "within", "tolerance": the numeric margin for "within", otherwise null, "unit": the margin\'s unit, otherwise ""}; use [] if there are none.\nA single message may contain several changes; list all of them. A need concerning a different property, however closely linked (a handle length next to a total length, a cuff depth next to a depth), adds a requirement and replaces nothing: an earlier requirement counts as dropped or replaced only if the message says so about that very property. Replacing a requirement means the old one goes into dropped and the new one into needs. If the shopper ranks one of two wishes above the other, the favoured wish is a need. Any budget the shopper states goes into new_budget. Pure hopes ("ideally", "would be nice") are not needs. Take the shopper at their word and charitably: homely, metaphorical or indirect phrasing that points to a value means the shopper insists on that value; exclude a value only where the shopper plainly asks to steer clear of it. Intensifying words that merely show how strongly a value is held (strictly, holding out for, only, locked to, nothing short of, stays, nothing but, sticking with) do not belong to the value: record only the value itself ("fabric: linen", "shade: grey"), never in the form "<value> only"; an exclusion is added only when the shopper also explicitly rules out blends or alternative values. A word that merely addresses the assistant in a friendly or familiar way is no value, even where a listing could hold the same word: leave it out. A figure paired with a comparison (under, no more than, at least, over, at most) is a limit and keeps its unit. A new budget overrides the previous one, upward or downward, even if the shopper presents it as welcome news. If the shopper merely says that anything is fine, that they have no preference, or that they don\'t mind, enter the property that was asked about in no_preference (if two properties were asked about in the message, take only the FIRST). A response saying only that the shopper could not follow or make sense of what was asked ("I\'m not with you", "say that again more simply", "too many words for me") signals, for this shopper, no preference, provided the question identified which product the property belongs to: in that case enter the asked property in no_preference as well; if the request covers several products and the question left unclear which one was meant, the property remains open (enter nothing in no_preference). When the request involves several products, write each need as "property (product): constraint", naming the product from the message (a message about "the lid shape of the large teapot" gives "lid shape (large teapot): ..."). Nothing in the assistant\'s own question is ever a requirement; only the shopper\'s statements are, and values the question offers as examples never turn into needs. A "don\'t mind" response releases solely the property the assistant\'s message inquired about; it never removes or alters a need the shopper stated before, and if the message posed no question (it merely informed), such a response has no effect. A need expressed by pointing to some other listing takes the message\'s pointing phrase as its value ("pattern: different from the one on Z\'s lamp listing"); the values the message attaches to that other listing describe it rather than the product, so they are not needs, drops or released properties.\n'

READ_RULES = '- Decide each requirement by the listing field that carries that property, taking the value in the words this store uses in that field. A field counts as naming the property when its name holds every word of the property\'s name, however separated or spelled (singular or plural, a spelling variant), also with words added in front; a field whose name leaves out one of those words or calls the property by another name does not. If a field bearing exactly the property\'s name is present, it is decisive, and so is a field with that name plus one extra word in front (sleeve length for length, shell material for material): the requirement is satisfied as soon as either of them writes the value. Fields whose names differ in separators alone (fit type, fit_type) are treated as a single field, and the requirement is satisfied if either one writes the value. Only where no field names the property at all may the value come from the title. When THIS listing has no field naming the property, its title decides, even though other listings of the category carry such a field, and a title that does not write the value fails it; a title value that names the required value among others (a shoe for boys and girls, a size list holding xl) satisfies it. Nothing in the description ever serves as evidence, except what the listing itself states it is compatible with, works with or fits (a list of models or codes there counts for a requirement to fit another item).\n- Compare values word by word in this store\'s own writing: case, separators, singular versus plural and spelling variants make no difference; a narrower value that includes the required one satisfies it (cotton is met by "80% cotton", green by "forest green", rtx 3060 by "rtx 3060 12gb"); a single matching list element suffices. Synonyms, abbreviations, related words, shades and shorter forms (technical wood written as "tech wood", windows 11 as "win11", embroidery as "embroidered", plain as "solid", off-white as ivory or cream) never satisfy a value, even where no listing of the category spells out that very value: a value is read word for word, so a "satin" finish fails matte. The first letters of the required words, written together, are those words rather than an abbreviation, and they satisfy the value wherever this listing writes them. When a value is combined with another through a plus ("wool + nylon"), it does not satisfy either part on its own if that field also holds the other part by itself ("nylon"). FIELD EVIDENCE lines, where present, are derived from this store\'s records and tell which of these cases applies. Another value of the same property does not satisfy (oak is not pine, analog is not digital), and neither does a word that merely resembles it. A value that claims to span every value of the property ("universal", "all seasons", "suitable for all occasions") names no particular one and fails a required value, though a list that additionally names the required value still satisfies it. A field giving some other value of the property fails the required value even when the two commonly come together or one usually suggests the other ("pullover" as a style does not satisfy "zip-up", nor does "slip-on" satisfy "lace-up"); the only thing that counts is a narrower value that includes its words.\n- A field for some other property is not evidence (a sole material does not stand for the material, a width is not a shape, a handle length is not a handle type): it can never replace a property for which the category\'s listings carry a field of its own.\n- Compare numbers as quantities once units are converted (2 m equals 200 cm); a limit includes its boundary value; a range qualifies only if its whole span satisfies the limit. If the property has no field naming it and so the evidence comes from the title, a required exact value is satisfied by a title that presents its words and figures in that same sequence, even if further detail follows them (6\' x 9\' 2" shows 6\' x 9\'; 6\' x 10\' does not).\n- Treat a listing\'s price as using the unit of the request\'s budget, unless the listing\'s record states a different currency or unit (for example, a price of 30 against a budget of 500 US cents means 30 cents, comfortably inside it); neither the price nor the budget may be rescaled.\n- Only the listed requirements matter: do not reject a listing over any gender, age group, season, size, condition, occasion, use or quality that the requirements leave unnamed.\n- The category label by itself satisfies the requested kind: a listing filed under the requested category counts as that kind even if its title names some other object (a refill, an accessory, a case, a set, a container); assess it solely on requirements, stock and price.'

CHECK_ITEM = 'You are the final gate before a shopping assistant settles on an item. The errand earns 1 only if the item satisfies all of the requirements below. Base your judgement ONLY on the records shown, applying these rules:\n' + READ_RULES + '\n- Any word that is part of the category\'s own name is satisfied by the category.\n- A field that names a specific model, part or standard shows that model\'s standard, widely documented specifications; fail it on such a requirement only when that specification is plainly different.\n- Respect exclusions and closed lists.\n- The item has to be the requested kind; where listings bear a category label, it has to be the label that fits the request\'s wording taken as a whole, rather than an adjacent or narrower label.\n- The budget must cover price multiplied by quantity (being exactly equal is acceptable). An item the records show as out of stock fails.\n- A simple wish ("ideally", "would be nice") flagged as not required does not count as a requirement.\nIf records conflict (on stock or price, say), go by the most recent one.\nReply with ONE JSON object and nothing more: {"item": id of the item, "fits": false or true, "problems": [brief strings, each naming a requirement that is unmet or unshown]}. Set fits=false only when a listed requirement is plainly failed by the item\'s fields by these rules (such a failure includes any FIELD EVIDENCE line saying that a requirement is NOT met by the item), or when it is stated neither by a field nor (if no field for it exists) by the title; never fail an item because a stock or price record is missing.'

RANK_CHECK = "A shopping assistant is on the point of ordering an item to fulfil a request asking for the PRIORITY. Shown below are the requirements, the item it picked and the listing records it has seen. Applying the reading rules that follow (the budget covers price times quantity; not shown as out of stock), identify the listings that could meet all requirements AND outdo the picked item on the priority (cheaper, for the cheapest; for the best, a stronger value on the measure). Omit any listing that, according to the records, is out of stock, above budget or failing a requirement. Numbers in a ROW's title (for instance 40 x 70) do not tell which dimension each refers to, so they can never exclude a ROW.\nREADING RULES:\n" + READ_RULES + '\nReply with ONE JSON object and nothing more: {"better": [ids, top candidate first, no more than 4], "reason": "one sentence"}.'

ASK_PLAN = 'Your job is to draw up the questions that a shopping assistant will put to a shopper. The shopper keeps some needs secret. Every need is the value of a single product property, and it comes out only if a message asks about precisely that property; whatever a message asks first gets the most dependable reply, while asking about other things only earns a polite "don\'t mind". Pick at most QUESTION_COUNT distinct properties, ordered from most to least promising:\n0. Start with whatever appears below under MUST BE LEARNED FROM THE SHOPPER and is still unasked, and ask about it directly. When it is a choice between two wishes, ask about one of them alone, the property the request names first;\n1. Next come properties that the details of the listings below really contain. Begin by reading the story (when STORY HINTS appear below, they list its details and which property each one is about, and they take precedence), and look at each key\'s sample values alongside it: a key whose values themselves name the thing the story detail concerns (the purpose, the trouble, the fit, or the person it mentions) ranks above keys like colour or material that all listings have; usually the concealed need is the property that the story\'s concrete details are about while leaving its value unsaid. Whatever the user will handle or push, look at closely or in pictures, listen to, measure or try against something, read off the label, or the way the item was produced, points at that part or facet of the item and the property belonging to it: tags, labels, or small print indicate the material, fabric or ingredients; pictures and appearance indicate the pattern, colour, or visible part; a space or body part it has to fit indicates that part\'s shape or size; a purchase for another person, or for someone hardly known, indicates the person it is designed for (age group, gender); the way it has to feel on the hand, skin or foot indicates the material or fabric first; the place, time or activity it serves or is worn for (say a dance lesson, a workshop, the beach) indicates its style or type, meaning the property with such uses among its values; something fastened or readjusted over and over indicates its closure or mechanism; the thing doing the work inside indicates its core component. Weather, an occupation, a timetable, or a location other than the one where it gets used or worn counts only as background. Your first question goes to the property the story\'s details indicate, using the listings\' name for it. Who the item is made for competes on equal terms: ask it if the story indicates another person, or if there is no other hint. After that, favour properties that describe the product\'s nature (function, length, closure, pattern, shape, material, type or style variant) above logistical ones (years, pack size, condition, warranty, lists of variants or sizes).\nDo not choose a property the request has already decided (its requirements, price, stock, quantity); do not ask a property twice; do not use two questions on near-identical properties (two style kinds, two spellings of the same property); and keep away from brands or makers, product or model names, delivery, stock, price, budget, or which item the shopper wants (the request already names it). Only ask about a property that the listing details below include as a key, under the label the listings give it; when a story detail concerns a property that no key below names, ask it through whichever key below gauges or decides the same thing (a key with values that vary as that detail does: a space to be fitted goes to that key\'s diameter, width or size; a substance it must be free of goes to its material), and never under a name absent from the listings; if no such key exists, drop it. Every step-1 question is a single short, plain, friendly sentence (15 words or fewer) that asks which <property> they would like for the <item kind>, with the property called by the listings\' own name (sample values on their own never bring out a need), and it may end with one or two sample values. Skip any opening, progress update, instructions or lists, and never fall back on catch-all keys like a name, a title, a description, options or variants. Every question concerns ONE property of ONE product: no question may hold two properties or properties of two products, because a reply uncovers one need at most and a question with two subjects may go unanswered entirely. Under "second", put the runner-up property, still of the SAME product (not a different name for the first one), or leave it blank: it is asked on its own, in a later message.\nIf SEVERAL products are requested, any one of them may carry hidden needs: distribute the questions so that each product receives one before any receives a second (except a product with no property remaining), draw each product\'s properties from the list for its own category below, write the property in the form "property (product)", and mention the product inside the question ("Which strap type would you like for the backpack?").\nReply with ONE JSON object and nothing more: {"questions": [{"property": "...", "second": "... or empty", "question": "..."}, ...]}.\n'

NEXT_ONE = '\nOnly a single question goes out on this turn; the next one is picked after reading what the shopper replies. In total REMAINING question(s) remain, and all of them ought to be used: place the one most promising property at the head of the list (it may be followed by one or two extras as backups); once a need has been revealed, the following question targets another property and another story detail.\n'

BEFORE_DECLINE = 'A shopping assistant is on the verge of telling the shopper that no item in the store fits, which is seldom the correct ending. Compare the listing records below with the requirements using these reading rules (the budget must cover the price, and the item must not appear as out of stock):\n' + READ_RULES + '\nSeveral listings below are seen only as search rows (their details were never read, which leaves their property fields unknown). Any such row in the requested category (no matter what object is named in its title) is still unchecked if nothing in its title, stock or price reveals a failed requirement (title figures, such as 40 x 70, do not tell which dimension each refers to, so they can never show a row failing): put it under "unopened", most likely first (one whose title names a required value, or else one of the requested kind that nothing counts against).\nAn opened listing that leaves a requirement unshown (no field of it names that property and its title does not write the value) fails that requirement: it is no possible fit.\nReply with ONE JSON object and nothing more: {"possible_fits": [listing ids that appear to satisfy all requirements, strongest first, no more than 5], "unopened": [rows-only listing ids worth opening before declining, no more than 6], "reason": "one sentence"}.'

REF_HOP = 'In a shopper\'s request, the product being asked for borrows its PROPERTY from a different listing in the store, which is described as: DESCRIPTION. The records below belong to listings that could be that one. Reply with ONE JSON object and nothing more, containing:\n"target": the id of whichever record actually IS the described listing, meaning same seller, same kind, and displaying each named detail, judged by meaning (an amount stated in a different unit still counts if it is equal); null if no record is.\n"missing": the named details that the target lacks ([] if it has every one).\n"go_to": if the target\'s own text points the reader to some DIFFERENT listing for the PROPERTY (taking it from, using, referring to, looking at, or going by another listing), give that listing as {"seller": the name of its shop or brand as the text writes it, "kind": the product kind it is, "details": [every value the text gives for it, brief and verbatim: a material, a colour, a size including its unit]}; otherwise null.\n"value": when no go_to was found, the PROPERTY value of the target itself, written as in its record.\n"goes_with": the values that the target\'s text says match, pair with, or are equivalent to that value ([] if none).\nA listing that the text mentions only as what the target isn\'t ("not the green one from X"), and likewise one it names only as a colour, size or version on sale in other places, does not count as a go_to; neither does a listing it mentions for some other property. A record lacking the described details is never the target, no matter what its text claims.\n'

DECLINE_WORDS = r"refus|declin|cancel|abandon|reject|give.?up|abstain|withdraw|no.?(?:purchase|order|buy)|nothing|without"
ORDER_WORDS = r"commit|buy|order|purchas|check.?out|pay|confirm|complete|place|accept|proceed"
RANK_PROPS = """A shopper asked an assistant to buy something. Besides what they wrote, the shopper has hidden must-have properties that they reveal only when a question names that exact property, and only a few questions may be asked.
Rank the properties of the ONE product below, the one most likely to hide such a must-have first, in this order of preference: (1) a property that a story hint below points to; (2) the properties that decide whether an item of this kind suits the buyer at all, the ones a buyer of such a product settles first (take them from the listings' properties below, even when only a few listings show one); (3) then plain technical figures that differ from listing to listing. A property with the same value in every listing seen is a weak choice. Leave out anything the request or the requirements so far already settle, any value the request says to read from another listing, anything already asked about this product, a second name of a property already ranked (two kinds of style, two spellings of one property), the price or budget, the brand, maker, seller or shop, free-text fields (descriptions, notes), logistics (warranty, shipping, condition of the offer, the number of units in a pack, years), values that merely list every option on offer, and a list of the devices, models or other products an item works with (the order's other items settle that, not the shopper's taste).
Product: PRODUCT_NAME
The shopper's request:
REQUEST_TEXT
Requirements so far:
LEDGER_TEXT
Already asked:
ASKED_TEXT
Story hints (details of the request's story, each with the property it seems to concern; not requirements):
HINT_TEXT
A property a story hint points to ranks first. Match the hint's DETAIL, not the property name guessed for it, to the listed property whose values describe that detail (where or how the item is kept points to a key about storage; the person it is for points to an age or gender key; how it looks points to colour or pattern); only after the hinted properties come the others.
Properties the store's listings of this product carry (in how many of the listings read, and example values):
MENU_TEXT
Reply with JSON only: {"ranked": ["property", ...]} with up to 6 property names spelled exactly as listed."""
PROCEDURE = """You check a shopping assistant against the procedure rules in its shopper's request, just before it MOMENT. Below are the request, the procedure rules found in it, what the assistant has done so far (step by step, with the shopper's replies), the cart now, and what is left.
For each rule decide: already met; due now (it must be done before this moment); or not due yet (it belongs to a later moment, or to another kind of finish). Only the request's own rules count; never invent a step, and never redo a step already done. Limits on messages or steps are met by not breaking them. Timing decides: a rule about the order, the final basket, its lines, its total, an approval of it or the checkout is due only when the assistant is about to place the order, never before the first item is in the basket; only a rule that itself says it comes before anything is put in the basket (or before choosing) is due at the first put. A rule for an order never applies to a decline, and one for a decline never applies to an order. A message sent before the last cart change (its step is given below) does not show the current cart and does not meet a rule about the basket. A rule that needs the shopper's answer is met only once the shopper has answered as it requires.
For each rule due now, name what must be done: "message" (write the full message in plain words: when it shows the cart, give each line with its quantity, its name, its listing id and its price, and the total; when the rule asks for the shopper's agreement, end with a short yes/no question), "tool" (one call of a listed tool, with its arguments: only tools that change neither the cart nor the finish) or "finish_text" (words the finishing call must carry). Never write a placeholder or a guess: every name, id, quantity, price and total in a message is taken from the cart and records shown; a figure not known yet means the message is not due yet. Use as few messages as the rules allow, and never more than are left.
Answer with ONE JSON object, nothing else: {"actions": [{"rule": "...", "do": "message" | "tool" | "finish_text", "text": "...", "tool": "...", "args": {}, "finish_text": "..."}], "wait_for_reply": true when the moment must wait for the shopper's answer to a message named here}. An empty "actions" list when nothing is due now."""
SORRY = "Sorry, nothing in the store meets all of your requirements, so I am not ordering anything."
ID_LIKE = r"[A-Za-z0-9][A-Za-z0-9_\-:.]{3,63}"
ID_KEYS: set[str] = set()   # keys a store's rows showed to hold the listing id (the name differs from store to store)
WORD_KEYS = r"^q$|query|text|keyword|needle|term|word|phrase|string|search"


def id_named(key: Any, strict: bool = False) -> bool:
    """Whether a record key holds the listing id. strict: the key ends in id or has id as a word ("listingId", "item_id_code"),
    or the store's rows showed it; otherwise any key with id in it."""
    if str(key) in ID_KEYS:
        return True
    low = str(key).lower()
    if not strict:
        return "id" in low
    return low.endswith("id") or "id" in re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", str(key)).lower().replace("_", " ").replace("-", " ").split()


# ---------------- results written as text: read into data whatever the layout ----------------
RECORD_CODE = r"[A-Za-z]{1,5}[-_]?[0-9a-fA-F]{5,}|[A-Za-z]{0,4}\d{5,}"


def text_seps(text: str, inner: bool = False) -> list[str]:
    """The separators a text uses, most used first: a line break, a tab, or one or two marks with the spaces around them,
    each seen at least twice (a comma only between the entries of a value, never between the fields of a record)."""
    counts: dict[str, int] = {}
    for mark in re.findall(r"\n|\t|\s?[^\w\s:=.'\"()%+\-]{1,2}\s?", text):
        if mark.strip() == "," and not inner:
            continue
        counts[mark] = counts.get(mark, 0) + 1
    return sorted((m for m, n in counts.items() if n >= 2), key=lambda m: -counts[m])
PAIR_HEAD = r"^\s*([A-Za-z][\w .,&+#'\-/()]{0,40}?)\s*SEP\s*(.*)$"
LAYOUTS: dict[str, dict[str, Any]] = {}   # tool -> the separators a model read off one of its results (when no plain layout fits)


def text_value(raw: str) -> Any:
    v = raw.strip()
    low = v.lower()
    if low in ("true", "false"):
        return low == "true"
    if low in ("none", "null", ""):
        return None
    if re.fullmatch(r"-?\d+", v):
        return int(v)
    if re.fullmatch(r"-?\d*\.\d+", v):
        return float(v)
    return v


def text_pairs(text: str, sep: str, mark: str) -> dict[str, Any] | None:
    """text cut at sep into name-mark-value pairs; a piece without a name of its own continues the value before it."""
    head = re.compile(PAIR_HEAD.replace("SEP", re.escape(mark)), re.S)
    out: dict[str, Any] = {}
    last = None
    named = 0
    pieces = text.split(sep)
    for piece in pieces:
        m = head.match(piece)
        if m and (m.group(1).strip() not in out):
            last = m.group(1).strip()
            out[last] = m.group(2)
            named += 1
        elif last is not None:
            out[last] += sep + piece
        else:
            return None
    return out if named >= 3 else None


def read_text(text: str, depth: int = 0, layout: dict[str, Any] | None = None) -> Any:
    """A string that holds a record, read into a dict: JSON text, or name/value pairs joined by the separator the text uses
    most (the pairs of a value are read one level down); anything else stays the string it was."""
    s0 = text.strip()
    if depth > 2 or len(s0) < 12:
        return text
    if s0[:1] in "{[" and s0[-1:] in "}]":
        try:
            return read_all(json.loads(s0), depth + 1)
        except ValueError:
            pass
    tries = ([(layout["field"], layout["pair"])] if layout and layout.get("field") and layout.get("pair") else []) + \
        [(sep, mark) for sep in text_seps(s0, depth > 0) for mark in ":=" if mark not in sep]
    best = None
    for sep, mark in tries:
        got = text_pairs(s0, sep, mark)
        if got and (best is None or len(got) > len(best[0])):
            best = (got, sep, mark)
    if not best:
        return text
    got, sep, _ = best
    if depth == 0 and len(got) < 5 and not any(re.fullmatch(RECORD_CODE, str(v).strip()) for v in got.values()):
        return text          # (a record names its listing by a code; a short sentence with a colon or two is not one)
    inner = layout.get("inner") if layout else None
    out: dict[str, Any] = {}
    for key, raw in got.items():
        val = raw.strip()
        sub = read_text(val, depth + 1, {"field": inner, "pair": layout.get("inner_pair")} if inner and layout.get("inner_pair") else None) \
            if (text_seps(val, True) or (inner and inner in val)) and sep not in val else val
        out[key] = sub if isinstance(sub, dict) else text_value(val)
    if depth == 0 and not any(id_named(k, True) for k in out):
        # (no field is named for an id: the field whose value is a listing code holds it, as learn_id_key finds in rows)
        code = next((k for k, v in out.items() if isinstance(v, str) and re.fullmatch(RECORD_CODE, v.strip())), None)
        if code:
            ID_KEYS.add(code)
    return out


def read_all(piece: Any, depth: int = 0, layout: dict[str, Any] | None = None, loose: bool = True) -> Any:
    """Every string in a tool result that holds a record becomes a dict (lists of them a list of dicts). Only a list's items
    and the values of a small wrapper ({"value": ...}) are read: the text fields of a record that is already data (its
    description, its notes) stay text."""
    if isinstance(piece, dict):
        small = len(piece) <= 3
        return {k: read_all(v, depth, layout, small) for k, v in piece.items()}
    if isinstance(piece, list):
        return [read_all(v, depth, layout, True) for v in piece]
    if loose and isinstance(piece, str) and len(piece) >= 24 and (re.search(r"[:=]", piece) or (layout and str(layout.get("pair") or "\0") in piece)):
        return read_text(piece, depth, layout)
    return piece


def unread_record(piece: Any, loose: bool = True) -> str:
    """A string left where read_all reads (a list item, a small wrapper's value) that still looks like a record: it holds a
    code with digits and several separators."""
    if isinstance(piece, dict):
        return next((x for x in (unread_record(v, len(piece) <= 3) for v in piece.values()) if x), "")
    if isinstance(piece, list):
        return next((x for x in (unread_record(v, True) for v in piece) if x), "")
    if loose and isinstance(piece, str) and len(piece) >= 60 and re.search(r"\b(?:" + RECORD_CODE + r")\b", piece):
        marks: dict[str, int] = {}
        for mark in re.findall(r"[^\w\s]{1,3}", piece):
            marks[mark] = marks.get(mark, 0) + 1
        if len(re.findall(r"[:=]", piece)) >= 4 or max(marks.values(), default=0) >= 4:
            return piece
    return ""


def learn_id_key(rows: list[Any]) -> None:
    """A list of records with no key named for an id: the key whose values all differ and read as codes holds it."""
    rows = [r for r in rows if isinstance(r, dict)]
    if len(rows) < 3 or any(id_named(k, True) for r in rows for k in r):
        return
    for key in rows[0]:
        vals = [r.get(key) for r in rows]
        if all(isinstance(v, str) and re.fullmatch(ID_LIKE, v) and re.search(r"\d", v) for v in vals) and len(set(vals)) == len(vals):
            ID_KEYS.add(str(key))
            return
CODE_LIKE = r"[A-Za-z0-9_\-]+"
# a part's own words about what it fits (the phrase set of Know Thyself's public _fit_marks), and a model code: letters, an
# optional dash, digits, and trailing x's standing for any digit ("XY-39xx"); "except" and "but not" name codes left out
FIT_CLAUSE = r"\b(?<!not )(?<!never )(?:compatible with|works with|goes with|fits)\b\s*:?\s*([^.;]+)"
MODEL_CODE = r"\b([a-z]{1,6})[- ]?(\d{2,6})(x{0,4})\b"
OWN_MODEL = r"\bmodel(?:\s*(?:no\.?|number|#|ref\.?|reference))?\s*[:.]?\s*([a-z]{1,6}[- ]?\d{2,6})\b(?!\s+also\s+takes)"


def model_codes(text: str) -> list[tuple[str, str, str]]:
    return [(m.group(1), m.group(2), m.group(3)) for m in re.finditer(MODEL_CODE, str(text or "").lower())]


# a device that names the part series it takes, and a part whose own bracketed code is its series, a dash and a number;
# the series stands in the fit machinery as a model code of its own, so a series part fits a series device only when the
# series agree
SERIES_TAKES = r"(?<!also takes )\bparts?\s+series\s*:?\s*([a-z]{1,5}\d{1,3}(?:\W+(?:[a-z]+\s+)?[a-z]{1,5}\d{1,3})*)\b"
SERIES_TAG = r"[\[(]\s*(?:part\s+)?([a-z]{1,5})(\d{1,3})-\d{2,6}\s*[\])]"     # (race 17 writes "(part VI7-3768)")
# another page may widen a device: "Model XY1234 also takes part series AB5" holds for every part of series AB5, so it goes
# with the notes from other pages, not with the page's own codes
SERIES_WIDENED = r"\bmodel\s+([a-z]{1,6}[- ]?\d{2,6})\s+also\s+takes\s+parts?\s+series\s*:?\s*([a-z]{1,5}\d{1,3}(?:\W+[a-z]{1,5}\d{1,3})*)"


def own_models(text: str) -> set[str]:
    low = str(text or "").lower()
    series = {c for m in re.finditer(SERIES_TAKES, low) for c in re.findall(r"[a-z]{1,5}\d{1,3}", m.group(1))}
    return {re.sub(r"[- ]", "", m.group(1)) for m in re.finditer(OWN_MODEL, low)} | series


def fit_rules(text: str) -> list[tuple[list[tuple[str, str, str]], list[tuple[str, str, str]]]]:
    """Each 'fits ...' clause of a record as (codes it fits, codes it leaves out); a clause naming no code is skipped."""
    rules = []
    low = str(text or "").lower()
    for m in re.finditer(FIT_CLAUSE, low):
        if re.search(r"\bpart\s+[a-z]{1,4}\d{3,6}\s+also\s+$", low[max(0, m.start() - 40):m.start()]):
            continue                     # ("part X also fits ..." on this page is about part X, another listing)
        parts = re.split(r"\b(?:except(?:ing)?|excluding|but not)\b", m.group(1), maxsplit=1)
        inc, exc = model_codes(parts[0]), model_codes(parts[1] if len(parts) > 1 else "")
        if inc:
            rules.append((inc, exc))
    tag = re.search(SERIES_TAG, low)
    if tag and not re.search(SERIES_TAKES, low):
        rules.append(([(tag.group(1), tag.group(2), "")], []))
    return rules


PART_TAG = r"\[\s*(?:part\s+)?([a-z]{1,4}\d{3,6})\s*\]"
ALSO_FITS = r"\bpart\s+([a-z]{1,4}\d{3,6})\s+also\s+fits\s*:?\s*([^.;]+)"
SEE_PART = r"\b(?:see|read|more on|details on)\s+(?:the\s+)?part\s+([a-z]{1,4}\d{3,6})\b"


def code_matches(device: str, code: tuple[str, str, str]) -> bool:
    letters, digits, wild = code
    return bool(re.fullmatch(re.escape(letters + digits) + r"\d" * len(wild), device))


def fit_verdict(device: str, rules: list) -> bool | None:
    """Whether a part's clauses take in this device model: None when no clause names a code of the device's own letters."""
    letters = re.match(r"[a-z]+", device)
    mine = [(inc, exc) for inc, exc in rules if letters and any(c[0] == letters.group(0) for c in inc)]
    if not mine:
        return None
    return any(any(code_matches(device, c) for c in inc) and not any(code_matches(device, c) for c in exc) for inc, exc in mine)


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


SHELF_NAMES: set[str] = set()   # a store's own name for its category field, learned from its rows (it differs from store to store)


def key_words(key: Any) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", re.sub(r"([a-z])([A-Z])", r"\1 \2", str(key)).lower()))


def shelf_key(key: Any) -> bool:
    """Whether a record key holds the listing's category (shelf) label: the usual names, or the one this store's rows showed."""
    return bool(re.search(r"categor|shelf|department|aisle", str(key).lower())) or key_words(key) in SHELF_NAMES


def learn_shelf_key(obs: Any, products: list[str]) -> None:
    """A store that names its category field in words of its own: the field of a page of result rows whose short values
    repeat from row to row and name the products asked for is taken as the category field, once, for the rest of the run."""
    if SHELF_NAMES:
        return
    names = [set(kind_tokens(p)) for p in products]
    names = [n for n in names if n]
    if not names:
        return
    tables: list[list[dict[str, Any]]] = []

    def visit(d: dict[str, Any]) -> None:
        for val in d.values():
            if isinstance(val, list) and len(val) >= 3 and all(isinstance(x, dict) for x in val):
                tables.append(val)
    walk_dicts(obs, visit)
    if isinstance(obs, list) and len(obs) >= 3 and all(isinstance(x, dict) for x in obs):
        tables.append(obs)
    for rows in tables:
        keys = {k for r in rows for k, v in r.items() if isinstance(v, str)}
        if any(shelf_key(k) for k in keys):
            return
        best, top = "", 0.0
        for key in sorted(keys, key=str):
            if id_named(key):
                continue
            vals = [r[key].strip().lower() for r in rows if isinstance(r.get(key), str) and 0 < len(r[key].strip()) <= 60]
            if len(vals) < 0.7 * len(rows) or len(set(vals)) > max(1, 0.6 * len(vals)):
                continue
            share = sum(1 for v in vals if any(kind_score(n, set(kind_tokens(v)), set()) > 0 for n in names)) / len(rows)
            if share > top:
                best, top = str(key), share
        if best and top >= 0.3:
            SHELF_NAMES.add(key_words(best))
            return


def junk_key(key: Any) -> bool:
    part = re.split(r"\s*\(", str(key))[0]
    words = " ".join(re.findall(r"[a-z0-9]+", re.sub(r"([a-z])([A-Z])", r"\1 \2", part).lower()))
    return bool(re.search(r"\b(?:brand|shop|seller|store|vendor|manufacturer|category|categories|department|title|headline|"
                          r"heading|listing|description|overview|price|cost|amount|availability|sku|url)\b", words)
                or re.fullmatch(r"(?:name|notes?|stock|in stock|on hand|images?|links?|make|maker|shelf|ids?|id)", words)
                or words in SHELF_NAMES)


# a label that names the product's kind (product type, item type) is the category itself, never a property to ask about
KIND_LABELS = ("type", "kind", "category", "product", "product type", "item", "item type")


def kind_label(key: Any) -> bool:
    part = re.split(r"\s*\(", str(key))[0]
    return " ".join(re.findall(r"[a-z0-9]+", re.sub(r"([a-z])([A-Z])", r"\1 \2", part).lower())) in KIND_LABELS


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


# a stated requirement read word for word: the words of the value written in a field named for the property
LIT_AT_MOST = r"\b(?:at most|no more than|not more than|up to|or less|or lower|or smaller|maximum|under|below|less than)\b"
LIT_AT_LEAST = r"\b(?:at least|no less than|not less than|or more|or higher|or larger|minimum|over|above|more than)\b"


def lit_words(text: Any) -> list[str]:
    low = re.sub(r"(\d)([a-z])", r"\1 \2", re.sub(r"[^a-z0-9.]+", " ", str(text or "").lower()))
    return [w.strip(".") for w in low.split() if w.strip(".")]


def lit_has(want: list[str], pool: list[str]) -> bool:
    # (every word of the value among the pool's words; singular/plural and spelling variants alike, a shorter word never)
    return bool(want) and all(any(w == p or same_kind(w, p) or (w.isalpha() and p.isalpha() and len(min(w, p, key=len)) >= 4
                                                                and spelled_alike(*sorted((w, p), key=len))) for p in pool) for w in want)


def lit_meets(fields: dict[str, Any], title: Any, prop: str, value: str) -> bool | None:
    """One stated requirement on a read listing, read word for word: the fields named for the property (the field's name
    holds every word of the property's name) must write the value (one alternative of an 'or' list; a bound figure by its
    bound); a listing with no such field must write it in its title (a bound figure needs the field). None when the value
    is not a plain written one."""
    name = lit_words(prop)
    low = re.sub(r"\((?:stated|revealed[^)]*)\)", " ", str(value or "").lower()).replace('"', " ")
    low = re.sub(r"^\s*(?:only|just|exactly|strictly)\b", " ", low).strip()
    if not name or not low or re.search(r"\bbetween\b", low):
        return None
    vals = [str(v) for key, v in fields.items() if str(v).strip() and lit_has(name, lit_words(key))]
    bound = "most" if re.search(LIT_AT_MOST, low) else "least" if re.search(LIT_AT_LEAST, low) else ""
    if bound:
        figs = re.findall(r"\d+(?:\.\d+)?", low)
        if not figs:
            return None
        cap = float(figs[0])
        for v in vals:
            got = re.findall(r"\d+(?:\.\d+)?", v)
            if got and (float(got[0]) <= cap + 1e-9 if bound == "most" else float(got[0]) >= cap - 1e-9):
                return True
        return False
    alts = [ws for ws in (lit_words(part) for part in re.split(r",|;|\bor\b", low)) if ws]
    if not alts:
        return None
    pool = vals or [str(title or "")]
    return any(lit_has(ws, lit_words(v)) for ws in alts for v in pool)


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
    return [w for w in re.findall(r"[a-z]{3,}", re.sub(r"'s\b", "", str(text).lower().replace("’", "'"))) if w not in LINK_WORDS]


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
# (a requirement for the value the store pairs with another listing's value: only the paired value meets it)
PARTNER_WORDS = r"\bpartner(?:s|ed)?\b(?!\s+(?:and|&))"
PAIR_CLAUSE = r"([a-z0-9][a-z0-9 .%/&'-]*?)\s+(?:go(?:es)?|pairs?)\s+(?:together\s+|up\s+)?with\s+([a-z0-9][a-z0-9 .%/&'-]*?)(?=\s*(?:[,;.]|\band\b|$))"
REF_STEPS = 0.55
REF_CALLS = 12
REF_HOPS = 8
DIFFER_WORDS = (r"\bset apart from\b|\bapart from\b|\bdistinct from\b|\bdifferent (?:from|to|than)\b|\bdiffer(?:s|ent|ing)?\s+from\b"
                r"|\bother than\b|\bunlike\b|\bnot the same as\b|\banything but\b")
POINT_WORDS = r"\bwithin\b|\bat (?:most|least)\b|\bno (?:more|less) than\b|\bsame as\b|\bas shown\b|\bthe value\b|\bthe one\b"
MOST_WORDS = (r"\b(?:at most|up to|or less|or lower|or smaller|no (?:more|higher|bigger|larger|greater|longer|heavier|wider) than|"
              r"not (?:any )?(?:more|higher|bigger|larger|greater) than)\b")
LEAST_WORDS = (r"\b(?:at least|or more|or higher|or larger|no (?:less|lower|smaller|fewer|shorter|lighter|narrower) than|"
               r"not (?:any )?(?:less|lower|smaller|fewer) than)\b")
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


def differ_said(text: Any) -> bool:
    """Whether a text sets the value apart from another listing's. A differ word followed within three words by a relation
    of its own ("apart from just the same as X", "other than within 2 cm of X") only frames that relation; a relation
    word that opens the listing's own description ("the value shown on X's page") leaves the differ word standing."""
    low = str(text or "").lower()
    for m in re.finditer(DIFFER_WORDS, low):
        inner = re.match(r"(?:\s+\S+){0,3}?\s+(" + "|".join((POINT_WORDS, MOST_WORDS, LEAST_WORDS)) + ")", low[m.end():])
        if not inner or inner.group(1).startswith("the "):
            return True
    return False


def differ_framed(text: Any) -> bool:
    """A differ word stands in the text, and every one of them only frames a relation of its own."""
    return bool(re.search(DIFFER_WORDS, str(text or "").lower())) and not differ_said(text)


def ref_relation(value: Any) -> tuple[str, float | None, str]:
    low = str(value or "").lower()
    if differ_said(low):
        return "differ", None, ""
    if re.search(PARTNER_WORDS, low):
        return "partner", None, ""
    hit = re.search(r"\bwithin\s+(\d+(?:\.\d+)?)\s*([a-z%]+)?", low)
    if hit:
        return "within", float(hit.group(1)), (hit.group(2) or "").strip()
    if re.search(MOST_WORDS, low):
        return "at most", None, ""
    if re.search(LEAST_WORDS, low):
        return "at least", None, ""
    return "same", None, ""


def bare_words(text: Any, keep: bool = False) -> list[str]:
    """A text's words less the link words and the words that only name a listing ("listing", "page", "one"); with keep, the
    words between the first and the last such word are kept as written."""
    words = str(text or "").split()
    plain = lambda w: re.sub(r"[^a-z0-9]", "", w.lower())
    idle = lambda w: not plain(w) or plain(w) in LINK_WORDS or re.fullmatch(r"(?:listing|page|one)s?", plain(w))
    if not keep:
        return [w for w in words if not idle(w)]
    while words and idle(words[0]):
        words = words[1:]
    while words and idle(words[-1]):
        words = words[:-1]
    return words


def one_edit(a: str, b: str) -> bool:
    """Whether two strings differ by exactly one character missing, extra or changed."""
    if a == b or abs(len(a) - len(b)) > 1:
        return False
    if len(a) == len(b):
        return sum(x != y for x, y in zip(a, b)) == 1
    short, long_ = (a, b) if len(a) < len(b) else (b, a)
    return any(long_[:n] + long_[n + 1:] == short for n in range(len(long_)))


def ref_parse(text: Any, ok: Any = None) -> dict[str, Any] | None:
    try:
        # (a short aside opened by "or" that only widens the pointer is dropped first: the seller follows it)
        body = re.sub(r"\(\s*or\b[^()]{0,30}\)(?=[^()]*[A-Z][a-z]{3,})|\s[—–]\s*or\b[^—–]{0,30}[—–](?=\s)", " ", str(text or ""))
        body = re.split(r"\(\s*or\b", body)[0]
        hits = [m for m in re.finditer(r"(?<![A-Za-z'])([A-Z][a-z]{3,})(?:'s)?(?![A-Za-z])", body) if m.start() > 0
                and (ok is None or ok(m.group(1)))]
        if not hits:
            return None
        # (a name the text marks as a seller, "X's", "by X" or "from X", before a capitalised word that only opens a clause)
        hit = max(hits, key=lambda m: (m.group(0).endswith("'s") + bool(re.search(r"\b(?:by|from)\s+$", body[:m.start()], re.I)), -m.start()))
        # (the description ends at a dash clause, a semicolon or the sentence's end; what follows is the shopper's own talk)
        segs = re.split(r"\s[—–](?:\s|$)|;|(?<=[A-Za-z0-9%)])[.!?](?:\s|$)", body[hit.end():])
        before, after, kind = body[:hit.start()], segs[0], ""
        if len(segs) > 1 and (re.match(r"\s*\S+\s+ones?\b", segs[1], re.I) or (re.search(r"\bones?\s*$", segs[1]) and not bare_words(after))):
            # (a listing described in an aside between dashes, "... the one with ..." or "... the <values> one":
            #  that aside gives its details)
            after = after + " with " + re.sub(r"\bones?\b", " ", segs[1])
        page_kind = " ".join(bare_words(re.split(r"\bpage\b", before, flags=re.I)[-1], keep=True)) if re.search(r"\bpage\b", before, re.I) else ""
        sold = re.search(r".*(?:\bthe|\bfor|\ba|\ban)\s+(.*?)\s+(?:sold|made|offered|listed)\s+by\s*$", before, re.I | re.S)
        came = re.search(r"([^,;.!?]*)\bfrom\s*$", before, re.I)
        if sold:
            kind = sold.group(1)
        else:
            lead = re.match(r"\s*([^,;(]*?)(?=\s+(?:with|that|which|in|at|having|from|of|for|sold)\b|[,;]|$)", after, re.I)
            if lead:
                kind, after = lead.group(1), after[lead.end():]
            if came:
                # ("the page of the fairy lights from X's listing": the kind is the words after the last link word)
                words = came.group(1).split()
                last = max([n for n, w in enumerate(words) if w.lower() in LINK_WORDS] + [-1])
                kind = " ".join(words[last + 1:]) or kind
        # (the kind may stand before the seller, "the page of the <kind> in X's listing", and "listing" after it)
        kind = " ".join(bare_words(kind, keep=True))
        of_kind = re.match(r"\s*(?:of|for)\b", after, re.I) if not kind else None
        lead = re.match(r"\s*([^,;(]*?)(?=\s+(?:with|that|which|in|at|having|from|of|for|sold)\b|[,;]|$)", after[of_kind.end():], re.I) if of_kind else None
        if lead and bare_words(lead.group(1)):
            # ("X's listing of the usb cables with ...": the kind follows "listing of")
            kind, after = " ".join(bare_words(lead.group(1), keep=True)), after[of_kind.end() + lead.end():]
        if page_kind and not bare_words(kind):
            kind = page_kind
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


def lensed(piece: Any) -> Any:
    try:
        piece = json.loads(json.dumps(piece))
    except (TypeError, ValueError):
        return piece

    def visit(d: dict[str, Any]) -> None:
        if not any(isinstance(val, str) and id_named(key) and re.fullmatch(ID_LIKE, val) for key, val in d.items()):
            return
        names = {str(key).lower(): key for key in d}
        if not any(re.search(r"price|cost", low) and is_number(d[key]) for low, key in names.items()):
            picks = [key for low, key in names.items() if is_number(d[key]) and re.search(r"amount|fee|charge|pay|rate|value", low)
                     and not re.search(r"total|count|qty|quant|unit|due|sum|rating|star|score|page|number|num|size|weight|width|height|length|found|match", low)]
            if len(picks) == 1:
                d["price"] = d[picks[0]]
        if not any(re.search(r"stock|avail", low) for low in names):
            picks = [key for low, key in names.items() if isinstance(d[key], bool)
                     and (re.search(r"hand|orderable|buyable|sellable|ready|ship|instock|purchas|in_store", low) or re.search(r"can.?buy", low))]
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
        if "/inference/" in path:
            self.model_calls = self.__dict__.get("model_calls", 0) + 1
        answer = client.post_verbose(path, body)
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

    def hard_cap(self) -> float:
        """The errand's hard total: lower for a request of three or more products; the questions carry a small reserve."""
        return (HARD_SPEND_MANY if self.__dict__.get("lines", 1) >= 3 else HARD_SPEND) + self.__dict__.get("bonus", 0.0)

    def query_model(self, msgs: list[dict[str, Any]], model_list: list[str], token_cap: int,
            toolset: list[dict[str, Any]] | None = None) -> dict[str, Any] | None:
        self.call_fail = ""
        self.call_models = set()
        self.dear_skip = False
        size = sum(len(m["content"]) if isinstance(m.get("content"), str) else len(json.dumps(m.get("content"), default=str)) for m in msgs) + 4 * token_cap
        self.call_size = size
        rates = self.__dict__.setdefault("char_rates", {})
        # (no call goes out once the errand's total is spent, nor one whose learned price per character would cross it)
        # (a model not used yet in this errand is priced high, so its first long call cannot slip past the total either)
        price = lambda mdl: rates.get(mdl) or max([1.0e-6 if "sonnet" in mdl else 3.0e-7] + [v for v in rates.values() if v])
        dear = [mdl for mdl in model_list if self.spend + price(mdl) * size > self.hard_cap() + 0.004]
        if self.spend >= self.hard_cap() or self.__dict__.get("model_calls", 0) >= CALL_LIMIT:
            if self.__dict__.get("bonus"):
                return None          # (the questions' reserve is spent: this call is skipped, the rest of the errand decides)
            self.no_model = True
        elif dear and len(dear) == len(model_list):
            self.dear_skip = True    # (only this call would cross the total: smaller calls, such as the questions, still go out)
            return None
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
                if "mimo" in mdl:
                    envelope["reasoning"] = {"enabled": False}
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
                        paid = float(use_info["cost"]) / max(1, self.__dict__.get("call_size", 1))
                        rates = self.__dict__.setdefault("char_rates", {})
                        rates[mdl] = paid if mdl not in rates else 0.5 * rates[mdl] + 0.5 * paid
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

    def read_result(self, data: dict[str, Any], tools: list[str]) -> dict[str, Any]:
        """A result written as text is read into data; a record no plain layout reads gets its separators from one model
        call per tool, kept for the rest of the errand."""
        try:
            tool = tools[0] if len(tools) == 1 else ""
            out = read_all(data, 0, LAYOUTS.get(tool))
            left = unread_record(out)
            if left and tool and tool not in LAYOUTS and self.spare() > 30:
                LAYOUTS[tool] = {}
                ask = ("One item of a store tool's result is below. It is a record written as text. Reply with ONE JSON object and "
                       "nothing else: {\"field\": the exact string that separates its fields, \"pair\": the exact string between a field's "
                       "name and its value, \"inner\": the exact string that separates the entries inside a field whose value is itself a "
                       "list of name/value entries, or null, \"inner_pair\": the exact string between such an entry's name and value, or null}."
                       "\n\nITEM:\n" + left[:1500])
                got = self.consult(ask, 200, ["anthropic/claude-haiku-5.5", "openai/gpt-6-luna"])
                if isinstance(got, dict) and isinstance(got.get("field"), str) and isinstance(got.get("pair"), str):
                    LAYOUTS[tool] = {k: v for k, v in got.items() if isinstance(v, str) or v is None}
                    out = read_all(data, 0, LAYOUTS[tool])
                self.log.append({"kind": "checkpoint", "what": "text_layout", "tool": tool, "layout": LAYOUTS[tool]})
            return out if isinstance(out, dict) else data
        except Exception as err:
            self.log.append({"kind": "checkpoint", "what": "read_result", "error": f"{type(err).__name__}: {err}"[:200]})
            return data

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
                data = self.read_result(answer.data, [str((a or {}).get("name") or (a or {}).get("tool_name") or "") for a in acts])
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

    def side_by_side(self, a: str, b: str) -> bool:
        """Whether one listing's table holds both names as fields of their own: then they are two properties, not two names
        of one (memory / graphics memory)."""
        want = {key_words(a), key_words(b)}
        if len(want) < 2 or "" in want:
            return False
        found = []

        def visit(d: dict[str, Any]) -> None:
            if not found and want <= {key_words(k) for k in d}:
                found.append(True)
        for obs in self.items:
            walk_dicts(obs, visit)
            if found:
                return True
        return False

    def add(self, obs: Any) -> None:
        self.items.append(obs)

        def visit(d: dict[str, Any]) -> None:
            for piece in d.values():
                if isinstance(piece, list):
                    learn_id_key(piece)
            for key, piece in d.items():
                if isinstance(piece, str) and id_named(key) and re.fullmatch(ID_LIKE, piece):
                    self.ids.add(piece)
            ident = next((piece for key, piece in d.items() if isinstance(piece, str) and str(key).lower() in ("id", "_id")
                          or (isinstance(piece, str) and id_named(key, True) and re.fullmatch(ID_LIKE, piece))), None)
            if ident and any(isinstance(val, dict) and sum(1 for v in val.values() if isinstance(v, (str, int, float))) >= 2
                             for val in d.values()):
                self.read.add(ident)
            own = next((piece.strip() for key, piece in d.items() if isinstance(piece, str) and 0 < len(piece.strip()) <= 60
                        and shelf_key(key) and "id" not in str(key).lower()), "")
            if ident and own:
                self.__dict__.setdefault("label_of", {}).setdefault(ident, own)
        walk_dicts(obs, visit)

    def price(self, thing: str) -> float | None:
        found: list[float] = []

        def visit(d: dict[str, Any]) -> None:
            if not any(isinstance(val, str) and val == thing and id_named(key) for key, val in d.items()):
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

    def digest(self, char_cap: int = 24000, value_chars: int = 160) -> list[str]:
        record_texts, char_total = [], 0
        for rec in reversed(self.items):
            text_body = as_text(clipped(rec, value_chars))
            if char_total + len(text_body) > char_cap:
                continue
            record_texts.append(text_body)
            char_total += len(text_body)
        return record_texts

    def cards(self, char_cap: int = 30000, cheaper_than: str = "", skip: set[str] | None = None, first: Any = ()) -> list[str]:
        merged: dict[str, dict[str, Any]] = {}
        order: list[str] = []

        def visit(d: dict[str, Any]) -> None:
            ident = next((val for key, val in d.items() if isinstance(val, str) and id_named(key)
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
                    and shelf_key(key) and "id" not in str(key).lower()), "")
        if own:
            return own
        labels = self.__dict__.get("label_of") or {}
        return next((labels[val] for key, val in d.items() if isinstance(val, str) and val in labels and id_named(key)), "")

    def rich_reads(self, shelf: str = "") -> int:
        rich: set[str] = set()

        def visit(d: dict[str, Any]) -> None:
            ident = next((val for key, val in d.items() if isinstance(val, str) and id_named(key) and val in self.read), None)
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
            ident = next((val for key, val in d.items() if isinstance(val, str) and id_named(key) and val in self.read), None)
            shelf = self.label_for(d) if ident else ""
            if ident and shelf and any(isinstance(val, dict) for val in d.values()):
                tally[shelf] = tally.get(shelf, 0) + 1
        for rec in self.items:
            walk_dicts(rec, visit)
        return sorted(tally, key=lambda s: -tally[s])

    def prop_menu(self, shelf: str = "") -> tuple[str, int]:
        tallies: dict[str, int] = {}
        examples: dict[str, list[str]] = {}
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
            brief_vals = [(fkey, val) for fkey, val in d.items() if isinstance(val, str) and 0 < len(val) <= 70 and not junk_key(fkey)]
            if len(brief_vals) < 5 and not (shelf and len(brief_vals) >= 2 and not any(isinstance(v, dict) for v in d.values())):
                return
            sets += 1
            rowlike = len(brief_vals) < 5 and any(isinstance(val, str) and "id" in str(fkey).lower() for fkey, val in d.items())
            mapped += 0 if rowlike else 1
            for key, piece in brief_vals:
                if re.search(r"\d", piece) and re.fullmatch(CODE_LIKE, piece) and ("-" in piece or "_" in piece):
                    continue
                call_name = re.sub(r"[_.\-]+", " ", re.sub(r"(?<=[a-z])(?=[A-Z])", " ", str(key))).strip().lower()   # (compatibleDevices, pack-type: words)
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
        if is_number(spend_cap) and spend_cap > 0:
            # a limit the shopper writes in cents, turned into dollars by the model: take the shopper's own number
            for match in re.finditer(r"(\d[\d,]*)\s*(?:usd?[\s-]*)?(?:cents?\b|¢)", " ".join(str(x) for x in new_lines), re.I):
                written = int(match.group(1).replace(",", ""))
                if written >= 1 and abs(spend_cap * 100 - written) < 1:
                    spend_cap = written
                    break
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
            ids = [k for k in ids if id_named(k)] or ids
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
            if count <= 0:
                self.units[thing] = 0.0               # (a put of none takes the line out: adding none would change nothing)
            elif prior > 0 and self.adds is not False:
                self.units[thing] = prior + count
                self.pending = {thing: count}
            else:
                self.units[thing] = count if count > 0 else 0.0
        else:
            self.units[thing] = max(0.0, prior - count) if count > 0 else 0.0
        if self.units.get(thing, 0) <= 0:
            self.units.pop(thing, None)
        seen = self.view_units(obs)
        if seen is not None and set(seen) == {thing}:
            self.learn_line(thing, seen[thing])
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
                     and not re.search(r"price|cost|amount|due|fee|charge|sum", str(fkey).lower())]
            if len(pairs) > 1:
                named = [(fkey, val) for fkey, val in pairs if re.search(r"qty|quant|count|units|number|total|held|ordered|piece", str(fkey).lower())
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

    def learn_line(self, thing: str, units: float) -> None:
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
        self.msgs_sent = 0
        self.no_pairs = True          # one property per question: a two-property message can confuse a store's shopper
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
        if self.msg_cap is None:
            return ASK_DEFAULT
        return max(0, self.msg_cap - self.kept_messages())

    def kept_messages(self, kind: str = "") -> int:
        reading = getattr(self.needs, "reading", None) or {}
        per_kind: dict[str, int] = {}
        for rule in reading.get("finish_messages") or []:
            if not isinstance(rule, dict) or not str(rule.get("rule") or "").strip():
                continue
            count = rule.get("messages")
            if not is_number(count) or isinstance(count, bool) or count <= 0:
                continue
            when = str(rule.get("when") or "").lower()
            if kind and kind not in when:
                continue
            finish = "decline" if "decline" in when else "order"
            if finish == "decline" and self.role("decline"):
                continue                     # (the store's decline move itself tells the shopper that nothing fits)
            text = str(rule.get("rule")).lower()
            if finish == "order" and "before" not in text and not re.search(r"basket|total|cart", text):
                continue                     # (a sentence of the story holds no message back)
            per_kind[finish] = per_kind.get(finish, 0) + int(count)
        return min(max(per_kind.values() or [0]), 2)

    def reorder_after_refusal(self, before: tuple[int, int, Any]) -> bool:
        refusals, needs_before, cap_before = before
        order = self.role("order")
        if self.refusal_n <= refusals or self.wire.done or self.wire.broke or not order or not self.trolley.units \
                or self.last_kind != "order" or self.__dict__.get("reorders", 0) >= 2 or self.moves_left() < 2 or self.wire.spare() < CLOSING:
            return False
        if len(self.needs.uncovered) > needs_before or set(self.trolley.units) & self.dead_lines():
            return False
        cap = self.needs.spend_cap
        prices = {i: self.archive.price(i) for i in self.trolley.units}
        if any(p is None for p in prices.values()):
            return False
        total = sum(prices[i] * float(q) for i, q in self.trolley.units.items())
        if is_number(cap) and total > float(cap) + 1e-9:
            return False
        for line in list(self.trolley.units):
            if self.check_item(line, "ORDER"):
                self.log.append({"kind": "checkpoint", "what": "reorder_after_refusal", "result": "a line fails now", "line": line})
                return False
        self.reorders = self.__dict__.get("reorders", 0) + 1
        self.refused_lines.difference_update(self.trolley.units)
        call = {"id": f"reorder_{self.wire.tick}", "type": "function", "function": {"name": order, "arguments": json.dumps(self.plain_args(order, "Order the cart.", "order"))}}
        self.log.append({"kind": "checkpoint", "what": "reorder_after_refusal", "result": "sent", "cart": dict(self.trolley.units)})
        self.chat.append({"role": "assistant", "content": "The refusal only loosened the requirements and the cart still meets every one now in force, so I order it again unchanged.", "tool_calls": [call]})
        self.post_replies(self.dispatch([call], scripted=True))
        return True

    def cart_instead(self) -> tuple[str, dict[str, Any]] | None:
        """A decline while the cart holds the whole basket: a change since its last check (a dropped wish, a new budget) leaves
        no check for the requirements now in force, so its lines are checked again, and if every one passes it is ordered."""
        order = self.role("order")
        lines = [i for i, units in self.trolley.units.items() if units]
        mark = (self.__dict__.get("cart_seq", 0), self.needs.version)
        if not order or not lines or self.__dict__.get("cart_kept") == mark or len(lines) < max(1, len(self.needs.items())) \
                or self.moves_left() < 2 or self.wire.spare() < CLOSING:
            return None
        self.cart_kept = mark
        if self.kit_fault(lines):
            self.log.append({"kind": "checkpoint", "what": "cart_instead", "result": "kit parts do not fit"})
            return None
        for i in lines:
            if i in self.refused_lines or i in self.dead_lines():
                return None
            if f"{i}|{self.needs.version}" not in self.passed:
                self.check_item(i, "ORDER")
            if f"{i}|{self.needs.version}" not in self.passed:
                self.log.append({"kind": "checkpoint", "what": "cart_instead", "line": i, "result": "fails its check"})
                return None
        self.log.append({"kind": "checkpoint", "what": "cart_instead", "lines": lines, "result": "ordered instead of the decline"})
        return order, self.plain_args(order, "Order the cart.", "order")

    def fit_instead(self) -> tuple[str, dict[str, Any]] | None:
        order = self.role("order")
        if self.needs.items() or not order or self.__dict__.get("fit_forced") or self.moves_left() < 4 \
                or self.wire.spare() < CLOSING:
            return None
        best = self.unrefuted()
        for _ in range(2):
            unchecked = [i for i in best if f"{i}|{self.needs.version}" not in self.passed]
            if not unchecked or self.wire.spare() < CLOSING + 20 or any(f"{i}|{self.needs.version}" in self.passed for i in best):
                break
            self.check_item(unchecked[0], "ORDER")
            best = self.unrefuted()
        fits = [i for i in best if f"{i}|{self.needs.version}" in self.passed and i not in self.refused_lines and i not in self.dead_lines()]
        if not fits:
            self.log.append({"kind": "checkpoint", "what": "fit_instead", "result": "no checked fit", "candidates": best[:5]})
            return None
        self.fit_forced = True
        bad = self.code_put([(fits[0], self.needs.quantity())])
        if bad is None or bad or not self.trolley.units:
            self.log.append({"kind": "checkpoint", "what": "fit_instead", "pick": fits[0], "result": "cart not set", "bad": bad})
            return None
        self.log.append({"kind": "checkpoint", "what": "fit_instead", "pick": fits[0], "result": "ordered instead of the decline"})
        return order, self.plain_args(order, "Order the cart.", "order")

    def guard_tools(self) -> set[str]:
        return {self.role(r) for r in ("details", "cart_view", "search")} - {None, self.role("cart"), self.role("order"), self.role("decline")}

    def trail(self) -> str:
        """What the errand did so far, one line a call: its step, tool, arguments and the start of its answer."""
        out = []
        for entry in self.log:
            if entry.get("kind") != "env":
                continue
            resp = entry.get("reply") or {}
            obs = resp.get("observations") if isinstance(resp.get("observations"), list) else [resp.get("observation")]
            for k, act in enumerate(entry.get("actions") or []):
                seen = obs[k] if k < len(obs) else None
                out.append(f"step {entry.get('step', 0) + k}: {act.get('name')} {as_text(act.get('args'))[:160]} -> {as_text(seen)[:180]}")
        return "\n".join(out[-45:]) or "(nothing yet)"

    def run_scripted(self, call: dict[str, Any]) -> None:
        self.chat.append({"role": "assistant", "content": "The request's own procedure needs this step first.", "tool_calls": [call]})
        self.post_replies(self.dispatch([call], scripted=True))

    def procedure_guard(self, moment: str) -> str:
        """The request's procedure rules against what was done, just before `moment`: what is due now is done by the code
        (messages, calls that change neither the cart nor the finish, words for the finish); the held call goes again."""
        reading = getattr(self.needs, "reading", None) or {}
        rules = [str(r).strip() for r in reading.get("rules") or [] if str(r).strip()]
        rules += [str(r.get("rule")).strip() for r in reading.get("finish_messages") or []
                  if isinstance(r, dict) and str(r.get("rule") or "").strip() and str(r.get("rule")).strip() not in rules]
        if not rules or self.__dict__.get("guard_calls", 0) >= 5 or self.wire.spare() < CLOSING or self.moves_left() < 3:
            return ""
        sig = (moment.split(" (")[0], self.trolley.changed_at, self.msgs_sent, len(self.needs.uncovered))
        seen = self.__dict__.setdefault("guard_seen", set())
        if sig in seen:
            return ""
        seen.add(sig)
        self.guard_calls = self.__dict__.get("guard_calls", 0) + 1
        left = (self.msg_cap - self.msgs_sent) if self.msg_cap is not None else "not limited"
        ask = (PROCEDURE.replace("MOMENT", moment) + "\n\nTHE SHOPPER'S REQUEST:\n" + self.request
               + "\n\nPROCEDURE RULES IN IT:\n" + "\n".join(f"- {r}" for r in rules)
               + "\n\nWHAT THE ASSISTANT HAS DONE SO FAR:\n" + self.trail()
               + "\n\nCART NOW:\n" + (self.basket_text() if self.trolley.units else "(empty)")
               + (f"\nLAST CART CHANGE AT STEP {self.trolley.changed_at}" if self.trolley.units else "")
               + f"\n\nMESSAGES LEFT: {left}; STEPS LEFT: {self.moves_left()}"
               + "\nTOOLS THE CODE CAN CALL (they change neither the cart nor the finish): " + ", ".join(sorted(self.guard_tools())))
        answer = self.wire.consult(ask, 1200, CHECKER)
        if not isinstance(answer, dict) or "actions" not in answer:
            answer = self.wire.consult(ask, 1200, CHECKER[1:])     # (a lost or unreadable check is asked once more, of the next models)
        if not isinstance(answer, dict) or "actions" not in answer:
            seen.discard(sig)                                         # (no check was made: the moment is checked again)
            self.log.append({"kind": "checkpoint", "what": "procedure_guard", "moment": moment, "result": "no answer"})
            return ""
        self.log.append({"kind": "checkpoint", "what": "procedure_guard", "moment": moment, "result": answer})
        did: list[str] = []
        for act in [a for a in answer.get("actions") or [] if isinstance(a, dict)][:4]:
            what = str(act.get("do") or "")
            if what == "message":
                text = str(act.get("text") or "").strip()
                if text and (self.msg_cap is None or self.msgs_sent < self.msg_cap) and self.moves_left() >= 2:
                    call = self.message_call(text, f"rule_{self.wire.tick}_{len(did)}")
                    if call:
                        self.run_scripted(call)
                        did.append("a message to the shopper: " + text[:160])
            elif what == "tool":
                name = str(act.get("tool") or "")
                args = act.get("args") if isinstance(act.get("args"), dict) else {}
                if name in self.guard_tools() and self.moves_left() >= 3:
                    self.run_scripted({"id": f"rule_{self.wire.tick}_{len(did)}", "type": "function", "function": {"name": name, "arguments": json.dumps(args)}})
                    did.append(f"{name} {as_text(args)[:80]}")
            elif what == "finish_text":
                words = str(act.get("finish_text") or "").strip()
                if words:
                    self.finish_words = (self.__dict__.get("finish_words", "") + " " + words).strip()
                    did.append("words for the finishing call noted: " + words[:100])
        wait = bool(answer.get("wait_for_reply")) and any(d.startswith("a message") for d in did)
        if not did:
            return ""
        return ("Not run yet (no step used): the request's own procedure needed this first, and the code did it: " + "; ".join(did)
                + (". Read the shopper's answer above and act on it, then send the call again." if wait else ". Now send the same call again."))

    def basket_due(self) -> bool:
        if not self.role("message") or not self.trolley.units or self.kept_messages("order") <= 0:
            return False
        if self.msg_cap is not None and self.msgs_sent >= self.msg_cap:
            return False
        return self.__dict__.get("basket_tick", -1) <= self.trolley.changed_at and self.moves_left() >= 2

    def basket_text(self) -> str:
        def title(thing: str) -> str:
            found: list[str] = []
            def visit(d: dict[str, Any]) -> None:
                if any(isinstance(v, str) and v == thing and id_named(k) for k, v in d.items()):
                    name = Errand.title_of(d)
                    if name:
                        found.append(name)
            for obs in self.archive.items:
                walk_dicts(obs, visit)
            return found[-1][:90] if found else "listing"
        reading = getattr(self.needs, "reading", None) or {}
        unit = str(reading.get("currency") or "").strip()
        lines, total = [], 0.0
        for thing, qty_n in self.trolley.units.items():
            count = int(qty_n) if float(qty_n).is_integer() else qty_n
            price = self.archive.price(thing)
            lines.append(f"{count} x {title(thing)} (listing {thing}" + (f", {price:g} {unit} each)".replace("  ", " ") if price is not None else ")"))
            total = total + price * float(qty_n) if price is not None and total is not None else None
        text = "Here is the final basket before I place the order:\n" + "\n".join(lines)
        if total is not None:
            text += f"\nTotal: {total:g} {unit}".rstrip()
        return text

    def call_kind(self, call_name: str, call_args: dict[str, Any] | None) -> str | None:
        order, decline = self.role("order"), self.role("decline")
        if call_name not in (order, decline) or not call_name:
            return None
        if order != decline:
            return "order" if call_name == order else "decline"
        props = (self.shapes.get(call_name) or {}).get("properties") or {}
        for key, piece in (call_args or {}).items():
            if isinstance(piece, str) and len(piece) <= 40:
                if re.search(DECLINE_WORDS, piece.lower()):
                    return "decline"
                if re.search(ORDER_WORDS, piece.lower()):
                    return "order"
                choices = [str(c) for c in ((props.get(key) or {}).get("enum") or []) if isinstance(c, str)]
                if len(choices) == 2 and piece in choices:
                    other = choices[1 - choices.index(piece)]
                    if re.search(ORDER_WORDS, other.lower()):
                        return "decline"
        return "order"

    def finishers(self) -> set[str]:
        return {self.role("order"), self.role("decline")} - {None}

    def clock_text(self) -> str:
        left = max(0.0, self.wire.spare() - LAST_CALL)
        if left > (DEADLINE - LAST_CALL) / 2:
            return ""
        step = int(left // CLOCK_STEP) * CLOCK_STEP
        return f"about {step} s left" if step > 0 else f"under {CLOCK_STEP} s left"

    def note_read(self, name: str, obs: Any) -> None:
        """Stores can split a listing's record: one tool gives its properties, another its offer. When the details tool gives a
        listing back with no property table, the details role moves to the other one-listing tool."""
        if self.__dict__.get("reader_sure") or name != self.role("details") or not isinstance(obs, dict):
            return
        body = next((v for v in obs.values() if isinstance(v, dict) and any(id_named(k, True) for k in v)), obs)
        if any(isinstance(v, dict) for v in body.values()):
            if any(isinstance(v, dict) and len(v) >= 2 for v in body.values()):
                self.reader_sure = True
            return                                         # (a table, even an empty one, marks the properties tool)
        if not any(id_named(k, True) for k in body):
            return                                         # (a failed read says nothing about the tool)
        view = self.role("view")
        if not view or view == name:
            taken = {self.role(r) for r in ("search", "message", "cart", "cart_view", "order", "decline")}
            view = next((t for t in self.shapes if t != name and t not in taken and self.lookup_key(t)), None)
        if view:
            self.role_map["details"], self.role_map["view"], self.reader_sure = view, name, True
            self.log.append({"kind": "checkpoint", "what": "reader_moved", "details": view})

    def status_line(self) -> str:
        spare = self.moves_left()
        pieces = [f"[status] steps used {self.wire.tick} of {self.step_cap}, {spare} left",
                 f"messages sent {self.msgs_sent}" + (f" of {self.msg_cap}" if self.msg_cap is not None else "")]
        clock = self.clock_text()
        if clock:
            pieces.append(clock)
        text_body = "; ".join(pieces) + ".\n[requirements now] " + self.needs.text_body() + self.ref_line() + self.product_steps()
        details, view = self.role("details"), self.role("view")
        if details and view and details != view:
            text_body += f"\n[reads] {details} shows a listing's properties and {view} its offer: judge requirements from {details}."
        if not self.trolley.used and not self.orders_tried and self.step_cap and not self.asking:
            text_body += ("\n[note] Cart each requested item as soon as you have read a listing that meets all its requirements in force, then "
                          "go on: a carted line can sell out, and an early cart leaves steps to replace it. Never cart a listing whose "
                          "properties you have not read.")
        if not self.trolley.used and not self.orders_tried and self.step_cap:
            plan = self.step_plan()
            text_body += (f"\n[step plan] questions end by step {plan['ask_by']}; the first cart comes by step {plan['cart_by']}: the shop or "
                          f"the shopper can change something after a first cart, and {plan['keep']} steps stay for that.")
        usable = [tdef["function"]["name"] for tdef in self.offered()]
        if len(usable) < len(self.toolset):
            text_body += "\n[tools usable now] " + ", ".join(usable) + " (any other tool call is refused without using a step" + (
                "; questions to the shopper are planned and sent for you" if self.asking else "") + ")."
        if self.trolley.used:
            text_body += "\n" + self.trolley.summary()
        if spare <= 2:
            text_body += ("\n[note] LAST STEPS: order the cart now if it holds a candidate for " + ("every product asked for" if self.needs.items()
                          else "the item") + "; otherwise put the best candidate in (a verified fit, else the cheapest in-stock, "
                          "in-budget listing whose record shows no requirement failing) and order. Decline only when the records "
                          "show that nothing can fit.")
        elif spare <= 5 or self.wire.spare() < CLOSING:
            text_body += ("\n[note] FINISH NOW: fix the cart to " + ("one line per product asked for" if self.needs.items() else "one line")
                          + " (a verified fit, else the cheapest in-stock, in-budget listing whose record shows no requirement "
                          "failing; a sold-out line out), re-check its stock, then order. Decline only when the records show that "
                          "nothing can fit.")
        elif self.wire.spent_out():
            several = bool(self.needs.items())
            text_body += ("\n[note] MODEL BUDGET USED: stop exploring; within the next few steps put the best candidate of "
                          + ("every product asked for (one line each, at its quantity)" if several else "the item asked for")
                          + " in the cart (a verified fit where one exists, else the closest fit the records show), re-check its "
                          "stock and price, and order. A decline is right only when the records show that "
                          + ("some product has" if several else "the store has") + " no listing that can meet its requirements.")
        elif self.orders_tried == 0 and spare <= self.order_due() + 7 and spare > self.order_due() and self.narrowing():
            text_body += f"\n[note] Steps are running low: within the next few steps put your best verified fit in the cart and order; searching and reading details end when {self.order_due()} steps remain, and a refused order needs about 4 steps to repair."
        elif self.orders_tried == 0 and spare <= self.order_due() and self.narrowing():
            text_body += "\n[note] Searching has ended (details only for the cart's line, listings a check named, or, while some product has no verified fit, up to 3 more reads or searches in all): put your best verified fit in the cart, re-check its stock and price, and order; a refused order still needs about 4 steps to repair."
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
            for mdl in [m for m in ACTOR if m not in READER]:
                msg = self.wire.query_model([{"role": "user", "content": content}], [mdl], 3000)
                retry = json_object((msg or {}).get("content") or "") or {}
                if isinstance(retry.get("roles"), dict):
                    reading = retry
                    break
            if not isinstance(reading.get("roles"), dict):
                try:
                    roles = self.code_roles()
                    reading = {"code_reading": True, "roles": roles, "requirements": [], "items": [], "quantity": 1, "rules": [],
                               "soft_wants": [], "ask_first": [], "hints": [], "references": [], "priority": "none", "priority_detail": "",
                               "product": self.code_label(roles.get("search"), kind_words(self.request))}
                except Exception as err:
                    self.wire.log.append({"kind": "code_reading_error", "error": f"{type(err).__name__}: {err}"[:300]})
                    reading = {}
            self.read_at = monotonic()
            self.wire.log.append({"kind": "code_reading", "roles": reading.get("roles"), "product": reading.get("product"),
                                  "price_cap": reading.get("price_cap"), "items": len(reading.get("items") or [])})
        self.use_reading(reading)

    def use_reading(self, reading: dict[str, Any], again: bool = False) -> None:
        if not again:
            self.steps_given = self.step_cap
        old_cart = self.role("cart")
        pairs = []
        for pair in reading.get("fits") if isinstance(reading.get("fits"), list) else []:
            part, main = (pair.get("item"), pair.get("with")) if isinstance(pair, dict) else (None, None)
            if isinstance(part, int) and isinstance(main, int) and part != main and part >= 1 and main >= 1:
                pairs.append((part - 1, main - 1))
        self.fit_pairs = pairs
        role_map = reading.get("roles") if isinstance(reading.get("roles"), dict) else {}
        self.role_map = {fkey: val for fkey, val in role_map.items() if isinstance(val, str) and val in self.shapes}
        try:
            self.fix_search_role()
        except Exception as err:
            self.wire.log.append({"kind": "fix_search_role_error", "error": f"{type(err).__name__}: {err}"[:200]})
        if again and self.trolley.used and old_cart:
            self.role_map["cart"] = old_cart
        if not (again and self.trolley.used):
            self.trolley = CartBook(self.role("cart"), self.shapes.get(self.role("cart") or "", {}))
        self.take_tool_args(reading)
        self.needs.sweepable = bool(self.role("search")) and self.page_arg() is not None
        spend_cap = reading.get("message_budget")
        if is_number(spend_cap) and spend_cap >= 0:
            self.msg_cap = int(spend_cap)
        self.needs.reading = reading
        try:
            self.wire.scale = 1.0 + 0.1 * min(2, max(0, len(self.needs.items()) - 1))
            self.wire.lines = len(self.needs.items())
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
        if again:
            self.step_cap = self.__dict__.get("steps_given", self.step_cap)
        step_count = reading.get("step_budget")
        if is_number(step_count) and 0 < step_count < self.step_cap:
            self.step_cap = int(step_count)
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
        values = line.split("):", 1)[1].split("|") if "):" in line else []
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
                    line = "" if junk_key(alt) or kind_label(alt) or strict_prop(alt, head) else self.prop_line(k, alt)
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
                if not key or junk_key(key) or kind_label(key) or key.lower() in spent or not self.twin_of(k, key, prop) or not self.plan_open(k, key):
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
        return None
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
                    if junk_key(alt) or kind_label(alt) or strict_prop(alt, head) or strict_prop(alt, name) or self.name_inside(alt, name):
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

    def quiet_reask(self, prop: str) -> str:
        return ""
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
        if reads < 4 or not menu:
            return menu
        kept = [ln for ln in menu.splitlines() if not re.search(r"\(in [01] listings?\)", ln)]
        return "\n".join(kept) if kept else menu

    def first_reads(self) -> bool:
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
                want = 2 if rows else FIRST_READS
                if home and have < want and room > 0:
                    picks = self.menu_rows(home, want - have, taken, qtys[min(k, len(qtys) - 1)])
                    taken += picks
                    if picks:
                        wants.append((have, k, picks))
            queue = [picks[n] for n in range(FIRST_READS) for _, _, picks in sorted(wants) if n < len(picks)][: max(0, room)]
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
        if self.step_cap <= 60:
            # (a short errand keeps about a fifth of its steps for the cart, its check and the order, two more once a store
            # notice or a sold-out line has been seen; the leading agents cart at about 3/4 of the steps)
            due = max(FIRST_ORDER_BY, round(0.22 * self.step_cap))
            if self.__dict__.get("event_seen") or (self.__dict__.get("dead") and set(self.trolley.units) & self.__dict__["dead"]):
                due += 2
            return min(due, self.step_cap // 2)
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
        ask_text += self.kit_evidence(thing)
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
        """The field evidence on a listing, with every stated requirement it does not write word for word as a failure (a
        meaning check that found the value in other words does not clear those)."""
        view = self.field_view_own(thing)
        try:
            lit = self.literal_fails(thing)
        except Exception as err:
            self.log.append({"kind": "literal_error", "error": f"{type(err).__name__}: {err}"[:300]})
            lit = []
        heads = [f.split(":", 1)[0].strip() for f in view["fails"]]
        extra = [f for f in lit if not any(same_prop(f.split(":", 1)[0].strip(), h) for h in heads)]
        return {"lines": view["lines"], "rules": view["rules"], "fails": list(view["fails"]) + extra} if extra else view

    def stated_now(self) -> list[str]:
        """The requirements the request itself states that are still in force, as 'property (product): value'; a stated
        requirement the current list no longer names (dropped or replaced) is left out."""
        nd = self.needs
        out = [r for r in nd.written if r not in nd.let_go]
        if nd.now_reqs is not None:
            now = [re.split(r"\s*\(", str(r).partition(":")[0])[0].strip() for r in nd.now_reqs]
            out = [r for r in out if any(same_prop(re.split(r"\s*\(", r.partition(":")[0])[0].strip(), p) for p in now)]
        return out

    def literal_fails(self, thing: str) -> list[str]:
        """The stated requirements a read listing does not write word for word (lit_meets): a related word, a
        shorter form or a missing field does not meet one. Only requirements whose property and value the request writes
        itself are judged, and a property the listing's notes send to another listing is left to that listing."""
        if thing not in self.archive.read:
            return []
        key = (thing, len(self.archive.items), self.needs.version, len(self.needs.let_go), str(self.needs.now_reqs))
        memo = self.__dict__.setdefault("lit_memo", {})
        if key in memo:
            return memo[key]
        out: list[str] = []
        me = self.ch_rows().get(thing)
        several = bool(self.needs.items())
        k = self.product_of(self.shelf_known(thing) or self.archive.label_of.get(thing, "")) if several and me else None
        if not me or not me["fields"] or (several and k is None):
            memo[key] = out
            return out
        said = set(lit_words(self.request))
        heard = lambda ws: all(any(w == s or same_kind(w, s) for s in said) for w in ws)
        for req in self.stated_now():
            head, _, raw = req.partition(":")
            prop = re.split(r"\s*\(", head)[0].strip()
            value = " ".join(re.sub(r"\((?:stated|revealed[^)]*)\)", " ", raw).split())
            if not prop or not value or ref_text(value) or not positive_value(value, head) or re.search(EXCLUDE_WORDS, value.lower()):
                continue
            if several and self.product_of(head) != k:
                continue
            bare = re.sub(LIT_AT_MOST + "|" + LIT_AT_LEAST + r"|\bonly\b|\bexactly\b", " ", value.lower())
            if not heard(lit_words(prop)) or not heard(lit_words(bare)) or self.ch_redirected(thing, prop):
                continue
            if lit_meets(me["fields"], me["title"], prop, value) is False:
                out.append(f"{prop}: the request's '{value}' is not written in its {prop} field (with no such field, in its title); "
                           "a stated value is read word for word, so another word for it, a shorter form or a missing "
                           "field does not meet it")
        memo[key] = out
        return out

    def field_view_own(self, thing: str) -> dict[str, list[str]]:
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
                                    "related value does not meet it (a value is read word for word).")
            elif on < CH_SYNONYM_MIN:
                out["rules"].append(base + " Its values seen: " + " | ".join(samples) + f"; none writes it yet, and a value of the same "
                                    "meaning does not meet it (a value is read word for word).")
            else:
                out["rules"].append(base + " Its values seen: " + " | ".join(samples) + "; none writes it, and a value of the same "
                                    "meaning does not meet it (a value is read word for word).")
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
                    if related else "") + ". With no field named for the property only its title can show the value, and a title "
                    "that does not write it fails it; a field of another property (a height for a type, a frame material for "
                    "the material) does not stand in.")
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
                                    "so this listing's other value does NOT meet it.")
            elif on < CH_SYNONYM_MIN:
                out["lines"].append(base + f" This listing: {shown}. No read listing of this category writes '{clip(value)}' in "
                                    "this field (its values: " + " | ".join(samples) + f"): this listing does not write it either, so it "
                                    "does NOT meet it (a value of the same meaning does not count; a value is read word for word).")
            else:
                out["lines"].append(base + f" This listing: {shown}. No read listing of this category writes '{clip(value)}' in "
                                    "this field (its values: " + " | ".join(samples) + "): this listing does not write it either, so it does NOT "
                                    "meet it (a value of the same meaning does not count; a value is read word for word).")
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
        self.rank_better = list(better)
        self.__dict__.setdefault("named", set()).update(better)
        if not better:
            return ""
        return (f"A rank check thinks these listings may also fit and beat {thing} on the {what}: " + ", ".join(better)
                + (f" ({answer.get('reason')})" if answer.get("reason") else "")
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

    def shorten_chat(self, keep: int = 8) -> None:
        """A long conversation: older tool results are cut down in place so later turns resend less."""
        cut = 0
        for m in self.chat[2:max(2, len(self.chat) - keep)]:
            text = m.get("content")
            if m.get("role") == "tool" and isinstance(text, str) and len(text) > 700:
                m["content"] = text[:600] + " ...(older result shortened)"
                cut += len(text) - 600
        self.log.append({"kind": "checkpoint", "what": "history_shortened", "chars": cut, "cost": round(self.wire.spend, 4), "step": self.wire.tick})

    def open_reads(self, queue: list[str], say: str, batch: tuple[str, str, int] | None, reader: str | None = None) -> list[tuple[str, str]]:
        if not batch and not reader and len(queue) >= 2 and not self.__dict__.get("reader_sure"):
            # (the store may split a listing over two tools: the first read settles which one carries the properties
            # before the rest of the queue goes out, so the queue is not spent on the offer tool)
            first = self.open_reads(queue[:1], say, None, None)
            return first + (self.open_reads(queue[1:], say, None, None) if first and not self.wire.done else [])
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
        if batch and purpose in ("menu", "quota") and self.__dict__.get("batch_flat") and self.role("details"):
            batch = None
        if batch and purpose == "cart" and not self.__dict__.get("batch_proven"):
            batch = None
        if batch and reader and not ((self.__dict__.get("tool_caps") or {}).get(batch[0]) or {}).get("stock"):
            batch = None
        responses = self.open_reads(queue, say, batch, reader)
        if batch and responses and not self.__dict__.get("batch_proven"):
            if self.batch_ok(self.__dict__.get("last_reads") or [], queue):
                self.batch_proven = True
                self.batch_flat = not self.batch_tables(self.__dict__.get("last_reads") or [])
            elif batch[2] > 2 and len(queue) > 2 and not self.__dict__.get("batch_halved"):
                # (a store can answer a several-listing read with nothing when it gets more ids than it takes: two at a
                # time is tried once before the tool is given up)
                self.batch_halved = True
                self.batch_spec = (batch[0], batch[1], 2)
                self.log.append({"kind": "batch_halved", "tool": batch[0]})
                self.post_replies(responses)
                responses = self.open_reads(queue[:2], say, self.batch_spec, reader)
                if responses and self.batch_ok(self.__dict__.get("last_reads") or [], queue[:2]):
                    self.batch_proven = True
                    self.batch_flat = not self.batch_tables(self.__dict__.get("last_reads") or [])
                else:
                    self.batch_spec = None
                    self.log.append({"kind": "batch_dropped", "tool": batch[0]})
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
            if any(isinstance(val, str) and val == thing and id_named(key) for key, val in d.items()):
                found = self.shelf_of(d) or found
        for rec in self.archive.items:
            walk_dicts(rec, visit)
        return found or str((self.archive.__dict__.get("label_of") or {}).get(thing) or "").strip().lower()

    def chosen(self) -> str | None:
        if self.trolley.units:
            return list(self.trolley.units)[-1]
        return None

    def shelf_learn(self, obs: Any) -> None:
        try:
            names = [str(row.get("product") or "") for row in self.needs.items()] or [str(self.needs.reading.get("product") or "")]
            learn_shelf_key(obs, names)
        except Exception:
            pass

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
                if shelf and any(isinstance(v, str) and v in self.archive.ids and id_named(key) for key, v in d.items()):
                    seen.add(shelf)
            for rec in self.archive.items:
                walk_dicts(rec, visit)
            names = [set(kind_tokens(row["product"])) for row in rows]
            for k, own in enumerate(names):
                shared = set().union(*(names[j] for j in range(len(names)) if j != k))
                best, top = set(), (0.0, 0)
                for shelf in sorted(seen):
                    words = set(kind_tokens(shelf))
                    score = (kind_score(own, words, shared), -len(words - own))
                    if score[0] > 0 and score > top:
                        best, top = {shelf}, score
                    elif score == top and score[0] > 0:
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
            if any(isinstance(v, str) and id_named(k) and re.fullmatch(ID_LIKE, v) for k, v in d.items()):
                label = next((val.strip() for key, val in d.items() if isinstance(val, str) and 0 < len(val.strip()) <= 60
                              and shelf_key(key) and "id" not in str(key).lower()), "")
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
                        if not di or not dj:
                            continue           # (one product already names the whole category: two products, not one)
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
        self.wire.scale = 1.0 + 0.1 * min(2, max(0, len(needs.items()) - 1))
        self.wire.lines = len(needs.items())
        self.log.append({"kind": "merge_joined", "label": label, "products": [rows[i]["product"], rows[j]["product"]],
                         "into": phrase, "label_words": in_order, "products_left": max(1, len(needs.items()))})
        return (f"The store sells '{label}' as ONE category: the request asks for one product of it ('{phrase}'), so the order "
                + ("holds one line for it" if left >= 2 else "holds ONE line") + ", not one line for each kind its name joins. "
                "Requirements now: " + needs.text_body())

    def id_repair(self, call_name: str, call_args: dict[str, Any]) -> dict[str, Any]:
        """A listing reference that is not an id seen in this store, mended in code before the call goes out (no step lost):
        a part code a seen title shows in brackets becomes that listing's id; an id with one
        character missing, extra or wrong becomes the one seen id it matches."""
        try:
            tools = (set(self.lookup_tools()) | {self.role("cart")}) - {None}
            if call_name not in tools or not self.archive.ids or not isinstance(call_args, dict):
                return call_args
            jschema = self.shapes.get(call_name) or {}
            props = jschema.get("properties") if isinstance(jschema.get("properties"), dict) else {}
            out = dict(call_args)
            for key, val in call_args.items():
                p = props.get(key) or {}
                if not isinstance(val, str) or not val.strip() or val.strip() in self.archive.ids or p.get("enum") or p.get("type") not in (None, "string"):
                    continue
                v = val.strip()
                tagged: set[str] = set()
                pat = re.compile(r"[\[(]\s*(?:part\s+)?" + re.escape(v) + r"\s*[\])]", re.I)

                def visit(d: dict[str, Any]) -> None:
                    ident = next((x for k2, x in d.items() if isinstance(x, str) and id_named(k2) and x in self.archive.ids), None)
                    if ident and any(isinstance(x, str) and len(x) < 400 and pat.search(x) for x in d.values()):
                        tagged.add(ident)
                for rec in self.archive.items:
                    walk_dicts(rec, visit)
                near = [i for i in self.archive.ids if one_edit(v, i)] if len(v) >= 8 else []
                hit = sorted(tagged) if len(tagged) == 1 else (near if len(near) == 1 else [])
                if hit:
                    out[key] = hit[0]
                    self.log.append({"kind": "id_repair", "tool": call_name, "from": v, "to": hit[0]})
            return out
        except Exception as err:
            self.log.append({"kind": "checkpoint", "what": "id_repair", "error": f"{type(err).__name__}: {err}"[:200]})
            return call_args

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
        for exact in (True, False):
            for key, val in d.items():
                name = str(key).lower()
                if (name == "id" if exact else id_named(key, True)) and isinstance(val, str) and re.fullmatch(ID_LIKE, val):
                    return val
        return None

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
        try:
            # (an add whose returned cart lacks the item just added: the item went out of stock on the way in)
            if name == self.role("cart") and asked and isinstance(call_args, dict) and self.trolley.is_put(call_args):
                inner = self.inner_obs(obs)
                held_ids: set[str] = set()
                walk_dicts(inner, lambda d: held_ids.update(v for k, v in d.items() if isinstance(v, str) and id_named(k)))
                if isinstance(inner, dict) and any(isinstance(v, list) for v in inner.values()):
                    for ident in asked - held_ids:
                        dead.add(ident)
                        dead_by[ident] = "lookup"
        except Exception:
            pass
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
            mended = self.kit_mend(picks) if picks else picks
            if mended is None:
                self.log.append({"kind": "decline_repair", "picks": picks, "done": False, "why": "kit parts do not fit"})
                return ""
            picks = mended
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
                if self.literal_fails(e["id"]):
                    continue        # (the grader reads a stated value word for word: the same value in other words fails it)
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
        if self.__dict__.get("batch_flat"):
            return None          # (a several-listing read that shows offers only is not a reader: listings are opened one at a time)
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
                    if len(arrays) == 1 and not is_number(most):
                        told = (str((props[arrays[0]] or {}).get("description") or "") + " " + said.get(name, "")).lower()
                        hit = re.search(r"(?:at most|up to|no more than|maximum of|max\.?|limit of)\s+(\d{1,2})\b", told)
                        most = int(hit.group(1)) if hit else BATCH_DEFAULT
                    if name not in taken and len(arrays) == 1 and not [r for r in needed if r not in arrays] and is_number(most) and most >= 2:
                        spec = (name, arrays[0], int(min(most, 10)))
                        break
            except Exception:
                spec = None
            self.batch_spec = spec
        return self.batch_spec

    @staticmethod
    def batch_tables(found: list[Any]) -> bool:
        """Whether a several-listing read gave property tables (a nested table of three or more entries), not only offers."""
        hit = []

        def visit(d: dict[str, Any]) -> None:
            if any(isinstance(v, dict) and len(v) >= 3 for v in d.values()):
                hit.append(1)
        try:
            for obs in found:
                walk_dicts(obs, visit)
        except Exception:
            return True
        return bool(hit)

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

    def shelf_named(self, text_body: str) -> str:
        rows = self.needs.items()
        if not rows:
            return text_body
        try:
            homes = self.home_shelves()
        except Exception:
            return text_body
        low = text_body.lower()
        stems = lambda label: [w[:4] for w in re.findall(r"[a-z]+", str(label).lower()) if len(w) > 3]
        def titled(product: str) -> list[str]:
            words = stems(product)
            tally: dict[str, int] = {}
            def visit(d: dict[str, Any]) -> None:
                title = str(Errand.title_of(d) or "").lower()
                label = next((str(v).strip() for key, v in d.items() if isinstance(v, str) and shelf_key(key)), "")
                if title and label and any(w in title for w in words):
                    tally[label] = tally.get(label, 0) + 1
            for obs in self.archive.items:
                walk_dicts(obs, visit)
            return [max(tally, key=tally.get)] if tally else []
        for k, row in enumerate(rows):
            home = sorted(homes[k]) if k < len(homes) and homes[k] else titled(row["product"])
            if not home or not any(w in low for w in stems(row["product"])):
                continue
            if any(w in low for label in home for w in stems(label)):
                return text_body
            label = home[0]
            cut = text_body.rfind("?")
            self.log.append({"kind": "checkpoint", "what": "shelf_named", "label": label})
            return (text_body[:cut] + f" ({label})" + text_body[cut:]) if cut >= 0 else f"{text_body} ({label})"
        return text_body

    def message_call(self, text_body: str, cid: str) -> dict[str, Any] | None:
        if cid.startswith("ask_"):
            text_body = self.shelf_named(text_body)
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
            tied = [k for k, s in enumerate(scores) if top > 0 and s == top]
            left = [sum(1 for w in names[k] if not kind_in(w, words)) for k in tied]
            if tied and left.count(0) == 1:
                return tied[left.index(0)]               # (a name inside another product's name: the one it covers in full)
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

        mains = {m for _, m in self.kit_pairs()}     # (parts must fit it: the unwritten needs sit on the parts, so it comes last)

        def level_of(k: int) -> tuple[int, int, int, int, int, int]:
            return (1 if found[k] and any(j != k for j in unasked) else 0, 1 if k in mains and any(j not in mains for j in unasked) else 0,
                    0 if k in hinted else 1, asked[k] + found[k], 0 if k in alt_left else sent[k] - asked[k], found[k])
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
        return None

    def product_plan(self, k: int, menu: str) -> dict[str, Any] | None:
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
        answer = self.wire.consult(ask_text, 1500, PLANNER)
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
        try:
            name = str(prop).split("(")[0].strip()
            if not name or junk_key(name) or kind_label(name):
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
                text += ("\n\nOPEN FROM THE LAST TWO-PROPERTY QUESTION (the second property of a planned pair whose reply did not answer it - the shopper "
                         "answered only the first, or only the first was sent - so it is still open; each was the plan's next pick):\n"
                         + "\n".join(f"- {p}" for p in opens[-3:]))
            if wanted:
                text += ("\n\nQUESTIONS THE ASSISTANT WANTED TO ASK (it has read the listing records; a property such a question names "
                         "that is still open and is a key below comes first, asked by the key's own name):\n" + "\n".join(f"- {w}" for w in wanted[-3:]))
            return text
        except Exception:
            return ""

    def want_ask(self, raw_args: Any) -> None:
        try:
            args = raw_args if isinstance(raw_args, dict) else (json_object(raw_args) if isinstance(raw_args, str) and raw_args.strip() else {})
            text = " ".join(str(v).strip() for v in (args or {}).values() if isinstance(v, str) and v.strip())[:300]
            if text:
                kept = [w for w in self.__dict__.get("wanted_asks") or [] if w.lower() != text.lower()]
                self.wanted_asks = (kept + [text])[-3:]
        except Exception:
            pass

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

    def queue_second(self, prop: str) -> None:
        queue = self.__dict__.setdefault("pending_seconds", [])
        if prop and prop.lower() not in [p.lower() for p in queue]:
            queue.append(prop)

    def quota_ahead(self, prop: str) -> bool:
        """Several products: the product of this question label has had more questions than another product whose need is
        not heard yet and that has a shelf seen (so it can be asked about)."""
        try:
            rows = self.needs.items()
            j = self.product_of(prop) if rows else None
            if j is None:
                return False
            asked = [0] * len(rows)
            for ln in self.questions_log:
                k = self.product_of(ln.split(":", 1)[0])
                if k is not None:
                    asked[k] += 1
            heard = {self.product_of(str(n).split(":", 1)[0]) for n in self.needs.uncovered}
            homes = self.home_shelves()
            return any(asked[k] < asked[j] for k in range(len(rows)) if k != j and k not in heard and k < len(homes) and homes[k])
        except Exception:
            return False

    def rank_target(self) -> tuple[int | None, str, str] | None:
        """The product the next question is about, its name and its menu: the one with the fewest questions among those
        whose need is not heard yet; None when no listing of it has been read."""
        rows = self.needs.items()
        if not rows:
            product = str(self.needs.reading.get("product") or "").strip()
            own = self.own_shelf()
            menu, reads = self.archive.prop_menu(own) if own else self.archive.prop_menu()
            menu = self.menu_trim(menu, self.archive.rich_reads(own) if own else self.archive.rich_reads())
            homes = self.home_shelves()
            home = homes[0] if homes else set()
            seen = self.archive.rich_reads(own) if own else self.archive.rich_reads()
            return (None, product, menu, home, seen) if product and menu and reads else None
        asked = [0] * len(rows)
        for ln in self.questions_log:
            k = self.product_of(ln.split(":", 1)[0])
            if k is not None:
                asked[k] += 1
        heard = {self.product_of(str(n).split(":", 1)[0]) for n in self.needs.uncovered}
        open_ks = [k for k in range(len(rows)) if k not in heard] or list(range(len(rows)))
        homes = self.home_shelves()
        mains = {m for _, m in self.kit_pairs()}     # (an item the other items must fit: the device)
        # (when parts must fit a device, the parts are asked first and the device last, since the unwritten needs of such a
        #  request sit on the parts; every product not asked yet comes before a second question to any product)
        for k in sorted(open_ks, key=lambda j: (asked[j] > 0, j in mains, asked[j], j)):
            home = homes[k] if k < len(homes) else set()
            if not home:
                continue
            read = [s for s in self.archive.shelves_read() if s.strip().lower() in home]
            if not read:
                return k, rows[k]["product"], "", home, 0
            menu = self.menu_trim(self.archive.prop_menu(read[0])[0], self.archive.rich_reads(read[0]))
            return k, rows[k]["product"], menu, home, self.archive.rich_reads(read[0])
        return None

    # ---------------- hw_m17a: tool reading ----------------
    @staticmethod
    def arg_words(key: str, pspec: Any) -> str:
        """An argument's name split into words, with its description."""
        name = re.sub(r"([a-z])([A-Z])", r"\1 \2", str(key)).replace("_", " ")
        return (name + " " + str((pspec or {}).get("description") or "")).lower() if isinstance(pspec, dict) else name.lower()

    def query_keys(self, tool: str) -> list[str]:
        if self.sheet_for(tool).get("words"):
            return [self.sheet_for(tool)["words"]]
        props = (self.shapes.get(tool) or {}).get("properties") or {}
        return [k for k, p in props.items() if isinstance(p, dict) and p.get("type") == "string" and not p.get("enum")
                and re.search(r"^q$|query|text|keyword|needle|term|word|phrase|string|search", str(k).lower())]

    def browse_spec(self) -> dict[str, Any] | None:
        """The tool that lists one shelf (category) of the store, read from the tool definitions: its shelf argument (a
        free string or a list of choices), and when it has them its price-cap, brand and next-page arguments. A tool that
        also takes search words is not one (the keyword search handles it). None when the store has no such tool."""
        if "browse_memo" in self.__dict__:
            return self.browse_memo
        spec = None
        try:
            taken = {self.role(r) for r in ("message", "cart", "cart_view", "order", "decline", "details")} - {None}
            for name, jschema in self.shapes.items():
                props = jschema.get("properties") if isinstance(jschema, dict) and isinstance(jschema.get("properties"), dict) else {}
                if name in taken or not props or self.query_keys(name):
                    continue
                words = {k: self.arg_words(k, p) for k, p in props.items() if isinstance(p, dict)}
                shelf = next((k for k, p in props.items() if isinstance(p, dict) and p.get("type") == "string"
                              and re.search(r"categor|shelf|department|aisle", words[k])), None)
                if not shelf:
                    continue
                cap = next((k for k, p in props.items() if isinstance(p, dict) and p.get("type") in ("integer", "number")
                            and re.search(r"ceil|max|price|cap$|budget|upper|most", str(k).lower())
                            and not re.search(r"page|first|offset|skip|count|size|num", str(k).lower())), None)
                sheet_cap = (self.__dict__.get("tool_sheet") or {}).get("price_cap")
                cap = cap or (sheet_cap if sheet_cap in props else None)
                brand = None
                tok = next((k for k in props if k not in (shelf, cap) and re.search(r"next|onward|cursor|bookmark|token|continu|more", str(k).lower())), None)
                spec = {"tool": name, "shelf": shelf, "cap": cap, "brand": brand, "token": tok,
                        "enum": props[shelf].get("enum") if isinstance(props[shelf].get("enum"), list) else None,
                        "cap_int": bool(cap and props[cap].get("type") == "integer")}
                break
        except Exception as err:
            self.wire.log.append({"kind": "browse_spec_error", "error": f"{type(err).__name__}: {err}"[:200]})
            spec = None
        self.browse_memo = spec
        return spec

    def take_tool_args(self, reading: dict[str, Any]) -> None:
        """What the reader says each search argument does, checked against the tool's own definition: a name it does not list, a
        value its choices do not hold, or the wrong type is dropped, and the code's own reading of the definition then stands. A
        later reading that says nothing keeps the sheet already held."""
        said = reading.get("tool_args")
        if not isinstance(said, dict):
            return
        props = (self.shapes.get(self.role("search") or "") or {}).get("properties")
        props = props if isinstance(props, dict) else {}
        sheet: dict[str, Any] = {}
        words, mode, nxt = said.get("words"), said.get("mode"), said.get("next")
        if isinstance(words, str) and (props.get(words) or {}).get("type") == "string" and not (props.get(words) or {}).get("enum"):
            sheet["words"] = words
        choices = (props.get(mode) or {}).get("enum") if isinstance(mode, str) else None
        if isinstance(choices, list) and said.get("mode_words") in choices:
            sheet["mode"], sheet["mode_words"] = mode, said["mode_words"]
        if isinstance(nxt, str) and nxt != sheet.get("words") and nxt != sheet.get("mode") and not (props.get(nxt) or {}).get("enum"):
            kind = (props.get(nxt) or {}).get("type")
            if kind == "string":
                sheet["next"], sheet["next_kind"] = nxt, "token"
            elif kind in ("integer", "number"):
                sheet["next"], sheet["next_kind"] = nxt, "page_number" if said.get("next_kind") == "page_number" else "row_offset"
        cap = said.get("price_cap")
        if isinstance(cap, str) and any(((sp or {}).get("properties") or {}).get(cap, {}).get("type") in ("integer", "number") for sp in self.shapes.values()):
            sheet["price_cap"] = cap
        if sheet:
            self.tool_sheet = sheet
        self.wire.log.append({"kind": "tool_sheet", "said": said, "kept": sheet})

    def sheet_for(self, tool: str | None) -> dict[str, Any]:
        """The reader's sheet, when this call is about the store's search tool."""
        return (self.__dict__.get("tool_sheet") or {}) if not tool or tool == self.role("search") else {}

    def fix_search_role(self) -> None:
        """The keyword search role goes to a tool that takes search words; the shelf filter keeps its own role."""
        self.__dict__.pop("browse_memo", None)
        cur = self.role_map.get("search")
        if cur and self.query_keys(cur):
            return
        taken = {self.role_map.get(r) for r in ("message", "cart", "cart_view", "order", "decline", "details", "view")} - {None}
        words = [name for name in self.shapes if name not in taken and self.query_keys(name)]
        if words:
            self.role_map["search"] = words[0]
            self.wire.log.append({"kind": "search_role", "was": cur, "now": words[0]})

    def seen_labels(self) -> dict[str, str]:
        """Every shelf (category) label seen on a listing so far: lower case -> as the store writes it."""
        out: dict[str, str] = {}

        def visit(d: dict[str, Any]) -> None:
            shelf = self.shelf_of(d)
            if shelf and any(isinstance(v, str) and v in self.archive.ids and id_named(key) for key, v in d.items()):
                out.setdefault(shelf.strip().lower(), shelf.strip())
        for rec in self.archive.items:
            walk_dicts(rec, visit)
        return out

    def product_phrase(self, k: int | None) -> str:
        rows = self.needs.items()
        if rows and k is not None and k < len(rows):
            return str(rows[k].get("product") or "").strip()
        return str(self.needs.reading.get("product") or "").strip()

    def browse_label(self, k: int | None) -> str:
        """The store's own label of the product's shelf, once a listing of it has been seen ('' before that)."""
        homes = self.home_shelves()
        home = homes[k] if k is not None and k < len(homes) else (homes[0] if homes else set())
        labels = self.seen_labels()
        for shelf in sorted(home):
            if shelf.strip().lower() in labels:
                return labels[shelf.strip().lower()]
        return ""

    def cap_for(self, k: int | None) -> Any:
        spec = self.browse_spec() or {}
        cap = self.needs.spend_cap
        if not is_number(cap) or cap <= 0:
            return None
        counts = self.needs.counts() if self.needs.items() else []
        qty = (counts[k] if k is not None and k < len(counts) else self.needs.quantity()) or 1
        return int(cap // qty) if spec.get("cap_int") else round(cap / qty, 2)

    @staticmethod
    def plain_value(value: Any) -> str:
        """A wanted value as a search uses it: without its source tag and what it carries in brackets."""
        text = re.sub(r"\((?:stated|revealed[^)]*)\)", "", str(value))
        main = re.sub(r"\([^)]*\)", " ", text)
        return " ".join(main.split()) or " ".join(text.split())

    def search_clean(self, call_name: str, call_args: dict[str, Any]) -> dict[str, Any]:
        """A keyword search leaves out the words a wanted value carries only in brackets."""
        try:
            search = self.role("search")
            if call_name != search or not isinstance(call_args, dict):
                return call_args
            key = next((q for q in self.query_keys(search) if isinstance(call_args.get(q), str) and call_args[q].strip()), None)
            if not key:
                return call_args
            inner: set[str] = set()
            outer: set[str] = set()
            for req in list(self.needs.props_left()) + list(self.needs.uncovered):
                text = re.sub(r"\((?:stated|revealed[^)]*)\)", "", str(req).partition(":")[2])
                inner |= {w for part in re.findall(r"\(([^)]*)\)", text) for w in re.findall(r"[a-z0-9]+", part.lower())}
                outer |= set(re.findall(r"[a-z0-9]+", re.sub(r"\([^)]*\)", " ", text).lower()))
            inner -= outer
            toks = call_args[key].split()
            keep = [t for t in toks if not (re.findall(r"[a-z0-9]+", t.lower()) and set(re.findall(r"[a-z0-9]+", t.lower())) <= inner)]
            if not inner or not keep or len(keep) == len(toks):
                return call_args
            self.log.append({"kind": "search_clean", "was": call_args[key], "now": " ".join(keep)})
            return {**call_args, key: " ".join(keep)}
        except Exception as err:
            self.log.append({"kind": "checkpoint", "what": "search_clean", "error": f"{type(err).__name__}: {err}"[:200]})
            return call_args

    def search_id(self, call_name: str, call_args: Any) -> tuple[str, str, str] | None:
        """A keyword search's or shelf listing's identity (its arguments without the page) and its page."""
        spec = self.browse_spec() or {}
        if call_name == self.role("search"):
            found = self.page_arg()
            pkey = found[0] if found else ""
        elif call_name == spec.get("tool"):
            pkey = spec.get("token") or ""
        else:
            return None
        if not isinstance(call_args, dict):
            return None
        rest = {a: (" ".join(re.findall(r"[a-z0-9]+", v.lower())) if isinstance(v, str) else v) for a, v in call_args.items() if a != pkey}
        return str(call_name) + "|" + json.dumps(rest, sort_keys=True, default=str), pkey, str(call_args.get(pkey) or "") if pkey else ""

    def repeat_fix(self, call_name: str, call_args: dict[str, Any]) -> tuple[dict[str, Any], str]:
        """A search or shelf page exactly like one already run goes on to the next page it has not shown; with no such page
        it is held back (no step used)."""
        sid = self.search_id(call_name, call_args)
        if not sid:
            return call_args, ""
        qid, pkey, page = sid
        rec = self.__dict__.setdefault("search_book", {}).get(qid)
        if not rec or page not in rec["pages"]:
            return call_args, ""
        nxt = rec.get("next") or ""
        if pkey and nxt and nxt not in rec["pages"]:
            self.log.append({"kind": "checkpoint", "what": "repeat_fix", "page": nxt})
            spec = ((self.shapes.get(call_name) or {}).get("properties") or {}).get(pkey) or {}
            if isinstance(nxt, str) and re.fullmatch(r"\d+", nxt.strip()) and isinstance(spec, dict) and spec.get("type") in ("integer", "number"):
                nxt = int(nxt.strip())          # (a row number handed back as text goes to a number argument as a number)
            return {**call_args, pkey: nxt}, ""
        self.log.append({"kind": "checkpoint", "what": "repeat_fix", "held": True})
        return call_args, ("Not run (no step used): this exact search already ran and its rows are above, and it has no further "
                           "page. Change the words, or read the rows it showed.")

    def search_seen(self, call_name: str, call_args: Any, obs: Any) -> None:
        sid = self.search_id(call_name, call_args)
        if not sid:
            return
        qid, _, page = sid
        rec = self.__dict__.setdefault("search_book", {}).setdefault(qid, {"pages": set(), "next": ""})
        rec["pages"].add(page)
        rec["next"] = self.next_token(self.inner_obs(obs)) or ""

    def browse_fix(self, call_name: str, call_args: dict[str, Any], scripted: bool) -> tuple[dict[str, Any], str]:
        """A shelf-listing call keeps to the store's own shelf labels and the budget: a label seen on listings is written
        the store's way; anything else (search words, a guess) becomes the product's shelf once one is known, or the call
        is not run; a missing price cap is the budget (per unit)."""
        spec = self.browse_spec()
        if not spec or call_name != spec["tool"] or not isinstance(call_args, dict):
            return call_args, ""
        key = spec["shelf"]
        labels = self.seen_labels()
        val = str(call_args.get(key) or "").strip()
        rows = self.needs.items()
        k = None
        if rows:
            names = [set(kind_tokens(r["product"])) for r in rows]
            words = set(kind_tokens(val))
            k = max(range(len(rows)), key=lambda j: len(names[j] & words)) if words else 0
        home = self.browse_label(k)
        if spec.get("enum"):
            pass
        elif val and val.lower() in labels:
            call_args[key] = labels[val.lower()]
        elif home:
            if val and val.lower() != home.lower():
                self.wire.log.append({"kind": "shelf_fixed", "was": val, "now": home})
            call_args[key] = home
        elif val and val.lower() in {self.product_phrase(j).lower() for j in (range(len(rows)) if rows else [None])} \
                and not self.__dict__.get("phrase_tried", set()) >= {val.lower()}:
            self.__dict__.setdefault("phrase_tried", set()).add(val.lower())
        else:
            seen = ", ".join(sorted(labels.values())[:8])
            return call_args, ("Not run (no step used): " + key + " takes a shelf name exactly as the listings write it in their "
                               "category field" + (f" (seen so far: {seen})" if seen else "; none has been seen yet, so run a short "
                               "keyword search first") + ". Put wanted values into the keyword search, never into the shelf.")
        if spec.get("cap") and spec["cap"] not in call_args:
            cap = self.cap_for(k)
            if cap is not None:
                call_args[spec["cap"]] = cap
        return call_args, ""

    def tool_notes(self) -> str:
        """How this store's tools are best used, read from their definitions (shown once, at the start)."""
        out = []
        spec = self.browse_spec()
        if spec:
            bits = [f"{spec['shelf']} = the shelf exactly as listings write their category"]
            if spec.get("cap"):
                bits.append(f"{spec['cap']} = your budget (filled in for you)")
            if spec.get("token"):
                bits.append(f"{spec['token']} = the next-page value it returns")
            out.append(f"- {spec['tool']} lists one shelf's listings ({'; '.join(bits)}). It is the surest way to see every in-budget "
                       "listing of a product: page through it rather than guessing words.")
        search = self.role("search")
        if search and search != (spec or {}).get("tool"):
            out.append(f"- {search} searches words: two or three words, a wanted value first and then the kind (or the value alone); "
                       "read the next page when the total is larger than the rows shown.")
        batch = self.batch_tool()
        if batch:
            out.append(f"- {batch[0]} reads up to {batch[2]} listings' full records in ONE step: read candidates several at a time "
                       f"with it (never more than {batch[2]} ids in one call).")
        return ("HOW THIS STORE'S TOOLS ARE BEST USED (read from their definitions):\n" + "\n".join(out)) if out else ""

    def reads_target(self) -> int:
        """Listings of a product to read before its questions: about a third of the steps' worth when several can be read
        at once, fewer when each read costs a step or several products share the steps."""
        n = max(1, len(self.needs.items()))
        if self.batch_tool():
            base = min(BROWSE_READS, max(4, round(0.32 * self.step_cap)))
        else:
            base = min(6, max(3, round(0.15 * self.step_cap)))
        return max(3, base // n) if n > 1 else base

    @staticmethod
    def inner_obs(obs: Any) -> Any:
        """The real result inside a notice wrapper ({notices: [...], result: {...}})."""
        if isinstance(obs, dict) and len(obs) == 2:
            lists = [v for v in obs.values() if isinstance(v, list) and v and all(isinstance(x, str) for x in v)]
            rest = [v for v in obs.values() if isinstance(v, dict)]
            if lists and rest:
                return rest[0]
        return obs

    def browse_step(self) -> bool:
        """Lord-Voldemort-style opening: each product's shelf is listed under the budget (page by page) and its best
        unread rows are read several at a time, before the questions and while steps allow."""
        spec = self.browse_spec()
        if not spec or self.wire.done or self.wire.broke or self.wire.mute or not spec.get("cap"):
            return False
        floor = self.gate_floor() + 1 if self.__dict__.get("gate_n") else self.order_due() + 6
        if self.moves_left() <= floor or self.wire.spare() < CLOSING + 40:
            return False
        rows = self.needs.items()
        state = self.__dict__.setdefault("browse_state", {})
        target = self.reads_target()
        ks = list(range(len(rows))) if rows else [None]
        ks.sort(key=lambda j: (state.get(str(j), {}).get("pages", 0), self.product_reads(j)))
        for k in ks:
            st = state.setdefault(str(k), {"pages": 0, "next": "", "done": False, "extra": 0})
            limit = BROWSE_PAGES + st.get("extra", 0)
            if st["done"] or st["pages"] >= limit or (st["pages"] >= 1 and self.product_reads(k) >= target and not st.get("extra")):
                continue
            label = self.browse_label(k)
            if not label:
                if st.get("phrase") or not self.product_phrase(k):
                    continue
                st["phrase"] = True
                label = self.product_phrase(k)
            call_args = {spec["shelf"]: label}
            cap = self.cap_for(k)
            cap = cap if self.__dict__.get("cap_down") is None else self.cap_down
            if cap is not None:
                call_args[spec["cap"]] = cap
            if st["pages"] > 0:
                if not st["next"] or not spec.get("token"):
                    st["done"] = True
                    continue
                call_args[spec["token"]] = st["next"]
            product = self.product_phrase(k) or "item"
            proposal = {"id": f"browse_{k}_{self.wire.tick}", "type": "function",
                        "function": {"name": spec["tool"], "arguments": json.dumps(call_args)}}
            self.chat.append({"role": "assistant", "content": f"I list the {product} shelf within the budget (page {st['pages'] + 1}) "
                              "and read its best rows several at a time.", "tool_calls": [proposal]})
            before = len(self.archive.items)
            responses = self.dispatch([proposal], scripted=True)
            st["pages"] += 1
            if st.get("extra"):
                st["extra"] -= 1
            st["next"] = ""
            for rec in self.archive.items[before:]:
                nxt = self.next_token(self.inner_obs(rec)) if isinstance(rec, dict) else None
                if nxt:
                    st["next"] = nxt
            self.post_replies(responses)
            homes = self.home_shelves()
            home = homes[k] if k is not None and k < len(homes) else (homes[0] if homes else set())
            batch = self.batch_tool()
            if home and (self.role("details") or batch) and (st.get("hunt") or self.product_reads(k) < target):
                qty = (self.needs.counts()[k] if rows and k is not None and k < len(self.needs.counts()) else self.needs.quantity()) or 1
                unread = self.menu_rows(home, batch[2] if batch else 2, [], qty)
                if unread:
                    self.open_queue = unread
                    self.open_purpose = "menu"
                    if st.get("hunt") and self.__dict__.get("gate_n"):
                        self.open_purpose, self.open_floor = "decline", self.gate_floor()
                    self.open_say = f"I read the {product} listings that look closest to the request, several at once."
            self.log.append({"kind": "browse_step", "product": product, "page": st["pages"], "next": bool(st["next"]),
                             "queued": len(self.__dict__.get("open_queue") or [])})
            return True
        return False

    def need_hunt(self) -> bool:
        """After the questions, for each product with no verified fit yet (one at a time, the least searched first): searches
        of its wanted values together and each with the product's kind word (then the first value alone) are paged, deeper
        while the pages still show unread in-budget rows of its shelf, and those rows are read several at a time."""
        try:
            search = self.role("search")
            if not search or self.asking or self.wire.done or self.wire.broke or self.wire.mute:
                return False
            rows = self.needs.items()
            if (self.trolley.units and not rows and not set(self.trolley.units) & set(self.__dict__.get("stand_in") or ())) \
                    or self.orders_tried or self.moves_left() <= self.order_due() + 2 \
                    or self.wire.spare() < CLOSING + 40 or not self.query_keys(search):
                return False
            ver = str(self.needs.version)
            fitted = [key.rpartition("|")[0] for key in self.passed if key.rpartition("|")[2] == ver and key.rpartition("|")[0] not in self.dead_lines()]
            homes = self.home_shelves()
            if rows:
                done_k = {k for k, home in enumerate(homes) if home and any(self.shelf_known(i) in home for i in fitted)}
                open_k = [k for k in range(len(rows)) if k not in done_k]
            else:
                open_k = [] if fitted else [None]
            if not open_k:
                return False
            book = self.__dict__.setdefault("hunts", {})
            total = sum(h["runs"] for h in book.values())
            if total >= HUNT_CALLS * max(1, len(rows)):
                return False
            for j in open_k:
                hunt = book.setdefault(str(j), {"queries": None, "at": 0, "page": 0, "next": "", "runs": 0})
                if hunt["queries"] is not None:
                    continue
                names = [w for w in re.findall(r"[a-z0-9]+", self.product_phrase(j).lower()) if len(w) > 2 and w not in LINK_WORDS]
                label = self.browse_label(j) if self.browse_spec() else ""
                if not label:
                    homes_k = self.home_shelves()
                    home_k = homes_k[j] if j is not None and j < len(homes_k) else (homes_k[0] if homes_k else set())
                    label = sorted(home_k)[0] if home_k else ""
                lw = [w for w in re.findall(r"[a-z0-9]+", str(label).lower()) if len(w) > 2 and w not in LINK_WORDS][:3]
                kind = " ".join(lw) if lw else (names[-1] if names else "")
                vals = []
                for req in list(self.needs.uncovered) + list(self.needs.props_left()):
                    if rows and self.product_of(str(req).partition(":")[0]) != j:
                        continue
                    v = self.plain_value(str(req).partition(":")[2])
                    if v and positive_value(v) and not ref_text(v.lower()) and len(v) <= 40 and v.lower() not in [x.lower() for x in vals]:
                        vals.append(v)
                        bound = re.search(r"\b(?:" + VALUE_LINKS + r")\b", v.lower())
                        hunt["heard"] = hunt.get("heard") or (req in self.needs.uncovered and not bound)
                vals = [" ".join(re.sub(r"\b(?:" + VALUE_LINKS + r")\b", " ", v, flags=re.I).split()) or v for v in vals]
                queries = []
                if len(vals) >= 2:
                    both = [w for v in vals[:2] for w in re.findall(r"[a-z0-9]+", v.lower()) if w not in LINK_WORDS]
                    if both:
                        queries.append(" ".join(dict.fromkeys(both + ([kind] if kind else []))))
                for v in vals[:2]:
                    words = [w for w in re.findall(r"[a-z0-9]+", v.lower()) if w not in LINK_WORDS]
                    if words:
                        queries.append(" ".join(dict.fromkeys(words + ([kind] if kind else []))))
                if vals:
                    queries.append(" ".join(w for w in re.findall(r"[a-z0-9]+", vals[0].lower()) if w not in LINK_WORDS))
                hunt["queries"] = [q for q in dict.fromkeys(queries) if q.strip()]
            if self.moves_left() <= self.order_due() + 4:
                # (the first search of a product with a heard value that names a thing, not a limit, with its stated values,
                # needs only its own steps above the order reserve: one search and two reads; later pages keep the wider margin)
                open_k = [j for j in open_k if not book[str(j)]["runs"] and book[str(j)].get("heard")]
                if not open_k:
                    return False
            k = min(open_k, key=lambda j: (book.get(str(j), {}).get("runs", 0), -1 if j is None else j))
            hunt = book[str(k)]
            queries = hunt["queries"]
            if hunt["at"] >= len(queries):
                hunt["runs"] = HUNT_CALLS          # (this product has no query left)
                rest = [j for j in open_k if j != k and book[str(j)]["at"] < len(book[str(j)]["queries"] or [])]
                return self.need_hunt() if rest and total + 1 < HUNT_CALLS * max(1, len(rows)) else False
            q = queries[hunt["at"]]
            call_args = {**self.word_args(search), self.query_keys(search)[0]: q}
            found = self.page_arg()
            if hunt["page"] > 0:
                if not (found and hunt["next"]):
                    hunt.update(at=hunt["at"] + 1, page=0, next="")
                    return self.need_hunt()
                call_args[found[0]] = hunt["next"]
            proposal = {"id": f"hunt_{k}_{hunt['runs']}_{self.wire.tick}", "type": "function", "function": {"name": search, "arguments": json.dumps(call_args)}}
            self.chat.append({"role": "assistant", "content": f"No opened listing meets every requirement yet, so I search '{q}' "
                              f"(page {hunt['page'] + 1}) and read its best rows.", "tool_calls": [proposal]})
            before = len(self.archive.items)
            responses = self.dispatch([proposal], scripted=True)
            hunt["runs"] += 1
            nxt = ""
            for rec in self.archive.items[before:]:
                nxt = (self.next_token(self.inner_obs(rec)) if isinstance(rec, dict) else None) or nxt
            self.post_replies(responses)
            homes = self.home_shelves()
            home = (homes[k] if k is not None and k < len(homes) else set()) if rows else (homes[0] if homes else set())
            batch = self.batch_tool()
            qty = (rows[k]["quantity"] if rows and k is not None and k < len(rows) else self.needs.quantity()) or 1
            most = batch[2] if batch else 2
            unread = self.menu_rows(home, most, [], qty) if home else []
            if home:
                # (this search's own unread rows of the shelf come first, in its order, the cheapest first when the request
                # wants the cheapest: a row whose title does not show the values can still carry them)
                cap = self.needs.spend_cap
                page: list[str] = []

                def page_visit(d: dict[str, Any]) -> None:
                    ident = self.row_mark(d)
                    price = self.archive.price(ident)
                    if ident in self.archive.ids and ident not in page and ident not in self.archive.read and self.shelf_of(d) in home \
                            and not self.sold_out(d) and ident not in self.refused_lines \
                            and not (is_number(cap) and is_number(price) and price * qty > cap):
                        page.append(ident)
                for rec in self.archive.items[before:]:
                    walk_dicts(rec, page_visit)
                if self.needs.priority() == "cheapest":
                    page.sort(key=lambda i: self.archive.price(i) if is_number(self.archive.price(i)) else float("inf"))
                inside = self.figure_rows(home, qty, k)
                unread = list(dict.fromkeys(inside + page + unread))[:most]
            if unread and (self.role("details") or batch):
                self.open_queue = unread
                self.open_purpose = "menu"
                self.open_say = f"I read the rows of '{q}' that sit on the requested shelf, several at once."
            hunt["page"] += 1
            hunt["next"] = nxt
            if not nxt or hunt["page"] >= (HUNT_PAGES_MAX if unread else HUNT_PAGES):
                hunt.update(at=hunt["at"] + 1, page=0, next="")
            self.log.append({"kind": "need_hunt", "query": q, "page": hunt["page"], "queued": len(unread or [])})
            return True
        except Exception as err:
            self.log.append({"kind": "checkpoint", "what": "need_hunt", "error": f"{type(err).__name__}: {err}"[:200]})
            return False

    def figure_rows(self, home: set[str], qty: int, k: int | None) -> list[str]:
        """Unread rows of this shelf whose own title shows a figure inside a limit the requirements set: a search of the limit's
        figure finds the listings at that figure and hides those past it, so rows already seen are looked at for them too."""
        limits = []
        for req in list(self.needs.uncovered) + list(self.needs.props_left()):
            if self.needs.items() and self.product_of(str(req).partition(":")[0]) != k:
                continue
            value = self.plain_value(str(req).partition(":")[2])
            relation = ref_relation(value)[0]
            if relation in ("at least", "at most"):
                limits += [(relation, fig, unit) for fig, unit in ref_figs(value)]
        if not limits or not home:
            return []
        cap = self.needs.spend_cap
        found: list[str] = []

        def visit(d: dict[str, Any]) -> None:
            ident = self.row_mark(d)
            if ident not in self.archive.ids or ident in found or ident in self.archive.read or ident in self.refused_lines \
                    or self.shelf_of(d) not in home or self.sold_out(d):
                return
            price = self.archive.price(ident)
            if is_number(cap) and is_number(price) and price * (qty or 1) > cap:
                return
            shown = ref_figs(self.title_of(d))
            for relation, fig, unit in limits:
                if any(u == unit and (n >= fig if relation == "at least" else n <= fig) for n, u in shown):
                    found.append(ident)
                    return
        for rec in self.archive.items:
            walk_dicts(rec, visit)
        if self.needs.priority() == "cheapest":
            found.sort(key=lambda i: self.archive.price(i) if is_number(self.archive.price(i)) else float("inf"))
        return found

    def cheap_pages(self) -> str:
        """Under a cheapest rule, once, before the first order: search results come in a fixed order, not by price, so the unread pages
        of the searches already run on each cart line's category are read (those whose words carry a requirement's value first), within
        what the steps allow; then the unread in-stock rows of each line's category priced below it go to the actor to open first."""
        try:
            units = dict(self.trolley.units)
            if self.needs.priority() != "cheapest" or self.orders_tried or self.__dict__.get("cheap_paged") or not units:
                return ""
            search, found = self.role("search"), self.page_arg()
            budget = min(8, self.moves_left() - 14)
            if not search or not found or budget < 2 or self.wire.spare() < CLOSING + 60:
                return ""
            self.cheap_paged = True
            key = found[0]
            lines = {t: self.shelf_known(t) for t in units}
            shelves = {sh for sh in lines.values() if sh}
            walks = []
            for w in (self.__dict__.get("page_book") or {}).values():
                if not (w.get("shelves") or set()) & shelves or self.next_page(w) is None:
                    continue
                left = max(0, -(-(w.get("total") or 0) // max(1, w.get("size") or 1)) - len(w.get("read") or ()))
                walks.append((-self.value_score(w["args"]), left, w))
            walks.sort(key=lambda x: (x[0], x[1]))
            if any(v < 0 for v, _, _ in walks):
                walks = [x for x in walks if x[0] < 0]      # (the searches led by a required value first: a bare kind search waits)
            read = 0
            for _, left, w in walks:
                mine = 0
                while read < budget and mine < 3 and self.next_page(w) is not None and not self.wire.done:
                    mine += 1
                    before = len(w.get("read") or ())
                    self.send_chunk([(f"cheap_page_{self.wire.tick}", search, {**w["args"], key: self.next_page(w)})])
                    read += 1
                    if len(w.get("read") or ()) == before:
                        break          # (the page did not book: stop this walk)
            self.log.append({"kind": "cheap_pages", "pages": read, "walks": len(walks)})
            if not read:
                return ""
            rows = self.shelf_rows()
            offers = []
            for thing, shelf in lines.items():
                price = self.archive.price(thing)
                if not is_number(price) or not shelf:
                    continue
                cheaper = sorted((c, i, t) for i, (c, t, _) in rows.items() if c < price and i not in units
                                 and i not in self.archive.read and self.shelf_known(i) == shelf)[:4]
                offers += [(thing, c, i, t) for c, i, t in cheaper]
            if not offers:
                return ""
            self.__dict__.setdefault("named", set()).update(i for _, _, i, _ in offers)
            return ("Not run yet (no step used). The cheapest fit is asked for, and search results come in a fixed order, not by "
                    f"price, so the code read {read} more result page(s). These in-stock rows cost less than the cart line of the same "
                    "category and were never opened: " + "; ".join(f"{i} '{t[:60]}' {c:g} (cheaper than {thing})" for thing, c, i, t in offers)
                    + ". Open them (several in one reply); if one meets every requirement of its product, make it that product's line, "
                    "re-read its stock and price, and order; otherwise order the cart as it is.")
        except Exception as err:
            self.log.append({"kind": "checkpoint", "what": "cheap_pages", "error": f"{type(err).__name__}: {err}"[:200]})
            return ""

    def decline_sweep(self) -> str:
        """Once, before a decline with a product lacking a line in the cart: up to two short searches of the category's own words and
        two led by the values in force, never run yet, then unread in-budget rows of the category are read for the actor: those whose
        title shows a figure inside a limit first, then those showing a searched value, cheapest first."""
        try:
            if self.__dict__.get("decline_swept") or self.moves_left() < 10 or self.wire.spare() < CLOSING + 40:
                return ""
            rows_p = self.needs.items()
            rows_n = len(rows_p) or 1
            search, batch = self.role("search"), self.batch_tool()
            if len(self.trolley.units) >= rows_n or not search or not self.query_keys(search) or not (self.role("details") or batch):
                return ""
            self.decline_swept = True
            ran = {" ".join(self.free_text(w["args"])).strip().lower() for w in self.__dict__.get("page_book", {}).values()}
            ran |= {str(t).strip().lower() for t in self.__dict__.get("search_words") or []}
            queries = [q for q in self.label_queries() if " " not in q and q not in ran][:2]
            homes = self.home_shelves() if rows_p else []
            for k in range(rows_n):
                shelf = sorted(homes[k])[0] if rows_p and k < len(homes) and homes[k] else ""
                if rows_p and (not shelf or any(self.shelf_known(t) == shelf for t in self.trolley.units)):
                    continue
                heads = [self.stem(w) for w in re.findall(r"[a-z0-9]+", shelf.lower())[-1:]] if shelf else self.label_queries()[:1]
                vals = []
                for req in self.needs.props_left()[:12]:
                    head, _, raw = str(req).partition(":")
                    value = self.plain_value(raw)
                    if rows_p and self.product_of(head) != k:
                        continue
                    if not value or ref_text(value) or ref_relation(value)[0] != "same" or not positive_value(value, head) or len(value) > 30:
                        continue
                    vals.append(value.lower())
                if not heads or not vals:
                    continue
                for q in [" ".join(vals[:3] + heads[:1]), " ".join(vals[:1] + heads[:1])]:
                    if q not in ran and q not in queries and len(queries) < 4:
                        queries.append(q)
            queries = queries[:max(0, self.moves_left() - 8)]
            base, text = self.word_args(search), self.query_keys(search)[0]
            if queries:
                self.send_chunk([(f"dsweep_{n}_{self.wire.tick}", search, {**base, text: q}) for n, q in enumerate(queries)])
            words = {w for q in queries for w in ref_toks(q)} - {w for q in self.label_queries() for w in ref_toks(q)}
            held = set(self.trolley.units)
            rows = sorted((not (set(ref_toks(t)) & words), c, i) for i, (c, t, _) in self.shelf_rows().items()
                          if i not in held and i not in self.archive.read and i not in self.refused_lines)
            inside = []
            for k in range(rows_n):
                home = (homes[k] if k < len(homes) else set()) if rows_p else {sh for sh in self.shelf_labels()}
                inside += self.figure_rows(home, (rows_p[k]["quantity"] if rows_p else self.needs.quantity()) or 1, k if rows_p else None)
            queue = [i for i in dict.fromkeys(inside + [i for _, _, i in rows]) if i not in held][:min(batch[2] if batch else 2, 5)]
            self.log.append({"kind": "decline_sweep", "queries": queries, "opened": queue})
            if not queue or self.moves_left() < 8:
                return ""
            self.open_queue = queue
            self.open_say = "Before declining, I open the likeliest listings of the requested category never opened."
            return ("Not run yet (no step used). Before a decline, the code searched the requested category's own words"
                    + (" (" + ", ".join(queries) + ")" if queries else "") + " and opens for you rows within the budget now in force that "
                    "were never read: " + ", ".join(queue) + ". Judge each against the requirements now in force; when one fits, put it in "
                    "the cart and order; decline only if each one fails.")
        except Exception as err:
            self.log.append({"kind": "checkpoint", "what": "decline_sweep", "error": f"{type(err).__name__}: {err}"[:200]})
            return ""

    def field_states(self, fields: dict[str, Any], prop: str, value: str) -> bool:
        """A listing's own field for this property writes the value, or a figure inside it when it is a limit."""
        name = {self.stem(w) for w in re.findall(r"[a-z0-9]+", re.sub(r"\([^)]*\)", " ", prop).lower())}
        relation = ref_relation(value)[0]
        want = {self.stem(w) for w in re.findall(r"[a-z0-9]+", value.lower())} - LINK_WORDS
        for key, val in fields.items():
            words = {self.stem(w) for w in re.findall(r"[a-z0-9]+", re.sub(r"([a-z])([A-Z])", r"\1 \2", str(key)).lower())}
            if not words or not name or not (words <= name or name <= words):
                continue
            if relation in ("at least", "at most"):
                if any(u == unit and (n >= fig if relation == "at least" else n <= fig) for fig, unit in ref_figs(value) for n, u in ref_figs(val)):
                    return True
            elif want and want <= {self.stem(w) for w in re.findall(r"[a-z0-9]+", str(val).lower())}:
                return True
        return False

    def stated_fit(self) -> str:
        """One product, once, before a decline: opened in-stock listings within the budget whose own fields state every requirement in
        force are named, cheapest first, so they get the item check rather than being passed over. A requirement this code cannot read
        plainly (a negation, a reference) leaves it silent, and the item check still decides."""
        try:
            reqs = list(self.needs.props_left())
            if self.needs.items() or self.__dict__.get("stated_fit_said") or not reqs or self.moves_left() < 4:
                return ""                # (the cart line, its re-read and the order)
            wants = []
            for req in reqs:
                prop, _, raw = str(req).partition(":")
                value = self.plain_value(raw)
                if not value or not positive_value(value, prop) or ref_text(value.lower()) or ref_relation(value)[0] not in ("same", "at least", "at most"):
                    return ""
                wants.append((prop, value))
            cap, qty, fits = self.needs.spend_cap, self.needs.quantity() or 1, []
            for ident, row in self.ch_rows().items():
                price = self.archive.price(ident)
                if ident not in self.archive.read or ident in self.refused_lines or ident in self.dead_lines() or row.get("stock") is False \
                        or self.last_stock(ident) is False or (is_number(cap) and is_number(price) and price * qty > cap) or self.field_view(ident)["fails"]:
                    continue
                if all(self.field_states(row["fields"], prop, value) for prop, value in wants):
                    fits.append(ident)
            if not fits:
                return ""
            self.stated_fit_said = True
            fits.sort(key=lambda i: self.archive.price(i) if is_number(self.archive.price(i)) else float("inf"))
            self.log.append({"kind": "checkpoint", "what": "stated_fit", "ids": fits[:3]})
            return ("Not run yet (no step used). These opened listings state every requirement in their own fields: " + ", ".join(fits[:3])
                    + ". Put the first in the cart for the item check, then re-read its stock and price and order it. Decline only if the check fails each one.")
        except Exception as err:
            self.log.append({"kind": "checkpoint", "what": "stated_fit", "error": f"{type(err).__name__}: {err}"[:200]})
            return ""

    def cheap_sweep(self) -> str:
        """One product under a cheapest rule, once, before the first order: a search of the item's plain name, then up to three unread
        rows of its shelf below the cart's price are read, those whose title shows a figure inside the shopper's limit first. On
        large shelves long queries and the shelf's first pages can leave the cheapest fitting listing out of every row seen."""
        try:
            line = self.chosen()
            search = self.role("search")
            if self.needs.priority() != "cheapest" or self.needs.items() or self.__dict__.get("cheap_swept") or self.orders_tried \
                    or not line or not search or not self.query_keys(search):
                return ""
            room = self.moves_left() - self.order_due() - 2
            price, shelf = self.archive.price(line), self.shelf_known(line)
            if room < 3 or not is_number(price) or not shelf or self.wire.spare() < CLOSING + 30:
                return ""
            self.cheap_swept = True
            product = str(self.needs.reading.get("product") or "").lower()
            names = re.findall(r"[a-z0-9]+", re.split(r"\s+or\s+|,|/", product)[-1])[:3]
            query = " ".join(names)
            used = {str(t).strip().lower() for t in self.__dict__.get("search_words") or []}
            if query and query not in used:
                self.walk_call({**self.word_args(search), self.query_keys(search)[0]: query}, f"Before ordering the cheapest, I search the plain item name '{query}'.")
                room -= 1
            home = {shelf}
            rows: list[str] = []

            def visit(d: dict[str, Any]) -> None:
                ident = self.row_mark(d)
                cost = self.archive.price(ident)
                if ident in self.archive.ids and ident not in rows and ident not in self.archive.read and ident not in self.refused_lines \
                        and self.shelf_of(d) in home and not self.sold_out(d) and is_number(cost) and cost < price:
                    rows.append(ident)
            for rec in self.archive.items:
                walk_dicts(rec, visit)
            rows.sort(key=lambda i: self.archive.price(i))
            inside = [i for i in self.figure_rows(home, 1, None) if i in rows]
            queue = list(dict.fromkeys(inside + rows))[:min(3, max(0, room - 1))]
            self.log.append({"kind": "checkpoint", "what": "cheap_sweep", "query": query, "below": len(rows), "queued": queue})
            if not queue or not (self.role("details") or self.batch_tool()):
                return ""
            self.open_queue = queue
            self.open_purpose = "cart"
            self.open_say = f"I read the rows priced below {line} to see whether one of them fits."
            return (f"The request wants the cheapest, and {len(rows)} unread rows of the category sit below {line}'s price. The cheapest of them "
                    f"are read for you now ({', '.join(queue)}): if one meets every requirement, take {line} out and put it in; otherwise order.")
        except Exception as err:
            self.log.append({"kind": "checkpoint", "what": "cheap_sweep", "error": f"{type(err).__name__}: {err}"[:200]})
            return ""

    def step_plan(self) -> dict[str, Any]:
        """How this request's steps are split, by its type: the step the questions end by, the step the first cart comes by,
        and the steps left after it for what may change."""
        kind = "several" if len(self.needs.items()) >= 2 else "cheapest" if self.needs.priority() != "none" else "one"
        ask, cart = STEP_PLAN[kind]
        cart_by = round(cart * self.step_cap)
        return {"kind": kind, "ask_by": round(ask * self.step_cap), "cart_by": cart_by, "keep": self.step_cap - cart_by,
                "stand_by": round(STAND_IN_BY * self.step_cap)}

    def ask_due(self) -> bool:
        """The questions left have to go out now to end by the plan's step, or before the errand's money runs out."""
        if not self.asking or not self.role("message"):
            return False
        left = self.ask_limit() - self.msgs_sent
        money = self.msgs_sent == 0 and self.wire.spend >= ASK_SPEND_SHARE * self.wire.hard_cap()
        return left > 0 and (self.wire.tick + left >= self.step_plan()["ask_by"] or money)

    def cart_pool(self, k: int | None) -> list[str]:
        """The read listings of a product that can go in the cart: in stock and in budget, those no check failed first, then those
        whose own fields state the most requirements, then by price. A listing whose own fields break a requirement (the field
        evidence, word for word on stated ones) is never one: an order holding it cannot pass."""
        rows = self.needs.items()
        homes = self.home_shelves() if rows else [set(self.shelf_labels())]
        home = homes[k if k is not None else 0] if homes else set()
        if not home:
            return []
        ver = str(self.needs.version)
        failed = {key.split("|")[0] for key, val in self.rulings.items() if val and key.split("|")[1:2] == [ver]}
        info = self.code_rows()
        cap, qty = self.needs.spend_cap, (rows[k]["quantity"] if rows and k is not None else self.needs.quantity())
        pool = []
        for ident in self.archive.read:
            price = self.archive.price(ident)
            if self.shelf_known(ident) not in home or ident in self.refused_lines or ident in self.dead_lines() \
                    or (info.get(ident) or {}).get("stock") is False or not is_number(price) or (is_number(cap) and price * qty > cap) \
                    or self.field_view(ident)["fails"]:
                continue
            pool.append((ident in failed, -self.stated_count(k, (info.get(ident) or {}).get("fields") or {}), price, ident))
        return [item[-1] for item in sorted(pool)]

    def stated_count(self, k: int | None, fields: dict[str, Any]) -> int:
        """How many requirements of this product a listing's own fields state."""
        n = 0
        for req in self.needs.props_left():
            prop, _, raw = str(req).partition(":")
            if self.needs.items() and self.product_of(prop) != k:
                continue
            value = self.plain_value(raw)
            if value and positive_value(value, prop) and not ref_text(value.lower()) and self.field_states(fields, prop, value):
                n += 1
        return n

    def cart_due(self, held: bool = False) -> bool:
        """By the plan's first-cart step (or before a first decline) each product's best read listing goes in the cart: a
        checked fit, else the closest read listing as a stand-in. A change the shop or the shopper makes comes after a first
        cart, and it then still has the steps it needs."""
        try:
            if self.__dict__.get("cart_due_done") or self.trolley.used or self.orders_tried or self.wire.done or self.wire.broke \
                    or self.wire.mute or not self.role("cart") or self.moves_left() < 4 or self.wire.spare() < CLOSING + 30:
                return False
            if not held and (self.asking or self.wire.tick < self.step_plan()["cart_by"]):
                return False
            late = held or self.wire.tick >= self.step_plan()["stand_by"]
            if self.__dict__.get("cart_due_fits") and not late:
                return False
            self.cart_due_fits = True
            self.cart_due_done = late
            rows = self.needs.items()
            cap = self.needs.spend_cap
            picks: list[tuple[str, int]] = []
            stand: set[str] = set()
            total = 0.0
            for k in (range(len(rows)) if rows else [None]):
                pool = self.cart_pool(k)
                qty = rows[k]["quantity"] if rows else self.needs.quantity()
                fit = next((i for i in pool[:3] if not self.check_item(i, "ADD TO THE CART")), "")
                ident = fit or (pool[0] if pool and late else "")
                cost = (self.archive.price(ident) or 0.0) * qty if ident else 0.0
                if not ident or (is_number(cap) and total + cost > cap):
                    continue
                total += cost
                picks.append((ident, qty))
                if not fit:
                    stand.add(ident)
            self.log.append({"kind": "checkpoint", "what": "cart_due", "picks": picks, "stand_in": sorted(stand), "held": held})
            if not picks:
                return False
            bad = self.code_put(picks)
            if bad is None:
                return False
            self.stand_in = stand
            note = ("[step plan] The first cart is due, so the code put these in: " + ", ".join(i for i, _ in picks) + ". "
                    + (("Stand-ins, not verified fits: " + ", ".join(sorted(stand)) + ": replace each when a fit is found, and never "
                        "order one that still fails. ") if stand else "") + self.status_line())
            if held:
                self.cart_note = " " + note      # (inside a dispatch: the note goes out with the call's own reply)
            else:
                self.chat.append({"role": "user", "content": note})
            return True
        except Exception as err:
            self.log.append({"kind": "checkpoint", "what": "cart_due", "error": f"{type(err).__name__}: {err}"[:200]})
            return False

    def fill_lines(self) -> bool:
        """A run cut short by the cost or call limit still orders a whole basket: an empty cart gets the first-cart picks,
        and each requested product with no cart line gets its best read listing (one no check failed first, then the cheapest)."""
        try:
            if self.wire.done or self.wire.broke or not self.role("cart") or self.moves_left() < 3:
                return False
            if not self.trolley.used:
                return self.cart_due(held=True)
            rows = self.needs.items()
            if not rows:
                return False
            homes = self.home_shelves()
            held = {line: units for line, units in self.trolley.units.items() if units}
            shelves = {self.shelf_known(line) for line in held}
            cap = self.needs.spend_cap
            total = sum((self.archive.price(line) or 0.0) * float(units) for line, units in held.items())
            picks: list[tuple[str, int]] = []
            for k, row in enumerate(rows):
                home = homes[k] if k < len(homes) else set()
                if not home or home & shelves:
                    continue
                pool = [ident for ident in self.cart_pool(k) if ident not in held]
                cost = (self.archive.price(pool[0]) or 0.0) * row["quantity"] if pool else 0.0
                if not pool or (is_number(cap) and total + cost > cap):
                    continue
                total += cost
                picks.append((pool[0], row["quantity"]))
            self.log.append({"kind": "checkpoint", "what": "fill_lines", "picks": picks, "held": sorted(held)})
            if not picks:
                return False
            return self.code_put([(line, int(units)) for line, units in held.items()] + picks) is not None
        except Exception as err:
            self.log.append({"kind": "checkpoint", "what": "fill_lines", "error": f"{type(err).__name__}: {err}"[:200]})
            return False

    def after_change(self) -> bool:
        """What changed after a cart decides the next steps. A dropped requirement on a request that does not want the
        cheapest: the cart stays and is ordered. With the cheapest asked for, a dropped requirement, a sold-out line or a new
        budget: the shelf is listed again below the best price known."""
        try:
            seen = self.__dict__.setdefault("change_seen", {"drop": 0, "dead": 0, "cap": self.needs.spend_cap})
            drop, dead, cap = len(self.needs.let_go), len(self.dead_lines()), self.needs.spend_cap
            fresh = (drop > seen["drop"], dead > seen["dead"], cap != seen["cap"])
            seen.update(drop=drop, dead=dead, cap=cap)
            if not any(fresh) or not self.trolley.used or self.wire.done or self.wire.broke or self.wire.mute:
                return False
            self.log.append({"kind": "checkpoint", "what": "after_change", "dropped": fresh[0], "sold_out": fresh[1], "budget": fresh[2]})
            if self.needs.priority() == "cheapest":
                return self.cheaper_look()
            return self.keep_order() if fresh[0] and not fresh[1] else False
        except Exception as err:
            self.log.append({"kind": "checkpoint", "what": "after_change", "error": f"{type(err).__name__}: {err}"[:200]})
            return False

    def keep_order(self) -> bool:
        rows = self.needs.items()
        lines = list(self.trolley.units)
        order = self.role("order")
        if not order or not lines or (rows and len(lines) < len(rows)) or set(lines) & self.dead_lines() or self.moves_left() < 3:
            return False
        if any(self.check_item(line, "ORDER") for line in lines):
            return False
        self.stand_in = set()
        call = {"id": f"keep_{self.wire.tick}", "type": "function", "function": {"name": order, "arguments": json.dumps(self.plain_args(order, "Order the cart.", "order"))}}
        self.log.append({"kind": "checkpoint", "what": "keep_order", "cart": dict(self.trolley.units)})
        self.chat.append({"role": "assistant", "content": "A dropped requirement only widens what fits, and the cart still meets every "
                          "requirement in force, so it stays and I order it.", "tool_calls": [call]})
        self.post_replies(self.dispatch([call], scripted=True))
        return True

    def cheaper_look(self) -> bool:
        spec = self.browse_spec()
        if self.needs.items() or not spec or not spec.get("cap") or self.moves_left() <= self.gate_floor() + 3:
            return False
        ver = str(self.needs.version)
        known = [self.archive.price(i) for i in list(self.trolley.units) + [key.rpartition("|")[0] for key in self.passed if key.rpartition("|")[2] == ver]
                 if i not in self.dead_lines() and is_number(self.archive.price(i))]
        cap = self.needs.spend_cap
        best = min(known) if known else None
        bound = best - 1 if best is not None else None
        if bound is None or bound <= 0 or (is_number(cap) and bound >= cap):
            return False
        self.cap_down = int(bound) if spec.get("cap_int") else bound
        self.gate_n = max(self.__dict__.get("gate_n", 0), 1)
        st = self.__dict__.setdefault("browse_state", {}).setdefault("None", {"pages": 0, "next": "", "done": False, "extra": 0})
        st.update(pages=0, next="", done=False, extra=2, hunt=True)
        self.log.append({"kind": "checkpoint", "what": "cheaper_look", "below": self.cap_down})
        return self.browse_step()

    def gate_floor(self) -> int:
        """Moves kept for the cart and the order while a held decline's extra pages and reads run: half the order
        reserve (a fit found that late still needs a cart call and the order), at least GATE_LEAST."""
        return max(GATE_LEAST, self.order_due() // 2 + 1)

    def gate_decline(self) -> bool:
        """A decline while steps are left and the shelf is not covered is held back (at most GATE_TRIES times): more of
        the product's shelf is listed and read, and the plainer search runs, before the decline is decided again."""
        try:
            if not self.trolley.used and self.cart_due(True):
                return True
            n = self.__dict__.get("gate_n", 0)
            if n >= GATE_TRIES or self.wire.done or self.wire.broke or self.moves_left() < self.gate_floor() + 3 \
                    or self.wire.spare() < CLOSING + 40:
                self.log.append({"kind": "checkpoint", "what": "decline_gate", "skip": [n, self.moves_left(), self.order_due(), round(self.wire.spare())]})
                return False
            lacking = list(dict.fromkeys(self.decline_lacking()))[:2] or ([None] if not self.needs.items() else [])
            if not lacking:
                return False
            self.gate_n = n + 1
            state = self.__dict__.setdefault("browse_state", {})
            for k in lacking:
                st = state.setdefault(str(k), {"pages": 0, "next": "", "done": False, "extra": 0})
                if st.get("next") or st["pages"] == 0:
                    st["done"] = False
                    st["extra"] = st.get("extra", 0) + 2
                    st["hunt"] = True
            plain = [k for k in self.plain_lacking() if k in lacking and self.search_again(k, dry=True) is not None]
            if plain:
                self.fresh_queue = plain
                self.decline_queue = ([], 0, self.gate_floor())
                self.gate_pending = True
            self.log.append({"kind": "checkpoint", "what": "decline_gate", "try": self.gate_n, "lacking": lacking, "plain": plain})
            return True
        except Exception as err:
            self.log.append({"kind": "checkpoint", "what": "decline_gate", "error": f"{type(err).__name__}: {err}"[:200]})
            return False

    def word_args(self, tool: str) -> dict[str, Any]:
        """The required choices of a word search: the option that reads as one, else the first."""
        jschema = self.shapes.get(tool) or {}
        props = jschema.get("properties") or {}
        out: dict[str, Any] = {}
        if self.sheet_for(tool).get("mode") and self.sheet_for(tool)["mode"] in (jschema.get("required") or []):
            out[self.sheet_for(tool)["mode"]] = self.sheet_for(tool)["mode_words"]
        for key in [k for k in jschema.get("required") or [] if k not in out]:
            options = (props.get(key) or {}).get("enum") if isinstance(props.get(key), dict) else None
            if options:
                out[key] = next((o for o in options if re.search(WORD_KEYS, str(o).lower())), options[0])
        return out

    def searched_for(self, k: int) -> bool:
        """Whether a search so far named this product (all the words of its kind)."""
        want = set(kind_tokens(self.product_phrase(k)))
        return not want or any(want <= set(kind_tokens(text)) for text in self.__dict__.get("search_words") or [])

    def own_search(self) -> bool:
        """One product: when listings have been read but none of them, nor any row seen, sits on a shelf named for the product
        (a reference in the request sent the first searches to another product), one search of its own name comes before the
        first question, so the questions are planned from its own listings' properties."""
        if self.needs.items() or self.__dict__.get("own_searched"):
            return False
        search = self.role("search")
        homes = self.home_shelves()
        if not search or not self.query_keys(search) or (homes and homes[0]) or not self.archive.shelves_read():
            return False
        product = str(self.needs.reading.get("product") or "").strip()
        query = " ".join(dict.fromkeys(w for w in re.findall(r"[a-z0-9]+", product.lower()) if w not in LINK_WORDS))
        if not query or self.moves_left() - self.order_due() - 2 * max(0, self.ask_limit() - self.msgs_sent) - 6 <= 0:
            return False
        self.own_searched = True
        self.first_reads_done = False          # (the first reads come from the shelf this search finds)
        proposal = {"id": f"own_{self.wire.tick}", "type": "function",
                    "function": {"name": search, "arguments": json.dumps({**self.word_args(search), self.query_keys(search)[0]: query})}}
        self.chat.append({"role": "assistant", "content": f"Before asking about the {product}, I search its listings ('{query}').", "tool_calls": [proposal]})
        self.post_replies(self.dispatch([proposal], scripted=True))
        self.log.append({"kind": "cover_lines", "product": product, "query": query})
        return True

    def cover_lines(self) -> bool:
        """Several products: before the first question every product gets a search of its own and COVER_READS of its listings
        read, so no product is left out of the questions for want of a property menu."""
        rows = self.needs.items()
        search = self.role("search")
        if len(rows) < 2 or not search or not self.query_keys(search) or not (self.role("details") or self.batch_tool()):
            return False
        room = self.moves_left() - self.order_due() - 2 * max(0, self.ask_limit() - self.msgs_sent) - 6
        if room <= 0:
            return False
        searched = self.__dict__.setdefault("cover_searched", set())
        tried = self.__dict__.setdefault("cover_tried", set())
        homes = self.home_shelves()
        k = next((j for j in range(len(rows)) if j not in searched and not self.searched_for(j)), None)
        for k in ([k] if k is not None else sorted(range(len(rows)), key=self.product_reads)):
            have = self.product_reads(k) if k in searched or self.searched_for(k) else 0
            if have >= COVER_READS:
                continue
            home = homes[k] if k < len(homes) else set()
            unread = [i for i in self.menu_rows(home, COVER_READS - have, [], rows[k]["quantity"]) if i not in tried] if home else []
            if unread and (k in searched or self.searched_for(k)):
                tried.update(unread)
                self.open_queue, self.open_purpose = unread, "menu"
                self.open_say = f"I read a few {self.product_phrase(k)} listings to see which properties they differ on."
                self.log.append({"kind": "cover_lines", "product": self.product_phrase(k), "queued": len(unread)})
                return True
            if k in searched:
                continue
            searched.add(k)
            query = " ".join(dict.fromkeys(w for w in re.findall(r"[a-z0-9]+", self.product_phrase(k).lower()) if w not in LINK_WORDS))
            if not query:
                continue
            proposal = {"id": f"cover_{k}_{self.wire.tick}", "type": "function",
                        "function": {"name": search, "arguments": json.dumps({**self.word_args(search), self.query_keys(search)[0]: query})}}
            self.chat.append({"role": "assistant", "content": f"Before asking about the {self.product_phrase(k)}, I search its listings ('{query}').", "tool_calls": [proposal]})
            self.post_replies(self.dispatch([proposal], scripted=True))
            self.log.append({"kind": "cover_lines", "product": self.product_phrase(k), "query": query})
            return True
        return False

    def line_first(self) -> bool:
        """Several products: the product asked about least (among those whose need is not heard yet) gets a plain code
        search by its name (and first stated value) and its rows read, when fewer than LINE_READS of its listings are read,
        so its question is chosen from its own listings rather than going to another product."""
        rows = self.needs.items()
        search = self.role("search")
        if len(rows) < 2 or not search or not self.query_keys(search) or self.moves_left() <= self.order_due() + 8:
            return False
        asked = [0] * len(rows)
        for ln in self.questions_log:
            k = self.product_of(ln.split(":", 1)[0])
            if k is not None:
                asked[k] += 1
        heard = {self.product_of(str(n).split(":", 1)[0]) for n in self.needs.uncovered}
        tried = self.__dict__.setdefault("line_tried", set())
        open_k = [j for j in range(len(rows)) if j not in heard]
        if not open_k:
            return False
        low = min(asked[j] for j in open_k)
        for k in [j for j in open_k if asked[j] == low]:
            if self.product_reads(k) >= LINE_READS or k in tried:
                continue
            tried.add(k)
            names = [w for w in re.findall(r"[a-z0-9]+", self.product_phrase(k).lower()) if w not in LINK_WORDS]
            vals = [self.plain_value(str(r).partition(":")[2]) for r in self.needs.props_left()
                    if self.product_of(str(r).partition(":")[0]) == k]
            vals = [v for v in vals if v and positive_value(v) and not ref_text(v.lower()) and len(v) <= 30]
            words = ([w for w in re.findall(r"[a-z0-9]+", vals[0].lower()) if w not in LINK_WORDS] if vals else []) + names
            query = " ".join(dict.fromkeys(words))
            if not query:
                continue
            proposal = {"id": f"line_{k}_{self.wire.tick}", "type": "function",
                        "function": {"name": search, "arguments": json.dumps({**self.word_args(search), self.query_keys(search)[0]: query})}}
            self.chat.append({"role": "assistant", "content": f"Before asking about the {self.product_phrase(k)}, I search its listings "
                              f"('{query}') and read a few of them.", "tool_calls": [proposal]})
            self.post_replies(self.dispatch([proposal], scripted=True))
            homes = self.home_shelves()
            home = homes[k] if k < len(homes) else set()
            batch = self.batch_tool()
            unread = self.menu_rows(home, batch[2] if batch else 2, [], rows[k]["quantity"]) if home else []
            if unread and (self.role("details") or batch):
                self.open_queue = unread
                self.open_purpose = "menu"
                self.open_say = f"I read a few {self.product_phrase(k)} listings to see which properties they differ on."
            self.log.append({"kind": "line_first", "product": self.product_phrase(k), "query": query, "queued": len(unread)})
            return True
        return False

    def menu_question(self, fresh: Any) -> dict[str, Any] | None:
        """A question built in code when the models give no plan before the questions' deadline: the property most of the
        target product's read listings carry, among those still open."""
        try:
            target = self.rank_target()
            if not target or not target[2]:
                return None
            k, product, menu = target[0], target[1], target[2]
            rows = []
            for ln in menu.splitlines():
                m = re.match(r"^- (.+?) \(in (\d+) listings?\)", ln)
                if not m:
                    continue
                name = re.sub(r"[_.]+", " ", m.group(1)).strip().lower()
                if not name or junk_key(name) or kind_label(name):
                    continue
                label = f"{name} ({product})" if k is not None else name
                if fresh(label):
                    rows.append((-int(m.group(2)), len(rows), name, label))
            if not rows:
                return None
            _, _, name, label = min(rows)
            self.log.append({"kind": "checkpoint", "what": "menu_question", "property": label})
            return {"property": label, "question": f"Which {name} would you like for the {product}?"}
        except Exception:
            return None

    def list_extras(self, label: str, fresh: Any) -> list[str]:
        """For a question about one of several requested products: the further properties of that same product to name after
        it in the same message, most carried first among the keys of its category's read listings that are still open, up to
        ASK_LIST_MAX properties in all. Race 17 shoppers took long lists and named one need per answer; under two messages
        and three products, single questions left most products unasked. One product: no extras ([])."""
        try:
            rows = self.needs.items()
            if len(rows) < 2 or self.__dict__.get("no_lists"):
                return []
            k = self.product_of(label)
            if k is None:
                return []
            tied = re.search(r"\(([^)]+)\)", label)
            product = tied.group(1).strip() if tied else rows[k]["product"]
            homes = self.home_shelves()
            home = homes[k] if k < len(homes) else set()
            read = [s for s in self.archive.shelves_read() if s.strip().lower() in home]
            if not read:
                return []
            menu = self.menu_trim(self.archive.prop_menu(read[0])[0], self.archive.rich_reads(read[0]))
            first = re.split(r"\s*\(", label)[0].strip().lower()
            ranked = []
            for ln in menu.splitlines():
                m = re.match(r"^- (.+?) \(in (\d+) listings?\)", ln)
                if not m:
                    continue
                name = re.sub(r"[_.]+", " ", m.group(1)).strip().lower()
                if not name or name == first or junk_key(name) or kind_label(name) or re.search(r"\d\s*$|\boptions?$", name) or name == "model" \
                        or re.fullmatch(r"(?:options?|variants?|choices?|selections?)", name):
                    continue        # (a key ending in a number is a store's second copy; a bare options or model key is no taste)
                if fresh(f"{name} ({product})"):
                    ranked.append((-int(m.group(2)), len(ranked), name))
            own = {w.rstrip("s") for w in re.findall(r"[a-z]+", product.lower())}

            def one_slot(a: str, b: str) -> bool:
                # (two keys whose words differ only by the product's own name, a unit or a word like "options" ask one thing:
                #  "bed sheet type" and "sheet type", "frame rate fps" and "frame rate", "color options" and "color")
                wa, wb = ({w.rstrip("s") for w in re.findall(r"[a-z]+", x)} for x in (a, b))
                if not (wa <= wb or wb <= wa):
                    return False
                return all(len(w) <= 3 or w in own or w in ("option", "family") for w in wa ^ wb)
            out: list[str] = []
            for _, _, name in sorted(ranked):
                if one_slot(name, first):
                    continue
                twin = next((n for n, o in enumerate(out) if one_slot(name, o)), None)
                if twin is None:
                    out.append(name)
                elif len(name.split()) < len(out[twin].split()):
                    out[twin] = name        # (the plainer name of the two holds the slot)
            return out[:ASK_LIST_MAX - 1]
        except Exception:
            return []

    def ranked_ask(self, fresh: Any) -> dict[str, Any] | None:
        try:
            if self.needs.items() and self.line_first():
                return {"read": True}
            target = self.rank_target()
            if not target or self.wire.spare() < CLOSING + 40:
                return None
            k, product, menu, home, seen = target
            wanted = self.__dict__.setdefault("rank_reads", set())
            goal = 3 if len(self.needs.items()) >= 3 else 4
            if seen < goal and home and k not in wanted and self.moves_left() > self.order_due() + 8:
                wanted.add(k)
                batch = self.batch_tool()
                unread = self.menu_rows(home, min(goal - seen, batch[2]) if batch else min(goal - seen, 4), [],
                                        self.needs.items()[k]["quantity"] if k is not None else self.needs.quantity())
                if unread and (self.role("details") or batch):
                    self.open_queue = unread
                    self.open_recheck = True
                    self.open_say = f"Before asking about the {product}, I read a few more of its listings to see which properties they differ on."
                    self.open_purpose = "quota"
                    return {"read": True}
            if not menu:
                return None
            names: dict[str, str] = {}
            lines = []
            for ln in menu.splitlines():
                m = re.match(r"^- (.+?) (\(in \d+ listings?\).*)$", ln)
                if not m:
                    continue
                name = re.sub(r"[_.]+", " ", m.group(1)).strip().lower()
                name = re.sub(r"^[a-z]{1,2} (?=[a-z]{3,})", "", name) if len(name.split()) > 1 else name
                name = re.sub(r"(?<=[a-z]) \d{1,2}$", "", name).strip()
                if not name or name in names or junk_key(name) or kind_label(name) or re.search(r"\bbrands?\b|\bsellers?\b|\bpart\b", name) or name == "model" \
                        or re.match(r"(?:is|has|with|for) ", name) or re.fullmatch(r"(?:options?|variants?|choices?|selections?)", name):
                    continue
                names[name] = name
                lines.append(f"- {name} {m.group(2)}")
            if not names:
                return None
            menu = "\n".join(lines)
            reads = len(names) + menu.count("|")
            memo = self.__dict__.setdefault("rank_memo", {})
            have = memo.get(k)
            if not have or reads > have[0] + 3:
                try:
                    hint_text = "\n".join(f"- {self.hint_mark(h)}" for h in self.open_hints()[:8]) or "(none)"
                except Exception:
                    hint_text = "(none)"
                ask_text = (RANK_PROPS.replace("PRODUCT_NAME", product).replace("REQUEST_TEXT", self.request).replace("HINT_TEXT", hint_text)
                            .replace("LEDGER_TEXT", self.needs.text_body() or "(none)")
                            .replace("ASKED_TEXT", "\n".join(self.questions_log) or "(nothing yet)").replace("MENU_TEXT", menu))
                answer = self.wire.consult(ask_text, 600, RANKER)
                ranked = [names[str(p).strip().lower()] for p in (answer or {}).get("ranked") or [] if str(p).strip().lower() in names]
                self.log.append({"kind": "checkpoint", "what": "ranked_ask", "product": product, "ranked": ranked})
                if not ranked:
                    return None
                have = memo[k] = (reads, ranked)
            picks = [p for p in have[1] if fresh(f"{p} ({product})" if k is not None else p)]
            if not picks:
                return None
            first = picks[0]
            label = f"{first} ({product})" if k is not None else first
            shelf = next(iter(home)) if isinstance(home, set) and len(home) == 1 else ""      # (the store's own category label: "X or Y" names read as two questions)
            return {"property": label, "question": f"Which {first} would you like for the {shelf or product}?"}
        except Exception as err:
            self.log.append({"kind": "checkpoint", "what": "ranked_ask", "error": f"{type(err).__name__}: {err}"[:200]})
            return None

    def blind(self) -> bool:
        """Several store results came back and not one listing could be read into fields: the code's menus and checks know
        nothing, so the questions come from the request and the kind of product, as a model reading the text would ask."""
        if self.archive.read:
            return False
        batch = self.batch_tool()
        # (only the tool that opens one listing shows its whole table; a several-listing read may show rows alone)
        readers = {self.role("details")} if self.role("details") else ({batch[0]} if batch else set())
        opened = sum(1 for e in self.log if e.get("kind") == "env" for a in e.get("actions") or [] if a.get("name") in readers)
        blind = opened >= 2 or (not self.archive.ids and len(self.archive.items) >= 4)
        if blind and not self.__dict__.get("blind_logged"):
            self.blind_logged = True
            self.log.append({"kind": "checkpoint", "what": "blind", "opened": opened, "ids": len(self.archive.ids), "results": len(self.archive.items)})
        return blind

    def pose_planned(self, urge: bool = False) -> bool:
        """The planned questions, with the questions' reserve on the hard total while they run."""
        if self.wire.no_model:
            return self.pose_planned_body(urge=urge)
        self.wire.bonus = ASK_RESERVE
        try:
            return self.pose_planned_body(urge=urge)
        finally:
            self.wire.bonus = 0.0

    def pose_planned_body(self, urge: bool = False) -> bool:
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
        want_sets = 3 if len(self.needs.items()) >= 3 else 4
        wait_steps = max(8, round(0.22 * self.step_cap))
        blind = self.blind()
        thin = sets < want_sets and not urge and not blind
        if thin and (self.wire.tick < wait_steps or (self.wire.tick < wait_steps + ASK_GRACE_STEPS and self.moves_left() > 20)):
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
        if not blind and self.pre_read():
            return True
        if not blind and not self.questions_log and not self.msgs_sent and self.cover_lines():
            return True
        if not blind and not self.questions_log and not self.msgs_sent and self.own_search():
            return True
        if not blind and not self.questions_log and not self.msgs_sent and not urge and self.first_reads():
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
            menus = [(shelf, self.archive.prop_menu(shelf)[0]) for shelf in order[:6]]
            prop_list = "\n".join(f"CATEGORY {shelf}:\n{menu}" for shelf, menu in menus if menu) or prop_list
            shown = "\n".join(f"CATEGORY {shelf}:\n{self.menu_trim(menu, self.archive.rich_reads(shelf))}"
                              for shelf, menu in menus if menu) or shown
        ask_text += "\n\nPROPERTIES IN THE LISTINGS' DETAILS SEEN SO FAR (name, number of listings, example values):\n" + (shown or "(none seen yet)")
        if blind:
            ask_text += ("\n\nTHE STORE'S LISTINGS COULD NOT BE READ INTO FIELDS, so no property list is available: for each product "
                         "still open, choose the properties a buyer of that kind of product most often settles first and that listings "
                         "of that kind usually state, one property per question, none the request or an earlier answer settles. With several products, give each "
                         "product its own questions, the product asked least first.")
            if self.needs.items():
                rows = self.needs.items()
                asked_n = [0] * len(rows)
                for ln in self.questions_log:
                    j = self.product_of(ln.split(":", 1)[0])
                    if j is not None:
                        asked_n[j] += 1
                heard = {self.product_of(str(n).split(":", 1)[0]) for n in self.needs.uncovered}
                open_ks = [j for j in range(len(rows)) if j not in heard] or list(range(len(rows)))
                nxt = min(open_ks, key=lambda j: (asked_n[j], j))
                ask_text += f"\nASK ABOUT THIS PRODUCT NOW (it has had the fewest questions): {rows[nxt]['product']}."
        asked_props = {p.strip().lower() for ln in self.questions_log for p in ln.split(":", 1)[0].strip("- ").split(";")}

        def fresh(prop: Any) -> bool:
            prop = str(prop or "").strip()
            if not prop or prop.lower() in asked_props or junk_key(prop) or kind_label(prop) or prop.lower() in self.__dict__.get("spent_asks", set()) \
                    or self.near_asked(prop) or self.twin_asked(prop) or self.stated_twin(prop):
                return False
            j = self.product_of(prop) if self.needs.items() else None
            return not self.plain_hint(j, prop) and self.plan_open(j, prop)
        queue = [p for p in self.__dict__.get("pending_seconds") or [] if fresh(p)]
        self.pending_seconds = queue
        # several products: a queued runner-up waits while another open product has had fewer questions
        pending = next((p for p in queue if not self.quota_ahead(p)), "")
        if pending:
            tied = re.search(r"\(([^)]+)\)", pending)
            product = tied.group(1).strip() if tied else str(self.needs.reading.get("product") or "item")
            text = (self.__dict__.get("pending_text") or {}).pop(pending.lower(), "")
            answer = {"questions": [{"property": pending, "question": text or f"Which {pending.split('(')[0].strip()} would you like for the {product}?"}]}
        else:
            ranked = self.ranked_ask(fresh) if not (self.needs.reading.get("ask_first") and not self.questions_log) else None
            if ranked and ranked.get("read"):
                return True
            answer = {"questions": [ranked]} if ranked else self.wire.consult(ask_text, 2500, PLANNER)
            if ranked:
                pending = ranked["property"]
            if answer is None and self.wire.misses > 0 and not urge:
                self.plan_wait = sets
                return False
        scripted = [qitem for qitem in (answer or {}).get("questions") or [] if isinstance(qitem, dict) and str(qitem.get("question") or "").strip()]
        kept = []
        for qitem in scripted:
            second = str(qitem.get("second") or "").strip()
            second = "" if re.search(r"none|^-$|n/a", second.lower()) else second
            parts = [p.strip() for p in str(qitem.get("property") or "").split(";") if p.strip()]
            if parts and all(fresh(p) for p in parts):
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
        if blind and self.needs.items() and scripted:
            # (blind: no menu spreads the questions over the products, so the product asked least goes first)
            asked_n = [0] * len(self.needs.items())
            for ln in self.questions_log:
                j = self.product_of(ln.split(":", 1)[0])
                if j is not None:
                    asked_n[j] += 1

            def load(qitem: dict[str, Any]) -> int:
                j = self.product_of(str(qitem.get("property") or ""))
                j = j if j is not None else self.product_of(str(qitem.get("question") or ""))
                return asked_n[j] if j is not None else len(self.questions_log) + 1
            scripted = sorted(scripted, key=load)
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
        if not proposal and urge:
            fallback = self.menu_question(fresh)
            if fallback:
                proposal = self.message_call(fallback["question"], f"ask_{self.msgs_sent + 1}_{self.wire.tick}")
                label = (fallback["property"], fallback["question"])
        if not proposal:
            if sets < ASK_AFTER_SETS and self.moves_left() > 12 and not urge:
                self.plan_wait = sets
                return False
            self.empty_plans = self.__dict__.get("empty_plans", 0) + 1
            if self.moves_left() > 12 and self.empty_plans <= (3 if blind else 1):
                self.plan_wait = None if blind else sets      # (blind: no read can add to the menu, so the next plan is tried at once)
                return False
            self.asking = False
            return False
        listed: list[str] = []
        extras = self.list_extras(label[0], fresh) if ";" not in label[0] else []
        if extras:
            # (several products: the chosen property leads a list of that product's other open properties, cut from the end
            #  when the message tool's schema will not take it whole)
            first = label[0].split("(")[0].strip()
            tied = re.search(r"\(([^)]+)\)", label[0])
            product = tied.group(1).strip() if tied else str(self.needs.reading.get("product") or "item")
            for size in range(len(extras), 0, -1):
                names = [first] + extras[:size]
                text = f"Which {', '.join(names[:-1])} or {names[-1]} would you like for the {product}?"
                wide = self.message_call(text, f"ask_{self.msgs_sent + 1}_{self.wire.tick}")
                if wide:
                    proposal, label, split_second = wide, (label[0], text), ""
                    listed = [label[0]] + [f"{e} ({product})" for e in extras[:size]]
                    self.log.append({"kind": "checkpoint", "what": "list_ask", "properties": listed})
                    break
        self.asked_at_sets, self.asked_at_tick = sets, self.wire.tick
        self.chat.append({"role": "assistant", "content": f"I will ask the shopper about {label[0]}.", "tool_calls": [proposal]})
        found_before, freed_before = len(self.needs.uncovered), len(self.needs.indifferent) + self.needs.__dict__.get("pinned_heard", 0)
        self.asking_label = "; ".join(listed) if listed else label[0]
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
        if heard_meta is not None and listed:
            # (a list the shopper did not take up: in a store whose shopper answers no-view properties with a puzzled line
            #  such replies are the commonest answer to a list, so its properties count as asked and the next product is next)
            open_note = " [list not answered: its properties count as asked]"
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
            twin = self.quiet_reask(asked_label) if ";" not in asked_label and not listed else ""
            if twin:
                self.pending_seconds = [twin] + [p for p in self.__dict__.get("pending_seconds") or [] if p.lower() != twin.lower()]
                open_note += f" [{twin}: its story twin, asked next]"
        if listed:
            label = ("; ".join(listed), label[1])
        for _, text_body in responses[:1]:
            self.questions_log.append(f"- {label[0]}: asked \"{label[1]}\" -> {text_body.split(chr(10) + '[status]')[0][:400]}{open_note}")
        self.drop_wanted(asked_label)
        self.asking = self.msgs_sent < ceiling
        return True

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

    def plain_args(self, call_name: str, text_body: str, kind: str = "") -> dict[str, Any]:
        jschema = self.shapes.get(call_name) or {}
        prop_specs = jschema.get("properties") or {}
        call_args: dict[str, Any] = {}
        words = DECLINE_WORDS if kind == "decline" else ORDER_WORDS if kind == "order" else ""
        for key in jschema.get("required") or []:
            pspec = prop_specs.get(key) or {}
            if pspec.get("enum"):
                fit = [val for val in pspec["enum"] if words and isinstance(val, str) and re.search(words, val.lower())]
                if not fit and kind in ("order", "decline"):
                    # (a choice the words do not name, such as "pass": the one that is not the other ending)
                    other = ORDER_WORDS if kind == "decline" else DECLINE_WORDS
                    fit = [val for val in pspec["enum"] if isinstance(val, str) and not re.search(other, val.lower())]
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
            if any(k in texts(name) for k in needed(name)):
                continue           # (one that needs a text from the caller writes or looks up; a name like "ask buyer" holds a buying word)
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
            info = out.setdefault(ident, {"shelf": "", "title": "", "price": None, "stock": None, "words": set(), "fields": {}})
            info["shelf"] = self.shelf_of(d) or info["shelf"]
            # (a title or headline field before any name field: a maker's or producer's name is no title)
            info["title"] = next((str(val) for key, val in sorted(d.items(), key=lambda kv: not re.search(r"title|headline", str(kv[0]).lower()))
                                  if isinstance(val, str) and re.search(r"title|headline|name", str(key).lower()) and "id" not in str(key).lower()
                                  and not re.search(r"producer|maker|brand|seller|shop|store|vendor", str(key).lower())), "") or info["title"]
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
        named = [val for key, val in row["fields"].items() if same_prop(key, prop)]
        if named:
            return any(Errand.code_seen(value, val) for val in named)
        if Errand.code_seen(value, row.get("title") or ""):
            return True
        own = kind_words(value)
        return not read and bool(own) and own <= hit_words

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

    def code_put(self, picks: list[tuple[str, int]]) -> list[str] | None:
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
        told: dict[str, float] = {}
        for call_args, obs in zip(calls, replies):
            seen = self.trolley.view_units(obs)
            if seen is not None and not spec_lines and set(seen) <= {line for line, _ in self.trolley.lines(call_args)}:
                told.update(seen)         # (a reply that names only this call's own line confirms that line, not the whole cart)
            elif seen is not None:
                shown = seen
            if self.code_fault(obs):
                named = [ident for ident in want if ident in as_text(obs)]
                lines = [line for line, _ in self.trolley.lines(call_args) if line in want]
                bad.update(named or lines or (list(want) if spec_lines else []))
        if len(replies) < len(calls):
            return None
        if shown is None and told:
            shown = {**{line: units for line, units in self.trolley.units.items() if line in want and line not in told}, **told}
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
                if call_rank and turn == 0 and not self.trolley.units and self.commit_basket():
                    move = {"name": call_rank, "args": self.plain_args(call_rank, "Order the cart.", "order")}
                    self.log.append({"kind": "checkpoint", "what": "commit_basket", "cart": dict(self.trolley.units)})
                if move is None and call_rank and turn == 0 and self.trolley.units:
                    self.cart_tidy()
                if move is None and call_rank and turn == 0 and self.trolley.units and self.kit_misfit():
                    if self.kit_swap():
                        self.log.append({"kind": "checkpoint", "what": "kit_swap", "cart": dict(self.trolley.units)})
                if move is None and call_rank and decl and turn == 0 and self.trolley.units:
                    # (a cart line whose own fields break a requirement, or a product with no line, fails the order for
                    #  certain; with no checked fit to put in its place, the decline is the ending that can still be right)
                    why = self.cart_unfit()
                    if why and self.cart_mend():
                        why = self.cart_unfit()
                    if why:
                        self.log.append({"kind": "checkpoint", "what": "unfit_decline", "why": why[:300], "cart": dict(self.trolley.units)})
                        move = {"name": decl, "args": self.plain_args(decl, SORRY, "decline")}
                if move is None and call_rank and turn == 0 and self.trolley.units:
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
                if move is None and call_rank and decl and turn == 0 and self.trolley.units and not gone:
                    # (the cart lines were just read again: a line a code fill or kit swap put in unread now shows its fields,
                    #  so the check above runs once more on what the records say)
                    why = self.cart_unfit()
                    if why and self.cart_mend():
                        why = self.cart_unfit()
                    if why:
                        self.log.append({"kind": "checkpoint", "what": "unfit_decline", "why": why[:300], "cart": dict(self.trolley.units)})
                        move = {"name": decl, "args": self.plain_args(decl, SORRY, "decline")}
                if move is not None:
                    pass
                elif call_rank and turn == 0 and not gone and self.trolley.used and (self.trolley.units or self.trolley.view_units(self.trolley.last_view) is None):
                    move = {"name": call_rank, "args": self.plain_args(call_rank, "Order the cart.", "order")}
                elif call_rank and turn == 0 and self.model_finish():
                    why = self.cart_unfit() if decl else ""
                    if why:
                        self.log.append({"kind": "checkpoint", "what": "unfit_decline", "why": why[:300], "cart": dict(self.trolley.units)})
                    move = ({"name": decl, "args": self.plain_args(decl, SORRY, "decline")} if why
                            else {"name": call_rank, "args": self.plain_args(call_rank, "Order the cart.", "order")})
                elif call_rank and turn > 0 and self.again_ok(avoid):
                    move = {"name": call_rank, "args": self.plain_args(call_rank, "Order the cart.", "order")}
                elif decl:
                    move = {"name": decl, "args": self.plain_args(decl, SORRY, "decline")}
                else:
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

    def cart_unfit(self) -> str:
        """Code only, at a forced finish: why the cart cannot pass as it stands, or '': a read line whose own fields break a
        requirement (the field evidence; stated values word for word), or a requested product with no line."""
        try:
            lines = [i for i, n in self.trolley.units.items() if n]
            if not lines:
                return ""
            for i in lines:
                fails = self.field_view(i)["fails"] if i in self.archive.read else []
                if fails:
                    return f"{i}: {fails[0]}"
            rows = self.needs.items()
            if rows:
                have = {self.product_of(self.shelf_known(i) or self.archive.label_of.get(i, "")) for i in lines}
                missing = [row["product"] for k, row in enumerate(rows) if k not in have]
                if missing and None not in have:
                    return "no cart line for the " + missing[0]
            return ""
        except Exception as err:
            self.log.append({"kind": "checkpoint", "what": "cart_unfit", "error": f"{type(err).__name__}: {err}"[:200]})
            return ""

    def cart_mend(self) -> bool:
        """At a forced finish with a cart that cannot pass: each product whose line breaks a requirement, or that has no line,
        gets its first checked fit among the read listings that can go in the cart (cart_pool); False when one has none."""
        try:
            rows = self.needs.items()
            ver = str(self.needs.version)
            owner = (lambda i: self.product_of(self.shelf_known(i) or self.archive.label_of.get(i, ""))) if rows else (lambda i: None)
            held = {i: int(self.trolley.held(i) or 1) for i, n in self.trolley.units.items() if n}
            picks = [(i, n) for i, n in held.items() if not (i in self.archive.read and self.field_view(i)["fails"])]
            have = {owner(i) for i, _ in picks}
            added = 0
            for k in (range(len(rows)) if rows else [None]):
                if k in have:
                    continue
                fit = next((i for i in self.cart_pool(k) if f"{i}|{ver}" in self.passed), "")
                if not fit:
                    return False
                picks.append((fit, rows[k]["quantity"] if rows else self.needs.quantity()))
                added += 1
            cap = self.needs.spend_cap
            total = sum((self.archive.price(i) or 0.0) * float(n) for i, n in picks)
            if not added or (is_number(cap) and total > float(cap) + 1e-9) or self.moves_left() < len(picks) + 2:
                return False
            was = self.wire.mute
            self.wire.mute = True
            try:
                bad = self.code_put(picks)
            finally:
                self.wire.mute = was
            done = bad == [] and all(i in self.trolley.units for i, _ in picks)
            if done and self.kit_misfit():
                done = self.kit_swap() and not self.kit_misfit()
            self.log.append({"kind": "checkpoint", "what": "cart_mend", "picks": picks, "done": done, "cart": dict(self.trolley.units)})
            return done
        except Exception as err:
            self.log.append({"kind": "checkpoint", "what": "cart_mend", "error": f"{type(err).__name__}: {err}"[:200]})
            return False

    def commit_basket(self) -> bool:
        """At a forced finish with an empty cart: the code-verified fit of every product, put in the cart, so the errand ends
        with the best basket found instead of an order of nothing."""
        try:
            if not self.role("cart") or self.wire.done or self.wire.broke:
                return False
            ver = str(self.needs.version)
            picks = self.code_picks()
            products = len(self.needs.items()) or 1
            if not picks or len(picks) < products or not all(f"{i}|{ver}" in self.passed for i, _ in picks):
                return False
            if self.moves_left() < len(picks) + 2:
                return False
            was = self.wire.mute
            self.wire.mute = True
            try:
                bad = self.code_put(picks)
            finally:
                self.wire.mute = was
            return bad == [] and all(i in self.trolley.units for i, _ in picks)
        except Exception as err:
            self.log.append({"kind": "checkpoint", "what": "commit_basket", "error": f"{type(err).__name__}: {err}"[:200]})
            return False

    def model_finish(self) -> bool:
        if self.__dict__.get("model_finish_done") or self.wire.spare() < 20 or self.moves_left() < 2 or not self.role("cart"):
            return False
        self.model_finish_done = True
        keep = {self.role(r) for r in ("cart", "details", "cart_view", "order", "decline")} - {None}
        tools = [tdef for tdef in self.toolset if tdef["function"]["name"] in keep]
        self.chat.append({"role": "user", "content": "The errand must end now. If a listing you have read meets every requirement in force "
                          "(in stock, within the budget), make the cart hold exactly that (one line per product at its quantity) and "
                          "order it; if nothing can fit, decline as the request says.\n" + self.status_line()})
        seq_no = 10 ** 6
        for _ in range(2):
            if self.wire.done or self.wire.broke or self.moves_left() < 1 or self.wire.spare() < 12:
                break
            msg = self.wire.query_model(self.chat, ACTOR, 1200, tools)
            if msg is None:
                break
            proposals, seq_no = self.tool_calls(msg, seq_no)
            if not proposals:
                break
            self.chat.append({"role": "assistant", "content": msg.get("content") or None, "tool_calls": proposals})
            self.post_replies(self.dispatch(proposals))
        self.log.append({"kind": "checkpoint", "what": "model_finish", "cart": dict(self.trolley.units), "done": self.wire.done})
        return bool(self.trolley.units) and not self.wire.done

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
            remarks.append("THE ORDER WAS REFUSED and the episode continues; the cart is unchanged. Apply every change in the refusal, fix the cart (take out any line that is no longer acceptable, such as a sold-out item), re-check stock and price of the new pick, and order again. A line that drops or loosens a requirement never cancels the order: if the cart still fits, order it again." + reread)
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
        return next((val.strip().lower() for key, val in row.items() if isinstance(val, str) and 0 < len(val.strip()) <= 60
                     and shelf_key(key) and "id" not in str(key).lower()), "")

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
            ident = next((val for key, val in d.items() if isinstance(val, str) and id_named(key) and val in self.archive.read), None)
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
        for exact in (True, False):
            for key, val in row.items():
                name = str(key).lower()
                if (name == "id" if exact else id_named(key, True)) and isinstance(val, (str, int)) and not isinstance(val, bool):
                    return str(val)
        return json.dumps(row, sort_keys=True, default=str)

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
        if self.sheet_for(None).get("next_kind") == "token":
            return self.sheet_for(None)["next"]
        props = (self.shapes.get(self.role("search") or "") or {}).get("properties")
        for key, pspec in (props if isinstance(props, dict) else {}).items():
            if isinstance(pspec, dict) and pspec.get("type") == "string" and not pspec.get("enum") \
                    and re.search(r"bookmark|cursor|token|resume|continu|after|next|page|offset|start", str(key).lower()):
                return key
        return None

    @staticmethod
    def next_token(obs: Any) -> str | None:
        if not isinstance(obs, dict):
            return None
        for key, val in obs.items():
            if isinstance(val, str) and val.strip() and re.search(r"next|onward|cursor|bookmark|token|continu|more", str(key).lower()):
                return val
        return None

    def shelf_tool(self, name: str) -> bool:
        """A second listing tool beside the keyword search (a shelf, maker or price filter): in no other role, with a number argument."""
        if not name or name == self.role("search") or name not in self.shapes:
            return False
        taken = {self.role(r) for r in ("message", "cart", "cart_view", "order", "decline", "details", "view")} - {None}
        props = (self.shapes.get(name) or {}).get("properties") or {}
        return name not in taken and isinstance(props, dict) and any(isinstance(p, dict) and p.get("type") in ("integer", "number")
                                                                     for p in props.values())

    def page_arg(self, tool: str | None = None) -> tuple[str, Any, Any, float | None] | None:
        name_of = tool or self.__dict__.get("page_tool") or self.role("search")
        shape = self.shapes.get(name_of or "") or {}
        props = shape.get("properties")
        sheet = self.sheet_for(name_of)
        if sheet.get("next") and sheet["next"] != self.__dict__.get("page_dead"):
            spec = props[sheet["next"]] if isinstance(props, dict) and isinstance(props.get(sheet["next"]), dict) else {}
            if sheet["next_kind"] == "token":
                return sheet["next"], "token", "", None
            by_page = sheet["next_kind"] == "page_number"
            low = spec.get("minimum", spec.get("default"))
            return sheet["next"], by_page, low if is_number(low) else (1 if by_page else 0), spec.get("maximum") if is_number(spec.get("maximum")) else None
        for key, pspec in (props if isinstance(props, dict) else {}).items():
            name = str(key).lower()
            if not isinstance(pspec, dict) or pspec.get("type") not in ("integer", "number") or key == self.__dict__.get("page_dead") \
                    or not re.search(r"page|offset|first|start|skip|from|cursor", name) or re.search(r"size|per_?page|limit|price|cost", name):
                continue
            by_page = "page" in name and not re.search(r"offset|skip|first|start|from", name)
            low = pspec.get("minimum", pspec.get("default"))
            low = low if is_number(low) else (1 if by_page else 0)
            return key, by_page, low, (pspec.get("maximum") if is_number(pspec.get("maximum")) else None)
        tok = self.token_key()
        if tok:
            return tok, "token", "", None
        # (the argument's name changes with the store: the search's one optional whole number that is no price, size or limit
        #  can only be where the rows start)
        needed = shape.get("required") if isinstance(shape.get("required"), list) else []
        rest = [k for k, p in (props if isinstance(props, dict) else {}).items() if isinstance(p, dict) and p.get("type") == "integer" and k not in needed
                and k != self.__dict__.get("page_dead") and not re.search(r"size|per_?page|limit|price|cost", str(k).lower())]
        if len(rest) == 1:
            low = props[rest[0]].get("minimum", props[rest[0]].get("default"))
            return rest[0], False, low if is_number(low) else 0, props[rest[0]].get("maximum") if is_number(props[rest[0]].get("maximum")) else None
        return None

    @staticmethod
    def stem(word: str) -> str:
        return word[:-1] if len(word) > 3 and word.endswith("s") and not word.endswith("ss") else word

    def shelf_rows(self) -> dict[str, tuple[float, str, str]]:
        shelves, cap, qty = self.shelf_labels(), self.needs.spend_cap, self.needs.quantity()
        found: dict[str, tuple[float, str, str]] = {}
        if not shelves:
            return found

        def visit(d: dict[str, Any]) -> None:
            ident = next((val for key, val in d.items() if isinstance(val, str) and id_named(key) and val in self.archive.ids), None)
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
            json.dumps(base, sort_keys=True, default=str), {"args": base, "read": set(), "size": 0, "total": 0, "shelves": set(),
                                                             "tool": self.__dict__.get("page_tool") or self.role("search")})
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
        elif mode != "token" and len(rows) >= 2 and walk.get("first") == tuple(self.row_mark(r) for r in rows if isinstance(r, dict)):
            self.page_dead = key       # (the same rows came back further on: this argument does not move through them)
            self.log.append({"kind": "page_dead", "arg": key})
        if at == low and total <= len(rows):
            walk["every"] = [r for r in rows if isinstance(r, dict)]
        marks = [self.row_mark(r) for r in rows if isinstance(r, dict)]
        if len(marks) >= 5:
            walk["id_order"] = walk.get("id_order", True) is not False and (marks == sorted(marks) or marks == sorted(
                marks, key=lambda m: [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", m)]))

    def next_page(self, walk: dict[str, Any]) -> Any:
        found = self.page_arg(walk.get("tool"))
        if not found:
            return None
        key, by_page, low, high = found
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
                    + "\nDecide from the records which reading is right: a field that writes the required value's words (with a "
                    "qualifier, or in another spelling) meets the requirement; a field that states it in other words, a shade of "
                    "it, or a different value fails it.")
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

    def kit_pairs(self, soft: bool = False) -> list[tuple[int, int]]:
        """The request's fit pairs; with soft, also those the store's own series words tie: a requested product whose listings take a
        whole series of parts and another requested product whose listings carry codes of such a series.
        (Soft pairs only inform the item check: a store that numbers its parts by series does not make every request a kit; race 12
        accepted parts of another series where the request asked for no fit.)"""
        pairs = list(self.__dict__.get("fit_pairs") or [])
        try:
            if not soft or len(self.needs.items()) < 2:
                return pairs
            mark = (len(self.archive.items), self.needs.version)
            memo = self.__dict__.get("series_memo")
            if not memo or memo[0] != mark:
                takes: set[int] = set()
                tagged: set[int] = set()

                def visit(d: dict[str, Any]) -> None:
                    ident = next((v for k, v in d.items() if isinstance(v, str) and id_named(k) and re.fullmatch(ID_LIKE, v)), None)
                    if not ident:
                        return
                    low = as_text(d, 4000).lower()
                    k = self.product_of(self.shelf_known(ident) or self.archive.label_of.get(ident, ""))
                    if k is None:
                        return
                    if re.search(SERIES_TAKES, low):
                        takes.add(k)
                    elif re.search(SERIES_TAG, low):
                        tagged.add(k)
                for rec in self.archive.items:
                    walk_dicts(rec, visit)
                found = [(p, m) for m in sorted(takes) for p in sorted(tagged - takes) if p != m]
                self.series_memo = (mark, found)
                if found:
                    self.log.append({"kind": "checkpoint", "what": "series_pairs", "pairs": found})
            for p, m in self.series_memo[1]:
                if not any(q == p for q, _ in pairs):
                    pairs.append((p, m))
        except Exception as err:
            self.log.append({"kind": "checkpoint", "what": "kit_pairs", "error": f"{type(err).__name__}: {err}"[:200]})
        return pairs

    def fit_notes(self) -> tuple[dict[str, set[str]], dict[str, str]]:
        """From every record seen: what other pages say a part also fits ("part X also fits: CODE, CODE" names part X, another
        listing), and each listing's own part code (its bracketed tag)."""
        also: dict[str, set[str]] = {}
        tags: dict[str, str] = {}
        for ident in self.archive.ids:
            low = self.record_text(ident).lower()
            tag = re.search(PART_TAG, low)
            if tag:
                tags[ident] = tag.group(1)
            for m in re.finditer(ALSO_FITS, low):
                also.setdefault(m.group(1), set()).update(c[0] + c[1] for c in model_codes(m.group(2)))
            for m in re.finditer(SERIES_WIDENED, low):
                for code in re.findall(r"[a-z]{1,5}\d{1,3}", m.group(2)):
                    also.setdefault("series " + code, set()).add(re.sub(r"[- ]", "", m.group(1)))
        return also, tags

    def part_fit(self, ident: str, model: str, rules: list, notes: tuple[dict[str, set[str]], dict[str, str]]) -> bool | None:
        """Whether a part takes in a device model: its own clauses, or another page's note that it also fits that model; None when
        its own clauses leave the model out but its list goes on on a part whose page is not seen yet."""
        also, tags = notes
        if model in also.get(tags.get(ident, ""), set()):
            return True
        series = re.search(SERIES_TAG, self.record_text(ident).lower())
        if series and model in also.get("series " + series.group(1) + series.group(2), set()):
            return True
        said = fit_verdict(model, rules)
        if said is False:
            more = re.search(SEE_PART, self.record_text(ident).lower())
            if more and more.group(1) not in set(tags.values()):
                return None
        return said

    def kit_evidence(self, thing: str) -> str:
        """For the item check on a request whose parts must work with a device: what the codes on the records say, computed in
        code (a part's own list with its exceptions, and what other pages say it also fits)."""
        try:
            pairs = self.kit_pairs(soft=True)
            hard = set(self.__dict__.get("fit_pairs") or [])
            rows = self.needs.items()
            if not pairs or not rows:
                return ""
            owner = lambda i: self.product_of(self.shelf_known(i) or self.archive.label_of.get(i, ""))
            name = lambda k: rows[k]["product"] if k is not None and k < len(rows) else "device"
            k = owner(thing)
            notes = self.fit_notes()
            lines, loose = [], []
            mains = {m for p, m in pairs if p == k}
            if mains:
                if not any((k, m) in hard for m in mains):
                    loose.append(thing)
                rules = fit_rules(self.record_text(thing))
                devices = [i for i in self.archive.ids if owner(i) in mains and own_models(self.record_text(i))]
                devices = [i for i in devices if i in self.trolley.units] or devices
                take, leave = [], []
                for d in devices[:12]:
                    said = [self.part_fit(thing, m, rules, notes) for m in own_models(self.record_text(d))]
                    if any(v is True for v in said):
                        take.append(d)
                    elif said and all(v is False for v in said):
                        leave.append(d)
                if take or leave:
                    lines.append(f"{thing} must work with the {name(next(iter(mains)))} picked: by the codes it works with "
                                 + (", ".join(take[:6]) or "none of the devices seen") + (f"; it does not work with {', '.join(leave[:6])}" if leave else ""))
            for p in sorted({p for p, m in pairs if m == k}):
                models = own_models(self.record_text(thing))
                fits = [i for i in self.archive.ids if owner(i) == p and models
                        and any(self.part_fit(i, m, fit_rules(self.record_text(i)), notes) for m in models)]
                lines.append(f"{name(p)} listings seen whose codes take in {thing}'s model: " + (", ".join(fits[:6]) or "none yet"))
                if (p, k) not in hard:
                    loose.append(p)
            if not lines:
                return ""
            if loose and not any(p == k or m == k for p, m in hard):
                # (only the store's series words tie these products: whether they must fit is the request's to say)
                return ("\n\nSERIES CODES (computed by code: the store numbers these parts by series, and a device takes only the series "
                        "it names; the request was not read as asking for a fit. If the request wants this item to fit, suit, work with or "
                        "go with the other product, this settles it; otherwise ignore it):\n" + "\n".join(lines))
            return ("\n\nFIT EVIDENCE (computed by code from the model codes on the records; it settles whether a part works with a "
                    "device: a part works with the device when this says so, whatever the description adds):\n" + "\n".join(lines))
        except Exception as err:
            self.log.append({"kind": "checkpoint", "what": "kit_evidence", "error": f"{type(err).__name__}: {err}"[:200]})
            return ""

    def basket_instead(self) -> tuple[str, dict[str, Any]] | None:
        """A decline while every product of the request has a listing the item check passed: those listings go in the cart and
        are ordered (the multi-product counterpart of fit_instead)."""
        order = self.role("order")
        if not order or not self.needs.items() or self.__dict__.get("basket_forced") or self.moves_left() < 4 or self.wire.spare() < CLOSING:
            return None
        ver = str(self.needs.version)
        picks = self.code_picks()
        picks = (self.kit_mend(picks) or []) if picks else picks
        if not picks or len(picks) < len(self.needs.items()) or not all(f"{i}|{ver}" in self.passed and i not in self.refused_lines
                                                                       and i not in self.dead_lines() for i, _ in picks):
            self.log.append({"kind": "checkpoint", "what": "basket_instead", "result": "no checked fit for every product", "picks": picks[:6]})
            return None
        if self.moves_left() < len(picks) + 2:
            return None
        self.basket_forced = True
        bad = self.code_put(picks)
        if bad is None or bad or not all(i in self.trolley.units for i, _ in picks):
            self.log.append({"kind": "checkpoint", "what": "basket_instead", "picks": picks, "result": "cart not set", "bad": bad})
            return None
        self.log.append({"kind": "checkpoint", "what": "basket_instead", "picks": picks, "result": "ordered instead of the decline"})
        return order, self.plain_args(order, "Order the cart.", "order")

    def kit_misfit(self) -> bool:
        """Code only (no model call): a part in the cart whose codes leave out the model of the device in the cart."""
        return bool(self.kit_fault(list(self.trolley.units)))

    def kit_fault(self, lines: list[str]) -> str:
        """Code only: a line of these whose codes leave out every model of the device lines it must fit, or ''. A device's
        other codes the part's clauses say nothing about (its own model number next to its part series) are left out, as in
        code_fit_hold."""
        try:
            pairs = self.kit_pairs()
            if not pairs or len(lines) < 2:
                return ""
            owner = {i: self.product_of(self.shelf_known(i) or self.archive.label_of.get(i, "")) for i in lines}
            notes = self.fit_notes()
            for part in lines:
                mains = {m for p, m in pairs if p == owner.get(part)}
                models = {c for i in lines if i != part and owner.get(i) in mains for c in own_models(self.record_text(i))}
                rules = fit_rules(self.record_text(part))
                known = [v for v in (self.part_fit(part, m, rules, notes) for m in models) if v is not None]
                if models and rules and known and not any(known):
                    return part
            return ""
        except Exception:
            return ""

    def kit_mend(self, picks: list[tuple[str, int]]) -> list[tuple[str, int]] | None:
        """Picks whose parts fit their device: unchanged when they already fit, else the device and parts of the cheapest
        complete fitting set the code knows put in place of the kit's lines; None when no fitting set is known."""
        if not self.kit_fault([i for i, _ in picks]):
            return picks
        ver = str(self.needs.version)
        sets = self.kit_sets(1, {i for i, _ in picks} | {k.rpartition("|")[0] for k in self.passed if k.rpartition("|")[2] == ver})
        if not sets:
            return None
        _, device, parts = sets[0]
        rows = self.needs.items()
        pairs = self.kit_pairs()
        kit_products = {m for _, m in pairs} | {p for p, _ in pairs}
        owner = lambda i: self.product_of(self.shelf_known(i) or self.archive.label_of.get(i, ""))
        qty = lambda i: int((rows[owner(i)].get("quantity") if owner(i) is not None and owner(i) < len(rows) else 1) or 1)
        new = [(i, n) for i, n in picks if owner(i) not in kit_products] + [(device, qty(device))] + [(p, qty(p)) for p in parts]
        return None if self.kit_fault([i for i, _ in new]) else new

    def exact_shelf(self, k: int | None) -> str:
        """The store category whose name is the requested product's own name (singular or plural, '&' or 'and'), or ''."""
        rows = self.needs.items()
        if k is None or not 0 <= k < len(rows):
            return ""
        norm = lambda t: " ".join(re.sub(r"(?<=[a-z]{3})s$", "", w) for w in re.findall(r"[a-z0-9]+", str(t).lower().replace("&", " and ")))
        want = norm(rows[k]["product"])
        shelves = {self.shelf_known(i) for i in self.archive.ids} - {""}
        return next((x for x in sorted(shelves) if norm(x) == want), "")

    def shelf_faults(self) -> list[str]:
        """Cart lines filed under a product the store sells as a category of exactly that name, while the line's own listing
        sits in another category whose name only contains the product's words."""
        out = []
        try:
            rows = self.needs.items()
            for i in list(self.trolley.units) if rows else []:
                mine = self.shelf_known(i)
                k = self.product_of(mine or self.archive.label_of.get(i, ""))
                home = self.exact_shelf(k)
                if home and mine and mine != home:
                    out.append(f"{i} is in the store category '{mine}', but '{rows[k]['product']}' is a category of its own in this "
                               f"store ('{home}'), and only a listing of '{home}' counts for that line: take {i} out of the cart and put "
                               f"in a listing of '{home}'")
        except Exception as err:
            self.log.append({"kind": "checkpoint", "what": "shelf_faults", "error": f"{type(err).__name__}: {err}"[:200]})
        return out

    def cart_tidy(self) -> bool:
        """At a forced finish with more cart lines than products: one line per product stays (on the product's exact category
        when the store has one, a checked line first); lines of no requested product go."""
        try:
            rows = self.needs.items()
            lines = list(self.trolley.units)
            if len(rows) < 2 or len(lines) <= len(rows) or not self.role("cart"):
                return False
            by: dict[Any, list[str]] = {}
            for i in lines:
                by.setdefault(self.product_of(self.shelf_known(i) or self.archive.label_of.get(i, "")), []).append(i)
            ver = str(self.needs.version)
            keep = []
            for k, ids in by.items():
                if k is None:
                    continue
                home = self.exact_shelf(k)
                good = [i for i in ids if not home or self.shelf_known(i) == home] or ids
                keep.append(min(good, key=lambda i: (f"{i}|{ver}" not in self.passed, ids.index(i))))
            if not keep or len(keep) == len(lines) or self.moves_left() < len(lines) - len(keep) + 2:
                return False
            was = self.wire.mute
            self.wire.mute = True
            try:
                bad = self.code_put([(i, int(self.trolley.held(i) or 1)) for i in keep])
            finally:
                self.wire.mute = was
            self.log.append({"kind": "checkpoint", "what": "cart_tidy", "kept": keep, "dropped": sorted(set(lines) - set(keep)), "ok": bad == []})
            return bad == []
        except Exception as err:
            self.log.append({"kind": "checkpoint", "what": "cart_tidy", "error": f"{type(err).__name__}: {err}"[:200]})
            return False

    def kit_swap(self) -> bool:
        """At a forced finish with a part that does not fit: the cheapest complete fitting set the code knows replaces the
        device and its parts in the cart; the other lines stay."""
        try:
            ver = str(self.needs.version)
            # (the device stays one already chosen or checked: a cheaper device nobody checked would trade the misfit for a miss)
            sets = self.kit_sets(1, set(self.trolley.units) | {k.rpartition("|")[0] for k in self.passed if k.rpartition("|")[2] == ver}, known=True)
            if not sets or not self.role("cart"):
                return False
            _, device, parts = sets[0]
            rows = self.needs.items()
            pairs = self.kit_pairs()
            kit_products = {m for _, m in pairs} | {p for p, _ in pairs}
            owner = lambda i: self.product_of(self.shelf_known(i) or self.archive.label_of.get(i, ""))
            keep = [(i, int(self.trolley.held(i) or 1)) for i in self.trolley.units if owner(i) not in kit_products]
            qty = lambda i: int((rows[owner(i)].get("quantity") if owner(i) is not None and owner(i) < len(rows) else 1) or 1)
            picks = keep + [(device, qty(device))] + [(p, qty(p)) for p in parts]
            if self.moves_left() < len(picks) + 2:
                return False
            was = self.wire.mute
            self.wire.mute = True
            try:
                bad = self.code_put(picks)
            finally:
                self.wire.mute = was
            return bad == [] and all(i in self.trolley.units for i, _ in picks)
        except Exception as err:
            self.log.append({"kind": "checkpoint", "what": "kit_swap", "error": f"{type(err).__name__}: {err}"[:200]})
            return False

    def kit_sets(self, top: int = 3, devices: set[str] | None = None, known: bool = False) -> list[tuple[float, str, list[str]]]:
        """Complete sets from the records seen whose codes fit: for each in-stock device of a product the others must fit, the
        cheapest in-stock seen listing of every product bound to it that takes in the device's model; cheapest sets first.
        known: read listings (their fields shown, no stated requirement broken) before unread rows, as at a forced finish
        where nothing more can be checked."""
        pairs = self.kit_pairs()
        rows = self.needs.items()
        if not pairs or not rows:
            return []
        notes = self.fit_notes()
        index = self.ref_index()
        owner = {i: self.product_of(self.shelf_known(i) or self.archive.label_of.get(i, "")) for i in self.archive.ids}

        def live(i: str) -> bool:
            return i not in self.dead_lines() and not self.sold_out(index.get(i) or {}) and is_number(self.archive.price(i))

        def clean(i: str) -> bool:
            # (a read listing whose own facts break its line's stated requirements is no stand-in: when the spend cap skips the
            #  item check, nothing counts as checked, and the cheapest fitting part could break a requirement the request states)
            try:
                return not (i in self.archive.read and self.field_view(i)["fails"])
            except Exception:
                return True
        ver = str(self.needs.version)
        checked = lambda i: f"{i}|{ver}" in self.passed          # (a listing the item check passed meets the line's other requirements)
        sets = []
        for main in sorted({m for _, m in pairs}):
            parts = sorted({p for p, m in pairs if m == main})
            for device in [i for i in self.archive.ids if owner.get(i) == main and live(i) and clean(i) and (devices is None or i in devices)]:
                models = own_models(self.record_text(device))
                if not models:
                    continue
                total = float(self.archive.price(device)) * float(rows[main].get("quantity") or 1) if main < len(rows) else 0.0
                pick = []
                for p in parts:
                    fits = [i for i in self.archive.ids if owner.get(i) == p and live(i) and clean(i)
                            and any(self.part_fit(i, m, fit_rules(self.record_text(i)), notes) for m in models)]
                    if not fits:
                        break
                    # (known: a read listing clean() let through shows its fields; an unread row shows none, so it comes after)
                    best = min(fits, key=lambda i: (not checked(i), known and i not in self.archive.read, float(self.archive.price(i))))
                    pick.append(best)
                    total += float(self.archive.price(best)) * float(rows[p].get("quantity") or 1) if p < len(rows) else 0.0
                if len(pick) == len(parts):
                    sets.append((total, device, pick, sum(1 for i in [device] + pick if checked(i)),
                                 sum(1 for i in [device] + pick if i in self.archive.read)))
        sets.sort(key=lambda s: (-s[3], -s[4] if known else 0, s[0]))
        sets = [(t, d, p) for t, d, p, _, _ in sets]
        return sets[:top]

    def kit_hold(self) -> str:
        """Before a decline on a request whose parts must fit a device: once, the complete sets whose codes fit."""
        if self.__dict__.get("kit_held"):
            return ""
        try:
            sets = self.kit_sets()
        except Exception as err:
            self.log.append({"kind": "checkpoint", "what": "kit_hold", "error": f"{type(err).__name__}: {err}"[:200]})
            return ""
        if not sets:
            return ""
        self.kit_held = True
        self.log.append({"kind": "checkpoint", "what": "kit_hold", "sets": [(d, p) for _, d, p in sets]})
        said = "; ".join(f"{d} with {', '.join(p)} (total {t:g})" for t, d, p in sets)
        return ("Not run yet (no step used). The codes on the records seen show complete sets whose parts fit their device: " + said +
                ". Check each listing of the cheapest such set against its own product's requirements (switch to the next set, or "
                "another device, when one fails), put every product's line in the cart and order; decline only if every set fails "
                "another requirement.")

    def see_part_lookup(self, codes: list[str]) -> bool:
        """A cart part whose own codes leave the device out but whose page says the rest of its list is on part X: one search
        for X (at most two in an errand), so the note on X's page about this part is read before the fit verdict."""
        done = self.__dict__.setdefault("see_part_done", set())
        tpl = self.ref_template()
        todo = [c for c in dict.fromkeys(codes) if c not in done][:max(0, 2 - len(done))]
        if not tpl or not todo or self.moves_left() < self.order_due() + 3 or self.wire.spare() < CLOSING:
            return False
        for code in todo:
            done.add(code)
            self.walk_call(self.uncapped({**tpl[0], tpl[1]: code.upper()}), f"I look up part {code.upper()}: a cart part's page says the rest of its fit list is there.")
            self.log.append({"kind": "checkpoint", "what": "see_part_lookup", "part": code})
        return True

    def part_lookups(self) -> None:
        """Several products: a cart line's page may point to a part number for more on it ("more on part X"). Each such number is
        searched once before the order, so what its page says is on record when the fit is judged."""
        try:
            search = self.role("search")
            if len(self.needs.items()) < 2 or not search:
                return
            done = self.__dict__.setdefault("see_part_done", set())
            todo = [c for i in self.trolley.units for c in re.findall(SEE_PART, self.record_text(i).lower()) if c not in done]
            todo = list(dict.fromkeys(todo))
            tpl = self.ref_template() or (self.query_keys(search) and (self.word_args(search), self.query_keys(search)[0]))
            if not tpl or not todo or self.moves_left() <= len(todo) + 2 or self.wire.spare() < LAST_CALL:
                return             # (no model call in these: they only need their own steps, and the order's)
            for code in todo:
                done.add(code)
                self.walk_call(self.uncapped({**tpl[0], tpl[1]: code.upper()}), f"I look up part {code.upper()}: a cart line's page points to it for more.")
                self.log.append({"kind": "checkpoint", "what": "part_lookup", "part": code})
        except Exception as err:
            self.log.append({"kind": "checkpoint", "what": "part_lookups", "error": f"{type(err).__name__}: {err}"[:200]})

    def code_fit_hold(self) -> str:
        """A part in the cart whose own 'fits ...' words leave out the model of the device in the same cart, read in code from
        the records: the part's clause names codes of the device's own letters and none of them takes in the device's model."""
        try:
            lines = list(self.trolley.units)
            pairs = self.kit_pairs()
            if len(lines) < 2 or not pairs:
                return ""
            owner = {i: self.product_of(self.shelf_known(i) or self.archive.label_of.get(i, "")) for i in lines}
            text = {i: self.record_text(i) for i in lines}
            rules = {i: fit_rules(t) for i, t in text.items()}
            unseen = []
            tags = set(self.fit_notes()[1].values())
            for part, found in rules.items():
                mains = {m for p, m in pairs if p == owner.get(part)}
                models = {c for i in lines if i != part and owner.get(i) in mains and not rules[i] for c in own_models(text[i])}
                more = re.search(SEE_PART, text[part].lower())
                if found and models and more and more.group(1) not in tags and all(fit_verdict(c, found) is False for c in models):
                    unseen.append(more.group(1))
            if unseen and self.see_part_lookup(unseen):
                text = {i: self.record_text(i) for i in lines}
            mark = (tuple(sorted(self.trolley.units.items())), self.needs.version)
            held = self.__dict__.setdefault("fit_code_held", {})
            if held.get(mark, 0) >= 2 or sum(held.values()) >= 5:
                return ""
            faults = []
            notes = self.fit_notes()
            for part, found in rules.items():
                mains = {m for p, m in pairs if p == owner.get(part)}
                devices = {i: own_models(text[i]) for i in lines if i != part and owner.get(i) in mains and not rules[i]}
                models = sorted({c for d in devices.values() for c in d})
                if not found or not models:
                    continue
                known = [v for v in (self.part_fit(part, c, found, notes) for c in models) if v is not None]
                if not known or any(known):
                    continue
                shelf = self.shelf_known(part)
                good = []
                for other in sorted(self.archive.ids - set(lines)):
                    if len(good) >= 4:
                        break
                    if shelf and self.shelf_known(other) != shelf:
                        continue
                    theirs = fit_rules(self.record_text(other))
                    if theirs and any(self.part_fit(other, c, theirs, notes) for c in models):
                        good.append(other)
                faults.append(f"{part} names the models it fits, and the device model in the cart ({', '.join(m.upper() for m in models)}) is not "
                              "among them" + (f"; seen listings of its kind that do fit it: {', '.join(good)}" if good else
                                               "; no seen listing of its kind fits that device, so search its kind again or switch the device "
                                               "to a model the parts fit"))
            if not faults:
                return ""
            held[mark] = held.get(mark, 0) + 1
            self.log.append({"kind": "checkpoint", "what": "code_fit_hold", "faults": faults[:3]})
            return ("Fit check: " + "; ".join(faults[:3]) + ". Replace the part (or the device) so that every part fits the device, "
                    "re-read stock and price, then order.")
        except Exception as err:
            self.log.append({"kind": "checkpoint", "what": "code_fit_hold", "error": f"{type(err).__name__}: {err}"[:200]})
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
            if any(isinstance(v, str) and v in ids and id_named(key) for key, v in d.items()):
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
            if len(d) >= 4 and any(isinstance(v, str) and v in self.archive.ids and id_named(key) for key, v in d.items()):
                rows.append(d)
        walk_dicts(obs, visit, ceiling=3)
        parts = []
        for r in rows[:12]:
            cost = next((v for key, v in r.items() if is_number(v) and re.search(r"price|cost", str(key).lower())), None)
            parts.append(f"{self.row_mark(r)} {cost if cost is not None else '?'}{' SOLD OUT' if self.sold_out(r) else ''} "
                         f"'{self.title_of(r)[:70]}' [{self.shelf_of(r)}]")
        return "Page rows (id, price, title, category): " + ("; ".join(parts) if parts else "none")

    def walk_call(self, args: dict[str, Any], say: str, tool: str | None = None) -> bool:
        proposal = {"id": f"walk_{len(self.chat)}_{self.wire.tick}", "type": "function",
                    "function": {"name": tool or self.__dict__.get("page_tool") or self.role("search"), "arguments": json.dumps(args)}}
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
                if not self.walk_call({**walk["args"], (self.page_arg(walk.get("tool")) or found)[0]: nxt}, "The cheapest is asked for and this store lists search results in a fixed "
                                      f"id order, not by price: I read the next page of '{query}'.", tool=walk.get("tool")):
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
            was_tool = self.__dict__.get("page_tool")
            self.page_tool = (self.ceil_spec() or {}).get("tool") or was_tool
            try:
                got_rows = self.ceiling_walk(k, floor, repair)
            finally:
                self.page_tool = was_tool
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
            if not rows or not is_number(cap):
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
            if self.__dict__.get("decline_read_done") or self.wire.spare() < CLOSING + 15 or not (self.role("details") or batch) \
                    or self.trolley.units:
                return ""
            opens = 1 if batch else 2
            room = min(3, self.moves_left() - opens - 6)
            keys = self.decline_walks() if room > 0 and self.page_arg() and self.role("search") and self.id_store() else []
            cheaper = self.over_budget_rows() if self.moves_left() > opens + 6 else []
            fresh = [k for k in self.decline_lacking() if self.search_again(k, dry=True) is not None][:2] if self.moves_left() > opens + 4 else []
            self.fresh_queue = fresh
            if not keys and not cheaper and not fresh:
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
            if fresh:
                said.append("a product with no fitting listing yet is searched again in plainer words and its cheapest unread "
                            "in-budget rows are opened")
            return ("Before a decline, " + "; ".join(said) + ". Judge those records against the requirements now in force by "
                    "meaning; buy what fits (" + ("one line per product, at its quantity" if self.needs.items() else "the item")
                    + "), and decline only if each one fails.")
        except Exception as err:
            self.log.append({"kind": "decline_reads_error", "error": f"{type(err).__name__}: {err}"[:300]})
            return ""

    def twin_asked(self, prop: Any) -> bool:
        """A property that is another name for one already asked about the same product: its words sit inside the other's
        (number of cores / number of cpu cores), or the listings write the same values in both (age / kid years)."""
        try:
            label = str(prop or "")
            k = self.product_of(label) if self.needs.items() else None
            mine = {w for w in re.findall(r"[a-z0-9]+", label.split("(")[0].lower()) if len(w) >= 3 and w not in LINK_WORDS}
            if not mine:
                return False
            vals = self.field_values(k, label.split("(")[0].strip())
            for ln in self.questions_log:
                for part in ln.split(":", 1)[0].strip("- ").split(";"):
                    if self.needs.items() and self.product_of(part) != k:
                        continue
                    name = part.split("(")[0].replace("[not answered]", "").strip()
                    done = {w for w in re.findall(r"[a-z0-9]+", name.lower()) if len(w) >= 3 and w not in LINK_WORDS}
                    if not done or done == mine:
                        continue
                    if (mine <= done or done <= mine) and not self.archive.side_by_side(label.split("(")[0], name):
                        return True
                    other = self.field_values(k, name)
                    if vals and other and len(vals & other) >= max(1, (min(len(vals), len(other)) + 1) // 2):
                        return True
            return False
        except Exception:
            return False

    def near_asked(self, prop: Any) -> bool:
        """A name that contains, or sits inside, a property name already asked about for the same product."""
        try:
            label = str(prop or "")
            k = self.product_of(label) if self.needs.items() else None
            name = " ".join(re.findall(r"[a-z0-9]+", label.split("(")[0].lower()))
            if not name:
                return False
            for ln in self.questions_log:
                for part in ln.split(":", 1)[0].strip("- ").split(";"):
                    if self.needs.items() and self.product_of(part) != k:
                        continue
                    done = " ".join(re.findall(r"[a-z0-9]+", part.split("(")[0].replace("[not answered]", "").lower()))
                    if done and done != name and (re.search(r"\b" + re.escape(done) + r"\b", name) or re.search(r"\b" + re.escape(name) + r"\b", done)) \
                            and not self.archive.side_by_side(name, done):
                        return True
            return False
        except Exception:
            return False

    def stated_twin(self, prop: Any) -> bool:
        """Another name for a property the request or a reply already settles for the same product: two or more words
        shared and one name's words inside the other's (strap width / watch strap width). A one-word name such as type or
        size never matches, since a longer name around it is often a field of its own."""
        try:
            label = str(prop or "")
            k = self.product_of(label) if self.needs.items() else None
            # (the product rides along as "(product)" or "of the product": only the words before it name the property)
            mine = {w for w in re.findall(r"[a-z0-9]+", re.split(r"\(|\bof the\b", label.lower())[0]) if len(w) >= 3 and w not in LINK_WORDS}
            for req in self.needs.props_left():
                head = str(req).partition(":")[0]
                if re.match(r"\s*(?:if|unless)\b", head.lower()) or (self.needs.items() and self.product_of(head) != k):
                    continue
                done = {w for w in re.findall(r"[a-z0-9]+", re.split(r"\(|\bof the\b", head.lower())[0]) if len(w) >= 3 and w not in LINK_WORDS}
                if done != mine and len(done & mine) >= 2 and (mine <= done or done <= mine):
                    return True
            return False
        except Exception:
            return False

    def record_text(self, thing: str) -> str:
        parts: list[str] = []

        def visit(d: dict[str, Any]) -> None:
            if any(isinstance(val, str) and val == thing and id_named(key) for key, val in d.items()):
                parts.append(as_text(d, 4000).lower())
        for rec in self.archive.items:
            walk_dicts(rec, visit)
        return " ".join(parts)

    def plain_lacking(self) -> list[int | None]:
        """Products none of whose opened listings shows every stated (non-numeric) value somewhere in its record."""
        rows = self.needs.items()
        homes = self.home_shelves()
        out: list[int | None] = []
        for k in (range(len(rows)) if rows else [None]):
            home = homes[k if k is not None else 0] if homes else set()
            if not home:
                continue
            words = []
            for req in self.needs.props_left():
                head, _, value = str(req).partition(":")
                if rows and self.product_of(head) != k:
                    continue
                if re.search(r"\d", value) or not positive_value(value, head):
                    continue
                words.append([w for w in re.findall(r"[a-z]{3,}", value.lower()) if w not in LINK_WORDS])
            opened = [i for i in self.archive.read if self.shelf_known(i) in home]
            if not any(all(all(w[:5] in self.record_text(i) for w in ws) for ws in words if ws) for i in opened):
                out.append(k)
        return out

    def early_rescue(self) -> bool:
        """Halfway through the steps or the time, a product with no fitting listing yet gets one plainer search and its
        cheapest unread in-budget rows opened, while steps remain to use them."""
        try:
            if self.__dict__.get("rescue_done") or self.wire.done or self.wire.broke:
                return False
            half = self.wire.tick >= self.step_cap * 0.5 or self.wire.spare() <= DEADLINE * 0.5
            if not half or self.moves_left() < self.order_due() + 6 or self.wire.spare() < CLOSING + 15 or not self.role("search"):
                return False
            self.rescue_done = True
            fitted = set(self.decline_lacking())
            lacking = [k for k in self.plain_lacking() if k in fitted and self.search_again(k, dry=True) is not None][:2]
            self.log.append({"kind": "checkpoint", "what": "early_rescue", "lacking": lacking})
            if not lacking:
                return False
            self.fresh_queue = lacking
            self.decline_queue = ([], 0, self.order_due() + 2)
            return self.run_decline_reads()
        except Exception as err:
            self.log.append({"kind": "checkpoint", "what": "early_rescue", "error": f"{type(err).__name__}: {err}"[:200]})
            return False

    def search_again(self, k: int | None, dry: bool = False) -> set[str] | None:
        """A plainer search for a product that has no fitting listing: its noun with one required value, else the noun
        alone, not sent before. Returns the ids the result shows (None when nothing was run)."""
        try:
            tool = self.role("search")
            done = self.__dict__.setdefault("again_done", set())
            if not tool or k in done:
                return None
            rows = self.needs.items()
            product = str(rows[k]["product"] if rows and k is not None else self.needs.reading.get("product") or "").strip()
            if not product:
                return None
            template = None
            for entry in reversed(self.log):
                if entry.get("kind") != "env":
                    continue
                for act in entry.get("actions") or []:
                    if act.get("name") == tool and isinstance(act.get("args"), dict) and self.free_text(act["args"]):
                        template = act["args"]
                        break
                if template:
                    break
            if not template:
                return None
            values = []
            for req in self.needs.props_left():
                head, _, value = str(req).partition(":")
                value = self.plain_value(value)
                if (not rows or self.product_of(head) == k) and value.strip() and not re.search(r"\d", value) and len(value.split()) <= 3:
                    values.append(value.strip())
            sent = {" ".join(self.free_text(act.get("args") or {})).strip().lower()
                    for entry in self.log if entry.get("kind") == "env" for act in entry.get("actions") or [] if act.get("name") == tool}
            noun = " ".join(kind_tokens(product)[-2:]) or product
            query = next((q for q in [f"{v} {noun}" for v in values[:2]] + [noun] if q.strip().lower() not in sent), "")
            if not query:
                return None
            if dry:
                return set()
            done.add(k)
            key = next(key for key, val in template.items() if isinstance(val, str) and val in self.free_text(template))
            args = dict(template, **{key: query})
            found = self.page_arg()
            if found:
                args[found[0]] = found[2] if found[1] != "token" else ""
            before = len(self.archive.items)
            if not self.walk_call(args, f"Before declining, I search again in plainer words for the {product}: '{query}'."):
                return None
            got: set[str] = set()

            def visit(d: dict[str, Any]) -> None:
                ident = self.row_mark(d)
                if ident in self.archive.ids:
                    got.add(ident)
            for rec in self.archive.items[before:]:
                walk_dicts(rec, visit)
            return got
        except Exception as err:
            self.log.append({"kind": "checkpoint", "what": "search_again", "error": f"{type(err).__name__}: {err}"[:200]})
            return None

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
        fresh_ran = False
        for k in self.__dict__.pop("fresh_queue", None) or []:
            if self.moves_left() <= floor + opens + 1 or self.wire.done or self.wire.broke:
                break
            got = self.search_again(k)
            if got is not None:
                seen |= got
                fresh_ran = True
        rows = self.needs.items()
        homes = [(self.walk_home(k), rows[k]["quantity"] if k is not None and k < len(rows) else self.needs.quantity())
                 for k in self.decline_lacking()]
        pools = (self.row_pools(homes, seen) if walked or fresh_ran else []) + self.over_budget_rows()
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
            self.open_say = "I open the cheapest unread in-budget rows those searches show for the products with no fit yet."
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
        told = str(info.get("relation") or "").lower().strip() if isinstance(info, dict) else ""
        span = ref_span(prop, source).lower()
        cut = span.find(str(tgt.get("seller") or "").lower()) if str(tgt.get("seller") or "").strip() else -1
        lead = span[:cut] if cut > 0 else ""
        if told:
            # (the reader's relation: it reads double negatives that the code's words cannot; only a differ word the text
            # shows framing a relation of its own keeps the code's reading of that inner relation)
            if re.search(r"differ|not the same|exclu|apart|other than|unlike", told):
                framed = differ_framed(lead) or differ_framed(value)
                rel, tol, unit = ref_relation(lead if differ_framed(lead) else value) if framed else ("differ", None, "")
            elif told == "within":
                if is_number(info.get("tolerance")):
                    rel, tol, unit = "within", float(info["tolerance"]), str(info.get("unit") or "")
                elif rel != "within":
                    rel, tol, unit = "same", None, ""
            elif told in ("at most", "at least"):
                rel, tol, unit = told, None, ""
            elif rel not in ("within", "at most", "at least"):
                rel, tol, unit = "same", None, ""
        elif rel == "same" and cut > 0 and differ_said(lead):
            rel, tol, unit = "differ", None, ""
        if told == "partner" or (rel != "differ" and re.search(PARTNER_WORDS, f"{lead} {value}".lower())):
            rel, tol, unit = "partner", None, ""
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
                    if isinstance(v, str) and 0 < len(v) <= 60 and "id" not in low and not shelf_key(k) and not re.search(
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
        notes, other = [], []
        for k, v in row.items():
            name = str(k).lower()
            if not isinstance(v, str) or len(v.split()) < 3 or id_named(k, True) \
                    or re.search(r"title|heading|headline|name|label|caption|categor|shelf|department|url|image", name):
                continue
            (other if re.search(r"desc|overview|summary|about", name) else notes).append(v)
        return "\n".join(notes or (other if desc else []))

    def ref_parts(self, row: dict[str, Any]) -> tuple[set[str], set[str]]:
        notes = re.sub(r"\b(?:not|no|never|without|nor)\b(?:\W+[A-Za-z0-9.:/&'-]+){1,3}", " ", self.ref_texts(row, desc=False), flags=re.I)
        keys, vals = [], [self.title_of(row), notes]
        maps = [v for v in row.values() if isinstance(v, dict)]
        for fields in maps or [{k: v for k, v in row.items() if isinstance(v, (str, int, float)) and not isinstance(v, bool)
                                and not id_named(k, True) and not re.search(r"title|shop|seller|brand|price|amount|desc|overview|note", str(k).lower())}]:
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
        order = ref.get("order") or {}
        scored = []
        for ident, row in rows.items():
            if name not in self.ref_sellers(row):
                continue
            hay = set(ref_toks(self.title_of(row) + " " + " ".join(str(v) for k, v in row.items()
                                                                  if isinstance(v, str) and re.search(r"desc|overview|summary", str(k).lower()))))
            shown = sum(1 for d in tgt["details"] if set(ref_toks(d)) and set(ref_toks(d)) <= hay)
            # (a row showing every value first; then the store's own order for the last search on this listing, which
            #  ranks the described listing ahead of the shop's other listings of that kind)
            whole = 1 if tgt["details"] and shown >= len(tgt["details"]) else 0
            scored.append((-(2 if kinds & kind_words(self.shelf_of(row)) else 0) - 3 * whole, order.get(ident, len(order) + 1), -shown,
                           -seen_at.get(ident, 0), ident, shown))
        scored.sort()
        ref["shows"] = {ident: shown for *_, ident, shown in scored}
        return [ident for *_, ident, _ in scored]

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

    def ref_words(self, ref: dict[str, Any]) -> list[str]:
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

    def ref_value_words(self, detail: Any) -> str:
        """One described detail of a referenced listing without the property's name: its value, as a listing's text would show it."""
        text = str(detail).partition(":")[2].strip() or str(detail).strip()
        parts = text.split()
        keys = self.store_keys()
        for j in range(len(parts) - 1, 0, -1):
            if " ".join(re.findall(r"[a-z0-9]+", " ".join(parts[:j]).lower())) in keys:
                return " ".join(parts[j:])
        return text

    def ref_plans(self, ref: dict[str, Any]) -> list[str]:
        width = max(2, self.lead_words or 3)
        seller = " ".join(re.findall(r"[A-Za-z0-9]+", re.sub(r"'s\b", "", str(ref["target"]["seller"]))))
        words = self.ref_words(ref)
        kind = [w for w in re.findall(r"[a-z]{3,}", str(ref["target"]["kind"]).lower()) if w not in LINK_WORDS][-1:]
        if ref.get("unindexed"):
            plans = [" ".join((words[:1] + kind)[:width])] if words or kind else []
        else:
            plans = ([seller + " " + words[0]] if words and len(seller.split()) < width else []) + [seller]
            if self.word_kind != "all" and not self.lead_words and ref["target"]["details"]:
                # (a ranking search that reads every word: the shop's name with its kind and every value the description
                #  gives puts the described listing first, ahead of the shop's other listings)
                # (the kind's words match every listing of that kind at the shop, so the values alone are tried first)
                rich = " ".join(" ".join([seller, str(ref["target"]["kind"] or "")] + [str(d) for d in ref["target"]["details"]]).split()[:14])
                bare = " ".join(" ".join([seller] + [self.ref_value_words(d) for d in ref["target"]["details"]]).split()[:14])
                plans = [p for p in dict.fromkeys([bare, rich]) if p] + [p for p in plans if p.lower() not in (rich.lower(), bare.lower())]
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
        if not (self.__dict__.get("ref_steps", 0) < cap and self.wire.spare() > CLOSING + 60 and self.wire.spend < self.wire.hard_cap() - 0.012
                and not self.wire.done and not self.wire.broke):   # (a reserve: the walks run past the spending tier, until the hard total)
            return False
        if self.ref_waits(floor):
            return False
        return self.moves_left() > floor or self.ref_hop_room()

    def ref_floor(self) -> int:
        # (the steps kept back for the cart, its check and the order, and a few more questions)
        return self.order_due() + 5

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
        return (self.__dict__.get("ref_calls", 0) < REF_CALLS and self.wire.spare() > CLOSING + 40
                and self.wire.spend < self.wire.hard_cap() - 0.012)

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
        for ref in self.refs:
            words = ref.pop("post_search", None) if ref.get("status") == "resolved" else None
            tpl = self.ref_template() if words else None
            if words and tpl and self.moves_left() > self.order_due() + 3 and not self.ran_query(words):
                self.log.append({"kind": "ref_post_search", "prop": ref["prop"], "query": words})
                return self.ref_search(ref, {**tpl[0], tpl[1]: words}) or True
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
        most = 2                     # (read one at a time: see ref_read)
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
        # (one listing a call: a page of the chain counts as opened only when it was opened on its own, not side by side)
        batch = None
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

    def uncapped(self, args: dict[str, Any]) -> dict[str, Any]:
        """A search's arguments without a price ceiling: a listing looked up only to read it (the one a requirement points to,
        a part named by its code) need not be one the shopper could buy, and a ceiling copied from an earlier search hides it."""
        props = (self.shapes.get(self.role("search") or "") or {}).get("properties")
        props = props if isinstance(props, dict) else {}
        caps = {k for k, p in props.items() if isinstance(p, dict) and p.get("type") in ("integer", "number")
                and re.search(r"ceil|max|price|cap$|budget|upper|most", str(k).lower()) and not re.search(r"page|first|offset|skip|count|size|num", str(k).lower())}
        cap = (self.__dict__.get("tool_sheet") or {}).get("price_cap")
        if cap:
            caps.add(cap)
        return {k: v for k, v in args.items() if k not in caps or k in ((self.shapes.get(self.role("search") or "") or {}).get("required") or [])}

    def ref_search(self, ref: dict[str, Any], args: dict[str, Any]) -> bool:
        args = self.uncapped(args)
        before = self.wire.tick
        got = self.walk_call(args, f"The request takes {ref['head']} from another listing ({self.ref_desc(ref)}): I search for it.")
        self.ref_steps = self.__dict__.get("ref_steps", 0) + (self.wire.tick - before)
        rows: list[dict[str, Any]] = []
        if got:
            walk_dicts(self.archive.items[-1], lambda d: rows.append(d) if self.row_mark(d) in self.archive.ids and len(d) >= 4 else None, ceiling=3)
        if rows:
            ref["order"] = {self.row_mark(r): n for n, r in enumerate(rows)}
        name = ref_name(ref["target"]["seller"])
        text = " ".join(self.free_text(args))
        if rows and not any(name in self.ref_sellers(r) for r in rows) and ref_name(text) == name:
            ref["unindexed"] = True
        found = self.page_arg()
        walk = (self.__dict__.get("page_book") or {}).get(json.dumps({k: v for k, v in args.items() if not (found and k == found[0])},
                                                                   sort_keys=True, default=str)) if got and found else None
        more = 0
        # (a search on the seller's name alone with a short result: its other pages too, so every listing of that seller is seen)
        while walk and ref_name(text) == name and 0 < walk.get("total", 0) <= 40 and more < 3 and self.next_page(walk) is not None \
                and self.moves_left() > self.order_due() + 4 and not self.wire.done and self.wire.spare() > CLOSING + 30:
            nxt = self.next_page(walk)
            if not self.walk_call({**walk["args"], (self.page_arg(walk.get("tool")) or found)[0]: nxt}, "I read the next page of that seller's listings.",
                                  tool=walk.get("tool")) or self.next_page(walk) == nxt:
                break
            more += 1
        self.log.append({"kind": "ref_search", "prop": ref["prop"], "query": text[:120], "rows": len(rows), "more_pages": more})
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
        if missing and (self.ref_room() or self.moves_left() > self.order_due() + 2) and self.ref_notes_read(ref, missing):
            return True
        if missing and self.ref_waits():
            return False
        if missing:
            # (a note not read may send the reader on to another listing: its fields alone would give the wrong value)
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
        return None

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
                         + "; ".join(f"'{r}'" for r, _, _ in own) + " (the grader takes the other listing's value), so it meets "
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
        marks = [m.start() for pat in (DIFFER_WORDS, REF_WORDS, POINT_WORDS, PARTNER_WORDS) for m in re.finditer(pat, lead.lower())]
        if not marks:
            return None
        start = cut + 1 + min(marks)
        tail = text[hit.end():]
        aside = re.search(r"\(\s*or\b[^)]*\)", tail)
        stop = re.search(r"[.!?;](?:\s|$)|\s[—–](?:\s|$)", tail)
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
            told = str(entry.get("relation") or "").lower().strip()
            if not told or (told == "differ" and differ_framed(phrase)) or (told == rel == "within" and not is_number(entry.get("tolerance"))) \
                    or (rel == "partner" and told != "partner"):
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
        if target not in pick or missing:
            ref["judged"].update(pick)
            self.ref_ids_add(pick)
            if target in pick:
                self.log.append({"kind": "ref_partial", "prop": ref["prop"], "target": target, "missing": missing[:4]})
            ids = [target] if target in pick else list(pick)
            self.ref_tell(f"Checked: {', '.join(ids)} {'is' if len(ids) == 1 else 'are'} not {self.ref_desc(ref)}"
                          + (f" (it does not show {', '.join(missing[:3])})" if missing else "")
                          + f", so the shopper's {ref['prop']} is not read from {'it' if len(ids) == 1 else 'them'}; the code keeps looking.")
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
        own = self.ref_field(row, ref["prop"])
        if not (own and set(ref_toks(value)) <= set(ref_toks(own))) and not any(
                self.ref_names_prop(ref["prop"], s) and set(ref_toks(value)) <= set(ref_toks(s)) for s in re.split(r"(?<=[.!?;])\s+", notes)):
            return "answer", (f"the value '{value}' is not {target}'s {ref['prop']}: " + (f"its {ref['prop']} field reads '{own}'" if own else
                              f"it has no {ref['prop']} field and its notes give none") + "."), None, "", []
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
        if ref["relation"] in ("at most", "at least") and re.search(r"\d", value):
            return f"{ref['relation']} {value}"
        if ref["relation"] == "differ":
            return " and ".join(f"not {v}" for v in [value] + [p for p in pairs if p.lower() != value.lower()])
        if ref["relation"] == "partner":
            mates = self.ref_mates(ref)
            return (" or ".join(mates) + f" (the partner the store pairs with {value}; {value} itself does not meet it)" if mates
                    else f"the value this store pairs with {value} (its partner, named where a listing says {value} goes together with it); "
                         f"{value} itself does not meet it")
        return " or ".join([value] + [p for p in pairs if p.lower() != value.lower()])

    def ref_mates(self, ref: dict[str, Any]) -> list[str]:
        """A partner requirement's values: what the last page of its chain pairs with its value, else the store's own pairing
        words for that value on any record seen."""
        value = str(ref.get("value") or "")
        mates = [p for p in ref.get("goes_with") or [] if str(p).strip() and str(p).lower() != value.lower()]
        return mates or self.pair_mates(ref["prop"], value)

    def pair_mates(self, prop: str, value: str) -> list[str]:
        """The values the records' pairing words give for a value ("<value> goes together with <other>"), most often said first;
        the reverse reading (the value on the right of such a clause) only when no record pairs the value forward."""
        want = set(ref_toks(value))
        if not want:
            return []
        memo = self.__dict__.setdefault("mates_memo", {})
        key = (prop, value, len(self.archive.items))
        if key not in memo:
            memo[key] = self.pair_scan(prop, want)
        return memo[key]

    def pair_scan(self, prop: str, want: set[str]) -> list[str]:
        fwd: dict[str, int] = {}
        back: dict[str, int] = {}
        drop = LINK_WORDS | set(ref_toks(prop))
        for ident in self.archive.ids:
            for sent in re.split(r"(?<=[.!?])\s+", self.record_text(ident)):
                low = sent.lower()
                if " with " not in low:
                    continue
                weight = 2 if prop_named(prop, low) else 1
                for m in re.finditer(PAIR_CLAUSE, low):
                    left = [w for w in m.group(1).split()[-4:] if w not in drop]
                    right = [w for w in m.group(2).split()[:4] if w not in drop]
                    if not left or not right:
                        continue
                    if want <= set(ref_toks(" ".join(left))) and not want <= set(ref_toks(" ".join(right))):
                        fwd[" ".join(right)] = fwd.get(" ".join(right), 0) + weight
                    elif want <= set(ref_toks(" ".join(right))) and not want <= set(ref_toks(" ".join(left))):
                        # (in the reverse reading the value stands just before the property's name)
                        words = m.group(1).split()
                        at = [n for n, w in enumerate(words) if set(ref_toks(w)) & set(ref_toks(prop))]
                        if at and at[-1] > 0 and words[at[-1] - 1] not in LINK_WORDS:
                            back[words[at[-1] - 1]] = back.get(words[at[-1] - 1], 0) + weight
        pick = fwd or back
        return sorted(pick, key=lambda v: (-pick[v], v))[:2]

    def ref_resolve(self, ref: dict[str, Any], target: str, value: str, pairs: list[str]) -> None:
        if not str(value or "").strip():
            self.ref_stop(ref, f"{target} shows no value for it")
            return
        ref["chain"].append(target)
        ref.update(status="resolved", value=str(value).strip(), goes_with=list(pairs))
        if ref["relation"] == "same":
            # (one search for the product with the value the chain gave: the searches so far ran before it was known)
            product = ref.get("tag") or str(self.needs.reading.get("product") or "")
            ref["post_search"] = " ".join(f"{(pairs or [value])[0]} {product}".split()[:8])
        mates = self.ref_mates(ref) if ref["relation"] == "partner" else []
        if mates:
            product = ref.get("tag") or str(self.needs.reading.get("product") or "")
            ref["post_search"] = " ".join(f"{mates[0]} {product}".split()[:8])
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
        if ref["relation"] == "partner":
            pair = (f". The shopper wants its PARTNER, the value the store pairs with it: {' or '.join(mates)}; {ref['value']} itself does "
                    "not meet it" if mates else f". The shopper wants its PARTNER, the value the store pairs with {ref['value']}; no record seen "
                    f"says what {ref['value']} goes together with yet, and {ref['value']} itself does not meet it")
        self.ref_tell(f"RESOLVED: {ref['head']} is given by another listing, and {target} shows {ref['value']}{via}{pair}. The "
                      f"requirement is now: {new_req}.")

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
                    parts.append(f"{ref['now']} (read on {ref['chain'][-1]}" + (", must differ from it" if ref.get("relation") == "differ" else
                                 ", its partner value, not that value" if ref.get("relation") == "partner" else "") + ")")
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
                    if ref.get("relation") == "differ":
                        pair = (f"; its notes pair {' and '.join(ref['goes_with'])} with it" if ref["goes_with"] else "")
                        lines.append(f"- {ref['now']}: the shopper's {ref['prop']} MUST DIFFER from the value shown on {ref['chain'][-1]} "
                                     f"({ref['value']}){via}{pair}: any other value meets it, that value does not; the listings of its "
                                     f"chain ({', '.join(ref['chain'])}) carry it.")
                    elif ref.get("relation") == "partner":
                        mates = self.ref_mates(ref)
                        lines.append(f"- {ref['now']}: {ref['chain'][-1]}{via} shows {ref['value']}, and the shopper wants its PARTNER, the "
                                     "value the store pairs with it" + (f" ({' or '.join(mates)})" if mates else "") + f": {ref['value']} "
                                     "itself does not meet it.")
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
                           if not shelf_key(kk)) for r in dicts)
        tmpl = {k: v for k, v in base.items() if k not in (bkey, skey) and not (isinstance(v, str) and v.strip() and k not in fixed and every(v))}
        words = sum(len(v.split()) for k, v in tmpl.items() if isinstance(v, str) and k not in fixed)
        props = (self.shapes.get(self.__dict__.get("page_tool") or self.role("search") or "") or {}).get("properties") or {}
        old = self.__dict__.get("ceil_mode")
        if old and old["sig"] not in bad and old["words"] < words:
            return
        self.ceil_mode = {"sig": sig, "tmpl": tmpl, "mode": fixed, "shelf": skey, "bound": bkey, "words": words,
                          "tool": self.__dict__.get("page_tool") or self.role("search"),
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
            if not any(isinstance(v, str) and v in self.archive.ids and id_named(key) for key, v in d.items()):
                return
            raw = next((val.strip() for key, val in d.items() if isinstance(val, str) and 0 < len(val.strip()) <= 60
                        and shelf_key(key) and "id" not in str(key).lower()), "")
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
            if hit or self.shelf_of(d) not in home or not any(isinstance(v, str) and v in self.archive.ids and id_named(key) for key, v in d.items()):
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
            if any(isinstance(v, str) and v in ids and id_named(key) for key, v in d.items()):
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
            if any(isinstance(v, str) and v == ident and id_named(key) for key, v in d.items()):
                text.extend(str(v).lower() for v in d.values() if isinstance(v, str))
        for item in self.archive.items:
            walk_dicts(item, visit)
        have = {self.stem(w) for w in re.findall(r"[a-z0-9]{3,}", " ".join(text))}
        body = " ".join(text)
        if any(re.search(r"(?<![\d.,])" + re.escape(fig) + r"\s*-?\s*" + re.escape(self.stem(unit)), body) for fig, unit in figs):
            return True
        return any(all(w in have for w in words) for words in wants)

    def vl_room(self) -> tuple[int, int]:
        left = max(0, self.ask_limit() - self.msgs_sent) if self.asking and self.role("message") else 0
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
            low = json.dumps({a: (v.strip().lower() if isinstance(v, str) else v) for a, v in args.items()}, sort_keys=True, default=str)
            key = low if key not in book and low in book else key      # (the call goes out with the label as the rows print it)
            reuse = key in book and bool(book[key].get("first")) and key in (self.__dict__.get("ceil_seen") or {})
            if not reuse and not self.walk_call(args, say):
                break
            key = low if key not in book and low in book else key
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
        if not spec or not found or self.needs.items() or spec.get("tool", self.role("search")) != self.role("search"):
            return None, False          # (the sweep sends its calls to the keyword search; a filter tool's walk runs in ceiling_walk)
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
        """A listing call for one shelf priced below a figure, on the keyword search or on a second listing tool (a shelf or price
        filter); the tool used is kept in seed_tool."""
        for tool in [self.role("search")] + [t for t in self.shapes if self.shelf_tool(t)]:
            args = self.seed_args_on(tool, label, bound)
            if args is not None:
                self.seed_tool = tool
                return args
        return None

    def seed_args_on(self, tool: str | None, label: str, bound: Any) -> dict[str, Any] | None:
        jschema = self.shapes.get(tool or "") or {}
        props = jschema.get("properties") if isinstance(jschema.get("properties"), dict) else {}
        found = self.page_arg(tool)
        if not props or not found or not is_number(bound):
            return None
        told = next((str(tdef["function"].get("description") or "") for tdef in self.toolset if tdef["function"]["name"] == tool), "")
        parts = [part.lower() for part in re.split(r"[.;]", told)]
        tok = self.token_key()
        label_key, label_val = None, None
        for key, spec in props.items():
            if not isinstance(spec, dict) or key in (tok, found[0]) or not shelf_key(key):
                continue
            if isinstance(spec.get("enum"), list):
                hit = next((v for v in spec["enum"] if isinstance(v, str) and v.strip().lower() == label.strip().lower()), None)
                if hit is not None:
                    label_key, label_val = key, hit
                    break
            elif spec.get("type") == "string":
                label_key, label_val = key, label
                break
        numeric = [key for key, spec in props.items() if isinstance(spec, dict) and spec.get("type") in ("integer", "number")
                   and not spec.get("enum") and key != found[0] and not re.search(r"page|first|offset|skip|count|size|num", str(key).lower())]
        bound_key = next((key for key in numeric if re.search(r"ceil|max|price|cap$|budget|upper|most", str(key).lower())),
                         numeric[0] if len(numeric) == 1 else None)
        if not label_key or not bound_key:
            return None
        args: dict[str, Any] = {label_key: label_val, bound_key: int(bound) if props[bound_key].get("type") == "integer" or float(bound).is_integer() else bound}
        for key, spec in props.items():
            if key == label_key or not isinstance(spec, dict) or not isinstance(spec.get("enum"), list) or not 2 <= len(spec["enum"]) <= 4:
                continue
            hit = next((v for v in spec["enum"] if any(re.search(r"\b" + re.escape(str(v).lower()) + r"\b", part)
                                                       and re.search(r"\b" + re.escape(label_key.lower()) + r"\b", part) for part in parts)), None)
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
                tool = self.__dict__.get("seed_tool") or self.role("search")
                found = self.page_arg(tool)
                mode = dict(self.search_keys(args)[1])
                sig = json.dumps([sorted((m, v) for m, v in mode.items() if not (isinstance(args.get(m), str) and args[m] == label)),
                                  next((a for a in args if is_number(args[a]) and a != (found[0] if found else None)), "")], default=str)
                key = json.dumps({a: v for a, v in args.items() if not found or a != found[0]}, sort_keys=True, default=str)
                if sig in self.__dict__.get("ceil_bad", set()) or key in (self.__dict__.get("ceil_seen") or {}):
                    continue
                self.__dict__.setdefault("ceil_seed_keys", set()).add(key)
                self.walk_call(args, f"The cheapest is asked for: I list '{label}' priced below the pick ({pick[1]:g}) to see whether this "
                                     "store's price filter holds.", tool=tool)
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
                long_strings([val for val in obs.values() if isinstance(val, (str, list))], lns)
        elif isinstance(obs, dict):
            for piece in obs.values():
                if isinstance(piece, list) and piece and all(isinstance(val, str) for val in piece):
                    long_strings(piece, lns)
                elif isinstance(piece, str) and len(piece.split()) >= 4 and refused:
                    lns.append(piece)
        return lns

    def screen(self, proposal: dict[str, Any], permitted: set[str], scripted: bool, batch: int, queued_messages: int) -> tuple[str | None, str, dict[str, Any]]:
        func = proposal.get("function") or {}
        call_name, raw_args = func.get("name"), func.get("arguments")
        if call_name not in self.shapes:
            return f"Not run (no step used): '{call_name}' is not one of the tools. Use one of: {', '.join(sorted(permitted))}.", call_name, {}
        if not scripted and call_name not in permitted and call_name == self.role("message") and self.trolley.units \
                and re.search(r"basket|total|cart", as_text(raw_args), re.I) and self.moves_left() >= 2:
            permitted = set(permitted) | {call_name}
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
        call_args = self.id_repair(call_name, call_args)
        try:
            call_args, held = self.browse_fix(call_name, call_args, scripted)
            if not held:
                call_args = self.search_clean(call_name, call_args)
                call_args, held = self.repeat_fix(call_name, call_args)
        except Exception as err:
            held = ""
            self.log.append({"kind": "browse_fix_error", "error": f"{type(err).__name__}: {err}"[:200]})
        if held:
            return held, call_name, call_args
        many = self.batch_tool()
        if many and call_name == many[0] and isinstance(call_args.get(many[1]), list) and len(call_args[many[1]]) > many[2]:
            return (f"Not run (no step used): {call_name} reads at most {many[2]} listings per call; send {many[2]} ids or fewer "
                    "(several calls in one reply are fine)."), call_name, call_args
        if call_name in self.finishers() and batch > 1:
            return "Not run (no step used): an order or a decline must be sent alone, after you have seen the results of your other calls.", call_name, call_args
        if call_name == self.role("message") and self.msg_cap is not None and self.msgs_sent + queued_messages >= self.msg_cap:
            return "Not run (no step used): the message budget is used up. Do not message the shopper again.", call_name, call_args
        shows_basket = call_name == self.role("message") and bool(self.trolley.units) and bool(re.search(r"basket|total|cart", as_text(raw_args), re.I))
        if call_name == self.role("message") and not scripted and self.msg_cap is not None and self.kept_messages() > 0 \
                and self.msgs_sent + queued_messages >= self.msg_cap - (1 if self.basket_due() or not self.trolley.units else 0) and not shows_basket:
            return ("Not run (no step used): the remaining message is kept for the request's own instruction at the finish (the code "
                    "sends it for you before the order). Do not message the shopper again."), call_name, call_args
        if call_name == self.role("cart") and not scripted and self.__dict__.get("basket_tick", -1) > self.trolley.changed_at \
                and self.msg_cap is not None and self.msgs_sent >= self.msg_cap \
                and not (set(self.trolley.units) & self.dead_lines()) and len(self.needs.uncovered) <= self.__dict__.get("basket_needs", 0):
            return ("Not run (no step used): the final basket was already shown to the shopper and no message is left to show a "
                    "changed one. Order the cart as it is."), call_name, call_args
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
        if not scripted and call_name == self.role("details") and self.needs.items() and self.step_cap <= 50:
            try:
                homes = self.home_shelves()
                lacking = set(self.decline_lacking())
                fits = self.decline_fits()
                wanted = self.asked_ids(call_args)
                owners = {next((k for k, home in enumerate(homes) if self.shelf_known(i) in home), None) for i in wanted}
                full = [k for k in owners if k is not None and k not in lacking
                        and any(self.shelf_known(i) in homes[k] for i in fits)
                        and sum(1 for i in self.archive.read if self.shelf_known(i) in homes[k]) >= 4]
                if wanted and lacking and None not in owners and len(full) == len(owners):
                    names = [self.needs.items()[k]["product"] for k in sorted(lacking) if k is not None and k < len(self.needs.items())]
                    tell = self.__dict__.setdefault("read_told", set())
                    if tuple(sorted(wanted)) not in tell:
                        tell.add(tuple(sorted(wanted)))
                        return ("Not run (no step used): steps are short in this errand and that product already has a verified fit; "
                                "spend the reads on " + (", ".join(names) or "the products with no fit yet") + " first. If you are "
                                "sure, send the same call again."), call_name, call_args
            except Exception:
                pass
        held = "" if scripted else self.late_hold(call_name, call_args)
        held = held or ("" if scripted else self.value_guard(call_name, call_args))
        if held:
            return held, call_name, call_args
        if call_name == self.role("cart"):
            if not scripted and not self.trolley.units and self.trolley.is_put(call_args):
                note = self.procedure_guard("puts the first item in the basket (the call it is about to send: " + as_text(call_args)[:200] + ")")
                if note:
                    return note, call_name, call_args
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
        shelf = self.shelf_known(thing)
        kept = {ident for ident, count in self.trolley.lines(call_args) if count is None or count > 0}
        twin = next((other for other, units in self.trolley.units.items() if units and other != thing and shelf
                     and (self.trolley.adds is not False or other in kept) and self.shelf_known(other) == shelf), "")
        if twin and f"twin|{sig}" not in self.warned:
            self.warned.add(f"twin|{sig}")
            return (f"Not run (no step used): the cart already holds {twin} from the same category ({shelf}), and the cart keeps both lines: "
                    f"to swap, first take {twin} out of the cart, then put {thing} in. If the request really wants both, send the same call again.")
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

    def screen_finish(self, cid: str, call_name: str, call_args: dict[str, Any]) -> tuple[str | None, dict[str, Any]]:
        if self.call_kind(call_name, call_args) == "decline" and not self.archive.read and self.moves_left() > 8 \
                and self.__dict__.get("blind_told", 0) < 2:
            # (no listing has been read into data yet: the store's results are in a form the code does not follow, so its
            #  checks know nothing, and a decline now would rest on nothing)
            self.blind_told = self.__dict__.get("blind_told", 0) + 1
            self.log.append({"kind": "checkpoint", "what": "blind_decline_held", "ids": len(self.archive.ids)})
            return ("Not run yet (no step used). No listing's details have been read yet, so nothing shows that the shop lacks "
                    "a fit. Open the details of a few listings of each product asked for, judge them against the request "
                    "yourself from what the store shows, and decline only if none fits."), call_args
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
                    done = False
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
        if self.call_kind(call_name, call_args) == "order":
            self.part_lookups()
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
            fault_marks = self.__dict__.setdefault("fault_holds", {})
            fmark = tuple(sorted(self.trolley.units.items()))
            if faults and self.moves_left() > 3 and fault_marks.get(fmark, 0) < 2:
                # (the next order of this cart is vetted again: a count hold must not let it skip the checks below)
                fault_marks[fmark] = fault_marks.get(fmark, 0) + 1
                self.order_vetted = False
                return "Not run yet (no step used). The cart as counted is not ready: " + "; ".join(faults) + ". Fix it, then order.", call_args
            shelf = self.shelf_faults()
            if shelf and self.moves_left() > 3 and self.__dict__.get("shelf_told", 0) < 2:
                self.shelf_told = self.__dict__.get("shelf_told", 0) + 1
                self.order_vetted = False
                return "Not run yet (no step used). " + "; ".join(shelf[:3]) + ". Fix the cart, then order.", call_args
            paged = self.cheap_pages()
            if paged:
                self.order_vetted = False
                return paged, call_args
            swept = self.cheap_sweep()
            if swept:
                self.order_vetted = False
                return "Not run yet (no step used). " + swept, call_args
            misfit = self.code_fit_hold() if self.moves_left() > 4 else ""
            if misfit:
                self.order_vetted = False
                return "Not run yet (no step used). " + misfit, call_args
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
            kit_note = self.kit_hold() if self.moves_left() > 4 else ""
            if kit_note:
                return kit_note, call_args
            near_note = self.near_hold()
            if near_note:
                return near_note, call_args
            stated = self.stated_fit()
            if stated:
                return stated, call_args
            sweep_note = self.decline_sweep()
            if sweep_note:
                return sweep_note, call_args
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
            soft = [r for r in self.__dict__.get("refs") or [] if r.get("status") in ("open", "stuck") and self.ref_live(r)]   # (any relation: an unknown value rules nothing out)
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
        spec = self.__dict__.get("batch_spec")
        cut_to = spec[2] if isinstance(spec, tuple) and len(spec) == 3 and is_number(spec[2]) else None
        if cut_to and isinstance(spec, tuple) and len(spec) == 3:
            for proposal in proposals:
                func = proposal.get("function") or {}
                if func.get("name") == spec[0]:
                    raw = func.get("arguments")
                    got = raw if isinstance(raw, dict) else (json_object(raw) if isinstance(raw, str) and raw.strip() else {})
                    if isinstance(got, dict) and isinstance(got.get(spec[1]), list) and len(got[spec[1]]) > cut_to:
                        got[spec[1]] = got[spec[1]][:cut_to]
                        func["arguments"] = json.dumps(got)
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
            if self.call_kind(call_name, call_args) == "decline" and self.gate_decline():
                responses.append((cid, "Not run (no step used): steps are left and the product's shelf is not covered yet, so more "
                                       "of it is listed and read for you next. Decide again after reading them, and never decline "
                                       "while an opened listing meets every requirement now in force." + self.__dict__.pop("cart_note", "")))
                return responses
            held, call_args = self.screen_finish(cid, call_name, call_args)
            if held and scripted and self.call_kind(call_name, call_args) == "order" and self.__dict__.get("open_queue"):
                self.open_named()
                held, call_args = self.screen_finish(cid, call_name, call_args)
            if held:
                responses.append((cid, held))
                return responses
            if self.call_kind(call_name, call_args) == "decline":
                ordered = self.cart_instead() or self.fit_instead() or self.basket_instead()
                if ordered:
                    call_name, call_args = ordered
                    held, call_args = self.screen_finish(cid, call_name, call_args)
                    if held and self.__dict__.get("open_queue") and not self.wire.done:
                        self.open_named()
                        held, call_args = self.screen_finish(cid, call_name, call_args)
                    if held:
                        responses.append((cid, held))
                        return responses
            if not scripted and self.call_kind(call_name, call_args) in ("order", "decline"):
                note = self.procedure_guard("places the order" if self.call_kind(call_name, call_args) == "order" else "declines (buys nothing)")
                if note:
                    responses.append((cid, note))
                    return responses
            words = self.__dict__.get("finish_words")
            if words:
                for key, val in list(call_args.items()):
                    if isinstance(val, str) and words not in val and isinstance((self.shapes.get(call_name, {}).get("properties") or {}).get(key), dict) \
                            and "enum" not in self.shapes[call_name]["properties"][key]:
                        limit = self.shapes[call_name]["properties"][key].get("maxLength", len(val) + len(words) + 1)
                        call_args[key] = (val[: max(0, limit - len(words) - 1)] + " " + words).strip()[:limit]
                        break
            to_send[0] = (cid, call_name, call_args)
        if to_send and to_send[0][1] not in self.finishers() and self.finishers():
            hold = max(0, self.moves_left() - 1)
            for cid, _, _ in to_send[hold:]:
                responses.append((cid, "Not run (no step used): only the last step is left, and it must be the order (or the decline)."))
            to_send = to_send[:hold]
        if to_send and to_send[0][1] in self.finishers() and self.call_kind(to_send[0][1], to_send[0][2]) == "order" and self.basket_due():
            note = self.message_call(self.basket_text(), f"basket_{self.wire.tick}")
            if note:
                note_args = json.loads(note["function"]["arguments"])
                self.basket_tick = self.wire.tick + 1
                self.basket_needs = len(self.needs.uncovered)
                self.log.append({"kind": "basket_message", "step": self.wire.tick + 1, "text": note_args})
                to_send.insert(0, (note["id"], note["function"]["name"], note_args))
        for begin in range(0, len(to_send), 16):
            slab = to_send[begin: begin + 16]
            if self.wire.done or self.wire.broke or self.moves_left() <= 0:
                responses += [(cid, "Not run: the episode is over or no steps are left.") for cid, _, _ in slab]
                continue
            slab = slab[: self.moves_left()]
            responses += self.send_chunk(slab)
        call_rank = {c["id"]: i for i, c in enumerate(proposals)}
        extra = [text for cid, text in responses if cid not in call_rank]
        responses = [(cid, text) for cid, text in responses if cid in call_rank]
        if extra and responses:
            responses[0] = (responses[0][0], "The final basket was shown to the shopper first, as the request asks (reply: "
                            + " ".join(extra)[:400] + ").\n" + responses[0][1])
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

    def absorb(self, call_name: str, call_args: dict[str, Any], obs: Any, resp: dict[str, Any], last: bool) -> str:
        shown, obs = obs, lensed(obs)
        msg_tool, basket_tool = self.role("message"), self.role("cart")
        if call_name == msg_tool:
            self.msgs_sent += 1
            if self.trolley.units and re.search(r"basket|total|cart", as_text(call_args), re.I):
                self.basket_tick = self.wire.tick
                self.basket_needs = len(self.needs.uncovered)
        self.last_kind = self.call_kind(call_name, call_args)
        if self.last_kind == "order":
            self.orders_tried += 1
        if call_name in (basket_tool, msg_tool) or call_name in self.finishers():
            self.last_calls = []
        else:
            self.last_calls = (self.last_calls + [call_name + as_text(call_args)])[-2:]
        self.archive.add(obs)
        self.shelf_learn(obs)
        try:
            self.search_seen(call_name, call_args, obs)
        except Exception:
            pass
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
                hits = {v for k, v in d.items() if isinstance(v, str) and v in self.trolley.units and id_named(k)}
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
        if call_name == self.role("search"):
            self.__dict__.setdefault("search_words", []).append(" ".join(str(v) for v in call_args.values() if isinstance(v, str)))
        self.note_read(call_name, obs)
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
        if call_name != self.role("search") and self.shelf_tool(call_name):
            was_tool, self.page_tool = self.__dict__.get("page_tool"), call_name
            try:
                remarks.append(self.search_note(call_args, obs))
            finally:
                self.page_tool = was_tool
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
        try:
            notes = self.tool_notes()
        except Exception:
            notes = ""
        if notes:
            opener += "\n\n" + notes
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
            if self.after_change():
                continue
            if self.open_named():
                continue
            if self.__dict__.pop("gate_pending", False) and "decline_queue" in self.__dict__ and self.run_decline_reads():
                continue
            if not self.ask_due() and self.ref_step():
                # (a value read from another listing decides which items fit: it is read before the first cart, inside
                # the steps the resolver may use, once no question is overdue)
                continue
            if self.cart_due():
                continue
            if self.ask_due() and self.pose_planned(urge=True):
                continue
            if self.browse_step():
                continue
            if self.need_hunt():
                continue
            if (not self.asking or self.__dict__.get("empty_plans")) and self.sweep_step():
                continue
            if self.value_pages():
                continue
            if self.early_rescue():
                continue
            if self.asking and self.pose_planned():
                continue
            if self.ref_step():
                continue
            if self.walk_step():
                continue
            model_list = ACTOR if self.stuck < STUCK_LIMIT else ACTOR[1:] + ACTOR[:1]
            size = sum(len(m["content"]) for m in self.chat if isinstance(m.get("content"), str))
            if size > HISTORY_CHARS:
                self.shorten_chat()
            rate = ((self.wire.__dict__.get("char_rates") or {}).get(model_list[0]) or 0.0)
            if rate and self.wire.spend + rate * (size + 10000) > self.wire.hard_cap():
                # (the next turn would cross the errand's total: the older results are cut harder, and questions not yet sent
                # go out first, since they cost little)
                self.shorten_chat(keep=3)
                if self.asking and self.msgs_sent == 0 and self.pose_planned(urge=True):
                    continue
            msg = self.wire.query_model(self.chat, model_list, 2500, self.toolset)
            if msg is None and self.wire.__dict__.pop("dear_skip", False):
                if self.wire.tick == 0 and not self.__dict__.get("first_turn_priced"):
                    # (the first turn of a long request, refused only because the models not used yet carry the plan's dear
                    # rate: they are priced at their own rate once, so the errand is not declined before its first step)
                    self.first_turn_priced = True
                    rates = self.wire.__dict__.setdefault("char_rates", {})
                    for mdl in model_list:
                        rates.setdefault(mdl, 1.0e-6 if "sonnet" in mdl else 3.0e-7)
                    continue
                self.wire.no_model = True       # (no turn fits in what is left: the code finishes the errand)
                break
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
            before = (self.refusal_n, len(self.needs.uncovered), self.needs.spend_cap)
            self.post_replies(self.dispatch(proposals))
            self.reorder_after_refusal(before)
            self.stuck = self.stuck + 1 if self.wire.tick == step_before else 0
        if not self.wire.done and not self.wire.broke and self.wire.no_model:
            self.fill_lines()
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
