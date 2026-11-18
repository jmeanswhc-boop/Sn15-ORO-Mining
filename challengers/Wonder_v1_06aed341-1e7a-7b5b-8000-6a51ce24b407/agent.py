from __future__ import annotations
import json
import re
from time import monotonic,sleep
from typing import Any
from src.agent.proxy_client import ProxyClient
_sma_HAIKU,_sma_SONNET,_sma_LUNA="anthropic/claude-haiku-5.5","anthropic/claude-sonnet-5.5","openai/gpt-6-luna"
ACTOR=[_sma_HAIKU,_sma_LUNA,_sma_HAIKU]
READER=[_sma_HAIKU,_sma_SONNET,_sma_LUNA]
READ_TOKENS=8000
CHECK_TOKENS=6000
PLANNER,LISTENER,CHECKER,AUDITOR,FINISHER=([_sma_HAIKU,_sma_SONNET]for _ in range(5))
ASKER=[_sma_HAIKU,_sma_SONNET]
DEADLINE,CLOSING,LAST_CALL,CLOCK_STEP=265.0,70.0,28.0,30
PACE_GAP,SLOW_GAP,SLOW_FOR,DOWN_AFTER=0.15,1.0,30.0,45.0
MODEL_TRIES,WORLD_TRIES=2,5
SOFT_SPEND,SPEND_CAP,HARD_SPEND=0.03,0.045,0.10
TURN_LIMIT,IDLE_LIMIT,STUCK_LIMIT=80,3,4
FINISH_STEPS,FINISH_TURNS,CODE_FINISHES=2,2,4
FIRST_ORDER_BY,FIRST_ORDER_DUE=7,10
ASK_DEFAULT,ASK_AFTER_SETS,ASK_GRACE_STEPS,ASK_AFTER_STEPS=3,2,3,5
SOLO_SETS=1
VIEW_CHARS,SEARCH_TEXT,RECORD_TEXT=20000,600,2500
CH_SYNONYM_MIN=4
CODE_PAGES,CODE_READS,SWEEP_READS=3,8,12
SWEEP_PAGES=3
SOLO_PAGES=3
FEW_PAGES,FEW_MAX=1,3
REVEAL_PAGES,REVEAL_OPENS,REVEAL_MAX=2,4,2
REACH_ROWS=24
CAP_ROWS,CAP_CALLS=12,6
FEW_ROWS,FEW_KEEP,PAIR_OPENS,PAIR_ANCHORS,PAIR_ALTS=16,6,3,3,2
LIST_READ_CAP=5
LIST_FAIL_CAP=2
SHOW_RULES_IN_FINISH=True
LOST_REASKS=1
WALK_PAGES,WALK_TOTAL=6,8
RUNG_COUNT,RUNG_PAGES,RUNG_CALLS,RUNG_OPENS=4,3,6,4
VALUE_PAGES,VALUE_TOTAL,VALUE_TOTAL_PER=4,8,5
LABEL_TRIES,LABEL_OPENS=4,3
CEIL_CALLS,CEIL_OPENS=4,3
FIT_TRIES,FIT_OPENS=3,4
WIDE_TRIES,WIDE_OPENS,WIDE_KEEP=3,2,6
JOINERS=r"\s+and\s+|\s*&\s*|\s*/\s*|\s+\+\s+"
FAULT_KEYS=r"error|fault|refus|reject|denied|invalid|fail"
PUT_WORDS={"add","put","hold","set","insert","place","keep"}
DROP_WORDS={"remove","drop","release","delete","take","out","clear"}
_SMA_JOINERS={"and","or","not","no","all","any"}
_SMA_PLACERS={"in","on","to","by","from","at"}
_SMA_POINTERS={"the","a","an","this","that"}
_SMA_OWNERS={"it","its","your","our","you"}
LINK_WORDS=_SMA_JOINERS|_SMA_PLACERS|_SMA_POINTERS|_SMA_OWNERS|{"as","of","for","with","is","are","be"}
ACT_RULES="""GRADING. You shop on behalf of a single person inside a simulated store, and the tools supplied to you are your only means of acting. Each run earns 1 or 0. It earns 1 only if the finish is flawless: what gets ordered satisfies every condition the shopper wrote, every hidden need they disclose once asked, and every change they announce along the way; quantity and budget are correct; when the request wants the cheapest (or best) one, that is what you chose; and each procedural rule the request sets was obeyed. If no item in the store satisfies everything, then the correct ending is a decline, carried out precisely the way the request describes.

EACH REPLY: include one tool call or more. Any text stays brief: today's conditions, the verified candidates with price and stock, and what you do next.

A. DECIDING WHETHER A LISTING QUALIFIES (your grade depends on it)
A1. The grader settles a condition by looking at the field for its property and comparing the wanted value with the words this store writes there. Field names are matched by meaning, not spelling: separators, prefixes, longer or shorter forms, and familiar alternative names all count (a pants fit type may be called "fit", a collar type "neckline", a watch dial size "dial size", a colour "color family"). A field named exactly after the property is decisive, and so is a field that adds one word in front of that name (length -> pants length, material -> clothing material); the value showing in either satisfies the condition. size class and size_class, differing only by separators, count as one field, satisfied when either spelling holds the value. Without such fields the alternative-name field decides, and if no field names the property whatsoever, the value may come from the options list or the title. Descriptions are never evidence.
A2. Comparing words: case, separators, singular versus plural and spelling variants do not matter. A more specific value that contains the wanted one passes (cotton <- "80% cotton", green <- "forest green", rtx 3060 <- "rtx 3060 12gb"), and in a list one element is enough. Synonyms, abbreviations, related words and shades (technical wood ~ "tech wood", windows 11 ~ "win11", embroidery ~ "embroidered", plain ~ "solid", off-white ~ ivory or cream) pass only while no listing of the category writes the wanted value itself in that field; once other listings show "matte" for finish, a "satin" finish fails matte. A plus-joined value ("wool + nylon") fails one of its parts when the same field also turns up with the other part alone ("nylon"). Wherever FIELD EVIDENCE lines appear, code derived them from this store's records, and they settle which case holds. These fail: another value of that property (quartz is not mechanical, rosewood is not maple); a word that only resembles the wanted one; a catch-all that claims every value ("suitable for all occasions", "all seasons", "universal") and so names none, though a list that also names the wanted value still passes; a value that merely tends to accompany or imply the wanted one (a "pullover" style is not "zip-up", "slip-on" is not "lace-up"). The only stand-ins accepted are a shade commonly sold as the wanted colour and a more specific value that contains it.
A3. Fields of some other property never stand in for a property whose own field the category's listings carry (lining material is not material, length is not style, strap length is not strap type).
A4. Conditions come only from what the shopper stated or disclosed. Do not drop a listing over anything unasked, such as an age group, gender, size, occasion, condition, quality or use you infer from the story or the description. Treat the story as context, and when it seems to matter, ask the shopper about that property instead of assuming. Questions you yourself ask create no conditions; only the shopper's answers do.
A5. Figures: convert units, then compare amounts (45 mm equals 4.5 cm). Edges are inside a limit, and a range passes only when every part of it does. If the title is the evidence because no field names the property, an exact wanted value passes when the title shows its words and figures in the same order, extra detail after them allowed (9' x 12' 4" shows 9' x 12'; 9' x 13' does not).
A6. Numbers in a title ("55 x 94 Inch", "117 x 228 cm") do not say which dimension each one is, so they never justify rejecting a listing or not opening it; read the fields in its details instead.
A7. Exclusions ("leave out anything with X") and closed lists ("A or B and nothing else") are binding.
A8. Category alone decides whether a listing is the requested kind. A listing in the requested category IS that kind as far as grading goes, even when its title or description names a different object (a container, a case, a refill, a set, an accessory); never pass over, reject or demote it for what it looks like, and judge only conditions, price and stock. No property needs to repeat the category name. When the store labels its categories, the item belongs in the category whose label fits the wording of the whole request (asking for "A or B" means a category "A & B", not the narrower neighbour "B").
A9. When the request wants items on hand, an item counts only while its stock shows it available; its price multiplied by the quantity has to stay inside the budget, and equal is fine. Prices are in the budget's unit unless the record names another (45 against a budget of 1000 US cents means 45 cents); rescale neither number.
A9b. Declining is rarely correct. If nothing matches the literal wording but some listing satisfies each condition under these rules (a more specific value, the same field under another name, or a synonym or shade where no listing of the category writes the value itself in that field), that listing is your fit. Decline only if the search truly found nothing that satisfies every condition under these rules.

B. CHEAPEST, OR BEST BY A MEASURE
B1. Cheapest: you must order the lowest price among ALL qualifying items; finding one qualifying item does not settle it.
 - If keyword search needs every word and also hits property values, then a value-plus-kind search lists the candidates holding that value: go through all its pages and open them in order of price.
 - A category view with a price limit reveals every price in the category but none of the values: use it to see which cheaper rows exist (lower the limit until the total fits), then check what properties those rows have.
 - Run each wanted value with the item kind as a short query, the value leading ("blue chair", "30m watch"), and read all of its pages when the total fits on the readable ones (if not, add a second wanted value or use a narrower word for the kind). Repeat with other names of the kind ("stool", "bench") and with the bare value.
 - When a value cannot be searched (a figure the store strips), also search the product line of every fit or near-fit: its maker plus one distinctive title word, two words in all, every page. Other sizes and colours of a line are separate listings with the same property values, and they may be cheaper.
 - Rank the visible candidates by price yourself and open details from the cheapest up. The answer is the first whose property fields meet everything, provided every cheaper one was opened and rejected; hold the next qualifying ones in reserve.
B2. Best by a measure (most, largest, longest, highest...): line up ALL qualifying items on that measure the same way and choose the extreme. An item with no value for the measure cannot win.

C. ENDING THE ERRAND
C1. Maintain a ranked list of verified fits that includes at least one reserve, because the first choice may sell out at checkout.
C2. Ahead of each order attempt: the cart contains just one line per requested product (one line if only one product is wanted), each the chosen item at its required quantity, and nothing beyond that; and after your latest cart change you have re-read the chosen item's stock and price and seen it in stock and inside the budget.
C3. A decline comes only after a thorough search turned up nothing that meets every condition, and it follows the request's decline rule to the letter. If the request forbids offering substitutes, the decline remark names, describes or points to no listing at all: no ids, titles or prices.
C4. Spend steps wisely: the first order attempt goes out while seven or more steps are left, so that a refusal can still be handled. At 7 steps left, the cart, stock-check and finishing tools are all that remain. Make the decision once rather than switching the cart back and forth.

D. WHEN THINGS CHANGE
D1. The shopper may alter the errand: a different budget (larger or smaller), a condition removed, swapped or added, a price increase, an item running out. One line may hold several such changes, and those lines may show up in any tool result (an extra field containing sentences) or together with an order attempt. From that moment each change applies: check the pick and the reserves again, and if the cheapest or best is wanted, choose anew from the updated acceptable set.
D2. An order attempt may be REFUSED. The episode then continues with the cart unchanged. Read the refusal, apply all the changes it names, take out of the cart anything that stopped fitting (for instance a sold-out item), add the next verified fit, check its stock and price again, and order once more.
D3. Stock and prices can also shift silently, so both get checked again right before every order attempt; an item whose current price multiplied by the quantity exceeds the current budget no longer fits.

E. HIDDEN NEEDS AND THE SHOPPER
E1. Some of the shopper's needs were never written down, and each comes out only when a message asks about its property. Answers are most dependable for whatever a message asks FIRST, so a message covers one property (two at most, the likelier one first) through a short, plain, friendly question that names the property; giving one or two example values is fine. Leave out instructions, lists, labels, positions and rules.
E2. Replies like "Don't mind", "no preference" or "anything is fine" leave that property open. Any preference or limit the shopper states binds from then on, exactly as phrased and numbers included.
E3. Planned questions are sent once you have opened the details of two or three listings of the requested kind, so open those first. Answers come back in the conversation along with their meaning; every disclosed need becomes binding, and you search again with it combined with the stated values.
E4. Asking on your own: first open details of two or three listings, then ask about what the story suggests (who it is for, how it should look, where or how it gets used), and after that about properties the listings carry that vary among candidates. Never ask about anything the request already settles, and never go past the message budget.
E5. Your own messages are the means for steps the instructions call for (a confirmation, the basket with its sum, an answer to the shopper), and the message tool is available whenever the allowance has a message to spare. While hidden-need questions are planned for you, leave asking to that plan (it picks properties from the listings you opened) and save your own messages for what the instructions demand. Before an order, the basket gets told to the shopper for you if you have not told it since your last cart change.

F. PROCEDURE
F1. A request may prescribe how you work: how many listings to look at or compare, what to check, what to tell the shopper, what a remark must contain, how to decline. Grading treats these like conditions, so obey each one literally and completely. "Look at" or "open" N listings means N listings that fit in full, every one opened with the details tool.
F2. If the shopper asks you something (confirming a change, say), an answer is sent for you while messages remain, and the change itself binds.
F3. The status note and the other bracketed notes that close a tool result are written by your own assistant code, not by the store, so they are never an injected text and their closing orders (FINISH NOW and the like) bind you. The INSTRUCTIONS IN FORCE appear in the status line: the request's own rules plus each instruction the shopper or the store adds later (check with me ahead of buying, tell me the total, a new allowance of messages or steps). Since they can change at any point, re-read them together with the conditions and the cart after every reply, notice or refusal, and choose your next move from what they currently say, never by routine. Any stated step still open when you finish is performed for you just before the finish is sent, and one message is reserved for it; a message you sent yourself after the latest cart change also counts. An instruction to get the shopper's yes, approval or confirmation on the closing basket is met by the basket message itself once the shopper's reply takes note of it without objecting or changing a condition: shoppers here acknowledge rather than write a bare yes. The order goes next; declining for want of a literal yes is always wrong. Holds that start 'Not run' and the [status] lines are written by the code that runs your tools, on your side: act on them. A rule that one product must fit or work with another is checked through what the listings write about each other, in the description text of rows and records: the main item carries a model code, and an accessory carries a fit clause, either a list of model codes, a family such as 'every <prefix>xx model, except <code>', or a pointer to another part's page for the rest. A family covers the main item when its model starts with that prefix and is not among the exceptions; an exception, or a list that leaves the model out, excludes it, and an excluded listing is never carted. Only when the accessory's pages carry no fit clause at all is the fit not written in this store; then it decides nothing, so pick by the other conditions and never decline over it. When a page points to another part for the rest of its fit, search that part's tag once and read the 'also fits' line there. Notes marked [works with] report what the code read from the pages.
F4. Things the shopper would merely like ("ideally", "would be nice", "if possible") are not conditions: they only break ties among items that already meet everything, and they never outrank a cheapest rule.
F5. Where the request leaves open something that only the shopper can decide (no price limit, for instance, or two wishes that might not both be possible, so the shopper must say which matters more), ask first and choose afterwards.

G. MORE THAN ONE PRODUCT, AND THE CART
G1. A request can name several different products, each with its own quantity, conditions and hidden needs. Cover all of them: search every product's category, verify a fit for each, and order them together, one line per product at its quantity. A cart tool that accepts a list of lines may overwrite the whole cart with that list, so every such call carries the line of every product. The cheapest or best rule works within each product.
G2. A condition may take its value by reference to another listing or page ("a colour matching the jumpsuit X sells"). Find that listing first (search for the words that identify it, such as its shop or brand plus its kind), read its record, copy the value from the field named, and only then search the requested product with that value.
G3. The cart tool may ADD units to a line already present rather than set the count: one more of a held line makes 2. Never add again a line the cart already contains; change units by taking out or putting in the difference. Following each cart call, a note reports what the code counts in the cart.

H. TOOLS, STEPS AND SEARCH
H1. Each tool call costs a step, failed and invalid calls included, and the number of steps is limited; thinking costs nothing.
H2. A reply may hold several tool calls if none needs the output of another (opening the details of several listings, for example), and they execute in order. An order or a decline is always sent alone.
H3. When the details tool takes a LIST of listing ids, it shows all of their full records in ONE step (up to its item limit); use it whenever two or more listings are to be opened. If such a call comes back with an error, the list was too long for this store: resend it with two ids and keep to that limit afterwards (the status line reports the learned limit).
H4. Tool names, argument names and result shapes differ in every store. Read each tool's description and JSON schema and use exactly the names and allowed values stated there. Learn from the first results which tool shows which facts (price, stock, detailed properties). When one tool shows a listing's description and another its property table, the model code and the fit clause sit in the description, which the search rows already show; open the property tool only for shortlisted rows. A cart tool's integer argument is the number of units held on that line whatever it is called; a repeated identical cart call changes nothing.
H5. Learn early which category labels this store puts on listings of the requested kind. Where neighbouring labels exist, pick the label closest to the request's wording as a whole, then search and buy inside it.
H6. Search works differently per store, so learn it from the first results. Rows can come best match first or in a fixed id order. Keyword search can require every word (one more word shrinks the total) or take any word (one more word grows the total, and listings hitting more words rank higher). Perhaps only a few rows or pages are visible. A filter may function in one mode only: keyword mode can ignore category or price filters, and filter mode can ignore the words.
H7. Hold any reported total against the rows you can see: when it is bigger, the fitting or cheapest listing may lie on a later page or past the visible ones. Under every-word search, narrow with more exact words (or category and price filters with a lower price limit) until the total fits on the visible pages.
H8. Keep queries to one, two or three words, most distinctive FIRST. Some stores read just the first word or two and ignore the remainder, so a long query may only search its opening words; check a longer query's total and rows against a shorter one that begins the same way (the notes report what that shows).
H9. Under every-word search, a wanted value with the item kind, value leading ("blue chair"), returns exactly the listings that carry it. Under any-word search, common words (the item kind, a unit) hit most of the shelf and hide everything else: search a wanted value's own distinctive word or figure alone ("herringbone", "30m", "0.4") to get only the listings that carry it, plus the other spellings listings use ("30 m", "3 atm"); read every page of such a search when its total fits on them, and add the item kind to the values only to bring the closest matches up front.
H10. A keyword search may also hit category names and property values. Notes after each search explain what the totals so far reveal about this store's search."""
READ_REQUEST="""OUTPUT: exactly one JSON object written compactly on a single line (no indentation), with no other text. It carries twenty keys: roles, message_budget, step_budget, price_cap, currency, in_stock_only, priority, priority_detail, rules, instructions, decline_no_names, soft_wants, ask_first, hints, references, reference_targets, product, quantity, requirements, items. The input is the shopper's request together with the store's tool list, both further down.

PART 1 - TOOLS
"roles" = an object that maps each role to the exact name of a listed tool, or to null where none fits; the roles are "search", "details", "view", "message", "cart", "cart_view", "order", "decline". Here "message" means writing to the shopper, "cart" means putting items into or taking them out of the cart, "order" means buying or checking out, and "decline" means closing the errand with no purchase.

PART 2 - LIMITS AND RANKING
"message_budget" = integer count of messages the shopper permits you to send; null if the request is silent on it.
"step_budget" = integer count of moves or steps permitted by the request; null if it is silent.
"price_cap" = the largest total price, as the very number written in the request and in the unit it uses, never converted ("3050 US cents" gives 3050); null if no cap appears.
"currency" = the unit the request prices in, or null.
"in_stock_only" = true if the request wants only things that are in stock or on hand.
"priority" = one of three strings. Use "cheapest" solely if the shopper unmistakably asks for the lowest-priced (cheapest) fitting item or for the lowest total on the order, and "best" solely if they unmistakably ask for the top item by some measure; every other request gets "none". A shopper who says money is tight or every penny counts has not asked for the cheapest: the remark just explains the cap.
"priority_detail" = for "best", the measure in question; for anything else, "".

PART 3 - HOW YOU MUST PROCEED
"rules" = list of the request's procedural rules for the assistant, each copied verbatim: limits on how many listings get looked at, what gets checked or told, the way to decline, prohibitions, things to include. Plans the shopper has for their own part are story and stay out.
"instructions" = those rules once more, now as objects (one per instruction, [] when there are none, and also covering a rule that only tells how to finish or what to say at the finish): {"text": up to 20 words in the wording of the request, "before_finish": set true if this must happen no later than the finish, i.e. before the order or decline goes (first confirming with the shopper, reporting the basket contents or the total, a closing remark, a final check), else false, "needs_message": set true if carrying it out takes a message to the shopper, else false, "before_change": set true if the assistant first has to check with the shopper before it changes, replaces or removes anything in the cart, else false}.
"decline_no_names" = true when the request's rule for declining forbids offering, naming or suggesting substitutes or alternatives; false when it wants something named; null when there is no such rule.

PART 4 - STORY, WISHES AND OPEN POINTS
"hints" = list of "property (product): the detail in a few words" strings ("property: detail" with a single product), one per story detail that aims at a property without saying its value: what the item gets used for or with, where, by whom, what it must fit, match or survive. Name the property the way listings would most likely name it. If a detail could point at two properties, give the likelier one, a slash, then the other ("colour / style (chair): must suit a room with one painted wall"): a room, a wall, an outfit or companion things the item must match, suit or sit well with, and its appearance to the eye, all point at colour first and at pattern or style next. Omit a detail whose only job is to justify a stated condition (why the stated material or size). Never hints, only background: weather, a job, a schedule, travel, a place where the item is not worn or used, and any purpose shared by every listing of that kind (storage for a dresser, warmth for a coat). [] if none.
"soft_wants" = what the shopper would like but does not strictly need (ideally, preferably, would be nice, if possible); these never go into requirements.
"ask_first" = whatever only the shopper can answer and should be asked about before a choice is made, e.g. the budget when no limit is set, which of two wishes counts more when the request admits both may not be possible (in that case both wishes go into soft_wants and not into requirements), or anything the request says to check with the shopper before the choice is made. A check that needs the chosen basket (approving it, its total, the order itself) is a finishing step: it goes into instructions with before_finish, never here. [] if none.

PART 5 - VALUES BORROWED FROM ANOTHER LISTING
"references" = list of conditions whose value comes only from a pointer to ANOTHER listing or page, e.g. "a colour matching the jumpsuit X sells", "within 5 of what the page for Y states", "a colour other than the one on Z's page". Format: "property (product): as shown on <how the request names that listing>"; when the value has to be unlike the pointed listing's, "property (product): different from the one shown on <how the request names that listing>"; when it must be that listing's counterpart (its opposite number, the value the store pairs with it, or any value that has to suit, get along with or be compatible with that listing's rather than equal it), "property (product): counterpart of what is shown on <how the request names that listing>". Put every entry into requirements as well, in this exact form. [] if none.
"reference_targets" = list, parallel to references, describing each listing pointed at: {"property": the property concerned, "product": the product concerned, "seller": shop or brand of that listing, copied exactly from the request, "kind": that listing's sort of product, "details": [short strings in the request's words for any further values it gives that listing ("blue", "4gb", "net weight 167.262186 g")], "relation": "same", "pairs" (our value is the one the store pairs with that listing's value: its counterpart or opposite number, or a value that has to suit or get along with it rather than equal it; neither the same value nor merely any other), "differ" (must NOT equal that listing's value: set apart from, different from, other than), "within", "at most" or "at least" - how our value relates to that listing's, "tolerance": number belonging to "within", else null, "unit": its unit, else ""}. [] if none.

PART 6 - THE PRODUCTS AND THEIR CONDITIONS
"product" = what kind of product is wanted, in the wording of the request; for more than one product, all of them linked with "and".
"quantity" = integer number of units of a single listing to order, defaulting to 1. Something sold as one piece, like a pair of shoes or a set, is 1. Only an outright request for several units changes it ("give me three of them", "order 2 packs"); the number of people who will share or use it, or of old ones there used to be, is story and leaves it at 1.
"requirements" = list of strings, one per firm condition on the product, each "property: constraint" with the shopper's exact value, limit, exclusion or closed list; for several products, include all of them, each written "property (product): constraint".
 * What a condition is: a product property plus the value the request insists on ("the pattern has to be herringbone", "with a mechanical movement", "at least 4 drawers", "the occasion has to be party"). Its value must be words or figures the request itself writes as the item's value. A count of things inside one sold pack, box or set is its pack size, valued with the figure and the noun the request writes after it ("a tin holding 40 tea bags" gives "pack size: 40 tea bags", never "tea bags per tin: 40").
 * Not conditions: price, stock, quantity, budgets and mere wishes; a restatement of the product kind or category (words that form part of the category name, such as a season, an occasion or an audience in it, are covered by the category already). Story is never one: the person it is for, an event where it will be worn or used ("to wear at his wedding", "for the staff party"), where, when or how it will be used, how many share it, or the fact that it replaces an old one ("a new watch" asks for a replacement, not a condition). An occasion, use or audience counts only when the request makes it the item's own property ("the occasion has to be business", "a blazer made for business"). A quality, reason or wish found in the story (natural and breathable, slim enough for a case, works with what I already have, covers the whole drop) is a hint, never a condition. A saying or comparison that names no property and no value of its own (do it right, like a craftsman would; sturdy as an old oak) only sets the tone: it is neither a condition nor a hint.
"items" = per DIFFERENT product the request wants bought, in request order: {"product": that product's kind as the request words it, "quantity": units of one listing of it (counted like quantity), "requirements": its firm conditions (written like requirements), "depends_on": 1-based position of the other entry this product must work with, fit, suit or be compatible with by the request's words, null otherwise; a condition that switches on only when the other product turns out a certain way (written as a requirement with "if") ties nothing and leaves null}. With a single product there is one entry.

LAST CHECK: no sentence of the request is left unplaced; each is a requirement, the priority, a soft want, a rule, a budget, a hint, or plain story. A sentence that might be a rule goes into rules.
"""
HEAR="""OUTPUT: a single JSON object and no other text. A shopping assistant needs the shopper's latest line(s) decoded; the object you return does that through twelve keys, defined in part B. Part A says how to read the shopper.

PART A - READING THE SHOPPER
A1. Report each change a line holds; one line may hold several.
A2. Treat folksy, figurative or roundabout wording that names a value as a firm wish for that value, reading the shopper literally and kindly. Exclude a value only when the shopper plainly says to avoid it.
A3. Emphasis is not part of a value. Words such as only, nothing but, nothing short of, strictly, stays, sticking with, locked to, holding out for just show how firmly it is held, so record the bare value ("material: jute", "color: white") and never "<value> only". An exclusion appears only if the shopper also rules out mixes or other values in so many words.
A4. Comparisons with numbers (at most, at least, under, over, no more than) are limits and keep their unit. Any budget the shopper names is the new_budget and supersedes the old one, upward or downward, even if announced as good news.
A5. When the shopper tells which of two wishes matters more, the favoured one becomes a need. Mere wishes ("ideally", "would be nice") never do.
A6. A need on some other property, however related (seat height beside an overall height, strap width beside a width), adds a condition and replaces nothing. Earlier conditions are dropped or replaced only if the line says so for that very property. On a replacement, the old condition goes to dropped and the new one to needs.
A7. "Don't mind", "no preference" or "anything works" puts the property that was asked about into no_preference, and if the message asked about two properties, only the FIRST. It frees solely that property: needs stated earlier are never dropped or altered by it, and if the assistant's message asked nothing (it merely told something) such a reply changes nothing.
A8. A line saying only that the shopper did not follow or understand the question ("that was rather dense", "you've lost me", "could you put it another way", "explain again slowly") states no preference at all: no_preference gets nothing from it, needs and dropped stay empty, and not_understood is set true, so the property stays open and gets asked once more in plainer words.
A9. With several products, every need is written "property (product): constraint", the product taken from the line (talk of the "toe style of the women's boots" gives "toe style (women's boots): ...").
A9b. Only the shopper's words create conditions. The assistant's question never does, and example values inside that question do not become needs.
A11. A need expressed by pointing at another listing keeps the line's own pointing words as its value ("finish: unlike the one shown on Y's kettle page"). Values the line gives for that other listing describe it, not our product, so they are neither needs nor drops nor freed properties.

PART B - THE KEYS (any order in your output)
"references": for every need whose value the shopper supplies only by pointing at another listing or page, one object describing the listing pointed at: {"property": the property concerned, "seller": shop or brand of that listing, copied exactly from the line, "kind": that listing's sort of product, "details": [short strings in the line's words for other values it gives that listing], "relation": "same", "pairs" (our value is the one the store pairs with that listing's value: its counterpart or opposite number, or a value that has to suit or get along with it rather than equal it), "differ" (must NOT equal that listing's value: set apart from, different from, other than), "within", "at most" or "at least", "tolerance": number belonging to "within", else null, "unit": its unit, else ""}; [] if there are none.
"allowance": null unless the lines put a fresh limit on the assistant's messages, questions, steps, moves or calls; then {"messages_total": overall messages now permitted, or null, "messages_left": messages still sendable from this point, or null, "steps_total": overall steps now permitted, or null, "steps_left": steps remaining from this point, or null}.
"instructions": procedural instructions that these lines give the assistant from now on (check with me ahead of buying, tell me the total or the basket, do or do not do something, how to finish or what to say at the finish), one object apiece: {"text": the instruction in the lines' own words, "before_finish": set true if it must be done no later than the finish (an order or a decline), else false, "needs_message": set true if it takes a message to the shopper, else false, "before_change": set true if the assistant has to check with the shopper first before it changes, replaces or removes anything in the cart, else false}; [] if the lines contain none. Needs, budgets and dropped conditions are never instructions.
"reply": "" unless the shopper puts a question to the assistant or wants something confirmed; in that case a brief plain answer to send back, naming the change in substance (confirming the new value, for instance).
"requirements_now": the full current list of firm conditions on the product, each "property: constraint". Take REQUIREMENTS SO FAR, keep every entry the shopper has not dropped or replaced, keep just the latest value of anything replaced, and add each new need stated in the shopper's lines. Nothing comes from the request's story unless REQUIREMENTS SO FAR already holds it: the shopper's lines are the only source of additions. Leave out price, stock, quantity and product category.
"not_understood": true if the line only says the shopper did not follow or understand the assistant's message, or asks for it again in shorter or simpler words, and names no value; false otherwise.
"meta_only": true if the line merely asks the assistant to put fewer things at once (one question at a time, less in one go, the first now and the rest later) and names no value; such a line answers nothing so far, adds nothing to needs, no_preference or dropped, and its figures of speech name no value. False otherwise. Not meta_only: a line saying only that the shopper did not follow or understand the question, or asking for it again in shorter or simpler words.
"unavailable": whether the shopper reports an item as no longer purchasable (sold out, gone, out of stock, button disabled): true, else false.
"new_budget": null, or the new ceiling for the total price as a number in the unit of the request's budget (take the line's own figure when it is in that unit: "13075 cents" gives 13075 against a budget in cents).
"dropped": earlier conditions the shopper has given up.
"no_preference": any property the shopper called unimportant to them.
"needs": firm conditions on the product that are new in these lines, each "property: constraint" carrying the exact value, limit, exclusion or closed list. A word value (not a figure) after a comparison word such as falls short of, nothing less than or at least is the value itself, spelled as the shopper spells it: "falls short of not available" yields "property: not available" (a field can really hold "not available" or "none").
"meaning": one short sentence giving the shopper's intent."""
READ_RULES="""(a) Kind. Sitting in the requested category makes a listing the requested kind, even if its title names some other object (a container, a case, a refill, a set, an accessory); only requirements, price and stock get judged.
(b) Scope. Nothing outside the listed requirements is judged: an age group, gender, size, occasion, condition, season, quality or use that they leave unnamed never makes a listing fail.
(c) Currency. Prices are read in the budget's own unit unless the record states some other currency or unit (45 against a budget of 1000 US cents means 45 cents, comfortably inside); rescale neither side.
(d) Figures. Convert units, then compare amounts (45 mm equals 4.5 cm). Edges lie inside a limit; a range passes only if every part of it does. If the title is the evidence because no field names the property, an exact wanted value passes when the title shows its words and figures in the same order, extra detail after them allowed (9' x 12' 4" shows 9' x 12', while 9' x 13' does not).
(e) Wrong property. Lining material vs material, length vs style, strap length vs strap type: a field about another property never substitutes for a property the category's listings carry in a field of its own.
(f) Where to look. Each requirement is settled by the field carrying its property, comparing the wanted value word by word with how this store spells it there. The spelling of a field name is irrelevant: separators, prefixes, a longer or shorter form and well-known alternative names all count (pants fit type -> "fit", collar type -> "neckline", colour -> "color family"). Decisive whenever present: a field named exactly after the property, or one with a single extra word in front of that name (length -> pants length, material -> clothing material); the value written in either one is enough. size class and size_class differ only by separators and are one field, so either spelling holding the value suffices. Lacking those, the alternative-name field decides; the options list or the title can show the value only when no field at all names the property, or when a FIELD EVIDENCE line lets the title stand in for a field this listing lacks. The description never serves as evidence.
(g) Matching words. Case, separators, singular versus plural and spelling variants are ignored. One element of a list is enough, and a longer, more specific value containing the wanted one matches (cotton <- "80% cotton", green <- "forest green", rtx 3060 <- "rtx 3060 12gb"). Synonyms, abbreviations, related words and shades (technical wood ~ "tech wood", windows 11 ~ "win11", embroidery ~ "embroidered", plain ~ "solid", off-white ~ ivory or cream) match only while no listing of the category writes the wanted value itself in that field: once others write "matte" for finish, "satin" stops matching matte. When the same field also turns up with one part alone ("nylon"), a plus-joined value ("wool + nylon") fails its other part. Which of these cases holds is settled by FIELD EVIDENCE lines when they exist, since code computes them from this store's records. Also no match: a different value of that property (quartz is not mechanical, rosewood is not maple), a word that just looks alike, a catch-all that claims every value and so names none ("suitable for all occasions", "all seasons", "universal") unless its list also names the wanted value, and a value that only tends to accompany or hint at the wanted one (a "pullover" style is not "zip-up", "slip-on" is not "lace-up"). The sole stand-ins allowed: a shade commonly sold as the wanted colour, and a more specific value containing it."""
def _sma_with_rules(text):
 return text.replace("@RULES@",READ_RULES)
CHECK_ITEM=_sma_with_rules("""OUTPUT: one JSON object, nothing around it, keyed "item" (the item's id), "fits" (true or false) and "problems" (a list of short strings, each naming one requirement not met or not shown).

You are the last gate before a shopping assistant commits to this item, and the errand is worth 1 only if every requirement below holds. Base the verdict on the records shown and nothing else.

Say false in exactly two situations: (1) the item's fields clearly break a listed requirement under the rules, as when a FIELD EVIDENCE line states the item does NOT meet one; (2) nothing states a listed requirement: no field does, and where the store has no field for it (or a FIELD EVIDENCE line lets the title stand in), neither do the options list and the title. Missing stock or price records alone never justify false.

RULES
@RULES@
(h) Words that are part of the category's own name are satisfied by the category.
(i) Exclusions and closed lists hold.
(j) The item must be of the requested kind; with category labels on listings, its label must match the request's wording as a whole, not a narrower or neighbouring one.
(k) Price multiplied by quantity must fall within the budget, equality included. Records showing it out of stock mean a fail.
(l) A wish marked as not required ("ideally", "would be nice") is no requirement.
(m) Conflicting records (about price or stock, for example): the most recent wins.""")
RANK_CHECK=_sma_with_rules("""OUTPUT: one JSON object, nothing around it: {"better": [ids, best first, at most 4], "reason": "one sentence"}.

This request wants the PRIORITY, and the assistant doing the shopping has already picked an item to order (the pick). Given below: requirements, that pick, and all listing records seen. Name listings that could satisfy all requirements while also beating the pick on that priority (cheaper, if the cheapest is wanted; a stronger value of the measure, if the best is). Judge with the rules here: price times quantity has to stay in budget, and no record may show it without stock. Drop listings whose records show a failed requirement, no stock or an over-budget price. Numbers in the title of a ROW (55 x 94, for example) give no clue to the dimension each measures, hence they never exclude any ROW.

RULES
@RULES@""")
ASK_PLAN="""OUTPUT: one JSON object with no other text: {"questions": [{"property": "...", "second": "... or empty", "question": "..."}, ...]}. List at most QUESTION_COUNT distinct properties, strongest candidate first.

WHY THIS MATTERS: a shopping assistant will put these questions to a shopper whose hidden needs are each a value of one product property. A need comes out only if a message asks about precisely that property; whatever a message asks first gets the most dependable answer, and asking about anything else just earns a polite "don't mind".

1. OFF LIMITS
 - a property the request already settles (its requirements, the price, stock, quantity); one that has been asked before; two questions on near-duplicates (two kinds of style, two spellings of one property);
 - a property whose earlier question drew a reply that it was not understood is asked once more, in plainer words with one or two example values, before anything else; after a second such reply it is spent and never comes back; a property already re-asked that way is off limits;
 - brand, maker, a product or model name, price, stock or delivery;
 - option or variant pickers, numbered duplicates of another key, and catch-all keys such as options, variants, a description, a title or a name;
 - any property that is not a key in the listings' details further down. Exception: a property under STORY HINTS without a matching key may be asked by its plain name, except when that hint carries the mark [no key] (enough of the product's listings were read without such a field); in that case ask the key below whose values vary the way the hint's detail describes, or let it go.

2. HOW A QUESTION LOOKS
 - Step-1 questions are one short, plain, friendly sentence of no more than 15 words that asks which <property> they would like for the <item kind>; one or two example values may follow. Example values alone never draw out a need, so the property itself has to be named. Nothing before it, no progress report, no instructions, no lists.
 - The property is named with the exact key that one of that product's listings below writes, copied as written; "second" stays empty in that case.
 - ONE property of ONE product per question (apart from the step-0 choice between two wishes): never two properties, never two products, since any reply discloses at most one need and a two-subject question can go unanswered entirely.
 - "second" may hold the next strongest property of the SAME product (not a rename of the first) or be left empty; the assistant may tack it onto its final message.

3. SEVERAL PRODUCTS
If the request wants SEVERAL products, hidden needs can sit on any of them. Spread the questions: every product gets one before any product gets a second (products with no property left are skipped). Draw each product's properties from its own category's list below, write them as "property (product)", and mention the product inside the question ("Which toe style would you like for the boots?").

4. WHAT TO ASK, IN ORDER
Step 0 - put plainly whatever is listed below as MUST BE LEARNED FROM THE SHOPPER and still unasked. For the budget, the question is the largest sum they are willing to spend. For a pick between two wishes, mention each value with its property, as in "If one listing can't have both, would you rather have the <value> <property> or the <value> <property>?".
Step 1 - after that, properties really present in the listings' details below, chosen as follows:
 a) Begin with the story. When STORY HINTS are given they come first; each names a detail of the story and the property it bears on. Normally the hidden need is the property that the story's concrete details are about while its value stays unsaid.
 b) Map cues to properties. A cue tells which part or aspect of the item, and so which property, the need sits on: something the user will touch or press, inspect up close or in photos, hear, measure or fit it against, read on its label, or how it was made. Typical pairs (cue => property):
    an activity, place or time of wearing or use (a gym class, the office, outdoors) => style or type, i.e. the property whose values name such uses;
    the thing doing the work inside => core component;
    repeated fastening or adjusting => closure or mechanism;
    a gift for another person, or for somebody hardly known => whom it is made for (gender, age group);
    feel against hand, skin or foot => first material or fabric;
    a body part or a space to fit => the shape or size of that part;
    appearance and photos => pattern, colour or the visible part;
    tags, labels or the fine print => material, fabric or ingredients.
    Not cues, merely background: weather, a job, a schedule, or a place where the item is neither worn nor used.
 c) The first question goes to the property those story details point at, under the listings' own name for it. Whom the item is made for is one candidate among others; raise it when the story involves someone else, or when no other hint exists.
 d) Where no story cue is left, ask first the property a buyer of this kind of item fixes ahead of all others, namely the property whose values split the kind into sub-kinds that cannot stand in for each other in use (what it runs on, the size it must have against a body or a space, the form or container it comes in, its main material, how it connects). Use the listings' own key for it. A key whose values differ only because each listing writes its own free text is never that property.
    Then descriptive properties win over logistics: pattern, material, shape, type or style variant, closure, length, function before warranty, condition, pack size, variant or size lists, years.
 e) Key order below (widest coverage across listings, most varied values) only breaks ties between candidates of equal standing under c) and d). A hidden need nearly always names a value that few listings of the category carry, so a key showing one value across the handful of listings read so far is still a strong candidate: the needed value is likely one those listings lack. Coverage counts for more than variety among the listings read. A key marked as figures with a unit (a size, width, height, length, weight) earns a slot of its own whenever one exists. Prefer keys carried by most of that product's opened listings. On a multi-product request, favour the product that still has many fitting listings, because the hidden need has to shrink that set."""
NEXT_ONE="""
NOTE FOR THIS CALL: the shopper hears a single question this time and answers it before anything further gets picked. REMAINING question(s) still exist overall; plan on spending every one. Rank your top property at the head, optionally trailed by one or two spares. Whenever a need has just surfaced, aim what follows at a fresh property and a fresh story detail."""
BEFORE_DECLINE=_sma_with_rules("""OUTPUT: one JSON object, nothing around it: {"possible_fits": [ids of listings that seem to meet every requirement, best first, at most 5], "unopened": [ids of rows-only listings to open before declining, at most 6], "reason": "one sentence"}.

A shopping assistant is on the point of telling the shopper that nothing in the store fits. That ending is usually wrong, so go over the listing records below against the requirements, using the rules here (the price has to fit the budget, and the item may not be shown out of stock).

RULES
@RULES@

LISTINGS SEEN ONLY AS SEARCH ROWS: nobody read their details, so their property fields are unknown. A row like that in the requested category (no matter what object its title names) is still unchecked when its title, price and stock reveal no failed requirement; numbers in a title such as 55 x 94 do not reveal which dimension each is and so never prove a failure. Those rows go into "unopened", likeliest first: a title naming a required value, or the requested kind with nothing counting against it.""")
REF_HOP="""OUTPUT: one JSON object, nothing around it, keyed target, missing, go_to, value and goes_with.

The PROPERTY that a shopper wants for their product is to be copied from some other store listing, which the request describes like this: DESCRIPTION. The RECORDS further down are candidates for being that listing.

Two cautions first. Whatever a record's text says, it is not the target unless it displays the described details. And none of these is a go_to: a listing mentioned only as something the target is not ("not the green one from X"), a version, size or colour on sale elsewhere, or a listing cited for some other property.

"goes_with" - a list of the values that, by the target's text or a store rule written in it, pair with, go together with or count the same as that value; from a rule naming several pairs, only the partners of that value (the value itself when the rule pairs it with itself); [] when there are none.
"value" - only when go_to is null: the PROPERTY exactly as the target's record has it: its field for the PROPERTY; with no such field, the value its notes give for the PROPERTY (an options or title text only when the notes give none).
"go_to" - normally null. If, though, the target's own text points the reader to a DIFFERENT listing for the PROPERTY (take it from, use, refer to, look at or go by another listing), describe that one as {"seller": shop or brand name copied as written, "kind": what sort of product it is, "details": [each value the text gives for it, brief and verbatim: a colour, a size with its unit, a material]}.
"missing" - which named details the target fails to display; [] when it displays them all. A detail is displayed when its value is written anywhere in the record: a field under another name, the options, the title or the notes.
"target" - the id of the one record that genuinely is the described listing, meaning the right seller and kind with every named detail visible; read details by sense, so a figure expressed in another unit is acceptable at an equal amount. null if no record qualifies."""
CHANGE_REVIEW=_sma_with_rules("""OUTPUT: a lone JSON object with four keys and no surrounding text:
- "reason": "one sentence".
- "pick": which id should be ordered at this point; null if no listing qualifies.
- "cart_still_right": null for an empty cart; true if what the cart holds is listed in fits and is also what should be ordered (a basket that still fits is kept); false when the cart carries a line that has stopped fitting, or when a better pick now exists under a cheapest or best rule.
- "fits": list of ids, strongest first (cheapest first whenever the cheapest is wanted), covering each OPENED listing that satisfies all conditions in force today, is priced within the budget and is not shown out of stock.

Background: the errand's terms just moved, because the store posted a notice, the shopper wrote a line, or a finish got refused, bringing a fresh budget, an added or removed requirement, or a vanished item. Supplied below: the request, the change, today's requirements, the PRIORITY, THE CART, and one card per listing encountered (a DETAILS card is a complete record). Re-assess every OPENED listing under the terms that hold NOW, applying these rules:
@RULES@""")
FINISH_CHECK="""OUTPUT: one JSON object, nothing around it, with two keys.
- "ready": false while any stated step is still owed; true once nothing is pending.
- "pending": the owed steps in the order they should run, two at most, each {"why": a few words naming the stated step, "tool": the precise name of the tool that does it, "args": arguments in exactly the form that tool's schema demands; when the step is a message to the shopper, write it out completely in one or two short plain sentences carrying whatever facts the step requires (for the basket: each line of the cart with its id, count and price, then the total)}. Leave out steps that no listed tool can do.

Context: you are the last look before a shopping assistant closes its errand by ordering or declining. Steps may have been stated by the request, by later shopper lines or by store notices, and some fall due before or at the finish, such as a prior confirmation with the shopper, a report of the basket contents or total, a closing remark, or one more look at something. Make nothing up: count only steps that really were stated by the request, the shopper or the store, and only steps a listed tool can perform.
Method: go through INSTRUCTIONS IN FORCE, the NEWEST SHOPPER LINES AND STORE NOTICES and DONE AFTER THE LAST CART CHANGE, then judge whether THIS finish still has a stated step outstanding. Basket steps carried out before the cart last changed do not count as done.

Basket note: if this finish is an order and nothing was messaged to the shopper since the cart last changed, code automatically sends them the basket together with its total; list such a step only when a stated one requires more or something else, like answering a question or awaiting a yes."""
DECLINE_WORDS=r"refus|declin|cancel|abandon|reject|give.?up|abstain|withdraw|no.?(?:purchase|order|buy)|nothing|without"
ORDER_WORDS=r"commit|buy|order|purchas|check.?out|pay|confirm|complete|place|accept|proceed"
SORRY="Sorry, nothing in the store meets all of your requirements, so I am not ordering anything."
ID_LIKE=r"[A-Za-z0-9][A-Za-z0-9_\-:.]{3,63}"
CODE_LIKE=r"[A-Za-z0-9_\-]+"
def _smb_cut(text,limit,mark):
 if len(text)<=limit:
  return text
 return text[:limit]+mark
def _smb_low(text):
 return str(text or"").lower()
def _smb_lower(thing):
 return str(thing).lower()
_SMB_DUMP={"ensure_ascii":False,"separators":(",",":")}
def as_text(piece,ceiling=VIEW_CHARS):
 try:
  dumped=json.dumps(piece,**_SMB_DUMP)
 except(TypeError,ValueError):
  dumped=str(piece)
 return _smb_cut(dumped,ceiling,"...(cut)")
_SMB_CLIP_DEPTH=8
def clipped(piece,chars,lvl=0):
 if lvl>_SMB_CLIP_DEPTH:
  return None
 if isinstance(piece,str):
  return _smb_cut(piece,chars,"...")
 below=lvl+1
 if isinstance(piece,dict):
  copy={}
  for field,sub in piece.items():
   copy[field]=clipped(sub,chars,below)
  return copy
 if not isinstance(piece,list):
  return piece
 return[clipped(sub,chars,below)for sub in piece]
_SMB_FENCE=re.compile(r"```[A-Za-z]*")
def _smb_unfence(answer):
 if not answer.startswith("```"):
  return answer
 inner=answer[_SMB_FENCE.match(answer).end():].strip()
 if inner.endswith("```"):
  inner=inner[:-3].strip()
 return inner
def _smb_brace_span(answer):
 if"{"not in answer:
  return""
 return answer[answer.find("{"):answer.rfind("}")+1]
def _smb_parsed(candidate):
 if candidate:
  try:
   return json.loads(candidate)
  except ValueError:
   pass
 return None
_SMB_JSON_STATES={
"whole":(lambda answer:answer,"span"),
"span":(_smb_brace_span,"done"),
}
def json_object(text_body):
 if not isinstance(text_body,str):
  return None
 answer=_smb_unfence(text_body.strip())
 state="whole"
 while state in _SMB_JSON_STATES:
  take,state=_SMB_JSON_STATES[state]
  found=_smb_parsed(take(answer))
  if isinstance(found,dict):
   return found
 return None
def is_number(piece):
 return not isinstance(piece,bool)and isinstance(piece,(int,float))
def long_strings(piece,collected,lvl=0):
 todo=[(piece,lvl)]
 while todo:
  item,depth=todo.pop()
  if depth>4:
   continue
  if isinstance(item,list):
   todo.extend((sub,depth+1)for sub in reversed(item[:10]))
  elif isinstance(item,str)and len(item.split())>3:
   collected.append(item)
def long_strings_all(d,out):
 for val in d.values():
  if isinstance(val,dict):
   source=val.values()
  elif isinstance(val,str):
   source=(val,)
  else:
   continue
  out.extend(s.strip()for s in source if isinstance(s,str)and s.strip())
_SMB_END=object()
def walk_dicts(piece,visit,lvl=0,ceiling=6):
 trail=[iter((piece,))]
 while trail:
  node=next(trail[-1],_SMB_END)
  if node is _SMB_END:
   trail.pop()
   continue
  if lvl+len(trail)-1>ceiling:
   continue
  if isinstance(node,dict):
   visit(node)
   trail.append(iter(node.values()))
  elif isinstance(node,list):
   trail.append(iter(node))
_SMB_PAREN=re.compile(r"\s*\(")
_SMB_CAMEL=re.compile(r"([a-z])([A-Z])")
_SMB_ALPHA=re.compile(r"[a-z]+")
_SMB_ALNUM=re.compile(r"[a-z0-9]+")
def _smb_head(name):
 return _SMB_PAREN.split(str(name),1)[0]
def _smb_name_runs(name,pattern):
 return pattern.findall(_SMB_CAMEL.sub(r"\1 \2",_smb_head(name)).lower())
_SMB_JUNK_ANY=re.compile(r"\b(?:brand|shop|seller|store|vendor|manufacturer|category|categories|department|title|headline|"
r"heading|listing|description|overview|price|cost|amount|availability|sku|url)\b")
_SMB_JUNK_ALL=re.compile(r"(?:name|notes?|stock|in stock|on hand|images?|links?|make|maker|shelf|ids?|id)")
def junk_key(key):
 words=" ".join(_smb_name_runs(key,_SMB_ALNUM))
 return _SMB_JUNK_ANY.search(words)is not None or _SMB_JUNK_ALL.fullmatch(words)is not None
def numbered_twin(key):
 runs=_smb_name_runs(key,_SMB_ALNUM)
 return len(runs)>=2 and len(runs[-1])==1 and runs[-1].isdigit()and runs[-2].isalpha()
def _smb_drop_s(word,keep_ss=True):
 if len(word)<4 or word[-1]!="s"or(keep_ss and word[-2]=="s"):
  return word
 return word[:-1]
def prop_words(name):
 return list(map(_smb_drop_s,_smb_name_runs(name,_SMB_ALPHA)))
def prop_key(name):
 return"".join(prop_words(name))
def edits_within(a,b,most):
 gap=len(a)-len(b)
 if max(gap,-gap)>most:
  return False
 row=list(range(len(b)+1))
 for i,ca in enumerate(a):
  corner,row[0]=row[0],i+1
  for j,cb in enumerate(b,1):
   corner,row[j]=row[j],min(corner+(ca!=cb),row[j-1]+1,row[j]+1)
  if min(row)>most:
   return False
 return row[-1]<=most
def _smb_by_length(x,y):
 return(y,x)if len(y)<len(x)else(x,y)
def same_prop(a,b):
 ka,kb=prop_key(a),prop_key(b)
 if not ka or not kb:
  return False
 if ka==kb:
  return True
 short,full=_smb_by_length(ka,kb)
 size=len(short)
 if size<4:
  return False
 if ka[:3]==kb[:3]and edits_within(ka,kb,2 if size>5 else 1):
  return True
 return size>4 and full.startswith(short)
def _smb_soft(word,i):
 letter=word[i]
 if letter in"aeiouy":
  return True
 return(i>0 and word[i-1]==letter)or(i+1<len(word)and word[i+1]==letter)
def _smb_reaches(word,goal,steps):
 if steps==0:
  return word==goal
 return any(_smb_reaches(word[:i]+word[i+1:],goal,steps-1)for i in range(len(word))if _smb_soft(word,i))
def spelled_alike(lo,hi):
 extra=len(hi)-len(lo)
 if extra in(1,2):
  return _smb_reaches(hi,lo,extra)
 if extra!=0:
  return False
 spots=[i for i,(p,q)in enumerate(zip(lo,hi))if p!=q]
 if len(spots)!=2 or spots[1]-spots[0]!=1:
  return False
 pair=slice(spots[0],spots[1]+1)
 return lo[pair]==hi[pair][::-1]
def strict_prop(a,b):
 left,right=prop_words(a),prop_words(b)
 ka,kb="".join(left),"".join(right)
 if not ka or not kb:
  return False
 if ka==kb:
  return True
 short,full=_smb_by_length(ka,kb)
 if len(short)>3 and short[:3]==full[:3]and spelled_alike(short,full):
  return True
 fewer,more=_smb_by_length(left,right)
 if len(more)-len(fewer)!=1 or more[:-1]!=fewer:
  return False
 return len(more[-1])<4 and len("".join(fewer))>4
CH_SKIP={"and","the","a","an","of"}
CH_BOUND=(r"\b(?:at least|no more than|at most|no less than|more than|less than|up to|or more|or less|or larger|or smaller|or higher|"
r"or lower|minimum|maximum|between|within|under|over|above|below|than)\b")
VALUE_LINKS=(r"at least|no more than|at most|no less than|more than|less than|up to|or more|or less|or larger|or smaller|"
r"or higher|or lower|minimum|maximum|between|within|under|over|above|below|exactly|only|either|neither|nor|than|plus")
BOUND_WORDS=r"\b(?:no|not)\s+(?:more|less|fewer|over|under|above|below|beyond|exceeding)\b"
EXCLUDE_WORDS=r"\b(?:no(?!-)|not|without|nothing|never|neither|nor|except|excluding|avoid\w*|anything but|other than|rather than|instead of)\b"
CJ_COMPARE=(r"\b(?:falls?\s+short\s+of|short\s+of|nothing\s+(?:less|other)\s+than|nothing\s+short\s+of|nothing\s+but|"
r"no\s+(?:less|worse)\s+than|at\s+least|(?:less|more|better|worse)\s+than|exactly|only|just)\b")
LITERAL_VALUES=set()
STORE_TWINS={}
def vowel_twin(a,b):
 if len(a)!=len(b)or len(a)<4 or a[:2]!=b[:2]or not(a.isalpha()and b.isalpha()):
  return False
 spots=[i for i,(p,q)in enumerate(zip(a,b))if p!=q]
 return len(spots)==1 and a[spots[0]]in"aeiouy"and b[spots[0]]in"aeiouy"
def spelling_twin(a,b):
 if a==b:
  return False
 if vowel_twin(a,b):
  return True
 short,full=_smb_by_length(a,b)
 return len(short)>3 and short[:3]==full[:3]and short.isalpha()and full.isalpha()and spelled_alike(short,full)
_SMB_VALUE_WORD=re.compile(r"[a-z0-9]+(?:\.[0-9]+)?")
_SMB_MARKS=re.compile(r"\((?:stated|revealed[^)]*)\)")
_SMB_EMPHASIS=re.compile(r"\b(?:only|just|exactly|strictly)\b")
_SMB_OR=re.compile(r"\bor\b")
_SMB_DIGIT=re.compile(r"\d")
_SMB_FIGURE=re.compile(r"\d+(?:\.\d+)?")
_SMB_LIST_CUT=re.compile(r"[,;|]")
def ch_words(text):
 found=_SMB_VALUE_WORD.findall(_smb_low(text))
 return[token for token in found if token not in CH_SKIP]
def _smb_near_word(word,other):
 if not(word.isalpha()and other.isalpha()):
  return False
 if same_kind(word,other)or other in STORE_TWINS.get(word,()):
  return True
 return min(len(word),len(other))>5 and word[:2]==other[:2]and edits_within(word,other,1)
def ch_word_in(word,pool):
 return any(candidate==word or _smb_near_word(word,candidate)for candidate in pool)
def ch_alts(value):
 plain=_SMB_EMPHASIS.sub(" ",_SMB_MARKS.sub(" ",_smb_low(value)).replace('"'," "))
 options=map(ch_words,_SMB_OR.split(plain))
 return[option for option in options if option]
def ch_writes(value,text):
 return ch_match(ch_alts(value),ch_words(text))
def _smb_glued(pool,start,target):
 pos,end=0,len(target)
 for word in pool[start:]:
  if pos>=end:
   break
  if not target.startswith(word,pos):
   return False
  pos+=len(word)
 return pos==end
def _smb_glue_worth(option,glued,pool):
 if len(option)>1 or _SMB_DIGIT.search(glued):
  return True
 return any(_SMB_DIGIT.search(word)for word in pool)
def ch_match(alts,pool):
 if not pool:
  return False
 for option in alts:
  if all(ch_word_in(part,pool)for part in option):
   return True
  glued="".join(option)
  if glued and _smb_glue_worth(option,glued,pool):
   if any(_smb_glued(pool,start,glued)for start in range(len(pool))):
    return True
 return False
def _smb_lone_figure(fig):
 return r"(?<![\d.])"+re.escape(fig)+r"(?!\d|\.\d)"
def ch_touches(value,text):
 field_words=ch_words(text)
 raw=_smb_low(text)
 openings={token[:3]for token in field_words if len(token)>2}
 def shares(part):
  if ch_word_in(part,field_words):
   return True
  if _SMB_FIGURE.fullmatch(part)and re.search(_smb_lone_figure(part),raw):
   return True
  return len(part)>2 and part[:3]in openings
 return any(shares(part)for option in ch_alts(value)for part in option)
def ch_parts(text):
 whole=str(text or"").strip().lower()
 items=[item for item in map(str.strip,_SMB_LIST_CUT.split(whole))if item]
 if not items and whole:
  items.append(whole)
 return items
def ch_sides(part):
 sides=[side for side in map(str.strip,str(part).split("+"))if side]
 return sides if len(sides)>1 else[]
_SMB_UNIT_AFTER=re.compile(r"\s*([a-z]+)")
def figure_written(fig,said):
 for hit in re.finditer(_smb_lone_figure(fig),said):
  if"."in fig:
   return True
  unit=_SMB_UNIT_AFTER.match(said,hit.end())
  if unit is None or len(unit.group(1))<4:
   return True
 return False
_SMB_WORD3=re.compile(r"[a-z]{3,}")
_SMB_FLAG_POSITIVE=re.compile(r"\W*(?:yes|no|true|false|none|required|included|needed)\W*")
_SMB_LINKS=re.compile(r"\b(?:"+VALUE_LINKS+r")\b")
_SMB_CLAUSES=re.compile(r",|;|/|&|\(|\)|\bor\b|\band\b")
_SMB_EXCLUDE=re.compile(EXCLUDE_WORDS)
_SMB_BOUND_CUTS=(
(re.compile(BOUND_WORDS),False),
(re.compile(r"\b(?:no|not)\s+[a-z]+er\b(?=\s+than\b)"),False),
(re.compile(r"\b(?:no|not)\s+[a-z]+er\b"),True),
)
def _smb_content(text):
 return[w for w in _SMB_WORD3.findall(text)if w not in LINK_WORDS]
def positive_value(value,name=""):
 text=str(value).lower().strip()
 if _SMB_FLAG_POSITIVE.fullmatch(text):
  return True
 if LITERAL_VALUES:
  flat=" ".join(_SMB_ALNUM.findall(_SMB_MARKS.sub(" ",text)))
  if flat in LITERAL_VALUES:
   return True
 for cut,needs_figure in _SMB_BOUND_CUTS:
  if needs_figure and not _SMB_DIGIT.search(text):
   continue
  text=cut.sub(" ",text)
 clauses=_SMB_CLAUSES.split(_SMB_LINKS.sub(" ",text))
 if max(len(_smb_content(clause))for clause in clauses)>3:
  return False
 neg=_SMB_EXCLUDE.search(text)
 if not neg or _smb_content(text[:neg.start()]):
  return True
 tail_words=_smb_content(text[neg.end():])
 own=set(prop_words(name)).union(_SMB_ALPHA.findall(_smb_lower(name)))
 if not tail_words or not own:
  return False
 return all(w in own or w.rstrip("s")in own for w in tail_words)
_SMB_FIGURE_RUN=re.compile(r"\d+(?:\.\d+)?(?:\s*[:x/]\s*\d+(?:\.\d+)?)*")
def _smb_run_pattern(run,tail):
 figs=_SMB_FIGURE.findall(run)
 return figs,r"(?<![\d.])"+r"\D{0,4}".join(map(re.escape,figs))+tail
def value_seen(value,samples):
 seen=_smb_low(samples)
 if not seen.strip():
  return False
 core=_SMB_LINKS.sub(" ",_smb_lower(value))
 stems=(w if len(w)<5 else w[:-1]for w in _smb_content(core))
 if any(re.search(r"\b"+re.escape(stem),seen)for stem in stems):
  return True
 return any(re.search(_smb_run_pattern(run,r"(?![\d.])")[1],seen)for run in _SMB_FIGURE_RUN.findall(core))
_SMB_POSSESSIVE=re.compile(r"'s\b")
def kind_words(text):
 return{token.rstrip("s")for token in _smb_content(_smb_lower(text))}
def kind_tokens(text):
 plain=_smb_lower(text).replace("\u2019","'")
 return _smb_content(_SMB_POSSESSIVE.sub("",plain))
_SMB_PLURAL_RULES=(
("",0,("","s","es")),
("y",1,("ies",)),
("fe",2,("ves",)),
("f",1,("ves",)),
("ies",3,("y",)),
("ves",3,("f","fe")),
("es",2,("",)),
("s",1,("",)),
)
def plural_forms(word):
 w=str(word)
 forms=set()
 for ending,cut,endings in _SMB_PLURAL_RULES:
  if w.endswith(ending):
   stem=w[:len(w)-cut]
   forms.update(stem+e for e in endings)
 return{f for f in forms if len(f)>2}
def same_kind(a,b):
 return a==b or any(str(y)in plural_forms(x)for x,y in((a,b),(b,a)))
def kind_in(word,pool):
 for other in pool:
  if same_kind(word,other):
   return True
 return False
def kinds_within(words,pool):
 for w in words:
  if not kind_in(w,pool):
   return False
 return True
def kind_score(own,words,shared):
 total=0
 for w in own:
  if kind_in(w,words):
   total+=0.5 if kind_in(w,shared)else 1.0
 return total
def _smb_kind_alternation(word):
 forms=sorted(plural_forms(word),key=lambda f:(-len(f),f))
 return"(?:"+"|".join(map(re.escape,forms))+r")(?:'s\b|'|\b)"
def name_pattern(name):
 return r"[\s'-]+".join(map(_smb_kind_alternation,kind_tokens(name)))
_SMB_HINT_TAG=re.compile(r"\(([^()]*)\)\s*$")
def one_prop_hint(hint):
 try:
  head,colon,detail=hint.partition(":")
  if colon and"/"in head:
   tag=_SMB_HINT_TAG.search(head)
   listed=head if tag is None else head[:tag.start()]
   names=[n for n in map(str.strip,listed.split("/"))if n]
   if len(names)>1:
    label=names[0]if tag is None else names[0]+" ("+tag.group(1).strip()+")"
    others=", ".join(names[1:])
    return f"{label}:{detail} (or {others})"
 except Exception:
  pass
 return hint
_SMB_LEAD_WORD=re.compile(r"\s*([a-z][a-z-]*[a-z])(?:\s*[,;.()]|\s*$|\s+(?:"+"|".join(sorted(LINK_WORDS))+r")\b)")
_SMB_REQUEST_TOKEN=re.compile(r"[a-z][a-z'-]*")
def lead_value(detail,request,product):
 try:
  opening=_SMB_LEAD_WORD.match(_smb_lower(detail))
  if opening is None:
   return""
  lead=opening.group(1)
  names=_smb_content(_smb_lower(product))
  if not names or len(lead)<3 or lead in LINK_WORDS or lead.rstrip("s")in{n.rstrip("s")for n in names}:
   return""
  head_kind=names[0].rstrip("s")
  tokens=_SMB_REQUEST_TOKEN.findall(_smb_lower(request))
  for left,right in zip(tokens,tokens[1:]):
   if left==lead and right.partition("'")[0].rstrip("s")==head_kind:
    return lead
 except Exception:
  pass
 return""
_SMB_POINTING=re.compile(r"as shown|same as|the page|listing|sold by|redirect|refer")
_SMB_FLAG_STORY=re.compile(r"\W*(?:yes|no|true|false|required|included|needed)\W*")
def _smb_root(word):
 for ending,longer_than in(("es",4),("s",3)):
  if word.endswith(ending)and len(word)>longer_than:
   return word[:-len(ending)]
 return word
def _smb_written(word,told_words,told_roots):
 root=_smb_root(word)
 long_word=len(word)>3
 def covers(told):
  if told==word:
   return True
  if told.startswith(root)and len(told)-len(root)<=3:
   return True
  if root.startswith(told)and len(told)>=max(4,len(root)-2):
   return True
  if not long_word:
   return False
  return told.startswith(word)or(len(told)>3 and told[0]==word[0]and edits_within(told,word,1))
 if any(map(covers,told_words)):
  return True
 return any(len(told)>2 and root.startswith(told)and len(root)-len(told)<=3 for told in told_roots)
def story_guess(req,request,refs):
 whole=str(req)
 name_part,_colon,stated=whole.partition(":")
 wanted=stated.strip().lower()
 if not wanted or whole in refs or _SMB_POINTING.search(wanted):
  return[]
 if _SMB_FLAG_STORY.fullmatch(wanted):
  wanted=_smb_head(name_part).lower()
 core=_SMB_LINKS.sub(" ",wanted)
 told=_smb_lower(request)
 numbers=_SMB_FIGURE.findall(core)
 if numbers and all(figure_written(n,told)for n in numbers):
  return[]
 told_words=set(_SMB_ALPHA.findall(told))
 told_roots={_smb_drop_s(t,keep_ss=False)for t in told_words}
 words=_smb_content(core)
 unseen=[w for w in words if not _smb_written(w,told_words,told_roots)]
 if len(words)<3 and any(len(w)>3 and w not in unseen for w in words):
  return[]
 for run in _SMB_FIGURE_RUN.findall(core):
  parts,pattern=_smb_run_pattern(run,r"(?!\d)")
  if len(parts)>1 and re.search(pattern,told)is None:
   unseen.append(run)
 return unseen
REF_HOPS=8
REF_CALLS=12
REF_STEPS=0.4
REF_DIFFER_BLIND=9
REF_DIFFER_STEPS=0.4
REF_WORDS=r"\bas shown\b|\bshown on\b|\bsame as\b|\bpage\b|\bsold by\b|\bredirect|\brefer(?:s|red|ring)?\b|\blisting points\b|\bpoints you\b"
SELLER_KEYS=r"shop|seller|store|vendor|brand|make|manufactur|merchant|retailer|supplier|sold.?by"
POINT_WORDS=r"\bwithin\b|\bat (?:most|least)\b|\bno(?:t any)? (?:more|less) than\b|\bsame as\b|\bas shown\b|\bthe value\b|\bthe one\b"
DIFFER_WORDS=(r"\bset apart from\b|\bapart from\b|\bdistinct from\b|\bdifferent (?:from|to|than)\b|\bdiffer(?:s|ent|ing)?\s+from\b"
r"|\bother than\b|\bunlike\b|\bnot the same as\b|\banything but\b")
PAIR_WORDS=r"\bcounterparts?\b|\bopposite numbers?\b|\bpartner value\b|\bpaired value\b|\bcompatible with\b"
_SMB_REF=re.compile(REF_WORDS)
_SMB_UNIT_GAP=re.compile(r"(\d)(?:\s+|\s*-\s*)([a-z]{1,8})\b")
_SMB_CODE_HEAD=re.compile(r"(?<![0-9])[a-z]+\d*$")
_SMB_REF_TOKEN=re.compile(r"\d+(?:[.,:/]\d+)*[a-z]*|[a-z][a-z0-9]*")
_SMB_ES_END=re.compile(r"(?:ch|sh|x|z)es$")
_SMB_FIG_UNIT=re.compile(r"(\d+(?:\.\d+)?(?:\s*/\s*\d+(?:\.\d+)?)*)\s*-?\s*([a-z]{1,8})\b")
_SMB_SLASH=re.compile(r"\s*/\s*")
_SMB_NOT_ALNUM=re.compile(r"[^a-z0-9]")
def ref_text(value):
 return _SMB_REF.search(_smb_low(value))is not None
def _smb_join_unit(hit):
 figure,unit=hit.groups()
 code=_SMB_CODE_HEAD.search(hit.string[max(0,hit.start()-16):hit.start()])
 return hit.group(0)if unit in LINK_WORDS or code else figure+unit
def _smb_plural_cut(tok,shortest,digit_guard):
 if len(tok)<shortest or tok[-1]!="s"or tok[-2]=="s"or(digit_guard and tok[-2].isdigit()):
  return tok
 return tok[:-2]if _SMB_ES_END.search(tok)else tok[:-1]
def ref_toks(text):
 fused=_SMB_UNIT_GAP.sub(_smb_join_unit,_smb_low(text))
 kept=(t for t in _SMB_REF_TOKEN.findall(fused)if t not in LINK_WORDS and(len(t)!=1 or t.isdigit()))
 return[_smb_plural_cut(t,4,True)for t in kept]
def ref_figs(text):
 pairs=set()
 try:
  flat=_smb_low(text).replace(",","")
  for hit in _SMB_FIG_UNIT.finditer(flat):
   numbers,word=hit.groups()
   word=_smb_plural_cut(word,3,False)
   if word in LINK_WORDS:
    continue
   for piece in _SMB_SLASH.split(numbers):
    pairs.add((float(piece),word))
 except Exception:
  pass
 return pairs
def ref_name(text):
 return _SMB_NOT_ALNUM.sub("",_SMB_POSSESSIVE.sub("",_smb_low(text)))
_SMB_RELATIONS=(
("pairs",re.compile(PAIR_WORDS)),
("differ",re.compile(DIFFER_WORDS)),
("within",re.compile(r"\bwithin\s+(\d+(?:\.\d+)?)\s*([a-z%]+)?")),
("at most",re.compile(r"\b(?:no more than|at most|up to|or less|or lower|or smaller)\b")),
("at least",re.compile(r"\b(?:at least|no(?:t any)? less than|or more|or higher|or larger)\b")),
)
def ref_relation(value):
 low=_smb_low(value)
 for relation,pattern in _SMB_RELATIONS:
  hit=pattern.search(low)
  if hit is None:
   continue
  if relation!="within":
   return relation,None,""
  amount,unit=hit.groups()
  return relation,float(amount),(unit or"").strip()
 return"same",None,""
_SMC_ALT_CUT=re.compile(r"\(\s*or\b")
_SMC_TAIL_CUT=re.compile("\\s[\u2014\u2013]\\s|\\s-\\s|\\s(?:shows?|says|lists)\\b")
_SMC_SELLER=re.compile(r"(?<![A-Za-z'])([A-Z][a-z]{3,})(?:'s)?(?![A-Za-z])")
_SMC_SOLD_BY=re.compile(r"(?is).*(?:\bthe|\bfor|\ba|\ban)\s+(.*?)\s+(?:sold|made|offered|listed)\s+by\s*$")
_SMC_KIND_LEAD=re.compile(r"(?i)\s*([^,;(]*?)(?=\s+(?:with|that|which|in|at|having|from|of|for|sold)\b|[,;]|$)")
_SMC_DETAIL_CUT=re.compile(r"(?i),|;|\band\b|\bwith\b|\bthat\s+(?:is|are|has|have|comes?(?:\s+as)?)\b|\bwhich\b|\bin\b|\bat\b")
_SMC_DETAIL_WORD=re.compile(r"[A-Za-z0-9][A-Za-z0-9.%:/&'-]*")
def _smc_seller_hit(body,ok):
 for found in _SMC_SELLER.finditer(body):
  if found.start()==0:
   continue
  if ok is None or ok(found.group(1)):
   return found
 return None
def _smc_kind_and_rest(before,after):
 sold=_SMC_SOLD_BY.search(before)
 if sold:
  return sold.group(1),after
 lead=_SMC_KIND_LEAD.match(after)
 if lead is None:
  return"",after
 return lead.group(1),after[lead.end():]
def _smc_detail_runs(rest):
 runs=[]
 for piece in _SMC_DETAIL_CUT.split(rest):
  kept=[w for w in _SMC_DETAIL_WORD.findall(piece)if w.lower()not in LINK_WORDS]
  if kept:
   runs.append(" ".join(kept))
 return runs
def ref_parse(text,ok=None):
 try:
  body=_SMC_ALT_CUT.split(str(text or""))[0]
  hit=_smc_seller_hit(body,ok)
  if hit is None:
   return None
  start,end=hit.span()
  kind,rest=_smc_kind_and_rest(body[:start],body[end:])
  rest=_SMC_TAIL_CUT.split(rest)[0]
  return dict(seller=hit.group(1),kind=kind.strip()[:80],details=_smc_detail_runs(rest)[:6])
 except Exception:
  return None
def _smc_low_words(text):
 return re.findall(r"[a-z0-9]+",str(text or"").lower())
def prop_named(prop,text,exact=False):
 head=re.split(r"\s*\(",str(prop or""))[0]
 wanted=_smc_low_words(re.sub(r"([a-z])([A-Z])",r"\1 \2",head))
 seen=_smc_low_words(text)
 if not wanted or len(" ".join(wanted))<3:
  return False
 def forms(word):
  return(word,)if exact or len(word)<3 else(word,word+"s",word+"es")
 width=len(wanted)
 for start in range(len(seen)-width+1):
  if all(seen[start+k]in forms(w)or w in forms(seen[start+k])for k,w in enumerate(wanted)):
   return True
 return False
def ref_span(prop,source):
 try:
  keys=[w for w in _smc_low_words(re.split(r"\s*\(",str(prop))[0])if w not in LINK_WORDS]
  if not keys:
   return""
  said=str(source)
  mention=re.search(r"\b"+r"\W+".join(map(re.escape,keys))+r"\b",said.lower())
  if mention is None:
   return""
  stop=mention.end()
  return re.split(r"\(\s*or\b|(?<=[.!?])\s",said[stop:stop+500])[0]
 except Exception:
  return""
def _smc_id_pairs(rec):
 return[(key,val)for key,val in rec.items()
if isinstance(val,str)and"id"in str(key).lower()and re.fullmatch(ID_LIKE,val)]
_SMC_LENS=(
("price",
lambda low,val:re.search(r"price|cost",low)and is_number(val),
lambda low,val:is_number(val)and re.search(r"amount|fee|charge|pay|rate|value",low)
and not re.search(r"total|count|qty|quant|unit|due|sum|rating|star|score|page|number|num|found|match|dimension|measure|_cm|_mm|_in\b|_kg|_g\b",low)),
("in_stock",
lambda low,val:re.search(r"stock|avail",low),
lambda low,val:isinstance(val,bool)and re.search(r"hand|orderable|buyable|sellable|ready|ship|instock|purchas|in_store",low)),
("title",
lambda low,val:re.search(r"title|heading",low),
lambda low,val:isinstance(val,str)and re.search(r"headline|name|label|caption",low)
and not re.search(r"shop|seller|store|vendor|brand|make|categor|shelf|department|id",low)),
)
def _smc_lens_fill(rec):
 if not _smc_id_pairs(rec):
  return
 lowered={}
 for orig in rec:
  lowered[str(orig).lower()]=orig
 for plain_name,carries,stands_in in _SMC_LENS:
  if any(carries(low,rec[orig])for low,orig in lowered.items()):
   continue
  found=[orig for low,orig in lowered.items()if stands_in(low,rec[orig])]
  if len(found)==1:
   rec[plain_name]=rec[found[0]]
def lensed(piece):
 try:
  copied=json.loads(json.dumps(piece))
 except(TypeError,ValueError):
  return piece
 walk_dicts(copied,_smc_lens_fill)
 return copied
_SMC_JSON_TYPES={
"string":lambda val:isinstance(val,str),
"integer":lambda val:isinstance(val,int)and not isinstance(val,bool),
"number":is_number,
"boolean":lambda val:isinstance(val,bool),
"object":lambda val:isinstance(val,dict),
"array":lambda val:isinstance(val,list),
"null":lambda val:val is None,
}
def _smc_fits(val,kinds):
 return any(_SMC_JSON_TYPES.get(kind,lambda _v:True)(val)for kind in kinds)
def _smc_as_integer(val):
 if isinstance(val,float)and val.is_integer():
  return True,int(val)
 if isinstance(val,str)and re.fullmatch(r"\s*-?\d+(\.0+)?\s*",val):
  return True,int(float(val))
 return False,val
def _smc_as_number(val):
 if not(isinstance(val,str)and re.fullmatch(r"\s*-?\d+(\.\d+)?\s*",val)):
  return False,val
 num=float(val)
 return True,(int(num)if num.is_integer()else num)
def _smc_as_string(val):
 return(True,str(val))if is_number(val)else(False,val)
def _smc_as_boolean(val):
 if isinstance(val,str)and val.lower()in("true","false"):
  return True,val.lower()=="true"
 return False,val
_SMC_COERCE={"integer":_smc_as_integer,"number":_smc_as_number,"string":_smc_as_string,"boolean":_smc_as_boolean}
def _smc_coerce(val,kinds):
 if not kinds:
  return val
 if _smc_fits(val,kinds):
  return int(val)if"integer"in kinds and isinstance(val,float)and val.is_integer()else val
 for kind in kinds:
  convert=_SMC_COERCE.get(kind)
  if convert is not None:
   done,out=convert(val)
   if done:
    return out
 return val
def _smc_rule_enum(val,schema,word,where):
 allowed=schema[word]
 if val in allowed:
  return val,[]
 if isinstance(val,str):
  folded=val.strip().lower()
  for opt in allowed:
   if isinstance(opt,str)and opt.lower()==folded:
    return opt,[]
 return val,[f"{where}: must be one of {allowed}"]
def _smc_rule_const(val,schema,word,where):
 want=schema[word]
 return val,([]if val==want else[f"{where}: must equal {want!r}"])
def _smc_rule_cut(val,schema,word,where):
 if isinstance(val,str):
  val=val[:schema[word]]
 return val,[]
def _smc_sized(kind,too_far,text):
 def rule(val,schema,word,where):
  limit=schema[word]
  if isinstance(val,kind)and too_far(len(val),limit):
   return val,[f"{where}: "+text.format(limit)]
  return val,[]
 return rule
def _smc_bounded(too_far,text):
 def rule(val,schema,word,where):
  limit=schema[word]
  if is_number(val)and is_number(limit)and too_far(val,limit):
   return val,[f"{where}: {text} {limit}"]
  return val,[]
 return rule
def _smc_rule_items(val,schema,word,where):
 inner=schema[word]
 if not(isinstance(val,list)and isinstance(inner,dict)):
  return val,[]
 fixed,faults=[],[]
 for idx,thing in enumerate(val):
  thing,more=ArgCheck.execute(thing,inner,f"{where}[{idx}]")
  fixed.append(thing)
  faults.extend(more)
 return fixed,faults
def _smc_rule_fields(val,schema,word,where):
 if not isinstance(val,dict):
  return val,[]
 specs=schema.get("properties")or{}
 faults=[f"{where}.{name}: required"for name in schema.get("required")or[]if name not in val]
 if schema.get("additionalProperties")is False:
  faults.extend(f"{where}.{name}: not an allowed argument"for name in val if name not in specs)
 fixed={}
 for name,thing in val.items():
  if name in specs:
   thing,more=ArgCheck.execute(thing,specs[name],f"{where}.{name}")
   faults.extend(more)
  fixed[name]=thing
 return fixed,faults
_SMC_RULES=(
("enum",_smc_rule_enum),
("const",_smc_rule_const),
("minLength",_smc_sized(str,lambda n,lim:n<lim,"shorter than {} characters")),
("maxLength",_smc_rule_cut),
("minimum",_smc_bounded(lambda val,lim:val<lim,"below minimum")),
("maximum",_smc_bounded(lambda val,lim:val>lim,"above maximum")),
("exclusiveMinimum",_smc_bounded(lambda val,lim:val<=lim,"must be above")),
("exclusiveMaximum",_smc_bounded(lambda val,lim:val>=lim,"must be below")),
("minItems",_smc_sized(list,lambda n,lim:n<lim,"needs at least {} items")),
("maxItems",_smc_sized(list,lambda n,lim:n>lim,"allows at most {} items")),
("items",_smc_rule_items),
("type",_smc_rule_fields),
)
def _smc_type_names(named):
 return[named]if isinstance(named,str)else list(named or[])
class ArgCheck:
 @classmethod
 def execute(cls,piece,jschema,where="args"):
  if not isinstance(jschema,dict):
   return piece,[]
  kinds=_smc_type_names(jschema.get("type"))
  piece=_smc_coerce(piece,kinds)
  if kinds and not _smc_fits(piece,kinds):
   return piece,[where+": expected "+"/".join(kinds)]
  faults=[]
  for word,rule in _SMC_RULES:
   if word not in jschema:
    continue
   piece,more=rule(piece,jschema,word,where)
   faults.extend(more)
  return piece,faults
class SendFault:
 data=None
 def __init__(self,why):
  self.error={"status":None,"body":why[:300]}
_SMC_MODEL_FLOOR=LAST_CALL-10
def _smc_drive(table,state):
 while state is not None:
  state=table[state]()
def _smc_clamp(value,low,high):
 return int(min(max(value,low),high))
def _smc_client(seconds):
 return ProxyClient(timeout=seconds,max_retries=1,retry_delay=1)
_SMC_VERDICTS=(
("spent",lambda code,low:code==402 or(code==403 and re.search(r"key limit|insufficient|credit",low))),
("refused",lambda code,low:code in(401,403,404)),
("body",lambda code,low:code in(400,413,422)),
("client",lambda code,low:isinstance(code,int)and 400<=code<500 and code not in(408,425,429)),
)
class _smc_ModelCall:
 def __init__(self,wire,msgs,models,cap,tools):
  self.wire,self.msgs,self.models,self.cap,self.tools=wire,msgs,models,cap,tools
  self.kinds=set()
  self.skip=set()
  self.ladder=list(models)+[m for m in wire.extra_models if m not in models]
  self.lone=self.few_left()
  self.seed=self.model=self.text=""
  self.pos,self.left,self.started=-1,0,0.0
  self.plain=False
  self.answer=self.status=self.out=None
 def run(self):
  _smc_drive({"pick":self.pick,"send":self.send,"reply":self.reply,"miss":self.miss,
"spent":self.spent,"refused":self.refused,"body":self.body_refused,
"client":self.client_refused,"soft":self.soft},"pick")
  return self.out
 def few_left(self):
  return len({m for m in self.ladder if m not in self.wire.dropped})<=1
 def stop(self):
  self.out=self.wire._smc_give_up(self.kinds)
  return None
 def ends_here(self):
  return self.lone or len(self.wire.call_models)>=2
 def too_dear(self):
  wire=self.wire
  cheaper_next=self.pos+1<len(self.models)and"sonnet"not in self.models[self.pos+1]
  guess=sum(len(json.dumps(m,default=str))for m in self.msgs)/2.3*2e-6+min(self.cap,1500)*1e-5
  scale=wire._smc_scale()
  if wire.spent_out():
   return True
  return cheaper_next and(wire.spend>=SOFT_SPEND*scale or wire.spend+guess>SPEND_CAP*scale)
 def barred(self,name):
  wire=self.wire
  return name in wire.dropped or name in self.skip or bool(self.tools and name in wire.no_tools)
 def pick(self):
  wire=self.wire
  self.pos+=1
  if self.pos>=len(self.ladder):
   return self.stop()
  name=self.ladder[self.pos]
  if not wire.no_model and wire.spend>=HARD_SPEND*wire._smc_scale():
   wire.no_model=True
   wire.log.append(dict(kind="spend_stop",spend=round(wire.spend,4)))
  if wire.no_model:
   return self.stop()
  if self.barred(name)or("sonnet"in name and self.too_dear()):
   return"pick"
  self.model,self.plain=name,False
  self.left=1 if wire.down()else MODEL_TRIES
  return"send"
 def send(self):
  if not self.left:
   return"pick"
  self.left-=1
  wire=self.wire
  spare=wire.spare()
  if spare<_SMC_MODEL_FLOOR:
   return self.stop()
  ceiling=30 if wire.down()else 75
  wire.llm.timeout=_smc_clamp((spare-LAST_CALL)+15,8,ceiling)
  messages=wire._smc_flat(self.msgs)if self.plain else wire._smc_marked(self.msgs,self.model)
  request=dict(model=self.model,messages=messages,temperature=0,max_tokens=self.cap)
  if self.tools:
   request.update(tools=self.tools,tool_choice="auto")
  wire.call_models.add(self.model)
  self.started=monotonic()
  self.answer=wire._smc_post(lambda:wire.llm.post_verbose("/inference/chat/completions",request),True)
  got=self.answer.data
  return"reply"if isinstance(got,dict)and got.get("choices")else"miss"
 def reply(self):
  wire,got=self.wire,self.answer.data
  wire.last_reply,wire.misses=monotonic(),0
  usage=got.get("usage")or{}
  if is_number(usage.get("cost")):
   wire.spend+=float(usage["cost"])
  msg=got["choices"][0].get("message")or{}
  if msg.get("content")or msg.get("tool_calls"):
   self.out=msg
   return None
  self.kinds.add("empty")
  wire.log.append(dict(kind="inference_failure",model=self.model,error="empty"))
  return"pick"
 def miss(self):
  wire,fault=self.wire,self.answer.error
  wire.misses+=1
  wire.log.append(dict(kind="inference_failure",model=self.model,error=str(fault)[:300]if fault else"empty"))
  fault=fault or{}
  self.status,self.text=fault.get("status"),str(fault.get("body")or"")
  low=self.text.lower()
  return next((state for state,test in _SMC_VERDICTS if test(self.status,low)),"soft")
 def spent(self):
  self.wire.no_model=True
  return self.stop()
 def refused(self):
  wire,low=self.wire,self.text.lower()
  self.kinds.add("hard")
  self.skip.add(self.model)
  if self.status==404 and self.tools and"tool"in low:
   wire.no_tools.add(self.model)
  elif self.status!=403 or not re.search(r"moderat|flagged",low):
   wire.dropped.add(self.model)
  if self.status==403 and not wire.extra_models:
   self.take_spares()
  return"pick"
 def take_spares(self):
  wire=self.wire
  listed=re.search(r'allowed_models"?\s*:\s*\[([^\]]*)',self.text)
  if not listed:
   return
  named=re.findall(r'"([^"\s]{3,100})"',listed.group(1))
  wire.extra_models=[m for m in named if m not in wire.dropped and m not in self.models][:2]
  self.ladder.extend([m for m in wire.extra_models if m not in self.ladder])
  self.lone=self.few_left()
 def body_refused(self):
  self.kinds.add("hard")
  if self.plain:
   self.skip.add(self.model)
   return"pick"
  self.plain=True
  return"send"
 def client_refused(self):
  self.kinds.add("hard")
  self.skip.add(self.model)
  return"pick"
 def soft(self):
  wire=self.wire
  self.kinds.add("soft")
  if monotonic()-self.started>20:
   if wire.down()and self.ends_here():
    return self.stop()
   return"pick"
  if not self.seed:
   self.seed=json.dumps(self.msgs[-1:],default=str)[-600:]
  wait=wire._smc_pause(self.seed+self.model,min(wire.misses-1,6),self.status)
  if wire.spare()-wait<_SMC_MODEL_FLOOR:
   return self.stop()
  wire.pace.model_at=monotonic()+wait
  if wire.down(wire.pace.model_at)and self.ends_here():
   return self.stop()
  return"send"
class _smc_WorldCall:
 def __init__(self,wire,acts):
  self.wire,self.acts=wire,acts
  self.body={**wire.bind,"step":wire.tick+1,"action":acts[0]}
  self.sends_left=3+WORLD_TRIES
  self.waits=self.bumps=0
  self.answer=self.status=self.out=None
  self.text=""
 def run(self):
  _smc_drive({"send":self.send,"booked":self.booked,"conflict":self.conflict,"backoff":self.backoff},"send")
  return self.out
 def send(self):
  if not self.sends_left:
   return None
  self.sends_left-=1
  wire=self.wire
  spare=wire.spare()
  if spare<2:
   return None
  wire.env_client.timeout=_smc_clamp(spare-1,3,95)
  self.answer=wire._smc_post(lambda:wire.env_client.post_verbose("/environment/call",self.body))
  if isinstance(self.answer.data,dict):
   return"booked"
  fault=self.answer.error or{}
  self.status,self.text=fault.get("status"),str(fault.get("body")or"")
  wire.log.append(dict(kind="env_failure",status=self.status,body=self.text[:300]))
  if self.status==402:
   wire.broke=True
   return None
  return"conflict"if self.status==409 else"backoff"
 def booked(self):
  wire,got=self.wire,self.answer.data
  wire.tick=got["step"]if isinstance(got.get("step"),int)else wire.tick+len(self.acts)
  if got.get("finished"):
   wire.done=True
  self.out=got
  return None
 def conflict(self):
  wire=self.wire
  if"finished"in self.text:
   wire.done=True
   return None
  named=re.search(r"expected (\d+)",self.text)
  if named:
   wire.tick=int(named.group(1))-1
   self.body["step"]=wire.tick+1
   return"send"
  if"reused"in self.text and self.bumps<2:
   self.bumps+=1
   self.body["step"]+=1
   return"send"
  return None
 def backoff(self):
  wire,code=self.wire,self.status
  if code in(403,422)or self.waits>=WORLD_TRIES-1 or(code in(400,401,404)and self.waits>=1):
   return None
  wait=wire._smc_pause(json.dumps(self.body,default=str)[-600:],self.waits,code)
  self.waits+=1
  if wire.spare()-wait<2:
   return None
  sleep(wait)
  return"send"
class _smc_Pacer:
 def __init__(self):
  self.last=self.calm_at=self.model_at=0.0
 def delay(self,for_model):
  gap=SLOW_GAP if monotonic()<self.calm_at else PACE_GAP
  hold=self.last+gap-monotonic()
  if for_model:
   hold=max(hold,self.model_at-monotonic())
  return hold
 def after(self,answer):
  if answer.data is None and(answer.error or{}).get("status")in(429,503):
   self.calm_at=monotonic()+SLOW_FOR
class Wire:
 def __init__(self,bind):
  self.t0=self.last_reply=monotonic()
  self.bind=dict(bind)
  self.llm,self.env_client=_smc_client(70),_smc_client(95)
  self.pace=_smc_Pacer()
  self.spend,self.tick,self.misses=0.0,0,0
  self.done=self.broke=self.mute=self.no_model=False
  self.log=[]
  self.dropped,self.no_tools,self.call_models=set(),set(),set()
  self.extra_models=[]
  self.call_fail=""
 def _smc_scale(self):
  return self.__dict__.get("scale",1.0)
 def spare(self):
  elapsed=monotonic()-self.t0
  return DEADLINE-elapsed
 def spent_out(self):
  return self.spend>=SPEND_CAP*self._smc_scale()
 def _smc_post(self,send,model=False):
  hold=min(self.pace.delay(model),self.spare()-1)
  if hold>0:
   sleep(hold)
  self.pace.last=monotonic()
  try:
   answer=send()
  except Exception as err:
   answer=SendFault(type(err).__name__+": "+str(err))
  self.pace.after(answer)
  return answer
 def _smc_pause(self,seed,tries,status):
  weight=7*len(seed)+sum(map(ord,seed[-64:]))+31*tries+13*self.tick
  mix=weight%101/100.0
  first=2.0 if status==429 else 1.0
  return min(8.0,first*2**tries)*(0.75+mix/2)
 def down(self,at=None):
  if self.no_model:
   return True
  if self.misses<=0:
   return False
  moment=monotonic()if at is None else at
  return moment-self.last_reply>DOWN_AFTER
 def usable(self,models,tools=False):
  if self.no_model or self.mute:
   return False
  blocked=self.dropped|self.no_tools if tools else self.dropped
  return any(m not in blocked for m in[*models,*self.extra_models])
 @staticmethod
 def _smc_flat(msgs):
  def flatten(one):
   out=dict(one)
   if isinstance(out.get("content"),list):
    out["content"]="\n\n".join(str(p.get("text")or"")for p in out["content"]if isinstance(p,dict))
   return out
  return[flatten(one)for one in msgs]
 @staticmethod
 def _smc_marked(msgs,model):
  if not model.startswith("anthropic/")or len(msgs)<=2:
   return msgs
  marked=list(map(dict,msgs))
  for end in(marked[0],marked[-1]):
   text=end.get("content")
   if isinstance(text,str)and text:
    end["content"]=[dict(type="text",text=text,cache_control=dict(type="ephemeral"))]
  return marked
 def _smc_give_up(self,kinds):
  if"empty"in kinds:
   verdict="empty"
  elif kinds=={"hard"}:
   verdict="refused"
  else:
   verdict=""
  self.call_fail=verdict
  return None
 def query_model(self,msgs,models,cap,tools=None):
  self.call_fail=""
  self.call_models=set()
  if self.mute or self.no_model:
   return None
  return _smc_ModelCall(self,msgs,models,cap,tools).run()
 def consult(self,prompt,cap,models):
  msg=self.query_model([{"role":"user","content":prompt}],models,cap)
  return json_object(msg.get("content")or""if msg else"")
 def act(self,acts):
  return _smc_WorldCall(self,acts).run()
def _smc_record_id(rec):
 for key,val in rec.items():
  if not isinstance(val,str):
   continue
  low=str(key).lower()
  if low in("id","_id")or(low.endswith("id")and re.fullmatch(ID_LIKE,val)):
   return val
 return None
def _smc_shelf_label(rec):
 for key,val in rec.items():
  if not isinstance(val,str):
   continue
  low,label=str(key).lower(),val.strip()
  if 0<len(label)<=60 and"id"not in low and re.search(r"categor|shelf|department|aisle",low):
   return label
 return""
def _smc_has_prop_map(rec):
 return any(isinstance(sub,dict)and sum(isinstance(v,(str,int,float))for v in sub.values())>=2
for sub in rec.values())
def _smc_carries(rec,ident):
 return any(isinstance(val,str)and val==ident and"id"in str(key).lower()for key,val in rec.items())
def _smc_price_field(rec):
 return next((float(val)for key,val in rec.items()if is_number(val)and re.search(r"price|cost",str(key).lower())),None)
class Records:
 def __init__(self):
  self.items=[]
  self.ids,self.read=set(),set()
  self.label_of={}
 def add(self,obs):
  self.items.append(obs)
  walk_dicts(obs,self._smc_take)
 def _smc_take(self,rec):
  self.ids.update(val for _key,val in _smc_id_pairs(rec))
  ident=_smc_record_id(rec)
  if not ident:
   return
  if _smc_has_prop_map(rec):
   self.read.add(ident)
  label=_smc_shelf_label(rec)
  if label:
   self.__dict__.setdefault("label_of",{}).setdefault(ident,label)
 def price(self,thing):
  newest=None
  def look(rec):
   nonlocal newest
   if _smc_carries(rec,thing):
    found=_smc_price_field(rec)
    if found is not None:
     newest=found
  for obs in self.items:
   walk_dicts(obs,look)
  return newest
 def about(self,thing,hold=4):
  found=[]
  for obs in self.items:
   stack=[(obs,0)]
   while stack:
    node,depth=stack.pop()
    if depth>6:
     continue
    if isinstance(node,list):
     below=node
    elif not isinstance(node,dict):
     continue
    elif thing in _smd_texts(node):
     shown=as_text(clipped(node,400))[:2500]
     if shown not in found:
      found.append(shown)
      if len(found)>=8:
       return found[-hold:]
     continue
    else:
     below=list(node.values())
    stack.extend((part,depth+1)for part in reversed(below))
  return found[-hold:]
 def _smd_each(self,take):
  for obs in self.items:
   walk_dicts(obs,take)
 def _smd_card_index(self):
  merged={}
  def take(rec):
   owner=next((cell for name,cell in _smd_named(rec)if isinstance(cell,str)and cell in self.ids
and"id"in name and re.fullmatch(ID_LIKE,cell)),None)
   if owner is None:
    return
   sheet=max(_smd_map_sizes(rec),default=0)>=2
   if sheet or len(rec)>=4:
    merged.setdefault(owner,{}).update(rec)
   if sheet:
    self.read.add(owner)
  self._smd_each(take)
  return merged
 def cards(self,char_cap=30000,cheaper_than="",skip=None,first=()):
  merged=self._smd_card_index()
  def price(owner):
   return next((float(cell)for name,cell in _smd_named(merged.get(owner,{}))
if is_number(cell)and _smd_price_key.search(name)),None)
  anchor=cheaper_than
  ceiling=price(anchor)if anchor else None
  kept=[]
  for owner in merged:
   if owner!=anchor:
    if skip and owner in skip:
     continue
    if ceiling is not None:
     paid=price(owner)
     if paid is not None and not paid<ceiling:
      continue
   kept.append(owner)
  lead=set(first or())
  line={o:("DETAILS "if o in self.read else"ROW ")+as_text(clipped(merged[o],160))for o in kept}
  short=sum(map(len,line.values()))>char_cap
  def rank(owner):
   gone=short and any(Errand.key_stock(k,v)is False for k,v in merged[owner].items())
   return owner not in lead,owner not in self.read,gone
  picked=[]
  room=char_cap
  for owner in sorted(kept,key=rank):
   if len(line[owner])<=room:
    room-=len(line[owner])
    picked.append(line[owner])
  return picked
 def label_for(self,d):
  pairs=list(_smd_named(d))
  for name,cell in pairs:
   if isinstance(cell,str)and _smd_shelf_key.search(name)and"id"not in name and 0<len(cell.strip())<=60:
    return cell.strip()
  carried=self.__dict__.get("label_of")or{}
  return next((carried[cell]for name,cell in pairs if isinstance(cell,str)and cell in carried and"id"in name),"")
 def _smd_read_id(self,rec):
  return next((cell for name,cell in _smd_named(rec)if isinstance(cell,str)and cell in self.read and"id"in name),None)
 def rich_reads(self,shelf=""):
  wanted=shelf.strip().lower()if shelf else""
  owners=set()
  def take(rec):
   owner=self._smd_read_id(rec)
   if not owner or(shelf and self.label_for(rec).strip().lower()!=wanted):
    return
   sizes=_smd_map_sizes(rec)
   if(max(sizes)>=3)if sizes else len(_smd_short_texts(rec,70))>=5:
    owners.add(owner)
  self._smd_each(take)
  return len(owners)
 def shelves_read(self):
  seen={}
  def take(rec):
   if self._smd_read_id(rec)and _smd_map_sizes(rec):
    label=self.label_for(rec)
    if label:
     seen[label]=seen.get(label,0)+1
  self._smd_each(take)
  return sorted(seen,key=lambda label:-seen[label])
 def prop_menu(self,shelf=""):
  wanted=shelf.strip().lower()if shelf else""
  def in_category(rec):
   labelled=(isinstance(cell,str)and _smd_shelf_key.search(name)and cell.strip().lower()==shelf.lower()
for name,cell in _smd_named(rec))
   return any(labelled)or self.label_for(rec).strip().lower()==wanted
  tally=_smd_PropTally(bool(shelf))
  for obs in self.items:
   stack=[(obs,False,0)]
   while stack:
    node,inside,depth=stack.pop()
    if depth>6:
     continue
    if isinstance(node,dict):
     inside=inside or not shelf or in_category(node)
     if inside:
      tally.take(node)
     below=[part for part in node.values()if isinstance(part,(dict,list))]
    elif isinstance(node,list):
     below=node
    else:
     continue
    stack.extend((part,inside,depth+1)for part in reversed(below))
  return tally.menu(),tally.records
 def label_lines(self):
  per_key={}
  def take(rec):
   if len(rec)>=4:
    for key,cell in _smd_short_texts(rec,40):
     if not(re.fullmatch(CODE_LIKE,cell)and re.search(r"\d",cell)):
      bucket=per_key.setdefault(str(key),{})
      bucket[cell]=bucket.get(cell,0)+1
  self._smd_each(take)
  out=[]
  for key,bucket in per_key.items():
   hits=list(bucket.values())
   if sum(hits)>=4 and 2<=len(hits)<=8 and max(hits)>=2:
    ranked=sorted(bucket.items(),key=lambda pair:-pair[1])
    out.append(f"- {key}: "+" | ".join(f"{cell} (x{n})"for cell,n in ranked))
    if len(out)==12:
     break
  return"\n".join(out)
_smd_shelf_key=re.compile(r"categor|shelf|department|aisle")
_smd_price_key=re.compile(r"price|cost")
_smd_money_words=re.compile(r"\bprice\b|budget|quantity|\bstock\b|\bcents?\b")
_smd_marks=re.compile(r"\((?:stated|revealed[^)]*)\)")
_smd_paren=re.compile(r"\(([^)]*)\)")
_smd_told_flags=("before_finish","needs_message","before_change")
_smd_told_marks=(("before_finish","before the finish"),("needs_message","by a message"))
def _smd_named(rec):
 return((str(k).lower(),cell)for k,cell in rec.items())
def _smd_texts(rec):
 return[cell for cell in rec.values()if isinstance(cell,str)]
def _smd_map_sizes(rec):
 return[len(cell)for cell in rec.values()if isinstance(cell,dict)]
def _smd_short_texts(rec,most):
 return[(k,cell)for k,cell in rec.items()if isinstance(cell,str)and 0<len(cell)<=most]
def _smd_units(value):
 return value if isinstance(value,int)and not isinstance(value,bool)and value>=1 else None
def _smd_alnum(text):
 return re.findall(r"[a-z0-9]+",text)
def _smd_letters(text):
 return re.findall(r"[a-z]+",text)
def _smd_squash(text):
 return" ".join(str(text or"").split())
def _smd_head(need):
 return need.partition(":")[0]
class _smd_PropTally:
 def __init__(self,scoped):
  self.scoped=scoped
  self.records=self.full=0
  self.hits,self.row_hits,self.values,self.samples={},{},{},{}
 def take(self,rec):
  short=[(k,cell)for k,cell in _smd_short_texts(rec,70)if not junk_key(k)]
  few=len(short)<5
  if few and not(self.scoped and len(short)>=2 and not _smd_map_sizes(rec)):
   return
  self.records+=1
  row=few and any("id"in name for name,cell in _smd_named(rec)if isinstance(cell,str))
  self.full+=not row
  for key,cell in short:
   if("-"in cell or"_"in cell)and re.search(r"\d",cell)and re.fullmatch(CODE_LIKE,cell):
    continue
   name=re.sub(r"[_.]+"," ",str(key)).strip().lower()
   low=cell.lower()
   self.hits[name]=self.hits.get(name,0)+1
   if row:
    self.row_hits[name]=self.row_hits.get(name,0)+1
   per_value=self.values.setdefault(name,{})
   per_value[low]=per_value.get(low,0)+1
   kept=self.samples.setdefault(name,[])
   if len(kept)<5 and low not in kept:
    kept.append(low)
 def spread(self,name):
  return len(self.samples.get(name,[]))
 def differs(self,name):
  return self.spread(name)>=2 and self.hits[name]>=2
 def measured(self,name):
  samples=self.samples.get(name)or[]
  with_digit=[s for s in samples if re.search(r"\d",s)]
  return bool(samples)and len(with_digit)==len(samples)and any(re.search(r"\d\s*[a-z\"']",s)for s in samples)
 def split_score(self,name):
  per_value=self.values.get(name)or{}
  total=sum(per_value.values())
  top=max(per_value.values())/total if total else 1.0
  return min(1.0,self.hits[name]/max(1,self.records))*(1.0-top)
 def menu(self):
  if self.full:
   for name in[n for n,c in self.hits.items()if self.row_hits.get(n,0)>=c]:
    del self.hits[name]
  order=sorted(self.hits,key=lambda n:(-self.hits[n],-self.split_score(n),n))
  out=[]
  for name in order[:28]:
   notes=f" [varies: {self.spread(name)} distinct values]"if self.differs(name)else""
   if self.measured(name):
    notes+=" [figures with a unit]"
   out.append(f"- {name} (in {self.hits[name]} listings){notes}: "+" | ".join(self.samples.get(name,[])))
  return"\n".join(out)
class Needs:
 def __init__(self,wire,request):
  self.wire,self.request,self.reading=wire,request,{}
  self.written,self.uncovered,self.indifferent,self.let_go=[],[],[],[]
  self.now_reqs=self.spend_cap=None
  self.version=0
  self.caught,self.exchanges,self.reply_due=set(),[],""
  self.widened_at,self.widened_tick=-1,-10**6
  self.sweepable=False
  self.resolved=[]
  self.ref_news=[]
  self.told=[]
  self.told_version=0
  self.allowance_news=None
 def add_told(self,items,source):
  if not isinstance(items,list):
   return[]
  held={_smd_squash(entry.get("text")).lower()for entry in self.told}
  fresh=[]
  for raw in items[:12]:
   given={"text":raw}if isinstance(raw,str)else raw
   if not isinstance(given,dict):
    continue
   text=_smd_squash(given.get("text"))
   if text and text.lower()not in held:
    held.add(text.lower())
    flags={flag:given.get(flag)if isinstance(given.get(flag),bool)else None for flag in _smd_told_flags}
    self.told.append(dict(text=text[:400],source=source,**flags))
    fresh.append(text[:120])
  self.told_version+=bool(fresh)
  return fresh
 def finish_told(self):
  return list(filter(lambda entry:entry.get("before_finish")is not False,self.told))
 @staticmethod
 def _smd_told_row(n,entry):
  tagged=[label for flag,label in _smd_told_marks if entry.get(flag)]
  where="from the "+str(entry.get("source"))
  return f"{n}) {entry['text']} ["+"; ".join([", ".join(tagged),where]if tagged else[where])+"]"
 def told_text(self):
  listed=[]
  for n,entry in enumerate(self.told[-10:],1):
   row=self._smd_told_row(n,entry)
   if self.__dict__.get("basket_told")and entry.get("before_finish")and entry.get("needs_message"):
    row+=(" [carried out: the cart as it stands was told to the shopper, whose reply took note of it without "
"objecting or changing anything; that reply is the approval asked for, so the order goes next]")
   listed.append(row)
  return"\n[instructions in force] "+" ".join(listed)if listed else""
 @staticmethod
 def _smd_product_row(row):
  if not isinstance(row,dict)or not str(row.get("product")or"").strip():
   return None
  asked=row.get("requirements")
  after=_smd_units(row.get("depends_on"))
  return{"product":str(row["product"]).strip(),"quantity":_smd_units(row.get("quantity"))or 1,
"requirements":[str(r)for r in(asked if isinstance(asked,list)else[])if r and":"in str(r)],
"depends_on":None if after is None else after-1}
 def items(self):
  found=[entry for entry in map(self._smd_product_row,self.reading.get("items")or[])if entry]
  return found if len(found)>1 else[]
 def quantity(self):
  return 1 if self.items()else _smd_units(self.reading.get("quantity"))or 1
 def counts(self):
  units=[entry["quantity"]for entry in self.items()]
  return units if units else[self.quantity()]
 def items_text(self):
  products=self.items()
  if not products:
   return""
  listed=" | ".join(f"{n}) {entry['product']} x{entry['quantity']}"for n,entry in enumerate(products,1))
  return(f"ITEMS IN THIS ORDER ({len(products)}): {listed}. Cart rule: a single line for each of them at its own "
"count, and no extra lines. Judge any listing solely by the needs of the item its category matches; a "
"lowest-price or top-option wish is settled per item, never across items.")
 def priority(self):
  rule=self.reading.get("priority")
  return rule if rule in("cheapest","best")else"none"
 def props_left(self):
  pool=self.written+self.uncovered if self.now_reqs is None else self.now_reqs
  return[line for line in pool if line not in self.let_go and not _smd_money_words.search(line.lower())]
 def text_body(self):
  if self.now_reqs is None:
   alive=[line for line in self.written if line not in self.let_go]
   heard=[line for line in self.uncovered if line not in self.let_go]
   parts=[f"{line} (stated)"for line in alive]+[f"{line} (revealed by the shopper)"for line in heard]
  else:
   parts=list(self.now_reqs)
  pieces=["; ".join(parts)if parts else
"nothing left to match on properties (all were dropped): any in-stock listing of the asked category is acceptable"if self.let_go else
"see the request"]
  if self.spend_cap is not None:
   pieces.append(f"budget, whole order, upper limit: {self.spend_cap} {self.reading.get('currency') or ''}".rstrip())
  pieces.append(self.items_text()if self.items()else f"quantity wanted: {self.quantity()}")
  if self.let_go:
   pieces.append("dropped by the shopper: "+"; ".join(self.let_go))
  if self.indifferent:
   pieces.append("no preference on: "+", ".join(self.indifferent))
  return"; ".join(pieces)
 def changes(self):
  if not self.let_go:
   return""
  return("\n\nLATER DROPS (the shopper withdrew these after writing the request; they beat its text): "
+"; ".join(self.let_go)+". Withdrawn means void: a listing is neither rejected nor ranked nor preferred "
"on any of them.")
 def overview(self,step_count,msgs):
  got=self.reading
  if not got:
   return""
  products=self.items()
  def bullets(key,lead):
   return[f"- {lead}: {entry}"for entry in got.get(key)or[]]
  out=["REQUEST NOTES (a draft parse; wherever it and the request text disagree, the text wins)"]
  if got.get("product"):
   out.append(f"- product: {got.get('product')}")
  if products:
   out+=[f"- item {n}/{len(products)}: {entry['product']} | count {entry['quantity']} | gets its own cart row"
for n,entry in enumerate(products,1)]
   out.append("- cart: nothing beyond these item lines; with a list-style cart tool, resend the full set each "
"time it is called")
  else:
   out.append(f"- quantity: {self.quantity()}")
  out+=bullets("requirements","firm need")
  out+=bullets("references","need tied to another listing (locate it, read the named field, and use what it "
"says as the search value)")
  if got.get("price_cap")is not None:
   out.append(f"- spend ceiling: {got.get('price_cap')} {got.get('currency') or ''}".rstrip())
  if got.get("in_stock_only"):
   out.append("- availability: currently stocked goods")
  rule=self.priority()
  if rule=="none":
   out.append("- choice rule: none given. Any pick that clears all needs within budget is correct; searching "
"on for a cheaper one burns steps, unless the request text demands it")
  else:
   detail=f" ({got.get('priority_detail')})"if got.get("priority_detail")else""
   out.append(f"- pick: the {rule}{detail} fit")
  out+=bullets("soft_wants","soft wish, tie-break only")
  out+=bullets("ask_first","ask the shopper first")
  out+=bullets("rules","rule")
  if msgs is not None:
   out.append(f"- message allowance: {msgs}")
  out.append(f"- step allowance: {step_count}")
  return"\n".join(out)
 @staticmethod
 def line_names(needs,lns):
  heard=_smd_letters(" ".join(lns).lower())
  candidates=(n for n in needs if isinstance(n,str)and":"in n)
  found=((n,Needs._smd_long_form(n,heard," ".join(heard)))for n in candidates)
  return{n:longer for n,longer in found if longer is not None}
 @staticmethod
 def _smd_long_form(need,words,spoken):
  prop,_,value=need.partition(":")
  cut=prop.find("(")
  tag=prop[cut:].strip()if cut>=0 else""
  short=_smd_letters(re.split(r"\s*\(",prop)[0].lower())
  if not short or min(map(len,short))<3 or re.search(r"\b"+r"\s+".join(short)+r"\b",spoken):
   return None
  width=len(short)
  for start in range(len(words)-width+1):
   run=words[start:start+width]
   if run!=short and all(map(str.startswith,run,short)):
    return" ".join(run)+(" "+tag if tag else"")+":"+value
  return None
 def drop_named(self,old_need,lns):
  try:
   prop=str(old_need).split(":",1)[0]
   case={"need":old_need,"prop":prop,"tags":_smd_paren.findall(prop)}
   for check in(self._smd_drop_unheld,self._smd_drop_named,self._smd_drop_by_field):
    verdict=check(case,lns)
    if verdict is not None:
     return verdict
   return False
  except Exception:
   return True
 @staticmethod
 def _smd_same_need(other,case):
  prop=_smd_head(other)
  return same_prop(prop,case["prop"])and(not case["tags"]or _smd_paren.findall(prop)in([],case["tags"]))
 def _smd_drop_unheld(self,case,lns):
  held=[line for line in self.written if self._smd_same_need(line,case)]
  if not held:
   fresh=[n for n in self.__dict__.get("fresh_needs")or[]if isinstance(n,str)and":"in n]
   held=[line for line in self.uncovered
if self._smd_same_need(line,case)and line not in self.written and line not in fresh]
   if not held or any(self._smd_same_need(n,case)and n!=case["need"]for n in fresh):
    return True
  case["held"]=held
  return None
 @staticmethod
 def _smd_drop_named(case,lns):
  words=_smd_letters(" ".join(lns).lower())
  runs={" ".join(words[i:i+size])for size in(1,2,3)for i in range(len(words)-size+1)}
  case["phrases"]={run for run in runs if len(run)>=3 and run not in LINK_WORDS}
  props=[_smd_head(line)for line in case["held"]]
  return True if any(same_prop(prop,run)for prop in props for run in case["phrases"])else None
 def _smd_drop_by_field(self,case,lns):
  hook=self.__dict__.get("field_hook")
  if not hook:
   return None
  for line in case["held"]:
   prop,_,raw=line.partition(":")
   value=_smd_marks.sub(" ",raw)
   if ref_text(value)or not ch_alts(value):
    continue
   fields=hook(prop)or{}
   rivals=[_smd_head(r)for r in self.written+self.uncovered
if isinstance(r,str)and":"in r and not same_prop(_smd_head(r),case["prop"])]
   for run in case["phrases"]:
    if any(strict_prop(rival,run)for rival in rivals):
     continue
    if any(strict_prop(field,run)and value_seen(value," | ".join(shown))for field,shown in fields.items()):
     self.wire.log.append({"kind":"drop_by_field","need":line[:200],"phrase":run})
     return True
  return None
 @staticmethod
 def _smd_store_spelling(need,hook):
  prop,_,value=need.partition(":")
  if re.search(r"\d",value)or ref_text(value):
   return None
  low=value.lower()
  bare=" ".join(_smd_alnum(re.sub(CJ_COMPARE," ",low)))
  if not bare or(bare==" ".join(_smd_alnum(low))and positive_value(value,prop)):
   return None
  name=re.split(r"\s*\(",prop)[0].strip()
  for field,shown in(hook(prop)or{}).items():
   if strict_prop(field,name):
    match=next((s for s in shown if" ".join(_smd_alnum(str(s).lower()))==bare),None)
    if match is not None:
     LITERAL_VALUES.add(bare)
     return f"{prop.strip()}: {str(match).strip().lower()}"
  return None
 def literal_values(self,answer):
  hook=self.__dict__.get("field_hook")
  if not hook:
   return
  spelled=[(n,self._smd_store_spelling(n,hook))for n in answer.get("needs")or[]if isinstance(n,str)and":"in n]
  swaps={n:alt for n,alt in spelled if alt is not None}
  if not swaps:
   return
  for field in("needs","requirements_now"):
   seq=answer.get(field)or[]if field=="needs"else answer.get(field)
   if field=="needs"or isinstance(seq,list):
    answer[field]=[swaps.get(x,x)if isinstance(x,str)else x for x in seq]
  self.wire.log.append({"kind":"literal_value","needs":list(swaps.values())[:4]})
 def _smd_pins(self,req,asked,tags,asked_key,refs):
  if req in self.let_go or":"not in req:
   return False
  prop,_,value=req.partition(":")
  own_tags=_smd_paren.findall(prop)
  if tags and own_tags and not kind_words(tags[-1])&kind_words(own_tags[-1]):
   return False
  if story_guess(req,self.request,refs)if positive_value(value,prop)else True:
   return False
  own_key=prop_key(prop)
  return bool(same_prop(prop,asked)or(len(own_key)>=4 and asked_key.endswith(own_key)))
 def pinned(self,prop_name,qtext):
  try:
   given=self.reading.get("references")
   refs=list(map(str,given))if isinstance(given,list)else[]
   refs.extend(self.__dict__.get("resolved")or[])
   tags=_smd_paren.findall(str(prop_name))
   asked_key=prop_key(prop_name)
   return any(self._smd_pins(req,prop_name,tags,asked_key,refs)for req in self.written)
  except Exception:
   return False
 @staticmethod
 def valued(needs,lns):
  said=" ".join(map(str,lns)).lower()
  def spelled_out(need):
   prop,_,value=str(need).partition(":")
   wanted=[w for w in prop_words(prop)if len(w)>=3]
   given=[w for w in _smd_alnum(value.lower())if w not in LINK_WORDS]
   return bool(wanted and given)and all(w in said for w in wanted+given)
  return any(map(spelled_out,needs if isinstance(needs,list)else()))
 def hear(self,heard,asked):
  turn=_sme_Turn(heard,asked)
  stage="fresh"
  while stage:
   stage=self._sme_hear_steps[stage](self,turn)
  return turn.note
 def _sme_log(self,kind,**fields):
  self.wire.log.append(dict(kind=kind,**fields))
 def _sme_hear_fresh(self,turn):
  turn.fresh=[said for said in turn.lines if said not in self.caught]
  if not turn.fresh:
   return""
  self.caught.update(turn.fresh)
  return""if self.wire.spare()<CLOSING else"consult"
 def _sme_hear_consult(self,turn):
  turn.so_far=self.text_body()
  parts=[HEAR,"ORIGINAL ASK (BACKGROUND):\n"+self.request,"REQUIREMENTS SO FAR:\n"+turn.so_far]
  history=self.exchanges[-12:]
  if history:
   parts.append("EARLIER BACK-AND-FORTH, OLDEST ON TOP:\n"+"\n".join(history))
  if turn.qtext:
   parts.append("THE ASSISTANT'S MESSAGE:\n"+turn.qtext)
  parts.append("SHOPPER'S NEWEST WORDS:\n"+"\n".join(turn.fresh))
  entry="shopper: "+" / ".join(turn.fresh)
  self.exchanges.append(f"asked: {turn.qtext} -> {entry}"if turn.qtext else entry)
  wire=self.wire
  turn.answer=wire.consult("\n\n".join(parts),2500,LISTENER)
  if turn.answer:
   return"meta"
  if wire.mute or wire.misses>0:
   parked=list(self.__dict__.get("pending")or[])
   self.caught.difference_update(turn.fresh)
   del self.exchanges[-1]
   parked.append((list(turn.fresh),turn.qtext))
   self.pending=parked
   self.pending_at=monotonic()
  return""
 def _sme_hear_meta(self,turn):
  try:
   got=turn.answer
   if got.get("meta_only")is True and turn.qtext and not self.valued(got.get("needs"),turn.fresh):
    turn.answer=dict(got,**{field:blank()for field,blank in _SME_QUIET_FIELDS})
    self.meta_last=turn.qtext
    self._sme_log("meta_only",line=" / ".join(turn.fresh)[:200])
   elif self._sme_lost_line(got,turn):
    turn.answer=dict(got,**{field:blank()for field,blank in _SME_QUIET_FIELDS})
    self.lost_last=turn.qtext
    self._sme_log("not_understood",line=" / ".join(turn.fresh)[:200])
  except Exception:
   pass
  return"rename"
 def _sme_lost_line(self,got,turn):
  if not turn.qtext or self.valued(got.get("needs"),turn.fresh):
   return False
  if got.get("not_understood")is True:
   return True
  said=" ".join(turn.fresh)
  return bool(_SME_LOST.search(said))and not(got.get("needs")or got.get("dropped")or is_number(got.get("new_budget")))
 def _sme_hear_rename(self,turn):
  try:
   swaps=self.line_names(turn.answer.get("needs")or[],turn.fresh)
  except Exception:
   swaps={}
  if not swaps:
   return"mask"
  got=turn.answer
  def swapped(seq):
   return[swaps.get(x,x)if isinstance(x,str)else x for x in seq]
  got["needs"]=swapped(got.get("needs")or[])
  if isinstance(got.get("requirements_now"),list):
   got["requirements_now"]=swapped(got["requirements_now"])
  self._sme_log("need_renamed",names=list(swaps.values())[:4])
  return"mask"
 def _sme_hear_mask(self,turn):
  turn.masked=turn.fresh
  reader=self.__dict__.get("ref_hook")
  if reader:
   try:
    turn.masked=reader(turn.answer,turn.fresh,turn.qtext)or turn.fresh
   except Exception as err:
    self._sme_log("ref_line_error",error=_sme_err(err))
    turn.masked=turn.fresh
  return"record"
 def _sme_hear_record(self,turn):
  turn.said=" ".join([turn.so_far," ".join(self.exchanges),turn.qtext or""]).lower()
  try:
   self.literal_values(turn.answer)
  except Exception as err:
   self._sme_log("literal_value_error",error=_sme_err(err))
  got=turn.answer
  self.fresh_needs=[x for x in got.get("needs")or[]if isinstance(x,str)]
  rebuilt=got.get("requirements_now")
  if isinstance(rebuilt,list)and all(isinstance(x,str)for x in rebuilt):
   self.now_reqs=[x for x in rebuilt if x.strip()and _sme_backed(x,turn.said)]
  for fresh_need in got.get("needs")or[]:
   if isinstance(fresh_need,str)and fresh_need and fresh_need not in self.uncovered and _sme_backed(fresh_need,turn.said):
    self.uncovered.append(fresh_need)
    turn.remarks.append("need added: "+fresh_need)
  try:
   self._sme_queue_pointers(turn)
  except Exception:
   pass
  return"free"
 def _sme_queue_pointers(self,turn):
  got=turn.answer
  pointed=got.get("references")
  infos=[ref for ref in pointed if isinstance(ref,dict)]if isinstance(pointed,list)else[]
  for fresh_need in got.get("needs")or[]:
   if not isinstance(fresh_need,str)or fresh_need not in self.uncovered:
    continue
   if ref_text(fresh_need.partition(":")[2])or fresh_need in(got.get("ref_needs")or[]):
    queued=list(self.__dict__.get("ref_news")or[])
    queued.append({"need":fresh_need,"infos":infos,"line":" ".join(turn.fresh)})
    self.ref_news=queued
 def _sme_hear_free(self,turn):
  in_force=self.written+self.uncovered if self.now_reqs is None else self.now_reqs
  ruled=set()
  for req in in_force:
   if isinstance(req,str)and":"in req and req not in self.let_go:
    ruled.add(req.partition(":")[0].strip().lower())
  for freed in turn.answer.get("no_preference")or[]:
   if not isinstance(freed,str)or not freed or freed in self.indifferent or freed.strip().lower()in ruled:
    continue
   if not self.pinned(freed,turn.qtext):
    self.indifferent.append(freed)
   else:
    self.pinned_heard=self.__dict__.get("pinned_heard",0)+1
  return"drop"
 def _sme_hear_drop(self,turn):
  got=turn.answer
  for gone in got.get("dropped")or[]:
   if not isinstance(gone,str)or not gone or gone in self.let_go:
    continue
   if is_number(got.get("new_budget"))and _SME_PRICE_TALK.search(gone.lower()):
    continue
   if self.drop_named(gone,turn.masked):
    self.let_go.append(gone)
    turn.remarks.append("need dropped: "+gone)
    turn.widened=turn.widened or self.priority()!="none"
   else:
    self._sme_log("drop_kept",need=gone[:200])
  turn.cap=got.get("new_budget")
  if is_number(turn.cap)and turn.cap>0 and turn.cap!=self.spend_cap:
   self.spend_cap=turn.cap
   turn.remarks.append(f"budget is now {turn.cap}")
  if got.get("unavailable"):
   turn.remarks.append("one product is no longer for sale - remove that line, and never check it out")
  try:
   self._sme_take_rules(turn)
  except Exception as err:
   self._sme_log("told_error",error=_sme_err(err))
  if turn.remarks:
   self.version+=1
   if turn.widened:
    self.widened_at,self.widened_tick=self.version,self.wire.tick
  return"widen"if self.now_reqs is None else"restore"
 def _sme_take_rules(self,turn):
  got=turn.answer
  added=self.add_told(got.get("instructions"),"store"if turn.qtext is None else"shopper")
  if added:
   turn.remarks.append("instruction added: "+"; ".join(added))
  limits=got.get("allowance")
  figures={k:v for k,v in limits.items()if is_number(v)}if isinstance(limits,dict)else{}
  if figures:
   self.allowance_news=figures
   turn.remarks.append("limit changed: "+json.dumps(figures))
 def _sme_hear_restore(self,turn):
  for revealed in self.uncovered:
   prop=revealed.partition(":")[0].strip().lower()
   if prop:
    listed=any(prop in req.lower()for req in self.now_reqs)
    released=any(prop in old.lower()for old in self.let_go)
    if not listed and not released:
     self.now_reqs.append(revealed)
  for stated in self.written:
   try:
    self._sme_restore_stated(stated,turn.masked)
   except Exception:
    continue
  return"widen"
 def _sme_restore_stated(self,stated,masked):
  if stated in self.let_go or stated in self.now_reqs or self.drop_named(stated,masked):
   return
  slot=_sme_head_tags(stated)
  def same_slot(other):
   head,tags=_sme_head_tags(other)
   return bool(same_prop(slot[0],head))and tags==slot[1]
  if any(same_slot(old)for old in self.let_go):
   return
  if not any(same_slot(req)for req in self.now_reqs if isinstance(req,str)):
   self.now_reqs.append(stated)
 def _sme_hear_widen(self,turn):
  if turn.widened:
   if self.props_left():
    turn.remarks.append(_SME_WIDER_NOTE)
   elif self.priority()=="cheapest"and self.sweepable:
    turn.remarks.append(_SME_OPEN_SHELF+_SME_SHELF_SWEPT+_SME_SHELF_SELF+_SME_SHELF_LAST)
   else:
    turn.remarks.append(_SME_OPEN_SHELF+_SME_SHELF_SELF+_SME_SHELF_BUY)
  return"close"
 def _sme_hear_close(self,turn):
  got=turn.answer
  sizes={key:len(got.get(field)or[])for key,field in(("needs","needs"),("free","no_preference"),
("dropped","dropped"))}
  self._sme_log("heard",question=turn.qtext is not None,**sizes,budget=is_number(turn.cap))
  asked_back=turn.qtext is None and any("?"in said for said in turn.fresh)and not got.get("no_preference")
  self.reply_due=str(got.get("reply")or"").strip()if asked_back else""
  turn.note="Meaning: "+str(got.get("meaning")or"")
  if turn.remarks:
   turn.note+=" ("+"; ".join(turn.remarks)+")"
  return""
 _sme_hear_steps={"fresh":_sme_hear_fresh,"consult":_sme_hear_consult,"meta":_sme_hear_meta,
"rename":_sme_hear_rename,"mask":_sme_hear_mask,"record":_sme_hear_record,
"free":_sme_hear_free,"drop":_sme_hear_drop,"restore":_sme_hear_restore,
"widen":_sme_hear_widen,"close":_sme_hear_close}
_SmeTexts=list[str]
_SmeArgs=dict[str,Any]
_SmeLine=tuple[str,float|None]
_SmeLines=list[_SmeLine]
class _sme_Turn:
 def __init__(self,lines,qtext):
  self.lines,self.qtext=lines,qtext
  self.fresh:list[str]=[]
  self.so_far=self.said=self.note=""
  self.answer:dict[str,Any]={}
  self.masked:list[str]=[]
  self.remarks:list[str]=[]
  self.widened=False
  self.cap:Any=None
_SME_QUIET_FIELDS=(("needs",list),("no_preference",list),("dropped",list),("references",list),
("requirements_now",lambda:None),("new_budget",lambda:None),("unavailable",bool),("reply",str))
_SME_PRICE_TALK=re.compile(r"\b(?:prices?|budget|costs?|spend\w*|cents?|dollars?)\b")
_SME_LOST=re.compile(r"lost me|confus|muddled|baffl|puzzl|stumped|didn.?t (?:follow|catch|get|understand)|did not (?:follow|catch|understand)|not (?:quite )?follow|(?:in )?(?:simpler|plainer|plain) (?:words|terms|talk|english|version)|(?:explain|say|put|run) (?:that|it)? ?(?:by me )?(?:again|another way|differently)|put it another way|come again|what (?:do|did) you mean|over my head|not with you|slow down|say that again|try (?:that )?again",re.I)
_SME_OPEN_SHELF=("SHELF OPEN - with no property requirement in force, FITS = anything in stock under the requested "
"category, accessories, parts and sets placed there included, and the single lowest price over the full "
"shelf takes it. ORDER OF ROWS = relevance or id, so price order can never be assumed")
_SME_SHELF_SWEPT=("; more slices of that shelf get fetched for you automatically (pages of the category listing, the "
"label's words, object-kind words found in titles); if no '[shelf sweep done]' list turns up after this")
_SME_SHELF_SELF=("; cover it yourself: page 1 of brief searches for the category label, for each single kind word in the "
"label and for each item name repeated across titles, plus the category listing's pages.")
_SME_SHELF_LAST=(" FINISH = cart that shelf's cheapest affordable row still in stock as the only line, confirm its stock "
"and price once more, then check out")
_SME_SHELF_BUY=" FINISH = order whichever in-stock item costs least"
_SME_WIDER_NOTE=("WIDER NOW - that requirement is gone, so a listing rejected only for it qualifies again, and "
"listings not yet seen might undercut the present choice. NEXT = search afresh for that kind of item, "
"leaving it out: one brief query per value still required, the value leading and the kind word after "
"it, all pages read; then the cheapest of every qualifier wins")
def _sme_err(err):
 return f"{type(err).__name__}: {err}"[:300]
def _sme_backed(req,said):
 long_words=re.findall(r"[a-z]{3,}",req.lower())
 return not long_words or any(tok in said for tok in long_words)
def _sme_head_tags(req):
 head=req.partition(":")[0]
 return head,re.findall(r"\(([^)]*)\)",head)
def _sme_kind(spec):
 return spec.get("type")if isinstance(spec,dict)else None
def _sme_lead(found):
 for entry in found:
  if entry[1]is None or entry[1]>0:
   return entry
 return found[0]if found else None
def _sme_qty(qty):
 return int(qty)if float(qty).is_integer()else qty
_SME_COUNT_NAME=re.compile(r"qty|quant|count|units|number|total|held|ordered")
_SME_MONEY_NAME=re.compile(r"price|cost|due|sum|fee|charge")
class CartBook:
 def __init__(self,tool,schema):
  self.tool=tool
  self._sme_schema=(schema or{}).get("properties")or{}
  self.put_form=self.adds=self.last_view=None
  self.used=False
  self.changed_at,self.viewed_at=0,-1
  self.units:dict[str,float]={}
  self._sme_if_set:dict[str,float]={}
 def _sme_enum_arg(self,key):
  spec=self._sme_schema.get(key)
  return isinstance(spec,dict)and"enum"in spec
 def form(self,args):
  return as_text({key:args[key]for key in args if self._sme_enum_arg(key)})
 def line_spec(self):
  for arg,spec in self._sme_schema.items():
   row=spec.get("items")if _sme_kind(spec)=="array"else None
   fields=row.get("properties")if isinstance(row,dict)else None
   if not isinstance(fields,dict):
    continue
   texts=[f for f,sub in fields.items()if _sme_kind(sub)=="string"and"enum"not in sub]
   keyed=[f for f in texts if"id"in f.lower()]
   amounts=[f for f,sub in fields.items()if _sme_kind(sub)in("integer","number")]
   if(keyed or texts)and amounts:
    return arg,(keyed or texts)[0],amounts[0]
  return None
 def _sme_rows_arg(self,args):
  spec=self.line_spec()
  if spec and isinstance(args.get(spec[0]),list):
   return spec
  return None
 def lines(self,args):
  spec=self._sme_rows_arg(args)
  if spec is None:
   target=self.aim_flat(args)
   return[(target,self.count_arg(args))]if target else[]
  arg,id_key,qty_key=spec
  picked=[]
  for entry in args[arg]:
   label=entry.get(id_key)if isinstance(entry,dict)else None
   if isinstance(label,str)and label.strip():
    amount=entry.get(qty_key)
    picked.append((label.strip(),float(amount)if is_number(amount)else None))
  return picked
 def aim(self,args):
  first=_sme_lead(self.lines(args))
  return None if first is None else first[0]
 def aim_flat(self,args):
  for key,value in args.items():
   if isinstance(value,str)and not self._sme_enum_arg(key):
    word=value.strip()
    if word and" "not in word:
     return value
  return None
 def count_key(self):
  for key,spec in self._sme_schema.items():
   if _sme_kind(spec)in("integer","number")and"enum"not in spec:
    return key
  return None
 def count_arg(self,args):
  if self._sme_rows_arg(args):
   first=_sme_lead(self.lines(args))
   return None if first is None else first[1]
  for key,spec in self._sme_schema.items():
   if _sme_kind(spec)in("integer","number")and"enum"not in spec:
    amount=args.get(key,spec.get("default"))
    if is_number(amount):
     return float(amount)
  return None
 def is_put(self,args):
  if self.put_form is None:
   return True
  return self.form(args)==self.put_form
 def held(self,label):
  return self.units.get(label,0.0)
 def booked(self,args,reply):
  self.used=True
  if isinstance(reply,dict)and reply.get("error"):
   return""
  if self.put_form is None:
   self.put_form=self.form(args)
  book=self._sme_book_basket if self._sme_rows_arg(args)else self._sme_book_line
  return book(args,reply)
 def _sme_book_basket(self,args,reply):
  self.adds=False
  basket={}
  for label,amount in self.lines(args):
   if amount is None:
    basket[label]=basket.get(label,0.0)+1.0
   elif amount>0:
    basket[label]=basket.get(label,0.0)+amount
  self.units=basket
  shown=self.view_units(reply)
  if shown is not None:
   self.learn(shown)
  return self.summary()
 def _sme_book_line(self,args,reply):
  label=self.aim(args)
  if not label:
   return""
  amount=self.count_arg(args)
  if amount is None:
   amount=1.0
  tally=self.units
  had=tally.get(label,0.0)
  putting=self.is_put(args)
  if not putting:
   tally[label]=max(0.0,had-amount)if amount>0 else 0.0
  elif had>0 and self.adds is not False:
   tally[label]=had+amount
   self._sme_if_set={label:amount}
  else:
   tally[label]=amount if amount>0 else 0.0
  if tally.get(label,0)<=0:
   del tally[label]
  shown=self.view_units(reply)
  if shown is not None:
   if set(shown)<={label}:
    self.settle(label,shown.get(label,0.0))
   else:
    self.learn(shown)
  elif(amount<=0 or not putting)and self.shows_empty(reply,label):
   self.units.pop(label,None)
   self._sme_if_set.pop(label,None)
  return self.summary()
 def shows_empty(self,reply,label):
  if not isinstance(reply,dict)or label not in self.units:
   return False
  has_list=has_figure=False
  for value in reply.values():
   if isinstance(value,list):
    if value:
     return False
    has_list=True
   elif is_number(value):
    if value!=0:
     return False
    has_figure=True
  if not(has_list and has_figure):
   return False
  text=as_text(reply)
  return all(other==label or other not in text for other in self.units)
 def view_units(self,reply):
  known=set(self.units)|set(self._sme_if_set)
  found={}
  walk_dicts(reply,lambda rec:self._sme_read_row(rec,known,found))
  if not found or(known and not known&set(found)):
   return None
  return found
 def _sme_read_row(self,rec,known,found):
  labels=[v for v in rec.values()if isinstance(v,str)and" "not in v and(v in known or re.fullmatch(ID_LIKE,v))]
  if not labels:
   return
  label=next((v for v in labels if v in known),labels[0])
  amount=self._sme_units_field(rec,label)
  if amount is not None:
   found[label]=found.get(label,0)+amount
 def _sme_units_field(self,rec,label):
  cand=[(k,v)for k,v in rec.items()if is_number(v)and isinstance(v,int)and 0<v<1000 and str(k)!="price"]
  if len(cand)>1:
   named=[(k,v)for k,v in cand
if _SME_COUNT_NAME.search(str(k).lower())and not _SME_MONEY_NAME.search(str(k).lower())]
   if len(named)==1:
    cand=named
  if len(cand)>1:
   counted=self.units.get(label,self._sme_if_set.get(label))
   equal=[(k,v)for k,v in cand if counted is not None and abs(v-counted)<1e-9]
   cand=equal if len(equal)==1 else[min(cand,key=lambda kv:kv[1])]
  return cand[0][1]if len(cand)==1 else None
 def _sme_judge_adds(self,label,shown,if_set):
  near=abs(shown-if_set)<1e-9
  if near and abs(self.units.get(label,0)-if_set)>1e-9:
   self.adds=False
  elif shown>if_set:
   self.adds=True
 def settle(self,label,amount):
  if label in self._sme_if_set:
   self._sme_judge_adds(label,amount,self._sme_if_set.pop(label))
  tally=self.units
  if amount>0:
   tally[label]=amount
  elif label in tally:
   del tally[label]
 def learn(self,shown):
  for label in[x for x in self._sme_if_set if x in shown]:
   self._sme_judge_adds(label,shown[label],self._sme_if_set[label])
  self._sme_if_set={}
  self.units={label:amount for label,amount in shown.items()if amount>0}
 def viewed(self,reply,tick):
  self.last_view,self.viewed_at=reply,tick
  shown=self.view_units(reply)
  if shown is not None:
   self.learn(shown)
  elif isinstance(reply,(dict,list))and not self.ids_in(reply):
   self.units={}
 def ids_in(self,reply):
  text=as_text(reply)
  return any(label in text for label in self.units)
 def summary(self):
  parts=[f"{label} x{_sme_qty(amount)}"for label,amount in self.units.items()]
  return"[cart, counted by code] "+(", ".join(parts)if parts else"no lines")+"."
 def faults(self,wanted,per_product=None):
  tally=self.units
  if not tally:
   return[]
  if per_product and len(per_product)>1:
   return self._sme_product_faults(per_product)
  if len(tally)>1:
   return[f"{len(tally)} separate lines sit in your basket ({self._sme_listing()}); clear all but your pick"]
  (label,amount),=tally.items()
  gap=amount-wanted
  if not abs(gap)>1e-9:
   return[]
  return[f"quantity off on {label}: {int(amount)} held, {wanted} needed - {'remove' if gap > 0 else 'add'} "
f"{int(abs(gap))} via the cart tool"]
 def _sme_listing(self):
  return", ".join(f"{label} x{int(amount)}"for label,amount in self.units.items())
 def _sme_product_faults(self,per_product):
  have=sorted(map(int,self.units.values()))
  if len(per_product)!=len(self.units):
   return[f"basket mismatch: {len(per_product)} products requested yet {len(self.units)} line(s) present "
f"({self._sme_listing()}); trim to a single entry per requested product"]
  want=sorted(per_product)
  if have==want:
   return[]
  return[f"line counts {have}, wanted {want}: correct them"]
def _sme_capped_messages(cap,sent,limits):
 whole,rest=limits.get("messages_total"),limits.get("messages_left")
 if is_number(whole):
  return max(sent,int(whole))
 if is_number(rest):
  return sent+max(0,int(rest))
 return cap
def _sme_capped_steps(cap,tick,limits):
 whole,rest=limits.get("steps_total"),limits.get("steps_left")
 if is_number(whole)and int(whole)>tick:
  return min(cap,int(whole))
 if is_number(rest)and int(rest)>=0:
  return min(cap,tick+int(rest))
 return cap
_SME_ERRAND_START=(
("role_map",dict),
("msgs_sent",int),
("msg_cap",lambda:None),
("orders_tried",int),
("refusal_n",int),
("order_vetted",bool),
("rank_checked",set),
("decline_vetted",bool),
("audit_done",bool),
("questions_log",list),
("asking",lambda:True),
("rulings",dict),
("checked_adds",set),
("passed",set),
("refused_lines",set),
("refusals_seen",dict),
("warned",set),
("word_seen",list),
("word_kind",str),
("lead_seen",list),
("lead_bounds",dict),
("lead_words",int),
("lead_ctx",lambda:None),
("last_calls",list),
("chat",list),
("stuck",int),
("refs",list),
)
class Errand:
 def __init__(self,payload):
  env=payload["environment"]
  wire=self.wire=Wire(env["binding"])
  view=self.pview=env["policy_view"]
  self.request=str(view.get("query")or"")
  self.toolset=[entry for entry in view.get("tools")or[]
if isinstance(entry,dict)and isinstance(entry.get("function"),dict)]
  shapes={}
  for entry in self.toolset:
   fn=entry["function"]
   shapes[fn["name"]]=fn.get("parameters")or{"type":"object","properties":{}}
  self.shapes=shapes
  self.step_cap=int(view.get("max_steps")or 30)
  wire.done=bool(view.get("finished"))
  self.archive=Records()
  ledger=self.needs=Needs(wire,self.request)
  ledger.ref_hook=self.ref_fix
  for name,make in _SME_ERRAND_START:
   setattr(self,name,make())
  self.trolley=CartBook(None,{})
  LITERAL_VALUES.clear()
  STORE_TWINS.clear()
  ledger.field_hook=self.store_fields
 @property
 def log(self):
  return self.wire.log
 def moves_left(self):
  return self.step_cap-self.wire.tick
 def role(self,kind):
  return self.role_map.get(kind)
 def ask_limit(self):
  return ASK_DEFAULT if self.msg_cap is None else self.msg_cap
 def msg_reserve(self):
  return 1 if self.role("message")else 0
 def ask_room(self):
  return max(0,self.ask_limit()-self.msg_reserve())
 def step_reserve(self):
  keep=bool(self.role("message"))or self.needs.finish_told()
  return FINISH_STEPS if keep else 0
 def apply_allowance(self,limits):
  was=(self.msg_cap,self.step_cap)
  self.msg_cap=_sme_capped_messages(self.msg_cap,self.msgs_sent,limits)
  self.step_cap=_sme_capped_steps(self.step_cap,self.wire.tick,limits)
  caps=[self.msg_cap,self.step_cap]
  if tuple(caps)!=was:
   self.asking=self.asking and self.msgs_sent<self.ask_room()
   self.log.append({"kind":"allowance","news":limits,"caps":caps})
 def call_kind(self,name,args):
  ender,dropper=self.role("order"),self.role("decline")
  if not name or name not in(ender,dropper):
   return None
  if ender!=dropper:
   return"order"if name==ender else"decline"
  modes=[v.lower()for v in(args or{}).values()if isinstance(v,str)and len(v)<=40]
  for low in modes:
   for kind,pattern in(("decline",DECLINE_WORDS),("order",ORDER_WORDS)):
    if re.search(pattern,low):
     return kind
  return self._sme_choice_side(name,modes)or"order"
 def _sme_choice_side(self,name,modes):
  layout=(self.shapes.get(name)or{}).get("properties")or{}
  for spec in layout.values():
   options=[str(v).lower()for v in spec.get("enum")or[]]if isinstance(spec,dict)else[]
   for low in modes:
    if len(options)==2 and low in options:
     other=options[1-options.index(low)]
     if re.search(ORDER_WORDS,other):
      return"decline"
     if re.search(DECLINE_WORDS,other):
      return"order"
  return None
 def finishers(self):
  tools=set()
  for kind in("order","decline"):
   tool=self.role(kind)
   if tool is not None:
    tools.add(tool)
  return tools
 def clock_text(self):
  left=max(0.0,self.wire.spare()-LAST_CALL)
  if 2*left>DEADLINE-LAST_CALL:
   return""
  shown=CLOCK_STEP*int(left//CLOCK_STEP)
  return f"about {shown} s left"if shown>0 else f"under {CLOCK_STEP} s left"
 def _smf_guard(fallback):
  def wrap(method):
   def run(*args,**kwargs):
    try:
     return method(*args,**kwargs)
    except Exception:
     return fallback
   run.__name__,run.__qualname__,run.__doc__=method.__name__,method.__qualname__,method.__doc__
   return run
  return wrap
 def _smf_note(self,kind,**fields):
  self.wire.log.append({"kind":kind,**fields})
 def _smf_fault(self,kind,err):
  self.wire.log.append({"kind":kind,"error":f"{type(err).__name__}: {err}"[:300]})
 def status_line(self):
  left=self.moves_left()
  sent=f"messages sent {self.msgs_sent}"
  if self.msg_cap is not None:
   sent+=f" of {self.msg_cap}"
  counts=[f"[status] steps used {self.wire.tick} of {self.step_cap}, {left} left",sent]
  clock=self.clock_text()
  counts+=[clock]if clock else[]
  note="; ".join(counts)+".\n[requirements now] "
  self.needs.basket_told=self._smw_told_current()
  for piece in(self.needs.text_body,self.ref_line,self.product_steps,self.needs.told_text):
   note+=piece()
  try:
   for piece in(self.bound_hint,self.dependency_hint):
    note+=piece()
  except Exception:
   pass
  note+=self._smf_side_notes()
  stage,due=self._smf_urgency(left)
  return note+(self._smf_closing(stage,bool(self.needs.items()),due)if stage else"")
 def _smf_side_notes(self):
  out=""
  if self.msg_reserve()and self.msgs_sent<self.ask_limit():
   out+=(" (the planned questions leave one message for the stated finishing step: yours to send, or sent for "
"you at the finish)")
  offered=[spec["function"]["name"]for spec in self.offered()]
  if len(offered)<len(self.toolset):
   plan_note="; hidden-need questions are also planned and sent for you"if self.asking else""
   out+=f"\n[tools usable now] {', '.join(offered)} (any other tool call is refused without using a step{plan_note})."
  if self.trolley.used:
   out+="\n"+self.trolley.summary()
  out+=str(self.__dict__.get("list_cap_note")or"")
  return out
 def _smf_urgency(self,left):
  if left<=2:
   return"last",0
  if left<=5 or self.wire.spare()<CLOSING:
   return"finish",0
  if self.wire.spent_out():
   return"budget",0
  if self.orders_tried==0:
   due=self.order_due()
   gap=left-due
   if 0<gap<=7 and self.narrowing():
    return"low",due
   if gap<=0 and self.narrowing():
    return"closed",due
  return"",0
 @staticmethod
 def _smf_closing(stage,multi,due):
  fallback="a verified fit, else the cheapest in-stock, in-budget listing whose record shows no requirement failing"
  never=" Decline only when the records show that nothing can fit."
  if stage=="last":
   whom="every product asked for"if multi else"the item"
   return(f" LAST STEPS: order the cart now if it holds a candidate for {whom}; otherwise put the best "
f"candidate in ({fallback}) and order."+never)
  if stage=="finish":
   lines="one line per product asked for"if multi else"one line"
   return f" FINISH NOW: fix the cart to {lines} ({fallback}; a sold-out line out), re-check its stock, then order."+never
  if stage=="budget":
   whom="every product asked for (one line each, at its quantity)"if multi else"the item asked for"
   lack="some product has"if multi else"the store has"
   return(f" MODEL BUDGET USED: stop exploring; within the next few steps put the best candidate of {whom} in "
"the cart (a verified fit where one exists, else the closest fit the records show), re-check its stock "
f"and price, and order. A decline is right only when the records show that {lack} no listing that "
"can meet its requirements.")
  repair="a refused order needs about 4 steps to repair."
  if stage=="low":
   return(" Steps are running low: within the next few steps put your best verified fit in the cart and order; "
f"searching and reading details end when {due} steps remain, and {repair}")
  return(" Searching has ended (details only for the cart's line, listings a check named, or, while some product "
"has no verified fit, up to 3 more reads or searches in all): put your best verified fit in the cart, "
"re-check its stock and price, and order; "+repair.replace("needs","still needs"))
 def read_request(self):
  catalogue="\n".join(self._smf_tool_row(spec["function"])for spec in self.toolset)
  shown="REQUEST:\n"+self.request
  if self.pview.get("opening")is not None:
   shown+="\n\nOPENING OBSERVATION:\n"+as_text(self.pview.get("opening"))
  self.read_content=[dict(type="text",text=READ_REQUEST+"\n\nTOOLS:\n"+catalogue,
cache_control={"type":"ephemeral"}),
dict(type="text",text=shown)]
  ladder=READER+LISTENER[:1]
  walk={"state":"ask","turn":0,"got":{}}
  while walk["state"]!="done":
   walk["state"]=getattr(self,"_smf_read_"+walk["state"])(walk,ladder)
 @staticmethod
 def _smf_tool_row(fn):
  return(f"- {fn['name']}: {fn.get('description', '')} | arguments: "
f"{as_text(fn.get('parameters') or {})}")
 def _smf_read_ask(self,walk,ladder):
  turn=walk["turn"]
  if turn>=3*len(ladder):
   return"fallback"
  if turn>=len(ladder):
   wire=self.wire
   if(wire.misses==0 or wire.call_fail=="refused"or wire.down()or not wire.usable(ladder)
or wire.spare()<CLOSING+60):
    return"fallback"
  model=ladder[turn%len(ladder)]
  reply=self.wire.query_model([{"role":"user","content":self.read_content}],[model],READ_TOKENS)or{}
  walk["got"]=json_object(reply.get("content")or"")or{}
  if isinstance(walk["got"].get("roles"),dict):
   return"apply"
  self._smf_note("setup_retry",model=model)
  walk["turn"]=turn+1
  return"ask"
 def _smf_read_fallback(self,walk,ladder):
  try:
   got={**self.code_reading()}
   got.update((kname,val)for kname,val in walk["got"].items()if kname!="roles")
  except Exception as err:
   self._smf_fault("code_reading_error",err)
   got={}
  self.read_at=monotonic()
  self._smf_note("code_reading",roles=got.get("roles"),product=got.get("product"),
price_cap=got.get("price_cap"),items=len(got.get("items")or[]))
  walk["got"]=got
  return"apply"
 def _smf_read_apply(self,walk,ladder):
  self.use_reading(walk["got"])
  return"done"
 def use_reading(self,plan,again=False):
  faults={"_smf_seed_told":"told_seed_error","_smf_run_story":"story_values_error",
"_smf_run_refs":"ref_register_error","_smf_take_written_counts":"written_counts_error"}
  for step in("_smf_take_roles","_smf_take_counts","_smf_take_written_counts","_smf_seed_told","_smf_scale_products",
"_smf_split_written","_smf_run_story","_smf_run_refs","_smf_trim_asks","_smf_take_price",
"_smf_take_steps"):
   run,kind=getattr(self,step),faults.get(step)
   if kind is None:
    run(plan,again)
    continue
   try:
    run(plan,again)
   except Exception as err:
    self._smf_fault(kind,err)
 def _smf_run_story(self,plan,again):
  self.story_values(plan)
 def _smf_run_refs(self,plan,again):
  self.ref_register()
 def _smf_take_roles(self,plan,again):
  if not again:
   self.steps_given=self.step_cap
  cart_before=self.role("cart")
  offered=plan.get("roles")
  pairs=offered.items()if isinstance(offered,dict)else()
  self.role_map=dict((role,tool)for role,tool in pairs if isinstance(tool,str)and tool in self.shapes)
  if not(again and self.trolley.used):
   cart_tool=self.role("cart")
   self.trolley=CartBook(cart_tool,self.shapes.get(cart_tool or"",{}))
  elif cart_before:
   self.role_map["cart"]=cart_before
  self.search_role_fix()
  self.needs.sweepable=self.page_arg()is not None if self.role("search")else False
 def _smf_take_counts(self,plan,again):
  allowance=plan.get("message_budget")
  if is_number(allowance)and allowance>=0:
   self.msg_cap=int(allowance)
  self.needs.reading=plan
 _smf_NOT_COUNT=re.compile(r"(?:\b(?:of|x|the|by|per|than|under|over|to)|\u00d7)$")
 def _smf_written_count(self,product):
  words=re.findall(r"[a-z0-9]+",str(product or"").lower())
  if not words:
   return None
  head=re.escape(words[0][:-1]if len(words[0])>3 and words[0].endswith("s")else words[0])
  text,seen=self.request.lower(),set()
  for hit in re.finditer(r"(?<![\w.$-])"+Errand._smm_FIGURE+r"\s+"+head+r"(?:e?s)?\b",text):
   if not Errand._smf_NOT_COUNT.search(text[:hit.start()].rstrip()):
    seen.add(Errand._smm_figure(hit.group(1)))
  units=seen.pop()if len(seen)==1 else None
  return units if units is not None and 1<units<100 else None
 def _smf_take_written_counts(self,plan,again):
  if plan.get("code_reading"):
   return
  rows=[row for row in plan.get("items")or[]if isinstance(row,dict)]
  rows=rows if len(rows)>1 else[plan]
  raised=[]
  for row in rows:
   units=self._smf_written_count(row.get("product"))
   if units is not None and(_smd_units(row.get("quantity"))or 1)==1:
    row["quantity"]=units
    raised.append(f"{str(row.get('product'))[:40]} x{units}")
  if raised:
   self.log.append({"kind":"written_counts","raised":raised})
 def _smf_seed_told(self,plan,again):
  told=self.needs.add_told
  told(plan.get("instructions"),"request")
  told(list(filter(lambda rule:isinstance(rule,str),plan.get("rules")or[])),"request")
 def _smf_scale_products(self,plan,again):
  try:
   past_first=sorted((0,len(self.needs.items())-1,2))[1]
   self.wire.scale=1.0+0.3*past_first
  except Exception:
   self.wire.scale=1.0
 def _smf_split_written(self,plan,again):
  named=[str(entry)for entry in plan.get("requirements")or[]if entry]
  firm,stray=[],[]
  for entry in named:
   field,colon,value=entry.partition(":")
   (firm if colon and field.strip()and value.strip()else stray).append(entry)
  self.needs.written=firm
  if stray:
   plan["soft_wants"]=[str(wish)for wish in plan.get("soft_wants")or[]if wish]+stray
   plan["requirements"]=list(firm)
 def _smf_trim_asks(self,plan,again):
  late=[entry for entry in plan.get("instructions")or[]if isinstance(entry,dict)
and entry.get("before_finish")is True and entry.get("before_change")is not True]
  marks=[set(re.findall(r"[a-z]{4,}",str(entry.get("text")or"").lower()))for entry in late]
  marks=[mark for mark in marks if mark]
  if not marks:
   return
  kept,moved=[],[]
  for point in plan.get("ask_first")or[]:
   got=set(re.findall(r"[a-z]{4,}",str(point).lower()))
   shared=max(len(got&mark)for mark in marks)
   (moved if got and(shared>=2 or 2*shared>=len(got))else kept).append(point)
  if moved:
   plan["ask_first"]=kept
   self.log.append({"kind":"ask_first_to_finish","moved":[str(point)[:120]for point in moved]})
 def _smf_take_price(self,plan,again):
  if not again or self.needs.version<=0:
   self.needs.spend_cap=plan.get("price_cap")
  if not plan or self.needs.spend_cap is not None or not self.role("message"):
   return
  known=[str(item)for item in plan.get("ask_first")or[]if item]
  plan["ask_first"]=["the most the shopper wants to spend, since the request states no price limit (skip if the "
"same thing is listed again here)"]+known
 def _smf_take_steps(self,plan,again):
  if again:
   self.step_cap=self.__dict__.get("steps_given",self.step_cap)
  stated=plan.get("step_budget")
  if is_number(stated)and 0<stated<self.step_cap:
   self.step_cap=int(stated)
  thin=40.0/max(1,self.step_cap)
  self.wire.scale=self.wire.__dict__.get("scale",1.0)*min(1.0,max(0.5,thin))
  if again:
   self.needs.version+=1
 @staticmethod
 def _smf_bare(text):
  return re.split(r"\s*\(",str(text))[0].strip()
 @staticmethod
 def _smf_key(text_line):
  return text_line.strip("- ").split(" (in ")[0].strip()
 @staticmethod
 def _smf_norm(value):
  return" ".join(re.findall(r"[a-z0-9]+",value.lower()))
 @staticmethod
 def _smf_samples(text_line):
  return text_line.split("):",1)[1].split("|")
 @staticmethod
 def _smf_words(wanted):
  return[word for word in prop_words(wanted)if len(word)>=3 and word not in LINK_WORDS]
 def _smf_hints(self):
  return self.needs.reading.get("hints")or[]
 def _smf_hint_rows(self,pk,multi,loose):
  for clue in self._smf_hints():
   front,_,tail=str(clue).partition(":")
   if multi and not(loose and pk is None)and self.product_of(front)not in(pk,None):
    continue
   yield clue,front,tail
 def _smf_kept_menu(self,pk):
  kept=self.__dict__.get("menu_memo",{}).get(pk)
  return kept[1]if kept else""
 def _smf_menu_keys(self,pk):
  return[self._smf_key(text_line)for text_line in self._smf_kept_menu(pk).splitlines()]
 def _smf_home(self,pk):
  homes=self.home_shelves()
  if pk is None or pk>=len(homes):
   return set()
  return homes[pk]
 def _smf_spent(self):
  return self.__dict__.get("spent_asks",set())
 def story_values(self,plan):
  clues=[one_prop_hint(text)for text in self._smf_flat_hints(plan.get("hints"))]
  reader_hints=list(clues)
  refs=plan.get("references")
  refs=[str(ref)for ref in refs or[]]if isinstance(refs,list)else[]
  guesses=[rq for rq in self.needs.written if story_guess(rq,self.request,refs)]
  total=len(self.needs.written)
  if guesses and total>=2 and len(guesses)*2<=total:
   clues+=self._smf_demote(plan,guesses)
  try:
   promoted=self.written_hints(plan,reader_hints)
   clues=[clue for clue in clues if clue not in promoted]
  except Exception as err:
   self._smf_fault("written_hint_error",err)
  plan["hints"]=clues[:12]
 @staticmethod
 def _smf_flat_hints(source):
  def pairs_of(mapping):
   for subject,held in mapping.items():
    for val in held if isinstance(held,list)else(held,):
     if val:
      text=str(val)
      yield text if":"in text else f"{subject}: {val}"
  if isinstance(source,dict):
   source=list(pairs_of(source))
  kept=[]
  for item in source if isinstance(source,(list,tuple))else():
   if isinstance(item,(str,int,float))and str(item).strip():
    kept.append(str(item).strip())
  return kept
 def _smf_demote(self,plan,guesses):
  self.needs.written=[rq for rq in self.needs.written if rq not in guesses]
  plan["requirements"]=list(self.needs.written)
  dropped={self._smf_ident(rq)for rq in guesses}
  for entry in plan.get("items")or[]:
   if not isinstance(entry,dict)or not isinstance(entry.get("requirements"),list):
    continue
   entry["requirements"]=[rq for rq in entry["requirements"]if self._smf_ident(rq)not in dropped]
  pieces=[rq.partition(":")for rq in guesses]
  self._smf_note("story_values",moved=guesses)
  return[f"{subject.strip()}: the story suggests {said.strip()} (not stated as a value; ask)"
for subject,_,said in pieces]
 @staticmethod
 def _smf_ident(rq):
  field,value=(str(rq).split(":",1)+[""])[:2]
  return Errand._smf_bare(field).lower(),value.strip().lower()
 def written_hints(self,plan,clues):
  entries=[entry for entry in plan.get("items")or[]if isinstance(entry,dict)]
  promoted=[]
  for clue in clues:
   stated,kinds=self._smf_stated_form(plan,entries,str(clue))
   if not stated:
    continue
   self.needs.written.append(stated)
   plan["requirements"]=list(self.needs.written)
   home_row=next((entry for entry in entries if isinstance(entry.get("requirements"),list)
and(len(entries)==1 or kind_words(entry.get("product")or"")&kinds)),None)
   if home_row is not None:
    home_row["requirements"].append(stated)
   promoted.append(clue)
   self._smf_note("written_hint",hint=str(clue)[:200],requirement=stated)
  return promoted
 def _smf_stated_form(self,plan,entries,clue):
  subject,colon,tail=clue.partition(":")
  field=self._smf_bare(subject)
  if not colon or not field or junk_key(field):
   return"",set()
  tags=re.findall(r"\(([^)]*)\)",subject)
  owners=[tags[-1]]if tags else([str(entry.get("product")or"")for entry in entries]
or[str(plan.get("product")or"")])
  word,owner="",""
  for owner in owners:
   word=lead_value(tail,self.request,owner)
   if word:
    break
  if not word:
   return"",set()
  kinds=kind_words(owner)
  pattern=r"\b"+re.escape(word)+r"\b"
  verdicts=[]
  for rq in self.needs.written:
   rhead,_,rvalue=rq.partition(":")
   rtags=re.findall(r"\(([^)]*)\)",rhead)
   other_product=bool(rtags)and bool(kinds)and not(kind_words(rtags[-1])&kinds)
   verdicts.append(not other_product and bool(same_prop(rhead,field)or re.search(pattern,rvalue.lower())))
  if any(verdicts):
   return"",set()
  return f"{subject.strip()}: {word}",kinds
 def open_hints(self,pk=None):
  reopened,fresh=[],[]
  for clue in self._smf_hints():
   field=str(clue).split(":",1)[0].strip()
   if not field:
    continue
   owner=self.product_of(field)if self.needs.items()else None
   if None not in(pk,owner)and owner!=pk:
    continue
   if self.open_prop(owner,field,self.prop_line(owner,field),by_samples=True)and not self.asked_inside(owner,field):
    fresh.append(str(clue))
   else:
    other=self.hint_alt(owner,str(clue),field)
    if other:
     reopened.append(other)
  return reopened+fresh
 @_smf_guard("")
 def prop_line(self,pk,field):
  kept=self.__dict__.setdefault("menu_memo",{}).get(pk)
  if kept and kept[0]==len(self.archive.items):
   menu=kept[1]
  else:
   menu=self._smf_build_menu(pk)
   self.menu_memo[pk]=(len(self.archive.items),menu)
  return next(filter(lambda text_line:strict_prop(self._smf_key(text_line),field),menu.splitlines()),"")
 def _smf_build_menu(self,pk):
  if not self.needs.items():
   kind_set=self.own_shelf()
   return(self.archive.prop_menu(kind_set)if kind_set else self.archive.prop_menu())[0]
  home=self._smf_home(pk)
  first=next((shelf for shelf in self.archive.shelves_read()if shelf.strip().lower()in home),None)
  return""if first is None else self.archive.prop_menu(first)[0]
 def field_values(self,pk,field):
  text_line=self.prop_line(pk,field)
  found={self._smf_norm(val)for val in self._smf_samples(text_line)}if"):"in text_line else set()
  found.discard("")
  return found
 def same_field(self,pk,first,second):
  return True if strict_prop(first,second)else self._smf_shared_samples(pk,first,second)
 @_smf_guard(False)
 def _smf_shared_samples(self,pk,first,second):
  if not same_prop(first,second):
   return False
  return not self.field_values(pk,first).isdisjoint(self.field_values(pk,second))
 def hinted_prop(self,pk,field):
  wanted=str(field).split("(")[0].strip()
  for clue in self._smf_hints():
   front=str(clue).split(":",1)[0]
   if strict_prop(front,wanted)and(pk is None or not self.needs.items()
or self.product_of(front)in(pk,None)):
    return True
  return False
 @_smf_guard(0)
 def product_reads(self,pk):
  if self.needs.items():
   return sum(map(self.archive.rich_reads,self._smf_home(pk)))
  kind_set=self.own_shelf()
  return self.archive.rich_reads(kind_set)if kind_set else self.archive.rich_reads()
 @_smf_guard(False)
 def plain_hint(self,pk,field,hinted=False):
  wanted=self._smf_bare(field)
  if not wanted or not(hinted or self.hinted_prop(pk,wanted)):
   return False
  if self.prop_line(pk,wanted)or self.product_reads(pk)<2:
   return False
  return all(not same_prop(kname,wanted)for kname in self._smf_menu_keys(pk))
 def hint_mark(self,clue):
  shown=str(clue)
  try:
   field=shown.split(":",1)[0].strip()
   if self.plain_hint(self.product_of(field)if self.needs.items()else None,field,hinted=True):
    shown+=" [no key]"
  except Exception:
   pass
  return shown
 @staticmethod
 def hint_alts_of(tail):
  found=[]
  for group in re.findall(r"\(or ([^()]+)\)",str(tail)):
   found.extend(piece.strip()for piece in re.split(r",|\bor\b|/",group)if piece.strip())
  return found
 @staticmethod
 def name_inside(first,second):
  left,right=set(Errand._smf_words(first)),set(Errand._smf_words(second))
  if not left or not right:
   return False
  return left<=right or right<=left
 def _smf_logged_parts(self,multi,answered_only):
  if answered_only:
   def pieces(label):
    return None if"[not answered]"in label else label.split(";")
  else:
   def pieces(label):
    plain=re.sub(r"\s*\[[^\]]*\]","",label)
    return[bit.strip()for bit in re.split(r";|\band\b",plain)if bit.strip()]
  for logged in self.questions_log:
   parts=pieces(logged.split(":",1)[0].strip("- "))
   if parts is None:
    continue
   asked=re.search(r'asked "(.*?)" -> ',logged)
   for part in parts:
    owner=self.product_of(part)if multi else None
    if owner is None and multi and asked:
     owner=self.product_of(asked.group(1))
    yield part,owner
 @_smf_guard(False)
 def asked_inside(self,pk,subject):
  multi=bool(self.needs.items())
  for part,j in self._smf_logged_parts(multi,True):
   if(pk is None or not multi or j in(None,pk))and self.name_inside(part,subject):
    return True
  return False
 @_smf_guard("")
 def hint_alt(self,pk,clue,subject):
  chosen=self.__dict__.setdefault("hint_alts",{})
  multi=bool(self.needs.items())
  front,_,tail=clue.partition(":")
  if not chosen.get(clue):
   if not self._smf_went_quiet(pk,subject,multi):
    return""
   for other in self.hint_alts_of(tail):
    text_line=""if junk_key(other)or strict_prop(other,subject)else self.prop_line(pk,other)
    if text_line:
     chosen[clue]=self._smf_key(text_line)
     break
  kname=chosen.get(clue)or""
  if not kname or kname.lower()in self._smf_spent()or not self.plan_open(pk,kname):
   return""
  tag=re.search(r"\(([^()]*)\)\s*$",front)
  label=kname if not tag else f"{kname} ({tag.group(1).strip()})"
  rest=re.sub(r"\s*\(or [^()]+\)","",tail).strip()
  return f"{label}: {rest} (its first property, {subject}, drew no preference)"
 def _smf_went_quiet(self,pk,subject,multi):
  for asked in self.__dict__.get("quiet_asks")or[]:
   if not(self.same_field(pk,asked,subject)or self.name_inside(asked,subject)):
    continue
   if not multi or self.product_of(asked)in(pk,None):
    return True
  return False
 @_smf_guard(False)
 def hint_mapped(self,pk,field):
  wanted=self._smf_bare(field)
  needed=[r"\b"+re.escape(word)for word in self._smf_words(wanted)]
  for clue,front,tail in self._smf_hint_rows(pk,bool(self.needs.items()),True):
   if any(strict_prop(cand,wanted)for cand in[self._smf_bare(front)]+self.hint_alts_of(tail)if cand):
    return True
   text=str(clue).lower()
   if needed and all(re.search(pat,text)for pat in needed):
    return True
  return False
 @staticmethod
 def name_twin(first,second):
  if strict_prop(first,second):
   return False
  words_a,words_b=Errand._smf_words(first),Errand._smf_words(second)
  pairs=[(p,q)for p in(words_a[:-1]if len(words_a)>1 else words_a)for q in words_b]
  pairs+=[(p,q)for p in words_a for q in(words_b[:-1]if len(words_b)>1 else words_b)]
  for p,q in pairs:
   if p==q or(min(len(p),len(q))>=4 and(p.startswith(q)or q.startswith(p))):
    return True
  return False
 @_smf_guard(False)
 def twin_of(self,pk,first,second):
  if strict_prop(first,second):
   return False
  if self.name_twin(first,second):
   return True
  return self.same_field(pk,first,second)
 @_smf_guard("")
 def twin_key(self,pk,field):
  entries=self.needs.items()
  self.prop_line(pk,field)
  keys=self._smf_menu_keys(pk)
  goods=entries[pk]["product"]if entries and pk is not None else str(self.needs.reading.get("product")or"")
  kind_set=kind_tokens(goods)
  story=" ".join([self.request.lower()]+[str(clue).lower()for clue,_,_ in self._smf_hint_rows(pk,bool(entries),False)])
  spent=self._smf_spent()
  base_words=set(prop_words(field))
  for kname in keys:
   if not kname or junk_key(kname)or kname.lower()in spent:
    continue
   if not self.twin_of(pk,kname,field)or not self.plan_open(pk,kname)or self.name_inside(kname,field):
    continue
   extra=[word for word in self._smf_words(kname)if word not in base_words and not kind_in(word,kind_set)]
   if any(re.search(r"\b"+re.escape(word),story)for word in extra)or self.hint_mapped(pk,kname):
    return kname
  return""
 @_smf_guard("")
 def values_form(self,pk,field,goods):
  text_line=self.prop_line(pk,field)
  if"):"not in text_line:
   return""
  taken=set()
  for other in self._smf_kept_menu(pk).splitlines():
   if"):"in other and other.strip()!=text_line.strip():
    taken.update(map(self._smf_norm,self._smf_samples(other)))
  picks=[]
  for val in map(str.strip,self._smf_samples(text_line)):
   if len(picks)==3:
    break
   usable=val and len(val)<=30 and re.search(r"[a-z]",val.lower())and not re.search(r"[,;/]",val)
   if usable and self._smf_norm(val)not in taken|{self._smf_norm(p)for p in picks}:
    picks.append(val)
  if len(picks)<2:
   return""
  return f"For the {goods}, would you like {', '.join(picks[:-1])} or {picks[-1]}?"
 @_smf_guard(None)
 def distinct_ask(self,field):
  entries=self.needs.items()
  pk=self.product_of(field)if entries else None
  done=self.__dict__.setdefault("twin_asks",set())
  slot=pk if entries else-1
  if(entries and pk is None)or slot in done:
   return None
  goods=entries[pk]["product"]if entries else str(self.needs.reading.get("product")or"item")
  target,text,step=self._smf_bare(field),"","own"
  while step:
   if step=="own":
    text=self.values_form(pk,target,goods)
    step=""if text else"twin"
   elif step=="twin":
    twin=self.twin_key(pk,field)
    if not twin:
     break
    target=twin
    text=self.values_form(pk,twin,goods)
    step=""if text else"plain"
   else:
    text,step=f"Which {target} would you like for the {goods}?",""
  if not text:
   return None
  done.add(slot)
  return(f"{target} ({goods})"if entries else target),text
 @_smf_guard(False)
 def asked_for(self,pk,field):
  multi=bool(self.needs.items())
  wanted=self._smf_bare(field or"")
  if wanted:
   for part,j in self._smf_logged_parts(multi,False):
    if(pk is None or not multi or j==pk)and strict_prop(self._smf_bare(part),wanted):
     return True
  return False
 @_smf_guard("")
 def alt_key(self,pk,field):
  wanted=self._smf_bare(field)
  if not wanted:
   return""
  spent=self._smf_spent()
  for _,front,tail in self._smf_hint_rows(pk,bool(self.needs.items()),False):
   subject=self._smf_bare(front)
   if not subject or not(self.same_field(pk,wanted,subject)or self.name_inside(wanted,subject)):
    continue
   for other in self.hint_alts_of(tail):
    if junk_key(other)or strict_prop(other,subject)or strict_prop(other,wanted)or self.name_inside(other,wanted):
     continue
    text_line=self.prop_line(pk,other)
    kname=self._smf_key(text_line)if text_line else""
    if not kname or strict_prop(kname,wanted)or kname.lower()in spent:
     continue
    if self.asked_for(pk,kname)or not self.plan_open(pk,kname):
     continue
    return kname
  return""
 def _smg_got(self,name,default=None):
  return self.__dict__.get(name,default)
 def _smg_fault(self,kind,err):
  self.log.append({"kind":kind,"error":": ".join((type(err).__name__,str(err)))[:300]})
 def collide_alt(self,pname):
  try:
   wanted=self.needs.items()
   pi=self.product_of(pname)if wanted else None
   spent=self.__dict__.setdefault("twin_asks",set())
   slot=pi if wanted else-1
   if wanted and pi is None or slot in spent:
    return""
   alt=self.alt_key(pi,pname)
   if not alt:
    return""
   spent.add(slot)
   return f"{alt} ({wanted[pi]['product']})"if wanted else alt
  except Exception:
   return""
 _smg_FRAME=frozenset(("which","would","you","like","for","the"))
 def holds_plain(self,text,plain_q):
  def stems(src,keep=None):
   return{self.stem(tok)for tok in re.findall(r"[a-z0-9]+",str(src).lower())if keep is None or keep(tok)}
  have=stems(text)
  need=stems(plain_q,lambda tok:tok not in self._smg_FRAME and tok not in LINK_WORDS)
  return bool(need)and not need-have
 def alt_untried(self,pi):
  try:
   return any(self.asked_for(pi,name)and self.alt_key(pi,name)for name in self._smg_hint_heads(pi))
  except Exception:
   return False
 def _smg_hint_heads(self,pi):
  for hint in self.needs.reading.get("hints")or[]:
   front=str(hint).split(":",1)[0]
   if self.product_of(front)not in(pi,None):
    continue
   name=re.split(r"\s*\(",front)[0].strip()
   if name:
    yield name
 @staticmethod
 def menu_trim(menu_text,seen_n):
  if not menu_text:
   return menu_text
  twin=re.compile(r"^- (.+?) \(in \d+ listings?\)")
  rows=[row for row in menu_text.splitlines()
if not((hit:=twin.match(row.strip()))and numbered_twin(hit.group(1)))]
  if seen_n>=4:
   rare=re.compile(r"\(in [01] listings?\)")
   rows=[row for row in rows if not rare.search(row)]
  return"\n".join(rows)if rows else menu_text
 def first_reads(self):
  try:
   blocked=self._smg_got("first_reads_done")or self.batch_tool()or not self.role("details")
   if blocked:
    return False
   self.first_reads_done=True
   queue=self._smg_survey()
   if queue:
    self.__dict__.update(open_queue=queue,open_recheck=True,open_purpose="menu",
open_say="To learn which fields these products carry, I open up to two listings of "
"each before my first message.")
    self.log.append({"kind":"first_reads","ids":queue})
   return bool(queue)
  except Exception as err:
   self._smg_fault("first_reads_error",err)
   return False
 def _smg_survey(self):
  wanted=self.needs
  lines=wanted.items()
  shelves=self.home_shelves()
  units=[line["quantity"]for line in lines]or[wanted.quantity()]
  budget=self.moves_left()-self.order_due()
  budget-=2*max(0,self.ask_room()-self.msgs_sent)+8
  claimed=[]
  plans=[]
  for pi in range(max(1,len(lines))):
   shelf=shelves[pi]if pi<len(shelves)else set()
   count=self.product_reads(pi if lines else None)
   if shelf and count<2 and budget>0:
    got=self.menu_rows(shelf,2-count,claimed,units[min(pi,len(units)-1)])
    claimed.extend(got)
    plans+=[(count,pi,got)]if got else[]
  ranked=[got for _,_,got in sorted(plans)]
  return([got[0]for got in ranked]+[got[1]for got in ranked if len(got)>1])[:max(0,budget)]
 def plan_open(self,pi,pname):
  try:
   hint_on=self.hinted_prop(pi,pname)
   menu_line=self.prop_line(pi,pname)if hint_on else""
   opened=self.open_prop(pi,pname,menu_line,by_samples=hint_on)
  except Exception:
   opened=self.open_prop(pi,pname)
  return opened
 _smg_POINTING=re.compile(r"as shown|same as|the page|sold by|redirect|refer")
 def settles(self,rule_text,sample_text=""):
  head,_,value=str(rule_text).partition(":")
  low=value.lower()
  tests=(
(True,lambda:self._smg_POINTING.search(low)or ref_text(low)),
(False,lambda:not positive_value(value,head)),
(False,lambda:self._smg_told_guess(rule_text)),
(False,lambda:sample_text and not re.search(r"\d",value)and self._smg_unechoed(low,sample_text)),
)
  for verdict,fires in tests:
   if fires():
    return verdict
  return True
 def _smg_reading_refs(self,keep_empty):
  found=self.needs.reading.get("references")
  if not isinstance(found,list):
   return[]
  return[str(ref_s)for ref_s in found if keep_empty or ref_s]
 def _smg_told_guess(self,rule_text):
  pointers=self._smg_reading_refs(True)+list(self.needs.__dict__.get("resolved")or[])
  heard=self.request+" "+" ".join(self.needs.exchanges)
  return story_guess(rule_text,heard,pointers)
 @staticmethod
 def _smg_unechoed(low,sample_text):
  words=[tok for tok in re.findall(r"[a-z]{3,}",low)if tok not in LINK_WORDS]
  pool=set(re.findall(r"[a-z]+",sample_text))
  if not words:
   return False
  def echoed(word):
   if re.search(r"\b"+re.escape(word[:-1]if len(word)>4 else word),sample_text):
    return True
   return len(word)>=4 and any(len(smp)>=4 and edits_within(smp,word,1)for smp in pool)
  return not any(map(echoed,words))
 def order_due(self):
  cap=self.step_cap
  floor=max(FIRST_ORDER_BY,min(FIRST_ORDER_DUE,cap//4+1))
  held=max(floor,min(16,cap//3))
  if self._smg_had_trouble():
   held=max(held,self._smg_repair_steps())
  return max(floor,min(held,cap//2))+self.step_reserve()
 def _smg_had_trouble(self):
  if self._smg_got("event_seen"):
   return True
  gone=self._smg_got("dead")
  return bool(gone)and bool(gone&set(self.trolley.units))
 def _smg_repair_steps(self):
  try:
   products=len(self.needs.items())
  except Exception:
   return 9
  return 8+max(1,products)
 def _smg_can_finish(self):
  return bool(self.role("order")and self.role("cart"))
 def narrowing(self):
  return self._smg_can_finish()
 def _smg_reserve_on(self,left_n):
  if left_n<=self.order_due()and self.orders_tried==0:
   return True
  return left_n<=FIRST_ORDER_BY and self.refusal_n>0
 def _smg_phase(self,left_n):
  if left_n<=1:
   return"last"
  if left_n<=3:
   return"closing"
  return"reserve"if self._smg_reserve_on(left_n)else"open"
 def _smg_read_gate(self,left_n,hit):
  return(self.unverified()and left_n>5)or(hit()and left_n>=min(FIRST_ORDER_BY+2,self.order_due()))
 def _smg_search_gate(self,left_n,paged):
  return(self.widened_late()or paged()or(self.unverified()and left_n>5))and left_n>4
 def _smg_reserve_unlocks(self,left_n,names):
  names.update((self.role("cart_view"),self.role("view")or self.role("details"),self.stock_tool()))
  reader=self.role("details")
  if self.swap_open():
   names.add(reader)
  quota_left=self.__dict__.get("late_n",0)<3
  if quota_left and self._smg_read_gate(left_n,self.named_open):
   names.add(reader)
  if quota_left and self._smg_search_gate(left_n,lambda:self.__dict__.get("named_pages")):
   names.add(self.role("search"))
  if left_n>=7 and self.backup_rows():
   names.add(reader)
 def offered(self):
  left_n=self.moves_left()
  closers=(self.role("order"),self.role("decline"),self.role("cart"))
  phase=self._smg_phase(left_n)
  if phase=="open":
   return self.toolset
  names=set(closers[:2])if phase=="last"else set(closers)
  if phase=="reserve":
   self._smg_reserve_unlocks(left_n,names)
  if not(closers[0]and closers[2]):
   return self.toolset
  speak=self.role("message")
  if speak and left_n>=2 and(self.msg_cap is None or self.msgs_sent<self.msg_cap):
   names.add(speak)
  kept=[spec for spec in self.toolset if spec["function"]["name"]in names]
  return kept or self.toolset
 def in_reserve(self):
  left_n=self.moves_left()
  return self._smg_can_finish()and left_n>3 and self._smg_reserve_on(left_n)
 def unverified(self):
  blocked=self.dead_lines()
  live=[]
  for mark in self.passed:
   ident,_,tag=mark.rpartition("|")
   if tag==str(self.needs.version)and ident not in self.refused_lines and ident not in blocked:
    live.append(ident)
  if not live:
   return True
  if not self.needs.items():
   return False
  try:
   seen={shelf for shelf in map(self.shelf_known,live)if shelf!=""}
   return bool(seen)and any(home and not home&seen for home in self.home_shelves())
  except Exception:
   return False
 def named_open(self):
  store=self.archive
  return{rid for rid in self.__dict__.get("named",set())
if rid in store.ids and rid not in store.read and rid not in self.refused_lines}
 def widened_late(self):
  wanted=self.needs
  fresh=wanted.widened_at==wanted.version
  return fresh and self.order_due()>=self.wire.tick-wanted.widened_tick
 def begin_batch(self):
  self.__dict__.update(late_q=0,late_ids=set())
 def late_hold(self,call_name,call_args):
  run={"tool":call_name,"args":call_args,"say":""}
  state="route"
  while state:
   state=self._smg_LATE_GATES[state](self,run)
  return run["say"]
 def _smg_lh_route(self,run):
  reader,finder=self.role("details"),self.role("search")
  run["reader"]=reader
  if run["tool"]in(reader,finder):
   return"reserve"
  if run["tool"]in self.lookup_tools()and reader:
   run["tool"]=reader
   return"reserve"
  return None
 def _smg_lh_reserve(self,run):
  if not self.in_reserve():
   return None
  targets={val for val in run["args"].values()if isinstance(val,str)}
  reading=run["tool"]==run["reader"]
  if reading and targets&(set(self.trolley.units)|{mark.rpartition("|")[0]for mark in self.passed}):
   return None
  queued=self.__dict__.get("late_q",0)
  run.update(targets=targets,reading=reading,queued=queued,spare=self.moves_left()-queued)
  return"quota"
 def _smg_lh_quota(self,run):
  if self.__dict__.get("late_n",0)+run["queued"]>=3:
   return"backup"
  left_n=run["spare"]
  if run["reading"]:
   ok=self._smg_read_gate(left_n,lambda:bool(run["targets"]&self.named_open()))
  else:
   ok=self._smg_search_gate(left_n,lambda:json.dumps(run["args"],sort_keys=True,default=str)
in self.__dict__.get("named_pages",set()))
  return"grant"if ok else"backup"
 def _smg_backup_room(self,left_n):
  return left_n-1>=6 and self.__dict__.get("backup_n",0)<2
 def _smg_fresh_backups(self):
  state=self.__dict__
  return[rid for rid in self.backup_rows()if rid not in state.get("late_ids",set())][:2-state.get("backup_n",0)]
 def _smg_lh_backup(self,run):
  if run["reading"]and self._smg_backup_room(run["spare"])and run["targets"]&set(self._smg_fresh_backups()):
   self.backup_n=self.__dict__.get("backup_n",0)+1
   return"grant"
  return"swap"
 def _smg_lh_swap(self,run):
  if run["reading"]:
   row=self.swap_open()
   if row and row in run["targets"]and run["spare"]>2:
    self.swap_done=True
    return"grant"
  return"deny"
 def _smg_lh_grant(self,run):
  self.late_q=run["queued"]+1
  self.__dict__.setdefault("late_ids",set()).update(run["targets"])
 def _smg_lh_deny(self,run):
  spare_rows=self._smg_fresh_backups()if run["reading"]and self._smg_backup_room(run["spare"])else[]
  say=("Not run (no step used). Checkout mode: three kinds of read are left - (1) a line that is in the cart; (2) any "
"listing an item check called out; (3) when no fit has passed verification yet, 3 reads overall. Recommended "
"now: basket the top verified fit, confirm availability and cost once more, place the order.")
  if spare_rows:
   say+=" Also allowed, as cheaper stand-ins nobody opened yet: "+", ".join(spare_rows)+"."
  run["say"]=say
 _smg_LATE_GATES={"route":_smg_lh_route,"reserve":_smg_lh_reserve,"quota":_smg_lh_quota,
"backup":_smg_lh_backup,"swap":_smg_lh_swap,"grant":_smg_lh_grant,"deny":_smg_lh_deny}
 _smg_SHUT=("read","gone","over","pointer","ruled","clash")
 def backup_rows(self):
  if self.needs.priority()!="cheapest"or self._smg_got("backup_n",0)>=2:
   return[]
  triggered=self._smg_got("event_seen")or self.dead_lines()or self.named_open()
  if not triggered:
   return[]
  found=[]
  try:
   for pi in self.cg_products():
    found.extend(self._smg_cheaper(self.ed_pool(pi)))
  except Exception as err:
   self._smg_fault("backup_error",err)
   return[]
  found.sort()
  return[ident for _,ident in found[:6]]
 @classmethod
 def _smg_cheaper(cls,pool):
  fits=[row["price"]for row in pool if row["verified"]and row["fit"]and row["price"]is not None]
  out=[]
  for row in pool if fits else():
   if any(row[flag]for flag in cls._smg_SHUT):
    continue
   cost=row["price"]
   if cost is not None and cost<min(fits):
    out.append((cost,row["id"]))
  return out
 def field_names(self):
  menu_text,_=self.archive.prop_menu()
  if not menu_text:
   return""
  return("\n\n## FIELD NAME COUNTS (from opened listings; a name and its variants that differ only in separators are "
"merged; the count is the number of opened listings with that name - the most common one is a "
"property's main field)\n"+menu_text)
 def check_item(self,item_id,aim_text,second=""):
  case={"id":item_id,"aim":aim_text,"extra":second,"verdict":""}
  stage="entry"
  while stage:
   stage=self._smg_CHECK_STAGES[stage](self,case)
  return case["verdict"]
 def _smg_ci_entry(self,case):
  if self.wire.spare()<CLOSING+20:
   return None
  item_id=case["id"]
  key=f"{item_id}|{self.needs.version}"
  if key in self.__dict__.get("near_ok",set()):
   return None
  refs=self._smg_reading_refs(False)
  tracked=list(self.__dict__.get("refs")or[])
  own=self.own_redirects(item_id)
  memo=key
  if refs or any(ref_s.get("status")!="resolved"for ref_s in tracked):
   memo+=f"|{len(self.archive.read)}"
  if case["extra"]:
   memo+="|second"
  if own:
   memo+=f"|own{len(own)}"
  case.update(key=key,refs=refs,tracked=tracked,own=own,memo=memo)
  return"memo"
 def _smg_ci_memo(self,case):
  if case["memo"]not in self.rulings:
   return"chain"
  case["verdict"]=self.rulings[case["memo"]]
  return None
 def _smg_ci_chain(self,case):
  item_id=case["id"]
  pointer=self.ref_pointer_note(item_id)
  if pointer:
   self._smg_rule_out(case,pointer,["reference chain"])
   return None
  case["records"]=self.archive.about(item_id)
  return"evidence"if case["records"]else None
 def _smg_ci_evidence(self,case):
  view=case["view"]=self.field_view(case["id"])
  if not view["fails"]:
   return"model"
  failing=view["fails"][:3]
  self._smg_rule_out(case,"; ".join(failing),failing,by="field evidence")
  return None
 def _smg_rule_out(self,case,why,problems,**extra):
  item_id=case["id"]
  ruling=f"Item check on {item_id}: rejected - {why}. Ruled out by code; find another."
  self.rulings[case["memo"]]=ruling
  self.log.append({"kind":"verify","item":item_id,"result":{"item":item_id,"fits":False,"problems":problems,**extra}})
  case["verdict"]=ruling
 def _smg_ci_model(self,case):
  item_id=case["id"]
  reply_obj=self.wire.consult(self._smg_check_text(case),3000,CHECKER)
  self.log.append({"kind":"verify","item":item_id,"result":reply_obj})
  refused=bool(reply_obj)and reply_obj.get("fits")is False
  if not refused:
   self.rulings[case["memo"]]=""
   if reply_obj:
    self.passed.add(case["key"])
    prices=self.__dict__.setdefault("pass_price",{})
    prices[item_id]=self.archive.price(item_id)
   return None
  faults="; ".join([str(pn)for pn in reply_obj.get("problems")or[]][:5])
  case["verdict"]=self.rulings[case["memo"]]=(
f"Item check on {item_id}: rejected - {faults}. If you are sure this verdict is mistaken you may keep it; "
"if not, take a fit proven on all requirements.")
  return None
 _smg_CHECK_STAGES={"entry":_smg_ci_entry,"memo":_smg_ci_memo,"chain":_smg_ci_chain,
"evidence":_smg_ci_evidence,"model":_smg_ci_model}
 def _smg_check_text(self,case):
  item_id,wanted=case["id"],self.needs
  parts=[CHECK_ITEM,
"\n\n## REQUEST (the shopper's words; later changes follow)\n"+self.request+wanted.changes(),
"\n\n## THE ITEM MUST MEET (as first stated, revealed later, or changed)\n"+wanted.text_body()]
  if wanted.items():
   held=self.trolley.held(item_id)
   parts.append("\n\n## WHICH PRODUCT: the request covers several products, and the item's category shows which "
"one this item would be. Judge it on that product's own constraints and quantity only - nothing "
"the other products need applies."+(f" Already carted: {int(held)} units."if held>0 else""))
  parts.append(self.ref_note(item_id)or"")
  parts.append(self._smg_pointed_records(case))
  parts.append(case["extra"])
  parts.append(self._smg_own_notes(case["own"]))
  parts.append(f"\n\n## CANDIDATE, next step {case['aim']}: {item_id}\nStore records for it, from oldest to newest:\n"
+"\n".join(case["records"]))
  if self.trolley.last_view is not None:
   parts.append("\n\n## CART, NEWEST VIEW\n"+as_text(self.trolley.last_view)[:3000])
  parts.append(self.field_names())
  if case["view"]["lines"]:
   parts.append("\n\nFIELD EVIDENCE, compiled by code. Per required value: how listings in this category "
"spell it here, then the item's entry:\n"+"\n".join(case["view"]["lines"]))
  labels=self.archive.label_lines()
  if labels:
   parts.append("\n\n## RECURRING CATEGORY LABELS (count each). The item must sit under the label whose words "
"match the request's phrase taken whole - an adjacent label does not qualify.\n"+labels)
  return"".join(parts)
 def _smg_pointed_records(self,case):
  loose=self.ref_untracked(case["refs"],case["tracked"])
  if not loose:
   return""
  cues={tok for ref_s in loose for tok in re.findall(r"[a-z]{4,}",ref_s.lower().split(":",1)[-1])}
  store,item_id=self.archive,case["id"]
  hits=[rid for rid in sorted(store.read)
if rid!=item_id and cues and any(tok in" ".join(store.about(rid,1)).lower()for tok in cues)][:3]
  if not hits:
   return""
  return("\n\n## READ LISTINGS THE REQUEST MAY POINT AT. A value given by pointing at one of them is read off that "
"listing, in the field the requirement names.\n"
+"\n".join(rec for rid in hits for rec in store.about(rid,1)))
 @staticmethod
 def _smg_own_notes(own):
  quoted=[]
  for entry in own or():
   rule_text,pname,sent=entry
   quoted.append(f"\n- {pname}, for '{rule_text}': \"{sent[:240]}\"")
  if not quoted:
   return""
  return("\n\nREDIRECT CHECK - the code quotes this item's notes:"+"".join(quoted)
+"\nReading rule. Some quotes may hand a property over to a different listing (wording such as take from, "
"refer to, go by, look to). Where that happens, the grader uses the value the other listing gives, so the "
"field on this item is no proof of the requirement; only that other listing's record can satisfy it. If a "
"quote names another listing merely as a version on sale elsewhere, or says this item is not it, ignore "
"that quote.")
 def ch_rows(self):
  stamp,table=self._smg_got("ch_rows_memo")or(None,None)
  if stamp==len(self.archive.items):
   return table
  table=self.code_rows()
  self.ch_rows_memo=(len(self.archive.items),table)
  return table
 def store_fields(self,label=""):
  try:
   table=self.ch_rows()
   scope=self._smg_label_shelves(label)
   picked=[row["fields"]for row in table.values()if row["fields"]and(not scope or row["shelf"]in scope)]
   pairs=[(str(key),str(val))for fields in picked for key,val in fields.items()]
   gathered={}
   for key,text in pairs:
    if text.strip()and not junk_key(key):
     gathered.setdefault(key,[]).append(text)
   return gathered
  except Exception:
   return{}
 def _smg_label_shelves(self,label):
  homes=self.home_shelves()
  if not self.needs.items():
   own=self.own_shelf()
   if own:
    return{own}
   return set(homes[0]if homes else())
  pi=self.product_of(str(label))
  if pi is None or pi>=len(homes):
   return set()
  return set(homes[pi])
 def ch_redirected(self,item_id,pname):
  try:
   notes=self.own_redirects(item_id)or[]
   return any(self._smg_names_prop(note,pname)for note in notes)
  except Exception:
   return True
 @staticmethod
 def _smg_names_prop(note,pname):
  heads=note[:2]if isinstance(note,(list,tuple))else[note]
  return any(same_prop(str(h).partition(":")[0],pname)for h in heads)
 def field_view(self,item_id):
  wanted=self.needs
  stamp=(item_id,len(self.archive.items),wanted.version,"|".join(wanted.props_left()))
  cache=self.__dict__.setdefault("ch_view_memo",{})
  if stamp not in cache:
   cache[stamp]=self._smg_facts_or_blank(item_id)
  view=cache[stamp]
  confirmed=self._smg_got("near_said")or{}
  cleared=confirmed.get(f"{item_id}|{wanted.version}")
  if cleared:
   try:
    return self._smg_mark_cleared(view,cleared)
   except Exception:
    pass
  return view
 def _smg_facts_or_blank(self,item_id):
  try:
   return self.field_facts(item_id)
  except Exception as err:
   self._smg_fault("field_facts_error",err)
  return{part:[]for part in("lines","rules","fails")}
 @staticmethod
 def _smg_mark_cleared(view,cleared):
  names=[pn for pn,_,_ in cleared]
  alike=lambda one,two:same_prop(one,two)or strict_prop(one,two)
  def note_for(line):
   head=re.match(r"- (.+?) = '",line)
   for pn,val,said in cleared if head else():
    if alike(head.group(1),pn):
     return f" (Meaning check: '{said[:80]}' counts as '{val[:60]}' - requirement met.)"
   return""
  marked=[line+note_for(line)for line in view["lines"]]
  kept=[fault for fault in view["fails"]if not any(alike(fault.split(":",1)[0].strip(),pn)for pn in names)]
  return{"lines":marked,"rules":list(view["rules"]),"fails":kept}
 @staticmethod
 def _smh_clip(text,n=70):
  flat=" ".join(str(text).split())
  return flat if len(flat)<=n else flat[:n-3]+"..."
 @staticmethod
 def _smh_set(raw):
  return bool(str(raw).strip())
 @staticmethod
 def _smh_over(cost,units,cap):
  return is_number(cap)and is_number(cost)and cost*units>cap
 @staticmethod
 def _smh_fault(err):
  return("%s: %s"%(type(err).__name__,err))[:300]
 @staticmethod
 def _smh_by_count(carry,keep):
  return sorted(filter(keep,carry),key=lambda name:-carry[name])
 @classmethod
 def _smh_coverage(cls,peers,keys):
  return len([1 for row in peers.values()if any(cls._smh_set(row["fields"].get(name,""))for name in keys)])
 @classmethod
 def _smh_census(cls,peers):
  carry={}
  table={}
  for owner,row in peers.items():
   for name,raw in row["fields"].items():
    if junk_key(name)if cls._smh_set(raw)else True:
     continue
    carry[name]=1+carry.get(name,0)
    bits=ch_parts(raw)
    whole=[str(raw).strip().lower()]if len(bits)>1 else[]
    table.setdefault(name,[]).extend((owner,bit,ch_words(bit))for bit in bits+whole)
  return carry,table
 @staticmethod
 def _smh_parse_need(line):
  label,_,rest=str(line).partition(":")
  want=" ".join(re.sub(r"\((?:stated|revealed[^)]*)\)"," ",rest).split())
  attr=re.split(r"\s*\(",label)[0].strip()
  if attr and want and not ref_text(want)and not re.search(Errand._SMH_POINTER,want.lower()):
   return label,attr,want
  return None
 _SMH_POINTER="|".join(("as shown","same as","the page","sold by","redirect","refer"))
 def field_facts(self,pick):
  found=dict(lines=[],rules=[],fails=[])
  table_rows=self.ch_rows()
  own=table_rows.get(pick)
  if not(own and own["fields"]):
   return found
  multi=bool(self.needs.items())
  label=own["shelf"]
  if multi and not label:
   return found
  slot=None
  if multi:
   slot=next((n for n,labels in enumerate(self.home_shelves())if label in labels),None)
  peers={}
  for other,row in table_rows.items():
   if row["fields"]and(not label or row["shelf"]==label):
    peers[other]=row
  if len(peers)<3:
   return found
  carry,table=self._smh_census(peers)
  scope=dict(pick=pick,rows=table_rows,me=own,label=label,multi=multi,slot=slot,peers=peers,carry=carry,
table=table,found=found)
  for line in self.needs.props_left()[:12]:
   need=self._smh_parse_need(line)
   if need:
    self._smh_weigh_need(scope,*need)
  return found
 def _smh_foreign(self,scope,label):
  if not scope["multi"]:
   return False,None
  owner=self.product_of(label)
  return owner is not None and owner!=scope["slot"],owner
 def _smh_weigh_need(self,scope,label,attr,want):
  lowered=want.lower()
  if re.search(r"\d",want)and re.search(CH_BOUND,lowered):
   self._smh_limit_need(scope,label,attr,want)
   return
  if re.fullmatch(r"\s*\d+(?:\.\d+)?\s+[a-z]+\s*",lowered):
   return
  foreign,owner=self._smh_foreign(scope,label)
  if foreign:
   return
  plus=positive_value(want,label)
  if not(plus or re.search(EXCLUDE_WORDS,lowered)):
   return
  forms=ch_alts(want)
  if not forms:
   return
  if plus and self._smh_shelf_names(scope,want):
   return
  keys=self._smh_pick_keys(scope,attr,want,forms,plus,owner)
  if keys is None:
   return
  named,titled,plus=keys
  facts=self._smh_category_facts(scope,attr,want,forms,named,titled,plus)
  scope["found"]["rules"].append(facts["rule"])
  self._smh_listing_verdict(scope,attr,want,forms,facts)
 def _smh_shelf_names(self,scope,want):
  shelf=str(scope.get("label")or"")
  if not shelf or not ch_writes(want,shelf):
   return False
  line=("- '%s' | the category '%s' itself names this value: every listing on it meets it, whether or not a "
"field or the title repeats it -> MEETS"%(self._smh_clip(want),self._smh_clip(shelf,60)))
  scope["found"]["rules"].append(line)
  scope["found"]["lines"].append(line)
  return True
 def _smh_limit_need(self,scope,label,attr,want):
  foreign,_=self._smh_foreign(scope,label)
  tally=scope["carry"]
  stems={w for w in set(prop_words(attr))if len(w)>=3 and w not in LINK_WORDS}
  homes=[name for name in tally if tally[name]>=2 and strict_prop(name,attr)]
  if foreign or not(attr and stems and homes):
   return
  for name,raw in scope["me"]["fields"].items():
   if self._smh_set(raw)and(strict_prop(name,attr)or set(prop_words(name))&stems):
    return
  brief,peers=self._smh_clip(want),scope["peers"]
  scope["found"]["lines"].append(
"- %s = '%s' | field in this store: %s | category listings read that fill it: %d of %d | this listing: field "
"absent -> FAILS the limit (no figure from a different field can stand in)"
%(attr,brief," / ".join("'%s'"%x for x in homes[:3]),self._smh_coverage(peers,homes),len(peers)))
  scope["found"]["fails"].append("%s: no %s field here, though the store's category listings carry one -> '%s' "
"unproven"%(attr,homes[0],brief))
 def _smh_pick_keys(self,scope,attr,want,forms,plus,
owner):
  tally,table=scope["carry"],scope["table"]
  named=self._smh_by_count(tally,lambda name:strict_prop(name,attr))
  if named:
   if not plus:
    flat=" ".join(re.findall(r"[a-z0-9]+",want.lower()))
    if flat in(" ".join(entry[2])for name in named for entry in table.get(name,[])):
     plus=True
     LITERAL_VALUES.add(flat)
   return named,True,plus
  if not plus or(scope["multi"]and owner is None):
   return None
  holders=self._smh_by_count(tally,lambda name:tally[name]>=2 and any(
ch_match(forms,entry[2])for entry in table.get(name,[])))
  if not holders:
   return None
  return self._smh_by_count(tally,lambda name:strict_prop(name,holders[0])),False,plus
 def _smh_category_facts(self,scope,attr,want,forms,named,
titled,plus):
  clip,pick,peers=self._smh_clip,scope["pick"],scope["peers"]
  others=[entry for name in named for entry in scope["table"].get(name,[])if entry[0]!=pick]
  writers={entry[0]for entry in others if ch_match(forms,entry[2])}
  facts={"plus":plus,"titled":titled,"named":named,"writers":writers,
"alone":{" ".join(entry[2])for entry in others},"filled":self._smh_coverage(peers,named)}
  seen=[]
  for row in peers.values():
   seen+=[clip(row["fields"][name],40)for name in named if self._smh_set(row["fields"].get(name,""))]
  facts["samples"]=" | ".join(list(dict.fromkeys(seen))[:6])
  brief=clip(want)
  tail="other listings with '%s' verbatim in it: %d"%(brief,len(writers))
  if not plus:
   bare=" ".join(re.sub(EXCLUDE_WORDS," ",want.lower()).split())
   bare_forms=ch_alts(bare)
   holding={entry[0]for entry in others if bare_forms and ch_match(bare_forms,entry[2])}
   if bare_forms:
    tail=("other listings holding '%s', the excluded value, there: %d (holding it = failing)"
%(clip(bare),len(holding)))
  where=" / ".join("'%s'"%x for x in named[:3])
  aside=""if titled else" [picked by its values; no field is called '%s']"%attr
  facts["lead"]="- %s = '%s' | field in this store: %s%s | category listings read that fill it: %d of %d | %s"%(
attr,brief,where,aside,facts["filled"],len(peers),tail)
  kinds=kind_words(want)
  facts["naming"]=[]
  if kinds and not writers:
   facts["naming"]=[other for other,row in scope["rows"].items()if other!=pick
and(not scope["label"]or row["shelf"]==scope["label"])and kinds<=row["words"]][:3]
  facts["rule"]=facts["lead"]+self._smh_rule_tail(facts,brief)
  return facts
 @staticmethod
 def _smh_rule_tail(facts,brief):
  seen=" | values on record: "+facts["samples"]
  if not facts["plus"]:
   return""
  if facts["writers"]:
   extra=""if facts["titled"]else(
" | no field bears the property's name, so a title with the value in it matches too, unless the "
"listing's own field sharing a word of that name contradicts it")
   return(" | so: a match needs those very words (contained, or as one list element); synonyms and kindred "
"words miss")+extra
  if facts["naming"]:
   return(seen+" | titles/texts in this category mention '%s' (%s), proof of a house term: synonyms or kindred "
"values count only after one of those listings' records shows this field phrasing it otherwise"
%(brief,", ".join(facts["naming"])))
  if facts["filled"]<CH_SYNONYM_MIN:
   return(seen+" | '%s' absent, yet with this few read listings filling the field nobody can tell if the "
"store ever writes it verbatim: an equivalent stays unproven for now"%brief)
  return seen+" | the value is absent from all of them: an equivalent may count"
 def _smh_fuller_keys(self,fields,named,attr,want):
  base_words=prop_words(attr)
  fuller=[]
  for name,raw in fields.items():
   if not base_words or name in named or not self._smh_set(raw)or junk_key(name):
    continue
   words=prop_words(name)
   if len(words)==len(base_words)+1 and words[1:]==base_words:
    fuller.append((name,str(raw)))
  hit=[pair for pair in fuller if ch_writes(want,pair[1])]
  near=bool(hit)or any(ch_touches(want,pair[1])for pair in fuller)
  note=""
  if hit:
   note=(" fuller-named key %s = '%s' has the value (one extra leading word, still this property) -> MEETS"
%(hit[0][0],self._smh_clip(hit[0][1],60)))
  return hit,near,note
 def _smh_listing_verdict(self,scope,attr,want,forms,facts):
  clip,me,named=self._smh_clip,scope["me"],facts["named"]
  fields,lead,brief=me["fields"],facts["lead"],clip(want)
  mine=[(name,str(fields[name]))for name in named if self._smh_set(fields.get(name,""))]
  shown="; ".join("%s: '%s'"%(name,clip(raw,90))for name,raw in mine)
  hit,near,note=self._smh_fuller_keys(fields,named,attr,want)
  lines,fails=scope["found"]["lines"],scope["found"]["fails"]
  if self.ch_redirected(scope["pick"],attr):
   lines.append(lead+" | caution: per its notes, this property might be defined by a different "
"listing; its own field is then not final")
   return
  if not facts["plus"]:
   lines.append(lead+" | this listing: %s | exclusion: carrying the ruled-out value fails"
%(shown or"field absent"))
   return
  stems={w for w in prop_words(attr)if len(w)>=3 and w not in LINK_WORDS}
  rival=[name for name,raw in fields.items()if self._smh_set(raw)and not junk_key(name)
and set(prop_words(name))&stems and not ch_writes(want,raw)]
  by_title=not rival and ch_writes(want,me.get("title")or"")
  if not mine:
   lines.append(lead+self._smh_missing_field(fields,named,attr,want,by_title,hit,note,facts["titled"]))
   return
  by_title=by_title and not facts["titled"]
  clean,joined=self._smh_split_parts(mine,forms,facts["alone"])
  plain_fail=facts["titled"]and not near
  mine_text=" | this listing: "+shown+" | "
  if clean:
   text="contains '%s' -> MEETS"%brief
  elif hit:
   text=note.strip()
  elif by_title:
   text="title has '%s' and no field bears the property's name -> MEETS"%brief
  elif joined:
   piece,sides=joined[0]
   if plain_fail:
    fails.append("%s: %s is '%s', coupling %s with %s (a value the store also writes by itself there) -> not %s"
%(attr,named[0],clip(piece),brief,sides[0],brief))
   text=("'%s' couples '%s' with '%s', a value other listings hold by itself in this field -> FAILS"
%(clip(piece),brief,sides[0]))
  elif facts["writers"]:
   if plain_fail and not any(ch_touches(want,raw)for _,raw in mine):
    fails.append("%s: %s is '%s'; the store writes '%s' verbatim in that field elsewhere"
%(attr,named[0],clip(mine[0][1],60),brief))
   text=("'%s' missing, though other category listings put it in this field -> FAILS (once the store writes "
"the value verbatim, synonyms and kindred words do not count)"%brief)
  elif facts["naming"]:
   text=("'%s' missing; no read listing has it in this field yet (values on record: %s), but category "
"titles/texts mention '%s' (%s): the store has its own term, so the different value here is unproven"
%(brief,facts["samples"],brief,", ".join(facts["naming"])))
  elif facts["filled"]<CH_SYNONYM_MIN:
   text=("'%s' missing from this field on every listing read (values on record: %s), and only %d read "
"listing(s) fill it: too few to know if the store writes '%s' verbatim, so an equivalent value stays "
"unproven for now"%(brief,facts["samples"],facts["filled"],brief))
  else:
   text=("'%s' missing from this field on every listing read (values on record: %s): an equivalent value may "
"count"%(brief,facts["samples"]))
  lines.append(lead+mine_text+text)
 def _smh_missing_field(self,fields,named,attr,want,by_title,
hit,note,titled=False):
  if hit:
   return" | this listing: field absent |"+note
  if by_title and titled:
   return(" | this listing: field absent, and none of its own fields shares the property's name; the title has "
"'%s' -> MEETS (the title stands in for a field this listing lacks)"%self._smh_clip(want))
  if by_title:
   return(" | this listing: field absent, and no field carries the property's name; the title has '%s' -> MEETS"
%self._smh_clip(want))
  stems={w for w in prop_words(attr)if len(w)>=3}
  maybe=[(name,raw)for name,raw in fields.items()if name not in named and self._smh_set(raw)
and not junk_key(name)and(set(prop_words(name))&stems or ch_writes(want,raw))]
  maybe.sort(key=lambda pair:not ch_writes(want,pair[1]))
  listed="; ".join("%s: '%s'"%(name,self._smh_clip(raw,60))for name,raw in maybe[:4])
  return(" | this listing: field absent"+(" | candidate fields of its own: "+listed if listed else"")
+" | only an alias of this SAME property may count; fields about other properties (height for a "
"type, frame material for material) never do")
 @staticmethod
 def _smh_split_parts(mine,forms,alone):
  pieces=[piece for _,raw in mine for piece in ch_parts(raw)]
  pieces+=[raw.strip().lower()for _,raw in mine if len(ch_parts(raw))>1]
  clean=[]
  joined=[]
  held=[set(one.split())for one in alone if one]
  for piece in filter(lambda p:ch_match(forms,ch_words(p)),pieces):
   sides=ch_sides(piece)
   hits=[ch_match(forms,ch_words(side))for side in sides]
   extra=[side for side,ok in zip(sides,hits)if not ok and Errand._smh_held_alone(side,alone,held)]if any(hits)else[]
   (joined.append((piece,extra))if extra else clean.append(piece))
  return clean,joined
 @staticmethod
 def _smh_held_alone(side,alone,held):
  words=ch_words(side)
  if" ".join(words)in alone:
   return True
  mine=set(words)
  return any(one and one<=mine for one in held)
 def rank_check(self,pick):
  needs=self.needs
  goal=needs.priority()
  mark="%s|%s|%s"%(pick,needs.version,self.refusal_n)
  blocked=goal=="none"or mark in self.rank_checked or pick in self.pair_best()
  if blocked or self.wire.spare()<CLOSING+30 or self.moves_left()<4:
   return""
  self.rank_checked.add(mark)
  low_first=goal=="cheapest"
  detail=needs.reading.get("priority_detail")or""
  if low_first:
   what="cheapest (lowest price)"
  else:
   what="best (%s)"%detail if detail else"best"
  parts=[RANK_CHECK.replace("PRIORITY",what),"\n\nTHE SHOPPER'S REQUEST:\n",self.request,needs.changes(),
"\n\nCURRENT REQUIREMENTS:\n",needs.text_body(),self.ref_note()]
  if needs.items():
   parts.append("\n\nSEVERAL PRODUCTS"+": rivals must share the chosen item's product (category); whatever "
"belongs to another product is never an upgrade.")
  legend=["card legend: DETAILS = full record read","ROW = search-row sighting only, fields unknown",
"a ROW within the requested category MAY fit, and goes in the answer, if it outdoes the chosen item "
"and neither title, price nor stock counts against it, whatever object the title names"]
  if low_first:
   legend.append("cards limited to listings cheaper than the pick")
  parts.append("\n\nTHE CHOSEN ITEM: "+pick+"\n\nLISTINGS SEEN"+" ("+"; ".join(legend)+"):\n")
  cards=self.archive.cards(20000,pick if low_first else"",self.ref_foreign(),{pick})
  parts+=["\n".join(cards),self.field_names()]
  evidence=self.field_view(pick)["rules"]
  if evidence:
   parts.append("\n\nFIELD EVIDENCE"+" (per required value, a code-made summary of how listings in the pick's "
"category write it here):\n"+"\n".join(evidence))
  reply=self.wire.consult("".join(parts),CHECK_TOKENS,CHECKER)
  self.log.append({"kind":"rank","result":reply})
  gone=set(self.dead_lines())|set(self.refused_lines)
  picks=[]
  for raw in(reply or{}).get("better")or[]:
   if isinstance(raw,(str,int))and str(raw)not in gone and str(raw)!=pick:
    picks.append(str(raw))
  named=list(filter(lambda ident:not self.field_view(ident)["fails"],picks[:4]))
  self.rank_better=named[:]
  self.__dict__.setdefault("named",set()).update(named)
  if not named:
   return""
  why=reply.get("reason")
  return("Rank check: %s may also fit and outdo %s on the %s%s. Before ordering, inspect each (details, stock, "
"price); should every one fail, go ahead with the current pick."%(", ".join(named),pick,what,
" (%s)"%why if why else""))
 def review_decline(self,early=False):
  job={"early":early,"note":""}
  step="gate"
  while step!="end":
   step=getattr(self,"_smh_audit_"+step)(job)
  return job["note"]
 def _smh_audit_gate(self,job):
  kv=self.__dict__
  again=kv.get("audits",0)<2 if len(self.archive.read)>kv.get("audit_reads",10**9)else False
  if(again or not self.audit_done)and self.wire.spare()>=CLOSING+30:
   self.audit_done=True
   if not job["early"]:
    self.audits=1+kv.get("audits",0)
   return"cards"
  return"end"
 def _smh_audit_cards(self,job):
  shop,last=self.archive,self.__dict__.get("audit_ids")
  newer=()if last is None else set(shop.read)-set(last or())
  job["cards"]=shop.cards(16000,"",self.ref_foreign(),newer)
  self.audit_reads=len(shop.read)
  if not job["cards"]:
   return"end"
  job["fresh"]=sorted(shop.read-last)if last is not None else[]
  self.audit_ids=set(shop.read)
  return"ask"
 def _smh_audit_ask(self,job):
  needs=self.needs
  parts=[BEFORE_DECLINE,"\n\nTHE SHOPPER'S REQUEST:\n",self.request,needs.changes(),
"\n\nCURRENT REQUIREMENTS:\n",needs.text_body(),self.ref_note(),
"\n\nLISTINGS SEEN"," (cards marked DETAILS come from a full record; cards marked ROW from a "
"search row alone):\n",
"\n".join(job["cards"])]
  if job["fresh"]:
   parts.append("\n\nNEWLY OPENED"+" (just opened; look at these cards ahead of all others and test every "
"field under the reading rules): "+", ".join(job["fresh"]))
  parts.append(self.field_names())
  reply=self.wire.consult("".join(parts),6000,AUDITOR)
  self.log.append({"kind":"audit","result":reply})
  job["reply"]=reply or{}
  offered=[str(x)for x in job["reply"].get("possible_fits")or[]if isinstance(x,(str,int))]
  job["fits"]=self.audit_fits=[ident for ident in offered[:5]if not self.field_view(ident)["fails"]]
  return"leads"
 def _smh_audit_leads(self,job):
  kv,shop=self.__dict__,self.archive
  asked=[]
  for raw in job["reply"].get("unopened")or[]:
   if isinstance(raw,(str,int))and str(raw)not in shop.read and str(raw)in shop.ids:
    asked.append(str(raw))
  labels,cap,units=self.shelf_labels(),self.needs.spend_cap,self.needs.quantity()
  where=kv.get("hit_shelf",{})
  def wanted(ident):
   if ident in shop.read or(labels and where.get(ident)and where.get(ident)not in labels):
    return False
   return not self._smh_over(shop.price(ident),units,cap)
  ranked=sorted(kv.get("all_hits",[]),key=lambda i:-kv.get("hit_score",{}).get(i,0))
  hits=list(filter(wanted,ranked))
  try:
   spare_rows=[ident for ident in self.slack_rows()if ident not in shop.read]
  except Exception:
   spare_rows=[]
  merged=spare_rows+hits[:4]+[ident for ident in asked[:6]if ident not in hits]
  job["unopened"]=list(dict.fromkeys(merged))[:6]
  return"searches"
 def _smh_audit_searches(self,job):
  kv=self.__dict__
  def remember(found):
   kv.setdefault("named_pages",set()).update(json.dumps(q,sort_keys=True,default=str)for q in found)
  job["pages"]=[]if self.moves_left()<=5 else self.open_pages(2)
  if job["pages"]:
   job["unopened"]=job["unopened"][:4]
   remember(job["pages"])
  try:
   job["combos"]=[]if self.moves_left()<=5 else self.combined_queries()
  except Exception as err:
   self.log.append({"kind":"combo_error","error":self._smh_fault(err)})
   job["combos"]=[]
  if job["combos"]:
   remember(job["combos"])
  leads=job["fits"]+job["unopened"]
  kv.setdefault("named",set()).update(leads)
  ordered=kv.setdefault("named_list",[])
  ordered.extend([ident for ident in leads if ident not in ordered])
  return"speak"if leads or job["pages"]or job["combos"]else"end"
 def _smh_audit_speak(self,job):
  said=[]
  def runs(found):
   return"; ".join("%s %s"%(self.role("search"),json.dumps(q,ensure_ascii=False))for q in found)
  if job["fits"]:
   why=job["reply"].get("reason")
   said.append("Second opinion - possibly meeting all requirements: %s%s."
%(", ".join(job["fits"])," (%s)"%why if why else""))
  if job["pages"]:
   said.append("Unread result pages - searches with required values whose matches outnumber the listings "
"read, so fits may hide further down: "+runs(job["pages"])+". A full page of rows takes one "
"step: go through those pages first, then open every requested-kind row inside the budget.")
  if job["combos"]:
   said.append("Combined searches nobody has run yet (all words mandatory: results show only listings bearing "
"every one): "+runs(job["combos"])+". These still go through the search tool; "
"run them, then open every requested-kind row inside the budget.")
  if job["unopened"]:
   said.append("Sighted only as search rows, never opened, so fields unchecked (requested kind): "
+", ".join(job["unopened"])+". Start with their details; one reply can hold several.")
  said.append("Settle each lead by its details, stock and price, and buy the one that holds up; declining is right "
"only once all of them truly fail.")
  job["note"]=" ".join(said)
  return"end"
 def named_unread(self):
  kv,shop=self.__dict__,self.archive
  cap,units=self.needs.spend_cap,self.needs.quantity()
  labels,where=self.shelf_labels(),kv.get("hit_shelf",{})
  score=kv.get("hit_score",{})
  order=sorted(kv.get("all_hits",[]),key=lambda i:-score.get(i,0))+list(kv.get("named_list",[]))
  kept=[]
  for ident in dict.fromkeys(order):
   cost=shop.price(ident)
   if ident in shop.read or ident in self.refused_lines or ident not in shop.ids:
    continue
   if self._smh_over(cost,units,cap)or(labels and where.get(ident)and where[ident]not in labels):
    continue
   kept.append(ident)
  return kept
 def details_call(self,ident):
  tool,known=self.role("details"),self.archive.ids
  if not tool or tool==self.role("search"):
   return None
  def is_id(v):
   return isinstance(v,str)and v in known
  envs=(e.get("actions")or[]for e in reversed(self.log)if e.get("kind")=="env")
  for acts in envs:
   for act in acts:
    given=act.get("args")if act.get("name")==tool else None
    if isinstance(given,dict)and any(map(is_id,given.values())):
     return{name:ident if is_id(v)else v for name,v in given.items()}
  shape=self.shapes.get(tool or"")or{}
  spec=shape.get("properties")
  spec=spec if isinstance(spec,dict)else{}
  texts=[name for name,p in spec.items()if isinstance(p,dict)and p.get("type")=="string"and not p.get("enum")]
  needed=shape.get("required")or[]
  choice=next(filter(lambda name:name in needed,texts),None)
  if choice is None and len(texts)==1:
   choice=texts[0]
  return{choice:ident}if choice else None
 def open_reads(self,queue,say,batch,reader=None):
  if batch:
   name,field,size=batch
   payloads=[{field:queue[at:at+size]}for at in range(0,len(queue),size)]
  else:
   name=reader or self.role("details")
   payloads=list(filter(None,(self.lookup_call(ident,name)for ident in queue)))if name else[]
  if not payloads:
   return[]
  stamp=self.wire.tick
  calls=[]
  for n,payload in enumerate(payloads):
   calls.append({"id":"open_%d_%s"%(n,stamp),"type":"function",
"function":{"name":name,"arguments":json.dumps(payload)}})
  self.chat.append({"role":"assistant","content":say,"tool_calls":calls})
  mark=len(self.archive.items)
  self.covering=True
  try:
   answers=self.dispatch(calls,scripted=True)
  finally:
   self.covering=False
  self.last_reads=self.archive.items[mark:]
  return answers
 _SMH_OPEN_KEYS=(("open_recheck",False),("open_purpose",""),("open_floor",None),("open_tool",None),("open_say",""))
 def open_named(self):
  job={}
  step="take"
  while isinstance(step,str):
   step=getattr(self,"_smh_open_"+step)(job)
  return step
 def _smh_open_take(self,job):
  kv,wire=self.__dict__,self.wire
  queue=kv.pop("open_queue",None)
  job["put_read"]=False
  if not queue and kv.get("put_queue")and not(wire.done or wire.broke):
   queue,job["put_read"]=kv.pop("put_queue"),True
   kv.update(open_recheck=True,open_purpose="cart",open_tool=kv.pop("put_tool",None),
open_say="Stock re-read of the item just added to the cart, since a listing may sell out moments "
"after being added.")
  if not queue or wire.done or wire.broke:
   return False
  job["queue"]=list(dict.fromkeys(map(str,queue)))
  job.update((name,kv.pop(name,empty))for name,empty in self._SMH_OPEN_KEYS)
  job["recheck"]=job["open_recheck"]
  job["purpose"]=str(job["open_purpose"]or"")
  if not job["open_say"]:
   job["open_say"]=("Re-reading the cart lines ahead of the order."if job["recheck"]
else"Checking flagged listings first; no decline yet.")
  return"trim"
 def _smh_open_trim(self,job):
  optional=job["purpose"]in("menu","quota","walk")
  if not(optional and self.role("order")and self.role("cart"))or self.wire.mute:
   return"cart"
  queue=job["queue"]
  try:
   reserve=self.order_due()if self.orders_tried==0 else FIRST_ORDER_BY
   room=self.moves_left()-reserve-1
   listed=lambda:bool(self.batch_tool())and len(queue)>=2
   if room<(1 if listed()else len(queue))and(self.trolley.units or not self.unverified()):
    kept=[]if listed()else queue[:max(0,room)]
    self.log.append({"kind":"reserve_skip","purpose":job["purpose"],"ids":queue[:6],"kept":kept})
    job["queue"]=kept
    if not kept:
     return False
  except Exception as err:
   self.log.append({"kind":"reserve_skip_error","error":self._smh_fault(err)})
  return"cart"
 def _smh_open_cart(self,job):
  if job["purpose"]!="cart":
   return"read"
  stale=set(self.stale_lines())
  job["queue"]=[i for i in job["queue"]if i in stale or(not job["put_read"]and i not in self.trolley.units)]
  return"read"if job["queue"]else False
 def _smh_open_read(self,job):
  queue,reader=job["queue"],job["open_tool"]
  lister=self.batch_tool()if len(queue)>=2 else None
  if lister and job["purpose"]=="cart"and not self.__dict__.get("batch_proven"):
   lister=None
  if lister and reader:
   caps=(self.__dict__.get("tool_caps")or{}).get(lister[0])or{}
   if not caps.get("stock"):
    lister=None
  job["batch"]=lister
  job["responses"]=self.open_reads(queue,job["open_say"],lister,reader)
  return"prove"
 def _smh_open_prove(self,job):
  kv=self.__dict__
  if not(job["batch"]and job["responses"])or kv.get("batch_proven"):
   return"audit"
  rows_back=self.batch_ok(kv.get("last_reads")or[],job["queue"])
  if rows_back and self.batch_richer(kv.get("last_reads")or[],job["queue"]):
   self.batch_proven=True
   return"audit"
  if not self.ids_back(kv.get("last_reads")or[],job["queue"])and self._smk_batch_shrink(len(job["queue"])):
   return"audit"
  self.batch_spec=None
  self.log.append({"kind":"batch_dropped","tool":job["batch"][0]})
  purpose,floor=job["purpose"],job["open_floor"]
  if is_number(floor):
   reserve=int(floor)
  elif purpose in("menu","quota"):
   reserve=self.order_due()
  else:
   reserve=3
  count=min({"menu":0,"quota":2,"walk":2}.get(purpose,3),len(job["queue"]),self.moves_left()-reserve-1)
  if count<=0 or self.wire.done or not(job["open_tool"]or self.role("details")):
   return"audit"
  self.post_replies(job["responses"])
  job["responses"]=self.open_reads(job["queue"][:count],job["open_say"],None,job["open_tool"])
  return"audit"if job["responses"]else True
 def _smh_open_audit(self,job):
  answers=job["responses"]
  if not answers:
   return False
  extra=self.review_decline()if self.wire.spare()>CLOSING+30 and not job["recheck"]else""
  if extra:
   last_id,text=answers[-1]
   answers[-1]=(last_id,"%s\n[check after opening] %s"%(text,extra))
  self.post_replies(answers)
  return True
 def _smh_scan(self,test,take):
  got=[]
  def visit(d):
   if test(d):
    got.append(take(d))
  for rec in self.archive.items:
   walk_dicts(rec,visit)
  return got
 def last_stock(self,ident):
  states=self._smh_scan(lambda d:self.row_mark(d)==ident,self.stock_state)
  return next((s for s in states[::-1]if s is not None),None)
 def _smh_rank_ready(self,ident,version,cap):
  if ident not in self.archive.read or"%s|%s"%(ident,version)in self.passed or ident in self.refused_lines:
   return False
  if ident in self.dead_lines()or self.last_stock(ident)is False or self.field_view(ident)["fails"]:
   return False
  wants=self.needs.items()
  slot=self.product_index(ident)if wants else None
  units=self.needs.items()[slot]["quantity"]if slot is not None and slot<len(self.needs.items())else self.needs.quantity()
  cost=self.archive.price(ident)
  return is_number(cost)and not(is_number(units)and self._smh_over(cost,units,cap))
 def rank_verify(self,pick):
  try:
   if self.needs.priority()=="cheapest":
    version,cap,left=str(self.needs.version),self.needs.spend_cap,2
    for ident in self.__dict__.get("rank_better")or[]:
     if not left or self.wire.spare()<CLOSING+30:
      break
     if self._smh_rank_ready(ident,version,cap):
      left-=1
      self.check_item(ident,"ORDER")
    return self.cheaper_fit(pick)
   return""
  except Exception as err:
   self.log.append({"kind":"rank_verify_error","error":self._smh_fault(err)})
   return""
 def cheaper_fit(self,pick):
  if self.needs.priority()!="cheapest"or not pick or pick in self.pair_best():
   return""
  shop=self.archive
  ceiling=shop.price(pick)
  if ceiling is None:
   return""
  label=self.shelf_known(pick)
  gone=self.dead_lines()
  winner,low="",None
  for mark in self.passed:
   ident,_,version=mark.rpartition("|")
   if ident==pick or version!=str(self.needs.version)or ident in self.refused_lines or ident in gone:
    continue
   if self.needs.items()and(not label or label!=self.shelf_known(ident)):
    continue
   cost=shop.price(ident)
   undercuts=cost is not None and cost<ceiling and(low is None or cost<low)
   if undercuts and not self.field_view(ident)["fails"]:
    winner,low=ident,cost
  return winner if low is not None else""
 def shelf_known(self,pick):
  def carries(d):
   return any(isinstance(v,str)and v==pick and"id"in str(k).lower()for k,v in d.items())
  labels=self._smh_scan(carries,self.shelf_of)
  found=next(filter(None,reversed(labels)),"")
  return found or str((self.archive.__dict__.get("label_of")or{}).get(pick)or"").strip().lower()
 def chosen(self):
  return next(reversed(self.trolley.units),None)
 def home_shelves(self):
  cached=self.__dict__.get("home_memo")
  if cached and cached[0]==len(self.archive.items):
   return[set(part)for part in cached[1]]
  try:
   result=self._smi_rank_homes()
  except Exception:
   result=[set()for _ in(self.needs.items()or[0])]
  self.home_memo=(len(self.archive.items),[set(part)for part in result])
  return result
 def _smi_rank_homes(self):
  wanted=[prod["product"]for prod in self.needs.items()]or[str(self.needs.reading.get("product")or"")]
  labels=self._smi_listing_labels()
  kinds=[set(kind_tokens(name))for name in wanted]
  result=[]
  for n,mine in enumerate(kinds):
   others=set().union(*(group for m,group in enumerate(kinds)if m!=n))
   scored={label:kind_score(mine,set(kind_tokens(label)),others)for label in sorted(labels)}
   peak=max(scored.values(),default=0.0)
   leaders={label for label,got in scored.items()if got==peak}if peak>0 else set()
   exact={label for label in leaders if self._smi_same_kind(mine,set(kind_tokens(label)))}if len(kinds)>1 else set()
   result.append(exact or leaders)
  return result
 @staticmethod
 def _smi_same_kind(one,two):
  return all(kind_in(word,two)for word in one)and all(kind_in(word,one)for word in two)
 def _smi_listing_labels(self):
  known,labels=self.archive.ids,set()
  def spot(record):
   label=self.shelf_of(record)
   if label and any("id"in str(field).lower()and isinstance(value,str)and value in known
for field,value in record.items()):
    labels.add(label)
  for obs in self.archive.items:
   walk_dicts(obs,spot)
  return labels
 def own_shelf(self):
  try:
   if self.needs.items():
    return""
   ours=self.home_shelves()[0]
   opened=[label.strip().lower()for label in self.archive.shelves_read()]
   strays=[label for label in opened if label not in ours]
   if not ours or not strays:
    return""
   return next((label for label in opened if label in ours),min(ours))
  except Exception:
   return""
 _smi_GLUE=r"(?:\band\b|&|\bor\b|/|\+)"
 _smi_GAP_GLUE=r"\band\b|&|\bor\b|,|/|\+"
 def joined_phrase(self,a,b,label):
  said=re.sub(r"\s+"," ",self.request.lower().replace("\u2019","'"))
  exact=self._smi_label_in_text(str(label).lower(),said)
  if exact:
   return exact,True
  loose=self._smi_names_in_text(a,b,said)
  return(loose,False)if loose else None
 @classmethod
 def _smi_label_in_text(cls,label,said):
  halves=[h for h in re.split(JOINERS,label)if h.strip()]
  if len(halves)!=2:
   return""
  front,back=name_pattern(halves[0]),name_pattern(halves[1])
  if not(front and back):
   return""
  found=re.search(r"\b%s\s*%s\s*%s"%(front,cls._smi_GLUE,back),said)
  return found.group(0).strip()if found else""
 @classmethod
 def _smi_names_in_text(cls,a,b,said):
  ka,kb=kind_tokens(a),kind_tokens(b)
  if not ka or not kb or same_kind(ka[-1],kb[-1]):
   return""
  for lead,tail in((a,b),(b,a)):
   span=re.compile(r"\b(?:%s)(?P<mid>[^.;:!?]{0,40}?)(?:%s)"%(name_pattern(lead),name_pattern(tail)))
   for found in span.finditer(said):
    gap=found.group("mid")
    if len(re.findall(r"[a-z0-9]+",gap))>3 or not re.search(cls._smi_GAP_GLUE,gap)or re.search(r"\d|\bx\b",gap):
     continue
    outside=said[:found.start()]+" | "+said[found.end():]
    if not all(re.search(r"\b"+name_pattern(name),outside)for name in(a,b)):
     return found.group(0).strip()
  return""
 def product_question(self,entry,prods):
  written=str(entry)
  parsed=re.match(r'-\s*(.*?): asked "(.*?)" -> ',written,re.S)
  about,asked=parsed.groups()if parsed else(written,written)
  pools=[kind_tokens(prod["product"])for prod in prods]
  for tag in re.findall(r"\(([^()]*)\)",about):
   if any(kind_in(term,pool)for term in kind_tokens(tag)for pool in pools):
    return True
  plain=asked.lower().replace("\u2019","'")
  for prod in prods:
   pat=name_pattern(prod["product"])
   if pat and re.search(r"\b"+pat,plain):
    return True
  return False
 @staticmethod
 def joined_clash(a,b):
  def settings(prod):
   got=[]
   for req in prod.get("requirements")or[]:
    name,_,value=str(req).partition(":")
    got.append((re.split(r"\s*\(",name)[0].strip()," ".join(re.findall(r"[a-z0-9]+",value.lower()))))
   return got
  theirs=settings(b)
  return any(same_prop(name_a,name_b)and said_a!=said_b for name_a,said_a in settings(a)for name_b,said_b in theirs)
 def merge_joined(self,obs):
  book=self.__dict__.setdefault("joined_seen",{})
  walk_dicts(obs,lambda record:self._smi_keep_label(record,book))
  said=[]
  run={"rounds":0,"prods":[],"pick":None}
  def gate():
   if run["rounds"]>=2:
    return"done"
   run["prods"]=self.needs.items()
   return"scan"if self._smi_merge_allowed(run["prods"],book)else"done"
  def scan():
   run["pick"]=self._smi_joined_pair(run["prods"],book)
   return"merge"if run["pick"]else"done"
  def merge():
   said.append(self.merge_items(*run["pick"]))
   run["rounds"]+=1
   return"gate"
  handlers={"gate":gate,"scan":scan,"merge":merge}
  state="gate"
  while state in handlers:
   state=handlers[state]()
  return" ".join(part for part in said if part)
 @staticmethod
 def _smi_keep_label(record,book):
  if not any("id"in str(field).lower()and isinstance(value,str)and re.fullmatch(ID_LIKE,value)
for field,value in record.items()):
   return
  for field,value in record.items():
   name=str(field).lower()
   if not isinstance(value,str)or not 0<len(value.strip())<=60 or"id"in name:
    continue
   if re.search(r"categor|shelf|department|aisle",name):
    text=value.strip()
    if len([h for h in re.split(JOINERS,text.lower())if h.strip()])==2:
     book.setdefault(text.lower(),text)
    return
 def _smi_merge_allowed(self,prods,book):
  cart,wants=self.trolley,self.needs
  if len(prods)<2 or not book or cart.used or cart.units or wants.uncovered or wants.indifferent:
   return False
  return not any(self.product_question(entry,prods)for entry in self.questions_log)
 def _smi_joined_pair(self,prods,book):
  terms=[kind_tokens(prod["product"])for prod in prods]
  couples=[(p,q)for p in range(len(prods))for q in range(p+1,len(prods))]
  for low in sorted(book):
   halves=[kind_tokens(h)for h in re.split(JOINERS,low)if h.strip()]
   if len(halves)!=2 or not all(halves):
    continue
   for p,q in couples:
    one,two=prods[p],prods[q]
    if not self._smi_halves_cover(terms[p],terms[q],halves)or one["quantity"]!=two["quantity"]:
     continue
    if self.joined_clash(one,two):
     continue
    got=self.joined_phrase(one["product"],two["product"],book[low])
    if got:
     return p,q,book[low],got[0],got[1]
  return None
 @staticmethod
 def _smi_same_kinds(x,y):
  return kinds_within(x,y)and kinds_within(y,x)
 @staticmethod
 def _smi_sided(terms,side,away):
  return all(kind_in(t,side)and not kind_in(t,away)for t in terms)
 @classmethod
 def _smi_halves_cover(cls,ta,tb,halves):
  if not(ta and tb)or cls._smi_same_kinds(ta,tb)or not kinds_within(ta+tb,halves[0]+halves[1]):
   return False
  solo_a=[t for t in ta if not kind_in(t,tb)]
  solo_b=[t for t in tb if not kind_in(t,ta)]
  for near,far in(halves,halves[::-1]):
   if cls._smi_sided(solo_a,near,far)and cls._smi_sided(solo_b,far,near)and any(kind_in(t,near)for t in ta)and any(kind_in(t,far)for t in tb):
    return True
  return False
 def merge_items(self,first,other,label,phrase,in_order=True):
  doc=self.needs.reading
  prods=self.needs.items()
  listed=[entry for entry in doc.get("items")or[]if isinstance(entry,dict)and str(entry.get("product")or"").strip()]
  if len(listed)!=len(prods)or not 0<=first<other<len(prods):
   return""
  remaining=len(prods)-1
  new_tag=phrase if remaining>=2 else""
  ours,retag=self._smi_tag_tools(prods,first,other,label,new_tag)
  size=prods[first]["quantity"]
  unit={"product":phrase,"quantity":size,
"requirements":self._smi_union_reqs(prods[first]["requirements"]+prods[other]["requirements"])}
  survivors=[unit if n==first else entry for n,entry in enumerate(listed)if n!=other]
  self._smi_retag_everywhere(ours,retag,phrase,new_tag)
  if remaining>=2:
   doc["items"]=survivors
   doc["product"]=" and ".join(str(entry.get("product"))for entry in survivors)
  else:
   doc.pop("items",None)
   doc.update(product=phrase,quantity=size)
  wants=self.needs
  wants.version+=1
  own=self.__dict__
  for cache in("home_memo","menu_memo","quota_reads","walked"):
   own.pop(cache,None)
  count=len(wants.items())
  pace=40.0/max(1,self.step_cap)
  self.wire.scale=(1.0+0.3*sorted((0,count-1,2))[1])*max(0.5,min(1.0,pace))
  self.log.append(dict(kind="merge_joined",label=label,products=[prods[first]["product"],prods[other]["product"]],
into=phrase,label_words=in_order,products_left=max(1,count)))
  layout="one line for that product"if remaining>=2 else"exactly ONE line"
  return(f"[merged products] This store lists '{label}' as just one category, so '{phrase}' is a single product "
f"here, not two. The order therefore carries {layout} - never a separate line per kind named in that "
"category title. Current requirement list: "+wants.text_body())
 def _smi_tag_tools(self,prods,first,other,label,new_tag):
  pair=kind_tokens(prods[first]["product"])+kind_tokens(prods[other]["product"])
  span=kind_tokens(label)
  rivals=[kind_tokens(prod["product"])for n,prod in enumerate(prods)if n not in(first,other)]
  fill=" (%s)"%new_tag if new_tag else""
  def ours(tag):
   words=kind_tokens(tag)
   if not words or not kinds_within(words,span):
    return False
   if not any(kind_in(t,pair)for t in words):
    return False
   return all(not self._smi_same_kinds(words,rival)for rival in rivals)
  def swap(match):
   return fill if ours(match.group(1))else match.group(0)
  def retag(text):
   if isinstance(text,str):
    name,colon,value=text.partition(":")
    return re.sub(r"\s*\(([^()]*)\)",swap,name)+colon+value
   return text
  return ours,retag
 @staticmethod
 def _smi_union_reqs(reqs):
  merged=[]
  lowered=set()
  for req in reqs:
   name,_,value=str(req).partition(":")
   text="%s:%s"%(re.sub(r"\s*\([^()]*\)","",name).strip(),value)
   if text.lower()not in lowered:
    lowered.add(text.lower())
    merged.append(text)
  return merged
 def _smi_retag_everywhere(self,ours,retag,phrase,new_tag):
  def fresh(seq):
   out=[]
   for item in map(retag,seq):
    if item not in out:
     out.append(item)
   return out
  wants=self.needs
  for field in("written","uncovered","let_go","indifferent"):
   setattr(wants,field,fresh(getattr(wants,field)))
  if wants.now_reqs is not None:
   wants.now_reqs=fresh(wants.now_reqs)
  wants.resolved=list(map(retag,wants.__dict__.get("resolved")or[]))
  doc=wants.reading
  for field in("requirements","hints","references","soft_wants"):
   if isinstance(doc.get(field),list):
    doc[field]=fresh(doc[field])
  aims=doc.get("reference_targets")
  for aim in aims if isinstance(aims,list)else[]:
   if isinstance(aim,dict)and ours(str(aim.get("product")or"")):
    aim["product"]=phrase
  own=self.__dict__
  for pointer in own.get("refs")or[]:
   pointer.update({field:retag(pointer[field])for field in("req","head","now")if isinstance(pointer.get(field),str)})
   if ours(str(pointer.get("tag")or"")):
    pointer["tag"]=new_tag
  for field in("pending_seconds","open_pair"):
   if isinstance(own.get(field),list):
    own[field]=list(map(retag,own[field]))
 def lookup_key(self,tool):
  spec=self.shapes.get(tool or"")or{}
  params=spec.get("properties")if isinstance(spec.get("properties"),dict)else{}
  required=spec.get("required")
  required=[name for name in required or[]if name in params]if isinstance(required,list)else[]
  free=[name for name,info in params.items()
if isinstance(info,dict)and info.get("type")=="string"and not info.get("enum")]
  pick=next(filter(required.__contains__,free),None)or(free[0]if len(free)==1 else None)
  if not pick or any(name!=pick for name in required):
   return None
  return pick
 def lookup_tools(self):
  busy=set(map(self.role,("search","message","cart","cart_view","order","decline")))
  busy.discard(None)
  return[tool for tool in self.shapes if tool not in busy and self.lookup_key(tool)]
 def lookup_call(self,ident,tool):
  if not tool:
   return None
  if tool==self.role("details"):
   return self.details_call(ident)
  known=self.archive.ids
  def listing(value):
   return isinstance(value,str)and value in known
  for sent in self._smi_past_args(tool):
   if any(map(listing,sent.values())):
    return{name:ident if listing(value)else value for name,value in sent.items()}
  slot=self.lookup_key(tool)
  return{slot:ident}if slot else None
 def _smi_past_args(self,tool):
  for record in self.log[::-1]:
   if record.get("kind")!="env":
    continue
   for step in record.get("actions")or[]:
    if step.get("name")==tool and isinstance(step.get("args"),dict):
     yield step["args"]
 @staticmethod
 def rec_id(d):
  fallback=None
  for field,value in d.items():
   low=str(field).lower()
   if not low.endswith("id")or not isinstance(value,str)or not re.fullmatch(ID_LIKE,value):
    continue
   if low=="id":
    return value
   if fallback is None:
    fallback=value
  return fallback
 @staticmethod
 def asked_ids(call_args):
  found=set()
  for value in(call_args.values()if isinstance(call_args,dict)else()):
   for piece in(value if isinstance(value,list)else[value]):
    if isinstance(piece,str)and re.fullmatch(ID_LIKE,piece):
     found.add(piece)
  return found
 _smi_CAP_KEYS=("id","stock","props","notes","price")
 _smi_OTHER_SOURCE={"row":"lookup","lookup":"row"}
 def learn_caps(self,call_name,call_args,obs):
  tool=str(call_name)
  own=self.__dict__
  tally=own.setdefault("tool_caps",{}).setdefault(tool,{"calls":0,**dict.fromkeys(self._smi_CAP_KEYS,0)})
  tally["calls"]+=1
  job={name:own.setdefault(name,make())for name,make in(
("dead",set),("dead_by",dict),("stock_by",dict),("row_text",dict),("row_keys",set))}
  job.update(tool=tool,call=call_name,named=self.asked_ids(call_args),shown=dict.fromkeys(self._smi_CAP_KEYS,False),
keys=own.setdefault("tool_keys",{}).setdefault(tool,set()),
note_keys=own.setdefault("note_keys",{}).setdefault(tool,set()),
carts={self.role("cart"),self.role("cart_view")}-{None},epoch=own.get("cart_seq",0),newly=[])
  if not(isinstance(obs,dict)and obs.get("error")):
   walk_dicts(obs,lambda record:self._smi_caps_visit(job,record))
  for cap,flag in job["shown"].items():
   tally[cap]+=int(flag)
  return job["newly"]
 def _smi_caps_visit(self,job,record):
  lid=self.rec_id(record)
  if not lid:
   return
  stocked=self.stock_state(record)
  named=lid in job["named"]
  via="lookup"if named or job["call"]in job["carts"]else"row"
  if stocked is not None:
   self._smi_caps_stock(job,lid,stocked,via)
  if named:
   self._smi_caps_named(job,lid,record,stocked)
  elif job["call"]not in job["carts"]:
   job["row_keys"].update(map(str,record))
   job["row_text"].setdefault(lid,{}).update((str(field),value)for field,value in record.items()
if isinstance(value,str))
 def _smi_caps_stock(self,job,lid,stocked,via):
  self.stock_seen=True
  history=job["stock_by"].setdefault(lid,{})
  opposite,last_row=history.get(self._smi_OTHER_SOURCE[via]),history.get("row")
  if opposite and opposite[0]==job["epoch"]and opposite[1]!=stocked:
   self.rows_lag=True
  if via=="row"and last_row and last_row[1]!=stocked:
   self.rows_live=True
  history[via]=(job["epoch"],stocked)
  gone,fresh=job["dead"],job["newly"]
  if stocked is not False:
   if via=="lookup"or job["dead_by"].get(lid)!="lookup":
    gone.discard(lid)
   return
  if lid not in gone and lid in self.trolley.units and lid not in fresh:
   fresh.append(lid)
  gone.add(lid)
  job["dead_by"][lid]=via
 @staticmethod
 def _smi_has_map(record):
  return any(isinstance(inner,dict)and len([x for x in inner.values()if isinstance(x,(str,int,float))])>=2
for inner in record.values())
 def _smi_caps_named(self,job,lid,record,stocked):
  shown,tool,own=job["shown"],job["tool"],self.__dict__
  shown["id"]=True
  own.setdefault("read_with",{}).setdefault(tool,set()).add(lid)
  own.setdefault("text_keys",{}).setdefault(tool,set()).update(
str(field)for field,value in record.items()if value is None or isinstance(value,str))
  if stocked is not None:
   shown["stock"]=True
  if not shown["props"]:
   shown["props"]=self._smi_has_map(record)
  if not shown["price"]:
   shown["price"]=any(is_number(value)and bool(re.search(r"price|cost",str(field).lower()))
for field,value in record.items())
  job["keys"].update(map(str,record))
  seen_row=job["row_text"].get(lid)
  for field,value in record.items():
   if not isinstance(value,str)or len(value.split())<8:
    continue
   name=str(field)
   if(name not in job["row_keys"])if seen_row is None else seen_row.get(name)!=value:
    job["note_keys"].add(name)
    shown["notes"]=True
 def stock_tool(self):
  tally=self.__dict__.get("tool_caps")or{}
  pool=self.lookup_tools()
  def seen(tool,cap):
   return(tally.get(tool or"")or{}).get(cap,0)
  def proven():
   showed=[tool for tool in pool if seen(tool,"stock")>0]
   if not showed:
    return None
   view,details=self.role("view"),self.role("details")
   return max(showed,key=lambda tool:(tally[tool]["stock"],tool==view,tool==details,tool))
  def named():
   guess=self._smi_role_guess()
   for cand in(self.role("view"),guess.get("view"),self.role("details")):
    if cand and cand in pool and(seen(cand,"id")<2 or seen(cand,"stock")!=0):
     return cand
   return None
  def probe():
   guess=self._smi_role_guess()
   claimed={guess.get(r)for r in("cart","cart_view","order","decline","message")}-{None}
   return next((tool for tool in pool if tool not in claimed and not seen(tool,"id")),None)
  for stage in(proven,named,probe):
   found=stage()
   if found is not None:
    return found
  return None
 def _smi_role_guess(self):
  guess=self.__dict__.get("role_guess")
  if guess is None:
   try:
    guess=self.code_roles()
   except Exception:
    guess={}
   self.role_guess=guess
  return guess
 def notes_tool(self):
  own=self.__dict__
  tally=own.get("tool_caps")or{}
  reader=self.role("details")
  usual=set((own.get("tool_keys")or{}).get(reader or"",set()))
  noted=own.get("note_keys")or{}
  extra=[tool for tool in self.lookup_tools()if tool!=reader and set(noted.get(tool)or set())-usual]
  if not extra:
   return None
  return max(extra,key=lambda tool:((tally.get(tool)or{}).get("notes",0),tool))
 def sheet_role_check(self):
  own=self.__dict__
  reader=self.role("details")
  if own.get("sheet_moved")or not reader:
   return
  tally=own.get("tool_caps")or{}
  mine=tally.get(reader)or{}
  if mine.get("id",0)<2 or mine.get("props",0):
   return
  richer=[tool for tool in self.lookup_tools()if tool!=reader and(tally.get(tool)or{}).get("props",0)]
  if not richer:
   return
  sheet=max(richer,key=lambda tool:(tally[tool].get("props",0),tool))
  self.sheet_moved=True
  self.role_map["details"]=sheet
  if self.role("view")==sheet:
   self.role_map["view"]=reader
  mapped=set()
  def take(rec):
   ident=_smc_record_id(rec)
   if ident and _smc_has_prop_map(rec):
    mapped.add(ident)
  for obs in self.archive.items:
   walk_dicts(obs,take)
  self.archive.read&=mapped
  self.log.append({"kind":"sheet_role","from":reader,"to":sheet})
 def dead_lines(self):
  return set(self.__dict__.get("dead")or())
 def stale_lines(self):
  own=self.__dict__
  last_seen,mark=own.get("line_seq")or{},own.get("cart_seq",0)
  return[held for held in self.trolley.units if last_seen.get(held,-1)<mark]
 _smi_LOOSE=re.compile(r"\b(?:"+VALUE_LINKS+r")\b|\d\s*(?:-|to)\s*\d|[+<>]|\b(?:about|around|approx\w*|roughly)\b")
 def dead_swap(self,line):
  job=self._smi_swap_context(line)
  pick=self._smi_swap_checked(job)
  if pick:
   return pick,""
  if job["several"]and not job["kind"]:
   return"",""
  return"",self._smi_swap_rows(job)
 def _smi_unit_cost(self,cand):
  quoted=self.archive.price(cand)
  return float("inf")if not is_number(quoted)else float(quoted)
 def _smi_swap_context(self,line):
  job={"line":line,"dead":self.dead_lines(),"ver":str(self.needs.version)}
  job["kind"]=self.shelf_known(line)
  job["several"]=len(self.needs.items())>=2
  job["cap"],job["qty"]=self.needs.spend_cap,max(1.0,self.trolley.held(line))
  total=0
  for held in self.trolley.units:
   if held!=line and self._smi_unit_cost(held)!=float("inf"):
    total+=self._smi_unit_cost(held)*self.trolley.held(held)
  job["rest"]=total
  return job
 def _smi_swap_blocked(self,job,cand):
  return cand==job["line"]or cand in job["dead"]or cand in self.refused_lines or cand in self.trolley.units
 def _smi_over_cap(self,job,amount):
  return is_number(job["cap"])and amount*job["qty"]+job["rest"]>job["cap"]
 def _smi_swap_checked(self,job):
  kind,several=job["kind"],job["several"]
  options=[]
  for mark in self.passed:
   cand,_,ver=mark.rpartition("|")
   if ver!=job["ver"]or self._smi_swap_blocked(job,cand):
    continue
   if self.field_view(cand)["fails"]:
    continue
   label=self.shelf_known(cand)
   if kind and(label!=kind if several else label not in("",kind)):
    continue
   amount=self._smi_unit_cost(cand)
   if amount!=float("inf")and self._smi_over_cap(job,amount):
    continue
   options.append((amount,cand))
  return min(options)[1]if options else""
 def _smi_swap_rows(self,job):
  kind=job["kind"]
  k=next((n for n,home in enumerate(self.home_shelves())if kind in home),None)if job["several"]else None
  stated=[(name,wanted)for name,wanted in self.code_values(k)if not self._smi_LOOSE.search(wanted.lower())]
  ver=job["ver"]
  failed={mark.split("|")[0]for mark,bad in self.rulings.items()if bad and mark.split("|")[1:2]==[ver]}
  pointers=self.ref_pointers()
  ranked=[]
  for cand,card in self.code_rows().items():
   amount=card["price"]if is_number(card["price"])else self._smi_unit_cost(cand)
   if self._smi_swap_blocked(job,cand)or card["stock"]is not True or(kind and card["shelf"]!=kind):
    continue
   if amount==float("inf")or self._smi_over_cap(job,amount):
    continue
   if cand in failed or cand in pointers or self.ref_excluded(card["fields"],k):
    continue
   opened=cand in self.archive.read
   if opened and not self._smi_row_shows_all(cand,card,stated):
    continue
   ranked.append((0 if opened else 1,amount,cand))
  ranked.sort()
  unread=[entry for entry in ranked if entry[0]==1]
  if unread and self.needs.priority()=="cheapest"and all(unread[0][1]<cost for flag,cost,_ in ranked if flag==0):
   ranked.remove(unread[0])
   ranked.insert(0,unread[0])
  top=ranked[:3]
  self.swap_unread=[cand for flag,_,cand in top if flag==1]
  self.swap_page=None
  if all(flag!=0 for flag,_,_ in ranked):
   try:
    self.swap_page=self.dead_page(k)
   except Exception:
    self.swap_page=None
  return", ".join(self._smi_row_label(*entry)for entry in top)
 def _smi_row_shows_all(self,cand,card,stated):
  if self.field_view(cand)["fails"]:
   return False
  for name,wanted in stated:
   if self.code_clash(card["fields"],name,wanted):
    return False
  return all(self.code_shows(card,name,wanted,set(),True)for name,wanted in stated)
 @staticmethod
 def _smi_row_label(flag,amount,cand):
  shown=int(amount)if float(amount).is_integer()else amount
  return f"{cand} at {shown}"+(" [opened; shows every stated value]"if flag==0 else"")
 def dead_page(self,k):
  pager=self.page_arg()
  if not pager:
   return None
  best,best_rank=None,None
  for trail in self.__dict__.get("page_book",{}).values():
   carried=self.walk_carries(trail["args"],k)
   if not carried or self.next_page(trail)is None:
    continue
   rank=(len(carried),-len(trail["read"]))
   if best is None or rank>best_rank:
    best,best_rank=trail,rank
  return None if best is None else{**best["args"],pager[0]:self.next_page(best)}
 def swap_open(self):
  own=self.__dict__
  if own.get("swap_done"):
   return""
  gone=self.dead_lines()
  for cand in own.get("swap_unread")or[]:
   if cand not in self.archive.read and cand not in gone and cand not in self.trolley.units:
    return cand
  return""
 def dead_note(self,lines):
  self.event_seen=True
  return" ".join(self._smi_sold_out_text(held)for held in lines)
 def _smi_sold_out_text(self,held):
  self.swap_page,self.swap_unread=None,[]
  pick,rows=self.dead_swap(held)
  head=(f"[stock] Cart line {held} can no longer be bought: the latest record of it reports no availability, "
"and a checkout that still carries it would be rejected. Plan:")
  close=" (c) confirm its availability and cost with one more read; (d) check out."
  if pick:
   least=", and it is also the lowest-priced approved option"if self.needs.priority()=="cheapest"else""
   text=(f"{head} (a) drop {held}; (b) add {pick} in its place - the item check approved it under the "
f"current requirement list{least};"+close)
  elif rows:
   text=(f"{head} (a) drop {held}; (b) no approved substitute for this product exists yet, so try these "
f"low-priced, available rows of its own category that the spending limit still permits, in order: "
f"{rows} (rows whose opened record lacks or breaks a stated requirement are already filtered out; "
"open the details of one not yet opened before adding it) and add the first that passes;"+close)
  else:
   text=f"{head} drop {held} and search for a different listing that satisfies the request."
  page=self.__dict__.get("swap_page")
  if page and not pick:
   text+=(" Another page of rows of this kind is available - the search led by a stated value still has an "
"unread page; call it with "+json.dumps(page,default=str)+".")
  return text
 def refusal_reread(self,obs=None):
  try:
   note,gone=self.refusal_gone(obs)
  except Exception as fault:
   self.log.append(dict(kind="refusal_gone_error",error=(fault.__class__.__name__+": "+str(fault))[:300]))
   return self.refusal_queue()
  return note+(gone if gone else self.refusal_queue())
 _smj_LOOSE=re.compile(r"\b(?:"+VALUE_LINKS+r")\b|\d\s*(?:-|to)\s*\d|[+<>]|\b(?:about|around|approx\w*|roughly)\b")
 _smj_MARKS=re.compile(r"\((?:stated|revealed[^)]*)\)")
 _smj_HEAD_CUT=re.compile(r"\s*\(")
 _smj_DONE_NOTE=(" What to do: send the order for exactly this cart. Keep the decline for the case where something "
"you opened proves a cart line wrong under the current requirements.")
 def _smj_run_flow(self,table,ctx,state):
  while state in table:
   state=getattr(self,table[state])(ctx)
  return state
 def _smj_log(self,kind,**fields):
  self.log.append(dict(kind=kind,**fields))
 def _smj_fail(self,kind,err):
  self._smj_log(kind,error=("%s: %s"%(type(err).__name__,err))[:300])
 def _smj_quiet(self):
  return bool(self.wire.mute or self.__dict__.get("covering"))
 def _smj_for_product(self,req,product):
  head=str(req).partition(":")[0]
  return not self.needs.items()or self.product_of(head)in(product,None)
 def _smj_need_stamp(self):
  n=self.needs
  return n.version,len(n.let_go),len(n.uncovered),str(n.spend_cap),"|".join(n.props_left())
 def refusal_gone(self,reply):
  cart=self.trolley
  single=[line for line in cart.units if cart.held(line)>0]
  if reply is None or len(single)!=1 or self._smj_quiet()or self.__dict__.get("last_kind")!="order":
   return"",""
  texts=self.shopper_lines(self.role("order")or"",reply,True)
  if not texts:
   return"",""
  first=self._smj_need_stamp()
  note=self.needs.hear(texts,None)
  then=self._smj_need_stamp()
  lead=" "+note if note else""
  if not note or then[0]==first[0]or then[1:]!=first[1:]:
   return lead,""
  only=single[0]
  mem=self.__dict__
  mem.setdefault("dead",set()).add(only)
  mem.setdefault("dead_by",{})[only]="lookup"
  self._smj_log("refusal_dead",line=only)
  why=(" Marked sold out without a new read: %s was the single line in the cart, and the store's refusal says "
"an item can no longer be had. "%only)
  return lead,why+self.dead_note([only])
 def refusal_queue(self):
  reader=self.stock_tool()
  first_four=list(self.trolley.units)[:4]
  if not(first_four and reader)or self._smj_quiet()or self.__dict__.get("open_queue"):
   return""
  if self.moves_left()<3+len(first_four)or self.wire.spare()<CLOSING:
   return""
  self.__dict__.update(open_queue=first_four,open_recheck=True,open_purpose="cart",open_tool=reader,
open_say="Order refused, so I check each cart line's stock again in case one sold out.")
  return" Stock re-read queued: each cart line is looked up again with %s."%reader
 def code_judges_all(self):
  count=len(self.needs.items())
  scopes=list(range(count))if count else[None]
  judged=sum(self.ed_plain(pair[1])for scope in scopes for pair in self.code_values(scope))
  owners=self.differ_done(None)
  in_force=list(self.needs.props_left())
  judged+=len([req for req in self.needs.props_left()if any(self.ref_match(o,req)for o in owners)])
  return judged>=len(in_force)
 def differ_done(self,product):
  several,in_force=bool(self.needs.items()),self.needs.props_left()
  found=[]
  for ref in self.__dict__.get("refs")or[]:
   settled=(ref.get("relation"),ref.get("status"))==("differ","resolved")
   if not settled or str(ref.get("value")or"").strip()==""or not self.ref_live(ref):
    continue
   if any(self.ref_match(ref,req)for req in in_force):
    tag=ref.get("tag")if several and product is not None else None
    if not tag or self.product_of("x ("+str(ref["tag"])+")")in(product,None):
     found.append(ref)
  return found
 _smj_REPAIR=dict(gate="_smj_dr_gate",proof="_smj_dr_proof",fill="_smj_dr_fill")
 def decline_repair(self):
  if self._smj_run_flow(self._smj_REPAIR,{},"gate")!="ordered":
   return""
  plural=bool(self.needs.items())
  whom,put=(" across all products","them")if plural else("","it")
  return("Not run yet (no step used). Decline held: a fitting candidate exists%s in the records - the item "
"check passed it, or its opened record writes every requested value - and turning the shopper away suits "
"only a case where nothing fits. The code has put %s in the cart and checked the stock again. Cart: %s"
%(whom,put,self.trolley.summary())+self._smj_DONE_NOTE)
 def _smj_dr_gate(self,ctx):
  refused=self.refusal_n
  several=len(self.needs.items())>=2
  reserve=self.moves_left()<=self.order_due()
  if refused>0 or reserve:
   mark=refused
  elif refused==0:
   mark=("early",refused)
  else:
   return"pass"
  w=self.wire
  if any((not self.role("cart"),not self.role("order"),self.moves_left()<3,w.spare()<=CLOSING,
self._smj_quiet(),w.done)):
   return"pass"
  tried=self.__dict__.setdefault("repair_tried",set())
  if mark in tried:
   return"pass"
  tried.add(mark)
  ctx["several"]=several
  return"proof"if several else"fill"
 def _smj_dr_proof(self,ctx):
  try:
   unfit=self.ed_proof()
  except Exception as err:
   unfit=[]
   self._smj_fail("decline_proof_error",err)
  if unfit:
   self._smj_log("decline_proof",products=unfit)
   return"pass"
  return"fill"
 def _smj_dr_fits(self,picks,ver,own):
  evidence=self.__dict__.get("code_fit")or{}
  def holds(ident):
   if f"{ident}|{ver}"in self.passed:
    return True
   return bool(own and ident in self.archive.read and evidence.get(ident))
  return bool(picks)and all(holds(pick[0])for pick in picks)
 def _smj_dr_fill(self,ctx):
  sold_out=self.dead_lines()
  unread=set(self.stale_lines())
  backup=set(self.refused_lines)
  keep={line for line in self.trolley.units if not(line in sold_out or line in unread)}
  seek={"ver":str(self.needs.version),"own":self.code_judges_all(),"several":ctx["several"],
"avoid":set(sold_out),"seen":{},"checks":0,"rounds":0}
  placed=False
  try:
   self.refused_lines.difference_update(keep)
   if seek["several"]:
    seen=seek["seen"]
    for product in range(len(self.needs.items())):
     seen.update((e["id"],e)for e in self.ed_pool(product))
    seek["avoid"].update(ident for ident,e in seen.items()if e["clash"])
   self._smj_run_flow(self._smj_SEEK,seek,"pick")
   chosen=seek["picks"]
   if not self._smj_dr_fits(chosen,seek["ver"],seek["own"]):
    self._smj_log("decline_repair",picks=chosen,done=False)
   elif self.moves_left()<2+len(chosen):
    self._smj_log("decline_repair",picks=chosen,done=False,why="steps")
   else:
    placed=self._smj_place(chosen)
    self._smj_log("decline_repair",picks=chosen,done=placed,cart=dict(self.trolley.units))
  except Exception as err:
   placed=False
   self._smj_fail("decline_repair_error",err)
  finally:
   if not placed:
    self.refused_lines.clear()
    self.refused_lines.update(backup)
  return"ordered"if placed else"pass"
 _smj_SEEK=dict(pick="_smj_seek_pick",judge="_smj_seek_judge",drop="_smj_seek_drop",check="_smj_seek_check")
 def _smj_seek_pick(self,seek):
  seek["picks"]=self.code_picks(seek["avoid"])
  return"judge"
 def _smj_seek_judge(self,seek):
  if not(seek["several"]and seek["picks"])or seek["checks"]>=3 or seek["rounds"]>=30:
   return"end"
  if self.wire.spare()<=CLOSING+30:
   return"end"
  ver,own,seen=seek["ver"],seek["own"],seek["seen"]
  waiting=[pick[0]for pick in seek["picks"]if not self._smj_dr_fits([(pick[0],1)],ver,own)]
  if not waiting:
   return"end"
  seek["bare"]=[ident for ident in waiting if not seen.get(ident)or self._smj_bare(seen[ident])]
  seek["first"]=waiting[0]
  return"drop"if seek["bare"]else"check"
 @staticmethod
 def _smj_bare(entry):
  return not entry["revealed_ok"]or len(entry["misses"])>0
 def _smj_seek_drop(self,seek):
  ver=seek["ver"]
  barred=seek["avoid"]
  barred.update(seek["bare"])
  barred.update(ident for ident,e in seek["seen"].items()
if not e["read"]and f"{ident}|{ver}"not in self.passed and self._smj_bare(e))
  seek["rounds"]+=1
  return"pick"
 def _smj_seek_check(self,seek):
  seek["checks"]+=1
  self.check_item(seek["first"],"ORDER")
  return"pick"
 def _smj_place(self,picks):
  cart,w=self.trolley,self.wire
  muted,w.mute=w.mute,True
  try:
   in_cart=self.code_put(picks)==[]and all(p[0]in cart.units for p in picks)
   wanted={p[0]for p in picks}
   reread=[line for line in self.stale_lines()if line in wanted]
   if in_cart and reread and self.moves_left()-len(reread)>=2:
    _,sold=self.code_recheck(reread)
    in_cart=not sold
  finally:
   w.mute=muted
  return in_cart and len(cart.units)>0 and not(set(cart.units)&self.dead_lines())
 @staticmethod
 def ed_plain(text):
  return Errand._smj_LOOSE.search(str(text).lower())is None
 @staticmethod
 def ed_word(text):
  low=str(text).lower()
  tests=(lambda t:re.search(r"[a-z]{3,}",t)is not None,lambda t:re.search(r"\d",t)is None,Errand.ed_plain,
lambda t:re.search(EXCLUDE_WORDS,t)is None,lambda t:not ref_text(t))
  return all(test(low)for test in tests)
 def ed_reqs(self,product):
  written=self.needs.written
  rows=[]
  for req in filter(lambda r:self._smj_for_product(r,product),self.needs.props_left()):
   name,_,said=str(req).partition(":")
   rows.append((self._smj_HEAD_CUT.split(name)[0].strip(),self._smj_MARKS.sub("",said).strip(),req not in written))
  return rows
 def ed_pool(self,product):
  needs,files=self.needs,self.archive
  several=bool(needs.items())
  ver=str(needs.version)
  struck=[said for said in self.rulings.values()if said]
  key=(product,ver,len(files.items),len(files.read),len(self.passed),len(struck),"|".join(needs.props_left()),
str(needs.spend_cap),len(self.__dict__.get("refs")or[]),frozenset(self.dead_lines()),
frozenset(self.refused_lines))
  cache=self.__dict__.setdefault("ed_memo",{})
  if key not in cache:
   if len(cache)>64:
    cache.clear()
   shelves=self._smj_pool_home(product,several)
   cache[key]=self._smj_pool_rows(product,several,ver,shelves)if shelves else[]
  return cache[key]
 def _smj_pool_home(self,product,several):
  homes=self.home_shelves()
  if not several:
   labels=set(self.shelf_labels())
   return labels if labels else set(homes[0]if homes else())
  if product is None or product>=len(homes):
   return set()
  return set(homes[product])
 def _smj_pool_rules(self,k,several,ver):
  counts=self.needs.counts()
  if several and k is not None and k<len(counts):
   units=counts[k]
  else:
   units=counts[0]if counts else 1
  scope=k if several else None
  values=self.code_values(scope)
  joined=[]
  for prop,value in values:
   if"+"not in value:
    continue
   parts=[bit.strip()for bit in value.split("+")if bit.strip()]
   if len(parts)>=2 and all(re.search(r"[a-z]{2,}",bit.lower())for bit in parts):
    joined.append((prop,parts))
  budget=self.needs.spend_cap
  failed,live=set(),set()
  for name,said in self.rulings.items():
   bits=name.split("|")
   if said and bits[1:2]==[ver]:
    failed.add(bits[0])
  for name in self.passed:
   head,_,tail=name.rpartition("|")
   if tail==ver:
    live.add(head)
  return dict(units=units,scope=scope,failed=failed,live=live,joined=joined,
cap=float(budget)if is_number(budget)else None,
judged=[pair for pair in values if self.ed_plain(pair[1])],
shown=[(prop,value)for prop,value,revealed in self.ed_reqs(scope)if revealed],
pointers=self.ref_pointers(),dead=self.dead_lines())
 def _smj_clashes(self,fields,rules):
  found=[]
  for prop,value in rules["judged"]:
   if self.code_clash(fields,prop,value):
    written=[said for name,said in fields.items()if same_prop(name,prop)]
    if not any(self.code_seen(value,said)for said in written):
     found.append(prop)
  for prop,parts in rules["joined"]:
   named=[said for name,said in fields.items()if same_prop(name,prop)]
   if named and not all(any(self.code_seen(part,said)for said in named)for part in parts):
    found.append(prop)
  return found
 def _smj_pool_rows(self,k,several,ver,home):
  rules=self._smj_pool_rules(k,several,ver)
  cap,files=rules["cap"],self.archive
  pool=[]
  for ident,row in[pair for pair in self.code_rows().items()if pair[1]["shelf"]in home]:
   seen=ident in files.read
   listed=files.price(ident)if row["price"]is None else row["price"]
   price=float(listed)if is_number(listed)else None
   gone=ident in rules["dead"]or row["stock"]is False or ident in self.refused_lines
   over=not(cap is None or price is None)and price*rules["units"]>cap+1e-9
   pointer=ident in rules["pointers"]or self.ref_excluded(row["fields"],rules["scope"])
   clash=self._smj_clashes(row["fields"],rules)
   misses=[pair for pair in rules["judged"]if not self.code_shows(row,pair[0],pair[1],set(),True)]
   ruled=ident in rules["failed"]or bool(seen and self.field_view(ident)["fails"])
   verified=ident in rules["live"]and not gone
   excluded=bool(gone or over or pointer or ruled or clash)
   shows_needs=all(self.code_shows(row,prop,value,set(),True)for prop,value in rules["shown"])
   pool.append(dict(id=ident,price=price,read=seen,verified=verified,
fit=(verified or(seen and not misses and not clash))and not(ruled or pointer or gone or over),
gone=gone,over=over,pointer=pointer,ruled=ruled,clash=clash,misses=misses,row=row,
out=excluded,revealed_ok=shows_needs))
  return pool
 def ed_proof(self):
  unfit=[]
  for product in range(len(self.needs.items())):
   verdicts={e["out"]for e in self.ed_pool(product)if e["read"]}
   if verdicts=={True}:
    unfit.append(product)
  return unfit
 def slack_rows(self):
  needs=self.needs
  n_products=len(needs.items())
  budget=needs.spend_cap
  if n_products<2 or not is_number(budget):
   return[]
  try:
   return self._smj_slack(n_products,budget)
  except Exception as err:
   self._smj_fail("slack_error",err)
   return[]
 def _smj_slack(self,n_products,budget):
  counts=self.needs.counts()
  each=[counts[k]if k<len(counts)else 1 for k in range(n_products)]
  floors=[]
  for k in range(n_products):
   cheapest=self._smj_cheapest_fit(self.ed_pool(k))
   if cheapest is None:
    return[]
   floors.append(cheapest[0]*each[k])
  cap,total=float(budget),sum(floors)
  if total<=cap+1e-9:
   return[]
  walks=list(self.__dict__.get("page_book",{}).values())
  hits=set(self.__dict__.get("all_hits")or[])
  queues=[]
  for k in range(n_products):
   room=(cap-(total-floors[k]))/max(1,each[k])
   lead=self._smj_lead_scores(k,walks,hits)
   eligible=[e for e in self.ed_pool(k)if self._smj_slack_ok(e,room)]
   eligible.sort(key=lambda e:(-lead.get(e["id"],0),e["price"],e["id"]))
   queues.append([e["id"]for e in eligible])
  taken=[]
  depth=0
  while len(taken)<3 and any(len(q)>depth for q in queues):
   taken.extend(q[depth]for q in queues if len(q)>depth)
   depth+=1
  taken=taken[:3]
  if taken:
   self.slack_total=total
  return taken
 @staticmethod
 def _smj_slack_ok(entry,room):
  if any(entry[flag]for flag in("read","gone","pointer","ruled","clash")):
   return False
  return entry["price"]is not None and room>0 and entry["price"]<=room+1e-9
 def _smj_lead_scores(self,k,walks,hits):
  word=re.compile(r"[a-z0-9]{2,}")
  wanted=[set(word.findall(trio[1].lower()))-LINK_WORDS for trio in self.ed_reqs(k)]
  lead=dict.fromkeys(hits,1)
  for walk in walks:
   said=[w for text in self.free_text(walk["args"])for w in word.findall(text.lower())]
   cut=self.lead_words
   heard=set(said[:cut]if cut else said)
   score=len([1 for group in wanted if group&heard])
   for ident in walk.get("row_shelf")or{}:
    if score>0:
     lead[str(ident)]=max(score,lead.get(str(ident),0))
  return lead
 def verify_problems(self,listing):
  for entry in reversed(self.log):
   if not isinstance(entry,dict)or(entry.get("kind"),entry.get("item"))!=("verify",listing):
    continue
   outcome=entry.get("result")
   if not isinstance(outcome,dict)or outcome.get("fits")is not False:
    break
   return[text for text in map(str,outcome.get("problems")or[])if text.strip()]
  return None
 def ed_news(self,product):
  if self.refusal_n>0:
   return True
  return any(self._smj_for_product(req,product)for req in[*self.needs.uncovered,*self.needs.let_go])
 def near_hold(self):
  try:
   return self._smj_hold_order()
  except Exception as err:
   self._smj_fail("near_hold_error",err)
   return""
 def _smj_hold_order(self):
  basket=self.__dict__.get("near_cart")
  if not basket:
   return""
  if(basket.get("ver"),basket.get("refusals"))!=(self.needs.version,self.refusal_n)or not self.role("order"):
   return""
  ids=list(basket.get("lines")or[])
  if not ids:
   return""
  for ident in ids:
   if ident not in self.trolley.units or ident in self.dead_lines()or ident in self.refused_lines:
    return""
  w=self.wire
  if self.moves_left()<1 or w.done or w.broke or w.spare()<2:
   return""
  self.near_cart=None
  self._smj_log("near_order",lines=ids)
  muted=w.mute
  try:
   self.finish_now()
  finally:
   w.mute=muted
  return("Not run (no step used). The code sent the order on its own rather than declining: the wording of the "
"cart lines (%s) was accepted earlier, by a check of meaning, as naming the requested values, and nothing "
"in the needs or the cart has moved since."%", ".join(ids))
 def near_check(self):
  w=self.wire
  stops=(self.__dict__.get("near_done"),not self.role("cart"),not self.role("order"),self.moves_left()<4,
w.spare()<=CLOSING+30,self._smj_quiet(),w.done,w.spent_out(),w.no_model)
  if any(stops):
   return""
  try:
   return self.near_basket()
  except Exception as err:
   self._smj_fail("near_error",err)
   return""
 _smj_NEAR=dict(gather="_smj_nb_gather",ask="_smj_nb_ask",settle="_smj_nb_settle",commit="_smj_nb_commit")
 def near_basket(self):
  count=len(self.needs.items())
  budget=self.needs.spend_cap
  ctx=dict(prods=list(range(count))if count else[None],counts=self.needs.counts(),
cap=float(budget)if is_number(budget)else None,chosen={},pools={})
  if self._smj_run_flow(self._smj_NEAR,ctx,"gather")!="placed":
   return""
  winners={n:pick[0]for n,pick in ctx["chosen"].items()}
  matched=["%s: its %s field reads '%s', taken as '%s'"%(row[1],row[2],row[4],row[3][:60])
for row in ctx["flat"]if winners.get(row[0],"")==row[1]]
  return("Not run yet (no step used). Decline held. Wording matches confirmed by a meaning check: %s. These "
"listings carry every need the shopper revealed, have stock and keep the basket within the spending "
"limit; the code has put them in the cart and checked their stock again. Cart: %s"
%("; ".join(matched),self.trolley.summary())+self._smj_DONE_NOTE)
 def _smj_units(self,ctx,n):
  counts=ctx["counts"]
  return counts[n]if n<len(counts)else 1
 def _smj_near_lines(self,entry,reqs):
  row=entry["row"]
  found=[]
  for prop,value,revealed in reqs:
   if self.code_shows(row,prop,value,set(),True):
    continue
   wording=[str(cell).strip()for name,cell in row["fields"].items()if same_prop(name,prop)]
   wording=[text for text in wording if text]
   if revealed or not wording or not self.ed_word(value):
    return None
   if any(re.search(r"\d",text)for text in wording):
    return None
   found.append((prop,value,"; ".join(dict.fromkeys(wording))[:120]))
  return found
 def _smj_near_cand(self,entry,reqs):
  lines=self._smj_near_lines(entry,reqs)
  heads,probs=set(),[]
  if entry["ruled"]:
   heads={str(fault).partition(":")[0].strip()for fault in self.field_view(entry["id"])["fails"]}
   probs=self.verify_problems(entry["id"])or[]
  if not lines:
   return None
  props=[line[0]for line in lines]
  if not all(any(same_prop(head,prop)for prop in props)for head in heads):
   return None
  terms=set()
  for prop,value,_ in lines:
   terms|=set(prop_words(prop))|set(re.findall(r"[a-z0-9]+",value.lower()))
  terms={term for term in terms if len(term)>2 and term not in LINK_WORDS}
  starts=[re.compile(r"\b"+re.escape(term))for term in terms]
  named=all(any(start.search(problem.lower())for start in starts)for problem in probs)
  return lines if named else None
 @staticmethod
 def _smj_cheapest_fit(pool):
  priced=[(e["price"],e["id"])for e in pool if e["fit"]and e["price"]is not None]
  return min(priced)if priced else None
 def _smj_near_pool(self,product,pool,keep):
  reqs=self.ed_reqs(product)
  ranked=[]
  for e in pool:
   if e["read"]and e["price"]is not None and not(e["gone"]or e["over"]or e["pointer"]):
    lines=self._smj_near_cand(e,reqs)
    if lines:
     ranked.append((len(lines),e["price"],e["id"],lines))
  return sorted(ranked)[:keep]
 def _smj_nb_gather(self,ctx):
  prods,chosen,pools=ctx["prods"],ctx["chosen"],ctx["pools"]
  keep=6 if len(prods)==1 else 3
  for n,product in enumerate(prods):
   pool=self.ed_pool(product)
   best=self._smj_cheapest_fit(pool)
   if best is not None:
    chosen[n]=(best[1],best[0],self._smj_units(ctx,n))
   elif not self.ed_news(product):
    return"drop"
   else:
    pools[n]=self._smj_near_pool(product,pool,keep)
    if not pools[n]:
     return"drop"
  if not pools:
   return"drop"
  least=sum(pick[1]*pick[2]for pick in chosen.values())
  least+=sum(min(c[1]for c in pools[n])*self._smj_units(ctx,n)for n in pools)
  if ctx["cap"]is not None and least>ctx["cap"]+1e-9:
   return"drop"
  flat=[]
  for n in sorted(pools):
   for cand in pools[n]:
    flat.extend((n,cand[2])+line for line in cand[3])
  ctx["flat"]=flat[:12]
  return"ask"
 def _smj_nb_ask(self,ctx):
  flat=ctx["flat"]
  listed="\n".join("%d. %s: %s = '%s' | listing field: %s"%(pos,row[1],row[2],row[3][:60],row[4])
for pos,row in enumerate(flat,1))
  prompt=("OUTPUT: exactly one JSON object and no other text: {\"lines\": [{\"n\": 1, \"same\": true}]} - one entry "
"per numbered line below; \"same\" is false whenever you are not sure.\n\n"
"TASK: an order is about to be declined because some listings word a requested value differently from "
"the request. Decide, line by line, whether the listing's field value means the requested value (a "
"synonym, a different spelling or phrasing, or a more precise form of the same thing) - same: true - or "
"names some other value of that property - same: false.\n\n"
"REQUEST FROM THE SHOPPER (for context):\n"+self.request[:1500]
+"\n\nLINES (number. listing id: property = 'requested value' | listing field: its own wording):\n"
+listed)
  self.near_done=True
  reply=self.wire.consult(prompt,600,CHECKER)
  said=reply.get("lines")if isinstance(reply,dict)else None
  agreed={int(item["n"])-1 for item in(said if isinstance(said,list)else[])
if isinstance(item,dict)and item.get("same")is True and is_number(item.get("n"))}
  self._smj_log("near_check",lines=[("%s:%s=%s|%s"%row[1:])[:120]for row in flat],same=sorted(agreed))
  ctx["agreed"]=agreed
  return"settle"
 def _smj_nb_settle(self,ctx):
  flat,agreed,chosen=ctx["flat"],ctx["agreed"],ctx["chosen"]
  for n in sorted(ctx["pools"]):
   for cand in ctx["pools"][n]:
    spots=[pos for pos,row in enumerate(flat)if(row[0],row[1])==(n,cand[2])]
    if len(spots)==len(cand[3])and agreed.issuperset(spots):
     chosen[n]=(cand[2],cand[1],self._smj_units(ctx,n))
     break
   else:
    return"drop"
  if ctx["cap"]is not None and sum(pick[1]*pick[2]for pick in chosen.values())>ctx["cap"]+1e-9:
   return"drop"
  picks=ctx["picks"]=[chosen[n][::2]for n in range(len(ctx["prods"]))]
  return"drop"if self.moves_left()<len(picks)+2 else"commit"
 def _smj_nb_commit(self,ctx):
  ver=str(self.needs.version)
  chosen,mem=ctx["chosen"],self.__dict__
  for n in ctx["pools"]:
   ident=chosen[n][0]
   key=f"{ident}|{ver}"
   self.rulings.update(dict.fromkeys([m for m in self.rulings if(m+"|").startswith(key+"|")],""))
   self.passed.add(key)
   mem.setdefault("near_ok",set()).add(key)
   mem.setdefault("pass_price",{})[ident]=self.archive.price(ident)
  picks=ctx["picks"]
  placed=self._smj_place(picks)
  self._smj_log("near_basket",picks=picks,done=placed,cart=dict(self.trolley.units))
  if not placed:
   return"drop"
  self.near_cart=dict(ver=self.needs.version,refusals=self.refusal_n,lines=[pick[0]for pick in picks])
  notes=mem.setdefault("near_said",{})
  for row in ctx["flat"]:
   if row[1]==chosen.get(row[0],("",))[0]:
    notes.setdefault(f"{row[1]}|{ver}",[]).append(row[2:])
  return"placed"
 def cart_moved(self,prior):
  mem=self.__dict__
  self.cart_ver,self.since_cart=mem.get("cart_ver",0)+1,0
  now=self.trolley.units
  if not set(now)<=set(prior):
   self.cart_seq=mem.get("absorb_n",0)
  reads=mem.setdefault("line_seq",{})
  for line in[line for line in now if line in prior and now[line]>prior[line]+1e-9]:
   reads.pop(line,None)
 def after_put(self,args,reply):
  cart,mem=self.trolley,self.__dict__
  if cart.view_units(reply)is not None:
   self.cart_shown=mem.get("cart_ver",0)
  known=mem.setdefault("first_puts",set())
  fresh=[line for line,count in cart.lines(args)
if(count is None or count>0)and line in cart.units and line not in known]
  if not fresh or not cart.is_put(args)or self.code_fault(reply):
   return
  known.update(fresh)
  unread=set(self.stale_lines())
  fresh=[line for line in fresh if line in unread]
  reader=self.stock_tool()
  if not(fresh and reader)or self._smj_quiet()or self.moves_left()<4 or self.wire.spare()<CLOSING:
   return
  queued=list(mem.get("put_queue")or[])
  self.put_queue,self.put_tool=(queued+[line for line in fresh if line not in queued])[:4],reader
  self._smj_log("after_put_read",lines=list(self.put_queue),tool=reader)
 def mark_fresh(self,reply):
  mem=self.__dict__
  if not mem.get("stock_seen"):
   return
  reads=mem.setdefault("line_seq",{})
  held=self.trolley.units
  def stamp(rec):
   ident=self.rec_id(rec)
   if ident in held and self.stock_state(rec)is not None:
    reads[ident]=mem.get("absorb_n",0)
  walk_dicts(reply,stamp)
 def cart_gone(self,held_ids):
  wanted=set(held_ids)
  gone=wanted&self.dead_lines()
  unread=[line for line in self.stale_lines()if line in wanted and line not in gone]
  if not unread or not self.stock_tool()or self.moves_left()-len(unread)<2:
   return gone
  _,sold=self.code_recheck(unread)
  return gone|set(sold)|(wanted&self.dead_lines())
 _smk_ROLE_NAMES=("search","message","cart","cart_view","order","decline")
 _smk_WRITE_WORDS=re.compile(r"\b(?:remov|delet|cancel|clear|discard|releas|refund|purchas|checkout|wish|basket|cart|bag)")
 _smk_WORD3=re.compile(r"[a-z0-9]{3,}")
 _smk_VALUE_TAG=re.compile(r"\((?:stated|revealed[^)]*)\)")
 _smk_POINTER=re.compile(r"as shown|same as|the page|sold by|redirect|refer")
 _smk_TITLE_KEY=re.compile(r"title|heading|headline")
 _smk_SCALAR=(str,int,float)
 _smk_PRE_STAGES={0:"first",1:"top_up"}
 def _smk_products(self):
  return self.needs.items()
 def _smk_fault(self,kind,err):
  self.log.append({"kind":kind,"error":f"{type(err).__name__}: {err}"[:300]})
 def _smk_walk(self,visit,newest_first=False):
  store=self.archive.items
  for obs in(reversed(store)if newest_first else store):
   walk_dicts(obs,visit)
 def batch_tool(self):
  if"batch_spec"in self.__dict__:
   return self.batch_spec
  try:
   self.batch_spec=self._smk_find_list_reader()
  except Exception:
   self.batch_spec=None
  return self.batch_spec
 def _smk_find_list_reader(self):
  claimed=set(map(self.role,self._smk_ROLE_NAMES))
  claimed.discard(None)
  blurb={}
  for spec in self.toolset or():
   body=spec.get("function")if isinstance(spec,dict)else None
   called=body.get("name")if isinstance(body,dict)else None
   if isinstance(called,str):
    spaced=re.sub(r"([a-z])([A-Z])",r"\1 \2",called).replace("_"," ")
    blurb[called]="%s %s"%(spaced,str(body.get("description")or""))
  claimed|={tool for tool in list(self.shapes)if self._smk_WRITE_WORDS.search(blurb.get(tool,tool).lower())}
  for tool,params in self.shapes.items():
   arg=self._smk_id_list_arg(params)
   if arg is not None and tool not in claimed:
    return(tool,)+arg
  return None
 @staticmethod
 def _smk_id_list_arg(params):
  if not isinstance(params,dict):
   return None
  props=params.get("properties")
  props=props if isinstance(props,dict)else{}
  def text_list(entry):
   inner=entry.get("items")if isinstance(entry,dict)and entry.get("type")=="array"else None
   return isinstance(inner,dict)and inner.get("type")=="string"and not inner.get("enum")
  names=[name for name,entry in props.items()if text_list(entry)]
  if len(names)!=1:
   return None
  demanded=params["required"]if isinstance(params.get("required"),list)else[]
  if[extra for extra in demanded if extra not in names]:
   return None
  limit=props[names[0]].get("maxItems")
  if not(is_number(limit)and limit>=2):
   return None
  return names[0],int(min(limit,LIST_READ_CAP))
 def _smk_batch_shrink(self,asked):
  spec=self.__dict__.get("batch_spec")
  if not spec or asked<=LIST_FAIL_CAP:
   return False
  self.batch_spec=(spec[0],spec[1],max(LIST_FAIL_CAP,min(int(spec[2]),LIST_FAIL_CAP)))
  self.list_cap_note=("\n[list limit] %s returned an error for %d ids; it is now sent %d ids at a time, and longer lists "
"are cut to that."%(spec[0],int(asked),self.batch_spec[2]))
  self.log.append({"kind":"batch_shrunk","tool":spec[0],"cap":self.batch_spec[2]})
  return True
 @staticmethod
 def batch_ok(found,ids):
  asked,matched=set(ids),set()
  def record_like(node):
   maps=[part for part in node.values()if isinstance(part,dict)]
   if maps:
    return any(len([b for b in m.values()if isinstance(b,(str,int,float))])>=2 for m in maps)
   return len([t for t in node.values()if isinstance(t,str)and 0<len(t)<=70])>=5
  def visit(node):
   if record_like(node):
    matched.update(t for t in node.values()if isinstance(t,str)and t in asked)
  try:
   for obs in found:
    walk_dicts(obs,visit)
  except Exception:
   return False
  return 0<len(matched)>=len(asked)/2
 @staticmethod
 def ids_back(found,ids):
  return any(i in json.dumps(found,default=str)for i in ids)if ids else False
 def batch_richer(self,found,ids):
  asked,keys,maps=set(ids),set(),False
  def visit(node):
   nonlocal maps
   if any(isinstance(t,str)and t in asked for t in node.values()):
    keys.update(map(str,node))
    maps=maps or any(isinstance(part,dict)and len(part)>=2 for part in node.values())
  try:
   for obs in found:
    walk_dicts(obs,visit)
  except Exception:
   return True
  seen=set(self.__dict__.get("row_keys")or set())
  return maps or not seen or bool(keys-seen)
 @staticmethod
 def title_of(row):
  named=[row.get("title")]+[val for name,val in row.items()if Errand._smk_TITLE_KEY.search(str(name).lower())]
  return next(filter(lambda val:isinstance(val,str),named),"")
 def _smk_title_words(self,node):
  return set(self._smk_WORD3.findall(self.title_of(node).lower()))
 def _smk_ranked_hits(self):
  weight=getattr(self,"hit_score",{})
  ranked=list(getattr(self,"all_hits",[]))
  ranked.sort(key=lambda rid:-weight.get(rid,0))
  return ranked
 def _smk_too_dear(self,rid,units,cap):
  if not is_number(cap):
   return False
  cost=self.archive.price(rid)
  return is_number(cost)and cost*units>cap
 @staticmethod
 def _smk_deal(queues,limit):
  out=[]
  depth=0
  while len(out)<limit:
   layer=[q[depth]for q in queues if depth<len(q)]
   if not layer:
    break
   out.extend(layer)
   depth+=1
  return out[:limit]
 def menu_rows(self,cats,most,skip,units=1,led_only=False):
  shop,cap=self.archive,self.needs.spend_cap
  fresh={}
  gone=set()
  seen_words=[]
  def visit(node):
   rid=self.row_mark(node)
   if rid not in shop.ids or self.shelf_of(node)not in cats:
    return
   if rid in shop.read:
    seen_words.append(self._smk_title_words(node))
    return
   if self.sold_out(node):
    gone.add(rid)
   fresh.setdefault(rid,self.title_of(node))
  self._smk_walk(visit)
  hits=self._smk_ranked_hits()
  ordered=[rid for rid in hits if rid in fresh]+[rid for rid in fresh if rid not in hits]
  pool=[rid for rid in ordered if not(rid in gone or rid in skip or rid in self.refused_lines)
and not self._smk_too_dear(rid,units,cap)]
  bag={rid:set(self._smk_WORD3.findall(fresh[rid].lower()))for rid in pool}
  led=getattr(self,"row_score",{})
  chosen=[]
  opener=self._smk_lead_row(cats,most,bag)
  if opener:
   chosen.append(opener)
   seen_words.append(bag[opener])
  elif led_only:
   return[]
  def rank(rid):
   tier=0 if rid in hits else(1 if led.get(rid)else 2)
   mine=bag[rid]
   alike=max((len(mine&other)/max(1,len(mine|other))for other in seen_words),default=0.0)
   return tier,alike,pool.index(rid)
  while len(chosen)<most:
   window=[rid for rid in pool if rid not in chosen][:16]
   if not window:
    break
   best=min(window,key=rank)
   chosen.append(best)
   seen_words.append(bag[best])
  return chosen
 def _smk_lead_row(self,cats,most,bag):
  try:
   several=bool(self._smk_products())
   pi=None
   if several:
    pi=next((n for n,group in enumerate(self.home_shelves())if group==cats),None)
    if pi is None:
     return None
   if most<=0:
    return None
   wanted=self._smk_lead_values(several,pi)
   if not wanted or self._smk_fit_read(cats,wanted):
    return None
   carried=[]
   for walk in getattr(self,"page_book",{}).values():
    hold=self.walk_carries(walk["args"],pi)
    if hold:
     carried.append((len(hold),walk))
   carried.sort(key=lambda pair:-pair[0])
   for _,walk in carried:
    first=next(filter(bag.__contains__,walk.get("row_shelf")or{}),None)
    if first:
     return first
   return None
  except Exception:
   return None
 def _smk_lead_values(self,several,pi):
  out=[]
  for req in self.needs.props_left():
   label,_,said=req.partition(":")
   if several and self.product_of(label)not in(pi,None):
    continue
   said=self._smk_VALUE_TAG.sub("",said)
   if said.strip()and positive_value(said)and not self._smk_POINTER.search(said.lower()):
    out.append(said)
  return out
 def _smk_fit_read(self,cats,wanted):
  facts={}
  scalar=self._smk_SCALAR
  def visit(node):
   rid=self.row_mark(node)
   if rid not in self.archive.read or self.shelf_of(node)not in cats:
    return
   maps=[part for part in node.values()if isinstance(part,dict)]
   if maps:
    facts.setdefault(rid,[]).extend(str(b)for m in maps for b in m.values()if isinstance(b,scalar))
  self._smk_walk(visit)
  blobs=map(" | ".join,facts.values())
  return any(all(value_seen(one,blob)for one in wanted)for blob in blobs)
 def _smk_queue_reads(self,ids,purpose,say):
  self.open_queue,self.open_recheck,self.open_say,self.open_purpose=ids,True,say,purpose
 def _smk_rich_count(self,cats):
  if not cats:
   return 4
  return sum(map(self.archive.rich_reads,cats))
 def pre_read(self):
  try:
   reader=self.batch_tool()
   stage=self._smk_PRE_STAGES.get(getattr(self,"pre_rounds",0),"spent")
   floor=25 if stage=="top_up"else 20
   if not reader or stage=="spent"or len(self.questions_log)>1 or self.moves_left()<=floor or self.wire.spare()<CLOSING+60:
    return False
   groups=self.home_shelves()
   return getattr(self,"_smk_pre_"+stage)(reader[2],groups,self.needs.counts())
  except Exception as err:
   self._smk_fault("pre_read_error",err)
   return False
 def fit_reads(self):
  try:
   kv=self.__dict__
   if kv.get("fit_read")or self.questions_log or self.msgs_sent or self.moves_left()<=20 or self.wire.spare()<CLOSING+60:
    return False
   several,counts,chosen=bool(self._smk_products()),self.needs.counts(),[]
   for pi,cats in enumerate(self.home_shelves()):
    wanted=self._smk_lead_values(several,pi)
    if cats and wanted and not self._smk_fit_read(cats,wanted):
     chosen+=self.menu_rows(cats,2,chosen,counts[min(pi,len(counts)-1)],led_only=True)
   if not chosen:
    return False
   kv["fit_read"]=True
   self._smk_queue_reads(chosen,"menu","The listings read so far show none with the stated values, so the rows a "
"search for those values found come next; their keys shape the questions.")
   self.log.append({"kind":"fit_reads","ids":chosen})
   return True
  except Exception as err:
   self._smk_fault("fit_reads_error",err)
   return False
 def _smk_pre_first(self,per_call,groups,counts):
  units_of=lambda pi:counts[min(pi,len(counts)-1)]
  bare,thin=[],[]
  for pi,cats in enumerate(groups):
   have=self._smk_rich_count(cats)
   if not have:
    bare+=self.menu_rows(cats,2 if self._smk_products()else 4,bare,units_of(pi))
   elif have<4:
    thin.append((pi,4-have))
  chosen=list(bare)
  for pi,gap in thin:
   chosen+=self.menu_rows(groups[pi],gap,chosen,units_of(pi))
  rounds=2 if len(bare)>per_call and self.moves_left()>25 else 1
  del chosen[per_call*rounds:]
  if not chosen:
   return False
  self.pre_rounds,self.pre_picked=1,list(chosen)
  self._smk_queue_reads(chosen,"menu","Reading a small batch of listings per requested item comes first; their "
"keys shape the questions.")
  return True
 def _smk_pre_top_up(self,per_call,groups,counts):
  self.pre_rounds=2
  queues=[]
  held=list(getattr(self,"pre_picked",None)or[])
  for pi,cats in enumerate(groups):
   have=self._smk_rich_count(cats)
   if have>=3:
    continue
   more=self.menu_rows(cats,4-have,held,counts[min(pi,len(counts)-1)])
   held=more+held
   queues.append(more)
  chosen=self._smk_deal(queues,per_call)
  if not chosen:
   return False
  self._smk_queue_reads(chosen,"menu","A few more listings get read first: the earlier reads of some products "
"carried only a handful of fields.")
  return True
 def lacking_rows(self):
  try:
   products=self._smk_products()
   if not products:
    return[]
   groups=self.home_shelves()
   short=self._smk_unfit_products(groups)
   if not short:
    return[]
   reader=self.batch_tool()
   pools=self._smk_lacking_pools(products,groups,short)
   return self._smk_deal([pools[pi]for pi in short],reader[2]if reader else 3)
  except Exception as err:
   self._smk_fault("lacking_error",err)
   return[]
 def _smk_unfit_products(self,groups):
  live=set()
  for key in self.passed:
   line,_,ver=key.rpartition("|")
   if ver==str(self.needs.version):
    live.add(line)
  live=live-self.refused_lines-self.dead_lines()
  found={self.shelf_known(rid)for rid in live|set(getattr(self,"audit_fits",None)or[])}
  found.discard("")
  return[pi for pi,cats in enumerate(groups)if cats and cats.isdisjoint(found)]
 def _smk_lacking_pools(self,products,groups,short):
  shop,cap=self.archive,self.needs.spend_cap
  trail,gone=[],set()
  shelf_at={}
  def visit(node):
   rid,label=self.row_mark(node),self.shelf_of(node)
   if rid not in shop.ids or not label:
    return
   trail.append(rid)
   shelf_at.setdefault(rid,label)
   if self.sold_out(node):
    gone.add(rid)
  self._smk_walk(visit,newest_first=True)
  owner={}
  for pi in short:
   for label in groups[pi]:
    owner.setdefault(label,pi)
  pools={pi:[]for pi in short}
  hits=[rid for rid in self._smk_ranked_hits()if rid in shelf_at]
  for rid in dict.fromkeys(hits+trail):
   if rid in shop.read or rid in self.refused_lines or rid in gone:
    continue
   pi=owner.get(shelf_at[rid])
   if pi is not None and not self._smk_too_dear(rid,products[pi]["quantity"],cap):
    pools[pi].append(rid)
  return pools
 def text_arg(self,tool):
  params=self.shapes.get(tool or"")or{}
  props=params.get("properties")or{}
  demanded=list(params.get("required")or[])
  def plain_text(entry):
   if not isinstance(entry,dict):
    return False
   kind=entry.get("type")
   return(kind=="string"or"string"in(kind or[]))and"enum"not in entry
  texts=[name for name,entry in props.items()if plain_text(entry)]
  sole=texts[0]if len(texts)==1 else None
  chosen=next(filter(texts.__contains__,demanded),None)or sole
  return chosen if chosen and all(name==chosen for name in demanded)else None
 def free_text_arg(self,tool):
  params=self.shapes.get(tool or"")or{}
  props=params.get("properties")or{}
  demanded=list(params.get("required")or[])
  names=[name for name,entry in props.items()if isinstance(entry,dict)and"enum"not in entry
and(entry.get("type")=="string"or"string"in(entry.get("type")or[]))]
  return next((name for name in demanded if name in names),names[0]if names else None)
 def message_call(self,body,call_id):
  tool=self.role("message")
  field=tool and self.text_arg(tool)
  if not field:
   return None
  fixed,problems=ArgCheck.execute({field:body},self.shapes[tool])
  if problems:
   return None
  call={"name":tool,"arguments":json.dumps(fixed)}
  return{"id":call_id,"type":"function","function":call}
 def product_of(self,text):
  products=self._smk_products()
  if not products:
   return None
  kinds=[set(kind_tokens(row["product"]))for row in products]
  span=range(len(kinds))
  rivals=[set().union(*[kinds[o]for o in span if o!=pi])for pi in span]
  whole=str(text)
  for piece in re.findall(r"\(([^)]*)\)",whole)[:1]+[whole]:
   said=set(kind_tokens(piece))
   tally=[kind_score(kinds[pi],said,rivals[pi])for pi in span]
   best=max(tally)
   leaders=[pi for pi in span if tally[pi]==best]
   if len(leaders)>1:
    leaders=[pi for pi in leaders if all(kind_in(word,said)for word in kinds[pi])]
   if best>0 and len(leaders)==1:
    return leaders[0]
  return None
 def plan_product(self,entry):
  tagged=self.product_of(str(entry.get("property")or""))
  return tagged if tagged is not None else self.product_of(str(entry.get("question")or""))
 @staticmethod
 def _smk_which(field,product):
  return f"Which {field} would you like for the {product}?"
 def _smk_ask_tally(self,count):
  answered,messaged=[0]*count,[0]*count
  for entry in self.questions_log:
   head=entry.partition(":")[0]
   owner=self.product_of(head)
   if owner is None:
    quoted=re.search(r'asked "(.*?)" -> ',entry)
    owner=None if quoted is None else self.product_of(quoted.group(1))
   if owner is None:
    continue
   messaged[owner]+=1
   answered[owner]+=0 if"[not answered]"in head else 1
  return answered,messaged
 def _smk_hinted_products(self,count,answered):
  try:
   by_key,keyless=set(),[0]*count
   for hint in self.open_hints():
    head=str(hint).partition(":")[0]
    owner=self.product_of(head)
    if owner is None:
     continue
    if self.plain_hint(owner,head.strip(),hinted=True):
     keyless[owner]+=1
    else:
     by_key.add(owner)
   return{o for o in range(count)if o in by_key or keyless[o]>answered[o]}
  except Exception:
   return set()
 def quota_first(self,plan,ok=None):
  products=self._smk_products()
  if not products:
   return plan
  answered,messaged=self._smk_ask_tally(len(products))
  if not any(answered)and plan and self.plan_product(plan[0])is None:
   return plan
  keys,hinted=self._smk_turn_keys(len(products),answered,messaged)
  if self._smk_lead_jumps(plan,keys,messaged):
   return plan
  ctx={"plan":plan,"products":products,"answered":answered,"hinted":hinted,"ok":ok}
  for level in sorted(set(keys)):
   ctx["due"]=[pi for pi,mark in enumerate(keys)if mark==level]
   for stage in(self._smk_due_planned,self._smk_due_second,self._smk_due_earlier,self._smk_due_pick):
    out=stage(ctx)
    if out is not None:
     return out
  return plan
 def _smk_turn_keys(self,count,answered,messaged):
  revealed=[0]*count
  for need in self.needs.uncovered:
   owner=self.product_of(str(need).partition(":")[0])
   if owner is not None:
    revealed[owner]+=1
  hinted=self._smk_hinted_products(count,answered)
  silent=[o for o in range(count)if not answered[o]]
  alt_open={o for o in range(count)if messaged[o]>answered[o]and self.alt_untried(o)}
  keys=[]
  for pi in range(count):
   waits=1 if revealed[pi]and any(o!=pi for o in silent)else 0
   unheard=0 if pi in alt_open else messaged[pi]-answered[pi]
   keys.append((waits,answered[pi]+revealed[pi],int(pi not in hinted),unheard,revealed[pi]))
  return keys,hinted
 def _smk_lead_jumps(self,plan,keys,messaged):
  try:
   head=self.plan_product(plan[0])if plan else None
   if head is None:
    return False
   low=min(keys)
   due={pi for pi,mark in enumerate(keys)if mark==low}
   if head in due or keys[head][:3]!=low[:3]or messaged[head]>min(messaged)+1:
    return False
   if not any(self.plan_product(entry)in due for entry in plan[1:]):
    return False
   silent=sum(1 for pi,sent in enumerate(messaged)if not sent and pi!=head)
   if self.ask_room()-self.msgs_sent-1<silent:
    return False
   self.log.append({"kind":"lead_jump","property":str(plan[0].get("property"))})
   return True
  except Exception:
   return False
 def _smk_due_earlier(self,ctx):
  ok=ctx["ok"]
  for entry in self.__dict__.get("plan_spares")or[]:
   owner=self.plan_product(entry)
   label=entry.get("property")
   if owner not in ctx["due"]or(ok is not None and not ok(label))or self.asked_for(owner,label):
    continue
   return[entry]+ctx["plan"]
  return None
 def _smk_due_planned(self,ctx):
  plan=ctx["plan"]
  spot=next((n for n,entry in enumerate(plan)if self.plan_product(entry)in ctx["due"]),None)
  if spot is None:
   return None
  rest=list(plan)
  return[rest.pop(spot)]+rest
 def _smk_due_second(self,ctx):
  for entry in ctx["plan"]:
   spare=str(entry.get("second")or"").strip()
   owner=self.product_of(spare)if spare else None
   if owner not in ctx["due"]:
    continue
   tag=re.search(r"\(([^)]+)\)",spare)
   product=ctx["products"][owner]["product"]if tag is None else tag.group(1).strip()
   ask={"property":spare,"question":self._smk_which(spare.partition("(")[0].strip(),product)}
   return[ask]+ctx["plan"]
  return None
 def _smk_due_pick(self,ctx):
  hinted,ok=ctx["hinted"],ctx["ok"]
  for pi in sorted(ctx["due"],key=lambda o:(o not in hinted,o)):
   try:
    found=self.quota_pick(pi,ctx["answered"][pi]==0,ok)
   except Exception:
    found=None
   if found and ok is not None:
    label=found.get("property")
    if not ok(label)or self.asked_for(pi,label):
     found=None
   if found:
    return[found]+ctx["plan"]
   if getattr(self,"quota_wait",None):
    return ctx["plan"]
  return None
 def quota_pick(self,pi,fallback=True,ok=None):
  product=self._smk_products()[pi]["product"]
  groups=self.home_shelves()
  cats=groups[pi]if pi<len(groups)else set()
  shelf=self._smk_quota_shelf(pi,cats)
  if shelf is None:
   self._smk_quota_open(pi,cats,product)
   return None
  listing,_=self.archive.prop_menu(shelf)
  menu=self.menu_trim(listing,self.archive.rich_reads(shelf))
  def fresh(label):
   if ok is None:
    return True
   return bool(ok(label))and not self.asked_for(pi,label)
  planned=self.product_plan(pi,menu)
  if planned and fresh(planned.get("property")):
   return planned
  for field in self._smk_quota_props(pi,menu,fallback):
   if fresh(f"{field} ({product})"):
    return{"property":f"{field} ({product})","question":self._smk_which(field,product)}
  return None
 def _smk_quota_shelf(self,pi,cats):
  shop=self.archive
  known=next((label for label in shop.shelves_read()if label.strip().lower()in cats),None)
  if known is not None:
   return known
  if not cats or not getattr(self,"quota_reads",{}).get(pi):
   return None
  return next((label for label in sorted(cats)if shop.prop_menu(label)[0]),None)
 def _smk_quota_open(self,pi,cats,product):
  reader=self.batch_tool()
  if not cats:
   return
  size=min(4,reader[2])if reader else 2
  unread=self.menu_rows(cats,size,[],self._smk_products()[pi]["quantity"])
  if not unread or not(self.role("details")or reader):
   return
  if getattr(self,"quota_reads",{}).get(pi)or self.moves_left()<=12:
   return
  self.__dict__.setdefault("quota_reads",{})[pi]=True
  self._smk_queue_reads(unread,"quota",f"Nothing about the {product} has been read so far; opening some of its "
"listings now lets its question use a real key.")
  self.quota_wait=True
 def _smk_quota_props(self,pi,menu,fallback):
  try:
   hints=self.open_hints(pi)
  except Exception:
   hints=[]
  for hint in hints:
   field=hint.partition(":")[0].partition("(")[0].strip()
   if field and not junk_key(field)and not self.plain_hint(pi,field,hinted=True):
    yield field
  if not fallback:
   return
  rows=menu.splitlines()
  for strict in(False,True):
   for row in rows:
    field=row.strip("- ").partition(" (in ")[0].strip()
    if field and self.open_prop(pi,field,row,by_samples=strict):
     yield field
 def product_plan(self,pi,menu):
  try:
   used=getattr(self,"product_plans",0)
   if not menu or used>=1 or self.wire.spare()<CLOSING+40:
    return None
   self.product_plans=used+1
   product=self._smk_products()[pi]["product"]
   reply=self.wire.consult(self._smk_plan_prompt(pi,product,menu),1500,PLANNER[:1])
   self.log.append({"kind":"product_plan","product":product,"result":reply})
   entries=(reply or{}).get("questions")or[]
   picks=(self._smk_plan_pick(pi,product,e)for e in entries if isinstance(e,dict))
   return next(filter(None,picks),None)
  except Exception:
   return None
 def _smk_plan_prompt(self,pi,product,menu):
  loud=product.upper()
  room=max(1,self.ask_room()-self.msgs_sent)
  parts=[ASK_PLAN.replace("QUESTION_COUNT",str(min(3,room))),
f"FOCUS: {loud}. Plan for this item and no other: of all the requested items it alone is still without "
f"any question. Take its properties solely from the {product} key list placed last, and put first "
"whichever key the story's concrete details are about.",
"SHOPPER REQUEST:\n"+self.request,
"REQUIREMENTS SO FAR:\n"+self.needs.text_body()]
  if self.questions_log:
   parts.append("QUESTIONS SENT SO FAR, EACH WITH THE SHOPPER'S REPLY:\n"+"\n".join(self.questions_log))
  try:
   hints=self.open_hints(pi)
  except Exception:
   hints=[]
  if hints:
   parts.append("STORY HINTS\nGuesses rather than requirements: each entry names a property, then the story "
"detail that suggests it; the likeliest entry comes first.\n"+"\n".join(f"- {self.hint_mark(h)}"for h in hints[:6]))
  text="\n\n".join(parts)+self.plan_notes(pi)
  return text+(f"\n\nAVAILABLE KEYS - {loud}\nEvery row gives a key name, then in how many of this item's "
"listings it occurs so far, then sample values:\n"+menu)
 def _smk_plan_pick(self,pi,product,entry):
  lead,spare=(str(entry.get(key)or"").partition("(")[0].strip()for key in("property","second"))
  if re.search(r"none|^-$|n/a",spare.lower()):
   spare=""
  wording=str(entry.get("question")or"").strip()
  lead=""if self.plain_hint(pi,lead)else lead
  if spare and self.plain_hint(pi,spare):
   spare=""
  if lead and self.plan_open(pi,lead):
   on_topic=bool(wording)and bool(kind_words(product)&kind_words(wording))
   paired=on_topic and spare and not strict_prop(lead,spare)and self.plan_open(pi,spare)
   return{"property":f"{lead} ({product})","second":f"{spare} ({product})"if paired else"",
"question":wording if on_topic else self._smk_which(lead,product)}
  if not spare or not self.plan_open(pi,spare):
   return None
  return{"property":f"{spare} ({product})","question":self._smk_which(spare,product)}
 def open_prop(self,who,field,menu_line="",by_samples=False):
  try:
   name=str(field).partition("(")[0].strip()
   if not name or junk_key(name):
    return False
   several=bool(self._smk_products())
   def ours(text,tagged_only=False):
    if who is None or not several:
     return True
    owner=self.product_of(text)
    return not tagged_only if owner is None else owner==who
   if self._smk_was_asked(who,name,ours):
    return False
   samples=menu_line.partition("):")[2].lower()
   for req in map(str,self.needs.props_left()):
    if ours(req.partition(":")[0])and self._smk_settled(who,name,req,samples,by_samples):
     return False
   calm=map(str,self.needs.indifferent)
   return not any(ours(c,tagged_only=True)and self.same_field(who,c,name)for c in calm)
  except Exception:
   return True
 def _smk_was_asked(self,who,name,ours):
  heads=(entry.partition(":")[0].strip("- ")for entry in self.questions_log)
  parts=(bit for head in heads if"[not answered]"not in head for bit in head.split(";"))
  return any(ours(bit)and self.same_field(who,bit,name)for bit in parts)
 def _smk_settled(self,who,name,req,samples,by_samples):
  head,_,said=req.partition(":")
  if strict_prop(head,name):
   return bool(self.settles(req,samples if by_samples else""))
  if same_prop(head,name):
   pool=samples or" | ".join(self.field_values(who,name))
   return bool(positive_value(said,head)and value_seen(said,pool))
  terms=[t for t in re.findall(r"[a-z]{3,}",said.lower())if t not in LINK_WORDS]
  if not samples or not terms or re.search(r"\d",said)or not positive_value(said,head):
   return False
  return all(re.search(r"\b"+re.escape(t),samples)for t in terms)
 def plan_notes(self,who):
  try:
   several=bool(self._smk_products())
   heads=(entry.partition(":")[0].strip("- ")for entry in self.questions_log)
   asked={bit.strip().lower()for head in heads for bit in head.split(";")}
   def concerns(text):
    return who is None or not several or self.product_of(text)in(who,None)
   still=[]
   for pending_prop in getattr(self,"open_pair",None)or[]:
    owner=self.product_of(pending_prop)if several else None
    if who is not None and several and owner not in(who,None):
     continue
    if pending_prop.lower()not in asked and self.plan_open(owner,pending_prop):
     still.append(pending_prop)
   hoped=list(filter(concerns,getattr(self,"wanted_asks",None)or[]))
   sections=(
("\n\nSTILL OPEN AFTER A PAIRED QUESTION (each one was the plan's next pick: the second half of a planned "
"pair that the reply left unanswered, because only the first half was sent or only it got an answer):\n",
still),
("\n\nUNSENT DRAFTS (the acting assistant composed these after opening listing records). Any draft whose "
"property is still open and appears among the keys below takes the first slot, worded with that key's "
"own spelling:\n",hoped))
   return"".join(title+"\n".join(f"- {x}"for x in items[-3:])for title,items in sections if items)
  except Exception:
   return""
 _sml_ALNUM=re.compile(r"[a-z0-9]+")
 _sml_ALPHA=re.compile(r"[a-z]+")
 _sml_TAG=re.compile(r"\(([^)]+)\)")
 _sml_LONG4=re.compile(r"[a-z]{4,}")
 _sml_LONG3=re.compile(r"[a-z]{3,}")
 _sml_NOT_SECOND=re.compile(r"none|^-$|n/a")
 _sml_ID_CUT=re.compile(r"[\s.,;:!?()\[\]{}<>\"'`]+")
 _sml_STOCK=re.compile(r"\bin[- ]stock\b|\bon hand\b|\bon the shelf\b|\bsold[- ]out\b|\bback[- ]?order|"
r"\bout of stock\b|\bavailable now\b")
 _sml_CHEAP=re.compile(r"\bcheapest\b|\blowest[- ](?:price|priced|cost)\b|\bleast expensive\b|"
r"\blowest\b[^.]{0,20}\bprice\b|\bsmallest (?:total|price)\b")
 _sml_LIMITS=(("message_budget",r"messages?|questions?|notes?|repl(?:y|ies)|texts?",0),
("step_budget",r"steps?|moves?|actions?|(?:tool )?calls?|turns?",FIRST_ORDER_BY+3))
 _sml_SENTENCE=re.compile(r"(?<=[.!?;])\s+")
 _sml_FIGURE=re.compile(r"([$\u20ac\u00a3]\s*)?(\d[\d,]*(?:\.\d+)?)")
 _sml_WHOLE=re.compile(r"\b(?:order|total|whole|lot|sum|bill|altogether|overall|everything)\b")
 _sml_COUNTED=re.compile(r"\s*(?:x\b|\u00d7|%|:\s*\d|-\s*\d|/\s*\d|(?:[a-z-]+\s+){0,2}?(?:messages?|questions?|notes?|"
r"repl\w*|texts?|steps?|moves?|actions?|(?:tool )?calls?|turns?|units?|items?|pieces?|packs?|"
r"pairs?|lots?|sets?|of\b|times?)\b)")
 _sml_IN_RUN=re.compile(r"[\d.,:/-]\s*$|\bx\s*$")
 _sml_SPEND=re.compile(r"\b(?:spend\w*|budget\w*|afford\w*|pay\w*|price[ds]?|costs?|cash|money)\b(?:\W+[\w']+){0,4}\W*$")
 _sml_CUE_BEFORE=re.compile(r"\b(?:under|below|no more than|at most|not more than|up to|within|max\w*|less than|"
r"cheaper than|not (?:more|over|above)|limit\w*|reach\w*|go to|to|at|cap\w*|past|exceed\w*|"
r"beyond|go over)\W+(?:[\w']+\W+)?$")
 _sml_CUE_AFTER=re.compile(r"\s*(?:[a-z.()]+\s*){0,3}?(?:or (?:lower|less|under|below|cheaper)|at (?:the (?:very )?)?most|"
r"max\w*|tops\b|,?\s*(?:and )?not (?:one|a) \w+ (?:over|more|above))")
 _sml_UNIT=re.compile(r"\s*((?:[a-z][a-z.]*\s*){0,2})")
 _sml_CAP_BEFORE=re.compile(r"\b(?:no more than|at most|not more than|up to|only|max\w*|use|using|take|send|allow\w*|"
r"receive|get|total of|limit\w*|inside|within|under|fewer than|less than|you've|you have|have|"
r"keep\w*|stick to|go past|past|exceed\w*|beyond)\W+(?:[\w']+\W+){0,3}$")
 _sml_CAP_AFTER=re.compile(r"\W*(?:[\w']+\W+){0,3}?(?:in total|total|no more|at most|max\w*|tops|or fewer|or less|allowed|"
r"limit)\b")
 _sml_WORD_NUMS=(("one",1),("two",2),("three",3),("four",4),("five",5),("six",6),("seven",7),("eight",8),
("nine",9),("ten",10),("twelve",12),("fifteen",15),("twenty",20))
 _sml_HINTS_HEAD=("\n\nSTORY HINTS. Not requirements. Per line: a detail from the tale => what it likely concerns. "
"Usually one of these properties holds what the shopper keeps unsaid.\n")
 _sml_MENU_HEAD="\n\nKEYS IN THE LISTINGS' DETAILS SEEN SO FAR (name, number of listings, example values):\n"
 _sml_ASKERS=frozenset(("which","would","you","like"))
 def _sml_kept(self,name,default=None):
  return self.__dict__.get(name,default)
 def _sml_rows(self):
  return self.needs.items()
 def _sml_said(self,slot):
  return self.needs.reading.get(slot)
 def _sml_k(self,name):
  return self.product_of(name)if self._sml_rows()else None
 @staticmethod
 def _sml_field(item,slot):
  return str(item.get(slot)or"")
 @staticmethod
 def _sml_flat(text):
  return" ".join(Errand._sml_ALNUM.findall(str(text).lower()))
 @staticmethod
 def _sml_note(name,what):
  return f" [{name}: {what}]"
 def want_ask(self,given):
  try:
   if isinstance(given,dict):
    fields=given
   elif isinstance(given,str)and given.strip():
    fields=json_object(given)
   else:
    fields={}
   stripped=(v.strip()for v in(fields or{}).values()if isinstance(v,str))
   said=" ".join(filter(None,stripped))[:300]
   if said:
    folded=said.lower()
    self.wanted_asks=[*(wd for wd in self._sml_kept("wanted_asks")or[]if wd.lower()!=folded),said][-3:]
  except Exception:
   pass
 def drop_wanted(self,label):
  try:
   stem=self.stem
   def covers(text,name):
    need=prop_words(name)
    bag={stem(wd)for wd in self._sml_ALPHA.findall(text.lower())}
    return bool(need)and all(wd in bag or stem(wd)in bag for wd in need)
   cleaned=str(label).replace("[not answered]","")
   names=[re.split(r"\s*\(",part)[0]for part in re.split(r";|\band\b",cleaned)]
   pool=self._sml_kept("wanted_asks")or[]
   self.wanted_asks=[text for text in pool if not any(covers(text,name)for name in names)]
  except Exception:
   pass
 def names_other(self,question,subject,sheet,other=""):
  try:
   flat=self._sml_flat(question)
   bag=set(flat.split())
   heard=bag|{wd[:-1]for wd in bag if len(wd)>3 and wd[-1]=="s"and wd[-2]!="s"}
   mine=self._sml_own_words(subject)
   if other and not strict_prop(other,subject)and self._sml_spells_out(other,heard,mine):
    return True
   return any(self._sml_spells_out(slot,heard,mine)or self._sml_quotes_sample(line,flat,mine)
for slot,line in self._sml_other_keys(sheet,subject))
  except Exception:
   return False
 @staticmethod
 def _sml_other_keys(sheet,subject):
  for line in str(sheet or"").splitlines():
   head=line.strip().lstrip("-").strip()
   cut=head.find(" (in ")
   slot=(head if cut<0 else head[:cut]).strip()
   if slot and not slot.endswith(":")and not strict_prop(slot,subject)and not junk_key(slot):
    yield slot,line
 def _sml_own_words(self,subject):
  names=[self._sml_field(rec,"product")for rec in self._sml_rows()]or[str(self._sml_said("product")or"")]
  mine=set(prop_words(subject))|self._sml_ASKERS
  for wd in self._sml_ALNUM.findall(" ".join(names).lower()):
   mine.update((wd,wd[:-1])if len(wd)>3 and wd[-1]=="s"else(wd,))
  return mine
 @staticmethod
 def _sml_spells_out(name,heard,mine):
  core={wd for wd in prop_words(name)if wd not in LINK_WORDS and len(wd)>2}
  return len(core)>0 and core<=heard and not core<=mine
 @staticmethod
 def _sml_quotes_sample(line,flat,mine):
  _,sep,tail=line.partition("):")
  if not sep:
   return False
  free=mine|LINK_WORDS
  padded=" "+flat+" "
  for sample in tail.split("|"):
   toks=Errand._sml_ALNUM.findall(sample.lower())
   if sum(map(len,toks))>=3 and not set(toks)<=free and" "+" ".join(toks)+" "in padded:
    return True
  return False
 def queue_second(self,subject):
  line=self.__dict__.setdefault("pending_seconds",[])
  if not subject:
   return
  lc=subject.lower()
  if not any(q.lower()==lc for q in line):
   line.append(subject)
 def _sml_tag(self,label,fallback=None):
  hit=self._sml_TAG.search(label)
  if hit:
   return hit.group(1).strip()
  if fallback is None:
   rows=self.needs.items()
   if len(rows)>1:
    bare=label.partition("(")[0].strip()
    owners=[k for k in range(len(rows))if bare and self.prop_line(k,bare)]
    if owners:
     return str(rows[owners[0]].get("product")or"item")
  return fallback if fallback is not None else str(self._sml_said("product")or"item")
 @staticmethod
 def _sml_plain(subject,product):
  name=subject.partition("(")[0].strip()
  return f"Which {name} would you like for the {product}?"
 def _sml_plain_for(self,label):
  return self._sml_plain(label,self._sml_tag(label))
 def _sml_queue_without(self,name):
  lc=name.lower()
  return[q for q in self._sml_kept("pending_seconds")or[]if q.lower()!=lc]
 def _sml_to_front(self,name):
  self.pending_seconds=[name,*self._sml_queue_without(name)]
 def _sml_keep_open(self,names):
  held=self.__dict__.setdefault("open_pair",[])
  for name in names:
   if all(q.lower()!=name.lower()for q in held):
    held.append(name)
 def _sml_spend(self,name):
  self.__dict__.setdefault("spent_asks",set()).add(name.lower())
 def _sml_freed(self):
  return len(self.needs.indifferent)+self.needs.__dict__.get("pinned_heard",0)
 def _sml_new_needs(self,walk):
  return" ".join(self.needs.uncovered[walk["found0"]:]).lower()
 @staticmethod
 def _sml_mentions(found_text,name):
  stems=Errand._sml_LONG4.findall(name.partition("(")[0].lower())
  return any(wd in found_text for wd in stems)
 @staticmethod
 def _sml_head(label):
  return label.partition(";")[0].strip()
 def pose_planned(self,urge=False):
  walk={"urge":urge,"out":False}
  step="gate"
  while step:
   step=getattr(self,"_sml_ask_"+step)(walk)
  return walk["out"]
 def _sml_ask_end(self,walk,value,idle=False):
  if idle:
   self.asking=False
  walk["out"]=value
  return None
 def _sml_ask_gate(self,walk):
  tool=self.role("message")
  walk["ceiling"]=top=self.ask_room()
  if not tool or self.msgs_sent>=top:
   return self._sml_ask_end(walk,False,idle=True)
  sheet,_=self.archive.prop_menu()
  reads=self.archive.rich_reads()
  cat=self.own_shelf()
  if cat:
   sheet,reads=self.archive.prop_menu(cat)[0],self.archive.rich_reads(cat)
  elif not self._sml_rows():
   try:
    ours=sorted(self.home_shelves()[0])
   except Exception:
    ours=[]
   strays=[label for label in self.archive.shelves_read()if label.strip().lower()not in ours]
   if ours and strays:
    sheet="\n".join(filter(None,(self.archive.prop_menu(label)[0]for label in ours)))
    reads=sum(self.archive.rich_reads(label)for label in ours)
  walk["menu"],walk["sets"]=sheet,reads
  walk["pressing"]=bool(self._sml_said("ask_first"))and not self.questions_log and(reads>=1 or self.wire.tick>=ASK_AFTER_STEPS)
  return"timing"
 def ask_sets(self):
  return ASK_AFTER_SETS+(0 if self.needs.items()else SOLO_SETS)
 def _sml_ask_timing(self,walk):
  sets,urge,tick=walk["sets"],walk["urge"],self.wire.tick
  if not(sets>=self.ask_sets()or walk["pressing"]or urge):
   if tick<ASK_AFTER_STEPS or(tick<ASK_AFTER_STEPS+ASK_GRACE_STEPS and self.moves_left()>20):
    return self._sml_ask_end(walk,False)
  if self.moves_left()<=max(10,self.order_due())or self.wire.spare()<CLOSING+40:
   return self._sml_ask_end(walk,False,idle=True)
  if self._sml_cold_product(walk):
   return self._sml_ask_end(walk,False)
  held=self._sml_kept("plan_wait")
  if(held is not None and sets<=held and not urge)or self._sml_ask_paced(walk):
   return self._sml_ask_end(walk,False)
  return"reads"
 def _sml_cold_product(self,walk):
  rows=self._sml_rows()
  if not rows or len(rows)>FEW_MAX or walk["urge"]:
   return False
  _,sent=self._smk_ask_tally(len(rows))
  owed=max(0,walk["ceiling"]-self.msgs_sent)
  if not any(sent)or self.moves_left()<=self.order_due()+2*owed+12:
   return False
  return any(not sent[pk]and self.product_reads(pk)<1 for pk in range(len(rows)))
 def _sml_ask_paced(self,walk):
  prior=self._sml_kept("asked_at_sets")
  if prior is None or walk["sets"]>prior or walk["urge"]or walk["pressing"]:
   return False
  left=self.moves_left()
  owed=max(0,walk["ceiling"]-self.msgs_sent)
  if left<=self.order_due()+2*owed+14:
   return False
  since=self.wire.tick-self._sml_kept("asked_at_tick",0)
  return since<4 and not self._sml_rows()
 def _sml_ask_reads(self,walk):
  if self.pre_read()or self.fit_reads():
   return self._sml_ask_end(walk,True)
  opening=not(self.questions_log or self.msgs_sent or walk["pressing"]or walk["urge"])
  if opening and self.own_survey():
   return self._sml_ask_end(walk,True)
  if opening and self.first_reads():
   return self._sml_ask_end(walk,True)
  return"brief"
 def own_survey(self):
  kv=self.__dict__
  if kv.get("own_surveyed")or self.needs.items()or not self.role("search")or not kv.get("refs"):
   return False
  try:
   home=self.home_shelves()[0]
   if not self.archive.ids or(home and(self.menu_rows(home,1,[],self.needs.quantity())
or any(self.archive.rich_reads(label)for label in home))):
    return False
  except Exception:
   return False
  kv["own_surveyed"]=True
  kind=" ".join(str(self.needs.reading.get("product")or"").split())
  if not kind or self.ran_query(kind)or self.moves_left()<=self.order_due()+8:
   return False
  shape=self.search_template()
  args={**shape[0],shape[1]:kind}if shape else self.plain_args(self.role("search"),kind)
  args,faults=ArgCheck.execute(args,self._smt_search_schema())
  if faults or not isinstance(args,dict)or kind not in self.free_text(args):
   return False
  tick=self.wire.tick
  self.log.append({"kind":"own_survey","query":kind})
  self.walk_call(args,"Searching for the asked-for product itself, so the fields its own listings carry are "
"known before any question.")
  return self.wire.tick!=tick
 def _sml_ask_brief(self,walk):
  left=str(walk["ceiling"]-self.msgs_sent)
  pieces=[ASK_PLAN.replace("QUESTION_COUNT",left),NEXT_ONE.replace("REMAINING",left),
"\n\nREQUEST, AS THE SHOPPER WROTE IT:\n"+self.request,
"\n\nREQUIREMENTS IN FORCE NOW:\n"+self.needs.text_body()]
  log=self.questions_log
  if log:
   pieces.append("\n\nQUESTIONS SENT EARLIER (each with the shopper's reply):\n"+"\n".join(log))
  if self._sml_said("ask_first"):
   musts=map(str,self._sml_said("ask_first")or[])
   pieces.append("\n\nMUST BE LEARNED FROM THE SHOPPER: "+"; ".join(musts))
  pieces.append(self.plan_notes(None))
  try:
   hints=self.open_hints()
  except Exception:
   hints=[]
  if hints:
   pieces.append(self._sml_HINTS_HEAD+"\n".join("- "+self.hint_mark(h)for h in hints[:8]))
  shown=self.menu_trim(walk["menu"],walk["sets"])
  rows=self._sml_rows()
  if rows:
   listed=[]
   for n,rec in enumerate(rows,1):
    listed.append(f"{n}) {rec['product']} x{rec['quantity']}")
   pieces.append("\n\nPRODUCTS IN THE REQUEST: "+" | ".join(listed))
   shown=self._sml_shelf_menus(walk,shown)
  pieces.append(self._sml_MENU_HEAD+(shown or"(none seen yet)"))
  walk["brief"]="".join(pieces)
  asked=set()
  for entry in log:
   subject=entry.split(":",1)[0].strip("- ")
   asked.update(name.strip().lower()for name in subject.split(";"))
  walk["asked"]=asked
  return"pick"
 def _sml_shelf_menus(self,walk,shown):
  seen=self.archive.shelves_read()
  homes=self.home_shelves()
  pool=set().union(*homes)if seen else set()
  own=[s for s in seen if s.strip().lower()in pool]
  if all(homes):
   order=own or seen
  else:
   order=own+[s for s in seen if s not in own]
  menus=[]
  for cat in order[:4]:
   menus.append((cat,self.archive.prop_menu(cat)[0]))
  filled=[(cat,sheet)for cat,sheet in menus if sheet]
  whole="\n".join("CATEGORY "+cat+":\n"+sheet for cat,sheet in filled)
  trimmed="\n".join("CATEGORY "+cat+":\n"+self.menu_trim(sheet,self.archive.rich_reads(cat))
for cat,sheet in filled)
  walk["menu"]=whole or walk["menu"]
  return trimmed or shown
 def _sml_fresh(self,walk,subject):
  name=str(subject or"").strip()
  if not name:
   return False
  lc=name.lower()
  if lc in walk["asked"]or junk_key(name)or numbered_twin(name)or lc in self._sml_kept("spent_asks",set()):
   return False
  k=self._sml_k(name)
  if self.plain_hint(k,name):
   return False
  return self.plan_open(k,name)
 def _sml_ask_pick(self,walk):
  line=[q for q in self._sml_kept("pending_seconds")or[]if self._sml_fresh(walk,q)]
  self.pending_seconds=line
  walk["pending"]=head=line[0]if line else""
  if not head:
   return"consult"
  texts=self._sml_kept("pending_text")or{}
  own_text=texts.pop(head.lower(),"")
  walk["answer"]={"questions":[{"property":head,"question":own_text or self._sml_plain_for(head)}]}
  return"script"
 def _sml_ask_consult(self,walk):
  plan=self.wire.consult(walk["brief"],2500,ASKER)
  walk["answer"]=plan
  if plan is not None or self.wire.misses<=0:
   return"script"
  self.plan_wait=walk["sets"]
  return self._sml_ask_end(walk,False)
 def _sml_vet(self,walk,item):
  extra=self._sml_field(item,"second").strip()
  if self._sml_NOT_SECOND.search(extra.lower()):
   extra=""
  if not self._sml_fresh(walk,item.get("property")):
   if extra and self._sml_fresh(walk,extra):
    return[{"property":extra,"question":self._sml_plain_for(extra)}]
   return[]
  usable=extra if extra and self._sml_fresh(walk,extra)else""
  entry=dict(item,second=usable)
  if extra and self.plain_hint(self._sml_k(extra),extra):
   entry["question"]=self._sml_plain_for(self._sml_field(item,"property").strip())
  return[entry]
 def _sml_ask_script(self,walk):
  picks=[]
  for item in(walk["answer"]or{}).get("questions")or[]:
   if isinstance(item,dict)and self._sml_field(item,"question").strip():
    picks+=self._sml_vet(walk,item)
  if not picks and not walk["pending"]and not self._sml_rows()and walk["sets"]>=self.ask_sets()and not self._sml_unread():
   picks=self._sml_menu_fallback(walk)
  elif picks and not walk["pending"]and not self._sml_rows():
   picks=self._sml_widest_first(walk,picks)
  walk["script"]=picks
  if walk["pending"]or not self._sml_rows():
   return"send"
  walk["script"]=self.quota_first(picks,lambda q:self._sml_fresh(walk,q))
  earlier=[entry for entry in self._sml_kept("plan_spares")or[]if entry not in picks]
  self.plan_spares=picks+earlier
  if self.__dict__.pop("quota_wait",False):
   return self._sml_ask_end(walk,True)
  return"send"
 _sml_MENU_LINE=re.compile(r"^- (.+?) \(in (\d+) listings\)(?: \[varies: (\d+) distinct values\])?")
 def _sml_menu_keys(self,every=False):
  try:
   ours=sorted(self.home_shelves()[0])
  except Exception:
   ours=[]
  menu="\n".join(self.archive.prop_menu(label)[0]for label in ours)
  keys=[]
  for text_line in menu.splitlines():
   hit=self._sml_MENU_LINE.match(text_line.strip())
   if not hit:
    continue
   name,count,spread=hit.group(1).strip(),int(hit.group(2)),int(hit.group(3)or 1)
   if every or not(count>=3 and spread>=count):
    keys.append((name,count))
  return sorted(keys,key=lambda pair:-pair[1])
 def _sml_menu_fallback(self,walk):
  for name,_ in self._sml_menu_keys():
   if self._sml_fresh(walk,name):
    self.log.append({"kind":"menu_fallback","key":name})
    return[{"property":name,"second":"","question":self._sml_plain_for(name)}]
  return[]
 def _sml_widest_first(self,walk,picks):
  try:
   asked=len(self.questions_log)
   if not asked or len(self.__dict__.get("quiet_asks")or[])<asked:
    return picks
   subject=self._sml_field(picks[0],"property").strip()
   hinted=[part for hint in self._smf_hints()for part in re.split(r"\s*/\s*",str(hint).partition(":")[0])]
   if any(same_prop(re.split(r"\s*\(",part)[0],subject)for part in hinted if part.strip()):
    return picks
   keys=self._sml_menu_keys()
   carried=next((n for name,n in self._sml_menu_keys(True)if same_prop(name,subject)),0)
   wide=next(((name,n)for name,n in keys if self._sml_fresh(walk,name)),None)
   if wide is None or wide[1]<carried+2 or same_prop(wide[0],subject):
    return picks
   self.log.append({"kind":"widest_first","key":wide[0],"over":subject})
   return[{"property":wide[0],"second":"","question":self._sml_plain_for(wide[0])}]+picks
  except Exception:
   return picks
 def _sml_shape(self,walk,item):
  if self._sml_rows():
   subject=self._sml_field(item,"property")
   product=self._sml_tag(subject,"")
   words=self._sml_LONG3.findall(product.lower())
   said=self._sml_field(item,"question").lower()
   if product and words and all(wd.rstrip("s")not in said for wd in words):
    item=dict(item,question=self._sml_plain(subject,product))
  if self._sml_said("ask_first")and not self.questions_log:
   return item
  text,subject,extra=(self._sml_field(item,slot)for slot in("question","property","second"))
  if self._sml_long_list(text,subject,walk["menu"]):
   return dict(item,question=self._sml_plain_for(subject))
  if walk["pending"]:
   return item
  if self.names_other(text,subject,walk["menu"],extra):
   item=dict(item,question=self._sml_plain_for(subject))
  return item
 _sml_EXAMPLES=re.compile(r"(?::|,|\bsuch as\b|\bfor example\b|\be\.g\.)(.+)$",re.I)
 def _sml_long_list(self,text,subject,sheet):
  try:
   hit=self._sml_EXAMPLES.search(text)
   if not hit:
    return False
   listed=any(strict_prop(line.strip().lstrip("-").strip().split(" (in ")[0],subject)
for line in str(sheet or"").splitlines()if" (in "in line)
   bits=[bit for bit in re.split(r",|\bor\b",hit.group(1))if bit.strip(" ?.!")]
   return listed and len(bits)>=3
  except Exception:
   return False
 def _sml_try_send(self,walk,item):
  text=str(item.get("question")).strip()
  cid=f"ask_{self.msgs_sent + 1}_{self.wire.tick}"
  walk["call"]=self.message_call(text,cid)
  if not walk["call"]:
   return False
  subject=str(item.get("property"))
  walk["label"]=(subject,text)
  extra=self._sml_field(item,"second").strip()
  if not extra or extra.lower()==subject.lower()or self._sml_NOT_SECOND.search(extra.lower()):
   return True
  single=self._sml_plain_for(subject)
  walk["call"]=self.message_call(single,cid)
  walk["label"]=(subject,single)
  if not walk["call"]:
   return False
  walk["split"]=extra
  return True
 def _sml_ask_send(self,walk):
  walk["split"]=""
  for item in walk["script"][:2]:
   if self._sml_try_send(walk,self._sml_shape(walk,item)):
    return"book"
  return"empty"
 def _sml_unread(self):
  try:
   several=len(self._sml_rows())
   return[k for k in range(several or 1)if self.product_reads(k if self._sml_rows()else None)<2]
  except Exception:
   return[]
 def _sml_ask_empty(self,walk):
  roomy=self.moves_left()>12
  if roomy and not walk["urge"]and(walk["sets"]<self.ask_sets()or self._sml_unread()):
   waiting=True
  else:
   self.empty_plans=self._sml_kept("empty_plans",0)+1
   waiting=roomy and self.empty_plans<=1
  if not waiting:
   return self._sml_ask_end(walk,False,idle=True)
  self.plan_wait=walk["sets"]
  return self._sml_ask_end(walk,False)
 def _sml_say(self,content,call):
  self.chat.append(dict(role="assistant",content=content,tool_calls=[call]))
 def _sml_ask_book(self,walk):
  subject=walk["label"][0]
  self.asked_at_sets=walk["sets"]
  self.asked_at_tick=self.wire.tick
  self._sml_say(f"I will ask the shopper about {subject}.",walk["call"])
  walk["found0"],walk["freed0"]=len(self.needs.uncovered),self._sml_freed()
  self.asking_label=subject
  try:
   walk["replies"]=self.dispatch([walk["call"]],scripted=True)
  finally:
   self.asking_label=""
  self.post_replies(walk["replies"])
  walk["asked_label"]=subject
  head=walk["pending"]
  if head and self._sml_head(subject).lower()==head.lower():
   self.pending_seconds=self._sml_queue_without(head)
  walk["note"]=""
  walk["meta"]=self.needs.__dict__.pop("meta_last",None)
  walk["lost"]=self.needs.__dict__.pop("lost_last",None)
  return"reply"
 def _sml_ask_reply(self,walk):
  subject,text=walk["label"]
  meta=walk["meta"]
  heard=self._sml_flat(text)
  paired=";"in subject
  lost=walk.get("lost")
  cases=((lambda:lost is not None and bool(heard)and heard in self._sml_flat(lost),self._sml_rp_lost),
(lambda:meta is not None and bool(heard)and heard in self._sml_flat(meta),self._sml_rp_single),
(lambda:paired and len(self.needs.uncovered)>walk["found0"],self._sml_rp_revealed),
(lambda:paired and self._sml_freed()>walk["freed0"],self._sml_rp_waived),
(lambda:paired and self._sml_freed()==walk["freed0"],self._sml_rp_missed),
(lambda:bool(walk["split"]),self._sml_rp_split))
  for test,act in cases:
   if test():
    act(walk)
    break
  return"log"
 def _sml_rp_lost(self,walk):
  subject,text=walk["label"]
  pieces=[q.strip()for q in subject.split(";")if q.strip()]or[subject]
  lead=pieces[0]
  lead_low=lead.lower()
  again=self.__dict__.setdefault("lost_asks",{})
  times=again.get(lead_low,0)
  if times>=LOST_REASKS:
   self._sml_spend(lead)
   walk["note"]=self._sml_note(lead,"not understood again, spent")
  else:
   again[lead_low]=times+1
   redo=self.distinct_ask(lead)
   name,plain=redo if redo else(lead,self._sml_simple_for(lead))
   if self._sml_flat(plain)==self._sml_flat(text):
    plain=self._sml_simple_for(lead)
   self.__dict__.setdefault("pending_text",{})[name.lower()]=plain
   self._sml_to_front(name)
   walk["note"]=self._sml_note(lead,"not understood, asked once more in plainer words next")
  extra=[walk["split"]]if walk["split"]else[]
  self._sml_keep_open(pieces[1:]+extra)
  walk["label"]=(" and ".join(pieces)+" [not understood]",text)
 def _sml_simple_for(self,label):
  name=label.partition("(")[0].strip()
  product=self._sml_tag(label)
  samples=[]
  try:
   pk=self.product_of(label)if self._sml_rows()else None
   for val in sorted(self.field_values(pk,name)or[],key=len):
    val=str(val).strip()
    if val and len(val)<=30 and re.search(r"[a-z]",val.lower())and not re.search(r"[,;/]",val)and val.lower()not in[x.lower()for x in samples]:
     samples.append(val)
    if len(samples)==2:
     break
  except Exception:
   samples=[]
  tail=f" For example {samples[0]} or {samples[1]}."if len(samples)==2 else""
  return f"Sorry, let me put it simply: for the {product}, what {name} do you want it to have?{tail}"
 def _sml_rp_single(self,walk):
  subject,text=walk["label"]
  self.no_pairs=True
  pieces=[q.strip()for q in subject.split(";")if q.strip()]or[subject]
  lead=pieces[0]
  lead_low=lead.lower()
  again=self.__dict__.setdefault("reasked",set())
  plain_q=self._sml_plain_for(lead)
  alone=len(pieces)==1
  alt=self.collide_alt(lead)if alone else""
  if alone and self._sml_rows()and not alt:
   if lead_low in again:
    self._sml_spend(lead)
    walk["note"]=self._sml_note(lead,"not answered twice, spent; another product is asked next")
   else:
    again.add(lead_low)
    self.__dict__.setdefault("pending_text",{})[lead_low]=plain_q
    self._sml_to_front(lead)
    walk["note"]=self._sml_note(lead,"not answered, asked again alone next")
  elif alt:
   again.add(lead_low)
   self._sml_spend(lead)
   self._sml_to_front(alt)
   walk["note"]=self._sml_note(lead,f"not answered, its story hint's alternative {alt} asked next")
  elif lead_low in again or self._sml_flat(plain_q)==self._sml_flat(text)or(alone and self.holds_plain(text,plain_q)):
   walk["note"]=self._sml_reask(lead,again)
  else:
   again.add(lead_low)
   self._sml_to_front(lead)
   walk["note"]=self._sml_note(lead,"not answered, asked again alone next")
  extra=[walk["split"]]if walk["split"]else[]
  self._sml_keep_open(pieces[1:]+extra)
  walk["label"]=(" and ".join(pieces)+" [not answered] [one at a time]",text)
 def _sml_reask(self,lead,again):
  redo=None if lead.lower()in again else self.distinct_ask(lead)
  if not redo:
   self._sml_spend(lead)
   return self._sml_note(lead,"not answered, not asked again")
  name=redo[0]
  again.update({lead.lower(),name.lower()})
  self.__dict__.setdefault("pending_text",{})[name.lower()]=redo[1]
  self._sml_to_front(name)
  same=re.split(r"\s*\(",name)[0].strip().lower()==re.split(r"\s*\(",lead)[0].strip().lower()
  if same:
   return self._sml_note(lead,"not answered, asked again by its values next")
  self._sml_spend(lead)
  return self._sml_note(lead,"not answered, asked again as "+name+" next")
 def _sml_rp_revealed(self,walk):
  subject,text=walk["label"]
  other=subject.partition(";")[2].strip()
  if not self._sml_mentions(self._sml_new_needs(walk),other):
   self.queue_second(other)
   walk["note"]=self._sml_note(other,"not answered, still open")
  walk["label"]=(self._sml_head(subject),text)
 def _sml_rp_waived(self,walk):
  subject,text=walk["label"]
  other=subject.partition(";")[2].strip()
  walk["label"]=(self._sml_head(subject),text)
  gone={other.lower(),other.partition("(")[0].strip().lower()}
  self.needs.indifferent=[q for q in self.needs.indifferent if q.strip().lower()not in gone]
  self._sml_keep_open([other])
  walk["note"]=self._sml_note(other,"not answered, still open")
 def _sml_rp_missed(self,walk):
  subject,text=walk["label"]
  self.no_pairs=True
  self._sml_keep_open([q.strip()for q in subject.split(";")if q.strip()])
  walk["label"]=(subject.replace(";"," and")+" [not answered]",text)
 def _sml_rp_split(self,walk):
  split=walk["split"]
  settled=self._sml_mentions(self._sml_new_needs(walk),split)
  self._sml_keep_open([]if settled else[split])
 def _sml_ask_log(self,walk):
  self.reveal_note(self.needs.uncovered[walk["found0"]:])
  if walk["meta"]is None and len(self.needs.uncovered)==walk["found0"]:
   self.__dict__.setdefault("quiet_asks",[]).append(self._sml_head(walk["asked_label"]))
  subject,text=walk["label"]
  for _,body in walk["replies"][:1]:
   reply=body.split("\n[status]")[0][:400]
   self.questions_log.append("- "+subject+": asked \""+text+"\" -> "+reply+walk["note"])
  self.drop_wanted(walk["asked_label"])
  self.asking=self.msgs_sent<walk["ceiling"]
  return self._sml_ask_end(walk,True)
 def reply_back(self,phrase):
  if not self.role("message"):
   return False
  room,steps,clock=self.ask_room,self.moves_left,self.wire.spare
  blocked=(lambda:self.msgs_sent>=room(),
lambda:steps()<=4,
lambda:clock()<CLOSING,
lambda:self.asking and self.msgs_sent>=room()-1 and steps()>10 and clock()>=CLOSING+40)
  if any(gate()for gate in blocked):
   return False
  call=self.message_call(phrase,f"reply_{self.wire.tick}")
  if not call:
   return False
  self._sml_say("I will answer the shopper's question.",call)
  sent=self.dispatch([call],scripted=True)
  self.post_replies(sent)
  return True
 def post_replies(self,responses):
  last=len(responses)-1
  for n,(cid,body)in enumerate(responses):
   content=body+"\n"+self.status_line()if n==last else body
   self.chat.append(dict(role="tool",tool_call_id=cid,content=content))
 @staticmethod
 def _sml_cut(text,spec):
  return text[:spec.get("maxLength",len(text))]
 @staticmethod
 def _sml_fill(spec,phrase,cue):
  if spec.get("enum"):
   options=spec["enum"]
   for option in options:
    if cue and isinstance(option,str)and re.search(cue,option.lower()):
     return[option]
   if cue==DECLINE_WORDS:
    for option in options:
     if isinstance(option,str)and not re.search(ORDER_WORDS,option.lower()):
      return[option]
   return[options[0]]
  typed=spec.get("type")
  if typed=="string"or"string"in(typed or[]):
   return[Errand._sml_cut(phrase,spec)]
  if typed=="boolean":
   return[True]
  return[max(1,spec.get("minimum",1))]if typed in("integer","number")else[]
 def plain_args(self,call_name,phrase,kind=""):
  schema=self.shapes.get(call_name)or{}
  props=schema.get("properties")or{}
  cue={"decline":DECLINE_WORDS,"order":ORDER_WORDS}.get(kind,"")
  made={}
  for name in schema.get("required")or[]:
   made.update((name,value)for value in self._sml_fill(props.get(name)or{},phrase,cue))
  if len(props)!=1:
   return made
  (only,spec),=props.items()
  free_text=isinstance(spec,dict)and spec.get("type")=="string"and"enum"not in spec
  if free_text and only not in made:
   made[only]=self._sml_cut(phrase,spec)
  return made
 def _sml_names_id(self,args_in):
  known=self.archive.ids
  texts=[piece for piece in args_in.values()if isinstance(piece,str)]
  if any(tok and tok in known for piece in texts for tok in self._sml_ID_CUT.split(piece)):
   return True
  return self._sml_names_title(texts)
 def _sml_names_title(self,texts):
  try:
   info=self.code_rows()
  except Exception:
   return False
  low=" ".join(" ".join(t.lower().split())for t in texts)
  if not low:
   return False
  for row in info.values():
   words=[w for w in re.findall(r"[a-z0-9]+",str(row.get("title")or"").lower())if len(w)>2]
   if len(words)>=3 and" ".join(words[:4])in low:
    return True
  return False
 def scrub_decline(self,args_in,call_name):
  rule=self._sml_said("decline_no_names")
  if rule is False:
   return args_in
  named=self._sml_names_id(args_in)
  if not(rule or named):
   return args_in
  layout=(self.shapes.get(call_name)or{}).get("properties")or{}
  def scrubbed(slot,piece):
   if not isinstance(piece,str):
    return piece
   spec=layout.get(slot)or{}
   return piece if"enum"in spec else self._sml_cut(SORRY,spec)
  return{slot:scrubbed(slot,piece)for slot,piece in args_in.items()}
 def code_reading(self):
  request=self.request
  lc=request.lower()
  cap,unit=self.code_budget(request)
  plan=dict(code_reading=True,requirements=[],items=[],quantity=1,rules=[],soft_wants=[],
ask_first=[],hints=[],references=[],priority_detail="",price_cap=cap,currency=unit,
in_stock_only=bool(self._sml_STOCK.search(lc)),
priority="cheapest"if self._sml_CHEAP.search(lc)else"none")
  for slot,nouns,floor in self._sml_LIMITS:
   figure=self.code_limit(lc,nouns)
   if figure is not None and figure>=floor:
    plan[slot]=figure
  roles=self.code_roles()
  plan.update(roles=roles)
  found=self.code_items(roles.get("search"))
  if len(found)>1:
   plan.update(items=[dict(product=rec["product"],quantity=rec["quantity"],requirements=rec["requirements"])
for rec in found],
product=" and ".join(rec["product"]for rec in found),
requirements=[need for rec in found for need in rec["requirements"]])
   return plan
  if found:
   plan.update(product=found[0]["label"],quantity=found[0]["quantity"])
  else:
   plan.update(product=self.code_label(roles.get("search"),kind_words(request)),quantity=self.code_count(lc))
  plan.update(requirements=[f"{subject}: {value}"for subject,value in self.code_needs(lc)])
  return plan
 @staticmethod
 def _sml_budget_score(sentence,hit,bonus):
  after=sentence[hit.end():]
  sign=hit.group(1)
  if Errand._sml_COUNTED.match(after)or(not sign and Errand._sml_IN_RUN.search(sentence[:hit.start()])):
   return None
  before=sentence[max(0,hit.start()-60):hit.start()]
  points=bonus
  if sign or Errand._sml_SPEND.search(before):
   points+=3
  if Errand._sml_CUE_BEFORE.search(before):
   points+=2
  if Errand._sml_CUE_AFTER.match(after):
   points+=2
  return points
 @staticmethod
 def _sml_budget_unit(after,sign):
  head=Errand._sml_UNIT.match(after)
  kept=[]
  for raw in(head.group(1).split()if head else[]):
   word=raw.strip(".")
   if word not in LINK_WORDS and len(word)>=2 and not re.search(r"\d",word):
    kept.append(word)
  return" ".join(kept)or(sign or"").strip()or None
 @staticmethod
 def _sml_scored(phrase):
  for sentence in Errand._sml_SENTENCE.split(str(phrase).lower()):
   bonus=1 if Errand._sml_WHOLE.search(sentence)else 0
   for hit in Errand._sml_FIGURE.finditer(sentence):
    score=Errand._sml_budget_score(sentence,hit,bonus)
    if score is not None:
     yield hit,sentence,score
 @staticmethod
 def code_budget(phrase):
  ranked=[(score,float("".join(hit.group(2).split(","))),hit,sentence)
for hit,sentence,score in Errand._sml_scored(phrase)if score>=2]
  if not ranked:
   return None,None
  _,figure,hit,sentence=max(ranked,key=lambda rec:rec[:2])
  amount=int(figure)if figure.is_integer()else figure
  return amount,Errand._sml_budget_unit(sentence[hit.end():],hit.group(1))
 @staticmethod
 def code_limit(lc,nouns):
  spelled=dict(Errand._sml_WORD_NUMS)
  counted=r"\b(\d+|"+"|".join(spelled)+r")\s+(?:[a-z-]+\s+){0,2}?(?:"+nouns+r")\b"
  for hit in re.finditer(counted,lc):
   start,end=hit.span()
   capped=Errand._sml_CAP_BEFORE.search(lc[max(0,start-60):start])or Errand._sml_CAP_AFTER.match(lc[end:end+40])
   if capped:
    said=hit.group(1)
    return int(said)if said.isdigit()else spelled[said]
  return None
 _smm_SMALL=("one","two","three","four","five")+("six","seven","eight","nine","ten")
 _smm_FIGURE=r"(\d+|one|two|three|four|five|six|seven|eight|nine|ten)"
 _smm_TYPES={"text":("string",),"number":("integer","number")}
 _smm_BLANK=(None,False,"",[],{})
 _smm_INF=float("inf")
 _smm_MARKS=re.compile(r"\((?:stated|revealed[^)]*)\)")
 _smm_LINKED=r"\b(?:"+VALUE_LINKS+r")\b"
 _smm_SPLIT_VALUE=re.compile(r"\bor\b|/|,|"+_smm_LINKED)
 _smm_BOUND=re.compile(_smm_LINKED+r"|\d\s*(?:-|to)\s*\d|[+<>]|\b(?:about|around|approx\w*|roughly)\b")
 _smm_PAGING=re.compile(r"page|first|offset|skip|count|size|num")
 _smm_CEILING=re.compile(r"ceil|max|price|cap$|budget|upper|most")
 @staticmethod
 def _smm_figure(token):
  if token.isdigit():
   return int(token)
  return Errand._smm_SMALL.index(token)+1
 @staticmethod
 def _smm_content(text,pattern=r"[a-z0-9]+"):
  return[w for w in re.findall(pattern,text.lower())if w not in LINK_WORDS]
 @staticmethod
 def _smm_join_once(words):
  kept=[]
  for w in words:
   if w not in kept:
    kept.append(w)
  return" ".join(kept)
 @staticmethod
 def _smm_choices(spec):
  return spec.get("enum")if isinstance(spec,dict)else None
 @staticmethod
 def _smm_decamel(text):
  return re.sub(r"([a-z])([A-Z])",r"\1 \2",text)
 @staticmethod
 def _smm_says(pattern,texts):
  return any(re.search(pattern,t)for t in texts)
 def _smm_schema(self,tool):
  form=self.shapes.get(tool or"")or{}
  args=form.get("properties")
  return form,(args if isinstance(args,dict)else{})
 @staticmethod
 def _smm_plain_args(args,kind):
  accepted=Errand._smm_TYPES[kind]
  return[name for name,spec in args.items()if isinstance(spec,dict)and not spec.get("enum")and spec.get("type")in accepted]
 @staticmethod
 def code_count(said):
  found=re.search(r"\b"+Errand._smm_FIGURE+r"\s+(?:units?|of (?:them|those|these)|copies)\b",said)
  units=Errand._smm_figure(found.group(1))if found else 1
  return units if 0<units<100 else 1
 _smm_NEED_VERBS=r"(?:is|are|has to be|have to be|must be|should be|needs to be|need to be|can be)"
 _smm_NEED_END=(r"(?=\s*(?:[,.;:!?()]|$)|\s+(?:so|since|as|because|for|to|which|that|nothing|otherwise|but|when|while|if|"
r"then|though|please|and (?:the|its|their|a|an|one|two|three|\d))\b)")
 _smm_NEED_LEAD=(r"(?:\b(?:where|whose)\s+(?:the\s+|its\s+|their\s+)?|(?:\b(?:and|with|but)\s+|[,;:]\s*|^|[.!?]\s+)"
r"(?:the|its|their)\s+)")
 _smm_NOT_PROP=(r"\b(?:i|you|we|it|he|she|they|my|your|our|there|this|that|which|what|who|me|us|them|one|ones|price|budget|"
r"cost|total|order|stock)\b")
 _smm_NOT_VALUE=(r"(?:a|an|the|my|your|our|his|her|their|me|us|for|to|in|on|at|of|see|go|get|be|have|only|too|very|so|not|"
r"no|what|how|it|all|mostly|always|never|just|really|still|also|from|about|like|one|ones)\b")
 @staticmethod
 def _smm_need_kept(stretch,hit):
  prop,value=(part.strip()for part in hit.groups())
  after=stretch[hit.end():hit.end()+60]
  if ref_text(value)or(ref_text(after)and re.search(r"\b(?:page|listing|shown|sold by)\b",after)):
   return False
  if not value or any(len(t.split())>4 for t in(value,prop)):
   return False
  return not(re.search(Errand._smm_NOT_PROP,prop)or re.match(Errand._smm_NOT_VALUE,value))
 @staticmethod
 def code_needs(stretch):
  shape=(Errand._smm_NEED_LEAD+r"([a-z][a-z'&/ -]{1,32}?)\s+"+Errand._smm_NEED_VERBS
+r"\s+(?:either\s+|just\s+)?([^,.;:!?()]{1,60}?)"+Errand._smm_NEED_END)
  pairs=[]
  for hit in re.finditer(shape,stretch):
   if Errand._smm_need_kept(stretch,hit):
    pairs.append(tuple(part.strip()for part in hit.groups()))
  return pairs
 def _smm_shelf_list(self,tool):
  longest=[]
  for spec in self._smm_schema(tool)[1].values():
   choices=Errand._smm_choices(spec)
   if isinstance(choices,list)and len(choices)>=4 and len(choices)>len(longest):
    longest=choices
  return longest
 @staticmethod
 def _smm_label_count(text,toks,own):
  width=len(own)
  before=r"\b"+Errand._smm_FIGURE+r"\s*(?:x\s+|\u00d7\s*)?(?:(?:of|[a-z'&-]{3,})\s+){0,4}$"
  for at in range(len(toks)-width+1):
   window=toks[at:at+width]
   if[word for word,_,_ in window]!=own:
    continue
   start,end=window[0][1],window[-1][2]
   if sum(1 for _ in re.finditer(r"[a-z0-9]+",text[start:end]))>width*2:
    continue
   figure=re.search(before,text[start-56 if start>56 else 0:start])
   if figure:
    return start,end,Errand._smm_figure(figure.group(1))
  return None
 def code_items(self,tool):
  shelf=[str(label)for label in self._smm_shelf_list(tool)]
  text=self.request.lower()
  toks=[]
  for hit in re.finditer(r"[a-z]{3,}",text):
   if hit.group(0)not in LINK_WORDS:
    toks.append((hit.group(0).rstrip("s"),*hit.span()))
  spans={}
  for label in shelf:
   own=[w.rstrip("s")for w in self._smm_content(label,r"[a-z]{3,}")]
   if own and label not in spans:
    place=self._smm_label_count(text,toks,own)
    if place and 0<place[2]<100:
     spans[label]=place
  found=sorted(spans.items(),key=lambda pair:pair[1][0])
  result=[]
  for idx,(label,(_,end,units))in enumerate(found):
   if idx+1<len(found):
    stretch=text[end:found[idx+1][1][0]]
   else:
    stretch=re.split(r"(?<=[.!?])\s",text[end:])[0]
   product=label.lower()
   result.append({"product":product,"label":label,"quantity":units,
"requirements":["%s (%s): %s"%(prop,product,value)for prop,value in self.code_needs(stretch)]})
  return result
 def _smm_tool_card(self,name,text):
  form,args=self._smm_schema(name)
  wanted=form.get("required")
  need=[key for key in wanted if key in args]if isinstance(wanted,list)else[]
  both=False
  for key in need:
   spec=args[key]
   if not isinstance(spec,dict):
    continue
   lowered=list(map(lambda v:str(v).lower(),spec.get("enum")or[]))
   pair=self._smm_says(ORDER_WORDS,lowered)and self._smm_says(DECLINE_WORDS,lowered)
   both=both or pair
  specs=[spec for spec in args.values()if isinstance(spec,dict)]
  return{"name":name,"text":text,"head":text.split(" | ")[0],"need":need,"both":both,
"free":self._smm_plain_args(args,"text"),
"array":any(spec.get("type")=="array"for spec in specs),
"choice":any(spec.get("enum")for spec in specs)}
 @staticmethod
 def _smm_roles_paired(cards,roles):
  for card in filter(lambda c:c["both"],cards):
   roles.setdefault("order",card["name"])
   roles.setdefault("decline",card["name"])
 @staticmethod
 def _smm_roles_finish(cards,roles,final):
  for card in cards:
   if card["name"]in roles.values():
    continue
   if final and(card["need"]or not re.search(r"\bfinal\b",card["text"])):
    continue
   words=card["text"]if final else card["head"]
   role=next((role for role,cue in(("decline",DECLINE_WORDS),("order",ORDER_WORDS))
if role not in roles and re.search(cue,words)),None)
   if role:
    roles[role]=card["name"]
 @staticmethod
 def _smm_roles_cart(cards,roles):
  for card in cards:
   if card["name"]not in roles.values()and re.search(r"\b(?:cart|basket|bag|trolley)\b",card["text"]):
    roles.setdefault("cart"if card["need"]or card["array"]else"cart_view",card["name"])
 _smm_PERSON=r"\b(?:shoppers?|customers?|buyers?|clients?|users?)\b"
 _smm_SPEAK=r"\b(?:send\w*|repl\w*|notes?|tell\w*|messag\w*|chat\w*|writ\w*|talk\w*)\b"
 _smm_CUES=(("search",r"\b(?:search\w*|find\w*|look\w*|locat\w*|hunt\w*|brows\w*|quer\w*)\b"),
("message",r"\b(?:send\w*|repl\w*|notes?|tell\w*|messag\w*|chat\w*|writ\w*|talk\w*)\b"),
("details",r"\b(?:details?|records?|spec\w*|propert\w*|attribut\w*|facts?|full)\b"),
("view",r"\b(?:offer|price|stock|availab\w*)\b"))
 @staticmethod
 def _smm_cue_fits(role,card):
  text=card["text"]
  to_person=re.search(Errand._smm_PERSON,text)
  if role=="search":
   if to_person and re.search(Errand._smm_SPEAK,text):
    return False
   return bool(card["free"])or card["choice"]
  if role=="message"and not(to_person or re.search(r"\b(?:messag\w*|chat\w*)\b",text)):
   return False
  return len(card["need"])==1 and card["need"][0]in card["free"]
 @staticmethod
 def _smm_roles_cued(cards,roles):
  for role,cue in Errand._smm_CUES:
   if role in roles:
    continue
   for card in cards:
    if card["name"]not in roles.values()and re.search(cue,card["text"])and Errand._smm_cue_fits(role,card):
     roles[role]=card["name"]
     break
 def _smm_cards(self):
  words={}
  for entry in self.toolset:
   fn=entry["function"]
   spaced=self._smm_decamel(fn["name"]).replace("_"," ")
   words[fn["name"]]="%s | %s"%(spaced.lower(),str(fn.get("description")or"").lower())
  return[self._smm_tool_card(name,text)for name,text in words.items()]
 def code_roles(self):
  cards=self._smm_cards()
  found={}
  self._smm_roles_paired(cards,found)
  self._smm_roles_finish(cards,found,False)
  self._smm_roles_finish(cards,found,True)
  self._smm_roles_cart(cards,found)
  self._smm_roles_cued(cards,found)
  return found
 def search_role_fix(self):
  held=self.role("search")
  try:
   cards=self._smm_cards()
  except Exception:
   return
  mine=next((card for card in cards if card["name"]==held),None)
  if mine is None or(mine["free"]and not self._smv_cap_tool(held)):
   return
  cue=dict(self._smm_CUES)["search"]
  for card in cards:
   if card["name"]in self.role_map.values()or not card["free"]or not re.search(cue,card["text"]):
    continue
   if self._smm_cue_fits("search",card):
    self.role_map["search"]=card["name"]
    self.log.append({"kind":"search_role","from":held,"to":card["name"]})
    return
 def code_label(self,tool,want):
  best,top="",None
  for label in self._smm_shelf_list(tool):
   own=kind_words(label)
   shared=len(own&want)
   score=(shared,shared/max(1,len(own)))
   if shared and(top is None or score>top):
    best,top=str(label),score
  return best
 def code_values(self,k):
  several=bool(self.needs.items())
  kept=[]
  for need in map(str,self.needs.props_left()):
   head,_,value=need.partition(":")
   owner=self.product_of(head)if several else None
   if owner is not None and owner!=k:
    continue
   bare=Errand._smm_MARKS.sub("",value).strip()
   lowered=bare.lower()
   named=positive_value(value,head)or(len(re.findall(r"[a-z0-9]+",lowered))<=5 and not re.search(EXCLUDE_WORDS,lowered))
   if named and bare and not ref_text(bare):
    kept.append((head.split("(")[0].strip(),bare))
  return kept
 def _smm_search_form(self,tool):
  form,args=self._smm_schema(tool)
  need=[key for key in form.get("required")or[]if key in args]
  about=""
  for entry in self.toolset:
   fn=entry["function"]
   if fn["name"]==tool:
    about=str(fn.get("description")or"")
    break
  clauses=[piece.lower()for piece in re.split(r"[.;]",about)]
  tok=self.token_key()
  free=[key for key in self._smm_plain_args(args,"text")if key!=tok]
  worded=[key for key in free if re.search(r"^q$|query|text|keyword|needle|term|word|phrase|string|search",key.lower())]
  lists={}
  for key,spec in args.items():
   choices=Errand._smm_choices(spec)
   if isinstance(choices,list)and choices:
    lists[key]=choices
  shelves=[key for key in lists if len(lists[key])>=4]
  shelf_key=max(shelves,key=lambda key:len(lists[key]))if shelves else None
  modes=[key for key,choices in lists.items()if key!=shelf_key and len(choices)<=4]
  def priced(key):
   low=key.lower()
   if Errand._smm_PAGING.search(low):
    return False
   return bool(Errand._smm_CEILING.search(low))or any(low in part and re.search(r"\bpric|\bcost",part)for part in clauses)
  return{"schema":form,"args":args,"need":need,"clauses":clauses,"word":(worded or free or[None])[0],
"lists":lists,"shelf":shelf_key,"mode":([key for key in modes if key in need]or modes or[None])[0],
"ceiling":next(filter(priced,self._smm_plain_args(args,"number")),None)}
 @staticmethod
 def _smm_mode_value(form,arg,last):
  def bounded(word):
   return r"\b"+re.escape(word.lower())+r"\b"
  options=form["lists"].get(form["mode"]or"",[])
  naming=[part for part in form["clauses"]if re.search(bounded(arg),part)]
  for val in options:
   if any(re.search(bounded(str(val)),part)for part in naming):
    return val
  return options[-1 if last else 0]
 def _smm_query_texts(self,k,product,label,room):
  names=[w for w in self._smm_content(product)if len(w)>2]
  kind=(names or[""])[-1]
  solid=[value for _,value in self.code_values(k)if not Errand._smm_SPLIT_VALUE.search(value.lower())]
  lead=self._smm_content(solid[0])if solid else[]
  if lead:
   texts=[self._smm_join_once(lead+[kind])]
   if len(lead)>1:
    texts.append(self._smm_join_once([max(lead,key=len),kind]))
  elif names:
   texts=[self._smm_join_once([names[0],names[-1]])]
  else:
   texts=[" ".join(sorted(kind_words(label))[:2])]
  if not is_number(room):
   return texts
  cut=[]
  for text in texts:
   while" "in text and len(text)>room:
    text=text[:text.rindex(" ")]
   cut.append(text[:int(room)])
  return cut
 def code_queries(self,k):
  tool=self.role("search")
  form=self._smm_search_form(tool)
  args,word_key,shelf_key,mode_key,ceiling=form["args"],form["word"],form["shelf"],form["mode"],form["ceiling"]
  listed=self.needs.items()
  if listed and k is not None and k<len(listed):
   product=str(listed[k]["product"])
  else:
   reading=self.needs.reading
   product=str(reading.get("product")or"")
  want=kind_words(product)or kind_words(self.request)
  label=self.code_label(tool,want)
  cap=self.needs.spend_cap
  per=self.needs.counts()if k is not None else[]
  units=(per[k]if k is not None and k<len(per)else self.needs.quantity())or 1
  limit={}
  if ceiling and is_number(cap)and cap>0:
   limit[ceiling]=int(cap//units)if args[ceiling].get("type")=="integer"else round(cap/units,2)
  word_spec=args.get(word_key)if word_key else None
  texts=self._smm_query_texts(k,product,label,word_spec.get("maxLength")if isinstance(word_spec,dict)else None)
  plan=[]
  if word_key and texts[0]:
   plan.append((word_key,texts[0],False))
  if label and shelf_key:
   plan.append((shelf_key,label,True))
  if word_key and len(texts)>1:
   plan.append((word_key,texts[1],False))
  made=[]
  for key,what,last in plan:
   call_args={key:what}
   if mode_key:
    call_args[mode_key]=self._smm_mode_value(form,key,last)
   call_args.update(limit)
   self._smm_fill_required(call_args,form,texts[0]or label)
   result=ArgCheck.execute(call_args,form["schema"])
   if result[1]and limit:
    result=ArgCheck.execute({name:val for name,val in call_args.items()if name not in limit},form["schema"])
   if not result[1]and result[0]not in made:
    made.append(result[0])
  return made
 @staticmethod
 def _smm_fill_required(call_args,form,words):
  for name in filter(lambda key:key not in call_args,form["need"]):
   spec=form["args"][name]
   spec=spec if isinstance(spec,dict)else{}
   choices,kind=spec.get("enum"),spec.get("type")
   if choices:
    call_args[name]=choices[0]
   elif kind in("integer","number"):
    call_args[name]=spec["default"]if"default"in spec else spec.get("minimum",1)
   elif kind=="string":
    call_args[name]=words
 @staticmethod
 def _smm_price_key(name):
  return bool(re.search(r"price|cost|amount",name))and not re.search(r"count|qty|quant|total|units|line|due|payable|sum",name)
 def _smm_take_record(self,seen,d):
  mark=self.row_mark(d)
  if mark in self.archive.ids:
   self._smm_merge_row(seen.setdefault(mark,{"shelf":"","title":"","price":None,"stock":None,"words":set(),"fields":{}}),d)
 def _smm_merge_row(self,row,d):
  row["shelf"]=self.shelf_of(d)or row["shelf"]
  for field,val in d.items():
   low=str(field).lower()
   if isinstance(val,str)and"id"not in low and re.search(r"title|name",low):
    row["title"]=str(val)or row["title"]
    break
  unit=next((float(val)for key,val in d.items()if is_number(val)and self._smm_price_key(str(key).lower())),None)
  if unit is not None:
   row["price"]=unit
  said=[state for state in(self.key_stock(key,val)for key,val in d.items())if state is not None]
  if said:
   row["stock"]=said[-1]
  scalar=(str,int,float)
  flat,nested=[],[]
  for piece in d.values():
   if isinstance(piece,dict):
    nested.append(piece)
   elif isinstance(piece,scalar)and not isinstance(piece,bool):
    flat.append(str(piece))
  flat+=[str(x)for sub in nested for x in sub.values()if isinstance(x,scalar)]
  row["words"].update(kind_words(" ".join(flat)))
  for sub in nested:
   for key,x in sub.items():
    if isinstance(x,scalar)and not isinstance(x,bool):
     row["fields"][str(key)]=str(x)
 def code_rows(self):
  seen={}
  for stored in self.archive.items:
   walk_dicts(stored,lambda d:self._smm_take_record(seen,d))
  return seen
 @staticmethod
 def _smm_prop_fields(fields,prop):
  return[text for name,text in fields.items()if same_prop(name,prop)]
 @staticmethod
 def code_shows(row,prop,value,hit_words,read):
  fielded=Errand._smm_prop_fields(row["fields"],prop)
  if fielded:
   return any(Errand.code_seen(value,text)for text in fielded)
  if Errand._smm_title_shows(row,value):
   return True
  wanted=set()if read else kind_words(value)
  return bool(wanted)and wanted<=hit_words
 @staticmethod
 def _smm_title_shows(row,value):
  return Errand.code_seen(value,row.get("title")or"")
 @staticmethod
 def _smm_alt_in(alt,pool):
  figures=re.findall(r"\d+(?:\.\d+)?",alt)
  stems=[w if len(w)<=4 else w[:-1]for w in re.findall(r"[a-z]{2,}",alt)if w not in LINK_WORDS]
  if not(stems or figures):
   return False
  return all(re.search(r"\b"+re.escape(stem),pool)for stem in stems)and all(re.search(r"(?<![\d.])"+re.escape(fig)+r"(?![\d.])",pool)for fig in figures)
 @staticmethod
 def code_seen(value,text):
  pool=str(text).lower()if text else""
  if not pool or pool.isspace():
   return False
  body=re.sub(Errand._smm_LINKED," ",str(value).lower())
  return any(Errand._smm_alt_in(alt,pool)for alt in re.split(r"\s*(?:\bor\b|/|,|;)\s*",body))
 @staticmethod
 def _smm_value_rank(row,prop,value):
  def joined(text):
   return" ".join(ch_words(text))
  goal={joined(alt)for alt in re.split(r"\s*(?:\bor\b|/)\s*",value.lower())if ch_words(alt)}
  ranks=[]
  for name,text in row["fields"].items():
   if not(same_prop(name,prop)and Errand.code_seen(value,text)):
    continue
   tier=0 if joined(text)in goal else(1 if any(joined(part)in goal for part in ch_parts(text))else 3)
   ranks.append((int(prop_key(name)!=prop_key(prop)),tier))
  if ranks:
   return min(ranks)
  return(2,2)if Errand._smm_title_shows(row,value)else(2,3)
 def _smm_pick_context(self,avoid,keep):
  state=self.__dict__
  version=str(self.needs.version)
  failed=set()
  for tag,ruled in self.rulings.items():
   if not ruled:
    continue
   parts=tag.split("|")
   if parts[1:2]==[version]:
    failed.add(parts[0])
  live=set()
  for tag in self.passed:
   head,_,tail=tag.rpartition("|")
   if tail==version:
    live.add(head)
  reviewed=state.get("audit_ids")
  if state.get("audit_done"):
   stale=reviewed is None or len(self.archive.read)<=state.get("audit_reads",-1)
   if stale:
    reviewed=set(self.archive.read)
  audits=[e for e in self.log if isinstance(e,dict)and e.get("kind")=="audit"]
  audited=set()
  if audits and isinstance(audits[-1].get("result"),dict):
   audited=set(reviewed or())-set(state.get("audit_fits")or())-live
  spend=self.needs.spend_cap
  return{"failed":failed,"live":live,"audited":audited,
"own_all":bool(audited)and self.code_judges_all(),
"cap":float(spend)if is_number(spend)else None,
"asked":kind_words(self.request),"hits":state.get("code_hits")or{},
"cheapest":self.needs.priority()=="cheapest",
"avoid":set(avoid or())|self.dead_lines()|self.ref_pointers(),
"keep":set(keep or())|set(self.pair_best())}
 @staticmethod
 def _smm_apart_ok(fields,apart):
  for prop,bad in apart:
   ok=False
   for name,text in fields.items():
    if same_prop(name,prop)and str(text).strip()and all(not ch_writes(v,text)for v in bad):
     ok=True
     break
   if not ok:
    return False
  return True
 def _smm_product_pool(self,ctx,info,k,home,qty):
  listed=self.needs.items()
  slot=k if listed else None
  want=kind_words(listed[k]["product"]if listed else(self.needs.reading.get("product")or""))
  vals=set()
  judged=[]
  for prop,plain in self.code_values(slot):
   vals.update(kind_words(plain))
   if not Errand._smm_BOUND.search(plain.lower()):
    judged.append((prop,plain))
  apart=[]
  for ref in self.differ_done(slot):
   prop=ref["prop"]
   options=[ref.get("value"),*(ref.get("goes_with")or[])]
   apart.append((prop,[str(v)for v in options if str(v or"").strip()]))
  cap,hits,read=ctx["cap"],ctx["hits"],self.archive.read
  ranked=[]
  for ident in info:
   row=info[ident]
   if ident in ctx["failed"]or ident in self.refused_lines or ident in ctx["avoid"]or row["stock"]is False:
    continue
   on_shelf=row["shelf"]in home if home else bool(want&row["words"])
   if not on_shelf or any(self.code_clash(row["fields"],p,v)for p,v in judged)or self.ref_excluded(row["fields"],slot):
    continue
   cost=self.archive.price(ident)if row["price"]is None else row["price"]
   priced=is_number(cost)
   if priced and cap is not None and qty*cost>cap:
    continue
   seen=hits.get(ident)or set()
   was_read=ident in read
   fits=all(self.code_shows(row,p,v,seen,was_read)for p,v in judged)and self._smm_apart_ok(row["fields"],apart)
   shows=[self._smm_value_rank(row,p,v)for p,v in judged]if was_read else[]
   if ident in ctx["audited"]and not(ctx["own_all"]and fits and(judged or apart)and all(max(s)<=1 for s in shows)):
    continue
   words=row["words"]
   score=3*len(vals&words)+2*len(want&words)+len(ctx["asked"]&words)+4*len(vals&seen)
   price=cost if priced else self._smm_INF
   quality=tuple(map(sum,zip(*shows)))if shows else(9,9)
   ranked.append((int(ident not in ctx["keep"]),int(ident not in ctx["live"]),not fits,cap is not None and not priced,
price if ctx["cheapest"]else 0.0,int(not was_read),row["stock"]is not True,
(quality,-score),price,ident))
  ranked.sort()
  return ranked
 _smm_PICK_STATES={"preferred":(None,"cheapest"),
"cheapest":(lambda e:(e[2],e[3],e[8],e[-1]),"")}
 def code_picks(self,avoid=None,keep=None):
  several=self.needs.items()
  per_product=self.needs.counts()
  if several:
   homes=self.home_shelves()
  else:
   labels=set(self.shelf_labels())
   homes=[labels or(self.home_shelves()or[set()])[0]]
  info=self.code_rows()
  ctx=self._smm_pick_context(avoid,keep)
  units=[per_product[k]if k<len(per_product)else 1 for k in range(len(homes))]
  pools=[]
  for k,home in enumerate(homes):
   pool=self._smm_product_pool(ctx,info,k,home,units[k])
   if not pool:
    return[]
   pools.append(pool)
  state="preferred"
  while state:
   order,after=self._smm_PICK_STATES[state]
   chosen=self._smm_choose(pools,order)
   if chosen is None:
    return[]
   spent=0.0
   for entry,qty in zip(chosen,units):
    spent+=(0.0 if entry[8]==self._smm_INF else entry[8])*qty
   if ctx["cap"]is None or spent<=ctx["cap"]:
    self.code_fit={entry[-1]:not entry[2]for entry in chosen}
    return list(zip((entry[-1]for entry in chosen),units))
   state=after
  return[]
 def _smm_choose(self,pools,order):
  taken=[]
  for pool in pools:
   used=[entry[-1]for entry in taken]
   entry=None
   for cand in(pool if order is None else sorted(pool,key=order)):
    ident=cand[-1]
    if ident in used or(ident in self.archive.read and self.field_view(ident)["fails"]):
     continue
    entry=cand
    break
   if entry is None:
    return None
   taken.append(entry)
  return taken
 @staticmethod
 def code_clash(fields,prop,value):
  fielded=Errand._smm_prop_fields(fields,prop)
  if not fielded:
   return False
  return all(not value_seen(value,text)for text in fielded)
 @staticmethod
 def _smm_split_words(v):
  return set(re.findall(r"[a-z]+",Errand._smm_decamel(str(v)).lower()))
 def _smm_cart_schema(self):
  form,args=self._smm_schema(self.role("cart"))
  first=json_object(self.trolley.put_form)if self.trolley.put_form else None
  put,drop={},{}
  for key in form.get("required")or[]:
   choices=Errand._smm_choices(args.get(key))
   if not isinstance(choices,list)or not choices:
    continue
   spelled=[self._smm_split_words(v)for v in choices]
   if isinstance(first,dict)and first.get(key)in choices:
    put[key]=first[key]
   else:
    put[key]=next((v for v,w in zip(choices,spelled)if w&PUT_WORDS and not w&DROP_WORDS),choices[0])
   take=[v for v,w in zip(choices,spelled)if w&DROP_WORDS and v!=put[key]]
   if take:
    drop[key]=take[0]
   elif len(choices)==2:
    drop[key]=[v for v in choices if v!=put[key]][0]
   else:
    drop[key]=None
  free=self._smm_plain_args(args,"text")
  named=[key for key in free if re.search(r"id|item|product|listing|sku|target|which|number",key.lower())]
  return{"schema":form,"args":args,"put":put,"drop":drop,"id":(named or free or[None])[0],
"units":self._smm_plain_args(args,"number")}
 @staticmethod
 def _smm_checked(calls,schema):
  fixed_calls=[]
  for call_args in calls:
   result=ArgCheck.execute(call_args,schema)
   if result[1]:
    return None
   fixed_calls.append(result[0])
  return fixed_calls
 def cart_args(self,picks):
  cart=self._smm_cart_schema()
  layout=self.trolley.line_spec()
  if layout:
   arr,id_field,count_field=layout
   calls=[{**cart["put"],arr:[{id_field:ident,count_field:qty}for ident,qty in picks]}]
  elif not cart["id"]:
   return[]
  else:
   calls=[]
   for ident,qty in picks:
    one=dict(cart["put"])
    one[cart["id"]]=ident
    if cart["units"]:
     one[cart["units"][0]]=qty
    calls.append(one)
  return self._smm_checked(calls,cart["schema"])or[]
 def cart_drops(self,lines):
  if not lines:
   return[]
  if self.trolley.line_spec():
   return[]
  cart=self._smm_cart_schema()
  id_key,units=cart["id"],cart["units"]
  if not id_key:
   return None
  if cart["put"]:
   if any(cart["drop"].get(key)is None for key in cart["put"]):
    return None
   zeroing=False
  else:
   floor=(cart["args"].get(units[0])or{}).get("minimum")if units else None
   if not units or(is_number(floor)and not cart["args"][units[0]]["minimum"]<=0):
    return None
   zeroing=True
  calls=[]
  for ident in lines:
   if zeroing:
    calls.append({id_key:ident,units[0]:0})
    continue
   one=dict(cart["drop"])
   one[id_key]=ident
   if units:
    one[units[0]]=max(1,int(self.trolley.held(ident)or 1))
   calls.append(one)
  return self._smm_checked(calls,cart["schema"])
 def code_replies(self,n):
  newest=next(filter(lambda e:isinstance(e,dict)and e.get("kind")=="env",reversed(self.log)),{})
  resp=newest.get("reply")
  if isinstance(resp,dict):
   many=resp.get("observations")
   return list(many if isinstance(many,list)else[resp.get("observation")])[:n]
  return[]
 @staticmethod
 def code_fault(obs):
  hits=[0]
  def note(d):
   for name,val in d.items():
    if val in Errand._smm_BLANK:
     continue
    if re.search(FAULT_KEYS,str(name).lower()):
     hits[0]+=1
     return
  walk_dicts(obs,note,0,2)
  return hits[0]>0
 _smn_NEXT_KEYS=r"next|onward|cursor|bookmark|token|continu|more"
 _smn_COUNT_KEYS=r"count|total|matched|found|hits"
 _smn_BLANK_CUES=(None,False,"",0)
 _smn_SOFT_ERRORS=("invalid_args","unknown_tool")
 _smn_PAGE_KEYS=r"page|offset|first|start|skip|from|cursor"
 _smn_NOT_PAGE_KEYS=r"size|per_?page|limit|price|cost"
 _smn_EDGE_PUNCT="()[]{}\"'.,;:!?/|"
 _smn_SHELF_KEYS=r"categor|shelf|department|aisle"
 _smn_GONE_WORDS=r"\b(?:out|sold|none|unavailable|not|no|false)\b"
 _smn_LATE_WORDS=r"back.?order|pre.?order|discontinu"
 _smn_SHOP_ROUNDS=4
 def _smn_book(self,kind,**fields):
  self.log.append({"kind":kind,**fields})
 def _smn_trip(self,kind,err):
  self._smn_book(kind,error=f"{type(err).__name__}: {err}"[:300])
 def _smn_burst(self,stem,tool,numbered):
  tick=self.wire.tick
  self.send_chunk([(f"{stem}_{i}_{tick}",tool,params)for i,params in numbered])
 def code_search(self,products,more=False):
  tool=self.role("search")
  run={"tool":tool,"free":self._smn_free_keys(),"hold":2+2*products,"n":products,"round":0}
  (self._smn_plan_more if more else self._smn_plan_fresh)(run)
  if run["calls"]:
   self.code_searched=True
   stage="send"
   while stage:
    stage=getattr(self,"_smn_search_"+stage)(run)
 def _smn_search_schema(self):
  return self.shapes.get(self.role("search")or"")or{}
 def _smn_free_keys(self):
  fields=self._smn_search_schema().get("properties")or{}
  return[name for name,kind in fields.items()
if isinstance(kind,dict)and kind.get("type")=="string"and not kind.get("enum")]
 def _smn_plan_more(self,run):
  kept=list(self.__dict__.get("code_more")or[])
  room=self.moves_left()-run["hold"]-2*run["n"]
  run.update(lead=kept,calls=kept[:max(0,room)],rounds=1)
 def _smn_plan_fresh(self,run):
  queries=[self.code_queries(k if self.needs.items()else None)for k in range(run["n"])]
  order=[]
  for depth in range(max(map(len,queries),default=0)):
   tier=[q[depth]for q in queries if len(q)>depth]
   order+=[query for query in tier if query not in order]
  room=self.moves_left()-run["hold"]
  cheap=self.needs.priority()=="cheapest"
  solo=self.deep_pages()
  run.update(calls=order[:max(0,room)],lead=[q[0]for q in queries if q],rounds=1+max(CODE_PAGES+2*cheap,solo))
 def deep_pages(self):
  count=len(self.needs.items())
  return SOLO_PAGES if not count else FEW_PAGES if count<=FEW_MAX else 0
 def _smn_search_send(self,run):
  self._smn_burst("code_search",run["tool"],list(enumerate(run["calls"])))
  run["replies"]=self.code_replies(len(run["calls"]))
  return"book"
 def _smn_search_book(self,run):
  own=self.__dict__
  if own.get("word_kind")!="any":
   tags=own.setdefault("code_hits",{})
   for params,reply in zip(run["calls"],run["replies"]):
    said=" ".join(map(str,(params.get(name)or""for name in run["free"])))
    if said.strip():
     self._smn_tag_rows(reply,kind_words(said),tags)
  return"page"
 def _smn_tag_rows(self,reply,words,tags):
  known=self.archive.ids
  def tag(row):
   pid=self.row_mark(row)
   if pid in known:
    tags.setdefault(pid,set()).update(words)
  walk_dicts(reply,tag)
 def _smn_search_page(self,run):
  nexts=(self.code_next(params,reply)for params,reply in zip(run["calls"],run["replies"])
if params in run["lead"])
  run["ahead"]=self.code_more=[call for call in nexts if call]
  run["round"]+=1
  return"judge"if run["round"]<run["rounds"]else""
 def _smn_search_judge(self,run):
  plan=self.code_picks()
  fits=self.__dict__.get("code_fit")or{}
  settled=run["round"]>self.deep_pages()
  if plan and self.needs.priority()!="cheapest"and all(fits.get(pid)for pid,_ in plan)and settled:
   return""
  room=self.moves_left()-run["hold"]-3*run["n"]
  run["lead"],run["calls"]=run["ahead"],run["ahead"][:max(0,room)]
  return"send"if run["calls"]else""
 def code_next(self,params,reply):
  paging=self.page_arg()
  if not paging:
   return None
  field,per_page,floor,ceiling=paging
  page=self._smn_result_page(reply)
  if page is None:
   return None
  sizes,cues,total=[],[],None
  for name,value in page.items():
   label=str(name).lower()
   if isinstance(value,list):
    sizes.append(len(value))
   if re.search(self._smn_NEXT_KEYS,label):
    cues.append(value)
   if total is None and is_number(value)and re.search(self._smn_COUNT_KEYS,label):
    total=value
  if cues and all(cue in self._smn_BLANK_CUES for cue in cues):
   return None
  if per_page=="token":
   follow=self._smn_token_follow(params,field,cues)
  else:
   follow=self._smn_count_follow(params,field,per_page,floor,ceiling,max(sizes),total,bool(cues))
  if follow is not None:
   fixed,faults=ArgCheck.execute(follow,self._smn_search_schema())
   if not faults:
    return fixed
  return None
 @staticmethod
 def _smn_result_page(reply):
  found=[]
  def probe(d):
   if not found and any(map(Errand._smn_is_rows,d.values())):
    found.append(d)
  walk_dicts(reply,probe,0,3)
  return found[0]if found else None
 @staticmethod
 def _smn_is_rows(value):
  return isinstance(value,list)and bool(value)and all(isinstance(x,dict)for x in value)
 @staticmethod
 def _smn_token_follow(params,field,cues):
  token=next((cue for cue in cues if isinstance(cue,str)and cue.strip()),None)
  return{**params,field:token}if token else None
 @staticmethod
 def _smn_count_follow(params,field,per_page,floor,ceiling,size,
total,cued):
  at=params.get(field,floor)
  at=at if is_number(at)else floor
  ahead=at+(1 if per_page else size)
  if ceiling is not None and ahead>ceiling:
   return None
  if not cued and(total is None or((at-floor+1)*size if per_page else at+size)>=total):
   return None
  return{**params,field:int(ahead)if float(ahead).is_integer()else ahead}
 def code_read(self,plan,barred,prefer):
  rounds=0
  while rounds<CODE_READS:
   rounds+=1
   slab=self._smn_read_slab(plan)
   if not slab:
    break
   self.send_chunk(slab)
   fresh=self.code_picks(barred,prefer)
   if fresh==plan:
    break
   plan=fresh
  return plan
 def _smn_read_slab(self,plan):
  opened=self.archive.read
  todo=[pid for pid,_qty in plan if pid not in opened]
  lister=self.batch_tool()
  joint=bool(lister)and len(todo)>1
  reserve=3+2*len(plan)+(1 if joint else len(todo))
  if not todo or not(lister or self.role("details"))or reserve>self.moves_left()or self.wire.spare()<=8:
   return[]
  stamp=self.wire.tick
  if joint:
   tool,key,most=lister[0],lister[1],lister[2]
   return[(f"code_read_{stamp}",tool,{key:todo[:most]})]
  opener=self.role("details")
  made=map(self.details_call,todo)
  return[(f"code_read_{n}_{stamp}",opener,call)for n,call in enumerate(made)if call]
 def code_put(self,plan):
  wanted={pid for pid,_ in plan}
  cart=self.trolley
  whole=cart.line_spec()
  others=[line for line in cart.units if line not in wanted]
  if not self._smn_swap_cleared(others,plan):
   return None
  calls=self._smn_put_calls(plan,others,whole)
  if calls is None or(calls and self.moves_left()<len(calls)+1):
   return None
  if not calls:
   return[]
  self._smn_burst("code_cart",self.role("cart"),list(enumerate(calls)))
  replies=self.code_replies(len(calls))
  bad,shown=self._smn_cart_faults(calls,replies,wanted,whole)
  if len(replies)<len(calls):
   return None
  return self._smn_settle(bad,shown if shown is not None or bad else self._smn_cart_look(bad,wanted),wanted)
 def _smn_settle(self,bad,shown,wanted):
  if shown is not None:
   bad|={pid for pid in wanted if pid not in shown}
  units=self.trolley.units
  for pid in bad:
   units.pop(pid,None)
  return sorted(bad)
 def _smn_swap_cleared(self,others,plan):
  ledger=self.needs.told
  stated=any(entry.get("before_change")is True for entry in ledger)
  unknown=any(entry.get("before_change")is None for entry in ledger)
  if others and(stated or unknown):
   return self.code_asks_swap(others,[pid for pid,_ in plan])or not stated
  return True
 def _smn_put_calls(self,plan,others,whole):
  cart=self.trolley
  todo=list(plan)if whole else[(pid,n)for pid,n in plan if abs(cart.held(pid)-n)>1e-9]
  drops,puts=self.cart_drops(others),(self.cart_args(todo)if todo else[])
  return None if drops is None or(todo and not puts)else drops+puts
 def _smn_cart_faults(self,calls,replies,wanted,whole):
  cart=self.trolley
  blanket=list(wanted)if whole else[]
  bad=set()
  shown=None
  for params,reply in zip(calls,replies):
   view=cart.view_units(reply)
   if view is not None:
    shown=view
   if not self.code_fault(reply):
    continue
   text=as_text(reply)
   named=[pid for pid in wanted if pid in text]
   carried=[line for line,_n in cart.lines(params)if line in wanted]
   bad.update(named or carried or blanket)
  return bad,shown
 def _smn_cart_look(self,bad,wanted):
  if not self.role("cart_view")or self.moves_left()<3:
   return None
  self.send_chunk([(f"code_view_{self.wire.tick}",self.role("cart_view"),{})])
  got=self.code_replies(1)
  shown=self.trolley.view_units(got[0])if got else None
  if got and shown is None:
   bad.update(pid for pid in wanted if pid not in self.trolley.units)
  return shown
 def code_recheck(self,held):
  tool=self.stock_tool()or next((name for name in map(self.role,("view","details"))if name),None)
  miss=(False,set())
  if not(held and tool)or len(held)+2>self.moves_left()or self.wire.spare()<8:
   return miss
  numbered=[(i,params)for i,params in enumerate(self.lookup_call(pid,tool)for pid in held)if params]
  if numbered:
   self._smn_burst("code_check",tool,numbered)
   stock=self.code_rows()
   return True,{pid for pid in held if(stock.get(pid)or{}).get("stock")is False}
  return miss
 def code_news(self,fresh):
  backlog=[line for batch,_ in self.needs.__dict__.get("pending")or[]for line in batch]
  done=self.__dict__.setdefault("news_read",set())
  for line in(*fresh,*backlog):
   if line in done:
    continue
   done.add(line)
   amount,_=self.code_budget(line)
   if is_number(amount)and amount>0:
    self.needs.spend_cap=amount
    self._smn_book("code_news",budget=amount)
 def code_cart(self,avoid=None,keep=None):
  search=self.role("search")
  if not self.role("cart")or self._smn_blocked(2,4):
   return False
  self.wire.mute=True
  try:
   run={"barred":set(avoid or()),"prefer":keep,"search":search,"round":0,
"n":max(1,len(self.needs.items()))}
   stage="look"
   while isinstance(stage,str):
    stage=getattr(self,"_smn_shop_"+stage)(run)
   return stage
  except Exception as err:
   self._smn_trip("code_finish_error",err)
   return False
 def _smn_shop_look(self,run):
  own,n=self.__dict__,run["n"]
  plan=self.code_picks(run["barred"],run["prefer"])
  unfit=not plan or not all(own.get("code_fit",{}).get(pid)for pid,_qty in plan)
  if run["search"]:
   steps=self.moves_left()
   first_go=not own.get("code_searched")and steps>=4 and unfit
   page_on=not first_go and bool(own.get("code_more"))and steps>=6+2*n and(unfit or self.needs.priority()=="cheapest")
   if first_go or page_on:
    self.code_search(n,page_on)
    plan=self.code_picks(run["barred"],run["prefer"])
  run["plan"]=plan
  return"read"
 def _smn_shop_read(self,run):
  run["plan"]=self.code_read(run["plan"],run["barred"],run["prefer"])
  if run["plan"]:
   return"put"
  self._smn_book("code_finish",picks=[],result="nothing to order")
  return False
 def _smn_shop_put(self,run):
  run["bad"]=self.code_put(run["plan"])
  if run["bad"]is None:
   self._smn_book("code_finish",picks=run["plan"],result="no cart call")
   return False
  return"swap"if run["bad"]else"check"
 def _smn_shop_check(self,run):
  plan=run["plan"]
  ids=[pid for pid,_qty in plan]
  sold=self.code_recheck(ids)[1]if self.moves_left()>=len(plan)+3 else set()
  run["bad"]=sorted(sold)
  if run["bad"]:
   return"swap"
  units=self.trolley.units
  self._smn_book("code_finish",picks=plan,cart=dict(units))
  return bool(units)
 def _smn_shop_swap(self,run):
  run["barred"]|=set(run["bad"])
  self._smn_book("code_swap",lines=run["bad"])
  run["round"]+=1
  return"look"if run["round"]<self._smn_SHOP_ROUNDS else False
 def finish_now(self):
  if self._smn_cut_off():
   return
  run={"order":self.role("order"),"decline":self.role("decline"),"barred":set(),
"prefer":None,"move":None,"kind":None,"resp":None,"turn":0}
  stage="gate"
  while stage:
   stage=getattr(self,"_smn_fin_"+stage)(run)
 def _smn_blocked(self,steps,seconds):
  wire=self.wire
  return wire.done or wire.broke or self.moves_left()<steps or wire.spare()<seconds
 def _smn_cut_off(self):
  return self._smn_blocked(1,2)
 def _smn_order_move(self,tool):
  return{"name":tool,"args":self.plain_args(tool,"Order the cart.","order")}
 def _smn_decline_move(self,tool):
  return{"name":tool,"args":self.plain_args(tool,SORRY,"decline")}
 def _smn_fin_gate(self,run):
  if run["turn"]>=CODE_FINISHES:
   return"close"
  if self._smn_cut_off():
   return""
  return"pick"if run["move"]is None else"steps"
 def _smn_fin_pick(self,run):
  orderer,first=run["order"],run["turn"]==0
  cart=self.trolley
  gone=self._smn_fin_dead(run)if orderer and first and cart.units else set()
  rules=(

("order",lambda:orderer and first and not gone and cart.used and self._smn_covers_all()
and(cart.units or cart.view_units(cart.last_view)is None)and not self.cart_misfits()),

("order",lambda:orderer and self.code_cart(run["barred"],run["prefer"])),

("order",lambda:orderer and not first and self.again_ok(run["barred"])),
("decline",lambda:run["decline"]),
)
  if first:
   run["barred"]|={m.group(1)for m in map(re.compile(r"(?:line )?(\S+?):? ").match,self.cart_misfits())
if m and m.group(1)in cart.units}
  choice=next((what for what,holds in rules if holds()),None)
  if choice is None:
   return""
  run["move"]=self._smn_order_move(orderer)if choice=="order"else self._smn_decline_move(run["decline"])
  return"steps"
 def _smn_covers_all(self):
  rows=self.needs.items()
  if not rows:
   return True
  slots={self.product_index(line)for line in self.trolley.units}
  return all(k in slots for k in range(len(rows)))
 def _smn_fin_dead(self,run):
  try:
   self.wire.mute=True
   held=list(self.trolley.units)
   gone=self.cart_gone(held)
   if gone:
    run["barred"]|=gone
    run["prefer"]=set(held)-gone
    self._smn_book("code_dead",lines=sorted(gone))
   return gone
  except Exception as err:
   self._smn_trip("cart_gone_error",err)
   return set()
 def _smn_fin_steps(self,run):
  move=run["move"]
  run["kind"]=self.call_kind(move["name"],move["args"])
  try:
   self.finish_steps(run["kind"]or"order")
  except Exception as err:
   self._smn_trip("finish_steps_error",err)
  wire=self.wire
  return""if wire.done or wire.broke or self.moves_left()<=0 else"send"
 def _smn_fin_send(self,run):
  answer=run["resp"]=self.wire.act([run["move"]])
  self._smn_book("forced_finish",action=run["move"],reply=answer)
  last_turn=run["turn"]==CODE_FINISHES-1
  if not answer or answer.get("finished")or self.wire.done or last_turn or self.moves_left()<=1:
   return"close"
  return"refused"
 def _smn_fin_refused(self,run):
  move,answer=run["move"],run["resp"]
  self.wire.mute=True
  held=list(self.trolley.units)
  seen=answer.get("observation")
  self.absorb(move["name"],move["args"],seen,answer,True)
  self.code_news(self.shopper_lines(move["name"],seen,True))
  run["move"]=None
  if run["kind"]=="order"and held:
   self._smn_fin_repair(run,held)
  run["turn"]+=1
  return"gate"
 def _smn_fin_repair(self,run,held):
  looked,gone=self.code_recheck(held)
  if not looked:
   return
  self.refused_lines.difference_update(set(held)-gone)
  rows,cart=self.code_rows(),self.trolley
  unit_price={line:(rows.get(line)or{}).get("price")or 0.0 for line in held}
  cost=sum(unit_price[line]*cart.held(line)for line in held)
  cap=self.needs.spend_cap
  over=is_number(cap)and cost>cap
  if not(gone or over):
   run["move"]=self._smn_order_move(run["order"])
   return
  run["barred"]=set(gone)
  run["prefer"]=None if over else set(held)-gone
 def _smn_fin_close(self,run):
  answer,kind=run["resp"],run["kind"]
  still_open=not self.wire.done and answer and not answer.get("finished")and run["decline"]and kind!="decline"and self.moves_left()>0
  if not still_open:
   return""
  try:
   retry=bool(run["order"])and kind=="order"and self.again_ok(run["barred"])
  except Exception as err:
   self._smn_trip("again_error",err)
   retry=False
  move=self._smn_order_move(run["order"])if retry else self._smn_decline_move(run["decline"])
  reply=self.wire.act([move])
  self._smn_book("forced_finish",action=move,reply=reply)
  return""
 def again_ok(self,avoid):
  held=set(self.trolley.units)
  shut=set(avoid)|self.dead_lines()
  return bool(held)and held.isdisjoint(shut)and bool(self.code_picks(shut))
 def catch_up(self):
  try:
   if self.wire.mute or self.wire.spare()<CLOSING:
    return
   notes=[note for step in(self._smn_redo_reading,self._smn_hear_backlog)for note in step()]
   if notes:
    self._smn_tell_actor(notes)
  except Exception as err:
   self._smn_trip("catch_up_error",err)
 def _smn_redo_reading(self):
  wire,own,needs=self.wire,self.__dict__,self.needs
  if not needs.reading.get("code_reading")or own.get("reread"):
   return[]
  if wire.last_reply<=own.get("read_at",wire.last_reply)or wire.spare()<=CLOSING+60:
   return[]
  self.reread=True
  answer=wire.query_model([{"role":"user","content":self.read_content}],READER,READ_TOKENS)
  said=(answer or{}).get("content")or""
  parsed=json_object(said)or{}
  usable=isinstance(parsed.get("roles"),dict)
  self._smn_book("re_read",ok=usable)
  if usable:
   self.use_reading(parsed,True)
   return["[fresh plan] Model service is back; it re-parsed the shopper's request. Prefer it to the old one:\n"
+needs.overview(self.step_cap,self.msg_cap)]
  return[]
 def _smn_hear_backlog(self):
  wire=self.wire
  queued=self.needs.__dict__.get("pending")or[]
  if not queued or wire.last_reply<=self.needs.__dict__.get("pending_at",0.0)or wire.spare()<CLOSING:
   return[]
  self.needs.pending=[]
  heard=(self.needs.hear(batch,asked)for batch,asked in queued[-3:])
  return["[read late] lines once sent by shopper: "+said for said in heard if said]
 def _smn_tell_actor(self,notes):
  body="\n".join([*notes,self.status_line()])
  newest=self.chat[-1:]
  tail=newest[0]if newest else None
  if bool(tail)and tail.get("role")in("user","tool")and isinstance(tail.get("content"),str):
   tail["content"]=tail["content"]+"\n"+body
  else:
   self.chat.append({"role":"user","content":body})
 def refused(self,call_name,obs,resp):
  boxed=isinstance(obs,dict)
  if boxed and"refusal"in obs:
   return True
  if call_name!=self.role("order")or resp.get("finished"):
   return False
  return not(boxed and obs.get("error")in self._smn_SOFT_ERRORS)
 def notes_for(self,call_name,obs,refused):
  parts=[]
  if isinstance(obs,dict)and obs.get("error")=="budget_exceeded":
   parts.append(self._smn_note_spent(call_name))
  case=self._smn_note_case(call_name,obs,refused)
  text=getattr(self,"_smn_note_"+case)(obs)if case else""
  return" ".join(parts+[text]if text else parts)
 def _smn_note_case(self,tool,reply,turned):
  if turned:
   redo=tool==self.role("decline")and getattr(self,"last_kind",None)=="decline"
   return"decline"if redo else"order"
  if isinstance(reply,dict)and len(reply)>=2 and tool!=self.role("message"):
   return"notice"
  return""
 def _smn_note_spent(self,tool):
  cap=self.msg_cap
  if tool==self.role("message")or cap is not None:
   self.msg_cap=self.msgs_sent if cap is None else min(cap,self.msgs_sent)
  return"This tool's allowance is spent; place no more calls."
 def _smn_note_decline(self,reply):
  self.audit_done=False
  self.decline_vetted=False
  self.event_seen=True
  if self._smn_long_texts(reply):
   return self._smn_news_hold()
  return("DECLINE TURNED DOWN; the errand goes on. The store holds that a fitting item exists. Next: (a) apply what "
"the refusal says; (b) reopen your closest options, judging each by its own record (every property, stock, "
"price) versus today's requirements; (c) buy whichever passes. Decline again only if every one fails.")
 def _smn_news_hold(self):
  ver=str(self.needs.version)
  struck=sorted({memo.split("|")[0]for memo,said in self.rulings.items()
if said and memo.split("|")[1:2]==[ver]and memo.split("|")[0]+"|"+ver not in self.passed})
  named=(" Item check rejections still standing: %s."%", ".join(struck[:6]))if struck else""
  return("DECLINE PAUSED FOR NEWS: the store passed on a shopper message before closing; the errand goes on. The pause "
"itself says nothing about whether any listing fits. Next: (a) apply the message; (b) re-judge only what it "
"moves: a listing that failed on a condition the message drops or widens, or on a price a raised budget now "
"covers; a listing that failed on any other condition stays out.%s (c) buy one that passes on every "
"condition now in force; (d) when none does, send the decline again."%named)
 def _smn_note_order(self,reply):
  self.refusal_n+=1
  self.order_vetted=False
  self.event_seen=True
  hint=""
  try:
   hint=self.refusal_reread(reply)
  except Exception as err:
   self._smn_trip("refusal_reread_error",err)
  self.refused_lines.update(self.trolley.units)
  self.refusal_texts={" ".join(text.lower().split())for text in self._smn_long_texts(reply)}
  return("ORDER TURNED DOWN - no sale made; work continues on this cart. Steps: (a) carry out each change it lists; "
"(b) mend the cart, dropping any line failing now (one out of stock, say); (c) recheck stock and cost of any "
"new item; (d) send the order once more."+hint)
 @staticmethod
 def _smn_long_texts(reply):
  found=[]
  stack=[(reply,0)]
  while stack:
   piece,depth=stack.pop()
   if depth>6:
    continue
   if isinstance(piece,dict):
    stack.extend((part,depth+1)for part in piece.values())
   elif isinstance(piece,list):
    stack.extend((part,depth+1)for part in piece[:20])
   elif isinstance(piece,str)and len(piece.split())>=4:
    found.append(piece)
  return found
 def _smn_note_notice(self,reply):
  said=self.notice_lines(reply)
  if not said:
   return""
  echoed=self.__dict__.pop("refusal_texts",None)or set()
  flat=[" ".join(str(line).lower().split())for line in said]
  if echoed and all(any(bit in text for text in echoed)for bit in flat):
   return"Only an echo: the refused finish already brought this same message, dealt with above; nothing here concerns this call."
  self.event_seen=True
  return"[shopper message: a change] This result bears news to apply at once: "+" | ".join(said[:4])
 @staticmethod
 def shelf_of(row):
  for name,value in row.items():
   label=str(name).lower()
   if isinstance(value,str)and 0<len(value.strip())<=60 and"id"not in label and re.search(Errand._smn_SHELF_KEYS,label):
    return value.strip().lower()
  return""
 @staticmethod
 def key_stock(field,value):
  label=str(field).lower()
  if value is None or not re.search(r"stock|avail|on.?hand",label):
   return None
  stockish=bool(re.search(r"stock|avail",label))
  inverted=stockish and bool(re.search(r"out|sold",label))
  nil=value is False or(is_number(value)and value<=0)
  words=value.lower()if isinstance(value,str)else None
  signs=(
inverted and value is True,
stockish and not inverted and(nil or(words is not None and bool(re.search(Errand._smn_GONE_WORDS,words)))),
bool(re.search(r"on.?hand",label))and nil,
words is not None and bool(re.search(Errand._smn_LATE_WORDS,words)),
)
  return not any(signs)
 @staticmethod
 def stock_state(row):
  said={Errand.key_stock(name,value)for name,value in row.items()}-{None}
  return(False not in said)if said else None
 @staticmethod
 def sold_out(row):
  return False in{Errand.key_stock(name,value)for name,value in row.items()}
 def shelf_labels(self):
  tally=self._smn_shelf_tally()
  peak=max(tally.values(),default=0)
  common={label for label,seen in tally.items()if seen==peak}
  if not self.needs.items()or not tally:
   return common
  try:
   return self._smn_product_shelves(tally)or common
  except Exception:
   return common
 def _smn_shelf_tally(self):
  found=[]
  done=set()
  foreign=self.ref_foreign()
  opened=self.archive
  def count(rec):
   pid=next((value for name,value in rec.items()
if isinstance(value,str)and"id"in str(name).lower()and value in opened.read),None)
   label=self.shelf_of(rec)if pid and pid not in done and pid not in foreign else""
   if label:
    done.add(pid)
    found.append(label)
  for item in opened.items:
   walk_dicts(item,count)
  return{label:found.count(label)for label in dict.fromkeys(found)}
 def _smn_product_shelves(self,tally):
  homes=self.home_shelves()
  chosen=set()
  for home in homes:
   seen={label:tally[label]for label in home if label in tally}
   if seen:
    best=max(seen.values())
    chosen.update(label for label,hits in seen.items()if hits==best)
   else:
    chosen.update(home)
  if not all(homes):
   ranked=iter(sorted(tally,key=tally.__getitem__,reverse=True))
   while len(chosen)<len(homes):
    label=next(ranked,None)
    if label is None:
     break
    chosen.add(label)
  return chosen
 @staticmethod
 def row_mark(row):
  usable=[(str(name).lower(),value)for name,value in row.items()
if isinstance(value,(str,int))and not isinstance(value,bool)]
  hit=next((value for name,value in usable if name=="id"),None)
  if hit is None:
   hit=next((value for name,value in usable if name.endswith("id")),None)
  return json.dumps(row,sort_keys=True,default=str)if hit is None else str(hit)
 def search_keys(self,params):
  schema=self._smn_search_schema()
  fields=schema.get("properties")if isinstance(schema,dict)else None
  fields=fields if isinstance(fields,dict)else{}
  cursor=self.token_key()
  texts,mode,paging,firsts=[],[],set(),[]
  for name,value in params.items():
   spec=fields.get(name)
   spec=spec if isinstance(spec,dict)else{}
   kind=self._smn_arg_kind(name,value,cursor,spec)
   if kind in("cursor","page"):
    paging.add(name)
    firsts.append(not value.strip()if kind=="cursor"
else value<=self._smn_page_floor(str(name).lower(),spec))
   elif kind=="mode":
    mode.append((name,value))
   elif kind=="text"and(words:=self._smn_plain_words(value)):
    texts.append((name,words))
  return texts,tuple(sorted(mode)),all(firsts),paging
 @staticmethod
 def _smn_arg_kind(name,value,cursor,spec):
  if isinstance(value,str):
   if name==cursor:
    return"cursor"
   return"mode"if spec.get("enum")else"text"
  label=str(name).lower()
  if is_number(value)and re.search(Errand._smn_PAGE_KEYS,label)and not re.search(Errand._smn_NOT_PAGE_KEYS,label):
   return"page"
  return""
 @staticmethod
 def _smn_plain_words(value):
  cleaned=(token.strip(Errand._smn_EDGE_PUNCT)for token in value.lower().split())
  words=tuple(dict.fromkeys(word for word in cleaned if re.search(r"[a-z0-9]",word)))
  return words if words and all(re.fullmatch(r"[a-z0-9]{2,}",word)for word in words)else()
 @staticmethod
 def _smn_page_floor(label,spec):
  low=spec.get("minimum",spec.get("default"))
  if is_number(low):
   return low
  return 1 if"page"in label else 0
 _smo_page_name=r"page|offset|first|start|skip|from|cursor"
 _smo_not_page=r"size|per_?page|limit|price|cost"
 _smo_seller=r"shop|seller|store|vendor|brand|make|manufactur"
 _smo_said_tag=r"\((?:stated|revealed[^)]*)\)"
 _smo_plain=r"[a-z0-9]+"
 _smo_pair2=r"[a-z0-9]{2,}"
 @staticmethod
 def _smo_sig(call):
  return json.dumps(call,sort_keys=True,default=str)
 @staticmethod
 def _smo_words(text,pattern=r"[a-z0-9]+"):
  return re.findall(pattern,text.lower())
 @staticmethod
 def _smo_named(record,pattern):
  return[(name,record[name])for name in record if re.search(pattern,str(name).lower())]
 @staticmethod
 def _smo_without(call,drop):
  return{name:call[name]for name in call if name!=drop}
 @staticmethod
 def _smo_near(a,b,figures=""):
  if figures and any(ch.isdigit()for ch in a+b):
   if a.lstrip("0")==b.lstrip("0"):
    return True
   return figures=="lead"and a.isdigit()and b.startswith(a)and not b[len(a):][:1].isdigit()
  short,longer=sorted((a,b),key=len)
  return len(short)>=3 and longer.startswith(short)
 def _smo_book(self):
  return self.__dict__.get("page_book",{})
 def _smo_query_of(self,call):
  return" ".join(self.free_text(call))
 def _smo_search_props(self):
  tool=self.role("search")or""
  props=(self.shapes.get(tool)or{}).get("properties")
  return props if isinstance(props,dict)else{}
 def _smo_live_reqs(self,need_colon):
  out=[]
  for req in self.needs.props_left():
   name,colon,value=req.partition(":")
   if(colon or not need_colon)and not ref_text(value):
    out.append((name,value))
  return out
 def _smo_first_look(self,call_args,obs,rows):
  texts,mode,first,paging=self.search_keys(call_args)
  if not first:
   return None
  counts=[n for n in obs.values()if isinstance(n,int)and not isinstance(n,bool)]
  counts.append(len(rows))
  return texts,mode,paging,max(counts)
 @staticmethod
 def _smo_frame(call_args,text_key,mode,paging):
  rest=sorted((name,str(call_args[name]))for name in call_args if name!=text_key and name not in paging)
  return(text_key,mode),tuple(rest)
 @staticmethod
 def _smo_shared_lead(a,b):
  n=0
  for x,y in zip(a,b):
   if x!=y:
    break
   n+=1
  return n
 def _smo_pair_says(self,mine,mine_total,theirs,their_total):
  (short,short_total),(longer,long_total)=sorted(((mine,mine_total),(theirs,their_total)),key=lambda p:len(p[0]))
  if len(longer)>len(short)and longer[:len(short)]==short:
   return"any"if long_total>short_total else"all"
  reach=self.lead_words or None
  for(big,big_total),(small,small_total)in(((mine,mine_total),(theirs,their_total)),
((theirs,their_total),(mine,mine_total))):
   if set(map(self.stem,big[:reach]))>set(map(self.stem,small[:reach])):
    return"subset"if big_total<small_total else""
  return""
 def word_rule(self,call_args,obs,rows):
  look=self._smo_first_look(call_args,obs,rows)
  if look is None:
   return self.word_kind
  texts,mode,paging,total=look
  for text_key,words in texts:
   here=self._smo_frame(call_args,text_key,mode,paging)
   narrower=False
   for entry in self.word_seen:
    if entry[:2]!=here or entry[3]==total:
     continue
    says=self._smo_pair_says(words,total,entry[2],entry[3])
    if says=="subset":
     narrower=True
    elif says:
     self.word_kind=says
   if narrower and not self.word_kind:
    self.word_kind="all"
   self.word_seen.append(here+(words,total))
  return self.word_kind
 def lead_rule(self,call_args,obs,rows):
  look=self._smo_first_look(call_args,obs,rows)
  if look is None:
   return self.lead_words
  texts,mode,paging,total=look
  seen_rows=tuple(self.row_mark(r)for r in rows)
  varied=len(set(seen_rows))>=3
  for text_key,words in texts:
   here=self._smo_frame(call_args,text_key,mode,paging)
   bound=self.lead_bounds.setdefault(here[0],[-1,set()])
   for entry in self.lead_seen:
    if entry[:2]!=here or entry[2]==words:
     continue
    shared=self._smo_shared_lead(words,entry[2])
    if(total,seen_rows)!=entry[3:]:
     bound[0]=max(bound[0],shared)
    elif shared and varied and shared==min(len(words),len(entry[2])):
     bound[1].add(shared)
   self.lead_seen.append(here+(words,total,seen_rows))
  ready=[(low+1,frame)for frame,(low,same)in self.lead_bounds.items()if low+1 in same]
  self.lead_words,self.lead_ctx=min(ready)if ready else(0,None)
  return self.lead_words
 def token_key(self,props=None):
  props=self._smo_search_props()if props is None else props
  for name in props:
   spec=props[name]
   if not isinstance(spec,dict)or spec.get("type")!="string"or spec.get("enum"):
    continue
   if re.search(r"bookmark|cursor|token|resume|continu|after|next|page|offset|start",str(name).lower()):
    return name
  return None
 @staticmethod
 def next_token(obs):
  if isinstance(obs,dict):
   for name in obs:
    found=obs[name]
    if isinstance(found,str)and found.strip()and re.search(r"next|onward|cursor|bookmark|token|continu|more",str(name).lower()):
     return found
  return None
 def page_arg(self):
  props=self._smo_search_props()
  for name in props:
   spec,low=props[name],str(name).lower()
   if not(isinstance(spec,dict)and spec.get("type")in("integer","number")):
    continue
   if re.search(self._smo_not_page,low)or not re.search(self._smo_page_name,low):
    continue
   whole="page"in low and not re.search(r"offset|skip|first|start|from",low)
   start=spec.get("minimum",spec.get("default"))
   top=spec.get("maximum")
   return(name,whole,start if is_number(start)else int(whole),
top if is_number(top)else None)
  tok=self.token_key()
  return(tok,"token","",None)if tok else None
 @staticmethod
 def stem(word):
  plural=len(word)>3 and word[-1:]=="s"and word[-2:]!="ss"
  return word[:-1]if plural else word
 @staticmethod
 def _smo_short_text(value):
  return isinstance(value,str)and 0<len(value)<=300
 def _smo_row_title(self,record):
  own=record.get("title")
  if self._smo_short_text(own):
   return own
  for name,value in self._smo_named(record,r"title|heading|headline|name"):
   if self._smo_short_text(value)and"id"not in str(name).lower()and not self.shelf_of({name:value})and not junk_key(name):
    return value
  return""
 def _smo_record_id(self,record):
  for name in record:
   value=record[name]
   if isinstance(value,str)and"id"in str(name).lower()and value in self.archive.ids:
    return value
  return None
 def _smo_shelf_seen(self,record,shelves,seen):
  ident=self._smo_record_id(record)
  if not ident or ident in self.refused_lines or self.shelf_of(record)not in shelves:
   return
  if self.sold_out(record):
   seen.pop(ident,None)
   return
  prices=[v for _,v in self._smo_named(record,r"price|cost")if is_number(v)and not isinstance(v,bool)]
  title=self._smo_row_title(record)
  brand=" ".join(v for _,v in self._smo_named(record,r"brand|make|manufactur|vendor")if isinstance(v,str))
  if prices:
   seen[ident]=(float(prices[0]),title,brand)
 def shelf_rows(self):
  shelves=self.shelf_labels()
  cap=self.needs.spend_cap
  qty=self.needs.quantity()
  seen={}
  if shelves:
   for rec in self.archive.items:
    walk_dicts(rec,lambda d:self._smo_shelf_seen(d,shelves,seen))
  kept={}
  for ident,row in seen.items():
   newest=self.archive.price(ident)
   price=row[0]if newest is None else newest
   if not(is_number(cap)and price*qty>cap):
    kept[ident]=(float(price),)+row[1:]
  return kept
 def label_queries(self):
  out=[]
  for label in sorted(self.shelf_labels()):
   runs=[[]]
   for piece in label.lower().split():
    word=piece.strip(".,;:!?'\"()[]")
    plain=word not in LINK_WORDS and re.fullmatch(r"[a-z0-9][a-z0-9'-]*",word)
    if plain:
     runs[-1].append(word)
    elif runs[-1]:
     runs.append([])
   runs=[run for run in runs if run]
   flat=[w for run in runs for w in run]
   tails=[run[-1]for run in runs]
   out.extend(map(self.stem,tails))
   if len(flat)>1:
    out.append(flat[0]+" "+flat[1])
   out.extend(tails)
   out.extend(w for w in flat if w not in tails)
  return list(dict.fromkeys(out))
 def kind_queries(self,pairs):
  label={self.stem(w)for s in self.shelf_labels()for w in re.findall(self._smo_plain,s)}
  by_title={}
  for price,title,maker in self.shelf_rows().values():
   low=title.lower().strip()
   if low:
    before=by_title.get(low)
    by_title[low]=(price if before is None else min(price,before[0]),maker.lower())
  width=2 if pairs else 1
  tally={}
  for title,(price,maker)in by_title.items():
   skip=set(re.findall(r"[a-z]+",maker))
   words=re.findall(r"[a-z]+",title)
   ok=[len(w)>=3 and w not in LINK_WORDS and w not in skip for w in words]
   counted=set()
   for at in range(len(words)-width+1):
    gram=words[at:at+width]
    if not all(ok[at:at+width]):
     continue
    stems=[self.stem(w)for w in gram]
    joined=" ".join(stems)
    if all(s in label for s in stems)or joined in counted:
     continue
    counted.add(joined)
    entry=tally.setdefault(joined,[0,price," ".join(gram)])
    entry[0]+=1
    entry[1]=min(entry[1],price)
  common=[joined for joined in tally if tally[joined][0]>=2]
  common.sort(key=lambda j:(tally[j][1],-tally[j][0],j))
  return[tally[j][2]for j in common]
 def _smo_two_stems(self,text):
  return tuple(map(self.stem,self._smo_words(text,self._smo_pair2)[:2]))
 def _smo_sweep_plan(self,turn,key,start,shelves,book):
  plan={"turn":turn,"key":key,"shelves":shelves,"book":book,"tried":self.__dict__.get("sweep_tried",set())}
  walks=list(book.values())
  open_lists=[w for w in walks if w.get("id_order")and w["shelves"]and w["shelves"]<=shelves
and self.next_page(w)is not None]
  plan["listing"]=max(open_lists,key=lambda w:w["total"],default=None)
  ranked=[w for w in walks if w.get("id_order")is False and self.free_text(w["args"])]or[w for w in walks if w.get("id_order")is None and self.free_text(w["args"])]
  plan["template"]=plan["text_key"]=None
  if ranked:
   props=self._smo_search_props()
   call=ranked[-1]["args"]
   fixed={}
   for name in call:
    value=call[name]
    fixed[name]=(not isinstance(value,str)or bool((props.get(name)or{}).get("enum"))
or value.strip().lower()in shelves or name==key)
   free=[name for name in call if not fixed[name]]
   lead=self.lead_ctx
   if lead and lead[0]in free:
    plan["text_key"]=lead[0]
   else:
    plan["text_key"]=max(free,key=lambda name:len(call[name].split()),default=None)
   if plan["text_key"]:
    plan["template"]={name:call[name]for name in call if name!=plan["text_key"]and fixed[name]}
  plan["done"]={self._smo_two_stems(self._smo_query_of(w["args"]))for w in walks
if start in w["read"]and w.get("id_order")is not True}
  plan["heads"]={self.stem(q)for q in self.label_queries()if" "not in q}
  return plan
 def _smo_untried(self,plan,calls):
  return[c for c in calls if self._smo_sig(c)not in plan["tried"]]
 def _smo_from_texts(self,plan,texts):
  if plan["template"]is None:
   return[]
  calls=[]
  for text in texts:
   sig=self._smo_two_stems(text)
   if sig and sig not in plan["done"]:
    calls.append({**plan["template"],plan["text_key"]:text})
  return self._smo_untried(plan,calls)
 def _smo_shelf_tool(self):
  label=""if self.ceil_spec()else self.raw_label(self.shelf_labels())
  if not label:
   return None
  taken=set(self.role_map.values())
  for tool,shape in self.shapes.items():
   props=(shape or{}).get("properties")
   if tool in taken or not isinstance(props,dict):
    continue
   slot=self.token_key(props)
   shelf,value=self._smu_pick_arg(props,label,(slot,))
   bound=self._smu_cap_arg(props,slot)
   if slot and shelf and bound:
    return dict(tool=tool,shelf=shelf,value=value,bound=bound,slot=slot)
  return None
 def _smo_shelf_ceiling(self,own,tried):
  rows=self.shelf_rows()
  if self.needs.items()or not rows:
   return None,False
  last,token=self.__dict__.get("sweep_listed")or(None,None)
  if last and token is None:
   return None,True
  cap=self.below(min(cost for cost,_shown,_stock in rows.values()))
  if cap is None:
   return None,True
  call={own["shelf"]:own["value"],own["bound"]:cap}
  if self._smo_sig(call)not in tried:
   return call,False
  more=dict(last or call,**{own["slot"]:token})
  return(None,True)if token is None or self._smo_sig(more)in tried else(more,False)
 def _smo_page_token(self,seen):
  token=None
  for obs in self.archive.items[seen:]:
   parts=[obs]+list(obs.values())if isinstance(obs,dict)else[]
   token=next(filter(None,map(self.next_token,parts)),token)
  return token
 def _smo_sweep_ceiling(self,plan):
  own=self._smo_shelf_tool()
  call,ended=self._smo_shelf_ceiling(own,plan["tried"])if own else self.ceiling_sweep(plan["tried"])
  if call is not None:
   return call,True
  return(None,False)if ended else None
 def _smo_sweep_second(self,plan):
  calls=[]
  for walk in plan["book"].values():
   said=[self.stem(t.strip().lower())for t in self.free_text(walk["args"])]
   if len(walk["read"])!=1 or not said or said[0]not in plan["heads"]:
    continue
   if walk["shelves"]&plan["shelves"]and self.next_page(walk)is not None:
    calls.append({**walk["args"],plan["key"]:self.next_page(walk)})
  calls=self._smo_untried(plan,calls)
  return(calls[0],False)if calls and plan["turn"]%2==1 else None
 def _smo_sweep_listing(self,plan):
  walk=plan["listing"]
  calls=self._smo_untried(plan,[{**walk["args"],plan["key"]:self.next_page(walk)}])if walk else[]
  return(calls[0],True)if calls else None
 def _smo_sweep_pairs(self,plan):
  return self._smo_sweep_texts(plan,self.kind_queries(True)[:6])
 def _smo_sweep_words(self,plan):
  return self._smo_sweep_texts(plan,self.kind_queries(False)[:6])
 def _smo_sweep_labels(self,plan):
  return self._smo_sweep_texts(plan,self.label_queries())
 def _smo_sweep_texts(self,plan,texts):
  calls=self._smo_from_texts(plan,texts)
  return(calls[0],False)if calls else None
 _smo_rotating=("pairs","words","labels")
 def sweep_next(self,turn):
  paging=self.page_arg()
  shelves=self.shelf_labels()
  book=self._smo_book()
  if not(paging and shelves and book):
   return None,False
  plan=self._smo_sweep_plan(turn,paging[0],paging[2],shelves,book)
  handlers={"ceiling":self._smo_sweep_ceiling,"second":self._smo_sweep_second,
"listing":self._smo_sweep_listing,"pairs":self._smo_sweep_pairs,
"words":self._smo_sweep_words,"labels":self._smo_sweep_labels}
  route=["ceiling","second","listing"]+[self._smo_rotating[(turn+n)%3]for n in range(3)]
  for state in route:
   outcome=handlers[state](plan)
   if outcome is not None:
    return outcome
  return None,False
 def _smo_sweep_room(self,reads,reserve):
  if reads>=SWEEP_READS or self.moves_left()<=reserve:
   return False
  floor=CLOSING+40
  return self.wire.spare()>=floor
 def _smo_sweep_wanted(self):
  nd=self.needs
  checks=(lambda:nd.priority()=="cheapest",lambda:bool(nd.let_go),lambda:not nd.props_left(),
lambda:bool(self.role("search")),lambda:not self.wire.done)
  return all(check()for check in checks)
 def sweep_step(self):
  if not self._smo_sweep_wanted():
   return False
  reserve=4 if self.trolley.units else 3
  if 0==self.orders_tried:
   reserve=max(reserve+3,FIRST_ORDER_BY)
  reads=self.__dict__.get("swept",0)
  if not self._smo_sweep_room(reads,reserve):
   return False
  turn=self.__dict__.get("sweep_turn",0)
  picked,of_listing=self.sweep_next(turn)
  if picked is None:
   return False
  self.swept=reads+1
  self.sweep_turn=turn if of_listing else turn+1
  self.__dict__.setdefault("sweep_tried",set()).add(self._smo_sig(picked))
  own=self._smo_shelf_tool()
  listing=bool(own)and own["bound"]in picked
  search=own["tool"]if listing else self.role("search")
  call={"id":"sweep_%d_%s"%(self.swept,self.wire.tick),"type":"function",
"function":{"name":search,"arguments":json.dumps(picked)}}
  self.chat.append({"role":"assistant","content":"Price alone decides (no property requirement remains): the cheapest "
"item in this category that can be bought wins, so one more part gets read.","tool_calls":[call]})
  booked=len(self.archive.items)
  try:
   self.covering=True
   answers=self.dispatch([call],scripted=True)
  finally:
   self.covering=False
  if listing:
   self.sweep_listed=picked,self._smo_page_token(booked)
  final=not(self._smo_sweep_room(self.swept,reserve)and self.sweep_next(self.sweep_turn)[0]is not None)
  if final and answers:
   self._smo_sweep_summary(answers)
  self.post_replies(answers)
  return True
 def wide_step(self):
  kv,nd=self.__dict__,self.needs
  mark=nd.widened_at
  if mark<0 or nd.items()or self.orders_tried or self.wire.done or self.wire.broke:
   return False
  if kv.get("wide_at")!=mark:
   kv["wide_at"]=mark
   kv["wide_todo"]=self._smo_wide_texts()if nd.priority()=="cheapest"and self.role("search")else[]
   kv["wide_ids"]=[]if kv["wide_todo"]else None
   if kv["wide_todo"]:
    self.log.append({"kind":"wide_plan","queries":list(kv["wide_todo"])})
  todo=kv.get("wide_todo")or[]
  if todo and self.moves_left()>WIDE_KEEP+len(todo)+1 and self.wire.spare()>CLOSING+40:
   return self._smo_wide_search(todo.pop(0))
  if todo or kv.get("wide_ids")is None:
   kv["wide_todo"]=[]
   return False
  found,kv["wide_ids"]=kv["wide_ids"],None
  return self._smo_wide_open(found)
 def _smo_wide_texts(self):
  values=[alts[0]for alts in self._smt_values(None)if alts and alts[0]][:2]
  lead=" ".join(word for words in values for word in words)
  if not lead:
   return[]
  texts=["%s %s"%(lead,kind)for kind in self._smo_wide_kinds(set(lead.split()))[:2]]+[lead]
  return[text for text in dict.fromkeys(texts)if not self.ran_query(text)][:WIDE_TRIES]
 def _smo_wide_kinds(self,skip):
  heads={self.stem(w)for s in self.shelf_labels()for w in re.findall(self._smo_plain,s.lower())
if len(w)>=3 and w not in LINK_WORDS}
  tally={}
  for _,title,_ in self.shelf_rows().values():
   for word in set(re.findall(r"[a-z]{3,}",title.lower()))-LINK_WORDS-skip:
    root=self.stem(word)
    if any(head in root or(len(root)>=4 and root in head)for head in heads):
     tally.setdefault(root,[0,word])[0]+=1
  ranked=sorted(tally,key=lambda root:(root in heads,-tally[root][0],root))
  return[tally[root][1]for root in ranked if tally[root][0]>=2 or root in heads]
 def _smo_wide_search(self,text):
  shape=self.search_template()
  if not shape:
   self.__dict__["wide_todo"]=[]
   return False
  args,faults=ArgCheck.execute({**shape[0],shape[1]:text},self._smt_search_schema())
  if faults or not isinstance(args,dict):
   return False
  home,start,tick=self.walk_home(None),len(self.archive.items),self.wire.tick
  self.walk_call(args,"A requirement was dropped, so listings the first searches left out may fit now: "
"searching '%s' without the dropped words."%text)
  seen=self.__dict__.setdefault("wide_ids",[])
  for rec in self.archive.items[start:]:
   walk_dicts(rec,lambda d:seen.append(self.row_mark(d))if self._smp_is_row(d)
and(not home or self.shelf_of(d)in home)else None,ceiling=3)
  self.log.append({"kind":"wide_search","query":text,"rows":len(seen)})
  return self.wire.tick!=tick
 def _smo_wide_open(self,found):
  prices=[self.archive.price(line)for line in self.trolley.units]
  below=min([float(p)for p in prices if is_number(p)],default=None)if self.trolley.units else None
  if self.trolley.units and below is None:
   return False
  ranked=[mark for _,mark in self.cheap_unread(list(dict.fromkeys(found)),None,below)]
  room=self.moves_left()-WIDE_KEEP
  queue=[mark for mark in ranked if mark not in self.trolley.units][:max(0,min(WIDE_OPENS,room))]
  self.log.append({"kind":"wide_open","below":below,"rows":ranked[:6],"opened":queue})
  if not queue:
   return False
  self.__dict__.setdefault("named",set()).update(queue)
  self.rank_checked=set()
  for name,value in(("order_vetted",False),("open_queue",queue),("open_recheck",True),
("open_purpose","wide"),
("open_say","Reading the cheaper rows the searches without the dropped requirement showed: "
"the lowest price among the listings that meet every condition now in force "
"wins, so one of them may replace the cart line before the basket message.")):
   setattr(self,name,value)
  sent=self.open_named()
  held=", ".join(sorted(self.trolley.units))
  if sent and held:
   self._smt_tack("[after the drop] Opened, each priced under the cart line %s: %s. Judge each by its record against "
"the conditions now in force only (the dropped one no longer counts). One that meets them all "
"takes the cart line's place - the cheapest such one - BEFORE any basket message, so the "
"basket you tell the shopper is the one you order."%(held,self._smt_priced(queue)),True)
  return sent
 def _smo_sweep_summary(self,answers):
  cheapest=sorted((price,ident,title)for ident,(price,title,_)in self.shelf_rows().items())[:6]
  if not cheapest:
   return
  listed="; ".join("%s '%s' %s"%(ident,title[:80],format(price,"g"))for price,ident,title in cheapest)
  head,body=answers[-1]
  answers[-1]=(head,body+"\n[shelf sweep done] Top prices seen across the asked-for category, only rows you "
"can buy now and not over budget: "+listed+". Plan: barring a change against it, the top row is "
"the pick; it becomes the sole basket line, has stock and price rechecked, then ordered.")
 def free_text(self,call_args):
  props=self._smo_search_props()
  tok=self.token_key()
  texts=[]
  for name in call_args:
   value=call_args[name]
   if isinstance(value,str)and name!=tok and not(props.get(name)or{}).get("enum"):
    texts.append(value)
  return texts
 @staticmethod
 def _smo_ascending(marks):
  def natural(mark):
   return[int(piece)if piece.isdigit()else piece for piece in re.split(r"(\d+)",mark)]
  return marks==sorted(marks)or marks==sorted(marks,key=natural)
 def book_page(self,call_args,rows,total,obs=None):
  paging=self.page_arg()
  if not paging or not rows:
   return
  key,how,start=paging[:3]
  spot=call_args.get(key,start)
  if how=="token":
   spot=spot.strip()if isinstance(spot,str)else""
  elif not is_number(spot):
   return
  rest=self._smo_without(call_args,key)
  book=self.__dict__.setdefault("page_book",{})
  walk=book.setdefault(self._smo_sig(rest),dict(args=rest,read=set(),size=0,total=0,shelves=set()))
  walk["read"].add(spot)
  if how=="token":
   tokens=walk.setdefault("tokens",{})
   tokens[spot]=self.next_token(obs)
  records=[r for r in rows if isinstance(r,dict)]
  walk.update(size=max(walk["size"],len(rows)),total=max(walk["total"],total))
  walk["shelves"]|=set(map(self.shelf_of,records))-{""}
  where=walk.setdefault("row_shelf",{})
  for rec in records:
   where.setdefault(self.row_mark(rec),self.shelf_of(rec))
  order=[self.row_mark(rec)for rec in records]
  if spot==start:
   walk["first"]=tuple(order)
   if total<=len(rows):
    walk["every"]=records
  if total>len(rows):
   walk["full"]=max(walk.get("full",0),len(rows))
  if len(order)>=min(5,max(4,walk.get("full",0)or 5)):
   walk["id_order"]=walk.get("id_order",True)is not False and self._smo_ascending(order)
 def learn_twins(self,call_args,rows):
  if self.word_kind=="any":
   return
  asked={w for t in self.free_text(call_args)for w in re.findall(r"[a-z]{4,}",t.lower())}
  proof=self.__dict__.setdefault("twin_rows",{})
  for row in filter(lambda r:isinstance(r,dict),rows or[]):
   texts=[]
   long_strings_all(row,texts)
   said=set(re.findall(r"[a-z]+"," ".join(texts).lower()))
   for word in asked:
    if any(w==word or same_kind(w,word)for w in said):
     continue
    for other in filter(lambda w:spelling_twin(word,w),said):
     seen=proof.setdefault((word,other),set())
     seen.add(self.row_mark(row))
     if len(seen)<2:
      continue
     STORE_TWINS.setdefault(word,set()).add(other)
     STORE_TWINS.setdefault(other,set()).add(word)
 def twin_query(self,k):
  shape=self.search_template()
  if not shape or not STORE_TWINS:
   return None
  values={w for w in self._smp_rx_a3.findall(" ".join(r.partition(":")[2].lower()for r in self.needs.props_left()
if not self.needs.items()or self.product_of(r.partition(":")[0])in(k,None)))}
  home=self.walk_home(k)
  best=None
  for entry in self._smp_book().values():
   if home and not set((entry.get("row_shelf")or{}).values())&home:
    continue
   words=self._smo_words(" ".join(self.free_text(entry["args"])))
   for at,word in enumerate(words):
    for other in sorted(STORE_TWINS.get(word,()))if word in values else():
     text=" ".join(words[:at]+[other]+words[at+1:])
     if not self.ran_query(text)and(best is None or len(text)<len(best)):
      best=text
  return None if best is None else{**shape[0],shape[1]:best}
 def next_page(self,walk):
  _,how,start,top=self.page_arg()
  read=walk["read"]
  if how=="token":
   for tok in(walk.get("tokens")or{}).values():
    if isinstance(tok,str)and tok not in read:
     return tok
   return None
  floor=min([start]+[v for v in read if is_number(v)])
  size=max(1,walk["size"])
  stride=1 if how else size
  for n in range(1,400):
   spot=floor+n*stride
   if n*size>=walk["total"]or(top is not None and spot>top):
    break
   if spot not in read:
    return spot
  return None
 def value_score(self,call_args):
  wanted=[self._smo_words(value,r"[a-z0-9]{3,}|\b[0-9][a-z]\b|\b[a-z][0-9]\b")for _,value in self._smo_live_reqs(False)]
  lead=self._smo_words(self._smo_query_of(call_args),self._smo_pair2)
  if self.lead_words:
   lead=lead[:self.lead_words]
  near=self._smo_near
  return len([vw for vw in wanted if any(w==v or near(w,v)for v in vw for w in lead)])
 def _smo_sold_by_pointed(self,row,asked):
  for name,value in self._smo_named(row,self._smo_seller):
   if not isinstance(value,str)or not value.strip()or"id"in str(name).lower():
    continue
   seller=set(self._smo_words(value))
   if seller and seller<=asked and seller<=self.pointed_words():
    return True
  return False
 def _smo_hit_shelf(self,row,cap,qty,shelves):
  prices=[v for _,v in self._smo_named(row,r"price|cost")if is_number(v)]
  price=prices[0]if prices else None
  shelf=self.shelf_of(row)
  if is_number(cap)and is_number(price)and price*qty>cap:
   return None
  if self.sold_out(row)or(shelves and shelf and shelf not in shelves):
   return None
  return shelf
 def row_hits(self,rows,words=None):
  cap,qty=self.needs.spend_cap,self.needs.quantity()
  shelves=self.shelf_labels()
  refs=self.needs.reading.get("references")
  asked=set(self._smo_words(" ".join(str(said)for said in words or[])))if refs else set()
  pointers=self.__dict__.setdefault("ref_hits",[])
  hits=[]
  for row in filter(lambda r:isinstance(r,dict),rows):
   ident=self.row_mark(row)
   if ident not in self.archive.ids or ident in self.archive.read:
    continue
   if asked and self._smo_sold_by_pointed(row,asked)and ident not in pointers:
    pointers.append(ident)
   shelf=self._smo_hit_shelf(row,cap,qty,shelves)
   if shelf is not None:
    hits.append(ident)
    self.__dict__.setdefault("hit_shelf",{})[ident]=shelf
  return hits
 def _smo_field_role(self,low,value):
  if re.search(self._smo_seller,low)and"id"not in low and len(value)<=40 and not re.search(r"note|text|desc|comment|detail|info|message|section",low):
   return"seller"
  return""if re.search(r"title|heading|headline|name|label|caption",low)else"text"
 def _smo_note_words(self,record,found):
  if self.row_mark(record)not in self.archive.read:
   return
  own,notes=set(),[]
  for name in record:
   value=record[name]
   if isinstance(value,str):
    role=self._smo_field_role(str(name).lower(),value)
    if role=="seller":
     own.update(self._smo_words(value))
    elif role:
     notes.append(value.lower())
  for note in notes:
   found.update(set(self._smo_words(note))-own)
 def pointed_words(self):
  size=len(self.archive.items)
  cached=self.__dict__.get("pointed_memo")
  if cached and cached[0]==size:
   return cached[1]
  found=set()
  try:
   self._smo_gather_pointed(found)
  except Exception:
   pass
  self.pointed_memo=size,found
  return found
 def _smo_gather_pointed(self,found):
  refs=self.needs.reading.get("references")or[]
  for ref in refs:
   found.update(self._smo_words(str(ref)))
  for rec in self.archive.items:
   walk_dicts(rec,lambda d:self._smo_note_words(d,found))
 def open_pages(self,most=2):
  paging=self.page_arg()
  if not paging:
   return[]
  shelves=self.shelf_labels()
  calls,order=[],[]
  for walk in self._smo_book().values():
   spot=self.next_page(walk)
   if spot is None:
    continue
   if shelves and walk["shelves"]and not walk["shelves"]&shelves:
    continue
   weight=self.value_score(walk["args"])
   if weight>0:
    call=dict(walk["args"])
    call[paging[0]]=spot
    order.append((-weight,walk["total"],len(calls)))
    calls.append(call)
  return[calls[slot]for _,_,slot in sorted(order)[:most]]
 def _smo_lead(self,text,reach):
  return[w for w in self._smo_words(text)if len(w)>1][:reach]
 _smo_blank=("","")
 def _smo_probe_reqs(self):
  pairs=[]
  for name,value in self._smo_live_reqs(True):
   untagged=re.sub(self._smo_said_tag,"",value)
   pairs.append((name.strip(),re.split(r"[,;]",untagged)[0].strip().lower()))
  return pairs
 def _smo_probe_ran(self,head,reach):
  close=self._smo_near
  for walk in self._smo_book().values():
   ran=self._smo_lead(self._smo_query_of(walk["args"]),reach)
   if len(ran)==len(head)and all(close(a,b,"lead")for a,b in zip(head,ran)):
    return True
  return False
 def value_probe(self,raw):
  if self.word_kind=="any":
   return self._smo_blank
  reach=self.lead_words or 2
  lead=self._smo_lead(" ".join(raw),reach)
  product=" ".join(self._smo_words(str(self.needs.reading.get("product")or""))[:3])
  reqs=self._smo_probe_reqs()
  def touches(pool):
   return any(self._smo_near(w,k,"lead")for k in pool for w in lead)
  known=[w for _,value in reqs for w in re.findall(self._smo_pair2,value)]+product.split()
  if not lead or touches(known):
   return self._smo_blank
  hit=next(((name,value)for name,value in reqs if value and touches(self._smo_words(name,self._smo_pair2))),None)
  if hit is None:
   return self._smo_blank
  digit=re.search(r"\d",hit[1])
  core=hit[1][digit.start():]if digit else hit[1]
  probe=(core.strip(" .;")+" "+product).strip()
  return self._smo_blank if self._smo_probe_ran(self._smo_lead(probe,reach),reach)else(hit[0],probe)
 def _smo_stem_key(self,text,width):
  return sorted(map(self.stem,self._smo_words(text)[:width]))
 def ran_query(self,text):
  width=self.lead_words or 3
  target=self._smo_stem_key(str(text),width)
  return any(self._smo_stem_key(self._smo_query_of(w["args"]),width)==target for w in self._smo_book().values())
 def search_template(self):
  newest_first=list(self._smo_book().values())[::-1]
  for call in(walk["args"]for walk in newest_first):
   if self.cg_listing(call):
    continue
   free=self.free_text(call)
   slots=[name for name in call if isinstance(call[name],str)and call[name]in free and call[name].strip()]
   if len(free)==len(slots)==1:
    return self._smo_without(call,slots[0]),slots[0]
  return None
 def _smo_bits(self,token):
  return[p for p in re.findall(self._smo_plain,token)if len(p)>1]
 def _smo_trap_note(self,why,word,query):
  if why=="name":
   head=(f"In this store every word must match, and '{word}' belongs to a property's NAME rather than to any "
"required value: only listings that spell out that name in a field or in the title come back, so one "
"giving nothing but the value there is lost.")
  else:
   head=(f"'{word}' comes from nothing but a requirement the shopper has let go of, yet the search still asks "
"every listing to carry it, hiding items now acceptable.")
  return head+f" Next try '{query}' too, going through all pages."
 def _smo_word_pool(self,texts,pattern):
  pool=set()
  for text in texts:
   pool.update(self._smo_words(text,pattern))
  return pool-LINK_WORDS
 def _smo_trap(self,raw):
  if self.word_kind=="any":
   return self._smo_blank
  cleaned=(piece.strip("\"'()[]{},.;!?|").lower()for piece in" ".join(raw).split())
  toks=[t for t in cleaned if re.search(r"[a-z0-9]",t)]
  if self.lead_words:
   width=self.lead_words
  else:
   width=max(3,len(toks))if self.word_kind=="all"else 3
  toks=toks[:width]
  close=self._smo_near
  reqs=[]
  for name,value in self._smo_live_reqs(True):
   reqs.append((re.split(r"\s*\(",name)[0].lower(),name,re.sub(self._smo_said_tag,"",value).lower()))
  vals=set()
  for req in reqs:
   vals.update(re.findall(self._smo_pair2,req[2]))
  lines=self.needs.items()
  products=[line["product"]for line in lines]or[str(self.needs.reading.get("product")or"")]
  kinds=self._smo_word_pool([" ".join(products+sorted(self.shelf_labels()))],r"[a-z0-9]{3,}")
  names=self._smo_word_pool([req[0]for req in reqs],r"[a-z]{3,}")
  dropped=self._smo_word_pool([old.partition(":")[2]for old in self.needs.let_go],r"[a-z]{3,}")
  rules=(("",vals|kinds),("name",names),("dropped",dropped))
  flagged,why=[],""
  for t in toks:
   bits=self._smo_bits(t)
   verdict=next((tag for tag,pool in rules if any(close(b,w,"eq")for b in bits for w in pool)),"")
   if verdict:
    flagged.append(t)
    why=why or verdict
  kept=list(filter(lambda t:t not in flagged,toks))
  kept_bits=[b for t in kept for b in self._smo_bits(t)]
  held=[req[1]for req in reqs if any(close(b,w,"eq")for b in kept_bits for w in re.findall(self._smo_pair2,req[2]))]
  if not(flagged and held):
   return self._smo_blank
  if not any(close(b,k,"eq")for b in kept_bits for k in kinds):
   at=self.product_of(held[0])if self.needs.items()else None
   named=products[at]if at is not None and at<len(products)else products[0]
   kept+=[w for w in self._smo_words(named,r"[a-z]{3,}")if w not in LINK_WORDS][-1:]
  query=" ".join(kept[:width])
  if query and not self.ran_query(query):
   return self._smo_trap_note(why,flagged[0],query),query
  return self._smo_blank
 def word_trap(self,raw):
  try:
   return self._smo_trap(raw)
  except Exception as err:
   self.log.append(dict(kind="trap_error",error=(type(err).__name__+": "+str(err))[:300]))
   return"",""
 _smp_tag=re.compile(r"\((?:stated|revealed[^)]*)\)")
 _smp_odd_value=re.compile(r"\d|as[ ]shown|same[ ]as|page|redirect|refer|\bnot\b|\bno\b|without|except|anything|other[ ]than")
 _smp_first_alt=re.compile(r"[,;/]|\bor\b|\band\b")
 _smp_alts=re.compile(r"[,;/]|\bor\b|\beither\b|\bnor\b")
 _smp_by_ref=re.compile(r"as[ ]shown|same[ ]as|the[ ]page|sold[ ]by|redirect|refer")
 _smp_rx_an=re.compile(r"[a-z0-9]+")
 _smp_rx_an2=re.compile(r"[a-z0-9]{2,}")
 _smp_rx_an3=re.compile(r"[a-z0-9]{3,}")
 _smp_rx_a=re.compile(r"[a-z]+")
 _smp_rx_a2=re.compile(r"[a-z]{2,}")
 _smp_rx_a3=re.compile(r"[a-z]{3,}")
 _smp_rx_code2=re.compile(r"\b(?:[0-9][a-z]|[a-z][0-9])\b")
 _smp_rx_stock=re.compile(r"stock|avail")
 _smp_rx_maker=re.compile(r"brand|make|manufactur")
 _smp_rx_cost=re.compile(r"price|cost")
 _smp_rx_news=re.compile(r"shopper|Meaning:|REFUSED")
 _smp_digest_head="Page rows (id, price, title, category): "
 def _smp_drive(self,steps,state,job):
  while state is not None:
   state=steps[state](job)
  return job
 def _smp_fault(self,kind,err):
  text="%s: %s"%(err.__class__.__name__,err)
  self.log.append({"kind":kind,"error":text[:300]})
 def _smp_attr(self,name,default=None):
  return self.__dict__.get(name,default)
 def _smp_book(self):
  return self._smp_attr("page_book",{})
 def _smp_hurry(self):
  return self.wire.spare()<CLOSING+30
 def _smp_cleared(self):
  now=str(self.needs.version)
  out=set()
  for entry in self.passed:
   ident,_,ver=entry.rpartition("|")
   if ver==now:
    out.add(ident)
  return out
 def _smp_struck(self):
  now=[str(self.needs.version)]
  out=set()
  for memo,said in self.rulings.items():
   parts=memo.split("|")
   if said and parts[1:2]==now:
    out.add(parts[0])
  return out
 def _smp_asked_product(self):
  return str(self.needs.reading.get("product")or"")
 def _smp_shelves_of(self,ids):
  found={self.shelf_known(i)for i in ids}
  found.discard("")
  return found
 def _smp_stock_marks(self,keep):
  marks={}
  def note(d):
   if keep(d):
    marks[self.row_mark(d)]=self.sold_out(d)
  for rec in self.archive.items:
   walk_dicts(rec,note)
  return marks
 def _smp_short(self,margin,broke=False):
  if self.moves_left()<=margin()or self.wire.done:
   return True
  return bool(broke and self.wire.broke)or self._smp_hurry()
 def combined_queries(self):
  shape=self.search_template()
  if not shape:
   return[]
  products=self.needs.items()
  room=self.lead_words or 3
  verified_on=self._smp_shelves_of(self._smp_cleared()-self.refused_lines)
  if products:
   shelves=self.home_shelves()
   open_ones=[n for n in range(len(products))if n<len(shelves)and shelves[n]and not shelves[n]&verified_on]
  else:
   open_ones=[None]if self.unverified()else[]
  ready=[]
  for n in open_ones:
   groups=self._smp_value_words(n,bool(products))
   if len(groups)<2:
    continue
   label=self._smp_asked_product()if n is None else products[n]["product"]
   query=self._smp_pack(groups,room,label)
   phrase=" ".join(query)
   if len(query)>1 and not self.ran_query(phrase):
    ready.append({**shape[0],shape[1]:phrase})
  return ready[:2]
 def _smp_value_words(self,n,several):
  groups=[]
  for req in self.needs.props_left():
   name,_,said=req.partition(":")
   if several and self.product_of(name)not in(n,None):
    continue
   plain=self._smp_tag.sub("",said).strip().lower()
   if plain and not self._smp_odd_value.search(plain):
    kept=[w for w in self._smp_rx_a2.findall(self._smp_first_alt.split(plain)[0])if w not in LINK_WORDS]
    if kept and len(kept)<=2:
     groups.append(kept)
  return groups
 def _smp_pack(self,groups,room,label):
  query=[]
  for group in groups:
   if len(query)+len(group)<=room:
    query+=group
  nouns=[w for w in self._smp_rx_a3.findall(label.lower())if w not in LINK_WORDS]
  if nouns and len(query)<room and nouns[-1]not in query:
   query.append(nouns[-1])
  return query
 def arbitrate(self,thing,ruling):
  try:
   redo=self.check_item(thing,"ORDER",second=self._smp_dispute(ruling))
   if redo:
    return redo
   prefix=f"{thing}|{self.needs.version}"
   stale=[m for m in self.rulings if m==prefix or m.startswith(prefix+"|")]
   self.rulings.update(dict.fromkeys(stale,""))
   return""
  except Exception as err:
   self._smp_fault("arbitrate_error",err)
   return ruling
 def _smp_dispute(self,ruling):
  why=""
  for entry in reversed(self.log):
   if isinstance(entry,dict)and entry.get("kind")=="audit"and isinstance(entry.get("result"),dict):
    why=str(entry["result"].get("reason")or"")
    break
  return("\n\nSECOND OPINION NEEDED\n- Audit verdict: the pre-decline audit judged this item compliant"
+(f" ({why[:400]})"if why else"")+".\n- Earlier check verdict: "+ruling[:500]
+"\n- How to settle: go by the records alone. Treat a field as satisfying a requirement when it expresses the "
"requested value with different wording, or as a shade, qualifier or alternate spelling of that very value; treat "
"it as failing when it expresses some other value.")
 def unrefuted(self):
  try:
   products=self.needs.items()
   shelves=self.home_shelves()if products else[set(self.shelf_labels())]
   if not shelves or not all(shelves):
    return[]
   barred=self._smp_struck()
   verified=self._smp_cleared()
   barred|=self._smp_audited_out()
   gone=self._smp_stock_marks(self._smp_has_stock_field)
   limit=self.needs.spend_cap
   barred|=self.refused_lines|self.ref_pointers()
   return self._smp_cheapest_set(shelves,products,verified,barred,gone,limit)
  except Exception as err:
   self._smp_fault("unrefuted_error",err)
   return[]
 def _smp_has_stock_field(self,d):
  return self.row_mark(d)in self.archive.ids and any(self._smp_rx_stock.search(str(f).lower())for f in d)
 def _smp_cheapest_set(self,shelves,products,verified,barred,gone,limit):
  picks=[]
  outlay=0.0
  capped=is_number(limit)
  for n,shelf in enumerate(shelves):
   units=products[n]["quantity"]if products else self.needs.quantity()
   ranked=[]
   for ident in sorted(self.archive.read):
    if ident in barred or gone.get(ident)or ident in picks or self.shelf_known(ident)not in shelf:
     continue
    price=self.archive.price(ident)
    if is_number(price)and not(capped and price*units>limit):
     ranked.append((ident not in verified,price,ident))
   if not ranked:
    return[]
   _,price,ident=min(ranked)
   picks.append(ident)
   outlay+=price*units
  return[]if capped and outlay>limit else picks
 def _smp_audited_out(self):
  seen=self._smp_attr("audit_ids")
  reads=self.archive.read
  whole=self._smp_attr("audit_done")and(seen is None or len(reads)<=self._smp_attr("audit_reads",-1))
  reviewed=set(reads)if whole else set(seen or())
  return reviewed.difference(self._smp_attr("audit_fits")or[])
 def fit_hold(self):
  try:
   steps={"gate":self._smp_fit_gate,"check":self._smp_fit_check,"cover":self._smp_fit_cover,
"word":self._smp_fit_word}
   return self._smp_drive(steps,"gate",{"good":[],"note":""})["note"]
  except Exception as err:
   self._smp_fault("fit_hold_error",err)
   return""
 def _smp_fit_gate(self,job):
  if self._smp_attr("fit_held")or self.trolley.units or self.moves_left()<5 or self._smp_hurry():
   return None
  known=self.archive.ids
  job["todo"]=[i for i in(self._smp_attr("audit_fits")or[])if i in known and i not in self.refused_lines][:2]
  if not job["todo"]:
   return None
  self.fit_held=True
  return"check"
 def _smp_fit_check(self,job):
  if not job["todo"]:
   return"cover"
  thing=job["todo"].pop(0)
  if thing not in self.archive.read:
   return"check"
  verdict=self.check_item(thing,"ORDER")
  if verdict and CLOSING+30<self.wire.spare():
   verdict=self.arbitrate(thing,verdict)
  if not verdict and thing in self._smp_cleared():
   job["good"].append(thing)
   if not self.needs.items():
    return"cover"
  return"check"
 def _smp_fit_cover(self,job):
  if not job["good"]:
   return None
  if self.needs.items():
   have=self._smp_shelves_of(self._smp_cleared()-self.refused_lines)
   if any(shelf and not shelf&have for shelf in self.home_shelves()):
    return None
  return"word"
 def _smp_fit_word(self,job):
  good=job["good"]
  one=len(good)==1
  what="one passed line for each product"if self.needs.items()else good[0]
  job["note"]=("Decline on hold: the pre-decline audit flagged "+", ".join(good)
+(" as a candidate; a full-record item check then cleared it on"
if one else" as candidates; full-record item checks then cleared them both on")
+" each rule now applying. Reconsider. If it truly qualifies, place "+what
+" into the cart, refresh its stock plus price, then send the order;"
" the decline is for use once a detail shows it failing one rule.")
 def id_store(self):
  flags=[entry.get("id_order")for entry in self._smp_book().values()if"id_order"in entry]
  return False not in flags and flags.count(True)>1
 def ranked_store(self):
  flags=[entry.get("id_order")for entry in self._smp_book().values()if"id_order"in entry]
  return True not in flags and flags.count(False)>1
 def walk_store(self):
  flags=[entry.get("id_order")for entry in self._smp_book().values()if"id_order"in entry]
  return self.id_store()or self.ranked_store()or(True in flags and False in flags and len(flags)>1)
 def walk_listing(self,entry):
  order=entry.get("id_order")
  return order is True or(order is False and(self.ranked_store()or self._smp_ranked_home(entry)))
 def _smp_ranked_home(self,entry):
  if not self.free_text(entry.get("args")or{}):
   return False
  homes=set().union(*self.home_shelves())if self.home_shelves()else set()
  return bool(homes&set(entry.get("shelves")or()))
 def product_index(self,thing):
  if not self.needs.items():
   return None
  shelf=self.shelf_known(thing)
  shelves=self.home_shelves()
  if shelf:
   for n,home in enumerate(shelves):
    if shelf in home:
     return n
  return None
 def walk_home(self,k):
  at=0 if k is None else k
  shelves=self.home_shelves()
  own=set(shelves[at])if shelves and at<len(shelves)else set()
  if own or k is not None:
   return own
  return set(self.shelf_labels())
 @staticmethod
 def fig_amount(text):
  low=str(text).lower()
  hit=None if re.search(CH_BOUND,low)else re.match(r"\s*(\d+)\s+([a-z]+)",low)
  return hit.groups()if hit else None
 @staticmethod
 def amount_part(text):
  low=str(text).lower()
  if not any(ch.isdecimal()for ch in low):
   return False
  return re.search(CH_BOUND,low)is not None or re.match(r"\s*\d+(?:[.,]\d+)?\s+[a-z]",low)is not None
 @staticmethod
 def _smp_near(a,b):
  return min(len(a),len(b))>2 and(a[:len(b)]==b or b[:len(a)]==a)
 def walk_carries(self,args,k,alts=False):
  lowered=[t.lower()for t in self.free_text(args)]
  span=self.lead_words or None
  words=[w for t in lowered for w in self._smp_rx_an2.findall(t)][:span]
  tokens=[w for t in lowered for w in self._smp_rx_an.findall(t)][:span]
  many=bool(self.needs.items())
  only_k=many and k is not None
  found=set()
  for n,req in enumerate(self.needs.props_left()):
   name,_,said=req.partition(":")
   if only_k and self.product_of(name)not in(k,None):
    continue
   pieces=self._smp_alts.split(self._smp_tag.sub("",said).lower())
   found.update((n,a)if alts else n for a,piece in enumerate(pieces)if self._smp_part_carried(piece,words,tokens))
  return found
 def _smp_part_carried(self,piece,words,tokens):
  fig=self.fig_amount(piece)
  if fig:
   num,unit=fig
   return num in tokens and any(t==unit or self._smp_near(t,unit)for t in tokens)
  if self.amount_part(piece):
   return False
  if any(v in words for v in self._smp_rx_code2.findall(piece)):
   return True
  return any(self._smp_near(w,v)for w in words for v in self._smp_rx_an3.findall(piece))
 def walk_score(self,args,k):
  carried=self.walk_carries(args,k)
  return len(carried)
 def _smp_lead(self,entry):
  found=self._smp_rx_an.findall(" ".join(self.free_text(entry["args"])).lower())
  if self.lead_words:
   found=found[:self.lead_words]
  return set(map(self.stem,found))
 def _smp_value_stems(self,k):
  narrow=bool(self.needs.items())and k is not None
  values=[said for name,_,said in(r.partition(":")for r in self.needs.props_left())
if not narrow or self.product_of(name)in(k,None)]
  return{self.stem(w)for said in values for w in self._smp_rx_an2.findall(said.lower())}
 def _smp_read_out(self,entry):
  if not self.walk_listing(entry)or not entry.get("first")or self.next_page(entry)is not None:
   return False
  return bool(self._smp_lead(entry))and len(entry.get("row_shelf")or{})>=entry["total"]
 def walk_options(self,k,deep=False):
  shelf=self.walk_home(k)
  if not shelf:
   return[]
  wide={self.stem(w)for s in shelf for w in self._smp_rx_an2.findall(s)}
  wide|=self._smp_value_stems(k)
  book=self._smp_book()
  finished=[(self._smp_lead(e),self.walk_carries(e["args"],k,alts=True),self.vl_mode(e["args"]))
for e in book.values()if self._smp_read_out(e)]
  bare=self._smp_bounds_only(k)
  deep=deep or bare
  twins={}
  for name,entry in book.items():
   if not self._smp_live_walk(entry):
    continue
   carried=self.walk_carries(entry["args"],k)or(bare and self._smp_on_shelf(entry,shelf))
   showing=any(s in shelf for s in(entry.get("row_shelf")or{}).values())
   if not(showing and carried)or self._smp_shadowed(entry,k,wide,finished,strict=bare):
    continue
   left=self._smp_pages_left(entry)
   twin=(entry["total"],entry["first"])
   held=twins.get(twin)
   short=0<entry.get("size",0)<5
   reach=0<left<=2*WALK_PAGES or(left>0 and(short or(deep and self._smp_on_shelf(entry,shelf))))
   if reach and(held is None or left<held[0]):
    twins[twin]=(left,name,entry)
  options=list(twins.values())
  if self.all_words():
   options=[o for o in options if not any(self._smp_nests(p,o,wide)for p in options if p is not o)]
  ranked=sorted((o[0],-self.walk_score(o[2]["args"],k),o[1],o)for o in options)
  return[o for*_,o in ranked]
 @staticmethod
 def _smp_on_shelf(entry,shelf):
  rows=entry.get("row_shelf")or{}
  return len(rows)>=max(1,entry.get("size",0))and all(s in shelf for s in rows.values())
 def _smp_live_walk(self,entry):
  return self.walk_listing(entry)and bool(entry.get("first"))and self.next_page(entry)is not None
 def _smp_shadowed(self,entry,k,wide,finished,strict=False):
  mine=self._smp_lead(entry)
  pairs=self.walk_carries(entry["args"],k,alts=True)
  how=self.vl_mode(entry["args"])
  for seen,got,mode in finished:
   if mode==how and(seen<=mine or(not strict and seen<=wide and pairs<=got)):
    return True
  return False
 def _smp_bounds_only(self,k):
  if k is None or not self.needs.items():
   return False
  tails=[said for name,_,said in(r.partition(":")for r in self.needs.props_left())if self.product_of(name)==k]
  return bool(tails)and all(re.search(CH_BOUND,self._smp_tag.sub("",t).lower())and self.amount_part(t)for t in tails)
 def _smp_pages_left(self,entry):
  size=max(1,entry["size"])
  full,rest=divmod(entry["total"],size)
  pages=full+(1 if rest else 0)
  paging=self.page_arg()
  if paging and paging[1]!="token"and is_number(paging[3])and is_number(paging[2]):
   pages=min(pages,int((paging[3]-paging[2])//(1 if paging[1]else size))+1)
  return pages-len(entry["read"])
 def _smp_nests(self,p,o,wide):
  small,big=self._smp_lead(p[2]),self._smp_lead(o[2])
  if small<big and not(big-small)<=wide and p[0]<=WALK_PAGES:
   return True
  return big<small and(small-big)<=wide
 def walk_plan(self,budget,only=None,rule="cheapest"):
  gates=(lambda:budget>0,lambda:self.needs.priority()==rule,lambda:bool(self.role("search")),
lambda:bool(self.page_arg()),self.walk_store)
  if not all(gate()for gate in gates):
   return[]
  walked=self._smp_attr("walked",set())
  single=not self.needs.items()
  wants=[]
  for n in list(range(len(self.needs.items())))or[None]:
   if n in walked or(only is not None and n not in only):
    continue
   best=next(iter(self.walk_options(n,deep=single)),None)
   if best:
    wants.append((best[0],-1 if n is None else n,n,best[1]))
  wants.sort(key=lambda w:w[:2])
  plan=[]
  left=budget
  for pages,_,n,name in wants:
   take=min(pages,WALK_PAGES,left)
   if take<=0:
    break
   plan.append((n,name,take))
   left-=take
  if single and plan:
   for pages,name,_ in self.walk_options(None,deep=True)[1:]:
    take=min(pages,WALK_PAGES,left)
    if take<=0:
     break
    plan.append((None,name,take))
    left-=take
  return plan
 def walk_pick(self,k):
  gone=self.dead_lines()
  def ours(ident):
   return k is None or self.product_index(ident)==k
  pool=[i for i in self.trolley.units if ours(i)and i not in gone]
  if not pool:
   pool=[i for i in self._smp_cleared()if i not in self.refused_lines and i not in gone and ours(i)]
  best=None
  for ident in pool:
   price=self.archive.price(ident)
   if is_number(price)and(best is None or(float(price),ident)<best):
    best=(float(price),ident)
  return(best[1],best[0])if best else None
 def _smp_titled_record(self,ident):
  hits=[]
  for rec in self.archive.items:
   walk_dicts(rec,lambda d:hits.append(d)if self.row_mark(d)==ident and self.title_of(d)else None)
  return hits[-1]if hits else{}
 def _smp_line_stopwords(self,k,maker_words):
  texts=list(self.walk_home(k))
  texts+=[req.lower()for req in self.needs.props_left()]
  kinds=self.needs.items()or[{"product":self._smp_asked_product()}]
  texts+=[r["product"].lower()for r in kinds]
  stop=LINK_WORDS|set(maker_words)
  for text in texts:
   stop.update(self._smp_rx_a.findall(text))
  return stop
 def line_query(self,k):
  pick=self.walk_pick(k)
  shape=self.search_template()
  if not pick or not shape:
   return None
  rec=self._smp_titled_record(pick[0])
  maker=""
  for field,val in rec.items():
   tag=str(field).lower()
   if isinstance(val,str)and val.strip()and self._smp_rx_maker.search(tag)and"id"not in tag:
    maker=val
    break
  maker_words=self._smp_rx_an.findall(maker.lower())
  if not maker_words:
   return self.title_line(k,rec,shape)
  brand=maker_words[0]
  stop=self._smp_line_stopwords(k,maker_words)
  title=re.findall(r"[a-z][a-z'-]{3,}",self.title_of(rec).lower())
  rest=title[title.index(brand)+1:]if brand in title else title
  name=next((w for w in rest if"'"not in w and w.strip("'-")not in stop),None)
  if name is None:
   return None
  text=brand+" "+name
  return None if self.ran_query(text)else{**shape[0],shape[1]:text}
 def walk_rows(self,k,keys,most,read_bar=True):
  shelf=self.walk_home(k)
  book=self._smp_book()
  counts={name:len(self.walk_carries(e["args"],k))for name,e in book.items()if self.walk_listing(e)}
  top=max(counts.values(),default=0)
  bare=not top and self._smp_bounds_only(k)
  shown=set()
  for name,entry in book.items():
   hits=counts.get(name,0)
   plain=bare and name in counts and self._smp_on_shelf(entry,shelf)
   if name in keys or(hits and hits==top)or plain:
    shown.update(i for i,s in(entry.get("row_shelf")or{}).items()if s in shelf)
  bar=self._smp_row_bar(k,shown)if read_bar else(self.walk_pick(k)or(None,None))[1]
  products=self.needs.items()
  units=self.needs.quantity()if k is None or k>=len(products)else products[k]["quantity"]
  limit=self.needs.spend_cap
  gone=self._smp_stock_marks(lambda d:self._smp_names_one(d,shown))
  cheap=[]
  for ident in shown:
   price=self.archive.price(ident)
   skip=ident in self.archive.read or ident in self.refused_lines or gone.get(ident)or not is_number(price)
   if skip or(is_number(limit)and price*units>limit)or(bar is not None and price>=bar):
    continue
   cheap.append((price,ident))
  cheap.sort()
  return[ident for _,ident in cheap[:most]]
 def _smp_row_bar(self,k,shown):
  pick=self.walk_pick(k)
  if pick:
   return pick[1]
  low=None
  for ident in shown:
   if ident in self.archive.read and is_number(self.archive.price(ident)):
    here=(float(self.archive.price(ident)),ident)
    low=here if low is None or here<low else low
  return None if low is None else low[0]
 @staticmethod
 def _smp_names_one(d,pool):
  return any("id"in str(f).lower()for f,v in d.items()if isinstance(v,str)and v in pool)
 def _smp_is_row(self,d):
  return len(d)>3 and self._smp_names_one(d,self.archive.ids)
 def page_digest(self,obs):
  rows=[]
  walk_dicts(obs,lambda d:rows.append(d)if self._smp_is_row(d)else None,ceiling=3)
  cells=[]
  for row in rows[:12]:
   price=next((v for f,v in row.items()if is_number(v)and self._smp_rx_cost.search(str(f).lower())),None)
   mark=self.row_mark(row)
   flag=" SOLD OUT"if self.sold_out(row)else""
   cells.append(f"{mark} {'?' if price is None else price}{flag} '{self.title_of(row)[:70]}' [{self.shelf_of(row)}]")
  return self._smp_digest_head+("; ".join(cells)if cells else"none")
 def _smp_tool_reply(self,call_id,content):
  self.chat.append({"role":"tool","tool_call_id":call_id,"content":content})
 def walk_call(self,args,say,tool=None):
  call=dict(id=f"walk_{len(self.chat)}_{self.wire.tick}",type="function",
function=dict(name=tool or self.role("search"),arguments=json.dumps(args)))
  self.chat.append(dict(role="assistant",content=say,tool_calls=[call]))
  count=len(self.archive.items)
  self.covering=True
  try:
   answers=self.dispatch([call],scripted=True)
  finally:
   self.covering=False
  if not answers:
   self._smp_tool_reply(call["id"],"Not run.")
   return False
  reply_id,reply=answers[0]
  grew=count<len(self.archive.items)
  _,_,news=reply.partition("\n[note] ")
  content=reply
  if grew:
   content=self.page_digest(self.archive.items[-1])
   if news and self._smp_rx_news.search(news):
    content+="\n[note] "+news
  self._smp_tool_reply(reply_id,content)
  return grew
 def run_walk(self,plan,floor,ceil=None,repair=False,twins=None,rungs=None):
  paging=self.page_arg()
  if not(plan or ceil or twins or rungs)or not paging or not self.role("search"):
   return False
  job={"plan":plan,"floor":floor,"ceil":ceil,"repair":repair,"key":paging[0],"book":self._smp_book(),
"walked":self.__dict__.setdefault("walked",set()),"report":[],"ran":False,"lines":{},
"ceil_rows":{},"queue":[],"ok":False,"twins":dict(twins or{}),"twin_keys":{},
"rungs":dict(rungs or{}),"rung_keys":{}}
  steps={"rungs":self._smp_walk_rungs,"twins":self._smp_walk_twins,"pages":self._smp_walk_pages,
"lines":self._smp_walk_lines,"ceiling":self._smp_walk_ceiling,"queue":self._smp_walk_queue,
"close":self._smp_walk_close}
  return self._smp_drive(steps,"rungs",job)["ok"]
 def _smp_walk_rungs(self,job):
  calls=0
  for n,searches in job["rungs"].items():
   keys=job["rung_keys"].setdefault(n,[])
   for search in searches:
    if calls>=RUNG_CALLS or self._smp_short(lambda:job["floor"]+2):
     break
    words=" ".join(self.free_text(search))
    calls+=1
    if not self.walk_call(search,"Only a figure limits this product, so I search the whole figures just "
f"past the limit: '{words}'."):
     continue
    job["ran"]=True
    key=self._smo_sig({a:v for a,v in search.items()if a!=job["key"]})
    keys.append(key)
    entry=self._smp_book().get(key)
    while entry and calls<RUNG_CALLS and len(entry["read"])<RUNG_PAGES and self.next_page(entry)is not None:
     if self._smp_short(lambda:job["floor"]+2):
      break
     calls+=1
     if not self.walk_call({**entry["args"],job["key"]:self.next_page(entry)},
"Reading that figure search to its end."):
      break
    job["report"].append(f"the figure search '{words}' ({len((entry or {}).get('row_shelf') or {})} rows seen)")
  return"twins"
 def _smp_walk_twins(self,job):
  for n,search in job["twins"].items():
   if self._smp_short(lambda:job["floor"]+2):
    break
   words=" ".join(self.free_text(search))
   if not self.walk_call(search,f"The store spells a required value two ways, so I also search '{words}'."):
    continue
   job["ran"]=True
   job["twin_keys"][n]=json.dumps({a:v for a,v in search.items()if a!=job["key"]},sort_keys=True,default=str)
   job["report"].append(f"the other spelling '{words}'")
  return"pages"
 def _smp_walk_pages(self,job):
  for n,name,pages in job["plan"]:
   job["walked"].add(n)
   entry=job["book"].get(name)
   if not entry:
    continue
   query=" ".join(self.free_text(entry["args"]))
   order="ranks its search results by match"if self.ranked_store()else"sorts its search results by id"
   say=(f"Lowest price wanted, but this store {order} instead of by their cost, so I keep "
f"paging on through '{query}'.")
   done=0
   for _ in range(pages):
    token=self.next_page(entry)
    if token is None or self._smp_short(lambda:job["floor"]+1,broke=True):
     break
    if not self.walk_call({**entry["args"],job["key"]:token},say):
     break
    done+=1
   job["ran"]=job["ran"]or done>0
   seen=len(entry.get("row_shelf")or{})
   job["report"].append(f"'{query}' paged {done} further time(s); rows now seen {seen} of {entry['total']}")
  return"lines"
 def _smp_walk_lines(self,job):
  for n in dict.fromkeys(n for n,_,_ in job["plan"]):
   if self._smp_short(lambda:job["floor"]+2):
    break
   search=self.line_query(n)
   if not search or not self.walk_call(search,"Then a search for the pick's own product family (maker plus "
"family name): sibling sizes or colours appear as their own listings and "
"might well come in cheaper."):
    continue
   job["ran"]=True
   job["lines"][n]=json.dumps({a:v for a,v in search.items()if a!=job["key"]},sort_keys=True,default=str)
   self._smp_line_to_end(self._smp_book().get(job["lines"][n]),job)
   words=" ".join(self.free_text(search))
   job["report"].append(f"the line '{words}' (pick)")
  return"ceiling"
 def _smp_line_to_end(self,line,job):
  for _ in range(2):
   if not line or line["total"]>3*max(1,line["size"])or self.next_page(line)is None:
    return
   if self.moves_left()<=job["floor"]+3 or self.wire.done or not self.wire.spare()>CLOSING+30:
    return
   if not self.walk_call({**line["args"],job["key"]:self.next_page(line)},"Going on to the line search's next page."):
    return
 def _smp_walk_ceiling(self,job):
  floor,repair=job["floor"],job["repair"]
  def margin():
   return(floor if repair else max(floor,self.order_due()+2))+1
  for n in dict.fromkeys(job["ceil"]or[]):
   if self._smp_short(margin):
    break
   got=self.ceiling_walk(n,floor,repair)
   if got:
    job["ran"]=True
    job["report"].append(got[0])
    job["ceil_rows"][n]=list(got[1])
  return"queue"if job["ran"]else None
 @staticmethod
 def _smp_interleave(pools,most):
  taken=[]
  while len(taken)<most and any(pools):
   for pool in pools:
    fresh=next((n for n,i in enumerate(pool)if i not in taken),len(pool))
    del pool[:fresh]
    if pool and len(taken)<most:
     taken.append(pool.pop(0))
  return taken
 def _smp_walk_queue(self,job):
  batch=self.batch_tool()
  most=batch[2]if batch else(CEIL_OPENS if job["ceil_rows"]else 2)
  planned=list(dict.fromkeys(n for n,_,_ in job["plan"]))
  twinned=[n for n in job["twin_keys"]if n not in planned]
  laddered=[n for n in job["rung_keys"]if n not in planned+twinned]
  pools=[self.walk_rows(n,[job["lines"].get(n,""),job["twin_keys"].get(n,"")],most)for n in planned+twinned]
  pools+=[self.walk_rows(n,job["rung_keys"][n],min(most,RUNG_OPENS),read_bar=False)for n in laddered]
  queue=self._smp_interleave(pools+list(job["ceil_rows"].values()),most)
  while queue:
   cost=1 if batch and len(queue)>=2 else len(queue)
   if cost<self.moves_left()-job["floor"]:
    break
   queue.pop()
  job["queue"]=queue
  return"close"
 def _smp_walk_close(self,job):
  queue=job["queue"]
  walked=list(dict.fromkeys(n for n,_,_ in job["plan"]))+list(job["ceil_rows"])+list(job["twin_keys"])
  walked+=list(job["rung_keys"])
  reset=set()
  for n in walked:
   reset|=self.walk_home(n)
  self.rank_checked=set(filter(lambda seal:self.shelf_known(seal.partition("|")[0])not in reset,self.rank_checked))
  self.order_vetted=False
  if queue:
   listed=", ".join(f"{i} ({self.archive.price(i):g})"for i in queue)
   middle=("Opened now, cheapest first: the unread rows from these searches that were in stock, affordable, and "
"under each pick: "+listed+". ")
  else:
   middle="No row they list here is unread, in stock, affordable, and under the pick. "
  summary=("[cheapest walk] "+"; ".join(job["report"])+". "+middle+"Settle each product on its least expensive "
"listing whose own fields fulfil all of the rules that apply; ranking is rerun later at each cart move and at the order.")
  if queue:
   for attr,value in(("open_queue",queue),("open_recheck",True),("open_purpose","walk"),("open_floor",job["floor"]),
("open_say","Lowest price wanted: now opening the cheapest unread rows the walk located.")):
    setattr(self,attr,value)
   self.open_named()
  last=self.chat[-1]if self.chat else None
  if last is not None and last.get("role")=="tool":
   tail=summary if queue else summary+"\n"+self.status_line()
   last["content"]="%s\n%s"%(last["content"],tail)
  record={"kind":"walk","plan":[[n,pages]for n,_,pages in job["plan"]]}
  record.update(ceiling=list(job["ceil_rows"]),opened=queue)
  self.log.append(record)
  job["ok"]=True
 def decline_fits(self):
  named=set(self._smp_attr("audit_fits")or[])-self._smp_struck()
  return(self._smp_cleared()|named)-self.refused_lines
 def decline_lacking(self):
  have=self._smp_shelves_of(self.decline_fits())
  if self.needs.items():
   return[n for n,shelf in enumerate(self.home_shelves())if shelf and not shelf&have]
  return[None]if not set(self.shelf_labels())&have else[]
 def decline_walks(self):
  plain={n for n,req in enumerate(self.needs.props_left())if not self._smp_by_ref.search(req.lower())}
  def worded(entry,k):
   return len(self.walk_carries(entry["args"],k)&plain)
  chosen=[]
  for k in self.decline_lacking():
   scored=[]
   for pages,name,entry in self.walk_options(k):
    hits=worded(entry,k)
    if hits:
     scored.append((-hits,-self.walk_score(entry["args"],k),pages,name))
   scored.sort()
   fresh=next((name for*_,name in scored if name not in chosen),None)
   if fresh is not None:
    chosen.append(fresh)
  return chosen[:2]
 _smq_KEEP=6
 def _smq_fault(self,kind,err):
  self.log.append(dict(kind=kind,error=("%s: %s"%(type(err).__name__,err))[:300]))
 def _smq_census(self):
  label,gone={},{}
  known=self.archive.ids
  def note(blob):
   lid=self.row_mark(blob)
   where=self.shelf_of(blob)if lid in known else""
   if where:
    label[lid]=where
    if any(re.search(r"stock|avail",str(fname).lower())for fname in blob):
     gone[lid]=self.sold_out(blob)
  for entry in self.archive.items:
   walk_dicts(entry,note)
  return label,gone
 def row_pools(self,homes,ids,below=None):
  cap=self.needs.spend_cap
  label,gone=self._smq_census()
  store=self.archive
  out=[]
  for slot,(home,units)in enumerate(homes):
   drops=(lambda lid,where,cost:where not in home,
lambda lid,where,cost:lid in store.read,
lambda lid,where,cost:lid in self.refused_lines,
lambda lid,where,cost:gone.get(lid),
lambda lid,where,cost:ids is not None and lid not in ids,
lambda lid,where,cost:not is_number(cost),
lambda lid,where,cost:is_number(cap)and cost*units>cap,
lambda lid,where,cost:below is not None and not cost<below[slot])
   ranked=[]
   for lid,where in label.items():
    cost=store.price(lid)
    if not any(drop(lid,where,cost)for drop in drops):
     ranked.append((float(cost),lid))
   ranked.sort()
   out.append([lid for _,lid in ranked])
  return out
 def over_budget_rows(self):
  try:
   lines=self.needs.items()
   cap=self.needs.spend_cap
   if not lines or self.needs.priority()=="cheapest"or not is_number(cap):
    return[]
   shelves=self.home_shelves()
   fits=self.decline_fits()
   lows,weight=[],[]
   spent=0.0
   for slot,shelf in enumerate(shelves):
    costs=[]
    for thing in fits:
     if self.shelf_known(thing)not in shelf:
      continue
     seen=self.archive.price(thing)
     if is_number(seen):
      costs.append(float(seen))
    if not(shelf and costs):
     return[]
    lows.append(min(costs))
    weight.append(lows[-1]*lines[slot]["quantity"])
    spent+=weight[-1]
   if spent<=cap:
    return[]
   found=self.row_pools([(shelf,lines[slot]["quantity"])for slot,shelf in enumerate(shelves)],None,lows)
   order=sorted(range(len(found)),key=lambda slot:-weight[slot])
   found=[found[slot]for slot in order]
   return found if any(found)else[]
  except Exception as err:
   self._smq_fault("over_budget_error",err)
   return[]
 def _smq_decline_note(self,searches,room,cheaper):
  parts=[]
  if searches:
   walks=self.__dict__.get("page_book",{})
   asked="; ".join("'"+" ".join(self.free_text(walks[s]["args"]))+"'"for s in searches)
   parts.append("[pages] There are unread result pages for search texts that hold a required value ("+asked
+"). This shop sorts rows by id, never by match or cost, so a good one is just as likely to hide "
"past the pages already seen; up to "+str(room)+" further pages get fetched at once, and the "
"lowest-priced affordable rows on them that still have stock get opened")
  if cheaper:
   parts.append("[cheaper rows] Summed, the lowest-priced fits per product overshoot the allowance; stocked rows "
"nobody has opened that undercut each one's pick get opened as well")
  buy="one apiece in quantities as requested"if self.needs.items()else"the item"
  return("Decline held for code reads. "+"; ".join(parts)+". Then weigh each record on sense against the active "
"requirement list; buy one which passes ("+buy+"); otherwise decline.")
 def _smq_queue_decline_reads(self):
  lister=self.batch_tool()
  shut=(lambda:self.__dict__.get("decline_read_done"),
lambda:self.wire.spare()<CLOSING+60,
lambda:not(self.role("details")or lister),
lambda:self.trolley.units)
  if any(test()for test in shut):
   return""
  spend=2-bool(lister)+self._smq_KEEP
  room=min(3,self.moves_left()-spend)
  paged=room>0 and self.page_arg()and self.role("search")and self.id_store()
  searches=self.decline_walks()if paged else[]
  cheaper=self.over_budget_rows()if self.moves_left()>spend else[]
  if not searches and not cheaper:
   return""
  self.__dict__.update(decline_read_done=True,decline_queue=(searches,room,self._smq_KEEP))
  return self._smq_decline_note(searches,room,cheaper)
 def decline_reads(self):
  try:
   return self._smq_queue_decline_reads()
  except Exception as err:
   self._smq_fault("decline_reads_error",err)
   return""
 def run_decline_reads(self):
  searches,page_cap,keep=self.__dict__.pop("decline_queue")
  paging=self.page_arg()
  lister=self.batch_tool()
  job={"todo":list(searches)if paging else[],"paging":paging,"book":self.__dict__.get("page_book",{}),
"batch":lister,"opens":2-bool(lister),"limit":page_cap,"keep":keep,"read":0,"walked":[],
"walk":None,"query":"","queue":[]}
  state="next"
  while state!="end":
   state=self._smq_DECLINE_FLOW[state](self,job)
  return job["read"]>0 or bool(job["queue"])
 def _smq_dr_next(self,job):
  while job["todo"]:
   sig=job["todo"].pop(0)
   plan=job["book"].get(sig)
   if plan:
    job["walked"].append(sig)
    job["walk"],job["query"]=plan," ".join(self.free_text(plan["args"]))
    return"page"
  return"pick"
 def _smq_dr_page(self,job):
  room=(lambda:job["read"]<job["limit"],lambda:self.moves_left()>job["keep"]+job["opens"],
lambda:not self.wire.done,lambda:not self.wire.broke,lambda:self.wire.spare()>CLOSING+30)
  if not all(ok()for ok in room):
   return"next"
  plan=job["walk"]
  token=self.next_page(plan)
  if token is None:
   return"next"
  said="Paging on for '"+job["query"]+"' first, since this shop sorts by id and a match may come later."
  if not self.walk_call({**plan["args"],job["paging"][0]:token},said):
   return"next"
  job["read"]+=1
  return"page"
 @staticmethod
 def _smq_round_robin(pools):
  given=set()
  heads=[0]*len(pools)
  while True:
   moved=False
   for n,pool in enumerate(pools):
    while heads[n]<len(pool)and pool[heads[n]]in given:
     heads[n]+=1
    if heads[n]<len(pool):
     moved=True
     given.add(pool[heads[n]])
     yield pool[heads[n]]
     heads[n]+=1
   if not moved:
    return
 def _smq_dr_pick(self,job):
  shown=set()
  for sig in job["walked"]:
   shown.update((job["book"].get(sig)or{}).get("row_shelf")or{})
  lines=self.needs.items()
  wanted=[]
  for slot in self.decline_lacking():
   home=self.walk_home(slot)
   wanted.append((home,lines[slot]["quantity"]if slot is not None and slot<len(lines)else self.needs.quantity()))
  pools=self.row_pools(wanted,shown)if job["walked"]else[]
  pools+=self.over_budget_rows()
  cap=job["batch"][2]if job["batch"]else 2
  feed=self._smq_round_robin(pools)
  done=object()
  while len(job["queue"])<cap:
   lid=next(feed,done)
   if lid is done:
    break
   job["queue"].append(lid)
  return"fit"
 def _smq_dr_fit(self,job):
  lids=job["queue"]
  while lids:
   cost=1 if job["batch"]and len(lids)>=2 else len(lids)
   if cost<self.moves_left()-job["keep"]:
    break
   lids.pop()
  self.log.append(dict(kind="decline_reads",pages=job["read"],opened=lids))
  return"open"
 def _smq_dr_open(self,job):
  if job["queue"]and not self.wire.done:
   self.__dict__.update(open_queue=job["queue"],open_purpose="decline",open_floor=job["keep"],
open_say="Opening, ahead of a decline, the lowest-priced affordable rows nobody has read "
"yet from the new pages and the picks.")
   self.open_named()
   return"end"
  tail=self.chat[-1]if self.chat else None
  if tail is not None and tail.get("role")=="tool":
   tail["content"]="%s\nNothing worth opening came of it: no stocked, in-cap row.\n%s"%(tail["content"],self.status_line())
  return"end"
 _smq_DECLINE_FLOW={"next":_smq_dr_next,"page":_smq_dr_page,"pick":_smq_dr_pick,"fit":_smq_dr_fit,
"open":_smq_dr_open}
 @staticmethod
 def _smq_plain(x):
  return isinstance(x,(str,int,float))and not isinstance(x,bool)
 @staticmethod
 def _smq_uncamel(name):
  return re.sub(r"([a-z])([A-Z])",r"\1 \2",str(name))
 @staticmethod
 def _smq_prop_head(req):
  front,_,said=str(req).partition(":")
  tags=re.findall(r"\(([^)]*)\)",front)
  return front,front.split("(",1)[0].strip(),(tags[-1]if tags else"").strip(),said
 def ref_register(self):
  reading=self.needs.reading
  pool=reading.get("reference_targets")
  entries=[t for t in pool if isinstance(t,dict)]if isinstance(pool,list)else[]
  named=reading.get("references")
  listed=set(map(str,named or()))if isinstance(named,list)else set()
  for text in map(str,list(self.needs.written)):
   tail=text.partition(":")[2]
   pointing=ref_text(tail)or(text in listed and re.search(DIFFER_WORDS,tail.lower()))
   if pointing:
    self.ref_add(text,entries,self.request,"request")
 def ref_untracked(self,refs,tracked):
  try:
   live=[rec for rec in tracked if rec.get("status")=="resolved"or rec.get("chain")]
   return[one for one in refs if all(not self.ref_match(rec,one)for rec in live)]
  except Exception:
   return refs
 def _smq_restore_resolved(self):
  for rec in self.refs:
   if rec["status"]=="resolved":
    now=rec["now"]
    if any(p!=now and self.ref_match(rec,p)for p in self.needs.props_left()):
     self.ref_swap(rec,now)
 def _smq_take_news(self):
  news=list(self.needs.__dict__.get("ref_news")or[])
  self.needs.ref_news=[]
  for item in news:
   need,entries,line=(item.get(k)for k in("need","infos","line"))
   opened=self.ref_add(str(need or""),list(entries or[]),str(line or""),"reply")
   if not opened:
    continue
   reading=self.needs.reading
   if not isinstance(reading.get("references"),list):
    reading["references"]=[]
   if opened["req"]not in reading["references"]:
    reading["references"].append(opened["req"])
 def ref_take(self):
  try:
   self._smq_restore_resolved()
   self._smq_take_news()
  except Exception as err:
   self._smq_fault("ref_take_error",err)
 _smq_DIFFER_TOLD="|".join(("differ","not the same","exclu","apart","other than","unlike"))
 def _smq_relation(self,said,entry,aim,prop,source):
  found=ref_relation(said)
  given=entry if isinstance(entry,dict)else{}
  told=str(given.get("relation")or"").lower()
  lead=ref_span(prop,source)
  who_at=lead.lower().find(str(aim.get("seller")or"").lower())if str(aim.get("seller")or"").strip()else-1
  lead=lead[:who_at]if who_at>0 else lead
  pointer=_SMB_REF.search(lead.lower())
  lead=lead[:pointer.end()]if pointer else lead
  if found[0]=="pairs"or told=="pairs"or re.search(PAIR_WORDS,lead.lower()):
   return"pairs",None,""
  if found[0]=="same"and given and told=="within"and is_number(given.get("tolerance")):
   found=("within",float(given["tolerance"]),str(given.get("unit")or""))
  span=ref_span(prop,source).lower()
  who=str(aim.get("seller")or"")
  cut=span.find(who.lower())if who.strip()else-1
  if found[0]!="same":
   return found
  if re.search(self._smq_DIFFER_TOLD,told)or(cut>0 and re.search(DIFFER_WORDS,span[:cut])):
   return"differ",None,""
  return found
 def _smq_ref_info(self,entries,prop,tag,tag_words):
  for entry in entries:
   if not same_prop(str(entry.get("property")or""),prop):
    continue
   product=entry.get("product")
   if not tag or not product or not self.needs.items()or kind_words(product)&tag_words:
    return entry
  return None
 def ref_add(self,req,infos,source,origin):
  front,prop,tag,said=self._smq_prop_head(req)
  if not prop or junk_key(prop):
   return None
  tag_words=kind_words(tag)
  clash=lambda rec:rec["req"]==req or(same_prop(rec["prop"],prop)and kind_words(rec["tag"])==tag_words)
  if any(clash(rec)for rec in self.refs):
   return None
  entry=self._smq_ref_info(infos,prop,tag,tag_words)
  aim=self.ref_target(entry,said,source,prop)
  text=str(req)
  if not aim:
   self.log.append(dict(kind="ref_skip",req=text[:200]))
   return None
  rel,tol,unit=self._smq_relation(said,entry,aim,prop,source)
  fresh=dict(req=text,head=front.strip(),prop=prop,tag=tag,relation=rel,tol=tol,unit=unit,target=aim,
first=dict(aim),chain=[],seen={self.ref_key(aim)},judged=set(),queries=[],pages=0,hops=0,
tries=0,fails=0,status="open",value="",goes_with=[],now="",origin=origin,unindexed=False)
  self.refs.append(fresh)
  self.log.append(dict(kind="ref_open",req=text[:200],target=aim,relation=rel,origin=origin))
  return fresh
 def ref_target(self,entry,said,source,prop):
  given=entry if isinstance(entry,dict)else{}
  who=str(given.get("seller")or"").strip()
  sort=str(given.get("kind")or"").strip()
  raw=given.get("details")
  bits=[str(b).strip()for b in(raw if isinstance(raw,list)else())if self._smq_plain(b)and str(b).strip()]
  writes=lambda nm:self.ref_seller_ok(nm,source)
  tries=(ref_parse(text,writes)for text in(ref_span(prop,source),said))
  hit=next((p for p in tries if p and writes(p["seller"])),None)
  if writes(who):
   if hit and not bits and ref_name(hit["seller"])==ref_name(who):
    bits=hit["details"]
  elif hit:
   who,sort,bits=hit["seller"],sort or hit["kind"],bits or hit["details"]
  else:
   return None
  heard=set(ref_toks(source))
  kept=[b for b in dict.fromkeys(bits)if heard.intersection(ref_toks(b))]
  sort=(sort or hit["kind"])if hit else sort
  return{"seller":who,"kind":sort[:80],"details":kept[:6]}
 def _smq_kind_vocab(self):
  common=set()
  asked=self.needs.items()or[{"product":str(self.needs.reading.get("product")or"")}]
  for line in asked:
   common.update(kind_words(line["product"]))
  spec=self.shapes.get(self.role("search")or"")or{}
  for field in(spec.get("properties")or{}).values():
   choices=field.get("enum")if isinstance(field,dict)else None
   for choice in choices if isinstance(choices,list)else():
    common.update(kind_words(choice))
  return common
 def ref_seller_ok(self,seller,source):
  nm=str(seller or"").strip()
  if sum(ch.isascii()and ch.isalpha()for ch in nm)<3:
   return False
  bounded=r"(?<![a-z0-9])%s(?:'s)?(?![a-z0-9])"%re.escape(nm.lower())
  if re.search(bounded,str(source or"").lower())is None:
   return False
  common=self._smq_kind_vocab()
  own=kind_words(nm)
  return bool(own)and not own.issubset(common)
 @staticmethod
 def ref_key(tgt):
  bag=set()
  for bit in tgt.get("details")or():
   bag.update(ref_toks(bit))
  return ref_name(tgt.get("seller")or""),tuple(sorted(bag))
 @staticmethod
 def ref_desc_of(tgt):
  words=["the",str(tgt.get("kind")or"listing"),"sold by",str(tgt.get("seller"))]
  if tgt.get("details"):
   words.append("(%s)"%"; ".join(tgt["details"]))
  return" ".join(words)
 def ref_desc(self,ref):
  return Errand.ref_desc_of(ref["target"])
 def ref_index(self):
  recs=self.archive.items
  memo=self.__dict__.get("ref_memo")
  if memo and memo[0]==len(recs):
   return memo[1]
  hits=[]
  for at,entry in enumerate(recs):
   walk_dicts(entry,lambda blob,at=at:hits.append((at,blob)))
  merged,last={},{}
  for at,blob in hits:
   lid=self.row_mark(blob)
   full=len(blob)>3 or any(isinstance(sub,dict)and len(sub)>1 for sub in blob.values())
   if full and lid in self.archive.ids:
    merged.setdefault(lid,{}).update(blob)
    last[lid]=at
  self.ref_memo=(len(recs),merged)
  self.ref_seen_at=last
  return merged
 _smq_NOT_SELLER=r"note|text|desc|comment|detail|info|message|section"
 def ref_sellers(self,row):
  learned=self.seller_keys()
  def holds_seller(fname):
   low=str(fname).lower()
   named=re.search(SELLER_KEYS,low)or str(fname)in learned
   return named and"id"not in low and not re.search(self._smq_NOT_SELLER,low)
  names={ref_name(txt)for fname,txt in row.items()
if isinstance(txt,str)and txt.strip()and len(txt)<=60 and holds_seller(fname)}
  return names-{""}
 _smq_NOT_SELLER_FIELD=(r"title|heading|name|label|caption|categor|shelf|department|aisle|note|text|desc|comment|detail|"
r"info|message|section|url|image|summary|overview")
 @staticmethod
 def _smq_caps_spans(sentence):
  spans=set()
  runs=[[]]
  for word in re.findall(r"[A-Za-z][A-Za-z'-]*",sentence)[1:]:
   if word[:1].isupper()and re.search(r"[a-z]",word):
    runs[-1].append(word)
   elif runs[-1]:
    runs.append([])
  for run in runs:
   for lo in range(len(run)):
    for hi in range(lo+1,min(len(run),lo+3)+1):
     spans.add(ref_name(" ".join(run[lo:hi])))
  return spans
 def _smq_known_sellers(self):
  out=set()
  for rec in self.__dict__.get("refs")or[]:
   for spot in(rec.get("target")or{},rec.get("first")or{}):
    if isinstance(spot,dict):
     out.add(ref_name(spot.get("seller")))
  out.discard("")
  return out
 def _smq_noted_names(self,cards):
  heard,map_vals=set(),set()
  for card in cards:
   for sub in card.values():
    if isinstance(sub,dict):
     map_vals.update(ref_name(x)for x in sub.values()if isinstance(x,str))
   prose=self.ref_texts(card,desc=False)
   for piece in re.split(r"(?<=[.!?;])\s+",prose):
    heard|=self._smq_caps_spans(piece)
  return{nm for nm in heard-map_vals if len(nm)>3}
 def seller_keys(self):
  try:
   sellers=self._smq_known_sellers()
   stamp=len(self.archive.items),tuple(sorted(sellers))
   memo=self.__dict__.get("seller_memo")
   if memo and memo[0]==stamp:
    return memo[1]
   cards=list(self.ref_index().values())
   heard=self._smq_noted_names(cards)
   def seller_field(fname,fval):
    low=str(fname).lower()
    if not isinstance(fval,str)or not fval or len(fval)>60:
     return False
    if"id"in low or re.search(self._smq_NOT_SELLER_FIELD,low):
     return False
    short=ref_name(fval)
    if short in sellers:
     return True
    return fval.strip()[:1].isupper()and len(fval.split())<4 and short in heard
   found=set()
   for card in cards:
    found.update(str(fname)for fname,fval in card.items()if seller_field(fname,fval))
   self.seller_memo=(stamp,found)
   return found
  except Exception:
   return set()
 @staticmethod
 def ref_texts(row,desc=True):
  bins={"note":[],"about":[]}
  for fname,txt in row.items():
   low=str(fname).lower()
   if not isinstance(txt,str)or len(txt.split())<3 or low.endswith("id"):
    continue
   if re.search(r"title|heading|headline|name|label|caption|categor|shelf|department|url|image",low):
    continue
   bins["about"if re.search(r"desc|overview|summary|about",low)else"note"].append(txt)
  chosen=bins["note"]or(bins["about"]if desc else[])
  return"\n".join(chosen)
 _smq_NEGATED=r"\b(?:not|no|never|without|nor)\b(?:\W+[A-Za-z0-9.:/&'-]+){1,3}"
 _smq_FLAT_SKIP=r"id$|title|shop|seller|brand|price|amount|desc|overview|note"
 _smq_FUSED_WORD=re.compile(r"\d+(?:[.,:/]\d+)*([a-z]{3,})")
 def _smq_maps(self,row):
  return[sub for sub in row.values()if isinstance(sub,dict)]or[
{fname:x for fname,x in row.items()if self._smq_plain(x)and not re.search(self._smq_FLAT_SKIP,str(fname).lower())}]
 @staticmethod
 def _smq_value_words(words):
  return[t for t in words if len(t)>=3 and not re.search(r"\d",t)]
 def _smq_led_value(self,row,words):
  longest=0
  for fields in self._smq_maps(row):
   for fname in fields:
    lead=self._smq_value_words(ref_toks(self._smq_uncamel(fname).replace("_"," ")))
    if len(lead)>longest and words[:len(lead)]==lead:
     longest=len(lead)
  return set(words[longest:])if longest else set()
 def ref_parts(self,row):
  prose=re.sub(self._smq_NEGATED," ",self.ref_texts(row,desc=False),flags=re.I)
  said=[self.title_of(row),prose]
  maps=self._smq_maps(row)
  heads=[]
  for fields in maps:
   heads+=[self._smq_uncamel(fname).replace("_"," ")for fname in fields]
   said+=[str(x)for x in fields.values()if self._smq_plain(x)]
  shown=set(ref_toks(" | ".join(said)))
  shown|={hit.group(1)for hit in map(self._smq_FUSED_WORD.fullmatch,shown)if hit}
  return set(ref_toks(" ".join(heads))),shown
 def _smq_units_agree(self,row,figs):
  own=[self.title_of(row)]
  for sub in filter(lambda x:isinstance(x,dict),row.values()):
   own.extend(str(x)for x in sub.values()if isinstance(x,(str,int,float)))
  written=ref_figs(" | ".join(own))
  for fig,unit in figs:
   same=[num for num,u in written if u==unit]
   if same and not any(abs(num-fig)<1e-9 for num in same):
    return False
  return True
 def ref_shows(self,row,detail,parts=None):
  heads,said=parts or self.ref_parts(row)
  words=self._smq_value_words(ref_toks(detail))
  want=set(words)-heads
  if not want and len(words)>1:
   want=self._smq_led_value(row,words)
  if want and want.isdisjoint(said):
   return False
  figs=ref_figs(detail)
  return self._smq_units_agree(row,figs)if figs else True
 @staticmethod
 def ref_field(row,prop):
  filled=[(fname,str(x).strip())for sub in row.values()if isinstance(sub,dict)for fname,x in sub.items()
if Errand._smq_plain(x)and str(x).strip()]
  for match in(strict_prop,same_prop):
   hit=next((txt for fname,txt in filled if match(fname,prop)),None)
   if hit is not None:
    return hit
  return""
 def ref_ids_add(self,ids):
  self.ref_ids=set(self.__dict__.get("ref_ids")or()).union(ids)
 def ref_foreign(self):
  try:
   asked=self.__dict__.get("ref_ids")or set()
   shelves=self.home_shelves()if asked else[]
   if not(shelves and all(shelves)):
    return set()
   homes=set().union(*shelves)
   index=self.ref_index()
   carted=set(self.trolley.units)
   def away(lid):
    where=self.shelf_of(index.get(lid)or{})
    return bool(where)and where not in homes
   return{lid for lid in asked if lid not in carted and away(lid)}
  except Exception:
   return set()
 def ref_cands(self,ref):
  aim=ref["target"]
  who=ref_name(aim["seller"])
  sorts=kind_words(aim["kind"])
  index=self.ref_index()
  newest=self.__dict__.get("ref_seen_at")or{}
  wanted=[set(ref_toks(bit))for bit in aim["details"]]
  board=[]
  for lid,card in index.items():
   if who not in self.ref_sellers(card):
    continue
   blurb=[str(txt)for fname,txt in card.items()if isinstance(txt,str)and re.search(r"desc|overview|summary",str(fname).lower())]
   hay=set(ref_toks(" ".join([self.title_of(card)]+blurb)))
   hits=len([1 for bag in wanted if bag and bag.issubset(hay)])
   near=2 if sorts&kind_words(self.shelf_of(card))else 0
   board.append((-near-hits,-newest.get(lid,0),lid,hits))
  board.sort()
  ref["shows"]={entry[2]:entry[3]for entry in board}
  return[entry[2]for entry in board]
 @staticmethod
 def _smq_unlabel(call,must,shelfset):
  def is_label(val):
   return isinstance(val,str)and val.strip().lower()in shelfset
  return{arg:val for arg,val in call.items()if arg in must or not is_label(val)}
 def _smq_lone_text(self,call,texts):
  slots=[arg for arg,val in call.items()if isinstance(val,str)and val in texts and val.strip()]
  return slots[0]if len(texts)==1 and len(slots)==1 else None
 def ref_modes(self,needed,labels):
  groups={}
  for at,plan in enumerate(self.__dict__.get("page_book",{}).values()):
   call=plan.get("args")
   call=call if isinstance(call,dict)else{}
   texts=self.free_text(call)
   slot=self._smq_lone_text(call,texts)
   if slot is None:
    continue
   fixed=self._smq_unlabel({arg:val for arg,val in call.items()if arg!=slot},needed,labels)
   sig=json.dumps(fixed,sort_keys=True,default=str)
   group=groups.get(sig)
   if group is None:
    group=groups[sig]={"tpl":(fixed,slot),"seen":[],"at":at}
   stems={self.stem(tok)for tok in re.findall(r"[a-z0-9]+",texts[0].lower())}
   group["seen"].append((stems,tuple(plan.get("first")or())))
  def blind_group(seen):
   for n,(w1,r1)in enumerate(seen):
    if any(len(r1)>=2 and r1==r2 and not w1&w2 for w2,r2 in seen[n+1:]):
     return True
   return False
  blind={sig for sig,group in groups.items()if blind_group(group["seen"])}
  live=[(-len({tuple(sorted(stems))for stems,_ in group["seen"]}),group["at"],group["tpl"])
for sig,group in groups.items()if sig not in blind]
  live.sort(key=lambda row:row[:2])
  return[row[2]for row in live],blind
 def _smq_sent_searches(self):
  finder=lambda:self.role("search")
  for entry in reversed(self.log):
   if entry.get("kind")=="env":
    for act in reversed(entry.get("actions")or[]):
     call=act.get("args")if act.get("name")==finder()else None
     if isinstance(call,dict):
      yield call
 def _smq_logged_search(self,must,shelfset,blind):
  paging=self.page_arg()
  for call in self._smq_sent_searches():
   slots=[arg for arg,val in call.items()if isinstance(val,str)and val.strip()and val in self.free_text(call)]
   if len(slots)!=1:
    continue
   rest={arg:val for arg,val in call.items()if arg!=slots[0]and not(paging and arg==paging[0])}
   if json.dumps(self._smq_unlabel(rest,must,shelfset),sort_keys=True,default=str)not in blind:
    return rest,slots[0]
  return None
 def ref_template(self):
  spec=self.shapes.get(self.role("search")or"")or{}
  listed=spec.get("required")
  must=set(listed or[])if isinstance(listed,list)else set()
  shelfset=set(map(self.shelf_of,self.ref_index().values()))
  shelfset.discard("")
  modes,blind=self.ref_modes(must,shelfset)
  mode=modes[0]if modes else self._smq_logged_search(must,shelfset,blind)
  if mode:
   return self._smq_unlabel(mode[0],must,shelfset),mode[1]
  fields=spec.get("properties")
  paging_tok=self.token_key()
  free=[arg for arg,field in(fields.items()if isinstance(fields,dict)else())
if isinstance(field,dict)and field.get("type")=="string"and not field.get("enum")and arg!=paging_tok]
  if len(free)==1 and must<={free[0]}:
   return{},free[0]
  return None
 _smq_EDGE=".,;:!?()[]{}\"'"
 def _smq_field_words(self):
  heads=[]
  for card in self.ref_index().values():
   for sub in filter(lambda x:isinstance(x,dict),card.values()):
    heads+=[self._smq_uncamel(fname)for fname in sub]
  heads+=[self._smq_prop_head(req)[1]for req in self.needs.written+self.needs.uncovered]
  return{tok for text in heads for tok in re.findall(r"[a-z]+",text.lower())}
 @staticmethod
 def _smq_word_bin(tok,after_figure):
  digit=re.search(r"\d",tok)is not None
  if after_figure and not digit:
   return None
  if digit:
   return"joined"if re.search(r"[a-z]",tok)else("figure"if len(tok)>1 else None)
  return"word"
 def ref_words(self,ref):
  known=self._smq_field_words()
  aim=ref["target"]
  owner=set(re.findall(r"[a-z0-9]+",str(aim["seller"]).lower()))
  bins={"joined":[],"figure":[],"word":[]}
  for bit in aim["details"]:
   prior=False
   for chunk in str(bit).lower().split():
    for tok in re.findall(r"[a-z0-9]+",chunk):
     if tok in LINK_WORDS or tok in owner:
      continue
     kind=self._smq_word_bin(tok,prior)
     if kind=="word"and not(len(tok)>2 and tok==chunk.strip(self._smq_EDGE)
and tok not in known and tok.rstrip("s")not in known):
      kind=None
     if kind:
      bins[kind].append(tok)
    prior=re.fullmatch(r"\d+(?:[.,]\d+)?",chunk)is not None
  ranked=sorted(bins["word"],key=lambda tok:-len(tok))
  fields=self.store_keys()
  pieces=[[tok for tok in re.findall(r"[a-z0-9]+",str(bit).lower())if tok not in LINK_WORDS]for bit in aim["details"]]
  all_pairs=ref.get("hops",0)==0 and bool(pieces)and all(len(seq)>1 for seq in pieces)
  def value_first(seq):
   if len(seq)<2 or seq[-1]not in ranked:
    return False
   return all_pairs or any(" ".join(seq[:j])in fields for j in range(1,len(seq)))
  lead=[seq[-1]for seq in pieces if value_first(seq)]
  ranked=lead+[tok for tok in ranked if tok not in lead]
  return list(dict.fromkeys(bins["joined"]+bins["figure"]+ranked))
 _SMR_SENTENCE=r"(?<=[.!?;])\s+"
 _SMR_WORD=r"[A-Za-z][A-Za-z'-]*"
 _SMR_NEGATORS=frozenset(("not","nor","never","no","neither"))
 _SMR_READ_SAY="Now opening the rows that could be the page %s is drawn from, per the shopper."
 _SMR_SEARCH_SAY="Store query for the very page that %s is pegged to, namely [%s]."
 _SMR_FACTS_SAY="Reading with %s the property sheet of the page %s is drawn from, since its record shows no such field."
 _SMR_NOTES_SAY=("Fetching with %s the remarks of a page that might be the source of %s: such text can pass the pointer "
"on to a further page, in turn.")
 _SMR_POINTER=("%s belongs to the pointer chain for %s (the source page itself, or one its remarks route via), thus its "
"%s carries exactly the value%s to be avoided by the wanted %s; it can never qualify")
 _SMR_LISTS="%s lists %s in its %s, the value the wanted %s has to go with rather than be; it can never qualify"
 _SMR_REMARK=("%s: the remarks on it might move %s to a separate page. If so, its own field proves nothing for %s "
"(grading reads that other page instead); it satisfies %s only where that page's record shows the "
"value; a fresh item check comes before any order%s")
 @staticmethod
 def _smr_err(err):
  return("%s: %s"%(type(err).__name__,err))[:300]
 @staticmethod
 def _smr_prop(text):
  return re.split(r"\s*\(",str(text).partition(":")[0])[0]
 @staticmethod
 def _smr_key_name(key):
  seamed=re.sub(r"(?<=[a-z])(?=[A-Z])"," ",str(key)).lower()
  return" ".join(re.findall(r"[a-z0-9]+",seamed))
 def _smr_note(self,kind,**facts):
  self.log.append({"kind":kind,**facts})
 def _smr_of_tag(self,tag):
  return self.product_of(f"x ({tag})")
 def _smr_links(self):
  return self.__dict__.get("refs")or[]
 def _smr_open(self):
  return[piece for piece in self._smr_links()if piece.get("status")=="open"]
 def _smr_booked(self,start):
  used=self.wire.tick-start
  self._smr_steps=self.__dict__.get("_smr_steps",0)+used
  return used>0
 def store_keys(self):
  records=self.archive.items
  cached=self.__dict__.get("_smr_keys_at")
  if cached and cached[0]==len(records):
   return cached[1]
  maps=[]
  for record in records:
   walk_dicts(record,maps.append)
  names=set()
  for node in maps:
   for inner in node.values():
    if isinstance(inner,dict)and len([piece for piece in inner.values()if isinstance(piece,(str,int,float))])>=2:
     names|=set(filter(None,map(self._smr_key_name,inner)))
  self._smr_keys_at=(len(records),names)
  return names
 def ref_plans(self,link):
  aim=link["target"]
  room=max(2,self.lead_words or 3)
  shop=" ".join(re.findall(r"[A-Za-z0-9]+",re.sub(r"'s\b","",str(aim["seller"]))))
  detail=self.ref_words(link)
  noun=[w for w in re.findall(r"[a-z]{3,}",str(aim["kind"]).lower())if w not in LINK_WORDS][-1:]
  if link.get("unindexed"):
   lead=(detail[:1]+noun)[:room]
   drafts=[" ".join(lead)]if lead else[]
  elif detail and len(shop.split())<room:
   drafts=[f"{shop} {detail[0]}",shop]
  else:
   drafts=[shop]
  return[q for q in drafts if q.strip()and q.lower()not in link["queries"]]
 def ref_page(self,link):
  paging=self.page_arg()
  if not paging or link["pages"]>=3:
   return None
  shop=ref_name(link["target"]["seller"])
  if not shop:
   return None
  walks=list(self.__dict__.get("page_book",{}).values())
  while walks:
   walk=walks.pop()
   if shop in ref_name(" ".join(self.free_text(walk["args"]))):
    after=self.next_page(walk)
    if after is not None:
     return{**walk["args"],paging[0]:after}
  return None
 def ref_room(self):
  live=self._smr_open()
  tally=self.__dict__.get("_smr_steps",0)
  differ_only=bool(live)and all(piece.get("relation")=="differ"for piece in live)
  if differ_only and tally>=REF_DIFFER_BLIND and not any(piece.get("chain")for piece in live):
   return False
  floor=self.ref_floor()
  limit=max(6,int(REF_DIFFER_STEPS*self.step_cap))if differ_only else max(8,int(REF_STEPS*self.step_cap))
  margin=60
  if any(piece.get("chain")for piece in live)or self.ref_solo():
   limit=self.step_cap
   margin=10
  wire=self.wire
  if any((tally>=limit,wire.spare()<=CLOSING+margin,wire.spent_out(),wire.done,wire.broke)):
   return False
  if self.ref_waits(floor):
   return False
  return self.moves_left()>floor or self.ref_hop_room()
 def ref_floor(self):
  keep=min(FIRST_ORDER_DUE,self.step_cap//4+1)
  if self.ref_solo():
   return FIRST_ORDER_BY
  if any(piece.get("chain")for piece in self._smr_open()):
   return max(FIRST_ORDER_BY+5,self.order_due()+max(0,len(self.needs.items())-1))
  return max(FIRST_ORDER_BY+5,keep+5,self.order_due()+7)
 def ref_solo(self):
  return not self.needs.items()and bool(self._smr_open())and not self._smg_had_trouble()
 def ref_product(self,link):
  if not self.needs.items():
   return None
  tag=link.get("tag")
  return self._smr_of_tag(tag)if tag else None
 def ref_waits(self,floor=None):
  products=self.needs.items()
  live=self._smr_open()
  if not(products and live)or any(piece.get("relation")!="differ"or not piece.get("chain")for piece in live):
   return False
  left=self.moves_left()
  if left<=4+(self.ref_floor()if floor is None else floor):
   return False
  try:
   owners=set(map(self.ref_product,live))
   lacking=[n for n,_ in enumerate(products)if not any(rec["fit"]for rec in self.ed_pool(n))]
  except Exception as err:
   self._smr_note("ref_wait_error",error=self._smr_err(err))
   return False
  foreign=[n for n in lacking if n not in owners]
  holding=len(foreign)>(1 if None in owners else 0)
  if holding and self.__dict__.get("_smr_wait_for")!=tuple(foreign):
   self._smr_wait_for=tuple(foreign)
   self._smr_note("ref_wait",lacking=foreign)
  return holding
 @staticmethod
 def _smr_named_hop(link):
  aim=link.get("target")or{}
  if link.get("status")!="open"or link.get("hops",0)<1:
   return False
  return all(str(aim.get(part)or"").strip()for part in("seller","kind"))and bool(aim.get("details"))
 def ref_hop_room(self):
  if self.moves_left()-self.order_due()<=2:
   return False
  tally=self.__dict__.get("_smr_steps",0)
  for link in filter(self._smr_named_hop,self._smr_links()):
   since=link.get("late_from")
   if since is not None:
    if tally-since<3:
     return True
    continue
   if link.get("queries")or not self.ed_others_verified(link):
    continue
   link["late_from"]=tally
   self._smr_note("ref_late",prop=link.get("prop"),to=str((link.get("target")or{}).get("seller"))[:80])
   return True
  return False
 def ed_others_verified(self,link):
  products=self.needs.items()
  if not products:
   return True
  try:
   unsure=[n for n,_ in enumerate(products)if not any(rec["verified"]and rec["fit"]for rec in self.ed_pool(n))]
  except Exception:
   return False
  home=self.ref_product(link)
  return sum(1 for n in unsure if n!=home)<=(1 if home is None else 0)
 def ref_calls_ok(self):
  wire=self.wire
  under_way=any(piece.get("chain")for piece in self._smr_open())
  margin=20 if under_way else 40
  if self.__dict__.get("ref_calls",0)>=REF_CALLS or wire.spare()<=CLOSING+margin or wire.spent_out():
   return False
  return wire.spend<SPEND_CAP*wire.__dict__.get("scale",1.0)
 def ref_match(self,link,rule):
  whole=str(rule)
  label,_,said=whole.partition(":")
  if whole==link["req"]or whole==link.get("now"):
   return True
  if not ref_text(said)and(link.get("relation")!="differ"or not re.search(DIFFER_WORDS,said.lower())):
   return False
  if not same_prop(self._smr_prop(label),link["prop"]):
   return False
  marks=re.findall(r"\(([^)]*)\)",label)
  return not marks or not link["tag"]or bool(kind_words(marks[-1])&kind_words(link["tag"]))
 def ref_live(self,link):
  def ours(rule):
   return same_prop(self._smr_prop(rule),link["prop"])
  for rule in self.needs.props_left():
   if self.ref_match(link,rule)or ours(rule):
    return True
  return all(not ours(rule)for rule in self.needs.let_go)
 def ref_step(self):
  wire=self.wire
  if wire.done or wire.broke or not self.__dict__.get("refs"):
   return False
  try:
   return self._smr_turn()
  except Exception as err:
   self._smr_note("ref_error",error=self._smr_err(err))
   for link in self._smr_open():
    link.update(status="stuck",why="chase logic raised a fault")
   return False
 def _smr_turn(self):
  turn={"acted":False}
  state="prune"
  while state!="done":
   state=getattr(self,"_smr_t_"+state)(turn)
  return turn["result"]
 @staticmethod
 def _smr_t_end(turn,result):
  turn["result"]=result
  return"done"
 def _smr_t_prune(self,turn):
  kept=[]
  for link in self.refs:
   if link["status"]!="open":
    continue
   if self.ref_live(link):
    kept.append(link)
   else:
    link["status"]="dropped"
    self._smr_note("ref_dropped",req=link["req"][:200])
  turn["live"]=kept
  return"judge"if kept else self._smr_t_end(turn,False)
 def _smr_t_judge(self,turn):
  wire=self.wire
  if any(self.ref_facts_read(link)for link in turn["live"]if link.get("facts_due")):
   return self._smr_t_end(turn,True)
  start=wire.tick
  for link in sorted(turn["live"],key=lambda piece:piece["hops"]):
   turn["acted"]=self.ref_judge(link)or turn["acted"]
  if wire.tick>start:
   return self._smr_t_end(turn,True)
  turn["live"]=[piece for piece in turn["live"]if piece["status"]=="open"]
  if any(piece.get("facts_due")for piece in turn["live"]):
   return self._smr_t_end(turn,True)
  if turn["live"]and self.ref_room():
   return"plan"
  return self._smr_t_end(turn,turn["acted"])
 def _smr_pool(self,link,most):
  known,judged=self.archive.read,link["judged"]
  fresh=[pid for pid in self.ref_cands(link)if not(pid in known or pid in judged)]
  if link["queries"]or not self.ref_plans(link)or len(fresh)<=most:
   return fresh
  return fresh if fresh and(link.get("shows")or{}).get(fresh[0])else[]
 @staticmethod
 def _smr_merge(pools,most):
  picked=[]
  streams=[iter(p)for p in pools]
  while streams and len(picked)<most:
   for stream in list(streams):
    if len(picked)>=most:
     break
    nxt=next((pid for pid in stream if pid not in picked),None)
    if nxt is None:
     streams.remove(stream)
    else:
     picked.append(nxt)
  return picked
 def _smr_t_plan(self,turn):
  live=turn["live"]
  lister=self.batch_tool()
  most=lister[2]if lister else 2
  pools=[self._smr_pool(link,most)for link in live]
  turn["tpl"]=self.ref_template()
  idle=[piece for piece,pool in zip(live,pools)if not pool and self.ref_plans(piece)]if turn["tpl"]else[]
  turn["queue"]=self._smr_merge(pools,most)
  turn["first"]=idle if 0<len(turn["queue"])<most else[]
  if turn["queue"]and not turn["first"]:
   return"read"
  if turn["tpl"]:
   return"search"
  return self._smr_t_end(turn,turn["acted"])
 def _smr_t_read(self,turn):
  return self._smr_t_end(turn,self.ref_read(turn["queue"],turn["live"])or turn["acted"])
 def _smr_t_search(self,turn):
  order=turn["first"]or sorted(turn["live"],key=lambda piece:(piece["hops"],len(piece["queries"])))
  for link in order:
   plans=self.ref_plans(link)
   if plans:
    text=plans[0]
    link["queries"].append(text.lower())
    if self.ran_query(text):
     return self._smr_t_end(turn,True)
    fixed,slot=turn["tpl"]
    args=dict(fixed)
    args[slot]=text
    return self._smr_t_end(turn,self.ref_search(link,args)or turn["acted"])
   nxt=self.ref_page(link)
   if nxt:
    link["pages"]+=1
    return self._smr_t_end(turn,self.ref_search(link,nxt)or turn["acted"])
  return"settle"
 def _smr_t_settle(self,turn):
  for link in turn["live"]:
   judged=link["judged"]
   if any(pid not in judged for pid in self.ref_cands(link))or self.ref_plans(link)or self.ref_page(link):
    continue
   self.ref_stop(link,"no store query showed the page it names, after each try made")
   turn["acted"]=True
  return self._smr_t_end(turn,turn["acted"])
 def _smr_fetch(self,ids,say,lister,reader,
kind,**extra):
  start=self.wire.tick
  replies=self.open_reads(ids,say,lister,reader)
  trusted=True
  if lister and replies and not self.__dict__.get("batch_proven"):
   found=self.__dict__.get("last_reads")or[]
   rows_back=self.batch_ok(found,ids)
   trusted=rows_back and self.batch_richer(found,ids)
   if trusted:
    self.batch_proven=True
   elif self.ids_back(found,ids)or not self._smk_batch_shrink(len(ids)):
    self.batch_spec=None
    self._smr_note("batch_dropped",tool=lister[0])
  if replies:
   self.post_replies(replies)
  went=self._smr_booked(start)
  self.ref_ids_add(ids)
  self._smr_note(kind,ids=ids,**extra)
  return went,trusted
 def ref_read(self,queue,opens):
  lister=self.batch_tool()if len(queue)>=2 else None
  named=[]
  for piece in opens:
   if piece["head"]not in named:
    named.append(piece["head"])
  heads=", ".join(named)
  went,trusted=self._smr_fetch(queue,self._SMR_READ_SAY%heads,
lister,None,"ref_read")
  if trusted:
   missed=[pid for pid in queue if pid not in self.archive.read]
   for link in opens:
    link["judged"].update(missed)
  return went
 def ref_search(self,link,args):
  start=self.wire.tick
  landed=self.walk_call(args,self._SMR_SEARCH_SAY%(link["head"],self.ref_desc(link)))
  went=self._smr_booked(start)
  hits=[]
  if landed:
   walk_dicts(self.archive.items[-1],hits.append,ceiling=3)
   hits=[bit for bit in hits if self.row_mark(bit)in self.archive.ids and len(bit)>=4]
  shop=ref_name(link["target"]["seller"])
  query=" ".join(self.free_text(args))
  if hits and ref_name(query)==shop and all(shop not in self.ref_sellers(h)for h in hits):
   link["unindexed"]=True
  self._smr_note("ref_search",prop=link["prop"],query=query[:120],rows=len(hits))
  return went
 _SMR_JUDGE=("_smr_j_notes_first","_smr_j_notes_wait","_smr_j_notes_lost","_smr_j_plain","_smr_j_no_calls","_smr_j_hop")
 def ref_judge(self,link):
  aim=link["target"]
  known=self.archive.read
  seen=[pid for pid in self.ref_cands(link)if pid in known]
  fresh=[pid for pid in seen if pid not in link["judged"]]
  if not fresh:
   return False
  table=self.ref_index()
  parts={pid:self.ref_parts(table.get(pid)or{})for pid in seen}
  loose=link.setdefault("loose",set())
  details=aim["details"]
  def shown(pid,bit):
   return self.ref_shows(table.get(pid)or{},bit,parts[pid])
  if len(seen)>=4:
   loose|={bit for bit in details if not any(shown(pid,bit)for pid in seen)}
  need=[set(ref_toks(bit))for bit in details]
  full={pid:len([t for t in need if t and t<=parts[pid][0]|parts[pid][1]])for pid in fresh}
  fits={pid:all(shown(pid,bit)for bit in details if bit not in loose)for pid in fresh}
  ranked=sorted(fresh,key=lambda pid:(not fits[pid],-full[pid]))
  if not fits[ranked[0]]and(self.ref_room()or self.ref_waits())and self._smr_more_rows(link):
   return False
  good=[pid for pid in ranked if fits[pid]]
  pick=good[:4]or ranked[:4]
  exact=[pid for pid in pick if details and fits[pid]and full[pid]==len(details)]
  sample=exact if len(exact)==1 else(pick[:2]if good else pick[:1])
  case={"link":link,"aim":aim,"fresh":fresh,"fits":fits,"pick":pick,"exact":exact,"index":table,
"missing":self.ref_notes_missing(link,sample)}
  for stage in self._SMR_JUDGE:
   verdict=getattr(self,stage)(case)
   if verdict is not None:
    return verdict
  return True
 def _smr_more_rows(self,link):
  known,judged=self.archive.read,link["judged"]
  rows=[pid for pid in self.ref_cands(link)if not(pid in known or pid in judged)]
  return bool(rows or self.ref_plans(link)or self.ref_page(link))
 def ref_facts_read(self,link):
  due=link.pop("facts_due",None)
  tool=self.ref_facts_tool(due)if due else None
  if not(tool and self.ref_room()):
   return False
  return self._smr_fetch([due],self._SMR_FACTS_SAY%(tool,link["head"]),None,tool,"ref_facts",tool=tool)[0]
 def ref_facts_tool(self,lid):
  own=self.__dict__
  done,tally=own.get("read_with")or{},own.get("tool_caps")or{}
  fresh=[tool for tool in self.lookup_tools()if lid not in(done.get(tool)or set())]
  mapped=[tool for tool in fresh if(tally.get(tool)or{}).get("props",0)>0]
  if mapped:
   return max(mapped,key=lambda tool:((tally.get(tool)or{}).get("props",0),tool))
  return next((tool for tool in(self.role("view"),self.role("details"))if tool and tool in fresh),None)
 def _smr_j_notes_first(self,case):
  if case["missing"]and self.ref_room()and self.ref_notes_read(case["link"],case["missing"]):
   return True
  return None
 def _smr_j_notes_wait(self,case):
  return False if case["missing"]and self.ref_waits()else None
 def _smr_j_notes_lost(self,case):
  link=case["link"]
  if not case["missing"]or self.ref_room()or link.get("relation")!="differ":
   return None
  link["judged"].update(case["pick"])
  self.ref_stop(link,"too few steps remained to fetch the remarks on the page that it names now")
  return True
 def _smr_j_plain(self,case):
  exact,link=case["exact"],case["link"]
  if len(exact)!=1 or not self.ref_plain(link,exact[0]):
   return None
  sole=exact[0]
  link["judged"].update(case["pick"])
  self.ref_ids_add(case["pick"])
  self.ref_resolve(link,sole,self.ref_field(case["index"].get(sole)or{},link["prop"]),[])
  return True
 def _smr_j_no_calls(self,case):
  if self.ref_calls_ok():
   return None
  case["link"]["judged"].update(case["pick"])
  self.ref_stop(case["link"],"the hop checks ran out")
  return True
 def _smr_j_hop(self,case):
  link,fits=case["link"],case["fits"]
  self.ref_apply(link,case["pick"],self.ref_ask(link,case["pick"],""))
  if link["status"]=="open"and link["target"]is case["aim"]:
   words=[bit for bit in case["aim"]["details"]if bit not in(link.get("loose")or set())]
   table=case["index"]
   link["judged"].update(pid for pid in case["fresh"]
if not fits[pid]and self.ref_word_miss(table.get(pid)or{},words))
  return True
 def ref_word_miss(self,row,details):
  heads,said=self.ref_parts(row)
  for bit in details:
   want={t for t in ref_toks(bit)if len(t)>=3 and not re.search(r"\d",t)}-heads
   if want and want.isdisjoint(said):
    return True
  return False
 def _smr_row_names(self,rec,table):
  sold_by=self.ref_sellers(rec)
  foreign=set().union(*map(self.ref_sellers,table.values()))-sold_by
  names,_=self.ref_parts(rec)
  texts=[self.title_of(rec),self.shelf_of(rec)]
  for inner in rec.values():
   if isinstance(inner,dict):
    texts+=[str(piece)for piece in inner.values()if isinstance(piece,(str,int,float))]
  return sold_by,foreign,names|set(ref_toks(" | ".join(texts)))
 @staticmethod
 def _smr_foreign(word,low,sold_by,foreign,facts,strict):
  if low in foreign:
   return True
  if not word[:1].isupper()or len(low)<4 or low in sold_by:
   return False
  if strict and set(ref_toks(word))&facts:
   return False
  return low not in facts
 def ref_plain(self,link,lid):
  attr=link["prop"]
  rec=self.ref_index().get(lid)or{}
  value=self.ref_field(rec,attr)
  remarks=self.ref_texts(rec,desc=False)
  if not value or self.ref_names_prop(attr,remarks)or set(ref_toks(value))&set(ref_toks(remarks)):
   return False
  sold_by,foreign,facts=self._smr_row_names(rec,self.ref_index())
  for line in re.split(self._SMR_SENTENCE,remarks):
   if any(self._smr_foreign(w,ref_name(w),sold_by,foreign,facts,False)for w in re.findall(self._SMR_WORD,line)[1:]):
    return False
  return True
 def ref_notes_tool(self):
  learned=self.notes_tool()
  if learned:
   return learned
  mine=self.__dict__
  details=self.role("details")
  if(mine.get("note_keys")or{}).get(details or""):
   return None
  per_tool=mine.get("tool_keys")or{}
  texty=mine.get("text_keys")or{}
  known=set(per_tool.get(details or"")or set())|set(mine.get("row_keys")or set())
  usage=mine.get("tool_caps")or{}
  rivals=[t for t in self.lookup_tools()if t!=details]
  richer=[t for t in rivals if(set(per_tool.get(t)or set())-known)&set(texty.get(t)or set())]
  if richer:
   return max(richer,key=lambda t:((usage.get(t)or{}).get("calls",0),t))
  fallback=(self.role("view"),self.stock_tool())
  return next((t for t in fallback if t in rivals and not(usage.get(t)or{}).get("id",0)),None)
 def ref_notes_missing(self,link,ids):
  via=self.ref_notes_tool()
  if not(via and ids):
   return[]
  skip=set((self.__dict__.get("read_with")or{}).get(via)or set())|link.setdefault("noted",set())
  out=[]
  for pid in dict.fromkeys(ids):
   if len(out)==2:
    break
   if pid not in skip:
    out.append(pid)
  return out
 def ref_notes_read(self,link,need):
  via=self.ref_notes_tool()
  if not(via and need):
   return False
  link.setdefault("noted",set()).update(need)
  say=self._SMR_NOTES_SAY%(via,link["head"])
  return self._smr_fetch(need,say,None,via,"ref_notes",tool=via)[0]
 def _smr_differs(self,statuses=None,also=None):
  for link in self._smr_links():
   if link.get("relation")!="differ":
    continue
   if statuses is not None and link.get("status")not in statuses:
    continue
   if also is not None and not also(link):
    continue
   if self.ref_live(link):
    yield link
 def _smr_barring(self,statuses=None,also=None):
  yield from self._smr_differs(statuses,also)
  if statuses is not None and"resolved"not in statuses:
   return
  for link in self._smr_links():
   if link.get("relation")!="pairs"or link.get("status")!="resolved":
    continue
   if any(set(ref_toks(mate))==set(ref_toks(link.get("value")))for mate in link.get("goes_with")or[]):
    continue
   if(also is None or also(link))and self.ref_live(link):
    yield link
 def ref_pointers(self):
  return{pid for link in self._smr_barring(("open","stuck","resolved"))
for pid in link.get("chain")or[]if self.ref_own_product(link,pid)}
 def ref_own_product(self,link,lid):
  if not self.needs.items():
   return True
  shelf=self.shelf_known(lid)
  home=self._smr_of_tag(shelf)if shelf else None
  if home is None or not link.get("tag"):
   return True
  return home==self._smr_of_tag(link["tag"])
 def ref_pointer_note(self,item):
  for link in self._smr_barring(also=lambda piece:item in(piece.get("chain")or[])):
   if not self.ref_own_product(link,item):
    continue
   known=f" ({link['value']})"if link.get("status")=="resolved"and link.get("value")else""
   attr=link["prop"]
   return self._SMR_POINTER%(item,link["head"],attr,known,attr)
  return self._smr_lists_read(item)
 def _smr_lists_read(self,item):
  for link in self._smr_barring(("resolved",)):
   if link.get("relation")!="pairs"or not self.ref_own_product(link,item):
    continue
   read=set(ref_toks(link.get("value")))
   attr=link["prop"]
   for text in self._smm_prop_fields((self.code_rows().get(item)or{}).get("fields")or{},attr):
    if any(set(ref_toks(part))==read for part in re.split(r"[,;/&]",str(text))):
     return self._SMR_LISTS%(item,link["value"],attr,attr)
  return""
 @staticmethod
 def _smr_hits_value(link,facts):
  def flat(v):
   return" ".join(ref_toks(v))
  avoid={flat(v)for v in[link.get("value"),*(link.get("goes_with")or[])]if str(v or"").strip()}
  return any(flat(val)in avoid for key,val in(facts or{}).items()if same_prop(key,link["prop"]))
 def ref_excluded(self,facts,which):
  products=self.needs.items()
  for link in self._smr_differs(("resolved",)):
   tag=link.get("tag")
   if products and tag and self._smr_of_tag(tag)not in(which,None):
    continue
   if self._smr_hits_value(link,facts):
    return True
  return False
 def _smr_sends_on(self,sentence,attr,sold_by,foreign,facts):
  marks=list(re.finditer(self._SMR_WORD,sentence))
  masked=re.sub(r"[A-Za-z0-9]+(?=\s+by\s+[A-Z])",lambda h:" "*(h.end()-h.start()),sentence)
  anchor=next((m.start()for m in marks if prop_named(attr,masked[:m.end()])),None)
  if anchor is None:
   return False
  for n in range(1,len(marks)):
   word=marks[n].group(0)
   low=ref_name(word)
   if marks[n].start()<=anchor or not low or not self._smr_foreign(word,low,sold_by,foreign,facts,True):
    continue
   if not any(w.group(0).lower()in self._SMR_NEGATORS for w in marks[max(0,n-5):n]):
    return True
  return False
 def own_redirects(self,item):
  try:
   table=self.ref_index()
   rec=table.get(item)or{}
   remarks=self.ref_texts(rec,desc=False)
   if not remarks:
    return[]
   sold_by,foreign,facts=self._smr_row_names(rec,table)
   products=self.needs.items()
   shelf=self.shelf_of(rec)
   home=self._smr_of_tag(shelf)if products and shelf else None
   lines=re.split(self._SMR_SENTENCE,remarks)
   found=[]
   for rule in self.needs.props_left():
    label=str(rule).partition(":")[0]
    if home is not None and self.product_of(label)not in(home,None):
     continue
    attr=re.split(r"\s*\(",label)[0].strip()
    hit=next((s for s in lines if self._smr_sends_on(s,attr,sold_by,foreign,facts)),None)if attr else None
    if hit is not None:
     found.append((str(rule),attr,hit.strip()))
   return found
  except Exception:
   return[]
 def notes_remark(self,call_name,call_args):
  quiet=tuple(map(self.role,("search","cart","cart_view","message")))
  if call_name in quiet or call_name in self.finishers():
   return""
  version=self.needs.version
  units,passed=self.trolley.units,self.passed
  said=[]
  for lid in sorted(self.asked_ids(call_args)):
   stamp=f"{lid}|{version}"
   if not(lid in units or stamp in passed):
    continue
   told=self.__dict__.setdefault("_smr_told",set())
   if stamp in told:
    continue
   found=self.own_redirects(lid)
   if found:
    told.add(stamp)
    passed.discard(stamp)
    self._smr_note("own_redirect",id=lid,props=[entry[1]for entry in found])
    said.append(self._smr_remark(lid,found,lid in units))
  return" ".join(said)
 @classmethod
 def _smr_remark(cls,lid,found,held):
  quoted="; ".join('%s ["%s"]'%(attr,words[:160])for _,attr,words in found)
  reqs="; ".join("'%s'"%entry[0]for entry in found)
  them="it"if len(found)==1 else"them"
  fix=". Were that true, drop this line and place a different confirmed pick."if held else"."
  return cls._SMR_REMARK%(lid,quoted,reqs,them,fix)
 def ref_phrase(self,line):
  spoken=str(line or"")
  if not ref_text(spoken):
   return None
  parsed=ref_parse(spoken,lambda name:self.ref_seller_ok(name,spoken))
  if not parsed or not self.ref_seller_ok(parsed["seller"],spoken):
   return None
  spot=re.search(r"(?<![A-Za-z'])"+re.escape(parsed["seller"])+r"(?:'s)?(?![A-Za-z])",spoken)
  if not spot:
   return None
  lead=spoken[:spot.start()]
  clause=max((m.start()for m in re.finditer(r"[,;!?]|[.](?=\s|$)",lead)),default=-1)+1
  low=lead[clause:].lower()
  hits=[m.start()for m in(re.search(p,low)for p in(DIFFER_WORDS,REF_WORDS,POINT_WORDS,PAIR_WORDS))if m]
  if not hits:
   return None
  start=clause+min(hits)
  trail=spoken[spot.end():]
  stop=re.search(r"[.!?;](?:\s|$)",trail)
  reach=stop.start()if stop else len(trail)
  aside=re.search(r"\(\s*or\b[^)]*\)",trail)
  if aside and aside.start()<reach:
   reach=aside.end()
  end=spot.end()+reach
  phrase=spoken[start:end].strip(" ,;")
  return(phrase,start,end,parsed)if phrase else None
 def ref_line_words(self,question,rest):
  if not question:
   return[]
  products=self.needs.items()or[{"product":str(self.needs.reading.get("product")or"")}]
  kind_set=set()
  for entry in products:
   kind_set.update(re.findall(r"[a-z]+",str(entry["product"]).lower()))
  terms=re.findall(r"[a-z0-9]+",str(question).lower())
  def plain(term):
   return term not in LINK_WORDS and len(term)>=3 and term not in kind_set and term.rstrip("s")not in kind_set
  ok=list(map(plain,terms))
  runs=[]
  for width in(3,2,1):
   starts=[i for i in range(len(terms)-width+1)if all(ok[i:i+width])]
   runs.extend(" ".join(terms[i:i+width])for i in starts)
  return[run for run in runs if prop_named(run,rest)]
 class _sms_Cut:
  def __init__(self,outside,inside):
   self.outside,self.inside=outside,inside
   self.in_toks,self.out_toks=set(ref_toks(inside)),set(ref_toks(outside))
  @staticmethod
  def label(req):
   return re.split(r"\s*\(",str(req).partition(":")[0])[0].strip()
  def only_inside(self,name):
   return bool(name)and prop_named(name,self.inside)and not prop_named(name,self.outside)
  def describes_other(self,req):
   words=set(ref_toks(str(req).partition(":")[2]))
   if self.only_inside(self.label(req)):
    return True
   return bool(words)and words<=self.in_toks and not words&self.out_toks
 @staticmethod
 def _sms_tags(head):
  return re.findall(r"\(([^)]*)\)",head)
 @staticmethod
 def _sms_points(value):
  return bool(ref_text(value)or re.search(DIFFER_WORDS,value.lower()))
 def _sms_trace(self,kind,fields):
  self.log.append({"kind":kind,**fields})
 def ref_fix(self,reply,said,question):
  return[self._sms_fix_line(reply,text,question)for text in said]
 def _sms_fix_line(self,reply,text,question):
  found=self.ref_phrase(text)
  if not found:
   return text
  phrase,lo,hi,info=found
  cut=self._sms_Cut(text[:lo]+" "+text[hi:],text[lo:hi])
  listed,given=reply.get("needs"),reply.get("references")
  stated=[req for req in listed if isinstance(req,str)and":"in req]if isinstance(listed,list)else[]
  entries=[e for e in given if isinstance(e,dict)]if isinstance(given,list)else[]
  prop=self._sms_line_prop(cut,stated,entries,question,info)
  if not prop:
   self._sms_trace("ref_line_unplaced",{"line":text[:200]})
   return cut.outside
  several=self.needs.items()
  tag=self._sms_line_tag(cut,stated,prop,question,several)
  def same_product(req):
   marks=self._sms_tags(str(req).partition(":")[0])
   if several and tag and marks:
    return bool(kind_words(marks[-1])&kind_words(tag))
   return True
  own=[req for req in stated if same_prop(cut.label(req),prop)and same_product(req)]
  relation,tol,unit=ref_relation(phrase)
  who=ref_name(info["seller"])
  reuse=next((req for req in own if self._sms_same_pointer(req,relation,who)),None)
  if own:
   head=own[0].partition(":")[0].strip()
  elif tag:
   head=f"{prop} ({tag})"
  else:
   head=prop
  if several and tag and not self._sms_tags(head):
   head=f"{head} ({tag})"
  need=reuse or f"{head}: {phrase}"
  foreign=[req for req in stated if req not in own and(cut.describes_other(req)or(
same_product(req)and self._sms_points_at(req.partition(":")[2],who)and not prop_named(cut.label(req),text)))]
  slot=stated.index(own[0])if own else len(stated)
  remaining=[req for req in stated if req not in own and req not in foreign]
  remaining.insert(slot,need)
  reply["needs"]=remaining
  self._sms_prune_answer(reply,cut,prop,need,foreign,same_product)
  self._sms_merge_entry(entries,prop,info,text,who,(relation,tol,unit))
  reply["references"]=entries
  reply["ref_needs"]=[*(reply.get("ref_needs")or[]),need]
  self._sms_trace("ref_line",{"prop":prop,"tag":tag,"relation":relation,"need":need[:200],"kept":bool(reuse),
"details_out":[f[:80]for f in foreign]})
  return cut.outside
 def _sms_points_at(self,value,who):
  return self._sms_points(value)and who in ref_name(value)
 def _sms_same_pointer(self,req,relation,who):
  tail=req.partition(":")[2]
  return self._sms_points_at(tail,who)and ref_relation(tail)[0]==relation
 def _sms_merge_entry(self,entries,prop,info,text,who,
rel):
  entry=next((e for e in entries if same_prop(str(e.get("property")or""),prop)),None)
  if entry is None:
   entry={"property":prop}
   entries.append(entry)
  if not(self.ref_seller_ok(entry.get("seller"),text)and ref_name(entry.get("seller"))==who):
   entry.update(seller=info["seller"],kind=info["kind"],details=list(info["details"]))
  elif not entry.get("details"):
   entry["details"]=list(info["details"])
  entry.setdefault("kind",info["kind"])
  if rel[0]!="same"or not entry.get("relation"):
   entry.update(relation=rel[0],tolerance=rel[1],unit=rel[2])
 def _sms_line_prop(self,cut,stated,entries,question,info):
  label=cut.label
  asked=list(filter(None,map(label,str(self.__dict__.get("asking_label")or"").split(";"))))
  sources=[[str(e.get("property")or"")for e in entries],
[label(req)for req in stated if self._sms_points(req.partition(":")[2])],
[label(req)for req in stated],
asked,
sorted(self.store_keys())+[label(w)for w in self.needs.written],
self.ref_line_words(question,cut.outside)]
  def named_in(words):
   def rank(name):
    return prop_named(name,words,exact=True),len(re.findall(r"[a-z0-9]+",name.lower())),len(name)
   for source in sources:
    heard=[name for name in source if name and prop_named(name,words)]
    if heard:
     return sorted(heard,key=rank,reverse=True)[0]
   return""
  found=named_in(cut.outside)or named_in(self._sms_after_listing(cut.inside,info))
  if found:
   return found
  fallback=sources[0]+sources[1]+(asked if len(asked)==1 else[])
  return next((name for name in fallback if name and not cut.only_inside(name)),"")
 @staticmethod
 def _sms_after_listing(phrase,info):
  low=phrase.lower()
  tail=low.rfind(str(info.get("seller")or"").lower())
  if tail<0:
   return""
  tail+=len(str(info.get("seller")))
  for bit in info.get("details")or():
   for word in re.findall(r"[a-z0-9]+",str(bit).lower()):
    spot=low.find(word,tail)
    if spot>=0:
     tail=spot+len(word)
  return phrase[tail:]
 def _sms_line_tag(self,cut,stated,prop,question,several):
  if not several:
   return""
  def by_listener():
   marks=[m for req in stated if same_prop(cut.label(req),prop)for m in self._sms_tags(req.partition(":")[0])]
   return self.product_of(f"x ({marks[-1]})")if marks else None
  tries=(lambda:self.product_of(cut.outside),by_listener,lambda:self.product_of(question)if question else None)
  idx=next((got for got in(attempt()for attempt in tries)if got is not None),None)
  return""if idx is None else several[idx]["product"]
 def _sms_prune_answer(self,reply,cut,prop,need,foreign,same_product
):
  label,written=cut.label,self.needs.written
  current=reply.get("requirements_now")
  if isinstance(current,list):
   survivors=[]
   for item in filter(lambda x:isinstance(x,str),current):
    if item not in written:
     if item in foreign or cut.describes_other(item):
      continue
     if same_prop(label(item),prop)and same_product(item):
      continue
    survivors.append(item)
   reply["requirements_now"]=survivors if need in survivors else survivors+[need]
  if isinstance(reply.get("dropped"),list):
   reply["dropped"]=[x for x in reply["dropped"]
if not isinstance(x,str)or not(cut.only_inside(label(x))or cut.describes_other(x))]
  if isinstance(reply.get("no_preference"),list):
   reply["no_preference"]=[x for x in reply["no_preference"]
if not isinstance(x,str)or not(cut.only_inside(label(x))or same_prop(label(x),prop))]
 @staticmethod
 def ref_names_prop(name,passage):
  keys=[w for w in prop_words(name)if len(w)>=3]
  vocab=set(re.findall(r"[a-z]+",str(passage).lower()))
  if not keys:
   return True
  stems={t[:4]for t in vocab if len(t)>=4}
  return any(k in vocab or(len(k)>=4 and k[:4]in stems)for k in keys)
 def ref_card(self,rid):
  rec=self.ref_index().get(rid)or{}
  flat={}
  for group in filter(lambda v:isinstance(v,dict),rec.values()):
   flat.update((str(name),str(x)[:90])for name,x in list(group.items())[:40]
if isinstance(x,(str,int,float))and not isinstance(x,bool))
  known=self.seller_keys()
  names=set()
  for field,text in rec.items():
   low=str(field).lower()
   if not isinstance(text,str)or not text.strip()or len(text)>60 or"id"in low:
    continue
   if(re.search(SELLER_KEYS,low)or str(field)in known)and not re.search(
r"note|text|desc|comment|detail|info|message|section",low):
    names.add(str(text))
  card=dict(id=rid)
  card["sold by"]=" / ".join(sorted(names))
  card.update(category=self.shelf_of(rec),title=self.title_of(rec)[:160],facts=flat,notes=self.ref_texts(rec)[:900])
  return as_text(card,5000)
 def ref_ask(self,link,batch,refusal):
  if not self.ref_calls_ok():
   return None
  self.ref_calls=1+self.__dict__.get("ref_calls",0)
  cards="\n".join(map(self.ref_card,batch))
  prompt=f"{REF_HOP.replace('PROPERTY', link['prop']).replace('DESCRIPTION', self.ref_desc(link))}\n\nCANDIDATE RECORDS:\n{cards}"
  if refusal:
   prompt=f"{prompt}\n\nTHE PREVIOUS ANSWER WAS REJECTED: {refusal} Answer again from the records."
  reply=self.wire.consult(prompt,3000,CHECKER[:1])
  self._sms_trace("ref_hop",{"prop":link["prop"],"ids":batch,"answer":reply})
  return reply
 def ref_apply(self,link,batch,reply,retry=True):
  hop={"link":link,"batch":batch,"reply":reply,"retry_left":retry}
  state="read"
  while state!="end":
   state=getattr(self,"_sms_hop_"+state)(hop)
 def _sms_hop_read(self,hop):
  link,batch,reply=hop["link"],hop["batch"],hop["reply"]
  if not isinstance(reply,dict):
   link["fails"]+=1
   if link["fails"]>=2:
    link["judged"].update(batch)
    self.ref_stop(link,"the hop check never answered")
   return"end"
  named=reply.get("target")
  cand=""if isinstance(named,bool)or not isinstance(named,(str,int))else str(named).strip()
  absent=reply.get("missing")
  lacking=[str(x)for x in absent if str(x).strip()]if isinstance(absent,list)else[]
  if cand in batch and not lacking:
   hop["cand"]=cand
   return"check"
  rescued=self._sms_rescue(link,batch,reply)
  if rescued:
   hop["cand"]=rescued
   return"check"
  link["judged"].update(batch)
  self.ref_ids_add(batch)
  if cand in batch:
   self._sms_trace("ref_partial",{"prop":link["prop"],"target":cand,"missing":lacking[:4]})
  return"end"
 def _sms_rescue(self,link,batch,reply):
  onward=reply.get("go_to")
  if not(isinstance(onward,dict)and str(onward.get("seller")or"").strip()):
   return""
  details,index=link["target"]["details"],self.ref_index()
  hits=[]
  for pid in batch:
   rec=index.get(pid)or{}
   notes=self.ref_texts(rec)
   toks=set(ref_toks(notes))
   if self._sms_check_go(link,pid,rec,notes,toks,onward)[1]or self.ref_word_miss(rec,details):
    continue
   if all(set(ref_toks(bit))<=toks for bit in details):
    hits.append(pid)
  if len(hits)!=1:
   return""
  rec=index.get(hits[0])or{}
  link.setdefault("loose",set()).update(bit for bit in details if not self.ref_shows(rec,bit))
  self._sms_trace("ref_rescued",{"prop":link["prop"],"target":hits[0]})
  return hits[0]
 def _sms_hop_check(self,hop):
  verdict=self.ref_validate(hop["link"],hop["cand"],hop["reply"],hop["batch"])
  hop["verdict"]=verdict
  if verdict[1]and self._sms_facts_first(hop):
   return"end"
  return"retry"if verdict[1]and hop["retry_left"]and self.ref_calls_ok()else"settle"
 def _sms_facts_first(self,hop):
  link,cand,reply=hop["link"],hop["cand"],hop["reply"]
  onward=reply.get("go_to")
  if hop["verdict"][0]!="answer"or(isinstance(onward,dict)and str(onward.get("seller")or"").strip()):
   return False
  if self.ref_field(self.ref_index().get(cand)or{},link["prop"]):
   return False
  tried=link.setdefault("mapped",set())
  if cand in tried or not self.ref_facts_tool(cand):
   return False
  tried.add(cand)
  link["facts_due"]=cand
  link["judged"].update(pid for pid in hop["batch"]if pid!=cand)
  self.ref_ids_add(hop["batch"])
  self._sms_trace("ref_facts_due",{"prop":link["prop"],"target":cand})
  return True
 def _sms_hop_retry(self,hop):
  hop["reply"]=self.ref_ask(hop["link"],hop["batch"],hop["verdict"][1])
  hop["retry_left"]=False
  return"read"
 def _sms_hop_settle(self,hop):
  link,batch,cand=hop["link"],hop["batch"],hop["cand"]
  where,problem,onward,value,pairs=hop["verdict"]
  self.ref_ids_add(batch)
  if not problem:
   link["judged"].update(batch)
   if cand in link["chain"]:
    self.ref_stop(link,"its notes loop back to a listing already followed")
   elif onward:
    self.ref_go(link,cand,onward)
   else:
    self.ref_resolve(link,cand,value,pairs)
   return"end"
  self._sms_trace("ref_invalid",{"prop":link["prop"],"target":cand,"problem":problem[:300]})
  if where=="target":
   link["judged"].add(cand)
   return"end"
  link["judged"].update(batch)
  link["chain"].append(cand)
  self.ref_stop(link,problem)
  return"end"
 def ref_validate(self,link,cand,reply,batch=None):
  index=self.ref_index()
  rec=index.get(cand)or{}
  names,values=self.ref_parts(rec)
  wanted=link["target"]["details"]
  lax=link.get("loose")or set()
  missed=next((d for d in wanted if d not in lax and not self.ref_shows(rec,d,(names,values))),None)
  if missed is not None:
   return"target",f"{cand} fails to show '{missed}', so it is not the listing described.",None,"",[]
  groups=[set(ref_toks(d))for d in wanted]
  def score(ident):
   kk,vv=self.ref_parts(self.ref_index().get(ident)or{})
   seen=kk|vv
   return len([g for g in groups if g and g<=seen])
  rival=max(batch or[cand],key=score)
  if score(rival)>score(cand):
   return"target",f"{rival} holds more of the details named than {cand}.",None,"",[]
  notes=self.ref_texts(rec)
  note_toks=set(ref_toks(notes))
  onward=reply.get("go_to")
  if isinstance(onward,dict)and str(onward.get("seller")or"").strip():
   return self._sms_check_go(link,cand,rec,notes,note_toks,onward)
  unstated=self._sms_notes_value(link["prop"],cand,rec,notes,reply)
  if unstated:
   return"answer",unstated,None,"",[]
  mute=self._sms_mute_target(link["prop"],cand,rec,notes)
  if mute:
   return"target",mute,None,"",[]
  return self._sms_check_value(cand,reply,names|values,note_toks)
 def _smr_sets_aside(self,line):
  return bool(self._SMR_NEGATORS&set(re.findall(r"[a-z]+",line.lower().replace("n't"," not"))))
 def _sms_notes_value(self,prop,cand,rec,notes,answer):
  if self.ref_field(rec,prop):
   return""
  told=[line for line in re.split(self._SMR_SENTENCE,notes)if prop_named(prop,line)]
  own=set(ref_toks(str(answer.get("value")or"")))
  if not told or(own and own<=set(ref_toks(" ".join(told)))):
   return""
  return(f"{cand} has no field for the {prop}, but its notes state the {prop}; the value must be the one those "
"notes give, and goes_with its partners there.")
 def _sms_mute_target(self,prop,cand,rec,notes):
  if self.ref_field(rec,prop)or prop_named(prop,f"{self.title_of(rec)} {notes}"):
   return""
  shelf,known=self.shelf_of(rec),self.archive.read
  peers=[row for pid,row in self.ref_index().items()if pid!=cand and pid in known and self.shelf_of(row)==shelf]
  if not peers or any(self.ref_field(row,prop)for row in peers):
   return""
  return f"{cand} says nothing of the {prop}, nor does any record read on its shelf, so it is not the listing described."
 def _sms_check_go(self,link,cand,rec,notes,note_toks,
onward):
  shop=str(onward["seller"]).strip()
  mention=re.compile(r"(?<![a-z0-9])"+re.escape(shop.lower())+r"(?:'s)?(?![a-z0-9])")
  if mention.search(notes.lower())is None:
   return"answer",f"the go_to seller '{shop}' is not written in {cand}'s notes.",None,"",[]
  lines=re.split(r"(?<=[.!?])\s+",notes)
  about=[s for s in lines if mention.search(s.lower())]
  about+=[was for was,s in zip(lines,lines[1:])if mention.search(s.lower())and self._smr_sets_aside(was)]
  if not any(self.ref_names_prop(link["prop"],s)for s in about):
   return"answer",f"the line naming {shop} is about another property, not {link['prop']}.",None,"",[]
  listed=onward.get("details")
  kept=[]
  for item in listed if isinstance(listed,list)else[]:
   if isinstance(item,bool)or not isinstance(item,(str,int,float)):
    continue
   item=str(item).strip()
   words=set(ref_toks(item))
   if item and words and 2*len(words&note_toks)>=len(words):
    kept.append(item)
  nxt={"seller":shop,"kind":str(onward.get("kind")or"").strip()[:80],"details":kept[:6]}
  if ref_name(shop)in self.ref_sellers(rec)and self.ref_key(nxt)==self.ref_key(link["target"]):
   return"answer",f"the go_to is {cand} itself.",None,"",[]
  return"","",nxt,"",[]
 @staticmethod
 def _sms_check_value(cand,answer,record_words,note_toks
):
  read=str(answer.get("value")or"").strip()
  own=set(ref_toks(read))
  if not(read and own and own<=record_words):
   return"answer",f"its value '{read}' is not written in {cand}'s record.",None,"",[]
  extra=answer.get("goes_with")
  texts=[]
  for item in extra if isinstance(extra,list)else[]:
   if isinstance(item,(str,int,float))and not isinstance(item,bool)and str(item).strip():
    texts.append(str(item).strip())
  mates=[]
  for text in dict.fromkeys(texts):
   words=set(ref_toks(text))
   if words and words<=note_toks:
    mates.append(text)
  return"","",None,read,mates[:3]
 @staticmethod
 def _sms_fresh_hop():
  return dict(judged=set(),queries=[],pages=0,tries=0,fails=0,unindexed=False,loose=set(),late_from=None)
 def ref_go(self,link,cand,onward):
  link["chain"].append(cand)
  mark=self.ref_key(onward)
  if mark in link["seen"]:
   halt="its notes loop back to a listing already followed"
  elif link["hops"]+1>=REF_HOPS:
   halt=f"more than {REF_HOPS} listings in a row each point somewhere else"
  else:
   halt=""
  if halt:
   self.ref_stop(link,halt)
   return
  link["seen"].add(mark)
  link.update(hops=link["hops"]+1,target=onward,**self._sms_fresh_hop())
  self._sms_trace("ref_go",{"prop":link["prop"],"from":cand,"to":onward})
  self.ref_tell(f"{cand} is the listing meant for {link['head']}; its own notes pass the reader on to {self.ref_desc(link)}: "
"the code follows them.")
 def ref_value(self,link):
  read,how=str(link["value"]),link["relation"]
  alts=[read]+[p for p in list(link.get("goes_with")or[])if p.lower()!=read.lower()]
  if how=="within"and is_number(link.get("tol")):
   num=re.search(r"(-?\d+(?:\.\d+)?)\s*([a-z%]*)",read.lower().replace(",",""))
   if num:
    centre,width=float(num.group(1)),float(link["tol"])
    unit=num.group(2)or str(link.get("unit")or"")
    low,high=round(centre-width,6),round(centre+width,6)
    return f"{low:g} to {high:g} {unit}".strip()+f" (within {width:g} of {read})"
  if how in("at most","at least"):
   return f"{how} {read}"
  if how=="differ":
   return" and ".join(f"not {a}"for a in alts)
  if how=="pairs":
   return" or ".join(dict.fromkeys(str(p)for p in link.get("goes_with")or[]))
  return" or ".join(alts)
 def ref_resolve(self,link,cand,read,mates):
  if not str(read or"").strip():
   self.ref_stop(link,f"{cand} gives no value for it")
   return
  if link["relation"]!="pairs":
   mates=[m for m in mates if set(ref_toks(m))!=set(ref_toks(read))]
  if link["relation"]=="pairs"and not mates:
   self.ref_stop(link,f"{cand} shows {read}, but no store rule saying which value pairs with it was found, so its "
"counterpart is unknown")
   return
  link["chain"].append(cand)
  link.update(status="resolved",value=str(read).strip(),goes_with=list(mates))
  settled=f"{link['head']}: {self.ref_value(link)}"
  link["now"]=settled
  self.ref_swap(link,settled)
  self.needs.resolved.append(settled)
  self.needs.version+=1
  self.reveal_settle(link)
  trail=link["chain"]
  self._sms_trace("ref_resolved",{"req":link["req"][:200],"now":settled,"chain":[*trail]})
  via=""if len(trail)<2 else f", reached from {trail[0]} through the listings' notes"
  joined=" and ".join(mates)
  if link["relation"]=="pairs":
   tail=(f"; the store pairs {link['value']} with {joined}. The shopper wants that counterpart: {joined} meets it"
+(""if any(m.lower()==link["value"].lower()for m in mates)else f", {link['value']} itself does not"))
  elif link["relation"]!="differ":
   tail=f"; its notes say {link['value']} goes together with {joined}, so each of them meets it"if mates else""
  else:
   tail=((f"; its notes pair {joined} with it"if mates else"")
+". The shopper's value MUST DIFFER from that one: any other value meets it, that value does not")
  self.ref_tell(f"SETTLED: {link['head']} is fixed by a linked listing, and {cand} shows {link['value']}{via}{tail}. The "
f"requirement is now: {settled}.")
 def ref_swap(self,link,settled):
  book=self.needs
  shop=ref_name(link["first"]["seller"])
  def stands_for(req):
   if self.ref_match(link,req):
    return True
   name,_,tail=str(req).partition(":")
   marks=self._sms_tags(name)
   if marks and link["tag"]and not kind_words(marks[-1])&kind_words(link["tag"]):
    return False
   return bool(shop)and shop in ref_name(tail)and same_prop(re.split(r"\s*\(",name)[0],link["prop"])
  def swapped(reqs):
   return[settled if stands_for(r)else r for r in reqs]
  touched=0
  for attr in("written","uncovered"):
   old=list(getattr(book,attr))
   new=swapped(old)
   touched+=new!=old
   setattr(book,attr,list(dict.fromkeys(new)))
  if book.now_reqs is not None:
   fresh=swapped(book.now_reqs)
   book.now_reqs=list(dict.fromkeys(fresh if settled in fresh else fresh+[settled]))
  if not touched:
   book.uncovered.append(settled)
  reading=book.reading
  if isinstance(reading.get("requirements"),list):
   reading["requirements"]=list(book.written)
  tail=settled.partition(":")[2].strip()
  slots=[s for s in reading.get("items")or[]if isinstance(s,dict)and isinstance(s.get("requirements"),list)]
  for slot in slots:
   if len(slots)>1 and link["tag"]and not(kind_words(slot.get("product")or"")&kind_words(link["tag"])):
    continue
   slot["requirements"]=[str(r).partition(":")[0].strip()+": "+tail if stands_for(r)else r
for r in slot["requirements"]]
 def ref_stop(self,link,cause):
  if link["status"]=="open":
   link.update(status="stuck",why=cause)
   self._sms_trace("ref_stuck",{"req":link["req"][:200],"why":cause[:300],"chain":[*link["chain"]]})
   self.ref_tell(f"UNRESOLVED: {link['head']} ({cause}). Its value is unknown: if notes of a listing you saw show where the "
"chain ends, use that value; otherwise never decline or reject a listing over it.")
 def ref_tell(self,note):
  newest=self.chat[-1]if self.chat else{}
  if self.chat and len(self.chat)>2 and newest.get("role")in("tool","user")and isinstance(newest.get("content"),str):
   newest["content"]+="\n[reference] "+note
  else:
   self.ref_told=[*(self.__dict__.get("ref_told")or[]),note]
 _sms_ref_states={
"resolved":lambda self,r:f"{r['now']} (read on {r['chain'][-1]}"
+(", must differ from it"if r.get("relation")=="differ"else"")+")"
+self._sms_ref_apart(r),
"open":lambda self,r:f"{r['head']}: the code is following it (now: {self.ref_desc(r)})",
"stuck":lambda self,r:f"{r['head']}: not resolved, value unknown (never decline or reject a listing over it)",
}
 @staticmethod
 def _sms_ref_apart(link):
  details=(link.get("first")or{}).get("details")
  if not details:
   return""
  return" [%s: details of the pointed listing, not conditions on the item to buy]"%"; ".join(map(str,details))
 def ref_line(self):
  try:
   shown=[]
   for link in self.__dict__.get("refs")or[]:
    render=self._sms_ref_states.get(link["status"])
    if render is not None:
     shown.append(render(self,link))
   queued=self.__dict__.pop("ref_told",None)or[]
   lead="\n[references] "+"; ".join(shown)if shown else""
   return lead+"".join("\n[reference] "+t for t in queued)
  except Exception:
   return""
 def ref_note(self,item=""):
  try:
   entries,ends,unsure=[],[],False
   for link in self.__dict__.get("refs")or[]:
    state=link["status"]
    if state=="resolved":
     entries.append(self._sms_settled_entry(link))
    elif state in("open","stuck"):
     unsure=True
     trail=link["chain"]
     entries.append(f"- {link['head']} = OPEN: "+("the code got as far as "+" -> ".join(trail)if trail
else"the code never found the listing that pointer names")+".")
     ends.extend(i for i in trail[-2:]if i!=item)
   if not entries:
    return""
   text=("\n\nREFERENCE REQUIREMENTS (each value comes from another listing; the code traced it through the notes "
"in the store's records):\n"+"\n".join(entries))
   if unsure:
    text+=("\nOPEN entries: if a record further down is that listing, or the last listing its chain got to, and "
"states its own value without pointing elsewhere, that value is the requirement (for a counterpart, "
"the value the store pairs with it); otherwise the value is unknown and an item never fails on it alone.")
   recs=[]
   for rid in dict.fromkeys(reversed(ends)):
    recs.extend(self.archive.about(rid,1))
   if recs:
    text+="\nRECORDS WHERE AN OPEN CHAIN STOPPED:\n"+"\n".join(recs[:2])
   return text
  except Exception:
   return""
 @staticmethod
 def _sms_settled_entry(link):
  trail,mates=link["chain"],link["goes_with"]
  via=f", traced from {trail[0]} along the listings' notes"if len(trail)>1 else""
  joined=" and ".join(mates)
  if link.get("relation")=="differ":
   tie=f"; its notes also tie {joined} to it"if mates else""
   return(f"- {link['now']} = EXCLUSION: the shopper's {link['prop']} has to be other than {link['value']}, the value "
f"{trail[-1]} shows{via}{tie}. Every other value is acceptable and that one is not; the listings along the "
f"chain ({', '.join(trail)}) all carry it.")
  if link.get("relation")=="pairs":
   bare=""if any(str(m).lower()==str(link["value"]).lower()for m in mates)else f"; {link['value']} itself does not"
   return(f"- {link['now']} = COUNTERPART: a listing showing {' or '.join(mates)} meets it{bare}. {trail[-1]} shows {link['value']}"
f"{via}, and the store pairs it with {joined}; the shopper's {link['prop']} is that partner value.")
  tie=f"; its notes treat {joined} as equal to {link['value']}, so any of them satisfies it"if mates else""
  return f"- {link['now']} = SETTLED: taken from {trail[-1]}{via}{tie}."
 def ceil_learn(self,args,found_rows,count,raw):
  pg=self.page_arg()
  page_key=pg[0]if pg else None
  plain={name:v for name,v in args.items()if name!=page_key}
  totals=self.__dict__.setdefault("ceil_seen",{})
  totals[json.dumps(plain,sort_keys=True,default=str)]=int(count)
  hits=list(filter(lambda r:isinstance(r,dict),found_rows))
  if pg and pg[1]is True and hits and isinstance(raw,dict):
   self._sms_note_depth(args.get(page_key,pg[2]),pg[2],hits,count,raw)
  _,mode,_,paging=self.search_keys(args)
  limits=[name for name,v in args.items()if self._sms_limit_arg(name,v,page_key,paging)]
  if len(limits)==1 and hits:
   self._sms_learn_mode(args,plain,hits,mode,page_key,limits[0])
 @staticmethod
 def _sms_limit_arg(name,val,page_key,paging):
  if isinstance(val,bool)or not is_number(val)or name==page_key or name in paging or not val>1:
   return False
  return re.search(r"page|size|count|result|min|low|from|start|offset",str(name).lower())is None
 def _sms_note_depth(self,at,origin,hits,total,obs):
  flags=[v for k,v in obs.items()
if re.search(r"next|more|onward|continu|cursor",str(k).lower())and not isinstance(v,(list,dict))]
  per_page=max(len(hits),self.page_size())
  ended=bool(flags)and all(v is None or v is False or v==""for v in flags)
  if is_number(at)and ended and len(hits)>=per_page:
   depth=int((at-origin+1)*per_page)
   if total>depth:
    self.page_cap=min(self.__dict__.get("page_cap")or depth,depth)
 def _sms_learn_mode(self,args,plain,hits,mode,page_key,
limit_key):
  ceiling=float(args[limit_key])
  shelves={self.shelf_of(row)for row in hits}
  shelf_args=[name for name,val in args.items()
if isinstance(val,str)and name!=page_key and val.strip()and val.strip().lower()in shelves]
  shelf_key=shelf_args[0]if len(shelf_args)==1 else None
  fixed=dict(pair for pair in mode if pair[0]!=shelf_key)
  sig=json.dumps([sorted(fixed.items()),limit_key],default=str)
  costs=list(map(self.row_cost,hits))
  broken=self.__dict__.setdefault("ceil_bad",set())
  if any(c is not None and c>ceiling+1e-9 for c in costs):
   broken.add(sig)
   return
  if sig in broken or shelf_key is None or len(shelves)!=1 or""in shelves or None in costs:
   return
  def on_every_row(text):
   want=text.strip().lower()
   return all(any(isinstance(cell,str)and cell.strip().lower()==want for f,cell in row.items()if not self.shelf_of({f:cell}))
for row in hits)
  template={}
  for name,val in plain.items():
   if name==limit_key or name==shelf_key:
    continue
   maker=isinstance(val,str)and val.strip()and name not in fixed and on_every_row(val)
   if not maker:
    template[name]=val
  free=sum(len(val.split())for name,val in template.items()if isinstance(val,str)and name not in fixed)
  schema=(self.shapes.get(self.role("search")or"")or{}).get("properties")or{}
  prior=self.__dict__.get("ceil_mode")
  if prior and prior["sig"]not in broken and prior["words"]<free:
   return
  arg_spec=schema.get(limit_key)or{}if isinstance(schema,dict)else{}
  self.ceil_mode=dict(sig=sig,tmpl=template,mode=fixed,shelf=shelf_key,bound=limit_key,words=free,
int=arg_spec.get("type")=="integer")
 @staticmethod
 def row_cost(entry):
  for name,v in entry.items():
   if not isinstance(v,bool)and is_number(v)and re.search(r"price|cost",str(name).lower()):
    return float(v)
  return None
 def ceil_spec(self):
  known=self.__dict__.get("ceil_mode")
  if known and known["sig"]not in self.__dict__.get("ceil_bad",set()):
   return known
  return None
 def cg_listing(self,call):
  known=self.ceil_spec()
  if not known or not isinstance(call,dict):
   return False
  fixed=known["mode"]
  if not fixed:
   return known["bound"]in call
  return all(call.get(name)==v for name,v in fixed.items())
 def page_size(self):
  return max((entry.get("size",0)for entry in self.__dict__.get("page_book",{}).values()),default=0)
 def visible_depth(self):
  pg=self.page_arg()
  per_page=self.page_size()
  if pg and per_page>0 and pg[1]!="token"and pg[3]is not None:
   span=pg[3]-pg[2]
   return int((span+1)*per_page)if pg[1]else int(span+per_page)
  return self.__dict__.get("page_cap")
 def raw_label(self,wanted):
  if not wanted:
   return""
  counts={}
  ids=self.archive.ids
  def visit(d):
   if not any("id"in str(k).lower()and isinstance(v,str)and v in ids for k,v in d.items()):
    return
   for k,v in d.items():
    low=str(k).lower()
    if not isinstance(v,str)or not 0<len(v.strip())<=60 or"id"in low:
     continue
    if re.search(r"categor|shelf|department|aisle",low):
     spelled=v.strip()
     if spelled.lower()in wanted:
      counts[spelled]=counts.get(spelled,0)+1
     return
  for rec in self.archive.items:
   walk_dicts(rec,visit)
  return min(counts,key=lambda s:(-counts[s],s),default="")
 def below(self,amount):
  if not is_number(amount):
   return None
  whole_limit=(self.ceil_spec()or{}).get("int")
  val=float(amount)
  if val.is_integer():
   under=int(val)-1
  else:
   under=int(val)if whole_limit else round(val-0.01,2)
  return None if under<=0 else under
 def ceil_mid(self,lo,hi):
  whole=bool((self.ceil_spec()or{}).get("int"))or all(float(x).is_integer()for x in(lo,hi))
  return(int(lo)+int(hi))//2 if whole else round((float(lo)+float(hi))/2,2)
 def ceil_args(self,known,shelf,limit):
  made=dict(known["tmpl"])
  made[known["shelf"]]=shelf
  made[known["bound"]]=int(limit)if known.get("int")or float(limit).is_integer()else limit
  return made
 def cg_products(self):
  count=len(self.needs.items())
  return list(range(count))if count else[None]
 def ceiling_ready(self,prod):
  if prod is None and self.needs.items():
   return False
  gates=(lambda:self.needs.priority()=="cheapest",
lambda:prod not in self.__dict__.get("ceil_walked",set()),
lambda:self.ceil_spec()is not None,
lambda:self.page_arg()is not None,
lambda:bool(self.raw_label(self.walk_home(prod))))
  return all(gate()for gate in gates)
 def ceil_aim(self,prod):
  chosen=self.walk_pick(prod)
  held=(self.__dict__.get("ceil_aims")or{}).get(prod)
  marks=[chosen[1]]if chosen else[]
  if is_number(held):
   marks.append(float(held))
  if marks or not self.bound_need(prod):
   return min(marks)if marks else None
  shelves=self.walk_home(prod)
  prices=[c for c,_ in self.cheap_unread(self.home_rows(shelves),prod)]
  book=self.archive
  for ident in book.read:
   if self.shelf_known(ident)not in shelves or not is_number(book.price(ident)):
    continue
   if ident in self.dead_lines()or ident in self.refused_lines or self.last_stock(ident)is False:
    continue
   prices.append(float(book.price(ident)))
  return min(prices)if prices else None
 def bound_need(self,prod):
  if not self.needs.items()or prod is None:
   return False
  def bounded(line):
   name,_,tail=str(line).partition(":")
   return self.product_of(name)==prod and bool(re.search(r"\d",tail))and bool(re.search(CH_BOUND,tail.lower()))
  return any(map(bounded,self.needs.props_left()))
 def _smt_rows(self,recs=None,ceiling=6):
  seen=[]
  for rec in(self.archive.items if recs is None else recs):
   walk_dicts(rec,seen.append,ceiling=ceiling)
  return seen
 @staticmethod
 def _smt_id_values(row):
  return[val for name,val in row.items()if isinstance(val,str)and"id"in str(name).lower()]
 @staticmethod
 def _smt_shelves(walk):
  return walk.get("row_shelf")or{}
 def _smt_on_home(self,walk,home):
  return any(shelf in home for shelf in self._smt_shelves(walk).values())
 def _smt_home_ids(self,walk,home):
  return[mark for mark,shelf in self._smt_shelves(walk).items()if shelf in home]
 def _smt_store(self,name,make):
  return self.__dict__.setdefault(name,make())
 def _smt_indexing(self):
  return self.__dict__.get("label_indexed")
 def _smt_search_schema(self):
  return self.shapes.get(self.role("search")or"")or{}
 @staticmethod
 def _smt_query_words(text):
  return[w for w in Errand._smt_toks(text)if w not in LINK_WORDS and len(w)>1]
 def _smt_book(self):
  return self.__dict__.get("page_book",{})
 @staticmethod
 def _smt_toks(text,pattern='[a-z0-9]+'):
  return re.findall(pattern,str(text).lower())
 def _smt_stemset(self,words):
  return set(map(self.stem,words))
 def _smt_carries(self,walk,k,alts=False):
  return self.walk_carries(walk["args"],k,alts=alts)
 def _smt_led(self,walk,k):
  found=self._smt_carries(walk,k)
  if found or self._smt_values(k):
   return found
  return{"bound"}if self._smt_bound_led(walk,k)else set()
 def _smt_bound_led(self,walk,k):
  lowered=[t.lower()for t in self.free_text(walk["args"])]
  tokens=[w for t in lowered for w in self._smp_rx_an.findall(t)][:self.lead_words or None]
  for _,req in self._smt_own_reqs(k):
   for piece in self._smt_choices(self._smt_bare_value(req)):
    if not re.search(CH_BOUND,piece):
     continue
    fig=self.fig_amount(re.sub(CH_BOUND," ",piece))
    if fig and fig[0]in tokens and any(t==fig[1]or self._smp_near(t,fig[1])for t in tokens):
     return True
  return False
 def _smt_id_ordered(self,walk):
  return walk.get("id_order")is True and not self.cg_listing(walk["args"])
 @staticmethod
 def _smt_add_new(seq,item):
  if item and item not in seq:
   seq.append(item)
 @staticmethod
 def _smt_bump(tally,key,step):
  tally[key]=tally.get(key,0)+step
 @staticmethod
 def _smt_unparen(value):
  text=str(value).lower().replace("\u2019","'")
  return re.sub(r"\([^)]*\)"," ",text).strip()
 @staticmethod
 def _smt_value_pieces(low):
  closed=re.search(r"\bor\b|\beither\b|/",low)is not None
  parts=re.split(r"[,;/]|\bor\b|\beither\b"if closed else r"[,;&+]|\band\b",low)
  return[p for p in parts if p.strip()][:3]if closed else parts[:1]
 def _smt_own_reqs(self,k):
  if not self.needs.items()or k is None:
   return list(enumerate(self.needs.props_left()))
  return[(n,req)for n,req in enumerate(self.needs.props_left())if self.product_of(req.partition(":")[0])in(k,None)]
 @staticmethod
 def _smt_bare_value(req):
  return re.sub(r"\((?:stated|revealed[^)]*)\)","",req.partition(":")[2]).lower()
 @staticmethod
 def _smt_choices(value):
  return re.split(r"[,;/]|\bor\b|\beither\b|\bnor\b",value)
 @staticmethod
 def _smt_ceil_div(num,den):
  return-(-num//den)
 def home_rows(self,home):
  marked=((self.row_mark(row),row)for row in self._smt_rows())
  return{mark for mark,row in marked if mark in self.archive.ids and self.shelf_of(row)in home}
 def ceil_clash(self,home,bound):
  for row in self._smt_rows():
   if self.shelf_of(row)not in home or not any(val in self.archive.ids for val in self._smt_id_values(row)):
    continue
   price=self.row_cost(row)
   if price is not None and price<=bound and self.stock_state(row)is True:
    return True
  return False
 def _smt_last_sold(self,pool):
  verdict={}
  for row in self._smt_rows():
   if not any(val in pool for val in self._smt_id_values(row)):
    continue
   state=self.stock_state(row)
   if state is not None:
    verdict[self.row_mark(row)]=state is False
  return verdict
 def _smt_vp_take(self,run,k,walk,composed):
  depth=run["depth"]
  reach=walk["total"]if depth is None else min(walk["total"],depth)
  pages=self._smt_ceil_div(reach,max(1,walk["size"]))-len(walk["read"])
  spent=run["used"].get(k,0)
  allowed=min(run["per"]-spent,run["cap"]-sum(run["used"].values()))
  if run["several"]:
   done_opens=run["opened"].get(k,0)
   allowed=min(allowed,run["share"]-spent-done_opens-(0 if run["opened"].get(k)else run["opens"]))
  if pages<=allowed:
   return pages
  return allowed if composed else 0
 def _smt_flush_pool(self,prod,book,booked,tally,each):
  home=self.walk_home(prod)
  ids=set()
  for key in booked.get(str(prod),[]):
   if key in book and self._smt_led(book[key],prod):
    ids.update(self._smt_home_ids(book[key],home))
  pick=self.walk_pick(prod)
  ranked=[mark for _,mark in self.cheap_unread(ids,prod,pick and pick[1])]
  return ranked[:max(0,min(each,CEIL_OPENS-tally.get(prod,0)))]
 @staticmethod
 def _smt_deal(due,pools,limit):
  owner={}
  queue=[]
  slot=0
  while slot or(len(queue)<limit and any(pools)):
   pool=pools[slot]
   while pool and pool[0]in queue:
    pool.pop(0)
   if pool and len(queue)<limit:
    mark=pool.pop(0)
    owner[mark]=due[slot]
    queue.append(mark)
   slot=(slot+1)%len(due)
  return owner,queue
 def cheap_unread(self,ids,k,below=None):
  pool=set(ids)
  if not pool:
   return[]
  lines=self.needs.items()
  units=self.needs.quantity()if k is None or k>=len(lines)else lines[k]["quantity"]
  cap=self.needs.spend_cap
  limited=is_number(cap)and is_number(units)
  sold=self._smt_last_sold(pool)
  def keep(mark,price):
   if mark in self.archive.read or mark in self.refused_lines or sold.get(mark)or not is_number(price):
    return False
   if limited and price*units>cap:
    return False
   return below is None or price<below
  return sorted((float(price),mark)for mark,price in((m,self.archive.price(m))for m in pool)if keep(mark,price))
 def all_words(self):
  if self.word_kind=="all":
   return True
  return self.word_kind!="any"and bool(self.lead_words)
 def valued_ids(self,k):
  if not self.all_words():
   return set()
  home=self.walk_home(k)
  chosen=(w for w in self._smt_book().values()if self._smt_id_ordered(w)and self._smt_carries(w,k))
  return{mark for w in chosen for mark in self._smt_home_ids(w,home)}
 def valued_complete(self,k):
  try:
   if not self.all_words():
    return False
   home=self.walk_home(k)
   single=set()
   for n,req in self._smt_own_reqs(k):
    if sum(1 for alt in self._smt_choices(self._smt_bare_value(req))if alt.strip())==1:
     single.add(n)
   for walk in self._smt_book().values():
    done=self._smt_id_ordered(walk)and walk.get("first")and self.next_page(walk)is None and len(self._smt_shelves(walk))>=walk["total"]
    if done and self._smt_on_home(walk,home)and self._smt_carries(walk,k)&single:
     return True
  except Exception:
   return False
  return False
 def row_carries(self,ident,k):
  word_sets=[]
  amounts=[]
  for _,req in self._smt_own_reqs(k):
   value=self._smt_bare_value(req)
   if ref_text(value):
    continue
   for alt in self._smt_choices(value):
    amount=self.fig_amount(alt)
    if amount:
     amounts.append(amount)
    elif not self.amount_part(alt):
     stems=[self.stem(w)for w in re.findall(r"[a-z0-9]{3,}",alt)]
     if stems:
      word_sets.append(stems)
  if not(word_sets or amounts):
   return False
  texts=[]
  for row in self._smt_rows():
   if ident in self._smt_id_values(row):
    texts+=[str(val).lower()for val in row.values()if isinstance(val,str)]
  body=" ".join(texts)
  for fig,unit in amounts:
   spelled=r"(?<![\d.,])%s\s*-?\s*%s"%(re.escape(fig),re.escape(self.stem(unit)))
   if re.search(spelled,body):
    return True
  present={self.stem(w)for w in re.findall(r"[a-z0-9]{3,}",body)}
  return any(set(stems)<=present for stems in word_sets)
 def _smt_room(self):
  burst=0
  if self.asking and self.role("message"):
   burst=max(self.ask_room()-self.msgs_sent,0)
  spare=self.moves_left()-self.order_due()
  return burst,spare-burst*2
 def _smt_piece_words(self,piece,own):
  amount=self.fig_amount(piece)
  if re.search(EXCLUDE_WORDS,piece)or(self.amount_part(piece)and not amount):
   return[]
  lead,unit=amount if amount else(None,None)
  text=re.sub(r"\b(?:"+VALUE_LINKS+r")\b"," ",re.sub(r"'s\b","",piece))
  kept=[]
  for word in re.findall(r"\d+(?:[.,]\d+)+|[a-z0-9]+",text):
   if"."in word or","in word or word in LINK_WORDS:
    continue
   if(len(word)>=2 or word==lead)and(word not in own or word==unit):
    kept.append(word)
  return kept
 def _smt_values(self,k):
  found=[]
  for head,value in self.code_values(k):
   low=self._smt_unparen(value)
   if low and not ref_text(low):
    own=set(self._smt_toks(head))
    options=[]
    for piece in self._smt_value_pieces(low):
     self._smt_add_new(options,self._smt_piece_words(piece,own))
    self._smt_add_new(found,options)
  return found
 def _smt_label_words(self,k,label):
  plain=re.sub(r"'s\b","",label.lower().replace("\u2019","'"))
  words=list(dict.fromkeys(self._smt_query_words(plain)))
  if self._smt_indexing()in(True,None):
   return words
  home=self.walk_home(k)
  titles=[]
  for row in self._smt_rows():
   if self.shelf_of(row)in home and self.row_mark(row)in self.archive.ids:
    titles.append(self._smt_stemset(self._smt_toks(self.title_of(row))))
  if titles:
   common=[w for w in words if 2*sum(self.stem(w)in t for t in titles)>=len(titles)]
   if common:
    return common
  lines=self.needs.items()
  if lines and k is not None and k<len(lines):
   kind=lines[k]["product"]
  else:
   kind=self.needs.reading.get("product")or""
  named=[w for w in self._smt_toks(kind)if len(w)>2 and w not in LINK_WORDS]
  return named[-1:]
 def _smt_forms(self,k,label,cap_len):
  values=self._smt_values(k)
  if not values:
   return[],[]
  tail=self._smt_label_words(k,label)
  spread=self.needs.priority()=="cheapest"
  def compose(chosen):
   firsts={}
   for alt in chosen+[tail]:
    for word in alt:
     firsts.setdefault(self.stem(word),word)
   kept=list(firsts.values())
   text=" ".join(kept[:self.lead_words]if self.lead_words else kept)
   if not is_number(cap_len):
    return text
   while" "in text and len(text)>cap_len:
    text=text.rsplit(" ",1)[0]
   return text[:int(cap_len)]
  def usable(options):
   return options[:]if spread else options[:1]
  def unique(texts):
   return list(dict.fromkeys(filter(None,texts)))
  single=unique([compose([alt])for alt in usable(values[0])])
  if len(values)==1:
   return single,[]
  wide=0
  for n,options in enumerate(values):
   if len(usable(options))>1:
    wide=n
    break
  heads=[options[0]for options in values]
  combined=[compose(heads[:wide]+[alt]+heads[wide+1:])for alt in usable(values[wide])]
  return unique(combined),single
 def _smt_next_form(self,first,second,tkey,mode):
  for forms in(first,second):
   totals=[self._smt_sent_total(form,tkey,mode)for form in forms]
   unsent=[form for form,total in zip(forms,totals)if total is None]
   if unsent:
    return unsent[0]
   if any(totals):
    return""
  return None if(first or second)else""
 def vl_mode(self,args):
  fields=self._smt_search_schema().get("properties")
  if not isinstance(fields,dict):
   fields={}
  picked=[]
  for name,val in args.items():
   spec=fields.get(name)
   if isinstance(val,str)and isinstance(spec,dict)and isinstance(spec.get("enum"),list)and len(spec["enum"])<=4:
    picked.append((name,val))
  return tuple(sorted(picked))
 def _smt_stems(self,text):
  words=re.findall(r"[a-z0-9]+",str(text).lower())
  return frozenset(map(self.stem,words[:self.lead_words or None]))
 def _smt_sent_total(self,text,tkey,mode):
  target=self._smt_stems(text)
  for args,total in self._smt_sent_calls():
   said=args.get(tkey)
   if isinstance(said,str)and all(args.get(name)==want for name,want in mode.items())and self._smt_stems(said)==target:
    return int(total)
  return None
 def _smt_sent_calls(self):
  calls=[]
  for sent,total in(self.__dict__.get("ceil_seen")or{}).items():
   try:
    args=json.loads(sent)
   except Exception:
    continue
   if isinstance(args,dict):
    calls.append((args,total))
  return calls
 def _smt_note_indexing(self,label):
  if self._smt_indexing()is not None or not self.archive.items:
   return
  wanted=self._smt_stemset(self._smt_query_words(re.sub(r"'s\b","",label.lower())))
  shelf=label.strip().lower()
  rows=[row for row in self._smt_rows([self.archive.items[-1]],3)
if self.row_mark(row)in self.archive.ids and self.shelf_of(row)==shelf]
  count=0
  for row in rows:
   texts=[val for name,val in row.items()
if isinstance(val,str)and not re.search(r"categor|shelf|department|aisle",str(name).lower())]
   elsewhere=self._smt_stemset(w for val in texts for w in self._smt_toks(val))
   count+=1 if wanted-elsewhere else 0
  if count>1:
   self.label_indexed=True
 def _smt_trim(self,queue,batch,floor):
  while queue:
   calls=1 if batch and len(queue)>=2 else len(queue)
   if calls<self.moves_left()-floor:
    break
   queue.pop()
  return queue
 def _smt_priced(self,queue):
  return", ".join(f"{mark} ({self.archive.price(mark):g})"for mark in queue)
 def _smt_tack(self,text,status):
  if not(text and self.chat and self.chat[-1].get("role")=="tool"):
   return
  last=self.chat[-1]
  last["content"]=str(last["content"])+"\n"+text+("\n"+self.status_line()if status else"")
 def vl_queue(self,k,ids,floor,most):
  cheapest=self.needs.priority()=="cheapest"
  under=None
  if cheapest:
   pick=self.walk_pick(k)
   under=pick[1]if pick else None
  batch=self.batch_tool()
  tally=self._smt_store("valued_opens",dict)
  ranked=[mark for _,mark in self.cheap_unread(ids,k,under)]
  quota=CEIL_OPENS-tally.get(k,0)if cheapest else most
  queue=self._smt_trim(ranked[:max(0,min(batch[2]if batch else most,quota))],batch,floor)
  if cheapest:
   self._smt_bump(tally,k,len(queue))
  return queue
 def vl_launch(self,queue,homes,floor,say):
  stale={mark for mark in self.rank_checked if self.shelf_known(mark.partition("|")[0])in homes}
  self.rank_checked=self.rank_checked-stale
  for name,value in(("order_vetted",False),("open_queue",queue),("open_recheck",True),("open_say",say),
("open_purpose","walk"),("open_floor",floor)):
   setattr(self,name,value)
  self.open_named()
 def vl_search(self):
  if self._smt_room()[1]<=4:
   return False
  schema=self._smt_search_schema()
  fields=schema.get("properties")
  shared={"schema":schema,"props":fields if isinstance(fields,dict)else{},"found":self.page_arg(),
"tries":self.__dict__.setdefault("label_tries",{}),
"settled":self.__dict__.setdefault("label_done",{}),
"sets":self.__dict__.setdefault("label_sets",{})}
  shared["products"]=self.cg_products()
  handlers={"open":self._smt_vs_open,"shape":self._smt_vs_shape,"forms":self._smt_vs_forms,
"fallback":self._smt_vs_fallback,"check":self._smt_vs_check,"send":self._smt_vs_send,
"report":self._smt_vs_report}
  for k in shared["products"]:
   job=dict(shared,k=k)
   state="open"
   while state in handlers:
    state=handlers[state](job)
   if state=="sent":
    return True
   if state=="settle":
    job["settled"][str(k)].append(job["sig"])
  return False
 def _smt_vs_open(self,job):
  k=job["k"]
  if job["tries"].get(k,0)>=LABEL_TRIES:
   return"skip"
  job["sig"]=json.dumps(self._smt_values(k))
  if job["sig"]in job["settled"].setdefault(str(k),[]):
   return"skip"
  job["home"]=self.walk_home(k)
  job["label"]=self.raw_label(job["home"])
  return"shape"if job["label"]else"skip"
 def _smt_vs_shape(self,job):
  chosen=None
  for call in self.code_queries(job["k"]):
   texts=self.free_text(call)
   keys=[name for name,val in call.items()if isinstance(val,str)and val.strip()and val in texts]
   if len(keys)==1 and not self.cg_listing(call):
    chosen=(call,keys[0])
    break
  if chosen is None:
   return"settle"
  call,tkey=chosen
  props,schema=job["props"],job["schema"]
  job["mode"]=mode={name:val for name,val in call.items()if isinstance(props.get(name),dict)and props[name].get("enum")}
  needed=schema.get("required")or[]
  job["tmpl"]={name:val for name,val in call.items()if name==tkey or name in mode or name in needed}
  job["tkey"]=tkey
  job["cap_len"]=props[tkey].get("maxLength")if isinstance(props.get(tkey),dict)else None
  return"forms"
 def _smt_vs_plan(self,job):
  k=job["k"]
  job["first"],job["second"]=self._smt_forms(k,job["label"],job["cap_len"])
  job["sets"].setdefault(str(k),set()).update(self._smt_stems(q)for q in job["first"]+job["second"])
  return self._smt_next_form(job["first"],job["second"],job["tkey"],job["mode"])
 def _smt_vs_forms(self,job):
  job["todo"]=self._smt_vs_plan(job)
  return"fallback"if job["todo"]is None else"check"
 def _smt_vs_fallback(self,job):
  job["todo"]=""
  k,home=job["k"],job["home"]
  hit=any(self._smt_carries(walk,k)and self._smt_on_home(walk,home)for walk in self._smt_book().values())
  if hit and self.__dict__.get("label_indexed")is None:
   self.label_indexed=False
   job["todo"]=self._smt_vs_plan(job)or""
  return"check"
 def _smt_vs_check(self,job):
  if not job["todo"]:
   return"settle"
  args,faults=ArgCheck.execute({**job["tmpl"],job["tkey"]:job["todo"]},job["schema"])
  if faults or not isinstance(args,dict):
   return"settle"
  job["args"]=args
  job["tries"][job["k"]]=job["tries"].get(job["k"],0)+1
  return"send"
 def _smt_vs_send(self,job):
  k,todo,args=job["k"],job["todo"],job["args"]
  plural="s"if len(self._smt_values(k))>1 else""
  self.walk_call(args,f"Running '{todo}' as a keyword query: required value{plural} first, then the category name in "
"the store's own spelling. Matching covers titles, categories and listing fields, so the reply "
"lists in-category items with that value.")
  job["total"]=self._smt_sent_total(todo,job["tkey"],job["mode"])
  self._smt_note_indexing(job["label"])
  found=job["found"]
  trimmed=dict(args)
  if found:
   trimmed.pop(found[0],None)
  job["key"]=key=json.dumps(trimmed,sort_keys=True,default=str)
  job["walk"]=walk=self._smt_book().get(key)
  job["every"]=bool(walk)and len(self._smt_shelves(walk))>=walk["total"]and self._smt_on_home(walk,job["home"])
  job["floor"]=self.order_due()+2
  return"report"
 def _smt_vs_report(self,job):
  k,todo,total,walk,home=job["k"],job["todo"],job["total"],job["walk"],job["home"]
  if total is None:
   note=""
  elif not total:
   note=f"[value search] '{todo}': zero matches found."
  else:
   note=f"[value search] '{todo}': {total} match(es)"+(" - all of them on the page."if job["every"]else".")
  queue=[]
  if job["every"]:
   if self.needs.priority()=="cheapest":
    if job["found"]and walk.get("id_order")is not False:
     queue=self._smt_vs_book(job)
   elif todo in(job["first"]or job["second"]):
    queue=self.vl_queue(k,self._smt_home_ids(walk,home),job["floor"],LABEL_OPENS)
  if queue:
   below=" below the pick"if self.walk_pick(k)and self.needs.priority()=="cheapest"else""
   note+=(" Still unread, in stock, inside the budget, on the requested category"+below
+", cheapest first, are opened next: "+self._smt_priced(queue)+".")
  self._smt_tack(note,not queue)
  self.log.append({"kind":"label_search","product":k,"query":todo,"total":total,"every":job["every"],
"opened":queue})
  if queue:
   self.vl_launch(queue,home,job["floor"],"Reading the cheapest unread hits next; every category item with "
"the required values is among these matches.")
  return"sent"
 def _smt_vs_book(self,job):
  k,key=job["k"],job["key"]
  self._smt_store("valued",set).add(key)
  self._smt_store("valued_keys",dict).setdefault(str(k),[]).append(key)
  if len(job["products"])>1:
   self._smt_store("valued_wait",dict)[k]=True
   return[]
  return self.vl_queue(k,self._smt_home_ids(job["walk"],job["home"]),job["floor"],CEIL_OPENS)
 def vl_flush(self,floor):
  flags=self.__dict__.setdefault("valued_wait",{})
  due=[prod for prod,flag in flags.items()if flag]
  flags.clear()
  if not due:
   return False
  book,booked=self._smt_book(),self.__dict__.get("valued_keys")or{}
  tally=self.__dict__.setdefault("valued_opens",{})
  batch=self.batch_tool()
  each=2 if self.moves_left()-floor-1>=2*len(due)else 1
  pools=[self._smt_flush_pool(prod,book,booked,tally,each)for prod in due]
  owner,queue=self._smt_deal(due,pools,each*len(due)if not batch else batch[2])
  if not self._smt_trim(queue,batch,floor):
   return False
  for mark in queue:
   self._smt_bump(tally,owner[mark],1)
  homes=set()
  for prod in due:
   homes|=self.walk_home(prod)
  self.vl_launch(queue,homes,floor,"Cheapest asked: each product now gets its lowest-priced unread hits "
"from its value query read.")
  self._smt_tack("[value searches read] Now reading, per product, under its pick: unread hits in stock, inside the "
"budget, cheapest first, from its value query - "+self._smt_priced(queue)+". Per product, pick "
"the lowest-priced listing whose own fields pass each requirement that is now in force; a fit that "
"costs more needs every cheaper candidate opened and rejected first.",False)
  self.log.append({"kind":"value_flush","products":due,"opened":queue})
  return True
 def product_steps(self):
  try:
   lines=self.needs.items()
   if len(lines)<2 or self.needs.priority()!="cheapest":
    return""
   pages=self.__dict__.get("valued_by")or{}
   parts=[]
   lacking=""
   for k,line in enumerate(lines):
    has=bool(self.walk_pick(k))
    name=str(line["product"])
    parts.append(f"{name}: {pages.get(k, 0)} value page(s) read, pick {('no', 'yes')[has]}")
    if not lacking and not has:
     lacking=name
   tail=f"; next: {lacking}"if lacking else""
   return"\n[product steps] "+"; ".join(parts)+tail+"."
  except Exception:
   return""
 def value_pages(self):
  phases={"gate":self._smt_vp_gate,"label":self._smt_vp_label,"setup":self._smt_vp_setup,
"scan":self._smt_vp_scan,"pick":self._smt_vp_pick}
  run={"result":False}
  phase="gate"
  try:
   while phase in phases:
    phase=phases[phase](run)
  except Exception as err:
   reason="%s: %s"%(type(err).__name__,err)
   self.log.append({"kind":"value_pages_error","error":reason[:300]})
   return False
  return run["result"]
 def _smt_vp_gate(self,run):
  if any(ref.get("status")=="open"for ref in self.__dict__.get("refs")or[])and self.ref_room():
   return"stop"
  blocked=(lambda:self.orders_tried,lambda:self.refusal_n,lambda:self.wire.done,lambda:self.wire.broke,
lambda:self.wire.spare()<CLOSING+60,lambda:not self.role("search"))
  return"stop"if any(test()for test in blocked)else"label"
 def _smt_vp_label(self,run):
  if self.vl_search():
   run["result"]=True
   return"stop"
  if self.needs.priority()!="cheapest"or self.word_kind=="any":
   return"stop"
  run["found"]=self.page_arg()
  return"setup"if run["found"]else"stop"
 def _smt_vp_setup(self,run):
  store=self.__dict__
  run["book"]=store.get("page_book",{})
  run["done"]=self._smt_store("valued",set)
  run["used"]=used=self._smt_store("valued_by",dict)
  run["opened"]=self._smt_store("valued_opens",dict)
  run["wait"]=self._smt_store("valued_wait",dict)
  run["sets"]=store.get("label_sets")or{}
  run["depth"]=depth=self.visible_depth()
  rows_per_page=max(1,self.page_size())
  run["per"]=VALUE_PAGES
  if depth:
   run["per"]=max(run["per"],self._smt_ceil_div(depth,rows_per_page)-1)
  run["products"]=products=self.cg_products()
  count=len(products)
  run["several"]=count>1
  run["cap"]=max(VALUE_TOTAL,count*VALUE_TOTAL_PER)
  batched=self.batch_tool()
  run["opens"]=1 if batched else 2 if count>1 else CEIL_OPENS
  run["floor"]=2+self.order_due()
  run["known"]=self.all_words()
  run["left"],run["room"]=self._smt_room()
  untouched=sum(1 for k in products if self.walk_home(k)and not used.get(k))
  run["share"]=max(4,run["room"]//(untouched or 1))
  return"scan"
 def _smt_vp_scan(self,run):
  run["options"],run["pending"]=[],False
  for n,k in enumerate(run["products"]):
   home=self.walk_home(k)
   if home:
    for take,score,key in self._smt_vp_product(run,k,home):
     run["options"].append((run["used"].get(k,0),take,score,n,key,k))
  return"pick"
 def _smt_vp_pick(self,run):
  if run["options"]:
   _,take,_,_,key,k=min(run["options"],key=lambda o:o[:5])
   run["done"].add(key)
   run["result"]=self.read_value_search(k,key,run["floor"],take,run["several"])
  elif run["wait"]and not run["pending"]and not(run["left"]and run["room"]>run["opens"]*len(run["wait"])+4):
   run["result"]=self.vl_flush(run["floor"])
  return"stop"
 def _smt_vp_product(self,run,k,home):
  prod={"k":k,"home":home,"mine":run["sets"].get(str(k))or set()}
  prod["values"]={self.stem(w)for _,req in self._smt_own_reqs(k)
for w in re.findall(r"[a-z0-9]{2,}",req.partition(":")[2].lower())}
  prod["label"]={self.stem(w)for shelf in home for w in re.findall(r"[a-z0-9]{2,}",shelf)}
  prod["whole"]=[(self._smt_stems(" ".join(self.free_text(w["args"]))),self._smt_carries(w,k,alts=True),
self.vl_mode(w["args"]))for w in run["book"].values()if self._smt_vp_whole(w)]
  picked=[]
  for key,walk in run["book"].items():
   weighed=self._smt_vp_weigh(run,prod,key,walk)
   if weighed=="burst":
    run["pending"]=True
   elif weighed:
    picked.append(weighed)
  if run["known"]:
   own=prod["values"]|prod["label"]
   picked=[c for c in picked if not self._smt_vp_nested(c[3],picked,own)]
  return[(take,score,key)for take,score,key,_ in picked]
 def _smt_vp_whole(self,walk):
  return walk.get("id_order")is True and bool(walk.get("first"))and self.next_page(walk)is None and not self.cg_listing(walk["args"])and len(self._smt_shelves(walk))>=walk["total"]
 @staticmethod
 def _smt_vp_nested(words,picked,own):
  for other in picked:
   theirs=other[3]
   if theirs<words and not(words-theirs)<=own:
    return True
   if words<theirs and(theirs-words)<=own:
    return True
  return False
 def _smt_vp_weigh(self,run,prod,key,walk):
  k=prod["k"]
  gates=(lambda:key not in run["done"],lambda:walk.get("id_order")is True,lambda:bool(walk.get("first")),
lambda:self.next_page(walk)is not None,lambda:not self.cg_listing(walk["args"]),
lambda:bool(self._smt_led(walk,k)),lambda:self._smt_on_home(walk,prod["home"]))
  if not all(gate()for gate in gates):
   return None
  words=self._smt_stems(" ".join(self.free_text(walk["args"])))
  composed=words in prod["mine"]
  known,depth=run["known"],run["depth"]
  if depth is None:
   if not known:
    return None
  elif walk["total"]>depth and not(known and composed):
   return None
  pairs=self._smt_carries(walk,k,alts=True)
  how=self.vl_mode(walk["args"])
  own=prod["values"]|prod["label"]
  if known:
   for seen,got,mode in prod["whole"]:
    if mode==how and(seen<=words or(pairs and seen<=own and pairs<=got)):
     return None
  take=self._smt_vp_take(run,k,walk,composed)
  if take<=0 or take+run["opens"]+run["floor"]>=self.moves_left():
   return None
  if(take>2 or run["several"])and run["left"]and run["room"]-take-run["opens"]>4:
   return"burst"
  return(take,-self.walk_score(walk["args"],k),key,words)
 _smu_skip=object()
 _smu_low_rule=("Choosing: the cheapest offer whose own record meets all requirements in force wins; any pricier "
"match first needs each cheaper candidate opened and turned down.")
 @staticmethod
 def _smu_call_key(call):
  return json.dumps(call,default=str,sort_keys=True)
 def _smu_store(self,name,make):
  return self.__dict__.setdefault(name,make())
 def _smu_own(self,name,empty=None):
  return self.__dict__.get(name,empty)
 def _smu_trouble(self,kind,err):
  self.log.append({"kind":kind,"error":(type(err).__name__+": "+str(err))[:300]})
 def _smu_lowest_wanted(self):
  return self.needs.priority()=="cheapest"
 def _smu_price_of(self,ident):
  price=self.archive.price(ident)
  return float(price)if is_number(price)else None
 @staticmethod
 def _smu_row_shelves(walk):
  return walk.get("row_shelf")or{}
 def _smu_can_list(self,floor):
  line=self.wire
  return not(line.done or line.broke)and line.spare()>=CLOSING+30 and self.moves_left()-1>floor
 def _smu_wire_short(self):
  return self.wire.done or self.wire.spare()<CLOSING+60
 def _smu_open_cost(self):
  return 1 if self.batch_tool()else 2
 @staticmethod
 def _smu_raise(table,key,value):
  old=table.setdefault(key,0)
  if value>old:
   table[key]=value
 def _smu_trace(self,kind,**fields):
  self.log.append(dict(kind=kind,**fields))
 def _smu_last_result(self):
  last=self.chat[-1]if self.chat else None
  return last if last is not None and last.get("role")=="tool"else None
 def _smu_extend_result(self,text,status=False):
  last=self._smu_last_result()
  if last is not None:
   last["content"]="%s\n%s"%(last["content"],text+"\n"+self.status_line()if status else text)
 def _smu_page_run(self,walk,base,slot,say,floor,room,on_try=None):
  tried=got=0
  state="probe"
  while state!="stop":
   if state=="probe":
    ok_to_go=tried<room and self.next_page(walk)is not None and self._smu_can_list(floor)
    state="fetch"if ok_to_go else"stop"
    continue
   tried+=1
   if on_try is not None:
    on_try()
   here=self.next_page(walk)
   if self.walk_call(dict(base,**{slot:here}),say):
    got+=1
    state="stop"if here==self.next_page(walk)else"probe"
   else:
    state="stop"
  return tried,got
 def read_value_search(self,prod,search_key,floor,most=VALUE_PAGES,defer=False):
  slot=self.page_arg()
  walk=self._smu_own("page_book",{}).get(search_key)
  if not(slot and walk):
   return False
  words=" ".join(self.free_text(walk["args"]))
  counts=self._smu_store("valued_by",dict)
  strict=self.all_words()
  if strict:
   say=("Paging on through '%s' (store sorts by id, not price; text queries here need every word), so that each "
"offer with those words is seen - cheapest wanted."%words)
  else:
   say="Paging on through '%s' (store sorts by id, not price), so that each offer it matched is seen - cheapest wanted."%words
  def bump():
   counts[prod]=1+counts.get(prod,0)
  tried,got=self._smu_page_run(walk,walk["args"],slot[0],say,floor,most,bump)
  if got==0:
   return tried>0
  self._smu_store("valued_keys",dict).setdefault(str(prod),[]).append(search_key)
  shown=len(self._smu_row_shelves(walk))
  complete=self.next_page(walk)is None and walk["total"]<=shown
  home=self.walk_home(prod)
  if complete:
   tail=" (complete for those words)"if strict else" (complete)"
  elif self.next_page(walk)is None:
   tail=" (further pages unreachable: something cheaper may lie past them)"
  else:
   tail=""
  head="[value search read] '%s' -> %d/%s%s. "%(words,shown,walk["total"],tail)
  opened=[]
  if defer:
   self._smu_store("valued_wait",dict)[prod]=True
   body="Deferred: after every product's value search is read, its cheapest unread entries (in stock, inside budget) get opened. "
  else:
   has_pick=self.walk_pick(prod)
   mine=[i for i,shelf in self._smu_row_shelves(walk).items()if shelf in home]
   opened=self.vl_queue(prod,mine,floor,CEIL_OPENS)
   if not opened:
    body="To open: nothing - every requested-category entry is read, sold out or over budget. "
   else:
    priced=", ".join("%s (%s)"%(i,format(self.archive.price(i),"g"))for i in opened)
    body=("To open now (requested category, cheapest first"+(", all below the pick"if has_pick else"")
+"): "+priced+". ")
  summary=head+body+self._smu_low_rule
  if opened:
   self.vl_launch(opened,home,floor,"Opening that query's cheapest unread entries.")
  self._smu_extend_result(summary,not opened)
  self._smu_trace("value_pages",product=prod,query=words,pages=got,complete=complete,opened=opened,defer=defer)
  return True
 def ceiling_walk(self,prod,floor,repair=False):
  spec,slot=self.ceil_spec(),self.page_arg()
  home=self.walk_home(prod)
  label=self.raw_label(home)
  aim=self.ceil_aim(prod)
  if aim is None or not(spec and slot and label):
   return None
  self._smu_store("ceil_walked",set).add(prod)
  pick_now=self.walk_pick(prod)
  aimed_at=pick_now[0]if pick_now else""
  self._smu_store("ceil_aimed",dict)[prod]=float(aim),aimed_at
  limit=self.below(aim)
  if limit is None:
   return None
  if repair:
   floor+=2
  elif not self.orders_tried:
   floor=max(self.order_due()+2,floor)
  held_aim=is_number((self._smu_own("ceil_aims")or{}).get(prod))
  what="the pick"if pick_now or held_aim else"the cheapest row seen"
  say="Category '%s' via the store's price cap, below %s (%s) - cheapest wanted."%(label,what,format(aim,"g"))
  run={"spec":spec,"slot":slot[0],"label":label,"home":home,"floor":floor}
  run["depth"]=self.visible_depth()
  run["book"]=self._smu_store("page_book",dict)
  run.update({"keys":[],"calls":0,"top":None,"complete":False,"ran":False,"bound":limit,"last":limit,
"say":say})
  handlers={"list":self._smu_cw_list,"check":self._smu_cw_check,"empty":self._smu_cw_empty,
"tail":self._smu_cw_tail,"lower":self._smu_cw_lower}
  state="list"
  while state in handlers:
   state=handlers[state](run)
  if not run["ran"]:
   return None
  seen=set()
  for done_key in run["keys"]:
   shelf=self._smu_row_shelves(run["book"].get(done_key,{}))
   seen.update(i for i in shelf if shelf[i]in home)
  cands,ranked=self._smu_rank_rows(seen,prod,aim)
  top=ranked[:CEIL_OPENS]
  aim_txt=format(aim,"g")
  if not run["complete"]:
   report=("cap listing '%s' (< %s): %d cheaper found; openable (unread, stocked, in budget): %d"
%(label,aim_txt,len(seen),len(cands)))
  elif seen:
   report=("cap listing '%s': complete to %s, %d found; openable (unread, stocked, in budget): %d"
%(label,format(run["last"],"g"),len(seen),len(cands)))
  else:
   report="cap listing '%s' (< %s): zero listings undercut %s"%(label,aim_txt,what)
  self._smu_trace("ceiling_walk",product=prod,label=label,aim=aim,calls=run["calls"],complete=run["complete"],
rows=len(seen),ranked=top)
  return report,top
 def _smu_cw_list(self,run):
  if not(run["calls"]<CEIL_CALLS and self._smu_can_list(run["floor"])):
   return"end"
  call=self.ceil_args(run["spec"],run["label"],run["bound"])
  ckey=self._smu_call_key(call)
  self._smu_store("ceil_keys",set).add(ckey)
  run["calls"]+=1
  pages=run["book"]
  reuse=ckey in pages and bool(pages[ckey].get("first"))and ckey in(self._smu_own("ceil_seen")or{})
  if not(reuse or self.walk_call(call,run["say"])):
   return"end"
  run.update(ran=True,last=run["bound"],args=call,key=ckey)
  run["keys"].append(ckey)
  return"check"
 def _smu_cw_check(self,run):
  walk=run["walk"]=run["book"].get(run["key"])
  if walk:
   for ident in self._smu_row_shelves(walk):
    cost=self.archive.price(ident)
    if is_number(cost)and cost>run["bound"]:
     self._smu_store("ceil_bad",set).add(run["spec"]["sig"])
     break
  if self.ceil_spec()is None:
   return"end"
  total=self._smu_own("ceil_seen",{}).get(run["key"])
  if total is None:
   return"end"
  run["total"]=total
  if not(total and walk):
   return"empty"
  pages,part=divmod(total,max(1,walk["size"]))
  if part:
   pages+=1
  shallow=run["depth"]is None or run["depth"]>=total
  if shallow and CEIL_CALLS-run["calls"]>=pages-len(walk["read"]):
   return"tail"
  return"lower"
 def _smu_cw_empty(self,run):
  if self.ceil_clash(run["home"],run["bound"]):
   self._smu_store("ceil_bad",set).add(run["spec"]["sig"])
   return"end"
  if run["top"]is None:
   run["complete"]=True
   return"end"
  mid=self.ceil_mid(run["bound"],run["top"])
  if not run["bound"]<mid<run["top"]:
   return"end"
  run["bound"]=mid
  return"list"
 def _smu_cw_tail(self,run):
  walk=run["walk"]
  spent,_=self._smu_page_run(walk,run["args"],run["slot"],"Paging on (price-cap listing).",
run["floor"],CEIL_CALLS-run["calls"])
  run["calls"]+=spent
  run["complete"]=self.next_page(walk)is None
  return"end"
 def _smu_cw_lower(self,run):
  walk,cap=run["walk"],run["bound"]
  run["top"]=cap
  costs=[c for c in map(self._smu_price_of,self._smu_row_shelves(walk))if c is not None]
  floor_price=min(costs)if costs else 0.0
  readable=max(1,walk["size"])*max(1,CEIL_CALLS-run["calls"])
  spread=floor_price+1.8*(cap-floor_price)*readable/max(run["total"],1)
  nxt=min(self.ceil_mid(floor_price,cap),self.ceil_mid(floor_price,spread))
  if nxt<cap:
   run["bound"]=nxt
   return"list"
  return"end"
 def _smu_rank_rows(self,pool,prod,aim):
  cands=[ident for _,ident in self.cheap_unread(pool,prod,aim)]
  valued=self.valued_ids(prod)
  first=[c for c in cands if c in valued]
  others=[c for c in cands if c not in valued]
  carried=[c for c in others if self.row_carries(c,prod)]
  tail=[]if self.valued_complete(prod)else[c for c in others if c not in carried]
  return cands,first+carried+tail
 def ceiling_sweep(self,tried):
  spec,slot=self.ceil_spec(),self.page_arg()
  if self.needs.items()if spec and slot else True:
   return None,False
  rows,shelves=self.shelf_rows(),self.shelf_labels()
  label=self.raw_label(shelves)
  if not(rows and label):
   return None,False
  cap=self.below(min(cost for cost,_shown,_stock in rows.values()))
  if cap is None:
   return None,True
  call=self.ceil_args(spec,label,cap)
  ckey=self._smu_call_key(call)
  if ckey not in tried:
   self._smu_store("ceil_keys",set).add(ckey)
   return call,False
  walk=self._smu_own("page_book",{}).get(ckey)
  if not walk:
   return None,self._smu_own("ceil_seen",{}).get(ckey)==0 and not self.ceil_clash(set(shelves),cap)
  here=self.next_page(walk)
  if here is None:
   return None,walk["total"]<=len(self._smu_row_shelves(walk))
  more=dict(call,**{slot[0]:here})
  return(more if self._smu_call_key(more)not in tried else None),False
 def _smu_guard_target(self,name,args):
  gates=(lambda:name==self.role("details"),self._smu_lowest_wanted,self.all_words,lambda:not self.in_reserve())
  if not all(gate()for gate in gates):
   return None
  known=self.archive.ids
  named=[v for v in args.values()if isinstance(v,str)and v in known]
  if len(named)!=1:
   return None
  ident=named[0]
  blocks=(lambda:ident in self.archive.read,lambda:ident in self.trolley.units,lambda:ident in self.named_open(),
lambda:self._smu_own("refs"),lambda:self.needs.reading.get("references"))
  if any(block()for block in blocks):
   return None
  prod=self.product_index(ident)
  return None if prod is None and self.needs.items()else(ident,prod)
 def value_guard(self,tool_name,sent):
  try:
   target=self._smu_guard_target(tool_name,sent)
   if target is None:
    return""
   ident,prod=target
   stamp=(prod,self.needs.version)
   told=self._smu_store("vguard_told",set)
   if stamp in told:
    return""
   valued=self.valued_ids(prod)
   if ident in valued or not valued:
    return""
   pick=self.walk_pick(prod)
   own=self._smu_price_of(ident)
   prices=([pick[1]]if pick else[])+([]if own is None else[own])
   cheaper=self.cheap_unread(valued,prod,min(prices))[:6]if prices else[]
   if not cheaper:
    return""
   told.add(stamp)
   listed=", ".join("%s (%s)"%(i,format(c,"g"))for c,i in cheaper)
   below=" (priced below it and the pick)"if pick else" (priced below it)"
   return("Not run (no step used). Do first: open these cheaper unread entries%s, cheapest first: %s; then resend "
"this call if still wanted. Why: cheapest wanted, and text queries here need every word, so entries "
"found by value-led queries carry those words in their records - %s was not one of them."
%(below,listed,ident))
  except Exception as err:
   self._smu_trouble("value_guard_error",err)
   return""
 def title_line(self,prod,rec,template):
  avoid=set(LINK_WORDS)
  for shelf in self.walk_home(prod):
   avoid.update(re.findall(r"[a-z]+",shelf))
  for req in self.needs.props_left():
   avoid.update(re.findall(r"[a-z0-9]+",req.lower()))
  wanted=self.needs.items()
  if not wanted:
   wanted=[{"product":str(self.needs.reading.get("product")or"")}]
  for item in wanted:
   kind_text=str(item.get("product")or"")
   avoid.update(re.findall(r"[a-z]+",kind_text.lower()))
  stems=set(map(self.stem,avoid))
  def free(word):
   return not(word in avoid or self.stem(word)in stems)
  title=self.title_of(rec).lower()
  words=re.findall(r"[a-z0-9]+",title)
  for at,word in enumerate(words):
   if not(re.fullmatch(r"[a-z]{3,}",word)and free(word)):
    continue
   follow=(words[at+1:]or[""])[0]
   text=word+" "+follow if follow and free(follow)else word
   if self.ran_query(text):
    return None
   ready=dict(template[0])
   ready[template[1]]=text
   return ready
  return None
 def _smu_rearm_due(self,aimed,gone,opens):
  least=FIRST_ORDER_BY+3+opens
  for prod,entry in list(aimed.items()):
   aim,ident=entry
   if not ident or ident not in gone:
    continue
   if self.ceil_spec()is None or self.page_arg()is None:
    continue
   pick=self.walk_pick(prod)
   if not pick:
    continue
   price=pick[1]
   if price<=aim+1e-9 or self.moves_left()<least:
    continue
   return prod,ident,price
  return None
 def rearm_walk(self):
  if not self._smu_lowest_wanted()or self._smu_wire_short()or not self.role("search"):
   return None
  aimed=self._smu_own("ceil_aimed")or{}
  due=self._smu_rearm_due(aimed,self.dead_lines()|self.refused_lines,self._smu_open_cost())
  if due is None:
   return None
  prod,dead,price=due
  aimed.pop(prod,None)
  held=self._smu_own("ceil_aims")or{}
  held.pop(prod,None)
  self._smu_trace("ceil_rearm",product=prod,dead=dead,aim=price)
  return self.run_walk([],FIRST_ORDER_BY,[prod],repair=True)
 def walk_step(self):
  try:
   for move in(self._smu_ws_decline,self._smu_ws_queued,self._smu_ws_rearm,self._smu_ws_best,self._smu_ws_rungs,
self._smu_ws_cheap):
    out=move()
    if out is not self._smu_skip:
     return out
   return False
  except Exception as err:
   self._smu_trouble("walk_error",err)
   return False
 def _smu_ws_decline(self):
  if not self._smu_own("decline_queue"):
   return self._smu_skip
  return self.run_decline_reads()
 def _smu_ws_queued(self):
  if"walk_queue"not in self.__dict__:
   return self._smu_skip
  queued=self.__dict__.pop("walk_queue")
  if not queued:
   return self._smu_skip
  plan,floor,ceil=queued[:3]
  return self.run_walk(plan,floor,ceil,twins=queued[3]if len(queued)>3 else None)
 def _smu_ws_rearm(self):
  again=self.rearm_walk()
  return self._smu_skip if again is None else again
 def _smu_ceil_due(self,prod):
  checks=(lambda:self.walk_pick(prod)or self.bound_need(prod),lambda:self.ceiling_ready(prod),
lambda:self.ceil_aim(prod)is not None)
  return all(check()for check in checks)
 def _smu_ws_cheap(self):
  if self.orders_tried or not self._smu_lowest_wanted()or self._smu_wire_short():
   return False
  only=None
  if self.unfound_product():
   only={prod for prod,_,_ in self.walk_plan(WALK_TOTAL)if self._smp_bounds_only(prod)}or None
   if only is None:
    return False
  opens=self._smu_open_cost()
  spare=self.moves_left()
  ready=list(filter(self._smu_ceil_due,self.cg_products()))
  reserve=self.order_due()+2
  if only is None and spare-opens-CEIL_CALLS>reserve:
   if ready:
    pages_read=self._smu_own("valued_by")or{}
    def order(prod):
     return(1 if self.walk_pick(prod)else 0),pages_read.get(prod,0),(prod if prod is not None else-1)
    return self.run_walk([],self.order_due()+2,[min(ready,key=order)])
   if self.ceil_seed():
    return True
  plan=self.walk_plan(WALK_TOTAL,only)
  if not plan:
   return False
  need=0
  for prod,_,n in plan:
   need+=n+(1 if self.walk_pick(prod)else 0)
  early=2 if any(self._smp_bounds_only(prod)for prod,_,_ in plan)else 0
  if spare-opens-need>reserve+early:
   return False
  fit=spare-opens-(reserve-1)
  plan=self.walk_plan(fit if fit<WALK_TOTAL else WALK_TOTAL,only)
  if not plan:
   return False
  self.walk_early=True
  return self.run_walk(plan,self.order_due())
 _smu_rx_no=re.compile(r"\bno (more|less) than\b")
 _smu_rx_up=re.compile(r"\b(?:at least|or more|or higher|or larger|minimum|more than|over|above)\b")
 _smu_rx_down=re.compile(r"\b(?:at most|up to|or less|or lower|or smaller|maximum|less than|under|below)\b")
 _smu_rx_strict=re.compile(r"\b(?:more than|less than|over|above|under|below)\b")
 def figure_rungs(self,k):
  for _,req in self._smt_own_reqs(k):
   low=self._smt_bare_value(req)
   figs=re.findall(r"(?<![\w.])(\d+(?:\.\d+)?)\s*[a-z\"']",low)
   if len(figs)!=1 or len(re.findall(r"\d+(?:\.\d+)?",low))!=1 or"between"in low:
    continue
   plain=self._smu_rx_no.sub(" ",low)
   no_word=self._smu_rx_no.search(low)
   down=(no_word.group(1)=="more")if no_word else self._smu_rx_down.search(plain)is not None
   if not(down or no_word or self._smu_rx_up.search(plain)):
    continue
   value=float(figs[0])
   if not 3<=value<100:
    return[]
   whole=int(value)
   exact=value==whole and not self._smu_rx_strict.search(plain)
   if down:
    top=whole if exact else(whole-1 if value==whole else whole)
    return[str(top-n)for n in range(RUNG_COUNT)if top-n>0]
   first=whole if exact else whole+1
   return[str(first+n)for n in range(RUNG_COUNT)]
  return[]
 def _smu_rung_base(self,k):
  home=self.walk_home(k)
  best=None
  for entry in self._smp_book().values():
   free=self.free_text(entry["args"])
   if len(free)!=1 or entry.get("id_order")is not True or re.search(r"\d",free[0])or self.cg_listing(entry["args"]):
    continue
   if not(home and self._smp_on_shelf(entry,home)):
    continue
   if best is None or entry["total"]>best["total"]:
    best=entry
  return best
 def _smu_ws_rungs(self):
  if not self._smu_lowest_wanted()or self.needs.items()or"rungs_run"in self.__dict__:
   return self._smu_skip
  orders=[entry.get("id_order")for entry in self._smp_book().values()if"id_order"in entry]
  if self.orders_tried or self._smu_wire_short()or not self.page_arg()or False in orders:
   return self._smu_skip
  if self._smt_values(None)and self.walk_options(None,deep=True):
   return self._smu_skip
  rungs=self.figure_rungs(None)
  base=self._smu_rung_base(None)if rungs else None
  if base is None or self.next_page(base)is None:
   return self._smu_skip
  room=self.moves_left()-self._smu_open_cost()-(self.order_due()+2)
  if room<2 or(self.asking and not self.trolley.units and room>RUNG_CALLS+6):
   return self._smu_skip
  slot=next(name for name in base["args"]if base["args"][name]==self.free_text(base["args"])[0])
  text=base["args"][slot].strip()
  searches=[{**base["args"],slot:"%s %s"%(text,rung)}for rung in rungs]
  searches=[one for one in searches if not self.ran_query(one[slot])]
  self.rungs_run=[one[slot]for one in searches]
  self.log.append({"kind":"rungs","base":text,"queries":self.rungs_run})
  if not searches:
   return self._smu_skip
  return self.run_walk([],self.order_due(),rungs={None:searches})
 def _smu_ws_best(self):
  if self.needs.priority()!="best"or self.needs.items():
   return self._smu_skip
  if self.orders_tried or self._smu_wire_short():
   return False
  spare,reserve=self.moves_left()-4,self.order_due()+2
  plan=self.walk_plan(WALK_TOTAL,rule="best")
  need=sum(p for _,_,p in plan)
  if not plan or(spare-need>reserve and self.__dict__.get("asking")):
   return False
  fit=spare-(reserve-1)
  plan=self.walk_plan(fit if fit<WALK_TOTAL else WALK_TOTAL,rule="best")
  paging=self.page_arg()
  if not(plan and paging):
   return False
  job={"plan":plan,"floor":self.order_due(),"key":paging[0],"book":self._smp_book(),
"walked":self.__dict__.setdefault("walked",set()),"report":[],"ran":False}
  self._smp_walk_pages(job)
  if not job["ran"]:
   return False
  shown=set()
  for _,name,_ in plan:
   shown.update((job["book"].get(name)or{}).get("row_shelf")or{})
  measure=str(self.needs.reading.get("priority_detail")or"").strip()
  marks=[w for w in re.findall(r"[a-z]{2,}",measure.lower())if w not in LINK_WORDS]
  fresh=[]
  for ident,(cost,title,*_)in self.shelf_rows().items():
   if ident in shown and ident not in self.archive.read and ident not in self.trolley.units:
    named=any(w in str(title).lower()for w in marks)
    fresh.append((not named,cost,ident,title))
  listed="; ".join("%s '%s' %g"%(ident,str(title)[:70],cost)for _,cost,ident,title in sorted(fresh)[:12])
  measure=measure or"the measure asked for"
  self._smu_extend_result("[best walk] "+"; ".join(job["report"])+". "+(
"Never-opened rows of the asked-for category, in stock and in budget: "+listed+". Ranked on "+measure
+": read the property lists ("+(self.role("details")or"the details tool")+") of the ones whose title "
"may beat the current pick on it before settling."
if listed else"No unread row of the asked-for category is left in them."),True)
  self.log.append({"kind":"best_walk","plan":[[n,pages]for n,_,pages in plan],"rows":len(fresh)})
  return True
 def _smu_pick_arg(self,props,label,skip):
  want=label.strip().lower()
  for name,spec in props.items():
   if not isinstance(spec,dict)or name in skip:
    continue
   if not self._smu_shelf_names.search(str(name).lower()):
    continue
   choices=spec.get("enum")
   if not isinstance(choices,list):
    if spec.get("type")=="string":
     return name,label
    continue
   for choice in choices:
    if isinstance(choice,str)and choice.strip().lower()==want:
     return name,choice
  return None,None
 _smu_shelf_names=re.compile(r"categor|shelf|department|aisle")
 _smu_cap_names=re.compile(r"ceil|max|price|cap$|budget|upper|most")
 _smu_not_cap=re.compile(r"page|first|offset|skip|count|size|num")
 _smu_not_limit=re.compile(r"page|size|count|result|min|low|from|start")
 @classmethod
 def _smu_cap_arg(cls,props,slot):
  for name,spec in props.items():
   if name==slot or not isinstance(spec,dict)or spec.get("enum"):
    continue
   if spec.get("type")not in("integer","number"):
    continue
   low=str(name).lower()
   if cls._smu_cap_names.search(low)and not cls._smu_not_cap.search(low):
    return name
  return None
 @staticmethod
 def _smu_mode_pick(choices,clauses,near):
  for choice in choices:
   mine=r"\b"+re.escape(str(choice).lower())+r"\b"
   if any(re.search(mine,c)and re.search(near,c)for c in clauses):
    return choice
  return None
 def seed_args(self,label,cap):
  tool=self.role("search")
  shape=self.shapes.get(tool or"")or{}
  props=shape.get("properties")
  if not isinstance(props,dict):
   props={}
  slot=self.page_arg()
  if not(props and slot and is_number(cap)):
   return None
  about=""
  for tdef in self.toolset:
   if tdef["function"]["name"]==tool:
    about=str(tdef["function"].get("description")or"")
    break
  clauses=[c.lower()for c in re.split(r"[.;]",about)]
  shelf_key,shelf_val=self._smu_pick_arg(props,label,(self.token_key(),slot[0]))
  cap_key=self._smu_cap_arg(props,slot[0])
  if not(shelf_key and cap_key):
   return None
  whole=props[cap_key].get("type")=="integer"or float(cap).is_integer()
  call={shelf_key:shelf_val,cap_key:int(cap)if whole else cap}
  near=r"\b"+re.escape(shelf_key.lower())+r"\b"
  for name,spec in props.items():
   if name==shelf_key or not isinstance(spec,dict):
    continue
   choices=spec.get("enum")
   if not isinstance(choices,list)or len(choices)<2 or len(choices)>4:
    continue
   hit=self._smu_mode_pick(choices,clauses,near)
   if hit is None:
    return None
   call[name]=hit
  if any(need not in call for need in shape.get("required")or[]):
   return None
  fixed,faults=ArgCheck.execute(call,shape)
  if faults or not isinstance(fixed,dict):
   return None
  return fixed
 def ceil_seed(self):
  try:
   if self.ceil_spec()is not None or self._smu_own("ceil_mode")or not self._smu_lowest_wanted():
    return False
   seeded=self._smu_store("ceil_seeded",set)
   slot=self.page_arg()
   skip=slot[0]if slot else None
   for prod in self.cg_products():
    if prod in seeded or(prod is None and self.needs.items()):
     continue
    pick,label=self.walk_pick(prod),self.raw_label(self.walk_home(prod))
    if not(pick and label):
     continue
    seeded.add(prod)
    call=self.seed_args(label,self.below(pick[1]))
    if not call:
     return False
    sig,ckey=self._smu_seed_marks(call,label,skip)
    bad,seen=self._smu_own("ceil_bad",set()),self._smu_own("ceil_seen")or{}
    if sig in bad or ckey in seen:
     continue
    self._smu_store("ceil_seed_keys",set).add(ckey)
    self.walk_call(call,"Testing the store's price cap: category '%s' below the pick (%s) - cheapest wanted."
%(label,format(pick[1],"g")))
    self._smu_trace("ceil_seed",product=prod,args=call,learned=self.ceil_spec()is not None)
    if self._smu_last_result()is not None and self.ceil_spec()is None:
     self._smu_extend_result(self.status_line())
    return True
   return False
  except Exception as err:
   self._smu_trouble("ceil_seed_error",err)
   return False
 def _smu_seed_marks(self,call,label,skip):
  mode=dict(self.search_keys(call)[1])
  kept=[]
  for m,v in mode.items():
   if isinstance(call.get(m),str)and call[m]==label:
    continue
   kept.append((m,v))
  cap=""
  for name in call:
   if name!=skip and is_number(call[name]):
    cap=name
    break
  plain={name:val for name,val in call.items()if skip is None or name!=skip}
  return json.dumps([sorted(kept),cap],default=str),self._smu_call_key(plain)
 def unfound_product(self):
  if not self.needs.items():
   return False
  try:
   opened={shelf.strip().lower()for shelf in self.archive.shelves_read()}
   if not all(home and home&opened for home in self.home_shelves()):
    return True
   tag=str(self.needs.version)
   live=[mark for mark in self.passed if mark.split("|")[-1]==tag]
   return self.unverified()if live else False
  except Exception:
   return False
 def _smu_aim_at(self,aims,prod,ids):
  prices=[]
  for ident in ids:
   if ident and self.product_index(ident)==prod:
    cost=self._smu_price_of(ident)
    if cost is not None:
     prices.append(cost)
  if not prices:
   return
  if is_number(aims.get(prod)):
   prices.append(float(aims[prod]))
  aims[prod]=min(prices)
 def walk_hold(self,sent):
  try:
   if not self._smu_lowest_wanted()or self._smu_own("walk_queue")or self.wire.spare()<CLOSING+60:
    return""
   if self.refusal_n and not self.moves_left()>15:
    return""
   if sent is None:
    ids=list(self.trolley.units)
   else:
    ids=[i for i,n in self.trolley.lines(sent)if n is None or n>0]
   only=set(self.product_index(i)for i in ids if i)
   floor=4 if self.orders_tried!=0 else FIRST_ORDER_BY+1
   left=self.moves_left()
   half=(left-1-floor)//2
   room=half-self._smu_open_cost()
   ceil=[]
   aims=self._smu_store("ceil_aims",dict)
   for prod in sorted(only,key=lambda p:-1 if p is None else p):
    if room<2 or not self.ceiling_ready(prod):
     continue
    ceil.append(prod)
    room-=CEIL_CALLS
    self._smu_aim_at(aims,prod,ids)
   twins={}
   for prod in sorted(only,key=lambda p:-1 if p is None else p):
    search=self.twin_query(prod)if room>0 else None
    if search:
     twins[prod]=search
     room-=1
   plan=self.walk_plan(room if room<WALK_TOTAL else WALK_TOTAL,only)
   if not(plan or ceil or twins):
    return""
   stop=self.moves_left()-1-half
   self.walk_queue=(plan,stop if stop>floor else floor,ceil,twins)
   book=self._smu_own("page_book",{})
   queries=["'"+" ".join(self.free_text(book[search_key]["args"]))+"'"for _,search_key,_ in plan if search_key in book]
   jobs=[]
   if twins:
    jobs.append("a search in the spelling this store also uses for a required value (%s)"
%", ".join("'"+" ".join(self.free_text(s))+"'"for s in twins.values()))
   if queries:
    jobs.append("pages still unread in %s"%", ".join(queries))
   if ceil:
    jobs.append("listing that category under the pick's price via the price cap")
   order="ranked by match"if self.ranked_store()else"in id order"
   return("Not run yet (no step used). The code now does for you: %s; then it opens the cheapest unread entries "
"below the pick. Afterwards resend this call, or one naming a cheaper match. Why: cheapest wanted, and "
"results here come %s, not price order, so pages not yet read may hold a cheaper match."
%("; ".join(jobs),order))
  except Exception as err:
   self._smu_trouble("walk_hold_error",err)
   return""
 @staticmethod
 def _smu_rows_of(piece):
  def table(val):
   return isinstance(val,list)and not any(not isinstance(r,dict)for r in val)
  tables=list(filter(table,piece.values()))
  if not tables:
   return None
  filled=[t for t in tables if t]
  return filled[0]if filled else tables[0]
 @staticmethod
 def _smu_add_new(found,extra):
  found+=[i for i in extra if i not in found]
 def search_note(self,sent,obs):
  if not isinstance(obs,dict):
   return""
  rows=self._smu_rows_of(obs)
  if rows is None:
   for part in obs.values():
    if isinstance(part,dict)and self._smu_rows_of(part)is not None:
     obs,rows=part,self._smu_rows_of(part)or[]
     break
   else:
    return""
  shown=len(rows)
  totals=[n for n in obs.values()if type(n)is int and n>shown]
  note=dict(args=sent,obs=obs,rows=rows,totals=totals,count=max(totals)if totals else shown,remarks=[])
  for stage in(self._smu_sn_book,self._smu_sn_lead,self._smu_sn_query,self._smu_sn_paging,self._smu_sn_limit):
   if not stage(note):
    break
  return" ".join(note["remarks"])
 def _smu_sn_book(self,note):
  call,rows,count,obs=note["args"],note["rows"],note["count"],note["obs"]
  self.book_page(call,rows,count,obs)
  try:
   self.learn_twins(call,rows)
   self.ceil_learn(call,rows,count,obs)
  except Exception as err:
   self._smu_trouble("ceil_learn_error",err)
  weight=self.value_score(call)if rows else 0
  if weight:
   best=self._smu_store("row_score",dict)
   for row in rows:
    if isinstance(row,dict):
     self._smu_raise(best,self.row_mark(row),weight)
  return True
 def _smu_sn_lead(self,note):
  call,obs,rows=note["args"],note["obs"],note["rows"]
  note["kind"]=self.word_rule(call,obs,rows)
  lead=self.lead_rule(call,obs,rows)
  if not lead:
   return True
  texts,mode,_,_=self.search_keys(call)
  field,lead_mode=self.lead_ctx[:2]
  words=()
  if mode==lead_mode:
   for name,said in texts:
    if name==field:
     words=said
     break
  counted=""
  if len(words)>lead:
   counted=" (this one: '%s')"%" ".join(words[:lead])
  note["remarks"].append("Query length: only the opening %s word(s) count for this store's text query (%s), the rest "
"is ignored%s. Put the %s most distinctive words up front (required value, then item kind) "
"and give each other value its own brief query."%(lead,field,counted,lead))
  return True
 def _smu_sn_query(self,note):
  call=note["args"]
  texts,mode,first,paging_keys=self.search_keys(call)
  note["paging"]=paging_keys
  modes=dict(mode)
  raw=[val for name,val in call.items()if isinstance(val,str)and name not in modes and name not in paging_keys]
  if first and raw:
   self._smu_sn_words(note,raw)
   self._smu_sn_repeat(note,raw,mode,modes)
   self._smu_sn_back(note,self._smu_sn_hits(note,raw,texts))
  return bool(note["rows"])
 def _smu_sn_words(self,note,raw):
  say=note["remarks"].append
  bits=re.findall(r"[a-z0-9]+"," ".join(raw).lower())
  if bits and max(map(len,bits))==1 and not note["rows"]and not self._smu_own("short_told"):
   self.short_told=True
   say("Empty result. One-letter or one-digit tokens get dropped by this store, hence 0.4 or 5 x 7 adds nothing to "
"a query except its other words. Spell the value as a word, or pair its unit word with the item kind, then "
"page through it fully.")
  prop,probe=self.value_probe(raw)
  named=self._smu_own("name_told",0)
  room=lambda:self.moves_left()-3>self.order_due()
  if probe and named<2 and room():
   self.name_told=named+1
   if named:
    why="Property name (%s) leading the query again."%prop
   else:
    why=("Offers whose %s field holds only the value never show up for this query: it opens with that "
"property's NAME and lacks a required value, while matching runs on values, titles and categories "
"(a field name counts only where a value or title repeats it)."%prop)
   say(why+" Suggested query: '%s' (value ahead of item kind; unreadable figures vanish and their unit words "
"remain). Page through it fully."%probe)
  elif not probe and self._smu_own("trap_told",0)<2 and room():
   trap,_=self.word_trap(raw)
   if trap:
    self.trap_told=1+self._smu_own("trap_told",0)
    say(trap)
 def _smu_sn_repeat(self,note,raw,mode,modes):
  skip=note["paging"]
  marks=tuple(map(self.row_mark,note["rows"]))
  fixed=sorted((name,str(val))for name,val in note["args"].items()
if name not in skip and not(isinstance(val,str)and name not in modes))
  sig=(mode,tuple(fixed),note["count"],marks)
  seen=self._smu_store("page_sigs",dict)
  query=" ".join(raw)
  if len(set(marks))>=3 and sig in seen and seen[sig]!=query:
   note["remarks"].append("Identical total and entries to '%s' earlier - to this store both spellings are one "
"query. Read its next page instead, or change the opening words."%seen[sig])
  seen.setdefault(sig,query)
 def _smu_sn_hits(self,note,raw,texts):
  kind,rows=note["kind"],note["rows"]
  full=rows and not note["totals"]
  told=bool(full and(kind=="all"or self.lead_words))
  if not(told or(full and kind!="any")):
   return[]
  held=None if told else list(self._smu_own("ref_hits")or[])
  self.ref_hits=[]
  hits=self.row_hits(rows,raw)
  weight=self.value_score(note["args"])
  ranks=self._smu_store("hit_score",dict)
  for ident in hits:
   self._smu_raise(ranks,ident,weight)
  self._smu_add_new(self._smu_store("all_hits",list),hits)
  if held is not None:
   self.ref_hits=held
  if told and hits and len(texts)and len(rows)<=12:
   note["remarks"].append("Open these in-budget entries before querying on: %s. Why: all query words are "
"mandatory and nothing is hidden here, hence each entry contains them somewhere in its "
"record, frequently outside its title."%", ".join(hits[:6]))
  pointers=[]
  if told:
   pointers=[i for i in self.__dict__.pop("ref_hits",[])if i not in hits][:6]
  if pointers:
   note["remarks"].append("Reference entries (read for a value, never bought): %s. Open each regardless of price, "
"stock or category and act on its notes - the request fixes a value via another offer, "
"and these come from the shop or brand this query names."%", ".join(pointers))
  return hits
 def _smu_sn_back(self,note,hits):
  if self._smu_own("hits_back")or not(note["kind"]=="all"or self.lead_words):
   return
  self.hits_back=True
  skip=note["paging"]
  here=self._smu_call_key({name:val for name,val in note["args"].items()if name not in skip})
  back=[]
  for search_key,walk in self._smu_own("page_book",{}).items():
   if search_key==here or not walk.get("every"):
    continue
   weight=self.value_score(walk["args"])
   if not weight:
    continue
   fresh=[i for i in self.row_hits(walk["every"])if i not in back and i not in hits]
   ranks=self._smu_store("hit_score",dict)
   for ident in fresh:
    self._smu_raise(ranks,ident,weight)
   back.extend(fresh)
  self._smu_add_new(self._smu_store("all_hits",list),back)
  if back:
   note["remarks"].append("Open the unread in-budget ones before querying on: %s. Why: mandatory words mean earlier "
"complete queries could only have listed entries containing all their words, often "
"outside the titles."%", ".join(back[:6]))
 _smu_widen={
"any":("Loose-match store: one shared word suffices and offers sharing more words come first, so additional "
"words enlarge the result. Alone, a required value's distinctive word or figure yields only offers "
"containing it (frequently a short enough result to page completely and compare prices); kind-plus-values "
"queries put the nearest offers on top; category mode, or a working price cap, reaches the remainder."),
"all":"Mandatory-words store: each extra exact word cuts the result, and so do category plus a lower price cap.",
}
 _smu_widen_other=("cut it down: when words are mandatory, append exact ones; when one shared word suffices (a longer "
"query returning more offers shows this), include the required values so matching offers come "
"first; a working price cap is useful too.")
 def _smu_sn_paging(self,note):
  if note["totals"]:
   how=self._smu_widen.get(note["kind"],self._smu_widen_other)
   note["remarks"].append("Partial view: %d matches, %d shown here, and only a few pages are readable - relevant "
"offers may be unseen. First page on through this exact query (as far as possible); if "
"the total outruns those pages, "%(max(note["totals"]),len(note["rows"]))+how)
   ahead=self._smu_next_hint(note["args"],note["obs"])
   if ahead:
    note["remarks"].append(ahead)
  return True
 def _smu_next_hint(self,sent,obs):
  paging=self.page_arg()
  try:
   follow=self.code_next(sent,obs)if paging else None
  except Exception:
   return""
  if not follow or paging[0]not in follow:
   return""
  return"To read its next page, send the same call with %s = %s."%(paging[0],json.dumps(follow[paging[0]]))
 def _smu_sn_limit(self,note):
  skip=self.search_keys(note["args"])[3]
  caps=[val for name,val in note["args"].items()
if is_number(val)and val>1 and name not in skip
and not self._smu_not_limit.search(str(name).lower())]
  if not caps:
   return True
  top=max(caps)
  def breaks(row):
   return any(is_number(val)and val>top and"price"in str(name).lower()for name,val in row.items())
  over=sum(1 for row in note["rows"]if breaks(row))
  if over:
   note["remarks"].append("Cap not applied: %d entr(ies) above %s although this call set that cap - this search "
"mode drops that filter, maybe the category one as well. Switch to a mode whose filters "
"work, or vet each entry by hand."%(over,top))
  return True
 def shopper_lines(self,call_name,obs,refused):
  if call_name==self.role("message"):
   said=self._smv_reply_text(obs)
  else:
   said=self.notice_lines(obs,refused)if isinstance(obs,dict)else[]
  return list(dict.fromkeys(said))
 @staticmethod
 def _smv_reply_text(obs):
  if isinstance(obs,str):
   return[obs]
  bag=[]
  if isinstance(obs,dict)and"error"not in obs:
   walk_dicts(obs,lambda layer:long_strings(list(filter(lambda x:isinstance(x,(str,list)),layer.values())),bag),
ceiling=4)
  return bag
 @staticmethod
 def notice_lines(obs,refused=False):
  if not isinstance(obs,dict):
   return[]
  bag=[]
  def tagged(layer):
   return any(isinstance(x,str)and"id"in str(name).lower()and re.fullmatch(ID_LIKE,x)for name,x in layer.items())
  def gather(layer):
   if len(layer)>6 and layer is not obs:
    return
   small=len(layer)<=3
   for x in layer.values():
    if isinstance(x,list):
     if x and all(isinstance(s,str)for s in x):
      long_strings(x,bag)
     continue
    if not isinstance(x,str)or len(x.split())<4:
     continue
    if refused or(small and len(x)<=600 and not tagged(layer)):
     bag.append(x)
  walk_dicts(obs,gather,ceiling=4)
  return list(dict.fromkeys(bag))
 _smv_NOW="Not run (no step used)"
 _smv_YET="Not run yet (no step used). "
 def _smv_have(self,name,default=None):
  return self.__dict__.get(name,default)
 def _smv_tag(self,ident):
  return"%s|%s"%(ident,self.needs.version)
 def _smv_room(self,steps,pad=None):
  return self.moves_left()>steps and(pad is None or self.wire.spare()>CLOSING+pad)
 @staticmethod
 def _smv_list(seq):
  return", ".join(seq)
 _smv_screen_gates=("_smv_gate_known","_smv_gate_offered","_smv_gate_decode","_smv_gate_shape","_smv_gate_list","_smv_gate_handles","_smv_gate_alone","_smv_gate_messages","_smv_gate_view","_smv_gate_repeat","_smv_gate_lookup","_smv_gate_guards","_smv_gate_cart")
 def screen(self,proposal,permitted,scripted,batch,queued_messages):
  spec=proposal.get("function")or{}
  st=dict(name=spec.get("name"),raw=spec.get("arguments"),args={},sig="",menu=permitted,own=not scripted,
many=batch,queued=queued_messages)
  for gate in self._smv_screen_gates:
   verdict=getattr(self,gate)(st)
   if verdict:
    return verdict,st["name"],st["args"]
  return None,st["name"],st["args"]
 def _smv_gate_known(self,st):
  if st["name"]in self.shapes:
   return""
  return"%s: no tool here is called '%s'. Pick a tool from this list: %s."%(
self._smv_NOW,st["name"],self._smv_list(sorted(st["menu"])))
 def _smv_gate_offered(self,st):
  if not st["own"]or st["name"]in st["menu"]:
   return""
  return"%s: the steps still open are too few for that tool; for now the only usable tools are: %s."%(
self._smv_NOW,self._smv_list(sorted(st["menu"])))
 def _smv_gate_decode(self,st):
  raw=st["raw"]
  if isinstance(raw,dict):
   parsed=raw
  elif(raw or"").strip():
   parsed=json_object(raw)
  else:
   parsed={}
  if parsed is None:
   return self._smv_NOW+": unreadable arguments; send one well-formed JSON object."
  st["args"]=parsed
  return""
 def _smv_gate_shape(self,st):
  st["args"],faults=ArgCheck.execute(st["args"],self.shapes[st["name"]])
  if not faults:
   return""
  return self._smv_NOW+": the arguments break the tool's schema: "+"; ".join(faults[:6])
 def _smv_gate_list(self,st):
  spec=self.batch_tool()
  ids=st["args"].get(spec[1])if spec and st["own"]and st["name"]==spec[0]else None
  if isinstance(ids,list)and len(ids)>spec[2]:
   self.ref_index()
   seen=self.__dict__.get("ref_seen_at")or{}
   kept=set(sorted(ids,key=lambda pid:-seen.get(pid,-1))[:spec[2]])
   st["args"][spec[1]]=[pid for pid in ids if pid in kept]
   self.log.append({"kind":"list_cut","asked":len(ids),"cap":spec[2]})
  return""
 def _smv_gate_handles(self,st):
  if not st["own"]:
   return""
  return self._smv_page_token(st)or self._smv_title_ids(st)
 def _smv_page_token(self,st):
  paging=self.page_arg()
  if st["name"]!=self.role("search")or not paging or paging[1]!="token":
   return""
  key,args=paging[0],st["args"]
  token=args.get(key)
  if not isinstance(token,str)or not token.strip():
   return""
  handed=set()
  for walk in self._smp_book().values():
   for spot,nxt in(walk.get("tokens")or{}).items():
    handed.update(t for t in(spot,nxt)if isinstance(t,str)and t)
  if token.strip()in handed:
   return""
  rest=self._smo_without(args,key)
  walk=self._smp_book().get(self._smo_sig(rest))
  if walk is None:
   st["args"]=rest
   self.log.append({"kind":"token_dropped","token":token[:40]})
   return""
  nxt=self.next_page(walk)
  where=("its next page comes with %s='%s'"%(key,nxt))if nxt is not None else"it has no unread page left"
  return("%s: '%s' is no page token this store handed out; tokens are copied from a result's own next-page field. "
"That search already ran: %s."%(self._smv_NOW,token[:40],where))
 def _smv_title_ids(self,st):
  if st["name"]not in{self.role("cart"),self.role("details"),self.role("view")}|{(self.batch_tool()or("",))[0]}:
   return""
  known=self.archive.ids
  texts=[]
  self._smv_swap_strings(st["args"],{},texts)
  strays={t for t in texts if t not in known and len(t.strip())>8}
  if not strays:
   return""
  owners={}
  for row in self._smt_rows():
   name,mark=self.title_of(row).strip().lower(),self.row_mark(row)
   if name and mark in known:
    owners.setdefault(name,set()).add(mark)
  swap,clash={},[]
  for text in strays:
   low=text.strip().lower()
   ids=owners.get(low,set())
   if not ids and len(low)>=20:
    ids=set().union(*(marks for name,marks in owners.items()if name.startswith(low)))
   if len(ids)==1:
    swap[text]=next(iter(ids))
   elif ids:
    clash.append("'%s' is the title of %s"%(text[:60],", ".join(sorted(ids))))
  if clash:
   return"%s: this tool takes listing ids, not titles: %s. Resend with the id meant."%(self._smv_NOW,"; ".join(clash))
  if swap:
   st["args"]=self._smv_swap_strings(st["args"],swap)
   self.log.append({"kind":"title_to_id","swapped":sorted(swap.values())})
  return""
 def _smv_swap_strings(self,node,swap,seen=None):
  if isinstance(node,dict):
   return{k:self._smv_swap_strings(v,swap,seen)for k,v in node.items()}
  if isinstance(node,list):
   return[self._smv_swap_strings(v,swap,seen)for v in node]
  if isinstance(node,str)and seen is not None:
   seen.append(node)
  return swap.get(node,node)if isinstance(node,str)else node
 def _smv_gate_alone(self,st):
  if st["name"]not in self.finishers()or st["many"]<=1:
   return""
  return(self._smv_NOW+": a finish (order or decline) goes out as the only call of its reply, once the results of "
"your other calls are back.")
 def _smv_gate_messages(self,st):
  if st["name"]!=self.role("message"):
   return""
  spent=st["queued"]+self.msgs_sent
  cap=self.msg_cap
  if cap is not None and spent>=cap:
   return self._smv_NOW+": zero messages remain; the shopper gets nothing further."
  if st["own"]and not self.trolley.units and self.msg_reserve()and spent>=self.ask_room():
   return(self._smv_NOW+": message allowance on hold. Remaining sends go to the basket summary owed after "
"something is carted, and to any closing step the instructions require. Cart a choice first; afterwards "
"you may write again.")
  return""
 def _smv_gate_view(self,st):
  tool=st["name"]
  if not st["own"]or tool!=self.role("cart_view")or tool==self.role("cart")or self._smv_have("batch_cart"):
   return""
  cart=self.trolley
  if not cart.units or self._smv_have("cart_shown",-1)!=self._smv_have("cart_ver",-2)or self._smv_have("since_cart",1):
   return""
  seen=self.__dict__.setdefault("view_told",set())
  if self.cart_shown in seen:
   return""
  seen.add(self.cart_shown)
  basket=cart.summary()
  return"%s: a cart view would add nothing; the latest cart reply already lists every line. %s For the current ""stock of one line, look up that listing's record using %s."%(
self._smv_NOW,basket,self.stock_tool()or"a tool that reports stock")
 def _smv_gate_repeat(self,st):
  tool=st["name"]
  st["sig"]=tool+as_text(st["args"])
  exempt=tool in self.finishers()or tool in(self.role("message"),self.role("cart"))
  if exempt or st["sig"]not in self.last_calls:
   return""
  return(self._smv_NOW+": an identical call just ran and nothing has moved since; scroll up for what it returned. "
"Vary the arguments or pick a different step.")
 def _smv_gate_lookup(self,st):
  tool,args=st["name"],st["args"]
  if not isinstance(args,dict)or tool in self.finishers()or tool in(self.role("message"),self.role("cart")):
   return""
  texts={str(k):v.strip()for k,v in args.items()if isinstance(v,str)and v.strip()}
  if tool==self.role("search")and any(v in self.archive.ids for v in texts.values()):
   return("%s: search text is matched against listing words, so a listing id returns nothing. Open that listing "
"with %s."%(self._smv_NOW,self.role("details")or"the record tool"))
  labels=self.shelf_names()
  for name,value in texts.items():
   if not labels or not re.search(r"categor|shelf|department|aisle",name.lower()):
    continue
   if value.lower().rstrip("s")not in{low.rstrip("s")for low in labels}:
    return("%s: %s takes a shelf name spelled as the listings show it; free words there match nothing. Shelves "
"seen so far: %s. Put descriptive words in the search text instead."
%(self._smv_NOW,name,", ".join(sorted(labels.values())[:8])))
  return""
 def shelf_names(self):
  found={}
  ids=self.archive.ids
  def visit(d):
   if not any("id"in str(k).lower()and isinstance(v,str)and v in ids for k,v in d.items()):
    return
   for k,v in d.items():
    low=str(k).lower()
    if isinstance(v,str)and 0<len(v.strip())<=60 and"id"not in low and re.search(r"categor|shelf|department|aisle",low):
     found.setdefault(v.strip().lower(),v.strip())
     return
  for rec in self.archive.items:
   walk_dicts(rec,visit)
  return found
 def _smv_gate_guards(self,st):
  if not st["own"]:
   return""
  tool,args=st["name"],st["args"]
  return self.late_hold(tool,args)or self.value_guard(tool,args)
 def _smv_gate_cart(self,st):
  if st["name"]!=self.role("cart"):
   return""
  return self.screen_cart(st["args"],st["sig"])
 _smv_cart_steps=("_smv_cart_same","_smv_cart_stated","_smv_cart_pointer","_smv_cart_kept","_smv_cart_takes","_smv_cart_double","_smv_cart_walk","_smv_cart_others","_smv_cart_own","_smv_cart_fit","_smv_cart_ranked","_smv_cart_undercut","_smv_cart_units")
 def screen_cart(self,call_args,sig):
  st=dict(args=call_args,sig=sig,line=self.trolley.aim(call_args),point="")
  for step in self._smv_cart_steps:
   verdict=getattr(self,step)(st)
   if verdict is not None:
    return verdict
  return""
 def _smv_cart_same(self,st):
  cart,line=self.trolley,st["line"]
  tool=self.role("cart")
  if not line or not tool or tool+as_text(st["args"])!=self.__dict__.get("last_cart_sig"):
   return None
  if not cart.is_put(st["args"])or st["sig"]in self.warned:
   return None
  asked=cart.count_arg(st["args"])
  held=cart.held(line)
  if cart.adds and held>0:
   return None
  self.warned.add(st["sig"])
  key=cart.count_key()or"the count argument"
  return("%s: this exact cart call just ran, and the cart holds %s x%s; sending it again changes nothing. %s is the "
"number of units held on that line, so to hold N units send N once. Resend the identical call only if you are sure."
%(self._smv_NOW,line,_sme_qty(held),key)) if asked is not None else None
 def _smv_cart_stated(self,st):
  cart=self.trolley
  if self.ask_limit()-self.msgs_sent>0 or not self._smw_told_current()or self.dead_lines()&set(cart.units):
   return None
  if any(self._smv_tag(line)not in self.passed for line in cart.units):
   return None
  cap,grand=self.needs.spend_cap,0.0
  for line_id,count in cart.units.items():
   each=self.archive.price(line_id)
   grand+=float(each)*float(count)if is_number(each)else 0.0
  if is_number(cap)and grand>float(cap)+1e-9:
   return None
  times=self.__dict__.get("stated_holds",0)
  if times>=3:
   return None
  self.stated_holds=times+1
  self.log.append({"kind":"stated_hold","lines":[str(line)for line in cart.units]})
  return("%s: the shopper has already been told this exact cart, and no message is left to tell a changed one, so "
"an edit now would send an order the shopper never saw. Nothing since forces a change (no notice, refusal, "
"sold-out line or total over the budget). Place the order with the cart as it stands."%self._smv_NOW)
 def _smv_cart_pointer(self,st):
  line=st["line"]
  st["point"]=self.ref_pointer_note(line)if line and self.trolley.is_put(st["args"])else""
  tally=self.__dict__.setdefault("pointer_told",{})
  times=tally.get(line,0)
  if not st["point"]or times>=2:
   return None
  tally[line]=times+1
  return"%s: %s. Pick a different listing."%(self._smv_NOW,st["point"])
 def _smv_undercut_in_batch(self,line,price):
  for other in self._smv_have("batch_puts",[]):
   if other==line:
    continue
   if is_number(price)and is_number(self.archive.price(other))and self.archive.price(other)<price:
    return True
  return False
 def _smv_cart_kept(self,st):
  line,args,book=st["line"],st["args"],self.archive
  stamp="%s#%s"%(st["sig"],self._smv_have("puts",{}).get(line,0))
  price=book.price(line)if line else None
  beaten=self._smv_undercut_in_batch(line,price)
  replacing=self.needs.priority()=="cheapest"and(beaten or bool(self.cheaper_fit(line)))
  priced=self._smv_have("pass_price",{})
  repriced=line in priced and self.__dict__["pass_price"][line]!=price
  if not line or self.trolley.is_put(args)or self._smv_tag(line)not in self.passed or replacing or repriced:
   return None
  if self.refusals_seen.get(line,self.refusal_n)!=self.refusal_n or stamp in self.warned:
   return None
  self.warned.add(stamp)
  lowest=""
  if self.needs.priority()=="cheapest"and not self.cheaper_fit(line):
   lowest=" Among verified fits, none is cheaper."
  return("%s: removal paused. Independent check verdict on %s: passes every condition now in force (properties "
"judged by the value rules, a dedicated value field of this store counting only by its wording; item kind "
"taken from the category label alone, whatever object title or text mention).%s Nothing new since. Repeat "
"this exact call only for: a change or refusal that excludes it, a field clearly breaking a condition, or a "
"cheaper listing replacing it."%(self._smv_NOW,line,lowest))
 def _smv_cart_takes(self,st):
  return None if self.trolley.is_put(st["args"])and st["line"]else""
 def _smv_cart_double(self,st):
  line,sig,cart=st["line"],st["sig"],self.trolley
  if cart.held(line)<=0 or cart.adds is False or sig in self.warned:
   return None
  self.warned.add(sig)
  return("%s: the cart has %s x%d, and with this tool adding units onto a held line the call would increase that "
"count. Ordering needs no cart change; adjusting units means adding or removing just the gap. Resend the "
"identical call if more units are truly intended."%(self._smv_NOW,line,int(cart.held(line))))
 def _smv_cart_walk(self,st):
  if st["point"]:
   return None
  return self.walk_hold(st["args"])or None
 def _smv_item_gate(self,line):
  tag=self._smv_tag(line)
  if not self._smv_room(3)or tag in self.checked_adds:
   return None
  self.checked_adds.add(tag)
  ruling=self.check_item(line,"ADD TO THE CART")
  if not ruling:
   return None
  return"%s. %s Resend this exact call only when you are certain of the fit."%(self._smv_NOW,ruling)
 def _smv_live_lines(self,args):
  return[ident for ident,n in self.trolley.lines(args)if n is None or not n<=0]
 def _smv_cart_others(self,st):
  for other,n in self.trolley.lines(st["args"])[1:]:
   verdict=self._smv_item_gate(other)if n is None or n>0 else None
   if verdict:
    return verdict
  return None
 def _smv_cart_own(self,st):
  return self._smv_item_gate(st["line"])
 def _smv_cart_fit(self,st):
  rows=self.needs.items()
  links=self._smv_links(rows)
  if not links or not self._smv_room(5,0):
   return None
  first={}
  for line in[*self._smv_live_lines(st["args"]),*self.trolley.units]:
   slot=self.product_index(line)
   if slot is not None:
    first.setdefault(slot,line)
  told=self.__dict__.setdefault("code_fit_told",set())
  found=[]
  for k,row,j in links:
   if k not in first or j not in first or len(told)>=5:
    continue
   base_line,dep_line=first[j],first[k]
   base_page,dep_page=self._smv_page(base_line),self._smv_page(dep_line)
   if(base_line,dep_line)in told or self._smv_coded(base_page,dep_page)is not False:
    continue
   told.add((base_line,dep_line))
   stems={self._smv_stem(head)for head in self._smv_codes(dep_page)[1]}
   codes=[code for code in sorted(self._smv_codes(base_page)[0])if self._smv_stem(code)in stems]
   self.log.append({"kind":"code_fit_hold","anchor":base_line,"dependent":dep_line,"codes":codes})
   found.append((k,row,j,base_line,dep_line,codes))
  if not found:
   return None
  queue=self.__dict__.setdefault("fit_queue",[])
  said=[]
  for k,row,j,base_line,dep_line,codes in found:
   if codes and not any(k==q[0]for q in queue):
    queue.append((k,codes[0]))
   said.append("line %s (the %s) is meant for %s (the %s), whose page states %s, while %s carries a tag of "
"that series with another number"%(dep_line,row["product"],base_line,rows[j]["product"],
self._smv_list(codes),dep_line))
  return("%s: this store writes fits as codes, and %s; so %s not fit. The code now queries %s with the code as "
"the search text and opens tagged hits. Cart, for each, a hit whose tag starts with that code and that "
"meets the other conditions; or switch to an anchor whose stated code some tag of each carries. Resend "
"this exact call only if no such listing exists."%(
self._smv_NOW,"; ".join(said),"those lines do"if len(found)>1 else"it does",
self._smv_list([row["product"]+" listings"for _k,row,*_rest in found])))
 def _smv_cart_ranked(self,st):
  if not self._smv_room(3):
   return None
  for ident in self._smv_live_lines(st["args"]):
   beaten=self.rank_check(ident)
   if not beaten:
    continue
   rival=self.rank_verify(ident)
   text=beaten.replace("before ordering","prior to adding it").replace("order the current pick","resend the identical call")
   if rival:
    text+=(" Lower-priced verified alternative: %s (independent item check: passes current conditions; price "
"below %s) - cart %s in its place.%s"%(rival,ident,rival,self.shelf_met_note(rival)))
   return self._smv_NOW+". "+text
  return None
 def shelf_met_note(self,ident):
  try:
   met=[line[2:]for line in self.field_view(ident)["lines"]if"itself names this value"in line]
  except Exception:
   return""
  return(" Met by its category alone: "+"; ".join(met[:3])+".")if met else""
 def _smv_cart_undercut(self,st):
  if self.needs.priority()!="cheapest":
   return None
  for ident in self._smv_live_lines(st["args"]):
   rival=self.cheaper_fit(ident)
   tally=self.__dict__.setdefault("cheaper_told",{})
   pair=("cart",ident,rival)
   if not rival or rival in self.trolley.units or tally.get(pair):
    continue
   tally[pair]=1
   return("%s. Lower-priced verified alternative: %s (independent item check: passes the conditions now in force; "
"price below %s). Lowest price is the priority and that check overrides your reading, so cart %s instead. "
"Repeat this exact call only if a record shows %s breaking a current condition.%s"
%(self._smv_NOW,rival,ident,rival,rival,self.shelf_met_note(rival)))
  return None
 def _smv_cart_units(self,st):
  line,sig,args,cart,needs=st["line"],st["sig"],st["args"],self.trolley,self.needs
  wanted=needs.quantity()
  asked=cart.count_arg(args)
  target=wanted if cart.adds is False else wanted-cart.held(line)
  if needs.items():
   off=[(ident,int(n))for ident,n in cart.lines(args)if n is not None and n>0 and int(n)not in needs.counts()]
   if not off or sig in self.warned:
    return""
   self.warned.add(sig)
   return("%s: per-product units should be %s (one line apiece), and this call has %s. Resend the identical call "
"if this is intended."%(self._smv_NOW,needs.counts(),off))
  if asked is None or abs(asked-target)<=1e-9 or sig in self.warned:
   return""
  self.warned.add(sig)
  return("%s: %s units of a single listing are needed, and this call has %d. Resend the identical call if this is "
"intended."%(self._smv_NOW,wanted,int(asked)))
 _smv_finish_paths={
"order":("_smv_fin_sold_out","_smv_fin_cheaper","_smv_fin_walk","_smv_fin_stale","_smv_fin_sweep","_smv_fin_depends","_smv_fin_vet","_smv_fin_misfit","_smv_fin_steps"),
"decline":("_smv_fin_told","_smv_fin_dropped","_smv_fin_early","_smv_fin_near_hold","_smv_fin_pairswap","_smv_fin_whole","_smv_fin_kept","_smv_fin_slack","_smv_fin_fill","_smv_fin_named","_smv_fin_pages","_smv_fin_reaudit","_smv_fin_audit","_smv_fin_fit_hold","_smv_fin_best","_smv_fin_unknown_ref","_smv_fin_repair","_smv_fin_meaning","_smv_fin_steps"),
}
 def screen_finish(self,cid,call_name,call_args):
  mode=self.call_kind(call_name,call_args)
  st={"mode":mode,"spread":None}
  for stage in self._smv_finish_paths.get(mode,()):
   verdict=getattr(self,stage)(st)
   if verdict:
    return verdict,call_args
  final=self.scrub_decline(call_args,call_name)if mode=="decline"else call_args
  return None,self._smv_rules_tail(call_name,final)
 def _smv_rules_tail(self,call_name,call_args):
  if not SHOW_RULES_IN_FINISH or not isinstance(call_args,dict):
   return call_args
  try:
   field=self.free_text_arg(call_name)
   if not field or not isinstance(call_args.get(field),str)or not call_args[field].strip():
    return call_args
   told=[str(entry.get("text")or"")[:60]for entry in self.needs.told][:4]
   answer=self.__dict__.get("finish_answer")
   pending=[]
   if isinstance(answer,dict):
    pending=[str(x.get("why")or"")[:40]for x in(answer.get("pending")or[])if isinstance(x,dict)][:2]
   tail=" [rules read: %s; finish check: %s]"%("; ".join(told)or"none","ready"if not pending else"pending "+"; ".join(pending))
   spec=((self.shapes.get(call_name)or{}).get("properties")or{}).get(field)or{}
   out=dict(call_args)
   out[field]=self._sml_cut(call_args[field].rstrip()+tail,spec)
   return out
  except Exception:
   return call_args
 @staticmethod
 def _smv_fault(err):
  return("%s: %s"%(type(err).__name__,err))[:300]
 def _smv_fin_sold_out(self,st):
  if not self._smv_room(2):
   return""
  cart=self.trolley
  gone=sorted(self.dead_lines()&set(cart.units))
  stamp=(tuple(gone),self._smv_have("cart_seq",0))
  tally=self.__dict__.setdefault("dead_told",{})
  if not gone:
   return""
  times=tally.get(stamp,0)
  if times<2:
   tally[stamp]=times+1
   return self._smv_YET+self.dead_note(gone)
  if times!=2 or not self.role("cart")or self.moves_left()<6 or self.wire.spare()<=CLOSING:
   return""
  tally[stamp]=3
  muted=self.wire.mute
  try:
   swapped=self.code_cart(set(gone)|self.dead_lines(),set(cart.units)-set(gone))
  except Exception as err:
   swapped=False
   self.log.append({"kind":"dead_swap_error","error":self._smv_fault(err)})
  finally:
   self.wire.mute=muted
  self.log.append({"kind":"dead_swap","lines":gone,"done":swapped,"cart":dict(cart.units)})
  if not swapped or not cart.units or set(cart.units)&self.dead_lines():
   return""
  return(self._smv_YET+self._smv_list(gone)+": no stock left, yet carted. Swap done by code - a candidate "
"replaced it, remaining lines untouched - followed by a stock read of each line: "+cart.summary()
+" Compare against the request; order if it matches.")
 def _smv_fin_cheaper(self,st):
  if not self._smv_room(3):
   return""
  line=self.chosen()
  rival=self.cheaper_fit(line)if line else""
  if not rival:
   for other in[i for i in self.trolley.units if i!=line]:
    rival=self.cheaper_fit(other)
    if rival:
     line=other
     break
  tally=self.__dict__.setdefault("cheaper_told",{})
  pair=(line,rival)
  if not rival or tally.get(pair,0)>=2:
   return""
  tally[pair]=tally.get(pair,0)+1
  return("%sLower-priced verified alternative: %s (independent item check: passes every condition; price below %s). "
"Lowest price is the priority and that check overrides your reading: take %s out, cart %s, refresh its stock "
"and price, order.%s"%(self._smv_YET,rival,line,line,rival,self.shelf_met_note(rival)))
 def _smv_fin_walk(self,st):
  if not self.trolley.units:
   return""
  return self.walk_hold(None)
 def _smv_fin_stale(self,st):
  reader=self.stock_tool()
  if not reader or not self._smv_room(len(self.trolley.units)+2,0):
   return""
  unread=self.stale_lines()
  stamp=(tuple(sorted(self.trolley.units)),self._smv_have("cart_seq",0))
  if not unread or self._smv_have("recheck_mark")==stamp:
   return""
  self.recheck_mark=stamp
  self.open_queue=unread[:4]
  self.open_recheck=True
  self.open_purpose="cart"
  self.open_tool=reader
  return("%sNo notice is needed for stock or prices to move, and these lines have gone unread since the cart last "
"changed: %s. The code now reads their stock for you through %s, the stock-showing tool. Then: with all lines "
"available and the sum inside the budget, order; otherwise replace the failing line."
%(self._smv_YET,self._smv_list(unread[:4]),reader))
 def _smv_fin_sweep(self,st):
  try:
   return self.cheap_sweep()
  except Exception as err:
   self.log.append({"kind":"cheap_sweep_error","error":self._smv_fault(err)})
   return""
 def _smv_fin_depends(self,st):
  if not self._smv_room(5,0):
   return""
  try:
   faults=self.dependency_faults()if self._smv_fit_written()else[]
  except Exception:
   faults=[]
  stamp=(tuple(sorted(self.trolley.units)),tuple(faults))
  tally=self.__dict__.setdefault("dep_told",{})
  if not faults or tally.get(stamp,0)>=2 or len(tally)>=4:
   return""
  tally[stamp]=tally.get(stamp,0)+1
  return("%sCompatibility gap: %s. Query for the dependent item using every identifier in turn as the search text, "
"open the hits and cart a hit whose page mentions an identifier; alternatively switch to an anchor that some "
"dependent page cites by identifier. Order once that holds."%(self._smv_YET,"; ".join(faults)))
 def _smv_fin_vet(self,st):
  if self.order_vetted or not self._smv_room(2,0):
   return""
  self.order_vetted=True
  needs=self.needs
  faults=self.trolley.faults(needs.quantity(),needs.counts())
  if faults and self._smv_room(3):
   return"%sThe cart fails the count check: %s. Correct it and order afterwards."%(self._smv_YET,"; ".join(faults))
  if needs.items():
   lines=list(self.trolley.units)
  else:
   lines=[self.chosen()]if self.chosen()else[]
  rulings=(self.check_item(line,"ORDER")for line in lines)
  ruling=next((r for r in rulings if r),"")
  if ruling:
   return self._smv_YET+ruling
  for line in lines:
   beaten=self.rank_check(line)
   if not beaten:
    continue
   rival=self.rank_verify(line)
   if rival:
    beaten+=(" Lower-priced verified alternative: %s (independent item check: passes current conditions; price "
"below %s) - drop %s, cart %s, refresh its stock and price, order.%s"
%(rival,line,line,rival,self.shelf_met_note(rival)))
   return self._smv_YET+beaten
  shape=("exactly one line per requested product, each in its quantity"if needs.items()
else"one line only: the picked item, quantity as asked")
  return(self._smv_YET+"Pre-order checklist - [cart] "+shape+", nothing extra. [fresh read] latest cart change "
"-> item stock/price looked up again via a tool reporting them; available; unit price x quantity <= current "
"budget. [fit] own properties satisfy every stated condition, disclosed shopper need, announced change. [rank] "
"best/cheapest requested -> nothing reachable beats it. [rules] request's procedural steps all done. Anything "
"unverified: verify it now. Everything verified: repeat the order call.")
 def _smv_fin_steps(self,st):
  return self.finish_steps(st["mode"])
 def _smv_fin_misfit(self,st):
  faults=self.cart_misfits()
  stamp=tuple(sorted(self.trolley.units))
  tally=self.__dict__.setdefault("misfit_told",{})
  if not faults or tally.get(stamp,0)>=2 or self.moves_left()<1:
   return""
  tally[stamp]=tally.get(stamp,0)+1
  self.log.append({"kind":"misfit_hold","lines":list(stamp),"faults":faults})
  return("%sThe records rule this cart out: %s. Such a basket is never accepted. Swap that line for an already read "
"listing that passes (for a fit, one whose page names its anchor's identifiers), then order; when no read "
"listing does, or time is short, decline instead."%(self._smv_YET,"; ".join(faults)[:900]))
 def cart_misfits(self):
  out=[]
  try:
   for line in self.trolley.units:
    fails=self.field_view(line)["fails"]
    if fails:
     out.append("%s: %s"%(line,fails[0]))
   if self._smv_dep_contradicted(self.needs.items()):
    out+=self.dependency_faults()or["a dependent line's page names other models than its anchor line's"]
  except Exception as err:
   self.log.append({"kind":"misfit_error","error":self._smv_fault(err)})
  return out
 def _smv_fin_near_hold(self,st):
  return""if self.cart_misfits()else self.near_hold()
 def _smv_fin_told(self,st):
  cart=self.trolley.units
  if not self._smw_told_current()or self.dead_lines()&set(cart)or self.__dict__.get("told_held",0)>=1:
   return""
  if any(self._smv_tag(line)not in self.passed for line in cart)or self._smv_dep_contradicted(self.needs.items()):
   return""
  cap,grand=self.needs.spend_cap,0.0
  for line,count in cart.items():
   each=self.archive.price(line)
   grand+=float(each)*float(count)if is_number(each)else 0.0
  if is_number(cap)and grand>float(cap)+1e-9:
   return""
  self.told_held=self.__dict__.get("told_held",0)+1
  self.log.append({"kind":"decline_told","lines":[str(line)for line in cart]})
  return("%sThe shopper has been told this exact basket (%s) and answered without any objection, condition or change; "
"nothing moved since. A reply that takes note of the told basket without objecting is the approval the request asks "
"for; this shopper never types a bare yes. Order this cart now. Decline only if a record shows a cart line "
"breaking a stated condition."
%(self._smv_YET,self._smv_list(list(cart))))
 def _smv_fin_dropped(self,st):
  led=self.needs
  if self.trolley.units or self.needs.items()or not led.let_go or not self._smv_room(3):
   return""
  if self.__dict__.get("dropped_held")==len(led.let_go):
   return""
  rows=sorted((cost,ident)for ident,(cost,_,_)in self.shelf_rows().items()
if ident in self.archive.read and ident not in self.refused_lines)
  if not rows:
   return""
  self.dropped_held=len(led.let_go)
  named=[ident for _,ident in rows[:3]]
  self.__dict__.setdefault("named",set()).update(named)
  self.log.append({"kind":"decline_dropped","rows":named})
  return("%sConditions were dropped since the search began (%s), so listings already opened may now fit. Lowest-priced "
"opened, in-stock rows of the asked category within the budget: %s. Check them against the conditions still "
"in force; cart the cheapest that fits, tell the basket if a message is left, then order. Decline only when "
"none of them fits."%(self._smv_YET,"; ".join(map(str,led.let_go))[:200],self._smv_list(named)))
 def _smv_fin_early(self,st):
  several=len(self.needs.items())
  if self.trolley.units or several or self.moves_left()<max(6,self.step_cap//4):
   return""
  times=self.__dict__.get("early_held",0)
  if times>=(1 if several else 2):
   return""
  self.early_held=times+1
  led={}
  for entry in self._smp_book().values():
   score=self.walk_score(entry["args"],None)
   for ident in entry.get("row_shelf")or{}:
    led[ident]=max(led.get(ident,0),score)
  unread=sorted((-led.get(ident,0),cost,ident)for ident,(cost,_,_)in self.shelf_rows().items()
if ident not in self.archive.read and ident not in self.refused_lines)
  named=[ident for _,_,ident in unread[:3]]
  self.log.append({"kind":"decline_early","left":self.moves_left(),"rows":named})
  extra=(" Unopened rows of the asked category, those of searches for a stated value first, then lowest price: "
"%s."%self._smv_list(named))if named else""
  return("%sNothing is carted and %d steps remain, so this decline comes too early. Before giving up: page further "
"through the search that returned the closest rows, search each stated value with the product word, and "
"browse the product's shelf under the budget; open the best unread candidates.%s %s"
%(self._smv_YET,self.moves_left(),extra,self._smv_early_goal(several)))
 @staticmethod
 def _smv_early_goal(several):
  if not several:
   return"Cart the cheapest one that meets every stated condition; decline only when none does."
  return("Do this for each of the %d products, cart the cheapest fit of each, then order the full basket; decline "
"only when some product has no fitting row at all."%several)
 def _smv_fin_pairswap(self,st):
  rows,cart=self.needs.items(),self.trolley.units
  if len(rows)<2 or not cart or not self._smv_room(3)or self.__dict__.get("pairswap_held",0)>=2:
   return""
  first={}
  for line in cart:
   slot=self.product_index(line)
   if slot is not None and slot not in first:
    first[slot]=line
  said=[]
  for k,row,j in self._smv_links(rows):
   if k not in first or j not in first:
    continue
   main_codes=self._smv_main_codes(self._smv_page(first[j]))
   verdict,_=self._smv_fit_verdict(main_codes,self._smv_page(first[k]))
   if verdict is not False:
    continue
   options=self._smv_fit_options(k,main_codes,skip=(first[k],))
   if not options:
    continue
   said.append("take out %s (the %s; its clause leaves out %s's model) and cart %s instead (%s listings whose clause "
"names it, cheapest first: %s), at the requested quantity"%(first[k],row["product"],first[j],options[0],row["product"],
self._smv_list(options)))
  if not said:
   return""
  self.pairswap_held=self.__dict__.get("pairswap_held",0)+1
  self.log.append({"kind":"decline_pairswap","lines":[str(line)for line in cart]})
  return("%sA fitting pair is already in view, so a decline would throw it away: %s. Confirm its stock and price, "
"then order."%(self._smv_YET,"; ".join(said)))
 def _smv_fin_whole(self,st):
  rows,cart=self.needs.items(),self.trolley.units
  if len(rows)<2 or not cart or not self._smv_room(1)or self.__dict__.get("whole_held",0)>=2:
   return""
  slots,grand=set(),0.0
  for line,count in cart.items():
   slot=self.product_index(line)
   if slot is None or self._smv_tag(line)not in self.passed or line in self.refused_lines or line in self.dead_lines():
    return""
   slots.add(slot)
   each=self.archive.price(line)
   grand+=float(each)*float(count)if is_number(each)else 0.0
  cap=self.needs.spend_cap
  if len(slots)<len(rows)or(is_number(cap)and grand>float(cap)+1e-9)or self._smv_dep_contradicted(rows):
   return""
  self.whole_held=self.__dict__.get("whole_held",0)+1
  self.log.append({"kind":"decline_whole","lines":[str(line)for line in cart]})
  return("%sEvery requested product already has a cart line that passed the independent item check on the "
"conditions now in force, each is on sale and the sum fits the budget (%s). A decline would throw a complete "
"basket away. No dependent line's page names a model other than its anchor's, so a fit the pages merely "
"leave unconfirmed is no proof of a misfit. Order this cart. Decline only if a record shows a line breaking "
"a stated condition."%(self._smv_YET,self._smv_list(list(cart))))
 def _smv_dep_contradicted(self,rows):
  if all(row.get("depends_on")is None for row in rows):
   return False
  first={}
  for line in self.trolley.units:
   slot=self.product_index(line)
   if slot is not None and slot not in first:
    first[slot]=line
  for k,row,j in self._smv_links(rows):
   if k not in first or j not in first:
    continue
   base_page,dep_page=self._smv_page(first[j]),self._smv_page(first[k])
   verdict,_=self._smv_fit_verdict(self._smv_main_codes(base_page),dep_page)
   if verdict is not None:
    if not verdict:
     return True
    continue
   coded=self._smv_coded(base_page,dep_page)
   if coded is not None:
    if not coded:
     return True
    continue
   marks=self._smv_idents(base_page)
   own=self._smv_idents(dep_page)
   body=self._smv_flat(dep_page)
   if marks and own and not any(self._smv_flat(m)in body for m in marks):
    return True
  return False
 def _smv_fin_kept(self,st):
  verified=[i for i in self.trolley.units if self._smv_tag(i)in self.passed and i not in self.refused_lines
and i not in self.dead_lines()]
  if not verified or not self._smv_room(1)or self._smv_have("decline_kept")or self.cart_misfits():
   return""
  self.decline_kept=True
  every=" with lines for all products"if self.needs.items()else""
  return("%sVerified cart line present (%s): independent check passed it on every condition now in force; a decline "
"would throw it away. Refresh its stock and price, then order%s. Decline only if a later change clearly "
"excludes it."%(self._smv_YET,self._smv_list(verified),every))
 @staticmethod
 def _smv_reads(queue,spread):
  if not spread or len(queue)<2:
   return len(queue)
  return-(-len(queue)//spread[2])
 def _smv_fin_slack(self,st):
  rows=[]
  if not self._smv_have("slack_done")and(self.role("details")or self.batch_tool()):
   rows=self.slack_rows()
  cost=self._smv_reads(rows,self.batch_tool())
  if not rows or not self._smv_room(cost+4,40):
   return""
  self.slack_done=True
  self.open_queue=rows
  self.open_say="Reading lower-priced rows first: together they might bring all products under the budget."
  total=self._smv_have("slack_total")
  summed="%g"%total if is_number(total)else"more than the budget"
  return("%sBudget squeeze: each product has fits, but the cheapest ones together cost %s, past the %s budget. "
"Unopened rows fall inside each product's budget share (budget less the other products' cheapest fits); the "
"code opens them for you now: %s. Rate each on its own product's conditions by the value rules (a dedicated "
"value field of this store counts only by its wording). A combination covering all products and staying "
"affordable -> a line per product, then order. Decline only when no combination works."
%(self._smv_YET,summed,self.needs.spend_cap,self._smv_list(rows)))
 def _smv_fin_fill(self,st):
  spread=st["spread"]=self.batch_tool()
  rows=[]
  if not self._smv_have("decline_fill")and(self.role("details")or spread):
   rows=self.lacking_rows()
  if not rows or not self._smv_room(self._smv_reads(rows,spread)+3,40):
   return""
  self.decline_fill=True
  self.open_queue=rows
  self.open_say="Opening listings never read for products lacking a verified fit, prior to any decline."
  return("%sNo verified fit yet for some requested products, while affordable listings from their categories sit "
"unopened. The code opens them for you now, before any decline: %s. Rate each on its own product's conditions "
"by the value rules (a dedicated value field of this store counts only by its wording). Every product covered "
"by a fitting listing -> a line per product, then order. Decline only for a product with truly none."
%(self._smv_YET,self._smv_list(rows)))
 def _smv_fin_named(self,st):
  spread=st["spread"]
  queue=self.named_unread()[:spread[2]if spread else 3]
  if not queue or self._smv_have("decline_opens")or not(self.role("details")or spread):
   return""
  if not self._smv_room(self._smv_reads(queue,spread)+3,40):
   return""
  self.decline_opens=True
  self.open_queue=queue
  return("%sUnread leads: %s came up in the checks yet were never opened. Their records get fetched by the code "
"right now, ahead of declining. Value rules: the property's namesake field is decisive (a store-written value "
"in it counts by its wording alone); without that field, a different field, the option menu or the listing "
"title can supply the value. Buy whichever fits; decline only when every lead fails."
%(self._smv_YET,self._smv_list(queue)))
 def _smv_prefixed(self,note):
  return self._smv_YET+note if note else""
 def _smv_fin_pages(self,st):
  return self._smv_prefixed(self.decline_reads())
 def _smv_fin_reaudit(self,st):
  grew=len(self.archive.read)>self._smv_have("audit_reads",10**9)and self._smv_have("audits",0)<2
  if grew and self.decline_vetted and self._smv_room(3,30):
   return self._smv_prefixed(self.review_decline())
  return""
 def _smv_fin_audit(self,st):
  if self.decline_vetted or not self._smv_room(3,30):
   return""
  self.decline_vetted=True
  return self._smv_prefixed(self.review_decline())or(
self._smv_YET+"Decline gate - confirm: [search] several angles tried (varied words, filters, later pages); "
"[read] every candidate's own properties opened; [fail] each candidate breaks a stated condition, a disclosed "
"need, the budget or stock. Field keys count under any spelling. Fitting item found -> buy it. None -> repeat "
"the decline call.")
 def _smv_fin_fit_hold(self,st):
  return self._smv_prefixed(self.fit_hold())
 def _smv_fin_best(self,st):
  pool=[]if self._smv_have("best_held")or not self._smv_room(4)else self.unrefuted()
  tries=0
  while tries<3:
   tries+=1
   todo=[i for i in pool if self._smv_tag(i)not in self.passed]
   if not todo or self.wire.spare()<CLOSING+30:
    break
   self.check_item(todo[0],"ORDER")
   pool=self.unrefuted()
  if not pool or not self._smv_room(len(pool)+3,20):
   return""
  self.best_held=True
  several=bool(self.needs.items())
  return("%sAn option still stands. A decline is correct only when no listing can satisfy the conditions; yet the "
"records still hold %s (in stock, inside the budget, excluded by no check): %s. Unless some record shows one "
"of them breaking a condition now in force, add %s to the cart, read stock and price again, and order; decline "
"only when every option plainly fails."
%(self._smv_YET,"options covering all products"if several else"an option",
self._smv_list(pool),"every product's line (each in its quantity)"if several else"it"))
 def _smv_fin_unknown_ref(self,st):
  open_refs=[]
  for ref in self._smv_have("refs")or[]:
   if ref.get("relation")=="differ"and ref.get("status")in("open","stuck")and self.ref_live(ref):
    open_refs.append(ref)
  if not open_refs or not self._smv_room(3)or self._smv_have("ref_decline_held"):
   return""
  self.ref_decline_held=True
  heads=self._smv_list(dict.fromkeys(ref["head"]for ref in open_refs))
  per_product=", one per product"if self.needs.items()else""
  return("%sUnresolved reference: %s hinges on a listing the code could not pin down, value unknown. Unknown means "
"it excludes nothing and gives no ground for declining. Cart the listing meeting all remaining conditions (in "
"stock, affordable)%s, refresh its stock and price, then order; decline only if no listing meets those "
"remaining conditions."%(self._smv_YET,heads,per_product))
 def _smv_fin_repair(self,st):
  return self.decline_repair()
 def _smv_fin_meaning(self,st):
  return self.near_check()
 def change_review(self):
  wire,book,cart,needs=self.wire,self.archive,self.trolley,self.needs
  held=self._smv_have("reviews",0)
  if held>=3 or not book.read or wire.mute or wire.spare()<CLOSING+20 or not wire.usable(CHECKER)or self.moves_left()<3:
   return""
  self.reviews=held+1
  cards=book.cards(14000,"",self.ref_foreign())
  if not cards:
   return""
  detail=needs.reading.get("priority_detail")or""
  parts=[CHANGE_REVIEW,
"THE SHOPPER'S REQUEST:\n"+self.request+needs.changes(),
"WHAT JUST CHANGED: "+" / ".join(needs.exchanges[-2:]),
"REQUIREMENTS NOW: "+needs.text_body(),
("PRIORITY: %s %s"%(needs.priority(),detail)).rstrip(),
"THE CART: "+cart.summary(),
"LISTINGS SEEN - DETAILS card = whole record read; ROW card = search-row data only:\n"+"\n".join(cards)]
  reply=wire.consult("\n\n".join(parts)+self.field_names(),CHECK_TOKENS,CHECKER)
  self.log.append(dict(kind="change_review",answer=reply))
  if not reply:
   return""
  def known(x):
   return isinstance(x,(str,int))and str(x)in book.ids
  good=list(map(str,filter(known,reply.get("fits")or[])))[:6]
  choice=reply.get("pick")
  choice=str(choice)if known(choice)else""
  self.__dict__.setdefault("named",set()).update(good)
  still=reply.get("cart_still_right")
  out=["[after the change]"]
  out.append("fits among the opened records, strongest first: "+self._smv_list(good)if good
else"nothing opened so far is shown to satisfy the current conditions")
  if choice:
   out.append("pick now: "+choice)
  if cart.units and still is True:
   out.append("no cart change: its line still qualifies, and the basket described to the shopper should be what ships")
  elif cart.units and still is False:
   out.append("replace the cart's line: under the current terms it stopped being the right pick (ask the shopper "
"first when the instructions require a check before any replacement)")
  why=reply.get("reason")
  if why:
   out.append(str(why)[:300])
  out.append("Priority now: re-choose among current fits (cheapest/best of them if the request sets that priority), fix "
"the cart, refresh stock and price, finish. An opened record still fits -> no decline.")
  return" ".join(out)
 def code_asks_swap(self,old,new):
  tool=self.role("message")
  field=self.text_arg(tool)if tool else None
  spare_msgs=self.ask_limit()-self.msgs_sent
  if not field or spare_msgs<self.msg_reserve()+1 or self.moves_left()<4:
   return False
  gone,put=old[:3],new[:3]
  question=("Before I change the basket: I would like to take out %s and put in %s instead, since the first no longer "
"fits. Is that all right with you?"%(self._smv_list(gone),self._smv_list(put)))
  replies=self.send_chunk([("code_swap_ask_%s"%self.wire.tick,tool,{field:question})])
  self.log.append({"kind":"code_swap_ask","old":gone,"new":put})
  return bool(replies)and not str(replies[0][1]).startswith("Not run")
 @staticmethod
 def _smv_names(layer,ident):
  return any(isinstance(x,str)and x==ident and"id"in str(name).lower()for name,x in layer.items())
 def line_seller(self,thing):
  learned=self.seller_keys()
  latest=""
  def scan(layer):
   nonlocal latest
   if not self._smv_names(layer,thing):
    return
   for name,x in layer.items():
    label=str(name)
    low=label.lower()
    if isinstance(x,str)and x.strip()and len(x)<=60 and"id"not in low and(re.search(SELLER_KEYS,low)or label in learned):
     latest=x.strip()
  for rec in self.archive.items:
   walk_dicts(rec,scan)
  return latest
 def _smv_page(self,ident):
  texts=[]
  for rec in self.archive.items:
   walk_dicts(rec,lambda layer:long_strings_all(layer,texts)if self._smv_names(layer,ident)else None)
  return" ".join(dict.fromkeys(texts))
 @staticmethod
 def _smv_idents(text):
  hits=re.findall(r"\b([A-Z][A-Za-z]{1,12}[ -]?\d{3,6})\b",text)
  return[*dict.fromkeys(hits)][:8]
 @staticmethod
 def _smv_codes(text):
  alone=set(re.findall(r"(?<![\w-])([A-Z]{1,4}\d{1,3})(?![\w-])",text))
  heads=set(re.findall(r"(?<![\w-])([A-Z]{1,4}\d{1,3})-\d{3,6}(?![\w-])",text))
  return alone,heads
 @staticmethod
 def _smv_stem(code):
  return code.rstrip("0123456789")
 def _smv_coded(self,base_page,dep_page):
  base,dep=self._smv_codes(base_page),self._smv_codes(dep_page)
  if base[0]&dep[1]or dep[0]&base[1]:
   return True
  stems={self._smv_stem(code)for code in base[0]}
  if any(self._smv_stem(head)in stems for head in dep[1]):
   return False
  return None
 def _smv_series(self,page):
  memo=self.__dict__.setdefault("_smv_stem_memo",{})
  size=len(self.archive.items)
  if memo.get("size")!=size:
   texts=[]
   for rec in self.archive.items:
    walk_dicts(rec,lambda layer:long_strings_all(layer,texts))
   heads=self._smv_codes(" ".join(dict.fromkeys(texts)))[1]
   memo.update(size=size,stems={self._smv_stem(head)for head in heads})
  return sorted(code for code in self._smv_codes(page)[0]if self._smv_stem(code)in memo["stems"])
 @staticmethod
 def _smv_flat(text):
  return"".join(ch for ch in str(text).lower()if ch in"abcdefghijklmnopqrstuvwxyz0123456789")
 _smv_CODE=r"(?:[A-Z][A-Za-z]{0,5}[- ]?\d{2,6}|[A-Z][a-z]{1,12} \d{2,5})"
 _smv_CODES=re.compile(r"("+_smv_CODE+r"(?:\s*(?:,|/|&|\band\b|\bor\b)\s*"+_smv_CODE+r")*)")
 _smv_FIT_LEAD=re.compile(r"\b(?:works?|compatible|goes|go|fits?|suits?|suitable|made|designed|intended)\s+(?:with|for|on)\s*:?\s*",re.I)
 _smv_FAMILY=re.compile(r"\b(?:every|all|any|each)\s+([A-Z][A-Za-z]{0,5}[- ]?\d{0,5})([xX*#?]{1,4})\b")
 _smv_EXCEPT=re.compile(r"\b(?:except(?:ing)?|excluding|but not|other than|save|apart from|not)\s+(?:for\s+)?(?:the\s+)?",re.I)
 _smv_CHAIN=re.compile(r"\b(?:more on|see|refer to|check|look at|details on|rest on|listed on)\s+part\s+([A-Za-z0-9-]{3,20})",re.I)
 _smv_REST=re.compile(r"\bpart\s+([A-Za-z0-9-]{3,20})\s+for\s+the\s+rest",re.I)
 _smv_ALSOFITS=re.compile(r"\bpart\s+([A-Za-z0-9-]{3,20})\s+(?:also\s+)?(?:fits?|works with|suits|goes with)\s*:?\s*",re.I)
 _smv_PARTTAG=re.compile(r"\[([A-Za-z]{1,4}\d{3,8})\]|\(part\s+([A-Za-z0-9-]{3,20})\)|\bpart\s+([A-Za-z]{1,4}\d{3,8})\b",re.I)
 @staticmethod
 def _smv_norm_code(code):
  return"".join(ch for ch in str(code).upper()if ch.isalnum())
 @classmethod
 def _smv_code_run(cls,text):
  m=cls._smv_CODES.match(text or"")
  if not m:
   return[],""
  codes=[cls._smv_norm_code(c)for c in re.findall(cls._smv_CODE,m.group(1))]
  return codes,text[m.end():]
 @classmethod
 def _smv_fit_clause(cls,text):
  text=str(text or"")
  out={"codes":set(),"banned":set(),"families":[],"chain":[],"clause":False}
  for m in cls._smv_FAMILY.finditer(text):
   prefix=cls._smv_norm_code(m.group(1))
   if not prefix:
    continue
   lead=text[max(0,m.start()-40):m.start()]
   if not cls._smv_FIT_LEAD.search(lead):
    continue
   tail=text[m.end():m.end()+160]
   banned=set()
   em=cls._smv_EXCEPT.search(tail)
   if em and em.start()<=40:
    banned=set(cls._smv_code_run(tail[em.end():])[0])
   out["families"].append((prefix,banned))
   out["clause"]=True
  for m in cls._smv_FIT_LEAD.finditer(text):
   tail=text[m.end():m.end()+220]
   if cls._smv_FAMILY.match(tail.lstrip()):
    continue
   codes,rest=cls._smv_code_run(tail)
   if not codes:
    continue
   out["codes"].update(codes)
   out["clause"]=True
   em=cls._smv_EXCEPT.match(rest.lstrip())
   if em:
    out["banned"].update(cls._smv_code_run(rest.lstrip()[em.end():])[0])
  for m in cls._smv_CHAIN.finditer(text):
   out["chain"].append(cls._smv_norm_code(m.group(1)))
  for m in cls._smv_REST.finditer(text):
   out["chain"].append(cls._smv_norm_code(m.group(1)))
  out["chain"]=[*dict.fromkeys(out["chain"])]
  return out
 @classmethod
 def _smv_page_tag(cls,text):
  for m in cls._smv_PARTTAG.finditer(str(text or"")):
   tag=next((g for g in m.groups()if g),"")
   if tag:
    return cls._smv_norm_code(tag)
  return""
 def _smv_also_fits(self):
  memo=self.__dict__.setdefault("_smv_also_memo",{})
  size=len(self.archive.items)
  if memo.get("size")!=size:
   found={}
   texts=[]
   for rec in self.archive.items:
    walk_dicts(rec,lambda layer:long_strings_all(layer,texts))
   for text in dict.fromkeys(texts):
    for m in self._smv_ALSOFITS.finditer(text):
     codes=self._smv_code_run(text[m.end():m.end()+200])[0]
     if codes:
      found.setdefault(self._smv_norm_code(m.group(1)),set()).update(codes)
   memo.update(size=size,map=found)
  return memo["map"]
 def _smv_main_codes(self,base_page):
  codes={self._smv_norm_code(c)for c in self._smv_idents(base_page)}
  return{c for c in codes if c}
 def _smv_fit_verdict(self,main_codes,dep_page):
  clause=self._smv_fit_clause(dep_page)
  tag=self._smv_page_tag(dep_page)
  extra=self._smv_also_fits().get(tag,set())if tag else set()
  named=clause["codes"]|extra
  if named or clause["families"]:
   clause["clause"]=True
  if not clause["clause"]:
   return None,clause
  for code in main_codes:
   if code in clause["banned"]:
    return False,clause
  for code in main_codes:
   if code in named:
    return True,clause
   for prefix,banned in clause["families"]:
    if code.startswith(prefix)and code not in banned:
     return True,clause
  return False,clause
 def _smv_fit_options(self,k,main_codes,skip=()):
  where=self._smv_row_shelves()
  mine={line for line in self.trolley.units if self.product_index(line)==k}
  shelves={where.get(line)for line in mine if where.get(line)}
  if not shelves:
   shelves={where.get(i)for i in where if self.product_index(i)==k}
  found=[]
  for ident,label in where.items():
   if label not in shelves or ident in mine or ident in skip or ident in self.dead_lines()or ident in self.refused_lines:
    continue
   verdict,_=self._smv_fit_verdict(main_codes,self._smv_shelf_page(ident))
   if verdict is not True or self.last_stock(ident)is False:
    continue
   price=self.archive.price(ident)
   found.append((float(price)if is_number(price)else float("inf"),ident))
  return[ident for _,ident in sorted(found)][:4]
 @staticmethod
 def _smv_links(rows):
  if not rows or all(r.get("depends_on")is None for r in rows):
   return[]
  ties=[]
  for k,row in enumerate(rows):
   anchor=row.get("depends_on")
   if anchor is not None and anchor!=k and not anchor>=len(rows):
    ties.append((k,row,anchor))
  return ties
 def dependency_faults(self):
  rows=self.needs.items()
  if not rows or all(r.get("depends_on")is None for r in rows):
   return[]
  first={}
  for line in self.trolley.units:
   slot=self.product_index(line)
   if slot is not None and slot not in first:
    first[slot]=line
  out=[]
  for k,row,j in self._smv_links(rows):
   if k not in first or j not in first:
    continue
   base_line,dep_line=first[j],first[k]
   base_page,dep_page=self._smv_page(base_line),self._smv_page(dep_line)
   main_codes=self._smv_main_codes(base_page)
   verdict,clause=self._smv_fit_verdict(main_codes,dep_page)
   if verdict is True:
    continue
   if verdict is False:
    options=self._smv_fit_options(k,main_codes,skip=(dep_line,))
    said=("line %s (the %s) is meant for %s (the %s), but its fit clause leaves out %s's model (%s)"
%(dep_line,row["product"],base_line,rows[j]["product"],base_line,self._smv_list(sorted(main_codes)[:4])))
    if options:
     said+="; same-shelf listings whose clause names it, cheapest first: "+self._smv_list(options)
    if clause["chain"]:
     said+="; its page also points to part %s for the rest, search that tag once"%self._smv_list(clause["chain"][:2])
    out.append(said)
    continue
   coded=self._smv_coded(base_page,dep_page)
   names=self._smv_idents(base_page)
   marks=self._smv_series(base_page)+names
   if coded or not marks:
    continue
   body=self._smv_flat(dep_page)
   if coded is None and any(self._smv_flat(m)in body for m in names):
    continue
   if clause["chain"]:
    out.append("line %s (the %s) is meant for %s (the %s); its page states no fit itself but points to part %s for it: "
"search that tag once and read the 'also fits' line there"%(dep_line,row["product"],base_line,rows[j]["product"],
self._smv_list(clause["chain"][:2])))
    continue
   out.append("line %s (the %s) is meant for %s (the %s), and its page mentions none of the identifiers %s carries (%s)"
%(dep_line,row["product"],base_line,rows[j]["product"],base_line,self._smv_list(marks)))
  return out
 def _smv_fit_written(self):
  rows=self.needs.items()
  if not rows:
   return False
  where=self._smv_row_shelves()
  lines={}
  for line in self.trolley.units:
   slot=self.product_index(line)
   if slot is not None:
    lines.setdefault(slot,line)
  for k,row,j in self._smv_links(rows):
   if k not in lines or j not in lines:
    continue
   base_page=self._smv_page(lines[j])
   marks=[self._smv_flat(m)for m in self._smv_idents(base_page)]
   coded=bool(self._smv_series(base_page))
   shelf=where.get(lines[k])
   if not(marks or coded)or not shelf:
    continue
   main_codes=self._smv_main_codes(base_page)
   for ident,label in where.items():
    if label!=shelf or ident==lines[k]:
     continue
    if main_codes and self._smv_fit_clause(self._smv_shelf_page(ident))["clause"]:
     return True
    if coded and self._smv_coded(base_page,self._smv_shelf_page(ident))is not None:
     return True
    if any(m and m in self._smv_shelf_text(ident)for m in marks):
     return True
  return False
 def _smv_shelf_page(self,ident):
  memo=self.__dict__.setdefault("_smv_flat_memo",{})
  size=len(self.archive.items)
  if memo.get("size")!=size:
   memo.clear()
   memo["size"]=size
  key=("page",ident)
  if key not in memo:
   memo[key]=self._smv_page(ident)
  return memo[key]
 def _smv_shelf_text(self,ident):
  page=self._smv_shelf_page(ident)
  memo=self.__dict__["_smv_flat_memo"]
  if ident not in memo:
   memo[ident]=self._smv_flat(page)
  return memo[ident]
 def dependency_hint(self):
  rows=self.needs.items()
  said=[]
  for k,row,j in self._smv_links(rows):
   base_line=next((line for line in self.trolley.units if self.product_index(line)==j),"")
   page=self._smv_page(base_line)if base_line else""
   marks=self._smv_series(page)+self._smv_idents(page)if base_line else[]
   dep,anchor=row["product"],rows[j]["product"]
   coded=self._smv_coded_anchors(j)if not marks else[]
   tail=self._smv_code_words(k)
   if marks:
    said.append("%s (goes with cart line %s, the %s): a %s page fits when its clause names one of %s or a family "
"covering it (prefix before the x's, exceptions excluded); a clause that leaves them out excludes the page; "
"read the clause in the rows' descriptions%s"%(dep,base_line,anchor,dep,self._smv_list(marks),tail))
    try:
     options=self._smv_fit_options(k,self._smv_main_codes(page))
    except Exception:
     options=[]
    if options:
     said.append("%s pages whose clause names cart line %s, cheapest first: %s"%(dep,base_line,self._smv_list(options)))
   elif coded:
    said.append("%s (goes with the %s): this store writes the fit as a code. A %s page states a series code "
"(read so far: %s), and a %s fits only when its listing tag starts with that same code; a tag of "
"that series with another number does not fit. Cart a pair sharing one code, querying %s "
"listings with the code as the search text%s"%(dep,anchor,anchor,self._smv_list(coded),dep,
dep,tail))
   else:
    said.append("%s (goes with the %s): settle the %s line first; afterwards query %s listings by model "
"names/numbers and keep a page quoting one"%(dep,anchor,anchor,dep))
  if not said:
   return""
  pair=self.pair_best()
  if pair:
   said.append("lowest total so far: %s, %g in all; tied products are priced together, so a cheaper line whose "
"code only a dearer partner shares does not beat it"%(self._smt_priced(list(pair)),sum(
self.archive.price(mark)*units for mark,units in pair.items())))
  return"\n[works with] "+"; ".join(said)
 def pair_best(self):
  rows=self.needs.items()
  links=self._smv_links(rows)if self.needs.priority()=="cheapest"else[]
  if not links or not self.archive.read or self._smr_open():
   return{}
  stamp=(len(self.archive.items),self.needs.version,"|".join(self.refused_lines),len(self.dead_lines()))
  memo=self.__dict__.get("pair_memo")
  if memo and memo[0]==stamp:
   return memo[1]
  anchor=links[0][2]
  tied=[k for k,_row,j in links if j==anchor]
  info,units=self.code_rows(),self.needs.counts()
  fits={k:self._smv_pair_fits(k,info)for k in[anchor,*tied]}
  best,lowest={},None
  for cost,base in fits[anchor]:
   page=self._smv_page(base)
   codes=set(self._smv_series(page))
   main_codes=self._smv_main_codes(page)
   def mate_of(k):
    for row in fits[k]:
     if codes and self._smv_codes(info[row[1]]["title"])[1]&codes:
      return row
     if main_codes and self._smv_fit_verdict(main_codes,self._smv_shelf_page(row[1]))[0]is True:
      return row
    return None
   mates=[mate_of(k)for k in tied]
   if not(codes or main_codes)or None in mates:
    continue
   total=cost*units[anchor]+sum(mate[0]*units[k]for mate,k in zip(mates,tied))
   if lowest is None or total<lowest:
    lowest=total
    best={base:units[anchor],**{mate[1]:units[k]for mate,k in zip(mates,tied)}}
  self.pair_memo=(stamp,best)
  return best
 def _smv_pair_fits(self,k,info):
  judged=[(prop,value)for prop,value in self.code_values(k)if not Errand._smm_BOUND.search(value.lower())]
  gone=self.dead_lines()|set(self.refused_lines)
  found=[]
  for ident in self.archive.read:
   row,cost=info.get(ident),self.archive.price(ident)
   if row is None or not is_number(cost)or ident in gone or row["stock"]is False or self.product_index(ident)!=k:
    continue
   if not self.field_view(ident)["fails"]and all(self.code_shows(row,prop,value,set(),True)for prop,value in judged):
    found.append((float(cost),ident))
  return sorted(found)
 def _smv_code_words(self,k):
  values=self._smt_values(k)
  words=values[0][0]if values and values[0]else[]
  if not words:
   return""
  return" (query '<identifier> %s' to list only hits holding that value too, a short list to read to its end)"%" ".join(words)
 def reveal_note(self,lines):
  kv=self.__dict__
  due=kv.setdefault("reveal_due",[])
  for line in lines if not self.needs.items()else[]:
   value=self._smt_bare_value(str(line)).strip()
   first=self._smt_choices(value)[0].strip()if value else""
   if re.search(CH_BOUND,value.lower()):
    continue
   if first and not ref_text(value)and len(first.split())<=4 and kv.get("reveal_sent",0)+len(due)<REVEAL_MAX:
    due.append(first)
 def reveal_settle(self,link):
  if self.needs.items()or link["relation"]not in("same","pairs"):
   return
  said=link.get("goes_with")if link["relation"]=="pairs"else[link.get("value")]
  value=str((said or[""])[0]).strip().lower()
  head,_,tail=value.rpartition(" ")
  shown=(text for row in self.code_rows().values()for text in self._smm_prop_fields(row.get("fields")or{},link["prop"]))
  if head and same_prop(tail,link["prop"])and any(self.code_seen(head,text)for text in shown):
   value=head
  if not value or ref_text(value)or re.search(CH_BOUND,value)or len(value.split())>4:
   return
  kv=self.__dict__
  due=kv.setdefault("reveal_due",[])
  due[:]=[" ".join([value,*due])]
  kv["reveal_deep"]=due[0]
 def reveal_reach(self):
  kv=self.__dict__
  if not kv.get("reveal_job")and not kv.get("reveal_due"):
   return False
  if self.needs.items()or self.trolley.units or self.orders_tried or not self.page_arg()or not self._smv_room(self.order_due()+4,50):
   kv.pop("reveal_job",None)
   kv["reveal_due"]=[]
   return False
  job=kv.get("reveal_job")or self._smv_reveal_job()
  if not job:
   return self.reveal_reach()if kv.get("reveal_due")else False
  kv["reveal_job"]=job
  entry=self._smp_book().get(self._smo_sig(job["args"]))
  if entry is None:
   return self.walk_call(job["args"],"Searching the value just revealed with the product words.")
  turn=self.next_page(entry)if job["pages"]<(REACH_ROWS if job["deep"]else REVEAL_PAGES*4)//max(4,entry["size"])else None
  if turn is not None:
   job["pages"]+=1
   return self.walk_call({**job["args"],self.page_arg()[0]:turn},"Next page of that search.")
  kv.pop("reveal_job",None)
  kv["reach_done"]=kv.get("reach_done")or job["deep"]
  return self._smv_reveal_open([entry],job["deep"])
 def _smv_reveal_job(self):
  kv=self.__dict__
  value=kv["reveal_due"].pop(0)
  form=self.search_template()
  product=str(self.needs.reading.get("product")or"").strip()
  if not form or not product:
   return None
  others,slot=form
  args,faults=ArgCheck.execute({**others,slot:f"{value} {product}"},self._smt_search_schema())
  if faults or not isinstance(args,dict):
   return None
  kv["reveal_sent"]=kv.get("reveal_sent",0)+1
  return{"args":args,"pages":0,"deep":value==kv.get("reveal_deep")}
 def _smv_reveal_open(self,entries,whole=False):
  home=self.walk_home(None)
  shown=[ident for entry in entries for ident,shelf in(entry.get("row_shelf")or{}).items()if not home or shelf in home]
  pick=self.walk_pick(None)
  rows=[mark for _,mark in self.cheap_unread(shown,None,pick[1]if pick else None)]
  rows=rows[:2*REVEAL_OPENS]if whole else sorted(rows,key=lambda mark:not self.row_carries(mark,None))[:REVEAL_OPENS]
  if not rows:
   return False
  self.open_recheck,self.open_purpose=False,"walk"
  self.open_queue,self.open_say=rows,"Opening the cheapest rows of that search."
  return self.open_named()
 def solo_reach(self):
  kv=self.__dict__
  if kv.get("reach_done")or not(self.msgs_sent or not self.role("message")):
   return False
  if self.needs.items()or self.trolley.units or self.orders_tried or not self.page_arg()or not self._smv_room(self.order_due()+3,50)or self._smv_cap_settled():
   return False
  spent=kv.get("reach_rows",0)
  entry=self._smv_reach_lead()if spent<REACH_ROWS else None
  if entry is None:
   kv["reach_done"]=True
   return self._smv_reveal_open(list(self._smp_book().values()))if spent else False
  kv["reach_rows"]=spent+max(1,entry["size"])
  return self.walk_call({**entry["args"],self.page_arg()[0]:self.next_page(entry)},"Next page of the closest search.")
 def _smv_reach_lead(self):
  home=self.walk_home(None)
  best,most=None,(0,0)
  chased={text for link in self._smr_links()for text in link.get("queries")or[]}
  for entry in self._smp_book().values():
   if" ".join(self.free_text(entry["args"])).lower()in chased:
    continue
   shelves=(entry.get("row_shelf")or{}).values()
   count=sum(1 for shelf in shelves if not home or shelf in home)
   rank=(self.walk_score(entry["args"],None),count)
   if count and rank>most and self.next_page(entry)is not None:
    best,most=entry,rank
  return best
 def _smv_cap_tool(self,only=None):
  for name,shape in self.shapes.items():
   props=shape.get("properties")if isinstance(shape,dict)and name==(only or name)else None
   if not isinstance(props,dict):
    continue
   texts=[key for key,spec in props.items()if isinstance(spec,dict)and spec.get("type")=="string"
and not spec.get("enum")]
   shelf=next((key for key in texts if self._smu_shelf_names.search(key.lower())),None)
   turn=next((key for key in texts if re.search(r"token|cursor|next|continu",key.lower())),None)
   limit=self._smu_cap_arg(props,turn)if shelf and turn else None
   if limit and not set(shape.get("required")or[])-{shelf,limit}:
    return name,shelf,limit,turn
  return None
 def _smv_cap_page(self,mark):
  for obs in self.archive.items[mark:]:
   rows=self._smu_rows_of(obs)if isinstance(obs,dict)else None
   if rows is None and isinstance(obs,dict):
    obs=next((part for part in obs.values()if isinstance(part,dict)and self._smu_rows_of(part)is not None),obs)
    rows=self._smu_rows_of(obs)
   if rows:
    totals=[n for n in obs.values()if type(n)is int and n>len(rows)]
    return len(rows),max(totals)if totals else len(rows),self.next_token(obs)
  return 0,0,None
 def _smv_cap_settled(self):
  job=self.__dict__.get("cap_job")or{}
  if not job.get("complete"):
   return False
  rows=self.shelf_rows()
  return any(ident in self.archive.read and rows[ident][0]<=job["bound"]and not self.field_view(ident)["fails"]
for ident in rows)
 def cap_reach(self):
  kv=self.__dict__
  if kv.get("cap_done")or not self._smu_lowest_wanted()or not(self.msgs_sent or not self.role("message")):
   return False
  found,label=self._smv_cap_tool(),self.raw_label(self.walk_home(None))
  prices=[row[0]for row in self.shelf_rows().values()]
  if self.needs.items()or self.trolley.units or self.orders_tried or not(found and label and prices)or not self._smv_room(self.order_due()+3,50):
   kv["cap_done"]=True
   return False
  job=kv.setdefault("cap_job",{"calls":0,"bound":self.below(min(prices)),"token":None,"lifted":False})
  if job["bound"]is None or job["calls"]>=CAP_CALLS or job.get("complete"):
   return self._smv_cap_open()
  tool,shelf,limit,turn=found
  args={shelf:label,limit:job["bound"]}
  if job["token"]:
   args[turn]=job["token"]
  mark,tick=len(self.archive.items),self.wire.tick
  self.walk_call(args,"Listing the category under the cheapest row seen: results come in id order, cheapest wanted.",tool)
  if self.wire.tick==tick:
   return self._smv_cap_open()
  job["calls"]+=1
  shown,total,token=self._smv_cap_page(mark)
  if not shown and not job["lifted"]:
   job.update(lifted=True,bound=min(prices)*2)
  elif not shown or total<=CAP_ROWS:
   job.update(token=token,complete=token is None)
  else:
   lower=self.ceil_mid(min(row[0]for row in self.shelf_rows().values()),job["bound"])
   job.update(bound=lower,complete=not lower<job["bound"])
  return True
 def _smv_cap_open(self):
  kv=self.__dict__
  kv["cap_done"]=True
  batch=self.batch_tool()
  rows=[mark for _,mark in self.cheap_unread(list(self.shelf_rows()),None)][:2*batch[2]if batch else REVEAL_OPENS]
  if not rows:
   return False
  self.open_recheck,self.open_purpose=False,"walk"
  self.open_queue,self.open_say=rows,"Opening the cheapest rows of the category."
  return self.open_named()
 def few_reach(self):
  kv=self.__dict__
  if kv.get("few_done")or not(self.msgs_sent or not self.role("message"))or(self._smr_open()and self.ref_room()):
   return False
  rows=self.needs.items()
  if not(self._smu_lowest_wanted()and 2<=len(rows)<=FEW_MAX)or self.trolley.units or self.orders_tried or not self.page_arg()or not self._smv_few_room():
   kv["few_done"]=True
   return False
  tied={k for k,_row,_j in self._smv_links(rows)}
  job=kv.setdefault("few_job",{"todo":[k for k in range(len(rows))if k not in tied and self._smt_values(k)],"args":None,
"rows":0,"sent":False,"read":[],"hunts":None})
  tick=self.wire.tick
  for phase in(self._smv_few_page,self._smv_few_open,self._smv_few_hunt,self._smv_few_tagged):
   phase(job)
   if self.wire.tick!=tick:
    return True
  kv["few_done"]=True
  return False
 def _smv_few_room(self):
  return self._smv_room(self.order_due()+FEW_KEEP,LAST_CALL-CLOSING)
 def _smv_few_page(self,job):
  while job["todo"]:
   k=job["todo"][0]
   if job["args"]is None:
    job.update(args=self._smv_few_args(k)or{},rows=0)
    job["sent"]=self._smo_sig(job["args"])in self._smp_book()
   args=job["args"]
   if args and not job["sent"]:
    job["sent"]=True
    self.walk_call(args,"Searching the required value with the category words: the cheapest fit may sit deep.")
    return
   entry=self._smp_book().get(self._smo_sig(args))if args else None
   turn=self.next_page(entry)if entry and job["rows"]<FEW_ROWS and self._smv_few_room()else None
   if turn is not None:
    job["rows"]+=max(1,entry["size"])
    self.walk_call({**args,self.page_arg()[0]:turn},"Next page of that search.")
    return
   job["read"].append(job["todo"].pop(0))
   job["args"]=None
 def _smv_few_args(self,k):
  form,label=self.search_template(),self.raw_label(self.walk_home(k))
  texts=self._smt_forms(k,label,None)[0]if label else[]
  if not(form and texts):
   return None
  args,faults=ArgCheck.execute({**form[0],form[1]:texts[0]},self._smt_search_schema())
  return None if faults or not isinstance(args,dict)else args
 def _smv_few_open(self,job):
  due,job["read"]=job["read"],[]
  queue=[]
  for k in due:
   queue+=[mark for _,mark in self.cheap_unread(self.home_rows(self.walk_home(k)),k)][:LIST_READ_CAP]
  self._smv_few_launch(queue,due,"Reading the cheapest rows of the value searches.")
 def _smv_few_launch(self,queue,products,say):
  floor=self.order_due()+FEW_KEEP
  queue=self._smt_trim(queue,self.batch_tool(),floor)
  if queue:
   homes=set().union(*[self.walk_home(k)for k in products])
   self.vl_launch(queue,homes,floor,say)
 def _smv_few_hunt(self,job):
  if job["hunts"]is None:
   job["hunts"]=self._smv_few_hunts()
  form=self.search_template()
  while job["hunts"]and form and self._smv_few_room():
   text=job["hunts"].pop(0)
   args,faults=ArgCheck.execute({**form[0],form[1]:text},self._smt_search_schema())
   if not faults and isinstance(args,dict)and not self.ran_query(text):
    self.walk_call(args,"Running '%s' as a keyword query: the anchor's series code first."%text)
    return
 def _smv_few_hunts(self):
  if self._smr_open():
   return[]
  rows=self.needs.items()
  info=self.code_rows()
  out=[]
  for k,_row,j in self._smv_links(rows):
   codes=[]
   for _cost,base in self._smv_pair_fits(j,info):
    codes+=[code for code in self._smv_series(self._smv_page(base))if code not in codes]
   values=self._smt_values(k)
   alts=values[0][:PAIR_ALTS]if values and values[0]else[[]]
   out+=[" ".join([code,*words])for code in codes[:PAIR_ANCHORS]for words in alts]
  return out
 def _smv_few_tagged(self,job):
  if job.get("tagged"):
   return
  job["tagged"]=True
  rows=self.needs.items()
  info=self.code_rows()
  queue=[]
  products=[]
  for k,_row,j in self._smv_links(rows):
   products.append(k)
   codes={code for _cost,base in self._smv_pair_fits(j,info)for code in self._smv_series(self._smv_page(base))}
   shelf=self.home_rows(self.walk_home(k))
   for code in sorted(codes):
    tagged=[mark for mark in shelf if mark in info and code in self._smv_codes(info[mark]["title"])[1]]
    queue+=[mark for _,mark in self.cheap_unread(tagged,k)][:PAIR_OPENS]
  self._smv_few_launch(queue,products,"Reading the cheapest tagged rows of each code.")
 def fit_search(self):
  kv=self.__dict__
  queue=kv.get("fit_queue")or[]
  if not queue:
   return False
  floor=self.order_due()+2
  if not self._smv_room(floor+2,15):
   kv.pop("fit_queue",None)
   return False
  k,code=queue[0]
  job={"k":k,"schema":self._smt_search_schema()}
  fields=job["schema"].get("properties")
  job["props"]=fields if isinstance(fields,dict)else{}
  tried=kv.setdefault("fit_tried",{})
  forms=self._smv_fit_forms(k,code,job)
  todo=[text for text in forms if text not in tried.get(k,[])][:max(0,FIT_TRIES-len(tried.get(k,[])))]
  if not todo:
   queue.pop(0)
   return self.fit_search()
  text=todo[0]
  tried.setdefault(k,[]).append(text)
  args,faults=ArgCheck.execute({**job["tmpl"],job["tkey"]:text},job["schema"])
  if faults or not isinstance(args,dict):
   return self.fit_search()
  mark,tick=len(self.archive.items),self.wire.tick
  if not self.walk_call(args,"Running '%s' as a keyword query: the series code first, so the reply lists listings "
"whose tag carries it."%text):
   return self.wire.tick!=tick or self.fit_search()
  home=self.walk_home(k)
  tagged=[]
  for rec in self.archive.items[mark:]:
   walk_dicts(rec,lambda d:tagged.append(self.row_mark(d))if self._smp_is_row(d)and self.shelf_of(d)in home
and code in self._smv_codes(self.title_of(d))[1]else None,ceiling=3)
  tagged=list(dict.fromkeys(tagged))
  if not tagged and len(tried[k])<FIT_TRIES and len(todo)>1:
   self._smt_tack("[code fit] '%s': no %s row tagged %s."%(text,self.needs.items()[k]["product"],code),True)
   return True
  queue.pop(0)
  batch=self.batch_tool()
  opens=self._smt_trim([mark for _,mark in self.cheap_unread(tagged,k)][:min(batch[2]if batch else FIT_OPENS,FIT_OPENS)],
batch,floor)
  note="[code fit] '%s': %s row(s) tagged %s: %s."%(text,self.needs.items()[k]["product"],code,
self._smv_list(tagged)or"none")
  if opens:
   note+=" Unread ones opened next: %s."%self._smt_priced(opens)
  self._smt_tack(note,not opens)
  self.log.append({"kind":"fit_search","product":k,"code":code,"query":text,"tagged":tagged,"opened":opens})
  if opens:
   self.vl_launch(opens,home,floor,"Reading the code-tagged hits: each carries the series code its anchor states.")
  return True
 def _smv_fit_forms(self,k,code,job):
  home=self.walk_home(k)
  label=self.raw_label(home)
  if self._smt_vs_shape(job)!="forms":
   return[]
  tail=self._smt_label_words(k,label)if label else[w for w in self._smt_query_words(self.needs.items()[k]["product"])if len(w)>2]
  values=self._smt_values(k)
  alts=values[0][:FIT_TRIES-1]if values and values[0]else[]
  out=[]
  for words in[*alts,[]]:
   kept=list(dict.fromkeys([code,*words,*tail]))
   text=" ".join(kept[:self.lead_words]if self.lead_words else kept)
   cap_len=job.get("cap_len")
   if is_number(cap_len):
    while" "in text and len(text)>cap_len:
     text=text.rsplit(" ",1)[0]
    text=text[:int(cap_len)]
   out.append(text)
  return list(dict.fromkeys(filter(None,out)))
 def _smv_coded_anchors(self,j):
  out=[]
  for ident in self._smv_row_shelves():
   if self.product_index(ident)!=j:
    continue
   codes=self._smv_series(self._smv_shelf_page(ident))
   if codes:
    out.append("%s %s"%(ident,"/".join(codes)))
  return out[-4:]
 def cheap_sweep(self):
  phase,memo="gate",{}
  while phase!="end":
   phase=getattr(self,"_smv_sweep_"+phase)(memo)
  return memo.get("note","")
 def _smv_sweep_gate(self,memo):
  needs=self.needs
  if needs.priority()!="cheapest"or self.orders_tried or self._smv_have("cheap_swept"):
   return"end"
  finder=self.role("search")
  if not finder or not self.trolley.units or self.moves_left()<12 or self.wire.spare()<CLOSING+40:
   return"end"
  self.cheap_swept=True
  memo["tool"]=finder
  return"schema"
 def _smv_sweep_schema(self,memo):
  shape=self.shapes.get(memo["tool"])or{}
  fields=shape.get("properties")or{}
  must=set(shape.get("required")or[])
  free=[name for name,spec in fields.items()if isinstance(spec,dict)and spec.get("type")=="string"and not spec.get("enum")]
  if len(free)!=1 or must-{free[0]}:
   return"end"
  memo["field"]=free[0]
  return"words"
 def _smv_sweep_words(self,memo):
  memo["queries"]=[]
  if self.needs.items():
   return"pages"
  done=set()
  for page in self._smv_have("page_book",{}).values():
   done.add(" ".join(self.free_text(page["args"])).strip().lower())
  fresh=[q for q in self.label_queries()if" "not in q and q not in done][:2]
  if not fresh:
   return"pages"
  memo["queries"]=fresh
  return"search"
 def _smv_sweep_search(self,memo):
  tick,tool,field=self.wire.tick,memo["tool"],memo["field"]
  self.send_chunk([("cheap_%d_%s"%(n,tick),tool,{field:q})for n,q in enumerate(memo["queries"])])
  return"pages"
 def _smv_sweep_pages(self,memo):
  paging,shelves,done=self.page_arg(),self.shelf_labels(),0
  if not paging or not shelves:
   return"rows"
  for walk in list(self._smv_have("page_book",{}).values()):
   size=max(1,walk.get("size",0))
   if not set(walk.get("shelves")or())&shelves or walk.get("total",0)>4*size:
    continue
   while done<SWEEP_PAGES and self.moves_left()>=10 and self.wire.spare()>CLOSING+40:
    spot=self.next_page(walk)
    if spot is None or not self.walk_call({**walk["args"],paging[0]:spot},"Reading the rest of a short "
"result set before the order: a lower-priced fit can sit further on."):
     break
    done+=1
  memo["pages"]=done
  return"rows"
 def _smv_row_shelves(self):
  found={}
  def note(layer):
   ident,label=self._smo_record_id(layer),self.shelf_of(layer)
   if ident and label:
    found.setdefault(ident,label)
  for rec in self.archive.items:
   walk_dicts(rec,note)
  return found
 def _smv_sweep_rows(self,memo):
  cart,book=self.trolley,self.archive
  several=bool(self.needs.items())
  where=self._smv_row_shelves()if several else{}
  ceilings={}
  for line in cart.units:
   price=book.price(line)
   if price is not None:
    ceilings[where.get(line,"")if several else""]=float(price)
  if not ceilings or(several and""in ceilings):
   return"end"
  found=[]
  for ident,(cost,title,_)in self.shelf_rows().items():
   ceiling=ceilings.get(where.get(ident,"?")if several else"")
   if ceiling is not None and cost<ceiling and ident not in cart.units and ident not in book.read:
    found.append((cost,ident,title))
  found=sorted(found)[:4]
  idents=[row[1]for row in found]
  self.log.append({"kind":"cheap_sweep","queries":memo["queries"],"pages":memo.get("pages",0),"rows":idents})
  if not found:
   return"end"
  self.__dict__.setdefault("named",set()).update(idents)
  listed="; ".join("%s '%s' %g"%(ident,title[:70],cost)for cost,ident,title in found)
  memo["note"]=("%sLowest price matters here, so extra code searches went out (%s). Never-opened rows of the "
"asked-for category, available, that undercut the current cart line%s: %s. Batch-open them in "
"one reply. Should one satisfy every condition, swap it into the cart, refresh stock and price, then "
"order; otherwise order what the cart holds."%(
self._smv_YET,self._smv_list(memo["queries"])or"further pages of short result sets",
" of their own category"if several else"",listed))
  return"end"
 def bound_hint(self):
  wanted=self.needs.priority()=="cheapest"and self.wire.spare()>=CLOSING
  if not wanted:
   return""
  figures={}
  for req in self.needs.props_left():
   raw=str(req)
   _,colon,rest=raw.partition(":")
   low=(rest if colon else raw).lower()
   bound=re.search(CH_BOUND,low)is not None
   for hit in _SMW_FIGURE.finditer(low):
    if bound and"."in hit[1]:
     continue
    figures.setdefault(hit[1]+hit[2])
  picks=[*figures][:3]
  if not picks:
   return""
  return("\n[cheapest pick, limit given as a measure] The lowest-priced fit tends to print that very figure in its own "
"text. Look up "+", ".join(picks)+" next to the product word - each figure glued to its unit and also "
"with a gap - and for every such lookup page through two sets of results; then open the cheapest rows whose "
"price keeps to the budget, all before the pick is settled.")
 def _smw_cart_mark(self):
  led=self.needs
  needs=tuple(tuple(map(str,part))for part in(led.written,led.uncovered,led.let_go))
  return(self.__dict__.get("cart_edits",0),tuple(sorted(self.trolley.units.items())),needs,led.spend_cap,
led.told_version,self.refusal_n)
 def _smw_told_current(self):
  mark=self.__dict__.get("stated_at")
  return bool(mark)and bool(self.trolley.units)and mark==self._smw_cart_mark()
 def _smw_states_basket(self,said):
  cart=self.trolley.units
  if not cart:
   return False
  figures=set(re.findall(r"\d+(?:\.\d+)?",str(said).replace(",","")))
  grand,priced,named=0.0,True,True
  for line_id,count in cart.items():
   each=self.archive.price(line_id)
   if is_number(each):
    grand+=float(each)*float(count)
   else:
    priced=False
   if line_id not in said and not(is_number(each)and f"{float(each):g}"in figures):
    named=False
  return named or(priced and f"{grand:g}"in figures)
 def basket_text(self):
  try:
   known=self.code_rows()
  except Exception:
   known={}
  cart=self.trolley.units
  pieces=[]
  grand,missing=0.0,False
  for line_id in cart:
   count=cart[line_id]
   facts=known.get(line_id)or{}
   each=self.archive.price(line_id)
   each=facts.get("price")if each is None else each
   shown=int(count)if float(count).is_integer()else count
   entry=f"{shown} x {line_id}{self._smw_line_tag(line_id, facts)}"
   if each is None:
    missing=True
   else:
    grand+=float(each)*float(count)
    entry=f"{entry} at {each:g} each"
   pieces.append(entry)
  money=self.needs.reading.get("currency")or""
  money=str(money).strip()
  if not pieces:
   return"Before I finish: the basket is empty, so nothing will be ordered."
  words=["Before I finish, here is the basket: "+"; ".join(pieces)+"."]
  if not missing:
   words.append(f" The total is {grand:g}{' ' + money if money else ''}.")
  words.append(" I will place the order now.")
  return"".join(words)
 def _smw_line_tag(self,line_id,facts):
  title=facts.get("title")
  tag=f" ({title[:60]})"if title else""
  try:
   seller=self.line_seller(line_id)
  except Exception:
   seller=""
  return f"{tag} from {seller}"if seller else tag
 def finish_steps(self,kind):
  job={"kind":kind,"out":""}
  stage="gate"
  while stage is not None:
   stage=getattr(self,"_smw_fin_"+stage)(job)
  return job["out"]
 def _smw_fin_gate(self,job):
  led,mine=self.needs,self.__dict__
  job["quiet"]=not(led.told or led.exchanges or mine.get("event_seen"))
  w=self.wire
  if w.done or w.broke or self.moves_left()<2:
   return None
  job["edits"]=mine.get("cart_edits",0)
  lines_now=tuple(sorted(self.trolley.units.items()))
  mark=(job["kind"],lines_now,job["edits"],led.told_version,self.refusal_n,led.version)
  done_marks=mine.setdefault("finish_checked",set())
  if mark in done_marks:
   return None
  done_marks.add(mark)
  return"brief"
 def _smw_fin_brief(self,job):
  led=self.needs
  speaker=job["tool"]=self.role("message")
  left=0
  if speaker:
   left=max(0,self.ask_limit()-self.msgs_sent)
  job["left"]=left
  fresh_cart=job["edits"]>0 or not self.trolley.units
  since=[]
  for tick,said,at in self.__dict__.get("msg_log",[]):
   if at==job["edits"]and fresh_cart:
    since.append("- [%s] sent: %s"%(tick,said))
  job["since"]=since
  tools=[]
  for spec in self.toolset:
   fname=spec["function"]["name"]
   if fname in self.finishers()or fname==self.role("cart")or(left<=0 and fname==speaker):
    continue
   tools.append(spec)
  job["tools"]=tools
  rules=[f"- {entry['text']} (from the {entry.get('source')})"for entry in led.told]
  offer=[]
  for spec in tools:
   fn=spec["function"]
   offer.append(f"- {fn['name']}: {fn.get('description', '')} | arguments: {as_text(fn.get('parameters') or {})}")
  sections=(
("INSTRUCTIONS IN FORCE","\n".join(rules)or"(none stated)"),
("SHOPPER'S ORIGINAL WORDS",self.request),
("RECENT WORDS FROM SHOPPER OR STORE (earliest on top)","\n".join(led.exchanges[-8:])or"(none)"),
("REQUIREMENTS AT THIS POINT",led.text_body()),
("CART RIGHT NOW",self.trolley.summary()+" "+self.basket_text()),
("FINISH ABOUT TO BE SENT",job["kind"]),
("ACTIVITY SINCE THE CART'S LATEST EDIT",
"\n".join(since)or"(nothing: no message went out and no other step ran after it)"),
("MOVES LEFT",f"{left} message(s), {self.moves_left()} step(s); one step goes to the finish"),
("CALLABLE TOOLS FOR AN OPEN STEP","\n".join(offer)),
)
  job["brief"]=FINISH_CHECK+"".join(f"\n\n[{n}] {head}:\n{body}"for n,(head,body)in enumerate(sections,1))
  return"consult"
 def _smw_fin_consult(self,job):
  job["answer"]=None
  w=self.wire
  if job["quiet"]or not w.usable(FINISHER)or w.spare()<=LAST_CALL+10:
   return"collect"
  job["answer"]=w.consult(job["brief"],2500,FINISHER)
  self.finish_answer=job["answer"]
  return"collect"
 def _smw_fin_collect(self,job):
  answer,speaker=job["answer"],job["tool"]
  steps=[]
  if answer is not None:
   known={spec["function"]["name"]for spec in job["tools"]}
   wanted=answer.get("pending")or[]
   for item in wanted[:FINISH_TURNS]:
    if isinstance(item,dict)and item.get("tool")in known and isinstance(item.get("args"),dict):
     tool,given=item["tool"],dict(item["args"])
     fixed,faults=ArgCheck.execute(given,self.shapes[tool])
     if not faults:
      steps.append((tool,fixed,str(item.get("why")or"")[:120]))
  field=self.text_arg(speaker)if speaker else None
  speaks=any(tool==speaker for tool,_,_ in steps)
  if job["kind"]=="decline"and field and job["left"]>0 and not speaks and not self.free_text_arg(self.role("decline")):
   said=[text for tick,text,at in self.__dict__.get("msg_log",[])if self._smw_decline_said(text)]
   if not said:
    steps.append((speaker,{field:self.decline_remark()},"telling the shopper the errand ends without a purchase"))
  if job["kind"]=="order"and self.trolley.units and field and not job["since"]and job["left"]>0:
   if not speaks:
    steps.append((speaker,{field:self.basket_text()},"telling the shopper what the cart holds, pre-order"))
   else:
    told=[]
    for tool,args,why in steps:
     if tool==speaker and not self._smw_states_basket(str(args.get(field)or"")):
      args=dict(args)
      args[field]=(str(args.get(field)or"").strip()+" "+self.basket_text()).strip()
     told.append((tool,args,why))
    steps=sorted(told,key=lambda step:step[0]!=speaker)
  self.log.append({"kind":"finish_check","finish":job["kind"],"answer":answer,
"calls":[tool for tool,_,_ in steps]})
  room=max(0,self.moves_left()-1)
  job["steps"]=steps[:room]
  return"run"if job["steps"]else None
 @staticmethod
 def _smw_decline_said(text):
  return bool(re.search(r"not ordering|no order|nothing (?:in the store |here )?(?:meets|fits|matches)|not (?:place|placing) (?:the|an|this) order|declin",str(text or""),re.I))
 def decline_remark(self):
  wants=[]
  try:
   for req in self.needs.text_body().splitlines():
    req=req.strip(" -*")
    if":"in req and len(req)<=70:
     wants.append(req)
  except Exception:
   wants=[]
  text=SORRY
  if wants:
   text=SORRY[:-1]+" (looked for: "+"; ".join(wants[:4])+")."
  if self._sml_said("decline_no_names")is False:
   closest=self._smw_closest_line()
   if closest:
    text+=" "+closest
  scrubbed=self.scrub_decline({"text":text},self.role("decline")or"")
  return str(scrubbed.get("text")or SORRY)
 def _smw_closest_line(self):
  try:
   info=self.code_rows()
   rows=self.backup_rows()or[]
   for ident in rows[:1]:
    row=info.get(ident)or{}
    title=str(row.get("title")or"").strip()
    price=row.get("price")
    if title:
     return"The closest listing is %s%s."%(title[:80],(" at %g"%price)if is_number(price)else"")
  except Exception:
   return""
  return""
 def _smw_fin_run(self,job):
  tick=self.wire.tick
  batch=[_smw_proposal({"id":f"finish_step_{tick}_{n}","function":{"name":tool,"arguments":args}},n)
for n,(tool,args,_)in enumerate(job["steps"])]
  mark=self._smw_ledger_mark()
  prior=self.__dict__.get("scripted_batch")
  try:
   replies=self.dispatch(batch,scripted=True)
  finally:
   self.scripted_batch=prior
  job["changed"]=mark!=self._smw_ledger_mark()
  outcomes=[]
  for(tool,args,_),(_,text)in zip(job["steps"],replies):
   head=text.split("\n[status]")[0]
   outcomes.append(f"{tool} {json.dumps(args, ensure_ascii=False)[:400]} -> {head[:700]}")
  reasons="; ".join(why for _,_,why in job["steps"])
  job["note"]=f"[code steps done first: {reasons}] "+" | ".join(outcomes)
  self.log.append({"kind":"finish_steps_done","note":job["note"][:600]})
  return"report"
 def _smw_fin_report(self,job):
  speaker=job["tool"]
  asked=bool(speaker)and any(tool==speaker for tool,_,_ in job["steps"])
  if asked and(job["changed"]or self.needs.reply_due or self.wire.done):
   job["out"]=("Not run yet (no step used). "+job["note"]+"\nSomething moved in that reply. Compare it with the "
"requirements plus every standing instruction, act on what it asks, and only after that try the "
"finish a second time.")
  else:
   self.finish_note=job["note"]
  return None
 def _smw_ledger_mark(self):
  led=self.needs
  return led.version,led.told_version,self.msg_cap,self.step_cap,self.refusal_n,len(self.trolley.units)
 def dispatch(self,batch,scripted=False):
  run=_smw_Batch(batch,scripted)
  self.scripted_batch=scripted
  self.begin_batch()
  run.permitted={spec["function"]["name"]for spec in self.offered()}
  self._smw_batch_intent(batch)
  phase=_SMW_SCREEN
  while phase!=_SMW_BOOKED:
   phase=getattr(self,"_smw_batch_"+phase)(run)
  return run.replies
 def _smw_batch_intent(self,batch):
  cart_tool=self.role("cart")
  self.batch_puts=[]
  self.batch_cart=cart_tool in[_smw_fn_of(item).get("name")for item in batch if isinstance(item,dict)]
  for item in batch:
   target=self._smw_put_target(_smw_fn_of(item),cart_tool)
   if target:
    self.batch_puts.append(target)
 def _smw_put_target(self,fn,cart_tool):
  if fn.get("name")!=cart_tool:
   return None
  raw=fn.get("arguments")
  parsed=raw if isinstance(raw,dict)else{}
  if isinstance(raw,str)and raw.strip():
   parsed=json_object(raw)
  if not isinstance(parsed,dict)or not self.trolley.is_put(parsed):
   return None
  return self.trolley.aim(parsed)
 def _smw_batch_screen(self,run):
  for proposal in run.proposals:
   if run.cart_held and proposal["function"]["name"]==self.role("cart"):
    run.hold(proposal["id"],_SMW_STALE_CART)
    continue
   refusal,name,args=self.screen(proposal,run.permitted,run.scripted,len(run.proposals),run.messages)
   if refusal:
    if not run.cart_held and name==self.role("cart")and isinstance(args,dict)and args and not self.trolley.is_put(args):
     run.cart_held=True
    run.hold(proposal["id"],refusal)
    continue
   if name==self.role("message"):
    run.messages+=1
   run.queue.append(_smw_Call(proposal["id"],name,args))
  return _SMW_FINISH if run.queue and run.queue[0].name in self.finishers()else _SMW_TRIM
 def _smw_batch_finish(self,run):
  lead=run.queue[0]
  held,lead.args=self.screen_finish(lead.cid,lead.name,lead.args)
  if held:
   run.hold(lead.cid,held)
   return _SMW_BOOKED
  return _SMW_TRIM
 def _smw_batch_trim(self,run):
  if run.queue and run.queue[0].name not in self.finishers()and self.finishers():
   room=max(0,self.moves_left()-1)
   for late in run.queue[room:]:
    late.state="trimmed"
    run.hold(late.cid,_SMW_LAST_STEP)
  return _SMW_SEND
 def _smw_batch_send(self,run):
  live=[call for call in run.queue if call.state=="queued"]
  for start in range(0,len(live),16):
   slab=live[start:start+16]
   if self.wire.done or self.wire.broke or self.moves_left()<=0:
    for call in slab:
     call.state="over"
     run.hold(call.cid,_SMW_OVER)
    continue
   slab=slab[:self.moves_left()]
   for call in slab:
    call.state="sent"
   run.replies+=self.send_chunk([(call.cid,call.name,call.args)for call in slab])
  return _SMW_LABEL
 def _smw_batch_label(self,run):
  note=self.__dict__.pop("finish_note","")
  kept=[call for call in run.queue if call.state!="trimmed"]
  lead=kept[0]if kept else None
  if note and lead is not None and lead.name in self.finishers():
   run.replies=[(cid,f"{note}\n{text}")if cid==lead.cid else(cid,text)for cid,text in run.replies]
  rank={}
  for n,proposal in enumerate(run.proposals):
   rank[proposal["id"]]=n
  run.replies.sort(key=lambda pair:rank.get(pair[0],0))
  return _SMW_BOOKED
 def send_chunk(self,slab):
  if len(slab)>1:
   return[reply for call in slab for reply in self.send_chunk([call])]
  acts=[dict(name=name,args=args)for _,name,args in slab]
  number=self.wire.tick+1
  answer=self.wire.act(acts)
  self.log.append(dict(kind="env",step=number,actions=acts,reply=answer))
  if answer is None:
   return[(cid,"No answer came back for this call (a network or server fault); it can be sent again.")for cid,*_ in slab]
  many=answer.get("observations")
  views=many if isinstance(many,list)else[answer.get("observation")]
  tail=len(views)-1
  out=[]
  for n,(cid,name,args)in enumerate(slab):
   if n>tail:
    out.append((cid,"Not run: the store closed up shop while earlier calls of this step were answered, and never got to this one."))
   else:
    out.append((cid,self.absorb(name,args,views[n],answer,n==tail)))
  return out
 def absorb(self,name,args,obs,resp,last):
  bk=_smw_Booking(name,args,obs,resp,last,self.role("message"),self.role("cart"))
  for stage in _SMW_BOOK_STAGES:
   getattr(self,stage)(bk)
  cap=SEARCH_TEXT if name==self.role("search")else RECORD_TEXT
  shown=as_text(clipped(bk.shown,cap))
  if last and resp.get("finished"):
   shown+=" [episode over: nothing more can be sent]"
  said=" ".join(r for r in bk.remarks if r)
  return f"{shown}\n[note] {said}"if said else shown
 def _smw_bump(self,attr):
  setattr(self,attr,self.__dict__.get(attr,0)+1)
 def _smw_try(self,tag,work,fallback=None):
  try:
   return work()
  except Exception as err:
   self.log.append({"kind":tag,"error":_smw_fault(err,300)})
   return fallback
 def _smw_book_counts(self,bk):
  mine=self.__dict__
  if bk.name==bk.speaker:
   self.msgs_sent+=1
   said=_smw_said(bk.args)
   entry=(self.wire.tick,said[:600],mine.get("cart_edits",0))
   mine.setdefault("msg_log",[]).append(entry)
   if self._smw_states_basket(said):
    self.stated_at=self._smw_cart_mark()
   if not mine.get("scripted_batch"):
    self.questions_log.append("- (the assistant's own message): \"%s\" -> %s"%(said[:300],as_text(bk.obs)[:300]))
  if bk.name==bk.cart:
   self._smw_bump("cart_edits")
  kind=self.last_kind=self.call_kind(bk.name,bk.args)
  if kind=="order":
   self.orders_tried+=1
  if bk.name in(bk.cart,bk.speaker)or bk.name in self.finishers():
   self.last_calls=[]
  else:
   self.last_calls=self.last_calls[-1:]+[bk.name+as_text(bk.args)]
  self.archive.add(bk.obs)
  self._smw_bump("absorb_n")
  if bk.name!=bk.cart and not mine.get("covering"):
   self._smw_bump("since_cart")
  bk.gone=self._smw_try("caps_error",lambda:self.learn_caps(bk.name,bk.args,bk.obs),[])
  self._smw_try("sheet_role_error",self.sheet_role_check)
 def _smw_book_lines(self,bk):
  cart=self.trolley.units
  if not cart:
   return
  mine=self.__dict__
  fresh=set()
  needs_stock=bool(mine.get("stock_seen"))
  named=self.asked_ids(bk.args)
  trusted=bool(mine.get("rows_live"))and not mine.get("rows_lag")
  lookup_only=not trusted and bk.name not in(bk.cart,self.role("cart_view"))
  def look(rec):
   on_cart=[(key,val)for key,val in rec.items()if isinstance(val,str)and val in cart]
   by_id={val for key,val in on_cart if"id"in str(key).lower()}
   if not by_id:
    return
   if needs_stock:
    ok=self.stock_state(rec)is not None and not(lookup_only and by_id.isdisjoint(named))
   else:
    ok=any(isinstance(val,dict)for val in rec.values())
   if ok:
    fresh.update(val for _,val in on_cart)
  walk_dicts(bk.obs,look)
  for line_id in fresh:
   mine.setdefault("line_read",{})[line_id]=self.wire.tick
   mine.setdefault("line_seq",{})[line_id]=self.absorb_n
 def _smw_book_reads(self,bk):
  failed=isinstance(bk.obs,dict)and bk.obs.get("error")
  if bk.name==self.role("details")and not failed:
   store=self.archive
   store.read.update([val for val in bk.args.values()if isinstance(val,str)and val in store.ids])
  if self._smw_late_read(bk.name,bk.args):
   self._smw_bump("late_n")
  if bk.gone:
   bk.remarks.append("[stock] "+self.dead_note(bk.gone))
  moved=self._smw_try("notes_remark_error",lambda:self.notes_remark(bk.name,bk.args))
  if moved:
   bk.remarks.append("[notes] "+moved)
 def _smw_late_read(self,name,args):
  details,search=self.role("details"),self.role("search")
  other_lookup=bool(details)and name not in(details,search)and name in self.lookup_tools()
  texts={val for val in args.values()if isinstance(val,str)}
  if self.moves_left()>=self.order_due()or not(name in(details,search)or other_lookup):
   return False
  if self.__dict__.get("covering"):
   return False
  held=set(self.trolley.units)
  if name==details and texts&held:
   return False
  if not other_lookup:
   return True
  return not texts&(held|{key.rpartition("|")[0]for key in self.passed})
 def _smw_book_cart(self,bk):
  if bk.name!=bk.cart:
   return
  book=self.trolley
  self.order_vetted=False
  book.changed_at=self.wire.tick
  before=dict(book.units)
  bk.remarks.append(book.booked(bk.args,bk.obs))
  self.last_cart_sig=bk.name+as_text(bk.args)
  def follow():
   self.cart_moved(before)
   self.mark_fresh(bk.obs)
   self.after_put(bk.args,bk.obs)
  self._smw_try("after_put_error",follow)
  book=self.trolley
  item=book.aim(bk.args)
  if not item or not book.is_put(bk.args):
   return
  self.refusals_seen[item]=self.refusal_n
  tally=self.__dict__.setdefault("puts",{})
  tally[item]=tally.get(item,0)+1
  units=book.count_arg(bk.args)
  if not book.is_put(bk.args)or f"{item}|{self.needs.version}"not in self.passed:
   return
  if(units is None or units>0)and book.held(item)>0 and item not in self.dead_lines():
   bk.remarks.append(f"Keep {item} in the cart unless something later excludes it: an outside check has tested it on "
"every current condition under the value rules (a value the store writes in a field of its own "
"is met only by that field's words).")
 def _smw_book_view(self,bk):
  if bk.name==self.role("cart_view"):
   self.trolley.viewed(bk.obs,self.wire.tick)
   bk.remarks.append(self.trolley.summary())
   self.last_cart_sig=None
  if bk.name in(self.role("cart_view"),self.role("cart"))and not self.__dict__.get("count_told"):
   key=self.trolley.count_key()
   if key and self.trolley.adds is not None:
    self.count_told=True
    bk.remarks.append("[cart tool] its integer argument %s %s; one line per product at its quantity, and an identical call "
"repeated changes nothing."%(key,"adds that many units to the line"if self.trolley.adds else"sets the units held on that line (2 means two units)"))
  bk.refused=self.refused(bk.name,bk.obs,bk.resp if bk.last else{"finished":False})
  if bk.refused and self.last_kind=="order":
   self.cart_seq=self.absorb_n
  bk.remarks.append(self.notes_for(bk.name,bk.obs,bk.refused))
 def _smw_book_search(self,bk):
  if bk.name!=self.role("search"):
   return
  bk.remarks.append(self.search_note(bk.args,bk.obs))
  pending=self.__dict__.get("named_pages",set())
  pending.discard(json.dumps(bk.args,sort_keys=True,default=str))
  joined=self._smw_try("merge_joined_error",lambda:self.merge_joined(bk.obs),_SMW_NOTHING)
  if joined is not _SMW_NOTHING:
   bk.remarks.append(joined)
 def _smw_book_audit(self,bk):
  if not bk.last or self.__dict__.get("early_audit"):
   return
  if bk.name in(bk.cart,bk.speaker)or bk.name in self.finishers():
   return
  if not FIRST_ORDER_BY<self.moves_left()<=self.step_cap//3+2 or self.trolley.units:
   return
  now=str(self.needs.version)
  if any(key.rpartition("|")[2]==now for key in self.passed):
   return
  self.early_audit=True
  ahead=self.review_decline(early=True)
  if ahead:
   bk.remarks.append("No checked fit so far. "+ahead)
 def _smw_book_shopper(self,bk):
  question=_smw_said(bk.args)if bk.name==bk.speaker else None
  caught=self.shopper_lines(bk.name,bk.obs,bk.refused)
  if not caught:
   return
  led=self.needs
  version,loose,cap=led.version,len(led.let_go),led.spend_cap
  bk.remarks.append(led.hear(caught,question))
  self.ref_take()
  news=led.__dict__.get("allowance_news")
  if news:
   led.allowance_news=None
   self._smw_try("allowance_error",lambda:self.apply_allowance(news))
  if led.told_version!=self.__dict__.get("told_seen"):
   self.told_seen=led.told_version
   self.order_vetted=False
  if led.version<=version:
   return
  if question is not None and len(led.let_go)<=loose and led.spend_cap==cap:
   return
  review=self._smw_try("change_review_error",self.change_review)
  if review:
   bk.remarks.append(review)
 def tool_calls(self,msg,seq_no):
  out=[]
  for raw in msg.get("tool_calls")or[]:
   if isinstance(raw,dict):
    seq_no+=1
    out.append(_smw_proposal(raw,seq_no))
  return out,seq_no
 def execute(self):
  loop=_smw_Loop()
  handlers={"open":self._smw_on_open,"turn":self._smw_on_turn,"code":self._smw_on_code,
"model":self._smw_on_model,"missed":self._smw_on_missed,"reply":self._smw_on_reply,
"idle":self._smw_on_idle,"calls":self._smw_on_calls,"close":self._smw_on_close}
  phase="open"
  while phase!="done":
   phase=handlers[phase](loop)
  return self.log
 def _smw_on_open(self,loop):
  if self.wire.done:
   return"done"
  self.read_request()
  self.log.append(dict(kind="plan",plan=self.needs.reading,roles=self.role_map))
  parts=["SHOPPER REQUEST:\n"+self.request]
  first=self.pview.get("opening")
  if first is not None:
   parts.append("OPENING OBSERVATION:\n"+as_text(first))
   self.archive.add(first)
  brief=self.needs.overview(self.step_cap,self.msg_cap)
  if brief:
   parts.append(brief)
  parts.append(self.status_line())
  self.chat=[dict(role="system",content=ACT_RULES),dict(role="user",content="\n\n".join(parts))]
  loop.turn_ok=monotonic()
  return"turn"
 def _smw_on_turn(self,loop):
  if loop.turns>=TURN_LIMIT:
   return"close"
  loop.turns+=1
  w=self.wire
  if w.done or w.broke or self.moves_left()<=0 or w.spare()<LAST_CALL:
   return"close"
  return"code"
 def _smw_on_code(self,loop):
  self.catch_up()
  if self.needs.reply_due:
   due,self.needs.reply_due=self.needs.reply_due,""
   if self.reply_back(due):
    return"turn"
  for move,guard in _SMW_CODE_MOVES:
   if(guard is None or getattr(self,guard)())and getattr(self,move)():
    return"turn"
  return"model"
 def _smw_may_sweep(self):
  return not self.asking or bool(self.__dict__.get("empty_plans"))
 def _smw_is_asking(self):
  return bool(self.asking)
 def _smw_on_model(self,loop):
  loop.models=[*ACTOR[1:],*ACTOR[:1]]if self.stuck>=STUCK_LIMIT else ACTOR
  loop.msg=self.wire.query_model(self.chat,loop.models,2500,self.toolset)
  return"missed"if loop.msg is None else"reply"
 def _smw_on_missed(self,loop):
  loop.missed+=1
  if self._smw_models_gone(loop):
   return"close"
  if self.wire.call_fail not in("empty","refused"):
   return"turn"
  loop.fails+=1
  return"close"if loop.fails>=2 else"turn"
 def _smw_models_gone(self,loop):
  w=self.wire
  if not w.usable(loop.models,True):
   return True
  if monotonic()-loop.turn_ok<=DOWN_AFTER:
   return False
  return loop.missed>1 or(w.down()and len(w.call_models)>1)
 def _smw_on_reply(self,loop):
  loop.turn_ok,loop.missed=monotonic(),0
  loop.calls,loop.seq=self.tool_calls(loop.msg,loop.seq)
  said=loop.msg.get("content")or""
  loop.text=said if isinstance(said,str)else as_text(said)
  binding={self.role("cart"),self.role("order"),self.role("decline")}-{None}
  if self.asking and any(call["function"]["name"]in binding for call in loop.calls)and self.pose_planned(urge=True):
   return"turn"
  return"calls"if loop.calls else"idle"
 def _smw_on_idle(self,loop):
  loop.idle+=1
  self.chat.append(dict(role="assistant",content=loop.text or"(no tool call)"))
  self.chat.append(dict(role="user",content="That reply called no tool. A turn without a tool call "
"does nothing; pick one of those given and carry on.\n"+self.status_line()))
  return"close"if loop.idle>=IDLE_LIMIT else"turn"
 def _smw_on_calls(self,loop):
  loop.idle=0
  self.log.append(dict(kind="model",content=loop.text[:2000],calls=[call["function"]for call in loop.calls]))
  self.chat.append(dict(role="assistant",content=loop.text or None,tool_calls=loop.calls))
  before=self.wire.tick
  self.post_replies(self.dispatch(loop.calls))
  self.stuck=0 if self.wire.tick!=before else self.stuck+1
  return"turn"
 def _smw_on_close(self,loop):
  w=self.wire
  if not(w.done or w.broke):
   self.finish_now()
  self.log.append(dict(kind="end",finished=w.done,steps=w.tick,cost=round(w.spend,5),
seconds=round(monotonic()-w.t0,1)))
  return"done"
_SMW_FIGURE=re.compile(r"(\d+(?:\.\d+)?)\s?([a-z\"']{1,4})\b")
_SMW_NOTHING=object()
_SMW_SCREEN,_SMW_FINISH,_SMW_TRIM,_SMW_SEND,_SMW_LABEL,_SMW_BOOKED="screen","finish","trim","send","label","booked"
_SMW_STALE_CART=("Not run, and no step was spent: this reply changed the cart earlier and that change was held back, which "
"leaves this one out of date; read the reason given for it and resend the cart changes.")
_SMW_LAST_STEP="Not run, and no step was spent: just one step remains, kept for the order or the decline alone."
_SMW_OVER="Not run, since sending is over: the step budget is gone or the store ended the episode."
_SMW_BOOK_STAGES=("_smw_book_counts","_smw_book_lines","_smw_book_reads","_smw_book_cart","_smw_book_view",
"_smw_book_search","_smw_book_audit","_smw_book_shopper")
_SMW_CODE_MOVES=(("open_named",None),("fit_search",None),("wide_step",None),("sweep_step","_smw_may_sweep"),
("value_pages",None),
("pose_planned","_smw_is_asking"),("ref_step",None),("reveal_reach",None),("cap_reach",None),
("solo_reach",None),("few_reach",None),
("walk_step",None))
class _smw_Call:
 __slots__=("cid","name","args","state")
 def __init__(self,cid,name,args):
  self.cid,self.name,self.args,self.state=cid,name,args,"queued"
class _smw_Batch:
 def __init__(self,proposals,scripted):
  self.proposals,self.scripted=proposals,scripted
  self.permitted:set[str]=set()
  self.replies:list[tuple[str,str]]=[]
  self.queue:list[_smw_Call]=[]
  self.messages=0
  self.cart_held=False
 def hold(self,cid,text):
  self.replies.append((cid,text))
class _smw_Booking:
 def __init__(self,name,args,obs,resp,last,speaker,
cart):
  self.name,self.args,self.resp,self.last=name,args,resp,last
  self.shown,self.obs=obs,lensed(obs)
  self.speaker,self.cart=speaker,cart
  self.gone:list[str]=[]
  self.remarks:list[str]=[]
  self.refused=False
class _smw_Loop:
 def __init__(self):
  self.turns=self.idle=self.fails=self.seq=self.missed=0
  self.turn_ok=0.0
  self.models:list[str]=ACTOR
  self.msg:dict={}
  self.calls:list[dict]=[]
  self.text=""
def _smw_said(args):
 return" ".join([val for val in args.values()if isinstance(val,str)])
def _smw_fn_of(call):
 return call.get("function")or{}
def _smw_fault(err,cap):
 return f"{type(err).__name__}: {err}"[:cap]
def _smw_proposal(raw,n):
 fn=raw.get("function")or{}
 args=fn.get("arguments")
 if not isinstance(args,str):
  args=json.dumps(args or{})
 return{"id":raw.get("id")or f"call_{n}","type":"function","function":{"name":fn.get("name"),"arguments":args}}
def agent_main(pdata):
 job=Errand(pdata)
 try:
  return job.execute()
 except Exception as err:
  job.log.append({"kind":"crash","error":_smw_fault(err,500)})
 try:
  if not(job.wire.done or job.wire.broke):
   job.finish_now()
 except Exception as err:
  job.log.append({"kind":"crash","error":_smw_fault(err,500)})
 return job.log
__all__=["agent_main"]
