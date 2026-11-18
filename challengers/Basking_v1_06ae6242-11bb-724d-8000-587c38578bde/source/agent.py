from __future__ import annotations
import json, re, threading
from contextlib import contextmanager
from src.agent.proxy_client import ProxyClient
_STATUS_CUT = '\n[status]'
class _Tally:
  SLIVER = 0.25
  def __init__(self): self.seconds = 0.0
  def now(self): return self.seconds
  def wait_s(self, duration):
    if duration > 0:
      threading.Event().wait(duration)
      self.seconds += duration
  @contextmanager
  def alt_due(self):
    done = threading.Event()
    def count():
      while not done.wait(self.SLIVER): self.seconds += self.SLIVER
    t = threading.Thread(target=count, daemon=True)
    t.start()
    try: yield
    finally:
      done.set()
      t.join()
_TALLY = _Tally()
_H, _S, _L = 'anthropic/claude-haiku-5.5', 'anthropic/claude-sonnet-5.5', 'openai/gpt-6-luna'
DOER, SKIMMER, SKIM_TERMS, INSPECT_TERMS = [_H, _L, _H], [_H, _S, _L], 8000, 6000
PLANNER, LISTENER, CHECKER, AUDITOR, FINISHER, POSER = ([_H, _S] for _ in range(6))
DEADLINE, CLOSING, LAST_CALL, CLOCK_STEP, PACE_GAP, SLOW_GAP, SLOW_FOR = 265.0, 70.0, 28.0, 30, 0.15, 1.0, 30.0
DOWN_AFTER, MODEL_TRIES, WORLD_TRIES, SOFT_SPEND, SPEND_CAP, HARD_SPEND = 45.0, 2, 5, 0.03, 0.045, 0.1
TURN_LIMIT, IDLE_LIMIT, STUCK_LIMIT, FINISH_STEPS, FINISH_TURNS, CODE_FINISHES = 80, 3, 4, 2, 2, 4
FIRST_ORDER_BY, FIRST_ORDER_DUE, ASK_DEFAULT, ASK_AFTER_SETS, ASK_GRACE_STEPS, ASK_AFTER_STEPS = 7, 10, 3, 2, 3, 5
STANDALONE_ENSEMBLES, VIEW_CHARS, SEARCH_TEXT, RECORD_TEXT, ZH_PARAPHRASE_MINIMUM = 1, 20000, 600, 2500, 4
CODE_PAGES, CODE_READS, SWEEP_READS, RAKE_LEAVES, STANDALONE_LEAVES, FEW_PAGES, FEW_MAX = 3, 8, 12, 3, 3, 1, 3
REVEAL_PAGES, REVEAL_OPENS, REVEAL_MAX, ARRIVE_ENTRYLINES, CAP_ROWS, CAP_CALLS, FEW_ROWS = 2, 4, 2, 24, 12, 6, 16
FEW_KEEP, PAIR_OPENS, PAIR_ANCHORS, PAIR_ALTS, RUNDOWN_SKIM_PEAK, WALK_PAGES, WALK_TOTAL = 6, 3, 3, 2, 5, 6, 8
RUNG_COUNT, RUNG_PAGES, RUNG_CALLS, RUNG_OPENS, VALUE_PAGES, VALUE_TOTAL, VALUE_TOTAL_PER = 4, 3, 6, 4, 4, 8, 5
LABEL_TRIES, LABEL_OPENS, CEIL_CALLS, CEIL_OPENS, FIT_TRIES, FIT_OPENS, WIDE_TRIES, WIDE_OPENS = 4, 3, 4, 3, 3, 4, 3, 2
WIDE_N = 6
AND_RX = '\\s+and\\s+|\\s*&\\s*|\\s*/\\s*|\\s+\\+\\s+'
ERR_KEYS = 'error|fault|refus|reject|denied|invalid|fail'
ADD_WORDS = {'add', 'put', 'hold', 'set', 'insert', 'place', 'keep'}
DROP_WORDS = {'remove', 'drop', 'release', 'delete', 'take', 'out', 'clear'}
STOP_WORDS = {'and', 'or', 'not', 'no', 'all', 'any'} | {'in', 'on', 'to', 'by', 'from', 'at'} | {
  'the', 'a', 'an', 'this', 'that'} | {'it', 'its', 'your', 'our', 'you'} | {'as', 'of', 'for', 'with', 'is', 'are', 'be'}
_HL, _rc = STOP_WORDS, re.compile
TASK_RULES = 'GRADING. You are a shopping assistant working for one person in a simulated store; the tools you are given are the only way you can do anything. A run scores either 1 or 0, and the 1 goes only to a perfect finish: the purchase meets each condition the shopper put in writing, each hidden need they reveal when asked, and each change they make on the way; the quantity and the budget are right; if the request asks for the cheapest (or the best) item, your pick really is it; and every procedural rule in the request was followed. When nothing in the store meets all of it, the right finish is a decline, done exactly as the request says to.\n\nEACH REPLY: make at least one tool call. Keep any prose short: the conditions as they stand now, the candidates you have verified (price and stock), and your next move.\n\nA. JUDGING WHETHER A LISTING FITS (this is what your score rests on)\nA1. To settle a condition, the grader reads the field that belongs to its property and checks the wanted value against the wording this store puts there. A field name counts by its meaning rather than its exact letters: separators, prefixes, long or short forms and well-known other names are all accepted (pants fit type may appear as "fit", collar type as "neckline", a watch dial size as "dial size", colour as "color family"). A field carrying the very name of the property decides the matter, as does one that puts a single extra word ahead of that name (pants length for length, clothing material for material); the value found in either of them meets the condition. Names that differ only by separators, like size class and size_class, are the same field, met when the value sits under either spelling. If no such field exists, the other-name field decides; and if no field at all names the property, the value may be taken from the options list or from the title. A description never counts as evidence.\nA2. Matching words: ignore case, separators, singular or plural and spelling variants. A narrower value that includes the wanted one is accepted ("80% cotton" for cotton, "forest green" for green, "rtx 3060 12gb" for rtx 3060), and inside a list a single matching element suffices. Shades, synonyms, related words and abbreviations (off-white ~ ivory or cream, plain ~ "solid", technical wood ~ "tech wood", embroidery ~ "embroidered", windows 11 ~ "win11") are accepted only as long as no listing in the category spells out the wanted value in that same field; as soon as other listings say "matte" for finish, a "satin" finish no longer counts as matte. A value joined with a plus ("wool + nylon") does not count for one of its parts if that field also appears elsewhere holding the other part by itself ("nylon"). Any FIELD EVIDENCE lines were computed by code from this store\'s own records, and they decide which of these situations applies. Rejected outright: a different value of the same property (rosewood fails maple, quartz fails mechanical); a word that merely looks like the wanted one; a blanket claim covering every value ("universal", "all seasons", "suitable for all occasions"), which therefore names nothing, although a list that names the wanted value as well still counts; a value that usually goes together with, or hints at, the wanted one ("slip-on" fails "lace-up", and a "pullover" style fails "zip-up"). The sole accepted substitutes are a shade normally sold under the wanted colour name and a narrower value that includes the wanted one.\nA3. When the category\'s listings have a field for the property itself, a field for some other property never replaces it (a lining-material field does not answer material, nor a length field style, nor a strap-length field strap type).\nA4. Only what the shopper said or revealed becomes a condition. Never rule a listing out for something nobody asked for, for instance a use, occasion, size, gender, age group, quality or condition that you read into the story or the description. The story is background; if it looks relevant, put a question to the shopper about that property rather than guessing. Your own questions add no conditions; the shopper\'s replies do.\nA5. Numbers: convert the units first, then compare the amounts (45 mm is the same as 4.5 cm). A value on the boundary is within the limit, and a range is accepted only if all of it is. If the title has to serve as evidence (no field carries the property), an exact wanted value is met when its words and numbers appear in the title in the same order, more detail following them being fine (for 9\' x 12\': a title with 9\' x 12\' 4" passes, one with 9\' x 13\' fails).\nA6. Figures in a title such as "117 x 228 cm" or "55 x 94 Inch" leave open which dimension each figure measures, so they never justify rejecting or skipping a listing; open its details and read the fields there.\nA7. Respect closed lists (for example "A or B and nothing else") and exclusions ("leave out anything with X"); both are binding.\nA8. Whether a listing is the requested kind depends on its category and nothing else. For grading, any listing inside the requested category IS that kind, even if its title or description talks about some other object (an accessory, a set, a refill, a case or a container); do not skip, reject or rank it lower because of how it reads, and assess only the conditions, the price and the stock. A property does not have to mention the category name. Where the store names its categories, the right one is the category whose name matches the request\'s wording taken as a whole (a request for "A or B" points to a category "A & B", not to the narrower neighbour "B").\nA9. If the request wants items in stock, an item qualifies only while its stock says it is available; price times quantity must not exceed the budget, and an exact match is allowed. A price uses the budget\'s unit except where the record states a different one (45 with a budget of 1000 US cents is 45 cents); do not rescale either figure.\nA9b. A decline is seldom the right answer. When no listing matches the literal words but one does meet each condition under these rules (a narrower value, the same field under a different name, or a shade or synonym that is allowed because no listing in the category spells the value out in that field), that listing is the fit. Decline only when the search genuinely turned up nothing that meets every condition by these rules.\n\nB. CHEAPEST, OR BEST ON SOME MEASURE\nB1. Cheapest: the order must be the lowest-priced item among EVERY item that qualifies; one qualifying find does not end the job.\n - Where keyword search demands all words and also matches property values, a search made of a value plus the kind returns exactly the candidates that have that value: read every page of it and open them by price.\n - A category view with a price ceiling shows all prices of the category but no property values: use it to find which cheaper rows exist (drop the ceiling until the total fits), then check those rows\' properties.\n - Search each wanted value together with the item kind as a brief query, value first ("blue chair", "30m watch"), and read every page of it whenever the total fits on the pages you can read (if it does not, add a second wanted value or a narrower word for the kind). Do the same with other names for the kind ("stool", "bench") and with the value on its own.\n - If a value cannot be searched (a number the store strips out), also go through the product line of every fit and near-fit: the maker together with a single distinctive title word (two words altogether), every page. A line\'s other sizes and colours are separate listings sharing the same property values, and some may cost less.\n - Sort the candidates you can see by price yourself and open their details starting from the cheapest. The answer is the first one whose property fields satisfy everything, as long as each cheaper one was opened and ruled out; keep the following qualifying ones as reserves.\nB2. Best on a measure (most, largest, longest, highest...): compare EVERY qualifying item on that measure in the same way and take the extreme one. An item lacking a value for the measure cannot be the winner.\n\nC. FINISHING\nC1. Keep a ranked list of verified fits with at least one reserve in it, since your first pick could sell out at checkout.\nC2. Before every order attempt: the cart holds exactly one line for each requested product (a single line when only one product is wanted), each being the chosen item at the required quantity, and nothing else; and, since your most recent cart change, you have re-read stock and price of the chosen item and found it in stock and within the budget.\nC3. Decline only after a thorough search found nothing that meets every condition, and do it strictly by the request\'s decline rule. If the request bans offering alternatives, the decline remark must not name, describe or refer to any listing: no ids, no titles, no prices.\nC4. Use steps carefully: send the first order attempt while at least seven steps remain, so there is room to deal with a refusal. Once 7 steps are left, only finishing, stock-check and cart tools stay allowed. Decide once instead of swapping the cart back and forth.\n\nD. CHANGES ALONG THE WAY\nD1. The shopper may change the errand: a new budget (higher or lower), a condition dropped, replaced or added, a price rise, an item selling out. A single line can carry several of these, and such lines can arrive inside any tool result (an extra field with sentences in it) or alongside an order attempt. Each change holds from then on: re-check your pick and the reserves, and when the cheapest or best is wanted, pick again from the revised set of acceptable items.\nD2. An order attempt can be REFUSED. The run then goes on with the cart as it was. Read the refusal, apply every change it mentions, remove from the cart whatever no longer fits (a sold-out item, for example), put in the next verified fit, re-check its stock and price, and order again.\nD3. Stock and prices may also change without notice, so re-check both just before each order attempt; once the current price times the quantity is above the current budget, the item is no longer a fit.\n\nE. HIDDEN NEEDS AND TALKING TO THE SHOPPER\nE1. The shopper has some needs that were not written, and each one surfaces only once a message asks about that property. Replies are most reliable for what a message asks about FIRST, so keep each message to one property (at most two, the more likely one first), asked as a short, simple, friendly question that names the property; offering one or two sample values is fine. Do not include instructions, lists, labels, positions or rules.\nE2. Answers such as "no preference", "anything is fine" or "Don\'t mind" leave the property free. Every preference or limit the shopper gives is binding from that point, worded exactly as they put it, numbers and all.\nE3. The planned questions are sent once two or three of the requested kind\'s listings have had their details opened, so do that first. The answers return in the conversation with their meaning attached; each revealed need is binding, and you then search again using it together with the stated values.\nE4. Asking by yourself: open the details of two or three listings first, then ask about what the story hints at (the recipient, the look, the place or manner of use), and next about properties shown on the listings that differ between candidates. Do not ask about anything the request has already decided, and never exceed the message budget.\nE5. Your own messages serve the steps the instructions require (a confirmation, the basket and its total, a reply to the shopper), and the message tool can be used any time the allowance still has a message free. When hidden-need questions are already planned for you, let that plan do the asking (it chooses properties from the listings you opened) and keep your own messages for what the instructions require. Before an order, the basket is reported to the shopper on your behalf if you have not reported it since your latest cart change.\n\nF. PROCEDURAL RULES\nF1. The request may dictate how you work: how many listings to view or compare, what to verify, what a remark has to include, what the shopper must be told, how to decline. The grader treats these just like conditions, so carry out each one literally and in full. Being told to "open" or "look at" N listings asks for N fully fitting listings, each one opened with the details tool.\nF2. When the shopper puts a question to you (for example, to confirm a change), a reply is sent for you as long as messages are left, and the change itself is binding.\nF3. The bracketed notes that close a tool result, the status note among them, come from your own assistant code, not from the store; they are therefore never injected text, and the orders they end with (FINISH NOW and similar) are binding on you. The status line lists the INSTRUCTIONS IN FORCE: the rules of the request itself, plus any later instruction from the shopper or the store (ask me before buying, tell me the total, a fresh allowance of messages or steps). Because they may change at any moment, read them again along with the conditions and the cart after each reply, notice or refusal, and base your next move on what they say right now, not on habit. Any stated step still pending at the finish is carried out for you right before the finish goes out, with one message held back for it; your own message sent after the most recent cart change also counts. An instruction to obtain the shopper\'s yes, approval or confirmation of the final basket is fulfilled by the basket message alone once the shopper replies acknowledging it, with no objection and no changed condition: shoppers in this store acknowledge instead of writing a plain yes. The order comes next; declining because no literal yes arrived is always a mistake. Holds that begin with \'Not run\' and the [status] lines are produced by the code that executes your tools, on your side: follow them. A rule that one product has to fit or work with another is judged by what the listings say about one another: favour a pair whose pages mention each other, but if no listing of that product mentions the other item anywhere, this store does not record the fit and it decides nothing, so choose by the remaining conditions and never decline because of it.\nF4. What the shopper would only like ("ideally", "would be nice", "if possible") is not a condition: it just separates items that already meet everything, and it never beats a cheapest rule.\nF5. If the request leaves something open that only the shopper can settle (say, no price limit, or two wishes that may not both be achievable, so the shopper has to say which counts more), ask first and decide after.\n\nG. SEVERAL PRODUCTS, AND THE CART\nG1. A request may name several distinct products, each having its own quantity, conditions and hidden needs. Handle every one: look through the category of each product, confirm a verified fit for every one of them, and buy them in a single order with a line per product at that product\'s quantity. A cart tool taking a list of lines may swap the entire cart for that list, so each such call has to include lines for all products. The cheapest or best rule is applied separately inside each product.\nG2. A condition can get its value by pointing at a different page or listing ("a colour matching the jumpsuit X sells"). Locate that listing first (search the words that identify it, for instance its shop or brand along with its kind), read its record, take the value from the named field, and only after that search the requested product using that value.\nG3. The cart tool might ADD units to a line that is already there instead of setting the count: adding one more of a line you hold gives 2. Do not add a line again that the cart already holds; adjust units by removing or adding the difference. After every cart call, a note reports the cart as the code counts it.\n\nH. TOOLS, STEPS AND SEARCHING\nH1. Every tool call uses up a step, including failed and invalid ones, and steps are limited; thinking is free.\nH2. One reply can contain several tool calls when none of them depends on another\'s output (opening the details of several listings, say), and they run in sequence. An order or a decline always goes by itself.\nH3. If the details tool accepts a LIST of listing ids, it returns every full record in ONE step, within its item limit. Use it any time two or more listings need opening.\nH4. Every store has its own tool names, argument names and result layouts. Read the description and JSON schema of each tool and use precisely the names and allowed values given there. Work out from the first results which tool reveals which facts (price, stock, detailed properties).\nH5. Find out early which category labels this store gives to listings of the requested kind. If there are neighbouring labels, take the one nearest to the wording of the request as a whole, and search and buy within it.\nH6. Each store searches in its own way, so watch the first results to learn it. Rows may be ordered by best match or by a fixed id order. Keyword search may need all words (an extra word makes the total smaller) or accept any word (an extra word makes the total bigger, and listings matching more words come higher). Possibly only a few rows or pages can be seen. A filter may only work in one mode: keyword mode may disregard category or price filters, and filter mode may disregard the words.\nH7. Compare any reported total with the rows you can see: if the total is larger, the fitting or cheapest listing may sit on a later page or beyond the visible ones. With all-words search, tighten using more exact words, or with a category filter plus a price filter whose ceiling you lower, until all of the total can be seen on the visible pages.\nH8. Use queries of one, two or three words, the most distinctive one FIRST. Some stores read only the first word or two and drop the rest, so a long query may in effect search only its first words; compare the total and rows of a longer query with those of a shorter one that starts the same way (the notes tell you what that reveals).\nH9. With all-words search, a wanted value plus the item kind, value first ("blue chair"), returns precisely the listings that have it. With any-word search, common words (a unit, the item kind) match most of the shelf and bury everything else: search the wanted value\'s own distinctive word or number by itself ("0.4", "30m", "herringbone") so that only listings holding it come back, and also the alternative spellings that listings use ("3 atm", "30 m"); when such a search\'s total fits on its pages, read them all, and put the item kind next to the values only to bring the nearest matches to the top.\nH10. A keyword search can also match category names and property values. The notes after each search say what the totals so far show about how this store searches.'
READ_ASK = 'OUTPUT: a single JSON object, compact and on one line (no indentation), with nothing else at all. It has twenty-one keys, grouped by the part that defines them: (part 1) roles; (part 2) message_budget, step_budget, price_cap, currency, in_stock_only, priority, priority_detail; (part 3) rules, instructions, decline_no_names, fallback_pick; (part 4) soft_wants, ask_first, hints; (part 5) references, reference_targets; (part 6) product, quantity, requirements, items. What you read is given further down: the shopper\'s request and the list of the store\'s tools.\n\nPART 1 - TOOLS\n"roles" = an object that pairs each role with the exact name of the listed tool playing it, or with null if no tool does; the roles: "search", "details", "view", "cart", "cart_view", "message", "order", "decline". By "message" we mean writing to the shopper, by "cart" placing items in the cart or removing them, by "order" buying or checking out, and by "decline" ending the errand with nothing bought.\n\nPART 2 - BUDGETS AND RANKING\n"message_budget" = whole number: how many messages the shopper lets you send; null when the request is silent about it.\n"step_budget" = whole number: how many steps or moves the request lets you take; null when unstated.\n"price_cap" = the ceiling on the total price, given as the exact number the request writes, in its own unit and never converted ("3050 US cents" means 3050); null when no ceiling is given.\n"currency" = the unit the request uses for prices, or null.\n"in_stock_only" = true when the request is only after items available on hand or in stock.\n"priority" = exactly one of three strings. "cheapest" only when the shopper clearly asks for the lowest-priced (cheapest) matching item or for the lowest order total; "best" only when they clearly ask for the top item on some measure; "none" for every other request. A remark that money is tight or that each penny matters does not ask for the cheapest: it merely gives the reason for the cap.\n"priority_detail" = the measure meant when priority is "best"; "" in every other case.\n\nPART 3 - PROCEDURE THE REQUEST IMPOSES\n"rules" = list of the procedural rules the request lays down for the assistant, each quoted word for word: caps on how many listings are viewed, what must be checked or reported, how to decline, prohibitions, what has to be included. What the shopper means to do on their own side is story and is left out.\n"instructions" = the same rules once again, now one object per instruction ([] if there are none; also include a rule that only says how to finish or what to say when finishing): {"text": no more than 20 words, worded as the request words it, "before_finish": true when it must be done by the finish at the latest, i.e. before the order or the decline goes out (confirming with the shopper beforehand, telling them the total or what the basket holds, a closing remark, one last check), otherwise false, "needs_message": true when doing it calls for a message to the shopper, otherwise false, "before_change": true when the assistant must clear it with the shopper before changing, replacing or removing anything in the cart, otherwise false}. A step that applies only in case no listing meets every condition (what to say or propose then) gets before_finish false and needs_message false: the order of the nearest listing settles it.\n"decline_no_names" = true if the request\'s decline rule forbids suggesting, naming or offering alternatives or substitutes; false if it wants something named; null if the request has no rule of that kind.\n"fallback_pick" = true if the request wants a nearest-match listing proposed in case no listing meets every condition (the one missing the fewest conditions, perhaps within some price margin); false otherwise.\n\nPART 4 - BACKGROUND, WISHES, OPEN QUESTIONS\n"hints" = list of strings shaped "property (product): short wording of the detail" (just "property: detail" when one product is wanted), one for every story detail that bears on a property but leaves its value unstated: what the item serves for or is used with, the place, the user, what it must fit, match or withstand. Name the property as listings would most probably name it. If a detail could concern two properties, give the likelier one first, then a slash and the other ("colour / style (chair): should go with a room that has one painted wall"). Whatever the item has to match, suit or sit nicely beside (a room, a wall, an outfit, companion items) and its look to the eye point first to colour and then to pattern or style. Leave out a detail that only justifies a condition already stated (why that size or that material). Pure background, never a hint: the weather, travel, a schedule or a job, places where the item will not be worn or used, and purposes common to every listing of that kind (warmth for a coat, storage for a dresser). [] if none.\n"soft_wants" = nice-to-haves the shopper does not strictly require (preferably, ideally, if possible, would be nice); none of them ever goes into requirements.\n"ask_first" = points only the shopper can settle that need raising before a choice is made: for example the budget when no limit is set, or which of two wishes weighs more when the request admits both might not be possible (both wishes then go into soft_wants rather than requirements), or any point the request wants cleared with the shopper before choosing. Checks that need the final basket (its approval, its total, the order as such) are finishing steps and go in instructions with before_finish, not here. [] if none.\n\nPART 5 - VALUES THAT POINT ELSEWHERE\n"references" = list of conditions whose value is given solely through a pointer to ANOTHER page or listing, like "any colour except the one on Z\'s page", "within 5 of the figure Y\'s page states", "a colour matching the jumpsuit X sells". Write each as "property (product): as shown on <how the request names that listing>"; if our value has to differ from the pointed listing\'s, as "property (product): different from the one shown on <how the request names that listing>"; if it must be the counterpart of that listing\'s value (the opposite number, the store\'s paired value, or any value meant to suit, get on with or be compatible with it instead of matching it), as "property (product): counterpart of what is shown on <how the request names that listing>". Every entry also goes into requirements, in this exact form. [] if none.\n"reference_targets" = one object per references entry, in the same order, for the listing it points at: {"property": which property, "product": which product, "seller": the listing\'s shop or brand exactly as the request writes it, "kind": the listing\'s product type, "details": [brief strings, worded as in the request, for further values it gives that listing ("4gb", "blue", "net weight 167.262186 g")], "relation": how our value relates to the listing\'s, one of "same"; "pairs" (we want the value the store pairs with the listing\'s: its opposite number or counterpart, or else a value meant to suit or get on with it rather than match it; not the same value and not just any different one); "differ" (must NOT be equal to the listing\'s value: other than, different from, set apart from); "within"; "at most"; "at least", "tolerance": the figure that "within" allows, else null, "unit": the unit of that figure, else ""}. [] when there are none.\n\nPART 6 - WHAT TO BUY AND ITS CONDITIONS\n"product" = the kind of product wanted, in the request\'s own words; with several products, all of them joined by "and".\n"quantity" = whole number of units of one listing to order, 1 unless stated. A thing sold as one piece, such as a set or a pair of shoes, is 1. Only an explicit ask for more units changes it ("order 2 packs", "give me three of them"); how many people are going to share or use it, or how many old ones existed, belongs to the story and keeps it at 1.\n"requirements" = a list holding one string per hard condition on the product, written "property: constraint" with exactly the value, limit, exclusion or closed list the shopper gave; when there are several products, include every one of them, using the form "property (product): constraint".\n * Condition means: a property of the product joined to a value the request demands ("with a mechanical movement", "the pattern has to be herringbone", "the occasion has to be party", "at least 4 drawers"). That value has to consist of words or figures that the request writes for the item itself. When one sold box, pack or set contains several things, their number is the pack size, written with the figure and the noun that follow it in the request ("a box of 40 tea bags" -> "pack size: 40 tea bags", not "tea bags per box: 40").\n * Never conditions: budgets, price, stock, quantity and mere wishes; repeating the category or product kind (category-name words such as a season, occasion or audience it contains are already handled by the category). Story is never a condition: the intended recipient, an event where it gets worn or used ("for the staff party", "to wear at his wedding"), where, when or how it is used, how many people it is shared by, or its being a replacement for an old one ("a new watch" signals a replacement, no condition). Use, occasion or audience count only when the request turns them into a property of the item ("a blazer made for business", "the occasion has to be business"). Qualities, reasons or wishes found in the story (covers the whole drop, natural and breathable, works with what I already have, slim enough for a case) are hints, never conditions. Proverbs and comparisons naming neither a property nor a value of their own (sturdy as an old oak; do it right, like a craftsman would) only set a mood: no condition, no hint.\n"items" = one entry for each DIFFERENT product to be bought, following the request\'s order: {"product": that product\'s kind in the request\'s wording, "quantity": units of one of its listings (counted as for quantity), "requirements": its hard conditions (written as for requirements), "depends_on": the 1-based position of the other entry that, by the request\'s words, this product must fit, suit, work with or be compatible with, else null; a condition that only kicks in depending on how the other product turns out (a requirement containing "if") links nothing, so leave null}. One product means one entry.\n\nFINAL CHECK: each sentence of the request must land somewhere: a rule, a budget, the priority, a requirement, a soft want, a hint, or just story. If a sentence could be a rule, put it in rules.\n'
HEED = 'OUTPUT: one JSON object and no surrounding text. A shopping assistant has to understand the shopper\'s newest line(s); your object explains them in twelve keys, all defined in part B. Part A tells you how to interpret the shopper.\n\nPART A - HOW TO READ THE SHOPPER\nA1. Every change in a line gets reported; one line can carry several.\nA2. Homely, figurative or roundabout phrasing that names a value counts as a firm wish for that value: take the shopper at their word, kindly. A value is excluded only when the shopper clearly says to avoid it.\nA3. Emphasis is no part of a value. Words like strictly, only, stays, locked to, sticking with, nothing but, holding out for or nothing short of merely express how firmly the wish is held, so write the plain value ("color: white", "material: jute"), never "<value> only". Record an exclusion only when the shopper explicitly rules out other values or mixes too.\nA4. Numeric comparisons (no more than, under, over, at least, at most) set limits, and their unit stays with them. A budget mentioned by the shopper is the new_budget and overrides the earlier one, up or down, even when it comes as good news.\nA5. When the shopper says which of two wishes counts for more, the favoured one turns into a need. Plain wishes ("would be nice", "ideally") never do.\nA6. A need about another property, however close (a seat height next to an existing overall height, a strap width next to a width), adds a condition and replaces none. Earlier conditions are dropped or replaced only when the line says so for that same property. In a replacement, dropped receives the old condition and needs receives the new one.\nA7. "No preference", "anything works" or "Don\'t mind" places the asked-about property in no_preference; if two properties were asked in one message, only the FIRST one goes there. Nothing but that property is freed: needs stated before are never removed or altered by it, and a reply like that has no effect when the assistant\'s message only informed and asked nothing.\nA8. Saying the question went over their head is how this shopper shows no preference ("you\'ve lost me there", "that was rather dense", "could you say that another way"), as long as the question mentioned the product owning that property; then that property goes into no_preference as well. When several products are wanted and the question left open which one, the property stays open and no_preference stays empty.\nA9. With several products, write each need in the form "property (product): constraint", with the product taken as the line names it (talk of the "women\'s boots\' toe style" yields "toe style (women\'s boots): ...").\nA9b. Conditions arise from the shopper\'s words only. The assistant\'s question never creates one, and example values offered in that question are not needs.\nA11. A need expressed as a pointer to another listing takes the line\'s pointing phrase itself as its value (e.g. "finish: unlike what Y\'s kettle page shows"). Values the line attaches to that other listing are about that listing rather than our product: they are not needs, not drops and not freed properties.\n\nPART B - KEYS (output them in any order)\n"references": one object per need that the shopper defines purely by pointing to a different page or listing, describing that listing: {"property": which property, "seller": its shop or brand exactly as the line writes it, "kind": its type of product, "details": [brief strings, in the line\'s wording, for other values the line gives it], "relation": "same", "pairs" (we want the value the store pairs with the listing\'s: its counterpart or opposite number, or else a value meant to suit or get on with it rather than match it), "differ" (must NOT be equal to the listing\'s value: other than, different from, set apart from), "within", "at most" or "at least", "tolerance": the figure "within" allows, else null, "unit": the unit of that figure, else ""}; [] if there is no such need.\n"allowance": null unless the lines set a new cap on how many messages, questions, steps, moves or calls the assistant may use; then {"messages_total": the overall number of messages now allowed, or null, "messages_left": how many may still be sent from here, or null, "steps_total": the overall number of steps now allowed, or null, "steps_left": how many steps remain from here, or null}.\n"instructions": working instructions for the assistant, set by these lines and valid from now on (check with me before you buy, give me the total or the basket, do this or don\'t do that, how to end or what to say at the end), one object for each: {"text": the instruction as the lines word it, "before_finish": true when it must be done by the finish (an order or a decline) at the latest, otherwise false, "needs_message": true when it calls for a message to the shopper, otherwise false, "before_change": true when the assistant must clear it with the shopper before changing, replacing or removing anything in the cart, otherwise false}; [] if the lines have none. Needs, budgets and dropped conditions never count as instructions.\n"reply": "" except when the shopper asks the assistant something or wants a point confirmed; then a brief, plain answer to send back that names the change itself (confirming the new value, say).\n"requirements_now": the whole up-to-date list of hard conditions on the product, each "property: constraint". Begin with REQUIREMENTS SO FAR; retain whatever the shopper has neither dropped nor replaced, hold only the latest value of what was replaced, and append each new need from the shopper\'s lines. Story details of the request enter only if REQUIREMENTS SO FAR has them already: the shopper\'s lines are the sole source of anything new. Leave out the product category, quantity, stock and price.\n"meta_only": true when the line merely asks the assistant to send less at a time (a single question per message, less in one go, the first part now and the rest later) without naming any value; a line like that answers no question so far, contributes nothing to needs, no_preference or dropped, and any figure of speech in it names no value. Else false. Not meta_only: a line only saying the question went over the shopper\'s head, or asking for it again shorter or in simpler words.\n"unavailable": true when the shopper says an item can no longer be bought (sold out, out of stock, gone, the button disabled), else false.\n"new_budget": either null or the fresh ceiling on the overall price as a number expressed in the unit of the original budget (take the shopper\'s figure as written when it already uses that unit: against a budget in cents, "13075 cents" is 13075).\n"dropped": conditions from before that the shopper has abandoned.\n"no_preference": each property that the shopper said they do not care about.\n"needs": the hard conditions these lines introduce for the first time, each one a "property: constraint" holding exactly the value, limit, exclusion or closed list given. A value made of words (not a number) after a comparison such as at least, nothing less than or falls short of is the value itself, spelled exactly like the shopper does, e.g. "property: not available" from "falls short of not available" (a field really can contain "none" or "not available").\n"meaning": a single short sentence on what the shopper intends.'
READ_RULES = '(a) Kind. A listing that sits in the requested category is the requested kind, even when its title names a different object (an accessory, a set, a refill, a case or a container); only the requirements, the price and the stock are judged.\n(b) Scope. Judging covers the listed requirements and nothing more: a listing never fails over something they do not name, be it a use, season, occasion, size, gender, age group, quality or condition.\n(c) Currency. A price is read in the budget\'s own unit except where the record states a different currency or unit (45 with a budget of 1000 US cents is 45 cents, well within it); do not rescale either side.\n(d) Figures. Convert the units first, then compare the amounts (45 mm is the same as 4.5 cm). A value on the boundary is within the limit; a range is accepted only if all of it is. If the title has to serve as evidence (no field carries the property), an exact wanted value is met when its words and numbers appear in the title in the same order, more detail following them being fine (for 9\' x 12\': a title with 9\' x 12\' 4" passes, one with 9\' x 13\' fails).\n(e) Wrong property. When the category\'s listings have a field of its own for a property, a field about some other property never replaces it (a lining-material field does not answer material, nor a length field style, nor a strap-length field strap type).\n(f) Where to look. A requirement is settled by the field that belongs to its property, checking the wanted value word by word against the way this store writes it there. How a field name is spelled does not matter: separators, prefixes, long or short forms and well-known other names are all accepted (pants fit type as "fit", collar type as "neckline", colour as "color family"). Decisive whenever it exists: a field carrying the very name of the property, or one that puts a single extra word ahead of that name (pants length for length, clothing material for material); the value found in either of them is enough. Names that differ only by separators, like size class and size_class, are the same field, so the value under either spelling suffices. If no such field exists, the field under the other name decides; the title or the options list may supply the value only if not one field names the property, or if a FIELD EVIDENCE line allows the title to stand in for a field missing from this listing. A description never counts as evidence.\n(g) Matching words. Ignore case, separators, singular or plural and spelling variants. A single matching element of a list suffices, and a narrower value that includes the wanted one is accepted ("80% cotton" for cotton, "forest green" for green, "rtx 3060 12gb" for rtx 3060). Shades, synonyms, related words and abbreviations (off-white ~ ivory or cream, plain ~ "solid", technical wood ~ "tech wood", embroidery ~ "embroidered", windows 11 ~ "win11") are accepted only as long as no listing in the category spells out the wanted value in that same field: as soon as other listings say "matte" for finish, "satin" no longer counts as matte. A value joined with a plus ("wool + nylon") does not count for one of its parts if that field also appears elsewhere holding the other part by itself ("nylon"). Where FIELD EVIDENCE lines exist they decide which of these situations applies, because code computed them from this store\'s own records. Rejected as well: a different value of the same property (rosewood fails maple, quartz fails mechanical), a word that merely looks like the wanted one, a blanket claim covering every value and therefore naming nothing ("universal", "all seasons", "suitable for all occasions") unless its list names the wanted value as well, and a value that usually goes together with, or hints at, the wanted one ("slip-on" fails "lace-up", and a "pullover" style fails "zip-up"). The sole accepted substitutes are a shade normally sold under the wanted colour name and a narrower value that includes the wanted one.'
def _rules(text): return text.replace('@RULES@', READ_RULES)
ITEM_CHECK = _rules('OUTPUT: just one JSON object, with no text around it, whose keys are "item" = the id of the item, "fits" = true or false, and "problems" = brief strings, one per requirement that is unmet or unshown.\n\nA shopping assistant is about to commit to this item and you are its last gate; the errand earns 1 only when every requirement below holds. Judge from the records shown here and nothing else.\n\nSay false in precisely two cases: (1) under the rules, the item\'s fields plainly break a listed requirement, for instance when a FIELD EVIDENCE line says the item does NOT meet one; (2) no record states a listed requirement: no field states it and, where no field for it exists in the store or a FIELD EVIDENCE line allows the title in its place, neither the title nor the options list does. A stock or price record being absent is never on its own a reason for false.\n\nRULES\n@RULES@\n(h) The category itself takes care of words that form part of its own name.\n(i) Closed lists and exclusions are binding.\n(j) Only the requested kind will do; where listings bear category labels, its label must suit the wording of the whole request rather than a narrower or neighbouring label.\n(k) Price times quantity must not exceed the budget; being equal is fine. A record showing no stock means fail.\n(l) Wishes flagged as optional ("ideally", "would be nice") are not requirements.\n(m) If records disagree (about stock or price, say), go by the newest one.')
RANK_CHECK = _rules('OUTPUT: just one JSON object, {"better": [up to 4 ids, best first], "reason": "one sentence"}, and nothing else.\n\nThe request wants the PRIORITY, and the shopping assistant has already settled on an item to order (the pick). You get the requirements, the pick and every listing record seen so far. Name the listings that could satisfy all the requirements and also outdo the pick on that priority (a lower price when the cheapest is wanted, a stronger measure value when the best is wanted). Use the rules given here: price times quantity has to stay inside the budget, and no record may show the item out of stock. Exclude any listing whose record shows it failing a requirement, out of stock, or over budget. Title numbers on a ROW, such as 55 x 94, do not tell which dimension each one is, so they can never exclude a ROW.\n\nRULES\n@RULES@')
ASK_PLAN = 'OUTPUT: only one JSON object, shaped {"questions": [{"property": "...", "second": "... or empty", "question": "..."}, ...]}, with no other text. Include QUESTION_COUNT distinct properties at most, strongest candidate first.\n\nWHY IT MATTERS: a shopping assistant will ask a shopper these questions, and each of the shopper\'s hidden needs is the value of a single property of the product. Only a message that asks about exactly that property brings a need out; the first thing a message asks gets the most trustworthy answer, and asking about anything else earns no more than a polite "don\'t mind".\n\n1. NEVER ASK ABOUT\n - a property the request has already settled (through its requirements, price, stock or quantity); a property asked before; both members of a near-duplicate pair (two style kinds, one property spelled two ways);\n - a property where the shopper answered an earlier question by saying they did not understand it: the shopper has no need there, so it is spent and must not come back reworded; move on to a different property or product;\n - the product\'s OWN brand, maker, product name or model name, price, stock, delivery (the make or model of ANOTHER thing the product must fit, such as the vehicle or device it goes on, is a fine property to ask);\n - keys that pick an option or variant, keys that are numbered copies of another, and catch-alls (options, variants, features or any other plural key naming a list of things, description, title, name);\n - any property missing from the keys of the listing details below. Exception: a STORY HINTS property without a matching key may be asked under its plain name, unless the hint is tagged [no key] (meaning enough listings of the product were read and none had such a field); then use the key below with values that differ in the way the hint\'s detail suggests, or skip it.\n\n2. THE SHAPE OF A QUESTION\n - A step-1 question is one brief, plain, friendly sentence of 15 words or fewer asking which <property> the shopper wants for the <item kind>, optionally followed by one sample value (never two values joined by or). Sample values alone never bring a need out, so the question must name the property itself. No lead-in: no status update, no instructions, no lists.\n - Use for the property the exact key written by one of that product\'s listings below, copied as is; "second" is then left empty.\n - ONE property of ONE product in each question (the step-0 question that weighs two wishes is the only exception): a question never covers two properties or two products, since a reply reveals at most one need and a question about two things may go wholly unanswered.\n - "second" may name the SAME product\'s runner-up property (a different property, not a rewording of the first) or remain empty; the assistant may add it to its closing message.\n\n3. MORE THAN ONE PRODUCT\nWhen SEVERAL products are wanted, any of them may carry hidden needs. Spread the questions out: no product gets a second question until each has had one (passing over products with no property left). Draw every product\'s properties from the list of its own category below, write them "property (product)", and say the product inside the question ("For the boots, which toe style would you like?").\n\n4. WHAT TO ASK, AND IN WHAT ORDER\nStep 0 - before anything else, ask plainly about whatever appears below under MUST BE LEARNED FROM THE SHOPPER and is not yet asked. For the budget, ask for the highest amount they are prepared to pay. For a choice between two wishes, name both values together with their property, e.g. "If no single listing offers both, would you prefer the <value> <property>, or rather the <value> <property>?".\nStep 1 - next, properties that really occur in the listing details below, chosen this way:\n a) The story first. Any STORY HINTS go first; every hint names a detail from the story and the property it touches. Normally the hidden need lies on the property the concrete story details concern, with its value left unspoken.\n b) From cue to property. A cue reveals the part or side of the item, hence the property, that the need sits on: something the user is going to touch or press, look at up close or in pictures, hear, measure or fit against something, something read on its label, the way it was made. Usual mappings (cue => property):\n    when, where or for which activity it is worn or used (the office, a gym class, outdoors) => style or type, meaning the property whose values name uses like these;\n    the inner part doing the work => core component;\n    fastening or adjusting over and over => closure or mechanism;\n    a present for another person or a near-stranger => who it is made for (age group, gender);\n    feel on skin, hand or foot => material or fabric first;\n    fitting a body part or a space => size or shape of that part;\n    how it looks in person or in pictures => colour, pattern or the visible part;\n    labels, tags or small print => ingredients, fabric or material.\n    Mere background, never cues: a job, a schedule, the weather, places where it will not be worn or used.\n c) The opening question goes to the property those details point at, named as the listings name it. The recipient (who it is made for) is merely one candidate; raise it if the story brings in another person, or if no other hint exists.\n d) With no story cue left, start with the property that buyers of this kind of item fix before any other: the one whose values break the kind into sub-kinds that are not interchangeable in use (what it runs on, which vehicle or device model it must fit, how big it must be for a body or a space, the form it takes or the container it comes in, the main material, the way it connects), asked under the listings\' own key. A key with values that vary only because each listing fills it with free text can never be it.\n    Beyond that, descriptive properties (pattern, material, shape, type or style variant, closure, length, function) outrank logistics (warranty, condition, pack size, variant or size lists, years).\n e) The key order below (widest coverage over listings, most varied values) is used only to break ties between candidates that c) and d) rank alike. A hidden need nearly always points to a value carried by few listings in the category, so a key showing the same single value in the handful of listings read stays a strong candidate, since the wanted value is probably absent from them. Among the listings read, coverage beats variety. Any key flagged as a figure with a unit (a size, width, height, length, weight) deserves its own slot if there is one. Favour keys present in most of that product\'s opened listings. When several products are requested, prefer the one with the most fitting listings left, since the hidden need has to cut that set down.'
ONE_ASK = '\nNOTE FOR THIS CALL: just one question reaches the shopper this time, and they answer it before anything more is chosen. REMAINING question(s) are still available in all; plan on using them all. Lead with your top property; one or two spares may follow it. After a need has just surfaced, point the next questions at a property not yet covered and a story detail not yet used.'
EARLY_DECL = _rules('OUTPUT: just one JSON object, with no text around it: {"possible_fits": [up to 5 ids, best first, of listings that look like they meet every requirement], "unopened": [up to 6 ids of listings known only from rows, to open before declining], "reason": "one sentence"}.\n\nA shopping assistant is ready to report to the shopper that the store has nothing suitable. Such an ending is mostly wrong, so test the listing records below against the requirements by the rules given here (the budget must cover the price, and no record may show the item out of stock).\n\nRULES\n@RULES@\n\nNOTE ON LISTINGS SEEN ONLY AS SEARCH ROWS: nobody opened their details, so their property fields are not known. A row of that sort inside the requested category (whatever object its title mentions) counts as unchecked when neither its title nor its price nor its stock reveals a failed requirement; title numbers like 55 x 94 do not tell which dimension each one is and therefore never prove a failure. List such rows in "unopened", likeliest first: titles that name a required value, or rows of the requested kind that nothing counts against.')
REF_BOUNCE = 'OUTPUT: just one JSON object, with no text around it, holding the keys goes_with, value, go_to, missing and target.\n\nFor their product, a shopper wants the PROPERTY copied from another listing in the store, which the request describes in these words: DESCRIPTION. Each of the RECORDS below may be that listing.\n\nTwo warnings before you start. Whatever its text may claim, a record only counts as the target when the described details can be seen on it. And these never make a go_to: a listing brought up only to say the target is not it ("not the green one from X"); other versions, sizes or colours sold elsewhere; a listing cited for another property.\n\n"goes_with" - list of values that pair with that value, go with it or count as the same, going by the target\'s own text or by a store rule written into it; if the rule lists several pairs, take just that value\'s partners (or the value itself when the rule pairs the value with itself); [] if there is none.\n"value" - filled only if go_to is null: the PROPERTY as written in the target\'s record, taken from its field for the PROPERTY; failing such a field, the value its notes state for the PROPERTY (fall back on the options or the title only when the notes state nothing).\n"go_to" - null in the normal case. If, however, the target\'s own text sends readers elsewhere for the PROPERTY, to a DIFFERENT listing (go by, look at, refer to, use or take it from another listing), give that listing as {"seller": its shop or brand name exactly as written, "kind": what type of product it is, "details": [each value it is given in that text, brief and word for word: a size with its unit, a colour, a material]}.\n"missing" - named details that the target fails to show; [] when it shows every one. A detail is shown if the record contains its value in any place, be it the notes, the title, the options or a field with another name.\n"target" - id of the one record that truly is the described listing: seller and kind both right, with every named detail to be seen on it; interpret details by sense, so the same amount in another unit counts. null when none qualifies.'
VARIANT_JUDGE = _rules('OUTPUT: just one JSON object with four keys and no text around it:\n- "reason": "one sentence".\n- "pick": the id to order now; null if none of the listings qualifies.\n- "cart_still_right": null if the cart is empty; true if the cart\'s contents are listed in fits and are the right thing to order (keep a basket that still fits); false if a cart line no longer fits or a cheapest or best rule now points to a better pick.\n- "fits": ids of every OPENED listing that meets all conditions currently in force, costs no more than the budget and has no record showing it out of stock, strongest first (cheapest first when the cheapest is wanted).\n\nBackground: something has just changed the errand\'s terms (a store notice, a line from the shopper, or a refused finish), bringing a different budget, a requirement added or taken away, or an item gone. Supplied below are the request, what changed, the current requirements, the PRIORITY, THE CART and a card for each listing encountered (a DETAILS card holds a complete record). Judge every OPENED listing afresh under the terms that apply NOW, with these rules:\n@RULES@')
FINISH_CHECK = 'OUTPUT: just one JSON object, with no text around it, holding two keys.\n- "ready": false as long as a stated step is still owed; true once none remains.\n- "pending": up to two owed steps, in the order to run them, each {"why": a short label of the stated step, "tool": the exact name of the tool doing it, "args": the arguments, shaped precisely as that tool\'s schema requires; for a message to the shopper, give the full text: one or two brief plain sentences holding all facts the step calls for (for the basket: each cart line with id, count and price, followed by the total)}. Steps none of the listed tools can do are left out.\n\nContext: this is the last review before a shopping assistant wraps up its errand with an order or a decline. Stated steps can come from the request, from later lines of the shopper or from store notices; some fall due at or before the finish (confirming with the shopper ahead of time, a report of the total or of what the basket holds, a closing remark, a further check of something). Invent nothing: count only steps the request, the shopper or the store actually stated, and only those a listed tool can perform.\nMethod: look through DONE AFTER THE LAST CART CHANGE, INSTRUCTIONS IN FORCE and the NEWEST SHOPPER LINES AND STORE NOTICES, then judge whether a stated step is still open for THIS finish. A basket step done before the latest cart change does not count as done.\n\nBasket note: if the finish is an order and the shopper has received no message since the latest cart change, the code itself sends the basket and its total; only list a step of that sort if a stated step demands more or something else (answering a question, waiting for a yes).'
NO_BUY_WORDS = 'refus|declin|cancel|abandon|reject|give.?up|abstain|withdraw|no.?(?:purchase|order|buy)|nothing|without'
ORDER_WORDS = 'commit|buy|order|purchas|check.?out|pay|confirm|complete|place|accept|proceed'
PARDON = 'Sorry, nothing in the store meets all of your requirements, so I am not ordering anything.'
ID_RX = '[A-Za-z0-9][A-Za-z0-9_\\-:.]{3,63}'
CODE_RX = '[A-Za-z0-9_\\-]+'
def _cut(t, n, tail): return t if len(t) <= n else t[:n] + tail
def _lo(t): return str(t or '').lower()
def _ls(t): return str(t).lower()
def as_text(morsel, ceiling=VIEW_CHARS):
  try: t = json.dumps(morsel, ensure_ascii=False, separators=(',', ':'))
  except (TypeError, ValueError): t = str(morsel)
  return _cut(t, ceiling, '...(cut)')
def snip(morsel, characters, rankno=0):
  if rankno > 8: return None
  if isinstance(morsel, str): return _cut(morsel, characters, '...')
  if isinstance(morsel, dict): return {k: snip(v, characters, rankno + 1) for k, v in morsel.items()}
  return [snip(v, characters, rankno + 1) for v in morsel] if isinstance(morsel, list) else morsel
_FENCE = _rc('```[A-Za-z]*')
def _js(t):
  try: return json.loads(t) if t else None
  except ValueError: return None
def as_json(body_text):
  if not isinstance(body_text, str): return None
  r = body_text.strip()
  if r.startswith('```'):
    r = r[_FENCE.match(r).end():].strip()
    r = r[:-3].strip() if r.endswith('```') else r
  if not isinstance(v := _js(r), dict):
    i = r.find('{')
    v = _js(r[i:r.rfind('}') + 1] if i >= 0 else '')
  return v if isinstance(v, dict) else None
def is_num(morsel): return isinstance(morsel, (int, float)) and not isinstance(morsel, bool)
def long_strs(morsel, garnered, rankno=0):
  if rankno > 4: return
  if isinstance(morsel, list):
    for v in morsel[:10]: long_strs(v, garnered, rankno + 1)
  elif isinstance(morsel, str) and len(morsel.split()) > 3: garnered.append(morsel)
def all_strs(dx, out):
  for v in dx.values():
    if isinstance(v, (dict, str)):
      v = v.values() if isinstance(v, dict) else (v,)
      out.extend(t.strip() for t in v if isinstance(t, str) and t.strip())
def walk_dicts(morsel, attend, rankno=0, ceiling=6):
  if rankno > ceiling: return
  if isinstance(morsel, dict):
    attend(morsel)
    morsel = morsel.values()
  elif not isinstance(morsel, list): return
  for v in morsel: walk_dicts(v, attend, rankno + 1, ceiling)
_BRK, _CAMEL, _ALPHA, _ALNUM = _rc('\\s*\\('), _rc('([a-z])([A-Z])'), _rc('[a-z]+'), _rc('[a-z0-9]+')
def _crest(name): return _BRK.split(str(name), 1)[0]
def _parts(name, rx): return rx.findall(_CAMEL.sub('\\1 \\2', _crest(name)).lower())
_CLUT1 = _rc('\\b(?:brand|shop|seller|store|vendor|manufacturer|category|categories|department|title|headline|heading|listing|description|overview|price|cost|amount|availability|sku|url)\\b')
_CLUT2 = _rc('(?:name|notes?|stock|in stock|on hand|images?|links?|make|maker|shelf|ids?|id)')
def junk_key(key):
  w = ' '.join(_parts(key, _ALNUM))
  return bool(_CLUT1.search(w) or _CLUT2.fullmatch(w))
def enum_pair(key):
  t = _parts(key, _ALNUM)
  return len(t) > 1 and len(t[-1]) == 1 and t[-1].isdigit() and t[-2].isalpha()
def _shed(w, keep=True): return w[:-1] if len(w) > 3 and w[-1] == 's' and not (keep and w[-2] == 's') else w
def prop_words(name): return [_shed(w) for w in _parts(name, _ALPHA)]
def prop_key(name): return ''.join(prop_words(name))
def edits_in(a, bv, utmost):
  if abs(len(a) - len(bv)) > utmost: return False
  up = list(range(len(bv) + 1))
  for i, c in enumerate(a, 1):
    row = [i]
    for j, d in enumerate(bv, 1): row.append(min(up[j - 1] + (c != d), row[j - 1] + 1, up[j] + 1))
    if min(row) > utmost: return False
    up = row
  return up[-1] <= utmost
def _sl(x, y): return (y, x) if len(y) < len(x) else (x, y)
def same_prop(a, bv):
  x, y = prop_key(a), prop_key(bv)
  if not (x and y) or x == y: return bool(x and y)
  s, f = _sl(x, y)
  n = len(s)
  return n > 3 and (x[:3] == y[:3] and edits_in(x, y, 2 if n > 5 else 1) or n > 4 and f.startswith(s))
def _arr(w, t, k):
  return w == t if k == 0 else any((w[i] in 'aeiouy' or w[i] in w[max(i - 1, 0):i] + w[i + 1:i + 2])
                  and _arr(w[:i] + w[i + 1:], t, k - 1) for i in range(len(w)))
def case_alike(lo, up):
  if k := len(up) - len(lo): return k in (1, 2) and _arr(up, lo, k)
  p = [i for i, (x, y) in enumerate(zip(lo, up)) if x != y]
  return len(p) == 2 and p[1] == p[0] + 1 and lo[p[0]:p[0] + 2] == up[p[0]:p[0] + 2][::-1]
def exact_prop(a, bv):
  l, r = prop_words(a), prop_words(bv)
  x, y = ''.join(l), ''.join(r)
  if not (x and y) or x == y: return bool(x and y)
  s, f = _sl(x, y)
  if len(s) > 3 and s[:3] == f[:3] and case_alike(s, f): return True
  s, m = _sl(l, r)
  return len(m) == len(s) + 1 and m[:-1] == s and len(m[-1]) < 4 and len(''.join(s)) > 4
ZH_SKIP = {'and', 'the', 'a', 'an', 'of'}
BOUND_RX = '\\b(?:at least|no more than|at most|no less than|more than|less than|up to|or more|or less|or larger|or smaller|or higher|or lower|minimum|maximum|between|within|under|over|above|below|than)\\b'
BOUND_WORDS = 'at least|no more than|at most|no less than|more than|less than|up to|or more|or less|or larger|or smaller|or higher|or lower|minimum|maximum|between|within|under|over|above|below|exactly|only|either|neither|nor|than|plus'
BOUND_NEG = '\\b(?:no|not)\\s+(?:more|less|fewer|over|under|above|below|beyond|exceeding)\\b'
NEG_WORDS = '\\b(?:no(?!-)|not|without|nothing|never|neither|nor|except|excluding|avoid\\w*|anything but|other than|rather than|instead of)\\b'
CONTRAST_RX = '\\b(?:falls?\\s+short\\s+of|short\\s+of|nothing\\s+(?:less|other)\\s+than|nothing\\s+short\\s+of|nothing\\s+but|no\\s+(?:less|worse)\\s+than|at\\s+least|(?:less|more|better|worse)\\s+than|exactly|only|just)\\b'
TEXTUAL_DATA, EMPORIUM_DOUBLES = set(), {}
def vowel_pair(a, bv):
  if len(a) != len(bv) or len(a) < 4 or a[:2] != bv[:2] or not (a.isalpha() and bv.isalpha()): return False
  p = [i for i, (x, y) in enumerate(zip(a, bv)) if x != y]
  return len(p) == 1 and a[p[0]] in 'aeiouy' and bv[p[0]] in 'aeiouy'
def spell_pair(a, bv):
  if a == bv: return False
  s, f = _sl(a, bv)
  return vowel_pair(a, bv) or (len(s) > 3 and s[:3] == f[:3] and s.isalpha() and f.isalpha()
                  and case_alike(s, f))
_DLEX, _NOTCH = _rc('[a-z0-9]+(?:\\.[0-9]+)?'), _rc('\\((?:stated|revealed[^)]*)\\)')
_STRESS, _OR, _NUM = _rc('\\b(?:only|just|exactly|strictly)\\b'), _rc('\\bor\\b'), _rc('\\d')
_DPT, _RSEV = _rc('\\d+(?:\\.\\d+)?'), _rc('[,;|]')
def txt_words(text): return [t for t in _DLEX.findall(_lo(text)) if t not in ZH_SKIP]
def word_in(word, stash):
  return any(c == word or word.isalpha() and c.isalpha() and (
    same_kind(word, c) or c in EMPORIUM_DOUBLES.get(word, ()) or
    min(len(word), len(c)) > 5 and word[:2] == c[:2] and edits_in(word, c, 1)) for c in stash)
def vsubs(value):
  t = _STRESS.sub(' ', _NOTCH.sub(' ', _lo(value)).replace('"', ' '))
  return [w for w in (txt_words(h) for h in _OR.split(t)) if w]
def val_in_txt(value, text): return zh_match(vsubs(value), txt_words(text))
def _glued(stash, i, t):
  at = 0
  for m in stash[i:]:
    if at >= len(t): break
    if not t.startswith(m, at): return False
    at += len(m)
  return at == len(t)
def zh_match(subs, stash):
  if not stash: return False
  for sel in subs:
    if all(word_in(s, stash) for s in sel): return True
    j = ''.join(sel)
    if j and (len(sel) > 1 or _NUM.search(j) or any(_NUM.search(w) for w in stash)):
      if any(_glued(stash, i, j) for i in range(len(stash))): return True
  return False
def _iso(f): return '(?<![\\d.])' + re.escape(f) + '(?!\\d|\\.\\d)'
def val_touch(value, text):
  lex, raw = txt_words(text), _lo(text)
  heads = {t[:3] for t in lex if len(t) > 2}
  return any(word_in(s, lex) or _DPT.fullmatch(s) and re.search(_iso(s), raw) or len(s) > 2 and s[:3] in heads
       for sel in vsubs(value) for s in sel)
def text_parts(text):
  w = str(text or '').strip().lower()
  m = [h.strip() for h in _RSEV.split(w) if h.strip()]
  return m if m or not w else [w]
def zh_facets(segment):
  a = [x.strip() for x in str(segment).split('+') if x.strip()]
  return a if len(a) > 1 else []
_UNIT = _rc('\\s*([a-z]+)')
def num_built(fgr, voiced):
  return any('.' in fgr or not (u := _UNIT.match(voiced, m.end())) or len(u[1]) < 4
       for m in re.finditer(_iso(fgr), voiced))
_LEX3, _YESNO = _rc('[a-z]{3,}'), _rc('\\W*(?:yes|no|true|false|none|required|included|needed)\\W*')
_HYP, _PROV = _rc('\\b(?:' + BOUND_WORDS + ')\\b'), _rc(',|;|/|&|\\(|\\)|\\bor\\b|\\band\\b')
_ELIM = _rc(NEG_WORDS)
_CSEV = ((_rc(BOUND_NEG), False), (_rc('\\b(?:no|not)\\s+[a-z]+er\\b(?=\\s+than\\b)'), False),
    (_rc('\\b(?:no|not)\\s+[a-z]+er\\b'), True))
def _mat(t): return [w for w in _LEX3.findall(t) if w not in _HL]
def confirm_val(value, name=''):
  t = str(value).lower().strip()
  if _YESNO.fullmatch(t) or TEXTUAL_DATA and ' '.join(_ALNUM.findall(_NOTCH.sub(' ', t))) in TEXTUAL_DATA: return True
  for rx, num in _CSEV:
    if not num or _NUM.search(t): t = rx.sub(' ', t)
  if max(len(_mat(c)) for c in _PROV.split(_HYP.sub(' ', t))) > 3: return False
  if not (d := _ELIM.search(t)) or _mat(t[:d.start()]): return True
  after, own = _mat(t[d.end():]), set(prop_words(name)) | set(_ALPHA.findall(_ls(name)))
  return bool(after and own) and all(w in own or w.rstrip('s') in own for w in after)
_DTRIAL = _rc('\\d+(?:\\.\\d+)?(?:\\s*[:x/]\\s*\\d+(?:\\.\\d+)?)*')
def _trx(run, tail):
  n = _DPT.findall(run)
  return n, '(?<![\\d.])' + '\\D{0,4}'.join(re.escape(f) for f in n) + tail
def val_seen(value, egvals):
  if not (seen := _lo(egvals)).strip(): return False
  hub = _HYP.sub(' ', _ls(value))
  return (any(re.search('\\b' + re.escape(w if len(w) < 5 else w[:-1]), seen) for w in _mat(hub))
      or any(re.search(_trx(r, '(?![\\d.])')[1], seen) for r in _DTRIAL.findall(hub)))
_POSS = _rc("'s\\b")
def kind_words(text): return {w.rstrip('s') for w in _mat(_ls(text))}
def kind_toks(text): return _mat(_POSS.sub('', _ls(text).replace('’', "'")))
_PLR = (('', 0, ('', 's', 'es')), ('y', 1, ('ies',)), ('fe', 2, ('ves',)), ('f', 1, ('ves',)), ('ies', 3, ('y',)),
    ('ves', 3, ('f', 'fe')), ('es', 2, ('',)), ('s', 1, ('',)))
def multi_forms(word):
  b, forms = str(word), set()
  for end, cut, tails in _PLR:
    if b.endswith(end): forms.update(b[:len(b) - cut] + t for t in tails)
  return {f for f in forms if len(f) > 2}
def same_kind(a, bv): return a == bv or str(bv) in multi_forms(a) or str(a) in multi_forms(bv)
def kind_in(word, stash): return any(same_kind(word, c) for c in stash)
def kinds_in(words, stash): return all(kind_in(w, stash) for w in words)
def kind_score(own, words, mutual):
  return sum(0.5 if kind_in(w, mutual) else 1.0 for w in own if kind_in(w, words))
def _osc(word):
  f = sorted(multi_forms(word), key=lambda x: (-len(x), x))
  return '(?:' + '|'.join(re.escape(x) for x in f) + ")(?:'s\\b|'|\\b)"
def name_rx(name): return "[\\s'-]+".join(_osc(t) for t in kind_toks(name))
_TAG = _rc('\\(([^()]*)\\)\\s*$')
def one_prop_hint(allusion):
  try:
    head, colon, spec = allusion.partition(':')
    if not colon or '/' not in head: return allusion
    tag = _TAG.search(head)
    names = [s.strip() for s in (head[:tag.start()] if tag else head).split('/') if s.strip()]
    if len(names) < 2: return allusion
    first = names[0] + (' (' + tag.group(1).strip() + ')' if tag else '')
    return f'{first}:{spec} (or {", ".join(names[1:])})'
  except Exception: return allusion
_VLEX = _rc('\\s*([a-z][a-z-]*[a-z])(?:\\s*[,;.()]|\\s*$|\\s+(?:' + '|'.join(sorted(_HL)) + ')\\b)')
_PTERM = _rc("[a-z][a-z'-]*")
def first_val(specific, request, product):
  try:
    if not (m := _VLEX.match(_ls(specific))): return ''
    lead, names = m.group(1), _mat(_ls(product))
    if not names or len(lead) < 3 or lead in _HL or lead.rstrip('s') in {n.rstrip('s') for n in names}: return ''
    head, t = names[0].rstrip('s'), _PTERM.findall(_ls(request))
    return lead if any(w == lead and n.partition("'")[0].rstrip('s') == head for w, n in zip(t, t[1:])) else ''
  except Exception: return ''
_REFER = _rc('as shown|same as|the page|listing|sold by|redirect|refer')
_YESNO2 = _rc('\\W*(?:yes|no|true|false|required|included|needed)\\W*')
def _known(w, lexes, roots):
  f = w[:-2] if w.endswith('es') and len(w) > 4 else w[:-1] if w.endswith('s') and len(w) > 3 else w
  return any(a == w or a.startswith(f) and len(a) - len(f) <= 3 or f.startswith(a) and len(a) >= max(4, len(f) - 2)
       or len(w) > 3 and (a.startswith(w) or len(a) > 3 and a[0] == w[0] and edits_in(a, w, 1))
       for a in lexes) or any(len(a) > 2 and f.startswith(a) and len(f) - len(a) <= 3 for a in roots)
def story_guess(req, request, refs):
  whole = str(req)
  nm, _, said = whole.partition(':')
  sought = said.strip().lower()
  if not sought or whole in refs or _REFER.search(sought): return []
  if _YESNO2.fullmatch(sought): sought = _crest(nm).lower()
  hub, heard = _HYP.sub(' ', sought), _ls(request)
  digits = _DPT.findall(hub)
  if digits and all(num_built(n, heard) for n in digits): return []
  lexes = set(_ALPHA.findall(heard))
  roots, words = {_shed(t, False) for t in lexes}, _mat(hub)
  missing = [w for w in words if not _known(w, lexes, roots)]
  if len(words) < 3 and any(len(w) > 3 and w not in missing for w in words): return []
  for r in _DTRIAL.findall(hub):
    n, rx = _trx(r, '(?!\\d)')
    if len(n) > 1 and re.search(rx, heard) is None: missing.append(r)
  return missing
HANDLE_BOUNCES, HANDLE_SUMMONS, HANDLE_STRIDES, HANDLE_DEVIATE_UNSEEING, HANDLE_DEVIATE_STRIDES = 8, 12, 0.4, 9, 0.4
REF_WORDS = '\\bas shown\\b|\\bshown on\\b|\\bsame as\\b|\\bpage\\b|\\bsold by\\b|\\bredirect|\\brefer(?:s|red|ring)?\\b|\\blisting points\\b|\\bpoints you\\b'
SELLER_KEYS = 'shop|seller|store|vendor|brand|make|manufactur|merchant|retailer|supplier|sold.?by'
BOUND_TERMS = '\\bwithin\\b|\\bat (?:most|least)\\b|\\bno(?:t any)? (?:more|less) than\\b|\\bsame as\\b|\\bas shown\\b|\\bthe value\\b|\\bthe one\\b'
DIFFER_WORDS = '\\bset apart from\\b|\\bapart from\\b|\\bdistinct from\\b|\\bdifferent (?:from|to|than)\\b|\\bdiffer(?:s|ent|ing)?\\s+from\\b|\\bother than\\b|\\bunlike\\b|\\bnot the same as\\b|\\banything but\\b'
PAIR_WORDS = '\\bcounterparts?\\b|\\bopposite numbers?\\b|\\bpartner value\\b|\\bpaired value\\b|\\bcompatible with\\b'
_PTR_RX = _rc(REF_WORDS)
_GHOLE, _CODEX = _rc('(\\d)(?:\\s+|\\s*-\\s*)([a-z]{1,8})\\b'), _rc('(?<![0-9])[a-z]+\\d*$')
_HTERM, _ZS = _rc('\\d+(?:[.,:/]\\d+)*[a-z]*|[a-z][a-z0-9]*'), _rc('(?:ch|sh|x|z)es$')
_FGAUGE = _rc('(\\d+(?:\\.\\d+)?(?:\\s*/\\s*\\d+(?:\\.\\d+)?)*)\\s*-?\\s*([a-z]{1,8})\\b')
_SOL, _NALNUM = _rc('\\s*/\\s*'), _rc('[^a-z0-9]')
def is_ref_text(value): return bool(_PTR_RX.search(_lo(value)))
def _unite(m):
  fig, unit = m.groups()
  return m[0] if unit in _HL or _CODEX.search(m.string[max(0, m.start() - 16):m.start()]) else fig + unit
def _depl(x, n, guard):
  if len(x) < n or x[-1] != 's' or x[-2] == 's' or guard and x[-2].isdigit(): return x
  return x[:-2] if _ZS.search(x) else x[:-1]
def ref_terms(text):
  return [_depl(t, 4, True) for t in _HTERM.findall(_GHOLE.sub(_unite, _lo(text)))
      if t not in _HL and (len(t) != 1 or t.isdigit())]
def ref_figs(text):
  found = set()
  try:
    for m in _FGAUGE.finditer(_lo(text).replace(',', '')):
      u = _depl(m[2], 3, False)
      if u not in _HL: found.update((float(n), u) for n in _SOL.split(m[1]))
  except Exception: pass
  return found
def norm_name(text): return _NALNUM.sub('', _POSS.sub('', _lo(text)))
_CONN = (('pairs', _rc(PAIR_WORDS)), ('differ', _rc(DIFFER_WORDS)),
    ('within', _rc('\\bwithin\\s+(\\d+(?:\\.\\d+)?)\\s*([a-z%]+)?')),
    ('at most', _rc('\\b(?:no more than|at most|up to|or less|or lower|or smaller)\\b')),
    ('at least', _rc('\\b(?:at least|no(?:t any)? less than|or more|or higher|or larger)\\b')))
def ref_link(value):
  for rel, rx in _CONN:
    if m := rx.search(_lo(value)):
      return (rel, float(m[1]), (m[2] or '').strip()) if rel == 'within' else (rel, None, '')
  return 'same', None, ''
_SUBSEV, _TRAIL = _rc('\\(\\s*or\\b'), _rc('\\s[—–]\\s|\\s-\\s|\\s(?:shows?|says|lists)\\b')
_RET = _rc("(?<![A-Za-z'])([A-Z][a-z]{3,})(?:'s)?(?![A-Za-z])")
_MKT = _rc('(?is).*(?:\\bthe|\\bfor|\\ba|\\ban)\\s+(.*?)\\s+(?:sold|made|offered|listed)\\s+by\\s*$')
_SVAN = _rc('(?i)\\s*([^,;(]*?)(?=\\s+(?:with|that|which|in|at|having|from|of|for|sold)\\b|[,;]|$)')
_SSEV = _rc('(?i),|;|\\band\\b|\\bwith\\b|\\bthat\\s+(?:is|are|has|have|comes?(?:\\s+as)?)\\b|\\bwhich\\b|\\bin\\b|\\bat\\b')
_SLEX = _rc("[A-Za-z0-9][A-Za-z0-9.%:/&'-]*")
def parse_ref(text, ok=None):
  try:
    body = _SUBSEV.split(str(text or ''))[0]
    m = next((x for x in _RET.finditer(body) if x.start() and (ok is None or ok(x[1]))), None)
    if m is None: return None
    rest, kind = body[m.end():], ''
    if k := _MKT.search(body[:m.start()]): kind = k[1]
    elif k := _SVAN.match(rest): kind, rest = k[1], rest[k.end():]
    ws = ([w for w in _SLEX.findall(c) if w.lower() not in _HL] for c in _SSEV.split(_TRAIL.split(rest)[0]))
    return dict(seller=m.group(1), kind=kind.strip()[:80], details=[' '.join(w) for w in ws if w][:6])
  except Exception: return None
def _alnums(text): return re.findall('[a-z0-9]+', str(text or '').lower())
def prop_named(prop, text, exact=False):
  sought, seen = _alnums(re.sub('([a-z])([A-Z])', '\\1 \\2', re.split('\\s*\\(', str(prop or ''))[0])), _alnums(text)
  if not sought or len(' '.join(sought)) < 3: return False
  def forms(w): return (w,) if exact or len(w) < 3 else (w, w + 's', w + 'es')
  n = len(sought)
  return any(all(a in forms(b) or b in forms(a) for a, b in zip(seen[i:i + n], sought))
       for i in range(len(seen) - n + 1))
def ref_scope(prop, source):
  try:
    keys = [w for w in _alnums(re.split('\\s*\\(', str(prop))[0]) if w not in _HL]
    if not keys: return ''
    s = str(source)
    m = re.search('\\b' + '\\W+'.join(re.escape(k) for k in keys) + '\\b', s.lower())
    return re.split('\\(\\s*or\\b|(?<=[.!?])\\s', s[m.end():m.end() + 500])[0] if m else ''
  except Exception: return ''
def _id_pairs(rcr):
  return [(k, v) for k, v in rcr.items()
      if isinstance(v, str) and 'id' in str(k).lower() and re.fullmatch(ID_RX, v)]
_PRISM = (
  ('price', lambda k, v: re.search('price|cost', k) and is_num(v),
  lambda k, v: is_num(v) and re.search('amount|fee|charge|pay|rate|value', k) and not re.search(
    'total|count|qty|quant|unit|due|sum|rating|star|score|page|number|num|size|weight|width|height|length|found|match', k)),
  ('in_stock', lambda k, v: re.search('stock|avail', k), lambda k, v: isinstance(v, bool) and re.search(
    'hand|orderable|buyable|sellable|ready|ship|instock|purchas|in_store', k)),
  ('title', lambda k, v: re.search('title|heading', k), lambda k, v: isinstance(v, str) and re.search(
    'headline|name|label|caption', k) and not re.search('shop|seller|store|vendor|brand|make|categor|shelf|department|id', k)))
def _prism_pad(rcr):
  if not _id_pairs(rcr): return
  low = {str(k).lower(): k for k in rcr}
  for name, has, can in _PRISM:
    if not any(has(lk, rcr[k]) for lk, k in low.items()) and len(
        m := [k for lk, k in low.items() if can(lk, rcr[k])]) == 1:
      rcr[name] = rcr[m[0]]
def padded(morsel):
  try: d = json.loads(json.dumps(morsel))
  except (TypeError, ValueError): return morsel
  walk_dicts(d, _prism_pad)
  return d
_GENERA = {'string': lambda v: isinstance(v, str), 'integer': lambda v: isinstance(v, int) and not isinstance(v, bool),
     'number': is_num, 'boolean': lambda v: isinstance(v, bool), 'object': lambda v: isinstance(v, dict),
     'array': lambda v: isinstance(v, list), 'null': lambda v: v is None}
def _fits_type(v, sorts): return any((q := _GENERA.get(k)) is None or q(v) for k in sorts)
def _coerce(v, sorts):
  if not sorts: return v
  fi, s = isinstance(v, float) and v.is_integer(), isinstance(v, str)
  if _fits_type(v, sorts): return int(v) if fi and 'integer' in sorts else v
  for k in sorts:
    if k == 'integer' and fi: return int(v)
    if k == 'integer' and s and re.fullmatch('\\s*-?\\d+(\\.0+)?\\s*', v): return int(float(v))
    if k == 'number' and s and re.fullmatch('\\s*-?\\d+(\\.\\d+)?\\s*', v):
      return int(f) if (f := float(v)).is_integer() else f
    if k == 'string' and is_num(v): return str(v)
    if k == 'boolean' and s and v.lower() in ('true', 'false'): return v.lower() == 'true'
  return v
def _rule_enum(v, schema, word, where):
  if v in (ok := schema[word]): return v, []
  if isinstance(v, str):
    for s in ok:
      if isinstance(s, str) and s.lower() == v.strip().lower(): return s, []
  return v, [f'{where}: must be one of {ok}']
def _rule_const(v, schema, word, at):
  req = schema[word]
  return v, [] if v == req else [f'{at}: must equal {req!r}']
def _rule_cut(v, schema, word, at): return v[:schema[word]] if isinstance(v, str) else v, []
def _lenrule(kind, bad, text):
  def rule(v, schema, word, at):
    n = schema[word]
    if kind: return v, [f'{at}: ' + text.format(n)] if isinstance(v, kind) and bad(len(v), n) else []
    return v, [f'{at}: {text} {n}'] if is_num(v) and is_num(n) and bad(v, n) else []
  return rule
def _rule_items(v, schema, word, at):
  sub = schema[word]
  if not isinstance(v, list) or not isinstance(sub, dict): return v, []
  got = [ArgCheck.vet(x, sub, f'{at}[{i}]') for i, x in enumerate(v)]
  return [x for x, _ in got], [e for _, more in got for e in more]
def _rule_fields(v, schema, word, at):
  if not isinstance(v, dict): return v, []
  props = schema.get('properties') or {}
  errs = [f'{at}.{k}: required' for k in schema.get('required') or [] if k not in v]
  if schema.get('additionalProperties') is False:
    errs += [f'{at}.{k}: not an allowed argument' for k in v if k not in props]
  out = {}
  for k, x in v.items():
    if k in props:
      x, more = ArgCheck.vet(x, props[k], f'{at}.{k}')
      errs += more
    out[k] = x
  return out, errs
_RULES = (('enum', _rule_enum), ('const', _rule_const),
         ('minLength', _lenrule(str, lambda a, b: a < b, 'shorter than {} characters')),
         ('maxLength', _rule_cut), ('minimum', _lenrule(None, lambda a, b: a < b, 'below minimum')),
         ('maximum', _lenrule(None, lambda a, b: a > b, 'above maximum')),
         ('exclusiveMinimum', _lenrule(None, lambda a, b: a <= b, 'must be above')),
         ('exclusiveMaximum', _lenrule(None, lambda a, b: a >= b, 'must be below')),
         ('minItems', _lenrule(list, lambda a, b: a < b, 'needs at least {} items')),
         ('maxItems', _lenrule(list, lambda a, b: a > b, 'allows at most {} items')),
         ('items', _rule_items), ('type', _rule_fields))

class ArgCheck:
  @classmethod
  def vet(cls, morsel, jsonspec, whereabouts='args'):
    if not isinstance(jsonspec, dict): return morsel, []
    t = jsonspec.get('type')
    sorts = [t] if isinstance(t, str) else list(t or [])
    morsel = _coerce(morsel, sorts)
    if sorts and not _fits_type(morsel, sorts): return morsel, [whereabouts + ': expected ' + '/'.join(sorts)]
    errs = []
    for word, rule in _RULES:
      if word in jsonspec:
        morsel, more = rule(morsel, jsonspec, word, whereabouts)
        errs += more
    return morsel, errs

class EmitErr:
  data = None
  def __init__(self, why): self.error = {'status': None, 'body': why[:300]}
_MODEL_FLOOR = LAST_CALL - 10
def _clamp(v, lo, hi): return int(min(max(v, lo), hi))
_VERDICTS = (('spent', lambda c, low: c == 402 or (c == 403 and re.search('key limit|insufficient|credit', low))),
        ('refused', lambda c, low: c in (401, 403, 404)), ('body', lambda c, low: c in (400, 413, 422)),
        ('client', lambda c, low: isinstance(c, int) and 400 <= c < 500 and c not in (408, 425, 429)))

class _Clock:
  def __init__(self): self.last = self.idle_time = self.model_time = 0.0
  def lag(self, for_model):
    now = _TALLY.now()
    hold = self.last + (SLOW_GAP if now < self.idle_time else PACE_GAP) - _TALLY.now()
    return max(hold, self.model_time - _TALLY.now()) if for_model else hold
  def after_ans(self, answer):
    if answer.data is None and (answer.error or {}).get('status') in (429, 503):
      self.idle_time = _TALLY.now() + SLOW_FOR

class Cable:
  def __init__(self, tiemap):
    self.z0 = self.last_reply = _TALLY.now()
    self.tiemap, self.clock = dict(tiemap), _Clock()
    self.nlm, self.env_client = (ProxyClient(timeout=t, max_retries=1, retry_delay=1) for t in (70, 95))
    self.spend, self.ticks, self.misses, self.log, self.extra_models, self.call_fail = 0.0, 0, 0, [], [], ''
    self.done = self.broken = self.hush = self.no_model = False
    self.dropped, self.no_tools, self.call_models = set(), set(), set()
  def _scale(self): return self.__dict__.get('scale', 1.0)
  def spare(self): return DEADLINE - (_TALLY.now() - self.z0)
  def spent_out(self): return self.spend >= SPEND_CAP * self._scale()
  def _tail_send(self, send, model=False):
    hold = min(self.clock.lag(model), self.spare() - 1)
    if hold > 0: _TALLY.wait_s(hold)
    self.clock.last = _TALLY.now()
    try:
      with _TALLY.alt_due(): answer = send()
    except Exception as e: answer = EmitErr(type(e).__name__ + ': ' + str(e))
    self.clock.after_ans(answer)
    return answer
  def _backoff(self, germ, tries, status):
    mass = 7 * len(germ) + sum(map(ord, germ[-64:])) + 31 * tries + 13 * self.ticks
    return min(8.0, (2.0 if status == 429 else 1.0) * 2 ** tries) * (0.75 + mass % 101 / 100.0 / 2)
  def downed(self, at=None):
    if self.no_model: return True
    if self.misses <= 0: return False
    return (_TALLY.now() if at is None else at) - self.last_reply > DOWN_AFTER
  def can_serve(self, engines, tools=False):
    if self.no_model or self.hush: return False
    bad = self.dropped | self.no_tools if tools else self.dropped
    return any(m not in bad for m in [*engines, *self.extra_models])
  @staticmethod
  def _flatten(comms):
    def flat(m):
      m = dict(m)
      parts = m.get('content')
      if isinstance(parts, list):
        m['content'] = '\n\n'.join(str(p.get('text') or '') for p in parts if isinstance(p, dict))
      return m
    return list(map(flat, comms))
  @staticmethod
  def _keyed(comms, model):
    if len(comms) <= 2 or not model.startswith('anthropic/'): return comms
    out = [dict(m) for m in comms]
    for end in (out[0], out[-1]):
      txt = end.get('content')
      if isinstance(txt, str) and txt:
        end['content'] = [dict(type='text', text=txt, cache_control=dict(type='ephemeral'))]
    return out
  def _fails(self, sorts):
    self.call_fail = 'empty' if 'empty' in sorts else 'refused' if sorts == {'hard'} else ''
  def _spendy(self, comms, engines, cap, nx):
    if self.spent_out(): return True
    if not (nx < len(engines) and 'sonnet' not in engines[nx]): return False
    est = sum(len(json.dumps(m, default=str)) for m in comms) / 2.3 * 2e-06 + min(cap, 1500) * 1e-05
    k = self._scale()
    return self.spend >= SOFT_SPEND * k or self.spend + est > SPEND_CAP * k
  def query_model(self, comms, engines, cap, tools=None):
    self.call_fail, self.call_models = '', set()
    if self.hush or self.no_model: return None
    sorts, skip, germ = set(), set(), ''
    stair = list(engines) + [m for m in self.extra_models if m not in engines]
    alone = len({m for m in stair if m not in self.dropped}) <= 1
    for pos, model in enumerate(stair):
      if not self.no_model and self.spend >= HARD_SPEND * self._scale():
        self.no_model = True
        self.log.append(dict(kind='spend_stop', spend=round(self.spend, 4)))
      if self.no_model: break
      if model in self.dropped or model in skip or tools and model in self.no_tools: continue
      if 'sonnet' in model and self._spendy(comms, engines, cap, pos + 1): continue
      plain, left = False, 1 if self.downed() else MODEL_TRIES
      while left:
        left, spare = left - 1, self.spare()
        if spare < _MODEL_FLOOR: return self._fails(sorts)
        self.nlm.timeout = _clamp(spare - LAST_CALL + 15, 8, 30 if self.downed() else 75)
        msgs = self._flatten(comms) if plain else self._keyed(comms, model)
        req = dict(model=model, messages=msgs, temperature=0, max_tokens=cap)
        if tools: req.update(tools=tools, tool_choice='auto')
        self.call_models.add(model)
        t0 = _TALLY.now()
        got = self._tail_send(lambda: self.nlm.post_verbose('/inference/chat/completions', req), True)
        if isinstance(body := got.data, dict) and body.get('choices'):
          self.last_reply, self.misses = _TALLY.now(), 0
          if is_num(cost := (body.get('usage') or {}).get('cost')): self.spend += float(cost)
          msg = body['choices'][0].get('message') or {}
          if msg.get('content') or msg.get('tool_calls'): return msg
          sorts.add('empty')
          self.log.append(dict(kind='inference_failure', model=model, error='empty'))
          break
        err, self.misses = got.error, self.misses + 1
        self.log.append(dict(kind='inference_failure', model=model, error=str(err)[:300] if err else 'empty'))
        code, text = (err or {}).get('status'), str((err or {}).get('body') or '')
        low = text.lower()
        st = next((s for s, quiz in _VERDICTS if quiz(code, low)), 'soft')
        if st == 'spent':
          self.no_model = True
          return self._fails(sorts)
        if st == 'soft':
          sorts.add('soft')
          if _TALLY.now() - t0 > 20:
            if self.downed() and (alone or len(self.call_models) >= 2): return self._fails(sorts)
            break
          germ = germ or json.dumps(comms[-1:], default=str)[-600:]
          wait = self._backoff(germ + model, min(self.misses - 1, 6), code)
          if self.spare() - wait < _MODEL_FLOOR: return self._fails(sorts)
          self.clock.model_time = _TALLY.now() + wait
          if self.downed(self.clock.model_time) and (alone or len(self.call_models) >= 2):
            return self._fails(sorts)
          continue
        sorts.add('hard')
        if st == 'body' and not plain:
          plain = True
          continue
        skip.add(model)
        if st == 'refused':
          if code == 404 and tools and 'tool' in low: self.no_tools.add(model)
          elif code != 403 or not re.search('moderat|flagged', low): self.dropped.add(model)
          hit = code == 403 and not self.extra_models and re.search(
            'allowed_models"?\\s*:\\s*\\[([^\\]]*)', text)
          if hit:
            named = re.findall('"([^"\\s]{3,100})"', hit.group(1))
            self.extra_models = [m for m in named if m not in self.dropped and m not in engines][:2]
            stair.extend(m for m in self.extra_models if m not in stair)
            alone = len({m for m in stair if m not in self.dropped}) <= 1
        break
    return self._fails(sorts)
  def consult(self, instruction, cap, engines):
    reply = self.query_model([{'role': 'user', 'content': instruction}], engines, cap)
    return as_json(reply.get('content') or '' if reply else '')
  def doing(self, doings):
    body, stalls, nudges = {**self.tiemap, 'step': self.ticks + 1, 'action': doings[0]}, 0, 0
    for _ in range(3 + WORLD_TRIES):
      spare = self.spare()
      if spare < 2: return None
      self.env_client.timeout = _clamp(spare - 1, 3, 95)
      got = self._tail_send(lambda: self.env_client.post_verbose('/environment/call', body))
      if isinstance(out := got.data, dict):
        step = out.get('step')
        self.ticks = step if isinstance(step, int) else self.ticks + len(doings)
        if out.get('finished'): self.done = True
        return out
      code, text = (err := got.error or {}).get('status'), str(err.get('body') or '')
      self.log.append(dict(kind='env_failure', status=code, body=text[:300]))
      if code == 402:
        self.broken = True
        return None
      if code == 409:
        if 'finished' in text:
          self.done = True
          return None
        hit = re.search('expected (\\d+)', text)
        if hit:
          self.ticks = int(hit.group(1)) - 1
          body['step'] = self.ticks + 1
          continue
        if 'reused' not in text or nudges >= 2: return None
        nudges += 1
        body['step'] += 1
        continue
      if code in (403, 422) or stalls >= WORLD_TRIES - 1 or code in (400, 401, 404) and stalls >= 1: return None
      wait = self._backoff(json.dumps(body, default=str)[-600:], stalls, code)
      stalls += 1
      if self.spare() - wait < 2: return None
      _TALLY.wait_s(wait)
    return None
def _rec_id(rcr):
  for k, v in rcr.items():
    if isinstance(v, str):
      low = str(k).lower()
      if low in ('id', '_id') or low.endswith('id') and re.fullmatch(ID_RX, v): return v
  return None
def _has_pmap(rcr):
  return any(isinstance(s, dict) and sum(isinstance(x, (str, int, float)) for x in s.values()) >= 2
       for s in rcr.values())

class Records:
  def __init__(self): self.items, self.ids, self.read, self.label_of = [], set(), set(), {}
  def add(self, obs):
    self.items.append(obs)
    walk_dicts(obs, self._grab)
  def _grab(self, rcr):
    self.ids.update(v for _, v in _id_pairs(rcr))
    sid = _rec_id(rcr)
    if not sid: return
    if _has_pmap(rcr): self.read.add(sid)
    for k, v in rcr.items():
      if isinstance(v, str):
        low, lab = str(k).lower(), v.strip()
        if 0 < len(lab) <= 60 and 'id' not in low and _shelfkey_rx.search(low):
          return self.label_of.setdefault(sid, lab)
  def price(self, widget):
    last = None
    def note(rcr):
      nonlocal last
      if any(isinstance(v, str) and v == widget and 'id' in str(k).lower() for k, v in rcr.items()):
        fig = next((float(v) for k, v in _named(rcr) if is_num(v) and _price_rx.search(k)), None)
        if fig is not None: last = fig
    self._each(note)
    return last
  def assoc(self, widget, hold=4):
    notes = []
    for obs in self.items:
      todo = [(obs, 0)]
      while todo:
        node, depth = todo.pop()
        if depth > 6: continue
        if isinstance(node, dict):
          if widget in [v for v in node.values() if isinstance(v, str)]:
            slip = as_text(snip(node, 400))[:2500]
            if slip not in notes:
              notes.append(slip)
              if len(notes) >= 8: return notes[-hold:]
            continue
          node = node.values()
        elif not isinstance(node, list): continue
        todo.extend((s, depth + 1) for s in reversed(node))
    return notes[-hold:]
  def _each(self, take):
    for obs in self.items: walk_dicts(obs, take)
  def cards(self, letterform_peak=30000, lowerpriced_compared='', skip=None, first=()):
    merged = {}
    def take(rcr):
      own = next((v for k, v in _named(rcr) if isinstance(v, str) and v in self.ids and 'id' in k
            and re.fullmatch(ID_RX, v)), None)
      if own is None: return
      sheet = max(_sizes(rcr), default=0) >= 2
      if sheet or len(rcr) >= 4: merged.setdefault(own, {}).update(rcr)
      if sheet: self.read.add(own)
    self._each(take)
    def price(own):
      return next((float(v) for k, v in _named(merged.get(own, {}))
            if is_num(v) and _price_rx.search(k)), None)
    anchor = lowerpriced_compared
    ceil = price(anchor) if anchor else None
    kept = [o for o in merged if o == anchor or not (skip and o in skip)
        and (ceil is None or (p := price(o)) is None or p < ceil)]
    lead = set(first or ())
    line = {o: ('DETAILS ' if o in self.read else 'ROW ') + as_text(snip(merged[o], 160)) for o in kept}
    snug = sum(map(len, line.values())) > letterform_peak
    def rank(o):
      gone = snug and any(Errand.key_stock(k, v) is False for k, v in merged[o].items())
      return o not in lead, o not in self.read, gone
    out, room = [], letterform_peak
    for o in sorted(kept, key=rank):
      if len(line[o]) <= room:
        room -= len(line[o])
        out.append(line[o])
    return out
  def label_for(self, dx):
    pairs = list(_named(dx))
    for k, v in pairs:
      if isinstance(v, str) and 'id' not in k and _shelfkey_rx.search(k) and 0 < len(v.strip()) <= 60:
        return v.strip()
    borne = self.__dict__.get('label_of') or {}
    return next((borne[v] for k, v in pairs if isinstance(v, str) and v in borne and 'id' in k), '')
  def _read_id(self, rcr):
    return next((v for k, v in _named(rcr) if isinstance(v, str) and v in self.read and 'id' in k), None)
  def dense_n(self, shelf=''):
    want, owners = shelf.strip().lower() if shelf else '', set()
    def take(rcr):
      own = self._read_id(rcr)
      if not own or shelf and self.label_for(rcr).strip().lower() != want: return
      sizes = _sizes(rcr)
      if max(sizes) >= 3 if sizes else len(_snips(rcr, 70)) >= 5: owners.add(own)
    self._each(take)
    return len(owners)
  def shelf_read(self):
    counts = {}
    def take(rcr):
      if self._read_id(rcr) and _sizes(rcr):
        lab = self.label_for(rcr)
        if lab: counts[lab] = counts.get(lab, 0) + 1
    self._each(take)
    return sorted(counts, key=lambda k: -counts[k])
  def prop_list(self, shelf=''):
    want = shelf.strip().lower() if shelf else ''
    def on_shelf(rcr):
      return any(isinstance(v, str) and _shelfkey_rx.search(k) and v.strip().lower() == shelf.lower()
           for k, v in _named(rcr)) or self.label_for(rcr).strip().lower() == want
    tally = _PropCount(bool(shelf))
    for obs in self.items:
      todo = [(obs, False, 0)]
      while todo:
        node, inside, depth = todo.pop()
        if depth > 6: continue
        if isinstance(node, dict):
          inside = inside or not shelf or on_shelf(node)
          if inside: tally.take(node)
          node = [s for s in node.values() if isinstance(s, (dict, list))]
        elif not isinstance(node, list): continue
        todo.extend((s, inside, depth + 1) for s in reversed(node))
    return tally.menu(), tally.rec_ls
  def label_lines(self):
    per = {}
    def take(rcr):
      for k, v in _snips(rcr, 40) if len(rcr) >= 4 else ():
        if not (re.fullmatch(CODE_RX, v) and re.search('\\d', v)):
          pail = per.setdefault(str(k), {})
          pail[v] = pail.get(v, 0) + 1
    self._each(take)
    lines = []
    for k, pail in per.items():
      hits = list(pail.values())
      if sum(hits) < 4 or not 2 <= len(hits) <= 8 or max(hits) < 2: continue
      lines.append(f'- {k}: ' + ' | '.join(f'{v} (x{n})' for v, n in sorted(pail.items(), key=lambda d: -d[1])))
      if len(lines) == 12: break
    return '\n'.join(lines)
_shelfkey_rx = re.compile('categor|shelf|department|aisle')
_price_rx = re.compile('price|cost')
_money_rx = re.compile('\\bprice\\b|budget|quantity|\\bstock\\b|\\bcents?\\b')
_marks_rx = re.compile('\\((?:stated|revealed[^)]*)\\)')
_paren = re.compile('\\(([^)]*)\\)')
_told_keys = ('before_finish', 'needs_message', 'before_change')
_told_tags = (('before_finish', 'before the finish'), ('needs_message', 'by a message'))
def _named(rcr): return ((str(k).lower(), v) for k, v in rcr.items())
def _sizes(rcr): return [len(v) for v in rcr.values() if isinstance(v, dict)]
def _snips(rcr, utmost): return [(k, v) for k, v in rcr.items() if isinstance(v, str) and 0 < len(v) <= utmost]
def _pos_int(v): return v if isinstance(v, int) and not isinstance(v, bool) and v >= 1 else None
def _alnum(text): return re.findall('[a-z0-9]+', text)
def _letters(text): return re.findall('[a-z]+', text)
def _flat_txt(text): return ' '.join(str(text or '').split())
def _head(need): return need.partition(':')[0]
def _longer(need, words, said):
  prop, _, value = need.partition(':')
  b = prop.find('(')
  tag = prop[b:].strip() if b >= 0 else ''
  short = _letters(re.split('\\s*\\(', prop)[0].lower())
  if not short or min(map(len, short)) < 3 or re.search('\\b' + '\\s+'.join(short) + '\\b', said): return None
  g = len(short)
  for i in range(len(words) - g + 1):
    run = words[i:i + g]
    if run != short and all(f.startswith(s) for f, s in zip(run, short)):
      return ' '.join(run) + (' ' + tag if tag else '') + ':' + value
  return None
def _spelled(need, hook):
  prop, _, value = need.partition(':')
  if re.search('\\d', value) or is_ref_text(value): return None
  low = value.lower()
  bare = ' '.join(_alnum(re.sub(CONTRAST_RX, ' ', low)))
  if not bare or bare == ' '.join(_alnum(low)) and confirm_val(value, prop): return None
  name = re.split('\\s*\\(', prop)[0].strip()
  for field, shown in (hook(prop) or {}).items():
    if exact_prop(field, name):
      hit = next((s for s in shown if ' '.join(_alnum(str(s).lower())) == bare), None)
      if hit is not None:
        TEXTUAL_DATA.add(bare)
        return f'{prop.strip()}: {str(hit).strip().lower()}'
  return None
def _subst(got, subs):
  def sw(seq): return [subs.get(x, x) if isinstance(x, str) else x for x in seq]
  got['needs'] = sw(got.get('needs') or [])
  if isinstance(got.get('requirements_now'), list): got['requirements_now'] = sw(got['requirements_now'])

class _PropCount:
  def __init__(self, parted):
    self.parted, self.rec_ls, self.full = parted, 0, 0
    self.hits, self.by_row, self.values, self.egvals = {}, {}, {}, {}
  def take(self, rcr):
    brief = [(k, v) for k, v in _snips(rcr, 70) if not junk_key(k)]
    few = len(brief) < 5
    if few and not (self.parted and len(brief) >= 2 and not _sizes(rcr)): return
    self.rec_ls += 1
    row = few and any('id' in k for k, v in _named(rcr) if isinstance(v, str))
    self.full += not row
    for k, v in brief:
      if ('-' in v or '_' in v) and re.search('\\d', v) and re.fullmatch(CODE_RX, v): continue
      name, low = re.sub('[_.]+', ' ', str(k)).strip().lower(), v.lower()
      self.hits[name] = self.hits.get(name, 0) + 1
      if row: self.by_row[name] = self.by_row.get(name, 0) + 1
      vals = self.values.setdefault(name, {})
      vals[low] = vals.get(low, 0) + 1
      ex = self.egvals.setdefault(name, [])
      if len(ex) < 5 and low not in ex: ex.append(low)
  def menu(self):
    hits = self.hits
    if self.full:
      for n in [n for n, c in hits.items() if self.by_row.get(n, 0) >= c]: del hits[n]
    def grade(n):
      total = sum((vals := self.values.get(n) or {}).values())
      return min(1.0, hits[n] / max(1, self.rec_ls)) * (1.0 - (max(vals.values()) / total if total else 1.0))
    lines = []
    for n in sorted(hits, key=lambda n: (-hits[n], -grade(n), n))[:28]:
      ex = self.egvals.get(n, [])
      note = f' [varies: {len(ex)} distinct values]' if len(ex) >= 2 and hits[n] >= 2 else ''
      if ex and all(re.search('\\d', s) for s in ex) and any(re.search('\\d\\s*[a-z\\"\']', s) for s in ex):
        note += ' [figures with a unit]'
      lines.append(f'- {n} (in {hits[n]} listings){note}: ' + ' | '.join(ex))
    return '\n'.join(lines)

class Wants:
  def __init__(self, cable, request):
    self.cable, self.request, self.parsed = cable, request, {}
    self.written, self.uncovered, self.indifferent, self.let_go = [], [], [], []
    self.cur_reqs = self.spend_cap = None
    self.rev_no, self.caught, self.trades, self.reply_due = 0, set(), [], ''
    self.grown_at, self.grown_tick, self.can_pg = -1, -10 ** 6, False
    self.resolved, self.ref_news, self.tolds, self.told_rev, self.allowance_news = [], [], [], 0, None
  def add_told(self, items, source):
    if not isinstance(items, list): return []
    held = {_flat_txt(e.get('text')).lower() for e in self.tolds}
    added = []
    for raw in items[:12]:
      got = {'text': raw} if isinstance(raw, str) else raw
      if not isinstance(got, dict): continue
      txt = _flat_txt(got.get('text'))
      if not txt or txt.lower() in held: continue
      held.add(txt.lower())
      flags = {m: got.get(m) if isinstance(got.get(m), bool) else None for m in _told_keys}
      self.tolds.append(dict(text=txt[:400], source=source, **flags))
      added.append(txt[:120])
    self.told_rev += bool(added)
    return added
  def finish_told(self): return [e for e in self.tolds if e.get('before_finish') is not False]
  def told_text(self):
    rows = []
    for n, e in enumerate(self.tolds[-10:], 1):
      tags = [lab for m, lab in _told_tags if e.get(m)]
      origin = 'from the ' + str(e.get('source'))
      row = f'{n}) {e["text"]} [' + ('; '.join([', '.join(tags), origin]) if tags else origin) + ']'
      if vars(self).get('basket_told') and e.get('before_finish') and e.get('needs_message'):
        row += ' [carried out: the shopper was told what the cart now holds and answered by noting it, with no objection and no change asked; that answer is the approval this step wanted, so placing the order comes next]'
      rows.append(row)
    return '\n[instructions in force] ' + ' '.join(rows) if rows else ''
  def items(self):
    rows = []
    for row in self.parsed.get('items') or []:
      if not isinstance(row, dict) or not str(row.get('product') or '').strip(): continue
      asked, dep = row.get('requirements'), _pos_int(row.get('depends_on'))
      rows.append({'product': str(row['product']).strip(), 'quantity': _pos_int(row.get('quantity')) or 1,
            'requirements': [str(r) for r in (asked if isinstance(asked, list) else []) if r and ':' in str(r)],
            'depends_on': None if dep is None else dep - 1})
    return rows if len(rows) >= 2 else []
  def quantity(self): return 1 if self.items() else _pos_int(self.parsed.get('quantity')) or 1
  def counts(self): return [e['quantity'] for e in self.items()] or [self.quantity()]
  def priority(self):
    rule = self.parsed.get('priority')
    return rule if rule in ('cheapest', 'best') else 'none'
  def props_left(self):
    stash = self.cur_reqs if self.cur_reqs is not None else self.written + self.uncovered
    return [x for x in stash if x not in self.let_go and not _money_rx.search(x.lower())]
  def body_text(self):
    drop = self.let_go
    if self.cur_reqs is not None: segs = list(self.cur_reqs)
    else:
      segs = [f'{x} (stated)' for x in self.written if x not in drop]
      segs += [f'{x} (revealed by the shopper)' for x in self.uncovered if x not in drop]
    parts = ['; '.join(segs) if segs else 'see the request' if not drop else
        'no property left to match (every one was dropped): any in-stock listing in the requested category will do']
    if self.spend_cap is not None:
      parts.append(f'budget, whole order, upper limit: {self.spend_cap} {self.parsed.get("currency") or ""}'.rstrip())
    rows = self.items()
    if rows:
      cat = ' | '.join(f'{n}) {e["product"]} x{e["quantity"]}' for n, e in enumerate(rows, 1))
      parts.append(f"ITEMS IN THIS ORDER ({len(rows)}): {cat}. Cart rule: exactly one line per item at that item's own count, nothing more. A listing is judged only by the needs of the item whose category it matches; a cheapest or best-option wish applies inside each item, never between items.")
    else: parts.append(f'quantity wanted: {self.quantity()}')
    if drop: parts.append('dropped by the shopper: ' + '; '.join(drop))
    if self.indifferent: parts.append('no preference on: ' + ', '.join(self.indifferent))
    return '; '.join(parts)
  def var_opts(self):
    if not self.let_go: return ''
    return '\n\nLATER DROPS (the shopper took these back after the request was written; they override its text): ' + '; '.join(self.let_go) + '. A withdrawn requirement no longer exists: no listing is rejected, ranked or favoured on any of them.'
  def rq_view(self, stride_cardinality, comms):
    got = self.parsed
    if not got: return ''
    rows = self.items()
    def dots(key, lead): return [f'- {lead}: {e}' for e in got.get(key) or []]
    lines = ['REQUEST NOTES (a first-pass parse; where it disagrees with the request text, trust the text)']
    if got.get('product'): lines.append(f'- product: {got.get("product")}')
    if rows:
      lines += [f'- item {n}/{len(rows)}: {e["product"]} | count {e["quantity"]} | has a cart row of its own'
           for n, e in enumerate(rows, 1)]
      lines.append('- cart: only these item lines, nothing more; if the cart tool takes a whole list, send the complete set on every call')
    else: lines.append(f'- quantity: {self.quantity()}')
    lines += dots('requirements', 'firm need')
    lines += dots('references', 'need tied to another listing (find that listing, read the field it names, and search with the value found there)')
    if got.get('price_cap') is not None:
      lines.append(f'- spend ceiling: {got.get("price_cap")} {got.get("currency") or ""}'.rstrip())
    if got.get('in_stock_only'): lines.append('- availability: only goods in stock now')
    rule = self.priority()
    if rule != 'none':
      spec = f' ({got.get("priority_detail")})' if got.get('priority_detail') else ''
      lines.append(f'- pick: the {rule}{spec} fit')
    else:
      lines.append('- choice rule: none given. Any pick meeting every need inside the budget is right; hunting further for a cheaper one only wastes steps, unless the request text asks for it')
    lines += dots('soft_wants', 'soft wish, used only to break ties') + dots('ask_first', 'ask the shopper first')
    lines += dots('rules', 'rule')
    if comms is not None: lines.append(f'- message allowance: {comms}')
    lines.append(f'- step allowance: {stride_cardinality}')
    return '\n'.join(lines)
  def drop_named(self, outdated_demand, rowlines):
    try:
      need = outdated_demand
      prop = str(need).split(':', 1)[0]
      tags = _paren.findall(prop)
      def same(o):
        p = _head(o)
        return same_prop(p, prop) and (not tags or _paren.findall(p) in ([], tags))
      held = [x for x in self.written if same(x)]
      if not held:
        fresh = [n for n in vars(self).get('fresh_needs') or [] if isinstance(n, str) and ':' in n]
        held = [x for x in self.uncovered if same(x) and x not in fresh]
        if not held or any(same(n) and n != need for n in fresh): return True
      words = _letters(' '.join(rowlines).lower())
      runs = {' '.join(words[i:i + n]) for n in (1, 2, 3) for i in range(len(words) - n + 1)}
      phrases = {r for r in runs if len(r) >= 3 and r not in STOP_WORDS}
      if any(same_prop(_head(x), r) for x in held for r in phrases): return True
      hook = vars(self).get('field_hook')
      for line in held if hook else ():
        p, _, raw = line.partition(':')
        value = _marks_rx.sub(' ', raw)
        if is_ref_text(value) or not vsubs(value): continue
        fields = hook(p) or {}
        rivals = [_head(r) for r in self.written + self.uncovered
             if isinstance(r, str) and ':' in r and not same_prop(_head(r), prop)]
        for run in phrases:
          if any(exact_prop(c, run) for c in rivals): continue
          for field, shown in fields.items():
            if exact_prop(field, run) and val_seen(value, ' | '.join(shown)):
              self.cable.log.append({'kind': 'drop_by_field', 'need': line[:200], 'phrase': run})
              return True
      return False
    except Exception: return True
  def tie_to(self, quality_appellation, qtxt):
    try:
      got = self.parsed.get('references')
      refs = [str(h) for h in got] if isinstance(got, list) else []
      refs += vars(self).get('resolved') or []
      asked = quality_appellation
      tags, posed = _paren.findall(str(asked)), prop_key(asked)
      for req in self.written:
        if req in self.let_go or ':' not in req: continue
        prop, _, value = req.partition(':')
        own = _paren.findall(prop)
        if tags and own and not kind_words(tags[-1]) & kind_words(own[-1]): continue
        if not confirm_val(value, prop) or story_guess(req, self.request, refs): continue
        mine = prop_key(prop)
        if same_prop(prop, asked) or len(mine) >= 4 and posed.endswith(mine): return True
      return False
    except Exception: return False
  @staticmethod
  def valued(needs, rowlines):
    said = ' '.join(map(str, rowlines)).lower()
    def full(need):
      prop, _, value = str(need).partition(':')
      want = [w for w in prop_words(prop) if len(w) >= 3]
      given = [w for w in _alnum(value.lower()) if w not in STOP_WORDS]
      return bool(want and given) and all(w in said for w in want + given)
    return isinstance(needs, list) and any(full(n) for n in needs)
  def _qe_log(self, kind, **fields): self.cable.log.append(dict(kind=kind, **fields))
  def heed(self, heard, asked):
    fresh = [x for x in heard if x not in self.caught]
    if not fresh: return ''
    self.caught.update(fresh)
    c, log = self.cable, self._qe_log
    if c.spare() < CLOSING: return ''
    segs = [HEED, 'THE ORIGINAL ASK (BACKGROUND):\n' + self.request,
        'REQUIREMENTS SO FAR:\n' + (trunk := self.body_text())]
    if self.trades: segs.append('EARLIER BACK-AND-FORTH, OLDEST FIRST:\n' + '\n'.join(self.trades[-12:]))
    if asked: segs.append('WHAT THE ASSISTANT SENT:\n' + asked)
    segs.append("THE SHOPPER'S LATEST WORDS:\n" + '\n'.join(fresh))
    self.trades.append((f'asked: {asked} -> ' if asked else '') + 'shopper: ' + ' / '.join(fresh))
    got = c.consult('\n\n'.join(segs), 2500, LISTENER)
    if not got:
      if c.hush or c.misses > 0:
        self.caught.difference_update(fresh)
        del self.trades[-1]
        self.pending = [*(vars(self).get('pending') or []), (list(fresh), asked)]
        self.pending_at = _TALLY.now()
      return ''
    try:
      if got.get('meta_only') is True and asked and not self.valued(got.get('needs'), fresh):
        got = dict(got, **{k: f() for k, f in _QUIET_FLDS})
        self.meta_last = asked
        log('meta_only', line=' / '.join(fresh)[:200])
    except Exception: pass
    try:
      said = ' '.join(heard := _letters(' '.join(fresh).lower()))
      subs = {n: v for n in got.get('needs') or [] if isinstance(n, str) and ':' in n
          and (v := _longer(n, heard, said)) is not None}
    except Exception: subs = {}
    if subs:
      _subst(got, subs)
      log('need_renamed', appellations=list(subs.values())[:4])
    shroud, reader = fresh, vars(self).get('ref_hook')
    if reader:
      try: shroud = reader(got, fresh, asked) or fresh
      except Exception as e: log('ref_line_error', error=_err_txt(e))
    voiced = ' '.join([trunk, ' '.join(self.trades), asked or '']).lower()
    try:
      hook = vars(self).get('field_hook')
      subs = {n: v for n in (got.get('needs') or [] if hook else ()) if isinstance(n, str) and ':' in n
          and (v := _spelled(n, hook)) is not None}
      if subs:
        _subst(got, subs)
        c.log.append({'kind': 'literal_value', 'needs': list(subs.values())[:4]})
    except Exception as e: log('literal_value_error', error=_err_txt(e))
    needs, notes, wide = got.get('needs') or [], [], False
    self.fresh_needs = [x for x in needs if isinstance(x, str)]
    if isinstance(now := got.get('requirements_now'), list) and all(isinstance(x, str) for x in now):
      self.cur_reqs = [x for x in now if x.strip() and _backed(x, voiced)]
    for x in needs:
      if isinstance(x, str) and x and x not in self.uncovered and _backed(x, voiced):
        self.uncovered.append(x)
        notes.append('need added: ' + x)
    try:
      refs = got.get('references')
      infos = [h for h in refs if isinstance(h, dict)] if isinstance(refs, list) else []
      for x in needs:
        if isinstance(x, str) and x in self.uncovered and (
            is_ref_text(x.partition(':')[2]) or x in (got.get('ref_needs') or [])):
          self.ref_news = [*(vars(self).get('ref_news') or []),
                  {'need': x, 'infos': infos, 'line': ' '.join(fresh)}]
    except Exception: pass
    into = self.cur_reqs if self.cur_reqs is not None else self.written + self.uncovered
    ruled = {r.partition(':')[0].strip().lower() for r in into
        if isinstance(r, str) and ':' in r and r not in self.let_go}
    for x in got.get('no_preference') or []:
      if not isinstance(x, str) or not x or x in self.indifferent or x.strip().lower() in ruled: continue
      if self.tie_to(x, asked): self.pinned_heard = vars(self).get('pinned_heard', 0) + 1
      else: self.indifferent.append(x)
    for x in got.get('dropped') or []:
      if not isinstance(x, str) or not x or x in self.let_go: continue
      if is_num(got.get('new_budget')) and _MONEY_RX.search(x.lower()): continue
      if not self.drop_named(x, shroud):
        log('drop_kept', need=x[:200])
        continue
      self.let_go.append(x)
      notes.append('need dropped: ' + x)
      wide = wide or self.priority() != 'none'
    if is_num(cap := got.get('new_budget')) and cap > 0 and cap != self.spend_cap:
      self.spend_cap = cap
      notes.append(f'budget is now {cap}')
    if got.get('unavailable'):
      notes.append('a product has stopped being sold - take its line out of the cart, and never check it out')
    try:
      added = self.add_told(got.get('instructions'), 'store' if asked is None else 'shopper')
      if added: notes.append('instruction added: ' + '; '.join(added))
      caps = got.get('allowance')
      nums = {k: v for k, v in caps.items() if is_num(v)} if isinstance(caps, dict) else {}
      if nums:
        self.allowance_news = nums
        notes.append('limit changed: ' + json.dumps(nums))
    except Exception as e: log('told_error', error=_err_txt(e))
    if notes:
      self.rev_no += 1
      if wide: self.grown_at, self.grown_tick = self.rev_no, c.ticks
    if (rqs := self.cur_reqs) is not None:
      for x in self.uncovered:
        prop = x.partition(':')[0].strip().lower()
        if prop and not any(prop in r.lower() for r in rqs) and not any(prop in o.lower() for o in self.let_go):
          rqs.append(x)
      for x in self.written:
        try:
          if x in self.let_go or x in rqs or self.drop_named(x, shroud): continue
          head, tags = _head_tags(x)
          pool = self.let_go + [r for r in rqs if isinstance(r, str)]
          if not any(same_prop(head, h) and t == tags for h, t in map(_head_tags, pool)):
            rqs.append(x)
        except Exception: continue
    if wide:
      if self.props_left(): notes.append(_WIDE_NOTE)
      elif self.priority() == 'cheapest' and self.can_pg:
        notes.append(_OPEN_SHELF + _SHELF_SWEPT + _SHELF_THIS + _SHELF_LAST)
      else: notes.append(_OPEN_SHELF + _SHELF_THIS + _SHELF_GET)
    log('heard', question=asked is not None, needs=len(got.get('needs') or []), free=len(got.get('no_preference') or []),
      dropped=len(got.get('dropped') or []), budget=is_num(cap))
    back = asked is None and any('?' in x for x in fresh) and not got.get('no_preference')
    self.reply_due = str(got.get('reply') or '').strip() if back else ''
    return 'Meaning: ' + str(got.get('meaning') or '') + (' (' + '; '.join(notes) + ')' if notes else '')
_QUIET_FLDS = (('needs', list), ('no_preference', list), ('dropped', list), ('references', list),
          ('requirements_now', lambda: None), ('new_budget', lambda: None),
          ('unavailable', bool), ('reply', str))
_MONEY_RX = re.compile('\\b(?:prices?|budget|costs?|spend\\w*|cents?|dollars?)\\b')
_OPEN_SHELF = 'SHELF OPEN - no property requirement is left, so FITS = every in-stock listing filed under the requested category (accessories, parts and sets filed there count as well), and the one lowest price across the whole shelf wins. ORDER OF ROWS = by relevance or by id, never by price, so do not assume one'
_SHELF_SWEPT = "; further slices of that shelf are fetched for you by code (pages of the category listing, the label's words, object-kind words seen in titles); if no '[shelf sweep done]' list shows up after this"
_SHELF_THIS = '; cover it yourself: read page 1 of short searches for the category label, for every single kind word in the label and for every item name that recurs across titles, plus the pages of the category listing.'
_SHELF_LAST = " FINISH = make the shelf's cheapest affordable row that is still in stock the cart's only line, confirm its stock and price one more time, then check out"
_SHELF_GET = ' FINISH = order the lowest-priced item that is in stock'
_WIDE_NOTE = 'WIDER NOW - that requirement no longer applies: a listing turned down only because of it qualifies again, and listings not seen yet may beat the present choice. NEXT = search again for that kind of item without it: one short query per value still required, the value first and the kind word after it, every page read; then the cheapest qualifier of them all wins'
def _err_txt(e): return f'{type(e).__name__}: {e}'[:300]
def _backed(req, voiced):
  words = re.findall('[a-z]{3,}', req.lower())
  return not words or any(w in voiced for w in words)
def _head_tags(req):
  head = req.partition(':')[0]
  return head, re.findall('\\(([^)]*)\\)', head)
def _kind(spec): return spec.get('type') if isinstance(spec, dict) else None
def _first_ok(found): return next((d for d in found if d[1] is None or d[1] > 0), found[0] if found else None)
def _qe_qnt(qty): return int(qty) if float(qty).is_integer() else qty
_QTY_RX = re.compile('qty|quant|count|units|number|total|held|ordered')
_COST_RX = re.compile('price|cost|due|sum|fee|charge')

class CartLog:
  def __init__(self, tool, schema):
    self.tool, self._spec = tool, (schema or {}).get('properties') or {}
    self.add_form = self.incl = self.last_view = None
    self.used, self.changed_at, self.seen_at, self.units, self._guess = False, 0, -1, {}, {}
  def _enum(self, key):
    s = self._spec.get(key)
    return isinstance(s, dict) and 'enum' in s
  def form_of(self, args): return as_text({k: args[k] for k in args if self._enum(k)})
  def line_spec(self):
    for prm, spec in self._spec.items():
      if _kind(spec) != 'array': continue
      row = spec.get('items')
      fields = row.get('properties') if isinstance(row, dict) else None
      if not isinstance(fields, dict): continue
      names = [f for f, s in fields.items() if _kind(s) == 'string' and 'enum' not in s]
      sums = [f for f, s in fields.items() if _kind(s) in ('integer', 'number')]
      ids = [f for f in names if 'id' in f.lower()] or names
      if ids and sums: return prm, ids[0], sums[0]
    return None
  def _rows(self, args):
    spec = self.line_spec()
    return spec if spec and isinstance(args.get(spec[0]), list) else None
  def lines(self, args):
    spec = self._rows(args)
    if spec is None:
      t = next((v for k, v in args.items() if isinstance(v, str) and not self._enum(k)
           and v.strip() and ' ' not in v.strip()), None)
      return [(t, self.count_arg(args))] if t else []
    prm, sid, qk = spec
    return [(r[sid].strip(), float(r.get(qk)) if is_num(r.get(qk)) else None) for r in args[prm]
        if isinstance(r, dict) and isinstance(r.get(sid), str) and r[sid].strip()]
  def aim(self, args):
    lead = _first_ok(self.lines(args))
    return None if lead is None else lead[0]
  def count_arg(self, args):
    if self._rows(args):
      lead = _first_ok(self.lines(args))
      return None if lead is None else lead[1]
    for k, spec in self._spec.items():
      if _kind(spec) in ('integer', 'number') and 'enum' not in spec:
        v = args.get(k, spec.get('default'))
        if is_num(v): return float(v)
    return None
  def is_add(self, args): return self.add_form is None or self.form_of(args) == self.add_form
  def held(self, label): return self.units.get(label, 0.0)
  def booked(self, args, reply):
    self.used = True
    if isinstance(reply, dict) and reply.get('error'): return ''
    if self.add_form is None: self.add_form = self.form_of(args)
    rows = self._rows(args)
    if rows:
      self.incl, tote = False, {}
      for lab, v in self.lines(args):
        if v is None or v > 0: tote[lab] = tote.get(lab, 0.0) + (1.0 if v is None else v)
      self.units = tote
      shown = self.view_counts(reply)
      if shown is not None: self.infer(shown)
      return self.recap()
    lab = self.aim(args)
    if not lab: return ''
    v = self.count_arg(args)
    if v is None: v = 1.0
    units = self.units
    prev = units.get(lab, 0.0)
    placing = self.is_add(args)
    if not placing: units[lab] = max(0.0, prev - v) if v > 0 else 0.0
    elif prev > 0 and self.incl is not False:
      units[lab] = prev + v
      self._guess = {lab: v}
    else: units[lab] = v if v > 0 else 0.0
    if units[lab] <= 0: del units[lab]
    shown = self.view_counts(reply)
    if shown is not None:
      if set(shown) <= {lab}: self.settle(lab, shown.get(lab, 0.0))
      else: self.infer(shown)
    elif (v <= 0 or not placing) and self._empty(reply, lab):
      self.units.pop(lab, None)
      self._guess.pop(lab, None)
    return self.recap()
  def _empty(self, reply, label):
    if not isinstance(reply, dict) or label not in self.units: return False
    lst = num = False
    for v in reply.values():
      if isinstance(v, list):
        if v: return False
        lst = True
      elif is_num(v):
        if v != 0: return False
        num = True
    if not (lst and num): return False
    text = as_text(reply)
    return all(o == label or o not in text for o in self.units)
  def view_counts(self, reply):
    known, found = set(self.units) | set(self._guess), {}
    def skim(rcr):
      labs = [v for v in rcr.values()
          if isinstance(v, str) and ' ' not in v and (v in known or re.fullmatch(ID_RX, v))]
      if labs:
        lab = next((v for v in labs if v in known), labs[0])
        n = self._qty(rcr, lab)
        if n is not None: found[lab] = found.get(lab, 0) + n
    walk_dicts(reply, skim)
    return None if not found or known and not known & set(found) else found
  def _qty(self, rcr, lab):
    st = [(k, v) for k, v in rcr.items()
       if is_num(v) and isinstance(v, int) and 0 < v < 1000 and str(k) != 'price']
    if len(st) > 1:
      named = [(k, v) for k, v in st if _QTY_RX.search(str(k).lower())
          and not _COST_RX.search(str(k).lower())]
      if len(named) == 1: st = named
    if len(st) > 1:
      want = self.units.get(lab, self._guess.get(lab))
      same = [v for k, v in st if want is not None and abs(v - want) < 1e-09]
      return same[0] if len(same) == 1 else min(v for k, v in st)
    return st[0][1] if st else None
  def _incl(self, lab, shown, guess):
    if abs(shown - guess) < 1e-09 and abs(self.units.get(lab, 0) - guess) > 1e-09: self.incl = False
    elif shown > guess: self.incl = True
  def settle(self, label, volume):
    if label in self._guess: self._incl(label, volume, self._guess.pop(label))
    if volume > 0: self.units[label] = volume
    else: self.units.pop(label, None)
  def infer(self, shown):
    for lab, g in self._guess.items():
      if lab in shown: self._incl(lab, shown[lab], g)
    self._guess, self.units = {}, {k: v for k, v in shown.items() if v > 0}
  def beheld(self, reply, ticks):
    self.last_view, self.seen_at = reply, ticks
    shown = self.view_counts(reply)
    if shown is not None: self.infer(shown)
    elif isinstance(reply, (dict, list)):
      text = as_text(reply)
      if not any(k in text for k in self.units): self.units = {}
  def recap(self):
    body = ', '.join(f'{k} x{_qe_qnt(v)}' for k, v in self.units.items())
    return '[cart, counted by code] ' + (body or 'no lines') + '.'
  def faults(self, sought, perunit_article=None):
    units = self.units
    if not units: return []
    def lot(): return ', '.join(f'{k} x{int(v)}' for k, v in units.items())
    if perunit_article and len(perunit_article) > 1:
      have = sorted(map(int, units.values()))
      if len(perunit_article) != len(units):
        return [f'the basket does not match the request: {len(perunit_article)} products were asked for but {len(units)} line(s) are in it ({lot()}); keep exactly one entry per requested product']
      want = sorted(perunit_article)
      return [] if have == want else [f'line quantities are {have} but {want} are wanted: fix them']
    if len(units) > 1:
      return [f'the basket holds {len(units)} separate lines ({lot()}); remove every one except your pick']
    (lab, v), = units.items()
    gap = v - sought
    if not abs(gap) > 1e-09: return []
    return [f'wrong quantity on {lab}: {int(v)} in the cart, {sought} wanted - {"remove" if gap > 0 else "add"} {int(abs(gap))} with the cart tool']
def _guard(fallback):
  def envelope(technique):
    def run(*args, **kwopts):
      try: return technique(*args, **kwopts)
      except Exception: return fallback
    run.__name__, run.__qualname__, run.__doc__ = technique.__name__, technique.__qualname__, technique.__doc__
    return run
  return envelope

class TaskCore:
  def __init__(self, freight):
    env = freight['environment']
    cable = self.cable = Cable(env['binding'])
    view = self.pvw = env['policy_view']
    self.request = str(view.get('query') or '')
    self.tool_specs = [e for e in view.get('tools') or []
               if isinstance(e, dict) and isinstance(e.get('function'), dict)]
    self.argspec = {e['function']['name']: e['function'].get('parameters') or {'type': 'object', 'properties': {}}
            for e in self.tool_specs}
    self.step_cap = int(view.get('max_steps') or 30)
    cable.done = bool(view.get('finished'))
    self.rec_db = Records()
    book = self.needs = Wants(cable, self.request)
    book.ref_hook = self.ref_fix
    vars(self).update(role_map={}, msgs_sent=0, msg_cap=None, orders_tried=0, refusal_n=0, order_vetted=False,
             rank_checked=set(), decline_vetted=False, audit_done=False, questions_log=[], asking=True,
             rulings={}, checked_adds=set(), passed=set(), refused_lines=set(), refusals_seen={},
             warned=set(), word_seen=[], word_kind='', lead_seen=[], lead_bounds={}, lead_words=0,
             lead_ctx=None, last_calls=[], chat=[], stuck=0, refs=[])
    self.barrow = CartLog(None, {})
    TEXTUAL_DATA.clear()
    EMPORIUM_DOUBLES.clear()
    book.field_hook = self.store_fields
  @property
  def log(self): return self.cable.log
  def steps_rem(self): return self.step_cap - self.cable.ticks
  def role(self, kind): return self.role_map.get(kind)
  def ask_cap(self): return ASK_DEFAULT if self.msg_cap is None else self.msg_cap
  # a message is held back only for a stated finishing step that may need one (no such step: every message may ask)
  def msg_reserve(self): return 1 if self.role('message') and any(e.get('needs_message') is not False for e in self.needs.finish_told()) else 0
  def ask_room(self): return max(0, self.ask_cap() - self.msg_reserve())
  def step_reserve(self): return FINISH_STEPS if self.role('message') or self.needs.finish_told() else 0
  def apply_slack(self, maximums):
    before, g, beat = (self.msg_cap, self.step_cap), maximums.get, self.cable.ticks
    tot, left = g('messages_total'), g('messages_left')
    if is_num(tot): self.msg_cap = max(self.msgs_sent, int(tot))
    elif is_num(left): self.msg_cap = self.msgs_sent + max(0, int(left))
    tot, left = g('steps_total'), g('steps_left')
    if is_num(tot) and int(tot) > beat: self.step_cap = min(self.step_cap, int(tot))
    elif is_num(left) and int(left) >= 0: self.step_cap = min(self.step_cap, beat + int(left))
    caps = [self.msg_cap, self.step_cap]
    if tuple(caps) == before: return
    self.asking = self.asking and self.msgs_sent < self.ask_room()
    self.log.append({'kind': 'allowance', 'news': maximums, 'caps': caps})
  def call_kind(self, name, args):
    fin, dec = self.role('order'), self.role('decline')
    if not name or name not in (fin, dec): return None
    if fin != dec: return 'order' if name == fin else 'decline'
    modes = [v.lower() for v in (args or {}).values() if isinstance(v, str) and len(v) <= 40]
    for m in modes:
      if re.search(NO_BUY_WORDS, m): return 'decline'
      if re.search(ORDER_WORDS, m): return 'order'
    for spec in ((self.argspec.get(name) or {}).get('properties') or {}).values():
      ch = [str(o).lower() for o in spec.get('enum') or []] if isinstance(spec, dict) else []
      for m in modes if len(ch) == 2 else ():
        if m in ch:
          other = ch[1 - ch.index(m)]
          if re.search(ORDER_WORDS, other): return 'decline'
          if re.search(NO_BUY_WORDS, other): return 'order'
    return 'order'
  def end_tools(self): return {t for t in map(self.role, ('order', 'decline')) if t is not None}
  def _qf_jot(self, kind, **fields): self.cable.log.append({'kind': kind, **fields})
  def _ferr(self, kind, mistake): self.cable.log.append({'kind': kind, 'error': _err_txt(mistake)})
  def state_line(self):
    left = self.steps_rem()
    sent = f'messages sent {self.msgs_sent}' + (f' of {self.msg_cap}' if self.msg_cap is not None else '')
    counts = [f'[status] steps used {self.cable.ticks} of {self.step_cap}, {left} left', sent]
    t = max(0.0, self.cable.spare() - LAST_CALL)
    if 2 * t <= DEADLINE - LAST_CALL:
      n = CLOCK_STEP * int(t // CLOCK_STEP)
      counts.append(f'about {n} s left' if n > 0 else f'under {CLOCK_STEP} s left')
    note = '; '.join(counts) + '.\n[requirements now] '
    self.needs.basket_told = self._told_now()
    for f in (self.needs.body_text, self.ref_line, self.prod_steps, self.needs.told_text):
      note += f()
    try:
      for f in (self.bound_hint, self.dep_hint): note += f()
    except Exception: pass
    if self.msg_reserve() and self.msgs_sent < self.ask_cap():
      note += ' (the planned questions keep one message free for the stated finishing step: you may send it yourself, or it goes out for you at the finish)'
    usable = [s['function']['name'] for s in self.on_offer()]
    if len(usable) < len(self.tool_specs):
      more = '; the hidden-need questions are planned and sent for you as well' if self.asking else ''
      note += f'\n[tools usable now] {", ".join(usable)} (a call to any other tool is refused and costs no step{more}).'
    if self.barrow.used: note += '\n' + self.barrow.recap()
    juncture, due = self._urgency(left)
    return note + self._wrap_note(juncture, bool(self.needs.items()), due) if juncture else note
  def _urgency(self, left):
    if left <= 2: return 'last', 0
    if left <= 5 or self.cable.spare() < CLOSING: return 'finish', 0
    if self.cable.spent_out(): return 'budget', 0
    if self.orders_tried == 0:
      due = self.order_due()
      if left - due <= 7 and self.endgame(): return 'low' if left - due > 0 else 'closed', due
    return '', 0
  @staticmethod
  def _wrap_note(juncture, multi, due):
    fallback = 'a verified fit, failing that the cheapest in-stock listing within budget whose record shows no requirement failing'
    never = ' Decline only if the records prove that no listing can fit.'
    if juncture == 'last':
      who = 'every product asked for' if multi else 'the item'
      return f' LAST STEPS: if the cart holds a candidate for {who}, order it now; if not, put the best candidate in ({fallback}) and order.' + never
    if juncture == 'finish':
      lines = 'a line for each requested product' if multi else 'one line'
      return f' FINISH NOW: bring the cart to {lines} ({fallback}; take out any line without stock), re-check its stock, then order.' + never
    if juncture == 'budget':
      who = 'each requested product (a line apiece, at its quantity)' if multi else 'the item asked for'
      short = 'some product has' if multi else 'the store has'
      return f' MODEL BUDGET USED: exploring is over; within the next few steps put the best candidate for {who} in the cart (a verified fit if one exists, otherwise the closest fit the records show), re-check its stock and price, and order. Declining is right only when the records show that {short} no listing able to meet its requirements.'
    if juncture == 'low':
      return f' Few steps are left: in the coming steps add your strongest verified fit to the cart and order; searches and detail reads stop once {due} steps remain, and a refused order takes about 4 steps to repair.'
    return " Searching is over (details only for the cart's lines, for listings a check named, or - while some product still lacks a verified fit - for up to 3 more reads or searches in total): put your best verified fit in the cart, re-check its stock and price, and order; a refused order still takes about 4 steps to repair."
  def read_ask(self):
    inventory = '\n'.join(f'- {f["name"]}: {f.get("description", "")} | arguments: {as_text(f.get("parameters") or {})}'
               for f in [s['function'] for s in self.tool_specs])
    shown = 'REQUEST:\n' + self.request
    if self.pvw.get('opening') is not None:
      shown += '\n\nOPENING OBSERVATION:\n' + as_text(self.pvw.get('opening'))
    self.read_prompt = [dict(type='text', text=READ_ASK + '\n\nTOOLS:\n' + inventory,
                 cache_control={'type': 'ephemeral'}), dict(type='text', text=shown)]
    stair, walk = SKIMMER + LISTENER[:1], {'state': 'ask', 'turn': 0, 'got': {}}
    while walk['state'] != 'done': walk['state'] = getattr(self, '_smf_read_' + walk['state'])(walk, stair)
  def _smf_read_ask(self, walk, stair):
    turn, c = walk['turn'], self.cable
    if turn >= 3 * len(stair) or turn >= len(stair) and (c.misses == 0 or c.call_fail == 'refused' or c.downed()
                              or not c.can_serve(stair) or c.spare() < CLOSING + 60):
      return 'fallback'
    model = stair[turn % len(stair)]
    reply = c.query_model([{'role': 'user', 'content': self.read_prompt}], [model], SKIM_TERMS) or {}
    walk['got'] = as_json(reply.get('content') or '') or {}
    if isinstance(walk['got'].get('roles'), dict): return 'apply'
    self._qf_jot('setup_retry', model=model)
    walk['turn'] = turn + 1
    return 'ask'
  def _smf_read_fallback(self, walk, stair):
    try:
      got = {**self.code_reading()}
      got.update((k, v) for k, v in walk['got'].items() if k != 'roles')
    except Exception as e:
      self._ferr('code_reading_error', e)
      got = {}
    self.read_at = _TALLY.now()
    self._qf_jot('code_reading', roles=got.get('roles'), product=got.get('product'), price_cap=got.get('price_cap'),
          items=len(got.get('items') or []))
    walk['got'] = got
    return 'apply'
  def _smf_read_apply(self, walk, stair):
    self.use_reads(walk['got'])
    return 'done'
  def use_reads(self, plan, encore=False):
    for step, err in (('_smf_take_roles', None), ('_smf_take_counts', None),
             ('_smf_take_written_counts', 'written_counts_error'), ('_smf_seed_told', 'told_seed_error'),
             ('_smf_scale_products', None), ('_smf_split_written', None),
             ('_smf_run_story', 'story_values_error'), ('_smf_run_refs', 'ref_register_error'),
             ('_smf_trim_asks', None), ('_smf_take_price', None), ('_smf_take_steps', None)):
      try: getattr(self, step)(plan, encore)
      except Exception as e:
        if err is None: raise
        self._ferr(err, e)
  def _smf_run_story(self, plan, encore): self.story_values(plan)
  def _smf_run_refs(self, plan, encore): self.ref_write()
  def _smf_take_roles(self, plan, encore):
    if not encore: self.steps_given = self.step_cap
    prev, got = self.role('cart'), plan.get('roles')
    self.role_map = {r: t for r, t in (got.items() if isinstance(got, dict) else ())
            if isinstance(t, str) and t in self.argspec}
    if not (encore and self.barrow.used):
      tool = self.role('cart')
      self.barrow = CartLog(tool, self.argspec.get(tool or '', {}))
    elif prev: self.role_map['cart'] = prev
    self._attempt('cart_note_error', self.cart_count_note)
    self.fix_cap()
    self.needs.can_pg = self.pg_arg() is not None if self.role('search') else False
  # a cart tool with no add/remove choice and one whole-number argument that may be 0: that number is the item's
  # count, and 0 is the only way to take an item out - say so in the tool's description for the driver
  def cart_count_note(self):
    tool = self.role('cart')
    spec = self.argspec.get(tool or '') or {}
    props = spec.get('properties') or {}
    if any(isinstance(s, dict) and 'enum' in s for s in props.values()) or vars(self).get('cart_noted'): return
    ints = [n for n, s in props.items() if _kind(s) == 'integer' and s.get('minimum') == 0]
    others = [n for n in props if n not in ints]
    if len(ints) != 1 or len(others) != 1 or _kind(props[others[0]]) != 'string': return
    for e in self.tool_specs:
      if e['function'].get('name') == tool:
        e['function'] = dict(e['function'], description='%s (code note: %s is the count of that item in the cart; set it to 0 to take the item out)' % (str(e['function'].get('description') or '').rstrip(), ints[0]))
        self.cart_noted = True
  def _smf_take_counts(self, plan, encore):
    budget = plan.get('message_budget')
    if is_num(budget) and budget >= 0: self.msg_cap = int(budget)
    self.needs.parsed = plan
  _smf_NOT_COUNT = re.compile('(?:\\b(?:of|x|the|by|per|than|under|over|to)|\\u00d7)$')
  def _sum_count(self, product):
    words = re.findall('[a-z0-9]+', str(product or '').lower())
    if not words: return None
    w = words[0]
    lemma = w[:-1] if len(w) > 3 and w.endswith('s') else w
    regex = '(?<![\\w.$-])' + Errand._smm_FIGURE + '\\s+' + re.escape(lemma) + '(?:e?s)?\\b'
    text = self.request.lower()
    nums = {Errand._qm_datapoint(m.group(1)) for m in re.finditer(regex, text)
        if not Errand._smf_NOT_COUNT.search(text[:m.start()].rstrip())}
    n = nums.pop() if len(nums) == 1 else None
    return n if n is not None and 1 < n < 100 else None
  def _smf_take_written_counts(self, plan, encore):
    if plan.get('code_reading'): return
    rows, raised = [r for r in plan.get('items') or [] if isinstance(r, dict)], []
    for row in rows if len(rows) > 1 else [plan]:
      n = self._sum_count(row.get('product'))
      if n is not None and (_pos_int(row.get('quantity')) or 1) == 1:
        row['quantity'] = n
        raised.append(f'{str(row.get("product"))[:40]} x{n}')
    if raised: self.log.append({'kind': 'written_counts', 'raised': raised})
  def _smf_seed_told(self, plan, encore):
    add = self.needs.add_told
    add(plan.get('instructions'), 'request')
    # the instructions restate the rules with their flags; bare rule texts (no flags, so each would hold back a
    # finishing message) only stand in when the reader gave no instructions
    if not self.needs.tolds:
      add([r for r in plan.get('rules') or [] if isinstance(r, str)], 'request')
  def _smf_scale_products(self, plan, encore):
    try: self.cable.scale = 1.0 + 0.3 * min(2, max(0, len(self.needs.items()) - 1))
    except Exception: self.cable.scale = 1.0
  def _smf_split_written(self, plan, encore):
    firm, loose = [], []
    for e in (str(r) for r in plan.get('requirements') or [] if r):
      k, colon, v = e.partition(':')
      (firm if colon and k.strip() and v.strip() else loose).append(e)
    self.needs.written = firm
    if loose:
      plan['soft_wants'] = [str(w) for w in plan.get('soft_wants') or [] if w] + loose
      plan['requirements'] = list(firm)
  def _smf_trim_asks(self, plan, encore):
    def bag(t): return set(re.findall('[a-z]{4,}', str(t).lower()))
    notches = [w for w in (bag(e.get('text') or '') for e in plan.get('instructions') or [] if isinstance(e, dict)
               and e.get('before_finish') is True and e.get('before_change') is not True) if w]
    if not notches: return
    kept, moved = [], []
    for p in plan.get('ask_first') or []:
      w = bag(p)
      m = max(len(w & n) for n in notches)
      (moved if w and (m >= 2 or 2 * m >= len(w)) else kept).append(p)
    if moved:
      plan['ask_first'] = kept
      self.log.append({'kind': 'ask_first_to_finish', 'moved': [str(p)[:120] for p in moved]})
  def _smf_take_price(self, plan, encore):
    if not encore or self.needs.rev_no <= 0: self.needs.spend_cap = plan.get('price_cap')
    if plan and self.needs.spend_cap is None and self.role('message'):
      plan['ask_first'] = ["the shopper's spending limit, as the request names no price cap (leave it out if this same point is listed again here)"] + [str(p) for p in plan.get('ask_first') or [] if p]
  def _smf_take_steps(self, plan, encore):
    if encore: self.step_cap = vars(self).get('steps_given', self.step_cap)
    v = plan.get('step_budget')
    if is_num(v) and 0 < v < self.step_cap: self.step_cap = int(v)
    c = self.cable
    c.scale = c.__dict__.get('scale', 1.0) * min(1.0, max(0.5, 40.0 / max(1, self.step_cap)))
    if encore: self.needs.rev_no += 1
  def _hints(self):
    return self.needs.parsed.get('hints') or []
  def _fhints(self, pk, multi, loose):
    for h in self._hints():
      head, _, tail = str(h).partition(':')
      if not multi or loose and pk is None or self.prod_of(head) in (pk, None):
        yield h, head, tail
  def _fkeys(self, pk):
    m = vars(self).get('menu_memo', {}).get(pk)
    return (m[1] if m else '').splitlines()
  def _home(self, pk):
    homes = self.home_cats()
    return set() if pk is None or pk >= len(homes) else homes[pk]
  def _qf_own(self, field):
    return self.prod_of(field) if self.needs.items() else None
  def story_values(self, plan):
    src, nd = plan.get('hints'), self.needs
    if isinstance(src, dict):
      src = [str(v) if ':' in str(v) else f'{k}: {v}' for k, h in src.items() for v in (h if isinstance(h, list)
        else (h,)) if v]
    src = src if isinstance(src, (list, tuple)) else []
    hs = [one_prop_hint(str(i).strip()) for i in src if isinstance(i, (str, int, float)) and str(i).strip()]
    sk, refs = list(hs), plan.get('references')
    refs = [str(r) for r in refs] if isinstance(refs, list) else []
    hunch = [r for r in nd.written if story_guess(r, self.request, refs)]
    if hunch and len(nd.written) >= 2 and len(hunch) * 2 <= len(nd.written):
      def sym(r):
        f, _, v = str(r).partition(':')
        return _bare(f).lower(), v.strip().lower()
      nd.written = [r for r in nd.written if r not in hunch]
      plan['requirements'] = list(nd.written)
      drop = {sym(r) for r in hunch}
      for e in plan.get('items') or []:
        if isinstance(e, dict) and isinstance(e.get('requirements'), list):
          e['requirements'] = [r for r in e['requirements'] if sym(r) not in drop]
      parts = [r.partition(':') for r in hunch]
      self._qf_jot('story_values', moved=hunch)
      hs += [f'{theme.strip()}: the story suggests {voiced.strip()} (not stated as a value; ask)' for theme, _, voiced in parts]
    try:
      up = self.built_hints(plan, sk)
      hs = [h for h in hs if h not in up]
    except Exception as e: self._ferr('written_hint_error', e)
    plan['hints'] = hs[:12]
  def built_hints(self, plan, indications):
    posts, up = [e for e in plan.get('items') or [] if isinstance(e, dict)], []
    for h in indications:
      req, sorts = self._said_form(plan, posts, str(h))
      if not req: continue
      self.needs.written.append(req)
      plan['requirements'] = list(self.needs.written)
      e = next((e for e in posts if isinstance(e.get('requirements'), list)
           and (len(posts) == 1 or kind_words(e.get('product') or '') & sorts)), None)
      if e:
        e['requirements'].append(req)
      up.append(h)
      self._qf_jot('written_hint', allusion=str(h)[:200], requirement=req)
    return up
  def _said_form(self, plan, posts, h):
    theme, colon, tail = h.partition(':')
    field = _bare(theme)
    if not colon or not field or junk_key(field): return '', set()
    tags = re.findall('\\(([^)]*)\\)', theme)
    word = owner = ''
    for owner in [tags[-1]] if tags else [str(e.get('product') or '') for e in posts] or [str(plan.get('product')
      or '')]:
      if (word := first_val(tail, self.request, owner)): break
    if not word: return '', set()
    sorts, rx = kind_words(owner), '\\b' + re.escape(word) + '\\b'
    def hit(r):
      head, _, val = r.partition(':')
      tg = re.findall('\\(([^)]*)\\)', head)
      return not (tg and sorts and not kind_words(tg[-1]) & sorts) and bool(same_prop(head, field)
        or re.search(rx, val.lower()))
    return ('', set()) if any([hit(r) for r in self.needs.written]) else (f'{theme.strip()}: {word}', sorts)
  def open_hints(self, primekey=None):
    old, new = [], []
    for h in self._hints():
      if not (f := str(h).split(':', 1)[0].strip()): continue
      own = self._qf_own(f)
      if None not in (primekey, own) and own != primekey: continue
      if self.open_prop(own, f, self.prop_line(own, f),
        per_exemplars=True) and not self.asked_inner(own, f):
        new.append(str(h))
      elif (alt := self.hint_alt(own, str(h), f)): old.append(alt)
    return old + new
  @_guard('')
  def prop_line(self, primekey, field):
    kept, st = vars(self).setdefault('menu_memo', {}).get(primekey), self.rec_db
    if kept and kept[0] == len(st.items): menu = kept[1]
    else:
      if not self.needs.items():
        s = self.own_shelf()
        menu = (st.prop_list(s) if s else st.prop_list())[0]
      else:
        home = self._home(primekey)
        first = next((x for x in st.shelf_read() if x.strip().lower() in home), None)
        menu = '' if first is None else st.prop_list(first)[0]
      self.menu_memo[primekey] = (len(st.items), menu)
    return next((r for r in menu.splitlines() if exact_prop(_qx_key(r), field)), '')
  def fld_vals(self, primekey, field):
    row = self.prop_line(primekey, field)
    return {_norm(v) for v in _vals(row)} - {''} if '):' in row else set()
  def same_field(self, primekey, first, second):
    return exact_prop(first, second) or self._shared_egs(primekey, first, second)
  @_guard(False)
  def _shared_egs(self, pk, a, b):
    return bool(same_prop(a, b)) and not self.fld_vals(pk, a).isdisjoint(self.fld_vals(pk, b))
  def hinted_prop(self, primekey, field):
    s = str(field).split('(')[0].strip()
    return any(exact_prop(f, s) and (primekey is None or not self.needs.items()
                        or self.prod_of(f) in (primekey, None))
         for f in (str(h).split(':', 1)[0] for h in self._hints()))
  @_guard(0)
  def prod_reads(self, primekey):
    st = self.rec_db
    if self.needs.items(): return sum(map(st.dense_n, self._home(primekey)))
    s = self.own_shelf()
    return st.dense_n(s) if s else st.dense_n()
  @_guard(False)
  def plain_hint(self, primekey, field, hinted=False):
    s = _bare(field)
    if not s or not (hinted or self.hinted_prop(primekey, s)) or self.prop_line(primekey,
      s) or self.prod_reads(primekey) < 2:
      return False
    return not any(same_prop(_qx_key(r), s) for r in self._fkeys(primekey))
  def hint_key(self, indication):
    shown = str(indication)
    try:
      f = shown.split(':', 1)[0].strip()
      if self.plain_hint(self._qf_own(f), f, hinted=True): shown += ' [no key]'
    except Exception: pass
    return shown
  def _fparts(self, multi, replied):
    for c in self.questions_log:
      lab = c.split(':', 1)[0].strip('- ')
      if replied and '[not answered]' in lab: continue
      asked = re.search('asked "(.*?)" -> ', c)
      for s in lab.split(';') if replied else [s.strip() for s in re.split(';|\\band\\b',
        re.sub('\\s*\\[[^\\]]*\\]', '', lab)) if s.strip()]:
        own = self.prod_of(s) if multi else None
        yield s, self.prod_of(asked.group(1)) if own is None and multi and asked else own
  @_guard(False)
  def asked_inner(self, primekey, theme):
    m = bool(self.needs.items())
    return any((primekey is None or not m or own in (None, primekey)) and _within(s, theme) for s,
      own in self._fparts(m, True))
  @_guard('')
  def hint_alt(self, primekey, indication, theme):
    chosen, m = vars(self).setdefault('hint_alts', {}), bool(self.needs.items())
    forefront, _, tail = indication.partition(':')
    if not chosen.get(indication):
      if not any((self.same_field(primekey, a, theme) or _within(a, theme))
           and (not m or self.prod_of(a) in (primekey, None))
           for a in vars(self).get('quiet_asks') or []):
        return ''
      for alt in _alts(tail):
        if not (junk_key(alt) or exact_prop(alt,
          theme)) and (row := self.prop_line(primekey, alt)):
          chosen[indication] = _qx_key(row)
          break
    key = chosen.get(indication) or ''
    if not key or key.lower() in vars(self).get('spent_asks', set()) or not self.plan_open(primekey, key):
      return ''
    tag = re.search('\\(([^()]*)\\)\\s*$', forefront)
    label = f'{key} ({tag.group(1).strip()})' if tag else key
    rest = re.sub('\\s*\\(or [^()]+\\)', '', tail).strip()
    return f'{label}: {rest} (its first property, {theme}, drew no preference)'
  @_guard(False)
  def hint_set(self, pk, field):
    s = _bare(field)
    req = ['\\b' + re.escape(w) for w in _xwords(s)]
    return any(any(exact_prop(n, s) for n in [_bare(head)] + _alts(tail) if n)
         or req and all(re.search(x, str(h).lower()) for x in req)
         for h, head, tail in self._fhints(pk, bool(self.needs.items()), True))
  @_guard(False)
  def pair_of(self, pk, first, second):
    if exact_prop(first, second): return False
    a, b = _xwords(first), _xwords(second)
    kin = lambda p, q: p == q or min(len(p), len(q)) >= 4 and (p.startswith(q) or q.startswith(p))
    top = lambda w: w[:-1] if len(w) > 1 else w
    return any(kin(p, q) for p in top(a) for q in b) or any(kin(p, q) for p in a
      for q in top(b)) or self.same_field(pk, first, second)
  @_guard('')
  def pair_key(self, pk, field):
    posts = self.needs.items()
    self.prop_line(pk, field)
    keys = [_qx_key(r) for r in self._fkeys(pk)]
    sorts = kind_toks(posts[pk]['product'] if posts and pk is not None else str(self.needs.parsed.get('product')
      or ''))
    story = ' '.join([self.request.lower()] + [str(h).lower() for h, _, _ in self._fhints(pk, bool(posts),
      False)])
    spent, base = vars(self).get('spent_asks', set()), set(prop_words(field))
    for key in keys:
      if not key or junk_key(key) or key.lower() in spent or not self.pair_of(pk, key,
        field) or not self.plan_open(pk, key) or _within(key, field):
        continue
      if any(re.search('\\b' + re.escape(w), story) for w in _xwords(key) if w not in base and not kind_in(w,
        sorts)) or self.hint_set(pk, key):
        return key
    return ''
  @_guard('')
  def val_form(self, pk, field, wares):
    row = self.prop_line(pk, field)
    if '):' not in row: return ''
    seized, picks = {_norm(v) for o in self._fkeys(pk) if '):' in o and o.strip() != row.strip()
      for v in _vals(o)}, []
    for v in map(str.strip, _vals(row)):
      if len(picks) == 3: break
      if v and len(v) <= 30 and re.search('[a-z]', v.lower()) and not re.search('[,;/]',
        v) and _norm(v) not in seized | {_norm(p) for p in picks}:
        picks.append(v)
    return f'For the {wares}, would you like {", ".join(picks[:-1])} or {picks[-1]}?' if len(picks) >= 2 else ''
  def _twin(self, name):
    posts = self.needs.items()
    pk = self.prod_of(name) if posts else None
    done, slot = vars(self).setdefault('twin_asks', set()), pk if posts else -1
    return None if posts and pk is None or slot in done else (posts, pk, done, slot)
  @_guard(None)
  def solo_ask(self, field):
    if not (t := self._twin(field)): return None
    posts, pk, done, slot = t
    wares = posts[pk]['product'] if posts else str(self.needs.parsed.get('product') or 'item')
    target = _bare(field)
    if not (text := self.val_form(pk, target, wares)):
      if not (target := self.pair_key(pk, field)): return None
      text = self.val_form(pk, target, wares) or f'Which {target} would you like for the {wares}?'
    done.add(slot)
    return f'{target} ({wares})' if posts else target, text
  @_guard(False)
  def asked_for(self, primekey, field):
    m, s = bool(self.needs.items()), _bare(field or '')
    return bool(s) and any((primekey is None or not m or own == primekey) and exact_prop(_bare(g), s)
      for g, own in self._fparts(m, False))
  @_guard('')
  def sub_key(self, pk, field):
    if not (s := _bare(field)): return ''
    spent = vars(self).get('spent_asks', set())
    for _, head, tail in self._fhints(pk, bool(self.needs.items()), False):
      theme = _bare(head)
      if not theme or not (self.same_field(pk, s, theme) or _within(s, theme)): continue
      for alt in _alts(tail):
        if junk_key(alt) or exact_prop(alt, theme) or exact_prop(alt,
          s) or _within(alt, s):
          continue
        key = _qx_key(row) if (row := self.prop_line(pk, alt)) else ''
        if key and not exact_prop(key, s) and key.lower() not in spent and not self.asked_for(pk,
          key) and self.plan_open(pk, key):
          return key
    return ''

def _bare(text):
  return re.split('\\s*\\(', str(text))[0].strip()
def _qx_key(row):
  return row.strip('- ').split(' (in ')[0].strip()
def _norm(v):
  return ' '.join(re.findall('[a-z0-9]+', v.lower()))
def _vals(row):
  return row.split('):', 1)[1].split('|')
def _xwords(s):
  return [w for w in prop_words(s) if len(w) >= 3 and w not in STOP_WORDS]
def _alts(tail):
  return [s.strip() for g in re.findall('\\(or ([^()]+)\\)', str(tail)) for s in re.split(',|\\bor\\b|/', g)
    if s.strip()]
def _within(a, b):
  x, y = set(_xwords(a)), set(_xwords(b))
  return bool(x and y) and (x <= y or y <= x)
def _qx_cut(text, n=70):
  flat = ' '.join(str(text).split())
  return flat if len(flat) <= n else flat[:n - 3] + '...'
def _full(raw):
  return bool(str(raw).strip())
def _over(cost, units, cap):
  return is_num(cap) and is_num(cost) and cost * units > cap
def _rank(carry, keep):
  return sorted(filter(keep, carry), key=lambda n: -carry[n])
def _xfill(peers, keys):
  return sum(1 for r in peers.values() if any(_full(r['fields'].get(n, '')) for n in keys))

class AskPace:
  def _gerr(self, kind, mistake):
    self.log.append({'kind': kind, 'error': ': '.join((type(mistake).__name__, str(mistake)))[:300]})
  def hit_alt(self, prodname):
    try:
      if not (t := self._twin(prodname)) or not (alt := self.sub_key(t[1], prodname)): return ''
      t[2].add(t[3])
      return f'{alt} ({t[0][t[1]]["product"]})' if t[0] else alt
    except Exception: return ''
  def keeps_plain(self, text, unadorned_qy):
    toks = lambda s: re.findall('[a-z0-9]+', str(s).lower())
    have = {self.lemma(w) for w in toks(text)}
    need = {self.lemma(w) for w in toks(unadorned_qy) if w not in {'which', 'would', 'you', 'like', 'for', 'the'}
      and w not in STOP_WORDS}
    return bool(need) and not need - have
  def alt_untried(self, pq):
    try:
      heads = (str(h).split(':', 1)[0] for h in self.needs.parsed.get('hints') or [])
      names = (_bare(f) for f in heads if self.prod_of(f) in (pq, None))
      return any(self.asked_for(pq, n) and self.sub_key(pq, n) for n in names if n)
    except Exception: return False
  @staticmethod
  def menu_trim(repertoire_writing, witnessed_nx):
    if not repertoire_writing: return repertoire_writing
    rows = [r for r in repertoire_writing.splitlines() if not ((m := re.match('^- (.+?) \\(in \\d+ listings?\\)',
      r.strip())) and enum_pair(m.group(1)))]
    if witnessed_nx >= 4: rows = [r for r in rows if not re.search('\\(in [01] listings?\\)', r)]
    return '\n'.join(rows) if rows else repertoire_writing
  def first_reads(self):
    try:
      if vars(self).get('first_reads_done') or self.batch_tool() or not self.role('details'): return False
      self.first_reads_done = True
      nd = self.needs
      lines, shelves = nd.items(), self.home_cats()
      units = [x['quantity'] for x in lines] or [nd.quantity()]
      budget = self.steps_rem() - self.order_due()
      budget -= 2 * max(0, self.ask_room() - self.msgs_sent) + 8
      seen, maps = [], []
      for pq in range(max(1, len(lines))):
        shelf = shelves[pq] if pq < len(shelves) else set()
        n = self.prod_reads(pq if lines else None)
        if shelf and n < 2 and budget > 0:
          got = self.menu_rows(shelf, 2 - n, seen, units[min(pq, len(units) - 1)])
          seen.extend(got)
          if got: maps.append((n, pq, got))
      got = [g for _, _, g in sorted(maps)]
      if (queue := ([g[0] for g in got] + [g[1] for g in got if len(g) > 1])[:max(0, budget)]):
        vars(self).update(open_queue=queue, open_recheck=True, open_purpose='menu', open_say='Before my first message I open up to two listings of each product, to see which fields they carry.')
        self.log.append({'kind': 'first_reads', 'ids': queue})
      return bool(queue)
    except Exception as e:
      self._gerr('first_reads_error', e)
      return False
  def plan_open(self, pq, prodname):
    try:
      hinted = self.hinted_prop(pq, prodname)
      return self.open_prop(pq, prodname, self.prop_line(pq, prodname) if hinted else '',
        per_exemplars=hinted)
    except Exception: return self.open_prop(pq, prodname)
  _REFER_RX = re.compile('as shown|same as|the page|sold by|redirect|refer')
  def rule_ok(self, guideline_writing, exemplar_writing=''):
    head, _, value = str(guideline_writing).partition(':')
    low, ex = value.lower(), exemplar_writing
    if self._REFER_RX.search(low) or is_ref_text(low): return True
    if not confirm_val(value, head): return False
    ptr = self._read_refs(True) + list(self.needs.__dict__.get('resolved') or [])
    if story_guess(guideline_writing, self.request + ' ' + ' '.join(self.needs.trades), ptr): return False
    if not ex or re.search('\\d', value): return True
    words, stash = [w for w in re.findall('[a-z]{3,}', low)
      if w not in STOP_WORDS], set(re.findall('[a-z]+', ex))
    seen = lambda w: re.search('\\b' + re.escape(w[:-1] if len(w) > 4 else w),
      ex) or len(w) >= 4 and any(len(x) >= 4 and edits_in(x, w, 1) for x in stash)
    return not words or any(map(seen, words))
  def _read_refs(self, keep_empty):
    f = self.needs.parsed.get('references')
    return [str(h) for h in f if keep_empty or h] if isinstance(f, list) else []
  def order_due(self):
    cap = self.step_cap
    floor = max(FIRST_ORDER_BY, min(FIRST_ORDER_DUE, cap // 4 + 1))
    held = max(floor, min(16, cap // 3))
    if self._own_issue():
      try:
        n = 8 + max(1, len(self.needs.items()))
      except Exception: n = 9
      held = max(held, n)
    return max(floor, min(held, cap // 2)) + self.step_reserve()
  def _own_issue(self):
    d = vars(self).get('dead')
    return bool(vars(self).get('event_seen')) or bool(d) and bool(d & set(self.barrow.units))
  def endgame(self):
    return bool(self.role('order') and self.role('cart'))
  def _order_time(self, n):
    return n <= self.order_due() and self.orders_tried == 0 or n <= FIRST_ORDER_BY and self.refusal_n > 0
  def _read_gate(self, n, success):
    return self.alt_unseen() and n > 5 or success() and n >= min(FIRST_ORDER_BY + 2, self.order_due())
  def _search_gate(self, n, paginated):
    return (self.grow_late() or paginated() or self.alt_unseen() and n > 5) and n > 4
  def on_offer(self):
    n, kit, role = self.steps_rem(), self.tool_specs, self.role
    ends = (role('order'), role('decline'), role('cart'))
    stage = 'last' if n <= 1 else 'closing' if n <= 3 else 'reserve' if self._order_time(n) else 'open'
    if stage == 'open': return kit
    names = set(ends[:2]) if stage == 'last' else set(ends)
    if stage == 'reserve':
      names.update((role('cart_view'), role('view') or role('details'), self.stock_tool()))
      reader = role('details')
      if self.swap_open(): names.add(reader)
      share = vars(self).get('late_n', 0) < 3
      if share and self._read_gate(n, self.name_open): names.add(reader)
      if share and self._search_gate(n, lambda: vars(self).get('named_pages')): names.add(role('search'))
      if n >= 7 and self.backup_rows(): names.add(reader)
    if not (ends[0] and ends[2]): return kit
    if (speak := role('message')) and n >= 2 and (self.msg_cap is None or self.msgs_sent < self.msg_cap):
      names.add(speak)
    return [s for s in kit if s['function']['name'] in names] or kit
  def in_reserve(self):
    n = self.steps_rem()
    return self.endgame() and n > 3 and self._order_time(n)
  def alt_unseen(self):
    dead, rev = self.dead_lines(), str(self.needs.rev_no)
    live = [s for s, _, t in (p.rpartition('|') for p in self.passed) if t == rev and s not in self.refused_lines
      and s not in dead]
    if not live: return True
    if not self.needs.items(): return False
    try:
      seen = {x for x in map(self.known_shelf, live) if x != ''}
      return bool(seen) and any(h and not h & seen for h in self.home_cats())
    except Exception: return False
  def name_open(self):
    st = self.rec_db
    return {r for r in vars(self).get('named', set()) if r in st.ids and r not in st.read
      and r not in self.refused_lines}
  def grow_late(self):
    nd = self.needs
    return (nd.grown_at == nd.rev_no
        and self.order_due() >= self.cable.ticks - nd.grown_tick)
  def start_batch(self):
    vars(self).update(late_q=0, late_ids=set())
  def late_keep(self, summon_appellation, summon_prms):
    tool, kv = summon_appellation, vars(self)
    reader = self.role('details')
    if tool not in (reader, self.role('search')):
      if not (tool in self.fetch_tools() and reader): return ''
      tool = reader
    if not self.in_reserve(): return ''
    targets, parsed = {v for v in summon_prms.values() if isinstance(v, str)}, tool == reader
    if parsed and targets & (set(self.barrow.units) | {p.rpartition('|')[0] for p in self.passed}): return ''
    queued = kv.get('late_q', 0)
    spare = self.steps_rem() - queued
    room = lambda: parsed and spare - 1 >= 6 and kv.get('backup_n', 0) < 2
    def fresh():
      asked = kv.get('late_ids', set())
      return [r for r in self.backup_rows() if r not in asked][:2 - kv.get('backup_n', 0)]
    page = lambda: json.dumps(summon_prms, sort_keys=True, default=str) in kv.get('named_pages', set())
    if not (kv.get('late_n', 0) + queued < 3 and (self._read_gate(spare,
      lambda: bool(targets & self.name_open())) if parsed else self._search_gate(spare, page))):
      if room() and targets & set(fresh()): self.backup_n = kv.get('backup_n', 0) + 1
      elif parsed and (row := self.swap_open()) and row in targets and spare > 2:
        self.swap_done = True
      else:
        extra = fresh() if room() else []
        say = 'Not run (no step used). Checkout mode: only three kinds of read remain - (1) a line already in the cart; (2) a listing an item check pointed to; (3) while no fit has passed verification, 3 reads in total. Best move now: put the top verified fit in the cart, confirm its stock and price once more, and place the order.'
        return say + (' Cheaper stand-ins that nobody has opened yet are allowed as well: ' + ', '.join(extra) + '.' if extra else '')
    self.late_q = queued + 1
    kv.setdefault('late_ids', set()).update(targets)
    return ''
  def backup_rows(self):
    if self.needs.priority() != 'cheapest' or vars(self).get('backup_n', 0) >= 2: return []
    if not (vars(self).get('event_seen') or self.dead_lines() or self.name_open()): return []
    found = []
    try:
      for pq in self.prod_slots():
        stash = self.zd_pool(pq)
        if (fits := [r['price'] for r in stash if r['verified'] and r['fit'] and r['price'] is not None]):
          found += [(r['price'], r['id']) for r in stash
               if not any(r[m] for m in ('read', 'gone', 'over', 'pointer', 'ruled', 'clash'))
               and r['price'] is not None and r['price'] < min(fits)]
    except Exception as e:
      self._gerr('backup_error', e)
      return []
    return [s for _, s in sorted(found)[:6]]
  def field_names(self):
    w, _ = self.rec_db.prop_list()
    return '\n\n## FIELD NAME COUNTS (taken from opened listings; a name is merged with variants that differ only in separators; each count is the number of opened listings carrying that name - the most frequent one is the main field of a property)\n' + w if w else ''
  def check_item(self, good_sid, intention_writing, second=''):
    if self.cable.spare() < CLOSING + 20: return ''
    nd, st, kv = self.needs, self.rec_db, vars(self)
    if (key := f'{good_sid}|{nd.rev_no}') in kv.get('near_ok', set()): return ''
    refs, tracked, own = self._read_refs(False), list(kv.get('refs')
      or []), self.own_detours(good_sid)
    memo = key + (f'|{len(st.read)}' if refs or any(h.get('status') != 'resolved' for h in tracked)
      else '') + ('|second' if second else '') + (f'|own{len(own)}' if own else '')
    if memo in self.rulings: return self.rulings[memo]
    if (pointer := self.ref_ptr_note(good_sid)):
      return self._rule_spent(good_sid, memo, pointer, ['reference chain'])
    if not (rec_ls := st.assoc(good_sid)): return ''
    view = self.item_view(good_sid)
    if (bad := view['fails'][:3]):
      return self._rule_spent(good_sid, memo, '; '.join(bad), bad, by='field evidence')
    seg = [ITEM_CHECK, "\n\n## REQUEST (the shopper's own words; later changes follow)\n" + self.request + nd.var_opts(), '\n\n## THE ITEM MUST MEET (as first stated, revealed later or changed)\n' + nd.body_text()]
    if nd.items():
      held = self.barrow.held(good_sid)
      seg.append("\n\n## WHICH PRODUCT: the request names several products, and the item's category tells which one this item would be. Judge it only on that product's own constraints and quantity - nothing the other products need applies here." + (f' Already carted: {int(held)} units.' if held > 0 else ''))
    seg += [self.ref_note(good_sid) or '', self._ref_records(good_sid, refs, tracked), second,
        self._own_notes(own),
        f'\n\n## CANDIDATE, next step {intention_writing}: {good_sid}\nIts store records, oldest first:\n' + '\n'.join(rec_ls)]
    if self.barrow.last_view is not None:
      seg.append('\n\n## CART, NEWEST VIEW\n' + as_text(self.barrow.last_view)[:3000])
    seg.append(self.field_names())
    if view['lines']:
      seg.append("\n\nFIELD EVIDENCE, compiled by code. For each required value: how this store's listings in the category write it, then the item's own entry:\n" + '\n'.join(view['lines']))
    if (labels := st.label_lines()):
      seg.append("\n\n## RECURRING CATEGORY LABELS (with counts). The item has to sit under the label whose words match the request's phrase as a whole - a neighbouring label does not qualify.\n" + labels)
    verdict = self.cable.consult(''.join(seg), 3000, CHECKER)
    self.log.append({'kind': 'verify', 'item': good_sid, 'result': verdict})
    if not (verdict and verdict.get('fits') is False):
      self.rulings[memo] = ''
      if verdict:
        self.passed.add(key)
        kv.setdefault('pass_price', {})[good_sid] = st.price(good_sid)
      return ''
    problems = '; '.join([str(p) for p in verdict.get('problems') or []][:5])
    self.rulings[memo] = edict = f'Item check on {good_sid}: rejected - {problems}. Keep the item only if you are certain this verdict is wrong; otherwise take a fit proven on every requirement.'
    return edict
  def _rule_spent(self, good_sid, memo, why, problems, **extra):
    self.rulings[memo] = edict = f'Item check on {good_sid}: rejected - {why}. The code ruled it out; look for another.'
    self.log.append({'kind': 'verify', 'item': good_sid, 'result': {'item': good_sid, 'fits': False,
      'problems': problems, **extra}})
    return edict
  def _ref_records(self, good_sid, refs, tracked):
    if not (loose := self.ref_untracked(refs, tracked)): return ''
    tips, st = {w for h in loose for w in re.findall('[a-z]{4,}', h.lower().split(':', 1)[-1])}, self.rec_db
    hits = [r for r in sorted(st.read) if r != good_sid and tips and any(w in t for t in [' '.join(st.assoc(r,
      1)).lower()] for w in tips)][:3]
    return '\n\n## READ LISTINGS THAT THE REQUEST MIGHT REFER TO. A value set by pointing at one of these comes from that listing, taken from the field the requirement names.\n' + '\n'.join(x for r in hits for x in st.assoc(r, 1)) if hits else ''
  @staticmethod
  def _own_notes(own):
    recited = ''.join((f'''\n- {prodname}, for '{guideline_writing}': "{sent[:240]}"''' for guideline_writing,
      prodname, sent in own or ()))
    return "\n\nREDIRECT CHECK - quoted by the code from this item's notes:" + recited + "\nReading rule. A quote may pass a property on to a different listing (wording such as take from, refer to, go by, look to). When it does, the grader uses the value that other listing gives, so the field on this item proves nothing about the requirement; only the other listing's record can satisfy it. A quote that names another listing just as a version sold elsewhere, or says this item is not it, is to be ignored." if recited else ''
  def zh_rows(self):
    imprint, table = vars(self).get('ch_rows_memo') or (None, None)
    if imprint != len(self.rec_db.items):
      table = self.all_rows()
      self.ch_rows_memo = (len(self.rec_db.items), table)
    return table
  def store_fields(self, label=''):
    try:
      table, homes, out = self.zh_rows(), self.home_cats(), {}
      if not self.needs.items():
        scope = {own} if (own := self.own_shelf()) else set(homes[0] if homes else ())
      else:
        pq = self.prod_of(str(label))
        scope = set() if pq is None or pq >= len(homes) else set(homes[pq])
      for row in table.values():
        if row['fields'] and not (scope and row['shelf'] not in scope):
          for k, v in row['fields'].items():
            if str(v).strip() and not junk_key(str(k)): out.setdefault(str(k), []).append(str(v))
      return out
    except Exception: return {}
  def zh_detour(self, good_sid, prodname):
    try:
      return any(any(same_prop(str(h).partition(':')[0], prodname)
             for h in (n[:2] if isinstance(n, (list, tuple)) else [n]))
           for n in self.own_detours(good_sid) or [])
    except Exception: return True
  def item_view(self, good_sid):
    nd = self.needs
    imprint = (good_sid, len(self.rec_db.items), nd.rev_no, '|'.join(nd.props_left()))
    hoard = vars(self).setdefault('ch_view_memo', {})
    if imprint not in hoard:
      try:
        hoard[imprint] = self.fld_facts(good_sid)
      except Exception as e:
        self._gerr('field_facts_error', e)
        hoard[imprint] = {s: [] for s in ('lines', 'rules', 'fails')}
    view = hoard[imprint]
    if not (erased := (vars(self).get('near_said') or {}).get(f'{good_sid}|{nd.rev_no}')): return view
    try:
      names = [p for p, _, _ in erased]
      kin = lambda a, b: same_prop(a, b) or exact_prop(a, b)
      def jot(line):
        head = re.match("- (.+?) = '", line)
        return next((f" (Meaning check: '{voiced[:80]}' was accepted as '{magnitude[:60]}' - requirement met.)"
              for partno, magnitude, voiced in erased if kin(head.group(1), partno)), '') if head else ''
      return {'lines': [x + jot(x) for x in view['lines']], 'rules': list(view['rules']), 'fails': [g
        for g in view['fails'] if not any(kin(g.split(':', 1)[0].strip(), p) for p in names)]}
    except Exception: return view
  # the code's own per-requirement verdicts for opened listings, for judges that weigh many cards:
  # listings with no failing requirement first (cheapest first), then the rest
  def code_reads(self, most=14, scan=60):
    st, rows = self.rec_db, []
    try:
      for sid in sorted(st.read, key=lambda s: (st.price(s) is None, st.price(s) or 0, s))[:scan]:
        bits = []
        for ln in self.item_view(sid)['lines']:
          head, tail = re.match("- (.+?) = '", ln), ln.split(' | this listing: ', 1)
          if head and len(tail) > 1: bits.append('%s: %s' % (head.group(1), tail[1][:170]))
        if not bits: continue
        bad = any('FAILS' in b for b in bits)
        tag = '' if bad else ' [ALL MEET]' if all('-> MEETS' in b for b in bits) else ' [none fails]'
        rows.append((bad, '%s%s: %s' % (sid, tag, ' || '.join(bits))))
    except Exception as e:
      self._gerr('code_reads_error', e)
    out = [r for bad, r in rows if not bad] + [r for bad, r in rows if bad]
    return ('\n\nCODE READS OF OPENED LISTINGS (per requirement, worked out by code from each record under the matching rules; '
      "a 'MEETS' or 'FAILS' here is decided, other notes say what is still open; listings where nothing fails come first, "
      'cheapest first):\n' + '\n'.join(out[:most])) if out else ''

class ItemCheck:
  def fld_facts(self, pick):
    found = dict(lines=[], rules=[], fails=[])
    rows = self.zh_rows()
    if not ((me := rows.get(pick)) and me['fields']): return found
    multi, label = bool(self.needs.items()), me['shelf']
    if multi and not label: return found
    slot = next((n for n, ls in enumerate(self.home_cats()) if label in ls), None) if multi else None
    if len(peers := {k: r for k, r in rows.items() if r['fields'] and (not label or r['shelf'] == label)}) < 3:
      return found
    carry, table = {}, {}
    for k, r in peers.items():
      for n, raw in r['fields'].items():
        if _full(raw) and not junk_key(n):
          carry[n] = 1 + carry.get(n, 0)
          bits = text_parts(raw)
          table.setdefault(n, []).extend((k, s, txt_words(s)) for s in bits + ([str(raw).strip().lower()]
            if len(bits) > 1 else []))
    sc = (pick, rows, me, label, multi, slot, peers, carry, table, found)
    for line in self.needs.props_left()[:12]:
      lab, _, rest = str(line).partition(':')
      req = ' '.join(re.sub('\\((?:stated|revealed[^)]*)\\)', ' ', rest).split())
      if (a := _bare(lab)) and req and not is_ref_text(req) and not self._REFER_RX.search(req.lower()):
        self._weigh_need(sc, lab, a, req)
    return found
  def _weigh_need(self, sc, label, a, req):
    pick, rows, me, shelf, multi, slot, peers, carry, table, found = sc
    low, lines, fails, rules = req.lower(), found['lines'], found['fails'], found['rules']
    cut, has = _qx_cut, _full
    own = self.prod_of(label) if multi else None
    out, brief, fields = own is not None and own != slot, cut(req), me['fields']
    if re.search('\\d', req) and re.search(BOUND_RX, low):
      stems, homes = set(_xwords(a)), [n for n in carry if carry[n] >= 2 and exact_prop(n, a)]
      if out or not (a and stems and homes) or any(has(raw) and (exact_prop(n, a)
        or set(prop_words(n)) & stems) for n, raw in fields.items()):
        return
      lines.append("- %s = '%s' | field in this store: %s | category listings read that fill it: %d of %d | this listing: field absent -> FAILS the limit (a figure from some other field cannot stand in)"
            % (a, brief, ' / '.join("'%s'" % n for n in homes[:3]), _xfill(peers, homes), len(peers)))
      return fails.append("%s: this listing has no %s field, though the store's category listings carry one -> '%s' unproven" % (a, homes[0], brief))
    if re.fullmatch('\\s*\\d+(?:\\.\\d+)?\\s+[a-z]+\\s*', low) or out: return
    plus = confirm_val(req, label)
    if not plus and not re.search(NEG_WORDS, low) or not (forms := vsubs(req)): return
    if plus and (sh := str(shelf or '')) and val_in_txt(req, sh):
      line = "- '%s' | the category '%s' itself names this value, so every listing filed there meets it, whether or not a field or the title repeats it -> MEETS" % (brief, cut(sh, 60))
      rules.append(line)
      return lines.append(line)
    if (titled := bool(named := _rank(carry, lambda n: exact_prop(n, a)))):
      if not plus and (flat := _norm(req)) in (' '.join(e[2]) for n in named for e in table.get(n, [])):
        plus = True
        TEXTUAL_DATA.add(flat)
    elif not plus or multi and own is None or not (keep := _rank(carry, lambda n: carry[n] >= 2
      and any(zh_match(forms, e[2]) for e in table.get(n, [])))):
      return
    else: named = _rank(carry, lambda n: exact_prop(n, keep[0]))
    cps = [e for n in named for e in table.get(n, []) if e[0] != pick]
    writers, alone, filled = {e[0] for e in cps if zh_match(forms, e[2])}, {' '.join(e[2])
      for e in cps}, _xfill(peers, named)
    smp = ' | '.join(list(dict.fromkeys(cut(r['fields'][n], 40) for r in peers.values() for n in named
      if has(r['fields'].get(n, ''))))[:6])
    tail = "other listings with '%s' verbatim in it: %d" % (brief, len(writers))
    if not plus and (sf := vsubs(bare := ' '.join(re.sub(NEG_WORDS, ' ', low).split()))):
      tail = "other listings holding '%s', the excluded value, there: %d (holding it = failing)" % (cut(bare), len({e[0] for e in cps if zh_match(sf, e[2])}))
    lead = "- %s = '%s' | field in this store: %s%s | category listings read that fill it: %d of %d | %s" % (
      a, brief, ' / '.join("'%s'" % n for n in named[:3]), '' if titled else " [chosen by its values; no field is called '%s']" % a,
      filled, len(peers), tail)
    sorts = kind_words(req)
    naming = [k for k, r in rows.items() if k != pick and (not shelf or r['shelf'] == shelf)
      and sorts <= r['words']][:3] if sorts and not writers else []
    seen = ' | values on record: ' + smp
    if not plus: trail = ''
    elif writers:
      trail = ' | so: a match needs these exact words (contained, or as one list element); synonyms and related words miss' + ('' if titled else " | no field carries the property's name, so a title holding the value matches as well, unless the listing's own field sharing a word of that name contradicts it")
    elif naming:
      trail = seen + " | titles/texts in this category mention '%s' (%s), which shows a house term: synonyms or related values count only once one of those listings' records shows this field writing it differently" % (brief, ', '.join(naming))
    elif filled < ZH_PARAPHRASE_MINIMUM:
      trail = seen + " | '%s' absent, but with this few read listings filling the field nobody can tell whether the store ever writes it word for word: an equivalent is not yet proven" % brief
    else: trail = seen + ' | the value appears in none of them: an equivalent may count'
    rules.append(lead + trail)
    mine = [(n, str(fields[n])) for n in named if has(fields.get(n, ''))]
    shown, base = '; '.join("%s: '%s'" % (n, cut(raw, 90)) for n, raw in mine), prop_words(a)
    amp = [(n, str(raw)) for n, raw in fields.items() if base and n not in named and has(raw)
      and not junk_key(n) and len(w := prop_words(n)) == len(base) + 1 and w[1:] == base]
    ok = [d for d in amp if val_in_txt(req, d[1])]
    near = bool(ok) or any(val_touch(req, raw) for _, raw in amp)
    note = " fuller-named key %s = '%s' holds the value (one extra leading word, still this property) -> MEETS" % (ok[0][0], cut(ok[0][1], 60)) if ok else ''
    if self.zh_detour(pick, a):
      return lines.append(lead + ' | caution: its notes suggest another listing may settle this property, so its own field may not be the last word')
    if not plus:
      return lines.append(lead + ' | this listing: %s | exclusion: holding the ruled-out value fails' % (shown or 'field absent'))
    stems = set(_xwords(a))
    cap = not [n for n, raw in fields.items() if has(raw) and not junk_key(n)
      and set(prop_words(n)) & stems and not val_in_txt(req,
      raw)] and val_in_txt(req, me.get('title') or '')
    if not mine:
      if ok: return lines.append(lead + ' | this listing: field absent |' + note)
      if cap and titled:
        return lines.append(lead + " | this listing: field absent, and none of its own fields shares the property's name; the title has '%s' -> MEETS (the title takes the place of the field this listing is missing)" % brief)
      if cap:
        return lines.append(lead + " | this listing: field absent, and no field carries the property's name; the title has '%s' -> MEETS" % brief)
      st = {w for w in base if len(w) >= 3}
      maybe = sorted([(n, raw) for n, raw in fields.items() if n not in named and has(raw)
        and not junk_key(n) and (set(prop_words(n)) & st or val_in_txt(req,
        raw))], key=lambda d: not val_in_txt(req, d[1]))
      cat = '; '.join("%s: '%s'" % (n, cut(raw, 60)) for n, raw in maybe[:4])
      return lines.append(lead + ' | this listing: field absent' + (' | its own fields that might hold it: ' + cat if cat else '') + ' | only an alias of this SAME property may count; fields about other properties never do (e.g. a height for a type, a frame material for a material)')
    neat, joined, held = [], [], [set(o.split()) for o in alone if o]
    def single(x):
      w = txt_words(x)
      return ' '.join(w) in alone or any(o and o <= set(w) for o in held)
    for m in [m for _, raw in mine for m in text_parts(raw)] + [raw.strip().lower() for _, raw in mine
      if len(text_parts(raw)) > 1]:
      if zh_match(forms, txt_words(m)):
        asp = zh_facets(m)
        hits = [zh_match(forms, txt_words(x)) for x in asp]
        extra = [x for x, y in zip(asp, hits) if not y and single(x)] if any(hits) else []
        joined.append((m, extra)) if extra else neat.append(m)
    flunk = titled and not near
    if neat: text = "contains '%s' -> MEETS" % brief
    elif ok: text = note.strip()
    elif cap and not titled: text = "the title has '%s' and no field carries the property's name -> MEETS" % brief
    elif joined:
      morsel, aspects = joined[0]
      if flunk:
        fails.append("%s: %s is '%s', which couples %s with %s (a value the store also writes alone there) -> not %s" % (a, named[0], cut(morsel), brief, aspects[0], brief))
      text = "'%s' couples '%s' with '%s', a value other listings hold alone in this field -> FAILS" % (cut(morsel), brief, aspects[0])
    elif writers:
      if flunk and not any(val_touch(req, raw) for _, raw in mine):
        fails.append("%s: %s is '%s', while the store writes '%s' verbatim in that field elsewhere" % (a, named[0], cut(mine[0][1], 60), brief))
      text = "'%s' missing, though other category listings put it in this field -> FAILS (once the store writes the value verbatim, synonyms and related words do not count)" % brief
    elif naming:
      text = "'%s' missing; no read listing has it in this field yet (values on record: %s), but category titles/texts mention '%s' (%s): the store uses a term of its own, so the differing value here is not proven" % (brief, smp, brief, ', '.join(naming))
    elif filled < ZH_PARAPHRASE_MINIMUM:
      text = "'%s' missing from this field on every listing read (values on record: %s), and only %d read listing(s) fill it: too few to tell whether the store writes '%s' word for word, so an equivalent value is not yet proven" % (brief, smp, filled, brief)
    else:
      text = "'%s' missing from this field on every listing read (values on record: %s): a value with the same meaning may count" % (brief, smp)
    lines.append(lead + ' | this listing: ' + shown + ' | ' + text)
  def _hhead(self, first):
    nd = self.needs
    return [first, "\n\nTHE SHOPPER'S REQUEST:\n", self.request, nd.var_opts(), '\n\nCURRENT REQUIREMENTS:\n',
      nd.body_text(), self.ref_note()]
  def _qh_ask(self, kind, seg, terms, who):
    reply = self.cable.consult(''.join(seg), terms, who)
    self.log.append({'kind': kind, 'result': reply})
    return reply
  def rank_check(self, pick):
    nd = self.needs
    aim, notch = nd.priority(), '%s|%s|%s' % (pick, nd.rev_no, self.refusal_n)
    if (aim == 'none' or notch in self.rank_checked or pick in self.pair_best()
        or self.cable.spare() < CLOSING + 30 or self.steps_rem() < 4):
      return ''
    self.rank_checked.add(notch)
    cheap, detail = aim == 'cheapest', nd.parsed.get('priority_detail') or ''
    what = 'cheapest (lowest price)' if cheap else 'best (%s)' % detail if detail else 'best'
    seg = self._hhead(RANK_CHECK.replace('PRIORITY', what))
    if nd.items():
      seg.append('\n\nSEVERAL PRODUCTS' + ': a rival has to be the same product (category) as the chosen item; anything belonging to another product never counts as an upgrade.')
    gloss = ['card legend: DETAILS = the whole record was read', 'ROW = seen only as a search row, fields unknown', 'a ROW within the requested category MAY fit, and belongs in the answer, if it beats the chosen item and neither its title, price nor stock speaks against it, whatever object the title names']
    if cheap: gloss.append('only listings cheaper than the pick get a card')
    seg += ['\n\nTHE CHOSEN ITEM: ' + pick + '\n\nLISTINGS SEEN' + ' (' + '; '.join(gloss) + '):\n',
      '\n'.join(self.rec_db.cards(20000, pick if cheap else '', self.ref_extra(),
      {pick})), self.field_names()]
    if (ev := self.item_view(pick)['rules']):
      seg.append('\n\nFIELD EVIDENCE' + " (for each required value, a code-made summary of how listings in the pick's category write it here):\n" + '\n'.join(ev))
    seg.append(self.code_reads())
    reply = self._qh_ask('rank', seg, INSPECT_TERMS, CHECKER)
    gone = set(self.dead_lines()) | set(self.refused_lines)
    picks = [str(r) for r in (reply or {}).get('better') or [] if isinstance(r, (str, int)) and str(r) not in gone
      and str(r) != pick]
    self.rank_better = (named := [s for s in picks[:4] if not self.item_view(s)['fails']])[:]
    vars(self).setdefault('named', set()).update(named)
    if not named: return ''
    why = reply.get('reason')
    return 'Rank check: %s might also fit and beat %s on the %s%s. Inspect each one (details, stock, price) before ordering; if every one of them fails, go ahead with the current pick.' % (', '.join(named), pick, what, ' (%s)' % why if why else '')
  def _leads(self, strict):
    kv, st, nd = vars(self), self.rec_db, self.needs
    labels, where, units, cap, score = self.shelf_labels(), kv.get('hit_shelf',
      {}), nd.quantity(), nd.spend_cap, kv.get('hit_score', {})
    order = sorted(kv.get('all_hits', []), key=lambda s: -score.get(s, 0))
    if strict: order = dict.fromkeys(order + list(kv.get('named_list', [])))
    return [s for s in order if s not in st.read and not (strict and (s in self.refused_lines or s not in st.ids))
        and not (labels and where.get(s) and where[s] not in labels) and not _over(st.price(s), units, cap)]
  def judge_decline(self, early=False):
    kv, st = vars(self), self.rec_db
    encore = len(st.read) > kv.get('audit_reads', 10 ** 9) and kv.get('audits', 0) < 2
    if not ((encore or not self.audit_done) and self.cable.spare() >= CLOSING + 30): return ''
    self.audit_done = True
    if not early: self.audits = 1 + kv.get('audits', 0)
    last = kv.get('audit_ids')
    fresher = () if last is None else set(st.read) - set(last or ())
    cards = st.cards(16000, '', self.ref_extra(), fresher)
    self.audit_reads = len(st.read)
    if not cards: return ''
    fresh = [] if last is None else sorted(st.read - last)
    self.audit_ids = set(st.read)
    seg = self._hhead(EARLY_DECL) + ['\n\nLISTINGS SEEN', ' (a card marked DETAILS comes from a full record, one marked ROW from a search row alone):\n', '\n'.join(cards)]
    if fresh:
      seg.append('\n\nNEWLY OPENED' + ' (just opened; look at these cards before all others and test every field under the reading rules): ' + ', '.join(fresh))
    seg += [self.code_reads(), self.field_names()]
    reply = self._qh_ask('audit', seg, 6000, AUDITOR) or {}
    ids = lambda k: [str(r) for r in reply.get(k) or [] if isinstance(r, (str, int))]
    fits = self.audit_fits = [s for s in ids('possible_fits')[:5] if not self.item_view(s)['fails']]
    asked = [s for s in ids('unopened') if s not in st.read and s in st.ids]
    hits = self._leads(False)
    try:
      extra = [s for s in self.wide_rows() if s not in st.read]
    except Exception: extra = []
    unopened = list(dict.fromkeys(extra + hits[:4] + [s for s in asked[:6] if s not in hits]))[:6]
    keep = lambda qs: kv.setdefault('named_pages', set()).update(json.dumps(q, sort_keys=True, default=str)
      for q in qs)
    if (pages := [] if self.steps_rem() <= 5 else self.open_pages(2)):
      unopened = unopened[:4]
      keep(pages)
    try:
      combos = [] if self.steps_rem() <= 5 else self.combo_qs()
    except Exception as e:
      self._gerr('combo_error', e)
      combos = []
    if combos: keep(combos)
    leads = fits + unopened
    kv.setdefault('named', set()).update(leads)
    ordered = kv.setdefault('named_list', [])
    ordered.extend([s for s in leads if s not in ordered])
    if not (leads or pages or combos): return ''
    call = lambda qs: '; '.join('%s %s' % (self.role('search'), json.dumps(q, ensure_ascii=False)) for q in qs)
    out = []
    if fits:
      why = reply.get('reason')
      out.append('Second opinion - these may meet every requirement: %s%s.' % (', '.join(fits), ' (%s)' % why if why else ''))
    if pages:
      out.append('Unread result pages - searches with required values that match more listings than were read, so fits may sit further down: ' + call(pages) + '. Each full page of rows costs one step: work through those pages first, then open every row of the requested kind the budget allows.')
    if combos:
      out.append('Combined searches not run yet (every word is mandatory, so results show only listings carrying all of them): ' + call(combos) + '. They still go through the search tool; run them, then open every row of the requested kind the budget allows.')
    if unopened:
      out.append('Seen only as search rows and never opened, so their fields are unchecked (requested kind): ' + ', '.join(unopened) + '. Read their details first; one reply can cover several.')
    return ' '.join(out + ['Settle every lead by its details, stock and price, and buy the one that holds up; a decline is right only once all of them really fail.'])
  def named_unread(self):
    return self._leads(True)
  def details_call(self, symbol):
    tool, known = self.role('details'), self.rec_db.ids
    if not tool or tool == self.role('search'): return None
    sid = lambda v: isinstance(v, str) and v in known
    for d in (d for e in reversed(self.log) if e.get('kind') == 'env' for d in e.get('actions') or []):
      a = d.get('args') if d.get('name') == tool else None
      if isinstance(a, dict) and any(map(sid, a.values())):
        return {k: symbol if sid(v) else v for k, v in a.items()}
    shape = self.argspec.get(tool) or {}
    spec, req = shape.get('properties'), shape.get('required') or []
    strs = [k for k, p in spec.items() if isinstance(p, dict) and p.get('type') == 'string'
      and not p.get('enum')] if isinstance(spec, dict) else []
    choice = next((k for k in strs if k in req), strs[0] if len(strs) == 1 else None)
    return {choice: symbol} if choice else None
  def open_reads(self, queue, say, batch, reader=None):
    if batch:
      name, field, size = batch
      loads = [{field: queue[i:i + size]} for i in range(0, len(queue), size)]
    else:
      name = reader or self.role('details')
      loads = [f for f in (self.fetch_call(s, name) for s in queue) if f] if name else []
    if not loads: return []
    hb, st = self.cable.ticks, self.rec_db
    calls = [{'id': 'open_%d_%s' % (n, hb), 'type': 'function', 'function': {'name': name,
      'arguments': json.dumps(f)}} for n, f in enumerate(loads)]
    self.chat.append({'role': 'assistant', 'content': say, 'tool_calls': calls})
    mark, self.covering = len(st.items), True
    try:
      out = self.relay(calls, programmed=True)
    finally:
      self.covering = False
    self.last_reads = st.items[mark:]
    return out
  def open_named(self):
    kv, cable = vars(self), self.cable
    queue, put_read = kv.pop('open_queue', None), False
    if not queue and kv.get('put_queue') and not (cable.done or cable.broken):
      queue, put_read = kv.pop('put_queue'), True
      kv.update(open_recheck=True, open_purpose='cart', open_tool=kv.pop('put_tool', None), open_say='Reading the stock of the item just put in the cart once more, since a listing can run out right after it is added.')
    if not queue or cable.done or cable.broken: return False
    queue = list(dict.fromkeys(map(str, queue)))
    recheck, purpose, floor, reader, say = (kv.pop('open_recheck', False), str(kv.pop('open_purpose', '') or ''),
      kv.pop('open_floor', None), kv.pop('open_tool', None), kv.pop('open_say', ''))
    say = say or ('Reading the cart lines again before the order.' if recheck else 'Looking at the flagged listings first; no decline yet.')
    if purpose in ('menu', 'quota', 'walk') and self.role('order') and self.role('cart') and not cable.hush:
      try:
        reserve = self.order_due() if self.orders_tried == 0 else FIRST_ORDER_BY
        room = self.steps_rem() - reserve - 1
        batched = lambda: bool(self.batch_tool()) and len(queue) >= 2
        if room < (1 if batched() else len(queue)) and (self.barrow.units or not self.alt_unseen()):
          kept = [] if batched() else queue[:max(0, room)]
          self.log.append({'kind': 'reserve_skip', 'purpose': purpose, 'ids': queue[:6], 'kept': kept})
          if not (queue := kept): return False
      except Exception as e: self._gerr('reserve_skip_error', e)
    if purpose == 'cart':
      expired = set(self.stale_lines())
      if not (queue := [s for s in queue if s in expired or not put_read and s not in self.barrow.units]):
        return False
    en = self.batch_tool() if len(queue) >= 2 else None
    if en and (purpose == 'cart' and not kv.get('batch_proven') or reader and not ((kv.get('tool_caps')
      or {}).get(en[0]) or {}).get('stock')):
      en = None
    resp = self.open_reads(queue, say, en, reader)
    if en and resp and not kv.get('batch_proven'):
      reads = kv.get('last_reads') or []
      if self.batch_ok(reads, queue) and self.batch_denser(reads, queue): self.batch_proven = True
      elif self.ids_back(reads, queue) or not self._batch_shrink(len(queue)):
        self.batch_spec = None
        self.log.append({'kind': 'batch_dropped', 'tool': en[0]})
        reserve = int(floor) if is_num(floor) else self.order_due() if purpose in ('menu',
          'quota') else 3
        n = min({'menu': 0, 'quota': 2, 'walk': 2}.get(purpose, 3), len(queue),
          self.steps_rem() - reserve - 1)
        if n > 0 and not cable.done and (reader or self.role('details')):
          self.last_replies(resp)
          if not (resp := self.open_reads(queue[:n], say, None, reader)): return True
    if not resp: return False
    if (extra := self.judge_decline() if cable.spare() > CLOSING + 30 and not recheck else ''):
      sid, text = resp[-1]
      resp[-1] = (sid, '%s\n[check after opening] %s' % (text, extra))
    self.last_replies(resp)
    return True
  def _scan(self, quiz, take):
    got = []
    for r in self.rec_db.items:
      walk_dicts(r, lambda k: got.append(take(k)) if quiz(k) else None)
    return got
  def last_stock(self, symbol):
    return next((s for s in reversed(self._scan(lambda k: self.row_key(k) == symbol,
      self.stock_state)) if s is not None), None)
  def rank_verify(self, pick):
    try:
      if (nd := self.needs).priority() != 'cheapest': return ''
      rev, cap, left = str(nd.rev_no), nd.spend_cap, 2
      for s in vars(self).get('rank_better') or []:
        if not left or self.cable.spare() < CLOSING + 30: break
        if (s not in self.rec_db.read or '%s|%s' % (s, rev) in self.passed or s in self.refused_lines
            or s in self.dead_lines() or self.last_stock(s) is False
            or self.item_view(s)['fails']):
          continue
        slot = self.prod_idx(s) if nd.items() else None
        units = nd.items()[slot]['quantity'] if slot is not None and slot < len(nd.items()) else nd.quantity()
        if is_num(cost := self.rec_db.price(s)) and not (is_num(units) and _over(cost,
          units, cap)):
          left -= 1
          self.check_item(s, 'ORDER')
      return self.cheaper_fit(pick)
    except Exception as e:
      self._gerr('rank_verify_error', e)
      return ''
  def cheaper_fit(self, pick):
    if self.needs.priority() != 'cheapest' or not pick or pick in self.pair_best(): return ''
    st = self.rec_db
    if (top := st.price(pick)) is None: return ''
    label, gone, rev = self.known_shelf(pick), self.dead_lines(), str(self.needs.rev_no)
    best, low = '', None
    for s, _, t in (p.rpartition('|') for p in self.passed):
      if s == pick or t != rev or s in self.refused_lines or s in gone or self.needs.items() and (not label
        or label != self.known_shelf(s)):
        continue
      if (cost := st.price(s)) is not None and cost < top and (low is None
        or cost < low) and not self.item_view(s)['fails']:
        best, low = s, cost
    return best
  def known_shelf(self, pick):
    labels = self._scan(lambda k: any(isinstance(v, str) and v == pick and 'id' in str(key).lower() for key,
      v in k.items()), self.shelf_for)
    return next((x for x in reversed(labels) if x), '') or str((vars(self.rec_db).get('label_of')
      or {}).get(pick) or '').strip().lower()
  def chosen(self):
    return next(reversed(self.barrow.units), None)
  def home_cats(self):
    if (h := vars(self).get('home_memo')) and h[0] == len(self.rec_db.items): return [set(s) for s in h[1]]
    try:
      res = self._rank_homes()
    except Exception: res = [set() for _ in self.needs.items() or [0]]
    self.home_memo = (len(self.rec_db.items), [set(s) for s in res])
    return res

class ShelfLookups:
  def _rank_homes(self):
    names = [e['product'] for e in self.needs.items()] or [str(self.needs.parsed.get('product') or '')]
    tags, sets = self._lot_labels(), [set(kind_toks(n)) for n in names]
    def best(i, own):
      rivals = set().union(*(w for j, w in enumerate(sets) if j != i))
      score = {t: kind_score(own, set(kind_toks(t)), rivals) for t in sorted(tags)}
      top = max(score.values(), default=0.0)
      peaks = {t for t, v in score.items() if v == top} if top > 0 else set()
      if len(sets) < 2: return peaks
      return {t for t in peaks for x in [set(kind_toks(t))] if all(kind_in(w, x) for w in own)
        and all(kind_in(w, own) for w in x)} or peaks
    return [best(i, w) for i, w in enumerate(sets)]
  def _lot_labels(self):
    ids, found = self.rec_db.ids, set()
    def note(r):
      if (shelf := self.shelf_for(r)) and any('id' in str(k).lower() and isinstance(v, str) and v in ids
        for k, v in r.items()):
        found.add(shelf)
    for r in self.rec_db.items:
      walk_dicts(r, note)
    return found
  @_guard('')
  def own_shelf(self):
    if self.needs.items(): return ''
    home = self.home_cats()[0]
    seen = [s.strip().lower() for s in self.rec_db.shelf_read()]
    if not home or all(s in home for s in seen): return ''
    hit = next((s for s in seen if s in home), None)
    return min(home) if hit is None else hit
  def joint_phrase(self, a, bv, label):
    text = re.sub('\\s+', ' ', self.request.lower().replace('’', "'"))
    if (hit := _label(str(label).lower(), text)): return hit, True
    alt = _pair_txt(a, bv, text)
    return (alt, False) if alt else None
  def prod_question(self, entry, prods):
    line = str(entry)
    m = re.match('-\\s*(.*?): asked "(.*?)" -> ', line, re.S)
    matter, said = m.groups() if m else (line, line)
    lex = [kind_toks(p['product']) for p in prods]
    tags = re.findall('\\(([^()]*)\\)', matter)
    if any(kind_in(e, w) for t in tags for e in kind_toks(t) for w in lex): return True
    low = said.lower().replace('’', "'")
    return any((rx := name_rx(p['product'])) and re.search('\\b' + rx, low) for p in prods)
  def merge_joined(self, obs):
    book = vars(self).setdefault('joined_seen', {})
    walk_dicts(obs, lambda r: _label_log(r, book))
    notes = []
    for _ in range(2):
      prods, cart, nd = self.needs.items(), self.barrow, self.needs
      if len(prods) < 2 or not book or cart.used or cart.units or nd.uncovered or nd.indifferent: break
      if any(self.prod_question(e, prods) for e in self.questions_log): break
      if not (dyad := self._joint_pair(prods, book)): break
      notes.append(self.merge_items(*dyad))
    return ' '.join(n for n in notes if n)
  def _joint_pair(self, prods, book):
    words = [kind_toks(p['product']) for p in prods]
    pairs = [(i, j) for i in range(len(prods)) for j in range(i + 1, len(prods))]
    for key in sorted(book):
      halves = [kind_toks(s) for s in re.split(AND_RX, key) if s.strip()]
      if len(halves) != 2 or not all(halves): continue
      for i, j in pairs:
        x, y = prods[i], prods[j]
        if not _halves(words[i], words[j], halves): continue
        if x['quantity'] != y['quantity'] or _clash(x, y): continue
        found = self.joint_phrase(x['product'], y['product'], book[key])
        if found: return i, j, book[key], found[0], found[1]
  def merge_items(self, first, counterpart, label, phrase, into_requisition=True):
    nd = self.needs
    doc, prods = nd.parsed, nd.items()
    listed = [e for e in doc.get('items') or [] if isinstance(e, dict) and str(e.get('product') or '').strip()]
    if len(listed) != len(prods) or not 0 <= first < counterpart < len(prods): return ''
    many = len(prods) - 1 >= 2
    tag = phrase if many else ''
    ourown, rebadge = self._tag_tools(prods, first, counterpart, label, tag)
    a, b = prods[first], prods[counterpart]
    size = a['quantity']
    joint = dict(product=phrase, quantity=size, requirements=_merge_reqs(a['requirements'] + b['requirements']))
    kept = [joint if i == first else e for i, e in enumerate(listed) if i != counterpart]
    self._relabel_all(ourown, rebadge, phrase, tag)
    if many:
      doc['items'] = kept
      doc['product'] = ' and '.join(str(e.get('product')) for e in kept)
    else:
      doc.pop('items', None)
      doc.update(product=phrase, quantity=size)
    nd.rev_no += 1
    for k in ('home_memo', 'menu_memo', 'quota_reads', 'walked'): vars(self).pop(k, None)
    left, pace = len(nd.items()), 40.0 / max(1, self.step_cap)
    self.cable.scale = (1.0 + 0.3 * min(max(left - 1, 0), 2)) * max(0.5, min(1.0, pace))
    self.log.append(dict(kind='merge_joined', label=label, products=[a['product'], b['product']], into=phrase,
              label_words=into_requisition, products_left=max(1, left)))
    lines = 'one line for that product' if many else 'exactly ONE line'
    txt = f"[merged products] The store files '{label}' as a single category, so '{phrase}' counts as one product here rather than two. The order therefore holds {lines} - do not split it into one line per kind named in that category title. Current requirement list: "
    return txt + nd.body_text()
  def _tag_tools(self, prods, first, counterpart, label, tag):
    pair = kind_toks(prods[first]['product']) + kind_toks(prods[counterpart]['product'])
    lw = kind_toks(label)
    rivals = [kind_toks(p['product']) for i, p in enumerate(prods) if i not in (first, counterpart)]
    sub = ' (%s)' % tag if tag else ''
    def ourown(t):
      w = kind_toks(t)
      return (w and kinds_in(w, lw) and any(kind_in(x, pair) for x in w)
          and not any(_q4_eqs(w, r) for r in rivals))
    def rebadge(text):
      if not isinstance(text, str): return text
      head, colon, body = text.partition(':')
      swap = lambda m: sub if ourown(m.group(1)) else m.group(0)
      return re.sub('\\s*\\(([^()]*)\\)', swap, head) + colon + body
    return ourown, rebadge
  def _relabel_all(self, ourown, rebadge, phrase, tag):
    def redo(seq):
      out = []
      for t in map(rebadge, seq):
        if t not in out: out.append(t)
      return out
    nd = self.needs
    for n in ('written', 'uncovered', 'let_go', 'indifferent'): setattr(nd, n, redo(getattr(nd, n)))
    if nd.cur_reqs is not None: nd.cur_reqs = redo(nd.cur_reqs)
    nd.resolved = list(map(rebadge, vars(nd).get('resolved') or []))
    doc = nd.parsed
    for n in ('requirements', 'hints', 'references', 'soft_wants'):
      if isinstance(doc.get(n), list): doc[n] = redo(doc[n])
    aims = doc.get('reference_targets')
    for t in aims if isinstance(aims, list) else ():
      if isinstance(t, dict) and ourown(str(t.get('product') or '')): t['product'] = phrase
    own = vars(self)
    for h in own.get('refs') or []:
      h.update({n: rebadge(h[n]) for n in ('req', 'head', 'now') if isinstance(h.get(n), str)})
      if ourown(str(h.get('tag') or '')): h['tag'] = tag
    for n in ('pending_seconds', 'open_pair'):
      if isinstance(own.get(n), list): own[n] = list(map(rebadge, own[n]))
  def fetch_key(self, tool):
    spec = self.argspec.get(tool or '') or {}
    props = spec.get('properties')
    if not isinstance(props, dict): props = {}
    req = spec.get('required')
    req = [n for n in req or [] if n in props] if isinstance(req, list) else []
    free = [n for n, i in props.items() if isinstance(i, dict) and i.get('type') == 'string' and not i.get('enum')]
    pick = next((n for n in free if n in req), None) or (free[0] if len(free) == 1 else None)
    return None if not pick or any(n != pick for n in req) else pick
  def fetch_tools(self):
    taken = {self.role(n) for n in _ROLES} - {None}
    return [t for t in self.argspec if t not in taken and self.fetch_key(t)]
  def fetch_call(self, symbol, tool):
    if not tool: return None
    if tool == self.role('details'): return self.details_call(symbol)
    ids = self.rec_db.ids
    lot = lambda v: isinstance(v, str) and v in ids
    past = (d['args'] for e in self.log[::-1] if e.get('kind') == 'env'
        for d in e.get('actions') or [] if d.get('name') == tool and isinstance(d.get('args'), dict))
    for old in past:
      if any(map(lot, old.values())): return {k: symbol if lot(v) else v for k, v in old.items()}
    key = self.fetch_key(tool)
    return {key: symbol} if key else None
  @staticmethod
  def asked_ids(summon_prms):
    if not isinstance(summon_prms, dict): return set()
    return {x for v in summon_prms.values() for x in (v if isinstance(v, list) else [v])
        if isinstance(x, str) and re.fullmatch(ID_RX, x)}
  def infer_caps(self, summon_appellation, summon_prms, obs):
    tool, own = str(summon_appellation), vars(self)
    counts = own.setdefault('tool_caps', {}).setdefault(tool, {'calls': 0, **dict.fromkeys(_CAPS, 0)})
    counts['calls'] += 1
    kinds = (('dead', set), ('dead_by', dict), ('stock_by', dict), ('row_text', dict), ('row_keys', set))
    st = {n: own.setdefault(n, e()) for n, e in kinds}
    st.update(tool=tool, call=summon_appellation, named=self.asked_ids(summon_prms),
         shown=dict.fromkeys(_CAPS, False), keys=own.setdefault('tool_keys', {}).setdefault(tool, set()),
         note_keys=own.setdefault('note_keys', {}).setdefault(tool, set()),
         carts={self.role('cart'), self.role('cart_view')} - {None}, epoch=own.get('cart_seq', 0), newly=[])
    if not (isinstance(obs, dict) and obs.get('error')): walk_dicts(obs, lambda r: self._ipeek(st, r))
    for cap, on in st['shown'].items(): counts[cap] += int(on)
    return st['newly']
  def _ipeek(self, st, rec):
    if not (sid := _q4_sid(rec)): return
    state = self.stock_state(rec)
    asked, cart = sid in st['named'], st['call'] in st['carts']
    if state is not None:
      src = 'lookup' if asked or cart else 'row'
      self.stock_seen = True
      hist = st['stock_by'].setdefault(sid, {})
      other, prev = hist.get('row' if src == 'lookup' else 'lookup'), hist.get('row')
      if other and other[0] == st['epoch'] and other[1] != state: self.rows_lag = True
      if src == 'row' and prev and prev[1] != state: self.rows_live = True
      hist[src] = (st['epoch'], state)
      dead, newly = st['dead'], st['newly']
      if state is not False:
        if src == 'lookup' or st['dead_by'].get(sid) != 'lookup': dead.discard(sid)
      else:
        if sid not in dead and sid in self.barrow.units and sid not in newly: newly.append(sid)
        dead.add(sid)
        st['dead_by'][sid] = src
    if asked:
      shown, tool, own = st['shown'], st['tool'], vars(self)
      shown['id'] = True
      own.setdefault('read_with', {}).setdefault(tool, set()).add(sid)
      texts = (str(k) for k, v in rec.items() if v is None or isinstance(v, str))
      own.setdefault('text_keys', {}).setdefault(tool, set()).update(texts)
      if state is not None: shown['stock'] = True
      if not shown['props']: shown['props'] = any(isinstance(d, dict) and _atoms(d) >= 2 for d in rec.values())
      if not shown['price']:
        shown['price'] = any(is_num(v) and re.search('price|cost', str(k).lower())
                  for k, v in rec.items())
      st['keys'].update(map(str, rec))
      seen = st['row_text'].get(sid)
      for k, v in rec.items():
        if not isinstance(v, str) or len(v.split()) < 8: continue
        if (str(k) not in st['row_keys']) if seen is None else seen.get(str(k)) != v:
          st['note_keys'].add(str(k))
          shown['notes'] = True
    elif not cart:
      st['row_keys'].update(map(str, rec))
      st['row_text'].setdefault(sid, {}).update((str(k), v) for k, v in rec.items() if isinstance(v, str))
  def stock_tool(self):
    counts = vars(self).get('tool_caps') or {}
    stash = self.fetch_tools()
    seen = lambda t, c: (counts.get(t or '') or {}).get(c, 0)
    if (shown := [t for t in stash if seen(t, 'stock') > 0]):
      view, det = self.role('view'), self.role('details')
      return max(shown, key=lambda t: (counts[t]['stock'], t == view, t == det, t))
    guess = self._iguess()
    for c in (self.role('view'), guess.get('view'), self.role('details')):
      if c and c in stash and (seen(c, 'id') < 2 or seen(c, 'stock') != 0): return c
    told = {guess.get(n) for n in ('cart', 'cart_view', 'order', 'decline', 'message')} - {None}
    return next((t for t in stash if t not in told and not seen(t, 'id')), None)
  def _iguess(self):
    if (guess := vars(self).get('role_guess')) is None:
      try: guess = self.guess_limits()
      except Exception: guess = {}
      self.role_guess = guess
    return guess
  def notes_tool(self):
    own = vars(self)
    counts, reader = own.get('tool_caps') or {}, self.role('details')
    usual = set((own.get('tool_keys') or {}).get(reader or '', set()))
    noted = own.get('note_keys') or {}
    extra = [t for t in self.fetch_tools() if t != reader and set(noted.get(t) or set()) - usual]
    return max(extra, key=lambda t: ((counts.get(t) or {}).get('notes', 0), t)) if extra else None
  def sheet_cap(self):
    own, reader = vars(self), self.role('details')
    if own.get('sheet_moved') or not reader: return
    counts = own.get('tool_caps') or {}
    mine = counts.get(reader) or {}
    if mine.get('id', 0) < 2 or mine.get('props', 0): return
    denser = [t for t in self.fetch_tools() if t != reader and (counts.get(t) or {}).get('props', 0)]
    if not denser: return
    sheet = max(denser, key=lambda t: (counts[t].get('props', 0), t))
    self.sheet_moved = True
    self.role_map['details'] = sheet
    if self.role('view') == sheet: self.role_map['view'] = reader
    keep = set()
    def take(r):
      if (sid := _rec_id(r)) and _has_pmap(r): keep.add(sid)
    for obs in self.rec_db.items: walk_dicts(obs, take)
    self.rec_db.read &= keep
    self.log.append({'kind': 'sheet_role', 'from': reader, 'to': sheet})
  def dead_lines(self): return set(vars(self).get('dead') or ())
  def stale_lines(self):
    own = vars(self)
    seq, notch = own.get('line_seq') or {}, own.get('cart_seq', 0)
    return [ln for ln in self.barrow.units if seq.get(ln, -1) < notch]
  def _cost(self, cand):
    p = self.rec_db.price(cand)
    return float(p) if is_num(p) else float('inf')
  def dead_swap(self, line):
    dead, ver, cart, nd = self.dead_lines(), str(self.needs.rev_no), self.barrow, self.needs
    kind, many = self.known_shelf(line), len(nd.items()) >= 2
    cap, qty, inf = nd.spend_cap, max(1.0, cart.held(line)), float('inf')
    rest = sum(p * cart.held(h) for h in cart.units if h != line and (p := self._cost(h)) != inf)
    barred = lambda c: c == line or c in dead or c in self.refused_lines or c in cart.units
    over = lambda v: is_num(cap) and v * qty + rest > cap
    opts = []
    for n in self.passed:
      c, _, v = n.rpartition('|')
      if v != ver or barred(c) or self.item_view(c)['fails']: continue
      shelf = self.known_shelf(c)
      if kind and (shelf != kind if many else shelf not in ('', kind)): continue
      if (vol := self._cost(c)) == inf or not over(vol): opts.append((vol, c))
    pick, cheap = min(opts)[1] if opts else '', nd.priority() == 'cheapest'
    self.swap_mix = ''
    if pick and not cheap: return pick, ''
    if many and not kind: return pick, ''
    k = next((i for i, home in enumerate(self.home_cats()) if kind in home), None) if many else None
    told = [(n, s) for n, s in self.need_vals(k) if not _LOOSE.search(s.lower())]
    failed, ptrs, ranked = self._failed_at(ver), self.ref_ptrs(), []
    for c, slip in self.all_rows().items():
      f, vol = slip['fields'], slip['price'] if is_num(slip['price']) else self._cost(c)
      if barred(c) or slip['stock'] is not True or (kind and slip['shelf'] != kind): continue
      if vol == inf or over(vol) or c in failed or c in ptrs or self.ref_excluded(f, k): continue
      opened = c in self.rec_db.read
      if opened and (self.item_view(c)['fails'] or any(self.field_clash(f, n, s) for n, s in told)
             or not all(self.row_shows(slip, n, s, set(), True) for n, s in told)): continue
      ranked.append((0 if opened else 1, vol, c))
    ranked.sort()
    unread = [e for e in ranked if e[0] == 1]
    if unread and nd.priority() == 'cheapest' and all(unread[0][1] < p for m, p, _ in ranked if m == 0):
      ranked.remove(unread[0])
      ranked.insert(0, unread[0])
    say = lambda m, v, c: f'{c} at {int(v) if float(v).is_integer() else v}' + (
      ' [opened; every stated value shows]' if m == 0 else '')
    self.swap_page = None
    # cheapest wanted and steps to spare: a further page of the stated-value search may hold a cheaper fit
    if all(m != 0 for m, _, _ in ranked) or cheap and self.steps_rem() >= 8:
      try: self.swap_page = self.dead_page(k)
      except Exception: self.swap_page = None
    if pick:
      # an approved line is not the cheapest when opened rows that show every stated value cost less
      pv, self.swap_unread = self._cost(pick), []
      lower = [e for e in ranked if e[0] == 0 and e[1] < pv][:2] if pv != inf else []
      if not lower: return pick, ''
      self.swap_mix = say(0, pv, pick).replace(' [opened; every stated value shows]', '')
      return '', ', '.join(say(*e) for e in lower)
    top = ranked[:3]
    self.swap_unread = [c for m, _, c in top if m == 1]
    return '', ', '.join(say(*e) for e in top)
  def dead_page(self, k):
    if not (pg := self.pg_arg()): return None
    best = top = None
    for w in vars(self).get('page_book', {}).values():
      if not (borne := self.walk_has(w['args'], k)) or self.next_page(w) is None: continue
      rank = (len(borne), -len(w['read']))
      if best is None or rank > top: best, top = w, rank
    return None if best is None else {**best['args'], pg[0]: self.next_page(best)}
  def swap_open(self):
    own = vars(self)
    if own.get('swap_done'): return ''
    dead = self.dead_lines()
    return next((c for c in own.get('swap_unread') or []
          if not (c in self.rec_db.read or c in dead or c in self.barrow.units)), '')
  def dead_note(self, lines):
    self.event_seen = True
    return ' '.join(map(self._dead_say, lines))
  def _dead_say(self, held):
    self.swap_page, self.swap_unread = None, []
    pick, rows = self.dead_swap(held)
    head = f'[stock] Cart line {held} cannot be bought any more: its newest record shows it unavailable, and a checkout that still includes it would be refused. Plan:'
    tail = ' (c) confirm its availability and price with one more read; (d) place the order.'
    if pick:
      cheap = ', and it is also the cheapest approved option' if self.needs.priority() == 'cheapest' else ''
      txt = f'{head} (a) remove {held}; (b) add {pick} instead - the item check approved it under the current requirement list{cheap};' + tail
    elif rows and (mix := vars(self).get('swap_mix')):
      txt = (f'{head} (a) remove {held}; (b) the cheapest wish decides: the item check approved {mix}, but these opened rows '
        f'cost less and the code finds every stated value in their records: {rows}. Add the cheapest of them that its record '
        f'confirms (item check), or {mix.split(" at ")[0]} if none does;' + tail)
    elif rows:
      txt = f'{head} (a) remove {held}; (b) no approved replacement for this product exists yet, so work through these cheap, available rows of the same category that still fit the spending limit, in this order: {rows} (rows whose opened record misses or contradicts a stated requirement have already been left out; open the details of an unopened one before adding it) and add the first one that passes;' + tail
    else:
      txt = f'{head} remove {held} and search for another listing that meets the request.'
    if not (page := vars(self).get('swap_page')) or pick and self.needs.priority() != 'cheapest': return txt
    more = ' More rows of this kind exist - the search led by a stated value still has a page not yet read; call it with '
    return txt + more + json.dumps(page, default=str) + '.'
  def refuse_redo(self, obs=None):
    try: note, gone = self.refuse_stale(obs)
    except Exception as e:
      self._qj_err('refusal_gone_error', e)
      return self.refuse_queue()
    return note + (gone or self.refuse_queue())
_LOOSE = re.compile(
  '\\b(?:' + BOUND_WORDS + ')\\b|\\d\\s*(?:-|to)\\s*\\d|[+<>]|\\b(?:about|around|approx\\w*|roughly)\\b')
_MARK_RX = re.compile('\\((?:stated|revealed[^)]*)\\)')
_DONE = ' Next: place the order for exactly this cart. Use the decline only if a record you open shows a cart line to be wrong under the current requirements.'
_CAPS = ('id', 'stock', 'props', 'notes', 'price')
_ROLES = ('search', 'message', 'cart', 'cart_view', 'order', 'decline')
_WORD3 = re.compile('[a-z0-9]{3,}')
def _catch(kind, empty):
  def wrap(fn):
    def run(self):
      try: return fn(self)
      except Exception as e:
        self._qj_err(kind, e)
        return empty()
    return run
  return wrap
def _label(label, voiced):
  segs = [s for s in re.split(AND_RX, label) if s.strip()]
  if len(segs) != 2: return ''
  left, right = name_rx(segs[0]), name_rx(segs[1])
  if not left or not right: return ''
  hit = re.search('\\b%s\\s*%s\\s*%s' % (left, '(?:\\band\\b|&|\\bor\\b|/|\\+)', right), voiced)
  return hit.group(0).strip() if hit else ''
def _pair_txt(a, bv, voiced):
  ta, tb = kind_toks(a), kind_toks(bv)
  if not ta or not tb or same_kind(ta[-1], tb[-1]): return ''
  for x, y in ((a, bv), (bv, a)):
    rx = '\\b(?:%s)(?P<mid>[^.;:!?]{0,40}?)(?:%s)' % (name_rx(x), name_rx(y))
    for m in re.finditer(rx, voiced):
      mid = m.group('mid')
      if len(re.findall('[a-z0-9]+', mid)) > 3 or not re.search('\\band\\b|&|\\bor\\b|,|/|\\+', mid): continue
      if re.search('\\d|\\bx\\b', mid): continue
      rest = voiced[:m.start()] + ' | ' + voiced[m.end():]
      if not all(re.search('\\b' + name_rx(n), rest) for n in (a, bv)): return m.group(0).strip()
  return ''
def _clash(a, bv):
  def pairs(p):
    return [(re.split('\\s*\\(', h)[0].strip(), ' '.join(re.findall('[a-z0-9]+', b.lower())))
        for h, _, b in (str(r).partition(':') for r in p.get('requirements') or [])]
  two = pairs(bv)
  return any(same_prop(x, y) and u != v for x, u in pairs(a) for y, v in two)
def _label_log(rec, book):
  if not any('id' in str(k).lower() and isinstance(v, str) and re.fullmatch(ID_RX, v) for k, v in rec.items()):
    return
  for k, v in rec.items():
    low = str(k).lower()
    if not isinstance(v, str) or not 0 < len(v.strip()) <= 60 or 'id' in low: continue
    if not re.search('categor|shelf|department|aisle', low): continue
    text = v.strip()
    if len([s for s in re.split(AND_RX, text.lower()) if s.strip()]) == 2: book.setdefault(text.lower(), text)
    return
def _q4_eqs(x, y): return kinds_in(x, y) and kinds_in(y, x)
def _halves(tav, tbl, halves):
  if not (tav and tbl) or _q4_eqs(tav, tbl) or not kinds_in(tav + tbl, halves[0] + halves[1]): return False
  oa, ob = [w for w in tav if not kind_in(w, tbl)], [w for w in tbl if not kind_in(w, tav)]
  side = lambda ws, near, far: all(kind_in(w, near) and not kind_in(w, far) for w in ws)
  return any(side(oa, n, r) and side(ob, r, n) and any(kind_in(w, n) for w in tav)
       and any(kind_in(w, r) for w in tbl) for n, r in (halves, halves[::-1]))
def _merge_reqs(rqs):
  kept = {}
  for req in rqs:
    head, _, body = str(req).partition(':')
    line = '%s:%s' % (re.sub('\\s*\\([^()]*\\)', '', head).strip(), body)
    kept.setdefault(line.lower(), line)
  return list(kept.values())
def _q4_sid(dx):
  ids = [(str(k).lower(), v) for k, v in dx.items()
     if str(k).lower().endswith('id') and isinstance(v, str) and re.fullmatch(ID_RX, v)]
  return next((v for k, v in ids if k == 'id'), ids[0][1] if ids else None)
def _atoms(d): return len([v for v in d.values() if isinstance(v, (str, int, float))])
def _firm(text): return _LOOSE.search(str(text).lower()) is None
def _word(text):
  low = str(text).lower()
  return (re.search('[a-z]{3,}', low) is not None and re.search('\\d', low) is None and _firm(low)
      and re.search(NEG_WORDS, low) is None and not is_ref_text(low))
def _lowest(stash):
  return min(((e['price'], e['id']) for e in stash if e['fit'] and e['price'] is not None), default=None)
def _deal(lineups, limit):
  got = []
  for depth in range(max(map(len, lineups), default=0)):
    if len(got) >= limit: break
    got.extend(q[depth] for q in lineups if depth < len(q))
  return got[:limit]
def _batch_arg(params):
  if not isinstance(params, dict): return None
  props = params.get('properties')
  if not isinstance(props, dict): props = {}
  names = [n for n, e in props.items() if isinstance(e, dict) and e.get('type') == 'array'
      and isinstance(i := e.get('items'), dict) and i.get('type') == 'string' and not i.get('enum')]
  if len(names) != 1: return None
  req = params['required'] if isinstance(params.get('required'), list) else []
  if any(n not in names for n in req): return None
  top = props[names[0]].get('maxItems')
  return (names[0], int(min(top, RUNDOWN_SKIM_PEAK))) if is_num(top) and top >= 2 else None
class RefusalFixes:
  def _qj_jot(self, kind, **fields):
    self.log.append(dict(kind=kind, **fields))
  def _qj_err(self, kind, e):
    self._qj_jot(kind, error=('%s: %s' % (type(e).__name__, e))[:300])
  def _muted(self): return bool(self.cable.hush or vars(self).get('covering'))
  def _qj_for(self, req, product):
    head = str(req).partition(':')[0]
    return not self.needs.items() or self.prod_of(head) in (product, None)
  def _failed_at(self, ver):
    return {n.split('|')[0] for n, bad in self.rulings.items() if bad and n.split('|')[1:2] == [ver]}
  def _live_at(self, ver): return {h for n in self.passed for h, _, t in [n.rpartition('|')] if t == ver}
  def _stamp(self):
    nd = self.needs
    return nd.rev_no, len(nd.let_go), len(nd.uncovered), str(nd.spend_cap), '|'.join(nd.props_left())
  def refuse_stale(self, reply):
    cart = self.barrow
    held = [ln for ln in cart.units if cart.held(ln) > 0]
    if reply is None or len(held) != 1 or self._muted() or vars(self).get('last_kind') != 'order': return '', ''
    if not (texts := self.buyer_lines(self.role('order') or '', reply, True)): return '', ''
    before = self._stamp()
    note = self.needs.heed(texts, None)
    after = self._stamp()
    lead = ' ' + note if note else ''
    if not note or after[0] == before[0] or after[1:] != before[1:]: return lead, ''
    line, own = held[0], vars(self)
    own.setdefault('dead', set()).add(line)
    own.setdefault('dead_by', {})[line] = 'lookup'
    self._qj_jot('refusal_dead', line=line)
    why = " Marked sold out without a fresh read: %s was the only line in the cart, and the store's refusal says an item is no longer available. " % line
    return lead, why + self.dead_note([line])
  def refuse_queue(self):
    reader = self.stock_tool()
    lines = list(self.barrow.units)[:4]
    if not lines or not reader or self._muted() or vars(self).get('open_queue'): return ''
    if self.steps_rem() < 3 + len(lines) or self.cable.spare() < CLOSING: return ''
    vars(self).update(open_queue=lines, open_recheck=True, open_purpose='cart', open_tool=reader,
             open_say="The order was refused, so every cart line's stock gets read again in case one has sold out.")
    return ' Stock re-read queued: every cart line is looked up again with %s.' % reader
  def rate_all(self):
    count = len(self.needs.items())
    judged = sum(sum(_firm(d[1]) for d in self.need_vals(s)) for s in (range(count) if count else [None]))
    done = self.differ_done(None)
    need = len(list(self.needs.props_left()))
    hit = sum(1 for r in self.needs.props_left() if any(self.ref_match(h, r) for h in done))
    return judged + hit >= need
  def differ_done(self, product):
    many, want = bool(self.needs.items()), self.needs.props_left()
    found = []
    for h in vars(self).get('refs') or []:
      if (h.get('relation'), h.get('status')) != ('differ', 'resolved'): continue
      if str(h.get('value') or '').strip() == '' or not self.ref_live(h): continue
      if not any(self.ref_match(h, r) for r in want): continue
      tag = h.get('tag') if many and product is not None else None
      if not tag or self.prod_of('x (' + str(h['tag']) + ')') in (product, None): found.append(h)
    return found
  def decline_repair(self):
    refused = self.refusal_n
    many = len(self.needs.items()) >= 2
    reserve = self.steps_rem() <= self.order_due()
    notch = refused if refused > 0 or reserve else ('early', refused) if refused == 0 else None
    if notch is None: return ''
    wx = self.cable
    if any((not self.role('cart'), not self.role('order'), self.steps_rem() < 3, wx.spare() <= CLOSING,
        self._muted(), wx.done)): return ''
    tried = vars(self).setdefault('repair_tried', set())
    if notch in tried: return ''
    tried.add(notch)
    if many:
      try: bad = self.zd_verify()
      except Exception as e:
        bad = []
        self._qj_err('decline_proof_error', e)
      if bad:
        self._qj_jot('decline_proof', products=bad)
        return ''
    dead, stale, saved = self.dead_lines(), set(self.stale_lines()), set(self.refused_lines)
    good = {ln for ln in self.barrow.units if not (ln in dead or ln in stale)}
    ver, own, avoid, seen = str(self.needs.rev_no), self.rate_all(), set(dead), {}
    fits = lambda s: own and s in self.rec_db.read and (vars(self).get('code_fit') or {}).get(s)
    ok = lambda s: f'{s}|{ver}' in self.passed or bool(fits(s))
    bare = lambda e: not e['revealed_ok'] or len(e['misses']) > 0
    placed = False
    try:
      self.refused_lines.difference_update(good)
      if many:
        for p in range(len(self.needs.items())): seen.update((e['id'], e) for e in self.zd_pool(p))
        avoid.update(s for s, e in seen.items() if e['clash'])
      checks = rounds = 0
      while True:
        picks = self.code_picks(avoid)
        if not (many and picks) or checks >= 3 or rounds >= 30 or self.cable.spare() <= CLOSING + 30: break
        if not (untested := [p[0] for p in picks if not ok(p[0])]): break
        if (drop := [s for s in untested if not seen.get(s) or bare(seen[s])]):
          avoid.update(drop)
          avoid.update(s for s, e in seen.items()
                if not e['read'] and f'{s}|{ver}' not in self.passed and bare(e))
          rounds += 1
        else:
          checks += 1
          self.check_item(untested[0], 'ORDER')
      if not (picks and all(ok(p[0]) for p in picks)): self._qj_jot('decline_repair', picks=picks, done=False)
      elif self.steps_rem() < 2 + len(picks):
        self._qj_jot('decline_repair', picks=picks, done=False, why='steps')
      else:
        placed = self._place(picks)
        self._qj_jot('decline_repair', picks=picks, done=placed, cart=dict(self.barrow.units))
    except Exception as e:
      placed = False
      self._qj_err('decline_repair_error', e)
    finally:
      if not placed:
        self.refused_lines.clear()
        self.refused_lines.update(saved)
    if not placed: return ''
    scope, pron = (' for every product', 'them') if self.needs.items() else ('', 'it')
    txt = 'Not run yet (no step used). Decline held: the records contain a fitting candidate%s - the item check passed it, or its opened record shows every requested value - and sending the shopper away is only right when nothing fits. The code has added %s to the cart and re-checked the stock. Cart: %s'
    return txt % (scope, pron, self.barrow.recap()) + _DONE
  def _place(self, picks):
    cart, wx = self.barrow, self.cable
    hush, wx.hush = wx.hush, True
    try:
      ok = self.plan_bad(picks) == [] and all(p[0] in cart.units for p in picks)
      want = {p[0] for p in picks}
      stale = [ln for ln in self.stale_lines() if ln in want]
      if ok and stale and self.steps_rem() - len(stale) >= 2:
        _, sold = self.code_recheck(stale)
        ok = not sold
    finally:
      wx.hush = hush
    return ok and len(cart.units) > 0 and not set(cart.units) & self.dead_lines()
  def zd_rqs(self, product):
    w = self.needs.written
    return [(_BRK.split(h)[0].strip(), _MARK_RX.sub('', b).strip(), r not in w)
        for r in self.needs.props_left() if self._qj_for(r, product)
        for h, _, b in [str(r).partition(':')]]
  def zd_pool(self, product):
    nd, docs = self.needs, self.rec_db
    many, ver = bool(nd.items()), str(nd.rev_no)
    key = (product, ver, len(docs.items), len(docs.read), len(self.passed),
       len([v for v in self.rulings.values() if v]), '|'.join(nd.props_left()), str(nd.spend_cap),
       len(vars(self).get('refs') or []), frozenset(self.dead_lines()), frozenset(self.refused_lines))
    memo = vars(self).setdefault('ed_memo', {})
    if key in memo: return memo[key]
    if len(memo) > 64: memo.clear()
    homes = self.home_cats()
    if many: shelves = set(homes[product]) if product is not None and product < len(homes) else set()
    else: shelves = set(self.shelf_labels()) or set(homes[0] if homes else ())
    memo[key] = self._jrows(product, many, ver, shelves) if shelves else []
    return memo[key]
  def _jrows(self, k, many, ver, home):
    counts = self.needs.counts()
    units = counts[k] if many and k is not None and k < len(counts) else counts[0] if counts else 1
    scope = k if many else None
    values = self.need_vals(scope)
    joined = [(p, segs) for p, v in values if '+' in v for segs in [[s.strip() for s in v.split('+') if s.strip()]]
         if len(segs) >= 2 and all(re.search('[a-z]{2,}', s.lower()) for s in segs)]
    budget = self.needs.spend_cap
    failed, live = self._failed_at(ver), self._live_at(ver)
    cap = float(budget) if is_num(budget) else None
    judged = [d for d in values if _firm(d[1])]
    shown = [(p, v) for p, v, new in self.zd_rqs(scope) if new]
    ptrs, dead, docs, out = self.ref_ptrs(), self.dead_lines(), self.rec_db, []
    for sid, row in [d for d in self.all_rows().items() if d[1]['shelf'] in home]:
      seen = sid in docs.read
      p = docs.price(sid) if row['price'] is None else row['price']
      price = float(p) if is_num(p) else None
      gone = sid in dead or row['stock'] is False or sid in self.refused_lines
      over = not (cap is None or price is None) and price * units > cap + 1e-09
      ptr = sid in ptrs or self.ref_excluded(row['fields'], scope)
      clash = self._jclash(row['fields'], judged, joined)
      misses = [d for d in judged if not self.row_shows(row, d[0], d[1], set(), True)]
      ruled = sid in failed or bool(seen and self.item_view(sid)['fails'])
      verified = sid in live and not gone
      fit = (verified or (seen and not misses and not clash)) and not (ruled or ptr or gone or over)
      out.append(dict(id=sid, price=price, read=seen, verified=verified, fit=fit, gone=gone, over=over,
              pointer=ptr, ruled=ruled, clash=clash, misses=misses, row=row,
              out=bool(gone or over or ptr or ruled or clash),
              revealed_ok=all(self.row_shows(row, p, v, set(), True) for p, v in shown)))
    return out
  def _jclash(self, fields, judged, joined):
    found = []
    for p, v in judged:
      if not self.field_clash(fields, p, v): continue
      said = [x for n, x in fields.items() if same_prop(n, p)]
      if not any(self.code_seen(v, x) for x in said): found.append(p)
    for p, segs in joined:
      said = [x for n, x in fields.items() if same_prop(n, p)]
      if said and not all(any(self.code_seen(s, x) for x in said) for s in segs): found.append(p)
    return found
  def zd_verify(self):
    return [p for p in range(len(self.needs.items()))
        if {e['out'] for e in self.zd_pool(p) if e['read']} == {True}]
  def wide_rows(self):
    nd = self.needs
    n, budget = len(nd.items()), nd.spend_cap
    if n < 2 or not is_num(budget): return []
    try:
      counts = nd.counts()
      units = [counts[k] if k < len(counts) else 1 for k in range(n)]
      mins = []
      for k in range(n):
        if (low := _lowest(self.zd_pool(k))) is None: return []
        mins.append(low[0] * units[k])
      cap, total = float(budget), sum(mins)
      if total <= cap + 1e-09: return []
      treks = list(vars(self).get('page_book', {}).values())
      hits = set(vars(self).get('all_hits') or [])
      lineups = []
      for k in range(n):
        room = (cap - (total - mins[k])) / max(1, units[k])
        lead = self._jlead(k, treks, hits)
        ok = [e for e in self.zd_pool(k) if not any(map(e.get, ('read', 'gone', 'pointer', 'ruled', 'clash')))
           and e['price'] is not None and room > 0 and e['price'] <= room + 1e-09]
        ok.sort(key=lambda e: (-lead.get(e['id'], 0), e['price'], e['id']))
        lineups.append([e['id'] for e in ok])
      if (picked := _deal(lineups, 3)): self.slack_total = total
      return picked
    except Exception as e:
      self._qj_err('slack_error', e)
      return []
  def _jlead(self, k, treks, hits):
    word = re.compile('[a-z0-9]{2,}')
    sought = [set(word.findall(t[1].lower())) - STOP_WORDS for t in self.zd_rqs(k)]
    lead = dict.fromkeys(hits, 1)
    for w in treks:
      toks = [x for t in self.arg_texts(w['args']) for x in word.findall(t.lower())]
      cut = self.lead_words
      heard = set(toks[:cut] if cut else toks)
      if (g := sum(1 for s in sought if s & heard)) <= 0: continue
      for s in w.get('row_shelf') or {}: lead[str(s)] = max(g, lead.get(str(s), 0))
    return lead
  def verify_issues(self, listing):
    for e in self.log[::-1]:
      if not isinstance(e, dict) or (e.get('kind'), e.get('item')) != ('verify', listing): continue
      r = e.get('result')
      if not isinstance(r, dict) or r.get('fits') is not False: return None
      return [t for t in map(str, r.get('problems') or []) if t.strip()]
  @_catch('near_hold_error', str)
  def near_keep(self):
    tote = vars(self).get('near_cart')
    if not tote or (tote.get('ver'), tote.get('refusals')) != (self.needs.rev_no, self.refusal_n): return ''
    if not self.role('order'): return ''
    if not (ids := list(tote.get('lines') or [])): return ''
    if any(s not in self.barrow.units or s in self.dead_lines() or s in self.refused_lines for s in ids):
      return ''
    wx = self.cable
    if self.steps_rem() < 1 or wx.done or wx.broken or wx.spare() < 2: return ''
    self.near_cart = None
    self._qj_jot('near_order', lines=ids)
    hush = wx.hush
    try: self.finish_now()
    finally: wx.hush = hush
    txt = 'Not run (no step used). The code placed the order itself instead of declining: the wording of the cart lines (%s) was accepted earlier by a meaning check as naming the requested values, and neither the needs nor the cart have changed since.'
    return txt % ', '.join(ids)
  def near_check(self):
    wx = self.cable
    if any((vars(self).get('near_done'), not self.role('cart'), not self.role('order'), self.steps_rem() < 4,
        wx.spare() <= CLOSING + 30, self._muted(), wx.done, wx.spent_out(), wx.no_model)):
      return ''
    try: return self.near_basket()
    except Exception as e:
      self._qj_err('near_error', e)
      return ''
  def near_basket(self):
    nd = self.needs
    count, budget = len(nd.items()), nd.spend_cap
    prods, counts = list(range(count)) if count else [None], nd.counts()
    cap = float(budget) if is_num(budget) else None
    chosen, pools = {}, {}
    units = lambda n: counts[n] if n < len(counts) else 1
    for n, product in enumerate(prods):
      stash = self.zd_pool(product)
      if (best := _lowest(stash)) is not None:
        chosen[n] = (best[1], best[0], units(n))
        continue
      if not (self.refusal_n > 0 or any(self._qj_for(r, product) for r in [*nd.uncovered, *nd.let_go])): return ''
      pools[n] = self._near_pool(product, stash, 6 if len(prods) == 1 else 3)
      if not pools[n]: return ''
    if not pools: return ''
    low = sum(p[1] * p[2] for p in chosen.values()) + sum(min(c[1] for c in pools[n]) * units(n) for n in pools)
    if cap is not None and low > cap + 1e-09: return ''
    flat = [(n, c[2]) + line for n in sorted(pools) for c in pools[n] for line in c[3]][:12]
    listed = '\n'.join("%d. %s: %s = '%s' | store wording: %s" % (i, r[1], r[2], r[3][:60], r[4]) for i, r in enumerate(flat, 1))
    head = 'OUTPUT: exactly one JSON object and nothing else: {"lines": [{"n": 1, "same": true}]} - one entry per numbered line below; set "same" to false whenever you are unsure.\n\nTASK: an order is about to be declined because some listings word a requested value differently from the request. For each line, decide whether the value in the listing\'s field means the requested value (a synonym, another spelling or phrasing, or a more precise form of the same thing) - same: true - or names a different value of that property - same: false.\n\nSHOPPER REQUEST (context only):\n'
    tail = "\n\nLINES (number. listing id: property = 'requested value' | store wording: the text the listing itself uses):\n"
    prompt = head + self.request[:1500] + tail + listed
    self.near_done = True
    reply = self.cable.consult(prompt, 600, CHECKER)
    got = reply.get('lines') if isinstance(reply, dict) else None
    agreed = {int(i['n']) - 1 for i in (got if isinstance(got, list) else [])
         if isinstance(i, dict) and i.get('same') is True and is_num(i.get('n'))}
    self._qj_jot('near_check', lines=[('%s:%s=%s|%s' % r[1:])[:120] for r in flat], same=sorted(agreed))
    for n in sorted(pools):
      for c in pools[n]:
        at = [i for i, r in enumerate(flat) if (r[0], r[1]) == (n, c[2])]
        if len(at) == len(c[3]) and agreed.issuperset(at):
          chosen[n] = (c[2], c[1], units(n))
          break
      else: return ''
    if cap is not None and sum(p[1] * p[2] for p in chosen.values()) > cap + 1e-09: return ''
    picks = [chosen[n][::2] for n in range(len(prods))]
    if self.steps_rem() < len(picks) + 2: return ''
    ver, own = str(nd.rev_no), vars(self)
    for n in pools:
      sid = chosen[n][0]
      key = f'{sid}|{ver}'
      self.rulings.update(dict.fromkeys([x for x in self.rulings if (x + '|').startswith(key + '|')], ''))
      self.passed.add(key)
      own.setdefault('near_ok', set()).add(key)
      own.setdefault('pass_price', {})[sid] = self.rec_db.price(sid)
    placed = self._place(picks)
    self._qj_jot('near_basket', picks=picks, done=placed, cart=dict(self.barrow.units))
    if not placed: return ''
    self.near_cart = dict(ver=self.needs.rev_no, refusals=self.refusal_n, lines=[p[0] for p in picks])
    notes = own.setdefault('near_said', {})
    for r in flat:
      if r[1] == chosen.get(r[0], ('',))[0]: notes.setdefault(f'{r[1]}|{ver}', []).append(r[2:])
    won = {n: p[0] for n, p in chosen.items()}
    said = "%s: its %s field says '%s', read as '%s'"
    aligned = [said % (r[1], r[2], r[4], r[3][:60]) for r in flat if won.get(r[0], '') == r[1]]
    txt = 'Not run yet (no step used). Decline held. A meaning check confirmed these wording matches: %s. The listings cover every need the shopper revealed, are in stock and keep the basket inside the spending limit; the code has added them to the cart and re-checked their stock. Cart: %s'
    return txt % ('; '.join(aligned), self.barrow.recap()) + _DONE
  def _near_pool(self, product, stash, keep):
    rqs, ranked = self.zd_rqs(product), []
    for e in stash:
      if not e['read'] or e['price'] is None or e['gone'] or e['over'] or e['pointer']: continue
      row, lines = e['row'], []
      for p, v, new in rqs:
        if self.row_shows(row, p, v, set(), True): continue
        said = [t for t in (str(c).strip() for n, c in row['fields'].items() if same_prop(n, p)) if t]
        if new or not said or not _word(v) or any(re.search('\\d', t) for t in said):
          lines = None
          break
        lines.append((p, v, '; '.join(dict.fromkeys(said))[:120]))
      heads, probs = set(), []
      if e['ruled']:
        heads = {str(g).partition(':')[0].strip() for g in self.item_view(e['id'])['fails']}
        probs = self.verify_issues(e['id']) or []
      if not lines or not all(any(same_prop(h, ln[0]) for ln in lines) for h in heads): continue
      words = set()
      for p, v, _ in lines: words |= set(prop_words(p)) | set(re.findall('[a-z0-9]+', v.lower()))
      starts = [re.compile('\\b' + re.escape(w)) for w in words if len(w) > 2 and w not in STOP_WORDS]
      if all(any(s.search(pr.lower()) for s in starts) for pr in probs):
        ranked.append((len(lines), e['price'], e['id'], lines))
    return sorted(ranked)[:keep]
  def cart_done(self, former):
    own = vars(self)
    self.cart_ver, self.since_cart = own.get('cart_ver', 0) + 1, 0
    now = self.barrow.units
    if not set(now) <= set(former): self.cart_seq = own.get('absorb_n', 0)
    reads = own.setdefault('line_seq', {})
    for ln in [ln for ln in now if ln in former and now[ln] > former[ln] + 1e-09]: reads.pop(ln, None)
  def after_add(self, args, reply):
    cart, own = self.barrow, vars(self)
    if cart.view_counts(reply) is not None: self.cart_shown = own.get('cart_ver', 0)
    known = own.setdefault('first_puts', set())
    new = [ln for ln, c in cart.lines(args) if (c is None or c > 0) and ln in cart.units and ln not in known]
    if not new or not cart.is_add(args) or self.reply_err(reply): return
    known.update(new)
    stale = set(self.stale_lines())
    new = [ln for ln in new if ln in stale]
    reader = self.stock_tool()
    if not new or not reader or self._muted(): return
    if self.steps_rem() < 4 or self.cable.spare() < CLOSING: return
    q = list(own.get('put_queue') or [])
    self.put_queue, self.put_tool = (q + [ln for ln in new if ln not in q])[:4], reader
    self._qj_jot('after_put_read', lines=list(self.put_queue), tool=reader)
  def key_fresh(self, reply):
    own = vars(self)
    if not own.get('stock_seen'): return
    reads, held = own.setdefault('line_seq', {}), self.barrow.units
    def mark(r):
      sid = _q4_sid(r)
      if sid in held and self.stock_state(r) is not None: reads[sid] = own.get('absorb_n', 0)
    walk_dicts(reply, mark)
  def cart_stale(self, grasped_sids):
    want = set(grasped_sids)
    gone = want & self.dead_lines()
    stale = [ln for ln in self.stale_lines() if ln in want and ln not in gone]
    if not stale or not self.stock_tool() or self.steps_rem() - len(stale) < 2: return gone
    _, sold = self.code_recheck(stale)
    return gone | set(sold) | want & self.dead_lines()
class BatchReads:
  def _prods(self): return self.needs.items()
  def _kwalk(self, attend, youngest_earliest=False):
    store = self.rec_db.items
    for obs in reversed(store) if youngest_earliest else store: walk_dicts(obs, attend)
  def batch_tool(self):
    if 'batch_spec' not in vars(self):
      try: found = self._find_batch()
      except Exception: found = None
      self.batch_spec = found
    return self.batch_spec
  def _find_batch(self):
    told, about = {self.role(n) for n in _ROLES} - {None}, {}
    for spec in self.tool_specs or ():
      body = spec.get('function') if isinstance(spec, dict) else None
      name = body.get('name') if isinstance(body, dict) else None
      if not isinstance(name, str): continue
      words = re.sub('([a-z])([A-Z])', '\\1 \\2', name).replace('_', ' ')
      about[name] = '%s %s' % (words, str(body.get('description') or ''))
    rx = '\\b(?:remov|delet|cancel|clear|discard|releas|refund|purchas|checkout|wish|basket|cart|bag)'
    told |= {t for t in list(self.argspec) if re.search(rx, about.get(t, t).lower())}
    for tool, params in self.argspec.items():
      if (found := _batch_arg(params)) is not None and tool not in told: return (tool,) + found
  # a list call naming known listings that brings none of them back means the store takes fewer ids than its
  # schema says: halve the size for the code and write it into this errand's copy of the schema for the driver
  def batch_learn(self, name, args, obs):
    arg = (_batch_arg(self.argspec.get(name)) or (None,))[0]
    ids = args.get(arg) if arg and isinstance(args, dict) else None
    if not isinstance(ids, list) or len(ids) < 2 or not all(i in self.rec_db.ids for i in ids): return
    hit = []
    walk_dicts(obs, lambda r: _q4_sid(r) and hit.append(1))
    if hit: return
    cap = (len(ids) + 1) // 2
    sch = self.argspec[name] = json.loads(json.dumps(self.argspec[name]))
    sch['properties'][arg]['maxItems'] = cap
    if (spec := self.batch_tool()) and spec[0] == name:
      self.batch_spec = (name, spec[1], min(int(spec[2]), cap)) if cap >= 2 else None
    self.log.append({'kind': 'batch_learned', 'tool': name, 'asked': len(ids), 'cap': cap})
  def _batch_shrink(self, asked):
    if not (spec := vars(self).get('batch_spec')) or asked <= 2: return False
    self.batch_spec = (spec[0], spec[1], max(2, min(int(spec[2]), asked - 1)))
    self.log.append({'kind': 'batch_shrunk', 'tool': spec[0], 'cap': self.batch_spec[2]})
    return True
  @staticmethod
  def batch_ok(found, ids):
    asked, hit = set(ids), set()
    def visit(k):
      maps = [s for s in k.values() if isinstance(s, dict)]
      short = [v for v in k.values() if isinstance(v, str) and 0 < len(v) <= 70]
      if any(_atoms(s) >= 2 for s in maps) if maps else len(short) >= 5:
        hit.update(v for v in k.values() if isinstance(v, str) and v in asked)
    try:
      for obs in found: walk_dicts(obs, visit)
    except Exception: return False
    return 0 < len(hit) >= len(asked) / 2
  @staticmethod
  def ids_back(found, ids):
    if not ids: return False
    text = json.dumps(found, default=str)
    return any(s in text for s in ids)
  def batch_denser(self, found, ids):
    asked, keys, maps = set(ids), set(), [False]
    def visit(k):
      if not any(isinstance(v, str) and v in asked for v in k.values()): return
      keys.update(map(str, k))
      maps[0] = maps[0] or any(isinstance(s, dict) and len(s) >= 2 for s in k.values())
    try:
      for obs in found: walk_dicts(obs, visit)
    except Exception: return True
    rk = set(vars(self).get('row_keys') or set())
    return maps[0] or not rk or bool(keys - rk)
  @staticmethod
  def title_of(row):
    opts = [row.get('title')] + [v for n, v in row.items() if re.search('title|heading|headline', str(n).lower())]
    return next((v for v in opts if isinstance(v, str)), '')
  def _hits(self):
    mass = getattr(self, 'hit_score', {})
    return sorted(getattr(self, 'all_hits', []), key=lambda r: -mass.get(r, 0))
  def _pricey(self, rowid, units, cap):
    return is_num(cap) and is_num(c := self.rec_db.price(rowid)) and c * units > cap
  def menu_rows(self, genres, utmost, skip, units=1, headed_merely=False):
    shop, cap = self.rec_db, self.needs.spend_cap
    heads, sold, known = {}, set(), []
    def visit(k):
      if (r := self.row_key(k)) not in shop.ids or self.shelf_for(k) not in genres: return
      if r in shop.read:
        known.append(set(_WORD3.findall(self.title_of(k).lower())))
        return
      if self.sold_out(k): sold.add(r)
      heads.setdefault(r, self.title_of(k))
    self._kwalk(visit)
    hits = self._hits()
    ordered = [r for r in hits if r in heads] + [r for r in heads if r not in hits]
    stash = [r for r in ordered if not (r in sold or r in skip or r in self.refused_lines)
        and not self._pricey(r, units, cap)]
    words = {r: set(_WORD3.findall(heads[r].lower())) for r in stash}
    scored, chosen = getattr(self, 'row_score', {}), []
    if (first := self._lead_row(genres, utmost, words)):
      chosen.append(first)
      known.append(words[first])
    elif headed_merely: return []
    def rank(r):
      mine = words[r]
      near = max((len(mine & o) / max(1, len(mine | o)) for o in known), default=0.0)
      return (0 if r in hits else 1 if scored.get(r) else 2, near, stash.index(r))
    while len(chosen) < utmost:
      if not (view := [r for r in stash if r not in chosen][:16]): break
      best = min(view, key=rank)
      chosen.append(best)
      known.append(words[best])
    return chosen
  @_guard(None)
  def _lead_row(self, genres, utmost, pouch):
    many = bool(self._prods())
    pq = next((n for n, g in enumerate(self.home_cats()) if g == genres), None) if many else None
    if many and pq is None or utmost <= 0: return None
    if not (sought := self._told(many, pq)) or self._shown(genres, sought): return None
    borne = [(len(h), w) for w in getattr(self, 'page_book', {}).values() if (h := self.walk_has(w['args'], pq))]
    for _, w in sorted(borne, key=lambda d: -d[0]):
      if (first := next((r for r in w.get('row_shelf') or {} if r in pouch), None)): return first
  def _told(self, many, pq):
    out = []
    for req in self.needs.props_left():
      label, _, v = req.partition(':')
      if many and self.prod_of(label) not in (pq, None): continue
      v = _MARK_RX.sub('', v)
      if not v.strip() or not confirm_val(v): continue
      if not re.search('as shown|same as|the page|sold by|redirect|refer', v.lower()): out.append(v)
    return out
  def _shown(self, genres, sought):
    facts = {}
    def visit(k):
      r = self.row_key(k)
      if r not in self.rec_db.read or self.shelf_for(k) not in genres: return
      if not (maps := [s for s in k.values() if isinstance(s, dict)]): return
      facts.setdefault(r, []).extend(str(v) for s in maps for v in s.values() if isinstance(v, (str, int, float)))
    self._kwalk(visit)
    return any(all(val_seen(o, lump) for o in sought) for lump in map(' | '.join, facts.values()))
  def _queue_reads(self, ids, purpose, say):
    self.open_queue, self.open_recheck, self.open_say, self.open_purpose = (ids, True, say, purpose)
  def _thick(self, genres): return sum(map(self.rec_db.dense_n, genres)) if genres else 4
  @_catch('pre_read_error', bool)
  def first_read(self):
    reader = self.batch_tool()
    stage = {0: 'first', 1: 'top_up'}.get(getattr(self, 'pre_rounds', 0), 'spent')
    if not reader or stage == 'spent' or len(self.questions_log) > 1: return False
    if self.steps_rem() <= (25 if stage == 'top_up' else 20): return False
    if self.cable.spare() < CLOSING + 60: return False
    groups, counts, per = self.home_cats(), self.needs.counts(), reader[2]
    at = lambda pq: counts[min(pq, len(counts) - 1)]
    if stage == 'top_up':
      self.pre_rounds = 2
      lineups, held = [], list(getattr(self, 'pre_picked', None) or [])
      for pq, genres in enumerate(groups):
        if (have := self._thick(genres)) >= 3: continue
        more = self.menu_rows(genres, 4 - have, held, at(pq))
        held = more + held
        lineups.append(more)
      if not (chosen := _deal(lineups, per)): return False
      say = 'A few more listings are read first, because earlier reads of some products showed only a handful of fields.'
      self._queue_reads(chosen, 'menu', say)
      return True
    bare, slim = [], []
    for pq, genres in enumerate(groups):
      have = self._thick(genres)
      if not have: bare += self.menu_rows(genres, 2 if self._prods() else 4, bare, at(pq))
      elif have < 4: slim.append((pq, 4 - have))
    chosen = list(bare)
    for pq, gap in slim: chosen += self.menu_rows(groups[pq], gap, chosen, at(pq))
    del chosen[per * (2 if len(bare) > per and self.steps_rem() > 25 else 1):]
    if not chosen: return False
    self.pre_rounds, self.pre_picked = 1, list(chosen)
    say = 'A few listings of each requested item are read first; their field keys guide the questions.'
    self._queue_reads(chosen, 'menu', say)
    return True
  @_catch('fit_reads_error', bool)
  def fit_reads(self):
    own = vars(self)
    if own.get('fit_read') or self.questions_log or self.msgs_sent: return False
    if self.steps_rem() <= 20 or self.cable.spare() < CLOSING + 60: return False
    many, counts, chosen = bool(self._prods()), self.needs.counts(), []
    for pq, genres in enumerate(self.home_cats()):
      sought = self._told(many, pq)
      if not (genres and sought) or self._shown(genres, sought): continue
      unit = counts[min(pq, len(counts) - 1)]
      chosen += self.menu_rows(genres, 2, chosen, unit, headed_merely=True)
    if not chosen: return False
    own['fit_read'] = True
    say = 'None of the listings read so far shows the stated values, so the rows that a search for those values turned up are read next; their field keys guide the questions.'
    self._queue_reads(chosen, 'menu', say)
    self.log.append({'kind': 'fit_reads', 'ids': chosen})
    return True
  @_catch('lacking_error', list)
  def empty_rows(self):
    if not (products := self._prods()): return []
    groups = self.home_cats()
    live = self._live_at(str(self.needs.rev_no)) - self.refused_lines - self.dead_lines()
    found = {self.known_shelf(r) for r in live | set(getattr(self, 'audit_fits', None) or [])} - {''}
    if not (short := [pq for pq, g in enumerate(groups) if g and g.isdisjoint(found)]): return []
    reader = self.batch_tool()
    shop, cap = self.rec_db, self.needs.spend_cap
    track, sold, shelf_of, owner = [], set(), {}, {}
    def visit(k):
      r, label = self.row_key(k), self.shelf_for(k)
      if r not in shop.ids or not label: return
      track.append(r)
      shelf_of.setdefault(r, label)
      if self.sold_out(k): sold.add(r)
    self._kwalk(visit, youngest_earliest=True)
    for pq in short:
      for label in groups[pq]: owner.setdefault(label, pq)
    pools = {pq: [] for pq in short}
    for r in dict.fromkeys([r for r in self._hits() if r in shelf_of] + track):
      if r in shop.read or r in self.refused_lines or r in sold: continue
      pq = owner.get(shelf_of[r])
      if pq is not None and not self._pricey(r, products[pq]['quantity'], cap): pools[pq].append(r)
    return _deal([pools[pq] for pq in short], reader[2] if reader else 3)
  def text_arg(self, tool):
    params = self.argspec.get(tool or '') or {}
    props, req = params.get('properties') or {}, list(params.get('required') or [])
    texts = [n for n, e in props.items() if isinstance(e, dict)
        and (e.get('type') == 'string' or 'string' in (e.get('type') or [])) and 'enum' not in e]
    pick = next((n for n in req if n in texts), None) or (texts[0] if len(texts) == 1 else None)
    return pick if pick and all(n == pick for n in req) else None
  def msg_call(self, body, summon_sid):
    tool = self.role('message')
    if not (field := tool and self.text_arg(tool)): return None
    args, problems = ArgCheck.vet({field: body}, self.argspec[tool])
    if problems: return None
    return {'id': summon_sid, 'type': 'function', 'function': {'name': tool, 'arguments': json.dumps(args)}}
  def prod_of(self, text):
    if not (products := self._prods()): return None
    sorts = [set(kind_toks(r['product'])) for r in products]
    span = range(len(sorts))
    rivals = [set().union(*[sorts[o] for o in span if o != pq]) for pq in span]
    def pick(words):
      g = [kind_score(sorts[pq], words, rivals[pq]) for pq in span]
      best = max(g)
      lead = [pq for pq in span if g[pq] == best]
      if len(lead) > 1: lead = [pq for pq in lead if all(kind_in(w, words) for w in sorts[pq])]
      return lead[0] if best > 0 and len(lead) == 1 else None
    whole = str(text)
    for m in re.findall('\\(([^)]*)\\)', whole)[:1] + [whole]:
      if (found := pick(set(kind_toks(m)))) is not None: return found
  def plan_prod(self, entry):
    own = self.prod_of(str(entry.get('property') or ''))
    return self.prod_of(str(entry.get('question') or '')) if own is None else own
  @staticmethod
  def _which_q(field, product): return f'Which {field} would you like for the {product}?'
  def _ask_count(self, count):
    answered, notified = [0] * count, [0] * count
    for e in self.questions_log:
      head = e.partition(':')[0]
      if (who := self.prod_of(head)) is None and (m := re.search('asked "(.*?)" -> ', e)) is not None:
        who = self.prod_of(m.group(1))
      if who is None: continue
      notified[who] += 1
      if '[not answered]' not in head: answered[who] += 1
    return answered, notified
  def share_first(self, plan, ok=None):
    if not (products := self._prods()): return plan
    count = len(products)
    answered, notified = self._ask_count(count)
    if not any(answered) and plan and self.plan_prod(plan[0]) is None: return plan
    shown = [0] * count
    for need in self.needs.uncovered:
      if (who := self.prod_of(str(need).partition(':')[0])) is not None: shown[who] += 1
    try:
      keyed, unkeyed = set(), [0] * count
      for a in self.open_hints():
        head = str(a).partition(':')[0]
        if (who := self.prod_of(head)) is None: continue
        if self.plain_hint(who, head.strip(), hinted=True): unkeyed[who] += 1
        else: keyed.add(who)
      hinted = {pq for pq in range(count) if pq in keyed or unkeyed[pq] > answered[pq]}
    except Exception: hinted = set()
    quiet = [pq for pq in range(count) if not answered[pq]]
    sub = {pq for pq in range(count) if notified[pq] > answered[pq] and self.alt_untried(pq)}
    keys = [(1 if shown[pq] and any(o != pq for o in quiet) else 0, answered[pq] + shown[pq], int(pq not in hinted),
        0 if pq in sub else notified[pq] - answered[pq], shown[pq]) for pq in range(count)]
    if self._jump(plan, keys, notified): return plan
    st = {'plan': plan, 'products': products, 'answered': answered, 'hinted': hinted, 'ok': ok}
    steps = (self._due_planned, self._due_second, self._due_prev, self._due_pick)
    for grade in sorted(set(keys)):
      st['due'] = [pq for pq, k in enumerate(keys) if k == grade]
      for step in steps:
        if (out := step(st)) is not None: return out
    return plan
  @_guard(False)
  def _jump(self, plan, keys, notified):
    if (head := self.plan_prod(plan[0]) if plan else None) is None: return False
    low = min(keys)
    due = {pq for pq, k in enumerate(keys) if k == low}
    if head in due or keys[head][:3] != low[:3] or notified[head] > min(notified) + 1: return False
    if not any(self.plan_prod(e) in due for e in plan[1:]): return False
    silent = sum(1 for pq, sent in enumerate(notified) if not sent and pq != head)
    if silent > self.ask_room() - self.msgs_sent - 1: return False
    self.log.append({'kind': 'lead_jump', 'property': str(plan[0].get('property'))})
    return True
  def _due_prev(self, s):
    ok = s['ok']
    for e in vars(self).get('plan_spares') or []:
      w, lab = self.plan_prod(e), e.get('property')
      if w in s['due'] and (ok is None or ok(lab)) and not self.asked_for(w, lab): return [e] + s['plan']
  def _due_planned(self, s):
    p = s['plan']
    i = next((n for n, e in enumerate(p) if self.plan_prod(e) in s['due']), None)
    return None if i is None else [p[i]] + p[:i] + p[i + 1:]
  def _due_second(self, s):
    for e in s['plan']:
      sp = str(e.get('second') or '').strip()
      w = self.prod_of(sp) if sp else None
      if w not in s['due']: continue
      tag = re.search('\\(([^)]+)\\)', sp)
      prod = tag.group(1).strip() if tag else s['products'][w]['product']
      return [{'property': sp, 'question': self._which_q(sp.partition('(')[0].strip(), prod)}] + s['plan']
  def _due_pick(self, s):
    hinted, ok = s['hinted'], s['ok']
    for pq in sorted(s['due'], key=lambda c: (c not in hinted, c)):
      try: got = self.share_pick(pq, s['answered'][pq] == 0, ok)
      except Exception: got = None
      if got and ok is not None and (not ok(lab := got.get('property')) or self.asked_for(pq, lab)): got = None
      if got: return [got] + s['plan']
      if getattr(self, 'quota_wait', None): return s['plan']
  def share_pick(self, pq, fallback=True, ok=None):
    prod = self._prods()[pq]['product']
    groups, rep = self.home_cats(), self.rec_db
    genres = groups[pq] if pq < len(groups) else set()
    shelf = next((v for v in rep.shelf_read() if v.strip().lower() in genres), None)
    if shelf is None and genres and getattr(self, 'quota_reads', {}).get(pq):
      shelf = next((v for v in sorted(genres) if rep.prop_list(v)[0]), None)
    if shelf is None:
      reader = self.batch_tool()
      if not genres: return None
      todo = self.menu_rows(genres, min(4, reader[2]) if reader else 2, [],
                       self._prods()[pq]['quantity'])
      if (todo and (self.role('details') or reader) and not getattr(self, 'quota_reads', {}).get(pq)
          and self.steps_rem() > 12):
        vars(self).setdefault('quota_reads', {})[pq] = True
        self._queue_reads(todo, 'quota',
                    f'No listing of the {prod} has been read yet; opening a few of them now lets its question name a real key.')
        self.quota_wait = True
      return None
    menu = self.menu_trim(rep.prop_list(shelf)[0], rep.dense_n(shelf))
    def fresh(lab): return ok is None or bool(ok(lab)) and not self.asked_for(pq, lab)
    plan = self.product_plan(pq, menu)
    if plan and fresh(plan.get('property')): return plan
    for f in self._fields(pq, menu, fallback):
      if fresh(f'{f} ({prod})'): return {'property': f'{f} ({prod})', 'question': self._which_q(f, prod)}
    return None
  def _fields(self, pq, menu, fallback):
    for a in self._lhints(pq):
      f = a.partition(':')[0].partition('(')[0].strip()
      if f and not junk_key(f) and not self.plain_hint(pq, f, hinted=True): yield f
    for exact in (False, True) if fallback else ():
      for row in menu.splitlines():
        f = row.strip('- ').partition(' (in ')[0].strip()
        if f and self.open_prop(pq, f, row, per_exemplars=exact): yield f
  def product_plan(self, pq, menu):
    try:
      used = getattr(self, 'product_plans', 0)
      if not menu or used >= 1 or self.cable.spare() < CLOSING + 40: return None
      self.product_plans = used + 1
      prod = self._prods()[pq]['product']
      up, room = prod.upper(), max(1, self.ask_room() - self.msgs_sent)
      segs = [ASK_PLAN.replace('QUESTION_COUNT', str(min(3, room))),
          f"FOCUS: {up}. Plan for this item only: it is the one requested item that has not had a question yet. Draw its properties only from the {prod} key list at the end, and put first the key that the story's concrete details are about.",
          'SHOPPER REQUEST:\n' + self.request, 'REQUIREMENTS SO FAR:\n' + self.needs.body_text()]
      if self.questions_log:
        segs.append("QUESTIONS SENT SO FAR, EACH WITH THE SHOPPER'S REPLY:\n" + '\n'.join(self.questions_log))
      if hints := self._lhints(pq):
        segs.append('STORY HINTS\nThese are guesses, not requirements: each line names a property, then the story detail pointing at it; the most likely line is first.\n' + '\n'.join(f'- {self.hint_key(h)}' for h in hints[:6]))
      text = '\n\n'.join(segs) + self.plan_notes(pq)
      tail = f"\n\nAVAILABLE KEYS - {up}\nEach row shows a key name, how many of this item's listings read so far carry it, and sample values:\n" + menu
      reply = self.cable.consult(text + tail, 1500, PLANNER[:1])
      self.log.append({'kind': 'product_plan', 'product': prod, 'result': reply})
      qs = (reply or {}).get('questions') or []
      return next((p for e in qs if isinstance(e, dict) and (p := self._pick(pq, prod, e))), None)
    except Exception: return None
  def _pick(self, pq, prod, entry):
    lead, sp = (str(entry.get(k) or '').partition('(')[0].strip() for k in ('property', 'second'))
    if _NEVER.search(sp.lower()): sp = ''
    q = str(entry.get('question') or '').strip()
    if self.plain_hint(pq, lead): lead = ''
    if sp and self.plain_hint(pq, sp): sp = ''
    if lead and self.plan_open(pq, lead):
      top = bool(q) and bool(kind_words(prod) & kind_words(q))
      alt = top and sp and not exact_prop(lead, sp) and self.plan_open(pq, sp)
      return {'property': f'{lead} ({prod})', 'second': f'{sp} ({prod})' if alt else '',
          'question': q if top else self._which_q(lead, prod)}
    if sp and self.plan_open(pq, sp):
      return {'property': f'{sp} ({prod})', 'question': self._which_q(sp, prod)}
  def open_prop(self, whoever, field, repertoire_itemrow='', per_exemplars=False):
    try:
      name = str(field).partition('(')[0].strip()
      if not name or junk_key(name): return False
      several = bool(self._prods())
      def ours(t, tagged=False):
        if whoever is None or not several: return True
        w = self.prod_of(t)
        return not tagged if w is None else w == whoever
      def same(t): return ours(t) and self.same_field(whoever, t, name)
      heads = (e.partition(':')[0].strip('- ') for e in self.questions_log)
      if any(same(sp) for h in heads if '[not answered]' not in h for sp in h.split(';')): return False
      egvals = repertoire_itemrow.partition('):')[2].lower()
      for r in map(str, self.needs.props_left()):
        if ours(r.partition(':')[0]) and self._settled(whoever, name, r, egvals, per_exemplars):
          return False
      idle = map(str, self.needs.indifferent)
      return not any(ours(p, True) and self.same_field(whoever, p, name) for p in idle)
    except Exception: return True
  def _settled(self, whoever, name, req, egvals, per_exemplars):
    head, _, said = req.partition(':')
    if exact_prop(head, name): return bool(self.rule_ok(req, egvals if per_exemplars else ''))
    if same_prop(head, name):
      pool = egvals or ' | '.join(self.fld_vals(whoever, name))
      return bool(confirm_val(said, head) and val_seen(said, pool))
    ws = [w for w in re.findall('[a-z]{3,}', said.lower()) if w not in STOP_WORDS]
    if not egvals or not ws or re.search('\\d', said) or not confirm_val(said, head): return False
    return all(re.search('\\b' + re.escape(w), egvals) for w in ws)
  def plan_notes(self, whoever):
    try:
      several = bool(self._prods())
      heads = (e.partition(':')[0].strip('- ') for e in self.questions_log)
      asked = {x.strip().lower() for h in heads for x in h.split(';')}
      free = whoever is None or not several
      keep = []
      for p in getattr(self, 'open_pair', None) or []:
        w = self.prod_of(p) if several else None
        if (free or w in (whoever, None)) and p.lower() not in asked and self.plan_open(w, p):
          keep.append(p)
      drafts = getattr(self, 'wanted_asks', None) or []
      drafts = [d for d in drafts if free or self.prod_of(d) in (whoever, None)]
      a = "\n\nSTILL OPEN AFTER A PAIRED QUESTION (each was the plan's next pick: the second half of a planned pair that the reply left unanswered, because only the first half was sent or only it got an answer):\n"
      b = "\n\nUNSENT DRAFTS (written by the acting assistant after it opened listing records). A draft whose property is still open and shows up among the keys below takes the first slot, worded with that key's own spelling:\n"
      return ''.join(t + '\n'.join(f'- {i}' for i in v[-3:]) for t, v in ((a, keep), (b, drafts)) if v)
    except Exception: return ''

_AN, _NEVER = re.compile('[a-z0-9]+'), re.compile('none|^-$|n/a')
_LK = '\\b(?:' + BOUND_WORDS + ')\\b'
_BOUND = re.compile(_LK + '|\\d\\s*(?:-|to)\\s*\\d|[+<>]|\\b(?:about|around|approx\\w*|roughly)\\b')
_FIG = '(\\d+|one|two|three|four|five|six|seven|eight|nine|ten)'
_INF, _SMALL = float('inf'), ('one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten')
_LIMITS = (('message_budget', 'messages?|questions?|notes?|repl(?:y|ies)|texts?', 0),
      ('step_budget', 'steps?|moves?|actions?|(?:tool )?calls?|turns?', FIRST_ORDER_BY + 3))
_AVAIL = '\\bin[- ]stock\\b|\\bon hand\\b|\\bon the shelf\\b|\\bsold[- ]out\\b|\\bback[- ]?order|\\bout of stock\\b|\\bavailable now\\b'
_LOWEST = '\\bcheapest\\b|\\blowest[- ](?:price|priced|cost)\\b|\\bleast expensive\\b|\\blowest\\b[^.]{0,20}\\bprice\\b|\\bsmallest (?:total|price)\\b'
_N_CUE = '\\s*(?:x\\b|\\u00d7|%|:\\s*\\d|-\\s*\\d|/\\s*\\d|(?:[a-z-]+\\s+){0,2}?(?:messages?|questions?|notes?|repl\\w*|texts?|steps?|moves?|actions?|(?:tool )?calls?|turns?|units?|items?|pieces?|packs?|pairs?|lots?|sets?|of\\b|times?)\\b)'
_SPEND = "\\b(?:spend\\w*|budget\\w*|afford\\w*|pay\\w*|price[ds]?|costs?|cash|money)\\b(?:\\W+[\\w']+){0,4}\\W*$"
_CUE_PRE = "\\b(?:under|below|no more than|at most|not more than|up to|within|max\\w*|less than|cheaper than|not (?:more|over|above)|limit\\w*|reach\\w*|go to|to|at|cap\\w*|past|exceed\\w*|beyond|go over)\\W+(?:[\\w']+\\W+)?$"
_CUE_POST = '\\s*(?:[a-z.()]+\\s*){0,3}?(?:or (?:lower|less|under|below|cheaper)|at (?:the (?:very )?)?most|max\\w*|tops\\b|,?\\s*(?:and )?not (?:one|a) \\w+ (?:over|more|above))'
_CAP_PRE = "\\b(?:no more than|at most|not more than|up to|only|max\\w*|use|using|take|send|allow\\w*|receive|get|total of|limit\\w*|inside|within|under|fewer than|less than|you've|you have|have|keep\\w*|stick to|go past|past|exceed\\w*|beyond)\\W+(?:[\\w']+\\W+){0,3}$"
_CAP_POST = "\\W*(?:[\\w']+\\W+){0,3}?(?:in total|total|no more|at most|max\\w*|tops|or fewer|or less|allowed|limit)\\b"
_NUMWORDS = {'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5, 'six': 6, 'seven': 7, 'eight': 8, 'nine': 9,
       'ten': 10, 'twelve': 12, 'fifteen': 15, 'twenty': 20}
_NEED_VB = '(?:is|are|has to be|have to be|must be|should be|needs to be|need to be|can be)'
_END_REQ = '(?=\\s*(?:[,.;:!?()]|$)|\\s+(?:so|since|as|because|for|to|which|that|nothing|otherwise|but|when|while|if|then|though|please|and (?:the|its|their|a|an|one|two|three|\\d))\\b)'
_NEED_PRE = '(?:\\b(?:where|whose)\\s+(?:the\\s+|its\\s+|their\\s+)?|(?:\\b(?:and|with|but)\\s+|[,;:]\\s*|^|[.!?]\\s+)(?:the|its|their)\\s+)'
_NEED_IN = "([a-z][a-z'&/ -]{1,32}?)\\s+"
_NEED_VAL = '\\s+(?:either\\s+|just\\s+)?([^,.;:!?()]{1,60}?)'
_NO_PROP = '\\b(?:i|you|we|it|he|she|they|my|your|our|there|this|that|which|what|who|me|us|them|one|ones|price|budget|cost|total|order|stock)\\b'
_NOT_VAL = '(?:a|an|the|my|your|our|his|her|their|me|us|for|to|in|on|at|of|see|go|get|be|have|only|too|very|so|not|no|what|how|it|all|mostly|always|never|just|really|still|also|from|about|like|one|ones)\\b'
_ID_CUT = '[\\s.,;:!?()\\[\\]{}<>\\"\'`]+'
_TALK = '\\b(?:send\\w*|repl\\w*|notes?|tell\\w*|messag\\w*|chat\\w*|writ\\w*|talk\\w*)\\b'
_TIPS = (('search', '\\b(?:search\\w*|find\\w*|look\\w*|locat\\w*|hunt\\w*|brows\\w*|quer\\w*)\\b'),
    ('message', '\\b(?:send\\w*|repl\\w*|notes?|tell\\w*|messag\\w*|chat\\w*|writ\\w*|talk\\w*)\\b'),
    ('details', '\\b(?:details?|records?|spec\\w*|propert\\w*|attribut\\w*|facts?|full)\\b'),
    ('view', '\\b(?:offer|price|stock|availab\\w*)\\b'))
def _cell(item, slot): return str(item.get(slot) or '')
def _flat(text): return ' '.join(_AN.findall(str(text).lower()))
def _jot(name, what): return f' [{name}: {what}]'
def _which(theme, product): return f'Which {theme.partition("(")[0].strip()} would you like for the {product}?'
def _headpt(label): return label.partition(';')[0].strip()
def _q5_cut(text, spec): return text[:spec.get('maxLength', len(text))]
def _addnew(held, names):
  for n in names:
    if all(x.lower() != n.lower() for x in held): held.append(n)
def _spent(name, heard, mine):
  hub = {w for w in prop_words(name) if len(w) > 2 and w not in STOP_WORDS}
  return bool(hub) and hub <= heard and not hub <= mine
def _quoted(line, flat, mine):
  _, colon, tail = line.partition('):')
  if not colon: return False
  skip, pad = mine | STOP_WORDS, f' {flat} '
  for t in (_AN.findall(x.lower()) for x in tail.split('|')):
    if sum(len(x) for x in t) >= 3 and not set(t) <= skip and ' ' + ' '.join(t) + ' ' in pad: return True
  return False
def _digest(text, theme, menu):
  try:
    m = re.search('(?::|,|\\bsuch as\\b|\\bfor example\\b|\\be\\.g\\.)(.+)$', text, re.I)
    if not m: return False
    keys = (v.strip().lstrip('-').strip().split(' (in ')[0] for v in str(menu or '').splitlines() if ' (in ' in v)
    shown = [x for x in re.split(',|\\bor\\b', m.group(1)) if x.strip(' ?.!')]
    return any(exact_prop(k, theme) for k in keys) and len(shown) >= 3
  except Exception: return False
def _pad(spec, phrase, tip):
  if spec.get('enum'):
    opts = spec['enum']
    for v in opts if tip else ():
      if isinstance(v, str) and re.search(tip, v.lower()): return [v]
    for v in opts if tip == NO_BUY_WORDS else ():
      if isinstance(v, str) and not re.search(ORDER_WORDS, v.lower()): return [v]
    return [opts[0]]
  kind = spec.get('type')
  if kind == 'string' or 'string' in (kind or []): return [_q5_cut(phrase, spec)]
  if kind == 'boolean': return [True]
  return [max(1, spec.get('minimum', 1))] if kind in ('integer', 'number') else []
def _grade(st, m, prem):
  after, sig = st[m.end():], m.group(1)
  if re.match(_N_CUE, after) or not sig and re.search('[\\d.,:/-]\\s*$|\\bx\\s*$', st[:m.start()]): return None
  before = st[max(0, m.start() - 60):m.start()]
  return (prem + (3 if sig or re.search(_SPEND, before) else 0) + (2 if re.search(_CUE_PRE, before) else 0)
      + (2 if re.match(_CUE_POST, after) else 0))
def _gauge(after, sig):
  h = re.match('\\s*((?:[a-z][a-z.]*\\s*){0,2})', after)
  ws = (r.strip('.') for r in h.group(1).split()) if h else ()
  ws = [w for w in ws if w not in STOP_WORDS and len(w) >= 2 and not re.search('\\d', w)]
  return ' '.join(ws) or (sig or '').strip() or None
def _cap_of(low, nominals):
  pat = '\\b(\\d+|' + '|'.join(_NUMWORDS) + ')\\s+(?:[a-z-]+\\s+){0,2}?(?:' + nominals + ')\\b'
  for m in re.finditer(pat, low):
    s, e = m.span()
    if re.search(_CAP_PRE, low[max(0, s - 60):s]) or re.match(_CAP_POST, low[e:e + 40]):
      return int(m.group(1)) if m.group(1).isdigit() else _NUMWORDS[m.group(1)]
  return None
def _num(token): return int(token) if token.isdigit() else _SMALL.index(token) + 1
def _count_of(low):
  m = re.search('\\b' + _FIG + '\\s+(?:units?|of (?:them|those|these)|copies)\\b', low)
  n = _num(m.group(1)) if m else 1
  return n if 0 < n < 100 else 1
def _keep_req(ext, m):
  prop, val = (x.strip() for x in m.groups())
  tail = ext[m.end():m.end() + 60]
  if is_ref_text(val) or is_ref_text(tail) and re.search('\\b(?:page|listing|shown|sold by)\\b', tail):
    return False
  if not val or any(len(t.split()) > 4 for t in (val, prop)) or re.search(_NO_PROP, prop): return False
  return not re.match(_NOT_VAL, val)
def _needs(ext):
  pat = _NEED_PRE + _NEED_IN + _NEED_VB + _NEED_VAL + _END_REQ
  return [tuple(x.strip() for x in m.groups()) for m in re.finditer(pat, ext) if _keep_req(ext, m)]
def _terms(text, regex='[a-z0-9]+'): return [w for w in re.findall(regex, text.lower()) if w not in STOP_WORDS]
def _uniq(words): return ' '.join(dict.fromkeys(words))
def _enum(spec): return spec.get('enum') if isinstance(spec, dict) else None
def _spaced(text): return re.sub('([a-z])([A-Z])', '\\1 \\2', text)
def _free(args, kind):
  ok = {'text': ('string',), 'number': ('integer', 'number')}[kind]
  return [n for n, s in args.items() if isinstance(s, dict) and not s.get('enum') and s.get('type') in ok]
def _qty_at(text, trms, own):
  n = len(own)
  lead = '\\b' + _FIG + "\\s*(?:x\\s+|\\u00d7\\s*)?(?:(?:of|[a-z'&-]{3,})\\s+){0,4}$"
  for at in range(len(trms) - n + 1):
    win = trms[at:at + n]
    s, e = win[0][1], win[-1][2]
    if [w for w, _, _ in win] == own and len(re.findall('[a-z0-9]+', text[s:e])) <= n * 2:
      if f := re.search(lead, text[max(0, s - 56):s]): return (s, e, _num(f.group(1)))
  return None
def _fits(role, c):
  t = c['text']
  human = re.search('\\b(?:shoppers?|customers?|buyers?|clients?|users?)\\b', t)
  if role == 'search': return not (human and re.search(_TALK, t)) and (bool(c['free']) or c['choice'])
  if role == 'message' and not (human or re.search('\\b(?:messag\\w*|chat\\w*)\\b', t)): return False
  return len(c['need']) == 1 and c['need'][0] in c['free']
def _wb(word): return '\\b' + re.escape(word.lower()) + '\\b'
def _style(lf, prm, last):
  opts = lf['lists'].get(lf['mode'] or '', [])
  naming = [s for s in lf['clauses'] if re.search(_wb(prm), s)]
  return next((o for o in opts if any(re.search(_wb(str(o)), s) for s in naming)), opts[-1 if last else 0])
def _fill(p, lf, words):
  for n in lf['need']:
    if n in p: continue
    s = lf['args'][n] if isinstance(lf['args'][n], dict) else {}
    if s.get('enum'): p[n] = s['enum'][0]
    elif s.get('type') in ('integer', 'number'): p[n] = s['default'] if 'default' in s else s.get('minimum', 1)
    elif s.get('type') == 'string': p[n] = words
def _is_price(name):
  if not re.search('price|cost|amount', name): return False
  return not re.search('count|qty|quant|total|units|line|due|payable|sum', name)
def _cells(fields, prop): return [text for name, text in fields.items() if same_prop(name, prop)]
def _sub_in(alt, pool):
  nums = re.findall('\\d+(?:\\.\\d+)?', alt)
  stems = [w[:-1] if len(w) > 4 else w for w in re.findall('[a-z]{2,}', alt) if w not in STOP_WORDS]
  if not stems and not nums or not all(re.search('\\b' + re.escape(w), pool) for w in stems): return False
  return all(re.search('(?<![\\d.])' + re.escape(f) + '(?![\\d.])', pool) for f in nums)
def _seen_in(value, text):
  pool = str(text).lower() if text else ''
  if not pool.strip(): return False
  return any(_sub_in(a, pool) for a in re.split('\\s*(?:\\bor\\b|/|,|;)\\s*', re.sub(_LK, ' ', str(value).lower())))
def _in_title(row, value): return _seen_in(value, row.get('title') or '')
def _prec(row, prop, value):
  def joined(t): return ' '.join(txt_words(t))
  alts = {joined(a) for a in re.split('\\s*(?:\\bor\\b|/)\\s*', value.lower()) if txt_words(a)}
  ranks = []
  for name, t in row['fields'].items():
    if same_prop(name, prop) and _seen_in(value, t):
      tier = 0 if joined(t) in alts else 1 if any(joined(x) in alts for x in text_parts(t)) else 3
      ranks.append((int(prop_key(name) != prop_key(prop)), tier))
  return min(ranks) if ranks else (2, 2) if _in_title(row, value) else (2, 3)
def _apart(fields, apart):
  def clear(p, bad):
    return any(same_prop(n, p) and str(t).strip() and all(not val_in_txt(v, t) for v in bad)
         for n, t in fields.items())
  return all(clear(p, bad) for p, bad in apart)
def _vetted(calls, schema):
  out = []
  for p in calls:
    r = ArgCheck.vet(p, schema)
    if r[1]: return None
    out.append(r[0])
  return out

class QuestionFlow:
  _smm_FIGURE, _smm_BOUND = _FIG, _BOUND
  def _ql_w(self, name): return self.prod_of(name) if self.needs.items() else None
  def _lhints(self, *a):
    try: return self.open_hints(*a)
    except Exception: return []
  def _lhome(self):
    try: return sorted(self.home_cats()[0])
    except Exception: return []
  def need_ask(self, furnished):
    try:
      if isinstance(furnished, dict): f = furnished
      else: f = as_json(furnished) if isinstance(furnished, str) and furnished.strip() else {}
      want = ' '.join(p for p in (v.strip() for v in (f or {}).values() if isinstance(v, str)) if p)[:300]
      if want:
        kept = [w for w in vars(self).get('wanted_asks') or [] if w.lower() != want.lower()]
        self.wanted_asks = (kept + [want])[-3:]
    except Exception: pass
  def drop_want(self, label):
    try:
      lemma = self.lemma
      def hit(want, prop):
        req = prop_words(prop)
        said = {lemma(w) for w in re.findall('[a-z]+', want.lower())}
        return bool(req) and all(w in said or lemma(w) in said for w in req)
      parts = re.split(';|\\band\\b', str(label).replace('[not answered]', ''))
      props = [re.split('\\s*\\(', h)[0] for h in parts]
      self.wanted_asks = [w for w in vars(self).get('wanted_asks') or [] if not any(hit(w, p) for p in props)]
    except Exception: pass
  def name_pair(self, question, theme, worksheet, counterpart=''):
    try:
      flat = _flat(question)
      words = set(flat.split())
      heard = words | {w[:-1] for w in words if len(w) > 3 and w[-1] == 's' and w[-2] != 's'}
      names = [_cell(r, 'product') for r in self.needs.items()] or [str(self.needs.parsed.get('product') or '')]
      mine = set(prop_words(theme)) | {'which', 'would', 'you', 'like'}
      for w in _AN.findall(' '.join(names).lower()):
        mine.add(w)
        if len(w) > 3 and w[-1] == 's': mine.add(w[:-1])
      if counterpart and not exact_prop(counterpart, theme) and _spent(counterpart, heard, mine):
        return True
      for line in str(worksheet or '').splitlines():
        k = line.strip().lstrip('-').strip().partition(' (in ')[0].strip()
        if not k or k.endswith(':') or exact_prop(k, theme) or junk_key(k): continue
        if _spent(k, heard, mine) or _quoted(line, flat, mine): return True
      return False
    except Exception: return False
  def queue_second(self, theme): _addnew(vars(self).setdefault('pending_seconds', []), [theme] if theme else [])
  def _ql_tag(self, label, fallback=None):
    if m := re.search('\\(([^)]+)\\)', label): return m.group(1).strip()
    if fallback is not None: return fallback
    rows = self.needs.items()
    if len(rows) > 1:
      bare = label.partition('(')[0].strip()
      own = [k for k in range(len(rows)) if bare and self.prop_line(k, bare)]
      if own: return str(rows[own[0]].get('product') or 'item')
    return str(self.needs.parsed.get('product') or 'item')
  def _lplain(self, label): return _which(label, self._ql_tag(label))
  def _minus(self, name): return [q for q in vars(self).get('pending_seconds') or [] if q.lower() != name.lower()]
  def _tofront(self, name): self.pending_seconds = [name, *self._minus(name)]
  def _hold(self, names): _addnew(vars(self).setdefault('open_pair', []), names)
  def _spent_q(self, name): vars(self).setdefault('spent_asks', set()).add(name.lower())
  def _free_n(self): return len(self.needs.indifferent) + vars(self.needs).get('pinned_heard', 0)
  def _cited(self, w, name):
    txt = ' '.join(self.needs.uncovered[w['found0']:]).lower()
    return any(x in txt for x in re.findall('[a-z]{4,}', name.partition('(')[0].lower()))
  def _idle(self): self.asking = False
  def pose_planned(self, urge=False):
    w, step = {'urge': urge}, 'gate'
    while isinstance(step, str): step = getattr(self, f'_sml_ask_{step}')(w)
    return bool(step)
  def _sml_ask_gate(self, w):
    tool, w['ceiling'] = self.role('message'), self.ask_room()
    if not tool or self.msgs_sent >= w['ceiling']: return self._idle()
    rep = self.rec_db
    menu, reads, shelf = rep.prop_list()[0], rep.dense_n(), self.own_shelf()
    if shelf: menu, reads = rep.prop_list(shelf)[0], rep.dense_n(shelf)
    elif not self.needs.items():
      own = self._lhome()
      if [s for s in rep.shelf_read() if s.strip().lower() not in own] and own:
        menu = '\n'.join(m for m in (rep.prop_list(s)[0] for s in own) if m)
        reads = sum(rep.dense_n(s) for s in own)
    w['menu'], w['sets'] = menu, reads
    first = bool(self.needs.parsed.get('ask_first')) and not self.questions_log
    w['pressing'] = first and (reads >= 1 or self.cable.ticks >= ASK_AFTER_STEPS)
    return 'timing'
  def ask_sets(self): return ASK_AFTER_SETS + (0 if self.needs.items() else STANDALONE_ENSEMBLES)
  def _tight(self, w, n):
    return self.steps_rem() <= self.order_due() + 2 * max(0, w['ceiling'] - self.msgs_sent) + n
  def _sml_ask_timing(self, w):
    sets, urge, hb = w['sets'], w['urge'], self.cable.ticks
    if not (sets >= self.ask_sets() or w['pressing'] or urge):
      if hb < ASK_AFTER_STEPS or hb < ASK_AFTER_STEPS + ASK_GRACE_STEPS and self.steps_rem() > 20:
        return None
    if self.steps_rem() <= max(10, self.order_due()) or self.cable.spare() < CLOSING + 40:
      return self._idle()
    rows = self.needs.items()
    if not urge and rows and len(rows) <= FEW_MAX:
      _, sent = self._ask_count(len(rows))
      if any(sent) and not self._tight(w, 12):
        if any(not sent[k] and self.prod_reads(k) < 1 for k in range(len(rows))): return None
    paused, last = vars(self).get('plan_wait'), vars(self).get('asked_at_sets')
    if paused is not None and sets <= paused and not urge: return None
    if last is not None and sets <= last and not (urge or w['pressing'] or self._tight(w, 14)):
      if self.cable.ticks - vars(self).get('asked_at_tick', 0) < 4 and not self.needs.items(): return None
    return 'reads'
  def _sml_ask_reads(self, w):
    if self.first_read() or self.fit_reads(): return True
    fresh = not (self.questions_log or self.msgs_sent or w['pressing'] or w['urge'])
    return True if fresh and (self.own_survey() or self.first_reads()) else 'brief'
  def own_survey(self):
    state, rep = vars(self), self.rec_db
    if state.get('own_surveyed') or self.needs.items() or not self.role('search') or not state.get('refs'):
      return False
    try:
      home = self.home_cats()[0]
      if not rep.ids: return False
      if home and (self.menu_rows(home, 1, [], self.needs.quantity())
            or any(rep.dense_n(v) for v in home)): return False
    except Exception: return False
    state['own_surveyed'] = True
    kind = ' '.join(str(self.needs.parsed.get('product') or '').split())
    if not kind or self.used_query(kind) or self.steps_rem() <= self.order_due() + 8:
      return False
    shape = self.search_shape()
    args = {**shape[0], shape[1]: kind} if shape else self.plain_args(self.role('search'), kind)
    args, faults = ArgCheck.vet(args, self._search_form())
    if faults or not isinstance(args, dict) or kind not in self.arg_texts(args): return False
    hb = self.cable.ticks
    self.log.append({'kind': 'own_survey', 'query': kind})
    self.walk_call(args,
            'Looking up the requested product by its own name first, so that the fields its listings carry are known before any question is asked.')
    return self.cable.ticks != hb
  def _sml_ask_brief(self, w):
    left, log = str(w['ceiling'] - self.msgs_sent), self.questions_log
    segs = [ASK_PLAN.replace('QUESTION_COUNT', left), ONE_ASK.replace('REMAINING', left),
        "\n\nTHE REQUEST IN THE SHOPPER'S OWN WORDS:\n" + self.request,
        '\n\nREQUIREMENTS THAT APPLY RIGHT NOW:\n' + self.needs.body_text()]
    if log: segs.append("\n\nQUESTIONS ALREADY SENT (each followed by the shopper's reply):\n" + '\n'.join(log))
    if self.needs.parsed.get('ask_first'):
      segs.append('\n\nMUST BE LEARNED FROM THE SHOPPER: ' + '; '.join(str(m) for m in self.needs.parsed.get('ask_first') or []))
    segs.append(self.plan_notes(None))
    if hints := self._lhints():
      segs.append('\n\nSTORY HINTS. These are not requirements. Each line: a detail from the story => what it probably concerns. Usually one of these properties carries what the shopper leaves unsaid.\n' + '\n'.join('- ' + self.hint_key(h) for h in hints[:8]))
    shown = self.menu_trim(w['menu'], w['sets'])
    if rows := self.needs.items():
      segs.append('\n\nPRODUCTS THE REQUEST ASKS FOR: ' + ' | '.join(f'{n}) {r["product"]} x{r["quantity"]}' for n, r in enumerate(rows, 1)))
      shown = self._cats(w, shown)
    segs.append("\n\nKEYS IN THE LISTINGS' DETAILS SEEN SO FAR (key name, how many listings carry it, sample values):\n" + (shown or '(nothing seen so far)'))
    w['brief'] = ''.join(segs)
    w['asked'] = {n.strip().lower() for e in log for n in e.split(':', 1)[0].strip('- ').split(';')}
    return 'pick'
  def _cats(self, w, shown):
    rep = self.rec_db
    seen, abodes = rep.shelf_read(), self.home_cats()
    pool = set().union(*abodes) if seen else set()
    own = [s for s in seen if s.strip().lower() in pool]
    order = (own or seen) if all(abodes) else own + [s for s in seen if s not in own]
    filled = [(g, m) for g, m in [(g, rep.prop_list(g)[0]) for g in order[:4]] if m]
    whole = '\n'.join('CATEGORY ' + g + ':\n' + m for g, m in filled)
    trimmed = '\n'.join('CATEGORY ' + g + ':\n' + self.menu_trim(m, rep.dense_n(g)) for g, m in filled)
    w['menu'] = whole or w['menu']
    return trimmed or shown
  def _fresh(self, w, theme):
    name = str(theme or '').strip()
    low = name.lower()
    if not name or low in w['asked'] or junk_key(name) or enum_pair(name): return False
    if low in vars(self).get('spent_asks', ()): return False
    # a plural catch-all key ("mixer features", "options") reads to the shopper as several questions at once
    if re.search('(?:features|options)$', _crest(name).strip().lower()): return False
    k = self._ql_w(name)
    return not self.plain_hint(k, name) and self.plan_open(k, name)
  def _sml_ask_pick(self, w):
    q = self.pending_seconds = [x for x in vars(self).get('pending_seconds') or [] if self._fresh(w, x)]
    head = w['pending'] = q[0] if q else ''
    if not head: return 'consult'
    text = (vars(self).get('pending_text') or {}).pop(head.lower(), '') or self._lplain(head)
    w['answer'] = {'questions': [{'property': head, 'question': text}]}
    return 'script'
  def _sml_ask_consult(self, w):
    plan = w['answer'] = self.cable.consult(w['brief'], 2500, POSER)
    if plan is None and self.cable.misses > 0: self.plan_wait = w['sets']
    else: return 'script'
  def _audit(self, w, item):
    extra = _cell(item, 'second').strip()
    if _NEVER.search(extra.lower()): extra = ''
    if not self._fresh(w, item.get('property')):
      ok = extra and self._fresh(w, extra)
      return [{'property': extra, 'question': self._lplain(extra)}] if ok else []
    e = dict(item, second=extra if extra and self._fresh(w, extra) else '')
    if extra and self.plain_hint(self._ql_w(extra), extra):
      e['question'] = self._lplain(_cell(item, 'property').strip())
    return [e]
  def _sml_ask_script(self, w):
    qs = (w['answer'] or {}).get('questions') or []
    picks = [e for x in qs if isinstance(x, dict) and _cell(x, 'question').strip() for e in self._audit(w, x)]
    if not w['pending'] and not self.needs.items():
      if picks: picks = self._widest(w, picks)
      elif w['sets'] >= self.ask_sets() and not self._unread():
        if (name := next((n for n, _ in self._lkeys() if self._fresh(w, n)), None)) is not None:
          self.log.append({'kind': 'menu_fallback', 'key': name})
          picks = self._ql_one(name)
    w['script'] = picks
    if w['pending'] or not self.needs.items(): return 'send'
    w['script'] = self.share_first(picks, lambda q: self._fresh(w, q))
    self.plan_spares = picks + [e for e in vars(self).get('plan_spares') or [] if e not in picks]
    return True if vars(self).pop('quota_wait', False) else 'send'
  def _lkeys(self, every=False):
    menu = '\n'.join(self.rec_db.prop_list(v)[0] for v in self._lhome())
    keys = []
    for raw in menu.splitlines():
      m = re.match('^- (.+?) \\(in (\\d+) listings\\)(?: \\[varies: (\\d+) distinct values\\])?', raw.strip())
      if m and (every or int(m.group(2)) < 3 or int(m.group(3) or 1) < int(m.group(2))):
        keys.append((m.group(1).strip(), int(m.group(2))))
    return sorted(keys, key=lambda d: -d[1])
  def _ql_one(self, name): return [{'property': name, 'second': '', 'question': self._lplain(name)}]
  def _widest(self, w, picks):
    try:
      asked = len(self.questions_log)
      if not asked or len(vars(self).get('quiet_asks') or []) < asked: return picks
      theme = _cell(picks[0], 'property').strip()
      hinted = [m for a in self._hints() for m in re.split('\\s*/\\s*', str(a).partition(':')[0])]
      if any(same_prop(re.split('\\s*\\(', m)[0], theme) for m in hinted if m.strip()): return picks
      keys = self._lkeys()
      borne = next((n for name, n in self._lkeys(True) if same_prop(name, theme)), 0)
      wide = next(((name, n) for name, n in keys if self._fresh(w, name)), None)
      if wide is None or wide[1] < borne + 2 or same_prop(wide[0], theme): return picks
      self.log.append({'kind': 'widest_first', 'key': wide[0], 'over': theme})
      return self._ql_one(wide[0]) + picks
    except Exception: return picks
  def _profile(self, w, item):
    if self.needs.items():
      theme = _cell(item, 'property')
      prod = self._ql_tag(theme, '')
      words, q = re.findall('[a-z]{3,}', prod.lower()), _cell(item, 'question').lower()
      if prod and words and all(x.rstrip('s') not in q for x in words):
        item = dict(item, question=_which(theme, prod))
    if self.needs.parsed.get('ask_first') and not self.questions_log: return item
    text, theme, extra = (_cell(item, s) for s in ('question', 'property', 'second'))
    menu = w['menu']
    if _digest(text, theme, menu) or not w['pending'] and self.name_pair(text, theme, menu, extra):
      return dict(item, question=self._lplain(theme))
    return item
  def _emit(self, w, item):
    text = str(item.get('question')).strip()
    cid = f'ask_{self.msgs_sent + 1}_{self.cable.ticks}'
    w['call'] = self.msg_call(text, cid)
    if not w['call']: return False
    theme = str(item.get('property'))
    w['label'] = (theme, text)
    extra = _cell(item, 'second').strip()
    if not extra or extra.lower() == theme.lower() or _NEVER.search(extra.lower()): return True
    one = self._lplain(theme)
    w['call'], w['label'] = self.msg_call(one, cid), (theme, one)
    if w['call']: w['split'] = extra
    return bool(w['call'])
  def _sml_ask_send(self, w):
    w['split'] = ''
    return 'book' if any(self._emit(w, self._profile(w, it)) for it in w['script'][:2]) else 'empty'
  def _unread(self):
    try:
      n = len(self.needs.items())
      return [k for k in range(n or 1) if self.prod_reads(k if n else None) < 2]
    except Exception: return []
  def _sml_ask_empty(self, w):
    room = self.steps_rem() > 12
    if not (room and not w['urge'] and (w['sets'] < self.ask_sets() or self._unread())):
      self.empty_plans = vars(self).get('empty_plans', 0) + 1
      if not (room and self.empty_plans <= 1): return self._idle()
    self.plan_wait = w['sets']
  def _say(self, text, call): self.chat.append({'role': 'assistant', 'content': text, 'tool_calls': [call]})
  def _sml_ask_book(self, w):
    theme = w['asked_label'] = w['label'][0]
    self.asked_at_sets, self.asked_at_tick = w['sets'], self.cable.ticks
    self._say(f'Asking the shopper about {theme}.', w['call'])
    w['found0'], w['freed0'] = len(self.needs.uncovered), self._free_n()
    self.asking_label = theme
    try: w['replies'] = self.relay([w['call']], programmed=True)
    finally: self.asking_label = ''
    self.last_replies(w['replies'])
    if (head := w['pending']) and _headpt(theme).lower() == head.lower(): self.pending_seconds = self._minus(head)
    w['note'], w['meta'] = '', vars(self.needs).pop('meta_last', None)
    return 'reply'
  def _sml_ask_reply(self, w):
    theme, text = w['label']
    meta, heard, alt = w['meta'], _flat(text), ';' in theme
    if meta is not None and heard and heard in _flat(meta): self._zp_one(w)
    elif alt and len(self.needs.uncovered) > w['found0']: self._zp_alt(w, False)
    elif alt and self._free_n() > w['freed0']: self._zp_alt(w, True)
    elif alt and self._free_n() == w['freed0']:
      self.no_pairs = True
      self._hold([q.strip() for q in theme.split(';') if q.strip()])
      w['label'] = (theme.replace(';', ' and') + ' [not answered]', text)
    elif w['split']: self._hold([] if self._cited(w, w['split']) else [w['split']])
    return 'log'
  def _zp_one(self, w):
    theme, text = w['label']
    self.no_pairs = True
    segs = [q.strip() for q in theme.split(';') if q.strip()] or [theme]
    lead = segs[0]
    low, again = lead.lower(), vars(self).setdefault('reasked', set())
    plain, alone = self._lplain(lead), len(segs) == 1
    sub = self.hit_alt(lead) if alone else ''
    if sub or alone and self.needs.items():
      again.add(low)
      self._spent_q(lead)
      if sub: self._tofront(sub)
      w['note'] = _jot(lead, f"no answer; its story hint's alternative {sub} is asked next" if sub else
              'no answer; dropped, and the next question goes to another product')
    elif low in again or _flat(plain) == _flat(text) or alone and self.keeps_plain(text, plain):
      w['note'] = self._rest(lead, again)
    else:
      again.add(low)
      self._tofront(lead)
      w['note'] = _jot(lead, 'no answer; asked again on its own next')
    self._hold(segs[1:] + ([w['split']] if w['split'] else []))
    w['label'] = (' and '.join(segs) + ' [not answered] [one at a time]', text)
  def _rest(self, lead, again):
    redo = None if lead.lower() in again else self.solo_ask(lead)
    if not redo:
      self._spent_q(lead)
      return _jot(lead, 'no answer; it will not be asked again')
    name = redo[0]
    again.update({lead.lower(), name.lower()})
    vars(self).setdefault('pending_text', {})[name.lower()] = redo[1]
    self._tofront(name)
    if re.split('\\s*\\(', name)[0].strip().lower() == re.split('\\s*\\(', lead)[0].strip().lower():
      return _jot(lead, 'no answer; asked again next, this time through its values')
    self._spent_q(lead)
    return _jot(lead, 'no answer; asked again next as ' + name)
  def _zp_alt(self, w, freed):
    theme, text = w['label']
    other = theme.partition(';')[2].strip()
    w['label'] = (_headpt(theme), text)
    if freed:
      gone = {other.lower(), other.partition('(')[0].strip().lower()}
      self.needs.indifferent = [q for q in self.needs.indifferent if q.strip().lower() not in gone]
      self._hold([other])
    elif self._cited(w, other): return
    else: self.queue_second(other)
    w['note'] = _jot(other, 'no answer; still open')
  def _sml_ask_log(self, w):
    self.show_note(self.needs.uncovered[w['found0']:])
    if w['meta'] is None and len(self.needs.uncovered) == w['found0']:
      vars(self).setdefault('quiet_asks', []).append(_headpt(w['asked_label']))
    theme, text = w['label']
    for _, body in w['replies'][:1]:
      self.questions_log.append(f'- {theme}: asked "{text}" -> {body.split(_STATUS_CUT)[0][:400]}' + w['note'])
    self.drop_want(w['asked_label'])
    self.asking = self.msgs_sent < w['ceiling']
    return True
  def reply_back(self, phrase):
    if (not self.role('message') or self.msgs_sent >= self.ask_room() or self.steps_rem() <= 4
        or self.cable.spare() < CLOSING): return False
    if self.asking and self.msgs_sent >= self.ask_room() - 1 and self.steps_rem() > 10:
      if self.cable.spare() >= CLOSING + 40: return False
    call = self.msg_call(phrase, f'reply_{self.cable.ticks}')
    if not call: return False
    self._say("Answering the shopper's question.", call)
    self.last_replies(self.relay([call], programmed=True))
    return True
  def last_replies(self, responses):
    last = len(responses) - 1
    for at, (cid, body) in enumerate(responses):
      body = body + '\n' + self.state_line() if at == last else body
      self.chat.append({'role': 'tool', 'tool_call_id': cid, 'content': body})
  def plain_args(self, summon_appellation, phrase, kind=''):
    schema = self.argspec.get(summon_appellation) or {}
    props = schema.get('properties') or {}
    out, tip = {}, {'decline': NO_BUY_WORDS, 'order': ORDER_WORDS}.get(kind, '')
    for n in schema.get('required') or []:
      for v in _pad(props.get(n) or {}, phrase, tip): out[n] = v
    if len(props) == 1:
      (only, spec), = props.items()
      if only not in out and isinstance(spec, dict) and spec.get('type') == 'string' and 'enum' not in spec:
        out[only] = _q5_cut(phrase, spec)
    return out
  def clean_decl(self, prms_into, summon_appellation):
    rule = self.needs.parsed.get('decline_no_names')
    if rule is False: return prms_into
    known = self.rec_db.ids
    def named(m): return isinstance(m, str) and any(x and x in known for x in re.split(_ID_CUT, m))
    if not any(named(m) for m in prms_into.values()) and not rule: return prms_into
    lay = (self.argspec.get(summon_appellation) or {}).get('properties') or {}
    def void(k, m): return m if not isinstance(m, str) or 'enum' in (s := lay.get(k) or {}) else _q5_cut(PARDON, s)
    return {k: void(k, m) for k, m in prms_into.items()}
  def code_reading(self):
    request = self.request
    low = request.lower()
    cap, unit = self.budget_of(request)
    plan = dict(code_reading=True, requirements=[], items=[], quantity=1, rules=[], soft_wants=[], ask_first=[],
          hints=[], references=[], priority_detail='', price_cap=cap, currency=unit,
          in_stock_only=bool(re.search(_AVAIL, low)),
          priority='cheapest' if re.search(_LOWEST, low) else 'none')
    for slot, nominals, floor in _LIMITS:
      if (f := _cap_of(low, nominals)) is not None and f >= floor: plan[slot] = f
    roles = plan['roles'] = self.guess_limits()
    found = self.find_items(tool := roles.get('search'))
    if len(found) > 1:
      plan.update(items=[{k: r[k] for k in ('product', 'quantity', 'requirements')} for r in found],
            product=' and '.join(r['product'] for r in found),
            requirements=[n for r in found for n in r['requirements']])
      return plan
    if found: plan.update(product=found[0]['label'], quantity=found[0]['quantity'])
    else: plan.update(product=self.prod_label(tool, kind_words(request)), quantity=_count_of(low))
    plan['requirements'] = [f'{prop}: {value}' for prop, value in _needs(low)]
    return plan
  @staticmethod
  def budget_of(phrase):
    best = []
    for st in re.split('(?<=[.!?;])\\s+', str(phrase).lower()):
      prem = 1 if re.search('\\b(?:order|total|whole|lot|sum|bill|altogether|overall|everything)\\b', st) else 0
      for m in re.finditer('([$\\u20ac\\u00a3]\\s*)?(\\d[\\d,]*(?:\\.\\d+)?)', st):
        if (g := _grade(st, m, prem)) is not None and g >= 2:
          best.append((g, float(m.group(2).replace(',', '')), m, st))
    if not best: return (None, None)
    _, f, m, st = max(best, key=lambda r: r[:2])
    return (int(f) if f.is_integer() else f, _gauge(st[m.end():], m.group(1)))

class PilotSetup:
  _qm_datapoint, _prop_fields = staticmethod(_num), staticmethod(_cells)
  code_seen = staticmethod(_seen_in)
  def _shape(self, tool):
    lf = self.argspec.get(tool or '') or {}
    return (lf, props if isinstance(props := lf.get('properties'), dict) else {})
  def _tshelves(self, tool):
    opts = (_enum(s) for s in self._shape(tool)[1].values())
    return max((o for o in opts if isinstance(o, list) and len(o) >= 4), key=len, default=[])
  def find_items(self, tool):
    shelf, text = [str(v) for v in self._tshelves(tool)], self.request.lower()
    trms = [(m.group(0).rstrip('s'), *m.span()) for m in re.finditer('[a-z]{3,}', text)
        if m.group(0) not in STOP_WORDS]
    at = {}
    for v in shelf:
      own = [x.rstrip('s') for x in _terms(v, '[a-z]{3,}')]
      if not own or v in at: continue
      if (p := _qty_at(text, trms, own)) and 0 < p[2] < 100: at[v] = p
    found = sorted(at.items(), key=lambda d: d[1][0])
    out = []
    for i, (v, (_, end, units)) in enumerate(found):
      ext = text[end:found[i + 1][1][0]] if i + 1 < len(found) else re.split('(?<=[.!?])\\s', text[end:])[0]
      prod = v.lower()
      needs = ['%s (%s): %s' % (p, prod, x) for p, x in _needs(ext)]
      out.append({'product': prod, 'label': v, 'quantity': units, 'requirements': needs})
    return out
  def _slip(self, name, text):
    lf, args = self._shape(name)
    need = [k for k in sought if k in args] if isinstance(sought := lf.get('required'), list) else []
    def two(k):
      low = [str(v).lower() for v in args[k].get('enum') or []] if isinstance(args[k], dict) else []
      return any(re.search(ORDER_WORDS, t) for t in low) and any(re.search(NO_BUY_WORDS, t) for t in low)
    both = any([two(k) for k in need])
    specs = [s for s in args.values() if isinstance(s, dict)]
    return {'name': name, 'text': text, 'head': text.split(' | ')[0], 'need': need, 'both': both,
        'free': _free(args, 'text'), 'array': any(s.get('type') == 'array' for s in specs),
        'choice': any(s.get('enum') for s in specs)}
  def _slips(self):
    texts = {}
    for fx in (e['function'] for e in self.tool_specs):
      head = _spaced(fx['name']).replace('_', ' ').lower()
      texts[fx['name']] = '%s | %s' % (head, str(fx.get('description') or '').lower())
    return [self._slip(n, t) for n, t in texts.items()]
  def guess_limits(self):
    cards, roles = self._slips(), {}
    for n in [c['name'] for c in cards if c['both']]:
      roles.setdefault('order', n)
      roles.setdefault('decline', n)
    tips = (('decline', NO_BUY_WORDS), ('order', ORDER_WORDS))
    for last in (False, True):
      for c in cards:
        if c['name'] in roles.values() or last and (c['need'] or not re.search('\\bfinal\\b', c['text'])):
          continue
        words = c['text'] if last else c['head']
        if r := next((r for r, tip in tips if r not in roles and re.search(tip, words)), None):
          roles[r] = c['name']
    for c in cards:
      if c['name'] not in roles.values() and re.search('\\b(?:cart|basket|bag|trolley)\\b', c['text']):
        roles.setdefault('cart' if c['need'] or c['array'] else 'cart_view', c['name'])
    for r, tip in _TIPS:
      if r in roles: continue
      def fit(c): return c['name'] not in roles.values() and re.search(tip, c['text']) and _fits(r, c)
      if (hit := next(filter(fit, cards), None)) is not None: roles[r] = hit['name']
    return roles
  def fix_cap(self):
    held = self.role('search')
    try: cards = self._slips()
    except Exception: return
    mine = next((c for c in cards if c['name'] == held), None)
    if mine is None or mine['free'] and not self._cap_tool(held): return
    for c in cards:
      if c['name'] in self.role_map.values() or not c['free']: continue
      if re.search(_TIPS[0][1], c['text']) and _fits('search', c):
        self.role_map['search'] = c['name']
        self.log.append({'kind': 'search_role', 'from': held, 'to': c['name']})
        return
  def prod_label(self, tool, require):
    best, top = '', None
    for v in self._tshelves(tool):
      own = kind_words(v)
      n = len(own & require)
      g = (n, n / max(1, len(own)))
      if n and (top is None or g > top): best, top = str(v), g
    return best
  def need_vals(self, k):
    several, kept = bool(self.needs.items()), []
    for need in map(str, self.needs.props_left()):
      head, _, value = need.partition(':')
      if several and (w := self.prod_of(head)) is not None and w != k: continue
      bare = re.sub('\\((?:stated|revealed[^)]*)\\)', '', value).strip()
      low = bare.lower()
      short = len(re.findall('[a-z0-9]+', low)) <= 5 and not re.search(NEG_WORDS, low)
      if (confirm_val(value, head) or short) and bare and not is_ref_text(bare):
        kept.append((head.split('(')[0].strip(), bare))
    return kept
  def _form(self, tool):
    lf, args = self._shape(tool)
    need = [k for k in lf.get('required') or [] if k in args]
    about = next((str(e['function'].get('description') or '') for e in self.tool_specs
           if e['function']['name'] == tool), '')
    clauses = [m.lower() for m in re.split('[.;]', about)]
    lex = self.term_key()
    free = [k for k in _free(args, 'text') if k != lex]
    phrased = [k for k in free if re.search(
      '^q$|query|text|keyword|needle|term|word|phrase|string|search', k.lower())]
    lists = {k: o for k, o in ((k, _enum(s)) for k, s in args.items()) if isinstance(o, list) and o}
    shelf = max((k for k in lists if len(lists[k]) >= 4), key=lambda k: len(lists[k]), default=None)
    styles = [k for k, o in lists.items() if k != shelf and len(o) <= 4]
    def rated(name):
      low = name.lower()
      if re.search('page|first|offset|skip|count|size|num', low): return False
      if re.search('ceil|max|price|cap$|budget|upper|most', low): return True
      return any(low in s and re.search('\\bpric|\\bcost', s) for s in clauses)
    return {'schema': lf, 'args': args, 'need': need, 'clauses': clauses, 'word': (phrased or free or [None])[0],
        'lists': lists, 'shelf': shelf, 'mode': ([k for k in styles if k in need] or styles or [None])[0],
        'ceiling': next(filter(rated, _free(args, 'number')), None)}
  def _phrases(self, k, product, label, room):
    names = [x for x in _terms(product) if len(x) > 2]
    kind = names[-1] if names else ''
    robust = [v for _, v in self.need_vals(k) if not re.search('\\bor\\b|/|,|' + _LK, v.lower())]
    lead = _terms(robust[0]) if robust else []
    if lead: ws = [_uniq(lead + [kind])] + ([_uniq([max(lead, key=len), kind])] if len(lead) > 1 else [])
    elif names: ws = [_uniq([names[0], names[-1]])]
    else: ws = [' '.join(sorted(kind_words(label))[:2])]
    if not is_num(room): return ws
    def fit(t):
      while ' ' in t and len(t) > room: t = t[:t.rindex(' ')]
      return t[:int(room)]
    return [fit(t) for t in ws]
  def code_queries(self, k):
    lf = self._form(tool := self.role('search'))
    args, word, shelf, mode, ceil = lf['args'], lf['word'], lf['shelf'], lf['mode'], lf['ceiling']
    rows = self.needs.items()
    ok = rows and k is not None and k < len(rows)
    product = str(rows[k]['product'] if ok else self.needs.parsed.get('product') or '')
    label = self.prod_label(tool, kind_words(product) or kind_words(self.request))
    cap, per = self.needs.spend_cap, self.needs.counts() if k is not None else []
    units = (per[k] if k is not None and k < len(per) else self.needs.quantity()) or 1
    limit = {}
    if ceil and is_num(cap) and cap > 0:
      limit[ceil] = int(cap // units) if args[ceil].get('type') == 'integer' else round(cap / units, 2)
    spec = args.get(word) if word else None
    ws = self._phrases(k, product, label, spec.get('maxLength') if isinstance(spec, dict) else None)
    plan = []
    if word and ws[0]: plan.append((word, ws[0], False))
    if label and shelf: plan.append((shelf, label, True))
    if word and len(ws) > 1: plan.append((word, ws[1], False))
    out = []
    for key, val, last in plan:
      p = {key: val}
      if mode: p[mode] = _style(lf, key, last)
      p.update(limit)
      _fill(p, lf, ws[0] or label)
      r = ArgCheck.vet(p, lf['schema'])
      if r[1] and limit: r = ArgCheck.vet({n: v for n, v in p.items() if n not in limit}, lf['schema'])
      if not r[1] and r[0] not in out: out.append(r[0])
    return out
  def _merge(self, seen, n, dx):
    row = seen.setdefault(n, {'shelf': '', 'title': '', 'price': None, 'stock': None, 'words': set(), 'fields': {}})
    row['shelf'] = self.shelf_for(dx) or row['shelf']
    def is_title(f, m):
      return isinstance(m, str) and 'id' not in (f := str(f).lower()) and re.search('title|name', f)
    if (title := next((str(m) for f, m in dx.items() if is_title(f, m)), None)) is not None:
      row['title'] = title or row['title']
    price = next((float(m) for k, m in dx.items() if is_num(m) and _is_price(str(k).lower())), None)
    if price is not None: row['price'] = price
    stock = [s for s in (self.key_stock(k, m) for k, m in dx.items()) if s is not None]
    if stock: row['stock'] = stock[-1]
    atomic = (str, int, float)
    nest = [m for m in dx.values() if isinstance(m, dict)]
    flat = [str(m) for m in dx.values() if not isinstance(m, (dict, bool)) and isinstance(m, atomic)]
    flat.extend(str(x) for sub in nest for x in sub.values() if isinstance(x, atomic))
    row['words'].update(kind_words(' '.join(flat)))
    for sub in nest:
      for k, x in sub.items():
        if isinstance(x, atomic) and not isinstance(x, bool): row['fields'][str(k)] = str(x)
  def all_rows(self):
    seen = {}
    def take(dx):
      if (n := self.row_key(dx)) in self.rec_db.ids: self._merge(seen, n, dx)
    for banked in self.rec_db.items: walk_dicts(banked, take)
    return seen
  @staticmethod
  def row_shows(row, prop, value, success_lexemes, read):
    if cells := _cells(row['fields'], prop): return any(_seen_in(value, t) for t in cells)
    if _in_title(row, value): return True
    want = kind_words(value) if not read else set()
    return bool(want) and want <= success_lexemes
  def _qm_ctx(self, avoid, keep):
    state, rev, read = self.__dict__, str(self.needs.rev_no), self.rec_db.read
    failed = {t.split('|')[0] for t, ruled in self.rulings.items() if ruled and t.split('|')[1:2] == [rev]}
    live = {h for h, _, tl in (t.rpartition('|') for t in self.passed) if tl == rev}
    seen = state.get('audit_ids')
    if state.get('audit_done') and (seen is None or len(read) <= state.get('audit_reads', -1)): seen = set(read)
    last = next((e for e in reversed(self.log) if isinstance(e, dict) and e.get('kind') == 'audit'), None)
    audited = set()
    if last is not None and isinstance(last.get('result'), dict):
      audited = set(seen or ()) - set(state.get('audit_fits') or ()) - live
    spend = self.needs.spend_cap
    return {'failed': failed, 'live': live, 'audited': audited,
        'own_all': bool(audited) and self.rate_all(),
        'cap': float(spend) if is_num(spend) else None, 'asked': kind_words(self.request),
        'hits': state.get('code_hits') or {}, 'cheapest': self.needs.priority() == 'cheapest',
        'avoid': set(avoid or ()) | self.dead_lines() | self.ref_ptrs(),
        'keep': set(keep or ()) | set(self.pair_best())}
  def _mpool(self, s, info, k, home, qty):
    slot = k if (rows := self.needs.items()) else None
    require = kind_words(rows[k]['product'] if rows else self.needs.parsed.get('product') or '')
    mags, judged = set(), []
    for p, v in self.need_vals(slot):
      mags.update(kind_words(v))
      if not _BOUND.search(v.lower()): judged.append((p, v))
    def opts(h): return [str(v) for v in [h.get('value'), *(h.get('goes_with') or [])] if str(v or '').strip()]
    apart = [(h['prop'], opts(h)) for h in self.differ_done(slot)]
    cap, hits, read = s['cap'], s['hits'], self.rec_db.read
    def rank(sym, row):
      if sym in s['failed'] or sym in self.refused_lines or sym in s['avoid'] or row['stock'] is False: return
      if not (row['shelf'] in home if home else bool(require & row['words'])): return
      if any(self.field_clash(row['fields'], p, v) for p, v in judged): return
      if self.ref_excluded(row['fields'], slot): return
      cost = self.rec_db.price(sym) if row['price'] is None else row['price']
      rated = is_num(cost)
      if rated and cap is not None and qty * cost > cap: return
      seen, was = hits.get(sym) or set(), sym in read
      fits = all(self.row_shows(row, p, v, seen, was) for p, v in judged)
      fits = fits and _apart(row['fields'], apart)
      shows = [_prec(row, p, v) for p, v in judged] if was else []
      if sym in s['audited']:
        if not (s['own_all'] and fits and (judged or apart) and all(max(x) <= 1 for x in shows)): return
      w = row['words']
      g = 3 * len(mags & w) + 2 * len(require & w) + len(s['asked'] & w) + 4 * len(mags & seen)
      price = cost if rated else _INF
      return (int(sym not in s['keep']), int(sym not in s['live']), not fits, cap is not None and not rated,
          price if s['cheapest'] else 0.0, int(not was), row['stock'] is not True,
          (tuple(map(sum, zip(*shows))) if shows else (9, 9), -g), price, sym)
    return sorted(e for e in (rank(sym, info[sym]) for sym in info) if e is not None)
  def code_picks(self, avoid=None, keep=None):
    several, per = self.needs.items(), self.needs.counts()
    abodes = self.home_cats() if several else [set(self.shelf_labels()) or (self.home_cats() or [set()])[0]]
    info, s = self.all_rows(), self._qm_ctx(avoid, keep)
    units = [per[k] if k < len(per) else 1 for k in range(len(abodes))]
    pools = []
    for k, home in enumerate(abodes):
      if not (pool := self._mpool(s, info, k, home, units[k])): return []
      pools.append(pool)
    for order in (None, lambda e: (e[2], e[3], e[8], e[-1])):
      if (chosen := self._qm_opt(pools, order)) is None: return []
      spent = 0.0
      for e, q in zip(chosen, units): spent += (0.0 if e[8] == _INF else e[8]) * q
      if s['cap'] is None or spent <= s['cap']:
        self.code_fit = {e[-1]: not e[2] for e in chosen}
        return list(zip((e[-1] for e in chosen), units))
    return []
  def _qm_opt(self, pools, order):
    got, read = [], self.rec_db.read
    for pool in pools:
      used = [e[-1] for e in got]
      def ok(c): return c[-1] not in used and not (c[-1] in read and self.item_view(c[-1])['fails'])
      if (pick := next(filter(ok, pool if order is None else sorted(pool, key=order)), None)) is None: return None
      got.append(pick)
    return got
  @staticmethod
  def field_clash(fields, prop, value):
    cells = _cells(fields, prop)
    return bool(cells) and not any(val_seen(value, t) for t in cells)
  def _cart(self):
    lf, args = self._shape(self.role('cart'))
    first = as_json(self.barrow.add_form) if self.barrow.add_form else None
    put, drop = {}, {}
    for key in lf.get('required') or []:
      opts = _enum(args.get(key))
      if not isinstance(opts, list) or not opts: continue
      lw = [set(re.findall('[a-z]+', _spaced(str(v)).lower())) for v in opts]
      if isinstance(first, dict) and first.get(key) in opts: put[key] = first[key]
      else: put[key] = next((v for v, x in zip(opts, lw) if x & ADD_WORDS and not x & DROP_WORDS), opts[0])
      take = [v for v, x in zip(opts, lw) if x & DROP_WORDS and v != put[key]]
      drop[key] = take[0] if take else [v for v in opts if v != put[key]][0] if len(opts) == 2 else None
    free = _free(args, 'text')
    named = [k for k in free if re.search('id|item|product|listing|sku|target|which|number', k.lower())]
    return {'schema': lf, 'args': args, 'put': put, 'drop': drop, 'id': (named or free or [None])[0],
        'units': _free(args, 'number')}
  def cart_args(self, picks):
    cart = self._cart()
    put, sid, units = cart['put'], cart['id'], cart['units']
    if lay := self.barrow.line_spec():
      vec, sid, cnt = lay
      calls = [{**put, vec: [{sid: s, cnt: q} for s, q in picks]}]
    elif not sid: return []
    else: calls = [{**put, sid: s, **({units[0]: q} if units else {})} for s, q in picks]
    return _vetted(calls, cart['schema']) or []
  def cart_drops(self, lines):
    if not lines or self.barrow.line_spec(): return []
    cart = self._cart()
    sid, units, put = cart['id'], cart['units'], cart['put']
    if not sid: return None
    if put:
      if any(cart['drop'].get(k) is None for k in put): return None
    else:
      floor = (cart['args'].get(units[0]) or {}).get('minimum') if units else None
      if not units or is_num(floor) and not floor <= 0: return None
    calls = []
    for s in lines:
      one = {**cart['drop'], sid: s} if put else {sid: s, units[0]: 0}
      if put and units: one[units[0]] = max(1, int(self.barrow.held(s) or 1))
      calls.append(one)
    return _vetted(calls, cart['schema'])
  def last_obs(self, n):
    resp = next((e for e in reversed(self.log) if isinstance(e, dict) and e.get('kind') == 'env'), {}).get('reply')
    if not isinstance(resp, dict): return []
    many = resp.get('observations')
    return list(many if isinstance(many, list) else [resp.get('observation')])[:n]
  @staticmethod
  def reply_err(obs):
    hits = []
    def note(dx):
      for name, m in dx.items():
        if m not in (None, False, '', [], {}) and re.search(ERR_KEYS, str(name).lower()):
          hits.append(name)
          return
    walk_dicts(obs, note, 0, 2)
    return bool(hits)
  _NEXT_KEYS = 'next|onward|cursor|bookmark|token|continu|more'
  _COUNT_KEYS = 'count|total|matched|found|hits'
  _EMPTY_VALS, _SOFT_ERRS = (None, False, '', 0), ('invalid_args', 'unknown_tool')
  _PG_KEYS, _NONPAGE_KEYS = 'page|offset|first|start|skip|from|cursor', 'size|per_?page|limit|price|cost'
  _smn_EDGE_PUNCT, _smn_SHELF_KEYS = '()[]{}"\'.,;:!?/|', 'categor|shelf|department|aisle'
  _smn_GONE_WORDS = '\\b(?:out|sold|none|unavailable|not|no|false)\\b'
  _smn_LATE_WORDS, _SHOP_LAPS = 'back.?order|pre.?order|discontinu', 4

class PilotMoves:
  def _log_event(self, kind, **fields): self.log.append({'kind': kind, **fields})
  def _log_err(self, kind, mistake): self._log_event(kind, error=f'{type(mistake).__name__}: {mistake}'[:300])
  def _volley(self, lemma, tool, enumerated):
    hb = self.cable.ticks
    self.emit_all([(f'{lemma}_{ix}_{hb}', tool, parameters) for ix, parameters in enumerated])
  def code_search(self, products, more=False):
    tool, hold = self.role('search'), 2 + 2 * products
    free = [k for k, v in (self._tschema().get('properties') or {}).items()
        if isinstance(v, dict) and v.get('type') == 'string' and not v.get('enum')]
    if more:
      lead = list(vars(self).get('code_more') or [])
      calls, rounds = lead[:max(0, self.steps_rem() - hold - 2 * products)], 1
    else:
      qs = [self.code_queries(k if self.needs.items() else None) for k in range(products)]
      calls = []
      for d in range(max(map(len, qs), default=0)):
        calls += [q for q in [x[d] for x in qs if len(x) > d] if q not in calls]
      calls, lead = calls[:max(0, self.steps_rem() - hold)], [x[0] for x in qs if x]
      rounds = 1 + max(CODE_PAGES + 2 * (self.needs.priority() == 'cheapest'), self.deep_pages())
    if not calls: return
    self.code_searched = True
    rnd = 0
    while calls:
      self._volley('code_search', tool, list(enumerate(calls)))
      replies = self.last_obs(len(calls))
      st = vars(self)
      if st.get('word_kind') != 'any':
        tags = st.setdefault('code_hits', {})
        for p, r in zip(calls, replies):
          txt = ' '.join(str(p.get(n) or '') for n in free)
          if txt.strip():
            ws, known = kind_words(txt), self.rec_db.ids
            walk_dicts(r, lambda row: (i := self.row_key(row)) in known
                   and tags.setdefault(i, set()).update(ws))
      ahead = [c for c in (self.next_cursor(p, r) for p, r in zip(calls, replies) if p in lead) if c]
      self.code_more = ahead
      rnd += 1
      if rnd >= rounds: return
      plan, fits = self.code_picks(), vars(self).get('code_fit') or {}
      settled = rnd > self.deep_pages()
      if plan and self.needs.priority() != 'cheapest' and all(fits.get(i) for i, _ in plan) and settled: return
      lead, calls = ahead, ahead[:max(0, self.steps_rem() - hold - 3 * products)]
  def _tschema(self):
    return self.argspec.get(self.role('search') or '') or {}
  def deep_pages(self):
    n = len(self.needs.items())
    return (FEW_PAGES if n <= FEW_MAX else 0) if n else STANDALONE_LEAVES
  def next_cursor(self, parameters, reply):
    paging = self.pg_arg()
    if not paging: return
    field, per, floor, ceiling = paging
    found = []
    walk_dicts(reply, lambda d: found or any(isinstance(v, list) and v and all(isinstance(x, dict) for x in v)
                          for v in d.values()) and found.append(d), 0, 3)
    if not found: return
    sizes, tips, total = [], [], None
    for k, v in found[0].items():
      lab = str(k).lower()
      if isinstance(v, list): sizes.append(len(v))
      if re.search(self._NEXT_KEYS, lab): tips.append(v)
      if total is None and is_num(v) and re.search(self._COUNT_KEYS, lab): total = v
    if tips and all(t in self._EMPTY_VALS for t in tips): return
    if per == 'token':
      tok = next((t for t in tips if isinstance(t, str) and t.strip()), None)
      if not tok: return
      nxt = {**parameters, field: tok}
    else:
      size, at = max(sizes), parameters.get(field, floor)
      if not is_num(at): at = floor
      nxt = at + (1 if per else size)
      if ceiling is not None and nxt > ceiling: return
      if not tips and (total is None or ((at - floor + 1) * size if per else at + size) >= total): return
      nxt = {**parameters, field: int(nxt) if float(nxt).is_integer() else nxt}
    anchored, faults = ArgCheck.vet(nxt, self._tschema())
    return None if faults else anchored
  def pick_read(self, plan, barred, prefer):
    for _ in range(CODE_READS):
      todo = [i for i, _ in plan if i not in self.rec_db.read]
      en = self.batch_tool()
      joint = bool(en) and len(todo) > 1
      if not todo or not (en or self.role('details')): break
      if 3 + 2 * len(plan) + (1 if joint else len(todo)) > self.steps_rem() or self.cable.spare() <= 8:
        break
      hb = self.cable.ticks
      if joint: block = [(f'code_read_{hb}', en[0], {en[1]: todo[:en[2]]})]
      else:
        tool = self.role('details')
        block = [(f'code_read_{n}_{hb}', tool, c) for n, c in enumerate(map(self.details_call, todo)) if c]
      if not block: break
      self.emit_all(block)
      fresh = self.code_picks(barred, prefer)
      if fresh == plan: break
      plan = fresh
    return plan
  def plan_bad(self, plan):
    sought, cart = {i for i, _ in plan}, self.barrow
    whole = cart.line_spec()
    others = [ln for ln in cart.units if ln not in sought]
    book = self.needs.tolds
    said = any(e.get('before_change') is True for e in book)
    if others and (said or any(e.get('before_change') is None for e in book))\
        and not self.asks_swap(others, [i for i, _ in plan]) and said:
      return
    todo = list(plan) if whole else [(i, n) for i, n in plan if abs(cart.held(i) - n) > 1e-09]
    sheds = self.cart_drops(others)
    puts = self.cart_args(todo) if todo else []
    if sheds is None or (todo and not puts): return
    calls = sheds + puts
    if calls and self.steps_rem() < len(calls) + 1: return
    if not calls: return []
    self._volley('code_cart', self.role('cart'), list(enumerate(calls)))
    replies = self.last_obs(len(calls))
    bad, shown, wide = set(), None, list(sought) if whole else []
    for p, r in zip(calls, replies):
      view = cart.view_counts(r)
      if view is not None: shown = view
      if self.reply_err(r):
        txt = as_text(r)
        named, borne = [i for i in sought if i in txt], [ln for ln, _ in cart.lines(p) if ln in sought]
        bad.update(named or borne or wide)
    if len(replies) < len(calls): return
    if shown is None and not bad and self.role('cart_view') and self.steps_rem() >= 3:
      self.emit_all([(f'code_view_{self.cable.ticks}', self.role('cart_view'), {})])
      got = self.last_obs(1)
      shown = cart.view_counts(got[0]) if got else None
      if got and shown is None: bad.update(i for i in sought if i not in cart.units)
    if shown is not None: bad.update(i for i in sought if i not in shown)
    for i in bad: cart.units.pop(i, None)
    return sorted(bad)
  def code_recheck(self, held):
    tool = self.stock_tool() or next((t for t in map(self.role, ('view', 'details')) if t), None)
    if not (held and tool) or len(held) + 2 > self.steps_rem() or self.cable.spare() < 8: return False, set()
    calls = [(ix, p) for ix, p in enumerate(self.fetch_call(i, tool) for i in held) if p]
    if not calls: return False, set()
    self._volley('code_check', tool, calls)
    stock = self.all_rows()
    return True, {i for i in held if (stock.get(i) or {}).get('stock') is False}
  def code_news(self, fresh):
    pile = [ln for b, _ in vars(self.needs).get('pending') or [] for ln in b]
    done = vars(self).setdefault('news_read', set())
    for ln in (*fresh, *pile):
      if ln in done: continue
      done.add(ln)
      vol, _ = self.budget_of(ln)
      if is_num(vol) and not vol <= 0:
        self.needs.spend_cap = vol
        self._log_event('code_news', budget=vol)
  def code_cart(self, avoid=None, keep=None):
    search = self.role('search')
    if not self.role('cart') or self._stuck(2, 4): return False
    self.cable.hush = True
    try:
      barred, n, st = set(avoid or ()), max(1, len(self.needs.items())), vars(self)
      for _ in range(self._SHOP_LAPS):
        plan = self.code_picks(barred, keep)
        unsuited = not plan or not all(st.get('code_fit', {}).get(i) for i, _ in plan)
        if search:
          steps = self.steps_rem()
          first = not st.get('code_searched') and steps >= 4 and unsuited
          more = not first and bool(st.get('code_more')) and steps >= 6 + 2 * n\
            and (unsuited or self.needs.priority() == 'cheapest')
          if first or more:
            self.code_search(n, more)
            plan = self.code_picks(barred, keep)
        plan = self.pick_read(plan, barred, keep)
        if not plan:
          self._log_event('code_finish', picks=[], result='nothing to order')
          return False
        bad = self.plan_bad(plan)
        if bad is None:
          self._log_event('code_finish', picks=plan, result='no cart call')
          return False
        if not bad:
          ok = self.steps_rem() >= len(plan) + 3
          bad = sorted(self.code_recheck([i for i, _ in plan])[1] if ok else set())
          if not bad:
            units = self.barrow.units
            self._log_event('code_finish', picks=plan, cart=dict(units))
            return bool(units)
        barred |= set(bad)
        self._log_event('code_swap', lines=bad)
      return False
    except Exception as e:
      self._log_err('code_finish_error', e)
      return False
  def finish_now(self):
    if self._stuck(1, 2): return
    order, decline = self.role('order'), self.role('decline')
    buy = lambda: {'name': order, 'args': self.plain_args(order, 'Order the cart.', 'order')}
    drop = lambda: {'name': decline, 'args': self.plain_args(decline, PARDON, 'decline')}
    barred, prefer, move, kind, resp, turn = set(), None, None, None, None, 0
    while turn < CODE_FINISHES:
      if self._stuck(1, 2): return
      if move is None:
        first, cart, gone = turn == 0, self.barrow, set()
        if order and first and cart.units:
          try:
            self.cable.hush = True
            held = list(self.barrow.units)
            gone = self.cart_stale(held)
            if gone:
              barred |= gone
              prefer = set(held) - gone
              self._log_event('code_dead', lines=sorted(gone))
          except Exception as e:
            self._log_err('cart_gone_error', e)
            gone = set()
        if first:
          rx = re.compile('(?:line )?(\\S+?):? ')
          barred |= {m.group(1) for m in map(rx.match, self.cart_misses())
               if m and m.group(1) in cart.units}
        if order and first and not gone and cart.used and self._covers()\
            and (cart.units or cart.view_counts(cart.last_view) is None)\
            and not self.cart_misses() or order and self.code_cart(barred, prefer)\
            or order and not first and self.retry_ok(barred):
          move = buy()
        # the request wants the nearest listing proposed: a filled cart is ordered even if it misses a condition
        elif order and cart.units and self.needs.parsed.get('fallback_pick'): move = buy()
        elif decline: move = drop()
        else: return
      kind = self.call_kind(move['name'], move['args'])
      try:
        self.finish_steps(kind or 'order')
      except Exception as e:
        self._log_err('finish_steps_error', e)
      if self.cable.done or self.cable.broken or self.steps_rem() <= 0: return
      resp = self.cable.doing([move])
      self._log_event('forced_finish', action=move, reply=resp)
      if not resp or resp.get('finished') or self.cable.done or turn == CODE_FINISHES - 1\
          or self.steps_rem() <= 1:
        break
      self.cable.hush = True
      held, seen = list(self.barrow.units), resp.get('observation')
      self.imbibe(move['name'], move['args'], seen, resp, True)
      self.code_news(self.buyer_lines(move['name'], seen, True))
      move = None
      if kind == 'order' and held:
        looked, gone = self.code_recheck(held)
        if looked:
          self.refused_lines.difference_update(set(held) - gone)
          rows = self.all_rows()
          cost = sum(((rows.get(ln) or {}).get('price') or 0.0) * self.barrow.held(ln) for ln in held)
          cap = self.needs.spend_cap
          over = is_num(cap) and cost > cap
          if not gone and not over: move = buy()
          else: barred, prefer = set(gone), None if over else set(held) - gone
      turn += 1
    if self.cable.done or not resp or resp.get('finished'): return
    if not decline or kind == 'decline' or self.steps_rem() <= 0: return
    try:
      retry = bool(order) and kind == 'order' and self.retry_ok(barred)
    except Exception as e:
      self._log_err('again_error', e)
      retry = False
    move = buy() if retry else drop()
    self._log_event('forced_finish', action=move, reply=self.cable.doing([move]))
  def _stuck(self, steps, seconds):
    cable = self.cable
    return cable.done or cable.broken or self.steps_rem() < steps or (cable.spare() < seconds)
  def _covers(self):
    rows = self.needs.items()
    return not rows or {self.prod_idx(ln) for ln in self.barrow.units}.issuperset(range(len(rows)))
  def retry_ok(self, avoid):
    held, closed = set(self.barrow.units), set(avoid) | self.dead_lines()
    return bool(held) and held.isdisjoint(closed) and bool(self.code_picks(closed))
  def grab_more(self):
    try:
      c = self.cable
      if c.hush or c.spare() < CLOSING: return
      notes, st, needs = [], vars(self), self.needs
      if needs.parsed.get('code_reading') and not st.get('reread')\
          and not (c.last_reply <= st.get('read_at', c.last_reply) or c.spare() <= CLOSING + 60):
        self.reread = True
        ans = c.query_model([{'role': 'user', 'content': self.read_prompt}], SKIMMER, SKIM_TERMS)
        got = as_json((ans or {}).get('content') or '') or {}
        ok = isinstance(got.get('roles'), dict)
        self._log_event('re_read', ok=ok)
        if ok:
          self.use_reads(got, True)
          notes.append("[fresh plan] The model service has recovered and read the shopper's request again; go by this reading rather than the earlier one:\n" + needs.rq_view(self.step_cap, self.msg_cap))
      q = vars(self.needs).get('pending') or []
      if q and not (c.last_reply <= vars(self.needs).get('pending_at', 0.0) or c.spare() < CLOSING):
        self.needs.pending = []
        heard = (self.needs.heed(b, a) for b, a in q[-3:])
        notes += ['[read late] lines the shopper sent earlier: ' + v for v in heard if v]
      if notes:
        body = '\n'.join([*notes, self.state_line()])
        tail = self.chat[-1] if self.chat else None
        if tail and tail.get('role') in ('user', 'tool') and isinstance(tail.get('content'), str):
          tail['content'] = tail['content'] + '\n' + body
        else: self.chat.append({'role': 'user', 'content': body})
    except Exception as e:
      self._log_err('catch_up_error', e)
  def refused(self, summon_appellation, obs, resp):
    box = isinstance(obs, dict)
    if box and 'refusal' in obs: return True
    if summon_appellation != self.role('order') or resp.get('finished'): return False
    return not (box and obs.get('error') in self._SOFT_ERRS)
  def notes_for(self, summon_appellation, obs, refused):
    tool, parts = summon_appellation, []
    if isinstance(obs, dict) and obs.get('error') == 'budget_exceeded':
      cap = self.msg_cap
      if tool == self.role('message') or cap is not None:
        self.msg_cap = self.msgs_sent if cap is None else min(cap, self.msgs_sent)
      parts.append("Nothing is left of this tool's allowance; make no further calls to it.")
    if refused:
      case = 'decline' if tool == self.role('decline') and getattr(self, 'last_kind', None) == 'decline' else 'order'
    else: case = 'notice' if isinstance(obs, dict) and len(obs) >= 2 and tool != self.role('message') else ''
    txt = getattr(self, f'_smn_note_{case}')(obs) if case else ''
    if txt: parts.append(txt)
    return ' '.join(parts)
  def _smn_note_decline(self, reply):
    self.audit_done = self.decline_vetted = False
    self.event_seen = True
    if not self._long_txts(reply):
      return "DECLINE TURNED DOWN; the errand is not over. The store maintains that a fitting item exists. Next: (a) act on whatever the refusal says; (b) go back to your nearest candidates and judge each one on its own record (every property, stock, price) against today's requirements; (c) buy the one that passes. Decline again only when every one of them fails."
    ver = str(self.needs.rev_no)
    hit = sorted(s for s in self._struck() if s + '|' + ver not in self.passed)
    named = ' Item check rejections that still stand: %s.' % ', '.join(hit[:6]) if hit else ''
    return 'DECLINE PAUSED FOR NEWS: before closing, the store handed on a message from the shopper; the errand is not over. The pause by itself tells nothing about whether any listing fits. Next: (a) act on the message; (b) re-judge only what it changes: a listing that failed on a condition the message removes or loosens, or on a price that a raised budget now covers; a listing that failed on any other condition stays out.%s (c) buy one that passes on every condition now in force; (d) if none does, send the decline again.' % named
  def _smn_note_order(self, reply):
    self.refusal_n += 1
    self.order_vetted = False
    self.event_seen = True
    hint = ''
    try:
      hint = self.refuse_redo(reply)
    except Exception as e:
      self._log_err('refusal_reread_error', e)
    self.refused_lines.update(self.barrow.units)
    self.refusal_texts = {' '.join(t.lower().split()) for t in self._long_txts(reply)}
    return 'ORDER TURNED DOWN - nothing was bought; keep working on this cart. Steps: (a) make each change it lists; (b) fix the cart, removing any line that fails now (for instance one out of stock); (c) check stock and cost of any new item again; (d) send the order again.' + hint
  @staticmethod
  def _long_txts(reply):
    found, pile = [], [(reply, 0)]
    while pile:
      m, d = pile.pop()
      if d > 6: continue
      if isinstance(m, dict): pile.extend((x, d + 1) for x in m.values())
      elif isinstance(m, list): pile.extend((x, d + 1) for x in m[:20])
      elif isinstance(m, str) and len(m.split()) >= 4: found.append(m)
    return found
  def _smn_note_notice(self, reply):
    said = self.warn_lines(reply)
    if not said: return ''
    echo = vars(self).pop('refusal_texts', None) or set()
    flat = [' '.join(str(ln).lower().split()) for ln in said]
    if echo and all(any(s in t for t in echo) for s in flat):
      return 'Just an echo: the refused finish already carried this same message, handled above; nothing here bears on this call.'
    self.event_seen = True
    return '[shopper message: a change] This result brings news to act on right away: ' + ' | '.join(said[:4])
  @staticmethod
  def shelf_for(row):
    return next((v.strip().lower() for k, v in row.items() if isinstance(v, str) and 'id' not in str(k).lower()
          and 0 < len(v.strip()) <= 60 and re.search(Errand._smn_SHELF_KEYS, str(k).lower())), '')
  @staticmethod
  def key_stock(field, value):
    lab = str(field).lower()
    if value is None or not re.search('stock|avail|on.?hand', lab): return
    sl = bool(re.search('stock|avail', lab))
    flip = sl and bool(re.search('out|sold', lab))
    nada = value is False or (is_num(value) and value <= 0)
    w = value.lower() if isinstance(value, str) else None
    return not (flip and value is True or sl and not flip and (nada or w is not None and bool(
      re.search(Errand._smn_GONE_WORDS, w))) or bool(re.search('on.?hand', lab)) and nada
          or w is not None and bool(re.search(Errand._smn_LATE_WORDS, w)))
  @staticmethod
  def stock_state(row):
    said = {Errand.key_stock(k, v) for k, v in row.items()} - {None}
    return (False not in said) if said else None
  @staticmethod
  def sold_out(row):
    return False in {Errand.key_stock(name, value) for name, value in row.items()}
  def shelf_labels(self):
    found, done, extra, repo = [], set(), self.ref_extra(), self.rec_db
    def count(r):
      i = next((v for k, v in r.items() if isinstance(v, str) and 'id' in str(k).lower() and v in repo.read),
          None)
      if i and i not in done and i not in extra and (lab := self.shelf_for(r)):
        done.add(i)
        found.append(lab)
    for it in repo.items:
      walk_dicts(it, count)
    head = {lab: found.count(lab) for lab in dict.fromkeys(found)}
    top = max(head.values(), default=0)
    freq = {lab for lab, n in head.items() if n == top}
    if not self.needs.items() or not head: return freq
    try:
      homes, chosen = self.home_cats(), set()
      for h in homes:
        seen = {lab: head[lab] for lab in h if lab in head}
        best = max(seen.values(), default=None)
        chosen.update([lab for lab, n in seen.items() if n == best] if seen else h)
      if not all(homes):
        for lab in sorted(head, key=head.__getitem__, reverse=True):
          if len(chosen) >= len(homes): break
          chosen.add(lab)
      return chosen or freq
    except Exception:
      return freq
  @staticmethod
  def row_key(row):
    ok = [(str(k).lower(), v) for k, v in row.items() if isinstance(v, (str, int)) and not isinstance(v, bool)]
    got = next((v for k, v in ok if k == 'id'), None)
    if got is None: got = next((v for k, v in ok if k.endswith('id')), None)
    return json.dumps(row, sort_keys=True, default=str) if got is None else str(got)
  def search_keys(self, parameters):
    sch = self._tschema()
    fields = sch.get('properties') if isinstance(sch, dict) else None
    if not isinstance(fields, dict): fields = {}
    cur = self.term_key()
    texts, mode, paging, firsts = [], [], set(), []
    for k, v in parameters.items():
      spec, lab = fields.get(k), str(k).lower()
      if not isinstance(spec, dict): spec = {}
      if isinstance(v, str):
        if k == cur:
          paging.add(k)
          firsts.append(not v.strip())
        elif spec.get('enum'): mode.append((k, v))
        else:
          ws = tuple(dict.fromkeys(t for t in (t.strip(Errand._smn_EDGE_PUNCT) for t in v.lower().split())
                      if re.search('[a-z0-9]', t)))
          if ws and all(re.fullmatch('[a-z0-9]{2,}', t) for t in ws): texts.append((k, ws))
      elif is_num(v) and re.search(Errand._PG_KEYS, lab)\
          and not re.search(Errand._NONPAGE_KEYS, lab):
        paging.add(k)
        low = spec.get('minimum', spec.get('default'))
        firsts.append(v <= (low if is_num(low) else 1 if 'page' in lab else 0))
    return texts, tuple(sorted(mode)), all(firsts), paging
  _seller_rx = 'shop|seller|store|vendor|brand|make|manufactur'
  _said_tag_rx = '\\((?:stated|revealed[^)]*)\\)'
  _plain_rx = '[a-z0-9]+'
  _tok2 = '[a-z0-9]{2,}'

class SearchKeys:
  @staticmethod
  def _qo_sgn(call):
    return json.dumps(call, sort_keys=True, default=str)
  @staticmethod
  def _qo_lx(text, regex='[a-z0-9]+'):
    return re.findall(regex, text.lower())
  @staticmethod
  def _qo_dub(record, regex):
    return [(k, record[k]) for k in record if re.search(regex, str(k).lower())]
  @staticmethod
  def _without(call, drop):
    return {key: call[key] for key in call if key != drop}
  @staticmethod
  def _near(a, bv, numerals=''):
    if numerals and any(c.isdigit() for c in a + bv):
      if a.lstrip('0') == bv.lstrip('0'): return True
      if numerals != 'lead' or not a.isdigit() or not bv.startswith(a): return False
      return not bv[len(a):][:1].isdigit()
    lo, hi = sorted((a, bv), key=len)
    return len(lo) >= 3 and hi.startswith(lo)
  def _qo_say(self, call):
    return ' '.join(self.arg_texts(call))
  def _props(self):
    found = self._tschema().get('properties')
    return found if isinstance(found, dict) else {}
  def _qo_rqs(self, demand_twodot):
    return [(h, t) for h, d, t in (n.partition(':') for n in self.needs.props_left())
        if (d or not demand_twodot) and not is_ref_text(t)]
  def _gaze(self, summon_prms, obs, rows):
    texts, mode, first, paging = self.search_keys(summon_prms)
    if first:
      return texts, mode, paging, max([v for v in obs.values() if isinstance(v, int) and not isinstance(v, bool)]
                      + [len(rows)])
  @staticmethod
  def _frame(summon_prms, text_key, mode, paging):
    return (text_key, mode), tuple(sorted((k, str(summon_prms[k])) for k in summon_prms
                       if k != text_key and k not in paging))
  def _pair(self, mine, mg, theirs, tg):
    (s, sg), (ln, lg) = sorted(((mine, mg), (theirs, tg)), key=lambda d: len(d[0]))
    if len(ln) > len(s) and ln[:len(s)] == s: return 'any' if lg > sg else 'all'
    lim = self.lead_words or None
    stems = lambda ws: set(map(self.lemma, ws[:lim]))
    for (w, wg), (n, ng) in (((mine, mg), (theirs, tg)), ((theirs, tg), (mine, mg))):
      if stems(w) > stems(n): return 'subset' if wg < ng else ''
    return ''
  def word_rule(self, summon_prms, obs, rows):
    look = self._gaze(summon_prms, obs, rows)
    if look is None: return self.word_kind
    texts, mode, paging, total = look
    for key, words in texts:
      sc = self._frame(summon_prms, key, mode, paging)
      taper = False
      for old in self.word_seen:
        if old[:2] != sc or old[3] == total: continue
        v = self._pair(words, total, old[2], old[3])
        if v == 'subset': taper = True
        elif v: self.word_kind = v
      if taper and not self.word_kind: self.word_kind = 'all'
      self.word_seen.append(sc + (words, total))
    return self.word_kind
  def first_rule(self, summon_prms, obs, rows):
    look = self._gaze(summon_prms, obs, rows)
    if look is None: return self.lead_words
    texts, mode, paging, total = look
    notches = tuple(self.row_key(r) for r in rows)
    mixed = len(set(notches)) >= 3
    for key, words in texts:
      sc = self._frame(summon_prms, key, mode, paging)
      cf = self.lead_bounds.setdefault(sc[0], [-1, set()])
      for old in self.lead_seen:
        if old[:2] != sc or old[2] == words: continue
        n = min(len(words), len(old[2]))
        f = next((i for i, (x, y) in enumerate(zip(words, old[2])) if x != y), n)
        if (total, notches) != old[3:]: cf[0] = max(cf[0], f)
        elif f and mixed and f == n: cf[1].add(f)
      self.lead_seen.append(sc + (words, total, notches))
    ok = [(lo + 1, sc) for sc, (lo, same) in self.lead_bounds.items() if lo + 1 in same]
    self.lead_words, self.lead_ctx = min(ok) if ok else (0, None)
    return self.lead_words
  def term_key(self, props=None):
    if props is None: props = self._props()
    return next((k for k in props if isinstance(props[k], dict) and props[k].get('type') == 'string'
          and not props[k].get('enum') and re.search(
            'bookmark|cursor|token|resume|continu|after|next|page|offset|start', str(k).lower())), None)
  @staticmethod
  def next_token(obs):
    if isinstance(obs, dict):
      return next((v for k, v in obs.items() if isinstance(v, str) and v.strip()
            and re.search('next|onward|cursor|bookmark|token|continu|more', str(k).lower())), None)
  def pg_arg(self):
    props = self._props()
    for k in props:
      spec, low = props[k], str(k).lower()
      if not (isinstance(spec, dict) and spec.get('type') in ('integer', 'number'))\
          or re.search(self._NONPAGE_KEYS, low) or not re.search(self._PG_KEYS, low):
        continue
      pages = 'page' in low and not re.search('offset|skip|first|start|from', low)
      first, top = spec.get('minimum', spec.get('default')), spec.get('maximum')
      return k, pages, first if is_num(first) else int(pages), top if is_num(top) else None
    tok = self.term_key()
    return (tok, 'token', '', None) if tok else None
  @staticmethod
  def lemma(word):
    return word if len(word) <= 3 or word[-1:] != 's' or word[-2:] == 'ss' else word[:-1]
  def _record_id(self, record):
    return next((v for k, v in record.items() if isinstance(v, str) and 'id' in str(k).lower()
          and v in self.rec_db.ids), None)
  def shelf_rows(self):
    shelves, cap, qty, seen = self.shelf_labels(), self.needs.spend_cap, self.needs.quantity(), {}
    brief = lambda v: isinstance(v, str) and 0 < len(v) <= 300
    def attend(r):
      sid = self._record_id(r)
      if not sid or sid in self.refused_lines or self.shelf_for(r) not in shelves: return
      if self.sold_out(r): return seen.pop(sid, None)
      prices = [v for _, v in self._qo_dub(r, 'price|cost') if is_num(v) and not isinstance(v, bool)]
      title = r.get('title')
      if not brief(title):
        title = next((v for k, v in self._qo_dub(r, 'title|heading|headline|name') if brief(v)
               and 'id' not in str(k).lower() and not self.shelf_for({k: v})
               and not junk_key(k)), '')
      maker = ' '.join(v for _, v in self._qo_dub(r, 'brand|make|manufactur|vendor') if isinstance(v, str))
      if prices: seen[sid] = (float(prices[0]), title, maker)
    if shelves:
      for it in self.rec_db.items:
        walk_dicts(it, attend)
    out = {}
    for sid, (old, *rest) in seen.items():
      new = self.rec_db.price(sid)
      p = old if new is None else new
      if not (is_num(cap) and p * qty > cap): out[sid] = (float(p),) + tuple(rest)
    return out
  def label_queries(self):
    out = []
    for label in sorted(self.shelf_labels()):
      groups = [[]]
      for raw in label.lower().split():
        t = raw.strip('.,;:!?\'"()[]')
        if t not in STOP_WORDS and re.fullmatch("[a-z0-9][a-z0-9'-]*", t): groups[-1].append(t)
        elif groups[-1]: groups.append([])
      groups = [g for g in groups if g]
      words, tails = [t for g in groups for t in g], [g[-1] for g in groups]
      out += map(self.lemma, tails)
      if len(words) > 1: out.append(words[0] + ' ' + words[1])
      out += tails + [t for t in words if t not in tails]
    return list(dict.fromkeys(out))
  def kind_queries(self, pairs):
    heads = {self.lemma(w) for lab in self.shelf_labels() for w in re.findall(self._plain_rx, lab)}
    best, grams, size = {}, {}, 2 if pairs else 1
    for price, title, maker in self.shelf_rows().values():
      key = title.lower().strip()
      if key:
        old = best.get(key)
        best[key] = (price if old is None else min(price, old[0]), maker.lower())
    for title, (price, maker) in best.items():
      mk, words, mine = set(re.findall('[a-z]+', maker)), re.findall('[a-z]+', title), set()
      ok = [len(w) >= 3 and w not in STOP_WORDS and w not in mk for w in words]
      for s in range(len(words) - size + 1):
        if not all(ok[s:s + size]): continue
        bit = words[s:s + size]
        stems = [self.lemma(w) for w in bit]
        key = ' '.join(stems)
        if all(x in heads for x in stems) or key in mine: continue
        mine.add(key)
        slot = grams.setdefault(key, [0, price, ' '.join(bit)])
        slot[0] += 1
        slot[1] = min(slot[1], price)
    rec = sorted((k for k in grams if grams[k][0] >= 2), key=lambda k: (grams[k][1], -grams[k][0], k))
    return [grams[k][2] for k in rec]
  def _qo_duo(self, text):
    return tuple(map(self.lemma, self._qo_lx(text, self._tok2)[:2]))
  def _shelf_tool(self):
    label = '' if self.ceil_cfg() else self.raw_label(self.shelf_labels())
    if not label: return
    taken = set(self.role_map.values())
    for tool, shape in self.argspec.items():
      props = (shape or {}).get('properties')
      if tool in taken or not isinstance(props, dict): continue
      slot = self.term_key(props)
      shelf, value = self._pick_arg(props, label, (slot,))
      bound = self._cap_arg(props, slot)
      if slot and shelf and bound: return dict(tool=tool, shelf=shelf, value=value, bound=bound, slot=slot)
    return
  def _shelf_cap(self, own, tried):
    rows = self.shelf_rows()
    if self.needs.items() or not rows: return None, False
    last, tok = vars(self).get('sweep_listed') or (None, None)
    if last and tok is None: return None, True
    cap = self.below(min(p for p, _, _ in rows.values()))
    if cap is None: return None, True
    first = {own['shelf']: own['value'], own['bound']: cap}
    if self._qo_sgn(first) not in tried: return first, False
    if tok is None: return None, True
    nxt = dict(last or first, **{own['slot']: tok})
    return (None, True) if self._qo_sgn(nxt) in tried else (nxt, False)
  def sweep_next(self, turn):
    paging, shelves, book = self.pg_arg(), self.shelf_labels(), self._book()
    if not (paging and shelves and book): return None, False
    key, start, tried, walks = paging[0], paging[2], vars(self).get('sweep_tried', set()), list(book.values())
    listing = max((w for w in walks if w.get('id_order') and w['shelves'] and w['shelves'] <= shelves
           and self.next_page(w) is not None), key=lambda w: w['total'], default=None)
    pri = [w for w in walks if w.get('id_order') is False and self.arg_texts(w['args'])]\
      or [w for w in walks if w.get('id_order') is None and self.arg_texts(w['args'])]
    tmpl = tkey = None
    if pri:
      props, args = self._props(), pri[-1]['args']
      fixed = {p: not isinstance(args[p], str) or (props.get(p) or {}).get('enum')
          or args[p].strip().lower() in shelves or p == key for p in args}
      free, lead = [p for p in args if not fixed[p]], self.lead_ctx
      tkey = lead[0] if lead and lead[0] in free else max(free, key=lambda p: len(args[p].split()), default=None)
      if tkey: tmpl = {p: args[p] for p in args if p != tkey and fixed[p]}
    done = {self._qo_duo(self._qo_say(w['args'])) for w in walks
        if start in w['read'] and w.get('id_order') is not True}
    heads = {self.lemma(t) for t in self.label_queries() if ' ' not in t}
    fresh = lambda calls: [c for c in calls if self._qo_sgn(c) not in tried]
    own = self._shelf_tool()
    call, end = self._shelf_cap(own, tried) if own else self.cap_sweep(tried)
    if call is not None: return call, True
    if end: return None, False
    cands = []
    for w in book.values():
      said = [self.lemma(t.strip().lower()) for t in self.arg_texts(w['args'])]
      if len(w['read']) != 1 or not said or said[0] not in heads or not w['shelves'] & shelves: continue
      if self.next_page(w) is not None: cands.append({**w['args'], key: self.next_page(w)})
    cands = fresh(cands)
    if cands and turn % 2 == 1: return cands[0], False
    nxt = listing and fresh([{**listing['args'], key: self.next_page(listing)}])
    if nxt: return nxt[0], True
    for o in range(3):
      name = ('pairs', 'words', 'labels')[(turn + o) % 3]
      ws = self.label_queries() if name == 'labels' else self.kind_queries(name == 'pairs')[:6]
      got = fresh([{**tmpl, tkey: t} for t in ws if (h := self._qo_duo(t)) and h not in done])\
        if tmpl is not None else []
      if got: return got[0], False
    return None, False
  def sweep_step(self):
    n = self.needs
    if n.priority() != 'cheapest' or not n.let_go or n.props_left() or not self.role('search')\
        or self.cable.done:
      return False
    reserve = 4 if self.barrow.units else 3
    if self.orders_tried == 0: reserve = max(reserve + 3, FIRST_ORDER_BY)
    margin = lambda reads: reads < SWEEP_READS and self.steps_rem() > reserve\
      and self.cable.spare() >= CLOSING + 40
    reads = vars(self).get('swept', 0)
    if not margin(reads): return False
    turn = vars(self).get('sweep_turn', 0)
    pick, lot = self.sweep_next(turn)
    if pick is None: return False
    self.swept = reads + 1
    self.sweep_turn = turn if lot else turn + 1
    vars(self).setdefault('sweep_tried', set()).add(self._qo_sgn(pick))
    own = self._shelf_tool()
    use = bool(own) and own['bound'] in pick
    call = {'id': 'sweep_%d_%s' % (self.swept, self.cable.ticks), 'type': 'function', 'function': {
      'name': own['tool'] if use else self.role('search'), 'arguments': json.dumps(pick)}}
    say = 'Only price is left to decide (every property requirement is settled): the cheapest buyable item in this category wins, so another part of it is read.'
    self.chat.append({'role': 'assistant', 'content': say, 'tool_calls': [call]})
    before = len(self.rec_db.items)
    try:
      self.covering = True
      out = self.relay([call], programmed=True)
    finally:
      self.covering = False
    if use:
      tok = None
      for b in self.rec_db.items[before:]:
        tok = next(filter(None, map(self.next_token, [b] + list(b.values()) if isinstance(b, dict) else [])),
             tok)
      self.sweep_listed = (pick, tok)
    if not (margin(self.swept) and self.sweep_next(self.sweep_turn)[0] is not None) and out:
      low = sorted((p, s, t) for s, (p, t, _) in self.shelf_rows().items())[:6]
      if low:
        listed = '; '.join("%s '%s' %s" % (s, t[:80], format(p, 'g')) for p, s, t in low)
        head, body = out[-1]
        recap = '\n[shelf sweep done] Lowest prices seen over the asked-for category, counting only rows you can buy now within budget: ' + listed + '. Plan: unless something speaks against it, the first row is the pick; it becomes the only basket line, gets its stock and price checked again, and is then ordered.'
        out[-1] = (head, body + recap)
    self.last_replies(out)
    return True
  def wide_step(self):
    st, n = vars(self), self.needs
    at = n.grown_at
    if at < 0 or n.items() or self.orders_tried or self.cable.done or self.cable.broken: return False
    if st.get('wide_at') != at:
      st['wide_at'] = at
      st['wide_todo'] = self._wide_words() if n.priority() == 'cheapest' and self.role('search') else []
      st['wide_ids'] = [] if st['wide_todo'] else None
      if st['wide_todo']: self.log.append({'kind': 'wide_plan', 'queries': list(st['wide_todo'])})
    q = st.get('wide_todo') or []
    if q and self.steps_rem() > WIDE_N + len(q) + 1 and (self.cable.spare() > CLOSING + 40):
      return self._wide_find(q.pop(0))
    if q or st.get('wide_ids') is None:
      st['wide_todo'] = []
      return False
    got, st['wide_ids'] = st['wide_ids'], None
    return self._wide_open(got)
  def _wide_words(self):
    lead = ' '.join(w for ws in [s[0] for s in self._tvals(None) if s and s[0]][:2] for w in ws)
    if not lead: return []
    heads = {self.lemma(w) for lab in self.shelf_labels() for w in re.findall(self._plain_rx, lab.lower())
        if len(w) >= 3 and w not in STOP_WORDS}
    counts = {}
    for _, title, _ in self.shelf_rows().values():
      for w in set(re.findall('[a-z]{3,}', title.lower())) - STOP_WORDS - set(lead.split()):
        f = self.lemma(w)
        if any(h in f or (len(f) >= 4 and f in h) for h in heads): counts.setdefault(f, [0, w])[0] += 1
    order = sorted(counts, key=lambda f: (f in heads, -counts[f][0], f))
    sorts = [counts[f][1] for f in order if counts[f][0] >= 2 or f in heads][:2]
    ws = ['%s %s' % (lead, kind) for kind in sorts] + [lead]
    return [t for t in dict.fromkeys(ws) if not self.used_query(t)][:WIDE_TRIES]
  def _wide_find(self, text):
    shape = self.search_shape()
    if not shape:
      vars(self)['wide_todo'] = []
      return False
    base, key = shape
    args, faults = ArgCheck.vet({**base, key: text}, self._search_form())
    if faults or not isinstance(args, dict): return False
    home, start, hb = self.walk_home(None), len(self.rec_db.items), self.cable.ticks
    say = "One requirement was dropped, so listings the earlier searches skipped could fit now: searching '%s' with the dropped words removed." % text
    self.walk_call(args, say)
    got = vars(self).setdefault('wide_ids', [])
    def attend(d):
      if self._is_row(d) and (not home or self.shelf_for(d) in home):
        got.append(self.row_key(d))
    for b in self.rec_db.items[start:]:
      walk_dicts(b, attend, ceiling=3)
    self.log.append({'kind': 'wide_search', 'query': text, 'rows': len(got)})
    return self.cable.ticks != hb
  def _wide_open(self, found):
    units = self.barrow.units
    prices, below = [self.rec_db.price(ln) for ln in units], None
    if units:
      below = min([float(p) for p in prices if is_num(p)], default=None)
      if below is None: return False
    pri = [i for _, i in self.cheap_unread(list(dict.fromkeys(found)), None, below)]
    q = [i for i in pri if i not in self.barrow.units]
    q = q[:max(0, min(WIDE_OPENS, self.steps_rem() - WIDE_N))]
    self.log.append({'kind': 'wide_open', 'below': below, 'rows': pri[:6], 'opened': q})
    if not q: return False
    self.__dict__.setdefault('named', set()).update(q)
    self.rank_checked = set()
    self.order_vetted, self.open_queue, self.open_recheck, self.open_purpose = False, q, True, 'wide'
    self.open_say = 'Reading the cheaper rows that the searches without the dropped requirement turned up: among listings meeting every condition still in force the lowest price wins, so one of them may take over the cart line before the basket message.'
    sent = self.open_named()
    held = ', '.join(sorted(self.barrow.units))
    if sent and held:
      self._append('[after the drop] Opened, each priced below the cart line %s: %s. Judge each one from its record against the conditions in force now only (the dropped one does not count any more). One that meets them all - the cheapest such one - replaces the cart line BEFORE any basket message, so the basket you tell the shopper is the one you order.' % (held, self._rated(q)), True)
    return sent
  def arg_texts(self, summon_prms):
    props, tok = self._props(), self.term_key()
    return [v for k, v in summon_prms.items()
        if isinstance(v, str) and k != tok and not (props.get(k) or {}).get('enum')]
  def log_page(self, summon_prms, rows, total, obs=None):
    paging = self.pg_arg()
    if not paging or not rows: return
    key, manner, start = paging[:3]
    at = summon_prms.get(key, start)
    if manner == 'token': at = at.strip() if isinstance(at, str) else ''
    elif not is_num(at): return
    rest = self._without(summon_prms, key)
    fresh = dict(args=rest, read=set(), size=0, total=0, shelves=set())
    walk = vars(self).setdefault('page_book', {}).setdefault(self._qo_sgn(rest), fresh)
    walk['read'].add(at)
    if manner == 'token': walk.setdefault('tokens', {})[at] = self.next_token(obs)
    recs = [r for r in rows if isinstance(r, dict)]
    walk.update(size=max(walk['size'], len(rows)), total=max(walk['total'], total))
    walk['shelves'] |= set(map(self.shelf_for, recs)) - {''}
    per = walk.setdefault('row_shelf', {})
    for r in recs:
      per.setdefault(self.row_key(r), self.shelf_for(r))
    ids = [self.row_key(r) for r in recs]
    if at == start:
      walk['first'] = tuple(ids)
      if total <= len(rows): walk['every'] = recs
    if total > len(rows): walk['full'] = max(walk.get('full', 0), len(rows))
    if len(ids) >= min(5, max(4, walk.get('full', 0) or 5)):
      walk['id_order'] = walk.get('id_order', True) is not False and (ids == sorted(ids) or ids == sorted(
        ids, key=lambda s: [int(x) if x.isdigit() else x for x in re.split('(\\d+)', s)]))
  def guess_pairs(self, summon_prms, rows):
    if self.word_kind == 'any': return
    asked = {w for t in self.arg_texts(summon_prms) for w in re.findall('[a-z]{4,}', t.lower())}
    ev = vars(self).setdefault('twin_rows', {})
    for row in filter(lambda r: isinstance(r, dict), rows or []):
      strands = []
      all_strs(row, strands)
      lex = set(re.findall('[a-z]+', ' '.join(strands).lower()))
      for word in asked:
        if any(w == word or same_kind(w, word) for w in lex): continue
        for d in (w for w in lex if spell_pair(word, w)):
          seen = ev.setdefault((word, d), set())
          seen.add(self.row_key(row))
          if len(seen) >= 2:
            EMPORIUM_DOUBLES.setdefault(word, set()).add(d)
            EMPORIUM_DOUBLES.setdefault(d, set()).add(word)
  def pair_query(self, k):
    shape = self.search_shape()
    if not shape or not EMPORIUM_DOUBLES: return
    n = self.needs
    joined = ' '.join(r.partition(':')[2].lower() for r in n.props_left()
             if not n.items() or self.prod_of(r.partition(':')[0]) in (k, None))
    values, home, best = set(self._alph3.findall(joined)), self.walk_home(k), None
    for e in self._book().values():
      if home and not set((e.get('row_shelf') or {}).values()) & home: continue
      words = self._qo_lx(' '.join(self.arg_texts(e['args'])))
      for pos, w in enumerate(words):
        if w not in values: continue
        for d in sorted(EMPORIUM_DOUBLES.get(w, ())):
          t = ' '.join(words[:pos] + [d] + words[pos + 1:])
          if not self.used_query(t) and (best is None or len(t) < len(best)): best = t
    return None if best is None else {**shape[0], shape[1]: best}
  def next_page(self, walk):
    _, manner, start, top = self.pg_arg()
    read = walk['read']
    if manner == 'token':
      return next((t for t in (walk.get('tokens') or {}).values() if isinstance(t, str) and t not in read), None)
    low, size = min([start] + [v for v in read if is_num(v)]), max(1, walk['size'])
    for n in range(1, 400):
      at = low + n * (1 if manner else size)
      if n * size >= walk['total'] or top is not None and at > top: break
      if at not in read: return at
    return
  def val_score(self, summon_prms):
    sought = [self._qo_lx(v, '[a-z0-9]{3,}|\\b[0-9][a-z]\\b|\\b[a-z][0-9]\\b') for _, v in self._qo_rqs(False)]
    lead = self._qo_lx(self._qo_say(summon_prms), self._tok2)
    if self.lead_words: lead = lead[:self.lead_words]
    return sum(1 for s in sought if any(w == v or self._near(w, v) for v in s for w in lead))
  def row_hits(self, rows, words=None):
    cap, qty, shelves = self.needs.spend_cap, self.needs.quantity(), self.shelf_labels()
    refd = self.needs.parsed.get('references')
    asked = set(self._qo_lx(' '.join(str(v) for v in words or []))) if refd else set()
    refs, hits = vars(self).setdefault('ref_hits', []), []
    for row in filter(lambda r: isinstance(r, dict), rows):
      sid = self.row_key(row)
      if sid not in self.rec_db.ids or sid in self.rec_db.read: continue
      if asked and any(isinstance(v, str) and v.strip() and 'id' not in str(k).lower()
              and (s := set(self._qo_lx(v))) and s <= asked and s <= self.cited_words()
              for k, v in self._qo_dub(row, self._seller_rx)) and sid not in refs:
        refs.append(sid)
      prices = [v for _, v in self._qo_dub(row, 'price|cost') if is_num(v)]
      price, shelf = prices[0] if prices else None, self.shelf_for(row)
      if is_num(cap) and is_num(price) and price * qty > cap or self.sold_out(row)\
          or shelves and shelf and shelf not in shelves:
        continue
      hits.append(sid)
      vars(self).setdefault('hit_shelf', {})[sid] = shelf
    return hits
  def cited_words(self):
    size, memo = len(self.rec_db.items), vars(self).get('pointed_memo')
    if memo and memo[0] == size: return memo[1]
    found = set()
    try:
      for h in self.needs.parsed.get('references') or []:
        found.update(self._qo_lx(str(h)))
      def attend(r):
        if self.row_key(r) not in self.rec_db.read: return
        sel, texts = set(), []
        for k, v in r.items():
          low = str(k).lower()
          if not isinstance(v, str): continue
          if re.search(self._seller_rx, low) and 'id' not in low and len(v) <= 40\
              and not re.search('note|text|desc|comment|detail|info|message|section', low):
            sel.update(self._qo_lx(v))
          elif not re.search('title|heading|headline|name|label|caption', low): texts.append(v.lower())
        for t in texts:
          found.update(set(self._qo_lx(t)) - sel)
      for b in self.rec_db.items:
        walk_dicts(b, attend)
    except Exception:
      pass
    self.pointed_memo = (size, found)
    return found
  def open_pages(self, utmost=2):
    paging = self.pg_arg()
    if not paging: return []
    shelves, pri = self.shelf_labels(), []
    for w in self._book().values():
      at = self.next_page(w)
      if at is None or shelves and w['shelves'] and not w['shelves'] & shelves: continue
      mass = self.val_score(w['args'])
      if mass > 0: pri.append(((-mass, w['total'], len(pri)), {**w['args'], paging[0]: at}))
    pri.sort(key=lambda x: x[0])
    return [c for _, c in pri[:utmost]]
  def _ohead(self, text, arrive):
    return [w for w in self._qo_lx(text) if len(w) > 1][:arrive]
  def val_scan(self, raw):
    if self.word_kind == 'any': return ('', '')
    arrive = self.lead_words or 2
    lead = self._ohead(' '.join(raw), arrive)
    product = ' '.join(self._qo_lx(str(self.needs.parsed.get('product') or ''))[:3])
    rqs = [(n.strip(), re.split('[,;]', re.sub(self._said_tag_rx, '', v))[0].strip().lower())
       for n, v in self._qo_rqs(True)]
    brushes = lambda st: any(self._near(w, k, 'lead') for k in st for w in lead)
    if not lead or brushes([w for _, v in rqs for w in re.findall(self._tok2, v)] + product.split()):
      return ('', '')
    hit = next(((n, v) for n, v in rqs if v and brushes(self._qo_lx(n, self._tok2))), None)
    if hit is None: return ('', '')
    name, value = hit
    m = re.search('\\d', value)
    scout = ((value[m.start():] if m else value).strip(' .;') + ' ' + product).strip()
    head = self._ohead(scout, arrive)
    if any(len(r) == len(head) and all(self._near(a, b, 'lead') for a, b in zip(head, r))
       for r in (self._ohead(self._qo_say(w['args']), arrive)
            for w in self._book().values())):
      return ('', '')
    return (name, scout)
  def used_query(self, text):
    g = self.lead_words or 3
    key = lambda t: sorted(map(self.lemma, self._qo_lx(t)[:g]))
    target = key(str(text))
    return any(key(self._qo_say(w['args'])) == target for w in self._book().values())
  def search_shape(self):
    for w in reversed(list(self._book().values())):
      call = w['args']
      if self.zg_lot(call): continue
      free = self.arg_texts(call)
      slots = [k for k in call if isinstance(call[k], str) and call[k] in free and call[k].strip()]
      if len(free) == 1 and len(slots) == 1: return (self._without(call, slots[0]), slots[0])
    return
  def word_trap(self, raw):
    try:
      if self.word_kind == 'any': return ('', '')
      toks = (m.strip('"\'()[]{},.;!?|').lower() for m in ' '.join(raw).split())
      toks = [t for t in toks if re.search('[a-z0-9]', t)]
      girth = self.lead_words or (max(3, len(toks)) if self.word_kind == 'all' else 3)
      toks = toks[:girth]
      near = lambda cr, st: any(self._near(b, w, 'eq') for b in cr for w in st)
      crumbs = lambda t: [s for s in re.findall(self._plain_rx, t) if len(s) > 1]
      stash = lambda ws, rx: {w for t in ws for w in self._qo_lx(t, rx)} - STOP_WORDS
      rqs = [(re.split('\\s*\\(', n)[0].lower(), n, re.sub(self._said_tag_rx, '', v).lower())
         for n, v in self._qo_rqs(True)]
      vals = {w for r in rqs for w in re.findall(self._tok2, r[2])}
      products = [ln['product'] for ln in self.needs.items()] or [str(self.needs.parsed.get('product') or '')]
      sorts = stash([' '.join(products + sorted(self.shelf_labels()))], '[a-z0-9]{3,}')
      pools = (('', vals | sorts), ('name', stash([r[0] for r in rqs], '[a-z]{3,}')),
          ('dropped', stash([o.partition(':')[2] for o in self.needs.let_go], '[a-z]{3,}')))
      hit, why = [], ''
      for t in toks:
        cr = crumbs(t)
        v = next((tag for tag, st in pools if near(cr, st)), '')
        if v:
          hit.append(t)
          why = why or v
      kept = [t for t in toks if t not in hit]
      kc = [b for t in kept for b in crumbs(t)]
      held = [r[1] for r in rqs if near(kc, re.findall(self._tok2, r[2]))]
      if not hit or not held: return ('', '')
      if not near(kc, sorts):
        at = self.prod_of(held[0]) if self.needs.items() else None
        named = products[at] if at is not None and at < len(products) else products[0]
        kept += [w for w in self._qo_lx(named, '[a-z]{3,}') if w not in STOP_WORDS][-1:]
      query = ' '.join(kept[:girth])
      if not query or self.used_query(query): return ('', '')
      word = hit[0]
      if why == 'name':
        lead = f"This store needs every search word to match, and '{word}' is part of a property's NAME, not of any required value: only listings that write that name out in a field or the title come back, so a listing showing just the value there is missed."
      else:
        lead = f"'{word}' stems only from a requirement the shopper has given up, yet the search still demands it of every listing, which hides items that are acceptable now."
      return (lead + f" Also try '{query}' next, reading every page.", query)
    except Exception as e:
      self._log_err('trap_error', e)
      return ('', '')
  _said_rx = re.compile('\\((?:stated|revealed[^)]*)\\)')
  _odd_val_rx = re.compile('\\d|as[ ]shown|same[ ]as|page|redirect|refer|\\bnot\\b|\\bno\\b|without|except|anything|other[ ]than')
  _first_alt_rx = re.compile('[,;/]|\\bor\\b|\\band\\b')
  _alt_rx = re.compile('[,;/]|\\bor\\b|\\beither\\b|\\bnor\\b')
  _ptr_rx = re.compile('as[ ]shown|same[ ]as|the[ ]page|sold[ ]by|redirect|refer')
  _alnum_rx = re.compile('[a-z0-9]+')
  _tok2_rx = re.compile('[a-z0-9]{2,}')
  _tok3_rx = re.compile('[a-z0-9]{3,}')
  _alph_rx = re.compile('[a-z]+')
  _alph2 = re.compile('[a-z]{2,}')
  _alph3 = re.compile('[a-z]{3,}')
  _code_rx = re.compile('\\b(?:[0-9][a-z]|[a-z][0-9])\\b')
  _stock_rx = re.compile('stock|avail')
  _brand_rx = re.compile('brand|make|manufactur')
  _cost_rx = re.compile('price|cost')
  _notice_rx = re.compile('shopper|Meaning:|REFUSED')
  _rows_head = 'Page rows (id, price, title, category): '

class WalkStore:
  def _steer(self, steps, state, assignment):
    while state is not None:
      state = steps[state](assignment)
    return assignment
  def _getv(self, name, default=None):
    return self.__dict__.get(name, default)
  def _book(self):
    return self._getv('page_book', {})
  def _removed(self):
    rev = str(self.needs.rev_no)
    return {s for s, _, v in (e.rpartition('|') for e in self.passed) if v == rev}
  def _struck(self):
    rev = [str(self.needs.rev_no)]
    return {f[0] for f in (m.split('|') for m, v in self.rulings.items() if v) if f[1:2] == rev}
  def _asked_prod(self):
    return str(self.needs.parsed.get('product') or '')
  def _shelves_of(self, ids):
    return {self.known_shelf(symbol) for symbol in ids} - {''}
  def _stock_keys(self, keep):
    out = {}
    def note(d):
      if keep(d): out[self.row_key(d)] = self.sold_out(d)
    for b in self.rec_db.items:
      walk_dicts(b, note)
    return out
  def _brief(self, headroom, broken=False):
    return bool(self.steps_rem() <= headroom() or self.cable.done or broken and self.cable.broken
          or self.cable.spare() < CLOSING + 30)
  def combo_qs(self):
    shape = self.search_shape()
    if not shape: return []
    prods, room = self.needs.items(), self.lead_words or 3
    have = self._shelves_of(self._removed() - self.refused_lines)
    if prods:
      homes = self.home_cats()
      todo = [n for n in range(len(prods)) if n < len(homes) and homes[n] and not homes[n] & have]
    else: todo = [None] if self.alt_unseen() else []
    (base, key), calls = shape, []
    for n in todo:
      groups = []
      for req in self.needs.props_left():
        name, _, said = req.partition(':')
        if prods and self.prod_of(name) not in (n, None): continue
        bare = self._said_rx.sub('', said).strip().lower()
        if not bare or self._odd_val_rx.search(bare): continue
        ws = self._alph2.findall(self._first_alt_rx.split(bare)[0])
        ws = [w for w in ws if w not in STOP_WORDS]
        if ws and len(ws) <= 2: groups.append(ws)
      if len(groups) < 2: continue
      label, words = self._asked_prod() if n is None else prods[n]['product'], []
      for g in groups:
        if len(words) + len(g) <= room: words += g
      noms = [w for w in self._alph3.findall(label.lower()) if w not in STOP_WORDS]
      if noms and len(words) < room and noms[-1] not in words: words.append(noms[-1])
      text = ' '.join(words)
      if len(words) > 1 and not self.used_query(text): calls.append({**base, key: text})
    return calls[:2]
  def umpire(self, widget, edict):
    try:
      reason = next((str(e['result'].get('reason') or '') for e in reversed(self.log) if isinstance(e, dict)
             and e.get('kind') == 'audit' and isinstance(e.get('result'), dict)), '')
      since_alt = f' ({reason[:400]})' if reason else ''
      second = '\n\nSECOND OPINION NEEDED\n- Audit verdict: the audit run before a decline found this item compliant' + since_alt + '.\n- Earlier check verdict: ' + edict[:500] + '\n- How to settle: decide from the records only. A field meets a requirement when it states the requested value in other words, or as a shade, qualifier or variant spelling of that same value; it fails when it states a different value.'
      again = self.check_item(widget, 'ORDER', second=second)
      if again: return again
      preamble = f'{widget}|{self.needs.rev_no}'
      gone = [m for m in self.rulings if m == preamble or m.startswith(preamble + '|')]
      self.rulings.update(dict.fromkeys(gone, ''))
      return ''
    except Exception as e:
      self._log_err('arbitrate_error', e)
      return edict
  def undisputed(self):
    try:
      prods = self.needs.items()
      shelves = self.home_cats() if prods else [set(self.shelf_labels())]
      if not shelves or not all(shelves): return []
      barred, verified = self._struck(), self._removed()
      cat, reads = self._getv('audit_ids'), self.rec_db.read
      whole = self._getv('audit_done') and (cat is None or len(reads) <= self._getv('audit_reads', -1))
      barred |= (set(reads) if whole else set(cat or ())).difference(self._getv('audit_fits') or [])
      gone = self._stock_keys(lambda d: self.row_key(d) in self.rec_db.ids and any(
        self._stock_rx.search(str(f).lower()) for f in d))
      limit = self.needs.spend_cap
      barred |= self.refused_lines | self.ref_ptrs()
      chosen, spent, capped = [], 0.0, is_num(limit)
      for n, shelf in enumerate(shelves):
        units = prods[n]['quantity'] if prods else self.needs.quantity()
        opts = []
        for s in sorted(self.rec_db.read):
          if s in barred or gone.get(s) or s in chosen or self.known_shelf(s) not in shelf: continue
          p = self.rec_db.price(s)
          if is_num(p) and not (capped and p * units > limit): opts.append((s not in verified, p, s))
        if not opts: return []
        _, p, s = min(opts)
        chosen.append(s)
        spent += p * units
      return [] if capped and spent > limit else chosen
    except Exception as e:
      self._log_err('unrefuted_error', e)
      return []
  def fit_keep(self):
    try:
      steps = {'gate': self._fit_gate, 'check': self._fit_check,
          'cover': self._fit_cover, 'word': self._fit_word}
      return self._steer(steps, 'gate', {'good': [], 'note': ''})['note']
    except Exception as e:
      self._log_err('fit_hold_error', e)
      return ''
  def _fit_gate(self, assignment):
    if self._getv('fit_held') or self.barrow.units or self.steps_rem() < 5\
        or self.cable.spare() < CLOSING + 30:
      return
    known = self.rec_db.ids
    todo = [s for s in self._getv('audit_fits') or [] if s in known and s not in self.refused_lines]
    assignment['todo'] = todo[:2]
    if not assignment['todo']: return
    self.fit_held = True
    return 'check'
  def _fit_check(self, a):
    if not a['todo']: return 'cover'
    w = a['todo'].pop(0)
    if w not in self.rec_db.read: return 'check'
    v = self.check_item(w, 'ORDER')
    if v and CLOSING + 30 < self.cable.spare(): v = self.umpire(w, v)
    if v or w not in self._removed(): return 'check'
    a['good'].append(w)
    return 'check' if self.needs.items() else 'cover'
  def _fit_cover(self, a):
    if not a['good']: return
    if self.needs.items():
      sn = self._shelves_of(self._removed() - self.refused_lines)
      if any(s and not s & sn for s in self.home_cats()): return
    return 'word'
  def _fit_word(self, a):
    g = a['good']
    n = 'Decline on hold: the audit run before a decline flagged ' + ', '.join(g) + (' as a candidate; an item check of its full record then passed it on' if len(g) == 1 else ' as candidates; item checks of their full records then passed both on') + ' every rule in force now. Think again. If it really qualifies, put ' + ('a passed line for every product' if self.needs.items() else g[0]) + ' into the cart, refresh its stock and price, then send the order; save the decline for when some detail shows it breaking a rule.'
    a['note'] = n
  def _flags(self):
    return [w.get('id_order') for w in self._book().values() if 'id_order' in w]
  def id_store(self):
    f = self._flags()
    return False not in f and f.count(True) > 1
  def top_store(self):
    f = self._flags()
    return True not in f and f.count(False) > 1
  def lot_of(self, e):
    o = e.get('id_order')
    if o is True or o is False and self.top_store(): return True
    if o is not False or not self.arg_texts(e.get('args') or {}): return False
    return bool(set().union(*self.home_cats()) & set(e.get('shelves') or ()))
  def prod_idx(self, w):
    if not self.needs.items(): return
    s, ab = self.known_shelf(w), self.home_cats()
    return next((n for n, h in enumerate(ab) if s in h), None) if s else None
  def walk_home(self, k):
    ab, i = self.home_cats(), k or 0
    own = set(ab[i]) if ab and i < len(ab) else set()
    return set(self.shelf_labels()) if not own and k is None else own
  @staticmethod
  def fig_qty(text):
    lw = str(text).lower()
    m = None if re.search(BOUND_RX, lw) else re.match('\\s*(\\d+)\\s+([a-z]+)', lw)
    return m.groups() if m else None
  @staticmethod
  def qty_part(text):
    lw = str(text).lower()
    if not any(c.isdecimal() for c in lw): return False
    return re.search(BOUND_RX, lw) is not None or re.match('\\s*\\d+(?:[.,]\\d+)?\\s+[a-z]', lw) is not None
  @staticmethod
  def _close(a, bv):
    return min(len(a), len(bv)) > 2 and (a[:len(bv)] == bv or bv[:len(a)] == a)
  def _reqs(self, k):
    only = bool(self.needs.items()) and k is not None
    for n, r in enumerate(self.needs.props_left()):
      nm, _, v = r.partition(':')
      if not (only and self.prod_of(nm) not in (k, None)): yield n, v
  def walk_has(self, args, k, subs=False):
    lw, sp, nb = [t.lower() for t in self.arg_texts(args)], self.lead_words or None, self._close
    ws = [w for t in lw for w in self._tok2_rx.findall(t)][:sp]
    tk, out = [w for t in lw for w in self._alnum_rx.findall(t)][:sp], set()
    for n, v in self._reqs(k):
      for p, m in enumerate(self._alt_rx.split(self._said_rx.sub('', v).lower())):
        if u := self.fig_qty(m): ok = u[0] in tk and any(t == u[1] or nb(t, u[1]) for t in tk)
        else: ok = not self.qty_part(m) and (any(c in ws for c in self._code_rx.findall(m)) or any(
          nb(w, x) for w in ws for x in self._tok3_rx.findall(m)))
        if ok: out.add((n, p) if subs else n)
    return out
  def walk_score(self, args, k):
    return len(self.walk_has(args, k))
  def _first(self, e):
    f = self._alnum_rx.findall(' '.join(self.arg_texts(e['args'])).lower())
    return set(map(self.lemma, f[:self.lead_words or None]))
  def walk_opts(self, k, deep=False):
    if not (sh := self.walk_home(k)): return []
    lm, fa, van, bs, zl = self.lemma, self._tok2_rx.findall, self._first, self.walk_has, self.enumsig
    wide = {lm(w) for s in sh for w in fa(s)} | {lm(w) for _, v in self._reqs(k) for w in fa(v.lower())}
    book, lot, up, atop = self._book(), self.lot_of, self.next_page, self._above_shelf
    done = [(van(e), bs(e['args'], k, subs=True), zl(e['args'])) for e in book.values() if lot(e) and e.get('first')
        and up(e) is None and van(e) and len(e.get('row_shelf') or {}) >= e['total']]
    bare = self._only_bounds(k)
    deep, best = deep or bare, {}
    for nm, e in book.items():
      if not (lot(e) and e.get('first') and up(e) is not None): continue
      if not (bs(e['args'], k) or bare and atop(e, sh)): continue
      if not any(s in sh for s in (e.get('row_shelf') or {}).values()): continue
      mi, pr, md = van(e), bs(e['args'], k, subs=True), zl(e['args'])
      if any(st == md and (ld <= mi or not bare and ld <= wide and pr <= g) for ld, g, st in done): continue
      z, p = max(1, e['size']), self.pg_arg()
      lf = -(-e['total'] // z)
      if p and p[1] != 'token' and is_num(p[3]) and is_num(p[2]):
        lf = min(lf, int((p[3] - p[2]) // (1 if p[1] else z)) + 1)
      lf, d = lf - len(e['read']), (e['total'], e['first'])
      ok = 0 < lf <= 2 * WALK_PAGES or lf > 0 and (0 < e.get('size', 0) < 5 or deep and atop(e, sh))
      if ok and (d not in best or lf < best[d][0]): best[d] = (lf, nm, e)
    op = list(best.values())
    if self.all_words():
      def emb(p, o):
        i, x = van(p[2]), van(o[2])
        return i < x and not x - i <= wide and p[0] <= WALK_PAGES or x < i and i - x <= wide
      op = [o for o in op if not any(emb(p, o) for p in op if p is not o)]
    return [o for *_, o in sorted((o[0], -self.walk_score(o[2]['args'], k), o[1], o) for o in op)]
  @staticmethod
  def _above_shelf(entry, shelf):
    rows = entry.get('row_shelf') or {}
    return len(rows) >= max(1, entry.get('size', 0)) and all(s in shelf for s in rows.values())
  def _only_bounds(self, k):
    if k is None or not self.needs.items(): return False
    rs = [v for r in self.needs.props_left() for n, _, v in [r.partition(':')]
       if self.prod_of(n) == k]
    return bool(rs) and all(re.search(BOUND_RX, self._said_rx.sub('', t).lower()) and self.qty_part(t)
                for t in rs)
  def walk_plan(self, budget, merely=None, rule='cheapest'):
    if not budget > 0 or self.needs.priority() != rule or not self.role('search') or not self.pg_arg(): return []
    f, tp = self._flags(), self.walk_opts
    if not (self.id_store() or self.top_store() or True in f and False in f): return []
    wk, solo, noms, plan, left = self._getv('walked', set()), not self.needs.items(), [], [], budget
    for n in range(len(self.needs.items())) or [None]:
      if n in wk or merely is not None and n not in merely: continue
      if b := next(iter(tp(n, deep=solo)), None): noms.append((b[0], -1 if n is None else n, n, b[1]))
    noms.sort(key=lambda c: c[:2])
    def add(n, nm, pg):
      nonlocal left
      if (t := min(pg, WALK_PAGES, left)) > 0: plan.append((n, nm, t)); left -= t
      return t > 0
    for pg, _, n, nm in noms:
      if not add(n, nm, pg): break
    for pg, nm, _ in tp(None, deep=True)[1:] if solo and plan else ():
      if not add(None, nm, pg): break
    return plan
  def walk_pick(self, k):
    dd, rel = self.dead_lines(), lambda s: k is None or self.prod_idx(s) == k
    st = [s for s in self.barrow.units if rel(s) and s not in dd] or [
      s for s in self._removed() if s not in self.refused_lines and s not in dd and rel(s)]
    b = min(((float(p), s) for s in st for p in [self.rec_db.price(s)] if is_num(p)), default=None)
    return (b[1], b[0]) if b else None
  def line_query(self, k):
    pk, sp, fd = self.walk_pick(k), self.search_shape(), []
    if not pk or not sp: return
    cc = self.title_of
    ad = lambda d: fd.append(d) if self.row_key(d) == pk[0] and cc(d) else None
    for it in self.rec_db.items: walk_dicts(it, ad)
    rc = fd[-1] if fd else {}
    mk = next((v for f, v in rc.items() if isinstance(v, str) and v.strip()
         and self._brand_rx.search(str(f).lower()) and 'id' not in str(f).lower()), '')
    if not (lx := self._alnum_rx.findall(mk.lower())): return self.title_line(k, rc, sp)
    ws = [*self.walk_home(k), *(r.lower() for r in self.needs.props_left()),
       *(r['product'].lower() for r in self.needs.items() or [{'product': self._asked_prod()}])]
    stop, tm = STOP_WORDS | set(lx) | {w for t in ws for w in self._alph_rx.findall(t)}, lx[0]
    ti = re.findall("[a-z][a-z'-]{3,}", cc(rc).lower())
    cl = next((w for w in (ti[ti.index(tm) + 1:] if tm in ti else ti)
         if "'" not in w and w.strip("'-") not in stop), None)
    if cl is None or self.used_query(t := tm + ' ' + cl): return
    return {**sp[0], sp[1]: t}
  def walk_rows(self, k, keys, utmost, skim_hurdle=True):
    sh, book, rp, sn = self.walk_home(k), self._book(), self.rec_db, set()
    bn = {n: len(self.walk_has(e['args'], k)) for n, e in book.items() if self.lot_of(e)}
    top = max(bn.values(), default=0)
    bare = not top and self._only_bounds(k)
    for nm, e in book.items():
      if nm in keys or (h := bn.get(nm, 0)) and h == top or bare and nm in bn and self._above_shelf(e, sh):
        sn.update(s for s, x in (e.get('row_shelf') or {}).items() if x in sh)
    if not skim_hurdle: hd = (self.walk_pick(k) or (None, None))[1]
    elif pk := self.walk_pick(k): hd = pk[1]
    else: hd = min((float(rp.price(s)) for s in sn if s in rp.read and is_num(rp.price(s))), default=None)
    pr, lim = self.needs.items(), self.needs.spend_cap
    u = self.needs.quantity() if k is None or k >= len(pr) else pr[k]['quantity']
    gone = self._stock_keys(lambda d: self._one_name(d, sn))
    return [s for _, s in sorted((p, s) for s in sn for p in [rp.price(s)] if not (
      s in rp.read or s in self.refused_lines or gone.get(s) or not is_num(p)
      or is_num(lim) and p * u > lim or hd is not None and p >= hd))[:utmost]]
  @staticmethod
  def _one_name(dx, stash):
    return any('id' in str(f).lower() for f, v in dx.items() if isinstance(v, str) and v in stash)
  def _is_row(self, dx):
    return len(dx) > 3 and self._one_name(dx, self.rec_db.ids)
  def page_brief(self, obs):
    rows = []
    walk_dicts(obs, lambda d: rows.append(d) if self._is_row(d) else None, ceiling=3)
    def cell(r):
      p = next((v for f, v in r.items() if is_num(v) and self._cost_rx.search(str(f).lower())), None)
      t = f"{self.row_key(r)} {'?' if p is None else p}{' SOLD OUT' if self.sold_out(r) else ''} '{self.title_of(r)[:70]}' [{self.shelf_for(r)}]"
      return t
    return self._rows_head + ('; '.join(cell(r) for r in rows[:12]) or 'none')
  def walk_call(self, args, say, tool=None):
    ch, rp = self.chat, self.rec_db
    call = dict(id=f'walk_{len(ch)}_{self.cable.ticks}', type='function',
          function=dict(name=tool or self.role('search'), arguments=json.dumps(args)))
    ch.append(dict(role='assistant', content=say, tool_calls=[call]))
    n, self.covering = len(rp.items), True
    try: rs = self.relay([call], programmed=True)
    finally: self.covering = False
    sid, t = rs[0] if rs else (call['id'], 'Not run.')
    if gr := bool(rs) and n < len(rp.items):
      nw, t = t.partition('\n[note] ')[2], self.page_brief(rp.items[-1])
      if nw and self._notice_rx.search(nw): t += '\n[note] ' + nw
    ch.append({'role': 'tool', 'tool_call_id': sid, 'content': t})
    return gr
  def try_walk(self, plan, floor, ceil=None, repair=False, twins=None, rungs=None):
    pg = self.pg_arg()
    if not (plan or ceil or twins or rungs) or not pg or not self.role('search'): return False
    ky, sm, up, rg, te = pg[0], self.walk_call, self.next_page, self._book, self.walk_rows
    a = {'plan': plan, 'floor': floor, 'key': ky, 'book': rg(),
      'walked': self.__dict__.setdefault('walked', set()), 'report': [], 'ran': False}
    rep, wd = a['report'], lambda s: ' '.join(self.arg_texts(s))
    sig = lambda s: self._qo_sgn({p: v for p, v in s.items() if p != ky})
    br = lambda: self._brief(lambda: floor + 2)
    n_, rk, tk, li, cr = 0, {}, {}, {}, {}
    for n, ss in dict(rungs or {}).items():
      ks = rk.setdefault(n, [])
      for s in ss:
        if n_ >= RUNG_CALLS or br(): break
        w, n_ = wd(s), n_ + 1
        say = f"This product is limited by a figure alone, so I search the whole figures just beyond that limit: '{w}'."
        if not sm(s, say): continue
        a['ran'] = True; ks.append(sig(s)); e = rg().get(ks[-1])
        while e and n_ < RUNG_CALLS and len(e['read']) < RUNG_PAGES and up(e) is not None and not br():
          n_ += 1
          if not sm({**e['args'], ky: up(e)}, 'Reading this figure search through to its last page.'): break
        rep.append(f"the figure search '{w}' ({len((e or {}).get('row_shelf') or {})} rows seen)")
    for n, s in dict(twins or {}).items():
      if br(): break
      if sm(s, f"This store writes a required value in two ways, so '{(w := wd(s))}' gets searched as well."):
        a['ran'], tk[n] = True, sig(s); rep.append(f"the other spelling '{w}'")
    self._walk_pages(a)
    for n in (env := list(dict.fromkeys(n for n, _, _ in plan))):
      if br(): break
      s = self.line_query(n)
      say = "Next, a search for the pick's own product line (maker plus family name): sibling sizes or colours are listed separately and may well be cheaper."
      if not s or not sm(s, say): continue
      a['ran'], li[n] = True, sig(s)
      ln = rg().get(li[n])
      for _ in range(2):
        if not ln or ln['total'] > 3 * max(1, ln['size']) or up(ln) is None: break
        if self.steps_rem() <= floor + 3 or self.cable.done: break
        if not self.cable.spare() > CLOSING + 30: break
        if not sm({**ln['args'], ky: up(ln)}, 'Moving on to the next page of the line search.'): break
      rep.append(f"the line '{wd(s)}' (pick)")
    for n in dict.fromkeys(ceil or []):
      if self._brief(lambda: (floor if repair else max(floor, self.order_due() + 2)) + 1): break
      if g := self.ceiling_walk(n, floor, repair): a['ran'] = True; rep.append(g[0]); cr[n] = list(g[1])
    if not a['ran']: return False
    b = self.batch_tool()
    ut, nd = b[2] if b else CEIL_OPENS if cr else 2, env + [n for n in tk if n not in env]
    po = [te(n, [li.get(n, ''), tk.get(n, '')], ut) for n in nd]
    po += [te(n, rk[n], min(ut, RUNG_OPENS), skim_hurdle=False) for n in rk if n not in nd]
    _trim(self, q := _h7_rr(po + list(cr.values()), ut), b, floor)
    ri = set().union(*map(self.walk_home, [*env, *cr, *tk, *rk]))
    self.rank_checked = {c for c in self.rank_checked if self.known_shelf(c.partition('|')[0]) not in ri}
    self.order_vetted = False
    m = 'Opened now, cheapest first: the unread rows of these searches that were in stock, affordable and below each pick: ' + ', '.join(f'{x} ({self.rec_db.price(x):g})' for x in q) + '. ' if q else 'None of the rows listed here is unread, in stock, affordable and below the pick. '
    ru = '[cheapest walk] ' + '; '.join(rep) + '. ' + m + 'Settle every product on its lowest-priced listing whose own fields meet all the rules that apply; the ranking is run again at each cart change and at the order.'
    if q:
      vars(self).update(open_queue=q, open_recheck=True, open_purpose='walk', open_floor=floor,
               open_say='The lowest price is wanted: opening the cheapest unread rows the walk found.')
      self.open_named()
    if la := _tool_tail(self):
      la['content'] = '%s\n%s' % (la['content'], ru if q else ru + '\n' + self.state_line())
    self.log.append({'kind': 'walk', 'plan': [[n, p] for n, _, p in plan], 'ceiling': list(cr), 'opened': q})
    return True
  def _walk_pages(self, a):
    for n, nm, pages in a['plan']:
      a['walked'].add(n)
      if not (e := a['book'].get(nm)): continue
      q, d = ' '.join(self.arg_texts(e['args'])), 0
      o = 'orders its search results by relevance' if self.top_store() else 'orders its search results by id'
      say = f"The lowest price is wanted, yet this store {o} rather than by price, so I go on paging through '{q}'."
      for _ in range(pages):
        if (t := self.next_page(e)) is None or self._brief(lambda: a['floor'] + 1, broken=True): break
        if not self.walk_call({**e['args'], a['key']: t}, say): break
        d += 1
      a['ran'] = a['ran'] or d > 0
      sn = len(e.get('row_shelf') or {})
      a['report'].append(f"'{q}' paged {d} more time(s); rows seen so far {sn} of {e['total']}")
    return 'lines'
  def decline_fits(self):
    return (self._removed() | set(self._getv('audit_fits') or []) - self._struck()) - self.refused_lines
  def decl_empty(self):
    sp = self._shelves_of(self.decline_fits())
    if self.needs.items(): return [n for n, s in enumerate(self.home_cats()) if s and not s & sp]
    return [] if set(self.shelf_labels()) & sp else [None]
  def decl_walks(self):
    pl = {n for n, r in enumerate(self.needs.props_left()) if not self._ptr_rx.search(r.lower())}
    ch = []
    for k in self.decl_empty():
      ev = sorted((-h, -self.walk_score(e['args'], k), p, nm) for p, nm, e in self.walk_opts(k)
            if (h := len(self.walk_has(e['args'], k) & pl)))
      if (f := next((x[-1] for x in ev if x[-1] not in ch), None)) is not None: ch.append(f)
    return ch[:2]
  _KEEP_N = 6
def _h7_err(m):
  return ('%s: %s' % (type(m).__name__, m))[:300]
def _trim(self, q, b, keep):
  while q and (1 if b and len(q) >= 2 else len(q)) >= self.steps_rem() - keep: q.pop()
def _h7_rr(pools, utmost):
  got, sups = [], [iter(s) for s in pools]
  while sups and len(got) < utmost:
    for s in list(sups):
      if len(got) >= utmost: break
      if (p := next((x for x in s if x not in got), None)) is None: sups.remove(s)
      else: got.append(p)
  return got
def _atom(x):
  return isinstance(x, (str, int, float)) and not isinstance(x, bool)
def _unmix(n):
  return re.sub('([a-z])([A-Z])', '\\1 \\2', str(n))
def _reqhead(req):
  hd, _, sd = str(req).partition(':')
  tg = re.findall('\\(([^)]*)\\)', hd)
  return hd, hd.split('(', 1)[0].strip(), (tg[-1] if tg else '').strip(), sd
def _dlex(ws):
  return [t for t in ws if len(t) >= 3 and not re.search('\\d', t)]
def _unlabel(call, mand, rack):
  return {p: v for p, v in call.items() if p in mand or not (isinstance(v, str) and v.strip().lower() in rack)}
def _qual(t):
  return re.split('\\s*\\(', str(t).partition(':')[0])[0]
def _tool_tail(self):
  return self.chat[-1] if self.chat and self.chat[-1].get('role') == 'tool' else None

class DeclineReads:
  _DIFFER_TOLD = '|'.join(('differ', 'not the same', 'exclu', 'apart', 'other than', 'unlike'))
  _NOT_SELLER = 'note|text|desc|comment|detail|info|message|section'
  _NOT_SELL_FLD = 'title|heading|name|label|caption|categor|shelf|department|aisle|note|text|desc|comment|detail|info|message|section|url|image|summary|overview'
  _NEGATED = "\\b(?:not|no|never|without|nor)\\b(?:\\W+[A-Za-z0-9.:/&'-]+){1,3}"
  _FLAT_SKIP = 'id$|title|shop|seller|brand|price|amount|desc|overview|note'
  def _qerr(self, kind, mistake):
    self.log.append(dict(kind=kind, error=_h7_err(mistake)))
  def row_pools(self, abodes, ids, below=None):
    cap, rp, sh, gone = self.needs.spend_cap, self.rec_db, {}, {}
    kn = rp.ids
    def note(d):
      t = self.row_key(d)
      if s := (self.shelf_for(d) if t in kn else ''):
        sh[t] = s
        if any(re.search('stock|avail', str(f).lower()) for f in d): gone[t] = self.sold_out(d)
    for it in rp.items: walk_dicts(it, note)
    return [[t for _, t in sorted((float(c), t) for t, s in sh.items() for c in [rp.price(t)] if not (
      s not in hm or t in rp.read or t in self.refused_lines or gone.get(t) or ids is not None and t not in ids
      or not is_num(c) or is_num(cap) and c * u > cap or below is not None and not c < below[i]))]
      for i, (hm, u) in enumerate(abodes)]
  def over_rows(self):
    try:
      ln, cap = self.needs.items(), self.needs.spend_cap
      if not ln or self.needs.priority() == 'cheapest' or not is_num(cap): return []
      sh, fits, low, ms = self.home_cats(), self.decline_fits(), [], []
      for i, s in enumerate(sh):
        ts = [float(p) for w in fits if self.known_shelf(w) in s
           for p in [self.rec_db.price(w)] if is_num(p)]
        if not (s and ts): return []
        low.append(min(ts)); ms.append(low[-1] * ln[i]['quantity'])
      if sum(ms, 0.0) <= cap: return []
      po = self.row_pools([(s, ln[i]['quantity']) for i, s in enumerate(sh)], None, low)
      po = [po[i] for i in sorted(range(len(po)), key=lambda i: -ms[i])]
      return po if any(po) else []
    except Exception as m:
      self._qerr('over_budget_error', m)
      return []
  def decline_reads(self):
    try:
      en = self.batch_tool()
      if vars(self).get('decline_read_done') or self.cable.spare() < CLOSING + 60: return ''
      if not (self.role('details') or en) or self.barrow.units: return ''
      sp = 2 - bool(en) + self._KEEP_N
      rm = min(3, self.steps_rem() - sp)
      se = rm > 0 and self.pg_arg() and self.role('search') and self.id_store()
      se = self.decl_walks() if se else []
      lw = self.over_rows() if self.steps_rem() > sp else []
      if not se and not lw: return ''
      self.__dict__.update(decline_read_done=True, decline_queue=(se, rm, self._KEEP_N))
      tk, sg = self._book(), []
      ak = '; '.join("'" + ' '.join(self.arg_texts(tk[s]['args'])) + "'" for s in se)
      g = '[pages] Some searches whose text holds a required value still have unread result pages (' + ak + '). This shop orders rows by id, not by match or price, so a good one may just as well sit beyond the pages seen so far; up to ' + str(rm) + ' more pages are fetched in one go, and the cheapest affordable rows on them that are still in stock get opened'
      if se: sg.append(g)
      g = "[cheaper rows] Added up, the cheapest fits of the products exceed the allowance; unopened rows in stock that undercut each product's pick get opened too"
      if lw: sg.append(g)
      g = 'one of each, in the requested quantities' if self.needs.items() else 'the item'
      g = 'Decline held for code reads. ' + '; '.join(sg) + '. Then weigh each record by meaning against the requirements in force; buy one that passes (' + g + '); if none does, decline.'
      return g
    except Exception as m:
      self._qerr('decline_reads_error', m)
      return ''
  def try_decl_reads(self):
    se, cap, keep = self.__dict__.pop('decline_queue')
    pg, en, c, aw = self.pg_arg(), self.batch_tool(), self.cable, self.arg_texts
    book, op, rd, wk = self._book(), 2 - bool(en), 0, []
    for sig in list(se) if pg else []:
      if not (w := book.get(sig)): continue
      wk.append(sig)
      say = "First paging further through '" + ' '.join(aw(w['args'])) + "', as this shop sorts by id and a match may still come later."
      while rd < cap and self.steps_rem() > keep + op and not (c.done or c.broken):
        if not c.spare() > CLOSING + 30 or (t := self.next_page(w)) is None: break
        if not self.walk_call({**w['args'], pg[0]: t}, say): break
        rd += 1
    sn, ln = set(), self.needs.items()
    for s in wk: sn.update((book.get(s) or {}).get('row_shelf') or {})
    so = [(self.walk_home(s), ln[s]['quantity'] if s is not None and s < len(ln) else self.needs.quantity())
       for s in self.decl_empty()]
    q = _h7_rr((self.row_pools(so, sn) if wk else []) + self.over_rows(),
         en[2] if en else 2)
    _trim(self, q, en, keep)
    self.log.append(dict(kind='decline_reads', pages=rd, opened=q))
    if q and not c.done:
      say = 'Before any decline, opening the cheapest affordable unread rows from the new pages and the picks.'
      self.__dict__.update(open_queue=q, open_purpose='decline', open_floor=keep, open_say=say)
      self.open_named()
    elif la := _tool_tail(self):
      g = '%s\nNothing came of it that is worth opening: no row in stock and within the cap.\n%s'
      la['content'] = g % (la['content'], self.state_line())
    return rd > 0 or bool(q)
  def ref_write(self):
    rd = self.needs.parsed
    st, nm = rd.get('reference_targets'), rd.get('references')
    po = [t for t in st if isinstance(t, dict)] if isinstance(st, list) else []
    ct = set(map(str, nm)) if isinstance(nm, list) else set()
    for t in map(str, list(self.needs.written)):
      v = t.partition(':')[2]
      if is_ref_text(v) or t in ct and re.search(DIFFER_WORDS, v.lower()):
        self.ref_add(t, po, self.request, 'request')
  def ref_untracked(self, refs, tracked):
    try:
      eg = [r for r in tracked if r.get('status') == 'resolved' or r.get('chain')]
      return [o for o in refs if not any(self.ref_match(r, o) for r in eg)]
    except Exception:
      return refs
  def ref_grab(self):
    try:
      for r in self.refs:
        if r['status'] == 'resolved' and any(n != r['now'] and self.ref_match(r, n)
                          for n in self.needs.props_left()):
          self.ref_swap(r, r['now'])
      nw = list(vars(self.needs).get('ref_news') or [])
      self.needs.ref_news = []
      for it in nw:
        rc = self.ref_add(str(it.get('need') or ''), list(it.get('infos') or []),
                    str(it.get('line') or ''), 'reply')
        if not rc: continue
        rd = self.needs.parsed
        if not isinstance(rd.get('references'), list): rd['references'] = []
        if rc['req'] not in rd['references']: rd['references'].append(rc['req'])
    except Exception as m:
      self._qerr('ref_take_error', m)
  def ref_add(self, req, infos, source, origin):
    hd, prop, tag, said = _reqhead(req)
    if not prop or junk_key(prop): return
    sl, eq = kind_words(tag), same_prop
    if any(r['req'] == req or eq(r['prop'], prop) and kind_words(r['tag']) == sl for r in self.refs): return
    e = next((x for x in infos if eq(str(x.get('property') or ''), prop) and (
      not tag or not x.get('product') or not self.needs.items() or kind_words(x.get('product')) & sl)), None)
    aim, tx = self.ref_target(e, said, source, prop), str(req)
    if not aim: return self.log.append(dict(kind='ref_skip', req=tx[:200]))
    rl, fu = ref_link(said), e if isinstance(e, dict) else {}
    hw, bw, sel = str(fu.get('relation') or '').lower(), ref_scope(prop, source), str(aim.get('seller') or '')
    at = bw.lower().find(sel.lower()) if sel.strip() else -1
    ld = bw[:at] if at > 0 else bw
    if pt := _PTR_RX.search(ld.lower()): ld = ld[:pt.end()]
    if rl[0] == 'pairs' or hw == 'pairs' or re.search(PAIR_WORDS, ld.lower()): rl = ('pairs', None, '')
    else:
      if rl[0] == 'same' and hw == 'within' and is_num(fu.get('tolerance')):
        rl = ('within', float(fu['tolerance']), str(fu.get('unit') or ''))
      if rl[0] == 'same' and (re.search(self._DIFFER_TOLD, hw)
                  or at > 0 and re.search(DIFFER_WORDS, bw.lower()[:at])):
        rl = ('differ', None, '')
    rc = dict(req=tx, head=hd.strip(), prop=prop, tag=tag, relation=rl[0], tol=rl[1], unit=rl[2], target=aim,
         first=dict(aim), chain=[], seen={self.ref_key(aim)}, judged=set(), queries=[], pages=0,
         hops=0, tries=0, fails=0, status='open', value='', goes_with=[], now='', origin=origin,
         unindexed=False)
    self.refs.append(rc)
    self.log.append(dict(kind='ref_open', req=tx[:200], target=aim, relation=rl[0], origin=origin))
    return rc
  def ref_target(self, e, said, source, prop):
    fu = e if isinstance(e, dict) else {}
    sel, kd, raw = str(fu.get('seller') or '').strip(), str(fu.get('kind') or '').strip(), fu.get('details')
    dt = [str(b).strip() for b in (raw if isinstance(raw, list) else ()) if _atom(b) and str(b).strip()]
    ok = lambda n: self.ref_seller_ok(n, source)
    win = next((p for p in (parse_ref(t, ok) for t in (ref_scope(prop, source), said))
          if p and ok(p['seller'])), None)
    if ok(sel):
      if win and not dt and norm_name(win['seller']) == norm_name(sel): dt = win['details']
    elif win: sel, dt = win['seller'], dt or win['details']
    else: return
    hr = set(ref_terms(source))
    if win: kd = kd or win['kind']
    dt = [d for d in dict.fromkeys(dt) if hr.intersection(ref_terms(d))]
    return {'seller': sel, 'kind': kd[:80], 'details': dt[:6]}
  def ref_seller_ok(self, seller, source):
    nm, fq = str(seller or '').strip(), set()
    if sum(c.isascii() and c.isalpha() for c in nm) < 3: return False
    if re.search("(?<![a-z0-9])%s(?:'s)?(?![a-z0-9])" % re.escape(nm.lower()), str(source or '').lower()) is None:
      return False
    for r in self.needs.items() or [{'product': str(self.needs.parsed.get('product') or '')}]:
      fq.update(kind_words(r['product']))
    for f in ((self.argspec.get(self.role('search') or '') or {}).get('properties') or {}).values():
      ch = f.get('enum') if isinstance(f, dict) else None
      for x in ch if isinstance(ch, list) else (): fq.update(kind_words(x))
    own = kind_words(nm)
    return bool(own) and not own.issubset(fq)
  @staticmethod
  def ref_key(intended):
    ts = {t for s in intended.get('details') or () for t in ref_terms(s)}
    return norm_name(intended.get('seller') or ''), tuple(sorted(ts))
  def ref_desc(self, handle):
    t = handle['target']
    ws = ['the', str(t.get('kind') or 'listing'), 'sold by', str(t.get('seller'))]
    if t.get('details'): ws.append('(%s)' % '; '.join(t['details']))
    return ' '.join(ws)
  def ref_idx(self):
    bk, fd, un, lt = self.rec_db.items, [], {}, {}
    if (mm := vars(self).get('ref_memo')) and mm[0] == len(bk): return mm[1]
    for i, it in enumerate(bk): walk_dicts(it, lambda d: fd.append((i, d)))
    for i, d in fd:
      t = self.row_key(d)
      if (len(d) > 3 or any(isinstance(s, dict) and len(s) > 1 for s in d.values())) and t in self.rec_db.ids:
        un.setdefault(t, {}).update(d); lt[t] = i
    self.ref_memo, self.ref_seen_at = (len(bk), un), lt
    return un
  def ref_sellers(self, row):
    aq = self.seller_keys()
    def has(f):
      lw = str(f).lower()
      return (re.search(SELLER_KEYS, lw) or str(f) in aq) and 'id' not in lw and not re.search(
        self._NOT_SELLER, lw)
    return {norm_name(t) for f, t in row.items() if isinstance(t, str) and t.strip() and len(t) <= 60
        and has(f)} - {''}
  def seller_keys(self):
    try:
      ap = norm_name
      rt = {ap(t.get('seller')) for r in vars(self).get('refs') or []
         for t in (r.get('target') or {}, r.get('first') or {}) if isinstance(t, dict)} - {''}
      im = (len(self.rec_db.items), tuple(sorted(rt)))
      if (mm := vars(self).get('seller_memo')) and mm[0] == im: return mm[1]
      cards, hr, cs = list(self.ref_idx().values()), set(), set()
      for s in cards:
        for sb in s.values():
          if isinstance(sb, dict): cs.update(ap(x) for x in sb.values() if isinstance(x, str))
        for st in re.split('(?<=[.!?;])\\s+', self.ref_texts(s, dsx=False)):
          rn = [[]]
          for w in re.findall("[A-Za-z][A-Za-z'-]*", st)[1:]:
            if w[:1].isupper() and re.search('[a-z]', w): rn[-1].append(w)
            elif rn[-1]: rn.append([])
          hr |= {ap(' '.join(r[i:j])) for r in rn for i in range(len(r))
             for j in range(i + 1, min(len(r), i + 3) + 1)}
      hr = {n for n in hr - cs if len(n) > 3}
      def cell(f, v):
        lw = str(f).lower()
        if not isinstance(v, str) or not v or len(v) > 60: return False
        if 'id' in lw or re.search(self._NOT_SELL_FLD, lw): return False
        return (b := ap(v)) in rt or v.strip()[:1].isupper() and len(v.split()) < 4 and b in hr
      ks = {str(f) for s in cards for f, v in s.items() if cell(f, v)}
      self.seller_memo = (im, ks)
      return ks
    except Exception:
      return set()
  @staticmethod
  def ref_texts(row, dsx=True):
    nt, ab = [], []
    for f, t in row.items():
      lw = str(f).lower()
      if not isinstance(t, str) or len(t.split()) < 3 or lw.endswith('id'): continue
      if re.search('title|heading|headline|name|label|caption|categor|shelf|department|url|image', lw): continue
      (ab if re.search('desc|overview|summary|about', lw) else nt).append(t)
    return '\n'.join(nt or (ab if dsx else []))
  def _row_maps(self, row):
    return [s for s in row.values() if isinstance(s, dict)] or [
      {f: x for f, x in row.items() if _atom(x) and not re.search(self._FLAT_SKIP, str(f).lower())}]
  def ref_parts(self, row):
    sd, hs = [self.title_of(row), re.sub(self._NEGATED, ' ', self.ref_texts(row, dsx=False),
                           flags=re.I)], []
    for fs in self._row_maps(row):
      hs += [_unmix(f).replace('_', ' ') for f in fs]; sd += [str(x) for x in fs.values() if _atom(x)]
    sn = set(ref_terms(' | '.join(sd)))
    sn |= {m.group(1) for x in sn if (m := re.fullmatch('\\d+(?:[.,:/]\\d+)*([a-z]{3,})', x))}
    return set(ref_terms(' '.join(hs))), sn
  def ref_shows(self, row, specific, segments=None):
    hs, sd = segments or self.ref_parts(row)
    ws = _dlex(ref_terms(specific))
    rq = set(ws) - hs
    if not rq and len(ws) > 1:
      far = max((len(h) for fs in self._row_maps(row) for f in fs
           for h in [_dlex(ref_terms(_unmix(f).replace('_', ' ')))] if ws[:len(h)] == h), default=0)
      rq = set(ws[far:]) if far else set()
    if rq and rq.isdisjoint(sd): return False
    if not (fg := ref_figs(specific)): return True
    own = [self.title_of(row)]
    for s in filter(lambda x: isinstance(x, dict), row.values()):
      own.extend(str(x) for x in s.values() if isinstance(x, (str, int, float)))
    wr = ref_figs(' | '.join(own))
    return all(not (eq := [d for d, u in wr if u == un]) or any(abs(d - g) < 1e-09 for d in eq) for g, un in fg)
  @staticmethod
  def row_field(row, prop):
    fl = [(f, str(x).strip()) for s in row.values() if isinstance(s, dict) for f, x in s.items()
       if _atom(x) and str(x).strip()]
    return next((t for c in (exact_prop, same_prop) for f, t in fl if c(f, prop)), '')
  def ref_ids_in(self, ids):
    self.ref_ids = set(vars(self).get('ref_ids') or ()).union(ids)
  def ref_extra(self):
    try:
      ak = vars(self).get('ref_ids') or set()
      sh = self.home_cats() if ak else []
      if not (sh and all(sh)): return set()
      ab, ix, bk = set().union(*sh), self.ref_idx(), set(self.barrow.units)
      return {t for t in ak if t not in bk and (s := self.shelf_for(ix.get(t) or {})) and s not in ab}
    except Exception:
      return set()
  def ref_cands(self, handle):
    aim = handle['target']
    sel, so, ix = norm_name(aim['seller']), kind_words(aim['kind']), self.ref_idx()
    lt, pn = vars(self).get('ref_seen_at') or {}, []
    sg = [set(ref_terms(d)) for d in aim['details']]
    for t, s in ix.items():
      if sel not in self.ref_sellers(s): continue
      tz = [str(x) for f, x in s.items()
         if isinstance(x, str) and re.search('desc|overview|summary', str(f).lower())]
      hy = set(ref_terms(' '.join([self.title_of(s)] + tz)))
      h = sum(1 for p in sg if p and p.issubset(hy))
      pn.append((-(2 if so & kind_words(self.shelf_for(s)) else 0) - h, -lt.get(t, 0), t, h))
    pn.sort()
    handle['shows'] = {t: h for _, _, t, h in pn}
    return [t for _, _, t, _ in pn]
  def ref_template(self):
    spec, aw, sg = self.argspec.get(self.role('search') or '') or {}, self.arg_texts, self._qo_sgn
    rq = spec.get('required')
    mand, gr = set(rq or []) if isinstance(rq, list) else set(), {}
    rack = set(map(self.shelf_for, self.ref_idx().values())) - {''}
    for i, w in enumerate(self._book().values()):
      cl = w.get('args') if isinstance(w.get('args'), dict) else {}
      wr = aw(cl)
      ns = [p for p, v in cl.items() if isinstance(v, str) and v in wr and v.strip()]
      if len(wr) != 1 or len(ns) != 1: continue
      an = _unlabel({p: v for p, v in cl.items() if p != ns[0]}, mand, rack)
      g = gr.setdefault(sg(an), {'tpl': (an, ns[0]), 'seen': [], 'at': i})
      g['seen'].append(({self.lemma(x) for x in re.findall('[a-z0-9]+', wr[0].lower())},
               tuple(w.get('first') or ())))
    uns = {s for s, g in gr.items() if any(len(ea) >= 2 and ea == eb and not la & lb
                        for n, (la, ea) in enumerate(g['seen'])
                        for lb, eb in g['seen'][n + 1:])}
    live = sorted(((-len({tuple(sorted(st)) for st, _ in g['seen']}), g['at'], g['tpl']) for s, g in gr.items()
           if s not in uns), key=lambda r: r[:2])
    md = live[0][2] if live else None
    if not live:
      pg = self.pg_arg()
      for e in reversed(self.log):
        if e.get('kind') != 'env': continue
        for d in reversed(e.get('actions') or []):
          cl = d.get('args') if d.get('name') == self.role('search') else None
          if not isinstance(cl, dict): continue
          ns = [p for p, v in cl.items() if isinstance(v, str) and v.strip() and v in aw(cl)]
          if len(ns) != 1: continue
          rs = {p: v for p, v in cl.items() if p != ns[0] and not (pg and p == pg[0])}
          if sg(_unlabel(rs, mand, rack)) not in uns: md = (rs, ns[0]); break
        if md: break
    if md: return _unlabel(md[0], mand, rack), md[1]
    fs, tk = spec.get('properties'), self.term_key()
    fr = [p for p, f in (fs.items() if isinstance(fs, dict) else ()) if isinstance(f, dict)
       and f.get('type') == 'string' and not f.get('enum') and p != tk]
    return ({}, fr[0]) if len(fr) == 1 and mand <= {fr[0]} else None
  def ref_words(self, handle):
    hs = [_unmix(f) for s in self.ref_idx().values() for sb in s.values() if isinstance(sb, dict) for f in sb]
    hs += [_reqhead(r)[1] for r in self.needs.written + self.needs.uncovered]
    kn, aim = {x for t in hs for x in re.findall('[a-z]+', t.lower())}, handle['target']
    hp = {'joined': [], 'figure': [], 'word': []}
    own, fa = set(re.findall('[a-z0-9]+', str(aim['seller']).lower())), re.findall
    for sp in aim['details']:
      af = False
      for h in str(sp).lower().split():
        for x in fa('[a-z0-9]+', h):
          if x in STOP_WORDS or x in own: continue
          if re.search('\\d', x): kd = 'joined' if re.search('[a-z]', x) else 'figure' if len(x) > 1 else None
          elif af or len(x) <= 2 or x != h.strip('.,;:!?()[]{}"\''): kd = None
          else: kd = None if x in kn or x.rstrip('s') in kn else 'word'
          if kd: hp[kd].append(x)
        af = re.fullmatch('\\d+(?:[.,]\\d+)?', h) is not None
    pd, fs = sorted(hp['word'], key=lambda x: -len(x)), self.store_keys()
    ms = [[x for x in fa('[a-z0-9]+', str(sp).lower()) if x not in STOP_WORDS] for sp in aim['details']]
    dy = handle.get('hops', 0) == 0 and bool(ms) and all(len(s) > 1 for s in ms)
    ld = [s[-1] for s in ms if len(s) >= 2 and s[-1] in pd
       and (dy or any(' '.join(s[:j]) in fs for j in range(1, len(s))))]
    return list(dict.fromkeys(hp['joined'] + hp['figure'] + ld + [x for x in pd if x not in ld]))
  _SENT_SPLIT = '(?<=[.!?;])\\s+'
  _WORD_RX = "[A-Za-z][A-Za-z'-]*"
  _NEGATORS = frozenset(('not', 'nor', 'never', 'no', 'neither'))
  _READ_SAY = 'Opening now the rows that might be the page %s takes its value from, as the shopper said.'
  _SEARCH_SAY = 'Store search for the exact page that %s is tied to, that is [%s].'
  _FACTS_SAY = 'Reading, with %s, the property sheet of the page %s takes its value from, because its record has no such field.'
  _NOTES_SAY = 'Fetching, with %s, the remarks of a page that may be the source of %s: such text can in turn hand the pointer on to yet another page.'
  _REF_SAY = '%s is part of the pointer chain for %s (the source page itself, or one its remarks lead through), so its %s holds exactly the value%s that the wanted %s must avoid; it can never qualify'
  _LISTS_SAY = '%s lists %s in its %s, the value the wanted %s must go with rather than be; it can never qualify'
  _SMR_REMARK = "%s: the remarks on it may move %s to another page. In that case its own field proves nothing for %s (grading reads that other page instead); it meets %s only where that page's record shows the value; a new item check comes before any order%s"

class RefPlan:
  def _qr_jot(self, kind, **facts):
    self.log.append({'kind': kind, **facts})
  def _tag_of(self, tag):
    return self.prod_of(f'x ({tag})')
  def _rlinks(self):
    return vars(self).get('refs') or []
  def _open(self):
    return [link for link in self._rlinks() if link.get('status') == 'open']
  def _logged(self, start):
    sp = self.cable.ticks - start
    self._smr_steps = vars(self).get('_smr_steps', 0) + sp
    return sp > 0
  def store_keys(self):
    rc, mt, nm = self.rec_db.items, [], set()
    if (mm := vars(self).get('_smr_keys_at')) and mm[0] == len(rc): return mm[1]
    for r in rc: walk_dicts(r, mt.append)
    for t in mt:
      for v in t.values():
        if isinstance(v, dict) and sum(isinstance(x, (str, int, float)) for x in v.values()) >= 2:
          nm.update(filter(None, (' '.join(re.findall('[a-z0-9]+', re.sub('(?<=[a-z])(?=[A-Z])', ' ', str(x))
                               .lower())) for x in v)))
    self._smr_keys_at = (len(rc), nm)
    return nm
  def ref_plans(self, link):
    t, gi = link['target'], max(2, self.lead_words or 3)
    sel = ' '.join(re.findall('[A-Za-z0-9]+', re.sub("'s\\b", '', str(t['seller']))))
    dt = self.ref_words(link)
    nom = [w for w in re.findall('[a-z]{3,}', str(t['kind']).lower()) if w not in STOP_WORDS][-1:]
    if link.get('unindexed'): op = [' '.join(ws)] if (ws := (dt[:1] + nom)[:gi]) else []
    else: op = [f'{sel} {dt[0]}', sel] if dt and len(sel.split()) < gi else [sel]
    return [x for x in op if x.strip() and x.lower() not in link['queries']]
  def ref_page(self, link):
    p = self.pg_arg()
    if not p or link['pages'] >= 3 or not (s := norm_name(link['target']['seller'])): return
    for w in reversed(list(self._book().values())):
      if s in norm_name(' '.join(self.arg_texts(w['args']))) and (
          n := self.next_page(w)) is not None:
        return {**w['args'], p[0]: n}
  def ref_room(self):
    pur, used = self._open(), vars(self).get('_smr_steps', 0)
    only, ch = bool(pur) and all(l.get('relation') == 'differ' for l in pur), any(l.get('chain') for l in pur)
    if only and used >= HANDLE_DEVIATE_UNSEEING and not ch: return False
    fl, sk = self.ref_floor(), self.step_cap
    cap, rs = max(6, int(HANDLE_DEVIATE_STRIDES * sk)) if only else max(8, int(HANDLE_STRIDES * sk)), 60
    if ch or self.ref_alone(): cap, rs = sk, 10
    c = self.cable
    if used >= cap or c.spare() <= CLOSING + rs or c.spent_out() or c.done or c.broken: return False
    if self.ref_stalls(fl): return False
    if self.steps_rem() > fl: return True
    if self.steps_rem() - self.order_due() <= 2: return False
    for l in self._rlinks():
      t = l.get('target') or {}
      if l.get('status') != 'open' or l.get('hops', 0) < 1: continue
      if not all(str(t.get(f) or '').strip() for f in ('seller', 'kind')): continue
      if not t.get('details'): continue
      if l.get('late_from') is not None:
        if used - l['late_from'] < 3: return True
      elif not l.get('queries') and self.pairs_ok(l):
        l['late_from'] = used
        self._qr_jot('ref_late', prop=l.get('prop'), to=str(t.get('seller'))[:80])
        return True
    return False
  def ref_floor(self):
    if self.ref_alone(): return FIRST_ORDER_BY
    if any(l.get('chain') for l in self._open()):
      return max(FIRST_ORDER_BY + 5, self.order_due() + max(0, len(self.needs.items()) - 1))
    return max(FIRST_ORDER_BY + 5, min(FIRST_ORDER_DUE, self.step_cap // 4 + 1) + 5,
         self.order_due() + 7)
  def ref_alone(self):
    return not self.needs.items() and bool(self._open()) and not self._own_issue()
  def ref_prod(self, link):
    return self._tag_of(link['tag']) if self.needs.items() and link.get('tag') else None
  def ref_stalls(self, floor=None):
    pr, pur = self.needs.items(), self._open()
    if not (pr and pur) or any(l.get('relation') != 'differ' or not l.get('chain') for l in pur): return False
    if self.steps_rem() <= 4 + (self.ref_floor() if floor is None else floor): return False
    try:
      own = set(map(self.ref_prod, pur))
      cp = [o for o, _ in enumerate(pr) if not any(r['fit'] for r in self.zd_pool(o))]
    except Exception as m:
      self._qr_jot('ref_wait_error', error=_h7_err(m))
      return False
    cp = [o for o in cp if o not in own]
    if (hold := len(cp) > (1 if None in own else 0)) and vars(self).get('_smr_wait_for') != tuple(cp):
      self._smr_wait_for = tuple(cp)
      self._qr_jot('ref_wait', devoid=cp)
    return hold
  def pairs_ok(self, link):
    if not (pr := self.needs.items()): return True
    try: dub = [o for o, _ in enumerate(pr) if not any(r['verified'] and r['fit'] for r in self.zd_pool(o))]
    except Exception: return False
    own = self.ref_prod(link)
    return sum(o != own for o in dub) <= (1 if own is None else 0)
  def ref_calls_ok(self):
    c, rs = self.cable, 20 if any(l.get('chain') for l in self._open()) else 40
    if vars(self).get('ref_calls', 0) >= HANDLE_SUMMONS: return False
    if c.spare() <= CLOSING + rs or c.spent_out(): return False
    return c.spend < SPEND_CAP * c.__dict__.get('scale', 1.0)
  def ref_match(self, link, rule):
    tx = str(rule)
    lb, _, v = tx.partition(':')
    if tx in (link['req'], link.get('now')): return True
    if not (is_ref_text(v) or link.get('relation') == 'differ' and re.search(DIFFER_WORDS, v.lower())):
      return False
    if not same_prop(_qual(lb), link['prop']): return False
    tg = re.findall('\\(([^)]*)\\)', lb)
    return not tg or not link['tag'] or bool(kind_words(tg[-1]) & kind_words(link['tag']))
  def ref_live(self, link):
    eq = lambda r: same_prop(_qual(r), link['prop'])
    if any(self.ref_match(link, r) or eq(r) for r in self.needs.props_left()): return True
    return not any(map(eq, self.needs.let_go))
  def ref_step(self):
    if self.cable.done or self.cable.broken or not vars(self).get('refs'): return False
    try: return self._rvolley()
    except Exception as m:
      self._qr_jot('ref_error', error=_h7_err(m))
      for l in self._open(): l.update(status='stuck', why='the chain-following code hit an error')
      return False
  def _rvolley(self):
    live, rm, lf = [], self.ref_plans, self.ref_page
    for l in self.refs:
      if l['status'] != 'open': continue
      if self.ref_live(l): live.append(l)
      else: l['status'] = 'dropped'; self._qr_jot('ref_dropped', req=l['req'][:200])
    if not live: return False
    if any(self.ref_facts_read(l) for l in live if l.get('facts_due')): return True
    hb, ac = self.cable.ticks, False
    for l in sorted(live, key=lambda x: x['hops']): ac = self.ref_judge(l) or ac
    if self.cable.ticks > hb: return True
    live = [l for l in live if l['status'] == 'open']
    if any(l.get('facts_due') for l in live): return True
    if not (live and self.ref_room()): return ac
    en = self.batch_tool()
    ut = en[2] if en else 2
    po, tpl = [self._pool(l, ut) for l in live], self.ref_template()
    idle = [l for l, st in zip(live, po) if not st and rm(l)] if tpl else []
    q = _h7_rr(po, ut)
    fi = idle if 0 < len(q) < ut else []
    if q and not fi: return self.ref_read(q, live) or ac
    if not tpl: return ac
    for l in fi or sorted(live, key=lambda x: (x['hops'], len(x['queries']))):
      if r := rm(l):
        l['queries'].append(r[0].lower())
        return True if self.used_query(r[0]) else self.ref_search(l, {**tpl[0], tpl[1]: r[0]}) or ac
      if p := lf(l): l['pages'] += 1; return self.ref_search(l, p) or ac
    for l in live:
      if any(p not in l['judged'] for p in self.ref_cands(l)) or rm(l) or lf(l): continue
      self.ref_stop(l, 'after every attempt, no store query brought up the page it names')
      ac = True
    return ac
  def _pool(self, link, utmost):
    sn, jd = self.rec_db.read, link['judged']
    fr = [p for p in self.ref_cands(link) if p not in sn and p not in jd]
    if link['queries'] or not self.ref_plans(link) or len(fr) <= utmost: return fr
    return fr if fr and (link.get('shows') or {}).get(fr[0]) else []
  def _pull(self, ids, say, enumerator, reader, kind, **extra):
    hb, ok = self.cable.ticks, True
    rp = self.open_reads(ids, say, enumerator, reader)
    if enumerator and rp and not vars(self).get('batch_proven'):
      got = vars(self).get('last_reads') or []
      if ok := self.batch_ok(got, ids) and self.batch_denser(got, ids): self.batch_proven = True
      elif self.ids_back(got, ids) or not self._batch_shrink(len(ids)):
        self.batch_spec = None
        self._qr_jot('batch_dropped', tool=enumerator[0])
    if rp: self.last_replies(rp)
    dn = self._logged(hb)
    self.ref_ids_in(ids)
    self._qr_jot(kind, ids=ids, **extra)
    return dn, ok
  def ref_read(self, queue, opens):
    en = self.batch_tool() if len(queue) >= 2 else None
    say = self._READ_SAY % ', '.join(dict.fromkeys(l['head'] for l in opens))
    dn, ok = self._pull(queue, say, en, None, 'ref_read')
    if ok:
      un = [p for p in queue if p not in self.rec_db.read]
      for l in opens: l['judged'].update(un)
    return dn
  def ref_search(self, link, args):
    hb = self.cable.ticks
    hit = self.walk_call(args, self._SEARCH_SAY % (link['head'], self.ref_desc(link)))
    dn, rows = self._logged(hb), []
    if hit:
      walk_dicts(self.rec_db.items[-1], rows.append, ceiling=3)
      rows = [r for r in rows if self.row_key(r) in self.rec_db.ids and len(r) >= 4]
    s, q = norm_name(link['target']['seller']), ' '.join(self.arg_texts(args))
    if rows and norm_name(q) == s and all(s not in self.ref_sellers(r) for r in rows):
      link['unindexed'] = True
    self._qr_jot('ref_search', prop=link['prop'], query=q[:120], rows=len(rows))
    return dn
  _JUDGE_SEQ = ('_smr_j_notes_first', '_smr_j_notes_wait', '_smr_j_notes_lost', '_smr_j_plain', '_smr_j_no_calls',
          '_smr_j_hop')
  def ref_judge(self, link):
    t, rd, jd = link['target'], self.rec_db.read, link['judged']
    op = [p for p in self.ref_cands(link) if p in rd]
    if not (fr := [p for p in op if p not in jd]): return False
    ix = self.ref_idx()
    sp = {p: self.ref_parts(ix.get(p) or {}) for p in op}
    lo, dt = link.setdefault('loose', set()), t['details']
    sw = lambda p, s: self.ref_shows(ix.get(p) or {}, s, sp[p])
    if len(op) >= 4: lo.update({s for s in dt if not any(sw(p, s) for p in op)})
    sg = [set(ref_terms(s)) for s in dt]
    sc = {p: sum(1 for x in sg if x and x <= st) for p in fr for st in [sp[p][0] | sp[p][1]]}
    fits = {p: all(sw(p, s) for s in dt if s not in lo) for p in fr}
    pri = sorted(fr, key=lambda p: (not fits[p], -sc[p]))
    if not fits[pri[0]] and (self.ref_room() or self.ref_stalls()) and (
        any(p not in rd and p not in jd for p in self.ref_cands(link))
        or self.ref_plans(link) or self.ref_page(link)): return False
    apt = [p for p in pri if fits[p]]
    pk = apt[:4] or pri[:4]
    ex = [p for p in pk if dt and fits[p] and sc[p] == len(dt)]
    sn = {'link': link, 'aim': t, 'fresh': fr, 'fits': fits, 'pick': pk, 'exact': ex, 'index': ix,
       'missing': self.ref_gaps(link, ex if len(ex) == 1 else pk[:2] if apt else pk[:1])}
    for j in self._JUDGE_SEQ:
      if (v := getattr(self, j)(sn)) is not None: return v
    return True
  def ref_facts_read(self, link):
    due = link.pop('facts_due', None)
    tl = self.facts_tool(due) if due else None
    if not tl or not self.ref_room(): return False
    return self._pull([due], self._FACTS_SAY % (tl, link['head']), None, tl, 'ref_facts', tool=tl)[0]
  def facts_tool(self, topper):
    us, cp = vars(self).get('read_with') or {}, vars(self).get('tool_caps') or {}
    pw = lambda t: (cp.get(t) or {}).get('props', 0)
    idle = [t for t in self.fetch_tools() if topper not in (us.get(t) or set())]
    if hq := [t for t in idle if pw(t) > 0]: return max(hq, key=lambda t: (pw(t), t))
    return next((t for t in (self.role('view'), self.role('details')) if t and t in idle), None)
  def _smr_j_notes_first(self, sc):
    if sc['missing'] and self.ref_room() and self.ref_notes_read(sc['link'], sc['missing']): return True
  def _smr_j_notes_wait(self, sc):
    if sc['missing'] and self.ref_stalls(): return False
  def _smr_j_notes_lost(self, sc):
    l = sc['link']
    if not sc['missing'] or self.ref_room() or l.get('relation') != 'differ': return
    l['judged'].update(sc['pick'])
    self.ref_stop(l, 'not enough steps were left to read the remarks on the page it now names')
    return True
  def _smr_j_plain(self, scenario):
    link, exact = scenario['link'], scenario['exact']
    if len(exact) != 1 or not self.ref_plain(link, exact[0]): return None
    link['judged'].update(scenario['pick'])
    self.ref_ids_in(scenario['pick'])
    self.ref_clarify(link, exact[0], self.row_field(scenario['index'].get(exact[0]) or {}, link['prop']), [])
    return True
  def _smr_j_no_calls(self, scenario):
    if self.ref_calls_ok(): return None
    link = scenario['link']
    link['judged'].update(scenario['pick'])
    self.ref_stop(link, 'no hop checks were left')
    return True
  def _smr_j_hop(self, scenario):
    link, fits, pick = scenario['link'], scenario['fits'], scenario['pick']
    self.ref_apply(link, pick, self.ref_ask(link, pick, ''))
    if link['status'] == 'open' and link['target'] is scenario['aim']:
      loose = link.get('loose') or set()
      words = [d for d in scenario['aim']['details'] if d not in loose]
      idx = scenario['index']
      link['judged'].update(p for p in scenario['fresh']
                 if not fits[p] and self.ref_lacks(idx.get(p) or {}, words))
    return True
  def ref_lacks(self, row, details):
    heads, voiced = self.ref_parts(row)
    return any((w := {x for x in ref_terms(d) if len(x) >= 3 and not re.search('\\d', x)} - heads)
         and w.isdisjoint(voiced) for d in details)
  def _rnames(self, rcr, table):
    sellers = self.ref_sellers(rcr)
    rivals = set().union(*map(self.ref_sellers, table.values())) - sellers
    heads, _ = self.ref_parts(rcr)
    txt = [self.title_of(rcr), self.shelf_for(rcr)]
    txt += [str(t) for v in rcr.values() if isinstance(v, dict) for t in v.values() if isinstance(t, (str, int,
        float))]
    return (sellers, rivals, heads | set(ref_terms(' | '.join(txt))))
  @staticmethod
  def _extra(word, low, sellers, rivals, facts, exacting):
    if low in rivals: return True
    if not word[:1].isupper() or len(low) < 4 or low in sellers: return False
    return not (exacting and set(ref_terms(word)) & facts) and low not in facts
  def ref_plain(self, link, topper):
    prop = link['prop']
    rcr = self.ref_idx().get(topper) or {}
    value = self.row_field(rcr, prop)
    txt = self.ref_texts(rcr, dsx=False)
    if not value or self.ref_about(prop, txt) or set(ref_terms(value)) & set(ref_terms(txt)):
      return False
    a, b, c = self._rnames(rcr, self.ref_idx())
    return not any(self._extra(w, norm_name(w), a, b, c, False)
           for s in re.split(self._SENT_SPLIT, txt) for w in re.findall(self._WORD_RX, s)[1:])
  def ref_notes_tool(self):
    got = self.notes_tool()
    if got: return got
    mine = vars(self)
    details = self.role('details')
    if (mine.get('note_keys') or {}).get(details or ''): return None
    ids, txt, caps = mine.get('tool_keys') or {}, mine.get('text_keys') or {}, mine.get('tool_caps') or {}
    seen = set(ids.get(details or '') or ()) | set(mine.get('row_keys') or ())
    rivals = [t for t in self.fetch_tools() if t != details]
    denser = [t for t in rivals if (set(ids.get(t) or ()) - seen) & set(txt.get(t) or ())]
    if denser: return max(denser, key=lambda t: ((caps.get(t) or {}).get('calls', 0), t))
    return next((t for t in (self.role('view'), self.stock_tool())
          if t in rivals and not (caps.get(t) or {}).get('id', 0)), None)
  def ref_gaps(self, link, ids):
    tool = self.ref_notes_tool()
    if not (tool and ids): return []
    done = set((vars(self).get('read_with') or {}).get(tool) or ()) | link.setdefault('noted', set())
    return [p for p in dict.fromkeys(ids) if p not in done][:2]
  def ref_notes_read(self, link, need):
    tool = self.ref_notes_tool()
    if not (tool and need): return False
    link.setdefault('noted', set()).update(need)
    return self._pull(need, self._NOTES_SAY % (tool, link['head']), None, tool, 'ref_notes', tool=tool)[0]
  def _differs(self, conditions=None, likewise=None):
    for link in self._rlinks():
      if (link.get('relation') == 'differ' and (conditions is None or link.get('status') in conditions)
          and (likewise is None or likewise(link)) and self.ref_live(link)):
        yield link
  def _excepts(self, conditions=None, likewise=None):
    yield from self._differs(conditions, likewise)
    if conditions is not None and 'resolved' not in conditions: return
    for link in self._rlinks():
      if link.get('relation') != 'pairs' or link.get('status') != 'resolved': continue
      v = set(ref_terms(link.get('value')))
      if (not any(set(ref_terms(c)) == v for c in link.get('goes_with') or [])
          and (likewise is None or likewise(link)) and self.ref_live(link)):
        yield link
  def ref_ptrs(self):
    return {p for link in self._excepts(('open', 'stuck', 'resolved')) for p in link.get('chain') or []
        if self.ref_same(link, p)}
  def ref_same(self, link, topper):
    if not self.needs.items(): return True
    shelf = self.known_shelf(topper)
    home = self._tag_of(shelf) if shelf else None
    return home is None or not link.get('tag') or home == self._tag_of(link['tag'])
  def ref_ptr_note(self, item):
    for link in self._excepts(likewise=lambda link: item in (link.get('chain') or [])):
      if self.ref_same(link, item):
        known = f' ({link["value"]})' if link.get('status') == 'resolved' and link.get('value') else ''
        return self._REF_SAY % (item, link['head'], link['prop'], known, link['prop'])
    return self._list_read(item)
  def _list_read(self, item):
    for link in self._excepts(('resolved',)):
      if link.get('relation') != 'pairs' or not self.ref_same(link, item): continue
      v = set(ref_terms(link.get('value')))
      prop = link['prop']
      fields = (self.all_rows().get(item) or {}).get('fields') or {}
      if any(set(ref_terms(s)) == v for t in self._prop_fields(fields, prop) for s in re.split('[,;/&]',
          str(t))):
        return self._LISTS_SAY % (item, link['value'], prop, prop)
    return ''
  @staticmethod
  def _hit_value(link, facts):
    flat = lambda v: ' '.join(ref_terms(v))
    avoid = {flat(v) for v in (link.get('value'), *(link.get('goes_with') or [])) if str(v or '').strip()}
    return any(same_prop(k, link['prop']) and flat(v) in avoid for k, v in (facts or {}).items())
  def ref_excluded(self, facts, which):
    products = self.needs.items()
    return any(self._hit_value(link, facts) for link in self._differs(('resolved',))
         if not (products and link.get('tag')
             and self._tag_of(link.get('tag')) not in (which, None)))
  def _emits_over(self, statement, attrb, sellers, rivals, facts):
    words = list(re.finditer(self._WORD_RX, statement))
    hid = re.sub('[A-Za-z0-9]+(?=\\s+by\\s+[A-Z])', lambda m: ' ' * (m.end() - m.start()), statement)
    anchor = next((w.start() for w in words if prop_named(attrb, hid[:w.end()])), None)
    if anchor is None: return False
    for pos, m in enumerate(words[1:], 1):
      low = norm_name(m.group(0))
      if (m.start() > anchor and low and self._extra(m.group(0), low, sellers, rivals, facts, True)
          and not any(f.group(0).lower() in self._NEGATORS for f in words[max(0, pos - 5):pos])):
        return True
    return False
  def own_detours(self, item):
    try:
      index = self.ref_idx()
      rcr = index.get(item) or {}
      txt = self.ref_texts(rcr, dsx=False)
      if not txt: return []
      a, b, c = self._rnames(rcr, index)
      products = self.needs.items()
      shelf = self.shelf_for(rcr)
      home = self._tag_of(shelf) if products and shelf else None
      lines = re.split(self._SENT_SPLIT, txt)
      found = []
      for rule in self.needs.props_left():
        label = str(rule).partition(':')[0]
        if home is not None and self.prod_of(label) not in (home, None): continue
        prop = re.split('\\s*\\(', label)[0].strip()
        if prop and (hit := next((s for s in lines if self._emits_over(s, prop, a, b, c)),
            None)) is not None:
          found.append((str(rule), prop, hit.strip()))
      return found
    except Exception:
      return []
  def notes_msg(self, summon_appellation, summon_prms):
    if (summon_appellation in tuple(map(self.role, ('search', 'cart', 'cart_view', 'message')))
        or summon_appellation in self.end_tools()):
      return ''
    rev, units, passed = self.needs.rev_no, self.barrow.units, self.passed
    out = []
    for top in sorted(self.asked_ids(summon_prms)):
      mark = f'{top}|{rev}'
      if top not in units and mark not in passed or mark in (told := vars(self).setdefault('_smr_told', set())):
        continue
      if not (found := self.own_detours(top)): continue
      told.add(mark)
      passed.discard(mark)
      self._qr_jot('own_redirect', id=top, props=[e[1] for e in found])
      out.append(self._say_line(top, found, top in units))
    return ' '.join(out)
  @classmethod
  def _say_line(cls, topper, found, held):
    return cls._SMR_REMARK % (topper, '; '.join('%s ["%s"]' % (p, s[:160]) for _, p, s in found),
                 '; '.join("'%s'" % e[0] for e in found), 'them' if len(found) != 1 else 'it',
                 '. If so, remove this line and put in another confirmed pick.' if held else '.')
  def ref_phrase(self, line):
    text = str(line or '')
    if not is_ref_text(text): return None
    info = parse_ref(text, lambda name: self.ref_seller_ok(name, text))
    if not info or not self.ref_seller_ok(info['seller'], text): return None
    at = re.search("(?<![A-Za-z'])" + re.escape(info['seller']) + "(?:'s)?(?![A-Za-z])", text)
    if not at: return None
    ahead = text[:at.start()]
    cut = max((m.start() for m in re.finditer('[,;!?]|[.](?=\\s|$)', ahead)), default=-1) + 1
    low = ahead[cut:].lower()
    hits = [m.start() for m in (re.search(rx, low) for rx in (DIFFER_WORDS, REF_WORDS, BOUND_TERMS,
        PAIR_WORDS)) if m]
    if not hits: return None
    start = cut + min(hits)
    rest = text[at.end():]
    stop = re.search('[.!?;](?:\\s|$)', rest)
    arrive = stop.start() if stop else len(rest)
    alt = re.search('\\(\\s*or\\b[^)]*\\)', rest)
    if alt and alt.start() < arrive: arrive = alt.end()
    end = at.end() + arrive
    phrase = text[start:end].strip(' ,;')
    return (phrase, start, end, info) if phrase else None
  def line_words(self, question, rest):
    if not question: return []
    seen = set()
    for e in self.needs.items() or [{'product': str(self.needs.parsed.get('product') or '')}]:
      seen.update(re.findall('[a-z]+', str(e['product']).lower()))
    ws = re.findall('[a-z0-9]+', str(question).lower())
    ok = [w not in STOP_WORDS and len(w) >= 3 and w not in seen and w.rstrip('s') not in seen for w in ws]
    return [r for n in (3, 2, 1) for i in range(len(ws) - n + 1)
        if all(ok[i:i + n]) and prop_named(r := ' '.join(ws[i:i + n]), rest)]
  class _Split:
    def __init__(self, outer, inner_t):
      self.outer, self.inner_t = outer, inner_t
      self.in_terms, self.spent_terms = set(ref_terms(inner_t)), set(ref_terms(outer))
    @staticmethod
    def label(req):
      return re.split('\\s*\\(', str(req).partition(':')[0])[0].strip()
    def only_inner(self, name):
      return bool(name) and prop_named(name, self.inner_t) and not prop_named(name, self.outer)
    def describes_other(self, req):
      words = set(ref_terms(str(req).partition(':')[2]))
      if self.only_inner(self.label(req)): return True
      return bool(words) and words <= self.in_terms and not words & self.spent_terms
  def _negated(self, line):
    return bool(self._NEGATORS & set(re.findall('[a-z]+', line.lower().replace("n't", ' not'))))

class RefResolve:
  _LOOP_MSG = 'its notes lead back to a listing the chain already visited'
  @staticmethod
  def _tags(head):
    return re.findall('\\(([^)]*)\\)', head)
  @staticmethod
  def _refers(value):
    return bool(is_ref_text(value) or re.search(DIFFER_WORDS, value.lower()))
  @staticmethod
  def _ahead(reply):
    f = reply.get('go_to')
    return f if isinstance(f, dict) and str(f.get('seller') or '').strip() else None
  def _qs_log(self, kind, fields):
    self.log.append({'kind': kind, **fields})
  def ref_fix(self, reply, voiced, question):
    return [self._fix_line(reply, line, question) for line in voiced]
  def _fix_line(self, reply, text, question):
    found = self.ref_phrase(text)
    if not found: return text
    phrase, lo, hi, info = found
    sever = self._Split(text[:lo] + ' ' + text[hi:], text[lo:hi])
    needs, refs = reply.get('needs'), reply.get('references')
    asserted = [r for r in needs if isinstance(r, str) and ':' in r] if isinstance(needs, list) else []
    postings = [e for e in refs if isinstance(e, dict)] if isinstance(refs, list) else []
    prop = self._line_prop(sever, asserted, postings, question, info)
    if not prop:
      self._qs_log('ref_line_unplaced', {'line': text[:200]})
      return sever.outer
    several = self.needs.items()
    tag = self._line_tag(sever, asserted, prop, question, several)
    def same(req):
      tags = self._tags(str(req).partition(':')[0])
      return not (several and tag and tags) or bool(kind_words(tags[-1]) & kind_words(tag))
    own = [r for r in asserted if same_prop(sever.label(r), prop) and same(r)]
    rel = ref_link(phrase)
    who = norm_name(info['seller'])
    reuse = next((r for r in own if self._same_ref(r, rel[0], who)), None)
    head = own[0].partition(':')[0].strip() if own else f'{prop} ({tag})' if tag else prop
    if several and tag and not self._tags(head): head = f'{head} ({tag})'
    need = reuse or f'{head}: {phrase}'
    extra = [r for r in asserted if r not in own and (sever.describes_other(r) or same(r) and self._named_in(
      r.partition(':')[2], who) and not prop_named(sever.label(r), text))]
    slot = asserted.index(own[0]) if own else len(asserted)
    kept = [r for r in asserted if r not in own and r not in extra]
    kept.insert(slot, need)
    reply['needs'] = kept
    self._trim_reply(reply, sever, prop, need, extra, same)
    self._merge_post(postings, prop, info, text, who, rel)
    reply['references'] = postings
    reply['ref_needs'] = [*(reply.get('ref_needs') or []), need]
    self._qs_log('ref_line', {'prop': prop, 'tag': tag, 'relation': rel[0], 'need': need[:200], 'kept': bool(reuse),
                    'details_out': [r[:80] for r in extra]})
    return sever.outer
  def _named_in(self, value, whoever):
    return self._refers(value) and whoever in norm_name(value)
  def _same_ref(self, req, relation, whoever):
    v = req.partition(':')[2]
    return self._named_in(v, whoever) and ref_link(v)[0] == relation
  def _merge_post(self, postings, prop, info, text, who, rel):
    e = next((x for x in postings if same_prop(str(x.get('property') or ''), prop)), None)
    if e is None:
      e = {'property': prop}
      postings.append(e)
    if not (self.ref_seller_ok(e.get('seller'), text) and norm_name(e.get('seller')) == who):
      e.update(seller=info['seller'], kind=info['kind'], details=list(info['details']))
    elif not e.get('details'): e['details'] = list(info['details'])
    e.setdefault('kind', info['kind'])
    relation, tolerance, unit = rel
    if relation != 'same' or not e.get('relation'): e.update(relation=relation, tolerance=tolerance, unit=unit)
  def _line_prop(self, sever, asserted, postings, question, info):
    label = sever.label
    asked = list(filter(None, map(label, str(vars(self).get('asking_label') or '').split(';'))))
    src = [[str(e.get('property') or '') for e in postings], [label(r) for r in asserted
        if self._refers(r.partition(':')[2])],
       [label(r) for r in asserted], asked, sorted(self.store_keys()) + [label(x)
           for x in self.needs.written],
       self.line_words(question, sever.outer)]
    def into(words):
      for s in src:
        if heard := [n for n in s if n and prop_named(n, words)]:
          return max(heard, key=lambda n: (prop_named(n, words, exact=True), len(re.findall('[a-z0-9]+',
              n.lower())), len(n)))
      return ''
    return (into(sever.outer) or into(self._after_lot(sever.inner_t, info))
        or next((n for n in src[0] + src[1] + (asked if len(asked) == 1 else []) if n
            and not sever.only_inner(n)), ''))
  @staticmethod
  def _after_lot(phrase, information):
    low = phrase.lower()
    pos = low.rfind(str(information.get('seller') or '').lower())
    if pos < 0: return ''
    pos += len(str(information.get('seller')))
    for d in information.get('details') or ():
      for w in re.findall('[a-z0-9]+', str(d).lower()):
        if (hit := low.find(w, pos)) >= 0: pos = hit + len(w)
    return phrase[pos:]
  def _line_tag(self, sever, asserted, prop, question, several):
    if not several: return ''
    n = self.prod_of(sever.outer)
    if n is None:
      tags = [t for r in asserted if same_prop(sever.label(r), prop)
          for t in self._tags(r.partition(':')[0])]
      n = self.prod_of(f'x ({tags[-1]})') if tags else None
    if n is None and question: n = self.prod_of(question)
    return '' if n is None else several[n]['product']
  def _trim_reply(self, reply, sever, prop, need, extraneous, same):
    label, written = sever.label, self.needs.written
    now = reply.get('requirements_now')
    if isinstance(now, list):
      kept = [x for x in now if isinstance(x, str) and (x in written or not (
        x in extraneous or sever.describes_other(x) or same_prop(label(x), prop) and same(x)))]
      reply['requirements_now'] = kept if need in kept else kept + [need]
    for key, test in (('dropped', sever.describes_other), ('no_preference', lambda x: same_prop(label(x),
        prop))):
      if isinstance(reply.get(key), list):
        reply[key] = [x for x in reply[key] if not isinstance(x, str) or not (sever.only_inner(label(x))
            or test(x))]
  @staticmethod
  def ref_about(name, paragraph):
    keys = [w for w in prop_words(name) if len(w) >= 3]
    lex = set(re.findall('[a-z]+', str(paragraph).lower()))
    if not keys: return True
    stems = {w[:4] for w in lex if len(w) >= 4}
    return any(k in lex or len(k) >= 4 and k[:4] in stems for k in keys)
  def ref_slip(self, rowid):
    rcr = self.ref_idx().get(rowid) or {}
    facts = {str(n): str(v)[:90] for g in rcr.values() if isinstance(g, dict)
        for n, v in list(g.items())[:40] if isinstance(v, (str, int, float)) and not isinstance(v, bool)}
    cells = self.seller_keys()
    sellers = {str(t) for f, t in rcr.items() if isinstance(t, str) and t.strip() and len(t) <= 60
         and 'id' not in (low := str(f).lower()) and (re.search(SELLER_KEYS, low) or str(f) in cells)
         and not re.search('note|text|desc|comment|detail|info|message|section', low)}
    return as_text({'id': rowid, 'sold by': ' / '.join(sorted(sellers)),
        'category': self.shelf_for(rcr),
               'title': self.title_of(rcr)[:160], 'facts': facts,
               'notes': self.ref_texts(rcr)[:900]}, 5000)
  def ref_ask(self, link, batch, refusal):
    if not self.ref_calls_ok(): return None
    self.ref_calls = vars(self).get('ref_calls', 0) + 1
    cards = '\n'.join(map(self.ref_slip, batch))
    duty = REF_BOUNCE.replace('PROPERTY', link['prop']).replace('DESCRIPTION', self.ref_desc(link))
    ask = f'{duty}\n\nCANDIDATE RECORDS:\n{cards}'
    if refusal: ask += f'\n\nTHE PREVIOUS ANSWER WAS REJECTED: {refusal} Check the records and answer again.'
    reply = self.cable.consult(ask, 3000, CHECKER[:1])
    self._qs_log('ref_hop', {'prop': link['prop'], 'ids': batch, 'answer': reply})
    return reply
  def ref_apply(self, link, batch, reply, retry=True):
    bounce, step = {'link': link, 'batch': batch, 'reply': reply, 'retry_left': retry}, 'read'
    while step != 'end': step = getattr(self, '_sms_hop_' + step)(bounce)
  def _sms_hop_read(self, bounce):
    link, batch, reply = bounce['link'], bounce['batch'], bounce['reply']
    if not isinstance(reply, dict):
      link['fails'] += 1
      if link['fails'] >= 2:
        link['judged'].update(batch)
        self.ref_stop(link, 'the hop check gave no usable answer')
      return 'end'
    named = reply.get('target')
    cand = '' if isinstance(named, bool) or not isinstance(named, (str, int)) else str(named).strip()
    miss = reply.get('missing')
    devoid = [str(x) for x in miss if str(x).strip()] if isinstance(miss, list) else []
    if cand in batch and not devoid:
      bounce['cand'] = cand
      return 'check'
    if kept := self._salvage(link, batch, reply):
      bounce['cand'] = kept
      return 'check'
    link['judged'].update(batch)
    self.ref_ids_in(batch)
    if cand in batch: self._qs_log('ref_partial', {'prop': link['prop'], 'target': cand, 'missing': devoid[:4]})
    return 'end'
  def _salvage(self, link, batch, reply):
    fwd = self._ahead(reply)
    if fwd is None: return ''
    details = link['target']['details']
    index = self.ref_idx()
    hits = []
    for p in batch:
      rcr = index.get(p) or {}
      notes = self.ref_texts(rcr)
      trms = set(ref_terms(notes))
      if (not self._probe(link, p, rcr, notes, trms, fwd)[1] and not self.ref_lacks(rcr, details)
          and all(set(ref_terms(d)) <= trms for d in details)):
        hits.append(p)
    if len(hits) != 1: return ''
    rcr = index.get(hits[0]) or {}
    link.setdefault('loose', set()).update(d for d in details if not self.ref_shows(rcr, d))
    self._qs_log('ref_rescued', {'prop': link['prop'], 'target': hits[0]})
    return hits[0]
  def _sms_hop_check(self, bounce):
    bounce['verdict'] = v = self.ref_verify(bounce['link'], bounce['cand'], bounce['reply'], bounce['batch'])
    if v[1] and self._facts_first(bounce): return 'end'
    return 'retry' if v[1] and bounce['retry_left'] and self.ref_calls_ok() else 'settle'
  def _facts_first(self, bounce):
    link, cand = bounce['link'], bounce['cand']
    if (bounce['verdict'][0] != 'answer' or self._ahead(bounce['reply'])
        or self.row_field(self.ref_idx().get(cand) or {}, link['prop'])):
      return False
    tried = link.setdefault('mapped', set())
    if cand in tried or not self.facts_tool(cand): return False
    tried.add(cand)
    link['facts_due'] = cand
    link['judged'].update(p for p in bounce['batch'] if p != cand)
    self.ref_ids_in(bounce['batch'])
    self._qs_log('ref_facts_due', {'prop': link['prop'], 'target': cand})
    return True
  def _sms_hop_retry(self, bounce):
    bounce['reply'] = self.ref_ask(bounce['link'], bounce['batch'], bounce['verdict'][1])
    bounce['retry_left'] = False
    return 'read'
  def _sms_hop_settle(self, bounce):
    link, batch, cand = bounce['link'], bounce['batch'], bounce['cand']
    where, problem, fwd, value, pairs = bounce['verdict']
    self.ref_ids_in(batch)
    if problem: self._qs_log('ref_invalid', {'prop': link['prop'], 'target': cand, 'problem': problem[:300]})
    if problem and where == 'target':
      link['judged'].add(cand)
      return 'end'
    link['judged'].update(batch)
    if problem:
      link['chain'].append(cand)
      self.ref_stop(link, problem)
    elif cand in link['chain']: self.ref_stop(link, self._LOOP_MSG)
    elif fwd: self.ref_go(link, cand, fwd)
    else: self.ref_clarify(link, cand, value, pairs)
    return 'end'
  def ref_verify(self, link, cand, reply, batch=None):
    rcr = self.ref_idx().get(cand) or {}
    names, values = self.ref_parts(rcr)
    sought = link['target']['details']
    loose = link.get('loose') or set()
    missed = next((d for d in sought if d not in loose and not self.ref_shows(rcr, d, (names, values))),
        None)
    if missed is not None:
      return ('target', f"{cand} does not show '{missed}', so it cannot be the listing described.", None, '', [])
    groups = [set(ref_terms(d)) for d in sought]
    def scope(sid):
      heads, voiced = self.ref_parts(self.ref_idx().get(sid) or {})
      return sum(1 for g in groups if g and g <= heads | voiced)
    rival = max(batch or [cand], key=scope)
    if scope(rival) > scope(cand):
      return ('target', f'{rival} shows more of the named details than {cand} does.', None, '', [])
    notes = self.ref_texts(rcr)
    jt = set(ref_terms(notes))
    if fwd := self._ahead(reply): return self._probe(link, cand, rcr, notes, jt, fwd)
    if msg := self._notes_val(link['prop'], cand, rcr, notes, reply): return ('answer', msg, None, '', [])
    if msg := self._quiet_target(link['prop'], cand, rcr, notes): return ('target', msg, None, '', [])
    return self._check_val(cand, reply, names | values, jt)
  def _notes_val(self, prop, cand, rcr, notes, answer):
    if self.row_field(rcr, prop): return ''
    told = [x for x in re.split(self._SENT_SPLIT, notes) if prop_named(prop, x)]
    got = set(ref_terms(str(answer.get('value') or '')))
    if not told or got and got <= set(ref_terms(' '.join(told))): return ''
    return f'{cand} has no {prop} field, yet its notes do state the {prop}; take the value from those notes, with goes_with listing the partners they give for it.'
  def _quiet_target(self, prop, cand, rcr, notes):
    if self.row_field(rcr, prop) or prop_named(prop, f'{self.title_of(rcr)} {notes}'): return ''
    shelf, read = self.shelf_for(rcr), self.rec_db.read
    peers = [r for p, r in self.ref_idx().items() if p != cand and p in read
        and self.shelf_for(r) == shelf]
    if not peers or any(self.row_field(r, prop) for r in peers): return ''
    return f'{cand} gives no {prop} at all, and neither does any record read from its shelf, so it cannot be the listing described.'
  def _probe(self, link, cand, rcr, notes, jot_trms, forward):
    seller = str(forward['seller']).strip()
    hit = re.compile('(?<![a-z0-9])' + re.escape(seller.lower()) + "(?:'s)?(?![a-z0-9])").search
    if hit(notes.lower()) is None:
      return ('answer', f"{cand}'s notes never mention the go_to seller '{seller}'.", None, '', [])
    lines = re.split('(?<=[.!?])\\s+', notes)
    near = [x for x in lines if hit(x.lower())]
    near += [p for p, x in zip(lines, lines[1:]) if hit(x.lower()) and self._negated(p)]
    if not any(self.ref_about(link['prop'], x) for x in near):
      return ('answer', f'the sentence that names {seller} concerns some other property, not {link["prop"]}.', None, '', [])
    items = forward.get('details')
    kept = [t for x in (items if isinstance(items, list) else []) if not isinstance(x, bool) and isinstance(x,
        (str, int, float))
        and (t := str(x).strip()) and (w := set(ref_terms(t))) and 2 * len(w & jot_trms) >= len(w)]
    aim = {'seller': seller, 'kind': str(forward.get('kind') or '').strip()[:80], 'details': kept[:6]}
    if (norm_name(seller) in self.ref_sellers(rcr)
        and self.ref_key(aim) == self.ref_key(link['target'])):
      return ('answer', f'the go_to points at {cand} itself.', None, '', [])
    return ('', '', aim, '', [])
  @staticmethod
  def _check_val(cand, answer, dossier_lexemes, jot_trms):
    value = str(answer.get('value') or '').strip()
    trms = set(ref_terms(value))
    if not (value and trms and trms <= dossier_lexemes):
      return ('answer', f"the value '{value}' does not appear in {cand}'s record.", None, '', [])
    extra = answer.get('goes_with')
    txt = [str(x).strip() for x in (extra if isinstance(extra, list) else [])
       if isinstance(x, (str, int, float)) and not isinstance(x, bool) and str(x).strip()]
    return ('', '', None, value, [t for t in dict.fromkeys(txt) if (w := set(ref_terms(t))) and w <= jot_trms][:3])
  def ref_go(self, link, cand, forward):
    link['chain'].append(cand)
    notch = self.ref_key(forward)
    if notch in link['seen']: return self.ref_stop(link, self._LOOP_MSG)
    if link['hops'] + 1 >= HANDLE_BOUNCES:
      return self.ref_stop(link,
          f'over {HANDLE_BOUNCES} listings in a row each sent the chain somewhere else')
    link['seen'].add(notch)
    link.update(hops=link['hops'] + 1, target=forward, judged=set(), queries=[], pages=0, tries=0, fails=0,
          unindexed=False, loose=set(), late_from=None)
    self._qs_log('ref_go', {'prop': link['prop'], 'from': cand, 'to': forward})
    self.ref_tell(f'{cand} is the listing {link["head"]} means; its own notes send the reader on to {self.ref_desc(link)}, and the code is following them.')
  def ref_val(self, link):
    read, how = str(link['value']), link['relation']
    subs = [read] + [c for c in list(link.get('goes_with') or []) if c.lower() != read.lower()]
    if how == 'within' and is_num(link.get('tol')):
      if m := re.search('(-?\\d+(?:\\.\\d+)?)\\s*([a-z%]*)', read.lower().replace(',', '')):
        mid, w = float(m.group(1)), float(link['tol'])
        unit = m.group(2) or str(link.get('unit') or '')
        return f'{round(mid - w, 6):g} to {round(mid + w, 6):g} {unit}'.strip() + f' (within {w:g} of {read})'
    if how in ('at most', 'at least'): return f'{how} {read}'
    if how == 'differ': return ' and '.join(f'not {s}' for s in subs)
    if how == 'pairs': return ' or '.join(dict.fromkeys(str(c) for c in link.get('goes_with') or []))
    return ' or '.join(subs)
  def ref_clarify(self, link, cand, read, companions):
    if not str(read or '').strip(): return self.ref_stop(link, f'{cand} states no value for it')
    pairs = link['relation'] == 'pairs'
    if not pairs:
      t = set(ref_terms(read))
      companions = [c for c in companions if set(ref_terms(c)) != t]
    elif not companions:
      return self.ref_stop(link,
          f'{cand} shows {read}, but no store rule telling which value goes with it turned up, so the counterpart stays unknown')
    link['chain'].append(cand)
    link.update(status='resolved', value=str(read).strip(), goes_with=list(companions))
    link['now'] = settled = f'{link["head"]}: {self.ref_val(link)}'
    self.ref_swap(link, settled)
    self.needs.resolved.append(settled)
    self.needs.rev_no += 1
    self.reveal_done(link)
    track = link['chain']
    self._qs_log('ref_resolved', {'req': link['req'][:200], 'now': settled, 'chain': [*track]})
    using = f", reached from {track[0]} by way of the listings' notes" if len(track) >= 2 else ''
    joined, value = ' and '.join(companions), link['value']
    if pairs:
      tail = f'; the store pairs {value} with {joined}. What the shopper wants is that counterpart: {joined} satisfies it' + (
        '' if any(c.lower() == value.lower() for c in companions) else f', while {value} itself does not')
    elif link['relation'] == 'differ':
      tail = (f'; its notes pair {joined} with it' if companions
          else '') + ". The shopper's value MUST NOT EQUAL that one: any other value satisfies it, that one alone does not"
    else: tail = f'; its notes say {value} goes with {joined}, so any of them satisfies it' if companions else ''
    self.ref_tell(f'SETTLED: a linked listing fixes {link["head"]}, and {cand} shows {value}{using}{tail}. The requirement now reads: {settled}.')
  def ref_swap(self, link, settled):
    book = self.needs
    seller = norm_name(link['first']['seller'])
    def mine(req):
      if self.ref_match(link, req): return True
      name, _, value = str(req).partition(':')
      tags = self._tags(name)
      if (tags and link['tag'] and not kind_words(tags[-1]) & kind_words(link['tag'])
          or not seller or seller not in norm_name(value)):
        return False
      return same_prop(re.split('\\s*\\(', name)[0], link['prop'])
    changed = 0
    for attr in ('written', 'uncovered'):
      old = list(getattr(book, attr))
      new = [settled if mine(r) else r for r in old]
      changed += new != old
      setattr(book, attr, list(dict.fromkeys(new)))
    if book.cur_reqs is not None:
      now = [settled if mine(r) else r for r in book.cur_reqs]
      book.cur_reqs = list(dict.fromkeys(now if settled in now else now + [settled]))
    if not changed: book.uncovered.append(settled)
    parsed = book.parsed
    if isinstance(parsed.get('requirements'), list): parsed['requirements'] = list(book.written)
    value = settled.partition(':')[2].strip()
    slots = [s for s in parsed.get('items') or [] if isinstance(s, dict) and isinstance(s.get('requirements'),
        list)]
    for s in slots:
      if not (len(slots) > 1 and link['tag'] and not kind_words(s.get('product')
          or '') & kind_words(link['tag'])):
        s['requirements'] = [str(r).partition(':')[0].strip() + ': ' + value if mine(r) else r
            for r in s['requirements']]
  def ref_stop(self, link, reason_alt):
    if link['status'] != 'open': return
    link.update(status='stuck', why=reason_alt)
    self._qs_log('ref_stuck', {'req': link['req'][:200], 'why': reason_alt[:300], 'chain': [*link['chain']]})
    self.ref_tell(f'UNRESOLVED: {link["head"]} ({reason_alt}). Its value is not known: when the notes of a listing you saw show where the chain ends, use that value; otherwise never decline or reject a listing because of it.')
  def ref_tell(self, note):
    chat = self.chat
    last = chat[-1] if chat else {}
    if chat and len(chat) > 2 and last.get('role') in ('tool', 'user') and isinstance(last.get('content'), str):
      last['content'] += '\n[reference] ' + note
    else: self.ref_told = [*(vars(self).get('ref_told') or []), note]
  _ref_states = {
    'resolved': lambda self, rr: f'{rr["now"]} (read on {rr["chain"][-1]}' + (
      ', must differ from it' if rr.get('relation') == 'differ' else '') + ')' + self._ref_apart(rr),
    'open': lambda self, rr: f'{rr["head"]}: still being traced by the code (currently: {self.ref_desc(rr)})',
    'stuck': lambda self, rr: f'{rr["head"]}: unresolved, value unknown (never decline or reject a listing because of it)'}
  @staticmethod
  def _ref_apart(link):
    details = (link.get('first') or {}).get('details')
    return ' [%s: these describe the pointed-to listing; they are not conditions on the item to buy]' % '; '.join(map(str, details)) if details else ''
  def ref_line(self):
    try:
      shown = [f(self, link) for link in vars(self).get('refs') or []
          if (f := self._ref_states.get(link['status'])) is not None]
      queued = vars(self).pop('ref_told', None) or []
      return ('\n[references] ' + '; '.join(shown) if shown else '') + ''.join('\n[reference] ' + n for n in queued)
    except Exception:
      return ''
  def ref_note(self, item=''):
    try:
      posts, ends, unsure = [], [], False
      for link in vars(self).get('refs') or []:
        status = link['status']
        if status == 'resolved': posts.append(self._done_post(link))
        elif status in ('open', 'stuck'):
          unsure = True
          track = link['chain']
          posts.append(f'- {link["head"]} = OPEN: ' + ('the code reached ' + ' -> '.join(track) if track else 'the code never located the listing that pointer names') + '.')
          ends.extend(r for r in track[-2:] if r != item)
      if not posts: return ''
      text = "\n\nREFERENCE REQUIREMENTS (each value is taken from another listing; the code traced it through the notes in the store's records):\n" + '\n'.join(posts)
      if unsure:
        text += '\nOPEN entries: when a record further down is that listing, or the last listing its chain reached, and gives its own value without pointing elsewhere, that value is the requirement (for a counterpart, the value the store pairs with it); otherwise the value is unknown and no item fails on it alone.'
      tips = [s for r in dict.fromkeys(reversed(ends)) for s in self.rec_db.assoc(r, 1)]
      if tips: text += '\nRECORDS WHERE AN OPEN CHAIN STOPPED:\n' + '\n'.join(tips[:2])
      return text
    except Exception:
      return ''
  @staticmethod
  def _done_post(link):
    track, comps, value = link['chain'], link['goes_with'], link['value']
    using = f", traced from {track[0]} through the listings' notes" if len(track) > 1 else ''
    joined = ' and '.join(comps)
    if link.get('relation') == 'differ':
      tie = f'; its notes also link {joined} to it' if comps else ''
      return f"- {link['now']} = EXCLUSION: the shopper's {link['prop']} must be something other than {value}, the value {track[-1]} shows{using}{tie}. Any other value is fine and that one is not; every listing along the chain ({', '.join(track)}) carries it."
    if link.get('relation') == 'pairs':
      bare = '' if any(str(c).lower() == str(value).lower() for c in comps) else f'; {value} itself does not'
      return f"- {link['now']} = COUNTERPART: a listing that shows {' or '.join(comps)} satisfies it{bare}. {track[-1]} shows {value}{using}, and the store pairs it with {joined}; the shopper's {link['prop']} is that partner value."
    tie = f'; its notes count {joined} as the same as {value}, so any of them satisfies it' if comps else ''
    return f'- {link["now"]} = SETTLED: read from {track[-1]}{using}{tie}.'
  def ceil_guess(self, args, located_entrylines, count, raw):
    paging = self.pg_arg()
    leaf = paging[0] if paging else None
    plain = {n: v for n, v in args.items() if n != leaf}
    seen = vars(self).setdefault('ceil_seen', {})
    seen[json.dumps(plain, sort_keys=True, default=str)] = int(count)
    rows = [r for r in located_entrylines if isinstance(r, dict)]
    if paging and paging[1] is True and rows and isinstance(raw, dict):
      self._note_level(args.get(leaf, paging[2]), paging[2], rows, count, raw)
    _, mode, _, pages = self.search_keys(args)
    top = [n for n, v in args.items() if self._is_cap_arg(n, v, leaf, pages)]
    if len(top) == 1 and rows: self._guess_mode(args, plain, rows, mode, leaf, top[0])
  @staticmethod
  def _is_cap_arg(name, v, leaf, paging):
    return not (isinstance(v, bool) or not is_num(v) or name == leaf or name in paging
        or not v > 1) and re.search(
      'page|size|count|result|min|low|from|start|offset', str(name).lower()) is None
  def _note_level(self, at, origin, hits, total, obs):
    flags = [v for k, v in obs.items() if re.search('next|more|onward|continu|cursor', str(k).lower())
        and not isinstance(v, (list, dict))]
    per = max(len(hits), self.page_len())
    done = bool(flags) and all(f is None or f is False or f == '' for f in flags)
    if is_num(at) and done and len(hits) >= per and total > (depth := int((at - origin + 1) * per)):
      self.page_cap = min(vars(self).get('page_cap') or depth, depth)
  def _guess_mode(self, args, plain, hits, mode, leaf, top):
    ceiling = float(args[top])
    shelves = {self.shelf_for(r) for r in hits}
    lp = [n for n, v in args.items() if isinstance(v, str) and n != leaf and v.strip()
        and v.strip().lower() in shelves]
    shelf = lp[0] if len(lp) == 1 else None
    anchored = dict(d for d in mode if d[0] != shelf)
    sig = json.dumps([sorted(anchored.items()), top], default=str)
    costs = list(map(self.price_of, hits))
    bad = vars(self).setdefault('ceil_bad', set())
    if any(c is not None and c > ceiling + 1e-09 for c in costs): return bad.add(sig)
    if sig in bad or shelf is None or len(shelves) != 1 or '' in shelves or None in costs: return
    def everywhere(text):
      want = text.strip().lower()
      return all(any(isinstance(c, str) and c.strip().lower() == want for f, c in r.items()
             if not self.shelf_for({f: c})) for r in hits)
    tmpl = {n: v for n, v in plain.items() if n not in (top, shelf)
        and not (isinstance(v, str) and v.strip() and n not in anchored and everywhere(v))}
    free = sum(len(v.split()) for n, v in tmpl.items() if isinstance(v, str) and n not in anchored)
    schema = (self.argspec.get(self.role('search') or '') or {}).get('properties') or {}
    prev = vars(self).get('ceil_mode')
    if prev and prev['sig'] not in bad and prev['words'] < free: return
    spec = schema.get(top) or {} if isinstance(schema, dict) else {}
    self.ceil_mode = dict(sig=sig, tmpl=tmpl, mode=anchored, shelf=shelf, bound=top, words=free,
        int=spec.get('type') == 'integer')
  @staticmethod
  def price_of(entry):
    return next((float(v) for n, v in entry.items()
          if not isinstance(v, bool) and is_num(v) and re.search('price|cost', str(n).lower())), None)
  def ceil_cfg(self):
    got = vars(self).get('ceil_mode')
    return None if not got or got['sig'] in vars(self).get('ceil_bad', set()) else got
  def zg_lot(self, call):
    got = self.ceil_cfg()
    if not got or not isinstance(call, dict): return False
    return all(call.get(n) == v for n, v in got['mode'].items()) if got['mode'] else got['bound'] in call
  def page_len(self):
    return max((e.get('size', 0) for e in vars(self).get('page_book', {}).values()), default=0)
  def view_level(self):
    p = self.pg_arg()
    per = self.page_len()
    if p and per > 0 and p[1] != 'token' and p[3] is not None:
      return int((p[3] - p[2] + 1) * per) if p[1] else int(p[3] - p[2] + per)
    return vars(self).get('page_cap')
  def raw_label(self, sought):
    if not sought: return ''
    counts = {}
    ids = self.rec_db.ids
    def attend(knot):
      if not any('id' in str(k).lower() and isinstance(v, str) and v in ids for k, v in knot.items()): return
      for k, v in knot.items():
        low = str(k).lower()
        if (isinstance(v, str) and 0 < len(v.strip()) <= 60 and 'id' not in low
            and re.search('categor|shelf|department|aisle', low)):
          if v.strip().lower() in sought: counts[v.strip()] = counts.get(v.strip(), 0) + 1
          return
    for rcr in self.rec_db.items: walk_dicts(rcr, attend)
    return min(counts, key=lambda t: (-counts[t], t), default='')
  def below(self, volume):
    if not is_num(volume): return None
    whole = (self.ceil_cfg() or {}).get('int')
    m = float(volume)
    v = int(m) - 1 if m.is_integer() else int(m) if whole else round(m - 0.01, 2)
    return None if v <= 0 else v
  def ceil_mid(self, lowerend, upperend):
    if (self.ceil_cfg() or {}).get('int') or all(float(x).is_integer() for x in (lowerend, upperend)):
      return (int(lowerend) + int(upperend)) // 2
    return round((float(lowerend) + float(upperend)) / 2, 2)
  def ceil_args(self, known, shelf, limit):
    return {**known['tmpl'], known['shelf']: shelf,
        known['bound']: int(limit) if known.get('int') or float(limit).is_integer() else limit}
  def prod_slots(self):
    return list(range(n)) if (n := len(self.needs.items())) else [None]
  def cap_ready(self, merchandise):
    if merchandise is None and self.needs.items(): return False
    return (self.needs.priority() == 'cheapest' and merchandise not in vars(self).get('ceil_walked', set())
        and self.ceil_cfg() is not None and self.pg_arg() is not None
        and bool(self.raw_label(self.walk_home(merchandise))))
  def ceil_aim(self, merchandise):
    chosen = self.walk_pick(merchandise)
    held = (vars(self).get('ceil_aims') or {}).get(merchandise)
    notches = [chosen[1]] if chosen else []
    if is_num(held): notches.append(float(held))
    if notches: return min(notches)
    if not self.bound_need(merchandise): return None
    shelves = self.walk_home(merchandise)
    costs = [c for c, _ in self.cheap_unread(self.home_rows(shelves), merchandise)]
    book = self.rec_db
    costs += [float(book.price(s)) for s in book.read if self.known_shelf(s) in shelves
        and is_num(book.price(s))
         and not (s in self.dead_lines() or s in self.refused_lines
             or self.last_stock(s) is False)]
    return min(costs, default=None)
  def bound_need(self, merchandise):
    if not self.needs.items() or merchandise is None: return False
    return any(self.prod_of((p := str(x).partition(':'))[0]) == merchandise and re.search('\\d', p[2])
         and re.search(BOUND_RX, p[2].lower()) for x in self.needs.props_left())

class ValueRows:
  def _trows(self, suggestions=None, ceiling=6):
    found = []
    for rcr in self.rec_db.items if suggestions is None else suggestions:
      walk_dicts(rcr, found.append, ceiling=ceiling)
    return found
  @staticmethod
  def _id_vals(row):
    return [v for n, v in row.items() if isinstance(v, str) and 'id' in str(n).lower()]
  @staticmethod
  def _shelves(walk):
    return walk.get('row_shelf') or {}
  def _above_home(self, walk, home):
    return any(s in home for s in self._shelves(walk).values())
  def _home_ids(self, walk, home):
    return [n for n, s in self._shelves(walk).items() if s in home]
  def _store(self, name, produce):
    return self.__dict__.setdefault(name, produce())
  def _search_form(self):
    return self.argspec.get(self.role('search') or '') or {}
  @staticmethod
  def _query_words(text):
    return [w for w in Errand._tterms(text) if w not in STOP_WORDS and len(w) > 1]
  def _ledger(self):
    return self.__dict__.get('page_book', {})
  @staticmethod
  def _tterms(text, regex='[a-z0-9]+'):
    return re.findall(regex, str(text).lower())
  def _stems(self, words):
    return set(map(self.lemma, words))
  def _has(self, walk, k, subs=False):
    return self.walk_has(walk['args'], k, subs=subs)
  def _headed(self, walk, k):
    found = self._has(walk, k)
    if found or self._tvals(k): return found
    return {'bound'} if self._bound_head(walk, k) else set()
  def _bound_head(self, walk, k):
    words = [w for t in self.arg_texts(walk['args']) for w in self._alnum_rx.findall(t.lower())]
    tokens = words[:self.lead_words or None]
    return any(re.search(BOUND_RX, m) and (f := self.fig_qty(re.sub(BOUND_RX, ' ', m))) and f[0] in tokens
         and any(x == f[1] or self._close(x, f[1]) for x in tokens)
         for _, req in self._own_reqs(k) for m in self._prefs(self._bare_val(req)))
  def _id_walk(self, walk):
    return walk.get('id_order') is True and not self.zg_lot(walk['args'])
  def _own_reqs(self, k):
    rules = self.needs.props_left()
    if not self.needs.items() or k is None: return list(enumerate(rules))
    return [(i, r) for i, r in enumerate(rules) if self.prod_of(r.partition(':')[0]) in (k, None)]
  @staticmethod
  def _bare_val(req):
    return re.sub('\\((?:stated|revealed[^)]*)\\)', '', req.partition(':')[2]).lower()
  @staticmethod
  def _prefs(value):
    return re.split('[,;/]|\\bor\\b|\\beither\\b|\\bnor\\b', value)
  @staticmethod
  def _cdiv(digitval, denominator):
    return -(-digitval // denominator)
  def home_rows(self, home):
    ids = self.rec_db.ids
    return {n for r in self._trows() if (n := self.row_key(r)) in ids
        and self.shelf_for(r) in home}
  def ceil_clash(self, home, bound):
    ids = self.rec_db.ids
    return any(self.shelf_for(r) in home and any(v in ids for v in self._id_vals(r))
         and (p := self.price_of(r)) is not None and p <= bound and self.stock_state(r) is True
         for r in self._trows())
  def _zv_grab(self, run, k, walk, constructed):
    depth = run['depth']
    pages = self._cdiv(walk['total'] if depth is None else min(walk['total'], depth), max(1,
        walk['size'])) - len(walk['read'])
    used = run['used'].get(k, 0)
    ok = min(run['per'] - used, run['cap'] - sum(run['used'].values()))
    if run['several']:
      ok = min(ok, run['share'] - used - run['opened'].get(k, 0) - (0 if run['opened'].get(k) else run['opens']))
    return pages if pages <= ok else ok if constructed else 0
  def _clear_pool(self, merchandise, book, booked, count, apiece):
    home = self.walk_home(merchandise)
    ids = set()
    for key in booked.get(str(merchandise), []):
      if key in book and self._headed(book[key], merchandise): ids.update(self._home_ids(book[key], home))
    pick = self.walk_pick(merchandise)
    ranked = [n for _, n in self.cheap_unread(ids, merchandise, pick and pick[1])]
    return ranked[:max(0, min(apiece, CEIL_OPENS - count.get(merchandise, 0)))]
  def cheap_unread(self, ids, k, below=None):
    stash = set(ids)
    if not stash: return []
    lines = self.needs.items()
    units = self.needs.quantity() if k is None or k >= len(lines) else lines[k]['quantity']
    cap = self.needs.spend_cap
    peaked = is_num(cap) and is_num(units)
    sold = self._last_sold(stash)
    book = self.rec_db
    return sorted((float(p), n) for n, p in ((n, book.price(n)) for n in stash)
           if not (n in book.read or n in self.refused_lines or sold.get(n)) and is_num(p)
           and not (peaked and p * units > cap) and (below is None or p < below))
  def _last_sold(self, stash):
    return {self.row_key(r): s is False for r in self._trows()
        if any(v in stash for v in self._id_vals(r))
            and (s := self.stock_state(r)) is not None}
  def all_words(self):
    return self.word_kind == 'all' or self.word_kind != 'any' and bool(self.lead_words)
  def counted_ids(self, k):
    if not self.all_words(): return set()
    home = self.walk_home(k)
    return {n for w in self._ledger().values() if self._id_walk(w) and self._has(w, k)
        for n in self._home_ids(w, home)}
  def counted_all(self, k):
    try:
      if not self.all_words(): return False
      home = self.walk_home(k)
      solo = {i for i, r in self._own_reqs(k) if len([s
          for s in self._prefs(self._bare_val(r)) if s.strip()]) == 1}
      return any(self._id_walk(w) and w.get('first') and self.next_page(w) is None
           and len(self._shelves(w)) >= w['total'] and self._above_home(w, home)
               and self._has(w, k) & solo
           for w in self._ledger().values())
    except Exception:
      return False
  def row_has(self, symbol, k):
    groups, sums = [], []
    for _, req in self._own_reqs(k):
      value = self._bare_val(req)
      if is_ref_text(value): continue
      for s in self._prefs(value):
        if v := self.fig_qty(s): sums.append(v)
        elif not self.qty_part(s) and (stems := [self.lemma(w) for w in re.findall('[a-z0-9]{3,}', s)]):
          groups.append(stems)
    if not groups and not sums: return False
    body = ' '.join(str(v).lower() for r in self._trows() if symbol in self._id_vals(r) for v in r.values() if isinstance(v, str))
    if any(re.search('(?<![\\d.,])%s\\s*-?\\s*%s' % (re.escape(f), re.escape(self.lemma(u))), body) for f,
        u in sums):
      return True
    have = {self.lemma(w) for w in re.findall('[a-z0-9]{3,}', body)}
    return any(set(s) <= have for s in groups)
  def _spare(self):
    burst = max(self.ask_room() - self.msgs_sent, 0) if self.asking and self.role('message') else 0
    return (burst, self.steps_rem() - self.order_due() - burst * 2)
  def _item_words(self, morsel, own):
    volume = self.fig_qty(morsel)
    if re.search(NEG_WORDS, morsel) or self.qty_part(morsel) and not volume: return []
    lead, unit = volume if volume else (None, None)
    text = re.sub('\\b(?:' + BOUND_WORDS + ')\\b', ' ', re.sub("'s\\b", '', morsel))
    return [w for w in re.findall('\\d+(?:[.,]\\d+)+|[a-z0-9]+', text) if '.' not in w and ',' not in w
        and w not in STOP_WORDS and (len(w) >= 2 or w == lead) and (w not in own or w == unit)]
  def _tvals(self, k):
    found = []
    for head, value in self.need_vals(k):
      low = re.sub('\\([^)]*\\)', ' ', str(value).lower().replace('’', "'")).strip()
      if not low or is_ref_text(low): continue
      own = set(self._tterms(head))
      if re.search('\\bor\\b|\\beither\\b|/', low) is not None:
        parts = [s for s in re.split('[,;/]|\\bor\\b|\\beither\\b', low) if s.strip()][:3]
      else: parts = re.split('[,;&+]|\\band\\b', low)[:1]
      opts = []
      for m in parts:
        if (w := self._item_words(m, own)) and w not in opts: opts.append(w)
      if opts and opts not in found: found.append(opts)
    return found
  def _label_words(self, k, label):
    words = list(dict.fromkeys(self._query_words(re.sub("'s\\b", '', label.lower().replace('’', "'")))))
    if self.__dict__.get('label_indexed') in (True, None): return words
    home = self.walk_home(k)
    ids = self.rec_db.ids
    heads = [self._stems(self._tterms(self.title_of(r))) for r in self._trows()
        if self.shelf_for(r) in home and self.row_key(r) in ids]
    if heads and (freq := [w for w in words if 2 * sum(self.lemma(w) in t for t in heads) >= len(heads)]):
      return freq
    lines = self.needs.items()
    kind = (lines[k]['product'] if lines and k is not None and k < len(lines)
        else self.needs.parsed.get('product') or '')
    return [w for w in self._tterms(kind) if len(w) > 2 and w not in STOP_WORDS][-1:]
  def _forms(self, k, label, cap_len):
    values = self._tvals(k)
    if not values: return ([], [])
    tail = self._label_words(k, label)
    cut = None if self.needs.priority() == 'cheapest' else 1
    def build(chosen):
      lead = {}
      for sel in chosen + [tail]:
        for w in sel: lead.setdefault(self.lemma(w), w)
      text = ' '.join(list(lead.values())[:self.lead_words or None])
      if not is_num(cap_len): return text
      while ' ' in text and len(text) > cap_len: text = text.rsplit(' ', 1)[0]
      return text[:int(cap_len)]
    uniq = lambda xs: list(dict.fromkeys(filter(None, xs)))
    solo = uniq([build([s]) for s in values[0][:cut]])
    if len(values) == 1: return (solo, [])
    wide = next((i for i, o in enumerate(values) if len(o[:cut]) > 1), 0)
    heads = [o[0] for o in values]
    return (uniq([build(heads[:wide] + [s] + heads[wide + 1:]) for s in values[wide][:cut]]), solo)
  def _next_form(self, first, second, tkey, mode):
    for forms in (first, second):
      totals = [self._sent_sum(f, tkey, mode) for f in forms]
      for f, t in zip(forms, totals):
        if t is None: return f
      if any(totals): return ''
    return None if first or second else ''
  def enumsig(self, args):
    fields = self._search_form().get('properties')
    fields = fields if isinstance(fields, dict) else {}
    return tuple(sorted((n, v) for n, v in args.items() if isinstance(v, str) and isinstance(s := fields.get(n),
        dict)
              and isinstance(s.get('enum'), list) and len(s['enum']) <= 4))
  def _lemmas(self, text):
    return frozenset(map(self.lemma, re.findall('[a-z0-9]+', str(text).lower())[:self.lead_words or None]))
  def _sent_sum(self, text, tkey, mode):
    target = self._lemmas(text)
    for sent, total in (self.__dict__.get('ceil_seen') or {}).items():
      try:
        args = json.loads(sent)
      except Exception:
        continue
      if (isinstance(args, dict) and isinstance(v := args.get(tkey), str)
          and all(args.get(n) == r for n, r in mode.items()) and self._lemmas(v) == target):
        return int(total)
    return None
  def _note_index(self, label):
    if self.__dict__.get('label_indexed') is not None or not self.rec_db.items: return
    sought = self._stems(self._query_words(re.sub("'s\\b", '', label.lower())))
    shelf = label.strip().lower()
    ids = self.rec_db.ids
    rows = [r for r in self._trows([self.rec_db.items[-1]], 3) if self.row_key(r) in ids
        and self.shelf_for(r) == shelf]
    if sum(1 for r in rows if sought - self._stems(w for n, v in r.items() if isinstance(v, str)
        and not re.search(
        'categor|shelf|department|aisle', str(n).lower()) for w in self._tterms(v))) > 1:
      self.label_indexed = True
  def _pare(self, queue, batch, floor):
    while queue and (1 if batch and len(queue) >= 2 else len(queue)) >= self.steps_rem() - floor: queue.pop()
    return queue
  def _rated(self, queue):
    return ', '.join(f'{n} ({self.rec_db.price(n):g})' for n in queue)
  def _append(self, text, status):
    chat = self.chat
    if text and chat and chat[-1].get('role') == 'tool':
      extra = '\n' + self.state_line() if status else ''
      chat[-1]['content'] = str(chat[-1]['content']) + '\n' + text + extra
  def zl_queue(self, k, ids, floor, utmost):
    cheap = self.needs.priority() == 'cheapest'
    pick = self.walk_pick(k) if cheap else None
    batch = self.batch_tool()
    count = self._store('valued_opens', dict)
    ranked = [n for _, n in self.cheap_unread(ids, k, pick[1] if pick else None)]
    quota = CEIL_OPENS - count.get(k, 0) if cheap else utmost
    queue = self._pare(ranked[:max(0, min(batch[2] if batch else utmost, quota))], batch, floor)
    if cheap: count[k] = count.get(k, 0) + len(queue)
    return queue
  def zl_start(self, queue, abodes, floor, say):
    self.rank_checked = self.rank_checked - {n for n in self.rank_checked
        if self.known_shelf(n.partition('|')[0]) in abodes}
    self.order_vetted = False
    self.open_queue, self.open_recheck, self.open_say = queue, True, say
    self.open_purpose, self.open_floor = 'walk', floor
    self.open_named()
  def zl_find(self):
    if self._spare()[1] <= 4: return False
    schema = self._search_form()
    props = schema.get('properties')
    d = self.__dict__
    mutual = {'schema': schema, 'props': props if isinstance(props, dict) else {}, 'found': self.pg_arg(),
         'tries': d.setdefault('label_tries', {}), 'settled': d.setdefault('label_done', {}),
         'sets': d.setdefault('label_sets', {}), 'products': self.prod_slots()}
    return any(self._vs_step(dict(mutual, k=k)) for k in mutual['products'])
  def _vs_step(self, a):
    k = a['k']
    if a['tries'].get(k, 0) >= LABEL_TRIES: return False
    a['sig'] = json.dumps(self._tvals(k))
    if a['sig'] in a['settled'].setdefault(str(k), []): return False
    home = self.walk_home(k)
    if not (label := self.raw_label(home)): return False
    a['label'] = label
    if self._vs_schema(a) == 'forms':
      todo = self._vs_plan(a)
      if todo is None:
        todo = ''
        if any(self._has(w, k) and self._above_home(w, home)
            for w in self._ledger().values()) and (
            self.__dict__.get('label_indexed') is None):
          self.label_indexed = False
          todo = self._vs_plan(a) or ''
      if todo:
        args, faults = ArgCheck.vet({**a['tmpl'], a['tkey']: todo}, a['schema'])
        if not faults and isinstance(args, dict):
          a['tries'][k] = a['tries'].get(k, 0) + 1
          return self._vs_emit(a, k, todo, args, home)
    a['settled'][str(k)].append(a['sig'])
    return False
  def _vs_schema(self, assignment):
    for call in self.code_queries(assignment['k']):
      txt = self.arg_texts(call)
      keys = [n for n, v in call.items() if isinstance(v, str) and v.strip() and v in txt]
      if len(keys) == 1 and not self.zg_lot(call): break
    else:
      return 'settle'
    tkey = keys[0]
    props = assignment['props']
    assignment['mode'] = mode = {n: v for n, v in call.items() if isinstance(props.get(n), dict)
        and props[n].get('enum')}
    need = assignment['schema'].get('required') or []
    assignment['tmpl'] = {n: v for n, v in call.items() if n == tkey or n in mode or n in need}
    assignment['tkey'] = tkey
    assignment['cap_len'] = props[tkey].get('maxLength') if isinstance(props.get(tkey), dict) else None
    return 'forms'
  def _vs_plan(self, a):
    a['first'], a['second'] = first, second = self._forms(a['k'], a['label'], a['cap_len'])
    a['sets'].setdefault(str(a['k']), set()).update(self._lemmas(f) for f in first + second)
    return self._next_form(first, second, a['tkey'], a['mode'])
  def _vs_emit(self, a, k, todo, args, home):
    self.walk_call(args, f"Sending '{todo}' as a keyword search: the required value{'s' if len(self._tvals(k)) > 1 else ''} first, then the category name as the store spells it. The search looks at titles, categories and listing fields, so the reply shows in-category items carrying that value.")
    total = self._sent_sum(todo, a['tkey'], a['mode'])
    self._note_index(a['label'])
    found = a['found']
    trimmed = dict(args)
    if found: trimmed.pop(found[0], None)
    key = json.dumps(trimmed, sort_keys=True, default=str)
    walk = self._ledger().get(key)
    every = bool(walk) and len(self._shelves(walk)) >= walk['total'] and self._above_home(walk, home)
    floor = self.order_due() + 2
    if total is None: note = ''
    elif not total: note = f"[value search] '{todo}': no matches at all."
    else:
      note = f"[value search] '{todo}': {total} match(es)" + (', every one of them on this page.' if every else '.')
    queue, most = [], None
    if every:
      if self.needs.priority() == 'cheapest':
        if found and walk.get('id_order') is not False:
          self._store('valued', set).add(key)
          self._store('valued_keys', dict).setdefault(str(k), []).append(key)
          if len(a['products']) > 1: self._store('valued_wait', dict)[k] = True
          else: most = CEIL_OPENS
      elif todo in (a['first'] or a['second']): most = LABEL_OPENS
    if most is not None: queue = self.zl_queue(k, self._home_ids(walk, home), floor, most)
    if queue:
      below = ' below the pick' if self.walk_pick(k) and self.needs.priority() == 'cheapest' else ''
      note += ' Opened next, cheapest first: the still-unread, in-stock, within-budget items of the requested category' + below + ': ' + self._rated(queue) + '.'
    self._append(note, not queue)
    self.log.append({'kind': 'label_search', 'product': k, 'query': todo, 'total': total, 'every': every,
        'opened': queue})
    if queue:
      self.zl_start(queue, home, floor,
          'Next I read the cheapest unread hits; each category item having the required values is among these matches.')
    return True
  def zl_clear(self, floor):
    flags = self.__dict__.setdefault('valued_wait', {})
    due = [m for m, f in flags.items() if f]
    flags.clear()
    if not due: return False
    book = self._ledger()
    booked = self.__dict__.get('valued_keys') or {}
    count = self.__dict__.setdefault('valued_opens', {})
    batch = self.batch_tool()
    apiece = 2 if self.steps_rem() - floor - 1 >= 2 * len(due) else 1
    pools = [self._clear_pool(m, book, booked, count, apiece) for m in due]
    limit = batch[2] if batch else apiece * len(due)
    owner, queue, turn = {}, [], 0
    while turn or (len(queue) < limit and any(pools)):
      stash = pools[turn]
      while stash and stash[0] in queue: stash.pop(0)
      if stash and len(queue) < limit:
        queue.append(stash.pop(0))
        owner[queue[-1]] = due[turn]
      turn = (turn + 1) % len(due)
    if not self._pare(queue, batch, floor): return False
    for n in queue: count[owner[n]] = count.get(owner[n], 0) + 1
    self.zl_start(queue, set().union(*map(self.walk_home, due)), floor,
        'The shopper asked for the cheapest: every product now has the lowest-priced unread hits of its value search read.')
    self._append('[value searches read] Reading now, for each product and under its pick, the unread in-stock hits inside the budget from its value search, cheapest first - ' + self._rated(queue) + '. For each product choose the lowest-priced listing whose own fields meet every requirement now in force; a pricier fit first needs every cheaper candidate opened and turned down.', False)
    self.log.append({'kind': 'value_flush', 'products': due, 'opened': queue})
    return True
  def prod_steps(self):
    try:
      lines = self.needs.items()
      if len(lines) < 2 or self.needs.priority() != 'cheapest': return ''
      pages = self.__dict__.get('valued_by') or {}
      parts, devoid = [], ''
      for k, line in enumerate(lines):
        owns = bool(self.walk_pick(k))
        name = str(line['product'])
        parts.append(f'{name}: {pages.get(k, 0)} value page(s) read, pick {("no", "yes")[owns]}')
        if not owns and not devoid: devoid = name
      return '\n[product steps] ' + '; '.join(parts) + (f'; next: {devoid}' if devoid else '') + '.'
    except Exception:
      return ''
  def value_pages(self):
    periods = {'gate': self._zv_gate, 'label': self._zv_label, 'setup': self._zv_setup,
         'scan': self._zv_scan, 'pick': self._zv_pick}
    run, stage = {'result': False}, 'gate'
    try:
      while stage in periods: stage = periods[stage](run)
    except Exception as mistake:
      self.log.append({'kind': 'value_pages_error', 'error': ('%s: %s' % (type(mistake).__name__,
          mistake))[:300]})
      return False
    return run['result']
  def _zv_gate(self, run):
    if any(h.get('status') == 'open' for h in self.__dict__.get('refs') or []) and self.ref_room():
      return 'stop'
    cable = self.cable
    if (self.orders_tried or self.refusal_n or cable.done or cable.broken or cable.spare() < CLOSING + 60
        or not self.role('search')):
      return 'stop'
    return 'label'
  def _zv_label(self, run):
    if self.zl_find():
      run['result'] = True
      return 'stop'
    if self.needs.priority() != 'cheapest' or self.word_kind == 'any': return 'stop'
    run['found'] = self.pg_arg()
    return 'setup' if run['found'] else 'stop'
  def _zv_setup(self, run):
    store = self.__dict__
    run.update(book=store.get('page_book', {}), done=self._store('valued', set),
         used=self._store('valued_by', dict), opened=self._store('valued_opens', dict),
         wait=self._store('valued_wait', dict), sets=store.get('label_sets') or {},
             depth=self.view_level())
    depth, per, products = run['depth'], max(1, self.page_len()), self.prod_slots()
    n = len(products)
    run.update(per=max(VALUE_PAGES, self._cdiv(depth, per) - 1) if depth else VALUE_PAGES, products=products,
         several=n > 1, cap=max(VALUE_TOTAL, n * VALUE_TOTAL_PER),
         opens=1 if self.batch_tool() else 2 if n > 1 else CEIL_OPENS,
             floor=2 + self.order_due(),
         known=self.all_words())
    run['left'], run['room'] = self._spare()
    fresh = sum(1 for k in products if self.walk_home(k) and not run['used'].get(k))
    run['share'] = max(4, run['room'] // (fresh or 1))
    return 'scan'
  def _zv_scan(self, r):
    r['options'], r['pending'] = [], False
    for pos, k in enumerate(r['products']):
      for t, g, key in self._zv_prod(r, k, home) if (home := self.walk_home(k)) else ():
        r['options'].append((r['used'].get(k, 0), t, g, pos, key, k))
    return 'pick'
  def _zv_pick(self, r):
    if r['options']:
      _, take, _, _, key, k = min(r['options'], key=lambda o: o[:5])
      r['done'].add(key)
      r['result'] = self.val_search(k, key, r['floor'], take, r['several'])
    elif r['wait'] and not r['pending'] and not (r['left'] and r['room'] > r['opens'] * len(r['wait']) + 4):
      r['result'] = self.zl_clear(r['floor'])
    return 'stop'
  def _zv_prod(self, r, k, home):
    lem = lambda txt: {self.lemma(w) for w in re.findall('[a-z0-9]{2,}', txt)}
    m = {'k': k, 'home': home, 'mine': r['sets'].get(str(k)) or set()}
    m['own'] = lem(' '.join(q.partition(':')[2].lower() for _, q in self._own_reqs(k))) | lem(' '.join(home))
    m['whole'] = [(self._lemmas(' '.join(self.arg_texts(w['args']))), self._has(w, k, subs=True),
           self.enumsig(w['args'])) for w in r['book'].values()
           if w.get('id_order') is True and w.get('first') and self.next_page(w) is None
           and not self.zg_lot(w['args']) and len(self._shelves(w)) >= w['total']]
    alt = []
    for key, walk in r['book'].items():
      if (p := self._zv_mull(r, m, key, walk)) == 'burst': r['pending'] = True
      elif p: alt.append(p)
    if r['known']:
      o = m['own']
      alt = [e for e in alt if not any(c[3] < e[3] and not e[3] - c[3] <= o or e[3] < c[3] and c[3] - e[3] <= o
                      for c in alt)]
    return [e[:3] for e in alt]
  def _zv_mull(self, r, m, key, walk):
    k, known, depth = m['k'], r['known'], r['depth']
    if (key in r['done'] or walk.get('id_order') is not True or not walk.get('first')
        or self.next_page(walk) is None or self.zg_lot(walk['args']) or not self._headed(walk, k)
        or not self._above_home(walk, m['home'])):
      return None
    words = self._lemmas(' '.join(self.arg_texts(walk['args'])))
    built = words in m['mine']
    if not known if depth is None else walk['total'] > depth and not (known and built): return None
    pairs, manner = self._has(walk, k, subs=True), self.enumsig(walk['args'])
    if known and any(md == manner and (seen <= words or pairs and seen <= m['own'] and pairs <= got)
            for seen, got, md in m['whole']):
      return None
    take = self._zv_grab(r, k, walk, built)
    if take <= 0 or take + r['opens'] + r['floor'] >= self.steps_rem(): return None
    if (take > 2 or r['several']) and r['left'] and r['room'] - take - r['opens'] > 4: return 'burst'
    return take, -self.walk_score(walk['args'], k), key, words
  _skip_mark = object()
  _cheap_rule = 'Choosing: the lowest-priced offer whose own record satisfies every requirement in force wins; before any pricier match, each cheaper candidate has to be opened and rejected.'

class ValueSearch:
  _cat_rx = re.compile('categor|shelf|department|aisle')
  _smu_cap_names = re.compile('ceil|max|price|cap$|budget|upper|most')
  _smu_not_cap = re.compile('page|first|offset|skip|count|size|num')
  _not_cap_rx = re.compile('page|size|count|result|min|low|from|start')
  _nomore_rx = re.compile('\\bno (more|less) than\\b')
  _over_rx = re.compile('\\b(?:at least|or more|or higher|or larger|minimum|more than|over|above)\\b')
  _under_rx = re.compile('\\b(?:at most|up to|or less|or lower|or smaller|maximum|less than|under|below)\\b')
  _strict_rx = re.compile('\\b(?:more than|less than|over|above|under|below)\\b')
  @staticmethod
  def _qu_sig(call): return json.dumps(call, sort_keys=True, default=str)
  def _memo(self, name, produce): return vars(self).setdefault(name, produce())
  def _peek(self, name, empty=None): return vars(self).get(name, empty)
  def _qu_err(self, kind, e): self.log.append({'kind': kind, 'error': self._verr(e)})
  def _wants_cheap(self): return self.needs.priority() == 'cheapest'
  def _ucost(self, s): return float(raw) if is_num(raw := self.rec_db.price(s)) else None
  @staticmethod
  def _row_shelf(walk): return walk.get('row_shelf') or {}
  def _can_list(self, floor):
    c = self.cable
    return not (c.done or c.broken or c.spare() < CLOSING + 30) and self.steps_rem() - 1 > floor
  def _lowtime(self): return self.cable.done or self.cable.spare() < CLOSING + 60
  def _opens(self): return 1 if self.batch_tool() else 2
  @staticmethod
  def _lift(table, key, value):
    if table.setdefault(key, 0) < value: table[key] = value
  def _note(self, kind, **fields): self.log.append({'kind': kind, **fields})
  def _tail(self):
    return tail if self.chat and (tail := self.chat[-1]) is not None and tail.get('role') == 'tool' else None
  def _add_text(self, text, status=False):
    if (tail := self._tail()) is not None:
      extra = text + '\n' + self.state_line() if status else text
      tail['content'] = '%s\n%s' % (tail['content'], extra)
  def _pages(self, walk, base, slot, say, floor, room, each=None):
    n = got = 0
    while n < room and self.next_page(walk) is not None and self._can_list(floor):
      n += 1
      if each is not None: each()
      pin = self.next_page(walk)
      if not self.walk_call({**base, slot: pin}, say): break
      got += 1
      if pin == self.next_page(walk): break
    return n, got
  def val_search(self, k, key, floor, utmost=VALUE_PAGES, postpone=False):
    slot, walk = self.pg_arg(), self._peek('page_book', {}).get(key)
    if not slot or not walk: return False
    query = ' '.join(self.arg_texts(walk['args']))
    per, every = self._memo('valued_by', dict), self.all_words()
    say = ("Reading the rest of '%s' page by page (results come in id order, not by price, and a text query here must match every word) so no offer carrying those words is missed - the cheapest is wanted." if every else
       "Reading the rest of '%s' page by page (results come in id order, not by price) so no offer it matched is missed - the cheapest is wanted.") % query
    bump = lambda: per.__setitem__(k, per.get(k, 0) + 1)
    tried, got = self._pages(walk, walk['args'], slot[0], say, floor, utmost, bump)
    if not got: return tried > 0
    self._memo('valued_keys', dict).setdefault(str(k), []).append(key)
    n, fin = len(self._row_shelf(walk)), self.next_page(walk) is None
    complete, home = fin and walk['total'] <= n, self.walk_home(k)
    tail = ((' (all of it, for those words)' if every else ' (all of it)') if complete else
        ' (later pages out of reach: a cheaper offer may sit beyond them)' if fin else '')
    head, opened = "[value search read] '%s' -> %d/%s%s. " % (query, n, walk['total'], tail), []
    if postpone:
      self._memo('valued_wait', dict)[k] = True
      body = 'Held back: once the value search of every product has been read, the cheapest unread entries of each (in stock, within budget) are opened. '
    else:
      pick = self.walk_pick(k)
      opened = self.zl_queue(k, [s for s, shelf in self._row_shelf(walk).items() if shelf in home], floor,
                   CEIL_OPENS)
      rated = ', '.join('%s (%s)' % (s, format(self.rec_db.price(s), 'g')) for s in opened)
      below = ', all cheaper than the pick' if pick else ''
      body = ('Opening now (requested category, cheapest first%s): %s. ' % (below, rated) if opened else
          'Nothing to open - each entry of the requested category is already read, sold out or over budget. ')
    if opened: self.zl_start(opened, home, floor, 'Opening the cheapest unread entries of that query.')
    self._add_text(head + body + self._cheap_rule, not opened)
    self._note('value_pages', product=k, query=query, pages=got, complete=complete, opened=opened,
           postpone=postpone)
    return True
  def ceiling_walk(self, merchandise, floor, repair=False):
    k, spec, slot = merchandise, self.ceil_cfg(), self.pg_arg()
    home = self.walk_home(k)
    label, aim = self.raw_label(home), self.ceil_aim(k)
    if aim is None or not (spec and slot and label): return None
    self._memo('ceil_walked', set).add(k)
    pick = self.walk_pick(k)
    self._memo('ceil_aimed', dict)[k] = (float(aim), pick[0] if pick else '')
    if (limit := self.below(aim)) is None: return None
    if repair: floor += 2
    elif not self.orders_tried: floor = max(self.order_due() + 2, floor)
    w = 'the pick' if pick or is_num((self._peek('ceil_aims') or {}).get(k)) else 'the cheapest row seen'
    say = "Listing category '%s' under the store's price cap, below %s (%s) - the cheapest is wanted." % (label, w, format(aim, 'g'))
    depth, book = self.view_level(), self._memo('page_book', dict)
    keys, calls, top, complete, ran, bound, last = [], 0, None, False, False, limit, limit
    while calls < CEIL_CALLS and self._can_list(floor):
      ck = self._qu_sig(call := self.ceil_args(spec, label, bound))
      self._memo('ceil_keys', set).add(ck)
      calls += 1
      reuse = ck in book and book[ck].get('first') and ck in (self._peek('ceil_seen') or {})
      if not reuse and not self.walk_call(call, say): break
      ran, last = True, bound
      keys.append(ck)
      walk = book.get(ck)
      if walk and any(is_num(c) and c > bound for c in map(self.rec_db.price, self._row_shelf(walk))):
        self._memo('ceil_bad', set).add(spec['sig'])
      if self.ceil_cfg() is None or (total := self._peek('ceil_seen', {}).get(ck)) is None: break
      if not total or not walk:
        if self.ceil_clash(home, bound):
          self._memo('ceil_bad', set).add(spec['sig'])
          break
        if top is None:
          complete = True
          break
        mid = self.ceil_mid(bound, top)
        if not bound < mid < top: break
        bound = mid
        continue
      pages = -(-total // max(1, walk['size']))
      if (depth is None or depth >= total) and CEIL_CALLS - calls >= pages - len(walk['read']):
        calls += self._pages(walk, call, slot[0], 'Reading the next page (price-capped listing).', floor,
                    CEIL_CALLS - calls)[0]
        complete = self.next_page(walk) is None
        break
      top, low = bound, min([c for c in map(self._ucost, self._row_shelf(walk)) if c is not None], default=0.0)
      spread = low + 1.8 * (top - low) * (max(1, walk['size']) * max(1, CEIL_CALLS - calls)) / max(total, 1)
      nxt = min(self.ceil_mid(low, top), self.ceil_mid(low, spread))
      if not nxt < top: break
      bound = nxt
    if not ran: return None
    found = {s for key in keys for s, shelf in self._row_shelf(book.get(key, {})).items() if shelf in home}
    cdts, valued = [s for _, s in self.cheap_unread(found, k, aim)], self.counted_ids(k)
    rest = [s for s in cdts if s not in valued]
    borne = [s for s in rest if self.row_has(s, k)]
    tail = [] if self.counted_all(k) else [s for s in rest if s not in borne]
    top, a = ([s for s in cdts if s in valued] + borne + tail)[:CEIL_OPENS], format(aim, 'g')
    report = ("cap listing '%s' (< %s): %d cheaper rows; can be opened (unread, in stock, within budget): %d" % (label, a, len(found), len(cdts)) if not complete else
         "cap listing '%s': read in full down to %s, %d found; openable (unread, in stock, in budget): %d" % (label, format(last, 'g'), len(found), len(cdts)) if found else
         "cap listing '%s' (< %s): no listing costs less than %s" % (label, a, w))
    self._note('ceiling_walk', product=k, label=label, aim=aim, calls=calls, complete=complete, rows=len(found),
           prioritized=top)
    return report, top
  def cap_sweep(self, tried):
    spec, slot = self.ceil_cfg(), self.pg_arg()
    if not (spec and slot) or self.needs.items(): return None, False
    rows, shelves = self.shelf_rows(), self.shelf_labels()
    label = self.raw_label(shelves)
    if not rows or not label: return None, False
    if (cap := self.below(min(c for c, _, _ in rows.values()))) is None: return None, True
    ck = self._qu_sig(call := self.ceil_args(spec, label, cap))
    if ck not in tried:
      self._memo('ceil_keys', set).add(ck)
      return call, False
    if not (walk := self._peek('page_book', {}).get(ck)):
      return None, self._peek('ceil_seen', {}).get(ck) == 0 and not self.ceil_clash(set(shelves), cap)
    if (nxt := self.next_page(walk)) is None: return None, walk['total'] <= len(self._row_shelf(walk))
    more = {**call, slot[0]: nxt}
    return (None, False) if self._qu_sig(more) in tried else (more, False)
  def val_guard(self, name, sent):
    try:
      if name != self.role('details') or not self._wants_cheap(): return ''
      if not self.all_words() or self.in_reserve(): return ''
      known = self.rec_db.ids
      named = [v for v in sent.values() if isinstance(v, str) and v in known]
      if len(named) != 1: return ''
      s = named[0]
      if s in self.rec_db.read or s in self.barrow.units or s in self.name_open(): return ''
      if self._peek('refs') or self.needs.parsed.get('references'): return ''
      if (k := self.prod_idx(s)) is None and self.needs.items(): return ''
      mark, told = (k, self.needs.rev_no), self._memo('vguard_told', set)
      if mark in told or not (valued := self.counted_ids(k)) or s in valued: return ''
      pick, own = self.walk_pick(k), self._ucost(s)
      fees = ([pick[1]] if pick else []) + ([] if own is None else [own])
      lower = self.cheap_unread(valued, k, min(fees))[:6] if fees else []
      if not lower: return ''
      told.add(mark)
      return 'Not run (no step used). First open these cheaper unread entries%s, cheapest first: %s; afterwards resend this call if you still want it. Reason: the cheapest is wanted, and a text query here must match every word, so the entries that value-led queries returned carry those words in their records - %s was not among them.' % (' (priced below it and the pick)' if pick else ' (priced below it)', ', '.join('%s (%s)' % (cid, format(c, 'g')) for c, cid in lower), s)
    except Exception as e:
      self._qu_err('value_guard_error', e)
      return ''
  def title_line(self, merchandise, rcr, template):
    avoid = set(STOP_WORDS)
    for shelf in self.walk_home(merchandise): avoid.update(re.findall('[a-z]+', shelf))
    for q in self.needs.props_left(): avoid.update(re.findall('[a-z0-9]+', q.lower()))
    for item in self.needs.items() or [{'product': str(self.needs.parsed.get('product') or '')}]:
      avoid.update(re.findall('[a-z]+', str(item.get('product') or '').lower()))
    stems = set(map(self.lemma, avoid))
    free = lambda w: w not in avoid and self.lemma(w) not in stems
    words = re.findall('[a-z0-9]+', self.title_of(rcr).lower())
    at = next((i for i, w in enumerate(words) if re.fullmatch('[a-z]{3,}', w) and free(w)), None)
    if at is None: return None
    nxt = words[at + 1] if at + 1 < len(words) else ''
    phrase = words[at] + ' ' + nxt if nxt and free(nxt) else words[at]
    return None if self.used_query(phrase) else {**dict(template[0]), template[1]: phrase}
  def rewalk(self):
    if not self._wants_cheap() or self._lowtime() or not self.role('search'): return None
    directed = self._peek('ceil_aimed') or {}
    gone, least = self.dead_lines() | self.refused_lines, FIRST_ORDER_BY + 3 + self._opens()
    for k, (aim, s) in list(directed.items()):
      if not s or s not in gone or self.ceil_cfg() is None or self.pg_arg() is None: continue
      if (pick := self.walk_pick(k)) and not (pick[1] <= aim + 1e-09 or self.steps_rem() < least): break
    else: return None
    directed.pop(k, None)
    (self._peek('ceil_aims') or {}).pop(k, None)
    self._note('ceil_rearm', product=k, dead=s, aim=pick[1])
    return self.try_walk([], FIRST_ORDER_BY, [k], repair=True)
  def walk_step(self):
    try:
      if self._peek('decline_queue'): return self.try_decl_reads()
      if q := vars(self).pop('walk_queue', None):
        plan, floor, ceil = q[:3]
        return self.try_walk(plan, floor, ceil, twins=q[3] if len(q) > 3 else None)
      if (got := self.rewalk()) is not None: return got
      for move in (self._wh_best, self._wh_steps):
        if (got := move()) is not self._skip_mark: return got
      return self._wh_cheap()
    except Exception as e:
      self._qu_err('walk_error', e)
      return False
  def _wh_cheap(self):
    if self.orders_tried or not self._wants_cheap() or self._lowtime(): return False
    merely = None
    if self.unseen_prod():
      merely = {k for k, _, _ in self.walk_plan(WALK_TOTAL) if self._only_bounds(k)} or None
      if merely is None: return False
    opens, spare = self._opens(), self.steps_rem()
    poised = [k for k in self.prod_slots() if (self.walk_pick(k) or self.bound_need(k))
         and self.cap_ready(k) and self.ceil_aim(k) is not None]
    reserve = self.order_due() + 2
    if merely is None and spare - opens - CEIL_CALLS > reserve:
      if poised:
        by = self._peek('valued_by') or {}
        rank = lambda k: (bool(self.walk_pick(k)), by.get(k, 0), -1 if k is None else k)
        return self.try_walk([], self.order_due() + 2, [min(poised, key=rank)])
      if self.ceil_seed(): return True
    if not (plan := self.walk_plan(WALK_TOTAL, merely)): return False
    need = sum(n + bool(self.walk_pick(k)) for k, _, n in plan)
    early = 2 if any(self._only_bounds(k) for k, _, _ in plan) else 0
    if spare - opens - need > reserve + early: return False
    if not (plan := self.walk_plan(min(spare - opens - (reserve - 1), WALK_TOTAL), merely)): return False
    self.walk_new = True
    return self.try_walk(plan, self.order_due())
  def num_steps(self, k):
    for _, q in self._own_reqs(k):
      text = self._bare_val(q)
      fgrs = re.findall('(?<![\\w.])(\\d+(?:\\.\\d+)?)\\s*[a-z\\"\']', text)
      if len(fgrs) != 1 or len(re.findall('\\d+(?:\\.\\d+)?', text)) != 1 or 'between' in text: continue
      bare, inv = self._nomore_rx.sub(' ', text), self._nomore_rx.search(text)
      below = inv.group(1) == 'more' if inv else self._under_rx.search(bare) is not None
      if not (below or inv or self._over_rx.search(bare)): continue
      if not 3 <= (v := float(fgrs[0])) < 100: return []
      n = int(v)
      exact = v == n and not self._strict_rx.search(bare)
      if below:
        top = n - 1 if v == n and not exact else n
        return [str(top - i) for i in range(RUNG_COUNT) if top - i > 0]
      return [str((n if exact else n + 1) + i) for i in range(RUNG_COUNT)]
    return []
  def _wh_steps(self):
    skip = self._skip_mark
    if not self._wants_cheap() or self.needs.items() or 'rungs_run' in vars(self): return skip
    orders = [e.get('id_order') for e in self._book().values() if 'id_order' in e]
    if self.orders_tried or self._lowtime() or not self.pg_arg() or False in orders: return skip
    if self._tvals(None) and self.walk_opts(None, deep=True): return skip
    if not (rungs := self.num_steps(None)): return skip
    home = self.walk_home(None)
    base = max((e for e in self._book().values() if len(f := self.arg_texts(e['args'])) == 1
          and e.get('id_order') is True and not re.search('\\d', f[0]) and not self.zg_lot(e['args'])
          and home and self._above_shelf(e, home)), key=lambda e: e['total'], default=None)
    if base is None or self.next_page(base) is None: return skip
    room = self.steps_rem() - self._opens() - (self.order_due() + 2)
    if room < 2 or self.asking and not self.barrow.units and room > RUNG_CALLS + 6: return skip
    word = self.arg_texts(args := base['args'])[0]
    slot = next(n for n in args if args[n] == word)
    text = args[slot].strip()
    searches = [q for q in ({**args, slot: '%s %s' % (text, g)} for g in rungs)
          if not self.used_query(q[slot])]
    self.rungs_run = [q[slot] for q in searches]
    self.log.append({'kind': 'rungs', 'base': text, 'queries': self.rungs_run})
    return self.try_walk([], self.order_due(), rungs={None: searches}) if searches else skip
  def _wh_best(self):
    if self.needs.priority() != 'best' or self.needs.items(): return self._skip_mark
    if self.orders_tried or self._lowtime(): return False
    spare, reserve = self.steps_rem() - 4, self.order_due() + 2
    plan = self.walk_plan(WALK_TOTAL, rule='best')
    need = sum(p for _, _, p in plan)
    if not plan or spare - need > reserve and vars(self).get('asking'): return False
    plan, paging = self.walk_plan(min(spare - (reserve - 1), WALK_TOTAL), rule='best'), self.pg_arg()
    if not plan or not paging: return False
    a = {'plan': plan, 'floor': self.order_due(), 'key': paging[0], 'book': self._book(),
      'walked': vars(self).setdefault('walked', set()), 'report': [], 'ran': False}
    self._walk_pages(a)
    if not a['ran']: return False
    shown = {s for _, name, _ in plan for s in (a['book'].get(name) or {}).get('row_shelf') or {}}
    metric = str(self.needs.parsed.get('priority_detail') or '').strip()
    notches = [w for w in re.findall('[a-z]{2,}', metric.lower()) if w not in STOP_WORDS]
    fresh = [(not any(w in str(t).lower() for w in notches), c, s, t)
        for s, (c, t, *_) in self.shelf_rows().items()
        if s in shown and s not in self.rec_db.read and s not in self.barrow.units]
    cat = '; '.join("%s '%s' %g" % (s, str(t)[:70], c) for _, c, s, t in sorted(fresh)[:12])
    metric = metric or 'the requested measure'
    guide = ('Rows of the requested category never opened, in stock and in budget: ' + cat + '. Ranked on ' + metric + ': before settling, read the property lists (' + (self.role('details') or 'the details tool') + ') of those whose title suggests they might beat the current pick on it.'
        if cat else 'They hold no unread row of the requested category.')
    self._add_text('[best walk] ' + '; '.join(a['report']) + '. ' + guide, True)
    self.log.append({'kind': 'best_walk', 'plan': [[n, p] for n, _, p in plan], 'rows': len(fresh)})
    return True
  def _pick_arg(self, props, label, skip):
    want = label.strip().lower()
    for name, spec in props.items():
      if not isinstance(spec, dict) or name in skip: continue
      if not self._cat_rx.search(str(name).lower()): continue
      if isinstance(choices := spec.get('enum'), list):
        hit = next((c for c in choices if isinstance(c, str) and c.strip().lower() == want), None)
        if hit is not None: return name, hit
      elif spec.get('type') == 'string': return name, label
    return None, None
  @classmethod
  def _cap_arg(cls, props, slot):
    return next((n for n, spec in props.items() if n != slot and isinstance(spec, dict) and not spec.get('enum')
          and spec.get('type') in ('integer', 'number') and cls._smu_cap_names.search(str(n).lower())
          and not cls._smu_not_cap.search(str(n).lower())), None)
  def seed_arg(self, label, cap):
    tool = self.role('search')
    shape = self.argspec.get(tool or '') or {}
    props, slot = shape.get('properties'), self.pg_arg()
    if not (isinstance(props, dict) and props) or not slot or not is_num(cap): return None
    about = next((str(t['function'].get('description') or '') for t in self.tool_specs
           if t['function']['name'] == tool), '')
    clauses = [seg.lower() for seg in re.split('[.;]', about)]
    (lid, lval), pid = (self._pick_arg(props, label, (self.term_key(), slot[0])),
              self._cap_arg(props, slot[0]))
    if not lid or not pid: return None
    call = {lid: lval, pid: int(cap) if props[pid].get('type') == 'integer' or float(cap).is_integer() else cap}
    near = '\\b' + re.escape(lid.lower()) + '\\b'
    for name, spec in props.items():
      opts = spec.get('enum') if name != lid and isinstance(spec, dict) else None
      if not isinstance(opts, list) or not 2 <= len(opts) <= 4: continue
      hit = next((c for c in opts if any(re.search('\\b' + re.escape(str(c).lower()) + '\\b', s)
                       and re.search(near, s) for s in clauses)), None)
      if hit is None: return None
      call[name] = hit
    if any(n not in call for n in shape.get('required') or []): return None
    out, faults = ArgCheck.vet(call, shape)
    return None if faults or not isinstance(out, dict) else out
  def ceil_seed(self):
    try:
      if self.ceil_cfg() is not None or self._peek('ceil_mode') or not self._wants_cheap():
        return False
      primed, slot = self._memo('ceil_seeded', set), self.pg_arg()
      skip = slot[0] if slot else None
      for k in self.prod_slots():
        if k in primed or k is None and self.needs.items(): continue
        pick, label = self.walk_pick(k), self.raw_label(self.walk_home(k))
        if not pick or not label: continue
        primed.add(k)
        if not (call := self.seed_arg(label, self.below(pick[1]))): return False
        kept = [(m, v) for m, v in dict(self.search_keys(call)[1]).items()
            if not (isinstance(call.get(m), str) and call[m] == label)]
        cap = next((n for n in call if n != skip and is_num(call[n])), '')
        ck = self._qu_sig({n: v for n, v in call.items() if skip is None or n != skip})
        sig = json.dumps([sorted(kept), cap], default=str)
        if sig in self._peek('ceil_bad', set()) or ck in (self._peek('ceil_seen') or {}): continue
        self._memo('ceil_seed_keys', set).add(ck)
        self.walk_call(call, "Checking the store's price cap: category '%s' below the pick (%s) - the cheapest is wanted." % (label, format(pick[1], 'g')))
        self._note('ceil_seed', product=k, args=call, acquired=self.ceil_cfg() is not None)
        if self._tail() is not None and self.ceil_cfg() is None: self._add_text(self.state_line())
        return True
      return False
    except Exception as e:
      self._qu_err('ceil_seed_error', e)
      return False
  def unseen_prod(self):
    if not self.needs.items(): return False
    try:
      opened = {s.strip().lower() for s in self.rec_db.shelf_read()}
      if any(not (home and home & opened) for home in self.home_cats()): return True
      tag = str(self.needs.rev_no)
      return self.alt_unseen() if [n for n in self.passed if n.split('|')[-1] == tag] else False
    except Exception: return False
  def walk_keep(self, sent):
    try:
      if not self._wants_cheap() or self._peek('walk_queue'): return ''
      if self.cable.spare() < CLOSING + 60 or self.refusal_n and self.steps_rem() <= 15: return ''
      ids = list(self.barrow.units) if sent is None else [
        s for s, n in self.barrow.lines(sent) if n is None or n > 0]
      merely, floor = {self.prod_idx(s) for s in ids if s}, FIRST_ORDER_BY + 1 if self.orders_tried == 0 else 4
      demi = (self.steps_rem() - 1 - floor) // 2
      room = demi - self._opens()
      order = sorted(merely, key=lambda k: -1 if k is None else k)
      ceil, twins, intents = [], {}, self._memo('ceil_aims', dict)
      for k in order:
        if room < 2 or not self.cap_ready(k): continue
        ceil.append(k)
        room -= CEIL_CALLS
        fees = [c for c in (self._ucost(s) for s in ids if s and self.prod_idx(s) == k) if c is not None]
        if fees:
          if is_num(intents.get(k)): fees.append(float(intents[k]))
          intents[k] = min(fees)
      for k in order:
        if room > 0 and (q := self.pair_query(k)):
          twins[k], room = q, room - 1
      plan = self.walk_plan(min(room, WALK_TOTAL), merely)
      if not plan and not ceil and not twins: return ''
      self.walk_queue = (plan, hi if (hi := self.steps_rem() - 1 - demi) > floor else floor, ceil, twins)
      book, rec = self._peek('page_book', {}), lambda args: "'" + ' '.join(self.arg_texts(args)) + "'"
      queries = [rec(book[key]['args']) for _, key, _ in plan if key in book]
      chores = [t for c, t in (
        (twins, 'a search in the other spelling this store uses for a required value (%s)' % ', '.join(rec(s) for s in twins.values())),
        (queries, 'the unread pages of %s' % ', '.join(queries)),
        (ceil, "a listing of that category below the pick's price through the price cap")) if c]
      return 'Not run yet (no step used). The code is now doing this for you: %s; after that it opens the cheapest unread entries below the pick. Then resend this call, or one that names a cheaper match. Reason: the cheapest is wanted, and results here come %s rather than by price, so pages not read yet may hold a cheaper match.' % ('; '.join(chores), 'ranked by match' if self.top_store() else 'in id order')
    except Exception as e:
      self._qu_err('walk_hold_error', e)
      return ''
  @staticmethod
  def _rows_of(morsel):
    m = [v for v in morsel.values() if isinstance(v, list) and all(isinstance(r, dict) for r in v)]
    return next((t for t in m if t), m[0]) if m else None
  @staticmethod
  def _add_new(found, extra): found.extend([i for i in extra if i not in found])
  def find_nt(self, sent, obs):
    if not isinstance(obs, dict): return ''
    if (rows := self._rows_of(obs)) is None:
      if (inner := next((seg for seg in obs.values() if isinstance(seg, dict)
               and self._rows_of(seg) is not None), None)) is None:
        return ''
      obs, rows = inner, self._rows_of(inner) or []
    totals = [n for n in obs.values() if type(n) is int and n > len(rows)]
    count, say = max(totals, default=len(rows)), (out := []).append
    self.log_page(sent, rows, count, obs)
    try:
      self.guess_pairs(sent, rows)
      self.ceil_guess(sent, rows, count, obs)
    except Exception as e: self._qu_err('ceil_learn_error', e)
    mass = self.val_score(sent) if rows else 0
    if mass:
      best = self._memo('row_score', dict)
      for row in rows:
        if isinstance(row, dict): self._lift(best, self.row_key(row), mass)
    kind, lead = self.word_rule(sent, obs, rows), self.first_rule(sent, obs, rows)
    if lead:
      writings, mode, _, _ = self.search_keys(sent)
      field, style = self.lead_ctx[:2]
      words = next((v for n, v in writings if n == field), ()) if mode == style else ()
      say('Query length: this store reads only the first %s word(s) of its text query (%s) and drops the rest%s. Lead with the %s most distinctive words (the required value, then the item kind) and give every other value a short query of its own.' % (lead, field, " (in this query: '%s')" % ' '.join(words[:lead]) if len(words) > lead else '', lead))
    writings, mode, first, paging = self.search_keys(sent)
    styles = dict(mode)
    raw = [v for n, v in sent.items() if isinstance(v, str) and n not in styles and n not in paging]
    if first and raw: self._zn_words(sent, rows, totals, count, kind, raw, writings, mode, styles, paging, say)
    if not rows: return ' '.join(out)
    if totals:
      say('Partial result: %d matches, %d of them shown, and only a few pages can be read - relevant offers may be hidden. First read on through this same query (as far as it goes); if the total exceeds those pages, ' % (max(totals), len(rows)) + self._widen.get(kind, self._widen_pair))
      lp = self.pg_arg()
      try: pursue = self.next_cursor(sent, obs) if lp else None
      except Exception: pursue = None
      if pursue and lp[0] in pursue:
        say('For its next page, resend the same call with %s = %s.' % (lp[0], json.dumps(pursue[lp[0]])))
    skip = self.search_keys(sent)[3]
    caps = [v for n, v in sent.items() if is_num(v) and v > 1 and n not in skip
        and not self._not_cap_rx.search(str(n).lower())]
    if caps:
      top = max(caps)
      over = sum(1 for row in rows
           if any(is_num(v) and v > top and 'price' in str(n).lower() for n, v in row.items()))
      if over: say('Cap ignored: %d entr(ies) cost more than %s even though this call set that cap - this search mode skips that filter, possibly the category one too. Use a mode whose filters hold, or check each entry yourself.' % (over, top))
    return ' '.join(out)
  def _zn_words(self, sent, rows, totals, count, kind, raw, writings, mode, styles, paging, say):
    crumbs = re.findall('[a-z0-9]+', ' '.join(raw).lower())
    if crumbs and max(map(len, crumbs)) == 1 and not rows and not self._peek('short_told'):
      self.short_told = True
      say('No results. This store discards one-letter and one-digit tokens, so 0.4 or 5 x 7 contributes nothing to a query beyond its other words. Write the value as a word, or combine its unit word with the item kind, then read that query to the end.')
    prop, scout = self.val_scan(raw)
    named, excess = self._peek('name_told', 0), lambda: self.steps_rem() - 3 > self.order_due()
    if scout:
      if named < 2 and excess():
        self.name_told = named + 1
        say(('Again the property name (%s) opens the query.' if named else
          "This query cannot surface offers whose %s field holds just the value: it begins with the property's NAME and carries no required value, whereas matching runs on values, titles and categories (a field name only counts where a value or the title repeats it).") % prop
          + " Suggested query: '%s' (the value before the item kind; figures the store cannot read drop out and their unit words stay). Read it to the end." % scout)
    elif self._peek('trap_told', 0) < 2 and excess():
      gotcha, _ = self.word_trap(raw)
      if gotcha:
        self.trap_told = 1 + self._peek('trap_told', 0)
        say(gotcha)
    notches = tuple(map(self.row_key, rows))
    sig = (mode, tuple(sorted((n, str(v)) for n, v in sent.items()
                 if n not in paging and not (isinstance(v, str) and n not in styles))), count, notches)
    seen, query = self._memo('page_sigs', dict), ' '.join(raw)
    if len(set(notches)) >= 3 and sig in seen and seen[sig] != query:
      say("Same total and same entries as '%s' before - this store treats both spellings as one query. Read its next page instead, or change the leading words." % seen[sig])
    seen.setdefault(sig, query)
    full = rows and not totals
    told, hits = bool(full and (kind == 'all' or self.lead_words)), []
    if told or full and kind != 'any':
      held, self.ref_hits = None if told else list(self._peek('ref_hits') or []), []
      hits = self.row_hits(rows, raw)
      mass, scores = self.val_score(sent), self._memo('hit_score', dict)
      for s in hits: self._lift(scores, s, mass)
      self._add_new(self._memo('all_hits', list), hits)
      if held is not None: self.ref_hits = held
      if told and hits and len(writings) and len(rows) <= 12:
        say('Before querying further, open these in-budget entries: %s. Reason: every query word is required and nothing is hidden here, so each entry contains them somewhere in its record, often outside the title.' % ', '.join(hits[:6]))
      pointers = [s for s in vars(self).pop('ref_hits', []) if s not in hits][:6] if told else []
      if pointers: say('Reference entries (read them for a value, never buy them): %s. Open every one whatever its price, stock or category and follow its notes - the request pins a value through another offer, and these come from the shop or brand this query names.' % ', '.join(pointers))
    if self._peek('hits_back') or not (kind == 'all' or self.lead_words): return
    self.hits_back, here, back = True, self._qu_sig({n: v for n, v in sent.items() if n not in paging}), []
    for key, walk in self._peek('page_book', {}).items():
      if key == here or not walk.get('every') or not (mass := self.val_score(walk['args'])): continue
      fresh = [s for s in self.row_hits(walk['every']) if s not in back and s not in hits]
      scores = self._memo('hit_score', dict)
      for s in fresh: self._lift(scores, s, mass)
      back.extend(fresh)
    self._add_new(self._memo('all_hits', list), back)
    if back: say('Before querying further, open the unread in-budget ones: %s. Reason: with every word required, earlier complete queries could only list entries holding all their words, often outside the titles.' % ', '.join(back[:6]))
  _widen = {'any': 'Loose-match store: a single shared word is enough and offers sharing more words rank higher, so extra words make the result larger. On its own, the distinctive word or figure of a required value returns only offers containing it (often few enough to read to the end and compare prices); queries of kind plus values put the closest offers first; category mode, or a working price cap, covers what is left.', 'all': 'Every-word store: each extra exact word shrinks the result, as do the category and a lower price cap.'}
  _widen_pair = 'narrow it: if every word is required, add exact words; if one shared word is enough (a longer query that returns more offers reveals this), add the required values so matching offers rank first; a working price cap helps as well.'
  def buyer_lines(self, summon_appellation, obs, refused):
    return list(dict.fromkeys(self._reply_text(obs) if summon_appellation == self.role('message') else
                 self.warn_lines(obs, refused) if isinstance(obs, dict) else []))

class CallFilter:
  _qv_NR, _NOT_RUN = 'Not run (no step used)', 'Not run yet (no step used). '
  _gate_steps = ('_smv_gate_known', '_smv_gate_offered', '_smv_gate_decode', '_smv_gate_shape',
             '_smv_gate_list', '_smv_gate_handles', '_smv_gate_alone', '_smv_gate_messages',
             '_smv_gate_view', '_smv_gate_repeat', '_smv_gate_lookup', '_smv_gate_guards',
             '_smv_gate_cart')
  _cart_steps = ('_smv_cart_stated', '_smv_cart_pointer', '_smv_cart_kept', '_smv_cart_takes',
             '_smv_cart_double', '_smv_cart_walk', '_smv_cart_others', '_smv_cart_own', '_smv_cart_fit',
             '_smv_cart_ranked', '_smv_cart_undercut', '_smv_cart_units')
  _finish_paths = {'order': ('_smv_fin_empty', '_smv_fin_sold_out', '_smv_fin_cheaper', '_smv_fin_walk', '_smv_fin_stale',
                  '_smv_fin_sweep', '_smv_fin_depends', '_smv_fin_vet', '_smv_fin_misfit',
                  '_smv_fin_steps'),
              'decline': ('_smv_fin_closest', '_smv_fin_told','_smv_fin_dropped', '_smv_fin_early', '_smv_fin_near_hold',
                   '_smv_fin_whole', '_smv_fin_kept', '_smv_fin_slack', '_smv_fin_fill',
                   '_smv_fin_named', '_smv_fin_pages', '_smv_fin_reaudit', '_smv_fin_audit',
                   '_smv_fin_fit_hold', '_smv_fin_best', '_smv_fin_unknown_ref',
                   '_smv_fin_repair', '_smv_fin_meaning', '_smv_fin_steps')}
  @staticmethod
  def _reply_text(obs):
    if isinstance(obs, str): return [obs]
    pouch = []
    if isinstance(obs, dict) and 'error' not in obs:
      grab = lambda d: long_strs([v for v in d.values() if isinstance(v, (str, list))], pouch)
      walk_dicts(obs, grab, ceiling=4)
    return pouch
  @staticmethod
  def warn_lines(obs, refused=False):
    if not isinstance(obs, dict): return []
    pouch = []
    def gather(d):
      if d is not obs and len(d) > 6: return
      minor = len(d) <= 3
      for v in d.values():
        if isinstance(v, list):
          if v and all(isinstance(m, str) for m in v): long_strs(v, pouch)
        elif isinstance(v, str) and len(v.split()) >= 4 and (refused or minor and len(v) <= 600 and not any(
            isinstance(x, str) and 'id' in str(key).lower() and re.fullmatch(ID_RX, x)
            for key, x in d.items())):
          pouch.append(v)
    walk_dicts(obs, gather, ceiling=4)
    return list(dict.fromkeys(pouch))
  def _attr(self, name, default=None): return vars(self).get(name, default)
  def _tag(self, symbol): return '%s|%s' % (symbol, self.needs.rev_no)
  def _room(self, steps, filler=None):
    return not self.steps_rem() <= steps and (filler is None or self.cable.spare() > CLOSING + filler)
  @staticmethod
  def _join_list(sqc): return ', '.join(sqc)
  def _costly(self, units):
    cap, total = self.needs.spend_cap, 0.0
    for s, n in units.items():
      total += float(p) * float(n) if is_num(p := self.rec_db.price(s)) else 0.0
    return is_num(cap) and total > float(cap) + 1e-09
  def screen(self, bid, authorized, programmed, batch, pipelined_communications):
    spec = bid.get('function') or {}
    st = {'name': spec.get('name'), 'raw': spec.get('arguments'), 'args': {}, 'sig': '', 'menu': authorized,
       'own': not programmed, 'many': batch, 'queued': pipelined_communications}
    return next(filter(None, (getattr(self, g)(st) for g in self._gate_steps)), None), st['name'], st['args']
  def _smv_gate_known(self, st):
    return '' if st['name'] in self.argspec else "%s: this store has no tool named '%s'. Choose one of these tools: %s." % (self._qv_NR, st['name'], self._join_list(sorted(st['menu'])))
  def _smv_gate_offered(self, st):
    return '' if not st['own'] or st['name'] in st['menu'] else '%s: too few steps remain for that tool; the tools usable right now are only: %s.' % (self._qv_NR, self._join_list(sorted(st['menu'])))
  def _smv_gate_decode(self, st):
    got = raw if isinstance(raw := st['raw'], dict) else as_json(raw) if (raw or '').strip() else {}
    if got is None: return self._qv_NR + ': the arguments could not be read; send a single well-formed JSON object.'
    st['args'] = got
    return ''
  def _smv_gate_shape(self, st):
    st['args'], faults = ArgCheck.vet(st['args'], self.argspec[st['name']])
    return self._qv_NR + ': the arguments violate the tool schema: ' + '; '.join(faults[:6]) if faults else ''
  def _smv_gate_list(self, st):
    spec = self.batch_tool()
    ids = st['args'].get(spec[1]) if spec and st['own'] and st['name'] == spec[0] else None
    if not isinstance(ids, list) or len(ids) <= spec[2]: return ''
    self.ref_idx()
    seen = vars(self).get('ref_seen_at') or {}
    kept = set(sorted(ids, key=lambda p: -seen.get(p, -1))[:spec[2]])
    st['args'][spec[1]] = [p for p in ids if p in kept]
    self.log.append({'kind': 'list_cut', 'asked': len(ids), 'cap': spec[2]})
    return ''
  def _smv_gate_handles(self, st): return (self._page_word(st) or self._title_ids(st)) if st['own'] else ''
  def _page_word(self, st):
    paging = self.pg_arg()
    if st['name'] != self.role('search') or not paging or paging[1] != 'token': return ''
    key, args = paging[0], st['args']
    token = args.get(key)
    if not isinstance(token, str) or not token.strip() or token.strip() in {
        t for w in self._book().values() for p in (w.get('tokens') or {}).items()
        for t in p if isinstance(t, str) and t}:
      return ''
    rest = self._without(args, key)
    if (walk := self._book().get(self._qo_sgn(rest))) is None:
      st['args'] = rest
      self.log.append({'kind': 'token_dropped', 'token': token[:40]})
      return ''
    nxt = self.next_page(walk)
    return "%s: '%s' was never handed out as a page token by this store; tokens are copied from the next-page field of a result. That search has run already: %s." % (self._qv_NR, token[:40], 'no page of it is left unread' if nxt is None else "its next page is requested with %s='%s'" % (key, nxt))
  def _title_ids(self, st):
    tools = {self.role('cart'), self.role('details'), self.role('view')} | {(self.batch_tool() or ('',))[0]}
    if st['name'] not in tools: return ''
    known, writings = self.rec_db.ids, []
    self._swap(st['args'], {}, writings)
    strays = {t for t in writings if t not in known and len(t.strip()) > 8}
    if not strays: return ''
    owners, swap, clash = {}, {}, []
    for row in self._trows():
      name, n = self.title_of(row).strip().lower(), self.row_key(row)
      if name and n in known: owners.setdefault(name, set()).add(n)
    for t in strays:
      ids = owners.get(low := t.strip().lower(), set())
      if not ids and len(low) >= 20: ids = set().union(*(v for name, v in owners.items() if name.startswith(low)))
      if len(ids) == 1: swap[t] = next(iter(ids))
      elif ids: clash.append("'%s' is the title of %s" % (t[:60], ', '.join(sorted(ids))))
    if clash: return '%s: this tool expects listing ids, not titles: %s. Resend it with the intended id.' % (self._qv_NR, '; '.join(clash))
    if swap:
      st['args'] = self._swap(st['args'], swap)
      self.log.append({'kind': 'title_to_id', 'swapped': sorted(swap.values())})
    return ''
  def _swap(self, knot, swap, seen=None):
    if isinstance(knot, dict): return {k: self._swap(v, swap, seen) for k, v in knot.items()}
    if isinstance(knot, list): return [self._swap(v, swap, seen) for v in knot]
    if not isinstance(knot, str): return knot
    if seen is not None: seen.append(knot)
    return swap.get(knot, knot)
  def _smv_gate_alone(self, st):
    return self._qv_NR + ': a finish (order or decline) must be the single call of its reply, sent once the results of your other calls have arrived.' if st['name'] in self.end_tools() and st['many'] > 1 else ''
  def _smv_gate_messages(self, st):
    if st['name'] != self.role('message'): return ''
    spent, cap = st['queued'] + self.msgs_sent, self.msg_cap
    if cap is not None and spent >= cap:
      return self._qv_NR + ': no messages are left; nothing more reaches the shopper.'
    if st['own'] and not self.barrow.units and self.msg_reserve() and spent >= self.ask_room():
      return self._qv_NR + ': message allowance on hold. The remaining sends are kept for the basket summary due once something is carted, and for any closing step the instructions demand. Cart a choice first; after that you may write again.'
    return ''
  def _smv_gate_view(self, st):
    if (not st['own'] or (tool := st['name']) != self.role('cart_view') or tool == self.role('cart')
        or self._attr('batch_cart') or not self.barrow.units):
      return ''
    if (self._attr('cart_shown', -1) != self._attr('cart_ver', -2) or self._attr('since_cart', 1)
        or self.cart_shown in (seen := vars(self).setdefault('view_told', set()))):
      return ''
    seen.add(self.cart_shown)
    return "%s: viewing the cart would add nothing; the latest cart reply already shows every line. %s To see the current stock of a line, read that listing's record with %s." % (self._qv_NR, self.barrow.recap(), self.stock_tool() or 'whichever tool shows stock')
  def _smv_gate_repeat(self, st):
    st['sig'] = (tool := st['name']) + as_text(st['args'])
    if tool in self.end_tools() or tool in (self.role('message'), self.role('cart')): return ''
    if st['sig'] not in self.last_calls: return ''
    return self._qv_NR + ': the very same call just ran and nothing has changed since; its result is further up. Change the arguments or choose another step.'
  def _smv_gate_lookup(self, st):
    tool, args = st['name'], st['args']
    if not isinstance(args, dict) or tool in self.end_tools(): return ''
    if tool in (self.role('message'), self.role('cart')): return ''
    writings = {str(k): v.strip() for k, v in args.items() if isinstance(v, str) and v.strip()}
    if tool == self.role('search') and any(v in self.rec_db.ids for v in writings.values()):
      return '%s: search text is matched against the words of listings, so a listing id finds nothing. Open that listing with %s.' % (self._qv_NR, self.role('details') or 'the record tool')
    labels = self.shelf_names()
    shelves = {x.rstrip('s') for x in labels}
    for name, value in writings.items() if labels else ():
      if re.search('categor|shelf|department|aisle', name.lower()) and value.lower().rstrip('s') not in shelves:
        return '%s: %s expects a shelf name written the way the listings show it; free words there match nothing. Shelves seen so far: %s. Put descriptive words into the search text instead.' % (self._qv_NR, name, ', '.join(sorted(labels.values())[:8]))
    return ''
  def shelf_names(self):
    found, ids = {}, self.rec_db.ids
    def attend(d):
      if not any('id' in str(k).lower() and isinstance(v, str) and v in ids for k, v in d.items()): return
      for k, v in d.items():
        if not isinstance(v, str) or not 0 < len(v.strip()) <= 60 or 'id' in (low := str(k).lower()): continue
        if re.search('categor|shelf|department|aisle', low):
          found.setdefault(v.strip().lower(), v.strip())
          return
    for rcr in self.rec_db.items: walk_dicts(rcr, attend)
    return found
  def _smv_gate_guards(self, st):
    if not st['own']: return ''
    return self.late_keep(st['name'], st['args']) or self.val_guard(st['name'], st['args'])
  def _smv_gate_cart(self, st):
    return self.gate_cart(st['args'], st['sig']) if st['name'] == self.role('cart') else ''
  def gate_cart(self, summon_prms, sig):
    st = {'args': summon_prms, 'sig': sig, 'line': self.barrow.aim(summon_prms), 'point': ''}
    return next((v for v in (getattr(self, s)(st) for s in self._cart_steps) if v is not None), '')
  def _smv_cart_stated(self, st):
    cart = self.barrow
    if (self.ask_cap() - self.msgs_sent > 0 or not self._told_now()
        or self.dead_lines() & set(cart.units)
        or any(self._tag(x) not in self.passed for x in cart.units)
        or self._costly(cart.units) or (n := vars(self).get('stated_holds', 0)) >= 3):
      return None
    self.stated_holds = n + 1
    self.log.append({'kind': 'stated_hold', 'lines': [str(x) for x in cart.units]})
    return '%s: the shopper has already been told this exact cart, and no message is left to tell a changed one, so editing it now would send an order the shopper never saw. Nothing since calls for a change (no notice, refusal, sold-out line or total above the budget). Place the order with the cart as it is.' % self._qv_NR
  def _smv_cart_pointer(self, st):
    line = st['line']
    st['point'] = self.ref_ptr_note(line) if line and self.barrow.is_add(st['args']) else ''
    n = (told := vars(self).setdefault('pointer_told', {})).get(line, 0)
    if not st['point'] or n >= 2: return None
    told[line] = n + 1
    return '%s: %s. Choose another listing.' % (self._qv_NR, st['point'])
  def _smv_cart_kept(self, st):
    line = st['line']
    mark = '%s#%s' % (st['sig'], self._attr('puts', {}).get(line, 0))
    price = self.rec_db.price(line) if line else None
    bested = any(c != line and is_num(price) and is_num(r := self.rec_db.price(c)) and r < price
          for c in self._attr('batch_puts', []))
    swap = self.needs.priority() == 'cheapest' and (bested or bool(self.cheaper_fit(line)))
    rated = self._attr('pass_price', {})
    if not line or self.barrow.is_add(st['args']) or self._tag(line) not in self.passed: return None
    if swap or line in rated and rated[line] != price: return None
    if self.refusals_seen.get(line, self.refusal_n) != self.refusal_n or mark in self.warned: return None
    self.warned.add(mark)
    return '%s: removal paused. Independent check on %s: it meets every condition now in force (properties judged by the value rules, a value field dedicated by this store counting only by its wording; the item kind taken from the category label alone, whatever the title or text mention).%s Nothing new has happened since. Send this exact call again only for: a change or refusal ruling it out, a field plainly breaking a condition, or a cheaper listing replacing it.' % (self._qv_NR, line, ' No verified fit is cheaper.' if self.needs.priority() == 'cheapest' and not self.cheaper_fit(line) else '')
  def _smv_cart_takes(self, st): return None if self.barrow.is_add(st['args']) and st['line'] else ''
  def _smv_cart_double(self, st):
    line, sig, cart = st['line'], st['sig'], self.barrow
    if cart.held(line) <= 0 or cart.incl is False or sig in self.warned: return None
    self.warned.add(sig)
    return '%s: the cart holds %s x%d, and since this tool adds units onto a held line the call would raise that count. No cart change is needed to order; to change units, add or remove just the difference. Resend the identical call if more units really are intended.' % (self._qv_NR, line, int(cart.held(line)))
  def _smv_cart_walk(self, st): return None if st['point'] else self.walk_keep(st['args']) or None
  def _vet_add(self, line):
    if not self._room(3) or (tag := self._tag(line)) in self.checked_adds: return None
    self.checked_adds.add(tag)
    edict = self.check_item(line, 'ADD TO THE CART')
    return '%s. %s Resend this exact call only when you are sure it fits.' % (self._qv_NR, edict) if edict else None
  def _live(self, args): return [s for s, n in self.barrow.lines(args) if n is None or not n <= 0]
  def _smv_cart_others(self, st):
    live = (c for c, n in self.barrow.lines(st['args'])[1:] if n is None or n > 0)
    return next(filter(None, map(self._vet_add, live)), None)
  def _smv_cart_own(self, st): return self._vet_add(st['line'])
  def _smv_cart_fit(self, st):
    rows = self.needs.items()
    links = self._links(rows)
    if not links or not self._room(5, 0): return None
    first = {}
    for line in [*self._live(st['args']), *self.barrow.units]:
      if (slot := self.prod_idx(line)) is not None: first.setdefault(slot, line)
    told, found = vars(self).setdefault('code_fit_told', set()), []
    for k, row, jx in links:
      if k not in first or jx not in first or len(told) >= 5: continue
      al, dl = self._symtxt(a := first[jx]), self._symtxt(d := first[k])
      if (a, d) in told or self._coded(al, dl) is not False: continue
      told.add((a, d))
      stems = {self._cstem(h) for h in self._codes(dl)[1]}
      codes = [c for c in sorted(self._codes(al)[0]) if self._cstem(c) in stems]
      self.log.append({'kind': 'code_fit_hold', 'anchor': a, 'dependent': d, 'codes': codes})
      found.append((k, row, jx, a, d, codes))
    if not found: return None
    queue, said = vars(self).setdefault('fit_queue', []), []
    for k, row, jx, a, d, codes in found:
      if codes and not any(k == q[0] for q in queue): queue.append((k, codes[0]))
      said.append('line %s (the %s) is meant for %s (the %s), whose page states %s, yet %s carries a tag of that series with a different number' % (d, row['product'], a, rows[jx]['product'], self._join_list(codes), d))
    return '%s: this store writes fits as codes, and %s; so %s not fit. The code now searches %s using the code as the search text and opens the tagged hits. For each one, cart a hit whose tag begins with that code and that meets the remaining conditions; or switch to an anchor whose stated code appears in a tag of each. Send this exact call again only when no such listing exists.' % (self._qv_NR, '; '.join(said), 'those lines do' if len(found) > 1 else 'it does', self._join_list([row['product'] + ' listings' for _, row, *_ in found]))
  def _smv_cart_ranked(self, st):
    if not self._room(3): return None
    for s in self._live(st['args']):
      if bested := self.rank_check(s):
        rival = self.rank_verify(s)
        text = bested.replace('before ordering', 'prior to adding it').replace(
          'order the current pick', 'resend the identical call')
        if rival:
          text += ' Cheaper verified alternative: %s (independent item check: meets current conditions; priced below %s) - cart %s instead.%s' % (rival, s, rival, self.shelf_ok_note(rival))
        return self._qv_NR + '. ' + text
    return None
  def shelf_ok_note(self, symbol):
    try: got = [x[2:] for x in self.item_view(symbol)['lines'] if 'itself names this value' in x]
    except Exception: return ''
    return ' Satisfied by its category alone: ' + '; '.join(got[:3]) + '.' if got else ''
  def _smv_cart_undercut(self, st):
    if self.needs.priority() != 'cheapest': return None
    for s in self._live(st['args']):
      rival = self.cheaper_fit(s)
      told, pair = vars(self).setdefault('cheaper_told', {}), ('cart', s, rival)
      if not rival or rival in self.barrow.units or told.get(pair): continue
      told[pair] = 1
      return '%s. Cheaper verified alternative: %s (independent item check: meets the conditions now in force; priced below %s). The lowest price comes first and that check outranks your reading, so cart %s instead. Send this exact call again only when a record shows %s breaking a current condition.%s' % (self._qv_NR, rival, s, rival, rival, self.shelf_ok_note(rival))
    return None
  def _smv_cart_units(self, st):
    line, sig, args, cart, needs = st['line'], st['sig'], st['args'], self.barrow, self.needs
    sought, asked = needs.quantity(), cart.count_arg(args)
    target = sought if cart.incl is False else sought - cart.held(line)
    if needs.items():
      bad = [(s, int(n)) for s, n in cart.lines(args) if n is not None and n > 0 and int(n) not in needs.counts()]
      if not bad or sig in self.warned: return ''
      self.warned.add(sig)
      return '%s: units per product should be %s (one line each), and this call has %s. Resend the identical call if that is intended.' % (self._qv_NR, needs.counts(), bad)
    if asked is None or abs(asked - target) <= 1e-09 or sig in self.warned: return ''
    self.warned.add(sig)
    return '%s: a single listing is needed in %s units, and this call has %d. Resend the identical call if that is intended.' % (self._qv_NR, sought, int(asked))
  def gate_finish(self, cid, summon_appellation, summon_prms):
    mode = self.call_kind(summon_appellation, summon_prms)
    st = {'mode': mode, 'spread': None}
    # the nearest-listing request was already pressed twice (or no step is left to press it): with a filled cart
    # and one finish tool, order it
    if (mode == 'decline' and (vars(self).get('closest_held', 0) >= 2 or not self._room(2)) and self.barrow.units
        and self.needs.parsed.get('fallback_pick') and self.role('order') == summon_appellation):
      buy = self.plain_args(summon_appellation, 'order', 'order')
      if self.call_kind(summon_appellation, buy) == 'order':
        self.log.append({'kind': 'closest_order', 'cart': dict(self.barrow.units)})
        return None, buy
    if v := next(filter(None, (getattr(self, j)(st) for j in self._finish_paths.get(mode, ()))), None):
      return v, summon_prms
    return None, self.clean_decl(summon_prms, summon_appellation) if mode == 'decline' else summon_prms
  @staticmethod
  def _verr(mistake): return ('%s: %s' % (type(mistake).__name__, mistake))[:300]
  def _smv_fin_sold_out(self, st):
    if not self._room(2): return ''
    cart = self.barrow
    gone = sorted(self.dead_lines() & set(cart.units))
    mark, told = (tuple(gone), self._attr('cart_seq', 0)), vars(self).setdefault('dead_told', {})
    if not gone: return ''
    if (n := told.get(mark, 0)) < 2:
      told[mark] = n + 1
      return self._NOT_RUN + self.dead_note(gone)
    if n != 2 or not self.role('cart') or self.steps_rem() < 6 or self.cable.spare() <= CLOSING: return ''
    told[mark], hush = 3, self.cable.hush
    try: swapped = self.code_cart(set(gone) | self.dead_lines(), set(cart.units) - set(gone))
    except Exception as e:
      swapped = False
      self.log.append({'kind': 'dead_swap_error', 'error': self._verr(e)})
    finally: self.cable.hush = hush
    self.log.append({'kind': 'dead_swap', 'lines': gone, 'done': swapped, 'cart': dict(cart.units)})
    if not swapped or not cart.units or set(cart.units) & self.dead_lines(): return ''
    return self._NOT_RUN + self._join_list(gone) + ': out of stock, yet in the cart. The code swapped it - a candidate took its place, the other lines left as they were - then read the stock of each line: ' + cart.recap() + ' Check it against the request; order if it matches.'
  # an order with an empty cart can only fail: say so (twice at most) and show the opened listings to cart from
  def _smv_fin_empty(self, st):
    if self.barrow.units or not self.role('cart') or not self._room(1): return ''
    held = vars(self).get('empty_held', 0)
    if held >= 2: return ''
    self.empty_held = held + 1
    self.log.append({'kind': 'empty_order_hold', 'n': held + 1})
    return self._NOT_RUN + 'The cart is empty, so an order would fail. Cart the chosen listing first (a condition the shopper dropped does not make a carted fit wrong), then order.' + self.code_reads()
  # the request wants the nearest listing proposed when nothing meets every condition: only an order does that
  def _smv_fin_closest(self, st):
    if not self.needs.parsed.get('fallback_pick') or not self.role('cart') or not self._room(2): return ''
    held = vars(self).get('closest_held', 0)
    if held >= 2: return ''
    self.closest_held = held + 1
    self.log.append({'kind': 'closest_hold', 'n': held + 1, 'cart': dict(self.barrow.units)})
    head = self._NOT_RUN + 'The request asks for the nearest listing to be proposed when no listing meets every condition, so this errand ends with an order, never with nothing bought. '
    if self.barrow.units:
      return head + 'The cart already holds: ' + self.barrow.recap() + ' Order it now (or first swap a line for an opened listing that misses fewer conditions, stock and budget included).'
    return head + 'Cart the opened listing that misses the fewest conditions (being out of stock or over budget counts as a miss); among listings that tie, take the cheapest or one only slightly dearer. Then order it.' + self.code_reads()
  def _smv_fin_cheaper(self, st):
    if not self._room(3): return ''
    rival = self.cheaper_fit(line) if (line := self.chosen()) else ''
    if not rival:
      for c in [s for s in self.barrow.units if s != line]:
        rival = self.cheaper_fit(c)
        if rival:
          line = c
          break
    told, pair = vars(self).setdefault('cheaper_told', {}), (line, rival)
    if not rival or told.get(pair, 0) >= 2: return ''
    told[pair] = told.get(pair, 0) + 1
    return '%sCheaper verified alternative: %s (independent item check: meets every condition; priced below %s). The lowest price comes first and that check outranks your reading: remove %s, cart %s, refresh its stock and price, then order.%s' % (self._NOT_RUN, rival, line, line, rival, self.shelf_ok_note(rival))
  def _smv_fin_walk(self, st): return self.walk_keep(None) if self.barrow.units else ''
  def _smv_fin_stale(self, st):
    if not (tool := self.stock_tool()) or not self._room(len(self.barrow.units) + 2, 0): return ''
    stale, mark = self.stale_lines(), (tuple(sorted(self.barrow.units)), self._attr('cart_seq', 0))
    if not stale or self._attr('recheck_mark') == mark: return ''
    self.recheck_mark, self.open_queue = mark, stale[:4]
    self.open_recheck, self.open_purpose, self.open_tool = True, 'cart', tool
    return '%sStock and prices can change without any notice, and these lines have not been read since the cart last changed: %s. The code reads their stock for you now with %s, the tool that shows stock. Then: if every line is available and the total is within the budget, order; otherwise replace the line that fails.' % (self._NOT_RUN, self._join_list(stale[:4]), tool)
  def _smv_fin_sweep(self, st):
    try: return self.cheap_sweep()
    except Exception as e:
      self.log.append({'kind': 'cheap_sweep_error', 'error': self._verr(e)})
      return ''
  def _smv_fin_depends(self, st):
    if not self._room(5, 0): return ''
    try: faults = self.dep_errors() if self._fit_built() else []
    except Exception: faults = []
    mark, told = (tuple(sorted(self.barrow.units)), tuple(faults)), vars(self).setdefault('dep_told', {})
    if not faults or told.get(mark, 0) >= 2 or len(told) >= 4: return ''
    told[mark] = told.get(mark, 0) + 1
    return '%sCompatibility gap: %s. Search for the dependent item with each identifier in turn as the search text, open the hits and cart a hit whose page mentions an identifier; or switch to an anchor that some dependent page cites by identifier. Order once that holds.' % (self._NOT_RUN, '; '.join(faults))
  def _smv_fin_vet(self, st):
    if self.order_vetted or not self._room(2, 0): return ''
    self.order_vetted, needs = True, self.needs
    faults = self.barrow.faults(needs.quantity(), needs.counts())
    if faults and self._room(3): return '%sThe count check rejects the cart: %s. Correct it, then order.' % (self._NOT_RUN, '; '.join(faults))
    lines = list(self.barrow.units) if needs.items() else [self.chosen()] if self.chosen() else []
    edict = next(filter(None, (self.check_item(x, 'ORDER') for x in lines)), None)
    if edict: return self._NOT_RUN + edict
    for x in lines:
      if bested := self.rank_check(x):
        rival = self.rank_verify(x)
        if rival:
          bested += ' Cheaper verified alternative: %s (independent item check: meets current conditions; priced below %s) - remove %s, cart %s, refresh its stock and price, then order.%s' % (rival, x, x, rival, self.shelf_ok_note(rival))
        return self._NOT_RUN + bested
    return self._NOT_RUN + 'Checklist before ordering - [cart] ' + ('a single line for every requested product, each at its quantity' if needs.items() else 'a single line: the chosen item, in the quantity asked') + ', nothing more. [fresh read] since the latest cart change, item stock/price read again through a tool that reports them; available; unit price x quantity <= the current budget. [fit] its own properties meet every stated condition, disclosed shopper need and announced change. [rank] if best/cheapest was asked, nothing reachable beats it. [rules] every procedural step of the request is done. Anything unverified: verify it now. All verified: repeat the order call.'
  def _smv_fin_steps(self, st): return self.finish_steps(st['mode'])
  def _smv_fin_misfit(self, st):
    faults, mark = self.cart_misses(), tuple(sorted(self.barrow.units))
    told = vars(self).setdefault('misfit_told', {})
    if not faults or told.get(mark, 0) >= 2 or self.steps_rem() < 1: return ''
    told[mark] = told.get(mark, 0) + 1
    self.log.append({'kind': 'misfit_hold', 'lines': list(mark), 'faults': faults})
    return "%sThe records rule this cart out: %s. A basket like that is never accepted. Swap that line for an already read listing that passes (for a fit, one whose page names its anchor's identifiers), then order; if no read listing does, or time is short, decline instead." % (self._NOT_RUN, '; '.join(faults)[:900])
  def cart_misses(self):
    out = []
    try:
      for x in self.barrow.units:
        if fails := self.item_view(x)['fails']: out.append('%s: %s' % (x, fails[0]))
      if self._dpn_denied(self.needs.items()):
        out += self.dep_errors() or ['the page of a dependent line names models other than those of its anchor line']
    except Exception as e: self.log.append({'kind': 'misfit_error', 'error': self._verr(e)})
    return out
  def _smv_fin_near_hold(self, st): return '' if self.cart_misses() else self.near_keep()
  def _smv_fin_told(self, st):
    cart = self.barrow.units
    if not self._told_now() or self.dead_lines() & set(cart): return ''
    if vars(self).get('told_held', 0) >= 1 or any(self._tag(x) not in self.passed for x in cart): return ''
    if self._dpn_denied(self.needs.items()) or self._costly(cart): return ''
    self.told_held = vars(self).get('told_held', 0) + 1
    self.log.append({'kind': 'decline_told', 'lines': [str(x) for x in cart]})
    return '%sThe shopper was told this exact basket (%s) and replied without objection, condition or change; nothing has moved since. A reply that acknowledges the told basket without objecting is the approval the request asks for; this shopper never writes a bare yes. Order this cart now. Decline only if a record shows a cart line breaking a stated condition.' % (self._NOT_RUN, self._join_list(list(cart)))
  def _smv_fin_dropped(self, st):
    needs = self.needs
    if self.barrow.units or needs.items() or not needs.let_go or not self._room(3): return ''
    if vars(self).get('dropped_held') == len(needs.let_go): return ''
    rows = sorted((c, s) for s, (c, _, _) in self.shelf_rows().items()
           if s in self.rec_db.read and s not in self.refused_lines)
    if not rows: return ''
    self.dropped_held, named = len(needs.let_go), [s for _, s in rows[:3]]
    vars(self).setdefault('named', set()).update(named)
    self.log.append({'kind': 'decline_dropped', 'rows': named})
    return '%sConditions were dropped after the search started (%s), so listings opened already may fit now. Cheapest opened, in-stock rows of the requested category within the budget: %s. Check them against the conditions still in force; cart the cheapest that fits, tell the basket if a message is left, then order. Decline only if not one of them fits.' % (self._NOT_RUN, '; '.join(map(str, needs.let_go))[:200], self._join_list(named))
  def _smv_fin_early(self, stt):
    if self.barrow.units or self.needs.items() or self.steps_rem() < max(6, self.step_cap // 4): return ''
    n = vars(self).get('early_held', 0)
    if n >= 2: return ''
    self.early_held, top = n + 1, {}
    for e in self._book().values():
      g = self.walk_score(e['args'], None)
      for s in e.get('row_shelf') or {}: top[s] = max(top.get(s, 0), g)
    named = [s for _, _, s in sorted((-top.get(s, 0), c, s) for s, (c, _, _) in self.shelf_rows().items()
                    if s not in self.rec_db.read and s not in self.refused_lines)[:3]]
    self.log.append({'kind': 'decline_early', 'left': self.steps_rem(), 'rows': named})
    extra = ' Unopened rows of the requested category, rows of searches for a stated value first, then by lowest price: %s.' % self._join_list(named) if named else ''
    return "%sNothing is in the cart and %d steps remain, so this decline is premature. Before giving up: read further pages of the search that gave the closest rows, search every stated value together with the product word, and browse the product's shelf under the budget; open the strongest unread candidates.%s %s" % (self._NOT_RUN, self.steps_rem(), extra, 'Cart the cheapest one meeting every stated condition; decline only if none does.')
  def _smv_fin_whole(self, stt):
    rows, cart = self.needs.items(), self.barrow.units
    if len(rows) < 2 or not cart or not self._room(1) or vars(self).get('whole_held', 0) >= 2: return ''
    slots, total = set(), 0.0
    for ln, n in cart.items():
      slot = self.prod_idx(ln)
      if slot is None or self._tag(ln) not in self.passed: return ''
      if ln in self.refused_lines or ln in self.dead_lines(): return ''
      slots.add(slot)
      total += float(p) * float(n) if is_num(p := self.rec_db.price(ln)) else 0.0
    cap = self.needs.spend_cap
    if len(slots) < len(rows) or is_num(cap) and total > float(cap) + 1e-09 or self._dpn_denied(rows):
      return ''
    self.whole_held = vars(self).get('whole_held', 0) + 1
    self.log.append({'kind': 'decline_whole', 'lines': [str(ln) for ln in cart]})
    return "%sEach requested product already has a cart line that passed the independent item check on the conditions now in force; all are on sale and the total fits the budget (%s). Declining would discard a complete basket. No dependent line's page names a model other than its anchor's, so a fit the pages merely leave unconfirmed proves no misfit. Order this cart. Decline only if a record shows a line breaking a stated condition." % (self._NOT_RUN, self._join_list(list(cart)))
  def _firstln(self):
    first = {}
    for ln in self.barrow.units: first.setdefault(self.prod_idx(ln), ln)
    first.pop(None, None)
    return first
  def _dpn_denied(self, rows):
    if all(r.get('depends_on') is None for r in rows): return False
    first = self._firstln()
    for k, _, jx in self._links(rows):
      if k not in first or jx not in first: continue
      base, dpn = self._symtxt(first[jx]), self._symtxt(first[k])
      enc, marks = self._coded(base, dpn), self._marks(base)
      if enc is False or enc is None and marks and self._marks(dpn) and not any(
          self._planar(m) in self._planar(dpn) for m in marks):
        return True
    return False
  def _smv_fin_kept(self, stt):
    ok = [s for s in self.barrow.units if self._tag(s) in self.passed and s not in self.refused_lines
       and s not in self.dead_lines()]
    if not ok or not self._room(1) or self._attr('decline_kept') or self.cart_misses(): return ''
    self.decline_kept = True
    return '%sA verified cart line is present (%s): the independent check passed it on every condition now in force, and a decline would throw it away. Check its stock and price again, then order%s. Decline only if a later change plainly rules it out.' % (self._NOT_RUN, self._join_list(ok), ' with a line for every product' if self.needs.items() else '')
  @staticmethod
  def _nreads(queue, spread): return -(-len(queue) // spread[2]) if spread and len(queue) >= 2 else len(queue)
  def _smv_fin_slack(self, stt):
    rows = self.wide_rows() if not self._attr('slack_done') and (
      self.role('details') or self.batch_tool()) else []
    cost = self._nreads(rows, self.batch_tool())
    if not rows or not self._room(cost + 4, 40): return ''
    self.slack_done, self.open_queue, self.open_say = True, rows, 'Reading the cheaper rows first: together they may bring every product under the budget.'
    total = self._attr('slack_total')
    return "%sBudget squeeze: each product has fits, but the cheapest ones together cost %s, above the %s budget. Unopened rows lie within each product's share of the budget (the budget minus the other products' cheapest fits); the code is opening them for you now: %s. Judge each on its own product's conditions by the value rules (a value field dedicated by this store counts only by its wording). A combination that covers every product and stays affordable -> one line per product, then order. Decline only when no combination works." % (self._NOT_RUN, '%g' % total if is_num(total) else 'more than the budget', self.needs.spend_cap, self._join_list(rows))
  def _smv_fin_fill(self, stt):
    spread = stt['spread'] = self.batch_tool()
    rows = self.empty_rows() if not self._attr('decline_fill') and (self.role('details') or spread) else []
    if not rows or not self._room(self._nreads(rows, spread) + 3, 40): return ''
    self.decline_fill, self.open_queue, self.open_say = True, rows, 'Before any decline, opening never-read listings of products that lack a verified fit.'
    return "%sSome requested products have no verified fit yet, while affordable listings of their categories remain unopened. Ahead of any decline the code is opening them for you now: %s. Judge each on its own product's conditions by the value rules (a value field dedicated by this store counts only by its wording). Every product covered by a fitting listing -> one line per product, then order. Decline only for a product that truly has none." % (self._NOT_RUN, self._join_list(rows))
  def _smv_fin_named(self, stt):
    spread = stt['spread']
    q = self.named_unread()[:spread[2] if spread else 3]
    if not q or self._attr('decline_opens') or not (self.role('details') or spread): return ''
    if not self._room(self._nreads(q, spread) + 3, 40): return ''
    self.decline_opens, self.open_queue = True, q
    return '%sUnread leads: %s turned up in the checks but were never opened. The code fetches their records right now, before declining. Value rules: the field named after the property decides (a value the store wrote there counts by its wording alone); without such a field, another field, the option menu or the listing title can supply the value. Buy whichever fits; decline only if every lead fails.' % (self._NOT_RUN, self._join_list(q))
  def _prefixed(self, note): return self._NOT_RUN + note if note else ''
  def _smv_fin_pages(self, stt): return self._prefixed(self.decline_reads())
  def _smv_fin_reaudit(self, stt):
    ok = len(self.rec_db.read) > self._attr('audit_reads', 10 ** 9) and self._attr('audits', 0) < 2
    if not (ok and self.decline_vetted and self._room(3, 30)): return ''
    return self._prefixed(self.judge_decline())
  def _smv_fin_audit(self, stt):
    if self.decline_vetted or not self._room(3, 30): return ''
    self.decline_vetted = True
    return self._prefixed(self.judge_decline()) or self._NOT_RUN + 'Before declining, confirm: [search] several angles tried (different words, filters, later pages); [read] the own properties of every candidate opened; [fail] each candidate breaks a stated condition, a disclosed need, the budget or stock. A field key counts however it is spelt. A fitting item is found -> buy it. None -> send the decline call again.'
  def _smv_fin_fit_hold(self, stt): return self._prefixed(self.fit_keep())
  def _smv_fin_best(self, stt):
    stash = [] if self._attr('best_held') or not self._room(4) else self.undisputed()
    for _ in range(3):
      todo = [s for s in stash if self._tag(s) not in self.passed]
      if not todo or self.cable.spare() < CLOSING + 30: break
      self.check_item(todo[0], 'ORDER')
      stash = self.undisputed()
    if not stash or not self._room(len(stash) + 3, 20): return ''
    self.best_held, several = True, bool(self.needs.items())
    return '%sAn option is still standing. Declining is right only when no listing can meet the conditions; yet the records still hold %s (in stock, within the budget, excluded by no check): %s. Unless a record shows one of them breaking a condition now in force, put %s in the cart, read stock and price again, and order; decline only when every option clearly fails.' % (self._NOT_RUN, 'options covering all products' if several else 'an option', self._join_list(stash), "every product's line (each in its quantity)" if several else 'it')
  def _smv_fin_unknown_ref(self, stt):
    refs = [h for h in self._attr('refs') or [] if h.get('relation') == 'differ'
        and h.get('status') in ('open', 'stuck') and self.ref_live(h)]
    if not refs or not self._room(3) or self._attr('ref_decline_held'): return ''
    self.ref_decline_held = True
    return '%sUnresolved reference: %s depends on a listing the code could not identify, so its value is unknown. An unknown value excludes nothing and is no reason to decline. Cart the listing that meets all the other conditions (in stock, affordable)%s, refresh its stock and price, then order; decline only if no listing meets those other conditions.' % (self._NOT_RUN, self._join_list(dict.fromkeys(h['head'] for h in refs)), ', one per product' if self.needs.items() else '')
  def _smv_fin_repair(self, stt): return self.decline_repair()
  def _smv_fin_meaning(self, stt): return self.near_check()
  def change_review(self):
    cable, book, cart, needs = self.cable, self.rec_db, self.barrow, self.needs
    held = self._attr('reviews', 0)
    if held >= 3 or not book.read or cable.hush: return ''
    if cable.spare() < CLOSING + 20 or not cable.can_serve(CHECKER) or self.steps_rem() < 3: return ''
    self.reviews = held + 1
    cards = book.cards(14000, '', self.ref_extra())
    if not cards: return ''
    segs = [VARIANT_JUDGE, "THE SHOPPER'S REQUEST:\n" + self.request + needs.var_opts(),
        'WHAT JUST CHANGED: ' + ' / '.join(needs.trades[-2:]), 'REQUIREMENTS NOW: ' + needs.body_text(),
        ('PRIORITY: %s %s' % (needs.priority(), needs.parsed.get('priority_detail') or '')).rstrip(),
        'THE CART: ' + cart.recap(),
        'LISTINGS SEEN - DETAILS card = whole record read; ROW card = search-row data only:\n' + '\n'.join(cards)]
    reply = cable.consult('\n\n'.join(segs) + self.code_reads() + self.field_names(), INSPECT_TERMS, CHECKER)
    self.log.append({'kind': 'change_review', 'answer': reply})
    if not reply: return ''
    known = lambda x: isinstance(x, (str, int)) and str(x) in book.ids
    good = [str(x) for x in reply.get('fits') or [] if known(x)][:6]
    pick = str(reply.get('pick')) if known(reply.get('pick')) else ''
    self.__dict__.setdefault('named', set()).update(good)
    out = ['[after the change]', 'opened records that fit, best first: ' + self._join_list(good) if good else 'no record opened so far is shown to meet the current conditions']
    if pick: out.append('pick now: ' + pick)
    if cart.units and isinstance(reply.get('cart_still_right'), bool): out.append('no cart change: its line still qualifies, and the basket described to the shopper should be the one that ships' if reply.get('cart_still_right') else "replace the cart's line: under the current terms it is no longer the right pick (ask the shopper first when the instructions require a check before any replacement)")
    if reply.get('reason'): out.append(str(reply.get('reason'))[:300])
    out.append('Priority now: choose again among the current fits (the cheapest/best of them if the request sets that priority), fix the cart, refresh stock and price, finish. An opened record still fits -> no decline.')
    return ' '.join(out)
  def asks_swap(self, old, new):
    tool = self.role('message')
    field = self.text_arg(tool) if tool else None
    if not field or self.ask_cap() - self.msgs_sent < self.msg_reserve() + 1 or self.steps_rem() < 4:
      return False
    gone, put = old[:3], new[:3]
    question = 'Before I change the basket: I would like to take out %s and put in %s instead, since the first no longer fits. Is that all right with you?' % (self._join_list(gone), self._join_list(put))
    replies = self.emit_all([('code_swap_ask_%s' % self.cable.ticks, tool, {field: question})])
    self.log.append({'kind': 'code_swap_ask', 'old': gone, 'new': put})
    return bool(replies) and not str(replies[0][1]).startswith('Not run')
  @staticmethod
  def _names(stratum, symbol):
    return any(isinstance(v, str) and v == symbol and 'id' in str(k).lower() for k, v in stratum.items())
  def _walk(self, fn, start=0, **kw):
    for r in self.rec_db.items[start:]: walk_dicts(r, fn, **kw)
  def line_seller(self, widget):
    got, out = self.seller_keys(), []
    def scan(d):
      if self._names(d, widget):
        out.extend(v.strip() for k, v in d.items() if isinstance(v, str) and v.strip() and len(v) <= 60 and 'id'
             not in str(k).lower() and (re.search(SELLER_KEYS, str(k).lower()) or str(k) in got))
    self._walk(scan)
    return out[-1] if out else ''
  def _symtxt(self, symbol):
    got = []
    self._walk(lambda d: self._names(d, symbol) and all_strs(d, got))
    return ' '.join(dict.fromkeys(got))
  @staticmethod
  def _marks(text): return list(dict.fromkeys(re.findall('\\b([A-Z][A-Za-z]{1,12}[ -]?\\d{3,6})\\b', text)))[:8]
  @staticmethod
  def _codes(text):
    return (set(re.findall('(?<![\\w-])([A-Z]{1,4}\\d{1,3})(?![\\w-])', text)),
        set(re.findall('(?<![\\w-])([A-Z]{1,4}\\d{1,3})-\\d{3,6}(?![\\w-])', text)))
  @staticmethod
  def _cstem(code): return code.rstrip('0123456789')
  def _coded(self, basis_leaf, dpn_leaf):
    (bs, bc), (ds, dc) = self._codes(basis_leaf), self._codes(dpn_leaf)
    if bs & dc or ds & bc: return True
    clans = {self._cstem(c) for c in bs}
    return False if any(self._cstem(h) in clans for h in dc) else None
  def _page_seq(self, page):
    memo, n = vars(self).setdefault('_smv_stem_memo', {}), len(self.rec_db.items)
    if memo.get('size') != n:
      got = []
      self._walk(lambda d: all_strs(d, got))
      memo.update(size=n, stems={self._cstem(h) for h in self._codes(' '.join(dict.fromkeys(got)))[1]})
    return sorted(c for c in self._codes(page)[0] if self._cstem(c) in memo['stems'])
  @staticmethod
  def _planar(text): return ''.join(c for c in str(text).lower() if c in 'abcdefghijklmnopqrstuvwxyz0123456789')
  @staticmethod
  def _links(rows):
    return [(k, r, t) for k, r in enumerate(rows or []) for t in [r.get('depends_on')]
        if t is not None and t != k and t < len(rows)]
  def dep_errors(self):
    rows = self.needs.items()
    if not rows or all(r.get('depends_on') is None for r in rows): return []
    first, out = self._firstln(), []
    for k, row, jx in self._links(rows):
      if k not in first or jx not in first: continue
      bl, dl, base, dpn = first[jx], first[k], self._symtxt(first[jx]), self._symtxt(first[k])
      enc, names = self._coded(base, dpn), self._marks(base)
      marks = self._page_seq(base) + names
      if enc or not marks or enc is None and any(self._planar(m) in self._planar(dpn) for m in names):
        continue
      out.append('line %s (the %s) belongs with %s (the %s), yet its page cites none of the identifiers that %s carries (%s)' % (dl, row['product'], bl, rows[jx]['product'], bl, self._join_list(marks)))
    return out
  def _fit_built(self):
    rows = self.needs.items()
    if not rows: return False
    where, first = self._row_shelves(), self._firstln()
    for k, _, jx in self._links(rows):
      if k not in first or jx not in first: continue
      base = self._symtxt(first[jx])
      marks, enc = [self._planar(m) for m in self._marks(base)], bool(self._page_seq(base))
      own, shelf = first[k], where.get(first[k])
      if (marks or enc) and shelf and any(lab == shelf and s != own and (
          enc and self._coded(base, self._shelf_page(s)) is not None
          or any(m and m in self._shelf_text(s) for m in marks)) for s, lab in where.items()):
        return True
    return False
  def _shelf_page(self, symbol):
    memo, n = vars(self).setdefault('_smv_flat_memo', {}), len(self.rec_db.items)
    if memo.get('size') != n: memo.clear()
    memo['size'], key = n, ('page', symbol)
    return memo[key] if key in memo else memo.setdefault(key, self._symtxt(symbol))
  def _shelf_text(self, symbol):
    page, memo = self._shelf_page(symbol), vars(self)['_smv_flat_memo']
    return memo[symbol] if symbol in memo else memo.setdefault(symbol, self._planar(page))
  def dep_hint(self):
    rows, out = self.needs.items(), []
    for k, row, jx in self._links(rows):
      base = next((ln for ln in self.barrow.units if self.prod_idx(ln) == jx), '')
      page = self._symtxt(base) if base else ''
      marks = self._page_seq(page) + self._marks(page) if base else []
      dpn, anchor, enc = row['product'], rows[jx]['product'], [] if marks else self._anchors(jx)
      tail = self._code_words(k)
      if marks: out.append('%s (goes with cart line %s, the %s): search %s listings once for each identifier (%s), open the hits and keep one whose page cites any of them; if no page cites one, this store does not write the fit and it settles nothing%s' % (dpn, base, anchor, dpn, self._join_list(marks), tail))
      elif enc: out.append('%s (goes with the %s): here the fit is written as a code. A %s page gives a series code (seen so far: %s), and a %s fits only if its listing tag begins with that very code; a tag from the same series with a different number is no fit. Cart a pair that shares one code, searching %s listings with the code as the query text%s' % (dpn, anchor, anchor, self._join_list(enc), dpn, dpn, tail))
      else: out.append('%s (goes with the %s): settle the %s line first; then search %s listings by model names/numbers and keep a page that quotes one' % (dpn, anchor, anchor, dpn))
    if not out: return ''
    dyad = self.pair_best()
    if dyad: out.append('cheapest total so far: %s, %g altogether; linked products are priced as one, so a cheaper line whose code only a pricier partner shares does not beat it' % (self._rated(list(dyad)), sum(self.rec_db.price(s) * u for s, u in dyad.items())))
    return '\n[works with] ' + '; '.join(out)
  def pair_best(self):
    links = self._links(self.needs.items()) if self.needs.priority() == 'cheapest' else []
    if not links or not self.rec_db.read or self._open(): return {}
    stamp = (len(self.rec_db.items), self.needs.rev_no, '|'.join(self.refused_lines),
        len(self.dead_lines()))
    memo = vars(self).get('pair_memo')
    if memo and memo[0] == stamp: return memo[1]
    anchor = links[0][2]
    tied, info, units = [k for k, _, jx in links if jx == anchor], self.all_rows(), self.needs.counts()
    fits, best, low = {k: self._pair_fits(k, info) for k in [anchor, *tied]}, {}, None
    for cost, base in fits[anchor]:
      codes = set(self._page_seq(self._symtxt(base)))
      mates = [next((c for c in fits[k] if self._codes(info[c[1]]['title'])[1] & codes), None)
          for k in tied] if codes else [None]
      if None in mates: continue
      total = cost * units[anchor] + sum(m[0] * units[k] for m, k in zip(mates, tied))
      if low is None or total < low:
        low, best = total, {base: units[anchor], **{m[1]: units[k] for m, k in zip(mates, tied)}}
    self.pair_memo = (stamp, best)
    return best
  def _pair_fits(self, k, information):
    judged = [(p, v) for p, v in self.need_vals(k) if not Errand._smm_BOUND.search(v.lower())]
    out, found = self.dead_lines() | set(self.refused_lines), []
    for s in self.rec_db.read:
      row, cost = information.get(s), self.rec_db.price(s)
      if row is None or not is_num(cost) or s in out or row['stock'] is False or self.prod_idx(s) != k or (
          self.item_view(s)['fails']): continue
      if all(self.row_shows(row, p, v, set(), True) for p, v in judged): found.append((float(cost), s))
    return sorted(found)
  def _code_words(self, k):
    words = v[0][0] if (v := self._tvals(k)) and v[0] else []
    return " (search '<identifier> %s' so the list holds only hits that also carry that value; it stays short, read it to the end)" % ' '.join(words) if words else ''
  def show_note(self, lines):
    kv = vars(self)
    due = kv.setdefault('reveal_due', [])
    if self.needs.items(): return
    for line in lines:
      v = self._bare_val(str(line)).strip()
      first = self._prefs(v)[0].strip() if v else ''
      if not re.search(BOUND_RX, v.lower()) and first and not is_ref_text(v) and (
          len(first.split()) <= 4 and kv.get('reveal_sent', 0) + len(due) < REVEAL_MAX):
        due.append(first)
  def reveal_done(self, link):
    if self.needs.items() or link['relation'] not in ('same', 'pairs'): return
    said = link.get('goes_with') if link['relation'] == 'pairs' else [link.get('value')]
    value = str((said or [''])[0]).strip().lower()
    head, _, tail = value.rpartition(' ')
    shown = (t for row in self.all_rows().values()
        for t in self._prop_fields(row.get('fields') or {}, link['prop']))
    if head and same_prop(tail, link['prop']) and any(self.code_seen(head, t) for t in shown):
      value = head
    if not value or is_ref_text(value) or re.search(BOUND_RX, value) or len(value.split()) > 4: return
    due = vars(self).setdefault('reveal_due', [])
    due[:] = [' '.join([value, *due])]
    self.reveal_deep = due[0]
  def _vidle(self, extra):
    return (self.needs.items() or self.barrow.units or self.orders_tried or not self.pg_arg()
        or not self._room(self.order_due() + extra, 50))
  def _vpage(self, args, turn, say='Following page of that search.'):
    return self.walk_call({**args, self.pg_arg()[0]: turn}, say)
  def reveal_reach(self):
    kv = vars(self)
    if not kv.get('reveal_job') and not kv.get('reveal_due'): return False
    if self._vidle(4):
      kv.pop('reveal_job', None)
      kv['reveal_due'] = []
      return False
    job = kv.get('reveal_job') or self._reveal_job()
    if not job: return self.reveal_reach() if kv.get('reveal_due') else False
    kv['reveal_job'] = job
    e = self._book().get(self._qo_sgn(job['args']))
    if e is None: return self.walk_call(job['args'], 'Searching the newly revealed value together with the product words.')
    turn = self.next_page(e) if job['pages'] < (
      ARRIVE_ENTRYLINES if job['deep'] else REVEAL_PAGES * 4) // max(4, e['size']) else None
    if turn is not None:
      job['pages'] += 1
      return self._vpage(job['args'], turn)
    kv.pop('reveal_job', None)
    kv['reach_done'] = kv.get('reach_done') or job['deep']
    return self._reveal_open([e], job['deep'])
  def _vargs(self, form, text):
    args, faults = ArgCheck.vet({**form[0], form[1]: text}, self._search_form())
    return None if faults or not isinstance(args, dict) else args
  def _reveal_job(self):
    kv = vars(self)
    value, form = kv['reveal_due'].pop(0), self.search_shape()
    product = str(self.needs.parsed.get('product') or '').strip()
    args = self._vargs(form, f'{value} {product}') if form and product else None
    if args is None: return None
    kv['reveal_sent'] = kv.get('reveal_sent', 0) + 1
    return {'args': args, 'pages': 0, 'deep': value == kv.get('reveal_deep')}
  def _vwalk(self, rows, say):
    if not rows: return False
    self.open_recheck, self.open_purpose, self.open_queue, self.open_say = False, 'walk', rows, say
    return self.open_named()
  def _reveal_open(self, postings, whole=False):
    home = self.walk_home(None)
    shown = [s for e in postings for s, shelf in (e.get('row_shelf') or {}).items() if not home or shelf in home]
    pick = self.walk_pick(None)
    rows = [n for _, n in self.cheap_unread(shown, None, pick[1] if pick else None)]
    rows = rows[:2 * REVEAL_OPENS] if whole else sorted(
      rows, key=lambda n: not self.row_has(n, None))[:REVEAL_OPENS]
    return self._vwalk(rows, 'Opening the lowest-priced rows of that search.')
  def solo_reach(self):
    kv = vars(self)
    if kv.get('reach_done') or not (self.msgs_sent or not self.role('message')): return False
    if self._vidle(3) or self._cap_done(): return False
    spent = kv.get('reach_rows', 0)
    e = self._first_in() if spent < ARRIVE_ENTRYLINES else None
    if e is None:
      kv['reach_done'] = True
      return bool(spent) and self._reveal_open(list(self._book().values()))
    kv['reach_rows'] = spent + max(1, e['size'])
    return self._vpage(e['args'], self.next_page(e), 'Following page of the closest search.')
  def _first_in(self):
    home, hounded = self.walk_home(None), {t for link in self._rlinks() for t in link.get('queries') or []}
    best, top = None, (0, 0)
    for e in self._book().values():
      if ' '.join(self.arg_texts(e['args'])).lower() in hounded: continue
      n = sum(1 for shelf in (e.get('row_shelf') or {}).values() if not home or shelf in home)
      rank = (self.walk_score(e['args'], None), n)
      if n and rank > top and self.next_page(e) is not None: best, top = e, rank
    return best
  def _cap_tool(self, merely=None):
    for name, shape in self.argspec.items():
      props = shape.get('properties') if isinstance(shape, dict) and name == (merely or name) else None
      if not isinstance(props, dict): continue
      ws = [k for k, s in props.items() if isinstance(s, dict) and s.get('type') == 'string' and not s.get('enum')]
      shelf = next((k for k in ws if self._cat_rx.search(k.lower())), None)
      turn = next((k for k in ws if re.search('token|cursor|next|continu', k.lower())), None)
      limit = self._cap_arg(props, turn) if shelf and turn else None
      if limit and not set(shape.get('required') or []) - {shelf, limit}: return (name, shelf, limit, turn)
    return None
  def _cap_page(self, notch):
    for obs in self.rec_db.items[notch:]:
      if not isinstance(obs, dict): continue
      rows = self._rows_of(obs)
      if rows is None:
        obs = next((s for s in obs.values() if isinstance(s, dict) and self._rows_of(s)
              is not None), obs)
        rows = self._rows_of(obs)
      if rows:
        return (len(rows), max([n for n in obs.values() if type(n) is int and n > len(rows)],
                   default=len(rows)), self.next_token(obs))
    return (0, 0, None)
  def _cap_done(self):
    job = vars(self).get('cap_job') or {}
    if not job.get('complete'): return False
    rows = self.shelf_rows()
    return any(s in self.rec_db.read and rows[s][0] <= job['bound'] and not self.item_view(s)['fails']
         for s in rows)
  def cap_reach(self):
    kv = vars(self)
    if kv.get('cap_done') or not self._wants_cheap() or not (self.msgs_sent or not self.role('message')):
      return False
    found, label = self._cap_tool(), self.raw_label(self.walk_home(None))
    prices = [r[0] for r in self.shelf_rows().values()]
    if self.needs.items() or self.barrow.units or self.orders_tried or not (found and label and prices) or (
        not self._room(self.order_due() + 3, 50)):
      kv['cap_done'] = True
      return False
    job = kv.setdefault('cap_job', {'calls': 0, 'bound': self.below(min(prices)), 'token': None, 'lifted': False})
    if job['bound'] is None or job['calls'] >= CAP_CALLS or job.get('complete'): return self._cap_open()
    (tool, shelf, limit, turn), notch, beat = found, len(self.rec_db.items), self.cable.ticks
    self.walk_call({shelf: label, limit: job['bound'], **({turn: job['token']} if job['token'] else {})}, 'Listing the category below the cheapest row seen: replies come in id order, and the cheapest is wanted.', tool)
    if self.cable.ticks == beat: return self._cap_open()
    job['calls'] += 1
    shown, total, token = self._cap_page(notch)
    if not shown and not job['lifted']: job.update(lifted=True, bound=min(prices) * 2)
    elif not shown or total <= CAP_ROWS: job.update(token=token, complete=token is None)
    else:
      low = self.ceil_mid(min(r[0] for r in self.shelf_rows().values()), job['bound'])
      job.update(bound=low, complete=not low < job['bound'])
    return True
  def _cap_open(self):
    self.cap_done = True
    b = self.batch_tool()
    rows = [n for _, n in self.cheap_unread(list(self.shelf_rows()), None)][
      :2 * b[2] if b else REVEAL_OPENS]
    return self._vwalk(rows, 'Opening the lowest-priced rows of the category.')
  def few_reach(self):
    kv = vars(self)
    if kv.get('few_done') or not (self.msgs_sent or not self.role('message')): return False
    if self._open() and self.ref_room(): return False
    rows = self.needs.items()
    if not (self._wants_cheap() and 2 <= len(rows) <= FEW_MAX) or self.barrow.units or self.orders_tried or (
        not self.pg_arg()) or not self._few_room():
      kv['few_done'] = True
      return False
    tied = {k for k, _, _ in self._links(rows)}
    job = kv.setdefault('few_job', {'todo': [k for k in range(len(rows)) if k not in tied and self._tvals(k)],
                    'args': None, 'rows': 0, 'sent': False, 'read': [], 'hunts': None})
    beat = self.cable.ticks
    for step in (self._few_page, self._few_open, self._few_hunt,
          self._few_tagged):
      step(job)
      if self.cable.ticks != beat: return True
    kv['few_done'] = True
    return False
  def _few_room(self): return self._room(self.order_due() + FEW_KEEP, LAST_CALL - CLOSING)
  def _few_page(self, job):
    while job['todo']:
      if job['args'] is None:
        job.update(args=self._few_args(job['todo'][0]) or {}, rows=0)
        job['sent'] = self._qo_sgn(job['args']) in self._book()
      args = job['args']
      if args and not job['sent']:
        job['sent'] = True
        return self.walk_call(args, 'Searching for the required value together with the category words: the cheapest fit may sit deep in the list.')
      e = self._book().get(self._qo_sgn(args)) if args else None
      turn = self.next_page(e) if e and job['rows'] < FEW_ROWS and self._few_room() else None
      if turn is not None:
        job['rows'] += max(1, e['size'])
        return self._vpage(args, turn)
      job['read'].append(job['todo'].pop(0))
      job['args'] = None
  def _few_args(self, k):
    form, label = self.search_shape(), self.raw_label(self.walk_home(k))
    ws = self._forms(k, label, None)[0] if label else []
    return self._vargs(form, ws[0]) if form and ws else None
  def _few_open(self, job):
    due, job['read'], q = job['read'], [], []
    for k in due:
      q += [n for _, n in self.cheap_unread(self.home_rows(self.walk_home(k)), k)][
        :RUNDOWN_SKIM_PEAK]
    self._few_start(q, due, 'Reading the lowest-priced rows of the value searches.')
  def _few_start(self, queue, products, say):
    floor = self.order_due() + FEW_KEEP
    queue = self._pare(queue, self.batch_tool(), floor)
    if queue: self.zl_start(queue, set().union(*[self.walk_home(k) for k in products]), floor, say)
  def _few_hunt(self, job):
    if job['hunts'] is None: job['hunts'] = self._few_hunts()
    form = self.search_shape()
    while job['hunts'] and form and self._few_room():
      text = job['hunts'].pop(0)
      args = self._vargs(form, text)
      if args is not None and not self.used_query(text): return self.walk_call(args, "Sending '%s' as a keyword query: the anchor's series code leads." % text)
  def _pcodes(self, jx, info):
    return [c for _, base in self._pair_fits(jx, info) for c in self._page_seq(self._symtxt(base))]
  def _few_hunts(self):
    if self._open(): return []
    rows, out, info = self.needs.items(), [], self.all_rows()
    for k, _, jx in self._links(rows):
      codes, v = list(dict.fromkeys(self._pcodes(jx, info))), self._tvals(k)
      out += [' '.join([c, *w]) for c in codes[:PAIR_ANCHORS] for w in (v[0][:PAIR_ALTS] if v and v[0] else [[]])]
    return out
  def _few_tagged(self, job):
    if job.get('tagged'): return
    job['tagged'], rows = True, self.needs.items()
    info, q, products = self.all_rows(), [], []
    for k, _, jx in self._links(rows):
      products.append(k)
      codes, shelf = set(self._pcodes(jx, info)), self.home_rows(self.walk_home(k))
      for code in sorted(codes):
        tagged = [n for n in shelf if n in info and code in self._codes(info[n]['title'])[1]]
        q += [n for _, n in self.cheap_unread(tagged, k)][:PAIR_OPENS]
    self._few_start(q, products, 'Reading the lowest-priced tagged rows of each code.')
  def fit_search(self):
    kv = vars(self)
    queue = kv.get('fit_queue') or []
    if not queue: return False
    floor = self.order_due() + 2
    if not self._room(floor + 2, 15):
      kv.pop('fit_queue', None)
      return False
    (k, code), sch = queue[0], self._search_form()
    p, tried = sch.get('properties'), kv.setdefault('fit_tried', {})
    job = {'k': k, 'schema': sch, 'props': p if isinstance(p, dict) else {}}
    forms, used = self._fit_forms(k, code, job), tried.get(k, [])
    todo = [t for t in forms if t not in used][:max(0, FIT_TRIES - len(used))]
    if not todo:
      queue.pop(0)
      return self.fit_search()
    text = todo[0]
    tried.setdefault(k, []).append(text)
    args, faults = ArgCheck.vet({**job['tmpl'], job['tkey']: text}, sch)
    if faults or not isinstance(args, dict): return self.fit_search()
    notch, beat = len(self.rec_db.items), self.cable.ticks
    if not self.walk_call(args, "Sending '%s' as a keyword query: the series code leads, so the reply lists the listings whose tag carries it." % text):
      return self.cable.ticks != beat or self.fit_search()
    home, tagged = self.walk_home(k), []
    def pin(d):
      if self._is_row(d) and self.shelf_for(d) in home and code in self._codes(
          self.title_of(d))[1]:
        tagged.append(self.row_key(d))
    self._walk(pin, notch, ceiling=3)
    tagged = list(dict.fromkeys(tagged))
    if not tagged and len(tried[k]) < FIT_TRIES and len(todo) > 1:
      self._append("[code fit] '%s': no %s row carries the tag %s." % (text, self.needs.items()[k]['product'], code), True)
      return True
    queue.pop(0)
    batch = self.batch_tool()
    opens = self._pare([n for _, n in self.cheap_unread(tagged, k)][:min(batch[2] if batch else FIT_OPENS,
               FIT_OPENS)], batch, floor)
    note = "[code fit] '%s': %s row(s) carrying the tag %s: %s." % (text, self.needs.items()[k]['product'], code, self._join_list(tagged) or 'none')
    if opens: note += ' The unread ones are opened next: %s.' % self._rated(opens)
    self._append(note, not opens)
    self.log.append({'kind': 'fit_search', 'product': k, 'code': code, 'query': text, 'tagged': tagged, 'opened': opens})
    if opens: self.zl_start(opens, home, floor, 'Reading the hits tagged with the code: each carries the series code its anchor gives.')
    return True
  def _fit_forms(self, k, code, job):
    label = self.raw_label(self.walk_home(k))
    if self._vs_schema(job) != 'forms': return []
    tail = self._label_words(k, label) if label else [
      w for w in self._query_words(self.needs.items()[k]['product']) if len(w) > 2]
    v, cap, out = self._tvals(k), job.get('cap_len'), []
    for words in [*(v[0][:FIT_TRIES - 1] if v and v[0] else []), []]:
      kept = list(dict.fromkeys([code, *words, *tail]))
      text = ' '.join(kept[:self.lead_words] if self.lead_words else kept)
      if is_num(cap):
        while ' ' in text and len(text) > cap: text = text.rsplit(' ', 1)[0]
        text = text[:int(cap)]
      out.append(text)
    return list(dict.fromkeys(filter(None, out)))
  def _anchors(self, jx):
    return ['%s %s' % (s, '/'.join(c)) for s in self._row_shelves()
        if self.prod_idx(s) == jx and (c := self._page_seq(self._shelf_page(s)))][-4:]
  def cheap_sweep(self):
    if self.needs.priority() != 'cheapest' or self.orders_tried or self._attr('cheap_swept'): return ''
    tool = self.role('search')
    if not tool or not self.barrow.units or self.steps_rem() < 12 or self.cable.spare() < CLOSING + 40:
      return ''
    self.cheap_swept = True
    shape = self.argspec.get(tool) or {}
    free = [n for n, s in (shape.get('properties') or {}).items()
        if isinstance(s, dict) and s.get('type') == 'string' and not s.get('enum')]
    if len(free) != 1 or set(shape.get('required') or []) - {free[0]}: return ''
    queries = []
    if not self.needs.items():
      done = {' '.join(self.arg_texts(p['args'])).strip().lower() for p in self._attr('page_book', {}).values()}
      queries = [q for q in self.label_queries() if ' ' not in q and q not in done][:2]
      if queries:
        self.emit_all([('cheap_%d_%s' % (n, self.cable.ticks), tool, {free[0]: q}) for n, q in enumerate(queries)])
    paging, shelves, pages = self.pg_arg(), self.shelf_labels(), 0
    for walk in list(self._attr('page_book', {}).values()) if paging and shelves else []:
      if not set(walk.get('shelves') or ()) & shelves or walk.get('total', 0) > 4 * max(1, walk.get('size', 0)):
        continue
      while pages < RAKE_LEAVES and self.steps_rem() >= 10 and self.cable.spare() > CLOSING + 40:
        turn = self.next_page(walk)
        if turn is None or not self.walk_call({**walk['args'], paging[0]: turn}, 'Reading the remainder of a short result set before ordering: a cheaper fit can sit further along.'):
          break
        pages += 1
    cart, book, several = self.barrow, self.rec_db, bool(self.needs.items())
    where = self._row_shelves() if several else {}
    shelf = lambda s, miss: where.get(s, miss) if several else ''
    tops = {shelf(ln, ''): float(p) for ln in cart.units if (p := book.price(ln)) is not None}
    if not tops or several and '' in tops: return ''
    found = sorted((c, s, t) for s, (c, t, _) in self.shelf_rows().items() if (m := tops.get(shelf(s, '?')))
           is not None and c < m and s not in cart.units and s not in book.read)[:4]
    ids = [r[1] for r in found]
    self.log.append({'kind': 'cheap_sweep', 'queries': queries, 'pages': pages, 'rows': ids})
    if not found: return ''
    self.__dict__.setdefault('named', set()).update(ids)
    listed = '; '.join("%s '%s' %g" % (s, t[:70], c) for c, s, t in found)
    return '%sThe lowest price counts here, so further code searches were sent (%s). Rows of the requested category, in stock and never opened, that cost less than the current cart line%s: %s. Open them all in one batched reply. If one meets every condition, swap it into the cart, refresh its stock and price, then order; if none does, order what the cart already holds.' % (self._NOT_RUN, self._join_list(queries) or 'more pages of the short result lists', ' of their own category' if several else '', listed)
  def _row_shelves(self):
    found = {}
    def note(d):
      s, label = self._record_id(d), self.shelf_for(d)
      if s and label: found.setdefault(s, label)
    self._walk(note)
    return found
  def bound_hint(self):
    if self.needs.priority() != 'cheapest' or self.cable.spare() < CLOSING: return ''
    nums = {}
    for req in self.needs.props_left():
      low = str(req).split(':', 1)[-1].lower()
      for m in _NUM_RX.finditer(low):
        if not (re.search(BOUND_RX, low) and '.' in m[1]): nums.setdefault(m[1] + m[2])
    picks = list(nums)[:3]
    return '\n[cheapest pick, limit given as a measure] The cheapest fit tends to print that exact figure in its own text. Search ' + ', '.join(picks) + ' together with the product word - each figure joined to its unit and also with a space between - and for each such search page through two sets of results; then open the lowest-priced rows whose price stays within the budget, all before the pick is settled.' if picks else ''
class Relay:
  def _cart_key(self):
    n = self.needs
    return (vars(self).get('cart_edits', 0), tuple(sorted(self.barrow.units.items())), tuple(
      tuple(map(str, s)) for s in (n.written, n.uncovered, n.let_go)), n.spend_cap, n.told_rev,
      self.refusal_n)
  def _told_now(self):
    return bool((at := vars(self).get('stated_at')) and self.barrow.units) and at == self._cart_key()
  def _state_sum(self, voiced):
    cart = self.barrow.units
    if not cart: return False
    nums, total, rated, named = set(re.findall('\\d+(?:\\.\\d+)?', str(voiced).replace(',', ''))), 0.0, True, True
    for sid, n in cart.items():
      known = is_num(p := self.rec_db.price(sid))
      total += float(p) * float(n) if known else 0
      rated &= known
      named &= sid in voiced or known and f'{float(p):g}' in nums
    return named or rated and f'{total:g}' in nums
  def sum_text(self):
    try: known = self.all_rows()
    except Exception: known = {}
    cart, bits, total, missing = self.barrow.units, [], 0.0, False
    for sid in cart:
      n, facts, p = cart[sid], known.get(sid) or {}, self.rec_db.price(sid)
      if p is None: p = facts.get('price')
      entry = f'{int(n) if float(n).is_integer() else n} x {sid}{self._wline_tag(sid, facts)}'
      if p is None: missing = True
      else: total, entry = total + float(p) * float(n), entry + f' at {p:g} each'
      bits.append(entry)
    funds = str(self.needs.parsed.get('currency') or '').strip()
    if not bits: return 'Before I finish: the basket is empty, so nothing will be ordered.'
    text = 'Before I finish, here is the basket: ' + '; '.join(bits) + '.'
    if not missing: text += f" The total is {total:g}{' ' + funds if funds else ''}."
    return text + ' I will place the order now.'
  def _wline_tag(self, itemrow_sid, facts):
    tag = f" ({facts['title'][:60]})" if facts.get('title') else ''
    try: seller = self.line_seller(itemrow_sid)
    except Exception: seller = ''
    return f'{tag} from {seller}' if seller else tag
  def finish_steps(self, kind):
    n, mine, wx = self.needs, vars(self), self.cable
    quiet = not (n.tolds or n.trades or mine.get('event_seen'))
    if wx.done or wx.broken or self.steps_rem() < 2: return ''
    edits = mine.get('cart_edits', 0)
    stamp = (kind, tuple(sorted(self.barrow.units.items())), edits, n.told_rev, self.refusal_n, n.rev_no)
    seen = mine.setdefault('finish_checked', set())
    if stamp in seen: return ''
    seen.add(stamp)
    tool = self.role('message')
    left, fresh = max(0, self.ask_cap() - self.msgs_sent) if tool else 0, edits > 0 or not self.barrow.units
    since = ['- [%s] sent: %s' % (b, v) for b, v, at in mine.get('msg_log', []) if at == edits and fresh]
    tools = [s for s in self.tool_specs if (f := s['function']['name']) not in self.end_tools()
        and f != self.role('cart') and not (left <= 0 and f == tool)]
    rules = [f"- {e['text']} (from the {e.get('source')})" for e in n.tolds]
    offer = [f"- {fx['name']}: {fx.get('description', '')} | arguments: {as_text(fx.get('parameters') or {})}"
        for fx in (s['function'] for s in tools)]
    parts = (('INSTRUCTIONS IN FORCE', '\n'.join(rules) or '(none given)'), ("SHOPPER'S ORIGINAL WORDS", self.request), ('RECENT WORDS FROM SHOPPER OR STORE (earliest on top)', '\n'.join(n.trades[-8:]) or '(none)'), ('REQUIREMENTS AT THIS POINT', n.body_text()), ('CART RIGHT NOW', self.barrow.recap() + ' ' + self.sum_text()), ('FINISH ABOUT TO BE SENT', kind), ("ACTIVITY SINCE THE CART'S LATEST EDIT", '\n'.join(since) or '(nothing: no message was sent and no other step ran after it)'), ('MOVES LEFT', f'{left} message(s), {self.steps_rem()} step(s); the finish takes one step'), ('CALLABLE TOOLS FOR AN OPEN STEP', '\n'.join(offer)))
    brief = FINISH_CHECK + ''.join(f'\n\n[{i}] {h}:\n{b}' for i, (h, b) in enumerate(parts, 1))
    answer = wx.consult(brief, 2500, FINISHER) if not quiet and wx.can_serve(FINISHER) and (
      wx.spare() > LAST_CALL + 10) else None
    steps = []
    if answer is not None:
      known = {s['function']['name'] for s in tools}
      for it in (answer.get('pending') or [])[:FINISH_TURNS]:
        if isinstance(it, dict) and it.get('tool') in known and isinstance(it.get('args'), dict):
          args, faults = ArgCheck.vet(dict(it['args']), self.argspec[it['tool']])
          if not faults: steps.append((it['tool'], args, str(it.get('why') or '')[:120]))
    field = self.text_arg(tool) if tool else None
    if kind == 'order' and self.barrow.units and field and not since and left > 0:
      if any(t == tool for t, _, _ in steps):
        steps = sorted([(t, {**a, field: (str(a.get(field) or '').strip() + ' ' + self.sum_text()).strip()}
                if t == tool and not self._state_sum(str(a.get(field) or '')) else a, w)
                for t, a, w in steps], key=lambda s: s[0] != tool)
      else:
        steps.append((tool, {field: self.sum_text()}, 'telling the shopper the cart contents before ordering'))
    self.log.append({'kind': 'finish_check', 'finish': kind, 'answer': answer, 'calls': [t for t, _, _ in steps]})
    steps = steps[:max(0, self.steps_rem() - 1)]
    if not steps: return ''
    batch = [_qw_bid({'id': f'finish_step_{wx.ticks}_{i}', 'function': {'name': t, 'arguments': a}}, i)
        for i, (t, a, _) in enumerate(steps)]
    stamp, former = self._ledger_key(), mine.get('scripted_batch')
    try: replies = self.relay(batch, programmed=True)
    finally: self.scripted_batch = former
    changed = stamp != self._ledger_key()
    res = [f"{t} {json.dumps(a, ensure_ascii=False)[:400]} -> {txt.split(_STATUS_CUT)[0][:700]}"
       for (t, a, _), (_, txt) in zip(steps, replies)]
    note = f"[steps the code ran first: {'; '.join(w for _, _, w in steps)}] " + ' | '.join(res)
    self.log.append({'kind': 'finish_steps_done', 'note': note[:600]})
    if tool and any(t == tool for t, _, _ in steps) and (changed or n.reply_due or wx.done):
      return 'Not run yet (no step used). ' + note + '\nThat reply changed something. Check it against the requirements and every standing instruction, do what it asks, and only then attempt the finish again.'
    self.finish_note = note
    return ''
  def _ledger_key(self):
    n = self.needs
    return (n.rev_no, n.told_rev, self.msg_cap, self.step_cap, self.refusal_n, len(self.barrow.units))
  def relay(self, batch, programmed=False):
    self.scripted_batch = programmed
    self.start_batch()
    allowed = {s['function']['name'] for s in self.on_offer()}
    ct, self.batch_puts = self.role('cart'), []
    self.batch_cart = ct in [(i.get('function') or {}).get('name') for i in batch if isinstance(i, dict)]
    for item in batch:
      fx = item.get('function') or {}
      if fx.get('name') != ct: continue
      raw = fx.get('arguments')
      got = as_json(raw) if isinstance(raw, str) and raw.strip() else raw if isinstance(raw, dict) else {}
      put = self.barrow.aim(got) if isinstance(got, dict) and self.barrow.is_add(got) else None
      if put: self.batch_puts.append(put)
    replies, queue, msgs, grasped = [], [], 0, False
    for bid in batch:
      cid = bid['id']
      refusal, name, args = (_STALE_CART, None, None) if grasped and bid['function']['name'] == self.role(
        'cart') else self.screen(bid, allowed, programmed, len(batch), msgs)
      if refusal:
        if not grasped and name == self.role('cart') and isinstance(args, dict) and args and (
            not self.barrow.is_add(args)):
          grasped = True
        replies.append((cid, refusal))
        continue
      msgs += name == self.role('message')
      queue.append((cid, name, args))
    cut = len(queue)
    if queue and queue[0][1] in self.end_tools():
      held, args = self.gate_finish(*queue[0])
      queue[0] = queue[0][:2] + (args,)
      if held: return replies + [(queue[0][0], held)]
    elif queue and self.end_tools():
      cut = max(0, self.steps_rem() - 1)
      replies += [(c[0], _LAST_STEP) for c in queue[cut:]]
    live = queue[:cut]
    for i in range(0, len(live), 16):
      blk = live[i:i + 16]
      if self.cable.done or self.cable.broken or self.steps_rem() <= 0:
        replies += [(c[0], _OVER_MSG) for c in blk]
      else: replies += self.emit_all(blk[:self.steps_rem()])
    note = vars(self).pop('finish_note', '')
    if note and live and live[0][1] in self.end_tools():
      replies = [(c, f'{note}\n{t}' if c == live[0][0] else t) for c, t in replies]
    rank = {b['id']: i for i, b in enumerate(batch)}
    replies.sort(key=lambda d: rank.get(d[0], 0))
    return replies
  def emit_all(self, block_alt):
    if len(block_alt) > 1: return [r for call in block_alt for r in self.emit_all([call])]
    acts, num = [dict(name=n, args=a) for _, n, a in block_alt], self.cable.ticks + 1
    ans = self.cable.doing(acts)
    self.log.append(dict(kind='env', step=num, actions=acts, reply=ans))
    if ans is None: return [(cid, 'No reply arrived for this call (a network or server failure); it may be sent again.') for cid, *_ in block_alt]
    views = ans['observations'] if isinstance(ans.get('observations'), list) else [ans.get('observation')]
    return [(cid, self.imbibe(n, a, views[i], ans, i == len(views) - 1) if i < len(views) else 'Not run: the store shut down while answering the earlier calls of this step and never reached this one.') for i, (cid, n, a) in enumerate(block_alt)]
  def imbibe(self, name, args, obs, resp, last):
    mine, tell, ct, shown, obs, notes = vars(self), self.role('message'), self.role('cart'), obs, padded(obs), []
    if name == tell:
      self.msgs_sent += 1
      said = _said(args)
      mine.setdefault('msg_log', []).append((self.cable.ticks, said[:600], mine.get('cart_edits', 0)))
      if self._state_sum(said): self.stated_at = self._cart_key()
      if not mine.get('scripted_batch'):
        self.questions_log.append('- (the assistant\'s own message): "%s" -> %s' % (said[:300], as_text(obs)[:300]))
    if name == ct: self._nudge('cart_edits')
    self.last_kind = self.call_kind(name, args)
    if self.last_kind == 'order': self.orders_tried += 1
    reset = name in (ct, tell) or name in self.end_tools()
    self.last_calls = [] if reset else self.last_calls[-1:] + [name + as_text(args)]
    self.rec_db.add(obs)
    self._nudge('absorb_n')
    if name != ct and not mine.get('covering'): self._nudge('since_cart')
    gone = self._attempt('caps_error', lambda: self.infer_caps(name, args, obs), [])
    self._attempt('batch_learn_error', lambda: self.batch_learn(name, args, obs))
    self._attempt('sheet_role_error', self.sheet_cap)
    units = self.barrow.units
    if units:
      fresh, stock, named = set(), bool(mine.get('stock_seen')), self.asked_ids(args)
      only = not (mine.get('rows_live') and not mine.get('rows_lag')) and name not in (ct, self.role('cart_view'))
      def look(r):
        hit = [(k, v) for k, v in r.items() if isinstance(v, str) and v in units]
        ids = {v for k, v in hit if 'id' in str(k).lower()}
        if ids and (self.stock_state(r) is not None and not (only and ids.isdisjoint(named))
              if stock else any(isinstance(v, dict) for v in r.values())):
          fresh.update(v for _, v in hit)
      walk_dicts(obs, look)
      for sid in fresh:
        mine.setdefault('line_read', {})[sid] = self.cable.ticks
        mine.setdefault('line_seq', {})[sid] = self.absorb_n
    if name == self.role('details') and not (isinstance(obs, dict) and obs.get('error')):
      self.rec_db.read.update([v for v in args.values() if isinstance(v, str) and v in self.rec_db.ids])
    if self._late_read(name, args): self._nudge('late_n')
    if gone: notes.append('[stock] ' + self.dead_note(gone))
    moved = self._attempt('notes_remark_error', lambda: self.notes_msg(name, args))
    if moved: notes.append('[notes] ' + moved)
    if name == ct:
      book = self.barrow
      self.order_vetted, book.changed_at, before = False, self.cable.ticks, dict(book.units)
      notes.append(book.booked(args, obs))
      self._attempt('after_put_error', lambda: (self.cart_done(before), self.key_fresh(obs),
                             self.after_add(args, obs)))
      book = self.barrow
      item = book.aim(args)
      if item and book.is_add(args):
        self.refusals_seen[item] = self.refusal_n
        puts = mine.setdefault('puts', {})
        puts[item] = puts.get(item, 0) + 1
        n = book.count_arg(args)
        if book.is_add(args) and f'{item}|{self.needs.rev_no}' in self.passed and (
            n is None or n > 0) and book.held(item) > 0 and item not in self.dead_lines():
          notes.append(f"Leave {item} in the cart unless something later rules it out: an independent check tested it against every current condition under the value rules (a value the store writes in a field of its own counts as met only by that field's words).")
    if name == self.role('cart_view'):
      self.barrow.beheld(obs, self.cable.ticks)
      notes.append(self.barrow.recap())
    refused = self.refused(name, obs, resp if last else {'finished': False})
    if refused and self.last_kind == 'order': self.cart_seq = self.absorb_n
    notes.append(self.notes_for(name, obs, refused))
    if name == self.role('search'):
      notes.append(self.find_nt(args, obs))
      mine.get('named_pages', set()).discard(json.dumps(args, sort_keys=True, default=str))
      notes.append(self._attempt('merge_joined_error', lambda: self.merge_joined(obs)))
    if last and not mine.get('early_audit') and name not in (ct, tell) and name not in self.end_tools() and (
        FIRST_ORDER_BY < self.steps_rem() <= self.step_cap // 3 + 2 and not self.barrow.units) and (
        not any(k.rpartition('|')[2] == str(self.needs.rev_no) for k in self.passed)):
      self.early_audit = True
      ahead = self.judge_decline(early=True)
      if ahead: notes.append('No fit has been checked so far. ' + ahead)
    question, got = _said(args) if name == tell else None, self.buyer_lines(name, obs, refused)
    if got:
      n = self.needs
      rev, loose, cap = n.rev_no, len(n.let_go), n.spend_cap
      notes.append(n.heed(got, question))
      self.ref_grab()
      news = n.__dict__.get('allowance_news')
      if news:
        n.allowance_news = None
        self._attempt('allowance_error', lambda: self.apply_slack(news))
      if n.told_rev != mine.get('told_seen'):
        self.told_seen, self.order_vetted = n.told_rev, False
      if n.rev_no > rev and not (question is not None and len(n.let_go) <= loose and n.spend_cap == cap):
        notes.append(self._attempt('change_review_error', self.change_review))
    shown = as_text(snip(shown, SEARCH_TEXT if name == self.role('search') else RECORD_TEXT))
    if last and resp.get('finished'): shown += ' [episode over: no further call can be sent]'
    said = ' '.join(r for r in notes if r)
    return f'{shown}\n[note] {said}' if said else shown
  def _nudge(self, attrb): setattr(self, attrb, vars(self).get(attrb, 0) + 1)
  def _attempt(self, tag, labor, fallback=None):
    try: return labor()
    except Exception as mistake:
      self.log.append({'kind': tag, 'error': _werr(mistake, 300)})
      return fallback
  def _late_read(self, name, args):
    det, srch = self.role('details'), self.role('search')
    other = bool(det) and name not in (det, srch) and name in self.fetch_tools()
    ws = {v for v in args.values() if isinstance(v, str)}
    if self.steps_rem() >= self.order_due() or not (name in (det, srch) or other): return False
    if vars(self).get('covering'): return False
    held = set(self.barrow.units)
    return not (name == det and ws & held) and (
      not other or not ws & (held | {k.rpartition('|')[0] for k in self.passed}))
  def tool_calls(self, comm, sqc_none):
    raws = [r for r in comm.get('tool_calls') or [] if isinstance(r, dict)]
    return ([_qw_bid(r, sqc_none + i) for i, r in enumerate(raws, 1)], sqc_none + len(raws))
  def vet(self):
    wx = self.cable
    if wx.done: return self.log
    self.read_ask()
    self.log.append(dict(kind='plan', plan=self.needs.parsed, roles=self.role_map))
    segs, first = ['SHOPPER REQUEST:\n' + self.request], self.pvw.get('opening')
    if first is not None:
      segs.append('OPENING OBSERVATION:\n' + as_text(first))
      self.rec_db.add(first)
    brief = self.needs.rq_view(self.step_cap, self.msg_cap)
    if brief: segs.append(brief)
    segs.append(self.state_line())
    self.chat = [dict(role='system', content=TASK_RULES), dict(role='user', content='\n\n'.join(segs))]
    fine, volleys, idle, fails, sqc, missed = _TALLY.now(), 0, 0, 0, 0, 0
    while volleys < TURN_LIMIT:
      volleys += 1
      if wx.done or wx.broken or self.steps_rem() <= 0 or wx.spare() < LAST_CALL: break
      if self._on_code(): continue
      engines = [*DOER[1:], *DOER[:1]] if self.stuck >= STUCK_LIMIT else DOER
      comm = wx.query_model(self.chat, engines, 2500, self.tool_specs)
      if comm is None:
        missed += 1
        if not wx.can_serve(engines, True) or _TALLY.now() - fine > DOWN_AFTER and (
            missed > 1 or wx.downed() and len(wx.call_models) > 1):
          break
        if wx.call_fail in ('empty', 'refused'):
          fails += 1
          if fails >= 2: break
        continue
      fine, missed = _TALLY.now(), 0
      calls, sqc = self.tool_calls(comm, sqc)
      text = said if isinstance(said := comm.get('content') or '', str) else as_text(said)
      binding = {self.role('cart'), self.role('order'), self.role('decline')} - {None}
      if self.asking and any(c['function']['name'] in binding for c in calls) and self.pose_planned(urge=True):
        continue
      if not calls:
        idle += 1
        self.chat.append(dict(role='assistant', content=text or '(no tool call)'))
        self.chat.append(dict(role='user', content='That reply made no tool call. A turn without one achieves nothing; choose one of the tools given and go on.\n' + self.state_line()))
        if idle >= IDLE_LIMIT: break
        continue
      idle = 0
      self.log.append(dict(kind='model', content=text[:2000], calls=[c['function'] for c in calls]))
      self.chat.append(dict(role='assistant', content=text or None, tool_calls=calls))
      before = wx.ticks
      self.last_replies(self.relay(calls))
      self.stuck = self.stuck + 1 if wx.ticks == before else 0
    if not (wx.done or wx.broken): self.finish_now()
    self.log.append(dict(kind='end', finished=wx.done, steps=wx.ticks, cost=round(wx.spend, 5),
              seconds=round(_TALLY.now() - wx.z0, 1)))
    return self.log
  def _on_code(self):
    self.grab_more()
    n = self.needs
    if n.reply_due:
      due, n.reply_due = n.reply_due, ''
      if self.reply_back(due): return True
    return any((g is None or getattr(self, g)()) and getattr(self, m)() for m, g in _CODE_ACTS)
  def _smw_may_sweep(self): return not self.asking or bool(vars(self).get('empty_plans'))
  def _smw_is_asking(self): return bool(self.asking)
class Errand(TaskCore, AskPace, ItemCheck, ShelfLookups, RefusalFixes, BatchReads, QuestionFlow, PilotSetup, PilotMoves, SearchKeys, WalkStore, DeclineReads, RefPlan, RefResolve, ValueRows, ValueSearch, CallFilter, Relay):
  pass
_NUM_RX = re.compile('(\\d+(?:\\.\\d+)?)\\s?([a-z\\"\']{1,4})\\b')
_STALE_CART = 'Not run, and no step was spent: an earlier cart change in this reply was held back, so this one is out of date; read the reason given for that hold and send the cart changes again.'
_LAST_STEP = 'Not run, and no step was spent: only one step is left, and it is kept for the order or the decline.'
_OVER_MSG = 'Not run, as sending has ended: the step budget is used up or the store closed the episode.'
_CODE_ACTS = (('open_named', None), ('fit_search', None), ('wide_step', None), ('sweep_step', '_smw_may_sweep'), ('value_pages', None), ('pose_planned', '_smw_is_asking'), ('ref_step', None), ('reveal_reach', None), ('cap_reach', None), ('solo_reach', None), ('few_reach', None), ('walk_step', None))
def _said(args): return ' '.join([v for v in args.values() if isinstance(v, str)])
def _werr(mistake, cap): return f'{type(mistake).__name__}: {mistake}'[:cap]
def _qw_bid(raw, n):
  fx = raw.get('function') or {}
  return {'id': raw.get('id') or f'call_{n}', 'type': 'function', 'function': {'name': fx.get('name'), 'arguments': (
    a if isinstance(a := fx.get('arguments'), str) else json.dumps(a or {}))}}
def agent_main(pdata):
  job = Errand(pdata)
  try: return job.vet()
  except Exception as mistake: job.log.append({'kind': 'crash', 'error': _werr(mistake, 500)})
  try:
    if not (job.cable.done or job.cable.broken): job.finish_now()
  except Exception as mistake: job.log.append({'kind': 'crash', 'error': _werr(mistake, 500)})
  return job.log
__all__ = ['agent_main']
