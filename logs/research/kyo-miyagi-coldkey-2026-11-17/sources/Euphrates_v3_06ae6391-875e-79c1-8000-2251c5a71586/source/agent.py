from __future__ import annotations

import json
import re
from time import monotonic, sleep
from typing import Any

from src.agent.proxy_client import ProxyClient

ACTOR = ["anthropic/claude-haiku-5.5", "openai/gpt-6-luna", "anthropic/claude-haiku-5.5"]
READER = ["anthropic/claude-haiku-5.5", "anthropic/claude-sonnet-5.5", "openai/gpt-6-luna"]
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
ASK_DEFAULT = 3
FIRST_ORDER_BY = 7
FIRST_ORDER_DUE = 10
IDLE_LIMIT = 3
STUCK_LIMIT = 4
SWEEP_READS = 12
WALK_PAGES = 6
WALK_TOTAL = 8
VWALK_PAGES = 12
LADDER_PAGES = 3
LADDER_RUNGS = 3
LADDER_OPENS = 4
DEPTH_ROWS = 24
VSEARCH_MAX = 4
VALUE_PAGES = 4
VALUE_TOTAL = 8
VALUE_TOTAL_PER = 5
LABEL_TRIES = 4
LABEL_OPENS = 3
CEIL_CALLS = 4
CEIL_OPENS = 3
LINK_WORDS = {"and", "or", "the", "for", "with", "of", "a", "an", "in", "on", "to", "by", "from", "at", "as", "is", "are", "be",
              "this", "that", "it", "its", "your", "our", "you", "all", "any", "not", "no"}
ASK_AFTER_SETS = 3
ASK_GRACE_STEPS = 3
ASK_AFTER_STEPS = 5
MV_DEPTH = 3
MV_SPARE = 6
MV_PAGES = 2
MV_IDLE = 3
MV_READS = 10
MV_BUDGET = dict(search=(0.35, 3), read=(0.45, 4), cart=(0.0, 0), tool=(0.15, 2))
MV_LINE_PUTS = 3
MV_DIRECT_MAX = 14
MV_DIRECT_SPEND = 0.04
MV_STATE_CHARS = 16000
DIRECTOR = ["anthropic/claude-haiku-5.5"]
DIRECTOR_STRONG = ["anthropic/claude-sonnet-5.5", "anthropic/claude-haiku-5.5"]
STRONG_POINTS = ("PLAN", "FINAL DECLINE GATE")
MV_KINDS = (dict(search='search the store', browse="list the product's category", figures='search the measured figures', pages='read further result pages', read="open listings' records", cart='make the cart hold its picks', order='place the order', decline='decline'))
JSON_AS_TEXT = '\n\nYour answer is the JSON object itself, placed as the body of your reply.'
DIRECT_TOOL = (dict(type='function', function=dict(name='director_answer', description='What you decide now is the JSON object that the instructions call for.', parameters=dict(type='object', properties=dict(procedures=dict(type='array', items=dict(type='string')), keep_messages=dict(type='integer'), no_fit=dict(type='string'), no_fit_rule=dict(type='string'), acts=dict(type='array', items=dict(type='object')), move=dict(type='string'), query=dict(type='string'), ids=dict(type='array', items=dict(type='string')), text=dict(type='string'), tool=dict(type='string'), args=dict(type='object'), why=dict(type='string'), quotes=dict(type='array', items=dict(type='string'))), required=['move', 'keep_messages', 'why']))))

DIRECT_RULES = 'ROLE\nYou direct a shopping assistant working for a single shopper inside a simulated store. The assistant\'s own code handles the groundwork (running searches, opening listings, inspecting items, planning questions, loading the cart, placing or refusing an order). At each decision point you receive a digest of the current situation and choose what happens next. Raw store pages never reach you, just that digest.\n\nHOW THE ERRAND IS JUDGED\nThe result is graded 1 or 0, and a run earns 1 only when the ending is correct on every count:\n- each ordered item satisfies every requirement the shopper stated, every hidden need the shopper disclosed upon being asked, and every change mentioned during the errand;\n- quantities, budget and stock are all correct when the order goes in;\n- when the request calls for the cheapest (or the best) option, that one is chosen;\n- every working rule that the request imposes on the assistant is obeyed.\nWhen no store listing satisfies all requirements, end the way the THIS request directs (certain requests want the closest listing bought, others want nothing bought). A decline is the correct ending only if the request says nothing about this case, and even then only when the store has nothing able to meet the requirements.\n\nREADING THE REQUEST\'S WORKING RULES\n- Any sentence instructing the assistant on how to carry out the errand counts as a working rule, whatever its wording or dialect: a limit on messages or steps; an action tied to a particular moment (ahead of the first cart, ahead of ordering, ahead of declining, following an event); something to state, display, confirm or verify.\n- An errand can include rules never seen before. Regard each as binding, and judge from its own wording what it requires you to DO using this store\'s tools in this errand: a message, a step at a given moment, a limit, or what to purchase or skip in some circumstance. Never assume a rule\'s meaning because it resembles a rule from a previous errand.\n- Whatever the request wants stated at the finish (what was chosen, that nothing fits, which listing is closest) goes in the text of the finish call itself. The shopper reads that text, and it uses up no message. Such a rule calls for a message only if the request wants it sent ahead of the finish, or wants a reply back (a yes, an approval).\n- A rule requiring a message uses up one of the scarce messages, so from the first decision on, hold that many back from questions (keep_messages). Only a sentence that asks the assistant to tell, show or confirm something with the shopper at a particular moment qualifies. These do not: an open invitation to put questions to the shopper, a limit on messages or steps, a narrative sentence about what the shopper will do, see or check.\n- While keep_messages is greater than 0, put every sentence that requires the message into "quotes", reproduced exactly as written.\n\nWHAT EARLIER RACES SHOWED (apply a point only if THIS request or the present state makes it relevant)\n- Sign-off on the final basket, if a request demands it: only 2 of 16,857 orders lacking a message that displayed the final basket after the cart\'s last change succeeded. Successful runs held off until the cart was settled, then posted ONE brief message laying out the basket (per line: short name, quantity, price, followed by the total) to ask for approval, and placed the order afterward. The shopper never refuses: a simple acknowledgement ("Right so, grand.", "Understood, carry on.") IS the approval. If the reply asks you to say it again or make it simpler, that is no approval: write a briefer, plainer version. Once approval is given, any change to the cart (a swap, a new quantity, a line added or dropped, a re-pick following an event) calls for a new approval. Asking about a property never counts as confirming a basket.\n- Timing: send this message only at the last checkpoint (cart final, the code\'s pre-order checks cleared). If sent sooner, it is wasted once the cart changes.\n- FINAL ORDER GATE marks the spot for a rule\'s message, ahead of the order. When an active instruction calls for a message before finishing and nothing was sent since the cart last changed, send it at this point. Post a basket message only if a rule in THIS request asks for the basket, the total or approval: an unrequested one earns nothing and, with a message cap in place, uses up a question. Don\'t use the gate for further searching or reading unless the state reveals a genuine cart problem (a line sold out, over budget, a requirement clearly unmet); runs that did so frequently lacked steps or messages for the approval.\n- Agents that used up all their messages on questions were unable to follow a rule requiring one.\n- HIDDEN NEEDS take priority. The shopper has requirements the request doesn\'t mention, and an errand that never brings them out fails (scored as a missed process). On a race-5 errand with 345 orders, all orders whose questions surfaced the hidden need passed, and all the rest failed. So messages mostly serve the question engine ("ask"): keep_messages covers ONLY the messages a rule will require (approval on the basket = 1), no more than that, and equals 0 if no rule needs one.\n- Events: inside a tool result (frequently right after the first cart add) or as a rejection of the order, the shopper or the store can remove or alter a requirement, shift the budget, or say an item is sold out. The cart then has to suit the new state; the code picks again, and any rule bound to the final basket must be satisfied anew.\n- A message that runs past a few plain sentences, or holds multiple questions, gets no reply.\n- REFERENCES TO OTHER LISTINGS. A requirement can get its value from a different listing (the same as, or unlike, what a named seller\'s item of a certain kind shows, with a few identifying details), stated in the request or first in a shopper\'s reply. Each listing that such a requirement refers to has to be OPENED prior to ordering, and if the opened record points further to another listing, open that as well, until the chain stops. In race 5, 94-97% of such orders failed this check; the runs that passed searched \'<seller> <detail> <detail> <kind>\', opened the match using "read", read the named field and followed any later pointer. Do it the moment the pointer shows up, ahead of the cart: "search" using such a query, then "read" the id that matches; confirm the record fits the seller, the kind and every detail supplied; the value you find resolves that requirement.\n- When an order is refused or a decline is turned down, new information arrives (a changed budget, a dropped requirement, an item no longer available). Reassess with it: buy only when the news lets a CHECKED candidate fit; if not, finish in the same way once more. In race 6, after a declined decline was refused, declining a second time was correct over twice as often as buying.\n- The code\'s own add-to-cart step performs the item check on each pick before adding it, so a pick that wasn\'t checked may proceed ("continue"); never force through a listing whose check FAILED. Listings you specify yourself ("cart" with ids) get checked first and are rejected if they fail. Only decline after genuine searching turned up nothing that passes. Rows of the requested kind that appear only in search results, within budget and not marked sold out, count as unchecked and not as failing (a listing with a similar title tells nothing about their fields): prior to declining, "read" them, starting with the cheapest.\n- Leave yourself enough steps to wrap up: if a finish gets rejected, fixing it takes several steps, so end the session (by ordering or declining) well ahead of the step limit instead of continuing to read.\n\nMOVES (answer with exactly one)\n- "continue": have the code carry out the move it suggests. This is the default. At a FINAL gate, the suggested move IS the finish: "continue" submits that order or decline immediately. If a decline would be mistaken, pick "order" (or "cart") to block it; if an order would be mistaken, pick the move that corrects it.\n- "search" along with "query": your own keyword lookup (keep it brief: the kind of product plus one or two values that are required). If no query is given, the code performs its own lookups.\n- "read" along with "ids": pull up the records for these listings.\n- "ask": the code\'s question planner asks its next question (concerning needs the shopper never stated).\n- "message" along with "text": deliver precisely this text to the shopper (a basket awaiting approval, or anything the shopper asked to be told). Use simple words, make a single request, and keep it brief.\n- "cart" (with "ids" if you like): the code arranges for the cart to contain your picks, or these listings.\n- "order" / "decline": end the session right away (the code\'s checks before finishing still apply).\n- "tool" along with "tool" and "args": invoke a different tool from this store that none of the moves handles.\n\nTHE ANSWER\nOne JSON object and nothing else:\n{"procedures": ["each working rule of the request, short"], "acts": [{"rule": "...", "do": "...", "when": "...", "detail": "..."}], "no_fit": "order" | "decline" | "", "no_fit_rule": "...", "keep_messages": <messages to keep free from questions for those rules, counting what is still to be sent>, "move": "...", "query": "...", "ids": [...], "text": "...", "tool": "...", "args": {}, "why": "one short sentence"}\n- "keep_messages": every time.\n- "acts" (at the PLAN point): write one entry for each EVERY rule about how to work or finish that the request contains (the code reads product requirements, budget and stock itself, so leave those out). "rule": quote the request\'s sentence. "do": the action it requires of you, chosen from "message" (send the shopper something), "open" (open listings\' records), "compare" (compare listings side by side), "order_closest", "decline", "cart" (a rule about what the cart is allowed to contain), "limit" (a limit on messages, steps or anything else), "other". "when": choose from "before_first_cart", "before_order", "before_decline", "at_finish" (within the text of the finish call itself), "always". "detail": state precisely what and how many, in a handful of words. The code displays these acts to the assistant on every step and verifies the ones that are due before every finish: a rule omitted here is a rule no one carries out.\n- "no_fit" (at the PLAN point): the instruction THIS request gives for the case where nothing satisfies all requirements. "order" if it wants a listing ordered in that case (one it tells you to put forward, offer, suggest or settle for); "decline" if it wants no purchase made; "" if it is silent on the matter. A rule that merely excludes out-of-stock or back-ordered items is a stock requirement and not a no_fit rule: "" for it. "no_fit_rule": quote that sentence from the request.\n- Be brief: "procedures" only at the PLAN point and only when they change (otherwise []); "why" below 20 words; omit the fields a move has no use for.'

ACT_RULES = 'You are buying on behalf of a single shopper inside a mock store, and your provided tools are your sole means of acting. Each errand is graded 1 or 0. You get 1 only if the outcome is correct on all counts: every purchased item satisfies each requirement the shopper stated, each concealed need the shopper discloses when questioned, and each change mentioned during the conversation; the quantity and budget are correct; the lowest-priced (or top) item is chosen whenever the request calls for it; and each operating rule in the request is obeyed. If the store has nothing that satisfies everything, end the way the request specifies for that situation; if it says nothing on the matter, end by declining, done precisely in the manner the request describes.\n\nSTEPS AND TOOLS\n- Each tool call costs one step, including calls that fail or are badly formed, and you have a capped number of steps. Reasoning costs nothing.\n- You can put multiple tool calls in a single reply if none relies on the output of another (for instance, fetching details for several listings); they execute sequentially. Placing an order or declining must always be sent alone.\n- When a details tool accepts a LIST of ids, it gives back every full record at the cost of ONE step (within its item cap), so rely on it any time you look at two or more listings.\n- Naming of tools, arguments and result fields varies from store to store. Go through each tool\'s description and its JSON schema, and apply the exact names and permitted values specified. From the early results, work out which tool reveals which facts (price, stock, detailed properties).\n- Find out at the start which category labels this store assigns to listings of the kind asked for. If similar labels sit side by side, choose the one whose phrasing fits the whole request best, then do both your searching and buying within that label.\n\nSEARCHING THIS STORE\n- Stores differ in how search works; figure out this store\'s from the initial results. Rows might be sorted by relevance or by a fixed id order. A keyword query may demand all words (adding a word reduces the total) or accept any word (adding one raises the total, with listings matching more words ranked higher). Perhaps only a few rows or pages can be seen. A filter might operate in just one mode (keyword mode may disregard the category or price filter, while filter mode may disregard the words).\n- Check the stated total against the rows visible to you: if the total exceeds them, the matching or lowest-priced listing could be on a subsequent page or outside what you see. If all words are mandatory, tighten the query with more precise words (or apply category and price filters together with a smaller maximum price) until the total can be covered by the pages you can view.\n- Use brief queries (one to three words), putting the most distinctive word FIRST, since certain stores consider just the first word or two, meaning a lengthy query might in practice search only its start (test this by comparing the total and rows of a longer query against a shorter one with the same beginning; the notes record what they reveal).\n- Where all words must match, pair a needed value with the item kind ("blue chair", value first) and you get exactly the listings carrying it. Where any single word suffices, common words (the item kind, a unit) hit most of the shelf and drown out the rest: search the distinctive word or figure of a needed value by itself ("herringbone", "30m", "0.4") and in the other ways listings spell it ("30 m", "3 atm"), go through every page if the total fits on them, and add the item kind only to put the closest matches on top. Keyword searching can also hit category names and property values. The notes following each search explain what the totals so far reveal about how this store\'s search behaves.\n\nTHE SHOPPER AND HIDDEN NEEDS\n- The shopper has needs not covered in the request and discloses one only if a message inquires about that property. Whatever the FIRST thing a message asks about is answered most dependably, so put one property in each message, or at most two, placing the more probable one first: a brief, simple, warm question naming the property (one or two sample values are acceptable). Include no instructions, lists, labels, positions or rules.\n- "Don\'t mind", "no preference", "anything is fine" keep the property open. Once a preference or limit is voiced, treat it as a strict requirement from that point, with its precise wording and any numbers.\n- Whenever questions are scheduled on your behalf, they are sent after you have viewed the full details of two to three listings of the kind wanted, so make those views your first actions. Replies show up in the chat along with their meaning: treat every need that gets uncovered as a strict requirement, and run a fresh search that combines it with the values already given.\n- When you choose to pose questions yourself, begin by viewing the details of two to three listings. Next raise whatever the scenario implies (the intended person, the desired appearance, the place or manner of use). After that raise attributes the listings list which vary between the candidates. Skip anything the request has already decided, and stay within the cap on messages. You fulfil the working rules\' demands through your own messages (a confirmation, the cart contents, the total, a reply to the shopper), so the message tool is available to you as long as any message remains. During the time hidden-need questions are scheduled for you, let that schedule do the asking (it chooses attributes from the listings you viewed) and spend your messages only on the rules. Ahead of an order, the shopper hears the basket contents automatically if a rule demands it and you haven\'t mentioned it since the cart last changed.\n\nTHE REQUEST\'S WORKING RULES\n- The request can lay down working rules, such as the number of listings to view or compare, which things to verify, what to report to the shopper, the content a note must have, and how to wrap up if no listing fits. These count the same as requirements when scored, so obey every one to the letter and in full. Interpret each separately from its own wording, deciding what it demands of you using these tools on this task; do not infer it from resemblance to a rule seen in a previous task.\n- Anything the request asks you to state upon finishing (your selection, that no listing fits, the closest listing) belongs in the text of the finish call itself, which the shopper reads and which uses up no message.\n- In the status line you\'ll find the director\'s interpretation of the rules: an [acting on the request\'s rules] list (what each rule requires of you, and the timing) plus a [when nothing fits] line whenever the request tells you to order the closest listing if none fits. Follow both.\n- "Look at" or "open" N listings means N listings that meet every requirement, each one read with the details tool.\n- A question from the shopper (asking you to confirm a change, for instance) gets a reply on your behalf as long as messages are left; the change is binding regardless.\n- The status line additionally shows the INSTRUCTIONS IN FORCE, meaning the rules from the request along with each instruction the shopper or store gives afterward (confirm prior to purchase, state the total, a fresh limit on messages or steps). These may shift at any moment, so following each response, notice, or refusal, review them again along with the requirements and the cart, and pick your next action based on their present content, not on routine. If you end with a named step not yet done, it is performed on your behalf right before the finish is submitted; a message you yourself send following the final cart change also satisfies it.\n- Things the shopper only wishes for ("ideally", "would be nice", "if possible") are not requirements; they serve solely to decide between items that already satisfy all requirements, and they never override a rule asking for the cheapest.\n- If the request omits a point that only the shopper can decide (a complete lack of any price cap, or a pair of wishes that might conflict so the shopper has to rank them), put the question first and choose afterward.\n\nCHANGES ALONG THE WAY\n- Changes may be introduced by the shopper or the store: a revised budget (up or down), a requirement removed, swapped or newly added, a higher price, an item going out of stock. A single line may hold multiple changes, and these lines may appear within any tool output (an additional field containing sentences) or alongside an order attempt. Every change is binding immediately: verify the selection and the fallbacks again, and if the cheapest or best was requested, pick anew from the updated set of acceptable items.\n- An order attempt might be REFUSED, in which case the episode continues and the cart is unchanged. Go through the refusal, apply all the changes it names, remove from the cart anything that no longer qualifies (an item that sold out, for example), add the next fit you have verified, check its stock and price again, then place the order once more.\n- Stock and prices may shift with no warning, since products can sell out: verify availability and cost again immediately prior to each attempt to place an order. If an item\'s up-to-date price multiplied by its quantity exceeds the present budget, it is no longer a fit.\n\nSEVERAL PRODUCTS AND LINKED LISTINGS\n- One request can cover multiple distinct products, every one with its own quantity, requirements and hidden needs. Handle all of them: search the category of each, confirm a match for each, then place a single order covering them, with one line per product at its quantity. Some cart tools accepting a list of lines might overwrite the entire cart with that list, so include a line for every product in every such call. The rule about cheapest or best holds inside each product separately.\n- A requirement\'s value can be drawn from a different listing or page ("the same colour as the jumpsuit sold by X"): locate that listing first (search the words naming it, like its shop or brand along with its kind), open its record, pull the value from the field it names, and only then search for the requested product using that value. If a listing\'s notes direct you onward to yet another listing for that property, keep following, even into a totally different kind of product (a camera\'s voltage might be written on a pants listing): such chains are intentional, and each listing in the chain must be opened prior to ordering.\n\nTHE CART\n- If the status has a [cart tool] line, read it: it explains how this store interprets cart arguments (e.g. a numeric argument standing for the line\'s unit count rather than a position).\n- Instead of overwriting units, the cart tool may ADD them onto a line already present: adding 1 more to a held line gives 2. Do not submit again a line that is already in the cart; to adjust units, move in or out only the difference. After every cart call, a note reports what the code counts as in the cart.\n\nJUDGING A LISTING (the score hinges on this)\n- Grading checks every requirement against the listing\'s field for that property, using the value\'s wording exactly as this store records it there. Any field whose name denotes the same property is valid regardless of spelling: different separators, an added prefix, a shorter or longer name, or a common alternative name ("fit" in place of a pants fit type, "neckline" for a collar type, "dial size" for the dial size of a watch, "color family" for colour). If a field bearing the exact property name is present, it is authoritative, as is a field with that name plus a single extra word in front (for length, pants length; for material, clothing material): the requirement is satisfied if either states the value. Fields whose names vary only in separators (size class, size_class) count as a single field: satisfied if either states the value. In other cases the field under the alternative name is authoritative; if no field is named for the property at all, the options list or the title can reveal the value. Never treat the description as evidence.\n- Matching goes by the words in the store\'s own wording: differences in case, separators, plural versus singular, and spelling variants are ignored; a more specific value that includes the required one satisfies it ("80% cotton" for cotton, "forest green" for green, "rtx 3060 12gb" for rtx 3060); a single list element suffices. A synonym, abbreviation, related term or shade ("tech wood" for technical wood, "win11" for windows 11, "embroidered" for embroidery, "solid" for plain, and for off-white either ivory or cream) satisfies a value only if no listing in the category spells out the value itself in that field: if other listings give "matte" as their finish, a "satin" finish fails to satisfy matte. A value combined with another through a plus ("wool + nylon") fails to satisfy one of its components if the field also holds the other component by itself ("nylon"). Any FIELD EVIDENCE lines present are derived from this store\'s records and indicate which situation applies.\n- Another value of the same property fails to satisfy it (quartz isn\'t mechanical, rosewood isn\'t maple), and neither does a word that merely resembles it. A value that claims to span all values ("suitable for all occasions", "all seasons", "universal") specifies none of them and does not satisfy a required value; a list that additionally names the required value does satisfy it. A field stating a different value does not satisfy the required one just because the two frequently appear together (a "pullover" style isn\'t "zip-up", "slip-on" isn\'t "lace-up"); only a shade normally marketed under the required colour, or a more specific value that contains it, qualifies.\n- A field belonging to some other property cannot serve as the requirement\'s property (lining material isn\'t material, length isn\'t style, strap length isn\'t strap type): it can never substitute for a property when the listings of the category have a field of their own for it.\n- Count only what the shopper said or disclosed. Do not exclude a listing over anything the shopper never requested: an age group, gender, size, occasion, condition, quality or use inferred from the backstory or description. The backstory is just context; should it appear relevant, ask about that property rather than presuming. Questions you pose are not requirements: only the shopper\'s replies are.\n- Compare numbers as quantities once units are converted (4.5 cm equals 45 mm); boundary values are allowed; a range qualifies only if every part of it satisfies the limit. If no field is named for the property and the title serves as evidence, an exact required value is satisfied when the title displays its words and figures in the identical order, extra detail after them notwithstanding (9\' x 12\' 4" shows 9\' x 12\'; 9\' x 13\' does not).\n- Numbers that appear in a title ("55 x 94 Inch", "117 x 228 cm") fail to reveal which dimension each refers to, so a listing must never be excluded, or left unopened, because of the numbers in its title; check the fields in its details instead.\n- Stay within the exclusions ("leave out anything with X") and the closed lists ("A or B and nothing else").\n- A listing satisfies the requested product kind through its category alone: if it sits in the requested category, the grader treats it as IS that kind, even if its title or description mentions some other object (a container, a case, a refill, a set, an accessory). Never skip, reject or rank it lower because of what it appears to be; evaluate it only against the requirements, price and stock. No property needs to repeat the category name. Where the store labels its categories, choose the item from the category whose label matches the whole wording of the request (a request for "A or B" corresponds to a category "A & B", not to a narrower neighbouring one "B"). If the request wants items on hand, stock has to show it available, and its price multiplied by the quantity must not exceed the budget (hitting it exactly is acceptable). Treat a listing\'s price as being in the budget\'s unit unless its record states a different one (45 against a budget of 1000 US cents means 45 cents); do not convert either figure.\n- Prior to deciding that nothing matches: if no listing satisfies the strict wording, yet one satisfies every requirement under these rules (a more specific value, a different name for the same field, or a synonym or shade when no listing in the category states the value itself in that field), pick that listing.\n\nCHEAPEST OR BEST\n- When the request is for the cheapest, the order has to be the lowest-priced item among ALL the items that satisfy all requirements; having just one fitting item does not suffice. If the keyword search demands every word and also matches property values, use the value-and-kind search, which returns the candidates with the value: go through all of its pages and open the items starting from the lowest price. A category listing filtered by a price limit reveals prices for the entire category but not the values: use it to find which cheaper rows exist (keep lowering the limit until the total fits), then inspect the properties of those rows. For each required value, search with the item kind as a brief query and the value placed first ("blue chair", "30m watch"), reading every page if the total fits on the pages you can read (if not, add a second required value or a narrower kind word); repeat this using the kind\'s other names ("stool", "bench") and using the value alone. If a value can\'t be searched (a figure the store drops), also search the product line of every fit or near-fit you have found (the maker plus one distinctive word from its title, all pages): other sizes and colours in a line are separate listings with the same property values and may be cheaper. Order the visible candidates by price on your own and open their details starting with the cheapest; the answer is the first one whose property fields satisfy everything, once every cheaper one has been opened and rejected. Hold onto the next fitting ones as backups.\n- When the request is for the best on some measure (most, largest, longest, highest...), compare ALL the fitting items on that measure in the same manner and pick the extreme one. An item that gives no value for the measure cannot be the winner.\n\nFINISHING\n- Maintain a ranked list of verified fits that includes at least one backup, since the top choice might sell out at checkout.\n- Ahead of each order attempt: the cart must contain exactly one line for each product requested (a single line if only one product is requested), each being the selected item in its required quantity, and nothing besides; after your final cart change, re-check the selected item\'s stock and price, and confirm it is in stock and within budget.\n- If nothing fits after a thorough search, end the way the request prescribes for this situation (see the [when nothing fits] line); otherwise decline exactly as the request instructs. If the request forbids offering substitutes, the decline message must not name, describe or point to any listing: no ids, titles or prices.\n- Plan your steps: send the first order attempt while at least seven steps are still left, so that a refusal can be dealt with. When 7 steps remain, only the cart, stock-check and finishing tools are left. Make your decision once; do not keep switching the cart back and forth.\n- Each reply includes at least one tool call. Keep the text brief: the requirements for now, the verified fits with their price and stock, and the next move.'

READ_REQUEST = 'Here are a shopper\'s request and the store\'s tool list. Go through both, then answer with ONE compact JSON object (no indentation) and nothing besides. Its keys, organized by what they describe:\n\nWHAT IS BOUGHT\n"product": the product type the shopper requested, phrased as the request phrases it (if there are multiple products, link every one of them using "and").\n"items": an entry for each DIFFERENT product to be purchased, following the order of the request: {"product": the product\'s type as worded in the request, "quantity": the number of units in a single listing of it (same as for "quantity"), "requirements": its strict requirements (same as for "requirements"), "depends_on": the 1-based position of the other entry that this product has to work with, fit, suit or be compatible with according to the request, otherwise null}. A request for a single product gives a single entry. Every count that the request states for something to purchase ("2 ...", "3 x ...") begins a ONE product: when nouns are linked by "and" or "&" within a single counted phrase ("2 lamps and lamp shades", "1 x pots & pans"), they stand for one product (frequently the name of a store category) and never for two.\n"quantity": the number of units of a single listing to purchase (an integer; it is 1 unless the request explicitly wants multiple). An item that is sold as one piece, for instance a pair of shoes or a set, counts as 1. Only a simple desire to buy more than one unit raises it ("I need three of them", "order 2 packs"); the number of people who use or share it, or the number of old ones that existed, is background and keeps it at 1. A number placed beside a product ("add 2 binders", "3 coffee") gives its quantity; wording that merely describes the arrangement of cart lines ("one line each", "a single line per product") does not change that number.\n"requirements": a list of strings with one string per strict requirement for the product (for several products, include the requirements of each product, each written as "property (product): constraint"), and each "property: constraint" carrying the shopper\'s precise value, limit, exclusion or closed list. Omit price, stock, quantity, budgets and simple wishes, and do not repeat the product type or category: wording that is part of the category\'s name (a season, occasion or audience contained in it) is already covered by the category. A requirement is a characteristic of the product together with the value the request says it has to take ("the pattern has to be herringbone", "with a mechanical movement", "at least 4 drawers", "the occasion has to be party"). The surrounding story is never one: who it is meant for, an event it is meant for ("to wear at his wedding", "for the staff party"), the place, time or manner of use, the number of people sharing it, or the fact that it replaces an old one ("a new watch" asks for a replacement, not for a condition). An occasion, use or audience counts as a requirement only if the request turns it into a property of the item itself ("the occasion has to be business", "a blazer made for business"). A requirement\'s value consists of words or figures that the request writes as the item\'s own value; a quality, reason or wish that comes from the story (natural and breathable, slim enough for a case, works with what I already have, covers the whole drop) is a hint and not a requirement.\n"soft_wants": things the shopper would enjoy having but does not strictly need (ideally, preferably, would be nice, if possible). Do not put these in "requirements".\n"conditions": an entry for each requirement that applies only if another product being purchased turns out to be a particular way ("should A\'s P come out as X or Y, then B\'s Q must be Z", "if A has P X, B must be Z"): {"product": B as the request words it, "requirement": "Q (B): Z", "if_product": A as the request words it, "if_property": P, "if_values": [X, Y]}. Such a requirement belongs here and in no other place (not in "requirements", not in an item\'s requirements): it is binding only when the A that gets bought has one of those values. Leave empty if there are none.\n\nPRICE, PRIORITY AND LIMITS\n"price_cap": the maximum total price, given as the number the request writes, in the request\'s own unit (never converted: "3050 US cents" gives 3050), or null.\n"currency": the unit of price used in the request, or null.\n"in_stock_only": true if the request only wants items that are on hand or in stock.\n"priority": "cheapest" only if the request clearly asks for the cheapest or lowest-priced item that fits; "best" only if it clearly asks for the best according to some measure; otherwise "none". A remark that money is tight or that every penny matters does not make a cheapest request; it merely explains the cap.\n"priority_detail": for "best", the measure; otherwise "".\n"message_budget": the number of messages the shopper lets you send, as an integer, or null if this is not stated.\n"step_budget": the number of moves or steps the request permits, given as an integer, or null if it doesn\'t state one.\n\nTHE REQUEST\'S WORKING RULES\n"rules": all the rules the request lays down about how the assistant should operate (how many listings to examine, what to verify or report, how to wrap up if nothing matches, what is off limits, what must be included), copied verbatim. Anything the shopper plans to do personally is story and not a rule. If you can\'t tell whether a sentence is a rule, place it here.\n"instructions": the same rules once more, as one object per rule: {"text": the rule worded as the request words it, 20 words at most, "before_finish": true if it has to happen before or when finishing (prior to an order or a decline: getting the shopper\'s confirmation, stating the basket or total, a parting comment, a final check), otherwise false, "needs_message": true if carrying it out means messaging the shopper ahead of the finish (something to say or ask while the errand is still going), otherwise false - whatever gets said at the finish itself (the choice made, that nothing matches, the closest listing) belongs in the text of the finish call, so use false for it, "before_change": true if it tells the assistant to consult the shopper prior to altering, swapping or removing anything already in the cart, otherwise false}. Count a rule that merely specifies how to finish or what to say at the end. Leave empty if there are no rules.\n"decline_no_names": true if the request\'s decline rule bans offering, naming or suggesting substitutes or alternatives; false if it tells you to name something; null if the request has no such rule.\n"ask_first": whatever only the shopper can decide and must be asked prior to choosing (not any approval, yes or confirmation the request wants for the basket or order: that is sent at the end, when the basket is settled, so it goes in "rules"): for instance the budget if no cap is given, which of two wishes counts more when the request says both might be impossible (then place both wishes in soft_wants, not in requirements), or anything the request says to run past the shopper. Leave empty if nothing applies.\n\nTHE STORY\n"hints": story details that hint at a product property without stating its value (what the item is for or used with, where, by whom, what it has to fit, match or withstand), each "property (product): the detail in a few words" ("property: detail" for a single product), with the property named the way listings would probably name it. If a detail could relate to several properties, put the likeliest first and the next after a slash ("colour / style (chair): must suit a room with one painted wall"): what the item has to match, suit or go well with (a room, a wall, an outfit, things it\'s used alongside) and its look suggest colour first, then pattern or style; a pledge to measure or fetch the tape, or a space the item has to fill, suggests a dimension the request leaves unstated; a previous one that broke, wore out, seemed cheap or quickly failed suggests its material. Skip a detail that merely justifies a requirement already given (why the material or size is as stated). Weather, a job, a schedule, travel, or a place where the item isn\'t worn or used counts as background, never a hint; so does a purpose shared by every listing of that kind (storage for a dresser, warmth for a coat). Leave empty if none.\nEach sentence in the request ends up in one place: a requirement, the priority, a soft want, a rule, a budget, a hint, or simple story.\n\nVALUES TAKEN FROM OTHER LISTINGS\n"references": requirements whose value the request supplies only by referring to ANOTHER listing or page ("the same colour as the jumpsuit sold by X", "within 5 of the figure on the page for Y", "a colour other than the one on Z\'s page"), each "property (product): as shown on <how the request names that listing>" ("property (product): different from the one shown on <how the request names that listing>" when the value must differ); also put each in "requirements" written in that form. Leave empty if none.\n"reference_targets": for every entry in "references", the listing it refers to: {"property": its property, "product": its product, "seller": the shop or brand name the request gives for that listing, precisely as written, "kind": what kind of product that listing is, "details": [every other value the request mentions for that listing, brief and in the request\'s wording ("blue", "4gb", "net weight 167.262186 g")], "relation": "same", "differ" (the value has to NOT be that listing\'s: set apart from, different from, other than), "pair" (the desired value is the partner named in that listing\'s own text for its value: its counterpart, what goes with it, what it is compatible with or plays nicely with), "within", "at most" or "at least" (how the value has to relate to that listing\'s), "tolerance": the number for "within", otherwise null, "unit": its unit, otherwise ""}. Leave empty if none.\n\nTHE STORE\'S TOOLS\n"roles": an object that assigns a tool to each role, using the exact tool names from the list, or null if no tool suits: "search", "details", "view", "message", "cart", "cart_view", "order", "decline". "message" messages the shopper; "cart" puts items in or removes them from the cart; "order" purchases or checks out; "decline" closes the errand without a purchase.\n'

HEAR = 'A shopping assistant has to work out what the shopper\'s latest lines mean. Answer with ONE JSON object and nothing besides it, using these keys:\n"meaning": a single brief sentence stating what the shopper means.\n"needs": fresh hard conditions for the product, every "property: constraint" carrying the precise value, cap, exclusion or fixed list stated. When a comparison phrase comes first (falls short of, nothing less than, at least), a value expressed in words instead of a number counts as the value itself, copied the way the shopper phrased it: "falls short of not available" yields "property: not available" ("not available" and "none" may be a field\'s own values).\n"no_preference": attributes the shopper states they are indifferent to.\n"dropped": past requirements that the shopper has stopped wanting.\n"new_budget": the updated maximum overall price, given as a number in the unit used by the request\'s budget (use the figure the line gives when it is in that unit: "13075 cents" becomes 13075 for a budget in cents), or null.\n"unavailable": true if the shopper reports that an item cannot be purchased any longer (sold out, gone, out of stock, disabled button), otherwise false.\n"meta_only": true if the line merely asks the assistant to put fewer questions at once (a single question at a time, fewer items together, the first one now and the others afterward) and states no value: nothing is answered yet, so leave needs, no_preference and dropped empty for it, and its figurative wording names no value; otherwise false. A line that only says the shopper missed or misunderstood the question, or wants it repeated more briefly or simply, is not meta_only.\n"requirements_now": the complete set of hard requirements on the product as of now, each "property: constraint": begin with REQUIREMENTS SO FAR, retain every one unless the shopper removed or swapped it, hold only the latest value of a swapped one, and include each new need that the shopper\'s lines state. Do not add any requirement taken from the request\'s story that REQUIREMENTS SO FAR lacks: only the shopper\'s lines can add requirements. Leave out price, stock, quantity and product category.\n"reply": if the shopper poses a question to the assistant or wants something confirmed, a brief, simple reply to return that states what changed in essence (for instance, confirming the new value); otherwise "".\n"instructions": standing rules the lines establish for the assistant\'s behavior going forward (check with me prior to buying, give me the total or the basket, do or avoid some action, how to wrap up or what to say at the end), one object per rule: {"text": the rule worded as the lines word it, "before_finish": true if it has to happen before or at the finish (an order or a decline), otherwise false, "needs_message": true if carrying it out requires messaging the shopper ahead of the finish (something to say or ask while the errand is still open), otherwise false - whatever gets said at the finish itself (what was picked, that nothing matches, which listing is nearest) belongs in the finish call\'s own text, so false for that, "before_change": true if it asks the assistant to confirm with the shopper before altering, swapping or removing anything already in the cart, otherwise false}; [] if the lines give none. A need, a budget or a dropped requirement does not count as an instruction.\n"allowance": if the lines impose a fresh cap on the assistant\'s messages, questions, steps, moves or calls: {"messages_total": the new overall count of messages permitted, or null, "messages_left": the number of messages that can still be sent from this point, or null, "steps_total": the new overall count of steps permitted, or null, "steps_left": the number of steps remaining from this point, or null}; null if no such cap is given.\n"references": for every need in "needs" whose value the shopper supplies only by referring to a different listing or page, the listing referred to: {"property": its property, "seller": the shop or brand name the line gives for that listing, precisely as written, "kind": what type of product that listing is, "details": [every other value the line mentions for that listing, brief and in the line\'s words], "relation": "same", "differ" (the value has to NOT be that listing\'s: set apart from, different from, other than), "pair" (the desired value is the partner of that listing, as its own text states: its counterpart, what pairs with it, what it works with or is compatible with), "within", "at most" or "at least", "tolerance": the number for "within", otherwise null, "unit": its unit, otherwise ""}; [] if there are none.\n\nHOW TO READ THE LINES\n- A single line may hold multiple changes: report every one.\n- A requirement about a different property, even a closely related one (seat height next to total height, strap width next to width), counts as an additional need and overrides nothing. Only a line that explicitly addresses that very same property can remove or swap out an earlier requirement. When a requirement is swapped, the old value is dropped and the new value becomes a need.\n- If the shopper indicates which of two wishes carries more weight, treat the favored one as a need. Any budget they state becomes new_budget and supersedes the earlier figure, be it larger or smaller, even if the shopper describes it as good news. A simple wish ("ideally", "would be nice") does not count as a need.\n- Interpret the shopper generously and at face value: homespun, metaphorical or indirect phrasing that names a value signals a firm desire for it. Exclude a value only if the shopper plainly states it should be avoided.\n- Words that merely emphasize how firmly the shopper holds a value (locked to, strictly, nailed to, nothing short of, only, pinned to, come out as, sticking with, nothing but, fixed as, set on, stays, kept or held tight to, holding out for) are no part of it: write only the value words ("material: jute", "color: white"), never "<value> only". In "keep <property> <firm word> to <words>" the value consists solely of what follows "to" (the width kept tight to narrow yields "width: narrow"). Only when a shopper also explicitly rules out mixes or other values does an exclusion get added.\n- A figure paired with a comparison (under, over, at least, at most, up to) counts as a limit and keeps its unit.\n- If the shopper merely says they\'re fine either way, have no preference, or that anything will do, list the property in question under no_preference (if the message covered two properties, just the FIRST one).\n- A reply that does nothing except signal the shopper was confused by the question ("that was rather dense", "you\'ve lost me", "could you put it another way") is how this shopper expresses having no preference, provided the question named the product that owns the property; add the property in question to no_preference as well. If multiple products are requested and the question left unclear which one it meant, leave the property open and add nothing to no_preference.\n- When the shopper asks for multiple products, record each need as "property (product): constraint", with the product drawn from the line (a line concerning "toe style of the women\'s boots" yields "toe style (women\'s boots): ...").\n- Nothing the assistant asks counts as a requirement; only the shopper\'s words do, and example values mentioned inside a question do not turn into needs. Any "don\'t mind" answer releases just the property the assistant\'s message inquired about; it never removes or alters a previously stated need, and if the message posed no question (it merely informed), that kind of answer has no effect.\n- When a need is expressed by referring to a different listing, its value stays the line\'s own referring words ("finish: unlike the one shown on Y\'s kettle page"). Values the line mentions for that other listing characterize it rather than the product, so they yield no needs, no drops and no freed properties.'

READ_RULES = '- Check every requirement against the listing\'s field for that property, using the value\'s wording as this store records it in that field. A field naming the same property qualifies regardless of how it is spelled (different separators, a prefix, a shorter or longer name, or a common alternative name like "fit" for pants fit type, "neckline" for collar type, "color family" for colour). If a field bearing the property\'s exact name exists, it is authoritative, as is a field with that name plus one extra leading word (for length, pants length; for material, clothing material); the requirement is satisfied if either one contains the value. Two fields whose names vary only in separators (size class, size_class) count as a single field, satisfied when either contains the value. In any other case the differently named field is authoritative; the options list or the title may show the value only if no field names the property. Never treat the description as evidence.\n- Match values by the words this store uses: differences in case, separators, singular versus plural and spelling variants do not matter; a more specific value that contains the required one satisfies it ("80% cotton" for cotton, "forest green" for green, "rtx 3060 12gb" for rtx 3060); one entry in a list suffices. A synonym, an abbreviation, a related word or a shade ("tech wood" for technical wood, "win11" for windows 11, "embroidered" for embroidery, "solid" for plain, ivory or cream for off-white) satisfies a value only if no listing in the category writes that value itself in that field: if other listings record "matte" as their finish, a "satin" finish does not satisfy matte. A value joined to another with a plus ("wool + nylon") does not satisfy one of its parts if the field also holds the other part by itself ("nylon"). Where FIELD EVIDENCE lines are provided, they are computed from this store\'s records and indicate which case applies.\n- A value that differs from the required one fails it (quartz is not mechanical, rosewood is not maple), and so does a word that merely resembles it. A value that claims to span all values of the property ("suitable for all occasions", "all seasons", "universal") names no particular one and so fails a required value; but a list that also includes the required value passes. A field that names some other value does not satisfy the required one just because the two frequently come together or one tends to suggest the other (a "pullover" style is not "zip-up", "slip-on" is not "lace-up"); only a shade normally sold under the required colour name, or a narrower value that contains it, passes.\n- A field belonging to a different property is not evidence (a lining material is not the material, a length is not a style, a strap length is not a strap type): it can never replace a property when the category\'s listings have a field of their own for it.\n- Compare numbers as quantities once units are converted (4.5 cm equals 45 mm); the boundary value is allowed; a range passes only if every part of it satisfies the limit. If no field names the property and the title is the only evidence, an exact required value is satisfied when the title presents its words and figures in that same order, even if extra detail follows (9\' x 12\' 4" shows 9\' x 12\'; 9\' x 13\' does not).\n- Treat a listing\'s price as being in the unit of the request\'s budget, unless its record states a different currency or unit (a price of 45 against a budget of 1000 US cents means 45 cents, far under it); do not convert the price or the budget to a different scale.\n- Only the requirements that are listed matter: do not reject a listing over an age group, gender, size, occasion, condition, season, quality or use that the requirements leave unnamed.\n- The category label by itself settles whether the requested kind is satisfied: a listing filed under the requested category counts as that kind even if its title names a different object (a container, a case, a refill, a set, an accessory); evaluate it only against the requirements, price and stock.'

CHECK_ITEM = 'This is the last verification before a shopping assistant commits to an item. The task earns a score of 1 only if the item satisfies all of the requirements listed below. Assess it ONLY using the records provided, under these rules:\n' + READ_RULES + '\n- A word that forms part of the category\'s name is already satisfied by the category itself.\n- Exclusions and closed lists hold.\n- The listing must be of the kind that was asked for. If listings have a category label, the label must correspond to the wording of the whole request, not to a more specific or adjacent category.\n- The total of price multiplied by quantity has to stay inside the budget (hitting it exactly is acceptable). Any record that marks it as out of stock means it fails.\n- A simple preference ("ideally", "would be nice") flagged as optional does not count as a requirement.\nIf records conflict (over price or stock, for instance), go with the most recent one.\nRespond with ONE JSON object only, no other text: {"item": the item\'s id, "fits": true or false, "problems": [brief strings, one per requirement that is unmet or not evidenced]}. Use fits=false solely when a listed requirement is clearly violated by the item\'s fields according to these rules (a FIELD EVIDENCE line stating the item does NOT satisfy a requirement counts as such a violation). If a requirement is never touched on anywhere in the item\'s records (it has no field, and the options list, title and notes are all silent on it), treat it as unknown rather than failed: list it under "problems" as "unknown: <requirement>" and leave fits=true. Do not reject an item because its stock or price record is absent.'

RANK_CHECK = 'The shopping assistant is on the verge of placing an order for an item in response to a request seeking the PRIORITY. What follows are the requirements, the item picked, and the listing records encountered. Apply the reading rules given here (price multiplied by quantity must stay within budget; an item not marked out of stock counts as available) and identify the listings that could satisfy all requirements AND outdo the picked item on the priority (for the cheapest, a lower price; for the best, a superior value on the measure). Exclude any listing whose records show it missing a requirement, being out of stock, or exceeding the budget. Numbers in a ROW title (for instance 55 x 94) fail to indicate which measurement each one refers to, so they must never be grounds for dismissing a ROW.\nREADING RULES:\n' + READ_RULES + '\nGive ONE JSON object only, with no other text: {"better": [ids, best first, at most 4], "reason": "one sentence"}.'

ASK_PLAN = 'You plan the questions that a shopping assistant puts to a shopper. The shopper has hidden needs, each one a value of a single product property, revealed only when a message asks about exactly that property. Whatever a message asks first receives the most dependable answer; questions on any other topic receive a courteous "don\'t mind". Choose up to QUESTION_COUNT distinct properties, with the most promising ahead:\n0. To begin, take anything marked below as MUST BE LEARNED FROM THE SHOPPER that has not yet been asked, and ask it directly. For the budget, ask what the most is that they would like to spend. For a choice between two wishes, name both values along with their properties, for example "If one listing can\'t have both, would you rather have the <value> <property> or the <value> <property>?".\n1. Next comes the property that the STORY hints at. A sentence in the request that states no requirement is present for a purpose: it builds the shopper\'s situation so that one product property settles the choice, and in the recent race\'s errands with a hidden need the story indicated that property in most cases. Read the request as a situation and link each such sentence to the listing property it concerns (STORY HINTS below offer a first reading; rely on the request\'s own sentences over them).\n   - Whatever the user will touch or press, look at closely or in photos, listen to, measure or fit against, read on the label, or how it was manufactured, indicates that part or aspect of the item: tags, labels or fine print lead to the material, fabric or ingredients; photos and appearance lead to the pattern, colour or the visible part; a body part or a space it has to fit leads to that part\'s shape or size; a purchase made for another person, or for a person hardly known, leads to who it is made for (gender, age group); how it should feel against the hand, skin or foot leads first to the material or fabric; where, when or for which activity it is worn or used (a gym class, the office, outdoors) leads to its style or type, the property whose values name such uses; something fastened or adjusted over and over leads to its closure or mechanism; whatever does the work inside it leads to its core component.\n   - Weather, a job, a schedule, or a place where the item is not worn or used count as background only.\n   - A promise to measure, check the size or fetch the tape, or a space, wall or gap that the item has to fill, indicates a dimension the request has not yet given (width, height, length or depth, whichever is not already a requirement). How something appears or looks to other people indicates its pattern or colour ahead of a part\'s style. A previous one that broke, wore out, felt cheap or quit working soon indicates its material (in the recent race\'s stories of that kind the hidden need was the material 5 times in 8).\n   - Start by asking about whichever property the story\'s details hint at, using the listings\' own label for it. If the story hints at nothing, open with the product\'s main specifications: keys that most listings below have, and whose values VARY across them. The intended wearer or user counts as one candidate among the rest: ask it if the story is aimed at another person, or if there\'s no other hint. After that, favor properties that say what the product is like (pattern, material, shape, type or style variant, closure, length, function) over logistics (warranty, condition, pack size, variant or size lists, years).\nCHOOSING AMONG PROPERTIES\n- Favor a property whose values DIFFER across the listings that already satisfy the stated requirements (the hidden need is what separates them). Leave a general \'style\' for the end: in the record it exposed a need the least often (roughly 5% of the time, versus 8% overall and 15% for pattern).\n- Do not choose a property that the request has already fixed (its requirements, the price, stock, quantity; only REQUIREMENTS SO FAR fix a property - the values that identify a listing the request refers to, "the page of <seller>\'s <kind> with <property> <value>", describe that listing rather than this product, and fix nothing). Do not repeat a question already asked, do not use two questions on near-duplicates (two kinds of style, two spellings of the same property), and do not ask about brand, maker, product or model name, price, stock or delivery.\n- Only ask about a property that shows up as a key in the listings\' details below, using the listings\' own label for it. If the story refers to a part of the item whose word is contained in a key\'s name (the story\'s strap and a "strap width" key), ask that key under its complete name. A property listed under STORY HINTS that matches no key below can be asked by its plain name, except when the hint is flagged [no key] (enough of its product\'s listings were read lacking that field): in that case ask the key below whose values vary the way its detail describes, or skip it.\nHOW A QUESTION IS WORDED\n- Every step-1 question is a single short, plain sentence of AT MOST 8 WORDS that asks what <property> they want for the <item kind> ("What finish do you need for the door handles?" - in the record this form got rejected 42-44% of the time and exposed the most, whereas "Which <property> would you like for the <item>?" got rejected 52-56% of the time). Refer to the property by the listings\' own label and to the item by its shortest plain name, with NO example values: questions of 8 words or fewer were rejected as unclear around 31% of the time versus 42-50% for longer ones, while exposing needs equally often, and example values made no difference.\n- Skip any lead-in, status report, instructions or lists, and never use a catch-all key such as options, variants, a description, a title or a name.\n- Every question concerns ONE property of ONE product (setting aside the step-0 choice between two wishes): never put two properties in a question and never mention properties of two products, because a reply exposes one need at most and a question naming two subjects might get no answer whatsoever. In "second", offer the next most promising property of the SAME product (not a different name for the first), or leave it blank: the assistant may append it to its last message.\nSEVERAL PRODUCTS\n- If the request is for SEVERAL products, the hidden needs could be on any of them: distribute the questions so each product receives one before any receives a second (a product with no property remaining excepted), draw each product\'s properties from the list for its own category below, phrase the property as "property (product)", and mention the product in the question ("What toe style do you need for the boots?").\nAnswer with ONE JSON object and nothing besides it, putting the analysis first: {"analysis": "two or three sentences: the shopper\'s situation, and which story sentence hints at which listing property", "questions": [{"property": "...", "second": "... or empty", "question": "..."}, ...]}.'

NEXT_ONE = '\nQuestions are sent singly in the order you give, and each reply is read before the following one is sent. REMAINING question(s) remain in total and every one should be put to use, so list REMAINING properties: the single most promising property at the top, then the rest in order, up to REMAINING in total. They are sent in that sequence, one by one, and the plan is redone only once a reply exposes a need; once a need is exposed, the next question goes to a different property and a different story detail.'

BEFORE_DECLINE = 'The shopping assistant is preparing to inform the shopper that the store has no suitable match. Compare the listing records that follow with the requirements, using these reading rules (price must fall within the budget, and an item not marked out of stock counts as available):\n' + READ_RULES + """
Some listings below are known only from search rows (no details read, so their property fields are unknown). Such a row in the requested category (whatever object its title names) whose title, price and stock do not show it failing a requirement has not been checked: figures in a title, such as 55 x 94, do not say which dimension each one is, so they never show a row failing, and another listing's record never stands for it (listings with the same or a similar title, from another seller or under another id, often differ in their fields). List such rows under "unopened", likeliest first (a title naming a required value, or the requested kind with nothing against it).
Reply with ONE JSON object and nothing else: {"possible_fits": [ids of listings that seem to meet every requirement, best first, at most 5], "unopened": [ids of rows-only listings to open before declining, at most 6], "reason": "one sentence"}."""

REF_HOP = 'A shopper\'s request gets the PROPERTY of the product it wants from a different listing in the store, which is described as: DESCRIPTION. The records below are listings that could be that listing. Answer with ONE JSON object and nothing besides it, containing:\n"target": the id of the record that IS the described listing: from that seller, of that kind, and displaying every named detail, interpreted by meaning (a figure in a different unit counts if it is the same amount); null if no record fits.\n"missing": whichever details the target leaves out ([] if it shows every one).\n"go_to": if the target\'s description directs the reader to a DIFFERENT listing regarding the PROPERTY (phrases like borrow it from, use, consult, check or follow another listing), give that listing as {"seller": shop or brand name exactly as it appears, "kind": the product type, "details": [every value the text mentions for it, brief and verbatim: a colour, a size including unit, a material, plus the season, use or occasion it is meant for ("for autumn")]}; otherwise null.\n"value": if go_to is absent, the value the target itself has for the PROPERTY, worded the way its record words it.\n"goes_with": any values that the target\'s text describes as matching, paired with or equivalent to that value ([] if there are none).\nDon\'t treat as go_to a listing mentioned solely as something the target differs from ("not the green one from X"), or as a variant, size or colour available elsewhere; the same goes for one mentioned regarding a different property. Yet when a text states that the PROPERTY is derived from another listing rather than this one ("..., not from this item"), that other listing is the go_to: the "not" concerns the target itself. Whatever its text claims, a record lacking the described details isn\'t the target.'

CHANGE_REVIEW = "A shopping errand's conditions have just shifted: a store notice, something the shopper said, or a rejected checkout revealed a new budget, a requirement that was removed or added, or an item that is no longer available. What follows is the request, the change, the requirements that now apply, the priority, the cart, and a card for each listing seen up to now (a DETAILS card holds the complete record). Reassess the OPENED listings against the conditions as they currently are NOW, applying these reading rules:\n" + READ_RULES + '\nRespond with ONE JSON object only, no other text: {"fits": [ids of opened listings that satisfy all requirements currently active, fit the budget and are not shown as out of stock, best first (lowest price first if the cheapest is wanted)], "pick": the id to purchase now, or null if nothing fits, "cart_still_right": true if the cart\'s line is one of the fitting listings and is the correct choice (a basket that still fits stays), false if the cart has a line that no longer fits or if a cheapest or best rule points to a better choice, null if the cart is empty, "reason": "one sentence"}.'

FINISH_CHECK = 'This is the last checkpoint before a shopping assistant closes out an errand by placing an order or declining. Steps the assistant has to perform, some of them prior to or at the close, can be laid down by the request, later shopper messages and store notices: asking the shopper for confirmation beforehand, reporting the basket or total, a parting comment, a final inspection of something. Go through the active instructions, the most recent shopper messages and notices, and the actions taken since the last cart change, and judge whether any stated step remains undone before THIS finish.\n- A basket-related step carried out earlier than the most recent cart change is not counted as completed.\n- (If an order is coming, a message is being left, and nothing has reached the shopper since the latest cart change, the code automatically informs the shopper of the basket and its total; include that step only if a stated step requires something additional or different, like answering a question or waiting for a yes.)\n- Make nothing up: include only steps that the request, shopper or store really stated, and only if a tool on the list is able to perform it.\n- Anything to be communicated at the finish (the choice made, that nothing matches, the nearest listing) belongs in the text of the finish call itself, and is never a pending step.\nAnswer with ONE JSON object and nothing besides it: {"pending": [an entry for each undone step, ordered as they should be run: {"why": a brief phrase for the stated step, "tool": the precise name of the tool that performs it, "args": its arguments exactly in the form that tool\'s schema requires; for a message to the shopper, spell it out fully in one or two short plain sentences holding the facts the step requires (for the basket: every cart line with its id, count and price, followed by the total)}], "ready": true if nothing is pending, otherwise false}. Maximum two entries; leave out any step that no listed tool can perform.'

DECLINE_WORDS = r"refus|declin|cancel|abandon|reject|give.?up|abstain|withdraw|no.?(?:purchase|order|buy)|nothing|without"
ORDER_WORDS = r"commit|buy|order|purchas|check.?out|pay|confirm|complete|place|accept|proceed"
CONFUSED_REPLY = (r"confus|can'?t tell what|cannot tell what|didn'?t follow|did not follow|lost me|lost you|fresh try|another way"
                  r"|more simply|simpler|break it down|repeat that|say (?:that|it) again|not sure what you|rephrase|too dense|muddled"
                  r"|didn'?t catch|did not catch|one ear|sailed past|not following|easier terms|explain(?:ing)? your last")
SORRY = "Sorry, but no product in the store satisfies all of your requirements, which is why I won't place an order."
ID_LIKE = r"[A-Za-z0-9][A-Za-z0-9_\-:.]{3,63}"
CODE_LIKE = r"[A-Za-z0-9_\-]+"

def as_text(piece, ceiling = VIEW_CHARS):
    try:
        text_body = json.dumps(piece, ensure_ascii=False, separators=(",", ":"))
    except (TypeError, ValueError):
        text_body = str(piece)
    return text_body[:ceiling] + '...(cut)' if not len(text_body) <= ceiling else text_body

def clipped(piece, chars, lvl = 0):
    if lvl > 8:
        return None
    if isinstance(piece, (str,)):
        return piece[:chars] + '...' if not len(piece) <= chars else piece
    if isinstance(piece, (dict,)):
        return {fkey: clipped(val, chars, lvl + 1) for fkey, val in piece.items()}
    if isinstance(piece, (list,)):
        return [clipped(val, chars, lvl + 1) for val in piece]
    return piece

def json_object(text_body):
    if not isinstance(text_body, (str,)):
        return None
    envelope = text_body.strip(None)
    if envelope.startswith("```"):
        envelope = re.sub('^```[A-Za-z]*', '', envelope).strip(None)
        if envelope.endswith("```"):
            envelope = envelope[:-3].strip(None)
    for candidate in (envelope, '' if not '{' in envelope else envelope[envelope.find('{'):envelope.rfind('}') + 1]):
        if not candidate:
            continue
        try:
            piece = json.loads(candidate)
        except (ValueError,):
            continue
        if isinstance(piece, (dict,)):
            return piece
    return None

def is_number(piece):
    return isinstance(piece, (int, float)) and not isinstance(piece, (bool,))

def long_strings(piece, collected, lvl = 0):
    if lvl > 4:
        return
    if isinstance(piece, (str,)):
        if 4 <= len(piece.split()):
            collected.append(piece)
    elif isinstance(piece, (list,)):
        for thing in piece[:10]:
            long_strings(thing, collected, lvl + 1)

def long_strings_all(d, out):
    for val in d.values():
        if isinstance(val, (str,)) and val.strip(None):
            out.append(val.strip(None))
        elif isinstance(val, (dict,)):
            out.extend(str(x).strip(None) for x in val.values() if isinstance(x, (str,)) and x.strip(None))

def walk_dicts(piece, visit, lvl = 0, ceiling = 6):
    if ceiling < lvl:
        return
    if isinstance(piece, (dict,)):
        visit(piece)
        for thing in piece.values():
            if isinstance(thing, (dict, list)):
                walk_dicts(thing, visit, lvl + 1, ceiling)
    elif isinstance(piece, (list,)):
        for thing in piece:
            walk_dicts(thing, visit, lvl + 1, ceiling)

def junk_key(key):
    part = re.split(r"\s*\(", str(key))[0]
    words = " ".join(re.findall(r"[a-z0-9]+", re.sub(r"([a-z])([A-Z])", r"\1 \2", part).lower()))
    return bool(re.search(r"\b(?:brand|shop|seller|store|vendor|manufacturer|category|categories|department|title|headline|"
                          r"heading|listing|description|overview|price|cost|amount|availability|sku|url)\b", words)
                or re.fullmatch(r"(?:name|notes?|stock|in stock|on hand|images?|links?|make|maker|shelf|ids?|id)", words))

def prop_key(name):
    part = re.split(r"\s*\(", str(name))[0]
    words = re.findall(r"[a-z]+", re.sub(r"([a-z])([A-Z])", r"\1 \2", part).lower())
    return "".join(w if not (3 < len(w) and w.endswith('s') and (not w.endswith('ss'))) else w[:-1] for w in words)

def edits_within(a, b, most):
    if abs(len(a) - len(b)) > most:
        return False
    prev = list(range(0, len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (cb != ca)))
        if min(cur) > most:
            return False
        prev = cur
    return prev[-1] <= most

def same_prop(a, b):
    x, y = prop_key(a), prop_key(b)
    if not x or not y:
        return False
    if y == x:
        return True
    short = min(len(x), len(y))
    if short >= 4 and x[:3] == y[:3] and edits_within(x, y, 2 if not short < 6 else 1):
        return True
    return short >= 5 and (x.startswith(y) or y.startswith(x))

def prop_words(name):
    part = re.split(r"\s*\(", str(name))[0]
    words = re.findall(r"[a-z]+", re.sub(r"([a-z])([A-Z])", r"\1 \2", part).lower())
    return [w if not (3 < len(w) and w.endswith('s') and (not w.endswith('ss'))) else w[:-1] for w in words]

def spelled_alike(lo, hi):
    gap = len(hi) - len(lo)
    if gap == 0:
        diff = [i for i in range(0, len(lo)) if lo[i] != hi[i]]
        return 2 == len(diff) and diff[1] == diff[0] + 1 and lo[diff[0]] == hi[diff[1]] and lo[diff[1]] == hi[diff[0]]
    if not gap in (1, 2):
        return False

    def soft(word, i):
        return word[i] in "aeiouy" or (i > 0 and word[i - 1] == word[i]) or (i + 1 < len(word) and word[i + 1] == word[i])
    for i in range(0, len(hi)):
        if not soft(hi, i):
            continue
        one = hi[:i] + hi[i + 1:]
        if gap == 1:
            if lo == one:
                return True
            continue
        for j in range(0, len(one)):
            if soft(one, j) and one[:j] + one[j + 1:] == lo:
                return True
    return False

def strict_prop(a, b):
    x, y = prop_key(a), prop_key(b)
    if not x or not y:
        return False
    if y == x:
        return True
    lo, hi = sorted((x, y), key=len)
    if 4 <= len(lo) and lo[:3] == hi[:3] and spelled_alike(lo, hi):
        return True
    wa, wb = prop_words(a), prop_words(b)
    short, longer = sorted((wa, wb), key=len)
    return len(longer) == len(short) + 1 and longer[:len(short)] == short and 3 >= len(longer[-1]) and 5 <= len(''.join(short))

BOUND_WORDS = r"\b(?:no|not)\s+(?:more|less|fewer|over|under|above|below|beyond|exceeding)\b"
CH_BOUND = r"\b(?:at least|at most|no more than|no less than|more than|less than|up to|or more|or less|or larger|or smaller|or higher|" \
           r"or lower|minimum|maximum|between|within|under|over|above|below|than)\b"
CJ_COMPARE = (r"\b(?:falls?\s+short\s+of|short\s+of|nothing\s+(?:less|other)\s+than|nothing\s+short\s+of|nothing\s+but|"
              r"no\s+(?:less|worse)\s+than|at\s+least|(?:less|more|better|worse)\s+than|exactly|only|just)\b")
CH_SKIP = {"and", "the", "a", "an", "of"}
LITERAL_VALUES: set[str] = set()

def ch_words(text):
    return [w for w in re.findall(r"[a-z0-9]+(?:\.[0-9]+)?", str(text or "").lower()) if not w in CH_SKIP]

def ch_word_in(word, pool):
    for other in pool:
        if word == other:
            return True
        if word.isalpha() and other.isalpha():
            if same_kind(word, other):
                return True
            if 6 <= len(word) and 6 <= len(other) and word[:2] == other[:2] and edits_within(word, other, 1):
                return True
    return False

def ch_alts(value):
    plain = re.sub(r"\((?:stated|revealed[^)]*)\)", " ", str(value or "").lower()).replace('"', " ")
    plain = re.sub(r"\b(?:only|just|exactly|strictly)\b", " ", plain)
    return [ws for ws in (ch_words(part) for part in re.split(r"\bor\b", plain)) if ws]

def ch_writes(value, text):
    return ch_match(ch_alts(value), ch_words(text))

def ch_match(alts, pool):
    if not pool:
        return False
    for ws in alts:
        if all(ch_word_in(w, pool) for w in ws):
            return True
        target = "".join(ws)
        if 2 <= len(ws) or re.search(r"\d", target) or any(re.search(r"\d", p) for p in pool):
            for i in range(0, len(pool)):
                run = ""
                for j in range(i, len(pool)):
                    run += pool[j]
                    if target == run:
                        return True
                    if len(run) >= len(target):
                        break
    return False

def ch_touches(value, text):
    pool = ch_words(text)
    low = str(text or "").lower()
    for ws in ch_alts(value):
        for w in ws:
            if ch_word_in(w, pool):
                return True
            if re.fullmatch(r"\d+(?:\.\d+)?", w) and re.search(r"(?<![\d.])" + re.escape(w) + r"(?!\d|\.\d)", low):
                return True
            if 3 <= len(w) and any(3 <= len(p) and (p.startswith(w[:3]) or w.startswith(p[:3])) for p in pool):
                return True
    return False

def ch_parts(text):
    low = str(text or '').strip(None).lower()
    return [p.strip(None) for p in re.split(r"[,;|]", low) if p.strip(None)] or ([] if not low else [low])

def ch_sides(part):
    sides = [s.strip(None) for s in re.split(r"\s*\+\s*", str(part)) if s.strip(None)]
    return [] if not 2 <= len(sides) else sides

def figure_written(fig, said):
    for hit in re.finditer(r"(?<![\d.])" + re.escape(fig) + r"(?!\d|\.\d)", said):
        if "." in fig:
            return True
        tail = said[hit.end():]
        word = re.match(r"\s*([a-z]+)", tail)
        if not word or 3 >= len(word.group(1)):
            return True
    return False

EXCLUDE_WORDS = r"\b(?:no(?!-)|not|without|nothing|never|neither|nor|except|excluding|avoid\w*|anything but|other than|rather than|instead of)\b"

def positive_value(value, name = ""):
    low = str(value).lower().strip(None)
    if re.fullmatch(r"\W*(?:yes|no|true|false|none|required|included|needed)\W*", low):
        return True
    if LITERAL_VALUES and " ".join(re.findall(r"[a-z0-9]+", re.sub(r"\((?:stated|revealed[^)]*)\)", " ", low))) in LITERAL_VALUES:
        return True

    def content(text):
        return [w for w in re.findall(r"[a-z]{3,}", text) if not w in LINK_WORDS]
    low = re.sub(BOUND_WORDS, " ", low)
    low = re.sub(r"\b(?:no|not)\s+[a-z]+er\b(?=\s+than\b)", " ", low)
    if re.search(r"\d", low):
        low = re.sub(r"\b(?:no|not)\s+[a-z]+er\b", " ", low)
    body = re.sub(r"\b(?:" + VALUE_LINKS + r")\b", " ", low)
    if any(4 <= len(content(part)) for part in re.split(r",|;|/|&|\(|\)|\bor\b|\band\b", body)):
        return False
    hit = re.search(EXCLUDE_WORDS, low)
    if not hit:
        return True
    if content(low[:hit.start()]):
        return True
    rest = content(low[hit.end():])
    own = set(prop_words(name)) | set(re.findall(r"[a-z]+", str(name).lower()))
    return bool(rest) and bool(own) and all(w in own or w.rstrip("s") in own for w in rest)

def value_seen(value, samples):
    pool = str(samples or "").lower()
    if not pool.strip(None):
        return False
    body = re.sub(r"\b(?:" + VALUE_LINKS + r")\b", " ", str(value).lower())
    for w in re.findall(r"[a-z]{3,}", body):
        if not w in LINK_WORDS and re.search(r"\b" + re.escape(w if not 4 < len(w) else w[:-1]), pool):
            return True
    for run in re.findall(r"\d+(?:\.\d+)?(?:\s*[:x/]\s*\d+(?:\.\d+)?)*", body):
        figs = re.findall(r"\d+(?:\.\d+)?", run)
        if re.search(r"(?<![\d.])" + r"\D{0,4}".join(re.escape(f) for f in figs) + r"(?![\d.])", pool):
            return True
    return False

def kind_words(text):
    return {w.rstrip("s") for w in re.findall(r"[a-z]{3,}", str(text).lower()) if not w in LINK_WORDS}

def kind_tokens(text):
    return [w for w in re.findall(r"[a-z]{3,}", re.sub(r"'s\b", "", str(text).lower().replace("\u2019", "'"))) if not w in LINK_WORDS]

def plural_forms(word):
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
    return {x for x in out if 3 <= len(x)}

def same_kind(a, b):
    return b == a or str(b) in plural_forms(a) or str(a) in plural_forms(b)

def kind_in(word, pool):
    return any(same_kind(word, other) for other in pool)

def kinds_within(words, pool):
    return all(kind_in(w, pool) for w in words)

def kind_score(own, words, shared):
    return sum(1.0 if not kind_in(w, shared) else 0.5 for w in own if kind_in(w, words))

def name_pattern(name):
    pats = []
    for w in kind_tokens(name):
        forms = sorted(plural_forms(w), key=lambda x: (-len(x), x))
        pats.append(r"(?:" + "|".join(re.escape(x) for x in forms) + r")(?:'s\b|'|\b)")
    return r"[\s'-]+".join(pats)

VALUE_LINKS = 'plus|than|nor|neither|either|only|exactly|below|above|over|under|within|between|maximum|minimum|or lower|or higher|or smaller|or larger|or less|or more|up to|less than|more than|no less than|no more than|at most|at least'

def one_prop_hint(hint):
    try:
        head, sep, detail = hint.partition(":")
        if not '/' in head or not sep:
            return hint
        tag = re.search(r"\(([^()]*)\)\s*$", head)
        names = [n.strip(None) for n in (head if not tag else head[:tag.start()]).split("/") if n.strip(None)]
        if 2 > len(names):
            return hint
        return '{}{}:{} (or {})'.format(names[0], '' if not tag else ' (' + tag.group(1).strip(None) + ')', detail, ', '.join(names[1:]))
    except (Exception,):
        return hint

def lead_value(detail, request, product):
    try:
        low = str(detail).lower()
        hit = re.match(r"\s*([a-z][a-z-]*[a-z])(?:\s*[,;.()]|\s*$|\s+(?:" + "|".join(sorted(LINK_WORDS)) + r")\b)", low)
        if not hit:
            return ""
        word = hit.group(1)
        kinds = [w for w in re.findall(r"[a-z]{3,}", str(product).lower()) if not w in LINK_WORDS]
        if 3 > len(word) or word in LINK_WORDS or not kinds or word.rstrip("s") in {k.rstrip("s") for k in kinds}:
            return ""
        first = kinds[0].rstrip("s")
        toks = re.findall(r"[a-z][a-z'-]*", str(request).lower())
        for i in range(0, len(toks) - 1):
            if toks[i] == word and toks[i + 1].split("'")[0].rstrip("s") == first:
                return word
    except (Exception,):
        return ""
    return ""

def story_guess(req, request, refs):
    req = str(req)
    head, _, value = req.partition(":")
    low = value.strip(None).lower()
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
    said_stems = {s if not (s.endswith('s') and 3 < len(s)) else s[:-1] for s in said_words}
    missing = []
    content = [w for w in re.findall(r"[a-z]{3,}", body) if not w in LINK_WORDS]
    for w in content:
        stem = (w if not (w.endswith('s') and 3 < len(w)) else w[:-1]) if not (w.endswith('es') and 4 < len(w)) else w[:-2]
        if not any(w == s or (s.startswith(stem) and len(s) <= len(stem) + 3) or (stem.startswith(s) and len(s) >= max(4, len(stem) - 2))
                   or (4 <= len(w) and s.startswith(w)) or (4 <= len(w) and 4 <= len(s) and s[0] == w[0] and edits_within(s, w, 1))
                   for s in said_words) \
                and not any(3 <= len(s) and stem.startswith(s) and len(s) >= len(stem) - 3 for s in said_stems):
            missing.append(w)
    if 2 >= len(content) and any(4 <= len(w) and not w in missing for w in content):
        return []
    for run in re.findall(r"\d+(?:\.\d+)?(?:\s*[:x/]\s*\d+(?:\.\d+)?)*", body):
        figs = re.findall(r"\d+(?:\.\d+)?", run)
        if 2 <= len(figs) and not re.search(r"(?<![\d.])" + r"\D{0,4}".join(re.escape(f) for f in figs) + r"(?!\d)", said):
            missing.append(run)
    return missing

REF_WORDS = r"\bas shown\b|\bshown on\b|\bsame as\b|\bpage\b|\bsold by\b|\bredirect|\brefer(?:s|red|ring)?\b|\blisting points\b|\bpoints you\b"
REF_STEPS = 0.4
REF_CALLS = 12
NO_FIT_SAID = re.compile(r"\b(?:closest|nearest|falls? short|fallback|nothing (?:fully )?(?:fits|meets)|no listing (?:fully )?(?:fits|meets|satisfies))\b", re.I)
REF_HOPS = 8
DIFFER_WORDS = (r"\bset apart from\b|\bapart from\b|\bdistinct from\b|\bdifferent (?:from|to|than)\b|\bdiffer(?:s|ent|ing)?\s+from\b"
                r"|\bother than\b|\bunlike\b|\bnot the same as\b|\banything but\b")
POINT_WORDS = r"\bwithin\b|\bat (?:most|least)\b|\bno (?:more|less) than\b|\bsame as\b|\bas shown\b|\bthe value\b|\bthe one\b"
REF_DIFFER_STEPS = 0.3
REF_DIFFER_BLIND = 5
SELLER_KEYS = r"shop|seller|store|vendor|brand|make|manufactur|merchant|retailer|supplier|sold.?by"

def ref_text(value):
    return bool(re.search(REF_WORDS, str(value or "").lower()))

def ref_toks(text):
    low = re.sub(r"(\d)(?:\s+|\s*-\s*)([a-z]{1,8})\b", lambda h: h.group(1) + h.group(2) if not h.group(2) in LINK_WORDS else h.group(0),
                 str(text or "").lower())
    out = []
    for tok in re.findall(r"\d+(?:[.,:/]\d+)*[a-z]*|[a-z][a-z0-9]*", low):
        if tok in LINK_WORDS or (1 == len(tok) and not tok.isdigit()):
            continue
        if 3 < len(tok) and tok.endswith("s") and not tok.endswith("ss") and not tok[-2].isdigit():
            tok = tok[:-1] if not re.search('(?:ch|sh|x|z)es$', tok) else tok[:-2]
        out.append(tok)
    return out

def ref_figs(text):
    out: set[tuple[float, str]] = set()
    try:
        low = str(text or "").lower().replace(",", "")
        for hit in re.finditer(r"(\d+(?:\.\d+)?(?:\s*/\s*\d+(?:\.\d+)?)*)\s*-?\s*([a-z]{1,8})\b", low):
            unit = hit.group(2)
            if re.match(r"\s*/\s*(?:s|sec)\b", low[hit.end():]) or (2 < len(unit) and unit.endswith("ps")):
                continue
            if 3 <= len(unit) and unit.endswith("s") and not unit.endswith("ss"):
                unit = unit[:-1] if not re.search('(?:ch|sh|x|z)es$', unit) else unit[:-2]
            if unit in LINK_WORDS:
                continue
            for fig in re.split(r"\s*/\s*", hit.group(1)):
                out.add((float(fig), unit))
    except (Exception,):
        return out
    return out

def ref_name(text):
    return re.sub(r"[^a-z0-9]", "", re.sub(r"['\u2019]s\b.*$", "", str(text or "").lower()))

def ref_relation(value):
    low = str(value or "").lower()
    if re.search(r"\bcounterpart\b|\bgoes with\b|\bpairs? with\b|\bgo together with\b|" + COMPAT_WORDS, low) \
            and not re.search(DIFFER_WORDS, low):
        return "pair", None, ""
    if re.search(DIFFER_WORDS, low):
        return "differ", None, ""
    hit = re.search(r"\bwithin\s+(\d+(?:\.\d+)?)\s*([a-z%]+)?", low)
    if hit:
        return "within", float(hit.group(1)), (hit.group(2) or '').strip(None)
    if re.search(r"\b(?:at most|no more than|up to|or less|or lower|or smaller)\b", low):
        return "at most", None, ""
    if re.search(r"\b(?:at least|no less than|or more|or higher|or larger)\b", low):
        return "at least", None, ""
    return "same", None, ""

def ref_parse(text, ok = None):
    try:
        body = re.split(r"\(\s*or\b", str(text or ""))[0]
        hit = next((m for m in re.finditer(r"(?<![A-Za-z'])([A-Z][a-z]{3,})(?:'s)?(?![A-Za-z])", body) if 0 < m.start()
                    and (ok is None or ok(m.group(1)))), None)
        if not hit:
            return None
        before, after, kind = body[:hit.start()], body[hit.end():], ""
        sold = re.search(r".*(?:\bthe|\bfor|\ba|\ban)\s+(.*?)\s+(?:sold|made|offered|listed)\s+by\s*$", before, re.I | re.S)
        if not sold:
            lead = re.match('\\s*([^,;(]*?)(?=\\s+(?:with|that|which|in|at|having|from|of|for|sold)\\b|[,;]|$)', after, re.I)
            if lead:
                kind, after = (lead.group(1), after[lead.end():])
        else:
            kind = sold.group(1)
        parts = re.split(r",|;|\band\b|\bwith\b|\bthat\s+(?:is|are|has|have|comes?(?:\s+as)?)\b|\bwhich\b|\bin\b|\bat\b", after, flags=re.I)
        details = []
        for part in parts:
            words = [w for w in re.findall(r"[A-Za-z0-9][A-Za-z0-9.%:/&'-]*", part) if not w.lower() in LINK_WORDS]
            if words:
                details.append(" ".join(words))
        return dict(seller=hit.group(1), kind=kind.strip(None)[:80], details=details[:6])
    except (Exception,):
        return None

def prop_named(prop, text, exact = False):
    part = re.sub(r"([a-z])([A-Z])", r"\1 \2", re.split(r"\s*\(", str(prop or ""))[0])
    words = re.findall(r"[a-z0-9]+", part.lower())
    toks = re.findall(r"[a-z0-9]+", str(text or "").lower())
    if not words or 3 > len(' '.join(words)):
        return False

    def same(a, b):
        return b == a or (not exact and ((3 <= len(b) and a in (b + "s", b + "es")) or (3 <= len(a) and b in (a + "s", a + "es"))))
    n = len(words)
    return any(all(same(t, w) for t, w in zip(toks[i: i + n], words)) for i in range(0, len(toks) - n + 1))

COMPAT_WORDS = r"\bcompatib\w*|\bplay(?:s|ing)?\s+(?:nicely|well)\b|\bin tune\b|\bharmoni[sz]\w*"

def ref_clause(source, seller):
    try:
        text, name = str(source or ""), str(seller or '').strip(None)
        hit = None if not name else re.search('\\b' + re.escape(name), text, re.I)
        if not hit:
            return ""
        cuts = r"[.!?](?=\s)|;|\b(?:first|second|third|fourth|fifth|lastly|finally)\b\s*,"
        back = [m.end() for m in re.finditer(cuts, text[:hit.start()], re.I)]
        ahead = re.search(cuts, text[hit.end():], re.I)
        return text[(0 if not back else back[-1]): hit.end() + (len(text) - hit.end() if not ahead else ahead.start())]
    except (Exception,):
        return ""

def ref_span(prop, source):
    try:
        words = [w for w in re.findall(r"[a-z0-9]+", re.split(r"\s*\(", str(prop))[0].lower()) if not w in LINK_WORDS]
        hit = None if not words else re.search('\\b' + '\\W+'.join(map(re.escape, words)) + '\\b', str(source).lower())
        if not hit:
            return ""
        return re.split(r"\(\s*or\b|(?<=[.!?])\s", str(source)[hit.end(): hit.end() + 500])[0]
    except (Exception,):
        return ""

def lensed(piece):
    try:
        piece = json.loads(json.dumps(piece))
    except (TypeError, ValueError):
        return piece

    def visit(d):
        if not any(isinstance(val, (str,)) and "id" in str(key).lower() and re.fullmatch(ID_LIKE, val) for key, val in d.items()):
            return
        names = {str(key).lower(): key for key in d}
        if not any(re.search(r"price|cost", low) and is_number(d[key]) for low, key in names.items()):
            picks = [key for low, key in names.items() if is_number(d[key]) and re.search(r"amount|fee|charge|pay|rate|value", low)
                     and not re.search(r"total|count|qty|quant|unit|due|sum|rating|star|score|page|number|num|size|weight|width|height|length|found|match", low)]
            if 1 == len(picks):
                d["price"] = d[picks[0]]
        if not any(re.search(r"stock|avail", low) for low in names):
            picks = [key for low, key in names.items() if isinstance(d[key], (bool,))
                     and re.search(r"hand|orderable|buyable|sellable|ready|ship|instock|purchas|in_store", low)]
            if 1 == len(picks):
                d["in_stock"] = d[picks[0]]
        if not any(re.search(r"title|heading", low) for low in names):
            picks = [key for low, key in names.items() if isinstance(d[key], (str,)) and re.search(r"headline|name|label|caption", low)
                     and not re.search(r"shop|seller|store|vendor|brand|make|categor|shelf|department|id", low)]
            if 1 == len(picks):
                d["title"] = d[picks[0]]
    walk_dicts(piece, visit)
    return piece

class ArgCheck:

    TYPES = (dict(string=lambda val: isinstance(val, (str,)), integer=lambda val: isinstance(val, (int,)) and (not isinstance(val, (bool,))), number=is_number, boolean=lambda val: isinstance(val, (bool,)), object=lambda val: isinstance(val, (dict,)), array=lambda val: isinstance(val, (list,)), null=lambda val: val is None))

    @classmethod
    def type_list(cls, jschema):
        type_list = jschema.get('type', None)
        return list(type_list or []) if not isinstance(type_list, (str,)) else [type_list]

    @classmethod
    def conforms(cls, piece, ktype):
        test = cls.TYPES.get(ktype, None)
        return True if not test else test(piece)

    @classmethod
    def convert(cls, piece, type_list):
        if not type_list:
            return piece
        if any(cls.conforms(piece, fkey) for fkey in type_list):
            if "integer" in type_list and isinstance(piece, (float,)) and piece.is_integer():
                return int(piece)
            return piece
        for ktype in type_list:
            if ktype == 'integer':
                if isinstance(piece, (float,)) and piece.is_integer():
                    return int(piece)
                if isinstance(piece, (str,)) and re.fullmatch(r"\s*-?\d+(\.0+)?\s*", piece):
                    return int(float(piece))
            elif ktype == 'number' and isinstance(piece, (str,)) and re.fullmatch(r"\s*-?\d+(\.\d+)?\s*", piece):
                num = float(piece)
                return num if not num.is_integer() else int(num)
            elif ktype == 'string' and is_number(piece):
                return str(piece)
            elif ktype == 'boolean' and isinstance(piece, (str,)) and piece.lower() in ("true", "false"):
                return 'true' == piece.lower()
        return piece

    @classmethod
    def execute(cls, piece, jschema, where = "args"):
        if not isinstance(jschema, (dict,)):
            return piece, []
        type_list = cls.type_list(jschema)
        piece = cls.convert(piece, type_list)
        if type_list and not any(cls.conforms(piece, fkey) for fkey in type_list):
            return piece, ['{}: expected {}'.format(where, '/'.join(type_list))]
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
        if isinstance(piece, (str,)):
            for opt in options:
                if isinstance(opt, (str,)) and opt.lower() == piece.strip(None).lower():
                    return opt, []
        return piece, ['{}: must be one of {}'.format(where, options)]

    @staticmethod
    def _const(piece, jschema, where):
        return piece, (['{}: must equal {!r}'.format(where, jschema['const'])] if not piece == jschema['const'] else [])

    @staticmethod
    def _min_length(piece, jschema, where):
        if isinstance(piece, (str,)) and len(piece) < jschema["minLength"]:
            return piece, ['{}: shorter than {} characters'.format(where, jschema['minLength'])]
        return piece, []

    @staticmethod
    def _max_length(piece, jschema, where):
        return (piece if not isinstance(piece, (str,)) else piece[:jschema['maxLength']]), []

    @staticmethod
    def _bound(call_name, test, words):
        def rule_fn(piece, jschema, where):
            ceiling = jschema[call_name]
            if is_number(piece) and is_number(ceiling) and test(piece, ceiling):
                return piece, ['{}: {} {}'.format(where, words, ceiling)]
            return piece, []
        return rule_fn

    @staticmethod
    def _min_items(piece, jschema, where):
        if isinstance(piece, (list,)) and len(piece) < jschema["minItems"]:
            return piece, ['{}: needs at least {} items'.format(where, jschema['minItems'])]
        return piece, []

    @staticmethod
    def _max_items(piece, jschema, where):
        if isinstance(piece, (list,)) and len(piece) > jschema["maxItems"]:
            return piece, ['{}: allows at most {} items'.format(where, jschema['maxItems'])]
        return piece, []

    @classmethod
    def _items(cls, piece, jschema, where):
        if not isinstance(piece, (list,)) or not isinstance(jschema['items'], (dict,)):
            return piece, []
        faults, mended = [], []
        for idx, thing in enumerate(piece, 0):
            thing, matched = cls.execute(thing, jschema["items"], '{}[{}]'.format(where, idx))
            mended.append(thing)
            faults += matched
        return mended, faults

    @classmethod
    def _properties(cls, piece, jschema, where):
        if not isinstance(piece, (dict,)):
            return piece, []
        prop_specs = jschema.get('properties', None) or {}
        faults = ['{}.{}: required'.format(where, call_name) for call_name in jschema.get('required', None) or [] if not call_name in piece]
        if jschema.get('additionalProperties', None) is False:
            faults += ['{}.{}: not an allowed argument'.format(where, call_name) for call_name in piece if not call_name in prop_specs]
        mended = {}
        for call_name, thing in piece.items():
            if call_name in prop_specs:
                thing, matched = cls.execute(thing, prop_specs[call_name], '{}.{}'.format(where, call_name))
                faults += matched
            mended[call_name] = thing
        return mended, faults

ArgCheck.RULES = [
    ("enum", ArgCheck._enum),
    ("const", ArgCheck._const),
    ("minLength", ArgCheck._min_length),
    ("maxLength", ArgCheck._max_length),
    ("minimum", ArgCheck._bound("minimum", lambda val, lim: lim > val, "below minimum")),
    ("maximum", ArgCheck._bound("maximum", lambda val, lim: lim < val, "above maximum")),
    ("exclusiveMinimum", ArgCheck._bound("exclusiveMinimum", lambda val, lim: lim >= val, "must be above")),
    ("exclusiveMaximum", ArgCheck._bound("exclusiveMaximum", lambda val, lim: lim <= val, "must be below")),
    ("minItems", ArgCheck._min_items),
    ("maxItems", ArgCheck._max_items),
    ("items", ArgCheck._items),
    ("type", lambda val, s, p: (val, []) if not isinstance(val, (dict,)) else ArgCheck._properties(val, s, p)),
]

class SendFault:

    def __init__(self, why):
        self.data = None
        self.error = dict(status=None, body=why[:300])

class Wire:

    def __init__(self, bind):
        self.t0 = monotonic()
        self.bind = dict(bind)
        self.model_list = ProxyClient(retry_delay=1, max_retries=1, timeout=70)
        self.world = ProxyClient(retry_delay=1, max_retries=1, timeout=95)
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

    def spare(self):
        return DEADLINE - (monotonic() - self.t0)

    def spent_out(self):
        return self.spend >= SPEND_CAP * self.__dict__.get("scale", 1.0)

    def send(self, client, path, body):
        gap = PACE_GAP if not monotonic() < self.slow_until else SLOW_GAP
        wait = self.last_send + gap - monotonic()
        if "/inference/" in path:
            wait = max(wait, self.model_resume - monotonic())
        wait = min(wait, self.spare() - 1)
        if wait > 0:
            sleep(wait)
        self.last_send = monotonic()
        try:
            answer = client.post_verbose(path, body)
        except (Exception,) as err:
            answer = SendFault('{}: {}'.format(type(err).__name__, err))
        if answer.data is None and (answer.error or {}).get('status', None) in (429, 503):
            self.slow_until = monotonic() + SLOW_FOR
        return answer

    def pause(self, seed, tries, status):
        mix = (len(seed) * 7 + sum(ord(ch) for ch in seed[-64:]) + 31 * tries + 13 * self.tick) % 101 / 100.0
        return min(8.0, (1.0 if not status == 429 else 2.0) * 2 ** tries) * (0.75 + 0.5 * mix)

    def down(self, at = None):
        return self.no_model or (self.misses > 0 and (at if not at is None else monotonic()) - self.last_reply > DOWN_AFTER)

    def usable(self, model_list, tools = False):
        return not self.no_model and not self.mute and any(
            not mdl in self.banned and not (tools and mdl in self.tool_banned) for mdl in list(model_list) + self.spare_models)

    @staticmethod
    def plain(msgs):
        out = []
        for msg in msgs:
            msg = dict(msg)
            if isinstance(msg.get('content', None), (list,)):
                msg["content"] = "\n\n".join(str(part.get('text', None) or "") for part in msg["content"] if isinstance(part, (dict,)))
            out.append(msg)
        return out

    def query_model(self, msgs, model_list, token_cap,
            toolset = None, force = None):
        self.call_fail = ""
        self.call_models = set()
        if self.mute or self.no_model:
            return None
        self.asked_n = self.__dict__.get("asked_n", 0) + 1
        seed = ""
        kinds: set[str] = set()
        refused_here: set[str] = set()
        ladder = list(model_list) + [mdl for mdl in self.spare_models if not mdl in model_list]
        lone = 1 >= len({mdl for mdl in ladder if not mdl in self.banned})
        if (not toolset and msgs and 'user' == msgs[-1].get('role', None) and isinstance(msgs[-1].get('content', None), (str,))
                and any("json object" in str(m.get('content', None) or "").lower() for m in msgs)):
            msgs = msgs[:-1] + [dict(msgs[-1], content=msgs[-1]["content"].rstrip() + JSON_AS_TEXT)]
        for pos, mdl in enumerate(ladder, 0):
            if self.spend >= HARD_SPEND * self.__dict__.get("scale", 1.0) and not self.no_model:
                self.no_model = True
                self.log.append(dict(kind='spend_stop', spend=round(self.spend, 4)))
            if self.no_model:
                return self.failed(kinds)
            if mdl in self.banned or mdl in refused_here or (toolset and mdl in self.tool_banned):
                continue
            if "sonnet" in mdl:
                cheaper_next = pos + 1 < len(model_list) and not 'sonnet' in model_list[pos + 1]
                guess = sum(len(json.dumps(m, default=str)) for m in msgs) / 2.3 * 2e-6 + min(token_cap, 1500) * 1e-5
                scale = self.__dict__.get("scale", 1.0)
                if self.spent_out() or (cheaper_next and (self.spend >= SOFT_SPEND * scale or self.spend + guess > SPEND_CAP * scale)):
                    continue
            plain = False
            for _ in range(0, MODEL_TRIES if not self.down() else 1):
                spare = self.spare()
                if spare < LAST_CALL - 10:
                    return self.failed(kinds)
                self.model_list.timeout = int(max(8, min(75 if not self.down() else 30, spare - LAST_CALL + 15)))
                envelope: dict[str, Any] = (dict(model=mdl, messages=self.cached(msgs, mdl) if not plain else self.plain(msgs), temperature=0, max_tokens=token_cap))
                if toolset:
                    envelope["tools"] = toolset
                    envelope["tool_choice"] = (('auto' if not (force and not mdl in self.__dict__.get('no_force', set())) else dict(type='function', function=dict(name=force))))
                self.call_models.add(mdl)
                started = monotonic()
                answer = self.send(self.model_list, "/inference/chat/completions", envelope)
                data = answer.data
                if isinstance(data, (dict,)) and data.get('choices', None):
                    self.last_reply, self.misses = monotonic(), 0
                    use_info = data.get('usage', None) or {}
                    if is_number(use_info.get('cost', None)):
                        self.spend += float(use_info["cost"])
                    msg = data['choices'][0].get('message', None) or {}
                    if msg.get('content', None) or msg.get('tool_calls', None):
                        return msg
                    kinds.add("empty")
                    self.log.append(dict(kind='inference_failure', model=mdl, error='empty'))
                    if _ == 0 and not self.down():
                        continue
                    break
                self.misses = self.misses + 1
                self.log.append((dict(kind='inference_failure', model=mdl, error='empty' if not answer.error else str(answer.error)[:300])))
                error = answer.error or {}
                status, body = error.get('status', None), str(error.get('body', None) or "")
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
                    listed = None if not status == 403 else re.search('allowed_models"?\\s*:\\s*\\[([^\\]]*)', body)
                    if listed and not self.spare_models:
                        self.spare_models = [m for m in re.findall(r'"([^"\s]{3,100})"', listed.group(1))
                                             if not m in self.banned and not m in model_list][:2]
                        ladder += [m for m in self.spare_models if not m in ladder]
                        lone = 1 >= len({m for m in ladder if not m in self.banned})
                    break
                if status in (400, 413, 422):
                    if force and "tool_choice" in said and not mdl in self.__dict__.get('no_force', set()):
                        self.__dict__.setdefault("no_force", set()).add(mdl)
                        continue
                    kinds.add("hard")
                    if plain:
                        refused_here.add(mdl)
                        break
                    plain = True
                    continue
                if isinstance(status, (int,)) and 400 <= status < 500 and not status in (408, 425, 429):
                    kinds.add("hard")
                    refused_here.add(mdl)
                    break
                kinds.add("soft")
                if 20 < monotonic() - started:
                    if self.down() and (lone or 2 <= len(self.call_models)):
                        return self.failed(kinds)
                    break
                if not seed:
                    seed = json.dumps(msgs[-1:], default=str)[-600:]
                wait = self.pause(seed + mdl, min(self.misses - 1, 6), status)
                if self.spare() - wait < LAST_CALL - 10:
                    return self.failed(kinds)
                self.model_resume = monotonic() + wait
                if self.down(self.model_resume) and (lone or 2 <= len(self.call_models)):
                    return self.failed(kinds)
        return self.failed(kinds)

    def failed(self, kinds):
        self.call_fail = ('' if not kinds == {'hard'} else 'refused') if not 'empty' in kinds else 'empty'
        return None

    @staticmethod
    def cached(msgs, mdl):
        if not mdl.startswith("anthropic/") or 2 >= len(msgs):
            return msgs
        marked = [dict(m) for m in msgs]
        for idx in (0, len(marked) - 1):
            content = marked[idx].get('content', None)
            if isinstance(content, (str,)) and content:
                marked[idx]["content"] = [dict(type='text', text=content, cache_control=dict(type='ephemeral'))]
        return marked

    def consult(self, ask_text, token_cap, model_list):
        msg = self.query_model([dict(role='user', content=ask_text)], model_list, token_cap)
        return json_object((msg or {}).get('content', None) or "")

    def act(self, acts):
        env_body: dict[str, Any] = dict(self.bind)
        env_body["step"] = self.tick + 1
        if not 1 == len(acts):
            env_body['actions'] = acts
        else:
            env_body['action'] = acts[0]
        tries = reused = 0
        for _ in range(0, 3 + WORLD_TRIES):
            spare = self.spare()
            if spare < 2:
                return None
            self.world.timeout = int(max(3, min(95, spare - 1)))
            answer = self.send(self.world, "/environment/call", env_body)
            if isinstance(answer.data, (dict,)):
                data = answer.data
                self.tick = self.tick + len(acts) if not isinstance(data.get('step', None), (int,)) else data['step']
                if data.get('finished', None):
                    self.done = True
                return data
            error = answer.error or {}
            status_line, envelope = error.get('status', None), str(error.get('body', None) or "")
            self.log.append(dict(kind='env_failure', status=status_line, body=envelope[:300]))
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
                        reused = reused + 1
                        env_body["step"] += 1
                        continue
                    return None
                self.tick = int(expected.group(1)) - 1
                env_body["step"] = self.tick + 1
                continue
            if status_line in (403, 422) or tries >= WORLD_TRIES - 1 or (status_line in (400, 401, 404) and tries >= 1):
                return None
            wait = self.pause(json.dumps(env_body, default=str)[-600:], tries, status_line)
            tries = tries + 1
            if 2 > self.spare() - wait:
                return None
            sleep(wait)
        return None

class Records:

    def __init__(self):
        self.items: list[Any] = []
        self.ids: set[str] = set()
        self.read: set[str] = set()
        self.label_of: dict[str, str] = {}

    def add(self, obs):
        self.items.append(obs)

        def visit(d):
            for key, piece in d.items():
                if isinstance(piece, (str,)) and "id" in str(key).lower() and re.fullmatch(ID_LIKE, piece):
                    self.ids.add(piece)
            ident = next((piece for key, piece in d.items() if isinstance(piece, (str,)) and str(key).lower() in ("id", "_id")
                          or (isinstance(piece, (str,)) and str(key).lower().endswith("id") and re.fullmatch(ID_LIKE, piece))), None)
            if ident and any(isinstance(val, (dict,)) and 2 <= sum((1 for v in val.values() if isinstance(v, (str, int, float))))
                             for val in d.values()):
                self.read.add(ident)
            own = next((piece.strip(None) for key, piece in d.items() if isinstance(piece, (str,)) and 0 < len(piece.strip(None)) <= 60
                        and re.search(r"categor|shelf|department|aisle", str(key).lower()) and not 'id' in str(key).lower()), "")
            if ident and own:
                self.__dict__.setdefault("label_of", {}).setdefault(ident, own)
        walk_dicts(obs, visit)

    def price(self, thing):
        found: list[float] = []

        def visit(d):
            if not any(isinstance(val, (str,)) and thing == val and "id" in str(key).lower() for key, val in d.items()):
                return
            for key, val in d.items():
                if is_number(val) and not isinstance(val, (bool,)) and re.search(r"price|cost", str(key).lower()):
                    found.append(float(val))
                    return
        for rec in self.items:
            walk_dicts(rec, visit)
        return None if not found else found[-1]

    def about(self, thing, hold = 4):
        matched: list[str] = []

        def seek(piece, lvl):
            if lvl > 6 or 8 <= len(matched):
                return
            if isinstance(piece, (dict,)):
                if any(isinstance(val, (str,)) and thing == val for val in piece.values()):
                    text_body = as_text(clipped(piece, 400))[:2500]
                    if not text_body in matched:
                        matched.append(text_body)
                    return
                for val in piece.values():
                    seek(val, lvl + 1)
            elif isinstance(piece, (list,)):
                for val in piece:
                    seek(val, lvl + 1)

        for rec in self.items:
            seek(rec, 0)
        return matched[-hold:]

    def digest(self, char_cap = 24000, value_chars = 160):
        record_texts, char_total = [], 0
        for rec in reversed(self.items):
            text_body = as_text(clipped(rec, value_chars))
            if char_total + len(text_body) > char_cap:
                continue
            record_texts.append(text_body)
            char_total += len(text_body)
        return record_texts

    def cards(self, char_cap = 30000, cheaper_than = "", skip = None, first = ()):
        merged: dict[str, dict[str, Any]] = {}
        order: list[str] = []

        def visit(d):
            ident = next((val for key, val in d.items() if isinstance(val, (str,)) and "id" in str(key).lower()
                          and re.fullmatch(ID_LIKE, val) and val in self.ids), None)
            if not ident or (4 > len(d) and not any(isinstance(v, (dict,)) and 2 <= len(v) for v in d.values())):
                return
            card = merged.setdefault(ident, {})
            if not ident in order:
                order.append(ident)
            card.update({key: val for key, val in d.items()})
            if any(isinstance(val, (dict,)) and 2 <= len(val) for val in d.values()):
                self.read.add(ident)

        for rec in self.items:
            walk_dicts(rec, visit)

        def price_of(ident):
            for key, val in (merged.get(ident, None) or {}).items():
                if is_number(val) and not isinstance(val, (bool,)) and re.search(r"price|cost", str(key).lower()):
                    return float(val)
            return None

        cap_price = None if not cheaper_than else price_of(cheaper_than)
        if skip:
            order = [i for i in order if not i in skip or cheaper_than == i]
        if not cap_price is None:
            order = [i for i in order if cheaper_than == i or price_of(i) is None or price_of(i) < cap_price]
        lines, total = [], 0
        front = set(first or ())
        ranked = sorted(order, key=lambda i: (not i in front, not i in self.read, order.index(i)))
        texts = {i: ('ROW ' if not i in self.read else 'DETAILS ') + as_text(clipped(merged[i], 160)) for i in ranked}
        if sum(len(t) for t in texts.values()) > char_cap:
            def gone(i):
                return any(Errand.key_stock(key, val) is False for key, val in merged[i].items())
            ranked = sorted(ranked, key=lambda i: (not i in front, not i in self.read, gone(i), order.index(i)))
        for ident in ranked:
            text_body = texts[ident]
            if total + len(text_body) > char_cap:
                continue
            lines.append(text_body)
            total += len(text_body)
        return lines

    def label_for(self, d):
        own = next((val.strip(None) for key, val in d.items() if isinstance(val, (str,)) and 0 < len(val.strip(None)) <= 60
                    and re.search(r"categor|shelf|department|aisle", str(key).lower()) and not 'id' in str(key).lower()), "")
        if own:
            return own
        labels = self.__dict__.get('label_of', None) or {}
        return next((labels[val] for key, val in d.items() if isinstance(val, (str,)) and val in labels and "id" in str(key).lower()), "")

    def rich_reads(self, shelf = ""):
        rich: set[str] = set()

        def visit(d):
            ident = next((val for key, val in d.items() if isinstance(val, (str,)) and "id" in str(key).lower() and val in self.read), None)
            if not ident:
                return
            if shelf and self.label_for(d).strip(None).lower() != shelf.strip(None).lower():
                return
            maps = [val for val in d.values() if isinstance(val, (dict,))]
            if any(3 <= len(val) for val in maps) or (not maps and 5 <= sum((1 for val in d.values() if isinstance(val, (str,)) and 0 < len(val) <= 70))):
                rich.add(ident)
        for rec in self.items:
            walk_dicts(rec, visit)
        return len(rich)

    def shelves_read(self):
        tally: dict[str, int] = {}

        def visit(d):
            ident = next((val for key, val in d.items() if isinstance(val, (str,)) and "id" in str(key).lower() and val in self.read), None)
            shelf = '' if not ident else self.label_for(d)
            if ident and shelf and any(isinstance(val, (dict,)) for val in d.values()):
                tally[shelf] = tally.get(shelf, 0) + 1
        for rec in self.items:
            walk_dicts(rec, visit)
        return sorted(tally, key=lambda s: -tally[s])

    def prop_menu(self, shelf = ""):
        tallies: dict[str, int] = {}
        long_tally: dict[str, int] = {}
        examples: dict[str, list[str]] = {}
        counts: dict[str, dict[str, int]] = {}
        sets = 0
        mapped, row_tally = 0, {}

        def on_shelf(d):
            return any(isinstance(val, (str,)) and val.strip(None).lower() == shelf.lower() and re.search(r"categor|shelf|department|aisle", str(key).lower())
                       for key, val in d.items()) or self.label_for(d).strip(None).lower() == shelf.strip(None).lower()

        def seek(piece, inside, lvl):
            if lvl > 6:
                return
            if isinstance(piece, (dict,)):
                inside = inside or not shelf or on_shelf(piece)
                if inside:
                    visit(piece)
                for val in piece.values():
                    if isinstance(val, (dict, list)):
                        seek(val, inside, lvl + 1)
            elif isinstance(piece, (list,)):
                for val in piece:
                    seek(val, inside, lvl + 1)

        def visit(d):
            nonlocal sets, mapped
            brief_vals = [(fkey, val) for fkey, val in d.items() if isinstance(val, (str,)) and 0 < len(val) <= 70 and not junk_key(fkey)]
            for fkey, val in d.items():
                if isinstance(val, (str,)) and 70 < len(val) and 20 <= len(val.split()):
                    name = re.sub('[_.]+', ' ', str(fkey)).strip(None).lower()
                    long_tally[name] = long_tally.get(name, 0) + 1
            if 5 > len(brief_vals) and not (shelf and 2 <= len(brief_vals) and not any(isinstance(v, (dict,)) for v in d.values())):
                return
            sets = sets + 1
            rowlike = 5 > len(brief_vals) and any(isinstance(val, (str,)) and "id" in str(fkey).lower() for fkey, val in d.items())
            mapped += 1 if not rowlike else 0
            for key, piece in brief_vals:
                if re.search(r"\d", piece) and re.fullmatch(CODE_LIKE, piece) and ("-" in piece or "_" in piece):
                    continue
                call_name = re.sub('[_.]+', ' ', str(key)).strip(None).lower()
                tallies[call_name] = tallies.get(call_name, 0) + 1
                if rowlike:
                    row_tally[call_name] = row_tally.get(call_name, 0) + 1
                held = counts.setdefault(call_name, {})
                held[piece.lower()] = held.get(piece.lower(), 0) + 1
                value_counts = examples.setdefault(call_name, [])
                if not piece.lower() in value_counts and 5 > len(value_counts):
                    value_counts.append(piece.lower())

        for rec in self.items:
            seek(rec, False, 0)
        if mapped:
            for fkey in [fkey for fkey in tallies if row_tally.get(fkey, 0) >= tallies[fkey]]:
                tallies.pop(fkey)

        for fkey in [fkey for fkey in tallies if long_tally.get(fkey, 0) >= 2 * tallies[fkey]]:
            tallies.pop(fkey)

        def spread(fkey):
            return len(examples.get(fkey, []))

        def figured(fkey):
            vals = examples.get(fkey, [])
            return bool(vals) and all(re.search(r"\d", v) for v in vals) and any(re.search(r"\d\s*[a-z\"']", v) for v in vals)
        def score(fkey):
            held = counts.get(fkey, None) or {}
            seen = sum(held.values())
            top = 1.0 if not seen else max(held.values()) / seen
            return min(1.0, tallies[fkey] / max(1, sets)) * (1.0 - top)
        if not sets >= 3:
            top_props = sorted(tallies, key=lambda fkey: (-(2 <= spread(fkey) and 2 <= tallies[fkey]), -tallies[fkey], -spread(fkey)))[:28]
        else:
            top_props = sorted(tallies, key=lambda fkey: (-score(fkey), -tallies[fkey], fkey))[:28]
        rows = []
        for fkey in top_props:
            marks = '' if not (2 <= spread(fkey) and 2 <= tallies[fkey]) else ' [values differ among the listings: {} seen]'.format(spread(fkey))
            marks += '' if not figured(fkey) else ' [figures with a unit]'
            rows.append('- {} (in {} listings){}: '.format(fkey, tallies[fkey], marks) + " | ".join(examples.get(fkey, [])))
        return "\n".join(rows), sets

    def label_lines(self):
        tallies: dict[str, dict[str, int]] = {}

        def visit(d):
            if 4 > len(d):
                return
            for key, piece in d.items():
                if isinstance(piece, (str,)) and 0 < len(piece) <= 40 and not (re.search(r"\d", piece) and re.fullmatch(CODE_LIKE, piece)):
                    value_counts = tallies.setdefault(str(key), {})
                    value_counts[piece] = value_counts.get(piece, 0) + 1

        for rec in self.items:
            walk_dicts(rec, visit)
        lns = []
        for key, value_counts in tallies.items():
            if 4 <= sum(value_counts.values()) and 2 <= len(value_counts) <= 8 and 2 <= max(value_counts.values()):
                lns.append('- {}: '.format(key) + " | ".join('{} (x{})'.format(val, qty_n) for val, qty_n in sorted(value_counts.items(), key=lambda kv: -kv[1])))
        return "\n".join(lns[:12])

class Needs:

    def __init__(self, wire, request):
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

    def add_told(self, items, source):
        added: list[str] = []
        if not isinstance(items, (list,)):
            return added
        known = {" ".join(str(t.get('text', None) or "").lower().split()) for t in self.told}
        for item in items[:12]:
            if isinstance(item, (str,)):
                item = dict(text=item, before_finish=None, needs_message=None)
            if not isinstance(item, (dict,)):
                continue
            words = " ".join(str(item.get('text', None) or "").split())
            if not words or words.lower() in known:
                continue
            known.add(words.lower())
            flags = {k: (None if not isinstance(item.get(k, None), (bool,)) else item.get(k, None)) for k in ("before_finish", "needs_message", "before_change")}
            self.told.append({"text": words[:400], "source": source, **flags})
            added.append(words[:120])
        if added:
            self.told_version = self.told_version + 1
        return added

    def finish_told(self):
        return [t for t in self.told if not t.get('before_finish', None) is False]

    def told_text(self):
        if not self.told:
            return ""
        rows = []
        for n, t in enumerate(self.told[-10:], 0):
            tags = [tag for tag, on in (("before the finish", t.get('before_finish', None)), ("by a message", t.get('needs_message', None))) if on]
            rows.append('{}) {}'.format(n + 1, t['text']) + (' [from the {}]'.format(t.get('source', None)) if not tags else ' [{}; from the {}]'.format(', '.join(tags), t.get('source', None))))
        return "\n[instructions in force] " + " ".join(rows)

    def items(self):
        rows = []
        for row in self.reading.get('items', None) or []:
            if isinstance(row, (dict,)) and str(row.get('product', None) or '').strip(None):
                qty = row.get('quantity', None)
                reqs = [] if not isinstance(row.get('requirements', None), (list,)) else row.get('requirements', None)
                dep = row.get('depends_on', None)
                rows.append((dict(product=str(row['product']).strip(None), quantity=1 if not (isinstance(qty, (int,)) and (not isinstance(qty, (bool,))) and (qty >= 1)) else qty, requirements=[str(r) for r in reqs if r and ':' in str(r)])))
                rows[-1]["depends_on"] = None if not (isinstance(dep, (int,)) and (not isinstance(dep, (bool,))) and (dep >= 1)) else dep - 1
        return [] if not 2 <= len(rows) else rows

    def quantity(self):
        if self.items():
            return 1
        need_n = self.reading.get('quantity', None)
        return 1 if not (isinstance(need_n, (int,)) and (not isinstance(need_n, (bool,))) and (need_n >= 1)) else need_n

    def counts(self):
        return [row["quantity"] for row in self.items()] or [self.quantity()]

    def items_text(self):
        rows = self.items()
        if not rows:
            return ""
        return ('THE REQUEST ASKS FOR SEVERAL PRODUCTS; the order contains precisely one line for each product, at its quantity, with nothing more; every listing is evaluated solely against the requirements of the product it belongs to (its category determines this), and the cheapest or best rule is applied inside each product separately: ' + " | ".join(
                    '{}) {} x{}'.format(n + 1, row['product'], row['quantity']) for n, row in enumerate(rows, 0)))

    def priority(self):
        return 'none' if not self.reading.get('priority', None) in ('cheapest', 'best') else self.reading.get('priority', None)

    def props_left(self):
        left = [r for r in self.written + self.uncovered if not r in self.let_go] if not not self.now_reqs is None else self.now_reqs
        return [r for r in left if not r in self.let_go and not re.search(r"\bprice\b|budget|quantity|\bstock\b|\bcents?\b", r.lower())]

    def text_body(self):
        if not not self.now_reqs is None:
            items = ['{} (stated)'.format(req) for req in self.written if not req in self.let_go]
            items += ['{} (revealed by the shopper)'.format(req) for req in self.uncovered if not req in self.let_go]
        else:
            items = list(self.now_reqs)
        collected = (('see the request' if not self.let_go else 'nothing remains to be matched on properties: any listing in the requested category that is in stock qualifies') if not items else '; '.join(items))
        if not self.spend_cap is None:
            collected += '; total price at most {} {}'.format(self.spend_cap, self.reading.get('currency', None) or '').rstrip()
        collected += '; ' + self.items_text() if self.items() else '; quantity {}'.format(self.quantity())
        if self.let_go:
            collected += "; no longer required: " + "; ".join(self.let_go)
        if self.indifferent:
            collected += "; shopper does not care about: " + ", ".join(self.indifferent)
        return collected

    def changes(self):
        if not self.let_go:
            return ""
        return ('\n\nCHANGES SINCE THE REQUEST (their text is overridden by these): the shopper DROPPED ' + "; ".join(self.let_go)
                + '. Once a requirement has been removed, it is irrelevant: do not reject, rank, or favor any listing because of it.')

    def overview(self, step_count, msgs):
        reading = self.reading
        if not reading:
            return ""
        lns = ['An initial interpretation of the request (verify it against the request itself):']
        if reading.get('product', None):
            lns.append('- product: {}'.format(reading.get('product', None)))
        if not self.items():
            lns.append('- quantity: {}'.format(self.quantity()))
        else:
            lns += ['- product {}: {}, quantity {} (one cart line)'.format(n + 1, row['product'], row['quantity']) for n, row in enumerate(self.items(), 0)]
            lns.append('- the order consists of precisely these lines, a single line per product and no extras; when the cart tool accepts a list of lines, include all of them in every cart call')
        lns += ['- stated requirement: {}'.format(req) for req in reading.get('requirements', None) or []]
        lns += ['- the value is supplied by pointing to something else (open that listing first, then search using its value): {}'.format(req) for req in reading.get('references', None) or []]
        if not reading.get('price_cap', None) is None:
            lns.append('- price cap (total): {} {}'.format(reading.get('price_cap', None), reading.get('currency', None) or '').rstrip())
        if reading.get('in_stock_only', None):
            lns.append("- only items in stock / on hand")
        if not 'none' != self.priority():
            lns.append("- the request gave no rule for picking the cheapest or best: any item that satisfies all requirements and fits the budget is acceptable, so don't use up steps searching for the lowest price (unless the request itself states otherwise)")
        else:
            detail_note = '' if not reading.get('priority_detail', None) else ' ({})'.format(reading.get('priority_detail', None))
            lns.append('- choose the {}{} among all fitting items'.format(self.priority(), detail_note))
        lns += ['- desirable but NOT mandatory (use only to break ties): {}'.format(w) for w in reading.get('soft_wants', None) or []]
        lns += ['- has to be found out by asking the shopper first: {}'.format(a) for a in reading.get('ask_first', None) or []]
        lns += ['- rule: {}'.format(req) for req in reading.get('rules', None) or []]
        if not msgs is None:
            lns.append('- messages to the shopper allowed: {}'.format(msgs))
        lns.append('- steps allowed: {}'.format(step_count))
        return "\n".join(lns)

    @staticmethod
    def line_names(needs, lns):
        words = re.findall(r"[a-z]+", " ".join(lns).lower())
        joined = " ".join(words)
        out: dict[str, str] = {}
        for need in needs:
            if not isinstance(need, (str,)) or not ':' in need:
                continue
            head, _, value = need.partition(":")
            tag = '' if not '(' in head else head[head.find('('):].strip(None)
            parts = re.findall(r"[a-z]+", re.split(r"\s*\(", head)[0].lower())
            if not parts or any(3 > len(p) for p in parts) or re.search(r"\b" + r"\s+".join(parts) + r"\b", joined):
                continue
            for i in range(0, len(words) - len(parts) + 1):
                run = words[i: i + len(parts)]
                if parts != run and all(w.startswith(p) for p, w in zip(parts, run)):
                    out[need] = (" ".join(run) + ('' if not tag else ' ' + tag)) + ":" + value
                    break
        return out

    def drop_named(self, old_need, lns):
        try:
            name = str(old_need).split(":", 1)[0]
            tag = re.findall(r"\(([^)]*)\)", name)

            def fits_tag(w):
                return same_prop(w.split(":", 1)[0], name) and (not tag or re.findall(r"\(([^)]*)\)", w.split(":", 1)[0]) in ([], tag))
            stated = [w for w in self.written if fits_tag(w)]
            if not stated:
                fresh = [n for n in self.__dict__.get('fresh_needs', None) or [] if isinstance(n, (str,)) and ":" in n]
                stated = [w for w in self.uncovered if fits_tag(w) and not w in self.written and not w in fresh]
                if not stated:
                    return True
                if any(fits_tag(n) and old_need != n for n in fresh):
                    return True
            words = re.findall(r"[a-z]+", " ".join(lns).lower())
            phrases = {" ".join(words[i: i + n]) for n in (1, 2, 3) for i in range(0, len(words) - n + 1)}
            phrases = {p for p in phrases if 3 <= len(p) and not p in LINK_WORDS}
            if any(same_prop(w.split(":", 1)[0], p) for w in stated for p in phrases):
                return True
            hook = self.__dict__.get('field_hook', None)
            if hook:
                for w in stated:
                    head, _, value = w.partition(":")
                    value = re.sub(r"\((?:stated|revealed[^)]*)\)", " ", value)
                    if ref_text(value) or not ch_alts(value):
                        continue
                    fields = hook(head) or {}
                    others = [r.split(":", 1)[0] for r in self.written + self.uncovered if isinstance(r, (str,)) and ":" in r
                              and not same_prop(r.split(":", 1)[0], name)]
                    for p in phrases:
                        if any(strict_prop(o, p) for o in others):
                            continue
                        if any(strict_prop(key, p) and value_seen(value, " | ".join(vals)) for key, vals in fields.items()):
                            self.wire.log.append(dict(kind='drop_by_field', need=w[:200], phrase=p))
                            return True
            return False
        except (Exception,):
            return True

    def literal_values(self, answer):
        hook = self.__dict__.get('field_hook', None)
        if not hook:
            return
        swaps: dict[str, str] = {}
        for need in answer.get('needs', None) or []:
            if not isinstance(need, (str,)) or not ':' in need:
                continue
            head, _, value = need.partition(":")
            if re.search(r"\d", value) or ref_text(value):
                continue
            bare = " ".join(re.findall(r"[a-z0-9]+", re.sub(CJ_COMPARE, " ", value.lower())))
            if not bare or (bare == " ".join(re.findall(r"[a-z0-9]+", value.lower())) and positive_value(value, head)):
                continue
            prop = re.split('\\s*\\(', head)[0].strip(None)
            for key, vals in (hook(head) or {}).items():
                if not strict_prop(key, prop):
                    continue
                own = next((v for v in vals if " ".join(re.findall(r"[a-z0-9]+", str(v).lower())) == bare), None)
                if not own is None:
                    LITERAL_VALUES.add(bare)
                    swaps[need] = '{}: {}'.format(head.strip(None), str(own).strip(None).lower())
                    break
        if not swaps:
            return
        answer["needs"] = [n if not isinstance(n, (str,)) else swaps.get(n, n) for n in answer.get('needs', None) or []]
        if isinstance(answer.get('requirements_now', None), (list,)):
            answer["requirements_now"] = [r if not isinstance(r, (str,)) else swaps.get(r, r) for r in answer["requirements_now"]]
        self.wire.log.append(dict(kind='literal_value', needs=list(swaps.values())[:4]))

    def pinned(self, prop_name, qtext):
        try:
            refs = [] if not isinstance(self.reading.get('references', None), (list,)) else [str(r) for r in self.reading.get('references', None) or []]
            refs += list(self.__dict__.get('resolved', None) or [])
            tag = re.findall(r"\(([^)]*)\)", str(prop_name))
            pk = prop_key(prop_name)
            for req in self.written:
                if req in self.let_go or not ':' in req:
                    continue
                head, _, value = req.partition(":")
                rtag = re.findall(r"\(([^)]*)\)", head)
                if tag and rtag and not (kind_words(tag[-1]) & kind_words(rtag[-1])):
                    continue
                if not positive_value(value, head) or story_guess(req, self.request, refs):
                    continue
                rk = prop_key(head)
                if same_prop(head, prop_name) or (4 <= len(rk) and pk.endswith(rk)):
                    return True
        except (Exception,):
            return False
        return False

    @staticmethod
    def valued(needs, lns):
        said = " ".join(str(ln) for ln in lns).lower()
        for need in [] if not isinstance(needs, (list,)) else needs:
            head, _, value = str(need).partition(":")
            names = [w for w in prop_words(head) if 3 <= len(w)]
            values = [w for w in re.findall(r"[a-z0-9]+", value.lower()) if not w in LINK_WORDS]
            if names and values and all(w in said for w in names + values):
                return True
        return False

    def quick_hear(self, lines, qtext):
        if not qtext or 1 != len(lines) or self.items():
            return None
        line = str(lines[0])
        low = line.lower()
        if 240 < len(line):
            return None
        m = re.match(r"\s*(?:which|what)\s+(.+?)\s+(?:would you like|do you need|do you want|would you prefer)\b", str(qtext).strip(None), re.I)
        if not m or re.search(r"\band\b|,", m.group(1)):
            return None
        props = [p.strip(None) for p in re.split(r"\s+or\s+", m.group(1).strip(None)) if p.strip(None)]
        if not 1 <= len(props) <= 2:
            return None
        prop = " or ".join(props)
        guard_low = re.sub(r"\b(?:both |they |it )?sounds? the same to me\b|\bsame as a [a-z]+\b|\bi'?m only after [a-z]+\b|"
                           r"\btwo notes an octave apart\b", " ", low)
        if re.search(r"\d|must|has to|have to|\bneed|firm|fixed|lock|settled|\bonly\b|nothing but|anything but|except|instead|budget|"
                     r"price|cost|stock|sold|chang|actually|\bbut\b|however|though|rather|prefer|insist|requir|won'?t (?:budge|bend|go|take|accept|settle)|stays|"
                     r"set on|tight|match|same|differ|page|listing|cancel|drop|forget|now|also|another|extra|more|less", guard_low):
            return None
        plain = re.search(r"no (?:strong |real )?(?:opinion|preference|pick|side|view)|don'?t mind|do not mind|not bothered|"
                          r"any(?: of them| option| choice| one)? (?:works|goes|would work|is (?:grand|fine|good))|whatever|"
                          r"choose (?:freely|whichever|what you like)|pick (?:anything|whatever|freely)|leave (?:it|that) (?:open|loose|to you)|"
                          r"wide open|open to your judgment|equal for me|never (?:given|gave) (?:it|that) a thought|not something i|"
                          r"haven'?t (?:got|a notion)|nothing (?:special|in my head)|surprise me|happy (?:with whatever|to leave)|"
                          r"carry no opinion|no rule on that|not particular|up to you|trust you|i'?m open on|grand by me|"
                          r"no pull|hinges on (?:that|it)|go with any|any option|not fussed|no lean|sound(?:s)? the same|suits me|"
                          r"either way|all the same to me|no view", low)
        confused = re.search(CONFUSED_REPLY, low)
        if not (plain or confused):
            return None
        self.wire.log.append(dict(kind='quick_hear', prop=prop[:40]))
        return (dict(meaning='no preference on {}'.format(props[0]), needs=[], no_preference=list(props[:1]), dropped=[], new_budget=None, unavailable=False, meta_only=False, requirements_now=None, reply='', instructions=[], allowance=None, references=[]))

    def hear(self, lns, qtext):
        new_lines = [ln for ln in lns if not ln in self.caught]
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
        self.exchanges.append(('' if not qtext else 'asked: {} -> '.format(qtext)) + "shopper: " + " / ".join(new_lines))
        answer = self.quick_hear(new_lines, qtext)
        if answer is None:
            answer = self.wire.consult(ask_text, 2500, LISTENER)
        try:
            if isinstance(answer, (dict,)) and isinstance(answer.get('needs', None), (list,)):
                said = " ".join(new_lines).lower().replace("\u2019", "'")
                firm = re.findall(r"(?:not shift on|holding firm on|nothing else will do except|has got to come as|stays locked to|"
                                  r"only saying yes to|coming out as|landing on|sitting at|has to (?:be|hit|come)|must be)"
                                  r"[^,.;!?]{0,60}?\b(?:being|as|on|at|to|showing|hit|be)\s+([a-z0-9][a-z0-9 ./&'-]{0,40}?)(?=\s*(?:[,.;!?]|$|\s+(?:or|and|then|because|—|-)\b))", said)
                fixed = []
                for need in answer["needs"]:
                    head, sep, value = str(need).partition(":")
                    v = value.strip(None).lower()
                    hit = re.match(r"not\s+(.+)$", v)
                    if sep and hit and any(hit.group(1).strip(None) == f.strip(None) for f in firm):
                        need = '{}: {}'.format(head.strip(None), value.strip(None)[hit.start(1):].strip(None))
                        self.wire.log.append(dict(kind='heard_flip', now=need[:80]))
                    fixed.append(need)
                answer = dict(answer, needs=fixed)
        except (Exception,) as err:
            self.wire.log.append(dict(kind='heard_flip_error', error='{}: {}'.format(type(err).__name__, err)[:200]))
        if not answer:
            if self.wire.mute or self.wire.misses > 0:
                self.caught.difference_update(new_lines)
                self.exchanges.pop()
                self.pending = list(self.__dict__.get('pending', None) or []) + [(list(new_lines), qtext)]
                self.pending_at = monotonic()
            return ""
        try:
            if answer.get('meta_only', None) is True and qtext and not self.valued(answer.get('needs', None), new_lines):
                answer = dict(answer, needs=[], no_preference=[], dropped=[], references=[], requirements_now=None,
                              new_budget=None, unavailable=False, reply="")
                self.meta_last = qtext
                self.wire.log.append(dict(kind='meta_only', line=' / '.join(new_lines)[:200]))
        except (Exception,):
            pass
        try:
            renames = self.line_names(answer.get('needs', None) or [], new_lines)
        except (Exception,):
            renames = {}
        if renames:
            answer["needs"] = [n if not isinstance(n, (str,)) else renames.get(n, n) for n in answer.get('needs', None) or []]
            if isinstance(answer.get('requirements_now', None), (list,)):
                answer["requirements_now"] = [r if not isinstance(r, (str,)) else renames.get(r, r) for r in answer["requirements_now"]]
            self.wire.log.append(dict(kind='need_renamed', names=list(renames.values())[:4]))
        masked = new_lines
        hook = self.__dict__.get('ref_hook', None)
        if hook:
            try:
                masked = hook(answer, new_lines, qtext) or new_lines
            except (Exception,) as err:
                self.wire.log.append(dict(kind='ref_line_error', error='{}: {}'.format(type(err).__name__, err)[:300]))
                masked = new_lines
        remarks = []
        said = (so_far + " " + " ".join(self.exchanges) + " " + (qtext or "")).lower()

        def grounded(req):
            words = [w for w in re.findall(r"[a-z]+", req.lower()) if 3 <= len(w)]
            return not words or any(w in said for w in words)

        try:
            self.literal_values(answer)
        except (Exception,) as err:
            self.wire.log.append(dict(kind='literal_value_error', error='{}: {}'.format(type(err).__name__, err)[:300]))
        self.fresh_needs = [n for n in answer.get('needs', None) or [] if isinstance(n, (str,))]
        now_list = answer.get('requirements_now', None)
        if isinstance(now_list, (list,)) and all(isinstance(req, (str,)) for req in now_list):
            self.now_reqs = [req for req in now_list if req.strip(None) and grounded(req)]
        for new_need in answer.get('needs', None) or []:
            if isinstance(new_need, (str,)) and new_need and not new_need in self.uncovered and grounded(new_need):
                self.uncovered.append(new_need)
                remarks.append('new requirement: {}'.format(new_need))
        try:
            infos = [] if not isinstance(answer.get('references', None), (list,)) else [r for r in answer.get('references', None) or [] if isinstance(r, (dict,))]
            for new_need in answer.get('needs', None) or []:
                if isinstance(new_need, (str,)) and new_need in self.uncovered and (ref_text(new_need.partition(":")[2])
                                                                                  or new_need in (answer.get('ref_needs', None) or [])):
                    self.ref_news = list(self.__dict__.get('ref_news', None) or []) + [dict(need=new_need, infos=infos, line=' '.join(new_lines))]
        except (Exception,):
            pass
        held = {r.split(':', 1)[0].strip(None).lower() for r in (self.written + self.uncovered if not not self.now_reqs is None else self.now_reqs)
                if isinstance(r, (str,)) and ":" in r and not r in self.let_go}
        for prop_name in answer.get('no_preference', None) or []:
            if isinstance(prop_name, (str,)) and prop_name and not prop_name in self.indifferent and not prop_name.strip(None).lower() in held:
                if self.pinned(prop_name, qtext):
                    self.pinned_heard = self.__dict__.get("pinned_heard", 0) + 1
                    continue
                self.indifferent.append(prop_name)
        widened = False
        for old_need in answer.get('dropped', None) or []:
            if isinstance(old_need, (str,)) and old_need and not old_need in self.let_go \
                    and not (is_number(answer.get('new_budget', None))
                             and re.search(r"\b(?:prices?|budget|costs?|spend\w*|cents?|dollars?)\b", old_need.lower())):
                if not self.drop_named(old_need, masked):
                    self.wire.log.append(dict(kind='drop_kept', need=old_need[:200]))
                    continue
                self.let_go.append(old_need)
                remarks.append('dropped: {}'.format(old_need))
                widened = widened or 'none' != self.priority()
        spend_cap = answer.get('new_budget', None)
        if is_number(spend_cap) and spend_cap > 0 and self.spend_cap != spend_cap:
            self.spend_cap = spend_cap
            remarks.append('new budget: {}'.format(spend_cap))
        if answer.get('unavailable', None):
            remarks.append('one item is no longer available; remove it from the cart and do not order it')
        try:
            added = self.add_told(answer.get('instructions', None), 'store' if not not qtext is None else 'shopper')
            if added:
                remarks.append("new instruction: " + "; ".join(added))
            allowance = answer.get('allowance', None)
            if isinstance(allowance, (dict,)) and any(is_number(v) and not isinstance(v, (bool,)) for v in allowance.values()):
                self.allowance_news = {k: v for k, v in allowance.items() if is_number(v) and not isinstance(v, (bool,))}
                remarks.append("new allowance: " + json.dumps(self.allowance_news))
        except (Exception,) as err:
            self.wire.log.append(dict(kind='told_error', error='{}: {}'.format(type(err).__name__, err)[:300]))
        if remarks:
            self.version = self.version + 1
            if widened:
                self.widened_at = self.version
                self.widened_tick = self.wire.tick
        if not self.now_reqs is None:
            for need in self.uncovered:
                prop_name = need.split(':', 1)[0].strip(None).lower()
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
                               for req in self.now_reqs if isinstance(req, (str,))):
                        self.now_reqs.append(need)
                except (Exception,):
                    continue
        if widened:
            if not self.props_left() and 'cheapest' == self.priority() and self.sweepable:
                remarks.append("no property requirement remains: any in-stock listing under the requested category qualifies, regardless of what kind of object it is (an accessory, a part, or a set placed on that shelf all count), and whichever is cheapest across the full shelf is the winner. Results are ordered by relevance or id and not by price, so more portions of the shelf get loaded for you afterward (the pages of its listing, the label's own terms, terms for the types of object appearing in titles); should no '[shelf sweep done]' list come after, go through the shelf on your own using brief queries, taking page one of each: the category label, every kind word in the label by itself, every item name repeating across the titles you have seen, and then the pages of the category listing. Next, turn the cheapest in-stock row in the requested category that fits the budget into the sole line in the cart, verify its stock and price once more, and place the order")
            elif not self.props_left():
                remarks.append('no property condition remains: any in-stock listing within the requested category qualifies, regardless of what kind of object it is (an accessory, a part, or a set placed on that shelf all count), and the lowest price across the entire shelf is the winner. Results are sorted by relevance or id, not price: sweep the shelf using brief queries, taking page one of each - the category label, every kind word in the label by itself, each item name repeated among the titles already seen - plus the pages of the category listing; then place the order for the in-stock listing with the lowest price')
            else:
                remarks.append('a requirement that was dropped WIDENS what counts as acceptable: listings that missed only on that point now qualify, and listings that were never searched could undercut the current choice; run the item kind search again without it, using short queries pairing each remaining required value with the kind word (value placed first), go through every page, and pick the lowest price among everything now acceptable')
        self.wire.log.append((dict(kind='heard', question=not qtext is None, needs=len(answer.get('needs', None) or []), free=len(answer.get('no_preference', None) or []), dropped=len(answer.get('dropped', None) or []), budget=is_number(spend_cap))))
        asked_back = qtext is None and any("?" in ln for ln in new_lines) and not answer.get('no_preference', None)
        self.reply_due = '' if not asked_back else str(answer.get('reply', None) or '').strip(None)
        return "Meaning: " + str(answer.get('meaning', None) or "") + ('' if not remarks else ' (' + '; '.join(remarks) + ')')

class CartBook:

    def __init__(self, the_tool, jschema):
        self.the_tool = the_tool
        self.prop_specs = (jschema or {}).get('properties', None) or {}
        self.put_form: str | None = None
        self.units: dict[str, float] = {}
        self.adds: bool | None = None
        self.used = False
        self.changed_at = 0
        self.viewed_at = -1
        self.last_view: Any = None
        self.pending: dict[str, float] = {}

    def form(self, call_args):
        return as_text({fkey: val for fkey, val in call_args.items() if isinstance(self.prop_specs.get(fkey, None), (dict,)) and "enum" in self.prop_specs[fkey]})

    def line_spec(self):
        for key, pspec in self.prop_specs.items():
            items = None if not (isinstance(pspec, (dict,)) and 'array' == pspec.get('type', None)) else pspec.get('items', None)
            fields = None if not isinstance(items, (dict,)) else (items or {}).get('properties', None)
            if not isinstance(fields, (dict,)):
                continue
            ids = [k for k, s in fields.items() if isinstance(s, (dict,)) and 'string' == s.get('type', None) and not 'enum' in s]
            ids = [k for k in ids if "id" in k.lower()] or ids
            counts = [k for k, s in fields.items() if isinstance(s, (dict,)) and s.get('type', None) in ("integer", "number")]
            if ids and counts:
                return key, ids[0], counts[0]
        return None

    def lines(self, call_args):
        spec = self.line_spec()
        if spec and isinstance(call_args.get(spec[0], None), (list,)):
            out = []
            for row in call_args[spec[0]]:
                if isinstance(row, (dict,)) and isinstance(row.get(spec[1], None), (str,)) and row[spec[1]].strip(None):
                    out.append((row[spec[1]].strip(None), None if not is_number(row.get(spec[2], None)) else float(row[spec[2]])))
            return out
        thing = self.aim_flat(call_args)
        return [] if not thing else [(thing, self.count_arg(call_args))]

    def aim(self, call_args):
        found = self.lines(call_args)
        put = [ln for ln in found if ln[1] is None or 0 < ln[1]]
        return None if not found else (put or found)[0][0]

    def aim_flat(self, call_args):
        for key, piece in call_args.items():
            pspec = {} if not isinstance(self.prop_specs.get(key, None), (dict,)) else self.prop_specs.get(key, None)
            if isinstance(piece, (str,)) and not 'enum' in pspec and piece.strip(None) and not ' ' in piece.strip(None):
                return piece
        return None

    def count_arg(self, call_args):
        if self.line_spec() and isinstance(call_args.get(self.line_spec()[0], None), (list,)):
            found = self.lines(call_args)
            put = [ln for ln in found if ln[1] is None or 0 < ln[1]]
            return None if not found else (put or found)[0][1]
        for key, pspec in self.prop_specs.items():
            if isinstance(pspec, (dict,)) and pspec.get('type', None) in ("integer", "number") and not 'enum' in pspec:
                piece = call_args.get(key, pspec.get('default', None))
                if is_number(piece):
                    return float(piece)
        return None

    def is_put(self, call_args):
        return self.put_form is None or self.form(call_args) == self.put_form

    def held(self, thing):
        return self.units.get(thing, 0.0)

    def booked(self, call_args, obs):
        self.used = True
        if isinstance(obs, (dict,)) and obs.get('error', None):
            return ""
        if self.put_form is None:
            self.put_form = self.form(call_args)
        if self.line_spec() and isinstance(call_args.get(self.line_spec()[0], None), (list,)):
            self.adds = False
            self.units = {}
            for thing, count in self.lines(call_args):
                if count is None or count > 0:
                    self.units[thing] = self.units.get(thing, 0.0) + (count if not count is None else 1.0)
            seen = self.view_units(obs)
            if not seen is None:
                self.learn(seen)
            return self.summary()
        thing = self.aim(call_args)
        if not thing:
            return ""
        count = self.count_arg(call_args)
        count = count if not count is None else 1.0
        prior = self.held(thing)
        if self.is_put(call_args) and count <= 0:
            self.units[thing] = 0.0
        elif self.is_put(call_args):
            if prior > 0 and not self.adds is False:
                self.units[thing] = prior + count
                self.pending = {thing: count}
            else:
                self.units[thing] = 0.0 if not count > 0 else count
        else:
            self.units[thing] = 0.0 if not count > 0 else max(0.0, prior - count)
        if 0 >= self.units.get(thing, 0):
            self.units.pop(thing, None)
        seen = self.view_units(obs)
        if not seen is None and set(seen) <= {thing}:
            self.settle(thing, seen.get(thing, 0.0))
        elif not seen is None:
            self.learn(seen)
        elif (count <= 0 or not self.is_put(call_args)) and self.shows_empty(obs, thing):
            self.units.pop(thing, None)
            self.pending.pop(thing, None)
        return self.summary()

    def shows_empty(self, obs, thing):
        if not isinstance(obs, (dict,)) or not thing in self.units:
            return False
        lists = [v for v in obs.values() if isinstance(v, (list,))]
        nums = [v for v in obs.values() if isinstance(v, (int, float)) and not isinstance(v, (bool,))]
        text_body = as_text(obs)
        return bool(lists) and all(not v for v in lists) and bool(nums) and all(v == 0 for v in nums) \
            and not any(other in text_body for other in self.units if thing != other)

    def view_units(self, obs):
        rows: dict[str, float] = {}
        known = set(self.units) | set(self.pending)

        def visit(d):
            ids = [val for val in d.values() if isinstance(val, (str,)) and (val in known or re.fullmatch(ID_LIKE, val)) and not ' ' in val]
            pairs = [(fkey, val) for fkey, val in d.items() if isinstance(val, (int,)) and not isinstance(val, (bool,)) and 0 < val < 1000
                     and 'price' != str(fkey)]
            if 1 < len(pairs):
                named = [(fkey, val) for fkey, val in pairs if re.search(r"qty|quant|count|units|number|total|held|ordered", str(fkey).lower())
                         and not re.search(r"price|cost|due|sum|fee|charge", str(fkey).lower())]
                pairs = pairs if not 1 == len(named) else named
            if 1 < len(pairs) and ids:
                ident0 = next((i for i in ids if i in known), ids[0])
                held = self.units.get(ident0, self.pending.get(ident0, None))
                same = [(fkey, val) for fkey, val in pairs if not held is None and 1e-09 > abs(val - held)]
                pairs = [min(pairs, key=lambda kv: kv[1])] if not 1 == len(same) else same[:1]
            ints = [val for _, val in pairs]
            if 1 == len(ints) and ids:
                ident = next((i for i in ids if i in known), ids[0])
                rows[ident] = rows.get(ident, 0) + ints[0]
        walk_dicts(obs, visit)
        if not rows or not (set(rows) & (known or set(rows))):
            return None
        return rows

    def settle(self, thing, units):
        expected_if_set = self.pending.pop(thing, None)
        if not expected_if_set is None:
            if 1e-09 > abs(units - expected_if_set) and 1e-09 < abs(self.units.get(thing, 0) - expected_if_set):
                self.adds = False
            elif expected_if_set < units:
                self.adds = True
        if not units > 0:
            self.units.pop(thing, None)
        else:
            self.units[thing] = units

    def learn(self, seen):
        for thing, expected_if_set in self.pending.items():
            if thing in seen:
                if 1e-09 > abs(seen[thing] - expected_if_set) and 1e-09 < abs(self.units.get(thing, 0) - expected_if_set):
                    self.adds = False
                elif seen[thing] > expected_if_set:
                    self.adds = True
        self.pending = {}
        self.units = {fkey: val for fkey, val in seen.items() if val > 0}

    def viewed(self, obs, tick):
        self.last_view = obs
        self.viewed_at = tick
        seen = self.view_units(obs)
        if not seen is None:
            self.learn(seen)
        elif isinstance(obs, (dict, list)) and not self.ids_in(obs):
            self.units = {}

    def ids_in(self, obs):
        text_body = as_text(obs)
        return any(thing in text_body for thing in self.units)

    def summary(self):
        if not self.units:
            return "[cart as counted] empty."
        lns = ", ".join('{} x{}'.format(thing, qty_n if not float(qty_n).is_integer() else int(qty_n)) for thing, qty_n in self.units.items())
        return '[cart as counted] {}.'.format(lns)

    def faults(self, need_n, counts = None):
        if not self.units:
            return []
        if counts and 1 < len(counts):
            held = sorted(int(q) for q in self.units.values())
            if len(self.units) != len(counts):
                return [f"the cart contains {len(self.units)} line(s) ({', '.join((f'{i} x{int(q)}' for i, q in self.units.items()))}), yet the request calls for {len(counts)} products with a single line apiece; each product must get its own line, with no extras"]
            if held != sorted(counts):
                return ['the lines add up to {} units, whereas the products require {} (each product gets one line, set to its quantity)'.format(held, sorted(counts))]
            return []
        collected = []
        if 1 < len(self.units):
            collected.append("the cart holds more than one line (" + ", ".join('{} x{}'.format(i, int(qty_n)) for i, qty_n in self.units.items()) + '); remove all lines except the selected one')
        for thing, qty_n in self.units.items():
            if 1 == len(self.units) and 1e-09 < abs(qty_n - need_n):
                change = 'put in' if not need_n < qty_n else 'take out'
                collected.append('the line {} has {} units, while the request requires {}; use the cart tool to {} {}'.format(thing, int(qty_n), need_n, change, int(abs(qty_n - need_n))))
        return collected

class Errand:
    def __init__(self, pdata):
        environment = pdata["environment"]
        self.wire = Wire(environment["binding"])
        self.pview = environment["policy_view"]
        self.request = str(self.pview.get('query', None) or "")
        self.toolset = [tdef for tdef in self.pview.get('tools', None) or [] if isinstance(tdef, (dict,)) and isinstance(tdef.get('function', None), (dict,))]
        self.shapes = {tdef["function"]["name"]: tdef['function'].get('parameters', None) or dict(type='object', properties={}) for tdef in self.toolset}
        self.step_cap = int(self.pview.get('max_steps', None) or 30)
        self.wire.done = bool(self.pview.get('finished', None))
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

    @property
    def log(self):
        return self.wire.log

    def moves_left(self):
        return self.step_cap - self.wire.tick

    def role(self, call_name):
        return self.role_map.get(call_name, None)

    def ask_limit(self):
        base = ASK_DEFAULT if not not self.msg_cap is None else self.msg_cap
        keep = max(0, int(self.__dict__.get("mv_keep", 0) or 0) - self.msg_reserve())
        return max(0, base - min(base, keep))

    def msg_reserve(self):
        if not self.role("message"):
            return 0
        return 0 if not 1 <= int(self.__dict__.get('mv_keep', 0) or 0) else 1

    def ask_room(self):
        return max(0, self.ask_limit() - self.msg_reserve())

    def step_reserve(self):
        return 0 if not (self.role('message') or self.needs.finish_told()) else FINISH_STEPS

    def apply_allowance(self, news):
        before = (self.msg_cap, self.step_cap)
        total, left = news.get('messages_total', None), news.get('messages_left', None)
        if is_number(total):
            self.msg_cap = max(self.msgs_sent, int(total))
        elif is_number(left):
            self.msg_cap = self.msgs_sent + max(0, int(left))
        total, left = news.get('steps_total', None), news.get('steps_left', None)
        if is_number(total) and int(total) > self.wire.tick:
            self.step_cap = min(self.step_cap, int(total))
        elif is_number(left) and 0 <= int(left):
            self.step_cap = min(self.step_cap, self.wire.tick + int(left))
        if (self.msg_cap, self.step_cap) != before:
            self.asking = self.asking and self.msgs_sent < self.ask_room()
            self.log.append(dict(kind="allowance", news=news, caps=[self.msg_cap, self.step_cap]))

    def call_kind(self, call_name, call_args):
        order, decline = self.role("order"), self.role("decline")
        if not call_name in (order, decline) or not call_name:
            return None
        if decline != order:
            return 'decline' if not order == call_name else 'order'
        for piece in (call_args or {}).values():
            if isinstance(piece, (str,)) and re.fullmatch(r"\s*(?:none|no|nothing|null)\s*", piece.lower()):
                return "decline"
        for piece in (call_args or {}).values():
            if isinstance(piece, (str,)) and 40 >= len(piece):
                if re.search(DECLINE_WORDS, piece.lower()):
                    return "decline"
                if re.search(ORDER_WORDS, piece.lower()):
                    return "order"
        return "order"

    def finishers(self):
        return {self.role("order"), self.role("decline")} - {None}

    def clock_text(self):
        left = max(0.0, self.wire.spare() - LAST_CALL)
        if left > (DEADLINE - LAST_CALL) / 2:
            return ""
        step = int(left // CLOCK_STEP) * CLOCK_STEP
        return 'under {} s left'.format(CLOCK_STEP) if not step > 0 else 'about {} s left'.format(step)

    def cart_arg_note(self):
        tb = self.trolley
        if not tb.the_tool or tb.line_spec():
            return ""
        key = next((k for k, ps in tb.prop_specs.items() if isinstance(ps, (dict,)) and ps.get('type', None) in ("integer", "number")
                    and not 'enum' in ps), None)
        if not key or re.search(r"qty|quant|count|unit|amount|piece|number_of|howmany", key.lower()):
            return ""
        how = {True: "adds that many units to the line", False: "sets the line to that many units"}.get(tb.adds, "puts that many units in")
        return ("\n[cart tool] {}'s {} refers to the NUMBER OF UNITS of that listing and not to where the line sits: a call {} (0 removes the line). To get a line of 3 units, pass {}=3 a single time; every product forms a separate line.".format(tb.the_tool, key, how, key))

    def status_line(self):
        spare = self.moves_left()
        pieces = ['[status] steps used {} of {}, {} left'.format(self.wire.tick, self.step_cap, spare),
                 'messages sent {}'.format(self.msgs_sent) + ('' if not not self.msg_cap is None else ' of {}'.format(self.msg_cap))]
        clock = self.clock_text()
        if clock:
            pieces.append(clock)
        text_body = "; ".join(pieces) + ".\n[requirements now] " + self.needs.text_body() + self.ref_line() + self.product_steps()
        text_body += self.needs.told_text()
        if 'order' == self.__dict__.get('no_fit', None):
            text_body += ('\n[when nothing fits] the present request calls for ' + (('nothing to be bought (a decline)' if not self.no_fit == 'order' else "the nearest listing it describes, which must be ORDERED rather than declined, with the order's own wording noting what is lacking")) + ('' if not self.__dict__.get('no_fit_rule', None) else ': "{}"'.format(self.no_fit_rule[:200])) + ".")
        text_body += self.cart_arg_note()
        if self.__dict__.get('acts', None):
            text_body += "\n[acting on the request's rules] " + " ".join(
                '{}) {} ({}): {}.'.format(n + 1, a['do'], a['when'], a['detail'] or a['rule']) for n, a in enumerate(self.acts, 0))
        try:
            text_body += self.bound_hint()
            text_body += self.dependency_hint()
            text_body += self.unopened_cheap_note()
            text_body += self.rule_status_text()
        except (Exception,):
            pass
        if self.msg_reserve() and self.msgs_sent < self.ask_limit():
            text_body += ' (the questions already planned leave room for one message covering the stated final step: you either write it yourself, or it goes out automatically when the process ends)'
        usable = [tdef["function"]["name"] for tdef in self.offered()]
        if len(usable) < len(self.toolset):
            text_body += "\n[tools usable now] " + ", ".join(usable) + ' (any other tool call gets rejected and costs no step' + (
                '' if not self.asking else '; questions about hidden needs are likewise prepared and dispatched on your behalf') + ")."
        if self.trolley.used:
            text_body += "\n" + self.trolley.summary()
        if spare <= 2:
            text_body += (' LAST STEPS: if the cart contains a candidate for ' + (('the item' if not self.needs.items() else 'every product asked for')) + '; if not, add the strongest candidate to the cart (a confirmed match, or failing that the lowest-priced listing that is in stock and within budget and whose record shows no failing requirement) and place the order. Refuse only if the records make clear that nothing could meet the requirements.' + self.no_fit_say())
        elif spare <= 5 or self.wire.spare() < CLOSING:
            text_body += (" FINISH NOW: fix the cart to " + ('one line' if not self.needs.items() else 'one line per product asked for')
                          + ' (a confirmed match, otherwise the lowest-priced in-stock listing within budget whose record shows no failed requirement; exclude any sold-out line), check its stock once more, then place the order. Refuse only if the records make clear that nothing can match.' + self.no_fit_say())
        elif self.wire.spent_out():
            several = bool(self.needs.items())
            text_body += (' MODEL BUDGET USED: end the exploration; in the coming few steps, add the strongest candidate among '
                          + ('the item asked for' if not several else 'each product requested (a single line per product, with its quantity)')
                          + " in the cart (a verified fit where one exists, else the closest fit the records show), re-check its "
                          "stock and price, and order. A decline is right only when the records show that "
                          + ('the store has' if not several else 'some product has') + " no listing that can meet its requirements.")
        elif self.orders_tried == 0 and spare <= self.order_due() + 7 and spare > self.order_due() and self.narrowing():
            text_body += ' Remaining steps are few: in the next several steps, add your best verified match to the cart and place the order; once {} steps are left, searching and reading details stop, and fixing a rejected order costs roughly 4 steps.'.format(self.order_due())
        elif self.orders_tried == 0 and spare <= self.order_due() and self.narrowing():
            text_body += " The search phase is over (you may only look at details for the cart's item, for listings named by a check, or, when any product still lacks a verified match, make at most 3 more reads or searches combined): add your best verified match to the cart, confirm once more its stock and its current price, then place the order; if an order is rejected, fixing it takes roughly 4 steps."
        return text_body

    def read_request(self):
        tool_rows = ['- {}: {} | arguments: {}'.format(tdef['function']['name'], tdef['function'].get('description', ''), as_text(tdef['function'].get('parameters', None) or {})) for tdef in self.toolset]
        fixed_part = READ_REQUEST + "\n\nTOOLS:\n" + "\n".join(tool_rows)
        var_part = "REQUEST:\n" + self.request
        if not self.pview.get('opening', None) is None:
            var_part += "\n\nOPENING OBSERVATION:\n" + as_text(self.pview.get('opening', None))
        content = [dict(type='text', text=fixed_part, cache_control=dict(type='ephemeral')), dict(type='text', text=var_part)]
        reading: dict[str, Any] = {}
        self.read_content = content
        ladder = READER + LISTENER[:1]
        for turn, mdl in enumerate(ladder * 3, 0):
            if turn >= len(ladder) and (self.wire.misses == 0 or self.wire.call_fail == 'refused' or self.wire.down()
                                        or not self.wire.usable(ladder) or self.wire.spare() < CLOSING + 60):
                break
            msg = self.wire.query_model([dict(role='user', content=content)], [mdl], 5000)
            reading = json_object((msg or {}).get('content', None) or "") or {}
            if isinstance(reading.get('roles', None), (dict,)):
                break
            self.wire.log.append(dict(kind='setup_retry', model=mdl))
        if not isinstance(reading.get('roles', None), (dict,)):
            try:
                reading = {**self.code_reading(), **{key: val for key, val in reading.items() if key != 'roles'}}
            except (Exception,) as err:
                self.wire.log.append(dict(kind='code_reading_error', error='{}: {}'.format(type(err).__name__, err)[:300]))
                reading = {}
            self.read_at = monotonic()
            self.wire.log.append((dict(kind='code_reading', roles=reading.get('roles', None), product=reading.get('product', None), price_cap=reading.get('price_cap', None), items=len(reading.get('items', None) or []))))
        self.use_reading(reading)

    def use_reading(self, reading, again = False):
        if not again:
            self.steps_given = self.step_cap
        old_cart = self.role("cart")
        role_map = {} if not isinstance(reading.get('roles', None), (dict,)) else reading.get('roles', None)
        self.role_map = {fkey: val for fkey, val in role_map.items() if isinstance(val, (str,)) and val in self.shapes}
        try:
            def said(name):
                return str(next((t['function'].get('description', None) for t in self.toolset if t['function'].get('name', None) == name), "") or "").lower()
            cur = self.role_map.get('search', None)
            if cur and not 'keyword' in said(cur):
                alt = next((t["function"]["name"] for t in self.toolset if "keyword" in said(t['function'].get('name', None) or "")
                            and not t['function']['name'] in self.role_map.values()), None)
                if alt:
                    self.role_map["search"] = alt
                    self.__dict__.pop("mv_btool", None)
                    self.log.append({"kind": "search_role_fixed", "from": cur, "to": alt})
        except (Exception,) as err:
            self.log.append(dict(kind='search_role_error', error='{}: {}'.format(type(err).__name__, err)[:200]))
        if again and self.trolley.used and old_cart:
            self.role_map["cart"] = old_cart
        if not (again and self.trolley.used):
            self.trolley = CartBook(self.role("cart"), self.shapes.get(self.role("cart") or "", {}))
        self.needs.sweepable = bool(self.role("search")) and not self.page_arg() is None
        spend_cap = reading.get('message_budget', None)
        if is_number(spend_cap) and spend_cap >= 0:
            self.msg_cap = int(spend_cap)
        self.needs.reading = reading
        try:
            self.needs.add_told(reading.get('instructions', None), "request")
            self.needs.add_told([r for r in reading.get('rules', None) or [] if isinstance(r, (str,))], "request")
        except (Exception,) as err:
            self.wire.log.append(dict(kind='told_seed_error', error='{}: {}'.format(type(err).__name__, err)[:300]))
        try:
            self.wire.scale = 1.0 + 0.3 * min(2, max(0, len(self.needs.items()) - 1))
        except (Exception,):
            self.wire.scale = 1.0
        named = [str(req) for req in reading.get('requirements', None) or [] if req]
        self.needs.written = [req for req in named if ":" in req and req.split(':', 1)[0].strip(None) and req.split(':', 1)[1].strip(None)]
        loose = [req for req in named if not req in self.needs.written]
        if loose:
            reading["soft_wants"] = [str(w) for w in reading.get('soft_wants', None) or [] if w] + loose
            reading["requirements"] = list(self.needs.written)
        try:
            self.story_values(reading)
        except (Exception,) as err:
            self.wire.log.append(dict(kind='story_values_error', error='{}: {}'.format(type(err).__name__, err)[:300]))
        try:
            self.ref_register()
        except (Exception,) as err:
            self.wire.log.append(dict(kind='ref_register_error', error='{}: {}'.format(type(err).__name__, err)[:300]))
        af = reading.get('ask_first', None)
        if isinstance(af, (str,)):
            reading["ask_first"] = [] if not af.strip(None) else [af.strip(None)]
        elif not isinstance(af, (list,)):
            reading["ask_first"] = []
        if not (again and self.needs.version > 0):
            self.needs.spend_cap = reading.get('price_cap', None)
        if reading and self.needs.spend_cap is None and self.role("message"):
            reading["ask_first"] = ['the upper limit of what the shopper is willing to pay, given that the request names no price cap (leave out if the same item appears listed again here)'] + [str(a) for a in reading.get('ask_first', None) or [] if a]
        if again:
            self.step_cap = self.__dict__.get("steps_given", self.step_cap)
        step_count = reading.get('step_budget', None)
        if is_number(step_count) and 0 < step_count < self.step_cap:
            self.step_cap = int(step_count)
        self.wire.scale = self.wire.__dict__.get("scale", 1.0) * max(0.5, min(1.0, 40.0 / max(1, self.step_cap)))
        if again:
            self.needs.version += 1

    def but_repair(self, reading):
        try:
            low = re.sub(r"\s+", " ", self.request.lower().replace("\u2019", "'"))
            for m in re.finditer(r"\banything\s+(?:else\s+)?but\s+(?:the\s+)?([a-z0-9 &'./-]{2,40}?)(?=\s*[,.;!?]|\s+please\b|\s+for\b|$)", low):
                if re.search(r"\b(?:nothing|not|no|never|n't|don't|won't)\b[^.!?]{0,20}$", low[max(0, m.start() - 30):m.start()]):
                    continue
                phrase = m.group(1).strip(None)
                for req in list(self.needs.written):
                    head, _, value = str(req).partition(":")
                    v = value.strip(None).lower()
                    if 2 > len(v) or re.match(r"(?:not|no|anything but|except|never|other than)\b", v) or not phrase.endswith(v):
                        continue
                    new = '{}: not {}'.format(head.strip(None), value.strip(None))
                    self.needs.written = [r if not req == r else new for r in self.needs.written]
                    reading["requirements"] = list(self.needs.written)
                    for row in reading.get('items', None) or []:
                        if isinstance(row, (dict,)) and isinstance(row.get('requirements', None), (list,)):
                            row["requirements"] = [(r if not str(r).partition(':')[2].strip(None).lower() == v else '{}: not {}'.format(str(r).partition(':')[0].strip(None), value.strip(None))) for r in row["requirements"]]
                    self.wire.log.append(dict(kind='but_repair', was=str(req)[:80], now=new[:80]))
        except (Exception,) as err:
            self.wire.log.append(dict(kind='but_repair_error', error='{}: {}'.format(type(err).__name__, err)[:200]))

    def split_merge(self, reading):
        try:
            items = [r for r in reading.get('items', None) or [] if isinstance(r, (dict,)) and str(r.get('product', None) or '').strip(None)]
            if 2 > len(items):
                return
            low = re.sub(r"\s+", " ", self.request.lower().replace("\u2019", "'"))
            names = [str(r['product']).lower().strip(None) for r in items]
            for m in re.finditer(r"[a-z][a-z0-9' -]*(?:\s*(?:,|&|\band\b)\s*[a-z][a-z0-9' -]*){1,4}", low):
                span = m.group(0).strip(None)
                parts = [p.strip(None) for p in re.split(r"\s*(?:,|&|\band\b)\s*", span) if p.strip(None)]
                if 2 > len(parts) or not all(any(p == n or p.endswith(" " + n) for p in parts) for n in names):
                    continue
                if 60 < len(' '.join(parts)) or any(3 < len(p.split()) for p in parts[1:]):
                    continue
                lead = max((n for n in names if parts[0] == n or parts[0].endswith(" " + n)), key=len, default=parts[0].split()[-1])
                name = (lead + span[len(parts[0]):]).strip(None)
                qty = max([r.get('quantity', None) for r in items if isinstance(r.get('quantity', None), (int,)) and not isinstance(r.get('quantity', None), (bool,))] or [1])
                reqs = list(dict.fromkeys(str(x) for r in items for x in (r.get('requirements', None) or []) if isinstance(r.get('requirements', None), (list,))))
                reading["items"] = [dict(product=name, quantity=qty, requirements=reqs, depends_on=None)]
                kinds = [kind_words(n) for n in names]

                def retag(req):
                    head, sep, value = str(req).partition(":")
                    tags = re.findall(r"\(([^)]*)\)", head)
                    if tags and any(kind_words(tags[-1]) == k for k in kinds if k):
                        head = head.replace('({})'.format(tags[-1]), '({})'.format(name))
                    return head + sep + value
                self.needs.written = [retag(r) for r in self.needs.written]
                reading["requirements"] = list(self.needs.written)
                self.wire.log.append(dict(kind='split_merge', names=names[:4], now=name[:60]))
                return
        except (Exception,) as err:
            self.wire.log.append(dict(kind='split_merge_error', error='{}: {}'.format(type(err).__name__, err)[:200]))

    def story_values(self, reading):
        self.but_repair(reading)
        self.split_merge(reading)
        raw = reading.get('hints', None)
        if isinstance(raw, (dict,)):
            raw = ['{}: {}'.format(k, v) if not ':' in str(v) else '{}'.format(v) for k, vals in raw.items()
                   for v in ([vals] if not isinstance(vals, (list,)) else vals) if v]
        hints = [] if not isinstance(raw, (list, tuple)) else [str(h).strip(None) for h in raw or [] if isinstance(h, (str, int, float)) and str(h).strip(None)]
        hints = [one_prop_hint(h) for h in hints]
        own_hints = list(hints)
        refs = [] if not isinstance(reading.get('references', None), (list,)) else [str(r) for r in reading.get('references', None) or []]
        moved = [req for req in self.needs.written if story_guess(req, self.request, refs) and not ref_text(str(req).partition(":")[2])]
        if moved and 2 <= len(self.needs.written) and 2 * len(moved) <= len(self.needs.written):
            self.needs.written = [req for req in self.needs.written if not req in moved]
            reading["requirements"] = list(self.needs.written)

            def base(req):
                head, _, value = str(req).partition(":")
                return re.split('\\s*\\(', head)[0].strip(None).lower(), value.strip(None).lower()
            gone = {base(req) for req in moved}
            for row in reading.get('items', None) or []:
                if isinstance(row, (dict,)) and isinstance(row.get('requirements', None), (list,)):
                    row["requirements"] = [r for r in row["requirements"] if not base(r) in gone]
            hints += ['{}: the story hints at {} (no value is given outright; ask)'.format(req.split(':', 1)[0].strip(None), req.split(':', 1)[1].strip(None)) for req in moved]
            self.wire.log.append(dict(kind='story_values', moved=moved))
        try:
            written = self.written_hints(reading, own_hints)
            hints = [h for h in hints if not h in written]
        except (Exception,) as err:
            self.wire.log.append(dict(kind='written_hint_error', error='{}: {}'.format(type(err).__name__, err)[:300]))
        reading["hints"] = hints[:12]

    def written_hints(self, reading, hints):
        rows = [r for r in reading.get('items', None) or [] if isinstance(r, (dict,))]
        moved: list[str] = []
        for hint in hints:
            head, sep, detail = str(hint).partition(":")
            prop = re.split('\\s*\\(', head)[0].strip(None)
            if not sep or not prop or junk_key(prop):
                continue
            tag = re.findall(r"\(([^)]*)\)", head)
            names = [str(r.get('product', None) or '') for r in rows] or [str(reading.get('product', None) or '')] if not tag else [tag[-1]]
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
            req = '{}: {}'.format(head.strip(None), word)
            self.needs.written.append(req)
            reading["requirements"] = list(self.needs.written)
            for row in rows:
                if isinstance(row.get('requirements', None), (list,)) and (1 == len(rows) or kind_words(row.get('product', None) or "") & mine):
                    row["requirements"].append(req)
                    break
            moved.append(hint)
            self.wire.log.append(dict(kind='written_hint', hint=str(hint)[:200], requirement=req))
        return moved

    def open_hints(self, k = None):
        out, alts = [], []
        for hint in self.needs.reading.get('hints', None) or []:
            prop = str(hint).split(':', 1)[0].strip(None)
            if not prop:
                continue
            j = None if not self.needs.items() else self.product_of(prop)
            if not k is None and not j in (k, None):
                continue
            if self.open_prop(None if not self.needs.items() else j, prop, self.prop_line(j, prop), by_samples=True) \
                    and not self.asked_inside(None if not self.needs.items() else j, prop):
                out.append(str(hint))
                continue
            alt = self.hint_alt(None if not self.needs.items() else j, str(hint), prop)
            if alt:
                alts.append(alt)
        return alts + out

    def prop_line(self, k, prop):
        try:
            memo = self.__dict__.setdefault('menu_memo', {}).get(k, None)
            if not (memo and memo[0] == len(self.archive.items)):
                menu = ''
                if self.needs.items():
                    homes = self.home_shelves()
                    home = set() if not (not k is None and k < len(homes)) else homes[k]
                    shelves = [s for s in self.archive.shelves_read() if s.strip(None).lower() in home]
                    menu = '' if not shelves else self.archive.prop_menu(shelves[0])[0]
                else:
                    own = self.own_shelf()
                    menu = self.archive.prop_menu()[0] if not own else self.archive.prop_menu(own)[0]
                self.menu_memo[k] = (len(self.archive.items), menu)
            else:
                menu = memo[1]
            for line in menu.splitlines():
                if strict_prop(line.strip('- ').split(' (in ')[0].strip(None), prop):
                    return line
        except (Exception,):
            return ""
        return ""

    def field_values(self, k, prop):
        line = self.prop_line(k, prop)
        values = [] if not '):' in line else line.split('):', 1)[1].split('|')
        return {" ".join(re.findall(r"[a-z0-9]+", v.lower())) for v in values} - {""}

    def same_field(self, k, a, b):
        if strict_prop(a, b):
            return True
        try:
            return same_prop(a, b) and bool(self.field_values(k, a) & self.field_values(k, b))
        except (Exception,):
            return False

    def hinted_prop(self, k, prop):
        name = str(prop).split('(')[0].strip(None)
        for hint in self.needs.reading.get('hints', None) or []:
            head = str(hint).split(":", 1)[0]
            if strict_prop(head, name) and (k is None or not self.needs.items() or self.product_of(head) in (k, None)):
                return True
        return False

    def product_reads(self, k):
        try:
            if self.needs.items():
                homes = self.home_shelves()
                home = set() if not (not k is None and k < len(homes)) else homes[k]
                return 0 if not home else sum((self.archive.rich_reads(s) for s in home))
            own = self.own_shelf()
            return self.archive.rich_reads() if not own else self.archive.rich_reads(own)
        except (Exception,):
            return 0

    def plain_hint(self, k, prop, hinted = False):
        try:
            name = re.split('\\s*\\(', str(prop))[0].strip(None)
            if not name or not (hinted or self.hinted_prop(k, name)) or self.prop_line(k, name) or 2 > self.product_reads(k):
                return False
            menu = (self.__dict__.get('menu_memo', {}).get(k, None) or (0, ""))[1]
            return not any(same_prop(line.strip('- ').split(' (in ')[0].strip(None), name) for line in menu.splitlines())
        except (Exception,):
            return False

    def hint_mark(self, hint):
        try:
            prop = str(hint).split(':', 1)[0].strip(None)
            k = None if not self.needs.items() else self.product_of(prop)
            return str(hint) if not self.plain_hint(k, prop, hinted=True) else '{} [no key]'.format(hint)
        except (Exception,):
            return str(hint)

    @staticmethod
    def hint_alts_of(detail):
        return [a.strip(None) for m in re.findall(r"\(or ([^()]+)\)", str(detail)) for a in re.split(r",|\bor\b|/", m) if a.strip(None)]

    @staticmethod
    def name_inside(a, b):
        wa = {w for w in prop_words(a) if 3 <= len(w) and not w in LINK_WORDS}
        wb = {w for w in prop_words(b) if 3 <= len(w) and not w in LINK_WORDS}
        return bool(wa) and bool(wb) and (wb >= wa or wa >= wb)

    def asked_inside(self, k, head):
        try:
            several = bool(self.needs.items())
            for ln in self.questions_log:
                label = ln.split(":", 1)[0].strip("- ")
                if "[not answered]" in label:
                    continue
                said = re.search(r'asked "(.*?)" -> ', ln)
                for part in label.split(";"):
                    j = None if not several else self.product_of(part)
                    if several and j is None and said:
                        j = self.product_of(said.group(1))
                    if (not several or k is None or j in (k, None)) and self.name_inside(part, head):
                        return True
        except (Exception,):
            return False
        return False

    def hint_alt(self, k, hint, head):
        try:
            memo = self.__dict__.setdefault("hint_alts", {})
            several = bool(self.needs.items())
            if not memo.get(hint, None):
                quiet = self.__dict__.get('quiet_asks', None) or []
                if not any((self.same_field(k, q, head) or self.name_inside(q, head)) and (not several or self.product_of(q) in (k, None))
                           for q in quiet):
                    return ""
                for alt in self.hint_alts_of(hint.partition(":")[2]):
                    line = self.prop_line(k, alt) if not (junk_key(alt) or strict_prop(alt, head)) else ''
                    if line:
                        memo[hint] = line.strip('- ').split(' (in ')[0].strip(None)
                        break
            key = memo.get(hint, None) or ""
            if not key or key.lower() in self.__dict__.get("spent_asks", set()) or not self.plan_open(k, key):
                return ""
            tag = re.search(r"\(([^()]*)\)\s*$", hint.partition(":")[0])
            detail = re.sub('\\s*\\(or [^()]+\\)', '', hint.partition(':')[2]).strip(None)
            return '{}{}: {} (the shopper expressed no preference about its first property, {})'.format(key, '' if not tag else ' (' + tag.group(1).strip(None) + ')', detail, head)
        except (Exception,):
            return ""

    def hint_mapped(self, k, prop):
        try:
            name = re.split('\\s*\\(', str(prop))[0].strip(None)
            words = [w for w in prop_words(name) if 3 <= len(w) and not w in LINK_WORDS]
            several = bool(self.needs.items())
            for hint in self.needs.reading.get('hints', None) or []:
                head, _, detail = str(hint).partition(":")
                if several and not k is None and not self.product_of(head) in (k, None):
                    continue
                if any(strict_prop(n, name) for n in [re.split('\\s*\\(', head)[0].strip(None)] + self.hint_alts_of(detail) if n):
                    return True
                low = str(hint).lower()
                if words and all(re.search(r"\b" + re.escape(w), low) for w in words):
                    return True
        except (Exception,):
            return False
        return False

    @staticmethod
    def name_twin(a, b):
        if strict_prop(a, b):
            return False

        def words(name):
            ws = [w for w in prop_words(name) if 3 <= len(w) and not w in LINK_WORDS]
            return ws, (ws if not 1 < len(ws) else ws[:-1])

        def match(x, y):
            return y == x or (4 <= min(len(x), len(y)) and (x.startswith(y) or y.startswith(x)))
        wa, ha = words(a)
        wb, hb = words(b)
        return any(match(x, y) for x in ha for y in wb) or any(match(x, y) for x in wa for y in hb)

    def twin_of(self, k, a, b):
        try:
            return not strict_prop(a, b) and (self.name_twin(a, b) or self.same_field(k, a, b))
        except (Exception,):
            return False

    def twin_key(self, k, prop):
        try:
            several = bool(self.needs.items())
            self.prop_line(k, prop)
            menu = (self.__dict__.get('menu_memo', {}).get(k, None) or (0, ""))[1]
            product = str(self.needs.reading.get('product', None) or '') if not (several and not k is None) else self.needs.items()[k]['product']
            own = kind_tokens(product)
            story = " ".join([self.request.lower()] + [str(h).lower() for h in self.needs.reading.get('hints', None) or []
                                                       if not several or self.product_of(str(h).split(":", 1)[0]) in (k, None)])
            spent = self.__dict__.get("spent_asks", set())
            mine = set(prop_words(prop))
            for line in menu.splitlines():
                key = line.strip('- ').split(' (in ')[0].strip(None)
                if not key or junk_key(key) or key.lower() in spent or not self.twin_of(k, key, prop) or not self.plan_open(k, key):
                    continue
                if self.name_inside(key, prop):
                    continue
                words = [w for w in prop_words(key) if 3 <= len(w) and not w in LINK_WORDS and not w in mine and not kind_in(w, own)]
                if any(re.search(r"\b" + re.escape(w), story) for w in words) or self.hint_mapped(k, key):
                    return key
        except (Exception,):
            return ""
        return ""

    def values_form(self, k, prop, product):
        try:
            line = self.prop_line(k, prop)
            if not '):' in line:
                return ""
            menu = (self.__dict__.get('menu_memo', {}).get(k, None) or (0, ""))[1]

            def norm(v):
                return " ".join(re.findall(r"[a-z0-9]+", v.lower()))
            others = {norm(v) for ln in menu.splitlines() if "):" in ln and ln.strip(None) != line.strip(None)
                      for v in ln.split("):", 1)[1].split("|")}
            picks: list[str] = []
            for v in (v.strip(None) for v in line.split("):", 1)[1].split("|")):
                if v and 30 >= len(v) and re.search(r"[a-z]", v.lower()) and not re.search(r"[,;/]", v) \
                        and not norm(v) in others and not norm(v) in [norm(p) for p in picks]:
                    picks.append(v)
            picks = picks[:3]
            if 2 > len(picks):
                return ""
            return 'For the {}, would you like {} or {}?'.format(product, ', '.join(picks[:-1]), picks[-1])
        except (Exception,):
            return ""

    def distinct_ask(self, prop):
        try:
            several = bool(self.needs.items())
            k = None if not several else self.product_of(prop)
            done = self.__dict__.setdefault("twin_asks", set())
            if (several and k is None) or (-1 if not several else k) in done:
                return None
            product = str(self.needs.reading.get('product', None) or 'item') if not several else self.needs.items()[k]['product']
            target = re.split('\\s*\\(', str(prop))[0].strip(None)
            text = self.values_form(k, target, product)
            twin = self.twin_key(k, prop) if not text else ''
            if twin:
                target = twin
                text = self.values_form(k, twin, product) or 'Which {} would you like for the {}?'.format(twin, product)
            if not text:
                return None
            done.add(-1 if not several else k)
            return (target if not several else '{} ({})'.format(target, product)), text
        except (Exception,):
            return None

    def asked_for(self, k, prop):
        try:
            several = bool(self.needs.items())
            name = re.split('\\s*\\(', str(prop or ''))[0].strip(None)
            if not name:
                return False
            for ln in self.questions_log:
                label = re.sub(r"\s*\[[^\]]*\]", "", ln.split(":", 1)[0].strip("- "))
                said = re.search(r'asked "(.*?)" -> ', ln)
                for part in (p.strip(None) for p in re.split(r";|\band\b", label) if p.strip(None)):
                    j = None if not several else self.product_of(part)
                    if several and j is None and said:
                        j = self.product_of(said.group(1))
                    if (not several or k is None or k == j) and strict_prop(re.split('\\s*\\(', part)[0].strip(None), name):
                        return True
        except (Exception,):
            return False
        return False

    def alt_key(self, k, prop):
        try:
            name = re.split('\\s*\\(', str(prop))[0].strip(None)
            if not name:
                return ""
            several = bool(self.needs.items())
            spent = self.__dict__.get("spent_asks", set())
            for hint in self.needs.reading.get('hints', None) or []:
                head_full, _, detail = str(hint).partition(":")
                head = re.split('\\s*\\(', head_full)[0].strip(None)
                if not head or (several and not self.product_of(head_full) in (k, None)):
                    continue
                if not (self.same_field(k, name, head) or self.name_inside(name, head)):
                    continue
                for alt in self.hint_alts_of(detail):
                    if junk_key(alt) or strict_prop(alt, head) or strict_prop(alt, name) or self.name_inside(alt, name):
                        continue
                    line = self.prop_line(k, alt)
                    key = '' if not line else line.strip('- ').split(' (in ')[0].strip(None)
                    if not key or strict_prop(key, name) or key.lower() in spent or self.asked_for(k, key) or not self.plan_open(k, key):
                        continue
                    return key
        except (Exception,):
            return ""
        return ""

    def collide_alt(self, prop):
        try:
            several = bool(self.needs.items())
            k = None if not several else self.product_of(prop)
            done = self.__dict__.setdefault("twin_asks", set())
            if (several and k is None) or (-1 if not several else k) in done:
                return ""
            key = self.alt_key(k, prop)
            if not key:
                return ""
            done.add(-1 if not several else k)
            return key if not several else '{} ({})'.format(key, self.needs.items()[k]['product'])
        except (Exception,):
            return ""

    def holds_plain(self, text, plain_q):
        frame = {"which", "would", "you", "like", "for", "the"}
        have = {self.stem(w) for w in re.findall(r"[a-z0-9]+", str(text).lower())}
        need = [self.stem(w) for w in re.findall(r"[a-z0-9]+", str(plain_q).lower()) if not w in frame and not w in LINK_WORDS]
        return bool(need) and all(w in have for w in need)

    def alt_untried(self, k):
        try:
            for hint in self.needs.reading.get('hints', None) or []:
                head_full = str(hint).split(":", 1)[0]
                if not self.product_of(head_full) in (k, None):
                    continue
                head = re.split('\\s*\\(', head_full)[0].strip(None)
                if head and self.asked_for(k, head) and self.alt_key(k, head):
                    return True
        except (Exception,):
            return False
        return False

    def quiet_reask(self, prop):
        try:
            several = bool(self.needs.items())
            k = None if not several else self.product_of(prop)
            done = self.__dict__.setdefault("twin_asks", set())
            if (several and k is None) or (-1 if not several else k) in done or not self.hint_mapped(k, prop):
                return ""
            if any(" drew no preference)" in h for h in self.open_hints(k)):
                return ""
            if several:
                sent = {self.product_of(ln.split(":", 1)[0]) for ln in self.questions_log}
                sent |= {self.product_of(said.group(1)) for ln in self.questions_log for said in [re.search(r'asked "(.*?)" -> ', ln)] if said}
                if any(k != j and not j in sent for j in range(0, len(self.needs.items()))):
                    return ""
            twin = self.twin_key(k, prop)
            if not twin:
                return ""
            done.add(-1 if not several else k)
            return twin if not several else '{} ({})'.format(twin, self.needs.items()[k]['product'])
        except (Exception,):
            return ""

    @staticmethod
    def menu_trim(menu, reads):
        if reads < 4 or not menu:
            return menu
        kept = [ln for ln in menu.splitlines() if not re.search(r"\(in [01] listings?\)", ln)]
        return menu if not kept else '\n'.join(kept)

    def first_reads(self):
        try:
            if self.__dict__.get('first_reads_done', None) or (self.batch_tool() and not self.__dict__.get('list_no_fields', None)) \
                    or not self.role("details"):
                return False
            rows = self.needs.items()
            homes = self.home_shelves()
            qtys = [row["quantity"] for row in rows] or [self.needs.quantity()]
            room = self.moves_left() - self.order_due() - 2 * max(0, self.ask_room() - self.msgs_sent) - 8
            wants: list[tuple[int, int, list[str]]] = []
            taken: list[str] = []
            for k in range(0, len(rows) or 1):
                home = set() if not k < len(homes) else homes[k]
                have = self.product_reads(None if not rows else k)
                want_n = 3 if not rows else 2
                if home and want_n > have and room > 0:
                    picks = self.menu_rows(home, want_n - have, taken, qtys[min(k, len(qtys) - 1)])
                    taken += picks
                    if picks:
                        wants.append((have, k, picks))
            queue = [picks[n] for n in range(0, 3) for _, _, picks in sorted(wants) if n < len(picks)][: max(0, room)]
            if not queue:
                return False
            self.first_reads_done = True
            self.open_queue = queue
            self.open_recheck = True
            self.open_say = 'I look at a handful of listings for every product requested to understand their fields before I put questions to the shopper.'
            self.open_purpose = "menu"
            self.log.append(dict(kind='first_reads', ids=queue))
            return True
        except (Exception,) as err:
            self.log.append(dict(kind='first_reads_error', error='{}: {}'.format(type(err).__name__, err)[:300]))
            return False

    def plan_open(self, k, prop):
        try:
            hinted = self.hinted_prop(k, prop)
            return self.open_prop(k, prop, '' if not hinted else self.prop_line(k, prop), by_samples=hinted)
        except (Exception,):
            return self.open_prop(k, prop)

    def settles(self, req, samples = ""):
        value = str(req).partition(":")[2]
        low = value.lower()
        if re.search(r"as shown|same as|the page|sold by|redirect|refer", low) or ref_text(low):
            return True
        if not positive_value(value, str(req).partition(":")[0]):
            return False
        refs = [] if not isinstance(self.needs.reading.get('references', None), (list,)) else [str(r) for r in self.needs.reading.get('references', None) or []]
        refs += list(self.needs.__dict__.get('resolved', None) or [])
        if story_guess(req, self.request + " " + " ".join(self.needs.exchanges), refs):
            return False
        if samples and not re.search(r"\d", value):
            words = [w for w in re.findall(r"[a-z]{3,}", low) if not w in LINK_WORDS]
            pool = set(re.findall(r"[a-z]+", samples))
            if words and not any(re.search(r"\b" + re.escape(w if not 4 < len(w) else w[:-1]), samples)
                                 or (4 <= len(w) and any(4 <= len(s) and edits_within(s, w, 1) for s in pool)) for w in words):
                return False
        return True

    def order_due(self):
        base = max(FIRST_ORDER_BY, min(FIRST_ORDER_DUE, self.step_cap // 4 + 1))
        due = max(base, min(16, self.step_cap // 3))
        if self.__dict__.get('event_seen', None) or (self.__dict__.get('dead', None) and set(self.trolley.units) & self.__dict__["dead"]):
            try:
                due = max(due, 8 + max(1, len(self.needs.items())))
            except (Exception,):
                due = max(due, 9)
        return max(base, min(due, self.step_cap // 2)) + self.step_reserve()

    def narrowing(self):
        return bool(self.role("order") and self.role("cart"))

    def offered(self):
        spare = self.moves_left()
        call_rank, decl, trolley = self.role("order"), self.role("decline"), self.role("cart")
        name_set: set[str] | None = None
        if spare <= 1:
            name_set = {call_rank, decl}
        elif spare <= 3:
            name_set = {call_rank, decl, trolley}
        elif (spare <= self.order_due() and self.orders_tried == 0) or (FIRST_ORDER_BY >= spare and self.refusal_n > 0):
            name_set = {call_rank, decl, trolley, self.role("cart_view"), self.role("view") or self.role("details"), self.stock_tool()}
            if spare > 3 and self.swap_open():
                name_set.add(self.role("details"))
            late = 3 > self.__dict__.get('late_n', 0)
            if late and ((self.unverified() and spare > 5) or (self.named_open() and spare >= min(FIRST_ORDER_BY + 2, self.order_due()))):
                name_set.add(self.role("details"))
            if late and (self.widened_late() or self.__dict__.get('named_pages', None) or (self.unverified() and spare > 5)) and spare > 4:
                name_set.add(self.role("search"))
            if spare >= 7 and self.backup_rows():
                name_set.add(self.role("details"))
        toolset = self.toolset
        if not name_set is None and call_rank and trolley:
            if self.role("message") and spare >= 2 and (self.msg_cap is None or self.msg_cap > self.msgs_sent):
                name_set.add(self.role("message"))
            trimmed = [tdef for tdef in self.toolset if tdef["function"]["name"] in name_set]
            toolset = trimmed or toolset
        return toolset

    def in_reserve(self):
        spare = self.moves_left()
        return bool(self.role("order") and self.role("cart")) and spare > 3 and (
            (spare <= self.order_due() and self.orders_tried == 0) or (FIRST_ORDER_BY >= spare and self.refusal_n > 0))

    def unverified(self):
        dead = self.dead_lines()
        live = [k.rpartition("|")[0] for k in self.passed if k.rpartition("|")[2] == str(self.needs.version)
                and not k.rpartition('|')[0] in self.refused_lines and not k.rpartition('|')[0] in dead]
        if not live:
            return True
        if not self.needs.items():
            return False
        try:
            got = {self.shelf_known(i) for i in live} - {""}
            return bool(got) and any(home and not home & got for home in self.home_shelves())
        except (Exception,):
            return False

    def named_open(self):
        return {i for i in self.__dict__.get("named", set()) if i in self.archive.ids} - self.archive.read - self.refused_lines

    def widened_late(self):
        return self.needs.version == self.needs.widened_at and self.wire.tick - self.needs.widened_tick <= self.order_due()

    def begin_batch(self):
        self.late_q = 0
        self.late_ids = set()

    def late_hold(self, call_name, call_args):
        details, search = self.role("details"), self.role("search")
        if not call_name in (details, search) and call_name in self.lookup_tools() and details:
            return self.late_hold(details, call_args)
        if not call_name in (details, search) or not self.in_reserve():
            return ""
        named = {val for val in call_args.values() if isinstance(val, (str,))}
        if details == call_name and named & (set(self.trolley.units) | {k.rpartition("|")[0] for k in self.passed}):
            return ""
        queued = self.__dict__.get("late_q", 0)
        spare = self.moves_left() - queued
        allowed = False
        if 3 > self.__dict__.get('late_n', 0) + queued:
            if not details == call_name:
                allowed = (self.widened_late() or json.dumps(call_args, default=str, sort_keys=True) in self.__dict__.get('named_pages', set()) or (self.unverified() and spare > 5)) and spare > 4
            else:
                allowed = self.unverified() and spare > 5 or (bool(named & self.named_open()) and spare >= min(FIRST_ORDER_BY + 2, self.order_due()))
        if not allowed and details == call_name and 6 <= spare - 1 and 2 > self.__dict__.get('backup_n', 0):
            backups = [i for i in self.backup_rows() if not i in self.__dict__.get('late_ids', set())][: 2 - self.__dict__.get("backup_n", 0)]
            if named & set(backups):
                allowed = True
                self.backup_n = self.__dict__.get("backup_n", 0) + 1
        opened = '' if not (not allowed and details == call_name) else self.swap_open()
        if opened and opened in named and spare > 2:
            allowed = True
            self.swap_done = True
        if allowed:
            self.late_q = queued + 1
            self.__dict__.setdefault("late_ids", set()).update(named)
            return ""
        extra = []
        if details == call_name and 6 <= spare - 1 and 2 > self.__dict__.get('backup_n', 0):
            extra = [i for i in self.backup_rows() if not i in self.__dict__.get('late_ids', set())][: 2 - self.__dict__.get("backup_n", 0)]
        return ('Not run (no step used): the remaining steps are saved for placing the order. From here, details get read only for the item in the cart, for a listing that a check singled out, or, as long as nothing has been verified, for at most 3 listings overall: add the best verified match to the cart, confirm its stock and price once more, and place the order.'
                + ('' if not extra else " You may still open cheaper rows you haven't read, which could support the choice: " + ', '.join(extra) + '.'))

    def backup_rows(self):
        if 'cheapest' != self.needs.priority() or 2 <= self.__dict__.get('backup_n', 0):
            return []
        if not (self.__dict__.get('event_seen', None) or self.dead_lines() or self.named_open()):
            return []
        out: list[tuple[float, str]] = []
        try:
            for k in self.cg_products():
                pool = self.ed_pool(k)
                tops = [e["price"] for e in pool if e["verified"] and e["fit"] and not e['price'] is None]
                if not tops:
                    continue
                out += [(e["price"], e["id"]) for e in pool if not e["read"] and not e["gone"] and not e["over"] and not e["pointer"]
                        and not e["ruled"] and not e["clash"] and not e['price'] is None and e["price"] < min(tops)]
        except (Exception,) as err:
            self.log.append(dict(kind='backup_error', error='{}: {}'.format(type(err).__name__, err)[:300]))
            return []
        return [i for _, i in sorted(out)][:6]

    def field_names(self):
        prop_list, _ = self.archive.prop_menu()
        if not prop_list:
            return ""
        return ("\n\nFIELD NAMES ON THIS STORE'S LISTINGS (spellings differing only in separators are combined; the number of listings read showing each is given):\n"
                + prop_list)

    def check_item(self, thing, aim_text, second = ""):
        if self.wire.spare() < CLOSING + 20:
            return ""
        key = '{}|{}'.format(thing, self.needs.version)
        if key in self.__dict__.get("near_ok", set()):
            return ""
        refs = [] if not isinstance(self.needs.reading.get('references', None), (list,)) else [str(r) for r in self.needs.reading.get('references', None) or [] if r]
        tracked = list(self.__dict__.get('refs', None) or [])
        own = self.own_redirects(thing)
        memo = key + ('' if not (refs or any(('resolved' != r.get('status', None) for r in tracked))) else '|{}'.format(len(self.archive.read))) + ('' if not second else '|second') \
            + ('' if not own else '|own{}'.format(len(own)))
        if memo in self.rulings:
            return self.rulings[memo]
        pointer = self.ref_pointer_note(thing)
        if pointer:
            ruling = 'A separate check reports that {} does NOT satisfy all the requirements: {}. Pick a different listing.'.format(thing, pointer)
            self.rulings[memo] = ruling
            self.log.append(dict(kind='verify', item=thing, result=dict(item=thing, fits=False, problems=['reference chain'])))
            return ruling
        archive = self.archive.about(thing)
        if not archive:
            return ""
        evidence = self.field_view(thing)
        if evidence["fails"]:
            ruling = ('A separate verification reports that {} does NOT satisfy all requirements: '.format(thing) + "; ".join(evidence["fails"][:3])
                      + ". Choose another listing.")
            self.rulings[memo] = ruling
            self.log.append((dict(kind='verify', item=thing, result=dict(item=thing, fits=False, problems=evidence['fails'][:3], by='field evidence'))))
            return ruling
        ask_text = CHECK_ITEM + "\n\nTHE SHOPPER'S REQUEST:\n" + self.request + self.needs.changes()
        ask_text += '\n\nCURRENT REQUIREMENTS (as stated, as revealed, and as changed):\n' + self.needs.text_body()
        if self.needs.items():
            ask_text += ("\n\nThe item that follows is among the requested products (its category tells you which one): evaluate it solely against the requirements and quantity of that product; requirements belonging to other products don't apply to it."
                         + ('' if not 0 < self.trolley.held(thing) else ' It is in the cart with {} units.'.format(int(self.trolley.held(thing)))))
        ref_note = self.ref_note(thing)
        if ref_note:
            ask_text += ref_note
        refs = self.ref_untracked(refs, tracked)
        if refs:
            words = {w for r in refs for w in re.findall(r"[a-z]{4,}", r.lower().split(":", 1)[-1])}
            pointed = [i for i in sorted(self.archive.read) if thing != i and any(
                w in " ".join(self.archive.about(i, 1)).lower() for w in words)][:3]
            if pointed:
                ask_text += ('\n\nRECORDS OF READ LISTINGS THE REQUEST POINTS AT (when a requirement is given by reference, its value comes from the field it names in the listing it refers to):\n' + "\n".join(r for i in pointed for r in self.archive.about(i, 1)))
        ask_text += second
        if own:
            ask_text += '\n\nNOTE (taken from the notes on this very item, as read by the code): ' + " ".join(
                ('this sentence of its notes may send {} to another listing: "{}". If it does (it tells the reader to take {} from, refer to, go by or look to another listing), the grader takes {} from that listing, so this item\'s own {} field does not count for \'{}\', which is then met only when the record of that other listing shows it; a listing the notes name only as another version sold elsewhere, or as what this item is not, changes nothing.'.format(prop, sent[:240], prop, prop, prop, req))
                for req, prop, sent in own)
        ask_text += '\n\nTHE ITEM TO {}: {}\nRECORDS OF THIS ITEM SEEN IN THE STORE (oldest first):\n'.format(aim_text, thing) + "\n".join(archive)
        if not self.trolley.last_view is None:
            ask_text += "\n\nLATEST CART VIEW:\n" + as_text(self.trolley.last_view)[:3000]
        ask_text += self.field_names()
        if evidence["lines"]:
            ask_text += ("\n\nFIELD EVIDENCE (worked out by code from the store's records for the item's category; it describes the way this store phrases every required value and what the item displays):\n" + "\n".join(evidence["lines"]))
        label_lines = self.archive.label_lines()
        if label_lines:
            ask_text += "\n\nLABELS THAT RECUR ON THIS STORE'S LISTINGS (including counts; pick the category label for the item that fits the whole wording of the request, not one next to it):\n" + label_lines
        answer = self.wire.consult(ask_text, 3000, CHECKER)
        self.log.append(dict(kind='verify', item=thing, result=answer))
        if not answer or not answer.get('fits', None) is False:
            self.rulings[memo] = ""
            if answer:
                self.passed.add(key)
                self.__dict__.setdefault("pass_price", {})[thing] = self.archive.price(thing)
                said = sum(1 for p in answer.get('problems', None) or [] if str(p).strip(None).lower().startswith("unknown"))
                try:
                    unseen = self.unstated_n(thing)
                except (Exception,):
                    unseen = 0
                self.__dict__.setdefault("unknown_n", {})[key] = max(said, unseen)
            return ""
        faults = [str(p) for p in answer.get('problems', None) or []][:5]
        ruling = 'A separate verification states that {} does NOT satisfy all the requirements: '.format(thing) + "; ".join(faults) + '. Pick a different fit that has been fully verified, unless you are sure the check itself is mistaken.'
        self.rulings[memo] = ruling
        return ruling

    def ch_rows(self):
        memo = self.__dict__.get('ch_rows_memo', None)
        if memo and memo[0] == len(self.archive.items):
            return memo[1]
        rows = self.code_rows()
        self.ch_rows_memo = (len(self.archive.items), rows)
        return rows

    def store_fields(self, label = ""):
        try:
            rows = self.ch_rows()
            shelves: set[str] = set()
            homes = self.home_shelves()
            if not self.needs.items():
                own = self.own_shelf()
                shelves = (set() if not homes else set(homes[0])) if not own else {own}
            else:
                k = self.product_of(str(label))
                shelves = set() if not (not k is None and k < len(homes)) else set(homes[k])
            out: dict[str, list[str]] = {}
            for row in rows.values():
                if not row["fields"] or (shelves and not row['shelf'] in shelves):
                    continue
                for key, val in row["fields"].items():
                    if str(val).strip(None) and not junk_key(key):
                        out.setdefault(str(key), []).append(str(val))
            return out
        except (Exception,):
            return {}

    def ch_redirected(self, thing, prop):
        own = getattr(self, "own_redirects", None)
        if callable(own):
            try:
                return any(any(same_prop(str(x).partition(":")[0], prop) for x in ([got] if not isinstance(got, (list, tuple)) else got)[:2])
                           for got in own(thing) or [])
            except (Exception,):
                return True
        words = [w for w in prop_words(prop) if 3 <= len(w)]
        if not words:
            return False
        named = r"\b" + r"\w*[\s_-]+".join(re.escape(w) for w in words)
        for body in self.archive.about(thing, 8):
            for sentence in re.split(r"(?<=[.!?;])\s+", body.lower()):
                if re.search(named, sentence) and re.search(
                        r"\b(?:refer|taken|take it|go by|goes by|look to|look at|comes? from|read from)\b", sentence):
                    return True
        return False

    def field_view(self, thing):
        key = (thing, len(self.archive.items), self.needs.version, "|".join(self.needs.props_left()))
        memo = self.__dict__.setdefault("ch_view_memo", {})
        if not key in memo:
            try:
                memo[key] = self.field_facts(thing)
            except (Exception,) as err:
                self.log.append(dict(kind='field_facts_error', error='{}: {}'.format(type(err).__name__, err)[:300]))
                memo[key] = dict(lines=[], rules=[], fails=[])
        said = (self.__dict__.get('near_said', None) or {}).get('{}|{}'.format(thing, self.needs.version), None)
        if not said:
            return memo[key]
        try:
            view = memo[key]
            props = [p for p, _, _ in said]
            lines = []
            for ln in view["lines"]:
                head = re.match(r"- (.+?) = '", ln)
                hit = next(((p, v, s) for p, v, s in said if head and (same_prop(head.group(1), p) or strict_prop(head.group(1), p))), None)
                lines.append(ln + ('' if not hit else " A semantic check verified that its '{}' equals '{}' (identical value, worded differently): the requirement is satisfied.".format(hit[2][:80], hit[1][:60])))
            fails = [f for f in view["fails"] if not any(same_prop(f.split(':', 1)[0].strip(None), p) or strict_prop(f.split(':', 1)[0].strip(None), p)
                                                         for p in props)]
            return dict(lines=lines, rules=list(view['rules']), fails=fails)
        except (Exception,):
            return memo[key]

    def field_facts(self, thing):
        out: dict[str, list[str]] = dict(lines=[], rules=[], fails=[])
        rows = self.ch_rows()
        me = rows.get(thing, None)
        if not me or not me["fields"]:
            return out
        several = bool(self.needs.items())
        shelf = me["shelf"]
        if several and not shelf:
            return out
        k = None if not several else next((n for n, home in enumerate(self.home_shelves(), 0) if shelf in home), None)
        peers = {i: r for i, r in rows.items() if r["fields"] and (not shelf or r["shelf"] == shelf)}
        if 3 > len(peers):
            return out
        carry: dict[str, int] = {}
        table: dict[str, list[tuple[str, str, list[str]]]] = {}
        for i, r in peers.items():
            for key, val in r["fields"].items():
                if str(val).strip(None) and not junk_key(key):
                    carry[key] = carry.get(key, 0) + 1
                    elems = ch_parts(val)
                    elems += [] if not 1 < len(elems) else [str(val).strip(None).lower()]
                    table.setdefault(key, []).extend((i, p, ch_words(p)) for p in elems)

        def clip(text, n = 70):
            text = " ".join(str(text).split())
            return text[:n - 3] + '...' if not len(text) <= n else text
        for req in self.needs.props_left()[:12]:
            head, _, raw = str(req).partition(":")
            value = " ".join(re.sub(r"\((?:stated|revealed[^)]*)\)", " ", raw).split())
            prop = re.split('\\s*\\(', head)[0].strip(None)
            if not prop or not value or ref_text(value) or re.search(r"as shown|same as|the page|sold by|redirect|refer", value.lower()):
                continue
            if re.search(r"\d", value) and re.search(CH_BOUND, value.lower()):
                j = None if not several else self.product_of(head)
                own_w = set(prop_words(prop))
                named_b = [key for key in carry if 2 <= carry[key] and strict_prop(key, prop)]
                kin_b = {w for w in own_w if 3 <= len(w) and not w in LINK_WORDS}
                if prop and kin_b and named_b and not (several and not j is None and k != j) and not any(
                        str(v).strip(None) and (strict_prop(key, prop) or set(prop_words(key)) & kin_b) for key, v in me["fields"].items()):
                    keys_b = " / ".join("'{}'".format(x) for x in named_b[:3])
                    out["lines"].append("- {} = '{}': in this store the field is {}, seen in {} out of {} listings read for this category. This listing lacks that field, so the limit isn't shown as satisfied: it does NOT satisfy it (a number from some other field can't substitute).".format(prop, clip(value), keys_b, sum((1 for r in peers.values() if any((str(r['fields'].get(x, '')).strip(None) for x in named_b)))), len(peers)))
                    out["fails"].append("{}: it lacks a {} field, though other listings of this category in the store include one, so '{}' isn't shown as satisfied".format(prop, named_b[0], clip(value)))
                continue
            if re.fullmatch(r"\s*\d+(?:\.\d+)?\s+[a-z]+\s*", value.lower()):
                continue
            j = None if not several else self.product_of(head)
            if several and not j is None and k != j:
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
                found = [key for key in carry if 2 <= carry[key] and any(ch_match(alts, words) for _, _, words in table.get(key, []))]
                if not found:
                    continue
                top = max(found, key=lambda x: carry[x])
                named = sorted((key for key in carry if strict_prop(key, top)), key=lambda x: -carry[x])
            on = sum(1 for r in peers.values() if any(str(r['fields'].get(key, '')).strip(None) for key in named))
            writers = sorted({i for key in named for i, _, words in table.get(key, []) if thing != i and ch_match(alts, words)})
            alone = {" ".join(words) for key in named for i, _, words in table.get(key, []) if thing != i}
            samples = list(dict.fromkeys(clip(r["fields"][key], 40) for r in peers.values() for key in named
                                         if str(r['fields'].get(key, '')).strip(None)))[:6]
            keys_txt = " / ".join("'{}'".format(x) for x in named[:3])
            tail = "{} additional listing(s) include '{}' in their own text.".format(len(writers), clip(value))
            if not positive:
                bare = " ".join(re.sub(EXCLUDE_WORDS, " ", value.lower()).split())
                bare_alts = ch_alts(bare)
                held = {i for key in named for i, _, words in table.get(key, []) if thing != i and bare_alts and ch_match(bare_alts, words)}
                if bare_alts:
                    tail = ("{} other listing(s) use '{}' in that spot, which is the value it rules out (any listing showing it fails to satisfy it).".format(len(held), clip(bare)))
            base = ("- {} = '{}': this store's field for it is {}".format(prop, clip(value), keys_txt)
                    + (" (there is no field called '{}'; the values of this one spell out the value)".format(prop) if not by_name else '')
                    + ', among {} of the {} listings read in this category; '.format(on, len(peers)) + tail)
            own_kinds = kind_words(value)
            naming = ([i for i, r in rows.items() if thing != i and (not shelf or r['shelf'] == shelf) and (own_kinds <= r['words'])][:3] if not (writers or not own_kinds) else [])
            if not positive:
                out["rules"].append(base)
            elif writers:
                out["rules"].append(base + " So only a value that writes those words meets it (containment and one list element "
                                    "count; a synonym or related word does not)." + ((" With no field named for the property, a listing whose title writes the value meets it too, unless a field of its own that shares the property's name says otherwise." if not by_name else '')))
            elif naming:
                out["rules"].append(base + " Its values seen: " + " | ".join(samples) + "; listings in this category use '{}' in their title or text ({}), meaning the store has its own term for it: a synonym or related value satisfies it only after the record of such a listing shows that this field expresses it differently.".format(clip(value), ', '.join(naming)))
            elif CH_SYNONYM_MIN > on:
                out["rules"].append(base + " Its values seen: " + " | ".join(samples) + "; no listing states it, yet too few of the listings read include this field to say whether this store itself uses '{}': a value with the same meaning hasn't been shown to satisfy it so far.".format(clip(value)))
            else:
                out["rules"].append(base + " Its values seen: " + " | ".join(samples) + '; no listing states it, so a value with the same meaning might satisfy it.')
            mine = [(key, str(me["fields"][key])) for key in named if str(me['fields'].get(key, '')).strip(None)]
            shown = "; ".join("{}: '{}'".format(key, clip(v, 90)) for key, v in mine)
            own_words = prop_words(prop)
            qualified = [(key, str(v)) for key, v in me["fields"].items() if own_words and not key in named and str(v).strip(None)
                         and not junk_key(key) and len(prop_words(key)) == len(own_words) + 1 and prop_words(key)[1:] == own_words]
            qual_writes = [(key, v) for key, v in qualified if ch_writes(value, v)]
            qual_near = bool(qual_writes) or any(ch_touches(value, v) for _, v in qualified)
            qual_note = (('' if not qual_writes else " The {} '{}' expresses it (a qualified key for the same property: a field that carries the property's name plus one extra word at the front is that property's field under a longer name), so the requirement is satisfied.".format(qual_writes[0][0], clip(qual_writes[0][1], 60))))
            if self.ch_redirected(thing, prop):
                out["lines"].append(base + ' The notes of this listing may redirect this property to a different listing, so its own field might not be decisive.')
                continue
            if not positive:
                out["lines"].append(base + " This listing: " + (shown or "no such field") + " (this is an exclusion: the listing can't contain the excluded value).")
                continue
            kin_words = {w for w in prop_words(prop) if 3 <= len(w) and not w in LINK_WORDS}
            kin = [key for key, v in me["fields"].items() if str(v).strip(None) and not junk_key(key) and set(prop_words(key)) & kin_words
                   and not ch_writes(value, v)]
            title_says = not by_name and not kin and ch_writes(value, me.get('title', None) or "")
            if not mine and qual_writes:
                out["lines"].append(base + " This listing has no such field." + qual_note)
                continue
            if not mine and title_says:
                out["lines"].append(base + " This listing has no such field and none is named for the property: its title says '{}', so it satisfies it.".format(clip(value)))
                continue
            if not mine:
                pw = {w for w in prop_words(prop) if 3 <= len(w)}
                related = sorted(((key, v) for key, v in me["fields"].items() if not key in named and str(v).strip(None) and not junk_key(key)
                                  and (set(prop_words(key)) & pw or ch_writes(value, v))), key=lambda kv: not ch_writes(value, kv[1]))[:4]
                out["lines"].append(base + " This listing has no such field" + (
                    ('' if not related else ': the fields that could name the property are: ' + '; '.join(("{}: '{}'".format(key, clip(v, 60)) for key, v in related)))) + ". A different name for the SAME property is acceptable; a field belonging to another property (a height in place of a type, a frame material in place of the material) can't substitute.")
                continue
            parts = [p for _, v in mine for p in ch_parts(v)] + [v.strip(None).lower() for _, v in mine if 1 < len(ch_parts(v))]

            def knocked(part):
                sides = ch_sides(part)
                if not sides or not any(ch_match(alts, ch_words(s)) for s in sides):
                    return []
                return [s for s in sides if not ch_match(alts, ch_words(s)) and " ".join(ch_words(s)) in alone]
            clean = [p for p in parts if ch_match(alts, ch_words(p)) and not knocked(p)]
            joined = [(p, knocked(p)) for p in parts if ch_match(alts, ch_words(p)) and knocked(p)]
            if clean:
                out["lines"].append(base + " For this listing: {} - it states '{}', so it satisfies it.".format(shown, clip(value)))
            elif qual_writes:
                out["lines"].append(base + ' This listing: {}.'.format(shown) + qual_note)
            elif title_says:
                out["lines"].append(base + (" This listing: {} - with no field named for the property, its title writes '{}': it meets it.".format(shown, clip(value))))
            elif joined:
                part, others = joined[0]
                out["lines"].append(base + (" This listing: {} - '{}' joins '{}' with '{}', which other listings carry alone in this field: ".format(shown, clip(part), clip(value), others[0])) + (('it does NOT meet it.' if not qual_writes else 'this field does not meet it.' + qual_note)))
                if by_name and not qual_near:
                    out["fails"].append("{}: the {} it has is '{}', which combines {} and {} (this store also writes that value by itself in that spot), so it does not count as {}".format(prop, named[0], clip(part), clip(value), others[0], clip(value)))
            elif writers:
                out["lines"].append(base + " This item: {} - it lacks '{}', a value that other items in this category give in this field: ".format(shown, clip(value)) + (('it does NOT satisfy it (a synonym or related term is not enough when the store enters the value itself).' if not qual_writes else 'this field does not meet it.' + qual_note)))
                if by_name and not qual_near and not any(ch_touches(value, v) for _, v in mine):
                    out["fails"].append("{}: the {} reads '{}', yet on other listings this store enters '{}' itself in that field".format(prop, named[0], clip(mine[0][1], 60), clip(value)))
            elif naming:
                out["lines"].append(base + " Regarding this listing: {} - its field does not contain the term '{}'. So far none of the listings that were read use it in this field (the values seen there: ".format(shown, clip(value)) + " | ".join(samples) + "), yet other listings in this category mention '{}' in their title or description ({}): the store uses its own term for it, so the different value on this listing is not taken as meeting it.".format(clip(value), ', '.join(naming)))
            elif CH_SYNONYM_MIN > on:
                out["lines"].append(base + " Regarding this listing: {}. Among the read listings in this category, none uses '{}' in this field (the values seen there: ".format(shown, clip(value)) + " | ".join(samples) + "), yet just {} listing(s) read include it: that is too few to say whether this store itself uses '{}', so a value with the same meaning is not yet counted as satisfying it.".format(on, clip(value)))
            else:
                out["lines"].append(base + " Regarding this listing: {}. Among the read listings in this category, none uses '{}' in this field (the values seen there: ".format(shown, clip(value)) + " | ".join(samples) + '), meaning a value with the same meaning could satisfy it.')
        return out

    def rank_check(self, thing):
        priority = self.needs.priority()
        key = '{}|{}|{}'.format(thing, self.needs.version, self.refusal_n)
        if priority == 'none' or key in self.rank_checked or self.wire.spare() < CLOSING + 30 or 4 > self.moves_left():
            return ""
        self.rank_checked.add(key)
        detail_note = self.needs.reading.get('priority_detail', None) or ""
        what = ('best' if not detail_note else 'best ({})'.format(detail_note)) if not priority == 'cheapest' else 'cheapest (lowest price)'
        ask_text = RANK_CHECK.replace("PRIORITY", what) + "\n\nTHE SHOPPER'S REQUEST:\n" + self.request + self.needs.changes()
        ask_text += "\n\nCURRENT REQUIREMENTS:\n" + self.needs.text_body()
        ask_text += self.ref_note()
        if self.needs.items():
            ask_text += ('\n\nIf the request covers multiple products, the chosen item is challenged only by listings from the SAME product (category); listings belonging to the remaining products can never outrank it.')
        ask_text += ('\n\nTHE CHOSEN ITEM: {}\n\nLISTINGS SEEN (a single card per item; DETAILS = the complete record was read, ROW = only appeared in search rows, so its fields are not known: a ROW in the requested category (regardless of what object its title mentions) that outranks the selected item, with nothing in its title, price or stock excluding it, MAY fit and is listed by name'.format(thing) + (('' if not priority == 'cheapest' else '; the display includes only listings with a lower price than the selected one')) + "):\n"
                     + "\n".join(self.archive.cards(20000, '' if not priority == 'cheapest' else thing, self.ref_foreign(), {thing})))
        ask_text += self.field_names()
        rules = self.field_view(thing)["rules"]
        if rules:
            ask_text += ("\n\nFIELD EVIDENCE (worked out by code from the store's records for the selected item's category: the way this store phrases every required value):\n" + "\n".join(rules))
        answer = self.wire.consult(ask_text, 2500, CHECKER)
        self.log.append(dict(kind='rank', result=answer))
        dead = self.dead_lines()
        better = [str(x) for x in (answer or {}).get('better', None) or [] if isinstance(x, (str, int)) and str(x) != thing
                  and not str(x) in self.refused_lines and not str(x) in dead][:4]
        better = [x for x in better if not self.field_view(x)["fails"]]
        self.rank_better = list(better)
        self.__dict__.setdefault("named", set()).update(better)
        if not better:
            return ""
        return ('A ranking check suspects that these listings could also be suitable and outrank {} on the {}: '.format(thing, what) + ", ".join(better)
                + ('' if not answer.get('reason', None) else ' ({})'.format(answer.get('reason', None)))
                + '. Look into them (details, stock, price) prior to ordering; should every one fall short, go ahead and order the current pick.')

    def review_decline(self, early = False):
        reread = len(self.archive.read) > self.__dict__.get("audit_reads", 10 ** 9) and 2 > self.__dict__.get('audits', 0)
        if (self.audit_done and not reread) or self.wire.spare() < CLOSING + 30:
            return ""
        self.audit_done = True
        if not early:
            self.audits = self.__dict__.get("audits", 0) + 1
        cards = self.archive.cards(16000, "", self.ref_foreign(), (() if not not self.__dict__.get('audit_ids', None) is None else set(self.archive.read) - set(self.__dict__.get('audit_ids', None) or ())))
        self.audit_reads = len(self.archive.read)
        if not cards:
            return ""
        before = self.__dict__.get('audit_ids', None)
        fresh = [] if not not before is None else sorted(self.archive.read - before)
        self.audit_ids = set(self.archive.read)
        ask_text = BEFORE_DECLINE + "\n\nTHE SHOPPER'S REQUEST:\n" + self.request + self.needs.changes()
        ask_text += "\n\nCURRENT REQUIREMENTS:\n" + self.needs.text_body()
        ask_text += self.ref_note()
        ask_text += '\n\nLISTINGS SEEN (a single card per item; DETAILS = the complete record was read, ROW = only appeared in search rows):\n' + "\n".join(cards)
        if fresh:
            ask_text += ('\n\nOPENED SINCE THE LAST CHECK (the DETAILS cards they have above are newly added: evaluate each one first, going field by field, using the reading rules): ' + ", ".join(fresh))
        ask_text += self.field_names()
        answer = self.wire.consult(ask_text, 6000, AUDITOR)
        self.log.append(dict(kind='audit', result=answer))
        conforms = [str(x) for x in (answer or {}).get('possible_fits', None) or [] if isinstance(x, (str, int))][:5]
        conforms = [x for x in conforms if not self.field_view(x)["fails"]]
        self.audit_fits = conforms
        unopened = [str(x) for x in (answer or {}).get('unopened', None) or [] if isinstance(x, (str, int))
                    and not str(x) in self.archive.read and str(x) in self.archive.ids][:6]
        shelves, cap, qty, hit_shelf = self.shelf_labels(), self.needs.spend_cap, self.needs.quantity(), self.__dict__.get("hit_shelf", {})
        hits = [i for i in sorted(self.__dict__.get("all_hits", []), key=lambda i: -self.__dict__.get("hit_score", {}).get(i, 0)) if not i in self.archive.read
                and not (shelves and hit_shelf.get(i, None) and not hit_shelf.get(i, None) in shelves)
                and not (is_number(cap) and is_number(self.archive.price(i)) and self.archive.price(i) * qty > cap)]
        try:
            slack = [i for i in self.slack_rows() if not i in self.archive.read]
        except (Exception,):
            slack = []
        unopened = list(dict.fromkeys(slack + hits[:4] + [i for i in unopened if not i in hits]))[:6]
        pages = [] if not 5 < self.moves_left() else self.open_pages(2)
        if pages:
            unopened = unopened[:4]
            self.__dict__.setdefault("named_pages", set()).update(json.dumps(p, default=str, sort_keys=True) for p in pages)
        try:
            combos = [] if not 5 < self.moves_left() else self.combined_queries()
        except (Exception,) as err:
            self.log.append(dict(kind='combo_error', error='{}: {}'.format(type(err).__name__, err)[:300]))
            combos = []
        if combos:
            self.__dict__.setdefault("named_pages", set()).update(json.dumps(p, default=str, sort_keys=True) for p in combos)
        self.__dict__.setdefault("named", set()).update(conforms + unopened)
        named_list = self.__dict__.setdefault("named_list", [])
        named_list += [i for i in conforms + unopened if not i in named_list]
        if not conforms and not unopened and not pages and not combos:
            return ""
        said = []
        if conforms:
            said.append('A separate check judges that these listings could satisfy all the requirements: ' + ", ".join(conforms)
                        + ('' if not answer.get('reason', None) else ' ({})'.format(answer.get('reason', None))) + ".")
        if pages:
            said.append('Each of these searches includes required values and returned more listings than were actually read, so matching items could be on pages not yet viewed: ' + "; ".join('{} {}'.format(self.role('search'), json.dumps(p, ensure_ascii=False)) for p in pages)
                        + '. Start by reading those pages (a single step displays an entire page of rows), then open any rows within budget that match the kind requested.')
        if combos:
            said.append('Each of these searches merges the required values and none has been run yet (if every word is mandatory, a search returns only listings that contain all of them): ' + "; ".join('{} {}'.format(self.role('search'), json.dumps(p, ensure_ascii=False)) for p in combos)
                        + '. Execute them (the search tool remains available for that) and then open any rows within budget that match the kind requested.')
        if unopened:
            said.append('Listings of the requested type in this group appeared only in search result rows and were never opened, so their details have not been verified: ' + ", ".join(unopened) + '. Pull up their details first (you can do several in one reply).')
        return " ".join(said) + ' Check them (details, stock, price) and purchase the correct one; refuse only when every single one really does not qualify.'

    def named_unread(self):
        cap, qty = self.needs.spend_cap, self.needs.quantity()
        shelves, hit_shelf = self.shelf_labels(), self.__dict__.get("hit_shelf", {})
        out = []
        ranks = self.__dict__.get("hit_score", {})
        ranked = sorted(self.__dict__.get("all_hits", []), key=lambda i: -ranks.get(i, 0))
        for ident in dict.fromkeys(ranked + list(self.__dict__.get("named_list", []))):
            cost = self.archive.price(ident)
            if ident in self.archive.read or ident in self.refused_lines or not ident in self.archive.ids \
                    or (is_number(cap) and is_number(cost) and cost * qty > cap) or (shelves and hit_shelf.get(ident, None) and not hit_shelf[ident] in shelves):
                continue
            out.append(ident)
        return out

    def spec_tool(self):
        name = self.__dict__.get("spec_tool_name", 0)
        if name == 0:
            used = {v for v in self.role_map.values() if v} | self.finishers()
            name = None
            for tdef in self.toolset:
                tool = tdef["function"]["name"]
                sch = self.shapes.get(tool, None) or {}
                props = {} if not isinstance(sch.get('properties', None), (dict,)) else sch.get('properties', None)
                req = sch.get('required', None) or []
                if tool in used or 1 != len(props) or 1 != len(req):
                    continue
                arg = props.get(req[0], None) or {}
                if isinstance(arg, (dict,)) and 'string' == arg.get('type', None) and not arg.get('enum', None):
                    name = tool
                    break
            self.spec_tool_name = name
        tried = self.__dict__.get('spec_read', None) or set()
        if name and 2 <= len(tried) and not (tried & self.archive.read):
            self.spec_tool_name = name = None
        return name

    def details_call(self, ident):
        tool = self.role("details")
        for entry in reversed(self.log):
            for act in [] if not 'env' == entry.get('kind', None) else entry.get('actions', None) or []:
                args = None if not act.get('name', None) == tool else act.get('args', None)
                if isinstance(args, (dict,)) and any(isinstance(v, (str,)) and v in self.archive.ids for v in args.values()):
                    return {k: (v if not (isinstance(v, (str,)) and v in self.archive.ids) else ident) for k, v in args.items()}
        jschema = self.shapes.get(tool or '', None) or {}
        props = {} if not isinstance(jschema.get('properties', None), (dict,)) else jschema.get('properties', None)
        texts = [k for k, p in props.items() if isinstance(p, (dict,)) and 'string' == p.get('type', None) and not p.get('enum', None)]
        key = next((k for k in texts if k in (jschema.get('required', None) or [])), None) or (None if not 1 == len(texts) else texts[0])
        return None if not key else {key: ident}

    def open_reads(self, queue, say, batch, reader = None):
        if not batch:
            tool = reader or self.role('details')
            calls = [] if not tool else [args for args in (self.lookup_call(i, tool) for i in queue) if args]
        else:
            tool, arg, most = batch
            calls = [{arg: queue[i:i + most]} for i in range(0, len(queue), most)]
        proposals = [dict(id='open_{}_{}'.format(n, self.wire.tick), type='function', function=dict(name=tool, arguments=json.dumps(args)))
                     for n, args in enumerate(calls, 0)]
        if not proposals:
            return []
        self.chat.append(dict(role='assistant', content=say, tool_calls=proposals))
        before = len(self.archive.items)
        self.covering = True
        try:
            responses = self.dispatch(proposals, scripted=True)
        finally:
            self.covering = False
        self.last_reads = self.archive.items[before:]
        if batch and responses and not self.__dict__.get('list_no_fields', None):
            facts: list[int] = []
            for rec in self.last_reads:
                walk_dicts(rec, lambda d: None if not any((isinstance(v, (dict,)) and 2 <= len(v) for v in d.values())) else facts.append(1))
            if not facts:
                self.list_no_fields = True
        return responses

    def open_named(self):
        queue = self.__dict__.pop("open_queue", None)
        put_read = False
        if not queue and self.__dict__.get('put_queue', None) and not self.wire.done and not self.wire.broke:
            queue, put_read = self.__dict__.pop("put_queue"), True
            self.open_recheck, self.open_purpose, self.open_tool = True, "cart", self.__dict__.pop("put_tool", None)
            self.open_say = 'I checked the stock for the item I had just added to the cart, since a listing may become unavailable immediately after being added.'
        if not queue or self.wire.done or self.wire.broke:
            return False
        queue = list(dict.fromkeys(str(i) for i in queue))
        recheck = self.__dict__.pop("open_recheck", False)
        purpose = str(self.__dict__.pop("open_purpose", "") or "")
        floor = self.__dict__.pop("open_floor", None)
        reader = self.__dict__.pop("open_tool", None)
        say = self.__dict__.pop("open_say", "") or (('Prior to refusing, I look at the listings that the checks pointed to.' if not recheck else 'Prior to placing the order, I go over the cart lines once more.'))
        if purpose in ("menu", "quota", "walk") and self.role("order") and self.role("cart") and not self.wire.mute:
            try:
                keep = FIRST_ORDER_BY if not self.orders_tried == 0 else self.order_due()
                room = self.moves_left() - keep - 1
                if room < (len(queue) if not (self.batch_tool() and 2 <= len(queue)) else 1) and (self.trolley.units or not self.unverified()):
                    trimmed = [] if self.batch_tool() and 2 <= len(queue) else queue[:max(0, room)]
                    self.log.append(dict(kind='reserve_skip', purpose=purpose, ids=queue[:6], kept=trimmed))
                    queue = trimmed
                    if not queue:
                        return False
            except (Exception,) as err:
                self.log.append(dict(kind='reserve_skip_error', error='{}: {}'.format(type(err).__name__, err)[:300]))
        if purpose == 'cart':
            stale = set(self.stale_lines())
            queue = [i for i in queue if i in stale or (not i in self.trolley.units and not put_read)]
            if not queue:
                return False
        batch = None if not 2 <= len(queue) else self.batch_tool()
        if batch and purpose == 'cart' and not self.__dict__.get('batch_proven', None):
            batch = None
        if batch and reader and not ((self.__dict__.get('tool_caps', None) or {}).get(batch[0], None) or {}).get('stock', None):
            batch = None
        responses = self.open_reads(queue, say, batch, reader)
        if batch and responses and not self.__dict__.get('batch_proven', None):
            if self.batch_ok(self.__dict__.get('last_reads', None) or [], queue):
                self.batch_proven = True
            else:
                if (self.__dict__.get('id_caps', None) or {}).get(batch[0], 99) >= min(len(queue), batch[2]):
                    self.batch_spec = None
                self.log.append(dict(kind='batch_dropped', tool=batch[0]))
                most = dict(menu=0, quota=2, walk=2).get(purpose, 3)
                keep = (3 if not purpose in ('menu', 'quota') else self.order_due()) if not is_number(floor) else int(floor)
                n = min(most, len(queue), self.moves_left() - keep - 1)
                if n > 0 and not self.wire.done and (reader or self.role("details")):
                    self.post_replies(responses)
                    responses = self.open_reads(queue[:n], say, None, reader)
                    if not responses:
                        return True
        if not responses:
            return False
        note = '' if not (self.wire.spare() > CLOSING + 30 and (not recheck)) else self.review_decline()
        if note and responses:
            cid, text_body = responses[-1]
            responses[-1] = (cid, text_body + "\n[check after opening] " + note)
        self.post_replies(responses)
        return True

    def last_stock(self, ident):
        state: bool | None = None

        def visit(d):
            nonlocal state
            if self.row_mark(d) == ident:
                got = self.stock_state(d)
                if not got is None:
                    state = got
        for rec in self.archive.items:
            walk_dicts(rec, visit)
        return state

    def rank_verify(self, thing):
        try:
            if 'cheapest' != self.needs.priority():
                return ""
            ver = str(self.needs.version)
            cap = self.needs.spend_cap
            done = 0
            for x in self.__dict__.get('rank_better', None) or []:
                if done >= 2 or self.wire.spare() < CLOSING + 30:
                    break
                if not x in self.archive.read or '{}|{}'.format(x, ver) in self.passed or x in self.refused_lines or x in self.dead_lines() \
                        or self.last_stock(x) is False or self.field_view(x)["fails"]:
                    continue
                k = None if not self.needs.items() else self.product_index(x)
                qty = self.needs.quantity() if not (not k is None and k < len(self.needs.items())) else self.needs.items()[k]['quantity']
                cost = self.archive.price(x)
                if not is_number(cost) or (is_number(cap) and is_number(qty) and cost * qty > cap):
                    continue
                done = done + 1
                self.check_item(x, "ORDER")
            return self.cheaper_fit(thing)
        except (Exception,) as err:
            self.log.append(dict(kind='rank_verify_error', error='{}: {}'.format(type(err).__name__, err)[:300]))
            return ""

    def cheaper_fit(self, thing):
        if 'cheapest' != self.needs.priority() or not thing:
            return ""
        mine = self.archive.price(thing)
        if mine is None:
            return ""
        kind = self.shelf_known(thing)
        best: tuple[str, float] | None = None
        dead = self.dead_lines()
        for key in self.passed:
            ident, _, version = key.rpartition("|")
            if version != str(self.needs.version) or thing == ident or ident in self.refused_lines or ident in dead:
                continue
            if self.needs.items() and (not kind or self.shelf_known(ident) != kind):
                continue
            cost = self.archive.price(ident)
            if not cost is None and mine > cost and (best is None or cost < best[1]):
                if self.unknowns(ident) > self.unknowns(thing):
                    continue
                if self.field_view(ident)["fails"]:
                    continue
                best = (ident, cost)
        return '' if not best else best[0]

    def shelf_known(self, thing):
        found = ""

        def visit(d):
            nonlocal found
            if any(isinstance(val, (str,)) and thing == val and "id" in str(key).lower() for key, val in d.items()):
                found = self.shelf_of(d) or found
        for rec in self.archive.items:
            walk_dicts(rec, visit)
        return found or str((self.archive.__dict__.get('label_of', None) or {}).get(thing, None) or '').strip(None).lower()

    def chosen(self):
        if self.trolley.units:
            return list(self.trolley.units)[-1]
        return None

    def home_shelves(self):
        memo = self.__dict__.get('home_memo', None)
        if memo and memo[0] == len(self.archive.items):
            return [set(h) for h in memo[1]]
        homes: list[set[str]] = []
        try:
            rows = self.needs.items() or [dict(product=str(self.needs.reading.get('product', None) or ''))]
            seen: set[str] = set()

            def visit(d):
                shelf = self.shelf_of(d)
                if shelf and any(isinstance(v, (str,)) and v in self.archive.ids and "id" in str(key).lower() for key, v in d.items()):
                    seen.add(shelf)
            for rec in self.archive.items:
                walk_dicts(rec, visit)
            names = [set(kind_tokens(row["product"])) for row in rows]
            for k, own in enumerate(names, 0):
                shared = set().union(*(names[j] for j in range(0, len(names)) if k != j))
                best, top = set(), 0.0
                for shelf in sorted(seen):
                    words = set(kind_tokens(shelf))
                    score = kind_score(own, words, shared)
                    if top < score:
                        best, top = {shelf}, score
                    elif top == score and score > 0:
                        best.add(shelf)
                homes.append(best)
        except (Exception,):
            homes = [set() for _ in (self.needs.items() or [0])]
        self.home_memo = (len(self.archive.items), [set(h) for h in homes])
        return homes

    def own_shelf(self):
        try:
            if self.needs.items():
                return ""
            home = self.home_shelves()[0]
            read = [s.strip(None).lower() for s in self.archive.shelves_read()]
            if not home or all(s in home for s in read):
                return ""
            mine = [s for s in read if s in home]
            return sorted(home)[0] if not mine else mine[0]
        except (Exception,):
            return ""

    def joined_phrase(self, a, b, label):
        low = re.sub(r"\s+", " ", self.request.lower().replace("\u2019", "'"))
        parts = [p for p in re.split(JOINERS, str(label).lower()) if p.strip(None)]
        if 2 == len(parts):
            pa, pb = name_pattern(parts[0]), name_pattern(parts[1])
            if pa and pb:
                hit = re.search(r"\b" + pa + r"\s*(?:\band\b|&|\bor\b|/|\+)\s*" + pb, low)
                if hit:
                    return hit.group(0).strip(None), True
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
                if 3 < len(words) or not re.search(r"\band\b|&|\bor\b|,|/|\+", mid) or re.search(r"\d|\bx\b", mid):
                    continue
                rest = low[:hit.start()] + " | " + low[hit.end():]
                if all(re.search(r"\b" + name_pattern(z), rest) for z in (a, b)):
                    continue
                return hit.group(0).strip(None), False
        return None

    def product_question(self, line, rows):
        hit = re.match(r'-\s*(.*?): asked "(.*?)" -> ', str(line), re.S)
        label, question = (str(line), str(line)) if not hit else (hit.group(1), hit.group(2))
        kinds = [kind_tokens(r["product"]) for r in rows]
        for tag in re.findall(r"\(([^()]*)\)", label):
            words = kind_tokens(tag)
            if words and any(kind_in(w, k) for w in words for k in kinds):
                return True
        low = question.lower().replace("\u2019", "'")
        return any(name_pattern(r["product"]) and re.search(r"\b" + name_pattern(r["product"]), low) for r in rows)

    @staticmethod
    def joined_clash(a, b):
        def pairs(row):
            out = []
            for req in row.get('requirements', None) or []:
                head, _, value = str(req).partition(":")
                out.append((re.split('\\s*\\(', head)[0].strip(None), " ".join(re.findall(r"[a-z0-9]+", value.lower()))))
            return out
        return any(same_prop(pa, pb) and vb != va for pa, va in pairs(a) for pb, vb in pairs(b))

    def merge_joined(self, obs):
        seen = self.__dict__.setdefault("joined_seen", {})

        def visit(d):
            if any(isinstance(v, (str,)) and "id" in str(k).lower() and re.fullmatch(ID_LIKE, v) for k, v in d.items()):
                label = next((val.strip(None) for key, val in d.items() if isinstance(val, (str,)) and 0 < len(val.strip(None)) <= 60
                              and re.search(r"categor|shelf|department|aisle", str(key).lower()) and not 'id' in str(key).lower()), "")
                if label and 2 == len([p for p in re.split(JOINERS, label.lower()) if p.strip(None)]):
                    seen.setdefault(label.lower(), label)
        walk_dicts(obs, visit)
        remarks: list[str] = []
        for _ in range(0, 2):
            rows = self.needs.items()
            if 2 > len(rows) or not seen or self.trolley.used or self.trolley.units:
                break
            if self.needs.uncovered or self.needs.indifferent or any(self.product_question(ln, rows) for ln in self.questions_log):
                break
            found = None
            for low in sorted(seen):
                parts = [p for p in re.split(JOINERS, low) if p.strip(None)]
                sides = [kind_tokens(p) for p in parts]
                if 2 != len(sides) or not sides[0] or not sides[1]:
                    continue
                whole = sides[0] + sides[1]
                for i in range(0, len(rows)):
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

    def merge_items(self, i, j, label, phrase, in_order = True):
        reading = self.needs.reading
        rows = self.needs.items()
        raw = [r for r in reading.get('items', None) or [] if isinstance(r, (dict,)) and str(r.get('product', None) or '').strip(None)]
        if len(raw) != len(rows) or not (0 <= i < j < len(rows)):
            return ""
        ti, tj = kind_tokens(rows[i]["product"]), kind_tokens(rows[j]["product"])
        whole = kind_tokens(label)
        others = [kind_tokens(r["product"]) for k, r in enumerate(rows, 0) if not k in (i, j)]
        left = len(rows) - 1
        tag_now = '' if not left >= 2 else phrase

        def owned(tag):
            ts = kind_tokens(tag)
            return bool(ts) and kinds_within(ts, whole) and any(kind_in(w, ti + tj) for w in ts) \
                and not any(kinds_within(ts, o) and kinds_within(o, ts) for o in others)

        def retag(text):
            if not isinstance(text, (str,)):
                return text
            head, sep, value = text.partition(":")

            def put(hit):
                return hit.group(0) if not owned(hit.group(1)) else '' if not tag_now else ' ({})'.format(tag_now)
            return re.sub(r"\s*\(([^()]*)\)", put, head) + sep + value

        def unique(items):
            out: list[Any] = []
            for x in items:
                if not x in out:
                    out.append(x)
            return out
        reqs: list[str] = []
        for req in list(rows[i]["requirements"]) + list(rows[j]["requirements"]):
            head, _, value = str(req).partition(":")
            plain = re.sub('\\s*\\([^()]*\\)', '', head).strip(None) + ":" + value
            if not plain.lower() in [r.lower() for r in reqs]:
                reqs.append(plain)
        entry = dict(product=phrase, quantity=rows[i]['quantity'], requirements=reqs)
        kept = [r if not i == k else entry for k, r in enumerate(raw, 0) if j != k]
        needs = self.needs
        needs.written = unique([retag(r) for r in needs.written])
        needs.uncovered = unique([retag(r) for r in needs.uncovered])
        needs.let_go = unique([retag(r) for r in needs.let_go])
        needs.indifferent = unique([retag(r) for r in needs.indifferent])
        if not needs.now_reqs is None:
            needs.now_reqs = unique([retag(r) for r in needs.now_reqs])
        needs.resolved = [retag(r) for r in needs.__dict__.get('resolved', None) or []]
        for key in ("requirements", "hints", "references", "soft_wants"):
            if isinstance(reading.get(key, None), (list,)):
                reading[key] = unique([retag(r) for r in reading[key]])
        for tgt in [] if not isinstance(reading.get('reference_targets', None), (list,)) else reading.get('reference_targets', None):
            if isinstance(tgt, (dict,)) and owned(str(tgt.get('product', None) or "")):
                tgt["product"] = phrase
        for ref in self.__dict__.get('refs', None) or []:
            for key in ("req", "head", "now"):
                if isinstance(ref.get(key, None), (str,)):
                    ref[key] = retag(ref[key])
            if owned(str(ref.get('tag', None) or "")):
                ref["tag"] = tag_now
        for key in ("pending_seconds", "open_pair"):
            if isinstance(self.__dict__.get(key, None), (list,)):
                self.__dict__[key] = [retag(p) for p in self.__dict__[key]]
        if not left >= 2:
            reading.pop('items', None)
            reading['product'] = phrase
            reading['quantity'] = rows[i]['quantity']
        else:
            reading['items'] = kept
            reading['product'] = ' and '.join((str(r.get('product', None)) for r in kept))
        needs.version = needs.version + 1
        for key in ("home_memo", "menu_memo", "quota_reads", "walked"):
            self.__dict__.pop(key, None)
        self.wire.scale = (1.0 + 0.3 * min(2, max(0, len(needs.items()) - 1))) * max(0.5, min(1.0, 40.0 / max(1, self.step_cap)))
        self.log.append((dict(kind='merge_joined', label=label, products=[rows[i]['product'], rows[j]['product']], into=phrase, label_words=in_order, products_left=max(1, len(needs.items())))))
        return ("The store lists '{}' as ONE category, and the request wants a single product from it ('{}'), hence the order ".format(label, phrase)
                + ('holds ONE line' if not left >= 2 else 'holds one line for it') + ', rather than a single line per each kind that its name combines. Current requirements: ' + needs.text_body())

    def lookup_key(self, tool):
        jschema = self.shapes.get(tool or '', None) or {}
        props = {} if not isinstance(jschema.get('properties', None), (dict,)) else jschema.get('properties', None)
        needed = [] if not isinstance(jschema.get('required', None), (list,)) else [k for k in jschema.get('required', None) or [] if k in props]
        texts = [k for k, p in props.items() if isinstance(p, (dict,)) and 'string' == p.get('type', None) and not p.get('enum', None)]
        key = next((k for k in texts if k in needed), None) or (None if not 1 == len(texts) else texts[0])
        return None if not (key and all((key == k for k in needed))) else key

    def lookup_tools(self):
        taken = {self.role(r) for r in ("search", "message", "cart", "cart_view", "order", "decline")} - {None}
        return [name for name in self.shapes if not name in taken and self.lookup_key(name)]

    def lookup_call(self, ident, tool):
        if not tool:
            return None
        if tool == self.role("details"):
            return self.details_call(ident)
        for entry in reversed(self.log):
            for act in [] if not 'env' == entry.get('kind', None) else entry.get('actions', None) or []:
                args = None if not act.get('name', None) == tool else act.get('args', None)
                if isinstance(args, (dict,)) and any(isinstance(v, (str,)) and v in self.archive.ids for v in args.values()):
                    return {k: (v if not (isinstance(v, (str,)) and v in self.archive.ids) else ident) for k, v in args.items()}
        key = self.lookup_key(tool)
        return None if not key else {key: ident}

    @staticmethod
    def rec_id(d):
        for exact in (True, False):
            for key, val in d.items():
                name = str(key).lower()
                if (name.endswith('id') if not exact else name == 'id') and isinstance(val, (str,)) and re.fullmatch(ID_LIKE, val):
                    return val
        return None

    @staticmethod
    def asked_ids(call_args):
        args = {} if not isinstance(call_args, (dict,)) else call_args
        asked = {v for v in args.values() if isinstance(v, (str,)) and re.fullmatch(ID_LIKE, v)}
        asked |= {x for v in args.values() if isinstance(v, (list,)) for x in v if isinstance(x, (str,)) and re.fullmatch(ID_LIKE, x)}
        return asked

    def learn_caps(self, call_name, call_args, obs):
        asked = self.asked_ids(call_args)
        name = str(call_name)
        caps = self.__dict__.setdefault("tool_caps", {}).setdefault(name, (dict(calls=0, id=0, stock=0, props=0, notes=0, price=0)))
        caps["calls"] += 1
        hit = dict(id=False, stock=False, props=False, notes=False, price=False)
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

        def visit(d):
            ident = self.rec_id(d)
            if not ident:
                return
            state = self.stock_state(d)
            source = 'row' if not (ident in asked or call_name in carts) else 'lookup'
            if not state is None:
                self.stock_seen = True
                mine = heard.setdefault(ident, {})
                other = mine.get('lookup' if not source == 'lookup' else 'row', None)
                if other and other[0] == epoch and other[1] != state:
                    self.rows_lag = True
                if source == 'row' and mine.get('row', None) and mine["row"][1] != state:
                    self.rows_live = True
                mine[source] = (epoch, state)
                if state is False:
                    if not ident in dead and ident in self.trolley.units and not ident in newly:
                        newly.append(ident)
                    dead.add(ident)
                    dead_by[ident] = source
                elif source == 'lookup' or 'lookup' != dead_by.get(ident, None):
                    dead.discard(ident)
            if not ident in asked:
                if not call_name in carts:
                    row_keys.update(str(k) for k in d)
                    row_text.setdefault(ident, {}).update({str(k): v for k, v in d.items() if isinstance(v, (str,))})
                return
            hit["id"] = True
            self.__dict__.setdefault("read_with", {}).setdefault(name, set()).add(ident)
            self.__dict__.setdefault("text_keys", {}).setdefault(name, set()).update(
                str(k) for k, v in d.items() if v is None or isinstance(v, (str,)))
            hit["stock"] = hit["stock"] or not state is None
            hit["props"] = hit["props"] or any(isinstance(v, (dict,)) and 2 <= sum((1 for x in v.values() if isinstance(x, (str, int, float))))
                                               for v in d.values())
            hit["price"] = hit["price"] or any(is_number(v) and re.search(r"price|cost", str(k).lower()) for k, v in d.items())
            keys.update(str(k) for k in d)
            row = row_text.get(ident, None)
            for k, v in d.items():
                if isinstance(v, (str,)) and 8 <= len(v.split()) and (not str(k) in row_keys if not not row is None else row.get(str(k), None) != v):
                    note_keys.add(str(k))
                    hit["notes"] = True
        if not (isinstance(obs, (dict,)) and obs.get('error', None)):
            walk_dicts(obs, visit)
        for key, got in hit.items():
            caps[key] += int(got)
        return newly

    def stock_tool(self):
        caps = self.__dict__.get('tool_caps', None) or {}
        tools = self.lookup_tools()
        shown = [t for t in tools if 0 < (caps.get(t, None) or {}).get('stock', 0)]
        if shown:
            return max(shown, key=lambda t: (caps[t]["stock"], t == self.role("view"), t == self.role("details"), t))
        guess = self.__dict__.get('role_guess', None)
        if guess is None:
            try:
                guess = self.code_roles()
            except (Exception,):
                guess = {}
            self.role_guess = guess
        for tool in (self.role("view"), guess.get('view', None), self.role("details")):
            got = caps.get(tool or '', None) or {}
            if tool and tool in tools and not (2 <= got.get('id', 0) and 0 == got.get('stock', 0)):
                return tool
        busy = {guess.get(r, None) for r in ("cart", "cart_view", "order", "decline", "message")} - {None}
        return next((t for t in tools if not t in busy and not (caps.get(t, None) or {}).get("id", 0)), None)

    def notes_tool(self):
        caps = self.__dict__.get('tool_caps', None) or {}
        details = self.role("details")
        dkeys = (self.__dict__.get('tool_keys', None) or {}).get(details or "", set())
        nkeys = self.__dict__.get('note_keys', None) or {}
        shown = [t for t in self.lookup_tools() if details != t and set(nkeys.get(t, None) or set()) - set(dkeys)]
        return None if not shown else max(shown, key=lambda t: ((caps.get(t, None) or {}).get('notes', 0), t))

    def dead_lines(self):
        return set(self.__dict__.get('dead', None) or set())

    def stale_lines(self):
        seqs = self.__dict__.get('line_seq', None) or {}
        mark = self.__dict__.get("cart_seq", 0)
        return [i for i in self.trolley.units if seqs.get(i, -1) < mark]

    def dead_swap(self, line):
        dead, ver = self.dead_lines(), str(self.needs.version)
        kind = self.shelf_known(line)
        several = 2 <= len(self.needs.items())
        cap, qty = self.needs.spend_cap, max(1.0, self.trolley.held(line))

        def cost(ident):
            price = self.archive.price(ident)
            return float('inf') if not is_number(price) else float(price)
        rest = sum(cost(i) * self.trolley.held(i) for i in self.trolley.units if line != i and cost(i) != float("inf"))
        fits = []
        for key in self.passed:
            ident, _, version = key.rpartition("|")
            if ver != version or line == ident or ident in dead or ident in self.refused_lines or ident in self.trolley.units:
                continue
            if self.field_view(ident)["fails"]:
                continue
            shelf = self.shelf_known(ident)
            if kind and (not shelf in ('', kind) if not several else kind != shelf):
                continue
            if is_number(cap) and cost(ident) != float("inf") and cost(ident) * qty + rest > cap:
                continue
            fits.append(ident)
        if fits:
            return min(fits, key=lambda i: (cost(i), i)), ""
        if several and not kind:
            return "", ""
        k = None if not several else next((n for n, home in enumerate(self.home_shelves(), 0) if kind and kind in home), None)
        judged = [(prop, plain) for prop, plain in self.code_values(k) if not re.search(
            r"\b(?:" + VALUE_LINKS + r")\b|\d\s*(?:-|to)\s*\d|[+<>]|\b(?:about|around|approx\w*|roughly)\b", plain.lower())]
        failed = {key.split("|")[0] for key, val in self.rulings.items() if val and key.split("|")[1:2] == [ver]}
        pointers = self.ref_pointers()
        rows = []
        for ident, row in self.code_rows().items():
            price = cost(ident) if not is_number(row['price']) else row['price']
            if line == ident or ident in dead or ident in self.refused_lines or ident in self.trolley.units or not row['stock'] is True:
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
            rows.append((1 if not read else 0, price, ident))
        rows.sort()
        low = min((x for x in rows if 1 == x[0]), key=lambda x: (x[1], x[2]), default=None)
        if low and 'cheapest' == self.needs.priority() and all(low[1] < p for r, p, _ in rows if r == 0):
            rows.insert(0, rows.pop(rows.index(low)))
        self.swap_unread = [i for r, _, i in rows[:3] if r == 1]
        self.swap_page = None
        if not any(r == 0 for r, _, _ in rows):
            try:
                self.swap_page = self.dead_page(k)
            except (Exception,):
                self.swap_page = None
        return "", ", ".join('{} ({}{})'.format(i, p if not float(p).is_integer() else int(p), '' if not r == 0 else ', read: its record lists each stated value')
                             for r, p, i in rows[:3])

    def dead_page(self, k):
        found = self.page_arg()
        if not found:
            return None
        walks = [w for w in self.__dict__.get("page_book", {}).values() if self.walk_carries(w["args"], k) and not self.next_page(w) is None]
        if not walks:
            return None
        best = max(walks, key=lambda w: (len(self.walk_carries(w["args"], k)), -len(w["read"])))
        return {**best["args"], found[0]: self.next_page(best)}

    def swap_open(self):
        if self.__dict__.get('swap_done', None):
            return ""
        dead = self.dead_lines()
        return next((i for i in self.__dict__.get('swap_unread', None) or [] if not i in self.archive.read and not i in dead
                     and not i in self.trolley.units), "")

    def dead_note(self, lines):
        parts = []
        self.event_seen = True
        for line in lines:
            self.swap_page, self.swap_unread = None, []
            pick, rows = self.dead_swap(line)
            head = '{} is now OUT OF STOCK (according to its most recent record): any order containing it will fail. Remove it from the cart'.format(line)
            if pick:
                least = '' if not 'cheapest' == self.needs.priority() else ', the cheapest such listing'
                parts.append('{} and add {} (the item check determined it satisfies all requirements currently in effect{}), check its stock and price again, then place the order.'.format(head, pick, least))
            elif rows:
                parts.append('{}; there is no other confirmed match for its product, so look at the lowest-priced in-stock rows in its category that fall within the budget ({}; any row that was read and whose own record is missing or fails a stated requirement is excluded, and a row not yet read must have its details opened first), add the first one that fits, check its stock and price again, then place the order.'.format(head, rows))
            else:
                parts.append('{} and find another listing that fits.'.format(head))
            page = self.__dict__.get('swap_page', None)
            if page and not pick:
                parts.append('Additional rows of the same type: the following unread page of the search headed by the given value, '
                             + json.dumps(page, default=str) + ".")
        return " ".join(parts)

    def refusal_reread(self, obs = None):
        heard = ""
        try:
            heard, gone = self.refusal_gone(obs)
        except (Exception,) as err:
            gone = ""
            self.log.append(dict(kind='refusal_gone_error', error='{}: {}'.format(type(err).__name__, err)[:300]))
        if gone:
            return heard + gone
        return heard + self.refusal_queue()

    def refusal_gone(self, obs):
        lines = [i for i in self.trolley.units if 0 < self.trolley.held(i)]
        if obs is None or 1 != len(lines) or self.wire.mute or self.__dict__.get('covering', None) or 'order' != self.__dict__.get('last_kind', None):
            return "", ""
        texts = self.shopper_lines(self.role("order") or "", obs, True)
        if not texts:
            return "", ""
        needs = self.needs
        before = (needs.version, len(needs.let_go), len(needs.uncovered), str(needs.spend_cap), "|".join(needs.props_left()))
        note = needs.hear(texts, None)
        after = (needs.version, len(needs.let_go), len(needs.uncovered), str(needs.spend_cap), "|".join(needs.props_left()))
        heard = '' if not note else ' ' + note
        if not note or after[0] == before[0] or after[1:] != before[1:]:
            return heard, ""
        line = lines[0]
        self.__dict__.setdefault("dead", set()).add(line)
        self.__dict__.setdefault("dead_by", {})[line] = "lookup"
        self.log.append(dict(kind='refusal_dead', line=line))
        return heard, (' The refusal reports that an item became unavailable while the cart had a single line, so the code uses ' + line
                       + " as sold out (no re-read): " + self.dead_note([line]))

    def refusal_queue(self):
        tool = self.stock_tool()
        lines = list(self.trolley.units)[:4]
        if not lines or not tool or self.wire.mute or self.__dict__.get('covering', None) or self.__dict__.get('open_queue', None) \
                or self.moves_left() < len(lines) + 3 or self.wire.spare() < CLOSING:
            return ""
        self.open_queue = lines
        self.open_recheck, self.open_purpose, self.open_tool = True, "cart", tool
        self.open_say = 'The order got rejected, so I check the stock again on each line in the cart (some line might have sold out).'
        return ' The stock of the cart lines has just been read for you (' + tool + ")."

    def code_judges_all(self):
        items = self.needs.items()
        vals = [plain for k in ([None] if not items else range(0, len(items))) for _, plain in self.code_values(k)]
        plain = [v for v in vals if not re.search(r"\b(?:" + VALUE_LINKS + r")\b|\d\s*(?:-|to)\s*\d|[+<>]|\b(?:about|around|approx\w*|roughly)\b",
                                                  v.lower())]
        resolved = self.differ_done(None)
        judged = sum(1 for req in self.needs.props_left() if any(self.ref_match(ref, req) for ref in resolved))
        return len(plain) + judged >= len(list(self.needs.props_left()))

    def differ_done(self, k):
        rows, left, out = self.needs.items(), self.needs.props_left(), []
        for ref in self.__dict__.get('refs', None) or []:
            if 'differ' != ref.get('relation', None) or 'resolved' != ref.get('status', None) or not str(ref.get('value', None) or '').strip(None) \
                    or not self.ref_live(ref) or not any(self.ref_match(ref, req) for req in left):
                continue
            if rows and not k is None and ref.get('tag', None) and not self.product_of('x ({})'.format(ref['tag'])) in (k, None):
                continue
            out.append(ref)
        return out

    def decline_repair(self):
        several = 2 <= len(self.needs.items())
        early = several and self.refusal_n == 0 and self.moves_left() > self.order_due()
        if not (self.refusal_n > 0 or self.moves_left() <= self.order_due() or early):
            return ""
        if not self.role("cart") or not self.role("order") or 3 > self.moves_left() or self.wire.spare() <= CLOSING \
                or self.wire.mute or self.__dict__.get('covering', None) or self.wire.done:
            return ""
        tried = self.__dict__.setdefault("repair_tried", set())
        mark: Any = self.refusal_n if not early else ('early', self.refusal_n)
        if mark in tried:
            return ""
        tried.add(mark)
        if several:
            try:
                proof = self.ed_proof()
            except (Exception,) as err:
                proof = []
                self.log.append(dict(kind='decline_proof_error', error='{}: {}'.format(type(err).__name__, err)[:300]))
            if proof:
                self.log.append(dict(kind='decline_proof', products=proof))
                return ""
        dead = self.dead_lines()
        stale = set(self.stale_lines())
        alive = {i for i in self.trolley.units if not i in dead and not i in stale}
        saved = set(self.refused_lines)
        done = False
        ver = str(self.needs.version)

        own = self.code_judges_all()

        def fitting(picks):
            fit = self.__dict__.get('code_fit', None) or {}
            return bool(picks) and all('{}|{}'.format(i, ver) in self.passed or (own and i in self.archive.read and fit.get(i, None)) for i, _ in picks)
        try:
            self.refused_lines.difference_update(alive)
            avoid = set(dead)
            seen: dict[str, dict[str, Any]] = {}
            if several:
                seen = {e["id"]: e for k in range(0, len(self.needs.items())) for e in self.ed_pool(k)}
                avoid |= {i for i, e in seen.items() if e["clash"]}
            picks = self.code_picks(avoid)
            checks = rounds = 0
            while several and picks and checks < 3 and rounds < 30 and self.wire.spare() > CLOSING + 30:
                unchecked = [i for i, _ in picks if not fitting([(i, 1)])]
                if not unchecked:
                    break
                bare = [i for i in unchecked if not seen.get(i, None) or not seen[i]["revealed_ok"] or seen[i]["misses"]]
                if bare:
                    avoid.update(bare)
                    avoid.update(i for i, e in seen.items() if not e["read"] and not '{}|{}'.format(i, ver) in self.passed
                                 and (not e["revealed_ok"] or e["misses"]))
                    rounds = rounds + 1
                    picks = self.code_picks(avoid)
                    continue
                checks = checks + 1
                self.check_item(unchecked[0], "ORDER")
                picks = self.code_picks(avoid)
            if not fitting(picks):
                self.log.append(dict(kind='decline_repair', picks=picks, done=False))
                return ""
            if self.moves_left() < len(picks) + 2:
                self.log.append(dict(kind='decline_repair', picks=picks, done=False, why='steps'))
                return ""
            was = self.wire.mute
            self.wire.mute = True
            try:
                bad = self.code_put(picks)
                done = bad == [] and all(i in self.trolley.units for i, _ in picks)
                stale_now = [i for i in self.stale_lines() if any(p == i for p, _ in picks)]
                if done and stale_now and self.moves_left() >= len(stale_now) + 2:
                    _, sold = self.code_recheck(stale_now)
                    done = not sold
            finally:
                self.wire.mute = was
            done = done and bool(self.trolley.units) and not set(self.trolley.units) & self.dead_lines()
            self.log.append(dict(kind='decline_repair', picks=picks, done=done, cart=dict(self.trolley.units)))
        except (Exception,) as err:
            done = False
            self.log.append(dict(kind='decline_repair_error', error='{}: {}'.format(type(err).__name__, err)[:300]))
        finally:
            if not done:
                self.refused_lines.clear()
                self.refused_lines.update(saved)
        if not done:
            return ""
        several = bool(self.needs.items())
        return ('Not run yet (no step used). Declining is correct only if no option can fit, and the records contain a suitable candidate '
                + ('' if not several else 'for every product asked for') + ' (confirmed by the item check, or read with each stated value present in its own record), which is why the code placed ' + ('it' if not several else 'them') + " in the cart and read the stock: "
                + self.trolley.summary() + ' Place the order for this cart now; refuse only if a record you have read shows that one of these lines fails a requirement that currently applies.')

    @staticmethod
    def ed_plain(value):
        return not re.search(r"\b(?:" + VALUE_LINKS + r")\b|\d\s*(?:-|to)\s*\d|[+<>]|\b(?:about|around|approx\w*|roughly)\b",
                             str(value).lower())

    @staticmethod
    def ed_word(value):
        low = str(value).lower()
        return bool(re.search(r"[a-z]{3,}", low)) and not re.search(r"\d", low) and Errand.ed_plain(low) \
            and not re.search(EXCLUDE_WORDS, low) and not ref_text(low)

    def ed_reqs(self, k):
        rows = self.needs.items()
        out = []
        for req in self.needs.props_left():
            head, _, value = str(req).partition(":")
            if rows and not self.product_of(head) in (k, None):
                continue
            plain = re.sub('\\((?:stated|revealed[^)]*)\\)', '', value).strip(None)
            out.append((re.split('\\s*\\(', head)[0].strip(None), plain, not req in self.needs.written))
        return out

    def ed_pool(self, k):
        rows = self.needs.items()
        ver = str(self.needs.version)
        mark = (k, len(self.archive.items), len(self.archive.read), ver, "|".join(self.needs.props_left()), len(self.passed),
                sum(1 for v in self.rulings.values() if v), tuple(sorted(self.dead_lines())), tuple(sorted(self.refused_lines)),
                str(self.needs.spend_cap), len(self.__dict__.get('refs', None) or []))
        memo = self.__dict__.setdefault("ed_memo", {})
        if mark in memo:
            return memo[mark]
        if 64 < len(memo):
            memo.clear()
        homes = self.home_shelves()
        if not rows:
            home = set(self.shelf_labels()) or (set() if not homes else set(homes[0]))
        else:
            home = set() if not (not k is None and k < len(homes)) else set(homes[k])
        out: list[dict[str, Any]] = []
        if home:
            counts = self.needs.counts()
            qty = (1 if not counts else counts[0]) if not (rows and not k is None and (k < len(counts))) else counts[k]
            failed = {key.split("|")[0] for key, val in self.rulings.items() if val and key.split("|")[1:2] == [ver]}
            live = {key.rpartition("|")[0] for key in self.passed if key.rpartition("|")[2] == ver}
            cap = None if not is_number(self.needs.spend_cap) else float(self.needs.spend_cap)
            judged = [(p, v) for p, v in self.code_values(None if not rows else k) if self.ed_plain(v)]
            joins = [(p, parts) for p, v in self.code_values(None if not rows else k) if "+" in v
                     for parts in [[x.strip(None) for x in v.split("+") if x.strip(None)]] if 2 <= len(parts)
                     and all(re.search(r"[a-z]{2,}", x.lower()) for x in parts)]
            shown = [(p, v) for p, v, revealed in self.ed_reqs(None if not rows else k) if revealed]
            pointers, dead = self.ref_pointers(), self.dead_lines()
            for ident, row in self.code_rows().items():
                if not row['shelf'] in home:
                    continue
                read = ident in self.archive.read
                cost = self.archive.price(ident) if not not row['price'] is None else row['price']
                price = None if not is_number(cost) else float(cost)
                gone = ident in dead or row["stock"] is False or ident in self.refused_lines
                over = not cap is None and not price is None and price * qty > cap + 1e-9
                pointer = ident in pointers or self.ref_excluded(row["fields"], None if not rows else k)
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
                out.append((dict(id=ident, price=price, read=read, verified=verified, fit=fit, gone=gone, over=over, pointer=pointer, ruled=ruled, clash=clash, misses=misses, row=row, out=bool(gone or over or pointer or ruled or clash), revealed_ok=all((self.code_shows(row, p, v, set(), True) for p, v in shown)))))
        memo[mark] = out
        return out

    def ed_proof(self):
        out = []
        for k in range(0, len(self.needs.items())):
            reads = [e for e in self.ed_pool(k) if e["read"]]
            if reads and all(e["out"] for e in reads):
                out.append(k)
        return out

    def slack_rows(self):
        rows = self.needs.items()
        cap = self.needs.spend_cap
        if 2 > len(rows) or not is_number(cap):
            return []
        try:
            counts = self.needs.counts()
            floors = []
            for k in range(0, len(rows)):
                prices = [e["price"] for e in self.ed_pool(k) if e["fit"] and not e['price'] is None]
                if not prices:
                    return []
                floors.append(min(prices) * (1 if not k < len(counts) else counts[k]))
            if sum(floors) <= float(cap) + 1e-9:
                return []
            book = list(self.__dict__.get("page_book", {}).values())
            hits = set(self.__dict__.get('all_hits', None) or [])
            pools = []
            for k in range(0, len(rows)):
                slack = (float(cap) - (sum(floors) - floors[k])) / max(1, 1 if not k < len(counts) else counts[k])
                values = [{w for w in re.findall(r"[a-z0-9]{2,}", v.lower()) if not w in LINK_WORDS} for _, v, _ in self.ed_reqs(k)]
                led: dict[str, int] = {i: 1 for i in hits}
                for walk in book:
                    words = [w for text in self.free_text(walk["args"]) for w in re.findall(r"[a-z0-9]{2,}", text.lower())]
                    words = set(words if not self.lead_words else words[:self.lead_words])
                    score = sum(1 for vals in values if vals & words)
                    for i in (walk.get('row_shelf', None) or {}):
                        if score:
                            led[str(i)] = max(led.get(str(i), 0), score)
                pool = sorted((-led.get(e["id"], 0), e["price"], e["id"]) for e in self.ed_pool(k)
                              if not e["read"] and not e["gone"] and not e["pointer"] and not e["ruled"] and not e["clash"]
                              and not e['price'] is None and slack > 0 and e["price"] <= slack + 1e-9)
                pools.append([i for _, _, i in pool])
            out: list[str] = []
            while 3 > len(out) and any(pools):
                for pool in pools:
                    if pool and 3 > len(out):
                        out.append(pool.pop(0))
            if out:
                self.slack_total = sum(floors)
            return out
        except (Exception,) as err:
            self.log.append(dict(kind='slack_error', error='{}: {}'.format(type(err).__name__, err)[:300]))
            return []

    def verify_problems(self, ident):
        for entry in reversed(self.log):
            if isinstance(entry, (dict,)) and 'verify' == entry.get('kind', None) and entry.get('item', None) == ident:
                res = entry.get('result', None)
                if isinstance(res, (dict,)) and res.get('fits', None) is False:
                    return [str(p) for p in res.get('problems', None) or [] if str(p).strip(None)]
                return None
        return None

    def ed_news(self, k):
        if self.refusal_n > 0:
            return True
        rows = self.needs.items()
        return any(not rows or self.product_of(str(req).partition(":")[0]) in (k, None)
                   for req in list(self.needs.uncovered) + list(self.needs.let_go))

    def near_hold(self):
        try:
            info = self.__dict__.get('near_cart', None)
            if not info or info.get('ver', None) != self.needs.version or info.get('refusals', None) != self.refusal_n or not self.role("order"):
                return ""
            lines = list(info.get('lines', None) or [])
            if not lines or any(not i in self.trolley.units or i in self.dead_lines() or i in self.refused_lines for i in lines):
                return ""
            if 1 > self.moves_left() or self.wire.done or self.wire.broke or 2 > self.wire.spare():
                return ""
            self.near_cart = None
            self.log.append(dict(kind='near_order', lines=lines))
            was = self.wire.mute
            try:
                self.finish_now()
            finally:
                self.wire.mute = was
            return ('Not run (no step used). A meaning check verified that the lines in the cart (' + ", ".join(lines) + ') state the requested values using different wording, and nothing has changed since then, so the code placed the order for the cart rather than declining.')
        except (Exception,) as err:
            self.log.append(dict(kind='near_hold_error', error='{}: {}'.format(type(err).__name__, err)[:300]))
            return ""

    def near_check(self):
        if self.__dict__.get('near_done', None) or not self.role("cart") or not self.role("order") or 4 > self.moves_left() \
                or self.wire.spare() <= CLOSING + 30 or self.wire.mute or self.__dict__.get('covering', None) or self.wire.done \
                or self.wire.spent_out() or self.wire.no_model:
            return ""
        try:
            return self.near_basket()
        except (Exception,) as err:
            self.log.append(dict(kind='near_error', error='{}: {}'.format(type(err).__name__, err)[:300]))
            return ""

    def near_basket(self):
        rows = self.needs.items()
        prods: list[int | None] = [None] if not rows else list(range(0, len(rows)))
        counts = self.needs.counts()
        cap = None if not is_number(self.needs.spend_cap) else float(self.needs.spend_cap)
        chosen: dict[int, tuple[str, float, int]] = {}
        pools: dict[int, list[tuple[int, float, str, list[tuple[str, str, str]]]]] = {}
        for n, k in enumerate(prods, 0):
            qty = 1 if not n < len(counts) else counts[n]
            pool = self.ed_pool(k)
            fits = sorted((e["price"], e["id"]) for e in pool if e["fit"] and not e['price'] is None)
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
                    says = [str(v).strip(None) for key, v in e["row"]["fields"].items() if same_prop(key, prop) and str(v).strip(None)]
                    if revealed or not says or not self.ed_word(value) or any(re.search(r"\d", s) for s in says):
                        ok = False
                        break
                    lines.append((prop, value, "; ".join(dict.fromkeys(says))[:120]))
                fails = set() if not e['ruled'] else {str(x).split(':', 1)[0].strip(None) for x in self.field_view(e['id'])['fails']}
                probs = [] if not e['ruled'] else self.verify_problems(e['id']) or []
                marks = {w for prop, value, _ in lines for w in prop_words(prop) + re.findall(r"[a-z0-9]+", value.lower())
                         if 3 <= len(w) and not w in LINK_WORDS}
                if ok and lines and all(any(same_prop(f, prop) for prop, _, _ in lines) for f in fails) \
                        and all(any(re.search(r"\b" + re.escape(w), p.lower()) for w in marks) for p in probs):
                    cands.append((len(lines), e["price"], e["id"], lines))
            if not cands:
                return ""
            pools[n] = sorted(cands)[: 3 if not 1 == len(prods) else 6]
        if not pools:
            return ""
        least = sum(price * qty for _, price, qty in chosen.values()) + sum(
            min(c[1] for c in pools[n]) * (1 if not n < len(counts) else counts[n]) for n in pools)
        if not cap is None and least > cap + 1e-9:
            return ""
        flat = [(n, ident, prop, value, says) for n in sorted(pools) for _, _, ident, lines in pools[n] for prop, value, says in lines][:12]
        text = "\n".join("{}. {}: {} = '{}' | the listing's field says: {}".format(i + 1, ident, prop, value[:60], says)
                         for i, (_, ident, prop, value, says) in enumerate(flat, 0))
        ask = ("Below are listings that do not state a requested value in the same words as the request, and a shopping assistant is about to refuse an order because of that. THE SHOPPER'S REQUEST (context):\n" + self.request[:1500] + "\n\nEvery line gives a listing, a property that was requested along with its requested value, and what the listing's own field states for that property.\n" + text
               + '\n\nFor each line, decide: does the listing\'s value refer to the requested value using different terms (a synonym, an alternate spelling or phrasing, or a more precise version of the same thing), or is it a different value of that property? If you\'re uncertain, answer false.\nReply with ONE JSON object and nothing more: {"lines": [{"n": 1, "same": true}]}')
        self.near_done = True
        answer = self.wire.consult(ask, 600, CHECKER)
        yes: set[int] = set()
        said_lines = [] if not (isinstance(answer, (dict,)) and isinstance(answer.get('lines', None), (list,))) else answer.get('lines', None)
        for entry in said_lines:
            if isinstance(entry, (dict,)) and entry.get('same', None) is True and is_number(entry.get('n', None)):
                yes.add(int(entry["n"]) - 1)
        self.log.append(dict(kind='near_check', lines=['{}:{}={}|{}'.format(i, p, v, s)[:120] for _, i, p, v, s in flat], same=sorted(yes)))
        for n in sorted(pools):
            for _, price, ident, lines in pools[n]:
                idx = [i for i, row in enumerate(flat, 0) if row[0] == n and row[1] == ident]
                if len(idx) == len(lines) and all(i in yes for i in idx):
                    chosen[n] = (ident, price, 1 if not n < len(counts) else counts[n])
                    break
            if not n in chosen:
                return ""
        if not cap is None and sum(price * qty for _, price, qty in chosen.values()) > cap + 1e-9:
            return ""
        picks = [(chosen[n][0], chosen[n][2]) for n in range(0, len(prods))]
        if self.moves_left() < len(picks) + 2:
            return ""
        ver = str(self.needs.version)
        for n in pools:
            key = '{}|{}'.format(chosen[n][0], ver)
            for memo in [m for m in self.rulings if key == m or m.startswith(key + "|")]:
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
            stale_now = [i for i in self.stale_lines() if any(p == i for p, _ in picks)]
            if done and stale_now and self.moves_left() >= len(stale_now) + 2:
                _, sold = self.code_recheck(stale_now)
                done = not sold
        finally:
            self.wire.mute = was
        done = done and bool(self.trolley.units) and not set(self.trolley.units) & self.dead_lines()
        self.log.append(dict(kind='near_basket', picks=picks, done=done, cart=dict(self.trolley.units)))
        if not done:
            return ""
        self.near_cart = dict(ver=self.needs.version, refusals=self.refusal_n, lines=[i for i, _ in picks])
        said_by = self.__dict__.setdefault("near_said", {})
        for n, ident, prop, value, says in flat:
            if ident == chosen.get(n, ("",))[0]:
                said_by.setdefault('{}|{}'.format(ident, ver), []).append((prop, value, says))
        said = "; ".join("{} writes {} as '{}' for '{}'".format(ident, prop, says, value[:60]) for n, ident, prop, value, says in flat
                         if ident == chosen.get(n, ("",))[0])
        return ('Not run yet (no step used). A semantic check determined that ' + said + ' - the same value phrased differently - and every such listing satisfies all the needs the shopper disclosed, is in stock and inside the budget, so the code added the basket to the cart and checked its stock: ' + self.trolley.summary() + ' Place the order for this cart now; refuse only if a record you have read shows that one of these lines fails a requirement that currently applies.')

    def cart_moved(self, before):
        self.cart_ver = self.__dict__.get("cart_ver", 0) + 1
        self.since_cart = 0
        now = self.trolley.units
        if any(not line in before for line in now):
            self.cart_seq = self.__dict__.get("absorb_n", 0)
        seqs = self.__dict__.setdefault("line_seq", {})
        for line, units in now.items():
            if line in before and units > before[line] + 1e-9:
                seqs.pop(line, None)

    def after_put(self, call_args, obs):
        if not self.trolley.view_units(obs) is None:
            self.cart_shown = self.__dict__.get("cart_ver", 0)
        firsts = self.__dict__.setdefault("first_puts", set())
        new = [line for line, count in self.trolley.lines(call_args) if (count is None or count > 0) and line in self.trolley.units
               and not line in firsts]
        if not new or not self.trolley.is_put(call_args) or self.code_fault(obs):
            return
        firsts.update(new)
        stale = set(self.stale_lines())
        new = [line for line in new if line in stale]
        tool = self.stock_tool()
        if not new or not tool or self.wire.mute or self.__dict__.get('covering', None) or 4 > self.moves_left() or self.wire.spare() < CLOSING:
            return
        queue = list(self.__dict__.get('put_queue', None) or [])
        self.put_queue = (queue + [line for line in new if not line in queue])[:4]
        self.put_tool = tool
        self.log.append(dict(kind='after_put_read', lines=list(self.put_queue), tool=tool))

    def mark_fresh(self, obs):
        if not self.__dict__.get('stock_seen', None):
            return
        seqs = self.__dict__.setdefault("line_seq", {})

        def visit(d):
            ident = self.rec_id(d)
            if ident in self.trolley.units and not self.stock_state(d) is None:
                seqs[ident] = self.__dict__.get("absorb_n", 0)
        walk_dicts(obs, visit)

    def cart_gone(self, lines):
        gone = set(lines) & self.dead_lines()
        stale = [i for i in self.stale_lines() if i in lines and not i in gone]
        if stale and self.stock_tool() and self.moves_left() >= len(stale) + 2:
            _, sold = self.code_recheck(stale)
            gone |= set(sold) | (set(lines) & self.dead_lines())
        return gone

    def batch_tool(self):
        if not 'batch_spec' in self.__dict__:
            spec = None
            try:
                taken = {self.role(r) for r in ("search", "message", "cart", "cart_view", "order", "decline")} - {None}
                said = {}
                for tdef in self.toolset or []:
                    fn = None if not isinstance(tdef, (dict,)) else tdef.get('function', None)
                    if isinstance(fn, (dict,)) and isinstance(fn.get('name', None), (str,)):
                        said[fn["name"]] = re.sub(r"([a-z])([A-Z])", r"\1 \2", fn["name"]).replace("_", " ") + " " + str(fn.get('description', None) or "")
                for name in list(self.shapes):
                    if re.search(r"\b(?:remov|delet|cancel|clear|discard|releas|refund|purchas|checkout|wish|basket|cart|bag)", said.get(name, name).lower()):
                        taken.add(name)
                for name, jschema in self.shapes.items():
                    props = None if not isinstance(jschema, (dict,)) else jschema.get('properties', None)
                    props = {} if not isinstance(props, (dict,)) else props
                    arrays = [key for key, p in props.items() if isinstance(p, (dict,)) and 'array' == p.get('type', None)
                              and isinstance(p.get('items', None), (dict,)) and 'string' == p['items'].get('type', None) and not p['items'].get('enum', None)]
                    needed = [] if not (isinstance(jschema, (dict,)) and isinstance(jschema.get('required', None), (list,))) else jschema.get('required', None)
                    most = None if not 1 == len(arrays) else props[arrays[0]].get('maxItems', None)
                    if not name in taken and 1 == len(arrays) and not [r for r in needed if not r in arrays] and is_number(most) and most >= 2:
                        spec = (name, arrays[0], int(min(most, 10)))
                        break
            except (Exception,):
                spec = None
            self.batch_spec = spec
        spec = self.batch_spec
        cap = None if not spec else (self.__dict__.get('id_caps', None) or {}).get(spec[0], None)
        return spec if not (spec and cap) else (spec[0], spec[1], min(spec[2], cap))

    @staticmethod
    def list_read_ids(found, ids):
        got: set[str] = set()
        wanted = set(ids)

        def visit(d):
            nested = any(isinstance(v, (dict,)) and 2 <= sum((1 for x in v.values() if isinstance(x, (str, int, float)))) for v in d.values())
            flat = not any(isinstance(v, (dict,)) for v in d.values()) and 5 <= sum((1 for v in d.values() if isinstance(v, (str,)) and 0 < len(v) <= 70))
            if nested or flat or 4 <= len(d):
                got.update(v for v in d.values() if isinstance(v, (str,)) and v in wanted)
        try:
            for obs in found:
                walk_dicts(obs, visit)
        except (Exception,):
            return set()
        return got

    def link_record_known(self, ident):
        return ident in self.archive.read or ident in (self.__dict__.get('link_list_seen', None) or set())

    @staticmethod
    def batch_ok(found, ids):
        got: set[str] = set()
        wanted = set(ids)

        def visit(d):
            nested = any(isinstance(v, (dict,)) and 2 <= sum((1 for x in v.values() if isinstance(x, (str, int, float)))) for v in d.values())
            flat = not any(isinstance(v, (dict,)) for v in d.values()) and 5 <= sum((1 for v in d.values() if isinstance(v, (str,)) and 0 < len(v) <= 70))
            if nested or flat:
                got.update(v for v in d.values() if isinstance(v, (str,)) and v in wanted)
        try:
            for obs in found:
                walk_dicts(obs, visit)
        except (Exception,):
            return False
        return bool(got) and 2 * len(got) >= len(wanted)

    @staticmethod
    def title_of(row):
        if isinstance(row.get('title', None), (str,)):
            return row["title"]
        return next((v for key, v in row.items() if isinstance(v, (str,)) and re.search(r"title|heading|headline", str(key).lower())), "")

    def menu_rows(self, home, most, taken, qty = 1):
        cap = self.needs.spend_cap
        order: list[str] = []
        titles: dict[str, str] = {}
        sold: set[str] = set()
        known: list[set[str]] = []

        def visit(d):
            ident = self.row_mark(d)
            if not ident in self.archive.ids or not self.shelf_of(d) in home:
                return
            if ident in self.archive.read:
                known.append(set(re.findall(r"[a-z0-9]{3,}", self.title_of(d).lower())))
                return
            if self.sold_out(d):
                sold.add(ident)
            if not ident in order:
                order.append(ident)
                titles[ident] = self.title_of(d)
        for rec in self.archive.items:
            walk_dicts(rec, visit)
        hits = sorted(self.__dict__.get("all_hits", []), key=lambda i: -self.__dict__.get("hit_score", {}).get(i, 0))
        cands = [i for i in hits if i in titles] + [i for i in order if not i in hits]
        cands = [i for i in cands if not i in sold and not i in taken and not i in self.refused_lines
                 and not (is_number(cap) and is_number(self.archive.price(i)) and self.archive.price(i) * qty > cap)]
        words = {i: set(re.findall(r"[a-z0-9]{3,}", titles.get(i, "").lower())) for i in cands}
        lead = self.__dict__.get("row_score", {})
        picks: list[str] = []
        try:
            several = bool(self.needs.items())
            k = None if not several else next((n for n, h in enumerate(self.home_shelves(), 0) if home == h), None)
            vals = [re.sub(r"\((?:stated|revealed[^)]*)\)", "", r.partition(":")[2]) for r in self.needs.props_left()
                    if not several or self.product_of(r.partition(":")[0]) in (k, None)]
            vals = [v for v in vals if v.strip(None) and positive_value(v) and not re.search(r"as shown|same as|the page|sold by|redirect|refer", v.lower())]
            facts: dict[str, list[str]] = {}

            def fact_visit(d):
                ident = self.row_mark(d)
                if ident in self.archive.read and self.shelf_of(d) in home:
                    for val in d.values():
                        if isinstance(val, (dict,)):
                            facts.setdefault(ident, []).extend(str(x) for x in val.values() if isinstance(x, (str, int, float)))
            for rec in [] if not (vals and most > 0 and (not several or not k is None)) else self.archive.items:
                walk_dicts(rec, fact_visit)
            fit_read = any(all(value_seen(v, " | ".join(texts)) for v in vals) for texts in facts.values())
            if vals and most > 0 and (not several or not k is None) and not fit_read:
                led = sorted((w for w in self.__dict__.get("page_book", {}).values() if self.walk_carries(w["args"], k)),
                             key=lambda w: -len(self.walk_carries(w["args"], k)))
                for walk in led:
                    first = next((i for i in (walk.get('row_shelf', None) or {}) if i in words), None)
                    if first:
                        picks.append(first)
                        known.append(words[first])
                        break
        except (Exception,):
            picks = []
        while len(picks) < most:
            pool = [i for i in cands if not i in picks][:16]
            if not pool:
                break
            best = min(pool, key=lambda i: ((2 if not lead.get(i, None) else 1) if not i in hits else 0,
                                            max((len(words[i] & w) / max(1, len(words[i] | w)) for w in known), default=0.0), cands.index(i)))
            picks.append(best)
            known.append(words[best])
        return picks

    def pre_read(self):
        try:
            batch = self.batch_tool()
            again = 1 == self.__dict__.get('pre_rounds', 0)
            if self.__dict__.get('list_no_fields', None):
                return False
            if not batch or (self.__dict__.get('pre_read_done', None) and not again) or 1 < len(self.questions_log) \
                    or self.moves_left() <= (20 if not again else 25) or self.wire.spare() < CLOSING + 60:
                return False
            homes = self.home_shelves()
            qtys = [row["quantity"] for row in self.needs.items()] or [self.needs.quantity()]
            if again:
                self.pre_rounds = 2
                wants: list[list[str]] = []
                taken: list[str] = []
                for k, home in enumerate(homes, 0):
                    have = 4 if not home else sum((self.archive.rich_reads(s) for s in home))
                    if have < 3:
                        rows = self.menu_rows(home, 4 - have, taken + list(self.__dict__.get('pre_picked', None) or []), qtys[min(k, len(qtys) - 1)])
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
            for k, home in enumerate(homes, 0):
                have = 4 if not home else sum((self.archive.rich_reads(s) for s in home))
                if have == 0:
                    empty += self.menu_rows(home, 4 if not self.needs.items() else 2, empty, qtys[min(k, len(qtys) - 1)])
                elif have < 4:
                    thin.append((k, 4 - have))
            picks = list(empty)
            for k, n in thin:
                picks += self.menu_rows(homes[k], n, picks, qtys[min(k, len(qtys) - 1)])
            picks = picks[: batch[2] * (1 if not (len(empty) > batch[2] and 25 < self.moves_left()) else 2)]
            if not picks:
                return False
            self.pre_read_done = True
            self.pre_rounds = 1
            self.pre_picked = list(picks)
            self.open_queue = picks
            self.open_recheck = True
            self.open_say = 'I start by sampling several listings for every product the shopper requested, reading their fields, and only then do I put any questions to the shopper.'
            self.open_purpose = "menu"
            return True
        except (Exception,) as err:
            self.log.append(dict(kind='pre_read_error', error='{}: {}'.format(type(err).__name__, err)[:300]))
            return False

    def quick_rows(self, most = 2):
        if self.needs.items():
            return self.lacking_rows()[:most]
        try:
            homes = self.home_shelves()
            home = set() if not homes else homes[0]
            if not home:
                return []
            cap, qty = self.needs.spend_cap, self.needs.quantity()
            where: dict[str, str] = {}
            sold: set[str] = set()

            def visit(d):
                ident, shelf = self.row_mark(d), self.shelf_of(d)
                if ident in self.archive.ids and shelf:
                    where.setdefault(ident, shelf)
                    if self.sold_out(d):
                        sold.add(ident)
            for rec in reversed(self.archive.items):
                walk_dicts(rec, visit)
            hits = sorted(self.__dict__.get("all_hits", []), key=lambda i: -self.__dict__.get("hit_score", {}).get(i, 0))
            picks: list[str] = []
            for ident in dict.fromkeys([i for i in hits if i in where] + list(where)):
                if ident in self.archive.read or ident in self.refused_lines or ident in sold or not where.get(ident, None) in home:
                    continue
                cost = self.archive.price(ident)
                if is_number(cap) and is_number(cost) and is_number(qty) and cost * qty > cap:
                    continue
                picks.append(ident)
                if len(picks) >= most:
                    break
            return picks
        except (Exception,) as err:
            self.log.append(dict(kind='quick_rows_error', error='{}: {}'.format(type(err).__name__, err)[:300]))
            return []

    def lacking_rows(self):
        try:
            rows = self.needs.items()
            if not rows:
                return []
            homes = self.home_shelves()
            live = {k.rpartition("|")[0] for k in self.passed if k.rpartition("|")[2] == str(self.needs.version)} - self.refused_lines - self.dead_lines()
            got = {self.shelf_known(i) for i in live | set(self.__dict__.get('audit_fits', None) or [])} - {""}
            lacking = [k for k, home in enumerate(homes, 0) if home and not home & got]
            if not lacking:
                return []
            batch = self.batch_tool()
            most = 3 if not batch else batch[2]
            cap = self.needs.spend_cap
            newest: list[tuple[str, str]] = []
            sold: set[str] = set()

            def visit(d):
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
                    if where.get(ident, None) in homes[k]:
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
        except (Exception,) as err:
            self.log.append(dict(kind='lacking_error', error='{}: {}'.format(type(err).__name__, err)[:300]))
            return []

    def text_arg(self, the_tool):
        jschema = self.shapes.get(the_tool or '', None) or {}
        prop_specs = jschema.get('properties', None) or {}
        req_names = list(jschema.get('required', None) or [])
        str_params = [fkey for fkey, val in prop_specs.items() if isinstance(val, (dict,)) and ('string' == val.get('type', None) or "string" in (val.get('type', None) or [])) and not 'enum' in val]
        aim = next((fkey for fkey in req_names if fkey in str_params), None) or (None if not 1 == len(str_params) else str_params[0])
        if not aim or any(aim != fkey for fkey in req_names):
            return None
        return aim

    def message_call(self, text_body, cid):
        the_tool = self.role("message")
        aim = None if not the_tool else self.text_arg(the_tool)
        if not aim:
            return None
        call_args, faults = ArgCheck.execute({aim: text_body}, self.shapes[the_tool])
        if faults:
            return None
        return dict(id=cid, type='function', function=dict(name=the_tool, arguments=json.dumps(call_args)))

    def product_of(self, text):
        rows = self.needs.items()
        if not rows:
            return None
        names = [set(kind_tokens(row["product"])) for row in rows]
        for piece in re.findall(r"\(([^)]*)\)", str(text))[:1] + [str(text)]:
            words = set(kind_tokens(piece))
            scores = []
            for k, own in enumerate(names, 0):
                shared = set().union(*(names[j] for j in range(0, len(names)) if k != j))
                scores.append(kind_score(own, words, shared))
            top = max(scores)
            if top > 0 and 1 == scores.count(top):
                return scores.index(top)
        return None

    def ask_noun(self, label):
        rows = self.needs.items()
        if not rows:
            return str(self.needs.reading.get('product', None) or "item")
        k = self.product_of(str(label))
        if k is None:
            prop = re.split('\\s*\\(', str(label))[0].strip(None).lower()
            homes = self.home_shelves()
            read = self.archive.shelves_read()
            hits = []
            for j in range(0, len(rows)):
                own = set() if not j < len(homes) else homes[j]
                menu = " ".join(self.archive.prop_menu(sh)[0] for sh in read if sh.strip(None).lower() in own).lower()
                hits.append(0 if not prop else len(re.findall('(?<![a-z])' + re.escape(prop) + '(?![a-z])', menu)))
            if hits and 0 < max(hits) and 1 == hits.count(max(hits)):
                k = hits.index(max(hits))
        return str(rows[0 if not not k is None else k]["product"])

    def plan_product(self, qitem):
        k = self.product_of(str(qitem.get('property', None) or ""))
        return self.product_of(str(qitem.get('question', None) or '')) if not not k is None else k

    def quota_first(self, scripted, ok = None):
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
                k = None if not said else self.product_of(said.group(1))
            if not k is None:
                sent[k] += 1
                if not '[not answered]' in head:
                    asked[k] += 1
        if 0 == max(asked) and scripted and self.plan_product(scripted[0]) is None:
            return scripted
        found = [0] * len(rows)
        for need in self.needs.uncovered:
            k = self.product_of(str(need).split(":", 1)[0])
            if not k is None:
                found[k] += 1
        try:
            hinted = {self.product_of(str(h).split(":", 1)[0]) for h in self.open_hints()}
        except (Exception,):
            hinted = set()
        unasked = [j for j in range(0, len(rows)) if not asked[j]]
        try:
            named, keyless = set(), [0] * len(rows)
            for h in self.open_hints():
                j = self.product_of(str(h).split(":", 1)[0])
                if j is None:
                    continue
                if not self.plain_hint(j, str(h).split(':', 1)[0].strip(None), hinted=True):
                    named.add(j)
                else:
                    keyless[j] += 1
            hinted = {j for j in range(0, len(rows)) if j in named or keyless[j] > asked[j]}
        except (Exception,):
            hinted = set()

        alt_left = {j for j in range(0, len(rows)) if sent[j] > asked[j] and self.alt_untried(j)}

        def level_of(k):
            return (0 if not (found[k] and any((k != j for j in unasked))) else 1, 1 if not k in hinted else 0, asked[k] + found[k],
                    sent[k] - asked[k] if not k in alt_left else 0, found[k])
        ready = list(range(0, len(rows)))
        for level in sorted({level_of(k) for k in ready}):
            due = [k for k in ready if level_of(k) == level]
            for n, qitem in enumerate(scripted, 0):
                if self.plan_product(qitem) in due:
                    return [qitem] + scripted[:n] + scripted[n + 1:]
            for qitem in scripted:
                second = str(qitem.get('second', None) or '').strip(None)
                j = None if not second else self.product_of(second)
                if j in due:
                    tied = re.search(r"\(([^)]+)\)", second)
                    product = rows[j]['product'] if not tied else tied.group(1).strip(None)
                    return [dict(property=second, question='Which {} would you like for the {}?'.format(second.split('(')[0].strip(None), product))] + scripted
            for k in sorted(due, key=lambda k: (not k in hinted, k)):
                try:
                    pick = self.quota_pick(k, 0 == asked[k], ok)
                except (Exception,):
                    pick = None
                if pick and not ok is None and (not ok(pick.get('property', None)) or self.asked_for(k, pick.get('property', None))):
                    pick = None
                if pick:
                    return [pick] + scripted
                if self.__dict__.get('quota_wait', None):
                    return scripted
        return scripted

    def quota_pick(self, k, fallback = True, ok = None):
        product = self.needs.items()[k]["product"]
        homes = self.home_shelves()
        home = set() if not k < len(homes) else homes[k]
        read_home = [s for s in self.archive.shelves_read() if s.strip(None).lower() in home]
        if not read_home and home and self.__dict__.get('quota_reads', {}).get(k, None):
            read_home = [s for s in sorted(home) if self.archive.prop_menu(s)[0]][:1]
        if not read_home:
            batch = self.batch_tool()
            unread = [] if not home else self.menu_rows(home, 2 if not batch else min(4, batch[2]), [], self.needs.items()[k]['quantity'])
            if unread and (self.role("details") or batch) and not self.__dict__.get('quota_reads', {}).get(k, None) and 12 < self.moves_left():
                self.__dict__.setdefault("quota_reads", {})[k] = True
                self.open_queue = unread
                self.open_recheck = True
                self.open_say = 'To learn which fields its listings carry, I look through several of them before I put a question about the {}.'.format(product)
                self.open_purpose = "quota"
                self.quota_wait = True
            return None
        menu = self.menu_trim(self.archive.prop_menu(read_home[0])[0], self.archive.rich_reads(read_home[0]))

        def good(label):
            return ok is None or (bool(ok(label)) and not self.asked_for(k, label))
        pick = self.product_plan(k, menu)
        if pick and good(pick.get('property', None)):
            return pick
        try:
            hinted = self.open_hints(k)
        except (Exception,):
            hinted = []
        for hint in hinted:
            prop = re.split('\\s*\\(', hint.split(':', 1)[0])[0].strip(None)
            if prop and not junk_key(prop) and not self.plain_hint(k, prop, hinted=True) and good('{} ({})'.format(prop, product)):
                return dict(property='{} ({})'.format(prop, product), question='Which {} would you like for the {}?'.format(prop, product))
        if not fallback:
            return None
        for weigh in (False, True):
            for line in menu.splitlines():
                prop = line.strip('- ').split(' (in ')[0].strip(None)
                if prop and self.open_prop(k, prop, line, by_samples=weigh) and good('{} ({})'.format(prop, product)):
                    return dict(property='{} ({})'.format(prop, product), question='Which {} would you like for the {}?'.format(prop, product))
        return None

    def product_plan(self, k, menu):
        try:
            return self.product_plan_call(k, menu)
        except (Exception,):
            return None

    def product_plan_call(self, k, menu):
        if not menu or 1 <= self.__dict__.get('product_plans', 0) or self.wire.spare() < CLOSING + 40:
            return None
        self.product_plans = self.__dict__.get("product_plans", 0) + 1
        product = self.needs.items()[k]["product"]
        remaining = max(1, self.ask_room() - self.msgs_sent)
        ask_text = ASK_PLAN.replace("QUESTION_COUNT", str(min(3, remaining))) + "\n\nTHE SHOPPER'S REQUEST:\n" + self.request
        ask_text += "\n\nREQUIREMENTS SO FAR:\n" + self.needs.text_body()
        if self.questions_log:
            ask_text += "\n\nALREADY ASKED (along with the shopper's own words):\n" + "\n".join(self.questions_log)
        ask_text += ('\n\nTHIS TIME EVERY QUESTION IS ABOUT THE {}: all the other requested products already had a question, and this one lacked any. Pick properties of the {} alone, drawn from its own list below, starting with the one that the details of the story point to.'.format(product.upper(), product))
        try:
            hinted = self.open_hints(k)
        except (Exception,):
            hinted = []
        if hinted:
            ask_text += ('\n\nSTORY HINTS (particulars of the story behind the request and the attribute each one appears to relate to, with the most probable listed at the top; these are not requirements):\n'
                         + "\n".join('- {}'.format(self.hint_mark(h)) for h in hinted[:6]))
        ask_text += self.plan_notes(k)
        ask_text += ('\n\nPROPERTIES IN THE DETAILS OF THE {} LISTINGS SEEN SO FAR (name, listing count, sample values):\n'.format(product.upper()) + menu)
        answer = self.wire.consult(ask_text, 1500, PLANNER[:1])
        self.log.append(dict(kind='product_plan', product=product, result=answer))
        words = kind_words(product)
        for qitem in (answer or {}).get('questions', None) or []:
            if not isinstance(qitem, (dict,)):
                continue
            first = str(qitem.get('property', None) or '').split('(')[0].strip(None)
            second = str(qitem.get('second', None) or '').split('(')[0].strip(None)
            second = second if not re.search('none|^-$|n/a', second.lower()) else ''
            question = str(qitem.get('question', None) or '').strip(None)
            first = first if not self.plain_hint(k, first) else ''
            second = second if not (second and self.plain_hint(k, second)) else ''
            if first and self.plan_open(k, first):
                named = bool(question) and bool(words & kind_words(question))
                pair = '' if not (named and second and (not strict_prop(first, second)) and self.plan_open(k, second)) else second
                return (dict(property='{} ({})'.format(first, product), second='' if not pair else '{} ({})'.format(pair, product), question='Which {} would you like for the {}?'.format(first, product) if not named else question))
            if second and self.plan_open(k, second):
                return dict(property='{} ({})'.format(second, product), question='Which {} would you like for the {}?'.format(second, product))
        return None

    def open_prop(self, k, prop, line = "", by_samples = False):
        try:
            name = str(prop).split('(')[0].strip(None)
            if not name or junk_key(name):
                return False
            several = bool(self.needs.items())

            def mine(text, tagged_only = False):
                if k is None or not several:
                    return True
                j = self.product_of(text)
                return k == j or (j is None and not tagged_only)
            for ln in self.questions_log:
                head = ln.split(":", 1)[0].strip("- ")
                if not '[not answered]' in head and any(mine(part) and self.same_field(k, part, name) for part in head.split(";")):
                    return False
            samples = '' if not '):' in line else line.split('):', 1)[1].lower()
            for req in self.needs.props_left():
                head, _, value = str(req).partition(":")
                if not mine(head):
                    continue
                if strict_prop(head, name):
                    if self.settles(str(req), '' if not by_samples else samples):
                        return False
                    continue
                if same_prop(head, name):
                    if positive_value(value, head) and value_seen(value, samples or " | ".join(self.field_values(k, name))):
                        return False
                    continue
                words = [w for w in re.findall(r"[a-z]{3,}", value.lower()) if not w in LINK_WORDS]
                if samples and words and not re.search(r"\d", value) and positive_value(value, head) \
                        and all(re.search(r"\b" + re.escape(w), samples) for w in words):
                    return False
            return not any(mine(str(p), tagged_only=True) and self.same_field(k, str(p), name) for p in self.needs.indifferent)
        except (Exception,):
            return True

    def plan_notes(self, k):
        try:
            several = bool(self.needs.items())
            asked = {p.strip(None).lower() for ln in self.questions_log for p in ln.split(":", 1)[0].strip("- ").split(";")}
            opens = []
            for prop in self.__dict__.get('open_pair', None) or []:
                j = None if not several else self.product_of(prop)
                if (not k is None and several and not j in (k, None)) or prop.lower() in asked or not self.plan_open(j, prop):
                    continue
                opens.append(prop)
            wanted = [w for w in self.__dict__.get('wanted_asks', None) or [] if k is None or not several or self.product_of(w) in (k, None)]
            text = ""
            if opens:
                text += ("\n\nOPEN FROM LAST PAIR (this is the planned pair's second property, and the reply left it unanswered, either because the shopper responded only to the first or because only the first went out, so it remains open; in each case it was the plan's next choice):\n"
                         + "\n".join('- {}'.format(p) for p in opens[-3:]))
            if wanted:
                text += ("\n\nQUESTIONS THE ASSISTANT WANTED TO ASK (the listing records have been read by it; if such a question names a property that remains open and appears below as a key, that one goes first, and you ask about it using the key's exact name):\n" + "\n".join('- {}'.format(w) for w in wanted[-3:]))
            return text
        except (Exception,):
            return ""

    def want_ask(self, raw_args):
        try:
            args = ({} if not (isinstance(raw_args, (str,)) and raw_args.strip(None)) else json_object(raw_args)) if not isinstance(raw_args, (dict,)) else raw_args
            text = " ".join(str(v).strip(None) for v in (args or {}).values() if isinstance(v, (str,)) and v.strip(None))[:300]
            if text:
                kept = [w for w in self.__dict__.get('wanted_asks', None) or [] if w.lower() != text.lower()]
                self.wanted_asks = (kept + [text])[-3:]
        except (Exception,):
            pass

    def drop_wanted(self, label):
        try:
            props = [re.split(r"\s*\(", p)[0] for p in re.split(r";|\band\b", str(label).replace("[not answered]", ""))]
            keep = []
            for text in self.__dict__.get('wanted_asks', None) or []:
                words = {self.stem(w) for w in re.findall(r"[a-z]+", text.lower())}
                if any(prop_words(p) and all(w in words or self.stem(w) in words for w in prop_words(p)) for p in props):
                    continue
                keep.append(text)
            self.wanted_asks = keep
        except (Exception,):
            pass

    def names_other(self, question, prop, menu, second = ""):
        try:
            flat = " ".join(re.findall(r"[a-z0-9]+", str(question).lower()))
            said = set(flat.split())
            said |= {w[:-1] for w in said if 3 < len(w) and w.endswith("s") and not w.endswith("ss")}
            own = set(prop_words(prop)) | {"which", "would", "you", "like"}
            for row in self.needs.items() or [dict(product=str(self.needs.reading.get('product', None) or ''))]:
                for w in re.findall(r"[a-z0-9]+", str(row.get('product', None) or "").lower()):
                    own |= {w, w if not (3 < len(w) and w.endswith('s')) else w[:-1]}

            def named(name):
                words = [w for w in prop_words(name) if 3 <= len(w) and not w in LINK_WORDS]
                return bool(words) and all(w in said for w in words) and any(not w in own for w in words)
            if second and not strict_prop(second, prop) and named(second):
                return True
            for line in str(menu or "").splitlines():
                key = line.strip(None).lstrip('-').strip(None).split(' (in ')[0].strip(None)
                if not key or key.endswith(":") or strict_prop(key, prop) or junk_key(key):
                    continue
                if named(key):
                    return True
                for sample in ('' if not '):' in line else line.split('):', 1)[1]).split("|"):
                    toks = re.findall(r"[a-z0-9]+", sample.lower())
                    if any(not t in own and not t in LINK_WORDS for t in toks) and 3 <= len(''.join(toks)) \
                            and re.search(r"(?<![a-z0-9])" + re.escape(" ".join(toks)) + r"(?![a-z0-9])", flat):
                        return True
            return False
        except (Exception,):
            return False

    def pair_fits(self, first, second):
        try:
            several = bool(self.needs.items())
            k = None if not several else self.product_of(first)
            if several and (k is None or self.product_of(second) != k):
                return False
            if strict_prop(first, second) or same_prop(first, second) or self.same_field(k, first, second):
                return False
            a = {w for w in prop_words(first) if 3 <= len(w) and not w in LINK_WORDS}
            b = {w for w in prop_words(second) if 3 <= len(w) and not w in LINK_WORDS}
            if (a & b) or (self.field_values(k, first) & self.field_values(k, second)):
                return False
            menu = (self.__dict__.get('menu_memo', {}).get(k, None) or (0, ""))[1]
            for line in menu.splitlines():
                key = line.strip('- ').split(' (in ')[0].strip(None)
                if key and not strict_prop(key, first) and not strict_prop(key, second) and self.twin_of(k, key, second) \
                        and self.plan_open(k, key):
                    return False
            return True
        except (Exception,):
            return False

    def queue_second(self, prop):
        queue = self.__dict__.setdefault("pending_seconds", [])
        if prop and not prop.lower() in [p.lower() for p in queue]:
            queue.append(prop)

    def reply_ref_open(self):
        return any('open' == r.get('status', None) and 'reply' == r.get('origin', None) for r in self.__dict__.get('refs', None) or [])

    def ask_first_due(self):
        raw = self.needs.reading.get('ask_first', None)
        text = str(raw or '') if not isinstance(raw, (list,)) else ' '.join((str(x) for x in raw))
        return bool(text.strip(None)) and bool(re.search(r"budget|price|spend|cost|how much|afford|rather|\bprefer(?:s|red)?\b|which (?:of|one)|"
                                                     r"matters? more|more important|either", text, re.I))

    def pose_planned(self, urge = False):
        the_tool = self.role("message")
        ceiling = self.ask_room()
        if not the_tool or ceiling <= self.msgs_sent:
            self.asking = False
            return False
        prop_list, _ = self.archive.prop_menu()
        sets = self.archive.rich_reads()
        own = self.own_shelf()
        if own:
            prop_list, sets = self.archive.prop_menu(own)[0], self.archive.rich_reads(own)
        pressing = self.ask_first_due() and not self.questions_log
        foreign = False
        if not own and not self.needs.items() and not pressing and self.__dict__.get('refs', None):
            try:
                foreign = not self.home_shelves()[0] and bool(self.archive.shelves_read())
            except (Exception,):
                foreign = False
            if foreign and self.moves_left() > self.order_due() + 12:
                return False
            if foreign:
                prop_list = ""
        thin = ASK_AFTER_SETS > sets and not pressing and not urge
        if thin and (ASK_AFTER_STEPS > self.wire.tick or (self.wire.tick < ASK_AFTER_STEPS + ASK_GRACE_STEPS and 20 < self.moves_left())):
            return False
        if self.moves_left() <= max(10, self.order_due()) or self.wire.spare() < CLOSING + 40:
            self.asking = False
            return False
        if self.reply_ref_open() and not urge:
            return False
        waiting = self.__dict__.get('plan_wait', None)
        if not waiting is None and waiting >= sets and not urge:
            return False
        last_sets = self.__dict__.get('asked_at_sets', None)
        if not last_sets is None and last_sets >= sets and not urge and not pressing \
                and self.moves_left() > self.order_due() + 2 * max(0, ceiling - self.msgs_sent) + 14 \
                and 4 > self.wire.tick - self.__dict__.get('asked_at_tick', 0) \
                and not self.needs.items():
            return False
        if self.pre_read():
            return True
        if not pressing and ASK_AFTER_SETS > sets and self.moves_left() > self.order_due() + 10 and not self.first_reads() \
                and not self.archive.read:
            return False
        if not self.questions_log and not self.msgs_sent and not pressing and (not urge or ASK_AFTER_SETS > sets) and self.first_reads():
            return True
        remaining = ceiling - self.msgs_sent
        ask_text = ASK_PLAN.replace("QUESTION_COUNT", str(remaining)) + NEXT_ONE.replace("REMAINING", str(remaining))
        ask_text += "\n\nTHE SHOPPER'S REQUEST:\n" + self.request
        ask_text += "\n\nREQUIREMENTS SO FAR:\n" + self.needs.text_body()
        if self.questions_log:
            ask_text += "\n\nALREADY ASKED (along with the shopper's own words):\n" + "\n".join(self.questions_log)
        if self.ask_first_due():
            ask_text += "\n\nMUST BE LEARNED FROM THE SHOPPER: " + "; ".join(str(h) for h in self.needs.reading.get('ask_first', None) or [])
        ask_text += self.plan_notes(None)
        try:
            hinted = self.open_hints()
        except (Exception,):
            hinted = []
        if hinted:
            ask_text += ('\n\nSTORY HINTS (particulars of the story behind the request and the attribute each appears to relate to; these are not requirements; the concealed need is typically found in one of them):\n' + "\n".join('- {}'.format(self.hint_mark(h)) for h in hinted[:8]))
        shown = self.menu_trim(prop_list, sets)
        if self.needs.items():
            ask_text += "\n\nTHE PRODUCTS ASKED FOR: " + " | ".join('{}) {} x{}'.format(n + 1, row['product'], row['quantity']) for n, row in enumerate(self.needs.items(), 0))
            read = self.archive.shelves_read()
            homes = self.home_shelves()
            mine = [s for s in read if s.strip(None).lower() in set().union(*homes)]
            order = mine + [s for s in read if not s in mine] if not all(homes) else mine or read
            menus = [(shelf, self.archive.prop_menu(shelf)[0]) for shelf in order[:4]]
            prop_list = "\n".join('CATEGORY {}:\n{}'.format(shelf, menu) for shelf, menu in menus if menu) or prop_list
            shown = "\n".join('CATEGORY {}:\n{}'.format(shelf, self.menu_trim(menu, self.archive.rich_reads(shelf)))
                              for shelf, menu in menus if menu) or shown
        ask_text += "\n\nPROPERTIES IN THE LISTINGS' DETAILS SEEN SO FAR (name, listing count, sample values):\n" + (shown or "(none seen yet)")
        asked_props = {p.strip(None).lower() for ln in self.questions_log for p in ln.split(":", 1)[0].strip("- ").split(";")}

        def fresh(prop):
            prop = str(prop or '').strip(None)
            if not prop or prop.lower() in asked_props or junk_key(prop) or prop.lower() in self.__dict__.get("spent_asks", set()):
                return False
            if not self.needs.items() and re.split('\\s*\\(', prop)[0].strip(None).lower() in self.__dict__.get("spent_asks", set()):
                return False
            if re.search(r"\b(?:options?|producer|condition|warranty|origin|variants?|series|years?|version)\b",
                         re.split(r"\s*\(", prop)[0].lower()):
                return False
            j = None if not self.needs.items() else self.product_of(prop)
            return not self.plain_hint(j, prop) and self.plan_open(j, prop)
        queue = [p for p in self.__dict__.get('pending_seconds', None) or [] if fresh(p)]
        self.pending_seconds = queue
        pending = '' if not queue else queue[0]
        if pending:
            tied = re.search(r"\(([^)]+)\)", pending)
            product = self.ask_noun(pending) if not tied else tied.group(1).strip(None)
            text = (self.__dict__.get('pending_text', None) or {}).pop(pending.lower(), "")
            answer = dict(questions=[dict(property=pending, question=text or 'Which {} would you like for the {}?'.format(pending.split('(')[0].strip(None), product))])
        else:
            plan_sig = (self.needs.version, len(self.needs.uncovered), min(sets, 6) // 3)
            cached = ([] if not self.__dict__.get('plan_sig', None) == plan_sig else [q for q in self.__dict__.get('plan_cache', None) or [] if isinstance(q, (dict,)) and fresh(q.get('property', None))])
            if cached:
                answer = dict(questions=cached)
                self.log.append(dict(kind='plan_cached', left=len(cached)))
            else:
                try:
                    old_sig = self.__dict__.get('plan_sig', None)
                    why = self.__dict__.setdefault("plan_why", {})
                    if old_sig is None:
                        tag = "first"
                    elif plan_sig != old_sig:
                        tag = "sig_" + "".join(n for n, a, b in zip(("v", "u", "s"), old_sig, plan_sig) if b != a)
                    else:
                        raw = [q for q in self.__dict__.get('plan_cache', None) or [] if isinstance(q, (dict,))]
                        tag = 'stale' if raw else 'empty'
                    why[tag] = why.get(tag, 0) + 1
                except (Exception,):
                    tag = ""
                built: list[dict[str, Any]] = []
                if tag == 'stale' and not self.needs.items():
                    try:
                        counts: dict[str, int] = {}
                        for ln in str(prop_list).splitlines():
                            mm = re.match(r"\s*-\s*(.+?)\s+\(in (\d+) listings?\)", ln)
                            if mm:
                                counts[mm.group(1).strip(None)] = max(counts.get(mm.group(1).strip(None), 0), int(mm.group(2)))
                        product = str(self.needs.reading.get('product', None) or "item")
                        for name in sorted(counts, key=lambda c: -counts[c]):
                            if fresh(name) and not re.search(r"\b(?:options?|producer|condition|warranty|origin|variants?|series|years?|version|model|brand|price|stock)\b", name.lower()):
                                built.append(dict(property=name, question='Which {} would you like for the {}?'.format(name, product)))
                            if len(built) >= max(1, remaining):
                                break
                    except (Exception,) as err:
                        built = []
                        self.log.append(dict(kind='plan_menu_error', error='{}: {}'.format(type(err).__name__, err)[:200]))
                if not built:
                    answer = self.wire.consult(ask_text, 2500, PLANNER)
                else:
                    answer = dict(questions=built)
                    self.log.append(dict(kind='plan_menu', n=len(built)))
                if answer is None and self.wire.misses > 0:
                    self.plan_wait = sets
                    return False
                fresh_plan = [q for q in (answer or {}).get('questions', None) or [] if isinstance(q, (dict,))]
                named = {str(q.get('property', None) or '').strip(None).lower() for q in fresh_plan}
                older = [q for q in self.__dict__.get('plan_cache', None) or [] if isinstance(q, (dict,))
                         and not str(q.get('property', None) or '').strip(None).lower() in named and fresh(q.get('property', None))]
                self.plan_cache = fresh_plan + older
                self.plan_sig = plan_sig
        scripted = [qitem for qitem in (answer or {}).get('questions', None) or [] if isinstance(qitem, (dict,)) and str(qitem.get('question', None) or '').strip(None)]
        kept = []
        for qitem in scripted:
            second = str(qitem.get('second', None) or '').strip(None)
            second = second if not re.search('none|^-$|n/a', second.lower()) else ''
            if fresh(qitem.get('property', None)):
                kept.append(dict(qitem, second='' if not (second and fresh(second)) else second))
                if second and self.plain_hint(None if not self.needs.items() else self.product_of(second), second):
                    first = str(qitem.get('property', None) or '').strip(None)
                    tied = re.search(r"\(([^)]+)\)", first)
                    product = self.ask_noun(first) if not tied else tied.group(1).strip(None)
                    kept[-1]["question"] = 'Which {} would you like for the {}?'.format(first.split('(')[0].strip(None), product)
            elif second and fresh(second):
                tied = re.search(r"\(([^)]+)\)", second)
                product = self.ask_noun(second) if not tied else tied.group(1).strip(None)
                kept.append(dict(property=second, question='Which {} would you like for the {}?'.format(second.split('(')[0].strip(None), product)))
        scripted = kept
        if self.needs.items() and not pending:
            scripted = self.quota_first(scripted, fresh)
            if self.__dict__.pop("quota_wait", False):
                return True
        proposal = None
        split_second = ""
        for thing in scripted[:2]:
            if self.needs.items():
                prop = str(thing.get('property', None) or "")
                tied = re.search(r"\(([^)]+)\)", prop)
                product = '' if not tied else tied.group(1).strip(None)
                words = re.findall(r"[a-z]{3,}", product.lower())
                if product and words and not any(w.rstrip("s") in str(thing.get('question', None) or "").lower() for w in words):
                    thing = dict(thing, question='Which {} would you like for the {}?'.format(prop.split('(')[0].strip(None), product))
            if not pending and not (self.ask_first_due() and not self.questions_log) and self.names_other(str(thing.get('question', None) or ""), str(thing.get('property', None) or ""), prop_list,
                                                str(thing.get('second', None) or "")):
                prop = str(thing.get('property', None) or "")
                tied = re.search(r"\(([^)]+)\)", prop)
                product = self.ask_noun(prop) if not tied else tied.group(1).strip(None)
                thing = dict(thing, question='Which {} would you like for the {}?'.format(prop.split('(')[0].strip(None), product))
            prop_q = str(thing.get('property', None) or '').strip(None)
            if prop_q and not pending and not (self.ask_first_due() and not self.questions_log):
                tied = re.search(r"\(([^)]+)\)", prop_q)
                product = self.ask_noun(prop_q) if not tied else tied.group(1).strip(None)
                thing = dict(thing, question='Which {} would you like for the {}?'.format(prop_q.split('(')[0].strip(None), product))
            proposal = self.message_call(str(thing.get('question', None)).strip(None), 'ask_{}_{}'.format(self.msgs_sent + 1, self.wire.tick))
            if proposal:
                label = (str(thing.get('property', None)), str(thing.get('question', None)).strip(None))
                second = str(thing.get('second', None) or '').strip(None)
                if not self.__dict__.get("no_pairs", True):
                    counts: dict[str, int] = {}
                    for ln in str(prop_list).splitlines():
                        mm = re.match(r"\s*-\s*(.+?)\s+\(in (\d+) listings?\)", ln)
                        if mm:
                            counts[mm.group(1).strip(None)] = max(counts.get(mm.group(1).strip(None), 0), int(mm.group(2)))
                    tied0 = re.search(r"\(([^)]+)\)", label[0])
                    cands = [c if not tied0 else '{} ({})'.format(c, tied0.group(1).strip(None)) for c in sorted(counts, key=lambda c: -counts[c])]
                    nxt = [str(q.get('property', None) or '').strip(None) for q in scripted if not q is thing]
                    pick2 = next((p for p in cands + nxt if p and fresh(p) and self.pair_fits(label[0], p)
                                  and not re.search(r"\b(?:options?|producer|condition|warranty|origin|variants?|series|years?|version|model)\b", p.lower())), "")
                    if pick2:
                        second = pick2
                if second and second.lower() != label[0].lower() and not re.search(r"none|^-$|n/a", second.lower()):
                    if not pending and not self.__dict__.get("no_pairs", True) and not (self.ask_first_due() and not self.questions_log) \
                            and fresh(second) and self.pair_fits(label[0], second):
                        tied = re.search(r"\(([^)]+)\)", label[0])
                        product = self.ask_noun(label[0]) if not tied else tied.group(1).strip(None)
                        pair = 'Which {} or {} would you like for the {}?'.format(label[0].split('(')[0].strip(None), second.split('(')[0].strip(None), product)
                        proposal = self.message_call(pair, 'ask_{}_{}'.format(self.msgs_sent + 1, self.wire.tick))
                        if not proposal:
                            continue
                        label = (label[0] + "; " + second, pair)
                        spent = self.__dict__.setdefault("spent_asks", set())
                        for nm in (label[0].split(';')[0].strip(None), second):
                            spent.update({nm.lower(), re.split('\\s*\\(', nm)[0].strip(None).lower()})
                        self.pair_spent = [label[0].split(';')[0].strip(None), second]
                    else:
                        prop = label[0].split('(')[0].strip(None)
                        tied = re.search(r"\(([^)]+)\)", label[0])
                        product = self.ask_noun(label[0]) if not tied else tied.group(1).strip(None)
                        single = 'Which {} would you like for the {}?'.format(prop, product)
                        proposal = self.message_call(single, 'ask_{}_{}'.format(self.msgs_sent + 1, self.wire.tick))
                        label = (label[0], single)
                        if not proposal:
                            continue
                        split_second = second
                break
        if not proposal:
            if ASK_AFTER_SETS > sets and 12 < self.moves_left() and not urge:
                self.plan_wait = sets
                return False
            try:
                unread = [k for k in range(0, len(self.needs.items()) or 1) if 2 > self.product_reads(None if not self.needs.items() else k)]
            except (Exception,):
                unread = []
            if unread and 12 < self.moves_left() and not urge:
                self.plan_wait = sets
                return False
            self.empty_plans = self.__dict__.get("empty_plans", 0) + 1
            if 12 < self.moves_left() and self.empty_plans <= 1:
                self.plan_wait = sets
                return False
            self.asking = False
            return False
        self.asked_at_sets, self.asked_at_tick = sets, self.wire.tick
        self.chat.append(dict(role='assistant', content='I will ask the shopper about {}.'.format(label[0]), tool_calls=[proposal]))
        found_before, freed_before = len(self.needs.uncovered), len(self.needs.indifferent) + self.needs.__dict__.get("pinned_heard", 0)
        self.asking_label = label[0]
        try:
            responses = self.dispatch([proposal], scripted=True)
        finally:
            self.asking_label = ""
        self.post_replies(responses)
        asked_label = label[0]
        if pending and label[0].split(';')[0].strip(None).lower() == pending.lower():
            self.pending_seconds = [p for p in self.__dict__.get('pending_seconds', None) or [] if p.lower() != pending.lower()]
        open_note = ""
        heard_meta = self.needs.__dict__.pop("meta_last", None)

        def plain(text):
            return " ".join(re.findall(r"[a-z0-9]+", str(text).lower()))
        try:
            ps = list(self.__dict__.get('pair_spent', None) or [])
            if heard_meta is None and 2 == len(ps) and len(self.needs.uncovered) == found_before \
                    and 1 <= self.ask_room() - self.msgs_sent:
                sec = ps[1]
                low_sec = sec.lower()
                bare = re.split('\\s*\\(', sec)[0].strip(None).lower()
                spent = self.__dict__.setdefault("spent_asks", set())
                spent.discard(low_sec)
                spent.discard(bare)
                self.needs.indifferent = [x for x in self.needs.indifferent if not same_prop(re.split(r"\s*\(", str(x))[0], bare)]
                self.pending_seconds = [sec] + [p for p in self.__dict__.get('pending_seconds', None) or [] if p.lower() != low_sec]
                self.pair_spent = []
                self.log.append(dict(kind='pair_second_reask', prop=sec[:40]))
        except (Exception,) as err:
            self.log.append(dict(kind='pair_second_error', error='{}: {}'.format(type(err).__name__, err)[:200]))
        if not heard_meta is None and plain(label[1]) and plain(label[1]) in plain(heard_meta):
            self.no_pairs = True
            spent = self.__dict__.setdefault("spent_asks", set())
            for nm in self.__dict__.pop("pair_spent", []) or []:
                spent.discard(nm.lower())
                spent.discard(re.split('\\s*\\(', nm)[0].strip(None).lower())
            parts = [p.strip(None) for p in label[0].split(";") if p.strip(None)] or [label[0]]
            first = parts[0]
            tied = re.search(r"\(([^)]+)\)", first)
            product = self.ask_noun(first) if not tied else tied.group(1).strip(None)
            again = self.__dict__.setdefault("reasked", set())
            plain_q = 'Which {} would you like for the {}?'.format(first.split('(')[0].strip(None), product)
            alone = 1 == len(parts)
            alt = '' if not alone else self.collide_alt(first)
            if alt:
                again.add(first.lower())
                self.__dict__.setdefault("spent_asks", set()).add(first.lower())
                self.pending_seconds = [alt] + [p for p in self.__dict__.get('pending_seconds', None) or [] if p.lower() != alt.lower()]
                open_note = ' [{}: no answer given, so the alternative {} from its story hint gets asked next]'.format(first, alt)
            elif not first.lower() in again and plain(plain_q) != plain(label[1]) and not (alone and self.holds_plain(label[1], plain_q)):
                again.add(first.lower())
                rest = [p for p in parts[1:] if p.lower() != first.lower()]
                self.pending_seconds = [first] + rest + [p for p in self.__dict__.get('pending_seconds', None) or []
                                                         if p.lower() != first.lower() and not p.lower() in {r.lower() for r in rest}]
                open_note = ' [{}'.format(first) + ('' if not rest else ' and ' + ', '.join(rest)) + ": not answered, asked again alone next]"
            else:
                redo = None if not not first.lower() in again else self.distinct_ask(first)
                if redo:
                    again.update({first.lower(), redo[0].lower()})
                    self.__dict__.setdefault("pending_text", {})[redo[0].lower()] = redo[1]
                    self.pending_seconds = [redo[0]] + [p for p in self.__dict__.get('pending_seconds', None) or [] if p.lower() != redo[0].lower()]
                    itself = re.split('\\s*\\(', redo[0])[0].strip(None).lower() == re.split('\\s*\\(', first)[0].strip(None).lower()
                    if not itself:
                        self.__dict__.setdefault("spent_asks", set()).add(first.lower())
                    open_note = ' [{}: not answered, asked again '.format(first) + ('as {} next]'.format(redo[0]) if not itself else 'by its values next]')
                else:
                    self.__dict__.setdefault("spent_asks", set()).add(first.lower())
                    open_note = ' [{}: not answered, not asked again]'.format(first)
            pairs = self.__dict__.setdefault("open_pair", [])
            for second in parts[1:] + ([] if not split_second else [split_second]):
                if not second.lower() in [p.lower() for p in pairs]:
                    pairs.append(second)
            label = (" and ".join(parts) + " [not answered] [one at a time]", label[1])
        elif ";" in label[0] and len(self.needs.uncovered) > found_before:
            second = label[0].split(';', 1)[1].strip(None)
            found_text = " ".join(self.needs.uncovered[found_before:]).lower()
            if not any(w in found_text for w in re.findall(r"[a-z]{4,}", second.split("(")[0].lower())):
                self.queue_second(second)
                open_note = ' [{}: not answered, still open]'.format(second)
            label = (label[0].split(';')[0].strip(None), label[1])
        elif ";" in label[0] and len(self.needs.indifferent) + self.needs.__dict__.get("pinned_heard", 0) > freed_before:
            second = label[0].split(';', 1)[1].strip(None)
            label = (label[0].split(';')[0].strip(None), label[1])
            self.needs.indifferent = [p for p in self.needs.indifferent if not p.strip(None).lower() in (second.lower(), second.split('(')[0].strip(None).lower())]
            pairs = self.__dict__.setdefault("open_pair", [])
            if not second.lower() in [p.lower() for p in pairs]:
                pairs.append(second)
            open_note = ' [{}: not answered, still open]'.format(second)
        elif ";" in label[0] and len(self.needs.indifferent) + self.needs.__dict__.get("pinned_heard", 0) == freed_before:
            self.no_pairs = True
            pairs = self.__dict__.setdefault("open_pair", [])
            for part in (p.strip(None) for p in label[0].split(";")):
                if part and not part.lower() in [p.lower() for p in pairs]:
                    pairs.append(part)
            label = (label[0].replace(";", " and") + " [not answered]", label[1])
        elif split_second:
            found_text = " ".join(self.needs.uncovered[found_before:]).lower()
            pairs = self.__dict__.setdefault("open_pair", [])
            if not any(w in found_text for w in re.findall(r"[a-z]{4,}", split_second.split("(")[0].lower())) \
                    and not split_second.lower() in [p.lower() for p in pairs]:
                pairs.append(split_second)
        if heard_meta is None and len(self.needs.uncovered) == found_before:
            self.__dict__.setdefault("quiet_asks", []).append(asked_label.split(';')[0].strip(None))
            twin = ""
            if twin:
                self.pending_seconds = [twin] + [p for p in self.__dict__.get('pending_seconds', None) or [] if p.lower() != twin.lower()]
                open_note += ' [{}: its story twin, asked next]'.format(twin)
        for _, text_body in responses[:1]:
            self.questions_log.append('- {}: asked "{}" -> {}{}'.format(label[0], label[1], text_body.split(chr(10) + '[status]')[0][:400], open_note))
        self.drop_wanted(asked_label)
        self.asking = ceiling > self.msgs_sent
        return True

    @staticmethod
    def figure_keys(menu):
        found = set()
        for ln in str(menu).splitlines():
            m = re.match(r"\s*-\s*(.+?)\s+\(in \d+ listings?\)", ln)
            if m and "[figures with a unit]" in ln:
                found.add(prop_key(m.group(1)))
        return found

    def measured_last(self, scripted, menu, asked, remaining, pending):
        if remaining != 1 or pending or not scripted:
            return scripted
        figs = self.figure_keys(menu)
        if not figs or any(prop_key(p) in figs for p in asked):
            return scripted
        first = scripted[0]
        second = str(first.get('second', None) or '').strip(None)
        if prop_key(first.get('property', None)) in figs or not second or not prop_key(second) in figs:
            return scripted
        tied = re.search(r"\(([^)]+)\)", second) or re.search(r"\(([^)]+)\)", str(first.get('property', None) or ""))
        product = self.ask_noun(second) if not tied else tied.group(1).strip(None)
        name = second.split('(')[0].strip(None)
        self.log.append(dict(kind='measured_last', asked=second, instead_of=str(first.get('property', None) or '')[:80]))
        return [dict(property=second, second='', question='Which {} would you like for the {}?'.format(name, product))] + scripted

    def reply_back(self, text_body):
        if not self.role("message") or self.msgs_sent >= self.ask_room() or 4 >= self.moves_left() or self.wire.spare() < CLOSING \
                or (self.asking and self.msgs_sent >= self.ask_room() - 1 and 10 < self.moves_left() and self.wire.spare() >= CLOSING + 40):
            return False
        proposal = self.message_call(text_body, 'reply_{}'.format(self.wire.tick))
        if not proposal:
            return False
        self.chat.append(dict(role='assistant', content="I will answer the shopper's question.", tool_calls=[proposal]))
        self.post_replies(self.dispatch([proposal], scripted=True))
        return True

    def post_replies(self, responses):
        for idx, (cid, text_body) in enumerate(responses, 0):
            if idx == len(responses) - 1:
                text_body += "\n" + self.status_line()
            self.chat.append(dict(role='tool', tool_call_id=cid, content=text_body))

    def plain_args(self, call_name, text_body, kind = ""):
        jschema = self.shapes.get(call_name, None) or {}
        prop_specs = jschema.get('properties', None) or {}
        call_args: dict[str, Any] = {}
        words = ('' if not kind == 'order' else ORDER_WORDS) if not kind == 'decline' else DECLINE_WORDS
        for key in jschema.get('required', None) or []:
            pspec = prop_specs.get(key, None) or {}
            if pspec.get('enum', None):
                fit = [val for val in pspec["enum"] if words and isinstance(val, (str,)) and (re.search(words, val.lower()) or (
                    kind == 'decline' and re.fullmatch(r"\s*(?:none|no|nothing|null)\s*", val.lower())))]
                call_args[key] = pspec['enum'][0] if not fit else fit[0]
            elif 'string' == pspec.get('type', None) or "string" in (pspec.get('type', None) or []):
                call_args[key] = text_body[: pspec.get("maxLength", len(text_body))]
            elif 'boolean' == pspec.get('type', None):
                call_args[key] = True
            elif pspec.get('type', None) in ("integer", "number"):
                call_args[key] = max(1, pspec.get("minimum", 1))
        if 1 == len(prop_specs):
            key, pspec = next(iter(prop_specs.items()))
            if not key in call_args and isinstance(pspec, (dict,)) and 'string' == pspec.get('type', None) and not 'enum' in pspec:
                call_args[key] = text_body[: pspec.get("maxLength", len(text_body))]
        return call_args

    def scrub_decline(self, call_args, call_name):
        if self.needs.reading.get('decline_no_names', None) is False:
            return call_args
        named = any(tok and tok in self.archive.ids
                    for piece in call_args.values() if isinstance(piece, (str,))
                    for tok in re.split(r"[\s.,;:!?()\[\]{}<>\"'`]+", piece))
        if not (self.needs.reading.get('decline_no_names', None) or named):
            return call_args
        prop_specs = (self.shapes.get(call_name, None) or {}).get('properties', None) or {}
        return {key: ((piece if not (isinstance(piece, (str,)) and not 'enum' in (prop_specs.get(key, None) or {})) else SORRY[:(prop_specs.get(key, None) or {}).get('maxLength', len(SORRY))]))
                for key, piece in call_args.items()}

    def code_reading(self):
        text_body = self.request
        low = text_body.lower()
        reading: dict[str, Any] = (dict(code_reading=True, requirements=[], items=[], quantity=1, rules=[], soft_wants=[], ask_first=[], hints=[], references=[], priority_detail=''))
        reading["price_cap"], reading["currency"] = self.code_budget(text_body)
        reading["in_stock_only"] = bool(re.search(r"\bin[- ]stock\b|\bon hand\b|\bon the shelf\b|\bsold[- ]out\b|\bback[- ]?order|"
                                                  r"\bout of stock\b|\bavailable now\b", low))
        reading["priority"] = ('none' if not re.search('\\bcheapest\\b|\\blowest[- ](?:price|priced|cost)\\b|\\bleast expensive\\b|\\blowest\\b[^.]{0,20}\\bprice\\b|\\bsmallest (?:total|price)\\b', low) else 'cheapest')
        said = self.code_limit(low, r"messages?|questions?|notes?|repl(?:y|ies)|texts?")
        if not said is None:
            reading["message_budget"] = said
        moves = self.code_limit(low, r"steps?|moves?|actions?|(?:tool )?calls?|turns?")
        if not moves is None and moves >= FIRST_ORDER_BY + 3:
            reading["step_budget"] = moves
        roles = self.code_roles()
        reading["roles"] = roles
        items = self.code_items(roles.get('search', None))
        if 2 <= len(items):
            reading["items"] = [{key: row[key] for key in ("product", "quantity", "requirements")} for row in items]
            reading["product"] = " and ".join(row["product"] for row in items)
            reading["requirements"] = [req for row in items for req in row["requirements"]]
        else:
            reading["product"] = self.code_label(roles.get('search', None), kind_words(text_body)) if not items else items[0]['label']
            reading["quantity"] = self.code_count(low) if not items else items[0]['quantity']
            reading["requirements"] = ['{}: {}'.format(prop, value) for prop, value in self.code_needs(low)]
        return reading

    @staticmethod
    def code_budget(text_body):
        best: tuple[int, float] = (0, 0.0)
        found: tuple[Any, Any] = (None, None)
        for sentence in re.split(r"(?<=[.!?;])\s+", str(text_body).lower()):
            whole = 0 if not re.search('\\b(?:order|total|whole|lot|sum|bill|altogether|overall|everything)\\b', sentence) else 1
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
                    score = score + 3
                if re.search(r"\b(?:under|below|at most|no more than|not more than|up to|within|max\w*|less than|cheaper than|"
                             r"not (?:more|over|above)|limit\w*|reach\w*|go to|to|at|cap\w*|past|exceed\w*|beyond|go over)\W+(?:[\w']+\W+)?$", before):
                    score = score + 2
                if re.match(r"\s*(?:[a-z.()]+\s*){0,3}?(?:or (?:lower|less|under|below|cheaper)|at (?:the (?:very )?)?most|max\w*|tops\b|"
                            r",?\s*(?:and )?not (?:one|a) \w+ (?:over|more|above))", after):
                    score = score + 2
                figure = float(m.group(2).replace(",", ""))
                if score >= 2 and (score, figure) > best:
                    best = (score, figure)
                    unit = re.match(r"\s*((?:[a-z][a-z.]*\s*){0,2})", after)
                    words = [w.strip(".") for w in ([] if not unit else unit.group(1).split()) if not w.strip('.') in LINK_WORDS
                             and 2 <= len(w.strip('.')) and not re.search(r"\d", w)]
                    sign = (m.group(1) or '').strip(None)
                    found = (figure if not figure.is_integer() else int(figure), " ".join(words) or sign or None)
        return found

    @staticmethod
    def code_limit(low, nouns):
        figures = (dict(one=1, two=2, three=3, four=4, five=5, six=6, seven=7, eight=8, nine=9, ten=10, twelve=12, fifteen=15, twenty=20))
        for m in re.finditer(r"\b(\d+|one|two|three|four|five|six|seven|eight|nine|ten|twelve|fifteen|twenty)\s+(?:[a-z-]+\s+){0,2}?(?:"
                             + nouns + r")\b", low):
            before, after = low[max(0, m.start() - 60): m.start()], low[m.end(): m.end() + 40]
            if re.search(r"\b(?:at most|no more than|not more than|up to|only|max\w*|use|using|take|send|allow\w*|receive|get|total of|"
                         r"limit\w*|inside|within|under|fewer than|less than|you've|you have|have|keep\w*|stick to|go past|past|"
                         r"exceed\w*|beyond)\W+(?:[\w']+\W+){0,3}$", before) \
                    or re.match(r"\W*(?:[\w']+\W+){0,3}?(?:in total|total|no more|at most|max\w*|tops|or fewer|or less|allowed|limit)\b", after):
                return figures[m.group(1)] if not m.group(1).isdigit() else int(m.group(1))
        return None

    @staticmethod
    def code_count(low):
        figures = dict(one=1, two=2, three=3, four=4, five=5, six=6, seven=7, eight=8, nine=9, ten=10)
        said = re.search(r"\b(\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s+(?:units?|of (?:them|those|these)|copies)\b", low)
        if not said:
            return 1
        n = figures[said.group(1)] if not said.group(1).isdigit() else int(said.group(1))
        return 1 if not 1 <= n <= 99 else n

    @staticmethod
    def code_needs(seg):
        out: list[tuple[str, str]] = []
        verbs = r"(?:is|are|has to be|have to be|must be|should be|needs to be|need to be|can be)"
        stop = r"(?=\s*(?:[,.;:!?()]|$)|\s+(?:so|since|as|because|for|to|which|that|nothing|otherwise|but|when|while|if|then|though|please|and (?:the|its|their|a|an|one|two|three|\d))\b)"
        pattern = (r"(?:\b(?:where|whose)\s+(?:the\s+|its\s+|their\s+)?|(?:\b(?:and|with|but)\s+|[,;:]\s*|^|[.!?]\s+)(?:the|its|their)\s+)"
                   r"([a-z][a-z'&/ -]{1,32}?)\s+" + verbs + r"\s+(?:either\s+|just\s+)?([^,.;:!?()]{1,60}?)" + stop)
        for m in re.finditer(pattern, seg):
            prop, value = m.group(1).strip(None), m.group(2).strip(None)
            if ref_text(value) or ref_text(seg[m.end(): m.end() + 60]) and re.search(r"\b(?:page|listing|shown|sold by)\b", seg[m.end(): m.end() + 60]):
                continue
            if not value or 4 < len(value.split()) or 4 < len(prop.split()):
                continue
            if re.search(r"\b(?:i|you|we|it|he|she|they|my|your|our|there|this|that|which|what|who|me|us|them|one|ones|price|budget|cost|total|order|stock)\b", prop):
                continue
            if re.match(r"(?:a|an|the|my|your|our|his|her|their|me|us|for|to|in|on|at|of|see|go|get|be|have|only|too|very|so|not|no|what|how|it|all|"
                        r"mostly|always|never|just|really|still|also|from|about|like|one|ones)\b", value):
                continue
            out.append((prop, value))
        return out

    def code_items(self, tool):
        props = (self.shapes.get(tool or '', None) or {}).get('properties', None)
        lists = [spec["enum"] for spec in ([] if not isinstance(props, (dict,)) else props.values()) if isinstance(spec, (dict,))
                 and isinstance(spec.get('enum', None), (list,)) and 4 <= len(spec['enum'])]
        labels = [] if not lists else [str(label) for label in max(lists, key=len)]
        low = self.request.lower()
        toks = [(m.group(0).rstrip("s"), m.start(), m.end()) for m in re.finditer(r"[a-z]{3,}", low) if not m.group(0) in LINK_WORDS]
        words = [t[0] for t in toks]
        figures = dict(one=1, two=2, three=3, four=4, five=5, six=6, seven=7, eight=8, nine=9, ten=10)
        counted: dict[str, tuple[int, int, int]] = {}
        for label in labels:
            own = [w.rstrip("s") for w in re.findall(r"[a-z]{3,}", label.lower()) if not w in LINK_WORDS]
            if not own:
                continue
            for i in range(0, len(words) - len(own) + 1):
                if words[i: i + len(own)] != own:
                    continue
                start, end = toks[i][1], toks[i + len(own) - 1][2]
                if len(re.findall(r"[a-z0-9]+", low[start:end])) > 2 * len(own):
                    continue
                count = re.search(r"\b(\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s*(?:x\s+|\u00d7\s*)?(?:(?:of|[a-z'&-]{3,})\s+){0,4}$",
                                  low[max(0, start - 56): start])
                if count and not label in counted:
                    n = figures[count.group(1)] if not count.group(1).isdigit() else int(count.group(1))
                    if 1 <= n <= 99:
                        counted[label] = (start, end, n)
                    break
        ordered = sorted(counted.items(), key=lambda kv: kv[1][0])
        out = []
        for j, (label, (start, end, n)) in enumerate(ordered, 0):
            stop = len(low) if not j + 1 < len(ordered) else ordered[j + 1][1][0]
            seg = low[end: stop]
            seg = seg if not j + 1 == len(ordered) else re.split('(?<=[.!?])\\s', seg)[0]
            name = label.lower()
            out.append((dict(product=name, label=label, quantity=n, requirements=['{} ({}): {}'.format(prop, name, value) for prop, value in self.code_needs(seg)])))
        return out

    def code_roles(self):
        said: dict[str, str] = {}
        for tdef in self.toolset:
            fn = tdef["function"]
            said[fn["name"]] = re.sub(r"([a-z])([A-Z])", r"\1 \2", fn["name"]).replace("_", " ").lower() + " | " \
                + str(fn.get('description', None) or "").lower()

        def props(name):
            found = (self.shapes.get(name, None) or {}).get('properties', None)
            return {} if not isinstance(found, (dict,)) else found

        def needed(name):
            found = (self.shapes.get(name, None) or {}).get('required', None)
            return [] if not isinstance(found, (list,)) else [key for key in found if key in props(name)]

        def texts(name):
            return [key for key, spec in props(name).items() if isinstance(spec, (dict,)) and 'string' == spec.get('type', None) and not spec.get('enum', None)]
        roles: dict[str, str] = {}
        for name in said:
            for key in needed(name):
                vals = [] if not isinstance(props(name)[key], (dict,)) else [str(v).lower() for v in props(name)[key].get('enum', None) or []]
                if any(re.search(ORDER_WORDS, v) for v in vals) and any(re.search(DECLINE_WORDS, v) for v in vals):
                    roles.setdefault("order", name)
                    roles.setdefault("decline", name)
        for name in said:
            head = said[name].split(" | ")[0]
            if name in roles.values():
                continue
            if not 'decline' in roles and re.search(DECLINE_WORDS, head):
                roles["decline"] = name
            elif not 'order' in roles and re.search(ORDER_WORDS, head):
                roles["order"] = name
        for name in said:
            if name in roles.values() or needed(name):
                continue
            if not 'decline' in roles and re.search(r"\bfinal\b", said[name]) and re.search(DECLINE_WORDS, said[name]):
                roles["decline"] = name
            elif not 'order' in roles and re.search(r"\bfinal\b", said[name]) and re.search(ORDER_WORDS, said[name]):
                roles["order"] = name
        basket = r"\b(?:cart|basket|bag|trolley)\b"
        for name in said:
            if name in roles.values() or not re.search(basket, said[name]):
                continue
            if needed(name) or any(isinstance(spec, (dict,)) and 'array' == spec.get('type', None) for spec in props(name).values()):
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
                if role == 'search' and re.search(person, said[name]) and re.search(speak, said[name]):
                    continue
                if role == 'message' and not (re.search(person, said[name]) or re.search(r"\b(?:messag\w*|chat\w*)\b", said[name])):
                    continue
                if one_text and not (1 == len(needed(name)) and needed(name)[0] in texts(name)):
                    continue
                if not one_text and not (texts(name) or any(isinstance(spec, (dict,)) and spec.get('enum', None) for spec in props(name).values())):
                    continue
                roles[role] = name
        return roles

    def code_label(self, tool, want):
        props = (self.shapes.get(tool or '', None) or {}).get('properties', None)
        lists = [spec["enum"] for spec in ([] if not isinstance(props, (dict,)) else props.values()) if isinstance(spec, (dict,))
                 and isinstance(spec.get('enum', None), (list,)) and 4 <= len(spec['enum'])]
        best, top = "", (0, 0.0)
        for label in [] if not lists else max(lists, key=len):
            words = kind_words(label)
            hit = len(words & want)
            if hit and (hit, hit / max(1, len(words))) > top:
                best, top = str(label), (hit, hit / max(1, len(words)))
        return best

    def code_values(self, k):
        rows = self.needs.items()
        out = []
        for req in self.needs.props_left():
            head, _, value = str(req).partition(":")
            if rows and not self.product_of(head) in (k, None):
                continue
            plain = re.sub('\\((?:stated|revealed[^)]*)\\)', '', value).strip(None)
            short = 5 >= len(re.findall('[a-z0-9]+', plain.lower())) and not re.search(EXCLUDE_WORDS, plain.lower())
            if (positive_value(value, head) or short) and plain and not ref_text(plain):
                out.append((re.split('\\s*\\(', head)[0].strip(None), plain))
        return out

    def code_queries(self, k):
        tool = self.role("search")
        jschema = self.shapes.get(tool or '', None) or {}
        props = {} if not isinstance(jschema.get('properties', None), (dict,)) else jschema.get('properties', None)
        needed = [key for key in jschema.get('required', None) or [] if key in props]
        rows = self.needs.items()
        product = str(self.needs.reading.get('product', None) or '' if not (rows and not k is None and (k < len(rows))) else rows[k]['product'])
        label = self.code_label(tool, kind_words(product) or kind_words(self.request))
        told = next((str(tdef['function'].get('description', None) or "") for tdef in self.toolset if tdef["function"]["name"] == tool), "")
        parts = [part.lower() for part in re.split(r"[.;]", told)]
        tok = self.token_key()
        texts = [key for key, spec in props.items() if isinstance(spec, (dict,)) and 'string' == spec.get('type', None) and not spec.get('enum', None) and tok != key]
        text_key = next((key for key in texts if re.search(r"^q$|query|text|keyword|needle|term|word|phrase|string|search", key.lower())),
                        None if not texts else texts[0])
        enums = {key: spec["enum"] for key, spec in props.items() if isinstance(spec, (dict,)) and isinstance(spec.get('enum', None), (list,)) and spec["enum"]}
        shelf_key = max((key for key in enums if 4 <= len(enums[key])), key=lambda key: len(enums[key]), default=None)
        modes = [key for key in enums if shelf_key != key and 4 >= len(enums[key])]
        mode_key = next((key for key in modes if key in needed), None if not modes else modes[0])

        def mode_for(arg, last):
            vals = enums.get(mode_key or "", [])
            for val in vals:
                if any(re.search(r"\b" + re.escape(str(val).lower()) + r"\b", part) and re.search(r"\b" + re.escape(arg.lower()) + r"\b", part)
                       for part in parts):
                    return val
            return vals[0] if not last else vals[-1]

        def price_word(key):
            return bool(re.search(r"ceil|max|price|cap$|budget|upper|most", key.lower())) or any(
                key.lower() in part and re.search(r"\bpric|\bcost", part) for part in parts)
        cap_key = next((key for key, spec in props.items() if isinstance(spec, (dict,)) and spec.get('type', None) in ("integer", "number")
                        and not spec.get('enum', None) and price_word(key) and not re.search(r"page|first|offset|skip|count|size|num", key.lower())), None)
        cap = self.needs.spend_cap
        qty = (self.needs.quantity() if not (not k is None and k < len(self.needs.counts())) else self.needs.counts()[k]) or 1
        bound = ({} if not (cap_key and is_number(cap) and (cap > 0)) else {cap_key: round(cap / qty, 2) if not 'integer' == props[cap_key].get('type', None) else int(cap // qty)})
        cap_len = None if not (text_key and isinstance(props.get(text_key, None), (dict,))) else props.get(text_key, {}).get('maxLength', None)
        names = [w for w in re.findall(r"[a-z0-9]+", product.lower()) if 2 < len(w) and not w in LINK_WORDS]
        kind = '' if not names else names[-1]
        firm = [value for _, value in self.code_values(k) if not re.search(r"\bor\b|/|,|\b(?:" + VALUE_LINKS + r")\b", value.lower())]
        queries: list[str] = []
        if firm:
            vw = [w for w in re.findall(r"[a-z0-9]+", firm[0].lower()) if not w in LINK_WORDS]
            if vw:
                queries.append(" ".join(dict.fromkeys(vw + [kind])))
                if 2 <= len(vw):
                    queries.append(" ".join(dict.fromkeys([max(vw, key=len), kind])))
        if not queries:
            queries.append(' '.join(sorted(kind_words(label))[:2]) if not names else ' '.join(dict.fromkeys(names[:1] + names[-1:])))
        if is_number(cap_len):
            fitted = []
            for query in queries:
                while len(query) > cap_len and " " in query:
                    query = query.rsplit(" ", 1)[0]
                fitted.append(query[: int(cap_len)])
            queries = fitted
        calls: list[dict[str, Any]] = []
        if text_key and queries[0]:
            calls.append({text_key: queries[0], **({} if not mode_key else {mode_key: mode_for(text_key, False)}), **bound})
        if label and shelf_key:
            calls.append({shelf_key: label, **({} if not mode_key else {mode_key: mode_for(shelf_key, True)}), **bound})
        if text_key and 1 < len(queries):
            calls.append({text_key: queries[1], **({} if not mode_key else {mode_key: mode_for(text_key, False)}), **bound})
        out: list[dict[str, Any]] = []
        for call_args in calls:
            for key in needed:
                spec = {} if not isinstance(props[key], (dict,)) else props[key]
                if key in call_args:
                    continue
                if spec.get('enum', None):
                    call_args[key] = spec["enum"][0]
                elif spec.get('type', None) in ("integer", "number"):
                    call_args[key] = spec.get("default", spec.get("minimum", 1))
                elif 'string' == spec.get('type', None):
                    call_args[key] = queries[0] or label
            fixed, faults = ArgCheck.execute(call_args, jschema)
            if faults and bound:
                fixed, faults = ArgCheck.execute({key: val for key, val in call_args.items() if not key in bound}, jschema)
            if not faults and not fixed in out:
                out.append(fixed)
        return out

    def code_rows(self):
        out: dict[str, dict[str, Any]] = {}

        def visit(d):
            ident = self.row_mark(d)
            if not ident in self.archive.ids:
                return
            info = out.setdefault(ident, dict(shelf='', title='', price=None, stock=None, words=set(), fields={}))
            info["shelf"] = self.shelf_of(d) or info["shelf"]
            info["title"] = next((str(val) for key, val in d.items() if isinstance(val, (str,)) and re.search(r"title|name", str(key).lower())
                                  and not 'id' in str(key).lower()), "") or info["title"]
            for key, val in d.items():
                name = str(key).lower()
                if is_number(val) and not isinstance(val, (bool,)) and re.search(r"price|cost|amount", name) \
                        and not re.search(r"count|qty|quant|total|units|line|due|payable|sum", name):
                    info["price"] = float(val)
                    break
            for key, val in d.items():
                state = self.key_stock(key, val)
                if not state is None:
                    info["stock"] = state
            pieces = [str(v) for v in d.values() if isinstance(v, (str, int, float)) and not isinstance(v, (bool,))]
            pieces += [str(x) for v in d.values() if isinstance(v, (dict,)) for x in v.values() if isinstance(x, (str, int, float))]
            info["words"] |= kind_words(" ".join(pieces))
            for v in d.values():
                if isinstance(v, (dict,)):
                    info["fields"].update({str(key): str(x) for key, x in v.items() if isinstance(x, (str, int, float)) and not isinstance(x, (bool,))})
        for rec in self.archive.items:
            walk_dicts(rec, visit)
        return out

    @staticmethod
    def code_shows(row, prop, value, hit_words, read):
        named = [val for key, val in row["fields"].items() if same_prop(key, prop)]
        if named:
            return any(Errand.code_seen(value, val) for val in named)
        if Errand.code_seen(value, row.get('title', None) or ""):
            return True
        own = kind_words(value)
        return not read and bool(own) and hit_words >= own

    @staticmethod
    def code_seen(value, text):
        pool = str(text or "").lower()
        if not pool.strip(None):
            return False
        body = re.sub(r"\b(?:" + VALUE_LINKS + r")\b", " ", str(value).lower())
        for alt in re.split(r"\s*(?:\bor\b|/|,|;)\s*", body):
            words = [w for w in re.findall(r"[a-z]{2,}", alt) if not w in LINK_WORDS]
            figs = re.findall(r"\d+(?:\.\d+)?", alt)
            if (words or figs) and all(re.search(r"\b" + re.escape(w if not 4 < len(w) else w[:-1]), pool) for w in words) \
                    and all(re.search(r"(?<![\d.])" + re.escape(f) + r"(?![\d.])", pool) for f in figs):
                return True
        return False

    def code_picks(self, avoid = None, keep = None):
        rows = self.needs.items()
        counts = self.needs.counts()
        homes = [set(self.shelf_labels()) or (self.home_shelves() or [set()])[0]] if not rows else self.home_shelves()
        info = self.code_rows()
        ver = str(self.needs.version)
        failed = {key.split("|")[0] for key, val in self.rulings.items() if val and key.split("|")[1:2] == [ver]}
        live = {key.rpartition("|")[0] for key in self.passed if key.rpartition("|")[2] == ver}
        reviewed = self.__dict__.get('audit_ids', None)
        if self.__dict__.get('audit_done', None) and (reviewed is None or len(self.archive.read) <= self.__dict__.get("audit_reads", -1)):
            reviewed = set(self.archive.read)
        answered = next((isinstance(e.get('result', None), (dict,)) for e in reversed(self.log) if isinstance(e, (dict,)) and 'audit' == e.get('kind', None)), False)
        audited = set() if not answered else set(reviewed or set()) - set(self.__dict__.get('audit_fits', None) or []) - live
        own_all = bool(audited) and self.code_judges_all()
        cap = None if not is_number(self.needs.spend_cap) else float(self.needs.spend_cap)

        def value_rank(row, prop, value):
            want = {" ".join(ch_words(alt)) for alt in re.split(r"\s*(?:\bor\b|/)\s*", value.lower()) if ch_words(alt)}
            best = (2, 3)
            for key, val in row["fields"].items():
                if not same_prop(key, prop) or not Errand.code_seen(value, val):
                    continue
                tier = (3 if not any((' '.join(ch_words(p)) in want for p in ch_parts(val))) else 1) if not ' '.join(ch_words(val)) in want else 0
                best = min(best, (1 if not prop_key(key) == prop_key(prop) else 0, tier))
            if best == (2, 3) and Errand.code_seen(value, row.get('title', None) or ""):
                best = (2, 2)
            return best
        asked = kind_words(self.request)
        hits = self.__dict__.get('code_hits', None) or {}
        cheapest = 'cheapest' == self.needs.priority()
        avoid, keep = set(avoid or ()), set(keep or ())
        avoid |= self.dead_lines()
        avoid |= self.ref_pointers()
        pools: list[list[tuple]] = []
        for k, home in enumerate(homes, 0):
            qty = 1 if not k < len(counts) else counts[k]
            want = kind_words(self.needs.reading.get('product', None) or '' if not rows else rows[k]['product'])
            vals: set[str] = set()
            stated: list[tuple[str, str]] = []
            judged: list[tuple[str, str]] = []
            for prop, plain in self.code_values(None if not rows else k):
                vals |= kind_words(plain)
                if re.search(r"\b(?:" + VALUE_LINKS + r")\b|\d\s*(?:-|to)\s*\d|[+<>]|\b(?:about|around|approx\w*|roughly)\b", plain.lower()):
                    continue
                judged.append((prop, plain))
                stated.append((prop, plain))
            pool = []
            apart = [(ref["prop"], [str(v) for v in [ref.get('value', None)] + list(ref.get('goes_with', None) or []) if str(v or '').strip(None)])
                     for ref in self.differ_done(None if not rows else k)]
            for ident, row in info.items():
                if ident in failed or ident in self.refused_lines or ident in avoid or row["stock"] is False:
                    continue
                if (home and not row['shelf'] in home) or (not home and not (want & row["words"])):
                    continue
                if not self.__dict__.get('mv_relax', None) and any(self.code_clash(row["fields"], prop, value) for prop, value in stated):
                    continue
                if self.ref_excluded(row["fields"], None if not rows else k):
                    continue
                cost = self.archive.price(ident) if not not row['price'] is None else row['price']
                if not cap is None and is_number(cost) and cost * qty > cap:
                    continue
                fits = all(self.code_shows(row, prop, value, hits.get(ident, None) or set(), ident in self.archive.read) for prop, value in judged)
                fits = fits and all(any(same_prop(key, prop) and str(val).strip(None) and not any(ch_writes(v, val) for v in bad)
                                        for key, val in row["fields"].items()) for prop, bad in apart)
                shows = [] if not ident in self.archive.read else [value_rank(row, prop, value) for prop, value in judged]
                if ident in audited and not (own_all and fits and (judged or apart) and all(1 >= max(s) for s in shows)):
                    continue
                score = 3 * len(vals & row["words"]) + 2 * len(want & row["words"]) + len(asked & row["words"]) \
                    + 4 * len(vals & (hits.get(ident, None) or set()))
                price = float('inf') if not is_number(cost) else cost
                quality = (9, 9) if not shows else (sum((s[0] for s in shows)), sum((s[1] for s in shows)))
                pool.append((1 if not ident in keep else 0, 1 if not ident in live else 0, not fits, not cap is None and not is_number(cost),
                             0.0 if not cheapest else price, 1 if not ident in self.archive.read else 0, not row['stock'] is True,
                             (quality, -score), price, ident))
            if not pool:
                return []
            pools.append(sorted(pool))
        for rank in ((lambda e: e), (lambda e: (e[2], e[3], e[8], e[-1]))):
            picks: list[tuple[str, int]] = []
            total, chosen = 0.0, []
            for k, pool in enumerate(pools, 0):
                entry = next((e for e in sorted(pool, key=rank) if all(e[-1] != p for p, _ in picks)
                              and not (e[-1] in self.archive.read and self.field_view(e[-1])["fails"])), None)
                if entry is None:
                    return []
                picks.append((entry[-1], 1 if not k < len(counts) else counts[k]))
                chosen.append(entry)
                total += (0.0 if not entry[8] != float('inf') else entry[8]) * picks[-1][1]
            if cap is None or cap >= total:
                self.code_fit = {e[-1]: not e[2] for e in chosen}
                return picks
        return []

    @staticmethod
    def code_clash(fields, prop, value):
        named = [val for key, val in fields.items() if same_prop(key, prop)]
        return bool(named) and not any(value_seen(value, val) for val in named)

    def cart_form(self):
        jschema = self.shapes.get(self.role('cart') or '', None) or {}
        props = {} if not isinstance(jschema.get('properties', None), (dict,)) else jschema.get('properties', None)
        known = None if not self.trolley.put_form else json_object(self.trolley.put_form)

        def words(v):
            return set(re.findall(r"[a-z]+", re.sub(r"([a-z])([A-Z])", r"\1 \2", str(v)).lower()))
        put: dict[str, Any] = {}
        drop: dict[str, Any] = {}
        for key in jschema.get('required', None) or []:
            spec = {} if not isinstance(props.get(key, None), (dict,)) else props.get(key, None)
            vals = [] if not isinstance(spec.get('enum', None), (list,)) else spec.get('enum', None)
            if not vals:
                continue
            puts = [v for v in vals if words(v) & PUT_WORDS and not words(v) & DROP_WORDS]
            takes = [v for v in vals if words(v) & DROP_WORDS]
            put[key] = (vals[0] if not puts else puts[0]) if not (isinstance(known, (dict,)) and known.get(key, None) in vals) else known[key]
            others = [v for v in vals if v != put[key]]
            drop[key] = next((v for v in takes if v != put[key]), None if not 2 == len(vals) else others[0])
        texts = [key for key, spec in props.items() if isinstance(spec, (dict,)) and 'string' == spec.get('type', None) and not spec.get('enum', None)]
        id_key = next((key for key in texts if re.search(r"id|item|product|listing|sku|target|which|number", key.lower())),
                      None if not texts else texts[0])
        counts = [key for key, spec in props.items() if isinstance(spec, (dict,)) and spec.get('type', None) in ("integer", "number") and not spec.get('enum', None)]
        return jschema, put, drop, id_key, counts

    def cart_args(self, picks):
        jschema, base, _, id_key, counts = self.cart_form()
        spec_lines = self.trolley.line_spec()
        calls: list[dict[str, Any]] = []
        if not spec_lines:
            if not id_key:
                return []
            for ident, qty in picks:
                call_args = {**base, id_key: ident}
                if counts:
                    call_args[counts[0]] = qty
                calls.append(call_args)
        else:
            arr, id_field, count_field = spec_lines
            calls.append({**base, arr: [{id_field: ident, count_field: qty} for ident, qty in picks]})
        out = []
        for call_args in calls:
            fixed, faults = ArgCheck.execute(call_args, jschema)
            if faults:
                return []
            out.append(fixed)
        return out

    def cart_drops(self, lines):
        if not lines or self.trolley.line_spec():
            return []
        jschema, put, drop, id_key, counts = self.cart_form()
        props = {} if not isinstance(jschema.get('properties', None), (dict,)) else jschema.get('properties', None)
        if not id_key:
            return None
        out = []
        for ident in lines:
            if put:
                if any(drop.get(key, None) is None for key in put):
                    return None
                call_args = {**drop, id_key: ident}
                if counts:
                    call_args[counts[0]] = max(1, int(self.trolley.held(ident) or 1))
            elif counts and (not is_number((props.get(counts[0], None) or {}).get('minimum', None)) or 0 >= props[counts[0]]['minimum']):
                call_args = {id_key: ident, counts[0]: 0}
            else:
                return None
            fixed, faults = ArgCheck.execute(call_args, jschema)
            if faults:
                return None
            out.append(fixed)
        return out

    def code_replies(self, n):
        entry = next((e for e in reversed(self.log) if isinstance(e, (dict,)) and 'env' == e.get('kind', None)), None)
        resp = (entry or {}).get('reply', None)
        if not isinstance(resp, (dict,)):
            return []
        obs_list = [resp.get('observation', None)] if not isinstance(resp.get('observations', None), (list,)) else resp.get('observations', None)
        return list(obs_list)[:n]

    @staticmethod
    def code_fault(obs):
        found = []

        def visit(d):
            if any(re.search(FAULT_KEYS, str(key).lower()) and not val in (None, False, '', [], {}) for key, val in d.items()):
                found.append(d)
        walk_dicts(obs, visit, 0, 2)
        return bool(found)

    def code_search(self, products, more = False):
        search = self.role("search")
        texts = [key for key, spec in ((self.shapes.get(search or '', None) or {}).get('properties', None) or {}).items()
                 if isinstance(spec, (dict,)) and 'string' == spec.get('type', None) and not spec.get('enum', None)]
        reserve = 2 + 2 * products
        if not more:
            lists = [self.code_queries(None if not self.needs.items() else k) for k in range(0, products)]
            calls = []
            for rank in range(0, max((len(q) for q in lists), default=0)):
                calls += [q[rank] for q in lists if rank < len(q) and not q[rank] in calls]
            calls = calls[:max(0, self.moves_left() - reserve)]
            calls = calls[:self.mv_left('search')]
            lead = [q[0] for q in lists if q]
            rounds = 1 + CODE_PAGES + (0 if not 'cheapest' == self.needs.priority() else 2)
        else:
            lead = list(self.__dict__.get('code_more', None) or [])
            calls = lead[:max(0, self.moves_left() - reserve - 2 * products)]
            calls = calls[:self.mv_left('search')]
            rounds = 1
        if not calls:
            return
        self.code_searched = True
        for rnd in range(0, rounds):
            self.send_chunk([('code_search_{}_{}'.format(n, self.wire.tick), search, call_args) for n, call_args in enumerate(calls, 0)])
            replies = self.code_replies(len(calls))
            if 'any' != self.__dict__.get('word_kind', None):
                hits = self.__dict__.setdefault("code_hits", {})
                for call_args, obs in zip(calls, replies):
                    said = " ".join(str(call_args.get(key, None) or "") for key in texts)
                    if not said.strip(None):
                        continue
                    words = kind_words(said)

                    def visit(d):
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
            fit = self.__dict__.get('code_fit', None) or {}
            if 'cheapest' != self.needs.priority() and picks and all(fit.get(ident, None) for ident, _ in picks):
                break
            lead = following
            calls = following[: max(0, self.moves_left() - reserve - 3 * products)]
            calls = calls[: self.mv_left("search")]
            if not calls:
                break

    def code_next(self, call_args, obs):
        spec = self.page_arg()
        if not spec:
            return None
        key, by_page, low, high = spec
        pages: list[dict[str, Any]] = []

        def visit(d):
            if any(isinstance(v, (list,)) and v and all(isinstance(x, (dict,)) for x in v) for v in d.values()):
                pages.append(d)
        walk_dicts(obs, visit, 0, 3)
        if not pages:
            return None
        page = pages[0]
        rows = max(len(v) for v in page.values() if isinstance(v, (list,)))
        marks = [v for k, v in page.items() if re.search(r"next|onward|cursor|bookmark|token|continu|more", str(k).lower())]
        total = next((v for k, v in page.items() if is_number(v) and not isinstance(v, (bool,))
                      and re.search(r"count|total|matched|found|hits", str(k).lower())), None)
        if marks and all(m in (None, False, "", 0) for m in marks):
            return None
        if not by_page == 'token':
            cur = call_args.get(key, low)
            cur = low if not is_number(cur) else cur
            nxt = cur + rows if not by_page else cur + 1
            if not high is None and high < nxt or (not marks and (total is None or (cur + rows if not by_page else (cur - low + 1) * rows) >= total)):
                return None
            nxt_args = {**call_args, key: nxt if not float(nxt).is_integer() else int(nxt)}
        else:
            tok = next((m for m in marks if isinstance(m, (str,)) and m.strip(None)), None)
            nxt_args = None if not tok else {**call_args, key: tok}
        if not nxt_args:
            return None
        fixed, faults = ArgCheck.execute(nxt_args, self.shapes.get(self.role('search') or '', None) or {})
        return fixed if not faults else None

    def code_read(self, picks, avoid, keep):
        for _ in range(0, CODE_READS):
            unread = [ident for ident, _ in picks if not ident in self.archive.read]
            batch = self.batch_tool()
            reads = len(unread) if not (batch and 2 <= len(unread)) else 1
            if not unread or not (batch or self.role("details")) or self.moves_left() < reads + 2 * len(picks) + 3 or 8 >= self.wire.spare():
                break
            if not (batch and 2 <= len(unread)):
                slab = [('code_read_{}_{}'.format(n, self.wire.tick), self.role('details'), call_args) for n, call_args in enumerate((self.details_call(ident) for ident in unread), 0) if call_args]
            else:
                slab = [('code_read_{}'.format(self.wire.tick), batch[0], {batch[1]: unread[:batch[2]]})]
            if not slab:
                break
            self.send_chunk(slab)
            fresh = self.code_picks(avoid, keep)
            if picks == fresh:
                break
            picks = fresh
        return picks

    def code_put(self, picks):
        cart = self.role("cart")
        want = {ident for ident, _ in picks}
        spec_lines = self.trolley.line_spec()
        extra = [line for line in self.trolley.units if not line in want]
        stated = any(t.get('before_change', None) is True for t in self.needs.told)
        unread = any(t.get('before_change', None) is None for t in self.needs.told)
        if extra and (stated or unread):
            if not self.code_asks_swap(extra, [ident for ident, _ in picks]) and stated:
                return None
        todo = [(ident, qty) for ident, qty in picks if 1e-09 < abs(self.trolley.held(ident) - qty)] if not spec_lines else list(picks)
        drops = self.cart_drops(extra)
        puts = [] if not todo else self.cart_args(todo)
        if drops is None or (todo and not puts):
            return None
        calls = drops + puts
        if not calls:
            return []
        if self.moves_left() < len(calls) + 1:
            return None
        self.send_chunk([('code_cart_{}_{}'.format(n, self.wire.tick), cart, call_args) for n, call_args in enumerate(calls, 0)])
        replies = self.code_replies(len(calls))
        bad: set[str] = set()
        shown = None
        for call_args, obs in zip(calls, replies):
            seen = self.trolley.view_units(obs)
            if not seen is None:
                shown = seen
            if self.code_fault(obs):
                named = [ident for ident in want if ident in as_text(obs)]
                lines = [line for line, _ in self.trolley.lines(call_args) if line in want]
                bad.update(named or lines or ([] if not spec_lines else list(want)))
        if len(replies) < len(calls):
            return None
        if shown is None and not bad and self.role("cart_view") and 3 <= self.moves_left():
            self.send_chunk([('code_view_{}'.format(self.wire.tick), self.role("cart_view"), {})])
            view = self.code_replies(1)
            shown = None if not view else self.trolley.view_units(view[0])
            if view and shown is None:
                bad.update(ident for ident in want if not ident in self.trolley.units)
        if not shown is None:
            bad.update(ident for ident in want if not ident in shown)
        for ident in bad:
            self.trolley.units.pop(ident, None)
        return sorted(bad)

    def code_recheck(self, lines):
        tool = self.stock_tool() or self.role("view") or self.role("details")
        if not lines or not tool or self.moves_left() < len(lines) + 2 or 8 > self.wire.spare():
            return False, set()
        slab = []
        for n, ident in enumerate(lines, 0):
            call_args = self.lookup_call(ident, tool)
            if call_args:
                slab.append(('code_check_{}_{}'.format(n, self.wire.tick), tool, call_args))
        if not slab:
            return False, set()
        self.send_chunk(slab)
        info = self.code_rows()
        return True, {ident for ident in lines if (info.get(ident, None) or {}).get('stock', None) is False}

    def code_news(self, lines):
        waiting = [ln for lns, _ in self.needs.__dict__.get('pending', None) or [] for ln in lns]
        seen = self.__dict__.setdefault("news_read", set())
        for line in list(lines) + waiting:
            if line in seen:
                continue
            seen.add(line)
            figure, _ = self.code_budget(line)
            if is_number(figure) and figure > 0:
                self.needs.spend_cap = figure
                self.log.append(dict(kind='code_news', budget=figure))

    def code_cart(self, avoid = None, keep = None):
        cart, search = self.role("cart"), self.role("search")
        if not cart or self.wire.done or self.wire.broke or 2 > self.moves_left() or 4 > self.wire.spare():
            return False
        ver = str(self.needs.version)
        avoid = set(avoid or ()) | {key.split("|")[0] for key, val in self.rulings.items() if val and key.split("|")[1:2] == [ver]}
        self.wire.mute = True
        try:
            avoid = set(avoid or ())
            products = max(1, len(self.needs.items()))
            for _ in range(0, 4):
                picks = self.code_picks(avoid, keep)
                unfit = not picks or not all(self.__dict__.get('code_fit', {}).get(ident, None) for ident, _ in picks)
                if search and not self.__dict__.get('code_searched', None) and 4 <= self.moves_left() and unfit:
                    self.code_search(products)
                    picks = self.code_picks(avoid, keep)
                elif search and self.__dict__.get('code_more', None) and self.moves_left() >= 6 + 2 * products and (
                        unfit or 'cheapest' == self.needs.priority()):
                    self.code_search(products, True)
                    picks = self.code_picks(avoid, keep)
                picks = self.code_read(picks, avoid, keep)
                if not picks:
                    self.log.append(dict(kind='code_finish', picks=[], result='nothing to order'))
                    return False
                bad = self.code_put(picks)
                if bad is None:
                    self.log.append(dict(kind='code_finish', picks=picks, result='no cart call'))
                    return False
                if not bad:
                    _, sold = (False, set()) if not self.moves_left() >= len(picks) + 3 else self.code_recheck([ident for ident, _ in picks])
                    bad = sorted(sold)
                if not bad:
                    self.log.append(dict(kind='code_finish', picks=picks, cart=dict(self.trolley.units)))
                    return bool(self.trolley.units)
                avoid |= set(bad)
                self.log.append(dict(kind='code_swap', lines=bad))
            return False
        except (Exception,) as err:
            self.log.append(dict(kind='code_finish_error', error='{}: {}'.format(type(err).__name__, err)[:300]))
            return False

    def cart_short(self):
        rows = self.needs.items()
        if 2 > len(rows) or not self.trolley.units:
            return False
        got = {self.product_index(i) for i in self.trolley.units}
        return not None in got and any(not k in got for k in range(0, len(rows)))

    def finish_now(self):
        if self.wire.done or self.wire.broke or 0 >= self.moves_left() or 2 > self.wire.spare():
            return
        call_rank, decl = self.role("order"), self.role("decline")
        avoid: set[str] = set()
        keep: set[str] | None = None
        move: dict[str, Any] | None = None
        kind = None
        for turn in range(0, CODE_FINISHES):
            if self.wire.done or self.wire.broke or 0 >= self.moves_left() or 2 > self.wire.spare():
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
                            self.log.append(dict(kind='code_dead', lines=sorted(gone)))
                    except (Exception,) as err:
                        self.log.append(dict(kind='cart_gone_error', error='{}: {}'.format(type(err).__name__, err)[:300]))
                        gone = set()
                ver_now = str(self.needs.version)
                failed_all = bool(self.trolley.units) and all(self.rulings.get('{}|{}'.format(i, ver_now), None) for i in self.trolley.units)
                if failed_all:
                    avoid |= set(self.trolley.units)
                    keep = set()
                    self.log.append(dict(kind='finish_failed_cart', lines=sorted(self.trolley.units)))
                short_rule = 2 <= len(self.needs.items()) and 'order' != self.__dict__.get('no_fit', None)
                if call_rank and turn == 0 and not gone and not failed_all and self.trolley.used and not (short_rule and self.cart_short()) \
                        and (self.trolley.units or self.trolley.view_units(self.trolley.last_view) is None):
                    move = dict(name=call_rank, args=self.plain_args(call_rank, 'Order the cart.', 'order'))
                elif call_rank and self.code_cart(avoid, keep) and not (short_rule and self.cart_short()):
                    move = dict(name=call_rank, args=self.plain_args(call_rank, 'Order the cart.', 'order'))
                elif call_rank and turn > 0 and not (short_rule and self.cart_short()) and self.again_ok(avoid):
                    move = dict(name=call_rank, args=self.plain_args(call_rank, 'Order the cart.', 'order'))
                elif call_rank and not short_rule and self.role("cart") and self.passed_fits() and not (self.code_put([(self.passed_fits()[0], max(1, int(self.needs.quantity() or 1)))]) is None) and self.trolley.units:
                    move = dict(name=call_rank, args=self.plain_args(call_rank, 'Order the cart.', 'order'))
                elif call_rank and 'order' == self.__dict__.get('no_fit', None) and self.trolley.units:
                    move = dict(name=call_rank, args=self.plain_args(call_rank, 'No listing satisfies all the requirements; the nearest matches found are being ranked.', 'order'))
                elif decl:
                    move = dict(name=decl, args=self.plain_args(decl, SORRY, 'decline'))
                else:
                    return
            if 'order' == self.call_kind(move['name'], move['args']):
                self.gate_rules_step()
            if 'order' == self.call_kind(move['name'], move['args']) and (self.spec_read_cart() | self.open_link_pages()):
                if self.wire.done or self.wire.broke or 0 >= self.moves_left():
                    return
            if 2 <= self.moves_left() and not self.wire.no_model and self.wire.spare() > LAST_CALL + 8:
                was = self.wire.mute
                try:
                    self.wire.mute = False
                    if self.mv_gate(self.call_kind(move["name"], move["args"]) or "order"):
                        self.__dict__.pop("mv_next", None)
                except (Exception,) as err:
                    self.log.append(dict(kind='mv_gate_error', error='{}: {}'.format(type(err).__name__, err)[:300]))
                finally:
                    self.wire.mute = was
                if self.wire.done or self.wire.broke or 0 >= self.moves_left():
                    return
            elif 'order' == self.call_kind(move['name'], move['args']) and self.basket_due() and 2 <= self.moves_left():
                self.send_basket()
            resp = self.wire.act([move])
            self.log.append(dict(kind='forced_finish', action=move, reply=resp))
            if not resp or resp.get('finished', None) or self.wire.done or turn == CODE_FINISHES - 1 or 1 >= self.moves_left():
                break
            self.wire.mute = True
            lines = list(self.trolley.units)
            obs = resp.get('observation', None)
            self.absorb(move["name"], move["args"], obs, resp, True)
            self.code_news(self.shopper_lines(move["name"], obs, True))
            move = None
            if kind != 'order' or not lines:
                continue
            read, sold = self.code_recheck(lines)
            if not read:
                continue
            self.refused_lines.difference_update(set(lines) - sold)
            info = self.code_rows()
            total = sum(((info.get(line, None) or {}).get('price', None) or 0.0) * self.trolley.held(line) for line in lines)
            cap = self.needs.spend_cap
            if not sold and not (is_number(cap) and cap < total):
                move = dict(name=call_rank, args=self.plain_args(call_rank, 'Order the cart.', 'order'))
            elif is_number(cap) and cap < total:
                avoid, keep = set(sold), None
            else:
                avoid, keep = set(sold), set(lines) - sold
        if not self.wire.done and resp and not resp.get('finished', None) and decl and kind != 'decline' and 0 < self.moves_left():
            again = False
            try:
                again = bool(call_rank) and kind == 'order' and self.again_ok(avoid)
            except (Exception,) as err:
                self.log.append(dict(kind='again_error', error='{}: {}'.format(type(err).__name__, err)[:300]))
            move = ((dict(name=decl, args=self.plain_args(decl, SORRY, 'decline')) if not again else dict(name=call_rank, args=self.plain_args(call_rank, 'Order the cart.', 'order'))))
            self.log.append(dict(kind='forced_finish', action=move, reply=self.wire.act([move])))

    def again_ok(self, avoid):
        units = set(self.trolley.units)
        gone = set(avoid) | self.dead_lines()
        return bool(units) and not units & gone and bool(self.code_picks(gone))

    def catch_up(self):
        try:
            if self.wire.mute or self.wire.spare() < CLOSING:
                return
            notes = []
            if self.needs.reading.get('code_reading', None) and not self.__dict__.get('reread', None) \
                    and self.wire.last_reply > self.__dict__.get("read_at", self.wire.last_reply) and self.wire.spare() > CLOSING + 60:
                self.reread = True
                msg = self.wire.query_model([dict(role='user', content=self.read_content)], READER, 3000)
                reading = json_object((msg or {}).get('content', None) or "") or {}
                self.log.append(dict(kind='re_read', ok=isinstance(reading.get('roles', None), (dict,))))
                if isinstance(reading.get('roles', None), (dict,)):
                    self.use_reading(reading, True)
                    notes.append('[plan updated] the models respond once more and go through the request; this new reading supersedes the first one:\n'
                                 + self.needs.overview(self.step_cap, self.msg_cap))
            waiting = self.needs.__dict__.get('pending', None) or []
            if waiting and self.wire.last_reply > self.needs.__dict__.get("pending_at", 0.0) and self.wire.spare() >= CLOSING:
                self.needs.pending = []
                for lns, qtext in waiting[-3:]:
                    said = self.needs.hear(lns, qtext)
                    if said:
                        notes.append("[the shopper's earlier lines, read now] " + said)
            if notes:
                text_body = "\n".join(notes) + "\n" + self.status_line()
                last = None if not self.chat else self.chat[-1]
                if not (last and last.get('role', None) in ('user', 'tool') and isinstance(last.get('content', None), (str,))):
                    self.chat.append(dict(role='user', content=text_body))
                else:
                    last['content'] += '\n' + text_body
        except (Exception,) as err:
            self.log.append(dict(kind='catch_up_error', error='{}: {}'.format(type(err).__name__, err)[:300]))

    def refused(self, call_name, obs, resp):
        if isinstance(obs, (dict,)) and "refusal" in obs:
            return True
        return call_name == self.role("order") and not resp.get('finished', None) and not (isinstance(obs, (dict,)) and obs.get('error', None) in ("invalid_args", "unknown_tool"))

    def notes_for(self, call_name, obs, refused):
        remarks = []
        if isinstance(obs, (dict,)) and 'budget_exceeded' == obs.get('error', None):
            if call_name == self.role("message") or not self.msg_cap is None:
                self.msg_cap = min(self.msg_cap, self.msgs_sent) if not self.msg_cap is None else self.msgs_sent
            remarks.append('Nothing remains of that budget, so this tool must not be invoked any more.')
        if refused and call_name == self.role("decline") and 'decline' == getattr(self, 'last_kind', None):
            self.audit_done = False
            self.decline_vetted = False
            self.event_seen = True
            remarks.append('THE DECLINE WAS REFUSED and the episode goes on: the store will not accept the claim that nothing matches. Take in the refusal, act on what it states, re-examine the nearest candidates (comparing their own attributes, availability and cost with the current requirements), and purchase the one that suits; refuse again only if every one really falls short.')
        elif refused:
            self.refusal_n = self.refusal_n + 1
            self.order_vetted = False
            self.event_seen = True
            reread = ""
            try:
                reread = self.refusal_reread(obs)
            except (Exception,) as err:
                self.log.append(dict(kind='refusal_reread_error', error='{}: {}'.format(type(err).__name__, err)[:300]))
            self.refused_lines.update(self.trolley.units)
            said: list[str] = []

            def gather(piece, lvl = 0):
                if lvl > 6:
                    return
                if isinstance(piece, (dict,)):
                    for val in piece.values():
                        gather(val, lvl + 1)
                elif isinstance(piece, (list,)):
                    for val in piece[:20]:
                        gather(val, lvl + 1)
                elif isinstance(piece, (str,)) and 4 <= len(piece.split()):
                    said.append(piece)
            gather(obs)
            self.refusal_texts = {" ".join(str(s).lower().split()) for s in said}
            remarks.append('THE ORDER WAS REFUSED and the episode goes on; the cart stays as it was. Carry out every change the refusal lists, correct the cart (remove any line that can no longer be accepted, like a sold-out item), verify stock and price of the new choice again, and submit the order once more.' + reread)
        elif isinstance(obs, (dict,)) and 2 <= len(obs) and call_name != self.role("message"):
            lns: list[str] = self.notice_lines(obs)
            seen = set() if not lns else self.__dict__.pop('refusal_texts', None) or set()
            if lns and seen and all(any(" ".join(str(s).lower().split()) in t for t in seen) for s in lns):
                remarks.append('The message here duplicates what the rejected finish already included (dealt with earlier); nothing in it concerns this particular result.')
            elif lns:
                self.event_seen = True
                remarks.append('The shopper has sent a message within this result (a change). Put it into effect: ' + " | ".join(lns[:4]))
        return " ".join(remarks)

    @staticmethod
    def shelf_of(row):
        return next((val.strip(None).lower() for key, val in row.items() if isinstance(val, (str,)) and 0 < len(val.strip(None)) <= 60
                     and re.search(r"categor|shelf|department|aisle", str(key).lower()) and not 'id' in str(key).lower()), "")

    @staticmethod
    def key_stock(key, val):
        name = str(key).lower()
        if val is None or not re.search(r"stock|avail|on.?hand", name):
            return None
        gone = False
        if re.search(r"stock|avail", name):
            if not re.search('out|sold', name):
                gone = val is False or (is_number(val) and val <= 0) or (isinstance(val, (str,)) and bool(re.search('\\b(?:out|sold|none|unavailable|not|no|false)\\b', val.lower())))
            else:
                gone = val is True
        if re.search(r"on.?hand", name) and (val is False or (is_number(val) and val <= 0)):
            gone = True
        if isinstance(val, (str,)) and re.search(r"back.?order|pre.?order|discontinu", val.lower()):
            gone = True
        return not gone

    @staticmethod
    def stock_state(row):
        states = [s for s in (Errand.key_stock(key, val) for key, val in row.items()) if not s is None]
        return None if not states else all(states)

    @staticmethod
    def sold_out(row):
        return any(Errand.key_stock(key, val) is False for key, val in row.items())

    def shelf_labels(self):
        tally: dict[str, int] = {}
        counted: set[str] = set()
        skip = self.ref_foreign()

        def visit(d):
            ident = next((val for key, val in d.items() if isinstance(val, (str,)) and "id" in str(key).lower() and val in self.archive.read), None)
            shelf = '' if not (ident and not ident in counted and (not ident in skip)) else self.shelf_of(d)
            if shelf:
                counted.add(ident)
                tally[shelf] = tally.get(shelf, 0) + 1
        for rec in self.archive.items:
            walk_dicts(rec, visit)
        top = max(tally.values(), default=0)
        best = {shelf for shelf, n in tally.items() if top == n}
        if not self.needs.items() or not tally:
            return best
        try:
            homes = self.home_shelves()
            picked: set[str] = set()
            for home in homes:
                read = [s for s in home if s in tally]
                most = max((tally[s] for s in read), default=0)
                picked |= set(home) if not read else {s for s in read if tally[s] == most}
            for shelf in sorted(tally, key=lambda s: -tally[s]):
                if all(homes) or len(picked) >= len(homes):
                    break
                picked.add(shelf)
            return picked or best
        except (Exception,):
            return best

    @staticmethod
    def row_mark(row):
        for exact in (True, False):
            for key, val in row.items():
                name = str(key).lower()
                if (name.endswith('id') if not exact else name == 'id') and isinstance(val, (str, int)) and not isinstance(val, (bool,)):
                    return str(val)
        for key, val in row.items():
            if re.search(r"(?:^|_)id(?:_|$)|id$", str(key).lower()) and isinstance(val, (str,)) and re.fullmatch(ID_LIKE, val):
                return val
        return json.dumps(row, default=str, sort_keys=True)

    def search_keys(self, call_args):
        props = self.shapes.get(self.role('search') or '', None)
        props = None if not isinstance(props, (dict,)) else props.get('properties', None)
        props = {} if not isinstance(props, (dict,)) else props

        def spec(key):
            pspec = props.get(key, None)
            return {} if not isinstance(pspec, (dict,)) else pspec
        texts, mode, first, paging = [], [], True, set()
        tok = self.token_key()
        for key, val in call_args.items():
            name = str(key).lower()
            if tok == key and isinstance(val, (str,)):
                paging.add(key)
                first = first and not val.strip(None)
            elif isinstance(val, (str,)) and spec(key).get('enum', None):
                mode.append((key, val))
            elif isinstance(val, (str,)):
                words = tuple(dict.fromkeys(w for w in (t.strip("\"'()[]{},.;:!?|/") for t in val.lower().split())
                                            if re.search(r"[a-z0-9]", w)))
                if words and all(re.fullmatch(r"[a-z0-9]{2,}", w) for w in words):
                    texts.append((key, words))
            elif is_number(val) and re.search(r"page|offset|first|start|skip|from|cursor", name) \
                    and not re.search(r"size|per_?page|limit|price|cost", name):
                paging.add(key)
                low = spec(key).get("minimum", spec(key).get('default', None))
                first = first and val <= ((0 if not 'page' in name else 1) if not is_number(low) else low)
        return texts, tuple(sorted(mode)), first, paging

    def word_rule(self, call_args, obs, rows):
        texts, mode, first, paging = self.search_keys(call_args)
        if not first:
            return self.word_kind
        total = max([val for val in obs.values() if isinstance(val, (int,)) and not isinstance(val, (bool,))] + [len(rows)])
        for key, words in texts:
            ctx = (key, mode)
            others = tuple(sorted((k2, str(v2)) for k2, v2 in call_args.items() if key != k2 and not k2 in paging))
            subset = False
            for seen_ctx, seen_others, seen_words, seen_total in self.word_seen:
                if ctx != seen_ctx or others != seen_others or total == seen_total:
                    continue
                if len(words) > len(seen_words) and words[: len(seen_words)] == seen_words:
                    self.word_kind = 'all' if not seen_total < total else 'any'
                elif len(seen_words) > len(words) and seen_words[: len(words)] == words:
                    self.word_kind = 'all' if not seen_total > total else 'any'
                else:
                    cut = self.lead_words or None
                    mine, theirs = set(words[:cut]), set(seen_words[:cut])
                    subset = subset or (theirs < mine and seen_total > total) or (mine < theirs and total > seen_total)
            if subset and not self.word_kind:
                self.word_kind = "all"
            self.word_seen.append((ctx, others, words, total))
        return self.word_kind

    def lead_rule(self, call_args, obs, rows):
        texts, mode, first, paging = self.search_keys(call_args)
        if not first:
            return self.lead_words
        total = max([val for val in obs.values() if isinstance(val, (int,)) and not isinstance(val, (bool,))] + [len(rows)])
        marks = tuple(self.row_mark(r) for r in rows)
        for key, words in texts:
            ctx = (key, mode)
            others = tuple(sorted((k2, str(v2)) for k2, v2 in call_args.items() if key != k2 and not k2 in paging))
            bound = self.lead_bounds.setdefault(ctx, [-1, set()])
            for seen_ctx, seen_others, seen_words, seen_total, seen_marks in self.lead_seen:
                if ctx != seen_ctx or others != seen_others or words == seen_words:
                    continue
                k = 0
                while k < min(len(words), len(seen_words)) and words[k] == seen_words[k]:
                    k = k + 1
                if seen_total != total or seen_marks != marks:
                    bound[0] = max(bound[0], k)
                elif k and 3 <= len(set(marks)) and k == min(len(words), len(seen_words)):
                    bound[1].add(k)
            self.lead_seen.append((ctx, others, words, total, marks))
        claims = sorted((min(k for k in high if low < k), ctx) for ctx, (low, high) in self.lead_bounds.items()
                        if any(low < k for k in high))
        claims = [(k, ctx) for k, ctx in claims if k == self.lead_bounds[ctx][0] + 1]
        self.lead_words, self.lead_ctx = (0, None) if not claims else claims[0]
        return self.lead_words

    def token_key(self):
        props = (self.shapes.get(self.role('search') or '', None) or {}).get('properties', None)
        for key, pspec in ({} if not isinstance(props, (dict,)) else props).items():
            if isinstance(pspec, (dict,)) and 'string' == pspec.get('type', None) and not pspec.get('enum', None) \
                    and re.search(r"bookmark|cursor|token|resume|continu|after|next|page|offset|start", str(key).lower()):
                return key
        return None

    @staticmethod
    def next_token(obs):
        if not isinstance(obs, (dict,)):
            return None
        for key, val in obs.items():
            if isinstance(val, (str,)) and val.strip(None) and re.search(r"next|onward|cursor|bookmark|token|continu|more|further|follow|after|resume",
                                                                  str(key).lower()):
                return val
        return None

    def page_arg(self):
        props = (self.shapes.get(self.role('search') or '', None) or {}).get('properties', None)
        for key, pspec in ({} if not isinstance(props, (dict,)) else props).items():
            name = str(key).lower()
            if not isinstance(pspec, (dict,)) or not pspec.get('type', None) in ('integer', 'number') \
                    or not re.search(r"page|offset|first|start|skip|from|cursor", name) or re.search(r"size|per_?page|limit|price|cost", name):
                continue
            by_page = "page" in name and not re.search(r"offset|skip|first|start|from", name)
            low = pspec.get("minimum", pspec.get('default', None))
            low = (0 if not by_page else 1) if not is_number(low) else low
            return key, by_page, low, (None if not is_number(pspec.get('maximum', None)) else pspec.get('maximum', None))
        tok = self.token_key()
        return None if not tok else (tok, 'token', '', None)

    @staticmethod
    def stem(word):
        return word if not (3 < len(word) and word.endswith('s') and (not word.endswith('ss'))) else word[:-1]

    def shelf_rows(self):
        shelves, cap, qty = self.shelf_labels(), self.needs.spend_cap, self.needs.quantity()
        found: dict[str, tuple[float, str, str]] = {}
        if not shelves:
            return found

        def visit(d):
            ident = next((val for key, val in d.items() if isinstance(val, (str,)) and "id" in str(key).lower() and val in self.archive.ids), None)
            if not ident or ident in self.refused_lines or not self.shelf_of(d) in shelves:
                return
            if self.sold_out(d):
                found.pop(ident, None)
                return
            cost = next((val for key, val in d.items() if is_number(val) and not isinstance(val, (bool,)) and re.search(r"price|cost", str(key).lower())), None)
            title = (next((val for key, val in d.items() if isinstance(val, (str,)) and 0 < len(val) <= 300 and re.search('title|heading|headline|name', str(key).lower()) and (not 'id' in str(key).lower()) and (not self.shelf_of({key: val})) and (not junk_key(key))), '') if not (isinstance(d.get('title', None), (str,)) and 0 < len(d['title']) <= 300) else d.get('title', None))
            maker = " ".join(val for key, val in d.items() if isinstance(val, (str,)) and re.search(r"brand|make|manufactur|vendor", str(key).lower()))
            if not cost is None:
                found[ident] = (float(cost), title, maker)
        for rec in self.archive.items:
            walk_dicts(rec, visit)
        for ident, (cost, title, maker) in list(found.items()):
            newest = self.archive.price(ident)
            cost = cost if not not newest is None else newest
            if not (is_number(cap) and cost * qty > cap):
                found[ident] = (float(cost), title, maker)
            else:
                found.pop(ident)
        return found

    def label_queries(self):
        out: list[str] = []
        for label in sorted(self.shelf_labels()):
            parts, cur = [], []
            for word in label.lower().split():
                word = word.strip(".,;:!?'\"()[]")
                if not (word in LINK_WORDS or not re.fullmatch("[a-z0-9][a-z0-9'-]*", word)):
                    cur.append(word)
                else:
                    if cur:
                        parts.append(cur)
                    cur = []
            if cur:
                parts.append(cur)
            words = [w for part in parts for w in part]
            heads = [part[-1] for part in parts]
            ones = [self.stem(h) for h in heads]
            out += ones + ([] if not 2 <= len(words) else [' '.join(words[:2])]) + heads + [w for w in words if not w in heads]
        return list(dict.fromkeys(out))

    def kind_queries(self, pairs):
        label = {self.stem(w) for s in self.shelf_labels() for w in re.findall(r"[a-z0-9]+", s)}
        titles: dict[str, tuple[float, str]] = {}
        for cost, title, maker in self.shelf_rows().values():
            low = title.lower().strip(None)
            if low:
                titles[low] = (min(cost, titles.get(low, (cost, ""))[0]), maker.lower())
        count: dict[str, int] = {}
        cheapest: dict[str, float] = {}
        shown: dict[str, str] = {}
        for title, (cost, maker) in titles.items():
            makers = set(re.findall(r"[a-z]+", maker))
            words = re.findall(r"[a-z]+", title)
            plain = [3 <= len(w) and not w in LINK_WORDS and not w in makers for w in words]
            met: set[str] = set()
            grams = [(words[i],) for i in range(0, len(words)) if plain[i]] if not pairs else [(words[i], words[i + 1]) for i in range(0, len(words) - 1) if plain[i] and plain[i + 1]]
            for gram in grams:
                base = " ".join(self.stem(w) for w in gram)
                if all(self.stem(w) in label for w in gram) or base in met:
                    continue
                met.add(base)
                count[base] = count.get(base, 0) + 1
                cheapest[base] = min(cheapest.get(base, cost), cost)
                shown.setdefault(base, " ".join(gram))
        return [shown[b] for b in sorted((b for b in count if 2 <= count[b]), key=lambda b: (cheapest[b], -count[b], b))]

    def sweep_next(self, turn):
        found = self.page_arg()
        shelves = self.shelf_labels()
        book = self.__dict__.get("page_book", {})
        if not found or not shelves or not book:
            return None, False
        key, _, low, _ = found
        tried = self.__dict__.get("sweep_tried", set())
        lists = [w for w in book.values() if w.get('id_order', None) and w["shelves"] and w["shelves"] <= shelves and not self.next_page(w) is None]
        listing = max(lists, key=lambda w: w["total"], default=None)
        ranked = [w for w in book.values() if w.get('id_order', None) is False and self.free_text(w["args"])]
        ranked = ranked or [w for w in book.values() if w.get('id_order', None) is None and self.free_text(w["args"])]
        template, text_key = None, None
        if ranked:
            props = (self.shapes.get(self.role('search') or '', None) or {}).get('properties', None) or {}
            args = ranked[-1]["args"]

            def fixed(k, v):
                return not isinstance(v, (str,)) or bool((props.get(k, None) or {}).get('enum', None)) or v.strip(None).lower() in shelves or key == k
            texts = [k for k, v in args.items() if not fixed(k, v)]
            text_key = max(texts, key=lambda k: len(args[k].split()), default=None) if not (self.lead_ctx and self.lead_ctx[0] in texts) else self.lead_ctx[0]
            template = None if not text_key else {k: v for k, v in args.items() if text_key != k and fixed(k, v)}
        done = {tuple(self.stem(w) for w in re.findall(r"[a-z0-9]{2,}", " ".join(self.free_text(w["args"])).lower())[:2])
                for w in book.values() if low in w["read"] and not w.get('id_order', None) is True}

        def query(text):
            sig = tuple(self.stem(w) for w in re.findall(r"[a-z0-9]{2,}", text.lower())[:2])
            if template is None or not sig or sig in done:
                return None
            args = {**template, text_key: text}
            return args if not json.dumps(args, default=str, sort_keys=True) in tried else None
        heads = {self.stem(q) for q in self.label_queries() if not ' ' in q}

        def second_pages():
            out = []
            for w in book.values():
                said = [self.stem(t.strip(None).lower()) for t in self.free_text(w["args"])]
                if 1 == len(w['read']) and said and said[0] in heads and w["shelves"] & shelves and not self.next_page(w) is None:
                    out.append({**w["args"], key: self.next_page(w)})
            return out
        sources = [
            lambda: [] if not listing else [{**listing['args'], key: self.next_page(listing)}],
            lambda: [q for q in (query(t) for t in self.kind_queries(True)[:6]) if q],
            lambda: [q for q in (query(t) for t in self.kind_queries(False)[:6]) if q],
            lambda: [q for q in (query(t) for t in self.label_queries()) if q],
        ]
        ceil, ended = self.ceiling_sweep(tried)
        if not ceil is None:
            return ceil, True
        if ended:
            return None, False
        extra = [a for a in second_pages() if not json.dumps(a, default=str, sort_keys=True) in tried]
        if extra and 1 == turn % 2:
            return extra[0], False
        for pick in [0] + [1 + (turn + shift) % 3 for shift in range(0, 3)]:
            picks = [a for a in sources[pick]() if not json.dumps(a, default=str, sort_keys=True) in tried]
            if picks:
                return picks[0], pick == 0
        return None, False

    def sweep_step(self):
        nd = self.needs
        left = nd.props_left()
        blind = bool(left) and all(self.amount_part(r.partition(":")[2]) or re.search(CH_BOUND, r.partition(":")[2].lower()) for r in left)
        if not nd.priority() in ('cheapest', 'best') or not ((nd.let_go and not left) or blind) or not self.role("search") or self.wire.done:
            return False
        keep = 3 + (0 if not self.trolley.units else 1)
        if self.orders_tried == 0:
            keep = max(keep + 3, FIRST_ORDER_BY)
        swept = self.__dict__.get("swept", 0)
        if SWEEP_READS <= swept or self.moves_left() <= keep or self.wire.spare() < CLOSING + 40:
            return False
        turn = self.__dict__.get("sweep_turn", 0)
        call_args, listing = self.sweep_next(turn)
        if call_args is None:
            return False
        self.swept = swept + 1
        self.sweep_turn = turn + (1 if not listing else 0)
        self.__dict__.setdefault("sweep_tried", set()).add(json.dumps(call_args, default=str, sort_keys=True))
        proposal = (dict(id='sweep_{}_{}'.format(self.swept, self.wire.tick), type='function', function=dict(name=self.role('search'), arguments=json.dumps(call_args))))
        self.chat.append(dict(role='assistant', content='Since no attribute requirement remains, the lowest-priced listing that is in stock anywhere on the shelf is the winner, so I go through one more portion of the shelf prior to deciding.', tool_calls=[proposal]))
        self.covering = True
        try:
            responses = self.dispatch([proposal], scripted=True)
        finally:
            self.covering = False
        more = (SWEEP_READS > self.swept and self.moves_left() > keep and self.wire.spare() >= CLOSING + 40
                and not self.sweep_next(self.sweep_turn)[0] is None)
        if not more and responses:
            rows = sorted(self.shelf_rows().items(), key=lambda kv: (kv[1][0], kv[0]))[:6]
            if rows:
                cid, text_body = responses[-1]
                note = (None if not blind else '\n[shelf read] Among everything viewed, these are the least expensive in-stock entries in the requested category that fit the budget: ' + '; '.join((f"{i} '{t[:80]}' {c:g}" for i, (c, t, _) in rows)) + '. What remains are requirements expressed as figures that no search word captures: open these listings starting with the lowest price, and test each against all requirements; the lowest-priced one that fits (or the best one, when a best rule applies) is the choice.')
                responses[-1] = ((cid, text_body + '\n[shelf sweep done] Of all the rows seen so far, these are the lowest-priced ones in stock for the requested category that fit the budget: ' + '; '.join((f"{i} '{t[:80]}' {c:g}" for i, (c, t, _) in rows)) + '. Choose the one with the lowest price unless some change disqualifies it: put it alone in the cart, verify again that it is in stock and its price, and then place the order.') if not note else (cid, text_body + note))
        self.post_replies(responses)
        return True

    def free_text(self, call_args):
        props = (self.shapes.get(self.role('search') or '', None) or {}).get('properties', None)
        props = {} if not isinstance(props, (dict,)) else props
        tok = self.token_key()
        return [val for key, val in call_args.items() if isinstance(val, (str,)) and not (props.get(key, None) or {}).get('enum', None) and tok != key]

    def book_page(self, call_args, rows, total, obs = None):
        found = self.page_arg()
        if not found or not rows:
            return
        key, mode, low, _ = found
        at = call_args.get(key, low)
        if mode == 'token':
            at = '' if not isinstance(at, (str,)) else at.strip(None)
        elif not is_number(at):
            return
        base = {k: v for k, v in call_args.items() if key != k}
        walk = self.__dict__.setdefault("page_book", {}).setdefault(
            json.dumps(base, default=str, sort_keys=True), dict(args=base, read=set(), size=0, total=0, shelves=set()))
        walk["read"].add(at)
        walk["last_rows"] = [self.row_mark(r) for r in rows if isinstance(r, (dict,))]
        if mode == 'token':
            walk.setdefault("tokens", {})[at] = self.next_token(obs)
        walk["size"] = max(walk["size"], len(rows))
        walk["total"] = max(walk["total"], total)
        walk["shelves"] |= {self.shelf_of(r) for r in rows if isinstance(r, (dict,))} - {""}
        shown = walk.setdefault("row_shelf", {})
        for r in rows:
            if isinstance(r, (dict,)):
                shown.setdefault(self.row_mark(r), self.shelf_of(r))
        if low == at:
            walk["first"] = tuple(self.row_mark(r) for r in rows if isinstance(r, (dict,)))
        if low == at and total <= len(rows):
            walk["every"] = [r for r in rows if isinstance(r, (dict,))]
        marks = [self.row_mark(r) for r in rows if isinstance(r, (dict,))]
        if 5 <= len(marks):
            walk["id_order"] = not walk.get('id_order', True) is False and (marks == sorted(marks) or marks == sorted(
                marks, key=lambda m: [t if not t.isdigit() else int(t) for t in re.split(r"(\d+)", m)]))

    def next_page(self, walk):
        key, by_page, low, high = self.page_arg()
        if by_page == 'token':
            for nxt in (walk.get('tokens', None) or {}).values():
                if isinstance(nxt, (str,)) and not nxt in walk['read']:
                    return nxt
            return None
        low = min([low] + [v for v in walk["read"] if is_number(v)])
        size = max(1, walk["size"])
        step = size if not by_page else 1
        for i in range(1, 400):
            val = low + i * step
            if (not high is None and high < val) or (i * size >= walk["total"]):
                return None
            if not val in walk['read']:
                return val
        return None

    def value_score(self, call_args):
        def near(a, b):
            return 3 <= min(len(a), len(b)) and (a.startswith(b) or b.startswith(a))
        vals = [re.findall(r"[a-z0-9]{3,}", req.partition(":")[2].lower()) for req in self.needs.props_left() if not ref_text(req.partition(":")[2])]
        words = [w for text in self.free_text(call_args) for w in re.findall(r"[a-z0-9]{2,}", text.lower())]
        words = words if not self.lead_words else words[:self.lead_words]
        return sum(1 for value_words in vals if any(near(w, v) for w in words for v in value_words))

    def row_hits(self, rows, words = None):
        cap, qty = self.needs.spend_cap, self.needs.quantity()
        shelves = self.shelf_labels()
        hits = []
        asked = set() if not self.needs.reading.get('references', None) else {w for t in words or [] for w in re.findall('[a-z0-9]+', str(t).lower())}
        pointers = self.__dict__.setdefault("ref_hits", [])
        for r in rows:
            if not isinstance(r, (dict,)):
                continue
            ident = self.row_mark(r)
            makers = [set(re.findall(r"[a-z0-9]+", val.lower())) for k2, val in r.items() if isinstance(val, (str,)) and val.strip(None)
                      and re.search(r"shop|seller|store|vendor|brand|make|manufactur", str(k2).lower()) and not 'id' in str(k2).lower()]
            if asked and ident in self.archive.ids and not ident in self.archive.read \
                    and any(m and asked >= m and m <= self.pointed_words() for m in makers):
                if not ident in pointers:
                    pointers.append(ident)
            cost = next((val for k2, val in r.items() if is_number(val) and re.search(r"price|cost", str(k2).lower())), None)
            shelf = self.shelf_of(r)
            if ident in self.archive.ids and not ident in self.archive.read and not (is_number(cap) and is_number(cost) and cost * qty > cap) \
                    and not self.sold_out(r) and not (shelves and shelf and not shelf in shelves):
                hits.append(ident)
                self.__dict__.setdefault("hit_shelf", {})[ident] = shelf
        return hits

    def pointed_words(self):
        memo = self.__dict__.get('pointed_memo', None)
        if memo and memo[0] == len(self.archive.items):
            return memo[1]
        words: set[str] = set()
        try:
            for ref in self.needs.reading.get('references', None) or []:
                words |= set(re.findall(r"[a-z0-9]+", str(ref).lower()))

            def visit(d):
                if not self.row_mark(d) in self.archive.read:
                    return
                own: set[str] = set()
                texts: list[str] = []
                for key, val in d.items():
                    if not isinstance(val, (str,)):
                        continue
                    low = str(key).lower()
                    if re.search(r"shop|seller|store|vendor|brand|make|manufactur", low) and not 'id' in low and 40 >= len(val) \
                            and not re.search(r"note|text|desc|comment|detail|info|message|section", low):
                        own |= set(re.findall(r"[a-z0-9]+", val.lower()))
                    elif not re.search(r"title|heading|headline|name|label|caption", low):
                        texts.append(val.lower())
                for text in texts:
                    words.update(set(re.findall(r"[a-z0-9]+", text)) - own)
            for rec in self.archive.items:
                walk_dicts(rec, visit)
        except (Exception,):
            pass
        self.pointed_memo = (len(self.archive.items), words)
        return words

    def open_pages(self, most = 2):
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

    def value_probe(self, raw):
        if self.word_kind == 'any':
            return "", ""
        lead = [w for w in re.findall(r"[a-z0-9]+", " ".join(raw).lower()) if 2 <= len(w)][: self.lead_words or 2]
        product = " ".join(re.findall(r"[a-z0-9]+", str(self.needs.reading.get('product', None) or "").lower())[:3])

        def near(a, b):
            if any(c.isdigit() for c in a + b):
                return a.lstrip("0") == b.lstrip("0") or (a.isdigit() and b.startswith(a) and not b[len(a):len(a) + 1].isdigit())
            return 3 <= min(len(a), len(b)) and (a.startswith(b) or b.startswith(a))
        reqs = [(req.partition(':')[0].strip(None), re.split('[,;]', re.sub('\\((?:stated|revealed[^)]*)\\)', '', req.partition(':')[2]))[0].strip(None).lower())
                for req in self.needs.props_left() if ":" in req and not ref_text(req.partition(":")[2])]
        keep = [w for _, value in reqs for w in re.findall(r"[a-z0-9]{2,}", value)] + product.split()
        if not lead or any(near(w, k) for w in lead for k in keep):
            return "", ""
        for name, value in reqs:
            if value and any(near(w, n) for w in lead for n in re.findall(r"[a-z0-9]{2,}", name.lower())):
                digit = re.search(r"\d", value)
                probe = '{} {}'.format((value if not digit else value[digit.start():]).strip(' .;'), product).strip(None)
                head = [w for w in re.findall(r"[a-z0-9]+", probe.lower()) if 2 <= len(w)][: self.lead_words or 2]
                for walk in self.__dict__.get("page_book", {}).values():
                    ran = [w for t in self.free_text(walk["args"]) for w in re.findall(r"[a-z0-9]+", t.lower()) if 2 <= len(w)][: self.lead_words or 2]
                    if len(ran) == len(head) and all(near(a, b) for a, b in zip(head, ran)):
                        return "", ""
                return name, probe
        return "", ""

    def ran_query(self, text):
        width = self.lead_words or 3
        mine = sorted(self.stem(w) for w in re.findall(r"[a-z0-9]+", str(text).lower())[:width])
        for walk in self.__dict__.get("page_book", {}).values():
            ran = sorted(self.stem(w) for w in re.findall(r"[a-z0-9]+", " ".join(self.free_text(walk["args"])).lower())[:width])
            if mine == ran:
                return True
        return False

    def search_template(self):
        for walk in reversed(list(self.__dict__.get("page_book", {}).values())):
            args = walk["args"]
            if self.cg_listing(args):
                continue
            texts = self.free_text(args)
            keys = [k for k, v in args.items() if isinstance(v, (str,)) and v in texts and v.strip(None)]
            if 1 == len(texts) and 1 == len(keys):
                return {k: v for k, v in args.items() if k != keys[0]}, keys[0]
        return None

    def word_trap(self, raw):
        try:
            if self.word_kind == 'any':
                return "", ""
            toks = [t.strip("\"'()[]{},.;!?|").lower() for t in " ".join(raw).split()]
            toks = [t for t in toks if re.search(r"[a-z0-9]", t)]
            width = self.lead_words or (3 if not self.word_kind == 'all' else max(3, len(toks)))
            toks = toks[:width]

            def near(a, b):
                if any(c.isdigit() for c in a + b):
                    return a.lstrip("0") == b.lstrip("0")
                return 3 <= min(len(a), len(b)) and (a.startswith(b) or b.startswith(a))

            def parts(t):
                return [p for p in re.findall(r"[a-z0-9]+", t) if 2 <= len(p)]
            reqs = [(re.split(r"\s*\(", req.partition(":")[0])[0].lower(), req.partition(":")[0],
                     re.sub(r"\((?:stated|revealed[^)]*)\)", "", req.partition(":")[2]).lower()) for req in self.needs.props_left()
                    if ":" in req and not ref_text(req.partition(":")[2])]
            vals = {w for _, _, v in reqs for w in re.findall(r"[a-z0-9]{2,}", v)}
            products = [r["product"] for r in self.needs.items()] or [str(self.needs.reading.get('product', None) or "")]
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
            rest = [t for t in toks if not t in flagged]
            held = [(tag, v) for _, tag, v in reqs if any(near(p, w) for t in rest for p in parts(t) for w in re.findall(r"[a-z0-9]{2,}", v))]
            if not held:
                return "", ""
            if not any(near(p, k) for t in rest for p in parts(t) for k in kinds):
                k = None if not self.needs.items() else self.product_of(held[0][0])
                name = products[0] if not (not k is None and k < len(products)) else products[k]
                kind = [w for w in re.findall(r"[a-z]{3,}", name.lower()) if not w in LINK_WORDS][-1:]
                rest += kind
            query = " ".join(rest[:width])
            if not query or self.ran_query(query):
                return "", ""
            note = (("The word '{}' in this query relates solely to a requirement the shopper DROPPED: only listings that carry it are still kept, even though they are no longer obliged to.".format(flagged[0]) if not why == 'name' else "The word '{}' in this query comes from the NAME of a required attribute, not from a required value: all words must match here, so only listings whose fields or title echo that name are kept, and those whose field contains just the value are lost.".format(flagged[0])))
            return note + " Run a search for '{}' too, and go through all of its pages.".format(query), query
        except (Exception,) as err:
            self.log.append(dict(kind='trap_error', error='{}: {}'.format(type(err).__name__, err)[:300]))
            return "", ""

    def combined_queries(self):
        template = self.search_template()
        if not template:
            return []
        rows = self.needs.items()
        width = self.lead_words or 3
        live = {k.rpartition("|")[0] for k in self.passed if k.rpartition("|")[2] == str(self.needs.version)} - self.refused_lines
        got = {self.shelf_known(i) for i in live} - {""}
        homes = [] if not rows else self.home_shelves()
        out: list[dict[str, Any]] = []
        for k in ([None] if not rows else range(0, len(rows))):
            if rows and (k >= len(homes) or not homes[k] or homes[k] & got):
                continue
            if not rows and not self.unverified():
                continue
            vals: list[list[str]] = []
            for req in self.needs.props_left():
                head, _, value = req.partition(":")
                if rows and not self.product_of(head) in (k, None):
                    continue
                value = re.sub('\\((?:stated|revealed[^)]*)\\)', '', value).strip(None).lower()
                if not value or re.search(r"\d|as shown|same as|page|redirect|refer|\bnot\b|\bno\b|without|except|anything|other than", value):
                    continue
                first = re.split(r"[,;/]|\bor\b|\band\b", value)[0]
                words = [w for w in re.findall(r"[a-z]{2,}", first) if not w in LINK_WORDS]
                if 1 <= len(words) <= 2:
                    vals.append(words)
            if 2 > len(vals):
                continue
            query: list[str] = []
            for words in vals:
                if len(query) + len(words) <= width:
                    query += words
            product = str(self.needs.reading.get('product', None) or '') if not rows else rows[k]['product']
            kind = [w for w in re.findall(r"[a-z]{3,}", product.lower()) if not w in LINK_WORDS][-1:]
            if kind and len(query) < width and not kind[0] in query:
                query += kind
            text = " ".join(query)
            if 2 <= len(query) and not self.ran_query(text):
                out.append({**template[0], template[1]: text})
        return out[:2]

    def arbitrate(self, thing, ruling):
        try:
            reason = next((str((r.get('result', None) or {}).get('reason', None) or "") for r in reversed(self.log)
                           if isinstance(r, (dict,)) and 'audit' == r.get('kind', None) and isinstance(r.get('result', None), (dict,))), "")
            note = ('\n\nTWO READINGS DISAGREE ABOUT THIS ITEM. The audit of the decline, going through the same records, determined that it satisfies all the requirements' + ('' if not reason else ' ({})'.format(reason[:400])) + "; an earlier check found: " + ruling[:500]
                    + '\nUse the records to determine which interpretation holds: if a field expresses the needed value differently, or gives a tint, modifier or alternate spelling of the same named value, the requirement is satisfied; if a field gives a different value, the requirement is not satisfied.')
            again = self.check_item(thing, "ORDER", second=note)
            if again:
                return again
            key = '{}|{}'.format(thing, self.needs.version)
            for memo in [k for k in self.rulings if key == k or k.startswith(key + "|")]:
                self.rulings[memo] = ""
            return ""
        except (Exception,) as err:
            self.log.append(dict(kind='arbitrate_error', error='{}: {}'.format(type(err).__name__, err)[:300]))
            return ruling

    def unrefuted(self):
        try:
            rows = self.needs.items()
            homes = [set(self.shelf_labels())] if not rows else self.home_shelves()
            if not homes or any(not home for home in homes):
                return []
            ver = str(self.needs.version)
            failed = {k.split("|")[0] for k, v in self.rulings.items() if v and k.split("|")[1:2] == [ver]}
            live = {k.rpartition("|")[0] for k in self.passed if k.rpartition("|")[2] == ver}
            reviewed = self.__dict__.get('audit_ids', None)
            if self.__dict__.get('audit_done', None) and (reviewed is None or len(self.archive.read) <= self.__dict__.get("audit_reads", -1)):
                reviewed = set(self.archive.read)
            reviewed = set(reviewed or set())
            audited = reviewed - set(self.__dict__.get('audit_fits', None) or [])
            sold: dict[str, bool] = {}

            def visit(d):
                ident = self.row_mark(d)
                if ident in self.archive.ids and any(re.search(r"stock|avail", str(key).lower()) for key in d):
                    sold[ident] = self.sold_out(d)
            for rec in self.archive.items:
                walk_dicts(rec, visit)
            cap = self.needs.spend_cap
            picks: list[str] = []
            total = 0.0
            pointers = self.ref_pointers()
            for k, home in enumerate(homes, 0):
                qty = self.needs.quantity() if not rows else rows[k]['quantity']
                pool = []
                for ident in sorted(self.archive.read):
                    if ident in failed or ident in audited or ident in self.refused_lines or sold.get(ident, None) or ident in picks \
                            or ident in pointers or not self.shelf_known(ident) in home:
                        continue
                    cost = self.archive.price(ident)
                    if not is_number(cost) or (is_number(cap) and cost * qty > cap):
                        continue
                    pool.append((not ident in live, cost, ident))
                if not pool:
                    return []
                _, cost, ident = min(pool)
                picks.append(ident)
                total += cost * qty
            if is_number(cap) and cap < total:
                return []
            return picks
        except (Exception,) as err:
            self.log.append(dict(kind='unrefuted_error', error='{}: {}'.format(type(err).__name__, err)[:300]))
            return []

    def fit_hold(self):
        try:
            if self.__dict__.get('fit_held', None) or self.trolley.units or 4 >= self.moves_left() or self.wire.spare() < CLOSING + 30:
                return ""
            fits = [i for i in self.__dict__.get('audit_fits', None) or [] if i in self.archive.ids and not i in self.refused_lines]
            if not fits:
                return ""
            self.fit_held = True
            good = []
            for thing in fits[:2]:
                if not thing in self.archive.read:
                    continue
                ruling = self.check_item(thing, "ORDER")
                if ruling and self.wire.spare() > CLOSING + 30:
                    ruling = self.arbitrate(thing, ruling)
                if not ruling and '{}|{}'.format(thing, self.needs.version) in self.passed:
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
            return ("The decline audit listed " + ", ".join(good) + ' as a candidate that might suit, plus the separate verification of the item, which examines '
                    + ('their' if not 1 == len(good) else 'its') + " full record, found " + ('they meet' if not 1 == len(good) else 'it meets')
                    + " every requirement now in force. Weigh " + ('them' if not 1 == len(good) else 'it') + " before declining: if it fits, put "
                    + (good[0] if not self.needs.items() else 'a verified line of every product') + ' in the cart, check its stock and price again, then place the order; refuse if it breaks any requirement that applies.')
        except (Exception,) as err:
            self.log.append(dict(kind='fit_hold_error', error='{}: {}'.format(type(err).__name__, err)[:300]))
            return ""

    def id_store(self):
        orders = [w.get('id_order', None) for w in self.__dict__.get("page_book", {}).values() if "id_order" in w]
        return 2 <= orders.count(True) and not False in orders

    def product_index(self, thing):
        if not self.needs.items():
            return None
        shelf = self.shelf_known(thing)
        for k, home in enumerate(self.home_shelves(), 0):
            if shelf and shelf in home:
                return k
        return None

    def walk_home(self, k):
        homes = self.home_shelves()
        home = set() if not (homes and (k or 0) < len(homes)) else set(homes[0 if not not k is None else k])
        return home or (set() if not k is None else set(self.shelf_labels()))

    @staticmethod
    def fig_amount(text):
        low = str(text).lower()
        if re.search(CH_BOUND, low):
            return None
        found = re.match(r"\s*(\d+)\s+([a-z]+)", low)
        return None if not found else (found.group(1), found.group(2))

    @staticmethod
    def amount_part(text):
        low = str(text).lower()
        return bool(re.search(r"\d", low)) and bool(re.search(CH_BOUND, low) or re.match(r"\s*\d+(?:[.,]\d+)?\s+[a-z]", low))

    def walk_carries(self, args, k, alts = False):
        def near(a, b):
            return 3 <= min(len(a), len(b)) and (a.startswith(b) or b.startswith(a))
        several = bool(self.needs.items()) and not k is None
        words = [w for text in self.free_text(args) for w in re.findall(r"[a-z0-9]{2,}", text.lower())]
        words = words if not self.lead_words else words[:self.lead_words]
        toks = [w for text in self.free_text(args) for w in re.findall(r"[a-z0-9]+", text.lower())]
        toks = toks if not self.lead_words else toks[:self.lead_words]
        out: set[Any] = set()
        for n, req in enumerate(self.needs.props_left(), 0):
            if several and not self.product_of(req.partition(':')[0]) in (k, None):
                continue
            value = re.sub(r"\((?:stated|revealed[^)]*)\)", "", req.partition(":")[2]).lower()
            for a, part in enumerate(re.split('[,;/]|\\bor\\b|\\beither\\b|\\bnor\\b', value), 0):
                fig = self.fig_amount(part)
                if fig:
                    if fig[0] in toks and any(w == fig[1] or near(w, fig[1]) for w in toks):
                        out.add(n if not alts else (n, a))
                    continue
                if not self.amount_part(part) and any(near(w, v) for w in words for v in re.findall(r"[a-z0-9]{3,}", part)) \
                        or any(v == w for w in words for v in re.findall(r"\b(?:\d[a-z]|[a-z]\d)\b", part)):
                    out.add(n if not alts else (n, a))
        return out

    def walk_score(self, args, k):
        return len(self.walk_carries(args, k))

    def walk_options(self, k):
        home = self.walk_home(k)
        if not home:
            return []
        cut = self.lead_words or None

        def lead(walk):
            return {self.stem(w) for w in re.findall(r"[a-z0-9]+", " ".join(self.free_text(walk["args"])).lower())[:cut]}
        label = {self.stem(w) for s in home for w in re.findall(r"[a-z0-9]{2,}", s)}
        several = bool(self.needs.items()) and not k is None
        values = {self.stem(w) for req in self.needs.props_left() if not several or self.product_of(req.partition(":")[0]) in (k, None)
                  for w in re.findall(r"[a-z0-9]{2,}", req.partition(":")[2].lower())}
        book = self.__dict__.get("page_book", {})
        done = [(lead(w), self.walk_carries(w["args"], k, alts=True), self.vl_mode(w["args"])) for w in book.values()
                if w.get('id_order', None) is True and w.get('first', None) and self.next_page(w) is None and lead(w)
                and len(w.get('row_shelf', None) or {}) >= w["total"]]
        best: dict[tuple, tuple[int, str, dict[str, Any]]] = {}
        for key, walk in book.items():
            if not walk.get('id_order', None) is True or not walk.get('first', None) or self.next_page(walk) is None:
                continue
            shown = walk.get('row_shelf', None) or {}
            carries = self.walk_carries(walk["args"], k)
            if not any(s in home for s in shown.values()) or not carries:
                continue
            words = lead(walk)
            pairs = self.walk_carries(walk["args"], k, alts=True)
            how = self.vl_mode(walk["args"])
            if any(how == mode and (words >= seen or (seen <= values | label and got >= pairs)) for seen, got, mode in done):
                continue
            pages = -(-walk["total"] // max(1, walk["size"])) - len(walk["read"])
            twin = (walk["total"], walk["first"])
            if 0 < pages <= 2 * WALK_PAGES and (not twin in best or pages < best[twin][0]):
                best[twin] = (pages, key, walk)
        opts = list(best.values())
        if self.all_words():
            def nested(p, o):
                a, b = lead(p[2]), lead(o[2])
                return (b > a and not (b - a) <= values | label and p[0] <= WALK_PAGES) or (a > b and (a - b) <= values | label)
            opts = [o for o in opts if not any(nested(p, o) for p in opts if not p is o)]
        return sorted(opts, key=lambda p: (p[0], -self.walk_score(p[2]["args"], k), p[1]))

    def walk_plan(self, budget, only = None):
        if budget <= 0 or 'cheapest' != self.needs.priority() or not self.role("search") or not self.page_arg() or not self.id_store():
            return []
        done = self.__dict__.get("walked", set())
        wants = []
        for k in (list(range(0, len(self.needs.items()))) or [None]):
            if k in done or (not only is None and not k in only):
                continue
            opts = self.walk_options(k)
            if opts:
                wants.append((opts[0][0], k if not k is None else -1, k, opts[0][1]))
        plan, left = [], budget
        for pages, _, k, key in sorted(wants, key=lambda p: (p[0], p[1])):
            n = min(pages, WALK_PAGES, left)
            if n <= 0:
                break
            plan.append((k, key, n))
            left -= n
        return plan

    def walk_pick(self, k):
        gone = self.dead_lines()
        cands = [i for i in self.trolley.units if (k is None or self.product_index(i) == k) and not i in gone]
        if not cands:
            cands = [key.rpartition("|")[0] for key in self.passed if key.rpartition("|")[2] == str(self.needs.version)]
            cands = [i for i in cands if not i in self.refused_lines and not i in gone and (k is None or self.product_index(i) == k)]
        priced = [(i, self.archive.price(i)) for i in cands]
        priced = [(i, float(p)) for i, p in priced if is_number(p)]
        return None if not priced else min(priced, key=lambda ip: (ip[1], ip[0]))

    def line_query(self, k):
        pick = self.walk_pick(k)
        template = self.search_template()
        if not pick or not template:
            return None
        rec: dict[str, Any] = {}

        def visit(d):
            nonlocal rec
            if self.row_mark(d) == pick[0] and self.title_of(d):
                rec = d
        for item in self.archive.items:
            walk_dicts(item, visit)
        maker = next((v for key, v in rec.items() if isinstance(v, (str,)) and v.strip(None) and re.search(r"brand|make|manufactur", str(key).lower())
                      and not 'id' in str(key).lower()), "")
        maker_words = re.findall(r"[a-z0-9]+", maker.lower())
        if not maker_words:
            return self.title_line(k, rec, template)
        avoid = set(maker_words) | LINK_WORDS | {w for s in self.walk_home(k) for w in re.findall(r"[a-z]+", s)}
        avoid |= {w for req in self.needs.props_left() for w in re.findall(r"[a-z]+", req.lower())}
        avoid |= {w for r in (self.needs.items() or [dict(product=str(self.needs.reading.get('product', None) or ''))]) for w in re.findall(r"[a-z]+", r["product"].lower())}
        title_words = re.findall(r"[a-z][a-z'-]{3,}", self.title_of(rec).lower())
        after = title_words if not maker_words[0] in title_words else title_words[title_words.index(maker_words[0]) + 1:]
        words = [w for w in after if not w.strip("'-") in avoid and not "'" in w]
        if not words:
            return None
        text = '{} {}'.format(maker_words[0], words[0])
        if self.ran_query(text):
            return None
        return {**template[0], template[1]: text}

    def walk_rows(self, k, keys, most):
        home = self.walk_home(k)
        book = self.__dict__.get("page_book", {})
        top = max([len(self.walk_carries(w["args"], k)) for w in book.values() if w.get('id_order', None) is True] or [0])
        ids: set[str] = set()
        for key, walk in book.items():
            got = set() if not walk.get('id_order', None) is True else self.walk_carries(walk['args'], k)
            if (got and len(got) == top) or key in keys:
                ids |= {i for i, s in (walk.get('row_shelf', None) or {}).items() if s in home}
        pick = self.walk_pick(k)
        if not pick:
            seen = sorted((float(self.archive.price(i)), i) for i in ids if i in self.archive.read and is_number(self.archive.price(i)))
            pick = None if not seen else (seen[0][1], seen[0][0])
        qty = self.needs.quantity() if not (not k is None and k < len(self.needs.items())) else self.needs.items()[k]['quantity']
        cap = self.needs.spend_cap
        sold: dict[str, bool] = {}

        def visit(d):
            if any(isinstance(v, (str,)) and v in ids and "id" in str(key).lower() for key, v in d.items()):
                sold[self.row_mark(d)] = self.sold_out(d)
        for item in self.archive.items:
            walk_dicts(item, visit)
        out = []
        for ident in ids:
            cost = self.archive.price(ident)
            if ident in self.archive.read or ident in self.refused_lines or sold.get(ident, None) or not is_number(cost):
                continue
            if (is_number(cap) and cost * qty > cap) or (pick and cost >= pick[1]):
                continue
            out.append((cost, ident))
        return [ident for _, ident in sorted(out)[:most]]

    def page_digest(self, obs):
        rows: list[dict[str, Any]] = []

        def visit(d):
            if 4 <= len(d) and any(isinstance(v, (str,)) and v in self.archive.ids and "id" in str(key).lower() for key, v in d.items()):
                rows.append(d)
        walk_dicts(obs, visit, ceiling=3)
        parts = []
        for r in rows[:12]:
            cost = next((v for key, v in r.items() if is_number(v) and re.search(r"price|cost", str(key).lower())), None)
            parts.append(("{} {}{} '{}' [{}]".format(self.row_mark(r), '?' if not not cost is None else cost, '' if not self.sold_out(r) else ' SOLD OUT', self.title_of(r)[:70], self.shelf_of(r))))
        return 'Rows on the page (id, price, title, category): ' + ('none' if not parts else '; '.join(parts))

    def walk_call(self, args, say):
        proposal = (dict(id='walk_{}_{}'.format(len(self.chat), self.wire.tick), type='function', function=dict(name=self.role('search'), arguments=json.dumps(args))))
        self.chat.append(dict(role='assistant', content=say, tool_calls=[proposal]))
        before = len(self.archive.items)
        self.covering = True
        try:
            responses = self.dispatch([proposal], scripted=True)
        finally:
            self.covering = False
        if not responses:
            self.chat.append(dict(role='tool', tool_call_id=proposal['id'], content='Not run.'))
            return False
        cid, text_body = responses[0]
        got = len(self.archive.items) > before
        body = text_body if not got else self.page_digest(self.archive.items[-1])
        note = '' if not '\n[note] ' in text_body else text_body.split('\n[note] ', 1)[1]
        if got and note and re.search(r"shopper|Meaning:|REFUSED", note):
            body += "\n[note] " + note
        self.chat.append(dict(role='tool', tool_call_id=cid, content=body))
        return got

    def run_walk(self, plan, floor, ceil = None, repair = False):
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
            walk = book.get(wkey, None)
            if not walk:
                continue
            query = " ".join(self.free_text(walk["args"]))
            got = 0
            for _ in range(0, n):
                nxt = self.next_page(walk)
                if nxt is None or self.moves_left() <= floor + 1 or self.wire.done or self.wire.broke or self.wire.spare() < CLOSING + 30:
                    break
                if not self.walk_call({**walk["args"], key: nxt}, "The lowest price is requested, yet this store returns search results in a set id order rather than sorted by price, so I am reading the following page of '{}'.".format(query)):
                    break
                got = got + 1
                ran = True
            report.append("'{}': {} additional page(s) fetched, having seen {} out of {} matches".format(query, got, len(walk.get('row_shelf', None) or {}), walk['total']))
        lines: dict[Any, str] = {}
        for k, _, _ in plan:
            if self.moves_left() <= floor + 2 or self.wire.done or self.wire.spare() < CLOSING + 30:
                break
            args = self.line_query(k)
            if args and self.walk_call(args, "I additionally look up the pick's product line, using its manufacturer plus the word that names the line, since other sizes and colours in a line are listed separately and could be cheaper."):
                ran = True
                lines[k] = json.dumps({a: v for a, v in args.items() if key != a}, default=str, sort_keys=True)
                line = self.__dict__.get('page_book', {}).get(lines[k], None)
                more = 0
                while line and more < 2 and line["total"] <= 3 * max(1, line["size"]) and not self.next_page(line) is None \
                        and self.moves_left() > floor + 3 and not self.wire.done and self.wire.spare() > CLOSING + 30:
                    if not self.walk_call({**line["args"], key: self.next_page(line)}, 'I fetched the following page of the line search.'):
                        break
                    more = more + 1
                report.append("the pick's line '{}'".format(' '.join(self.free_text(args))))
        ceil_rows: dict[Any, list[str]] = {}
        for k in dict.fromkeys(ceil or []):
            if self.moves_left() <= (max(floor, self.order_due() + 2) if not repair else floor) + 1 or self.wire.done \
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
        most = (2 if not ceil_rows else CEIL_OPENS) if not batch else batch[2]
        pools = [self.walk_rows(k, [lines.get(k, "")], most) for k, _, _ in plan] + [rows for rows in ceil_rows.values()]
        queue: list[str] = []
        while len(queue) < most and any(pools):
            for pool in pools:
                while pool and pool[0] in queue:
                    pool.pop(0)
                if pool and len(queue) < most:
                    queue.append(pool.pop(0))
        while queue and (len(queue) if not (batch and 2 <= len(queue)) else 1) >= self.moves_left() - floor:
            queue.pop()
        homes = set().union(*(self.walk_home(k) for k in [k for k, _, _ in plan] + list(ceil_rows)))
        self.rank_checked = {mark for mark in self.rank_checked if not self.shelf_known(mark.split('|')[0]) in homes}
        self.order_vetted = False
        summary = ("[cheapest walk] " + "; ".join(report) + ". "
                   + (('Nothing was displayed that is unread, in stock, within the budget, and priced under the pick. ' if not queue else "Next to be opened are the least expensive in-stock rows not yet read that fall within the budget and appeared in these searches, listed beneath each pick's price: " + ', '.join((f'{i} ({self.archive.price(i):g})' for i in queue)) + '. '))
                   + 'Per product, pick the listing with the smallest price among those whose own fields satisfy all the requirements currently active; the rank check is repeated when the cart is called and when the order is placed.')
        if queue:
            self.open_queue = queue
            self.open_recheck = True
            self.open_say = "The cheapest is asked for: I open the cheapest unread rows the walk showed."
            self.open_purpose, self.open_floor = "walk", floor
            self.open_named()
        if self.chat and 'tool' == self.chat[-1].get('role', None):
            self.chat[-1]["content"] = str(self.chat[-1]["content"]) + "\n" + summary + ('\n' + self.status_line() if not queue else '')
        self.log.append(dict(kind='walk', plan=[[k, n] for k, _, n in plan], ceiling=list(ceil_rows), opened=queue))
        return True

    def decline_fits(self):
        ver = str(self.needs.version)
        live = {k.rpartition("|")[0] for k in self.passed if k.rpartition("|")[2] == ver}
        failed = {k.split("|")[0] for k, v in self.rulings.items() if v and k.split("|")[1:2] == [ver]}
        return (live | (set(self.__dict__.get('audit_fits', None) or []) - failed)) - self.refused_lines

    def decline_lacking(self):
        got = {self.shelf_known(i) for i in self.decline_fits()} - {""}
        if not self.needs.items():
            return [None] if not set(self.shelf_labels()) & got else []
        return [k for k, home in enumerate(self.home_shelves(), 0) if home and not home & got]

    def decline_walks(self):
        reqs = self.needs.props_left()

        def stated(args, k):
            return sum(1 for n in self.walk_carries(args, k) if n < len(reqs)
                       and not re.search(r"as shown|same as|the page|sold by|redirect|refer", reqs[n].lower()))
        keys: list[str] = []
        for k in self.decline_lacking():
            opts = sorted((o for o in self.walk_options(k) if stated(o[2]["args"], k)),
                          key=lambda p: (-stated(p[2]["args"], k), -self.walk_score(p[2]["args"], k), p[0], p[1]))
            for _, key, _ in opts:
                if not key in keys:
                    keys.append(key)
                    break
        return keys[:2]

    def row_pools(self, homes, ids, below = None):
        cap = self.needs.spend_cap
        shelf: dict[str, str] = {}
        sold: dict[str, bool] = {}

        def visit(d):
            ident = self.row_mark(d)
            if ident in self.archive.ids and self.shelf_of(d):
                shelf[ident] = self.shelf_of(d)
                if any(re.search(r"stock|avail", str(key).lower()) for key in d):
                    sold[ident] = self.sold_out(d)
        for rec in self.archive.items:
            walk_dicts(rec, visit)
        pools = []
        for n, (home, qty) in enumerate(homes, 0):
            pool = []
            for ident, where in shelf.items():
                cost = self.archive.price(ident)
                if not where in home or ident in self.archive.read or ident in self.refused_lines or sold.get(ident, None) \
                        or (not ids is None and not ident in ids) or not is_number(cost) or (is_number(cap) and cost * qty > cap) \
                        or (not below is None and not cost < below[n]):
                    continue
                pool.append((float(cost), ident))
            pools.append([ident for _, ident in sorted(pool)])
        return pools

    def over_budget_rows(self):
        try:
            rows, cap = self.needs.items(), self.needs.spend_cap
            if not rows or 'cheapest' == self.needs.priority() or not is_number(cap):
                return []
            homes = self.home_shelves()
            fits = self.decline_fits()
            picks: list[float] = []
            shares: list[float] = []
            total = 0.0
            for k, home in enumerate(homes, 0):
                priced = [float(self.archive.price(i)) for i in fits if self.shelf_known(i) in home and is_number(self.archive.price(i))]
                if not home or not priced:
                    return []
                picks.append(min(priced))
                shares.append(min(priced) * rows[k]["quantity"])
                total += min(priced) * rows[k]["quantity"]
            if cap >= total:
                return []
            pools = self.row_pools([(home, rows[k]["quantity"]) for k, home in enumerate(homes, 0)], None, picks)
            pools = [pools[k] for k in sorted(range(0, len(pools)), key=lambda k: -shares[k])]
            return [] if not any(pools) else pools
        except (Exception,) as err:
            self.log.append(dict(kind='over_budget_error', error='{}: {}'.format(type(err).__name__, err)[:300]))
            return []

    def decline_reads(self):
        try:
            batch = self.batch_tool()
            if self.__dict__.get('decline_read_done', None) or self.wire.spare() < CLOSING + 60 or not (self.role("details") or batch) \
                    or self.trolley.units:
                return ""
            opens = 2 if not batch else 1
            room = min(3, self.moves_left() - opens - 6)
            keys = [] if not (room > 0 and self.page_arg() and self.role('search') and self.id_store()) else self.decline_walks()
            cheaper = [] if not self.moves_left() > opens + 6 else self.over_budget_rows()
            if not keys and not cheaper:
                return ""
            self.decline_read_done = True
            self.decline_queue = (keys, room, 6)
            said = []
            if keys:
                book = self.__dict__.get("page_book", {})
                said.append('searches containing required values turned up more listings than were actually read ('
                            + "; ".join("'{}'".format(' '.join(self.free_text(book[key]['args']))) for key in keys) + '), and this store returns results in a set id order rather than ranked by fit or price, meaning matching listings are as likely to be on unread pages as read ones: the code now reads as many as {} of those pages for you and opens the lowest-priced in-stock rows within the budget that they show'.format(room))
            if cheaper:
                said.append("all the listings located for the products, added up, exceed the budget, so the cheapest in-stock rows not yet read, each priced under its product's pick, are also opened for you")
            return ("Before a decline, " + "; ".join(said) + '. Assess those records by meaning against the requirements currently in effect; purchase the one that fits (' + ('the item' if not self.needs.items() else 'one line per product, at its quantity')
                    + "), and decline only if each one fails.")
        except (Exception,) as err:
            self.log.append(dict(kind='decline_reads_error', error='{}: {}'.format(type(err).__name__, err)[:300]))
            return ""

    def run_decline_reads(self):
        keys, most_pages, floor = self.__dict__.pop("decline_queue")
        found = self.page_arg()
        book = self.__dict__.get("page_book", {})
        batch = self.batch_tool()
        opens = 2 if not batch else 1
        read = 0
        walked: list[str] = []
        for key in [] if not found else keys:
            walk = book.get(key, None)
            if not walk:
                continue
            walked.append(key)
            query = " ".join(self.free_text(walk["args"]))
            while most_pages > read and self.moves_left() > floor + opens and not self.wire.done and not self.wire.broke \
                    and self.wire.spare() > CLOSING + 30:
                nxt = self.next_page(walk)
                if nxt is None or not self.walk_call({**walk["args"], found[0]: nxt}, "Prior to declining, I fetched the following page of '{}': results in this store come in a set id order, so matching items could be on pages not yet read.".format(query)):
                    break
                read = read + 1
        seen: set[str] = set()
        for key in walked:
            seen |= set((book.get(key, None) or {}).get('row_shelf', None) or {})
        rows = self.needs.items()
        homes = [(self.walk_home(k), self.needs.quantity() if not (not k is None and k < len(rows)) else rows[k]['quantity'])
                 for k in self.decline_lacking()]
        pools = ([] if not walked else self.row_pools(homes, seen)) + self.over_budget_rows()
        most = 2 if not batch else batch[2]
        queue: list[str] = []
        while len(queue) < most and any(pools):
            for pool in pools:
                while pool and pool[0] in queue:
                    pool.pop(0)
                if pool and len(queue) < most:
                    queue.append(pool.pop(0))
        while queue and (len(queue) if not (batch and 2 <= len(queue)) else 1) >= self.moves_left() - floor:
            queue.pop()
        self.log.append(dict(kind='decline_reads', pages=read, opened=queue))
        if queue and not self.wire.done:
            self.open_queue = queue
            self.open_say = 'Prior to refusing, I read the lowest-priced unopened rows within budget that appear across those pages and picks.'
            self.open_purpose, self.open_floor = "decline", floor
            self.open_named()
        elif self.chat and 'tool' == self.chat[-1].get('role', None):
            self.chat[-1]["content"] = str(self.chat[-1]["content"]) + '\nThere, no unread in-stock row that fit the budget was displayed.\n' + self.status_line()
        return read > 0 or bool(queue)

    def ref_register(self):
        reading = self.needs.reading
        infos = [] if not isinstance(reading.get('reference_targets', None), (list,)) else reading.get('reference_targets', None)
        infos = [t for t in infos if isinstance(t, (dict,))]
        listed = set() if not isinstance(reading.get('references', None), (list,)) else {str(r) for r in reading.get('references', None) or []}
        for req in list(self.needs.written):
            value = str(req).partition(":")[2]
            if ref_text(value) or (str(req) in listed and re.search(DIFFER_WORDS, value.lower())):
                self.ref_add(str(req), infos, self.request, "request")
        try:
            self.ref_prune()
        except (Exception,) as err:
            self.log.append(dict(kind='ref_prune_error', error='{}: {}'.format(type(err).__name__, err)[:300]))

    def ref_prune(self):
        low = re.sub(r"\s+", " ", self.request.lower().replace("\u2019", "'"))
        spans = []
        for ref in self.refs:
            seller = str(ref['target'].get('seller', None) or '').lower().strip(None)
            if not seller or 'request' != ref.get('origin', None):
                continue
            for m in re.finditer(re.escape(seller), low):
                begin = max(low.rfind(". ", 0, m.start()), low.rfind("! ", 0, m.start()), low.rfind("? ", 0, m.start())) + 1
                anchors = [a.start() for a in re.finditer(r"page (?:of|for)|pointed to|listing (?:of|from|by)|sold by|\bfrom\b", low[begin:m.start()])]
                start = m.start() if not anchors else begin + anchors[-1]
                end = re.search(r"[.!?](?:\s|$)", low[m.end():])
                spans.append((start, len(low) if not end else m.end() + end.start()))
        if not spans:
            return

        def inside(pos):
            return any(a <= pos < b for a, b in spans)
        try:
            cuts = r"[.!?](?=\s)|;|\b(?:first|second|third|fourth|fifth|lastly|finally)\b\s*,"
            tight = []
            for a, b in spans:
                nxt = re.search(cuts, low[a:b])
                tight.append((a, b if not nxt else a + nxt.start()))
            rows = [] if not isinstance(self.needs.reading.get('items', None), (list,)) else self.needs.reading.get('items', None)
            drop = []
            for row in rows:
                prod = '' if not isinstance(row, (dict,)) else str(row.get('product', None) or '').lower().strip(None)
                if 3 > len(prod):
                    continue
                hits = [m.start() for m in re.finditer(r"(?<![a-z0-9])" + re.escape(prod) + r"(?![a-z0-9])", low)]
                if hits and all(any(x <= h < y for x, y in tight) for h in hits):
                    drop.append(row)
            if drop and len(drop) < len(rows):
                kinds = [kind_words(r["product"]) for r in drop]

                def tagged(req):
                    tags = re.findall(r"\(([^)]*)\)", str(req).partition(":")[0])
                    return bool(tags) and any(kind_words(tags[-1]) == k for k in kinds if k)
                self.needs.reading["items"] = [r for r in rows if not r in drop]
                self.needs.written = [r for r in self.needs.written if not tagged(r)]
                self.needs.reading["requirements"] = list(self.needs.written)
                self.log.append(dict(kind='ref_item_prune', dropped=[str(r.get('product', None))[:40] for r in drop]))
        except (Exception,) as err:
            self.log.append(dict(kind='ref_item_prune_error', error='{}: {}'.format(type(err).__name__, err)[:200]))
        moved = []
        for req in list(self.needs.written):
            head, _, value = str(req).partition(":")
            value, prop = value.strip(None).lower(), re.split('\\s*\\(', head)[0].strip(None).lower()
            if not value or 2 > len(value) or ref_text(value) or any(r["req"] == req for r in self.refs):
                continue
            hits = [m.start() for m in re.finditer(r"(?<![a-z0-9])" + re.escape(value) + r"(?![a-z0-9])", low)]
            if not hits:
                continue
            paired = [h for h in hits if prop and prop in low[max(0, h - 40): h + len(value) + 40]]
            use = paired or hits
            if all(inside(h) for h in use):
                moved.append(req)
        if not moved or len(moved) >= len(self.needs.written):
            return
        reading = self.needs.reading
        self.needs.written = [r for r in self.needs.written if not r in moved]
        reading["requirements"] = list(self.needs.written)
        gone = {str(r).strip(None).lower() for r in moved}
        for row in reading.get('items', None) or []:
            if isinstance(row, (dict,)) and isinstance(row.get('requirements', None), (list,)):
                row["requirements"] = [r for r in row["requirements"] if not str(r).strip(None).lower() in gone]
        self.log.append(dict(kind='ref_prune', moved=moved))

    def ref_untracked(self, refs, tracked):
        try:
            return [r for r in refs if not any(self.ref_match(t, r) and ('resolved' == t.get('status', None) or t.get('chain', None)) for t in tracked)]
        except (Exception,):
            return refs

    def ref_take(self):
        try:
            for ref in self.refs:
                if 'resolved' == ref['status'] and any(self.ref_match(ref, r) and r != ref["now"] for r in self.needs.props_left()):
                    self.ref_swap(ref, ref["now"])
            news = list(self.needs.__dict__.get('ref_news', None) or [])
            self.needs.ref_news = []
            for item in news:
                ref = self.ref_add(str(item.get('need', None) or ""), list(item.get('infos', None) or []), str(item.get('line', None) or ""), "reply")
                if ref:
                    refs = self.needs.reading.get('references', None)
                    if not isinstance(refs, (list,)):
                        refs = self.needs.reading["references"] = []
                    if not ref['req'] in refs:
                        refs.append(ref["req"])
        except (Exception,) as err:
            self.log.append(dict(kind='ref_take_error', error='{}: {}'.format(type(err).__name__, err)[:300]))

    def ref_add(self, req, infos, source, origin):
        head, _, value = str(req).partition(":")
        prop = re.split('\\s*\\(', head)[0].strip(None)
        tags = re.findall(r"\(([^)]*)\)", head)
        tag = '' if not tags else tags[-1].strip(None)
        if not prop or junk_key(prop):
            return None
        for old in self.refs:
            if old["req"] == req or (same_prop(old["prop"], prop) and kind_words(old["tag"]) == kind_words(tag)):
                return None
        info = None
        for cand in infos:
            if same_prop(str(cand.get('property', None) or ""), prop) and (not tag or not cand.get('product', None) or not self.needs.items()
                                                                    or kind_words(cand.get('product', None)) & kind_words(tag)):
                info = cand
                break
        tgt = self.ref_target(info, value, source, prop)
        if not tgt:
            self.log.append(dict(kind='ref_skip', req=str(req)[:200]))
            return None
        rel, tol, unit = ref_relation(value)
        if rel == 'same' and isinstance(info, (dict,)) and 'within' == str(info.get('relation', None) or '').lower() and is_number(info.get('tolerance', None)):
            rel, tol, unit = "within", float(info["tolerance"]), str(info.get('unit', None) or "")
        told = '' if not isinstance(info, (dict,)) else str(info.get('relation', None) or '').lower()
        span = ref_span(prop, source).lower()
        cut = -1 if not str(tgt.get('seller', None) or '').strip(None) else span.find(str(tgt.get('seller', None) or '').lower())
        if rel == 'same' and (re.search(r"differ|not the same|exclu|apart|other than|unlike", told)
                              or (cut > 0 and re.search(DIFFER_WORDS, span[:cut]))):
            rel, tol, unit = "differ", None, ""
        if rel != 'pair' and (told == 'pair' or re.search(r"\bcounterpart\b|\bgoes with\b|\bpairs? with\b|\bgo together with\b",
                                                          span if not cut > 0 else span[:cut])):
            rel, tol, unit = "pair", None, ""
        clause = ref_clause(source, tgt.get('seller', None))
        if not rel in ('pair', 'within', 'at most', 'at least') and re.search(COMPAT_WORDS, clause, re.I) \
                and not re.search(DIFFER_WORDS, clause, re.I):
            rel, tol, unit = "pair", None, ""
            self.log.append(dict(kind='ref_compat', seller=str(tgt.get('seller', None) or '')[:40]))
        ref = (dict(req=str(req), head=head.strip(None), prop=prop, tag=tag, relation=rel, tol=tol, unit=unit, target=tgt, first=dict(tgt), chain=[], seen={self.ref_key(tgt)}, judged=set(), queries=[], pages=0, hops=0, tries=0, fails=0, status='open', value='', goes_with=[], now='', origin=origin, unindexed=False))
        self.refs.append(ref)
        self.log.append(dict(kind='ref_open', req=str(req)[:200], target=tgt, relation=rel, origin=origin))
        return ref

    def ref_target(self, info, value, source, prop):
        seller, kind, details = "", "", []
        if isinstance(info, (dict,)):
            seller = str(info.get('seller', None) or '').strip(None)
            kind = str(info.get('kind', None) or '').strip(None)
            raw = [] if not isinstance(info.get('details', None), (list,)) else info.get('details', None)
            details = [str(d).strip(None) for d in raw if isinstance(d, (str, int, float)) and not isinstance(d, (bool,)) and str(d).strip(None)]
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
        return dict(seller=seller, kind=(kind if not parsed else kind or parsed['kind'])[:80], details=details)

    def ref_seller_ok(self, seller, source):
        name = str(seller or '').strip(None)
        if 3 > len(re.sub('[^A-Za-z]', '', name)):
            return False
        if not re.search(r"(?<![a-z0-9])" + re.escape(name.lower()) + r"(?:'s)?(?![a-z0-9])", str(source or "").lower()):
            return False
        kinds: set[str] = set()
        for row in self.needs.items() or [dict(product=str(self.needs.reading.get('product', None) or ''))]:
            kinds |= kind_words(row["product"])
        for pspec in ((self.shapes.get(self.role('search') or '', None) or {}).get('properties', None) or {}).values():
            if isinstance(pspec, (dict,)) and isinstance(pspec.get('enum', None), (list,)):
                kinds |= {w for v in pspec["enum"] for w in kind_words(v)}
        words = kind_words(name)
        return bool(words) and not kinds >= words

    @staticmethod
    def ref_key(tgt):
        return (ref_name(tgt.get('seller', None) or ""), tuple(sorted({x for d in tgt.get('details', None) or [] for x in ref_toks(d)})))

    @staticmethod
    def ref_desc_of(tgt):
        return ('the {} sold by {}'.format(tgt.get('kind', None) or 'listing', tgt.get('seller', None))
                + ('' if not tgt.get('details', None) else ' (' + '; '.join(tgt.get('details', None) or []) + ')'))

    def ref_desc(self, ref):
        return self.ref_desc_of(ref["target"])

    def ref_index(self):
        memo = self.__dict__.get('ref_memo', None)
        if memo and memo[0] == len(self.archive.items):
            return memo[1]
        rows: dict[str, dict[str, Any]] = {}
        seen_at: dict[str, int] = {}
        for n, rec in enumerate(self.archive.items, 0):
            def visit(d):
                ident = self.row_mark(d)
                if ident in self.archive.ids and (4 <= len(d) or any(isinstance(v, (dict,)) and 2 <= len(v) for v in d.values())):
                    rows.setdefault(ident, {}).update(d)
                    seen_at[ident] = n
            walk_dicts(rec, visit)
        self.ref_memo = (len(self.archive.items), rows)
        self.ref_seen_at = seen_at
        return rows

    def ref_sellers(self, row):
        learned = self.seller_keys()
        return {ref_name(v) for k, v in row.items() if isinstance(v, (str,)) and v.strip(None) and 60 >= len(v)
                and (re.search(SELLER_KEYS, str(k).lower()) or str(k) in learned) and not 'id' in str(k).lower()
                and not re.search(r"note|text|desc|comment|detail|info|message|section", str(k).lower())} - {""}

    def seller_keys(self):
        try:
            names = {ref_name(t.get('seller', None)) for r in self.__dict__.get('refs', None) or [] for t in (r.get('target', None) or {}, r.get('first', None) or {})
                     if isinstance(t, (dict,))} - {""}
            mark = (len(self.archive.items), tuple(sorted(names)))
            memo = self.__dict__.get('seller_memo', None)
            if memo and memo[0] == mark:
                return memo[1]
            rows = self.ref_index()
            noted: set[str] = set()
            vocab: set[str] = set()
            for row in rows.values():
                for v in row.values():
                    if isinstance(v, (dict,)):
                        vocab.update(ref_name(x) for x in v.values() if isinstance(x, (str,)))
                for sentence in re.split(r"(?<=[.!?;])\s+", self.ref_texts(row, desc=False)):
                    run: list[str] = []
                    for word in re.findall(r"[A-Za-z][A-Za-z'-]*", sentence)[1:] + [""]:
                        if word[:1].isupper() and re.search(r"[a-z]", word):
                            run.append(word)
                            continue
                        for n in range(0, len(run)):
                            for j in range(n + 1, min(len(run), n + 3) + 1):
                                noted.add(ref_name(" ".join(run[n:j])))
                        run = []
            noted = {w for w in noted if 4 <= len(w)} - vocab
            keys: set[str] = set()
            for row in rows.values():
                for k, v in row.items():
                    low = str(k).lower()
                    if isinstance(v, (str,)) and 0 < len(v) <= 60 and not 'id' in low and not re.search(
                            r"title|heading|name|label|caption|categor|shelf|department|aisle|note|text|desc|comment|detail|info|"
                            r"message|section|url|image|summary|overview", low) and (
                            ref_name(v) in names or (v.strip(None)[:1].isupper() and 3 >= len(v.split()) and ref_name(v) in noted)):
                        keys.add(str(k))
            self.seller_memo = (mark, keys)
            return keys
        except (Exception,):
            return set()

    @staticmethod
    def ref_texts(row, desc = True):
        notes, other = [], []
        for k, v in row.items():
            name = str(k).lower()
            if not isinstance(v, (str,)) or 3 > len(v.split()) or name == 'id' or name.endswith("id") \
                    or re.search(r"title|heading|headline|name|label|caption|categor|shelf|department|url|image", name):
                continue
            (notes if not re.search('desc|overview|summary|about', name) else other).append(v)
        return "\n".join(notes or ([] if not desc else other))

    def ref_parts(self, row):
        notes = re.sub(r"\b(?:not|no|never|without|nor)\b(?:\W+[A-Za-z0-9.:/&'-]+){1,3}", " ", self.ref_texts(row, desc=False), flags=re.I)
        keys, vals = [], [self.title_of(row), notes]
        maps = [v for v in row.values() if isinstance(v, (dict,))]
        for fields in maps or [{k: v for k, v in row.items() if isinstance(v, (str, int, float)) and not isinstance(v, (bool,))
                                and not re.search(r"id$|title|shop|seller|brand|price|amount|desc|overview|note", str(k).lower())}]:
            for k, x in fields.items():
                keys.append(re.sub(r"([a-z])([A-Z])", r"\1 \2", str(k)).replace("_", " "))
                if isinstance(x, (str, int, float)) and not isinstance(x, (bool,)):
                    vals.append(str(x))
        return set(ref_toks(" ".join(keys))), set(ref_toks(" | ".join(vals)))

    def ref_shows(self, row, detail, parts = None):
        keys, vals = parts or self.ref_parts(row)
        want = {t for t in ref_toks(detail) if 3 <= len(t) and not re.search(r"\d", t)} - keys
        if want and not want & vals:
            return False
        figs = ref_figs(detail)
        if figs:
            mine = ref_figs(" | ".join([self.title_of(row)] + [str(x) for v in row.values() if isinstance(v, (dict,))
                                                               for x in v.values() if isinstance(x, (str, int, float))]))
            for fig, unit in figs:
                same = [f for f, u in mine if unit == u]
                if same and not any(1e-09 > abs(f - fig) for f in same):
                    return False
        return True

    @staticmethod
    def ref_field(row, prop):
        for strict in (True, False):
            for val in row.values():
                if isinstance(val, (dict,)):
                    for k, x in val.items():
                        if isinstance(x, (str, int, float)) and not isinstance(x, (bool,)) and str(x).strip(None) \
                                and (same_prop(k, prop) if not strict else strict_prop(k, prop)):
                            return str(x).strip(None)
        return ""

    def ref_ids_add(self, ids):
        self.ref_ids = set(self.__dict__.get('ref_ids', None) or set()) | set(ids)

    def ref_foreign(self):
        try:
            ids = self.__dict__.get('ref_ids', None) or set()
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
            return {i for i in ids if not i in mine and self.shelf_of(rows.get(i, None) or {}) and not self.shelf_of(rows.get(i, None) or {}) in homes}
        except (Exception,):
            return set()

    def ref_cands(self, ref):
        tgt = ref["target"]
        name = ref_name(tgt["seller"])
        kinds = kind_words(tgt["kind"])
        rows = self.ref_index()
        seen_at = self.__dict__.get('ref_seen_at', None) or {}
        scored = []
        for ident, row in rows.items():
            if not name in self.ref_sellers(row):
                continue
            hay = set(ref_toks(self.title_of(row) + " " + " ".join(str(v) for k, v in row.items()
                                                                  if isinstance(v, (str,)) and re.search(r"desc|overview|summary", str(k).lower()))))
            shown = sum(1 for d in tgt["details"] if set(ref_toks(d)) and set(ref_toks(d)) <= hay)
            scored.append((-(0 if not kinds & kind_words(self.shelf_of(row)) else 2) - shown, -seen_at.get(ident, 0), ident, shown))
        scored.sort()
        ref["shows"] = {ident: shown for _, _, ident, shown in scored}
        return [ident for _, _, ident, _ in scored]

    def ref_modes(self, needed, labels):
        groups: dict[str, dict[str, Any]] = {}
        for n, walk in enumerate(self.__dict__.get('page_book', {}).values(), 0):
            args = {} if not isinstance(walk.get('args', None), (dict,)) else walk.get('args', None)
            texts = self.free_text(args)
            keys = [k for k, v in args.items() if isinstance(v, (str,)) and v in texts and v.strip(None)]
            if 1 != len(texts) or 1 != len(keys):
                continue
            other = {k: v for k, v in args.items() if k != keys[0] and (k in needed or not (isinstance(v, (str,)) and v.strip(None).lower() in labels))}
            group = groups.setdefault(json.dumps(other, default=str, sort_keys=True), dict(tpl=(other, keys[0]), seen=[], at=n))
            group["seen"].append(({self.stem(w) for w in re.findall(r"[a-z0-9]+", texts[0].lower())}, tuple(walk.get('first', None) or ())))
        ranked, blind = [], set()
        for mode, group in groups.items():
            seen = group["seen"]
            if any(2 <= len(f1) and f2 == f1 and not (w1 & w2) for i, (w1, f1) in enumerate(seen, 0) for w2, f2 in seen[i + 1:]):
                blind.add(mode)
                continue
            ranked.append((-len({tuple(sorted(w)) for w, _ in seen}), group["at"], group["tpl"]))
        ranked.sort(key=lambda x: (x[0], x[1]))
        return [tpl for _, _, tpl in ranked], blind

    def ref_template(self):
        jschema = self.shapes.get(self.role('search') or '', None) or {}
        needed = set() if not isinstance(jschema.get('required', None), (list,)) else set(jschema.get('required', None) or [])
        labels = {self.shelf_of(r) for r in self.ref_index().values()} - {""}
        modes, blind = self.ref_modes(needed, labels)
        tpl = None if not modes else modes[0]
        paging = self.page_arg()
        for entry in [] if tpl else reversed(self.log):
            for act in [] if not 'env' == entry.get('kind', None) else reversed(entry.get('actions', None) or []):
                args = None if not act.get('name', None) == self.role('search') else act.get('args', None)
                if not isinstance(args, (dict,)):
                    continue
                texts = [k for k, v in args.items() if isinstance(v, (str,)) and v.strip(None) and v in self.free_text(args)]
                if 1 == len(texts):
                    other = {k: v for k, v in args.items() if k != texts[0] and not (paging and k == paging[0])}
                    kept = {k: v for k, v in other.items() if k in needed or not (isinstance(v, (str,)) and v.strip(None).lower() in labels)}
                    if json.dumps(kept, default=str, sort_keys=True) in blind:
                        continue
                    tpl = (other, texts[0])
                    break
            if tpl:
                break
        if tpl:
            other, key = tpl
            return {k: v for k, v in other.items() if k in needed or not (isinstance(v, (str,)) and v.strip(None).lower() in labels)}, key
        props = {} if not isinstance(jschema.get('properties', None), (dict,)) else jschema.get('properties', None)
        tok = self.token_key()
        texts = [k for k, p in props.items() if isinstance(p, (dict,)) and 'string' == p.get('type', None) and not p.get('enum', None) and tok != k]
        if 1 == len(texts) and not (needed - {texts[0]}):
            return {}, texts[0]
        return None

    def ref_words(self, ref):
        keys: set[str] = set()
        for row in self.ref_index().values():
            for val in row.values():
                if isinstance(val, (dict,)):
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
                        if 2 <= len(w):
                            figures.append(w)
                    elif 3 <= len(w) and w == whole.strip(".,;:!?()[]{}\"'") and not w in keys and not w.rstrip('s') in keys:
                        words.append(w)
        words.sort(reverse=True, key=len)
        lasts = []
        skeys = self.store_keys()
        split = [[w for w in re.findall(r"[a-z0-9]+", str(d).lower()) if not w in LINK_WORDS] for d in ref["target"]["details"]]
        pairs = 0 == ref.get('hops', 0) and 1 <= len(split) and all(2 <= len(parts) for parts in split)
        for parts in split:
            if 2 <= len(parts) and parts[-1] in words and (pairs or any(" ".join(parts[:j]) in skeys for j in range(1, len(parts)))):
                lasts.append(parts[-1])
        words = [w for w in lasts if w in words] + [w for w in words if not w in lasts]
        return list(dict.fromkeys(joined + figures + words))

    def store_keys(self):
        mark = len(self.archive.items)
        memo = self.__dict__.get('store_keys_memo', None)
        if memo and memo[0] == mark:
            return memo[1]
        keys: set[str] = set()

        def visit(d):
            for v in d.values():
                if isinstance(v, (dict,)) and 2 <= sum((1 for x in v.values() if isinstance(x, (str, int, float)))):
                    for k in v:
                        name = " ".join(re.findall(r"[a-z0-9]+", re.sub(r"([a-z])([A-Z])", r"\1 \2", str(k)).lower()))
                        if name:
                            keys.add(name)
        for rec in self.archive.items:
            walk_dicts(rec, visit)
        self.store_keys_memo = (mark, keys)
        return keys

    def ref_plans(self, ref):
        width = max(2, self.lead_words or 3)
        seller = " ".join(re.findall(r"[A-Za-z0-9]+", re.sub(r"'s\b", "", str(ref["target"]["seller"]))))
        words = self.ref_words(ref)
        kind = [w for w in re.findall(r"[a-z]{3,}", str(ref["target"]["kind"]).lower()) if not w in LINK_WORDS][-1:]
        if not ref.get('unindexed', None):
            plans = ([] if not (words and len(seller.split()) < width) else [seller + ' ' + words[0]]) + [seller]
        else:
            plans = [] if not (words or kind) else [' '.join((words[:1] + kind)[:width])]
        return [p for p in plans if p.strip(None) and not p.lower() in ref['queries']]

    def ref_page(self, ref):
        found = self.page_arg()
        if not found or 3 <= ref['pages']:
            return None
        name = ref_name(ref["target"]["seller"])
        for walk in reversed(list(self.__dict__.get("page_book", {}).values())):
            if not name or not name in ref_name(' '.join(self.free_text(walk['args']))):
                continue
            nxt = self.next_page(walk)
            if not nxt is None:
                return {**walk["args"], found[0]: nxt}
        return None

    def ref_room(self):
        blind = [r for r in self.__dict__.get('refs', None) or [] if 'open' == r.get('status', None)]
        if blind and all('differ' == r.get('relation', None) and not r.get('chain', None) and not self.ref_cands(r) for r in blind) \
                and self.__dict__.get("ref_steps", 0) >= REF_DIFFER_BLIND:
            return False
        cap = max(8, int(REF_STEPS * self.step_cap))
        floor = self.ref_floor()
        opens = [r for r in self.__dict__.get('refs', None) or [] if 'open' == r.get('status', None)]
        if opens and all('differ' == r.get('relation', None) for r in opens):
            cap = max(6, int(REF_DIFFER_STEPS * self.step_cap))
        if not (self.wire.spare() > CLOSING + 60 and not self.wire.spent_out() and not self.wire.done and not self.wire.broke):
            return False
        if self.__dict__.get("ref_steps", 0) >= cap and self.reply_ref_open() and self.moves_left() > self.order_due() + 3:
            return True
        if self.__dict__.get("ref_steps", 0) >= cap:
            return self.ref_hop_room()
        if self.ref_waits(floor):
            return False
        return self.moves_left() > floor or self.ref_hop_room()

    def link_notes_room(self):
        return (self.moves_left() > self.order_due() + 2 and self.wire.spare() > CLOSING + 40 and not self.wire.spent_out()
                and not self.wire.done and not self.wire.broke)

    def ref_floor(self):
        return max(max(FIRST_ORDER_BY, min(FIRST_ORDER_DUE, self.step_cap // 4 + 1)) + 10, self.order_due() + 7)

    def ref_product(self, ref):
        return None if not (self.needs.items() and ref.get('tag', None)) else self.product_of('x ({})'.format(ref['tag']))

    def ref_waits(self, floor = None):
        rows = self.needs.items()
        opens = [r for r in self.__dict__.get('refs', None) or [] if 'open' == r.get('status', None)]
        if not rows or not opens or not all('differ' == r.get('relation', None) and r.get('chain', None) for r in opens):
            return False
        if self.moves_left() <= (floor if not floor is None else self.ref_floor()) + 4:
            return False
        try:
            owners = {self.ref_product(r) for r in opens}
            lacking = [k for k in range(0, len(rows)) if not any(e["fit"] for e in self.ed_pool(k))]
        except (Exception,) as err:
            self.log.append(dict(kind='ref_wait_error', error='{}: {}'.format(type(err).__name__, err)[:300]))
            return False
        known = owners - {None}
        rest = [k for k in lacking if not k in known]
        waits = bool(rest) if not None in owners else 2 <= len(rest)
        if waits and self.__dict__.get('ref_wait_said', None) != tuple(rest):
            self.ref_wait_said = tuple(rest)
            self.log.append(dict(kind='ref_wait', lacking=rest))
        return waits

    def ref_hop_room(self):
        if self.moves_left() <= self.order_due() + 2:
            return False
        steps = self.__dict__.get("ref_steps", 0)
        for ref in self.__dict__.get('refs', None) or []:
            tgt = ref.get('target', None) or {}
            if 'open' != ref.get('status', None) or 1 > ref.get('hops', 0) or not str(tgt.get('seller', None) or '').strip(None) \
                    or not str(tgt.get('kind', None) or '').strip(None) or not tgt.get('details', None):
                continue
            start = ref.get('late_from', None)
            if not start is None:
                if 3 > steps - start:
                    return True
                continue
            if ref.get('queries', None) or not self.ed_others_verified(ref):
                continue
            ref["late_from"] = steps
            self.log.append(dict(kind='ref_late', prop=ref.get('prop', None), to=str(tgt.get('seller', None))[:80]))
            return True
        return False

    def ed_others_verified(self, ref):
        rows = self.needs.items()
        if not rows:
            return True
        try:
            bare = [k for k in range(0, len(rows)) if not any(e["verified"] and e["fit"] for e in self.ed_pool(k))]
        except (Exception,):
            return False
        own = self.ref_product(ref)
        return 1 >= len(bare) if not not own is None else not [k for k in bare if own != k]

    def ref_calls_ok(self):
        return (self.__dict__.get("ref_calls", 0) < REF_CALLS and self.wire.spare() > CLOSING + 40 and not self.wire.spent_out()
                and self.wire.spend < 0.8 * SPEND_CAP * self.wire.__dict__.get("scale", 1.0))

    def ref_match(self, ref, req):
        head, _, value = str(req).partition(":")
        if str(req) in (ref["req"], ref.get('now', None)):
            return True
        pointing = ref_text(value) or ('differ' == ref.get('relation', None) and re.search(DIFFER_WORDS, value.lower()))
        if not pointing or not same_prop(re.split(r"\s*\(", head)[0], ref["prop"]):
            return False
        tags = re.findall(r"\(([^)]*)\)", head)
        return not tags or not ref["tag"] or bool(kind_words(tags[-1]) & kind_words(ref["tag"]))

    def ref_live(self, ref):
        left = self.needs.props_left()
        if any(self.ref_match(ref, r) or same_prop(re.split(r"\s*\(", str(r).partition(":")[0])[0], ref["prop"]) for r in left):
            return True
        return not any(same_prop(re.split(r"\s*\(", str(r).partition(":")[0])[0], ref["prop"]) for r in self.needs.let_go)

    def link_repair(self):
        try:
            for ref in self.refs:
                if 'resolved' != ref.get('status', None) or 'pair' != ref.get('relation', None) or ref.get('goes_with', None) or not ref.get('now', None):
                    continue
                pairs = self.stated_pairings(str(ref.get('value', None) or ""), ref)
                if not pairs:
                    continue
                old = ref["now"]
                ref["goes_with"] = pairs
                new_req = '{}: {}'.format(ref['head'], self.ref_value(ref))
                nd = self.needs
                for name in ("written", "uncovered", "resolved"):
                    setattr(nd, name, [r if not old == r else new_req for r in getattr(nd, name)])
                if not nd.now_reqs is None:
                    nd.now_reqs = [r if not old == r else new_req for r in nd.now_reqs]
                if isinstance(nd.reading.get('requirements', None), (list,)):
                    nd.reading["requirements"] = list(nd.written)
                old_v, new_v = old.partition(':')[2].strip(None), new_req.partition(':')[2].strip(None)
                for row in nd.reading.get('items', None) or []:
                    if isinstance(row, (dict,)) and isinstance(row.get('requirements', None), (list,)):
                        row["requirements"] = [(r if not (str(r).partition(':')[2].strip(None) == old_v and same_prop(re.split('\\s*\\(', str(r).partition(':')[0])[0], ref['prop'])) else '{}: {}'.format(str(r).partition(':')[0].strip(None), new_v)) for r in row["requirements"]]
                ref["now"] = new_req
                nd.version = nd.version + 1
                self.log.append(dict(kind='ref_repaired', req=old[:200], now=new_req))
                self.ref_tell('RESOLVED AGAIN: a note that has been read since then states that {} is paired with {}, meaning the shopper wants that counterpart, and the requirement now reads: {}.'.format(ref['value'], ' and '.join(pairs), new_req))
        except (Exception,) as err:
            self.log.append(dict(kind='ref_repair_error', error='{}: {}'.format(type(err).__name__, err)[:300]))

    def gate_rules_step(self):
        try:
            gate_rules = self.__dict__.get('gate_rules', None)
            if gate_rules is None:
                raw = self.needs.reading.get('conditions', None)
                gate_rules = []
                for c in [] if not isinstance(raw, (list,)) else raw:
                    if not isinstance(c, (dict,)):
                        continue
                    req = str(c.get('requirement', None) or '').strip(None)
                    vals = [] if not isinstance(c.get('if_values', None), (list,)) else [str(v).strip(None).lower() for v in c.get('if_values', None) or [] if str(v).strip(None)]
                    if not ':' in req or not vals or not str(c.get('if_property', None) or '').strip(None):
                        continue
                    gate_rules.append((dict(req=req, product=str(c.get('product', None) or ''), if_product=str(c.get('if_product', None) or ''), if_property=str(c.get('if_property', None)).strip(None), if_values=vals, state=None, seen='')))
                self.gate_rules = gate_rules
                for c in gate_rules:
                    self.rule_unbind(c, strip=True)
            for c in gate_rules:
                state, seen = self.rule_verdict(c)
                c["seen"] = seen
                if state is None or state == c["state"]:
                    continue
                prev, c["state"] = c["state"], state
                if state:
                    self.rule_bind(c)
                elif prev:
                    self.rule_unbind(c)
                self.needs.version += 1
                self.log.append(dict(kind='cond', req=c['req'][:160], state=state, seen=seen[:80]))
                self.ref_tell("CONDITION {}: in the cart, the {} displays {} '{}', which makes '{}' {}.".format('NOT MET' if not state else 'MET', c['if_product'], c['if_property'], seen, c['req'], 'not in force' if not state else 'now in force: re-check that line'))
        except (Exception,) as err:
            self.log.append(dict(kind='cond_error', error='{}: {}'.format(type(err).__name__, err)[:300]))

    def rule_product(self, product):
        rows = self.needs.items()
        words = kind_words(product)
        best = None if not rows else max(range(0, len(rows)), key=lambda k: len(words & kind_words(rows[k].get('product', None) or '')), default=None)
        return None if not (not best is None and words & kind_words(rows[best].get('product', None) or '')) else best

    def rule_verdict(self, c):
        k = self.rule_product(c["if_product"])
        lines = [i for i in self.trolley.units if (True if not not k is None else self.product_index(i) == k)]
        if not lines:
            return None, ""
        row = self.ref_index().get(lines[-1], None) or {}
        seen = self.ref_field(row, c["if_property"])
        if not seen:
            return None, ""
        low = seen.lower()
        met = any(low == v or (set(ref_toks(v)) and set(ref_toks(v)) <= set(ref_toks(low))) for v in c["if_values"])
        return met, seen

    def rule_owns(self, c, req, strip):
        text = str(req)
        if text.strip(None).lower() == c["req"].lower():
            return True
        if not strip:
            return False
        head, mine = text.partition(":")[0], c["req"].partition(":")[0]
        same = same_prop(re.split(r"\s*\(", head)[0], re.split(r"\s*\(", mine)[0])
        return same and bool(re.search(r"\b(?:only )?if\b|\bshould\b|\bturns? out\b|\bprovided\b|\bin case\b", text.lower()))

    def rule_bind(self, c):
        nd = self.needs
        if not c['req'] in nd.written:
            nd.written.append(c["req"])
        if not nd.now_reqs is None and not c['req'] in nd.now_reqs:
            nd.now_reqs.append(c["req"])
        if isinstance(nd.reading.get('requirements', None), (list,)):
            nd.reading["requirements"] = list(nd.written)
        k = self.rule_product(c["product"])
        rows = nd.reading.get('items', None) or []
        if not k is None and k < len(rows) and isinstance(rows[k], (dict,)) and isinstance(rows[k].get('requirements', None), (list,)) \
                and not c['req'] in rows[k]['requirements']:
            rows[k]["requirements"].append(c["req"])

    def rule_unbind(self, c, strip = False):
        nd = self.needs
        nd.written = [r for r in nd.written if not self.rule_owns(c, r, strip)]
        if not nd.now_reqs is None:
            nd.now_reqs = [r for r in nd.now_reqs if not self.rule_owns(c, r, strip)]
        if isinstance(nd.reading.get('requirements', None), (list,)):
            nd.reading["requirements"] = list(nd.written)
        for row in nd.reading.get('items', None) or []:
            if isinstance(row, (dict,)) and isinstance(row.get('requirements', None), (list,)):
                row["requirements"] = [r for r in row["requirements"] if not self.rule_owns(c, r, strip)]

    def rule_status_text(self):
        out = []
        for c in self.__dict__.get('gate_rules', None) or []:
            now = ('undecided until a {} is in the cart'.format(c['if_product']) if not not c['state'] is None else 'not in force' if not c['state'] else 'IN FORCE')
            out.append("'{}' only if the {}'s {} is {}: {}".format(c['req'], c['if_product'], c['if_property'], ' or '.join(c['if_values']), now)
                       + ('' if not c.get('seen', None) else " (the cart's {} shows '{}')".format(c['if_product'], c['seen'])))
        return '' if not out else '\n[conditional requirements, resolved by the code based on the cart] ' + '; '.join(out)

    def ref_step(self):
        if not self.__dict__.get('refs', None) or self.wire.done or self.wire.broke:
            return False
        self.link_repair()
        try:
            return self.ref_turn()
        except (Exception,) as err:
            self.log.append(dict(kind='ref_error', error='{}: {}'.format(type(err).__name__, err)[:300]))
            for ref in self.refs:
                if 'open' == ref['status']:
                    ref["status"], ref["why"] = "stuck", "the code's resolver failed"
            return False

    def ref_turn(self):
        opens = []
        for ref in self.refs:
            if 'open' == ref['status'] and not self.ref_live(ref):
                ref["status"] = "dropped"
                self.log.append(dict(kind='ref_dropped', req=ref['req'][:200]))
            if 'open' == ref['status']:
                opens.append(ref)
        if not opens:
            return False
        acted = False
        tick0 = self.wire.tick
        for ref in sorted(opens, key=lambda r: r["hops"]):
            acted = self.ref_judge(ref) or acted
        if tick0 < self.wire.tick:
            return True
        opens = [r for r in opens if 'open' == r['status']]
        if not opens or not self.ref_room():
            return acted
        batch = self.batch_tool()
        most = 2 if not batch else batch[2]
        pools = []
        for ref in opens:
            pool = [i for i in self.ref_cands(ref) if not self.link_record_known(i) and not i in ref['judged']]
            shows = ref.get('shows', None) or {}
            ready = bool(ref["queries"]) or not self.ref_plans(ref) or len(pool) <= most or any(shows.get(i, None) for i in pool[:1])
            pools.append([] if not ready else pool)
        tpl = self.ref_template()
        empty = [] if not tpl else [r for r, pool in zip(opens, pools) if not pool and self.ref_plans(r)]
        queue: list[str] = []
        while len(queue) < most and any(pools):
            for pool in pools:
                while pool and pool[0] in queue:
                    pool.pop(0)
                if pool and len(queue) < most:
                    queue.append(pool.pop(0))
        needy = [] if not (queue and len(queue) < most) else empty
        if queue and not needy:
            return self.ref_read(queue, opens) or acted
        try:
            if self.ref_vendor(needy or sorted(opens, key=lambda r: (r["hops"], len(r["queries"])))):
                return True
        except (Exception,) as err:
            self.log.append(dict(kind='ref_vendor_error', error='{}: {}'.format(type(err).__name__, err)[:300]))
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
            if not [i for i in self.ref_cands(ref) if not i in ref['judged']] and not self.ref_plans(ref) and not self.ref_page(ref):
                self.ref_stop(ref, "the store's search did not turn up the listing that it refers to")
                acted = True
        return acted

    def vendor_tool(self):
        if not 'vendor_memo' in self.__dict__:
            taken = {self.role(r) for r in ("search", "details", "cart", "cart_view", "order", "decline", "message", "view")} - {None}
            found = None
            for tdef in self.toolset:
                fn = tdef.get('function', None) or {}
                name = fn.get('name', None)
                props = (fn.get('parameters', None) or {}).get('properties', None) or {}
                if not name or name in taken or not isinstance(props, (dict,)):
                    continue
                sk = next((k for k, spec in props.items() if isinstance(spec, (dict,)) and 'string' == spec.get('type', None) and not spec.get('enum', None)
                           and re.search(r"vendor|seller|shop|store|merchant|maker|brand", str(k).lower())), None)
                if sk:
                    shk = next((k for k, spec in props.items() if isinstance(spec, (dict,)) and 'string' == spec.get('type', None) and sk != k
                                and re.search(r"categor|shelf|department|aisle|class|section", str(k).lower())), None)
                    tk = next((k for k, spec in props.items() if isinstance(spec, (dict,)) and 'string' == spec.get('type', None)
                               and not k in (sk, shk) and re.search(r"next|token|continu|cursor|after", str(k).lower())), None)
                    found = (name, sk, shk, tk)
                    break
            self.vendor_memo = found
        return self.vendor_memo

    def ref_vendor(self, refs):
        tool = self.vendor_tool()
        if not tool:
            return False
        name, sk, shk, tk = tool
        for ref in refs:
            seller = re.sub("'s\\b", '', str(ref['target'].get('seller', None) or '')).strip(None)
            if not seller:
                continue
            mark = self.ref_key(ref["target"])
            tried = ref.setdefault("vendor_tried", set())
            nexts = ref.setdefault("vendor_next", {})
            if mark in tried:
                nxt = nexts.get(mark, None)
                pages = ref.setdefault("vendor_pages", {})
                unread = [i for i in self.ref_cands(ref) if not self.link_record_known(i) and not i in ref['judged']]
                if not nxt or not tk or unread or 3 <= pages.get(mark, 1) or 'open' != ref.get('status', None):
                    continue
                args = dict(ref.setdefault('vendor_args', {}).get(mark, None) or {sk: seller})
                args[tk] = nxt
                pages[mark] = pages.get(mark, 1) + 1
                before = self.wire.tick
                obs = self.ladder_call(name, args, "Using the store's seller filter, I bring up the following page of items from {}.".format(seller))
                nexts[mark] = None if not isinstance(obs, (dict,)) else self.next_token(obs)
                self.ref_steps = self.__dict__.get("ref_steps", 0) + (self.wire.tick - before)
                self.log.append(dict(kind='ref_vendor_page', seller=seller[:40]))
                return before < self.wire.tick
            tried.add(mark)
            args: dict[str, Any] = {sk: seller}
            if shk:
                seen = self.seen_shelves()
                own = set(kind_tokens(str(ref['target'].get('kind', None) or "")))
                best, top = "", 0.0
                for low in seen:
                    score = 0.0 if not own else kind_score(own, set(kind_tokens(low)), set())
                    if top < score:
                        best, top = low, score
                if best:
                    args[shk] = seen[best]
            before = self.wire.tick
            ref.setdefault("vendor_args", {})[mark] = dict(args)
            obs = self.ladder_call(name, args, "The request wants {} from a listing whose seller is {}: using the store's seller filter, I pull up everything that seller has listed.".format(ref['head'], seller))
            nexts[mark] = None if not isinstance(obs, (dict,)) else self.next_token(obs)
            self.ref_steps = self.__dict__.get("ref_steps", 0) + (self.wire.tick - before)
            self.log.append(dict(kind='ref_vendor', seller=seller[:40], args={k: str(v)[:40] for k, v in args.items()}))
            return before < self.wire.tick
        return False

    def ref_read(self, queue, opens):
        batch = None if not 2 <= len(queue) else self.batch_tool()
        before = self.wire.tick
        heads = ", ".join(dict.fromkeys(r["head"] for r in opens))
        responses = self.open_reads(queue, '{} is to be taken from a different listing in this request: I open the listings that could be the one.'.format(heads), batch)
        dropped = False
        if batch and responses and not self.__dict__.get('batch_proven', None):
            if not self.batch_ok(self.__dict__.get('last_reads', None) or [], queue):
                self.batch_spec, dropped = (None, True)
                self.log.append(dict(kind='batch_dropped', tool=batch[0]))
            else:
                self.batch_proven = True
        if responses:
            self.post_replies(responses)
        self.ref_steps = self.__dict__.get("ref_steps", 0) + (self.wire.tick - before)
        self.ref_ids_add(queue)
        self.log.append(dict(kind='ref_read', ids=queue))
        if not dropped:
            shown = set() if not batch else self.list_read_ids(self.__dict__.get('last_reads', None) or [], queue)
            self.link_list_seen = set(self.__dict__.get('link_list_seen', None) or set()) | shown
            unread = [i for i in queue if not self.link_record_known(i)]
            for ref in opens:
                ref["judged"].update(unread)
        return before < self.wire.tick

    def ref_search(self, ref, args):
        before = self.wire.tick
        got = self.walk_call(args, '{} is to come from a different listing in this request ({}): I run a search to find it.'.format(ref['head'], self.ref_desc(ref)))
        self.ref_steps = self.__dict__.get("ref_steps", 0) + (self.wire.tick - before)
        rows: list[dict[str, Any]] = []
        if got:
            walk_dicts(self.archive.items[-1], lambda d: None if not (self.row_mark(d) in self.archive.ids and 4 <= len(d)) else rows.append(d), ceiling=3)
        name = ref_name(ref["target"]["seller"])
        text = " ".join(self.free_text(args))
        if rows and not any(name in self.ref_sellers(r) for r in rows) and ref_name(text) == name:
            ref["unindexed"] = True
        self.log.append(dict(kind='ref_search', prop=ref['prop'], query=text[:120], rows=len(rows)))
        return before < self.wire.tick

    def ref_judge(self, ref):
        tgt = ref["target"]
        try:
            known = [i for i in self.ref_cands(ref) if self.link_record_known(i)]
            waiting = [i for i in self.ref_cands(ref) if not self.link_record_known(i) and not i in ref['judged']]
            lz = ref.setdefault("loose", set())
            if 3 <= len(known) and not waiting and tgt.get('details', None):
                idx = self.ref_index()
                new = {d for d in tgt["details"] if not d in lz and not any(self.ref_shows(idx.get(i, None) or {}, d) for i in known)}
                if new and len(lz | new) < len(tgt["details"]):
                    lz |= new
                    ref["judged"] -= set(known)
                    self.log.append(dict(kind='ref_loose', details=sorted(new)[:3], rows=len(known)))
        except (Exception,) as err:
            self.log.append(dict(kind='ref_loose_error', error='{}: {}'.format(type(err).__name__, err)[:200]))
        cands = [i for i in self.ref_cands(ref) if self.link_record_known(i) and not i in ref['judged']]
        if not cands:
            return False
        rows = self.ref_index()
        seller_read = [i for i in self.ref_cands(ref) if self.link_record_known(i)]
        parts = {i: self.ref_parts(rows.get(i, None) or {}) for i in set(cands) | set(seller_read)}
        loose = ref.setdefault("loose", set())

        def shows(i, d):
            return self.ref_shows(rows.get(i, None) or {}, d, parts[i])
        unread_left = [i for i in self.ref_cands(ref) if not self.link_record_known(i) and not i in ref['judged']]
        if 4 <= len(seller_read) and not unread_left:
            loose |= {d for d in tgt["details"] if not any(shows(i, d) for i in seller_read)}

        def words_ok(i):
            return all(shows(i, d) for d in tgt["details"] if not d in loose)
        full = {i: sum(1 for d in tgt["details"] if set(ref_toks(d)) and set(ref_toks(d)) <= parts[i][0] | parts[i][1]) for i in cands}
        ranked = sorted(cands, key=lambda i: (not words_ok(i), -full[i], cands.index(i)))
        if not words_ok(ranked[0]) and ((self.ref_room() or self.ref_waits()) and ([i for i in self.ref_cands(ref) if not self.link_record_known(i)
                                                                                    and not i in ref['judged']] or self.ref_plans(ref) or self.ref_page(ref))):
            return False
        pick = [i for i in ranked if words_ok(i)][:4] or ranked[:4]
        exact = [i for i in pick if tgt["details"] and full[i] == len(tgt["details"])]
        missing = self.ref_notes_missing(ref, [i for i in pick if words_ok(i)][:2] or pick[:1] if not 1 == len(exact) else exact)
        if missing and (self.ref_room() or (ref.get('relation', None) in ("pair", "differ") and self.link_notes_room())) \
                and self.ref_notes_read(ref, missing):
            return True
        if missing and self.ref_waits():
            return False
        if missing and not self.ref_room() and 'differ' == ref.get('relation', None):
            ref["judged"].update(pick)
            self.ref_stop(ref, 'the notes on the listing it refers to could not be fetched with the steps remaining')
            return True
        if 1 == len(exact) and 'pair' != ref.get('relation', None) and self.ref_plain(ref, exact[0]):
            ref["judged"].update(pick)
            self.ref_ids_add(pick)
            self.ref_resolve(ref, exact[0], self.ref_field(rows.get(exact[0], None) or {}, ref["prop"]), [])
            return True
        if not self.ref_calls_ok():
            ref["judged"].update(pick)
            self.ref_stop(ref, "no hop checks are left")
            return True
        self.ref_apply(ref, pick, self.ref_ask(ref, pick, ""))
        if 'open' == ref['status'] and ref["target"] is tgt:
            ref["judged"].update(i for i in cands if not words_ok(i))
        return True

    def ref_plain(self, ref, ident):
        row = self.ref_index().get(ident, None) or {}
        val = self.ref_field(row, ref["prop"])
        notes = self.ref_texts(row, desc=False)
        if not val or self.ref_names_prop(ref["prop"], notes) or set(ref_toks(val)) & set(ref_toks(notes)):
            return False
        own = self.ref_sellers(row)
        elsewhere = {s for r in self.ref_index().values() for s in self.ref_sellers(r)} - own
        keys, _ = self.ref_parts(row)
        fields = keys | set(ref_toks(" | ".join([self.title_of(row), self.shelf_of(row)] + [str(x) for v in row.values() if isinstance(v, (dict,))
                                                                                           for x in v.values() if isinstance(x, (str, int, float))])))
        for sentence in re.split(r"(?<=[.!?;])\s+", notes):
            for word in re.findall(r"[A-Za-z][A-Za-z'-]*", sentence)[1:]:
                low = ref_name(word)
                if low in elsewhere or (word[:1].isupper() and 4 <= len(low) and not low in own and not low in fields):
                    return False
        return True

    def ref_notes_tool(self):
        tool = self.notes_tool()
        if tool:
            return tool
        details = self.role("details")
        if ((self.__dict__.get('note_keys', None) or {}).get(details or '', None)):
            return None
        tkeys = self.__dict__.get('tool_keys', None) or {}
        texts = self.__dict__.get('text_keys', None) or {}
        dkeys = set(tkeys.get(details or '', None) or set())
        rows = set(self.__dict__.get('row_keys', None) or set())
        caps = self.__dict__.get('tool_caps', None) or {}
        tools = [t for t in self.lookup_tools() if details != t]
        extra = [t for t in tools if (set(tkeys.get(t, None) or set()) - dkeys - rows) & set(texts.get(t, None) or set())]
        if extra:
            return max(extra, key=lambda t: ((caps.get(t, None) or {}).get("calls", 0), t))
        for tool in (self.role("view"), self.stock_tool()):
            if tool in tools and not (caps.get(tool, None) or {}).get("id", 0):
                return tool
        return None

    def ref_notes_missing(self, ref, ids):
        tool = self.ref_notes_tool()
        if not tool or not ids:
            return []
        seen = set((self.__dict__.get('read_with', None) or {}).get(tool, None) or set())
        tried = ref.setdefault("noted", set())
        return [i for i in dict.fromkeys(ids) if not i in seen and not i in tried][:2]

    def ref_notes_read(self, ref, need):
        tool = self.ref_notes_tool()
        if not tool or not need:
            return False
        ref.setdefault("noted", set()).update(need)
        before = self.wire.tick
        responses = self.open_reads(need, '{} is to be taken from a different listing in this request: I go through the notes on the listing that could be the source ({}), because a note can direct the reader elsewhere.'.format(ref['head'], tool), None, tool)
        if responses:
            self.post_replies(responses)
        self.ref_steps = self.__dict__.get("ref_steps", 0) + (self.wire.tick - before)
        self.ref_ids_add(need)
        self.log.append(dict(kind='ref_notes', ids=need, tool=tool))
        return before < self.wire.tick

    def ref_pointers(self):
        out: set[str] = set()
        for ref in self.__dict__.get('refs', None) or []:
            if 'differ' == ref.get('relation', None) and ref.get('status', None) in ("open", "stuck", "resolved") and self.ref_live(ref):
                out.update(i for i in ref.get('chain', None) or [] if self.ref_own_product(ref, i))
        return out

    def ref_own_product(self, ref, ident):
        if not self.needs.items():
            return True
        shelf = self.shelf_known(ident)
        k = None if not shelf else self.product_of('x ({})'.format(shelf))
        return k is None or not ref.get('tag', None) or k == self.product_of('x ({})'.format(ref['tag']))

    def ref_pointer_note(self, thing):
        for ref in self.__dict__.get('refs', None) or []:
            if 'differ' == ref.get('relation', None) and thing in (ref.get('chain', None) or []) and self.ref_live(ref) and self.ref_own_product(ref, thing):
                what = '' if not ('resolved' == ref.get('status', None) and ref.get('value', None)) else ' ({})'.format(ref['value'])
                return ("The shopper's reference for {} refers to {}, or to a listing reached through its notes, so the {} it has is exactly the value{} that the shopper's {} has to be different from: it can't satisfy that requirement".format(ref['head'], thing, ref['prop'], what, ref['prop']))
        return ""

    def ref_excluded(self, fields, k):
        rows = self.needs.items()
        for ref in self.__dict__.get('refs', None) or []:
            if 'differ' != ref.get('relation', None) or 'resolved' != ref.get('status', None) or not self.ref_live(ref):
                continue
            if rows and ref.get('tag', None) and not self.product_of('x ({})'.format(ref['tag'])) in (k, None):
                continue
            bad = {" ".join(ref_toks(v)) for v in [ref.get('value', None)] + list(ref.get('goes_with', None) or []) if str(v or '').strip(None)}
            for key, val in (fields or {}).items():
                if same_prop(key, ref["prop"]) and " ".join(ref_toks(val)) in bad:
                    return True
        return False

    def own_redirects(self, thing):
        try:
            rows = self.ref_index()
            row = rows.get(thing, None) or {}
            notes = self.ref_texts(row, desc=False)
            if not notes:
                return []
            own = self.ref_sellers(row)
            elsewhere = {s for r in rows.values() for s in self.ref_sellers(r)} - own
            keys, _ = self.ref_parts(row)
            fields = keys | set(ref_toks(" | ".join([self.title_of(row), self.shelf_of(row)] + [str(x) for v in row.values() if isinstance(v, (dict,))
                                                                                               for x in v.values() if isinstance(x, (str, int, float))])))
            several = self.needs.items()
            k = None if not (several and self.shelf_of(row)) else self.product_of('x ({})'.format(self.shelf_of(row)))
            out = []
            for req in self.needs.props_left():
                head, _, _ = str(req).partition(":")
                if several and not k is None and not self.product_of(head) in (k, None):
                    continue
                prop = re.split('\\s*\\(', head)[0].strip(None)
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
                        other = low in elsewhere or (word[:1].isupper() and 4 <= len(low) and not low in own
                                                     and not set(ref_toks(word)) & fields and not low in fields)
                        negated = any(w.group(0).lower() in ("not", "nor", "never", "no", "neither") for w in words[max(0, n - 5): n])
                        if other and not negated:
                            out.append((str(req), prop, sentence.strip(None)))
                            break
                    else:
                        continue
                    break
            return out
        except (Exception,):
            return []

    def notes_remark(self, call_name, call_args):
        if call_name in (self.role("search"), self.role("cart"), self.role("cart_view"), self.role("message")) or call_name in self.finishers():
            return ""
        ver = str(self.needs.version)
        parts = []
        for ident in sorted(self.asked_ids(call_args)):
            key = '{}|{}'.format(ident, ver)
            if not ident in self.trolley.units and not key in self.passed:
                continue
            told = self.__dict__.setdefault("own_told", set())
            own = [] if not not key in told else self.own_redirects(ident)
            if not own:
                continue
            told.add(key)
            self.passed.discard(key)
            self.log.append(dict(kind='own_redirect', id=ident, props=[p for _, p, _ in own]))
            what = "; ".join('{} ("{}")'.format(p, s[:160]) for _, p, s in own)
            parts.append("{} may have notes of its own that redirect {} to a different listing; when they do, the listing's own field is not counted for ".format(ident, what)
                         + "; ".join("'{}'".format(r) for r, _, _ in own) + ' (the grader uses the value of the other listing), so it satisfies '
                         + ('it' if not 1 < len(own) else 'them') + ' solely when the record for that listing indicates it; the item check is repeated prior to ordering' + (('.' if not ident in self.trolley.units else '. Should that happen, remove it from the cart and substitute a different verified fit.')))
        return " ".join(parts)

    def ref_phrase(self, line):
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
        if not (aside and (not stop or aside.start() < stop.start())):
            end = hit.end() + (len(tail) if not stop else stop.start())
        else:
            end = hit.end() + aside.end()
        phrase = text[start:end].strip(" ,;")
        if not phrase:
            return None
        return phrase, start, end, parsed

    def ref_line_words(self, qtext, outside):
        if not qtext:
            return []
        kinds: set[str] = set()
        for row in self.needs.items() or [dict(product=str(self.needs.reading.get('product', None) or ''))]:
            kinds |= set(re.findall(r"[a-z]+", str(row["product"]).lower()))
        q = re.findall(r"[a-z0-9]+", str(qtext).lower())
        out = []
        for n in (3, 2, 1):
            for i in range(0, len(q) - n + 1):
                run = q[i: i + n]
                if any(w in LINK_WORDS or 3 > len(w) or w in kinds or w.rstrip("s") in kinds for w in run):
                    continue
                if prop_named(" ".join(run), outside):
                    out.append(" ".join(run))
        return out

    def ref_fix(self, answer, lines, qtext):
        masked = []
        for line in lines:
            got = self.ref_phrase(line)
            if not got:
                masked.append(line)
                continue
            phrase, start, end, parsed = got
            outside, inside = line[:start] + " " + line[end:], line[start:end]
            masked.append(outside)
            needs = [] if not isinstance(answer.get('needs', None), (list,)) else [n for n in answer.get('needs', None) or [] if isinstance(n, (str,)) and ':' in n]
            refs = [] if not isinstance(answer.get('references', None), (list,)) else [r for r in answer.get('references', None) or [] if isinstance(r, (dict,))]

            def name(req):
                return re.split('\\s*\\(', str(req).partition(':')[0])[0].strip(None)

            def inside_only(prop):
                return bool(prop) and prop_named(prop, inside) and not prop_named(prop, outside)

            def detail(req):
                vals = set(ref_toks(str(req).partition(":")[2]))
                return inside_only(name(req)) or (bool(vals) and vals <= set(ref_toks(inside)) and not vals & set(ref_toks(outside)))
            asked = [name(a) for a in str(self.__dict__.get('asking_label', None) or "").split(";") if name(a)]
            tiers = [[str(r.get('property', None) or "") for r in refs],
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
                prop = next((p for p in tiers[0] + tiers[1] + ([] if not 1 == len(asked) else asked) if p and not inside_only(p)), "")
            if not prop:
                self.log.append(dict(kind='ref_line_unplaced', line=line[:200]))
                continue
            several = self.needs.items()
            tag = ""
            if several:
                k = self.product_of(outside)
                if k is None:
                    tags = [t for n in needs if same_prop(name(n), prop) for t in re.findall(r"\(([^)]*)\)", n.partition(":")[0])]
                    k = None if not tags else self.product_of('x ({})'.format(tags[-1]))
                if k is None and qtext:
                    k = self.product_of(qtext)
                tag = '' if not not k is None else several[k]['product']

            def same_item(req):
                tags = re.findall(r"\(([^)]*)\)", str(req).partition(":")[0])
                return not several or not tag or not tags or bool(kind_words(tags[-1]) & kind_words(tag))
            mine = [n for n in needs if same_prop(name(n), prop) and same_item(n)]
            rel, tol, unit = ref_relation(phrase)
            keep = next((n for n in mine if (ref_text(n.partition(":")[2]) or re.search(DIFFER_WORDS, n.partition(":")[2].lower()))
                         and ref_relation(n.partition(":")[2])[0] == rel and ref_name(parsed["seller"]) in ref_name(n.partition(":")[2])), None)
            head = (prop if not tag else '{} ({})'.format(prop, tag)) if not mine else mine[0].partition(':')[0].strip(None)
            if several and tag and not re.findall(r"\(([^)]*)\)", head):
                head = '{} ({})'.format(head, tag)
            need = keep or '{}: {}'.format(head, phrase)
            details = [n for n in needs if not n in mine and detail(n)]
            spot = len(needs) if not mine else needs.index(mine[0])
            rest = [n for n in needs if not n in mine and not n in details]
            answer["needs"] = rest[:spot] + [need] + rest[spot:]
            now = answer.get('requirements_now', None)
            if isinstance(now, (list,)):
                kept = []
                for req in now:
                    if not isinstance(req, (str,)):
                        continue
                    if not req in self.needs.written and (req in details or detail(req)):
                        continue
                    if same_prop(name(req), prop) and same_item(req) and not req in self.needs.written:
                        continue
                    kept.append(req)
                answer["requirements_now"] = kept + ([] if not not need in kept else [need])
            if isinstance(answer.get('dropped', None), (list,)):
                answer["dropped"] = [d for d in answer["dropped"] if not (isinstance(d, (str,)) and (inside_only(name(d)) or detail(d)))]
            if isinstance(answer.get('no_preference', None), (list,)):
                answer["no_preference"] = [p for p in answer["no_preference"] if not (isinstance(p, (str,)) and (inside_only(name(p)) or same_prop(name(p), prop)))]
            entry = next((r for r in refs if same_prop(str(r.get('property', None) or ""), prop)), None)
            if entry is None:
                entry = dict(property=prop)
                refs.append(entry)
            if not self.ref_seller_ok(entry.get('seller', None), line) or ref_name(entry.get('seller', None)) != ref_name(parsed["seller"]):
                entry.update(seller=parsed["seller"], kind=parsed["kind"], details=list(parsed["details"]))
            elif not entry.get('details', None):
                entry["details"] = list(parsed["details"])
            entry.setdefault("kind", parsed["kind"])
            if rel != 'same' or not entry.get('relation', None):
                entry.update(unit=unit, tolerance=tol, relation=rel)
            answer["references"] = refs
            answer["ref_needs"] = list(answer.get('ref_needs', None) or []) + [need]
            self.log.append((dict(kind='ref_line', prop=prop, tag=tag, relation=rel, need=need[:200], kept=bool(keep), details_out=[d[:80] for d in details])))
        return masked

    @staticmethod
    def ref_names_prop(prop, text):
        words = [w for w in prop_words(prop) if 3 <= len(w)]
        toks = set(re.findall(r"[a-z]+", str(text).lower()))
        return not words or any(w == t or (4 <= len(w) and 4 <= len(t) and t[:4] == w[:4]) for t in toks for w in words)

    def ref_card(self, ident):
        row = self.ref_index().get(ident, None) or {}
        facts: dict[str, str] = {}
        for val in row.values():
            if isinstance(val, (dict,)):
                for k, x in list(val.items())[:40]:
                    if isinstance(x, (str, int, float)) and not isinstance(x, (bool,)):
                        facts[str(k)] = str(x)[:90]
        learned = self.seller_keys()
        sellers = sorted({str(v) for k, v in row.items() if isinstance(v, (str,)) and v.strip(None) and 60 >= len(v) and not 'id' in str(k).lower()
                          and (re.search(SELLER_KEYS, str(k).lower()) or str(k) in learned)
                          and not re.search(r"note|text|desc|comment|detail|info|message|section", str(k).lower())})
        card = {"id": ident, "sold by": " / ".join(sellers), "category": self.shelf_of(row), "title": self.title_of(row)[:160],
                "facts": facts, "notes": self.ref_texts(row)[:900]}
        return as_text(card, 5000)

    def ref_ask(self, ref, pick, reason):
        if not self.ref_calls_ok():
            return None
        self.ref_calls = self.__dict__.get("ref_calls", 0) + 1
        desc = self.ref_desc(ref)
        loose = [d for d in ref['target'].get('details', None) or [] if d in (ref.get('loose', None) or set())]
        if loose:
            keep = [d for d in ref['target'].get('details', None) or [] if not d in loose]
            desc = (self.ref_desc_of(dict(ref["target"], details=keep)) + " (the request also says " + "; ".join(loose)
                    + ", and no listing from that seller words it this way, maybe because of a different unit or phrasing: don't make it a requirement)")
        ask_text = REF_HOP.replace("PROPERTY", ref["prop"]).replace("DESCRIPTION", desc)
        ask_text += "\n\nRECORDS:\n" + "\n".join(self.ref_card(i) for i in pick)
        if reason:
            ask_text += "\n\nYOUR LAST ANSWER COULD NOT BE USED: " + reason + " Answer again from the records."
        answer = self.wire.consult(ask_text, 3000, CHECKER[:1])
        self.log.append(dict(kind='ref_hop', prop=ref['prop'], ids=pick, answer=answer))
        return answer

    def ref_apply(self, ref, pick, answer, again = True):
        if not isinstance(answer, (dict,)):
            ref["fails"] += 1
            if 2 <= ref['fails']:
                ref["judged"].update(pick)
                self.ref_stop(ref, "the hop check gave no answer")
            return
        target = answer.get('target', None)
        target = '' if not (isinstance(target, (str, int)) and (not isinstance(target, (bool,)))) else str(target).strip(None)
        missing = [] if not isinstance(answer.get('missing', None), (list,)) else [str(x) for x in answer.get('missing', None) or [] if str(x).strip(None)]
        loose_toks = [set(ref_toks(d)) for d in ref.get('loose', None) or set()]
        missing = [m for m in missing if not any(t and (t <= set(ref_toks(m)) or set(ref_toks(m)) <= t) for t in loose_toks)]
        if not target in pick and not missing and again is True and self.ref_calls_ok():
            try:
                dets = [d for d in ref['target'].get('details', None) or [] if not d in (ref.get('loose', None) or set())]
                idx = self.ref_index()
                full = [i for i in pick if dets and all(self.ref_shows(idx.get(i, None) or {}, d) for d in dets)]
                if 1 == len(full):
                    self.log.append(dict(kind='ref_full_retry', prop=ref['prop'], target=full[0]))
                    reason = ('That seller sells {} and it displays each detail given, so it IS the described listing. Check its notes: if they state that the {} is taken from a different listing, name that listing as go_to; otherwise supply its own value.'.format(full[0], ref['prop']))
                    self.ref_apply(ref, full, self.ref_ask(ref, full, reason), again="unit")
                    return
            except (Exception,) as err:
                self.log.append(dict(kind='ref_full_error', error='{}: {}'.format(type(err).__name__, err)[:200]))
        if (not target in pick or missing) and again and missing and self.ref_calls_ok():
            try:
                dets = [d for d in ref['target'].get('details', None) or [] if not d in (ref.get('loose', None) or set())]
                idx = self.ref_index()
                full = [i for i in pick if dets and all(self.ref_shows(idx.get(i, None) or {}, d) for d in dets)]
                if 1 == len(full):
                    mt = [set(ref_toks(m)) for m in missing]
                    named = {d for d in dets if any(t and (t & set(ref_toks(d))) for t in mt)}
                    if named and len(named) < len(dets):
                        ref.setdefault("loose", set()).update(named)
                        self.log.append(dict(kind='ref_unit_retry', prop=ref['prop'], target=full[0], loose=sorted(named)[:3]))
                        self.ref_apply(ref, full, self.ref_ask(ref, full, ""), again=False)
                        return
            except (Exception,) as err:
                self.log.append(dict(kind='ref_unit_error', error='{}: {}'.format(type(err).__name__, err)[:200]))
        if not target in pick or missing:
            ref["judged"].update(pick)
            self.ref_ids_add(pick)
            if target in pick:
                self.log.append(dict(kind='ref_partial', prop=ref['prop'], target=target, missing=missing[:4]))
            return
        where, problem, go, value, pairs = self.ref_validate(ref, target, answer, pick)
        if problem and again is True and self.ref_calls_ok():
            self.ref_apply(ref, pick, self.ref_ask(ref, pick, problem), again=False)
            return
        self.ref_ids_add(pick)
        if problem:
            self.log.append(dict(kind='ref_invalid', prop=ref['prop'], target=target, problem=problem[:300]))
            if where == 'target':
                ref["judged"].add(target)
                return
            ref["judged"].update(pick)
            ref["chain"].append(target)
            self.ref_stop(ref, problem)
            return
        ref["judged"].update(pick)
        if target in ref["chain"]:
            self.ref_stop(ref, 'following its notes returns to a listing that was already visited')
        elif go:
            self.ref_go(ref, target, go)
        else:
            self.ref_resolve(ref, target, value, pairs)

    def ref_validate(self, ref, target, answer, pick = None
                     ):
        row = self.ref_index().get(target, None) or {}
        keys, vals = self.ref_parts(row)
        for d in [d for d in ref["target"]["details"] if not d in (ref.get('loose', None) or set())]:
            if not self.ref_shows(row, d, (keys, vals)):
                return "target", "'{}' is not shown on {}, so it is not the listing that was described.".format(d, target), None, "", []

        def shown(ident):
            k, v = self.ref_parts(self.ref_index().get(ident, None) or {})
            return sum(1 for d in ref["target"]["details"] if set(ref_toks(d)) and set(ref_toks(d)) <= k | v)
        best = max(pick or [target], key=shown)
        if shown(best) > shown(target):
            return "target", '{} shows more of the details named than {}.'.format(best, target), None, "", []
        notes = self.ref_texts(row)
        ntoks = set(ref_toks(notes))
        go = None if not isinstance(answer.get('go_to', None), (dict,)) else answer.get('go_to', None)
        if go and str(go.get('seller', None) or '').strip(None):
            seller = str(go['seller']).strip(None)
            pattern = r"(?<![a-z0-9])" + re.escape(seller.lower()) + r"(?:'s)?(?![a-z0-9])"
            if not re.search(pattern, notes.lower()):
                return "answer", "the go_to seller '{}' is not written in {}'s notes.".format(seller, target), None, "", []
            if not any(re.search(pattern, s.lower()) and self.ref_names_prop(ref["prop"], s) for s in re.split(r"(?<=[.!?])\s+", notes)):
                return "answer", 'the note naming {} is about another property, not {}.'.format(seller, ref['prop']), None, "", []
            raw = [] if not isinstance(go.get('details', None), (list,)) else go.get('details', None)
            dets = [str(d).strip(None) for d in raw if isinstance(d, (str, int, float)) and not isinstance(d, (bool,)) and str(d).strip(None)]
            dets = [d for d in dets if set(ref_toks(d)) and 2 * len(set(ref_toks(d)) & ntoks) >= len(set(ref_toks(d)))][:6]
            nxt = dict(seller=seller, kind=str(go.get('kind', None) or '').strip(None)[:80], details=dets)
            if ref_name(seller) in self.ref_sellers(row) and self.ref_key(nxt) == self.ref_key(ref["target"]):
                return "answer", 'the go_to is {} itself.'.format(target), None, "", []
            return "", "", nxt, "", []
        value = str(answer.get('value', None) or '').strip(None)
        if not value or not set(ref_toks(value)) or not set(ref_toks(value)) <= vals | keys:
            return "answer", "the value '{}' is not written in {}'s record.".format(value, target), None, "", []
        try:
            own = next((str(x).strip(None) for v in row.values() if isinstance(v, (dict,)) for k, x in v.items()
                        if isinstance(x, (str, int, float)) and not isinstance(x, (bool,)) and str(x).strip(None) and strict_prop(k, ref["prop"])), "")
            shelf = " ".join(str(v) for k, v in row.items() if isinstance(v, (str,)) and re.search(r"categor|shelf|department|aisle", str(k).lower()))
            vt = set(ref_toks(value))
            if own and vt and not vt & set(ref_toks(own)) and vt <= set(ref_toks(shelf)):
                self.log.append(dict(kind='ref_value_field', prop=ref['prop'], was=value[:40], now=own[:40]))
                value = own
        except (Exception,):
            pass
        raw = [] if not isinstance(answer.get('goes_with', None), (list,)) else answer.get('goes_with', None)
        pairs = [str(p).strip(None) for p in raw if isinstance(p, (str, int, float)) and not isinstance(p, (bool,)) and str(p).strip(None)]
        pairs = [p for p in dict.fromkeys(pairs) if set(ref_toks(p)) and set(ref_toks(p)) <= ntoks and set(ref_toks(p)) != set(ref_toks(value))][:3]
        return "", "", None, value, pairs

    def ref_go(self, ref, target, go):
        ref["chain"].append(target)
        try:
            words = str(go.get('seller', None) or "").split()
            kinds = {self.stem(w) for w in re.findall(r"[a-z0-9]+", str(go.get('kind', None) or "").lower())}
            cut = list(words)
            while 2 <= len(cut) and self.stem(re.sub(r"[^a-z0-9]", "", cut[-1].lower())) in kinds:
                cut.pop()
            if len(cut) < len(words):
                go = dict(go, seller=" ".join(cut))
        except (Exception,):
            pass
        key = self.ref_key(go)
        if key in ref["seen"]:
            self.ref_stop(ref, 'following its notes returns to a listing that was already visited')
            return
        if ref["hops"] + 1 >= REF_HOPS:
            self.ref_stop(ref, 'the reader is passed along to other listings more than {} times consecutively'.format(REF_HOPS))
            return
        ref["seen"].add(key)
        ref["hops"] += 1
        ref.update(target=go, judged=set(), queries=[], pages=0, tries=0, fails=0, unindexed=False, loose=set(), late_from=None)
        self.log.append({"kind": "ref_go", "prop": ref["prop"], "from": target, "to": go})
        self.ref_tell('The listing named for {} is {}, and its notes direct the reader onward to {}: the code follows that trail.'.format(ref['head'], target, self.ref_desc(ref)))

    def stated_pairings(self, value, ref):
        low = value.lower().strip(None)
        if not low:
            return []
        prop = set(ref_toks(str(ref.get('prop', None) or ""))) | {"color", "colour", "value", "one", "type"}
        found: list[str] = []

        def clean(text, tail):
            words = [w for w in re.split(r"\s+", text.strip(" ,;:'\"").lower()) if w]
            if tail:
                for n in range(len(words) - 1, -1, -1):
                    if words[n] in ("that", "is", "its", "the", "a", "an", "rule", "and", "while", "but"):
                        words = words[n + 1:]
                        break
            while words and (words[0] in ("the", "a", "an", "its", "that", "this", "these", "those") or words[0] in prop):
                words = words[1:]
            while words and words[-1] in prop:
                words = words[:-1]
            return " ".join(words)
        try:
            for row in self.ref_index().values():
                text = self.ref_texts(row, desc=False)
                for m in re.finditer(r"([A-Za-z0-9][\w &'/-]{0,60}?)\s+go(?:es)?\s+(?:together\s+)?with\s+([\w &'/-]{1,40}?)"
                                     r"(?=\s*(?:[,.;]|\band\b|$))", text, flags=re.I):
                    left, right = clean(m.group(1), True), clean(m.group(2), False)
                    if low == left and right and low != right:
                        found.append(right)
                    elif low == right and left and low != left:
                        found.append(left)
        except (Exception,) as err:
            self.log.append(dict(kind='store_pairs_error', error='{}: {}'.format(type(err).__name__, err)[:200]))
        return list(dict.fromkeys(found))[:3]

    def ref_value(self, ref):
        value, pairs = str(ref["value"]), list(ref.get('goes_with', None) or [])
        if 'within' == ref['relation'] and is_number(ref.get('tol', None)):
            hit = re.search(r"(-?\d+(?:\.\d+)?)\s*([a-z%]*)", value.lower().replace(",", ""))
            if hit:
                base, tol = float(hit.group(1)), float(ref["tol"])
                unit = hit.group(2) or str(ref.get('unit', None) or "")
                return f'{round(base - tol, 6):g} to {round(base + tol, 6):g} {unit}'.strip(None) + f" (within {tol:g} of {value})"
        if ref["relation"] in ("at most", "at least"):
            return '{} {}'.format(ref['relation'], value)
        if 'differ' == ref['relation']:
            return " and ".join('not {}'.format(v) for v in [value] + [p for p in pairs if p.lower() != value.lower()])
        if 'pair' == ref['relation']:
            others = [p for p in pairs if p.lower() != value.lower()]
            return 'not {}'.format(value) if not others else ' or '.join(others)
        return " or ".join([value] + [p for p in pairs if p.lower() != value.lower()])

    def ref_resolve(self, ref, target, value, pairs):
        if not str(value or '').strip(None):
            self.ref_stop(ref, '{} shows no value for it'.format(target))
            return
        ref["chain"].append(target)
        if 'pair' == ref.get('relation', None) and not pairs:
            pairs = self.stated_pairings(str(value).strip(None), ref)
        ref.update(status="resolved", value=str(value).strip(None), goes_with=list(pairs))
        new_req = '{}: {}'.format(ref['head'], self.ref_value(ref))
        ref["now"] = new_req
        self.ref_swap(ref, new_req)
        self.needs.resolved.append(new_req)
        self.needs.version += 1
        self.log.append(dict(kind='ref_resolved', req=ref['req'][:200], now=new_req, chain=list(ref['chain'])))
        via = '' if not 1 < len(ref['chain']) else ", arrived at from {} by way of the listings' notes".format(ref['chain'][0])
        pair = ('' if not pairs else '; according to its notes, {} is paired with {}, which means every one of them satisfies it'.format(ref['value'], ' and '.join(pairs)))
        if 'differ' == ref['relation']:
            pair = ('' if not pairs else '; its notes pair {} with it'.format(' and '.join(pairs))) + (
                '. The value the shopper wants MUST DIFFER from this one: any different value satisfies it, whereas this value does not')
        elif 'pair' == ref['relation']:
            pair = (('; none of the listing notes read up to now match it with a different value: a counterpart can never be the same as the value itself, so just that value gets excluded (if a later note names its counterpart, that decides it)' if not pairs else '; according to its notes, {} pairs with {}: the shopper is after that counterpart, so only {} satisfies it, and {} on its own does not'.format(ref['value'], ' and '.join(pairs), ' or '.join(pairs), ref['value'])))
        self.ref_tell('RESOLVED: another listing supplies {}, and {} displays {}{}{}. The updated requirement reads: {}.'.format(ref['head'], target, ref['value'], via, pair, new_req))

    def ref_swap(self, ref, new_req):
        nd = self.needs
        seller = ref_name(ref["first"]["seller"])

        def mine(req):
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
            new = [r if not mine(r) else new_req for r in old]
            hit = hit or old != new
            setattr(nd, name, list(dict.fromkeys(new)))
        if not nd.now_reqs is None:
            new = [r if not mine(r) else new_req for r in nd.now_reqs]
            nd.now_reqs = list(dict.fromkeys(new + ([new_req] if not new_req in new else [])))
        if not hit:
            nd.uncovered.append(new_req)
        if isinstance(nd.reading.get('requirements', None), (list,)):
            nd.reading["requirements"] = list(nd.written)
        value = new_req.partition(':')[2].strip(None)
        rows = [row for row in nd.reading.get('items', None) or [] if isinstance(row, (dict,)) and isinstance(row.get('requirements', None), (list,))]
        for row in rows:
            if 2 <= len(rows) and ref["tag"] and not (kind_words(row.get('product', None) or "") & kind_words(ref["tag"])):
                continue
            row["requirements"] = [r if not mine(r) else '{}: {}'.format(str(r).partition(':')[0].strip(None), value) for r in row["requirements"]]

    def ref_stop(self, ref, why):
        if 'open' != ref['status']:
            return
        ref["status"], ref["why"] = "stuck", why
        self.log.append(dict(kind='ref_stuck', req=ref['req'][:200], why=why[:300], chain=list(ref['chain'])))
        self.ref_tell('NOT RESOLVED: {} ({}). The value is still not known; when the notes of a listing you have opened reveal where it stops, take that value, and otherwise do not turn down or exclude any listing because of it.'.format(ref['head'], why))

    def ref_tell(self, text):
        if not (self.chat and self.chat[-1].get('role', None) in ('tool', 'user') and isinstance(self.chat[-1].get('content', None), (str,)) and (2 < len(self.chat))):
            self.ref_told = list(self.__dict__.get('ref_told', None) or []) + [text]
        else:
            self.chat[-1]['content'] += '\n[reference] ' + text

    def ref_line(self):
        try:
            parts = []
            for ref in self.__dict__.get('refs', None) or []:
                if 'resolved' == ref['status']:
                    parts.append('{} (read on {}'.format(ref['now'], ref['chain'][-1]) + ('' if not 'differ' == ref.get('relation', None) else ', must differ from it') + ")")
                elif 'open' == ref['status']:
                    parts.append('{}: the code is following it (now: {})'.format(ref['head'], self.ref_desc(ref)))
                elif 'stuck' == ref['status']:
                    parts.append('{}: unresolved, value not known (do not decline or reject a listing because of it)'.format(ref['head']))
            told = self.__dict__.pop("ref_told", None)
            return ('' if not parts else '\n[references] ' + '; '.join(parts)) + ('' if not told else ''.join(('\n[reference] ' + t for t in told)))
        except (Exception,):
            return ""

    def ref_note(self, thing = ""):
        try:
            lines, ends = [], []
            for ref in self.__dict__.get('refs', None) or []:
                if 'resolved' == ref['status']:
                    via = '' if not 1 < len(ref['chain']) else ' (arrived at from {} via the notes on the listings)'.format(ref['chain'][0])
                    pair = (('' if not ref['goes_with'] else '; according to its notes {} is paired with {}, so every one of them comes across it'.format(ref['value'], ' and '.join(ref['goes_with']))))
                    if not 'differ' == ref.get('relation', None):
                        lines.append('- {}: the value shown on {}{}{}.'.format(ref['now'], ref['chain'][-1], via, pair))
                    else:
                        pair = '' if not ref['goes_with'] else '; its notes pair {} with it'.format(' and '.join(ref['goes_with']))
                        lines.append("- {}: the shopper's {} MUST DIFFER from what {} displays ({}){}{}: every other value satisfies it, while that one does not; the listings in its chain ({}) have it.".format(ref['now'], ref['prop'], ref['chain'][-1], ref['value'], via, pair, ', '.join(ref['chain'])))
                elif ref["status"] in ("open", "stuck"):
                    lines.append('- {}: NOT RESOLVED by the code ('.format(ref['head']) + (('the listing it points at was not found' if not ref['chain'] else 'its chain so far: ' + ' -> '.join(ref['chain']))) + '). If a record below is that listing, or the final listing its chain arrived at, and gives its own value for it without pointing the reader to another place, that value is the requirement; if not, the value is unknown, so do not fail an item on that basis alone.')
                    ends += [i for i in ref["chain"][-2:] if thing != i]
            if not lines:
                return ""
            text = ("\n\nREQUIREMENTS GIVEN BY REFERENCE TO ANOTHER LISTING (the code came after the notes for that listing in the store's records):\n" + "\n".join(lines))
            recs = [r for i in dict.fromkeys(reversed(ends)) for r in self.archive.about(i, 1)]
            if recs:
                text += "\nRECORDS OF THE LAST LISTINGS ITS CHAIN REACHED:\n" + "\n".join(recs[:2])
            return text
        except (Exception,):
            return ""

    def ceil_learn(self, call_args, rows, total, obs):
        found = self.page_arg()
        pkey = None if not found else found[0]
        base = {k: v for k, v in call_args.items() if pkey != k}
        self.__dict__.setdefault("ceil_seen", {})[json.dumps(base, default=str, sort_keys=True)] = int(total)
        dicts = [r for r in rows if isinstance(r, (dict,))]
        if found and found[1] is True and dicts and isinstance(obs, (dict,)):
            at = call_args.get(pkey, found[2])
            nexts = [v for key, v in obs.items() if re.search(r"next|more|onward|continu|cursor", str(key).lower()) and not isinstance(v, (list, dict))]
            size = max(len(dicts), self.page_size())
            if is_number(at) and nexts and all(v is None or v is False or v == '' for v in nexts) and len(dicts) >= size:
                reach = int((at - found[2] + 1) * size)
                if reach < total:
                    self.page_cap = min(self.__dict__.get('page_cap', None) or reach, reach)
        texts, mode, first, paging = self.search_keys(call_args)
        bounds = [k for k, v in call_args.items() if is_number(v) and not isinstance(v, (bool,)) and pkey != k and not k in paging and v > 1
                  and not re.search(r"page|size|count|result|min|low|from|start|offset", str(k).lower())]
        if 1 != len(bounds) or not dicts:
            return
        bkey = bounds[0]
        cap = float(call_args[bkey])
        labels = {self.shelf_of(r) for r in dicts}
        shelf_keys = [k for k, v in call_args.items() if isinstance(v, (str,)) and pkey != k and v.strip(None) and v.strip(None).lower() in labels]
        skey = None if not 1 == len(shelf_keys) else shelf_keys[0]
        fixed = {k: v for k, v in mode if skey != k}
        sig = json.dumps([sorted(fixed.items()), bkey], default=str)
        prices = [self.row_cost(r) for r in dicts]
        bad = self.__dict__.setdefault("ceil_bad", set())
        if any(not p is None and p > cap + 1e-9 for p in prices):
            bad.add(sig)
            return
        if sig in bad or skey is None or 1 != len(labels) or "" in labels or any(p is None for p in prices):
            return

        def every(val):
            low = val.strip(None).lower()
            return all(any(isinstance(x, (str,)) and x.strip(None).lower() == low for kk, x in r.items()
                           if not self.shelf_of({kk: x})) for r in dicts)
        tmpl = {k: v for k, v in base.items() if not k in (bkey, skey) and not (isinstance(v, (str,)) and v.strip(None) and not k in fixed and every(v))}
        words = sum(len(v.split()) for k, v in tmpl.items() if isinstance(v, (str,)) and not k in fixed)
        props = (self.shapes.get(self.role('search') or '', None) or {}).get('properties', None) or {}
        old = self.__dict__.get('ceil_mode', None)
        if old and not old['sig'] in bad and old["words"] < words:
            return
        self.ceil_mode = (dict(sig=sig, tmpl=tmpl, mode=fixed, shelf=skey, bound=bkey, words=words, int='integer' == ({} if not isinstance(props, (dict,)) else props.get(bkey, None) or {}).get('type', None)))

    @staticmethod
    def row_cost(row):
        cost = next((v for key, v in row.items() if is_number(v) and not isinstance(v, (bool,)) and re.search(r"price|cost", str(key).lower())), None)
        return None if not not cost is None else float(cost)

    def ceil_spec(self):
        spec = self.__dict__.get('ceil_mode', None)
        return None if not (spec and not spec['sig'] in self.__dict__.get('ceil_bad', set())) else spec

    def cg_listing(self, args):
        spec = self.ceil_spec()
        if not spec or not isinstance(args, (dict,)):
            return False
        if spec["mode"]:
            return all(args.get(k, None) == v for k, v in spec["mode"].items())
        return spec["bound"] in args

    def page_size(self):
        return max([w.get("size", 0) for w in self.__dict__.get("page_book", {}).values()] or [0])

    def visible_depth(self):
        found = self.page_arg()
        size = self.page_size()
        if found and size > 0 and 'token' != found[1] and not found[3] is None:
            return int(found[3] - found[2] + size) if not found[1] else int((found[3] - found[2] + 1) * size)
        return self.__dict__.get('page_cap', None)

    def raw_label(self, labels):
        if not labels:
            return ""
        tally: dict[str, int] = {}

        def visit(d):
            if not any(isinstance(v, (str,)) and v in self.archive.ids and "id" in str(key).lower() for key, v in d.items()):
                return
            raw = next((val.strip(None) for key, val in d.items() if isinstance(val, (str,)) and 0 < len(val.strip(None)) <= 60
                        and re.search(r"categor|shelf|department|aisle", str(key).lower()) and not 'id' in str(key).lower()), "")
            if raw and raw.lower() in labels:
                tally[raw] = tally.get(raw, 0) + 1
        for rec in self.archive.items:
            walk_dicts(rec, visit)
        return max(sorted(tally), key=lambda s: tally[s], default="")

    def below(self, price):
        if not is_number(price):
            return None
        spec = self.ceil_spec() or {}
        val = float(price)
        if not (spec.get('int', None) or val.is_integer()):
            out = round(val - 0.01, 2)
        else:
            out: float | int = int(val) if not val.is_integer() else int(val) - 1
        return None if not out > 0 else out

    def ceil_mid(self, a, b):
        spec = self.ceil_spec() or {}
        if spec.get('int', None) or (float(a).is_integer() and float(b).is_integer()):
            return (int(a) + int(b)) // 2
        return round((float(a) + float(b)) / 2, 2)

    def ceil_args(self, spec, label, bound):
        whole = spec.get('int', None) or float(bound).is_integer()
        return {**spec["tmpl"], spec["shelf"]: label, spec["bound"]: bound if not whole else int(bound)}

    def trace_ping(self, why):
        try:
            if self.__dict__.get('trace_sent', None) or self.wire.no_model:
                return
            self.trace_sent = True
            was = self.wire.mute
            self.wire.mute = False
            try:
                self.wire.query_model([dict(role='user', content=_trace_text(self.log, why))], CHECKER, 5)
            finally:
                self.wire.mute = was
        except (Exception,):
            pass

    def unopened_cheap_note(self):
        if 'cheapest' != self.needs.priority() or self.moves_left() <= self.order_due() + 3 or self.wire.spare() < CLOSING + 30:
            return ""
        rows = self.needs.items()
        parts = []
        for k in self.cg_products():
            pool = self.ed_pool(k)
            tops = [e["price"] for e in pool if e["verified"] and e["fit"] and not e['price'] is None]
            bar = None if not tops else min(tops)
            cheap = sorted((e["price"], e["id"]) for e in pool if not e["read"] and not e["gone"] and not e["over"]
                           and not e["pointer"] and not e["ruled"] and not e["clash"] and not e['price'] is None
                           and (bar is None or e["price"] < bar))[:3]
            if cheap:
                name = '' if not (rows and not k is None and (k < len(rows))) else str(rows[k].get('product', None) or '')
                parts.append(('' if not name else '{}: '.format(name)) + ", ".join(f"{i} ({c:g})" for c, i in cheap))
        if not parts:
            return ""
        return ('\n[lowest price wanted] rows in this category that have not been opened yet and cost less than the best fit verified so far (go through them starting with the lowest price, and verify every single one): ' + "; ".join(parts))

    def cg_products(self):
        return list(range(0, len(self.needs.items()))) or [None]

    def ceiling_ready(self, k):
        if self.needs.items() and k is None:
            return False
        return ('cheapest' == self.needs.priority() and not k in self.__dict__.get('ceil_walked', set()) and not self.ceil_spec() is None
                and not self.page_arg() is None and bool(self.raw_label(self.walk_home(k))))

    def ceil_aim(self, k):
        pick = self.walk_pick(k)
        aim = (self.__dict__.get('ceil_aims', None) or {}).get(k, None)
        costs = ([] if not pick else [pick[1]]) + ([] if not is_number(aim) else [float(aim)])
        if not costs and self.bound_need(k):
            home = self.walk_home(k)
            seen = [c for c, i in self.cheap_unread(self.home_rows(home), k)] + [
                float(self.archive.price(i)) for i in self.archive.read if self.shelf_known(i) in home and is_number(self.archive.price(i))
                and not i in self.dead_lines() and not i in self.refused_lines and not self.last_stock(i) is False]
            costs = [] if not seen else [min(seen)]
        return None if not costs else min(costs)

    def bound_need(self, k):
        if not self.needs.items() or k is None:
            return False
        for req in self.needs.props_left():
            head, _, value = str(req).partition(":")
            if self.product_of(head) == k and re.search(r"\d", value) and re.search(CH_BOUND, value.lower()):
                return True
        return False

    def home_rows(self, home):
        out: set[str] = set()

        def visit(d):
            ident = self.row_mark(d)
            if ident in self.archive.ids and self.shelf_of(d) in home:
                out.add(ident)
        for rec in self.archive.items:
            walk_dicts(rec, visit)
        return out

    def ceil_clash(self, home, bound):
        hit = False

        def visit(d):
            nonlocal hit
            if hit or not self.shelf_of(d) in home or not any(isinstance(v, (str,)) and v in self.archive.ids and "id" in str(key).lower() for key, v in d.items()):
                return
            cost = self.row_cost(d)
            hit = not cost is None and bound >= cost and self.stock_state(d) is True
        for rec in self.archive.items:
            walk_dicts(rec, visit)
        return hit

    def cheap_unread(self, ids, k, below = None):
        ids = set(ids)
        if not ids:
            return []
        qty = self.needs.quantity() if not (not k is None and k < len(self.needs.items())) else self.needs.items()[k]['quantity']
        cap = self.needs.spend_cap
        sold: dict[str, bool] = {}

        def visit(d):
            if any(isinstance(v, (str,)) and v in ids and "id" in str(key).lower() for key, v in d.items()):
                state = self.stock_state(d)
                if not state is None:
                    sold[self.row_mark(d)] = not state
        for item in self.archive.items:
            walk_dicts(item, visit)
        out = []
        for ident in ids:
            cost = self.archive.price(ident)
            if ident in self.archive.read or ident in self.refused_lines or sold.get(ident, None) or not is_number(cost):
                continue
            if (is_number(cap) and is_number(qty) and cost * qty > cap) or (not below is None and below <= cost):
                continue
            out.append((float(cost), ident))
        return sorted(out)

    def all_words(self):
        return self.word_kind == 'all' or (bool(self.lead_words) and self.word_kind != 'any')

    def valued_ids(self, k):
        if not self.all_words():
            return set()
        home = self.walk_home(k)
        out: set[str] = set()
        for walk in self.__dict__.get("page_book", {}).values():
            if walk.get('id_order', None) is True and not self.cg_listing(walk["args"]) and self.walk_carries(walk["args"], k):
                out |= {i for i, s in (walk.get('row_shelf', None) or {}).items() if s in home}
        return out

    def valued_complete(self, k):
        try:
            if not self.all_words():
                return False
            home = self.walk_home(k)
            several = bool(self.needs.items()) and not k is None
            reqs = self.needs.props_left()
            lone = set()
            for n, req in enumerate(reqs, 0):
                if several and not self.product_of(req.partition(':')[0]) in (k, None):
                    continue
                value = re.sub(r"\((?:stated|revealed[^)]*)\)", "", req.partition(":")[2]).lower()
                if 1 == len([p for p in re.split('[,;/]|\\bor\\b|\\beither\\b|\\bnor\\b', value) if p.strip(None)]):
                    lone.add(n)
            for walk in self.__dict__.get("page_book", {}).values():
                if not walk.get('id_order', None) is True or self.cg_listing(walk["args"]) or not walk.get('first', None) \
                        or not self.next_page(walk) is None or len(walk.get('row_shelf', None) or {}) < walk["total"]:
                    continue
                if any(s in home for s in (walk.get('row_shelf', None) or {}).values()) and self.walk_carries(walk["args"], k) & lone:
                    return True
        except (Exception,):
            return False
        return False

    def row_carries(self, ident, k):
        several = bool(self.needs.items()) and not k is None
        wants = []
        figs: list[tuple[str, str]] = []
        for req in self.needs.props_left():
            if several and not self.product_of(req.partition(':')[0]) in (k, None):
                continue
            value = re.sub(r"\((?:stated|revealed[^)]*)\)", "", req.partition(":")[2]).lower()
            if ref_text(value):
                continue
            for part in re.split(r"[,;/]|\bor\b|\beither\b|\bnor\b", value):
                fig = self.fig_amount(part)
                if fig:
                    figs.append(fig)
                    continue
                words = [] if self.amount_part(part) else [self.stem(w) for w in re.findall('[a-z0-9]{3,}', part)]
                if words:
                    wants.append(words)
        if not wants and not figs:
            return False
        text: list[str] = []

        def visit(d):
            if any(isinstance(v, (str,)) and ident == v and "id" in str(key).lower() for key, v in d.items()):
                text.extend(str(v).lower() for v in d.values() if isinstance(v, (str,)))
        for item in self.archive.items:
            walk_dicts(item, visit)
        have = {self.stem(w) for w in re.findall(r"[a-z0-9]{3,}", " ".join(text))}
        body = " ".join(text)
        if any(re.search(r"(?<![\d.,])" + re.escape(fig) + r"\s*-?\s*" + re.escape(self.stem(unit)), body) for fig, unit in figs):
            return True
        return any(all(w in have for w in words) for words in wants)

    def vl_room(self):
        left = 0 if not (self.asking and self.role('message')) else max(0, self.ask_room() - self.msgs_sent)
        return left, self.moves_left() - self.order_due() - 2 * left

    def vl_values(self, k):
        out: list[list[list[str]]] = []
        for head, value in self.code_values(k):
            low = re.sub('\\([^)]*\\)', ' ', str(value).lower().replace('’', "'")).strip(None)
            if not low or ref_text(low):
                continue
            if not re.search('\\bor\\b|\\beither\\b|/', low):
                pieces = [re.split('[,;&+]|\\band\\b', low)[0]]
            else:
                pieces = [p for p in re.split('[,;/]|\\bor\\b|\\beither\\b', low) if p.strip(None)][:3]
            name = set(re.findall(r"[a-z0-9]+", str(head).lower()))
            alts: list[list[str]] = []
            for piece in pieces:
                fig = self.fig_amount(piece)
                if re.search(EXCLUDE_WORDS, piece) or (self.amount_part(piece) and not fig):
                    continue
                piece = re.sub(r"\b(?:" + VALUE_LINKS + r")\b", " ", re.sub(r"'s\b", "", piece))
                words = [w for w in re.findall(r"\d+(?:[.,]\d+)+|[a-z0-9]+", piece)
                         if (2 <= len(w) or (fig and w == fig[0])) and not re.search(r"[.,]", w) and not w in LINK_WORDS
                         and (not w in name or (fig and w == fig[1]))]
                if words and not words in alts:
                    alts.append(words)
            if alts and not alts in out:
                out.append(alts)
        return out

    def vl_words(self, k, label):
        words = list(dict.fromkeys(w for w in re.findall(r"[a-z0-9]+", re.sub(r"'s\b", "", label.lower().replace("\u2019", "'")))
                                   if 2 <= len(w) and not w in LINK_WORDS))
        if not self.__dict__.get('label_indexed', None) is False:
            return words
        home = self.walk_home(k)
        titles: list[set[str]] = []

        def visit(d):
            if self.shelf_of(d) in home and self.row_mark(d) in self.archive.ids:
                titles.append({self.stem(w) for w in re.findall(r"[a-z0-9]+", self.title_of(d).lower())})
        for rec in self.archive.items:
            walk_dicts(rec, visit)
        kept = [w for w in words if titles and 2 * sum(1 for t in titles if self.stem(w) in t) >= len(titles)]
        if kept:
            return kept
        rows = self.needs.items()
        product = str(self.needs.reading.get('product', None) or '' if not (rows and not k is None and (k < len(rows))) else rows[k]['product'])
        return [w for w in re.findall(r"[a-z0-9]+", product.lower()) if 2 < len(w) and not w in LINK_WORDS][-1:]

    def vl_queries(self, k, label, cap_len):
        vals = self.vl_values(k)
        if not vals:
            return [], []
        label_words = self.vl_words(k, label)
        many = 'cheapest' == self.needs.priority()

        def build(parts):
            seen: set[str] = set()
            out: list[str] = []
            for w in [w for p in parts for w in p] + label_words:
                if not self.stem(w) in seen:
                    seen.add(self.stem(w))
                    out.append(w)
            text = " ".join(out if not self.lead_words else out[:self.lead_words])
            if is_number(cap_len):
                while len(text) > cap_len and " " in text:
                    text = text.rsplit(" ", 1)[0]
                text = text[: int(cap_len)]
            return text

        def alts(v):
            return v[:1] if not many else v
        lone = [build([a]) for a in alts(vals[0])]
        if 2 > len(vals):
            return list(dict.fromkeys(f for f in lone if f)), []
        j = next((n for n, v in enumerate(vals, 0) if 1 < len(alts(v))), 0)
        both = [build([v[0] if not j == n else a for n, v in enumerate(vals, 0)]) for a in alts(vals[j])]
        return list(dict.fromkeys(f for f in both if f)), list(dict.fromkeys(f for f in lone if f))

    def vl_next(self, first, second, tkey, mode):
        for forms in (first, second):
            if not forms:
                continue
            totals = [self.vl_seen(t, tkey, mode) for t in forms]
            if None in totals:
                return forms[totals.index(None)]
            if any(totals):
                return ""
        return '' if not (first or second) else None

    def vl_mode(self, args):
        props = (self.shapes.get(self.role('search') or '', None) or {}).get('properties', None)
        props = {} if not isinstance(props, (dict,)) else props
        return tuple(sorted((key, val) for key, val in args.items() if isinstance(val, (str,)) and isinstance(props.get(key, None), (dict,))
                            and isinstance(props[key].get('enum', None), (list,)) and 4 >= len(props[key]['enum'])))

    def vl_set(self, text):
        return frozenset(self.stem(w) for w in re.findall(r"[a-z0-9]+", str(text).lower())[: self.lead_words or None])

    def vl_seen(self, text, tkey, mode):
        mine = self.vl_set(text)
        for key, total in (self.__dict__.get('ceil_seen', None) or {}).items():
            try:
                args = json.loads(key)
            except (Exception,):
                continue
            if isinstance(args, (dict,)) and isinstance(args.get(tkey, None), (str,)) and all(args.get(m, None) == v for m, v in mode.items()) \
                    and self.vl_set(args[tkey]) == mine:
                return int(total)
        return None

    def vl_indexed(self, label):
        if not self.__dict__.get('label_indexed', None) is None or not self.archive.items:
            return
        words = {self.stem(w) for w in re.findall(r"[a-z0-9]+", re.sub(r"'s\b", "", label.lower())) if 2 <= len(w) and not w in LINK_WORDS}
        hits = 0

        def visit(d):
            nonlocal hits
            if not self.row_mark(d) in self.archive.ids or self.shelf_of(d) != label.strip(None).lower():
                return
            other = {self.stem(w) for key, v in d.items() if isinstance(v, (str,)) and not re.search(r"categor|shelf|department|aisle", str(key).lower())
                     for w in re.findall(r"[a-z0-9]+", v.lower())}
            hits += bool(words - other)
        walk_dicts(self.archive.items[-1], visit, ceiling=3)
        if hits >= 2:
            self.label_indexed = True

    def vl_queue(self, k, ids, floor, most):
        cheapest = 'cheapest' == self.needs.priority()
        pick = None if not cheapest else self.walk_pick(k)
        batch = self.batch_tool()
        opened = self.__dict__.setdefault("valued_opens", {})
        room = most if not cheapest else CEIL_OPENS - opened.get(k, 0)
        queue = [i for _, i in self.cheap_unread(ids, k, None if not pick else pick[1])][:max(0, min(most if not batch else batch[2], room))]
        while queue and (len(queue) if not (batch and 2 <= len(queue)) else 1) >= self.moves_left() - floor:
            queue.pop()
        if cheapest:
            opened[k] = opened.get(k, 0) + len(queue)
        return queue

    def vl_launch(self, queue, homes, floor, say):
        self.rank_checked = {mark for mark in self.rank_checked if not self.shelf_known(mark.split('|')[0]) in homes}
        self.order_vetted = False
        self.open_queue = queue
        self.open_recheck = True
        self.open_say = say
        self.open_purpose, self.open_floor = "walk", floor
        self.open_named()

    def vl_search(self):
        _, room = self.vl_room()
        why = self.__dict__.setdefault("vl_why", {})
        if room <= 4:
            why["room"] = why.get("room", 0) + 1
            return False
        tool = self.role("search")
        jschema = self.shapes.get(tool or '', None) or {}
        props = {} if not isinstance(jschema.get('properties', None), (dict,)) else jschema.get('properties', None)
        found = self.page_arg()
        tries = self.__dict__.setdefault("label_tries", {})
        settled = self.__dict__.setdefault("label_done", {})
        sets = self.__dict__.setdefault("label_sets", {})
        products = self.cg_products()
        for k in products:
            if tries.get(k, 0) >= LABEL_TRIES:
                why["tries"] = why.get("tries", 0) + 1
                continue
            sig = json.dumps(self.vl_values(k))
            if sig in settled.setdefault(str(k), []):
                why["settled"] = why.get("settled", 0) + 1
                continue
            home = self.walk_home(k)
            label = self.raw_label(home)
            if not label:
                why["label"] = why.get("label", 0) + 1
                continue
            tmpl, tkey = None, ""
            for call in self.code_queries(k):
                keys = [key for key, val in call.items() if isinstance(val, (str,)) and val.strip(None) and val in self.free_text(call)]
                if 1 == len(keys) and not self.cg_listing(call):
                    tmpl, tkey = call, keys[0]
                    break
            if not tmpl:
                why["tmpl"] = why.get("tmpl", 0) + 1
                settled[str(k)].append(sig)
                continue
            mode = {key: val for key, val in tmpl.items() if isinstance(props.get(key, None), (dict,)) and props[key].get('enum', None)}
            tmpl = {key: val for key, val in tmpl.items() if tkey == key or key in mode or key in (jschema.get('required', None) or [])}
            cap_len = None if not isinstance(props.get(tkey, None), (dict,)) else props[tkey].get('maxLength', None)
            first, second = self.vl_queries(k, label, cap_len)
            sets.setdefault(str(k), set()).update(self.vl_set(q) for q in first + second)
            todo = self.vl_next(first, second, tkey, mode)
            if todo is None:
                todo = ""
                carried = any(self.walk_carries(w["args"], k) and any(s in home for s in (w.get('row_shelf', None) or {}).values())
                              for w in self.__dict__.get("page_book", {}).values())
                if carried and self.__dict__.get('label_indexed', None) is None:
                    self.label_indexed = False
                    first, second = self.vl_queries(k, label, cap_len)
                    sets[str(k)].update(self.vl_set(q) for q in first + second)
                    todo = self.vl_next(first, second, tkey, mode) or ""
            if not todo:
                why["todo"] = why.get("todo", 0) + 1
                settled[str(k)].append(sig)
                continue
            args, faults = ArgCheck.execute({**tmpl, tkey: todo}, jschema)
            if faults or not isinstance(args, (dict,)):
                settled[str(k)].append(sig)
                continue
            tries[k] = tries.get(k, 0) + 1
            vals = self.vl_values(k)
            self.walk_call(args, "I look up the needed value{} together with the category name exactly as this store's rows spell it ('{}'): keyword search checks listing values, titles and categories, so the results show those listings in the category that have the value.".format('' if not 1 < len(vals) else 's', todo))
            total = self.vl_seen(todo, tkey, mode)
            self.vl_indexed(label)
            key = json.dumps({a: v for a, v in args.items() if not found or a != found[0]}, default=str, sort_keys=True)
            walk = self.__dict__.get('page_book', {}).get(key, None)
            every = bool(walk) and len(walk.get('row_shelf', None) or {}) >= walk["total"] and any(s in home for s in (walk.get('row_shelf', None) or {}).values())
            floor = self.order_due() + 2
            queue: list[str] = []
            note = ("[value search] '{}': ".format(todo) + ('{} match(es)'.format(total) + ('.' if not every else ' - every one is shown here.') if total else 'no listing matched.') if not total is None else '')
            if every and 'cheapest' == self.needs.priority() and found and not walk.get('id_order', None) is False:
                self.__dict__.setdefault("valued", set()).add(key)
                self.__dict__.setdefault("valued_keys", {}).setdefault(str(k), []).append(key)
                if not 1 < len(products):
                    queue = self.vl_queue(k, [i for i, s in (walk.get('row_shelf', None) or {}).items() if s in home], floor, CEIL_OPENS)
                else:
                    self.__dict__.setdefault('valued_wait', {})[k] = True
            elif every and 'cheapest' != self.needs.priority() and todo in (first or second):
                queue = self.vl_queue(k, [i for i, s in (walk.get('row_shelf', None) or {}).items() if s in home], floor, LABEL_OPENS)
            if queue:
                note += (" Its unread in-stock rows within the budget on the requested category" + (('' if not (self.walk_pick(k) and 'cheapest' == self.needs.priority()) else ' below the pick')) + ", cheapest first, are opened next: "
                         + ", ".join(f"{i} ({self.archive.price(i):g})" for i in queue) + ".")
            if note and self.chat and 'tool' == self.chat[-1].get('role', None):
                self.chat[-1]["content"] = str(self.chat[-1]["content"]) + "\n" + note + ('\n' + self.status_line() if not queue else '')
            self.log.append(dict(kind='label_search', product=k, query=todo, total=total, every=every, opened=queue))
            if queue:
                self.vl_launch(queue, home, floor, "I go through the lowest-priced rows of that search that I haven't read yet: each listing in the category that has the required values shows up among its results.")
            return True
        return False

    def vl_flush(self, floor):
        wait = self.__dict__.setdefault("valued_wait", {})
        ks = [k for k, on in wait.items() if on]
        wait.clear()
        if not ks:
            return False
        book = self.__dict__.get("page_book", {})
        keys = self.__dict__.get('valued_keys', None) or {}
        opened = self.__dict__.setdefault("valued_opens", {})
        batch = self.batch_tool()
        per = 1 if not self.moves_left() - floor - 1 >= 2 * len(ks) else 2
        pools: list[list[str]] = []
        for k in ks:
            home = self.walk_home(k)
            ids = {i for key in keys.get(str(k), []) if key in book and self.walk_carries(book[key]["args"], k)
                   for i, s in (book[key].get('row_shelf', None) or {}).items() if s in home}
            pick = self.walk_pick(k)
            pools.append([i for _, i in self.cheap_unread(ids, k, None if not pick else pick[1])][:max(0, min(per, CEIL_OPENS - opened.get(k, 0)))])
        owner: dict[str, Any] = {}
        queue: list[str] = []
        most = per * len(ks) if not batch else batch[2]
        while len(queue) < most and any(pools):
            for k, pool in zip(ks, pools):
                while pool and pool[0] in queue:
                    pool.pop(0)
                if pool and len(queue) < most:
                    owner[pool[0]] = k
                    queue.append(pool.pop(0))
        while queue and (len(queue) if not (batch and 2 <= len(queue)) else 1) >= self.moves_left() - floor:
            queue.pop()
        if not queue:
            return False
        for i in queue:
            opened[owner[i]] = opened.get(owner[i], 0) + 1
        homes = set().union(*(self.walk_home(k) for k in ks))
        self.vl_launch(queue, homes, floor, "Since the lowest price is wanted, I go through the cheapest rows that the value search for each product turned up and that haven't been read yet.")
        summary = ("[value searches read] Below each product's pick, the cheapest in-stock rows not yet read and still under the budget, as found by that product's value search, were opened: " + ", ".join(f"{i} ({self.archive.price(i):g})" for i in queue) + '. Per product, pick the listing with the smallest price whose own fields satisfy all the requirements currently active; if a match is pricier, first open and rule out each cheaper candidate.')
        if self.chat and 'tool' == self.chat[-1].get('role', None):
            self.chat[-1]["content"] = str(self.chat[-1]["content"]) + "\n" + summary
        self.log.append(dict(kind='value_flush', products=ks, opened=queue))
        return True

    def product_steps(self):
        try:
            rows = self.needs.items()
            if 2 > len(rows) or 'cheapest' != self.needs.priority():
                return ""
            used = self.__dict__.get('valued_by', None) or {}
            parts, nxt = [], ""
            for k, row in enumerate(rows, 0):
                pick = self.walk_pick(k)
                parts.append('{}: {} value page(s) read, pick {}'.format(row['product'], used.get(k, 0), 'no' if not pick else 'yes'))
                nxt = nxt or (str(row['product']) if not pick else '')
            return "\n[product steps] " + "; ".join(parts) + ('' if not nxt else '; next: {}'.format(nxt)) + "."
        except (Exception,):
            return ""

    def seen_shelves(self):
        tally: dict[str, dict[str, int]] = {}

        def visit(d):
            if not any(isinstance(v, (str,)) and v in self.archive.ids and "id" in str(key).lower() for key, v in d.items()):
                return
            raw = next((val.strip(None) for key, val in d.items() if isinstance(val, (str,)) and 0 < len(val.strip(None)) <= 60
                        and re.search(r"categor|shelf|department|aisle", str(key).lower()) and not 'id' in str(key).lower()), "")
            if raw:
                tally.setdefault(raw.lower(), {})[raw] = tally.setdefault(raw.lower(), {}).get(raw, 0) + 1
        for rec in self.archive.items:
            walk_dicts(rec, visit)
        return {low: max(spell, key=spell.get) for low, spell in tally.items()}

    def shelf_fix(self, call_name, call_args):
        props = (self.shapes.get(call_name, None) or {}).get('properties', None)
        if not isinstance(props, (dict,)):
            return None
        key = next((k for k, spec in props.items() if isinstance(spec, (dict,)) and 'string' == spec.get('type', None) and not spec.get('enum', None)
                    and re.search(r"categor|shelf|department|aisle", str(k).lower())), None)
        if not key or not key in call_args or not isinstance(call_args.get(key, None), (str,)):
            return None
        given = call_args[key].strip(None)
        seen = self.seen_shelves()
        if not seen or given.lower() in seen:
            return None
        own = set() if not given else set(kind_tokens(given))
        best, top = "", 0.0
        for low in seen:
            score = 0.0 if not own else kind_score(own, set(kind_tokens(low)), set())
            if top < score:
                best, top = low, score
        if not best and given:
            return None
        if not best:
            home = set() if self.needs.items() else self.walk_home(None)
            best = '' if not (home and self.raw_label(home)) else self.raw_label(home).lower()
        if not best or not best in seen:
            return None
        self.log.append(dict(kind='shelf_fix', tool=call_name, given=given[:60], to=seen[best]))
        return dict(call_args, **{key: seen[best]})

    def ladder_tool(self):
        tool = self.mv_browse_tool()
        if not tool:
            return None
        name, cat, props = tool
        bound = next((k for k, spec in props.items() if isinstance(spec, (dict,)) and spec.get('type', None) in ("number", "integer")
                      and re.search(r"cap|max|ceil|price|cost|budget|upper|most", str(k).lower())
                      and not re.search(r"page|start|offset|size|count|min|low", str(k).lower())), None)
        tok = next((k for k, spec in props.items() if isinstance(spec, (dict,)) and spec.get('type', None) in ("string", "integer")
                    and not k in (cat, bound) and re.search(r"next|token|continu|cursor|after|start|offset|from", str(k).lower())), None)
        return None if not bound else (name, cat, bound, tok, props)

    @staticmethod
    def ladder_cap(top, spec):
        return round(float(top) - 0.01, 2) if not 'integer' == spec.get('type', None) else float(int(top) - 1)

    def title_of_id(self, ident):
        found = ""

        def visit(d):
            nonlocal found
            if not found and any(isinstance(v, (str,)) and ident == v and "id" in str(k).lower() for k, v in d.items()):
                found = self.title_of(d) or ""
        for rec in self.archive.items:
            walk_dicts(rec, visit)
            if found:
                break
        return found

    def unknown_hold(self):
        try:
            if 2 <= self.__dict__.get('unk_holds', 0) or self.moves_left() <= self.order_due() + 2 or self.wire.spare() < CLOSING + 20:
                return ""
            weak = [i for i in self.trolley.units if 0 < self.unknowns(i)]
            if not weak:
                return ""
            line = weak[0]
            k = None if not self.needs.items() else self.product_index(line)
            home = self.walk_home(k)
            ids: set[str] = set()

            def visit(d):
                ident = self.row_mark(d)
                if ident in self.archive.ids and line != ident and self.shelf_of(d) in home:
                    ids.add(ident)
            for rec in self.archive.items:
                walk_dicts(rec, visit, ceiling=3)
            vals = {w for req in self.needs.props_left() for w in re.findall(r"[a-z]{3,}", req.partition(":")[2].lower())}
            pool = self.cheap_unread(ids, k)
            pool.sort(key=lambda ci: (-len(vals & set(re.findall(r"[a-z]{3,}", self.title_of_id(ci[1]).lower()))), ci[0]))
            queue = [i for _, i in pool][:4]
            if not queue:
                return ""
            self.unk_holds = self.__dict__.get("unk_holds", 0) + 1
            self.log.append(dict(kind='unknown_hold', line=line, opens=queue))
            self.open_queue = queue
            self.open_say = 'Because the cart line leaves one requirement unverified, I first go through further listings in its category.'
            self.open_purpose, self.open_floor = "walk", self.order_due() + 1
            return ('Not run yet (no step used). The item check was unable to verify all requirements for {} (one requirement appears in none of its records), and such orders have not gone through. The code now opens additional listings from its category: should any of them display every requirement, add that one to the cart rather than '.format(line) + line + "; then order.")
        except (Exception,) as err:
            self.log.append(dict(kind='unknown_hold_error', error='{}: {}'.format(type(err).__name__, err)[:300]))
            return ""

    def ladder_floor(self):
        return max(FIRST_ORDER_BY + 3, self.order_due() - 4)

    def ladder_top(self, k):
        fits = self.passed_for(k)
        if fits:
            top = self.archive.price(fits[0])
            return (None if not is_number(top) else float(top)), None
        gone = self.dead_lines() | set(self.refused_lines)
        ids = [key.rpartition("|")[0] for key in self.passed]
        sold = [float(self.archive.price(i)) for i in dict.fromkeys(ids) if i in gone and (k is None or self.product_index(i) == k)
                and is_number(self.archive.price(i))]
        if not sold:
            return None, None
        floor = min(sold)
        qty = self.needs.quantity() if not (not k is None and k < len(self.needs.items())) else self.needs.items()[k]['quantity']
        cap = self.needs.spend_cap
        top = floor * 4 if not (is_number(cap) and float(cap) > floor) else float(cap) / max(1, int(qty or 1))
        return min(top, floor * 2), floor

    def ladder_due(self):
        try:
            why = self.__dict__.setdefault("ladder_why", {})
            if 'cheapest' == self.needs.priority():
                tag = (((((('' if any((self.ladder_top(k)[0] for k in self.cg_products())) else 'no_fit') if self.ladder_tool() else 'tool') if not self.wire.spare() < CLOSING + 30 else 'time') if not self.moves_left() <= self.ladder_floor() else 'steps') if not self.orders_tried else 'ordered'))
                if tag:
                    why[tag] = why.get(tag, 0) + 1
            if 'cheapest' != self.needs.priority() or self.orders_tried or self.moves_left() <= self.ladder_floor() \
                    or self.wire.spare() < CLOSING + 30 or self.wire.done or self.wire.broke:
                return False
            spec = self.ladder_tool()
            if not spec:
                return False
            state = self.__dict__.get('ladder', None) or {}
            for k in self.cg_products():
                top, _ = self.ladder_top(k)
                if not is_number(top) or not self.raw_label(self.walk_home(k)):
                    continue
                st = state.get(k, None)
                if st is None:
                    return True
                if st["cap"] != self.ladder_cap(float(top), spec[4].get(spec[2], None) or {}):
                    if st["rungs"] < LADDER_RUNGS:
                        return True
                    continue
                if not st["end"] and st["pages"] < LADDER_PAGES:
                    return True
                if [i for _, i in self.cheap_unread(st["ids"], k, below=float(top)) if not i in st['opened']]:
                    return True
            return False
        except (Exception,) as err:
            self.log.append(dict(kind='ladder_due_error', error='{}: {}'.format(type(err).__name__, err)[:300]))
            return False

    def ladder_call(self, name, args, say):
        proposal = (dict(id='ladder_{}_{}'.format(len(self.chat), self.wire.tick), type='function', function=dict(name=name, arguments=json.dumps(args))))
        self.chat.append(dict(role='assistant', content=say, tool_calls=[proposal]))
        before = len(self.archive.items)
        responses = self.dispatch([proposal], scripted=True)
        if not responses:
            self.chat.append(dict(role='tool', tool_call_id=proposal['id'], content='Not run.'))
            return None
        cid, text_body = responses[0]
        got = len(self.archive.items) > before
        self.chat.append(dict(role='tool', tool_call_id=cid, content=text_body if not got else self.page_digest(self.archive.items[-1])))
        return None if not got else self.archive.items[-1]

    def ladder_step(self):
        try:
            if not self.ladder_due():
                return False
            name, cat, bound, tok, props = self.ladder_tool()
            state = self.__dict__.setdefault("ladder", {})
            for k in self.cg_products():
                top, floor = self.ladder_top(k)
                label = self.raw_label(self.walk_home(k))
                if not is_number(top) or not label:
                    continue
                if not floor is None:
                    self.log.append(dict(kind='ladder_soldout', k=k, top=top, floor=floor))
                top = float(top)
                cap = self.ladder_cap(top, props.get(bound, None) or {})
                if cap <= 0:
                    continue
                st = state.setdefault(k, dict(cap=None, next=None, pages=0, end=False, ids=[], rungs=0, opened=set()))
                if st["cap"] != cap:
                    if st["rungs"] >= LADDER_RUNGS:
                        continue
                    st.update(cap=cap, probe=cap, next=None, pages=0, end=False, ids=[])
                    st["rungs"] += 1
                vals = {w for req in self.needs.props_left() for w in re.findall(r"[a-z]{3,}", req.partition(":")[2].lower())}
                pool = [(c, i) for c, i in self.cheap_unread(st["ids"], k, below=top) if not i in st['opened']]
                pool.sort(key=lambda ci: (-len(vals & set(re.findall(r"[a-z]{3,}", self.title_of_id(ci[1]).lower()))), ci[0]))
                cheaper = [i for _, i in pool][:LADDER_OPENS]
                if cheaper and (st["end"] or st["pages"] >= LADDER_PAGES or 2 <= len(cheaper)):
                    st["opened"].update(cheaper)
                    self.log.append(dict(kind='ladder_open', k=k, below=top, ids=cheaper))
                    self.open_queue = cheaper
                    self.open_say = f'The request is for the cheapest fit: I open the listings whose price is under {top:g}, the lowest-priced fit found up to now.'
                    self.open_purpose, self.open_floor = "walk", self.order_due() + 2
                    return self.open_named()
                if st["end"] or st["pages"] >= LADDER_PAGES:
                    continue
                probe = st.get("probe", cap)
                args: dict[str, Any] = {cat: label, bound: probe if not 'integer' == (props.get(bound, None) or {}).get('type', None) else int(probe)}
                if tok and st.get('next', None):
                    args[tok] = st.pop("next")
                obs = self.ladder_call(name, args, f"The request is for the cheapest fit: using the store's price filter, I list the category items priced under {top:g}, the lowest-priced fit found up to now.")
                st["pages"] += 1
                rows: list[str] = []
                if isinstance(obs, (dict, list)):
                    walk_dicts(obs, lambda d: None if not (self.row_mark(d) in self.archive.ids and 4 <= len(d)) else rows.append(self.row_mark(d)), ceiling=3)
                rows = list(dict.fromkeys(rows))
                st["ids"] = list(dict.fromkeys(st["ids"] + rows))
                more = None if not isinstance(obs, (dict,)) else self.next_token(obs)
                if floor is None and tok and more and 1 == st['pages'] and rows:
                    st["next"] = more
                    self.log.append(dict(kind='ladder_page', k=k, cap=cap, rows=len(rows), paged=True))
                    return True
                prices = [float(self.archive.price(i)) for i in rows if is_number(self.archive.price(i))]
                if not floor is None and any(floor <= x for x in prices):
                    prices = [x for x in prices if floor <= x]
                low = None if not prices else min(prices)
                if not rows or low is None or low <= 0:
                    st["end"] = True
                else:
                    nxt = self.ladder_cap(low, props.get(bound, None) or {})
                    if not floor is None and floor < nxt:
                        nxt = min(nxt, self.ladder_cap(floor + (probe - floor) * 0.5, props.get(bound, None) or {}))
                    if not (nxt <= 0 or probe <= nxt or (not floor is None and floor > nxt)):
                        st['probe'] = nxt
                    else:
                        st['end'] = True
                self.log.append(dict(kind='ladder_page', k=k, cap=cap, rows=len(rows)))
                return True
            return False
        except (Exception,) as err:
            self.log.append(dict(kind='ladder_error', error='{}: {}'.format(type(err).__name__, err)[:300]))
            return False

    def depth_step(self):
        try:
            if self.orders_tried or self.wire.done or self.wire.broke or not self.role("search") or self.wire.spare() < CLOSING + 60:
                return False
            if any('open' == r.get('status', None) for r in self.__dict__.get('refs', None) or []) and self.ref_room():
                return False
            found = self.page_arg()
            if not found or self.moves_left() <= self.order_due() + 6:
                return False
            book = self.__dict__.get("page_book", {})
            done = self.__dict__.setdefault("depth_done", set())
            ladder = 'cheapest' == self.needs.priority() and bool(self.ladder_tool())
            if ladder and self.__dict__.get("depth_pages", 0) >= max(2, self.step_cap // 8):
                return False
            if self.__dict__.get("depth_pages", 0) >= max(4, self.step_cap // 4):
                return False
            rows_cap = DEPTH_ROWS if not ladder else 12
            sellers = set()
            for ref in self.__dict__.get('refs', None) or []:
                for tgt in (ref.get('target', None) or {}, ref.get('first', None) or {}):
                    name = ref_name(str(tgt.get('seller', None) or ""))
                    if name:
                        sellers.add(name)
            rows = self.needs.items()
            for k in self.cg_products():
                home = self.walk_home(k)
                if not home:
                    continue
                several = bool(rows) and not k is None
                label = {self.stem(w) for h in home for w in re.findall(r"[a-z0-9]{2,}", h)}
                product = str(self.needs.reading.get('product', None) or '') if not (several and k < len(rows)) else str(rows[k]['product'])
                kind = {self.stem(w) for w in re.findall(r"[a-z0-9]{3,}", product.lower())}
                values = {self.stem(w) for req in self.needs.props_left() if not several or self.product_of(req.partition(":")[0]) in (k, None)
                          for w in re.findall(r"[a-z0-9]{2,}", req.partition(":")[2].lower())}
                mains = []
                for key, walk in book.items():
                    if self.cg_listing(walk["args"]):
                        continue
                    text = " ".join(self.free_text(walk["args"])).lower()
                    words = {self.stem(w) for w in re.findall(r"[a-z0-9]{2,}", text)}
                    if not words or not words & values or any(n and n in text for n in sellers):
                        continue
                    if not any(sh in home for sh in (walk.get('row_shelf', None) or {}).values()):
                        continue
                    mains.append((key, walk))
                for key, walk in mains[:2 if not (ladder or 1 < len(rows)) else 1]:
                    if key in done:
                        continue
                    size = max(1, int(walk.get('size', None) or 0))
                    last = walk.get('last_rows', None) or []
                    shelf = walk.get('row_shelf', None) or {}
                    nxt = self.next_page(walk)
                    if nxt is None or len(walk.get('read', None) or ()) * size >= rows_cap or len(last) < size \
                            or not any(shelf.get(m, None) in home for m in last):
                        done.add(key)
                        continue
                    args = dict(walk["args"])
                    args[found[0]] = nxt
                    self.depth_pages = self.__dict__.get("depth_pages", 0) + 1
                    self.log.append(dict(kind='depth_page', query=' '.join(self.free_text(walk['args']))[:80], page=str(nxt)[:20]))
                    return self.walk_call(args, 'Since the store displays {} rows per page, this search has more matches beyond the pages already read: the next page is what I read.'.format(size))
            return False
        except (Exception,) as err:
            self.log.append(dict(kind='depth_error', error='{}: {}'.format(type(err).__name__, err)[:300]))
            return False

    def value_pages(self):
        try:
            if any('open' == r.get('status', None) for r in self.__dict__.get('refs', None) or []) and self.ref_room():
                return False
            if self.orders_tried or self.refusal_n or self.wire.done or self.wire.broke or self.wire.spare() < CLOSING + 60 \
                    or not self.role("search"):
                why = self.__dict__.setdefault("vl_why", {})
                tag = ('other' if not self.wire.spare() < CLOSING + 60 else 'time') if not (self.orders_tried or self.refusal_n) else 'ordered'
                why[tag] = why.get(tag, 0) + 1
                return False
            if self.vl_search():
                return True
            if 'cheapest' != self.needs.priority() or self.word_kind == 'any':
                return False
            found = self.page_arg()
            if not found:
                return False
            book = self.__dict__.get("page_book", {})
            done = self.__dict__.setdefault("valued", set())
            used = self.__dict__.setdefault("valued_by", {})
            opened = self.__dict__.setdefault("valued_opens", {})
            wait = self.__dict__.setdefault("valued_wait", {})
            sets = self.__dict__.get('label_sets', None) or {}
            depth = self.visible_depth()
            size = max(1, self.page_size())
            per = VALUE_PAGES if not depth else max(VALUE_PAGES, -(-depth // size) - 1)
            products = self.cg_products()
            several = 1 < len(products)
            cap = max(VALUE_TOTAL, VALUE_TOTAL_PER * len(products))
            opens = (CEIL_OPENS if not several else 2) if not self.batch_tool() else 1
            floor = self.order_due() + 2
            known = self.all_words()
            left, room = self.vl_room()
            share = max(4, room // max(1, len([k for k in products if self.walk_home(k) and not used.get(k, None)])))
            options, pending = [], False
            for n, k in enumerate(products, 0):
                home = self.walk_home(k)
                if not home:
                    continue
                mine = sets.get(str(k), None) or set()
                several_k = bool(self.needs.items()) and not k is None
                values = {self.stem(w) for req in self.needs.props_left() if not several_k or self.product_of(req.partition(":")[0]) in (k, None)
                          for w in re.findall(r"[a-z0-9]{2,}", req.partition(":")[2].lower())}
                label = {self.stem(w) for s in home for w in re.findall(r"[a-z0-9]{2,}", s)}
                whole = [(self.vl_set(" ".join(self.free_text(w["args"]))), self.walk_carries(w["args"], k, alts=True), self.vl_mode(w["args"]))
                         for w in book.values() if w.get('id_order', None) is True and w.get('first', None) and self.next_page(w) is None
                         and not self.cg_listing(w["args"]) and len(w.get('row_shelf', None) or {}) >= w["total"]]
                cands = []
                for key, walk in book.items():
                    if key in done or not walk.get('id_order', None) is True or not walk.get('first', None) or self.next_page(walk) is None \
                            or self.cg_listing(walk["args"]) or not self.walk_carries(walk["args"], k) \
                            or not any(s in home for s in (walk.get('row_shelf', None) or {}).values()):
                        continue
                    words = self.vl_set(" ".join(self.free_text(walk["args"])))
                    composed = words in mine
                    if not known and (depth is None or walk["total"] > depth):
                        continue
                    if known and not depth is None and walk["total"] > depth and not composed:
                        continue
                    pairs = self.walk_carries(walk["args"], k, alts=True)
                    how = self.vl_mode(walk["args"])
                    if known and any(how == mode and (words >= seen or (seen <= values | label and got >= pairs)) for seen, got, mode in whole):
                        continue
                    reach = min(walk['total'], depth) if not depth is None else walk['total']
                    pages = -(-reach // max(1, walk["size"])) - len(walk["read"])
                    allowed = min(per - used.get(k, 0), cap - sum(used.values()))
                    if several:
                        allowed = min(allowed, share - used.get(k, 0) - opened.get(k, 0) - (opens if not opened.get(k, None) else 0))
                    take = (0 if not composed else allowed) if not allowed >= pages else pages
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
        except (Exception,) as err:
            self.log.append(dict(kind='value_pages_error', error='{}: {}'.format(type(err).__name__, err)[:300]))
            return False

    def read_value_search(self, k, key, floor, most = VALUE_PAGES, defer = False):
        found = self.page_arg()
        walk = self.__dict__.get('page_book', {}).get(key, None)
        if not found or not walk:
            return False
        query = " ".join(self.free_text(walk["args"]))
        used = self.__dict__.setdefault("valued_by", {})
        known = self.all_words()
        say = (("Since the lowest price is wanted, and this store returns search results sorted by a set id order rather than by price, I fetch the following page of '{}' so each listing it matched gets seen.".format(query) if not known else "Since the lowest price is wanted, and this store's keyword search demands all the words and returns results sorted by a set id order rather than price, I fetch the following page of '{}' so each listing containing those words gets seen.".format(query)))
        got, tried = 0, False
        while not self.next_page(walk) is None and most > got and self.moves_left() > floor + 1 and not self.wire.done \
                and not self.wire.broke and self.wire.spare() >= CLOSING + 30:
            tried = True
            used[k] = used.get(k, 0) + 1
            nxt = self.next_page(walk)
            if not self.walk_call({**walk["args"], found[0]: nxt}, say):
                break
            got = got + 1
            if self.next_page(walk) == nxt:
                break
        if not got:
            return tried
        self.__dict__.setdefault("valued_keys", {}).setdefault(str(k), []).append(key)
        seen = len(walk.get('row_shelf', None) or {})
        complete = self.next_page(walk) is None and seen >= walk["total"]
        home = self.walk_home(k)
        head = ("[value search read] '{}': {} out of {} matches viewed".format(query, seen, walk['total'])
                + ((('' if not self.next_page(walk) is None else ' - the others are out of reach, so a fit that costs less might sit beyond the rows that can be reached') if not complete else ' - all the listings it matched are included in them' if not known else ' - all listings containing these words are included in them')) + ". ")
        queue: list[str] = []
        if not defer:
            pick = self.walk_pick(k)
            queue = self.vl_queue(k, [i for i, s in (walk.get('row_shelf', None) or {}).items() if s in home], floor, CEIL_OPENS)
            summary = head + ('Among them, nothing unread and in stock within the budget remains in the requested category. ' if not queue else 'The lowest-priced in-stock rows not yet read that fall inside the budget for the category asked for' + ('' if not pick else ' below the pick') + ' are opened next: ' + ', '.join((f'{i} ({self.archive.price(i):g})' for i in queue)) + '. ') + 'Pick the listing with the lowest price whose own fields satisfy all requirements currently active; to pick a pricier match, first open and rule out every cheaper candidate.'
            if queue:
                self.vl_launch(queue, home, floor, "The request is for the cheapest: I open the lowest-priced rows of that search that I haven't read yet.")
        else:
            self.__dict__.setdefault('valued_wait', {})[k] = True
            summary = head + "Once every product's value search has been read, its cheapest in-stock rows under the budget that haven't been read yet get opened. Pick the listing with the lowest price whose own fields satisfy all requirements currently active; to pick a pricier match, first open and rule out every cheaper candidate."
        if self.chat and 'tool' == self.chat[-1].get('role', None):
            self.chat[-1]["content"] = str(self.chat[-1]["content"]) + "\n" + summary + ('\n' + self.status_line() if not queue else '')
        self.log.append((dict(kind='value_pages', product=k, query=query, pages=got, complete=complete, opened=queue, defer=defer)))
        return True

    def ceiling_walk(self, k, floor, repair = False):
        spec, found = self.ceil_spec(), self.page_arg()
        home = self.walk_home(k)
        label = self.raw_label(home)
        aim = self.ceil_aim(k)
        if not spec or not found or not label or aim is None:
            return None
        self.__dict__.setdefault("ceil_walked", set()).add(k)
        pick_now = self.walk_pick(k)
        self.__dict__.setdefault("ceil_aimed", {})[k] = (float(aim), '' if not pick_now else pick_now[0])
        bound = self.below(aim)
        if bound is None:
            return None
        floor = floor + (0 if not repair else 2) if not (not self.orders_tried and (not repair)) else max(floor, self.order_due() + 2)
        depth = self.visible_depth()
        book = self.__dict__.setdefault("page_book", {})
        keys: list[str] = []
        calls, top, complete, ran, last = 0, None, False, False, bound
        what = 'the cheapest row seen' if not (pick_now or is_number((self.__dict__.get('ceil_aims', None) or {}).get(k, None))) else 'the pick'
        say = f"The request is for the cheapest, so I use the store's price filter to list '{label}' costing less than {what} ({aim:g})."
        while CEIL_CALLS > calls and self.moves_left() > floor + 1 and not self.wire.done and not self.wire.broke \
                and self.wire.spare() >= CLOSING + 30:
            args = self.ceil_args(spec, label, bound)
            key = json.dumps(args, default=str, sort_keys=True)
            self.__dict__.setdefault("ceil_keys", set()).add(key)
            calls = calls + 1
            reuse = key in book and bool(book[key].get('first', None)) and key in (self.__dict__.get('ceil_seen', None) or {})
            if not reuse and not self.walk_call(args, say):
                break
            ran, last = True, bound
            keys.append(key)
            walk = book.get(key, None)
            if walk and any(is_number(self.archive.price(i)) and self.archive.price(i) > bound for i in (walk.get('row_shelf', None) or {})):
                self.__dict__.setdefault("ceil_bad", set()).add(spec["sig"])
            if self.ceil_spec() is None:
                break
            total = self.__dict__.get('ceil_seen', {}).get(key, None)
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
            if (depth is None or depth >= total) and pages - len(walk["read"]) <= CEIL_CALLS - calls:
                while not self.next_page(walk) is None and CEIL_CALLS > calls and self.moves_left() > floor + 1 \
                        and not self.wire.done and not self.wire.broke and self.wire.spare() >= CLOSING + 30:
                    calls = calls + 1
                    nxt = self.next_page(walk)
                    if not self.walk_call({**args, found[0]: nxt}, 'The following page of that price-filtered listing has now been read by me.') \
                            or self.next_page(walk) == nxt:
                        break
                complete = self.next_page(walk) is None
                break
            top = bound
            costs = [float(c) for c in (self.archive.price(i) for i in (walk.get('row_shelf', None) or {})) if is_number(c)]
            low = 0.0 if not costs else min(costs)
            fill = max(1, CEIL_CALLS - calls) * max(1, walk["size"])
            nb = min(self.ceil_mid(low, bound), self.ceil_mid(low, low + (bound - low) * 1.8 * fill / max(total, 1)))
            if not bound > nb:
                break
            bound = nb
        if not ran:
            return None
        ids: set[str] = set()
        for key in keys:
            ids |= {i for i, s in (book.get(key, {}).get('row_shelf', None) or {}).items() if s in home}
        cands = [i for _, i in self.cheap_unread(ids, k, aim)]
        valued = self.valued_ids(k)
        lead = [i for i in cands if i in valued]
        rest = [i for i in cands if not i in valued]
        texty = [i for i in rest if self.row_carries(i, k)]
        bare = [i for i in rest if not i in texty] if not self.valued_complete(k) else []
        ranked = lead + texty + bare
        if complete and not ids:
            report = f"with the price filter on '{label}' under {aim:g}, there is no listing of it priced below {what}"
        elif complete:
            report = (f"the price filter on '{label}': all listings costing up to {last:g} were seen ({len(ids)}, of which {len(cands)} are unread, in stock and under the budget)")
        else:
            report = f"the price filter on '{label}' under {aim:g}: {len(ids)} of the less expensive listings were seen ({len(cands)} unread, in stock and under the budget)"
        self.log.append((dict(kind='ceiling_walk', product=k, label=label, aim=aim, calls=calls, complete=complete, rows=len(ids), ranked=ranked[:CEIL_OPENS])))
        return report, ranked[:CEIL_OPENS]

    def ceiling_sweep(self, tried):
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
        key = json.dumps(args, default=str, sort_keys=True)
        if not key in tried:
            self.__dict__.setdefault("ceil_keys", set()).add(key)
            return args, False
        walk = self.__dict__.get('page_book', {}).get(key, None)
        if not walk:
            return None, 0 == self.__dict__.get('ceil_seen', {}).get(key, None) and not self.ceil_clash(set(shelves), bound)
        nxt = self.next_page(walk)
        if not nxt is None:
            more = {**args, found[0]: nxt}
            return (None, False) if not not json.dumps(more, default=str, sort_keys=True) in tried else (more, False)
        return None, len(walk.get('row_shelf', None) or {}) >= walk["total"]

    def value_guard(self, call_name, call_args):
        try:
            if call_name != self.role("details") or 'cheapest' != self.needs.priority() or not self.all_words() or self.in_reserve():
                return ""
            named = [v for v in call_args.values() if isinstance(v, (str,)) and v in self.archive.ids]
            if 1 != len(named) or named[0] in self.archive.read or named[0] in self.trolley.units:
                return ""
            ident = named[0]
            if ident in self.named_open() or self.__dict__.get('refs', None) or self.needs.reading.get('references', None):
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
            costs = ([] if not pick else [pick[1]]) + ([] if not is_number(mine) else [float(mine)])
            limit = None if not costs else min(costs)
            rows = [] if not not limit is None else self.cheap_unread(valued, k, limit)[:6]
            if not rows:
                return ""
            told.add(mark)
            return ('Not run (no step used). The lowest price is what was requested, and keyword search in this store requires all words to match, so the rows shown by its searches that began with a required value contain those words somewhere in their records; {} is not one of them, yet these rows from that set are cheaper'.format(ident) + (' than it' if not pick else ' than it and the pick') + " and are unread: "
                    + ", ".join(f"{i} ({c:g})" for c, i in rows) + '. Start by opening these, lowest price first; repeat this call afterwards if you still need it.')
        except (Exception,) as err:
            self.log.append(dict(kind='value_guard_error', error='{}: {}'.format(type(err).__name__, err)[:300]))
            return ""

    def title_line(self, k, rec, template):
        avoid = set(LINK_WORDS) | {w for s in self.walk_home(k) for w in re.findall(r"[a-z]+", s)}
        avoid |= {w for req in self.needs.props_left() for w in re.findall(r"[a-z0-9]+", req.lower())}
        avoid |= {w for r in (self.needs.items() or [dict(product=str(self.needs.reading.get('product', None) or ''))])
                  for w in re.findall(r"[a-z]+", str(r.get('product', None) or "").lower())}
        stems = {self.stem(w) for w in avoid}
        toks = re.findall(r"[a-z0-9]+", self.title_of(rec).lower())
        for n, tok in enumerate(toks, 0):
            if re.fullmatch(r"[a-z]{3,}", tok) and not tok in avoid and not self.stem(tok) in stems:
                nxt = '' if not n + 1 < len(toks) else toks[n + 1]
                text = tok + ('' if not (nxt and not nxt in avoid and (not self.stem(nxt) in stems)) else ' {}'.format(nxt))
                return {**template[0], template[1]: text} if not self.ran_query(text) else None
        return None

    def rearm_walk(self):
        if 'cheapest' != self.needs.priority() or self.wire.done or self.wire.spare() < CLOSING + 60 or not self.role("search"):
            return None
        aimed = self.__dict__.get('ceil_aimed', None) or {}
        gone = self.dead_lines() | self.refused_lines
        opens = 2 if not self.batch_tool() else 1
        for k, (aim, ident) in list(aimed.items()):
            if not ident or not ident in gone or self.ceil_spec() is None or self.page_arg() is None:
                continue
            pick = self.walk_pick(k)
            if not pick or pick[1] <= aim + 1e-9 or self.moves_left() < FIRST_ORDER_BY + 3 + opens:
                continue
            aimed.pop(k, None)
            (self.__dict__.get('ceil_aims', None) or {}).pop(k, None)
            self.log.append(dict(kind='ceil_rearm', product=k, dead=ident, aim=pick[1]))
            return self.run_walk([], FIRST_ORDER_BY, [k], repair=True)
        return None

    def walk_step(self):
        try:
            if self.__dict__.get('decline_queue', None):
                return self.run_decline_reads()
            queued = self.__dict__.pop("walk_queue", None)
            if queued:
                return self.run_walk(*queued)
            again = self.rearm_walk()
            if not again is None:
                return again
            if self.orders_tried or 'cheapest' != self.needs.priority() or self.wire.spare() < CLOSING + 60 or self.wire.done:
                return False
            if self.unfound_product():
                return False
            opens = 2 if not self.batch_tool() else 1
            spare = self.moves_left()
            ceil = [k for k in self.cg_products() if (self.walk_pick(k) or self.bound_need(k)) and self.ceiling_ready(k)
                    and not self.ceil_aim(k) is None]
            if ceil and spare > self.order_due() + 2 + CEIL_CALLS + opens:
                used = self.__dict__.get('valued_by', None) or {}
                ceil.sort(key=lambda k: (bool(self.walk_pick(k)), used.get(k, 0), k if not k is None else -1))
                return self.run_walk([], self.order_due() + 2, ceil[:1])
            if not ceil and spare > self.order_due() + 2 + CEIL_CALLS + opens and self.ceil_seed():
                return True
            plan = self.walk_plan(WALK_TOTAL)
            if not plan:
                return False
            pages = sum(n for _, _, n in plan) + sum(1 for k, _, _ in plan if self.walk_pick(k))
            pace = (monotonic() - self.wire.t0) / max(1, self.wire.tick)
            need = (pages + opens + self.order_due() + 2) * pace + CLOSING + 30
            if spare > self.order_due() + 2 + pages + opens and self.wire.spare() > need:
                return False
            plan = self.walk_plan(min(WALK_TOTAL, spare - self.order_due() - 1 - opens))
            if not plan:
                return False
            self.walk_early = True
            return self.run_walk(plan, self.order_due())
        except (Exception,) as err:
            self.log.append(dict(kind='walk_error', error='{}: {}'.format(type(err).__name__, err)[:300]))
            return False

    def unstated_n(self, thing):
        words: set[str] = set()

        def visit(d):
            if any(isinstance(v, (str,)) and thing == v and "id" in str(k).lower() for k, v in d.items()):
                for k, v in d.items():
                    words.update(re.findall(r"[a-z]+", str(k).lower()))
                    if isinstance(v, (str, int, float)):
                        words.update(re.findall(r"[a-z]+", str(v).lower()))
        for rec in self.archive.items:
            walk_dicts(rec, visit)
        if not words:
            return 0
        rows = self.needs.items()
        mine = None if not rows else self.product_index(thing)
        product = str(self.needs.reading.get('product', None) or '') if not (rows and not mine is None and (mine < len(rows))) else str(rows[mine]['product'])
        common = set(re.findall(r"[a-z]+", product.lower()))
        n = 0
        for req in self.needs.props_left():
            head, _, value = str(req).partition(":")
            low = value.strip(None).lower()
            if rows and not self.product_of(head) in (mine, None):
                continue
            if not low or re.search(r"\d", low) or ref_text(low) or re.search(CH_BOUND, low) \
                    or re.search(r"\b(?:not|no|none|without|other than|except|avoid)\b", low):
                continue
            alts = [{w for w in re.findall(r"[a-z]+", alt) if 3 < len(w)} - common
                    for alt in re.split(r"\s*(?:\||/|,|\bor\b)\s*", low)]
            alts = [a for a in alts if a]
            if alts and not any(a & words or {w.rstrip("s") for w in a} & {w.rstrip("s") for w in words} for a in alts):
                n = n + 1
        return n

    def unknowns(self, ident):
        return int((self.__dict__.get('unknown_n', None) or {}).get('{}|{}'.format(ident, self.needs.version), 0))

    def passed_for(self, k):
        gone = self.dead_lines()
        ids = [key.rpartition("|")[0] for key in self.passed if key.rpartition("|")[2] == str(self.needs.version)]
        ids = [i for i in dict.fromkeys(ids) if not i in self.refused_lines and not i in gone
               and (k is None or self.product_index(i) == k)]
        return sorted(ids, key=lambda i: (0 < self.unknowns(i), 1000000000000.0 if not is_number(self.archive.price(i)) else float(self.archive.price(i)), i))

    def spec_read_cart(self):
        try:
            details = self.role("details")
            sig = (tuple(sorted(self.trolley.units)), self.__dict__.get("cart_edits", 0))
            if not details or self.__dict__.get('cart_spec_mark', None) == sig or 4 > self.moves_left() or 8 > self.wire.spare():
                return False
            done = set((self.__dict__.get('read_with', None) or {}).get(details, None) or set())
            need = [i for i in self.trolley.units if not i in done][:min(3, self.moves_left() - 3)]
            self.cart_spec_mark = sig
            if not need:
                return False
            responses = self.open_reads(need, 'Before the order goes through, each of its lines is read again to check its details and stock.', None, details)
            if responses:
                self.post_replies(responses)
            self.log.append(dict(kind='lines_read', ids=need))
            return bool(responses)
        except (Exception,) as err:
            self.log.append(dict(kind='read_lines_error', error='{}: {}'.format(type(err).__name__, err)[:300]))
            return False

    def open_link_pages(self):
        try:
            if self.__dict__.get('link_pages_done', None) or 4 > self.moves_left() or 8 > self.wire.spare():
                return False
            tool = self.ref_notes_tool() or self.role("view")
            if not tool:
                return False
            done = self.__dict__.get('read_with', None) or {}
            batch = self.batch_tool()
            seen = set(done.get(tool, None) or set()) | (set() if not batch else set(done.get(batch[0], None) or set()))
            need = [i for r in self.__dict__.get('refs', None) or [] for i in r.get('chain', None) or [] if isinstance(i, (str,)) and not i in seen]
            need = list(dict.fromkeys(need))[:min(4, self.moves_left() - 3)]
            if not need:
                return False
            self.link_pages_done = True
            responses = self.open_reads(need, "Before the order, all listings that the request's reference led to get opened.", None, tool)
            if responses:
                self.post_replies(responses)
            self.log.append(dict(kind='link_pages_done', ids=need))
            return bool(responses)
        except (Exception,) as err:
            self.log.append(dict(kind='view_chain_error', error='{}: {}'.format(type(err).__name__, err)[:300]))
            return False

    def basket_answered(self, call_args):
        text = " ".join(str(v) for v in call_args.values() if isinstance(v, (str,)))
        if not re.search(r"basket|total|order|go.?ahead|\byes\b", text, re.I):
            return False
        edits = self.__dict__.get("cart_edits", 0)
        sent = [m for m in self.__dict__.get('msg_log', None) or [] if m[2] == edits]
        asked = [m for m in sent if "?" in m[1] and re.search(r"basket|total|order|go.?ahead", m[1], re.I)]
        if not asked:
            return False
        said = asked[-1][1][:120]
        reply = ""
        for line in reversed(self.questions_log):
            if "(the assistant's own message)" in line and said[:60] in line:
                reply = '' if not '->' in line else line.split('->', 1)[1]
                break
        if not reply.strip(None):
            return False
        return not re.search(r"\b(?:no|nope|don'?t|do not|wait|hold on|stop|change|instead|cancel|rather|wrong|remove|swap)\b"
                             r"|too (?:much|dear|expensive)|over (?:the|my) budget", reply, re.I)

    def basket_due(self):
        if 1 > int(self.__dict__.get('mv_keep', 0) or 0) or not self.trolley.units or not self.role("message"):
            return False
        if not self.msg_cap is None and self.msg_cap <= self.msgs_sent:
            return False
        sent = self.__dict__.get('msg_log', None) or []
        return not (sent and sent[-1][2] == self.__dict__.get("cart_edits", 0))

    def send_basket(self):
        proposal = self.message_call(self.basket_text(), 'basket_{}'.format(self.wire.tick))
        if not proposal or 2 > self.moves_left():
            return False
        self.send_chunk([(proposal["id"], proposal["function"]["name"], json.loads(proposal["function"]["arguments"]))])
        return True

    def passed_fits(self):
        if self.needs.items():
            return []
        gone = self.dead_lines()
        ids = [key.rpartition("|")[0] for key in self.passed if key.rpartition("|")[2] == str(self.needs.version)]
        ids = [i for i in dict.fromkeys(ids) if not i in self.refused_lines and not i in gone]
        return sorted(ids, key=lambda i: (1000000000000.0 if not is_number(self.archive.price(i)) else float(self.archive.price(i)), i))

    def value_search_step(self):
        if (self.orders_tried or not self.role("search")
                or self.wire.done or self.wire.spare() < CLOSING + 80 or self.moves_left() <= self.order_due() + 6):
            return False
        if not self.needs.priority() in ('cheapest', 'best') and (self.passed_fits() or self.trolley.units):
            return False
        done = self.__dict__.setdefault("vsearch_done", set())
        book = self.__dict__.get("page_book", {})
        if len(done) >= VSEARCH_MAX or not book:
            return False
        props = (self.shapes.get(self.role('search') or '', None) or {}).get('properties', None) or {}
        page, tok = (self.page_arg() or (None,))[0], self.token_key()
        text_key = next((k for k, spec in props.items() if isinstance(spec, (dict,)) and 'string' == spec.get('type', None)
                         and not spec.get('enum', None) and not k in (page, tok)), None)
        if not text_key:
            return False
        seen = [set(re.findall(r"[a-z0-9]+", " ".join(self.free_text(w["args"])).lower())) for w in book.values()]
        rows = self.needs.items()
        plist = ([(None, str(self.needs.reading.get('product', None) or '').strip(None).lower())] if not rows else [(k, str(rows[k]['product']).strip(None).lower()) for k in range(0, len(rows))])
        pairs = []
        for k, product in plist:
            for req in self.needs.props_left():
                if rows and not self.product_of(req.partition(':')[0]) in (k, None):
                    continue
                pairs.append((k, product, req))
        expanded = []
        for k, product, req in pairs:
            raw = re.sub('\\((?:stated|revealed[^)]*)\\)', '', req.partition(':')[2]).strip(None).lower()
            alts = ([raw] if not (re.search('\\bor\\b|,', raw) and (not re.search(CH_BOUND, raw)) and (not re.match('^(?:not|no|non|without)\\b', raw))) else [a.strip(None) for a in re.split('\\s*(?:,|\\bor\\b)\\s*', raw) if a.strip(None)])
            expanded += [(k, product, a) for a in alts[:3]]
        for k, product, value in expanded:
            if not product:
                continue
            pwords = set(re.findall(r"[a-z0-9]+", product))
            words = re.findall(r"[a-z0-9]+", value)
            if (not words or 4 < len(words) or (self.amount_part(value) and not self.fig_amount(value)) or re.search(CH_BOUND, value)
                    or re.search(r"\bor\b|,|^(?:not|no|non|without)\b", value)):
                continue
            if not any((set(words) <= s and s <= set(words) | pwords for s in seen)):
                call_args = {k: v for k, v in next(iter(book.values()))['args'].items() if not k in (page, tok)}
                call_args[text_key] = value + ' ' + product
            else:
                both = next((w for w in book.values() if set(re.findall('[a-z0-9]+', ' '.join(self.free_text(w['args'])).lower())) == set(words) | pwords), None)
                if not both or 3 <= both['total'] or value + '|alone' in done or any((s == set(words) for s in seen)):
                    continue
                call_args = {k: v for k, v in both['args'].items() if not k in (page, tok)}
                call_args[text_key] = value
                done.add(value + '|alone')
                value = value + '|alone'
            done.add(value)
            proposal = (dict(id='vsearch_{}_{}'.format(len(done), self.wire.tick), type='function', function=dict(name=self.role('search'), arguments=json.dumps(call_args))))
            self.chat.append(dict(role='assistant', content='When a search starts with one required value, it returns every listing that mentions that value; the cheapest fit might be on any of its pages, so I run it.', tool_calls=[proposal]))
            self.covering = True
            try:
                responses = self.dispatch([proposal], scripted=True)
            finally:
                self.covering = False
            self.post_replies(responses)
            return True
        return False

    def value_walk_step(self):
        found = self.page_arg()
        if (not self.needs.priority() in ('cheapest', 'best') or self.orders_tried or not found or not self.role("search") or self.wire.done
                or self.wire.spare() < CLOSING + 60 or self.moves_left() <= self.order_due() + 4):
            return False
        used = self.__dict__.get("vwalk_pages", 0)
        if VWALK_PAGES <= used:
            return False
        best = None
        for k in (list(range(0, len(self.needs.items()))) or [None]):
          home = self.walk_home(k) or (set() if not k is None else set(self.shelf_labels()))
          for walk in self.__dict__.get("page_book", {}).values():
            nxt = self.next_page(walk)
            if not walk.get('id_order', None) is True or not walk.get('first', None) or nxt is None:
                continue
            carries = self.walk_carries(walk["args"], k)
            if not carries or (home and not set((walk.get('row_shelf', None) or {}).values()) & home):
                continue
            left = -(-walk["total"] // max(1, walk["size"])) - len(walk["read"])
            if left <= 0:
                continue
            rank = (-len(carries), left, json.dumps(walk['args'], default=str, sort_keys=True))
            if best is None or rank < best[0]:
                best = (rank, walk, nxt)
        if best is None:
            return False
        _, walk, nxt = best
        call_args = dict(walk["args"])
        call_args[found[0]] = nxt
        self.vwalk_pages = used + 1
        proposal = (dict(id='vwalk_{}_{}'.format(self.vwalk_pages, self.wire.tick), type='function', function=dict(name=self.role('search'), arguments=json.dumps(call_args))))
        self.chat.append(dict(role='assistant', content='Because the store returns results in a set order rather than sorted by price, the lowest-priced suitable item might be on any page of this search, so I go on to the following page and read it.', tool_calls=[proposal]))
        self.covering = True
        try:
            responses = self.dispatch([proposal], scripted=True)
        finally:
            self.covering = False
        if responses and self.next_page(walk) is None and 'best' == self.needs.priority():
            rows = sorted(self.shelf_rows().items(), key=lambda kv: (kv[1][0], kv[0]))[:10]
            if rows:
                rule = str(self.needs.reading.get('priority_detail', None) or "the request's rule").strip(None)
                cid, text_body = responses[-1]
                responses[-1] = (cid, text_body + '\n[search read to its end] Rows from all that has been seen that are in stock, in the requested category, and inside the budget: ' + "; ".join(f"{i} '{t[:80]}' {c:g}" for i, (c, t, _) in rows)
                                 + '. The rule is {}: read the full records of the listings whose titles could outdo your pick on that rule and test each against all requirements; among those that fit, the one best on the rule becomes the pick.'.format(rule))
        elif responses and self.next_page(walk) is None:
            rows = sorted(self.shelf_rows().items(), key=lambda kv: (kv[1][0], kv[0]))[:5]
            if rows:
                cid, text_body = responses[-1]
                responses[-1] = (cid, text_body + '\n[search read to its end] Of all the rows seen so far, these are the lowest-priced ones in stock for the requested category that fit the budget: ' + "; ".join(f"{i} '{t[:80]}' {c:g}" for i, (c, t, _) in rows)
                                 + ". Open the least expensive rows that haven't been opened yet and test each against all requirements: whichever is cheapest and fits becomes the pick.")
        self.post_replies(responses)
        return True

    def seed_args(self, label, bound):
        tool = self.role("search")
        jschema = self.shapes.get(tool or '', None) or {}
        props = {} if not isinstance(jschema.get('properties', None), (dict,)) else jschema.get('properties', None)
        found = self.page_arg()
        if not props or not found or not is_number(bound):
            return None
        told = next((str(tdef['function'].get('description', None) or "") for tdef in self.toolset if tdef["function"]["name"] == tool), "")
        parts = [part.lower() for part in re.split(r"[.;]", told)]
        tok = self.token_key()
        shelf_key, shelf_val = None, None
        for key, spec in props.items():
            if not isinstance(spec, (dict,)) or key in (tok, found[0]) or not re.search(r"categor|shelf|department|aisle", str(key).lower()):
                continue
            if isinstance(spec.get('enum', None), (list,)):
                hit = next((v for v in spec["enum"] if isinstance(v, (str,)) and v.strip(None).lower() == label.strip(None).lower()), None)
                if not hit is None:
                    shelf_key, shelf_val = key, hit
                    break
            elif 'string' == spec.get('type', None):
                shelf_key, shelf_val = key, label
                break
        bound_key = next((key for key, spec in props.items() if isinstance(spec, (dict,)) and spec.get('type', None) in ("integer", "number")
                          and not spec.get('enum', None) and key != found[0] and re.search(r"ceil|max|price|cap$|budget|upper|most", str(key).lower())
                          and not re.search(r"page|first|offset|skip|count|size|num", str(key).lower())), None)
        if not shelf_key or not bound_key:
            return None
        args: dict[str, Any] = {shelf_key: shelf_val, bound_key: bound if not ('integer' == props[bound_key].get('type', None) or float(bound).is_integer()) else int(bound)}
        for key, spec in props.items():
            if shelf_key == key or not isinstance(spec, (dict,)) or not isinstance(spec.get('enum', None), (list,)) or not 2 <= len(spec["enum"]) <= 4:
                continue
            hit = next((v for v in spec["enum"] if any(re.search(r"\b" + re.escape(str(v).lower()) + r"\b", part)
                                                       and re.search(r"\b" + re.escape(shelf_key.lower()) + r"\b", part) for part in parts)), None)
            if hit is None:
                return None
            args[key] = hit
        for key in jschema.get('required', None) or []:
            if not key in args:
                return None
        fixed, faults = ArgCheck.execute(args, jschema)
        return None if not (not faults and isinstance(fixed, (dict,))) else fixed

    def ceil_seed(self):
        try:
            if not self.ceil_spec() is None or self.__dict__.get('ceil_mode', None) or 'cheapest' != self.needs.priority():
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
                sig = json.dumps([sorted((m, v) for m, v in mode.items() if not (isinstance(args.get(m, None), (str,)) and args[m] == label)),
                                  next((a for a in args if is_number(args[a]) and a != (None if not found else found[0])), "")], default=str)
                key = json.dumps({a: v for a, v in args.items() if not found or a != found[0]}, default=str, sort_keys=True)
                if sig in self.__dict__.get("ceil_bad", set()) or key in (self.__dict__.get('ceil_seen', None) or {}):
                    continue
                self.__dict__.setdefault("ceil_seed_keys", set()).add(key)
                self.walk_call(args, f"The request is for the cheapest: I list '{label}' with prices under the pick ({pick[1]:g}) to check if this store's price filter works.")
                self.log.append(dict(kind='ceil_seed', product=k, args=args, learned=not self.ceil_spec() is None))
                if self.chat and 'tool' == self.chat[-1].get('role', None) and self.ceil_spec() is None:
                    self.chat[-1]["content"] = str(self.chat[-1]["content"]) + "\n" + self.status_line()
                return True
            return False
        except (Exception,) as err:
            self.log.append(dict(kind='ceil_seed_error', error='{}: {}'.format(type(err).__name__, err)[:300]))
            return False

    def unfound_product(self):
        if not self.needs.items():
            return False
        try:
            read = {s.strip(None).lower() for s in self.archive.shelves_read()}
            if any(not home or not home & read for home in self.home_shelves()):
                return True
            live = [k for k in self.passed if k.rpartition("|")[2] == str(self.needs.version)]
            return bool(live) and self.unverified()
        except (Exception,):
            return False

    def walk_hold(self, call_args):
        try:
            if 'cheapest' != self.needs.priority() or self.__dict__.get('walk_queue', None) or self.wire.spare() < CLOSING + 60 \
                    or (self.refusal_n and 15 >= self.moves_left()):
                return ""
            ids = list(self.trolley.units) if not not call_args is None else [i for i, c in self.trolley.lines(call_args) if c is None or c > 0]
            only = {self.product_index(i) for i in ids if i}
            floor = 4 if not self.orders_tried == 0 else FIRST_ORDER_BY + 1
            half = (self.moves_left() - floor - 1) // 2
            room = half - (2 if not self.batch_tool() else 1)
            ceil = []
            aims = self.__dict__.setdefault("ceil_aims", {})
            for k in sorted(only, key=lambda x: x if not x is None else -1):
                if room >= 2 and self.ceiling_ready(k):
                    ceil.append(k)
                    room -= CEIL_CALLS
                    costs = [float(p) for p in (self.archive.price(i) for i in ids if i and self.product_index(i) == k) if is_number(p)]
                    if costs:
                        aims[k] = min(costs + ([] if not is_number(aims.get(k, None)) else [float(aims[k])]))
            plan = self.walk_plan(min(WALK_TOTAL, room), only)
            if not plan and not ceil:
                return ""
            self.walk_queue = (plan, max(floor, self.moves_left() - half - 1), ceil)
            book = self.__dict__.get("page_book", {})
            queries = [" ".join(self.free_text(book[key]["args"])) for _, key, _ in plan if key in book]
            parts = ([] if not queries else ['reads the remaining pages of ' + ', '.join(("'{}'".format(q) for q in queries))]) + (
                [] if not ceil else ["gives the category's listings that cost less than the pick, using the store's price filter"])
            return ('Not run yet (no step used). Cheapest option requested, yet this store sorts search results by a fixed id sequence rather than by price, so a lower-priced match could be on pages that were never opened: the code ' + " and ".join(parts)
                    + ' now on your behalf and opens the lowest-priced unread rows priced below the chosen one. After that, submit the call once more, or submit it with a less expensive fit.')
        except (Exception,) as err:
            self.log.append(dict(kind='walk_hold_error', error='{}: {}'.format(type(err).__name__, err)[:300]))
            return ""

    def search_note(self, call_args, obs):
        if not isinstance(obs, (dict,)):
            return ""

        def rows_of(piece):
            lists = [val for val in piece.values() if isinstance(val, (list,)) and all(isinstance(r, (dict,)) for r in val)]
            return next((val for val in lists if val), None if not lists else lists[0])
        rows = rows_of(obs)
        if rows is None:
            obs = next((val for val in obs.values() if isinstance(val, (dict,)) and not rows_of(val) is None), None)
            if obs is None:
                return ""
            rows = rows_of(obs) or []
        totals = [val for val in obs.values() if isinstance(val, (int,)) and not isinstance(val, (bool,)) and val > len(rows)]
        self.book_page(call_args, rows, max(totals or [len(rows)]), obs)
        try:
            self.ceil_learn(call_args, rows, max(totals or [len(rows)]), obs)
        except (Exception,) as err:
            self.log.append(dict(kind='ceil_learn_error', error='{}: {}'.format(type(err).__name__, err)[:300]))
        score = 0 if not rows else self.value_score(call_args)
        if score:
            lead_rows = self.__dict__.setdefault("row_score", {})
            for r in rows:
                if isinstance(r, (dict,)):
                    lead_rows[self.row_mark(r)] = max(lead_rows.get(self.row_mark(r), 0), score)
        remarks = []
        kind = self.word_rule(call_args, obs, rows)
        lead = self.lead_rule(call_args, obs, rows)
        if lead:
            texts, mode, _, _ = self.search_keys(call_args)
            words = () if not mode == self.lead_ctx[1] else next((w for key, w in texts if key == self.lead_ctx[0]), ())
            counted = '' if not len(words) > lead else ' (this query counted only: {})'.format(' '.join(words[:lead]))
            remarks.append("This store's keyword search ({}) only looks at the first FIRST {} word(s) of a query and disregards whatever follows{}: begin with the {} most distinctive words (a required value and the kind of item, with the value coming first), and look up any other values using their own separate short queries.".format(self.lead_ctx[0], lead, counted, lead))
        texts, mode, first, paging_keys = self.search_keys(call_args)
        raw = [val for key, val in call_args.items() if isinstance(val, (str,)) and not key in dict(mode) and not key in paging_keys]
        if first and raw:
            pieces = [p for t in raw for p in re.findall(r"[a-z0-9]+", t.lower())]
            if pieces and all(1 == len(p) for p in pieces) and not rows and not self.__dict__.get('short_told', None):
                self.short_told = True
                remarks.append("Nothing came back for this query: the store's search ignores words of a single character, so a value such as 0.4 or 5 x 7 yields no search at all (or only searches on the remaining words). Try a value written as a word, or pair the unit word with the kind of item, and go through all pages.")
            prop, probe = self.value_probe(raw)
            told = self.__dict__.get("name_told", 0)
            if probe and told < 2 and self.moves_left() > self.order_due() + 3:
                self.name_told = told + 1
                remarks.append((("Again led by a property's name ({}).".format(prop) if told else "This query begins with words from a property's NAME ({}) and includes no required value: keyword search compares against listing values, titles and categories, and only matches a field's name when some value or title echoes it, so listings whose {} field contains only the value get missed.".format(prop, prop)))
                               + (" Search the value first, then the item kind: '{}' (a figure the store cannot read drops out, leaving its unit word), and read every page of it.".format(probe)))
            elif not probe and 2 > self.__dict__.get('trap_told', 0) and self.moves_left() > self.order_due() + 3:
                trap, _ = self.word_trap(raw)
                if trap:
                    self.trap_told = self.__dict__.get("trap_told", 0) + 1
                    remarks.append(trap)
            marks = tuple(self.row_mark(r) for r in rows)
            others = tuple(sorted((k2, str(v2)) for k2, v2 in call_args.items() if not k2 in paging_keys and not (isinstance(v2, (str,)) and not k2 in dict(mode))))
            sig = (mode, others, max(totals or [len(rows)]), marks)
            book = self.__dict__.setdefault("page_sigs", {})
            query = " ".join(raw)
            if 3 <= len(set(marks)) and sig in book and book[sig] != query:
                remarks.append("Your earlier search '{}' produced the identical total and rows: the store treated the two as the same, so spelling it differently reveals nothing new. Go to its next page, or alter the first words.".format(book[sig]))
            book.setdefault(sig, query)
            hits = []
            told = bool((kind == 'all' or self.lead_words) and not totals and rows)
            if told or (kind != 'any' and not totals and rows):
                held = list(self.__dict__.get('ref_hits', None) or []) if not told else None
                self.ref_hits = []
                hits = self.row_hits(rows, raw)
                score = self.value_score(call_args)
                ranks = self.__dict__.setdefault("hit_score", {})
                ranks.update({i: max(ranks.get(i, 0), score) for i in hits})
                found = self.__dict__.setdefault("all_hits", [])
                found += [i for i in hits if not i in found]
                if not held is None:
                    self.ref_hits = held
                if told and hits and len(texts) and 12 >= len(rows):
                    remarks.append('All words are mandatory in this case and every match is displayed: each row has the words that the store found anywhere in its record, frequently in fields that the title leaves out. Before searching further, open the ones that fit the budget: ' + ", ".join(hits[:6]) + ".")
                pointers = [] if not told else [i for i in self.__dict__.pop('ref_hits', []) if not i in hits][:6]
                if pointers:
                    remarks.append('Here the request supplies a value by referring to a different listing, and the shop or brand named in this search is the seller of these rows: open them regardless of price, stock or category (a listing providing a value is only read, never purchased), and act on what its notes say: ' + ", ".join(pointers) + ".")
            if (kind == 'all' or self.lead_words) and not self.__dict__.get('hits_back', None):
                self.hits_back = True
                here = json.dumps({k: v for k, v in call_args.items() if not k in paging_keys}, default=str, sort_keys=True)
                back: list[str] = []
                for walk_key, walk in self.__dict__.get("page_book", {}).items():
                    if here != walk_key and walk.get('every', None) and self.value_score(walk["args"]):
                        fresh = [i for i in self.row_hits(walk["every"]) if not i in back and not i in hits]
                        ranks = self.__dict__.setdefault("hit_score", {})
                        ranks.update({i: max(ranks.get(i, 0), self.value_score(walk["args"])) for i in fresh})
                        back += fresh
                found = self.__dict__.setdefault("all_hits", [])
                found += [i for i in back if not i in found]
                if back:
                    remarks.append("Every word must match in this store's keyword search, so previous searches that displayed all matches only listed rows containing all of their words, frequently in fields the titles leave out. Before searching further, open the within-budget ones that haven't been read yet: " + ", ".join(back[:6]) + ".")
        if not rows:
            return " ".join(remarks)
        if totals:
            if kind == 'any':
                how = ('the keyword search in this store uses ANY of the words and puts listings that match more of them at the top, so each extra word broadens the results: a single distinctive word or figure from a required value returns just the listings that include it (frequently few enough that you can read all the pages and compare prices); combining the item kind with the values puts the nearest matches first; for everything else, switch to the category mode (or apply a price filter if one is relevant).')
            elif kind == 'all':
                how = ('keyword search in this store requires EVERY word, so adding more exact words narrows results, and so do the category filter and a price filter with a lower price cap.')
            else:
                how = ('after that, cut it down for the remainder: if every word has to match, add exact words; if matching any word is enough (which you can tell when a longer query returns more listings), include the needed values in the query so that fitting listings come first; a price filter, if one applies, helps too.')
            remarks.append('The search hit {} listings, but this page lists just {}, and only a handful of pages are readable, so relevant listings might be hidden among those not shown. First go through the later pages of this very search (as many as you are able to); if the total exceeds what those pages cover, '.format(max(totals), len(rows)) + how)
        paging = self.search_keys(call_args)[3]
        limits = [val for key, val in call_args.items() if is_number(val) and val > 1 and not key in paging
                  and not re.search(r"page|size|count|result|min|low|from|start", str(key).lower())]
        if limits:
            cap = max(limits)
            over = [r for r in rows if any(is_number(val) and cap < val and "price" in str(key).lower() for key, val in r.items())]
            if over:
                remarks.append('{} of these rows are priced above the {} limit set in this call: the search mode in use disregarded that filter (and could be disregarding others too, like the category). Switch to a mode that respects its filters, or verify each row on your own.'.format(len(over), cap))
        return " ".join(remarks)

    def shopper_lines(self, call_name, obs, refused):
        lns: list[str] = []
        if call_name == self.role("message"):
            if isinstance(obs, (str,)):
                lns = [obs]
            elif isinstance(obs, (dict,)) and not 'error' in obs:
                def visit(d):
                    long_strings([val for val in d.values() if isinstance(val, (str, list))], lns)
                walk_dicts(obs, visit, ceiling=4)
        elif isinstance(obs, (dict,)):
            lns = self.notice_lines(obs, refused)
        return list(dict.fromkeys(lns))

    @staticmethod
    def notice_lines(obs, refused = False):
        lns: list[str] = []
        if not isinstance(obs, (dict,)):
            return lns

        def visit(d):
            if not d is obs and 6 < len(d):
                return
            for piece in d.values():
                if isinstance(piece, (list,)) and piece and all(isinstance(val, (str,)) for val in piece):
                    long_strings(piece, lns)
                elif isinstance(piece, (str,)) and 4 <= len(piece.split()) and (refused or (3 >= len(d) and 600 >= len(piece) and not any(
                        isinstance(v, (str,)) and "id" in str(k).lower() and re.fullmatch(ID_LIKE, v) for k, v in d.items()))):
                    lns.append(piece)
        walk_dicts(obs, visit, ceiling=4)
        return list(dict.fromkeys(lns))

    def screen(self, proposal, permitted, scripted, batch, queued_messages):
        func = proposal.get('function', None) or {}
        call_name, raw_args = func.get('name', None), func.get('arguments', None)
        if not call_name in self.shapes:
            return "Not run (no step used): there is no tool called '{}'. Pick from these: {}.".format(call_name, ', '.join(sorted(permitted))), call_name, {}
        if not scripted and not call_name in permitted:
            return 'Not run (no step used): the remaining step count is too low for this, so for now you may call only these tools: ' + ", ".join(sorted(permitted)) + ".", call_name, {}
        call_args = ({} if not (raw_args or '').strip(None) else json_object(raw_args)) if not isinstance(raw_args, (dict,)) else raw_args
        if call_args is None:
            return 'Not run (no step used): the arguments passed in were not a well-formed JSON object.', call_name, {}
        call_args, faults = ArgCheck.execute(call_args, self.shapes[call_name])
        if faults:
            return 'Not run (no step used): the arguments were not valid: ' + "; ".join(faults[:6]), call_name, call_args
        if call_name in self.shapes and isinstance(call_args, (dict,)):
            fixed = self.shelf_fix(call_name, call_args)
            if not fixed is None:
                call_args = fixed
        if call_name == self.role("search"):
            tok = self.token_key()
            given = None if not tok else call_args.get(tok, None)
            if isinstance(given, (str,)) and given.strip(None):
                book = self.__dict__.get("page_book", {})
                known = {str(v) for w in book.values() for k, v in (w.get('tokens', None) or {}).items() if v}
                if known and not given.strip(None) in known:
                    base = {k: v for k, v in call_args.items() if tok != k}
                    walk = book.get(json.dumps(base, default=str, sort_keys=True), None)
                    nxt = None if not walk else self.next_page(walk)
                    self.log.append(dict(kind='token_fix', given=given[:40], to=str(nxt)[:40]))
                    if nxt:
                        call_args = dict(call_args, **{tok: nxt})
                    elif walk:
                        return ('Not run (no step used): the store never issued that page token, and all pages of this search that it did return have been read already. A next-page token must be copied exactly the way a result provides it.'), call_name, call_args
                    else:
                        call_args = base
        cap = (self.__dict__.get('id_caps', None) or {}).get(call_name, None)
        if cap and any(isinstance(v, (list,)) and len(v) > cap for v in call_args.values()):
            return (('Not run (no step used): {} answers at most {} ids per call in this store (a longer list came back as an error), so send at most {} ids per call.'.format(call_name, cap, cap))), call_name, call_args
        if call_name in self.finishers() and batch > 1:
            return 'Not run (no step used): you must send an order or a decline by itself, once the results from your other calls have come back.', call_name, call_args
        if call_name == self.role("message") and self.trolley.units and self.basket_answered(call_args):
            return ("Not run (no step used): since the last change to the cart, the shopper has already been shown this basket, was asked to approve it, and voiced no objection - that response counts as approval. Don't ask once more: place the order now."), call_name, call_args
        if call_name == self.role("message") and not self.msg_cap is None and self.msgs_sent + queued_messages >= self.msg_cap:
            return 'Not run (no step used): you have no messages left in the budget. Do not send the shopper any more messages.', call_name, call_args
        if call_name == self.role("message") and not scripted and self.msg_reserve() \
                and self.msgs_sent + queued_messages >= self.ask_room():
            return ('Not run (no step used): the remaining message is saved for the closing step described in the instructions; it goes out during the final check when you order or decline. Send the shopper nothing further: complete the cart, then order (or decline).'), call_name, call_args
        if call_name == self.role("message") and not self.trolley.units and self.moves_left() > self.order_due() + 6 \
                and not str(proposal.get('id', None) or "").startswith(("ask_", "basket_", "reply")) \
                and not (self.ask_first_due() and not self.questions_log) \
                and self.archive.rich_reads() < ASK_AFTER_SETS and any("?" in v for v in call_args.values() if isinstance(v, (str,))):
            return ("Not run (no step used): questions for the shopper are held back until you have read the details of a few listings, so that each question refers to a field the store's listings actually have; the question plan sends them at that point. Begin by searching and reading listings."), call_name, call_args
        if not scripted and call_name == self.role("cart_view") and call_name != self.role("cart") and not self.__dict__.get('batch_cart', None) \
                and self.trolley.units and self.__dict__.get("cart_shown", -1) == self.__dict__.get("cart_ver", -2) \
                and not self.__dict__.get("since_cart", 1):
            told = self.__dict__.setdefault("view_told", set())
            if not self.cart_shown in told:
                told.add(self.cart_shown)
                return ('Not run (no step used): the basket is already visible in the most recent cart result - ' + self.trolley.summary()
                        + " Viewing the cart again reveals nothing new now; to verify a line's stock again, open its record using "
                        + (self.stock_tool() or "the tool that shows stock") + "."), call_name, call_args
        sig = call_name + as_text(call_args)
        if not call_name in self.finishers() and not call_name in (self.role('message'), self.role('cart')) and sig in self.last_calls:
            return 'Not run (no step used): this identical call was made a moment ago and nothing is different now; the result appears above. Modify the arguments or take a different action.', call_name, call_args
        held = self.late_hold(call_name, call_args) if not scripted else ''
        held = held or (self.value_guard(call_name, call_args) if not scripted else '')
        if held:
            return held, call_name, call_args
        if call_name == self.role("cart"):
            refusal = self.screen_cart(call_args, sig)
            if refusal:
                return refusal, call_name, call_args
        return None, call_name, call_args

    def screen_cart(self, call_args, sig):
        thing = self.trolley.aim(call_args)
        point = '' if not (thing and self.trolley.is_put(call_args)) else self.ref_pointer_note(thing)
        pointed = self.__dict__.setdefault("pointer_told", {})
        if point and 2 > pointed.get(thing, 0):
            pointed[thing] = pointed.get(thing, 0) + 1
            return 'Not run (no step used): {}. Pick a different listing.'.format(point)
        mark = '{}#{}'.format(sig, self.__dict__.get('puts', {}).get(thing, 0))
        mine = None if not thing else self.archive.price(thing)
        cheaper_put = any(is_number(mine) and is_number(self.archive.price(p)) and self.archive.price(p) < mine
                          for p in self.__dict__.get("batch_puts", []) if thing != p)
        swap = 'cheapest' == self.needs.priority() and (cheaper_put or bool(self.cheaper_fit(thing)))
        moved = thing in self.__dict__.get("pass_price", {}) and self.__dict__["pass_price"][thing] != mine
        if thing and not self.trolley.is_put(call_args) and '{}|{}'.format(thing, self.needs.version) in self.passed and not swap and not moved \
                and self.refusal_n == self.refusals_seen.get(thing, self.refusal_n) and not mark in self.warned:
            self.warned.add(mark)
            least = '' if not ('cheapest' == self.needs.priority() and (not self.cheaper_fit(thing))) else ' It is also the cheapest verified fit.'
            return ('Not run (no step used): a separate check determined that {} satisfies all requirements currently active, with each property read according to the value rules (when this store spells out the value in a field, only those exact words count); the category label alone settles what kind of item it is, regardless of which object its title or description mentions.{} Nothing is different since then. Resend the identical call if a change or a refusal excludes it, if one of its fields plainly fails a requirement, or if you are swapping it for a cheaper listing.'.format(thing, least))
        if not self.trolley.is_put(call_args) or not thing:
            return ""
        if 0 < self.trolley.held(thing) and not self.trolley.adds is False and not sig in self.warned:
            self.warned.add(sig)
            return ("Not run (no step used): {} x{} is already in the cart, and since this cart tool increases the units on a line that exists, running this call would bump it higher. If the goal is to buy it, don't touch the cart; if the unit count needs fixing, add or remove just the gap. When adding is truly the intent, repeat the identical call.".format(thing, int(self.trolley.held(thing))))
        walk = self.walk_hold(call_args) if not point else ''
        if walk:
            return walk
        for other, count in self.trolley.lines(call_args)[1:]:
            if (count is None or count > 0) and 3 < self.moves_left() and not '{}|{}'.format(other, self.needs.version) in self.checked_adds:
                self.checked_adds.add('{}|{}'.format(other, self.needs.version))
                ruling = self.check_item(other, "ADD TO THE CART")
                if ruling:
                    return "Not run (no step used). " + ruling + ' Once you are sure it fits, repeat the identical call.'
        if 3 < self.moves_left() and not '{}|{}'.format(thing, self.needs.version) in self.checked_adds:
            self.checked_adds.add('{}|{}'.format(thing, self.needs.version))
            ruling = self.check_item(thing, "ADD TO THE CART")
            if ruling:
                return "Not run (no step used). " + ruling + ' If you are sure it is a fit, make the same call once more.'
        for line_id, count in [] if not 3 < self.moves_left() else self.trolley.lines(call_args):
            if not count is None and count <= 0:
                continue
            better = self.rank_check(line_id)
            if better:
                cheaper = self.rank_verify(line_id)
                return "Not run (no step used). " + better.replace("before ordering", "before putting it in the cart").replace(
                    "order the current pick", "send the same call again") + (
                    '' if not cheaper else ' The separate item check showed that {} satisfies all the requirements currently in force and is cheaper than {}, so swap in {} instead.'.format(cheaper, line_id, cheaper))
        for line_id, count in [] if not 'cheapest' == self.needs.priority() else self.trolley.lines(call_args):
            if not count is None and count <= 0:
                continue
            cheaper = self.cheaper_fit(line_id)
            told = self.__dict__.setdefault("cheaper_told", {})
            if cheaper and not cheaper in self.trolley.units and not told.get(('cart', line_id, cheaper), None):
                told[("cart", line_id, cheaper)] = 1
                return ("Not run (no step used). According to the independent item check, {} satisfies all the requirements currently in force and is priced below {}. Since the lowest-priced fit is what's wanted, and the check's judgment outranks yours, swap in {} instead. Should a record show {} failing any requirement currently in force, repeat the identical call.".format(cheaper, line_id, cheaper, cheaper))
        need_n = self.needs.quantity()
        count = self.trolley.count_arg(call_args)
        aim = need_n if not not self.trolley.adds is False else need_n - self.trolley.held(thing)
        if self.needs.items():
            odd = [(i, int(c)) for i, c in self.trolley.lines(call_args) if not c is None and c > 0 and not int(c) in self.needs.counts()]
            if odd and not sig in self.warned:
                self.warned.add(sig)
                return ("Not run (no step used): the requested products call for {} units (a single line apiece), yet this call requests {}. If you're certain, repeat the identical call.".format(self.needs.counts(), odd))
            return ""
        if not count is None and 1e-09 < abs(count - aim) and not sig in self.warned:
            self.warned.add(sig)
            return "Not run (no step used): what was asked for is a quantity of {} for a single listing, yet this call specifies {}. When you're certain, repeat the identical call.".format(need_n, int(count))
        return ""

    def nofit_vet(self, call_args):
        near_note = self.near_hold()
        if near_note:
            return near_note, call_args
        passed_lines = [i for i in self.trolley.units if '{}|{}'.format(i, self.needs.version) in self.passed and not i in self.refused_lines
                        and not i in self.dead_lines()]
        if passed_lines and 1 < self.moves_left() and not self.__dict__.get('decline_kept', None):
            self.decline_kept = True
            return ("Not run yet (no step used). The cart holds " + ", ".join(passed_lines) + ", which the independent check "
                    "found meets every requirement now in force. A decline would throw that away: re-read its stock and price and "
                    "order it" + ('' if not self.needs.items() else " with every product's line in the cart") + '. Refuse only when something that changed since then plainly makes it impossible.'), call_args
        slack = [] if not (not self.__dict__.get('slack_done', None) and (self.role('details') or self.batch_tool())) else self.slack_rows()
        spread = self.batch_tool()
        reads = len(slack) if not (spread and 2 <= len(slack)) else -(-len(slack) // spread[2])
        if slack and self.moves_left() > reads + 4 and self.wire.spare() > CLOSING + 40:
            self.slack_done = True
            self.open_queue = slack
            self.open_say = 'Prior to refusing, I check cheaper rows that could allow all the products to fit within the budget together.'
            total = self.__dict__.get('slack_total', None)
            return ('Not run yet (no step used). A suitable listing exists for each product, but the lowest-cost combination of them totals '
                    + ('more than the budget' if not is_number(total) else f'{total:g}') + ", exceeding the {} budget. Rows whose prices fit within what remains for each product once the other products' cheapest fits are subtracted were never opened, so you now get their records read out: ".format(self.needs.spend_cap) + ", ".join(slack) + '. Evaluate every one against the requirements of its own product using the value rules (if this store spells the value out in a field, only that wording counts as a match); once the options that fit the budget jointly cover all products, add a line for each product to the cart and order; refuse only when no such combination is available.'), call_args
        batch = self.batch_tool()
        fill = [] if not (not self.__dict__.get('decline_fill', None) and (self.role('details') or batch)) else self.lacking_rows()
        reads = len(fill) if not (batch and 2 <= len(fill)) else -(-len(fill) // batch[2])
        if fill and self.moves_left() > reads + 3 and self.wire.spare() > CLOSING + 40:
            self.decline_fill = True
            self.open_queue = fill
            self.open_say = 'Prior to refusing, I check listings not yet read for the products lacking a verified fit.'
            return ('Not run yet (no step used). A few requested products still lack a confirmed fit, and the in-budget listings for their categories were never looked at, so their records are now fetched for you ahead of any refusal: ' + ", ".join(fill) + '. Evaluate every one against the requirements of its own product using the value rules (if this store spells the value out in a field, only that wording counts as a match); once each product has a listing that fits, add a line for every product to the cart and order; refuse only when some product really has no fit.'), call_args
        queue = self.named_unread()[: 3 if not batch else batch[2]]
        reads = len(queue) if not (batch and 2 <= len(queue)) else -(-len(queue) // batch[2])
        if queue and not self.__dict__.get('decline_opens', None) and (self.role("details") or batch) and self.moves_left() > reads + 3 \
                and self.wire.spare() > CLOSING + 40:
            self.decline_opens = True
            self.open_queue = queue
            return ('Not run yet (no step used). The checks pointed to listings that were never looked at, so their records are now fetched for you ahead of any refusal: ' + ", ".join(queue) + '. Evaluate every one against the requirements using the value rules (the field named after the property decides, and if this store spells the value out there, only that wording counts as a match; if there is no such field, a fallback such as the options list or the title can reveal the value), purchase one that fits, and refuse only when all of them fail.'), call_args
        pages_note = self.decline_reads()
        if pages_note:
            return "Not run yet (no step used). " + pages_note, call_args
        reread = len(self.archive.read) > self.__dict__.get("audit_reads", 10 ** 9) and 2 > self.__dict__.get('audits', 0)
        if reread and self.decline_vetted and 3 < self.moves_left() and self.wire.spare() > CLOSING + 30:
            audit_note = self.review_decline()
            if audit_note:
                return "Not run yet (no step used). " + audit_note, call_args
        if not self.decline_vetted and 3 < self.moves_left() and self.wire.spare() > CLOSING + 30:
            self.decline_vetted = True
            audit_note = self.review_decline()
            if audit_note:
                return "Not run yet (no step used). " + audit_note, call_args
            return ("Not run yet (no step used). Prior to refusing, confirm that you ran searches in multiple ways (varied wording, filters, result pages), inspected the properties of each candidate yourself, and found that every one falls short on a stated requirement, a need the shopper revealed, the budget, or availability. A property's field counts under any of its key spellings. If an item does suit, purchase it rather than declining; if none suits, invoke the decline tool once more."), call_args
        fit_note = self.fit_hold()
        if fit_note:
            return "Not run yet (no step used). " + fit_note, call_args
        best = [] if not (not self.__dict__.get('best_held', None) and 4 < self.moves_left()) else self.unrefuted()
        for _ in range(0, 3):
            unchecked = [i for i in best if not '{}|{}'.format(i, self.needs.version) in self.passed]
            if not unchecked or self.wire.spare() < CLOSING + 30:
                break
            self.check_item(unchecked[0], "ORDER")
            best = self.unrefuted()
        if best and self.moves_left() > len(best) + 3 and self.wire.spare() > CLOSING + 20:
            self.best_held = True
            several = bool(self.needs.items())
            return ('Not run yet (no step used). Refusing is correct only if none of the listings can satisfy the requirements, and the records contain ' + ('a candidate' if not several else 'a candidate for every product asked for') + ' which is in stock, fits the budget, and was not excluded by any check: ' + ", ".join(best) + ". If a record doesn't show any of them failing a requirement currently in force, add " + (('it' if not several else 'a single line per product, with its quantity,')) + " in the cart, re-read stock and price, and order; decline only if each candidate clearly "
                    "fails."), call_args
        soft = [r for r in self.__dict__.get('refs', None) or [] if 'differ' == r.get('relation', None) and r.get('status', None) in ("open", "stuck")
                and self.ref_live(r)]
        if soft and 3 < self.moves_left() and not self.__dict__.get('ref_decline_held', None):
            self.ref_decline_held = True
            heads = ", ".join(dict.fromkeys(r["head"] for r in soft))
            return ("Not run yet (no step used). " + heads + " comes from a different listing that the code was unable to resolve, so nothing is known about its value: it can't exclude a listing and can't be a reason to decline. Add to the cart the listing that satisfies all remaining requirements (available in stock, inside the budget)" + ('' if not self.needs.items() else ', one line per product')
                    + ', check its stock and price again, then place the order; refuse only if none of the listings satisfies the remaining requirements.'), call_args
        repair = self.decline_repair()
        if repair:
            return repair, call_args
        near = self.near_check()
        if near:
            return near, call_args
        fits = self.passed_fits()
        if fits and not self.__dict__.get('fit_decline_held', None) and 3 < self.moves_left():
            self.fit_decline_held = True
            return ('Not run yet (no step used). The item check succeeded ' + ", ".join(fits[:3]) + ' given the requirements currently in force: declining is mistaken as long as a verified fit exists. Add ' + fits[0] + ' (the lowest-priced one) in the cart, check its stock and price again, then place the order for it.'), call_args
        return None

    def screen_finish(self, cid, call_name, call_args):
        if 'order' == self.call_kind(call_name, call_args) and 3 < self.moves_left() and not self.needs.items() \
                and 2 <= len(self.trolley.units):
            keep = self.chosen()
            mark = (tuple(sorted(self.trolley.units)), self.__dict__.get("cart_seq", 0))
            told = self.__dict__.setdefault("multi_told", {})
            if 1 > told.get(mark, 0):
                told[mark] = 1
                return ('Not run yet (no step used). Only one product was requested, yet the cart contains {} distinct listings ({}): retain just the one that is suitable ({} was added most recently), remove the rest, check its stock and price again, and then place the order.'.format(len(self.trolley.units), ', '.join(self.trolley.units), keep)), call_args
            if 1 == told.get(mark, None) and self.role("cart") and keep and 5 <= self.moves_left():
                told[mark] = 2
                was = self.wire.mute
                try:
                    self.wire.mute = True
                    bad = self.code_put([(keep, self.trolley.units.get(keep, None) or max(1, int(self.needs.quantity() or 1)))])
                except (Exception,) as err:
                    bad = None
                    self.log.append(dict(kind='multi_line_error', error='{}: {}'.format(type(err).__name__, err)[:200]))
                finally:
                    self.wire.mute = was
                self.log.append(dict(kind='multi_line_fix', keep=keep, bad=bad, cart=dict(self.trolley.units)))
                if bad == [] and list(self.trolley.units) == [keep]:
                    return ('Not run yet (no step used). Several listings for a single product were in the cart: the code retained {} and removed the rest: '.format(keep) + self.trolley.summary() + " Order it now."), call_args
        if 'order' == self.call_kind(call_name, call_args) and 2 < self.moves_left():
            dead = sorted(set(self.trolley.units) & self.dead_lines())
            mark = (tuple(dead), self.__dict__.get("cart_seq", 0))
            told = self.__dict__.setdefault("dead_told", {})
            if dead and 2 > told.get(mark, 0):
                told[mark] = told.get(mark, 0) + 1
                return "Not run yet (no step used). " + self.dead_note(dead), call_args
            if dead and 2 == told.get(mark, 0) and self.role("cart") and 6 <= self.moves_left() and self.wire.spare() > CLOSING:
                told[mark] = 3
                was = self.wire.mute
                try:
                    done = self.code_cart(set(dead) | self.dead_lines(), set(self.trolley.units) - set(dead))
                except (Exception,) as err:
                    done = False
                    self.log.append(dict(kind='dead_swap_error', error='{}: {}'.format(type(err).__name__, err)[:300]))
                finally:
                    self.wire.mute = was
                self.log.append(dict(kind='dead_swap', lines=dead, done=done, cart=dict(self.trolley.units)))
                if done and self.trolley.units and not set(self.trolley.units) & self.dead_lines():
                    return ('Not run yet (no step used). The cart continued to contain ' + ", ".join(dead) + ', which is out of stock, so the code removed it and swapped in a candidate, leaving the remaining lines untouched, and checked the stock of those lines: '
                            + self.trolley.summary() + ' Compare this cart with what was asked for, and then place the order.'), call_args
        if 'order' == self.call_kind(call_name, call_args) and 3 < self.moves_left():
            thing = self.chosen()
            cheaper = '' if not thing else self.cheaper_fit(thing)
            for line in ([i for i in self.trolley.units if thing != i] if not cheaper else []):
                cheaper = self.cheaper_fit(line)
                if cheaper:
                    thing = line
                    break
            told = self.__dict__.setdefault("cheaper_told", {})
            if cheaper and 2 > told.get((thing, cheaper), 0):
                told[(thing, cheaper)] = told.get((thing, cheaper), 0) + 1
                return ("Not run yet (no step used). The separate item check established that {} satisfies all requirements and is priced below {}. Since the cheapest fitting item is wanted, and the check's result outranks your own reading: remove {} from the cart, add {}, check its stock and price again, and order it.".format(cheaper, thing, thing, cheaper)), call_args
        if 'order' == self.call_kind(call_name, call_args) and self.trolley.units:
            held_unk = self.unknown_hold()
            if held_unk:
                return held_unk, call_args
        if 'order' == self.call_kind(call_name, call_args) and self.trolley.units and self.ladder_due() \
                and 2 > self.__dict__.get('ladder_held', 0):
            self.ladder_held = self.__dict__.get("ladder_held", 0) + 1
            return ("Not run yet (no step used). What's wanted is the lowest-priced fit: the code begins by listing the category with the store's price filter placed just under the lowest-priced fit found so far, then opens any cheaper listing that appears. After that, submit the order again, or submit the same order using a cheaper fit."), call_args
        if 'order' == self.call_kind(call_name, call_args) and self.trolley.units:
            walk = self.walk_hold(None)
            if walk:
                return walk, call_args
        reader = None if not 'order' == self.call_kind(call_name, call_args) else self.stock_tool()
        if reader and self.moves_left() > len(self.trolley.units) + 2 and self.wire.spare() > CLOSING:
            stale = self.stale_lines()
            mark = (tuple(sorted(self.trolley.units)), self.__dict__.get("cart_seq", 0))
            if stale and self.__dict__.get('recheck_mark', None) != mark:
                self.recheck_mark = mark
                self.open_queue = stale[:4]
                self.open_recheck = True
                self.open_purpose = "cart"
                self.open_tool = reader
                return ('Not run yet (no step used). Prices and availability may shift with no warning, and none of these cart lines have been re-checked since the cart was last modified: ' + ", ".join(stale[:4]) + '. The system now reads their stock on your behalf ({}, the stock-viewing tool); place the order if no line has gone out of stock and the sum stays inside the budget, otherwise swap out that line.'.format(reader)), call_args
        if 'order' == self.call_kind(call_name, call_args):
            try:
                cheap = self.cheap_sweep()
            except (Exception,) as err:
                cheap = ""
                self.log.append(dict(kind='cheap_sweep_error', error='{}: {}'.format(type(err).__name__, err)[:300]))
            if cheap:
                return cheap, call_args
        if 'order' == self.call_kind(call_name, call_args) and 5 < self.moves_left() and self.wire.spare() > CLOSING:
            try:
                deps = self.dependency_faults()
            except (Exception,):
                deps = []
            mark = (tuple(sorted(self.trolley.units)), tuple(deps))
            told = self.__dict__.setdefault("dep_told", {})
            if deps and 2 > told.get(mark, 0) and 4 > len(told):
                told[mark] = told.get(mark, 0) + 1
                return ('Not run yet (no step used). According to the request, one product has to be compatible with another: ' + "; ".join(deps)
                        + '. For the dependent product, run a search using each identifier as the query, open what comes back, and add one whose page lists any of them (or pick a different anchor whose identifiers appear on a dependent page); after that, order.'), call_args
        if 'order' == self.call_kind(call_name, call_args) and not self.order_vetted and 2 < self.moves_left() and self.wire.spare() > CLOSING:
            self.order_vetted = True
            faults = self.trolley.faults(self.needs.quantity(), self.needs.counts())
            if faults and 3 < self.moves_left():
                return 'Not run yet (no step used). The cart, going by its count, is not ready to go: ' + "; ".join(faults) + ". Fix it, then order.", call_args
            lines = ([] if not self.chosen() else [self.chosen()]) if not self.needs.items() else list(self.trolley.units)
            for thing in lines:
                ruling = self.check_item(thing, "ORDER")
                if ruling:
                    return "Not run yet (no step used). " + ruling, call_args
            for thing in lines:
                better = self.rank_check(thing)
                if better:
                    cheaper = self.rank_verify(thing)
                    return "Not run yet (no step used). " + better + (
                        '' if not cheaper else ' The separate item check determined that {} satisfies all requirements currently applying and is cheaper than {}: remove {}, add {}, check its stock and price again, then place the order.'.format(cheaper, thing, thing, cheaper)), call_args
            return ('Not run yet (no step used). Check before ordering: (1) the cart contains exactly one line ' + ('the selected item, in the quantity that is needed' if not self.needs.items() else 'for each requested product, at the quantity given for it') + ", and nothing more; (2) once your final cart change was made, you checked this item's stock and price using a tool that displays them, and it is available and price multiplied by quantity fits the current budget; (3) its own properties satisfy each stated requirement, each need the shopper disclosed and each announced change; (4) when the cheapest or best was requested, nothing better that fits exists among all you were able to find; (5) every procedural rule in the request was obeyed. If something remains unchecked, check it now; if not, call the order tool again."), call_args
        if 'order' == self.call_kind(call_name, call_args) and 'order' == self.__dict__.get('no_fit', None) and self.trolley.units \
                and self.__dict__.get('closest_vetted', None) != self.needs.version and 3 < self.moves_left() and self.wire.spare() > LAST_CALL + 20:
            loose = [i for i in self.trolley.units if not '{}|{}'.format(i, self.needs.version) in self.passed]
            said = " ".join(str(v) for v in (call_args or {}).values() if isinstance(v, (str,)))
            if not loose and not self.cart_short() and NO_FIT_SAID.search(said) and self.__dict__.get('fit_text_held', None) != self.needs.version:
                self.fit_text_held = self.needs.version
                return ("Not run yet (no step used). Every line in the cart passed the item check against the requirements in force, so "
                        "this order is a full fit, not the closest-listing fallback. Send the order again with its text saying so: "
                        "each line, its price and stock, and that it meets every requirement."), call_args
            if (loose or self.cart_short()) and self.wire.spare() <= CLOSING + 30:
                self.closest_vetted = self.needs.version
                rows = [] if not (self.role('details') or self.batch_tool()) else self.quick_rows()
                spread = self.batch_tool()
                reads = len(rows) if not (spread and 2 <= len(rows)) else -(-len(rows) // spread[2])
                if rows and self.moves_left() > reads + 3:
                    self.open_queue = rows
                    self.open_say = "Before settling for the closest listing, I read the likeliest unread rows that could meet every requirement."
                    self.log.append(dict(kind='closest_quick', rows=rows))
                    return ("Not run yet (no step used). This order settles for listings that miss a requirement, which is right only "
                            "when no listing meets them all. Time is short, so the code reads just the likeliest unread rows in budget: "
                            + ", ".join(rows) + ". If one meets every requirement, put it in the cart in place of the line it beats and "
                            "order; otherwise order the cart as it is."), call_args
            elif loose or self.cart_short():
                self.closest_vetted = self.needs.version
                vet = self.nofit_vet(call_args)
                if vet:
                    self.log.append(dict(kind='closest_vet', loose=loose))
                    return ("Not run yet (no step used). This order settles for listings that miss a requirement (the request's fallback for "
                            "when nothing fits), and that fallback is right only when no listing meets every requirement: where a full fit "
                            "exists, the request wants it. So the code first looks for one, the same way it does before a refusal. "
                            + vet[0].replace("Not run yet (no step used). ", "", 1)), vet[1]
        count_note = self.acts_count_hold(self.call_kind(call_name, call_args))
        if count_note:
            return count_note, call_args
        if 'decline' == self.call_kind(call_name, call_args):
            vet = self.nofit_vet(call_args)
            if vet:
                return vet
            held = self.__dict__.get("no_fit_holds", 0)
            if 'order' == self.__dict__.get('no_fit', None) and held >= 2 and self.trolley.units and self.role("order") == call_name:
                self.log.append(dict(kind='no_fit_order', cart=dict(self.trolley.units)))
                return None, self.plain_args(call_name, 'No listing satisfies all requirements, so the nearest matches I found are what I am ordering: ' + ", ".join(self.trolley.units) + ".", "order")
            if 'order' == self.__dict__.get('no_fit', None) and held < 2 and 2 < self.moves_left():
                self.no_fit_holds = held + 1
                return ('Not run yet (no step used). The request states how to proceed if no option satisfies all the requirements'
                        + ('' if not self.__dict__.get('no_fit_rule', None) else ': "{}"'.format(self.no_fit_rule)) + '. In that situation it wants a listing ordered, and buying nothing is not what it means. Add the listing it describes to the cart'
                        + ('' if not self.needs.items() else ' (a single line for each requested product, in its quantity: a confirmed match when there is one, otherwise the closest listing for that product as the request defines it)') + ", look at its stock and price again, then place the order; in the order's own text, state that no option fully fits and which requirement it fails."), call_args
            gate = self.mv_gate("decline")
            if gate:
                return gate, call_args
            told = self.__dict__.pop("mv_decline_text", "")
            return None, self.scrub_decline(call_args if not told else self.plain_args(call_name, told, 'decline'), call_name)
        if 'order' == self.call_kind(call_name, call_args):
            gate = self.mv_gate("order")
            if gate:
                return gate, call_args
        return None, call_args

    def change_review(self):
        if 3 <= self.__dict__.get('reviews', 0) or not self.archive.read or self.wire.mute or self.wire.spare() < CLOSING + 20 \
                or not self.wire.usable(CHECKER) or 3 > self.moves_left():
            return ""
        self.reviews = self.__dict__.get("reviews", 0) + 1
        cards = self.archive.cards(14000, "", self.ref_foreign())
        if not cards:
            return ""
        ask_text = CHANGE_REVIEW + "\n\nTHE SHOPPER'S REQUEST:\n" + self.request + self.needs.changes()
        ask_text += "\n\nWHAT JUST CHANGED: " + " / ".join(self.needs.exchanges[-2:])
        ask_text += "\n\nREQUIREMENTS NOW: " + self.needs.text_body()
        ask_text += '\n\nPRIORITY: {} {}'.format(self.needs.priority(), self.needs.reading.get('priority_detail', None) or '').rstrip()
        ask_text += "\n\nTHE CART: " + self.trolley.summary()
        ask_text += '\n\nLISTINGS SEEN (DETAILS = the complete record was fetched, ROW = only appeared in search result rows):\n' + "\n".join(cards)
        ask_text += self.field_names()
        answer = self.wire.consult(ask_text, 2500, CHECKER)
        self.log.append(dict(kind="change_review", answer=answer))
        if not answer:
            return ""
        fits = [str(x) for x in answer.get('fits', None) or [] if isinstance(x, (str, int)) and str(x) in self.archive.ids][:6]
        pick = answer.get('pick', None)
        pick = '' if not (isinstance(pick, (str, int)) and str(pick) in self.archive.ids) else str(pick)
        self.__dict__.setdefault("named", set()).update(fits)
        try:
            before = {key.rpartition("|")[0] for key in self.passed}
            unk = self.__dict__.setdefault("unknown_n", {})
            for ident in fits:
                if ident in before:
                    self.passed.add('{}|{}'.format(ident, self.needs.version))
                    olds = [v for k, v in unk.items() if k.rpartition("|")[0] == ident]
                    if olds:
                        unk['{}|{}'.format(ident, self.needs.version)] = min(olds)
            passed_now = [i for i in fits if '{}|{}'.format(i, self.needs.version) in self.passed]
            full = [i for i in passed_now if 0 == self.unknowns(i)]
            if pick and full and not pick in full and (pick in passed_now and 0 < self.unknowns(pick) or not pick in passed_now):
                old_pick = pick
                pick = (full[0] if not 'cheapest' == self.needs.priority() else min(full, key=lambda i: 1000000000000.0 if not is_number(self.archive.price(i)) else float(self.archive.price(i))))
                self.log.append({"kind": "review_repick", "from": old_pick, "to": pick})
        except (Exception,) as err:
            self.log.append(dict(kind='review_pass_error', error='{}: {}'.format(type(err).__name__, err)[:300]))
        parts = ["[re-check after the change]"]
        parts.append(('no opened listing has been shown to satisfy all the requirements currently in force' if not fits else 'listings you opened that satisfy all requirements currently applying, strongest first: ' + ', '.join(fits)))
        if pick:
            parts.append('the one to take now: {}'.format(pick))
        if answer.get('cart_still_right', None) is True and self.trolley.units:
            parts.append('the line in the cart is still a fit: leave it, so the basket already reported remains the one that gets ordered')
        elif answer.get('cart_still_right', None) is False and self.trolley.units:
            parts.append("under the present conditions the item on the cart's line is no longer the best choice: swap it out (if the instructions say to consult the shopper before replacing anything, do that first)")
        if answer.get('reason', None):
            parts.append(str(answer["reason"])[:300])
        parts.append('Deal with this first: choose again from the options that fit at this point (taking the cheapest or the best one if the request wants that), correct the cart, check its stock and price again, and then wrap up; as long as an opened listing fits, you must not refuse.')
        return " ".join(parts)

    def code_asks_swap(self, old, new):
        tool = self.role("message")
        aim = None if not tool else self.text_arg(tool)
        if not aim or self.ask_limit() - self.msgs_sent < 1 + self.msg_reserve() or 4 > self.moves_left():
            return False
        text = ('Prior to my altering the basket: I want to remove ' + ", ".join(old[:3]) + " and put in " + ", ".join(new[:3])
                + " in its place, because the first one doesn't fit anymore. Does that work for you?")
        answers = self.send_chunk([('code_swap_ask_{}'.format(self.wire.tick), tool, {aim: text})])
        self.log.append(dict(kind='code_swap_ask', old=old[:3], new=new[:3]))
        return bool(answers) and not str(answers[0][1]).startswith("Not run")

    def line_seller(self, thing):
        learned = self.seller_keys()
        found: list[str] = []

        def visit(d):
            if not thing in [v for k, v in d.items() if isinstance(v, (str,)) and 'id' in str(k).lower()]:
                return
            for k, v in d.items():
                if isinstance(v, (str,)) and v.strip(None) and 60 >= len(v) and not 'id' in str(k).lower() and (
                        re.search(SELLER_KEYS, str(k).lower()) or str(k) in learned):
                    found.append(v.strip(None))
        for rec in self.archive.items:
            walk_dicts(rec, visit)
        return '' if not found else found[-1]

    def page_text(self, thing):
        out: list[str] = []

        def visit(d):
            if thing in [v for k, v in d.items() if isinstance(v, (str,)) and "id" in str(k).lower()]:
                long_strings_all(d, out)
        for rec in self.archive.items:
            walk_dicts(rec, visit)
        return " ".join(dict.fromkeys(out))

    @staticmethod
    def ident_tokens(text):
        found = re.findall(r"\b([A-Z][A-Za-z]{1,12}[ -]?\d{3,6})\b", text)
        return list(dict.fromkeys(found))[:8]

    @staticmethod
    def squash(text):
        return re.sub(r"[^a-z0-9]", "", str(text).lower())

    def dependency_faults(self):
        rows = self.needs.items()
        if not rows or not any(not r.get('depends_on', None) is None for r in rows):
            return []
        held = {}
        for line in self.trolley.units:
            k = self.product_index(line)
            if not k is None:
                held.setdefault(k, line)
        faults = []
        for k, row in enumerate(rows, 0):
            j = row.get('depends_on', None)
            if j is None or k == j or j >= len(rows) or not k in held or not j in held:
                continue
            anchor, dep = held[j], held[k]
            tokens = self.ident_tokens(self.page_text(anchor))
            if not tokens:
                continue
            text = self.squash(self.page_text(dep))
            if not any(self.squash(t) in text for t in tokens):
                faults.append("the {} ({}) needs to be compatible with the {} ({}), yet the page for it lists none of that item's identifiers ({})".format(row['product'], dep, rows[j]['product'], anchor, ', '.join(tokens)))
        return faults

    def dependency_hint(self):
        rows = self.needs.items()
        if not rows or not any(not r.get('depends_on', None) is None for r in rows):
            return ""
        out = []
        for k, row in enumerate(rows, 0):
            j = row.get('depends_on', None)
            if j is None or k == j or j >= len(rows):
                continue
            anchor = next((line for line in self.trolley.units if self.product_index(line) == j), "")
            tokens = [] if not anchor else self.ident_tokens(self.page_text(anchor))
            if not tokens:
                out.append('the {} needs to be compatible with the {}: settle on that item before anything else, then look up the {} by its model names or numbers and pick a page that mentions at least one of them'.format(row['product'], rows[j]['product'], row['product']))
            else:
                out.append("the {} needs to be compatible with the cart's {} ({}): run a search on the {} using each of its identifiers on its own as the search text ({}), open what comes back, and pick one whose page mentions any of those identifiers".format(row['product'], rows[j]['product'], anchor, row['product'], ', '.join(tokens)))
        return '' if not out else '\n[works with] ' + '; '.join(out)

    def cheap_sweep(self):
        if 'cheapest' != self.needs.priority() or self.needs.items() or self.orders_tried or self.__dict__.get('cheap_swept', None):
            return ""
        search = self.role("search")
        if not search or not self.trolley.units or 12 > self.moves_left() or self.wire.spare() < CLOSING + 40:
            return ""
        self.cheap_swept = True
        props = (self.shapes.get(search, None) or {}).get('properties', None) or {}
        need = set((self.shapes.get(search, None) or {}).get('required', None) or [])
        texts = [k for k, v in props.items() if isinstance(v, (dict,)) and 'string' == v.get('type', None) and not v.get('enum', None)]
        if 1 != len(texts) or need - {texts[0]}:
            return ""
        ran = {' '.join(self.free_text(w['args'])).strip(None).lower() for w in self.__dict__.get("page_book", {}).values()}
        queries = [q for q in self.label_queries() if not ' ' in q and not q in ran][:2]
        if not queries:
            return ""
        self.send_chunk([('cheap_{}_{}'.format(n, self.wire.tick), search, {texts[0]: q}) for n, q in enumerate(queries, 0)])
        line = next(iter(self.trolley.units))
        price = self.archive.price(line)
        if price is None:
            return ""
        rows = sorted((c, i, t) for i, (c, t, _) in self.shelf_rows().items()
                      if price > c and not i in self.trolley.units and not i in self.archive.read)[:4]
        self.log.append(dict(kind='cheap_sweep', queries=queries, rows=[i for _, i, _ in rows]))
        if not rows:
            return ""
        self.__dict__.setdefault("named", set()).update(i for _, i, _ in rows)
        return ('Not run yet (no step used). Since the lowest-priced match was requested, the code looked through ' + ", ".join(queries)
                + ": the following rows from the requested category are in stock and priced below the cart's line, and none has been opened: "
                + "; ".join(f"{i} '{t[:70]}' {c:g}" for c, i, t in rows) + '. Open these (you can do many in one reply). If any satisfies all the requirements, set it as the cart line, check its stock and price again, and place the order; if none does, order the cart unchanged.')

    def bound_hint(self):
        if 'cheapest' != self.needs.priority() or self.wire.spare() < CLOSING:
            return ""
        figs = []
        for req in self.needs.props_left():
            val = str(req).split(":", 1)[-1]
            for m in re.finditer(r"(\d+(?:\.\d+)?)\s?([a-z\"']{1,4})\b", val.lower()):
                figs.append(m.group(1) + m.group(2))
        figs = list(dict.fromkeys(figs))[:3]
        if not figs:
            return ""
        return ('\n[cheapest with a measured limit] the lowest-priced fit often states the exact figure of the limit: search ' + ", ".join(figs)
                + ' using the kind word (try it written with a space before the unit and without one), go through two pages for each, and open the lowest-priced rows within budget before settling the choice.')

    def basket_text(self):
        info = {}
        try:
            info = self.code_rows()
        except (Exception,):
            info = {}
        parts, total, priced = [], 0.0, True
        for thing, qty_n in self.trolley.units.items():
            row = info.get(thing, None) or {}
            cost = self.archive.price(thing)
            cost = row.get('price', None) if not not cost is None else cost
            units = qty_n if not float(qty_n).is_integer() else int(qty_n)
            name = '' if not row.get('title', None) else ' ({})'.format(row['title'][:60])
            try:
                seller = self.line_seller(thing)
            except (Exception,):
                seller = ""
            name += '' if not seller else ' from {}'.format(seller)
            if not cost is None:
                total += float(cost) * float(qty_n)
                parts.append(f'{units} x {thing}{name} at {cost:g} each')
            else:
                priced = False
                parts.append('{} x {}{}'.format(units, thing, name))
        unit = str(self.needs.reading.get('currency', None) or '').strip(None)
        if not parts:
            return 'One thing before I wrap up: nothing is in the basket, so no order will be placed.'
        text_body = "Before I finish, here is the basket: " + "; ".join(parts) + "."
        if priced:
            text_body += f" The total is {total:g}{'' if not unit else ' ' + unit}."
        if all(not self.last_stock(i) is False for i in self.trolley.units):
            text_body += ' Every one of these lines is in stock and can be bought.'
        return text_body + " Do you want me to place the order? Reply to confirm and I'll proceed."

    def finish_steps(self, kind):
        quiet = not self.needs.told and not self.needs.exchanges and not self.__dict__.get('event_seen', None)
        said = " ".join(str(x).split("shopper:", 1)[-1] for x in self.needs.exchanges).lower()
        if not self.needs.told and not self.__dict__.get('event_seen', None) and not re.search(
                r"confirm|let me know|tell me|before (?:you|ordering|placing|checkout|buying)|basket|total|check with|run it by|"
                r"\bwait|ok to|go ahead|approv|sign off|heads[- ]up|show me|send me|summar|remind|note that|make sure|"
                r"read (?:it|them) back|list (?:it|them)|ask me|check in", said):
            quiet = True
        due = self.acts_due(kind)
        if due:
            quiet = False
        if 2 > self.moves_left() or self.wire.done or self.wire.broke:
            return ""
        edits = self.__dict__.get("cart_edits", 0)
        sig = (kind, tuple(sorted(self.trolley.units.items())), edits, self.needs.told_version, self.refusal_n, self.needs.version)
        checked = self.__dict__.setdefault("finish_checked", set())
        if sig in checked:
            return ""
        checked.add(sig)
        msg_tool = self.role("message")
        msgs_left = 0 if not msg_tool else max(0, self.ask_limit() - self.msgs_sent)
        since = ['- message sent at step {}: {}'.format(tick, said) for tick, said, at in self.__dict__.get("msg_log", [])
                 if edits == at and (edits > 0 or not self.trolley.units)]
        told = self.needs.told
        usable = [tdef for tdef in self.toolset if not tdef['function']['name'] in self.finishers() and tdef["function"]["name"] != self.role("cart")]
        if msgs_left <= 0:
            usable = [tdef for tdef in usable if tdef["function"]["name"] != msg_tool]
        ask_text = FINISH_CHECK
        ask_text += "\n\nINSTRUCTIONS IN FORCE:\n" + ("\n".join('- {} (from the {})'.format(t['text'], t.get('source', None)) for t in told) or "(none stated)")
        if due:
            ask_text += ("\n\nRULES TO ACT ON AT THIS FINISH (the director's interpretation of the request; any step that the completed actions haven't yet carried out counts as pending - a step 'at_finish' belongs in the text of the finish call itself and is never pending):\n"
                         + "\n".join('- {} ({}): {} - rule: "{}"'.format(a['do'], a['when'], a['detail'], a['rule']) for a in due))
        ask_text += "\n\nTHE SHOPPER'S REQUEST:\n" + self.request
        ask_text += '\n\nNEWEST SHOPPER LINES AND STORE NOTICES (starting with the oldest):\n' + ("\n".join(self.needs.exchanges[-8:]) or "(none)")
        ask_text += "\n\nREQUIREMENTS NOW: " + self.needs.text_body()
        ask_text += "\n\nTHE CART: " + self.trolley.summary() + " " + self.basket_text()
        ask_text += "\n\nTHE FINISH ABOUT TO HAPPEN: " + kind
        ask_text += "\n\nDONE AFTER THE LAST CART CHANGE:\n" + ("\n".join(since) or '(nothing has been sent or done since the cart was last modified)')
        ask_text += '\n\nMESSAGES LEFT: {}\nSTEPS LEFT: {} (the finish call itself counts as one)'.format(msgs_left, self.moves_left())
        ask_text += "\n\nTOOLS THAT MAY CARRY OUT A STEP:\n" + "\n".join(
            '- {}: {} | arguments: {}'.format(tdef['function']['name'], tdef['function'].get('description', ''), as_text(tdef['function'].get('parameters', None) or {})) for tdef in usable)
        answer = None
        if not quiet and self.wire.usable(FINISHER) and self.wire.spare() > LAST_CALL + 10:
            answer = self.wire.consult(ask_text, 2500, FINISHER)
        calls: list[tuple[str, dict[str, Any], str]] = []
        if not answer is None:
            names = {tdef["function"]["name"] for tdef in usable}
            for step in (answer.get('pending', None) or [])[:FINISH_TURNS]:
                if not isinstance(step, (dict,)) or not step.get('tool', None) in names or not isinstance(step.get('args', None), (dict,)):
                    continue
                call_args, faults = ArgCheck.execute(dict(step["args"]), self.shapes[step["tool"]])
                if faults:
                    continue
                calls.append((step["tool"], call_args, str(step.get('why', None) or "")[:120]))
        aim = None if not msg_tool else self.text_arg(msg_tool)
        if kind == 'order' and self.trolley.units and aim and not since and msgs_left > sum(1 for c in calls if c[0] == msg_tool) \
                and not any(c[0] == msg_tool for c in calls):
            calls.append((msg_tool, {aim: self.basket_text()}, "the basket told before the order"))
        self.log.append(dict(kind="finish_check", finish=kind, answer=answer, calls=[c[0] for c in calls]))
        calls = calls[: max(0, self.moves_left() - 1)]
        if not calls:
            return ""
        proposals = [(dict(id='finish_step_{}_{}'.format(self.wire.tick, n), type='function', function=dict(name=name, arguments=json.dumps(args)))) for n, (name, args, _) in enumerate(calls, 0)]
        state = (self.needs.version, self.needs.told_version, self.msg_cap, self.step_cap, self.refusal_n, len(self.trolley.units))
        was = self.__dict__.get('scripted_batch', None)
        try:
            responses = self.dispatch(proposals, scripted=True)
        finally:
            self.scripted_batch = was
        changed = state != (self.needs.version, self.needs.told_version, self.msg_cap, self.step_cap, self.refusal_n, len(self.trolley.units))
        done = ['{} {} -> {}'.format(name, json.dumps(args, ensure_ascii=False)[:400], text.split(chr(10) + '[status]')[0][:700])
                for (name, args, _), (_, text) in zip(calls, responses)]
        note = '[prior to this ending, the code did the following: ' + "; ".join(c[2] for c in calls) + "] " + " | ".join(done)
        self.log.append(dict(kind="finish_steps_done", note=note[:600]))
        if msg_tool and any(msg_tool == name for name, _, _ in calls) and (changed or self.needs.reply_due or self.wire.done):
            return ("Not run yet (no step used). " + note + '\nThe reply altered something: compare it with the requirements and instructions in force, correct whatever it calls for, and then finish once more.')
        self.finish_note = note
        return ""

    def dispatch(self, proposals, scripted = False):
        responses: list[tuple[str, str]] = []
        self.scripted_batch = scripted
        self.begin_batch()
        permitted = {tdef["function"]["name"] for tdef in self.offered()}
        to_send: list[tuple[str, str, dict[str, Any]]] = []
        queued_messages = 0
        cart_held = False
        self.batch_puts = []
        self.batch_cart = any((p.get('function', None) or {}).get('name', None) == self.role("cart") for p in proposals if isinstance(p, (dict,)))
        for proposal in proposals:
            func = proposal.get('function', None) or {}
            if func.get('name', None) == self.role("cart"):
                raw = func.get('arguments', None)
                put_args = ({} if not (isinstance(raw, (str,)) and raw.strip(None)) else json_object(raw)) if not isinstance(raw, (dict,)) else raw
                if isinstance(put_args, (dict,)) and self.trolley.is_put(put_args) and self.trolley.aim(put_args):
                    self.batch_puts.append(self.trolley.aim(put_args))
        for proposal in proposals:
            if cart_held and proposal["function"]["name"] == self.role("cart"):
                responses.append((proposal["id"], "Not run (no step used): a cart edit made earlier in this reply didn't execute; find out the reason, then submit the cart edits once more."))
                continue
            refusal, call_name, call_args = self.screen(proposal, permitted, scripted, len(proposals), queued_messages)
            if refusal:
                cart_held = cart_held or (call_name == self.role("cart") and isinstance(call_args, (dict,)) and bool(call_args)
                                          and not self.trolley.is_put(call_args))
                responses.append((proposal["id"], refusal))
                continue
            if call_name == self.role("message"):
                queued_messages = queued_messages + 1
            to_send.append((proposal["id"], call_name, call_args))
        if to_send and to_send[0][1] in self.finishers():
            cid, call_name, call_args = to_send[0]
            held, call_args = self.screen_finish(cid, call_name, call_args)
            if held:
                responses.append((cid, held))
                return responses
            to_send[0] = (cid, call_name, call_args)
        if to_send and not to_send[0][1] in self.finishers() and self.finishers():
            hold = max(0, self.moves_left() - 1)
            for cid, _, _ in to_send[hold:]:
                responses.append((cid, 'Not run (no step used): just the final step remains, and that has to be placing the order (or declining).'))
            to_send = to_send[:hold]
        for begin in range(0, len(to_send), 16):
            slab = to_send[begin: begin + 16]
            if self.wire.done or self.wire.broke or 0 >= self.moves_left():
                responses += [(cid, 'Not run: the episode has ended, or the step budget is used up.') for cid, _, _ in slab]
                continue
            slab = slab[: self.moves_left()]
            responses += self.send_chunk(slab)
        note = self.__dict__.pop("finish_note", "")
        if note and to_send and to_send[0][1] in self.finishers():
            responses = [(cid, text if not cid == to_send[0][0] else note + '\n' + text) for cid, text in responses]
        call_rank = {c["id"]: i for i, c in enumerate(proposals, 0)}
        responses.sort(key=lambda req: call_rank.get(req[0], 0))
        return responses

    def send_chunk(self, slab):
        acts = [dict(name=call_name, args=call_args) for _, call_name, call_args in slab]
        prior = self.wire.tick
        resp = self.wire.act(acts)
        self.log.append(dict(kind='env', step=prior + 1, actions=acts, reply=resp))
        if resp is None:
            return [(cid, "The call failed (network or server). Try again.") for cid, _, _ in slab]
        obs_list = [resp.get('observation', None)] if not isinstance(resp.get('observations', None), (list,)) else resp.get('observations', None)
        collected = []
        for idx, (cid, call_name, call_args) in enumerate(slab, 0):
            if idx >= len(obs_list):
                collected.append((cid, 'Not run: the episode had already ended before this call was made.'))
                continue
            collected.append((cid, self.absorb(call_name, call_args, obs_list[idx], resp, idx == len(obs_list) - 1)))
            self.mv_note("call", '{}({}) -> {}'.format(call_name, as_text(call_args)[:120], collected[-1][1][:160]))
            rnd = self.mv_round()
            eng = self.mv_engine(call_name)
            if eng == 'read':
                recent = self.__dict__.setdefault("mv_recent", [])
                for ident in sorted(self.asked_ids(call_args)):
                    if ident in recent:
                        recent.remove(ident)
                    recent.append(ident)
                del recent[:-8]
            rnd["used"][eng] = rnd["used"].get(eng, 0) + 1
            if call_name == self.role("cart") and self.trolley.aim(call_args):
                rnd["puts"][self.trolley.aim(call_args)] = rnd["puts"].get(self.trolley.aim(call_args), 0) + 1
        return collected

    def learn_id_cap(self, call_name, call_args, obs):
        sent = [v for v in (call_args or {}).values() if isinstance(v, (list,)) and 2 < len(v) and all(isinstance(x, (str,)) for x in v)]
        if not sent or call_name in self.finishers():
            return
        text = as_text(obs)
        if 300 > len(text) and not re.search(r"\bp-[0-9a-f]{6,}", text) \
                and re.search(r"couldn'?t|could not|cannot|can'?t|not (?:be )?found|no such|unknown|invalid|unable|locate|too many|error", text.lower()):
            caps = self.__dict__.setdefault("id_caps", {})
            if 2 < caps.get(call_name, 99):
                caps[call_name] = 2
                self.log.append(dict(kind='id_cap', tool=call_name, cap=2, sent=len(sent[0])))

    def absorb(self, call_name, call_args, obs, resp, last):
        shown, obs = obs, lensed(obs)
        msg_tool, basket_tool = self.role("message"), self.role("cart")
        if msg_tool == call_name:
            self.msgs_sent = self.msgs_sent + 1
            said = " ".join(str(val) for val in call_args.values() if isinstance(val, (str,)))
            self.__dict__.setdefault("msg_log", []).append((self.wire.tick, said[:600], self.__dict__.get("cart_edits", 0)))
            if not self.__dict__.get('scripted_batch', None):
                self.questions_log.append('- (the assistant\'s own message): "{}" -> {}'.format(said[:300], as_text(obs)[:300]))
        if basket_tool == call_name:
            self.cart_edits = self.__dict__.get("cart_edits", 0) + 1
        self.last_kind = self.call_kind(call_name, call_args)
        if self.last_kind == 'order':
            self.orders_tried = self.orders_tried + 1
        if not (call_name in (basket_tool, msg_tool) or call_name in self.finishers()):
            self.last_calls = (self.last_calls + [call_name + as_text(call_args)])[-2:]
        else:
            self.last_calls = []
        self.archive.add(obs)
        self.absorb_n = self.__dict__.get("absorb_n", 0) + 1
        if basket_tool != call_name and not self.__dict__.get('covering', None):
            self.since_cart = self.__dict__.get("since_cart", 0) + 1
        try:
            self.learn_id_cap(call_name, call_args, obs)
        except (Exception,) as err:
            self.log.append(dict(kind='id_cap_error', error='{}: {}'.format(type(err).__name__, err)[:300]))
        try:
            gone_now = self.learn_caps(call_name, call_args, obs)
        except (Exception,) as err:
            gone_now = []
            self.log.append(dict(kind='caps_error', error='{}: {}'.format(type(err).__name__, err)[:300]))
        if self.trolley.units:
            fresh: set[str] = set()
            stock_rule = bool(self.__dict__.get('stock_seen', None))
            asked = self.asked_ids(call_args)
            rows_ok = bool(self.__dict__.get('rows_live', None)) and not self.__dict__.get('rows_lag', None)
            lookups_only = not rows_ok and not call_name in (basket_tool, self.role('cart_view'))

            def seen_line(d):
                hits = {v for k, v in d.items() if isinstance(v, (str,)) and v in self.trolley.units and "id" in str(k).lower()}
                if not hits:
                    return
                if stock_rule:
                    if self.stock_state(d) is None or (lookups_only and not hits & asked):
                        return
                elif not any(isinstance(v, (dict,)) for v in d.values()):
                    return
                fresh.update(v for k, v in d.items() if isinstance(v, (str,)) and v in self.trolley.units)
            walk_dicts(obs, seen_line)
            for line_id in fresh:
                self.__dict__.setdefault("line_read", {})[line_id] = self.wire.tick
                self.__dict__.setdefault("line_seq", {})[line_id] = self.absorb_n
        if call_name == self.role("details") and not (isinstance(obs, (dict,)) and obs.get('error', None)):
            self.archive.read.update(val for val in call_args.values() if isinstance(val, (str,)) and val in self.archive.ids)
        looked = bool(self.role("details")) and not call_name in (self.role('details'), self.role('search')) and call_name in self.lookup_tools()
        named_ids = {val for val in call_args.values() if isinstance(val, (str,))}
        if self.moves_left() < self.order_due() and (call_name in (self.role("details"), self.role("search")) or looked) \
                and not self.__dict__.get('covering', None) \
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
        except (Exception,) as err:
            self.log.append(dict(kind='notes_remark_error', error='{}: {}'.format(type(err).__name__, err)[:300]))
        if basket_tool == call_name:
            self.order_vetted = False
            self.trolley.changed_at = self.wire.tick
            held_before = dict(self.trolley.units)
            remarks.append(self.trolley.booked(call_args, obs))
            try:
                self.cart_moved(held_before)
                self.mark_fresh(obs)
                self.after_put(call_args, obs)
            except (Exception,) as err:
                self.log.append(dict(kind='after_put_error', error='{}: {}'.format(type(err).__name__, err)[:300]))
            put_item = self.trolley.aim(call_args)
            if put_item and self.trolley.is_put(call_args):
                self.refusals_seen[put_item] = self.refusal_n
                puts = self.__dict__.setdefault("puts", {})
                puts[put_item] = puts.get(put_item, 0) + 1
            put_units = self.trolley.count_arg(call_args)
            if put_item and self.trolley.is_put(call_args) and '{}|{}'.format(put_item, self.needs.version) in self.passed \
                    and (put_units is None or put_units > 0) and 0 < self.trolley.held(put_item) and not put_item in self.dead_lines():
                remarks.append('A separate verification determined that {} satisfies all requirements currently applying, with each property read according to the value rules (when this store spells out the value in a field, only those exact words satisfy it); stick with it unless a later change excludes it.'.format(put_item))
        if call_name == self.role("cart_view"):
            self.trolley.viewed(obs, self.wire.tick)
            remarks.append(self.trolley.summary())
        refused = self.refused(call_name, obs, dict(finished=False) if not last else resp)
        if refused and self.last_kind == 'order':
            self.cart_seq = self.absorb_n
        remarks.append(self.notes_for(call_name, obs, refused))
        if call_name == self.role("search"):
            remarks.append(self.search_note(call_args, obs))
            self.__dict__.get("named_pages", set()).discard(json.dumps(call_args, default=str, sort_keys=True))
            try:
                remarks.append(self.merge_joined(obs))
            except (Exception,) as err:
                self.log.append(dict(kind='merge_joined_error', error='{}: {}'.format(type(err).__name__, err)[:300]))
        if (last and not self.__dict__.get('early_audit', None) and not call_name in (basket_tool, msg_tool) and not call_name in self.finishers()
                and FIRST_ORDER_BY < self.moves_left() <= self.step_cap // 3 + 2 and not self.trolley.units
                and not any(key.rpartition("|")[2] == str(self.needs.version) for key in self.passed)):
            self.early_audit = True
            ahead = self.review_decline(early=True)
            if ahead:
                remarks.append("Nothing verified fits yet. " + ahead)
        qtext = None if not msg_tool == call_name else ' '.join((str(val) for val in call_args.values() if isinstance(val, (str,))))
        caught = self.shopper_lines(call_name, obs, refused)
        talk = self.__dict__.setdefault("mv_talk", [])
        if msg_tool == call_name:
            self.mv_msg_tick = self.wire.tick
            talk.append('[step {}] YOU: {}'.format(self.wire.tick, (qtext or '')[:400]))
        elif basket_tool == call_name:
            self.mv_cart_tick = self.wire.tick
        if caught:
            if not msg_tool == call_name:
                self.mv_events = self.__dict__.get('mv_events', 0) + 1
                talk.append('[step {}] EVENT in the result of {}'.format(self.wire.tick, call_name) + ('' if not refused else ' (refused)') + ': ' + ' / '.join(caught)[:500])
            else:
                talk.append('[step {}] SHOPPER: '.format(self.wire.tick) + ' / '.join(caught)[:500])
        if caught:
            ver_before = self.needs.version
            loose_before, cap_before = len(self.needs.let_go), self.needs.spend_cap
            remarks.append(self.needs.hear(caught, qtext))
            self.ref_take()
            news = self.needs.__dict__.get('allowance_news', None)
            if news:
                self.needs.allowance_news = None
                try:
                    self.apply_allowance(news)
                except (Exception,) as err:
                    self.log.append(dict(kind='allowance_error', error='{}: {}'.format(type(err).__name__, err)[:300]))
            if self.needs.told_version != self.__dict__.get('told_seen', None):
                self.told_seen = self.needs.told_version
                self.order_vetted = False
            if ver_before < self.needs.version and (qtext is None or len(self.needs.let_go) > loose_before
                                                     or cap_before != self.needs.spend_cap):
                try:
                    review = self.change_review()
                    if review:
                        remarks.append(review)
                except (Exception,) as err:
                    self.log.append(dict(kind='change_review_error', error='{}: {}'.format(type(err).__name__, err)[:300]))
        text_body = as_text(clipped(shown, RECORD_TEXT if not call_name == self.role('search') else SEARCH_TEXT))
        if last and resp.get('finished', None):
            text_body += " [the episode is finished]"
        summary = " ".join(qty_n for qty_n in remarks if qty_n)
        return text_body + ('' if not summary else '\n[note] ' + summary)

    def mv_note(self, kind, text):
        book = self.__dict__.setdefault("mv_book", [])
        book.append('[step {}] {}: {}'.format(self.wire.tick, kind, text))
        del book[:-40]

    def mv_tools_text(self):
        if not 'mv_tools' in self.__dict__:
            lines = []
            for tdef in self.toolset:
                fn = tdef.get('function', None) or {}
                params = list(((fn.get('parameters', None) or {}).get('properties', None) or {}).keys())
                lines.append('- {}({}): {}'.format(fn.get('name', None), ', '.join(params), str(fn.get('description', None) or '')[:200]))
            self.mv_tools = "\n".join(lines)
        return self.mv_tools

    def mv_check_of(self, ident):
        ver = str(self.needs.version)
        if '{}|{}'.format(ident, ver) in self.passed:
            return "check passed"
        if any(val and key.split("|")[0] == ident and key.split("|")[1:2] == [ver] for key, val in self.rulings.items()):
            return "check FAILED"
        return "unchecked"

    def mv_row_text(self, ident, info):
        row = info.get(ident, None) or {}
        price = self.archive.price(ident) if not is_number(row.get('price', None)) else row.get('price', None)
        stock = {True: "in stock", False: "SOLD OUT"}.get(row.get('stock', None), "stock unknown")
        return (("{} '{}' price {}, {}, {}, {}".format(ident, str(row.get('title', None) or '')[:70], '?' if not not price is None else price, stock, 'row only' if not ident in self.archive.read else 'record read', self.mv_check_of(ident))))

    def mv_cart_text(self):
        if not self.trolley.units:
            return "empty"
        info = self.code_rows()
        total, lines = 0.0, []
        for ident, units in self.trolley.units.items():
            row = info.get(ident, None) or {}
            price = self.archive.price(ident) if not is_number(row.get('price', None)) else row.get('price', None)
            total += 0.0 if not is_number(price) else float(price) * float(units)
            lines.append(f"  {units:g} x " + self.mv_row_text(ident, info))
        changed = self.__dict__.get("mv_cart_tick", -1) > self.__dict__.get("mv_msg_tick", -1)
        return ("\n".join(lines) + f'\n  total {total:g}; budget {self.needs.spend_cap}; the cart was modified after the latest message: '
                + ('no' if not changed else 'yes'))

    def mv_cands_text(self):
        info = self.code_rows()
        out, avoid = [], set()
        for rank in range(0, 4):
            picks = self.code_picks(avoid)
            if not picks:
                break
            for ident, qty in picks:
                avoid.add(ident)
                out.append('  rank {}: {} x '.format(rank + 1, qty) + self.mv_row_text(ident, info))
        return '  none the code can see' if not out else '\n'.join(out)

    def mv_state(self, point, proposal = "", why = ""):
        nd = self.needs
        rd = {} if not isinstance(nd.reading, (dict,)) else nd.reading
        out = ['DECISION POINT: {}'.format(point)]
        if proposal:
            out.append('THE CODE PROPOSES: {}'.format(MV_KINDS.get(proposal, proposal)) + ('' if not why else ' ({})'.format(why)))
        rules = [] if not isinstance(rd.get('rules', None), (list,)) else [str(r) for r in rd.get('rules', None) or [] if r]
        if rules:
            out.append("RULES THE REQUEST READER FOUND:\n" + "\n".join("- " + r for r in rules))
        try:
            told = self.needs.told_text()
        except (Exception,):
            told = ""
        if told:
            out.append('INSTRUCTIONS IN FORCE (the list for the reader and listener; the flags tell what has to be done ahead of the finish and what calls for a message):\n' + told.replace("\n[instructions in force] ", ""))
        procs = self.__dict__.get('mv_procs', None) or []
        if procs:
            out.append("PROCEDURES YOU FOUND EARLIER:\n" + "\n".join("- " + p for p in procs))
        if "mv_keep" in self.__dict__ and not int(self.__dict__.get('mv_keep', None) or 0):
            out.append('MESSAGES RESERVED FOR The PROCEDURE entry reads none: the plan identified no step in the request that should be done by message prior to ordering, so skip the basket message unless a rule currently in effect requires one.')
        sent = self.__dict__.get('msg_log', None) or []
        after = [m for m in sent if m[2] == self.__dict__.get("cart_edits", 0)]
        out.append("MESSAGES AFTER THE LAST CART CHANGE: " + (('none (a message sent before the last cart change is not counted)' if not after else '{} (newest: "{}")'.format(len(after), str(after[-1][1])[:120]))))
        out.append("REQUIREMENTS NOW (stated, revealed, changed): " + nd.text_body())
        out.append('PRIORITY: {}'.format(nd.priority()) + ('' if not rd.get('priority_detail', None) else ' ({})'.format(rd.get('priority_detail', None))))
        if 'order' == self.__dict__.get('no_fit', None):
            out.append('WHEN NOTHING FITS (PLAN is how your reading of this request came out): ' + ('buy nothing (decline)' if not self.no_fit == 'order' else 'ORDER the nearest listing that matches its description, giving one line for each product and never a refusal')
                       + ('' if not self.__dict__.get('no_fit_rule', None) else ' - "{}"'.format(self.no_fit_rule[:200])))
        cap = '{} (no limit stated)'.format(self.msgs_sent) if not not self.msg_cap is None else '{} of {}'.format(self.msgs_sent, self.msg_cap)
        out.append('METERS: {} of {} steps spent ({} remaining; {} reserved for the order); {} messages spent, {} held back for procedures; {} seconds remaining; {} orders attempted; {} refusals'.format(self.wire.tick, self.step_cap, self.moves_left(), self.order_due(), cap, self.__dict__.get('mv_keep', 0), int(self.wire.spare()), self.orders_tried, self.refusal_n))
        out.append('ENGINE STEP BUDGETS THIS ROUND (shown as used/limit; reset at news; any engine that has hit its limit is unable to run): '
                   + self.mv_budget_text())
        talk = self.__dict__.get('mv_talk', None) or []
        out.append('MESSAGES AND SHOPPER LINES (earliest first):\n' + ('none yet' if not talk else '\n'.join(talk[-16:])))
        out.append("CART NOW:\n" + self.mv_cart_text())
        out.append('CANDIDATES (ranked by the code, for each product):\n' + self.mv_cands_text())
        recent = [i for i in (self.__dict__.get('mv_recent', None) or [])[-4:] if i in self.archive.ids]
        if recent:
            out.append("RECENTLY OPENED RECORDS (newest last):\n" + "\n".join(
                '{}: '.format(i) + " ".join(self.archive.about(i, 1))[:700] for i in recent))
        book = self.__dict__.get('mv_book', None) or []
        out.append("LAST MOVES (newest last):\n" + ('none yet' if not book else '\n'.join(book[-14:])))
        return "\n\n".join(out)[:MV_STATE_CHARS]

    def mv_direct(self, point, proposal = "", why = "", gate = False):
        n = self.__dict__.get("mv_direct_n", 0)
        if (not gate and (MV_DIRECT_MAX <= n or MV_DIRECT_SPEND < self.wire.spend or self.wire.spare() < CLOSING + 15)) \
                or self.wire.spare() < LAST_CALL + 8 or self.wire.mute or self.wire.no_model or self.wire.done or self.wire.broke:
            return None
        self.mv_direct_n = n + 1
        self.mv_seen = self.mv_info()
        task = "THE ERRAND (fixed):\nREQUEST (the shopper's own words):\n" + self.request + "\n\nTOOLS OF THIS STORE:\n" + self.mv_tools_text()
        msgs = [dict(role='system', content=DIRECT_RULES), dict(role='user', content=task),
                dict(role='assistant', content='Got it. Provide the state whenever a decision has to be made.'),
                dict(role='user', content=self.mv_state(point, proposal, why))]
        answer = None
        models = DIRECTOR if not point.startswith(STRONG_POINTS) else DIRECTOR_STRONG
        flat = [dict(role='user', content=DIRECT_RULES + '\n\n' + task + '\n\n' + msgs[-1]['content'])]
        for attempt in range(0, 2 if not gate else 3):
            msg = self.wire.query_model(flat if not attempt == 0 else msgs, DIRECTOR if not attempt == 0 else models, 1500,
                                        [DIRECT_TOOL], "director_answer")
            called = [c.get('function', None) or {} for c in (msg or {}).get('tool_calls', None) or []
                      if 'director_answer' == (c.get('function', None) or {}).get('name', None)]
            given = (msg or {}).get('content', None) if not called else called[0].get('arguments', None)
            answer = self.mv_parse(given or '') if not isinstance(given, (dict,)) else given
            if isinstance(answer, (dict,)):
                break
        self.log.append(dict(kind='director', point=point[:60], proposal=proposal, answer=answer))
        if not isinstance(answer, (dict,)):
            return None
        keep = answer.get('keep_messages', None)
        if is_number(keep) and not isinstance(keep, (bool,)):
            cap = min(2, ASK_DEFAULT if not not self.msg_cap is None else self.msg_cap)
            want = max(0, min(int(keep), cap))
            quotes = [str(x) for x in answer.get('quotes', None) or [] if isinstance(x, (str,)) and x.strip(None)]
            if want and quotes and point.startswith("PLAN"):
                said = " ".join(re.findall(r"[a-z0-9]+", self.request.lower()))
                real = [x for x in quotes if 4 <= len(re.findall('[a-z0-9]+', x.lower()))
                        and " ".join(re.findall(r"[a-z0-9]+", x.lower())) in said
                        and not re.search(r"\bask\b|question|what you need|whatever|\d+ messages", x.lower())]
                if not real:
                    self.log.append(dict(kind='keep_unquoted', quotes=quotes[:3]))
                    want = 0
            if point.startswith(("PLAN", "FINAL")) or want < self.__dict__.get("mv_keep", 0):
                self.mv_keep = want
        nofit = str(answer.get('no_fit', None) or '').strip(None).lower()
        if point.startswith("PLAN") and nofit in ("order", "decline", ""):
            self.no_fit = nofit
            self.no_fit_rule = str(answer.get('no_fit_rule', None) or "")[:300]
        acts = answer.get('acts', None)
        if point.startswith("PLAN") and isinstance(acts, (list,)):
            self.acts = [{k: str(a.get(k, None) or "")[:200] for k in ("rule", "do", "when", "detail")} for a in acts
                         if isinstance(a, (dict,)) and (a.get('rule', None) or a.get('detail', None))][:8]
        procs = answer.get('procedures', None)
        if isinstance(procs, (list,)) and procs:
            self.mv_procs = [str(p)[:200] for p in procs][:6]
        self.mv_note("director", '{} - {}'.format(answer.get('move', None), str(answer.get('why', None) or '')[:160]))
        return answer

    @staticmethod
    def mv_parse(text):
        if not isinstance(text, (str,)) or not text.strip(None):
            return None
        got = json_object(text)
        if isinstance(got, (dict,)):
            return got
        body = text if not '{' in text else text[text.find('{'):]
        for tail in ("}", ""):
            trimmed = re.sub(r"[)\]\s.]*$", "", body.rstrip()) + tail
            got = json_object(trimmed)
            if isinstance(got, (dict,)):
                return got
        out: dict[str, Any] = {}
        for key in ("move", "query", "text", "tool", "why"):
            m = re.search(r'"' + key + r'"\s*:\s*"((?:[^"\\]|\\.)*)"', body)
            if m:
                out[key] = m.group(1).replace('\\"', '"')
        m = re.search(r'"keep_messages"\s*:\s*(\d+)', body)
        if m:
            out["keep_messages"] = int(m.group(1))
        m = re.search(r'"ids"\s*:\s*\[([^\]]*)\]', body)
        if m:
            out["ids"] = re.findall(r'"([^"]+)"', m.group(1))
        m = re.search(r'"procedures"\s*:\s*\[([^\]]*)\]', body)
        if m:
            out["procedures"] = re.findall(r'"((?:[^"\\]|\\.)*)"', m.group(1))
        return None if not (out.get('move', None) or 'keep_messages' in out) else out

    def mv_info(self):
        return (self.__dict__.get("mv_events", 0), self.refusal_n, self.msgs_sent)

    def acts_due(self, kind):
        out = []
        for a in self.__dict__.get('acts', None) or []:
            when, do = a.get("when", ""), a.get("do", "")
            if do in ("limit", "order_closest", "decline"):
                continue
            if when in ("always", "at_finish") or (kind == 'order' and when == 'before_order') or (kind == 'decline' and when == 'before_decline'):
                out.append(a)
        return out

    def acts_count_hold(self, kind):
        if not kind in ('order', 'decline'):
            return ""
        held = self.__dict__.setdefault("acts_held", {})
        for a in self.acts_due(kind):
            if not a.get('do', None) in ('open', 'compare'):
                continue
            m = re.search(r"\b(\d+)\b", a.get("detail", "") + " " + a.get("rule", ""))
            want = 0 if not m else int(m.group(1))
            got = len(self.archive.read)
            key = a.get("rule", "")[:80]
            if 0 < want <= 20 and want > got and 2 > held.get(key, 0) and self.moves_left() > (want - got) + 2:
                held[key] = held.get(key, 0) + 1
                return (('Not run yet (no step used). The request\'s rule "{}" asks to {} {} listings before this {}; {} were opened so far. Open the missing ones (fitting candidates first), then {}.'.format(a.get('rule', '')[:160], a.get('do', None), want, kind, got, kind)))
        return ""

    def no_fit_say(self):
        if 'order' != self.__dict__.get('no_fit', None):
            return ""
        return (" THIS REQUEST, if nothing matches, is a request to order the listing it describes and not a request to buy nothing: place an order for the nearest match and state in the order's text what it lacks.")

    def mv_gate(self, kind):
        key = (kind, tuple(sorted((k, float(v)) for k, v in self.trolley.units.items())), self.needs.version, self.msgs_sent,
               self.refusal_n, len(self.__dict__.get('mv_talk', None) or []), self.wire.tick)
        gates = self.__dict__.setdefault("mv_gates", {})
        if 6 <= gates.get(kind, 0):
            return 
        if isinstance(self.__dict__.get('mv_next', None), (dict,)):
            return "Not run yet (no step used). A move the director put first has not run yet."
        if self.__dict__.get('mv_gate_key', None) == key:
            return ""
        self.mv_gate_key = key
        gates[kind] = gates.get(kind, 0) + 1
        ver = str(self.needs.version)
        if kind == 'order' and 3 < self.moves_left() and self.role("cart"):
            rows = self.needs.items()
            prod = (lambda i: None) if not rows else lambda i: self.product_index(i)
            def qty_of(k, held=None):
                if not held is None:
                    return max(1, int(float(held)))
                q = self.needs.quantity() if not (rows and not k is None and (k < len(rows))) else rows[k].get('quantity', None)
                return 1 if not (isinstance(q, (int, float)) and (not isinstance(q, (bool,)))) else max(1, int(q or 1))
            cart = {i: max(1, int(float(q or 1))) for i, q in self.trolley.units.items()}
            if rows:
                held = {prod(i) for i in cart}
                adds = [(self.passed_for(k)[0], qty_of(k)) for k in range(0, len(rows)) if not k in held and self.passed_for(k)]
                if adds and not self.code_put(list(cart.items()) + adds) is None:
                    self.mv_gate_key = None
                    return ('Not run yet (no step used). One product was missing from the cart, so the fit that was checked for it was added: '
                            + ", ".join(a for a, _ in adds) + '. Check the stock and prices in the cart again, and then place the order.')
            for thing in list(cart):
                if '{}|{}'.format(thing, ver) in self.passed:
                    continue
                ruling = '' if not self.wire.spare() > CLOSING + 20 else self.check_item(thing, 'ORDER')
                if '{}|{}'.format(thing, ver) in self.passed:
                    continue
                fits = [i for i in self.passed_for(prod(thing)) if thing != i and not i in cart]
                if fits:
                    picks = [(i, q) for i, q in cart.items() if thing != i] + [(fits[0], cart[thing])]
                    if not self.code_put(picks) is None:
                        self.mv_gate_key = None
                        return ("Not run yet (no step used). {} hasn't passed any item check".format(thing) + ('' if not ruling else ' ({})'.format(ruling[:200]))
                                + '; the checked fit {} replaced it in the cart; check its stock and price again, and then place the order.'.format(fits[0]))
                if ruling:
                    self.mv_gate_key = None
                    return "Not run yet (no step used). " + ruling
            if 'cheapest' == self.needs.priority() and self.wire.spare() > CLOSING + 40 and 4 < self.moves_left():
                tried = self.__dict__.setdefault("gate_checked", set())
                shelf = self.shelf_rows()
                for line in list(cart):
                    lp = self.archive.price(line)
                    if not is_number(lp):
                        continue
                    cands = sorted((float(c), i) for i, (c, _, _) in shelf.items()
                                   if i in self.archive.read and not i in cart and float(c) < float(lp) and prod(i) == prod(line)
                                   and not '{}|{}'.format(i, ver) in self.passed and not '{}|{}'.format(i, ver) in tried)[:2]
                    for _, i in cands:
                        tried.add('{}|{}'.format(i, ver))
                        try:
                            self.check_item(i, "COMPARE WITH THE CART'S PICK")
                        except (Exception,) as err:
                            self.log.append(dict(kind='gate_check_error', error='{}: {}'.format(type(err).__name__, err)[:300]))
        if kind == 'order':
            ver = self.needs.version
            self.gate_rules_step()
            if ver != self.needs.version:
                return ("Not run yet (no step used). One of the request's conditions was just settled using the cart (check the conditions line): look over the lines it affects again, then place the order.")
        if kind == 'order' and (self.spec_read_cart() | self.open_link_pages()):
            return ("Not run yet (no step used). Already done: the details of every order line and the page view of each listing the request's reference passed through were read; place the order now.")
        if kind == 'order' and 'cheapest' == self.needs.priority() and self.wire.spare() > CLOSING + 30 and 4 < self.moves_left():
            for thing in list(self.trolley.units):
                try:
                    cheaper = self.cheaper_fit(thing)
                    if cheaper and self.needs.items() and self.product_index(cheaper) != self.product_index(thing):
                        cheaper = ""
                    if not cheaper:
                        if not self.rank_check(thing):
                            continue
                        cheaper = self.rank_verify(thing)
                except (Exception,) as err:
                    self.log.append(dict(kind='gate_rank_error', error='{}: {}'.format(type(err).__name__, err)[:300]))
                    continue
                if cheaper and not cheaper in self.trolley.units:
                    qty = max(1, int(float(self.trolley.units.get(thing, None) or 1)))
                    picks = [(i, max(1, int(float(q)))) for i, q in self.trolley.units.items() if thing != i] + [(cheaper, qty)]
                    if not self.code_put(picks) is None:
                        self.mv_gate_key = None
                        return ('Not run yet (no step used). The cart swapped out {} for a lower-priced listing, {}, that got through the item check; check its stock and price again, then place the order.'.format(thing, cheaper))
        dec = self.mv_direct("FINAL {} GATE: the code's checks have all succeeded, so the {} is sent right away unless you pick a different action beforehand".format(kind.upper(), kind), kind, gate=True)
        early = str((dec or {}).get('move', None) or 'continue').strip(None).lower()
        if kind == 'order' and early in ("order", "continue", "ask", "tool", "") and self.basket_due() and 2 <= self.moves_left():
            if self.send_basket():
                return 'Not run yet (no step used). The shopper already received the basket and its total; place the order now.'
        if dec is None and kind == 'order' and 1 <= int(self.__dict__.get('mv_keep', 0) or 0) and self.trolley.units \
                and (self.msg_cap is None or self.msg_cap > self.msgs_sent) and 2 <= self.moves_left():
            sent = self.__dict__.get('msg_log', None) or []
            if not (sent and sent[-1][2] == self.__dict__.get("cart_edits", 0)):
                proposal = self.message_call(self.basket_text(), 'mv_gate_basket_{}'.format(self.wire.tick))
                if proposal:
                    self.send_chunk([(proposal["id"], proposal["function"]["name"], json.loads(proposal["function"]["arguments"]))])
                    return 'Not run yet (no step used). Shopper saw the basket and its total beforehand; place the order immediately.'
        move = str((dec or {}).get('move', None) or 'continue').strip(None).lower()
        if move == 'tool' and str((dec or {}).get('tool', None) or "") == self.role("message"):
            move, dec = "message", (dict(move='message', text=next((v for v in ((dec or {}).get('args', None) or {}).values() if isinstance(v, (str,))), '')))
        if kind == 'decline' and move == 'decline' and str((dec or {}).get('text', None) or '').strip(None):
            self.mv_decline_text = str(dec['text']).strip(None)[:600]
        if move in ("continue", kind, ""):
            return ""
        if move == 'message':
            proposal = self.message_call(str(dec.get('text', None) or '').strip(None)[:1500], 'mv_gate_msg_{}'.format(self.wire.tick))
            if proposal and (self.msg_cap is None or self.msg_cap > self.msgs_sent) and 2 <= self.moves_left():
                self.send_chunk([(proposal["id"], proposal["function"]["name"], json.loads(proposal["function"]["arguments"]))])
                return 'Not run yet (no step used). The director first messaged the shopper; go through the response.'
            return ""
        self.mv_next = dec
        return 'Not run yet (no step used). The director has put a different move ahead of it: {}.'.format(move)

    def mv_engine(self, call_name):
        if call_name == self.role("search") or call_name == (self.mv_browse_tool() or ("",))[0]:
            return "search"
        if call_name in (self.role("cart"), self.role("cart_view")):
            return "cart"
        if call_name == self.role("message"):
            return "message"
        if call_name in self.finishers():
            return "finish"
        if call_name in (self.role("details"), self.role("view"), self.stock_tool()) or (self.batch_tool() or ("",))[0] == call_name:
            return "read"
        return "tool"

    def mv_round(self):
        mark = (self.needs.version,) + self.mv_info()
        rnd = self.__dict__.get('mv_rnd', None)
        if not rnd or rnd["mark"] != mark:
            free = max(0, self.moves_left() - self.order_due())
            products = max(1, len(self.needs.items()))
            caps = {eng: max(least, int(share * free)) for eng, (share, least) in MV_BUDGET.items()}
            caps["cart"] = 2 * products + 3
            rnd = dict(mark=mark, caps=caps, used={}, puts={})
            self.mv_rnd = rnd
        return rnd

    def mv_left(self, engine):
        if self.wire.mute:
            return 10 ** 6
        rnd = self.mv_round()
        return max(0, rnd["caps"].get(engine, 10 ** 6) - rnd["used"].get(engine, 0))

    def mv_room(self, engine, calls = 1):
        rnd = self.mv_round()
        if not engine in rnd['caps']:
            return True
        return rnd["used"].get(engine, 0) + calls <= rnd["caps"][engine]

    def mv_budget_text(self):
        rnd = self.mv_round()
        return ", ".join('{} {}/{}'.format(eng, rnd['used'].get(eng, 0), cap) for eng, cap in rnd["caps"].items())

    def mv_call(self, call_name, call_args, n = 0):
        return dict(id='mv_{}_{}'.format(n, self.wire.tick), type='function', function=dict(name=call_name, arguments=json.dumps(call_args)))

    def mv_holds(self, responses):
        for _, text in responses or []:
            if isinstance(text, (str,)) and text.startswith("Not run"):
                self.mv_note("held", text[:260])
                self.mv_held = self.__dict__.get("mv_held", 0) + 1

    def mv_finish(self, kind):
        call_name = self.role(kind)
        if not call_name:
            return False
        if self.asking and self.pose_planned(urge=True):
            return True
        text = SORRY if not kind == 'order' else 'Order the cart.'
        self.mv_holds(self.dispatch([self.mv_call(call_name, self.plain_args(call_name, text, kind))]))
        return True

    def mv_search(self, query):
        search = self.role("search")
        if not search or self.moves_left() < self.order_due() + 2:
            return False
        tpl = self.search_template()
        if not tpl:
            props = (self.shapes.get(search, None) or {}).get('properties', None) or {}
            key = self.text_arg(search) or next((k for k, spec in props.items() if isinstance(spec, (dict,)) and 'string' == spec.get('type', None) and (not spec.get('enum', None))), None)
            if not key:
                return False
            call_args = {key: query}
        else:
            call_args = {**tpl[0], tpl[1]: query}
        call_args, faults = ArgCheck.execute(call_args, self.shapes[search])
        if faults:
            return False
        self.send_chunk([('mv_search_{}'.format(self.wire.tick), search, call_args)])
        return True

    def mv_browse_tool(self):
        if not 'mv_btool' in self.__dict__:
            taken = {self.role(r) for r in ("search", "details", "cart", "cart_view", "order", "decline", "message", "view")} - {None}
            found = None
            for tdef in self.toolset:
                fn = tdef.get('function', None) or {}
                name = fn.get('name', None)
                if not name or name in taken:
                    continue
                props = (fn.get('parameters', None) or {}).get('properties', None) or {}
                cat = next((k for k, spec in props.items() if isinstance(spec, (dict,)) and spec.get('type', None) in ("string", None)
                            and re.search(r"class|section|categor|depart|range|aisle|shelf|group", str(k).lower())), None)
                text = (str(fn.get('description', None) or "") + " " + name).lower()
                if cat and re.search(r"search|list|item|browse|explore|shelf|find", text):
                    found = (name, cat, props)
                    break
            self.mv_btool = found
        return self.mv_btool

    def mv_browse(self):
        tool = self.mv_browse_tool()
        if not tool or not self.mv_room("search") or 3 > self.moves_left() - self.order_due():
            return False
        name, cat, props = tool
        done = self.__dict__.setdefault("mv_browsed", set())
        homes = self.home_shelves()
        rows = self.needs.items() or [dict(product=str(self.needs.reading.get('product', None) or ''))]
        for k, row in enumerate(rows, 0):
            if k in done:
                continue
            done.add(k)
            label = str(row.get('product', None) or '').strip(None) if not (k < len(homes) and homes[k]) else sorted(homes[k])[0]
            enum = None if not isinstance(props.get(cat, None), (dict,)) else (props.get(cat, None) or {}).get('enum', None)
            if isinstance(enum, (list,)) and enum:
                want = kind_words(label) | kind_words(row.get('product', None) or "")
                label = max(enum, key=lambda v: len(want & kind_words(str(v))))
            if not label:
                continue
            call_args: dict[str, Any] = {cat: label}
            ceil = next((key for key, spec in props.items() if isinstance(spec, (dict,)) and spec.get('type', None) in ("number", "integer")
                         and re.search(r"cost|price|ceiling|max", str(key).lower())), None)
            if ceil and is_number(self.needs.spend_cap):
                call_args[ceil] = int(self.needs.spend_cap) if not 'number' == props[ceil].get('type', None) else self.needs.spend_cap
            call_args, faults = ArgCheck.execute(call_args, self.shapes.get(name, None) or {})
            if faults:
                continue
            self.send_chunk([('mv_browse_{}'.format(self.wire.tick), name, call_args)])
            return True
        return False

    def mv_figure_queries(self):
        rows = self.needs.items()
        plain_kind = str(self.needs.reading.get('product', None) or '').strip(None)
        out = []
        for req in self.needs.props_left():
            head, _, val = str(req).partition(":")
            k = None if not rows else self.product_of(head)
            kind = (plain_kind if not (rows and (not k is None) and (k < len(rows))) else str(rows[k].get('product', None) or '')).strip(None)
            for m in re.finditer(r"(\d+(?:\.\d+)?)\s*([a-z][a-z.]{0,11})", val.lower()):
                if kind and not m.group(2) in kind.lower().split():
                    out.append('{} {} {}'.format(m.group(1), m.group(2), kind))
        return list(dict.fromkeys(out))[:3]

    def mv_figures(self):
        done = self.__dict__.setdefault("mv_fig_done", set())
        for query in self.mv_figure_queries():
            if (self.needs.version, query) in done:
                continue
            done.add((self.needs.version, query))
            if self.mv_room("search") and self.mv_search(query):
                return True
        return False

    def mv_unread_rows(self):
        if 3 > self.moves_left() - self.order_due() or not self.mv_room("read"):
            return []
        info = self.code_rows()
        homes = [h for h in self.home_shelves() if h]
        want = set()
        for row in (self.needs.items() or [dict(product=str(self.needs.reading.get('product', None) or ''))]):
            want |= kind_words(row.get('product', None) or "")
        cap = None if not is_number(self.needs.spend_cap) else float(self.needs.spend_cap)
        tried = self.__dict__.get("mv_tried", set())
        rows = []
        for ident, row in info.items():
            if ident in self.archive.read or ident in tried or row["stock"] is False or ident in self.refused_lines:
                continue
            on_kind = bool(want & row['words']) if not homes else any((row['shelf'] in h for h in homes))
            if not on_kind:
                continue
            price = self.archive.price(ident) if not is_number(row['price']) else row['price']
            if not cap is None and is_number(price) and cap < price:
                continue
            rows.append((0.0 if not ('cheapest' == self.needs.priority() and is_number(price)) else price,
                         -len(want & row["words"]), ident))
        return [ident for *_, ident in sorted(rows)[:4]]

    def mv_reads(self):
        tried = self.__dict__.setdefault("mv_tried", set())
        room = self.moves_left() - self.order_due() - 2
        want = [] if not room > 0 else [i for i in sorted(self.named_open()) if not i in tried][:3]
        if want:
            return want
        products = max(1, len(self.needs.items()))
        done = self.__dict__.setdefault("mv_read_n", {})
        if done.get(self.needs.version, 0) >= MV_READS + 3 * (products - 1):
            return []
        avoid: set[str] = set()
        for rank in range(0, 2 if not products == 1 else MV_DEPTH):
            picks = self.code_picks(avoid)
            if not picks:
                break
            for ident, _ in picks:
                avoid.add(ident)
                if not ident in self.archive.read and not ident in tried and (rank == 0 or MV_SPARE < room):
                    want.append(ident)
        return want

    def mv_read(self, ids = None):
        reader, batch = self.role("details"), self.batch_tool()
        if not (reader or batch) or self.wire.spare() < CLOSING:
            return False
        want = self.mv_reads() if not ids else list(ids)
        if not want:
            return False
        if not (batch and 2 <= len(want)):
            ids = want[:4]
            slab = [('mv_read_{}_{}'.format(n, self.wire.tick), reader, call_args) for n, call_args in enumerate((self.details_call(i) for i in ids), 0) if call_args]
        else:
            ids = want[:batch[2]]
            slab = [('mv_read_{}'.format(self.wire.tick), batch[0], {batch[1]: ids})]
        spec = self.spec_tool()
        if spec and slab:
            seen = self.__dict__.setdefault("spec_read", set())
            for n, ident in enumerate([i for i in ids if not i in seen], 0):
                call_args = self.lookup_call(ident, spec)
                if call_args:
                    slab.append(('mv_spec_{}_{}'.format(n, self.wire.tick), spec, call_args))
                    seen.add(ident)
        if not slab or self.moves_left() < len(slab) + 2 * max(1, len(self.needs.items())) + 3:
            return False
        self.__dict__.setdefault("mv_tried", set()).update(ids)
        done = self.__dict__.setdefault("mv_read_n", {})
        done[self.needs.version] = done.get(self.needs.version, 0) + len(ids)
        self.send_chunk(slab)
        return True

    def mv_cart(self, ids = None):
        if not self.role("cart"):
            return False
        if ids:
            for ident in ids:
                if not '{}|{}'.format(ident, self.needs.version) in self.passed and self.wire.spare() > CLOSING:
                    self.check_item(ident, "ADD TO THE CART")
            refused = [i for i in ids if not '{}|{}'.format(i, self.needs.version) in self.passed]
            if refused:
                self.mv_note("held", 'the cart choice made by the director has not had any item check pass: ' + ", ".join(refused))
                return False
            dearer = [] if not 'cheapest' == self.needs.priority() else [(i, self.cheaper_fit(i)) for i in ids]
            dearer = [(i, c) for i, c in dearer if c and not c in ids]
            if dearer:
                self.mv_note("held", 'the cart choice made by the director is pricier than a listing that passed the item check for that same product ('
                             + ", ".join('{} is cheaper than {}'.format(c, i) for i, c in dearer) + "): ask for the lowest-priced match and let the check's result stand; if the store lists an item in the product's category, it counts as that product regardless of what its title calls it")
                return False
            counts = self.needs.counts()
            picks = [(i, (1 if self.needs.items() else self.needs.quantity()) if not (len(ids) == len(counts) and k < len(counts)) else counts[k])
                     for k, i in enumerate(ids, 0)]
            extra = [line for line in self.trolley.units if not line in ids] if not self.needs.items() else []
        else:
            picks = self.code_picks()
            if not picks:
                return False
            extra = [line for line in self.trolley.units if not line in {i for i, _ in picks}]
        todo = [(ident, qty) for ident, qty in picks if 1e-09 < abs(self.trolley.held(ident) - qty)]
        puts_n = self.mv_round()["puts"]
        stuck = [ident for ident, _ in todo if puts_n.get(ident, 0) >= MV_LINE_PUTS]
        if stuck:
            self.mv_note("held", 'phrases used too frequently in this round, kept unchanged: ' + ", ".join(stuck))
            todo = [(ident, qty) for ident, qty in todo if not ident in stuck]
        extra = [line for line in extra if puts_n.get(line, 0) < MV_LINE_PUTS]
        if not extra and not todo:
            return False
        if self.asking and self.pose_planned(urge=True):
            return True
        if self.trolley.line_spec():
            todo = list(picks)
        drops = self.cart_drops(extra)
        puts = [] if not todo else self.cart_args(todo)
        if drops is None or (todo and not puts):
            return False
        self.mv_holds(self.dispatch([self.mv_call(self.role("cart"), call_args, n) for n, call_args in enumerate(drops + puts, 0)]))
        return True

    def mv_run(self, dec):
        move = str(dec.get('move', None) or 'continue').strip(None).lower()
        if move == 'tool' and str(dec.get('tool', None) or "") == self.role("message"):
            move, dec = "message", dict(text=next((v for v in (dec.get('args', None) or {}).values() if isinstance(v, (str,))), ''))
        if move == 'message':
            text = str(dec.get('text', None) or '').strip(None)
            proposal = None if not text else self.message_call(text[:1500], 'mv_msg_{}'.format(self.wire.tick))
            if not proposal:
                return False
            self.mv_holds(self.dispatch([proposal], scripted=True))
            return True
        if move == 'ask':
            return bool(self.asking and self.pose_planned(urge=True))
        engine = dict(search='search', read='read', cart='cart', tool='tool').get(move, None)
        if engine and not self.mv_room(engine):
            self.mv_note("held", 'the {} engine has no steps left in its budget for this round'.format(engine))
            return False
        if move == 'search':
            query = str(dec.get('query', None) or '').strip(None)
            if query:
                return self.mv_search(query[:200])
            tick = self.wire.tick
            self.code_search(max(1, len(self.needs.items())))
            return tick < self.wire.tick
        if move == 'read':
            return self.mv_read([str(i) for i in dec.get('ids', None) or [] if str(i) in self.archive.ids][:4] or None)
        if move == 'cart':
            return self.mv_cart([str(i) for i in dec.get('ids', None) or [] if str(i) in self.archive.ids] or None)
        if move in ("order", "decline"):
            return self.mv_finish(move)
        if move == 'tool':
            name = str(dec.get('tool', None) or "")
            if not name in self.shapes or name in self.finishers() or name == self.role("cart"):
                return False
            self.mv_holds(self.dispatch([self.mv_call(name, {} if not isinstance(dec.get('args', None), (dict,)) else dec.get('args', None))]))
            return True
        return False

    def mv_propose(self, blocked):
        ver = self.needs.version
        products = max(1, len(self.needs.items()))
        spare = self.moves_left() - self.order_due()
        picks = self.code_picks()
        if False and not picks and self.archive.read:
            self.mv_relax = True
            try:
                picks = self.code_picks()
            finally:
                self.mv_relax = False
            if picks:
                self.mv_note("held", "the code's value check rejected every listing; the item check receives the candidates in their place")
        fit = self.__dict__.get('code_fit', None) or {}
        fits = bool(picks) and all(fit.get(ident, None) for ident, _ in picks)
        for eng in ("search", "read", "cart"):
            if not self.mv_room(eng):
                blocked.add(eng)
        if not self.mv_room("search"):
            blocked.add("pages")
        if not 'search' in blocked and self.role("search") and not fits and self.__dict__.get('mv_searched', None) != ver and spare > 2:
            return "search", 'so far no candidate meets all the requirements'
        if not 'browse' in blocked and self.mv_browse_tool() and self.__dict__.get('mv_searched', None) == ver and any(
                not k in self.__dict__.get('mv_browsed', set()) for k in range(0, products)) and spare > 3 and self.mv_room("search"):
            return "browse", "browse the product's category using the store's tool that searches by fields"
        if not 'figures' in blocked and self.__dict__.get('mv_searched', None) == ver and spare > 3 and self.mv_room("search") and any(
                not (ver, q) in self.__dict__.get('mv_fig_done', set()) for q in self.mv_figure_queries()):
            return "figures", 'run a search with the kind word plus the figures the requirements measure'
        if not 'read' in blocked and self.mv_reads():
            return "read", "candidates' records not read yet"
        paged = self.__dict__.get('mv_paged', None) or {}
        if not 'pages' in blocked and self.role("search") and not fits and self.__dict__.get('code_more', None) and spare > 2 + products \
                and paged.get(ver, 0) < MV_PAGES:
            return "pages", "no candidate fits yet"
        if not picks:
            if not 'read' in blocked and self.mv_unread_rows():
                return "read", 'none of the listings examined match; listings of the requested type are still unexamined'
            return (None, '') if not not 'decline' in blocked else ('decline', 'the records hold nothing that can fit')
        want = {ident for ident, _ in picks}
        if not 'cart' in blocked and (any(not line in want for line in self.trolley.units)
                                      or any(1e-09 < abs(self.trolley.held(i) - q) for i, q in picks)):
            return "cart", "picks: " + ", ".join('{} x {}'.format(q, i) for i, q in picks)
        return (None, '') if not not 'order' in blocked else ('order', 'the cart holds the picks')

    def mv_exec(self, kind):
        if kind == 'search':
            self.mv_searched = self.needs.version
            tick = self.wire.tick
            self.code_search(max(1, len(self.needs.items())))
            return tick < self.wire.tick
        if kind == 'browse':
            return self.mv_browse()
        if kind == 'figures':
            return self.mv_figures()
        if kind == 'read':
            return self.mv_read() or self.mv_read(self.mv_unread_rows())
        if kind == 'pages':
            paged = self.__dict__.setdefault("mv_paged", {})
            paged[self.needs.version] = paged.get(self.needs.version, 0) + 1
            tick = self.wire.tick
            self.code_search(max(1, len(self.needs.items())), True)
            return tick < self.wire.tick
        if kind == 'cart':
            return self.mv_cart()
        if kind in ("order", "decline"):
            return self.mv_finish(kind)
        return False

    def mv_step(self):
        held = self.__dict__.get("mv_held", 0)
        state = (self.wire.tick, self.needs.version, len(self.archive.read), len(self.rulings), len(self.passed),
                 tuple(sorted(self.trolley.units)), self.msgs_sent, self.refusal_n, 24 if not held < 24 else held)
        if not state == self.__dict__.get('mv_state_key', None):
            self.mv_state_key, self.mv_idle = (state, 0)
        else:
            self.mv_idle = self.__dict__.get('mv_idle', 0) + 1
            if MV_IDLE <= self.mv_idle:
                self.log.append(dict(kind='mv_idle', step=self.wire.tick))
                return False
        if self.refusal_n > self.__dict__.get("mv_refusals_seen", 0):
            self.mv_refusals_seen = self.refusal_n
            lines = sorted(set(self.refused_lines) & set(self.trolley.units)) or sorted(self.refused_lines)
            if lines and self.moves_left() >= len(lines) + 3:
                try:
                    read, sold = self.code_recheck(lines[:3])
                except (Exception,) as err:
                    read, sold = False, set()
                    self.log.append(dict(kind='mv_recheck_error', error='{}: {}'.format(type(err).__name__, err)[:300]))
                if read:
                    self.refused_lines.difference_update(set(lines[:3]) - set(sold))
                    self.mv_note("held", "refused lines re-read: " + ", ".join(lines[:3]) + ('; still in stock' if not sold else '; sold out: {}'.format(', '.join(sorted(sold)))))
                    return True
        queued = self.__dict__.pop("mv_next", None)
        if isinstance(queued, (dict,)):
            if self.mv_run(queued):
                return True
            self.mv_note("held", "the director's action '{}' failed to go through; choose again".format(queued.get('move', None)))
            self.mv_gate_key = None
        blocked: set[str] = set()
        for _ in range(0, 4):
            kind, why = self.mv_propose(blocked)
            if not kind:
                return False
            group = ('find' if not kind in ('order', 'decline') else 'finish') if not kind == 'cart' else 'cart'
            turning = (group == 'cart' and 'cart' != self.__dict__.get('mv_group', None)) or (
                group != 'finish' and self.mv_info() != self.__dict__.get('mv_seen', None))
            if turning:
                dec = self.mv_direct('TURNING POINT: the code is about to {}'.format(MV_KINDS[kind]), kind, why)
                if dec and not str(dec.get('move', None) or 'continue').strip(None).lower() in ('continue', kind) and self.mv_run(dec):
                    self.mv_group = None
                    return True
            if self.mv_exec(kind):
                self.mv_group = group
                self.log.append(dict(kind='mv', move=kind, step=self.wire.tick))
                return True
            blocked.add(kind)
        return False

    def tool_calls(self, msg, seq_no):
        proposals = []
        for proposal in msg.get('tool_calls', None) or []:
            if not isinstance(proposal, (dict,)):
                continue
            seq_no = seq_no + 1
            func = proposal.get('function', None) or {}
            arguments = json.dumps(func.get('arguments', None) or {}) if not isinstance(func.get('arguments', None), (str,)) else func.get('arguments', None)
            proposals.append(dict(id=proposal.get('id', None) or 'call_{}'.format(seq_no), type='function', function=dict(name=func.get('name', None), arguments=arguments)))
        return proposals, seq_no

    def execute(self):
        if self.wire.done:
            return self.log
        self.read_request()
        self.log.append(dict(kind='plan', plan=self.needs.reading, roles=self.role_map))
        opener_obs = self.pview.get('opening', None)
        opener = "SHOPPER REQUEST:\n" + self.request
        if not opener_obs is None:
            opener += "\n\nOPENING OBSERVATION:\n" + as_text(opener_obs)
            self.archive.add(opener_obs)
        overview = self.needs.overview(self.step_cap, self.msg_cap)
        if overview:
            opener += "\n\n" + overview
        opener += "\n\n" + self.status_line()
        self.chat = [dict(role='system', content=ACT_RULES), dict(role='user', content=opener)]
        try:
            plan = self.mv_direct('PLAN: ahead of the initial action; locate all the procedures and limits, plus the messages that need to stay unused')
            if plan and 'continue' != str(plan.get('move', None) or 'continue').strip(None).lower():
                self.mv_next = plan
        except (Exception,) as err:
            self.log.append(dict(kind='mv_plan_error', error='{}: {}'.format(type(err).__name__, err)[:300]))
        idle_n = fails = seq_no = 0
        turn_ok, missed = monotonic(), 0
        for _pass in range(0, 2):
            for _ in range(0, TURN_LIMIT):
                if self.wire.done or self.wire.broke or 0 >= self.moves_left():
                    break
                if self.wire.spare() < LAST_CALL:
                    break
                stepped = False
                tick_mark, chat_mark, ask_mark = self.wire.tick, len(self.chat), self.wire.__dict__.get("asked_n", 0)
                if 3 > self.__dict__.get('helper_errors', 0):
                    try:
                        self.catch_up()
                        if self.needs.reply_due:
                            text_body, self.needs.reply_due = self.needs.reply_due, ""
                            stepped = self.reply_back(text_body)
                        stepped = stepped or self.open_named() \
                            or ((not self.asking or bool(self.__dict__.get('empty_plans', None))) and self.sweep_step()) \
                            or self.value_pages() or self.depth_step() or (self.asking and self.pose_planned())
                        if not stepped:
                            self.gate_rules_step()
                            stepped = self.ref_step() or self.walk_step() or self.ladder_step() or self.value_search_step() \
                                or self.value_walk_step()
                    except (Exception,) as err:
                        self.helper_errors = self.__dict__.get("helper_errors", 0) + 1
                        self.log.append(dict(kind='step_fault', error='{}: {}'.format(type(err).__name__, err)[:300]))
                        stepped = False
                        if self.wire.done or self.wire.broke:
                            break
                if stepped and tick_mark == self.wire.tick and self.wire.__dict__.get("asked_n", 0) == ask_mark:
                    self.idle_helpers = self.__dict__.get("idle_helpers", 0) + 1
                    if self.idle_helpers >= 3:
                        self.log.append(dict(kind='idle_helpers', tick=self.wire.tick))
                        self.idle_helpers, stepped = 0, False
                elif stepped:
                    self.idle_helpers = 0
                if stepped:
                    continue
                queued = self.__dict__.pop("mv_next", None)
                if isinstance(queued, (dict,)):
                    try:
                        if self.mv_run(queued):
                            continue
                    except (Exception,) as err:
                        self.log.append(dict(kind='mv_error', error='{}: {}'.format(type(err).__name__, err)[:300]))
                    self.mv_note("held", "the director's action '{}' failed to go through; choose again".format(queued.get('move', None)))
                    self.mv_gate_key = None
                model_list = ACTOR[1:] + ACTOR[:1] if not STUCK_LIMIT > self.stuck else ACTOR
                msg = self.wire.query_model(self.chat, model_list, 2500, self.toolset)
                if msg is None:
                    missed = missed + 1
                    if not self.wire.usable(model_list, True) or (monotonic() - turn_ok > DOWN_AFTER and (
                            missed >= 2 or (self.wire.down() and 2 <= len(self.wire.call_models)))):
                        break
                    if self.wire.call_fail in ("empty", "refused"):
                        fails = fails + 1
                        if fails >= 2:
                            break
                    continue
                turn_ok, missed = monotonic(), 0
                proposals, seq_no = self.tool_calls(msg, seq_no)
                content = msg.get('content', None) or ""
                if not isinstance(content, (str,)):
                    content = as_text(content)
                binding_moves = {self.role("cart"), self.role("order"), self.role("decline")} - {None}
                if self.asking and any(c["function"]["name"] in binding_moves for c in proposals) and self.pose_planned(urge=True):
                    continue
                if not proposals:
                    idle_n = idle_n + 1
                    self.chat.append(dict(role='assistant', content=content or '(no tool call)'))
                    self.chat.append(dict(role='user', content='Each reply has to invoke at least one of the provided tools. Go on.\n' + self.status_line()))
                    if IDLE_LIMIT <= idle_n:
                        break
                    continue
                idle_n = 0
                self.log.append(dict(kind='model', content=content[:2000], calls=[c['function'] for c in proposals]))
                self.chat.append(dict(role='assistant', content=content or None, tool_calls=proposals))
                step_before = self.wire.tick
                self.post_replies(self.dispatch(proposals))
                self.stuck = 0 if not step_before == self.wire.tick else self.stuck + 1
            if _pass or self.wire.done or self.wire.broke or self.wire.spare() < CLOSING + 60 or self.moves_left() <= self.order_due() + 4:
                break
            self.trace_ping("loop ended early")
            self.log.append(dict(kind='resume', tick=self.wire.tick, spare=round(self.wire.spare(), 1)))
            self.wire.mute = False
            self.wire.banned.clear()
            self.wire.tool_banned.clear()
            self.stuck, idle_n, fails, missed, turn_ok = 0, 0, 0, 0, monotonic()
            self.chat.append(dict(role='user', content='The task is incomplete: there are still steps and time left. Keep going with the tools.\n' + self.status_line()))
        if self.wire.spare() > CLOSING + 60:
            self.trace_ping('loop ended early' if not self.wire.done else 'ended early')
        if not self.wire.done and not self.wire.broke:
            self.finish_now()
        try:
            kinds: dict[str, int] = {}
            for entry in self.log + list(self.wire.log):
                if isinstance(entry, (dict,)):
                    kinds[str(entry.get('kind', None))] = kinds.get(str(entry.get('kind', None)), 0) + 1
            tally = (dict(kind='engine_tally', kinds=kinds, ladder_why=dict(self.__dict__.get('ladder_why', None) or {}), vl_why=dict(self.__dict__.get('vl_why', None) or {}), plan_why=dict(self.__dict__.get('plan_why', None) or {})))
            self.log.append(tally)
            if not self.wire.no_model and 4 < self.wire.spare():
                was = self.wire.mute
                self.wire.mute = False
                try:
                    self.wire.query_model([dict(role='user', content='TRACE TALLY (reply ok): ' + json.dumps(tally)[:3000])], CHECKER, 5)
                finally:
                    self.wire.mute = was
        except (Exception,):
            pass
        self.log.append((dict(kind='end', finished=self.wire.done, steps=self.wire.tick, cost=round(self.wire.spend, 5), seconds=round(monotonic() - self.wire.t0, 1))))
        return self.log

def _trace_text(log, why):
    rows = []
    for item in log[-30:]:
        try:
            rows.append(json.dumps(item, default=str)[:240])
        except (Exception,):
            rows.append(str(item)[:240])
    return "DEBUG ({}) - answer with just the word ok. Here are the latest entries in the assistant's own log:\n".format(why) + "\n".join(rows)

def agent_main(pdata):
    errand = Errand(pdata)
    try:
        return errand.execute()
    except (Exception,) as err:
        errand.log.append(dict(kind='crash', error='{}: {}'.format(type(err).__name__, err)[:500]))
        try:
            if not errand.wire.done and not errand.wire.broke:
                errand.finish_now()
        except (Exception,) as second_err:
            errand.log.append(dict(kind='crash', error='{}: {}'.format(type(second_err).__name__, second_err)[:500]))
        return errand.log

__all__ = ["agent_main"]
