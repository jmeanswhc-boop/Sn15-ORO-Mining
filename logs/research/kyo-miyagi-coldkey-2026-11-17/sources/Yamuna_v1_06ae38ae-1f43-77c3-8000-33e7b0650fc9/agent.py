from __future__ import annotations

import json
import re
from functools import partial
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
RANKER = ["anthropic/claude-haiku-5.5", "anthropic/claude-sonnet-5.5"]

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
PUT_WORDS = set("add put hold set insert place keep include in plus increase append".split())
DROP_WORDS = set("remove drop release delete take out clear discard minus decrease subtract off unhold".split())
FAULT_KEYS = r"error|fault|refus|reject|denied|invalid|fail"
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
LINK_WORDS = set("and or the for with of a an in on to by from at as is are be this that it its your our you all any not no".split())
ASK_AFTER_SETS = 2
ASK_GRACE_STEPS = 3
ASK_AFTER_STEPS = 5


ACT_RULES = """Context: You shop for a single shopper inside a simulated store, and the given tools are your only means of acting there. Grading is all or nothing: 1 only if the finish is correct on every count, else 0. Correct means the item ordered satisfies each written requirement, each hidden need the shopper discloses on being asked, and each change the shopper announces as you go; quantity and budget are correct; when the request wants the cheapest (or best) item, that is the one picked; and every procedural rule the request sets is followed. If no listing satisfies all of it, the correct finish is a decline carried out precisely the way the request describes.

Your job: drive the errand to a correct order (or, only when nothing fits, a correct decline) through tool calls.

Constraints:
- How this store behaves:
  - Every tool call uses up one step, including calls that fail or are invalid, and the step supply is finite. Reasoning is free.
  - A single reply can hold several tool calls as long as none depends on the output of another (opening the details of a few listings, say); they execute in sequence. An order or a decline is always sent on its own.
  - If a details tool accepts a LIST of listing ids, it returns the full records of all of them for ONE step (within its item limit); use it whenever you open two or more listings.
  - Tool names, argument names and result shapes differ in every store. Study each tool's description and JSON schema and use exactly the names and permitted values given. From the first results, work out which tool reveals which facts (price, stock, detailed properties).
  - Find out early which category labels this store gives to listings of the kind requested. If similar labels sit side by side, pick the one whose label fits the wording of the request taken as a whole, and do your searching and buying within it.
  - Search behaviour varies by store; infer this one's from its first results. Rows can arrive best match first or in a fixed order by id; a keyword search can demand every word (an extra word makes the total smaller) or accept any word (an extra word makes the total bigger, and listings hitting more of the words rank above the rest); perhaps only a few rows or pages can be seen; a filter may only work in one mode (a keyword mode can ignore the category or price filter, a filter mode can ignore the words). Set any reported total against the rows in view: if the total is bigger, the fitting or cheapest listing might be on a later page or past the visible ones. If every word is demanded, tighten with more precise words (or with the category and price filters and a lower price limit) until the total fits within the visible pages. Use short queries (one to three words) and put the most telling words FIRST: some stores look only at the opening word or two and drop the rest, so a long query might only search its first words (set a longer query's total and rows against a shorter one that begins the same way; the notes say what they reveal). If every word is demanded, a required value together with the item kind ("blue chair", value first) returns exactly the listings bearing that value. If any word is accepted, generic words (the item kind, a unit) hit most of the shelf and hide everything else: the required value's own distinctive word or figure on its own ("herringbone", "30m", "0.4") returns only the listings bearing it, so search it alone and in the listings' other spellings ("30 m", "3 atm"), read all pages of that search when its total fits on them, and combine the item kind with the values only to bring the nearest matches to the top. Keyword searches can also hit category names and property values. The notes after every search tell you what the totals so far reveal about how this store searches.
- The shopper and hidden needs:
  - Some needs were never written down; each comes out only when a message asks about that exact property. The shopper answers the FIRST question in a message most dependably, so a message asks about a single property, or at most two with the likelier one first: a brief, plain, friendly question naming the property (one or two sample values are fine). Leave out instructions, lists, labels, positions and rules.
  - Replies like "Don't mind", "no preference" or "anything is fine" leave the property open. A preference or limit the shopper states becomes a hard requirement from that point, word for word, figures included.
  - If questions are being planned for you, they are sent once you have opened the details of two or three listings of the requested kind, so open those among your first moves. The replies show up in the conversation together with what they mean; every revealed need binds just like a stated one: search again with it next to the stated values.
  - If you do the asking yourself: open the details of two or three listings first, then ask about what the story suggests (who the item is for, how it should look, where or how it gets used), then about properties the listings carry that vary between candidates. Do not ask about anything the request already decides, and do not exceed the message budget. Your own messages are the way you perform the steps the instructions call for (a confirmation, the basket, the total, a reply to the shopper): you may use the message tool whenever the allowance still has a message. As long as hidden-need questions are planned for you, let the plan do the asking (it chooses properties from the listings you opened) and save your own messages for what the instructions require of you; before an order, if you have not told the shopper the basket since your last cart change, it is told to them for you.
- Rules of procedure:
  - The request can lay down working rules: how many listings to view or compare, what to verify, what to tell the shopper, what a remark has to say, how to decline. These are graded just like requirements; carry out each one literally and completely. To "Look at" or "open" N listings means N listings that fit fully, each one opened with the details tool.
  - If the shopper puts a question to you (such as confirming a change), it gets answered for you while messages are left; the change itself is binding.
  - The status line shows the INSTRUCTIONS IN FORCE: the rules of the request plus any instruction the shopper or the store adds later (confirm with me before buying, tell me the total, a new allowance of messages or steps). They may change at any moment, so after every reply, notice or refusal re-read them with the requirements and the cart, and pick the next move from what they say now, not from routine. Any stated step left undone when you finish is performed for you just before the finish is sent, and one message is held back for it; a message you send yourself after the last cart change also counts.
  - Things the shopper would only like ("ideally", "would be nice", "if possible") are not requirements: they merely settle ties among items that already meet everything, and they never override a cheapest rule.
  - If the request leaves open something only the shopper can decide (no price limit, for instance, or two wishes that may clash so the shopper must say which weighs more), ask before choosing.
- Changes along the way:
  - The shopper may alter things: a new budget (up or down), a requirement removed, swapped or added, a price increase, an item going out of stock. A single line can hold several changes. Lines like these can turn up inside any tool result (an extra field containing sentences) or come back with an order attempt. Every change applies from that moment: check the pick and the backups again, and if the cheapest or best is wanted, choose afresh among the new acceptable set.
  - An order attempt may be REFUSED: the episode continues and the cart stays as it was. Read the refusal, apply each change it mentions, remove from the cart anything that no longer fits (a sold-out item, for example), add the next verified fit, re-check its stock and price, and order again.
  - Items may also go out of stock or change price unannounced, so stock and price get re-checked just before each order attempt; an item whose present price times quantity exceeds the present budget no longer fits.
- Several products:
  - A single request can name a number of distinct products, every one carrying its own quantity, requirements and hidden needs. Cover them all: look through each product's category, confirm a fit for each, then place them together in one order as one line per product at its quantity. Some cart tools take a list of lines and may swap the entire cart for that list, so every such call has to carry the lines of all products. Cheapest or best is decided product by product.
  - Sometimes a value is given by reference to another listing or page ("a colour identical to the jumpsuit sold by X"): track down that listing before anything else (search for the words that identify it, e.g. its shop or brand together with its kind), read its record, copy the value out of the field it names, and only then search for the wanted product using that value.
- The cart:
  - Instead of setting a count, the cart tool might ADD units onto a line it holds already: one more of a held line turns into 2. Never re-add a line that is already there; adjust units by removing or adding just the gap. After each cart call a note tells you how many units the code believes are in the cart.
- Judging a listing (the score depends on this):
  - For every requirement the grader looks at the listing field belonging to that property and compares the value in this store's own wording there. Any field whose name denotes that same property qualifies, however written: other separators, a prefix, a shortened or lengthened name, or a familiar substitute name ("fit" for a pants fit type, "neckline" for a collar type, "dial size" for a watch dial size, "color family" for a colour). A field carrying the exact property name takes precedence, as does that name preceded by one more word (pants length for length, clothing material for material); should either of them contain the value, the requirement holds. Names that differ by separators alone (size class, size_class) are treated as a single field, satisfied when either holds the value. Without such fields, the substitute field governs; and only when no field at all names the property can the options list or the title supply the value. Descriptions are never proof.
  - Matching goes by words in this store's spelling: letter case, separators, singular or plural and variant spellings make no difference; something narrower that includes the wanted value counts ("80% cotton" for cotton, "forest green" for green, "rtx 3060 12gb" for rtx 3060); being one entry in a list is sufficient. Synonyms, abbreviations, related terms and shades ("tech wood" for technical wood, "win11" for windows 11, "embroidered" for embroidery, "solid" for plain, ivory or cream for off-white) count only while no listing in the category puts the value itself in that field: once others write "matte" for their finish, "satin" no longer counts as matte. Two values joined by a plus ("wool + nylon") do not count for one part if the field elsewhere holds that other part alone ("nylon"). FIELD EVIDENCE lines, when shown, come from this store's records and settle which of these cases you face. Another value of the same property never counts (quartz is not mechanical, rosewood is not maple), nor does a lookalike word. Catch-all values ("suitable for all occasions", "all seasons", "universal") name no particular value and do not count, though a list that names the wanted value as well does. Nor does a different value count because the two tend to go together or one hints at the other (a "pullover" style is not "zip-up", "slip-on" is not "lace-up"); apart from a contained narrower value, only a shade commonly sold under the wanted colour qualifies.
  - Fields about some other property never substitute for this one (lining material is not material, length is not style, strap length is not strap type) whenever the category's listings have a field of the property itself.
  - Reject listings only over what the shopper said or revealed. Guesses drawn from the story or description (age group, gender, size, occasion, condition, quality, use) are never grounds; the story is only background, and if something there looks important, put it to the shopper as a question. What you ask is not a requirement; what the shopper answers is.
  - Compare figures as amounts once units are converted (4.5 cm is 45 mm); a limit's edge is inside it; a range qualifies only when every part of it honours the limit. When the title is the only evidence because no field names the property, an exact wanted value holds if the title gives its words and figures in that order, extra detail after them allowed (9' x 12' 4" shows 9' x 12'; 9' x 13' does not).
  - Title figures ("55 x 94 Inch", "117 x 228 cm") never say which dimension is which, so they are no reason to drop a listing or leave it unopened; look at the fields of its details instead.
  - Exclusions ("leave out anything with X") and closed lists ("A or B and nothing else") must be honoured.
  - Category membership by itself makes a listing the requested kind: to the grader anything filed under the requested category IS that kind even if title or description speak of another object (a container, a case, a refill, a set, an accessory), so do not pass over, refuse or downgrade it for its apparent nature; assess only requirements, price and stock. No property must echo the category name. Where the store labels its categories, the item must belong to the label matching the request's full wording (a request for "A or B" goes to a category "A & B", not to a narrower neighbour "B"). If the request wants goods on hand, only an in-stock listing will do, and price times quantity must stay inside the budget (equal is fine). Prices are in the budget's unit unless the record states another (45 against a budget of 1000 US cents is 45 cents); rescale neither.
  - Declining is rarely right. A listing that misses the literal wording yet meets every requirement by these rules (narrower value, other name of the same field, or a synonym or shade where no category listing writes the value itself in that field) is the fit; decline only when the search genuinely found nothing meeting all requirements by these rules.
- Cheapest or best:
  - Cheapest means the lowest price among ALL items that meet everything; a single fitting item does not settle it. Where keyword search demands all words and matches property values, the value-and-kind search returns the candidates holding that value: go through every page and open them in price order; a price-limited category listing reveals prices across the category but not values: use it to learn which cheaper rows exist (drop the limit until the total fits), then inspect their properties. Query each wanted value plus the item kind briefly, value leading ("blue chair", "30m watch"), and read each page when the total fits within readable pages (otherwise add a second wanted value or a tighter kind word); do likewise with alternative kind names ("stool", "bench") and with the bare value; for a value that cannot be searched (a figure the store strips), also search the product line of every fit or near-fit found, as its maker plus one distinctive title word, all pages: sibling sizes and colours of a line are separate listings with the same property values and can be cheaper; rank visible candidates by price on your own and open details from the cheapest upward; the first one whose property fields meet everything, after every cheaper one was opened and turned down, wins. Keep the following fits as backups.
  - Best by a measure (most, largest, longest, highest...) means comparing ALL fitting items on it likewise and choosing the extreme. Items giving no figure for the measure cannot win.
- Finishing:
  - Hold a ranked set of verified fits including a backup or more; the top choice may be sold out by checkout.
  - Prior to any order attempt: one cart line per requested product (a single line for a single product), each the chosen item at the quantity required, nothing extra; and after your final cart change you looked again at the chosen item's stock and price and found it available and affordable.
  - A decline comes only after a full search turns up nothing meeting every requirement, done strictly by the request's decline rule. When substitutes must not be offered, the decline remark mentions, describes or hints at no listing at all: no ids, titles or prices.
  - Ration steps: the first order attempt must go while at least seven steps are left, leaving room to handle a refusal. At 7 steps left only cart, stock-check and finishing tools stay usable. Make the decision once; no swapping the cart to and fro.

Answer with: at least one tool call in every reply. Keep any text brief: the requirements as they stand, the verified fits with price and stock, and the next move."""

READ_REQUEST = """Context: below are a shopper's request to a store assistant and the store's tool list.
Your job: derive a shopping plan from both, as ONE compact JSON object (unindented), and nothing more.
Constraints:
- "product": the wanted product kind, phrased as the request phrases it; if more than one product is wanted, list them all, linked by "and".
- "items": one entry per DIFFERENT product to purchase, in request order: {"product": its kind as the request phrases it, "quantity": how many units of one of its listings (see "quantity"), "requirements": its firm requirements (see "requirements"), "depends_on": the 1-based index of the entry this product must work with, fit, suit or be compatible with per the request, or else null}. One product, one entry. A kind written as two nouns joined by "and" or "&" that share a single count and a single set of requirements, with one count before them or none ("2 lamps and lighting fixtures in brass", "find me pots & pans that are dishwasher safe"), is the name of one store category: it makes one entry, not two.
- "quantity": an integer, the unit count to buy of one listing; 1 unless the request wants several. A thing sold whole, e.g. a pair of shoes or a set, is 1. The count goes up only when the shopper plainly wants multiple units ("I need three of them", "order 2 packs"); a number of users or sharers, or of previous ones, is background; it stays 1.
- "requirements": a string list, one entry for every firm product requirement, shaped "property: constraint" and carrying the shopper's literal value, bound, exclusion or closed list (for several products, all of them, each shaped "property (product): constraint"). Omit price, stock, quantity, budgets, simple wishes, and never repeat the product kind or category: words that form part of the category's name (an audience, a season or an occasion in it) are covered there, not listed apart. An entry pairs a product property with the value it is required to have ("the pattern has to be herringbone", "with a mechanical movement", "at least 4 drawers", "the occasion has to be party"). When the shopper insists on some aspect but never names the value it must have ("the cut matters most to me", "I always look at the sole"), that is no requirement, since no listing can be checked against it: put the aspect in "hints" only. Background never qualifies: the person it is meant for, an event where it gets worn or used ("to wear at his wedding", "for the staff party"), the place, time or manner of use, how many people share it, or the fact that an old one is replaced ("a new watch" means a replacement, not a condition). An occasion, use or audience qualifies only when the request turns it into the item's own property ("the occasion has to be business", "a blazer made for business"). Values must be words or figures the request gives as the item's value; a quality, motive or wish taken from the background (natural and breathable, slim enough for a case, works with what I already have, covers the whole drop) is a hint and can never be a requirement.
- "price_cap": the maximum total price, copied as the request's figure in the request's unit with no conversion ("3050 US cents" yields 3050); null if absent.
- "currency": the request's price unit, or null.
- "in_stock_only": true if the request will accept nothing but items on hand or in stock.
- "priority": "cheapest" just when the request openly wants the least expensive fitting item; "best" just when it openly wants the top item by some measure; "none" in all other cases. Saying the money is limited or that every penny counts does not ask for the cheapest; it merely accounts for the cap.
- "priority_detail": for "best", the measure used; for anything else, "".
- "message_budget": the integer count of messages the shopper permits you to send, or null if unstated.
- "step_budget": the integer count of moves or steps the request permits, or null if unstated.
- "rules": all procedural rules the request lays down for the assistant (the number of listings to inspect, what to verify or report, the way to decline, things to avoid, things to include), quoted verbatim. What the shopper intends to do personally is background, not a rule.
- "soft_wants": things the shopper would enjoy but does not strictly require (ideally, preferably, would be nice, if possible); never in "requirements".
- "instructions": those procedural directions a second time, each as an object: {"text": the direction as worded in the request, no longer than 20 words, "before_finish": true if it is due before or at the finish (ahead of an order or a decline: confirming with the shopper first, reporting the basket's contents or the total, a parting remark, one last check), false if not, "needs_message": true if doing it requires a message to the shopper, false if not, "before_change": true if it requires asking the shopper before something already in the cart is changed, replaced or removed, false if not}. Directions about how to end or what to say on ending count too. Empty if the request has none.
- "ask_first": things the shopper alone can settle, to ask before picking: for instance the budget if no limit appears, which of two wishes counts more when the request allows that both may be impossible (both wishes then belong in soft_wants, not requirements), or whatever the request wants cleared with the shopper. Empty if nothing applies.
- "hints": background details that point toward a product property without giving its value (what it is used for or with, where, by whom, what it has to fit, match or withstand), each "property (product): the detail in a few words" ("property: detail" when only one product is wanted), with the property named the way listings would probably name it. If a detail might bear on two properties, give the likelier one first and the other after a slash ("colour / style (chair): must suit a room with one painted wall"): anything the item has to match, suit or go well with (a room, a wall, an outfit, items used alongside) and its look point first to colour, next to pattern or style. Drop details that merely justify a requirement already given (why it needs its stated material or size). Weather, work, timetables, trips, or places where it is never worn or used are background, never hints; likewise any purpose that all listings of the kind fulfil (storage for a dresser, warmth for a coat). Empty if there are none.
- Each request sentence lands as a requirement, the priority, a soft want, a rule, a budget, a hint, or simple background. Doubtful rule sentences go under "rules".
- "references": requirements whose value is stated only by pointing at ANOTHER listing or page ("a colour identical to the jumpsuit sold by X", "within 5 of the figure on the page for Y", "a colour other than the one on Z's page"), each as "property (product): as shown on <how the request names that listing>" (or "property (product): different from the one shown on <how the request names that listing>" if the value must not match that listing's); every one is also entered in "requirements" in that same shape. Empty if there are none.
- "reference_targets": per "references" entry, the listing pointed at: {"property": the property concerned, "product": the product concerned, "seller": that listing's shop or brand as the request writes it, verbatim, "kind": the product kind of that listing, "details": [any further values the request mentions for that listing, brief and in the request's wording ("blue", "4gb", "net weight 167.262186 g")], "relation": "same", "differ" (the value may NOT equal that listing's: set apart from, different from, other than), "within", "at most" or "at least" (how the value compares with that listing's), "tolerance": the amount for "within", or else null, "unit": its unit, or else ""}. Empty if there are none.
- "decline_no_names": true if the request's rule for declining bans offering, naming or suggesting substitutes or alternatives; false if it wants something named; null if it sets no such rule.
- "roles": an object mapping each role to the exact listed tool name, or to null if no tool serves it: "search", "details", "view", "message", "cart", "cart_view", "order", "decline". "message" talks to the shopper; "cart" puts items in or takes them out; "order" purchases or checks out; "decline" ends the errand with no purchase.
Answer with: one JSON object with all these keys.
"""

HEAR = """Context: a shopping assistant must learn what the shopper's newest lines change.
Your job: read those lines and report their effect as ONE JSON object, with no other text.
Constraints:
- "meaning": a brief sentence on the shopper's intent.
- "needs": requirements on the product that are new and firm, written "property: constraint", copying the value, bound, exclusion or closed list exactly. Following a comparing phrase (falls short of, nothing less than, at least), a worded value (not a figure) counts as the value as such, in the shopper's spelling: "falls short of not available" yields "property: not available" (a field can hold "not available" or "none").
- "no_preference": properties the shopper declares they are indifferent to.
- "dropped": requirements from before that the shopper now gives up.
- "new_budget": the fresh price limit covering the entire order, a number expressed in the budget's own unit (take the figure the line gives when it uses that unit: "13075 cents" yields 13075 for a cents budget), or null.
- "approval": "yes" if the lines accept the basket or order the assistant put to them (agree, go ahead, fine, place it), "no" if they turn it down or ask to hold off or change something first, "" if they do neither.
- "new_quantity": when the lines change how many units are wanted, one object per product affected: {"product": the product as the lines name it ("" when one product is ordered), "count": the new number of units}; [] when no count changes.
- "unavailable": true if the shopper reports that some item can't be purchased any more (sold out, gone, out of stock, button disabled); false if not.
- "meta_only": true if all the line does is ask for fewer questions per message (one at a time, less in one go, the first one now and others afterwards) without giving any value: such a line answers nothing so far, adds nothing to needs, no_preference or dropped, and its figures of speech carry no value; false if not. A line that merely says the question was not followed or understood, or wants it repeated shorter or more simply, is NOT meta_only.
- "requirements_now": every firm product requirement in force after these lines, as "property: constraint": take REQUIREMENTS SO FAR as the base, retain whatever was neither dropped nor replaced, for a replacement retain only the latest value, and append each fresh need stated in the shopper's lines. Never bring in anything from the request's story that REQUIREMENTS SO FAR is missing; additions come solely from what the shopper writes. Leave out price, stock, quantity and the product category.
- "reply": if the shopper puts a question to the assistant or wants something confirmed, a short plain answer to return that states the substance of the change (for instance confirming the new value); if not, "".
- "instructions": procedural directions these lines give the assistant for the rest of the errand (ask me before buying, let me know the basket or the total, do or skip something, how to end or what to say at the end), each as an object: {"text": the direction as the lines phrase it, "before_finish": true if it is due before or at the finish (an order or a decline), false if not, "needs_message": true if doing it requires a message to the shopper, false if not, "before_change": true if it requires asking the shopper before something in the cart is changed, replaced or removed, false if not}; [] if the lines contain none. Needs, budgets and dropped requirements do not count as directions.
- "allowance": if the lines impose a fresh cap on the assistant's messages, questions, steps, moves or calls: {"messages_total": the newly permitted overall number of messages, or null, "messages_left": the count of messages still permitted from this point, or null, "steps_total": the newly permitted overall number of steps, or null, "steps_left": the count of steps remaining from this point, or null}; null if no such cap appears.
- "references": one object for each need in "needs" whose value is given only by pointing to some other listing or page, describing that listing: {"property": the property concerned, "seller": the shop or brand the line names for it, copied verbatim, "kind": what product kind it is, "details": [any further values the line mentions for it, brief and in the line's wording], "relation": "same", "differ" (the value may NOT equal that listing's: set apart from, different from, other than), "within", "at most" or "at least", "tolerance": the amount for "within", or else null, "unit": its unit, or else ""}; [] if there are none.
- A line may hold several changes; list each. A need about a different property, related or not (seat height versus overall height, strap width versus width), is an addition that removes nothing: a past requirement counts as dropped or replaced only if the line states that about the identical property. A replacement appears twice: the old value in dropped, the new value among the needs.
- If the shopper ranks two wishes, the favoured one becomes a need. Any budget stated goes to new_budget. A mere hope ("ideally", "would be nice") does not make a need.
- Read the shopper at face value and generously: homely, metaphorical or indirect wording that names a value means that value is firmly wanted; a value is ruled out only on a clear instruction to avoid it. Emphasis words showing how firmly a value is held (only, nothing but, nothing short of, strictly, stays, sticking with, locked to, holding out for) are not part of the value: give the bare value ("material: jute", "color: white"), never "<value> only"; add an exclusion only when the shopper also explicitly rejects blends or other values.
- A number plus a comparison (at most, at least, under, over, no more than) makes a limit and keeps its unit. A fresh budget overrides the previous one whether it rises or falls, even if the shopper presents it as good news.
- If the shopper merely says they don't mind, have no preference, or that anything is fine, put the asked-about property in no_preference (just the FIRST one if the message raised two). An answer that merely says the question was hard to follow ("that was rather dense", "you've lost me", "could you put it another way") is how this shopper signals no preference, provided the question named the product owning the property: then that property also goes in no_preference; with several products wanted and no product named in the question, the property remains open (no_preference gets nothing).
- With several products, phrase every need "property (product): constraint", with the product taken from the line (a line on "toe style of the women's boots" yields "toe style (women's boots): ...").
- Questions the assistant itself asked never become requirements; only the shopper's words do, and sample values mentioned in a question are not needs. A "don't mind" frees just the property that the assistant's message asked about; it cannot remove or alter an earlier stated need, and if the message posed no question (it merely informed), that answer changes nothing.
- When a need is given by pointing to another listing, its value is the line's pointing phrase itself ("finish: unlike the one shown on Y's kettle page"); values the line mentions for that other listing describe it, not the product, so they are neither needs nor drops nor freed properties.
Answer with: one JSON object with all of these keys."""

READ_RULES = """- Settle every requirement from the listing field for that property, matching the value by the words this store writes there. Spelling of the field name does not matter: other separators, an added prefix, a name cut short or made longer, or a usual alternative name (a pants fit type may sit under "fit", a collar type under "neckline", a colour under "color family") all count. Where a field bearing the property's own name exists, it decides, together with any field of that name carrying one extra leading word (length may sit under pants length, material under clothing material): should either one write the value, the requirement holds. Two names apart only by separators (size_class versus size class) form a single field, and either writing the value is enough. Lacking those, an alternative field decides; the options list or the title may supply the value solely when not one field names the property. Descriptions never serve as evidence.
- Match values word by word as this store spells them: ignore case, separators, singular versus plural and spelling variants; a narrower value holding the required one is fine (cotton is met by "80% cotton", green by "forest green", rtx 3060 by "rtx 3060 12gb"); one matching member of a list suffices. Synonyms, abbreviations, related terms and shades (technical wood written "tech wood", windows 11 written "win11", embroidery written "embroidered", plain written "solid", off-white written ivory or cream) are accepted only while no listing of the category writes the value itself in that field: with "matte" written as the finish elsewhere, a finish given as "satin" fails matte. When a plus joins two values ("wool + nylon"), that pair fails either part if the same field also appears holding the other part alone ("nylon"). FIELD EVIDENCE lines, when supplied, are worked out from this store's records and state which case is true. Any other value of the property fails (mechanical is not quartz, maple is not rosewood), as does a word that just resembles it. Claims to cover the whole property ("suitable for all occasions", "all seasons", "universal") name no value and fail a required one; yet a list naming the required value besides still passes. A field holding another value fails even when that value commonly accompanies or hints at the required one (zip-up is not a "pullover" style, lace-up is not "slip-on"); only a shade commonly sold as the required colour, or a narrower value holding it, passes.
- Fields of other properties never count (the material is not a lining material, a style is not a length, a strap type is not a strap length): none may substitute for a property whose own field appears across the category's listings.
- Numbers compare as amounts after unit conversion (45 mm equals 4.5 cm); edges belong to the limit; a range passes solely if every part of it satisfies the limit. With no field naming the property and the title as evidence, an exact value required passes when the title gives its words and figures in that same order, even with more detail following (for 9' x 12', the title 9' x 12' 4" passes while 9' x 13' fails).
- Prices share the unit of the request's budget unless a record names another currency or unit (with a budget of 1000 US cents, a price of 45 means 45 cents, comfortably within); rescale neither the price nor the budget.
- Requirements outside the list are irrelevant: never reject a listing for a season, an occasion, a gender, an age group, a size, a condition, a quality or a use the list never mentions.
- Kind is settled by the category label alone: whatever object the title mentions (an accessory, a set, a refill, a case, a container), a listing in the requested category is that kind; judge it solely by requirements, price and stock."""

CHECK_ITEM = """Context: a shopping assistant is on the point of committing to an item, and you are its final gate; the errand earns 1 solely when that item satisfies each requirement below.
Your job: rule on the item using ONLY the records shown and the rules that follow.
Constraints:
""" + READ_RULES + """
- Words forming part of the category's own name are satisfied by the category.
- Exclusions and closed lists must hold.
- The item must belong to the requested kind; where listings bear a category label, the label has to match the request's wording taken whole, not a narrower label or a neighbour.
- Quantity times price must stay inside the budget (reaching it exactly is fine). Records showing it out of stock, or holding fewer units than the quantity ordered, mean a fail.
- A wish only ("ideally", "would be nice") that is marked not required is no requirement.
- Where records conflict (over stock or price, say), the latest one wins.
Answer with: a single JSON object and no other text: {"item": the id of the item, "fits": true or false, "problems": [brief strings, one for each requirement unmet or unshown]}. Use fits=false only for a listed requirement that the item's fields clearly fail by these rules (a FIELD EVIDENCE line saying the item does NOT meet a requirement counts as that), or one stated by no field and, if no field for it exists, by neither the title nor the options list; missing stock or price records are never a reason to fail it."""

RANK_CHECK = """Context: an order is imminent for a request whose aim is the PRIORITY. You receive the requirements, the item a shopping assistant chose and the listing records it saw.
Your job: name every listing that could satisfy all requirements AND outdo the chosen item on that aim (cheaper when the cheapest is wanted; a stronger value of the measure when the best is wanted).
Constraints:
- Judge by the reading rules below (quantity times price inside the budget; no out-of-stock showing).
- Omit listings whose records reveal a failed requirement, a missing stock or a price above budget.
- Numbers in a ROW's title (55 x 94, for instance) leave each dimension unnamed, so they can never disqualify a ROW.
READING RULES:
""" + READ_RULES + """
Answer with: a single JSON object and no other text: {"better": [ids, strongest first, 4 at most], "reason": "one sentence"}."""

ASK_PLAN = """Context: you choose what a shopping assistant will ask a shopper. The shopper keeps hidden needs, each one a value of a single product property that surfaces only if a message names exactly that property; whatever a message asks first gets the surest answer, and anything else just draws a polite "don't mind".
Your job: choose at most QUESTION_COUNT separate properties, best bet first.
Constraints:
- Step 0 comes before everything: any item below marked MUST BE LEARNED FROM THE SHOPPER that is still unasked; put it plainly. On the budget, ask for the top amount they are willing to pay. On a trade-off between two wishes, give both values together with their properties, as in "If one listing can't have both, would you rather have the <value> <property> or the <value> <property>?".
- Step 1 comes next: properties really present in the details of the listings below. Begin with the story (when STORY HINTS appear below, they list its details and which property each detail is about, and they lead): the concealed need tends to be the property its concrete details circle around while never stating the value. Things the user presses or handles, inspects up close or on photos, listens to, measures or tries for fit, reads off a label, or the way it was produced all point at that part or aspect of the item and at its property: fine print, tags or labels hint at ingredients, fabric or material; looks and photos at the visible part, colour or pattern; a space or body part it must match at the size or shape of that part; a gift for another person, maybe a near stranger, at who it is made for (age group, gender); the feel against foot, skin or hand first at fabric or material; the activity, time or place of use or wear (outdoors, the office, a gym class) at its type or style, i.e. the property whose values name such uses; anything opened, closed or adjusted many times at its mechanism or closure; the part doing the work inside at its core component. A place, a schedule, a job or weather unrelated to where it is used or worn is mere background. Open with the property those details aim at, called by the listings' term. Who the item is made for competes like any other property: ask about it if the story aims at another person, or if no other hint exists. Next, rank properties of the product itself (function, length, closure, type or style variant, shape, material, pattern) above logistics (years, size or variant lists, pack size, condition, warranty). The keys arrive ranked best first (widest coverage and most varied values); among those, lean toward a key flagged as differing across the listings read (a hidden need splits listings, which a key written identically everywhere never does), reserve a slot of its own for any key flagged as figures with a unit (weight, length, height, width, a size) when such a key is present, lean toward keys carried by most of that product's opened listings, and, with several products requested, toward the product that still has many fitting listings (the hidden need must narrow them); never waste a slot on a variant or option picker or on a key that merely duplicates another with a number.
- Skip any property the request has settled already (quantity, stock, price, its requirements), anything asked before, and near-duplicate pairs (two spellings of a property, two style kinds); brand, maker, product or model names, price, stock and delivery are never asked.
- Only properties appearing as keys in the listings' details below may be asked, under the listings' own name; a STORY HINTS property without a matching key below may be asked by its plain name, except when the hint says [no key] (enough listings of its product were read lacking that field): in that case ask the key below whose values vary as its detail describes, or drop it.
- A step-1 question is a single short, friendly, plain sentence of 15 words or fewer asking which <property> they would like for the <item kind>, with the property called by the listings' own term (example values on their own never reveal a need), and at most one or two example values after it. No intro, no progress report, no lists or instructions, and never a catch-all key (a name, a title, a description, variants, options). Use the key exactly as one of that product's listings below writes it, and keep "second" empty.
- One question, ONE property of ONE product (except the step-0 trade-off between two wishes): never pair two properties or two products in a question, because a reply discloses one need at most and a two-subject question can go unanswered. Fill "second" with the next best property of the SAME product (not a second name for the first), or keep it empty: the assistant may tack it onto its final message.
- With SEVERAL products requested, hidden needs can sit on any one: share out the questions so each product has one before another gets two (products with no property left are skipped), draw each product's properties from the list of its own category below, write the property as "property (product)", and mention the product in the question ("Which toe style would you like for the boots?").
Answer with: a single JSON object and no other text: {"questions": [{"property": "...", "second": "... or empty", "question": "..."}, ...]}."""

NEXT_ONE = """
Only ONE question goes out this round; the shopper's reply is read before picking the next. REMAINING question(s) stay available overall and each should get used: place the one strongest property at the top of the list (one or two spares may come after it); once a need is revealed, the next question turns to another property and another story detail."""

BASKET_RANK = """# Shopping assistant - this step: choose which properties to ask the customer about
- The customer keeps back one or more wishes, on any of the products (often all on one), and states each only when a question names that exact property.
- For each product (P1, P2 ...) you get the properties seen on the shop's listings for it, with how many listings carry each and some sample values.
# Rules
- Rank up to eight properties per product, the one most likely to hold an unstated wish first.
- A property the request already settles holds no unstated wish: leave it out. Leave out identifiers, titles and prices too.
- Lean on what the background implies without saying it: who uses the item, where or how it is used, how it should look or feel, what went wrong with the old one.
- A property such a detail points at goes in "story" as well as in "ranked". With no such detail, "story" stays empty.
- Use property names exactly as listed; two spellings of one property count once.
# Answer
- ONE JSON object and nothing else: {"P1": {"ranked": ["property", ...], "story": ["property", ...]}, "P2": {...}}"""
BASKET_FIRST = 2

BEFORE_DECLINE = """Context: a shopping assistant intends to tell the shopper the store has nothing suitable, an ending that is seldom correct.
Your job: test the listing records below against the requirements with these reading rules (its cost stays within the budget and it must not appear out of stock):
""" + READ_RULES + """
Constraints:
- A few listings below come from search rows alone (details never read, so their property fields are unknown). A row of that sort in the requested category (no matter which object its title names), whose title, stock and price reveal no failed requirement, remains unverified (title figures such as 55 x 94 leave each dimension unnamed, so they cannot prove a row fails): place it in "unopened", most promising first (a title naming a value that is required, or the requested kind with nothing speaking against it).
Answer with: a single JSON object and no other text: {"possible_fits": [up to 5 ids of listings that look like they satisfy all requirements, strongest first], "unopened": [up to 6 ids of listings known only from rows, to open ahead of any decline], "reason": "one sentence"}."""

REF_HOP = """Context: the shopper wants the requested product to copy the PROPERTY of a different store listing, which they describe as: DESCRIPTION. The records below come from candidates for that listing.
Your job: find that listing and report what it says.
Constraints:
- Text naming a listing only as what the target isn't ("not the green one from X"), or as a size, colour or version on sale elsewhere, gives no go_to; nor does a listing it names about some other property.
- A record missing the described details is never the target, whatever its text claims.
Answer with: a single JSON object and no other text, with:
"target": id of the record that IS the described listing (that seller, that kind, every named detail shown, read by meaning; a figure stated in another unit counts if the amount is equal); null if no record is.
"missing": described details absent from the target ([] if none are).
"go_to": if the target's own text directs the reader to some DIFFERENT listing for the PROPERTY (use it, take it from it, refer to it, look at it or go by it), that listing as {"seller": shop or brand name as written, "kind": what kind of product it is, "details": [every value the text gives it, brief and verbatim: a material, a size plus unit, a colour]}; null otherwise.
"value": with no go_to, the value of the PROPERTY written in the target's own record.
"goes_with": values that the target's text pairs, groups or equates with that value ([] if there are none)."""

CHANGE_REVIEW = """Context: conditions of a shopping errand shifted a moment ago: a refused finish, a shopper line or a store notice brought in a new budget, a requirement added or removed, or an item no longer there. You get the request, the change, the requirements now applying, the priority, the cart, and a card per listing met so far (DETAILS cards hold the full record).
Your job: rule again on the OPENED listings under the conditions that hold NOW.
Constraints:
""" + READ_RULES + """
Answer with: a single JSON object and no other text: {"fits": [ids of opened listings satisfying all requirements in force today, inside the budget and not out of stock, strongest first (cheapest first if the cheapest is requested)], "pick": id to order at once, or null if nothing qualifies, "cart_still_right": true if the line in the cart is one of the fits and is the correct pick (a still-fitting basket stays), false if the cart line stopped fitting or a cheapest or best rule favours another pick, null if the cart holds nothing, "reason": "one sentence"}."""

FINISH_CHECK = """Context: a shopping assistant is closing an errand with either a decline or an order, and you check it one final time. (Ahead of an order, when the shopper got no message after the latest cart change, the code itself reports the basket and its total to them; list that only if a stated step demands something extra or different, say a question to answer or waiting for a yes.) Steps the assistant must perform, some before or at the end, may be set by the request, later shopper lines or store notices: confirming with the shopper beforehand, reporting the basket contents or the total, a closing remark, one more look at something.
Your job: go through the instructions in force, the latest notices and shopper lines, and the actions since the latest cart change, and judge whether a stated step remains undone ahead of THIS finish.
Constraints:
- Basket steps carried out before the latest cart change are not done.
- Make nothing up: only steps really stated by the request, the shopper or the store, and only steps a listed tool can perform.
- Two entries at most; drop a step that none of the listed tools can perform.
Answer with: a single JSON object and no other text: {"pending": [an entry for each undone step, in running order: {"why": that stated step in a few words, "tool": exact name of the tool performing it, "args": its arguments just as that tool's schema requires; for a message to the shopper, spell it out as one or two brief plain sentences holding the facts the step wants (for the basket: every cart line with id, count and price, plus the total)}], "ready": true if nothing is pending, false otherwise}."""
DECLINE_WORDS = r"refus|declin|cancel|abandon|reject|give.?up|abstain|withdraw|no.?(?:purchase|order|buy)|nothing|without"
ORDER_WORDS = r"commit|buy|order|purchas|check.?out|pay|confirm|complete|place|accept|proceed"
SORRY = "Sorry, nothing in the store meets all of your requirements, so I am not ordering anything."
ID_LIKE = r"[A-Za-z0-9][A-Za-z0-9_\-:.]{3,63}"
CODE_LIKE = r"[A-Za-z0-9_\-]+"


BOUND_WORDS = r"\b(?:no|not)\s+(?:more|less|fewer|over|under|above|below|beyond|exceeding)\b"
CH_BOUND = r"\b(?:at least|no more than|at most|no less than|more than|less than|up to|or more|or less|or larger|or smaller|or higher|" \
           r"or lower|minimum|maximum|between|within|under|over|above|below|than)\b"
CJ_COMPARE = (r"\b(?:falls?\s+short\s+of|short\s+of|nothing\s+(?:less|other)\s+than|nothing\s+short\s+of|nothing\s+but|"
              r"no\s+(?:less|worse)\s+than|at\s+least|(?:less|more|better|worse)\s+than|exactly|only|just)\b")
CH_SKIP = {"and", "the", "a", "an", "of"}
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
# settings by order type, one product or several (counted after the category merge), read wherever the two kinds part ways
ROUTES = {
    "single": {"asker": "planner", "checker": CHECKER, "sweep_queries": 2, "wording_holds": 3, "ref_share": REF_STEPS,
               "menu_merge": True},
    "multi": {"asker": "ranker", "checker": CHECKER, "sweep_queries": 3, "wording_holds": 3, "ref_share": REF_STEPS,
              "menu_merge": False},
}
REF_DIFFER_BLIND = 5
SELLER_KEYS = r"shop|seller|store|vendor|brand|make|manufactur|merchant|retailer|supplier|sold.?by"


class TextKit:
    @staticmethod
    def as_text(piece: Any, ceiling: int = VIEW_CHARS) -> str:
        try:
            shown = json.dumps(piece, ensure_ascii=False, separators=(",", ":"))
        except (TypeError, ValueError):
            shown = str(piece)
        if len(shown) > ceiling:
            return shown[:ceiling] + "...(cut)"
        return shown

    @staticmethod
    def clipped(piece: Any, chars: int, lvl: int = 0) -> Any:
        if lvl > 8:
            return None
        deeper = lvl + 1
        if isinstance(piece, dict):
            return {fkey: TextKit.clipped(val, chars, deeper) for fkey, val in piece.items()}
        if isinstance(piece, list):
            return [TextKit.clipped(val, chars, deeper) for val in piece]
        if isinstance(piece, str) and len(piece) > chars:
            return piece[:chars] + "..."
        return piece

    @staticmethod
    def _unfenced(raw: str) -> str:
        inner = raw.strip()
        if not inner.startswith("```"):
            return inner
        inner = re.sub(r"^```[A-Za-z]*", "", inner).strip()
        return inner[:-3].strip() if inner.endswith("```") else inner

    @staticmethod
    def json_object(text_body: Any) -> dict[str, Any] | None:
        if not isinstance(text_body, str):
            return None
        inner = TextKit._unfenced(text_body)
        braced = inner[inner.find("{"): inner.rfind("}") + 1] if "{" in inner else ""
        for attempt in (inner, braced):
            if attempt:
                try:
                    parsed = json.loads(attempt)
                except ValueError:
                    parsed = None
                if isinstance(parsed, dict):
                    return parsed
        return None

    @staticmethod
    def is_number(piece: Any) -> bool:
        if isinstance(piece, bool):
            return False
        return isinstance(piece, (int, float))

    @staticmethod
    def long_strings(piece: Any, collected: list[str], lvl: int = 0) -> None:
        if lvl > 4:
            return
        if isinstance(piece, list):
            for entry in piece[:10]:
                TextKit.long_strings(entry, collected, lvl + 1)
        elif isinstance(piece, str) and len(piece.split()) >= 4:
            collected.append(piece)

    @staticmethod
    def long_strings_all(d: dict[str, Any], out: list[str]) -> None:
        for val in d.values():
            if isinstance(val, dict):
                for inner in val.values():
                    if isinstance(inner, str) and inner.strip():
                        out.append(str(inner).strip())
            elif isinstance(val, str) and val.strip():
                out.append(val.strip())

    @staticmethod
    def walk_dicts(piece: Any, visit, lvl: int = 0, ceiling: int = 6) -> None:
        if lvl > ceiling:
            return
        if isinstance(piece, list):
            for entry in piece:
                TextKit.walk_dicts(entry, visit, lvl + 1, ceiling)
            return
        if not isinstance(piece, dict):
            return
        visit(piece)
        for entry in piece.values():
            if isinstance(entry, (dict, list)):
                TextKit.walk_dicts(entry, visit, lvl + 1, ceiling)

    @staticmethod
    def junk_key(key: Any) -> bool:
        head = re.split(r"\s*\(", str(key))[0]
        spaced = re.sub(r"([a-z])([A-Z])", r"\1 \2", head).lower()
        words = " ".join(re.findall(r"[a-z0-9]+", spaced))
        if re.search(r"\b(?:brand|shop|seller|store|vendor|manufacturer|category|categories|department|title|headline|"
                     r"heading|listing|description|overview|price|cost|amount|availability|sku|url)\b", words):
            return True
        return bool(re.fullmatch(r"(?:name|notes?|stock|in stock|on hand|images?|links?|make|maker|shelf|ids?|id)", words))

    @staticmethod
    def _id_bearing(record: dict[str, Any]) -> bool:
        for key, val in record.items():
            if isinstance(val, str) and "id" in str(key).lower() and re.fullmatch(ID_LIKE, val):
                return True
        return False

    @staticmethod
    def _sole(names: dict[str, Any], test) -> Any:
        picks = [key for low, key in names.items() if test(low, key)]
        return picks[0] if len(picks) == 1 else None

    @staticmethod
    def _plain_names(record: dict[str, Any]) -> None:
        if not TextKit._id_bearing(record):
            return
        names = {str(key).lower(): key for key in record}
        priced = any(re.search(r"price|cost", low) and TextKit.is_number(record[key]) for low, key in names.items())
        if not priced:
            key = TextKit._sole(names, lambda low, key: TextKit.is_number(record[key])
                                and re.search(r"amount|fee|charge|pay|rate|value", low)
                                and not re.search(r"total|count|qty|quant|unit|due|sum|rating|star|score|page|number|num|size|weight|width|height|length|found|match", low))
            if key is not None:
                record["price"] = record[key]
        if not any(re.search(r"stock|avail", low) for low in names):
            key = TextKit._sole(names, lambda low, key: isinstance(record[key], bool)
                                and re.search(r"hand|orderable|buyable|sellable|ready|ship|instock|purchas|in_store", low))
            if key is not None:
                record["in_stock"] = record[key]
        if not any(re.search(r"title|heading", low) for low in names):
            key = TextKit._sole(names, lambda low, key: isinstance(record[key], str)
                                and re.search(r"headline|name|label|caption", low)
                                and not re.search(r"shop|seller|store|vendor|brand|make|categor|shelf|department|id", low))
            if key is not None:
                record["title"] = record[key]

    @staticmethod
    def lensed(piece: Any) -> Any:
        try:
            copy = json.loads(json.dumps(piece))
        except (TypeError, ValueError):
            return piece
        TextKit.walk_dicts(copy, TextKit._plain_names)
        return copy


class PropKit:
    @staticmethod
    def _split_words(name: Any) -> list[str]:
        head = re.split(r"\s*\(", str(name))[0]
        return re.findall(r"[a-z]+", re.sub(r"([a-z])([A-Z])", r"\1 \2", head).lower())

    @staticmethod
    def _singular(word: str) -> str:
        if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
            return word[:-1]
        return word

    @staticmethod
    def prop_key(name: Any) -> str:
        return "".join(PropKit._singular(w) for w in PropKit._split_words(name))

    @staticmethod
    def prop_words(name: Any) -> list[str]:
        return [PropKit._singular(w) for w in PropKit._split_words(name)]

    @staticmethod
    def edits_within(a: str, b: str, most: int) -> bool:
        if abs(len(a) - len(b)) > most:
            return False
        row = list(range(len(b) + 1))
        for i, left in enumerate(a, 1):
            nxt = [i]
            for j, right in enumerate(b, 1):
                nxt.append(min(row[j] + 1, nxt[j - 1] + 1, row[j - 1] + (left != right)))
            if min(nxt) > most:
                return False
            row = nxt
        return row[-1] <= most

    @staticmethod
    def same_prop(a: Any, b: Any) -> bool:
        first, second = PropKit.prop_key(a), PropKit.prop_key(b)
        if not (first and second):
            return False
        if first == second:
            return True
        shorter = min(len(first), len(second))
        allowed = 1 if shorter < 6 else 2
        if shorter >= 4 and first[:3] == second[:3] and PropKit.edits_within(first, second, allowed):
            return True
        if shorter < 5:
            return False
        return first.startswith(second) or second.startswith(first)

    @staticmethod
    def _soft_letter(word: str, i: int) -> bool:
        if word[i] in "aeiouy":
            return True
        if i > 0 and word[i - 1] == word[i]:
            return True
        return i + 1 < len(word) and word[i + 1] == word[i]

    @staticmethod
    def _soft_drops(word: str):
        for i in range(len(word)):
            if PropKit._soft_letter(word, i):
                yield word[:i] + word[i + 1:]

    @staticmethod
    def spelled_alike(lo: str, hi: str) -> bool:
        gap = len(hi) - len(lo)
        if gap == 0:
            spots = [i for i, (x, y) in enumerate(zip(lo, hi)) if x != y]
            if len(spots) != 2 or spots[1] != spots[0] + 1:
                return False
            return lo[spots[0]] == hi[spots[1]] and lo[spots[1]] == hi[spots[0]]
        if gap == 1:
            return any(shorter == lo for shorter in PropKit._soft_drops(hi))
        if gap == 2:
            for shorter in PropKit._soft_drops(hi):
                if any(least == lo for least in PropKit._soft_drops(shorter)):
                    return True
        return False

    @staticmethod
    def strict_prop(a, b):
        first, second = PropKit.prop_key(a), PropKit.prop_key(b)
        if not first or not second:
            return False
        if first == second:
            return True
        lo, hi = sorted((first, second), key=len)
        if len(lo) >= 4 and lo[:3] == hi[:3] and PropKit.spelled_alike(lo, hi):
            return True
        short, longer = sorted((PropKit.prop_words(a), PropKit.prop_words(b)), key=len)
        extends = len(longer) == len(short) + 1 and longer[:len(short)] == short
        return extends and len(longer[-1]) <= 3 and len("".join(short)) >= 5

    @staticmethod
    def kind_words(text: Any) -> set[str]:
        found = re.findall(r"[a-z]{3,}", str(text).lower())
        return {w.rstrip("s") for w in found if w not in LINK_WORDS}

    @staticmethod
    def kind_tokens(text: Any) -> list[str]:
        plain = re.sub(r"'s\b", "", str(text).lower().replace("’", "'"))
        return [w for w in re.findall(r"[a-z]{3,}", plain) if w not in LINK_WORDS]

    @staticmethod
    def plural_forms(word: Any) -> set[str]:
        w = str(word)
        forms = {w, w + "s", w + "es"}
        if w.endswith("y"):
            forms.add(w[:-1] + "ies")
        if w.endswith("fe"):
            forms.add(w[:-2] + "ves")
        elif w.endswith("f"):
            forms.add(w[:-1] + "ves")
        if w.endswith("ies"):
            forms.add(w[:-3] + "y")
        if w.endswith("ves"):
            forms |= {w[:-3] + "f", w[:-3] + "fe"}
        if w.endswith("es"):
            forms.add(w[:-2])
        if w.endswith("s"):
            forms.add(w[:-1])
        return {form for form in forms if len(form) >= 3}

    @staticmethod
    def same_kind(a: Any, b: Any) -> bool:
        if a == b:
            return True
        return str(b) in PropKit.plural_forms(a) or str(a) in PropKit.plural_forms(b)

    @staticmethod
    def kind_in(word: Any, pool: Any) -> bool:
        for other in pool:
            if PropKit.same_kind(word, other):
                return True
        return False

    @staticmethod
    def kinds_within(words: Any, pool: Any) -> bool:
        for w in words:
            if not PropKit.kind_in(w, pool):
                return False
        return True

    @staticmethod
    def kind_score(own: Any, words: Any, shared: Any) -> float:
        total = 0
        for w in own:
            if PropKit.kind_in(w, words):
                total += 0.5 if PropKit.kind_in(w, shared) else 1.0
        return total

    @staticmethod
    def name_pattern(name: Any) -> str:
        pieces = []
        for w in PropKit.kind_tokens(name):
            spellings = sorted(PropKit.plural_forms(w), key=lambda form: (-len(form), form))
            pieces.append(r"(?:" + "|".join(map(re.escape, spellings)) + r")(?:'s\b|'|\b)")
        return r"[\s'-]+".join(pieces)

    @staticmethod
    def _word_match(a: str, b: str, exact: bool) -> bool:
        if a == b:
            return True
        if exact:
            return False
        if len(b) >= 3 and a in (b + "s", b + "es"):
            return True
        return len(a) >= 3 and b in (a + "s", a + "es")

    @staticmethod
    def prop_named(prop, text, exact=False):
        head = re.sub(r"([a-z])([A-Z])", r"\1 \2", re.split(r"\s*\(", str(prop or ""))[0])
        words = re.findall(r"[a-z0-9]+", head.lower())
        toks = re.findall(r"[a-z0-9]+", str(text or "").lower())
        if not words or len(" ".join(words)) < 3:
            return False
        width = len(words)
        return any(all(PropKit._word_match(t, w, exact) for t, w in zip(toks[i: i + width], words))
                   for i in range(len(toks) - width + 1))

class ValueKit:
    @staticmethod
    def ch_words(text: Any) -> list[str]:
        found = re.findall(r"[a-z0-9]+(?:\.[0-9]+)?", str(text or "").lower())
        return [w for w in found if w not in CH_SKIP]

    @staticmethod
    def ch_word_in(word: str, pool: list[str]) -> bool:
        lettered = word.isalpha()
        for other in pool:
            if other == word:
                return True
            if not (lettered and other.isalpha()):
                continue
            if PropKit.same_kind(word, other):
                return True
            if min(len(word), len(other)) >= 6 and word[:2] == other[:2] and PropKit.edits_within(word, other, 1):
                return True
        return False

    @staticmethod
    def ch_alts(value: Any) -> list[list[str]]:
        plain = re.sub(r"\((?:stated|revealed[^)]*)\)", " ", str(value or "").lower()).replace('"', " ")
        plain = re.sub(r"\b(?:only|just|exactly|strictly)\b", " ", plain)
        alternatives = []
        for part in re.split(r"\bor\b", plain):
            words = ValueKit.ch_words(part)
            if words:
                alternatives.append(words)
        return alternatives

    @staticmethod
    def ch_writes(value: Any, text: Any) -> bool:
        return ValueKit.ch_match(ValueKit.ch_alts(value), ValueKit.ch_words(text))

    @staticmethod
    def _run_equals(pool: list[str], target: str) -> bool:
        for start in range(len(pool)):
            joined = ""
            for piece in pool[start:]:
                joined += piece
                if joined == target:
                    return True
                if len(joined) >= len(target):
                    break
        return False

    @staticmethod
    def ch_match(alts: list[list[str]], pool: list[str]) -> bool:
        if not pool:
            return False
        for words in alts:
            if all(ValueKit.ch_word_in(w, pool) for w in words):
                return True
            target = "".join(words)
            spaced = len(words) >= 2 or re.search(r"\d", target) or any(re.search(r"\d", p) for p in pool)
            if spaced and ValueKit._run_equals(pool, target):
                return True
        return False

    @staticmethod
    def ch_touches(value: Any, text: Any) -> bool:
        pool = ValueKit.ch_words(text)
        low = str(text or "").lower()
        for alternative in ValueKit.ch_alts(value):
            for w in alternative:
                if ValueKit.ch_word_in(w, pool):
                    return True
                if re.fullmatch(r"\d+(?:\.\d+)?", w) and re.search(r"(?<![\d.])" + re.escape(w) + r"(?!\d|\.\d)", low):
                    return True
                if len(w) < 3:
                    continue
                for p in pool:
                    if len(p) >= 3 and (p.startswith(w[:3]) or w.startswith(p[:3])):
                        return True
        return False

    @staticmethod
    def ch_parts(text: Any) -> list[str]:
        low = str(text or "").strip().lower()
        parts = [p.strip() for p in re.split(r"[,;|]", low) if p.strip()]
        if parts:
            return parts
        return [low] if low else []

    @staticmethod
    def ch_sides(part: Any) -> list[str]:
        sides = [s.strip() for s in re.split(r"\s*\+\s*", str(part)) if s.strip()]
        return [] if len(sides) < 2 else sides

    @staticmethod
    def figure_written(fig: str, said: str) -> bool:
        decimal = "." in fig
        for hit in re.finditer(r"(?<![\d.])" + re.escape(fig) + r"(?!\d|\.\d)", said):
            if decimal:
                return True
            unit = re.match(r"\s*([a-z]+)", said[hit.end():])
            if unit is None or len(unit.group(1)) <= 3:
                return True
        return False

    @staticmethod
    def _content(text: str) -> list[str]:
        return [w for w in re.findall(r"[a-z]{3,}", text) if w not in LINK_WORDS]

    @staticmethod
    def positive_value(value, name=""):
        low = str(value).lower().strip()
        if re.fullmatch(r"\W*(?:yes|no|true|false|none|required|included|needed)\W*", low) or ValueKit._literal_hit(low):
            return True
        low = ValueKit._drop_comparisons(low)
        body = re.sub(r"\b(?:" + VALUE_LINKS + r")\b", " ", low)
        if any(len(ValueKit._content(clause)) >= 4 for clause in re.split(r",|;|/|&|\(|\)|\bor\b|\band\b", body)):
            return False
        negation = re.search(EXCLUDE_WORDS, low)
        if not negation or ValueKit._content(low[:negation.start()]):
            return True
        after = ValueKit._content(low[negation.end():])
        own = set(PropKit.prop_words(name)) | set(re.findall(r"[a-z]+", str(name).lower()))
        return bool(after and own) and all(w in own or w.rstrip("s") in own for w in after)

    @staticmethod
    def _literal_hit(low):
        if not LITERAL_VALUES:
            return False
        unmarked = re.sub(r"\((?:stated|revealed[^)]*)\)", " ", low)
        return " ".join(re.findall(r"[a-z0-9]+", unmarked)) in LITERAL_VALUES

    @staticmethod
    def _drop_comparisons(low):
        low = re.sub(BOUND_WORDS, " ", low)
        low = re.sub(r"\b(?:no|not)\s+[a-z]+er\b(?=\s+than\b)", " ", low)
        if not re.search(r"\d", low):
            return low
        return re.sub(r"\b(?:no|not)\s+[a-z]+er\b", " ", low)

    @staticmethod
    def _figure_run_pattern(run: str) -> str:
        figs = re.findall(r"\d+(?:\.\d+)?", run)
        return r"\D{0,4}".join(re.escape(f) for f in figs)

    @staticmethod
    def value_seen(value: Any, samples: Any) -> bool:
        pool = str(samples or "").lower()
        if not pool.strip():
            return False
        body = re.sub(r"\b(?:" + VALUE_LINKS + r")\b", " ", str(value).lower())
        for w in re.findall(r"[a-z]{3,}", body):
            if w in LINK_WORDS:
                continue
            if re.search(r"\b" + re.escape(w[:-1] if len(w) > 4 else w), pool):
                return True
        for run in re.findall(r"\d+(?:\.\d+)?(?:\s*[:x/]\s*\d+(?:\.\d+)?)*", body):
            if re.search(r"(?<![\d.])" + ValueKit._figure_run_pattern(run) + r"(?![\d.])", pool):
                return True
        return False

    @staticmethod
    def one_prop_hint(hint: str) -> str:
        try:
            head, colon, detail = hint.partition(":")
            if not colon or "/" not in head:
                return hint
            tag = re.search(r"\(([^()]*)\)\s*$", head)
            names_part = head[: tag.start()] if tag else head
            names = [n.strip() for n in names_part.split("/") if n.strip()]
            if len(names) < 2:
                return hint
            tag_text = " (" + tag.group(1).strip() + ")" if tag else ""
            return f"{names[0]}{tag_text}:{detail} (or {', '.join(names[1:])})"
        except Exception:
            return hint

    @staticmethod
    def lead_value(detail: Any, request: str, product: str) -> str:
        try:
            linkers = "|".join(sorted(LINK_WORDS))
            opening = re.match(r"\s*([a-z][a-z-]*[a-z])(?:\s*[,;.()]|\s*$|\s+(?:" + linkers + r")\b)", str(detail).lower())
            if opening is None:
                return ""
            word = opening.group(1)
            kinds = [w for w in re.findall(r"[a-z]{3,}", str(product).lower()) if w not in LINK_WORDS]
            if len(word) < 3 or word in LINK_WORDS or not kinds:
                return ""
            if word.rstrip("s") in {k.rstrip("s") for k in kinds}:
                return ""
            head_kind = kinds[0].rstrip("s")
            toks = re.findall(r"[a-z][a-z'-]*", str(request).lower())
            for here, after in zip(toks, toks[1:]):
                if here == word and after.split("'")[0].rstrip("s") == head_kind:
                    return word
        except Exception:
            return ""
        return ""

    @staticmethod
    def _stem_of(w: str) -> str:
        if w.endswith("es") and len(w) > 4:
            return w[:-2]
        if w.endswith("s") and len(w) > 3:
            return w[:-1]
        return w

    @staticmethod
    def _said_like(w: str, stem: str, s: str) -> bool:
        if s == w or (s.startswith(stem) and len(s) <= len(stem) + 3):
            return True
        if stem.startswith(s) and len(s) >= max(4, len(stem) - 2):
            return True
        if len(w) >= 4 and s.startswith(w):
            return True
        return len(w) >= 4 and len(s) >= 4 and s[0] == w[0] and PropKit.edits_within(s, w, 1)

    @staticmethod
    def story_guess(req, request, refs):
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
        if figs and all(ValueKit.figure_written(f, said) for f in figs):
            return []
        content = ValueKit._content(body)
        missing = ValueKit._unsaid_words(content, said)
        if len(content) <= 2 and [w for w in content if len(w) >= 4 and w not in missing]:
            return []
        return missing + ValueKit._unsaid_runs(body, said)

    @staticmethod
    def _unsaid_words(content, said):
        said_words = set(re.findall(r"[a-z]+", said))
        said_stems = {s[:-1] if s.endswith("s") and len(s) > 3 else s for s in said_words}

        def heard(w):
            stem = ValueKit._stem_of(w)
            if any(ValueKit._said_like(w, stem, s) for s in said_words):
                return True
            return any(len(s) >= 3 and stem.startswith(s) and len(s) >= len(stem) - 3 for s in said_stems)
        return [w for w in content if not heard(w)]

    @staticmethod
    def _unsaid_runs(body, said):
        gaps = []
        for run in re.findall(r"\d+(?:\.\d+)?(?:\s*[:x/]\s*\d+(?:\.\d+)?)*", body):
            parts = re.findall(r"\d+(?:\.\d+)?", run)
            if len(parts) > 1 and not re.search(r"(?<![\d.])" + r"\D{0,4}".join(re.escape(f) for f in parts) + r"(?!\d)", said):
                gaps.append(run)
        return gaps


class RefKit:
    @staticmethod
    def ref_text(value: Any) -> bool:
        return re.search(REF_WORDS, str(value or "").lower()) is not None

    @staticmethod
    def _unit_joined(hit) -> str:
        if hit.group(2) in LINK_WORDS:
            return hit.group(0)
        return hit.group(1) + hit.group(2)

    @staticmethod
    def _plural_cut(tok: str) -> str:
        return tok[:-2] if re.search(r"(?:ch|sh|x|z)es$", tok) else tok[:-1]

    @staticmethod
    def ref_toks(text: Any) -> list[str]:
        low = re.sub(r"(\d)(?:\s+|\s*-\s*)([a-z]{1,8})\b", RefKit._unit_joined, str(text or "").lower())
        kept = []
        for tok in re.findall(r"\d+(?:[.,:/]\d+)*[a-z]*|[a-z][a-z0-9]*", low):
            if tok in LINK_WORDS:
                continue
            if len(tok) == 1 and not tok.isdigit():
                continue
            if len(tok) > 3 and tok.endswith("s") and not tok.endswith("ss") and not tok[-2].isdigit():
                tok = RefKit._plural_cut(tok)
            kept.append(tok)
        return kept

    @staticmethod
    def ref_figs(text: Any) -> set[tuple[float, str]]:
        pairs: set[tuple[float, str]] = set()
        try:
            low = str(text or "").lower().replace(",", "")
            for hit in re.finditer(r"(\d+(?:\.\d+)?(?:\s*/\s*\d+(?:\.\d+)?)*)\s*-?\s*([a-z]{1,8})\b", low):
                unit = hit.group(2)
                if len(unit) >= 3 and unit.endswith("s") and not unit.endswith("ss"):
                    unit = RefKit._plural_cut(unit)
                if unit in LINK_WORDS:
                    continue
                pairs.update((float(fig), unit) for fig in re.split(r"\s*/\s*", hit.group(1)))
        except Exception:
            return pairs
        return pairs

    @staticmethod
    def ref_name(text: Any) -> str:
        no_possessive = re.sub(r"'s\b", "", str(text or "").lower())
        return re.sub(r"[^a-z0-9]", "", no_possessive)

    @staticmethod
    def ref_relation(value: Any) -> tuple[str, float | None, str]:
        low = str(value or "").lower()
        if re.search(DIFFER_WORDS, low):
            return "differ", None, ""
        within = re.search(r"\bwithin\s+(\d+(?:\.\d+)?)\s*([a-z%]+)?", low)
        if within:
            return "within", float(within.group(1)), (within.group(2) or "").strip()
        if re.search(r"\b(?:no more than|at most|up to|or less|or lower|or smaller)\b", low):
            return "at most", None, ""
        if re.search(r"\b(?:at least|no less than|or more|or higher|or larger)\b", low):
            return "at least", None, ""
        return "same", None, ""

    @staticmethod
    def _seller_hit(body: str, ok: Any):
        for found in re.finditer(r"(?<![A-Za-z'])([A-Z][a-z]{3,})(?:'s)?(?![A-Za-z])", body):
            if found.start() > 0 and (ok is None or ok(found.group(1))):
                return found
        return None

    @staticmethod
    def ref_parse(text, ok=None):
        try:
            return RefKit._parsed_ref(str(text or ""), ok)
        except Exception:
            return None

    @staticmethod
    def _parsed_ref(text, ok):
        body = re.split(r"\(\s*or\b", text)[0]
        seller = RefKit._seller_hit(body, ok)
        if seller is None:
            return None
        kind, after = RefKit._ref_kind(body[:seller.start()], body[seller.end():])
        details = []
        for part in re.split(r",|;|\band\b|\bwith\b|\bthat\s+(?:is|are|has|have|comes?(?:\s+as)?)\b|\bwhich\b|\bin\b|\bat\b", after, flags=re.I):
            kept = [w for w in re.findall(r"[A-Za-z0-9][A-Za-z0-9.%:/&'-]*", part) if w.lower() not in LINK_WORDS]
            if kept:
                details.append(" ".join(kept))
        return {"seller": seller.group(1), "kind": kind.strip()[:80], "details": details[:6]}

    @staticmethod
    def _ref_kind(before, after):
        sold = re.search(r".*(?:\bthe|\bfor|\ba|\ban)\s+(.*?)\s+(?:sold|made|offered|listed)\s+by\s*$", before, re.I | re.S)
        if sold is not None:
            return sold.group(1), after
        lead = re.match(r"\s*([^,;(]*?)(?=\s+(?:with|that|which|in|at|having|from|of|for|sold)\b|[,;]|$)", after, re.I)
        if lead is None:
            return "", after
        return lead.group(1), after[lead.end():]

    @staticmethod
    def ref_span(prop: Any, source: Any) -> str:
        try:
            words = [w for w in re.findall(r"[a-z0-9]+", re.split(r"\s*\(", str(prop))[0].lower()) if w not in LINK_WORDS]
            if not words:
                return ""
            hit = re.search(r"\b" + r"\W+".join(map(re.escape, words)) + r"\b", str(source).lower())
            if hit is None:
                return ""
            window = str(source)[hit.end(): hit.end() + 500]
            return re.split(r"\(\s*or\b|(?<=[.!?])\s", window)[0]
        except Exception:
            return ""


class ArgCheck:
    TYPES = {
        "string": lambda val: isinstance(val, str),
        "integer": lambda val: isinstance(val, int) and not isinstance(val, bool),
        "number": TextKit.is_number,
        "boolean": lambda val: isinstance(val, bool),
        "object": lambda val: isinstance(val, dict),
        "array": lambda val: isinstance(val, list),
        "null": lambda val: val is None,
    }
    # keyword -> (comparison that breaks the limit, wording of the fault)
    LIMITS = {
        "minimum": (lambda val, lim: val < lim, "below minimum"),
        "maximum": (lambda val, lim: val > lim, "above maximum"),
        "exclusiveMinimum": (lambda val, lim: val <= lim, "must be above"),
        "exclusiveMaximum": (lambda val, lim: val >= lim, "must be below"),
    }
    ORDER = tuple("enum const minLength maxLength minimum maximum exclusiveMinimum exclusiveMaximum minItems maxItems items type".split())

    @classmethod
    def type_list(cls, jschema: dict[str, Any]) -> list[str]:
        declared = jschema.get("type")
        if isinstance(declared, str):
            return [declared]
        return list(declared or [])

    @classmethod
    def conforms(cls, piece: Any, ktype: str) -> bool:
        test = cls.TYPES.get(ktype)
        if test is None:
            return True
        return test(piece)

    @classmethod
    def _fits_any(cls, piece: Any, kinds: list[str]) -> bool:
        return any(cls.conforms(piece, kind) for kind in kinds)

    @staticmethod
    def _coerced(piece: Any, kind: str) -> tuple[bool, Any]:
        if kind == "integer":
            if isinstance(piece, float) and piece.is_integer():
                return True, int(piece)
            if isinstance(piece, str) and re.fullmatch(r"\s*-?\d+(\.0+)?\s*", piece):
                return True, int(float(piece))
        elif kind == "number":
            if isinstance(piece, str) and re.fullmatch(r"\s*-?\d+(\.\d+)?\s*", piece):
                figure = float(piece)
                return True, (int(figure) if figure.is_integer() else figure)
        elif kind == "string":
            if TextKit.is_number(piece):
                return True, str(piece)
        elif kind == "boolean":
            if isinstance(piece, str) and piece.lower() in ("true", "false"):
                return True, piece.lower() == "true"
        return False, piece

    @classmethod
    def convert(cls, piece: Any, type_list: list[str]) -> Any:
        if not type_list:
            return piece
        if not cls._fits_any(piece, type_list):
            for kind in type_list:
                changed, value = cls._coerced(piece, kind)
                if changed:
                    return value
            return piece
        whole_float = isinstance(piece, float) and piece.is_integer()
        return int(piece) if whole_float and "integer" in type_list else piece

    @classmethod
    def execute(cls, piece: Any, jschema: Any, where: str = "args") -> tuple[Any, list[str]]:
        if not isinstance(jschema, dict):
            return piece, []
        kinds = cls.type_list(jschema)
        piece = cls.convert(piece, kinds)
        if kinds and not cls._fits_any(piece, kinds):
            return piece, [f"{where}: expected {'/'.join(kinds)}"]
        faults: list[str] = []
        for keyword in cls.ORDER:
            if keyword not in jschema:
                continue
            piece, found = cls._apply(keyword, piece, jschema, where)
            faults += found
        return piece, faults

    @classmethod
    def _apply(cls, keyword: str, piece: Any, jschema: dict[str, Any], where: str) -> tuple[Any, list[str]]:
        rule = jschema[keyword]
        if keyword in cls.LIMITS:
            broken, words = cls.LIMITS[keyword]
            if TextKit.is_number(piece) and TextKit.is_number(rule) and broken(piece, rule):
                return piece, [f"{where}: {words} {rule}"]
            return piece, []
        if keyword == "enum":
            return cls._enum(piece, rule, where)
        if keyword == "const":
            return piece, ([] if piece == rule else [f"{where}: must equal {rule!r}"])
        if keyword == "maxLength":
            return (piece[:rule] if isinstance(piece, str) else piece), []
        if keyword == "minLength":
            short = isinstance(piece, str) and len(piece) < rule
            return piece, ([f"{where}: shorter than {rule} characters"] if short else [])
        if keyword == "minItems":
            few = isinstance(piece, list) and len(piece) < rule
            return piece, ([f"{where}: needs at least {rule} items"] if few else [])
        if keyword == "maxItems":
            many = isinstance(piece, list) and len(piece) > rule
            return piece, ([f"{where}: allows at most {rule} items"] if many else [])
        if keyword == "items":
            return cls._each_item(piece, rule, where)
        return cls._fields(piece, jschema, where)

    @staticmethod
    def _enum(piece: Any, options: Any, where: str) -> tuple[Any, list[str]]:
        if piece in options:
            return piece, []
        if isinstance(piece, str):
            wanted = piece.strip().lower()
            for option in options:
                if isinstance(option, str) and option.lower() == wanted:
                    return option, []
        return piece, [f"{where}: must be one of {options}"]

    @classmethod
    def _each_item(cls, piece: Any, item_schema: Any, where: str) -> tuple[Any, list[str]]:
        if not (isinstance(piece, list) and isinstance(item_schema, dict)):
            return piece, []
        faults: list[str] = []
        fixed = []
        for idx, entry in enumerate(piece):
            entry, found = cls.execute(entry, item_schema, f"{where}[{idx}]")
            fixed.append(entry)
            faults += found
        return fixed, faults

    @classmethod
    def _fields(cls, piece: Any, jschema: dict[str, Any], where: str) -> tuple[Any, list[str]]:
        if not isinstance(piece, dict):
            return piece, []
        known = jschema.get("properties") or {}
        faults = [f"{where}.{name}: required" for name in jschema.get("required") or [] if name not in piece]
        if jschema.get("additionalProperties") is False:
            faults.extend(f"{where}.{name}: not an allowed argument" for name in piece if name not in known)
        fixed = {}
        for name, entry in piece.items():
            if name in known:
                entry, found = cls.execute(entry, known[name], f"{where}.{name}")
                faults += found
            fixed[name] = entry
        return fixed, faults


class SendFault:
    def __init__(self, why: str):
        self.data = None
        self.error = {"status": None, "body": why[:300]}


class ModelCall:
    # bookkeeping of one query_model call across its models and attempts
    def __init__(self, wire: "Wire", model_list: list[str]):
        self.seed = ""
        self.kinds: set[str] = set()
        self.refused: set[str] = set()
        self.ladder = list(model_list) + [mdl for mdl in wire.spare_models if mdl not in model_list]
        self.lone = False
        self.recount(wire)

    def recount(self, wire: "Wire") -> None:
        self.lone = len({mdl for mdl in self.ladder if mdl not in wire.banned}) <= 1


class Wire:
    def __init__(self, bind: dict[str, Any]):
        self.t0 = monotonic()
        self.bind = dict(bind)
        self.model_list = ProxyClient(timeout=70, max_retries=1, retry_delay=1)
        self.world = ProxyClient(timeout=95, max_retries=1, retry_delay=1)
        self.spend = 0.0
        self.tick = 0
        self.done = self.broke = False
        self.log: list[dict[str, Any]] = []
        self.last_send = self.slow_until = self.model_resume = 0.0
        self.last_reply = self.t0
        self.misses = 0
        self.no_model = False
        self.banned: set[str] = set()
        self.tool_banned: set[str] = set()
        self.spare_models: list[str] = []
        self.call_models: set[str] = set()
        self.mute = False
        self.call_fail = ""

    def _scale(self) -> float:
        return vars(self).get("scale", 1.0)

    def spare(self) -> float:
        return DEADLINE - (monotonic() - self.t0)

    def spent_out(self) -> bool:
        return self.spend >= SPEND_CAP * self._scale()

    def send(self, client: Any, path: str, body: dict[str, Any]) -> Any:
        gap = PACE_GAP
        if monotonic() < self.slow_until:
            gap = SLOW_GAP
        hold = self.last_send + gap - monotonic()
        if "/inference/" in path:
            hold = max(hold, self.model_resume - monotonic())
        hold = min(hold, self.spare() - 1)
        if hold > 0:
            sleep(hold)
        self.last_send = monotonic()
        try:
            reply = client.post_verbose(path, body)
        except Exception as err:
            reply = SendFault(f"{type(err).__name__}: {err}")
        if reply.data is None:
            if (reply.error or {}).get("status") in (429, 503):
                self.slow_until = monotonic() + SLOW_FOR
        return reply

    def pause(self, seed: str, tries: int, status: Any) -> float:
        tail_sum = sum(ord(ch) for ch in seed[-64:])
        mix = (len(seed) * 7 + tail_sum + 31 * tries + 13 * self.tick) % 101 / 100.0
        first = 2.0 if status == 429 else 1.0
        return min(8.0, first * 2 ** tries) * (0.75 + 0.5 * mix)

    def down(self, at: float | None = None) -> bool:
        if self.no_model:
            return True
        if self.misses <= 0:
            return False
        now = monotonic() if at is None else at
        return now - self.last_reply > DOWN_AFTER

    def usable(self, model_list: list[str], tools: bool = False) -> bool:
        if self.no_model or self.mute:
            return False
        for mdl in list(model_list) + self.spare_models:
            if mdl in self.banned:
                continue
            if tools and mdl in self.tool_banned:
                continue
            return True
        return False

    @staticmethod
    def plain(msgs: list[dict[str, Any]]) -> list[dict[str, Any]]:
        flat = []
        for original in msgs:
            msg = dict(original)
            parts = msg.get("content")
            if isinstance(parts, list):
                msg["content"] = "\n\n".join(str(part.get("text") or "") for part in parts if isinstance(part, dict))
            flat.append(msg)
        return flat

    @staticmethod
    def cached(msgs: list[dict[str, Any]], mdl: str) -> list[dict[str, Any]]:
        if len(msgs) <= 2 or not mdl.startswith("anthropic/"):
            return msgs
        marked = [dict(m) for m in msgs]
        for idx in (0, len(marked) - 1):
            text = marked[idx].get("content")
            if text and isinstance(text, str):
                marked[idx]["content"] = [{"type": "text", "text": text, "cache_control": {"type": "ephemeral"}}]
        return marked

    def failed(self, kinds: set[str]) -> None:
        if "empty" in kinds:
            self.call_fail = "empty"
        elif kinds == {"hard"}:
            self.call_fail = "refused"
        else:
            self.call_fail = ""
        return None

    def _hard_stop(self) -> None:
        if self.no_model or self.spend < HARD_SPEND * self._scale():
            return
        self.no_model = True
        self.log.append({"kind": "spend_stop", "spend": round(self.spend, 4)})

    def _skip_sonnet(self, msgs: list[dict[str, Any]], model_list: list[str], pos: int, token_cap: int) -> bool:
        cheaper_next = pos + 1 < len(model_list) and "sonnet" not in model_list[pos + 1]
        prompt_chars = sum(len(json.dumps(m, default=str)) for m in msgs)
        guess = prompt_chars / 2.3 * 2e-6 + min(token_cap, 1500) * 1e-5
        scale = self._scale()
        if self.spent_out():
            return True
        if not cheaper_next:
            return False
        return self.spend >= SOFT_SPEND * scale or self.spend + guess > SPEND_CAP * scale

    def query_model(self, msgs: list[dict[str, Any]], model_list: list[str], token_cap: int,
            toolset: list[dict[str, Any]] | None = None) -> dict[str, Any] | None:
        self.call_fail = ""
        self.call_models = set()
        if self.mute or self.no_model:
            return None
        call = ModelCall(self, model_list)
        for pos, mdl in enumerate(call.ladder):
            self._hard_stop()
            if self.no_model:
                return self.failed(call.kinds)
            if mdl in self.banned or mdl in call.refused:
                continue
            if toolset and mdl in self.tool_banned:
                continue
            if "sonnet" in mdl and self._skip_sonnet(msgs, model_list, pos, token_cap):
                continue
            outcome, msg = self._try_model(call, mdl, msgs, model_list, token_cap, toolset)
            if outcome == "reply":
                return msg
            if outcome == "stop":
                return self.failed(call.kinds)
        return self.failed(call.kinds)

    def _envelope(self, mdl: str, msgs: list[dict[str, Any]], token_cap: int, plain: bool,
                  toolset: list[dict[str, Any]] | None) -> dict[str, Any]:
        body: dict[str, Any] = {"model": mdl, "messages": self.plain(msgs) if plain else self.cached(msgs, mdl),
                                "temperature": 0, "max_tokens": token_cap}
        if toolset:
            body["tools"] = toolset
            body["tool_choice"] = "auto"
        return body

    def _answered(self, call: ModelCall, mdl: str, data: dict[str, Any]) -> tuple[str, Any]:
        self.last_reply, self.misses = monotonic(), 0
        usage = data.get("usage") or {}
        if TextKit.is_number(usage.get("cost")):
            self.spend += float(usage["cost"])
        msg = data["choices"][0].get("message") or {}
        if msg.get("content") or msg.get("tool_calls"):
            return "reply", msg
        call.kinds.add("empty")
        self.log.append({"kind": "inference_failure", "model": mdl, "error": "empty"})
        return "next", None

    def _refused_model(self, call: ModelCall, mdl: str, status: int, body: str, said: str,
                       model_list: list[str], toolset: Any) -> None:
        call.kinds.add("hard")
        call.refused.add(mdl)
        if status == 404 and toolset and "tool" in said:
            self.tool_banned.add(mdl)
        elif not (status == 403 and re.search(r"moderat|flagged", said)):
            self.banned.add(mdl)
        if status != 403:
            return
        listed = re.search(r'allowed_models"?\s*:\s*\[([^\]]*)', body)
        if not listed or self.spare_models:
            return
        offered = re.findall(r'"([^"\s]{3,100})"', listed.group(1))
        self.spare_models = [m for m in offered if m not in self.banned and m not in model_list][:2]
        call.ladder += [m for m in self.spare_models if m not in call.ladder]
        call.recount(self)

    def _try_model(self, call, mdl, msgs, model_list, token_cap, toolset):
        plain = False
        attempts = 1 if self.down() else MODEL_TRIES
        while attempts > 0:
            attempts -= 1
            left = self.spare()
            if left < LAST_CALL - 10:
                return "stop", None
            cap = 30 if self.down() else 75
            self.model_list.timeout = int(max(8, min(cap, left - LAST_CALL + 15)))
            body = self._envelope(mdl, msgs, token_cap, plain, toolset)
            self.call_models.add(mdl)
            started = monotonic()
            answer = self.send(self.model_list, "/inference/chat/completions", body)
            if isinstance(answer.data, dict) and answer.data.get("choices"):
                outcome = self._answered(call, mdl, answer.data)
                # an empty reply is asked once more of the same model before a dearer one takes over
                if outcome[0] == "next" and attempts > 0 and self.spare() > LAST_CALL + 20:
                    continue
                return outcome
            self.misses += 1
            self.log.append({"kind": "inference_failure", "model": mdl,
                             "error": str(answer.error)[:300] if answer.error else "empty"})
            fault = answer.error or {}
            status = fault.get("status")
            verdict = self._judge_failure(call, mdl, status, str(fault.get("body") or ""), plain, model_list, toolset)
            if verdict == "plain":
                plain = True
                continue
            verdict = verdict or self._soft_wait(call, mdl, msgs, status, started)
            if verdict:
                return verdict, None
        return "next", None

    def _judge_failure(self, call, mdl, status, text, plain, model_list, toolset):
        said = text.lower()
        if status == 402 or (status == 403 and re.search(r"key limit|insufficient|credit", said)):
            self.no_model = True
            return "stop"
        if status in (401, 403, 404):
            self._refused_model(call, mdl, status, text, said, model_list, toolset)
            return "next"
        if not isinstance(status, int) or not 400 <= status < 500 or status in (408, 425, 429):
            call.kinds.add("soft")
            return ""
        call.kinds.add("hard")
        if not plain and status in (400, 413, 422):
            return "plain"
        call.refused.add(mdl)
        return "next"

    def _soft_wait(self, call, mdl, msgs, status, started):
        if monotonic() - started > 20:
            return "stop" if self.down() and (call.lone or len(self.call_models) >= 2) else "next"
        call.seed = call.seed or json.dumps(msgs[-1:], default=str)[-600:]
        delay = self.pause(call.seed + mdl, min(self.misses - 1, 6), status)
        if self.spare() - delay < LAST_CALL - 10:
            return "stop"
        self.model_resume = monotonic() + delay
        if self.down(self.model_resume) and (call.lone or len(self.call_models) >= 2):
            return "stop"
        return ""

    def consult(self, ask_text: str, token_cap: int, model_list: list[str]) -> dict[str, Any] | None:
        reply = self.query_model([{"role": "user", "content": ask_text}], model_list, token_cap)
        return TextKit.json_object((reply or {}).get("content") or "")

    def _envelope_for(self, acts: list[dict[str, Any]]) -> dict[str, Any]:
        body: dict[str, Any] = dict(self.bind)
        body["step"] = self.tick + 1
        if len(acts) != 1:
            body["actions"] = acts
        else:
            body["action"] = acts[0]
        return body

    def _taken(self, data: dict[str, Any], acts: list[dict[str, Any]]) -> dict[str, Any]:
        step = data.get("step")
        self.tick = step if isinstance(step, int) else self.tick + len(acts)
        if data.get("finished"):
            self.done = True
        return data

    def act(self, acts):
        body = self._envelope_for(acts)
        failed = 0
        reuses = {"n": 0}
        rounds = 3 + WORLD_TRIES
        while rounds:
            rounds -= 1
            left = self.spare()
            if left < 2:
                break
            self.world.timeout = int(max(3, min(95, left - 1)))
            answer = self.send(self.world, "/environment/call", body)
            if isinstance(answer.data, dict):
                return self._taken(answer.data, acts)
            fault = answer.error or {}
            status, said = fault.get("status"), str(fault.get("body") or "")
            self.log.append({"kind": "env_failure", "status": status, "body": said[:300]})
            if status == 409:
                if self._step_clash(said, body, reuses):
                    continue
                break
            if status == 402:
                self.broke = True
                break
            if status in (403, 422) or failed >= WORLD_TRIES - 1 or (failed >= 1 and status in (400, 401, 404)):
                break
            delay = self.pause(json.dumps(body, default=str)[-600:], failed, status)
            failed += 1
            if self.spare() - delay < 2:
                break
            sleep(delay)
        return None

    def _step_clash(self, said, body, reuses):
        if "finished" in said:
            self.done = True
            return False
        expected = re.search(r"expected (\d+)", said)
        if expected:
            self.tick = int(expected.group(1)) - 1
            body["step"] = self.tick + 1
            return True
        if reuses["n"] >= 2 or "reused" not in said:
            return False
        reuses["n"] += 1
        body["step"] += 1
        return True


class Records:
    SHELF_KEY = r"categor|shelf|department|aisle"

    def __init__(self):
        self.items: list[Any] = []
        self.ids: set[str] = set()
        self.read: set[str] = set()
        self.label_of: dict[str, str] = {}

    @staticmethod
    def _own_label(d: dict[str, Any]) -> str:
        for key, val in d.items():
            if not isinstance(val, str):
                continue
            text = val.strip()
            low = str(key).lower()
            if 0 < len(text) <= 60 and re.search(Records.SHELF_KEY, low) and "id" not in low:
                return text
        return ""

    @staticmethod
    def _has_map(d: dict[str, Any], least: int) -> bool:
        return any(isinstance(val, dict) and len(val) >= least for val in d.values())

    @staticmethod
    def _priced(d: dict[str, Any]) -> float | None:
        for key, val in d.items():
            if TextKit.is_number(val) and not isinstance(val, bool) and re.search(r"price|cost", str(key).lower()):
                return float(val)
        return None

    def _read_id(self, d: dict[str, Any]) -> str | None:
        for key, val in d.items():
            if isinstance(val, str) and "id" in str(key).lower() and val in self.read:
                return val
        return None

    def add(self, obs: Any) -> None:
        self.items.append(obs)

        def note(d: dict[str, Any]) -> None:
            ident = None
            for key, piece in d.items():
                if not isinstance(piece, str):
                    continue
                low = str(key).lower()
                if "id" in low and re.fullmatch(ID_LIKE, piece):
                    self.ids.add(piece)
            for key, piece in d.items():
                if not isinstance(piece, str):
                    continue
                low = str(key).lower()
                if low in ("id", "_id") or (low.endswith("id") and re.fullmatch(ID_LIKE, piece)):
                    ident = piece
                    break
            if not ident:
                return
            small_maps = (val for val in d.values() if isinstance(val, dict))
            if any(len([v for v in val.values() if isinstance(v, (str, int, float))]) >= 2 for val in small_maps):
                self.read.add(ident)
            own = self._own_label(d)
            if own:
                vars(self).setdefault("label_of", {}).setdefault(ident, own)
        TextKit.walk_dicts(obs, note)

    def price(self, thing: str) -> float | None:
        prices: list[float] = []

        def note(d: dict[str, Any]) -> None:
            names_it = any(isinstance(val, str) and val == thing and "id" in str(key).lower() for key, val in d.items())
            if names_it:
                figure = self._priced(d)
                if figure is not None:
                    prices.append(figure)
        for rec in self.items:
            TextKit.walk_dicts(rec, note)
        return prices[-1] if prices else None

    def about(self, thing: str, hold: int = 4) -> list[str]:
        hits: list[str] = []

        def dig(piece: Any, depth: int) -> None:
            if depth > 6 or len(hits) >= 8:
                return
            if isinstance(piece, list):
                for val in piece:
                    dig(val, depth + 1)
                return
            if not isinstance(piece, dict):
                return
            if thing in [val for val in piece.values() if isinstance(val, str)]:
                shown = TextKit.as_text(TextKit.clipped(piece, 400))[:2500]
                if shown not in hits:
                    hits.append(shown)
                return
            for val in piece.values():
                dig(val, depth + 1)

        for rec in self.items:
            dig(rec, 0)
        return hits[-hold:]

    def cards(self, char_cap: int = 30000, cheaper_than: str = "", skip: set[str] | None = None, first: Any = ()) -> list[str]:
        merged: dict[str, dict[str, Any]] = {}
        seen_order: list[str] = []

        def gather(d: dict[str, Any]) -> None:
            ident = None
            for key, val in d.items():
                if isinstance(val, str) and "id" in str(key).lower() and re.fullmatch(ID_LIKE, val) and val in self.ids:
                    ident = val
                    break
            if ident is None:
                return
            if len(d) < 4 and not self._has_map(d, 2):
                return
            if ident not in seen_order:
                seen_order.append(ident)
            merged.setdefault(ident, {}).update(dict(d.items()))
            if self._has_map(d, 2):
                self.read.add(ident)

        for rec in self.items:
            TextKit.walk_dicts(rec, gather)

        def cost(ident: str) -> float | None:
            return self._priced(merged.get(ident) or {})

        limit = cost(cheaper_than) if cheaper_than else None
        if skip:
            seen_order = [i for i in seen_order if i == cheaper_than or i not in skip]
        if limit is not None:
            kept = []
            for ident in seen_order:
                if ident == cheaper_than or cost(ident) is None or cost(ident) < limit:
                    kept.append(ident)
            seen_order = kept
        lead = set(first or ())

        def rank(ident: str) -> tuple:
            return (ident not in lead, ident not in self.read, seen_order.index(ident))
        ranked = sorted(seen_order, key=rank)
        texts = {}
        for ident in ranked:
            head = "DETAILS " if ident in self.read else "ROW "
            texts[ident] = head + TextKit.as_text(TextKit.clipped(merged[ident], 160))
        if sum(map(len, texts.values())) > char_cap:
            def sold(ident: str) -> bool:
                return any(Reading.key_stock(None, key, val) is False for key, val in merged[ident].items())
            ranked = sorted(ranked, key=lambda i: (i not in lead, i not in self.read, sold(i), seen_order.index(i)))
        out, used = [], 0
        for ident in ranked:
            text = texts[ident]
            if used + len(text) <= char_cap:
                out.append(text)
                used += len(text)
        return out

    def label_for(self, d: dict[str, Any]) -> str:
        own = self._own_label(d)
        if own:
            return own
        known = vars(self).get("label_of") or {}
        for key, val in d.items():
            if isinstance(val, str) and val in known and "id" in str(key).lower():
                return known[val]
        return ""

    def rich_reads(self, shelf: str = "") -> int:
        rich: set[str] = set()
        wanted = shelf.strip().lower()

        def note(d: dict[str, Any]) -> None:
            ident = self._read_id(d)
            if not ident or (shelf and self.label_for(d).strip().lower() != wanted):
                return
            maps = [val for val in d.values() if isinstance(val, dict)]
            if maps:
                if any(len(val) >= 3 for val in maps):
                    rich.add(ident)
                return
            if len([val for val in d.values() if isinstance(val, str) and 0 < len(val) <= 70]) >= 5:
                rich.add(ident)
        for rec in self.items:
            TextKit.walk_dicts(rec, note)
        return len(rich)

    def shelves_read(self) -> list[str]:
        counts: dict[str, int] = {}

        def note(d: dict[str, Any]) -> None:
            if not self._read_id(d):
                return
            label = self.label_for(d)
            if label and any(isinstance(val, dict) for val in d.values()):
                counts[label] = counts.get(label, 0) + 1
        for rec in self.items:
            TextKit.walk_dicts(rec, note)
        return sorted(counts, key=lambda label: -counts[label])

    def prop_menu(self, shelf=""):
        stats = {}
        counter = {"sets": 0, "mapped": 0}
        wanted = shelf.strip().lower()

        def belongs(d):
            for key, val in d.items():
                if isinstance(val, str) and val.strip().lower() == shelf.lower() and re.search(Records.SHELF_KEY, str(key).lower()):
                    return True
            return self.label_for(d).strip().lower() == wanted

        def tally(d):
            brief = [(key, val) for key, val in d.items() if isinstance(val, str) and 0 < len(val) <= 70 and not TextKit.junk_key(key)]
            flat = not any(isinstance(v, dict) for v in d.values())
            if len(brief) < 5 and not (shelf and len(brief) >= 2 and flat):
                return
            counter["sets"] += 1
            row_like = len(brief) < 5 and any(isinstance(val, str) and "id" in str(key).lower() for key, val in d.items())
            counter["mapped"] += 0 if row_like else 1
            for key, val in brief:
                if re.search(r"\d", val) and re.fullmatch(CODE_LIKE, val) and ("-" in val or "_" in val):
                    continue
                entry = stats.setdefault(re.sub(r"[_.]+", " ", re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", str(key))).strip().lower(),
                                         {"hits": 0, "rows": 0, "spread": {}, "shown": []})
                low = val.lower()
                entry["hits"] += 1
                entry["rows"] += 1 if row_like else 0
                entry["spread"][low] = entry["spread"].get(low, 0) + 1
                if low not in entry["shown"] and len(entry["shown"]) < 5:
                    entry["shown"].append(low)

        stack = [(rec, False, 0) for rec in reversed(self.items)]
        while stack:
            piece, inside, depth = stack.pop()
            if depth > 6:
                continue
            if isinstance(piece, dict):
                inside = inside or not shelf or belongs(piece)
                if inside:
                    tally(piece)
                kids = [val for val in piece.values() if isinstance(val, (dict, list))]
            elif isinstance(piece, list):
                kids = list(piece)
            else:
                continue
            stack.extend((kid, inside, depth + 1) for kid in reversed(kids))
        if counter["mapped"]:
            for name in [name for name, e in stats.items() if e["rows"] >= e["hits"]]:
                del stats[name]
        sets = counter["sets"]

        def worth(name):
            e = stats[name]
            total = sum(e["spread"].values())
            share = max(e["spread"].values()) / total if total else 1.0
            return min(1.0, e["hits"] / max(1, sets)) * (1.0 - share)

        def varied(name):
            return len(stats[name]["shown"]) >= 2 and stats[name]["hits"] >= 2

        def with_unit(name):
            vals = stats[name]["shown"]
            return bool(vals) and all(re.search(r"\d", v) for v in vals) and any(re.search(r"\d\s*[a-z\"']", v) for v in vals)

        if sets >= 3:
            order = sorted(stats, key=lambda name: (-worth(name), -stats[name]["hits"], name))
        else:
            order = sorted(stats, key=lambda name: (-varied(name), -stats[name]["hits"], -len(stats[name]["shown"])))
        lines = []
        for name in order[:28]:
            e = stats[name]
            tags = f" [values differ among the listings: {len(e['shown'])} seen]" if varied(name) else ""
            if with_unit(name):
                tags += " [figures with a unit]"
            lines.append(f"- {name} (in {e['hits']} listings){tags}: " + " | ".join(e["shown"]))
        return "\n".join(lines), sets

    def label_lines(self) -> str:
        counts: dict[str, dict[str, int]] = {}

        def note(d: dict[str, Any]) -> None:
            if len(d) >= 4:
                for key, val in d.items():
                    if not isinstance(val, str) or not 0 < len(val) <= 40:
                        continue
                    if re.search(r"\d", val) and re.fullmatch(CODE_LIKE, val):
                        continue
                    bucket = counts.setdefault(str(key), {})
                    bucket[val] = bucket.get(val, 0) + 1

        for rec in self.items:
            TextKit.walk_dicts(rec, note)
        out = []
        for key, bucket in counts.items():
            if sum(bucket.values()) < 4 or not 2 <= len(bucket) <= 8 or max(bucket.values()) < 2:
                continue
            ranked = sorted(bucket.items(), key=lambda kv: -kv[1])
            out.append(f"- {key}: " + " | ".join(f"{val} (x{n})" for val, n in ranked))
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
        self.approval = ""
        self.widened_at = -1
        self.widened_tick = -10 ** 6
        self.sweepable = False
        self.resolved: list[str] = []
        self.ref_news: list[dict[str, Any]] = []
        self.told: list[dict[str, Any]] = []
        self.told_version = 0
        self.allowance_news: dict[str, Any] | None = None

    @staticmethod
    def _squeezed(text: Any) -> str:
        return " ".join(str(text or "").split())

    @staticmethod
    def _tags(head: str) -> list[str]:
        return re.findall(r"\(([^)]*)\)", head)

    @staticmethod
    def _count(value: Any) -> bool:
        return isinstance(value, int) and not isinstance(value, bool) and value >= 1

    def add_told(self, items: Any, source: str) -> list[str]:
        fresh: list[str] = []
        if not isinstance(items, list):
            return fresh
        known = {self._squeezed(t.get("text")).lower() for t in self.told}
        for entry in items[:12]:
            if isinstance(entry, str):
                entry = {"text": entry, "before_finish": None, "needs_message": None}
            if not isinstance(entry, dict):
                continue
            words = self._squeezed(entry.get("text"))
            if not words or words.lower() in known:
                continue
            known.add(words.lower())
            row = {"text": words[:400], "source": source}
            for flag in ("before_finish", "needs_message", "before_change"):
                row[flag] = entry.get(flag) if isinstance(entry.get(flag), bool) else None
            self.told.append(row)
            fresh.append(words[:120])
        if fresh:
            self.told_version += 1
        return fresh

    def finish_told(self) -> list[dict[str, Any]]:
        return [t for t in self.told if t.get("before_finish") is not False]

    def told_text(self) -> str:
        if not self.told:
            return ""
        parts = []
        for n, t in enumerate(self.told[-10:], 1):
            marks = []
            if t.get("before_finish"):
                marks.append("before the finish")
            if t.get("needs_message"):
                marks.append("by a message")
            origin = f"from the {t.get('source')}"
            tail = f" [{', '.join(marks)}; {origin}]" if marks else f" [{origin}]"
            parts.append(f"{n}) {t['text']}" + tail)
        return "\n[instructions in force] " + " ".join(parts)

    def items(self) -> list[dict[str, Any]]:
        lines = []
        for row in self.reading.get("items") or []:
            if not isinstance(row, dict) or not str(row.get("product") or "").strip():
                continue
            qty, dep = row.get("quantity"), row.get("depends_on")
            reqs = row.get("requirements") if isinstance(row.get("requirements"), list) else []
            lines.append({"product": str(row["product"]).strip(),
                          "quantity": qty if self._count(qty) else 1,
                          "requirements": [str(r) for r in reqs if r and ":" in str(r)],
                          "depends_on": dep - 1 if self._count(dep) else None})
        return lines if len(lines) > 1 else []

    def quantity(self) -> int:
        if self.items():
            return 1
        wanted = self.reading.get("quantity")
        return wanted if self._count(wanted) else 1

    def counts(self) -> list[int]:
        return [row["quantity"] for row in self.items()] or [self.quantity()]

    def items_text(self) -> str:
        rows = self.items()
        if not rows:
            return ""
        listed = " | ".join(f"{n}) {row['product']} x{row['quantity']}" for n, row in enumerate(rows, 1))
        return ("THE REQUEST ASKS FOR SEVERAL PRODUCTS; the order holds exactly one line per product, at its quantity, and nothing "
                "else; each listing is judged only against the requirements of the product it is (its category decides), and the "
                "cheapest or best rule applies within each product: " + listed)

    def priority(self) -> str:
        rule = self.reading.get("priority")
        return rule if rule in ("cheapest", "best") else "none"

    def props_left(self) -> list[str]:
        if self.now_reqs is None:
            pool = [r for r in self.written + self.uncovered if r not in self.let_go]
        else:
            pool = self.now_reqs
        plain = []
        for req in pool:
            if req in self.let_go or re.search(r"\bprice\b|budget|quantity|\bstock\b|\bcents?\b", req.lower()):
                continue
            plain.append(req)
        return plain

    def text_body(self) -> str:
        if self.now_reqs is None:
            parts = [f"{req} (stated)" for req in self.written if req not in self.let_go]
            parts.extend(f"{req} (revealed by the shopper)" for req in self.uncovered if req not in self.let_go)
        else:
            parts = list(self.now_reqs)
        if parts:
            out = "; ".join(parts)
        elif self.let_go:
            out = "no property requirement is left: any listing of the requested category that is in stock fits"
        else:
            out = "see the request"
        if self.spend_cap is not None:
            out += f"; total price at most {self.spend_cap} {self.reading.get('currency') or ''}".rstrip()
        out += "; " + self.items_text() if self.items() else f"; quantity {self.quantity()}"
        if self.let_go:
            out += "; no longer required: " + "; ".join(self.let_go)
        if self.indifferent:
            out += "; shopper does not care about: " + ", ".join(self.indifferent)
        return out

    def changes(self) -> str:
        if not self.let_go:
            return ""
        return ("\n\nCHANGES AFTER THE REQUEST (these win over its wording): the shopper DROPPED " + "; ".join(self.let_go)
                + ". Ignore a dropped requirement entirely: no listing fails, ranks lower or wins because of it.")

    def overview(self, step_count: int, msgs: int | None) -> str:
        got = self.reading
        if not got:
            return ""
        out = ["How the request was first read (compare it with the request text itself):"]
        if got.get("product"):
            out.append(f"- product: {got.get('product')}")
        several = self.items()
        if not several:
            out.append(f"- quantity: {self.quantity()}")
        else:
            for n, row in enumerate(several, 1):
                out.append(f"- product {n}: {row['product']}, quantity {row['quantity']} (a cart line of its own)")
            out.append("- the order contains exactly these lines, a single one per product and nothing more; when the cart tool accepts a list of lines, include all of them in every cart call")
        out.extend(f"- stated requirement: {req}" for req in got.get("requirements") or [])
        out.extend(f"- value set by reference (open that listing first, then search using its value): {req}" for req in got.get("references") or [])
        if got.get("price_cap") is not None:
            out.append(f"- price cap (total): {got.get('price_cap')} {got.get('currency') or ''}".rstrip())
        if got.get("in_stock_only"):
            out.append("- in-stock / on-hand items only")
        if self.priority() == "none":
            out.append("- the request sets no cheapest/best rule: any item that meets every requirement within the budget is correct, so spend no steps chasing the lowest price (unless the request itself asks for it)")
        else:
            extra = f" ({got.get('priority_detail')})" if got.get("priority_detail") else ""
            out.append(f"- among every fitting item, pick the {self.priority()}{extra}")
        out.extend(f"- a wish, NOT a requirement (it only breaks ties): {w}" for w in got.get("soft_wants") or [])
        out.extend(f"- ask the shopper about this first: {a}" for a in got.get("ask_first") or [])
        out.extend(f"- rule: {req}" for req in got.get("rules") or [])
        if msgs is not None:
            out.append(f"- messages the shopper may receive: {msgs}")
        out.append(f"- step allowance: {step_count}")
        return "\n".join(out)

    @staticmethod
    def line_names(needs: list[Any], lns: list[str]) -> dict[str, str]:
        words = re.findall(r"[a-z]+", " ".join(lns).lower())
        joined = " ".join(words)
        renamed: dict[str, str] = {}
        for need in needs:
            if not (isinstance(need, str) and ":" in need):
                continue
            head, _, value = need.partition(":")
            tag = head[head.find("("):].strip() if "(" in head else ""
            parts = re.findall(r"[a-z]+", re.split(r"\s*\(", head)[0].lower())
            if not parts or min(len(p) for p in parts) < 3:
                continue
            if re.search(r"\b" + r"\s+".join(parts) + r"\b", joined):
                continue
            width = len(parts)
            for start in range(len(words) - width + 1):
                run = words[start: start + width]
                if run == parts or not all(w.startswith(p) for p, w in zip(parts, run)):
                    continue
                renamed[need] = " ".join(run) + (" " + tag if tag else "") + ":" + value
                break
        return renamed

    def _phrases(self, lns: list[str]) -> set[str]:
        words = re.findall(r"[a-z]+", " ".join(lns).lower())
        grams = set()
        for size in (1, 2, 3):
            for start in range(len(words) - size + 1):
                grams.add(" ".join(words[start: start + size]))
        return {g for g in grams if len(g) >= 3 and g not in LINK_WORDS}

    def drop_named(self, old_need: str, lns: list[str]) -> bool:
        try:
            name = str(old_need).split(":", 1)[0]
            tag = self._tags(name)

            def matches(other: str) -> bool:
                head = other.split(":", 1)[0]
                return PropKit.same_prop(head, name) and (not tag or self._tags(head) in ([], tag))
            stated = [w for w in self.written if matches(w)]
            if not stated:
                fresh = [n for n in vars(self).get("fresh_needs") or [] if isinstance(n, str) and ":" in n]
                stated = [w for w in self.uncovered if matches(w) and w not in self.written and w not in fresh]
                if not stated or any(matches(n) and n != old_need for n in fresh):
                    return True
            phrases = self._phrases(lns)
            for w in stated:
                if any(PropKit.same_prop(w.split(":", 1)[0], p) for p in phrases):
                    return True
            hook = vars(self).get("field_hook")
            if not hook:
                return False
            for w in stated:
                head, _, value = w.partition(":")
                value = re.sub(r"\((?:stated|revealed[^)]*)\)", " ", value)
                if RefKit.ref_text(value) or not ValueKit.ch_alts(value):
                    continue
                fields = hook(head) or {}
                others = [r.split(":", 1)[0] for r in self.written + self.uncovered
                          if isinstance(r, str) and ":" in r and not PropKit.same_prop(r.split(":", 1)[0], name)]
                for p in phrases:
                    if any(PropKit.strict_prop(o, p) for o in others):
                        continue
                    for key, vals in fields.items():
                        if PropKit.strict_prop(key, p) and ValueKit.value_seen(value, " | ".join(vals)):
                            self.wire.log.append({"kind": "drop_by_field", "need": w[:200], "phrase": p})
                            return True
            return False
        except Exception:
            return True

    @staticmethod
    def _bare(text: str) -> str:
        return " ".join(re.findall(r"[a-z0-9]+", text.lower()))

    def literal_values(self, answer: dict[str, Any]) -> None:
        hook = vars(self).get("field_hook")
        if not hook:
            return
        swaps: dict[str, str] = {}
        for need in answer.get("needs") or []:
            if not (isinstance(need, str) and ":" in need):
                continue
            head, _, value = need.partition(":")
            if re.search(r"\d", value) or RefKit.ref_text(value):
                continue
            stripped = self._bare(re.sub(CJ_COMPARE, " ", value.lower()))
            if not stripped:
                continue
            if stripped == self._bare(value) and ValueKit.positive_value(value, head):
                continue
            prop = re.split(r"\s*\(", head)[0].strip()
            for key, vals in (hook(head) or {}).items():
                if not PropKit.strict_prop(key, prop):
                    continue
                written = next((v for v in vals if self._bare(str(v)) == stripped), None)
                if written is not None:
                    LITERAL_VALUES.add(stripped)
                    swaps[need] = f"{head.strip()}: {str(written).strip().lower()}"
                    break
        if not swaps:
            return
        self._swap_needs(answer, swaps)
        self.wire.log.append({"kind": "literal_value", "needs": list(swaps.values())[:4]})

    @staticmethod
    def _swap_needs(answer: dict[str, Any], swaps: dict[str, str]) -> None:
        answer["needs"] = [swaps.get(n, n) if isinstance(n, str) else n for n in answer.get("needs") or []]
        if isinstance(answer.get("requirements_now"), list):
            answer["requirements_now"] = [swaps.get(r, r) if isinstance(r, str) else r for r in answer["requirements_now"]]

    def pinned(self, prop_name: str, qtext: str | None) -> bool:
        try:
            given = self.reading.get("references")
            refs = [str(r) for r in given or []] if isinstance(given, list) else []
            refs += list(vars(self).get("resolved") or [])
            tag = self._tags(str(prop_name))
            asked_key = PropKit.prop_key(prop_name)
            for req in self.written:
                if req in self.let_go or ":" not in req:
                    continue
                head, _, value = req.partition(":")
                own_tag = self._tags(head)
                if tag and own_tag and not (PropKit.kind_words(tag[-1]) & PropKit.kind_words(own_tag[-1])):
                    continue
                if not ValueKit.positive_value(value, head) or ValueKit.story_guess(req, self.request, refs):
                    continue
                own_key = PropKit.prop_key(head)
                if PropKit.same_prop(head, prop_name) or (len(own_key) >= 4 and asked_key.endswith(own_key)):
                    return True
        except Exception:
            return False
        return False

    @staticmethod
    def valued(needs: Any, lns: list[str]) -> bool:
        said = " ".join(str(ln) for ln in lns).lower()
        if not isinstance(needs, list):
            return False
        for need in needs:
            head, _, value = str(need).partition(":")
            names = [w for w in PropKit.prop_words(head) if len(w) >= 3]
            values = [w for w in re.findall(r"[a-z0-9]+", value.lower()) if w not in LINK_WORDS]
            if names and values and all(w in said for w in names + values):
                return True
        return False

    def _listener_prompt(self, so_far: str, qtext: str | None, new_lines: list[str]) -> str:
        sections = [HEAR, "THE SHOPPER'S REQUEST (for context):\n" + self.request, "REQUIREMENTS SO FAR:\n" + so_far]
        if self.exchanges:
            sections.append("EARLIER EXCHANGES WITH THE SHOPPER (oldest first):\n" + "\n".join(self.exchanges[-12:]))
        if qtext:
            sections.append("THE ASSISTANT ASKED:\n" + qtext)
        sections.append("THE SHOPPER NOW SAID:\n" + "\n".join(new_lines))
        return "\n\n".join(sections)

    def _park(self, new_lines: list[str], qtext: str | None) -> None:
        self.caught.difference_update(new_lines)
        self.exchanges.pop()
        self.pending = list(vars(self).get("pending") or []) + [(list(new_lines), qtext)]
        self.pending_at = monotonic()

    def _restore_lost(self, masked: list[str]) -> None:
        for need in self.uncovered:
            prop_name = need.split(":", 1)[0].strip().lower()
            if not prop_name:
                continue
            if any(prop_name in req.lower() for req in self.now_reqs) or any(prop_name in old.lower() for old in self.let_go):
                continue
            self.now_reqs.append(need)
        for need in self.written:
            try:
                if need in self.let_go or need in self.now_reqs or self.drop_named(need, masked):
                    continue
                name = need.split(":", 1)[0]
                tag = self._tags(name)

                def same_slot(other: str) -> bool:
                    head = other.split(":", 1)[0]
                    return PropKit.same_prop(name, head) and self._tags(head) == tag
                if any(same_slot(old) for old in self.let_go):
                    continue
                if not any(same_slot(req) for req in self.now_reqs if isinstance(req, str)):
                    self.now_reqs.append(need)
            except Exception:
                continue

    def _widened_remark(self) -> str:
        if self.props_left():
            return ("a dropped requirement WIDENS what is acceptable: listings that failed only on it fit now, and "
                    "listings never searched for may beat the pick; search the item kind again without it, as short "
                    "queries of each remaining required value with the kind word, value first, reading "
                    "every page, and choose the cheapest over everything acceptable now")
        shelf_rule = ("no property requirement is left: any listing of the requested category that is in stock fits, whatever "
                      "object it is (an accessory, a part or a set shelved there counts), and the cheapest on the whole shelf "
                      "wins. Rows come by relevance or id, not by price")
        if self.priority() == "cheapest" and self.sweepable:
            return (shelf_rule + ", so further slices of the shelf are read for you next "
                    "(its listing's pages, the label's own words, words naming the kinds of object in the titles); if no "
                    "'[shelf sweep done]' list follows, cover the shelf yourself with short queries, the first page of "
                    "each - the category label, each kind word of the label alone, each item name that recurs in the "
                    "titles seen - and the category listing's pages. Then make the lowest-priced in-stock row on the "
                    "requested category within the budget the cart's only line, re-check its stock and price, and order it")
        return (shelf_rule + ": cover the shelf with short queries, the first page "
                "of each - the category label, each kind word of the label alone, each item name that recurs in the "
                "titles seen - and the category listing's pages; then order the lowest in-stock price")

    def _screen_meta(self, answer, qtext, new_lines):
        try:
            only_meta = answer.get("meta_only") is True and qtext
            if only_meta and not self.valued(answer.get("needs"), new_lines):
                self.meta_last = qtext
                self.wire.log.append({"kind": "meta_only", "line": " / ".join(new_lines)[:200]})
                return dict(answer, needs=[], no_preference=[], dropped=[], references=[], requirements_now=None,
                            new_budget=None, unavailable=False, reply="")
        except Exception:
            pass
        return answer

    def _masked_lines(self, answer, new_lines, qtext):
        hook = vars(self).get("ref_hook")
        if not hook:
            return new_lines
        try:
            return hook(answer, new_lines, qtext) or new_lines
        except Exception as err:
            self.wire.log.append({"kind": "ref_line_error", "error": f"{type(err).__name__}: {err}"[:300]})
            return new_lines

    def _take_needs(self, answer, grounded, notes, new_lines):
        heard = answer.get("needs") or []
        self.fresh_needs = list(filter(lambda n: isinstance(n, str), heard))
        rebuilt = answer.get("requirements_now")
        if isinstance(rebuilt, list) and all(isinstance(req, str) for req in rebuilt):
            self.now_reqs = list(filter(lambda req: req.strip() and grounded(req), rebuilt))
        for need in heard:
            fresh = isinstance(need, str) and need and need not in self.uncovered
            if fresh and grounded(need):
                self.uncovered.append(need)
                notes.append(f"new requirement: {need}")
        try:
            refs = answer.get("references")
            infos = list(filter(lambda r: isinstance(r, dict), refs or [])) if isinstance(refs, list) else []
            pointing = answer.get("ref_needs") or []
            for need in answer.get("needs") or []:
                if isinstance(need, str) and need in self.uncovered:
                    if RefKit.ref_text(need.partition(":")[2]) or need in pointing:
                        entry = {"need": need, "infos": infos, "line": " ".join(new_lines)}
                        self.ref_news = list(vars(self).get("ref_news") or []) + [entry]
        except Exception:
            pass

    def _take_free(self, answer, qtext):
        source = self.written + self.uncovered if self.now_reqs is None else self.now_reqs
        held = set()
        for r in source:
            if isinstance(r, str) and ":" in r and r not in self.let_go:
                held.add(r.split(":", 1)[0].strip().lower())
        for prop_name in answer.get("no_preference") or []:
            usable = isinstance(prop_name, str) and prop_name
            if not usable or prop_name in self.indifferent or prop_name.strip().lower() in held:
                continue
            if not self.pinned(prop_name, qtext):
                self.indifferent.append(prop_name)
            else:
                self.pinned_heard = vars(self).get("pinned_heard", 0) + 1

    def _take_drops(self, answer, masked, notes):
        widened = False
        budget_moved = TextKit.is_number(answer.get("new_budget"))
        money = r"\b(?:prices?|budget|costs?|spend\w*|cents?|dollars?)\b"
        for old_need in answer.get("dropped") or []:
            if not isinstance(old_need, str) or not old_need or old_need in self.let_go:
                continue
            if budget_moved and re.search(money, old_need.lower()):
                continue
            if self.drop_named(old_need, masked):
                self.let_go.append(old_need)
                notes.append(f"dropped: {old_need}")
                widened = widened or self.priority() != "none"
            else:
                self.wire.log.append({"kind": "drop_kept", "need": old_need[:200]})
        return widened

    def _new_count(self, change: Any) -> str:
        if not isinstance(change, dict) or not self._count(change.get("count")):
            return ""
        count = int(change["count"])
        rows = [row for row in self.reading.get("items") or [] if isinstance(row, dict) and str(row.get("product") or "").strip()]
        if len(rows) <= 1:
            if self.reading.get("quantity") == count:
                return ""
            self.reading["quantity"] = count
            if rows:
                rows[0]["quantity"] = count
            return f"new quantity: {count}"
        said = set(PropKit.kind_tokens(str(change.get("product") or "")))
        kinds = [set(PropKit.kind_tokens(str(row["product"]))) for row in rows]
        marks = [PropKit.kind_score(own, said, set().union(*(kinds[j] for j in range(len(kinds)) if j != k))) for k, own in enumerate(kinds)]
        best = max(marks)
        if best <= 0 or marks.count(best) != 1:
            return ""
        row = rows[marks.index(best)]
        if row.get("quantity") == count:
            return ""
        row["quantity"] = count
        return f"new quantity of the {row['product']}: {count}"

    def _take_terms(self, answer, qtext, notes):
        cap = answer.get("new_budget")
        if TextKit.is_number(cap) and cap > 0 and cap != self.spend_cap:
            self.spend_cap = cap
            notes.append(f"new budget: {cap}")
        if answer.get("unavailable"):
            notes.append("an item became unavailable; take it out of the cart and do not order it")
        try:
            for change in answer.get("new_quantity") or []:
                said = self._new_count(change)
                if said:
                    notes.append(said)
        except Exception as err:
            self.wire.log.append({"kind": "quantity_error", "error": f"{type(err).__name__}: {err}"[:300]})
        try:
            origin = "store" if qtext is None else "shopper"
            added = self.add_told(answer.get("instructions"), origin)
            if added:
                notes.append("new instruction: " + "; ".join(added))
            allowance = answer.get("allowance")
            figures = {}
            if isinstance(allowance, dict):
                figures = {k: v for k, v in allowance.items() if TextKit.is_number(v) and not isinstance(v, bool)}
            if figures:
                self.allowance_news = figures
                notes.append("new allowance: " + json.dumps(self.allowance_news))
        except Exception as err:
            self.wire.log.append({"kind": "told_error", "error": f"{type(err).__name__}: {err}"[:300]})
        return cap

    def hear(self, lns: list[str], qtext: str | None) -> str:
        new_lines = [ln for ln in lns if ln not in self.caught]
        if not new_lines:
            return ""
        self.caught.update(new_lines)
        if self.wire.spare() < CLOSING:
            return ""
        so_far = self.text_body()
        prompt = self._listener_prompt(so_far, qtext, new_lines)
        self.exchanges.append((f"asked: {qtext} -> " if qtext else "") + "shopper: " + " / ".join(new_lines))
        answer = self.wire.consult(prompt, 2500, LISTENER)
        if not answer:
            if self.wire.mute or self.wire.misses > 0:
                self._park(new_lines, qtext)
            return ""
        answer = self._screen_meta(answer, qtext, new_lines)
        if str(answer.get("approval") or "").strip().lower() in ("yes", "no"):
            self.approval = str(answer.get("approval")).strip().lower()
        try:
            renames = self.line_names(answer.get("needs") or [], new_lines)
        except Exception:
            renames = {}
        if renames:
            self._swap_needs(answer, renames)
            self.wire.log.append({"kind": "need_renamed", "names": list(renames.values())[:4]})
        masked = self._masked_lines(answer, new_lines, qtext)
        notes = []
        said = " ".join([so_far, " ".join(self.exchanges), qtext or ""]).lower()

        def grounded(req: str) -> bool:
            words = [w for w in re.findall(r"[a-z]+", req.lower()) if len(w) >= 3]
            return not words or any(w in said for w in words)

        try:
            self.literal_values(answer)
        except Exception as err:
            self.wire.log.append({"kind": "literal_value_error", "error": f"{type(err).__name__}: {err}"[:300]})
        self._take_needs(answer, grounded, notes, new_lines)
        self._take_free(answer, qtext)
        widened = self._take_drops(answer, masked, notes)
        cap = self._take_terms(answer, qtext, notes)
        if notes:
            self.version += 1
            if widened:
                self.widened_at, self.widened_tick = self.version, self.wire.tick
        if self.now_reqs is not None:
            self._restore_lost(masked)
        if widened:
            notes.append(self._widened_remark())
        counts = {key: len(answer.get(key) or []) for key in ("needs", "no_preference", "dropped")}
        self.wire.log.append({"kind": "heard", "question": qtext is not None, "needs": counts["needs"],
                              "free": counts["no_preference"], "dropped": counts["dropped"],
                              "budget": TextKit.is_number(cap)})
        # a question the shopper puts back, in a store result or in the reply to our own message, gets its answer
        asked_back = not answer.get("no_preference") and any("?" in ln for ln in new_lines)
        self.reply_due = str(answer.get("reply") or "").strip() if asked_back else ""
        tail = " (" + "; ".join(notes) + ")" if notes else ""
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

    @staticmethod
    def _close(a: float, b: float) -> bool:
        return abs(a - b) < 1e-9

    def _spec_of(self, key: str) -> dict[str, Any]:
        spec = self.prop_specs.get(key)
        return spec if isinstance(spec, dict) else {}

    def form(self, call_args: dict[str, Any]) -> str:
        chosen = {}
        for key, val in call_args.items():
            if isinstance(self.prop_specs.get(key), dict) and "enum" in self.prop_specs[key]:
                chosen[key] = val
        return TextKit.as_text(chosen)

    def line_spec(self) -> tuple[str, str, str] | None:
        for key, spec in self.prop_specs.items():
            if not (isinstance(spec, dict) and spec.get("type") == "array"):
                continue
            inner = spec.get("items")
            fields = inner.get("properties") if isinstance(inner, dict) else None
            if not isinstance(fields, dict):
                continue
            texts = [k for k, s in fields.items() if isinstance(s, dict) and s.get("type") == "string" and "enum" not in s]
            id_like = [k for k in texts if "id" in k.lower()]
            numbers = [k for k, s in fields.items() if isinstance(s, dict) and s.get("type") in ("integer", "number")]
            ids = id_like or texts
            if ids and numbers:
                return key, ids[0], numbers[0]
        return None

    def _listed(self, call_args: dict[str, Any]) -> tuple[str, str, str] | None:
        spec = self.line_spec()
        return spec if spec and isinstance(call_args.get(spec[0]), list) else None

    def lines(self, call_args: dict[str, Any]) -> list[tuple[str, float | None]]:
        spec = self._listed(call_args)
        if spec is None:
            thing = self.aim_flat(call_args)
            return [(thing, self.count_arg(call_args))] if thing else []
        arg, id_field, count_field = spec
        out = []
        for row in call_args[arg]:
            if not isinstance(row, dict):
                continue
            ident = row.get(id_field)
            if isinstance(ident, str) and ident.strip():
                count = row.get(count_field)
                out.append((ident.strip(), float(row[count_field]) if TextKit.is_number(count) else None))
        return out

    @staticmethod
    def _first_put(found: list[tuple[str, float | None]]) -> tuple[str, float | None] | None:
        if not found:
            return None
        putting = [ln for ln in found if ln[1] is None or ln[1] > 0]
        return (putting or found)[0]

    def aim(self, call_args: dict[str, Any]) -> str | None:
        first = self._first_put(self.lines(call_args))
        return first[0] if first else None

    def aim_flat(self, call_args: dict[str, Any]) -> str | None:
        for key, val in call_args.items():
            if not isinstance(val, str) or "enum" in self._spec_of(key):
                continue
            word = val.strip()
            if word and " " not in word:
                return val
        return None

    def count_arg(self, call_args: dict[str, Any]) -> float | None:
        if self.line_spec() and isinstance(call_args.get(self.line_spec()[0]), list):
            first = self._first_put(self.lines(call_args))
            return first[1] if first else None
        for key, spec in self.prop_specs.items():
            if not isinstance(spec, dict) or "enum" in spec or spec.get("type") not in ("integer", "number"):
                continue
            given = call_args.get(key, spec.get("default"))
            if TextKit.is_number(given):
                return float(given)
        return None

    def is_put(self, call_args: dict[str, Any]) -> bool:
        return self.put_form is None or self.form(call_args) == self.put_form

    def held(self, thing: str) -> float:
        return self.units.get(thing, 0.0)

    def _book_list(self, call_args: dict[str, Any], obs: Any) -> str:
        self.adds = False
        self.units = {}
        for thing, count in self.lines(call_args):
            if count is not None and count <= 0:
                continue
            self.units[thing] = self.units.get(thing, 0.0) + (1.0 if count is None else count)
        seen = self.view_units(obs)
        if seen is not None:
            self.learn(seen)
        return self.summary()

    def booked(self, call_args: dict[str, Any], obs: Any) -> str:
        self.used = True
        if isinstance(obs, dict) and obs.get("error"):
            return ""
        if self.put_form is None:
            self.put_form = self.form(call_args)
        if self.line_spec() and isinstance(call_args.get(self.line_spec()[0]), list):
            return self._book_list(call_args, obs)
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
            self.units[thing] = count if count > 0 else 0.0
        if self.units.get(thing, 0) <= 0:
            self.units.pop(thing, None)
        seen = self.view_units(obs)
        if seen is None:
            if (count <= 0 or not self.is_put(call_args)) and self.shows_empty(obs, thing):
                self.units.pop(thing, None)
                self.pending.pop(thing, None)
        elif set(seen) <= {thing}:
            self.settle(thing, seen.get(thing, 0.0))
        else:
            self.learn(seen)
        return self.summary()

    def shows_empty(self, obs: Any, thing: str) -> bool:
        if not isinstance(obs, dict) or thing not in self.units:
            return False
        values = list(obs.values())
        lists = [v for v in values if isinstance(v, list)]
        if not lists or any(lists):
            return False
        figures = [v for v in values if isinstance(v, (int, float)) and not isinstance(v, bool)]
        if not figures or any(v != 0 for v in figures):
            return False
        shown = TextKit.as_text(obs)
        return all(other not in shown for other in self.units if other != thing)

    def view_units(self, obs: Any) -> dict[str, float] | None:
        found: dict[str, float] = {}
        known = set(self.units) | set(self.pending)

        def count_field(pairs: list[tuple[Any, int]]) -> list[tuple[Any, int]]:
            named = []
            for key, val in pairs:
                low = str(key).lower()
                if re.search(r"qty|quant|count|units|number|total|held|ordered", low) and not re.search(r"price|cost|due|sum|fee|charge", low):
                    named.append((key, val))
            return named if len(named) == 1 else pairs

        def note(d: dict[str, Any]) -> None:
            ids = []
            for val in d.values():
                if isinstance(val, str) and " " not in val and (val in known or re.fullmatch(ID_LIKE, val)):
                    ids.append(val)
            pairs = [(key, val) for key, val in d.items()
                     if isinstance(val, int) and not isinstance(val, bool) and 0 < val < 1000 and str(key) != "price"]
            if len(pairs) > 1:
                pairs = count_field(pairs)
            if len(pairs) > 1 and ids:
                lead = next((i for i in ids if i in known), ids[0])
                expected = self.units.get(lead, self.pending.get(lead))
                equal = [(key, val) for key, val in pairs if expected is not None and abs(val - expected) < 1e-9]
                pairs = equal[:1] if len(equal) == 1 else [min(pairs, key=lambda kv: kv[1])]
            if len(pairs) == 1 and ids:
                ident = next((i for i in ids if i in known), ids[0])
                found[ident] = found.get(ident, 0) + pairs[0][1]
        TextKit.walk_dicts(obs, note)
        if found and set(found) & (known or set(found)):
            return found
        return None

    def _judge_mode(self, thing: str, shown: float, if_set: float) -> None:
        if self._close(shown, if_set) and not self._close(self.units.get(thing, 0), if_set):
            self.adds = False
        elif shown > if_set:
            self.adds = True

    def settle(self, thing: str, units: float) -> None:
        if_set = self.pending.pop(thing, None)
        if if_set is not None:
            self._judge_mode(thing, units, if_set)
        if units <= 0:
            self.units.pop(thing, None)
        else:
            self.units[thing] = units

    def learn(self, seen: dict[str, float]) -> None:
        for thing, if_set in self.pending.items():
            if thing in seen:
                self._judge_mode(thing, seen[thing], if_set)
        self.pending = {}
        self.units = {key: val for key, val in seen.items() if val > 0}

    def viewed(self, obs: Any, tick: int) -> None:
        self.last_view = obs
        self.viewed_at = tick
        seen = self.view_units(obs)
        if seen is not None:
            self.learn(seen)
        elif isinstance(obs, (dict, list)) and not self.ids_in(obs):
            self.units = {}

    def ids_in(self, obs: Any) -> bool:
        shown = TextKit.as_text(obs)
        return any(thing in shown for thing in self.units)

    @staticmethod
    def _units_text(qty: float) -> Any:
        return int(qty) if float(qty).is_integer() else qty

    def summary(self) -> str:
        if not self.units:
            return "[cart as counted] empty."
        listed = ", ".join(f"{thing} x{self._units_text(qty)}" for thing, qty in self.units.items())
        return f"[cart as counted] {listed}."

    def faults(self, need_n: int, counts: list[int] | None = None) -> list[str]:
        if not self.units:
            return []
        if counts and len(counts) > 1:
            if len(self.units) != len(counts):
                held_lines = ", ".join(f"{i} x{int(q)}" for i, q in self.units.items())
                return [f"the cart holds {len(self.units)} line(s) ({held_lines}) but "
                        f"the request asks for {len(counts)} products, one line each; every product needs its line and nothing else"]
            held = sorted(int(q) for q in self.units.values())
            if held == sorted(counts):
                return []
            return [f"the lines hold {held} units but the products need {sorted(counts)} (one line per product at its quantity)"]
        problems = []
        if len(self.units) > 1:
            held_lines = ", ".join(f"{i} x{int(q)}" for i, q in self.units.items())
            problems.append("the cart holds more than one line (" + held_lines + "); take out every line but the chosen one")
        elif len(self.units) == 1:
            thing, qty = next(iter(self.units.items()))
            if abs(qty - need_n) > 1e-9:
                change = "take out" if qty > need_n else "put in"
                problems.append(f"the line {thing} holds {int(qty)} units but the request needs {need_n}; {change} {int(abs(qty - need_n))} with the cart tool")
        return problems


class Intake:
    @classmethod
    def read_request(cls, ctx) -> None:
        listing = []
        for tdef in ctx.shelf.toolset:
            fn = tdef["function"]
            listing.append(f"- {fn['name']}: {fn.get('description', '')} | arguments: {TextKit.as_text(fn.get('parameters') or {})}")
        steady = READ_REQUEST + "\n\nTOOLS:\n" + "\n".join(listing)
        varying = "REQUEST:\n" + ctx.request
        if ctx.pview.get("opening") is not None:
            varying += "\n\nOPENING OBSERVATION:\n" + TextKit.as_text(ctx.pview.get("opening"))
        content = [{"type": "text", "text": steady, "cache_control": {"type": "ephemeral"}}, {"type": "text", "text": varying}]
        ctx.budget.read_content = content
        reading: dict[str, Any] = {}
        ladder = READER + LISTENER[:1]
        wire = ctx.wire
        for turn, model in enumerate(ladder * 3):
            if turn >= len(ladder):
                hopeless = (wire.misses == 0 or wire.call_fail == "refused" or wire.down()
                            or not wire.usable(ladder) or wire.spare() < CLOSING + 60)
                if hopeless:
                    break
            reply = wire.query_model([{"role": "user", "content": content}], [model], 3000)
            reading = TextKit.json_object((reply or {}).get("content") or "") or {}
            if isinstance(reading.get("roles"), dict):
                break
            wire.log.append({"kind": "setup_retry", "model": model})
        if not isinstance(reading.get("roles"), dict):
            reading = cls._code_fallback(ctx, reading)
        cls.use_reading(ctx, reading)

    @classmethod
    def _code_fallback(cls, ctx, reading: dict[str, Any]) -> dict[str, Any]:
        try:
            kept = {key: val for key, val in reading.items() if key != "roles"}
            reading = {**Closing.code_reading(ctx), **kept}
        except Exception as err:
            ctx.wire.log.append({"kind": "code_reading_error", "error": f"{type(err).__name__}: {err}"[:300]})
            reading = {}
        ctx.budget.read_at = monotonic()
        ctx.wire.log.append({"kind": "code_reading", "roles": reading.get("roles"), "product": reading.get("product"),
                             "price_cap": reading.get("price_cap"), "items": len(reading.get("items") or [])})
        return reading

    @classmethod
    def _logged(cls, ctx, kind: str, err: Exception) -> None:
        ctx.wire.log.append({"kind": kind, "error": f"{type(err).__name__}: {err}"[:300]})

    @staticmethod
    def item_requirements(reading: dict[str, Any]) -> list[str]:
        # a reading that wrote its requirements only under the items still states them: each one is lifted, with its product's
        # tag when several products are wanted
        rows = [r for r in reading.get("items") or [] if isinstance(r, dict)]
        several = len([r for r in rows if str(r.get("product") or "").strip()]) > 1
        lifted = []
        for row in rows:
            product = str(row.get("product") or "").strip()
            for req in row.get("requirements") or []:
                head, colon, value = str(req).partition(":")
                if not colon or not head.strip() or not value.strip():
                    continue
                if several and product and "(" not in head:
                    req = f"{head.strip()} ({product}):{value}"
                lifted.append(str(req))
        return list(dict.fromkeys(lifted))

    @classmethod
    def use_reading(cls, ctx, reading: dict[str, Any], again: bool = False) -> None:
        needs, wire = ctx.needs, ctx.wire
        cart_kept = again and ctx.cart.trolley.used
        if not again:
            ctx.budget.steps_given = ctx.budget.step_cap
        old_cart = ctx.role("cart")
        roles = reading.get("roles") if isinstance(reading.get("roles"), dict) else {}
        ctx.shelf.role_map = {key: val for key, val in roles.items() if isinstance(val, str) and val in ctx.shelf.shapes}
        if cart_kept and old_cart:
            ctx.shelf.role_map["cart"] = old_cart
        if not cart_kept:
            ctx.cart.trolley = CartBook(ctx.role("cart"), ctx.shelf.shapes.get(ctx.role("cart") or "", {}))
        needs.sweepable = bool(ctx.role("search")) and Reading.page_arg(ctx) is not None
        allowance = reading.get("message_budget")
        if TextKit.is_number(allowance) and allowance >= 0:
            ctx.budget.msg_cap = int(allowance)
        needs.reading = reading
        try:
            needs.add_told(reading.get("instructions"), "request")
            needs.add_told([r for r in reading.get("rules") or [] if isinstance(r, str)], "request")
        except Exception as err:
            cls._logged(ctx, "told_seed_error", err)
        try:
            wire.scale = 1.0 + 0.3 * min(2, max(0, len(needs.items()) - 1))
        except Exception:
            wire.scale = 1.0
        if not reading.get("requirements"):
            reading["requirements"] = cls.item_requirements(reading)
        named = [str(req) for req in reading.get("requirements") or [] if req]
        proper = []
        for req in named:
            head, colon, value = req.partition(":")
            if colon and head.strip() and value.strip():
                proper.append(req)
        needs.written = proper
        loose = [req for req in named if req not in needs.written]
        if loose:
            reading["soft_wants"] = [str(w) for w in reading.get("soft_wants") or [] if w] + loose
            reading["requirements"] = list(needs.written)
        for step, kind in ((lambda: cls.story_values(ctx, reading), "story_values_error"),
                           (lambda: Referencing.ref_register(ctx), "ref_register_error")):
            try:
                step()
            except Exception as err:
                cls._logged(ctx, kind, err)
        if not (again and needs.version > 0):
            needs.spend_cap = reading.get("price_cap")
        if reading and needs.spend_cap is None and ctx.role("message"):
            reading["ask_first"] = ["the most the shopper wants to spend, since the request states no price limit (skip if the same thing is listed again here)"] + [str(a) for a in reading.get("ask_first") or [] if a]
        if again:
            ctx.budget.step_cap = vars(ctx.budget).get("steps_given", ctx.budget.step_cap)
        limit = reading.get("step_budget")
        if TextKit.is_number(limit) and 0 < limit < ctx.budget.step_cap:
            ctx.budget.step_cap = int(limit)
        wire.scale = vars(wire).get("scale", 1.0) * max(0.5, min(1.0, 40.0 / max(1, ctx.budget.step_cap)))
        if again:
            needs.version += 1

    @staticmethod
    def _hint_list(raw: Any) -> list[str]:
        if isinstance(raw, dict):
            flat = []
            for key, vals in raw.items():
                for v in vals if isinstance(vals, list) else [vals]:
                    if v:
                        flat.append(f"{v}" if ":" in str(v) else f"{key}: {v}")
            raw = flat
        if not isinstance(raw, (list, tuple)):
            return []
        return [str(h).strip() for h in raw or [] if isinstance(h, (str, int, float)) and str(h).strip()]

    @staticmethod
    def _bare_req(req: Any) -> tuple[str, str]:
        head, _, value = str(req).partition(":")
        return re.split(r"\s*\(", head)[0].strip().lower(), value.strip().lower()

    @classmethod
    def story_values(cls, ctx, reading: dict[str, Any]) -> None:
        needs = ctx.needs
        hints = [ValueKit.one_prop_hint(h) for h in cls._hint_list(reading.get("hints"))]
        own_hints = list(hints)
        given = reading.get("references")
        refs = [str(r) for r in given or []] if isinstance(given, list) else []
        guessed = [req for req in needs.written if ValueKit.story_guess(req, ctx.request, refs)]
        if guessed and len(needs.written) >= 2 and 2 * len(guessed) <= len(needs.written):
            needs.written = [req for req in needs.written if req not in guessed]
            reading["requirements"] = list(needs.written)
            gone = {cls._bare_req(req) for req in guessed}
            for row in reading.get("items") or []:
                if isinstance(row, dict) and isinstance(row.get("requirements"), list):
                    row["requirements"] = [r for r in row["requirements"] if cls._bare_req(r) not in gone]
            for req in guessed:
                prop, _, value = req.partition(":")
                hints.append(f"{prop.strip()}: the story suggests {value.strip()} (not stated as a value; ask)")
            ctx.wire.log.append({"kind": "story_values", "moved": guessed})
        try:
            written = cls.written_hints(ctx, reading, own_hints)
            hints = [h for h in hints if h not in written]
        except Exception as err:
            cls._logged(ctx, "written_hint_error", err)
        reading["hints"] = hints[:12]

    @classmethod
    def written_hints(cls, ctx, reading: dict[str, Any], hints: list[str]) -> list[str]:
        needs = ctx.needs
        rows = [r for r in reading.get("items") or [] if isinstance(r, dict)]
        promoted: list[str] = []
        for hint in hints:
            head, sep, detail = str(hint).partition(":")
            prop = re.split(r"\s*\(", head)[0].strip()
            if not (sep and prop) or TextKit.junk_key(prop):
                continue
            tag = re.findall(r"\(([^)]*)\)", head)
            if tag:
                names = [tag[-1]]
            else:
                names = [str(r.get("product") or "") for r in rows] or [str(reading.get("product") or "")]
            word, name = "", ""
            for name in names:
                word = ValueKit.lead_value(detail, ctx.request, name)
                if word:
                    break
            if not word:
                continue
            mine = PropKit.kind_words(name)
            if cls._hint_clashes(needs.written, prop, word, mine):
                continue
            req = f"{head.strip()}: {word}"
            needs.written.append(req)
            reading["requirements"] = list(needs.written)
            target = next((row for row in rows if isinstance(row.get("requirements"), list)
                           and (len(rows) == 1 or PropKit.kind_words(row.get("product") or "") & mine)), None)
            if target is not None:
                target["requirements"].append(req)
            promoted.append(hint)
            ctx.wire.log.append({"kind": "written_hint", "hint": str(hint)[:200], "requirement": req})
        return promoted

    @staticmethod
    def _hint_clashes(written: list[str], prop: str, word: str, mine: set[str]) -> bool:
        for req in written:
            head, _, value = req.partition(":")
            tags = re.findall(r"\(([^)]*)\)", head)
            if tags and mine and not (PropKit.kind_words(tags[-1]) & mine):
                continue
            if PropKit.same_prop(head, prop) or re.search(r"\b" + re.escape(word) + r"\b", value.lower()):
                return True
        return False

    @classmethod
    def open_hints(cls, ctx, k: int | None = None) -> list[str]:
        several = bool(ctx.needs.items())
        plain, reopened = [], []
        for hint in ctx.needs.reading.get("hints") or []:
            prop = str(hint).split(":", 1)[0].strip()
            if not prop:
                continue
            j = Asking.product_of(ctx, prop) if several else None
            if k is not None and j not in (k, None):
                continue
            slot = j if several else None
            if Asking.open_prop(ctx, slot, prop, cls.prop_line(ctx, j, prop), by_samples=True) and not cls.asked_inside(ctx, slot, prop):
                plain.append(str(hint))
                continue
            alt = cls.hint_alt(ctx, slot, str(hint), prop)
            if alt:
                reopened.append(alt)
        return reopened + plain

    @classmethod
    def _menu_for(cls, ctx, k: int | None) -> str:
        records = ctx.shelf.archive
        if not ctx.needs.items():
            own = Lookup.own_shelf(ctx)
            return records.prop_menu(own)[0] if own else records.prop_menu()[0]
        homes = Lookup.home_shelves(ctx)
        home = homes[k] if k is not None and k < len(homes) else set()
        shelves = [s for s in records.shelves_read() if s.strip().lower() in home]
        return records.prop_menu(shelves[0])[0] if shelves else ""

    @staticmethod
    def _line_key(line: str) -> str:
        return line.strip("- ").split(" (in ")[0].strip()

    @classmethod
    def _cached_menu(cls, ctx, k: int | None) -> str:
        return (vars(ctx.talk).get("menu_memo", {}).get(k) or (0, ""))[1]

    @classmethod
    def prop_line(cls, ctx, k: int | None, prop: str) -> str:
        try:
            stamp = len(ctx.shelf.archive.items)
            memo = vars(ctx.talk).setdefault("menu_memo", {}).get(k)
            if memo and memo[0] == stamp:
                menu = memo[1]
            else:
                menu = cls._menu_for(ctx, k)
                ctx.talk.menu_memo[k] = (len(ctx.shelf.archive.items), menu)
            return next((line for line in menu.splitlines() if PropKit.strict_prop(cls._line_key(line), prop)), "")
        except Exception:
            return ""

    @classmethod
    def field_values(cls, ctx, k: int | None, prop: str) -> set[str]:
        line = cls.prop_line(ctx, k, prop)
        if "):" not in line:
            return set()
        found = {" ".join(re.findall(r"[a-z0-9]+", v.lower())) for v in line.split("):", 1)[1].split("|")}
        found.discard("")
        return found

    @classmethod
    def same_field(cls, ctx, k: int | None, a: str, b: str) -> bool:
        if PropKit.strict_prop(a, b):
            return True
        try:
            if not PropKit.same_prop(a, b):
                return False
            return bool(cls.field_values(ctx, k, a) & cls.field_values(ctx, k, b))
        except Exception:
            return False

    @classmethod
    def hinted_prop(cls, ctx, k: int | None, prop: str) -> bool:
        name = str(prop).split("(")[0].strip()
        for hint in ctx.needs.reading.get("hints") or []:
            head = str(hint).split(":", 1)[0]
            if not PropKit.strict_prop(head, name):
                continue
            if k is None or not ctx.needs.items() or Asking.product_of(ctx, head) in (k, None):
                return True
        return False

    @classmethod
    def product_reads(cls, ctx, k: int | None) -> int:
        try:
            records = ctx.shelf.archive
            if not ctx.needs.items():
                own = Lookup.own_shelf(ctx)
                return records.rich_reads(own) if own else records.rich_reads()
            homes = Lookup.home_shelves(ctx)
            home = homes[k] if k is not None and k < len(homes) else set()
            return sum(records.rich_reads(s) for s in home) if home else 0
        except Exception:
            return 0

    @classmethod
    def plain_hint(cls, ctx, k: int | None, prop: str, hinted: bool = False) -> bool:
        try:
            name = re.split(r"\s*\(", str(prop))[0].strip()
            if not name or not (hinted or cls.hinted_prop(ctx, k, name)):
                return False
            if cls.prop_line(ctx, k, name) or cls.product_reads(ctx, k) < 2:
                return False
            keys = [cls._line_key(line) for line in cls._cached_menu(ctx, k).splitlines()]
            return not any(PropKit.same_prop(key, name) for key in keys)
        except Exception:
            return False

    @classmethod
    def hint_mark(cls, ctx, hint: str) -> str:
        try:
            prop = str(hint).split(":", 1)[0].strip()
            k = Asking.product_of(ctx, prop) if ctx.needs.items() else None
            if cls.plain_hint(ctx, k, prop, hinted=True):
                return f"{hint} [no key]"
            return str(hint)
        except Exception:
            return str(hint)

    @classmethod
    def hint_alts_of(cls, ctx, detail: str) -> list[str]:
        alts = []
        for group in re.findall(r"\(or ([^()]+)\)", str(detail)):
            alts.extend(a.strip() for a in re.split(r",|\bor\b|/", group) if a.strip())
        return alts

    @staticmethod
    def _name_set(name: Any) -> set[str]:
        return {w for w in PropKit.prop_words(name) if len(w) >= 3 and w not in LINK_WORDS}

    @classmethod
    def name_inside(cls, ctx, a: Any, b: Any) -> bool:
        left, right = cls._name_set(a), cls._name_set(b)
        if not left or not right:
            return False
        return left <= right or right <= left

    @classmethod
    def _asked_product(cls, ctx, part: str, said: Any, several: bool) -> int | None:
        if not several:
            return None
        j = Asking.product_of(ctx, part)
        if j is None and said:
            j = Asking.product_of(ctx, said.group(1))
        return j

    @classmethod
    def asked_inside(cls, ctx, k: int | None, head: str) -> bool:
        try:
            several = bool(ctx.needs.items())
            for ln in ctx.talk.questions_log:
                label = ln.split(":", 1)[0].strip("- ")
                if "[not answered]" in label:
                    continue
                said = re.search(r'asked "(.*?)" -> ', ln)
                for part in label.split(";"):
                    j = cls._asked_product(ctx, part, said, several)
                    if several and k is not None and j not in (k, None):
                        continue
                    if cls.name_inside(ctx, part, head):
                        return True
        except Exception:
            return False
        return False

    @classmethod
    def hint_alt(cls, ctx, k: int | None, hint: str, head: str) -> str:
        try:
            memo = vars(ctx.pointers).setdefault("hint_alts", {})
            several = bool(ctx.needs.items())
            if not memo.get(hint):
                quiet = vars(ctx.talk).get("quiet_asks") or []
                drew_nothing = any((cls.same_field(ctx, k, q, head) or cls.name_inside(ctx, q, head))
                                   and (not several or Asking.product_of(ctx, q) in (k, None)) for q in quiet)
                if not drew_nothing:
                    return ""
                for alt in cls.hint_alts_of(ctx, hint.partition(":")[2]):
                    if TextKit.junk_key(alt) or PropKit.strict_prop(alt, head):
                        continue
                    line = cls.prop_line(ctx, k, alt)
                    if line:
                        memo[hint] = cls._line_key(line)
                        break
            key = memo.get(hint) or ""
            if not key or key.lower() in vars(ctx.talk).get("spent_asks", set()) or not cls.plan_open(ctx, k, key):
                return ""
            tag = re.search(r"\(([^()]*)\)\s*$", hint.partition(":")[0])
            detail = re.sub(r"\s*\(or [^()]+\)", "", hint.partition(":")[2]).strip()
            label = f"{key} ({tag.group(1).strip()})" if tag else key
            return f"{label}: {detail} (its first property, {head}, drew no preference)"
        except Exception:
            return ""

    @classmethod
    def hint_mapped(cls, ctx, k: int | None, prop: str) -> bool:
        try:
            name = re.split(r"\s*\(", str(prop))[0].strip()
            words = [w for w in PropKit.prop_words(name) if len(w) >= 3 and w not in LINK_WORDS]
            several = bool(ctx.needs.items())
            for hint in ctx.needs.reading.get("hints") or []:
                head, _, detail = str(hint).partition(":")
                if several and k is not None and Asking.product_of(ctx, head) not in (k, None):
                    continue
                candidates = [re.split(r"\s*\(", head)[0].strip()] + cls.hint_alts_of(ctx, detail)
                if any(PropKit.strict_prop(n, name) for n in candidates if n):
                    return True
                low = str(hint).lower()
                if words and all(re.search(r"\b" + re.escape(w), low) for w in words):
                    return True
        except Exception:
            return False
        return False

    @classmethod
    def name_twin(cls, ctx, a: Any, b: Any) -> bool:
        if PropKit.strict_prop(a, b):
            return False

        def split(name: Any) -> tuple[list[str], list[str]]:
            ws = [w for w in PropKit.prop_words(name) if len(w) >= 3 and w not in LINK_WORDS]
            return ws, (ws[:-1] if len(ws) > 1 else ws)

        def alike(x: str, y: str) -> bool:
            if x == y:
                return True
            return min(len(x), len(y)) >= 4 and (x.startswith(y) or y.startswith(x))
        all_a, lead_a = split(a)
        all_b, lead_b = split(b)
        if any(alike(x, y) for x in lead_a for y in all_b):
            return True
        return any(alike(x, y) for x in all_a for y in lead_b)

    @classmethod
    def twin_of(cls, ctx, k: int | None, a: str, b: str) -> bool:
        try:
            if PropKit.strict_prop(a, b):
                return False
            return cls.name_twin(ctx, a, b) or cls.same_field(ctx, k, a, b)
        except Exception:
            return False

    @classmethod
    def twin_key(cls, ctx, k: int | None, prop: str) -> str:
        try:
            several = bool(ctx.needs.items())
            cls.prop_line(ctx, k, prop)
            menu = cls._cached_menu(ctx, k)
            if several and k is not None:
                product = ctx.needs.items()[k]["product"]
            else:
                product = str(ctx.needs.reading.get("product") or "")
            own = PropKit.kind_tokens(product)
            pieces = [ctx.request.lower()]
            for h in ctx.needs.reading.get("hints") or []:
                if not several or Asking.product_of(ctx, str(h).split(":", 1)[0]) in (k, None):
                    pieces.append(str(h).lower())
            story = " ".join(pieces)
            spent = vars(ctx.talk).get("spent_asks", set())
            mine = set(PropKit.prop_words(prop))
            for line in menu.splitlines():
                key = cls._line_key(line)
                if not key or TextKit.junk_key(key) or key.lower() in spent:
                    continue
                if not cls.twin_of(ctx, k, key, prop) or not cls.plan_open(ctx, k, key) or cls.name_inside(ctx, key, prop):
                    continue
                words = [w for w in PropKit.prop_words(key)
                         if len(w) >= 3 and w not in LINK_WORDS and w not in mine and not PropKit.kind_in(w, own)]
                if any(re.search(r"\b" + re.escape(w), story) for w in words) or cls.hint_mapped(ctx, k, key):
                    return key
        except Exception:
            return ""
        return ""

    @staticmethod
    def _norm(v: str) -> str:
        return " ".join(re.findall(r"[a-z0-9]+", v.lower()))

    @classmethod
    def values_form(cls, ctx, k: int | None, prop: str, product: str) -> str:
        try:
            line = cls.prop_line(ctx, k, prop)
            if "):" not in line:
                return ""
            menu = cls._cached_menu(ctx, k)
            others = set()
            for other in menu.splitlines():
                if "):" in other and other.strip() != line.strip():
                    others.update(cls._norm(v) for v in other.split("):", 1)[1].split("|"))
            picks: list[str] = []
            for raw in line.split("):", 1)[1].split("|"):
                v = raw.strip()
                if not v or len(v) > 30 or not re.search(r"[a-z]", v.lower()) or re.search(r"[,;/]", v):
                    continue
                if cls._norm(v) in others or cls._norm(v) in [cls._norm(p) for p in picks]:
                    continue
                picks.append(v)
            picks = picks[:3]
            if len(picks) < 2:
                return ""
            return f"For the {product}, would you like {', '.join(picks[:-1])} or {picks[-1]}?"
        except Exception:
            return ""

    @classmethod
    def _twin_slot(cls, ctx, prop: str) -> tuple[bool, int | None, set, Any]:
        several = bool(ctx.needs.items())
        k = Asking.product_of(ctx, prop) if several else None
        done = vars(ctx.talk).setdefault("twin_asks", set())
        mark = k if several else -1
        return several, k, done, mark

    @classmethod
    def distinct_ask(cls, ctx, prop: str) -> tuple[str, str] | None:
        try:
            several, k, done, mark = cls._twin_slot(ctx, prop)
            if (several and k is None) or mark in done:
                return None
            product = ctx.needs.items()[k]["product"] if several else str(ctx.needs.reading.get("product") or "item")
            target = re.split(r"\s*\(", str(prop))[0].strip()
            text = cls.values_form(ctx, k, target, product)
            twin = cls.twin_key(ctx, k, prop) if not text else ""
            if twin:
                target = twin
                text = cls.values_form(ctx, k, twin, product) or f"Which {twin} would you like for the {product}?"
            if not text:
                return None
            done.add(k if several else -1)
            return (f"{target} ({product})" if several else target), text
        except Exception:
            return None

    @classmethod
    def asked_for(cls, ctx, k: int | None, prop: Any) -> bool:
        try:
            several = bool(ctx.needs.items())
            name = re.split(r"\s*\(", str(prop or ""))[0].strip()
            if not name:
                return False
            for ln in ctx.talk.questions_log:
                label = re.sub(r"\s*\[[^\]]*\]", "", ln.split(":", 1)[0].strip("- "))
                said = re.search(r'asked "(.*?)" -> ', ln)
                parts = [p.strip() for p in re.split(r";|\band\b", label) if p.strip()]
                for part in parts:
                    j = cls._asked_product(ctx, part, said, several)
                    if several and k is not None and j != k:
                        continue
                    if PropKit.strict_prop(re.split(r"\s*\(", part)[0].strip(), name):
                        return True
        except Exception:
            return False
        return False

    @classmethod
    def alt_key(cls, ctx, k: int | None, prop: str) -> str:
        try:
            name = re.split(r"\s*\(", str(prop))[0].strip()
            if not name:
                return ""
            several = bool(ctx.needs.items())
            spent = vars(ctx.talk).get("spent_asks", set())
            for hint in ctx.needs.reading.get("hints") or []:
                head_full, _, detail = str(hint).partition(":")
                head = re.split(r"\s*\(", head_full)[0].strip()
                if not head or (several and Asking.product_of(ctx, head_full) not in (k, None)):
                    continue
                if not (cls.same_field(ctx, k, name, head) or cls.name_inside(ctx, name, head)):
                    continue
                for alt in cls.hint_alts_of(ctx, detail):
                    if TextKit.junk_key(alt) or PropKit.strict_prop(alt, head) or PropKit.strict_prop(alt, name):
                        continue
                    if cls.name_inside(ctx, alt, name):
                        continue
                    line = cls.prop_line(ctx, k, alt)
                    key = cls._line_key(line) if line else ""
                    if not key or PropKit.strict_prop(key, name) or key.lower() in spent:
                        continue
                    if cls.asked_for(ctx, k, key) or not cls.plan_open(ctx, k, key):
                        continue
                    return key
        except Exception:
            return ""
        return ""

    @classmethod
    def collide_alt(cls, ctx, prop: str) -> str:
        try:
            several, k, done, mark = cls._twin_slot(ctx, prop)
            if (several and k is None) or mark in done:
                return ""
            key = cls.alt_key(ctx, k, prop)
            if not key:
                return ""
            done.add(k if several else -1)
            return f"{key} ({ctx.needs.items()[k]['product']})" if several else key
        except Exception:
            return ""

    @classmethod
    def holds_plain(cls, ctx, text: Any, plain_q: str) -> bool:
        frame = {"which", "would", "you", "like", "for", "the"}
        have = {Reading.stem(ctx, w) for w in re.findall(r"[a-z0-9]+", str(text).lower())}
        wanted = [Reading.stem(ctx, w) for w in re.findall(r"[a-z0-9]+", str(plain_q).lower())
                  if w not in frame and w not in LINK_WORDS]
        return bool(wanted) and all(w in have for w in wanted)

    @classmethod
    def alt_untried(cls, ctx, k: int) -> bool:
        try:
            for hint in ctx.needs.reading.get("hints") or []:
                head_full = str(hint).split(":", 1)[0]
                if Asking.product_of(ctx, head_full) not in (k, None):
                    continue
                head = re.split(r"\s*\(", head_full)[0].strip()
                if head and cls.asked_for(ctx, k, head) and cls.alt_key(ctx, k, head):
                    return True
        except Exception:
            return False
        return False

    @classmethod
    def menu_trim(cls, ctx, menu: str, reads: int) -> str:
        if menu and ctx.route()["menu_merge"]:
            menu = cls.menu_merge(menu)
        if reads < 4 or not menu:
            return menu
        kept = [ln for ln in menu.splitlines() if not re.search(r"\(in [01] listings?\)", ln)]
        return "\n".join(kept) if kept else menu

    @staticmethod
    def menu_merge(menu: str) -> str:
        # a key whose name holds all the words of a key above it (or the reverse) and whose sample values all appear there is
        # the same property written again: it leaves the menu
        kept, seen = [], []
        for line in menu.splitlines():
            m = re.match(r"\s*-\s*(.+?)\s+\(in \d+ listings?\)", line)
            if not m:
                kept.append(line)
                continue
            words = set(PropKit.prop_words(m.group(1)))
            samples = {re.sub(r"\s+", "", v).lower() for v in line.rsplit(": ", 1)[-1].split(" | ") if v.strip()}
            if words and samples and any((words <= w or w <= words) and samples <= s for w, s in seen):
                continue
            seen.append((words, samples))
            kept.append(line)
        return "\n".join(kept)

    @classmethod
    def first_reads(cls, ctx) -> bool:
        try:
            if vars(ctx.shelf).get("first_reads_done") or Lookup.batch_tool(ctx) or not ctx.role("details"):
                return False
            ctx.shelf.first_reads_done = True
            rows = ctx.needs.items()
            homes = Lookup.home_shelves(ctx)
            qtys = [row["quantity"] for row in rows] or [ctx.needs.quantity()]
            room = ctx.moves_left() - cls.order_due(ctx) - 2 * max(0, ctx.ask_room() - ctx.budget.msgs_sent) - 8
            wants: list[tuple[int, int, list[str]]] = []
            taken: list[str] = []
            for k in range(len(rows) or 1):
                home = homes[k] if k < len(homes) else set()
                have = cls.product_reads(ctx, k if rows else None)
                if not home or have >= 2 or room <= 0:
                    continue
                picks = Lookup.menu_rows(ctx, home, 2 - have, taken, qtys[min(k, len(qtys) - 1)])
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
            shelf = ctx.shelf
            shelf.open_queue = queue
            shelf.open_recheck = True
            shelf.open_say = "First I look at a few listings per requested product to see which fields they carry, then I ask the shopper."
            shelf.open_purpose = "menu"
            ctx.log.append({"kind": "first_reads", "ids": queue})
            return True
        except Exception as err:
            ctx.log.append({"kind": "first_reads_error", "error": f"{type(err).__name__}: {err}"[:300]})
            return False

    @classmethod
    def plan_open(cls, ctx, k: int | None, prop: str) -> bool:
        try:
            hinted = cls.hinted_prop(ctx, k, prop)
            line = cls.prop_line(ctx, k, prop) if hinted else ""
            return Asking.open_prop(ctx, k, prop, line, by_samples=hinted)
        except Exception:
            return Asking.open_prop(ctx, k, prop)

    @classmethod
    def settles(cls, ctx, req: str, samples: str = "") -> bool:
        head, _, value = str(req).partition(":")
        low = value.lower()
        if re.search(r"as shown|same as|the page|sold by|redirect|refer", low) or RefKit.ref_text(low):
            return True
        if not ValueKit.positive_value(value, head):
            return False
        given = ctx.needs.reading.get("references")
        refs = [str(r) for r in given or []] if isinstance(given, list) else []
        refs += list(vars(ctx.needs).get("resolved") or [])
        if ValueKit.story_guess(req, ctx.request + " " + " ".join(ctx.needs.exchanges), refs):
            return False
        if not samples or re.search(r"\d", value):
            return True
        words = [w for w in re.findall(r"[a-z]{3,}", low) if w not in LINK_WORDS]
        if not words:
            return True
        pool = set(re.findall(r"[a-z]+", samples))

        def carried(w: str) -> bool:
            if re.search(r"\b" + re.escape(w[:-1] if len(w) > 4 else w), samples):
                return True
            return len(w) >= 4 and any(len(s) >= 4 and PropKit.edits_within(s, w, 1) for s in pool)
        return any(carried(w) for w in words)

    @classmethod
    def order_due(cls, ctx) -> int:
        cap = ctx.budget.step_cap
        floor = max(FIRST_ORDER_BY, min(FIRST_ORDER_DUE, cap // 4 + 1))
        due = max(floor, min(16, cap // 3))
        dead = vars(ctx.shelf).get("dead")
        if vars(ctx.talk).get("event_seen") or (dead and set(ctx.cart.trolley.units) & ctx.shelf.dead):
            try:
                due = max(due, 8 + max(1, len(ctx.needs.items())))
            except Exception:
                due = max(due, 9)
        return max(floor, min(due, cap // 2)) + ctx.step_reserve()

    @classmethod
    def narrowing(cls, ctx) -> bool:
        return bool(ctx.role("order") and ctx.role("cart"))

    @classmethod
    def _reserve_names(cls, ctx, spare: int, order: Any, decline: Any, cart: Any) -> set | None:
        if spare <= 1:
            return {order, decline}
        if spare <= 3:
            return {order, decline, cart}
        first_due = spare <= cls.order_due(ctx) and ctx.cart.orders_tried == 0
        if not first_due and not (spare <= FIRST_ORDER_BY and ctx.cart.refusal_n > 0):
            return None
        names = {order, decline, cart, ctx.role("cart_view"), ctx.role("view") or ctx.role("details"), Lookup.stock_tool(ctx)}
        if spare > 3 and Lookup.swap_open(ctx):
            names.add(ctx.role("details"))
        late = vars(ctx.budget).get("late_n", 0) < 3
        if late and ((cls.unverified(ctx) and spare > 5)
                     or (cls.named_open(ctx) and spare >= min(FIRST_ORDER_BY + 2, cls.order_due(ctx)))):
            names.add(ctx.role("details"))
        if late and (cls.widened_late(ctx) or vars(ctx.shelf).get("named_pages") or (cls.unverified(ctx) and spare > 5)) and spare > 4:
            names.add(ctx.role("search"))
        if spare >= 7 and cls.backup_rows(ctx):
            names.add(ctx.role("details"))
        return names

    @classmethod
    def offered(cls, ctx) -> list[dict[str, Any]]:
        spare = ctx.moves_left()
        order, decline, cart = ctx.role("order"), ctx.role("decline"), ctx.role("cart")
        names = cls._reserve_names(ctx, spare, order, decline, cart)
        tools = ctx.shelf.toolset
        if names is None or not (order and cart):
            return tools
        budget = ctx.budget
        if ctx.role("message") and spare >= 2 and (budget.msg_cap is None or budget.msgs_sent < budget.msg_cap):
            names.add(ctx.role("message"))
        trimmed = [tdef for tdef in ctx.shelf.toolset if tdef["function"]["name"] in names]
        return trimmed or tools

    @classmethod
    def in_reserve(cls, ctx) -> bool:
        spare = ctx.moves_left()
        if not (ctx.role("order") and ctx.role("cart")) or spare <= 3:
            return False
        if spare <= cls.order_due(ctx) and ctx.cart.orders_tried == 0:
            return True
        return spare <= FIRST_ORDER_BY and ctx.cart.refusal_n > 0

    @classmethod
    def unverified(cls, ctx) -> bool:
        dead = Lookup.dead_lines(ctx)
        version = str(ctx.needs.version)
        live = []
        for key in ctx.cart.passed:
            item, _, stamp = key.rpartition("|")
            if stamp == version and item not in ctx.cart.refused_lines and item not in dead:
                live.append(item)
        if not live:
            return True
        if not ctx.needs.items():
            return False
        try:
            shelves = {Judging.shelf_known(ctx, i) for i in live} - {""}
            if not shelves:
                return False
            return any(home and not home & shelves for home in Lookup.home_shelves(ctx))
        except Exception:
            return False

    @classmethod
    def named_open(cls, ctx) -> set[str]:
        records = ctx.shelf.archive
        seen = {i for i in vars(ctx.shelf).get("named", set()) if i in records.ids}
        return seen - records.read - ctx.cart.refused_lines

    @classmethod
    def widened_late(cls, ctx) -> bool:
        needs = ctx.needs
        if needs.widened_at != needs.version:
            return False
        return ctx.wire.tick - needs.widened_tick <= cls.order_due(ctx)

    @classmethod
    def begin_batch(cls, ctx) -> None:
        ctx.budget.late_q = 0
        ctx.budget.late_ids = set()

    @classmethod
    def _spare_backups(cls, ctx) -> list[str]:
        used = vars(ctx.budget).get("backup_n", 0)
        late_ids = vars(ctx.budget).get("late_ids", set())
        return [i for i in cls.backup_rows(ctx) if i not in late_ids][: 2 - used]

    @classmethod
    def late_hold(cls, ctx, call_name: str, call_args: dict[str, Any]) -> str:
        details, search = ctx.role("details"), ctx.role("search")
        if call_name not in (details, search) and call_name in Lookup.lookup_tools(ctx) and details:
            return cls.late_hold(ctx, details, call_args)
        if call_name not in (details, search) or not cls.in_reserve(ctx):
            return ""
        budget = ctx.budget
        named = {val for val in call_args.values() if isinstance(val, str)}
        if call_name == details:
            ours = set(ctx.cart.trolley.units) | {k.rpartition("|")[0] for k in ctx.cart.passed}
            if named & ours:
                return ""
        queued = vars(budget).get("late_q", 0)
        spare = ctx.moves_left() - queued
        allowed = False
        if vars(budget).get("late_n", 0) + queued < 3:
            if call_name == details:
                allowed = (cls.unverified(ctx) and spare > 5) or bool(named & cls.named_open(ctx)) and spare >= min(FIRST_ORDER_BY + 2, cls.order_due(ctx))
            else:
                pages = vars(ctx.shelf).get("named_pages", set())
                allowed = (cls.widened_late(ctx) or json.dumps(call_args, sort_keys=True, default=str) in pages
                           or (cls.unverified(ctx) and spare > 5)) and spare > 4
        if not allowed and call_name == details and spare - 1 >= 6 and vars(budget).get("backup_n", 0) < 2:
            if named & set(cls._spare_backups(ctx)):
                allowed = True
                budget.backup_n = vars(budget).get("backup_n", 0) + 1
        opened = Lookup.swap_open(ctx) if not allowed and call_name == details else ""
        if opened and opened in named and spare > 2:
            allowed = True
            ctx.shelf.swap_done = True
        if allowed:
            budget.late_q = queued + 1
            vars(budget).setdefault("late_ids", set()).update(named)
            return ""
        extra = []
        if call_name == details and spare - 1 >= 6 and vars(budget).get("backup_n", 0) < 2:
            extra = cls._spare_backups(ctx)
        note = ("Not run (no step used): the steps left are kept for the order. Now details are read only for the cart's line, a "
                "listing a check named, or (while nothing is verified) up to 3 listings in all: put your best verified fit in the "
                "cart, re-check its stock and price, and order.")
        if extra:
            note += " Cheaper unread rows that may back up the pick can still be read: " + ", ".join(extra) + "."
        return note

    @classmethod
    def backup_rows(cls, ctx) -> list[str]:
        if ctx.needs.priority() != "cheapest" or vars(ctx.budget).get("backup_n", 0) >= 2:
            return []
        if not (vars(ctx.talk).get("event_seen") or Lookup.dead_lines(ctx) or cls.named_open(ctx)):
            return []
        found: list[tuple[float, str]] = []
        try:
            for k in Coverage.cg_products(ctx):
                pool = Lookup.ed_pool(ctx, k)
                tops = [e["price"] for e in pool if e["verified"] and e["fit"] and e["price"] is not None]
                if not tops:
                    continue
                ceiling = min(tops)
                for e in pool:
                    if e["read"] or e["gone"] or e["over"] or e["pointer"] or e["ruled"] or e["clash"]:
                        continue
                    if e["price"] is not None and e["price"] < ceiling:
                        found.append((e["price"], e["id"]))
        except Exception as err:
            ctx.log.append({"kind": "backup_error", "error": f"{type(err).__name__}: {err}"[:300]})
            return []
        return [i for _, i in sorted(found)][:6]


class Judging:
    REF_HINT = r"as shown|same as|the page|sold by|redirect|refer"

    @classmethod
    def field_names(cls, ctx) -> str:
        menu, _ = ctx.shelf.archive.prop_menu()
        if not menu:
            return ""
        return ("\n\nPROPERTY FIELDS THIS STORE USES (names differing only in separators count as one; each with the number of read listings that have it):\n"
                + menu)

    @classmethod
    def _ruled_out(cls, ctx, memo: str, thing: str, why: str, problems: list[str], by: str = "") -> str:
        ruling = f"An independent check says {thing} does NOT meet every requirement: {why}. Choose another listing."
        ctx.cart.rulings[memo] = ruling
        result = {"item": thing, "fits": False, "problems": problems}
        if by:
            result["by"] = by
        ctx.log.append({"kind": "verify", "item": thing, "result": result})
        return ruling

    @classmethod
    def _pointed_records(cls, ctx, thing: str, refs: list[str]) -> str:
        records = ctx.shelf.archive
        words = {w for r in refs for w in re.findall(r"[a-z]{4,}", r.lower().split(":", 1)[-1])}
        pointed = []
        for i in sorted(records.read):
            if i != thing and any(w in " ".join(records.about(i, 1)).lower() for w in words):
                pointed.append(i)
        pointed = pointed[:3]
        if not pointed:
            return ""
        return ("\n\nREAD LISTINGS THE REQUEST REFERS TO (when a requirement is given by reference, its value is whatever the "
                "named field says on the listing it points to):\n" + "\n".join(r for i in pointed for r in records.about(i, 1)))

    @staticmethod
    def _own_note(own: list) -> str:
        parts = []
        for req, prop, sent in own:
            parts.append(f"one sentence in its notes may move {prop} to a different listing: \"{sent[:240]}\". When it really does that (the reader is told to "
                         f"take {prop} from another listing, refer to it, go by it or look to it), grading reads {prop} on that other listing; the item's "
                         f"own {prop} field then says nothing about '{req}', which holds only if the other listing's record shows it. Notes that "
                         "mention a listing merely as a version sold elsewhere, or as something this item is not, change nothing.")
        return "\n\nNOTE (from the code's reading of the item's own notes): " + " ".join(parts)

    @classmethod
    def _verify_prompt(cls, ctx, thing, aim_text, second, own, refs, tracked, seen, evidence):
        needs, records, trolley = ctx.needs, ctx.shelf.archive, ctx.cart.trolley
        parts = [CHECK_ITEM, "\n\nTHE SHOPPER'S REQUEST:\n", ctx.request, needs.changes(),
                 "\n\nCURRENT REQUIREMENTS (stated, revealed and changed):\n", needs.text_body()]
        if needs.items():
            held = trolley.held(thing)
            parts.append("\n\nThe item below belongs to one of the requested products (its category tells which one): measure it only "
                         "against that product's requirements and quantity, never against another product's.")
            if held > 0:
                parts.append(f" It is in the cart with {int(trolley.held(thing))} units.")
        parts.append(Referencing.ref_note(ctx, thing) or "")
        refs = Referencing.ref_untracked(ctx, refs, tracked)
        if refs:
            parts.append(cls._pointed_records(ctx, thing, refs))
        parts.append(second)
        if own:
            parts.append(cls._own_note(own))
        parts.append(f"\n\nTHE ITEM TO {aim_text}: {thing}\nRECORDS OF THIS ITEM SEEN IN THE STORE (oldest first):\n" + "\n".join(seen))
        if trolley.last_view is not None:
            parts.append("\n\nLATEST CART VIEW:\n" + TextKit.as_text(trolley.last_view)[:3000])
        parts.append(cls.field_names(ctx))
        if evidence["lines"]:
            parts.append("\n\nFIELD EVIDENCE (worked out by code from this store's records in the item's category: how the store "
                         "spells each required value, and what this item shows for it):\n" + "\n".join(evidence["lines"]))
        labels = records.label_lines()
        if labels:
            parts.append("\n\nCATEGORY LABELS SEEN REPEATEDLY IN THIS STORE (with counts; the item's label has to match the request's wording taken as a whole, not a nearby label):\n" + labels)
        return "".join(parts)

    @classmethod
    def check_item(cls, ctx, thing: str, aim_text: str, second: str = "") -> str:
        if ctx.wire.spare() < CLOSING + 20:
            return ""
        needs, records, trolley = ctx.needs, ctx.shelf.archive, ctx.cart.trolley
        key = f"{thing}|{needs.version}"
        if key in vars(ctx.cart).get("near_ok", set()):
            return ""
        given = needs.reading.get("references")
        refs = [str(r) for r in given or [] if r] if isinstance(given, list) else []
        tracked = list(vars(ctx.pointers).get("refs") or [])
        own = Referencing.own_redirects(ctx, thing)
        memo = key
        if refs or any(r.get("status") != "resolved" for r in tracked):
            memo += f"|{len(records.read)}"
        if second:
            memo += "|second"
        if own:
            memo += f"|own{len(own)}"
        rulings = ctx.cart.rulings
        if memo in rulings:
            return rulings[memo]
        pointer = Referencing.ref_pointer_note(ctx, thing)
        if pointer:
            return cls._ruled_out(ctx, memo, thing, pointer, ["reference chain"])
        seen = records.about(thing)
        if not seen:
            return ""
        evidence = cls.field_view(ctx, thing)
        if evidence["fails"]:
            return cls._ruled_out(ctx, memo, thing, "; ".join(evidence["fails"][:3]), evidence["fails"][:3], "field evidence")
        prompt = cls._verify_prompt(ctx, thing, aim_text, second, own, refs, tracked, seen, evidence)
        answer = ctx.wire.consult(prompt, 3000, ctx.route()["checker"])
        ctx.log.append({"kind": "verify", "item": thing, "result": answer})
        if answer and answer.get("fits") is False:
            faults = [str(p) for p in answer.get("problems") or []][:5]
            ruling = f"An independent check says {thing} does NOT meet every requirement: " + "; ".join(faults) + ". Unless you are certain the check is wrong, choose another fully verified fit."
            rulings[memo] = ruling
            return ruling
        rulings[memo] = ""
        if answer:
            ctx.cart.passed.add(key)
            vars(ctx.cart).setdefault("pass_price", {})[thing] = records.price(thing)
        return ""

    @classmethod
    def ch_rows(cls, ctx) -> dict[str, dict[str, Any]]:
        stamp = len(ctx.shelf.archive.items)
        memo = vars(ctx.shelf).get("ch_rows_memo")
        if memo and memo[0] == stamp:
            return memo[1]
        rows = Closing.code_rows(ctx)
        ctx.shelf.ch_rows_memo = (len(ctx.shelf.archive.items), rows)
        return rows

    @classmethod
    def store_fields(cls, ctx, label: Any = "") -> dict[str, list[str]]:
        try:
            rows = cls.ch_rows(ctx)
            homes = Lookup.home_shelves(ctx)
            if ctx.needs.items():
                k = Asking.product_of(ctx, str(label))
                shelves = set(homes[k]) if k is not None and k < len(homes) else set()
            else:
                own = Lookup.own_shelf(ctx)
                if own:
                    shelves = {own}
                else:
                    shelves = set(homes[0]) if homes else set()
            fields: dict[str, list[str]] = {}
            for row in rows.values():
                if not row["fields"]:
                    continue
                if shelves and row["shelf"] not in shelves:
                    continue
                for key, val in row["fields"].items():
                    if str(val).strip() and not TextKit.junk_key(key):
                        fields.setdefault(str(key), []).append(str(val))
            return fields
        except Exception:
            return {}

    @classmethod
    def ch_redirected(cls, ctx, thing: str, prop: str) -> bool:
        try:
            for got in Referencing.own_redirects(ctx, thing) or []:
                heads = got if isinstance(got, (list, tuple)) else [got]
                if any(PropKit.same_prop(str(x).partition(":")[0], prop) for x in heads[:2]):
                    return True
            return False
        except Exception:
            return True

    @classmethod
    def field_view(cls, ctx, thing: str) -> dict[str, list[str]]:
        needs = ctx.needs
        key = (thing, len(ctx.shelf.archive.items), needs.version, "|".join(needs.props_left()))
        memo = vars(ctx.shelf).setdefault("ch_view_memo", {})
        if key not in memo:
            try:
                memo[key] = cls.field_facts(ctx, thing)
            except Exception as err:
                ctx.log.append({"kind": "field_facts_error", "error": f"{type(err).__name__}: {err}"[:300]})
                memo[key] = {"lines": [], "rules": [], "fails": []}
        said = (vars(ctx.cart).get("near_said") or {}).get(f"{thing}|{needs.version}")
        if not said:
            return memo[key]
        try:
            view = memo[key]
            return {"lines": [cls._confirmed_line(ln, said) for ln in view["lines"]],
                    "rules": list(view["rules"]),
                    "fails": [f for f in view["fails"] if not cls._names_any(f.split(":", 1)[0].strip(), [p for p, _, _ in said])]}
        except Exception:
            return memo[key]

    @staticmethod
    def _names_any(name: str, props: list[str]) -> bool:
        return any(PropKit.same_prop(name, p) or PropKit.strict_prop(name, p) for p in props)

    @classmethod
    def _confirmed_line(cls, line: str, said: list) -> str:
        head = re.match(r"- (.+?) = '", line)
        if not head:
            return line
        for prop, value, shown in said:
            if cls._names_any(head.group(1), [prop]):
                return line + (f" A meaning check confirmed its '{shown[:80]}' as '{value[:60]}' (the same value in other words): it "
                               "meets it.")
        return line

    @staticmethod
    def _clip(text: Any, n: int = 70) -> str:
        text = " ".join(str(text).split())
        return text if len(text) <= n else text[: n - 3] + "..."

    @staticmethod
    def _field_table(peers: dict[str, dict[str, Any]]) -> tuple[dict[str, int], dict[str, list]]:
        carry: dict[str, int] = {}
        table: dict[str, list[tuple[str, str, list[str]]]] = {}
        for ident, row in peers.items():
            for key, val in row["fields"].items():
                if not str(val).strip() or TextKit.junk_key(key):
                    continue
                carry[key] = carry.get(key, 0) + 1
                pieces = ValueKit.ch_parts(val)
                if len(pieces) > 1:
                    pieces = pieces + [str(val).strip().lower()]
                table.setdefault(key, []).extend((ident, p, ValueKit.ch_words(p)) for p in pieces)
        return carry, table

    @classmethod
    def _limit_check(cls, ctx, out: dict, me: dict, peers: dict, carry: dict, prop: str, value: str, off_product: bool) -> None:
        own_words = set(PropKit.prop_words(prop))
        named = [key for key in carry if carry[key] >= 2 and PropKit.strict_prop(key, prop)]
        kin = {w for w in own_words if len(w) >= 3 and w not in LINK_WORDS}
        if not (prop and kin and named) or off_product:
            return
        for key, v in me["fields"].items():
            if str(v).strip() and (PropKit.strict_prop(key, prop) or set(PropKit.prop_words(key)) & kin):
                return
        clip = cls._clip
        keys = " / ".join(f"'{x}'" for x in named[:3])
        carrying = len([r for r in peers.values() if any(str(r["fields"].get(x, "")).strip() for x in named)])
        out["lines"].append(f"- {prop} = '{clip(value)}': this store's field for it is {keys}, on "
                            f"{carrying} of "
                            f"{len(peers)} read listings of this category. This listing has no such field, so the limit is not "
                            "shown met: it does NOT meet it (another field's figure does not stand in).")
        out["fails"].append(f"{prop}: it has no {named[0]} field, which this store's listings of the category carry, so "
                            f"'{clip(value)}' is not shown met")

    @classmethod
    def unproven_wording(cls, ctx, thing: str) -> str:
        # a field worded otherwise than the required value is taken only once enough listings of its kind show the store never writes it
        if ctx.wire.spare() < CLOSING + 60 or ctx.moves_left() <= 8:
            return ""
        holds = vars(ctx.cart).setdefault("wording_holds", set())
        rows = cls.ch_rows(ctx)
        me = rows.get(thing)
        if len(holds) >= ctx.route()["wording_holds"] or not me or not me["fields"]:
            return ""
        several = bool(ctx.needs.items())
        shelf = me["shelf"]
        k = None
        if several:
            k = next((n for n, home in enumerate(Lookup.home_shelves(ctx)) if shelf and shelf in home), None)
            if k is None:
                return ""
        kin = {ident: row for ident, row in rows.items() if not shelf or row["shelf"] == shelf}
        for req in ctx.needs.props_left()[:12]:
            head, _, raw = str(req).partition(":")
            value = " ".join(re.sub(r"\((?:stated|revealed[^)]*)\)", " ", raw).split())
            prop = re.split(r"\s*\(", head)[0].strip()
            if not prop or not value or re.search(r"\d", value) or RefKit.ref_text(value) or re.search(cls.REF_HINT, value.lower()):
                continue
            if re.search(EXCLUDE_WORDS, value.lower()) or not ValueKit.positive_value(value, head):
                continue
            if several and Asking.product_of(ctx, head) != k:
                continue
            mine = [(key, str(v)) for key, v in me["fields"].items() if str(v).strip() and PropKit.strict_prop(key, prop)]
            if not mine or any(ValueKit.ch_writes(value, v) for _, v in mine):
                continue
            own = PropKit.prop_words(prop)
            if any(ValueKit.ch_writes(value, v) for key, v in me["fields"].items()
                   if len(PropKit.prop_words(key)) == len(own) + 1 and PropKit.prop_words(key)[1:] == own):
                continue
            carriers = [ident for ident, row in kin.items()
                        if any(str(v).strip() and PropKit.strict_prop(key, prop) for key, v in row["fields"].items())]
            mark = f"{thing}|{prop}"
            if len(carriers) >= CH_SYNONYM_MIN or mark in holds:
                continue
            holds.add(mark)
            writers = [ident for ident in carriers if ident != thing and any(
                PropKit.strict_prop(key, prop) and ValueKit.ch_writes(value, v) for key, v in kin[ident]["fields"].items())]
            unread = [ident for ident, row in kin.items() if not row["fields"] and row["stock"] is not False][:6]
            clip = cls._clip
            text = (f"{thing} is held back for now: its {mine[0][0]} reads '{clip(mine[0][1], 60)}', not '{clip(value)}' "
                    f"({head.strip()}). A requirement holds on the store's own words for the value, and only {len(carriers)} "
                    "read listing(s) of this category carry that field, too few to tell whether this store writes "
                    f"'{clip(value)}' itself. ")
            if writers:
                text += f"{', '.join(writers[:3])} already write(s) '{clip(value)}' there: prefer such a listing."
            else:
                text += ("Read the details of more listings of this category first (a search with the value's words, then "
                         "several listings' details in one call" + (f", e.g. {', '.join(unread)}" if unread else "")
                         + f"); if none of them writes '{clip(value)}', this listing may stand and the same call goes through.")
            ctx.log.append({"kind": "wording_hold", "item": thing, "req": str(req)[:120], "carriers": len(carriers),
                            "writers": writers[:3]})
            return text
        return ""

    @classmethod
    def field_facts(cls, ctx, thing: str) -> dict[str, list[str]]:
        out: dict[str, list[str]] = {"lines": [], "rules": [], "fails": []}
        rows = cls.ch_rows(ctx)
        me = rows.get(thing)
        if not me or not me["fields"]:
            return out
        several = bool(ctx.needs.items())
        shelf = me["shelf"]
        if several and not shelf:
            return out
        k = None
        if several:
            k = next((n for n, home in enumerate(Lookup.home_shelves(ctx)) if shelf in home), None)
        peers = {}
        for ident, row in rows.items():
            if row["fields"] and (not shelf or row["shelf"] == shelf):
                peers[ident] = row
        if len(peers) < 3:
            return out
        carry, table = cls._field_table(peers)
        for req in ctx.needs.props_left()[:12]:
            head, _, raw = str(req).partition(":")
            value = " ".join(re.sub(r"\((?:stated|revealed[^)]*)\)", " ", raw).split())
            prop = re.split(r"\s*\(", head)[0].strip()
            if not prop or not value or RefKit.ref_text(value) or re.search(cls.REF_HINT, value.lower()):
                continue
            if re.search(r"\d", value) and re.search(CH_BOUND, value.lower()):
                j = Asking.product_of(ctx, head) if several else None
                cls._limit_check(ctx, out, me, peers, carry, prop, value, several and j is not None and j != k)
                continue
            if re.fullmatch(r"\s*\d+(?:\.\d+)?\s+[a-z]+\s*", value.lower()):
                continue
            j = Asking.product_of(ctx, head) if several else None
            if several and j is not None and j != k:
                continue
            cls._value_facts(ctx, out, thing, me, rows, peers, carry, table, shelf, head, prop, value, several and j is None)
        return out

    @classmethod
    def _value_facts(cls, ctx, out, thing, me, rows, peers, carry, table, shelf, head, prop, value, unplaced) -> None:
        clip = cls._clip
        positive = ValueKit.positive_value(value, head)
        if not positive and not re.search(EXCLUDE_WORDS, value.lower()):
            return
        alts = ValueKit.ch_alts(value)
        if not alts:
            return
        named = sorted((key for key in carry if PropKit.strict_prop(key, prop)), key=lambda x: -carry[x])
        by_name = bool(named)
        if by_name and not positive:
            flat = " ".join(re.findall(r"[a-z0-9]+", value.lower()))
            if any(" ".join(words) == flat for key in named for _, _, words in table.get(key, [])):
                positive = True
                LITERAL_VALUES.add(flat)
        if not named:
            if not positive or unplaced:
                return
            found = [key for key in carry if carry[key] >= 2
                     and any(ValueKit.ch_match(alts, words) for _, _, words in table.get(key, []))]
            if not found:
                return
            top = max(found, key=lambda x: carry[x])
            named = sorted((key for key in carry if PropKit.strict_prop(key, top)), key=lambda x: -carry[x])
        entries = [(i, words) for key in named for i, _, words in table.get(key, []) if i != thing]
        on = len([r for r in peers.values() if any(str(r["fields"].get(key, "")).strip() for key in named)])
        writers = sorted({i for i, words in entries if ValueKit.ch_match(alts, words)})
        alone = {" ".join(words) for _, words in entries}
        samples = list(dict.fromkeys(clip(r["fields"][key], 40) for r in peers.values() for key in named
                                     if str(r["fields"].get(key, "")).strip()))[:6]
        keys_txt = " / ".join(f"'{x}'" for x in named[:3])
        tail = f"{len(writers)} other listing(s) write '{clip(value)}' itself in it."
        if not positive:
            bare = " ".join(re.sub(EXCLUDE_WORDS, " ", value.lower()).split())
            bare_alts = ValueKit.ch_alts(bare)
            if bare_alts:
                held = {i for i, words in entries if ValueKit.ch_match(bare_alts, words)}
                tail = (f"{len(held)} other listing(s) write '{clip(bare)}' there, the value it excludes (a listing that writes "
                        "it does not meet it).")
        opening = (f"- {prop} = '{clip(value)}': this store's field for it is {keys_txt}"
                   + ("" if by_name else f" (no field is named '{prop}'; this one's values write the value)")
                   + f", on {on} of {len(peers)} read listings of this category; " + tail)
        own_kinds = PropKit.kind_words(value)
        naming = []
        if not writers and own_kinds:
            naming = [i for i, r in rows.items() if i != thing and (not shelf or r["shelf"] == shelf) and own_kinds <= r["words"]][:3]
        out["rules"].append(cls._rule_text(opening, positive, writers, naming, on, samples, by_name, value))
        cls._listing_facts(ctx, out, thing, me, prop, value, named, by_name, positive, alts, alone, writers, naming, on,
                           samples, opening)

    @classmethod
    def _rule_text(cls, opening, positive, writers, naming, on, samples, by_name, value) -> str:
        clip = cls._clip
        if not positive:
            return opening
        if writers:
            return (opening + " So only a value that writes those words meets it (containment and one list element "
                    "count; a synonym or related word does not)." + ("" if by_name else " With no field named for "
                    "the property, a listing whose title writes the value meets it too, unless a field of its own "
                    "that shares the property's name says otherwise."))
        seen = " Its values seen: " + " | ".join(samples)
        if naming:
            return (opening + seen + f"; listings of this category name '{clip(value)}' "
                    f"in their title or text ({', '.join(naming)}), so the store has its own word for it: a synonym or "
                    "related value meets it only once such a listing's record shows this field writes it otherwise.")
        if on < CH_SYNONYM_MIN:
            return (opening + seen + f"; none writes it, but too few read listings "
                    f"carry this field to tell whether this store writes '{clip(value)}' itself: a value of the same "
                    "meaning is not shown to meet it yet.")
        return opening + seen + "; none writes it, so a value of the same meaning may meet it."

    @classmethod
    def _listing_facts(cls, ctx, out, thing, me, prop, value, named, by_name, positive, alts, alone, writers, naming, on,
                       samples, opening) -> None:
        clip = cls._clip
        fields = me["fields"]
        mine = [(key, str(fields[key])) for key in named if str(fields.get(key, "")).strip()]
        shown = "; ".join(f"{key}: '{clip(v, 90)}'" for key, v in mine)
        own_words = PropKit.prop_words(prop)
        qualified = []
        for key, v in fields.items():
            if not own_words or key in named or not str(v).strip() or TextKit.junk_key(key):
                continue
            words = PropKit.prop_words(key)
            if len(words) == len(own_words) + 1 and words[1:] == own_words:
                qualified.append((key, str(v)))
        qual_writes = [(key, v) for key, v in qualified if ValueKit.ch_writes(value, v)]
        qual_near = bool(qual_writes) or any(ValueKit.ch_touches(value, v) for _, v in qualified)
        qual_note = ""
        if qual_writes:
            qual_note = (f" Its {qual_writes[0][0]} '{clip(qual_writes[0][1], 60)}' writes it (a qualified key of the same property: "
                         "a field named for the property with one more leading word is the same property's field under its fuller "
                         "name), so it meets it.")
        lines = out["lines"]
        if cls.ch_redirected(ctx, thing, prop):
            lines.append(opening + " This listing's own notes may send this property to another listing, so its own "
                         "field may not decide.")
            return
        if not positive:
            lines.append(opening + " This listing: " + (shown or "no such field") + " (an exclusion: the listing must not "
                         "carry the excluded value).")
            return
        kin_words = {w for w in PropKit.prop_words(prop) if len(w) >= 3 and w not in LINK_WORDS}
        kin = [key for key, v in fields.items() if str(v).strip() and not TextKit.junk_key(key)
               and set(PropKit.prop_words(key)) & kin_words and not ValueKit.ch_writes(value, v)]
        title_says = not by_name and not kin and ValueKit.ch_writes(value, me.get("title") or "")
        if not mine:
            cls._missing_field(lines, opening, fields, named, prop, value, qual_writes, qual_note, title_says)
            return
        parts = [p for _, v in mine for p in ValueKit.ch_parts(v)]
        parts += [v.strip().lower() for _, v in mine if len(ValueKit.ch_parts(v)) > 1]

        def knocked(part: str) -> list[str]:
            sides = ValueKit.ch_sides(part)
            if not sides or not any(ValueKit.ch_match(alts, ValueKit.ch_words(s)) for s in sides):
                return []
            return [s for s in sides if not ValueKit.ch_match(alts, ValueKit.ch_words(s)) and " ".join(ValueKit.ch_words(s)) in alone]
        matching = [p for p in parts if ValueKit.ch_match(alts, ValueKit.ch_words(p))]
        clean = [p for p in matching if not knocked(p)]
        joined = [(p, knocked(p)) for p in matching if knocked(p)]
        if clean:
            lines.append(opening + f" This listing: {shown} - it writes '{clip(value)}': it meets it.")
        elif qual_writes:
            lines.append(opening + f" This listing: {shown}." + qual_note)
        elif title_says:
            lines.append(opening + f" This listing: {shown} - with no field named for the property, its title writes "
                         f"'{clip(value)}': it meets it.")
        elif joined:
            part, others = joined[0]
            verdict = "it does NOT meet it."
            lines.append(opening + f" This listing: {shown} - '{clip(part)}' joins '{clip(value)}' with '{others[0]}', which "
                         "other listings carry alone in this field: " + verdict)
            if by_name and not qual_near:
                out["fails"].append(f"{prop}: its {named[0]} is '{clip(part)}', which joins {clip(value)} with {others[0]} (a "
                                    f"value this store also writes alone there), so it is not {clip(value)}")
        elif writers:
            lines.append(opening + f" This listing: {shown} - it does not write '{clip(value)}', which other listings of "
                         "this category write in this field: " + "it does NOT meet it (a synonym or related word does not "
                         "count where the store writes the value itself).")
            if by_name and not qual_near and not any(ValueKit.ch_touches(value, v) for _, v in mine):
                out["fails"].append(f"{prop}: its {named[0]} is '{clip(mine[0][1], 60)}', while this store writes "
                                    f"'{clip(value)}' itself in that field on other listings")
        elif naming:
            lines.append(opening + f" This listing: {shown} - it does not write '{clip(value)}'. No read listing writes it in "
                         "this field yet (its values: " + " | ".join(samples) + f"), but listings of this category name "
                         f"'{clip(value)}' in their title or text ({', '.join(naming)}): the store has its own word for it, "
                         "so this listing's other value is not shown to meet it.")
        elif on < CH_SYNONYM_MIN:
            lines.append(opening + f" This listing: {shown}. No read listing of this category writes '{clip(value)}' in "
                         "this field (its values: " + " | ".join(samples) + f"), but only {on} read listing(s) carry it: too "
                         f"few to tell whether this store writes '{clip(value)}' itself, so a value of the same meaning is "
                         "not shown to meet it yet.")
        else:
            lines.append(opening + f" This listing: {shown}. No read listing of this category writes '{clip(value)}' in "
                         "this field (its values: " + " | ".join(samples) + "), so a value of the same meaning may meet it.")

    @classmethod
    def _missing_field(cls, lines, opening, fields, named, prop, value, qual_writes, qual_note, title_says) -> None:
        clip = cls._clip
        if qual_writes:
            lines.append(opening + " This listing has no such field." + qual_note)
            return
        if title_says:
            lines.append(opening + f" This listing has no such field, and no field is named for the property: its title "
                         f"writes '{clip(value)}': it meets it.")
            return
        prop_set = {w for w in PropKit.prop_words(prop) if len(w) >= 3}
        related = [(key, v) for key, v in fields.items() if key not in named and str(v).strip() and not TextKit.junk_key(key)
                   and (set(PropKit.prop_words(key)) & prop_set or ValueKit.ch_writes(value, v))]
        related = sorted(related, key=lambda kv: not ValueKit.ch_writes(value, kv[1]))[:4]
        listed = ""
        if related:
            listed = ": its fields that may name the property: " + "; ".join(f"{key}: '{clip(v, 60)}'" for key, v in related)
        lines.append(opening + " This listing has no such field" + listed + ". Another name of the SAME property counts; a field "
                     "of another property (a height for a type, a frame material for the material) does not stand in.")

    @classmethod
    def _request_head(cls, ctx, frame: str, label: str) -> str:
        text = frame + "\n\nTHE SHOPPER'S REQUEST:\n" + ctx.request + ctx.needs.changes()
        return text + label + ctx.needs.text_body()

    @classmethod
    def rank_check(cls, ctx, thing: str) -> str:
        needs = ctx.needs
        priority = needs.priority()
        key = f"{thing}|{needs.version}|{ctx.cart.refusal_n}"
        if priority == "none" or key in ctx.cart.rank_checked or ctx.wire.spare() < CLOSING + 30 or ctx.moves_left() < 4:
            return ""
        ctx.cart.rank_checked.add(key)
        detail = needs.reading.get("priority_detail") or ""
        if priority == "cheapest":
            what = "cheapest (lowest price)"
        else:
            what = f"best ({detail})" if detail else "best"
        prompt = cls._request_head(ctx, RANK_CHECK.replace("PRIORITY", what), "\n\nCURRENT REQUIREMENTS:\n")
        prompt += Referencing.ref_note(ctx)
        if needs.items():
            prompt += ("\n\nThe request covers several products: a listing competes with the chosen item only when it is the SAME "
                       "product (category); listings of the other products can never beat it.")
        cheapest = priority == "cheapest"
        cards = ctx.shelf.archive.cards(20000, thing if cheapest else "", Referencing.ref_foreign(ctx), {thing})
        prompt += (f"\n\nTHE CHOSEN ITEM: {thing}\n\nLISTINGS SEEN (a card per listing; DETAILS = its whole record was read, ROW = known "
                   "only from search rows, fields unknown: a ROW in the requested category (whatever object its title names) that beats the chosen item and that its "
                   "title, price or stock does not exclude MAY fit and gets named" + ("; only listings cheaper than the chosen one appear"
                   if cheapest else "") + "):\n" + "\n".join(cards))
        prompt += cls.field_names(ctx)
        rules = cls.field_view(ctx, thing)["rules"]
        if rules:
            prompt += ("\n\nFIELD EVIDENCE (worked out by code from this store's records in the chosen item's category: how the "
                       "store spells each required value):\n" + "\n".join(rules))
        answer = ctx.wire.consult(prompt, 2500, CHECKER)
        ctx.log.append({"kind": "rank", "result": answer})
        dead = Lookup.dead_lines(ctx)
        better = []
        for x in (answer or {}).get("better") or []:
            if not isinstance(x, (str, int)):
                continue
            name = str(x)
            if name != thing and name not in ctx.cart.refused_lines and name not in dead:
                better.append(name)
        better = [x for x in better[:4] if not cls.field_view(ctx, x)["fails"]]
        ctx.cart.rank_better = list(better)
        vars(ctx.shelf).setdefault("named", set()).update(better)
        if not better:
            return ""
        reason = f" ({answer.get('reason')})" if answer.get("reason") else ""
        return (f"A rank check thinks these listings may also fit and beat {thing} on the {what}: " + ", ".join(better)
                + reason + ". Check them (details, stock, price) before ordering; if each fails, order the current pick.")

    @classmethod
    def _unread_hits(cls, ctx) -> list[str]:
        shelf = vars(ctx.shelf)
        records = ctx.shelf.archive
        labels, cap, qty, hit_shelf = Reading.shelf_labels(ctx), ctx.needs.spend_cap, ctx.needs.quantity(), shelf.get("hit_shelf", {})
        scores = shelf.get("hit_score", {})
        out = []
        for i in sorted(shelf.get("all_hits", []), key=lambda i: -scores.get(i, 0)):
            if i in records.read:
                continue
            if labels and hit_shelf.get(i) and hit_shelf.get(i) not in labels:
                continue
            if TextKit.is_number(cap) and TextKit.is_number(records.price(i)) and records.price(i) * qty > cap:
                continue
            out.append(i)
        return out

    @classmethod
    def _search_list(cls, ctx, pages: list) -> str:
        return "; ".join(f"{ctx.role('search')} {json.dumps(p, ensure_ascii=False)}" for p in pages)

    @classmethod
    def _audit_prompt(cls, ctx, cards, fresh):
        parts = [cls._request_head(ctx, BEFORE_DECLINE, "\n\nCURRENT REQUIREMENTS:\n"), Referencing.ref_note(ctx),
                 "\n\nLISTINGS SEEN (a card per listing; DETAILS = its whole record was read, ROW = known only from search rows):\n"
                 + "\n".join(cards)]
        if fresh:
            parts.append("\n\nOPENED AFTER THE PREVIOUS CHECK (these DETAILS cards are new: judge each of them first, field by field, "
                         "by the reading rules): " + ", ".join(fresh))
        parts.append(cls.field_names(ctx))
        return "".join(parts)

    @classmethod
    def _audit_names(cls, ctx, got):
        records = ctx.shelf.archive
        named = [str(x) for x in got.get("possible_fits") or [] if isinstance(x, (str, int))]
        conforms = [x for x in named[:5] if not cls.field_view(ctx, x)["fails"]]
        unopened = []
        for x in got.get("unopened") or []:
            if isinstance(x, (str, int)) and str(x) not in records.read and str(x) in records.ids:
                unopened.append(str(x))
        return conforms, unopened[:6]

    @classmethod
    def review_decline(cls, ctx, early: bool = False) -> str:
        cart, records = ctx.cart, ctx.shelf.archive
        again = len(records.read) > vars(cart).get("audit_reads", 10 ** 9) and vars(cart).get("audits", 0) < 2
        if (cart.audit_done and not again) or ctx.wire.spare() < CLOSING + 30:
            return ""
        cart.audit_done = True
        if not early:
            cart.audits = vars(cart).get("audits", 0) + 1
        before = vars(cart).get("audit_ids")
        first = set(records.read) - set(before or ()) if before is not None else ()
        cards = records.cards(16000, "", Referencing.ref_foreign(ctx), first)
        cart.audit_reads = len(records.read)
        if not cards:
            return ""
        before = vars(cart).get("audit_ids")
        fresh = sorted(records.read - before) if before is not None else []
        cart.audit_ids = set(records.read)
        answer = ctx.wire.consult(cls._audit_prompt(ctx, cards, fresh), 6000, AUDITOR)
        ctx.log.append({"kind": "audit", "result": answer})
        conforms, unopened = cls._audit_names(ctx, answer or {})
        cart.audit_fits = conforms
        hits = cls._unread_hits(ctx)
        try:
            slack = [i for i in Lookup.slack_rows(ctx) if i not in records.read]
        except Exception:
            slack = []
        unopened = list(dict.fromkeys(slack + hits[:4] + [i for i in unopened if i not in hits]))[:6]
        pages = Reading.open_pages(ctx, 2) if ctx.moves_left() > 5 else []
        if pages:
            unopened = unopened[:4]
            vars(ctx.shelf).setdefault("named_pages", set()).update(json.dumps(p, sort_keys=True, default=str) for p in pages)
        try:
            combos = Walking.combined_queries(ctx) if ctx.moves_left() > 5 else []
        except Exception as err:
            ctx.log.append({"kind": "combo_error", "error": f"{type(err).__name__}: {err}"[:300]})
            combos = []
        if combos:
            vars(ctx.shelf).setdefault("named_pages", set()).update(json.dumps(p, sort_keys=True, default=str) for p in combos)
        vars(ctx.shelf).setdefault("named", set()).update(conforms + unopened)
        listed = vars(ctx.shelf).setdefault("named_list", [])
        listed += [i for i in conforms + unopened if i not in listed]
        if not (conforms or unopened or pages or combos):
            return ""
        said = []
        if conforms:
            said.append("A second, independent reading suggests these listings could satisfy all requirements: " + ", ".join(conforms)
                        + (f" ({answer.get('reason')})" if answer.get("reason") else "") + ".")
        if pages:
            said.append("Some searches that include required values found more listings than you have read; fits may hide on "
                        "the pages not yet seen: " + cls._search_list(ctx, pages)
                        + ". Go through those pages first (a single step returns a full page of rows), then open each in-budget row of the wanted kind.")
        if combos:
            said.append("Searches joining the required values have not been tried yet (where the store needs every word, such a search "
                        "returns only listings holding all of them): " + cls._search_list(ctx, combos)
                        + ". Try them (search stays allowed for these) and then open each in-budget row of the wanted kind they return.")
        if unopened:
            said.append("Listings of the wanted kind that appeared only as search rows, never opened, so nobody has checked "
                        "their fields: " + ", ".join(unopened) + ". Look at their details before anything else (batch several per reply).")
        return " ".join(said) + " Confirm each one (details, stock, price) and buy the one that fits; decline only when every one of them fails."

    @classmethod
    def named_unread(cls, ctx) -> list[str]:
        shelf = vars(ctx.shelf)
        records = ctx.shelf.archive
        cap, qty = ctx.needs.spend_cap, ctx.needs.quantity()
        labels, hit_shelf = Reading.shelf_labels(ctx), shelf.get("hit_shelf", {})
        scores = shelf.get("hit_score", {})
        ranked = sorted(shelf.get("all_hits", []), key=lambda i: -scores.get(i, 0))
        out = []
        for ident in dict.fromkeys(ranked + list(shelf.get("named_list", []))):
            cost = records.price(ident)
            if ident in records.read or ident in ctx.cart.refused_lines or ident not in records.ids:
                continue
            if TextKit.is_number(cap) and TextKit.is_number(cost) and cost * qty > cap:
                continue
            if labels and hit_shelf.get(ident) and hit_shelf[ident] not in labels:
                continue
            out.append(ident)
        return out

    @classmethod
    def details_call(cls, ctx, ident: str) -> dict[str, Any] | None:
        tool = ctx.role("details")
        ids = ctx.shelf.archive.ids
        for entry in reversed(ctx.log):
            if entry.get("kind") != "env":
                continue
            for act in entry.get("actions") or []:
                args = act.get("args") if act.get("name") == tool else None
                if isinstance(args, dict) and any(isinstance(v, str) and v in ids for v in args.values()):
                    return {k: (ident if isinstance(v, str) and v in ids else v) for k, v in args.items()}
        schema = ctx.shelf.shapes.get(tool or "") or {}
        props = schema.get("properties") if isinstance(schema.get("properties"), dict) else {}
        texts = [k for k, p in props.items() if isinstance(p, dict) and p.get("type") == "string" and not p.get("enum")]
        key = next((k for k in texts if k in (schema.get("required") or [])), None)
        if key is None and len(texts) == 1:
            key = texts[0]
        return {key: ident} if key else None

    @classmethod
    def open_reads(cls, ctx, queue: list[str], say: str, batch: tuple[str, str, int] | None, reader: str | None = None) -> list[tuple[str, str]]:
        if batch:
            tool, arg, most = batch
            calls = [{arg: queue[i: i + most]} for i in range(0, len(queue), most)]
        else:
            tool = reader or ctx.role("details")
            calls = []
            if tool:
                for ident in queue:
                    args = Lookup.lookup_call(ctx, ident, tool)
                    if args:
                        calls.append(args)
        tick = ctx.wire.tick
        proposals = [{"id": f"open_{n}_{tick}", "type": "function", "function": {"name": tool, "arguments": json.dumps(args)}}
                     for n, args in enumerate(calls)]
        if not proposals:
            return []
        ctx.talk.chat.append({"role": "assistant", "content": say, "tool_calls": proposals})
        start = len(ctx.shelf.archive.items)
        ctx.shelf.covering = True
        try:
            responses = Checkout.dispatch(ctx, proposals, scripted=True)
        finally:
            ctx.shelf.covering = False
        ctx.shelf.last_reads = ctx.shelf.archive.items[start:]
        return responses

    @classmethod
    def _take_queue(cls, ctx) -> tuple[list | None, bool]:
        shelf, cart, wire = vars(ctx.shelf), vars(ctx.cart), ctx.wire
        queue = shelf.pop("open_queue", None)
        if queue or not cart.get("put_queue") or wire.done or wire.broke:
            return queue, False
        queue = cart.pop("put_queue")
        ctx.shelf.open_recheck, ctx.shelf.open_purpose, ctx.shelf.open_tool = True, "cart", cart.pop("put_tool", None)
        ctx.shelf.open_say = "Checking stock on the item just added to the cart, since a listing may sell out right after it goes in."
        return queue, True

    @classmethod
    def _reserve_trim(cls, ctx, queue: list[str], purpose: str) -> list[str]:
        try:
            keep = Intake.order_due(ctx) if ctx.cart.orders_tried == 0 else FIRST_ORDER_BY
            room = ctx.moves_left() - keep - 1
            listed = Lookup.batch_tool(ctx) and len(queue) >= 2
            need = 1 if listed else len(queue)
            if room < need and (ctx.cart.trolley.units or not Intake.unverified(ctx)):
                trimmed = [] if (Lookup.batch_tool(ctx) and len(queue) >= 2) else queue[: max(0, room)]
                ctx.log.append({"kind": "reserve_skip", "purpose": purpose, "ids": queue[:6], "kept": trimmed})
                return trimmed
        except Exception as err:
            ctx.log.append({"kind": "reserve_skip_error", "error": f"{type(err).__name__}: {err}"[:300]})
        return queue

    @classmethod
    def open_named(cls, ctx) -> bool:
        wire = ctx.wire
        queue, put_read = cls._take_queue(ctx)
        if not queue or wire.done or wire.broke:
            return False
        queue = list(dict.fromkeys(str(i) for i in queue))
        shelf = vars(ctx.shelf)
        recheck = shelf.pop("open_recheck", False)
        purpose = str(shelf.pop("open_purpose", "") or "")
        floor = shelf.pop("open_floor", None)
        reader = shelf.pop("open_tool", None)
        say = shelf.pop("open_say", "")
        if not say:
            say = "Before ordering, I re-read the cart lines." if recheck else "Before declining, I open the listings the checks named."
        if purpose in ("menu", "quota", "walk") and ctx.role("order") and ctx.role("cart") and not wire.mute:
            queue = cls._reserve_trim(ctx, queue, purpose)
            if not queue:
                return False
        if purpose == "cart":
            stale = set(Lookup.stale_lines(ctx))
            units = ctx.cart.trolley.units
            queue = [i for i in queue if i in stale or (i not in units and not put_read)]
            if not queue:
                return False
        batch = Lookup.batch_tool(ctx) if len(queue) >= 2 else None
        proven = vars(ctx.cart).get("batch_proven")
        if batch and purpose == "cart" and not proven:
            batch = None
        if batch and reader and not ((vars(ctx.shelf).get("tool_caps") or {}).get(batch[0]) or {}).get("stock"):
            batch = None
        responses = cls.open_reads(ctx, queue, say, batch, reader)
        if batch and responses and not vars(ctx.cart).get("batch_proven"):
            if Lookup.batch_ok(ctx, vars(ctx.shelf).get("last_reads") or [], queue):
                ctx.cart.batch_proven = True
            else:
                ctx.cart.batch_spec = None
                ctx.log.append({"kind": "batch_dropped", "tool": batch[0]})
                most = {"menu": 0, "quota": 2, "walk": 2}.get(purpose, 3)
                if TextKit.is_number(floor):
                    keep = int(floor)
                else:
                    keep = Intake.order_due(ctx) if purpose in ("menu", "quota") else 3
                n = min(most, len(queue), ctx.moves_left() - keep - 1)
                if n > 0 and not wire.done and (reader or ctx.role("details")):
                    Asking.post_replies(ctx, responses)
                    responses = cls.open_reads(ctx, queue[:n], say, None, reader)
                    if not responses:
                        return True
        if not responses:
            return False
        note = cls.review_decline(ctx) if wire.spare() > CLOSING + 30 and not recheck else ""
        if note:
            cid, text_body = responses[-1]
            responses[-1] = (cid, text_body + "\n[check after opening] " + note)
        Asking.post_replies(ctx, responses)
        return True

    @classmethod
    def last_stock(cls, ctx, ident: str) -> bool | None:
        states: list[bool] = []

        def note(d: dict[str, Any]) -> None:
            if Reading.row_mark(ctx, d) != ident:
                return
            got = Reading.stock_state(ctx, d)
            if got is not None:
                states.append(got)
        for rec in ctx.shelf.archive.items:
            TextKit.walk_dicts(rec, note)
        return states[-1] if states else None

    @classmethod
    def _rank_candidate(cls, ctx, x: str, version: str, cap: Any) -> bool:
        cart = ctx.cart
        if x not in ctx.shelf.archive.read or f"{x}|{version}" in cart.passed or x in cart.refused_lines:
            return False
        if x in Lookup.dead_lines(ctx) or cls.last_stock(ctx, x) is False or cls.field_view(ctx, x)["fails"]:
            return False
        several = ctx.needs.items()
        k = Walking.product_index(ctx, x) if several else None
        if k is not None and k < len(ctx.needs.items()):
            qty = ctx.needs.items()[k]["quantity"]
        else:
            qty = ctx.needs.quantity()
        cost = ctx.shelf.archive.price(x)
        if not TextKit.is_number(cost):
            return False
        return not (TextKit.is_number(cap) and TextKit.is_number(qty) and cost * qty > cap)

    @classmethod
    def rank_verify(cls, ctx, thing: str) -> str:
        try:
            if ctx.needs.priority() != "cheapest":
                return ""
            version = str(ctx.needs.version)
            cap = ctx.needs.spend_cap
            checked = 0
            for x in vars(ctx.cart).get("rank_better") or []:
                if checked >= 2 or ctx.wire.spare() < CLOSING + 30:
                    break
                if cls._rank_candidate(ctx, x, version, cap):
                    checked += 1
                    cls.check_item(ctx, x, "ORDER")
            return cls.cheaper_fit(ctx, thing)
        except Exception as err:
            ctx.log.append({"kind": "rank_verify_error", "error": f"{type(err).__name__}: {err}"[:300]})
            return ""

    @classmethod
    def cheaper_fit(cls, ctx, thing: str) -> str:
        if ctx.needs.priority() != "cheapest" or not thing:
            return ""
        records = ctx.shelf.archive
        mine = records.price(thing)
        if mine is None:
            return ""
        kind = cls.shelf_known(ctx, thing)
        best_id, best_cost = "", None
        dead = Lookup.dead_lines(ctx)
        version = str(ctx.needs.version)
        for key in ctx.cart.passed:
            ident, _, stamp = key.rpartition("|")
            if stamp != version or ident == thing or ident in ctx.cart.refused_lines or ident in dead:
                continue
            if ctx.needs.items() and (not kind or cls.shelf_known(ctx, ident) != kind):
                continue
            cost = records.price(ident)
            if cost is None or cost >= mine or (best_cost is not None and cost >= best_cost):
                continue
            if cls.field_view(ctx, ident)["fails"]:
                continue
            best_id, best_cost = ident, cost
        return best_id

    @classmethod
    def shelf_known(cls, ctx, thing: str) -> str:
        labels: list[str] = []

        def note(d: dict[str, Any]) -> None:
            if any(isinstance(val, str) and val == thing and "id" in str(key).lower() for key, val in d.items()):
                labels.append(Reading.shelf_of(ctx, d))
        for rec in ctx.shelf.archive.items:
            TextKit.walk_dicts(rec, note)
        found = ""
        for label in labels:
            found = label or found
        if found:
            return found
        return str((vars(ctx.shelf.archive).get("label_of") or {}).get(thing) or "").strip().lower()

    @classmethod
    def chosen(cls, ctx) -> str | None:
        units = ctx.cart.trolley.units
        return list(units)[-1] if units else None


class Reading:
    SHELF_KEY = r"categor|shelf|department|aisle"
    PAGE_KEY = r"page|offset|first|start|skip|from|cursor"
    NOT_PAGE = r"size|per_?page|limit|price|cost"

    @classmethod
    def refused(cls, ctx, call_name: str, obs: Any, resp: dict[str, Any]) -> bool:
        if isinstance(obs, dict) and "refusal" in obs:
            return True
        if call_name != ctx.role("order") or resp.get("finished"):
            return False
        return not (isinstance(obs, dict) and obs.get("error") in ("invalid_args", "unknown_tool"))

    @staticmethod
    def _sentences(obs: Any) -> list[str]:
        found: list[str] = []

        def dig(piece: Any, depth: int = 0) -> None:
            if depth > 6:
                return
            if isinstance(piece, dict):
                for val in piece.values():
                    dig(val, depth + 1)
            elif isinstance(piece, list):
                for val in piece[:20]:
                    dig(val, depth + 1)
            elif isinstance(piece, str) and len(piece.split()) >= 4:
                found.append(piece)
        dig(obs)
        return found

    @classmethod
    def _order_refused(cls, ctx, obs: Any) -> str:
        cart = ctx.cart
        cart.refusal_n += 1
        cart.order_vetted = False
        ctx.talk.event_seen = True
        extra = ""
        try:
            extra = Lookup.refusal_reread(ctx, obs)
        except Exception as err:
            ctx.log.append({"kind": "refusal_reread_error", "error": f"{type(err).__name__}: {err}"[:300]})
        cart.refused_lines.update(cart.trolley.units)
        ctx.talk.refusal_texts = {" ".join(str(s).lower().split()) for s in cls._sentences(obs)}
        return ("THE ORDER WAS REFUSED and the episode continues; the cart is unchanged. Apply every change in the refusal, fix the cart (take out any line that is no longer acceptable, such as a sold-out item), re-check stock and price of the new pick, and order again." + extra)

    @classmethod
    def notes_for(cls, ctx, call_name: str, obs: Any, refused: bool) -> str:
        notes = []
        budget = ctx.budget
        if isinstance(obs, dict) and obs.get("error") == "budget_exceeded":
            if call_name == ctx.role("message") or budget.msg_cap is not None:
                budget.msg_cap = budget.msgs_sent if budget.msg_cap is None else min(budget.msg_cap, budget.msgs_sent)
            notes.append("That budget is used up; do not call this tool again.")
        if refused and call_name == ctx.role("decline") and getattr(ctx.talk, "last_kind", None) == "decline":
            ctx.cart.audit_done = False
            ctx.cart.decline_vetted = False
            ctx.talk.event_seen = True
            notes.append("THE DECLINE WAS REFUSED and the episode continues: the store does not accept that nothing fits. Read the refusal, apply what it says, look again at the closest candidates (their own properties, stock and price against the requirements now), and order the one that fits; decline again only if each one truly fails.")
        elif refused:
            notes.append(cls._order_refused(ctx, obs))
        elif isinstance(obs, dict) and len(obs) >= 2 and call_name != ctx.role("message"):
            lines: list[str] = Coverage.notice_lines(ctx, obs, tool=call_name)
            if lines:
                carried = vars(ctx.talk).pop("refusal_texts", None) or set()
                repeats = bool(carried) and all(any(" ".join(str(s).lower().split()) in t for t in carried) for s in lines)
                if repeats:
                    notes.append("This result repeats the message the refused finish already carried (handled above); it says nothing "
                                 "about this result.")
                else:
                    ctx.talk.event_seen = True
                    notes.append("This result carries a message from the shopper (a change). Apply it: " + " | ".join(lines[:4]))
        return " ".join(notes)

    @classmethod
    def shelf_of(cls, ctx, row: dict[str, Any]) -> str:
        for key, val in row.items():
            if not isinstance(val, str):
                continue
            low = str(key).lower()
            if 0 < len(val.strip()) <= 60 and re.search(Reading.SHELF_KEY, low) and "id" not in low:
                return val.strip().lower()
        return ""

    @classmethod
    def key_stock(cls, ctx, key: Any, val: Any) -> bool | None:
        name = str(key).lower()
        if val is None or not re.search(r"stock|avail|on.?hand", name):
            return None
        figure = TextKit.is_number(val)
        gone = False
        if re.search(r"stock|avail", name):
            if re.search(r"out|sold", name):
                gone = val is True
            elif val is False or (figure and val <= 0):
                gone = True
            elif isinstance(val, str):
                gone = bool(re.search(r"\b(?:out|sold|none|unavailable|not|no|false)\b", val.lower()))
        if re.search(r"on.?hand", name) and (val is False or (figure and val <= 0)):
            gone = True
        if isinstance(val, str) and re.search(r"back.?order|pre.?order|discontinu", val.lower()):
            gone = True
        # a count of units on hand below what every ordered product needs leaves the listing unusable
        if figure and val is not True and not gone and ctx is not None and val < cls.fewest_units(ctx):
            gone = True
        return not gone

    @classmethod
    def fewest_units(cls, ctx) -> int:
        try:
            return max(1, min(int(q) for q in ctx.needs.counts()))
        except Exception:
            return 1

    @classmethod
    def stock_state(cls, ctx, row: dict[str, Any]) -> bool | None:
        states = []
        for key, val in row.items():
            said = cls.key_stock(ctx, key, val)
            if said is not None:
                states.append(said)
        if not states:
            return None
        return all(states)

    @classmethod
    def sold_out(cls, ctx, row: dict[str, Any]) -> bool:
        return any(cls.key_stock(ctx, key, val) is False for key, val in row.items())

    @classmethod
    def shelf_labels(cls, ctx) -> set[str]:
        records = ctx.shelf.archive
        tally: dict[str, int] = {}
        counted: set[str] = set()
        skip = Referencing.ref_foreign(ctx)

        def note(d: dict[str, Any]) -> None:
            ident = next((val for key, val in d.items() if isinstance(val, str) and "id" in str(key).lower() and val in records.read), None)
            if not ident or ident in counted or ident in skip:
                return
            label = cls.shelf_of(ctx, d)
            if label:
                counted.add(ident)
                tally[label] = tally.get(label, 0) + 1
        for rec in records.items:
            TextKit.walk_dicts(rec, note)
        top = max(tally.values(), default=0)
        best = {label for label, n in tally.items() if n == top}
        if not tally or not ctx.needs.items():
            return best
        try:
            homes = Lookup.home_shelves(ctx)
            picked: set[str] = set()
            for home in homes:
                read = [s for s in home if s in tally]
                if not read:
                    picked |= set(home)
                    continue
                most = max(tally[s] for s in read)
                picked |= {s for s in read if tally[s] == most}
            for label in sorted(tally, key=lambda s: -tally[s]):
                if all(homes) or len(picked) >= len(homes):
                    break
                picked.add(label)
            return picked or best
        except Exception:
            return best

    @classmethod
    def row_mark(cls, ctx, row: dict[str, Any]) -> str:
        names = [(str(key).lower(), val) for key, val in row.items()]
        for test in (lambda n: n == "id", lambda n: n.endswith("id")):
            for name, val in names:
                if test(name) and isinstance(val, (str, int)) and not isinstance(val, bool):
                    return str(val)
        return json.dumps(row, sort_keys=True, default=str)

    @classmethod
    def _search_props(cls, ctx) -> dict[str, Any]:
        props = ctx.shelf.shapes.get(ctx.role("search") or "")
        props = props.get("properties") if isinstance(props, dict) else None
        return props if isinstance(props, dict) else {}

    @classmethod
    def search_keys(cls, ctx, call_args: dict[str, Any]) -> tuple[list[tuple], tuple, bool, set]:
        props = cls._search_props(ctx)

        def spec(key: str) -> dict[str, Any]:
            found = props.get(key)
            return found if isinstance(found, dict) else {}
        texts, mode, paging = [], [], set()
        first = True
        token = cls.token_key(ctx)
        for key, val in call_args.items():
            name = str(key).lower()
            if key == token and isinstance(val, str):
                paging.add(key)
                first = first and not val.strip()
                continue
            if isinstance(val, str):
                if spec(key).get("enum"):
                    mode.append((key, val))
                    continue
                cleaned = (t.strip("\"'()[]{},.;:!?|/") for t in val.lower().split())
                words = tuple(dict.fromkeys(w for w in cleaned if re.search(r"[a-z0-9]", w)))
                if words and all(re.fullmatch(r"[a-z0-9]{2,}", w) for w in words):
                    texts.append((key, words))
                continue
            if TextKit.is_number(val) and re.search(cls.PAGE_KEY, name) and not re.search(cls.NOT_PAGE, name):
                paging.add(key)
                low = spec(key).get("minimum", spec(key).get("default"))
                if not TextKit.is_number(low):
                    low = 1 if "page" in name else 0
                first = first and val <= low
        return texts, tuple(sorted(mode)), first, paging

    @staticmethod
    def _total(obs: dict[str, Any], rows: list) -> int:
        counts = [val for val in obs.values() if isinstance(val, int) and not isinstance(val, bool)]
        return max(counts + [len(rows)])

    @staticmethod
    def _others(call_args: dict[str, Any], key: str, paging: set) -> tuple:
        return tuple(sorted((k2, str(v2)) for k2, v2 in call_args.items() if k2 != key and k2 not in paging))

    @classmethod
    def word_rule(cls, ctx, call_args: dict[str, Any], obs: dict[str, Any], rows: list) -> str:
        shelf = ctx.shelf
        texts, mode, first, paging = cls.search_keys(ctx, call_args)
        if not first:
            return shelf.word_kind
        total = cls._total(obs, rows)
        for key, words in texts:
            slot = (key, mode)
            others = cls._others(call_args, key, paging)
            subset = False
            for seen_slot, seen_others, seen_words, seen_total in shelf.word_seen:
                if seen_slot != slot or seen_others != others or seen_total == total:
                    continue
                if len(words) > len(seen_words) and words[: len(seen_words)] == seen_words:
                    shelf.word_kind = "any" if total > seen_total else "all"
                elif len(seen_words) > len(words) and seen_words[: len(words)] == words:
                    shelf.word_kind = "any" if total < seen_total else "all"
                else:
                    cut = shelf.lead_words or None
                    mine, theirs = set(words[:cut]), set(seen_words[:cut])
                    if (mine > theirs and total < seen_total) or (theirs > mine and seen_total < total):
                        subset = True
            if subset and not shelf.word_kind:
                shelf.word_kind = "all"
            shelf.word_seen.append((slot, others, words, total))
        return shelf.word_kind

    @classmethod
    def lead_rule(cls, ctx, call_args: dict[str, Any], obs: dict[str, Any], rows: list) -> int:
        shelf = ctx.shelf
        texts, mode, first, paging = cls.search_keys(ctx, call_args)
        if not first:
            return shelf.lead_words
        total = cls._total(obs, rows)
        marks = tuple(cls.row_mark(ctx, r) for r in rows)
        for key, words in texts:
            slot = (key, mode)
            others = cls._others(call_args, key, paging)
            bound = shelf.lead_bounds.setdefault(slot, [-1, set()])
            for seen_slot, seen_others, seen_words, seen_total, seen_marks in shelf.lead_seen:
                if seen_slot != slot or seen_others != others or seen_words == words:
                    continue
                shared = 0
                limit = min(len(words), len(seen_words))
                while shared < limit and words[shared] == seen_words[shared]:
                    shared += 1
                if total != seen_total or marks != seen_marks:
                    bound[0] = max(bound[0], shared)
                elif shared and len(set(marks)) >= 3 and shared == limit:
                    bound[1].add(shared)
            shelf.lead_seen.append((slot, others, words, total, marks))
        claims = []
        for slot, (low, high) in shelf.lead_bounds.items():
            above = [k for k in high if k > low]
            if above:
                claims.append((min(above), slot))
        claims = [(k, slot) for k, slot in sorted(claims) if k == shelf.lead_bounds[slot][0] + 1]
        shelf.lead_words, shelf.lead_ctx = claims[0] if claims else (0, None)
        return shelf.lead_words

    @classmethod
    def token_key(cls, ctx) -> str | None:
        props = (ctx.shelf.shapes.get(ctx.role("search") or "") or {}).get("properties")
        if not isinstance(props, dict):
            return None
        for key, spec in props.items():
            if not isinstance(spec, dict) or spec.get("type") != "string" or spec.get("enum"):
                continue
            if re.search(r"bookmark|cursor|token|resume|continu|after|next|page|offset|start", str(key).lower()):
                return key
        return None

    @classmethod
    def next_token(cls, ctx, obs: Any) -> str | None:
        if not isinstance(obs, dict):
            return None
        for key, val in obs.items():
            if isinstance(val, str) and val.strip() and re.search(r"next|onward|cursor|bookmark|token|continu|more", str(key).lower()):
                return val
        return None

    @classmethod
    def page_arg(cls, ctx) -> tuple[str, Any, Any, float | None] | None:
        props = (ctx.shelf.shapes.get(ctx.role("search") or "") or {}).get("properties")
        for key, spec in (props if isinstance(props, dict) else {}).items():
            name = str(key).lower()
            if not isinstance(spec, dict) or spec.get("type") not in ("integer", "number"):
                continue
            if not re.search(cls.PAGE_KEY, name) or re.search(cls.NOT_PAGE, name):
                continue
            by_page = "page" in name and not re.search(r"offset|skip|first|start|from", name)
            low = spec.get("minimum", spec.get("default"))
            if not TextKit.is_number(low):
                low = 1 if by_page else 0
            high = spec.get("maximum") if TextKit.is_number(spec.get("maximum")) else None
            return key, by_page, low, high
        token = cls.token_key(ctx)
        return (token, "token", "", None) if token else None

    @classmethod
    def stem(cls, ctx, word: str) -> str:
        if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
            return word[:-1]
        return word

    @classmethod
    def _row_title(cls, ctx, d: dict[str, Any]) -> str:
        title = d.get("title")
        if isinstance(title, str) and 0 < len(title) <= 300:
            return title
        for key, val in d.items():
            if not isinstance(val, str) or not 0 < len(val) <= 300:
                continue
            low = str(key).lower()
            if re.search(r"title|heading|headline|name", low) and "id" not in low and not cls.shelf_of(ctx, {key: val}) and not TextKit.junk_key(key):
                return val
        return ""

    @classmethod
    def shelf_rows(cls, ctx) -> dict[str, tuple[float, str, str]]:
        labels, cap, qty = cls.shelf_labels(ctx), ctx.needs.spend_cap, ctx.needs.quantity()
        found: dict[str, tuple[float, str, str]] = {}
        if not labels:
            return found
        records = ctx.shelf.archive

        def note(d: dict[str, Any]) -> None:
            ident = next((val for key, val in d.items() if isinstance(val, str) and "id" in str(key).lower() and val in records.ids), None)
            if not ident or ident in ctx.cart.refused_lines or cls.shelf_of(ctx, d) not in labels:
                return
            if cls.sold_out(ctx, d):
                found.pop(ident, None)
                return
            cost = next((val for key, val in d.items()
                         if TextKit.is_number(val) and not isinstance(val, bool) and re.search(r"price|cost", str(key).lower())), None)
            title = cls._row_title(ctx, d)
            maker = " ".join(val for key, val in d.items() if isinstance(val, str) and re.search(r"brand|make|manufactur|vendor", str(key).lower()))
            if cost is not None:
                found[ident] = (float(cost), title, maker)
        for rec in records.items:
            TextKit.walk_dicts(rec, note)
        for ident, (cost, title, maker) in list(found.items()):
            newest = records.price(ident)
            if newest is not None:
                cost = newest
            if TextKit.is_number(cap) and cost * qty > cap:
                del found[ident]
            else:
                found[ident] = (float(cost), title, maker)
        return found

    @classmethod
    def label_queries(cls, ctx) -> list[str]:
        out: list[str] = []
        for label in sorted(cls.shelf_labels(ctx)):
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
            words = [w for group in groups for w in group]
            heads = [group[-1] for group in groups]
            out.extend(cls.stem(ctx, h) for h in heads)
            if len(words) >= 2:
                out.append(" ".join(words[:2]))
            out.extend(heads)
            out.extend(w for w in words if w not in heads)
        return list(dict.fromkeys(out))

    @classmethod
    def kind_queries(cls, ctx, pairs: bool) -> list[str]:
        label = {cls.stem(ctx, w) for s in cls.shelf_labels(ctx) for w in re.findall(r"[a-z0-9]+", s)}
        titles: dict[str, tuple[float, str]] = {}
        for cost, title, maker in cls.shelf_rows(ctx).values():
            low = title.lower().strip()
            if low:
                titles[low] = (min(cost, titles.get(low, (cost, ""))[0]), maker.lower())
        count: dict[str, int] = {}
        cheapest: dict[str, float] = {}
        shown: dict[str, str] = {}
        for title, (cost, maker) in titles.items():
            makers = set(re.findall(r"[a-z]+", maker))
            words = re.findall(r"[a-z]+", title)
            usable = [len(w) >= 3 and w not in LINK_WORDS and w not in makers for w in words]
            if pairs:
                grams = [(words[i], words[i + 1]) for i in range(len(words) - 1) if usable[i] and usable[i + 1]]
            else:
                grams = [(words[i],) for i in range(len(words)) if usable[i]]
            met: set[str] = set()
            for gram in grams:
                stems = [cls.stem(ctx, w) for w in gram]
                joined = " ".join(stems)
                if all(s in label for s in stems) or joined in met:
                    continue
                met.add(joined)
                count[joined] = count.get(joined, 0) + 1
                cheapest[joined] = min(cheapest.get(joined, cost), cost)
                shown.setdefault(joined, " ".join(gram))
        common = [b for b in count if count[b] >= 2]
        return [shown[b] for b in sorted(common, key=lambda b: (cheapest[b], -count[b], b))]

    @staticmethod
    def _sig(args: dict[str, Any]) -> str:
        return json.dumps(args, sort_keys=True, default=str)

    @classmethod
    def _text_template(cls, ctx, book: dict, labels: set, page_key: str) -> tuple[dict | None, str | None]:
        ranked = [w for w in book.values() if w.get("id_order") is False and cls.free_text(ctx, w["args"])]
        if not ranked:
            ranked = [w for w in book.values() if w.get("id_order") is None and cls.free_text(ctx, w["args"])]
        if not ranked:
            return None, None
        props = (ctx.shelf.shapes.get(ctx.role("search") or "") or {}).get("properties") or {}
        args = ranked[-1]["args"]

        def fixed(k: str, v: Any) -> bool:
            if not isinstance(v, str) or (props.get(k) or {}).get("enum"):
                return True
            return v.strip().lower() in labels or k == page_key
        texts = [k for k, v in args.items() if not fixed(k, v)]
        lead = ctx.shelf.lead_ctx
        if lead and lead[0] in texts:
            text_key = lead[0]
        else:
            text_key = max(texts, key=lambda k: len(args[k].split()), default=None)
        if not text_key:
            return None, text_key
        return {k: v for k, v in args.items() if k != text_key and fixed(k, v)}, text_key

    @classmethod
    def sweep_next(cls, ctx, turn: int) -> tuple[dict[str, Any] | None, bool]:
        found = cls.page_arg(ctx)
        labels = cls.shelf_labels(ctx)
        book = vars(ctx.shelf).get("page_book", {})
        if not found or not labels or not book:
            return None, False
        key, _, low, _ = found
        tried = vars(ctx.shelf).get("sweep_tried", set())
        lists = [w for w in book.values()
                 if w.get("id_order") and w["shelves"] and w["shelves"] <= labels and cls.next_page(ctx, w) is not None]
        listing = max(lists, key=lambda w: w["total"], default=None)
        template, text_key = cls._text_template(ctx, book, labels, key)

        def lead_pair(text: str) -> tuple:
            return tuple(cls.stem(ctx, w) for w in re.findall(r"[a-z0-9]{2,}", text.lower())[:2])
        done = {lead_pair(" ".join(cls.free_text(ctx, w["args"]))) for w in book.values()
                if low in w["read"] and w.get("id_order") is not True}

        def query(text: str) -> dict[str, Any] | None:
            sig = lead_pair(text)
            if template is None or not sig or sig in done:
                return None
            args = {**template, text_key: text}
            return None if cls._sig(args) in tried else args
        heads = {cls.stem(ctx, q) for q in cls.label_queries(ctx) if " " not in q}

        def second_pages() -> list[dict[str, Any]]:
            out = []
            for w in book.values():
                said = [cls.stem(ctx, t.strip().lower()) for t in cls.free_text(ctx, w["args"])]
                if len(w["read"]) != 1 or not said or said[0] not in heads or not (w["shelves"] & labels):
                    continue
                if cls.next_page(ctx, w) is not None:
                    out.append({**w["args"], key: cls.next_page(ctx, w)})
            return out

        def from_texts(texts: list[str]) -> list[dict[str, Any]]:
            return [q for q in (query(t) for t in texts) if q]
        sources = [
            lambda: [{**listing["args"], key: cls.next_page(ctx, listing)}] if listing else [],
            lambda: from_texts(cls.kind_queries(ctx, True)[:6]),
            lambda: from_texts(cls.kind_queries(ctx, False)[:6]),
            lambda: from_texts(cls.label_queries(ctx)),
        ]
        ceiling, ended = Coverage.ceiling_sweep(ctx, tried)
        if ceiling is not None:
            return ceiling, True
        if ended:
            return None, False
        extra = [a for a in second_pages() if cls._sig(a) not in tried]
        if extra and turn % 2 == 1:
            return extra[0], False
        for pick in [0] + [1 + (turn + shift) % 3 for shift in range(3)]:
            fresh = [a for a in sources[pick]() if cls._sig(a) not in tried]
            if fresh:
                return fresh[0], pick == 0
        return None, False

    @classmethod
    def sweep_step(cls, ctx) -> bool:
        needs, wire = ctx.needs, ctx.wire
        if needs.priority() != "cheapest" or not needs.let_go or needs.props_left() or not ctx.role("search") or wire.done:
            return False
        keep = 4 if ctx.cart.trolley.units else 3
        if ctx.cart.orders_tried == 0:
            keep = max(keep + 3, FIRST_ORDER_BY)
        shelf = ctx.shelf
        swept = vars(shelf).get("swept", 0)
        if swept >= SWEEP_READS or ctx.moves_left() <= keep or wire.spare() < CLOSING + 40:
            return False
        turn = vars(shelf).get("sweep_turn", 0)
        call_args, listing = cls.sweep_next(ctx, turn)
        if call_args is None:
            return False
        shelf.swept = swept + 1
        shelf.sweep_turn = turn if listing else turn + 1
        vars(shelf).setdefault("sweep_tried", set()).add(cls._sig(call_args))
        proposal = {"id": f"sweep_{shelf.swept}_{wire.tick}", "type": "function",
                    "function": {"name": ctx.role("search"), "arguments": json.dumps(call_args)}}
        ctx.talk.chat.append({"role": "assistant", "content": "With every property requirement gone, the lowest-priced in-stock listing on the "
                              "entire shelf is the answer, so I scan another part of the shelf first.", "tool_calls": [proposal]})
        shelf.covering = True
        try:
            responses = Checkout.dispatch(ctx, [proposal], scripted=True)
        finally:
            shelf.covering = False
        more = (shelf.swept < SWEEP_READS and ctx.moves_left() > keep and wire.spare() >= CLOSING + 40
                and cls.sweep_next(ctx, shelf.sweep_turn)[0] is not None)
        if responses and not more:
            rows = sorted(cls.shelf_rows(ctx).items(), key=lambda kv: (kv[1][0], kv[0]))[:6]
            if rows:
                cid, text_body = responses[-1]
                listed = "; ".join(f"{i} '{t[:80]}' {c:g}" for i, (c, t, _) in rows)
                responses[-1] = (cid, text_body + "\n[shelf sweep done] Among all rows seen, these are the lowest-priced in-stock ones in the wanted "
                                 "category inside the budget: " + listed
                                 + ". Take the cheapest unless some change excludes it: keep it as the single cart line, "
                                 "read its stock and price again, then place the order.")
        Asking.post_replies(ctx, responses)
        return True

    @classmethod
    def free_text(cls, ctx, call_args: dict[str, Any]) -> list[str]:
        props = (ctx.shelf.shapes.get(ctx.role("search") or "") or {}).get("properties")
        if not isinstance(props, dict):
            props = {}
        token = cls.token_key(ctx)
        out = []
        for key, val in call_args.items():
            if isinstance(val, str) and not (props.get(key) or {}).get("enum") and key != token:
                out.append(val)
        return out

    @staticmethod
    def _natural(mark: str) -> list:
        return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", mark)]

    @classmethod
    def book_page(cls, ctx, call_args: dict[str, Any], rows: list, total: int, obs: Any = None) -> None:
        found = cls.page_arg(ctx)
        if not found or not rows:
            return
        key, mode, low, _ = found
        at = call_args.get(key, low)
        if mode == "token":
            at = at.strip() if isinstance(at, str) else ""
        elif not TextKit.is_number(at):
            return
        rest = {k: v for k, v in call_args.items() if k != key}
        walk = vars(ctx.shelf).setdefault("page_book", {}).setdefault(
            cls._sig(rest), {"args": rest, "read": set(), "size": 0, "total": 0, "shelves": set()})
        walk["read"].add(at)
        if mode == "token":
            walk.setdefault("tokens", {})[at] = cls.next_token(ctx, obs)
        dict_rows = [r for r in rows if isinstance(r, dict)]
        walk["size"] = max(walk["size"], len(rows))
        walk["total"] = max(walk["total"], total)
        walk["shelves"] |= {cls.shelf_of(ctx, r) for r in dict_rows} - {""}
        shown = walk.setdefault("row_shelf", {})
        for r in dict_rows:
            shown.setdefault(cls.row_mark(ctx, r), cls.shelf_of(ctx, r))
        if at == low:
            walk["first"] = tuple(cls.row_mark(ctx, r) for r in dict_rows)
            if total <= len(rows):
                walk["every"] = list(dict_rows)
        marks = [cls.row_mark(ctx, r) for r in dict_rows]
        if len(marks) >= 5:
            ordered = marks == sorted(marks) or marks == sorted(marks, key=cls._natural)
            walk["id_order"] = walk.get("id_order", True) is not False and ordered

    @classmethod
    def next_page(cls, ctx, walk: dict[str, Any]) -> Any:
        key, by_page, low, high = cls.page_arg(ctx)
        if by_page == "token":
            for following in (walk.get("tokens") or {}).values():
                if isinstance(following, str) and following not in walk["read"]:
                    return following
            return None
        low = min([low] + [v for v in walk["read"] if TextKit.is_number(v)])
        size = max(1, walk["size"])
        step = 1 if by_page else size
        for i in range(1, 400):
            val = low + i * step
            if (high is not None and val > high) or i * size >= walk["total"]:
                return None
            if val not in walk["read"]:
                return val
        return None

    @staticmethod
    def _prefix_near(a: str, b: str) -> bool:
        return min(len(a), len(b)) >= 3 and (a.startswith(b) or b.startswith(a))

    @classmethod
    def value_score(cls, ctx, call_args: dict[str, Any]) -> int:
        value_sets = []
        for req in ctx.needs.props_left():
            value = req.partition(":")[2]
            if not RefKit.ref_text(value):
                value_sets.append(re.findall(r"[a-z0-9]{3,}", value.lower()))
        words = [w for text in cls.free_text(ctx, call_args) for w in re.findall(r"[a-z0-9]{2,}", text.lower())]
        lead = ctx.shelf.lead_words
        if lead:
            words = words[:lead]
        return len([1 for value_words in value_sets if any(cls._prefix_near(w, v) for w in words for v in value_words)])

    @classmethod
    def row_hits(cls, ctx, rows: list, words: list[str] | None = None) -> list[str]:
        cap, qty = ctx.needs.spend_cap, ctx.needs.quantity()
        labels = cls.shelf_labels(ctx)
        records = ctx.shelf.archive
        hits = []
        asked = set()
        if ctx.needs.reading.get("references"):
            asked = {w for t in words or [] for w in re.findall(r"[a-z0-9]+", str(t).lower())}
        pointers = vars(ctx.pointers).setdefault("ref_hits", [])
        for r in rows:
            if not isinstance(r, dict):
                continue
            ident = cls.row_mark(ctx, r)
            makers = []
            for k2, val in r.items():
                low = str(k2).lower()
                if isinstance(val, str) and val.strip() and re.search(r"shop|seller|store|vendor|brand|make|manufactur", low) and "id" not in low:
                    makers.append(set(re.findall(r"[a-z0-9]+", val.lower())))
            unread = ident in records.ids and ident not in records.read
            if asked and unread and any(m and m <= asked and m <= cls.pointed_words(ctx) for m in makers):
                if ident not in pointers:
                    pointers.append(ident)
            cost = next((val for k2, val in r.items() if TextKit.is_number(val) and re.search(r"price|cost", str(k2).lower())), None)
            label = cls.shelf_of(ctx, r)
            if not unread:
                continue
            if TextKit.is_number(cap) and TextKit.is_number(cost) and cost * qty > cap:
                continue
            if cls.sold_out(ctx, r) or (labels and label and label not in labels):
                continue
            hits.append(ident)
            vars(ctx.shelf).setdefault("hit_shelf", {})[ident] = label
        return hits

    @classmethod
    def pointed_words(cls, ctx) -> set[str]:
        records = ctx.shelf.archive
        memo = vars(ctx.shelf).get("pointed_memo")
        if memo and memo[0] == len(records.items):
            return memo[1]
        words: set[str] = set()
        try:
            for ref in ctx.needs.reading.get("references") or []:
                words |= set(re.findall(r"[a-z0-9]+", str(ref).lower()))

            def note(d: dict[str, Any]) -> None:
                if cls.row_mark(ctx, d) not in records.read:
                    return
                own: set[str] = set()
                texts: list[str] = []
                for key, val in d.items():
                    if not isinstance(val, str):
                        continue
                    low = str(key).lower()
                    seller = (re.search(r"shop|seller|store|vendor|brand|make|manufactur", low) and "id" not in low and len(val) <= 40
                              and not re.search(r"note|text|desc|comment|detail|info|message|section", low))
                    if seller:
                        own |= set(re.findall(r"[a-z0-9]+", val.lower()))
                    elif not re.search(r"title|heading|headline|name|label|caption", low):
                        texts.append(val.lower())
                for text in texts:
                    words.update(set(re.findall(r"[a-z0-9]+", text)) - own)
            for rec in records.items:
                TextKit.walk_dicts(rec, note)
        except Exception:
            pass
        ctx.shelf.pointed_memo = (len(records.items), words)
        return words

    @classmethod
    def open_pages(cls, ctx, most: int = 2) -> list[dict[str, Any]]:
        found = cls.page_arg(ctx)
        if not found:
            return []
        labels = cls.shelf_labels(ctx)
        picks = []
        for walk in vars(ctx.shelf).get("page_book", {}).values():
            following = cls.next_page(ctx, walk)
            if following is None:
                continue
            if labels and walk["shelves"] and not walk["shelves"] & labels:
                continue
            score = cls.value_score(ctx, walk["args"])
            if score:
                picks.append((-score, walk["total"], {**walk["args"], found[0]: following}))
        picks.sort(key=lambda p: (p[0], p[1]))
        return [args for _, _, args in picks[:most]]

    @staticmethod
    def _figure_near(a: str, b: str) -> bool:
        if any(c.isdigit() for c in a + b):
            if a.lstrip("0") == b.lstrip("0"):
                return True
            return a.isdigit() and b.startswith(a) and not b[len(a):len(a) + 1].isdigit()
        return min(len(a), len(b)) >= 3 and (a.startswith(b) or b.startswith(a))

    @classmethod
    def value_probe(cls, ctx, raw: list[str]) -> tuple[str, str]:
        shelf = ctx.shelf
        if shelf.word_kind == "any":
            return "", ""
        depth = shelf.lead_words or 2
        lead = [w for w in re.findall(r"[a-z0-9]+", " ".join(raw).lower()) if len(w) >= 2][:depth]
        product = " ".join(re.findall(r"[a-z0-9]+", str(ctx.needs.reading.get("product") or "").lower())[:3])
        near = cls._figure_near
        reqs = []
        for req in ctx.needs.props_left():
            if ":" not in req or RefKit.ref_text(req.partition(":")[2]):
                continue
            head, _, value = req.partition(":")
            first = re.split(r"[,;]", re.sub(r"\((?:stated|revealed[^)]*)\)", "", value))[0].strip().lower()
            reqs.append((head.strip(), first))
        keep = [w for _, value in reqs for w in re.findall(r"[a-z0-9]{2,}", value)] + product.split()
        if not lead or any(near(w, k) for w in lead for k in keep):
            return "", ""
        for name, value in reqs:
            if not value or not any(near(w, n) for w in lead for n in re.findall(r"[a-z0-9]{2,}", name.lower())):
                continue
            digit = re.search(r"\d", value)
            probe = f"{(value[digit.start():] if digit else value).strip(' .;')} {product}".strip()
            head = [w for w in re.findall(r"[a-z0-9]+", probe.lower()) if len(w) >= 2][: shelf.lead_words or 2]
            for walk in vars(shelf).get("page_book", {}).values():
                ran = [w for t in cls.free_text(ctx, walk["args"]) for w in re.findall(r"[a-z0-9]+", t.lower()) if len(w) >= 2][: shelf.lead_words or 2]
                if len(ran) == len(head) and all(near(a, b) for a, b in zip(head, ran)):
                    return "", ""
            return name, probe
        return "", ""


class Asking:
    @staticmethod
    def _tag_of(label: Any, fallback: str) -> str:
        tied = re.search(r"\(([^)]+)\)", label)
        return tied.group(1).strip() if tied else fallback

    @classmethod
    def _plain_ask(cls, ctx, label: str) -> str:
        product = cls._tag_of(label, str(ctx.needs.reading.get("product") or "item"))
        return f"Which {label.split('(')[0].strip()} would you like for the {product}?"

    @staticmethod
    def _no_second(text: str) -> bool:
        return bool(re.search(r"none|^-$|n/a", text.lower()))

    @staticmethod
    def _bring_first(queue: list[str], prop: str) -> list[str]:
        return [prop] + [p for p in queue if p.lower() != prop.lower()]

    @staticmethod
    def _add_open(pairs: list[str], prop: str) -> None:
        if prop.lower() not in [p.lower() for p in pairs]:
            pairs.append(prop)

    @classmethod
    def text_arg(cls, ctx, the_tool: str) -> str | None:
        schema = ctx.shelf.shapes.get(the_tool or "") or {}
        specs = schema.get("properties") or {}
        required = list(schema.get("required") or [])
        texts = []
        for name, spec in specs.items():
            if not isinstance(spec, dict) or "enum" in spec:
                continue
            if spec.get("type") == "string" or "string" in (spec.get("type") or []):
                texts.append(name)
        target = next((name for name in required if name in texts), None)
        if target is None and len(texts) == 1:
            target = texts[0]
        if target and all(name == target for name in required):
            return target
        return None

    @classmethod
    def message_call(cls, ctx, text_body: str, cid: str) -> dict[str, Any] | None:
        tool = ctx.role("message")
        target = cls.text_arg(ctx, tool) if tool else None
        if not target:
            return None
        cleaned, faults = ArgCheck.execute({target: text_body}, ctx.shelf.shapes[tool])
        if faults:
            return None
        return {"id": cid, "type": "function", "function": {"name": tool, "arguments": json.dumps(cleaned)}}

    @classmethod
    def product_of(cls, ctx, text: str) -> int | None:
        rows = ctx.needs.items()
        if not rows:
            return None
        kinds = [set(PropKit.kind_tokens(row["product"])) for row in rows]
        pieces = re.findall(r"\(([^)]*)\)", str(text))[:1]
        pieces.append(str(text))
        for piece in pieces:
            said = set(PropKit.kind_tokens(piece))
            marks = []
            for k, own in enumerate(kinds):
                others = set().union(*(kinds[j] for j in range(len(kinds)) if j != k))
                marks.append(PropKit.kind_score(own, said, others))
            best = max(marks)
            if best > 0 and marks.count(best) == 1:
                return marks.index(best)
        return None

    @classmethod
    def plan_product(cls, ctx, qitem: dict[str, Any]) -> int | None:
        k = cls.product_of(ctx, str(qitem.get("property") or ""))
        if k is None:
            k = cls.product_of(ctx, str(qitem.get("question") or ""))
        return k

    @classmethod
    def _tally_asked(cls, ctx, count: int) -> tuple[list[int], list[int]]:
        answered, messaged = [0] * count, [0] * count
        for entry in ctx.talk.questions_log:
            head = entry.split(":", 1)[0]
            k = cls.product_of(ctx, head)
            if k is None:
                quoted = re.search(r'asked "(.*?)" -> ', entry)
                if quoted:
                    k = cls.product_of(ctx, quoted.group(1))
            if k is None:
                continue
            messaged[k] += 1
            if "[not answered]" not in head:
                answered[k] += 1
        return answered, messaged

    @classmethod
    def _story_products(cls, ctx, count: int, answered: list[int]) -> set[int]:
        named, keyless = set(), [0] * count
        for hint in Intake.open_hints(ctx):
            j = cls.product_of(ctx, str(hint).split(":", 1)[0])
            if j is None:
                continue
            if Intake.plain_hint(ctx, j, str(hint).split(":", 1)[0].strip(), hinted=True):
                keyless[j] += 1
            else:
                named.add(j)
        return {j for j in range(count) if j in named or keyless[j] > answered[j]}

    @classmethod
    def quota_first(cls, ctx, scripted: list[dict[str, Any]], ok: Any = None) -> list[dict[str, Any]]:
        rows = ctx.needs.items()
        if not rows:
            return scripted
        count = len(rows)
        answered, messaged = cls._tally_asked(ctx, count)
        if scripted and max(answered) == 0 and cls.plan_product(ctx, scripted[0]) is None:
            return scripted
        revealed = [0] * count
        for need in ctx.needs.uncovered:
            k = cls.product_of(ctx, str(need).split(":", 1)[0])
            if k is not None:
                revealed[k] += 1
        try:
            hinted = {cls.product_of(ctx, str(h).split(":", 1)[0]) for h in Intake.open_hints(ctx)}
        except Exception:
            hinted = set()
        never = [j for j in range(count) if not answered[j]]
        try:
            hinted = cls._story_products(ctx, count, answered)
        except Exception:
            hinted = set()
        untried = set()
        for j in range(count):
            if messaged[j] > answered[j] and Intake.alt_untried(ctx, j):
                untried.add(j)

        def rank(k: int) -> tuple[int, int, int, int, int]:
            behind = 1 if revealed[k] and any(j != k for j in never) else 0
            silent = 0 if k in untried else messaged[k] - answered[k]
            return behind, 0 if k in hinted else 1, answered[k] + revealed[k], silent, revealed[k]

        everyone = list(range(count))
        for level in sorted({rank(k) for k in everyone}):
            due = [k for k in everyone if rank(k) == level]
            for pos, qitem in enumerate(scripted):
                if cls.plan_product(ctx, qitem) in due:
                    return [qitem] + scripted[:pos] + scripted[pos + 1:]
            for qitem in scripted:
                second = str(qitem.get("second") or "").strip()
                j = cls.product_of(ctx, second) if second else None
                if j not in due:
                    continue
                product = cls._tag_of(second, rows[j]["product"])
                question = f"Which {second.split('(')[0].strip()} would you like for the {product}?"
                return [{"property": second, "question": question}] + scripted
            for k in sorted(due, key=lambda k: (k not in hinted, k)):
                try:
                    pick = cls.quota_pick(ctx, k, answered[k] == 0, ok)
                except Exception:
                    pick = None
                if pick and ok is not None:
                    label = pick.get("property")
                    if not ok(label) or Intake.asked_for(ctx, k, label):
                        pick = None
                if pick:
                    return [pick] + scripted
                if vars(ctx.talk).get("quota_wait"):
                    return scripted
        return scripted

    @classmethod
    def quota_pick(cls, ctx, k: int, fallback: bool = True, ok: Any = None) -> dict[str, Any] | None:
        records = ctx.shelf.archive
        product = ctx.needs.items()[k]["product"]
        homes = Lookup.home_shelves(ctx)
        home = homes[k] if k < len(homes) else set()
        own_read = [shelf for shelf in records.shelves_read() if shelf.strip().lower() in home]
        opened_before = vars(ctx.talk).get("quota_reads", {}).get(k)
        if home and not own_read and opened_before:
            own_read = [shelf for shelf in sorted(home) if records.prop_menu(shelf)[0]][:1]
        if not own_read:
            batch = Lookup.batch_tool(ctx)
            unread = []
            if home:
                unread = Lookup.menu_rows(ctx, home, min(4, batch[2]) if batch else 2, [], ctx.needs.items()[k]["quantity"])
            can_open = ctx.role("details") or batch
            if unread and can_open and not vars(ctx.talk).get("quota_reads", {}).get(k) and ctx.moves_left() > 12:
                vars(ctx.talk).setdefault("quota_reads", {})[k] = True
                ctx.shelf.open_queue = unread
                ctx.shelf.open_recheck = True
                ctx.shelf.open_say = f"To learn what fields the {product} listings have, I read a few of them before the question."
                ctx.shelf.open_purpose = "quota"
                ctx.talk.quota_wait = True
            return None
        menu = Intake.menu_trim(ctx, records.prop_menu(own_read[0])[0], records.rich_reads(own_read[0]))

        def usable(label: Any) -> bool:
            if ok is None:
                return True
            return bool(ok(label)) and not Intake.asked_for(ctx, k, label)

        def ask_about(prop: str) -> dict[str, Any]:
            return {"property": f"{prop} ({product})", "question": f"Which {prop} would you like for the {product}?"}

        planned = cls.product_plan(ctx, k, menu)
        if planned and usable(planned.get("property")):
            return planned
        try:
            story = Intake.open_hints(ctx, k)
        except Exception:
            story = []
        for hint in story:
            prop = re.split(r"\s*\(", hint.split(":", 1)[0])[0].strip()
            if not prop or TextKit.junk_key(prop) or Intake.plain_hint(ctx, k, prop, hinted=True):
                continue
            if usable(f"{prop} ({product})"):
                return ask_about(prop)
        if not fallback:
            return None
        for by_samples in (False, True):
            for line in menu.splitlines():
                prop = line.strip("- ").split(" (in ")[0].strip()
                if not prop or not cls.open_prop(ctx, k, prop, line, by_samples=by_samples):
                    continue
                if usable(f"{prop} ({product})"):
                    return ask_about(prop)
        return None

    @classmethod
    def basket_menus(cls, ctx) -> dict[int, str]:
        records = ctx.shelf.archive
        homes = Lookup.home_shelves(ctx)
        menus = {}
        for k in range(len(ctx.needs.items())):
            home = homes[k] if k < len(homes) else set()
            own_read = [shelf for shelf in records.shelves_read() if shelf.strip().lower() in home]
            if own_read:
                menu = Intake.menu_trim(ctx, records.prop_menu(own_read[0])[0], records.rich_reads(own_read[0]))
                if menu:
                    menus[k] = menu
        return menus

    @classmethod
    def basket_rank(cls, ctx) -> dict[int, list[str]]:
        # one ranking call per basket covers every product whose listings were read; a later call ranks the rest
        talk = ctx.talk
        queues = vars(talk).setdefault("basket_queues", {})
        menus = {k: menu for k, menu in cls.basket_menus(ctx).items() if k not in queues}
        if not menus or vars(talk).get("basket_ranks", 0) >= 3 or ctx.wire.spare() < CLOSING + 40:
            return queues
        talk.basket_ranks = vars(talk).get("basket_ranks", 0) + 1
        rows = ctx.needs.items()
        blocks = []
        for k, menu in sorted(menus.items()):
            stated = "; ".join(rows[k]["requirements"]) or "nothing"
            blocks.append(f"## P{k + 1}: {rows[k]['product']}\n- already stated: {stated}\n{menu}")
        answer = ctx.wire.consult(BASKET_RANK + "\n\n# Request\n" + ctx.request + "\n\n# Products\n" + "\n\n".join(blocks), 1500, RANKER)
        ctx.log.append({"kind": "basket_rank", "products": sorted(menus), "result": answer})
        for k, menu in menus.items():
            names = [line.strip().lstrip("-").strip().split(" (in ")[0].strip() for line in menu.splitlines()]
            names = [name for name in names if name]
            pick = (answer or {}).get(f"P{k + 1}")
            pick = pick if isinstance(pick, dict) else {}
            ranked = [str(p) for p in pick.get("ranked") or []][:BASKET_FIRST]
            story = [str(p) for p in pick.get("story") or []]
            order = []
            for said in ranked + story:
                known = next((name for name in names if PropKit.strict_prop(name, said)), None)
                if known and known not in order:
                    order.append(known)
            queues[k] = order + [name for name in names if name not in order]
        return queues

    @classmethod
    def basket_script(cls, ctx, fresh) -> list[dict[str, Any]]:
        try:
            queues = cls.basket_rank(ctx)
        except Exception as err:
            ctx.log.append({"kind": "basket_rank_error", "error": f"{type(err).__name__}: {err}"[:300]})
            return []
        rows = ctx.needs.items()
        script = []
        for k, queue in sorted(queues.items()):
            product = rows[k]["product"]
            for prop in queue:
                if TextKit.junk_key(prop):
                    continue
                label = f"{prop} ({product})"
                if fresh(label) and not Intake.asked_for(ctx, k, label):
                    script.append({"property": label, "second": "", "question": f"Which {prop} would you like for the {product}?"})
                    break
        return script

    @classmethod
    def product_plan(cls, ctx, k: int, menu: str) -> dict[str, Any] | None:
        try:
            return cls.product_plan_call(ctx, k, menu)
        except Exception:
            return None

    @classmethod
    def product_plan_call(cls, ctx, k: int, menu: str) -> dict[str, Any] | None:
        if not menu or vars(ctx.talk).get("product_plans", 0) >= 1 or ctx.wire.spare() < CLOSING + 40:
            return None
        ctx.talk.product_plans = vars(ctx.talk).get("product_plans", 0) + 1
        product = ctx.needs.items()[k]["product"]
        room = max(1, ctx.ask_room() - ctx.budget.msgs_sent)
        parts = [ASK_PLAN.replace("QUESTION_COUNT", str(min(3, room))), "\n\nTHE SHOPPER'S REQUEST:\n", ctx.request,
                 "\n\nREQUIREMENTS SO FAR:\n", ctx.needs.text_body()]
        if ctx.talk.questions_log:
            parts.append("\n\nQUESTIONS SENT ALREADY (each with the shopper's reply):\n" + "\n".join(ctx.talk.questions_log))
        parts.append(f"\n\nFOCUS ON THE {product.upper()} ONLY: each other requested product has had a question already, "
                     f"this one none. Pick properties of the {product} from its own field list below, beginning with "
                     "the one the story's details hint at.")
        try:
            story = Intake.open_hints(ctx, k)
        except Exception:
            story = []
        if story:
            parts.append("\n\nSTORY HINTS (story details paired with the property each seems to touch, likeliest first; "
                         "they are not requirements):\n" + "\n".join(f"- {Intake.hint_mark(ctx, h)}" for h in story[:6]))
        parts.append(cls.plan_notes(ctx, k))
        parts.append(f"\n\nPROPERTIES IN THE DETAILS OF THE {product.upper()} LISTINGS SEEN SO FAR (name, number of listings, "
                     "example values):\n" + menu)
        answer = ctx.wire.consult("".join(parts), 1500, PLANNER[:1])
        ctx.log.append({"kind": "product_plan", "product": product, "result": answer})
        kinds = PropKit.kind_words(product)
        for qitem in (answer or {}).get("questions") or []:
            if not isinstance(qitem, dict):
                continue
            first = str(qitem.get("property") or "").split("(")[0].strip()
            second = str(qitem.get("second") or "").split("(")[0].strip()
            if cls._no_second(second):
                second = ""
            question = str(qitem.get("question") or "").strip()
            if Intake.plain_hint(ctx, k, first):
                first = ""
            if second and Intake.plain_hint(ctx, k, second):
                second = ""
            if first and Intake.plan_open(ctx, k, first):
                names_it = bool(question) and bool(kinds & PropKit.kind_words(question))
                pair = ""
                if names_it and second and not PropKit.strict_prop(first, second) and Intake.plan_open(ctx, k, second):
                    pair = second
                return {"property": f"{first} ({product})", "second": f"{pair} ({product})" if pair else "",
                        "question": question if names_it else f"Which {first} would you like for the {product}?"}
            if second and Intake.plan_open(ctx, k, second):
                return {"property": f"{second} ({product})", "question": f"Which {second} would you like for the {product}?"}
        return None

    @classmethod
    def _prop_settled(cls, ctx, k, name: str, samples: str, by_samples: bool, mine) -> bool:
        for req in ctx.needs.props_left():
            head, _, value = str(req).partition(":")
            if not mine(head):
                continue
            if PropKit.strict_prop(head, name):
                if Intake.settles(ctx, str(req), samples if by_samples else ""):
                    return True
                continue
            if PropKit.same_prop(head, name):
                if ValueKit.positive_value(value, head) and ValueKit.value_seen(
                        value, samples or " | ".join(Intake.field_values(ctx, k, name))):
                    return True
                continue
            words = [w for w in re.findall(r"[a-z]{3,}", value.lower()) if w not in LINK_WORDS]
            if not (samples and words) or re.search(r"\d", value) or not ValueKit.positive_value(value, head):
                continue
            if all(re.search(r"\b" + re.escape(w), samples) for w in words):
                return True
        return False

    @classmethod
    def open_prop(cls, ctx, k: int | None, prop: str, line: str = "", by_samples: bool = False) -> bool:
        try:
            name = str(prop).split("(")[0].strip()
            if not name or TextKit.junk_key(name):
                return False
            several = bool(ctx.needs.items())

            def mine(text: str, tagged_only: bool = False) -> bool:
                if k is None or not several:
                    return True
                j = cls.product_of(ctx, text)
                if j is None:
                    return not tagged_only
                return j == k
            for entry in ctx.talk.questions_log:
                head = entry.split(":", 1)[0].strip("- ")
                if "[not answered]" in head:
                    continue
                if any(mine(part) and Intake.same_field(ctx, k, part, name) for part in head.split(";")):
                    return False
            samples = line.split("):", 1)[1].lower() if "):" in line else ""
            if cls._prop_settled(ctx, k, name, samples, by_samples, mine):
                return False
            for p in ctx.needs.indifferent:
                if mine(str(p), tagged_only=True) and Intake.same_field(ctx, k, str(p), name):
                    return False
            return True
        except Exception:
            return True

    @classmethod
    def plan_notes(cls, ctx, k: int | None) -> str:
        try:
            several = bool(ctx.needs.items())
            asked = set()
            for entry in ctx.talk.questions_log:
                for part in entry.split(":", 1)[0].strip("- ").split(";"):
                    asked.add(part.strip().lower())
            still_open = []
            for prop in vars(ctx.talk).get("open_pair") or []:
                j = cls.product_of(ctx, prop) if several else None
                if k is not None and several and j not in (k, None):
                    continue
                if prop.lower() in asked or not Intake.plan_open(ctx, j, prop):
                    continue
                still_open.append(prop)
            wanted = []
            for text in vars(ctx.talk).get("wanted_asks") or []:
                if k is None or not several or cls.product_of(ctx, text) in (k, None):
                    wanted.append(text)
            notes = ""
            if still_open:
                notes += ("\n\nSTILL OPEN AFTER A PAIRED QUESTION (a planned pair's second property that its reply never settled - "
                          "only the first got an answer, or only the first was sent - so it remains open; each one was "
                          "next in the plan):\n" + "\n".join(f"- {p}" for p in still_open[-3:]))
            if wanted:
                notes += ("\n\nQUESTIONS THE ASSISTANT MEANT TO SEND (it has seen the listing records; when such a question names "
                          "a property that is still open and appears as a key below, that key goes first under its own name):\n"
                          + "\n".join(f"- {w}" for w in wanted[-3:]))
            return notes
        except Exception:
            return ""

    @classmethod
    def drop_wanted(cls, ctx, label: str) -> None:
        try:
            props = [re.split(r"\s*\(", p)[0] for p in re.split(r";|\band\b", str(label).replace("[not answered]", ""))]
            left = []
            for text in vars(ctx.talk).get("wanted_asks") or []:
                stems = {Reading.stem(ctx, w) for w in re.findall(r"[a-z]+", text.lower())}

                def covered(p: str) -> bool:
                    words = PropKit.prop_words(p)
                    return bool(words) and all(w in stems or Reading.stem(ctx, w) in stems for w in words)
                if not any(covered(p) for p in props):
                    left.append(text)
            ctx.talk.wanted_asks = left
        except Exception:
            pass

    @classmethod
    def names_other(cls, ctx, question: str, prop: str, menu: str, second: str = "") -> bool:
        try:
            flat = " ".join(re.findall(r"[a-z0-9]+", str(question).lower()))
            said = set(flat.split())
            said |= {w[:-1] for w in said if len(w) > 3 and w.endswith("s") and not w.endswith("ss")}
            own = set(PropKit.prop_words(prop)) | {"which", "would", "you", "like"}
            rows = ctx.needs.items() or [{"product": str(ctx.needs.reading.get("product") or "")}]
            for row in rows:
                for w in re.findall(r"[a-z0-9]+", str(row.get("product") or "").lower()):
                    own.add(w)
                    own.add(w[:-1] if len(w) > 3 and w.endswith("s") else w)

            def mentions(name: str) -> bool:
                words = [w for w in PropKit.prop_words(name) if len(w) >= 3 and w not in LINK_WORDS]
                return bool(words) and all(w in said for w in words) and any(w not in own for w in words)

            def sample_named(line: str) -> bool:
                tail = line.split("):", 1)[1] if "):" in line else ""
                for sample in tail.split("|"):
                    toks = re.findall(r"[a-z0-9]+", sample.lower())
                    if len("".join(toks)) < 3 or all(t in own or t in LINK_WORDS for t in toks):
                        continue
                    if re.search(r"(?<![a-z0-9])" + re.escape(" ".join(toks)) + r"(?![a-z0-9])", flat):
                        return True
                return False
            if second and not PropKit.strict_prop(second, prop) and mentions(second):
                return True
            for line in str(menu or "").splitlines():
                key = line.strip().lstrip("-").strip().split(" (in ")[0].strip()
                if not key or key.endswith(":") or PropKit.strict_prop(key, prop) or TextKit.junk_key(key):
                    continue
                if mentions(key) or sample_named(line):
                    return True
            return False
        except Exception:
            return False

    @classmethod
    def queue_second(cls, ctx, prop: str) -> None:
        queue = vars(ctx.talk).setdefault("pending_seconds", [])
        if prop:
            cls._add_open(queue, prop)

    @classmethod
    def _plan_text(cls, ctx, room: int, prop_list: str, sets: int) -> tuple[str, str, str]:
        parts = [ASK_PLAN.replace("QUESTION_COUNT", str(room)), NEXT_ONE.replace("REMAINING", str(room)),
                 "\n\nTHE SHOPPER'S REQUEST:\n", ctx.request, "\n\nREQUIREMENTS SO FAR:\n", ctx.needs.text_body()]
        if ctx.talk.questions_log:
            parts.append("\n\nQUESTIONS SENT ALREADY (each with the shopper's reply):\n" + "\n".join(ctx.talk.questions_log))
        if ctx.needs.reading.get("ask_first"):
            parts.append("\n\nMUST BE LEARNED FROM THE SHOPPER: " + "; ".join(str(h) for h in ctx.needs.reading.get("ask_first") or []))
        parts.append(cls.plan_notes(ctx, None))
        try:
            story = Intake.open_hints(ctx)
        except Exception:
            story = []
        if story:
            parts.append("\n\nSTORY HINTS (story details paired with the property each seems to touch; they are not "
                         "requirements; the hidden need tends to be one of them):\n"
                         + "\n".join(f"- {Intake.hint_mark(ctx, h)}" for h in story[:8]))
        records = ctx.shelf.archive
        shown = Intake.menu_trim(ctx, prop_list, sets)
        rows = ctx.needs.items()
        if rows:
            parts.append("\n\nTHE PRODUCTS ASKED FOR: " + " | ".join(f"{n + 1}) {row['product']} x{row['quantity']}" for n, row in enumerate(rows)))
            read = records.shelves_read()
            homes = Lookup.home_shelves(ctx)
            own = [shelf for shelf in read if shelf.strip().lower() in set().union(*homes)]
            if all(homes):
                order = own or read
            else:
                order = own + [shelf for shelf in read if shelf not in own]
            menus = [(shelf, records.prop_menu(shelf)[0]) for shelf in order[:4]]
            prop_list = "\n".join(f"CATEGORY {shelf}:\n{menu}" for shelf, menu in menus if menu) or prop_list
            shown = "\n".join(f"CATEGORY {shelf}:\n{Intake.menu_trim(ctx, menu, records.rich_reads(shelf))}"
                              for shelf, menu in menus if menu) or shown
        parts.append("\n\nPROPERTIES IN THE LISTINGS' DETAILS SEEN SO FAR (name, number of listings, example values):\n"
                     + (shown or "(none seen yet)"))
        return "".join(parts), prop_list, shown

    @classmethod
    def _filtered_plan(cls, ctx, answer: Any, fresh) -> list[dict[str, Any]]:
        several = bool(ctx.needs.items())
        kept = []
        for qitem in (answer or {}).get("questions") or []:
            if not isinstance(qitem, dict) or not str(qitem.get("question") or "").strip():
                continue
            second = str(qitem.get("second") or "").strip()
            if cls._no_second(second):
                second = ""
            if fresh(qitem.get("property")):
                kept.append(dict(qitem, second=second if second and fresh(second) else ""))
                if second and Intake.plain_hint(ctx, cls.product_of(ctx, second) if several else None, second):
                    kept[-1]["question"] = cls._plain_ask(ctx, str(qitem.get("property") or "").strip())
            elif second and fresh(second):
                kept.append({"property": second, "question": cls._plain_ask(ctx, second)})
        return kept

    @classmethod
    def _choose_message(cls, ctx, scripted, pending: str, prop_list: str):
        proposal, label, split_second = None, None, ""
        several = bool(ctx.needs.items())
        for thing in scripted[:2]:
            if several:
                prop = str(thing.get("property") or "")
                product = cls._tag_of(prop, "")
                words = re.findall(r"[a-z]{3,}", product.lower())
                asked_text = str(thing.get("question") or "").lower()
                if product and words and not any(w.rstrip("s") in asked_text for w in words):
                    thing = dict(thing, question=f"Which {prop.split('(')[0].strip()} would you like for the {product}?")
            if not pending and not (ctx.needs.reading.get("ask_first") and not ctx.talk.questions_log) \
                    and cls.names_other(ctx, str(thing.get("question") or ""), str(thing.get("property") or ""),
                                        prop_list, str(thing.get("second") or "")):
                thing = dict(thing, question=cls._plain_ask(ctx, str(thing.get("property") or "")))
            proposal = cls.message_call(ctx, str(thing.get("question")).strip(), f"ask_{ctx.budget.msgs_sent + 1}_{ctx.wire.tick}")
            if not proposal:
                continue
            label = (str(thing.get("property")), str(thing.get("question")).strip())
            second = str(thing.get("second") or "").strip()
            if not second or second.lower() == label[0].lower() or cls._no_second(second):
                break
            single = cls._plain_ask(ctx, label[0])
            proposal = cls.message_call(ctx, single, f"ask_{ctx.budget.msgs_sent + 1}_{ctx.wire.tick}")
            label = (label[0], single)
            if proposal:
                split_second = second
                break
        return proposal, label, split_second

    @classmethod
    def _no_question(cls, ctx, sets: int, urge: bool) -> bool:
        talk = ctx.talk
        if sets < ASK_AFTER_SETS and ctx.moves_left() > 12 and not urge:
            talk.plan_wait = sets
            return False
        several = bool(ctx.needs.items())
        try:
            unread = [k for k in range(len(ctx.needs.items()) or 1) if Intake.product_reads(ctx, k if several else None) < 2]
        except Exception:
            unread = []
        if unread and ctx.moves_left() > 12 and not urge:
            talk.plan_wait = sets
            return False
        talk.empty_plans = vars(talk).get("empty_plans", 0) + 1
        if ctx.moves_left() > 12 and talk.empty_plans <= 1:
            talk.plan_wait = sets
            return False
        talk.asking = False
        return False

    @classmethod
    def _hold_question(cls, ctx, sets, pressing, urge, ceiling):
        talk = ctx.talk
        if not (sets >= ASK_AFTER_SETS or pressing or urge):
            tick = ctx.wire.tick
            if tick < ASK_AFTER_STEPS or (tick < ASK_AFTER_STEPS + ASK_GRACE_STEPS and ctx.moves_left() > 20):
                return True
        if ctx.moves_left() <= max(10, Intake.order_due(ctx)) or ctx.wire.spare() < CLOSING + 40:
            talk.asking = False
            return True
        if urge:
            return False
        waited = vars(talk).get("plan_wait")
        if waited is not None and sets <= waited:
            return True
        last = vars(talk).get("asked_at_sets")
        if last is None or sets > last or pressing:
            return False
        return ctx.moves_left() > Intake.order_due(ctx) + 2 * max(0, ceiling - ctx.budget.msgs_sent) + 14 \
            and ctx.wire.tick - vars(talk).get("asked_at_tick", 0) < 4 and not ctx.needs.items()

    @classmethod
    def _fresh_prop(cls, ctx, asked_props, prop):
        prop = str(prop or "").strip()
        if not prop or prop.lower() in asked_props or TextKit.junk_key(prop):
            return False
        if prop.lower() in vars(ctx.talk).get("spent_asks", set()):
            return False
        j = cls.product_of(ctx, prop) if ctx.needs.items() else None
        return not Intake.plain_hint(ctx, j, prop) and Intake.plan_open(ctx, j, prop)

    @classmethod
    def pose_planned(cls, ctx, urge: bool = False) -> bool:
        talk, records = ctx.talk, ctx.shelf.archive
        tool = ctx.role("message")
        ceiling = ctx.ask_room()
        if not tool or ctx.budget.msgs_sent >= ceiling:
            talk.asking = False
            return False
        prop_list, _ = records.prop_menu()
        sets = records.rich_reads()
        own = Lookup.own_shelf(ctx)
        if own:
            prop_list, sets = records.prop_menu(own)[0], records.rich_reads(own)
        elif ctx.moves_left() > 12 and Lookup.only_pointed_reads(ctx):
            # the listings read so far are the pages the request points at: their fields are no menu for this product
            return False
        pressing = bool(ctx.needs.reading.get("ask_first")) and not talk.questions_log
        if cls._hold_question(ctx, sets, pressing, urge, ceiling):
            return False
        if Lookup.pre_read(ctx):
            return True
        if not (talk.questions_log or ctx.budget.msgs_sent or pressing or urge) and Intake.first_reads(ctx):
            return True
        room = ceiling - ctx.budget.msgs_sent
        ask_text, prop_list, _ = cls._plan_text(ctx, room, prop_list, sets)
        asked_props = {part.strip().lower() for entry in talk.questions_log
                       for part in entry.split(":", 1)[0].strip("- ").split(";")}
        fresh = partial(cls._fresh_prop, ctx, asked_props)
        queue = [p for p in vars(talk).get("pending_seconds") or [] if fresh(p)]
        talk.pending_seconds = queue
        pending = queue[0] if queue else ""
        # several products: each product's question comes from the ranked queues; the planner keeps the must-learn step
        script = cls.basket_script(ctx, fresh) if ctx.route()["asker"] == "ranker" and not pending and not pressing else []
        if pending:
            own_text = (vars(talk).get("pending_text") or {}).pop(pending.lower(), "")
            answer = {"questions": [{"property": pending, "question": own_text or cls._plain_ask(ctx, pending)}]}
        elif script:
            answer = None
        else:
            answer = ctx.wire.consult(ask_text, 2500, PLANNER)
            if answer is None and ctx.wire.misses > 0:
                talk.plan_wait = sets
                return False
        scripted = script or cls._filtered_plan(ctx, answer, fresh)
        if ctx.needs.items() and not pending:
            scripted = cls.quota_first(ctx, scripted, fresh)
            if vars(talk).pop("quota_wait", False):
                return True
        proposal, label, split_second = cls._choose_message(ctx, scripted, pending, prop_list)
        if not proposal:
            return cls._no_question(ctx, sets, urge)
        talk.asked_at_sets, talk.asked_at_tick = sets, ctx.wire.tick
        talk.chat.append({"role": "assistant", "content": f"I will ask the shopper about {label[0]}.", "tool_calls": [proposal]})
        found_before = len(ctx.needs.uncovered)
        freed_before = len(ctx.needs.indifferent) + vars(ctx.needs).get("pinned_heard", 0)
        talk.asking_label = label[0]
        try:
            responses = Checkout.dispatch(ctx, [proposal], scripted=True)
        finally:
            talk.asking_label = ""
        cls.post_replies(ctx, responses)
        asked_label = label[0]
        if pending and label[0].split(";")[0].strip().lower() == pending.lower():
            talk.pending_seconds = [p for p in vars(talk).get("pending_seconds") or [] if p.lower() != pending.lower()]
        heard_meta = vars(ctx.needs).pop("meta_last", None)
        label, open_note = cls._after_reply(ctx, label, heard_meta, split_second, found_before, freed_before)
        if heard_meta is None and len(ctx.needs.uncovered) == found_before:
            vars(talk).setdefault("quiet_asks", []).append(asked_label.split(";")[0].strip())
        for _, text_body in responses[:1]:
            talk.questions_log.append(f"- {label[0]}: asked \"{label[1]}\" -> {text_body.split(chr(10) + '[status]')[0][:400]}{open_note}")
        cls.drop_wanted(ctx, asked_label)
        talk.asking = ctx.budget.msgs_sent < ceiling
        return True

    @staticmethod
    def _flat(text: Any) -> str:
        return " ".join(re.findall(r"[a-z0-9]+", str(text).lower()))

    @classmethod
    def _one_at_a_time(cls, ctx, label, split_second: str):
        talk = ctx.talk
        talk.no_pairs = True
        parts = [p.strip() for p in label[0].split(";") if p.strip()] or [label[0]]
        first = parts[0]
        again = vars(talk).setdefault("reasked", set())
        plain_q = cls._plain_ask(ctx, first)
        alone = len(parts) == 1
        alt = Intake.collide_alt(ctx, first) if alone else ""
        if alt:
            again.add(first.lower())
            vars(talk).setdefault("spent_asks", set()).add(first.lower())
            talk.pending_seconds = cls._bring_first(vars(talk).get("pending_seconds") or [], alt)
            note = f" [{first}: not answered, its story hint's alternative {alt} asked next]"
        elif first.lower() not in again and cls._flat(plain_q) != cls._flat(label[1]) \
                and not (alone and Intake.holds_plain(ctx, label[1], plain_q)):
            again.add(first.lower())
            talk.pending_seconds = cls._bring_first(vars(talk).get("pending_seconds") or [], first)
            note = f" [{first}: not answered, asked again alone next]"
        else:
            redo = Intake.distinct_ask(ctx, first) if first.lower() not in again else None
            if redo:
                again.update({first.lower(), redo[0].lower()})
                vars(talk).setdefault("pending_text", {})[redo[0].lower()] = redo[1]
                talk.pending_seconds = cls._bring_first(vars(talk).get("pending_seconds") or [], redo[0])
                itself = re.split(r"\s*\(", redo[0])[0].strip().lower() == re.split(r"\s*\(", first)[0].strip().lower()
                if not itself:
                    vars(talk).setdefault("spent_asks", set()).add(first.lower())
                note = f" [{first}: not answered, asked again " + ("by its values next]" if itself else f"as {redo[0]} next]")
            else:
                vars(talk).setdefault("spent_asks", set()).add(first.lower())
                note = f" [{first}: not answered, not asked again]"
        pairs = vars(talk).setdefault("open_pair", [])
        for second in parts[1:] + ([split_second] if split_second else []):
            cls._add_open(pairs, second)
        return (" and ".join(parts) + " [not answered] [one at a time]", label[1]), note

    @classmethod
    def _after_reply(cls, ctx, label, heard_meta, split_second: str, found_before: int, freed_before: int):
        needs, talk = ctx.needs, ctx.talk
        if heard_meta is not None and cls._flat(label[1]) and cls._flat(label[1]) in cls._flat(heard_meta):
            return cls._one_at_a_time(ctx, label, split_second)
        note = ""
        paired = ";" in label[0]
        if paired and len(needs.uncovered) > found_before:
            second = label[0].split(";", 1)[1].strip()
            found_text = " ".join(needs.uncovered[found_before:]).lower()
            if not any(w in found_text for w in re.findall(r"[a-z]{4,}", second.split("(")[0].lower())):
                cls.queue_second(ctx, second)
                note = f" [{second}: not answered, still open]"
            return (label[0].split(";")[0].strip(), label[1]), note
        if paired and len(needs.indifferent) + vars(needs).get("pinned_heard", 0) > freed_before:
            second = label[0].split(";", 1)[1].strip()
            label = (label[0].split(";")[0].strip(), label[1])
            gone = (second.lower(), second.split("(")[0].strip().lower())
            needs.indifferent = [p for p in needs.indifferent if p.strip().lower() not in gone]
            cls._add_open(vars(talk).setdefault("open_pair", []), second)
            return label, f" [{second}: not answered, still open]"
        if paired and len(needs.indifferent) + vars(needs).get("pinned_heard", 0) == freed_before:
            talk.no_pairs = True
            pairs = vars(talk).setdefault("open_pair", [])
            for part in (p.strip() for p in label[0].split(";")):
                if part:
                    cls._add_open(pairs, part)
            return (label[0].replace(";", " and") + " [not answered]", label[1]), note
        if split_second:
            found_text = " ".join(needs.uncovered[found_before:]).lower()
            pairs = vars(talk).setdefault("open_pair", [])
            hit = any(w in found_text for w in re.findall(r"[a-z]{4,}", split_second.split("(")[0].lower()))
            if not hit:
                cls._add_open(pairs, split_second)
        return label, note

    @classmethod
    def reply_back(cls, ctx, text_body: str) -> bool:
        if not ctx.role("message") or ctx.budget.msgs_sent >= ctx.ask_room() or ctx.moves_left() <= 4:
            return False
        if ctx.wire.spare() < CLOSING:
            return False
        if ctx.talk.asking and ctx.budget.msgs_sent >= ctx.ask_room() - 1 and ctx.moves_left() > 10 \
                and ctx.wire.spare() >= CLOSING + 40:
            return False
        proposal = cls.message_call(ctx, text_body, f"reply_{ctx.wire.tick}")
        if not proposal:
            return False
        ctx.talk.chat.append({"role": "assistant", "content": "I will answer the shopper's question.", "tool_calls": [proposal]})
        cls.post_replies(ctx, Checkout.dispatch(ctx, [proposal], scripted=True))
        return True

    @classmethod
    def post_replies(cls, ctx, responses: list[tuple[str, str]]) -> None:
        last = len(responses) - 1
        for pos, (cid, text_body) in enumerate(responses):
            if pos == last:
                text_body += "\n" + ctx.status_line()
            ctx.talk.chat.append({"role": "tool", "tool_call_id": cid, "content": text_body})
class Closing:
    NUMBER_WORDS = {word: n for n, word in enumerate("one two three four five six seven eight nine ten".split(), 1)}

    @classmethod
    def _spoken(cls, token: str, extra: dict[str, int] | None = None) -> int:
        if token.isdigit():
            return int(token)
        return {**cls.NUMBER_WORDS, **(extra or {})}[token]

    @classmethod
    def plain_args(cls, ctx, call_name: str, text_body: str, kind: str = "") -> dict[str, Any]:
        schema = ctx.shelf.shapes.get(call_name) or {}
        specs = schema.get("properties") or {}
        cue = ""
        if kind == "decline":
            cue = DECLINE_WORDS
        elif kind == "order":
            cue = ORDER_WORDS
        args: dict[str, Any] = {}
        for name in schema.get("required") or []:
            spec = specs.get(name) or {}
            kind_of = spec.get("type")
            if spec.get("enum"):
                matching = [v for v in spec["enum"] if cue and isinstance(v, str) and re.search(cue, v.lower())]
                args[name] = (matching or spec["enum"])[0]
            elif kind_of == "string" or "string" in (kind_of or []):
                args[name] = text_body[: spec.get("maxLength", len(text_body))]
            elif kind_of == "boolean":
                args[name] = True
            elif kind_of in ("integer", "number"):
                args[name] = max(1, spec.get("minimum", 1))
        if len(specs) != 1:
            return args
        name, spec = next(iter(specs.items()))
        if name in args or not isinstance(spec, dict):
            return args
        if spec.get("type") == "string" and "enum" not in spec:
            args[name] = text_body[: spec.get("maxLength", len(text_body))]
        return args

    @classmethod
    def scrub_decline(cls, ctx, call_args: dict[str, Any], call_name: str) -> dict[str, Any]:
        rule = ctx.needs.reading.get("decline_no_names")
        if rule is False:
            return call_args
        known = ctx.shelf.archive.ids
        mentions = False
        for piece in call_args.values():
            if not isinstance(piece, str):
                continue
            if any(tok and tok in known for tok in re.split(r"[\s.,;:!?()\[\]{}<>\"'`]+", piece)):
                mentions = True
                break
        if not (rule or mentions):
            return call_args
        specs = (ctx.shelf.shapes.get(call_name) or {}).get("properties") or {}
        cleaned = {}
        for name, piece in call_args.items():
            spec = specs.get(name) or {}
            if isinstance(piece, str) and "enum" not in spec:
                piece = SORRY[: spec.get("maxLength", len(SORRY))]
            cleaned[name] = piece
        return cleaned

    @classmethod
    def code_reading(cls, ctx) -> dict[str, Any]:
        text_body = ctx.request
        low = text_body.lower()
        reading: dict[str, Any] = {"code_reading": True, "requirements": [], "items": [], "quantity": 1, "rules": [],
                                   "soft_wants": [], "ask_first": [], "hints": [], "references": [], "priority_detail": ""}
        reading["price_cap"], reading["currency"] = cls.code_budget(ctx, text_body)
        reading["in_stock_only"] = bool(re.search(r"\bin[- ]stock\b|\bon hand\b|\bon the shelf\b|\bsold[- ]out\b|\bback[- ]?order|"
                                                  r"\bout of stock\b|\bavailable now\b", low))
        cheapest = re.search(r"\bcheapest\b|\blowest[- ](?:price|priced|cost)\b|\bleast expensive\b|"
                             r"\blowest\b[^.]{0,20}\bprice\b|\bsmallest (?:total|price)\b", low)
        reading["priority"] = "cheapest" if cheapest else "none"
        messages = cls.code_limit(ctx, low, r"messages?|questions?|notes?|repl(?:y|ies)|texts?")
        if messages is not None:
            reading["message_budget"] = messages
        steps = cls.code_limit(ctx, low, r"steps?|moves?|actions?|(?:tool )?calls?|turns?")
        if steps is not None and steps >= FIRST_ORDER_BY + 3:
            reading["step_budget"] = steps
        roles = cls.code_roles(ctx)
        reading["roles"] = roles
        items = cls.code_items(ctx, roles.get("search"))
        if len(items) < 2:
            if items:
                reading["product"], reading["quantity"] = items[0]["label"], items[0]["quantity"]
            else:
                reading["product"] = cls.code_label(ctx, roles.get("search"), PropKit.kind_words(text_body))
                reading["quantity"] = cls.code_count(ctx, low)
            reading["requirements"] = [f"{prop}: {value}" for prop, value in cls.code_needs(ctx, low)]
            return reading
        reading["items"] = [{key: row[key] for key in ("product", "quantity", "requirements")} for row in items]
        reading["product"] = " and ".join(row["product"] for row in items)
        reading["requirements"] = [req for row in items for req in row["requirements"]]
        return reading

    @classmethod
    def _budget_score(cls, sentence: str, m: Any, whole: int) -> int | None:
        after = sentence[m.end():]
        if re.match(r"\s*(?:x\b|×|%|:\s*\d|-\s*\d|/\s*\d|(?:[a-z-]+\s+){0,2}?(?:messages?|questions?|notes?|repl\w*|texts?|"
                    r"steps?|moves?|actions?|(?:tool )?calls?|turns?|units?|items?|pieces?|packs?|pairs?|lots?|sets?|of\b|times?)\b)", after):
            return None
        if not m.group(1) and re.search(r"[\d.,:/-]\s*$|\bx\s*$", sentence[:m.start()]):
            return None
        before = sentence[max(0, m.start() - 60): m.start()]
        score = whole
        if m.group(1) or re.search(r"\b(?:spend\w*|budget\w*|afford\w*|pay\w*|price[ds]?|costs?|cash|money)\b(?:\W+[\w']+){0,4}\W*$", before):
            score += 3
        if re.search(r"\b(?:under|below|no more than|at most|not more than|up to|within|max\w*|less than|cheaper than|"
                     r"not (?:more|over|above)|limit\w*|reach\w*|go to|to|at|cap\w*|past|exceed\w*|beyond|go over)\W+(?:[\w']+\W+)?$", before):
            score += 2
        if re.match(r"\s*(?:[a-z.()]+\s*){0,3}?(?:or (?:lower|less|under|below|cheaper)|at (?:the (?:very )?)?most|max\w*|tops\b|"
                    r",?\s*(?:and )?not (?:one|a) \w+ (?:over|more|above))", after):
            score += 2
        return score

    @classmethod
    def code_budget(cls, ctx, text_body: str) -> tuple[Any, Any]:
        best: tuple[int, float] = (0, 0.0)
        found: tuple[Any, Any] = (None, None)
        for sentence in re.split(r"(?<=[.!?;])\s+", str(text_body).lower()):
            whole = 1 if re.search(r"\b(?:order|total|whole|lot|sum|bill|altogether|overall|everything)\b", sentence) else 0
            for m in re.finditer(r"([$€£]\s*)?(\d[\d,]*(?:\.\d+)?)", sentence):
                score = cls._budget_score(sentence, m, whole)
                if score is None:
                    continue
                figure = float(m.group(2).replace(",", ""))
                if score < 2 or (score, figure) <= best:
                    continue
                best = (score, figure)
                unit = re.match(r"\s*((?:[a-z][a-z.]*\s*){0,2})", sentence[m.end():])
                words = []
                for w in (unit.group(1).split() if unit else []):
                    bare = w.strip(".")
                    if bare not in LINK_WORDS and len(bare) >= 2 and not re.search(r"\d", w):
                        words.append(bare)
                sign = (m.group(1) or "").strip()
                found = (int(figure) if figure.is_integer() else figure, " ".join(words) or sign or None)
        return found

    @classmethod
    def code_limit(cls, ctx, low: str, nouns: str) -> int | None:
        larger = {"twelve": 12, "fifteen": 15, "twenty": 20}
        counted = re.finditer(r"\b(\d+|one|two|three|four|five|six|seven|eight|nine|ten|twelve|fifteen|twenty)\s+(?:[a-z-]+\s+){0,2}?(?:"
                              + nouns + r")\b", low)
        for m in counted:
            before = low[max(0, m.start() - 60): m.start()]
            after = low[m.end(): m.end() + 40]
            cue_before = re.search(r"\b(?:no more than|at most|not more than|up to|only|max\w*|use|using|take|send|allow\w*|receive|get|total of|"
                                   r"limit\w*|inside|within|under|fewer than|less than|you've|you have|have|keep\w*|stick to|go past|past|"
                                   r"exceed\w*|beyond)\W+(?:[\w']+\W+){0,3}$", before)
            if cue_before or re.match(r"\W*(?:[\w']+\W+){0,3}?(?:in total|total|no more|at most|max\w*|tops|or fewer|or less|allowed|limit)\b", after):
                return cls._spoken(m.group(1), larger)
        return None

    @classmethod
    def code_count(cls, ctx, low: str) -> int:
        said = re.search(r"\b(\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s+(?:units?|of (?:them|those|these)|copies)\b", low)
        if said is None:
            return 1
        n = cls._spoken(said.group(1))
        return n if 1 <= n <= 99 else 1

    @classmethod
    def code_needs(cls, ctx, seg: str) -> list[tuple[str, str]]:
        verbs = r"(?:is|are|has to be|have to be|must be|should be|needs to be|need to be|can be)"
        stop = r"(?=\s*(?:[,.;:!?()]|$)|\s+(?:so|since|as|because|for|to|which|that|nothing|otherwise|but|when|while|if|then|though|please|and (?:the|its|their|a|an|one|two|three|\d))\b)"
        pattern = (r"(?:\b(?:where|whose)\s+(?:the\s+|its\s+|their\s+)?|(?:\b(?:and|with|but)\s+|[,;:]\s*|^|[.!?]\s+)(?:the|its|their)\s+)"
                   r"([a-z][a-z'&/ -]{1,32}?)\s+" + verbs + r"\s+(?:either\s+|just\s+)?([^,.;:!?()]{1,60}?)" + stop)
        found: list[tuple[str, str]] = []
        for m in re.finditer(pattern, seg):
            prop, value = m.group(1).strip(), m.group(2).strip()
            following = seg[m.end(): m.end() + 60]
            if RefKit.ref_text(value) or RefKit.ref_text(following) and re.search(r"\b(?:page|listing|shown|sold by)\b", following):
                continue
            if not value or len(value.split()) > 4 or len(prop.split()) > 4:
                continue
            if re.search(r"\b(?:i|you|we|it|he|she|they|my|your|our|there|this|that|which|what|who|me|us|them|one|ones|price|budget|cost|total|order|stock)\b", prop):
                continue
            if re.match(r"(?:a|an|the|my|your|our|his|her|their|me|us|for|to|in|on|at|of|see|go|get|be|have|only|too|very|so|not|no|what|how|it|all|"
                        r"mostly|always|never|just|really|still|also|from|about|like|one|ones)\b", value):
                continue
            found.append((prop, value))
        return found

    @classmethod
    def _shelf_choices(cls, ctx, tool: str | None) -> list[Any]:
        specs = (ctx.shelf.shapes.get(tool or "") or {}).get("properties")
        lists = []
        for spec in (specs.values() if isinstance(specs, dict) else []):
            if isinstance(spec, dict) and isinstance(spec.get("enum"), list) and len(spec["enum"]) >= 4:
                lists.append(spec["enum"])
        return max(lists, key=len) if lists else []

    @classmethod
    def code_items(cls, ctx, tool: str | None) -> list[dict[str, Any]]:
        labels = [str(label) for label in cls._shelf_choices(ctx, tool)]
        low = ctx.request.lower()
        toks = [(m.group(0).rstrip("s"), m.start(), m.end()) for m in re.finditer(r"[a-z]{3,}", low) if m.group(0) not in LINK_WORDS]
        words = [t[0] for t in toks]
        counted: dict[str, tuple[int, int, int]] = {}
        for label in labels:
            own = [w.rstrip("s") for w in re.findall(r"[a-z]{3,}", label.lower()) if w not in LINK_WORDS]
            if not own:
                continue
            span = len(own)
            for i in range(len(words) - span + 1):
                if words[i: i + span] != own:
                    continue
                start, end = toks[i][1], toks[i + span - 1][2]
                if len(re.findall(r"[a-z0-9]+", low[start:end])) > 2 * span:
                    continue
                count = re.search(r"\b(\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s*(?:x\s+|×\s*)?(?:(?:of|[a-z'&-]{3,})\s+){0,4}$",
                                  low[max(0, start - 56): start])
                if not count or label in counted:
                    continue
                n = cls._spoken(count.group(1))
                if 1 <= n <= 99:
                    counted[label] = (start, end, n)
                break
        ordered = sorted(counted.items(), key=lambda kv: kv[1][0])
        items = []
        last = len(ordered) - 1
        for j, (label, (start, end, n)) in enumerate(ordered):
            if j < last:
                seg = low[end: ordered[j + 1][1][0]]
            else:
                seg = re.split(r"(?<=[.!?])\s", low[end: len(low)])[0]
            name = label.lower()
            needs = [f"{prop} ({name}): {value}" for prop, value in cls.code_needs(ctx, seg)]
            items.append({"product": name, "label": label, "quantity": n, "requirements": needs})
        return items

    @classmethod
    def code_roles(cls, ctx):
        shapes = ctx.shelf.shapes
        said = {}
        for tdef in ctx.shelf.toolset:
            fn = tdef["function"]
            spaced = re.sub(r"([a-z])([A-Z])", r"\1 \2", fn["name"]).replace("_", " ").lower()
            said[fn["name"]] = spaced + " | " + str(fn.get("description") or "").lower()
        roles = {}
        for name in said:
            for key in cls._schema_needed(shapes, name):
                spec = cls._schema_props(shapes, name)[key]
                choices = [str(v).lower() for v in (spec.get("enum") or [])] if isinstance(spec, dict) else []
                if not any(re.search(ORDER_WORDS, v) for v in choices):
                    continue
                if any(re.search(DECLINE_WORDS, v) for v in choices):
                    roles.setdefault("order", name)
                    roles.setdefault("decline", name)
        for name, text in said.items():
            if name in roles.values():
                continue
            head = text.split(" | ")[0]
            if "decline" not in roles and re.search(DECLINE_WORDS, head):
                roles["decline"] = name
            elif "order" not in roles and re.search(ORDER_WORDS, head):
                roles["order"] = name
        for name, text in said.items():
            if name in roles.values() or cls._schema_needed(shapes, name):
                continue
            final = re.search(r"\bfinal\b", text)
            if "decline" not in roles and final and re.search(DECLINE_WORDS, text):
                roles["decline"] = name
            elif "order" not in roles and re.search(r"\bfinal\b", text) and re.search(ORDER_WORDS, text):
                roles["order"] = name
        for name, text in said.items():
            if name in roles.values() or not re.search(r"\b(?:cart|basket|bag|trolley)\b", text):
                continue
            listy = any(isinstance(spec, dict) and spec.get("type") == "array" for spec in cls._schema_props(shapes, name).values())
            if cls._schema_needed(shapes, name) or listy:
                roles.setdefault("cart", name)
            elif not cls._schema_needed(shapes, name):
                roles.setdefault("cart_view", name)
        cls._cue_roles(shapes, said, roles)
        return roles

    @staticmethod
    def _schema_props(shapes, name):
        found = (shapes.get(name) or {}).get("properties")
        return found if isinstance(found, dict) else {}

    @classmethod
    def _schema_needed(cls, shapes, name):
        found = (shapes.get(name) or {}).get("required")
        return [key for key in found if key in cls._schema_props(shapes, name)] if isinstance(found, list) else []

    @classmethod
    def _schema_texts(cls, shapes, name):
        return [key for key, spec in cls._schema_props(shapes, name).items()
                if isinstance(spec, dict) and spec.get("type") == "string" and not spec.get("enum")]

    @classmethod
    def _cue_roles(cls, shapes, said, roles):
        person = r"\b(?:shoppers?|customers?|buyers?|clients?|users?)\b"
        speak = r"\b(?:send\w*|repl\w*|notes?|tell\w*|messag\w*|chat\w*|writ\w*|talk\w*)\b"
        cues = (("search", r"\b(?:search\w*|find\w*|look\w*|locat\w*|hunt\w*|brows\w*|quer\w*)\b", False),
                ("message", speak, True),
                ("details", r"\b(?:details?|records?|spec\w*|propert\w*|attribut\w*|facts?|full)\b", True),
                ("view", r"\b(?:offer|price|stock|availab\w*)\b", True))
        for role, cue, single_text in cues:
            for name, text in said.items():
                if role in roles or name in roles.values() or not re.search(cue, text):
                    continue
                if role == "search" and re.search(person, text) and re.search(speak, text):
                    continue
                if role == "message" and not re.search(person, text) and not re.search(r"\b(?:messag\w*|chat\w*)\b", text):
                    continue
                if single_text:
                    needed = cls._schema_needed(shapes, name)
                    if len(needed) != 1 or needed[0] not in cls._schema_texts(shapes, name):
                        continue
                elif not cls._schema_texts(shapes, name) and not any(
                        isinstance(spec, dict) and spec.get("enum") for spec in cls._schema_props(shapes, name).values()):
                    continue
                roles[role] = name

    @classmethod
    def code_label(cls, ctx, tool: str | None, want: set[str]) -> str:
        chosen, top = "", (0, 0.0)
        for label in cls._shelf_choices(ctx, tool):
            words = PropKit.kind_words(label)
            hit = len(words & want)
            if not hit:
                continue
            mark = (hit, hit / max(1, len(words)))
            if mark > top:
                chosen, top = str(label), mark
        return chosen

    @classmethod
    def code_values(cls, ctx, k: int | None) -> list[tuple[str, str]]:
        several = ctx.needs.items()
        found = []
        for req in ctx.needs.props_left():
            head, _, value = str(req).partition(":")
            if several and Asking.product_of(ctx, head) not in (k, None):
                continue
            bare = re.sub(r"\((?:stated|revealed[^)]*)\)", "", value).strip()
            short = len(re.findall(r"[a-z0-9]+", bare.lower())) <= 5 and not re.search(EXCLUDE_WORDS, bare.lower())
            if not (ValueKit.positive_value(value, head) or short) or not bare or RefKit.ref_text(bare):
                continue
            found.append((re.split(r"\s*\(", head)[0].strip(), bare))
        return found

    @classmethod
    def _query_words(cls, ctx, k: int | None, product: str, label: str) -> tuple[list[str], list[str]]:
        names = [w for w in re.findall(r"[a-z0-9]+", product.lower()) if len(w) > 2 and w not in LINK_WORDS]
        kind = names[-1] if names else ""
        firm = [value for _, value in cls.code_values(ctx, k)
                if not re.search(r"\bor\b|/|,|\b(?:" + VALUE_LINKS + r")\b", value.lower())]
        queries: list[str] = []
        if firm:
            value_words = [w for w in re.findall(r"[a-z0-9]+", firm[0].lower()) if w not in LINK_WORDS]
            if value_words:
                queries.append(" ".join(dict.fromkeys(value_words + [kind])))
                if len(value_words) >= 2:
                    queries.append(" ".join(dict.fromkeys([max(value_words, key=len), kind])))
        if not queries:
            if names:
                queries.append(" ".join(dict.fromkeys(names[:1] + names[-1:])))
            else:
                queries.append(" ".join(sorted(PropKit.kind_words(label))[:2]))
        return queries, names

    @classmethod
    def code_queries(cls, ctx, k):
        tool = ctx.role("search")
        schema = ctx.shelf.shapes.get(tool or "") or {}
        props = schema.get("properties") if isinstance(schema.get("properties"), dict) else {}
        needed = [key for key in schema.get("required") or [] if key in props]
        rows = ctx.needs.items()
        product = str(rows[k]["product"]) if rows and k is not None and k < len(rows) else str(ctx.needs.reading.get("product") or "")
        label = cls.code_label(ctx, tool, PropKit.kind_words(product) or PropKit.kind_words(ctx.request))
        told = ""
        for tdef in ctx.shelf.toolset:
            if tdef["function"]["name"] == tool:
                told = str(tdef["function"].get("description") or "")
                break
        clauses = [part.lower() for part in re.split(r"[.;]", told)]
        text_key = cls._text_key(props, Reading.token_key(ctx))
        enums = {}
        for key, spec in props.items():
            if isinstance(spec, dict) and isinstance(spec.get("enum"), list) and spec["enum"]:
                enums[key] = spec["enum"]
        shelf_key = max((key for key in enums if len(enums[key]) >= 4), key=lambda key: len(enums[key]), default=None)
        modes = [key for key in enums if key != shelf_key and len(enums[key]) <= 4]
        mode_key = next((key for key in modes if key in needed), modes[0] if modes else None)
        cap_key = cls._cap_key(props, clauses)
        cap = ctx.needs.spend_cap
        counts = ctx.needs.counts()
        qty = (counts[k] if k is not None and k < len(ctx.needs.counts()) else ctx.needs.quantity()) or 1
        bound = {}
        if cap_key and TextKit.is_number(cap) and cap > 0:
            bound[cap_key] = round(cap / qty, 2) if props[cap_key].get("type") != "integer" else int(cap // qty)
        cap_len = props.get(text_key, {}).get("maxLength") if text_key and isinstance(props.get(text_key), dict) else None
        queries, _ = cls._query_words(ctx, k, product, label)
        if TextKit.is_number(cap_len):
            queries = [cls._fit_length(query, cap_len) for query in queries]

        def moded(arg, last):
            if not mode_key:
                return {}
            return {mode_key: cls._mode_choice(enums.get(mode_key or "", []), clauses, arg, last)}
        plans = []
        if text_key and queries[0]:
            plans.append({text_key: queries[0], **moded(text_key, False), **bound})
        if label and shelf_key:
            plans.append({shelf_key: label, **moded(shelf_key, True), **bound})
        if text_key and len(queries) > 1:
            plans.append({text_key: queries[1], **moded(text_key, False), **bound})
        built = []
        for call_args in plans:
            cls._fill_needed(call_args, needed, props, queries, label)
            fixed, faults = ArgCheck.execute(call_args, schema)
            if faults and bound:
                fixed, faults = ArgCheck.execute({key: val for key, val in call_args.items() if key not in bound}, schema)
            if faults or fixed in built:
                continue
            built.append(fixed)
        return built

    @staticmethod
    def _text_key(props, token):
        texts = []
        for key, spec in props.items():
            if isinstance(spec, dict) and spec.get("type") == "string" and not spec.get("enum") and key != token:
                texts.append(key)
        for key in texts:
            if re.search(r"^q$|query|text|keyword|needle|term|word|phrase|string|search", key.lower()):
                return key
        return texts[0] if texts else None

    @staticmethod
    def _mode_choice(choices, clauses, arg, last):
        for choice in choices:
            if any(re.search(r"\b" + re.escape(str(choice).lower()) + r"\b", part) and re.search(r"\b" + re.escape(arg.lower()) + r"\b", part)
                   for part in clauses):
                return choice
        return choices[-1] if last else choices[0]

    @staticmethod
    def _cap_key(props, clauses):
        for key, spec in props.items():
            if not isinstance(spec, dict) or spec.get("type") not in ("integer", "number") or spec.get("enum"):
                continue
            lowered = key.lower()
            priced = bool(re.search(r"ceil|max|price|cap$|budget|upper|most", lowered)) or any(
                lowered in part and re.search(r"\bpric|\bcost", part) for part in clauses)
            if priced and not re.search(r"page|first|offset|skip|count|size|num", lowered):
                return key
        return None

    @staticmethod
    def _fit_length(query, cap_len):
        while " " in query and len(query) > cap_len:
            query = query.rsplit(" ", 1)[0]
        return query[: int(cap_len)]

    @staticmethod
    def _fill_needed(call_args, needed, props, queries, label):
        for key in needed:
            if key in call_args:
                continue
            spec = props[key] if isinstance(props[key], dict) else {}
            if spec.get("enum"):
                call_args[key] = spec["enum"][0]
            elif spec.get("type") in ("integer", "number"):
                call_args[key] = spec.get("default", spec.get("minimum", 1))
            elif spec.get("type") == "string":
                call_args[key] = queries[0] or label

    @classmethod
    def code_rows(cls, ctx) -> dict[str, dict[str, Any]]:
        table: dict[str, dict[str, Any]] = {}
        known = ctx.shelf.archive.ids

        def visit(d: dict[str, Any]) -> None:
            ident = Reading.row_mark(ctx, d)
            if ident not in known:
                return
            info = table.setdefault(ident, {"shelf": "", "title": "", "price": None, "stock": None, "words": set(), "fields": {}})
            info["shelf"] = Reading.shelf_of(ctx, d) or info["shelf"]
            title = ""
            for key, val in d.items():
                if isinstance(val, str) and re.search(r"title|name", str(key).lower()) and "id" not in str(key).lower():
                    title = str(val)
                    break
            info["title"] = title or info["title"]
            for key, val in d.items():
                name = str(key).lower()
                if TextKit.is_number(val) and not isinstance(val, bool) and re.search(r"price|cost|amount", name) \
                        and not re.search(r"count|qty|quant|total|units|line|due|payable|sum", name):
                    info["price"] = float(val)
                    break
            for key, val in d.items():
                state = Reading.key_stock(ctx, key, val)
                if state is not None:
                    info["stock"] = state
            pieces = [str(v) for v in d.values() if isinstance(v, (str, int, float)) and not isinstance(v, bool)]
            for v in d.values():
                if isinstance(v, dict):
                    pieces.extend(str(x) for x in v.values() if isinstance(x, (str, int, float)))
            info["words"] |= PropKit.kind_words(" ".join(pieces))
            for v in d.values():
                if not isinstance(v, dict):
                    continue
                info["fields"].update({str(key): str(x) for key, x in v.items()
                                       if isinstance(x, (str, int, float)) and not isinstance(x, bool)})
        for rec in ctx.shelf.archive.items:
            TextKit.walk_dicts(rec, visit)
        return table

    @classmethod
    def code_shows(cls, ctx, row: dict[str, Any], prop: str, value: str, hit_words: set[str], read: bool) -> bool:
        named = [val for key, val in row["fields"].items() if PropKit.same_prop(key, prop)]
        if named:
            return any(cls.code_seen(ctx, value, val) for val in named)
        if cls.code_seen(ctx, value, row.get("title") or ""):
            return True
        if read:
            return False
        own = PropKit.kind_words(value)
        return bool(own) and own <= hit_words

    @classmethod
    def code_seen(cls, ctx, value: str, text: str) -> bool:
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

    @classmethod
    def _value_rank(cls, ctx, row: dict[str, Any], prop: str, value: str) -> tuple[int, int]:
        want = set()
        for alt in re.split(r"\s*(?:\bor\b|/)\s*", value.lower()):
            if ValueKit.ch_words(alt):
                want.add(" ".join(ValueKit.ch_words(alt)))
        best = (2, 3)
        for key, val in row["fields"].items():
            if not PropKit.same_prop(key, prop) or not cls.code_seen(ctx, value, val):
                continue
            if " ".join(ValueKit.ch_words(val)) in want:
                tier = 0
            elif any(" ".join(ValueKit.ch_words(p)) in want for p in ValueKit.ch_parts(val)):
                tier = 1
            else:
                tier = 3
            best = min(best, (0 if PropKit.prop_key(key) == PropKit.prop_key(prop) else 1, tier))
        if best == (2, 3) and cls.code_seen(ctx, value, row.get("title") or ""):
            best = (2, 2)
        return best

    @classmethod
    def _audited(cls, ctx, live: set[str]) -> set[str]:
        cart = ctx.cart
        reviewed = vars(cart).get("audit_ids")
        if vars(cart).get("audit_done") and (reviewed is None or len(ctx.shelf.archive.read) <= vars(cart).get("audit_reads", -1)):
            reviewed = set(ctx.shelf.archive.read)
        answered = False
        for entry in reversed(ctx.log):
            if isinstance(entry, dict) and entry.get("kind") == "audit":
                answered = isinstance(entry.get("result"), dict)
                break
        if not answered:
            return set()
        return set(reviewed or set()) - set(vars(cart).get("audit_fits") or []) - live

    @classmethod
    def code_picks(cls, ctx, avoid: set[str] | None = None, keep: set[str] | None = None) -> list[tuple[str, int]]:
        records = ctx.shelf.archive
        rows = ctx.needs.items()
        counts = ctx.needs.counts()
        if rows:
            homes = Lookup.home_shelves(ctx)
        else:
            homes = [set(Reading.shelf_labels(ctx)) or (Lookup.home_shelves(ctx) or [set()])[0]]
        info = cls.code_rows(ctx)
        ver = str(ctx.needs.version)
        failed = {key.split("|")[0] for key, val in ctx.cart.rulings.items() if val and key.split("|")[1:2] == [ver]}
        live = {key.rpartition("|")[0] for key in ctx.cart.passed if key.rpartition("|")[2] == ver}
        audited = cls._audited(ctx, live)
        own_all = bool(audited) and Lookup.code_judges_all(ctx)
        cap = float(ctx.needs.spend_cap) if TextKit.is_number(ctx.needs.spend_cap) else None
        asked = PropKit.kind_words(ctx.request)
        hits = vars(ctx.shelf).get("code_hits") or {}
        cheapest = ctx.needs.priority() == "cheapest"
        avoid, keep = set(avoid or ()), set(keep or ())
        avoid |= Lookup.dead_lines(ctx)
        avoid |= Referencing.ref_pointers(ctx)
        scope = dict(failed=failed, avoid=avoid, keep=keep, live=live, audited=audited, own_all=own_all, cap=cap,
                     asked=asked, hits=hits, cheapest=cheapest)
        pools = []
        for k, home in enumerate(homes):
            slot = k if rows else None
            want = PropKit.kind_words(rows[k]["product"] if rows else ctx.needs.reading.get("product") or "")
            vals, judged = set(), []
            for prop, bare in cls.code_values(ctx, slot):
                vals |= PropKit.kind_words(bare)
                if re.search(r"\b(?:" + VALUE_LINKS + r")\b|\d\s*(?:-|to)\s*\d|[+<>]|\b(?:about|around|approx\w*|roughly)\b", bare.lower()):
                    continue
                judged.append((prop, bare))
            apart = [(ref["prop"], [str(v) for v in [ref.get("value")] + list(ref.get("goes_with") or []) if str(v or "").strip()])
                     for ref in Lookup.differ_done(ctx, slot)]
            need = dict(qty=counts[k] if k < len(counts) else 1, want=want, vals=vals, judged=judged, apart=apart, slot=slot)
            pool = []
            for ident, row in info.items():
                entry = cls._pool_entry(ctx, ident, row, home, need, scope)
                if entry is not None:
                    pool.append(entry)
            if not pool:
                return []
            pools.append(sorted(pool))
        return cls._settle_picks(ctx, pools, counts, cap)

    @classmethod
    def _pool_entry(cls, ctx, ident, row, home, need, scope):
        if ident in scope["failed"] or ident in ctx.cart.refused_lines or ident in scope["avoid"] or row["stock"] is False:
            return None
        if not home:
            if not need["want"] & row["words"]:
                return None
        elif row["shelf"] not in home:
            return None
        judged, apart = need["judged"], need["apart"]
        if any(cls.code_clash(ctx, row["fields"], prop, value) for prop, value in judged):
            return None
        if Referencing.ref_excluded(ctx, row["fields"], need["slot"]):
            return None
        records = ctx.shelf.archive
        cap, hits = scope["cap"], scope["hits"]
        cost = records.price(ident) if row["price"] is None else row["price"]
        if cap is not None and TextKit.is_number(cost) and cost * need["qty"] > cap:
            return None
        was_read = ident in records.read
        fits = all(cls.code_shows(ctx, row, prop, value, hits.get(ident) or set(), was_read) for prop, value in judged)
        if fits:
            fits = all(any(PropKit.same_prop(key, prop) and str(val).strip() and not any(ValueKit.ch_writes(v, val) for v in bad)
                           for key, val in row["fields"].items()) for prop, bad in apart)
        shows = [cls._value_rank(ctx, row, prop, value) for prop, value in judged] if was_read else []
        if ident in scope["audited"]:
            trusted = scope["own_all"] and fits and (judged or apart) and all(max(s) <= 1 for s in shows)
            if not trusted:
                return None
        words = row["words"]
        score = 3 * len(need["vals"] & words) + 2 * len(need["want"] & words) + len(scope["asked"] & words) \
            + 4 * len(need["vals"] & (hits.get(ident) or set()))
        price = cost if TextKit.is_number(cost) else float("inf")
        quality = (sum(s[0] for s in shows), sum(s[1] for s in shows)) if shows else (9, 9)
        return (0 if ident in scope["keep"] else 1, 0 if ident in scope["live"] else 1, not fits,
                cap is not None and not TextKit.is_number(cost), price if scope["cheapest"] else 0.0, 0 if was_read else 1,
                row["stock"] is not True, (quality, -score), price, ident)

    @classmethod
    def _settle_picks(cls, ctx, pools: list[list[tuple]], counts: list[int], cap: float | None) -> list[tuple[str, int]]:
        records = ctx.shelf.archive
        orders = ((lambda e: e), (lambda e: (e[2], e[3], e[8], e[-1])))
        for order in orders:
            picks: list[tuple[str, int]] = []
            chosen = []
            total = 0.0
            for k, pool in enumerate(pools):
                entry = None
                for cand in sorted(pool, key=order):
                    if any(cand[-1] == p for p, _ in picks):
                        continue
                    if cand[-1] in records.read and Judging.field_view(ctx, cand[-1])["fails"]:
                        continue
                    entry = cand
                    break
                if entry is None:
                    return []
                picks.append((entry[-1], counts[k] if k < len(counts) else 1))
                chosen.append(entry)
                unit_price = entry[8] if entry[8] != float("inf") else 0.0
                total += unit_price * picks[-1][1]
            if cap is None or total <= cap:
                ctx.cart.code_fit = {e[-1]: not e[2] for e in chosen}
                return picks
        return []

    @classmethod
    def code_clash(cls, ctx, fields: dict[str, str], prop: str, value: str) -> bool:
        named = [val for key, val in fields.items() if PropKit.same_prop(key, prop)]
        if not named:
            return False
        return not any(ValueKit.value_seen(value, val) for val in named)

    @classmethod
    def cart_form(cls, ctx) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], str | None, list[str]]:
        schema = ctx.shelf.shapes.get(ctx.role("cart") or "") or {}
        props = schema.get("properties") if isinstance(schema.get("properties"), dict) else {}
        book = ctx.cart.trolley
        known = TextKit.json_object(book.put_form) if book.put_form else None

        def words(v: Any) -> set[str]:
            return set(re.findall(r"[a-z]+", re.sub(r"([a-z])([A-Z])", r"\1 \2", str(v)).lower()))
        put: dict[str, Any] = {}
        drop: dict[str, Any] = {}
        for key in schema.get("required") or []:
            spec = props.get(key) if isinstance(props.get(key), dict) else {}
            choices = spec.get("enum") if isinstance(spec.get("enum"), list) else []
            if not choices:
                continue
            adds = [v for v in choices if words(v) & PUT_WORDS and not words(v) & DROP_WORDS]
            takes = [v for v in choices if words(v) & DROP_WORDS]
            if isinstance(known, dict) and known.get(key) in choices:
                put[key] = known[key]
            else:
                put[key] = adds[0] if adds else choices[0]
            others = [v for v in choices if v != put[key]]
            drop[key] = next((v for v in takes if v != put[key]), others[0] if len(choices) == 2 else None)
        texts = [key for key, spec in props.items() if isinstance(spec, dict) and spec.get("type") == "string" and not spec.get("enum")]
        id_key = next((key for key in texts if re.search(r"id|item|product|listing|sku|target|which|number", key.lower())),
                      texts[0] if texts else None)
        units = [key for key, spec in props.items()
                 if isinstance(spec, dict) and spec.get("type") in ("integer", "number") and not spec.get("enum")]
        return schema, put, drop, id_key, units

    @classmethod
    def cart_args(cls, ctx, picks: list[tuple[str, int]]) -> list[dict[str, Any]]:
        schema, base, _, id_key, units = cls.cart_form(ctx)
        layout = ctx.cart.trolley.line_spec()
        if layout:
            arr, id_field, count_field = layout
            calls = [{**base, arr: [{id_field: ident, count_field: qty} for ident, qty in picks]}]
        elif not id_key:
            return []
        else:
            calls = []
            for ident, qty in picks:
                call_args = {**base, id_key: ident}
                if units:
                    call_args[units[0]] = qty
                calls.append(call_args)
        checked = []
        for call_args in calls:
            fixed, faults = ArgCheck.execute(call_args, schema)
            if faults:
                return []
            checked.append(fixed)
        return checked

    @classmethod
    def cart_drops(cls, ctx, lines: list[str]) -> list[dict[str, Any]] | None:
        book = ctx.cart.trolley
        if not lines or book.line_spec():
            return []
        schema, put, drop, id_key, units = cls.cart_form(ctx)
        props = schema.get("properties") if isinstance(schema.get("properties"), dict) else {}
        if not id_key:
            return None
        calls = []
        for ident in lines:
            if put:
                if any(drop.get(key) is None for key in put):
                    return None
                call_args = {**drop, id_key: ident}
                if units:
                    call_args[units[0]] = max(1, int(book.held(ident) or 1))
            else:
                if not units:
                    return None
                floor = (props.get(units[0]) or {}).get("minimum")
                if TextKit.is_number(floor) and props[units[0]]["minimum"] > 0:
                    return None
                call_args = {id_key: ident, units[0]: 0}
            fixed, faults = ArgCheck.execute(call_args, schema)
            if faults:
                return None
            calls.append(fixed)
        return calls

    @classmethod
    def code_replies(cls, ctx, n: int) -> list[Any]:
        entry = None
        for item in reversed(ctx.log):
            if isinstance(item, dict) and item.get("kind") == "env":
                entry = item
                break
        resp = (entry or {}).get("reply")
        if not isinstance(resp, dict):
            return []
        if isinstance(resp.get("observations"), list):
            return list(resp.get("observations"))[:n]
        return [resp.get("observation")][:n]

    @classmethod
    def code_fault(cls, ctx, obs: Any) -> bool:
        flagged = []

        def visit(d: dict[str, Any]) -> None:
            for key, val in d.items():
                if re.search(FAULT_KEYS, str(key).lower()) and val not in (None, False, "", [], {}):
                    flagged.append(d)
                    return
        TextKit.walk_dicts(obs, visit, 0, 2)
        return bool(flagged)

    @classmethod
    def _book_hits(cls, ctx, calls: list[dict[str, Any]], replies: list[Any], texts: list[str]) -> None:
        hits = vars(ctx.shelf).setdefault("code_hits", {})
        known = ctx.shelf.archive.ids
        for call_args, obs in zip(calls, replies):
            said = " ".join(str(call_args.get(key) or "") for key in texts)
            if not said.strip():
                continue
            words = PropKit.kind_words(said)

            def visit(d: dict[str, Any]) -> None:
                ident = Reading.row_mark(ctx, d)
                if ident in known:
                    hits.setdefault(ident, set()).update(words)
            TextKit.walk_dicts(obs, visit)

    @classmethod
    def code_search(cls, ctx, products: int, more: bool = False) -> None:
        search = ctx.role("search")
        specs = (ctx.shelf.shapes.get(search or "") or {}).get("properties") or {}
        texts = [key for key, spec in specs.items() if isinstance(spec, dict) and spec.get("type") == "string" and not spec.get("enum")]
        reserve = 2 + 2 * products
        if more:
            lead = list(vars(ctx.shelf).get("code_more") or [])
            calls = lead[: max(0, ctx.moves_left() - reserve - 2 * products)]
            rounds = 1
        else:
            several = bool(ctx.needs.items())
            lists = [cls.code_queries(ctx, k if several else None) for k in range(products)]
            calls = []
            deepest = max((len(q) for q in lists), default=0)
            for rank in range(deepest):
                layer = [q[rank] for q in lists if rank < len(q) and q[rank] not in calls]
                calls.extend(layer)
            calls = calls[: max(0, ctx.moves_left() - reserve)]
            lead = [q[0] for q in lists if q]
            rounds = 1 + CODE_PAGES + (2 if ctx.needs.priority() == "cheapest" else 0)
        if not calls:
            return
        ctx.shelf.code_searched = True
        for rnd in range(rounds):
            Checkout.send_chunk(ctx, [(f"code_search_{n}_{ctx.wire.tick}", search, call_args) for n, call_args in enumerate(calls)])
            replies = cls.code_replies(ctx, len(calls))
            if vars(ctx.shelf).get("word_kind") != "any":
                cls._book_hits(ctx, calls, replies, texts)
            following = []
            for call_args, obs in zip(calls, replies):
                if call_args not in lead:
                    continue
                nxt = cls.code_next(ctx, call_args, obs)
                if nxt:
                    following.append(nxt)
            ctx.shelf.code_more = following
            if rnd == rounds - 1:
                break
            picks = cls.code_picks(ctx)
            fit = vars(ctx.cart).get("code_fit") or {}
            if ctx.needs.priority() != "cheapest" and picks and all(fit.get(ident) for ident, _ in picks):
                break
            lead = following
            calls = following[: max(0, ctx.moves_left() - reserve - 3 * products)]
            if not calls:
                break

    @classmethod
    def code_next(cls, ctx, call_args: dict[str, Any], obs: Any) -> dict[str, Any] | None:
        spec = Reading.page_arg(ctx)
        if not spec:
            return None
        key, by_page, low, high = spec
        pages: list[dict[str, Any]] = []

        def visit(d: dict[str, Any]) -> None:
            for v in d.values():
                if isinstance(v, list) and v and all(isinstance(x, dict) for x in v):
                    pages.append(d)
                    return
        TextKit.walk_dicts(obs, visit, 0, 3)
        if not pages:
            return None
        page = pages[0]
        rows = max(len(v) for v in page.values() if isinstance(v, list))
        marks = [v for name, v in page.items() if re.search(r"next|onward|cursor|bookmark|token|continu|more", str(name).lower())]
        total = None
        for name, v in page.items():
            if TextKit.is_number(v) and not isinstance(v, bool) and re.search(r"count|total|matched|found|hits", str(name).lower()):
                total = v
                break
        if marks and all(m in (None, False, "", 0) for m in marks):
            return None
        if by_page == "token":
            tok = next((m for m in marks if isinstance(m, str) and m.strip()), None)
            if not tok:
                return None
            nxt_args = {**call_args, key: tok}
        else:
            cur = call_args.get(key, low)
            if not TextKit.is_number(cur):
                cur = low
            nxt = cur + 1 if by_page else cur + rows
            if high is not None and nxt > high:
                return None
            if not marks:
                shown = (cur - low + 1) * rows if by_page else cur + rows
                if total is None or shown >= total:
                    return None
            nxt_args = {**call_args, key: int(nxt) if float(nxt).is_integer() else nxt}
        fixed, faults = ArgCheck.execute(nxt_args, ctx.shelf.shapes.get(ctx.role("search") or "") or {})
        return None if faults else fixed

    @classmethod
    def code_read(cls, ctx, picks: list[tuple[str, int]], avoid: set[str], keep: set[str] | None) -> list[tuple[str, int]]:
        records = ctx.shelf.archive
        for _ in range(CODE_READS):
            unread = [ident for ident, _ in picks if ident not in records.read]
            batch = Lookup.batch_tool(ctx)
            grouped = bool(batch) and len(unread) >= 2
            reads = 1 if grouped else len(unread)
            if not unread or not (batch or ctx.role("details")):
                break
            if ctx.moves_left() < reads + 2 * len(picks) + 3 or ctx.wire.spare() <= 8:
                break
            if grouped:
                slab = [(f"code_read_{ctx.wire.tick}", batch[0], {batch[1]: unread[: batch[2]]})]
            else:
                details = ctx.role("details")
                slab = [(f"code_read_{n}_{ctx.wire.tick}", details, call_args)
                        for n, call_args in enumerate(Judging.details_call(ctx, ident) for ident in unread) if call_args]
            if not slab:
                break
            Checkout.send_chunk(ctx, slab)
            fresh = cls.code_picks(ctx, avoid, keep)
            if fresh == picks:
                break
            picks = fresh
        return picks

    @classmethod
    def _unseen_picks(cls, ctx, calls, replies, want: set[str], layout) -> tuple[set[str], Any]:
        book = ctx.cart.trolley
        bad: set[str] = set()
        shown = None
        for call_args, obs in zip(calls, replies):
            seen = book.view_units(obs)
            if seen is not None:
                shown = seen
            if not cls.code_fault(ctx, obs):
                continue
            named = [ident for ident in want if ident in TextKit.as_text(obs)]
            lines = [line for line, _ in book.lines(call_args) if line in want]
            bad.update(named or lines or (list(want) if layout else []))
        return bad, shown

    @classmethod
    def code_put(cls, ctx, picks: list[tuple[str, int]]) -> list[str] | None:
        book = ctx.cart.trolley
        cart = ctx.role("cart")
        want = {ident for ident, _ in picks}
        layout = book.line_spec()
        extra = [line for line in book.units if line not in want]
        stated = any(t.get("before_change") is True for t in ctx.needs.told)
        unknown = any(t.get("before_change") is None for t in ctx.needs.told)
        if extra and (stated or unknown):
            if not Checkout.code_asks_swap(ctx, extra, [ident for ident, _ in picks]) and stated:
                return None
        if layout:
            todo = list(picks)
        else:
            todo = [(ident, qty) for ident, qty in picks if abs(book.held(ident) - qty) > 1e-9]
        drops = cls.cart_drops(ctx, extra)
        puts = cls.cart_args(ctx, todo) if todo else []
        if drops is None or (todo and not puts):
            return None
        calls = drops + puts
        if not calls:
            return []
        if ctx.moves_left() < len(calls) + 1:
            return None
        Checkout.send_chunk(ctx, [(f"code_cart_{n}_{ctx.wire.tick}", cart, call_args) for n, call_args in enumerate(calls)])
        replies = cls.code_replies(ctx, len(calls))
        bad, shown = cls._unseen_picks(ctx, calls, replies, want, layout)
        if len(replies) < len(calls):
            return None
        viewer = ctx.role("cart_view")
        if shown is None and not bad and viewer and ctx.moves_left() >= 3:
            Checkout.send_chunk(ctx, [(f"code_view_{ctx.wire.tick}", ctx.role("cart_view"), {})])
            view = cls.code_replies(ctx, 1)
            shown = book.view_units(view[0]) if view else None
            if view and shown is None:
                bad.update(ident for ident in want if ident not in book.units)
        if shown is not None:
            bad.update(ident for ident in want if ident not in shown)
        for ident in bad:
            book.units.pop(ident, None)
        return sorted(bad)

    @classmethod
    def code_recheck(cls, ctx, lines: list[str]) -> tuple[bool, set[str]]:
        tool = Lookup.stock_tool(ctx) or ctx.role("view") or ctx.role("details")
        if not lines or not tool or ctx.moves_left() < len(lines) + 2 or ctx.wire.spare() < 8:
            return False, set()
        slab = []
        for n, ident in enumerate(lines):
            call_args = Lookup.lookup_call(ctx, ident, tool)
            if call_args:
                slab.append((f"code_check_{n}_{ctx.wire.tick}", tool, call_args))
        if not slab:
            return False, set()
        Checkout.send_chunk(ctx, slab)
        info = cls.code_rows(ctx)
        return True, {ident for ident in lines if (info.get(ident) or {}).get("stock") is False}

    @classmethod
    def code_news(cls, ctx, lines: list[str]) -> None:
        waiting = []
        for held_lines, _ in vars(ctx.needs).get("pending") or []:
            waiting.extend(held_lines)
        seen = vars(ctx.talk).setdefault("news_read", set())
        for line in list(lines) + waiting:
            if line in seen:
                continue
            seen.add(line)
            figure, _ = cls.code_budget(ctx, line)
            if TextKit.is_number(figure) and figure > 0:
                ctx.needs.spend_cap = figure
                ctx.log.append({"kind": "code_news", "budget": figure})

    @classmethod
    def _cart_round(cls, ctx, avoid: set[str], keep, search, products: int):
        picks = cls.code_picks(ctx, avoid, keep)
        fit = vars(ctx.cart).get("code_fit", {})
        unfit = not picks or not all(fit.get(ident) for ident, _ in picks)
        if search and not vars(ctx.shelf).get("code_searched") and ctx.moves_left() >= 4 and unfit:
            cls.code_search(ctx, products)
            picks = cls.code_picks(ctx, avoid, keep)
        elif search and vars(ctx.shelf).get("code_more") and ctx.moves_left() >= 6 + 2 * products \
                and (unfit or ctx.needs.priority() == "cheapest"):
            cls.code_search(ctx, products, True)
            picks = cls.code_picks(ctx, avoid, keep)
        return cls.code_read(ctx, picks, avoid, keep)

    @classmethod
    def code_cart(cls, ctx, avoid: set[str] | None = None, keep: set[str] | None = None) -> bool:
        cart, search = ctx.role("cart"), ctx.role("search")
        wire = ctx.wire
        if not cart or wire.done or wire.broke or ctx.moves_left() < 2 or wire.spare() < 4:
            return False
        wire.mute = True
        try:
            avoid = set(avoid or ())
            products = max(1, len(ctx.needs.items()))
            book = ctx.cart.trolley
            for _ in range(4):
                picks = cls._cart_round(ctx, avoid, keep, search, products)
                if not picks:
                    ctx.log.append({"kind": "code_finish", "picks": [], "result": "nothing to order"})
                    return False
                bad = cls.code_put(ctx, picks)
                if bad is None:
                    ctx.log.append({"kind": "code_finish", "picks": picks, "result": "no cart call"})
                    return False
                if not bad:
                    if ctx.moves_left() >= len(picks) + 3:
                        _, sold = cls.code_recheck(ctx, [ident for ident, _ in picks])
                    else:
                        sold = set()
                    bad = sorted(sold)
                if not bad:
                    ctx.log.append({"kind": "code_finish", "picks": picks, "cart": dict(book.units)})
                    return bool(book.units)
                avoid |= set(bad)
                ctx.log.append({"kind": "code_swap", "lines": bad})
            return False
        except Exception as err:
            ctx.log.append({"kind": "code_finish_error", "error": f"{type(err).__name__}: {err}"[:300]})
            return False

    @classmethod
    def _order_move(cls, ctx, tool: str) -> dict[str, Any]:
        return {"name": tool, "args": cls.plain_args(ctx, tool, "Order the cart.", "order")}

    @classmethod
    def _decline_move(cls, ctx, tool: str) -> dict[str, Any]:
        return {"name": tool, "args": cls.plain_args(ctx, tool, SORRY, "decline")}

    @classmethod
    def _over(cls, ctx) -> bool:
        return ctx.wire.done or ctx.wire.broke or ctx.moves_left() <= 0 or ctx.wire.spare() < 2

    @classmethod
    def _first_move(cls, ctx, turn: int, state: dict[str, Any]) -> dict[str, Any] | None:
        book = ctx.cart.trolley
        order_tool, decline_tool = state["order"], state["decline"]
        gone: set[str] = set()
        if order_tool and turn == 0 and book.units:
            try:
                ctx.wire.mute = True
                before = list(book.units)
                gone = Lookup.cart_gone(ctx, before)
                if gone:
                    state["avoid"] |= gone
                    state["keep"] = set(before) - gone
                    ctx.log.append({"kind": "code_dead", "lines": sorted(gone)})
            except Exception as err:
                ctx.log.append({"kind": "cart_gone_error", "error": f"{type(err).__name__}: {err}"[:300]})
                gone = set()
        if order_tool and turn == 0 and not gone and book.used \
                and (book.units or book.view_units(book.last_view) is None):
            return cls._order_move(ctx, order_tool)
        if order_tool and cls.code_cart(ctx, state["avoid"], state["keep"]):
            return cls._order_move(ctx, order_tool)
        if order_tool and turn > 0 and cls.again_ok(ctx, state["avoid"]):
            return cls._order_move(ctx, order_tool)
        if decline_tool:
            return cls._decline_move(ctx, decline_tool)
        return None

    @classmethod
    def _after_refusal(cls, ctx, move: dict[str, Any], resp: dict[str, Any], kind: Any, state: dict[str, Any]) -> dict[str, Any] | None:
        book = ctx.cart.trolley
        ctx.wire.mute = True
        lines = list(book.units)
        obs = resp.get("observation")
        Checkout.absorb(ctx, move["name"], move["args"], obs, resp, True)
        cls.code_news(ctx, Coverage.shopper_lines(ctx, move["name"], obs, True))
        if kind != "order" or not lines:
            return None
        read, sold = cls.code_recheck(ctx, lines)
        if not read:
            return None
        ctx.cart.refused_lines.difference_update(set(lines) - sold)
        info = cls.code_rows(ctx)
        total = 0.0
        for line in lines:
            total += ((info.get(line) or {}).get("price") or 0.0) * book.held(line)
        cap = ctx.needs.spend_cap
        over = TextKit.is_number(cap) and total > cap
        if not sold and not over:
            return cls._order_move(ctx, state["order"])
        state["avoid"] = set(sold)
        state["keep"] = None if over else set(lines) - sold
        return None

    @classmethod
    def finish_now(cls, ctx) -> None:
        if cls._over(ctx):
            return
        state: dict[str, Any] = {"order": ctx.role("order"), "decline": ctx.role("decline"), "avoid": set(), "keep": None}
        move: dict[str, Any] | None = None
        kind = None
        for turn in range(CODE_FINISHES):
            if cls._over(ctx):
                return
            if move is None:
                move = cls._first_move(ctx, turn, state)
                if move is None:
                    return
            kind = ctx.call_kind(move["name"], move["args"])
            try:
                Checkout.finish_steps(ctx, kind or "order")
            except Exception as err:
                ctx.log.append({"kind": "finish_steps_error", "error": f"{type(err).__name__}: {err}"[:300]})
            if ctx.wire.done or ctx.wire.broke or ctx.moves_left() <= 0:
                return
            resp = ctx.wire.act([move])
            ctx.log.append({"kind": "forced_finish", "action": move, "reply": resp})
            if not resp or resp.get("finished") or ctx.wire.done or turn == CODE_FINISHES - 1 or ctx.moves_left() <= 1:
                break
            move = cls._after_refusal(ctx, move, resp, kind, state)
        if ctx.wire.done or not resp or resp.get("finished") or not state["decline"] or kind == "decline" or ctx.moves_left() <= 0:
            return
        again = False
        try:
            again = bool(state["order"]) and kind == "order" and cls.again_ok(ctx, state["avoid"])
        except Exception as err:
            ctx.log.append({"kind": "again_error", "error": f"{type(err).__name__}: {err}"[:300]})
        move = cls._order_move(ctx, state["order"]) if again else cls._decline_move(ctx, state["decline"])
        ctx.log.append({"kind": "forced_finish", "action": move, "reply": ctx.wire.act([move])})

    @classmethod
    def again_ok(cls, ctx, avoid: set[str]) -> bool:
        units = set(ctx.cart.trolley.units)
        gone = set(avoid) | Lookup.dead_lines(ctx)
        if not units or units & gone:
            return False
        return bool(cls.code_picks(ctx, gone))

    @classmethod
    def catch_up(cls, ctx) -> None:
        try:
            wire = ctx.wire
            if wire.mute or wire.spare() < CLOSING:
                return
            notes = []
            if ctx.needs.reading.get("code_reading") and not vars(ctx.budget).get("reread") \
                    and wire.last_reply > vars(ctx.budget).get("read_at", wire.last_reply) and wire.spare() > CLOSING + 60:
                ctx.budget.reread = True
                msg = wire.query_model([{"role": "user", "content": ctx.budget.read_content}], READER, 3000)
                reading = TextKit.json_object((msg or {}).get("content") or "") or {}
                ctx.log.append({"kind": "re_read", "ok": isinstance(reading.get("roles"), dict)})
                if isinstance(reading.get("roles"), dict):
                    Intake.use_reading(ctx, reading, True)
                    notes.append("[plan updated] the models are reachable again and have read the request; this reading takes "
                                 "the place of the first one:\n" + ctx.needs.overview(ctx.budget.step_cap, ctx.budget.msg_cap))
            waiting = vars(ctx.needs).get("pending") or []
            if waiting and wire.last_reply > vars(ctx.needs).get("pending_at", 0.0) and wire.spare() >= CLOSING:
                ctx.needs.pending = []
                for held_lines, qtext in waiting[-3:]:
                    said = ctx.needs.hear(held_lines, qtext)
                    if said:
                        notes.append("[earlier shopper lines, read only now] " + said)
            if not notes:
                return
            text_body = "\n".join(notes) + "\n" + ctx.status_line()
            last = ctx.talk.chat[-1] if ctx.talk.chat else None
            if last and last.get("role") in ("user", "tool") and isinstance(last.get("content"), str):
                last["content"] += "\n" + text_body
            else:
                ctx.talk.chat.append({"role": "user", "content": text_body})
        except Exception as err:
            ctx.log.append({"kind": "catch_up_error", "error": f"{type(err).__name__}: {err}"[:300]})
class Lookup:
    PLAIN_VALUE = r"\b(?:" + VALUE_LINKS + r")\b|\d\s*(?:-|to)\s*\d|[+<>]|\b(?:about|around|approx\w*|roughly)\b"

    @classmethod
    def _seen_shelves(cls, ctx) -> set[str]:
        records = ctx.shelf.archive
        found: set[str] = set()

        def visit(d: dict[str, Any]) -> None:
            shelf = Reading.shelf_of(ctx, d)
            if not shelf:
                return
            if any(isinstance(v, str) and v in records.ids and "id" in str(key).lower() for key, v in d.items()):
                found.add(shelf)
        for rec in records.items:
            TextKit.walk_dicts(rec, visit)
        return found

    @classmethod
    def home_shelves(cls, ctx) -> list[set[str]]:
        records = ctx.shelf.archive
        memo = vars(ctx.shelf).get("home_memo")
        if memo and memo[0] == len(records.items):
            return [set(h) for h in memo[1]]
        homes: list[set[str]] = []
        try:
            rows = ctx.needs.items() or [{"product": str(ctx.needs.reading.get("product") or "")}]
            seen = cls._seen_shelves(ctx)
            kinds = [set(PropKit.kind_tokens(row["product"])) for row in rows]
            for k, own in enumerate(kinds):
                others = set().union(*(kinds[j] for j in range(len(kinds)) if j != k))
                best, top = set(), 0.0
                for shelf in sorted(seen):
                    score = PropKit.kind_score(own, set(PropKit.kind_tokens(shelf)), others)
                    if score > top:
                        best, top = {shelf}, score
                    elif score > 0 and score == top:
                        best.add(shelf)
                homes.append(best)
        except Exception:
            homes = [set() for _ in (ctx.needs.items() or [0])]
        ctx.shelf.home_memo = (len(records.items), [set(h) for h in homes])
        return homes

    @classmethod
    def only_pointed_reads(cls, ctx) -> bool:
        try:
            targets = [t for t in ctx.needs.reading.get("reference_targets") or [] if isinstance(t, dict)]
            kinds = [set(PropKit.kind_tokens(t.get("kind") or "")) for t in targets]
            kinds = [k for k in kinds if k]
            read = ctx.shelf.archive.shelves_read()
            if not kinds or not read:
                return False
            own = set(PropKit.kind_tokens(ctx.needs.reading.get("product") or ""))
            for shelf in read:
                words = set(PropKit.kind_tokens(shelf))
                if PropKit.kind_score(own, words, set()) > 0:
                    return False
                if not any(PropKit.kind_score(k, words, set()) > 0 for k in kinds):
                    return False
            return True
        except Exception:
            return False

    @classmethod
    def own_shelf(cls, ctx) -> str:
        try:
            if ctx.needs.items():
                return ""
            home = cls.home_shelves(ctx)[0]
            read = [shelf.strip().lower() for shelf in ctx.shelf.archive.shelves_read()]
            if not home or all(shelf in home for shelf in read):
                return ""
            for shelf in read:
                if shelf in home:
                    return shelf
            return sorted(home)[0]
        except Exception:
            return ""

    @classmethod
    def joined_phrase(cls, ctx, a: str, b: str, label: str) -> tuple[str, bool] | None:
        low = re.sub(r"\s+", " ", ctx.request.lower().replace("’", "'"))
        parts = [p for p in re.split(JOINERS, str(label).lower()) if p.strip()]
        if len(parts) == 2:
            first, second = PropKit.name_pattern(parts[0]), PropKit.name_pattern(parts[1])
            if first and second:
                hit = re.search(r"\b" + first + r"\s*(?:\band\b|&|\bor\b|/|\+)\s*" + second, low)
                if hit:
                    return hit.group(0).strip(), True
        toks_a, toks_b = PropKit.kind_tokens(a), PropKit.kind_tokens(b)
        if not toks_a or not toks_b or PropKit.same_kind(toks_a[-1], toks_b[-1]):
            return None
        for x, y in ((a, b), (b, a)):
            px, py = PropKit.name_pattern(x), PropKit.name_pattern(y)
            if not (px and py):
                continue
            for hit in re.finditer(r"\b(?:" + px + r")(?P<mid>[^.;:!?]{0,40}?)(?:" + py + r")", low):
                mid = hit.group("mid")
                if len(re.findall(r"[a-z0-9]+", mid)) > 3 or not re.search(r"\band\b|&|\bor\b|,|/|\+", mid):
                    continue
                if re.search(r"\d|\bx\b", mid):
                    continue
                rest = low[:hit.start()] + " | " + low[hit.end():]
                if not all(re.search(r"\b" + PropKit.name_pattern(z), rest) for z in (a, b)):
                    return hit.group(0).strip(), False
        return None

    @classmethod
    def product_question(cls, ctx, line: str, rows: list[dict[str, Any]]) -> bool:
        hit = re.match(r'-\s*(.*?): asked "(.*?)" -> ', str(line), re.S)
        if hit:
            label, question = hit.group(1), hit.group(2)
        else:
            label = question = str(line)
        kinds = [PropKit.kind_tokens(r["product"]) for r in rows]
        for tag in re.findall(r"\(([^()]*)\)", label):
            words = PropKit.kind_tokens(tag)
            if words and any(PropKit.kind_in(w, k) for w in words for k in kinds):
                return True
        low = question.lower().replace("’", "'")
        for r in rows:
            pattern = PropKit.name_pattern(r["product"])
            if pattern and re.search(r"\b" + PropKit.name_pattern(r["product"]), low):
                return True
        return False

    @classmethod
    def joined_clash(cls, ctx, a: dict[str, Any], b: dict[str, Any]) -> bool:
        def pairs(row: dict[str, Any]) -> list[tuple[str, str]]:
            found = []
            for req in row.get("requirements") or []:
                head, _, value = str(req).partition(":")
                found.append((re.split(r"\s*\(", head)[0].strip(), " ".join(re.findall(r"[a-z0-9]+", value.lower()))))
            return found
        for prop_a, val_a in pairs(a):
            for prop_b, val_b in pairs(b):
                if PropKit.same_prop(prop_a, prop_b) and val_a != val_b:
                    return True
        return False

    @classmethod
    def _sides_fit(cls, ti: list[str], tj: list[str], sides: list[list[str]]) -> bool:
        only_i = [w for w in ti if not PropKit.kind_in(w, tj)]
        only_j = [w for w in tj if not PropKit.kind_in(w, ti)]
        for x, y in ((sides[0], sides[1]), (sides[1], sides[0])):
            if not all(PropKit.kind_in(w, x) and not PropKit.kind_in(w, y) for w in only_i):
                continue
            if not all(PropKit.kind_in(w, y) and not PropKit.kind_in(w, x) for w in only_j):
                continue
            if any(PropKit.kind_in(w, x) for w in ti) and any(PropKit.kind_in(w, y) for w in tj):
                return True
        return False

    @classmethod
    def _merge_target(cls, ctx, seen: dict[str, str], rows: list[dict[str, Any]]):
        for low in sorted(seen):
            sides = [PropKit.kind_tokens(p) for p in re.split(JOINERS, low) if p.strip()]
            if len(sides) != 2 or not sides[0] or not sides[1]:
                continue
            whole = sides[0] + sides[1]
            for i in range(len(rows)):
                for j in range(i + 1, len(rows)):
                    ti, tj = PropKit.kind_tokens(rows[i]["product"]), PropKit.kind_tokens(rows[j]["product"])
                    if not ti or not tj:
                        continue
                    if PropKit.kinds_within(ti, tj) and PropKit.kinds_within(tj, ti):
                        continue
                    if not PropKit.kinds_within(ti + tj, whole) or rows[i]["quantity"] != rows[j]["quantity"]:
                        continue
                    if not cls._sides_fit(ti, tj, sides) or cls.joined_clash(ctx, rows[i], rows[j]):
                        continue
                    got = cls.joined_phrase(ctx, rows[i]["product"], rows[j]["product"], seen[low])
                    if got:
                        return i, j, seen[low], got[0], got[1]
        return None

    @classmethod
    def merge_joined(cls, ctx, obs: Any) -> str:
        seen = vars(ctx.shelf).setdefault("joined_seen", {})

        def visit(d: dict[str, Any]) -> None:
            if not any(isinstance(v, str) and "id" in str(k).lower() and re.fullmatch(ID_LIKE, v) for k, v in d.items()):
                return
            label = ""
            for key, val in d.items():
                if isinstance(val, str) and 0 < len(val.strip()) <= 60 and re.search(r"categor|shelf|department|aisle", str(key).lower()) \
                        and "id" not in str(key).lower():
                    label = val.strip()
                    break
            if label and len([p for p in re.split(JOINERS, label.lower()) if p.strip()]) == 2:
                seen.setdefault(label.lower(), label)
        TextKit.walk_dicts(obs, visit)
        book = ctx.cart.trolley
        remarks: list[str] = []
        for _ in range(2):
            rows = ctx.needs.items()
            if len(rows) < 2 or not seen or book.used or book.units:
                break
            if ctx.needs.uncovered or ctx.needs.indifferent:
                break
            if any(cls.product_question(ctx, entry, rows) for entry in ctx.talk.questions_log):
                break
            found = cls._merge_target(ctx, seen, rows)
            if not found:
                break
            remarks.append(cls.merge_items(ctx, *found))
        return " ".join(r for r in remarks if r)

    @staticmethod
    def _distinct(items: list[Any]) -> list[Any]:
        kept: list[Any] = []
        for x in items:
            if x not in kept:
                kept.append(x)
        return kept

    @staticmethod
    def _owned_tag(whole, pair, others, tag):
        toks = PropKit.kind_tokens(tag)
        if not toks or not PropKit.kinds_within(toks, whole) or not any(PropKit.kind_in(w, pair) for w in toks):
            return False
        return not any(PropKit.kinds_within(toks, o) and PropKit.kinds_within(o, toks) for o in others)

    @staticmethod
    def _retag(owned, tag_now, text):
        if not isinstance(text, str):
            return text
        head, sep, value = text.partition(":")

        def swap(hit):
            if owned(hit.group(1)):
                return f" ({tag_now})" if tag_now else ""
            return hit.group(0)
        return re.sub(r"\s*\(([^()]*)\)", swap, head) + sep + value

    @classmethod
    def merge_items(cls, ctx, i: int, j: int, label: str, phrase: str, in_order: bool = True) -> str:
        reading = ctx.needs.reading
        rows = ctx.needs.items()
        raw = [r for r in reading.get("items") or [] if isinstance(r, dict) and str(r.get("product") or "").strip()]
        if len(raw) != len(rows) or not (0 <= i < j < len(rows)):
            return ""
        ti, tj = PropKit.kind_tokens(rows[i]["product"]), PropKit.kind_tokens(rows[j]["product"])
        whole = PropKit.kind_tokens(label)
        others = [PropKit.kind_tokens(r["product"]) for n, r in enumerate(rows) if n not in (i, j)]
        left = len(rows) - 1
        tag_now = phrase if left >= 2 else ""
        owned = partial(cls._owned_tag, whole, ti + tj, others)
        retag = partial(cls._retag, owned, tag_now)
        reqs = []
        seen = []
        for req in list(rows[i]["requirements"]) + list(rows[j]["requirements"]):
            head, _, value = str(req).partition(":")
            bare = re.sub(r"\s*\([^()]*\)", "", head).strip() + ":" + value
            if bare.lower() in seen:
                continue
            reqs.append(bare)
            seen.append(bare.lower())
        merged = {"product": phrase, "quantity": rows[i]["quantity"], "requirements": reqs}
        kept = [merged if n == i else r for n, r in enumerate(raw) if n != j]
        needs = ctx.needs
        for name in ("written", "uncovered", "let_go", "indifferent"):
            setattr(needs, name, cls._distinct([retag(r) for r in getattr(needs, name)]))
        if needs.now_reqs is not None:
            needs.now_reqs = cls._distinct([retag(r) for r in needs.now_reqs])
        needs.resolved = [retag(r) for r in vars(needs).get("resolved") or []]
        for key in ("requirements", "hints", "references", "soft_wants"):
            if isinstance(reading.get(key), list):
                reading[key] = cls._distinct([retag(r) for r in reading[key]])
        targets = reading.get("reference_targets") if isinstance(reading.get("reference_targets"), list) else []
        for tgt in targets:
            if isinstance(tgt, dict) and owned(str(tgt.get("product") or "")):
                tgt["product"] = phrase
        for ref in vars(ctx.pointers).get("refs") or []:
            for key in ("req", "head", "now"):
                if isinstance(ref.get(key), str):
                    ref[key] = retag(ref[key])
            if owned(str(ref.get("tag") or "")):
                ref["tag"] = tag_now
        talk_state = vars(ctx.talk)
        for key in ("pending_seconds", "open_pair"):
            if isinstance(talk_state.get(key), list):
                talk_state[key] = [retag(p) for p in talk_state[key]]
        if left < 2:
            reading.pop("items", None)
            reading["product"] = phrase
            reading["quantity"] = rows[i]["quantity"]
        else:
            reading["items"] = kept
            reading["product"] = " and ".join(str(r.get("product")) for r in kept)
        needs.version += 1
        vars(ctx.shelf).pop("home_memo", None)
        vars(ctx.talk).pop("menu_memo", None)
        vars(ctx.talk).pop("quota_reads", None)
        vars(ctx.shelf).pop("walked", None)
        ctx.wire.scale = (1.0 + 0.3 * min(2, max(0, len(needs.items()) - 1))) * max(0.5, min(1.0, 40.0 / max(1, ctx.budget.step_cap)))
        ctx.log.append({"kind": "merge_joined", "label": label, "products": [rows[i]["product"], rows[j]["product"]],
                        "into": phrase, "label_words": in_order, "products_left": max(1, len(needs.items()))})
        return (f"'{label}' is a SINGLE category in this store, and the request wants one product from it ('{phrase}'); the order "
                + ("holds one line for it" if left >= 2 else "holds ONE line") + ", rather than a line for each kind the name "
                "combines. Requirements now: " + needs.text_body())

    @classmethod
    def lookup_key(cls, ctx, tool: str | None) -> str | None:
        schema = ctx.shelf.shapes.get(tool or "") or {}
        props = schema.get("properties") if isinstance(schema.get("properties"), dict) else {}
        required = schema.get("required")
        needed = [k for k in required or [] if k in props] if isinstance(required, list) else []
        texts = [k for k, p in props.items() if isinstance(p, dict) and p.get("type") == "string" and not p.get("enum")]
        key = next((k for k in texts if k in needed), None)
        if not key and len(texts) == 1:
            key = texts[0]
        if not key or any(k != key for k in needed):
            return None
        return key

    @classmethod
    def lookup_tools(cls, ctx) -> list[str]:
        taken = {ctx.role(r) for r in ("search", "message", "cart", "cart_view", "order", "decline")} - {None}
        return [name for name in ctx.shelf.shapes if name not in taken and cls.lookup_key(ctx, name)]

    @classmethod
    def lookup_call(cls, ctx, ident: str, tool: str | None) -> dict[str, Any] | None:
        if not tool:
            return None
        if tool == ctx.role("details"):
            return Judging.details_call(ctx, ident)
        known = ctx.shelf.archive.ids
        for entry in reversed(ctx.log):
            if entry.get("kind") != "env":
                continue
            for act in entry.get("actions") or []:
                args = act.get("args") if act.get("name") == tool else None
                if not isinstance(args, dict):
                    continue
                if any(isinstance(v, str) and v in known for v in args.values()):
                    return {k: (ident if isinstance(v, str) and v in known else v) for k, v in args.items()}
        key = cls.lookup_key(ctx, tool)
        return {key: ident} if key else None

    @classmethod
    def rec_id(cls, ctx, d: dict[str, Any]) -> str | None:
        for exact in (True, False):
            for key, val in d.items():
                name = str(key).lower()
                named = name == "id" if exact else name.endswith("id")
                if named and isinstance(val, str) and re.fullmatch(ID_LIKE, val):
                    return val
        return None

    @classmethod
    def asked_ids(cls, ctx, call_args: Any) -> set[str]:
        args = call_args if isinstance(call_args, dict) else {}
        named = set()
        for v in args.values():
            if isinstance(v, str) and re.fullmatch(ID_LIKE, v):
                named.add(v)
        for v in args.values():
            if isinstance(v, list):
                named.update(x for x in v if isinstance(x, str) and re.fullmatch(ID_LIKE, x))
        return named

    @classmethod
    def learn_caps(cls, ctx, call_name: str, call_args: Any, obs: Any) -> list[str]:
        shelf = ctx.shelf
        memo = vars(shelf)
        asked = cls.asked_ids(ctx, call_args)
        name = str(call_name)
        caps = memo.setdefault("tool_caps", {}).setdefault(name, {"calls": 0, "id": 0, "stock": 0, "props": 0, "notes": 0,
                                                                  "price": 0})
        caps["calls"] += 1
        hit = {"id": False, "stock": False, "props": False, "notes": False, "price": False}
        pass_state = dict(
            dead=memo.setdefault("dead", set()),
            dead_by=memo.setdefault("dead_by", {}),
            heard=memo.setdefault("stock_by", {}),
            row_text=memo.setdefault("row_text", {}),
            row_keys=memo.setdefault("row_keys", set()),
            keys=memo.setdefault("tool_keys", {}).setdefault(name, set()),
            note_keys=memo.setdefault("note_keys", {}).setdefault(name, set()),
            carts={ctx.role("cart"), ctx.role("cart_view")} - {None},
            epoch=vars(ctx.cart).get("cart_seq", 0),
            units=ctx.cart.trolley.units,
            newly=[], asked=asked, name=name, call_name=call_name, hit=hit)
        if not isinstance(obs, dict) or not obs.get("error"):
            TextKit.walk_dicts(obs, partial(cls._caps_visit, ctx, pass_state))
        for key, got in hit.items():
            caps[key] += int(got)
        return pass_state["newly"]

    @classmethod
    def _caps_stock(cls, ctx, ps, ident, state, source):
        ctx.shelf.stock_seen = True
        mine = ps["heard"].setdefault(ident, {})
        other = mine.get("lookup" if source != "lookup" else "row")
        if other and other[0] == ps["epoch"] and other[1] != state:
            ctx.shelf.rows_lag = True
        if source == "row" and mine.get("row") and mine["row"][1] != state:
            ctx.shelf.rows_live = True
        mine[source] = (ps["epoch"], state)
        dead = ps["dead"]
        if state is not False:
            if source == "lookup" or ps["dead_by"].get(ident) != "lookup":
                dead.discard(ident)
            return
        if ident not in dead and ident in ps["units"] and ident not in ps["newly"]:
            ps["newly"].append(ident)
        dead.add(ident)
        ps["dead_by"][ident] = source

    @classmethod
    def _caps_visit(cls, ctx, ps, d):
        ident = cls.rec_id(ctx, d)
        if not ident:
            return
        state = Reading.stock_state(ctx, d)
        name, hit = ps["name"], ps["hit"]
        source = "lookup" if ident in ps["asked"] or ps["call_name"] in ps["carts"] else "row"
        if state is not None:
            cls._caps_stock(ctx, ps, ident, state, source)
        if ident not in ps["asked"]:
            if ps["call_name"] not in ps["carts"]:
                ps["row_keys"].update(str(k) for k in d)
                ps["row_text"].setdefault(ident, {}).update({str(k): v for k, v in d.items() if isinstance(v, str)})
            return
        hit["id"] = True
        memo = vars(ctx.shelf)
        memo.setdefault("read_with", {}).setdefault(name, set()).add(ident)
        memo.setdefault("text_keys", {}).setdefault(name, set()).update(
            str(k) for k, v in d.items() if v is None or isinstance(v, str))
        hit["stock"] = hit["stock"] or state is not None
        if not hit["props"]:
            hit["props"] = any(isinstance(v, dict) and sum(1 for x in v.values() if isinstance(x, (str, int, float))) >= 2
                               for v in d.values())
        if not hit["price"]:
            hit["price"] = any(TextKit.is_number(v) and re.search(r"price|cost", str(k).lower()) for k, v in d.items())
        ps["keys"].update(str(k) for k in d)
        row = ps["row_text"].get(ident)
        for k, v in d.items():
            if isinstance(v, str) and len(v.split()) >= 8:
                if (row.get(str(k)) != v) if row is not None else (str(k) not in ps["row_keys"]):
                    ps["note_keys"].add(str(k))
                    hit["notes"] = True

    @classmethod
    def stock_tool(cls, ctx) -> str | None:
        caps = vars(ctx.shelf).get("tool_caps") or {}
        tools = cls.lookup_tools(ctx)
        shown = [t for t in tools if (caps.get(t) or {}).get("stock", 0) > 0]
        if shown:
            return max(shown, key=lambda t: (caps[t]["stock"], t == ctx.role("view"), t == ctx.role("details"), t))
        guess = vars(ctx.shelf).get("role_guess")
        if guess is None:
            try:
                guess = Closing.code_roles(ctx)
            except Exception:
                guess = {}
            ctx.shelf.role_guess = guess
        for tool in (ctx.role("view"), guess.get("view"), ctx.role("details")):
            if not tool or tool not in tools:
                continue
            got = caps.get(tool or "") or {}
            if got.get("id", 0) >= 2 and got.get("stock", 0) == 0:
                continue
            return tool
        busy = {guess.get(r) for r in ("cart", "cart_view", "order", "decline", "message")} - {None}
        for t in tools:
            if t not in busy and not (caps.get(t) or {}).get("id", 0):
                return t
        return None

    @classmethod
    def notes_tool(cls, ctx) -> str | None:
        memo = vars(ctx.shelf)
        caps = memo.get("tool_caps") or {}
        details = ctx.role("details")
        details_keys = (memo.get("tool_keys") or {}).get(details or "", set())
        note_keys = memo.get("note_keys") or {}
        shown = [t for t in cls.lookup_tools(ctx) if t != details and set(note_keys.get(t) or set()) - set(details_keys)]
        if not shown:
            return None
        return max(shown, key=lambda t: ((caps.get(t) or {}).get("notes", 0), t))

    @classmethod
    def dead_lines(cls, ctx) -> set[str]:
        return set(vars(ctx.shelf).get("dead") or set())

    @classmethod
    def stale_lines(cls, ctx) -> list[str]:
        seqs = vars(ctx.cart).get("line_seq") or {}
        mark = vars(ctx.cart).get("cart_seq", 0)
        return [line for line in ctx.cart.trolley.units if seqs.get(line, -1) < mark]

    @classmethod
    def _price_of(cls, ctx, ident: str) -> float:
        price = ctx.shelf.archive.price(ident)
        return float(price) if TextKit.is_number(price) else float("inf")

    @classmethod
    def _verified_swap(cls, ctx, line: str, kind: str, several: bool, rest: float) -> str:
        book, cart = ctx.cart.trolley, ctx.cart
        dead, ver = cls.dead_lines(ctx), str(ctx.needs.version)
        cap, qty = ctx.needs.spend_cap, max(1.0, book.held(line))
        fits = []
        for key in cart.passed:
            ident, _, version = key.rpartition("|")
            if version != ver or ident == line or ident in dead or ident in cart.refused_lines or ident in book.units:
                continue
            if Judging.field_view(ctx, ident)["fails"]:
                continue
            shelf = Judging.shelf_known(ctx, ident)
            if kind:
                if several and shelf != kind:
                    continue
                if not several and shelf not in ("", kind):
                    continue
            cost = cls._price_of(ctx, ident)
            if TextKit.is_number(cap) and cost != float("inf") and cost * qty + rest > cap:
                continue
            fits.append(ident)
        if not fits:
            return ""
        return min(fits, key=lambda i: (cls._price_of(ctx, i), i))

    @classmethod
    def _swap_entry(cls, ctx, ident, row, lim):
        price = cls._price_of(ctx, ident) if not TextKit.is_number(row["price"]) else row["price"]
        book, cart = ctx.cart.trolley, ctx.cart
        if row["stock"] is not True or ident == lim["line"] or ident in lim["dead"] or ident in cart.refused_lines or ident in book.units:
            return None
        cap = lim["cap"]
        if (lim["kind"] and row["shelf"] != lim["kind"]) or price == float("inf"):
            return None
        if TextKit.is_number(cap) and price * lim["qty"] + lim["rest"] > cap:
            return None
        if ident in lim["failed"] or ident in lim["pointers"] or Referencing.ref_excluded(ctx, row["fields"], lim["k"]):
            return None
        if ident not in ctx.shelf.archive.read:
            return (1, price, ident)
        if Judging.field_view(ctx, ident)["fails"]:
            return None
        judged = lim["judged"]
        if any(Closing.code_clash(ctx, row["fields"], prop, bare) for prop, bare in judged):
            return None
        if not all(Closing.code_shows(ctx, row, prop, bare, set(), True) for prop, bare in judged):
            return None
        return (0, price, ident)

    @classmethod
    def dead_swap(cls, ctx, line: str) -> tuple[str, str]:
        book, cart, records = ctx.cart.trolley, ctx.cart, ctx.shelf.archive
        dead, ver = cls.dead_lines(ctx), str(ctx.needs.version)
        kind = Judging.shelf_known(ctx, line)
        several = len(ctx.needs.items()) >= 2
        cap, qty = ctx.needs.spend_cap, max(1.0, book.held(line))
        rest = 0.0
        for ident in book.units:
            if ident != line and cls._price_of(ctx, ident) != float("inf"):
                rest += cls._price_of(ctx, ident) * book.held(ident)
        pick = cls._verified_swap(ctx, line, kind, several, rest)
        if pick:
            return pick, ""
        if several and not kind:
            return "", ""
        k = None
        if several:
            k = next((n for n, home in enumerate(cls.home_shelves(ctx)) if kind and kind in home), None)
        judged = [(prop, bare) for prop, bare in Closing.code_values(ctx, k) if not re.search(cls.PLAIN_VALUE, bare.lower())]
        failed = {key.split("|")[0] for key, val in cart.rulings.items() if val and key.split("|")[1:2] == [ver]}
        pointers = Referencing.ref_pointers(ctx)
        limits = dict(line=line, dead=dead, kind=kind, cap=cap, qty=qty, rest=rest, failed=failed, pointers=pointers, k=k,
                      judged=judged)
        rows = []
        for ident, row in Closing.code_rows(ctx).items():
            entry = cls._swap_entry(ctx, ident, row, limits)
            if entry:
                rows.append(entry)
        rows.sort()
        unread = [x for x in rows if x[0] == 1]
        low = min(unread, key=lambda x: (x[1], x[2]), default=None)
        if low and ctx.needs.priority() == "cheapest" and all(low[1] < p for r, p, _ in rows if r == 0):
            rows.insert(0, rows.pop(rows.index(low)))
        ctx.shelf.swap_unread = [i for r, _, i in rows[:3] if r == 1]
        ctx.shelf.swap_page = None
        if not any(r == 0 for r, _, _ in rows):
            try:
                ctx.shelf.swap_page = cls.dead_page(ctx, k)
            except Exception:
                ctx.shelf.swap_page = None
        listed = []
        for r, p, i in rows[:3]:
            figure = int(p) if float(p).is_integer() else p
            listed.append(f"{i} ({figure}{', read: its record shows every stated value' if r == 0 else ''})")
        return "", ", ".join(listed)

    @classmethod
    def dead_page(cls, ctx, k: int | None) -> dict[str, Any] | None:
        found = Reading.page_arg(ctx)
        if not found:
            return None
        walks = []
        for walk in vars(ctx.shelf).get("page_book", {}).values():
            if Walking.walk_carries(ctx, walk["args"], k) and Reading.next_page(ctx, walk) is not None:
                walks.append(walk)
        if not walks:
            return None
        best = max(walks, key=lambda w: (len(Walking.walk_carries(ctx, w["args"], k)), -len(w["read"])))
        return {**best["args"], found[0]: Reading.next_page(ctx, best)}

    @classmethod
    def swap_open(cls, ctx) -> str:
        if vars(ctx.shelf).get("swap_done"):
            return ""
        dead = cls.dead_lines(ctx)
        for ident in vars(ctx.shelf).get("swap_unread") or []:
            if ident not in ctx.shelf.archive.read and ident not in dead and ident not in ctx.cart.trolley.units:
                return ident
        return ""

    @classmethod
    def dead_note(cls, ctx, lines: list[str]) -> str:
        parts = []
        ctx.talk.event_seen = True
        for line in lines:
            ctx.shelf.swap_page, ctx.shelf.swap_unread = None, []
            pick, rows = cls.dead_swap(ctx, line)
            head = f"{line} is OUT OF STOCK now (so says its latest record), and an order that contains it will fail. Remove it from the cart"
            if pick:
                least = ", the cheapest such listing" if ctx.needs.priority() == "cheapest" else ""
                parts.append(f"{head} and add {pick} instead (the item check confirmed it meets every requirement in force now{least}), "
                             "check its stock and price again, then place the order.")
            elif rows:
                parts.append(f"{head}; no other verified fit for its product is on record, so look at the cheapest in-stock rows "
                             f"of its category inside the budget ({rows}; read rows whose own record misses or contradicts a "
                             "stated requirement were dropped, and an unread row needs its details read first), put in the first "
                             "one that fits, check its stock and price again, then place the order.")
            else:
                parts.append(f"{head} and look for a different listing that fits.")
            page = vars(ctx.shelf).get("swap_page")
            if page and not pick:
                parts.append("Further rows of its kind: the next unread page of the search that leads with the stated value, "
                             + json.dumps(page, default=str) + ".")
        return " ".join(parts)

    @classmethod
    def refusal_reread(cls, ctx, obs: Any = None) -> str:
        heard = ""
        try:
            heard, gone = cls.refusal_gone(ctx, obs)
        except Exception as err:
            gone = ""
            ctx.log.append({"kind": "refusal_gone_error", "error": f"{type(err).__name__}: {err}"[:300]})
        if gone:
            return heard + gone
        return heard + cls.refusal_queue(ctx)

    @classmethod
    def refusal_gone(cls, ctx, obs: Any) -> tuple[str, str]:
        book = ctx.cart.trolley
        lines = [i for i in book.units if book.held(i) > 0]
        if obs is None or len(lines) != 1 or ctx.wire.mute:
            return "", ""
        if vars(ctx.shelf).get("covering") or vars(ctx.talk).get("last_kind") != "order":
            return "", ""
        texts = Coverage.shopper_lines(ctx, ctx.role("order") or "", obs, True)
        if not texts:
            return "", ""
        needs = ctx.needs

        def snapshot() -> tuple:
            return needs.version, len(needs.let_go), len(needs.uncovered), str(needs.spend_cap), "|".join(needs.props_left())
        before = snapshot()
        note = needs.hear(texts, None)
        after = snapshot()
        heard = (" " + note) if note else ""
        if not note or after[0] == before[0] or after[1:] != before[1:]:
            return heard, ""
        line = lines[0]
        vars(ctx.shelf).setdefault("dead", set()).add(line)
        vars(ctx.shelf).setdefault("dead_by", {})[line] = "lookup"
        ctx.log.append({"kind": "refusal_dead", "line": line})
        return heard, (" The refusal reports that an item went unavailable and the cart had a single line, so the code treats "
                       + line + " as sold out (without a re-read): " + cls.dead_note(ctx, [line]))

    @classmethod
    def refusal_queue(cls, ctx) -> str:
        tool = cls.stock_tool(ctx)
        lines = list(ctx.cart.trolley.units)[:4]
        if not lines or not tool or ctx.wire.mute:
            return ""
        if vars(ctx.shelf).get("covering") or vars(ctx.shelf).get("open_queue"):
            return ""
        if ctx.moves_left() < len(lines) + 3 or ctx.wire.spare() < CLOSING:
            return ""
        shelf = ctx.shelf
        shelf.open_queue = lines
        shelf.open_recheck, shelf.open_purpose, shelf.open_tool = True, "cart", tool
        shelf.open_say = "The store refused the order, so I check the stock of every cart line again (one may be sold out)."
        return " The code is reading the stock of the cart lines for you now (" + tool + ")."

    @classmethod
    def code_judges_all(cls, ctx) -> bool:
        items = ctx.needs.items()
        plain = 0
        for k in (range(len(items)) if items else [None]):
            for _, value in Closing.code_values(ctx, k):
                if not re.search(cls.PLAIN_VALUE, value.lower()):
                    plain += 1
        resolved = cls.differ_done(ctx, None)
        judged = 0
        for req in ctx.needs.props_left():
            if any(Referencing.ref_match(ctx, ref, req) for ref in resolved):
                judged += 1
        return plain + judged >= len(list(ctx.needs.props_left()))

    @classmethod
    def differ_done(cls, ctx, k: int | None) -> list[dict[str, Any]]:
        rows, left = ctx.needs.items(), ctx.needs.props_left()
        found = []
        for ref in vars(ctx.pointers).get("refs") or []:
            if ref.get("relation") != "differ" or ref.get("status") != "resolved" or not str(ref.get("value") or "").strip():
                continue
            if not Referencing.ref_live(ctx, ref) or not any(Referencing.ref_match(ctx, ref, req) for req in left):
                continue
            if rows and k is not None and ref.get("tag") and Asking.product_of(ctx, f"x ({ref['tag']})") not in (k, None):
                continue
            found.append(ref)
        return found

    @classmethod
    def _held_picks(cls, ctx, picks: list[tuple[str, int]]) -> bool:
        book, wire = ctx.cart.trolley, ctx.wire
        was = wire.mute
        wire.mute = True
        try:
            bad = Closing.code_put(ctx, picks)
            done = bad == [] and all(i in book.units for i, _ in picks)
            stale_now = [i for i in cls.stale_lines(ctx) if any(i == p for p, _ in picks)]
            if done and stale_now and ctx.moves_left() >= len(stale_now) + 2:
                _, sold = Closing.code_recheck(ctx, stale_now)
                done = not sold
        finally:
            wire.mute = was
        return done and bool(book.units) and not set(book.units) & cls.dead_lines(ctx)

    @classmethod
    def _repair_picks(cls, ctx, several: bool, avoid: set[str], fitting) -> list[tuple[str, int]]:
        ver = str(ctx.needs.version)
        seen: dict[str, dict[str, Any]] = {}
        if several:
            seen = {e["id"]: e for k in range(len(ctx.needs.items())) for e in cls.ed_pool(ctx, k)}
            avoid |= {i for i, e in seen.items() if e["clash"]}
        picks = Closing.code_picks(ctx, avoid)
        checks = rounds = 0
        while several and picks and checks < 3 and rounds < 30 and ctx.wire.spare() > CLOSING + 30:
            unchecked = [i for i, _ in picks if not fitting([(i, 1)])]
            if not unchecked:
                break
            bare = [i for i in unchecked if not seen.get(i) or not seen[i]["revealed_ok"] or seen[i]["misses"]]
            if not bare:
                checks += 1
                Judging.check_item(ctx, unchecked[0], "ORDER")
                picks = Closing.code_picks(ctx, avoid)
                continue
            avoid.update(bare)
            avoid.update(i for i, e in seen.items() if not e["read"] and f"{i}|{ver}" not in ctx.cart.passed
                         and (not e["revealed_ok"] or e["misses"]))
            rounds += 1
            picks = Closing.code_picks(ctx, avoid)
        return picks

    @classmethod
    def decline_repair(cls, ctx) -> str:
        cart, wire = ctx.cart, ctx.wire
        several = len(ctx.needs.items()) >= 2
        early = several and cart.refusal_n == 0 and ctx.moves_left() > Intake.order_due(ctx)
        if cart.refusal_n <= 0 and ctx.moves_left() > Intake.order_due(ctx) and not early:
            return ""
        if not ctx.role("cart") or not ctx.role("order") or ctx.moves_left() < 3 or wire.spare() <= CLOSING:
            return ""
        if wire.mute or vars(ctx.shelf).get("covering") or wire.done:
            return ""
        tried = vars(cart).setdefault("repair_tried", set())
        mark: Any = ("early", cart.refusal_n) if early else cart.refusal_n
        if mark in tried:
            return ""
        tried.add(mark)
        if several:
            try:
                proof = cls.ed_proof(ctx)
            except Exception as err:
                proof = []
                ctx.log.append({"kind": "decline_proof_error", "error": f"{type(err).__name__}: {err}"[:300]})
            if proof:
                ctx.log.append({"kind": "decline_proof", "products": proof})
                return ""
        book = cart.trolley
        dead = cls.dead_lines(ctx)
        stale = set(cls.stale_lines(ctx))
        alive = {i for i in book.units if i not in dead and i not in stale}
        saved = set(cart.refused_lines)
        done = False
        ver = str(ctx.needs.version)
        own = cls.code_judges_all(ctx)

        def fitting(picks: list[tuple[str, int]]) -> bool:
            fit = vars(cart).get("code_fit") or {}
            if not picks:
                return False
            return all(f"{i}|{ver}" in cart.passed or (own and i in ctx.shelf.archive.read and fit.get(i)) for i, _ in picks)
        try:
            cart.refused_lines.difference_update(alive)
            picks = cls._repair_picks(ctx, several, set(dead), fitting)
            if not fitting(picks):
                ctx.log.append({"kind": "decline_repair", "picks": picks, "done": False})
                return ""
            if ctx.moves_left() < len(picks) + 2:
                ctx.log.append({"kind": "decline_repair", "picks": picks, "done": False, "why": "steps"})
                return ""
            done = cls._held_picks(ctx, picks)
            ctx.log.append({"kind": "decline_repair", "picks": picks, "done": done, "cart": dict(book.units)})
        except Exception as err:
            done = False
            ctx.log.append({"kind": "decline_repair_error", "error": f"{type(err).__name__}: {err}"[:300]})
        finally:
            if not done:
                cart.refused_lines.clear()
                cart.refused_lines.update(saved)
        if not done:
            return ""
        several = bool(ctx.needs.items())
        return ("Not run yet (no step used). Declining is correct only when nothing can fit, yet the records show a fitting "
                "candidate " + ("for every product asked for" if several else "") + " (passed by the item check, or read with "
                "every stated value in its own record), so the code placed " + ("them" if several else "it") + " in the cart and "
                "read the stock: " + book.summary() + " Place the order for this cart now; a decline is right only when a record you "
                "read shows one of these lines failing a requirement in force.")

    @classmethod
    def ed_plain(cls, ctx, value: str) -> bool:
        return not re.search(cls.PLAIN_VALUE, str(value).lower())

    @classmethod
    def ed_word(cls, ctx, value: str) -> bool:
        low = str(value).lower()
        if not re.search(r"[a-z]{3,}", low) or re.search(r"\d", low) or not cls.ed_plain(ctx, low):
            return False
        return not re.search(EXCLUDE_WORDS, low) and not RefKit.ref_text(low)

    @classmethod
    def ed_reqs(cls, ctx, k: int | None) -> list[tuple[str, str, bool]]:
        rows = ctx.needs.items()
        found = []
        for req in ctx.needs.props_left():
            head, _, value = str(req).partition(":")
            if rows and Asking.product_of(ctx, head) not in (k, None):
                continue
            bare = re.sub(r"\((?:stated|revealed[^)]*)\)", "", value).strip()
            found.append((re.split(r"\s*\(", head)[0].strip(), bare, req not in ctx.needs.written))
        return found

    @classmethod
    def _pool_entry(cls, ctx, ident: str, row: dict[str, Any], scope: dict[str, Any]) -> dict[str, Any]:
        records, cart = ctx.shelf.archive, ctx.cart
        read = ident in records.read
        cost = row["price"] if row["price"] is not None else records.price(ident)
        price = float(cost) if TextKit.is_number(cost) else None
        gone = ident in scope["dead"] or row["stock"] is False or ident in cart.refused_lines
        cap = scope["cap"]
        over = cap is not None and price is not None and price * scope["qty"] > cap + 1e-9
        pointer = ident in scope["pointers"] or Referencing.ref_excluded(ctx, row["fields"], scope["k"])
        fields = row["fields"]
        clash = []
        for p, v in scope["judged"]:
            if not Closing.code_clash(ctx, fields, p, v):
                continue
            if any(Closing.code_seen(ctx, v, val) for key, val in fields.items() if PropKit.same_prop(key, p)):
                continue
            clash.append(p)
        for p, parts in scope["joins"]:
            named = [val for key, val in fields.items() if PropKit.same_prop(key, p)]
            if named and not all(any(Closing.code_seen(ctx, x, val) for val in named) for x in parts):
                clash.append(p)
        misses = [(p, v) for p, v in scope["judged"] if not Closing.code_shows(ctx, row, p, v, set(), True)]
        ruled = ident in scope["failed"] or bool(read and Judging.field_view(ctx, ident)["fails"])
        verified = ident in scope["live"] and not gone
        fit = (verified or (read and not misses and not clash)) and not (ruled or pointer or gone or over)
        return {"id": ident, "price": price, "read": read, "verified": verified, "fit": fit, "gone": gone,
                "over": over, "pointer": pointer, "ruled": ruled, "clash": clash, "misses": misses, "row": row,
                "out": bool(gone or over or pointer or ruled or clash),
                "revealed_ok": all(Closing.code_shows(ctx, row, p, v, set(), True) for p, v in scope["shown"])}

    @classmethod
    def ed_pool(cls, ctx, k: int | None) -> list[dict[str, Any]]:
        records, cart = ctx.shelf.archive, ctx.cart
        rows = ctx.needs.items()
        ver = str(ctx.needs.version)
        mark = (k, len(records.items), len(records.read), ver, "|".join(ctx.needs.props_left()), len(cart.passed),
                sum(1 for v in cart.rulings.values() if v), tuple(sorted(cls.dead_lines(ctx))), tuple(sorted(cart.refused_lines)),
                str(ctx.needs.spend_cap), len(vars(ctx.pointers).get("refs") or []))
        memo = vars(ctx.shelf).setdefault("ed_memo", {})
        if mark in memo:
            return memo[mark]
        if len(memo) > 64:
            memo.clear()
        homes = cls.home_shelves(ctx)
        if not rows:
            home = set(Reading.shelf_labels(ctx)) or (set(homes[0]) if homes else set())
        elif k is not None and k < len(homes):
            home = set(homes[k])
        else:
            home = set()
        pool: list[dict[str, Any]] = []
        if home:
            counts = ctx.needs.counts()
            if rows and k is not None and k < len(counts):
                qty = counts[k]
            else:
                qty = counts[0] if counts else 1
            scope_k = k if rows else None
            joins = []
            for p, v in Closing.code_values(ctx, scope_k):
                if "+" not in v:
                    continue
                parts = [x.strip() for x in v.split("+") if x.strip()]
                if len(parts) >= 2 and all(re.search(r"[a-z]{2,}", x.lower()) for x in parts):
                    joins.append((p, parts))
            scope = {
                "k": scope_k, "qty": qty,
                "failed": {key.split("|")[0] for key, val in cart.rulings.items() if val and key.split("|")[1:2] == [ver]},
                "live": {key.rpartition("|")[0] for key in cart.passed if key.rpartition("|")[2] == ver},
                "cap": float(ctx.needs.spend_cap) if TextKit.is_number(ctx.needs.spend_cap) else None,
            }
            scope["judged"] = [(p, v) for p, v in Closing.code_values(ctx, scope_k) if cls.ed_plain(ctx, v)]
            scope["joins"] = joins
            scope["shown"] = [(p, v) for p, v, revealed in cls.ed_reqs(ctx, scope_k) if revealed]
            scope["pointers"], scope["dead"] = Referencing.ref_pointers(ctx), cls.dead_lines(ctx)
            for ident, row in Closing.code_rows(ctx).items():
                if row["shelf"] in home:
                    pool.append(cls._pool_entry(ctx, ident, row, scope))
        memo[mark] = pool
        return pool

    @classmethod
    def ed_proof(cls, ctx) -> list[int]:
        proven = []
        for k in range(len(ctx.needs.items())):
            reads = [e for e in cls.ed_pool(ctx, k) if e["read"]]
            if reads and all(e["out"] for e in reads):
                proven.append(k)
        return proven

    @classmethod
    def _led_scores(cls, ctx, k: int, hits: set[str]) -> dict[str, int]:
        values = []
        for _, v, _ in cls.ed_reqs(ctx, k):
            values.append({w for w in re.findall(r"[a-z0-9]{2,}", v.lower()) if w not in LINK_WORDS})
        led: dict[str, int] = {i: 1 for i in hits}
        lead_words = ctx.shelf.lead_words
        for walk in list(vars(ctx.shelf).get("page_book", {}).values()):
            words = [w for text in Reading.free_text(ctx, walk["args"]) for w in re.findall(r"[a-z0-9]{2,}", text.lower())]
            words = set(words[: lead_words] if lead_words else words)
            score = sum(1 for vals in values if vals & words)
            if not score:
                continue
            for i in (walk.get("row_shelf") or {}):
                led[str(i)] = max(led.get(str(i), 0), score)
        return led

    @classmethod
    def slack_rows(cls, ctx) -> list[str]:
        rows = ctx.needs.items()
        cap = ctx.needs.spend_cap
        if len(rows) < 2 or not TextKit.is_number(cap):
            return []
        try:
            counts = ctx.needs.counts()
            floors = []
            for k in range(len(rows)):
                prices = [e["price"] for e in cls.ed_pool(ctx, k) if e["fit"] and e["price"] is not None]
                if not prices:
                    return []
                floors.append(min(prices) * (counts[k] if k < len(counts) else 1))
            if sum(floors) <= float(cap) + 1e-9:
                return []
            hits = set(vars(ctx.shelf).get("all_hits") or [])
            pools = []
            for k in range(len(rows)):
                slack = (float(cap) - (sum(floors) - floors[k])) / max(1, counts[k] if k < len(counts) else 1)
                led = cls._led_scores(ctx, k, hits)
                ranked = []
                for e in cls.ed_pool(ctx, k):
                    if e["read"] or e["gone"] or e["pointer"] or e["ruled"] or e["clash"] or e["price"] is None:
                        continue
                    if 0 < slack and e["price"] <= slack + 1e-9:
                        ranked.append((-led.get(e["id"], 0), e["price"], e["id"]))
                pools.append([i for _, _, i in sorted(ranked)])
            chosen: list[str] = []
            while len(chosen) < 3 and any(pools):
                for pool in pools:
                    if pool and len(chosen) < 3:
                        chosen.append(pool.pop(0))
            if chosen:
                ctx.budget.slack_total = sum(floors)
            return chosen
        except Exception as err:
            ctx.log.append({"kind": "slack_error", "error": f"{type(err).__name__}: {err}"[:300]})
            return []

    @classmethod
    def verify_problems(cls, ctx, ident: str) -> list[str] | None:
        for entry in reversed(ctx.log):
            if not isinstance(entry, dict) or entry.get("kind") != "verify" or entry.get("item") != ident:
                continue
            res = entry.get("result")
            if not isinstance(res, dict) or res.get("fits") is not False:
                return None
            return [str(p) for p in res.get("problems") or [] if str(p).strip()]
        return None

    @classmethod
    def ed_news(cls, ctx, k: int | None) -> bool:
        if ctx.cart.refusal_n > 0:
            return True
        rows = ctx.needs.items()
        for req in list(ctx.needs.uncovered) + list(ctx.needs.let_go):
            if not rows or Asking.product_of(ctx, str(req).partition(":")[0]) in (k, None):
                return True
        return False

    @classmethod
    def near_hold(cls, ctx) -> str:
        try:
            cart = ctx.cart
            info = vars(cart).get("near_cart")
            if not info or info.get("ver") != ctx.needs.version or info.get("refusals") != cart.refusal_n or not ctx.role("order"):
                return ""
            lines = list(info.get("lines") or [])
            if not lines:
                return ""
            for i in lines:
                if i not in cart.trolley.units or i in cls.dead_lines(ctx) or i in cart.refused_lines:
                    return ""
            wire = ctx.wire
            if ctx.moves_left() < 1 or wire.done or wire.broke or wire.spare() < 2:
                return ""
            cart.near_cart = None
            ctx.log.append({"kind": "near_order", "lines": lines})
            was = wire.mute
            try:
                Closing.finish_now(ctx)
            finally:
                wire.mute = was
            return ("Not run (no step used). A meaning check found the cart's lines (" + ", ".join(lines) + ") give the "
                    "requested values in other wording, and nothing has changed since, so instead of declining the code placed the order.")
        except Exception as err:
            ctx.log.append({"kind": "near_hold_error", "error": f"{type(err).__name__}: {err}"[:300]})
            return ""

    @classmethod
    def near_check(cls, ctx) -> str:
        wire = ctx.wire
        if vars(ctx.cart).get("near_done") or not ctx.role("cart") or not ctx.role("order") or ctx.moves_left() < 4:
            return ""
        if wire.spare() <= CLOSING + 30 or wire.mute or vars(ctx.shelf).get("covering") or wire.done:
            return ""
        if wire.spent_out() or wire.no_model:
            return ""
        try:
            return cls.near_basket(ctx)
        except Exception as err:
            ctx.log.append({"kind": "near_error", "error": f"{type(err).__name__}: {err}"[:300]})
            return ""

    @classmethod
    def _near_candidates(cls, ctx, pool: list[dict[str, Any]], reqs) -> list[tuple]:
        cands = []
        for e in pool:
            if not e["read"] or e["gone"] or e["over"] or e["pointer"] or e["price"] is None:
                continue
            lines, ok = [], True
            for prop, value, revealed in reqs:
                if Closing.code_shows(ctx, e["row"], prop, value, set(), True):
                    continue
                says = [str(v).strip() for key, v in e["row"]["fields"].items() if PropKit.same_prop(key, prop) and str(v).strip()]
                if revealed or not says or not cls.ed_word(ctx, value) or any(re.search(r"\d", s) for s in says):
                    ok = False
                    break
                lines.append((prop, value, "; ".join(dict.fromkeys(says))[:120]))
            fails = set()
            problems = []
            if e["ruled"]:
                fails = {str(x).split(":", 1)[0].strip() for x in Judging.field_view(ctx, e["id"])["fails"]}
                problems = cls.verify_problems(ctx, e["id"]) or []
            marks = set()
            for prop, value, _ in lines:
                for w in PropKit.prop_words(prop) + re.findall(r"[a-z0-9]+", value.lower()):
                    if len(w) >= 3 and w not in LINK_WORDS:
                        marks.add(w)
            if not ok or not lines:
                continue
            if not all(any(PropKit.same_prop(f, prop) for prop, _, _ in lines) for f in fails):
                continue
            if all(any(re.search(r"\b" + re.escape(w), p.lower()) for w in marks) for p in problems):
                cands.append((len(lines), e["price"], e["id"], lines))
        return cands

    @classmethod
    def near_basket(cls, ctx) -> str:
        rows = ctx.needs.items()
        prods: list[int | None] = list(range(len(rows))) if rows else [None]
        counts = ctx.needs.counts()
        cap = float(ctx.needs.spend_cap) if TextKit.is_number(ctx.needs.spend_cap) else None

        def units(n: int) -> int:
            return counts[n] if n < len(counts) else 1
        chosen: dict[int, tuple[str, float, int]] = {}
        pools: dict[int, list[tuple[int, float, str, list[tuple[str, str, str]]]]] = {}
        for n, k in enumerate(prods):
            qty = units(n)
            pool = cls.ed_pool(ctx, k)
            fits = sorted((e["price"], e["id"]) for e in pool if e["fit"] and e["price"] is not None)
            if fits:
                chosen[n] = (fits[0][1], fits[0][0], qty)
                continue
            if not cls.ed_news(ctx, k):
                return ""
            cands = cls._near_candidates(ctx, pool, cls.ed_reqs(ctx, k))
            if not cands:
                return ""
            pools[n] = sorted(cands)[: 6 if len(prods) == 1 else 3]
        if not pools:
            return ""
        least = sum(price * qty for _, price, qty in chosen.values())
        least += sum(min(c[1] for c in pools[n]) * units(n) for n in pools)
        if cap is not None and least > cap + 1e-9:
            return ""
        flat = [(n, ident, prop, value, says) for n in sorted(pools) for _, _, ident, lines in pools[n] for prop, value, says in lines][:12]
        listing = "\n".join(f"{i + 1}. {ident}: {prop} = '{value[:60]}' | the listing's field says: {says}"
                            for i, (_, ident, prop, value, says) in enumerate(flat))
        ask = ("Context: a shopping assistant is close to declining an order, because the listings below do not use the words "
               "of the request for a requested value. THE SHOPPER'S REQUEST (for context):\n" + ctx.request[:1500]
               + "\n\nEvery line holds a listing, a requested property with its requested value, and the text of the listing's "
               "own field for that property.\n" + listing
               + "\n\nYour job: for every line, decide whether the listing's value is the requested value in other words.\n"
               "Constraints:\n- A synonym, a different spelling or phrasing, or a more precise form of the same thing counts as "
               "the same.\n- A different value of the property is not the same; when in doubt, use false.\n"
               "Answer with ONE JSON object, nothing else: {\"lines\": [{\"n\": 1, \"same\": true}]}")
        ctx.cart.near_done = True
        answer = ctx.wire.consult(ask, 600, CHECKER)
        yes: set[int] = set()
        said_lines = answer.get("lines") if isinstance(answer, dict) and isinstance(answer.get("lines"), list) else []
        for entry in said_lines:
            if isinstance(entry, dict) and entry.get("same") is True and TextKit.is_number(entry.get("n")):
                yes.add(int(entry["n"]) - 1)
        ctx.log.append({"kind": "near_check", "lines": [f"{i}:{p}={v}|{s}"[:120] for _, i, p, v, s in flat], "same": sorted(yes)})
        for n in sorted(pools):
            for _, price, ident, lines in pools[n]:
                idx = [i for i, row in enumerate(flat) if row[0] == n and row[1] == ident]
                if len(idx) == len(lines) and all(i in yes for i in idx):
                    chosen[n] = (ident, price, units(n))
                    break
            if n not in chosen:
                return ""
        if cap is not None and sum(price * qty for _, price, qty in chosen.values()) > cap + 1e-9:
            return ""
        picks = [(chosen[n][0], chosen[n][2]) for n in range(len(prods))]
        if ctx.moves_left() < len(picks) + 2:
            return ""
        return cls._near_commit(ctx, pools, chosen, picks, flat)

    @classmethod
    def _near_commit(cls, ctx, pools, chosen, picks, flat) -> str:
        cart = ctx.cart
        ver = str(ctx.needs.version)
        for n in pools:
            ident = chosen[n][0]
            key = f"{ident}|{ver}"
            for memo in [m for m in cart.rulings if m == key or m.startswith(key + "|")]:
                cart.rulings[memo] = ""
            cart.passed.add(key)
            vars(cart).setdefault("near_ok", set()).add(key)
            vars(cart).setdefault("pass_price", {})[ident] = ctx.shelf.archive.price(ident)
        done = cls._held_picks(ctx, picks)
        ctx.log.append({"kind": "near_basket", "picks": picks, "done": done, "cart": dict(cart.trolley.units)})
        if not done:
            return ""
        cart.near_cart = {"ver": ctx.needs.version, "refusals": cart.refusal_n, "lines": [i for i, _ in picks]}
        said_by = vars(cart).setdefault("near_said", {})
        confirmed = [(n, ident, prop, value, says) for n, ident, prop, value, says in flat
                     if ident == chosen.get(n, ("",))[0]]
        for n, ident, prop, value, says in confirmed:
            said_by.setdefault(f"{ident}|{ver}", []).append((prop, value, says))
        said = "; ".join(f"{ident} writes {prop} as '{says}' for '{value[:60]}'" for n, ident, prop, value, says in confirmed)
        return ("Not run yet (no step used). A meaning check showed that " + said + " - one value phrased differently - and "
                "every such listing meets each need the shopper revealed, is in stock and inside the budget, so the code placed "
                "the basket in the cart and read its stock: " + cart.trolley.summary() + " Place the order for this cart now; a decline is "
                "right only when a record you read shows one of these lines failing a requirement in force.")

    @classmethod
    def cart_moved(cls, ctx, before: dict[str, float]) -> None:
        cart = ctx.cart
        cart.cart_ver = vars(cart).get("cart_ver", 0) + 1
        cart.since_cart = 0
        now = cart.trolley.units
        if any(line not in before for line in now):
            cart.cart_seq = vars(ctx.budget).get("absorb_n", 0)
        seqs = vars(cart).setdefault("line_seq", {})
        for line, held in now.items():
            if line in before and held > before[line] + 1e-9:
                seqs.pop(line, None)

    @classmethod
    def after_put(cls, ctx, call_args: dict[str, Any], obs: Any) -> None:
        cart = ctx.cart
        book = cart.trolley
        if book.view_units(obs) is not None:
            cart.cart_shown = vars(cart).get("cart_ver", 0)
        firsts = vars(cart).setdefault("first_puts", set())
        new = []
        for line, count in book.lines(call_args):
            if (count is None or count > 0) and line in book.units and line not in firsts:
                new.append(line)
        if not new or not book.is_put(call_args) or Closing.code_fault(ctx, obs):
            return
        firsts.update(new)
        stale = set(cls.stale_lines(ctx))
        new = [line for line in new if line in stale]
        tool = cls.stock_tool(ctx)
        if not new or not tool or ctx.wire.mute or vars(ctx.shelf).get("covering"):
            return
        if ctx.moves_left() < 4 or ctx.wire.spare() < CLOSING:
            return
        queue = list(vars(cart).get("put_queue") or [])
        cart.put_queue = (queue + [line for line in new if line not in queue])[:4]
        cart.put_tool = tool
        ctx.log.append({"kind": "after_put_read", "lines": list(cart.put_queue), "tool": tool})

    @classmethod
    def mark_fresh(cls, ctx, obs: Any) -> None:
        if not vars(ctx.shelf).get("stock_seen"):
            return
        seqs = vars(ctx.cart).setdefault("line_seq", {})
        units = ctx.cart.trolley.units

        def visit(d: dict[str, Any]) -> None:
            ident = cls.rec_id(ctx, d)
            if ident in units and Reading.stock_state(ctx, d) is not None:
                seqs[ident] = vars(ctx.budget).get("absorb_n", 0)
        TextKit.walk_dicts(obs, visit)

    @classmethod
    def cart_gone(cls, ctx, lines: list[str]) -> set[str]:
        gone = set(lines) & cls.dead_lines(ctx)
        stale = [i for i in cls.stale_lines(ctx) if i in lines and i not in gone]
        if not stale or not cls.stock_tool(ctx) or ctx.moves_left() < len(stale) + 2:
            return gone
        _, sold = Closing.code_recheck(ctx, stale)
        return gone | set(sold) | (set(lines) & cls.dead_lines(ctx))

    @classmethod
    def _find_batch(cls, ctx) -> tuple[str, str, int] | None:
        shapes = ctx.shelf.shapes
        taken = {ctx.role(r) for r in ("search", "message", "cart", "cart_view", "order", "decline")} - {None}
        said = {}
        for tdef in ctx.shelf.toolset or []:
            fn = tdef.get("function") if isinstance(tdef, dict) else None
            if isinstance(fn, dict) and isinstance(fn.get("name"), str):
                spaced = re.sub(r"([a-z])([A-Z])", r"\1 \2", fn["name"]).replace("_", " ")
                said[fn["name"]] = spaced + " " + str(fn.get("description") or "")
        for name in list(shapes):
            if re.search(r"\b(?:remov|delet|cancel|clear|discard|releas|refund|purchas|checkout|wish|basket|cart|bag)", said.get(name, name).lower()):
                taken.add(name)
        for name, schema in shapes.items():
            props = schema.get("properties") if isinstance(schema, dict) else None
            if not isinstance(props, dict):
                props = {}
            arrays = []
            for key, p in props.items():
                if isinstance(p, dict) and p.get("type") == "array" and isinstance(p.get("items"), dict) \
                        and p["items"].get("type") == "string" and not p["items"].get("enum"):
                    arrays.append(key)
            needed = schema.get("required") if isinstance(schema, dict) and isinstance(schema.get("required"), list) else []
            most = props[arrays[0]].get("maxItems") if len(arrays) == 1 else None
            if name in taken or len(arrays) != 1 or [r for r in needed if r not in arrays]:
                continue
            if TextKit.is_number(most) and most >= 2:
                return name, arrays[0], int(min(most, 10))
        return None

    @classmethod
    def batch_tool(cls, ctx) -> tuple[str, str, int] | None:
        if "batch_spec" not in vars(ctx.cart):
            try:
                spec = cls._find_batch(ctx)
            except Exception:
                spec = None
            ctx.cart.batch_spec = spec
        return ctx.cart.batch_spec

    @classmethod
    def batch_ok(cls, ctx, found: list[Any], ids: list[str]) -> bool:
        got: set[str] = set()
        wanted = set(ids)

        def visit(d: dict[str, Any]) -> None:
            nested = any(isinstance(v, dict) and sum(1 for x in v.values() if isinstance(x, (str, int, float))) >= 2 for v in d.values())
            flat = False
            if not any(isinstance(v, dict) for v in d.values()):
                flat = sum(1 for v in d.values() if isinstance(v, str) and 0 < len(v) <= 70) >= 5
            if nested or flat:
                got.update(v for v in d.values() if isinstance(v, str) and v in wanted)
        try:
            for obs in found:
                TextKit.walk_dicts(obs, visit)
        except Exception:
            return False
        return bool(got) and 2 * len(got) >= len(wanted)

    @classmethod
    def title_of(cls, ctx, row: dict[str, Any]) -> str:
        if isinstance(row.get("title"), str):
            return row["title"]
        for key, v in row.items():
            if isinstance(v, str) and re.search(r"title|heading|headline", str(key).lower()):
                return v
        return ""

    @classmethod
    def _value_led_first(cls, ctx, home: set[str], most: int, words: dict[str, set[str]], known: list[set[str]]) -> list[str]:
        records = ctx.shelf.archive
        several = bool(ctx.needs.items())
        k = next((n for n, h in enumerate(cls.home_shelves(ctx)) if h == home), None) if several else None
        vals = []
        for r in ctx.needs.props_left():
            if several and Asking.product_of(ctx, r.partition(":")[0]) not in (k, None):
                continue
            vals.append(re.sub(r"\((?:stated|revealed[^)]*)\)", "", r.partition(":")[2]))
        vals = [v for v in vals if v.strip() and ValueKit.positive_value(v)
                and not re.search(r"as shown|same as|the page|sold by|redirect|refer", v.lower())]
        usable = bool(vals) and most > 0 and (not several or k is not None)
        facts: dict[str, list[str]] = {}

        def fact_visit(d: dict[str, Any]) -> None:
            ident = Reading.row_mark(ctx, d)
            if ident not in records.read or Reading.shelf_of(ctx, d) not in home:
                return
            for val in d.values():
                if isinstance(val, dict):
                    facts.setdefault(ident, []).extend(str(x) for x in val.values() if isinstance(x, (str, int, float)))
        for rec in records.items if usable else []:
            TextKit.walk_dicts(rec, fact_visit)
        fit_read = any(all(ValueKit.value_seen(v, " | ".join(texts)) for v in vals) for texts in facts.values())
        if not usable or fit_read:
            return []
        walks = [w for w in vars(ctx.shelf).get("page_book", {}).values() if Walking.walk_carries(ctx, w["args"], k)]
        walks.sort(key=lambda w: -len(Walking.walk_carries(ctx, w["args"], k)))
        for walk in walks:
            first = next((i for i in (walk.get("row_shelf") or {}) if i in words), None)
            if first:
                known.append(words[first])
                return [first]
        return []

    @classmethod
    def menu_rows(cls, ctx, home: set[str], most: int, taken: list[str], qty: int = 1) -> list[str]:
        records = ctx.shelf.archive
        cap = ctx.needs.spend_cap
        order: list[str] = []
        titles: dict[str, str] = {}
        sold: set[str] = set()
        known: list[set[str]] = []

        def visit(d: dict[str, Any]) -> None:
            ident = Reading.row_mark(ctx, d)
            if ident not in records.ids or Reading.shelf_of(ctx, d) not in home:
                return
            if ident in records.read:
                known.append(set(re.findall(r"[a-z0-9]{3,}", cls.title_of(ctx, d).lower())))
                return
            if Reading.sold_out(ctx, d):
                sold.add(ident)
            if ident not in order:
                order.append(ident)
                titles[ident] = cls.title_of(ctx, d)
        for rec in records.items:
            TextKit.walk_dicts(rec, visit)
        memo = vars(ctx.shelf)
        hits = sorted(memo.get("all_hits", []), key=lambda i: -memo.get("hit_score", {}).get(i, 0))
        cands = [i for i in hits if i in titles] + [i for i in order if i not in hits]

        def allowed(i: str) -> bool:
            if i in sold or i in taken or i in ctx.cart.refused_lines:
                return False
            price = records.price(i)
            return not (TextKit.is_number(cap) and TextKit.is_number(price) and records.price(i) * qty > cap)
        cands = [i for i in cands if allowed(i)]
        words = {i: set(re.findall(r"[a-z0-9]{3,}", titles.get(i, "").lower())) for i in cands}
        lead = memo.get("row_score", {})
        try:
            picks = cls._value_led_first(ctx, home, most, words, known)
        except Exception:
            picks = []
        while len(picks) < most:
            pool = [i for i in cands if i not in picks][:16]
            if not pool:
                break

            def spread(i: str) -> tuple:
                tier = 0 if i in hits else 1 if lead.get(i) else 2
                overlap = max((len(words[i] & w) / max(1, len(words[i] | w)) for w in known), default=0.0)
                return tier, overlap, cands.index(i)
            best = min(pool, key=spread)
            picks.append(best)
            known.append(words[best])
        return picks

    @classmethod
    def _interleave(cls, wants: list[list[str]]) -> list[str]:
        merged: list[str] = []
        while any(wants):
            for rows in wants:
                if rows:
                    merged.append(rows.pop(0))
        return merged

    @classmethod
    def _queue_reads(cls, ctx, picks: list[str], say: str) -> None:
        shelf = ctx.shelf
        shelf.open_queue = picks
        shelf.open_recheck = True
        shelf.open_say = say
        shelf.open_purpose = "menu"

    @classmethod
    def pre_read(cls, ctx) -> bool:
        try:
            shelf, records = ctx.shelf, ctx.shelf.archive
            batch = cls.batch_tool(ctx)
            again = vars(shelf).get("pre_rounds", 0) == 1
            if not batch or (vars(shelf).get("pre_read_done") and not again) or len(ctx.talk.questions_log) > 1:
                return False
            if ctx.moves_left() <= (25 if again else 20) or ctx.wire.spare() < CLOSING + 60:
                return False
            homes = cls.home_shelves(ctx)
            qtys = [row["quantity"] for row in ctx.needs.items()] or [ctx.needs.quantity()]

            def qty_of(k: int) -> int:
                return qtys[min(k, len(qtys) - 1)]
            if again:
                shelf.pre_rounds = 2
                wants: list[list[str]] = []
                taken: list[str] = []
                for k, home in enumerate(homes):
                    have = sum(records.rich_reads(s) for s in home) if home else 4
                    if have >= 3:
                        continue
                    rows = cls.menu_rows(ctx, home, 4 - have, taken + list(vars(shelf).get("pre_picked") or []), qty_of(k))
                    taken += rows
                    wants.append(rows)
                picks = cls._interleave(wants)[: batch[2]]
                if not picks:
                    return False
                cls._queue_reads(ctx, picks, "The first listings of some products carried few fields, so I read some more of them before the question.")
                return True
            empty: list[str] = []
            thin: list[tuple[int, int]] = []
            for k, home in enumerate(homes):
                have = sum(records.rich_reads(s) for s in home) if home else 4
                if have == 0:
                    empty += cls.menu_rows(ctx, home, 2 if ctx.needs.items() else 4, empty, qty_of(k))
                elif have < 4:
                    thin.append((k, 4 - have))
            picks = list(empty)
            for k, n in thin:
                picks += cls.menu_rows(ctx, homes[k], n, picks, qty_of(k))
            doubled = len(empty) > batch[2] and ctx.moves_left() > 25
            picks = picks[: batch[2] * (2 if doubled else 1)]
            if not picks:
                return False
            shelf.pre_read_done = True
            shelf.pre_rounds = 1
            shelf.pre_picked = list(picks)
            cls._queue_reads(ctx, picks, "Ahead of any question to the shopper, I read a few listings of every requested product to see which fields they carry.")
            return True
        except Exception as err:
            ctx.log.append({"kind": "pre_read_error", "error": f"{type(err).__name__}: {err}"[:300]})
            return False

    @classmethod
    def lacking_rows(cls, ctx) -> list[str]:
        try:
            rows = ctx.needs.items()
            if not rows:
                return []
            records, cart = ctx.shelf.archive, ctx.cart
            homes = cls.home_shelves(ctx)
            live = {k.rpartition("|")[0] for k in cart.passed if k.rpartition("|")[2] == str(ctx.needs.version)}
            live = live - cart.refused_lines - cls.dead_lines(ctx)
            got = {Judging.shelf_known(ctx, i) for i in live | set(vars(cart).get("audit_fits") or [])} - {""}
            lacking = [k for k, home in enumerate(homes) if home and not home & got]
            if not lacking:
                return []
            batch = cls.batch_tool(ctx)
            most = batch[2] if batch else 3
            cap = ctx.needs.spend_cap
            newest: list[tuple[str, str]] = []
            sold: set[str] = set()

            def visit(d: dict[str, Any]) -> None:
                ident, shelf = Reading.row_mark(ctx, d), Reading.shelf_of(ctx, d)
                if ident not in records.ids or not shelf:
                    return
                newest.append((ident, shelf))
                if Reading.sold_out(ctx, d):
                    sold.add(ident)
            for rec in reversed(records.items):
                TextKit.walk_dicts(rec, visit)
            where: dict[str, str] = {}
            for ident, shelf in newest:
                where.setdefault(ident, shelf)
            memo = vars(ctx.shelf)
            hits = sorted(memo.get("all_hits", []), key=lambda i: -memo.get("hit_score", {}).get(i, 0))
            pools: dict[int, list[str]] = {k: [] for k in lacking}
            for ident in dict.fromkeys([i for i in hits if i in where] + [i for i, _ in newest]):
                if ident in records.read or ident in cart.refused_lines or ident in sold:
                    continue
                k = next((k for k in lacking if where.get(ident) in homes[k]), None)
                if k is None:
                    continue
                cost = records.price(ident)
                if not (TextKit.is_number(cap) and TextKit.is_number(cost) and cost * rows[k]["quantity"] > cap):
                    pools[k].append(ident)
            picks: list[str] = []
            while len(picks) < most and any(pools.values()):
                for k in lacking:
                    if pools[k] and len(picks) < most:
                        picks.append(pools[k].pop(0))
            return picks
        except Exception as err:
            ctx.log.append({"kind": "lacking_error", "error": f"{type(err).__name__}: {err}"[:300]})
            return []


class Trail:
    @staticmethod
    def add(ctx, kind, **fields):
        entry = {"kind": kind}
        entry.update(fields)
        ctx.log.append(entry)

    @staticmethod
    def fault(ctx, kind, err):
        ctx.log.append({"kind": kind, "error": f"{type(err).__name__}: {err}"[:300]})

    @staticmethod
    def book(ctx):
        return vars(ctx.shelf).get("page_book", {})

    @staticmethod
    def live_ids(ctx):
        version = str(ctx.needs.version)
        found = set()
        for key in ctx.cart.passed:
            ident, _, tag = key.rpartition("|")
            if tag == version:
                found.add(ident)
        return found

    @staticmethod
    def failed_ids(ctx):
        version = str(ctx.needs.version)
        found = set()
        for key, why in ctx.cart.rulings.items():
            if why and key.split("|")[1:2] == [version]:
                found.add(key.split("|")[0])
        return found

    @staticmethod
    def append_note(ctx, text, closing=True):
        chat = ctx.talk.chat
        if not chat or chat[-1].get("role") != "tool":
            return
        tail = "\n" + ctx.status_line() if closing else ""
        chat[-1]["content"] = str(chat[-1]["content"]) + "\n" + text + tail

    @staticmethod
    def no_time(ctx, margin):
        return ctx.wire.done or ctx.wire.broke or ctx.wire.spare() < CLOSING + margin


class Walking:
    @classmethod
    def ran_query(cls, ctx, text):
        width = ctx.shelf.lead_words or 3

        def leading(raw):
            picked = re.findall(r"[a-z0-9]+", raw.lower())[:width]
            return sorted(map(lambda w: Reading.stem(ctx, w), picked))
        wanted = leading(str(text))
        for walk in Trail.book(ctx).values():
            if leading(" ".join(Reading.free_text(ctx, walk["args"]))) == wanted:
                return True
        return False

    @classmethod
    def search_template(cls, ctx):
        for walk in reversed(list(Trail.book(ctx).values())):
            args = walk["args"]
            if Coverage.cg_listing(ctx, args):
                continue
            texts = Reading.free_text(ctx, args)
            if len(texts) != 1:
                continue
            text_keys = [name for name, val in args.items() if isinstance(val, str) and val in texts and val.strip()]
            if len(text_keys) == 1:
                rest = dict(args)
                del rest[text_keys[0]]
                return rest, text_keys[0]
        return None

    @staticmethod
    def _close_words(a, b):
        if not any(ch.isdigit() for ch in a + b):
            return min(len(a), len(b)) >= 3 and (a.startswith(b) or b.startswith(a))
        return a.lstrip("0") == b.lstrip("0")

    @staticmethod
    def _word_bits(token):
        return list(filter(lambda bit: len(bit) >= 2, re.findall(r"[a-z0-9]+", token)))

    @classmethod
    def word_trap(cls, ctx, raw):
        try:
            return cls._trap_of(ctx, raw)
        except Exception as err:
            Trail.fault(ctx, "trap_error", err)
            return "", ""

    @classmethod
    def _trap_needs(cls, ctx):
        needs = []
        for req in ctx.needs.props_left():
            head, colon, tail = req.partition(":")
            if colon and not RefKit.ref_text(tail):
                cleaned = re.sub(r"\((?:stated|revealed[^)]*)\)", "", tail).lower()
                needs.append((re.split(r"\s*\(", head)[0].lower(), head, cleaned))
        return needs

    @classmethod
    def _trap_of(cls, ctx, raw):
        if ctx.shelf.word_kind == "any":
            return "", ""
        tokens = []
        for bit in " ".join(raw).split():
            piece = bit.strip("\"'()[]{},.;!?|").lower()
            if re.search(r"[a-z0-9]", piece):
                tokens.append(piece)
        width = ctx.shelf.lead_words
        if not width:
            width = max(3, len(tokens)) if ctx.shelf.word_kind == "all" else 3
        tokens = tokens[:width]
        close, bits = cls._close_words, cls._word_bits
        needs = cls._trap_needs(ctx)
        value_words = set()
        for _, _, val in needs:
            value_words.update(re.findall(r"[a-z0-9]{2,}", val))
        products = [row["product"] for row in ctx.needs.items()] or [str(ctx.needs.reading.get("product") or "")]
        kind_text = " ".join(products + sorted(Reading.shelf_labels(ctx))).lower()
        kind_words = set(re.findall(r"[a-z0-9]{3,}", kind_text)) - LINK_WORDS
        name_words = set()
        for low, _, _ in needs:
            name_words.update(re.findall(r"[a-z]{3,}", low))
        name_words -= LINK_WORDS
        gone_words = set()
        for old in ctx.needs.let_go:
            gone_words.update(re.findall(r"[a-z]{3,}", old.partition(":")[2].lower()))
        gone_words -= LINK_WORDS

        def near_any(tok_bits, pool):
            return any(close(b, w) for b in tok_bits for w in pool)
        flagged, reason = [], ""
        for tok in tokens:
            tok_bits = bits(tok)
            if not tok_bits or near_any(tok_bits, value_words | kind_words):
                continue
            if near_any(tok_bits, name_words):
                flagged.append(tok)
                reason = reason or "name"
            elif near_any(tok_bits, gone_words):
                flagged.append(tok)
                reason = reason or "dropped"
        if not flagged:
            return "", ""
        kept, kept_bits = [], []
        for tok in tokens:
            if tok in flagged:
                continue
            kept.append(tok)
            kept_bits.extend(bits(tok))
        first_held = None
        for _, head, val in needs:
            if near_any(kept_bits, re.findall(r"[a-z0-9]{2,}", val)):
                first_held = head
                break
        if first_held is None:
            return "", ""
        if not near_any(kept_bits, kind_words):
            index = Asking.product_of(ctx, first_held) if ctx.needs.items() else None
            if index is not None and index < len(products):
                product = products[index]
            else:
                product = products[0]
            kept += [w for w in re.findall(r"[a-z]{3,}", product.lower()) if w not in LINK_WORDS][-1:]
        query = " ".join(kept[:width])
        if not query or cls.ran_query(ctx, query):
            return "", ""
        if reason == "name":
            note = (f"This query's word '{flagged[0]}' is a word of a required property's NAME, not of a required value: every word is "
                    "required here, so it keeps only the listings whose fields or title repeat that name and misses those whose field "
                    "holds just the value.")
        else:
            note = (f"This query's word '{flagged[0]}' belongs only to a requirement the shopper DROPPED: it still keeps only the "
                    "listings carrying it, though they no longer need to.")
        return note + f" Also search '{query}' and read every page of it.", query

    @classmethod
    def combined_queries(cls, ctx):
        template = cls.search_template(ctx)
        if not template:
            return []
        rows = ctx.needs.items()
        width = ctx.shelf.lead_words or 3
        live = Trail.live_ids(ctx) - ctx.cart.refused_lines
        covered = set(map(lambda ident: Judging.shelf_known(ctx, ident), live)) - {""}
        homes = Lookup.home_shelves(ctx) if rows else []
        found = []
        for k in (range(len(rows)) if rows else [None]):
            if rows and (k >= len(homes) or not homes[k] or homes[k] & covered):
                continue
            if not rows and not Intake.unverified(ctx):
                continue
            value_sets = cls._word_values(ctx, k, rows)
            if len(value_sets) < 2:
                continue
            query = []
            for words in value_sets:
                if len(query) + len(words) > width:
                    continue
                query.extend(words)
            product = rows[k]["product"] if rows else str(ctx.needs.reading.get("product") or "")
            kind = [w for w in re.findall(r"[a-z]{3,}", product.lower()) if w not in LINK_WORDS][-1:]
            if kind and len(query) < width and kind[0] not in query:
                query.extend(kind)
            text = " ".join(query)
            if len(query) > 1 and not cls.ran_query(ctx, text):
                fresh = dict(template[0])
                fresh[template[1]] = text
                found.append(fresh)
        return found[:2]

    @classmethod
    def _word_values(cls, ctx, k, rows):
        picked = []
        for req in ctx.needs.props_left():
            head, _, value = req.partition(":")
            if rows and Asking.product_of(ctx, head) not in (k, None):
                continue
            value = re.sub(r"\((?:stated|revealed[^)]*)\)", "", value).strip().lower()
            if value and not re.search(r"\d|as shown|same as|page|redirect|refer|\bnot\b|\bno\b|without|except|anything|other than", value):
                lead = re.split(r"[,;/]|\bor\b|\band\b", value)[0]
                words = [w for w in re.findall(r"[a-z]{2,}", lead) if w not in LINK_WORDS]
                if words and len(words) <= 2:
                    picked.append(words)
        return picked

    @classmethod
    def arbitrate(cls, ctx, thing, ruling):
        try:
            audits = [entry for entry in ctx.log if isinstance(entry, dict) and entry.get("kind") == "audit"
                      and isinstance(entry.get("result"), dict)]
            reason = str((audits[-1].get("result") or {}).get("reason") or "") if audits else ""
            quoted = f" ({reason[:400]})" if reason else ""
            note = ("\n\nTWO READINGS DISAGREE ABOUT THIS ITEM. Reading the same records, the decline audit judged that it meets "
                    "every requirement" + quoted + "; an earlier check had found: " + ruling[:500]
                    + "\nSettle from the records which reading holds: a field giving the required value in other words, or a "
                    "shade, qualifier or spelling of that same named value, meets the requirement; a field giving a different "
                    "value fails it.")
            verdict = Judging.check_item(ctx, thing, "ORDER", second=note)
            if verdict:
                return verdict
            key = f"{thing}|{ctx.needs.version}"
            rulings = ctx.cart.rulings
            for memo in [memo for memo in rulings if memo == key or memo.startswith(key + "|")]:
                rulings[memo] = ""
            return ""
        except Exception as err:
            Trail.fault(ctx, "arbitrate_error", err)
            return ruling

    @classmethod
    def _stock_marks(cls, ctx):
        marks = {}
        archive = ctx.shelf.archive

        def visit(d):
            ident = Reading.row_mark(ctx, d)
            if ident not in archive.ids:
                return
            if any(re.search(r"stock|avail", str(key).lower()) for key in d):
                marks[ident] = Reading.sold_out(ctx, d)
        for rec in archive.items:
            TextKit.walk_dicts(rec, visit)
        return marks

    @classmethod
    def unrefuted(cls, ctx):
        try:
            return cls._unrefuted_picks(ctx)
        except Exception as err:
            Trail.fault(ctx, "unrefuted_error", err)
            return []

    @classmethod
    def _audited_out(cls, ctx):
        cart_state = vars(ctx.cart)
        reviewed = cart_state.get("audit_ids")
        read = ctx.shelf.archive.read
        if cart_state.get("audit_done") and (reviewed is None or len(read) <= cart_state.get("audit_reads", -1)):
            reviewed = set(read)
        return set(reviewed or set()) - set(cart_state.get("audit_fits") or [])

    @classmethod
    def _unrefuted_picks(cls, ctx):
        rows = ctx.needs.items()
        homes = Lookup.home_shelves(ctx) if rows else [set(Reading.shelf_labels(ctx))]
        if not homes or not all(homes):
            return []
        failed = Trail.failed_ids(ctx)
        live = Trail.live_ids(ctx)
        ruled_out = cls._audited_out(ctx)
        sold = cls._stock_marks(ctx)
        cap = ctx.needs.spend_cap
        archive = ctx.shelf.archive
        picks, total = [], 0.0
        pointers = Referencing.ref_pointers(ctx)
        for k, home in enumerate(homes):
            qty = ctx.needs.quantity() if not rows else rows[k]["quantity"]
            pool = []
            for ident in sorted(archive.read):
                if ident in failed or ident in ruled_out or ident in ctx.cart.refused_lines or sold.get(ident):
                    continue
                if ident in picks or ident in pointers or Judging.shelf_known(ctx, ident) not in home:
                    continue
                cost = archive.price(ident)
                if not TextKit.is_number(cost):
                    continue
                if TextKit.is_number(cap) and cost * qty > cap:
                    continue
                pool.append((ident not in live, cost, ident))
            if not pool:
                return []
            best = min(pool)
            picks.append(best[2])
            total += best[1] * qty
        if TextKit.is_number(cap) and total > cap:
            return []
        return picks

    @classmethod
    def fit_hold(cls, ctx):
        try:
            return cls._fit_note(ctx)
        except Exception as err:
            Trail.fault(ctx, "fit_hold_error", err)
            return ""

    @classmethod
    def _fit_note(cls, ctx):
        blocked = vars(ctx.cart).get("fit_held") or ctx.cart.trolley.units or ctx.moves_left() <= 4
        if blocked or ctx.wire.spare() < CLOSING + 30:
            return ""
        archive = ctx.shelf.archive
        fits = list(filter(lambda ident: ident in archive.ids and ident not in ctx.cart.refused_lines,
                           vars(ctx.cart).get("audit_fits") or []))
        if not fits:
            return ""
        ctx.cart.fit_held = True
        good = []
        for thing in fits[:2]:
            if thing not in archive.read:
                continue
            ruling = Judging.check_item(ctx, thing, "ORDER")
            if ruling and ctx.wire.spare() > CLOSING + 30:
                ruling = cls.arbitrate(ctx, thing, ruling)
            if not ruling and f"{thing}|{ctx.needs.version}" in ctx.cart.passed:
                good.append(thing)
                if not ctx.needs.items():
                    break
        if not good:
            return ""
        if ctx.needs.items():
            covered = {Judging.shelf_known(ctx, ident) for ident in Trail.live_ids(ctx) - ctx.cart.refused_lines} - {""}
            for home in Lookup.home_shelves(ctx):
                if home and not home & covered:
                    return ""
        one = len(good) == 1
        target = "a verified line of every product" if ctx.needs.items() else good[0]
        return ("The decline audit listed " + ", ".join(good) + " as a possible fit, and the independent item check, reading "
                + ("its" if one else "their") + " full record, found " + ("it meets" if one else "they meet")
                + " every requirement now in force. Weigh " + ("it" if one else "them") + " before declining: if it fits, put "
                + target + " in the cart, re-read its stock and price, "
                "and order; decline if you find that it fails a requirement in force.")

    @classmethod
    def id_store(cls, ctx):
        flags = [walk.get("id_order") for walk in Trail.book(ctx).values() if "id_order" in walk]
        if False in flags:
            return False
        return flags.count(True) >= 2

    @classmethod
    def product_index(cls, ctx, thing):
        if not ctx.needs.items():
            return None
        shelf = Judging.shelf_known(ctx, thing)
        homes = Lookup.home_shelves(ctx)
        return next((k for k, home in enumerate(homes) if shelf and shelf in home), None)

    @classmethod
    def walk_home(cls, ctx, k):
        homes = Lookup.home_shelves(ctx)
        index = 0 if k is None else k
        home = set(homes[index]) if homes and (k or 0) < len(homes) else set()
        if not home and k is None:
            return set(Reading.shelf_labels(ctx))
        return home

    @classmethod
    def fig_amount(cls, ctx, text):
        low = str(text).lower()
        if re.search(CH_BOUND, low):
            return None
        hit = re.match(r"\s*(\d+)\s+([a-z]+)", low)
        return (hit.group(1), hit.group(2)) if hit else None

    @classmethod
    def amount_part(cls, ctx, text):
        low = str(text).lower()
        return bool(re.search(r"\d", low)) and bool(re.search(CH_BOUND, low) or re.match(r"\s*\d+(?:[.,]\d+)?\s+[a-z]", low))

    @classmethod
    def walk_carries(cls, ctx, args, k, alts=False):
        def close(a, b):
            if min(len(a), len(b)) < 3:
                return False
            return a.startswith(b) or b.startswith(a)
        texts = Reading.free_text(ctx, args)
        words, tokens = [], []
        for text in texts:
            words += re.findall(r"[a-z0-9]{2,}", text.lower())
            tokens += re.findall(r"[a-z0-9]+", text.lower())
        if ctx.shelf.lead_words:
            words = words[: ctx.shelf.lead_words]
            tokens = tokens[: ctx.shelf.lead_words]
        by_product = bool(ctx.needs.items()) and k is not None
        carried = set()
        for n, req in enumerate(ctx.needs.props_left()):
            if by_product and Asking.product_of(ctx, req.partition(":")[0]) not in (k, None):
                continue
            value = re.sub(r"\((?:stated|revealed[^)]*)\)", "", req.partition(":")[2]).lower()
            parts = re.split(r"[,;/]|\bor\b|\beither\b|\bnor\b", value)
            for a in range(len(parts)):
                part = parts[a]
                amount = cls.fig_amount(ctx, part)
                if amount:
                    hit = amount[0] in tokens and any(w == amount[1] or close(w, amount[1]) for w in tokens)
                elif cls.amount_part(ctx, part):
                    hit = False
                else:
                    hit = any(close(w, v) for w in words for v in re.findall(r"[a-z0-9]{3,}", part))
                if hit:
                    carried.add((n, a) if alts else n)
        return carried

    @classmethod
    def walk_score(cls, ctx, args, k):
        return len(cls.walk_carries(ctx, args, k))

    @classmethod
    def _value_stems(cls, ctx, k):
        by_product = bool(ctx.needs.items()) and k is not None
        stems = set()
        for req in ctx.needs.props_left():
            if by_product and Asking.product_of(ctx, req.partition(":")[0]) not in (k, None):
                continue
            for w in re.findall(r"[a-z0-9]{2,}", req.partition(":")[2].lower()):
                stems.add(Reading.stem(ctx, w))
        return stems

    @classmethod
    def walk_options(cls, ctx, k):
        home = cls.walk_home(ctx, k)
        if not home:
            return []
        cut = ctx.shelf.lead_words or None

        def lead(walk):
            text = " ".join(Reading.free_text(ctx, walk["args"])).lower()
            return set(Reading.stem(ctx, w) for w in re.findall(r"[a-z0-9]+", text)[:cut])
        label = set()
        for name in home:
            label.update(Reading.stem(ctx, w) for w in re.findall(r"[a-z0-9]{2,}", name))
        values = cls._value_stems(ctx, k)
        book = Trail.book(ctx)

        def read_through(walk):
            return walk.get("id_order") is True and walk.get("first") and Reading.next_page(ctx, walk) is None and lead(walk) \
                and len(walk.get("row_shelf") or {}) >= walk["total"]
        finished = []
        for done_walk in book.values():
            if not read_through(done_walk):
                continue
            finished.append((lead(done_walk), cls.walk_carries(ctx, done_walk["args"], k, alts=True),
                             Coverage.vl_mode(ctx, done_walk["args"])))
        best = {}
        for key, walk in book.items():
            if walk.get("id_order") is not True or not walk.get("first") or Reading.next_page(ctx, walk) is None:
                continue
            carries = cls.walk_carries(ctx, walk["args"], k)
            on_home = any(name in home for name in (walk.get("row_shelf") or {}).values())
            if not (on_home and carries):
                continue
            words = lead(walk)
            pairs = cls.walk_carries(ctx, walk["args"], k, alts=True)
            mode = Coverage.vl_mode(ctx, walk["args"])
            shadowed = False
            for seen, got, seen_mode in finished:
                if seen_mode == mode and (seen <= words or (seen <= values | label and pairs <= got)):
                    shadowed = True
                    break
            if shadowed:
                continue
            pages = -(-walk["total"] // max(1, walk["size"])) - len(walk["read"])
            twin = (walk["total"], walk["first"])
            if pages <= 0 or pages > 2 * WALK_PAGES:
                continue
            if twin not in best or pages < best[twin][0]:
                best[twin] = (pages, key, walk)
        options = list(best.values())
        if Coverage.all_words(ctx):
            reach = values | label

            def nested(p, o):
                a, b = lead(p[2]), lead(o[2])
                if a < b:
                    return not (b - a) <= reach and p[0] <= WALK_PAGES
                return b < a and (a - b) <= reach
            survivors = []
            for option in options:
                if any(nested(other, option) for other in options if other is not option):
                    continue
                survivors.append(option)
            options = survivors
        options.sort(key=lambda p: (p[0], -cls.walk_score(ctx, p[2]["args"], k), p[1]))
        return options

    @classmethod
    def walk_plan(cls, ctx, budget, only=None):
        if budget <= 0 or ctx.needs.priority() != "cheapest":
            return []
        if not ctx.role("search") or not Reading.page_arg(ctx) or not cls.id_store(ctx):
            return []
        walked = vars(ctx.shelf).get("walked", set())
        wants = []
        for k in (list(range(len(ctx.needs.items()))) or [None]):
            if k in walked or (only is not None and k not in only):
                continue
            options = cls.walk_options(ctx, k)
            if options:
                wants.append((options[0][0], -1 if k is None else k, k, options[0][1]))
        plan, left = [], budget
        for pages, _, k, key in sorted(wants, key=lambda w: w[:2]):
            n = min(pages, WALK_PAGES, left)
            if n <= 0:
                break
            plan.append((k, key, n))
            left -= n
        return plan

    @classmethod
    def walk_pick(cls, ctx, k):
        gone = Lookup.dead_lines(ctx)

        def eligible(ident):
            return ident not in gone and (k is None or cls.product_index(ctx, ident) == k)
        candidates = [ident for ident in ctx.cart.trolley.units if (k is None or cls.product_index(ctx, ident) == k) and ident not in gone]
        if not candidates:
            version = str(ctx.needs.version)
            passed = [key.rpartition("|")[0] for key in ctx.cart.passed if key.rpartition("|")[2] == version]
            candidates = [ident for ident in passed if ident not in ctx.cart.refused_lines and eligible(ident)]
        best = None
        for ident in candidates:
            cost = ctx.shelf.archive.price(ident)
            if not TextKit.is_number(cost):
                continue
            entry = (ident, float(cost))
            if best is None or (entry[1], entry[0]) < (best[1], best[0]):
                best = entry
        return best

    @classmethod
    def _record_of(cls, ctx, ident):
        found = {}

        def visit(d):
            nonlocal found
            if Reading.row_mark(ctx, d) == ident and Lookup.title_of(ctx, d):
                found = d
        for item in ctx.shelf.archive.items:
            TextKit.walk_dicts(item, visit)
        return found

    @staticmethod
    def _maker_of(rec):
        for key, val in rec.items():
            low_key = str(key).lower()
            if not isinstance(val, str) or not val.strip():
                continue
            if re.search(r"brand|make|manufactur", low_key) and "id" not in low_key:
                return val
        return ""

    @classmethod
    def line_query(cls, ctx, k):
        pick = cls.walk_pick(ctx, k)
        template = cls.search_template(ctx)
        if not (pick and template):
            return None
        rec = cls._record_of(ctx, pick[0])
        maker_words = re.findall(r"[a-z0-9]+", cls._maker_of(rec).lower())
        if not maker_words:
            return Coverage.title_line(ctx, k, rec, template)
        avoid = set(maker_words) | LINK_WORDS
        for name in cls.walk_home(ctx, k):
            avoid.update(re.findall(r"[a-z]+", name))
        for req in ctx.needs.props_left():
            avoid.update(re.findall(r"[a-z]+", req.lower()))
        for row in ctx.needs.items() or [{"product": str(ctx.needs.reading.get("product") or "")}]:
            avoid.update(re.findall(r"[a-z]+", row["product"].lower()))
        title_words = re.findall(r"[a-z][a-z'-]{3,}", Lookup.title_of(ctx, rec).lower())
        if maker_words[0] in title_words:
            title_words = title_words[title_words.index(maker_words[0]) + 1:]
        line_word = next((w for w in title_words if w.strip("'-") not in avoid and "'" not in w), None)
        if line_word is None:
            return None
        text = f"{maker_words[0]} {line_word}"
        if cls.ran_query(ctx, text):
            return None
        query = dict(template[0])
        query[template[1]] = text
        return query

    @classmethod
    def walk_rows(cls, ctx, k, keys, most):
        home = cls.walk_home(ctx, k)
        book = Trail.book(ctx)
        archive = ctx.shelf.archive
        counts = [len(cls.walk_carries(ctx, w["args"], k)) for w in book.values() if w.get("id_order") is True]
        top = max(counts) if counts else 0
        ids = set()
        for key, walk in book.items():
            got = set()
            if walk.get("id_order") is True:
                got = cls.walk_carries(ctx, walk["args"], k)
            if (got and len(got) == top) or key in keys:
                ids.update(ident for ident, name in (walk.get("row_shelf") or {}).items() if name in home)
        pick = cls.walk_pick(ctx, k)
        if not pick:
            priced = sorted((float(archive.price(i)), i) for i in ids if i in archive.read and TextKit.is_number(archive.price(i)))
            if priced:
                pick = (priced[0][1], priced[0][0])
        rows = ctx.needs.items()
        qty = rows[k]["quantity"] if k is not None and k < len(rows) else ctx.needs.quantity()
        cap = ctx.needs.spend_cap
        sold = {}

        def visit(d):
            if any(isinstance(val, str) and val in ids and "id" in str(key).lower() for key, val in d.items()):
                sold[Reading.row_mark(ctx, d)] = Reading.sold_out(ctx, d)
        for item in archive.items:
            TextKit.walk_dicts(item, visit)
        cheaper = []
        for ident in ids:
            cost = archive.price(ident)
            skip = ident in archive.read or ident in ctx.cart.refused_lines or sold.get(ident) or not TextKit.is_number(cost)
            if skip or (TextKit.is_number(cap) and cost * qty > cap) or (pick and cost >= pick[1]):
                continue
            cheaper.append((cost, ident))
        return [pair[1] for pair in sorted(cheaper)[:most]]

    @classmethod
    def page_digest(cls, ctx, obs):
        rows = []

        def visit(d):
            if len(d) < 4:
                return
            if any(isinstance(val, str) and val in ctx.shelf.archive.ids and "id" in str(key).lower() for key, val in d.items()):
                rows.append(d)
        TextKit.walk_dicts(obs, visit, ceiling=3)
        lines = []
        for row in rows[:12]:
            cost = None
            for key, val in row.items():
                if TextKit.is_number(val) and re.search(r"price|cost", str(key).lower()):
                    cost = val
                    break
            shown = "?" if cost is None else cost
            stock = " SOLD OUT" if Reading.sold_out(ctx, row) else ""
            lines.append(f"{Reading.row_mark(ctx, row)} {shown}{stock} '{Lookup.title_of(ctx, row)[:70]}' [{Reading.shelf_of(ctx, row)}]")
        body = "; ".join(lines) if lines else "none"
        return "Page rows (id, price, title, category): " + body

    @classmethod
    def walk_call(cls, ctx, args, say):
        chat = ctx.talk.chat
        call_id = f"walk_{len(chat)}_{ctx.wire.tick}"
        call = {"name": ctx.role("search"), "arguments": json.dumps(args)}
        proposal = {"id": call_id, "type": "function", "function": call}
        chat.append({"role": "assistant", "content": say, "tool_calls": [proposal]})
        count_before = len(ctx.shelf.archive.items)
        ctx.shelf.covering = True
        try:
            responses = Checkout.dispatch(ctx, [proposal], scripted=True)
        finally:
            ctx.shelf.covering = False
        if not responses:
            chat.append({"role": "tool", "tool_call_id": call_id, "content": "Not run."})
            return False
        reply_id, text_body = responses[0]
        grew = len(ctx.shelf.archive.items) > count_before
        content = text_body
        if grew:
            content = cls.page_digest(ctx, ctx.shelf.archive.items[-1])
            _, marker, note = text_body.partition("\n[note] ")
            if marker and note and re.search(r"shopper|Meaning:|REFUSED", note):
                content += "\n[note] " + note
        chat.append({"role": "tool", "tool_call_id": reply_id, "content": content})
        return grew

    @staticmethod
    def _deal_rows(pools, most):
        queue = []
        while len(queue) < most and any(pools):
            for pool in pools:
                while pool and pool[0] in queue:
                    del pool[0]
                if pool and len(queue) < most:
                    queue.append(pool.pop(0))
        return queue

    @staticmethod
    def _trim_queue(ctx, queue, batch, floor):
        while queue and (1 if batch and len(queue) >= 2 else len(queue)) >= ctx.moves_left() - floor:
            queue.pop()

    @classmethod
    def _walk_pages(cls, ctx, plan, floor, page_key, report):
        book = Trail.book(ctx)
        walked = vars(ctx.shelf).setdefault("walked", set())
        ran = False
        for k, walk_key, n in plan:
            walked.add(k)
            walk = book.get(walk_key)
            if not walk:
                continue
            query = " ".join(Reading.free_text(ctx, walk["args"]))
            say = ("Only the cheapest will do, and this store returns search results in a fixed id order rather than "
                   f"by price: I read the next page of '{query}'.")
            read = 0
            for _ in range(n):
                token = Reading.next_page(ctx, walk)
                if token is None or ctx.moves_left() <= floor + 1 or Trail.no_time(ctx, 30):
                    break
                page_args = dict(walk["args"])
                page_args[page_key] = token
                if not cls.walk_call(ctx, page_args, say):
                    break
                read += 1
                ran = True
            report.append(f"'{query}': {read} more page(s) read, {len(walk.get('row_shelf') or {})} of its {walk['total']} matches seen")
        return ran

    @classmethod
    def _walk_lines(cls, ctx, plan, floor, page_key, report, lines):
        ran = False
        say = ("I also search the pick's product line (its maker plus the word that names the line): the line's other "
               "sizes and colours are listed separately and may be cheaper.")
        for k, _, _ in plan:
            if ctx.moves_left() <= floor + 2 or ctx.wire.done or ctx.wire.spare() < CLOSING + 30:
                break
            args = cls.line_query(ctx, k)
            if not args or not cls.walk_call(ctx, args, say):
                continue
            ran = True
            lines[k] = json.dumps({a: v for a, v in args.items() if a != page_key}, sort_keys=True, default=str)
            line = Trail.book(ctx).get(lines[k])
            for _ in range(2):
                if not line or line["total"] > 3 * max(1, line["size"]) or Reading.next_page(ctx, line) is None:
                    break
                if ctx.moves_left() <= floor + 3 or ctx.wire.done or ctx.wire.spare() <= CLOSING + 30:
                    break
                more = dict(line["args"])
                more[page_key] = Reading.next_page(ctx, line)
                if not cls.walk_call(ctx, more, "I read the next page of the line search."):
                    break
            report.append(f"the pick's line '{' '.join(Reading.free_text(ctx, args))}'")
        return ran

    @classmethod
    def run_walk(cls, ctx, plan, floor, ceil=None, repair=False):
        found = Reading.page_arg(ctx)
        if not (plan or ceil) or not found or not ctx.role("search"):
            return False
        page_key = found[0]
        report = []
        ran = cls._walk_pages(ctx, plan, floor, page_key, report)
        lines = {}
        ran = cls._walk_lines(ctx, plan, floor, page_key, report, lines) or ran
        ceil_rows = {}
        for k in dict.fromkeys(ceil or []):
            keep = floor if repair else max(floor, Intake.order_due(ctx) + 2)
            if ctx.moves_left() <= keep + 1 or ctx.wire.done or ctx.wire.spare() < CLOSING + 30:
                break
            got_rows = Coverage.ceiling_walk(ctx, k, floor, repair)
            if not got_rows:
                continue
            ran = True
            report.append(got_rows[0])
            ceil_rows[k] = list(got_rows[1])
        if not ran:
            return False
        batch = Lookup.batch_tool(ctx)
        most = batch[2] if batch else (CEIL_OPENS if ceil_rows else 2)
        pools = [cls.walk_rows(ctx, k, [lines.get(k, "")], most) for k, _, _ in plan]
        pools.extend(ceil_rows.values())
        queue = cls._deal_rows(pools, most)
        cls._trim_queue(ctx, queue, batch, floor)
        homes = set()
        for k in [k for k, _, _ in plan] + list(ceil_rows):
            homes |= cls.walk_home(ctx, k)
        ctx.cart.rank_checked = {mark for mark in ctx.cart.rank_checked if Judging.shelf_known(ctx, mark.split("|")[0]) not in homes}
        ctx.cart.order_vetted = False
        if queue:
            priced = ", ".join(f"{i} ({ctx.shelf.archive.price(i):g})" for i in queue)
            opened = ("The cheapest unread in-stock rows within the budget these searches showed, below each pick's price, are opened "
                      "next: " + priced + ". ")
        else:
            opened = "No unread in-stock row within the budget below the pick's price was shown. "
        summary = ("[cheapest walk] " + "; ".join(report) + ". " + opened
                   + "For each product, choose the lowest-priced listing whose own fields meet every requirement now in force; the "
                   "rank check runs again on the cart call and the order.")
        if queue:
            ctx.shelf.open_queue = queue
            ctx.shelf.open_recheck = True
            ctx.shelf.open_say = "The cheapest is required: I open the cheapest unread rows that the walk turned up."
            ctx.shelf.open_purpose, ctx.shelf.open_floor = "walk", floor
            Judging.open_named(ctx)
        Trail.append_note(ctx, summary, closing=not queue)
        Trail.add(ctx, "walk", plan=[[k, n] for k, _, n in plan], ceiling=list(ceil_rows), opened=queue)
        return True

    @classmethod
    def decline_fits(cls, ctx):
        audited = set(vars(ctx.cart).get("audit_fits") or []) - Trail.failed_ids(ctx)
        return (Trail.live_ids(ctx) | audited) - ctx.cart.refused_lines

    @classmethod
    def decline_lacking(cls, ctx):
        covered = {Judging.shelf_known(ctx, ident) for ident in cls.decline_fits(ctx)} - {""}
        if not ctx.needs.items():
            return [] if set(Reading.shelf_labels(ctx)) & covered else [None]
        lacking = []
        for k, home in enumerate(Lookup.home_shelves(ctx)):
            if home and not home & covered:
                lacking.append(k)
        return lacking

    @classmethod
    def decline_walks(cls, ctx):
        reqs = ctx.needs.props_left()

        def stated(args, k):
            count = 0
            for n in cls.walk_carries(ctx, args, k):
                if n < len(reqs) and not re.search(r"as shown|same as|the page|sold by|redirect|refer", reqs[n].lower()):
                    count += 1
            return count
        chosen = []
        for k in cls.decline_lacking(ctx):
            usable = [o for o in cls.walk_options(ctx, k) if stated(o[2]["args"], k)]
            usable.sort(key=lambda p: (-stated(p[2]["args"], k), -cls.walk_score(ctx, p[2]["args"], k), p[0], p[1]))
            for _, key, _ in usable:
                if key not in chosen:
                    chosen.append(key)
                    break
        return chosen[:2]

    @classmethod
    def row_pools(cls, ctx, homes, ids, below=None):
        cap = ctx.needs.spend_cap
        archive = ctx.shelf.archive
        shelf_by_id, sold = {}, {}

        def visit(d):
            ident = Reading.row_mark(ctx, d)
            if ident in archive.ids and Reading.shelf_of(ctx, d):
                shelf_by_id[ident] = Reading.shelf_of(ctx, d)
                if any(re.search(r"stock|avail", str(key).lower()) for key in d):
                    sold[ident] = Reading.sold_out(ctx, d)
        for rec in archive.items:
            TextKit.walk_dicts(rec, visit)

        def admitted(ident, where, n, qty):
            if where not in home_list[n] or ident in archive.read or ident in ctx.cart.refused_lines or sold.get(ident):
                return False
            if ids is not None and ident not in ids:
                return False
            cost = archive.price(ident)
            if not TextKit.is_number(cost) or (TextKit.is_number(cap) and cost * qty > cap):
                return False
            return below is None or cost < below[n]
        home_list = [home for home, _ in homes]
        pools = []
        for n, (_, qty) in enumerate(homes):
            pool = sorted((float(archive.price(ident)), ident) for ident, where in shelf_by_id.items() if admitted(ident, where, n, qty))
            pools.append([ident for _, ident in pool])
        return pools

    @classmethod
    def over_budget_rows(cls, ctx):
        try:
            rows, cap = ctx.needs.items(), ctx.needs.spend_cap
            if not rows or ctx.needs.priority() == "cheapest" or not TextKit.is_number(cap):
                return []
            homes = Lookup.home_shelves(ctx)
            fits = cls.decline_fits(ctx)
            archive = ctx.shelf.archive
            floors, shares = [], []
            for k, home in enumerate(homes):
                prices = [float(archive.price(i)) for i in fits if Judging.shelf_known(ctx, i) in home and TextKit.is_number(archive.price(i))]
                if not (home and prices):
                    return []
                floors.append(min(prices))
                shares.append(min(prices) * rows[k]["quantity"])
            if sum(shares) <= cap:
                return []
            pools = cls.row_pools(ctx, [(home, rows[k]["quantity"]) for k, home in enumerate(homes)], None, floors)
            ordered = [pools[k] for k in sorted(range(len(pools)), key=lambda k: -shares[k])]
            return ordered if any(ordered) else []
        except Exception as err:
            Trail.fault(ctx, "over_budget_error", err)
            return []

    @classmethod
    def decline_reads(cls, ctx):
        try:
            return cls._decline_hold(ctx)
        except Exception as err:
            Trail.fault(ctx, "decline_reads_error", err)
            return ""

    @classmethod
    def _decline_hold(cls, ctx):
        batch = Lookup.batch_tool(ctx)
        if vars(ctx.cart).get("decline_read_done") or ctx.wire.spare() < CLOSING + 60:
            return ""
        if not (ctx.role("details") or batch) or ctx.cart.trolley.units:
            return ""
        opens = 1 if batch else 2
        room = min(3, ctx.moves_left() - opens - 6)
        keys = []
        if room > 0 and Reading.page_arg(ctx) and ctx.role("search") and cls.id_store(ctx):
            keys = cls.decline_walks(ctx)
        cheaper = cls.over_budget_rows(ctx) if ctx.moves_left() > opens + 6 else []
        if not keys and not cheaper:
            return ""
        ctx.cart.decline_read_done = True
        ctx.cart.decline_queue = (keys, room, 6)
        said = []
        if keys:
            book = Trail.book(ctx)
            named = "; ".join(f"'{' '.join(Reading.free_text(ctx, book[key]['args']))}'" for key in keys)
            said.append("searches that carry required values matched more listings than were read (" + named + "), and this store lists "
                        "results in a fixed id order, not by fit or price, so fitting listings sit on unread pages as often as on "
                        f"read ones: the code reads up to {room} of those pages for you now and opens the cheapest in-stock rows "
                        "within the budget they show")
        if cheaper:
            said.append("the listings found for every product together cost more than the budget: the cheapest unread "
                        "in-stock rows priced below each product's pick are opened for you too")
        buy = "one line per product, at its quantity" if ctx.needs.items() else "the item"
        return ("Before a decline, " + "; ".join(said) + ". Judge those records against the requirements now in force by "
                "meaning; buy what fits (" + buy + "), and decline only if each one fails.")

    @classmethod
    def run_decline_reads(cls, ctx):
        keys, most_pages, floor = vars(ctx.cart).pop("decline_queue")
        found = Reading.page_arg(ctx)
        book = Trail.book(ctx)
        batch = Lookup.batch_tool(ctx)
        opens = 1 if batch else 2
        read = 0
        walked = []
        for key in (keys if found else []):
            walk = book.get(key)
            if not walk:
                continue
            walked.append(key)
            query = " ".join(Reading.free_text(ctx, walk["args"]))
            say = ("Before I decline, I read the next page of "
                   f"'{query}': this store returns results in a fixed id order, so fits can sit on unread pages.")
            while read < most_pages and ctx.moves_left() > floor + opens and not ctx.wire.done and not ctx.wire.broke \
                    and ctx.wire.spare() > CLOSING + 30:
                token = Reading.next_page(ctx, walk)
                if token is None:
                    break
                page_args = dict(walk["args"])
                page_args[found[0]] = token
                if not cls.walk_call(ctx, page_args, say):
                    break
                read += 1
        seen = set()
        for key in walked:
            seen.update((book.get(key) or {}).get("row_shelf") or {})
        rows = ctx.needs.items()
        homes = [(cls.walk_home(ctx, k), rows[k]["quantity"] if k is not None and k < len(rows) else ctx.needs.quantity())
                 for k in cls.decline_lacking(ctx)]
        pools = cls.row_pools(ctx, homes, seen) if walked else []
        pools += cls.over_budget_rows(ctx)
        queue = cls._deal_rows(pools, batch[2] if batch else 2)
        cls._trim_queue(ctx, queue, batch, floor)
        Trail.add(ctx, "decline_reads", pages=read, opened=queue)
        if queue and not ctx.wire.done:
            ctx.shelf.open_queue = queue
            ctx.shelf.open_say = "Before I decline, I open the cheapest unread in-budget rows that those pages and picks show."
            ctx.shelf.open_purpose, ctx.shelf.open_floor = "decline", floor
            Judging.open_named(ctx)
        else:
            Trail.append_note(ctx, "No unread in-stock row within the budget was shown there.")
        return read > 0 or bool(queue)

    @classmethod
    def walk_step(cls, ctx):
        try:
            return cls._next_walk(ctx)
        except Exception as err:
            Trail.fault(ctx, "walk_error", err)
            return False

    @classmethod
    def _next_walk(cls, ctx):
        if vars(ctx.cart).get("decline_queue"):
            return cls.run_decline_reads(ctx)
        queued = vars(ctx.shelf).pop("walk_queue", None)
        if queued:
            return cls.run_walk(ctx, *queued)
        again = Coverage.rearm_walk(ctx)
        if again is not None:
            return again
        if ctx.cart.orders_tried or ctx.needs.priority() != "cheapest" or ctx.wire.spare() < CLOSING + 60 or ctx.wire.done:
            return False
        if Coverage.unfound_product(ctx):
            return False
        opens = 1 if Lookup.batch_tool(ctx) else 2
        spare = ctx.moves_left()
        ceil = []
        for k in Coverage.cg_products(ctx):
            aimed = cls.walk_pick(ctx, k) or Coverage.bound_need(ctx, k)
            if aimed and Coverage.ceiling_ready(ctx, k) and Coverage.ceil_aim(ctx, k) is not None:
                ceil.append(k)
        if ceil and spare > Intake.order_due(ctx) + 2 + CEIL_CALLS + opens:
            used = vars(ctx.shelf).get("valued_by") or {}
            ceil.sort(key=lambda k: (bool(cls.walk_pick(ctx, k)), used.get(k, 0), -1 if k is None else k))
            return cls.run_walk(ctx, [], Intake.order_due(ctx) + 2, ceil[:1])
        if not ceil and spare > Intake.order_due(ctx) + 2 + CEIL_CALLS + opens and Coverage.ceil_seed(ctx):
            return True
        plan = cls.walk_plan(ctx, WALK_TOTAL)
        if not plan:
            return False
        pages = sum(n for _, _, n in plan)
        pages += sum(1 for k, _, _ in plan if cls.walk_pick(ctx, k))
        if spare > Intake.order_due(ctx) + 2 + pages + opens:
            return False
        plan = cls.walk_plan(ctx, min(WALK_TOTAL, spare - Intake.order_due(ctx) - 1 - opens))
        if not plan:
            return False
        ctx.shelf.walk_early = True
        return cls.run_walk(ctx, plan, Intake.order_due(ctx))
class Referencing:
    @staticmethod
    def _bare_prop(head):
        return re.split(r"\s*\(", str(head))[0]

    @staticmethod
    def _all_refs(ctx):
        return vars(ctx.pointers).get("refs") or []

    @classmethod
    def _open_refs(cls, ctx):
        return [ref for ref in cls._all_refs(ctx) if ref.get("status") == "open"]

    @staticmethod
    def _add_steps(ctx, since):
        ctx.pointers.ref_steps = vars(ctx.pointers).get("ref_steps", 0) + (ctx.wire.tick - since)

    @staticmethod
    def _toks(text):
        return set(RefKit.ref_toks(text))

    @staticmethod
    def _tag_of(head):
        tags = re.findall(r"\(([^)]*)\)", head)
        return tags[-1] if tags else None

    @classmethod
    def ref_register(cls, ctx):
        reading = ctx.needs.reading
        targets = reading.get("reference_targets")
        infos = list(filter(lambda entry: isinstance(entry, dict), targets)) if isinstance(targets, list) else []
        named = reading.get("references")
        listed = set(map(str, named or [])) if isinstance(named, list) else set()
        for req in list(ctx.needs.written):
            text = str(req)
            value = text.partition(":")[2]
            if RefKit.ref_text(value):
                cls.ref_add(ctx, text, infos, ctx.request, "request")
            elif text in listed and re.search(DIFFER_WORDS, value.lower()):
                cls.ref_add(ctx, text, infos, ctx.request, "request")

    @classmethod
    def ref_untracked(cls, ctx, refs, tracked):
        try:
            missing = []
            for req in refs:
                followed = False
                for entry in tracked:
                    if cls.ref_match(ctx, entry, req) and (entry.get("status") == "resolved" or entry.get("chain")):
                        followed = True
                        break
                if not followed:
                    missing.append(req)
            return missing
        except Exception:
            return refs

    @classmethod
    def ref_take(cls, ctx):
        try:
            for ref in filter(lambda r: r["status"] == "resolved", ctx.pointers.refs):
                if any(cls.ref_match(ctx, ref, req) and req != ref["now"] for req in ctx.needs.props_left()):
                    cls.ref_swap(ctx, ref, ref["now"])
            news = list(vars(ctx.needs).get("ref_news") or [])
            ctx.needs.ref_news = []
            for item in news:
                need = str(item.get("need") or "")
                opened = cls.ref_add(ctx, need, list(item.get("infos") or []), str(item.get("line") or ""), "reply")
                if opened is None:
                    continue
                reading = ctx.needs.reading
                if not isinstance(reading.get("references"), list):
                    reading["references"] = []
                if opened["req"] not in reading["references"]:
                    reading["references"].append(opened["req"])
        except Exception as err:
            Trail.fault(ctx, "ref_take_error", err)

    @classmethod
    def _duplicate(cls, ctx, req, prop, tag):
        for old in ctx.pointers.refs:
            if old["req"] == req:
                return True
            if PropKit.same_prop(old["prop"], prop) and PropKit.kind_words(old["tag"]) == PropKit.kind_words(tag):
                return True
        return False

    @classmethod
    def _matching_info(cls, ctx, infos, prop, tag):
        for cand in infos:
            if not PropKit.same_prop(str(cand.get("property") or ""), prop):
                continue
            loose = not tag or not cand.get("product") or not ctx.needs.items()
            if loose or PropKit.kind_words(cand.get("product")) & PropKit.kind_words(tag):
                return cand
        return None

    @classmethod
    def ref_add(cls, ctx, req, infos, source, origin):
        head, _, value = str(req).partition(":")
        prop = cls._bare_prop(head).strip()
        tag = (cls._tag_of(head) or "").strip()
        if not prop or TextKit.junk_key(prop) or cls._duplicate(ctx, req, prop, tag):
            return None
        info = cls._matching_info(ctx, infos, prop, tag)
        target = cls.ref_target(ctx, info, value, source, prop)
        if not target:
            Trail.add(ctx, "ref_skip", req=str(req)[:200])
            return None
        relation, tol, unit = RefKit.ref_relation(value)
        told = ""
        if isinstance(info, dict):
            told = str(info.get("relation") or "").lower()
            if relation == "same" and told == "within" and TextKit.is_number(info.get("tolerance")):
                relation, tol, unit = "within", float(info["tolerance"]), str(info.get("unit") or "")
        span = RefKit.ref_span(prop, source).lower()
        seller_text = str(target.get("seller") or "")
        cut = span.find(seller_text.lower()) if seller_text.strip() else -1
        differs = re.search(r"differ|not the same|exclu|apart|other than|unlike", told) or (cut > 0 and re.search(DIFFER_WORDS, span[:cut]))
        if relation == "same" and differs:
            relation, tol, unit = "differ", None, ""
        ref = dict(req=str(req), head=head.strip(), prop=prop, tag=tag, relation=relation, tol=tol, unit=unit,
                   target=target, first=dict(target), chain=[], seen={cls.ref_key(ctx, target)}, judged=set(),
                   queries=[], pages=0, hops=0, tries=0, fails=0, status="open", value="", goes_with=[],
                   now="", origin=origin, unindexed=False)
        ctx.pointers.refs.append(ref)
        Trail.add(ctx, "ref_open", req=str(req)[:200], target=target, relation=relation, origin=origin)
        return ref

    @staticmethod
    def _clean_list(raw):
        items = raw if isinstance(raw, list) else []
        cleaned = []
        for x in items:
            if isinstance(x, (str, int, float)) and not isinstance(x, bool) and str(x).strip():
                cleaned.append(str(x).strip())
        return cleaned

    @classmethod
    def ref_target(cls, ctx, info, value, source, prop):
        seller = kind = ""
        details = []
        if isinstance(info, dict):
            seller = str(info.get("seller") or "").strip()
            kind = str(info.get("kind") or "").strip()
            details = cls._clean_list(info.get("details"))

        def seller_ok(name):
            return cls.ref_seller_ok(ctx, name, source)
        parsed = None
        for text in (RefKit.ref_span(prop, source), value):
            got = RefKit.ref_parse(text, seller_ok)
            if got and seller_ok(got["seller"]):
                parsed = got
                break
        if not cls.ref_seller_ok(ctx, seller, source):
            if parsed is None:
                return None
            seller = parsed["seller"]
            kind = kind or parsed["kind"]
            details = details or parsed["details"]
        elif not details and parsed and RefKit.ref_name(parsed["seller"]) == RefKit.ref_name(seller):
            details = parsed["details"]
        written = cls._toks(source)
        kept = [d for d in dict.fromkeys(details) if cls._toks(d) & written]
        if parsed:
            kind = kind or parsed["kind"]
        return {"seller": seller, "kind": kind[:80], "details": kept[:6]}

    @classmethod
    def ref_seller_ok(cls, ctx, seller, source):
        name = str(seller or "").strip()
        if len(re.sub(r"[^A-Za-z]", "", name)) < 3:
            return False
        pattern = r"(?<![a-z0-9])" + re.escape(name.lower()) + r"(?:'s)?(?![a-z0-9])"
        if re.search(pattern, str(source or "").lower()) is None:
            return False
        kinds = set()
        for row in ctx.needs.items() or [{"product": str(ctx.needs.reading.get("product") or "")}]:
            kinds.update(PropKit.kind_words(row["product"]))
        search_shape = ctx.shelf.shapes.get(ctx.role("search") or "") or {}
        for pspec in (search_shape.get("properties") or {}).values():
            if not isinstance(pspec, dict) or not isinstance(pspec.get("enum"), list):
                continue
            for choice in pspec["enum"]:
                kinds.update(PropKit.kind_words(choice))
        own = PropKit.kind_words(name)
        return bool(own) and not own.issubset(kinds)

    @classmethod
    def ref_key(cls, ctx, tgt):
        tokens = set()
        for detail in tgt.get("details") or []:
            tokens.update(RefKit.ref_toks(detail))
        return RefKit.ref_name(tgt.get("seller") or ""), tuple(sorted(tokens))

    @classmethod
    def ref_desc_of(cls, ctx, tgt):
        text = f"the {tgt.get('kind') or 'listing'} sold by {tgt.get('seller')}"
        if tgt.get("details"):
            text += " (" + "; ".join(tgt.get("details") or []) + ")"
        return text

    @classmethod
    def ref_desc(cls, ctx, ref):
        return cls.ref_desc_of(ctx, ref["target"])

    @classmethod
    def ref_index(cls, ctx):
        archive = ctx.shelf.archive
        memo = vars(ctx.pointers).get("ref_memo")
        if memo and memo[0] == len(archive.items):
            return memo[1]
        rows, seen_at = {}, {}
        position = [0]

        def visit(d):
            ident = Reading.row_mark(ctx, d)
            if ident in archive.ids and (len(d) >= 4 or any(isinstance(v, dict) and len(v) >= 2 for v in d.values())):
                rows.setdefault(ident, {}).update(d)
                seen_at[ident] = position[0]
        for n, rec in enumerate(archive.items):
            position[0] = n
            TextKit.walk_dicts(rec, visit)
        ctx.pointers.ref_memo = (len(archive.items), rows)
        ctx.pointers.ref_seen_at = seen_at
        return rows

    @staticmethod
    def _seller_field(key, val, learned):
        low = str(key).lower()
        if not isinstance(val, str) or not val.strip() or len(val) > 60 or "id" in low:
            return False
        if not (re.search(SELLER_KEYS, low) or str(key) in learned):
            return False
        return not re.search(r"note|text|desc|comment|detail|info|message|section", low)

    @classmethod
    def ref_sellers(cls, ctx, row):
        learned = cls.seller_keys(ctx)
        names = {RefKit.ref_name(val) for key, val in row.items() if cls._seller_field(key, val, learned)}
        names.discard("")
        return names

    @staticmethod
    def _capital_runs(text):
        found = set()
        for sentence in re.split(r"(?<=[.!?;])\s+", text):
            run = []
            for word in re.findall(r"[A-Za-z][A-Za-z'-]*", sentence)[1:] + [""]:
                if word[:1].isupper() and re.search(r"[a-z]", word):
                    run.append(word)
                    continue
                for start in range(len(run)):
                    for stop in range(start + 1, min(len(run), start + 3) + 1):
                        found.add(RefKit.ref_name(" ".join(run[start:stop])))
                run = []
        return found

    @classmethod
    def seller_keys(cls, ctx):
        try:
            return cls._learn_seller_keys(ctx)
        except Exception:
            return set()

    @classmethod
    def _learn_seller_keys(cls, ctx):
        names = set()
        for ref in cls._all_refs(ctx):
            for tgt in (ref.get("target") or {}, ref.get("first") or {}):
                if isinstance(tgt, dict):
                    names.add(RefKit.ref_name(tgt.get("seller")))
        names.discard("")
        mark = (len(ctx.shelf.archive.items), tuple(sorted(names)))
        memo = vars(ctx.shelf).get("seller_memo")
        if memo and memo[0] == mark:
            return memo[1]
        rows = cls.ref_index(ctx)
        noted, vocab = set(), set()
        for row in rows.values():
            for val in row.values():
                if isinstance(val, dict):
                    vocab.update(RefKit.ref_name(x) for x in val.values() if isinstance(x, str))
            noted.update(cls._capital_runs(cls.ref_texts(ctx, row, desc=False)))
        noted = set(w for w in noted if len(w) >= 4) - vocab
        skip = (r"title|heading|name|label|caption|categor|shelf|department|aisle|note|text|desc|comment|detail|info|"
                r"message|section|url|image|summary|overview")
        keys = set()
        for row in rows.values():
            for key, val in row.items():
                low = str(key).lower()
                if not isinstance(val, str) or not 0 < len(val) <= 60 or "id" in low or re.search(skip, low):
                    continue
                reduced = RefKit.ref_name(val)
                if reduced in names or (val.strip()[:1].isupper() and len(val.split()) <= 3 and reduced in noted):
                    keys.add(str(key))
        ctx.shelf.seller_memo = (mark, keys)
        return keys

    @classmethod
    def ref_texts(cls, ctx, row, desc=True):
        notes, other = [], []
        for key, val in row.items():
            name = str(key).lower()
            if not isinstance(val, str) or len(val.split()) < 3 or name == "id" or name.endswith("id"):
                continue
            if re.search(r"title|heading|headline|name|label|caption|categor|shelf|department|url|image", name):
                continue
            (other if re.search(r"desc|overview|summary|about", name) else notes).append(val)
        if notes:
            return "\n".join(notes)
        return "\n".join(other) if desc else ""

    @classmethod
    def ref_parts(cls, ctx, row):
        notes = re.sub(r"\b(?:not|no|never|without|nor)\b(?:\W+[A-Za-z0-9.:/&'-]+){1,3}", " ", cls.ref_texts(ctx, row, desc=False), flags=re.I)
        names = []
        values = [Lookup.title_of(ctx, row), notes]
        maps = [v for v in row.values() if isinstance(v, dict)]
        if not maps:
            flat = {}
            for k, v in row.items():
                if isinstance(v, (str, int, float)) and not isinstance(v, bool) \
                        and not re.search(r"id$|title|shop|seller|brand|price|amount|desc|overview|note", str(k).lower()):
                    flat[k] = v
            maps = [flat]
        for fields in maps:
            for key, val in fields.items():
                names.append(re.sub(r"([a-z])([A-Z])", r"\1 \2", str(key)).replace("_", " "))
                if isinstance(val, (str, int, float)) and not isinstance(val, bool):
                    values.append(str(val))
        return cls._toks(" ".join(names)), cls._toks(" | ".join(values))

    @staticmethod
    def _own_values(row):
        values = []
        for v in row.values():
            if isinstance(v, dict):
                values.extend(str(x) for x in v.values() if isinstance(x, (str, int, float)))
        return values

    @classmethod
    def ref_shows(cls, ctx, row, detail, parts=None):
        keys, vals = parts or cls.ref_parts(ctx, row)
        wanted = set(t for t in RefKit.ref_toks(detail) if len(t) >= 3 and not re.search(r"\d", t)) - keys
        if wanted and wanted.isdisjoint(vals):
            return False
        figures = RefKit.ref_figs(detail)
        if not figures:
            return True
        written = RefKit.ref_figs(" | ".join([Lookup.title_of(ctx, row)] + cls._own_values(row)))
        for fig, unit in figures:
            same_unit = [f for f, u in written if u == unit]
            if same_unit and not any(abs(f - fig) < 1e-9 for f in same_unit):
                return False
        return True

    @classmethod
    def ref_field(cls, ctx, row, prop):
        maps = [val for val in row.values() if isinstance(val, dict)]
        for test in (PropKit.strict_prop, PropKit.same_prop):
            for fields in maps:
                hit = next((str(x).strip() for key, x in fields.items()
                            if isinstance(x, (str, int, float)) and not isinstance(x, bool) and str(x).strip() and test(key, prop)), None)
                if hit is not None:
                    return hit
        return ""

    @classmethod
    def ref_ids_add(cls, ctx, ids):
        known = set(vars(ctx.pointers).get("ref_ids") or set())
        known.update(ids)
        ctx.pointers.ref_ids = known

    @classmethod
    def ref_foreign(cls, ctx):
        try:
            ids = vars(ctx.pointers).get("ref_ids") or set()
            if not ids:
                return set()
            shelves = Lookup.home_shelves(ctx)
            if not shelves or not all(shelves):
                return set()
            homes = set().union(*shelves)
            rows = cls.ref_index(ctx)
            in_cart = set(ctx.cart.trolley.units)
            foreign = set()
            for ident in ids:
                if ident in in_cart:
                    continue
                shelf = Reading.shelf_of(ctx, rows.get(ident) or {})
                if shelf and shelf not in homes:
                    foreign.add(ident)
            return foreign
        except Exception:
            return set()

    @classmethod
    def ref_cands(cls, ctx, ref):
        target = ref["target"]
        name = RefKit.ref_name(target["seller"])
        kinds = PropKit.kind_words(target["kind"])
        rows = cls.ref_index(ctx)
        seen_at = vars(ctx.pointers).get("ref_seen_at") or {}
        scored = []
        for ident, row in rows.items():
            if name not in cls.ref_sellers(ctx, row):
                continue
            blurb = [str(v) for k, v in row.items() if isinstance(v, str) and re.search(r"desc|overview|summary", str(k).lower())]
            hay = cls._toks(Lookup.title_of(ctx, row) + " " + " ".join(blurb))
            shown = len([d for d in target["details"] if cls._toks(d) and cls._toks(d) <= hay])
            bonus = 2 if kinds & PropKit.kind_words(Reading.shelf_of(ctx, row)) else 0
            scored.append((-bonus - shown, -seen_at.get(ident, 0), ident, shown))
        scored.sort()
        ref["shows"] = dict((entry[2], entry[3]) for entry in scored)
        return [entry[2] for entry in scored]

    @classmethod
    def ref_modes(cls, ctx, needed, labels):
        groups = {}
        for n, walk in enumerate(Trail.book(ctx).values()):
            args = walk.get("args") if isinstance(walk.get("args"), dict) else {}
            texts = Reading.free_text(ctx, args)
            text_keys = [k for k, v in args.items() if isinstance(v, str) and v in texts and v.strip()]
            if len(texts) != 1 or len(text_keys) != 1:
                continue
            other = {}
            for k, v in args.items():
                if k != text_keys[0] and (k in needed or not (isinstance(v, str) and v.strip().lower() in labels)):
                    other[k] = v
            mode = json.dumps(other, sort_keys=True, default=str)
            if mode not in groups:
                groups[mode] = {"tpl": (other, text_keys[0]), "seen": [], "at": n}
            stems = set(Reading.stem(ctx, w) for w in re.findall(r"[a-z0-9]+", texts[0].lower()))
            groups[mode]["seen"].append((stems, tuple(walk.get("first") or ())))
        ranked, blind = [], set()
        for mode, group in groups.items():
            seen = group["seen"]
            ignores = False
            for i in range(len(seen)):
                words_a, first_a = seen[i]
                for words_b, first_b in seen[i + 1:]:
                    if len(first_a) >= 2 and first_a == first_b and not (words_a & words_b):
                        ignores = True
                if ignores:
                    break
            if ignores:
                blind.add(mode)
                continue
            distinct = set(tuple(sorted(w)) for w, _ in seen)
            ranked.append((-len(distinct), group["at"], group["tpl"]))
        ranked.sort(key=lambda entry: entry[:2])
        return [entry[2] for entry in ranked], blind

    @classmethod
    def ref_template(cls, ctx):
        shape = ctx.shelf.shapes.get(ctx.role("search") or "") or {}
        required = shape.get("required")
        needed = set(required or []) if isinstance(required, list) else set()
        labels = set(Reading.shelf_of(ctx, row) for row in cls.ref_index(ctx).values())
        labels.discard("")
        modes, blind = cls.ref_modes(ctx, needed, labels)
        template = modes[0] if modes else None
        paging = Reading.page_arg(ctx)

        def keepable(k, v):
            return k in needed or not (isinstance(v, str) and v.strip().lower() in labels)
        if template is None:
            template = cls._logged_template(ctx, paging, blind, keepable)
        if template:
            other, key = template
            return dict((k, v) for k, v in other.items() if keepable(k, v)), key
        props = shape.get("properties") if isinstance(shape.get("properties"), dict) else {}
        token = Reading.token_key(ctx)
        texts = [k for k, p in props.items() if isinstance(p, dict) and p.get("type") == "string" and not p.get("enum") and k != token]
        if len(texts) == 1 and not (needed - {texts[0]}):
            return {}, texts[0]
        return None

    @classmethod
    def _logged_template(cls, ctx, paging, blind, keepable):
        search = ctx.role("search")
        env_entries = [entry for entry in reversed(ctx.log) if entry.get("kind") == "env"]
        for entry in env_entries:
            for act in reversed(entry.get("actions") or []):
                if act.get("name") != search or not isinstance(act.get("args"), dict):
                    continue
                args = act.get("args")
                texts = [k for k, v in args.items() if isinstance(v, str) and v.strip() and v in Reading.free_text(ctx, args)]
                if len(texts) != 1:
                    continue
                other = {k: v for k, v in args.items() if k != texts[0] and not (paging and k == paging[0])}
                kept = {k: v for k, v in other.items() if keepable(k, v)}
                if json.dumps(kept, sort_keys=True, default=str) not in blind:
                    return other, texts[0]
        return None

    @classmethod
    def ref_words(cls, ctx, ref):
        field_words = set()
        for row in cls.ref_index(ctx).values():
            for val in filter(lambda v: isinstance(v, dict), row.values()):
                for key in val:
                    field_words.update(re.findall(r"[a-z]+", re.sub(r"([a-z])([A-Z])", r"\1 \2", str(key)).lower()))
        for req in ctx.needs.written + ctx.needs.uncovered:
            field_words.update(re.findall(r"[a-z]+", cls._bare_prop(str(req).partition(":")[0]).lower()))
        target = ref["target"]
        seller = set(re.findall(r"[a-z0-9]+", str(target["seller"]).lower()))
        joined, figures, words = [], [], []
        for detail in target["details"]:
            previous = ""
            for whole in str(detail).lower().split():
                after_figure = re.fullmatch(r"\d+(?:[.,]\d+)?", previous) is not None
                previous = whole
                bare = whole.strip(".,;:!?()[]{}\"'")
                for w in re.findall(r"[a-z0-9]+", whole):
                    digit = re.search(r"\d", w)
                    if w in LINK_WORDS or w in seller or (after_figure and not digit):
                        continue
                    if digit and re.search(r"[a-z]", w):
                        joined.append(w)
                    elif w.isdigit():
                        if len(w) > 1:
                            figures.append(w)
                    elif len(w) >= 3 and w == bare and w not in field_words and w.rstrip("s") not in field_words:
                        words.append(w)
        words.sort(key=len, reverse=True)
        store_names = cls.store_keys(ctx)
        split = [[w for w in re.findall(r"[a-z0-9]+", str(d).lower()) if w not in LINK_WORDS] for d in target["details"]]
        as_pairs = ref.get("hops", 0) == 0 and split and all(len(parts) > 1 for parts in split)
        lasts = []
        for parts in split:
            if len(parts) > 1 and parts[-1] in words:
                if as_pairs or any(" ".join(parts[:j]) in store_names for j in range(1, len(parts))):
                    lasts.append(parts[-1])
        ordered = [w for w in lasts if w in words] + [w for w in words if w not in lasts]
        return list(dict.fromkeys(joined + figures + ordered))

    @classmethod
    def store_keys(cls, ctx):
        mark = len(ctx.shelf.archive.items)
        memo = vars(ctx.shelf).get("store_keys_memo")
        if memo and memo[0] == mark:
            return memo[1]
        names = set()

        def visit(d):
            for val in d.values():
                if isinstance(val, dict) and len([x for x in val.values() if isinstance(x, (str, int, float))]) >= 2:
                    for key in val:
                        name = " ".join(re.findall(r"[a-z0-9]+", re.sub(r"([a-z])([A-Z])", r"\1 \2", str(key)).lower()))
                        if name:
                            names.add(name)
        for rec in ctx.shelf.archive.items:
            TextKit.walk_dicts(rec, visit)
        ctx.shelf.store_keys_memo = (mark, names)
        return names

    @classmethod
    def ref_plans(cls, ctx, ref):
        width = max(2, ctx.shelf.lead_words or 3)
        target = ref["target"]
        seller = " ".join(re.findall(r"[A-Za-z0-9]+", re.sub(r"'s\b", "", str(target["seller"]))))
        words = cls.ref_words(ctx, ref)
        kind = [w for w in re.findall(r"[a-z]{3,}", str(target["kind"]).lower()) if w not in LINK_WORDS][-1:]
        plans = []
        if ref.get("unindexed"):
            if words or kind:
                plans.append(" ".join((words[:1] + kind)[:width]))
        else:
            if words and len(seller.split()) < width:
                plans.append(seller + " " + words[0])
            plans.append(seller)
        return [plan for plan in plans if plan.strip() and plan.lower() not in ref["queries"]]

    @classmethod
    def ref_page(cls, ctx, ref):
        found = Reading.page_arg(ctx)
        if not found or ref["pages"] >= 3:
            return None
        name = RefKit.ref_name(ref["target"]["seller"])
        if not name:
            return None
        for walk in reversed(list(Trail.book(ctx).values())):
            if name in RefKit.ref_name(" ".join(Reading.free_text(ctx, walk["args"]))):
                token = Reading.next_page(ctx, walk)
                if token is not None:
                    args = dict(walk["args"])
                    args[found[0]] = token
                    return args
        return None

    @classmethod
    def ref_room(cls, ctx):
        steps = vars(ctx.pointers).get("ref_steps", 0)
        waiting = cls._open_refs(ctx)
        blind = all(r.get("relation") == "differ" and not r.get("chain") for r in waiting)
        if waiting and blind and steps >= REF_DIFFER_BLIND:
            return False
        floor = cls.ref_floor(ctx)
        open_now = cls._open_refs(ctx)
        share, least = ctx.route()["ref_share"], 8
        if open_now and all(r.get("relation") == "differ" for r in open_now):
            share, least = REF_DIFFER_STEPS, 6
        cap = max(least, int(share * ctx.budget.step_cap))
        if steps >= cap or ctx.wire.spare() <= CLOSING + 60 or ctx.wire.spent_out() or ctx.wire.done or ctx.wire.broke:
            return False
        if cls.ref_waits(ctx, floor):
            return False
        return ctx.moves_left() > floor or cls.ref_hop_room(ctx)

    @classmethod
    def ref_floor(cls, ctx):
        reserve = min(FIRST_ORDER_DUE, ctx.budget.step_cap // 4 + 1)
        return max(max(FIRST_ORDER_BY, reserve) + 10, Intake.order_due(ctx) + 7)

    @classmethod
    def ref_product(cls, ctx, ref):
        if ctx.needs.items() and ref.get("tag"):
            return Asking.product_of(ctx, f"x ({ref['tag']})")
        return None

    @classmethod
    def ref_waits(cls, ctx, floor=None):
        rows = ctx.needs.items()
        opens = cls._open_refs(ctx)
        if not (rows and opens):
            return False
        if any(r.get("relation") != "differ" or not r.get("chain") for r in opens):
            return False
        limit = cls.ref_floor(ctx) if floor is None else floor
        if ctx.moves_left() <= limit + 4:
            return False
        try:
            owners = set(cls.ref_product(ctx, r) for r in opens)
            lacking = []
            for k in range(len(rows)):
                if not any(entry["fit"] for entry in Lookup.ed_pool(ctx, k)):
                    lacking.append(k)
        except Exception as err:
            Trail.fault(ctx, "ref_wait_error", err)
            return False
        known = owners - {None}
        rest = [k for k in lacking if k not in known]
        waits = len(rest) >= 2 if None in owners else len(rest) > 0
        if waits and vars(ctx.pointers).get("ref_wait_said") != tuple(rest):
            ctx.pointers.ref_wait_said = tuple(rest)
            Trail.add(ctx, "ref_wait", lacking=rest)
        return waits

    @classmethod
    def ref_hop_room(cls, ctx):
        if ctx.moves_left() <= Intake.order_due(ctx) + 2:
            return False
        steps = vars(ctx.pointers).get("ref_steps", 0)
        for ref in cls._all_refs(ctx):
            target = ref.get("target") or {}
            if ref.get("status") != "open" or ref.get("hops", 0) < 1:
                continue
            if not str(target.get("seller") or "").strip() or not str(target.get("kind") or "").strip() or not target.get("details"):
                continue
            start = ref.get("late_from")
            if start is None:
                if ref.get("queries") or not cls.ed_others_verified(ctx, ref):
                    continue
                ref["late_from"] = steps
                Trail.add(ctx, "ref_late", prop=ref.get("prop"), to=str(target.get("seller"))[:80])
                return True
            if steps - start < 3:
                return True
        return False

    @classmethod
    def ed_others_verified(cls, ctx, ref):
        rows = ctx.needs.items()
        if not rows:
            return True
        try:
            bare = []
            for k in range(len(rows)):
                if not any(e["verified"] and e["fit"] for e in Lookup.ed_pool(ctx, k)):
                    bare.append(k)
        except Exception:
            return False
        own = cls.ref_product(ctx, ref)
        if own is None:
            return len(bare) < 2
        return not [k for k in bare if k != own]

    @classmethod
    def ref_calls_ok(cls, ctx):
        within = vars(ctx.pointers).get("ref_calls", 0) < REF_CALLS and ctx.wire.spare() > CLOSING + 40 and not ctx.wire.spent_out()
        return within and ctx.wire.spend < 0.8 * SPEND_CAP * vars(ctx.wire).get("scale", 1.0)

    @classmethod
    def ref_match(cls, ctx, ref, req):
        if str(req) in (ref["req"], ref.get("now")):
            return True
        head, _, value = str(req).partition(":")
        pointing = RefKit.ref_text(value) or (ref.get("relation") == "differ" and re.search(DIFFER_WORDS, value.lower()))
        if not (pointing and PropKit.same_prop(cls._bare_prop(head), ref["prop"])):
            return False
        tag = cls._tag_of(head)
        if tag is None or not ref["tag"]:
            return True
        return bool(PropKit.kind_words(tag) & PropKit.kind_words(ref["tag"]))

    @classmethod
    def ref_live(cls, ctx, ref):
        def same_property(req):
            return PropKit.same_prop(cls._bare_prop(str(req).partition(":")[0]), ref["prop"])
        if any(cls.ref_match(ctx, ref, req) or same_property(req) for req in ctx.needs.props_left()):
            return True
        return not any(map(same_property, ctx.needs.let_go))

    @classmethod
    def ref_step(cls, ctx):
        if not vars(ctx.pointers).get("refs") or ctx.wire.done or ctx.wire.broke:
            return False
        try:
            return cls.ref_turn(ctx)
        except Exception as err:
            Trail.fault(ctx, "ref_error", err)
            for ref in ctx.pointers.refs:
                if ref["status"] == "open":
                    ref["status"], ref["why"] = "stuck", "the code's resolver failed"
            return False

    @classmethod
    def ref_turn(cls, ctx):
        opens = []
        for ref in ctx.pointers.refs:
            if ref["status"] == "open" and not cls.ref_live(ctx, ref):
                ref["status"] = "dropped"
                Trail.add(ctx, "ref_dropped", req=ref["req"][:200])
            if ref["status"] == "open":
                opens.append(ref)
        if not opens:
            return False
        acted = False
        tick_before = ctx.wire.tick
        for ref in sorted(opens, key=lambda r: r["hops"]):
            if cls.ref_judge(ctx, ref):
                acted = True
        if ctx.wire.tick > tick_before:
            return True
        opens = [ref for ref in opens if ref["status"] == "open"]
        if not opens or not cls.ref_room(ctx):
            return acted
        batch = Lookup.batch_tool(ctx)
        most = batch[2] if batch else 2
        pools = [cls._ready_pool(ctx, ref, most) for ref in opens]
        template = cls.ref_template(ctx)
        empty = []
        if template:
            empty = [ref for ref, pool in zip(opens, pools) if not pool and cls.ref_plans(ctx, ref)]
        queue = Walking._deal_rows(pools, most)
        needy = empty if queue and len(queue) < most else []
        if queue and not needy:
            return cls.ref_read(ctx, queue, opens) or acted
        if not template:
            return acted
        order = needy or sorted(opens, key=lambda r: (r["hops"], len(r["queries"])))
        for ref in order:
            plans = cls.ref_plans(ctx, ref)
            if plans:
                ref["queries"].append(plans[0].lower())
                if Walking.ran_query(ctx, plans[0]):
                    return True
                query = dict(template[0])
                query[template[1]] = plans[0]
                return cls.ref_search(ctx, ref, query) or acted
            args = cls.ref_page(ctx, ref)
            if args:
                ref["pages"] += 1
                return cls.ref_search(ctx, ref, args) or acted
        for ref in opens:
            unjudged = [i for i in cls.ref_cands(ctx, ref) if i not in ref["judged"]]
            if unjudged or cls.ref_plans(ctx, ref) or cls.ref_page(ctx, ref):
                continue
            cls.ref_stop(ctx, ref, "the listing it points at was not found in the store's search")
            acted = True
        return acted

    @classmethod
    def _ready_pool(cls, ctx, ref, most):
        pool = [i for i in cls.ref_cands(ctx, ref) if i not in ctx.shelf.archive.read and i not in ref["judged"]]
        if ref["queries"] or not cls.ref_plans(ctx, ref) or len(pool) <= most:
            return pool
        shows = ref.get("shows") or {}
        return pool if pool and shows.get(pool[0]) else []

    @classmethod
    def ref_read(cls, ctx, queue, opens):
        batch = Lookup.batch_tool(ctx) if len(queue) > 1 else None
        before = ctx.wire.tick
        heads = ", ".join(dict.fromkeys(ref["head"] for ref in opens))
        responses = Judging.open_reads(ctx, queue, f"The request takes {heads} from a different listing, so I open the listings that could be that one.", batch)
        dropped = False
        if batch and responses and not vars(ctx.cart).get("batch_proven"):
            if Lookup.batch_ok(ctx, vars(ctx.shelf).get("last_reads") or [], queue):
                ctx.cart.batch_proven = True
            else:
                ctx.cart.batch_spec = None
                dropped = True
                Trail.add(ctx, "batch_dropped", tool=batch[0])
        if responses:
            Asking.post_replies(ctx, responses)
        cls._add_steps(ctx, before)
        cls.ref_ids_add(ctx, queue)
        Trail.add(ctx, "ref_read", ids=queue)
        if not dropped:
            unread = [ident for ident in queue if ident not in ctx.shelf.archive.read]
            for ref in opens:
                ref["judged"].update(unread)
        return ctx.wire.tick > before

    @classmethod
    def ref_search(cls, ctx, ref, args):
        before = ctx.wire.tick
        got = Walking.walk_call(ctx, args, f"The request takes {ref['head']} from a different listing ({cls.ref_desc(ctx, ref)}), so I search for that one.")
        cls._add_steps(ctx, before)
        rows = []
        if got:
            archive = ctx.shelf.archive
            TextKit.walk_dicts(archive.items[-1], lambda d: rows.append(d) if Reading.row_mark(ctx, d) in archive.ids and len(d) >= 4 else None,
                               ceiling=3)
        name = RefKit.ref_name(ref["target"]["seller"])
        text = " ".join(Reading.free_text(ctx, args))
        if rows and not any(name in cls.ref_sellers(ctx, row) for row in rows) and RefKit.ref_name(text) == name:
            ref["unindexed"] = True
        Trail.add(ctx, "ref_search", prop=ref["prop"], query=text[:120], rows=len(rows))
        return ctx.wire.tick > before

    @classmethod
    def ref_judge(cls, ctx, ref):
        target = ref["target"]
        archive = ctx.shelf.archive
        cands = [i for i in cls.ref_cands(ctx, ref) if i in archive.read and i not in ref["judged"]]
        if not cands:
            return False
        rows = cls.ref_index(ctx)
        seller_read = [i for i in cls.ref_cands(ctx, ref) if i in archive.read]
        parts = dict((i, cls.ref_parts(ctx, rows.get(i) or {})) for i in set(cands) | set(seller_read))
        loose = ref.setdefault("loose", set())
        details = target["details"]

        def shows(i, d):
            return cls.ref_shows(ctx, rows.get(i) or {}, d, parts[i])
        if len(seller_read) >= 4:
            loose.update(d for d in details if not any(shows(i, d) for i in seller_read))

        def words_ok(i):
            return all(shows(i, d) for d in details if d not in loose)
        full = {}
        for i in cands:
            hay = parts[i][0] | parts[i][1]
            full[i] = len([d for d in details if cls._toks(d) and cls._toks(d) <= hay])
        ranked = sorted(cands, key=lambda i: (not words_ok(i), -full[i], cands.index(i)))
        if not words_ok(ranked[0]) and (cls.ref_room(ctx) or cls.ref_waits(ctx)):
            more = [i for i in cls.ref_cands(ctx, ref) if i not in archive.read and i not in ref["judged"]]
            if more or cls.ref_plans(ctx, ref) or cls.ref_page(ctx, ref):
                return False
        good = [i for i in ranked if words_ok(i)]
        pick = good[:4] or ranked[:4]
        exact = [i for i in pick if details and full[i] == len(details)]
        to_note = exact if len(exact) == 1 else ([i for i in pick if words_ok(i)][:2] or pick[:1])
        missing = cls.ref_notes_missing(ctx, ref, to_note)
        return cls._settle(ctx, ref, target, cands, pick, exact, missing, rows, words_ok)

    @classmethod
    def _settle(cls, ctx, ref, target, cands, pick, exact, missing, rows, words_ok):
        if missing:
            if cls.ref_room(ctx) and cls.ref_notes_read(ctx, ref, missing):
                return True
            if cls.ref_waits(ctx):
                return False
            if not cls.ref_room(ctx) and ref.get("relation") == "differ":
                ref["judged"].update(pick)
                cls.ref_stop(ctx, ref, "the notes of the listing it points at could not be read in the steps left")
                return True
        # the pointed listing is also shown with the page tool before its value counts: a page can carry a note the
        # details tool leaves out
        if len(exact) == 1 and cls.ref_page_unseen(ctx, ref, exact[0]) and cls.ref_room(ctx):
            view = ctx.role("view")
            ref.setdefault("viewed", set()).add(exact[0])
            before = ctx.wire.tick
            say = f"The request takes {ref['head']} from {exact[0]}: I open its page as well, because a page can say more than its details."
            responses = Judging.open_reads(ctx, [exact[0]], say, None, view)
            if responses:
                Asking.post_replies(ctx, responses)
            cls._add_steps(ctx, before)
            if ctx.wire.tick > before:
                return True
        if len(exact) == 1 and cls.ref_plain(ctx, ref, exact[0]):
            ref["judged"].update(pick)
            cls.ref_ids_add(ctx, pick)
            cls.ref_resolve(ctx, ref, exact[0], cls.ref_field(ctx, rows.get(exact[0]) or {}, ref["prop"]), [])
            return True
        if not cls.ref_calls_ok(ctx):
            ref["judged"].update(pick)
            cls.ref_stop(ctx, ref, "no hop checks are left")
            return True
        cls.ref_apply(ctx, ref, pick, cls.ref_ask(ctx, ref, pick, ""))
        if ref["status"] == "open" and ref["target"] is target:
            ref["judged"].update(i for i in cands if not words_ok(i))
        return True

    @classmethod
    def _field_tokens(cls, ctx, row, keys):
        texts = [Lookup.title_of(ctx, row), Reading.shelf_of(ctx, row)] + cls._own_values(row)
        return keys | cls._toks(" | ".join(texts))

    @classmethod
    def ref_plain(cls, ctx, ref, ident):
        row = cls.ref_index(ctx).get(ident) or {}
        value = cls.ref_field(ctx, row, ref["prop"])
        notes = cls.ref_texts(ctx, row, desc=False)
        if not value or cls.ref_names_prop(ctx, ref["prop"], notes) or cls._toks(value) & cls._toks(notes):
            return False
        own = cls.ref_sellers(ctx, row)
        elsewhere = set()
        for other in cls.ref_index(ctx).values():
            elsewhere |= cls.ref_sellers(ctx, other)
        elsewhere -= own
        fields = cls._field_tokens(ctx, row, cls.ref_parts(ctx, row)[0])
        for sentence in re.split(r"(?<=[.!?;])\s+", notes):
            for word in re.findall(r"[A-Za-z][A-Za-z'-]*", sentence)[1:]:
                low = RefKit.ref_name(word)
                capital = word[:1].isupper() and len(low) >= 4 and low not in own and low not in fields
                if low in elsewhere or capital:
                    return False
        return True

    @classmethod
    def ref_notes_tool(cls, ctx):
        learned = Lookup.notes_tool(ctx)
        if learned:
            return learned
        details = ctx.role("details")
        shelf = vars(ctx.shelf)
        if (shelf.get("note_keys") or {}).get(details or ""):
            return None
        tool_keys = shelf.get("tool_keys") or {}
        text_keys = shelf.get("text_keys") or {}
        detail_keys = set(tool_keys.get(details or "") or set())
        row_keys = set(shelf.get("row_keys") or set())
        caps = shelf.get("tool_caps") or {}
        others = [tool for tool in Lookup.lookup_tools(ctx) if tool != details]
        extra = []
        for tool in others:
            fresh = set(tool_keys.get(tool) or set()) - detail_keys - row_keys
            if fresh & set(text_keys.get(tool) or set()):
                extra.append(tool)
        if extra:
            return max(extra, key=lambda tool: ((caps.get(tool) or {}).get("calls", 0), tool))
        for tool in (ctx.role("view"), Lookup.stock_tool(ctx)):
            if tool in others and not (caps.get(tool) or {}).get("id", 0):
                return tool
        return None

    @classmethod
    def ref_page_unseen(cls, ctx, ref, ident):
        view = ctx.role("view")
        if not view or view == ctx.role("details") or view not in ctx.shelf.shapes:
            return False
        seen = set((vars(ctx.shelf).get("read_with") or {}).get(view) or set())
        return ident not in seen and ident not in ref.get("viewed", set())

    @classmethod
    def ref_notes_missing(cls, ctx, ref, ids):
        tool = cls.ref_notes_tool(ctx)
        if not (tool and ids):
            return []
        seen = set((vars(ctx.shelf).get("read_with") or {}).get(tool) or set())
        tried = ref.setdefault("noted", set())
        return [i for i in dict.fromkeys(ids) if i not in seen and i not in tried][:2]

    @classmethod
    def ref_notes_read(cls, ctx, ref, need):
        tool = cls.ref_notes_tool(ctx)
        if not (tool and need):
            return False
        ref.setdefault("noted", set()).update(need)
        before = ctx.wire.tick
        say = (f"The request takes {ref['head']} from a different listing: I read the notes of the listing that could be it "
               f"({tool}), because a note can point the reader further.")
        responses = Judging.open_reads(ctx, need, say, None, tool)
        if responses:
            Asking.post_replies(ctx, responses)
        cls._add_steps(ctx, before)
        cls.ref_ids_add(ctx, need)
        Trail.add(ctx, "ref_notes", ids=need, tool=tool)
        return ctx.wire.tick > before

    @classmethod
    def ref_pointers(cls, ctx):
        found = set()
        for ref in cls._all_refs(ctx):
            if ref.get("relation") == "differ" and ref.get("status") in ("open", "stuck", "resolved") and cls.ref_live(ctx, ref):
                for ident in ref.get("chain") or []:
                    if cls.ref_own_product(ctx, ref, ident):
                        found.add(ident)
        return found

    @classmethod
    def ref_own_product(cls, ctx, ref, ident):
        if not ctx.needs.items():
            return True
        shelf = Judging.shelf_known(ctx, ident)
        k = Asking.product_of(ctx, f"x ({shelf})") if shelf else None
        return k is None or not ref.get("tag") or k == Asking.product_of(ctx, f"x ({ref['tag']})")

    @classmethod
    def ref_pointer_note(cls, ctx, thing):
        for ref in cls._all_refs(ctx):
            pointed = ref.get("relation") == "differ" and thing in (ref.get("chain") or [])
            if not pointed or not cls.ref_live(ctx, ref) or not cls.ref_own_product(ctx, ref, thing):
                continue
            what = f" ({ref['value']})" if ref.get("status") == "resolved" and ref.get("value") else ""
            return (f"{thing} is the listing the shopper's reference for {ref['head']} points at, or one its notes lead "
                    f"through, so its {ref['prop']} is the very value{what} the shopper's {ref['prop']} must differ from: it cannot "
                    "meet that requirement")
        return ""

    @classmethod
    def ref_excluded(cls, ctx, fields, k):
        several = ctx.needs.items()
        for ref in cls._all_refs(ctx):
            if ref.get("relation") != "differ" or ref.get("status") != "resolved" or not cls.ref_live(ctx, ref):
                continue
            if several and ref.get("tag") and Asking.product_of(ctx, f"x ({ref['tag']})") not in (k, None):
                continue
            banned = set()
            for val in [ref.get("value")] + list(ref.get("goes_with") or []):
                if str(val or "").strip():
                    banned.add(" ".join(RefKit.ref_toks(val)))
            for key, val in (fields or {}).items():
                if PropKit.same_prop(key, ref["prop"]) and " ".join(RefKit.ref_toks(val)) in banned:
                    return True
        return False

    @classmethod
    def own_redirects(cls, ctx, thing):
        try:
            return cls._redirects(ctx, thing)
        except Exception:
            return []

    @classmethod
    def _redirects(cls, ctx, thing):
        rows = cls.ref_index(ctx)
        row = rows.get(thing) or {}
        notes = cls.ref_texts(ctx, row, desc=False)
        if not notes:
            return []
        own = cls.ref_sellers(ctx, row)
        elsewhere = set()
        for other in rows.values():
            elsewhere |= cls.ref_sellers(ctx, other)
        elsewhere -= own
        fields = cls._field_tokens(ctx, row, cls.ref_parts(ctx, row)[0])
        several = ctx.needs.items()
        shelf = Reading.shelf_of(ctx, row)
        k = Asking.product_of(ctx, f"x ({shelf})") if several and shelf else None
        found = []
        for req in ctx.needs.props_left():
            head = str(req).partition(":")[0]
            if several and k is not None and Asking.product_of(ctx, head) not in (k, None):
                continue
            prop = cls._bare_prop(head).strip()
            if prop:
                sentence = cls._redirecting_sentence(prop, notes, own, elsewhere, fields)
                if sentence is not None:
                    found.append((str(req), prop, sentence))
        return found

    @staticmethod
    def _redirecting_sentence(prop, notes, own, elsewhere, fields):
        negators = ("not", "nor", "never", "no", "neither")
        for sentence in re.split(r"(?<=[.!?;])\s+", notes):
            words = list(re.finditer(r"[A-Za-z][A-Za-z'-]*", sentence))
            blanked = re.sub(r"[A-Za-z0-9]+(?=\s+by\s+[A-Z])", lambda hit: " " * len(hit.group(0)), sentence)
            first_prop = None
            for m in words:
                if PropKit.prop_named(prop, blanked[:m.end()]):
                    first_prop = m.start()
                    break
            if first_prop is None:
                continue
            for n in range(1, len(words)):
                match = words[n]
                word = match.group(0)
                low = RefKit.ref_name(word)
                if match.start() <= first_prop or not low:
                    continue
                if low in elsewhere:
                    other = True
                else:
                    other = word[:1].isupper() and len(low) >= 4 and low not in own \
                        and not set(RefKit.ref_toks(word)) & fields and low not in fields
                negated = any(w.group(0).lower() in negators for w in words[max(0, n - 5): n])
                if other and not negated:
                    return sentence.strip()
        return None

    @classmethod
    def notes_remark(cls, ctx, call_name, call_args):
        quiet = {ctx.role("search"), ctx.role("cart"), ctx.role("cart_view"), ctx.role("message")}
        if call_name in quiet or call_name in ctx.finishers():
            return ""
        version = str(ctx.needs.version)
        remarks = []
        for ident in sorted(Lookup.asked_ids(ctx, call_args)):
            key = f"{ident}|{version}"
            in_cart = ident in ctx.cart.trolley.units
            if not in_cart and key not in ctx.cart.passed:
                continue
            told = vars(ctx.talk).setdefault("own_told", set())
            own = cls.own_redirects(ctx, ident) if key not in told else []
            if not own:
                continue
            told.add(key)
            ctx.cart.passed.discard(key)
            Trail.add(ctx, "own_redirect", id=ident, props=[p for _, p, _ in own])
            sent = "; ".join(f"{p} (\"{s[:160]}\")" for _, p, s in own)
            reqs = "; ".join(f"'{r}'" for r, _, _ in own)
            tail = ". If they do, take it out of the cart and put another verified fit in." if ident in ctx.cart.trolley.units else "."
            remarks.append(f"{ident}'s own notes may point {sent} at a different listing: in that case its own field does not count for "
                           + reqs + " (the grader uses the other listing's value), so it meets " + ("them" if len(own) > 1 else "it")
                           + " only when that listing's record shows it; the item check runs once more before it is ordered" + tail)
        return " ".join(remarks)

    @classmethod
    def ref_phrase(cls, ctx, line):
        text = str(line or "")
        if not RefKit.ref_text(text):
            return None
        parsed = RefKit.ref_parse(text, lambda name: cls.ref_seller_ok(ctx, name, text))
        if not parsed or not cls.ref_seller_ok(ctx, parsed["seller"], text):
            return None
        hit = re.search(r"(?<![A-Za-z'])" + re.escape(parsed["seller"]) + r"(?:'s)?(?![A-Za-z])", text)
        if hit is None:
            return None
        head = text[:hit.start()]
        stops = [head.rfind(mark) for mark in (",", ";", "!", "?")]
        stops += [m.start() for m in re.finditer(r"[.](?:\s|$)", head)]
        cut = max(stops + [-1])
        clause = head[cut + 1:].lower()
        marks = []
        for pattern in (DIFFER_WORDS, REF_WORDS, POINT_WORDS):
            marks += [m.start() for m in re.finditer(pattern, clause)]
        if not marks:
            return None
        start = cut + 1 + min(marks)
        tail = text[hit.end():]
        aside = re.search(r"\(\s*or\b[^)]*\)", tail)
        stop = re.search(r"[.!?;](?:\s|$)", tail)
        if aside and (stop is None or aside.start() < stop.start()):
            end = hit.end() + aside.end()
        elif stop:
            end = hit.end() + stop.start()
        else:
            end = hit.end() + len(tail)
        phrase = text[start:end].strip(" ,;")
        if not phrase:
            return None
        return phrase, start, end, parsed

    @classmethod
    def ref_line_words(cls, ctx, qtext, outside):
        if not qtext:
            return []
        kinds = set()
        for row in ctx.needs.items() or [{"product": str(ctx.needs.reading.get("product") or "")}]:
            kinds.update(re.findall(r"[a-z]+", str(row["product"]).lower()))
        words = re.findall(r"[a-z0-9]+", str(qtext).lower())

        def usable(w):
            return w not in LINK_WORDS and len(w) >= 3 and w not in kinds and w.rstrip("s") not in kinds
        runs = []
        for size in (3, 2, 1):
            for i in range(len(words) - size + 1):
                run = " ".join(words[i: i + size])
                if all(map(usable, words[i: i + size])) and PropKit.prop_named(run, outside):
                    runs.append(run)
        return runs

    @classmethod
    def ref_fix(cls, ctx, answer, lines, qtext):
        masked = []
        for line in lines:
            got = cls.ref_phrase(ctx, line)
            if got is None:
                masked.append(line)
                continue
            outside = line[:got[1]] + " " + line[got[2]:]
            masked.append(outside)
            cls._fix_line(ctx, answer, line, got, outside, qtext)
        return masked

    @classmethod
    def _fix_line(cls, ctx, answer, line, got, outside, qtext):
        fix = PointingLine(ctx, answer, line, got, outside)
        asked = [fix.name(a) for a in str(vars(ctx.talk).get("asking_label") or "").split(";") if fix.name(a)]
        tiers = [[str(r.get("property") or "") for r in fix.refs],
                 [fix.name(n) for n in fix.needs if fix.points(n.partition(":")[2])],
                 [fix.name(n) for n in fix.needs],
                 asked,
                 sorted(cls.store_keys(ctx)) + [fix.name(r) for r in ctx.needs.written],
                 cls.ref_line_words(ctx, qtext, outside)]
        fix.prop = cls._pick_prop(tiers, outside, asked, fix.inside_only)
        if not fix.prop:
            Trail.add(ctx, "ref_line_unplaced", line=line[:200])
            return
        fix.settle_tag(qtext)
        fix.apply()

    @staticmethod
    def _pick_prop(tiers, outside, asked, inside_only):
        def rank(p):
            return PropKit.prop_named(p, outside, exact=True), len(re.findall(r"[a-z0-9]+", p.lower())), len(p)
        for tier in tiers:
            named = [p for p in tier if p and PropKit.prop_named(p, outside)]
            if named:
                return max(named, key=rank)
        fallback = tiers[0] + tiers[1] + (asked if len(asked) == 1 else [])
        return next((p for p in fallback if p and not inside_only(p)), "")

    @classmethod
    def ref_names_prop(cls, ctx, prop, text):
        words = [w for w in PropKit.prop_words(prop) if len(w) >= 3]
        if not words:
            return True
        for tok in set(re.findall(r"[a-z]+", str(text).lower())):
            if tok in words:
                return True
            if len(tok) >= 4 and any(len(w) >= 4 and tok[:4] == w[:4] for w in words):
                return True
        return False

    @classmethod
    def ref_card(cls, ctx, ident):
        row = cls.ref_index(ctx).get(ident) or {}
        facts = {}
        for val in filter(lambda v: isinstance(v, dict), row.values()):
            for key, x in list(val.items())[:40]:
                if isinstance(x, (str, int, float)) and not isinstance(x, bool):
                    facts[str(key)] = str(x)[:90]
        learned = cls.seller_keys(ctx)
        sellers = sorted(set(str(val) for key, val in row.items() if cls._seller_field(key, val, learned)))
        card = {"id": ident, "sold by": " / ".join(sellers), "category": Reading.shelf_of(ctx, row),
                "title": Lookup.title_of(ctx, row)[:160], "facts": facts, "notes": cls.ref_texts(ctx, row)[:900]}
        return TextKit.as_text(card, 5000)

    @classmethod
    def ref_ask(cls, ctx, ref, pick, reason):
        if not cls.ref_calls_ok(ctx):
            return None
        ctx.pointers.ref_calls = vars(ctx.pointers).get("ref_calls", 0) + 1
        ask_text = REF_HOP.replace("PROPERTY", ref["prop"]).replace("DESCRIPTION", cls.ref_desc(ctx, ref))
        ask_text += "\n\nRECORDS:\n" + "\n".join(map(lambda ident: cls.ref_card(ctx, ident), pick))
        if reason:
            ask_text += "\n\nYOUR LAST ANSWER COULD NOT BE USED: " + reason + " Answer again from the records."
        answer = ctx.wire.consult(ask_text, 3000, CHECKER[:1])
        Trail.add(ctx, "ref_hop", prop=ref["prop"], ids=pick, answer=answer)
        return answer

    @classmethod
    def ref_apply(cls, ctx, ref, pick, answer, again=True):
        if not isinstance(answer, dict):
            ref["fails"] += 1
            if ref["fails"] >= 2:
                ref["judged"].update(pick)
                cls.ref_stop(ctx, ref, "the hop check gave no answer")
            return
        raw_target = answer.get("target")
        target = ""
        if isinstance(raw_target, (str, int)) and not isinstance(raw_target, bool):
            target = str(raw_target).strip()
        missing = [str(x) for x in answer.get("missing") or [] if str(x).strip()] if isinstance(answer.get("missing"), list) else []
        if target not in pick or missing:
            ref["judged"].update(pick)
            cls.ref_ids_add(ctx, pick)
            if target in pick:
                Trail.add(ctx, "ref_partial", prop=ref["prop"], target=target, missing=missing[:4])
            return
        where, problem, go, value, pairs = cls.ref_validate(ctx, ref, target, answer, pick)
        if problem and again and cls.ref_calls_ok(ctx):
            cls.ref_apply(ctx, ref, pick, cls.ref_ask(ctx, ref, pick, problem), again=False)
            return
        cls.ref_ids_add(ctx, pick)
        if problem:
            Trail.add(ctx, "ref_invalid", prop=ref["prop"], target=target, problem=problem[:300])
            if where == "target":
                ref["judged"].add(target)
            else:
                ref["judged"].update(pick)
                ref["chain"].append(target)
                cls.ref_stop(ctx, ref, problem)
            return
        ref["judged"].update(pick)
        if target in ref["chain"]:
            cls.ref_stop(ctx, ref, "its notes lead back to a listing already followed")
        elif go:
            cls.ref_go(ctx, ref, target, go)
        else:
            cls.ref_resolve(ctx, ref, target, value, pairs)

    @classmethod
    def ref_validate(cls, ctx, ref, target, answer, pick=None):
        row = cls.ref_index(ctx).get(target) or {}
        keys, vals = cls.ref_parts(ctx, row)
        loose = ref.get("loose") or set()
        details = ref["target"]["details"]
        for d in details:
            if d not in loose and not cls.ref_shows(ctx, row, d, (keys, vals)):
                return "target", f"{target} does not show '{d}', so it is not the listing described.", None, "", []

        def shown(ident):
            k, v = cls.ref_parts(ctx, cls.ref_index(ctx).get(ident) or {})
            return len([d for d in details if cls._toks(d) and cls._toks(d) <= k | v])
        best = max(pick or [target], key=shown)
        if shown(best) > shown(target):
            return "target", f"{best} shows more of the details named than {target}.", None, "", []
        notes = cls.ref_texts(ctx, row)
        note_toks = cls._toks(notes)
        go = answer.get("go_to") if isinstance(answer.get("go_to"), dict) else None
        if go and str(go.get("seller") or "").strip():
            return cls._check_go(ctx, ref, target, go, row, notes, note_toks)
        value = str(answer.get("value") or "").strip()
        value_toks = cls._toks(value)
        if not value or not value_toks or not value_toks <= vals | keys:
            return "answer", f"the value '{value}' is not written in {target}'s record.", None, "", []
        pairs = []
        for pair in dict.fromkeys(cls._clean_list(answer.get("goes_with"))):
            pair_toks = cls._toks(pair)
            if pair_toks and pair_toks <= note_toks and pair_toks != value_toks:
                pairs.append(pair)
        return "", "", None, value, pairs[:3]

    @classmethod
    def _check_go(cls, ctx, ref, target, go, row, notes, note_toks):
        seller = str(go["seller"]).strip()
        pattern = r"(?<![a-z0-9])" + re.escape(seller.lower()) + r"(?:'s)?(?![a-z0-9])"
        if not re.search(pattern, notes.lower()):
            return "answer", f"the go_to seller '{seller}' is not written in {target}'s notes.", None, "", []
        about_prop = [s for s in re.split(r"(?<=[.!?])\s+", notes) if re.search(pattern, s.lower()) and cls.ref_names_prop(ctx, ref["prop"], s)]
        if not about_prop:
            return "answer", f"the note naming {seller} is about another property, not {ref['prop']}.", None, "", []
        details = []
        for d in cls._clean_list(go.get("details")):
            toks = cls._toks(d)
            if toks and 2 * len(toks & note_toks) >= len(toks):
                details.append(d)
        nxt = {"seller": seller, "kind": str(go.get("kind") or "").strip()[:80], "details": details[:6]}
        if RefKit.ref_name(seller) in cls.ref_sellers(ctx, row) and cls.ref_key(ctx, nxt) == cls.ref_key(ctx, ref["target"]):
            return "answer", f"the go_to is {target} itself.", None, "", []
        return "", "", nxt, "", []

    @classmethod
    def ref_go(cls, ctx, ref, target, go):
        ref["chain"].append(target)
        key = cls.ref_key(ctx, go)
        if key in ref["seen"]:
            cls.ref_stop(ctx, ref, "its notes lead back to a listing already followed")
        elif ref["hops"] + 1 >= REF_HOPS:
            cls.ref_stop(ctx, ref, f"more than {REF_HOPS} listings in a row send the reader elsewhere")
        else:
            ref["seen"].add(key)
            ref["hops"] += 1
            ref.update(target=go, judged=set(), queries=[], pages=0, tries=0, fails=0, unindexed=False, loose=set())
            ctx.log.append({"kind": "ref_go", "prop": ref["prop"], "from": target, "to": go})
            cls.ref_tell(ctx, f"{target} is the listing named for {ref['head']}, and its notes send the reader on to {cls.ref_desc(ctx, ref)}: the "
                         "code follows them.")

    @classmethod
    def ref_value(cls, ctx, ref):
        value = str(ref["value"])
        others = [p for p in list(ref.get("goes_with") or []) if p.lower() != value.lower()]
        relation = ref["relation"]
        if relation == "within" and TextKit.is_number(ref.get("tol")):
            hit = re.search(r"(-?\d+(?:\.\d+)?)\s*([a-z%]*)", value.lower().replace(",", ""))
            if hit:
                centre, tol = float(hit.group(1)), float(ref["tol"])
                unit = hit.group(2) or str(ref.get("unit") or "")
                span = f"{round(centre - tol, 6):g} to {round(centre + tol, 6):g} {unit}".strip()
                return span + f" (within {tol:g} of {value})"
        if relation in ("at most", "at least"):
            return f"{relation} {value}"
        if relation == "differ":
            return " and ".join("not " + v for v in [value] + others)
        return " or ".join([value] + others)

    @classmethod
    def ref_resolve(cls, ctx, ref, target, value, pairs):
        if not str(value or "").strip():
            cls.ref_stop(ctx, ref, f"{target} shows no value for it")
            return
        ref["chain"].append(target)
        ref.update(status="resolved", value=str(value).strip(), goes_with=list(pairs))
        new_req = f"{ref['head']}: {cls.ref_value(ctx, ref)}"
        ref["now"] = new_req
        cls.ref_swap(ctx, ref, new_req)
        ctx.needs.resolved.append(new_req)
        ctx.needs.version += 1
        Trail.add(ctx, "ref_resolved", req=ref["req"][:200], now=new_req, chain=list(ref["chain"]))
        via = ""
        if len(ref["chain"]) > 1:
            via = f", reached from {ref['chain'][0]} through the listings' notes"
        joined = " and ".join(pairs)
        if ref["relation"] != "differ":
            pair = f"; its notes say {ref['value']} goes together with {joined}, so each of them meets it" if pairs else ""
        else:
            pair = (f"; its notes pair {joined} with it" if pairs else "") + (
                ". The shopper's value MUST DIFFER from that one: any other value meets it, that value does not")
        cls.ref_tell(ctx, f"RESOLVED: {ref['head']} is given by another listing, and {target} shows {ref['value']}{via}{pair}. The "
                     f"requirement is now: {new_req}.")

    @classmethod
    def ref_swap(cls, ctx, ref, new_req):
        needs = ctx.needs
        seller = RefKit.ref_name(ref["first"]["seller"])

        def mine(req):
            if cls.ref_match(ctx, ref, req):
                return True
            head, _, value = str(req).partition(":")
            tag = cls._tag_of(head)
            if tag is not None and ref["tag"] and not PropKit.kind_words(tag) & PropKit.kind_words(ref["tag"]):
                return False
            return bool(seller) and seller in RefKit.ref_name(value) and PropKit.same_prop(cls._bare_prop(head), ref["prop"])

        def swapped(reqs):
            return [new_req if mine(req) else req for req in reqs]
        replaced = False
        for attr in ("written", "uncovered"):
            old = list(getattr(needs, attr))
            new = swapped(old)
            if new != old:
                replaced = True
            setattr(needs, attr, list(dict.fromkeys(new)))
        if needs.now_reqs is not None:
            new = swapped(needs.now_reqs)
            needs.now_reqs = list(dict.fromkeys(new + ([] if new_req in new else [new_req])))
        if not replaced:
            needs.uncovered.append(new_req)
        if isinstance(needs.reading.get("requirements"), list):
            needs.reading["requirements"] = list(needs.written)
        value = new_req.partition(":")[2].strip()
        rows = [row for row in needs.reading.get("items") or [] if isinstance(row, dict) and isinstance(row.get("requirements"), list)]
        for row in rows:
            if len(rows) >= 2 and ref["tag"] and not (PropKit.kind_words(row.get("product") or "") & PropKit.kind_words(ref["tag"])):
                continue
            row["requirements"] = [f"{str(req).partition(':')[0].strip()}: {value}" if mine(req) else req for req in row["requirements"]]

    @classmethod
    def ref_stop(cls, ctx, ref, why):
        if ref["status"] == "open":
            ref["status"], ref["why"] = "stuck", why
            Trail.add(ctx, "ref_stuck", req=ref["req"][:200], why=why[:300], chain=list(ref["chain"]))
            cls.ref_tell(ctx, f"NOT RESOLVED: {ref['head']} ({why}). Its value stays unknown: if a listing's notes you read show where it "
                         "ends, use that value; otherwise never decline or reject a listing over it.")

    @classmethod
    def ref_tell(cls, ctx, text):
        chat = ctx.talk.chat
        if chat and len(chat) > 2 and chat[-1].get("role") in ("tool", "user") and isinstance(chat[-1].get("content"), str):
            chat[-1]["content"] += "\n[reference] " + text
            return
        ctx.pointers.ref_told = list(vars(ctx.pointers).get("ref_told") or []) + [text]

    @classmethod
    def ref_line(cls, ctx):
        try:
            parts = []
            for ref in cls._all_refs(ctx):
                status = ref["status"]
                if status == "resolved":
                    differ = ", must differ from it" if ref.get("relation") == "differ" else ""
                    parts.append(f"{ref['now']} (read on {ref['chain'][-1]}{differ})")
                elif status == "open":
                    parts.append(f"{ref['head']}: the code is following it (now: {cls.ref_desc(ctx, ref)})")
                elif status == "stuck":
                    parts.append(f"{ref['head']}: not resolved, value unknown (never decline or reject a listing over it)")
            told = vars(ctx.pointers).pop("ref_told", None)
            text = "\n[references] " + "; ".join(parts) if parts else ""
            for item in told or []:
                text += "\n[reference] " + item
            return text
        except Exception:
            return ""

    @classmethod
    def ref_note(cls, ctx, thing=""):
        try:
            lines, ends = [], []
            for ref in cls._all_refs(ctx):
                status = ref["status"]
                if status == "resolved":
                    lines.append(cls._resolved_line(ref))
                elif status in ("open", "stuck"):
                    chain = ("its chain so far: " + " -> ".join(ref["chain"])) if ref["chain"] else "the listing it points at was not found"
                    lines.append(f"- {ref['head']}: NOT RESOLVED by the code (" + chain + "). If a record below is that listing, or the last "
                                 "one its chain reached, and it states its own value without pointing the reader further, take that value "
                                 "as the requirement; otherwise the value is unknown, so never fail an item on this point alone.")
                    ends.extend(i for i in ref["chain"][-2:] if i != thing)
            if not lines:
                return ""
            text = ("\n\nREQUIREMENTS GIVEN BY REFERENCE TO ANOTHER LISTING (the code traced that listing's notes through the store's "
                    "records):\n" + "\n".join(lines))
            records = []
            for ident in dict.fromkeys(reversed(ends)):
                records.extend(ctx.shelf.archive.about(ident, 1))
            if records:
                text += "\nRECORDS OF THE LAST LISTINGS ITS CHAIN REACHED:\n" + "\n".join(records[:2])
            return text
        except Exception:
            return ""

    @staticmethod
    def _resolved_line(ref):
        via = f" (followed from {ref['chain'][0]} by way of the listings' notes)" if len(ref["chain"]) > 1 else ""
        pairs = ref["goes_with"]
        joined = " and ".join(pairs)
        if ref.get("relation") != "differ":
            pair = f"; its notes pair {ref['value']} with {joined}, so any of them satisfies it" if pairs else ""
            return f"- {ref['now']}: taken from the value on {ref['chain'][-1]}{via}{pair}."
        pair = f"; its notes link {joined} to it" if pairs else ""
        return (f"- {ref['now']}: the shopper's {ref['prop']} MUST be something other than the value on {ref['chain'][-1]} "
                f"({ref['value']}){via}{pair}: every other value satisfies it, that one does not; each listing in its "
                f"chain ({', '.join(ref['chain'])}) carries that value.")


class PointingLine:
    def __init__(self, ctx, answer, line, got, outside):
        self.ctx, self.answer, self.line, self.outside = ctx, answer, line, outside
        self.phrase, start, end, self.parsed = got
        self.inside = line[start:end]
        raw_needs, raw_refs = answer.get("needs"), answer.get("references")
        self.needs = [n for n in raw_needs or [] if isinstance(n, str) and ":" in n] if isinstance(raw_needs, list) else []
        self.refs = [r for r in raw_refs or [] if isinstance(r, dict)] if isinstance(raw_refs, list) else []
        self.prop = ""
        self.tag = ""
        self.several = []

    @staticmethod
    def name(req):
        return Referencing._bare_prop(str(req).partition(":")[0]).strip()

    @staticmethod
    def points(value):
        return bool(RefKit.ref_text(value) or re.search(DIFFER_WORDS, value.lower()))

    def inside_only(self, prop):
        if not prop or not PropKit.prop_named(prop, self.inside):
            return False
        return not PropKit.prop_named(prop, self.outside)

    def detail(self, req):
        if self.inside_only(self.name(req)):
            return True
        words = set(RefKit.ref_toks(str(req).partition(":")[2]))
        if not words or not words <= set(RefKit.ref_toks(self.inside)):
            return False
        return words.isdisjoint(RefKit.ref_toks(self.outside))

    def same_item(self, req):
        if not self.several or not self.tag:
            return True
        tag_here = Referencing._tag_of(str(req).partition(":")[0])
        return tag_here is None or bool(PropKit.kind_words(tag_here) & PropKit.kind_words(self.tag))

    def settle_tag(self, qtext):
        ctx = self.ctx
        self.several = ctx.needs.items()
        if not self.several:
            return
        k = Asking.product_of(ctx, self.outside)
        if k is None:
            tags = []
            for n in self.needs:
                if PropKit.same_prop(self.name(n), self.prop):
                    tags.extend(re.findall(r"\(([^)]*)\)", n.partition(":")[0]))
            k = Asking.product_of(ctx, f"x ({tags[-1]})") if tags else None
        if k is None and qtext:
            k = Asking.product_of(ctx, qtext)
        if k is not None:
            self.tag = self.several[k]["product"]

    def apply(self):
        ctx, answer, prop = self.ctx, self.answer, self.prop
        mine = list(filter(lambda n: PropKit.same_prop(self.name(n), prop) and self.same_item(n), self.needs))
        relation, tol, unit = RefKit.ref_relation(self.phrase)
        seller_name = RefKit.ref_name(self.parsed["seller"])
        keep = None
        for n in mine:
            value = n.partition(":")[2]
            if self.points(value) and RefKit.ref_relation(value)[0] == relation and seller_name in RefKit.ref_name(value):
                keep = n
                break
        if mine:
            head = mine[0].partition(":")[0].strip()
        elif self.tag:
            head = f"{prop} ({self.tag})"
        else:
            head = prop
        if self.several and self.tag and not re.findall(r"\(([^)]*)\)", head):
            head = f"{head} ({self.tag})"
        need = keep or f"{head}: {self.phrase}"
        details = [n for n in self.needs if n not in mine and self.detail(n)]
        spot = self.needs.index(mine[0]) if mine else len(self.needs)
        rest = [n for n in self.needs if n not in mine and n not in details]
        answer["needs"] = rest[:spot] + [need] + rest[spot:]
        self.trim(need, details)
        self.record_entry(relation, tol, unit, seller_name)
        answer["references"] = self.refs
        answer["ref_needs"] = list(answer.get("ref_needs") or []) + [need]
        Trail.add(ctx, "ref_line", prop=prop, tag=self.tag, relation=relation, need=need[:200], kept=bool(keep),
                  details_out=[d[:80] for d in details])

    def trim(self, need, details):
        answer, written = self.answer, self.ctx.needs.written
        if isinstance(answer.get("requirements_now"), list):
            kept = []
            for req in answer["requirements_now"]:
                if not isinstance(req, str):
                    continue
                if req not in written and (req in details or self.detail(req)):
                    continue
                if PropKit.same_prop(self.name(req), self.prop) and self.same_item(req) and req not in written:
                    continue
                kept.append(req)
            answer["requirements_now"] = kept if need in kept else kept + [need]
        if isinstance(answer.get("dropped"), list):
            answer["dropped"] = [d for d in answer["dropped"]
                                 if not isinstance(d, str) or not (self.inside_only(self.name(d)) or self.detail(d))]
        if isinstance(answer.get("no_preference"), list):
            answer["no_preference"] = [p for p in answer["no_preference"]
                                       if not isinstance(p, str) or not (self.inside_only(self.name(p)) or PropKit.same_prop(self.name(p), self.prop))]

    def record_entry(self, relation, tol, unit, seller_name):
        parsed = self.parsed
        entry = None
        for ref in self.refs:
            if PropKit.same_prop(str(ref.get("property") or ""), self.prop):
                entry = ref
                break
        if entry is None:
            entry = {"property": self.prop}
            self.refs.append(entry)
        if not Referencing.ref_seller_ok(self.ctx, entry.get("seller"), self.line) or RefKit.ref_name(entry.get("seller")) != seller_name:
            entry.update(seller=parsed["seller"], kind=parsed["kind"], details=list(parsed["details"]))
        elif not entry.get("details"):
            entry["details"] = list(parsed["details"])
        entry.setdefault("kind", parsed["kind"])
        if relation != "same" or not entry.get("relation"):
            entry.update(relation=relation, tolerance=tol, unit=unit)
class Coverage:
    @staticmethod
    def _plain_value(req):
        return re.sub(r"\((?:stated|revealed[^)]*)\)", "", req.partition(":")[2]).lower()

    @staticmethod
    def _shown_on(walk, home):
        shelves = walk.get("row_shelf") or {}
        return [ident for ident in shelves if shelves[ident] in home]

    @staticmethod
    def _touches(walk, home):
        return any(shelf in home for shelf in (walk.get("row_shelf") or {}).values())

    @staticmethod
    def _by_product(ctx, k, req):
        if not ctx.needs.items() or k is None:
            return True
        return Asking.product_of(ctx, req.partition(":")[0]) in (k, None)

    @classmethod
    def ceil_learn(cls, ctx, call_args, rows, total, obs):
        found = Reading.page_arg(ctx)
        page_key = found[0] if found else None
        plain_args = {k: v for k, v in call_args.items() if k != page_key}
        seen = vars(ctx.shelf).setdefault("ceil_seen", {})
        seen[json.dumps(plain_args, sort_keys=True, default=str)] = int(total)
        dicts = list(filter(lambda row: isinstance(row, dict), rows))
        if found and found[1] is True and dicts and isinstance(obs, dict):
            cls._learn_depth(ctx, call_args, found, dicts, total, obs)
        _, mode, _, paging = Reading.search_keys(ctx, call_args)

        def limit_like(key, val):
            if not TextKit.is_number(val) or isinstance(val, bool) or key == page_key or key in paging or val <= 1:
                return False
            return not re.search(r"page|size|count|result|min|low|from|start|offset", str(key).lower())
        bounds = [key for key, val in call_args.items() if limit_like(key, val)]
        if not dicts or len(bounds) != 1:
            return
        bound_key = bounds[0]
        cap = float(call_args[bound_key])
        labels = set(Reading.shelf_of(ctx, row) for row in dicts)
        shelf_keys = [k for k, v in call_args.items() if isinstance(v, str) and k != page_key and v.strip() and v.strip().lower() in labels]
        shelf_key = None if len(shelf_keys) != 1 else shelf_keys[0]
        fixed = dict((k, v) for k, v in mode if k != shelf_key)
        sig = json.dumps([sorted(fixed.items()), bound_key], default=str)
        prices = [cls.row_cost(ctx, row) for row in dicts]
        bad = vars(ctx.shelf).setdefault("ceil_bad", set())
        if any(p is not None and p > cap + 1e-9 for p in prices):
            bad.add(sig)
            return
        if sig in bad or shelf_key is None or len(labels) != 1 or "" in labels or None in prices:
            return

        def on_every_row(val):
            low = val.strip().lower()
            for row in dicts:
                matched = False
                for kk, x in row.items():
                    if not Reading.shelf_of(ctx, {kk: x}) and isinstance(x, str) and x.strip().lower() == low:
                        matched = True
                        break
                if not matched:
                    return False
            return True
        template = {}
        for k, v in plain_args.items():
            if k == bound_key or k == shelf_key:
                continue
            if not (isinstance(v, str) and v.strip() and k not in fixed and on_every_row(v)):
                template[k] = v
        words = 0
        for k, v in template.items():
            if isinstance(v, str) and k not in fixed:
                words += len(v.split())
        props = (ctx.shelf.shapes.get(ctx.role("search") or "") or {}).get("properties") or {}
        old = vars(ctx.shelf).get("ceil_mode")
        if old and old["sig"] not in bad and old["words"] < words:
            return
        bound_spec = (props.get(bound_key) or {}) if isinstance(props, dict) else {}
        ctx.shelf.ceil_mode = {"sig": sig, "tmpl": template, "mode": fixed, "shelf": shelf_key, "bound": bound_key, "words": words,
                               "int": bound_spec.get("type") == "integer"}

    @classmethod
    def _learn_depth(cls, ctx, call_args, found, dicts, total, obs):
        at = call_args.get(found[0], found[2])
        nexts = []
        for key, val in obs.items():
            if re.search(r"next|more|onward|continu|cursor", str(key).lower()) and not isinstance(val, (list, dict)):
                nexts.append(val)
        size = max(len(dicts), cls.page_size(ctx))
        ended = all(val is None or val is False or val == "" for val in nexts)
        if TextKit.is_number(at) and nexts and ended and len(dicts) >= size:
            reach = int((at - found[2] + 1) * size)
            if total > reach:
                ctx.shelf.page_cap = min(vars(ctx.shelf).get("page_cap") or reach, reach)

    @classmethod
    def row_cost(cls, ctx, row):
        priced = [val for key, val in row.items()
                  if TextKit.is_number(val) and not isinstance(val, bool) and re.search(r"price|cost", str(key).lower())]
        return float(priced[0]) if priced else None

    @classmethod
    def ceil_spec(cls, ctx):
        spec = vars(ctx.shelf).get("ceil_mode")
        if spec and spec["sig"] not in vars(ctx.shelf).get("ceil_bad", set()):
            return spec
        return None

    @classmethod
    def cg_listing(cls, ctx, args):
        spec = cls.ceil_spec(ctx)
        if spec and isinstance(args, dict):
            wanted = spec["mode"]
            return all(args.get(k) == wanted[k] for k in wanted) if wanted else spec["bound"] in args
        return False

    @classmethod
    def page_size(cls, ctx):
        largest = 0
        for walk in Trail.book(ctx).values():
            largest = max(largest, walk.get("size", 0))
        return largest

    @classmethod
    def visible_depth(cls, ctx):
        found = Reading.page_arg(ctx)
        size = cls.page_size(ctx)
        known = found and size > 0 and found[1] != "token" and found[3] is not None
        if not known:
            return vars(ctx.shelf).get("page_cap")
        last, start = found[3], found[2]
        return int((last - start + 1) * size) if found[1] else int(last - start + size)

    @classmethod
    def raw_label(cls, ctx, labels):
        if not labels:
            return ""
        tally = {}
        archive = ctx.shelf.archive

        def visit(d):
            if not any(isinstance(v, str) and v in archive.ids and "id" in str(key).lower() for key, v in d.items()):
                return
            for key, val in d.items():
                low = str(key).lower()
                if not isinstance(val, str) or not 0 < len(val.strip()) <= 60:
                    continue
                if re.search(r"categor|shelf|department|aisle", low) and "id" not in low:
                    if val.strip().lower() in labels:
                        tally[val.strip()] = tally.get(val.strip(), 0) + 1
                    return
        for rec in archive.items:
            TextKit.walk_dicts(rec, visit)
        return max(sorted(tally), key=tally.get, default="")

    @classmethod
    def below(cls, ctx, price):
        if not TextKit.is_number(price):
            return None
        whole_spec = (cls.ceil_spec(ctx) or {}).get("int")
        val = float(price)
        if val.is_integer():
            lower = int(val) - 1
        elif whole_spec:
            lower = int(val)
        else:
            lower = round(val - 0.01, 2)
        return lower if lower > 0 else None

    @classmethod
    def ceil_mid(cls, ctx, a, b):
        if (cls.ceil_spec(ctx) or {}).get("int") or (float(a).is_integer() and float(b).is_integer()):
            return (int(a) + int(b)) // 2
        return round((float(a) + float(b)) / 2, 2)

    @classmethod
    def ceil_args(cls, ctx, spec, label, bound):
        args = dict(spec["tmpl"])
        args[spec["shelf"]] = label
        args[spec["bound"]] = int(bound) if spec.get("int") or float(bound).is_integer() else bound
        return args

    @classmethod
    def cg_products(cls, ctx):
        return list(range(len(ctx.needs.items()))) or [None]

    @classmethod
    def ceiling_ready(cls, ctx, k):
        if ctx.needs.items() and k is None:
            return False
        checks = (lambda: ctx.needs.priority() == "cheapest",
                  lambda: k not in vars(ctx.shelf).get("ceil_walked", set()),
                  lambda: cls.ceil_spec(ctx) is not None,
                  lambda: Reading.page_arg(ctx) is not None,
                  lambda: bool(cls.raw_label(ctx, Walking.walk_home(ctx, k))))
        return all(check() for check in checks)

    @classmethod
    def ceil_aim(cls, ctx, k):
        pick = Walking.walk_pick(ctx, k)
        held = (vars(ctx.shelf).get("ceil_aims") or {}).get(k)
        costs = [pick[1]] if pick else []
        if TextKit.is_number(held):
            costs.append(float(held))
        if costs or not cls.bound_need(ctx, k):
            return min(costs) if costs else None
        home = Walking.walk_home(ctx, k)
        seen = [pair[0] for pair in cls.cheap_unread(ctx, cls.home_rows(ctx, home), k)]
        archive = ctx.shelf.archive
        for ident in archive.read:
            if Judging.shelf_known(ctx, ident) not in home or not TextKit.is_number(archive.price(ident)):
                continue
            if ident in Lookup.dead_lines(ctx) or ident in ctx.cart.refused_lines or Judging.last_stock(ctx, ident) is False:
                continue
            seen.append(float(archive.price(ident)))
        return min(seen) if seen else None

    @classmethod
    def bound_need(cls, ctx, k):
        if k is None or not ctx.needs.items():
            return False
        for req in ctx.needs.props_left():
            head, _, value = str(req).partition(":")
            if Asking.product_of(ctx, head) != k:
                continue
            if re.search(r"\d", value) and re.search(CH_BOUND, value.lower()):
                return True
        return False

    @classmethod
    def home_rows(cls, ctx, home):
        ids = set()
        archive = ctx.shelf.archive

        def visit(d):
            ident = Reading.row_mark(ctx, d)
            if ident in archive.ids and Reading.shelf_of(ctx, d) in home:
                ids.add(ident)
        for rec in archive.items:
            TextKit.walk_dicts(rec, visit)
        return ids

    @classmethod
    def ceil_clash(cls, ctx, home, bound):
        state = {"hit": False}
        archive = ctx.shelf.archive

        def visit(d):
            if state["hit"] or Reading.shelf_of(ctx, d) not in home:
                return
            if any(isinstance(v, str) and v in archive.ids and "id" in str(key).lower() for key, v in d.items()):
                cost = cls.row_cost(ctx, d)
                state["hit"] = cost is not None and cost <= bound and Reading.stock_state(ctx, d) is True
        for rec in archive.items:
            TextKit.walk_dicts(rec, visit)
        return state["hit"]

    @classmethod
    def cheap_unread(cls, ctx, ids, k, below=None):
        ids = set(ids)
        if not ids:
            return []
        rows = ctx.needs.items()
        qty = ctx.needs.quantity() if k is None or k >= len(rows) else rows[k]["quantity"]
        cap = ctx.needs.spend_cap
        archive = ctx.shelf.archive
        sold = {}

        def visit(d):
            for key, v in d.items():
                if isinstance(v, str) and v in ids and "id" in str(key).lower():
                    state = Reading.stock_state(ctx, d)
                    if state is not None:
                        sold[Reading.row_mark(ctx, d)] = not state
                    return
        for item in archive.items:
            TextKit.walk_dicts(item, visit)

        def keep(ident):
            cost = archive.price(ident)
            if ident in archive.read or ident in ctx.cart.refused_lines or sold.get(ident) or not TextKit.is_number(cost):
                return False
            if TextKit.is_number(cap) and TextKit.is_number(qty) and cost * qty > cap:
                return False
            return below is None or cost < below
        return sorted((float(archive.price(ident)), ident) for ident in ids if keep(ident))

    @classmethod
    def all_words(cls, ctx):
        kind = ctx.shelf.word_kind
        return kind == "all" or (bool(ctx.shelf.lead_words) and kind != "any")

    @classmethod
    def valued_ids(cls, ctx, k):
        ids = set()
        if not cls.all_words(ctx):
            return ids
        home = Walking.walk_home(ctx, k)
        for walk in Trail.book(ctx).values():
            usable = walk.get("id_order") is True and not cls.cg_listing(ctx, walk["args"])
            if usable and Walking.walk_carries(ctx, walk["args"], k):
                ids.update(cls._shown_on(walk, home))
        return ids

    @classmethod
    def valued_complete(cls, ctx, k):
        try:
            return cls._complete_search(ctx, k)
        except Exception:
            return False

    @classmethod
    def _complete_search(cls, ctx, k):
        if not cls.all_words(ctx):
            return False
        home = Walking.walk_home(ctx, k)
        single = set()
        for n, req in enumerate(ctx.needs.props_left()):
            if not cls._by_product(ctx, k, req):
                continue
            parts = list(filter(str.strip, re.split(r"[,;/]|\bor\b|\beither\b|\bnor\b", cls._plain_value(req))))
            if len(parts) == 1:
                single.add(n)
        for walk in Trail.book(ctx).values():
            if walk.get("id_order") is not True or cls.cg_listing(ctx, walk["args"]) or not walk.get("first"):
                continue
            if Reading.next_page(ctx, walk) is not None or len(walk.get("row_shelf") or {}) < walk["total"]:
                continue
            if cls._touches(walk, home) and Walking.walk_carries(ctx, walk["args"], k) & single:
                return True
        return False

    @classmethod
    def row_carries(cls, ctx, ident, k):
        wanted, amounts = [], []
        for req in ctx.needs.props_left():
            if not cls._by_product(ctx, k, req):
                continue
            value = cls._plain_value(req)
            if RefKit.ref_text(value):
                continue
            for part in re.split(r"[,;/]|\bor\b|\beither\b|\bnor\b", value):
                amount = Walking.fig_amount(ctx, part)
                if amount:
                    amounts.append(amount)
                elif not Walking.amount_part(ctx, part):
                    stems = [Reading.stem(ctx, w) for w in re.findall(r"[a-z0-9]{3,}", part)]
                    if stems:
                        wanted.append(stems)
        if not (wanted or amounts):
            return False
        texts = []

        def visit(d):
            if any(isinstance(v, str) and v == ident and "id" in str(key).lower() for key, v in d.items()):
                texts.extend(str(v).lower() for v in d.values() if isinstance(v, str))
        for item in ctx.shelf.archive.items:
            TextKit.walk_dicts(item, visit)
        body = " ".join(texts)
        have = set(Reading.stem(ctx, w) for w in re.findall(r"[a-z0-9]{3,}", body))
        written = [re.search(r"(?<![\d.,])" + re.escape(fig) + r"\s*-?\s*" + re.escape(Reading.stem(ctx, unit)), body)
                   for fig, unit in amounts]
        if any(written):
            return True
        return any(set(stems) <= have for stems in wanted)

    @classmethod
    def vl_room(cls, ctx):
        burst = ctx.talk.asking and ctx.role("message")
        left = max(0, ctx.ask_room() - ctx.budget.msgs_sent) if burst else 0
        return left, ctx.moves_left() - Intake.order_due(ctx) - 2 * left

    @classmethod
    def _value_pieces(cls, low):
        if re.search(r"\bor\b|\beither\b|/", low):
            return [p for p in re.split(r"[,;/]|\bor\b|\beither\b", low) if p.strip()][:3]
        return [re.split(r"[,;&+]|\band\b", low)[0]]

    @classmethod
    def vl_values(cls, ctx, k):
        values = []
        for head, value in Closing.code_values(ctx, k):
            low = re.sub(r"\([^)]*\)", " ", str(value).lower().replace("’", "'")).strip()
            if not low or RefKit.ref_text(low):
                continue
            name = set(re.findall(r"[a-z0-9]+", str(head).lower()))
            alternatives = []
            for piece in cls._value_pieces(low):
                amount = Walking.fig_amount(ctx, piece)
                if re.search(EXCLUDE_WORDS, piece):
                    continue
                if Walking.amount_part(ctx, piece) and not amount:
                    continue
                piece = re.sub(r"\b(?:" + VALUE_LINKS + r")\b", " ", re.sub(r"'s\b", "", piece))
                words = []
                for w in re.findall(r"\d+(?:[.,]\d+)+|[a-z0-9]+", piece):
                    if re.search(r"[.,]", w) or w in LINK_WORDS:
                        continue
                    if len(w) < 2 and not (amount and w == amount[0]):
                        continue
                    if w in name and not (amount and w == amount[1]):
                        continue
                    words.append(w)
                if words and words not in alternatives:
                    alternatives.append(words)
            if alternatives and alternatives not in values:
                values.append(alternatives)
        return values

    @classmethod
    def vl_words(cls, ctx, k, label):
        cleaned = re.sub(r"'s\b", "", label.lower().replace("’", "'"))
        words = list(dict.fromkeys(w for w in re.findall(r"[a-z0-9]+", cleaned) if len(w) >= 2 and w not in LINK_WORDS))
        if vars(ctx.shelf).get("label_indexed") is not False:
            return words
        home = Walking.walk_home(ctx, k)
        titles = []

        def visit(d):
            if Reading.shelf_of(ctx, d) in home and Reading.row_mark(ctx, d) in ctx.shelf.archive.ids:
                title = Lookup.title_of(ctx, d).lower()
                titles.append(set(Reading.stem(ctx, w) for w in re.findall(r"[a-z0-9]+", title)))
        for rec in ctx.shelf.archive.items:
            TextKit.walk_dicts(rec, visit)
        common = []
        for w in words:
            if titles and 2 * len([t for t in titles if Reading.stem(ctx, w) in t]) >= len(titles):
                common.append(w)
        if common:
            return common
        rows = ctx.needs.items()
        use_row = rows and k is not None and k < len(rows)
        product = str(rows[k]["product"] if use_row else ctx.needs.reading.get("product") or "")
        return [w for w in re.findall(r"[a-z0-9]+", product.lower()) if len(w) > 2 and w not in LINK_WORDS][-1:]

    @classmethod
    def vl_queries(cls, ctx, k, label, cap_len):
        values = cls.vl_values(ctx, k)
        if not values:
            return [], []
        label_words = cls.vl_words(ctx, k, label)
        every_alt = ctx.needs.priority() == "cheapest"

        def build(parts):
            seen, kept = set(), []
            for w in sum(parts, []) + label_words:
                stem = Reading.stem(ctx, w)
                if stem in seen:
                    continue
                seen.add(stem)
                kept.append(w)
            lead = ctx.shelf.lead_words
            text = " ".join(kept[:lead] if lead else kept)
            if TextKit.is_number(cap_len):
                while len(text) > cap_len and " " in text:
                    text = text.rsplit(" ", 1)[0]
                text = text[: int(cap_len)]
            return text

        def options(value):
            return value if every_alt else value[:1]

        def unique(forms):
            return list(dict.fromkeys(filter(None, forms)))
        single = unique(build([alt]) for alt in options(values[0]))
        if len(values) == 1:
            return single, []
        widest = 0
        for n, value in enumerate(values):
            if len(options(value)) > 1:
                widest = n
                break
        combined = []
        for alt in options(values[widest]):
            combined.append(build([alt if n == widest else value[0] for n, value in enumerate(values)]))
        return unique(combined), single

    @classmethod
    def vl_next(cls, ctx, first, second, tkey, mode):
        for forms in filter(None, (first, second)):
            totals = [cls.vl_seen(ctx, text, tkey, mode) for text in forms]
            if None in totals:
                return forms[totals.index(None)]
            if any(totals):
                return ""
        return None if (first or second) else ""

    @classmethod
    def vl_mode(cls, ctx, args):
        props = (ctx.shelf.shapes.get(ctx.role("search") or "") or {}).get("properties")
        props = props if isinstance(props, dict) else {}

        def short_choice(key, val):
            spec = props.get(key)
            return isinstance(val, str) and isinstance(spec, dict) and isinstance(spec.get("enum"), list) and len(spec["enum"]) <= 4
        return tuple(sorted(item for item in args.items() if short_choice(*item)))

    @classmethod
    def vl_set(cls, ctx, text):
        limit = ctx.shelf.lead_words or None
        return frozenset(Reading.stem(ctx, w) for w in re.findall(r"[a-z0-9]+", str(text).lower())[:limit])

    @classmethod
    def vl_seen(cls, ctx, text, tkey, mode):
        target = cls.vl_set(ctx, text)
        for key, total in (vars(ctx.shelf).get("ceil_seen") or {}).items():
            try:
                args = json.loads(key)
            except Exception:
                continue
            same_mode = isinstance(args, dict) and isinstance(args.get(tkey), str) and all(args.get(m) == mode[m] for m in mode)
            if same_mode and cls.vl_set(ctx, args[tkey]) == target:
                return int(total)
        return None

    @classmethod
    def vl_indexed(cls, ctx, label):
        if vars(ctx.shelf).get("label_indexed") is not None or not ctx.shelf.archive.items:
            return
        cleaned = re.sub(r"'s\b", "", label.lower())
        label_words = set(Reading.stem(ctx, w) for w in re.findall(r"[a-z0-9]+", cleaned) if len(w) >= 2 and w not in LINK_WORDS)
        counter = [0]
        shelf_name = label.strip().lower()

        def visit(d):
            if Reading.row_mark(ctx, d) not in ctx.shelf.archive.ids or Reading.shelf_of(ctx, d) != shelf_name:
                return
            elsewhere = set()
            for key, val in d.items():
                if not isinstance(val, str) or re.search(r"categor|shelf|department|aisle", str(key).lower()):
                    continue
                elsewhere.update(Reading.stem(ctx, w) for w in re.findall(r"[a-z0-9]+", val.lower()))
            counter[0] += bool(label_words - elsewhere)
        TextKit.walk_dicts(ctx.shelf.archive.items[-1], visit, ceiling=3)
        if counter[0] >= 2:
            ctx.shelf.label_indexed = True

    @classmethod
    def vl_queue(cls, ctx, k, ids, floor, most):
        cheapest = ctx.needs.priority() == "cheapest"
        pick = Walking.walk_pick(ctx, k) if cheapest else None
        batch = Lookup.batch_tool(ctx)
        opened = vars(ctx.shelf).setdefault("valued_opens", {})
        room = most
        if cheapest:
            room = CEIL_OPENS - opened.get(k, 0)
        limit = max(0, min(batch[2] if batch else most, room))
        rows = cls.cheap_unread(ctx, ids, k, pick[1] if pick else None)
        queue = [pair[1] for pair in rows][:limit]
        Walking._trim_queue(ctx, queue, batch, floor)
        if cheapest:
            opened[k] = opened.get(k, 0) + len(queue)
        return queue

    @classmethod
    def vl_launch(cls, ctx, queue, homes, floor, say):
        kept = set()
        for mark in ctx.cart.rank_checked:
            if Judging.shelf_known(ctx, mark.split("|")[0]) not in homes:
                kept.add(mark)
        ctx.cart.rank_checked = kept
        ctx.cart.order_vetted = False
        shelf = ctx.shelf
        shelf.open_queue, shelf.open_recheck, shelf.open_say = queue, True, say
        shelf.open_purpose, shelf.open_floor = "walk", floor
        Judging.open_named(ctx)

    @classmethod
    def _keyword_template(cls, ctx, k):
        for call in Closing.code_queries(ctx, k):
            keys = [key for key, val in call.items() if isinstance(val, str) and val.strip() and val in Reading.free_text(ctx, call)]
            if len(keys) == 1 and not cls.cg_listing(ctx, call):
                return call, keys[0]
        return None, ""

    @classmethod
    def vl_search(cls, ctx):
        if cls.vl_room(ctx)[1] <= 4:
            return False
        shape = ctx.shelf.shapes.get(ctx.role("search") or "") or {}
        props = shape.get("properties") if isinstance(shape.get("properties"), dict) else {}
        found = Reading.page_arg(ctx)
        shelf = vars(ctx.shelf)
        tries = shelf.setdefault("label_tries", {})
        settled = shelf.setdefault("label_done", {})
        sets = shelf.setdefault("label_sets", {})
        products = cls.cg_products(ctx)
        for k in products:
            if tries.get(k, 0) >= LABEL_TRIES:
                continue
            sig = json.dumps(cls.vl_values(ctx, k))
            done_here = settled.setdefault(str(k), [])
            if sig in done_here:
                continue
            home = Walking.walk_home(ctx, k)
            label = cls.raw_label(ctx, home)
            if not label:
                continue
            template, text_key = cls._keyword_template(ctx, k)
            if not template:
                done_here.append(sig)
                continue
            mode, kept = {}, {}
            for key, val in template.items():
                if isinstance(props.get(key), dict) and props[key].get("enum"):
                    mode[key] = val
            required = shape.get("required") or []
            for key, val in template.items():
                if key == text_key or key in mode or key in required:
                    kept[key] = val
            template = kept
            cap_len = props[text_key].get("maxLength") if isinstance(props.get(text_key), dict) else None
            first, second = cls.vl_queries(ctx, k, label, cap_len)
            sets.setdefault(str(k), set()).update(cls.vl_set(ctx, q) for q in first + second)
            todo = cls.vl_next(ctx, first, second, text_key, mode)
            if todo is None:
                todo = ""
                carried = False
                for walk in shelf.get("page_book", {}).values():
                    if Walking.walk_carries(ctx, walk["args"], k) and cls._touches(walk, home):
                        carried = True
                        break
                if carried and shelf.get("label_indexed") is None:
                    ctx.shelf.label_indexed = False
                    first, second = cls.vl_queries(ctx, k, label, cap_len)
                    sets[str(k)].update(cls.vl_set(ctx, q) for q in first + second)
                    todo = cls.vl_next(ctx, first, second, text_key, mode) or ""
            if not todo:
                done_here.append(sig)
                continue
            query = dict(template)
            query[text_key] = todo
            args, faults = ArgCheck.execute(query, shape)
            if faults or not isinstance(args, dict):
                done_here.append(sig)
                continue
            tries[k] = tries.get(k, 0) + 1
            cls._label_search(ctx, k, args, todo, text_key, mode, label, home, found, first or second, len(products))
            return True
        return False

    @classmethod
    def _label_search(cls, ctx, k, args, todo, text_key, mode, label, home, found, leading_forms, product_count):
        plural = "s" if len(cls.vl_values(ctx, k)) > 1 else ""
        Walking.walk_call(ctx, args, f"I search for the required value{plural} together with the category name as this store's rows "
                                     f"spell it ('{todo}'): keyword search matches listing values, titles and categories, so this "
                                     "brings up the category's listings that carry the value.")
        total = cls.vl_seen(ctx, todo, text_key, mode)
        cls.vl_indexed(ctx, label)
        key = json.dumps({a: v for a, v in args.items() if not found or a != found[0]}, sort_keys=True, default=str)
        walk = Trail.book(ctx).get(key)
        every = bool(walk) and len(walk.get("row_shelf") or {}) >= walk["total"] and cls._touches(walk, home)
        floor = Intake.order_due(ctx) + 2
        cheapest = ctx.needs.priority() == "cheapest"
        queue = []
        note = ""
        if total is not None:
            if not total:
                note = f"[value search] '{todo}': no listing matched."
            else:
                note = f"[value search] '{todo}': {total} match(es)" + (" - every one is shown here." if every else ".")
        if every and cheapest and found and walk.get("id_order") is not False:
            shelf = vars(ctx.shelf)
            shelf.setdefault("valued", set()).add(key)
            shelf.setdefault("valued_keys", {}).setdefault(str(k), []).append(key)
            if product_count > 1:
                shelf.setdefault("valued_wait", {})[k] = True
            else:
                queue = cls.vl_queue(ctx, k, cls._shown_on(walk, home), floor, CEIL_OPENS)
        elif every and not cheapest and todo in leading_forms:
            queue = cls.vl_queue(ctx, k, cls._shown_on(walk, home), floor, LABEL_OPENS)
        if queue:
            below_pick = " below the pick" if Walking.walk_pick(ctx, k) and ctx.needs.priority() == "cheapest" else ""
            priced = ", ".join(f"{i} ({ctx.shelf.archive.price(i):g})" for i in queue)
            note += " Its unread in-stock rows within the budget on the requested category" + below_pick + ", cheapest first, are opened next: " + priced + "."
        if note:
            Trail.append_note(ctx, note, closing=not queue)
        Trail.add(ctx, "label_search", product=k, query=todo, total=total, every=every, opened=queue)
        if queue:
            cls.vl_launch(ctx, queue, home, floor, "I open the cheapest unread rows of that search: every listing in the category that "
                                                   "carries the required values is among its matches.")

    @classmethod
    def vl_flush(cls, ctx, floor):
        wait = vars(ctx.shelf).setdefault("valued_wait", {})
        waiting = [k for k in wait if wait[k]]
        wait.clear()
        if not waiting:
            return False
        book = Trail.book(ctx)
        keys = vars(ctx.shelf).get("valued_keys") or {}
        opened = vars(ctx.shelf).setdefault("valued_opens", {})
        batch = Lookup.batch_tool(ctx)
        per = 1 if ctx.moves_left() - floor - 1 < 2 * len(waiting) else 2
        pools = []
        for k in waiting:
            home = Walking.walk_home(ctx, k)
            ids = set()
            for key in keys.get(str(k), []):
                if key in book and Walking.walk_carries(ctx, book[key]["args"], k):
                    ids.update(cls._shown_on(book[key], home))
            pick = Walking.walk_pick(ctx, k)
            cheap = cls.cheap_unread(ctx, ids, k, pick[1] if pick else None)
            pools.append([pair[1] for pair in cheap][:max(0, min(per, CEIL_OPENS - opened.get(k, 0)))])
        owner, queue = {}, []
        most = batch[2] if batch else per * len(waiting)
        while len(queue) < most and any(pools):
            for k, pool in zip(waiting, pools):
                while pool and pool[0] in queue:
                    del pool[0]
                if pool and len(queue) < most:
                    owner[pool[0]] = k
                    queue.append(pool.pop(0))
        Walking._trim_queue(ctx, queue, batch, floor)
        if not queue:
            return False
        for ident in queue:
            opened[owner[ident]] = opened.get(owner[ident], 0) + 1
        homes = set()
        for k in waiting:
            homes |= Walking.walk_home(ctx, k)
        cls.vl_launch(ctx, queue, homes, floor, "Only the cheapest will do: I open the cheapest unread rows that each product's value search turned up.")
        priced = ", ".join(f"{i} ({ctx.shelf.archive.price(i):g})" for i in queue)
        summary = ("[value searches read] The cheapest unread in-stock rows within the budget that each product's value search showed, "
                   "below its pick, were opened: " + priced + ". For each product, "
                   "choose the lowest-priced listing whose own fields meet every requirement now in force; a fit that costs more needs "
                   "every cheaper candidate opened and rejected first.")
        Trail.append_note(ctx, summary, closing=False)
        Trail.add(ctx, "value_flush", products=waiting, opened=queue)
        return True

    @classmethod
    def product_steps(cls, ctx):
        try:
            rows = ctx.needs.items()
            if len(rows) < 2 or ctx.needs.priority() != "cheapest":
                return ""
            used = vars(ctx.shelf).get("valued_by") or {}
            parts, upcoming = [], ""
            for k, row in enumerate(rows):
                has_pick = bool(Walking.walk_pick(ctx, k))
                parts.append(f"{row['product']}: {used.get(k, 0)} value page(s) read, pick {'yes' if has_pick else 'no'}")
                if not upcoming and not has_pick:
                    upcoming = str(row["product"])
            tail = f"; next: {upcoming}" if upcoming else ""
            return "\n[product steps] " + "; ".join(parts) + tail + "."
        except Exception:
            return ""

    @classmethod
    def value_pages(cls, ctx):
        try:
            refs = vars(ctx.pointers).get("refs") or []
            if any(ref.get("status") == "open" for ref in refs) and Referencing.ref_room(ctx):
                return False
            if ctx.cart.orders_tried or ctx.cart.refusal_n or Trail.no_time(ctx, 60) or not ctx.role("search"):
                return False
            if cls.vl_search(ctx):
                return True
            if ctx.needs.priority() != "cheapest" or ctx.shelf.word_kind == "any" or not Reading.page_arg(ctx):
                return False
            return cls._value_page_turn(ctx)
        except Exception as err:
            Trail.fault(ctx, "value_pages_error", err)
            return False

    @classmethod
    def _value_page_turn(cls, ctx):
        shelf = vars(ctx.shelf)
        book = shelf.get("page_book", {})
        done = shelf.setdefault("valued", set())
        used = shelf.setdefault("valued_by", {})
        opened = shelf.setdefault("valued_opens", {})
        wait = shelf.setdefault("valued_wait", {})
        sets = shelf.get("label_sets") or {}
        depth = cls.visible_depth(ctx)
        size = max(1, cls.page_size(ctx))
        per = VALUE_PAGES if not depth else max(VALUE_PAGES, -(-depth // size) - 1)
        products = cls.cg_products(ctx)
        several = len(products) > 1
        cap = max(VALUE_TOTAL, VALUE_TOTAL_PER * len(products))
        opens = 1 if Lookup.batch_tool(ctx) else (2 if several else CEIL_OPENS)
        floor = Intake.order_due(ctx) + 2
        known = cls.all_words(ctx)
        left, room = cls.vl_room(ctx)
        fresh = [k for k in products if Walking.walk_home(ctx, k) and not used.get(k)]
        share = max(4, room // max(1, len(fresh)))
        limits = {"per": per, "cap": cap, "share": share, "opens": opens, "floor": floor, "depth": depth, "known": known,
                  "left": left, "room": room, "several": several}
        options, pending = [], False
        for n, k in enumerate(products):
            home = Walking.walk_home(ctx, k)
            if not home:
                continue
            cands, waits = cls._page_candidates(ctx, k, home, book, done, used, opened, sets.get(str(k)) or set(), limits)
            pending = waits or pending
            for take, score, key, _ in cands:
                options.append((used.get(k, 0), take, score, n, key, k))
        if options:
            chosen = min(options, key=lambda o: o[:5])
            done.add(chosen[4])
            return cls.read_value_search(ctx, chosen[5], chosen[4], floor, chosen[1], several)
        if wait and not pending and not (left and room > opens * len(wait) + 4):
            return cls.vl_flush(ctx, floor)
        return False

    @classmethod
    def _page_candidates(cls, ctx, k, home, book, done, used, opened, composed_sets, limits):
        depth, known = limits["depth"], limits["known"]
        values = Walking._value_stems(ctx, k)
        label = set()
        for name in home:
            label.update(Reading.stem(ctx, w) for w in re.findall(r"[a-z0-9]{2,}", name))
        reach_words = values | label
        finished = []
        for walk in book.values():
            if walk.get("id_order") is not True or not walk.get("first") or Reading.next_page(ctx, walk) is not None:
                continue
            if cls.cg_listing(ctx, walk["args"]) or len(walk.get("row_shelf") or {}) < walk["total"]:
                continue
            finished.append((cls.vl_set(ctx, " ".join(Reading.free_text(ctx, walk["args"]))),
                             Walking.walk_carries(ctx, walk["args"], k, alts=True), cls.vl_mode(ctx, walk["args"])))
        cands, pending = [], False
        for key, walk in book.items():
            if key in done or walk.get("id_order") is not True or not walk.get("first") or Reading.next_page(ctx, walk) is None:
                continue
            if cls.cg_listing(ctx, walk["args"]) or not Walking.walk_carries(ctx, walk["args"], k) or not cls._touches(walk, home):
                continue
            words = cls.vl_set(ctx, " ".join(Reading.free_text(ctx, walk["args"])))
            composed = words in composed_sets
            too_deep = depth is not None and walk["total"] > depth
            if (known and too_deep and not composed) or (not known and (depth is None or too_deep)):
                continue
            pairs = Walking.walk_carries(ctx, walk["args"], k, alts=True)
            how = cls.vl_mode(ctx, walk["args"])
            if known and any(mode == how and (seen <= words or (seen <= reach_words and pairs <= got)) for seen, got, mode in finished):
                continue
            reach = walk["total"] if depth is None else min(walk["total"], depth)
            pages = -(-reach // max(1, walk["size"])) - len(walk["read"])
            allowed = min(limits["per"] - used.get(k, 0), limits["cap"] - sum(used.values()))
            if limits["several"]:
                first_open = 0 if opened.get(k) else limits["opens"]
                allowed = min(allowed, limits["share"] - used.get(k, 0) - opened.get(k, 0) - first_open)
            take = pages if pages <= allowed else (allowed if composed else 0)
            if take <= 0 or ctx.moves_left() - take - limits["opens"] <= limits["floor"]:
                continue
            if (take > 2 or limits["several"]) and limits["left"] and limits["room"] > take + limits["opens"] + 4:
                pending = True
                continue
            cands.append((take, -Walking.walk_score(ctx, walk["args"], k), key, words))
        if known:
            def covered(c, o):
                if o[3] < c[3]:
                    return not (c[3] - o[3]) <= reach_words
                return c[3] < o[3] and (o[3] - c[3]) <= reach_words
            cands = [c for c in cands if not any(covered(c, o) for o in cands)]
        return cands, pending

    @classmethod
    def read_value_search(cls, ctx, k, key, floor, most=VALUE_PAGES, defer=False):
        found = Reading.page_arg(ctx)
        walk = Trail.book(ctx).get(key)
        if not (found and walk):
            return False
        query = " ".join(Reading.free_text(ctx, walk["args"]))
        used = vars(ctx.shelf).setdefault("valued_by", {})
        known = cls.all_words(ctx)
        if known:
            say = ("Only the cheapest will do, and this store's keyword search requires every word and returns results in a fixed id "
                   f"order rather than by price: I read the next page of '{query}' so every listing carrying these words gets seen.")
        else:
            say = ("Only the cheapest will do, and this store returns search results in a fixed id order rather than by price: I read "
                   f"the next page of '{query}' so every listing it matched gets seen.")
        got, tried = 0, False
        while Reading.next_page(ctx, walk) is not None and got < most and ctx.moves_left() > floor + 1 and not Trail.no_time(ctx, 30):
            tried = True
            used[k] = used.get(k, 0) + 1
            token = Reading.next_page(ctx, walk)
            page_args = dict(walk["args"])
            page_args[found[0]] = token
            if not Walking.walk_call(ctx, page_args, say):
                break
            got += 1
            if Reading.next_page(ctx, walk) == token:
                break
        if got == 0:
            return tried
        vars(ctx.shelf).setdefault("valued_keys", {}).setdefault(str(k), []).append(key)
        seen = len(walk.get("row_shelf") or {})
        exhausted = Reading.next_page(ctx, walk) is None
        complete = exhausted and seen >= walk["total"]
        home = Walking.walk_home(ctx, k)
        if complete:
            reach = " - every listing carrying these words is among them" if known else " - every listing it matched is among them"
        elif Reading.next_page(ctx, walk) is None:
            reach = " - the rest cannot be reached, so a cheaper fit may lie past the reachable rows"
        else:
            reach = ""
        head = f"[value search read] '{query}': {seen} of its {walk['total']} matches seen{reach}. "
        closing = ("Choose the lowest-priced listing whose own fields meet every requirement now in force; a fit that costs more "
                   "needs every cheaper candidate opened and rejected first.")
        queue = []
        if defer:
            vars(ctx.shelf).setdefault("valued_wait", {})[k] = True
            middle = "Its cheapest unread in-stock rows within the budget are opened once every product's value search is read. "
        else:
            pick = Walking.walk_pick(ctx, k)
            queue = cls.vl_queue(ctx, k, cls._shown_on(walk, home), floor, CEIL_OPENS)
            if queue:
                priced = ", ".join(f"{i} ({ctx.shelf.archive.price(i):g})" for i in queue)
                middle = ("Its cheapest unread in-stock rows within the budget on the requested category" + (" below the pick" if pick else "")
                          + " are opened next: " + priced + ". ")
                cls.vl_launch(ctx, queue, home, floor, "Only the cheapest will do: I open the cheapest unread rows of that search.")
            else:
                middle = "No unread in-stock row within the budget on the requested category is left among them. "
        Trail.append_note(ctx, head + middle + closing, closing=not queue)
        Trail.add(ctx, "value_pages", product=k, query=query, pages=got, complete=complete, opened=queue, defer=defer)
        return True

    @classmethod
    def ceiling_walk(cls, ctx, k, floor, repair=False):
        spec, found = cls.ceil_spec(ctx), Reading.page_arg(ctx)
        home = Walking.walk_home(ctx, k)
        label = cls.raw_label(ctx, home)
        aim = cls.ceil_aim(ctx, k)
        if not spec or not found or not label or aim is None:
            return None
        shelf = vars(ctx.shelf)
        shelf.setdefault("ceil_walked", set()).add(k)
        pick_now = Walking.walk_pick(ctx, k)
        shelf.setdefault("ceil_aimed", {})[k] = (float(aim), pick_now[0] if pick_now else "")
        bound = cls.below(ctx, aim)
        if bound is None:
            return None
        if repair:
            floor += 2
        elif not ctx.cart.orders_tried:
            floor = max(floor, Intake.order_due(ctx) + 2)
        walk_state = {"calls": 0, "top": None, "complete": False, "ran": False, "last": bound, "keys": []}
        what = "the pick" if pick_now or TextKit.is_number((shelf.get("ceil_aims") or {}).get(k)) else "the cheapest row seen"
        cls._lower_ceiling(ctx, spec, found, label, home, aim, bound, floor, what, walk_state)
        if not walk_state["ran"]:
            return None
        book = shelf.setdefault("page_book", {})
        ids = set()
        for key in walk_state["keys"]:
            ids.update(cls._shown_on(book.get(key, {}), home))
        cands = [pair[1] for pair in cls.cheap_unread(ctx, ids, k, aim)]
        valued = cls.valued_ids(ctx, k)
        lead = [i for i in cands if i in valued]
        rest = [i for i in cands if i not in valued]
        texty = [i for i in rest if cls.row_carries(ctx, i, k)]
        bare = [] if cls.valued_complete(ctx, k) else [i for i in rest if i not in texty]
        ranked = (lead + texty + bare)[:CEIL_OPENS]
        complete, last = walk_state["complete"], walk_state["last"]
        if complete and not ids:
            report = f"the price filter on '{label}' below {aim:g}: no listing of it costs less than {what}"
        elif complete:
            report = (f"the price filter on '{label}': every listing priced up to {last:g} seen ({len(ids)}, {len(cands)} of them unread, "
                      "in stock and within the budget)")
        else:
            report = f"the price filter on '{label}' below {aim:g}: {len(ids)} of the cheaper listings seen ({len(cands)} unread, in stock and within the budget)"
        Trail.add(ctx, "ceiling_walk", product=k, label=label, aim=aim, calls=walk_state["calls"], complete=complete,
                  rows=len(ids), ranked=ranked)
        return report, ranked

    @classmethod
    def _lower_ceiling(cls, ctx, spec, found, label, home, aim, bound, floor, what, state):
        shelf = vars(ctx.shelf)
        depth = cls.visible_depth(ctx)
        book = shelf.setdefault("page_book", {})
        say = f"Only the cheapest will do: I use the store's price filter to list '{label}' priced under {what} ({aim:g})."

        def can_call():
            return state["calls"] < CEIL_CALLS and ctx.moves_left() > floor + 1 and not ctx.wire.done and not ctx.wire.broke \
                and ctx.wire.spare() >= CLOSING + 30
        while can_call():
            args = cls.ceil_args(ctx, spec, label, bound)
            key = json.dumps(args, sort_keys=True, default=str)
            shelf.setdefault("ceil_keys", set()).add(key)
            state["calls"] += 1
            reuse = key in book and bool(book[key].get("first")) and key in (shelf.get("ceil_seen") or {})
            if not reuse and not Walking.walk_call(ctx, args, say):
                return
            state["ran"], state["last"] = True, bound
            state["keys"].append(key)
            walk = book.get(key)
            archive = ctx.shelf.archive
            if walk and any(TextKit.is_number(archive.price(i)) and archive.price(i) > bound for i in (walk.get("row_shelf") or {})):
                shelf.setdefault("ceil_bad", set()).add(spec["sig"])
            if cls.ceil_spec(ctx) is None:
                return
            total = shelf.get("ceil_seen", {}).get(key)
            if total is None:
                return
            if not total or not walk:
                if cls.ceil_clash(ctx, home, bound):
                    shelf.setdefault("ceil_bad", set()).add(spec["sig"])
                    return
                if state["top"] is None:
                    state["complete"] = True
                    return
                middle = cls.ceil_mid(ctx, bound, state["top"])
                if not bound < middle < state["top"]:
                    return
                bound = middle
                continue
            pages = -(-total // max(1, walk["size"]))
            if (depth is None or total <= depth) and pages - len(walk["read"]) <= CEIL_CALLS - state["calls"]:
                while Reading.next_page(ctx, walk) is not None and can_call():
                    state["calls"] += 1
                    token = Reading.next_page(ctx, walk)
                    more = dict(args)
                    more[found[0]] = token
                    if not Walking.walk_call(ctx, more, "I read the next page of that price-filtered listing.") \
                            or Reading.next_page(ctx, walk) == token:
                        break
                state["complete"] = Reading.next_page(ctx, walk) is None
                return
            state["top"] = bound
            costs = [float(c) for c in map(archive.price, walk.get("row_shelf") or {}) if TextKit.is_number(c)]
            low = min(costs) if costs else 0.0
            fill = max(1, CEIL_CALLS - state["calls"]) * max(1, walk["size"])
            lower = min(cls.ceil_mid(ctx, low, bound), cls.ceil_mid(ctx, low, low + (bound - low) * 1.8 * fill / max(total, 1)))
            if not lower < bound:
                return
            bound = lower

    @classmethod
    def ceiling_sweep(cls, ctx, tried):
        spec, found = cls.ceil_spec(ctx), Reading.page_arg(ctx)
        if not spec or not found or ctx.needs.items():
            return None, False
        rows = Reading.shelf_rows(ctx)
        shelves = Reading.shelf_labels(ctx)
        label = cls.raw_label(ctx, shelves)
        if not (rows and label):
            return None, False
        bound = cls.below(ctx, min(entry[0] for entry in rows.values()))
        if bound is None:
            return None, True
        args = cls.ceil_args(ctx, spec, label, bound)
        key = json.dumps(args, sort_keys=True, default=str)
        if key not in tried:
            vars(ctx.shelf).setdefault("ceil_keys", set()).add(key)
            return args, False
        walk = Trail.book(ctx).get(key)
        if not walk:
            if vars(ctx.shelf).get("ceil_seen", {}).get(key) != 0:
                return None, False
            return None, not cls.ceil_clash(ctx, set(shelves), bound)
        token = Reading.next_page(ctx, walk)
        if token is None:
            return None, len(walk.get("row_shelf") or {}) >= walk["total"]
        more = dict(args)
        more[found[0]] = token
        return (None, False) if json.dumps(more, sort_keys=True, default=str) in tried else (more, False)

    @classmethod
    def value_guard(cls, ctx, call_name, call_args):
        try:
            return cls._guard_note(ctx, call_name, call_args)
        except Exception as err:
            Trail.fault(ctx, "value_guard_error", err)
            return ""

    @classmethod
    def _guard_note(cls, ctx, call_name, call_args):
        if call_name != ctx.role("details") or ctx.needs.priority() != "cheapest":
            return ""
        if not cls.all_words(ctx) or Intake.in_reserve(ctx):
            return ""
        archive = ctx.shelf.archive
        named = [v for v in call_args.values() if isinstance(v, str) and v in archive.ids]
        if len(named) != 1:
            return ""
        ident = named[0]
        if ident in archive.read or ident in ctx.cart.trolley.units:
            return ""
        if ident in Intake.named_open(ctx) or vars(ctx.pointers).get("refs") or ctx.needs.reading.get("references"):
            return ""
        k = Walking.product_index(ctx, ident)
        if ctx.needs.items() and k is None:
            return ""
        mark = (k, ctx.needs.version)
        told = vars(ctx.talk).setdefault("vguard_told", set())
        if mark in told:
            return ""
        valued = cls.valued_ids(ctx, k)
        if not valued or ident in valued:
            return ""
        pick = Walking.walk_pick(ctx, k)
        own_price = archive.price(ident)
        costs = [pick[1]] if pick else []
        if TextKit.is_number(own_price):
            costs.append(float(own_price))
        rows = cls.cheap_unread(ctx, valued, k, min(costs))[:6] if costs else []
        if not rows:
            return ""
        told.add(mark)
        listed = ", ".join(f"{i} ({c:g})" for c, i in rows)
        return ("Not run (no step used). The cheapest is asked for, and this store's keyword search needs every word, so the rows "
                f"its searches led by a required value showed carry those words somewhere in their records; {ident} is not among "
                "them, while these rows of theirs cost less" + (" than it and the pick" if pick else " than it") + " and are unread: "
                + listed + ". Open them first, cheapest first; send this call again if you "
                "still want it.")

    @classmethod
    def title_line(cls, ctx, k, rec, template):
        avoid = set(LINK_WORDS)
        for name in Walking.walk_home(ctx, k):
            avoid.update(re.findall(r"[a-z]+", name))
        for req in ctx.needs.props_left():
            avoid.update(re.findall(r"[a-z0-9]+", req.lower()))
        for row in ctx.needs.items() or [{"product": str(ctx.needs.reading.get("product") or "")}]:
            avoid.update(re.findall(r"[a-z]+", str(row.get("product") or "").lower()))
        stems = set(map(lambda w: Reading.stem(ctx, w), avoid))

        def plain(word):
            return word not in avoid and Reading.stem(ctx, word) not in stems
        tokens = re.findall(r"[a-z0-9]+", Lookup.title_of(ctx, rec).lower())
        for n, tok in enumerate(tokens):
            if re.fullmatch(r"[a-z]{3,}", tok) and plain(tok):
                following = tokens[n + 1] if n + 1 < len(tokens) else ""
                text = tok + (f" {following}" if following and plain(following) else "")
                if Walking.ran_query(ctx, text):
                    return None
                query = dict(template[0])
                query[template[1]] = text
                return query
        return None

    @classmethod
    def rearm_walk(cls, ctx):
        if ctx.needs.priority() != "cheapest" or ctx.wire.done or ctx.wire.spare() < CLOSING + 60 or not ctx.role("search"):
            return None
        aimed = vars(ctx.shelf).get("ceil_aimed") or {}
        gone = Lookup.dead_lines(ctx) | ctx.cart.refused_lines
        opens = 1 if Lookup.batch_tool(ctx) else 2
        for k, (aim, ident) in list(aimed.items()):
            if not ident or ident not in gone:
                continue
            if cls.ceil_spec(ctx) is None or Reading.page_arg(ctx) is None:
                continue
            pick = Walking.walk_pick(ctx, k)
            if not pick or pick[1] <= aim + 1e-9 or ctx.moves_left() < FIRST_ORDER_BY + 3 + opens:
                continue
            aimed.pop(k, None)
            (vars(ctx.shelf).get("ceil_aims") or {}).pop(k, None)
            Trail.add(ctx, "ceil_rearm", product=k, dead=ident, aim=pick[1])
            return Walking.run_walk(ctx, [], FIRST_ORDER_BY, [k], repair=True)
        return None

    @staticmethod
    def _shelf_argument(props, label, skip):
        wanted = label.strip().lower()
        for key, spec in props.items():
            if not isinstance(spec, dict) or key in skip or not re.search(r"categor|shelf|department|aisle", str(key).lower()):
                continue
            choices = spec.get("enum")
            if isinstance(choices, list):
                for val in choices:
                    if isinstance(val, str) and val.strip().lower() == wanted:
                        return key, val
            elif spec.get("type") == "string":
                return key, label
        return None, None

    @staticmethod
    def _limit_argument(props, page_key):
        for key, spec in props.items():
            if not isinstance(spec, dict) or spec.get("type") not in ("integer", "number") or spec.get("enum") or key == page_key:
                continue
            low = str(key).lower()
            if re.search(r"ceil|max|price|cap$|budget|upper|most", low) and not re.search(r"page|first|offset|skip|count|size|num", low):
                return key
        return None

    @classmethod
    def seed_args(cls, ctx, label, bound):
        tool = ctx.role("search")
        shape = ctx.shelf.shapes.get(tool or "") or {}
        props = shape.get("properties") if isinstance(shape.get("properties"), dict) else {}
        found = Reading.page_arg(ctx)
        if not props or not found or not TextKit.is_number(bound):
            return None
        described = [tdef for tdef in ctx.shelf.toolset if tdef["function"]["name"] == tool]
        told = str(described[0]["function"].get("description") or "") if described else ""
        sentences = [part.lower() for part in re.split(r"[.;]", told)]
        token = Reading.token_key(ctx)
        shelf_key, shelf_val = cls._shelf_argument(props, label, (token, found[0]))
        bound_key = cls._limit_argument(props, found[0])
        if not (shelf_key and bound_key):
            return None
        whole = props[bound_key].get("type") == "integer" or float(bound).is_integer()
        args = {shelf_key: shelf_val, bound_key: int(bound) if whole else bound}
        shelf_word = r"\b" + re.escape(shelf_key.lower()) + r"\b"
        for key, spec in props.items():
            if key == shelf_key or not isinstance(spec, dict) or not isinstance(spec.get("enum"), list) or not 2 <= len(spec["enum"]) <= 4:
                continue
            paired = [val for val in spec["enum"]
                      if any(re.search(r"\b" + re.escape(str(val).lower()) + r"\b", part) and re.search(shelf_word, part) for part in sentences)]
            if not paired:
                return None
            args[key] = paired[0]
        missing = [key for key in shape.get("required") or [] if key not in args]
        if missing:
            return None
        fixed, faults = ArgCheck.execute(args, shape)
        if faults or not isinstance(fixed, dict):
            return None
        return fixed

    @classmethod
    def ceil_seed(cls, ctx):
        try:
            if cls.ceil_spec(ctx) is not None or vars(ctx.shelf).get("ceil_mode") or ctx.needs.priority() != "cheapest":
                return False
            seeded = vars(ctx.shelf).setdefault("ceil_seeded", set())
            found = Reading.page_arg(ctx)
            for k in cls.cg_products(ctx):
                if k in seeded or (ctx.needs.items() and k is None):
                    continue
                pick = Walking.walk_pick(ctx, k)
                label = cls.raw_label(ctx, Walking.walk_home(ctx, k))
                if not (pick and label):
                    continue
                seeded.add(k)
                args = cls.seed_args(ctx, label, cls.below(ctx, pick[1]))
                if not args:
                    return False
                if cls._send_seed(ctx, k, args, label, pick, found):
                    return True
            return False
        except Exception as err:
            Trail.fault(ctx, "ceil_seed_error", err)
            return False

    @classmethod
    def _send_seed(cls, ctx, k, args, label, pick, found):
        mode = dict(Reading.search_keys(ctx, args)[1])
        page_key = found[0] if found else None
        fixed = sorted((m, v) for m, v in mode.items() if not (isinstance(args.get(m), str) and args[m] == label))
        limit_key = next((a for a in args if TextKit.is_number(args[a]) and a != page_key), "")
        sig = json.dumps([fixed, limit_key], default=str)
        key = json.dumps({a: v for a, v in args.items() if not found or a != found[0]}, sort_keys=True, default=str)
        if sig in vars(ctx.shelf).get("ceil_bad", set()) or key in (vars(ctx.shelf).get("ceil_seen") or {}):
            return False
        vars(ctx.shelf).setdefault("ceil_seed_keys", set()).add(key)
        Walking.walk_call(ctx, args, f"Only the cheapest will do: I list '{label}' priced under the pick ({pick[1]:g}) to check whether "
                                     "this store's price filter holds.")
        Trail.add(ctx, "ceil_seed", product=k, args=args, learned=cls.ceil_spec(ctx) is not None)
        chat = ctx.talk.chat
        if chat and chat[-1].get("role") == "tool" and cls.ceil_spec(ctx) is None:
            chat[-1]["content"] = str(chat[-1]["content"]) + "\n" + ctx.status_line()
        return True

    @classmethod
    def unfound_product(cls, ctx):
        if not ctx.needs.items():
            return False
        try:
            read = set(name.strip().lower() for name in ctx.shelf.archive.shelves_read())
            for home in Lookup.home_shelves(ctx):
                if not home or not home & read:
                    return True
            return bool(Trail.live_ids(ctx)) and Intake.unverified(ctx)
        except Exception:
            return False

    @classmethod
    def walk_hold(cls, ctx, call_args):
        try:
            return cls._hold_note(ctx, call_args)
        except Exception as err:
            Trail.fault(ctx, "walk_hold_error", err)
            return ""

    @classmethod
    def _hold_note(cls, ctx, call_args):
        if ctx.needs.priority() != "cheapest" or vars(ctx.shelf).get("walk_queue") or ctx.wire.spare() < CLOSING + 60:
            return ""
        if ctx.cart.refusal_n and ctx.moves_left() <= 15:
            return ""
        trolley = ctx.cart.trolley
        if call_args is None:
            ids = list(trolley.units)
        else:
            ids = [ident for ident, count in trolley.lines(call_args) if count is None or count > 0]
        only = set(Walking.product_index(ctx, ident) for ident in ids if ident)
        floor = 4 if ctx.cart.orders_tried else FIRST_ORDER_BY + 1
        half = (ctx.moves_left() - floor - 1) // 2
        room = half - (1 if Lookup.batch_tool(ctx) else 2)
        ceil = []
        aims = vars(ctx.shelf).setdefault("ceil_aims", {})
        for k in sorted(only, key=lambda x: -1 if x is None else x):
            if room < 2 or not cls.ceiling_ready(ctx, k):
                continue
            ceil.append(k)
            room -= CEIL_CALLS
            prices = [ctx.shelf.archive.price(ident) for ident in ids if ident and Walking.product_index(ctx, ident) == k]
            costs = [float(p) for p in prices if TextKit.is_number(p)]
            if not costs:
                continue
            if TextKit.is_number(aims.get(k)):
                costs.append(float(aims[k]))
            aims[k] = min(costs)
        plan = Walking.walk_plan(ctx, min(WALK_TOTAL, room), only)
        if not (plan or ceil):
            return ""
        ctx.shelf.walk_queue = (plan, max(floor, ctx.moves_left() - half - 1), ceil)
        book = Trail.book(ctx)
        queries = [" ".join(Reading.free_text(ctx, book[key]["args"])) for _, key, _ in plan if key in book]
        parts = []
        if queries:
            parts.append("reads the remaining pages of " + ", ".join(f"'{q}'" for q in queries))
        if ceil:
            parts.append("lists the category's listings priced below the pick with the store's price filter")
        return ("Not run yet (no step used). The cheapest is asked for, and this store lists search results in a fixed id order, "
                "not by price, so a cheaper fit may sit on pages never read: the code " + " and ".join(parts)
                + " for you now and opens the cheapest unread rows below the pick. Then send the call again, or the same call "
                "with a cheaper fit.")

    @classmethod
    def search_note(cls, ctx, call_args, obs):
        if not isinstance(obs, dict):
            return ""
        rows = cls._row_list(obs)
        if rows is None:
            nested = next((val for val in obs.values() if isinstance(val, dict) and cls._row_list(val) is not None), None)
            if nested is None:
                return ""
            obs = nested
            rows = cls._row_list(obs) or []
        totals = []
        for val in obs.values():
            if isinstance(val, bool) or not isinstance(val, int):
                continue
            if val > len(rows):
                totals.append(val)
        biggest = max(totals) if totals else len(rows)
        Reading.book_page(ctx, call_args, rows, biggest, obs)
        try:
            cls.ceil_learn(ctx, call_args, rows, biggest, obs)
        except Exception as err:
            Trail.fault(ctx, "ceil_learn_error", err)
        score = Reading.value_score(ctx, call_args) if rows else 0
        if score:
            row_scores = vars(ctx.shelf).setdefault("row_score", {})
            for row in filter(lambda r: isinstance(r, dict), rows):
                mark = Reading.row_mark(ctx, row)
                row_scores[mark] = max(row_scores.get(Reading.row_mark(ctx, row), 0), score)
        remarks = []
        kind = Reading.word_rule(ctx, call_args, obs, rows)
        lead = Reading.lead_rule(ctx, call_args, obs, rows)
        if lead:
            texts, mode, _, _ = Reading.search_keys(ctx, call_args)
            lead_ctx = ctx.shelf.lead_ctx
            words = ()
            if mode == lead_ctx[1]:
                words = next((w for key, w in texts if key == lead_ctx[0]), ())
            counted = f" (this query counted only: {' '.join(words[:lead])})" if len(words) > lead else ""
            remarks.append(f"This store's keyword search ({lead_ctx[0]}) reads only the FIRST {lead} word(s) of a query and "
                           f"ignores the rest{counted}: put the {lead} most distinctive words first (a required value and the item "
                           "kind, the value first), and search other values in separate short queries.")
        texts, mode, first, paging_keys = Reading.search_keys(ctx, call_args)
        mode_keys = dict(mode)
        raw = [val for key, val in call_args.items() if isinstance(val, str) and key not in mode_keys and key not in paging_keys]
        if first and raw:
            cls._query_remarks(ctx, call_args, raw, rows, totals, texts, mode, paging_keys, kind, remarks)
        if not rows:
            return " ".join(remarks)
        if totals:
            remarks.append(cls._depth_remark(kind, max(totals), len(rows)))
        paging = Reading.search_keys(ctx, call_args)[3]
        cap = None
        for key, val in call_args.items():
            if not TextKit.is_number(val) or val <= 1 or key in paging:
                continue
            if re.search(r"page|size|count|result|min|low|from|start", str(key).lower()):
                continue
            cap = val if cap is None else max(cap, val)
        over = []
        if cap is not None:
            over = list(filter(lambda row: cls._priced_over(row, cap), rows))
        if over:
            remarks.append(f"{len(over)} of these rows cost more than the limit {cap} this call set: this search mode ignored that filter "
                           "(and may ignore others, such as the category). Use the mode whose filters hold, or check every row yourself.")
        return " ".join(remarks)

    @staticmethod
    def _row_list(piece):
        lists = [val for val in piece.values() if isinstance(val, list) and all(isinstance(r, dict) for r in val)]
        return next((val for val in lists if val), lists[0] if lists else None)

    @staticmethod
    def _priced_over(row, cap):
        return any(TextKit.is_number(val) and val > cap and "price" in str(key).lower() for key, val in row.items())

    @staticmethod
    def _depth_remark(kind, total, shown):
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
        return (f"This search matched {total} listings but shows only {shown} on this page, and only a few pages can "
                "be read: the listings that matter may be among the unseen ones. Read the further pages of this same search "
                "first (as many as can be read); if the total is larger than those pages, " + how)

    @classmethod
    def _query_remarks(cls, ctx, call_args, raw, rows, totals, texts, mode, paging_keys, kind, remarks):
        shelf = vars(ctx.shelf)
        talk = vars(ctx.talk)
        pieces = [p for text in raw for p in re.findall(r"[a-z0-9]+", text.lower())]
        if pieces and all(len(p) == 1 for p in pieces) and not rows and not talk.get("short_told"):
            ctx.talk.short_told = True
            remarks.append("This query found nothing: this store's search skips one-character words, so a figure like 0.4 or "
                           "5 x 7 searches nothing (or only its other words). Search a word value, or the unit word with the item "
                           "kind, and read every page.")
        prop, probe = Reading.value_probe(ctx, raw)
        told = talk.get("name_told", 0)
        roomy = lambda: ctx.moves_left() > Intake.order_due(ctx) + 3
        if probe and told < 2 and roomy():
            ctx.talk.name_told = told + 1
            if told:
                lead_in = f"Again led by a property's name ({prop})."
            else:
                lead_in = (f"This query leads with words of a property's NAME ({prop}) and none of a required value: a keyword "
                           "search matches listing values, titles and categories, and a field's name only where some value or "
                           f"title repeats it, so it misses listings whose {prop} field holds just the value.")
            remarks.append(lead_in + f" Search the value first, then the item kind: '{probe}' (a figure the store cannot read drops out, "
                           "leaving its unit word), and read every page of it.")
        elif not probe and talk.get("trap_told", 0) < 2 and roomy():
            trap = Walking.word_trap(ctx, raw)[0]
            if trap:
                ctx.talk.trap_told = talk.get("trap_told", 0) + 1
                remarks.append(trap)
        marks = tuple(Reading.row_mark(ctx, row) for row in rows)
        mode_keys = dict(mode)
        others = tuple(sorted((k2, str(v2)) for k2, v2 in call_args.items()
                              if k2 not in paging_keys and not (isinstance(v2, str) and k2 not in mode_keys)))
        sig = (mode, others, max(totals or [len(rows)]), marks)
        sigs = shelf.setdefault("page_sigs", {})
        query = " ".join(raw)
        if len(set(marks)) >= 3 and sigs.get(sig, query) != query:
            remarks.append(f"Same total and rows as your earlier search '{sigs[sig]}': the store read both alike, so re-spelling "
                           "it shows nothing new. Read its next page, or change its leading words.")
        sigs.setdefault(sig, query)
        hits = cls._booked_hits(ctx, call_args, raw, rows, totals, texts, kind, remarks)
        if (kind == "all" or ctx.shelf.lead_words) and not shelf.get("hits_back"):
            cls._hits_back(ctx, call_args, paging_keys, hits, remarks)

    @classmethod
    def _booked_hits(cls, ctx, call_args, raw, rows, totals, texts, kind, remarks):
        shelf = vars(ctx.shelf)
        hits = []
        told_all = bool((kind == "all" or ctx.shelf.lead_words) and not totals and rows)
        if not told_all and not (kind != "any" and not totals and rows):
            return hits
        held = None if told_all else list(vars(ctx.pointers).get("ref_hits") or [])
        ctx.pointers.ref_hits = []
        hits = Reading.row_hits(ctx, rows, raw)
        score = Reading.value_score(ctx, call_args)
        ranks = shelf.setdefault("hit_score", {})
        for ident in hits:
            ranks[ident] = max(ranks.get(ident, 0), score)
        found = shelf.setdefault("all_hits", [])
        found += [ident for ident in hits if ident not in found]
        if held is not None:
            ctx.pointers.ref_hits = held
        if told_all and hits and len(texts) and len(rows) <= 12:
            remarks.append("Every word is required here and every match is shown: each row carries the words the store read "
                           "somewhere in its record, often in fields its title does not show. Open the in-budget ones before "
                           "searching on: " + ", ".join(hits[:6]) + ".")
        pointers = []
        if told_all:
            pointers = [ident for ident in vars(ctx.pointers).pop("ref_hits", []) if ident not in hits][:6]
        if pointers:
            remarks.append("The request gives a value by pointing at another listing, and these rows are sold by the shop or brand "
                           "this search names: open them whatever their price, stock or category (a listing that gives a value "
                           "is read, not bought), and follow what its notes say: " + ", ".join(pointers) + ".")
        return hits

    @classmethod
    def _hits_back(cls, ctx, call_args, paging_keys, hits, remarks):
        shelf = vars(ctx.shelf)
        ctx.shelf.hits_back = True
        here = json.dumps({k: v for k, v in call_args.items() if k not in paging_keys}, sort_keys=True, default=str)
        back = []
        for walk_key, walk in shelf.get("page_book", {}).items():
            if walk_key == here or not walk.get("every") or not Reading.value_score(ctx, walk["args"]):
                continue
            fresh = [ident for ident in Reading.row_hits(ctx, walk["every"]) if ident not in back and ident not in hits]
            ranks = shelf.setdefault("hit_score", {})
            for ident in fresh:
                ranks[ident] = max(ranks.get(ident, 0), Reading.value_score(ctx, walk["args"]))
            back += fresh
        found = shelf.setdefault("all_hits", [])
        found += [ident for ident in back if ident not in found]
        if back:
            remarks.append("This store's keyword search needs every word, so earlier searches that showed every match listed only "
                           "rows carrying all of their words, often in fields the titles do not show. Open the in-budget ones not "
                           "read yet before searching on: " + ", ".join(back[:6]) + ".")

    @classmethod
    def shopper_lines(cls, ctx, call_name, obs, refused):
        lines = []
        if call_name == ctx.role("message"):
            if isinstance(obs, str):
                lines = [obs]
            elif isinstance(obs, dict) and "error" not in obs:
                TextKit.walk_dicts(obs, lambda d: TextKit.long_strings([val for val in d.values() if isinstance(val, (str, list))], lines),
                                   ceiling=4)
        elif isinstance(obs, dict):
            lines = cls.notice_lines(ctx, obs, refused, call_name)
        return list(dict.fromkeys(lines))

    @classmethod
    def notice_lines(cls, ctx, obs, refused=False, tool=""):
        lines = []
        if not isinstance(obs, dict):
            return lines
        memo = vars(ctx.talk).get("notice_memo") if ctx is not None else None
        if memo and memo[0] is obs and memo[1] == refused:
            return list(memo[2])

        def names_listing(d):
            return any(isinstance(v, str) and "id" in str(k).lower() and re.fullmatch(ID_LIKE, v) for k, v in d.items())

        def priced(d):
            return any(TextKit.is_number(v) and re.search(r"price|cost", str(k).lower()) for k, v in d.items())

        # a sentence under a key this tool's results never carried before (a notice beside results, a cart total or an
        # item id) is an event, heard once; the keys of a tool's first result are its usual shape
        talk = ctx.talk if ctx is not None else None
        shapes = vars(talk).setdefault("usual_keys", {}) if talk is not None else {}
        usual = shapes.get(tool)
        seen = vars(talk).setdefault("notices_seen", set()) if talk is not None else set()
        fresh_keys = set()

        def sentence(piece):
            return len(piece) <= 600 and (re.search(r"[.!?](?:\s|$)", piece) is not None or len(piece.split()) >= 8)

        def visit(d):
            if d is not obs and len(d) > 6:
                return
            for key, piece in d.items():
                if isinstance(piece, list) and piece and all(isinstance(val, str) for val in piece):
                    TextKit.long_strings(piece, lines)
                    continue
                if not isinstance(piece, str) or len(piece.split()) < 4:
                    continue
                name = str(key).lower()
                fresh_keys.add(name)
                if refused or (len(d) <= 3 and len(piece) <= 600 and not names_listing(d)):
                    lines.append(piece)
                elif usual is not None and name not in usual and not priced(d) and not (names_listing(d) and len(d) >= 5) and sentence(piece):
                    flat = " ".join(piece.lower().split())
                    if flat not in seen:
                        seen.add(flat)
                        lines.append(piece)
        TextKit.walk_dicts(obs, visit, ceiling=4)
        if talk is not None:
            shapes.setdefault(tool, set()).update(fresh_keys)
            talk.notice_memo = (obs, refused, list(dict.fromkeys(lines)))
        return list(dict.fromkeys(lines))


class Checkout:
    @classmethod
    def screen(cls, ctx, proposal: dict[str, Any], permitted: set[str], scripted: bool, batch: int,
               queued_messages: int) -> tuple[str | None, str, dict[str, Any]]:
        func = proposal.get("function") or {}
        call_name, raw_args = func.get("name"), func.get("arguments")
        shapes = ctx.shelf.shapes
        if call_name not in shapes:
            listed = ", ".join(sorted(permitted))
            return f"Not run (no step used): '{call_name}' is not one of the tools. Use one of: {listed}.", call_name, {}
        if call_name not in permitted and not scripted:
            return ("Not run (no step used): too few steps are left for this; only these tools can be used now: "
                    + ", ".join(sorted(permitted)) + "."), call_name, {}
        if isinstance(raw_args, dict):
            call_args = raw_args
        elif (raw_args or "").strip():
            call_args = TextKit.json_object(raw_args)
        else:
            call_args = {}
        if call_args is None:
            return "Not run (no step used): the arguments were not a valid JSON object.", call_name, {}
        call_args, faults = ArgCheck.execute(call_args, shapes[call_name])
        if faults:
            return "Not run (no step used): invalid arguments: " + "; ".join(faults[:6]), call_name, call_args
        refusal = cls._gate_kind(ctx, call_name, scripted, batch, queued_messages)
        if refusal:
            return refusal, call_name, call_args
        sig = call_name + TextKit.as_text(call_args)
        repeat_free = call_name in ctx.finishers() or call_name in (ctx.role("message"), ctx.role("cart"))
        if not repeat_free and sig in ctx.talk.last_calls:
            return ("Not run (no step used): you just made this exact call and nothing has changed since; its result is above. "
                    "Change the arguments or do something else."), call_name, call_args
        if not scripted:
            held = Intake.late_hold(ctx, call_name, call_args) or Coverage.value_guard(ctx, call_name, call_args)
            if held:
                return held, call_name, call_args
        if call_name == ctx.role("cart"):
            refusal = cls.screen_cart(ctx, call_args, sig)
            if refusal:
                return refusal, call_name, call_args
        if call_name == ctx.role("message") and not scripted:
            held = cls._last_word_sweep(ctx, queued_messages)
            if held:
                return held, call_name, call_args
        return None, call_name, call_args

    @classmethod
    def _last_word_sweep(cls, ctx, queued_messages: int) -> str:
        # a message spending the last allowance seals the told basket, so the cheapest sweep runs before it goes out
        if not ctx.cart.trolley.units or ctx.ask_limit() - ctx.budget.msgs_sent - queued_messages > 1:
            return ""
        if not any(isinstance(t, dict) and t.get("before_finish") and t.get("needs_message") for t in ctx.needs.told):
            return ""
        return Checkout._sweep_gate(ctx)

    @classmethod
    def _gate_kind(cls, ctx, call_name: str, scripted: bool, batch: int, queued_messages: int) -> str:
        budget, units = ctx.budget, ctx.cart.trolley.units
        if batch > 1 and call_name in ctx.finishers():
            return ("Not run (no step used): an order or a decline must be sent alone, after you have seen the results of "
                    "your other calls.")
        is_message = call_name == ctx.role("message")
        planned = budget.msgs_sent + queued_messages
        if is_message and budget.msg_cap is not None and planned >= budget.msg_cap:
            return "Not run (no step used): the message budget is used up. Do not message the shopper again."
        if is_message and not scripted and not units and ctx.msg_reserve() and planned >= ctx.ask_room():
            return ("Not run (no step used): the messages left are kept for telling the shopper the basket once something is in "
                    "the cart (and for any finishing step the instructions state). Put your pick in the cart first; after that "
                    "the message tool is yours again.")
        if scripted or call_name != ctx.role("cart_view") or call_name == ctx.role("cart"):
            return ""
        state = vars(ctx.cart)
        if state.get("batch_cart") or not units:
            return ""
        if state.get("cart_shown", -1) != state.get("cart_ver", -2) or state.get("since_cart", 1):
            return ""
        shown = vars(ctx.talk).setdefault("view_told", set())
        if ctx.cart.cart_shown in shown:
            return ""
        shown.add(ctx.cart.cart_shown)
        return ("Not run (no step used): the last cart result already shows the basket - " + ctx.cart.trolley.summary()
                + " A cart view now shows nothing new; to re-check a line's stock, read its record with "
                + (Lookup.stock_tool(ctx) or "the tool that shows stock") + ".")

    @classmethod
    def screen_cart(cls, ctx, call_args, sig):
        book = ctx.cart.trolley
        thing = book.aim(call_args)
        sealed = cls._sealed_basket(ctx)
        if sealed:
            return sealed
        point = ""
        if thing and book.is_put(call_args):
            point = Referencing.ref_pointer_note(ctx, thing)
        told_points = vars(ctx.talk).setdefault("pointer_told", {})
        if point and told_points.get(thing, 0) < 2:
            told_points[thing] = 1 + told_points.get(thing, 0)
            return f"Not run (no step used): {point}. Choose another listing."
        note = cls._kept_line_note(ctx, call_args, sig, thing)
        if note or not (thing and book.is_put(call_args)):
            return note or ""
        already = book.held(thing)
        if already > 0 and book.adds is not False and sig not in ctx.talk.warned:
            ctx.talk.warned.add(sig)
            return (f"Not run (no step used): the cart already holds {thing} x{int(book.held(thing))}, and this cart tool adds units to a held line, "
                    f"so this call would raise it. To order it, leave the cart as it is; to change the units, take the difference out or in. If you really mean to add, send the same call again.")
        steps = [] if point else [lambda: Coverage.walk_hold(ctx, call_args)]
        steps += [lambda: cls._added_lines_ruling(ctx, call_args, thing), lambda: cls._rank_hold(ctx, call_args)]
        for step in steps:
            answer = step()
            if answer:
                return answer
        return cls._count_hold(ctx, call_args, sig, thing)

    @classmethod
    def _kept_line_note(cls, ctx, call_args: dict[str, Any], sig: str, thing: str) -> str:
        book, cart, records = ctx.cart.trolley, ctx.cart, ctx.shelf.archive
        state = vars(cart)
        mark = f"{sig}#{state.get('puts', {}).get(thing, 0)}"
        mine = records.price(thing) if thing else None
        cheaper_put = False
        for other in state.get("batch_puts", []):
            if other == thing:
                continue
            if TextKit.is_number(mine) and TextKit.is_number(records.price(other)) and records.price(other) < mine:
                cheaper_put = True
                break
        cheapest = ctx.needs.priority() == "cheapest"
        swap = cheapest and (cheaper_put or bool(Judging.cheaper_fit(ctx, thing)))
        moved = thing in state.get("pass_price", {}) and state["pass_price"][thing] != mine
        if not thing or book.is_put(call_args) or f"{thing}|{ctx.needs.version}" not in cart.passed or swap or moved:
            return ""
        if cart.refusal_n != cart.refusals_seen.get(thing, cart.refusal_n) or mark in ctx.talk.warned:
            return ""
        ctx.talk.warned.add(mark)
        least = ""
        if ctx.needs.priority() == "cheapest" and not Judging.cheaper_fit(ctx, thing):
            least = " It is also the cheapest verified fit."
        return (f"Not run (no step used): {thing} passed an independent check against every requirement in force today, with each "
                "property read by the value rules (a field in which this store writes the value itself is met only by those words); the item kind comes from the category label alone, whatever object the title or description "
                f"mentions.{least} Nothing has changed since. If a change or a refusal rules it out, one of its fields clearly fails "
                "a requirement, or you are replacing it with a cheaper listing, send the same call again.")

    @classmethod
    def _added_lines_ruling(cls, ctx, call_args: dict[str, Any], thing: str) -> str:
        book, cart = ctx.cart.trolley, ctx.cart
        candidates = [other for other, count in book.lines(call_args)[1:] if count is None or count > 0]
        candidates.append(thing)
        for item in candidates:
            try:
                hold = Judging.unproven_wording(ctx, item)
            except Exception as err:
                ctx.log.append({"kind": "wording_hold_error", "error": f"{type(err).__name__}: {err}"[:300]})
                hold = ""
            if hold:
                return "Not run (no step used). " + hold
        for item in candidates:
            key = f"{item}|{ctx.needs.version}"
            if ctx.moves_left() <= 3 or key in cart.checked_adds:
                continue
            cart.checked_adds.add(key)
            ruling = Judging.check_item(ctx, item, "ADD TO THE CART")
            if ruling:
                return "Not run (no step used). " + ruling + " If you are certain it fits, send the same call again."
        return ""

    @classmethod
    def _rank_hold(cls, ctx, call_args: dict[str, Any]) -> str:
        book = ctx.cart.trolley
        live = lambda count: count is None or count > 0
        if ctx.moves_left() > 3:
            for line_id, count in book.lines(call_args):
                if not live(count):
                    continue
                better = Judging.rank_check(ctx, line_id)
                if not better:
                    continue
                cheaper = Judging.rank_verify(ctx, line_id)
                text = better.replace("before ordering", "before putting it in the cart").replace(
                    "order the current pick", "send the same call again")
                if cheaper:
                    text += (f" The independent item check found that {cheaper} meets every requirement now in force and costs less than "
                             f"{line_id}: put {cheaper} in instead.")
                return "Not run (no step used). " + text
        if ctx.needs.priority() != "cheapest":
            return ""
        for line_id, count in book.lines(call_args):
            if not live(count):
                continue
            cheaper = Judging.cheaper_fit(ctx, line_id)
            told = vars(ctx.talk).setdefault("cheaper_told", {})
            if not cheaper or cheaper in book.units or told.get(("cart", line_id, cheaper)):
                continue
            told[("cart", line_id, cheaper)] = 1
            return (f"Not run (no step used). By the independent item check, {cheaper} meets every requirement in "
                    f"force and is priced below {line_id}. The request wants the cheapest fit, and the check's reading outranks "
                    f"yours: put {cheaper} in instead. If a record shows {cheaper} failing a requirement now in force, send the "
                    "same call again.")
        return ""

    @classmethod
    def _count_hold(cls, ctx, call_args, sig, thing):
        book, needs, warned = ctx.cart.trolley, ctx.needs, ctx.talk.warned
        need_n = needs.quantity()
        count = book.count_arg(call_args)
        aim = need_n - book.held(thing) if book.adds is not False else need_n
        if not needs.items():
            live = [item for item, units in book.lines(call_args) if units is None or units > 0]
            if len(live) > 1 and sig not in warned:
                warned.add(sig)
                return (f"Not run (no step used): the request wants ONE product, a single cart line, but this call puts {len(live)} "
                        f"lines ({', '.join(map(str, live))}). Keep only the chosen listing. If you are sure, send the same call again.")
            if count is None or sig in warned or abs(count - aim) <= 1e-9:
                return ""
            warned.add(sig)
            return f"Not run (no step used): the request needs a quantity of {need_n} of one listing, but this call asks for {int(count)}. If you are sure, send the same call again."
        odd = [(item, int(units)) for item, units in book.lines(call_args)
               if units is not None and units > 0 and int(units) not in needs.counts()]
        if sig in warned or not odd:
            return ""
        warned.add(sig)
        return (f"Not run (no step used): the products asked for need {needs.counts()} units (one line each), but this call "
                f"asks for {odd}. If you are sure, send the same call again.")

    @classmethod
    def screen_finish(cls, ctx, cid: str, call_name: str, call_args: dict[str, Any]) -> tuple[str | None, dict[str, Any]]:
        kind = ctx.call_kind(call_name, call_args)
        if kind == "order":
            # a basket already told with the last message is ordered as told: no gate may ask for a change
            gates = () if cls._sealed_basket(ctx) else (cls._ref_gate, cls._dead_gate, cls._cheaper_gate, cls._walk_gate, cls._stock_gate, cls._sweep_gate,
                         cls._dependency_gate, cls._vet_gate)
            for gate in gates:
                held = gate(ctx)
                if held:
                    return held, call_args
            waited = cls.finish_steps(ctx, "order")
            return (waited, call_args) if waited else (None, call_args)
        if kind == "decline":
            return cls._decline_gates(ctx, call_name, call_args)
        return None, call_args

    @classmethod
    def _sealed_basket(cls, ctx) -> str:
        # the basket was told to the shopper with the last message a stated step can use: a cart change now could not be
        # told again, so the cart stays as told
        try:
            msg_tool = ctx.role("message")
            if not msg_tool or not ctx.cart.trolley.units:
                return ""
            if not any(isinstance(t, dict) and t.get("before_finish") and t.get("needs_message") for t in ctx.needs.told):
                return ""
            if ctx.ask_limit() - ctx.budget.msgs_sent > 0:
                return ""
            edits = vars(ctx.cart).get("cart_edits", 0)
            if not cls._sent_since(ctx, edits):
                return ""
            return ("Not run (no step used): the basket and its total were already told to the shopper with the last message "
                    "left, and a change now could not be told again. Order the cart exactly as it was told.")
        except Exception:
            return ""

    @classmethod
    def _ref_gate(cls, ctx):
        # a requirement still given only by another listing's page holds the order while there is room to follow it
        try:
            opens = [str(ref.get("req") or ref.get("head") or "") for ref in Referencing._all_refs(ctx)
                     if ref.get("status") == "open" and Referencing.ref_live(ctx, ref)]
            # a requirement whose value is still a pointer phrase counts too, registered or not
            opens += [req for req in ctx.needs.props_left() if RefKit.ref_text(str(req).partition(":")[2])]
            opens = list(dict.fromkeys(o for o in opens if o))
            if not opens or ctx.moves_left() <= 8 or ctx.wire.spare() <= CLOSING + 20:
                return ""
            held = vars(ctx.talk).setdefault("ref_held", 0)
            if held >= 2:
                return ""
            ctx.talk.ref_held = held + 1
            heads = "; ".join(o[:160] for o in opens[:2])
            return ("Not run yet (no step used). These requirements still take their value from another listing's page, and "
                    "that page has not been settled: " + heads + ". Find that listing (by the seller and the details the "
                    "shopper gives), open its page, take the value it shows, then choose the item and order.")
        except Exception as err:
            ctx.log.append({"kind": "ref_gate_error", "error": f"{type(err).__name__}: {err}"[:300]})
            return ""

    @classmethod
    def _dead_gate(cls, ctx):
        if ctx.moves_left() <= 2:
            return ""
        book = ctx.cart.trolley
        dead = sorted(set(book.units) & Lookup.dead_lines(ctx))
        mark = (tuple(dead), vars(ctx.cart).get("cart_seq", 0))
        told = vars(ctx.talk).setdefault("dead_told", {})
        if not dead:
            return ""
        times = told.get(mark, 0)
        if times < 2:
            told[mark] = times + 1
            return "Not run yet (no step used). " + Lookup.dead_note(ctx, dead)
        if times != 2 or not ctx.role("cart") or ctx.moves_left() < 6 or ctx.wire.spare() <= CLOSING:
            return ""
        told[mark] = 3
        swapped = cls._swap_dead(ctx, dead)
        ctx.log.append({"kind": "dead_swap", "lines": dead, "done": swapped, "cart": dict(book.units)})
        if swapped and book.units and not set(book.units) & Lookup.dead_lines(ctx):
            return ("Not run yet (no step used). The cart still held " + ", ".join(dead) + ", which is sold out, so the code "
                    "took it out and put a candidate in instead, keeping the other lines, and read the lines' stock: "
                    + book.summary() + " Check this cart against the request, then order it.")
        return ""

    @staticmethod
    def _swap_dead(ctx, dead):
        muted_before = ctx.wire.mute
        held = set(ctx.cart.trolley.units)
        try:
            return Closing.code_cart(ctx, set(dead) | Lookup.dead_lines(ctx), held - set(dead))
        except Exception as err:
            ctx.log.append({"kind": "dead_swap_error", "error": f"{type(err).__name__}: {err}"[:300]})
            return False
        finally:
            ctx.wire.mute = muted_before

    @classmethod
    def _cheaper_gate(cls, ctx) -> str:
        if ctx.moves_left() <= 3:
            return ""
        thing = Judging.chosen(ctx)
        cheaper = Judging.cheaper_fit(ctx, thing) if thing else ""
        if not cheaper:
            for line in [i for i in ctx.cart.trolley.units if i != thing]:
                cheaper = Judging.cheaper_fit(ctx, line)
                if cheaper:
                    thing = line
                    break
        told = vars(ctx.talk).setdefault("cheaper_told", {})
        if not cheaper or told.get((thing, cheaper), 0) >= 2:
            return ""
        told[(thing, cheaper)] = told.get((thing, cheaper), 0) + 1
        return (f"Not run yet (no step used). The independent item check found that {cheaper} meets every requirement, and "
                f"it costs less than {thing}. The cheapest fit is asked for and the check's reading stands over yours: take "
                f"{thing} out of the cart, put {cheaper} in, re-read its stock and price, and order it.")

    @classmethod
    def _walk_gate(cls, ctx) -> str:
        return Coverage.walk_hold(ctx, None) if ctx.cart.trolley.units else ""

    @classmethod
    def _stock_gate(cls, ctx) -> str:
        reader = Lookup.stock_tool(ctx)
        units = ctx.cart.trolley.units
        # the re-reads keep room for the basket message a stated step may still need, and the order itself
        if not reader or ctx.moves_left() <= len(units) + 3 or ctx.wire.spare() <= CLOSING:
            return ""
        stale = Lookup.stale_lines(ctx)
        mark = (tuple(sorted(units)), vars(ctx.cart).get("cart_seq", 0))
        if not stale or vars(ctx.cart).get("recheck_mark") == mark:
            return ""
        ctx.cart.recheck_mark = mark
        shelf = ctx.shelf
        shelf.open_queue = stale[:4]
        shelf.open_recheck = True
        shelf.open_purpose = "cart"
        shelf.open_tool = reader
        return ("Not run yet (no step used). Stock and prices can change without any notice, and these cart lines were not "
                "read since the last cart change: " + ", ".join(stale[:4]) + f". Their stock is read for you now ({reader}, the "
                "tool that shows stock); then order if each is still in stock and the total is within the budget, else replace "
                "the line.")

    @classmethod
    def _sweep_gate(cls, ctx) -> str:
        try:
            return cls.cheap_sweep(ctx)
        except Exception as err:
            ctx.log.append({"kind": "cheap_sweep_error", "error": f"{type(err).__name__}: {err}"[:300]})
            return ""

    @classmethod
    def _dependency_gate(cls, ctx) -> str:
        if ctx.moves_left() <= 5 or ctx.wire.spare() <= CLOSING:
            return ""
        try:
            deps = cls.dependency_faults(ctx)
        except Exception:
            deps = []
        mark = (tuple(sorted(ctx.cart.trolley.units)), tuple(deps))
        told = vars(ctx.talk).setdefault("dep_told", {})
        if not deps or told.get(mark, 0) >= 2 or len(told) >= 4:
            return ""
        told[mark] = told.get(mark, 0) + 1
        return ("Not run yet (no step used). The request says one product must work with another: " + "; ".join(deps)
                + ". Search the dependent product with each identifier as the phrase, open the results, and put in one whose "
                "page names one of them (or take another anchor whose identifiers a dependent page names); then order.")

    @classmethod
    def _vet_gate(cls, ctx) -> str:
        cart, needs = ctx.cart, ctx.needs
        if cart.order_vetted or ctx.moves_left() <= 2 or ctx.wire.spare() <= CLOSING:
            return ""
        cart.order_vetted = True
        faults = cart.trolley.faults(needs.quantity(), needs.counts())
        if faults and ctx.moves_left() > 3:
            return "Not run yet (no step used). The cart as counted is not ready: " + "; ".join(faults) + ". Fix it, then order."
        if needs.items():
            lines = list(cart.trolley.units)
        else:
            pick = Judging.chosen(ctx)
            lines = [pick] if pick else []
        for thing in lines:
            ruling = Judging.check_item(ctx, thing, "ORDER")
            if ruling:
                return "Not run yet (no step used). " + ruling
        for thing in lines:
            better = Judging.rank_check(ctx, thing)
            if not better:
                continue
            cheaper = Judging.rank_verify(ctx, thing)
            extra = (f" The independent item check found that {cheaper} meets every requirement now in force and costs less "
                     f"than {thing}: take {thing} out, put {cheaper} in, re-read its stock and price, and order.") if cheaper else ""
            return "Not run yet (no step used). " + better + extra
        shape = "per product asked for, each at its quantity" if needs.items() else "the chosen item at the required quantity"
        return ("Not run yet (no step used). Before the order, confirm: (1) the cart has exactly one line " + shape + " and no other line; "
                "(2) since your last cart change you re-read the item's stock and price with a tool that shows them, it is in stock, and price times quantity fits the current budget; "
                "(3) the item's own properties meet each stated requirement, each need the shopper revealed and each announced change; "
                "(4) when the cheapest or best was asked for, nothing you could find fits better; "
                "(5) every procedural rule in the request has been followed. If anything is unchecked, check it now; otherwise call the order tool again.")

    @classmethod
    def _decline_gates(cls, ctx, call_name: str, call_args: dict[str, Any]) -> tuple[str | None, dict[str, Any]]:
        note = Lookup.near_hold(ctx)
        if note:
            return note, call_args
        for gate in (cls._kept_fit_gate, cls._slack_gate, cls._fill_gate, cls._named_gate):
            note = gate(ctx)
            if note:
                return note, call_args
        pages_note = Walking.decline_reads(ctx)
        if pages_note:
            return "Not run yet (no step used). " + pages_note, call_args
        note = cls._audit_gate(ctx)
        if note:
            return note, call_args
        fit_note = Walking.fit_hold(ctx)
        if fit_note:
            return "Not run yet (no step used). " + fit_note, call_args
        for gate in (cls._best_gate, cls._differ_gate):
            note = gate(ctx)
            if note:
                return note, call_args
        for last_check in (Lookup.decline_repair, Lookup.near_check):
            note = last_check(ctx)
            if note:
                return note, call_args
        waited = cls.finish_steps(ctx, "decline")
        if waited:
            return waited, call_args
        return None, Closing.scrub_decline(ctx, call_args, call_name)

    @classmethod
    def _kept_fit_gate(cls, ctx) -> str:
        cart, needs = ctx.cart, ctx.needs
        passed_lines = []
        for item in cart.trolley.units:
            if f"{item}|{needs.version}" in cart.passed and item not in cart.refused_lines and item not in Lookup.dead_lines(ctx):
                passed_lines.append(item)
        if not passed_lines or ctx.moves_left() <= 1 or vars(cart).get("decline_kept"):
            return ""
        cart.decline_kept = True
        every = " with every product's line in the cart" if needs.items() else ""
        return ("Not run yet (no step used). In the cart is " + ", ".join(passed_lines) + ", and the independent check "
                "found it meets every requirement in force. Declining would discard that: re-read its stock and price and "
                "order it" + every + ". Decline only if a "
                "change since then clearly rules it out.")

    @classmethod
    def _slack_gate(cls, ctx) -> str:
        budget = vars(ctx.budget)
        slack = []
        if not budget.get("slack_done") and (ctx.role("details") or Lookup.batch_tool(ctx)):
            slack = Lookup.slack_rows(ctx)
        spread = Lookup.batch_tool(ctx)
        reads = -(-len(slack) // spread[2]) if spread and len(slack) >= 2 else len(slack)
        if not slack or ctx.moves_left() <= reads + 4 or ctx.wire.spare() <= CLOSING + 40:
            return ""
        ctx.budget.slack_done = True
        ctx.shelf.open_queue = slack
        ctx.shelf.open_say = "Ahead of any decline, I read cheaper rows that could let all products fit the budget at once."
        total = budget.get("slack_total")
        told_total = f"{total:g}" if TextKit.is_number(total) else "more than the budget"
        return ("Not run yet (no step used). Each product has a fitting listing, yet the cheapest fits added up cost "
                + told_total + f", over the budget of {ctx.needs.spend_cap}. "
                "Rows priced inside what the budget leaves each product, after the cheapest fits of the others, were never "
                "opened, so the code reads their records for you now: " + ", ".join(slack) + ". Judge each against its own "
                "product's requirements by the value rules (a field in which this store writes the value itself is met only by "
                "those words); when fits within the budget together cover every product, put every product's line "
                "in the cart and order; decline only if no such set exists.")

    @classmethod
    def _fill_gate(cls, ctx) -> str:
        batch = Lookup.batch_tool(ctx)
        able = ctx.role("details") or batch
        fill = Lookup.lacking_rows(ctx) if not vars(ctx.cart).get("decline_fill") and able else []
        reads = -(-len(fill) // batch[2]) if batch and len(fill) >= 2 else len(fill)
        if not fill or ctx.moves_left() <= reads + 3 or ctx.wire.spare() <= CLOSING + 40:
            return ""
        ctx.cart.decline_fill = True
        ctx.shelf.open_queue = fill
        ctx.shelf.open_say = "Ahead of any decline, I read unopened listings for the products still lacking a verified fit."
        return ("Not run yet (no step used). Some requested products still lack a verified fit, and in-budget listings of "
                "their categories were never opened, so the code reads their records for you now, ahead of any "
                "decline: " + ", ".join(fill) + ". Judge each against its own product's requirements "
                "by the value rules (a field in which this store writes the value itself is met only by those words); when "
                "every product has a fitting listing, put every product's line in the cart and order; decline only if "
                "a product truly has none.")

    @classmethod
    def _named_gate(cls, ctx) -> str:
        batch = Lookup.batch_tool(ctx)
        queue = Judging.named_unread(ctx)[: batch[2] if batch else 3]
        reads = -(-len(queue) // batch[2]) if batch and len(queue) >= 2 else len(queue)
        if not queue or vars(ctx.cart).get("decline_opens") or not (ctx.role("details") or batch):
            return ""
        if ctx.moves_left() <= reads + 3 or ctx.wire.spare() <= CLOSING + 40:
            return ""
        ctx.cart.decline_opens = True
        ctx.shelf.open_queue = queue
        return ("Not run yet (no step used). The checks named listings that nobody opened, so the code reads their records "
                "for you now, ahead of any decline: " + ", ".join(queue) + ". Judge each against the requirements by the value rules (the "
                "field named for the property decides, and where this store writes the value itself there only those words count; "
                "when there is no such field, an alternative field, the options list or the title may show the "
                "value), buy one that fits, and decline only if each one fails.")

    @classmethod
    def _audit_gate(cls, ctx) -> str:
        cart = ctx.cart
        state = vars(cart)
        reread = len(ctx.shelf.archive.read) > state.get("audit_reads", 10 ** 9) and state.get("audits", 0) < 2
        if reread and cart.decline_vetted and ctx.moves_left() > 3 and ctx.wire.spare() > CLOSING + 30:
            audit_note = Judging.review_decline(ctx)
            if audit_note:
                return "Not run yet (no step used). " + audit_note
        if cart.decline_vetted or ctx.moves_left() <= 3 or ctx.wire.spare() <= CLOSING + 30:
            return ""
        cart.decline_vetted = True
        audit_note = Judging.review_decline(ctx)
        if audit_note:
            return "Not run yet (no step used). " + audit_note
        return ("Not run yet (no step used). Make sure before declining: you searched more than one way (other words, filters, pages), you looked at each candidate's own properties, "
                "and every one fails a stated requirement, a revealed need, the budget or the stock. Any key spelling of a property's field counts. "
                "If something does fit, buy it instead; if nothing fits, call the decline tool again.")

    @classmethod
    def _best_gate(cls, ctx) -> str:
        cart, needs = ctx.cart, ctx.needs
        best = Walking.unrefuted(ctx) if not vars(cart).get("best_held") and ctx.moves_left() > 4 else []
        tries = 0
        while tries < 3:
            tries += 1
            unchecked = [i for i in best if f"{i}|{needs.version}" not in cart.passed]
            if not unchecked or ctx.wire.spare() < CLOSING + 30:
                break
            Judging.check_item(ctx, unchecked[0], "ORDER")
            best = Walking.unrefuted(ctx)
        if not best or ctx.moves_left() <= len(best) + 3 or ctx.wire.spare() <= CLOSING + 20:
            return ""
        cart.best_held = True
        several = bool(needs.items())
        holds = "a candidate for every product asked for" if several else "a candidate"
        lines = "one line of each product, at its quantity," if several else "it"
        return ("Not run yet (no step used). Declining fits only when no listing can meet the requirements, and the "
                "records show " + holds + " that is in "
                "stock, inside the budget and not ruled out by any check: " + ", ".join(best) + ". Unless a record shows one of "
                "them failing a requirement now in force, put " + lines + " in the cart, re-read stock and price, and order; decline only if each candidate clearly "
                "fails.")

    @classmethod
    def _differ_gate(cls, ctx) -> str:
        soft = []
        for ref in vars(ctx.pointers).get("refs") or []:
            if ref.get("relation") == "differ" and ref.get("status") in ("open", "stuck") and Referencing.ref_live(ctx, ref):
                soft.append(ref)
        if not soft or ctx.moves_left() <= 3 or vars(ctx.cart).get("ref_decline_held"):
            return ""
        ctx.cart.ref_decline_held = True
        heads = ", ".join(dict.fromkeys(ref["head"] for ref in soft))
        per_product = ", one line per product" if ctx.needs.items() else ""
        return ("Not run yet (no step used). " + heads + " comes from another listing the code could not resolve, so its "
                "value is unknown: it never excludes a listing and never justifies a decline. Put the listing that meets each "
                "other requirement (in stock, within the budget) in the cart" + per_product
                + ", re-read its stock and price, and order; decline only if no listing meets the other requirements.")

    @classmethod
    def change_review(cls, ctx) -> str:
        records, wire = ctx.shelf.archive, ctx.wire
        done = vars(ctx.talk).get("reviews", 0)
        if done >= 3 or not records.read or wire.mute or wire.spare() < CLOSING + 20:
            return ""
        if not wire.usable(CHECKER) or ctx.moves_left() < 3:
            return ""
        ctx.talk.reviews = done + 1
        cards = records.cards(14000, "", Referencing.ref_foreign(ctx))
        if not cards:
            return ""
        needs = ctx.needs
        sections = [
            CHANGE_REVIEW,
            "\n\nTHE SHOPPER'S REQUEST:\n" + ctx.request + needs.changes(),
            "\n\nWHAT JUST CHANGED: " + " / ".join(needs.exchanges[-2:]),
            "\n\nREQUIREMENTS NOW: " + needs.text_body(),
            f"\n\nPRIORITY: {needs.priority()} {needs.reading.get('priority_detail') or ''}".rstrip(),
            "\n\nTHE CART: " + ctx.cart.trolley.summary(),
            "\n\nLISTINGS SEEN (DETAILS = its full record was read, ROW = seen only in search rows):\n" + "\n".join(cards),
            Judging.field_names(ctx),
        ]
        answer = wire.consult("".join(sections), 2500, CHECKER)
        ctx.log.append(dict(kind="change_review", answer=answer))
        if not answer:
            return ""
        return cls._review_remark(ctx, answer)

    @classmethod
    def _review_remark(cls, ctx, answer: dict[str, Any]) -> str:
        ids = ctx.shelf.archive.ids
        fits = []
        for raw in answer.get("fits") or []:
            if isinstance(raw, (str, int)) and str(raw) in ids:
                fits.append(str(raw))
        fits = fits[:6]
        pick = answer.get("pick")
        pick = str(pick) if isinstance(pick, (str, int)) and str(pick) in ids else ""
        vars(ctx.shelf).setdefault("named", set()).update(fits)
        if fits:
            verdict = "opened listings that meet every requirement now in force, best first: " + ", ".join(fits)
        else:
            verdict = "none of the opened listings is shown to meet every requirement now in force"
        parts = ["[re-check after the change]", verdict]
        if pick:
            parts.append(f"the one to take now: {pick}")
        still = answer.get("cart_still_right")
        if ctx.cart.trolley.units and still is True:
            parts.append("the cart's line still fits: keep it, so the basket already told stays the one ordered")
        elif ctx.cart.trolley.units and still is False:
            parts.append("the cart's line is no longer the right pick under the conditions now: replace it (when the instructions "
                         "ask you to check with the shopper before replacing anything, ask first)")
        if answer.get("reason"):
            parts.append(str(answer["reason"])[:300])
        parts.append("Act on this before anything else: re-pick among what fits now (the cheapest or best of them when the request "
                     "asks for that), fix the cart, re-read its stock and price, then finish; never decline while an opened listing fits.")
        return " ".join(parts)

    @classmethod
    def code_asks_swap(cls, ctx, old: list[str], new: list[str]) -> bool:
        tool = ctx.role("message")
        target = Asking.text_arg(ctx, tool) if tool else None
        if not target:
            return False
        if ctx.ask_limit() - ctx.budget.msgs_sent < 1 + ctx.msg_reserve() or ctx.moves_left() < 4:
            return False
        question = ("Before I change the basket: I would like to take out " + ", ".join(old[:3]) + " and put in " + ", ".join(new[:3])
                    + " instead, since the first no longer fits. Is that all right with you?")
        answers = cls.send_chunk(ctx, [(f"code_swap_ask_{ctx.wire.tick}", tool, {target: question})])
        ctx.log.append({"kind": "code_swap_ask", "old": old[:3], "new": new[:3]})
        if not answers:
            return False
        return not str(answers[0][1]).startswith("Not run")

    @classmethod
    def _records_naming(cls, ctx, thing: str, visit) -> None:
        def own_record(d: dict[str, Any]) -> None:
            id_values = [v for k, v in d.items() if isinstance(v, str) and "id" in str(k).lower()]
            if thing in id_values:
                visit(d)
        for rec in ctx.shelf.archive.items:
            TextKit.walk_dicts(rec, own_record)

    @classmethod
    def line_seller(cls, ctx, thing: str) -> str:
        learned = Referencing.seller_keys(ctx)
        found: list[str] = []

        def sellers(d: dict[str, Any]) -> None:
            for k, v in d.items():
                if not isinstance(v, str) or not v.strip() or len(v) > 60 or "id" in str(k).lower():
                    continue
                if re.search(SELLER_KEYS, str(k).lower()) or str(k) in learned:
                    found.append(v.strip())
        cls._records_naming(ctx, thing, sellers)
        return found[-1] if found else ""

    @classmethod
    def page_text(cls, ctx, thing: str) -> str:
        pieces: list[str] = []
        cls._records_naming(ctx, thing, lambda d: TextKit.long_strings_all(d, pieces))
        return " ".join(dict.fromkeys(pieces))

    @classmethod
    def ident_tokens(cls, ctx, text: str) -> list[str]:
        found = re.findall(r"\b([A-Z][A-Za-z]{1,12}[ -]?\d{3,6})\b", text)
        return list(dict.fromkeys(found))[:8]

    @classmethod
    def link_tokens(cls, ctx, anchor: str) -> list[str]:
        # a fit key is a code on the anchor's page (letters then digits: a part series, a socket, a model line) that other
        # listings read also carry; a code no other listing names (the anchor's own model number) says nothing about fit
        text = cls.page_text(ctx, anchor)
        codes = list(dict.fromkeys(re.findall(r"\b([A-Z]{1,4}-?\d{1,4}[A-Za-z]?)\b", text)))
        if not codes:
            return cls.ident_tokens(ctx, text)
        others = [cls.squash(ctx, cls.page_text(ctx, ident)) for ident in list(ctx.shelf.archive.read)[:40] if ident != anchor]
        shared = [c for c in codes if any(cls.squash(ctx, c) in o for o in others)]
        return shared[:8] or cls.ident_tokens(ctx, text)

    @classmethod
    def squash(cls, ctx, text: str) -> str:
        return re.sub(r"[^a-z0-9]", "", str(text).lower())

    @classmethod
    def _dependent_rows(cls, ctx) -> list[dict[str, Any]]:
        rows = ctx.needs.items()
        if rows and any(row.get("depends_on") is not None for row in rows):
            return rows
        return []

    @classmethod
    def dependency_faults(cls, ctx) -> list[str]:
        rows = cls._dependent_rows(ctx)
        if not rows:
            return []
        held = {}
        for line in ctx.cart.trolley.units:
            slot = Walking.product_index(ctx, line)
            if slot is not None and slot not in held:
                held[slot] = line
        faults = []
        for k, row in enumerate(rows):
            j = row.get("depends_on")
            if j is None or j == k or j >= len(rows) or k not in held or j not in held:
                continue
            anchor, dep = held[j], held[k]
            tokens = cls.link_tokens(ctx, anchor)
            if not tokens:
                continue
            text = cls.squash(ctx, cls.page_text(ctx, dep))
            if any(cls.squash(ctx, t) in text for t in tokens):
                continue
            faults.append(f"the {row['product']} ({dep}) must work with the {rows[j]['product']} ({anchor}), but its page names "
                          f"none of that listing's identifiers ({', '.join(tokens)})")
        return faults

    @classmethod
    def dependency_hint(cls, ctx) -> str:
        rows = cls._dependent_rows(ctx)
        if not rows:
            return ""
        out = []
        for k, row in enumerate(rows):
            j = row.get("depends_on")
            if j is None or j == k or j >= len(rows):
                continue
            anchor = ""
            for line in ctx.cart.trolley.units:
                if Walking.product_index(ctx, line) == j:
                    anchor = line
                    break
            tokens = cls.link_tokens(ctx, anchor) if anchor else []
            mine, other = row["product"], rows[j]["product"]
            if tokens:
                out.append(f"the {mine} must work with the {other} in the cart ({anchor}): search the "
                           f"{mine} with each of its identifiers as the phrase ({', '.join(tokens)}), open the results, and "
                           "take one whose page names one of them")
            else:
                out.append(f"the {mine} must work with the {other}: choose that one first, then search the "
                           f"{mine} by its model names or numbers and take a page that names one of them")
        if not out:
            return ""
        return "\n[works with] " + "; ".join(out)

    @classmethod
    def cheap_sweep(cls, ctx):
        needs, cart = ctx.needs, ctx.cart
        book = cart.trolley
        if needs.priority() != "cheapest" or cart.orders_tried or vars(cart).get("cheap_swept"):
            return ""
        search = ctx.role("search")
        if not search or not book.units or ctx.moves_left() < 12 or ctx.wire.spare() < CLOSING + 40:
            return ""
        cart.cheap_swept = True
        paging = Reading.page_arg(ctx)
        arg = cls._lone_text_arg(ctx.shelf.shapes.get(search) or {}, paging[0] if paging else "")
        if arg is None:
            return ""
        ran = {" ".join(Reading.free_text(ctx, page["args"])).strip().lower()
               for page in vars(ctx.shelf).get("page_book", {}).values()}
        several = len(needs.items())
        queries = [q for q in Reading.label_queries(ctx) if " " not in q and q not in ran and len(re.sub(r"[^a-z]", "", q.lower())) >= 3][:min(ctx.route()["sweep_queries"], several) if several else ctx.route()["sweep_queries"]]
        pages = cls._unread_pages(ctx, book)
        if not queries and not pages:
            return ""
        tick = ctx.wire.tick
        cls.send_chunk(ctx, [(f"cheap_{n}_{tick}", search, {arg: q}) for n, q in enumerate(queries)]
                       + [(f"cheap_page_{n}_{tick}", search, args) for n, args in enumerate(pages)])
        if several:
            # a basket: every cart line is compared with the unopened rows of its own shelf
            rows = []
            for line in book.units:
                price = ctx.shelf.archive.price(line)
                shelf = cls._line_shelf(ctx, line)
                if price is not None and shelf:
                    rows.extend(cls._cheaper_unopened(ctx, book, price, shelf)[:3])
            rows = list(dict.fromkeys(rows))[:6]
        else:
            price = ctx.shelf.archive.price(next(iter(book.units)))
            if price is None:
                return ""
            rows = cls._cheaper_unopened(ctx, book, price)
        ctx.log.append({"kind": "cheap_sweep", "queries": queries, "rows": [i for _, i, _ in rows]})
        if not rows:
            return ""
        vars(ctx.shelf).setdefault("named", set()).update(i for _, i, _ in rows)
        listed = "; ".join(f"{i} '{t[:70]}' {c:g}" for c, i, t in rows)
        return ("Not run yet (no step used). The cheapest fit is asked for, so the code searched " + ", ".join(queries)
                + ": these in-stock rows of the requested category cost less than the cart's line" + (" of the same shelf" if several else "")
                + " and were never opened: " + listed + ". Open them (several in one reply), and if one meets "
                "every requirement of its product, make it the cart's line, re-read its stock and price, and order; otherwise "
                "order the cart as it is.")

    @classmethod
    def _unread_pages(cls, ctx, book) -> list[dict[str, Any]]:
        # a search whose next page was never read may hide a cheaper row of a cart line's shelf: one page per shelf
        found = Reading.page_arg(ctx)
        if not found:
            return []
        shelves = {cls._line_shelf(ctx, line) for line in book.units} - {""}
        pages, used = [], set()
        for walk in vars(ctx.shelf).get("page_book", {}).values():
            hit = walk["shelves"] & shelves - used
            if not hit:
                continue
            following = Reading.next_page(ctx, walk)
            if following is None:
                continue
            used |= hit
            pages.append({**walk["args"], found[0]: following})
        return pages[:3]

    @staticmethod
    def _lone_text_arg(schema, page_key=""):
        # the page cursor of a search is not its query text
        props = schema.get("properties") or {}
        texts = [k for k, v in props.items() if isinstance(v, dict) and v.get("type") == "string" and not v.get("enum") and k != page_key]
        if len(texts) == 1 and not set(schema.get("required") or []) - {texts[0]}:
            return texts[0]
        return None

    @classmethod
    def _cheaper_unopened(cls, ctx, book, price, shelf=""):
        read = ctx.shelf.archive.read
        shelves = cls._shelf_map(ctx) if shelf else {}
        found = [(cost, ident, title) for ident, (cost, title, _) in Reading.shelf_rows(ctx).items()
                 if cost < price and ident not in book.units and ident not in read
                 and (not shelf or shelves.get(ident) == shelf)]
        return sorted(found)[:4]

    @classmethod
    def _shelf_map(cls, ctx) -> dict[str, str]:
        found: dict[str, str] = {}

        def note(d: dict[str, Any]) -> None:
            label = Reading.shelf_of(ctx, d)
            if not label:
                return
            for k, v in d.items():
                if isinstance(v, str) and "id" in str(k).lower() and v not in found:
                    found[v] = label
        for rec in ctx.shelf.archive.items:
            TextKit.walk_dicts(rec, note)
        return found

    @classmethod
    def _line_shelf(cls, ctx, ident: str) -> str:
        return cls._shelf_map(ctx).get(ident, "")

    @classmethod
    def bound_hint(cls, ctx) -> str:
        if ctx.needs.priority() != "cheapest" or ctx.wire.spare() < CLOSING:
            return ""
        figs = []
        for req in ctx.needs.props_left():
            tail = str(req).split(":", 1)[-1].lower()
            figs.extend(m.group(1) + m.group(2) for m in re.finditer(r"(\d+(?:\.\d+)?)\s?([a-z\"']{1,4})\b", tail))
        figs = list(dict.fromkeys(figs))[:3]
        if not figs:
            return ""
        return ("\n[cheapest with a measured limit] the cheapest fit often writes the limit's own figure: search " + ", ".join(figs)
                + " with the kind word (with and without a space before the unit), read two pages of each, and open the cheapest "
                  "in-budget rows before fixing the pick.")

    @classmethod
    def basket_text(cls, ctx) -> str:
        try:
            info = Closing.code_rows(ctx)
        except Exception:
            info = {}
        parts, total, priced = [], 0.0, True
        for thing, qty_n in ctx.cart.trolley.units.items():
            row = info.get(thing) or {}
            cost = ctx.shelf.archive.price(thing)
            if cost is None:
                cost = row.get("price")
            units = int(qty_n) if float(qty_n).is_integer() else qty_n
            label = f" ({row['title'][:60]})" if row.get("title") else ""
            try:
                seller = cls.line_seller(ctx, thing)
            except Exception:
                seller = ""
            if seller:
                label += f" from {seller}"
            if cost is None:
                priced = False
                parts.append(f"{units} x {thing}{label}")
                continue
            total += float(cost) * float(qty_n)
            parts.append(f"{units} x {thing}{label} at {cost:g} each")
        unit = str(ctx.needs.reading.get("currency") or "").strip()
        if not parts:
            return "Before I finish: the basket is empty, so nothing will be ordered."
        said = "Before I finish, here is the basket: " + "; ".join(parts) + "."
        if priced:
            said += f" The total is {total:g}{(' ' + unit) if unit else ''}."
        # when a stated step wants the shopper's word before the finish, or they turned the basket down, the message asks
        asks_word = ctx.needs.approval == "no" or any(isinstance(t, dict) and t.get("before_finish") and t.get("needs_message")
                                                      for t in ctx.needs.told)
        return said + (" Shall I place this order?" if asks_word else " I will place the order now.")

    @classmethod
    def _finish_prompt(cls, ctx, kind: str, since: list[str], msgs_left: int, usable: list[dict[str, Any]]) -> str:
        needs = ctx.needs
        told_rows = "\n".join(f"- {t['text']} (from the {t.get('source')})" for t in needs.told) or "(none stated)"
        tool_rows = "\n".join(
            f"- {tdef['function']['name']}: {tdef['function'].get('description', '')} | arguments: {TextKit.as_text(tdef['function'].get('parameters') or {})}"
            for tdef in usable)
        return "".join([
            FINISH_CHECK,
            "\n\nINSTRUCTIONS IN FORCE:\n" + told_rows,
            "\n\nTHE SHOPPER'S REQUEST:\n" + ctx.request,
            "\n\nLATEST LINES FROM THE SHOPPER AND THE STORE (oldest first):\n" + ("\n".join(needs.exchanges[-8:]) or "(none)"),
            "\n\nREQUIREMENTS NOW: " + needs.text_body(),
            "\n\nTHE CART: " + ctx.cart.trolley.summary() + " " + cls.basket_text(ctx),
            "\n\nTHE FINISH ABOUT TO HAPPEN: " + kind,
            "\n\nDONE AFTER THE LAST CART CHANGE:\n" + ("\n".join(since) or "(no message or action since the cart last changed)"),
            f"\n\nMESSAGES LEFT: {msgs_left}\nSTEPS LEFT: {ctx.moves_left()} (the finish itself takes one)",
            "\n\nTOOLS THAT MAY CARRY OUT A STEP:\n" + tool_rows,
        ])

    @classmethod
    def _finish_calls(cls, ctx, answer: Any, usable: list[dict[str, Any]]) -> list[tuple[str, dict[str, Any], str]]:
        calls: list[tuple[str, dict[str, Any], str]] = []
        if answer is None:
            return calls
        names = {tdef["function"]["name"] for tdef in usable}
        for step in (answer.get("pending") or [])[:FINISH_TURNS]:
            if not isinstance(step, dict) or step.get("tool") not in names or not isinstance(step.get("args"), dict):
                continue
            call_args, faults = ArgCheck.execute(dict(step["args"]), ctx.shelf.shapes[step["tool"]])
            if not faults:
                calls.append((step["tool"], call_args, str(step.get("why") or "")[:120]))
        return calls

    @classmethod
    def finish_steps(cls, ctx, kind):
        needs, cart, talk, wire = ctx.needs, ctx.cart, ctx.talk, ctx.wire
        quiet = not (needs.told or needs.exchanges or vars(talk).get("event_seen"))
        if ctx.moves_left() < 2 or wire.done or wire.broke:
            return ""
        edits = vars(cart).get("cart_edits", 0)
        sig = (kind, tuple(sorted(cart.trolley.units.items())), edits, needs.told_version, cart.refusal_n, needs.version)
        checked = vars(talk).setdefault("finish_checked", set())
        # a standing "no" from the shopper keeps the check open until they agree or something changes
        if sig in checked and needs.approval != "no":
            return ""
        checked.add(sig)
        msg_tool = ctx.role("message")
        msgs_left = 0 if not msg_tool else max(0, ctx.ask_limit() - ctx.budget.msgs_sent)
        since = cls._sent_since(ctx, edits)
        usable = cls._finish_tools(ctx, msgs_left, msg_tool)
        ask_text = cls._finish_prompt(ctx, kind, since, msgs_left, usable)
        consulted = not quiet and wire.usable(FINISHER) and wire.spare() > LAST_CALL + 10
        answer = wire.consult(ask_text, 2500, FINISHER) if consulted else None
        calls = cls._finish_calls(ctx, answer, usable)
        target = Asking.text_arg(ctx, msg_tool) if msg_tool else None
        planned_messages = len([c for c in calls if c[0] == msg_tool])
        tell_basket = kind == "order" and cart.trolley.units and target and (not since or needs.approval == "no")
        if tell_basket and not planned_messages and msgs_left > planned_messages:
            calls.append((msg_tool, {target: cls.basket_text(ctx)}, "the basket told before the order"))
        ctx.log.append(dict(kind="finish_check", finish=kind, answer=answer, calls=[c[0] for c in calls]))
        calls = calls[: max(0, ctx.moves_left() - 1)]
        return cls._run_finish_calls(ctx, calls, msg_tool) if calls else ""

    @staticmethod
    def _sent_since(ctx, edits):
        bare = not ctx.cart.trolley.units
        return [f"- message sent at step {tick}: {said}" for tick, said, at in vars(ctx.talk).get("msg_log", [])
                if at == edits and (edits > 0 or bare)]

    @staticmethod
    def _finish_tools(ctx, msgs_left, msg_tool):
        barred = ctx.finishers() | {ctx.role("cart")}
        if msgs_left <= 0:
            barred.add(msg_tool)
        return [tdef for tdef in ctx.shelf.toolset if tdef["function"]["name"] not in barred]

    @classmethod
    def _state_mark(cls, ctx) -> tuple:
        needs, budget = ctx.needs, ctx.budget
        return (needs.version, needs.told_version, budget.msg_cap, budget.step_cap, ctx.cart.refusal_n, len(ctx.cart.trolley.units))

    @classmethod
    def _run_finish_calls(cls, ctx, calls: list[tuple[str, dict[str, Any], str]], msg_tool: str | None) -> str:
        proposals = []
        for n, (name, args, _) in enumerate(calls):
            proposals.append({"id": f"finish_step_{ctx.wire.tick}_{n}", "type": "function",
                              "function": {"name": name, "arguments": json.dumps(args)}})
        before = cls._state_mark(ctx)
        ctx.needs.approval = ""
        prior_flag = vars(ctx.talk).get("scripted_batch")
        try:
            responses = cls.dispatch(ctx, proposals, scripted=True)
        finally:
            ctx.talk.scripted_batch = prior_flag
        changed = before != cls._state_mark(ctx)
        done = []
        for (name, args, _), (_, text) in zip(calls, responses):
            done.append(f"{name} {json.dumps(args, ensure_ascii=False)[:400]} -> {text.split(chr(10) + '[status]')[0][:700]}")
        note = "[before this finish the code carried out: " + "; ".join(c[2] for c in calls) + "] " + " | ".join(done)
        ctx.log.append(dict(kind="finish_steps_done", note=note[:600]))
        messaged = msg_tool and any(name == msg_tool for name, _, _ in calls)
        if messaged and (changed or ctx.needs.reply_due or ctx.wire.done):
            return ("Not run yet (no step used). " + note + "\nThe answer changed something: read it against the requirements and "
                    "the instructions in force, fix what it asks for, and finish again.")
        if messaged and ctx.needs.approval == "no":
            return ("Not run yet (no step used). " + note + "\nThe shopper did not agree to finishing this way: read their answer, "
                    "do what it asks (change the cart, wait, answer them), and finish again.")
        ctx.talk.finish_note = note
        return ""

    @classmethod
    def _cart_puts(cls, ctx, proposals: list[dict[str, Any]]) -> list[str]:
        book, cart_tool = ctx.cart.trolley, ctx.role("cart")
        aims = []
        for proposal in proposals:
            func = proposal.get("function") or {}
            if func.get("name") != cart_tool:
                continue
            raw = func.get("arguments")
            if isinstance(raw, dict):
                put_args = raw
            elif isinstance(raw, str) and raw.strip():
                put_args = TextKit.json_object(raw)
            else:
                put_args = {}
            if isinstance(put_args, dict) and book.is_put(put_args) and book.aim(put_args):
                aims.append(book.aim(put_args))
        return aims

    @classmethod
    def dispatch(cls, ctx, proposals, scripted=False):
        cart = ctx.cart
        ctx.talk.scripted_batch = scripted
        Intake.begin_batch(ctx)
        permitted = {tdef["function"]["name"] for tdef in Intake.offered(ctx)}
        cart.batch_puts = []
        cart_tool = ctx.role("cart")
        cart.batch_cart = any((p.get("function") or {}).get("name") == cart_tool for p in proposals if isinstance(p, dict))
        cart.batch_puts.extend(cls._cart_puts(ctx, proposals))
        responses, to_send = cls._screen_all(ctx, proposals, permitted, scripted)
        closing = ctx.finishers()
        leads_finish = bool(to_send) and to_send[0][1] in closing
        if leads_finish:
            cid, call_name, call_args = to_send[0]
            held, call_args = cls.screen_finish(ctx, cid, call_name, call_args)
            if held:
                responses.append((cid, held))
                return responses
            to_send[0] = (cid, call_name, call_args)
        elif to_send and closing:
            keep = max(0, ctx.moves_left() - 1)
            for cid, _, _ in to_send[keep:]:
                responses.append((cid, "Not run (no step used): only the last step is left, and it must be the order (or the decline)."))
            del to_send[keep:]
        cls._send_slabs(ctx, to_send, responses)
        note = vars(ctx.talk).pop("finish_note", "")
        if note and leads_finish:
            first = to_send[0][0]
            responses = [(cid, note + "\n" + text if cid == first else text) for cid, text in responses]
        position = {c["id"]: i for i, c in enumerate(proposals)}
        return sorted(responses, key=lambda pair: position.get(pair[0], 0))

    @classmethod
    def _send_slabs(cls, ctx, to_send, responses):
        for start in range(0, len(to_send), 16):
            slab = to_send[start: start + 16]
            wire = ctx.wire
            if not (wire.done or wire.broke or ctx.moves_left() <= 0):
                responses.extend(cls.send_chunk(ctx, slab[: ctx.moves_left()]))
                continue
            responses.extend((cid, "Not run: the episode is over or no steps are left.") for cid, _, _ in slab)

    @classmethod
    def _screen_all(cls, ctx, proposals: list[dict[str, Any]], permitted: set[str], scripted: bool):
        responses: list[tuple[str, str]] = []
        to_send: list[tuple[str, str, dict[str, Any]]] = []
        queued_messages = 0
        cart_held = False
        cart_tool, book = ctx.role("cart"), ctx.cart.trolley
        for proposal in proposals:
            if cart_held and proposal["function"]["name"] == cart_tool:
                responses.append((proposal["id"], "Not run (no step used): an earlier cart change in this reply was not run; read why, "
                                                  "then send the cart changes again."))
                continue
            refusal, call_name, call_args = cls.screen(ctx, proposal, permitted, scripted, len(proposals), queued_messages)
            if refusal:
                if not cart_held:
                    cart_held = (call_name == cart_tool and isinstance(call_args, dict) and bool(call_args)
                                 and not book.is_put(call_args))
                responses.append((proposal["id"], refusal))
                continue
            if call_name == ctx.role("message"):
                queued_messages += 1
            to_send.append((proposal["id"], call_name, call_args))
        return responses, to_send

    @classmethod
    def send_chunk(cls, ctx, slab: list[tuple[str, str, dict[str, Any]]]) -> list[tuple[str, str]]:
        acts = [{"name": call_name, "args": call_args} for _, call_name, call_args in slab]
        step_no = ctx.wire.tick + 1
        resp = ctx.wire.act(acts)
        ctx.log.append({"kind": "env", "step": step_no, "actions": acts, "reply": resp})
        if resp is None:
            return [(cid, "The call failed (network or server). Try again.") for cid, _, _ in slab]
        if isinstance(resp.get("observations"), list):
            obs_list = resp.get("observations")
        else:
            obs_list = [resp.get("observation")]
        collected = []
        last_index = len(obs_list) - 1
        for idx, (cid, call_name, call_args) in enumerate(slab):
            if idx > last_index:
                collected.append((cid, "Not run: the episode finished before this call."))
            else:
                collected.append((cid, cls.absorb(ctx, call_name, call_args, obs_list[idx], resp, idx == last_index)))
        return collected

    @classmethod
    def _book_call(cls, ctx, call_name, call_args, obs):
        talk, cart, budget = ctx.talk, ctx.cart, ctx.budget
        msg_tool, basket_tool = ctx.role("message"), ctx.role("cart")
        if call_name == msg_tool:
            budget.msgs_sent += 1
            pieces = [val for val in call_args.values() if isinstance(val, str)]
            said = " ".join(str(val) for val in pieces)
            log_book = vars(talk).setdefault("msg_log", [])
            log_book.append((ctx.wire.tick, said[:600], vars(cart).get("cart_edits", 0)))
            if not vars(talk).get("scripted_batch"):
                outcome = TextKit.as_text(obs)[:300]
                talk.questions_log.append(f"- (the assistant's own message): \"{said[:300]}\" -> {outcome}")
        if call_name == basket_tool:
            cart.cart_edits = 1 + vars(cart).get("cart_edits", 0)
        kind = ctx.call_kind(call_name, call_args)
        talk.last_kind = kind
        if kind == "order":
            cart.orders_tried += 1
        resets = call_name in (basket_tool, msg_tool) or call_name in ctx.finishers()
        sig = call_name + TextKit.as_text(call_args)
        # a call whose result is a fault (a busy store, a timeout) may be repeated once as it was
        faulted = vars(talk).setdefault("fault_retries", set())
        retry_once = not resets and sig not in faulted and Closing.code_fault(ctx, obs)
        if retry_once:
            faulted.add(sig)
        elif not resets:
            talk.last_calls = (talk.last_calls + [sig])[-2:]
        else:
            talk.last_calls = []
        ctx.shelf.archive.add(obs)
        budget.absorb_n = 1 + vars(budget).get("absorb_n", 0)
        if call_name == basket_tool or vars(ctx.shelf).get("covering"):
            return
        cart.since_cart = 1 + vars(cart).get("since_cart", 0)

    @classmethod
    def _track_lines(cls, ctx, call_name: str, call_args: dict[str, Any], obs: Any) -> None:
        units = ctx.cart.trolley.units
        if not units:
            return
        shelf = vars(ctx.shelf)
        fresh: set[str] = set()
        stock_rule = bool(shelf.get("stock_seen"))
        asked = Lookup.asked_ids(ctx, call_args)
        rows_ok = bool(shelf.get("rows_live")) and not shelf.get("rows_lag")
        lookups_only = not rows_ok and call_name not in (ctx.role("cart"), ctx.role("cart_view"))

        def seen_line(d: dict[str, Any]) -> None:
            hits = {v for k, v in d.items() if isinstance(v, str) and v in units and "id" in str(k).lower()}
            if not hits:
                return
            if not stock_rule:
                if not any(isinstance(v, dict) for v in d.values()):
                    return
            elif Reading.stock_state(ctx, d) is None or (lookups_only and not hits & asked):
                return
            fresh.update(v for v in d.values() if isinstance(v, str) and v in units)
        TextKit.walk_dicts(obs, seen_line)
        cart = vars(ctx.cart)
        for line_id in fresh:
            cart.setdefault("line_read", {})[line_id] = ctx.wire.tick
            cart.setdefault("line_seq", {})[line_id] = ctx.budget.absorb_n

    @classmethod
    def _count_late(cls, ctx, call_name: str, call_args: dict[str, Any]) -> None:
        details, search = ctx.role("details"), ctx.role("search")
        looked = bool(details) and call_name not in (details, search) and call_name in Lookup.lookup_tools(ctx)
        named_ids = {val for val in call_args.values() if isinstance(val, str)}
        if ctx.moves_left() >= Intake.order_due(ctx):
            return
        if not (call_name in (details, search) or looked) or vars(ctx.shelf).get("covering"):
            return
        held = set(ctx.cart.trolley.units)
        if call_name == details and named_ids & held:
            return
        if looked and named_ids & (held | {k.rpartition("|")[0] for k in ctx.cart.passed}):
            return
        ctx.budget.late_n = vars(ctx.budget).get("late_n", 0) + 1

    @classmethod
    def _cart_remarks(cls, ctx, call_args, obs, remarks):
        cart = ctx.cart
        book = cart.trolley
        cart.order_vetted = False
        book.changed_at = ctx.wire.tick
        held_before = dict(book.units)
        remarks.append(book.booked(call_args, obs))
        cls._guarded(ctx, "after_put_error", lambda: cls._after_cart(ctx, held_before, call_args, obs))
        put_item = book.aim(call_args)
        putting = bool(put_item) and book.is_put(call_args)
        if putting:
            cart.refusals_seen[put_item] = cart.refusal_n
            tally = vars(cart).setdefault("puts", {})
            tally[put_item] = 1 + tally.get(put_item, 0)
        put_units = book.count_arg(call_args)
        verified = putting and f"{put_item}|{ctx.needs.version}" in cart.passed
        if not verified or (put_units is not None and put_units <= 0):
            return
        if book.held(put_item) > 0 and put_item not in Lookup.dead_lines(ctx):
            remarks.append(f"An independent check found {put_item} meets every requirement now in force, reading each property "
                           "under the value rules (where this store writes the value itself in a field, only those words meet it); keep it unless a new change rules it out.")

    @staticmethod
    def _after_cart(ctx, held_before, call_args, obs):
        Lookup.cart_moved(ctx, held_before)
        Lookup.mark_fresh(ctx, obs)
        Lookup.after_put(ctx, call_args, obs)

    @classmethod
    def _hear_lines(cls, ctx, call_name, call_args, obs, refused, remarks):
        needs = ctx.needs
        qtext = None
        if call_name == ctx.role("message"):
            qtext = " ".join(str(val) for val in call_args.values() if isinstance(val, str))
        caught = Coverage.shopper_lines(ctx, call_name, obs, refused)
        if not caught:
            return
        version_before, loose_before, cap_before = needs.version, len(needs.let_go), needs.spend_cap
        remarks.append(needs.hear(caught, qtext))
        Referencing.ref_take(ctx)
        news = vars(needs).get("allowance_news")
        if news:
            needs.allowance_news = None
            cls._guarded(ctx, "allowance_error", lambda: ctx.apply_allowance(news))
        if needs.told_version != vars(ctx.talk).get("told_seen"):
            ctx.talk.told_seen = needs.told_version
            ctx.cart.order_vetted = False
        if needs.version <= version_before:
            return
        unchanged_limits = len(needs.let_go) <= loose_before and needs.spend_cap == cap_before
        if qtext is not None and unchanged_limits:
            return
        review = cls._guarded(ctx, "change_review_error", lambda: cls.change_review(ctx))
        if review:
            remarks.append(review)

    @classmethod
    def absorb(cls, ctx, call_name, call_args, obs, resp, last):
        shown, obs = obs, TextKit.lensed(obs)
        cls._book_call(ctx, call_name, call_args, obs)
        gone_now = cls._guarded(ctx, "caps_error", lambda: Lookup.learn_caps(ctx, call_name, call_args, obs), [])
        cls._track_lines(ctx, call_name, call_args, obs)
        records = ctx.shelf.archive
        read_ok = not (isinstance(obs, dict) and obs.get("error"))
        if read_ok and call_name == ctx.role("details"):
            records.read.update(val for val in call_args.values() if isinstance(val, str) and val in records.ids)
        cls._count_late(ctx, call_name, call_args)
        remarks = ["[stock] " + Lookup.dead_note(ctx, gone_now)] if gone_now else []
        redirected = cls._guarded(ctx, "notes_remark_error", lambda: Referencing.notes_remark(ctx, call_name, call_args), "")
        if redirected:
            remarks.append("[notes] " + redirected)
        cls._outcome_remarks(ctx, call_name, call_args, obs, resp, last, remarks)
        clip_at = SEARCH_TEXT if call_name == ctx.role("search") else RECORD_TEXT
        shown_text = TextKit.as_text(TextKit.clipped(shown, clip_at))
        if last and resp.get("finished"):
            shown_text += " [the episode is finished]"
        summary = " ".join(filter(None, remarks))
        return f"{shown_text}\n[note] {summary}" if summary else shown_text

    @classmethod
    def _guarded(cls, ctx, kind, work, fallback=None):
        try:
            return work()
        except Exception as err:
            ctx.log.append({"kind": kind, "error": f"{type(err).__name__}: {err}"[:300]})
            return fallback

    @classmethod
    def _outcome_remarks(cls, ctx, call_name, call_args, obs, resp, last, remarks):
        book = ctx.cart.trolley
        if call_name == ctx.role("cart"):
            cls._cart_remarks(ctx, call_args, obs, remarks)
        if call_name == ctx.role("cart_view"):
            book.viewed(obs, ctx.wire.tick)
            remarks.append(book.summary())
        refused = Reading.refused(ctx, call_name, obs, resp if last else {"finished": False})
        if refused and ctx.talk.last_kind == "order":
            ctx.cart.cart_seq = ctx.budget.absorb_n
        remarks.append(Reading.notes_for(ctx, call_name, obs, refused))
        if call_name == ctx.role("search"):
            cls._search_remarks(ctx, call_args, obs, remarks)
        if last and cls._audit_due(ctx, call_name):
            ctx.cart.early_audit = True
            ahead = Judging.review_decline(ctx, early=True)
            if ahead:
                remarks.append("Nothing verified fits yet. " + ahead)
        cls._hear_lines(ctx, call_name, call_args, obs, refused, remarks)

    @classmethod
    def _search_remarks(cls, ctx, call_args: dict[str, Any], obs: Any, remarks: list[str]) -> None:
        remarks.append(Coverage.search_note(ctx, call_args, obs))
        vars(ctx.shelf).get("named_pages", set()).discard(json.dumps(call_args, sort_keys=True, default=str))
        try:
            remarks.append(Lookup.merge_joined(ctx, obs))
        except Exception as err:
            ctx.log.append({"kind": "merge_joined_error", "error": f"{type(err).__name__}: {err}"[:300]})

    @classmethod
    def _audit_due(cls, ctx, call_name: str) -> bool:
        if vars(ctx.cart).get("early_audit"):
            return False
        if call_name in (ctx.role("cart"), ctx.role("message")) or call_name in ctx.finishers():
            return False
        if not FIRST_ORDER_BY < ctx.moves_left() <= ctx.budget.step_cap // 3 + 2 or ctx.cart.trolley.units:
            return False
        version = str(ctx.needs.version)
        return not any(key.rpartition("|")[2] == version for key in ctx.cart.passed)


class Budget:
    def __init__(self, step_cap: int):
        self.step_cap = step_cap
        self.msg_cap: int | None = None
        self.msgs_sent = 0
        self.stuck = 0


class Transcript:
    def __init__(self):
        self.chat: list[dict[str, Any]] = []
        self.last_calls: list[str] = []
        self.warned: set[str] = set()
        self.questions_log: list[str] = []
        self.asking = True


class Shelf:
    def __init__(self, toolset: list[dict[str, Any]]):
        self.toolset = toolset
        self.shapes = {}
        for tdef in toolset:
            spec = tdef["function"]
            self.shapes[spec["name"]] = spec.get("parameters") or {"type": "object", "properties": {}}
        self.archive = Records()
        self.role_map: dict[str, str] = {}
        self.word_seen: list[tuple] = []
        self.word_kind = ""
        self.lead_seen: list[tuple] = []
        self.lead_bounds: dict[tuple, list] = {}
        self.lead_words = 0
        self.lead_ctx: tuple | None = None


class Basket:
    def __init__(self):
        self.trolley = CartBook(None, {})
        self.orders_tried = 0
        self.refusal_n = 0
        self.order_vetted = False
        self.rank_checked: set[str] = set()
        self.decline_vetted = False
        self.audit_done = False
        self.rulings: dict[str, str] = {}
        self.checked_adds: set[str] = set()
        self.passed: set[str] = set()
        self.refused_lines: set[str] = set()
        self.refusals_seen: dict[str, int] = {}


class Pointers:
    def __init__(self):
        self.refs: list[dict[str, Any]] = []


class Episode:
    def __init__(self, pdata: dict[str, Any]):
        env = pdata["environment"]
        self.wire = Wire(env["binding"])
        self.pview = env["policy_view"]
        self.request = str(self.pview.get("query") or "")
        offered_tools = []
        for tdef in self.pview.get("tools") or []:
            if isinstance(tdef, dict) and isinstance(tdef.get("function"), dict):
                offered_tools.append(tdef)
        self.shelf = Shelf(offered_tools)
        self.budget = Budget(int(self.pview.get("max_steps") or 30))
        self.wire.done = bool(self.pview.get("finished"))
        self.needs = Needs(self.wire, self.request)
        self.needs.ref_hook = partial(Referencing.ref_fix, self)
        self.cart = Basket()
        self.talk = Transcript()
        self.pointers = Pointers()
        LITERAL_VALUES.clear()
        self.needs.field_hook = partial(Judging.store_fields, self)

    @property
    def log(self) -> list[dict[str, Any]]:
        return self.wire.log

    def moves_left(self) -> int:
        return self.budget.step_cap - self.wire.tick

    def role(self, call_name: str) -> str | None:
        return self.shelf.role_map.get(call_name)

    def route(self) -> dict[str, Any]:
        return ROUTES["multi" if self.needs.items() else "single"]

    def ask_limit(self) -> int:
        cap = self.budget.msg_cap
        return ASK_DEFAULT if cap is None else cap

    def msg_reserve(self) -> int:
        return 1 if self.role("message") else 0

    def ask_room(self) -> int:
        return max(0, self.ask_limit() - self.msg_reserve())

    def step_reserve(self) -> int:
        if self.role("message") or self.needs.finish_told():
            return FINISH_STEPS
        return 0

    def apply_allowance(self, news: dict[str, Any]) -> None:
        plan = self.budget
        caps_before = (plan.msg_cap, plan.step_cap)
        msg_total, msg_left = news.get("messages_total"), news.get("messages_left")
        if TextKit.is_number(msg_total):
            plan.msg_cap = max(plan.msgs_sent, int(msg_total))
        elif TextKit.is_number(msg_left):
            plan.msg_cap = plan.msgs_sent + max(0, int(msg_left))
        step_total, step_left = news.get("steps_total"), news.get("steps_left")
        tick = self.wire.tick
        if TextKit.is_number(step_total) and int(step_total) > tick:
            plan.step_cap = min(plan.step_cap, int(step_total))
        elif TextKit.is_number(step_left) and int(step_left) >= 0:
            plan.step_cap = min(plan.step_cap, tick + int(step_left))
        if caps_before == (plan.msg_cap, plan.step_cap):
            return
        self.talk.asking = self.talk.asking and plan.msgs_sent < self.ask_room()
        self.log.append(dict(kind="allowance", news=news, caps=[plan.msg_cap, plan.step_cap]))

    def call_kind(self, call_name: str, call_args: dict[str, Any] | None) -> str | None:
        order_tool, decline_tool = self.role("order"), self.role("decline")
        if not call_name or call_name not in (order_tool, decline_tool):
            return None
        if order_tool != decline_tool:
            return "decline" if call_name != order_tool else "order"
        for piece in (call_args or {}).values():
            if not isinstance(piece, str) or len(piece) > 40:
                continue
            lowered = piece.lower()
            if re.search(DECLINE_WORDS, lowered):
                return "decline"
            if re.search(ORDER_WORDS, lowered):
                return "order"
        return "order"

    def finishers(self) -> set[str]:
        return {self.role("order"), self.role("decline")} - {None}

    def clock_text(self) -> str:
        remaining = max(0.0, self.wire.spare() - LAST_CALL)
        if remaining > (DEADLINE - LAST_CALL) / 2:
            return ""
        rounded = int(remaining // CLOCK_STEP) * CLOCK_STEP
        if rounded <= 0:
            return f"under {CLOCK_STEP} s left"
        return f"about {rounded} s left"

    def _closing_advice(self, spare: int) -> str:
        if spare <= 2:
            return (" LAST STEPS: order the cart now if it holds a candidate for " + ("every product asked for" if self.needs.items()
                    else "the item") + "; otherwise put the best candidate in (a verified fit, else the cheapest in-stock, "
                    "in-budget listing whose record shows no requirement failing) and order. Decline only when the records "
                    "show that nothing can fit.")
        if spare <= 5 or self.wire.spare() < CLOSING:
            return (" FINISH NOW: fix the cart to " + ("one line per product asked for" if self.needs.items() else "one line")
                    + " (a verified fit, else the cheapest in-stock, in-budget listing whose record shows no requirement "
                    "failing; a sold-out line out), re-check its stock, then order. Decline only when the records show that "
                    "nothing can fit.")
        if self.wire.spent_out():
            several = bool(self.needs.items())
            return (" MODEL BUDGET USED: stop exploring; within the next few steps put the best candidate of "
                    + ("every product asked for (one line each, at its quantity)" if several else "the item asked for")
                    + " in the cart (a verified fit where one exists, else the closest fit the records show), re-check its "
                    "stock and price, and order. A decline is right only when the records show that "
                    + ("some product has" if several else "the store has") + " no listing that can meet its requirements.")
        if self.cart.orders_tried != 0:
            return ""
        if spare <= Intake.order_due(self) + 7 and spare > Intake.order_due(self) and Intake.narrowing(self):
            return f" Steps are running low: within the next few steps put your best verified fit in the cart and order; searching and reading details end when {Intake.order_due(self)} steps remain, and a refused order needs about 4 steps to repair."
        if spare <= Intake.order_due(self) and Intake.narrowing(self):
            return " Searching has ended (details only for the cart's line, listings a check named, or, while some product has no verified fit, up to 3 more reads or searches in all): put your best verified fit in the cart, re-check its stock and price, and order; a refused order still needs about 4 steps to repair."
        return ""

    def status_line(self) -> str:
        spare = self.moves_left()
        plan = self.budget
        sent_part = f"messages sent {plan.msgs_sent}"
        if plan.msg_cap is not None:
            sent_part += f" of {plan.msg_cap}"
        pieces = [f"[status] steps used {self.wire.tick} of {plan.step_cap}, {spare} left", sent_part]
        clock = self.clock_text()
        if clock:
            pieces.append(clock)
        line = "; ".join(pieces) + ".\n[requirements now] " + self.needs.text_body() + Referencing.ref_line(self) \
            + Coverage.product_steps(self)
        line += self.needs.told_text()
        try:
            line += Checkout.bound_hint(self)
            line += Checkout.dependency_hint(self)
        except Exception:
            pass
        if self.msg_reserve() and plan.msgs_sent < self.ask_limit():
            line += " (the planned questions leave one message for the stated finishing step: yours to send, or sent for you at the finish)"
        usable = [tdef["function"]["name"] for tdef in Intake.offered(self)]
        if len(usable) < len(self.shelf.toolset):
            line += "\n[tools usable now] " + ", ".join(usable) + " (any other tool call is refused without using a step" + (
                "; hidden-need questions are also planned and sent for you" if self.talk.asking else "") + ")."
        if self.cart.trolley.used:
            line += "\n" + self.cart.trolley.summary()
        return line + self._closing_advice(spare)


class Replying:
    @staticmethod
    def applies(ctx: Episode) -> bool:
        return bool(ctx.needs.reply_due)

    @staticmethod
    def run(ctx: Episode) -> bool:
        reply, ctx.needs.reply_due = ctx.needs.reply_due, ""
        return Asking.reply_back(ctx, reply)


class Opening:
    @staticmethod
    def applies(ctx: Episode) -> bool:
        return True

    @staticmethod
    def run(ctx: Episode) -> bool:
        return Judging.open_named(ctx)


class Sweeping:
    @staticmethod
    def applies(ctx: Episode) -> bool:
        return not ctx.talk.asking or bool(vars(ctx.talk).get("empty_plans"))

    @staticmethod
    def run(ctx: Episode) -> bool:
        return Reading.sweep_step(ctx)


class ValuePaging:
    @staticmethod
    def applies(ctx: Episode) -> bool:
        return True

    @staticmethod
    def run(ctx: Episode) -> bool:
        return Coverage.value_pages(ctx)


class Questioning:
    @staticmethod
    def applies(ctx: Episode) -> bool:
        return ctx.talk.asking

    @staticmethod
    def run(ctx: Episode) -> bool:
        return Asking.pose_planned(ctx)


class Following:
    @staticmethod
    def applies(ctx: Episode) -> bool:
        return True

    @staticmethod
    def run(ctx: Episode) -> bool:
        return Referencing.ref_step(ctx)


class CheapWalk:
    @staticmethod
    def applies(ctx: Episode) -> bool:
        return True

    @staticmethod
    def run(ctx: Episode) -> bool:
        return Walking.walk_step(ctx)


class TurnEngine:
    CODE_STAGES = (Replying, Opening, Sweeping, ValuePaging, Questioning, Following, CheapWalk)

    def __init__(self, ctx: Episode):
        self.ctx = ctx
        self.idle = 0
        self.failures = 0
        self.seq_no = 0
        self.last_ok = 0.0
        self.missed = 0

    def open_chat(self) -> None:
        ctx = self.ctx
        first_view = ctx.pview.get("opening")
        opener = "SHOPPER REQUEST:\n" + ctx.request
        if first_view is not None:
            opener += "\n\nOPENING OBSERVATION:\n" + TextKit.as_text(first_view)
            ctx.shelf.archive.add(first_view)
        summary = ctx.needs.overview(ctx.budget.step_cap, ctx.budget.msg_cap)
        if summary:
            opener += "\n\n" + summary
        opener += "\n\n" + ctx.status_line()
        ctx.talk.chat = [{"role": "system", "content": ACT_RULES}, {"role": "user", "content": opener}]

    def may_go_on(self) -> bool:
        ctx = self.ctx
        if ctx.wire.done or ctx.wire.broke or ctx.moves_left() <= 0:
            return False
        return ctx.wire.spare() >= LAST_CALL

    def code_turn(self) -> bool:
        return any(stage.applies(self.ctx) and stage.run(self.ctx) for stage in self.CODE_STAGES)

    def collect_calls(self, msg: dict[str, Any]) -> list[dict[str, Any]]:
        calls = []
        for raw in msg.get("tool_calls") or []:
            if not isinstance(raw, dict):
                continue
            self.seq_no += 1
            func = raw.get("function") or {}
            arguments = func.get("arguments")
            if not isinstance(arguments, str):
                arguments = json.dumps(func.get("arguments") or {})
            calls.append({"id": raw.get("id") or f"call_{self.seq_no}", "type": "function",
                          "function": {"name": func.get("name"), "arguments": arguments}})
        return calls

    def lost_turn(self, model_list: list[str]) -> str:
        wire = self.ctx.wire
        self.missed += 1
        if not wire.usable(model_list, True):
            return "stop"
        if monotonic() - self.last_ok > DOWN_AFTER and (
                self.missed >= 2 or (wire.down() and len(wire.call_models) >= 2)):
            return "stop"
        if wire.call_fail not in ("empty", "refused"):
            return "next"
        self.failures += 1
        return "stop" if self.failures >= 2 else "next"

    def actor_turn(self) -> str:
        ctx = self.ctx
        talk = ctx.talk
        model_list = ACTOR[1:] + ACTOR[:1] if ctx.budget.stuck >= STUCK_LIMIT else ACTOR
        msg = ctx.wire.query_model(talk.chat, model_list, 2500, ctx.shelf.toolset)
        if msg is None:
            return self.lost_turn(model_list)
        self.last_ok, self.missed = monotonic(), 0
        calls = self.collect_calls(msg)
        said = msg.get("content") or ""
        if not isinstance(said, str):
            said = TextKit.as_text(said)
        binding = {ctx.role("cart"), ctx.role("order"), ctx.role("decline")} - {None}
        if talk.asking and any(c["function"]["name"] in binding for c in calls) and Asking.pose_planned(ctx, urge=True):
            return "next"
        if not calls:
            self.idle += 1
            talk.chat.append({"role": "assistant", "content": said or "(no tool call)"})
            talk.chat.append({"role": "user", "content": "Every reply must call at least one of the supplied tools. Continue.\n" + ctx.status_line()})
            return "stop" if self.idle >= IDLE_LIMIT else "next"
        self.idle = 0
        ctx.log.append({"kind": "model", "content": said[:2000], "calls": [c["function"] for c in calls]})
        talk.chat.append({"role": "assistant", "content": said or None, "tool_calls": calls})
        tick_before = ctx.wire.tick
        Asking.post_replies(ctx, Checkout.dispatch(ctx, calls))
        ctx.budget.stuck = 0 if ctx.wire.tick != tick_before else ctx.budget.stuck + 1
        return "next"

    def run(self) -> None:
        self.last_ok = monotonic()
        for _ in range(TURN_LIMIT):
            if not self.may_go_on():
                break
            Closing.catch_up(self.ctx)
            if self.code_turn():
                continue
            if self.actor_turn() == "stop":
                break


class Intro:
    @staticmethod
    def applies(ctx: Episode) -> bool:
        return not ctx.wire.done

    @staticmethod
    def run(ctx: Episode) -> None:
        Intake.read_request(ctx)
        ctx.log.append({"kind": "plan", "plan": ctx.needs.reading, "roles": ctx.shelf.role_map})


class Shopping:
    @staticmethod
    def applies(ctx: Episode) -> bool:
        return True

    @staticmethod
    def run(ctx: Episode) -> None:
        engine = TurnEngine(ctx)
        engine.open_chat()
        engine.run()


class Wrapup:
    @staticmethod
    def applies(ctx: Episode) -> bool:
        return not ctx.wire.done and not ctx.wire.broke

    @staticmethod
    def run(ctx: Episode) -> None:
        Closing.finish_now(ctx)


EPISODE_STAGES = (Intro, Shopping, Wrapup)


def run_episode(ctx: Episode) -> list[dict[str, Any]]:
    if ctx.wire.done:
        return ctx.log
    for stage in EPISODE_STAGES:
        if stage.applies(ctx):
            stage.run(ctx)
    wire = ctx.wire
    ctx.log.append({"kind": "end", "finished": wire.done, "steps": wire.tick,
                    "cost": round(wire.spend, 5), "seconds": round(monotonic() - wire.t0, 1)})
    return ctx.log


def agent_main(pdata: dict[str, Any]) -> list[dict[str, Any]]:
    errand = Episode(pdata)
    try:
        return run_episode(errand)
    except Exception as err:
        errand.log.append({"kind": "crash", "error": f"{type(err).__name__}: {err}"[:500]})
        try:
            if not errand.wire.done and not errand.wire.broke:
                Closing.finish_now(errand)
        except Exception as second_err:
            errand.log.append({"kind": "crash", "error": f"{type(second_err).__name__}: {second_err}"[:500]})
        return errand.log


__all__ = ["agent_main"]
