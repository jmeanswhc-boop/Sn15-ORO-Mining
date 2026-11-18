from __future__ import annotations

import json
import re
from time import monotonic, sleep
from typing import Any

from src.agent.proxy_client import ProxyClient

ACTOR = ["anthropic/claude-haiku-5.5", "openai/gpt-6-luna", "anthropic/claude-haiku-5.5"]
READER = ["anthropic/claude-sonnet-5.5", "openai/gpt-6-luna"]
CHECKER = ["anthropic/claude-haiku-5.5", "anthropic/claude-sonnet-5.5"]
AUDITOR = ["anthropic/claude-haiku-5.5", "anthropic/claude-sonnet-5.5"]
LISTENER = ["anthropic/claude-haiku-5.5", "anthropic/claude-sonnet-5.5"]
PLANNER = ["anthropic/claude-haiku-5.5", "anthropic/claude-sonnet-5.5"]
ASK_SEAT = ["anthropic/claude-haiku-5.5", "google/gemini-3.8-flash"]
OUR_ASK = True
ASK_FIRST_PICKS = 2
ASK_MENU = 24
ASK_SAMPLES = 3
ASK_RANKINGS = 4
ASK_CHARS = 70

DEADLINE = 265.0
CLOSING = 70.0
LAST_CALL = 28.0
CLOCK_FLOOR = 4.0
CHECK_FLOOR = LAST_CALL + 12
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
PUT_WORDS = frozenset("add put hold set insert place keep include in plus increase append".split())
DROP_WORDS = frozenset("remove drop release delete take out clear discard minus decrease subtract off unhold".split())
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
FOLLOW_PAGES = 2
FOLLOW_DEEP = 5
FOLLOW_TOTAL = 8
VALUE_TOTAL = 8
VALUE_TOTAL_PER = 5
LABEL_TRIES = 4
LABEL_OPENS = 3
CEIL_CALLS = 4
CEIL_MORE = 3
REF_QUERY_WORDS = 8
CEIL_OPENS = 3
LINK_WORDS = frozenset("and or the for with of a an in on to by from at as is are be this that it its your our you all any not no".split())
ASK_AFTER_SETS = 2
ASK_GRACE_STEPS = 3
ASK_AFTER_STEPS = 5

ACT_RULES = """Situation: you help the single customer of a simulated shop, using only the tools handed to you. The run is marked 0 or 1. It gets the 1 only if all of this holds: what you buy satisfies all conditions the request puts on paper, all hidden wishes the shopper lets out when you ask, and all changes announced as you go; the count and the spending limit are correct; if cheapest (or best) is demanded, the one you bought is exactly that; and you kept every procedural instruction of the request. If no item can satisfy all that, the proper close is a decline done exactly as the request asks.

Steps:
a) Get to know the shop's mechanics.
- Every tool call costs a step, rejected and malformed ones too, and steps are few. Thinking is free.
- One reply can hold several calls if none depends on another's output (opening the details of some listings, for example); they execute in turn. A purchase or a decline must always go out on its own.
- If the details tool is happy to receive a LIST of ids, a single call brings back the complete records of all of them for ONE step (as many as it allows): rely on this whenever you open two listings or more.
- Shops name tools, parameters and outputs in their own ways. Read every tool's JSON schema and description and use exactly the names and accepted values given there. The earliest outputs show you what each tool reveals (price, stock, detailed properties).
- Learn early which category labels the shop puts on listings of the needed kind. When labels neighbour each other, choose the one agreeing with the request's full phrasing; search and buy under it.
- No two shops search alike; work out this one's behaviour from its first output. Results may be ranked by relevance or listed in a set id order; keyword search may require all the words (each extra word cuts the total) or be content with any of them (each extra word swells the total, listings hitting more words coming higher); perhaps few rows or pages are readable; a filter may operate in just one mode (keyword mode may disregard the price or category filter, filter mode may disregard the words). Compare a reported total with the rows shown: a higher number means the cheapest or fitting listing may lie on a further page or beyond what is visible. If all words must match, narrow down with more exact words (or with price and category filters plus a reduced price ceiling) till the readable pages can hold the total. Write brief queries (between one word and three) whose most telling words come first: some shops only use one or two opening words and throw away the rest; hence a long query may effectively look up just its opening words (put the long query's rows and total next to those of a short one starting the same way; the notes explain the outcome). If all words must match, a required value written ahead of the item kind (value at the front: "blue chair") gives you precisely the listings with that value. If any one word suffices, generic words (a unit, the item kind) drag in nearly the whole shelf and drown out the rest: a required value's characteristic figure or word on its own ("0.4", "30m", "herringbone") yields only listings containing it, so look it up alone and also in the other forms listings write it in ("3 atm", "30 m"), go through all pages of that search whenever they can hold its total, and join the item kind to values only to push the nearest hits upward. Keyword lookups may hit property values and category names as well. After every search a note explains what its totals show about this shop's search.

b) Uncover the wishes the shopper keeps to themselves.
- Some needs of the shopper are unwritten; one is given away only to a message naming exactly its property. The question placed FIRST in a message is answered most reliably, so cover one property per message, two at most, likelier one in front: a short, simple, friendly question that mentions the property (example values, one or two, may be included). No rules, positions, labels, lists or instructions.
- Replies like "no preference", "anything is fine" or "don't mind" leave that property unconstrained. A limit or preference the shopper voices becomes a strict requirement from then on, kept in the shopper's own words, numbers included.
- Your planned questions go out automatically after you open a few (two or three) listings of the requested kind, so open them soon. Each answer shows up with its meaning; a wish revealed this way counts exactly like a written one: search again using it together with the written values.
- If you ask yourself: begin by opening a few (two or three) listings' details, next raise what the story suggests (for whom, which look, the place or manner of use), then the properties on which candidates vary. Never ask about what the request already fixed, nor exceed the message allowance.

c) Obey the procedural instructions.
- A request may say how to proceed: how many listings are to be viewed or compared, which points to confirm, which report goes to the shopper, which contents a remark needs, how a decline is done. These are scored like requirements; carry each out word for word and fully. Being told to "open" or "look at" N listings calls for N fully fitting listings, each opened via the details tool.
- When the shopper puts a question to you (for example, to confirm a change), the answer is given for you while there are messages left; the change binds you all the same.
- Plain wishes ("if possible", "would be nice", "ideally") do not count as requirements: they only settle ties between items that already satisfy everything, and they never beat a cheapest rule.
- If the request leaves something that only the shopper is able to settle (no spending limit, or a pair of wishes that may clash so that the shopper must say which comes first), ask them before you pick.

d) Keep up with changes.
- The errand may change: a different budget (raised or lowered), a requirement taken away, exchanged or added, a higher price, an item running out. One line may carry a number of changes, and such lines can turn up in any tool output (as an added field of sentences) or arrive alongside an attempt to order. Every change binds from then on: review your pick and reserves, and when the goal is the cheapest or best one, pick once more from the items acceptable now.
- A REFUSED order leaves the episode running and the cart unchanged. Study the refusal, apply its changes, remove from the cart whatever fails now (an item that sold out, say), add the verified fit that comes next, recheck its price and stock, and order again.
- Prices and stock may also move unannounced, so confirm both right before you try to order; once an item's current price times the quantity exceeds the current budget, it no longer fits.

e) Cope with requests for more than one product.
- Some requests ask for several distinct products, each having its own count, requirements and hidden wishes. Handle each: search every category, establish a fit for every product, and buy everything in one order carrying, for each product, one line at its count. A cart tool receiving a lines list may overwrite the cart with exactly that list, so every call of that kind must contain the lines for every product. Cheapest and best apply per product.
- A requirement may hand you its value only through a reference to a different page or listing ("the colour of the jumpsuit sold by X"): first locate that listing (search for what marks it out, e.g. its kind with its brand or shop), open the record, read the value from the field named, and only after that look for the requested product using that value. Once a "[reference] RESOLVED" line gives that value, it is final until the shopper changes it: the code followed every note sending the reader on to a further listing, as the request allows, so never put back a value from a page the chain went past.

f) Keep an eye on the cart.
- Some cart tools ADD units on top of what a line holds instead of setting the number: putting in 1 of a line held once leaves you with 2. Never add a line that is in the cart; to alter units, add or remove just the difference. After each cart call a note tells you what the code sees in the cart.

g) Judge each listing yourself (your score depends on this).
- The grader checks every requirement against whichever listing field holds its property, reading the value as this shop words it in that field. Spelling does not matter as long as the field's name denotes that property: extra separators or a prefix, a shorter or longer form of the name, or a well-known alias are all fine (a watch's dial size under "dial size", the fit type of pants under "fit", a colour under "color family", a collar type under "neckline"). A field carrying exactly the property's name is decisive, and so is a field carrying that name preceded by one more word (material as clothing material, length as pants length): the moment one of them shows the value, the requirement is met. Two names that differ by separators alone (size_class versus size class) count as a single field, and the value in either meets it. Lacking such fields, the alias field is decisive; the title or the options list may provide the value only if no field whatsoever names the property. A description never proves anything.
- Values are matched word for word in the shop's spelling, disregarding case, separators, spelling variants and whether it is singular or plural; a match also comes from a narrower value including the required one (green from "forest green", cotton from "80% cotton", rtx 3060 from "rtx 3060 12gb"); a single list entry suffices, even when the shopper wants that value and no other: a field offering a number of variants, among them the value, matches. Shades, related words, abbreviations and synonyms (embroidery as "embroidered", technical wood as "tech wood", off-white as cream or ivory, windows 11 as "win11", plain as "solid") match only while no listing in the category puts the value itself into that field: if "matte" is the finish other listings give, then "satin" is no match for matte. Two values with a plus between them ("wool + nylon") match neither one if somewhere else that field holds the other value by itself ("nylon"). Any lines titled FIELD EVIDENCE come from this shop's records and state which of these situations you are in. A different value for this same property is a mismatch (maple is not rosewood, mechanical is not quartz), and a word that just looks similar is one too. A value claiming to fit every possible value ("suitable for all occasions", "all seasons", "universal") in fact names no value and so fails a required one; still, a list naming the required value besides is a match. A field with a different value is no match for the required one merely because the two often go together or one suggests the other (a "zip-up" is not a "pullover" style, "lace-up" is not "slip-on"); the only exceptions are a narrower value including it and a shade that is usually sold as the required colour.
- A field on another property does not stand for the property of the requirement (a style is not shown by a length, the material not by a lining material, a strap type not by a strap length): where the category's listings have a field of the property's own, such a field never stands in for it.
- Count only what the shopper said or revealed. Do not turn a listing down over anything never brought up: a use, quality, condition, occasion, size, gender or age group you inferred from the story or the description. The story is background; if it looks important, ask the shopper about that property instead of guessing. Requirements arise from what the shopper answers, not from what you asked.
- Convert units, then compare quantities as amounts (45 mm and 4.5 cm are equal); the boundary of a limit lies inside it; a range is fine only if each part of it keeps to the limit. When the title has to serve as evidence since the property has no field, a precise required value is met by a title writing its figures and words in the same sequence, extra detail after them allowed (a title with 9' x 12' 4" shows 9' x 12'; one with 9' x 13' does not).
- Figures in titles like "55 x 94 Inch" or "117 x 228 cm" do not tell which measurement each one is: never turn a listing down, or skip opening it, because of them; look at the fields in its details instead.
- Respect an exclusion ("leave out anything with X") and a closed list ("A or B and nothing else").
- Only the category decides a listing's product kind: the grader sees a listing filed under the wanted category AS that kind, even if a different object appears in its description or title (an accessory, a refill, a set, a container, a case), so never skip it, turn it down or rank it lower because of its looks; weigh just price, stock and requirements. No property needs to repeat the category name. If the shop gives its categories labels, the item belongs in the category whose label fits the whole phrasing of the request ("A or B" in a request fits a category "A & B" rather than its narrower neighbour "B"). If the request needs goods on hand, stock has to be shown, and the price times the quantity may not go beyond the budget (being equal is fine). Take a price to be in the budget's unit unless its record states some other unit (with 1000 US cents as the budget, 45 means 45 cents); neither is rescaled.
- Declining is seldom right. When no listing meets the literal wording yet one meets each requirement by these rules (by a narrower value, by a field with a different name, or by a shade or synonym that no listing of the category contradicts through writing the value itself), that is the fitting listing; the only grounds to decline is a search that really produced nothing meeting all requirements by these rules.

h) Locate the cheapest or best if requested.
- For the cheapest, what you order must undercut ALL other items meeting every requirement; one fit found is not the end of the job. If keyword search insists on all words and also matches property values, searching value with kind returns the candidates having the value: walk through all of its pages, opening them by rising price; listing a category under a price ceiling reveals the prices all over the category but no values: from it, learn what cheaper rows exist (keep lowering the ceiling until the readable pages hold the total), and afterwards check the properties of those rows. Make a brief query of each required value plus the item kind, putting the value first ("30m watch", "blue chair"), and read each of its pages if the readable pages hold the total (otherwise add a narrower word for the kind or a further required value); do it again with the kind's other names ("bench", "stool") and with the bare value; when a value is not searchable (a figure the shop strips away), also search, over all pages, the line of each near-fit or fit you found, as two words: its maker and one telling word of its title; a line's other colours and sizes are listings of their own sharing the line's property values and can cost less; put the visible candidates in price order yourself, opening details upward from the lowest; the result is the earliest whose property fields meet everything once every cheaper one was opened and rejected. Hold the next fits in reserve.
- For the best on some measure (highest, longest, largest, most...), line up ALL fitting items on that measure in the same way and choose the extreme. An item that states nothing for the measure can never win.

i) End the run properly.
- Keep your verified fits in a ranked list with one backup or more: your first pick may be sold out when you check out.
- Before you try to order: the cart has exactly one line per requested product (a lone product, a lone line), each the picked item at the needed quantity and nothing else; since your most recent cart change you re-read the price and stock of the picked item and found it available and within budget.
- Decline only once a careful search turns up nothing meeting every requirement, and follow the decline instruction of the request exactly. If offering substitutes is not allowed, the decline remark neither names nor describes nor points to any listing: no ids, no titles, no prices.
- Plan your steps: try ordering for the first time while you still have seven steps or more, so a refusal can still be dealt with. From 7 steps left on, only the finishing, stock-check and cart tools can be used. Make up your mind once; do not keep switching the cart one way and back.

Output: each reply calls at least a single tool and adds brief text naming the current requirements, each verified fit along with its stock state and price, then the move that comes next."""

READ_REQUEST = """Situation: below are a shopper's request and the store's tool list; reply with ONE JSON object, compact (no indentation), with nothing around it.

Steps:
a) Fill in the keys listed here:

"product": the kind of goods sought, in the request's phrasing (when several products are sought, name each, joined with "and").
"items": one entry, in order, for each DIFFERENT product to be bought: {"product": its kind in the request's words, "quantity": the unit count of one listing of it (as "quantity" below), "requirements": the firm conditions set on it (as "requirements" below)}. A single product gives a single entry.
"quantity": how many whole units of one listing are bought; 1 unless more are asked for. A thing sold in one piece (shoes sold as a pair, a set) counts once. Only an express call for multiple units lifts it ("order 2 packs", "I need three of them"); the number of its users or sharers, or of the old ones it had, is part of the story, and the count is still 1.
"requirements": a string list with one string per firm condition on the product, written "property: constraint" and holding precisely the shopper's exclusion, limit, closed list or value (with several products: the conditions of all of them, each written "property (product): constraint"). Budgets, mere wishes, quantity, stock and price stay out, and so does anything that only repeats the category or product kind: words belonging to the category's name (an audience, season or occasion it contains) fall under the category, not under a condition of their own. A condition ties a product property to the demanded value ("with a mechanical movement", "the occasion has to be party", "the pattern has to be herringbone", "at least 4 drawers"). The story never yields a condition: who it is meant for, an event at which someone wears or uses it ("for the staff party", "to wear at his wedding"), the place, moment or manner of using it, how many share it, or the fact that it replaces something old ("a new watch" tells of a replacement and sets no condition). An audience, use or occasion is a condition only when the request makes it part of what the thing itself is ("a blazer made for business", "the occasion has to be business"). A condition's value consists solely of what the request writes down for the item, in words or figures; a quality, motive or wish that comes out of the story (works with what I already have, covers the whole drop, natural and breathable, slim enough for a case) is just a hint. Every word the request sets beside a figure stays in the value, a sizing system or unit too: "a uk 9 boot" gives "size: uk 9", "in eu 42" gives "size: eu 42", never a bare "size: 9"; such a word never tells the currency.
"price_cap": the highest total price, given as the exact number the request writes, in the request's own unit (do no converting: 3050 is what "3050 US cents" gives), or null where no cap appears.
"currency": the unit in which the request states prices, or null.
"in_stock_only": true when only goods in stock or on hand are acceptable.
"priority": "cheapest" only when the request plainly asks for the lowest-priced or cheapest fitting item; "best" only when it plainly asks for the item that scores best on a measure; "none" in all other cases. Saying funds are tight or each penny matters is no call for the cheapest; it only accounts for the cap.
"priority_detail": if "priority" is "best", that measure; "" in any other case.
"message_budget": the integer count of messages allowed by the shopper, null when it is not said.
"step_budget": the integer count of steps or moves the request allows, null when it is not said.
"figures": {"price_cap": the exact words of the request that fix its price ceiling, "message_budget": the exact words fixing its message ceiling, "step_budget": the exact words fixing its step ceiling}, with "" for any ceiling the request does not fix.
"if_then": an entry for each sentence that makes a requirement on one product depend on what another product of this order turns out to have: {"product": the product the requirement falls on, worded like its "items" entry, "property": that requirement's property, "values": [the value or values it must then have, in the request's words], "when": {"product": the product whose value decides it, worded like its "items" entry, "property": its deciding property, "values": [the deciding values, in the request's words]}, "unless": true when the requirement holds except when the deciding product has one of those values, false when it holds only when it has one, "quote": the sentence, word for word}. Empty when there is none.
"fit_with": an entry for each product which the request says must suit, fit or work with some other product chosen for this order too: {"product": the product required to fit, worded like its "items" entry, "with": the product to be fitted, worded like its "items" entry, "quote": the request's sentence stating this, word for word}. Empty when there is none.
"procedures": an entry for each step towards the shopper that the request demands of the assistant, needing a separate message and not being a product question (the shopper's approval of the basket ahead of ordering, a report ahead of acting): {"quote": the request's sentence, word for word, "needs_yes": true when the shopper's consent must come before the assistant acts, "final_cart": true when its subject is the finished cart or basket}. Asking about the products, their intended users or anything else useful never forms such a step, even when invited; neither are rules for choosing, buying or delivering. Empty when there is none.
"rules": all procedural instructions addressed to the assistant (the number of listings to go through, what must be verified or reported, the way to decline, things to avoid or to include), word for word. Whatever the shopper intends to do in person belongs to the story.
"soft_wants": what the shopper would welcome without truly needing it (preferably, ideally, if possible, would be nice). These never reappear in "requirements".
"ask_first": points the shopper alone can resolve, to be asked ahead of choosing: the budget where there is no limit, which wish weighs more where the request concedes that two wishes might be impossible together (then both wishes belong in soft_wants instead of requirements), or a wish the request wants confirmed by the shopper ahead of choosing (approving the order or basket is a "procedures" matter instead). Empty when there is none.
"hints": story details that bear on a product property but leave its value unsaid (where, by whom, or with what it gets used, what it serves, what it must fit, withstand or match), written "property (product): the detail in a few words" ("property: detail" if just one product is sought), the property named as listings would likely name it. When a detail could touch more than one property, give the most likely one first and the next after a slash ("colour / style (chair): must suit a room with one painted wall"): whatever the item must match, suit or go well with (a room, a wall, an outfit, things used along with it) and whatever concerns its appearance relates first of all to its colour and after that to its style or pattern. Drop a detail that only explains a stated condition (why the stated size or material is needed). The weather, a job, travelling, a timetable, or a spot where the item goes unworn and unused is just background, not a hint; so is a purpose that all listings of the kind serve (a coat keeps warm, a dresser stores). Empty when there is none.
No sentence may be left unfiled; each ends up as a budget, a condition, the priority, a hint, a rule, a soft want, or mere story. If it is unclear whether a sentence is a rule, it goes under "rules".
"references": conditions whose value is only given through a pointer to ANOTHER page or listing ("within 5 of the figure Y's page gives", "the colour of the jumpsuit X sells", "a colour other than the one on Z's page"), each written "property (product): as shown on <how the request names that listing>" (or "property (product): different from the one shown on <how the request names that listing>" when the value must not match that listing's); put each into "requirements" too, in identical form. Empty when there is none.
"reference_targets": for every entry of "references", one object describing the listing referred to: {"property": the property concerned, "product": the product concerned, "seller": the brand or shop that the request ties to that listing, word for word, "kind": what type of product the referred listing is, "details": [any other value the request states for it, short and in the request's phrasing ("net weight 167.262186 g", "blue", "4gb")], "marks": [the words that listing would itself show for those details, values only and never a property's name ("bag", "frozen", "156.21 cm", "5", "aluminum")], "relation": "same", "differ" (NOT that listing's value: other than, different from, set apart from), "pairs" (its value has to go with or be compatible with that listing's rather than equal it), "within", "at most" or "at least" (how this value relates to that listing's), "tolerance": with "within", the figure, else null, "unit": its unit, else ""}. Empty when there is none.
"decline_no_names": true if the decline rule of the request forbids naming, offering or suggesting alternatives or substitutes; false if it asks for something to be named; null if there is no rule of that sort.
"no_fit": what the request wants done should no listing meet every requirement, or null when it says nothing on that: {"quote": that sentence of the request word for word, "closest": true when it wants the listing coming nearest (missing the fewest requirements) offered or bought instead of nothing, "slack": how far above the price of the cheapest such nearest listing it still accepts, as a fraction (a quarter more is 0.25), else 0, "counts": the things the sentence says to count as requirements on top of the product's own (such as stock or the budget), else [], "choose": the words of the sentence that tell how to pick among several equally near listings, other than a price allowance, copied exactly, else ""}.
"roles": an object giving each role the exact name of a listed tool, or null when no tool fits it: "search", "details", "view", "message", "cart", "cart_view", "order", "decline". "message" sends text to the shopper; "cart" adds goods to the cart or removes them; "order" purchases or checks out; "decline" ends the errand without any purchase.
Output: that JSON object, alone.
"""

GO_AHEAD = """Situation: a shopping assistant laid the finished basket before the shopper ahead of buying it. Steps: a) read the reply; b) give ONE JSON object only: {"answer": "change"} if the shopper wants the basket altered in some way (a different product, quantity or property, a fresh price limit); {"answer": "no"} if the purchase is clearly turned down or forbidden; c) {"answer": "yes"} in any other case, since consent, thanks, an acknowledgement or a comment wanting nothing changed keep the basket as it is. Output: that object."""

STEP_CHECK = """Situation: rules from a shopper's request, listed below, bind a shopping assistant. Steps: a) find a rule making the assistant write to the shopper in a separate message not asking about the products (e.g. for the basket's approval before ordering); b) skip the others. Output: ONE JSON object and nothing else: {"steps": [the numbers of those rules]}, an empty list when none does."""

ASK_RANK = """Situation: a shopping assistant chooses properties to raise with a customer. Per product, the customer withholds a wish or two and mentions one only if a question names exactly its property. You get, for each product (P1, P2 ...), the properties that appear on the shop's listings of it, the number of listings showing each, and a handful of sample values.
Steps:
a) Order up to eight properties for each product, the one most likely to conceal an unspoken wish at the top.
b) Properties the request already fixes conceal no wish: drop them. Drop titles, prices and identifiers too.
c) Draw on what the background implies but leaves unsaid: the item's user, the place or manner of its use, the intended look or feel, what went wrong with the previous one. A concrete detail of that kind whose value the request never states marks the property most likely to hide the wish; that property goes first. A detail about how the item looks or what others see of it points at its appearance (colour, pattern, style, finish), not at what it is made of; a detail about how it feels, wears or holds up points at its material or build.
d) Each property pointed at by such a detail is put into "ranked" and into "story" as well. If no such detail exists, "story" remains empty.
e) Without such a detail, put first the property that divides this kind of item into sorts that cannot replace one another in use (what it runs on, how it connects or mounts, its form, its main material, the size it has to match). Properties describing the item itself come before shipping, warranty, condition, pack counts and catch-all fields (options, features, notes). A property whose samples are all different free text is rarely the one.
f) QUESTIONS SO FAR lists what was asked and what came back. Leave out every property asked already. A reply asking for the question again, saying it was not followed, or asking for one thing at a time means the shopper holds no wish on that property. Once a wish has come out for a product, look next at another product that has no wish yet, else at a property of a different aspect.
g) Copy the property names exactly; a property spelled two ways counts once.
Output: ONE JSON object and nothing else: {"P1": {"ranked": ["property", ...], "story": ["property", ...]}, "P2": {...}}"""

HEAR = """Situation: the shopper has just written again, and a shopping assistant must learn what those words mean for the errand.
Steps:
a) Return a single JSON object with the keys below and no text around it:
"meaning": in one brief sentence, what the shopper is after.
"needs": firm product requirements that did not exist before, written "property: constraint" and holding the value, limit, exclusion or closed list exactly as given. When a value made of words, not figures, comes after a comparing phrase (falls short of, nothing less than, at least), those words are the value and keep the shopper's wording: from "falls short of not available" write "property: not available" (some field may truly read "not available" or "none").
"no_preference": every property the shopper called unimportant to them.
"dropped": requirements from before that the shopper abandoned.
"new_budget": a number for the new top total price, in the unit the request's budget uses (when the line's figure is already in that unit, use it as is: a budget counted in cents and "13075 cents" give 13075), else null.
"unavailable": false unless the shopper reports an item can no longer be had (sold out, gone, out of stock, button disabled), then true.
"meta_only": true only for a line that just wants the assistant to put fewer questions at once (a single question per message, less per turn, this one first and the rest afterwards) and states no value: since nothing gets answered, needs, no_preference and dropped take nothing from it, and its figurative wording names no value; otherwise false. Saying the question was not followed or understood, or asking for it again in shorter or simpler words, does not make a line meta_only.
"requirements_now": the complete current set of firm product requirements, each "property: constraint": begin with REQUIREMENTS SO FAR, carry over whatever the shopper neither dropped nor replaced, keep just the newest value of anything replaced, and append each new need the shopper's lines bring. Nothing from the request's story that is absent from REQUIREMENTS SO FAR may be added; only the shopper's lines create requirements. Price, stock, quantity and product category stay out.
"reply": "" unless the shopper puts a question to the assistant or asks for a confirmation; then a brief plain answer to send that gives the substance of the change (for example confirming the new value).
"references": [] unless some need in "needs" takes its value only from another listing or page the shopper points at; for each such need, the listing meant: {"property": which property it concerns, "seller": the shop or brand named in the line for that listing, copied word for word, "kind": what sort of product that listing is, "details": [other values the line states for that listing, brief, in the line's own words], "marks": [the words that listing would itself show for those values, values only and never a property's name], "relation": "same", "differ" (NOT that listing's value: other than, different from, set apart from), "pairs" (its value has to go with or be compatible with that listing's rather than equal it), "within", "at most" or "at least", "tolerance": for "within" the figure, else null, "unit": the unit, else ""}.
b) Several changes may share one line; capture each. A need on some other property, however close (seat height next to overall height, strap width next to width), adds a requirement without replacing any: an earlier requirement leaves or changes only when the line says that about that same property. Every replacement is entered twice, the old value under dropped and the new one under needs. Whichever of two wishes the shopper ranks higher becomes a need. A budget from the shopper belongs in new_budget. Mere wishes ("ideally", "would be nice") never become needs.
c) Read the shopper charitably and literally: plain, figurative or indirect wording that names a value is a firm wish for it; treat a value as excluded only if the shopper openly says to keep away from it. Emphasis words about how strongly a value is held (strictly, only, locked to, nothing but, sticking with, stays, holding out for, nothing short of) are no part of the value: keep only the value words ("material: jute", "color: white"), never "<value> only"; add an exclusion only if the shopper also rules out blends or other values outright. A figure joined to a comparison (under, over, no more than, at least, at most) is a limit and carries its unit. Any new budget, higher or lower, takes over from the old one, even if it is announced as good news.
d) If all the shopper says is that they don't mind, have no preference or anything will do, the asked-about property goes in no_preference (only the FIRST, if the message asked about two). If the answer only says the question lost the shopper ("you've lost me", "could you put it another way", "that was rather dense"), take it as no preference when the question named the product owning the property, and put that property in no_preference too; but with several products wanted and no product named in the question, the property stays open and no_preference stays empty.
e) When there are several products, write each need "property (product): constraint" and take the product from the line (from "toe style of the women's boots" comes "toe style (women's boots): ...").
f) Only the shopper's words make requirements, never the assistant's own question, and sample values inside that question are not needs. A "don't mind" frees just the property the assistant's message asked about, never removing or changing a need given before; and when the message asked nothing (it merely reported), that answer changes nothing. When a need is given by pointing at another listing, its value is the line's pointing wording itself ("finish: unlike the one shown on Y's kettle page"); values the line states for the other listing describe it and not the product, so they are neither needs, nor drops, nor freed properties.
Output: the JSON object by itself."""

READ_RULES = """a) A requirement is decided by the listing field that carries its property, through the words this shop itself writes for the value in it. Spelling is irrelevant if the field's name means that property: separators, a prefix, a clipped or stretched name or a well-known alias are all acceptable (a colour may appear as "color family", a collar type as "neckline", a pants fit type as "fit"). A field named exactly after the property decides, and so does a field named after it with one extra word ahead (material as clothing material, length as pants length): the requirement is met once either of them shows the value. Names differing in separators alone (size class versus size_class) form a single field, and the value in either one meets it. Lacking those, the alias field decides; the title or options list may give the value only if not a single field of that same listing names the property (other listings having such a field does not matter). Descriptions never prove it, save for a property that no listing of the category carries in any field (what model or device the item fits or works with, for instance): there the listing's own description decides, a stated range or pattern of models covering the required one counts (every AB-12xx model covers AB-1234), and a model it names as an exception does not.
b) Values are matched word by word as this shop writes them, case, separators, spelling variants and singular or plural making no difference; a narrower value containing the required one is a match (green from "forest green", cotton from "80% cotton", rtx 3060 from "rtx 3060 12gb"); one list entry suffices, even when the shopper accepts that value only: a field offering a few variants, the value among them, is a match. Shades, related words, abbreviations and synonyms (embroidery as "embroidered", technical wood as "tech wood", off-white as cream or ivory, windows 11 as "win11", plain as "solid") match only so long as no category listing writes the value itself into that field: if "matte" is the finish other listings give, then "satin" is no match for matte. A value joined to another by a plus ("wool + nylon") matches neither if elsewhere the field carries the other one by itself ("nylon"). Any lines titled FIELD EVIDENCE come from this shop's records and state which of these situations applies. A different value for the very same property is a mismatch (maple is not rosewood, mechanical is not quartz), and a word that only looks similar is one as well. A value claiming to suit any value whatever ("suitable for all occasions", "all seasons", "universal") in truth names no value and fails a required one; still, a list naming the required value besides is a match. A field holding some other value is no match merely because the two often come together or one suggests the other (a "zip-up" is not a "pullover" style, "lace-up" is not "slip-on"); the only exceptions are a narrower value including it and a shade usually sold as the required colour.
c) A field about a different property is no evidence (a style is not shown by a length, the material not by a lining material, a strap type not by a strap length): where the listings of the category have a field of the property's own, it cannot take that field's place.
d) Quantities are compared as amounts after unit conversion (45 mm and 4.5 cm are equal); a limit includes its boundary; a range is fine only if each part of it keeps to the limit. When the title has to serve as evidence since the property has no field, a precise required value is met by a title writing its figures and words in the same sequence, extra detail after them allowed (a title with 9' x 12' 4" shows 9' x 12'; one with 9' x 13' does not).
e) Read a price in the unit of the request's budget unless its record states some other currency or unit (with 1000 US cents as the budget, 45 means 45 cents, well within it); rescale neither the budget nor the price.
f) Only the requirements in the list count: a listing never fails on a use, quality, season, condition, occasion, size, gender or age group that none of them names.
g) The kind of product asked for is decided by the category label and nothing else: a listing filed under that category counts as that kind whatever different object its title mentions (an accessory, a refill, a set, a container, a case); judge it only on price, stock and requirements."""

CHECK_ITEM = """Situation: you are the final gate before a shopping assistant settles on an item; the run earns 1 only if that item satisfies every requirement listed below. Decide from the records shown alone.
Steps:
""" + READ_RULES + """
h) Words that belong to the name of the category itself are satisfied by the category.
i) Closed lists and exclusions stay binding.
j) The item must be of the requested kind; if listings show a category label, the right label is whichever matches the whole wording of the request, not a neighbouring or narrower label.
k) The price multiplied by the quantity may not exceed the budget (reaching it exactly is allowed). If the records show the item out of stock, it does not fit.
l) A plain wish ("ideally", "would be nice") flagged as not required counts as no requirement.
m) If records contradict each other (about stock or price, for instance), go by the most recent one.
n) Answer fits=false only for a listed requirement that the item's fields clearly miss by these rules (a FIELD EVIDENCE line stating the item does NOT satisfy a requirement counts as such a miss), or one stated by no field and (where it has no field) by neither the title nor the options list; a missing price or stock record is never a reason to fail the item.
Output: a single JSON object with no other text: {"item": the id of the item, "fits": true or false, "problems": [brief strings, one for each requirement that is not met or not shown], "missed": [the requirements among those problems that a field, the title or the options list states differently, or that a FIELD EVIDENCE line marks as not met; a requirement that no record states at all stays out of this list]}."""

RANK_CHECK = """Situation: a shopping assistant is ready to order an item for a request asking for the PRIORITY; look for any listing that could do better. Below come the requirements, the item picked and the records of the listings seen.
Steps:
a) Judge by the reading rules further down (the price times the quantity may not exceed the budget; the item may not show as out of stock).
b) Name each listing that might satisfy all requirements AND outdo the picked item on the priority (being cheaper when the cheapest is wanted; having a better value of the measure when the best is wanted).
c) Skip listings whose records show a failed requirement, missing stock or a price above the budget. The figures in a ROW's title (say 55 x 94) never tell which dimension each one gives, so they are no reason to drop a ROW.
READING RULES:
""" + READ_RULES + """
Output: a single JSON object with no other text: {"better": [ids, strongest first, 4 at most], "reason": "one sentence"}."""

ASK_PLAN = """Situation: a shopping assistant needs its questions for a shopper chosen. Each hidden need of the shopper is a value of one property of one product, and it surfaces only when a message names exactly that property; a message's first question earns the surest answer, and a question on anything else earns nothing but a polite "don't mind". Choose up to QUESTION_COUNT properties, all different, in order of promise.
Steps:
a) Start from the items under MUST BE LEARNED FROM THE SHOPPER below that nobody has asked yet, and put them in plain words. On the budget, ask for the highest amount they would pay. On a choice between two wishes, state both values along with their properties, e.g. "If one listing can't have both, would you rather have the <value> <property> or the <value> <property>?".
b) After those, properties actually found in the listings' details below. The story comes first (STORY HINTS below, if given, list its details with the property each one touches; they take the lead): the hidden need tends to be the property that the story's concrete details touch on while its value is never said. Whatever the user is going to touch or press, inspect up close or in photos, listen to, measure against or fit to, read on its label, or how it was manufactured, singles out that part or aspect of the item and so its property: tags, labels and the fine print lead to the material, fabric or ingredients; photos and appearance lead to the colour, pattern or the visible part; a space or body part it must fit into leads to that part's shape or size; a present for another person, or for someone scarcely known, leads to whom it is made for (gender, age group); the feel against hand, skin or foot leads first to the material or fabric; the place, time or activity of wearing or using it (outdoors, the office, a gym class) leads to its style or type, meaning the property whose values name such uses; a part fastened or adjusted time after time leads to its closure or mechanism; whatever does the work within it leads to its core component. Weather, a job, a schedule, or any place other than where the item is worn or used is just background. Open with the property the story's details lead to, under the listings' own name for it. Whom the item is made for is one candidate among the others: ask it if the story points to another person, or if nothing else is suggested. Then put properties of the product itself (material, pattern, closure, shape, length, type or style variant, function) before logistics (warranty, condition, pack size, variant or size lists, years).
c) Drop properties already fixed by the request (its requirements, the price, stock, quantity), ones already asked, near-twins that would use up two questions (two kinds of style, one property spelt two ways), and anything about price, stock, delivery, brand, maker or a product or model name. A property qualifies only when it is a key in the listings' details below, and is asked under the listings' name; a STORY HINTS property that no key below matches may be asked under its plain name, except where the hint is marked [no key] (enough listings of that product were read and none had such a field): then ask instead the key below whose values vary in the way the detail describes, or leave it out.
d) A step-b question is a single short, plain, friendly sentence, 15 words at most, asking which <property> they would like for the <item kind> and calling the property what the listings call it (sample values by themselves never reveal a need); it may close with one or two sample values. Leave out any preamble, progress report, instructions or lists, and never ask about a catch-all key like options, variants, a description, a title or a name. Each question covers ONE property of ONE product (apart from the step-a choice between two wishes): never two properties together nor properties of two products, because an answer reveals at most one need and a question with two subjects may go unanswered. Under "second" put the runner-up property of the SAME product (not merely another name for the first), or nothing: the assistant may tack it onto its final message.
e) If the request wants SEVERAL products, any of them may carry the hidden needs: divide the questions so that every product has one before any has two (passing over a product with no property left), draw each product's properties from the list below for its own category, write the property as "property (product)", and mention the product in the question ("Which toe style would you like for the boots?").
Output: just ONE JSON object: {"questions": [{"property": "...", "second": "... or empty", "question": "..."}, ...]}."""

NEXT_ONE = """
f) Only ONE question goes out now; the next is chosen after the shopper's answer has been read. In total REMAINING question(s) are still available and all should be spent: lead with the single best property (one or two spares may follow it); once a need is revealed, the next question turns to a different property and story detail."""

BEFORE_DECLINE = """Situation: a shopping assistant is on the point of telling the shopper that no item of the store fits, which is seldom correct; test it. Compare each listing record below with the requirements by the reading rules that follow (the item may not be shown out of stock, with a price that stays inside the budget).
Steps:
""" + READ_RULES + """
h) A few listings below are known through search rows alone (their details were never read, so nothing is known of their property fields). A row like that inside the requested category (no matter what object its title mentions) whose price, stock and title show no failed requirement has simply not been checked yet (a title's figures, 55 x 94 for instance, never tell which dimension each one gives, so they cannot show that a row fails): put it under "unopened", most promising first (a title naming a required value, or one of the requested kind with nothing speaking against it).
Output: a single JSON object with no other text: {"possible_fits": [ids of listings that look like they meet all requirements, strongest first, 5 at most], "unopened": [ids of listings seen only as rows that should be opened before any decline, 6 at most], "reason": "one sentence"}."""

NEAREST_PICK = """Situation: no listing meets every requirement of a shopper's request, and the request explains which listing to buy then. The listings below all fall short on the same, smallest number of requirements.
Steps:
a) Read the picking part of the request's rule and apply only that to the listings below, using what their records show.
b) If the records cannot settle it, choose the first listing.
Output: one JSON object and nothing else: {"pick": "the id of the chosen listing", "reason": "one short sentence"}."""

REF_HOP = """Situation: in a shopper's request, the product takes its PROPERTY from a different listing in this store, which is described as: DESCRIPTION. Each record below is a listing that may be the one meant.
Steps:
a) Answer with one JSON object, nothing else, containing:
"target": the id of the record that IS the listing described (sold by that seller, of that kind, and showing every detail named, matched by sense: a figure in another unit is fine if the amount is equal), or null if none is.
"missing": the named details that the target lacks ([] when it has all of them).
"go_to": when the target's own text directs the reader to a DIFFERENT listing for the PROPERTY (to take it from, use, consult, look at or follow some other listing), that listing as {"seller": its brand or shop name exactly as the text gives it, "kind": the type of product it is, "details": [every value the text gives for it, brief and as worded: a material, a size with its unit, a colour]}; null otherwise.
"value": with no go_to, the PROPERTY's value as the target's own record gives it.
"goes_with": values the text of the target says belong with, match or are equal to that value ([] when there are none).
b) A listing brought up only as something the target is not ("not the green one from X"), as a colour, size or version sold somewhere else, or in connection with another property, gives no go_to. A record lacking the described details is never the target, whatever its text claims.
Output: the JSON object."""

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
REF_DIFFER_BLIND = 5
SELLER_KEYS = r"shop|seller|store|vendor|brand|make|manufactur|merchant|retailer|supplier|sold.?by"


def as_text(data, ceiling=VIEW_CHARS):
    try:
        flat = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    except (TypeError, ValueError):
        flat = str(data)
    if len(flat) > ceiling:
        return flat[:ceiling] + "...(cut)"
    return flat


def clipped(data, limit, lvl=0):
    if lvl > 8:
        return None
    deeper = lvl + 1
    if isinstance(data, dict):
        return dict((key, clipped(part, limit, deeper)) for key, part in data.items())
    if isinstance(data, list):
        return list(map(lambda part: clipped(part, limit, deeper), data))
    if isinstance(data, str) and len(data) > limit:
        return data[:limit] + "..."
    return data


def _unfenced(raw):
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[A-Za-z]*", "", text).strip()
        if text.endswith("```"):
            text = text[:-3].strip()
    return text


def json_object(raw):
    if not isinstance(raw, str):
        return None
    text = _unfenced(raw)
    candidates = [text]
    if "{" in text:
        candidates.append(text[text.find("{"): text.rfind("}") + 1])
    for candidate in candidates:
        if not candidate:
            continue
        try:
            loaded = json.loads(candidate)
        except ValueError:
            loaded = None
        if isinstance(loaded, dict):
            return loaded
    return None


def is_number(data):
    return not isinstance(data, bool) and isinstance(data, (int, float))


def number_words():
    return {word: at for at, word in enumerate("one two three four five six seven eight nine ten".split(), 1)}


def id_marks(raw):
    return set(re.findall(r"\b[A-Z][A-Za-z]*[ -]?\d{3,5}\b", str(raw or "")))


def ask_name(key):
    return " ".join(re.findall(r"[a-z0-9]+", str(key).replace("_", " ").lower()))


def _menu_skip(values, carriers):
    if all(re.fullmatch(r"[a-z]{0,3}[-_]?[0-9a-f]{8,}", text.lower()) for text in values):
        return True
    wordy = sum(len(text.split()) for text in values) >= 4 * len(values)
    if len(values) == carriers and wordy:
        return True
    return 2 * sum(times for text, times in values.items() if " / " in text) > carriers


def _value_counts(listings):
    tally = {}
    for facts in listings:
        for key in facts:
            counts = tally.setdefault(key, {})
            text = facts[key]
            counts[text] = 1 + counts.get(text, 0)
    return tally


def _menu_row(key, counts, settled, basket):
    label = ask_name(key)
    carriers = sum(counts.values())
    if not label or label in settled or _menu_skip(counts, carriers):
        return None
    weight = (len(counts) > 1, carriers) if basket else (True, carriers * min(len(counts), 4))
    ordered = sorted(counts, key=lambda text: -counts[text])
    return weight, key, carriers, ordered[:ASK_SAMPLES]


def ask_menu(listings, settled, basket):
    candidates = []
    for key, counts in _value_counts(listings).items():
        row = _menu_row(key, counts, settled, basket)
        if row is not None:
            candidates.append(row)
    candidates.sort(key=lambda row: row[0], reverse=True)
    menu = []
    squeezed_seen = set()
    for _, key, carriers, samples in candidates:
        squeezed = ask_name(key).replace(" ", "")
        if squeezed in squeezed_seen:
            continue
        if any(same_values(listings, key, kept[0]) for kept in menu):
            continue
        squeezed_seen.add(squeezed)
        menu.append((key, carriers, samples))
    return menu


def same_values(listings, one, other):
    both = [facts for facts in listings if one in facts and other in facts]
    if len(both) < 2:
        return False
    for facts in both:
        if " ".join(facts[one].lower().split()) != " ".join(facts[other].lower().split()):
            return False
    return True


def ask_merge(first, story, spread):
    by_name = dict((ask_name(key), key) for key in spread)
    picked = [by_name[ask_name(key)] for key in first + story if ask_name(key) in by_name]
    return list(dict.fromkeys(picked + spread))


def written_figures(raw):
    return [float(text.replace(",", "")) for text in re.findall(r"\d[\d,]*(?:\.\d+)?", str(raw or ""))]


def plain_words(raw):
    return re.findall(r"[a-z0-9]+", str(raw or "").lower())


def request_sentences(raw):
    return [bit.strip() for bit in re.split(r"(?<=[.!?;])\s+", raw or "") if bit.strip()]


def matching_sentence(quote, raw, share=0.7):
    wanted = plain_words(quote)
    if not wanted:
        return ""
    best, best_share = "", 0.0
    for sentence in request_sentences(raw):
        present = set(plain_words(sentence))
        portion = sum(1 for word in wanted if word in present) / len(wanted)
        if portion > best_share:
            best, best_share = sentence, portion
    return best if best_share >= share and best_share > 0 else ""


def long_strings(data, collected, lvl=0):
    if lvl > 4:
        return
    if isinstance(data, str):
        if len(data.split()) > 3:
            collected.append(data)
        return
    if isinstance(data, list):
        for part in data[:10]:
            long_strings(part, collected, lvl + 1)


def walk_dicts(data, visit, lvl=0, ceiling=6):
    if lvl > ceiling:
        return
    if isinstance(data, list):
        children = data
    elif isinstance(data, dict):
        visit(data)
        children = [part for part in data.values() if isinstance(part, (dict, list))]
    else:
        return
    for part in children:
        walk_dicts(part, visit, lvl + 1, ceiling)

def _bare_name(name):
    front = re.split(r"\s*\(", str(name))[0]
    return re.sub(r"([a-z])([A-Z])", r"\1 \2", front).lower()


def _one_of(word):
    if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


def junk_key(key):
    spaced = " ".join(re.findall(r"[a-z0-9]+", _bare_name(key)))
    whole = re.fullmatch(r"(?:name|notes?|stock|in stock|on hand|images?|links?|make|maker|shelf|ids?|id)", spaced)
    return bool(whole) or re.search(r"\b(?:brand|shop|seller|store|vendor|manufacturer|category|categories|department|title|headline|"
                                    r"heading|listing|description|overview|price|cost|amount|availability|sku|url)\b", spaced) is not None


def prop_words(name):
    return [_one_of(word) for word in re.findall(r"[a-z]+", _bare_name(name))]


def prop_key(name):
    return "".join(prop_words(name))


def edits_within(a, b, most):
    if abs(len(a) - len(b)) > most:
        return False
    above = list(range(len(b) + 1))
    for row in range(1, len(a) + 1):
        here = [row] * (len(b) + 1)
        for col in range(1, len(b) + 1):
            here[col] = min(above[col] + 1, here[col - 1] + 1, above[col - 1] + (a[row - 1] != b[col - 1]))
        if min(here) > most:
            return False
        above = here
    return above[-1] <= most


def same_prop(a, b):
    one, two = prop_key(a), prop_key(b)
    if not (one and two):
        return False
    least = min(len(one), len(two))
    if one == two:
        return True
    if least >= 4 and one[:3] == two[:3] and edits_within(one, two, 2 if least >= 6 else 1):
        return True
    return least >= 5 and (one.startswith(two) or two.startswith(one))


def _loose_spot(word, at):
    letter = word[at]
    return letter in "aeiouy" or letter in word[max(0, at - 1):at] or letter == word[at + 1:at + 2]


def _shortened(word):
    return [word[:at] + word[at + 1:] for at in range(len(word)) if _loose_spot(word, at)]


def spelled_alike(lo, hi):
    gap = len(hi) - len(lo)
    if gap == 1:
        return lo in _shortened(hi)
    if gap == 2:
        return any(lo in _shortened(once) for once in _shortened(hi))
    if gap:
        return False
    differ = [at for at in range(len(lo)) if lo[at] != hi[at]]
    if len(differ) != 2 or differ[1] != differ[0] + 1:
        return False
    first, second = differ
    return lo[first] == hi[second] and lo[second] == hi[first]


def _one_short_word_more(first, second):
    if len(first) > len(second):
        first, second = second, first
    if len(second) - len(first) != 1 or second[:-1] != first:
        return False
    return len(second[-1]) <= 3 and len("".join(first)) >= 5


def strict_prop(a, b):
    one, two = prop_key(a), prop_key(b)
    if not (one and two):
        return False
    if one == two:
        return True
    short, long_ = sorted((one, two), key=len)
    if len(short) > 3 and short[:3] == long_[:3] and spelled_alike(short, long_):
        return True
    return _one_short_word_more(prop_words(a), prop_words(b))


def ch_words(text):
    found = re.findall(r"[a-z0-9]+(?:\.[0-9]+)?", str(text or "").lower())
    return list(filter(lambda token: token not in CH_SKIP, found))


def ch_word_in(word, pool):
    if word in pool:
        return True
    if not word.isalpha():
        return False
    for seen in filter(str.isalpha, pool):
        if same_kind(word, seen):
            return True
        if min(len(word), len(seen)) >= 6 and word[:2] == seen[:2] and edits_within(word, seen, 1):
            return True
    return False


def ch_alts(value):
    text = re.sub(r"\((?:stated|revealed[^)]*)\)", " ", str(value or "").lower())
    text = re.sub(r"\b(?:only|just|exactly|strictly)\b", " ", text.replace('"', " "))
    return [option for option in map(ch_words, re.split(r"\bor\b", text)) if option]


def ch_writes(value, text):
    return ch_match(ch_alts(value), ch_words(text))


def _glued_somewhere(pool, target):
    for begin in range(len(pool)):
        built, at = "", begin
        while at < len(pool):
            built += pool[at]
            at += 1
            if built == target:
                return True
            if len(built) >= len(target):
                break
    return False


def ch_match(alts, pool):
    if not pool:
        return False
    memo = []

    def pool_has_digits():
        if not memo:
            memo.append(any(re.search(r"\d", word) for word in pool))
        return memo[0]

    for option in alts:
        if all(ch_word_in(word, pool) for word in option):
            return True
        joined = "".join(option)
        wordy = len(option) < 2 and re.search(r"\d", joined) is None
        if wordy and not pool_has_digits():
            continue
        if _glued_somewhere(pool, joined):
            return True
    return False


def _word_touches(word, pool, lowered):
    if ch_word_in(word, pool):
        return True
    if re.fullmatch(r"\d+(?:\.\d+)?", word) and re.search(r"(?<![\d.])" + re.escape(word) + r"(?!\d|\.\d)", lowered):
        return True
    shares_head = lambda seen: len(seen) >= 3 and (seen.startswith(word[:3]) or word.startswith(seen[:3]))
    return len(word) >= 3 and any(map(shares_head, pool))


def ch_touches(value, text):
    pool = ch_words(text)
    lowered = str(text or "").lower()
    return any(_word_touches(word, pool, lowered) for option in ch_alts(value) for word in option)


def ch_parts(text):
    lowered = str(text or "").strip().lower()
    bits = list(filter(None, (raw.strip() for raw in re.split(r"[,;|]", lowered))))
    if not bits and lowered:
        bits = [lowered]
    return bits


def ch_sides(part):
    sides = list(filter(None, (raw.strip() for raw in re.split(r"\s*\+\s*", str(part)))))
    return sides if len(sides) > 1 else []


def figure_written(fig, said):
    pattern = r"(?<![\d.])" + re.escape(fig) + r"(?!\d|\.\d)"
    if "." in fig:
        return re.search(pattern, said) is not None
    for found in re.finditer(pattern, said):
        unit = re.match(r"\s*([a-z]+)", said[found.end():])
        if unit is None or len(unit.group(1)) < 4:
            return True
    return False

def _meaning_words(text):
    return [word for word in re.findall(r"[a-z]{3,}", text) if word not in LINK_WORDS]


def _cleared_bounds(text):
    stripped = re.sub(r"\b(?:no|not)\s+[a-z]+er\b(?=\s+than\b)", " ", re.sub(BOUND_WORDS, " ", text))
    if re.search(r"\d", stripped) is None:
        return stripped
    return re.sub(r"\b(?:no|not)\s+[a-z]+er\b", " ", stripped)


def _plainly_positive(text):
    if re.fullmatch(r"\W*(?:yes|no|true|false|none|required|included|needed)\W*", text):
        return True
    if LITERAL_VALUES:
        unmarked = re.sub(r"\((?:stated|revealed[^)]*)\)", " ", text)
        return " ".join(re.findall(r"[a-z0-9]+", unmarked)) in LITERAL_VALUES
    return False


def _exclusion_is_own(after, name):
    named = set(prop_words(name))
    named.update(re.findall(r"[a-z]+", str(name).lower()))
    if not (after and named):
        return False
    return all(word in named or word.rstrip("s") in named for word in after)


def _has_long_clause(text):
    linked = re.sub(r"\b(?:" + VALUE_LINKS + r")\b", " ", text)
    return any(len(_meaning_words(clause)) >= 4 for clause in re.split(r",|;|/|&|\(|\)|\bor\b|\band\b", linked))


def positive_value(value, name=""):
    text = str(value).lower().strip()
    if _plainly_positive(text):
        return True
    text = _cleared_bounds(text)
    if _has_long_clause(text):
        return False
    found = re.search(EXCLUDE_WORDS, text)
    if found is None or _meaning_words(text[:found.start()]):
        return True
    return _exclusion_is_own(_meaning_words(text[found.end():]), name)


def _figure_groups(body):
    chains = re.findall(r"\d+(?:\.\d+)?(?:\s*[:x/]\s*\d+(?:\.\d+)?)*", body)
    return [re.findall(r"\d+(?:\.\d+)?", chain) for chain in chains]


def _value_patterns(body):
    for word in re.findall(r"[a-z]{3,}", body):
        if word not in LINK_WORDS:
            yield r"\b" + re.escape(word[:-1] if len(word) > 4 else word)
    for group in _figure_groups(body):
        yield r"(?<![\d.])" + r"\D{0,4}".join(map(re.escape, group)) + r"(?![\d.])"


def value_seen(value, samples):
    haystack = str(samples or "").lower()
    if not haystack.strip():
        return False
    body = re.sub(r"\b(?:" + VALUE_LINKS + r")\b", " ", str(value).lower())
    return any(re.search(pattern, haystack) for pattern in _value_patterns(body))


def kind_words(text):
    return set(word.rstrip("s") for word in _meaning_words(str(text).lower()))


def kind_tokens(text):
    lowered = str(text).lower().replace("’", "'")
    return _meaning_words(re.sub(r"'s\b", "", lowered))


_PLURAL_SWAPS = (("y", 1, "ies"), ("fe", 2, "ves"), ("ies", 3, "y"), ("ves", 3, "f"), ("ves", 3, "fe"), ("es", 2, ""), ("s", 1, ""))


def plural_forms(word):
    stem = str(word)
    spellings = {stem, stem + "s", stem + "es"}
    for ending, cut, tail in _PLURAL_SWAPS:
        if stem.endswith(ending):
            spellings.add(stem[:-cut] + tail)
    if stem.endswith("f") and not stem.endswith("fe"):
        spellings.add(stem[:-1] + "ves")
    return set(filter(lambda form: len(form) > 2, spellings))


def same_kind(a, b):
    if a == b:
        return True
    return str(b) in plural_forms(a) or str(a) in plural_forms(b)


def kind_in(word, pool):
    for seen in pool:
        if same_kind(word, seen):
            return True
    return False


def kinds_within(words, pool):
    return not any(not kind_in(word, pool) for word in words)


def kind_score(own, words, shared):
    return sum((0.5 if kind_in(word, shared) else 1.0) for word in own if kind_in(word, words))


def _word_regex(token):
    forms = sorted(plural_forms(token), key=lambda form: (-len(form), form))
    return r"(?:" + "|".join(re.escape(form) for form in forms) + r")(?:'s\b|'|\b)"


def name_pattern(name):
    return r"[\s'-]+".join(map(_word_regex, kind_tokens(name)))


def one_prop_hint(hint):
    try:
        if ":" not in hint:
            return hint
        front, rest = hint.split(":", 1)
        if "/" not in front:
            return hint
        tail = re.search(r"\(([^()]*)\)\s*$", front)
        names = front[:tail.start()] if tail else front
        choices = list(filter(None, map(str.strip, names.split("/"))))
        if len(choices) < 2:
            return hint
        owner = f" ({tail.group(1).strip()})" if tail else ""
        others = ", ".join(choices[1:])
        return choices[0] + owner + ":" + rest + f" (or {others})"
    except Exception:
        return hint


def _opening_word(detail):
    joiners = "|".join(sorted(LINK_WORDS))
    found = re.match(r"\s*([a-z][a-z-]*[a-z])(?:\s*[,;.()]|\s*$|\s+(?:" + joiners + r")\b)", str(detail).lower())
    return "" if found is None else found.group(1)


def _comes_before(word, request, noun):
    spoken = re.findall(r"[a-z][a-z'-]*", str(request).lower())
    for at in range(len(spoken) - 1):
        if spoken[at] == word and spoken[at + 1].split("'")[0].rstrip("s") == noun:
            return True
    return False


def lead_value(detail, request, product):
    try:
        word = _opening_word(detail)
        if not word:
            return ""
        product_words = _meaning_words(str(product).lower())
        if not product_words or len(word) < 3 or word in LINK_WORDS:
            return ""
        stems = set(kind.rstrip("s") for kind in product_words)
        if word.rstrip("s") in stems or not _comes_before(word, request, product_words[0].rstrip("s")):
            return ""
        return word
    except Exception:
        return ""


def _stem_of(word):
    for ending, shortest in (("es", 5), ("s", 4)):
        if word.endswith(ending) and len(word) >= shortest:
            return word[:-len(ending)]
    return word


def _close_said(word, stem, said):
    grown_stem = said.startswith(stem) and len(said) <= len(stem) + 3
    cut_stem = stem.startswith(said) and len(said) >= max(4, len(stem) - 2)
    if said == word or grown_stem or cut_stem:
        return True
    if len(word) < 4:
        return False
    if said.startswith(word):
        return True
    near = len(said) >= 4 and said[:1] == word[:1]
    return near and edits_within(said, word, 1)


def _word_heard(word, said_words, said_stems):
    stem = _stem_of(word)
    for said in said_words:
        if _close_said(word, stem, said):
            return True
    return any(len(said) >= 3 and stem.startswith(said) and len(said) >= len(stem) - 3 for said in said_stems)


def _requirement_body(req):
    front, _, rest = req.partition(":")
    text = rest.strip().lower()
    if re.fullmatch(r"\W*(?:yes|no|true|false|required|included|needed)\W*", text):
        text = re.split(r"\s*\(", front)[0].lower()
    return re.sub(r"\b(?:" + VALUE_LINKS + r")\b", " ", text)


def _runs_not_said(body, said):
    missing = []
    for chain in re.findall(r"\d+(?:\.\d+)?(?:\s*[:x/]\s*\d+(?:\.\d+)?)*", body):
        figs = re.findall(r"\d+(?:\.\d+)?", chain)
        if len(figs) < 2:
            continue
        spaced = r"\D{0,4}".join(map(re.escape, figs))
        if not re.search(r"(?<![\d.])" + spaced + r"(?!\d)", said):
            missing.append(chain)
    return missing


def _story_skipped(req, refs):
    tail = req.partition(":")[2].strip().lower()
    if tail and req not in refs:
        return re.search(r"as shown|same as|the page|listing|sold by|redirect|refer", tail) is not None
    return True


def _figures_all_said(body, said):
    numbers = re.findall(r"\d+(?:\.\d+)?", body)
    return bool(numbers) and all(figure_written(number, said) for number in numbers)


def story_guess(req, request, refs):
    text = str(req)
    if _story_skipped(text, refs):
        return []
    body = _requirement_body(text)
    said = str(request).lower()
    if _figures_all_said(body, said):
        return []
    vocabulary = set(re.findall(r"[a-z]+", said))
    trimmed = {word[:-1] if word.endswith("s") and len(word) > 3 else word for word in vocabulary}
    content = _meaning_words(body)
    heard, unheard = [], []
    for word in content:
        (heard if _word_heard(word, vocabulary, trimmed) else unheard).append(word)
    if len(content) < 3 and any(len(word) > 3 for word in heard):
        return []
    return unheard + _runs_not_said(body, said)


def ref_text(value):
    return re.search(REF_WORDS, str(value or "").lower()) is not None


def _unplural(token):
    if re.search(r"(?:ch|sh|x|z)es$", token):
        return token[:-2]
    return token[:-1]


def _glue_unit(found):
    unit = found.group(2)
    if unit in LINK_WORDS:
        return found.group(0)
    return found.group(1) + unit


def _ref_token(token):
    if token in LINK_WORDS or (len(token) == 1 and not token.isdigit()):
        return None
    plural = len(token) > 3 and token.endswith("s") and not token.endswith("ss") and not token[-2].isdigit()
    return _unplural(token) if plural else token


def ref_toks(text):
    lowered = re.sub(r"(\d)(?:\s+|\s*-\s*)([a-z]{1,8})\b", _glue_unit, str(text or "").lower())
    picked = map(_ref_token, re.findall(r"\d+(?:[.,:/]\d+)*[a-z]*|[a-z][a-z0-9]*", lowered))
    return [token for token in picked if token is not None]


def _figure_units(flat):
    for hit in re.finditer(r"(\d+(?:\.\d+)?(?:\s*/\s*\d+(?:\.\d+)?)*)\s*-?\s*([a-z]{1,8})\b", flat):
        unit = hit.group(2)
        if len(unit) >= 3 and unit.endswith("s") and not unit.endswith("ss"):
            unit = _unplural(unit)
        if unit in LINK_WORDS:
            continue
        for fig in re.split(r"\s*/\s*", hit.group(1)):
            yield float(fig), unit


def ref_figs(text):
    collected = set()
    try:
        collected.update(_figure_units(str(text or "").lower().replace(",", "")))
    except Exception:
        pass
    return collected


def ref_name(text):
    lowered = str(text or "").lower()
    return re.sub(r"[^a-z0-9]", "", re.sub(r"'s\b", "", lowered))


_BOUND_RELATIONS = ((r"\b(?:up to|no more than|at most|or lower|or less|or smaller)\b", "at most"),
                    (r"\b(?:at least|no less than|or more|or higher|or larger)\b", "at least"))


def ref_relation(value):
    text = str(value or "").lower()
    if re.search(DIFFER_WORDS, text):
        return "differ", None, ""
    near = re.search(r"\bwithin\s+(\d+(?:\.\d+)?)\s*([a-z%]+)?", text)
    if near is not None:
        return "within", float(near.group(1)), (near.group(2) or "").strip()
    for pattern, relation in _BOUND_RELATIONS:
        if re.search(pattern, text):
            return relation, None, ""
    return "same", None, ""


def _seller_word(body, ok):
    hits = re.finditer(r"(?<![A-Za-z'])([A-Z][a-z]{3,})(?:'s)?(?![A-Za-z])", body)
    return next((found for found in hits if found.start() > 0 and (ok is None or ok(found.group(1)))), None)


def _ref_details(rest):
    pieces = re.split(r",|;|\band\b|\bwith\b|\bthat\s+(?:is|are|has|have|comes?(?:\s+as)?)\b|\bwhich\b|\bin\b|\bat\b", rest, flags=re.I)
    details = []
    for piece in pieces:
        kept = [word for word in re.findall(r"[A-Za-z0-9][A-Za-z0-9.%:/&'-]*", piece) if word.lower() not in LINK_WORDS]
        if kept:
            details.append(" ".join(kept))
    return details[:6]


def _ref_kind(before, rest):
    sold = re.search(r".*(?:\bthe|\bfor|\ba|\ban)\s+(.*?)\s+(?:sold|made|offered|listed)\s+by\s*$", before, re.I | re.S)
    if sold:
        return sold.group(1), rest
    lead = re.match(r"\s*([^,;(]*?)(?=\s+(?:with|that|which|in|at|having|from|of|for|sold)\b|[,;]|$)", rest, re.I)
    if lead:
        return lead.group(1), rest[lead.end():]
    return "", rest


def ref_parse(text, ok=None):
    try:
        body = re.split(r"\(\s*or\b", str(text or ""))[0]
        seller = _seller_word(body, ok)
        if seller is None:
            return None
        kind, rest = _ref_kind(body[:seller.start()], body[seller.end():])
        return {"seller": seller.group(1), "kind": kind.strip()[:80], "details": _ref_details(rest)}
    except Exception:
        return None

def _names_match(seen, word, exact):
    if seen == word or exact:
        return seen == word
    return any(len(stem) >= 3 and other in (stem + "s", stem + "es") for stem, other in ((word, seen), (seen, word)))


def prop_named(prop, text, exact=False):
    wanted = re.findall(r"[a-z0-9]+", _bare_name(prop or ""))
    if not wanted or len(" ".join(wanted)) < 3:
        return False
    tokens = re.findall(r"[a-z0-9]+", str(text or "").lower())
    width = len(wanted)
    start = 0
    while start + width <= len(tokens):
        if all(map(lambda seen, word: _names_match(seen, word, exact), tokens[start:start + width], wanted)):
            return True
        start += 1
    return False


def _span_pattern(prop):
    head = re.split(r"\s*\(", str(prop))[0].lower()
    words = [word for word in re.findall(r"[a-z0-9]+", head) if word not in LINK_WORDS]
    return r"\b" + r"\W+".join(map(re.escape, words)) + r"\b" if words else None


def ref_span(prop, source):
    try:
        pattern = _span_pattern(prop)
        found = None if pattern is None else re.search(pattern, str(source).lower())
        if found is None:
            return ""
        tail = str(source)[found.end():][:500]
        return re.split(r"\(\s*or\b|(?<=[.!?])\s", tail)[0]
    except Exception:
        return ""


def _only_key(lowered, record, test):
    hits = [key for low, key in lowered.items() if test(low, record[key])]
    if len(hits) != 1:
        return None
    return hits[0]


def _price_like(low, val):
    if is_number(val) and re.search(r"amount|fee|charge|pay|rate|value", low):
        return re.search(r"total|count|qty|quant|unit|due|sum|rating|star|score|page|number|num|size|weight|width|height|length|found|match", low) is None
    return False


def _stock_like(low, val):
    return isinstance(val, bool) and re.search(r"hand|orderable|buyable|sellable|ready|ship|instock|purchas|in_store", low) is not None


def _title_like(low, val):
    if isinstance(val, str) and re.search(r"headline|name|label|caption", low):
        return re.search(r"shop|seller|store|vendor|brand|make|categor|shelf|department|id", low) is None
    return False


def _has_price(lowered, record):
    for low, key in lowered.items():
        if re.search(r"price|cost", low) and is_number(record[key]):
            return True
    return False


def _has_stock(lowered, record):
    return any(re.search(r"stock|avail", low) for low in lowered)


def _has_title(lowered, record):
    return any(re.search(r"title|heading", low) for low in lowered)


_PLAIN_FIELDS = (("price", _has_price, _price_like), ("in_stock", _has_stock, _stock_like), ("title", _has_title, _title_like))


def _carries_id(record):
    for key, val in record.items():
        if isinstance(val, str) and "id" in str(key).lower() and re.fullmatch(ID_LIKE, val):
            return True
    return False


def _add_plain_names(record):
    if not _carries_id(record):
        return
    lowered = dict((str(key).lower(), key) for key in record)
    for field, present, fits in _PLAIN_FIELDS:
        source = None if present(lowered, record) else _only_key(lowered, record, fits)
        if source is not None:
            record[field] = record[source]


def lensed(data):
    try:
        copy = json.loads(json.dumps(data))
    except (TypeError, ValueError):
        return data
    walk_dicts(copy, _add_plain_names)
    return copy


def _is_whole_float(value):
    return isinstance(value, float) and value.is_integer()


_NO_CHANGE = object()


def _to_integer(value):
    if _is_whole_float(value):
        return int(value)
    if isinstance(value, str) and re.fullmatch(r"\s*-?\d+(\.0+)?\s*", value):
        return int(float(value))
    return _NO_CHANGE


def _to_number(value):
    if not isinstance(value, str) or not re.fullmatch(r"\s*-?\d+(\.\d+)?\s*", value):
        return _NO_CHANGE
    figure = float(value)
    return int(figure) if figure.is_integer() else figure


def _to_string(value):
    return str(value) if is_number(value) else _NO_CHANGE


def _to_boolean(value):
    if isinstance(value, str) and value.lower() in ("true", "false"):
        return value.lower() == "true"
    return _NO_CHANGE


_CONVERTERS = {"integer": _to_integer, "number": _to_number, "string": _to_string, "boolean": _to_boolean}


def _converted(value, kind):
    convert = _CONVERTERS.get(kind)
    return _NO_CHANGE if convert is None else convert(value)


def _enum_rule(value, schema, where):
    options = schema["enum"]
    if value in options:
        return value, []
    folded = value.strip().lower() if isinstance(value, str) else None
    if folded is not None:
        for option in options:
            if isinstance(option, str) and option.lower() == folded:
                return option, []
    return value, [f"{where}: must be one of {options}"]


def _const_rule(value, schema, where):
    fixed = schema["const"]
    if value == fixed:
        return value, []
    return value, [f"{where}: must equal {fixed!r}"]


def _min_length_rule(value, schema, where):
    least = schema["minLength"]
    if isinstance(value, str) and len(value) < least:
        return value, [f"{where}: shorter than {least} characters"]
    return value, []


def _max_length_rule(value, schema, where):
    if isinstance(value, str):
        value = value[: schema["maxLength"]]
    return value, []


def _bound_rule(keyword, crosses, wording):
    def judge(value, schema, where):
        edge = schema[keyword]
        broken = is_number(value) and is_number(edge) and crosses(value, edge)
        return value, ([f"{where}: {wording} {edge}"] if broken else [])
    return judge


def _count_rule(keyword, too_many, wording):
    def judge(value, schema, where):
        limit = schema[keyword]
        if isinstance(value, list) and (len(value) > limit if too_many else len(value) < limit):
            return value, [f"{where}: {wording} {limit} items"]
        return value, []
    return judge


def _items_rule(value, schema, where):
    inner = schema["items"]
    if not (isinstance(value, list) and isinstance(inner, dict)):
        return value, []
    kept, issues = [], []
    for pos, element in enumerate(value):
        element, found = ArgCheck.execute(element, inner, f"{where}[{pos}]")
        kept.append(element)
        issues.extend(found)
    return kept, issues


def _object_rule(value, schema, where):
    if not isinstance(value, dict):
        return value, []
    declared = schema.get("properties") or {}
    missing = [name for name in schema.get("required") or [] if name not in value]
    issues = [f"{where}.{name}: required" for name in missing]
    if schema.get("additionalProperties") is False:
        issues += [f"{where}.{name}: not an allowed argument" for name in value if name not in declared]
    checked = {name: ArgCheck.execute(element, declared[name], f"{where}.{name}") if name in declared else (element, [])
               for name, element in value.items()}
    issues += [problem for _, found in checked.values() for problem in found]
    return {name: pair[0] for name, pair in checked.items()}, issues


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

    RULES = (
        ("enum", _enum_rule),
        ("const", _const_rule),
        ("minLength", _min_length_rule),
        ("maxLength", _max_length_rule),
        ("minimum", _bound_rule("minimum", lambda got, edge: got < edge, "below minimum")),
        ("maximum", _bound_rule("maximum", lambda got, edge: got > edge, "above maximum")),
        ("exclusiveMinimum", _bound_rule("exclusiveMinimum", lambda got, edge: got <= edge, "must be above")),
        ("exclusiveMaximum", _bound_rule("exclusiveMaximum", lambda got, edge: got >= edge, "must be below")),
        ("minItems", _count_rule("minItems", False, "needs at least")),
        ("maxItems", _count_rule("maxItems", True, "allows at most")),
        ("items", _items_rule),
        ("type", _object_rule),
    )

    @classmethod
    def matches(cls, value, kinds):
        return any(cls.TYPES.get(kind) is None or cls.TYPES[kind](value) for kind in kinds)

    @classmethod
    def adjusted(cls, value, kinds):
        if not kinds:
            return value
        if cls.matches(value, kinds):
            if "integer" in kinds and _is_whole_float(value):
                return int(value)
            return value
        for kind in kinds:
            changed = _converted(value, kind)
            if changed is not _NO_CHANGE:
                return changed
        return value

    @classmethod
    def execute(cls, piece, jschema, where="args"):
        if not isinstance(jschema, dict):
            return piece, []
        declared = jschema.get("type")
        kinds = [declared] if isinstance(declared, str) else list(declared or [])
        value = cls.adjusted(piece, kinds)
        if kinds and not cls.matches(value, kinds):
            return value, [f"{where}: expected {'/'.join(kinds)}"]
        pending = [rule for keyword, rule in cls.RULES if keyword in jschema]
        problems = []
        while pending:
            value, found = pending.pop(0)(value, jschema, where)
            problems.extend(found)
        return value, problems


class Knowledge:
    SHELF_KEY = r"categor|shelf|department|aisle"

    def __init__(self):
        self.items = []
        self.ids = set()
        self.read = set()
        self.label_of = {}

    @classmethod
    def shelf_name(cls, row):
        for field, value in row.items():
            lowered = str(field).lower()
            if "id" in lowered or not isinstance(value, str):
                continue
            trimmed = value.strip()
            if trimmed and len(trimmed) <= 60 and re.search(cls.SHELF_KEY, lowered):
                return trimmed
        return ""

    @staticmethod
    def row_ident(row):
        for field, value in row.items():
            if isinstance(value, str):
                lowered = str(field).lower()
                if lowered == "id" or lowered == "_id":
                    return value
                if lowered.endswith("id") and re.fullmatch(ID_LIKE, value):
                    return value
        return None

    @staticmethod
    def row_price(row):
        for field, value in row.items():
            if isinstance(value, bool) or not is_number(value):
                continue
            if re.search(r"price|cost", str(field).lower()):
                return float(value)
        return None

    def opened_ident(self, row):
        for field, value in row.items():
            if isinstance(value, str) and "id" in str(field).lower() and value in self.read:
                return value
        return None

    def remember_row(self, row):
        for field, value in row.items():
            if not isinstance(value, str) or "id" not in str(field).lower():
                continue
            if re.fullmatch(ID_LIKE, value):
                self.ids.add(value)
        ident = self.row_ident(row)
        if not ident:
            return
        for inner in row.values():
            if not isinstance(inner, dict):
                continue
            if len([v for v in inner.values() if isinstance(v, (str, int, float))]) >= 2:
                self.read.add(ident)
                break
        shelf = self.shelf_name(row)
        if shelf:
            self.label_of.setdefault(ident, shelf)

    def add(self, obs):
        self.items.append(obs)
        walk_dicts(obs, self.remember_row)

    def visit_all(self, visit):
        for stored in self.items:
            walk_dicts(stored, visit)

    def price(self, thing):
        last = [None]

        def spot(row):
            names_it = False
            for field, value in row.items():
                if isinstance(value, str) and value == thing and "id" in str(field).lower():
                    names_it = True
                    break
            if names_it:
                figure = self.row_price(row)
                if figure is not None:
                    last[0] = figure
        self.visit_all(spot)
        return last[0]

    def collect_mentions(self, piece, thing, bag, level):
        if level > 6 or len(bag) >= 8:
            return
        if isinstance(piece, list):
            nested = piece
        elif isinstance(piece, dict):
            if thing in [v for v in piece.values() if isinstance(v, str)]:
                snippet = as_text(clipped(piece, 400))[:2500]
                if snippet not in bag:
                    bag.append(snippet)
                return
            nested = list(piece.values())
        else:
            return
        for inner in nested:
            self.collect_mentions(inner, thing, bag, level + 1)

    def about(self, thing, hold=4):
        bag = []
        for stored in self.items:
            self.collect_mentions(stored, thing, bag, 0)
        return bag[-hold:]

    def merge_cards(self):
        joined = {}
        seen_order = []

        def fold(row):
            ident = None
            for field, value in row.items():
                if (isinstance(value, str) and "id" in str(field).lower() and re.fullmatch(ID_LIKE, value)
                        and value in self.ids):
                    ident = value
                    break
            if not ident:
                return
            has_map = False
            for value in row.values():
                if isinstance(value, dict) and len(value) >= 2:
                    has_map = True
                    break
            if not has_map and len(row) < 4:
                return
            if ident not in seen_order:
                seen_order.append(ident)
            joined.setdefault(ident, {}).update(dict(row.items()))
            if has_map:
                self.read.add(ident)
        self.visit_all(fold)
        return joined, seen_order

    def cards(self, char_cap=30000, cheaper_than="", skip=None, first=()):
        joined, seen_order = self.merge_cards()
        price_of = lambda ident: self.row_price(joined.get(ident) or {})
        if skip:
            seen_order = [i for i in seen_order if i == cheaper_than or i not in skip]
        limit = price_of(cheaper_than) if cheaper_than else None
        if limit is not None:
            seen_order = [i for i in seen_order
                          if i == cheaper_than or price_of(i) is None or price_of(i) < limit]
        place = dict(zip(seen_order, range(len(seen_order))))
        leaders = set(first or ())

        def rank(ident, with_stock=False):
            key = [ident not in leaders, ident not in self.read]
            if with_stock:
                key.append(any(Lookout.key_stock(f, v) is False for f, v in joined[ident].items()))
            key.append(place[ident])
            return tuple(key)

        order = sorted(seen_order, key=rank)
        lines = dict((ident, ("DETAILS " if ident in self.read else "ROW ") + as_text(clipped(joined[ident], 160)))
                     for ident in order)
        if sum(len(t) for t in lines.values()) > char_cap:
            order = sorted(order, key=lambda ident: rank(ident, True))
        picked, total = [], 0
        for ident in order:
            text = lines[ident]
            if total + len(text) <= char_cap:
                picked.append(text)
                total += len(text)
        return picked

    def shelf_for(self, row):
        own = self.shelf_name(row)
        if own:
            return own
        mapping = self.label_of or {}
        for field, value in row.items():
            if isinstance(value, str) and "id" in str(field).lower() and value in mapping:
                return mapping[value]
        return ""

    def rich_reads(self, shelf=""):
        target = shelf.strip().lower()
        rich = set()

        def weigh(row):
            ident = self.opened_ident(row)
            if not ident:
                return
            if shelf and self.shelf_for(row).strip().lower() != target:
                return
            nested = [value for value in row.values() if isinstance(value, dict)]
            if not nested:
                shorts = [value for value in row.values() if isinstance(value, str) and 0 < len(value) <= 70]
                if len(shorts) >= 5:
                    rich.add(ident)
            elif any(len(value) >= 3 for value in nested):
                rich.add(ident)
        self.visit_all(weigh)
        return len(rich)

    def shelves_read(self):
        tally = {}

        def count(row):
            if not self.opened_ident(row):
                return
            shelf = self.shelf_for(row)
            if not shelf:
                return
            if any(isinstance(value, dict) for value in row.values()):
                tally[shelf] = tally.get(shelf, 0) + 1
        self.visit_all(count)
        return sorted(tally, key=lambda shelf: -tally[shelf])

    def scan_shelf(self, piece, shelf, within, level, take):
        if level > 6:
            return
        if isinstance(piece, dict):
            within = within or not shelf or self.sits_on(piece, shelf)
            if within:
                take(piece)
            for inner in piece.values():
                if isinstance(inner, (dict, list)):
                    self.scan_shelf(inner, shelf, within, level + 1, take)
        elif isinstance(piece, list):
            for inner in piece:
                self.scan_shelf(inner, shelf, within, level + 1, take)

    def sits_on(self, row, shelf):
        wanted = shelf.lower()
        for field, value in row.items():
            if not isinstance(value, str) or value.strip().lower() != wanted:
                continue
            if re.search(self.SHELF_KEY, str(field).lower()):
                return True
        return self.shelf_for(row).strip().lower() == shelf.strip().lower()

    @staticmethod
    def short_fields(row):
        return [(field, value) for field, value in row.items()
                if isinstance(value, str) and 0 < len(value) <= 70 and not junk_key(field)]

    def fact_sets(self, shelf=""):
        found, thin = [], []

        def take(row):
            facts = {str(field): value for field, value in self.short_fields(row)}
            if facts in found or facts in thin:
                return
            if len(facts) >= 5:
                found.append(facts)
            elif len(facts) >= 2 and not any(isinstance(value, (dict, list)) for value in row.values()) \
                    and not any(str(value) in self.ids for value in facts.values()):
                thin.append(facts)
        for stored in self.items:
            self.scan_shelf(stored, shelf, False, 0, take)
        return found if len(found) >= 2 else found + thin

    @staticmethod
    def code_like(value):
        return ("-" in value or "_" in value) and bool(re.search(r"\d", value)) and bool(re.fullmatch(CODE_LIKE, value))

    def prop_menu(self, shelf=""):
        everywhere, in_rows, examples = {}, {}, {}
        seen = dict(sets=0, mapped=0)

        def take(row):
            facts = self.short_fields(row)
            if len(facts) < 5:
                flat = shelf and len(facts) >= 2 and not any(isinstance(v, dict) for v in row.values())
                if not flat:
                    return
                as_row = any(isinstance(v, str) and "id" in str(k).lower() for k, v in row.items())
            else:
                as_row = False
            seen["sets"] += 1
            seen["mapped"] += 0 if as_row else 1
            for field, value in facts:
                if self.code_like(value):
                    continue
                name = re.sub(r"[_.]+", " ", str(field)).strip().lower()
                everywhere[name] = everywhere.get(name, 0) + 1
                if as_row:
                    in_rows[name] = in_rows.get(name, 0) + 1
                shown = examples.setdefault(name, [])
                if len(shown) < 5 and value.lower() not in shown:
                    shown.append(value.lower())

        for stored in self.items:
            self.scan_shelf(stored, shelf, False, 0, take)
        if seen["mapped"]:
            everywhere = {name: n for name, n in everywhere.items() if in_rows.get(name, 0) < n}
        top = sorted(everywhere, key=lambda name: -everywhere[name])[:28]
        listing = "\n".join("- %s (in %d listings): %s" % (name, everywhere[name], " | ".join(examples.get(name, [])))
                            for name in top)
        return listing, seen["sets"]

    def label_lines(self):
        per_key = {}

        def count(row):
            if len(row) < 4:
                return
            for field, value in row.items():
                if not isinstance(value, str) or not value or len(value) > 40:
                    continue
                if re.search(r"\d", value) and re.fullmatch(CODE_LIKE, value):
                    continue
                bucket = per_key.setdefault(str(field), {})
                bucket[value] = bucket.get(value, 0) + 1
        self.visit_all(count)
        rows = []
        for field, bucket in per_key.items():
            if sum(bucket.values()) >= 4 and 2 <= len(bucket) <= 8 and max(bucket.values()) >= 2:
                ranked = sorted(bucket.items(), key=lambda pair: -pair[1])
                rows.append(f"- {field}: " + " | ".join(f"{value} (x{times})" for value, times in ranked))
        return "\n".join(rows[:12])


class Wants:
    def __init__(self, ep):
        self.ep = ep
        self.request = ep.request
        self.reading = {}
        self.written = []
        self.uncovered = []
        self.indifferent = []
        self.let_go = []
        self.now_reqs = None
        self.spend_cap = None
        self.version = 0
        self.caught = set()
        self.exchanges = []
        self.reply_due = ""
        self.widened_at = -1
        self.widened_tick = -10 ** 6
        self.sweepable = False
        self.resolved = []
        self.ref_news = []
        self.pending = None
        self.pending_at = 0.0
        self.pinned_heard = 0
        self.meta_last = None
        self.fresh_needs = None
        self.field_hook = None
        self.ref_hook = None

    def note(self, entry):
        self.ep.log.append(entry)

    @staticmethod
    def counted(value):
        if isinstance(value, bool) or not isinstance(value, int):
            return False
        return value >= 1

    def items(self):
        lines = []
        for entry in self.reading.get("items") or []:
            if not isinstance(entry, dict) or not str(entry.get("product") or "").strip():
                continue
            wanted = entry.get("quantity")
            reqs = entry.get("requirements")
            if not isinstance(reqs, list):
                reqs = []
            lines.append({"product": str(entry["product"]).strip(),
                          "quantity": wanted if self.counted(wanted) else 1,
                          "requirements": [str(req) for req in reqs if req and ":" in str(req)]})
        if len(lines) < 2:
            return []
        return lines

    def quantity(self):
        if self.items():
            return 1
        wanted = self.reading.get("quantity")
        if self.counted(wanted):
            return wanted
        return 1

    def counts(self):
        lines = self.items()
        if not lines:
            return [self.quantity()]
        return [entry["quantity"] for entry in lines]

    def several_text(self):
        lines = self.items()
        if not lines:
            return ""
        listing = " | ".join(f"{spot}) {entry['product']} x{entry['quantity']}" for spot, entry in enumerate(lines, 1))
        return ("SEVERAL PRODUCTS ARE IN THIS REQUEST; each product gets just one order line at its own quantity, "
                "and no further lines go in; each listing is measured solely by the requirements of its own product "
                "(its category tells which), and the cheapest or best rule works inside each product: " + listing)

    def priority(self):
        rule = self.reading.get("priority")
        if rule in ("cheapest", "best"):
            return rule
        return "none"

    def props_left(self):
        if self.now_reqs is not None:
            source = self.now_reqs
        else:
            source = [req for req in self.written + self.uncovered if req not in self.let_go]
        left = []
        for req in source:
            if req in self.let_go:
                continue
            if re.search(r"\bprice\b|budget|quantity|\bstock\b|\bcents?\b", req.lower()):
                continue
            left.append(req)
        return left

    def text_body(self):
        if self.now_reqs is not None:
            shown = list(self.now_reqs)
        else:
            shown = [f"{req} (stated)" for req in self.written if req not in self.let_go]
            shown.extend(f"{req} (revealed by the shopper)" for req in self.uncovered if req not in self.let_go)
        if not shown:
            body = ("no property requirement remains: any in-stock listing of the requested category fits" if self.let_go
                    else "see the request")
        else:
            body = "; ".join(shown)
        if self.spend_cap is not None:
            body += f"; total price at most {self.spend_cap} {self.reading.get('currency') or ''}".rstrip()
        if self.items():
            body += "; " + self.several_text()
        else:
            body += f"; quantity {self.quantity()}"
        if self.let_go:
            body += "; no longer required: " + "; ".join(self.let_go)
        if self.indifferent:
            body += "; shopper does not care about: " + ", ".join(self.indifferent)
        return body

    def changes(self):
        if self.let_go:
            return ("\n\nUPDATES AFTER THE REQUEST (they override what it says): the shopper DROPPED " + "; ".join(self.let_go)
                    + ". Act as if it was never asked: it makes no listing fail, rank or win now.")
        return ""

    def overview(self, step_count, msgs):
        reading = self.reading
        if not reading:
            return ""
        get = reading.get
        rows = ["A first take on the request (compare it with the request's wording):"]
        if get("product"):
            rows.append("- product: " + str(get("product")))
        lines = self.items()
        if not lines:
            rows.append(f"- quantity: {self.quantity()}")
        else:
            rows += [f"- product {n}: {line['product']}, quantity {line['quantity']} (one cart line)"
                     for n, line in enumerate(self.items(), 1)]
            rows.append("- the order is made of these lines only, one per product; whenever the cart tool takes a "
                        "list of lines, each call to the cart carries all of them")
        labelled = (("requirements", "- stated requirement: "),
                    ("references", "- value given by reference to another listing (open that one, then search by its value): "))
        for key, lead in labelled:
            rows += [lead + str(item) for item in get(key) or []]
        if get("price_cap") is not None:
            rows.append(f"- price cap (total): {get('price_cap')} {get('currency') or ''}".rstrip())
        if get("in_stock_only"):
            rows.append("- only items in stock / on hand")
        rule = self.priority()
        if rule == "none":
            rows.append("- the request sets no cheapest/best rule: any item within the budget that meets every requirement "
                        "will do, so spend no steps hunting for a lower price (except where the request itself says so)")
        else:
            detail = get("priority_detail")
            rows.append(f"- among all items that fit, pick the {rule}" + (f" ({detail})" if detail else ""))
        tails = (("soft_wants", "- a preference, NOT a must (it just settles ties): "),
                 ("ask_first", "- to find out from the shopper before picks: "), ("rules", "- rule: "))
        for key, lead in tails:
            rows += [lead + str(item) for item in get(key) or []]
        if msgs is not None:
            rows.append(f"- messages to the shopper allowed: {msgs}")
        rows.append(f"- steps allowed: {step_count}")
        return "\n".join(rows)

    @staticmethod
    def spoken_names(needs, lns):
        words = re.findall(r"[a-z]+", " ".join(lns).lower())
        joined = " ".join(words)
        found = {}
        for need in needs:
            if not isinstance(need, str) or ":" not in need:
                continue
            head, _, value = need.partition(":")
            stems = re.findall(r"[a-z]+", re.split(r"\s*\(", head)[0].lower())
            if not stems or min(map(len, stems)) < 3:
                continue
            if re.search(r"\b" + r"\s+".join(stems) + r"\b", joined):
                continue
            tag = head[head.find("("):].strip() if "(" in head else ""
            span = len(stems)
            windows = (words[i: i + span] for i in range(len(words) - span + 1))
            hit = next((w for w in windows if w != stems and all(a.startswith(b) for b, a in zip(stems, w))), None)
            if hit is not None:
                found[need] = " ".join(hit) + (" " + tag if tag else "") + ":" + value
        return found

    @staticmethod
    def tags_in(text):
        return re.findall(r"\(([^)]*)\)", text)

    def drop_named(self, old_need, lns):
        try:
            return self.may_drop(old_need, lns)
        except Exception:
            return True

    @staticmethod
    def short_phrases(lns):
        words = re.findall(r"[a-z]+", " ".join(lns).lower())
        found = set()
        for size in (1, 2, 3):
            start = 0
            while start + size <= len(words):
                found.add(" ".join(words[start: start + size]))
                start += 1
        return set(p for p in found if len(p) >= 3 and p not in LINK_WORDS)

    def may_drop(self, old_need, lns):
        name = str(old_need).split(":", 1)[0]
        tag = self.tags_in(name)

        def same_slot(other):
            head = other.split(":", 1)[0]
            return same_prop(head, name) and (not tag or self.tags_in(head) in ([], tag))

        matching = list(filter(same_slot, self.written))
        if not matching:
            fresh = [n for n in self.fresh_needs or [] if isinstance(n, str) and ":" in n]
            matching = [r for r in self.uncovered if same_slot(r) and r not in self.written and r not in fresh]
            rival = any(same_slot(n) and n != old_need for n in fresh)
            if not matching or rival:
                return True
        phrases = self.short_phrases(lns)
        heads = [req.split(":", 1)[0] for req in matching]
        if any(same_prop(head, p) for head in heads for p in phrases):
            return True
        return bool(self.field_hook) and self.field_shows_drop(matching, name, phrases)

    def field_shows_drop(self, matching, name, phrases):
        for req in matching:
            head, _, raw = req.partition(":")
            value = re.sub(r"\((?:stated|revealed[^)]*)\)", " ", raw)
            if ref_text(value) or not ch_alts(value):
                continue
            fields = self.field_hook(head) or {}
            others = [r.split(":", 1)[0] for r in self.written + self.uncovered
                      if isinstance(r, str) and ":" in r and not same_prop(r.split(":", 1)[0], name)]
            phrase = self.naming_phrase(phrases, others, fields, value)
            if phrase is not None:
                self.note(dict(kind="drop_by_field", need=req[:200], phrase=phrase))
                return True
        return False

    @staticmethod
    def naming_phrase(phrases, others, fields, value):
        for phrase in phrases:
            if any(strict_prop(other, phrase) for other in others):
                continue
            for field, values in fields.items():
                if not strict_prop(field, phrase):
                    continue
                if value_seen(value, " | ".join(values)):
                    return phrase
        return None

    @staticmethod
    def plain_words(text):
        return " ".join(re.findall(r"[a-z0-9]+", text))

    def spelled_by_store(self, head, prop, bare):
        for field, values in (self.field_hook(head) or {}).items():
            if not strict_prop(field, prop):
                continue
            hit = next((v for v in values if self.plain_words(str(v).lower()) == bare), None)
            if hit is not None:
                return hit
        return None

    def store_spellings(self, answer):
        if not self.field_hook:
            return
        swaps = {}
        for need in answer.get("needs") or []:
            if not (isinstance(need, str) and ":" in need):
                continue
            head, _, value = need.partition(":")
            if re.search(r"\d", value) or ref_text(value):
                continue
            lowered = value.lower()
            bare = self.plain_words(re.sub(CJ_COMPARE, " ", lowered))
            if not bare:
                continue
            if bare == self.plain_words(lowered) and positive_value(value, head):
                continue
            hit = self.spelled_by_store(head, re.split(r"\s*\(", head)[0].strip(), bare)
            if hit is None:
                continue
            LITERAL_VALUES.add(bare)
            swaps[need] = head.strip() + ": " + str(hit).strip().lower()
        if swaps:
            self.replace_needs(answer, swaps)
            self.note(dict(kind="literal_value", needs=list(swaps.values())[:4]))

    @staticmethod
    def replace_needs(answer, swaps):
        def swap(entry):
            return swaps.get(entry, entry) if isinstance(entry, str) else entry
        answer["needs"] = [swap(n) for n in answer.get("needs") or []]
        if isinstance(answer.get("requirements_now"), list):
            answer["requirements_now"] = [swap(r) for r in answer["requirements_now"]]

    def stated_firmly(self, prop_name, qtext):
        try:
            given = self.reading.get("references")
            refs = list(map(str, given or [])) if isinstance(given, list) else []
            refs = refs + list(self.resolved or [])
            tag = self.tags_in(str(prop_name))
            wanted_key = prop_key(prop_name)
            for req in [r for r in self.written if ":" in r and r not in self.let_go]:
                head, _, value = req.partition(":")
                own_tag = self.tags_in(head)
                clash = tag and own_tag and not kind_words(tag[-1]) & kind_words(own_tag[-1])
                if clash or not positive_value(value, head) or story_guess(req, self.request, refs):
                    continue
                own_key = prop_key(head)
                if same_prop(head, prop_name) or (len(own_key) >= 4 and wanted_key.endswith(own_key)):
                    return True
        except Exception:
            return False
        return False

    @staticmethod
    def names_a_value(needs, lns):
        if not isinstance(needs, list):
            return False
        said = " ".join(str(line) for line in lns).lower()
        for need in needs:
            head, _, value = str(need).partition(":")
            prop_bits = [bit for bit in prop_words(head) if len(bit) >= 3]
            value_bits = [bit for bit in re.findall(r"[a-z0-9]+", value.lower()) if bit not in LINK_WORDS]
            if prop_bits and value_bits and all(bit in said for bit in prop_bits + value_bits):
                return True
        return False

    def hear_prompt(self, fresh, qtext, so_far):
        pieces = [HEAR, "\n\nTHE SHOPPER'S ORIGINAL REQUEST (background only):\n", self.request,
                  "\n\nREQUIREMENTS KNOWN SO FAR:\n", so_far]
        if self.exchanges:
            pieces.append("\n\nEARLIER TURNS WITH THE SHOPPER (oldest first):\n")
            pieces.append("\n".join(self.exchanges[-12:]))
        if qtext:
            pieces.append("\n\nTHE ASSISTANT ASKED:\n")
            pieces.append(qtext)
        pieces.append("\n\nTHE SHOPPER NOW SAID:\n")
        pieces.append("\n".join(fresh))
        return "".join(pieces)

    def park(self, fresh, qtext):
        self.caught.difference_update(fresh)
        self.exchanges.pop()
        self.pending = list(self.pending or []) + [(list(fresh), qtext)]
        self.pending_at = monotonic()

    def strip_meta(self, answer, fresh, qtext):
        try:
            if answer.get("meta_only") is not True or not qtext:
                return answer
            if self.names_a_value(answer.get("needs"), fresh):
                return answer
            answer = dict(answer, needs=[], no_preference=[], dropped=[], references=[], requirements_now=None,
                          new_budget=None, unavailable=False, reply="")
            self.meta_last = qtext
            self.note({"kind": "meta_only", "line": " / ".join(fresh)[:200]})
        except Exception:
            pass
        return answer

    def use_spoken_names(self, answer, fresh):
        try:
            renames = self.spoken_names(answer.get("needs") or [], fresh)
        except Exception:
            renames = {}
        if renames:
            self.replace_needs(answer, renames)
            self.note({"kind": "need_renamed", "names": list(renames.values())[:4]})

    def unpointed_lines(self, answer, fresh, qtext):
        if not self.ref_hook:
            return fresh
        try:
            kept = self.ref_hook(answer, fresh, qtext)
        except Exception as err:
            self.note({"kind": "ref_line_error", "error": f"{type(err).__name__}: {err}"[:300]})
            return fresh
        return kept or fresh

    def queue_pointed(self, answer, fresh):
        try:
            given = answer.get("references")
            infos = [r for r in given or [] if isinstance(r, dict)] if isinstance(given, list) else []
            for need in answer.get("needs") or []:
                if not isinstance(need, str) or need not in self.uncovered:
                    continue
                if not ref_text(need.partition(":")[2]) and need not in (answer.get("ref_needs") or []):
                    continue
                self.ref_news = list(self.ref_news or []) + [{"need": need, "infos": infos, "line": " ".join(fresh)}]
        except Exception:
            pass

    def take_indifference(self, answer, qtext):
        pool = self.now_reqs if self.now_reqs is not None else self.written + self.uncovered
        held = set()
        for req in pool:
            if isinstance(req, str) and ":" in req and req not in self.let_go:
                held.add(req.split(":", 1)[0].strip().lower())
        for prop_name in answer.get("no_preference") or []:
            if not isinstance(prop_name, str) or not prop_name or prop_name in self.indifferent:
                continue
            if prop_name.strip().lower() in held:
                continue
            if self.stated_firmly(prop_name, qtext):
                self.pinned_heard += 1
            else:
                self.indifferent.append(prop_name)

    def take_drops(self, answer, masked, remarks):
        widened = False
        budget_moved = is_number(answer.get("new_budget"))
        money = r"\b(?:prices?|budget|costs?|spend\w*|cents?|dollars?)\b"
        for old_need in answer.get("dropped") or []:
            usable = isinstance(old_need, str) and old_need and old_need not in self.let_go
            if not usable or (budget_moved and re.search(money, old_need.lower())):
                continue
            if not self.drop_named(old_need, masked):
                self.note(dict(kind="drop_kept", need=old_need[:200]))
                continue
            self.let_go.append(old_need)
            remarks.append("dropped: " + old_need)
            if not widened:
                widened = self.priority() != "none"
        return widened

    def bring_back(self, masked):
        for need in self.uncovered:
            name = need.split(":", 1)[0].strip().lower()
            if not name:
                continue
            present = any(name in req.lower() for req in self.now_reqs) or any(name in old.lower() for old in self.let_go)
            if not present:
                self.now_reqs.append(need)
        for need in self.written:
            try:
                self.bring_back_stated(need, masked)
            except Exception:
                pass

    def bring_back_stated(self, need, masked):
        if need in self.let_go or need in self.now_reqs:
            return
        if self.drop_named(need, masked):
            return
        name = need.split(":", 1)[0]
        tag = self.tags_in(name)

        def same_slot(other):
            head = other.split(":", 1)[0]
            return same_prop(name, head) and self.tags_in(head) == tag

        if any(same_slot(old) for old in self.let_go):
            return
        if any(same_slot(req) for req in self.now_reqs if isinstance(req, str)):
            return
        self.now_reqs.append(need)

    def opened_field_remark(self):
        if self.props_left():
            return ("a requirement the shopper let go OPENS the field: rows that failed on that one point now pass, and "
                    "rows no search has shown yet may undercut the current pick; search the kind once more without it, "
                    "using short queries of each remaining required value with the kind word, value first, reading every "
                    "page, and choose the cheapest over everything acceptable now")
        if self.priority() != "cheapest" or not self.sweepable:
            return ("nothing about the properties is required any more: any in-stock row of the requested category "
                    "passes, of any sort (accessories, parts and sets on that shelf too), and the lowest price on the "
                    "full shelf takes it. Rows arrive in relevance or id order, never by price: cover the shelf with "
                    "short queries, the first page of each - the category label, each kind word of the label alone, each "
                    "item name repeated in the titles you saw - plus the category listing's pages; next, buy the cheapest "
                    "one in stock")
        return ("nothing about the properties is required any more: any in-stock row of the requested category "
                "passes, of any sort (accessories, parts and sets on that shelf too), and the lowest price on the "
                "full shelf takes it. Rows arrive in relevance or id order, never by price, so the next shelf slices "
                "are fetched for you (the listing pages, the label's words, the object words in titles); when no "
                "'[shelf sweep done]' list shows up, sweep the shelf on your own with brief queries, reading page one "
                "of each - the category label, every kind word in that label alone, every item name repeated in "
                "titles you saw - plus the category listing's pages. Then let the cheapest in-stock row of the "
                "requested category within budget be the cart's sole line, re-read its stock and price, and "
                "order it")

    def ask_listener(self, fresh, qtext):
        so_far = self.text_body()
        prompt = self.hear_prompt(fresh, qtext, so_far)
        opener = "asked: " + str(qtext) + " -> " if qtext else ""
        self.exchanges.append(opener + "shopper: " + " / ".join(fresh))
        return so_far, self.ep.counsel.ask_json(prompt, 2500, LISTENER)

    def adopt_needs(self, answer, grounded, remarks):
        for need in answer.get("needs") or []:
            fresh_need = isinstance(need, str) and need and need not in self.uncovered
            if fresh_need and grounded(need):
                self.uncovered.append(need)
                remarks.append("new requirement: " + need)

    def hear(self, lns, qtext):
        fresh = [line for line in lns if line not in self.caught]
        if not fresh:
            return ""
        self.caught.update(fresh)
        if self.ep.time_left() < CLOSING:
            return ""
        so_far, answer = self.ask_listener(fresh, qtext)
        if not answer:
            self.park_if_quiet(fresh, qtext)
            return ""
        answer = self.strip_meta(answer, fresh, qtext)
        self.use_spoken_names(answer, fresh)
        masked = self.unpointed_lines(answer, fresh, qtext)
        grounded = self.grounding(" ".join([so_far, " ".join(self.exchanges), qtext or ""]).lower())
        self.learn_spellings(answer)
        needs = answer.get("needs") or []
        self.fresh_needs = [n for n in needs if isinstance(n, str)]
        self.take_now_reqs(answer.get("requirements_now"), grounded)
        remarks = []
        self.adopt_needs(answer, grounded, remarks)
        self.queue_pointed(answer, fresh)
        self.take_indifference(answer, qtext)
        widened = self.take_drops(answer, masked, remarks)
        budget_ok = self.take_budget(answer.get("new_budget"), remarks)
        if answer.get("unavailable"):
            remarks.append("an item became unavailable; take it out of the cart and do not order it")
        if remarks:
            self.version += 1
            if widened:
                self.widened_at, self.widened_tick = self.version, self.ep.shop.tick
        if self.now_reqs is not None:
            self.bring_back(masked)
        if widened:
            remarks.append(self.opened_field_remark())
        self.note(dict(kind="heard", question=qtext is not None, needs=len(needs),
                       free=len(answer.get("no_preference") or []), dropped=len(answer.get("dropped") or []),
                       budget=budget_ok))
        asked_back = qtext is None and not answer.get("no_preference") and any("?" in line for line in fresh)
        self.reply_due = str(answer.get("reply") or "").strip() if asked_back else ""
        meaning = "Meaning: " + str(answer.get("meaning") or "")
        return meaning + " (" + "; ".join(remarks) + ")" if remarks else meaning

    def park_if_quiet(self, fresh, qtext):
        counsel = self.ep.counsel
        if counsel.mute or counsel.misses > 0:
            self.park(fresh, qtext)

    @staticmethod
    def grounding(context):
        def grounded(req):
            words = [w for w in re.findall(r"[a-z]+", req.lower()) if len(w) >= 3]
            return not words or any(map(context.__contains__, words))
        return grounded

    def learn_spellings(self, answer):
        try:
            self.store_spellings(answer)
        except Exception as err:
            self.note(dict(kind="literal_value_error", error=f"{type(err).__name__}: {err}"[:300]))

    def take_now_reqs(self, now, grounded):
        if not isinstance(now, list) or any(not isinstance(r, str) for r in now):
            return
        self.now_reqs = [r for r in now if r.strip() and grounded(r)]

    def take_budget(self, budget, remarks):
        known = is_number(budget)
        if known and budget > 0 and budget != self.spend_cap:
            self.spend_cap = budget
            remarks.append(f"new budget: {budget}")
        return known

class Basket:
    def __init__(self, jschema):
        self.prop_specs = (jschema or {}).get("properties") or {}
        self.put_form = None
        self.units = {}
        self.adds = None
        self.used = False
        self.last_view = None
        self.pending = {}
        self.count_seen = None

    def spec_of(self, key):
        spec = self.prop_specs.get(key)
        return spec if isinstance(spec, dict) else {}

    def mode_of(self, call_args):
        return as_text({key: val for key, val in call_args.items() if "enum" in self.spec_of(key)})

    def line_spec(self):
        for key, spec in self.prop_specs.items():
            if not isinstance(spec, dict) or spec.get("type") != "array":
                continue
            inner = spec.get("items")
            if not isinstance(inner, dict) or not isinstance(inner.get("properties"), dict):
                continue
            fields = inner["properties"]
            words = []
            figures = []
            for name, sub in fields.items():
                if not isinstance(sub, dict):
                    continue
                if sub.get("type") == "string" and "enum" not in sub:
                    words.append(name)
                if sub.get("type") in ("integer", "number"):
                    figures.append(name)
            ids = [name for name in words if "id" in name.lower()] or words
            if ids and figures:
                return key, ids[0], figures[0]
        return None

    def list_mode(self, call_args):
        spec = self.line_spec()
        if spec and isinstance(call_args.get(spec[0]), list):
            return spec
        return None

    def lines(self, call_args):
        spec = self.list_mode(call_args)
        if spec is not None:
            array_key, id_key, count_key = spec
            rows = [row for row in call_args[array_key] if isinstance(row, dict)]
            named = [(row[id_key].strip(), row) for row in rows if isinstance(row.get(id_key), str) and row[id_key].strip()]
            return [(ident, float(row[count_key]) if is_number(row.get(count_key)) else None) for ident, row in named]
        thing = self.flat_target(call_args)
        if not thing:
            return []
        return [(thing, self.count_arg(call_args))]

    @staticmethod
    def leading_put(found):
        for line in found:
            if line[1] is None or line[1] > 0:
                return line
        return found[0] if found else None

    def aim(self, call_args):
        line = self.leading_put(self.lines(call_args))
        return line[0] if line else None

    def flat_target(self, call_args):
        for key, piece in call_args.items():
            if not isinstance(piece, str) or "enum" in self.spec_of(key):
                continue
            core = piece.strip()
            if core and " " not in core:
                return piece
        return None

    def count_arg(self, call_args):
        if self.list_mode(call_args):
            line = self.leading_put(self.lines(call_args))
            return line[1] if line else None
        key = self.count_key(call_args)
        return None if key is None else float(call_args.get(key, self.spec_of(key).get("default")))

    def count_key(self, call_args):
        for key, spec in self.prop_specs.items():
            if not isinstance(spec, dict) or "enum" in spec:
                continue
            if spec.get("type") not in ("integer", "number"):
                continue
            if is_number(call_args.get(key, spec.get("default"))):
                return key
        return None

    def is_put(self, call_args):
        if self.put_form is None:
            return True
        return self.put_form == self.mode_of(call_args)

    def held(self, thing):
        return self.units.get(thing, 0.0)

    def match_reply(self, obs, whole=True):
        seen = self.view_units(obs)
        if seen is None:
            return False
        if whole or len(seen) != 1:
            self.absorb_counts(seen)
            return True
        others = {thing: qty for thing, qty in self.units.items() if thing not in seen}
        self.absorb_counts(seen)
        self.units.update(others)
        return True

    def book_lines(self, call_args, obs):
        self.adds = False
        self.units = {}
        for thing, count in self.lines(call_args):
            if count is not None and count <= 0:
                continue
            self.units[thing] = self.units.get(thing, 0.0) + (1.0 if count is None else count)
        self.match_reply(obs)
        return self.summary()

    def next_units(self, thing, count, putting):
        before = self.held(thing)
        if not putting:
            return 0.0 if count <= 0 else max(0.0, before - count)
        if before > 0 and self.adds is not False:
            self.pending = {thing: count}
            return before + count
        return 0.0 if count <= 0 else count

    def booked(self, call_args, obs):
        self.used = True
        if isinstance(obs, dict) and obs.get("error"):
            return ""
        if self.put_form is None:
            self.put_form = self.mode_of(call_args)
        if self.list_mode(call_args):
            return self.book_lines(call_args, obs)
        thing = self.aim(call_args)
        if not thing:
            return ""
        count = self.count_arg(call_args)
        count = 1.0 if count is None else count
        self.units[thing] = self.next_units(thing, count, self.is_put(call_args))
        expected = self.units[thing]
        if self.units.get(thing, 0) <= 0:
            del self.units[thing]
        shown = self.match_reply(obs, whole=False)
        if shown and count > 1 and self.is_put(call_args) and abs(self.units.get(thing, 0) - expected) < 1e-9:
            self.count_seen = self.count_key(call_args)
        if not shown:
            removing = count <= 0 or not self.is_put(call_args)
            if removing and self.reply_empty(obs, thing):
                self.units.pop(thing, None)
                self.pending.pop(thing, None)
        return self.summary()

    def reply_empty(self, obs, thing):
        if not isinstance(obs, dict) or thing not in self.units:
            return False
        lists = [val for val in obs.values() if isinstance(val, list)]
        if not lists or any(lists):
            return False
        figures = [val for val in obs.values() if isinstance(val, (int, float)) and not isinstance(val, bool)]
        if not figures or any(val != 0 for val in figures):
            return False
        shown = as_text(obs)
        for other in self.units:
            if other != thing and other in shown:
                return False
        return True

    @staticmethod
    def quantity_fields(row):
        numbers = [(k, v) for k, v in row.items()
                   if isinstance(v, int) and not isinstance(v, bool) and 0 < v < 1000 and str(k) != "price"]
        if len(numbers) < 2:
            return numbers
        counted = [(k, v) for k, v in numbers
                   if re.search(r"qty|quant|count|units|number|total|held|ordered", str(k).lower())
                   and not re.search(r"price|cost|due|sum|fee|charge", str(k).lower())]
        return counted if len(counted) == 1 else numbers

    def view_units(self, obs):
        tallies = {}
        known = set(self.units) | set(self.pending)

        def count_row(row):
            idents = [v for v in row.values()
                      if isinstance(v, str) and (v in known or re.fullmatch(ID_LIKE, v)) and " " not in v]
            numbers = self.quantity_fields(row)
            if not idents or len(numbers) != 1:
                return
            ident = next((i for i in idents if i in known), idents[0])
            tallies[ident] = numbers[0][1] + tallies.get(ident, 0)
        walk_dicts(obs, count_row)
        if not tallies or not set(tallies) & (known or set(tallies)):
            return None
        return tallies

    def absorb_counts(self, seen):
        for thing, expected in self.pending.items():
            if thing not in seen:
                continue
            shown = seen[thing]
            same_as_set = abs(shown - expected) < 1e-9
            if same_as_set and abs(self.units.get(thing, 0) - expected) > 1e-9:
                self.adds = False
            elif shown > expected:
                self.adds = True
        self.pending = {}
        self.units = {thing: qty for thing, qty in seen.items() if qty > 0}

    def viewed(self, obs):
        self.last_view = obs
        if self.match_reply(obs):
            return
        if not isinstance(obs, (dict, list)):
            return
        shown = as_text(obs)
        if not any(thing in shown for thing in self.units):
            self.units = {}

    @staticmethod
    def as_count(qty):
        if float(qty).is_integer():
            return int(qty)
        return qty

    def summary(self):
        if self.units:
            return "[cart as counted] " + ", ".join(f"{thing} x{self.as_count(qty)}" for thing, qty in self.units.items()) + "."
        return "[cart as counted] empty."

    def faults(self, need_n, counts=None):
        if not self.units:
            return []
        if counts and len(counts) > 1:
            return self.basket_faults(counts)
        if len(self.units) > 1:
            listing = ", ".join("%s x%d" % (ident, int(qty)) for ident, qty in self.units.items())
            return ["the cart holds more than one line (" + listing + "); take out every line but the chosen one"]
        wrong = [(thing, qty) for thing, qty in self.units.items() if abs(qty - need_n) > 1e-9]
        return [f"the line {thing} holds {int(qty)} units but the request needs {need_n}; "
                + ("take out" if qty > need_n else "put in") + f" {int(abs(qty - need_n))} with the cart tool"
                for thing, qty in wrong]

    def basket_faults(self, counts):
        if len(self.units) != len(counts):
            shown = ", ".join(f"{i} x{int(q)}" for i, q in self.units.items())
            return [f"the cart holds {len(self.units)} line(s) ({shown}) but "
                    f"the request asks for {len(counts)} products, one line each; every product needs its line and nothing else"]
        held = sorted(int(q) for q in self.units.values())
        if held == sorted(counts):
            return []
        return [f"the lines hold {held} units but the products need {sorted(counts)} (one line per product at its quantity)"]


class Assessor:
    def __init__(self, ep):
        self.ep = ep
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

    def field_names(self):
        menu, _ = self.ep.known.prop_menu()
        if not menu:
            return ""
        return ("\n\nFIELD NAMES ON THIS STORE'S LISTINGS (two names apart only in a separator are a single name; the "
                "figure gives the count of read listings showing it):\n" + menu)

    @staticmethod
    def refusal_text(thing, why):
        return f"An independent check says {thing} does NOT meet every requirement: {why}. Choose another listing."

    def pointed_cards(self, refs, thing):
        known = self.ep.known
        clues = set()
        for ref in refs:
            clues |= set(re.findall(r"[a-z]{4,}", ref.lower().split(":", 1)[-1]))
        hits = []
        for ident in sorted(known.read):
            if len(hits) == 3:
                break
            if ident == thing:
                continue
            text = " ".join(known.about(ident, 1)).lower()
            if any(clue in text for clue in clues):
                hits.append(ident)
        if not hits:
            return ""
        records = [line for ident in hits for line in known.about(ident, 1)]
        return ("\n\nREAD LISTINGS THE REQUEST POINTS AT (when a requirement refers to some other listing, its value comes "
                "from the named field of that very listing):\n" + "\n".join(records))

    @staticmethod
    def redirect_notes(own):
        sentences = []
        for req, prop, sent in own:
            sentences.append(f"this item's notes hold a line that may send {prop} elsewhere: \"{sent[:240]}\". Should it truly do "
                             f"so (have the reader take {prop} from, refer to, go by or look to another listing), the grader checks "
                             f"{prop} on that one, and the item's own {prop} field no longer settles '{req}': it is met only if the "
                             "record of that other listing shows it. A listing mentioned merely as a variant sold elsewhere, or as "
                             "what this item isn't, alters nothing.")
        return "\n\nNOTE (how the code reads the notes of this item): " + " ".join(sentences)

    def ruling_key(self, stamp, refs, tracked, second, own):
        key = stamp
        if refs or any(r.get("status") != "resolved" for r in tracked):
            key += "|" + str(len(self.ep.known.read))
        if second:
            key += "|second"
        if own:
            key += f"|own{len(own)}"
        return key

    def rule_against(self, memo, thing, verdict, entry):
        self.ep.rulings[memo] = verdict
        self.ep.log.append({"kind": "verify", "item": thing, "result": entry})
        return verdict

    def item_prompt(self, thing, aim_text, second, own, refs, tracked, history, evidence):
        ep = self.ep
        wants, basket, pointers = ep.wants, ep.basket, ep.pointers
        text = CHECK_ITEM + "\n\nSHOPPER REQUEST TEXT:\n" + ep.request + wants.changes()
        text += "\n\nCURRENT REQUIREMENTS (stated, revealed or changed since):\n" + wants.text_body()
        if wants.items():
            units = basket.held(thing)
            text += ("\n\nSeveral products are wanted, and the category of this item shows which one it belongs to: test it "
                     "solely against the requirements and quantity of that product, ignoring what the other products need.")
            if units > 0:
                text += f" It is in the cart with {int(basket.held(thing))} units."
        text += pointers.ref_note(thing) or ""
        outside = pointers.ref_untracked(refs, tracked)
        text += self.pointed_cards(outside, thing) if outside else ""
        text += second
        text += self.redirect_notes(own) if own else ""
        text += f"\n\nTHE ITEM TO {aim_text}: {thing}\nRECORDS OF THIS ITEM SEEN IN THE STORE (earliest first):\n"
        text += "\n".join(history)
        if basket.last_view is not None:
            text += "\n\nLATEST CART VIEW:\n" + as_text(basket.last_view)[:3000]
        text += self.field_names()
        if evidence["lines"]:
            text += ("\n\nFIELD EVIDENCE (derived by code from records in this item's category: the spelling this shop gives "
                     "each required value, and what this item itself shows):\n" + "\n".join(evidence["lines"]))
        labels = ep.known.label_lines()
        if labels:
            text += ("\n\nLABELS SEEN AGAIN AND AGAIN ON THE LISTINGS (counted; an item's category label has to fit the "
                     "whole wording of the request, never a nearby label):\n" + labels)
        return text

    def request_refs(self):
        given = self.ep.wants.reading.get("references")
        if not isinstance(given, list):
            return []
        return [str(r) for r in given or [] if r]

    def early_ruling(self, memo, thing):
        chain = self.ep.pointers.ref_pointer_note(thing)
        if chain:
            entry = dict(item=thing, fits=False, problems=["reference chain"])
            return self.rule_against(memo, thing, self.refusal_text(thing, chain), entry)
        return None

    def check_item(self, thing, aim_text, second=""):
        ep = self.ep
        if ep.time_left() < CLOSING + 20:
            return ""
        stamp = f"{thing}|{ep.wants.version}"
        if stamp in ep.shop.near_ok:
            return ""
        refs = self.request_refs()
        tracked = list(ep.refs or [])
        own = ep.pointers.own_redirects(thing)
        memo = self.ruling_key(stamp, refs, tracked, second, own)
        rulings = ep.rulings
        if memo in rulings:
            return rulings[memo]
        early = self.early_ruling(memo, thing)
        if early is not None:
            return early
        history = ep.known.about(thing)
        if not history:
            return ""
        evidence = self.field_view(thing)
        failing = evidence["fails"][:3]
        if failing:
            ep.shortfalls[stamp] = list(map(str, failing))
            entry = dict(item=thing, fits=False, problems=failing, by="field evidence")
            return self.rule_against(memo, thing, self.refusal_text(thing, "; ".join(failing)), entry)
        prompt = self.item_prompt(thing, aim_text, second, own, refs, tracked, history, evidence)
        answer = self.ask_json(prompt, 3000, CHECKER)
        ep.log.append(dict(kind="verify", item=thing, result=answer))
        rejected = bool(answer) and answer.get("fits") is False
        if rejected:
            problems = list(map(str, answer.get("problems") or []))[:5]
            ep.shortfalls[stamp] = problems or ["unstated"]
            if isinstance(answer.get("missed"), list) and not answer["missed"]:
                ep.near_fits.add(stamp)
            rulings[memo] = (f"An independent check says {thing} does NOT meet every requirement: " + "; ".join(problems)
                             + ". Pick a different, fully verified fit unless you know for sure the check erred")
            return rulings[memo]
        rulings[memo] = ""
        if answer:
            ep.passed.add(stamp)
            ep.pass_price[thing] = ep.known.price(thing)
        return ""

    def coded_rows(self):
        size = len(self.ep.known.items)
        memo = self.ch_rows_memo
        if memo and memo[0] == size:
            return memo[1]
        rows = self.ep.closer.code_rows()
        self.ch_rows_memo = (len(self.ep.known.items), rows)
        return rows

    def store_fields(self, label=""):
        ep = self.ep
        try:
            rows = self.coded_rows()
            homes = ep.shop.home_shelves()
            if ep.wants.items():
                slot = ep.talk.product_of(str(label))
                shelves = set(homes[slot]) if slot is not None and slot < len(homes) else set()
            else:
                own = ep.shop.own_shelf()
                shelves = {own} if own else (set(homes[0]) if homes else set())
            collected = {}
            for row in rows.values():
                if not row["fields"] or (shelves and row["shelf"] not in shelves):
                    continue
                for name, value in row["fields"].items():
                    if junk_key(name) or not str(value).strip():
                        continue
                    collected.setdefault(str(name), []).append(str(value))
            return collected
        except Exception:
            return {}

    def notes_redirect(self, thing, prop):
        try:
            for found in self.ep.pointers.own_redirects(thing) or []:
                heads = found if isinstance(found, (list, tuple)) else [found]
                for head in heads[:2]:
                    if same_prop(str(head).partition(":")[0], prop):
                        return True
            return False
        except Exception:
            return True

    def field_view(self, thing):
        ep = self.ep
        wants = ep.wants
        key = (thing, len(ep.known.items), wants.version, "|".join(wants.props_left()))
        memo = self.ch_view_memo
        view = memo.get(key)
        if view is None:
            try:
                view = self.field_facts(thing)
            except Exception as err:
                ep.log.append(dict(kind="field_facts_error", error=f"{type(err).__name__}: {err}"[:300]))
                view = dict(lines=[], rules=[], fails=[])
            memo[key] = view
        agreed = (ep.shop.near_said or {}).get(f"{thing}|{wants.version}")
        if not agreed:
            return view
        try:
            return self.with_agreement(view, agreed)
        except Exception:
            return view

    @staticmethod
    def with_agreement(view, agreed):
        def same_field(field, prop):
            return same_prop(field, prop) or strict_prop(field, prop)

        lines = []
        for line in view["lines"]:
            head = re.match(r"- (.+?) = '", line)
            match = None
            if head:
                for entry in agreed:
                    if same_field(head.group(1), entry[0]):
                        match = entry
                        break
            if match:
                line += (f" A check of meaning found its '{match[2][:80]}' as '{match[1][:60]}' (the same value in other words): it "
                         "meets it.")
            lines.append(line)
        props = [entry[0] for entry in agreed]
        fails = [fail for fail in view["fails"]
                 if not any(same_field(fail.split(":", 1)[0].strip(), prop) for prop in props)]
        return {"lines": lines, "rules": list(view["rules"]), "fails": fails}

    @staticmethod
    def shorten(text, n=70):
        flat = " ".join(str(text).split())
        return flat if len(flat) <= n else flat[: n - 3] + "..."

    def peer_table(self, peers):
        carry, table = {}, {}
        for ident, row in peers.items():
            for name, value in row["fields"].items():
                if junk_key(name) or not str(value).strip():
                    continue
                carry[name] = 1 + carry.get(name, 0)
                parts = ch_parts(value)
                if len(parts) > 1:
                    parts.append(str(value).strip().lower())
                table.setdefault(name, []).extend([(ident, part, ch_words(part)) for part in parts])
        return carry, table

    def field_facts(self, thing):
        result = dict(lines=[], rules=[], fails=[])
        rows = self.coded_rows()
        mine = rows.get(thing)
        if not (mine and mine["fields"]):
            return result
        several = bool(self.ep.wants.items())
        shelf = mine["shelf"]
        if not shelf and several:
            return result
        slot = None
        if several:
            homes = self.ep.shop.home_shelves()
            slot = next(filter(lambda n: shelf in homes[n], range(len(homes))), None)
        peers = dict((i, r) for i, r in rows.items() if r["fields"] and (r["shelf"] == shelf or not shelf))
        if len(peers) < 3:
            return result
        carry, table = self.peer_table(peers)
        scene = dict(rows=rows, mine=mine, shelf=shelf, several=several, slot=slot, peers=peers, carry=carry, table=table)
        for req in self.ep.wants.props_left()[:12]:
            self.judge_requirement(thing, req, scene, result)
        return result

    def bound_evidence(self, prop, value, head, scene, result):
        cut = self.shorten
        mine, peers, carry = scene["mine"], scene["peers"], scene["carry"]
        own_slot = self.ep.talk.product_of(head) if scene["several"] else None
        kin = set(filter(lambda w: len(w) >= 3 and w not in LINK_WORDS, set(prop_words(prop))))
        named = [name for name, count in carry.items() if count >= 2 and strict_prop(name, prop)]
        if not (prop and kin and named):
            return
        if scene["several"] and own_slot not in (None, scene["slot"]):
            return
        for name, shown in mine["fields"].items():
            if str(shown).strip() and (strict_prop(name, prop) or kin & set(prop_words(name))):
                return
        listed = " / ".join(f"'{x}'" for x in named[:3])
        holding = 0
        for row in peers.values():
            if any(str(row['fields'].get(x, '')).strip() for x in named):
                holding += 1
        result["lines"].append(f"- {prop} = '{cut(value)}': the shop records it under {listed}, found on {holding} of "
                               f"{len(peers)} read listings in this category. The field is missing on this item, so the limit is never "
                               "proven and it fails: it does NOT meet it (the figure taken from another field cannot replace it).")
        result["fails"].append(f"{prop}: it has no {named[0]} field, which this store's listings of the category carry, so "
                               f"'{cut(value)}' is not shown met")

    @staticmethod
    def split_requirement(req):
        head, _, raw = str(req).partition(":")
        value = " ".join(re.sub(r"\((?:stated|revealed[^)]*)\)", " ", raw).split())
        prop = re.split(r"\s*\(", head)[0].strip()
        if not prop or not value or ref_text(value):
            return None
        if re.search(r"as shown|same as|the page|sold by|redirect|refer", value.lower()):
            return None
        return head, prop, value

    def fields_for(self, fact, scene):
        carry, table = scene["carry"], scene["table"]
        heaviest_first = lambda names: sorted(names, key=lambda name: -carry[name])
        named = heaviest_first([name for name in carry if strict_prop(name, fact["prop"])])
        fact["by_name"] = len(named) > 0
        if not named:
            if not fact["positive"] or (scene["several"] and fact["slot"] is None):
                return None
            alts = fact["alts"]
            writing = [name for name in carry
                       if carry[name] >= 2 and any(ch_match(alts, words) for _, _, words in table.get(name, []))]
            if not writing:
                return None
            top = max(writing, key=lambda name: carry[name])
            return heaviest_first([name for name in carry if strict_prop(name, top)])
        if not fact["positive"]:
            flat = " ".join(re.findall(r"[a-z0-9]+", fact["value"].lower()))
            spelled = any(" ".join(words) == flat for name in named for _, _, words in table.get(name, []))
            if spelled:
                fact["positive"] = True
                LITERAL_VALUES.add(flat)
        return named

    def rule_line(self, fact):
        cut = self.shorten
        lead, value = fact["lead"], fact["value"]
        if not fact["positive"]:
            return lead
        met = " | ".join(fact["samples"])
        if fact["writers"]:
            extra = ""
            if not fact["by_name"]:
                extra = (" Since no field bears this property's name, a title spelling out the value counts as well, "
                         "unless one of its own fields with the property's name states otherwise.")
            return (lead + (" Just a value worded exactly like that is accepted (containment and one list element count; a "
                            "synonym or related word does not).") + extra)
        if fact["naming"]:
            return (lead + " Values met so far: " + met + f"; '{cut(value)}' appears in the title or text of "
                    f"some listings of this category ({', '.join(fact['naming'])}), so the store itself uses the word: other "
                    "wording counts only once such a listing's record shows this field writes it otherwise.")
        if fact["on"] >= CH_SYNONYM_MIN:
            return lead + " Values met so far: " + met + "; not one spells it out, so a value of the same meaning may meet it."
        return (lead + " Values met so far: " + met + f"; none of them writes it, yet too few read listings "
                f"have this field to say whether the store would spell '{cut(value)}' literally: a value of the same "
                "meaning is not yet shown to meet it.")

    def sample_values(self, peers, named):
        found = []
        for row in peers.values():
            fields = row["fields"]
            found += [self.shorten(fields[name], 40) for name in named if str(fields.get(name, "")).strip()]
        return list(dict.fromkeys(found))[:6]

    def exclusion_tail(self, value, others):
        cut = self.shorten
        bare = " ".join(re.sub(EXCLUDE_WORDS, " ", value.lower()).split())
        bare_alts = ch_alts(bare)
        if not bare_alts:
            return None
        carrying = set(ident for ident, words in others if ch_match(bare_alts, words))
        return (f"{len(carrying)} other listing(s) put '{cut(bare)}' there, the very value excluded (any listing writing "
                "it fails to satisfy it).")

    def judge_requirement(self, thing, req, scene, result):
        parsed = self.split_requirement(req)
        if not parsed:
            return
        head, prop, value = parsed
        lowered = value.lower()
        if re.search(r"\d", value) and re.search(CH_BOUND, lowered):
            return self.bound_evidence(prop, value, head, scene, result)
        if re.fullmatch(r"\s*\d+(?:\.\d+)?\s+[a-z]+\s*", lowered):
            return
        slot = None
        if scene["several"]:
            slot = self.ep.talk.product_of(head)
            if slot is not None and slot != scene["slot"]:
                return
        positive = positive_value(value, head)
        if not positive and not re.search(EXCLUDE_WORDS, lowered):
            return
        alts = ch_alts(value)
        if not alts:
            return
        fact = dict(prop=prop, value=value, slot=slot, positive=positive, alts=alts)
        named = self.fields_for(fact, scene)
        if named is None:
            return
        fact["named"] = named
        self.fill_fact(fact, thing, scene)
        result["rules"].append(self.rule_line(fact))
        self.own_field_line(thing, fact, scene, result)

    def fill_fact(self, fact, thing, scene):
        cut = self.shorten
        named, value, peers = fact["named"], fact["value"], scene["peers"]
        fact["on"] = len([row for row in peers.values() if any(str(row["fields"].get(n, "")).strip() for n in named)])
        others = [(ident, words) for name in named for ident, _, words in scene["table"].get(name, []) if ident != thing]
        fact["writers"] = sorted({ident for ident, words in others if ch_match(fact["alts"], words)})
        fact["alone"] = {" ".join(words) for _, words in others}
        fact["samples"] = self.sample_values(peers, named)
        tail = f"{len(fact['writers'])} other listing(s) write '{cut(value)}' itself in it."
        if not fact["positive"]:
            tail = self.exclusion_tail(value, others) or tail
        lead = f"- {fact['prop']} = '{cut(value)}': the store's field for it is " + " / ".join(f"'{x}'" for x in named[:3])
        if not fact["by_name"]:
            lead += f" (nothing is called '{fact['prop']}'; the values of this one carry the value)"
        fact["lead"] = lead + f", on {fact['on']} of {len(peers)} read listings of this category; " + tail
        fact["naming"] = self.naming_rows(value, fact["writers"], thing, scene)

    @staticmethod
    def naming_rows(value, writers, thing, scene):
        kinds = kind_words(value)
        if not kinds or writers:
            return []
        shelf = scene["shelf"]
        hits = [ident for ident, row in scene["rows"].items()
                if ident != thing and (not shelf or row["shelf"] == shelf) and kinds <= row["words"]]
        return hits[:3]

    @staticmethod
    def longer_keys(fields, named, prop):
        core = prop_words(prop)
        if not core:
            return []
        found = []
        for name, shown in fields.items():
            if name in named or junk_key(name) or not str(shown).strip():
                continue
            words = prop_words(name)
            if len(words) == len(core) + 1 and words[1:] == core:
                found.append((name, str(shown)))
        return found

    def absent_field_line(self, lead, prop, value, fields, named, qual_note, title_says):
        cut = self.shorten
        if qual_note:
            return lead + " This listing has no such field." + qual_note
        if title_says:
            return lead + (" The listing lacks that field and no field is named for the property, but its title "
                           f"spells '{cut(value)}': it meets it.")
        short = set(w for w in prop_words(prop) if len(w) >= 3)

        def related(pair):
            name, shown = pair
            if name in named or junk_key(name) or not str(shown).strip():
                return False
            return bool(set(prop_words(name)) & short) or bool(ch_writes(value, shown))

        candidates = sorted(filter(related, fields.items()), key=lambda pair: not ch_writes(value, pair[1]))[:4]
        listed = ""
        if candidates:
            listed = ": candidate fields for the property: " + "; ".join(
                f"{name}: '{cut(shown, 60)}'" for name, shown in candidates)
        return (lead + " No such field here" + listed + (". A field under a different name for this SAME property is fine; one for another property (a height "
                                                         "for a type, a frame material for the material) does not stand in."))

    def longer_key_note(self, fields, named, prop, value):
        longer = self.longer_keys(fields, named, prop)
        writing = [pair for pair in longer if ch_writes(value, pair[1])]
        near = bool(writing) or any(ch_touches(value, v) for _, v in longer)
        if not writing:
            return "", near
        key_name, key_value = writing[0]
        return (f" Its {key_name} '{self.shorten(key_value, 60)}' holds it (a longer key for the same "
                "property: one added word in front of the property's name keeps naming the property, just with a longer "
                "name), so the item meets it."), near

    @staticmethod
    def split_matches(parts, alts, alone):
        clean, joined = [], []
        for part in parts:
            if not ch_match(alts, ch_words(part)):
                continue
            sides = ch_sides(part)
            rest = []
            if sides and any(ch_match(alts, ch_words(side)) for side in sides):
                rest = [s for s in sides if not ch_match(alts, ch_words(s)) and " ".join(ch_words(s)) in alone]
            if rest:
                joined.append((part, rest))
            else:
                clean.append(part)
        return clean, joined

    def own_field_line(self, thing, fact, scene, result):
        prop, value, named, lead = fact["prop"], fact["value"], fact["named"], fact["lead"]
        mine = scene["mine"]
        fields = mine["fields"]
        own = [(name, str(fields[name])) for name in named if str(fields.get(name, "")).strip()]
        shown = "; ".join(f"{name}: '{self.shorten(v, 90)}'" for name, v in own)
        qual_note, longer_near = self.longer_key_note(fields, named, prop, value)
        lines = result["lines"]
        if self.notes_redirect(thing, prop):
            return lines.append(lead + (" The notes here may send the reader to a different listing for this property, so its own field may "
                                        "not decide."))
        if not fact["positive"]:
            return lines.append(lead + " Here: " + (shown or "no such field") + " (this is an exclusion, so the listing must not "
                                "carry the excluded value).")
        core = set(w for w in prop_words(prop) if len(w) >= 3 and w not in LINK_WORDS)
        kin = False
        for name, v in fields.items():
            if str(v).strip() and not junk_key(name) and set(prop_words(name)) & core and not ch_writes(value, v):
                kin = True
                break
        title_says = not fact["by_name"] and not kin and ch_writes(value, mine.get("title") or "")
        if not own:
            return lines.append(self.absent_field_line(lead, prop, value, fields, named, qual_note, title_says))
        parts = [p for _, v in own for p in ch_parts(v)] + [v.strip().lower() for _, v in own if len(ch_parts(v)) > 1]
        clean, joined = self.split_matches(parts, ch_alts(value), fact["alone"])
        line, fail = self.own_verdict(fact, shown, own, clean, joined, qual_note, longer_near, title_says)
        lines.append(line)
        if fail:
            result["fails"].append(fail)

    def own_verdict(self, fact, shown, own, clean, joined, qual_note, longer_near, title_says):
        cut = self.shorten
        prop, value, named, lead = fact["prop"], fact["value"], fact["named"], fact["lead"]
        said = cut(value)
        here = lead + " Here: " + shown
        if clean:
            return here + f" - it writes '{said}': it meets it.", ""
        if qual_note:
            return here + "." + qual_note, ""
        if title_says:
            return (here + " - as no field has the property's name, the title settles it, and that title spells "
                    f"'{said}': it meets it."), ""
        strict = fact["by_name"] and not longer_near
        if joined:
            part, others = joined[0]
            fail = ""
            if strict:
                fail = (f"{prop}: its {named[0]} is '{cut(part)}', which joins {said} with {others[0]} (a "
                        f"value this store also writes alone there), so it is not {said}")
            return (here + f" - '{cut(part)}' combines '{said}' with '{others[0]}', a value that "
                    "other listings carry alone in this field: it does NOT meet it."), fail
        if fact["writers"]:
            fail = ""
            if strict and not any(ch_touches(value, v) for _, v in own):
                fail = (f"{prop}: its {named[0]} is '{cut(own[0][1], 60)}', while this store writes "
                        f"'{said}' itself in that field on other listings")
            return (here + f" - '{said}' is missing, while other listings of "
                    "this category place it in this field: so it fails (different wording, a synonym or a related word does not "
                    "count where the store writes the value itself)."), fail
        met = " | ".join(fact["samples"])
        if fact["naming"]:
            return (here + f" - '{said}' is not there. No read listing has placed it in "
                    "this field yet (values: " + met + f"), though the titles or texts of this category's listings hold "
                    f"'{said}' ({', '.join(fact['naming'])}), so the shop has its own word: "
                    "so this listing's other value is not shown to meet it."), ""
        on = fact["on"]
        opening = here + f". No read listing of this category places '{said}' in this field (values: " + met
        if on >= CH_SYNONYM_MIN:
            return opening + "), so a value of the same meaning may meet it.", ""
        return (opening + f"); since just {on} read listing(s) carry it, no one can "
                f"say yet if '{said}' would be written literally, so a value of the same meaning is "
                "not yet shown to meet it."), ""

    def rank_wording(self):
        wants = self.ep.wants
        if wants.priority() == "cheapest":
            return "cheapest (lowest price)"
        detail = wants.reading.get("priority_detail") or ""
        return f"best ({detail})" if detail else "best"

    def rank_prompt(self, thing, what, cheapest):
        ep = self.ep
        wants = ep.wants
        parts = [RANK_CHECK.replace("PRIORITY", what), "\n\nSHOPPER REQUEST TEXT:\n", ep.request, wants.changes(),
                 "\n\nCURRENT REQUIREMENTS:\n", wants.text_body(), ep.pointers.ref_note()]
        if wants.items():
            parts.append("\n\nAs several products are wanted, the chosen item is weighed only against listings of the SAME "
                         "product (category); a listing of another product can never outrank it.")
        cards = ep.known.cards(20000, thing if cheapest else "", ep.pointers.ref_foreign(), {thing})
        parts.append(f"\n\nTHE CHOSEN ITEM: {thing}\n\nLISTINGS SEEN (one card each; DETAILS = whole record opened, ROW = met only "
                     "in search rows, its fields unknown: a ROW of the requested category (whatever thing its title names) "
                     "that outdoes the chosen item and that title, price or stock do not exclude MAY fit and gets named"
                     + ("; shown are only listings cheaper than the chosen one" if cheapest else "") + "):\n")
        parts.append("\n".join(cards))
        parts.append(self.field_names())
        rules = self.field_view(thing)["rules"]
        if rules:
            parts.append("\n\nFIELD EVIDENCE (derived by code from records in the chosen item's category: how this shop writes "
                         "each value that is required):\n" + "\n".join(rules))
        return "".join(parts)

    def rank_check(self, thing):
        ep = self.ep
        rule = ep.wants.priority()
        stamp = f"{thing}|{ep.wants.version}|{ep.refusal_n}"
        if rule == "none" or stamp in ep.rank_checked:
            return ""
        if ep.time_left() < CLOSING + 30 or ep.moves_left() < 4:
            return ""
        ep.rank_checked.add(stamp)
        what = self.rank_wording()
        answer = self.ask_json(self.rank_prompt(thing, what, rule == "cheapest"), 2500, CHECKER)
        ep.log.append(dict(kind="rank", result=answer))
        gone = ep.shop.dead_lines()
        named = [str(x) for x in (answer or {}).get("better") or [] if isinstance(x, (str, int))]
        named = [x for x in named if x != thing and x not in ep.refused_lines and x not in gone]
        rivals = list(filter(lambda x: not self.field_view(x)["fails"], named[:4]))
        self.rank_better = rivals[:]
        self.named.update(rivals)
        if not rivals:
            return ""
        why = f" ({answer.get('reason')})" if answer.get("reason") else ""
        return (f"A ranking pass thinks these listings fit as well and top {thing} on the {what}: "
                + ", ".join(rivals) + why + ". Look at them (details, stock, price) before ordering; if all fail, order the current pick.")

    def unopened_hits(self, cap, qty, shelves, hit_shelf):
        ep = self.ep
        known = ep.known
        scores = ep.scout.hit_score
        ordered = sorted(ep.scout.all_hits, key=lambda ident: -scores.get(ident, 0))

        def keep(ident):
            if ident in known.read:
                return False
            shelf = hit_shelf.get(ident)
            if shelves and shelf and shelf not in shelves:
                return False
            price = known.price(ident)
            return not (is_number(cap) and is_number(price) and price * qty > cap)

        return [ident for ident in ordered if keep(ident)]

    def audit_prompt(self, cards, fresh):
        ep = self.ep
        out = [BEFORE_DECLINE, "\n\nSHOPPER REQUEST TEXT:\n", ep.request, ep.wants.changes(),
               "\n\nCURRENT REQUIREMENTS:\n", ep.wants.text_body(), ep.pointers.ref_note(),
               "\n\nLISTINGS SEEN (one card each; DETAILS = record opened, ROW = known from search rows alone):\n",
               "\n".join(cards)]
        if fresh:
            out.append("\n\nNEWLY OPENED AFTER THE PRIOR AUDIT (their DETAILS cards are new: go through them prior to "
                       "the rest, field by field, using the reading rules): " + ", ".join(fresh))
        out.append(self.field_names())
        return "".join(out)

    def page_calls(self):
        ep = self.ep
        if ep.moves_left() <= 5:
            return []
        return ep.lookout.open_pages(2)

    def audit_leads(self, unopened):
        ep = self.ep
        known = ep.known
        shelves = ep.lookout.shelf_labels()
        cap, qty = ep.wants.spend_cap, ep.wants.quantity()
        hits = self.unopened_hits(cap, qty, shelves, ep.lookout.hit_shelf)
        try:
            slack = list(filter(lambda ident: ident not in known.read, ep.shop.slack_rows()))
        except Exception:
            slack = []
        merged = slack + hits[:4] + [ident for ident in unopened if ident not in hits]
        unopened = list(dict.fromkeys(merged))[:6]
        pages = self.page_calls()
        signature = lambda call: json.dumps(call, sort_keys=True, default=str)
        if pages:
            unopened = unopened[:4]
            self.named_pages.update(map(signature, pages))
        combos = []
        try:
            if ep.moves_left() > 5:
                combos = ep.bargain.combined_queries()
        except Exception as err:
            ep.log.append(dict(kind="combo_error", error=f"{type(err).__name__}: {err}"[:300]))
            combos = []
        if combos:
            self.named_pages.update(map(signature, combos))
        return unopened, pages, combos

    def audit_note(self, answer, fits, unopened, pages, combos):
        search = self.ep.role('search')

        def calls(found):
            return "; ".join(f"{search} {json.dumps(p, ensure_ascii=False)}" for p in found)

        said = []
        if fits:
            why = f" ({answer.get('reason')})" if answer.get("reason") else ""
            said.append("Another reviewer thinks these listings might meet all of the requirements: " + ", ".join(fits) + why + ".")
        if pages:
            said.append("A few searches with required values returned more results than were read; fits might lie "
                        "sitting on pages nobody has read yet: " + calls(pages) + (". Read those pages before anything else (one step brings a full page of rows), then open the "
                                                                            "in-budget rows of the kind asked for."))
        if combos:
            said.append("No search has yet joined the required values jointly (if the store needs every word, each "
                        "such search gives only the listings carrying all of them): " + calls(combos) + (". Run them (the search tool still allows them), then open the in-budget rows they list for the kind "
                                                                                                   "wanted here."))
        if unopened:
            said.append("Listings of the requested kind seen only as search rows, not once opened, so no one checked "
                        "their fields yet: " + ", ".join(unopened) + ". Open their details ahead of all (many per call).")
        return " ".join(said) + " Look them over (details, stock, price) and buy the one that fits; decline only if they all fail."

    def audit_allowed(self):
        ep = self.ep
        if ep.audit_done:
            limit = self.audit_reads if self.audit_reads is not None else 10 ** 9
            if self.audits >= 2 or len(ep.known.read) <= limit:
                return False
        return ep.time_left() >= CLOSING + 30

    def review_decline(self, early=False):
        ep = self.ep
        known = ep.known
        if not self.audit_allowed():
            return ""
        ep.audit_done = True
        self.audits += 0 if early else 1
        earlier = self.audit_ids
        skip = () if earlier is None else set(known.read) - set(earlier or ())
        leads = self.unopened_hits(ep.wants.spend_cap, ep.wants.quantity(), ep.lookout.shelf_labels(), ep.lookout.hit_shelf)
        cards = known.cards(16000, "", ep.pointers.ref_foreign(), set(skip) | set(leads[:6]))
        self.audit_reads = len(known.read)
        if not cards:
            return ""
        fresh = sorted(known.read - earlier) if earlier is not None else []
        self.audit_ids = set(known.read)
        answer = self.ask_json(self.audit_prompt(cards, fresh), 6000, AUDITOR)
        ep.log.append(dict(kind="audit", result=answer))
        reply = answer or {}
        fits = [str(x) for x in reply.get("possible_fits") or [] if isinstance(x, (str, int))][:5]
        self.audit_fits = fits = [x for x in fits if not self.field_view(x)["fails"]]
        offered = [str(x) for x in reply.get("unopened") or []
                   if isinstance(x, (str, int)) and str(x) not in known.read and str(x) in known.ids]
        unopened, pages, combos = self.audit_leads(offered[:6])
        found = fits + unopened
        self.named.update(found)
        self.named_list.extend([ident for ident in found if ident not in self.named_list])
        if not (fits or unopened or pages or combos):
            return ""
        return self.audit_note(answer, fits, unopened, pages, combos)

    def affordable(self, ident, cap, qty):
        price = self.ep.known.price(ident)
        return not (is_number(cap) and is_number(price) and price * qty > cap)

    def named_unread(self):
        ep = self.ep
        known = ep.known
        cap, qty = ep.wants.spend_cap, ep.wants.quantity()
        shelves, hit_shelf = ep.lookout.shelf_labels(), ep.lookout.hit_shelf
        scores = ep.scout.hit_score
        pool = sorted(ep.scout.all_hits, key=lambda i: -scores.get(i, 0)) + list(self.named_list)
        picked = []
        for ident in dict.fromkeys(pool):
            if ident in known.read or ident in ep.refused_lines or ident not in known.ids:
                continue
            if not self.affordable(ident, cap, qty):
                continue
            shelf = hit_shelf.get(ident)
            if shelves and shelf and shelf not in shelves:
                continue
            picked.append(ident)
        return picked

    def details_call(self, ident):
        ep = self.ep
        tool = ep.role("details")
        known = ep.known.ids
        envs = (entry for entry in reversed(ep.log) if entry.get("kind") == "env")
        for entry in envs:
            for act in entry.get("actions") or []:
                args = act.get("args")
                if act.get("name") != tool or not isinstance(args, dict):
                    continue
                naming = [k for k, v in args.items() if isinstance(v, str) and v in known]
                if naming:
                    return dict((k, ident if k in naming else v) for k, v in args.items())
        schema = ep.shapes.get(tool or "") or {}
        props = schema.get("properties")
        props = props if isinstance(props, dict) else {}
        texts = [k for k, p in props.items() if isinstance(p, dict) and p.get("type") == "string" and not p.get("enum")]
        required = schema.get("required") or []
        chosen = [k for k in texts if k in required]
        slot = chosen[0] if chosen else (texts[0] if len(texts) == 1 else None)
        return {slot: ident} if slot else None

    def read_calls(self, queue, batch, reader):
        if batch:
            tool, arg, most = batch
            return tool, [{arg: queue[at: at + most]} for at in range(0, len(queue), most)]
        tool = reader or self.ep.role("details")
        if not tool:
            return tool, []
        built = map(lambda ident: self.ep.shop.lookup_call(ident, tool), queue)
        return tool, [call for call in built if call]

    def open_reads(self, queue, say, batch, reader=None):
        ep = self.ep
        tool, calls = self.read_calls(queue, batch, reader)
        tick = ep.shop.tick
        proposals = [dict(id=f"open_{n}_{tick}", type="function", function=dict(name=tool, arguments=json.dumps(args)))
                     for n, args in enumerate(calls)]
        if not proposals:
            return []
        ep.chat.append({"role": "assistant", "content": say, "tool_calls": proposals})
        start = len(ep.known.items)
        ep.covering = True
        try:
            replies = ep.dispatch(proposals, scripted=True)
        finally:
            ep.covering = False
        self.last_reads = ep.known.items[start:]
        return replies

    def pull_queue(self):
        ep = self.ep
        reads = ep.reads
        queue, reads.open_queue = reads.open_queue, None
        if queue or not reads.put_queue or ep.shop.done or ep.shop.broke:
            return queue, False
        queue, reads.put_queue = reads.put_queue, None
        tool, reads.put_tool = reads.put_tool, None
        reads.open_recheck, reads.open_purpose, reads.open_tool = True, "cart", tool
        reads.open_say = "Reading the stock of the line just added again, as an item can sell out right as it enters the cart."
        return queue, True

    def keep_order_room(self, queue, purpose):
        ep = self.ep
        try:
            reserve = FIRST_ORDER_BY if ep.orders_tried else ep.brief.order_due()
            room = ep.moves_left() - reserve - 1
            batched = bool(ep.shop.batch_tool()) and len(queue) > 1
            short = room < (1 if batched else len(queue))
            if short and (ep.basket.units or not ep.brief.unverified()):
                cut = queue[: max(0, room)]
                if ep.shop.batch_tool() and len(queue) >= 2:
                    cut = []
                ep.log.append(dict(kind="reserve_skip", purpose=purpose, ids=queue[:6], kept=cut))
                return cut
        except Exception as err:
            ep.log.append(dict(kind="reserve_skip_error", error=f"{type(err).__name__}: {err}"[:300]))
        return queue

    def take_flags(self):
        reads = self.ep.reads
        flags = dict(recheck=reads.open_recheck, purpose=str(reads.open_purpose or ""), floor=reads.open_floor,
                     reader=reads.open_tool, say=reads.open_say)
        reads.open_recheck = False
        reads.open_purpose = ""
        reads.open_floor = None
        reads.open_tool = None
        reads.open_say = ""
        if not flags["say"]:
            flags["say"] = ("A last look at the lines in the cart before ordering." if flags["recheck"]
                            else "Some listings came up in the checks; I open them before declining.")
        return flags

    def trimmed_queue(self, queue, flags, from_put):
        ep = self.ep
        purpose = flags["purpose"]
        busy = purpose in ("menu", "quota", "walk")
        if busy and ep.role("order") and ep.role("cart") and not ep.counsel.mute:
            queue = self.keep_order_room(queue, purpose)
        elif purpose == "cart":
            stale = set(ep.shop.stale_lines())
            queue = [i for i in queue if i in stale or (not from_put and i not in ep.basket.units)]
        return queue

    def usable_batch(self, queue, flags):
        ep = self.ep
        shop = ep.shop
        if len(queue) < 2:
            return None
        batch = shop.batch_tool()
        if not batch:
            return batch
        if flags["purpose"] == "cart" and not ep.batch_proven:
            return None
        if flags["reader"] and not ((shop.tool_caps or {}).get(batch[0]) or {}).get("stock"):
            return None
        return batch

    def batch_fallback(self, queue, flags, batch, replies):
        ep = self.ep
        shop = ep.shop
        if shop.batch_ok(self.last_reads or [], queue):
            ep.batch_proven = True
            return replies, False
        ep.batch_spec = None
        ep.batch_spec_known = True
        ep.log.append(dict(kind="batch_dropped", tool=batch[0]))
        purpose, floor, reader = flags["purpose"], flags["floor"], flags["reader"]
        most = {"menu": 0, "quota": 2, "walk": 2}.get(purpose, 3)
        if is_number(floor):
            reserve = int(floor)
        else:
            reserve = ep.brief.order_due() if purpose in ("menu", "quota") else 3
        count = min(most, len(queue), ep.moves_left() - reserve - 1)
        if count <= 0 or shop.done or not (reader or ep.role("details")):
            return replies, False
        ep.talk.post_replies(replies)
        replies = self.open_reads(queue[:count], flags["say"], None, reader)
        return replies, not replies

    def open_named(self):
        ep = self.ep
        shop = ep.shop
        queue, from_put = self.pull_queue()
        if not queue or shop.done or shop.broke:
            return False
        queue = list(dict.fromkeys(map(str, queue)))
        flags = self.take_flags()
        queue = self.trimmed_queue(queue, flags, from_put)
        if not queue:
            return False
        batch = self.usable_batch(queue, flags)
        replies = self.open_reads(queue, flags["say"], batch, flags["reader"])
        if batch and replies and not ep.batch_proven:
            replies, emptied = self.batch_fallback(queue, flags, batch, replies)
            if emptied:
                return True
        if not replies:
            return False
        note = ""
        if ep.time_left() > CLOSING + 30 and not flags["recheck"]:
            note = self.review_decline()
        if note:
            cid, body = replies[-1]
            replies[-1] = (cid, body + "\n[check after opening] " + note)
        ep.talk.post_replies(replies)
        return True

    @staticmethod
    def dicts_in(pieces):
        found = []
        for piece in pieces:
            walk_dicts(piece, found.append)
        return found

    def last_stock(self, ident):
        lookout = self.ep.lookout
        state = None
        for row in self.dicts_in(self.ep.known.items):
            if lookout.row_mark(row) != ident:
                continue
            seen = lookout.stock_state(row)
            if seen is not None:
                state = seen
        return state

    def worth_checking(self, x, version, cap):
        ep = self.ep
        blocked = (x not in ep.known.read or f"{x}|{version}" in ep.passed or x in ep.refused_lines
                   or x in ep.shop.dead_lines() or self.last_stock(x) is False or self.field_view(x)["fails"])
        if blocked:
            return False
        lines = ep.wants.items()
        slot = ep.bargain.product_index(x) if lines else None
        qty = ep.wants.quantity()
        if slot is not None and slot < len(ep.wants.items()):
            qty = ep.wants.items()[slot]["quantity"]
        price = ep.known.price(x)
        if not is_number(price):
            return False
        return not (is_number(cap) and is_number(qty) and price * qty > cap)

    def rank_verify(self, thing):
        ep = self.ep
        try:
            if ep.wants.priority() != "cheapest":
                return ""
            version = str(ep.wants.version)
            cap = ep.wants.spend_cap
            done = 0
            for x in self.rank_better or []:
                if done >= 2 or ep.time_left() < CLOSING + 30:
                    break
                if not self.worth_checking(x, version, cap):
                    continue
                done += 1
                self.check_item(x, "ORDER")
            return self.cheaper_fit(thing)
        except Exception as err:
            ep.log.append({"kind": "rank_verify_error", "error": f"{type(err).__name__}: {err}"[:300]})
            return ""

    def cheaper_fit(self, thing):
        ep = self.ep
        if not thing or ep.wants.priority() != "cheapest":
            return ""
        ceiling = ep.known.price(thing)
        if ceiling is None:
            return ""
        label = self.shelf_known(thing)
        gone = ep.shop.dead_lines()
        version = str(ep.wants.version)
        several = ep.wants.items()
        best = ("", None)
        for key in ep.passed:
            ident, _, ver = key.rpartition("|")
            excluded = ver != version or ident == thing or ident in ep.refused_lines or ident in gone
            if excluded:
                continue
            if several and (not label or self.shelf_known(ident) != label):
                continue
            cost = ep.known.price(ident)
            if cost is None or cost >= ceiling:
                continue
            if best[1] is not None and cost >= best[1]:
                continue
            if not self.field_view(ident)["fails"]:
                best = (ident, cost)
        return best[0]

    def shelf_known(self, thing):
        lookout = self.ep.lookout
        label = ""
        for row in self.dicts_in(self.ep.known.items):
            for key, val in row.items():
                if isinstance(val, str) and val == thing and "id" in str(key).lower():
                    label = lookout.shelf_of(row) or label
                    break
        if label:
            return label
        return str((self.ep.known.label_of or {}).get(thing) or "").strip().lower()

    def chosen(self):
        units = self.ep.basket.units
        if not units:
            return None
        return list(units)[-1]


class Talk:
    def __init__(self, ep):
        self.ep = ep
        self.urging = False
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
        self.rank_memo = None

    @staticmethod
    def flat(text):
        return " ".join(re.findall(r"[a-z0-9]+", str(text).lower()))

    @staticmethod
    def plain_question(prop, product):
        return f"Which {prop.split('(')[0].strip()} would you like for the {product}?"

    def text_slot(self, tool):
        schema = self.ep.shapes.get(tool or "") or {}
        required = list(schema.get("required") or [])
        strings = []
        for name, spec in (schema.get("properties") or {}).items():
            if not isinstance(spec, dict) or "enum" in spec:
                continue
            kind = spec.get("type")
            if kind == "string" or "string" in (kind or []):
                strings.append(name)
        slot = None
        for name in required:
            if name in strings:
                slot = name
                break
        if not slot and len(strings) == 1:
            slot = strings[0]
        if not slot:
            return None
        for name in required:
            if name != slot:
                return None
        return slot

    def message_call(self, text_body, cid):
        tool = self.ep.role("message")
        slot = self.text_slot(tool) if tool else None
        if not slot:
            return None
        args, faults = ArgCheck.execute({slot: text_body}, self.ep.shapes[tool])
        if faults:
            return None
        return {"id": cid, "type": "function", "function": {"name": tool, "arguments": json.dumps(args)}}

    def product_of(self, text):
        lines = self.ep.wants.items()
        if not lines:
            return None
        kinds = [set(kind_tokens(line["product"])) for line in lines]
        everything = range(len(kinds))
        for piece in re.findall(r"\(([^)]*)\)", str(text))[:1] + [str(text)]:
            words = set(kind_tokens(piece))
            scores = [kind_score(kinds[spot], words, set().union(*[kinds[o] for o in everything if o != spot]))
                      for spot in everything]
            best = max(scores)
            if best > 0 and scores.count(best) == 1:
                return scores.index(best)
        return None

    def item_slot(self, qitem):
        slot = self.product_of(str(qitem.get("property") or ""))
        if slot is not None:
            return slot
        return self.product_of(str(qitem.get("question") or ""))

    def basket_covered(self):
        lines = self.ep.wants.items()
        return not lines or all(self.asked_counts(len(lines))[1])

    def asked_counts(self, count):
        answered = [0] * count
        sent = [0] * count
        for line in self.ep.questions_log:
            head = line.split(":", 1)[0]
            slot = self.product_of(head)
            if slot is None:
                quoted = re.search(r'asked "(.*?)" -> ', line)
                if quoted:
                    slot = self.product_of(quoted.group(1))
            if slot is None:
                continue
            sent[slot] += 1
            if "[not answered]" not in head:
                answered[slot] += 1
        return answered, sent

    def hinted_slots(self, count, answered):
        brief = self.ep.brief
        try:
            brief.open_hints()
        except Exception:
            pass
        try:
            named = set()
            keyless = [0] * count
            for hint in brief.open_hints():
                head = str(hint).split(":", 1)[0]
                slot = self.product_of(head)
                if slot is None:
                    continue
                if brief.plain_hint(slot, head.strip(), hinted=True):
                    keyless[slot] += 1
                else:
                    named.add(slot)
            return {j for j in range(count) if j in named or keyless[j] > answered[j]}
        except Exception:
            return set()

    def lone_second(self, scripted, due, lines):
        for qitem in scripted:
            second = str(qitem.get("second") or "").strip()
            if not second:
                continue
            slot = self.product_of(second)
            if slot not in due:
                continue
            tied = re.search(r"\(([^)]+)\)", second)
            product = tied.group(1).strip() if tied else lines[slot]["product"]
            return {"property": second, "question": f"Which {second.split('(')[0].strip()} would you like for the {product}?"}
        return None

    def quota_levels(self, count, answered, sent, revealed, hinted, alt_left):
        unanswered = [j for j in range(count) if not answered[j]]
        levels = {}
        covered = all(sent) or self.ep.route == "single"
        for k in range(count):
            first = 0 if covered or not sent[k] else 1
            waits = 0
            if revealed[k] and any(j != k for j in unanswered):
                waits = 1
            silent = 0 if k in alt_left else sent[k] - answered[k]
            levels[k] = (first, waits, 0 if k in hinted else 1, answered[k] + revealed[k], silent, revealed[k])
        return levels

    def quota_try(self, due, scripted, hinted, answered, ok):
        brief = self.ep.brief
        for k in sorted(due, key=lambda k: (k not in hinted, k)):
            try:
                pick = self.quota_pick(k, answered[k] == 0, ok)
            except Exception:
                pick = None
            if pick and ok is not None:
                if not ok(pick.get("property")) or brief.asked_for(k, pick.get("property")):
                    pick = None
            if pick:
                return [pick] + scripted
            if self.quota_wait:
                return scripted
        return None

    def revealed_counts(self, count):
        revealed = [0] * count
        for need in self.ep.wants.uncovered:
            slot = self.product_of(str(need).split(":", 1)[0])
            if slot is not None:
                revealed[slot] += 1
        return revealed

    def quota_first(self, scripted, ok=None):
        ep = self.ep
        lines = ep.wants.items()
        if not lines:
            return scripted
        count = len(lines)
        answered, sent = self.asked_counts(count)
        if scripted and max(answered) == 0 and self.item_slot(scripted[0]) is None:
            return scripted
        revealed = self.revealed_counts(count)
        hinted = self.hinted_slots(count, answered)
        alt_left = set(j for j in range(count) if sent[j] > answered[j] and ep.brief.alt_untried(j))
        levels = self.quota_levels(count, answered, sent, revealed, hinted, alt_left)
        for level in sorted(set(levels.values())):
            due = [k for k, value in sorted(levels.items()) if value == level]
            spot = next((n for n, qitem in enumerate(scripted) if self.item_slot(qitem) in due), None)
            if spot is not None:
                return [scripted[spot]] + scripted[:spot] + scripted[spot + 1:]
            alone = self.lone_second(scripted, due, lines)
            if alone:
                return [alone] + scripted
            ordered = self.quota_try(due, scripted, hinted, answered, ok)
            if ordered is not None:
                return ordered
        return scripted

    def queue_quota_reads(self, k, home, product):
        ep = self.ep
        batch = ep.shop.batch_tool()
        unread = []
        if home:
            size = min(4, batch[2]) if batch else 2
            unread = ep.shop.menu_rows(home, size, [], ep.wants.items()[k]["quantity"])
        ready = unread and (ep.role("details") or batch) and not self.quota_reads.get(k) and ep.moves_left() > 12
        if not ready:
            return
        self.quota_reads[k] = True
        self.quota_wait = True
        reads = ep.reads
        reads.open_queue, reads.open_recheck, reads.open_purpose = unread, True, "quota"
        reads.open_say = f"First I open a few {product} listings to learn their fields; the question on it follows."

    def home_labels(self, k, home):
        known = self.ep.known
        labels = [label for label in known.shelves_read() if label.strip().lower() in home]
        if labels or not home or not self.quota_reads.get(k):
            return labels
        return [label for label in sorted(home) if known.prop_menu(label)[0]][:1]

    def menu_property(self, k, product, menu, fallback, good):
        brief = self.ep.brief
        try:
            hints = brief.open_hints(k)
        except Exception:
            hints = []
        for hint in hints:
            prop = re.split(r"\s*\(", hint.split(":", 1)[0])[0].strip()
            if not prop or junk_key(prop) or brief.plain_hint(k, prop, hinted=True):
                continue
            if good(f"{prop} ({product})"):
                return prop
        if fallback:
            for weigh in (False, True):
                for entry in menu.splitlines():
                    prop = entry.strip("- ").split(" (in ")[0].strip()
                    if not prop or not self.open_prop(k, prop, entry, by_samples=weigh):
                        continue
                    if good(f"{prop} ({product})"):
                        return prop
        return None

    def quota_pick(self, k, fallback=True, ok=None):
        ep = self.ep
        brief, known = ep.brief, ep.known
        product = ep.wants.items()[k]["product"]
        homes = ep.shop.home_shelves()
        home = set() if k >= len(homes) else homes[k]
        labels = self.home_labels(k, home)
        if not labels:
            return self.queue_quota_reads(k, home, product)
        shelf = labels[0]
        menu = brief.menu_trim(known.prop_menu(shelf)[0], known.rich_reads(shelf))

        def good(label):
            return ok is None or (bool(ok(label)) and not brief.asked_for(k, label))

        planned = self.product_plan(k, menu)
        if planned and good(planned.get("property")):
            return planned
        prop = self.menu_property(k, product, menu, fallback, good)
        if prop is not None:
            return dict(property=f"{prop} ({product})", question=f"Which {prop} would you like for the {product}?")
        return None

    def product_plan(self, k, menu):
        try:
            return self.ask_product_plan(k, menu)
        except Exception:
            return None

    def ask_product_plan(self, k, menu):
        ep = self.ep
        brief = ep.brief
        if not menu or self.product_plans >= 1 or ep.time_left() < CLOSING + 40:
            return None
        self.product_plans += 1
        product = ep.wants.items()[k]["product"]
        shout = product.upper()
        count = min(3, max(1, ep.ask_limit() - ep.msgs_sent))
        blocks = [ASK_PLAN.replace("QUESTION_COUNT", str(count)), "\n\nSHOPPER REQUEST TEXT:\n" + ep.request,
                  "\n\nREQUIREMENTS KNOWN SO FAR:\n" + ep.wants.text_body()]
        if ep.questions_log:
            blocks.append("\n\nQUESTIONS ALREADY PUT (each with the shopper's answer):\n" + "\n".join(ep.questions_log))
        blocks.append(f"\n\nTHIS ROUND ASKS ONLY ABOUT THE {shout}: each other product was already asked about, "
                      f"this one not at all. Choose just {product} properties from its list below, "
                      "beginning with whichever one the details of the story hint at.")
        try:
            hints = brief.open_hints(k)
        except Exception:
            hints = []
        if hints:
            marked = [f"- {brief.hint_mark(h)}" for h in hints[:6]]
            blocks.append("\n\nSTORY HINTS (bits of the story in the request, each with the property it probably refers to, likeliest "
                          "first; none of them is a requirement):\n" + "\n".join(marked))
        blocks.append(self.plan_notes(k))
        blocks.append(f"\n\nPROPERTIES FOUND IN DETAILS OF {shout} LISTINGS SEEN SO FAR (name, number of listings, "
                      "example values):\n" + menu)
        answer = ep.counsel.ask_json("".join(blocks), 1500, PLANNER[:1])
        ep.log.append(dict(kind="product_plan", product=product, result=answer))
        return self.choose_from_plan(k, product, answer)

    @staticmethod
    def no_second(text):
        return bool(re.search(r"none|^-$|n/a", text.lower()))

    def choose_from_plan(self, k, product, answer):
        brief = self.ep.brief
        words = kind_words(product)
        asked = lambda prop: f"Which {prop} would you like for the {product}?"
        for qitem in filter(lambda q: isinstance(q, dict), (answer or {}).get("questions") or []):
            first = str(qitem.get("property") or "").split("(")[0].strip()
            second = str(qitem.get("second") or "").split("(")[0].strip()
            second = "" if self.no_second(second) else second
            question = str(qitem.get("question") or "").strip()
            first = "" if brief.plain_hint(k, first) else first
            if second and brief.plain_hint(k, second):
                second = ""
            if first and brief.plan_open(k, first):
                names_product = bool(question) and bool(words & kind_words(question))
                pair = ""
                if names_product and second and not strict_prop(first, second) and brief.plan_open(k, second):
                    pair = second
                return dict(property=f"{first} ({product})", second=f"{pair} ({product})" if pair else "",
                            question=question if names_product else asked(first))
            if second and brief.plan_open(k, second):
                return dict(property=f"{second} ({product})", question=asked(second))
        return None

    def requirement_settles(self, req, name, slot, samples, by_samples):
        brief = self.ep.brief
        head, _, value = str(req).partition(":")
        if strict_prop(head, name):
            return bool(brief.settles(str(req), samples if by_samples else ""))
        positive = positive_value(value, head)
        if same_prop(head, name):
            return positive and bool(value_seen(value, samples or " | ".join(brief.field_values(slot, name))))
        tokens = [t for t in re.findall(r"[a-z]{3,}", value.lower()) if t not in LINK_WORDS]
        usable = samples and tokens and not re.search(r"\d", value) and positive
        return bool(usable) and all(re.search(r"\b" + re.escape(t), samples) for t in tokens)

    def open_prop(self, k, prop, line="", by_samples=False):
        ep = self.ep
        brief = ep.brief
        try:
            name = str(prop).split("(")[0].strip()
            if not name or junk_key(name):
                return False
            several = bool(ep.wants.items())

            def about_k(text, tagged_only=False):
                if k is None or not several:
                    return True
                slot = self.product_of(text)
                return slot == k or (slot is None and not tagged_only)

            answered_heads = [logged.split(":", 1)[0].strip("- ") for logged in ep.questions_log]
            for head in answered_heads:
                if "[not answered]" not in head and any(
                        about_k(part) and brief.same_field(k, part, name) for part in head.split(";")):
                    return False
            samples = line.split("):", 1)[1].lower() if "):" in line else ""
            settled = any(about_k(str(req).partition(":")[0]) and self.requirement_settles(req, name, k, samples, by_samples)
                          for req in ep.wants.props_left())
            if settled:
                return False
            return not any(about_k(str(free), tagged_only=True) and brief.same_field(k, str(free), name)
                           for free in ep.wants.indifferent)
        except Exception:
            return True

    def plan_notes(self, k):
        ep = self.ep
        try:
            several = bool(ep.wants.items())
            asked = set()
            for line in ep.questions_log:
                asked |= {part.strip().lower() for part in line.split(":", 1)[0].strip("- ").split(";")}

            def mine(text):
                return k is None or not several or self.product_of(text) in (k, None)

            still_open = []
            for prop in self.open_pair or []:
                slot = self.product_of(prop) if several else None
                if k is not None and several and slot not in (k, None):
                    continue
                if prop.lower() not in asked and ep.brief.plan_open(slot, prop):
                    still_open.append(prop)
            wished = list(filter(mine, self.wanted_asks or []))
            pieces = []
            if still_open:
                pieces.append("\n\nSTILL OPEN AFTER THE PREVIOUS PAIR (the second part of planned pairs got no reply, since the "
                              "shopper replied to the first alone or just the first went out; they remain open, and each was the "
                              "planner's next choice):\n" + "\n".join("- " + p for p in still_open[-3:]))
            if wished:
                pieces.append("\n\nQUESTIONS THE ASSISTANT MEANT TO ASK (drafted once listing records had been read; when one names a "
                              "property that is still open and also a key below, ask that key first, under its own name):\n"
                              + "\n".join("- " + w for w in wished[-3:]))
            return "".join(pieces)
        except Exception:
            return ""

    def want_ask(self, raw_args):
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
            others = [w for w in self.wanted_asks or [] if w.lower() != text.lower()]
            others.append(text)
            self.wanted_asks = others[-3:]
        except Exception:
            pass

    def forget_wanted(self, label):
        stem = self.ep.lookout.stem
        try:
            cleaned = str(label).replace("[not answered]", "")
            props = [re.split(r"\s*\(", p)[0] for p in re.split(r";|\band\b", cleaned)]

            def covered(text):
                words = {stem(w) for w in re.findall(r"[a-z]+", text.lower())}
                for p in props:
                    bits = prop_words(p)
                    if bits and all(w in words or stem(w) in words for w in bits):
                        return True
                return False

            self.wanted_asks = [text for text in self.wanted_asks or [] if not covered(text)]
        except Exception:
            pass

    def question_words(self, prop):
        wants = self.ep.wants
        words = set(prop_words(prop)) | {"which", "would", "you", "like"}
        for line in wants.items() or [{"product": str(wants.reading.get("product") or "")}]:
            for token in re.findall(r"[a-z0-9]+", str(line.get("product") or "").lower()):
                words.add(token)
                words.add(token[:-1] if len(token) > 3 and token.endswith("s") else token)
        return words

    @staticmethod
    def heard_words(question):
        spoken = " ".join(re.findall(r"[a-z0-9]+", str(question).lower()))
        heard = set(spoken.split())
        heard |= {t[:-1] for t in heard if len(t) > 3 and t.endswith("s") and not t.endswith("ss")}
        return spoken, heard

    def example_named(self, entry, spoken, mine):
        if "):" not in entry:
            return False
        for example in entry.split("):", 1)[1].split("|"):
            tokens = re.findall(r"[a-z0-9]+", example.lower())
            outside = [t for t in tokens if t not in mine and t not in LINK_WORDS]
            if not outside or len("".join(tokens)) < 3:
                continue
            if re.search(r"(?<![a-z0-9])" + re.escape(" ".join(tokens)) + r"(?![a-z0-9])", spoken):
                return True
        return False

    def names_other(self, question, prop, menu, second=""):
        try:
            spoken, heard = self.heard_words(question)
            mine = self.question_words(prop)

            def mentions(field):
                core = [t for t in prop_words(field) if len(t) >= 3 and t not in LINK_WORDS]
                return bool(core) and set(core) <= heard and not set(core) <= mine

            if second and not strict_prop(second, prop) and mentions(second):
                return True
            for entry in str(menu or "").splitlines():
                field = entry.strip().lstrip("-").strip().split(" (in ")[0].strip()
                skip = not field or field.endswith(":") or strict_prop(field, prop) or junk_key(field)
                if skip:
                    continue
                if mentions(field) or self.example_named(entry, spoken, mine):
                    return True
            return False
        except Exception:
            return False

    def pair_fits(self, first, second):
        ep = self.ep
        brief = ep.brief
        try:
            slot = None
            if ep.wants.items():
                slot = self.product_of(first)
                if slot is None or self.product_of(second) != slot:
                    return False
            if strict_prop(first, second) or same_prop(first, second) or brief.same_field(slot, first, second):
                return False
            keep = lambda prop: set(w for w in prop_words(prop) if len(w) >= 3 and w not in LINK_WORDS)
            if keep(first) & keep(second):
                return False
            if brief.field_values(slot, first) & brief.field_values(slot, second):
                return False
            menu = (brief.menu_memo.get(slot) or (0, ""))[1]
            keys = [line.strip("- ").split(" (in ")[0].strip() for line in menu.splitlines()]
            for key in keys:
                if key and not strict_prop(key, first) and not strict_prop(key, second):
                    if brief.twin_of(slot, key, second) and brief.plan_open(slot, key):
                        return False
            return True
        except Exception:
            return False

    def queue_second(self, prop):
        if not prop:
            return
        if prop.lower() in [p.lower() for p in self.pending_seconds]:
            return
        self.pending_seconds.append(prop)

    def product_tag(self, label, fallback=None):
        tied = re.search(r"\(([^)]+)\)", label)
        if tied:
            return tied.group(1).strip()
        if fallback is None:
            return str(self.ep.wants.reading.get("product") or "item")
        return fallback

    def ask_later(self, sets, pressing, urge, ceiling):
        ep = self.ep
        tick = ep.shop.tick
        thin = not (pressing or urge) and sets < ASK_AFTER_SETS
        too_soon = tick < ASK_AFTER_STEPS or (tick < ASK_AFTER_STEPS + ASK_GRACE_STEPS and ep.moves_left() > 20)
        if thin and too_soon:
            return True
        closing = ep.moves_left() <= max(10, ep.brief.order_due()) or ep.time_left() < CLOSING + 40
        if closing:
            ep.asking = False
            return True
        if not urge and self.plan_wait is not None and sets <= self.plan_wait:
            return True
        if urge or pressing or self.asked_at_sets is None or sets > self.asked_at_sets:
            return False
        spare_moves = ep.moves_left() - ep.brief.order_due() - 2 * max(0, ceiling - ep.msgs_sent)
        if spare_moves <= 14:
            return False
        return not ep.wants.items() and ep.shop.tick - self.asked_at_tick < 4

    def shelf_menus(self, prop_list, shown):
        ep = self.ep
        known = ep.known
        brief = ep.brief
        labels = known.shelves_read()
        homes = ep.shop.home_shelves()
        every_home = set().union(*homes)
        own = [label for label in labels if label.strip().lower() in every_home]
        if all(homes):
            order = own or labels
        else:
            order = own + [label for label in labels if label not in own]
        filled = []
        for label in order[:4]:
            menu = known.prop_menu(label)[0]
            if menu:
                filled.append((label, menu))
        whole = "\n".join(f"CATEGORY {label}:\n{menu}" for label, menu in filled) or prop_list
        trimmed = "\n".join(f"CATEGORY {label}:\n{brief.menu_trim(menu, known.rich_reads(label))}"
                            for label, menu in filled) or shown
        return whole, trimmed

    def planner_prompt(self, remaining, prop_list, sets):
        ep = self.ep
        brief, wants = ep.brief, ep.wants
        text = ASK_PLAN.replace("QUESTION_COUNT", str(remaining)) + NEXT_ONE.replace("REMAINING", str(remaining))
        text += "\n\nSHOPPER REQUEST TEXT:\n" + ep.request
        text += "\n\nREQUIREMENTS KNOWN SO FAR:\n" + wants.text_body()
        if ep.questions_log:
            text += "\n\nQUESTIONS ALREADY PUT (each with the shopper's answer):\n" + "\n".join(ep.questions_log)
        must = wants.reading.get("ask_first")
        if must:
            text += "\n\nTO ASK THE SHOPPER ABOUT BEFORE ALL: " + "; ".join(map(str, must or []))
        text += self.plan_notes(None)
        try:
            hints = brief.open_hints()
        except Exception:
            hints = []
        if hints:
            text += ("\n\nSTORY HINTS (bits of the story in the request, each with the property it probably refers to; "
                     "none is a requirement, yet the hidden need tends to be one of them):\n"
                     + "\n".join(f"- {brief.hint_mark(h)}" for h in hints[:8]))
        shown = brief.menu_trim(prop_list, sets)
        lines = wants.items()
        if lines:
            listing = " | ".join(f"{n}) {line['product']} x{line['quantity']}" for n, line in enumerate(wants.items(), 1))
            text += "\n\nPRODUCTS REQUESTED: " + listing
            prop_list, shown = self.shelf_menus(prop_list, shown)
        text += ("\n\nPROPERTIES FOUND IN LISTING DETAILS SEEN SO FAR (name, number of listings, example values):\n"
                 + (shown or "(none seen yet)"))
        return text, prop_list

    def screen_plan(self, answer, fresh):
        several = bool(self.ep.wants.items())
        kept = []
        questions = [q for q in (answer or {}).get("questions") or []
                     if isinstance(q, dict) and str(q.get("question") or "").strip()]
        for qitem in questions:
            second = str(qitem.get("second") or "").strip()
            second = "" if self.no_second(second) else second
            if fresh(qitem.get("property")):
                entry = dict(qitem, second=second if second and fresh(second) else "")
                slot = self.product_of(second) if several else None
                if second and self.ep.brief.plain_hint(slot, second):
                    first = str(qitem.get("property") or "").strip()
                    entry["question"] = self.plain_question(first, self.product_tag(first))
                kept.append(entry)
            elif second and fresh(second):
                kept.append(dict(property=second, question=self.plain_question(second, self.product_tag(second))))
        return kept

    def name_the_product(self, thing):
        prop = str(thing.get("property") or "")
        product = self.product_tag(prop, "")
        words = re.findall(r"[a-z]{3,}", product.lower())
        if not product or not words:
            return thing
        asked = str(thing.get("question") or "").lower()
        for word in words:
            if word.rstrip("s") in asked:
                return thing
        return dict(thing, question=self.plain_question(prop, product))

    def paired_or_single(self, label, second, remaining):
        ep = self.ep
        first = label[0]
        slot = self.product_of(first) if ep.wants.items() else None
        cid = "ask_%d_%d" % (ep.msgs_sent + 1, ep.shop.tick)
        product = self.product_tag(first)
        if remaining == 1 and not self.no_pairs and self.pair_fits(first, second) and ep.brief.hint_mapped(slot, first):
            head, tail = first.split('(')[0].strip(), second.split('(')[0].strip()
            both = f"Which {head} would you like for the {product}, and which {tail}?"
            return self.message_call(both, cid), (first + "; " + second, both), ""
        alone = self.plain_question(first, product)
        return self.message_call(alone, cid), (first, alone), second

    def build_proposal(self, scripted, pending, prop_list, remaining):
        ep = self.ep
        wants = ep.wants
        strict_start = wants.reading.get("ask_first") and not ep.questions_log
        proposal = label = None
        split_second = ""
        for thing in scripted[:2]:
            thing = self.name_the_product(thing) if wants.items() else thing
            prop = str(thing.get("property") or "")
            if not (pending or strict_start) and self.names_other(str(thing.get("question") or ""), prop, prop_list,
                                                                  str(thing.get("second") or "")):
                thing = dict(thing, question=self.plain_question(prop, self.product_tag(prop)))
            question = str(thing.get("question")).strip()
            proposal = self.message_call(question, f"ask_{ep.msgs_sent + 1}_{ep.shop.tick}")
            if proposal is None:
                continue
            label = (str(thing.get("property")), question)
            second = str(thing.get("second") or "").strip()
            single = not second or second.lower() == label[0].lower() or self.no_second(second)
            if single:
                break
            proposal, candidate, split = self.paired_or_single(label, second, remaining)
            if proposal is not None:
                label, split_second = candidate, split
                break
        return proposal, label, split_second

    def menu_basis(self):
        known = self.ep.known
        whole_menu, whole_sets = known.prop_menu()[0], known.rich_reads()
        own = self.ep.shop.own_shelf()
        if own:
            return known.prop_menu(own)[0], known.rich_reads(own)
        return whole_menu, whole_sets

    def plan_answer(self, prompt, pending, sets):
        if pending:
            stored = (self.pending_text or {}).pop(pending.lower(), "")
            return {"questions": [{"property": pending,
                                   "question": stored or self.plain_question(pending, self.product_tag(pending))}]}
        if OUR_ASK:
            ours = self.our_plan()
            if ours is False:
                return False
            if ours.get("questions"):
                return ours
        counsel = self.ep.counsel
        answer = counsel.ask_json(prompt, 2500, PLANNER)
        if answer is None and counsel.misses > 0:
            self.plan_wait = sets
            return False
        return answer

    def shelf_facts(self, slot, basket, homes):
        ep = self.ep
        known = ep.known
        if basket:
            labels = sorted(homes[slot] if slot < len(homes) else [])
            return [found for label in labels for found in known.fact_sets(label)]
        home = (ep.shop.home_shelves() or [set()])[0]
        if not home and not self.urging and ep.moves_left() > ep.brief.order_due() + 12:
            ep.log.append({"kind": "our_plan", "wait": "no home shelf yet"})
            return None
        if home:
            return [found for label in sorted(home) for found in known.fact_sets(label)]
        return known.fact_sets(ep.shop.own_shelf())

    def settled_keys(self, slot, basket):
        settled = set()
        for line in self.ep.wants.written:
            head = line.partition(":")[0]
            if basket and self.product_of(head) != slot:
                continue
            settled.add(ask_name(head.split("(")[0]))
        return settled

    @staticmethod
    def ranked_queue(pick, menu):
        ranked = story = []
        if isinstance(pick, dict):
            ranked = list(map(str, pick.get("ranked") or []))
            story = list(map(str, pick.get("story") or []))
        return ask_merge(ranked[:ASK_FIRST_PICKS], story, [entry[0] for entry in menu])

    def our_plan(self):
        ep = self.ep
        wants = ep.wants
        names = [str(line["product"]) for line in wants.items()] or [str(wants.reading.get("product") or "item")]
        basket = len(names) > 1
        menus = self.plan_menus(names, basket, ep.shop.home_shelves() if basket else [])
        if menus is None:
            return False
        if basket and not all(menus):
            empty = [name for name, menu in zip(names, menus) if not menu]
            waiting = ep.moves_left() > ep.brief.order_due() + 12
            ep.log.append(dict(kind="our_plan", empty=empty, wait=waiting))
            return False if waiting else {}
        ranks = self.ask_ranks(names, menus)
        queues = [self.ranked_queue(ranks.get(f"P{slot + 1}") or {}, menu) for slot, menu in enumerate(menus)]
        sent = self.asked_counts(len(names))[1] if basket else [0]
        order = sorted(range(len(queues)), key=lambda slot: (sent[slot] if slot < len(sent) else 0, slot))
        questions = [self.plan_question(slot, key, menus, names, basket) for key, slot in self.round_robin(queues, order)]
        ep.log.append(dict(kind="our_plan", menus=[[entry[0] for entry in menu][:8] for menu in menus],
                           first=[q["property"] for q in questions[:4]]))
        return {"questions": questions}

    def plan_menus(self, names, basket, homes):
        menus = []
        for slot in range(len(names)):
            facts = self.shelf_facts(slot, basket, homes)
            if facts is None:
                return None
            menus.append(ask_menu(facts, self.settled_keys(slot, basket), basket)[:ASK_MENU])
        return menus

    @staticmethod
    def round_robin(queues, order):
        for spot in range(max(map(len, queues), default=0)):
            for slot in order:
                if spot < len(queues[slot]):
                    yield queues[slot][spot], slot

    def plan_question(self, slot, key, menus, names, basket):
        samples = next((found for listed, _, found in menus[slot] if listed == key), [])
        prop = ask_name(key)
        label = f"{prop} ({names[slot]})" if basket else prop
        return {"property": label, "question": self.our_question(prop, names[slot], samples, basket)}

    def ask_ranks(self, names, menus):
        asked = list(self.ep.questions_log)
        mark = (tuple(tuple(entry[0] for entry in menu[:12]) for menu in menus), len(asked))
        memo = self.rank_memo
        unranked = memo and memo[2] < ASK_RANKINGS + len(menus) and any(
            menu and not (memo[1] or {}).get(f"P{slot + 1}") for slot, menu in enumerate(menus))
        reuse = memo and (memo[0] == mark or (memo[2] >= ASK_RANKINGS and not unranked))
        if reuse or not any(menus):
            return memo[1] if memo else {}
        blocks = []
        for slot, name in enumerate(names):
            body = "\n".join("- %s (%s listings): %s" % (key, carriers, ", ".join(samples))
                             for key, carriers, samples in menus[slot])
            blocks.append("## P%d: %s\n" % (slot + 1, name) + body)
        so_far = "\n".join(asked) or "(none yet)"
        prompt = "".join([ASK_RANK, "\n\n# Shopper request\n", self.ep.request, "\n\n# Products\n", "\n\n".join(blocks),
                          "\n\n# QUESTIONS SO FAR\n", so_far])
        counsel = self.ep.counsel
        counsel.quiet = True
        try:
            answer = counsel.ask_json(prompt, 1500, ASK_SEAT) or {}
        finally:
            counsel.quiet = False
        used = memo[2] if memo else 0
        self.rank_memo = (mark, answer, used + 1)
        return answer

    @staticmethod
    def our_question(prop, product, samples, basket):
        bare = f"What {prop} do you need for the {product}?"
        if basket:
            return bare
        for count in (2, 1):
            shown = [value for value in samples if len(value) <= 24][:count]
            tail = f", such as {' or '.join(shown)}" if shown else ""
            worded = f"What {prop} do you need for the {product}" + tail + "?"
            if len(worded) <= ASK_CHARS:
                return worded
        return bare

    def nothing_to_ask(self, sets, urge):
        ep = self.ep
        if sets < ASK_AFTER_SETS and ep.moves_left() > 12 and not urge:
            self.plan_wait = sets
            return False
        self.empty_plans += 1
        if ep.moves_left() > 12 and self.empty_plans <= 1:
            self.plan_wait = sets
        else:
            ep.asking = False
        return False

    def send_question(self, proposal, label):
        ep = self.ep
        wants = ep.wants
        ep.chat.append({"role": "assistant", "content": f"Question for the shopper next: {label[0]}.", "tool_calls": [proposal]})
        counts = len(wants.uncovered), len(wants.indifferent) + wants.pinned_heard
        self.asking_label = label[0]
        try:
            replies = ep.dispatch([proposal], scripted=True)
        finally:
            self.asking_label = ""
        self.post_replies(replies)
        return replies, counts

    def asked_properties(self):
        names = set()
        for logged in self.ep.questions_log:
            head = logged.split(":", 1)[0].strip("- ")
            names |= {part.strip().lower() for part in head.split(";")}
        return names

    def freshness(self):
        wants, brief = self.ep.wants, self.ep.brief
        taken = self.asked_properties()

        def fresh(prop):
            text = str(prop or "").strip()
            low = text.lower()
            blocked = not text or low in taken or junk_key(text) or low in self.spent_asks
            if blocked:
                return False
            slot = None
            if wants.items():
                slot = self.product_of(text)
            return not brief.plain_hint(slot, text) and brief.plan_open(slot, text)
        return fresh

    def opening_move(self, urge):
        ep = self.ep
        ceiling = ep.ask_limit()
        if ep.msgs_sent >= ceiling or not ep.role("message"):
            ep.asking = False
            return False, None
        prop_list, sets = self.menu_basis()
        wants = ep.wants
        pressing = not ep.questions_log and bool(wants.reading.get("ask_first")) and not is_number(wants.spend_cap)
        if self.ask_later(sets, pressing, urge, ceiling):
            return False, None
        if ep.shop.pre_read():
            return True, None
        quiet_start = not (ep.questions_log or ep.msgs_sent or pressing or urge)
        if quiet_start and ep.brief.first_reads():
            return True, None
        return None, (ceiling, prop_list, sets)

    def pose_planned(self, urge=False):
        self.urging = urge
        verdict, setting = self.opening_move(urge)
        if setting is None:
            return verdict
        ceiling, prop_list, sets = setting
        ep = self.ep
        wants = ep.wants
        remaining = ceiling - ep.msgs_sent
        prompt, prop_list = self.planner_prompt(remaining, prop_list, sets)
        fresh = self.freshness()
        self.pending_seconds = list(filter(fresh, self.pending_seconds or []))
        pending = self.pending_seconds[0] if self.pending_seconds else ""
        answer = self.plan_answer(prompt, pending, sets)
        if answer is False:
            return False
        scripted = self.screen_plan(answer, fresh)
        if not pending and wants.items():
            scripted = self.quota_first(scripted, fresh)
            held = self.quota_wait
            self.quota_wait = False
            if held:
                return True
        proposal, label, split_second = self.build_proposal(scripted, pending, prop_list, remaining)
        if not proposal:
            return self.nothing_to_ask(sets, urge)
        self.asked_at_sets = sets
        self.asked_at_tick = ep.shop.tick
        replies, before = self.send_question(proposal, label)
        self.record_question(label, replies, before, pending, split_second)
        ep.asking = ep.msgs_sent < ceiling
        return True

    def record_question(self, label, replies, before, pending, split_second):
        ep, wants = self.ep, self.ep.wants
        asked_label = label[0]
        lead_part = asked_label.split(";")[0].strip()
        if pending and lead_part.lower() == pending.lower():
            gone = pending.lower()
            self.pending_seconds = [p for p in self.pending_seconds or [] if p.lower() != gone]
        heard_meta = wants.meta_last
        wants.meta_last = None
        found_before, freed_before = before
        label, note = self.read_answer(label, heard_meta, found_before, freed_before, split_second)
        if heard_meta is None and found_before == len(wants.uncovered):
            self.quiet_asks.append(lead_part)
            twin = ep.brief.quiet_reask(asked_label) if ";" not in asked_label else ""
            if twin:
                self.ask_next(twin)
                note = note + f" [{twin}: its story twin, asked next]"
        if replies:
            shown = replies[0][1].split(chr(10) + '[status]')[0][:400]
            ep.questions_log.append(f"- {label[0]}: asked \"{label[1]}\" -> {shown}{note}")
        self.forget_wanted(asked_label)

    def ask_next(self, prop):
        rest = [p for p in self.pending_seconds or [] if p.lower() != prop.lower()]
        self.pending_seconds = [prop] + rest

    def hold_open(self, prop):
        if prop.lower() in [p.lower() for p in self.open_pair]:
            return
        self.open_pair.append(prop)

    def retry_plan(self, first, label, alone):
        brief = self.ep.brief
        key = first.lower()
        if self.ep.route != "single" and not self.basket_covered():
            self.spent_asks.add(key)
            return f" [{first}: not answered, not asked again while another product still waits for its first question]"
        plain_q = self.plain_question(first, self.product_tag(first))
        alt = brief.collide_alt(first) if alone else ""
        if alone and not alt:
            self.spent_asks.add(key)
            return f" [{first}: not answered; one property alone drew this, so the shopper holds no wish on it]"
        if alt:
            self.reasked.add(key)
            self.spent_asks.add(key)
            self.ask_next(alt)
            return f" [{first}: not answered, its story hint's alternative {alt} asked next]"
        fresh_wording = self.flat(plain_q) != self.flat(label[1])
        if key not in self.reasked and fresh_wording and not (alone and brief.holds_plain(label[1], plain_q)):
            self.reasked.add(key)
            self.ask_next(first)
            return f" [{first}: not answered, asked again alone next]"
        redo = None if key in self.reasked else brief.distinct_ask(first)
        if not redo:
            self.spent_asks.add(key)
            return f" [{first}: not answered, not asked again]"
        self.reasked.update({key, redo[0].lower()})
        self.pending_text[redo[0].lower()] = redo[1]
        self.ask_next(redo[0])
        bare = lambda text: re.split(r"\s*\(", text)[0].strip().lower()
        same = bare(redo[0]) == bare(first)
        if not same:
            self.spent_asks.add(key)
        return f" [{first}: not answered, asked again " + ("by its values next]" if same else f"as {redo[0]} next]")

    def split_pair(self, label, split_second):
        self.no_pairs = True
        parts = [p.strip() for p in label[0].split(";") if p.strip()] or [label[0]]
        note = self.retry_plan(parts[0], label, len(parts) == 1)
        for second in parts[1:] + ([split_second] if split_second else []):
            self.hold_open(second)
        return (" and ".join(parts) + " [not answered] [one at a time]", label[1]), note

    @staticmethod
    def second_heard(second, found_text):
        return any(w in found_text for w in re.findall(r"[a-z]{4,}", second.split("(")[0].lower()))

    def read_answer(self, label, heard_meta, found_before, freed_before, split_second):
        wants = self.ep.wants
        if heard_meta is not None and self.flat(label[1]) and self.flat(label[1]) in self.flat(heard_meta):
            return self.split_pair(label, split_second)
        paired = ";" in label[0]
        if paired:
            first, second = label[0].split(";")[0].strip(), label[0].split(";", 1)[1].strip()
            if len(wants.uncovered) > found_before:
                note = ""
                if not self.second_heard(second, " ".join(wants.uncovered[found_before:]).lower()):
                    self.queue_second(second)
                    note = f" [{second}: not answered, still open]"
                return (first, label[1]), note
            freed_now = len(wants.indifferent) + wants.pinned_heard
            if freed_now > freed_before:
                bare = second.split("(")[0].strip().lower()
                wants.indifferent = [p for p in wants.indifferent if p.strip().lower() not in (second.lower(), bare)]
                self.hold_open(second)
                return (first, label[1]), f" [{second}: not answered, still open]"
            if freed_now == freed_before:
                self.no_pairs = True
                for part in (p.strip() for p in label[0].split(";")):
                    if part:
                        self.hold_open(part)
                return (label[0].replace(";", " and") + " [not answered]", label[1]), ""
        if split_second and not self.second_heard(split_second, " ".join(wants.uncovered[found_before:]).lower()):
            self.hold_open(split_second)
        return label, ""

    def reply_back(self, text_body):
        ep = self.ep
        limit = ep.ask_limit()
        blocked = not ep.role("message") or ep.msgs_sent >= limit or ep.moves_left() <= 4 or ep.time_left() < CLOSING
        if blocked:
            return False
        saving = ep.asking and ep.msgs_sent >= ep.ask_limit() - 1 and ep.moves_left() > 10
        if saving and ep.time_left() >= CLOSING + 40:
            return False
        proposal = self.message_call(text_body, "reply_%s" % ep.shop.tick)
        if proposal is None:
            return False
        ep.chat.append({"role": "assistant", "content": "Answering the shopper's question.", "tool_calls": [proposal]})
        self.post_replies(ep.dispatch([proposal], scripted=True))
        return True

    def post_replies(self, responses):
        ep = self.ep
        final = len(responses) - 1
        for spot, (cid, body) in enumerate(responses):
            if spot == final:
                body += "\n" + ep.status_line()
            ep.chat.append({"role": "tool", "tool_call_id": cid, "content": body})


class Lookout:
    PAGING_WORDS = r"page|offset|first|start|skip|from|cursor"
    NOT_PAGING = r"size|per_?page|limit|price|cost"

    def __init__(self, ep):
        self.ep = ep
        self.next_seen = None
        self.hit_shelf = {}
        self.pointed_memo = None
        self.refusal_texts = None
        self.sweep_tried = set()
        self.sweep_turn = 0
        self.swept = 0

    @staticmethod
    def signature(call_args):
        return json.dumps(call_args, sort_keys=True, default=str)

    def refused(self, call_name, obs, resp):
        if isinstance(obs, dict) and "refusal" in obs:
            return True
        if call_name != self.ep.role("order") or resp.get("finished"):
            return False
        if isinstance(obs, dict) and obs.get("error") in ("invalid_args", "unknown_tool"):
            return False
        return True

    @staticmethod
    def sentences_in(obs):
        found = []
        stack = [(obs, 0)]
        while stack:
            piece, depth = stack.pop()
            if depth > 6:
                continue
            if isinstance(piece, dict):
                children = list(piece.values())
            elif isinstance(piece, list):
                children = piece[:20]
            else:
                if isinstance(piece, str) and len(piece.split()) >= 4:
                    found.append(piece)
                continue
            stack.extend((child, depth + 1) for child in reversed(children))
        return found

    def order_refused(self, obs):
        ep = self.ep
        ep.refusal_n += 1
        ep.order_vetted = False
        ep.event_seen = True
        try:
            hint = ep.shop.refusal_reread(obs)
        except Exception as err:
            hint = ""
            ep.log.append(dict(kind="refusal_reread_error", error=f"{type(err).__name__}: {err}"[:300]))
        ep.refused_lines.update(ep.basket.units)
        squash = lambda text: " ".join(str(text).lower().split())
        self.refusal_texts = set(map(squash, self.sentences_in(obs)))
        return ("ORDER REFUSED; the episode goes on with the cart as it was. Carry out each change the refusal "
                "names, fix the cart (drop every line no longer acceptable, a sold-out item for example), check "
                "the stock and price of the new pick once more, then order again." + hint)

    def shopper_push(self, obs):
        pushed = []
        for value in obs.values():
            if isinstance(value, list) and value and not [x for x in value if not isinstance(x, str)]:
                long_strings(value, pushed)
        if not pushed:
            return None
        earlier, self.refusal_texts = self.refusal_texts or set(), None

        def echoed(text):
            squashed = " ".join(str(text).lower().split())
            return any(squashed in old for old in earlier)

        if earlier and all(map(echoed, pushed)):
            return ("This shopper text repeats what the refused finish already carried (handled above); it tells nothing "
                    "about this result.")
        self.ep.event_seen = True
        return "A shopper message (a change) is included with this result. Apply it: " + " | ".join(pushed[:4])

    def notes_for(self, call_name, obs, refused):
        ep = self.ep
        said = []
        is_dict = isinstance(obs, dict)
        if is_dict and obs.get("error") == "budget_exceeded":
            if ep.msg_cap is not None or call_name == ep.role("message"):
                ep.msg_cap = ep.msgs_sent if ep.msg_cap is None else min(ep.msg_cap, ep.msgs_sent)
            said.append("That budget is used up; do not call this tool again.")
        declined = refused and call_name == ep.role("decline") and ep.last_kind == "decline"
        if declined:
            ep.audit_done = ep.decline_vetted = False
            ep.event_seen = True
            ep.decline_refusals += 1
            said.append("DECLINE REFUSED; the episode goes on as the store holds that something might still be bought. "
                        "Do what the refusal says, look at the closest candidates again (their own properties, stock and "
                        "price measured against the requirements now in force), order the one that fits; decline again only if "
                        "they all really fail.")
        elif refused:
            said.append(self.order_refused(obs))
        elif is_dict and len(obs) > 1 and call_name != ep.role("message"):
            pushed = self.shopper_push(obs)
            if pushed:
                said.append(pushed)
        return " ".join(said)

    @staticmethod
    def shelf_of(row):
        for key, value in row.items():
            name = str(key).lower()
            if "id" in name or not isinstance(value, str):
                continue
            trimmed = value.strip()
            if trimmed and len(trimmed) <= 60 and re.search(r"categor|shelf|department|aisle", name):
                return trimmed.lower()
        return ""

    @staticmethod
    def key_stock(key, val):
        name = str(key).lower()
        if val is None or not re.search(r"stock|avail|on.?hand", name):
            return None
        empty_figure = val is False or (is_number(val) and val <= 0)
        gone = False
        if re.search(r"stock|avail", name):
            if re.search(r"out|sold", name):
                gone = val is True
            elif empty_figure:
                gone = True
            elif isinstance(val, str):
                gone = bool(re.search(r"\b(?:out|sold|none|unavailable|not|no|false)\b", val.lower()))
        if empty_figure and re.search(r"on.?hand", name):
            gone = True
        if isinstance(val, str) and re.search(r"back.?order|pre.?order|discontinu", val.lower()):
            gone = True
        return not gone

    @staticmethod
    def stock_state(row):
        verdicts = []
        for key, val in row.items():
            said = Lookout.key_stock(key, val)
            if said is not None:
                verdicts.append(said)
        if not verdicts:
            return None
        return all(verdicts)

    @staticmethod
    def sold_out(row):
        return any(Lookout.key_stock(key, val) is False for key, val in row.items())

    @staticmethod
    def id_among(row, pool):
        for key, value in row.items():
            if isinstance(value, str) and "id" in str(key).lower() and value in pool:
                return value
        return None

    def shelf_tally(self):
        known = self.ep.known
        tally = {}
        counted = set()
        foreign = self.ep.pointers.ref_foreign()

        def count(row):
            ident = self.id_among(row, known.read)
            if not ident or ident in counted or ident in foreign:
                return
            label = self.shelf_of(row)
            if not label:
                return
            counted.add(ident)
            tally[label] = tally.get(label, 0) + 1
        known.visit_all(count)
        return tally

    def shelf_labels(self):
        tally = self.shelf_tally()
        top = max(tally.values(), default=0)
        leaders = set(label for label in tally if tally[label] == top)
        if not (tally and self.ep.wants.items()):
            return leaders
        try:
            homes = self.ep.shop.home_shelves()
            picked = set()
            for home in homes:
                seen = [label for label in home if label in tally]
                best = max((tally[label] for label in seen), default=None)
                picked |= set(home) if best is None else set(label for label in seen if tally[label] == best)
            spare = iter(sorted(tally, key=lambda s: -tally[s]))
            while not all(homes) and len(picked) < len(homes):
                label = next(spare, None)
                if label is None:
                    break
                picked.add(label)
            return picked or leaders
        except Exception:
            return leaders

    @staticmethod
    def row_mark(row):
        for exact in (True, False):
            for key, val in row.items():
                name = str(key).lower()
                hit = name == "id" if exact else name.endswith("id")
                if not hit or isinstance(val, bool) or not isinstance(val, (str, int)):
                    continue
                return str(val)
        return json.dumps(row, sort_keys=True, default=str)

    def text_words(self, value):
        pieces = [piece.strip("\"'()[]{},.;:!?|/") for piece in value.lower().split()]
        return tuple(dict.fromkeys(w for w in pieces if re.search(r"[a-z0-9]", w)))

    def search_keys(self, call_args):
        fields = self.search_fields()
        token = self.token_key()
        texts, modes, paging = [], [], set()
        first_page = True
        for field, value in call_args.items():
            if isinstance(value, str):
                kind = self.string_arg_kind(field, value, token, fields)
                if kind == "token":
                    paging.add(field)
                    first_page = first_page and not value.strip()
                elif kind == "mode":
                    modes.append((field, value))
                elif kind:
                    texts.append((field, kind))
                continue
            floor = self.paging_floor(field, value, fields)
            if floor is not None:
                paging.add(field)
                first_page = first_page and value <= floor
        return texts, tuple(sorted(modes)), first_page, paging

    def string_arg_kind(self, field, value, token, fields):
        if field == token:
            return "token"
        spec = fields.get(field)
        if isinstance(spec, dict) and spec.get("enum"):
            return "mode"
        words = self.text_words(value)
        if words and all(re.fullmatch(r"[a-z0-9]{2,}", w) for w in words):
            return words
        return None

    def paging_floor(self, field, value, fields):
        name = str(field).lower()
        if not is_number(value) or not re.search(self.PAGING_WORDS, name) or re.search(self.NOT_PAGING, name):
            return None
        spec = fields.get(field)
        spec = spec if isinstance(spec, dict) else {}
        floor = spec.get("minimum", spec.get("default"))
        if is_number(floor):
            return floor
        return 1 if "page" in name else 0

    @staticmethod
    def result_total(obs, rows):
        figures = [v for v in obs.values() if isinstance(v, int) and not isinstance(v, bool)]
        return max(figures + [len(rows)])

    @staticmethod
    def other_args(call_args, field, paging):
        return tuple(sorted((k, str(v)) for k, v in call_args.items() if k != field and k not in paging))

    @staticmethod
    def word_reading(words, total, old_words, old_total, depth):
        if len(words) > len(old_words) and words[: len(old_words)] == old_words:
            return ("any" if total > old_total else "all"), False
        if len(old_words) > len(words) and old_words[: len(words)] == words:
            return ("any" if total < old_total else "all"), False
        ours, theirs = set(words[:depth]), set(old_words[:depth])
        narrowed = (ours > theirs and total < old_total) or (theirs > ours and old_total < total)
        return None, narrowed

    def word_rule(self, call_args, obs, rows):
        ep = self.ep
        texts, mode, first_page, paging = self.search_keys(call_args)
        if not first_page:
            return ep.word_kind
        total = self.result_total(obs, rows)
        for field, words in texts:
            context = (field, mode)
            rest = self.other_args(call_args, field, paging)
            hinted = False
            for seen in ep.word_seen:
                if seen[0] != context or seen[1] != rest or seen[3] == total:
                    continue
                kind, narrowed = self.word_reading(words, total, seen[2], seen[3], ep.lead_words or None)
                if kind is not None:
                    ep.word_kind = kind
                hinted = hinted or narrowed
            if hinted and not ep.word_kind:
                ep.word_kind = "all"
            ep.word_seen.append((context, rest, words, total))
        return ep.word_kind

    def lead_rule(self, call_args, obs, rows):
        ep = self.ep
        texts, mode, first_page, paging = self.search_keys(call_args)
        if not first_page:
            return ep.lead_words
        total = self.result_total(obs, rows)
        marks = tuple(map(self.row_mark, rows))
        for field, words in texts:
            context = (field, mode)
            rest = self.other_args(call_args, field, paging)
            bounds = ep.lead_bounds.setdefault(context, [-1, set()])
            for seen in ep.lead_seen:
                if seen[0] != context or seen[1] != rest or seen[2] == words:
                    continue
                old_words = seen[2]
                shorter = min(len(words), len(old_words))
                shared = next((n for n in range(shorter) if words[n] != old_words[n]), shorter)
                if total != seen[3] or marks != seen[4]:
                    bounds[0] = max(bounds[0], shared)
                elif shared and shared == shorter and len(set(marks)) >= 3:
                    bounds[1].add(shared)
            ep.lead_seen.append((context, rest, words, total, marks))
        claims = sorted((min(k for k in unread if k > proven), context)
                        for context, (proven, unread) in ep.lead_bounds.items()
                        if any(k > proven for k in unread))
        claims = [pair for pair in claims if pair[0] == ep.lead_bounds[pair[1]][0] + 1]
        ep.lead_words, ep.lead_ctx = claims[0] if claims else (0, None)
        return ep.lead_words

    def sweep_allowed(self):
        ep = self.ep
        wants = ep.wants
        if wants.priority() != "cheapest" or not wants.let_go or wants.props_left():
            return None
        if not ep.role("search") or ep.shop.done:
            return None
        reserve = 4 if ep.basket.units else 3
        if not ep.orders_tried:
            reserve = max(reserve + 3, FIRST_ORDER_BY)
        if self.swept >= SWEEP_READS or not self.sweep_room(reserve):
            return None
        return reserve

    def sweep_summary(self, replies):
        cheapest = sorted(self.shelf_rows().items(), key=lambda kv: (kv[1][0], kv[0]))[:6]
        if not cheapest:
            return
        cid, text = replies[-1]
        listed = "; ".join("%s '%s' %s" % (i, t[:80], format(c, "g")) for i, (c, t, _) in cheapest)
        replies[-1] = (cid, text + ("\n[shelf sweep done] Lowest-priced in-stock rows of the requested category, within the budget, "
                                    "collected from every page read: ") + listed + (". If no change rules it out, take the cheapest: let it be the cart's lone line, check its "
                                                                                    "stock and price again, then order it."))

    def sweep_step(self):
        ep = self.ep
        reserve = self.sweep_allowed()
        if reserve is None:
            return False
        call_args, from_listing = self.sweep_next(self.sweep_turn)
        if call_args is None:
            return False
        self.swept += 1
        self.sweep_turn += 0 if from_listing else 1
        self.sweep_tried.add(self.signature(call_args))
        search = ep.role("search")
        proposal = dict(id=f"sweep_{self.swept}_{ep.shop.tick}", type="function",
                        function=dict(name=search, arguments=json.dumps(call_args)))
        ep.chat.append({"role": "assistant", "content": ("No property requirement remains; the lowest-priced in-stock listing of the shelf wins: I read another "
                                                         "part of the shelf before I choose."), "tool_calls": [proposal]})
        ep.covering = True
        try:
            replies = ep.dispatch([proposal], scripted=True)
        finally:
            ep.covering = False
        if self.swept < SWEEP_READS and self.sweep_room(reserve) and self.sweep_next(self.sweep_turn)[0] is not None:
            pass
        elif replies:
            self.sweep_summary(replies)
        ep.talk.post_replies(replies)
        return True

    def book_page(self, call_args, rows, total, obs=None):
        onward = self.int_next(obs)
        if onward is not None and self.next_seen is None and rows:
            self.next_seen = onward <= 2 < len(rows)
        paging = self.page_arg()
        if not paging or not rows:
            return
        field, mode, first = paging[:3]
        position = call_args.get(field, first)
        if mode == "token":
            position = position.strip() if isinstance(position, str) else ""
        elif not is_number(position):
            return
        rest = {k: v for k, v in call_args.items() if k != field}
        blank = {"args": rest, "read": set(), "size": 0, "total": 0, "shelves": set()}
        walk = self.ep.page_book.setdefault(self.signature(rest), blank)
        walk["read"].add(position)
        if mode == "token":
            walk.setdefault("tokens", {})[position] = self.next_token(obs)
        records = [r for r in rows if isinstance(r, dict)]
        walk["size"] = max(walk["size"], len(rows))
        walk["total"] = max(walk["total"], total)
        self.file_rows(walk, records)
        marks = [self.row_mark(r) for r in records]
        gathered = walk.setdefault("rows", {})
        gathered.update((mark, record) for mark, record in zip(marks, records) if mark)
        if position == first:
            walk["first"] = tuple(marks)
            if total <= len(rows):
                walk["every"] = records
        elif "every" not in walk and first in walk["read"] and walk["total"] <= len(gathered):
            walk["every"] = list(gathered.values())
            walk["closed"] = True
        if len(marks) >= 5:
            walk["id_order"] = self.ids_ascend(marks) and walk.get("id_order", True) is not False
        return walk

    def file_rows(self, walk, records):
        walk["shelves"] |= {self.shelf_of(r) for r in records} - {""}
        by_row = walk.setdefault("row_shelf", {})
        for record in records:
            by_row.setdefault(self.row_mark(record), self.shelf_of(record))

    def ids_ascend(self, marks):
        return sorted(marks) == marks or sorted(marks, key=self.natural_key) == marks

    def search_fields(self):
        schema = self.ep.shapes.get(self.ep.role("search") or "") or {}
        found = schema.get("properties")
        if isinstance(found, dict):
            return found
        return {}

    def token_key(self):
        for field, spec in self.search_fields().items():
            if not isinstance(spec, dict) or spec.get("type") != "string" or spec.get("enum"):
                continue
            if re.search(r"bookmark|cursor|token|resume|continu|after|next|page|offset|start", str(field).lower()):
                return field
        return None

    def page_arg(self):
        for field, spec in self.search_fields().items():
            if not isinstance(spec, dict) or spec.get("type") not in ("integer", "number"):
                continue
            name = str(field).lower()
            if not re.search(self.PAGING_WORDS, name) or re.search(self.NOT_PAGING, name):
                continue
            whole_pages = "page" in name and not re.search(r"offset|skip|first|start|from", name)
            first = spec.get("minimum", spec.get("default"))
            if not is_number(first):
                first = 1 if whole_pages else 0
            top = spec.get("maximum")
            return field, whole_pages, first, top if is_number(top) else None
        token = self.token_key()
        if token:
            return token, "token", "", None
        return self.offset_arg()

    def offset_arg(self):
        if self.next_seen is None:
            return None
        schema = self.ep.shapes.get(self.ep.role("search") or "") or {}
        required = set(schema.get("required") or []) if isinstance(schema.get("required"), list) else set()
        loose = [(field, spec) for field, spec in self.search_fields().items()
                 if isinstance(spec, dict) and spec.get("type") in ("integer", "number") and not spec.get("enum")
                 and field not in required and not re.search(self.NOT_PAGING, str(field).lower())]
        if len(loose) != 1:
            return None
        field, spec = loose[0]
        first = spec.get("minimum", spec.get("default"))
        top = spec.get("maximum")
        return field, self.next_seen, first if is_number(first) else 0, top if is_number(top) else None

    @staticmethod
    def int_next(obs):
        if not isinstance(obs, dict):
            return None
        for key, value in obs.items():
            if isinstance(value, int) and not isinstance(value, bool) and value > 0 and re.search(r"next|onward|cursor|bookmark|token|continu|more", str(key).lower()):
                return value
        return None

    @staticmethod
    def stem(word):
        if len(word) <= 3 or not word.endswith("s") or word.endswith("ss"):
            return word
        return word[:-1]

    @staticmethod
    def row_title(record):
        title = record.get("title")
        if isinstance(title, str) and 0 < len(title) <= 300:
            return title
        for key, value in record.items():
            if not isinstance(value, str) or not 0 < len(value) <= 300:
                continue
            name = str(key).lower()
            if not re.search(r"title|heading|headline|name", name):
                continue
            if re.search(r"categor|shelf|brand|make|manufactur|vendor|shop|seller|store|id", name):
                continue
            return value
        return ""

    @staticmethod
    def first_price(record):
        for key, value in record.items():
            if isinstance(value, bool) or not is_number(value):
                continue
            if re.search(r"price|cost", str(key).lower()):
                return value
        return None

    def shelf_rows(self):
        ep = self.ep
        known = ep.known
        labels = self.shelf_labels()
        budget, units = ep.wants.spend_cap, ep.wants.quantity()
        rows = {}
        if not labels:
            return rows

        def take(record):
            ident = self.id_among(record, known.ids)
            if not ident or ident in ep.refused_lines or self.shelf_of(record) not in labels:
                return
            if self.sold_out(record):
                rows.pop(ident, None)
                return
            price = self.first_price(record)
            title = self.row_title(record)
            maker = " ".join(v for k, v in record.items()
                             if isinstance(v, str) and re.search(r"brand|make|manufactur|vendor", str(k).lower()))
            if price is not None:
                rows[ident] = (float(price), title, maker)

        known.visit_all(take)
        for ident in list(rows):
            price, title, maker = rows[ident]
            latest = known.price(ident)
            price = price if latest is None else latest
            if is_number(budget) and price * units > budget:
                rows.pop(ident)
            else:
                rows[ident] = (float(price), title, maker)
        return rows

    def label_queries(self):
        queries = []
        for label in sorted(self.shelf_labels()):
            groups = [[]]
            for raw in label.lower().split():
                word = raw.strip(".,;:!?'\"()[]")
                if word not in LINK_WORDS and re.fullmatch(r"[a-z0-9][a-z0-9'-]*", word):
                    groups[-1].append(word)
                elif groups[-1]:
                    groups.append([])
            groups = [group for group in groups if group]
            flat = sum(groups, [])
            heads = [group[-1] for group in groups]
            if len(flat) > 1:
                queries.append(" ".join(flat[:2]))
            queries += heads + [w for w in flat if w not in heads]
        return list(dict.fromkeys(queries))

    def kind_queries(self, pairs):
        stems_on_labels = set()
        for label in self.shelf_labels():
            stems_on_labels |= set(map(self.stem, re.findall(r"[a-z0-9]+", label)))
        titles = {}
        for price, title, maker in self.shelf_rows().values():
            key = title.lower().strip()
            if not key:
                continue
            earlier = titles.get(key)
            titles[key] = (price if earlier is None else min(price, earlier[0]), maker.lower())
        tallies, lowest, written = {}, {}, {}
        for title, (price, maker) in titles.items():
            skip = set(re.findall(r"[a-z]+", maker))
            words = re.findall(r"[a-z]+", title)
            fit = [len(w) >= 3 and w not in LINK_WORDS and w not in skip for w in words]
            span = 2 if pairs else 1
            grams = [tuple(words[i:i + span]) for i in range(len(words) - span + 1) if all(fit[i:i + span])]
            done = set()
            for gram in grams:
                stems = [self.stem(w) for w in gram]
                joined = " ".join(stems)
                if joined in done or all(s in stems_on_labels for s in stems):
                    continue
                done.add(joined)
                tallies[joined] = tallies.get(joined, 0) + 1
                lowest[joined] = min(lowest.get(joined, price), price)
                if joined not in written:
                    written[joined] = " ".join(gram)
        common = [g for g, n in tallies.items() if n >= 2]
        common.sort(key=lambda g: (lowest[g], -tallies[g], g))
        return list(map(written.get, common))

    def deepest_listing(self, book, labels):
        chosen = None
        for walk in book.values():
            usable = walk.get("id_order") and walk["shelves"] and walk["shelves"] <= labels
            if not usable or self.next_page(walk) is None:
                continue
            if chosen is None or chosen["total"] < walk["total"]:
                chosen = walk
        return chosen

    def query_template(self, book, labels, page_field):
        ep = self.ep
        ranked = [w for w in book.values() if w.get("id_order") is False and self.free_text(w["args"])]
        if not ranked:
            ranked = [w for w in book.values() if w.get("id_order") is None and self.free_text(w["args"])]
        if not ranked:
            return None, None
        fields = self.search_fields() or {}
        model_args = ranked[-1]["args"]

        def fixed(key, value):
            if isinstance(value, str) and not (fields.get(key) or {}).get("enum"):
                return key == page_field or value.strip().lower() in labels
            return True

        free = [key for key, value in model_args.items() if not fixed(key, value)]
        lead = ep.lead_ctx
        if lead and lead[0] in free:
            text_field = lead[0]
        else:
            text_field = max(free, key=lambda key: len(model_args[key].split()), default=None)
        if not text_field:
            return None, text_field
        return {key: value for key, value in model_args.items() if key != text_field and fixed(key, value)}, text_field

    def sweep_next(self, turn):
        ep = self.ep
        paging = self.page_arg()
        labels = self.shelf_labels()
        book = ep.page_book
        if not (paging and labels and book):
            return None, False
        page_field, first = paging[0], paging[2]
        tried = self.sweep_tried
        listing = self.deepest_listing(book, labels)
        template, text_field = self.query_template(book, labels, page_field)
        covered = set()
        for walk in book.values():
            if walk.get("id_order") is True or first not in walk["read"]:
                continue
            words = re.findall(r"[a-z0-9]{2,}", " ".join(self.free_text(walk["args"])).lower())
            covered.add(tuple(map(self.stem, words[:2])))

        def calls_from(texts):
            built = []
            for text in texts:
                mark = tuple(map(self.stem, re.findall(r"[a-z0-9]{2,}", text.lower())[:2]))
                if template is None or not mark or mark in covered:
                    continue
                call_args = dict(template)
                call_args[text_field] = text
                if self.signature(call_args) not in tried:
                    built.append(call_args)
            return built

        def source(n):
            if n == 0:
                if not listing:
                    return []
                paged = dict(listing["args"])
                paged[page_field] = self.next_page(listing)
                return [paged]
            if n == 1:
                return calls_from(self.kind_queries(True)[:6])
            if n == 2:
                return calls_from(self.kind_queries(False)[:6])
            return calls_from(self.label_queries())

        ceiling_call, finished = ep.scout.ceiling_sweep(tried)
        if ceiling_call is not None:
            return ceiling_call, True
        if finished:
            return None, False
        for n in [0, 1 + turn % 3, 1 + (turn + 1) % 3, 1 + (turn + 2) % 3]:
            for call_args in source(n):
                if self.signature(call_args) not in tried:
                    return call_args, n == 0
        return None, False

    def sweep_room(self, reserve):
        return self.ep.moves_left() > reserve and self.ep.time_left() >= CLOSING + 40

    def free_text(self, call_args):
        fields = self.search_fields()
        token = self.token_key()
        texts = []
        for key, value in call_args.items():
            if not isinstance(value, str) or key == token:
                continue
            if (fields.get(key) or {}).get("enum"):
                continue
            texts.append(value)
        return texts

    @staticmethod
    def next_token(obs):
        if not isinstance(obs, dict):
            return None
        for key, value in obs.items():
            if isinstance(value, str) and value.strip() and re.search(r"next|onward|cursor|bookmark|token|continu|more", str(key).lower()):
                return value
        return None

    @staticmethod
    def natural_key(mark):
        return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", mark)]

    def next_page(self, walk):
        field, whole_pages, first, highest = self.page_arg()
        if whole_pages == "token":
            tokens = (walk.get("tokens") or {}).values()
            return next((t for t in tokens if isinstance(t, str) and t not in walk["read"]), None)
        start = min([first] + [v for v in walk["read"] if is_number(v)])
        size = max(1, walk["size"])
        stride = size if not whole_pages else 1
        for n in range(1, 400):
            if n * size >= walk["total"]:
                return None
            candidate = start + n * stride
            if highest is not None and candidate > highest:
                return None
            if candidate not in walk["read"]:
                return candidate
        return None

    @staticmethod
    def close_words(a, b):
        if min(len(a), len(b)) < 3:
            return False
        return a.startswith(b) or b.startswith(a)

    def value_score(self, call_args):
        ep = self.ep
        value_sets = []
        for need in ep.wants.props_left():
            value = need.partition(":")[2]
            if ref_text(value):
                continue
            value_sets.append(re.findall(r"[a-z0-9]{3,}", value.lower()))
        typed = []
        for text in self.free_text(call_args):
            typed += re.findall(r"[a-z0-9]{2,}", text.lower())
        if ep.lead_words:
            typed = typed[: ep.lead_words]
        score = 0
        for words in value_sets:
            if any(self.close_words(t, v) for t in typed for v in words):
                score += 1
        return score

    def maker_sets(self, row):
        sets = []
        for key, value in row.items():
            lowered = str(key).lower()
            if isinstance(value, str) and value.strip() and "id" not in lowered and \
                    re.search(r"shop|seller|store|vendor|brand|make|manufactur", lowered):
                sets.append(set(re.findall(r"[a-z0-9]+", value.lower())))
        return sets

    def row_hits(self, rows, words=None):
        ep = self.ep
        known = ep.known
        budget, units = ep.wants.spend_cap, ep.wants.quantity()
        labels = self.shelf_labels()
        named = set()
        if ep.wants.reading.get("references"):
            named = {w for t in words or [] for w in re.findall(r"[a-z0-9]+", str(t).lower())}
        found = []
        for row in filter(lambda r: isinstance(r, dict), rows):
            ident = self.row_mark(row)
            makers = self.maker_sets(row)
            unread = ident in known.ids and ident not in known.read
            pointed = named and unread and any(m and m <= named and m <= self.pointed_words() for m in makers)
            if pointed and ident not in ep.ref_hits:
                ep.ref_hits.append(ident)
            price = next((v for k, v in row.items() if is_number(v) and re.search(r"price|cost", str(k).lower())), None)
            label = self.shelf_of(row)
            if not unread:
                continue
            too_dear = is_number(budget) and is_number(price) and price * units > budget
            off_shelf = labels and label and label not in labels
            if too_dear or self.sold_out(row) or off_shelf:
                continue
            found.append(ident)
            self.hit_shelf[ident] = label
        return found

    def pointed_words(self):
        known = self.ep.known
        memo = self.pointed_memo
        if memo and memo[0] == len(known.items):
            return memo[1]
        words = set()
        seller_key = r"shop|seller|store|vendor|brand|make|manufactur"
        prose_key = r"note|text|desc|comment|detail|info|message|section"
        name_key = r"title|heading|headline|name|label|caption"
        try:
            for ref in self.ep.wants.reading.get("references") or []:
                words |= set(re.findall(r"[a-z0-9]+", str(ref).lower()))

            def gather(record):
                if self.row_mark(record) not in known.read:
                    return
                sellers, bodies = set(), []
                for key, value in filter(lambda kv: isinstance(kv[1], str), record.items()):
                    lowered = str(key).lower()
                    if (re.search(seller_key, lowered) and "id" not in lowered and len(value) <= 40
                            and not re.search(prose_key, lowered)):
                        sellers |= set(re.findall(r"[a-z0-9]+", value.lower()))
                    elif not re.search(name_key, lowered):
                        bodies.append(value.lower())
                for body in bodies:
                    words.update(set(re.findall(r"[a-z0-9]+", body)) - sellers)
            known.visit_all(gather)
        except Exception:
            pass
        self.pointed_memo = (len(known.items), words)
        return words

    def open_pages(self, most=2):
        paging = self.page_arg()
        if not paging:
            return []
        labels = self.shelf_labels()
        scored = []
        for walk in self.ep.page_book.values():
            following = self.next_page(walk)
            off_shelf = labels and walk["shelves"] and not walk["shelves"] & labels
            if following is None or off_shelf:
                continue
            score = self.value_score(walk["args"])
            if not score:
                continue
            call_args = dict(walk["args"])
            call_args[paging[0]] = following
            scored.append(((-score, walk["total"]), call_args))
        scored.sort(key=lambda pair: pair[0])
        return [call_args for _, call_args in scored[:most]]

    @staticmethod
    def probe_near(a, b):
        if not any(c.isdigit() for c in a + b):
            return min(len(a), len(b)) >= 3 and (a.startswith(b) or b.startswith(a))
        if a.lstrip("0") == b.lstrip("0"):
            return True
        return a.isdigit() and b.startswith(a) and not b[len(a):len(a) + 1].isdigit()

    @staticmethod
    def long_words(text, depth):
        return [w for w in re.findall(r"[a-z0-9]+", text.lower()) if len(w) >= 2][:depth]

    def wanted_values(self):
        pairs = []
        for need in self.ep.wants.props_left():
            name, colon, value = need.partition(":")
            if colon and not ref_text(value):
                stripped = re.sub(r"\((?:stated|revealed[^)]*)\)", "", value)
                pairs.append((name.strip(), re.split(r"[,;]", stripped)[0].strip().lower()))
        return pairs

    def probe_already_run(self, head, depth):
        for walk in self.ep.page_book.values():
            ran = self.long_words(" ".join(self.free_text(walk["args"])), depth)
            if len(ran) != len(head):
                continue
            if all(self.probe_near(a, b) for a, b in zip(head, ran)):
                return True
        return False

    def value_probe(self, raw):
        ep = self.ep
        if ep.word_kind == "any":
            return "", ""
        depth = ep.lead_words or 2
        lead = self.long_words(" ".join(raw), depth)
        product = " ".join(re.findall(r"[a-z0-9]+", str(ep.wants.reading.get("product") or "").lower())[:3])
        wanted = self.wanted_values()
        known = []
        for _, value in wanted:
            known += re.findall(r"[a-z0-9]{2,}", value)
        known += product.split()
        if not lead:
            return "", ""
        for word in lead:
            if any(self.probe_near(word, k) for k in known):
                return "", ""
        for name, value in wanted:
            bits = re.findall(r"[a-z0-9]{2,}", name.lower())
            if not value or not any(self.probe_near(w, n) for w in lead for n in bits):
                continue
            digit = re.search(r"\d", value)
            core = value if not digit else value[digit.start():]
            probe = (core.strip(' .;') + " " + product).strip()
            if self.probe_already_run(self.long_words(probe, depth), depth):
                return "", ""
            return name, probe
        return "", ""


class ProductTagger:
    def __init__(self, pair_words, label_words, other_words, tag_after):
        self.pair_words = pair_words
        self.label_words = label_words
        self.other_words = other_words
        self.tag_after = tag_after

    def ours(self, tag):
        words = kind_tokens(tag)
        if not words or not kinds_within(words, self.label_words):
            return False
        if not any(kind_in(w, self.pair_words) for w in words):
            return False
        return not any(kinds_within(words, other) and kinds_within(other, words) for other in self.other_words)

    def _swap(self, hit):
        if self.ours(hit.group(1)):
            return f" ({self.tag_after})" if self.tag_after else ""
        return hit.group(0)

    def text(self, value):
        if not isinstance(value, str):
            return value
        head, colon, tail = value.partition(":")
        return re.sub(r"\s*\(([^()]*)\)", self._swap, head) + colon + tail

    def unique(self, values):
        out = []
        for value in map(self.text, values):
            if value not in out:
                out.append(value)
        return out


class ShelfKeeper:
    def __init__(self, ep):
        self.ep = ep
        self.home_memo = None
        self.joined_seen = {}
        self.dead = set()
        self.dead_by = {}
        self.stock_by = {}
        self.stock_seen = None
        self.rows_lag = None
        self.rows_live = None
        self.row_keys = set()
        self.row_text = {}
        self.read_with = {}
        self.text_keys = {}
        self.tool_caps = {}
        self.tool_keys = {}
        self.note_keys = {}
        self.role_guess = None
        self.swap_page = None
        self.swap_unread = None
        self.repair_tried = set()
        self.ed_memo = {}
        self.slack_total = None
        self.near_cart = None
        self.near_done = None
        self.near_ok = set()
        self.near_said = {}
        self.cart_shown = -1
        self.cart_ver = None
        self.first_puts = set()
        self.pre_picked = None
        self.pre_read_done = None
        self.pre_rounds = 0

    # which category labels belong to which product

    def _identified_labels(self):
        found = set()
        for rec in self.ep.counsel.dicts_in(self.ep.known.items):
            shelf = self.ep.lookout.shelf_of(rec)
            if shelf and any(isinstance(val, str) and val in self.ep.known.ids and "id" in str(key).lower()
                             for key, val in rec.items()):
                found.add(shelf)
        return found

    @staticmethod
    def _closest_labels(kind_sets, labels):
        result = []
        for spot, mine in enumerate(kind_sets):
            others = set().union(*(words for idx, words in enumerate(kind_sets) if idx != spot))
            best_score = 0.0
            chosen = set()
            for name in sorted(labels):
                score = kind_score(mine, set(kind_tokens(name)), others)
                if score > best_score:
                    chosen = {name}
                    best_score = score
                elif score > 0 and score == best_score:
                    chosen.add(name)
            result.append(chosen)
        return result

    def home_shelves(self):
        memo = self.home_memo
        if memo and memo[0] == len(self.ep.known.items):
            return [set(group) for group in memo[1]]
        wants = self.ep.wants
        try:
            rows = wants.items() or [{"product": str(wants.reading.get("product") or "")}]
            groups = self._closest_labels([set(kind_tokens(r["product"])) for r in rows], self._identified_labels())
        except Exception:
            groups = [set() for _ in wants.items() or [0]]
        self.home_memo = (len(self.ep.known.items), [set(group) for group in groups])
        return groups

    def own_shelf(self):
        try:
            if self.ep.wants.items():
                return ""
            home = self.home_shelves()[0]
            visited = [name.strip().lower() for name in self.ep.known.shelves_read()]
            if not home or all(name in home for name in visited):
                return ""
            for name in visited:
                if name in home:
                    return name
            return sorted(home)[0]
        except Exception:
            return ""

    # one store category that joins two requested products

    @staticmethod
    def _ordered_label_hit(text, label):
        pieces = [bit for bit in re.split(JOINERS, str(label).lower()) if bit.strip()]
        first = name_pattern(pieces[0]) if len(pieces) == 2 else ""
        second = name_pattern(pieces[1]) if len(pieces) == 2 else ""
        if not (first and second):
            return None
        return re.search(r"\b" + first + r"\s*(?:\band\b|&|\bor\b|/|\+)\s*" + second, text)

    @staticmethod
    def _gap_joins(between):
        if len(re.findall(r"[a-z0-9]+", between)) > 3:
            return False
        return bool(re.search(r"\band\b|&|\bor\b|,|/|\+", between)) and not re.search(r"\d|\bx\b", between)

    @staticmethod
    def _near_pair_hit(text, x, y, a, b):
        left, right = name_pattern(x), name_pattern(y)
        if not left or not right:
            return None
        for hit in re.finditer(r"\b(?:" + left + r")(?P<mid>[^.;:!?]{0,40}?)(?:" + right + r")", text):
            if not ShelfKeeper._gap_joins(hit.group("mid")):
                continue
            rest = text[:hit.start()] + " | " + text[hit.end():]
            if not (re.search(r"\b" + name_pattern(a), rest) and re.search(r"\b" + name_pattern(b), rest)):
                return hit
        return None

    def joined_phrase(self, a, b, label):
        flat = re.sub(r"\s+", " ", self.ep.request.lower().replace("’", "'"))
        exact = self._ordered_label_hit(flat, label)
        if exact:
            return exact.group(0).strip(), True
        a_kinds, b_kinds = kind_tokens(a), kind_tokens(b)
        if a_kinds and b_kinds and not same_kind(a_kinds[-1], b_kinds[-1]):
            loose = self._near_pair_hit(flat, a, b, a, b) or self._near_pair_hit(flat, b, a, a, b)
            if loose:
                return loose.group(0).strip(), False
        return None

    def product_question(self, line, rows):
        shaped = re.match(r'-\s*(.*?): asked "(.*?)" -> ', str(line), re.S)
        label = shaped.group(1) if shaped else str(line)
        question = shaped.group(2) if shaped else str(line)
        row_kinds = [kind_tokens(r["product"]) for r in rows]
        for inner in re.findall(r"\(([^()]*)\)", label):
            tag_words = kind_tokens(inner)
            if tag_words and any(kind_in(word, kinds) for word in tag_words for kinds in row_kinds):
                return True
        lowered = question.lower().replace("’", "'")
        for r in rows:
            if name_pattern(r["product"]) and re.search(r"\b" + name_pattern(r["product"]), lowered):
                return True
        return False

    @staticmethod
    def joined_clash(a, b):
        def prop_values(row):
            return [(re.split(r"\s*\(", str(req).partition(":")[0])[0].strip(),
                     " ".join(re.findall(r"[a-z0-9]+", str(req).partition(":")[2].lower())))
                    for req in row.get("requirements") or []]
        theirs = prop_values(b)
        for pa, va in prop_values(a):
            if any(same_prop(pa, pb) and va != vb for pb, vb in theirs):
                return True
        return False

    @staticmethod
    def _record_label(rec):
        for key, val in rec.items():
            if not isinstance(val, str) or not 0 < len(val.strip()) <= 60:
                continue
            lowered = str(key).lower()
            if "id" not in lowered and re.search(r"categor|shelf|department|aisle", lowered):
                return val.strip()
        return ""

    def _remember_joined_labels(self, obs):
        for rec in self.ep.counsel.dicts_in([obs]):
            if not any(isinstance(v, str) and "id" in str(k).lower() and re.fullmatch(ID_LIKE, v) for k, v in rec.items()):
                continue
            label = self._record_label(rec)
            if label and len([bit for bit in re.split(JOINERS, label.lower()) if bit.strip()]) == 2:
                self.joined_seen.setdefault(label.lower(), label)

    @staticmethod
    def _splits_by_sides(ti, tj, sides):
        own_i = [w for w in ti if not kind_in(w, tj)]
        own_j = [w for w in tj if not kind_in(w, ti)]
        for left, right in (sides, sides[::-1]):
            if not all(kind_in(w, left) and not kind_in(w, right) for w in own_i):
                continue
            if not all(kind_in(w, right) and not kind_in(w, left) for w in own_j):
                continue
            if any(kind_in(w, left) for w in ti) and any(kind_in(w, right) for w in tj):
                return True
        return False

    def _pair_joinable(self, rows, i, j, sides):
        ti, tj = kind_tokens(rows[i]["product"]), kind_tokens(rows[j]["product"])
        if not ti or not tj or (kinds_within(ti, tj) and kinds_within(tj, ti)):
            return False
        if not kinds_within(ti + tj, sides[0] + sides[1]) or rows[i]["quantity"] != rows[j]["quantity"]:
            return False
        return self._splits_by_sides(ti, tj, sides) and not self.joined_clash(rows[i], rows[j])

    def _find_joined_pair(self, rows):
        for low in sorted(self.joined_seen):
            sides = [kind_tokens(bit) for bit in re.split(JOINERS, low) if bit.strip()]
            if len(sides) != 2 or not sides[0] or not sides[1]:
                continue
            for i in range(len(rows)):
                for j in range(i + 1, len(rows)):
                    if not self._pair_joinable(rows, i, j, sides):
                        continue
                    label = self.joined_seen[low]
                    phrase = self.joined_phrase(rows[i]["product"], rows[j]["product"], label)
                    if phrase:
                        return i, j, label, phrase[0], phrase[1]
        return None

    def _merge_blocked(self, rows):
        wants, basket = self.ep.wants, self.ep.basket
        if len(rows) < 2 or not self.joined_seen or basket.used or basket.units:
            return True
        if wants.uncovered or wants.indifferent:
            return True
        return any(self.product_question(entry, rows) for entry in self.ep.questions_log)

    def merge_joined(self, obs):
        self._remember_joined_labels(obs)
        said = []
        for _ in range(2):
            rows = self.ep.wants.items()
            pair = None if self._merge_blocked(rows) else self._find_joined_pair(rows)
            if not pair:
                break
            said.append(self.merge_items(*pair))
        return " ".join(bit for bit in said if bit)

    @staticmethod
    def _union_requirements(first, second):
        out = []
        for req in [*first["requirements"], *second["requirements"]]:
            head, _, tail = str(req).partition(":")
            plain = re.sub(r"\s*\([^()]*\)", "", head).strip() + ":" + tail
            if plain.lower() not in (x.lower() for x in out):
                out.append(plain)
        return out

    def _wording_pools(self, slots, units, basket):
        pools = {}
        for n, k in enumerate(slots):
            entries = self.ed_pool(k)
            cheapest_fit = min(((e["price"], e["id"]) for e in entries if e["fit"] and e["price"] is not None), default=None)
            if cheapest_fit is not None:
                basket[n] = (cheapest_fit[1], cheapest_fit[0], units[n])
                continue
            options = self._reworded_options(entries, self.ed_reqs(k)) if self.ed_news(k) else []
            if not options:
                return None
            options.sort()
            del options[6 if len(slots) == 1 else 3:]
            pools[n] = options
        return pools

    def _fill_from_agreed(self, pools, flat, agreed, basket, units):
        for n in sorted(pools):
            accepted = None
            for _, price, ident, lines in pools[n]:
                spots = [at for at, row in enumerate(flat) if row[:2] == (n, ident)]
                if len(spots) == len(lines) and set(spots) <= agreed:
                    accepted = (ident, price, units[n])
                    break
            if accepted is None:
                return False
            basket[n] = accepted
        return True

    def _load_reviewed(self, picks, pools, basket, flat):
        version = str(self.ep.wants.version)
        for n in pools:
            chosen = basket[n][0]
            stamp = f"{chosen}|{version}"
            stale_memos = [m for m in self.ep.rulings if m == stamp or m.startswith(stamp + "|")]
            self.ep.rulings.update(dict.fromkeys(stale_memos, ""))
            self.ep.passed.add(stamp)
            self.near_ok.add(stamp)
            self.ep.pass_price[chosen] = self.ep.known.price(chosen)
        done = self._place_repair(picks)
        self.ep.log.append({"kind": "near_basket", "picks": picks, "done": done, "cart": dict(self.ep.basket.units)})
        if not done:
            return ""
        self.near_cart = {"ver": self.ep.wants.version, "refusals": self.ep.refusal_n, "lines": [p[0] for p in picks]}
        kept = [row[1:] for row in flat if row[1] == basket.get(row[0], ("",))[0]]
        said = []
        for ident, prop, value, says in kept:
            self.near_said.setdefault(f"{ident}|{version}", []).append((prop, value, says))
            said.append(f"{ident} writes {prop} as '{says}' for '{value[:60]}'")
        return ("Not run yet (no step used). A wording check found that " + "; ".join(said) + " - the same value, put another way - and "
                "each listing involved meets all needs the shopper disclosed, is in stock and within budget, so the code placed the "
                "whole basket in the cart and read its stock: " + self.ep.basket.summary() + " Order this cart now; decline just if a "
                "record you saw has one of these lines failing a requirement that is in force.")

    def near_basket(self):
        wants = self.ep.wants
        counts = wants.counts()
        slots = list(range(len(wants.items()))) or [None]
        units = [counts[n] if n < len(counts) else 1 for n in range(len(slots))]
        budget = float(wants.spend_cap) if is_number(wants.spend_cap) else None
        basket = {}
        pools = self._wording_pools(slots, units, basket)
        if not pools:
            return ""
        settled = sum(price * many for _, price, many in basket.values())
        if budget is not None:
            least = sum(min(option[1] for option in pools[n]) * units[n] for n in pools)
            if settled + least > budget + 1e-9:
                return ""
        flat = [(n, ident, prop, value, says) for n in sorted(pools) for _, _, ident, lines in pools[n]
                for prop, value, says in lines][:12]
        self.near_done = True
        agreed = self._agreed_lines(self.ep.counsel.ask_json(self._wording_prompt(flat), 600, CHECKER))
        self.ep.log.append({"kind": "near_check", "lines": [f"{row[1]}:{row[2]}={row[3]}|{row[4]}"[:120] for row in flat],
                            "same": sorted(agreed)})
        if not self._fill_from_agreed(pools, flat, agreed, basket, units):
            return ""
        if budget is not None and sum(price * many for _, price, many in basket.values()) + 0.0 > budget + 1e-9:
            return ""
        picks = [basket[n][::2] for n in range(len(slots))]
        if self.ep.moves_left() < len(picks) + 2:
            return ""
        return self._load_reviewed(picks, pools, basket, flat)

    def merge_items(self, i, j, label, phrase, in_order=True):
        wants = self.ep.wants
        products = wants.items()
        listed = [r for r in wants.reading.get("items") or [] if isinstance(r, dict) and str(r.get("product") or "").strip()]
        if len(listed) != len(products) or not 0 <= i < j < len(products):
            return ""
        several_left = len(products) > 2
        others = [kind_tokens(r["product"]) for n, r in enumerate(products) if n not in (i, j)]
        retag = ProductTagger(kind_tokens(products[i]["product"]) + kind_tokens(products[j]["product"]), kind_tokens(label),
                              others, phrase if several_left else "")
        merged_row = {"product": phrase, "quantity": products[i]["quantity"],
                      "requirements": self._union_requirements(products[i], products[j])}
        survivors = listed[:j] + listed[j + 1:]
        survivors[i] = merged_row
        self._retag_everything(retag, phrase)
        self._reshape_reading(survivors, phrase, products[i]["quantity"], several_left)
        wants.version += 1
        self.home_memo = None
        self.ep.brief.menu_memo = {}
        self.ep.talk.quota_reads = {}
        self.ep.bargain.walked = set()
        left = len(wants.items())
        self.ep.counsel.scale = 1.0 + 0.3 * min(2, max(0, left - 1))
        self.ep.log.append({"kind": "merge_joined", "label": label, "products": [products[i]["product"], products[j]["product"]],
                            "into": phrase, "label_words": in_order, "products_left": max(1, len(wants.items()))})
        return (f"'{label}' is a SINGLE category in this shop, and the request wants one product out of it ('{phrase}'), so the order "
                + ("holds one line for it" if several_left else "holds ONE line")
                + ", not a line for every kind its name links. Requirements now: " + wants.text_body())

    def _retag_everything(self, retag, phrase):
        wants = self.ep.wants
        reading = wants.reading
        for name in ("written", "uncovered", "let_go", "indifferent"):
            setattr(wants, name, retag.unique(getattr(wants, name)))
        if wants.now_reqs is not None:
            wants.now_reqs = retag.unique(wants.now_reqs)
        wants.resolved = list(map(retag.text, wants.resolved or []))
        for field in ("requirements", "hints", "references", "soft_wants"):
            if isinstance(reading.get(field), list):
                reading[field] = retag.unique(reading[field])
        self._retag_targets(reading.get("reference_targets"), retag, phrase)
        self._retag_refs(retag)
        talk = self.ep.talk
        if isinstance(talk.pending_seconds, list):
            talk.pending_seconds = list(map(retag.text, talk.pending_seconds))
        if isinstance(talk.open_pair, list):
            talk.open_pair = list(map(retag.text, talk.open_pair))

    def _reshape_reading(self, survivors, phrase, quantity, several_left):
        reading = self.ep.wants.reading
        if several_left:
            reading["items"] = survivors
            reading["product"] = " and ".join(str(row.get("product")) for row in survivors)
            return
        reading.pop("items", None)
        reading["product"] = phrase
        reading["quantity"] = quantity

    @staticmethod
    def _retag_targets(aims, retag, phrase):
        if not isinstance(aims, list):
            return
        for aim in aims:
            if isinstance(aim, dict) and retag.ours(str(aim.get("product") or "")):
                aim["product"] = phrase

    def _retag_refs(self, retag):
        for ref in self.ep.refs or []:
            for field in ("req", "head", "now"):
                if isinstance(ref.get(field), str):
                    ref[field] = retag.text(ref[field])
            if retag.ours(str(ref.get("tag") or "")):
                ref["tag"] = retag.tag_after

    # lookup tools and what they reveal

    def lookup_key(self, tool):
        schema = self.ep.shapes.get(tool or "") or {}
        props = schema.get("properties") if isinstance(schema.get("properties"), dict) else {}
        required = schema.get("required")
        must = [k for k in required or [] if k in props] if isinstance(required, list) else []
        strings = [k for k, p in props.items() if isinstance(p, dict) and p.get("type") == "string" and not p.get("enum")]
        named = [k for k in strings if k in must]
        if named:
            slot = named[0]
        elif len(strings) == 1:
            slot = strings[0]
        else:
            return None
        return slot if all(k == slot for k in must) else None

    def lookup_tools(self):
        taken = {self.ep.role(r) for r in ("search", "message", "cart", "cart_view", "order", "decline")}
        return [name for name in self.ep.shapes if name not in taken - {None} and self.lookup_key(name)]

    def _logged_lookup(self, ident, tool):
        known = self.ep.known.ids
        for entry in reversed(self.ep.log):
            if entry.get("kind") != "env":
                continue
            for act in entry.get("actions") or []:
                args = act.get("args") if act.get("name") == tool else None
                if isinstance(args, dict) and any(isinstance(v, str) and v in known for v in args.values()):
                    return {k: ident if isinstance(v, str) and v in known else v for k, v in args.items()}
        return None

    def lookup_call(self, ident, tool):
        if not tool:
            return None
        if tool == self.ep.role("details"):
            return self.ep.counsel.details_call(ident)
        logged = self._logged_lookup(ident, tool)
        if logged is not None:
            return logged
        slot = self.lookup_key(tool)
        return {slot: ident} if slot else None

    @staticmethod
    def rec_id(d):
        exact = [v for k, v in d.items() if str(k).lower() == "id" and isinstance(v, str) and re.fullmatch(ID_LIKE, v)]
        if exact:
            return exact[0]
        loose = [v for k, v in d.items() if str(k).lower().endswith("id") and isinstance(v, str) and re.fullmatch(ID_LIKE, v)]
        return loose[0] if loose else None

    @staticmethod
    def asked_ids(call_args):
        values = list(call_args.values()) if isinstance(call_args, dict) else []
        ids = set()
        for v in values:
            if isinstance(v, str) and re.fullmatch(ID_LIKE, v):
                ids.add(v)
        for v in values:
            if isinstance(v, list):
                ids.update(x for x in v if isinstance(x, str) and re.fullmatch(ID_LIKE, x))
        return ids

    def _note_stock(self, ident, state, source, epoch, newly):
        self.stock_seen = True
        past = self.stock_by.setdefault(ident, {})
        other = past.get("row" if source == "lookup" else "lookup")
        if other and other[0] == epoch and other[1] != state:
            self.rows_lag = True
        if source == "row" and past.get("row") and past["row"][1] != state:
            self.rows_live = True
        past[source] = (epoch, state)
        if state is not False:
            if source == "lookup" or self.dead_by.get(ident) != "lookup":
                self.dead.discard(ident)
            return
        fresh = ident not in self.dead and ident in self.ep.basket.units and ident not in newly
        if fresh:
            newly.append(ident)
        self.dead.add(ident)
        self.dead_by[ident] = source

    @staticmethod
    def _carries_property_map(record):
        for v in record.values():
            if isinstance(v, dict) and len([x for x in v.values() if isinstance(x, (str, int, float))]) > 1:
                return True
        return False

    def _refusal_sold_line(self, obs):
        holding = [x for x in self.ep.basket.units if self.ep.basket.held(x) > 0]
        blocked = obs is None or len(holding) != 1 or self.ep.counsel.mute or self.ep.covering
        texts = None
        if not blocked and self.ep.last_kind == "order":
            texts = self.ep.scout.shopper_lines(self.ep.role("order") or "", obs, True)
        if not texts:
            return "", ""
        before = self._wants_state()
        note = self.ep.wants.hear(texts, None)
        after = self._wants_state()
        heard = (" " + note) if note else ""
        only_version_moved = bool(note) and after[0] != before[0] and after[1:] == before[1:]
        if not only_version_moved:
            return heard, ""
        line = holding[0]
        self.dead.add(line)
        self.dead_by[line] = "lookup"
        self.ep.log.append({"kind": "refusal_dead", "line": line})
        told = self.dead_note([line])
        return heard, (" The refusal names an item that can no longer be bought, and the cart held just one line, so the code now takes " + line
                       + " as sold out (no re-read): " + told)

    def after_put(self, call_args, obs):
        basket = self.ep.basket
        if basket.view_units(obs) is not None:
            self.cart_shown = self.cart_ver if self.cart_ver is not None else 0
        added = []
        for line, count in basket.lines(call_args):
            if count is not None and count <= 0:
                continue
            if line in basket.units and line not in self.first_puts:
                added.append(line)
        if added and basket.is_put(call_args) and not self.ep.closer.code_fault(obs):
            self.first_puts.update(added)
            self._queue_put_reads(added)

    def _queue_put_reads(self, added):
        stale = set(self.stale_lines())
        fresh = [line for line in added if line in stale]
        tool = self.stock_tool()
        if not fresh or not tool or self.ep.counsel.mute or self.ep.covering:
            return
        if self.ep.moves_left() < 4 or self.ep.time_left() < CLOSING:
            return
        reads = self.ep.reads
        queue = list(reads.put_queue or [])
        reads.put_queue = (queue + [line for line in fresh if line not in queue])[:4]
        reads.put_tool = tool
        self.ep.log.append({"kind": "after_put_read", "lines": list(reads.put_queue), "tool": tool})

    def learn_caps(self, call_name, call_args, obs):
        tool = str(call_name)
        named = self.asked_ids(call_args)
        tally = self.tool_caps.setdefault(tool, {"calls": 0, "id": 0, "stock": 0, "props": 0, "notes": 0, "price": 0})
        tally["calls"] += 1
        flags = dict.fromkeys(("id", "stock", "props", "notes", "price"), False)
        sinks = (self.tool_keys.setdefault(tool, set()), self.note_keys.setdefault(tool, set()))
        cart_side = {self.ep.role("cart"), self.ep.role("cart_view")} - {None}
        newly = []
        records = []
        if not (isinstance(obs, dict) and obs.get("error")):
            records = self.ep.counsel.dicts_in([obs])
        for rec in records:
            self._take_record(rec, tool, call_name in cart_side, named, flags, sinks, newly)
        for key in tally:
            if key in flags and flags[key]:
                tally[key] += 1
        return newly

    def _take_record(self, rec, tool, from_cart, named, flags, sinks, newly):
        ident = self.rec_id(rec)
        if not ident:
            return
        state = self.ep.lookout.stock_state(rec)
        if state is not None:
            source = "lookup" if ident in named or from_cart else "row"
            self._note_stock(ident, state, source, self.ep.cart_seq, newly)
        if ident in named:
            self._absorb_named_record(tool, ident, rec, state is not None, flags, sinks[0], sinks[1])
            return
        if from_cart:
            return
        self.row_keys.update(map(str, rec))
        self.row_text.setdefault(ident, {}).update((str(k), v) for k, v in rec.items() if isinstance(v, str))

    def menu_rows(self, home, most, taken, qty=1):
        known = self.ep.known
        budget = self.ep.wants.spend_cap
        unread, titles, sold, read_words = self._unread_menu(home)
        scores = self.ep.scout.hit_score
        hits = sorted(self.ep.scout.all_hits, key=lambda i: -scores.get(i, 0))
        order = [i for i in hits if i in titles]
        order += [i for i in unread if i not in hits]

        def affordable(ident):
            price = known.price(ident)
            return not (is_number(budget) and is_number(price) and known.price(ident) * qty > budget)
        order = [i for i in order if i not in sold and i not in taken and i not in self.ep.refused_lines and affordable(i)]
        words = {}
        for ident in order:
            words[ident] = set(re.findall(r"[a-z0-9]{3,}", titles.get(ident, "").lower()))
        try:
            chosen = self._value_led_first(home, most, words, read_words)
        except Exception:
            chosen = []
        tiers = {ident: 0 if ident in hits else 1 if self.ep.scout.row_score.get(ident) else 2 for ident in order}
        while len(chosen) < most:
            window = [i for i in order if i not in chosen][:16]
            if not window:
                break
            best = min(window, key=lambda ident: (tiers[ident], self._title_overlap(words[ident], read_words), order.index(ident)))
            chosen.append(best)
            read_words.append(words[best])
        return chosen

    @staticmethod
    def _title_overlap(mine, read_words):
        return max((len(mine & other) / max(1, len(mine | other)) for other in read_words), default=0.0)

    def _absorb_named_record(self, tool, ident, rec, stocked, flags, keys_here, notes_here):
        flags["id"] = True
        self.read_with.setdefault(tool, set()).add(ident)
        self.text_keys.setdefault(tool, set()).update(str(k) for k, v in rec.items() if v is None or isinstance(v, str))
        flags["stock"] = flags["stock"] or stocked
        if not flags["props"]:
            flags["props"] = self._carries_property_map(rec)
        if not flags["price"]:
            flags["price"] = any(is_number(v) and re.search(r"price|cost", str(k).lower()) for k, v in rec.items())
        keys_here.update(map(str, rec))
        listed = self.row_text.get(ident)
        for k, v in rec.items():
            if isinstance(v, str) and len(v.split()) >= 8:
                changed = str(k) not in self.row_keys if listed is None else listed.get(str(k)) != v
                if changed:
                    notes_here.add(str(k))
                    flags["notes"] = True

    def _guessed_roles(self):
        if self.role_guess is None:
            try:
                self.role_guess = self.ep.closer.code_roles()
            except Exception:
                self.role_guess = {}
        return self.role_guess

    def stock_tool(self):
        caps = self.tool_caps or {}
        tools = self.lookup_tools()
        with_stock = [t for t in tools if (caps.get(t) or {}).get("stock", 0) > 0]
        if with_stock:
            ep = self.ep
            return max(with_stock, key=lambda t: (caps[t]["stock"], t == ep.role("view"), t == ep.role("details"), t))
        guess = self._guessed_roles()
        for tool in (self.ep.role("view"), guess.get("view"), self.ep.role("details")):
            if tool and tool in tools:
                seen = caps.get(tool) or {}
                if seen.get("id", 0) < 2 or seen.get("stock", 0) != 0:
                    return tool
        busy = {guess.get(r) for r in ("cart", "cart_view", "order", "decline", "message")} - {None}
        for tool in tools:
            if tool not in busy and not (caps.get(tool) or {}).get("id", 0):
                return tool
        return None

    def notes_tool(self):
        caps = self.tool_caps or {}
        details = self.ep.role("details")
        own_keys = set((self.tool_keys or {}).get(details or "", set()))
        notes = self.note_keys or {}
        found = [t for t in self.lookup_tools() if t != details and set(notes.get(t) or set()) - own_keys]
        if not found:
            return None
        return max(found, key=lambda t: ((caps.get(t) or {}).get("notes", 0), t))

    # sold-out lines and their replacements

    def dead_lines(self):
        return set(self.dead or set())

    def stale_lines(self):
        seqs = self.ep.line_seq or {}
        return [line for line in self.ep.basket.units if seqs.get(line, -1) < self.ep.cart_seq]

    def _cost(self, ident):
        price = self.ep.known.price(ident)
        return float(price) if is_number(price) else float("inf")

    def _over_cap(self, price, cap, qty, rest):
        return is_number(cap) and price * qty + rest > cap

    def _passed_swaps(self, line, shut, version, label, several, cap, qty, rest):
        counsel = self.ep.counsel
        out = []
        for stamp in self.ep.passed:
            ident, _, ver = stamp.rpartition("|")
            if ver != version or ident in shut or counsel.field_view(ident)["fails"]:
                continue
            theirs = counsel.shelf_known(ident)
            if label and ((several and theirs != label) or (not several and theirs not in ("", label))):
                continue
            price = self._cost(ident)
            if price == float("inf") or not self._over_cap(price, cap, qty, rest):
                out.append(ident)
        return out

    def _ranked_swap_rows(self, shut, version, label, slot, cap, qty, rest):
        closer = self.ep.closer
        plain_pairs = [(prop, plain) for prop, plain in closer.code_values(slot) if self.ed_plain(plain)]
        failed = {stamp.split("|")[0] for stamp, verdict in self.ep.rulings.items()
                  if verdict and stamp.split("|")[1:2] == [version]}
        pointed = self.ep.pointers.ref_pointers()
        ranked = []
        for ident, row in closer.code_rows().items():
            price = row["price"] if is_number(row["price"]) else self._cost(ident)
            if ident in shut or row["stock"] is not True or (label and row["shelf"] != label):
                continue
            if price == float("inf") or self._over_cap(price, cap, qty, rest):
                continue
            if ident in failed or ident in pointed or self.ep.pointers.ref_excluded(row["fields"], slot):
                continue
            was_read = ident in self.ep.known.read
            if not was_read or self._read_row_holds(ident, row, plain_pairs):
                ranked.append((int(not was_read), price, ident))
        return sorted(ranked)

    def _read_row_holds(self, ident, row, plain_pairs):
        closer = self.ep.closer
        if self.ep.counsel.field_view(ident)["fails"]:
            return False
        for prop, plain in plain_pairs:
            if closer.code_clash(row["fields"], prop, plain):
                return False
        return all(closer.code_shows(row, prop, plain, set(), True) for prop, plain in plain_pairs)

    def slack_rows(self):
        wanted = self.ep.wants.items()
        budget = self.ep.wants.spend_cap
        if len(wanted) < 2 or not is_number(budget):
            return []
        try:
            return self._slack_choice(len(wanted), float(budget))
        except Exception as err:
            self.ep.log.append({"kind": "slack_error", "error": f"{type(err).__name__}: {err}"[:300]})
            return []

    def _slack_choice(self, count, budget):
        counts = self.ep.wants.counts()
        multipliers = [counts[k] if k < len(counts) else 1 for k in range(count)]
        floors = self._fit_floors(count, multipliers.__getitem__)
        if floors is None:
            return []
        spent = sum(floors)
        if spent <= budget + 1e-9:
            return []
        book = list(self.ep.page_book.values())
        hits = set(self.ep.scout.all_hits or [])
        queues = [self._slack_queue(k, floors, budget, multipliers.__getitem__, hits, book) for k in range(count)]
        chosen = []
        while any(queues) and len(chosen) < 3:
            for queue in filter(None, queues):
                if len(chosen) < 3:
                    chosen.append(queue.pop(0))
        if chosen:
            self.slack_total = sum(floors)
        return chosen

    def _gather_repair_picks(self, several, gone, own, version):
        pool = self._repair_pool() if several else {}
        avoid = set(gone) | {ident for ident, entry in pool.items() if entry["clash"]}
        picks = self.ep.closer.code_picks(avoid)
        checks = 0
        rounds = 0
        while several and picks and checks < 3 and rounds < 30:
            if self.ep.time_left() <= CLOSING + 30:
                break
            unchecked = [ident for ident, _ in picks if not self._picks_proven([(ident, 1)], own, version)]
            if not unchecked:
                break
            weak = [x for x in unchecked if self._weak_entry(pool.get(x))]
            if weak:
                avoid.update(weak)
                avoid.update(x for x, entry in pool.items()
                             if not entry["read"] and f"{x}|{version}" not in self.ep.passed and self._weak_entry(entry))
                rounds += 1
            else:
                checks += 1
                self.ep.counsel.check_item(unchecked[0], "ORDER")
            picks = self.ep.closer.code_picks(avoid)
        return picks

    @staticmethod
    def _weak_entry(entry):
        return not entry or not entry["revealed_ok"] or bool(entry["misses"])

    def dead_swap(self, line):
        wants, basket = self.ep.wants, self.ep.basket
        version = str(wants.version)
        label = self.ep.counsel.shelf_known(line)
        several = len(wants.items()) > 1
        cap, qty = wants.spend_cap, max(1.0, basket.held(line))
        rest = 0
        for other in basket.units:
            price = self._cost(other)
            if other != line and price != float("inf"):
                rest += price * basket.held(other)
        shut = self.dead_lines() | set(self.ep.refused_lines) | set(basket.units) | {line}
        passed = self._passed_swaps(line, shut, version, label, several, cap, qty, rest)
        if passed:
            return min(passed, key=lambda x: (self._cost(x), x)), ""
        if several and not label:
            return "", ""
        slot = None
        for n, home in enumerate(self.home_shelves() if several else []):
            if label and label in home:
                slot = n
                break
        ranked = self._ranked_swap_rows(shut, version, label, slot, cap, qty, rest)
        self._cheapest_unread_first(ranked)
        top = ranked[:3]
        self.swap_unread = [entry[2] for entry in top if entry[0] == 1]
        self.swap_page = None
        if not [entry for entry in ranked if entry[0] == 0]:
            try:
                self.swap_page = self.dead_page(slot)
            except Exception:
                self.swap_page = None
        return "", ", ".join(self._swap_label(*entry) for entry in top)

    def _cheapest_unread_first(self, ranked):
        unread = [entry for entry in ranked if entry[0] == 1]
        if not unread:
            return
        lowest = min(unread, key=lambda entry: (entry[1], entry[2]))
        if self.ep.wants.priority() == "cheapest" and all(lowest[1] < p for r, p, _ in ranked if r == 0):
            ranked.remove(lowest)
            ranked.insert(0, lowest)

    @staticmethod
    def _swap_label(r, p, ident):
        shown_price = int(p) if float(p).is_integer() else p
        return f"{ident} ({shown_price}{', read: its record shows every stated value' if r == 0 else ''})"

    def dead_page(self, k):
        page_slot = self.ep.lookout.page_arg()
        if not page_slot:
            return None
        bargain, eyes = self.ep.bargain, self.ep.lookout
        walks = [w for w in self.ep.page_book.values()
                 if bargain.walk_carries(w["args"], k) and eyes.next_page(w) is not None]
        if not walks:
            return None
        pick = max(walks, key=lambda w: (len(bargain.walk_carries(w["args"], k)), -len(w["read"])))
        return {**pick["args"], page_slot[0]: eyes.next_page(pick)}

    def swap_open(self):
        if self.ep.brief.swap_done:
            return ""
        gone = self.dead_lines()
        for ident in self.swap_unread or []:
            if ident not in self.ep.known.read and ident not in gone and ident not in self.ep.basket.units:
                return ident
        return ""

    def _dead_line_note(self, line):
        pick, rows = self.dead_swap(line)
        head = f"{line} is SOLD OUT (its latest record says so), and every order with it fails. Remove it from the cart"
        if pick:
            cheapest = ", the cheapest such listing" if self.ep.wants.priority() == "cheapest" else ""
            text = (f"{head}, then add {pick} (the item check found it meets every requirement now in force{cheapest}), "
                    "re-read its stock and price, then order.")
        elif rows:
            text = (f"{head}; no other listing for that product has been verified, so use the category's lowest-priced "
                    f"in-stock rows within the budget ({rows}; read rows whose record misses or contradicts some stated "
                    "requirement are skipped, and each unread one needs its details opened first), add the first that fits, read "
                    "its stock and price again, then order.")
        else:
            text = f"{head} and look for a different fit there."
        if pick or not self.swap_page:
            return [text]
        return [text, "Still more rows of that kind wait on the next page not yet read of the search headed by the stated value, "
                + json.dumps(self.swap_page, default=str) + "."]

    def dead_note(self, lines):
        self.ep.event_seen = True
        notes = []
        for line in lines:
            self.swap_page, self.swap_unread = None, []
            notes += self._dead_line_note(line)
        return " ".join(notes)

    def refusal_reread(self, obs=None):
        heard = ""
        try:
            heard, gone = self._refusal_sold_line(obs)
        except Exception as err:
            gone = ""
            self.ep.log.append({"kind": "refusal_gone_error", "error": f"{type(err).__name__}: {err}"[:300]})
        return heard + (gone if gone else self._queue_cart_recheck())

    def _wants_state(self):
        wants = self.ep.wants
        return (wants.version, len(wants.let_go), len(wants.uncovered), str(wants.spend_cap), "|".join(wants.props_left()))


    def _queue_cart_recheck(self):
        tool = self.stock_tool()
        lines = list(self.ep.basket.units)[:4]
        if not lines or not tool or self.ep.counsel.mute or self.ep.covering or self.ep.reads.open_queue:
            return ""
        if self.ep.moves_left() < len(lines) + 3 or self.ep.time_left() < CLOSING:
            return ""
        reads = self.ep.reads
        reads.open_queue = lines
        reads.open_recheck, reads.open_purpose, reads.open_tool = True, "cart", tool
        reads.open_say = "The order was refused; the stock of each cart line is read again (one may be gone)."
        return " The stock of the cart lines is read again (" + tool + ")."

    # putting a fitting cart together instead of declining

    def code_judges_all(self):
        items = self.ep.wants.items()
        slots = range(len(items)) if items else [None]
        plain_count = len([plain for k in slots for _, plain in self.ep.closer.code_values(k) if self.ed_plain(plain)])
        settled = self.differ_done(None)
        by_refs = len([req for req in self.ep.wants.props_left() if any(self.ep.pointers.ref_match(ref, req) for ref in settled)])
        return plain_count + by_refs >= len(list(self.ep.wants.props_left()))

    def differ_done(self, k):
        desk = self.ep.pointers
        rows, left = self.ep.wants.items(), self.ep.wants.props_left()
        out = []
        for ref in self.ep.refs or []:
            if ref.get("relation") != "differ" or ref.get("status") != "resolved":
                continue
            if not str(ref.get("value") or "").strip() or not desk.ref_live(ref):
                continue
            if not any(desk.ref_match(ref, req) for req in left):
                continue
            if rows and k is not None and ref.get("tag") and self.ep.talk.product_of(f"x ({ref['tag']})") not in (k, None):
                continue
            out.append(ref)
        return out

    def _picks_proven(self, picks, own, version):
        fit = self.ep.closer.code_fit or {}
        return bool(picks) and all(f"{ident}|{version}" in self.ep.passed or (own and ident in self.ep.known.read and fit.get(ident))
                                   for ident, _ in picks)

    def _repair_pool(self):
        pool = {}
        for k in range(len(self.ep.wants.items())):
            pool.update((entry["id"], entry) for entry in self.ed_pool(k))
        return pool


    def _repair_moment(self):
        ep = self.ep
        several = len(ep.wants.items()) >= 2
        early = several and ep.refusal_n == 0 and ep.moves_left() > ep.brief.order_due()
        late = ep.refusal_n > 0 or ep.moves_left() <= ep.brief.order_due()
        if not late and not early:
            return None
        if not ep.role("cart") or not ep.role("order") or ep.moves_left() < 3:
            return None
        if ep.time_left() <= CLOSING or ep.counsel.mute or ep.covering or ep.shop.done:
            return None
        mark = ("early", ep.refusal_n) if early else ep.refusal_n
        if mark in self.repair_tried:
            return None
        self.repair_tried.add(mark)
        return several

    def _decline_proven(self):
        try:
            proof = self.ed_proof()
        except Exception as err:
            proof = []
            self.ep.log.append({"kind": "decline_proof_error", "error": f"{type(err).__name__}: {err}"[:300]})
        if proof:
            self.ep.log.append({"kind": "decline_proof", "products": proof})
        return len(proof) > 0

    def _try_repair(self, several, gone, own, version):
        picks = self._gather_repair_picks(several, gone, own, version)
        if not self._picks_proven(picks, own, version):
            self.ep.log.append({"kind": "decline_repair", "picks": picks, "done": False})
            return False
        if self.ep.moves_left() < len(picks) + 2:
            self.ep.log.append({"kind": "decline_repair", "picks": picks, "done": False, "why": "steps"})
            return False
        done = self._place_repair(picks)
        self.ep.log.append({"kind": "decline_repair", "picks": picks, "done": done, "cart": dict(self.ep.basket.units)})
        return done

    def decline_repair(self):
        several = self._repair_moment()
        if several is None:
            return ""
        if several and self._decline_proven():
            return ""
        gone = self.dead_lines()
        stale = set(self.stale_lines())
        alive = {x for x in self.ep.basket.units if x not in gone and x not in stale}
        kept_refusals = set(self.ep.refused_lines)
        version = str(self.ep.wants.version)
        own = self.code_judges_all()
        done = False
        try:
            self.ep.refused_lines.difference_update(alive)
            done = self._try_repair(several, gone, own, version)
        except Exception as err:
            done = False
            self.ep.log.append({"kind": "decline_repair_error", "error": f"{type(err).__name__}: {err}"[:300]})
        finally:
            if not done:
                self.ep.refused_lines.clear()
                self.ep.refused_lines.update(kept_refusals)
        if not done:
            return ""
        many = bool(self.ep.wants.items())
        return ("Not run yet (no step used). Declining is proper only when nothing can work, yet the records show a candidate that fits "
                + ("for every product asked for" if many else "") + " (passed by the item check, or opened with every stated "
                "value in its record), so the code has put " + ("them" if many else "it") + " into the cart, reading its stock: "
                + self.ep.basket.summary() + " Now go and order this cart; decline just when a read record proves one of these lines "
                "fails a requirement in force.")

    def _place_repair(self, picks):
        counsel = self.ep.counsel
        quiet_before = counsel.mute
        counsel.mute = True
        try:
            done = self.ep.closer.code_put(picks) == [] and all(ident in self.ep.basket.units for ident, _ in picks)
            picked = [p for p, _ in picks]
            stale_now = [x for x in self.stale_lines() if x in picked]
            if done and stale_now and self.ep.moves_left() >= len(stale_now) + 2:
                done = not self.ep.closer.code_recheck(stale_now)[1]
        finally:
            counsel.mute = quiet_before
        return done and bool(self.ep.basket.units) and not set(self.ep.basket.units) & self.dead_lines()

    # code-side fit evidence per product

    @staticmethod
    def ed_plain(value):
        return re.search(r"\b(?:" + VALUE_LINKS + r")\b|\d\s*(?:-|to)\s*\d|[+<>]|\b(?:about|around|approx\w*|roughly)\b",
                         str(value).lower()) is None

    @staticmethod
    def ed_word(value):
        lowered = str(value).lower()
        if not re.search(r"[a-z]{3,}", lowered) or re.search(r"\d", lowered):
            return False
        return ShelfKeeper.ed_plain(lowered) and not re.search(EXCLUDE_WORDS, lowered) and not ref_text(lowered)

    def ed_reqs(self, k):
        products = self.ep.wants.items()
        out = []
        for req in self.ep.wants.props_left():
            head, _, value = str(req).partition(":")
            if not products or self.ep.talk.product_of(head) in (k, None):
                out.append((re.split(r"\s*\(", head)[0].strip(), re.sub(r"\((?:stated|revealed[^)]*)\)", "", value).strip(),
                            req not in self.ep.wants.written))
        return out

    def _clashing_props(self, fields, plain_values, joins):
        closer = self.ep.closer

        def values_for(prop):
            return [val for key, val in fields.items() if same_prop(key, prop)]
        clash = [prop for prop, wanted in plain_values if closer.code_clash(fields, prop, wanted)
                 and not any(closer.code_seen(wanted, val) for val in values_for(prop))]
        for prop, parts in joins:
            present = values_for(prop)
            if present and not all(any(closer.code_seen(part, val) for val in present) for part in parts):
                clash.append(prop)
        return clash

    def _evidence_entry(self, ident, row, ctx):
        closer = self.ep.closer
        entry = {"id": ident}
        quoted = row["price"]
        if quoted is None:
            quoted = self.ep.known.price(ident)
        entry["price"] = float(quoted) if is_number(quoted) else None
        entry["read"] = ident in self.ep.known.read
        gone = ident in ctx["gone"] or row["stock"] is False or ident in self.ep.refused_lines
        over = None not in (ctx["budget"], entry["price"]) and entry["price"] * ctx["units"] > ctx["budget"] + 1e-9
        pointer = ident in ctx["pointers"] or self.ep.pointers.ref_excluded(row["fields"], ctx["slot"])
        clash = self._clashing_props(row["fields"], ctx["plain"], ctx["joins"])
        misses = [pair for pair in ctx["plain"] if not closer.code_shows(row, pair[0], pair[1], set(), True)]
        ruled = ident in ctx["failed"] or bool(entry["read"] and self.ep.counsel.field_view(ident)["fails"])
        entry["verified"] = ident in ctx["live"] and not gone
        held_back = ruled or pointer or gone or over
        entry["fit"] = not held_back and (entry["verified"] or (entry["read"] and not misses and not clash))
        entry.update(gone=gone, over=over, pointer=pointer, ruled=ruled, clash=clash, misses=misses, row=row)
        entry["out"] = bool(held_back or clash)
        entry["revealed_ok"] = all(closer.code_shows(row, pair[0], pair[1], set(), True) for pair in ctx["revealed"])
        return entry

    def _lacking_picks(self, wanted, homes, short, limit):
        known = self.ep.known
        budget = self.ep.wants.spend_cap
        sightings, sold = self._sightings()
        label_of = {}
        for ident, label in sightings:
            if ident not in label_of:
                label_of[ident] = label
        scores = self.ep.scout.hit_score
        ranked_hits = sorted(self.ep.scout.all_hits, key=lambda i: -scores.get(i, 0))
        candidates = dict.fromkeys([i for i in ranked_hits if i in label_of] + [pair[0] for pair in sightings])
        queues = {k: [] for k in short}
        for ident in candidates:
            if ident in known.read or ident in self.ep.refused_lines or ident in sold:
                continue
            slot = self._short_slot(ident, label_of, homes, short)
            if slot is None:
                continue
            price = known.price(ident)
            too_dear = is_number(budget) and is_number(price) and price * wanted[slot]["quantity"] > budget
            if not too_dear:
                queues[slot].append(ident)
        return self._round_robin(queues, short, limit)

    @staticmethod
    def _short_slot(ident, label_of, homes, short):
        for k in short:
            if label_of.get(ident) in homes[k]:
                return k
        return None

    @staticmethod
    def _round_robin(queues, order, limit):
        chosen = []
        while len(chosen) < limit and any(queues.values()):
            for k in order:
                if queues[k] and len(chosen) < limit:
                    chosen.append(queues[k].pop(0))
        return chosen

    def _evidence_home(self, k, several):
        homes = self.home_shelves()
        if not several:
            return set(self.ep.lookout.shelf_labels()) or (set(homes[0]) if homes else set())
        return set(homes[k]) if k is not None and k < len(homes) else set()

    def _pool_mark(self, k):
        wants = self.ep.wants
        return (k, len(self.ep.known.items), len(self.ep.known.read), str(wants.version), "|".join(wants.props_left()),
                len(self.ep.passed), sum(1 for v in self.ep.rulings.values() if v), tuple(sorted(self.dead_lines())),
                tuple(sorted(self.ep.refused_lines)), str(wants.spend_cap), len(self.ep.refs or []))

    def ed_pool(self, k):
        several = self.ep.wants.items()
        mark = self._pool_mark(k)
        if mark in self.ed_memo:
            return self.ed_memo[mark]
        if len(self.ed_memo) > 64:
            self.ed_memo.clear()
        home = self._evidence_home(k, several)
        entries = self._evidence_entries(k, home, several, mark[3]) if home else []
        self.ed_memo[mark] = entries
        return entries

    def _joined_values(self, slot):
        joins = []
        for p, v in self.ep.closer.code_values(slot):
            parts = [bit.strip() for bit in v.split("+") if bit.strip()] if "+" in v else []
            if len(parts) > 1 and all(re.search(r"[a-z]{2,}", bit.lower()) for bit in parts):
                joins.append((p, parts))
        return joins

    def _evidence_entries(self, k, home, several, version):
        wants = self.ep.wants
        counts = wants.counts()
        if several and k is not None and k < len(counts):
            units = counts[k]
        else:
            units = counts[0] if counts else 1
        ctx = {"units": units}
        ctx["failed"] = {stamp.split("|")[0] for stamp, verdict in self.ep.rulings.items()
                         if verdict and stamp.split("|")[1:2] == [version]}
        ctx["live"] = {stamp.rpartition("|")[0] for stamp in self.ep.passed if stamp.rpartition("|")[2] == version}
        ctx["budget"] = float(wants.spend_cap) if is_number(wants.spend_cap) else None
        slot = k if several else None
        ctx["slot"] = slot
        ctx["plain"] = [(p, v) for p, v in self.ep.closer.code_values(slot) if self.ed_plain(v)]
        ctx["joins"] = self._joined_values(slot)
        ctx["revealed"] = [(p, v) for p, v, shown in self.ed_reqs(slot) if shown]
        ctx["pointers"] = self.ep.pointers.ref_pointers()
        ctx["gone"] = self.dead_lines()
        return [self._evidence_entry(ident, row, ctx) for ident, row in self.ep.closer.code_rows().items() if row["shelf"] in home]

    def ed_proof(self):
        proven = []
        for k in range(len(self.ep.wants.items())):
            opened = [entry for entry in self.ed_pool(k) if entry["read"]]
            if opened and not [entry for entry in opened if not entry["out"]]:
                proven.append(k)
        return proven

    # cheaper combinations when the budget is tight

    def _search_lead_scores(self, k, hits, book):
        groups = []
        for _, value, _ in self.ed_reqs(k):
            groups.append({t for t in re.findall(r"[a-z0-9]{2,}", value.lower()) if t not in LINK_WORDS})
        scores = dict.fromkeys(hits, 1)
        for search in book:
            words = [w for text in self.ep.lookout.free_text(search["args"]) for w in re.findall(r"[a-z0-9]{2,}", text.lower())]
            front = set(words[: self.ep.lead_words] if self.ep.lead_words else words)
            carried = sum(1 for group in groups if group & front)
            for ident in (search.get("row_shelf") or {}) if carried else ():
                scores[str(ident)] = max(scores.get(str(ident), 0), carried)
        return scores

    def _fit_floors(self, count, per_unit):
        floors = []
        for k in range(count):
            prices = [e["price"] for e in self.ed_pool(k) if e["fit"] and e["price"] is not None]
            if not prices:
                return None
            floors.append(min(prices) * per_unit(k))
        return floors

    def _slack_queue(self, k, floors, budget, per_unit, hits, book):
        room = (float(budget) - (sum(floors) - floors[k])) / max(1, per_unit(k))
        scores = self._search_lead_scores(k, hits, book)
        ranked = []
        for e in self.ed_pool(k):
            if e["read"] or e["gone"] or e["pointer"] or e["ruled"] or e["clash"] or e["price"] is None:
                continue
            if room > 0 and e["price"] <= room + 1e-9:
                ranked.append((-scores.get(e["id"], 0), e["price"], e["id"]))
        return [ident for _, _, ident in sorted(ranked)]


    def verify_problems(self, ident):
        latest = next((entry for entry in reversed(self.ep.log)
                       if isinstance(entry, dict) and entry.get("kind") == "verify" and entry.get("item") == ident), None)
        result = (latest or {}).get("result")
        if isinstance(result, dict) and result.get("fits") is False:
            return [str(p) for p in result.get("problems") or [] if str(p).strip()]
        return None

    def ed_news(self, k):
        if self.ep.refusal_n > 0:
            return True
        rows = self.ep.wants.items()
        for req in [*self.ep.wants.uncovered, *self.ep.wants.let_go]:
            if not rows or self.ep.talk.product_of(str(req).partition(":")[0]) in (k, None):
                return True
        return False

    # wording review: listings that state a value in other words

    def _near_cart_intact(self, info):
        wants = self.ep.wants
        if not info or info.get("ver") != wants.version or info.get("refusals") != self.ep.refusal_n or not self.ep.role("order"):
            return []
        lines = list(info.get("lines") or [])
        for ident in lines:
            if ident not in self.ep.basket.units or ident in self.dead_lines() or ident in self.ep.refused_lines:
                return []
        return lines

    def near_hold(self):
        try:
            lines = self._near_cart_intact(self.near_cart)
            shop = self.ep.shop
            if not lines or self.ep.moves_left() < 1 or shop.done or shop.broke or self.ep.time_left() < 2:
                return ""
            self.near_cart = None
            self.ep.log.append({"kind": "near_order", "lines": lines})
            quiet_before = self.ep.counsel.mute
            try:
                self.ep.closer.finish_now()
            finally:
                self.ep.counsel.mute = quiet_before
            return ("Not run (no step used). A prior check of wording found that the cart lines (" + ", ".join(lines) + ") state the "
                    "requested values in other words, and nothing has changed after it, so the code placed the order, not a decline.")
        except Exception as err:
            self.ep.log.append({"kind": "near_hold_error", "error": f"{type(err).__name__}: {err}"[:300]})
            return ""

    def near_check(self):
        counsel, shop = self.ep.counsel, self.ep.shop
        if self.near_done or not self.ep.role("cart") or not self.ep.role("order") or self.ep.moves_left() < 4:
            return ""
        if (self.ep.time_left() <= CLOSING + 30 or counsel.mute or self.ep.covering or shop.done
                or counsel.over_budget() or counsel.no_model):
            return ""
        try:
            return self.near_basket()
        except Exception as err:
            self.ep.log.append({"kind": "near_error", "error": f"{type(err).__name__}: {err}"[:300]})
            return ""

    def _reworded_lines(self, entry, reqs):
        closer = self.ep.closer
        row = entry["row"]
        out = []
        for prop, value, revealed in reqs:
            if closer.code_shows(row, prop, value, set(), True):
                continue
            written = [str(v).strip() for key, v in row["fields"].items() if same_prop(key, prop) and str(v).strip()]
            if revealed or not written or not self.ed_word(value) or any(re.search(r"\d", text) for text in written):
                return None
            out.append((prop, value, "; ".join(dict.fromkeys(written))[:120]))
        return out

    def _problem_marks(self, lines):
        marks = set()
        for prop, value, _ in lines:
            for token in prop_words(prop) + re.findall(r"[a-z0-9]+", value.lower()):
                if len(token) > 2 and token not in LINK_WORDS:
                    marks.add(token)
        return marks

    def _reworded_options(self, pool, reqs):
        options = []
        for entry in pool:
            if not entry["read"] or entry["gone"] or entry["over"] or entry["pointer"] or entry["price"] is None:
                continue
            lines = self._reworded_lines(entry, reqs)
            failed_fields, problems = set(), []
            if entry["ruled"]:
                failed_fields = {str(x).split(":", 1)[0].strip() for x in self.ep.counsel.field_view(entry["id"])["fails"]}
                problems = self.verify_problems(entry["id"]) or []
            if not lines:
                continue
            marks = self._problem_marks(lines)
            props = [prop for prop, _, _ in lines]
            if any(not any(same_prop(f, prop) for prop in props) for f in failed_fields):
                continue
            if all(any(re.search(r"\b" + re.escape(t), problem.lower()) for t in marks) for problem in problems):
                options.append((len(lines), entry["price"], entry["id"], lines))
        return options

    def _wording_prompt(self, flat):
        rows = "\n".join(f"{i + 1}. {ident}: {prop} = '{value[:60]}' | the listing's field says: {says}"
                         for i, (_, ident, prop, value, says) in enumerate(flat))
        return ("Situation: some listings do not repeat a requested value word for word, and the shopping assistant would "
                "decline them unless they state that value in other words. Shopper's request, for context:\n"
                + self.ep.request[:1500] + "\n\nEach numbered line gives a listing, one requested property with the value asked "
                "for, and the text the listing's own field shows for it.\n" + rows + "\n\nSteps:\na) Go through the lines one "
                "by one.\nb) Say true only when the field gives the requested thing in other words (a synonym, a different "
                "spelling or phrasing, or a narrower form of that same thing); every other value of the property is false.\nc) When "
                "in doubt, say false.\nOutput: a single JSON object and nothing more: {\"lines\": [{\"n\": 1, \"same\": true}]}")


    @staticmethod
    def _agreed_lines(answer):
        agreed = set()
        verdicts = answer.get("lines") if isinstance(answer, dict) else None
        for verdict in verdicts if isinstance(verdicts, list) else []:
            if isinstance(verdict, dict) and verdict.get("same") is True and is_number(verdict.get("n")):
                agreed.add(int(verdict["n"]) - 1)
        return agreed



    # cart changes and fresh stock reads

    def cart_moved(self, before):
        self.cart_ver = (self.cart_ver or 0) + 1
        self.ep.since_cart = 0
        now = self.ep.basket.units
        if [line for line in now if line not in before]:
            self.ep.cart_seq = self.ep.absorb_n
        for line, count in now.items():
            if line in before and count > before[line] + 1e-9:
                self.ep.line_seq.pop(line, None)


    def mark_fresh(self, obs):
        if not self.stock_seen:
            return
        for rec in self.ep.counsel.dicts_in([obs]):
            ident = self.rec_id(rec)
            if ident in self.ep.basket.units and self.ep.lookout.stock_state(rec) is not None:
                self.ep.line_seq[ident] = self.ep.absorb_n

    def cart_gone(self, lines):
        gone = set(lines) & self.dead_lines()
        stale = [x for x in self.stale_lines() if x in lines and x not in gone]
        if stale and self.stock_tool() and self.ep.moves_left() >= len(stale) + 2:
            sold = self.ep.closer.code_recheck(stale)[1]
            gone |= set(sold) | (set(lines) & self.dead_lines())
        return gone

    # a lookup tool that opens several listings at once

    def _cart_like_tools(self):
        busy = {self.ep.role(r) for r in ("search", "message", "cart", "cart_view", "order", "decline")} - {None}
        spoken = {}
        for tdef in self.ep.toolset or []:
            fn = tdef.get("function") if isinstance(tdef, dict) else None
            if isinstance(fn, dict) and isinstance(fn.get("name"), str):
                spoken[fn["name"]] = (re.sub(r"([a-z])([A-Z])", r"\1 \2", fn["name"]).replace("_", " ") + " "
                                      + str(fn.get("description") or ""))
        pattern = r"\b(?:remov|delet|cancel|clear|discard|releas|refund|purchas|checkout|wish|basket|cart|bag)"
        for name in list(self.ep.shapes):
            if re.search(pattern, spoken.get(name, name).lower()):
                busy.add(name)
        return busy

    @staticmethod
    def _id_list_slot(schema):
        if not isinstance(schema, dict):
            schema = {}
        props = schema.get("properties") if isinstance(schema.get("properties"), dict) else {}
        arrays = [key for key, spec in props.items()
                  if isinstance(spec, dict) and spec.get("type") == "array" and isinstance(spec.get("items"), dict)
                  and spec["items"].get("type") == "string" and not spec["items"].get("enum")]
        if len(arrays) != 1:
            return None
        required = schema.get("required") if isinstance(schema.get("required"), list) else []
        if [r for r in required if r not in arrays]:
            return None
        most = props[arrays[0]].get("maxItems")
        return (arrays[0], int(min(most, 10))) if is_number(most) and most >= 2 else None

    def batch_tool(self):
        if not self.ep.batch_spec_known:
            spec = None
            try:
                busy = self._cart_like_tools()
                for name, schema in self.ep.shapes.items():
                    slot = None if name in busy else self._id_list_slot(schema)
                    if slot:
                        spec = (name,) + slot
                        break
            except Exception:
                spec = None
            self.ep.batch_spec = spec
            self.ep.batch_spec_known = True
        return self.ep.batch_spec

    @staticmethod
    def _looks_rich(values):
        nested = [v for v in values if isinstance(v, dict)]
        if nested:
            return any(sum(1 for x in v.values() if isinstance(x, (str, int, float))) >= 2 for v in nested)
        return len([v for v in values if isinstance(v, str) and 0 < len(v) <= 70]) >= 5

    def batch_ok(self, found, ids):
        wanted = set(ids)
        got = set()
        try:
            for rec in self.ep.counsel.dicts_in(found):
                values = list(rec.values())
                if self._looks_rich(values):
                    got |= {v for v in values if isinstance(v, str) and v in wanted}
        except Exception:
            return False
        return len(got) > 0 and 2 * len(got) >= len(wanted)

    @staticmethod
    def title_of(row):
        title = row.get("title")
        if isinstance(title, str):
            return title
        for key, v in row.items():
            if isinstance(v, str) and re.search(r"title|heading|headline", str(key).lower()):
                return v
        return ""

    # listings opened before the first question

    def _value_targets(self, several, slot):
        values = []
        for req in self.ep.wants.props_left():
            head, _, tail = req.partition(":")
            if several and self.ep.talk.product_of(head) not in (slot, None):
                continue
            value = re.sub(r"\((?:stated|revealed[^)]*)\)", "", tail)
            if value.strip() and positive_value(value) and not re.search(r"as shown|same as|the page|sold by|redirect|refer", value.lower()):
                values.append(value)
        return values

    def _read_facts(self, home):
        facts = {}
        for rec in self.ep.counsel.dicts_in(self.ep.known.items):
            ident = self.ep.lookout.row_mark(rec)
            if ident in self.ep.known.read and self.ep.lookout.shelf_of(rec) in home:
                for val in rec.values():
                    if isinstance(val, dict):
                        facts.setdefault(ident, []).extend(str(x) for x in val.values() if isinstance(x, (str, int, float)))
        return facts

    def _value_led_first(self, home, most, words, known):
        several = bool(self.ep.wants.items())
        slot = next((n for n, h in enumerate(self.home_shelves()) if h == home), None) if several else None
        values = self._value_targets(several, slot)
        if not values or most <= 0 or (several and slot is None):
            return []
        facts = self._read_facts(home)
        if any(all(value_seen(v, " | ".join(texts)) for v in values) for texts in facts.values()):
            return []
        bargain = self.ep.bargain
        carrying = sorted((w for w in self.ep.page_book.values() if bargain.walk_carries(w["args"], slot)),
                          key=lambda w: -len(bargain.walk_carries(w["args"], slot)))
        for walk in carrying:
            lead = next((ident for ident in walk.get("row_shelf") or {} if ident in words), None)
            if lead:
                known.append(words[lead])
                return [lead]
        return []

    def _unread_menu(self, home):
        unread = []
        titles = {}
        sold = set()
        read_words = []
        for rec in self.ep.counsel.dicts_in(self.ep.known.items):
            ident = self.ep.lookout.row_mark(rec)
            if ident not in self.ep.known.ids or self.ep.lookout.shelf_of(rec) not in home:
                continue
            if ident in self.ep.known.read:
                read_words.append(set(re.findall(r"[a-z0-9]{3,}", self.title_of(rec).lower())))
                continue
            if self.ep.lookout.sold_out(rec):
                sold.add(ident)
            if ident not in unread:
                unread.append(ident)
                titles[ident] = self.title_of(rec)
        return unread, titles, sold, read_words


    def _queue_menu_reads(self, picks, say):
        reads = self.ep.reads
        reads.open_queue = picks
        reads.open_recheck = True
        reads.open_say = say
        reads.open_purpose = "menu"

    def _rich_reads_on(self, home):
        if not home:
            return 4
        return sum(self.ep.known.rich_reads(label) for label in home)

    def pre_read(self):
        try:
            batch = self.batch_tool()
            again = self.pre_rounds == 1
            if not batch or (self.pre_read_done and not again) or len(self.ep.questions_log) > 1:
                return False
            if self.ep.moves_left() <= (25 if again else 20) or self.ep.time_left() < CLOSING + 60:
                return False
            homes = self.home_shelves()
            quantities = [row["quantity"] for row in self.ep.wants.items()] or [self.ep.wants.quantity()]

            def qty_for(k):
                return quantities[min(k, len(quantities) - 1)]
            if again:
                return self._second_menu_round(homes, batch, qty_for)
            return self._first_menu_round(homes, batch, qty_for)
        except Exception as err:
            self.ep.log.append({"kind": "pre_read_error", "error": f"{type(err).__name__}: {err}"[:300]})
            return False

    def _first_menu_round(self, homes, batch, qty_for):
        bare = []
        thin = []
        for k, home in enumerate(homes):
            have = self._rich_reads_on(home)
            if not have:
                bare += self.menu_rows(home, 2 if self.ep.wants.items() else 4, bare, qty_for(k))
            elif have < 4:
                thin.append((k, 4 - have))
        order = list(bare)
        for k, missing in thin:
            order += self.menu_rows(homes[k], missing, order, qty_for(k))
        rounds = 2 if len(bare) > batch[2] and self.ep.moves_left() > 25 else 1
        order = order[: batch[2] * rounds]
        if not order:
            return False
        self.pre_read_done = True
        self.pre_rounds = 1
        self.pre_picked = list(order)
        self._queue_menu_reads(order, "First I read some listings of each requested product, so their fields are known before questions are sent.")
        return True

    def _second_menu_round(self, homes, batch, qty_for):
        self.pre_rounds = 2
        queues = []
        claimed = []
        for k, home in enumerate(homes):
            have = self._rich_reads_on(home)
            if have < 3:
                more = self.menu_rows(home, 4 - have, claimed + list(self.pre_picked or []), qty_for(k))
                claimed += more
                queues.append(more)
        order = []
        while any(queues):
            order += [queue.pop(0) for queue in queues if queue]
        order = order[: batch[2]]
        if not order:
            return False
        self._queue_menu_reads(order, "Some more listings first: for a few products, the ones read before held only a few fields.")
        return True

    def lacking_rows(self):
        try:
            wanted = self.ep.wants.items()
            if not wanted:
                return []
            homes = self.home_shelves()
            version = str(self.ep.wants.version)
            verified = {stamp.rpartition("|")[0] for stamp in self.ep.passed if stamp.rpartition("|")[2] == version}
            verified -= self.ep.refused_lines
            verified -= self.dead_lines()
            covered = {self.ep.counsel.shelf_known(i) for i in verified | set(self.ep.counsel.audit_fits or [])} - {""}
            short = [k for k, home in enumerate(homes) if home and not home & covered]
            if not short:
                return []
            batch = self.batch_tool()
            return self._lacking_picks(wanted, homes, short, batch[2] if batch else 3)
        except Exception as err:
            self.ep.log.append({"kind": "lacking_error", "error": f"{type(err).__name__}: {err}"[:300]})
            return []

    def _sightings(self):
        seen = []
        sold = set()
        for rec in self.ep.counsel.dicts_in(reversed(self.ep.known.items)):
            ident, label = self.ep.lookout.row_mark(rec), self.ep.lookout.shelf_of(rec)
            if ident in self.ep.known.ids and label:
                seen.append((ident, label))
                if self.ep.lookout.sold_out(rec):
                    sold.add(ident)
        return seen, sold



class Closer:
    def __init__(self, ep):
        self.ep = ep
        self.reread = None
        self.news_read = set()
        self.code_fit = {}
        self.code_hits = {}
        self.code_more = None
        self.code_searched = None

    def _schema(self, tool):
        return self.ep.shapes.get(tool or "") or {}

    @staticmethod
    def _props(schema):
        found = schema.get("properties")
        return found if isinstance(found, dict) else {}

    @staticmethod
    def _free_text_fields(fields, skip=None):
        return [name for name, spec in fields.items()
                if isinstance(spec, dict) and spec.get("type") == "string" and not spec.get("enum") and name != skip]

    # arguments for the finishing tools

    def schema_args(self, tool, text, kind=""):
        schema = self.ep.shapes.get(tool) or {}
        fields = schema.get("properties") or {}
        wording = {"decline": DECLINE_WORDS, "order": ORDER_WORDS}.get(kind, "")
        filled = {}
        for field in schema.get("required") or []:
            known, value = self._required_value(fields.get(field) or {}, text, wording)
            if known:
                filled[field] = value
        if len(fields) == 1:
            (field, spec), = fields.items()
            if field not in filled and isinstance(spec, dict) and spec.get("type") == "string" and "enum" not in spec:
                filled[field] = text[: spec.get("maxLength", len(text))]
        return filled

    @staticmethod
    def _required_value(spec, text, wording):
        options = spec.get("enum")
        if options:
            fitting = [c for c in options if wording and isinstance(c, str) and re.search(wording, c.lower())]
            return True, (fitting or options)[0]
        ftype = spec.get("type")
        if ftype == "string" or "string" in (ftype or []):
            return True, text[: spec.get("maxLength", len(text))]
        if ftype == "boolean":
            return True, True
        if ftype in ("integer", "number"):
            return True, max(1, spec.get("minimum", 1))
        return False, None

    def code_picks(self, avoid=None, keep=None):
        wants = self.ep.wants
        products = wants.items()
        counts = wants.counts()
        homes = self.ep.shop.home_shelves() if products else [
            set(self.ep.lookout.shelf_labels()) or (self.ep.shop.home_shelves() or [set()])[0]]
        ctx = {"table": self.code_rows()}
        version = str(wants.version)
        ctx["failed"] = {key.split("|")[0] for key, verdict in self.ep.rulings.items()
                         if verdict and key.split("|")[1:2] == [version]}
        ctx["live"] = {key.rpartition("|")[0] for key in self.ep.passed if key.rpartition("|")[2] == version}
        ctx["audited"] = self._audit_silent(ctx["live"])
        ctx["trust"] = bool(ctx["audited"]) and self.ep.shop.code_judges_all()
        cap = ctx["cap"] = float(wants.spend_cap) if is_number(wants.spend_cap) else None
        ctx["request"] = kind_words(self.ep.request)
        ctx["hits"] = self.code_hits or {}
        ctx["cheapest"] = wants.priority() == "cheapest"
        ctx["keep"] = set(keep or ())
        ctx["avoid"] = set(avoid or ()) | self.ep.shop.dead_lines() | self.ep.pointers.ref_pointers()
        pools = []
        for k, home in enumerate(homes):
            pool = self._pick_pool(k, home, products, counts, ctx)
            if not pool:
                return []
            pools.append(sorted(pool))
        first = self._choose_from(pools, None, counts, cap)
        if first is None:
            return []
        if first:
            return first
        return self._choose_from(pools, lambda e: (e[2], e[3], e[8], e[-1]), counts, cap) or []

    def _names_listing(self, call_args):
        known = self.ep.known.ids
        for value in call_args.values():
            if isinstance(value, str):
                if any(bit and bit in known for bit in re.split(r"[\s.,;:!?()\[\]{}<>\"'`]+", value)):
                    return True
        return False

    def scrub_decline(self, call_args, call_name):
        rule = self.ep.wants.reading.get("decline_no_names")
        if rule is False or (not rule and not self._names_listing(call_args)):
            return call_args
        fields = (self.ep.shapes.get(call_name) or {}).get("properties") or {}
        cleaned = dict(call_args)
        for field, value in call_args.items():
            spec = fields.get(field) or {}
            if isinstance(value, str) and "enum" not in spec:
                cleaned[field] = SORRY[: spec.get("maxLength", len(SORRY))]
        return cleaned

    # reading the request without an LLM

    def code_reading(self):
        text = self.ep.request
        lowered = text.lower()
        plan = {"code_reading": True, "requirements": [], "items": [], "quantity": 1, "rules": [],
                "soft_wants": [], "ask_first": [], "hints": [], "references": [], "priority_detail": ""}
        plan["price_cap"], plan["currency"] = self._budget_from(text)
        stock_cue = re.search(r"\bin[- ]stock\b|\bon hand\b|\bon the shelf\b|\bsold[- ]out\b|\bback[- ]?order|"
                              r"\bout of stock\b|\bavailable now\b", lowered)
        plan["in_stock_only"] = stock_cue is not None
        cheapest = re.search(r"\bcheapest\b|\blowest[- ](?:price|priced|cost)\b|\bleast expensive\b|"
                             r"\blowest\b[^.]{0,20}\bprice\b|\bsmallest (?:total|price)\b", lowered)
        plan["priority"] = "none" if cheapest is None else "cheapest"
        limits = (("message_budget", r"messages?|questions?|notes?|repl(?:y|ies)|texts?", 0),
                  ("step_budget", r"steps?|moves?|actions?|(?:tool )?calls?|turns?", FIRST_ORDER_BY + 3))
        for key, nouns, least in limits:
            found = self._count_limit(lowered, nouns)
            if found is not None and found >= least:
                plan[key] = found
        roles = self.code_roles()
        plan["roles"] = roles
        products = self._counted_products(roles.get("search"))
        if len(products) > 1:
            plan["items"] = [{"product": row["product"], "quantity": row["quantity"], "requirements": row["requirements"]}
                             for row in products]
            plan["product"] = " and ".join(row["product"] for row in products)
            plan["requirements"] = sum((row["requirements"] for row in products), [])
            return plan
        if not products:
            plan["product"] = self._closest_label(roles.get("search"), kind_words(text))
            plan["quantity"] = self._unit_count(lowered)
        else:
            plan["product"], plan["quantity"] = products[0]["label"], products[0]["quantity"]
        plan["requirements"] = [name + ": " + value for name, value in self._stated_pairs(lowered)]
        return plan

    @staticmethod
    def _amount_score(sentence, hit, tail, base_score):
        lead = sentence[max(0, hit.start() - 60): hit.start()]
        score = base_score
        if hit.group(1) or re.search(r"\b(?:spend\w*|budget\w*|afford\w*|pay\w*|price[ds]?|costs?|cash|money)\b(?:\W+[\w']+){0,4}\W*$", lead):
            score += 3
        if re.search(r"\b(?:under|below|no more than|not more than|up to|within|at most|max\w*|less than|cheaper than|"
                     r"not (?:more|over|above)|limit\w*|reach\w*|go to|to|at|cap\w*|past|exceed\w*|beyond|go over)\W+(?:[\w']+\W+)?$", lead):
            score += 2
        if re.match(r"\s*(?:[a-z.()]+\s*){0,3}?(?:or (?:lower|less|under|below|cheaper)|at (?:the (?:very )?)?most|max\w*|tops\b|"
                    r",?\s*(?:and )?not (?:one|a) \w+ (?:over|more|above))", tail):
            score += 2
        return score

    @staticmethod
    def _skip_amount(sentence, hit, tail):
        if re.match(r"\s*(?:x\b|×|%|:\s*\d|-\s*\d|/\s*\d|(?:[a-z-]+\s+){0,2}?(?:messages?|questions?|notes?|repl\w*|texts?|"
                    r"steps?|moves?|actions?|(?:tool )?calls?|turns?|units?|items?|pieces?|packs?|pairs?|lots?|sets?|of\b|times?)\b)", tail):
            return True
        return not hit.group(1) and bool(re.search(r"[\d.,:/-]\s*$|\bx\s*$", sentence[:hit.start()]))

    @staticmethod
    def _unit_after(tail, sign):
        after = re.match(r"\s*((?:[a-z][a-z.]*\s*){0,2})", tail)
        words = after.group(1).split() if after else []
        unit = [w.strip(".") for w in words if w.strip(".") not in LINK_WORDS
                and not re.fullmatch(r"or|and|not|tops|max\w*|only|just|is|was", w.strip("."))]
        return " ".join(unit) or (sign or "").strip() or None

    @staticmethod
    def _budget_from(text):
        best = (0, 0.0)
        found = (None, None)
        for sentence in re.split(r"(?<=[.!?;])\s+", str(text).lower()):
            base_score = 1 if re.search(r"\b(?:order|total|whole|lot|sum|bill|altogether|overall|everything)\b", sentence) else 0
            for hit in re.finditer(r"([$€£]\s*)?(\d[\d,]*(?:\.\d+)?)", sentence):
                tail = sentence[hit.end():]
                if Closer._skip_amount(sentence, hit, tail):
                    continue
                score = Closer._amount_score(sentence, hit, tail, base_score)
                amount = float(hit.group(2).replace(",", ""))
                if score >= 2 and (score, amount) > best:
                    best = (score, amount)
                    found = (int(amount) if amount.is_integer() else amount, Closer._unit_after(tail, hit.group(1)))
        return found

    @staticmethod
    def _count_limit(lowered, nouns):
        spelled = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
                   "twelve": 12, "fifteen": 15, "twenty": 20}
        pattern = (r"\b(\d+|one|two|three|four|five|six|seven|eight|nine|ten|twelve|fifteen|twenty)\s+(?:[a-z-]+\s+){0,2}?(?:"
                   + nouns + r")\b")
        for hit in re.finditer(pattern, lowered):
            before = lowered[max(0, hit.start() - 60): hit.start()]
            after = lowered[hit.end(): hit.end() + 40]
            cued = re.search(r"\b(?:no more than|not more than|up to|at most|only|max\w*|use|using|take|send|allow\w*|receive|get|total of|"
                             r"limit\w*|inside|within|under|fewer than|less than|you've|you have|have|keep\w*|stick to|go past|past|"
                             r"exceed\w*|beyond)\W+(?:[\w']+\W+){0,3}$", before)
            if not cued and not re.match(r"\W*(?:[\w']+\W+){0,3}?(?:in total|total|no more|at most|max\w*|tops|or fewer|or less|allowed|limit)\b", after):
                continue
            word = hit.group(1)
            return int(word) if word.isdigit() else spelled[word]
        return None

    @staticmethod
    def _unit_count(lowered):
        hit = re.search(r"\b(\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s+(?:units?|of (?:them|those|these)|copies)\b", lowered)
        if not hit:
            return 1
        count = int(hit.group(1)) if hit.group(1).isdigit() else number_words()[hit.group(1)]
        return count if 0 < count < 100 else 1

    @staticmethod
    def _pair_rejected(name, value, following):
        if ref_text(value):
            return True
        if ref_text(following) and re.search(r"\b(?:page|listing|shown|sold by)\b", following):
            return True
        if not value or len(value.split()) > 4 or len(name.split()) > 4:
            return True
        if re.search(r"\b(?:i|you|we|it|he|she|they|my|your|our|there|this|that|which|what|who|me|us|them|one|ones|price|budget|cost|total|order|stock)\b", name):
            return True
        return bool(re.match(r"(?:a|an|the|my|your|our|his|her|their|me|us|for|to|in|on|at|of|see|go|get|be|have|only|too|very|so|not|no|what|how|it|all|"
                             r"mostly|always|never|just|really|still|also|from|about|like|one|ones)\b", value))

    @staticmethod
    def _stated_pairs(stretch):
        verbs = r"(?:is|are|has to be|have to be|must be|should be|needs to be|need to be|can be)"
        stop = r"(?=\s*(?:[,.;:!?()]|$)|\s+(?:so|since|as|because|for|to|which|that|nothing|otherwise|but|when|while|if|then|though|please|and (?:the|its|their|a|an|one|two|three|\d))\b)"
        pattern = (r"(?:\b(?:where|whose)\s+(?:the\s+|its\s+|their\s+)?|(?:\b(?:and|with|but)\s+|[,;:]\s*|^|[.!?]\s+)(?:the|its|their)\s+)"
                   r"([a-z][a-z'&/ -]{1,32}?)\s+" + verbs + r"\s+(?:either\s+|just\s+)?([^,.;:!?()]{1,60}?)" + stop)
        pairs = []
        for hit in re.finditer(pattern, stretch):
            name, value = hit.group(1).strip(), hit.group(2).strip()
            if not Closer._pair_rejected(name, value, stretch[hit.end(): hit.end() + 60]):
                pairs.append((name, value))
        return pairs

    def _category_choices(self, tool):
        fields = self._schema(tool).get("properties")
        best = []
        for spec in fields.values() if isinstance(fields, dict) else []:
            if isinstance(spec, dict) and isinstance(spec.get("enum"), list) and len(spec["enum"]) >= 4:
                if not best or len(spec["enum"]) > len(best):
                    best = spec["enum"]
        return best

    def _label_span(self, label, tokens, stems, lowered, spans):
        wanted = [w.rstrip("s") for w in re.findall(r"[a-z]{3,}", label.lower()) if w not in LINK_WORDS]
        width = len(wanted)
        if not width:
            return
        for at in range(len(stems) - width + 1):
            if stems[at: at + width] != wanted:
                continue
            begin, finish = tokens[at][1], tokens[at + width - 1][2]
            if len(re.findall(r"[a-z0-9]+", lowered[begin:finish])) > 2 * width:
                continue
            counter = re.search(r"\b(\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s*(?:x\s+|×\s*)?(?:(?:of|[a-z'&-]{3,})\s+){0,4}$",
                                lowered[max(0, begin - 56): begin])
            if not counter or label in spans:
                continue
            word = counter.group(1)
            count = int(word) if word.isdigit() else number_words()[word]
            if 1 <= count <= 99:
                spans[label] = (begin, finish, count)
            return

    def _counted_products(self, tool):
        lowered = self.ep.request.lower()
        tokens = [(hit.group(0).rstrip("s"), hit.start(), hit.end())
                  for hit in re.finditer(r"[a-z]{3,}", lowered) if hit.group(0) not in LINK_WORDS]
        stems = [tok[0] for tok in tokens]
        spans = {}
        for label in map(str, self._category_choices(tool)):
            self._label_span(label, tokens, stems, lowered, spans)
        ordered = sorted(spans.items(), key=lambda pair: pair[1][0])
        products = []
        for index, (label, (_, finish, count)) in enumerate(ordered):
            if index + 1 < len(ordered):
                stretch = lowered[finish: ordered[index + 1][1][0]]
            else:
                stretch = re.split(r"(?<=[.!?])\s", lowered[finish:])[0]
            name = label.lower()
            products.append({"product": name, "label": label, "quantity": count,
                             "requirements": [f"{prop} ({name}): {value}" for prop, value in self._stated_pairs(stretch)]})
        return products

    def code_roles(self):
        described = {}
        for tdef in self.ep.toolset:
            fn = tdef["function"]
            described[fn["name"]] = (re.sub(r"([a-z])([A-Z])", r"\1 \2", fn["name"]).replace("_", " ").lower()
                                     + " | " + str(fn.get("description") or "").lower())
        roles = {}
        self._roles_from_choices(described, roles)
        self._roles_from_titles(described, roles)
        self._roles_from_cues(described, roles)
        return roles

    def _required_of(self, tool):
        listed = self._schema(tool).get("required")
        if not isinstance(listed, list):
            return []
        return [field for field in listed if field in self._props(self._schema(tool))]

    def _roles_from_choices(self, described, roles):
        for tool in described:
            for field in self._required_of(tool):
                spec = self._props(self._schema(tool))[field]
                options = [str(c).lower() for c in spec.get("enum") or []] if isinstance(spec, dict) else []
                if any(re.search(ORDER_WORDS, c) for c in options) and any(re.search(DECLINE_WORDS, c) for c in options):
                    roles.setdefault("order", tool)
                    roles.setdefault("decline", tool)

    def _roles_from_titles(self, described, roles):
        for tool, words in described.items():
            if tool in roles.values():
                continue
            title = words.split(" | ")[0]
            if "decline" not in roles and re.search(DECLINE_WORDS, title):
                roles["decline"] = tool
            elif "order" not in roles and re.search(ORDER_WORDS, title):
                roles["order"] = tool
        for tool, words in described.items():
            if tool in roles.values() or self._required_of(tool):
                continue
            final = re.search(r"\bfinal\b", words)
            if "decline" not in roles and final and re.search(DECLINE_WORDS, words):
                roles["decline"] = tool
            elif "order" not in roles and final and re.search(ORDER_WORDS, words):
                roles["order"] = tool
        for tool, words in described.items():
            if tool in roles.values() or not re.search(r"\b(?:cart|basket|bag|trolley)\b", words):
                continue
            takes_list = any(isinstance(spec, dict) and spec.get("type") == "array"
                             for spec in self._props(self._schema(tool)).values())
            roles.setdefault("cart" if self._required_of(tool) or takes_list else "cart_view", tool)

    def _roles_from_cues(self, described, roles):
        person = r"\b(?:shoppers?|customers?|buyers?|clients?|users?)\b"
        speak = r"\b(?:send\w*|repl\w*|notes?|tell\w*|messag\w*|chat\w*|writ\w*|talk\w*)\b"
        cues = (("search", r"\b(?:search\w*|find\w*|look\w*|locat\w*|hunt\w*|brows\w*|quer\w*)\b", False),
                ("message", speak, True),
                ("details", r"\b(?:details?|records?|spec\w*|propert\w*|attribut\w*|facts?|full)\b", True),
                ("view", r"\b(?:offer|price|stock|availab\w*)\b", True))
        for role, cue, single_text in cues:
            for tool, words in described.items():
                if role in roles or tool in roles.values() or not re.search(cue, words):
                    continue
                if role == "search" and re.search(person, words) and re.search(speak, words):
                    continue
                if role == "message" and not re.search(person, words) and not re.search(r"\b(?:messag\w*|chat\w*)\b", words):
                    continue
                fields = self._props(self._schema(tool))
                texts = self._free_text_fields(fields)
                if single_text:
                    needed = self._required_of(tool)
                    if len(needed) != 1 or needed[0] not in texts:
                        continue
                elif not texts and not any(isinstance(spec, dict) and spec.get("enum") for spec in fields.values()):
                    continue
                roles[role] = tool

    def _closest_label(self, tool, wanted):
        chosen, best = "", (0, 0.0)
        for label in self._category_choices(tool):
            own = kind_words(label)
            shared = len(own & wanted)
            rank = (shared, shared / max(1, len(own)))
            if shared and rank > best:
                chosen, best = str(label), rank
        return chosen

    def code_values(self, k):
        products = self.ep.wants.items()
        pairs = []
        for need in self.ep.wants.props_left():
            head, _, value = str(need).partition(":")
            if products and self.ep.talk.product_of(head) not in (k, None):
                continue
            bare = re.sub(r"\((?:stated|revealed[^)]*)\)", "", value).strip()
            if not bare or ref_text(bare):
                continue
            short = len(re.findall(r"[a-z0-9]+", bare.lower())) <= 5 and not re.search(EXCLUDE_WORDS, bare.lower())
            if positive_value(value, head) or short:
                pairs.append((re.split(r"\s*\(", head)[0].strip(), bare))
        return pairs

    # searches the code runs on its own

    def _search_layout(self, tool, must, fields):
        described = [str(t["function"].get("description") or "") for t in self.ep.toolset if t["function"]["name"] == tool]
        clauses = [c.lower() for c in re.split(r"[.;]", described[0] if described else "")]
        free = self._free_text_fields(fields, self.ep.lookout.token_key())
        named = [f for f in free if re.search(r"^q$|query|text|keyword|needle|term|word|phrase|string|search", f.lower())]
        text_field = (named or free or [None])[0]
        choices = {}
        for f, spec in fields.items():
            if isinstance(spec, dict) and isinstance(spec.get("enum"), list) and spec["enum"]:
                choices[f] = spec["enum"]
        wide = [f for f in choices if len(choices[f]) >= 4]
        shelf_field = max(wide, key=lambda f: len(choices[f])) if wide else None
        mode_fields = [f for f in choices if f != shelf_field and len(choices[f]) <= 4]
        required_modes = [f for f in mode_fields if f in must]
        mode_field = (required_modes or mode_fields or [None])[0]
        return clauses, text_field, choices, shelf_field, mode_field

    def code_queries(self, k):
        wants = self.ep.wants
        tool = self.ep.role("search")
        schema = self._schema(tool)
        fields = self._props(schema)
        must = [f for f in schema.get("required") or [] if f in fields]
        listed = wants.items()
        if listed and k is not None and k < len(listed):
            product = str(listed[k]["product"])
        else:
            product = str(wants.reading.get("product") or "")
        label = self._closest_label(tool, kind_words(product) or kind_words(self.ep.request))
        clauses, text_field, choices, shelf_field, mode_field = self._search_layout(tool, must, fields)
        layout = {"schema": schema, "must": must, "fields": fields, "clauses": clauses, "choices": choices,
                  "mode": mode_field, "limit": self._price_ceiling_arg(fields, clauses, k)}
        spec = fields.get(text_field) if text_field else None
        phrases = self._query_phrases(k, product, label, spec.get("maxLength") if isinstance(spec, dict) else None)
        layout["fallback"] = phrases[0] or label
        wanted = []
        if text_field and phrases[0]:
            wanted.append((text_field, phrases[0], False))
        if label and shelf_field:
            wanted.append((shelf_field, label, True))
        if text_field and len(phrases) > 1:
            wanted.append((text_field, phrases[1], False))
        found = []
        for field, value, use_last in wanted:
            query = self._checked_query(layout, field, value, use_last)
            if query is not None and query not in found:
                found.append(query)
        return found

    def _checked_query(self, layout, field, value, use_last):
        schema, limit = layout["schema"], layout["limit"]
        draft = {field: value}
        mode_field = layout["mode"]
        if mode_field:
            draft[mode_field] = self._mode_choice(layout["choices"].get(mode_field or "", []), layout["clauses"], field,
                                                  use_last)
        draft.update(limit)
        self._fill_required(draft, layout["must"], layout["fields"], layout["fallback"])
        fixed, faults = ArgCheck.execute(draft, schema)
        if faults and limit:
            fixed, faults = ArgCheck.execute(dict((f, v) for f, v in draft.items() if f not in limit), schema)
        return None if faults else fixed

    def _price_ceiling_arg(self, fields, clauses, k):
        ceiling = self._ceiling_field(fields, clauses)
        wants = self.ep.wants
        budget = wants.spend_cap
        counts = wants.counts()
        units = (counts[k] if k is not None and k < len(counts) else wants.quantity()) or 1
        if not ceiling or not is_number(budget) or budget <= 0:
            return {}
        if fields[ceiling].get("type") == "integer":
            return {ceiling: int(budget // units)}
        return {ceiling: round(budget / units, 2)}

    def _next_page_args(self, call_args, obs):
        paging = self.ep.lookout.page_arg()
        page = self._first_listing(obs) if paging else None
        if page is None:
            return None
        field, per_page, lowest, highest = paging
        shown = max(len(v) for v in page.values() if isinstance(v, list))
        markers = []
        matched = None
        for key, value in page.items():
            lowered = str(key).lower()
            if re.search(r"next|onward|cursor|bookmark|token|continu|more", lowered):
                markers.append(value)
        for key, value in page.items():
            if is_number(value) and not isinstance(value, bool) and re.search(r"count|total|matched|found|hits", str(key).lower()):
                matched = value
                break
        if markers and not [m for m in markers if m not in (None, False, "", 0)]:
            return None
        target = self._page_target(call_args, field, per_page, lowest, highest, shown, markers, matched)
        if target is None:
            return None
        fixed, faults = ArgCheck.execute({**call_args, field: target}, self._schema(self.ep.role("search")))
        return fixed if not faults else None

    @staticmethod
    def _page_target(call_args, field, per_page, lowest, highest, shown, markers, matched):
        if per_page == "token":
            return next((m for m in markers if isinstance(m, str) and m.strip()), None) or None
        current = call_args.get(field, lowest)
        current = current if is_number(current) else lowest
        upcoming = current + 1 if per_page else current + shown
        if highest is not None and upcoming > highest:
            return None
        if not markers:
            reached = (current - lowest + 1) * shown if per_page else current + shown
            if matched is None or reached >= matched:
                return None
        return int(upcoming) if float(upcoming).is_integer() else upcoming

    def code_put(self, picks):
        basket = self.ep.basket
        wanted = {ident for ident, _ in picks}
        line_spec = basket.line_spec()
        if line_spec:
            to_put = list(picks)
        else:
            to_put = [pick for pick in picks if abs(basket.held(pick[0]) - pick[1]) > 1e-9]
        removals = self._drop_args([line for line in basket.units if line not in wanted])
        additions = self._put_args(to_put) if to_put else []
        if removals is None or (to_put and not additions):
            return None
        calls = removals + additions
        if not calls or self.ep.moves_left() < len(calls) + 1:
            return [] if not calls else None
        tool = self.ep.role("cart")
        self.ep.send_chunk([(f"code_cart_{n}_{self.ep.shop.tick}", tool, args) for n, args in enumerate(calls)])
        replies = self._latest_observations(len(calls))
        missing, held_view = self._missing_after_put(calls, replies, wanted, line_spec)
        if len(replies) < len(calls):
            return None
        if held_view is None and not missing:
            held_view = self._view_after_put(wanted, missing)
        if held_view is not None:
            missing |= {x for x in wanted if x not in held_view}
        for ident in missing:
            basket.units.pop(ident, None)
        return sorted(missing)

    def _view_after_put(self, wanted, missing):
        viewer = self.ep.role("cart_view")
        if not viewer or self.ep.moves_left() < 3:
            return None
        self.ep.send_chunk([(f"code_view_{self.ep.shop.tick}", self.ep.role("cart_view"), {})])
        view = self._latest_observations(1)
        if not view:
            return None
        held_view = self.ep.basket.view_units(view[0])
        if held_view is None:
            missing.update(x for x in wanted if x not in self.ep.basket.units)
        return held_view

    @staticmethod
    def _mode_choice(options, clauses, arg, use_last):
        for option in options:
            said = r"\b" + re.escape(str(option).lower()) + r"\b"
            for clause in clauses:
                if re.search(said, clause) and re.search(r"\b" + re.escape(arg.lower()) + r"\b", clause):
                    return option
        return options[-1] if use_last else options[0]

    @staticmethod
    def _ceiling_field(fields, clauses):
        for field, spec in fields.items():
            if not isinstance(spec, dict) or spec.get("type") not in ("integer", "number") or spec.get("enum"):
                continue
            lowered = field.lower()
            priced = re.search(r"ceil|max|price|cap$|budget|upper|most", lowered) or any(
                lowered in clause and re.search(r"\bpric|\bcost", clause) for clause in clauses)
            if priced and not re.search(r"page|first|offset|skip|count|size|num", lowered):
                return field
        return None


    @staticmethod
    def _fill_required(draft, must, fields, text):
        for field in must:
            if field in draft:
                continue
            spec = fields[field] if isinstance(fields[field], dict) else {}
            if spec.get("enum"):
                draft[field] = spec["enum"][0]
            elif spec.get("type") in ("integer", "number"):
                draft[field] = spec.get("default", spec.get("minimum", 1))
            elif spec.get("type") == "string":
                draft[field] = text

    def _query_phrases(self, k, product, label, longest):
        product_words = [w for w in re.findall(r"[a-z0-9]+", product.lower()) if len(w) > 2 and w not in LINK_WORDS]
        kind = product_words[-1] if product_words else ""
        phrases = []
        for _, value in self.code_values(k):
            if re.search(r"\bor\b|/|,|\b(?:" + VALUE_LINKS + r")\b", value.lower()):
                continue
            value_words = [w for w in re.findall(r"[a-z0-9]+", value.lower()) if w not in LINK_WORDS]
            if value_words:
                phrases.append(" ".join(dict.fromkeys(value_words + [kind])))
                if len(value_words) > 1:
                    phrases.append(" ".join(dict.fromkeys([max(value_words, key=len), kind])))
            break
        if not phrases:
            words = product_words[:1] + product_words[-1:] if product_words else []
            phrases.append(" ".join(dict.fromkeys(words)) if words else " ".join(sorted(kind_words(label))[:2]))
        if is_number(longest):
            phrases = [self._trim_phrase(phrase, longest) for phrase in phrases]
        return phrases

    @staticmethod
    def _trim_phrase(phrase, longest):
        while " " in phrase and len(phrase) > longest:
            phrase = phrase.rsplit(" ", 1)[0]
        return phrase[: int(longest)]

    # what the records say, per listing

    def code_rows(self):
        table = {}

        def take(record):
            ident = self.ep.lookout.row_mark(record)
            if ident in self.ep.known.ids:
                self._fold_record(table.setdefault(ident, {"shelf": "", "title": "", "price": None, "stock": None,
                                                           "words": set(), "fields": {}}), record)
        for record in self.ep.known.items:
            walk_dicts(record, take)
        return table

    @staticmethod
    def _record_title(record):
        for key, value in record.items():
            name = str(key).lower()
            if isinstance(value, str) and re.search(r"title|name", name) and "id" not in name:
                return str(value)
        return ""

    @staticmethod
    def _record_price(record):
        for key, value in record.items():
            name = str(key).lower()
            if not is_number(value) or isinstance(value, bool) or not re.search(r"price|cost|amount", name):
                continue
            if not re.search(r"count|qty|quant|total|units|line|due|payable|sum", name):
                return float(value)
        return None

    def _fold_record(self, entry, record):
        eyes = self.ep.lookout
        entry["shelf"] = eyes.shelf_of(record) or entry["shelf"]
        entry["title"] = self._record_title(record) or entry["title"]
        price = self._record_price(record)
        if price is not None:
            entry["price"] = price
        for key, value in record.items():
            state = eyes.key_stock(key, value)
            if state is not None:
                entry["stock"] = state
        nested = [v for v in record.values() if isinstance(v, dict)]
        texts = [str(v) for v in record.values() if isinstance(v, (str, int, float)) and not isinstance(v, bool)]
        texts += [str(x) for inner in nested for x in inner.values() if isinstance(x, (str, int, float))]
        entry["words"] |= kind_words(" ".join(texts))
        for inner in nested:
            entry["fields"].update((str(key), str(x)) for key, x in inner.items()
                                   if isinstance(x, (str, int, float)) and not isinstance(x, bool))

    @staticmethod
    def code_shows(row, prop, value, hit_words, read):
        matching = [text for key, text in row["fields"].items() if same_prop(key, prop)]
        if matching:
            named = set(re.findall(r"[a-z]{2,}", str(prop).lower()))
            trimmed = re.sub(r"\b(?:" + "|".join(map(re.escape, sorted(named))) + r")s?\b", " ", str(value).lower()) if named else ""
            values = [value] + ([trimmed] if re.search(r"[a-z]{2,}|\d", trimmed) else [])
            return any(Closer.code_seen(option, text) for option in values for text in matching)
        if Closer.code_seen(value, row.get("title") or ""):
            return True
        worded = re.search(r"[a-z]{3,}", re.sub(r"\b(?:" + VALUE_LINKS + r")\b", " ", str(value).lower()))
        if worded and any(Closer.code_seen(value, text) for key, text in row["fields"].items() if "option" in str(key).lower()):
            return True
        words = set() if read else kind_words(value)
        return bool(words) and words <= hit_words

    @staticmethod
    def code_seen(value, text):
        haystack = str(text or "").lower()
        if not haystack.strip():
            return False
        stripped = re.sub(r"\b(?:" + VALUE_LINKS + r")\b", " ", str(value).lower())
        for option in re.split(r"\s*(?:\bor\b|/|,|;)\s*", stripped):
            words = [w for w in re.findall(r"[a-z]{2,}", option) if w not in LINK_WORDS]
            numbers = re.findall(r"\d+(?:\.\d+)?", option)
            if not words and not numbers:
                continue
            spelled = all(re.search(r"\b" + re.escape(w if len(w) <= 4 else w[:-1]), haystack) for w in words)
            if spelled and all(re.search(r"(?<![\d.])" + re.escape(n) + r"(?![\d.])", haystack) for n in numbers):
                return True
        return False

    @staticmethod
    def _evidence_rank(row, prop, value):
        wanted = set()
        for option in re.split(r"\s*(?:\bor\b|/)\s*", value.lower()):
            if ch_words(option):
                wanted.add(" ".join(ch_words(option)))
        best = (2, 3)
        for key, text in row["fields"].items():
            if same_prop(key, prop) and Closer.code_seen(value, text):
                if " ".join(ch_words(text)) in wanted:
                    closeness = 0
                else:
                    closeness = 1 if any(" ".join(ch_words(part)) in wanted for part in ch_parts(text)) else 3
                best = min(best, (int(prop_key(key) != prop_key(prop)), closeness))
        if best == (2, 3) and Closer.code_seen(value, row.get("title") or ""):
            best = (2, 2)
        return best

    def _audit_silent(self, live):
        counsel = self.ep.counsel
        seen = counsel.audit_ids
        if self.ep.audit_done:
            limit = -1 if counsel.audit_reads is None else counsel.audit_reads
            if seen is None or len(self.ep.known.read) <= limit:
                seen = set(self.ep.known.read)
        for entry in reversed(self.ep.log):
            if isinstance(entry, dict) and entry.get("kind") == "audit":
                if isinstance(entry.get("result"), dict):
                    return set(seen or set()) - set(counsel.audit_fits or []) - live
                return set()
        return set()


    def _choose_from(self, pools, ordering, counts, cap):
        picks = []
        chosen = []
        total = 0.0
        for k, pool in enumerate(pools):
            taken = None
            for entry in sorted(pool, key=ordering):
                ident = entry[-1]
                if any(ident == got for got, _ in picks):
                    continue
                if ident in self.ep.known.read and self.ep.counsel.field_view(ident)["fails"]:
                    continue
                taken = entry
                break
            if taken is None:
                return None
            units = counts[k] if k < len(counts) else 1
            picks.append((taken[-1], units))
            chosen.append(taken)
            total += (0.0 if taken[8] == float("inf") else taken[8]) * units
        if cap is not None and total > cap:
            return []
        self.code_fit = {entry[-1]: not entry[2] for entry in chosen}
        return picks

    def _judged_values(self, slot):
        value_words = set()
        judged = []
        for prop, bare in self.code_values(slot):
            value_words |= kind_words(bare)
            if not re.search(r"\b(?:" + VALUE_LINKS + r")\b|\d\s*(?:-|to)\s*\d|[+<>]|\b(?:about|around|approx\w*|roughly)\b", bare.lower()):
                judged.append((prop, bare))
        return value_words, judged

    def _set_apart(self, slot):
        out = []
        for ref in self.ep.shop.differ_done(slot):
            excluded = [str(v) for v in [ref.get("value"), *(ref.get("goes_with") or [])] if str(v or "").strip()]
            out.append((ref["prop"], excluded))
        return out

    def finish_now(self):
        if self._out_of_room():
            return
        state = {"avoid": set(), "keep": None, "move": None, "kind": None, "reply": None}
        for turn in range(CODE_FINISHES):
            if self._out_of_room():
                return
            if state["move"] is None:
                state["move"], state["keep"] = self._opening_move(turn, state["avoid"], state["keep"])
                if state["move"] is None:
                    return
            if not self._finish_turn(turn, state):
                break
        self._last_word(state)

    def _finish_turn(self, turn, state):
        shop = self.ep.shop
        move = state["move"]
        if self.ep.call_kind(move["name"], move["args"]) == "order":
            self.ep.send_go_ahead(room=2)
        reply = shop.perform([move])
        state["reply"] = reply
        self.ep.log.append({"kind": "forced_finish", "action": move, "reply": reply})
        kind = state["kind"] = self.ep.call_kind(move["name"], move["args"])
        if not reply or reply.get("finished") or shop.done or turn == CODE_FINISHES - 1 or self.ep.moves_left() <= 1:
            return False
        self.ep.counsel.mute = True
        lines = list(self.ep.basket.units)
        obs = reply.get("observation")
        self.ep.absorb(move["name"], move["args"], obs, reply, True)
        self._budget_news(self.ep.scout.shopper_lines(move["name"], obs, True))
        state["move"] = None
        if kind == "order" and lines:
            state["move"], state["avoid"], state["keep"] = self._after_refused_order(
                lines, self.ep.role("order"), state["avoid"], state["keep"])
        return True

    def _last_word(self, state):
        reply, kind = state["reply"], state["kind"]
        order_tool, decline_tool = self.ep.role("order"), self.ep.role("decline")
        if self.ep.shop.done or not reply or reply.get("finished") or not decline_tool:
            return
        if kind == "decline" or self.ep.moves_left() <= 0:
            return
        again = False
        try:
            again = bool(order_tool) and kind == "order" and self._order_again(state["avoid"])
        except Exception as err:
            self.ep.log.append({"kind": "again_error", "error": f"{type(err).__name__}: {err}"[:300]})
        move = self._move(order_tool, "order") if again else self._move(decline_tool, "decline")
        self.ep.log.append({"kind": "forced_finish", "action": move, "reply": self.ep.shop.perform([move])})

    def _row_fits(self, row, judged, set_apart, own_hits, was_read):
        for prop, value in judged:
            if not self.code_shows(row, prop, value, own_hits, was_read):
                return False
        for prop, bad in set_apart:
            stated = [text for key, text in row["fields"].items() if same_prop(key, prop) and str(text).strip()]
            if not any(not any(ch_writes(v, text) for v in bad) for text in stated):
                return False
        return True

    def _pick_pool(self, k, home, products, counts, ctx):
        slot = k if products else None
        kind = kind_words(products[k]["product"] if products else self.ep.wants.reading.get("product") or "")
        value_words, judged = self._judged_values(slot)
        shared = {"units": counts[k] if k < len(counts) else 1, "kind": kind, "value_words": value_words,
                  "judged": judged, "apart": self._set_apart(slot)}
        pool = []
        for ident, row in ctx["table"].items():
            if self._row_allowed(ident, row, home, kind, judged, slot, ctx):
                entry = self._pool_row(ident, row, shared, ctx)
                if entry is not None:
                    pool.append(entry)
        return pool

    def _pool_row(self, ident, row, shared, ctx):
        known = self.ep.known
        cap = ctx["cap"]
        cost = row["price"] if row["price"] is not None else known.price(ident)
        priced = is_number(cost)
        if cap is not None and priced and cost * shared["units"] > cap:
            return None
        was_read = ident in known.read
        own_hits = ctx["hits"].get(ident) or set()
        judged = shared["judged"]
        fits = self._row_fits(row, judged, shared["apart"], own_hits, was_read)
        ranks = [self._evidence_rank(row, prop, value) for prop, value in judged] if was_read else []
        if ident in ctx["audited"]:
            if not (ctx["trust"] and fits and (judged or shared["apart"]) and all(max(r) <= 1 for r in ranks)):
                return None
        words = row["words"]
        value_words = shared["value_words"]
        score = (3 * len(value_words & words) + 2 * len(shared["kind"] & words) + len(ctx["request"] & words)
                 + 4 * len(value_words & own_hits))
        price = cost if priced else float("inf")
        if ranks:
            quality = (sum(r[0] for r in ranks), sum(r[1] for r in ranks))
        else:
            quality = (9, 9)
        return (int(ident not in ctx["keep"]), int(ident not in ctx["live"]), not fits, cap is not None and not priced,
                price if ctx["cheapest"] else 0.0, int(not was_read), row["stock"] is not True, (quality, -score), price,
                ident)

    def _row_allowed(self, ident, row, home, kind, judged, slot, ctx):
        if ident in ctx["failed"] or ident in self.ep.refused_lines or ident in ctx["avoid"] or row["stock"] is False:
            return False
        if home:
            if row["shelf"] not in home:
                return False
        elif not kind & row["words"]:
            return False
        for prop, value in judged:
            if self.code_clash(row["fields"], prop, value):
                return False
        return not self.ep.pointers.ref_excluded(row["fields"], slot)

    @staticmethod
    def code_clash(fields, prop, value):
        matching = [text for key, text in fields.items() if same_prop(key, prop)]
        if not matching:
            return False
        return not any(value_seen(value, text) for text in matching)

    # cart arguments built from the cart tool's schema

    @staticmethod
    def _option_words(value):
        return set(re.findall(r"[a-z]+", re.sub(r"([a-z])([A-Z])", r"\1 \2", str(value)).lower()))

    def _cart_form(self):
        schema = self._schema(self.ep.role("cart"))
        fields = self._props(schema)
        earlier = json_object(self.ep.basket.put_form) if self.ep.basket.put_form else None
        put_values = {}
        drop_values = {}
        for field in schema.get("required") or []:
            spec = fields.get(field) if isinstance(fields.get(field), dict) else {}
            options = spec.get("enum") if isinstance(spec.get("enum"), list) else []
            if options:
                put_values[field], drop_values[field] = self._cart_choice(field, options, earlier)
        free = self._free_text_fields(fields)
        id_field = next((f for f in free if re.search(r"id|item|product|listing|sku|target|which|number", f.lower())),
                        free[0] if free else None)
        unit_fields = [f for f, spec in fields.items()
                       if isinstance(spec, dict) and spec.get("type") in ("integer", "number") and not spec.get("enum")]
        return schema, put_values, drop_values, id_field, unit_fields

    def _cart_choice(self, field, options, earlier):
        adders = []
        removers = []
        for option in options:
            words = self._option_words(option)
            if words & DROP_WORDS:
                removers.append(option)
            elif words & PUT_WORDS:
                adders.append(option)
        if isinstance(earlier, dict) and earlier.get(field) in options:
            chosen = earlier[field]
        else:
            chosen = (adders or options)[0]
        others = [o for o in options if o != chosen]
        dropping = next((o for o in removers if o != chosen), others[0] if len(options) == 2 else None)
        return chosen, dropping

    def _put_args(self, picks):
        schema, put_values, _, id_field, unit_fields = self._cart_form()
        line_spec = self.ep.basket.line_spec()
        if line_spec:
            array_field, line_id, line_units = line_spec
            drafts = [{**put_values, array_field: [{line_id: ident, line_units: units} for ident, units in picks]}]
        elif not id_field:
            return []
        else:
            drafts = [{**put_values, id_field: ident, **({unit_fields[0]: units} if unit_fields else {})} for ident, units in picks]
        checked = []
        for draft in drafts:
            fixed, faults = ArgCheck.execute(draft, schema)
            if faults:
                return []
            checked.append(fixed)
        return checked

    def _drop_args(self, lines):
        basket = self.ep.basket
        if not lines or basket.line_spec():
            return []
        schema, put_values, drop_values, id_field, unit_fields = self._cart_form()
        fields = self._props(schema)
        if not id_field:
            return None
        checked = []
        for ident in lines:
            if put_values:
                if any(drop_values.get(field) is None for field in put_values):
                    return None
                draft = {**drop_values, id_field: ident}
                if unit_fields:
                    draft[unit_fields[0]] = max(1, int(basket.held(ident) or 1))
            elif not unit_fields:
                return None
            elif is_number((fields.get(unit_fields[0]) or {}).get("minimum")) and fields[unit_fields[0]]["minimum"] > 0:
                return None
            else:
                draft = {id_field: ident, unit_fields[0]: 0}
            fixed, faults = ArgCheck.execute(draft, schema)
            if faults:
                return None
            checked.append(fixed)
        return checked

    def _latest_observations(self, count):
        for entry in reversed(self.ep.log):
            if isinstance(entry, dict) and entry.get("kind") == "env":
                reply = entry.get("reply")
                break
        else:
            reply = None
        if not isinstance(reply, dict):
            return []
        many = reply.get("observations")
        return list(many if isinstance(many, list) else [reply.get("observation")])[:count]

    @staticmethod
    def code_fault(obs):
        flagged = []

        def look(record):
            if any(re.search(FAULT_KEYS, str(key).lower()) and value not in (None, False, "", [], {})
                   for key, value in record.items()):
                flagged.append(record)
        walk_dicts(obs, look, 0, 2)
        return bool(flagged)

    # the code's own search rounds

    def _note_search_hits(self, calls, replies, text_fields):
        for call_args, obs in zip(calls, replies):
            typed = " ".join(str(call_args.get(field) or "") for field in text_fields)
            if not typed.strip():
                continue
            words = kind_words(typed)

            def mark(record):
                ident = self.ep.lookout.row_mark(record)
                if ident in self.ep.known.ids:
                    self.code_hits.setdefault(ident, set()).update(words)
            walk_dicts(obs, mark)

    def _first_search_calls(self, products, reserve):
        per_product = [self.code_queries(k if self.ep.wants.items() else None) for k in range(products)]
        calls = []
        for rank in range(max((len(q) for q in per_product), default=0)):
            for queries in per_product:
                if rank < len(queries) and queries[rank] not in calls:
                    calls.append(queries[rank])
        leading = [q[0] for q in per_product if q]
        rounds = 1 + CODE_PAGES
        if self.ep.wants.priority() == "cheapest":
            rounds += 2
        return calls[: max(0, self.ep.moves_left() - reserve)], leading, rounds

    def _code_search(self, products, more=False):
        tool = self.ep.role("search")
        text_fields = self._free_text_fields(self._schema(tool).get("properties") or {})
        reserve = 2 + 2 * products
        if more:
            leading = list(self.code_more or [])
            calls = leading[: max(0, self.ep.moves_left() - reserve - 2 * products)]
            rounds = 1
        else:
            calls, leading, rounds = self._first_search_calls(products, reserve)
        if not calls:
            return
        self.code_searched = True
        for round_no in range(rounds):
            self.ep.send_chunk([(f"code_search_{n}_{self.ep.shop.tick}", tool, call_args) for n, call_args in enumerate(calls)])
            replies = self._latest_observations(len(calls))
            if self.ep.word_kind != "any":
                self._note_search_hits(calls, replies, text_fields)
            onward = [follow for follow in (self._next_page_args(call_args, obs)
                                            for call_args, obs in zip(calls, replies) if call_args in leading) if follow]
            self.code_more = onward
            if round_no == rounds - 1 or self._search_enough():
                return
            leading = onward
            calls = onward[: max(0, self.ep.moves_left() - reserve - 3 * products)]
            if not calls:
                return

    def _search_enough(self):
        picks = self.code_picks()
        fitting = self.code_fit or {}
        return self.ep.wants.priority() != "cheapest" and bool(picks) and all(fitting.get(ident) for ident, _ in picks)

    @staticmethod
    def _first_listing(obs):
        listings = []

        def gather(record):
            for value in record.values():
                if isinstance(value, list) and value and all(isinstance(x, dict) for x in value):
                    listings.append(record)
                    return
        walk_dicts(obs, gather, 0, 3)
        return listings[0] if listings else None


    def _open_picks(self, picks, avoid, keep):
        for _ in range(CODE_READS):
            unread = [ident for ident, _ in picks if ident not in self.ep.known.read]
            batch = self.ep.shop.batch_tool()
            as_batch = bool(batch) and len(unread) > 1
            if not unread or not (batch or self.ep.role("details")):
                break
            if self.ep.moves_left() < (1 if as_batch else len(unread)) + 2 * len(picks) + 3 or self.ep.time_left() <= 8:
                break
            if as_batch:
                slab = [(f"code_read_{self.ep.shop.tick}", batch[0], {batch[1]: unread[: batch[2]]})]
            else:
                slab = []
                for n, ident in enumerate(unread):
                    call_args = self.ep.counsel.details_call(ident)
                    if call_args:
                        slab.append((f"code_read_{n}_{self.ep.shop.tick}", self.ep.role("details"), call_args))
            if not slab:
                break
            self.ep.send_chunk(slab)
            again = self.code_picks(avoid, keep)
            if again == picks:
                break
            picks = again
        return picks

    def _missing_after_put(self, calls, replies, wanted, line_spec):
        basket = self.ep.basket
        missing = set()
        held_view = None
        for call_args, obs in zip(calls, replies):
            seen = basket.view_units(obs)
            if seen is not None:
                held_view = seen
            if self.code_fault(obs):
                named = [x for x in wanted if x in as_text(obs)]
                touched = [line for line, _ in basket.lines(call_args) if line in wanted]
                missing.update(named or touched or (list(wanted) if line_spec else []))
        return missing, held_view


    def code_recheck(self, lines):
        tool = self.ep.shop.stock_tool() or self.ep.role("view") or self.ep.role("details")
        if not lines or not tool or self.ep.moves_left() < len(lines) + 2 or self.ep.time_left() < 8:
            return False, set()
        slab = []
        for n, ident in enumerate(lines):
            call_args = self.ep.shop.lookup_call(ident, tool)
            if call_args:
                slab.append((f"code_check_{n}_{self.ep.shop.tick}", tool, call_args))
        if not slab:
            return False, set()
        self.ep.send_chunk(slab)
        table = self.code_rows()
        return True, {x for x in lines if (table.get(x) or {}).get("stock") is False}

    def _budget_news(self, lines):
        waiting = [line for group, _ in self.ep.wants.pending or [] for line in group]
        for line in list(lines) + waiting:
            if line in self.news_read:
                continue
            self.news_read.add(line)
            amount = self._budget_from(line)[0]
            if is_number(amount) and amount > 0:
                self.ep.wants.spend_cap = amount
                self.ep.log.append({"kind": "code_news", "budget": amount})

    # filling the cart and finishing without the LLM

    def code_cart(self, avoid=None, keep=None):
        shop = self.ep.shop
        if not self.ep.role("cart") or shop.done or shop.broke or self.ep.moves_left() < 2 or self.ep.time_left() < 4:
            return False
        self.ep.counsel.mute = True
        try:
            return self._cart_rounds(set(avoid or ()), keep, self.ep.role("search"))
        except Exception as err:
            self.ep.log.append({"kind": "code_finish_error", "error": f"{type(err).__name__}: {err}"[:300]})
            return False

    def _refresh_picks(self, picks, avoid, keep, finder, products):
        unfit = not picks or not all(self.code_fit.get(ident) for ident, _ in picks)
        if finder and not self.code_searched and self.ep.moves_left() >= 4 and unfit:
            self._code_search(products)
            return self.code_picks(avoid, keep)
        if finder and self.code_more and self.ep.moves_left() >= 6 + 2 * products and (
                unfit or self.ep.wants.priority() == "cheapest"):
            self._code_search(products, True)
            return self.code_picks(avoid, keep)
        return picks

    def _cart_rounds(self, avoid, keep, finder):
        products = max(1, len(self.ep.wants.items()))
        for _ in range(4):
            picks = self._refresh_picks(self.code_picks(avoid, keep), avoid, keep, finder, products)
            picks = self._open_picks(picks, avoid, keep)
            if not picks:
                self.ep.log.append({"kind": "code_finish", "picks": [], "result": "nothing to order"})
                return False
            faulty = self.code_put(picks)
            if faulty is None:
                self.ep.log.append({"kind": "code_finish", "picks": picks, "result": "no cart call"})
                return False
            if not faulty and self.ep.moves_left() >= len(picks) + 3:
                faulty = sorted(self.code_recheck([ident for ident, _ in picks])[1])
            if not faulty:
                self.ep.log.append({"kind": "code_finish", "picks": picks, "cart": dict(self.ep.basket.units)})
                return bool(self.ep.basket.units)
            avoid |= set(faulty)
            self.ep.log.append({"kind": "code_swap", "lines": faulty})
        return False

    def _move(self, tool, kind):
        text = (self.ep.closest_note or "Order the cart.") if kind == "order" else SORRY
        return {"name": tool, "args": self.schema_args(tool, text, kind)}

    def _out_of_room(self):
        shop = self.ep.shop
        return shop.done or shop.broke or self.ep.moves_left() <= 0 or self.ep.time_left() < 2

    def _sold_before_order(self, avoid):
        basket = self.ep.basket
        try:
            self.ep.counsel.mute = True
            held_before = list(basket.units)
            sold = self.ep.shop.cart_gone(held_before)
            if not sold:
                return sold, None
            avoid |= sold
            self.ep.log.append({"kind": "code_dead", "lines": sorted(sold)})
            return sold, set(held_before) - sold
        except Exception as err:
            self.ep.log.append({"kind": "cart_gone_error", "error": f"{type(err).__name__}: {err}"[:300]})
            return set(), None

    def _opening_move(self, turn, avoid, keep):
        basket = self.ep.basket
        order_tool, decline_tool = self.ep.role("order"), self.ep.role("decline")
        sold = set()
        if order_tool and turn == 0 and basket.units:
            sold, kept = self._sold_before_order(avoid)
            if kept is not None:
                keep = kept
        cart_ready = (order_tool and turn == 0 and not sold and basket.used
                      and (basket.units or basket.view_units(basket.last_view) is None))
        if cart_ready and self.ep.basket_short():
            cart_ready = False
            keep = set(basket.units) if keep is None else keep
            self.ep.log.append({"kind": "code_partial", "faults": self.ep.basket_short()})
        if cart_ready or (order_tool and self.code_cart(avoid, keep)):
            return self._move(order_tool, "order"), keep
        if order_tool and turn > 0 and self._order_again(avoid):
            return self._move(order_tool, "order"), keep
        if decline_tool:
            return self._move(decline_tool, "decline"), keep
        return None, keep


    def _after_refused_order(self, lines, order_tool, avoid, keep):
        reread, sold = self.code_recheck(lines)
        if not reread:
            return None, avoid, keep
        self.ep.refused_lines.difference_update(set(lines) - sold)
        table = self.code_rows()
        total = sum(((table.get(line) or {}).get("price") or 0.0) * self.ep.basket.held(line) for line in lines)
        budget = self.ep.wants.spend_cap
        if is_number(budget) and total > budget:
            return None, set(sold), None
        if sold:
            return None, set(sold), set(lines) - sold
        return self._move(order_tool, "order"), avoid, keep

    def _order_again(self, avoid):
        held = set(self.ep.basket.units)
        ruled_out = set(avoid) | self.ep.shop.dead_lines()
        return bool(held) and not held & ruled_out and bool(self.code_picks(ruled_out))

    def catch_up(self):
        counsel, wants = self.ep.counsel, self.ep.wants
        try:
            if counsel.mute or self.ep.time_left() < CLOSING:
                return
            notes = self._reread_note() + self._waiting_lines()
            if notes:
                self._append_to_chat("\n".join(notes) + "\n" + self.ep.status_line())
        except Exception as err:
            self.ep.log.append({"kind": "catch_up_error", "error": f"{type(err).__name__}: {err}"[:300]})

    def _reread_note(self):
        counsel, wants, brief = self.ep.counsel, self.ep.wants, self.ep.brief
        read_at = brief.read_at
        if not wants.reading.get("code_reading") or self.reread or read_at is None:
            return []
        if counsel.last_reply <= read_at or self.ep.time_left() <= CLOSING + 60:
            return []
        self.reread = True
        answer = counsel.converse([{"role": "user", "content": brief.read_content}], READER, 3000)
        fresh = json_object((answer or {}).get("content") or "") or {}
        usable = isinstance(fresh.get("roles"), dict)
        self.ep.log.append({"kind": "re_read", "ok": usable})
        if not usable:
            return []
        brief.use_reading(fresh, True)
        return ["[plan updated] LLMs answered again and the request was read anew; this reading replaces the last:\n"
                + wants.overview(self.ep.step_cap, self.ep.msg_cap)]

    def _waiting_lines(self):
        wants = self.ep.wants
        queued = wants.pending or []
        if not queued or self.ep.counsel.last_reply <= wants.pending_at or self.ep.time_left() < CLOSING:
            return []
        wants.pending = []
        notes = []
        for group, question in queued[-3:]:
            heard = wants.hear(group, question)
            if heard:
                notes.append("[the shopper's earlier lines, read now] " + heard)
        return notes

    def _append_to_chat(self, text):
        chat = self.ep.chat
        newest = chat[-1] if chat else None
        if newest and newest.get("role") in ("user", "tool") and isinstance(newest.get("content"), str):
            newest["content"] += "\n" + text
        else:
            chat.append({"role": "user", "content": text})


class Bargains:
    def __init__(self, ep):
        self.ep = ep
        self.walked = set()
        self.fit_held = None
        self.decline_queue = None
        self.decline_read_done = None

    def _version_passes(self):
        version = str(self.ep.wants.version)
        return {stamp.rpartition("|")[0] for stamp in self.ep.passed if stamp.rpartition("|")[2] == version}

    def _version_fails(self):
        version = str(self.ep.wants.version)
        return {stamp.split("|")[0] for stamp, verdict in self.ep.rulings.items()
                if verdict and stamp.split("|")[1:2] == [version]}

    # queries already run, and how to form a fresh one

    def ran_query(self, text):
        width = self.ep.lead_words or 3
        stem = self.ep.lookout.stem
        wanted = sorted(map(stem, re.findall(r"[a-z0-9]+", str(text).lower())[:width]))
        for walk in self.ep.page_book.values():
            typed = " ".join(self.ep.lookout.free_text(walk["args"])).lower()
            if sorted(map(stem, re.findall(r"[a-z0-9]+", typed)[:width])) == wanted:
                return True
        return False

    def _query_template(self):
        for walk in reversed(list(self.ep.page_book.values())):
            if self.ep.scout.cg_listing(walk["args"]):
                continue
            texts = self.ep.lookout.free_text(walk["args"])
            slots = [key for key, val in walk["args"].items() if isinstance(val, str) and val in texts and val.strip()]
            if len(texts) == 1 and len(slots) == 1:
                rest = dict(walk["args"])
                del rest[slots[0]]
                return rest, slots[0]
        return None

    @staticmethod
    def _words_near(a, b):
        if any(ch.isdigit() for ch in a + b):
            return a.lstrip("0") == b.lstrip("0")
        return min(len(a), len(b)) >= 3 and (a.startswith(b) or b.startswith(a))

    @staticmethod
    def _pieces(token):
        return [bit for bit in re.findall(r"[a-z0-9]+", token) if len(bit) >= 2]

    def word_trap(self, raw):
        try:
            return self._spot_trap(raw)
        except Exception as err:
            self.ep.log.append({"kind": "trap_error", "error": f"{type(err).__name__}: {err}"[:300]})
            return "", ""

    def _trap_width(self, tokens):
        if self.ep.lead_words:
            return self.ep.lead_words
        if self.ep.word_kind == "all":
            return max(3, len(tokens))
        return 3

    def _token_hits(self, token, pool):
        bits = self._pieces(token)
        return any(self._words_near(bit, other) for bit in bits for other in pool)

    def _live_needs(self):
        found = []
        for need in self.ep.wants.props_left():
            head, colon, value = need.partition(":")
            if not colon or ref_text(value):
                continue
            plain_value = re.sub(r"\((?:stated|revealed[^)]*)\)", "", value).lower()
            found.append((re.split(r"\s*\(", head)[0].lower(), head, plain_value))
        return found

    def _trap_vocabulary(self, live_needs):
        wants = self.ep.wants
        products = [row["product"] for row in wants.items()] or [str(wants.reading.get("product") or "")]
        value_words = set()
        prop_names = set()
        for name, _, value in live_needs:
            value_words.update(re.findall(r"[a-z0-9]{2,}", value))
        labels = sorted(self.ep.lookout.shelf_labels())
        kinds = set(re.findall(r"[a-z0-9]{3,}", " ".join(products + labels).lower())) - LINK_WORDS
        for name, _, value in live_needs:
            prop_names.update(re.findall(r"[a-z]{3,}", name))
        dropped = set()
        for old in wants.let_go:
            dropped.update(re.findall(r"[a-z]{3,}", old.partition(":")[2].lower()))
        return products, value_words | kinds, kinds, prop_names - LINK_WORDS, dropped - LINK_WORDS

    def _spot_trap(self, raw):
        if self.ep.word_kind == "any":
            return "", ""
        tokens = []
        for piece in " ".join(raw).split():
            piece = piece.strip("\"'()[]{},.;!?|").lower()
            if re.search(r"[a-z0-9]", piece):
                tokens.append(piece)
        width = self._trap_width(tokens)
        tokens = tokens[:width]
        live_needs = self._live_needs()
        products, safe, kinds, prop_names, dropped = self._trap_vocabulary(live_needs)
        flagged = []
        reasons = []
        for token in tokens:
            if not self._pieces(token) or self._token_hits(token, safe):
                continue
            why = "name" if self._token_hits(token, prop_names) else "dropped" if self._token_hits(token, dropped) else ""
            if why:
                flagged.append(token)
                reasons.append(why)
        if not flagged:
            return "", ""
        kept = [t for t in tokens if t not in flagged]
        carried = next((head for _, head, value in live_needs
                        if any(self._token_hits(t, re.findall(r"[a-z0-9]{2,}", value)) for t in kept)), None)
        if carried is None:
            return "", ""
        if not any(self._token_hits(t, kinds) for t in kept):
            slot = self.ep.talk.product_of(carried) if self.ep.wants.items() else None
            name = products[slot] if slot is not None and slot < len(products) else products[0]
            kept.extend([w for w in re.findall(r"[a-z]{3,}", name.lower()) if w not in LINK_WORDS][-1:])
        query = " ".join(kept[:width])
        if query and not self.ran_query(query):
            return self._trap_note(flagged[0], reasons[0]) + f" Also search '{query}' and read every page of it.", query
        return "", ""

    @staticmethod
    def _trap_note(word, reason):
        if reason == "name":
            return (f"The word '{word}' in this query comes from the NAME of a required property, not a required value. "
                    "Since the store needs all words, it returns only listings whose fields or title repeat that name, "
                    "missing those with just the value in a field.")
        return (f"Query word '{word}' stems only from a requirement the shopper DROPPED, so such a search keeps just "
                "the listings that carry it, though none has to anymore.")

    def _value_groups(self, k, several):
        groups = []
        for need in self.ep.wants.props_left():
            head, _, value = need.partition(":")
            if several and self.ep.talk.product_of(head) not in (k, None):
                continue
            value = re.sub(r"\((?:stated|revealed[^)]*)\)", "", value).strip().lower()
            if value and not re.search(r"\d|as shown|same as|page|redirect|refer|\bnot\b|\bno\b|without|except|anything|other than", value):
                words = [w for w in re.findall(r"[a-z]{2,}", re.split(r"[,;/]|\bor\b|\band\b", value)[0]) if w not in LINK_WORDS]
                if 0 < len(words) < 3:
                    groups.append(words)
        return groups

    def _needs_combined(self, k, several, homes, covered):
        if several:
            return k < len(homes) and bool(homes[k]) and not homes[k] & covered
        return bool(self.ep.brief.unverified())

    def combined_queries(self):
        template = self._query_template()
        if not template:
            return []
        several = self.ep.wants.items()
        width = self.ep.lead_words or 3
        covered = {self.ep.counsel.shelf_known(x) for x in self._version_passes() - self.ep.refused_lines} - {""}
        homes = self.ep.shop.home_shelves() if several else []
        built = []
        for k in (range(len(several)) if several else [None]):
            if not self._needs_combined(k, several, homes, covered):
                continue
            groups = self._value_groups(k, several)
            if len(groups) < 2:
                continue
            query = []
            for words in groups:
                if len(query) + len(words) <= width:
                    query += words
            product = several[k]["product"] if several else str(self.ep.wants.reading.get("product") or "")
            tail = [w for w in re.findall(r"[a-z]{3,}", product.lower()) if w not in LINK_WORDS][-1:]
            if tail and len(query) < width and tail[0] not in query:
                query += tail
            text = " ".join(query)
            if len(query) > 1 and not self.ran_query(text):
                built.append({**template[0], template[1]: text})
        return built[:2]

    # second opinion when the decline audit and the item check disagree

    def _audit_reason(self):
        for entry in reversed(self.ep.log):
            if isinstance(entry, dict) and entry.get("kind") == "audit" and isinstance(entry.get("result"), dict):
                return str((entry.get("result") or {}).get("reason") or "")
        return ""

    def _settle_conflict(self, thing, ruling):
        try:
            reason = self._audit_reason()
            said = f" ({reason[:400]})" if reason else ""
            extra = ("\n\nTWO READINGS OF THIS ITEM DISAGREE. Situation: the same records were read two ways and only one "
                     "reading holds. Steps: a) The decline audit found that it meets every requirement" + said
                     + "; an earlier check reported instead: " + ruling[:500]
                     + "\nb) Decide it from the records: a field giving the required value in different words, or naming a shade, "
                     "qualifier or spelling of that very value, meets it; a field giving some other value fails it.")
            verdict = self.ep.counsel.check_item(thing, "ORDER", second=extra)
            if not verdict:
                key = f"{thing}|{self.ep.wants.version}"
                for memo in [m for m in self.ep.rulings if m == key or m.startswith(key + "|")]:
                    self.ep.rulings[memo] = ""
            return verdict or ""
        except Exception as err:
            self.ep.log.append({"kind": "arbitrate_error", "error": f"{type(err).__name__}: {err}"[:300]})
            return ruling

    def unrefuted(self):
        try:
            return self._unrefuted_picks()
        except Exception as err:
            self.ep.log.append({"kind": "unrefuted_error", "error": f"{type(err).__name__}: {err}"[:300]})
            return []

    def _stock_marks(self):
        sold = {}

        def note(record):
            ident = self.ep.lookout.row_mark(record)
            if ident in self.ep.known.ids and any(re.search(r"stock|avail", str(key).lower()) for key in record):
                sold[ident] = self.ep.lookout.sold_out(record)
        for record in self.ep.known.items:
            walk_dicts(record, note)
        return sold

    def _audited_reads(self):
        counsel = self.ep.counsel
        reviewed = counsel.audit_ids
        if self.ep.audit_done:
            limit = -1 if counsel.audit_reads is None else counsel.audit_reads
            if reviewed is None or len(self.ep.known.read) <= limit:
                reviewed = set(self.ep.known.read)
        return set(reviewed or set()) - set(counsel.audit_fits or [])

    def _unrefuted_picks(self):
        products = self.ep.wants.items()
        homes = self.ep.shop.home_shelves() if products else [set(self.ep.lookout.shelf_labels())]
        if not homes or not all(homes):
            return []
        shut = self._version_fails()
        live = self._version_passes()
        shut |= self._audited_reads()
        shut |= set(self.ep.refused_lines)
        sold = self._stock_marks()
        budget = self.ep.wants.spend_cap
        pointers = self.ep.pointers.ref_pointers()
        chosen = []
        total = 0.0
        for k, home in enumerate(homes):
            units = products[k]["quantity"] if products else self.ep.wants.quantity()
            best = self._cheapest_unrefuted(home, units, shut, sold, pointers, live, chosen, budget)
            if best is None:
                return []
            chosen.append(best[2])
            total += best[1] * units
        return chosen if not (is_number(budget) and total > budget) else []

    def _cheapest_unrefuted(self, home, units, shut, sold, pointers, live, chosen, budget):
        pool = []
        for ident in sorted(self.ep.known.read):
            if ident in shut or sold.get(ident) or ident in chosen or ident in pointers:
                continue
            if self.ep.counsel.shelf_known(ident) not in home:
                continue
            price = self.ep.known.price(ident)
            if not is_number(price) or (is_number(budget) and price * units > budget):
                continue
            pool.append((ident not in live, price, ident))
        return min(pool) if pool else None

    def fit_hold(self):
        try:
            return self._hold_for_audit_fit()
        except Exception as err:
            self.ep.log.append({"kind": "fit_hold_error", "error": f"{type(err).__name__}: {err}"[:300]})
            return ""

    def _confirmed_fits(self, candidates):
        good = []
        for thing in candidates[:2]:
            if thing not in self.ep.known.read:
                continue
            verdict = self.ep.counsel.check_item(thing, "ORDER")
            if verdict and self.ep.time_left() > CLOSING + 30:
                verdict = self._settle_conflict(thing, verdict)
            if verdict or f"{thing}|{self.ep.wants.version}" not in self.ep.passed:
                continue
            good.append(thing)
            if not self.ep.wants.items():
                break
        return good

    def _hold_for_audit_fit(self):
        if self.fit_held or self.ep.basket.units or self.ep.moves_left() <= 4 or self.ep.time_left() < CLOSING + 30:
            return ""
        candidates = [x for x in self.ep.counsel.audit_fits or [] if x in self.ep.known.ids and x not in self.ep.refused_lines]
        if not candidates:
            return ""
        self.fit_held = True
        good = self._confirmed_fits(candidates)
        if not good:
            return ""
        several = self.ep.wants.items()
        if several:
            covered = {self.ep.counsel.shelf_known(x) for x in self._version_passes() - self.ep.refused_lines} - {""}
            if any(home and not home & covered for home in self.ep.shop.home_shelves()):
                return ""
        one = len(good) == 1
        target = good[0] if not several else "a verified line of every product"
        return ("The decline audit gave " + ", ".join(good) + " as a possible fit, and the separate item check went through "
                + ("its" if one else "their") + " full record and found " + ("it meets" if one else "they meet")
                + " each requirement now in force. Look at " + ("it" if one else "them") + " before a decline: when it fits, put "
                + target + " into the cart, read stock and price again, and order; decline only if it fails some "
                "requirement in force.")

    # walking result pages of an id-ordered store for the cheapest fit

    def _id_ordered_store(self):
        flags = [walk.get("id_order") for walk in self.ep.page_book.values() if "id_order" in walk]
        return False not in flags and flags.count(True) > 1

    def product_index(self, thing):
        if not self.ep.wants.items():
            return None
        label = self.ep.counsel.shelf_known(thing)
        if label:
            for k, home in enumerate(self.ep.shop.home_shelves()):
                if label in home:
                    return k
        return None

    def walk_home(self, k):
        homes = self.ep.shop.home_shelves()
        home = set(homes[0 if k is None else k]) if homes and (k or 0) < len(homes) else set()
        if not home and k is None:
            return set(self.ep.lookout.shelf_labels())
        return home

    @staticmethod
    def fig_amount(text):
        lowered = str(text).lower()
        hit = None if re.search(CH_BOUND, lowered) else re.match(r"\s*(\d+)\s+([a-z]+)", lowered)
        return None if hit is None else (hit.group(1), hit.group(2))

    @staticmethod
    def amount_part(text):
        lowered = str(text).lower()
        return bool(re.search(r"\d", lowered)) and bool(
            re.search(CH_BOUND, lowered) or re.match(r"\s*\d+(?:[.,]\d+)?\s+[a-z]", lowered))

    def _front_words(self, call_args, pattern):
        joined = []
        for text in self.ep.lookout.free_text(call_args):
            joined.extend(re.findall(pattern, text.lower()))
        return joined[: self.ep.lead_words] if self.ep.lead_words else joined

    def _part_carried(self, part, words, tokens):
        near = self.ep.lookout.close_words
        amount = self.fig_amount(part)
        if amount:
            return amount[0] in tokens and any(t == amount[1] or near(t, amount[1]) for t in tokens)
        if self.amount_part(part):
            return False
        return any(near(w, v) for w in words for v in re.findall(r"[a-z0-9]{3,}", part))

    def walk_carries(self, args, k, alts=False):
        split_by_product = k is not None and bool(self.ep.wants.items())
        words = self._front_words(args, r"[a-z0-9]{2,}")
        tokens = self._front_words(args, r"[a-z0-9]+")
        carried = set()
        for n, need in enumerate(self.ep.wants.props_left()):
            head, _, tail = need.partition(":")
            if split_by_product and self.ep.talk.product_of(head) not in (k, None):
                continue
            value = re.sub(r"\((?:stated|revealed[^)]*)\)", "", tail).lower()
            for a, part in enumerate(re.split(r"[,;/]|\bor\b|\beither\b|\bnor\b", value)):
                if self._part_carried(part, words, tokens):
                    carried.add((n, a) if alts else n)
        return carried

    def walk_score(self, args, k):
        return len(self.walk_carries(args, k))

    def _lead_stems(self, walk, depth):
        typed = " ".join(self.ep.lookout.free_text(walk["args"])).lower()
        return {self.ep.lookout.stem(w) for w in re.findall(r"[a-z0-9]+", typed)[:depth]}

    def _allowed_stems(self, k, home):
        stem = self.ep.lookout.stem
        allowed = {stem(w) for label in home for w in re.findall(r"[a-z0-9]{2,}", label)}
        split_by_product = k is not None and bool(self.ep.wants.items())
        for need in self.ep.wants.props_left():
            if split_by_product and self.ep.talk.product_of(need.partition(":")[0]) not in (k, None):
                continue
            allowed |= {stem(w) for w in re.findall(r"[a-z0-9]{2,}", need.partition(":")[2].lower())}
        return allowed

    def _walk_options(self, k):
        home = self.walk_home(k)
        if not home:
            return []
        depth = self.ep.lead_words or None
        allowed = self._allowed_stems(k, home)
        open_walks, finished = [], []
        for key, walk in self.ep.page_book.items():
            if walk.get("id_order") is not True or not walk.get("first"):
                continue
            if self.ep.lookout.next_page(walk) is not None:
                open_walks.append((key, walk))
            elif self._lead_stems(walk, depth) and len(walk.get("row_shelf") or {}) >= walk["total"]:
                finished.append(self._walk_mark(walk, k, depth))
        best = {}
        for key, walk in open_walks:
            if not self.walk_carries(walk["args"], k):
                continue
            if all(label not in home for label in (walk.get("row_shelf") or {}).values()):
                continue
            words, pairs, mode = self._walk_mark(walk, k, depth)
            if any(seen[2] == mode and (seen[0] <= words or (seen[0] <= allowed and pairs <= seen[1])) for seen in finished):
                continue
            pages = -(-walk["total"] // max(1, walk["size"])) - len(walk["read"])
            if not 0 < pages <= 2 * WALK_PAGES:
                continue
            twin = (walk["total"], walk["first"])
            held = best.get(twin)
            if held is None or pages < held[0]:
                best[twin] = (pages, key, walk)
        options = list(best.values())
        if self.ep.scout.all_words():
            options = [o for o in options if not any(self._shadows(p, o, depth, allowed) for p in options if p is not o)]
        return sorted(options, key=lambda p: (p[0], -self.walk_score(p[2]["args"], k), p[1]))

    def _walk_mark(self, walk, k, depth):
        return (self._lead_stems(walk, depth), self.walk_carries(walk["args"], k, alts=True),
                self.ep.scout.vl_mode(walk["args"]))

    def _shadows(self, p, o, depth, allowed):
        mine, other = self._lead_stems(p[2], depth), self._lead_stems(o[2], depth)
        if mine < other and p[0] <= WALK_PAGES and not (other - mine) <= allowed:
            return True
        return other < mine and (mine - other) <= allowed

    def walk_plan(self, budget, only=None):
        if budget <= 0 or self.ep.wants.priority() != "cheapest" or not self.ep.role("search"):
            return []
        if not self.ep.lookout.page_arg() or not self._id_ordered_store():
            return []
        slots = list(range(len(self.ep.wants.items()))) or [None]
        wanted = []
        for k in slots:
            if k in self.walked or (only is not None and k not in only):
                continue
            options = self._walk_options(k)
            if options:
                wanted.append((options[0][0], -1 if k is None else k, k, options[0][1]))
        plan = []
        room = budget
        for pages, _, k, key in sorted(wanted, key=lambda w: (w[0], w[1])):
            take = min(pages, WALK_PAGES, room)
            if take <= 0:
                break
            plan.append((k, key, take))
            room -= take
        return plan

    def walk_pick(self, k):
        sold = self.ep.shop.dead_lines()

        def ours(ident):
            return k is None or self.product_index(ident) == k
        options = [x for x in self.ep.basket.units if ours(x) and x not in sold]
        if not options:
            options = [x for x in (stamp.rpartition("|")[0] for stamp in self.ep.passed
                                   if stamp.rpartition("|")[2] == str(self.ep.wants.version))
                       if x not in self.ep.refused_lines and x not in sold and ours(x)]
        priced = [(x, float(self.ep.known.price(x))) for x in options if is_number(self.ep.known.price(x))]
        if priced:
            return min(priced, key=lambda pair: (pair[1], pair[0]))
        return None

    def _record_of(self, ident):
        found = [{}]

        def look(candidate):
            if self.ep.lookout.row_mark(candidate) == ident and self.ep.shop.title_of(candidate):
                found[0] = candidate
        for item in self.ep.known.items:
            walk_dicts(item, look)
        return found[0]

    @staticmethod
    def _maker_of(record):
        for key, value in record.items():
            name = str(key).lower()
            if isinstance(value, str) and value.strip() and re.search(r"brand|make|manufactur", name) and "id" not in name:
                return value
        return ""

    def _product_line_query(self, k):
        pick = self.walk_pick(k)
        template = self._query_template()
        if not (pick and template):
            return None
        record = self._record_of(pick[0])
        maker_words = re.findall(r"[a-z0-9]+", self._maker_of(record).lower())
        if not maker_words:
            return self.ep.scout.title_line(k, record, template)
        wants = self.ep.wants
        products = wants.items() or [{"product": str(wants.reading.get("product") or "")}]
        skip = set(maker_words) | LINK_WORDS
        skip |= {w for label in self.walk_home(k) for w in re.findall(r"[a-z]+", label)}
        skip |= {w for need in wants.props_left() for w in re.findall(r"[a-z]+", need.lower())}
        skip |= {w for row in products for w in re.findall(r"[a-z]+", row["product"].lower())}
        title_words = re.findall(r"[a-z][a-z'-]{3,}", self.ep.shop.title_of(record).lower())
        if maker_words[0] in title_words:
            del title_words[: title_words.index(maker_words[0]) + 1]
        line_words = [w for w in title_words if "'" not in w and w.strip("'-") not in skip]
        if not line_words:
            return None
        text = maker_words[0] + " " + line_words[0]
        if self.ran_query(text):
            return None
        return {**template[0], template[1]: text}

    def _shown_on_home(self, k, keys, home):
        ordered = [(key, walk) for key, walk in self.ep.page_book.items() if walk.get("id_order") is True]
        top = max([len(self.walk_carries(walk["args"], k)) for _, walk in ordered] or [0])
        shown = set()
        for key, walk in self.ep.page_book.items():
            carried = self.walk_carries(walk["args"], k) if walk.get("id_order") is True else set()
            if key in keys or (carried and len(carried) == top):
                shown.update(x for x, label in (walk.get("row_shelf") or {}).items() if label in home)
        return shown

    def _cheaper_rows(self, k, keys, most):
        known = self.ep.known
        shown = self._shown_on_home(k, keys, self.walk_home(k))
        pick = self.walk_pick(k) or self._cheapest_read(shown)
        units = self._units_of(k)
        budget = self.ep.wants.spend_cap
        sold = self._sold_marks(shown)
        ceiling = pick[1] if pick else None
        found = []
        for ident in shown:
            skip = ident in known.read or ident in self.ep.refused_lines or sold.get(ident)
            price = known.price(ident)
            if skip or not is_number(price):
                continue
            over = is_number(budget) and price * units > budget
            if not over and (ceiling is None or price < ceiling):
                found.append((price, ident))
        return [pair[1] for pair in sorted(found)[:most]]

    def _cheapest_read(self, shown):
        known = self.ep.known
        priced = [(float(known.price(x)), x) for x in shown if x in known.read and is_number(known.price(x))]
        if not priced:
            return None
        price, ident = min(priced)
        return ident, price

    def _units_of(self, k):
        wants = self.ep.wants
        products = wants.items()
        if k is not None and k < len(products):
            return products[k]["quantity"]
        return wants.quantity()

    def _sold_marks(self, shown):
        eyes = self.ep.lookout
        marks = {}

        def note(record):
            for key, v in record.items():
                if isinstance(v, str) and v in shown and "id" in str(key).lower():
                    marks[eyes.row_mark(record)] = eyes.sold_out(record)
                    return
        for item in self.ep.known.items:
            walk_dicts(item, note)
        return marks

    def _page_digest(self, obs):
        eyes = self.ep.lookout
        rows = []

        def gather(record):
            if len(record) > 3 and any(isinstance(v, str) and v in self.ep.known.ids and "id" in str(key).lower()
                                       for key, v in record.items()):
                rows.append(record)
        walk_dicts(obs, gather, ceiling=3)
        shown = []
        for row in rows[:12]:
            price = next((v for key, v in row.items() if is_number(v) and re.search(r"price|cost", str(key).lower())), None)
            sold = " SOLD OUT" if eyes.sold_out(row) else ""
            shown.append(f"{eyes.row_mark(row)} {'?' if price is None else price}{sold} "
                         f"'{self.ep.shop.title_of(row)[:70]}' [{eyes.shelf_of(row)}]")
        return "Page rows (id, price, title, category): " + ("; ".join(shown) or "none")

    def walk_call(self, args, say):
        chat = self.ep.chat
        proposal = {"id": f"walk_{len(chat)}_{self.ep.shop.tick}", "type": "function",
                    "function": {"name": self.ep.role("search"), "arguments": json.dumps(args)}}
        chat.append({"role": "assistant", "content": say, "tool_calls": [proposal]})
        before = len(self.ep.known.items)
        self.ep.covering = True
        try:
            replies = self.ep.dispatch([proposal], scripted=True)
        finally:
            self.ep.covering = False
        if not replies:
            chat.append({"role": "tool", "tool_call_id": proposal["id"], "content": "Not run."})
            return False
        cid, text = replies[0]
        grew = len(self.ep.known.items) > before
        body = text
        if grew:
            body = self._page_digest(self.ep.known.items[-1])
            _, marker, note = text.partition("\n[note] ")
            if marker and note and re.search(r"shopper|Meaning:|REFUSED", note):
                body += "\n[note] " + note
        chat.append({"role": "tool", "tool_call_id": cid, "content": body})
        return grew

    def _room_to_walk(self):
        return self.ep.time_left() >= CLOSING + 30 if not self.ep.shop.done else False

    @staticmethod
    def _interleave(pools, most):
        queue = []
        while len(queue) < most and any(pools):
            for pool in pools:
                while pool and pool[0] in queue:
                    del pool[0]
                if pool and len(queue) < most:
                    queue.append(pool.pop(0))
        return queue

    def _fit_queue(self, queue, batch, floor):
        while queue and (1 if batch and len(queue) > 1 else len(queue)) >= self.ep.moves_left() - floor:
            del queue[-1]

    def _walk_pages(self, plan, field, floor, report):
        shop = self.ep.shop
        progressed = False
        for k, walk_key, count in plan:
            self.walked.add(k)
            walk = self.ep.page_book.get(walk_key)
            if not walk:
                continue
            query = " ".join(self.ep.lookout.free_text(walk["args"]))
            read = 0
            while read < count:
                following = self.ep.lookout.next_page(walk)
                stop = (following is None or self.ep.moves_left() <= floor + 1 or shop.done or shop.broke
                        or self.ep.time_left() < CLOSING + 30)
                said = ("The lowest price is asked for, and this store orders search results by a fixed id rather than price, so I "
                        f"read the next page of '{query}'.")
                if stop or not self.walk_call({**walk["args"], field: following}, said):
                    break
                read += 1
                progressed = True
            seen = len(walk.get("row_shelf") or {})
            report.append(f"'{query}': {read} more page(s) read, {seen} of its {walk['total']} matches seen")
        return progressed

    def _line_more_pages(self, line, field, floor):
        shop = self.ep.shop
        extra = 0
        while line and extra < 2 and line["total"] <= 3 * max(1, line["size"]):
            following = self.ep.lookout.next_page(line)
            if following is None:
                return
            if self.ep.moves_left() <= floor + 3 or shop.done or self.ep.time_left() <= CLOSING + 30:
                return
            if not self.walk_call({**line["args"], field: self.ep.lookout.next_page(line)}, "Next page of the line search."):
                return
            extra += 1

    def _walk_lines(self, plan, field, floor, report, lines):
        progressed = False
        for k, _, _ in plan:
            if self.ep.moves_left() <= floor + 2 or not self._room_to_walk():
                break
            call_args = self._product_line_query(k)
            said = ("I also search the product line of the pick (its maker and the word that names the line): a line's other "
                    "sizes and colours are listings of their own and can be cheaper.")
            if not call_args or not self.walk_call(call_args, said):
                continue
            progressed = True
            lines[k] = json.dumps({a: v for a, v in call_args.items() if a != field}, sort_keys=True, default=str)
            self._line_more_pages(self.ep.page_book.get(lines[k]), field, floor)
            report.append(f"the pick's line '{' '.join(self.ep.lookout.free_text(call_args))}'")
        return progressed

    def _ceiling_rounds(self, ceil, floor, repair, report):
        rows = {}
        for k in dict.fromkeys(ceil or []):
            keep = floor if repair else max(floor, self.ep.brief.order_due() + 2)
            if self.ep.moves_left() <= keep + 1 or not self._room_to_walk():
                break
            outcome = self.ep.scout.ceiling_walk(k, floor, repair)
            if outcome:
                report.append(outcome[0])
                rows[k] = list(outcome[1])
        return rows

    def run_walk(self, plan, floor, ceil=None, repair=False):
        paging = self.ep.lookout.page_arg()
        if not paging or not (plan or ceil) or not self.ep.role("search"):
            return False
        field = paging[0]
        report = []
        lines = {}
        ran = self._walk_pages(plan, field, floor, report)
        ran = self._walk_lines(plan, field, floor, report, lines) or ran
        ceiling_rows = self._ceiling_rounds(ceil, floor, repair, report)
        if not (ran or ceiling_rows):
            return False
        batch = self.ep.shop.batch_tool()
        most = batch[2] if batch else CEIL_OPENS if ceiling_rows else 2
        pools = [self._cheaper_rows(k, [lines.get(k, "")], most) for k, _, _ in plan]
        queue = self._interleave(pools + list(ceiling_rows.values()), most)
        self._fit_queue(queue, batch, floor)
        slots = [k for k, _, _ in plan] + list(ceiling_rows)
        homes = set().union(*(self.walk_home(k) for k in slots))
        self.ep.rank_checked = {m for m in self.ep.rank_checked if self.ep.counsel.shelf_known(m.split("|")[0]) not in homes}
        self.ep.order_vetted = False
        self._walk_summary(report, queue, floor)
        self.ep.log.append({"kind": "walk", "plan": [[k, n] for k, _, n in plan], "ceiling": list(ceiling_rows), "opened": queue})
        return True

    def _walk_summary(self, report, queue, floor):
        if queue:
            listed = ", ".join(f"{x} ({self.ep.known.price(x):g})" for x in queue)
            middle = ("Now I open the unread in-stock rows within budget, cheapest first, that these searches showed under each "
                      "pick's price: " + listed + ". ")
        else:
            middle = "No unread in-stock row in budget under the price of the pick turned up here. "
        summary = ("[cheapest walk] " + "; ".join(report) + ". " + middle
                   + "Per product, pick the lowest-priced listing whose own fields satisfy all of the requirements in "
                   "force; a rank check repeats with the cart call and at the order.")
        if queue:
            reads = self.ep.reads
            reads.open_queue = queue
            reads.open_recheck = True
            reads.open_say = "The lowest price is asked for: I open the cheapest unread rows walked."
            reads.open_purpose, reads.open_floor = "walk", floor
            self.ep.counsel.open_named()
        chat = self.ep.chat
        if chat and chat[-1].get("role") == "tool":
            chat[-1]["content"] = str(chat[-1]["content"]) + "\n" + summary + ("" if queue else "\n" + self.ep.status_line())

    # reads queued before a decline

    def _decline_fits(self):
        named = set(self.ep.counsel.audit_fits or []) - self._version_fails()
        return (self._version_passes() | named) - self.ep.refused_lines

    def _uncovered_products(self):
        covered = {self.ep.counsel.shelf_known(x) for x in self._decline_fits()}
        covered.discard("")
        if not self.ep.wants.items():
            return [] if set(self.ep.lookout.shelf_labels()) & covered else [None]
        return [k for k, home in enumerate(self.ep.shop.home_shelves()) if home and not home & covered]

    def _decline_walk_keys(self):
        live_needs = self.ep.wants.props_left()

        def worded(call_args, k):
            carried = self.walk_carries(call_args, k)
            return len([n for n in carried if n < len(live_needs)
                        and not re.search(r"as shown|same as|the page|sold by|redirect|refer", live_needs[n].lower())])
        chosen = []
        for k in self._uncovered_products():
            options = sorted((o for o in self._walk_options(k) if worded(o[2]["args"], k)),
                             key=lambda p: (-worded(p[2]["args"], k), -self.walk_score(p[2]["args"], k), p[0], p[1]))
            for _, key, _ in options:
                if key not in chosen:
                    chosen.append(key)
                    break
        return chosen[:2]

    def _labels_and_stock(self):
        labels = {}
        sold = {}

        def note(record):
            ident = self.ep.lookout.row_mark(record)
            if ident not in self.ep.known.ids or not self.ep.lookout.shelf_of(record):
                return
            labels[ident] = self.ep.lookout.shelf_of(record)
            if any(re.search(r"stock|avail", str(key).lower()) for key in record):
                sold[ident] = self.ep.lookout.sold_out(record)
        for record in self.ep.known.items:
            walk_dicts(record, note)
        return labels, sold

    def _unread_pools(self, homes, ids, below=None):
        known = self.ep.known
        budget = self.ep.wants.spend_cap
        labels, sold = self._labels_and_stock()
        pools = []
        for n, (home, units) in enumerate(homes):
            ranked = []
            for ident, label in labels.items():
                price = known.price(ident)
                skip = (label not in home or ident in known.read or ident in self.ep.refused_lines or sold.get(ident)
                        or (ids is not None and ident not in ids) or not is_number(price))
                if skip or (is_number(budget) and price * units > budget) or (below is not None and not price < below[n]):
                    continue
                ranked.append((float(price), ident))
            pools.append([ident for _, ident in sorted(ranked)])
        return pools

    def _over_budget_rows(self):
        try:
            products, budget = self.ep.wants.items(), self.ep.wants.spend_cap
            if not products or self.ep.wants.priority() == "cheapest" or not is_number(budget):
                return []
            homes = self.ep.shop.home_shelves()
            fits = self._decline_fits()
            floors = []
            shares = []
            for k, home in enumerate(homes):
                prices = [float(self.ep.known.price(x)) for x in fits
                          if self.ep.counsel.shelf_known(x) in home and is_number(self.ep.known.price(x))]
                if not (home and prices):
                    return []
                floors.append(min(prices))
                shares.append(min(prices) * products[k]["quantity"])
            if sum(shares) <= budget:
                return []
            pools = self._unread_pools([(home, products[k]["quantity"]) for k, home in enumerate(homes)], None, floors)
            ranked = [pools[k] for k in sorted(range(len(pools)), key=lambda k: -shares[k])]
            return ranked if any(ranked) else []
        except Exception as err:
            self.ep.log.append({"kind": "over_budget_error", "error": f"{type(err).__name__}: {err}"[:300]})
            return []

    def decline_reads(self):
        try:
            return self._queue_decline_reads()
        except Exception as err:
            self.ep.log.append({"kind": "decline_reads_error", "error": f"{type(err).__name__}: {err}"[:300]})
            return ""

    def _queue_decline_reads(self):
        batch = self.ep.shop.batch_tool()
        if self.decline_read_done or self.ep.time_left() < CLOSING + 60:
            return ""
        if self.ep.basket.units or not (self.ep.role("details") or batch):
            return ""
        opens = 2 if not batch else 1
        room = min(3, self.ep.moves_left() - opens - 6)
        keys = []
        if room > 0 and self.ep.lookout.page_arg() and self.ep.role("search") and self._id_ordered_store():
            keys = self._decline_walk_keys()
        cheaper = self._over_budget_rows() if self.ep.moves_left() > opens + 6 else []
        if not keys and not cheaper:
            return ""
        self.decline_read_done = True
        self.decline_queue = (keys, room, 6)
        parts = []
        if keys:
            named = "; ".join(f"'{' '.join(self.ep.lookout.free_text(self.ep.page_book[key]['args']))}'" for key in keys)
            parts.append("searches holding required values listed more listings than were read (" + named + "), and this "
                         "store sorts its results by a fixed id, not by fit or by price, so a fit is as likely to be found on "
                         f"pages not read as on read ones: the code reads up to {room} of these pages for you at once and opens "
                         "the cheapest in-stock rows in the budget they show")
        if cheaper:
            parts.append("added up across all the products, the listings found exceed the budget: the cheapest unread in-stock "
                         "rows priced under the pick of each product get opened as well")
        target = "one line per product, at its quantity" if self.ep.wants.items() else "the item"
        return ("Before a decline, " + "; ".join(parts) + ". Hold those records up against every requirement in force, by "
                "meaning; buy what fits (" + target + "), and decline only if each one fails.")

    def _decline_pages(self, walked, paging, most_pages, floor, opens):
        shop = self.ep.shop
        read = 0
        for key in walked:
            walk = self.ep.page_book.get(key)
            query = " ".join(self.ep.lookout.free_text(walk["args"]))
            said = (f"Ahead of a decline, I read the next page of '{query}': this store lists by fixed id, "
                    "so fits may lie on pages not read.")
            while read < most_pages and self.ep.moves_left() > floor + opens and not shop.done and not shop.broke:
                if self.ep.time_left() <= CLOSING + 30:
                    break
                following = self.ep.lookout.next_page(walk)
                if following is None or not self.walk_call({**walk["args"], paging[0]: following}, said):
                    break
                read += 1
        return read

    def run_decline_reads(self):
        keys, most_pages, floor = self.decline_queue
        self.decline_queue = None
        paging = self.ep.lookout.page_arg()
        batch = self.ep.shop.batch_tool()
        walked = [key for key in (keys if paging else []) if self.ep.page_book.get(key)]
        read = self._decline_pages(walked, paging, most_pages, floor, 1 if batch else 2)
        seen = set()
        for key in walked:
            seen.update((self.ep.page_book.get(key) or {}).get("row_shelf") or {})
        products = self.ep.wants.items()

        def units_for(k):
            if k is not None and k < len(products):
                return products[k]["quantity"]
            return self.ep.wants.quantity()
        homes = [(self.walk_home(k), units_for(k)) for k in self._uncovered_products()]
        pools = self._unread_pools(homes, seen) if walked else []
        queue = self._interleave(pools + self._over_budget_rows(), batch[2] if batch else 2)
        self._fit_queue(queue, batch, floor)
        self.ep.log.append({"kind": "decline_reads", "pages": read, "opened": queue})
        chat = self.ep.chat
        if queue and not self.ep.shop.done:
            reads = self.ep.reads
            reads.open_queue = queue
            reads.open_say = "Before any decline, I open the cheapest unread rows in budget that pages and picks gave."
            reads.open_purpose, reads.open_floor = "decline", floor
            self.ep.counsel.open_named()
        elif chat and chat[-1].get("role") == "tool":
            chat[-1]["content"] = (str(chat[-1]["content"]) + "\nNo unread in-stock row in the budget was on these "
                                   "pages.\n" + self.ep.status_line())
        return read > 0 or bool(queue)


class PointerLine:
    def __init__(self, text, phrase, start, end, parsed):
        self.text, self.phrase, self.parsed = text, phrase, parsed
        self.inside = text[start:end]
        self.outside = " ".join((text[:start], text[end:]))

    def inside_only(self, prop):
        if prop:
            return prop_named(prop, self.inside) and not prop_named(prop, self.outside)
        return False

    def describes_other(self, req, name):
        if self.inside_only(name(req)):
            return True
        value_tokens = set(ref_toks(str(req).partition(":")[2]))
        if value_tokens:
            inner, outer = set(ref_toks(self.inside)), set(ref_toks(self.outside))
            return value_tokens <= inner and not value_tokens & outer
        return False


class Pointers:
    def __init__(self, ep):
        self.ep = ep
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

    def ref_step(self):
        ep = self.ep
        if not ep.refs or ep.shop.done or ep.shop.broke:
            return False
        try:
            return self.chase_turn()
        except Exception as err:
            ep.log.append({"kind": "ref_error", "error": f"{type(err).__name__}: {err}"[:300]})
            for ref in ep.refs:
                if ref["status"] == "open":
                    ref["status"], ref["why"] = "stuck", "the code's resolver failed"
            return False

    def chase_turn(self):
        ep = self.ep
        pending = self.still_open()
        if not pending:
            return False
        acted = False
        tick_before = ep.shop.tick
        for ref in sorted(pending, key=lambda entry: entry["hops"]):
            if self.judge_read(ref):
                acted = True
        if ep.shop.tick > tick_before:
            return True
        pending = [ref for ref in pending if ref["status"] == "open"]
        if not (pending and self.ref_room()):
            return acted
        batch = ep.shop.batch_tool()
        most = 2 if not batch else batch[2]
        pools = self.candidate_pools(pending, most)
        template = self.search_template()
        starved = [ref for ref, pool in zip(pending, pools) if not pool and self.planned_queries(ref)] if template else []
        queue = self.round_robin(pools, most)
        needy = starved if 0 < len(queue) < most else []
        if queue and not needy:
            return self.read_candidates(queue, pending) or acted
        return self.search_or_give_up(needy, pending, template, acted) if template else acted

    def still_open(self):
        kept = []
        for ref in self.ep.refs:
            if ref["status"] != "open":
                continue
            if self.ref_live(ref):
                kept.append(ref)
            else:
                ref["status"] = "dropped"
                self.ep.log.append({"kind": "ref_dropped", "req": ref["req"][:200]})
        return kept

    def ref_live(self, ref):
        def about_prop(req):
            return same_prop(self.bare_prop(str(req).partition(":")[0]), ref["prop"])
        for req in self.wants.props_left():
            if self.ref_match(ref, req) or about_prop(req):
                return True
        return not any(about_prop(req) for req in self.wants.let_go)

    @property
    def wants(self):
        return self.ep.wants

    @staticmethod
    def bare_prop(head):
        return re.split(r"\s*\(", str(head))[0]

    def ref_match(self, ref, req):
        line = str(req)
        if line in (ref["req"], ref.get("now")):
            return True
        head, _, value = line.partition(":")
        pointing = bool(ref_text(value)) or (ref.get("relation") == "differ" and re.search(DIFFER_WORDS, value.lower()))
        if not pointing or not same_prop(self.bare_prop(head), ref["prop"]):
            return False
        brackets = re.findall(r"\(([^)]*)\)", head)
        if brackets and ref["tag"]:
            return bool(kind_words(brackets[-1]) & kind_words(ref["tag"]))
        return True

    def candidate_pools(self, pending, most):
        read = self.known.read
        pools = []
        for ref in pending:
            pool = list(filter(lambda ident: ident not in read and ident not in ref["judged"], self.candidates(ref)))
            leading = (ref.get("shows") or {}).get(pool[0]) if pool else None
            if ref["queries"] or not self.planned_queries(ref) or len(pool) <= most or leading:
                pools.append(pool)
            else:
                pools.append([])
        return pools

    @property
    def known(self):
        return self.ep.known

    def candidates(self, ref):
        target = ref["target"]
        seller = ref_name(target["seller"])
        kinds = kind_words(target["kind"])
        listings = self.listing_index()
        seen_at = self.index_seen_at or {}
        ranking = []
        for ident, listing in listings.items():
            if seller in self.sellers_of(listing):
                shown = self.title_shows(listing, target["details"])
                bonus = 2 if kinds & kind_words(self.ep.lookout.shelf_of(listing)) else 0
                named = len(kinds & kind_words(self.ep.shop.title_of(listing)))
                ranking.append((-(bonus + shown + named), -seen_at.get(ident, 0), ident, shown))
        ranking.sort()
        ref["shows"] = dict((entry[2], entry[3]) for entry in ranking)
        return [entry[2] for entry in ranking]

    def listing_index(self):
        records = self.known.items
        size = len(records)
        if self.index_memo and self.index_memo[0] == size:
            return self.index_memo[1]
        merged, latest = {}, {}
        for at, record in enumerate(records):
            found = []
            walk_dicts(record, found.append)
            for fields in found:
                if self.indexable(fields):
                    merged.setdefault(self.ep.lookout.row_mark(fields), {}).update(fields)
                    latest[self.ep.lookout.row_mark(fields)] = at
        self.index_memo = (size, merged)
        self.index_seen_at = latest
        return merged

    def indexable(self, fields):
        if self.ep.lookout.row_mark(fields) not in self.known.ids:
            return False
        return len(fields) >= 4 or any(isinstance(value, dict) and len(value) >= 2 for value in fields.values())

    def sellers_of(self, row):
        learned = self.seller_fields()
        names = set()
        for key, value in row.items():
            lowered = str(key).lower()
            if not isinstance(value, str) or not value.strip() or len(value) > 60 or "id" in lowered:
                continue
            if not (re.search(SELLER_KEYS, lowered) or str(key) in learned):
                continue
            if not re.search(r"note|text|desc|comment|detail|info|message|section", lowered):
                names.add(ref_name(value))
        return names - {""}

    def seller_fields(self):
        try:
            chased = self.chased_sellers()
            stamp = (len(self.known.items), tuple(sorted(chased)))
            if self.seller_field_memo and self.seller_field_memo[0] == stamp:
                return self.seller_field_memo[1]
            listings = self.listing_index()
            map_words = set()
            for listing in listings.values():
                for inner in filter(lambda value: isinstance(value, dict), listing.values()):
                    map_words |= {ref_name(x) for x in inner.values() if isinstance(x, str)}
            noted = set(word for word in self.proper_names(listings) if len(word) >= 4) - map_words
            fields = set(self.seller_like_keys(listings, chased, noted))
            self.seller_field_memo = (stamp, fields)
            return fields
        except Exception:
            return set()

    def chased_sellers(self):
        found = set()
        for ref in self.ep.refs or []:
            for side in (ref.get("target") or {}, ref.get("first") or {}):
                if isinstance(side, dict):
                    found.add(ref_name(side.get("seller")))
        found.discard("")
        return found

    @staticmethod
    def seller_like_keys(listings, chased, noted):
        skip = (r"title|heading|name|label|caption|categor|shelf|department|aisle|note|text|desc|comment|detail|info|"
                r"message|section|url|image|summary|overview")
        for listing in listings.values():
            for key, value in listing.items():
                lowered = str(key).lower()
                if not isinstance(value, str) or not 0 < len(value) <= 60 or "id" in lowered or re.search(skip, lowered):
                    continue
                name = ref_name(value)
                if name in chased or (value.strip()[:1].isupper() and len(value.split()) <= 3 and name in noted):
                    yield str(key)

    def proper_names(self, rows):
        found = set()
        for row in rows.values():
            for sentence in re.split(r"(?<=[.!?;])\s+", self.note_text(row, desc=False)):
                for streak in self.capital_streaks(re.findall(r"[A-Za-z][A-Za-z'-]*", sentence)[1:]):
                    for first in range(len(streak)):
                        for last in range(first + 1, min(len(streak), first + 3) + 1):
                            found.add(ref_name(" ".join(streak[first:last])))
        return found

    @staticmethod
    def note_text(row, desc=True):
        notes, blurbs = [], []
        for key, val in row.items():
            lowered = str(key).lower()
            if not isinstance(val, str) or len(val.split()) < 3 or lowered == "id" or lowered.endswith("id"):
                continue
            if re.search(r"title|heading|headline|name|label|caption|categor|shelf|department|url|image", lowered):
                continue
            (blurbs if re.search(r"desc|overview|summary|about", lowered) else notes).append(val)
        if notes:
            return "\n".join(notes)
        return "\n".join(blurbs) if desc else ""

    @staticmethod
    def capital_streaks(words):
        streak = []
        for word in words + [""]:
            if word[:1].isupper() and re.search(r"[a-z]", word):
                streak.append(word)
            else:
                yield streak
                streak = []

    def title_shows(self, listing, details):
        blurbs = [str(text) for name, text in listing.items()
                  if isinstance(text, str) and re.search(r"desc|overview|summary", str(name).lower())]
        pool = set(ref_toks(self.ep.shop.title_of(listing) + " " + " ".join(blurbs)))
        return len([detail for detail in details if set(ref_toks(detail)) and set(ref_toks(detail)) <= pool])

    def planned_queries(self, ref):
        width = max(2, self.ep.lead_words or 3)
        target = ref["target"]
        seller = " ".join(re.findall(r"[A-Za-z0-9]+", re.sub(r"'s\b", "", str(target["seller"]))))
        words = self.detail_words(ref)
        kind = [word for word in re.findall(r"[a-z]{3,}", str(target["kind"]).lower()) if word not in LINK_WORDS][-1:]
        plans = []
        if ref.get("unindexed"):
            if words or kind:
                plans.append(" ".join((words[:1] + kind)[:width]))
        else:
            full = self.full_query_words(ref, words)
            if full and self.ep.word_kind != "all":
                plans.append(" ".join((seller.split() + full)[:REF_QUERY_WORDS]))
            if words and len(seller.split()) < width:
                plans.append(seller + " " + words[0])
            plans.append(seller)
        return list(filter(lambda plan: plan.strip() and plan.lower() not in ref["queries"], plans))

    @staticmethod
    def full_query_words(ref, filtered=()):
        target = ref["target"]
        marks = target.get("marks") or (list(filtered) if ref.get("hops") else [])
        seller = set(re.findall(r"[a-z0-9]+", str(target["seller"]).lower()))
        words = lambda texts: [w for text in texts for w in re.findall(r"[a-z0-9]+(?:\.[0-9]+)?", str(text).lower())
                               if w not in LINK_WORDS and w not in seller and len(w) >= 2]
        sort = words([target["kind"]])
        named = {k.rstrip("s") for k in sort}
        kept = [w for w in dict.fromkeys(words(marks)) if w.rstrip("s") not in named] + list(dict.fromkeys(sort))
        return kept if len(kept) >= 2 else []

    def detail_words(self, ref):
        vocabulary = self.known_words()
        details = ref["target"]["details"]
        seller_words = set(re.findall(r"[a-z0-9]+", str(ref["target"]["seller"]).lower()))
        buckets = {"mixed": [], "numbers": [], "plain": []}
        for detail in details:
            for word, bucket in self.sorted_detail_words(detail, seller_words, vocabulary):
                buckets[bucket].append(word)
        plain = sorted(buckets["plain"], key=len, reverse=True)
        first = self.value_words_first(ref, details, plain)
        ordered = [word for word in first if word in plain] + [word for word in plain if word not in first]
        return list(dict.fromkeys(buckets["mixed"] + buckets["numbers"] + ordered))

    def known_words(self):
        words = set()
        for row in self.listing_index().values():
            for val in row.values():
                if not isinstance(val, dict):
                    continue
                for key in val:
                    spaced = re.sub(r"([a-z])([A-Z])", r"\1 \2", str(key)).lower()
                    words.update(re.findall(r"[a-z]+", spaced))
        for req in self.wants.written + self.wants.uncovered:
            words.update(re.findall(r"[a-z]+", self.bare_prop(str(req).partition(":")[0]).lower()))
        return words

    def value_words_first(self, ref, details, plain):
        names = self.field_names()
        split = [[w for w in re.findall(r"[a-z0-9]+", str(d).lower()) if w not in LINK_WORDS] for d in details]
        paired = ref.get("hops", 0) == 0 and split and all(len(parts) >= 2 for parts in split)
        leading = []
        for parts in split:
            if len(parts) >= 2 and parts[-1] in plain:
                if paired or any(" ".join(parts[:cut]) in names for cut in range(1, len(parts))):
                    leading.append(parts[-1])
        return leading

    def field_names(self):
        stamp = len(self.known.items)
        memo = self.field_names_memo
        if memo and memo[0] == stamp:
            return memo[1]
        names = set()

        def collect(fields):
            for val in fields.values():
                if isinstance(val, dict) and sum(1 for x in val.values() if isinstance(x, (str, int, float))) >= 2:
                    for key in val:
                        spaced = re.sub(r"([a-z])([A-Z])", r"\1 \2", str(key)).lower()
                        joined = " ".join(re.findall(r"[a-z0-9]+", spaced))
                        if joined:
                            names.add(joined)
        for record in self.known.items:
            walk_dicts(record, collect)
        self.field_names_memo = (stamp, names)
        return names

    @staticmethod
    def sorted_detail_words(detail, seller_words, vocabulary):
        before = ""
        for whole in str(detail).lower().split():
            unit_slot = bool(re.fullmatch(r"\d+(?:[.,]\d+)?", before))
            before = whole
            bare = whole.strip(".,;:!?()[]{}\"'")
            for word in re.findall(r"[a-z0-9]+", whole):
                digits = re.search(r"\d", word)
                if word in LINK_WORDS or word in seller_words or (unit_slot and not digits):
                    continue
                if digits and re.search(r"[a-z]", word):
                    yield word, "mixed"
                elif word.isdigit():
                    if len(word) >= 2:
                        yield word, "numbers"
                elif len(word) >= 3 and word == bare and word not in vocabulary and word.rstrip("s") not in vocabulary:
                    yield word, "plain"

    def search_template(self):
        ep = self.ep
        shape = ep.shapes.get(ep.role("search") or "") or {}
        required = shape.get("required")
        needed = set(required or []) if isinstance(required, list) else set()
        labels = set(map(ep.lookout.shelf_of, self.listing_index().values()))
        labels.discard("")
        modes, blind = self.search_modes(needed, labels)
        paging = ep.lookout.page_arg()
        template = modes[0] if modes else self.logged_template(needed, labels, blind, paging)
        if template:
            rest, key = template
            kept = {}
            for name, value in rest.items():
                if name in needed or not (isinstance(value, str) and value.strip().lower() in labels):
                    kept[name] = value
            return kept, key
        return self.lone_text_key(shape, needed)

    def search_modes(self, needed, labels):
        lookout = self.ep.lookout
        groups = {}
        for order, (rest, text_key, query) in enumerate(self.single_text_searches(needed, labels)):
            if rest is None:
                continue
            slot = json.dumps(rest, sort_keys=True, default=str)
            if slot not in groups:
                groups[slot] = {"tpl": (rest, text_key), "seen": [], "at": order}
            stems = set(map(lookout.stem, re.findall(r"[a-z0-9]+", query[0].lower())))
            groups[slot]["seen"].append((stems, query[1]))
        ranked, blind = [], set()
        for slot, group in groups.items():
            if self.blind_to_words(group["seen"]):
                blind.add(slot)
            else:
                ranked.append((-len(set(tuple(sorted(words)) for words, _ in group["seen"])), group["at"], group["tpl"]))
        ranked.sort(key=lambda row: row[:2])
        return [row[2] for row in ranked], blind

    def single_text_searches(self, needed, labels):
        free_text = self.ep.lookout.free_text
        for entry in self.ep.page_book.values():
            args = entry.get("args")
            if not isinstance(args, dict):
                args = {}
            texts = free_text(args)
            text_keys = [name for name, value in args.items() if isinstance(value, str) and value in texts and value.strip()]
            if len(texts) != 1 or len(text_keys) != 1 or texts[0].strip().lower() in labels:
                yield None, None, None
                continue
            text_key = text_keys[0]
            rest = {}
            for name, value in args.items():
                if name == text_key:
                    continue
                if name in needed or not (isinstance(value, str) and value.strip().lower() in labels):
                    rest[name] = value
            yield rest, text_key, (texts[0], tuple(entry.get("first") or ()))

    @staticmethod
    def blind_to_words(seen):
        for at, (first_words, first_rows) in enumerate(seen):
            if len(first_rows) < 2:
                continue
            for later_words, later_rows in seen[at + 1:]:
                if first_rows == later_rows and not first_words & later_words:
                    return True
        return False

    def lone_text_key(self, shape, needed):
        props = shape.get("properties")
        if not isinstance(props, dict):
            props = {}
        token = self.ep.lookout.token_key()
        free = [name for name, spec in props.items()
                if isinstance(spec, dict) and spec.get("type") == "string" and not spec.get("enum") and name != token]
        if len(free) != 1 or needed - {free[0]}:
            return None
        return {}, free[0]

    def logged_template(self, needed, labels, blind, paging):
        for args in self.logged_searches():
            texts = [name for name, value in args.items()
                     if isinstance(value, str) and value.strip() and value in self.ep.lookout.free_text(args)]
            if len(texts) != 1 or str(args[texts[0]]).strip().lower() in labels:
                continue
            rest = dict((name, value) for name, value in args.items() if name != texts[0] and not (paging and name == paging[0]))
            kept = dict((name, value) for name, value in rest.items()
                        if name in needed or not (isinstance(value, str) and value.strip().lower() in labels))
            if json.dumps(kept, sort_keys=True, default=str) not in blind:
                return rest, texts[0]
        return None

    def logged_searches(self):
        search = self.ep.role("search")
        for entry in reversed(self.ep.log):
            if entry.get("kind") == "env":
                for act in reversed(entry.get("actions") or []):
                    if act.get("name") == search and isinstance(act.get("args"), dict):
                        yield act.get("args")

    @staticmethod
    def round_robin(pools, most):
        queue = []
        while len(queue) < most and any(pools):
            for pool in pools:
                while pool and pool[0] in queue:
                    del pool[0]
                if pool and len(queue) < most:
                    queue.append(pool.pop(0))
        return queue

    def judge_read(self, ref):
        read = self.known.read
        unjudged = [ident for ident in self.candidates(ref) if ident in read and ident not in ref["judged"]]
        if not unjudged:
            return False
        listings = self.listing_index()
        same_seller = [ident for ident in self.candidates(ref) if ident in read]
        tokens = {ident: self.token_parts(listings.get(ident) or {}) for ident in set(unjudged).union(same_seller)}
        details = ref["target"]["details"]
        lenient = ref.setdefault("loose", set())
        shows = lambda ident, detail: self.may_show(listings.get(ident) or {}, detail, tokens[ident])
        if len(same_seller) > 3:
            lenient |= {detail for detail in details if not any(shows(ident, detail) for ident in same_seller)}
        strict = [detail for detail in details if detail not in lenient]
        passes = {ident: all(shows(ident, detail) for detail in strict) for ident in unjudged}
        whole = {ident: self.covered_count(tokens[ident], details) for ident in unjudged}
        ranked = sorted(unjudged, key=lambda ident: (not passes[ident], -whole[ident], unjudged.index(ident)))
        if not passes[ranked[0]] and self.more_to_read(ref):
            return False
        pick = [ident for ident in ranked if passes[ident]][:4] or ranked[:4]
        exact = [ident for ident in pick if details and whole[ident] == len(details)]
        asked = exact if len(exact) == 1 else ([ident for ident in pick if passes[ident]][:2] or pick[:1])
        verdict = self.settle_by_notes(ref, pick, self.notes_unread(ref, asked))
        if verdict is None:
            failing = [ident for ident in unjudged if not passes[ident]]
            verdict = self.hop_or_settle(ref, ref["target"], pick, exact, listings, failing)
        return verdict

    def settle_by_notes(self, ref, pick, missing):
        if not missing:
            return None
        if self.ref_room() and self.read_notes(ref, missing):
            return True
        if self.waits_for_others():
            return False
        if not self.ref_room() and ref.get("relation") == "differ":
            ref["judged"].update(pick)
            self.give_up(ref, "the notes of the listing it points at could not be read in the steps left")
            return True
        return None

    def waits_for_others(self, floor=None):
        ep = self.ep
        rows = self.wants.items()
        pending = self.open_refs()
        if not (rows and pending) or any(ref.get("relation") != "differ" or not ref.get("chain") for ref in pending):
            return False
        limit = floor if floor is not None else self.step_floor()
        if ep.moves_left() <= limit + 4:
            return False
        try:
            owners = set(map(self.product_for, pending))
            lacking = [k for k in range(len(rows)) if not any(entry["fit"] for entry in ep.shop.ed_pool(k))]
        except Exception as err:
            ep.log.append({"kind": "ref_wait_error", "error": f"{type(err).__name__}: {err}"[:300]})
            return False
        others = [k for k in lacking if k not in owners - {None}]
        waiting = bool(others) if None not in owners else len(others) >= 2
        if waiting and tuple(others) != self.wait_logged:
            self.wait_logged = tuple(others)
            ep.log.append({"kind": "ref_wait", "lacking": others})
        return waiting

    def open_refs(self):
        return [ref for ref in self.ep.refs or [] if ref.get("status") == "open"]

    def step_floor(self):
        cap = self.ep.step_cap
        reserve = max(FIRST_ORDER_BY, min(FIRST_ORDER_DUE, cap // 4 + 1)) + 10
        return max(reserve, self.ep.brief.order_due() + 7)

    def product_for(self, ref):
        if self.wants.items() and ref.get("tag"):
            return self.ep.talk.product_of(f"x ({ref['tag']})")
        return None

    def ref_room(self):
        ep = self.ep
        pending = self.open_refs()
        differ_only = bool(pending) and all(r.get("relation") == "differ" for r in pending)
        if differ_only and all(not r.get("chain") for r in pending) and self.steps_spent >= REF_DIFFER_BLIND:
            return False
        cap = ep.step_cap
        floor = self.step_floor()
        limit = max(6, int(REF_DIFFER_STEPS * cap)) if differ_only else max(8, int(REF_STEPS * cap))
        shop = ep.shop
        if (self.steps_spent >= limit or ep.time_left() <= CLOSING + 60 or ep.counsel.over_budget()
                or shop.done or shop.broke):
            return False
        if self.waits_for_others(floor):
            return False
        return ep.moves_left() > floor or self.late_hop_room()

    def late_hop_room(self):
        ep = self.ep
        if ep.moves_left() <= ep.brief.order_due() + 2:
            return False
        for ref in filter(self.late_candidate, ep.refs or []):
            began = ref.get("late_from")
            if began is not None:
                if self.steps_spent - began < 3:
                    return True
                continue
            if ref.get("queries") or not self.others_verified(ref):
                continue
            ref["late_from"] = self.steps_spent
            ep.log.append({"kind": "ref_late", "prop": ref.get("prop"), "to": str((ref.get("target") or {}).get("seller"))[:80]})
            return True
        return False

    @staticmethod
    def late_candidate(ref):
        target = ref.get("target") or {}
        if ref.get("status") != "open" or ref.get("hops", 0) < 1:
            return False
        return bool(str(target.get("seller") or "").strip() and str(target.get("kind") or "").strip() and target.get("details"))

    def others_verified(self, ref):
        rows = self.wants.items()
        if not rows:
            return True
        try:
            unproven = [k for k in range(len(rows))
                        if not any(e["verified"] and e["fit"] for e in self.ep.shop.ed_pool(k))]
        except Exception:
            return False
        own = self.product_for(ref)
        if own is not None:
            return all(k == own for k in unproven)
        return len(unproven) <= 1

    def read_notes(self, ref, need):
        tool = self.notes_reader()
        if not (tool and need):
            return False
        ep = self.ep
        ref.setdefault("noted", set()).update(need)
        tick_before = ep.shop.tick
        say = (f"The request copies {ref['head']} from another listing: I open the notes of the listing that may be it ({tool}), as a note may send the reader on.")
        replies = ep.counsel.open_reads(need, say, None, tool)
        if replies:
            ep.talk.post_replies(replies)
        self.steps_spent += ep.shop.tick - tick_before
        self.remember_ids(need)
        ep.log.append(dict(kind="ref_notes", ids=need, tool=tool))
        return ep.shop.tick > tick_before

    def notes_reader(self):
        shop = self.ep.shop
        learned = shop.notes_tool()
        if learned:
            return learned
        details = self.ep.role("details") or ""
        if (shop.note_keys or {}).get(details):
            return None
        tool_keys, text_keys, caps = shop.tool_keys or {}, shop.text_keys or {}, shop.tool_caps or {}
        known_keys = set(tool_keys.get(details) or set()) | set(shop.row_keys or set())
        others = [tool for tool in shop.lookup_tools() if tool != self.ep.role("details")]

        def adds_text(tool):
            return bool((set(tool_keys.get(tool) or set()) - known_keys) & set(text_keys.get(tool) or set()))
        extra = list(filter(adds_text, others))
        if extra:
            return max(extra, key=lambda tool: ((caps.get(tool) or {}).get("calls", 0), tool))
        fallback = [tool for tool in (self.ep.role("view"), shop.stock_tool())
                    if tool in others and not (caps.get(tool) or {}).get("id", 0)]
        return fallback[0] if fallback else None

    def remember_ids(self, ids):
        self.candidate_ids = set(self.candidate_ids or set()) | set(ids)

    def give_up(self, ref, why):
        if ref["status"] == "open":
            ref["status"], ref["why"] = "stuck", why
            self.ep.log.append({"kind": "ref_stuck", "req": ref["req"][:200], "why": why[:300], "chain": list(ref["chain"])})
            self.tell_actor(f"NOT RESOLVED: {ref['head']} ({why}). Its value remains unknown: should the notes you have read show where it ends, use that value; otherwise never decline or reject a listing over it.")

    def tell_actor(self, text):
        chat = self.ep.chat
        newest = chat[-1] if chat else None
        if newest and newest.get("role") in ("tool", "user") and isinstance(newest.get("content"), str) and len(chat) > 2:
            newest["content"] += "\n[reference] " + text
        else:
            self.pending_tells = list(self.pending_tells or []) + [text]

    def hop_or_settle(self, ref, target, pick, exact, listings, failing):
        if len(exact) == 1 and self.plain_record(ref, exact[0]):
            only = exact[0]
            ref["judged"].update(pick)
            self.remember_ids(pick)
            self.resolve(ref, only, self.field_value(listings.get(only) or {}, ref["prop"]), [])
        elif self.hop_budget_ok():
            self.apply_hop(ref, pick, self.ask_hop(ref, pick, ""))
            if ref["target"] is target and ref["status"] == "open":
                ref["judged"].update(failing)
        else:
            ref["judged"].update(pick)
            self.give_up(ref, "no hop checks are left")
        return True

    def plain_record(self, ref, ident):
        listing = self.listing_index().get(ident) or {}
        value = self.field_value(listing, ref["prop"])
        notes = self.note_text(listing, desc=False)
        if not value or self.names_property(ref["prop"], notes):
            return False
        if set(ref_toks(value)).intersection(ref_toks(notes)):
            return False
        own = self.sellers_of(listing)
        elsewhere = set().union(*(self.sellers_of(other) for other in self.listing_index().values())) - own
        fields = self.field_tokens(listing, self.token_parts(listing)[0])
        for sentence in re.split(r"(?<=[.!?;])\s+", notes):
            for word in re.findall(r"[A-Za-z][A-Za-z'-]*", sentence)[1:]:
                lowered = ref_name(word)
                capital_stranger = word[:1].isupper() and len(lowered) >= 4 and lowered not in own and lowered not in fields
                if lowered in elsewhere or capital_stranger:
                    return False
        return True

    @staticmethod
    def field_value(row, prop):
        maps = [val for val in row.values() if isinstance(val, dict)]
        for matcher in (strict_prop, same_prop):
            for fields in maps:
                for key, x in fields.items():
                    if isinstance(x, (str, int, float)) and not isinstance(x, bool) and str(x).strip() and matcher(key, prop):
                        return str(x).strip()
        return ""

    def field_tokens(self, row, names):
        pieces = [self.ep.shop.title_of(row), self.ep.lookout.shelf_of(row)]
        for val in row.values():
            if isinstance(val, dict):
                pieces.extend(str(x) for x in val.values() if isinstance(x, (str, int, float)))
        return names | set(ref_toks(" | ".join(pieces)))

    @staticmethod
    def names_property(prop, text):
        words = [w for w in prop_words(prop) if len(w) >= 3]
        if not words:
            return True
        for token in set(re.findall(r"[a-z]+", str(text).lower())):
            for word in words:
                if token == word:
                    return True
                if len(word) >= 4 and len(token) >= 4 and token[:4] == word[:4]:
                    return True
        return False

    def token_parts(self, row):
        trimmed = re.sub(r"\b(?:not|no|never|without|nor)\b(?:\W+[A-Za-z0-9.:/&'-]+){1,3}", " ",
                         self.note_text(row, desc=False), flags=re.I)
        names, values = [], [self.ep.shop.title_of(row), trimmed]
        for fields in self.value_maps(row):
            for key, value in fields.items():
                names.append(re.sub(r"([a-z])([A-Z])", r"\1 \2", str(key)).replace("_", " "))
                if isinstance(value, (str, int, float)) and not isinstance(value, bool):
                    values.append(str(value))
        return set(ref_toks(" ".join(names))), set(ref_toks(" | ".join(values)))

    @staticmethod
    def value_maps(row):
        maps = [value for value in row.values() if isinstance(value, dict)]
        if maps:
            return maps
        flat = {}
        for key, value in row.items():
            if not isinstance(value, (str, int, float)) or isinstance(value, bool):
                continue
            if not re.search(r"id$|title|shop|seller|brand|price|amount|desc|overview|note", str(key).lower()):
                flat[key] = value
        return [flat]

    def resolve(self, ref, target, value, pairs):
        shown = str(value or "").strip()
        if not shown:
            self.give_up(ref, f"{target} shows no value for it")
            return
        ref["chain"].append(target)
        ref.update(status="resolved", value=shown, goes_with=list(pairs))
        ref["now"] = updated = f"{ref['head']}: {self.resolved_value(ref)}"
        self.swap_in(ref, updated)
        wants = self.wants
        wants.resolved.append(updated)
        wants.version += 1
        self.ep.log.append({"kind": "ref_resolved", "req": ref["req"][:200], "now": updated, "chain": list(ref["chain"])})
        tail = self.resolved_tail(ref, pairs)
        self.tell_actor(f"RESOLVED: another listing gives {ref['head']}, and {target} shows {ref['value']}{tail}. "
                        f"The requirement is now: {updated}.")

    def swap_in(self, ref, new_req):
        wants = self.wants
        covers = self.covered_by(ref)
        swap = lambda lines: [new_req if covers(line) else line for line in lines]
        changed = False
        for field in ("written", "uncovered"):
            old = list(getattr(wants, field))
            new = swap(old)
            if new != old:
                changed = True
            setattr(wants, field, list(dict.fromkeys(new)))
        if wants.now_reqs is not None:
            swapped = swap(wants.now_reqs)
            if new_req not in swapped:
                swapped += [new_req]
            wants.now_reqs = list(dict.fromkeys(swapped))
        if not changed:
            wants.uncovered.append(new_req)
        reading = wants.reading
        if isinstance(reading.get("requirements"), list):
            reading["requirements"] = list(wants.written)
        self.rewrite_items(reading, ref, covers, new_req.partition(":")[2].strip())

    def covered_by(self, ref):
        first_seller = ref_name(ref["first"]["seller"])
        tag_kinds = kind_words(ref["tag"]) if ref["tag"] else set()

        def covers(req):
            if self.ref_match(ref, req):
                return True
            head, _, value = str(req).partition(":")
            brackets = re.findall(r"\(([^)]*)\)", head)
            if ref["tag"] and brackets and not kind_words(brackets[-1]) & tag_kinds:
                return False
            if not first_seller or first_seller not in ref_name(value):
                return False
            return same_prop(self.bare_prop(head), ref["prop"])
        return covers

    @staticmethod
    def rewrite_items(reading, ref, covers, value):
        entries = [entry for entry in reading.get("items") or []
                   if isinstance(entry, dict) and isinstance(entry.get("requirements"), list)]
        for entry in entries:
            foreign = len(entries) >= 2 and ref["tag"] and not kind_words(entry.get("product") or "") & kind_words(ref["tag"])
            if foreign:
                continue
            fresh = []
            for line in entry["requirements"]:
                fresh.append(f"{str(line).partition(':')[0].strip()}: {value}" if covers(line) else line)
            entry["requirements"] = fresh

    @staticmethod
    def resolved_tail(ref, pairs):
        chain = ref["chain"]
        via = (f", reached from {chain[0]} through the listings' notes; each page before {chain[-1]} sends the reader on for "
               f"this property, so a value in that page's own fields does not count") if len(chain) > 1 else ""
        joined = " and ".join(pairs)
        if ref["relation"] == "differ":
            paired = f"; its notes pair {joined} with it" if pairs else ""
            return via + paired + ". The shopper's value MUST DIFFER from that one: any other value meets it, that value does not"
        if pairs and ref["relation"] == "pairs":
            return via + f"; its notes say {ref['value']} goes together with {joined}, and the shopper's value has to go with that one, so {joined} meets it, not {ref['value']} itself"
        if pairs:
            return via + f"; its notes say {ref['value']} goes together with {joined}, so each of them meets it"
        return via

    @staticmethod
    def resolved_value(ref):
        value, relation = str(ref["value"]), ref["relation"]
        if relation == "within" and is_number(ref.get("tol")):
            found = re.search(r"(-?\d+(?:\.\d+)?)\s*([a-z%]*)", value.lower().replace(",", ""))
            if found is not None:
                centre, spread = float(found.group(1)), float(ref["tol"])
                unit = found.group(2) or str(ref.get("unit") or "")
                span = "{:g} to {:g} {}".format(round(centre - spread, 6), round(centre + spread, 6), unit).strip()
                return "{} (within {:g} of {})".format(span, spread, value)
        if relation in ("at most", "at least"):
            return relation + " " + value
        partners = [other for other in ref.get("goes_with") or [] if other.lower() != value.lower()]
        if relation == "pairs" and partners:
            return " or ".join(partners)
        options = [value] + partners
        return " and ".join("not " + option for option in options) if relation == "differ" else " or ".join(options)

    def hop_budget_ok(self):
        counsel = self.ep.counsel
        if self.hop_calls < REF_CALLS and self.ep.time_left() > CLOSING + 40 and not counsel.over_budget():
            return counsel.spend < 0.8 * SPEND_CAP * counsel.scale
        return False

    def apply_hop(self, ref, pick, answer, again=True):
        if not isinstance(answer, dict):
            ref["fails"] += 1
            if ref["fails"] > 1:
                ref["judged"].update(pick)
                self.give_up(ref, "the hop check gave no answer")
            return
        given = answer.get("target")
        usable = isinstance(given, (str, int)) and not isinstance(given, bool)
        target = str(given).strip() if usable else ""
        missing = self.clean_missing(answer.get("missing"))
        if target in pick and not missing:
            self._take_hop(ref, pick, target, answer, again)
            return
        ref["judged"].update(pick)
        self.remember_ids(pick)
        if target in pick:
            self.ep.log.append({"kind": "ref_partial", "prop": ref["prop"], "target": target, "missing": missing[:4]})

    @staticmethod
    def clean_missing(listed):
        if not isinstance(listed, list):
            return []
        return [str(item) for item in listed or [] if str(item).strip()]

    def _take_hop(self, ref, pick, target, answer, again):
        where, problem, go, value, pairs = self.check_hop(ref, target, answer, pick)
        if problem and again and self.hop_budget_ok():
            self.apply_hop(ref, pick, self.ask_hop(ref, pick, problem), again=False)
            return
        self.remember_ids(pick)
        if problem:
            self.reject_hop(ref, pick, target, where, problem)
            return
        ref["judged"].update(pick)
        if target in ref["chain"]:
            self.give_up(ref, "its notes lead back to a listing already followed")
        elif go:
            self.follow(ref, target, go)
        else:
            self.resolve(ref, target, value, pairs)

    def check_hop(self, ref, target, answer, pick=None):
        listing = self.listing_index().get(target) or {}
        parts = self.token_parts(listing)
        complaint = self._hop_mismatch(ref, target, listing, parts, pick)
        if complaint:
            return "target", complaint, None, "", []
        notes = self.note_text(listing)
        note_tokens = set(ref_toks(notes))
        onward = answer.get("go_to")
        if isinstance(onward, dict) and str(onward.get("seller") or "").strip():
            return self.check_go_to(ref, target, onward, listing, notes, note_tokens)
        value = str(answer.get("value") or "").strip()
        value_tokens = set(ref_toks(value))
        if not (value and value_tokens and value_tokens <= parts[0] | parts[1]):
            return "answer", f"the value '{value}' is not written in {target}'s record.", None, "", []
        companions = [paired for paired in dict.fromkeys(self.clean_strings(answer.get("goes_with")))
                      if self._pairs_in_notes(paired, value_tokens, note_tokens)]
        return "", "", None, value, companions[:3]

    def _hop_mismatch(self, ref, target, listing, parts, pick):
        loose = ref.get("loose") or set()
        for detail in ref["target"]["details"]:
            if detail not in loose and not self.may_show(listing, detail, parts):
                return f"{target} does not show '{detail}', so it is not the listing described."
        best = max(pick or [target], key=lambda ident: self.details_shown(ref, ident))
        if self.details_shown(ref, best) > self.details_shown(ref, target):
            return f"{best} shows more of the details named than {target}."
        return ""

    def details_shown(self, ref, ident):
        return self.covered_count(self.token_parts(self.listing_index().get(ident) or {}), ref["target"]["details"])

    @staticmethod
    def covered_count(parts, details):
        pool = parts[0] | parts[1]
        count = 0
        for detail in details:
            wanted = set(ref_toks(detail))
            count += bool(wanted) and wanted <= pool
        return count

    def may_show(self, row, detail, parts=None):
        names, values = parts or self.token_parts(row)
        words = set(token for token in ref_toks(detail) if len(token) >= 3 and not re.search(r"\d", token)) - names
        if words and words.isdisjoint(values):
            return False
        wanted = ref_figs(detail)
        if not wanted:
            return True
        shown = ref_figs(self.figures_text(row))
        for figure, unit in wanted:
            alike = [number for number, other_unit in shown if other_unit == unit]
            if alike and not any(abs(number - figure) < 1e-9 for number in alike):
                return False
        return True

    def figures_text(self, row):
        pieces = [self.ep.shop.title_of(row)]
        for val in row.values():
            if isinstance(val, dict):
                pieces.extend(str(x) for x in val.values() if isinstance(x, (str, int, float)))
        return " | ".join(pieces)

    def check_go_to(self, ref, target, go, row, notes, note_tokens):
        seller = str(go["seller"]).strip()
        pattern = r"(?<![a-z0-9])" + re.escape(seller.lower()) + r"(?:'s)?(?![a-z0-9])"
        problem = None
        if not re.search(pattern, notes.lower()):
            problem = f"the go_to seller '{seller}' is not written in {target}'s notes."
        elif not any(re.search(pattern, sentence.lower()) and self.names_property(ref["prop"], sentence)
                     for sentence in re.split(r"(?<=[.!?])\s+", notes)):
            problem = f"the note naming {seller} is about another property, not {ref['prop']}."
        if problem:
            return "answer", problem, None, "", []
        kept = [detail for detail in self.clean_strings(go.get("details"))
                if set(ref_toks(detail)) and 2 * len(set(ref_toks(detail)) & note_tokens) >= len(set(ref_toks(detail)))]
        onward = {"seller": seller, "kind": str(go.get("kind") or "").strip()[:80], "details": kept[:6]}
        if ref_name(seller) in self.sellers_of(row) and self.target_key(onward) == self.target_key(ref["target"]):
            return "answer", f"the go_to is {target} itself.", None, "", []
        return "", "", onward, "", []

    @staticmethod
    def clean_strings(raw):
        if not isinstance(raw, list):
            return []
        return [str(x).strip() for x in raw if isinstance(x, (str, int, float)) and not isinstance(x, bool) and str(x).strip()]

    @staticmethod
    def target_key(target):
        tokens = set()
        for detail in target.get("details") or []:
            tokens |= set(ref_toks(detail))
        return ref_name(target.get("seller") or ""), tuple(sorted(tokens))

    @staticmethod
    def _pairs_in_notes(paired, value_tokens, note_tokens):
        tokens = set(ref_toks(paired))
        return bool(tokens) and tokens != value_tokens and tokens <= note_tokens

    def reject_hop(self, ref, pick, target, where, problem):
        self.ep.log.append({"kind": "ref_invalid", "prop": ref["prop"], "target": target, "problem": problem[:300]})
        if where == "target":
            ref["judged"].add(target)
            return
        ref["judged"].update(pick)
        ref["chain"].append(target)
        self.give_up(ref, problem)

    def ask_hop(self, ref, pick, reason):
        if not self.hop_budget_ok():
            return None
        self.hop_calls += 1
        prompt = REF_HOP.replace("PROPERTY", ref["prop"]).replace("DESCRIPTION", self.describe(ref))
        prompt += "\n\nRECORDS:\n" + "\n".join(self.record_card(i) for i in pick)
        if reason:
            prompt += "\n\nTHE PREVIOUS ANSWER WAS NOT USABLE: " + reason + " Answer again from the records."
        answer = self.ep.counsel.ask_json(prompt, 3000, CHECKER[:1])
        self.ep.log.append({"kind": "ref_hop", "prop": ref["prop"], "ids": pick, "answer": answer})
        return answer

    def describe(self, ref):
        return self.describe_target(ref["target"])

    @staticmethod
    def describe_target(target):
        details = target.get("details")
        text = f"the {target.get('kind') or 'listing'} sold by {target.get('seller')}"
        return text + (" (" + "; ".join(target.get("details") or []) + ")" if details else "")

    def record_card(self, ident):
        listing = self.listing_index().get(ident) or {}
        facts = {}
        for inner in (value for value in listing.values() if isinstance(value, dict)):
            for key, fact in list(inner.items())[:40]:
                if isinstance(fact, (str, int, float)) and not isinstance(fact, bool):
                    facts[str(key)] = str(fact)[:90]
        sellers = sorted(self.seller_values(listing))
        card = dict(id=ident)
        card["sold by"] = " / ".join(sellers)
        card.update(category=self.ep.lookout.shelf_of(listing), title=self.ep.shop.title_of(listing)[:160], facts=facts,
                    notes=self.note_text(listing)[:900])
        return as_text(card, 5000)

    def seller_values(self, listing):
        learned = self.seller_fields()
        found = set()
        for key, value in listing.items():
            lowered = str(key).lower()
            if not isinstance(value, str) or not value.strip() or len(value) > 60 or "id" in lowered:
                continue
            if re.search(r"note|text|desc|comment|detail|info|message|section", lowered):
                continue
            if re.search(SELLER_KEYS, lowered) or str(key) in learned:
                found.add(str(value))
        return found

    def follow(self, ref, target, go):
        ref["chain"].append(target)
        key = self.target_key(go)
        looped = key in ref["seen"]
        if looped or ref["hops"] + 1 >= REF_HOPS:
            reason = ("its notes lead back to a listing already followed" if looped
                      else f"more than {REF_HOPS} listings in a row send the reader elsewhere")
            self.give_up(ref, reason)
            return
        ref["seen"].add(key)
        ref["hops"] += 1
        ref.update(target=go, judged=set(), queries=[], pages=0, tries=0, fails=0, unindexed=False, loose=set())
        self.ep.log.append({"kind": "ref_go", "prop": ref["prop"], "from": target, "to": go})
        self.tell_actor(f"{target} is the listing given for {ref['head']}; its notes pass the reader on to {self.describe(ref)}: the "
                        "code follows them.")

    def more_to_read(self, ref):
        if not self.ref_room() and not self.waits_for_others():
            return False
        read = self.known.read
        if any(i not in read and i not in ref["judged"] for i in self.candidates(ref)):
            return True
        return bool(self.planned_queries(ref) or self.next_seller_page(ref))

    def next_seller_page(self, ref):
        lookout = self.ep.lookout
        paging = lookout.page_arg()
        seller = ref_name(ref["target"]["seller"])
        if not paging or ref["pages"] >= 3 or not seller:
            return None
        books = list(self.ep.page_book.values())
        while books:
            entry = books.pop()
            if seller in ref_name(" ".join(lookout.free_text(entry["args"]))):
                after = lookout.next_page(entry)
                if after is not None:
                    return dict(entry["args"], **{paging[0]: after})
        return None

    def notes_unread(self, ref, ids):
        tool = self.notes_reader()
        if not tool or not ids:
            return []
        done = set((self.ep.shop.read_with or {}).get(tool) or set())
        tried = ref.setdefault("noted", set())
        return [i for i in dict.fromkeys(ids) if i not in done and i not in tried][:2]

    def search_or_give_up(self, needy, pending, template, acted):
        for ref in needy or sorted(pending, key=lambda entry: (entry["hops"], len(entry["queries"]))):
            plans = self.planned_queries(ref)
            if not plans:
                page = self.next_seller_page(ref)
                if not page:
                    continue
                ref["pages"] += 1
                return self.search_target(ref, page) or acted
            ref["queries"].append(plans[0].lower())
            if self.ep.bargain.ran_query(plans[0]):
                return True
            args = dict(template[0])
            args[template[1]] = plans[0]
            return self.search_target(ref, args) or acted
        exhausted = [ref for ref in pending
                     if not any(ident not in ref["judged"] for ident in self.candidates(ref))
                     and not self.planned_queries(ref) and not self.next_seller_page(ref)]
        for ref in exhausted:
            self.give_up(ref, "the listing it points at was not found in the store's search")
        return acted or bool(exhausted)

    def search_target(self, ref, args):
        ep = self.ep
        tick_before = ep.shop.tick
        found = ep.bargain.walk_call(args, f"The request copies {ref['head']} from a different listing ({self.describe(ref)}): searching for it.")
        self.steps_spent += ep.shop.tick - tick_before
        rows = self.newest_rows() if found else []
        seller = ref_name(ref["target"]["seller"])
        text = " ".join(ep.lookout.free_text(args))
        if rows and not any(seller in self.sellers_of(row) for row in rows) and ref_name(text) == seller:
            ref["unindexed"] = True
        ep.log.append({"kind": "ref_search", "prop": ref["prop"], "query": text[:120], "rows": len(rows)})
        return ep.shop.tick > tick_before

    def newest_rows(self):
        mark, ids = self.ep.lookout.row_mark, self.known.ids
        rows = []

        def keep(fields):
            if len(fields) >= 4 and mark(fields) in ids:
                rows.append(fields)
        walk_dicts(self.known.items[-1], keep, ceiling=3)
        return rows

    def read_candidates(self, queue, pending):
        ep = self.ep
        batch = None if len(queue) < 2 else ep.shop.batch_tool()
        tick_before = ep.shop.tick
        heads = ", ".join(dict.fromkeys(ref["head"] for ref in pending))
        replies = ep.counsel.open_reads(queue, f"The request copies {heads} from a different listing: I read the ones that may be it.", batch)
        kept = True
        if batch and replies and not ep.batch_proven:
            kept = self.settle_batch(batch, queue)
        if replies:
            ep.talk.post_replies(replies)
        self.steps_spent += ep.shop.tick - tick_before
        self.remember_ids(queue)
        ep.log.append({"kind": "ref_read", "ids": queue})
        if kept:
            unread = [ident for ident in queue if ident not in self.known.read]
            for ref in pending:
                ref["judged"].update(unread)
        return ep.shop.tick > tick_before

    def settle_batch(self, batch, queue):
        ep = self.ep
        if ep.shop.batch_ok(ep.counsel.last_reads or [], queue):
            ep.batch_proven = True
            return True
        ep.batch_spec, ep.batch_spec_known = None, True
        ep.log.append({"kind": "batch_dropped", "tool": batch[0]})
        return False

    def ref_take(self):
        ep = self.ep
        try:
            for ref in ep.refs:
                if ref["status"] != "resolved":
                    continue
                left = self.wants.props_left()
                if any(req != ref["now"] and self.ref_match(ref, req) for req in left):
                    self.swap_in(ref, ref["now"])
            news, self.wants.ref_news = list(self.wants.ref_news or []), []
            for item in news:
                opened = self.ref_add(str(item.get("need") or ""), list(item.get("infos") or []), str(item.get("line") or ""), "reply")
                if opened:
                    self.note_reference(opened["req"])
        except Exception as err:
            ep.log.append({"kind": "ref_take_error", "error": f"{type(err).__name__}: {err}"[:300]})

    def ref_add(self, req, infos, source, origin):
        line = str(req)
        head, _, value = line.partition(":")
        prop = self.bare_prop(head).strip()
        tags = re.findall(r"\(([^)]*)\)", head)
        tag = tags[-1].strip() if tags else ""
        if not prop or junk_key(prop) or self.opened_before(req, prop, tag):
            return None
        info = self.matching_info(infos, prop, tag)
        target = self.find_target(info, value, source, prop)
        if target:
            return self._open_ref(line, head, prop, tag, value, info, source, target, origin)
        self.ep.log.append({"kind": "ref_skip", "req": line[:200]})
        return None

    def matching_info(self, infos, prop, tag):
        for info in infos:
            if same_prop(str(info.get("property") or ""), prop):
                product = info.get("product")
                if not tag or not product or not self.wants.items() or kind_words(product) & kind_words(tag):
                    return info
        return None

    def find_target(self, info, value, source, prop):
        seller, kind, details = "", "", []
        if isinstance(info, dict):
            seller = str(info.get("seller") or "").strip()
            kind = str(info.get("kind") or "").strip()
            details = self.clean_strings(info.get("details"))
        approve = lambda name: self.ref_seller_ok(name, source)
        parsed = next((guess for guess in map(lambda text: ref_parse(text, approve), (ref_span(prop, source), value))
                       if guess and approve(guess["seller"])), None)
        if approve(seller):
            if parsed and not details and ref_name(parsed["seller"]) == ref_name(seller):
                details = parsed["details"]
        elif parsed is None:
            return None
        else:
            seller, kind, details = parsed["seller"], kind or parsed["kind"], details or parsed["details"]
        source_tokens = set(ref_toks(source))
        kept = [detail for detail in dict.fromkeys(details) if source_tokens.intersection(ref_toks(detail))][:6]
        label = (kind or parsed["kind"]) if parsed else kind
        stated = self.clean_strings(info.get("marks")) if isinstance(info, dict) and approve(seller) and ref_name(info.get("seller")) == ref_name(seller) else []
        marks = [mark for mark in dict.fromkeys(stated) if set(ref_toks(mark)) and set(ref_toks(mark)) <= source_tokens][:8]
        return dict(seller=seller, kind=label[:80], details=kept, marks=marks)

    def ref_seller_ok(self, seller, source):
        name = str(seller or "").strip()
        if len(re.sub(r"[^A-Za-z]", "", name)) < 3:
            return False
        written = re.search(r"(?<![a-z0-9])" + re.escape(name.lower()) + r"(?:'s)?(?![a-z0-9])", str(source or "").lower())
        if written is None:
            return False
        kinds = self.asked_kinds()
        shape = self.ep.shapes.get(self.ep.role("search") or "") or {}
        enums = [spec["enum"] for spec in (shape.get("properties") or {}).values()
                 if isinstance(spec, dict) and isinstance(spec.get("enum"), list)]
        for options in enums:
            for option in options:
                kinds |= kind_words(option)
        own = kind_words(name)
        return bool(own) and not own.issubset(kinds)

    def asked_kinds(self):
        entries = self.wants.items() or [{"product": str(self.wants.reading.get("product") or "")}]
        found = set()
        for entry in entries:
            found |= kind_words(entry["product"])
        return found

    def opened_before(self, req, prop, tag):
        return any(ref["req"] == req or (same_prop(ref["prop"], prop) and kind_words(ref["tag"]) == kind_words(tag))
                   for ref in self.ep.refs)

    def _open_ref(self, line, head, prop, tag, value, info, source, target, origin):
        relation, tolerance, unit = self.read_relation(value, info, prop, source, target)
        fresh = {"req": line, "head": head.strip(), "prop": prop, "tag": tag, "relation": relation, "tol": tolerance,
                 "unit": unit, "target": target, "first": dict(target), "chain": [], "seen": {self.target_key(target)},
                 "judged": set(), "queries": [], "pages": 0, "hops": 0, "tries": 0, "fails": 0, "status": "open",
                 "value": "", "goes_with": [], "now": "", "origin": origin, "unindexed": False}
        self.ep.refs.append(fresh)
        self.ep.log.append({"kind": "ref_open", "req": line[:200], "target": target, "relation": relation, "origin": origin})
        return fresh

    def read_relation(self, value, info, prop, source, target):
        relation, tolerance, unit = ref_relation(value)
        stated = str(info.get("relation") or "").lower() if isinstance(info, dict) else ""
        if relation == "same" and stated == "within" and isinstance(info, dict) and is_number(info.get("tolerance")):
            return "within", float(info["tolerance"]), str(info.get("unit") or "")
        if relation != "same":
            return relation, tolerance, unit
        if re.search(r"differ|not the same|exclu|apart|other than|unlike", stated):
            return "differ", None, ""
        stretch = ref_span(prop, source).lower()
        seller = str(target.get("seller") or "")
        at = stretch.find(seller.lower()) if seller.strip() else -1
        if at > 0 and re.search(DIFFER_WORDS, stretch[:at]):
            return "differ", None, ""
        return ("pairs", None, "") if stated == "pairs" else (relation, tolerance, unit)

    def note_reference(self, req):
        reading = self.wants.reading
        listed = reading.get("references")
        if not isinstance(listed, list):
            listed = []
            reading["references"] = listed
        if req not in listed:
            listed.append(req)

    def ref_register(self):
        reading = self.wants.reading
        targets = reading.get("reference_targets")
        infos = [t for t in targets if isinstance(t, dict)] if isinstance(targets, list) else []
        listed_raw = reading.get("references")
        listed = set()
        if isinstance(listed_raw, list):
            listed = {str(r) for r in listed_raw or []}
        for written in list(self.wants.written):
            line = str(written)
            value = line.partition(":")[2]
            if ref_text(value) or (line in listed and re.search(DIFFER_WORDS, value.lower())):
                self.ref_add(line, infos, self.ep.request, "request")

    def ref_fix(self, answer, lines, qtext):
        masked = []
        for line in lines:
            located = self.pointing_phrase(line)
            if located:
                piece = PointerLine(line, *located)
                masked.append(piece.outside)
                self.place_pointing_need(answer, piece, qtext)
            else:
                masked.append(line)
        return masked

    def pointing_phrase(self, line):
        text = str(line or "")
        if not ref_text(text):
            return None
        parsed = ref_parse(text, lambda name: self.ref_seller_ok(name, text))
        if not parsed or not self.ref_seller_ok(parsed["seller"], text):
            return None
        seller_hit = re.search(r"(?<![A-Za-z'])" + re.escape(parsed["seller"]) + r"(?:'s)?(?![A-Za-z])", text)
        if seller_hit is None:
            return None
        start = self.phrase_start(text[:seller_hit.start()])
        if start is None:
            return None
        end = seller_hit.end() + self.phrase_length(text[seller_hit.end():])
        phrase = text[start:end].strip(" ,;")
        return (phrase, start, end, parsed) if phrase else None

    @staticmethod
    def phrase_start(before):
        breaks = [before.rfind(mark) for mark in ",;!?"]
        breaks.extend(hit.start() for hit in re.finditer(r"[.](?:\s|$)", before))
        cut = max(breaks + [-1])
        clause = before[cut + 1:].lower()
        marks = [hit.start() for pattern in (DIFFER_WORDS, REF_WORDS, POINT_WORDS) for hit in re.finditer(pattern, clause)]
        return cut + 1 + min(marks) if marks else None

    @staticmethod
    def phrase_length(tail):
        aside = re.search(r"\(\s*or\b[^)]*\)", tail)
        stop = re.search(r"[.!?;](?:\s|$)", tail)
        if aside is not None and (stop is None or aside.start() < stop.start()):
            return aside.end()
        return len(tail) if stop is None else stop.start()

    def place_pointing_need(self, answer, piece, qtext):
        needs, refs = self.answer_lists(answer)
        prop = self.line_property(piece, needs, refs, qtext)
        if not prop:
            self.ep.log.append({"kind": "ref_line_unplaced", "line": piece.text[:200]})
            return
        several = self.wants.items()
        tag = self.line_tag(piece, needs, prop, qtext, several) if several else ""
        same_item = self.item_filter(several, tag)
        mine = list(filter(lambda need: same_prop(self.need_name(need), prop) and same_item(need), needs))
        relation, tolerance, unit = ref_relation(piece.phrase)
        seller = ref_name(piece.parsed["seller"])
        keep = self.kept_pointing(mine, relation, seller)
        need = keep or self.need_head(mine, prop, tag, several) + ": " + piece.phrase
        details = [line for line in needs if line not in mine and piece.describes_other(line, self.need_name)]
        position = len(needs) if not mine else needs.index(mine[0])
        others = [line for line in needs if line not in mine and line not in details]
        others.insert(position, need)
        answer["needs"] = others
        self.trim_answer(answer, piece, prop, need, details, same_item)
        self.merge_entry(refs, prop, piece, relation, tolerance, unit, seller)
        answer["references"] = refs
        answer["ref_needs"] = list(answer.get("ref_needs") or []) + [need]
        self.ep.log.append(dict(kind="ref_line", prop=prop, tag=tag, relation=relation, need=need[:200], kept=bool(keep),
                                details_out=[line[:80] for line in details]))

    @staticmethod
    def answer_lists(answer):
        needs, refs = [], []
        listed, given = answer.get("needs"), answer.get("references")
        if isinstance(listed, list):
            needs = [line for line in listed or [] if isinstance(line, str) and ":" in line]
        if isinstance(given, list):
            refs = [entry for entry in given or [] if isinstance(entry, dict)]
        return needs, refs

    def line_property(self, piece, needs, refs, qtext):
        name = self.need_name
        outside = piece.outside
        asked = list(filter(None, map(name, str(self.ep.talk.asking_label or "").split(";"))))
        pointing = [name(need) for need in needs
                    if ref_text(need.partition(":")[2]) or re.search(DIFFER_WORDS, need.partition(":")[2].lower())]
        stated = [str(entry.get("property") or "") for entry in refs]
        for tier in self.property_tiers(stated, pointing, needs, asked, qtext, outside):
            named = [prop for prop in tier if prop and prop_named(prop, outside)]
            if named:
                return max(named, key=lambda prop: (prop_named(prop, outside, exact=True),
                                                    len(re.findall(r"[a-z0-9]+", prop.lower())), len(prop)))
        backup = stated + pointing + (asked if len(asked) == 1 else [])
        return next((prop for prop in backup if prop and not piece.inside_only(prop)), "")

    def need_name(self, req):
        return self.bare_prop(str(req).partition(":")[0]).strip()

    def property_tiers(self, stated, pointing, needs, asked, qtext, outside):
        name = self.need_name
        yield stated
        yield pointing
        yield [name(need) for need in needs]
        yield asked
        yield sorted(self.field_names()) + [name(req) for req in self.wants.written]
        yield self.question_words(qtext, outside)

    def question_words(self, qtext, outside):
        if not qtext:
            return []
        entries = self.wants.items() or [{"product": str(self.wants.reading.get("product") or "")}]
        product_words = set()
        for entry in entries:
            product_words |= set(re.findall(r"[a-z]+", str(entry["product"]).lower()))
        tokens = re.findall(r"[a-z0-9]+", str(qtext).lower())
        bad = lambda word: word in LINK_WORDS or len(word) < 3 or word in product_words or word.rstrip("s") in product_words
        found = []
        for size in (3, 2, 1):
            for at in range(len(tokens) - size + 1):
                window = tokens[at: at + size]
                if any(map(bad, window)):
                    continue
                phrase = " ".join(window)
                if prop_named(phrase, outside):
                    found.append(phrase)
        return found

    @staticmethod
    def item_filter(several, tag):
        def same_item(req):
            tags = re.findall(r"\(([^)]*)\)", str(req).partition(":")[0])
            return not (several and tag and tags) or bool(kind_words(tags[-1]) & kind_words(tag))
        return same_item

    def kept_pointing(self, mine, relation, seller):
        for need in mine:
            value = need.partition(":")[2]
            pointing = ref_text(value) or re.search(DIFFER_WORDS, value.lower())
            if pointing and ref_relation(value)[0] == relation and seller in ref_name(value):
                return need
        return None

    def trim_answer(self, answer, piece, prop, need, details, same_item):
        name = self.need_name
        current = answer.get("requirements_now")
        if isinstance(current, list):
            written = self.wants.written

            def stays(req):
                if req in written:
                    return True
                if req in details or piece.describes_other(req, name):
                    return False
                return not (same_prop(name(req), prop) and same_item(req))
            kept = [req for req in current if isinstance(req, str) and stays(req)]
            answer["requirements_now"] = kept if need in kept else kept + [need]
        for field, clash in (("dropped", lambda text: piece.describes_other(text, name)),
                             ("no_preference", lambda text: same_prop(name(text), prop))):
            listed = answer.get(field)
            if isinstance(listed, list):
                answer[field] = [text for text in listed
                                 if not isinstance(text, str) or not (piece.inside_only(name(text)) or clash(text))]

    def merge_entry(self, refs, prop, piece, relation, tolerance, unit, seller):
        parsed = piece.parsed
        entry = next((ref for ref in refs if same_prop(str(ref.get("property") or ""), prop)), None)
        if entry is None:
            entry = {"property": prop}
            refs.append(entry)
        trusted = self.ref_seller_ok(entry.get("seller"), piece.text) and ref_name(entry.get("seller")) == seller
        if not trusted:
            entry["seller"], entry["kind"], entry["details"] = parsed["seller"], parsed["kind"], list(parsed["details"])
        elif not entry.get("details"):
            entry["details"] = list(parsed["details"])
        entry.setdefault("kind", parsed["kind"])
        if not entry.get("relation") or relation != "same":
            entry["relation"], entry["tolerance"], entry["unit"] = relation, tolerance, unit

    def line_tag(self, piece, needs, prop, qtext, several):
        product_of = self.ep.talk.product_of
        k = product_of(piece.outside)
        if k is None:
            tags = []
            for need in needs:
                if same_prop(self.need_name(need), prop):
                    tags.extend(re.findall(r"\(([^)]*)\)", need.partition(":")[0]))
            if tags:
                k = product_of(f"x ({tags[-1]})")
        if k is None and qtext:
            k = product_of(qtext)
        return "" if k is None else several[k]["product"]

    @staticmethod
    def need_head(mine, prop, tag, several):
        if mine:
            head = mine[0].partition(":")[0].strip()
        elif tag:
            head = f"{prop} ({tag})"
        else:
            head = prop
        if several and tag and not re.findall(r"\(([^)]*)\)", head):
            head = f"{head} ({tag})"
        return head

    def ref_line(self):
        try:
            parts = list(filter(None, map(self.status_part, self.ep.refs or [])))
            told, self.pending_tells = self.pending_tells, None
            text = "\n[references] " + "; ".join(parts) if parts else ""
            return text + "".join("\n[reference] " + line for line in told or [])
        except Exception:
            return ""

    def status_part(self, ref):
        status = ref["status"]
        if status == "resolved":
            return f"{ref['now']} (read on {ref['chain'][-1]}" + (", must differ from it" if ref.get("relation") == "differ" else "") + ")"
        if status == "open":
            return f"{ref['head']}: the code is following it (now: {self.describe(ref)})"
        if status == "stuck":
            return f"{ref['head']}: not resolved, value unknown (never decline or reject a listing over it)"
        return ""

    def ref_note(self, thing=""):
        try:
            lines, ends = [], []
            for ref in self.ep.refs or []:
                if ref["status"] == "resolved":
                    lines.append(self.resolved_line(ref))
                elif ref["status"] in ("open", "stuck"):
                    lines.append(self.unresolved_line(ref))
                    ends += [ident for ident in ref["chain"][-2:] if ident != thing]
            if not lines:
                return ""
            text = ("\n\nREQUIREMENTS TAKEN FROM ANOTHER LISTING'S VALUE (the code traced that listing's notes through the "
                    "store's records):\n" + "\n".join(lines))
            records = [card for ident in dict.fromkeys(reversed(ends)) for card in self.known.about(ident, 1)]
            if not records:
                return text
            return text + "\nRECORDS OF THE LAST LISTINGS ITS CHAIN REACHED:\n" + "\n".join(records[:2])
        except Exception:
            return ""

    @staticmethod
    def resolved_line(ref):
        chain = ref["chain"]
        via = (f" (reached from {chain[0]} through the listings' notes; each page before {chain[-1]} sends the reader on for "
               f"this property, so a value in that page's own fields does not count)") if len(chain) > 1 else ""
        partners = ref["goes_with"]
        joined = " and ".join(partners)
        if ref.get("relation") == "differ":
            pair = f"; its notes pair {joined} with it" if partners else ""
            return (f"- {ref['now']}: the shopper's {ref['prop']} has to DIFFER from what {chain[-1]} shows "
                    f"({ref['value']}){via}{pair}: each different value meets it, that value does not; its chain's listings ({', '.join(chain)}) carry it.")
        if partners and ref.get("relation") == "pairs":
            return (f"- {ref['now']}: a value that goes with what {chain[-1]} shows ({ref['value']}){via}; its notes pair "
                    f"{ref['value']} with {joined}, so {joined} meets it and {ref['value']} itself does not.")
        pair = f"; its notes say {ref['value']} goes together with {joined}, so each of them meets it" if partners else ""
        return f"- {ref['now']}: the value shown on {chain[-1]}{via}{pair}."

    @staticmethod
    def unresolved_line(ref):
        chain = ref["chain"]
        where = "its chain so far: " + " -> ".join(chain) if chain else "the listing it points at was not found"
        return (f"- {ref['head']}: the code could NOT RESOLVE it (" + where + "). Should a record below be that listing, or the final one the chain got to, and state its own value "
                                                                              "for it without a further pointer, that value is the requirement; else the value is still unknown, so "
                                                                              "never fail an item on it alone.")

    def ref_pointers(self):
        pointed = set()
        for ref in self.ep.refs or []:
            if ref.get("relation") == "differ" and ref.get("status") in ("open", "stuck", "resolved") and self.ref_live(ref):
                pointed.update(i for i in ref.get("chain") or [] if self.same_product(ref, i))
        return pointed

    def same_product(self, ref, ident):
        if not self.wants.items():
            return True
        shelf = self.ep.counsel.shelf_known(ident)
        product_of = self.ep.talk.product_of
        own = product_of(f"x ({shelf})") if shelf else None
        if own is not None and ref.get("tag"):
            return own == product_of(f"x ({ref['tag']})")
        return True

    def ref_pointer_note(self, thing):
        for ref in self.ep.refs or []:
            if ref.get("relation") != "differ" or thing not in (ref.get("chain") or []):
                continue
            if self.ref_live(ref) and self.same_product(ref, thing):
                what = f" ({ref['value']})" if ref.get("status") == "resolved" and ref.get("value") else ""
                return (f"{thing} is where the shopper's reference for {ref['head']} leads, or a listing its notes go through, hence its {ref['prop']} is the very value{what} the shopper's {ref['prop']} must differ from: it cannot "
                        "meet that requirement")
        return ""

    def ref_excluded(self, fields, k):
        several = self.wants.items()
        for ref in self.ep.refs or []:
            if not (ref.get("relation") == "differ" and ref.get("status") == "resolved" and self.ref_live(ref)):
                continue
            if several and ref.get("tag") and self.ep.talk.product_of(f"x ({ref['tag']})") not in (k, None):
                continue
            barred = set()
            for value in [ref.get("value")] + list(ref.get("goes_with") or []):
                if str(value or "").strip():
                    barred.add(" ".join(ref_toks(value)))
            if any(" ".join(ref_toks(value)) in barred for key, value in (fields or {}).items() if same_prop(key, ref["prop"])):
                return True
        return False

    def ref_foreign(self):
        try:
            ids = self.candidate_ids or set()
            shelves = self.ep.shop.home_shelves() if ids else None
            if not ids or not shelves or not all(shelves):
                return set()
            homes = set().union(*shelves)
            listings = self.listing_index()
            held = set(self.ep.basket.units)
            shelf_of = self.ep.lookout.shelf_of
            away = set()
            for ident in ids:
                if ident in held:
                    continue
                label = shelf_of(listings.get(ident) or {})
                if label and label not in homes:
                    away.add(ident)
            return away
        except Exception:
            return set()

    def ref_untracked(self, refs, tracked):
        def chased(req):
            return any(self.ref_match(ref, req) and (ref.get("status") == "resolved" or ref.get("chain"))
                       for ref in tracked)
        try:
            return [req for req in refs if not chased(req)]
        except Exception:
            return refs

    def notes_remark(self, call_name, call_args):
        ep = self.ep
        if call_name in ep.finishers() or call_name in (ep.role("search"), ep.role("cart"), ep.role("cart_view"), ep.role("message")):
            return ""
        version = str(self.wants.version)
        held = ep.basket.units
        remarks = []
        for ident in sorted(ep.shop.asked_ids(call_args)):
            key = "|".join((str(ident), version))
            if ident not in held and key not in ep.passed:
                continue
            found = self.own_redirects(ident) if key not in self.redirect_told else []
            if found:
                self.redirect_told.add(key)
                ep.passed.discard(key)
                ep.log.append({"kind": "own_redirect", "id": ident, "props": [prop for _, prop, _ in found]})
                remarks.append(self.redirect_text(ident, found, ident in ep.basket.units))
        return " ".join(remarks)

    def own_redirects(self, thing):
        try:
            listings = self.listing_index()
            listing = listings.get(thing) or {}
            notes = self.note_text(listing, desc=False)
            if not notes:
                return []
            own = self.sellers_of(listing)
            elsewhere = set()
            for other in listings.values():
                elsewhere |= self.sellers_of(other)
            elsewhere -= own
            fields = self.field_tokens(listing, self.token_parts(listing)[0])
            sentences = re.split(r"(?<=[.!?;])\s+", notes)
            return list(self.redirected_needs(listing, sentences, own, elsewhere, fields))
        except Exception:
            return []

    def redirected_needs(self, listing, sentences, own, elsewhere, fields):
        several = self.wants.items()
        product_of = self.ep.talk.product_of
        shelf = self.ep.lookout.shelf_of(listing)
        k = product_of(f"x ({shelf})") if several and shelf else None
        for req in self.wants.props_left():
            head = str(req).partition(":")[0]
            if several and k is not None and product_of(head) not in (k, None):
                continue
            prop = self.bare_prop(head).strip()
            if not prop:
                continue
            hit = next((sentence for sentence in sentences if self.sends_elsewhere(sentence, prop, own, elsewhere, fields)), None)
            if hit is not None:
                yield str(req), prop, hit.strip()

    def sends_elsewhere(self, sentence, prop, own, elsewhere, fields):
        words = list(re.finditer(r"[A-Za-z][A-Za-z'-]*", sentence))
        masked = re.sub(r"[A-Za-z0-9]+(?=\s+by\s+[A-Z])", lambda hit: " " * len(hit.group(0)), sentence)
        named_at = next((match.start() for match in words if prop_named(prop, masked[:match.end()])), None)
        if named_at is None:
            return False
        negators = ("not", "nor", "never", "no", "neither")
        for position, match in enumerate(words[1:], 1):
            word = match.group(0)
            lowered = ref_name(word)
            if not lowered or match.start() <= named_at:
                continue
            stranger = word[:1].isupper() and len(lowered) >= 4 and lowered not in own \
                and not set(ref_toks(word)) & fields and lowered not in fields
            if lowered not in elsewhere and not stranger:
                continue
            if all(w.group(0).lower() not in negators for w in words[max(0, position - 5): position]):
                return True
        return False

    @staticmethod
    def redirect_text(ident, found, in_cart):
        what = "; ".join(f"{prop} (\"{sentence[:160]}\")" for _, prop, sentence in found)
        quoted = "; ".join(f"'{req}'" for req, _, _ in found)
        ending = ". If so, get it right out of the cart and put another verified fit in." if in_cart else "."
        return (f"{ident}'s notes may point {what} at a different listing: its own field then fails to count for "
                + quoted + " (grading takes the figure on that other listing), so it passes "
                + ("them" if len(found) > 1 else "it") + " just once the other listing's record says so; before the order, the item check runs anew" + ending)


class Scout:
    def __init__(self, ep):
        self.ep = ep
        self.all_hits = []
        self.hit_score = {}
        self.row_score = {}
        self.aimed_at = {}
        self.put_aims = {}
        self.broken_modes = set()
        self.price_mode = None
        self.seeded_products = set()
        self.totals_seen = {}
        self.walked_products = set()
        self.back_hits_done = None
        self.settled_values = {}
        self.label_indexed = None
        self.composed_sets = {}
        self.label_attempts = {}
        self.followed = 0
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
        self.walk_named = set()

    def search_note(self, call_args, obs):
        page, rows = self.unwrap_rows(obs) if isinstance(obs, dict) else (None, None)
        if page is None:
            return ""
        lookout = self.ep.lookout
        totals = [count for count in page.values() if isinstance(count, int) and not isinstance(count, bool) and count > len(rows)]
        biggest = max(totals or [len(rows)])
        walk = lookout.book_page(call_args, rows, biggest, page)
        self.ep.guarded("ceil_learn_error", None, self.learn_filter, call_args, rows, biggest, page)
        self.score_rows(call_args, rows)
        kind = lookout.word_rule(call_args, page, rows)
        lead = lookout.lead_rule(call_args, page, rows)
        remarks = [self.lead_remark(call_args, lead)] if lead else []
        texts, mode, first, paging_keys = lookout.search_keys(call_args)
        fixed = set(dict(mode)) | set(paging_keys)
        loose = [value for name, value in call_args.items() if isinstance(value, str) and name not in fixed]
        if loose and first:
            self.query_remarks(call_args, rows, loose, totals, mode, paging_keys, texts, kind, remarks)
        if rows:
            remarks.extend([self.unseen_remark(kind, totals, rows)] if totals else [])
            remarks.append(self.over_limit_remark(call_args, rows))
        remarks.append(self.closed_walk_remark(walk))
        return " ".join(filter(None, remarks))

    def closed_walk_remark(self, walk):
        ep = self.ep
        if not walk or not walk.pop("closed", None):
            return ""
        values = [need.partition(":")[2] for need in ep.wants.props_left()]
        wordy = [value for value in values if not ref_text(value) and re.search(r"[a-z0-9]{3,}", value.lower())]
        score = ep.lookout.value_score(walk["args"])
        if not wordy or score < len(wordy):
            return ""
        hits = ep.lookout.row_hits(walk["every"])
        self.score_hits(hits, score)
        if not hits or ep.moves_left() <= ep.brief.order_due() + 3:
            return ""
        price = lambda ident: ep.known.price(ident) if is_number(ep.known.price(ident)) else float("inf")
        hits = sorted([ident for ident in hits if ident not in self.walk_named], key=price)[:6]
        if not hits:
            return ""
        self.walk_named.update(hits)
        return (f"This search holds every required value and its last page is now read: all {len(walk['every'])} matches are listed. "
                "Not opened yet, and not ruled out by price, stock or category: " + ", ".join(hits) + " (cheapest first). "
                "A row's title seldom shows its fields, so open these (several per call) before searching further.")

    def learn_filter(self, call_args, rows, total, obs):
        lookout = self.ep.lookout
        paging = lookout.page_arg()
        page_key = paging[0] if paging else None
        plain_args = dict((name, value) for name, value in call_args.items() if name != page_key)
        self.totals_seen[json.dumps(plain_args, sort_keys=True, default=str)] = int(total)
        listed = list(filter(lambda row: isinstance(row, dict), rows))
        if listed and paging and paging[1] is True and isinstance(obs, dict):
            self.note_depth_cap(call_args, paging, listed, total, obs)
        keys_found = lookout.search_keys(call_args)
        mode, paging_keys = keys_found[1], keys_found[3]
        bound_key = self.bound_argument(call_args, page_key, paging_keys)
        if bound_key is not None and listed:
            self.adopt_price_mode(call_args, plain_args, listed, page_key, mode, bound_key)

    @staticmethod
    def bound_argument(call_args, page_key, paging_keys):
        found = [k for k, v in call_args.items()
                 if is_number(v) and not isinstance(v, bool) and k != page_key and k not in paging_keys and v > 1
                 and not re.search(r"page|size|count|result|min|low|from|start|offset", str(k).lower())]
        return found[0] if len(found) == 1 else None

    def note_depth_cap(self, call_args, paging, dicts, total, obs):
        page_at = call_args.get(paging[0], paging[2])
        if not is_number(page_at):
            return
        markers = [value for key, value in obs.items()
                   if re.search(r"next|more|onward|continu|cursor", str(key).lower()) and not isinstance(value, (list, dict))]
        width = max(len(dicts), self.widest_page())
        if not markers or len(dicts) < width:
            return
        quiet = [value for value in markers if value is None or value is False or value == ""]
        if len(quiet) != len(markers):
            return
        reach = int((page_at - paging[2] + 1) * width)
        if total > reach:
            self.depth_cap = min(self.depth_cap or reach, reach)

    def widest_page(self):
        return max([entry.get("size", 0) for entry in self.ep.page_book.values()] or [0])

    def adopt_price_mode(self, call_args, plain_args, listed, page_key, mode, bound_key):
        ceiling = float(call_args[bound_key])
        labels = set(map(self.ep.lookout.shelf_of, listed))
        matching = [name for name, value in call_args.items()
                    if isinstance(value, str) and name != page_key and value.strip() and value.strip().lower() in labels]
        shelf_key = None if len(matching) != 1 else matching[0]
        fixed = dict((name, value) for name, value in mode if name != shelf_key)
        signature = json.dumps([sorted(fixed.items()), bound_key], default=str)
        prices = list(map(self.row_price, listed))
        if any(price is not None and price > ceiling + 1e-9 for price in prices):
            self.broken_modes.add(signature)
            return
        steady = not (signature in self.broken_modes or shelf_key is None)
        if not steady or len(labels) != 1 or "" in labels or None in prices:
            return
        template = self.without_shared_words(plain_args, listed, bound_key, shelf_key, fixed)
        word_count = 0
        for name, value in template.items():
            if isinstance(value, str) and name not in fixed:
                word_count += len(value.split())
        shape = self.ep.shapes.get(self.ep.role("search") or "") or {}
        props = shape.get("properties") or {}
        held = self.price_mode
        if held and held["sig"] not in self.broken_modes and held["words"] < word_count:
            return
        bound_spec = {}
        if isinstance(props, dict):
            bound_spec = props.get(bound_key) or {}
        self.price_mode = dict(sig=signature, tmpl=template, mode=fixed, shelf=shelf_key, bound=bound_key, words=word_count,
                               int=bound_spec.get("type") == "integer")

    @staticmethod
    def without_shared_words(base, dicts, bound_key, shelf_key, fixed):
        def every_row_says(val):
            wanted = val.strip().lower()
            return all(any(isinstance(x, str) and x.strip().lower() == wanted for key, x in row.items()
                           if not re.search(r"categor|shelf|department|aisle", str(key).lower()))
                       for row in dicts)
        kept = {}
        for k, v in base.items():
            if k == bound_key or k == shelf_key:
                continue
            if isinstance(v, str) and v.strip() and k not in fixed and every_row_says(v):
                continue
            kept[k] = v
        return kept

    @staticmethod
    def row_price(row):
        for key, val in row.items():
            if is_number(val) and not isinstance(val, bool) and re.search(r"price|cost", str(key).lower()):
                return float(val)
        return None

    def score_rows(self, call_args, rows):
        lookout = self.ep.lookout
        score = lookout.value_score(call_args) if rows else 0
        if score:
            for row in rows:
                if isinstance(row, dict):
                    mark = lookout.row_mark(row)
                    self.row_score[mark] = max(self.row_score.get(mark, 0), score)

    def unwrap_rows(self, obs):
        rows = self.result_rows(obs)
        if rows is not None:
            return obs, rows
        for val in obs.values():
            if isinstance(val, dict) and self.result_rows(val) is not None:
                return val, self.result_rows(val) or []
        return None, None

    @staticmethod
    def result_rows(piece):
        lists = [val for val in piece.values() if isinstance(val, list) and all(isinstance(r, dict) for r in val)]
        for val in lists:
            if val:
                return val
        return lists[0] if lists else None

    def query_remarks(self, call_args, rows, raw, totals, mode, paging_keys, texts, kind, remarks):
        ep = self.ep
        self.hint_remarks(rows, raw, remarks)
        marks = tuple(map(ep.lookout.row_mark, rows))
        self.repeat_remark(call_args, rows, raw, totals, mode, paging_keys, marks, remarks)
        every_word = kind == "all" or ep.lead_words
        told = bool(every_word and not totals and rows)
        hits = []
        if told or (kind != "any" and not totals and rows):
            hits = self.book_hits(call_args, rows, raw, told, texts, remarks)
        if every_word and not self.back_hits_done:
            self.back_hits_done = True
            self.earlier_hits(call_args, paging_keys, hits, remarks)

    def hint_remarks(self, rows, raw, remarks):
        ep = self.ep
        pieces = [piece for text in raw for piece in re.findall(r"[a-z0-9]+", text.lower())]
        if not rows and not self.short_hint_given and pieces and max(map(len, pieces)) == 1:
            self.short_hint_given = True
            remarks.append("Zero rows came back: this store's search drops one-letter words, so a figure like 0.4 or 5 x 7 "
                           "yields nothing (or just the other words). Query a value spelled as a word, or the unit word with the "
                           "item kind, and read every page.")
        prop, probe = ep.lookout.value_probe(raw)
        roomy = lambda: ep.moves_left() > ep.brief.order_due() + 3
        if probe:
            if self.name_hints < 2 and roomy():
                remarks.append(self.probe_hint(prop, probe))
        elif self.trap_hints < 2 and roomy():
            trap = ep.bargain.word_trap(raw)[0]
            if trap:
                self.trap_hints += 1
                remarks.append(trap)

    def probe_hint(self, prop, probe):
        repeated = self.name_hints > 0
        self.name_hints += 1
        if repeated:
            opening = f"Again led by a property's name ({prop})."
        else:
            opening = (f"The query opens with a property's NAME ({prop}) but contains no required value: the keyword search finds listing values, titles and categories, and a field name only if a value or title repeats it, so listings whose {prop} field holds just the value are missed.")
        return opening + f" Search the value first, then the item kind: '{probe}' (a figure the store can't parse falls out, keeping its unit word), and read every page of it."

    def repeat_remark(self, call_args, rows, raw, totals, mode, paging_keys, marks, remarks):
        mode_keys = dict(mode)
        fixed = []
        for name, value in call_args.items():
            if name in paging_keys or (isinstance(value, str) and name not in mode_keys):
                continue
            fixed.append((name, str(value)))
        signature = (mode, tuple(sorted(fixed)), max(totals or [len(rows)]), marks)
        query = " ".join(raw)
        earlier = self.page_signatures.get(signature)
        if signature in self.page_signatures and earlier != query and len(set(marks)) >= 3:
            remarks.append(f"Your earlier search '{earlier}' gave the same total and rows: the store treated both alike, so another spelling adds nothing. Open its next page, or change its leading words.")
        self.page_signatures.setdefault(signature, query)

    def book_hits(self, call_args, rows, raw, told, texts, remarks):
        ep = self.ep
        stashed = None if told else list(ep.ref_hits or [])
        ep.ref_hits = []
        hits = ep.lookout.row_hits(rows, raw)
        self.score_hits(hits, ep.lookout.value_score(call_args))
        if stashed is not None:
            ep.ref_hits = stashed
        if told and hits and len(texts) and len(rows) <= 12:
            remarks.append("All words must match, and all matches are listed: every row holds the searched words in some field "
                           "of its record, often in fields the tittle does not show. Before searching further, open the "
                           "in-budget ones: " + ", ".join(hits[:6]) + ".")
        if not told:
            return hits
        popped, ep.ref_hits = ep.ref_hits, []
        pointers = [ident for ident in popped if ident not in hits][:6]
        if pointers:
            remarks.append("A value in the request comes from another listing, and these rows belong to the shop or brand named "
                           "by this search: read each one whatever price, stock or category (a listing giving a value is read, "
                           "not bought), and go where its notes lead: " + ", ".join(pointers) + ".")
        return hits

    def score_hits(self, hits, score):
        self.hit_score.update(dict((ident, max(self.hit_score.get(ident, 0), score)) for ident in hits))
        self.all_hits.extend([ident for ident in hits if ident not in self.all_hits])

    def earlier_hits(self, call_args, paging_keys, hits, remarks):
        lookout = self.ep.lookout
        here = json.dumps(dict((name, value) for name, value in call_args.items() if name not in paging_keys),
                          sort_keys=True, default=str)
        back = []
        for walk_key, entry in self.ep.page_book.items():
            if walk_key != here and entry.get("every") and lookout.value_score(entry["args"]):
                fresh = [ident for ident in lookout.row_hits(entry["every"]) if ident not in back and ident not in hits]
                self.hit_score.update(dict((ident, max(self.hit_score.get(ident, 0), lookout.value_score(entry["args"])))
                                           for ident in fresh))
                back += fresh
        self.all_hits.extend([ident for ident in back if ident not in self.all_hits])
        if back:
            remarks.append("Every word must match in this store's keyword search, so the earlier searches shown in full gave "
                           "only rows holding all the words, often in fields the titles never show. Before you search further, "
                           "open the unread in-budget ones: " + ", ".join(back[:6]) + ".")

    def lead_remark(self, call_args, lead):
        context = self.ep.lead_ctx
        texts, mode, _, _ = self.ep.lookout.search_keys(call_args)
        words = ()
        if mode == context[1]:
            for key, found in texts:
                if key == context[0]:
                    words = found
                    break
        counted = f" (this query counted only: {' '.join(words[:lead])})" if len(words) > lead else ""
        return (f"This store's keyword search ({context[0]}) counts just the FIRST {lead} word(s) of a query and drops "
                f"the rest{counted}: put the {lead} most telling words up front (a required value plus the item kind, the value first), and search other values in separate short queries.")

    def over_limit_remark(self, call_args, rows):
        paging = self.ep.lookout.search_keys(call_args)[3]
        limits = []
        for name, value in call_args.items():
            if not is_number(value) or not value > 1 or name in paging:
                continue
            if not re.search(r"page|size|count|result|min|low|from|start", str(name).lower()):
                limits.append(value)
        if not limits:
            return ""
        cap = max(limits)
        over = sum(1 for row in rows
                   if any(is_number(value) and value > cap and "price" in str(name).lower() for name, value in row.items()))
        if over == 0:
            return ""
        return (f"{over} of these rows are priced above the limit {cap} set by this call: this search mode ignored the filter (perhaps other ones too, like the category). Pick a mode whose filters hold, or check every row yourself.")

    @staticmethod
    def unseen_remark(kind, totals, rows):
        if kind == "all":
            how = ("this store's keyword search needs each word, so more precise words narrow it, as do the category and "
                   "price filters with a lower price limit.")
        elif kind == "any":
            how = ("one matching word suffices for the keyword search here, and rows hitting more query words sit "
                   "higher, so each extra word broadens it: query only the decisive word or figure of a required value "
                   "to get just the rows that have it (often few enough to page through and compare prices); the kind "
                   "with the values puts the best matches first; then use the category mode (or a price filter if one "
                   "holds) for the rest.")
        else:
            how = ("afterwards cut down the remainder: when each word is required, add exact words; if any word suffices "
                   "(seen as a longer query finds more listings), put the required values in the query so fitting "
                   "listings come first; a price filter, when one holds, also helps.")
        return (f"{max(totals)} listings matched this search, yet this page shows only {len(rows)}, and just a few pages can be read: the listings that matter may lie in the unseen ones. First read this same search's later pages (as many as can be read); should the total go beyond what those pages hold, " + how)

    def shopper_lines(self, call_name, obs, refused):
        found = []
        if call_name != self.ep.role("message"):
            if not isinstance(obs, dict):
                return found
            for piece in obs.values():
                if isinstance(piece, str):
                    if refused and len(piece.split()) >= 4:
                        found.append(piece)
                elif isinstance(piece, list) and piece and all(isinstance(item, str) for item in piece):
                    long_strings(piece, found)
            return found
        if isinstance(obs, str):
            return [obs]
        if isinstance(obs, dict) and "error" not in obs:
            long_strings([item for item in obs.values() if isinstance(item, (str, list))], found)
        return found

    def value_pages(self):
        ep = self.ep
        try:
            chasing = any(ref.get("status") == "open" for ref in ep.refs or [])
            if chasing and ep.pointers.ref_room():
                return False
            if ep.orders_tried or ep.refusal_n or ep.shop.done or ep.shop.broke:
                return False
            if ep.time_left() < CLOSING + 60 or not ep.role("search"):
                return False
            if self.value_label_search():
                return True
            if not (self.cheapest_rule() and ep.word_kind != "any" and ep.lookout.page_arg()):
                return False
            return self.page_or_flush(self.cg_products())
        except Exception as err:
            ep.log.append({"kind": "value_pages_error", "error": f"{type(err).__name__}: {err}"[:300]})
            return False

    def follow_pages(self):
        ep = self.ep
        try:
            paging = ep.lookout.page_arg()
            if not paging or not ep.role("search") or ep.orders_tried or ep.refusal_n or ep.shop.done or ep.shop.broke:
                return False
            if self.followed >= FOLLOW_TOTAL or ep.time_left() < CLOSING + 40 or self.burst_room()[1] <= 6:
                return False
            walk = self.follow_target(paging)
            if walk is None:
                return False
            field, position = paging[0], self.follow_position(walk, paging)
            self.followed += 1
            walk["followed"] = walk.get("followed", 0) + 1
            walk["home_at"] = self.home_rows(walk)
            args = dict(walk["args"], **{field: position})
            words = " ".join(ep.lookout.free_text(walk["args"]))
            ep.log.append({"kind": "follow_page", "query": words, "position": position})
            ep.bargain.walk_call(args, f"The search '{words}' holds a required value and shows listings of the requested "
                                       "category, with more results than one page shows, so the code reads its next page "
                                       "before any new wording.")
            return True
        except Exception as err:
            ep.log.append({"kind": "follow_error", "error": f"{type(err).__name__}: {err}"[:300]})
            return False

    def follow_target(self, paging):
        ep = self.ep
        homes = set().union(*(ep.shop.home_shelves() or [set()])) or {ep.shop.own_shelf()}
        valued = set()
        for req in ep.wants.props_left():
            value = re.sub(r"\((?:stated|revealed[^)]*)\)", "", str(req).partition(":")[2])
            valued |= {ep.lookout.stem(w) for w in plain_words(value) if len(w) >= 3 and not w.isdigit()}
        if not valued:
            return None
        scored = []
        for order, walk in enumerate(ep.page_book.values()):
            said = {ep.lookout.stem(w) for w in plain_words(" ".join(ep.lookout.free_text(walk["args"])))}
            if said & valued and walk.get("size"):
                scored.append((len(said & valued), order, walk))
        scored.sort(key=lambda entry: (-entry[0], -entry[1]))
        for rank, (_, _, walk) in enumerate(scored):
            cap = FOLLOW_DEEP if rank == 0 else FOLLOW_PAGES
            if walk.get("followed", 0) >= cap:
                continue
            on_home = self.home_rows(walk, homes)
            if on_home < 2 or walk["total"] <= walk["size"] * len(walk["read"]):
                continue
            if "home_at" in walk and on_home <= walk["home_at"]:
                continue
            if self.follow_position(walk, paging) is not None:
                return walk
        return None

    def home_rows(self, walk, homes=None):
        if homes is None:
            homes = set().union(*(self.ep.shop.home_shelves() or [set()])) or {self.ep.shop.own_shelf()}
        return sum(1 for shelf in (walk.get("row_shelf") or {}).values() if shelf in homes)

    @staticmethod
    def follow_position(walk, paging):
        read = walk["read"]
        if paging[1] == "token":
            ahead = [token for token in (walk.get("tokens") or {}).values() if token and token not in read]
            return ahead[-1] if ahead else None
        numbers = [p for p in read if is_number(p)]
        if not numbers:
            return None
        position = max(numbers) + (1 if paging[1] else max(1, walk.get("size") or 1))
        return None if is_number(paging[3]) and position > paging[3] else position

    def value_label_search(self):
        if self.burst_room()[1] <= 4:
            return False
        ep = self.ep
        shape = ep.shapes.get(ep.role("search") or "") or {}
        props = shape["properties"] if isinstance(shape.get("properties"), dict) else {}
        paging = ep.lookout.page_arg()
        products = self.cg_products()
        for k, home, label, template, text_key, signature, settled in self.label_search_targets(products):
            built = self._label_attempt(k, home, label, template, text_key, shape, props)
            if built is None:
                settled.append(signature)
                continue
            args, todo, mode, forms = built
            self.label_attempts[k] = self.label_attempts.get(k, 0) + 1
            plural = "s" if len(self.firm_values(k)) > 1 else ""
            ep.bargain.walk_call(args, f"Searching the required value{plural} with the category name as this store's rows spell it "
                                       f"('{todo}'): keyword search finds listing values, titles and categories, so this shows the category's listings that carry the value.")
            self.after_label_search(k, args, todo, text_key, mode, label, home, forms, paging, len(products) > 1)
            return True
        return False

    def cg_products(self):
        return list(range(len(self.wants.items()))) or [None]

    @property
    def wants(self):
        return self.ep.wants

    def label_search_targets(self, products):
        for k in products:
            if self.label_attempts.get(k, 0) >= LABEL_TRIES:
                continue
            signature = json.dumps(self.firm_values(k))
            settled = self.settled_values.setdefault(str(k), [])
            if signature in settled:
                continue
            home = self.ep.bargain.walk_home(k)
            label = self.shelf_spelling(home)
            if not label:
                continue
            template, text_key = self.label_template(k)
            if template:
                yield k, home, label, template, text_key, signature, settled
            else:
                settled.append(signature)

    def shelf_spelling(self, labels):
        if not labels:
            return ""
        tally = {}
        ids = self.known.ids
        shelf_key = lambda key: re.search(r"categor|shelf|department|aisle", str(key).lower()) and "id" not in str(key).lower()

        def count(fields):
            if not any(isinstance(value, str) and value in ids and "id" in str(key).lower() for key, value in fields.items()):
                return
            spelled = next((value.strip() for key, value in fields.items()
                            if isinstance(value, str) and 0 < len(value.strip()) <= 60 and shelf_key(key)), "")
            if spelled and spelled.lower() in labels:
                tally[spelled] = tally.get(spelled, 0) + 1
        for record in self.known.items:
            walk_dicts(record, count)
        return max(sorted(tally), key=tally.get, default="")

    @property
    def known(self):
        return self.ep.known

    def label_template(self, k):
        free_text = self.ep.lookout.free_text
        for call in self.ep.closer.code_queries(k):
            keys = [key for key, val in call.items() if isinstance(val, str) and val.strip() and val in free_text(call)]
            if len(keys) == 1 and not self.cg_listing(call):
                return call, keys[0]
        return None, ""

    def cg_listing(self, args):
        spec = self.price_spec()
        if not spec or not isinstance(args, dict):
            return False
        mode = spec["mode"]
        if mode:
            return all(args.get(k) == v for k, v in mode.items())
        return spec["bound"] in args

    def price_spec(self):
        spec = self.price_mode
        if spec and spec["sig"] not in self.broken_modes:
            return spec
        return None

    def firm_values(self, k):
        values = []
        for head, value in self.ep.closer.code_values(k):
            options = self.value_options(head, value)
            if options and options not in values:
                values.append(options)
        return values

    def value_options(self, head, value):
        lowered = re.sub(r"\([^)]*\)", " ", str(value).lower().replace("’", "'")).strip()
        if not lowered or ref_text(lowered):
            return []
        if re.search(r"\bor\b|\beither\b|/", lowered):
            pieces = [piece for piece in re.split(r"[,;/]|\bor\b|\beither\b", lowered) if piece.strip()][:3]
        else:
            pieces = re.split(r"[,;&+]|\band\b", lowered)[:1]
        own_name = set(re.findall(r"[a-z0-9]+", str(head).lower()))
        options = []
        for piece in pieces:
            words = self.option_words(piece, own_name)
            if words and words not in options:
                options.append(words)
        return options

    def option_words(self, piece, own_name):
        bargain = self.ep.bargain
        figure = bargain.fig_amount(piece)
        if re.search(EXCLUDE_WORDS, piece) or (bargain.amount_part(piece) and not figure):
            return None
        piece = re.sub(r"\b(?:" + VALUE_LINKS + r")\b", " ", re.sub(r"'s\b", "", piece))
        words = []
        for word in re.findall(r"\d+(?:[.,]\d+)+|[a-z0-9]+", piece):
            if len(word) < 2 and not (figure and word == figure[0]):
                continue
            if re.search(r"[.,]", word) or word in LINK_WORDS:
                continue
            if word not in own_name or (figure and word == figure[1]):
                words.append(word)
        return words

    def _label_attempt(self, k, home, label, template, text_key, shape, props):
        mode = {key: val for key, val in template.items() if isinstance(props.get(key), dict) and props[key].get("enum")}
        needed = shape.get("required") or []
        bare = {key: val for key, val in template.items() if key == text_key or key in mode or key in needed}
        spec = props.get(text_key)
        todo, first, second = self.pending_form(k, label, home, spec.get("maxLength") if isinstance(spec, dict) else None,
                                                text_key, mode)
        if not todo:
            return None
        args, faults = ArgCheck.execute({**bare, text_key: todo}, shape)
        if faults or not isinstance(args, dict):
            return None
        return args, todo, mode, first or second

    def pending_form(self, k, label, home, cap_len, text_key, mode):
        composed = self.composed_sets.setdefault(str(k), set())
        first, second = self.remember_forms(k, label, cap_len, composed)
        todo = self.next_form(first, second, text_key, mode)
        if todo is not None:
            return todo, first, second
        carried = False
        for entry in self.ep.page_book.values():
            if self.ep.bargain.walk_carries(entry["args"], k) and self.on_home(entry, home):
                carried = True
                break
        if self.label_indexed is not None or not carried:
            return "", first, second
        self.label_indexed = False
        first, second = self.remember_forms(k, label, cap_len, composed)
        return self.next_form(first, second, text_key, mode) or "", first, second

    def remember_forms(self, k, label, cap_len, composed):
        first, second = self.composed_queries(k, label, cap_len)
        for query in first + second:
            composed.add(self.word_set(query))
        return first, second

    def composed_queries(self, k, label, cap_len):
        values = self.firm_values(k)
        if not values:
            return [], []
        label_terms = self.label_words(k, label)
        limit = None if self.cheapest_rule() else 1
        compose = lambda parts: self.query_text(parts, label_terms, cap_len)
        dedupe = lambda forms: list(dict.fromkeys(filter(None, forms)))
        alone = dedupe(compose([option]) for option in values[0][:limit])
        if len(values) == 1:
            return alone, []
        varied = next((n for n, value in enumerate(values) if len(value[:limit]) > 1), 0)
        together = []
        for option in values[varied][:limit]:
            together.append(compose([option if n == varied else value[0] for n, value in enumerate(values)]))
        return dedupe(together), alone

    def label_words(self, k, label):
        cleaned = re.sub(r"'s\b", "", label.lower().replace("’", "'"))
        words = list(dict.fromkeys(word for word in re.findall(r"[a-z0-9]+", cleaned) if len(word) >= 2 and word not in LINK_WORDS))
        if self.label_indexed is not False:
            return words
        stem = self.ep.lookout.stem
        titles = self.title_stems_on(k)
        common = [word for word in words if titles and 2 * len([t for t in titles if stem(word) in t]) >= len(titles)]
        return common or self.product_word(k)

    def title_stems_on(self, k):
        lookout = self.ep.lookout
        home = self.ep.bargain.walk_home(k)
        titles = []

        def collect(fields):
            if lookout.shelf_of(fields) in home and lookout.row_mark(fields) in self.known.ids:
                title = self.ep.shop.title_of(fields).lower()
                titles.append({lookout.stem(w) for w in re.findall(r"[a-z0-9]+", title)})
        for record in self.known.items:
            walk_dicts(record, collect)
        return titles

    def product_word(self, k):
        rows = self.wants.items()
        if rows and k is not None and k < len(rows):
            product = str(rows[k]["product"])
        else:
            product = str(self.wants.reading.get("product") or "")
        long_words = [word for word in re.findall(r"[a-z0-9]+", product.lower()) if len(word) > 2 and word not in LINK_WORDS]
        return long_words[-1:]

    def cheapest_rule(self):
        return self.wants.priority() == "cheapest"

    def query_text(self, parts, label_terms, cap_len):
        stem = self.ep.lookout.stem
        lead = self.ep.lead_words
        chosen, roots = [], set()
        for word in [word for part in parts for word in part] + label_terms:
            if stem(word) in roots:
                continue
            roots.add(stem(word))
            chosen.append(word)
        text = " ".join(chosen[:lead] if lead else chosen)
        if not is_number(cap_len):
            return text
        while " " in text and len(text) > cap_len:
            text = text.rsplit(" ", 1)[0]
        return text[: int(cap_len)]

    def word_set(self, text):
        stem = self.ep.lookout.stem
        words = re.findall(r"[a-z0-9]+", str(text).lower())[: self.ep.lead_words or None]
        return frozenset(stem(w) for w in words)

    def next_form(self, first, second, text_key, mode):
        for forms in (first, second):
            if forms:
                totals = [self.sent_total(text, text_key, mode) for text in forms]
                if None in totals:
                    return forms[totals.index(None)]
                if any(totals):
                    return ""
        return None if first or second else ""

    def sent_total(self, text, text_key, mode):
        wanted = self.word_set(text)
        for key, total in (self.totals_seen or {}).items():
            try:
                args = json.loads(key)
            except Exception:
                continue
            if isinstance(args, dict) and isinstance(args.get(text_key), str):
                if all(args.get(m) == v for m, v in mode.items()) and self.word_set(args[text_key]) == wanted:
                    return int(total)
        return None

    @staticmethod
    def on_home(entry, home):
        return [ident for ident, label in (entry.get("row_shelf") or {}).items() if label in home]

    def after_label_search(self, k, args, todo, text_key, mode, label, home, forms, paging, several):
        ep = self.ep
        total = self.sent_total(todo, text_key, mode)
        self.note_label_indexing(label)
        trimmed = dict((name, value) for name, value in args.items() if not paging or name != paging[0])
        key = json.dumps(trimmed, sort_keys=True, default=str)
        entry = ep.page_book.get(key)
        every = bool(entry) and len(entry.get("row_shelf") or {}) >= entry["total"] and bool(self.on_home(entry, home))
        floor = ep.brief.order_due() + 2
        note = self.label_note(todo, total, every)
        queue = self.label_queue(k, key, entry, every, forms, todo, paging, several, home, floor)
        if queue:
            below = " below the pick" if ep.bargain.walk_pick(k) and self.cheapest_rule() else ""
            note += (" The unread in-stock rows inside the budget in the requested category" + below + ", cheapest first, are opened next: "
                     + self.priced_list(queue) + ".")
        if note and self.tool_is_last():
            self.append_to_tool(note + ("\n" + ep.status_line() if not queue else ""))
        ep.log.append({"kind": "label_search", "product": k, "query": todo, "total": total, "every": every, "opened": queue})
        if queue:
            self.launch_reads(queue, home, floor, ("I now open that search's lowest-priced unread rows - all listings of the category carrying the "
                                                   "required values are among its results."))

    def note_label_indexing(self, label):
        if self.label_indexed is not None or not self.known.items:
            return
        lookout = self.ep.lookout
        spelled = label.strip().lower()
        label_stems = set(lookout.stem(word) for word in re.findall(r"[a-z0-9]+", re.sub(r"'s\b", "", label.lower()))
                          if len(word) >= 2 and word not in LINK_WORDS)
        indexed = []

        def count(fields):
            if lookout.row_mark(fields) in self.known.ids and lookout.shelf_of(fields) == spelled:
                elsewhere = set()
                for key, text in fields.items():
                    if isinstance(text, str) and not re.search(r"categor|shelf|department|aisle", str(key).lower()):
                        elsewhere.update(map(lookout.stem, re.findall(r"[a-z0-9]+", text.lower())))
                if label_stems - elsewhere:
                    indexed.append(fields)
        walk_dicts(self.known.items[-1], count, ceiling=3)
        if len(indexed) >= 2:
            self.label_indexed = True

    @staticmethod
    def label_note(todo, total, every):
        if total is None:
            return ""
        if not total:
            return f"[value search] '{todo}': no listing matched."
        return f"[value search] '{todo}': {total} match(es)" + (" - every one is shown here." if every else ".")

    def label_queue(self, k, key, entry, every, forms, todo, paging, several, home, floor):
        if not every:
            return []
        if not self.cheapest_rule():
            return self.cheap_queue(k, self.on_home(entry, home), floor, LABEL_OPENS) if todo in forms else []
        if not paging or entry.get("id_order") is False:
            return []
        self.valued_walks.add(key)
        self.valued_walk_keys.setdefault(str(k), []).append(key)
        if several:
            self.waiting_flush[k] = True
            return []
        return self.cheap_queue(k, self.on_home(entry, home), floor, CEIL_OPENS)

    def cheap_queue(self, k, ids, floor, most):
        bargain = self.ep.bargain
        cheapest = self.cheapest_rule()
        pick = bargain.walk_pick(k) if cheapest else None
        batch = self.ep.shop.batch_tool()
        opened = self.opens_by_product
        room = most if not cheapest else CEIL_OPENS - opened.get(k, 0)
        take = max(0, min(most if not batch else batch[2], room))
        cheap = self.cheapest_unread(ids, k, pick[1] if pick else None)
        queue = self.trim_to_steps([pair[1] for pair in cheap[:take]], batch, floor)
        if cheapest:
            opened[k] = len(queue) + opened.get(k, 0)
        return queue

    def cheapest_unread(self, ids, k, below=None):
        pool = set(ids)
        if not pool:
            return []
        rows = self.wants.items()
        qty = self.wants.quantity() if k is None or k >= len(self.wants.items()) else rows[k]["quantity"]
        budget = self.wants.spend_cap
        sold_out = self.sold_out_marks(pool)
        known, refused = self.known, self.ep.refused_lines
        ranked = []
        for ident in pool:
            price = known.price(ident)
            if ident in known.read or ident in refused or sold_out.get(ident) or not is_number(price):
                continue
            over = is_number(budget) and is_number(qty) and price * qty > budget
            if not over and (below is None or price < below):
                ranked.append((float(price), ident))
        ranked.sort()
        return ranked

    def sold_out_marks(self, ids):
        lookout = self.ep.lookout
        marks = {}

        def note_stock(fields):
            if any(isinstance(v, str) and v in ids and "id" in str(key).lower() for key, v in fields.items()):
                in_stock = lookout.stock_state(fields)
                if in_stock is not None:
                    marks[lookout.row_mark(fields)] = not in_stock
        for record in self.known.items:
            walk_dicts(record, note_stock)
        return marks

    def trim_to_steps(self, queue, batch, floor):
        while queue:
            cost = 1 if batch and len(queue) >= 2 else len(queue)
            if cost < self.ep.moves_left() - floor:
                break
            queue.pop()
        return queue

    def tool_is_last(self):
        chat = self.ep.chat
        return bool(chat) and chat[-1].get("role") == "tool"

    def append_to_tool(self, text):
        newest = self.ep.chat[-1]
        newest["content"] = str(newest["content"]) + "\n" + text

    def launch_reads(self, queue, homes, floor, say):
        ep = self.ep
        shelf_known = ep.counsel.shelf_known
        ep.rank_checked = {mark for mark in ep.rank_checked if shelf_known(mark.split("|")[0]) not in homes}
        ep.order_vetted = False
        orders = ep.reads
        orders.open_queue, orders.open_recheck, orders.open_say = queue, True, say
        orders.open_purpose, orders.open_floor = "walk", floor
        ep.counsel.open_named()

    def priced_list(self, ids):
        return ", ".join(f"{ident} ({self.known.price(ident):g})" for ident in ids)

    def burst_room(self):
        ep = self.ep
        left = 0
        if ep.asking and ep.role("message"):
            left = max(0, ep.ask_limit() - ep.msgs_sent)
        return left, ep.moves_left() - ep.brief.order_due() - 2 * left

    def page_or_flush(self, products):
        several = len(products) > 1
        plan = self.paging_plan(products, several)
        options = []
        for n, k in enumerate(products):
            if home := self.ep.bargain.walk_home(k):
                options.extend(self.page_candidates(k, n, home, plan))
        if options:
            best = min(options, key=lambda option: option[:5])
            self.valued_walks.add(best[4])
            return self.read_value_walk(best[5], best[4], plan["floor"], best[1], several)
        waiting = self.waiting_flush
        if not waiting or plan["pending"]:
            return False
        if plan["left"] and plan["room"] > plan["opens"] * len(waiting) + 4:
            return False
        return self.flush_waiting(plan["floor"])

    def paging_plan(self, products, several):
        ep = self.ep
        depth = self.reachable_depth()
        size = max(1, self.widest_page())
        left, room = self.burst_room()
        unread_homes = len(list(filter(lambda k: ep.bargain.walk_home(k) and not self.value_pages_read.get(k), products)))
        plan = dict(depth=depth, several=several, pending=False, left=left, room=room)
        plan["per"] = max(VALUE_PAGES, -(-depth // size) - 1) if depth else VALUE_PAGES
        plan["cap"] = max(VALUE_TOTAL, VALUE_TOTAL_PER * len(products))
        plan["opens"] = 1 if ep.shop.batch_tool() else (2 if several else CEIL_OPENS)
        plan["floor"] = ep.brief.order_due() + 2
        plan["known"] = self.all_words()
        plan["share"] = max(4, room // max(1, unread_homes))
        return plan

    def reachable_depth(self):
        paging = self.ep.lookout.page_arg()
        size = self.widest_page()
        if not paging or size <= 0 or paging[1] == "token" or paging[3] is None:
            return self.depth_cap
        first, last = paging[2], paging[3]
        return int((last - first + 1) * size) if paging[1] else int(last - first + size)

    def all_words(self):
        kind = self.ep.word_kind
        if kind == "all":
            return True
        return bool(self.ep.lead_words) and kind != "any"

    def flush_waiting(self, floor):
        waiting = [k for k, on in self.waiting_flush.items() if on]
        self.waiting_flush.clear()
        if not waiting:
            return False
        ep = self.ep
        batch = ep.shop.batch_tool()
        per = 1
        if ep.moves_left() - floor - 1 >= 2 * len(waiting):
            per = 2
        owner, queue = self.take_turns(waiting, self.flush_pools(waiting, per), batch[2] if batch else per * len(waiting))
        if not self.trim_to_steps(queue, batch, floor):
            return False
        counts = self.opens_by_product
        for product in map(owner.get, queue):
            counts[product] = counts.get(product, 0) + 1
        homes = set()
        for k in waiting:
            homes |= ep.bargain.walk_home(k)
        self.launch_reads(queue, homes, floor, "Cheapest wanted: I open the cheapest unread rows that every product's value search listed.")
        if self.tool_is_last():
            self.append_to_tool("[value searches read] For each pick, the code opened the cheapest unread in-stock rows inside the "
                                "budget from every product's value search: " + self.priced_list(queue) + (". Per product, choose the lowest-priced listing that by its own fields meets every requirement now "
                                                                                                         "in force; any costlier fit requires every cheaper candidate opened and rejected first."))
        ep.log.append({"kind": "value_flush", "products": waiting, "opened": queue})
        return True

    @staticmethod
    def take_turns(waiting, pools, most):
        owner, queue = {}, []
        while len(queue) < most and any(pools):
            for k, pool in zip(waiting, pools):
                while pool and pool[0] in queue:
                    del pool[0]
                if pool and len(queue) < most:
                    ident = pool.pop(0)
                    owner[ident] = k
                    queue.append(ident)
        return owner, queue

    def flush_pools(self, waiting, per):
        bargain = self.ep.bargain
        book = self.ep.page_book
        walks = self.valued_walk_keys or {}
        pools = []
        for k in waiting:
            home = bargain.walk_home(k)
            listed = set()
            for key in walks.get(str(k), []):
                if key in book and bargain.walk_carries(book[key]["args"], k):
                    listed |= set(self.on_home(book[key], home))
            pick = bargain.walk_pick(k)
            room = max(0, min(per, CEIL_OPENS - self.opens_by_product.get(k, 0)))
            cheap = self.cheapest_unread(listed, k, None if not pick else pick[1])
            pools.append([pair[1] for pair in cheap][:room])
        return pools

    def read_value_walk(self, k, key, floor, most=VALUE_PAGES, defer=False):
        ep = self.ep
        lookout = ep.lookout
        paging = lookout.page_arg()
        entry = ep.page_book.get(key)
        if not (paging and entry):
            return False
        query = " ".join(lookout.free_text(entry["args"]))
        every_word = self.all_words()
        got, tried = self.read_pages(k, entry, paging, most, floor, self.walk_say(query, every_word))
        if got == 0:
            return tried
        self.valued_walk_keys.setdefault(str(k), []).append(key)
        seen = len(entry.get("row_shelf") or {})
        ended = lookout.next_page(entry) is None
        complete = ended and seen >= entry["total"]
        home = ep.bargain.walk_home(k)
        if complete:
            reach = " - all listings carrying these words are among them" if every_word else " - all listings it matched are among them"
        else:
            reach = " - the remainder is unreachable, so cheaper fits may sit past what is reachable" if lookout.next_page(entry) is None else ""
        queue, middle = self.walk_middle(k, entry, home, floor, defer)
        if queue:
            self.launch_reads(queue, home, floor, "The cheapest is wanted: I open the search's cheapest unread rows.")
        if self.tool_is_last():
            summary = (f"[value search read] '{query}': {seen} of its {entry['total']} matches seen" + reach + ". " + middle
                       + ("Choose the cheapest listing that by its own fields meets every requirement now in force; any "
                          "costlier fit requires every cheaper candidate opened and rejected first."))
            self.append_to_tool(summary + ("" if queue else "\n" + ep.status_line()))
        ep.log.append({"kind": "value_pages", "product": k, "query": query, "pages": got, "complete": complete, "opened": queue,
                       "defer": defer})
        return True

    def read_pages(self, k, entry, paging, most, floor, say):
        lookout = self.ep.lookout
        got = 0
        tried = False
        while got < most:
            if lookout.next_page(entry) is None:
                break
            if not self.may_page(floor):
                break
            tried = True
            self.value_pages_read[k] = 1 + self.value_pages_read.get(k, 0)
            after = lookout.next_page(entry)
            if not self.ep.bargain.walk_call(dict(entry["args"], **{paging[0]: after}), say):
                break
            got += 1
            if after == lookout.next_page(entry):
                break
        return got, tried

    def may_page(self, floor):
        ep = self.ep
        shop = ep.shop
        if ep.moves_left() <= floor + 1 or shop.done or shop.broke:
            return False
        return ep.time_left() >= CLOSING + 30

    def walk_middle(self, k, entry, home, floor, defer):
        if defer:
            self.waiting_flush[k] = True
            return [], ("Once each product's value search is read, I open its lowest-priced unread in-stock rows inside the "
                        "budget. ")
        pick = self.ep.bargain.walk_pick(k)
        queue = self.cheap_queue(k, self.on_home(entry, home), floor, CEIL_OPENS)
        if not queue:
            return queue, "Among them no unread in-stock row inside the budget in the requested category is left. "
        below = " below the pick" if pick else ""
        return queue, ("The cheapest unread in-stock rows inside the budget in the requested category" + below
                       + " are opened next: " + self.priced_list(queue) + ". ")

    @staticmethod
    def walk_say(query, every_word):
        if every_word:
            return ("The cheapest is wanted; this store's keyword search wants each word and orders results in a fixed id order, not by price, so I read the next page of '" f"{query}' so every listing carrying these words gets seen.")
        return ("The cheapest is wanted; the store orders search results in a fixed id order, not price, reading the next page of '" f"{query}' so every listing it matched gets seen.")

    def page_candidates(self, k, n, home, plan):
        ep = self.ep
        mine = (self.composed_sets or {}).get(str(k)) or set()
        terms = self.product_terms(k, home)
        finished = self.finished_searches(k)
        found = []
        for key, entry in ep.page_book.items():
            if not self.open_value_walk(key, entry, k, home):
                continue
            words = self.word_set(" ".join(ep.lookout.free_text(entry["args"])))
            row = self._page_row(k, plan, key, entry, words, words in mine, finished, terms)
            if row is not None:
                found.append(row)
        if plan["known"]:
            found = [row for row in found if not self.nested_better(row[3], found, terms)]
        used = self.value_pages_read.get(k, 0)
        return [(used, take, score, n, key, k) for take, score, key, _ in found]

    def product_terms(self, k, home):
        stem = self.ep.lookout.stem
        several = bool(self.wants.items()) and k is not None
        product_of = self.ep.talk.product_of
        terms = set()
        for req in self.wants.props_left():
            if several and product_of(req.partition(":")[0]) not in (k, None):
                continue
            terms |= {stem(w) for w in re.findall(r"[a-z0-9]{2,}", req.partition(":")[2].lower())}
        for label in home:
            terms |= {stem(w) for w in re.findall(r"[a-z0-9]{2,}", label)}
        return terms

    def finished_searches(self, k):
        lookout, bargain = self.ep.lookout, self.ep.bargain
        done = []
        for entry in self.ep.page_book.values():
            closed = entry.get("id_order") is True and entry.get("first") and lookout.next_page(entry) is None
            if not closed or self.cg_listing(entry["args"]):
                continue
            if len(entry.get("row_shelf") or {}) >= entry["total"]:
                args = entry["args"]
                done.append((self.word_set(" ".join(lookout.free_text(args))), bargain.walk_carries(args, k, alts=True),
                             self.vl_mode(args)))
        return done

    def vl_mode(self, args):
        shape = self.ep.shapes.get(self.ep.role("search") or "") or {}
        props = shape.get("properties")
        if not isinstance(props, dict):
            props = {}

        def small_enum(key, value):
            spec = props.get(key)
            options = spec.get("enum") if isinstance(spec, dict) else None
            return isinstance(value, str) and isinstance(options, list) and len(options) <= 4
        return tuple(sorted((key, value) for key, value in args.items() if small_enum(key, value)))

    def _page_row(self, k, plan, key, entry, words, composed, finished, terms):
        bargain = self.ep.bargain
        if not self.depth_allows(plan, entry, composed):
            return None
        if plan["known"]:
            carries = bargain.walk_carries(entry["args"], k, alts=True)
            if self.shown_before(finished, words, terms, carries, self.vl_mode(entry["args"])):
                return None
        take = self.pages_to_take(plan, entry, k, composed)
        if take <= 0 or self.ep.moves_left() - take - plan["opens"] <= plan["floor"]:
            return None
        if (take > 2 or plan["several"]) and plan["left"] and plan["room"] > take + plan["opens"] + 4:
            plan["pending"] = True
            return None
        return take, -bargain.walk_score(entry["args"], k), key, words

    def pages_to_take(self, plan, entry, k, composed):
        used, opened = self.value_pages_read, self.opens_by_product
        depth = plan["depth"]
        reach = min(entry["total"], depth) if depth is not None else entry["total"]
        pages = -(-reach // max(1, entry["size"])) - len(entry["read"])
        allowed = min(plan["per"] - used.get(k, 0), plan["cap"] - sum(used.values()))
        if plan["several"]:
            already = opened.get(k, 0)
            allowed = min(allowed, plan["share"] - used.get(k, 0) - already - (plan["opens"] if not opened.get(k) else 0))
        if pages > allowed:
            return allowed if composed else 0
        return pages

    @staticmethod
    def depth_allows(plan, entry, composed):
        depth = plan["depth"]
        too_deep = depth is not None and entry["total"] > depth
        if plan["known"]:
            return composed or not too_deep
        return depth is not None and not too_deep

    @staticmethod
    def shown_before(finished, words, terms, pairs, how):
        for seen, got, mode in finished:
            if mode != how:
                continue
            if seen <= words or (seen <= terms and pairs <= got):
                return True
        return False

    def open_value_walk(self, key, entry, k, home):
        if key in self.valued_walks or entry.get("id_order") is not True or not entry.get("first"):
            return False
        if self.ep.lookout.next_page(entry) is None or self.cg_listing(entry["args"]):
            return False
        return bool(self.ep.bargain.walk_carries(entry["args"], k)) and bool(self.on_home(entry, home))

    @staticmethod
    def nested_better(words, found, terms):
        for row in found:
            other = row[3]
            if other < words and not (words - other) <= terms:
                return True
            if words < other and (other - words) <= terms:
                return True
        return False

    def product_steps(self):
        try:
            rows = self.wants.items()
            if len(rows) < 2 or not self.cheapest_rule():
                return ""
            pages = self.value_pages_read or {}
            picks = [self.ep.bargain.walk_pick(k) for k in range(len(rows))]
            parts = ["{}: {} value page(s) read, pick {}".format(row["product"], pages.get(k, 0), "yes" if picks[k] else "no")
                     for k, row in enumerate(rows)]
            missing = next((str(row["product"]) for k, row in enumerate(rows) if not picks[k]), "")
            return "\n[product steps] " + "; ".join(parts) + (f"; next: {missing}" if missing else "") + "."
        except Exception:
            return ""

    def walk_step(self):
        ep = self.ep
        bargain = ep.bargain
        try:
            if bargain.decline_queue:
                return bargain.run_decline_reads()
            queued, self.queued_walk = self.queued_walk, None
            if queued:
                return bargain.run_walk(*queued)
            if (again := self.rewalk_dead_aim()) is not None:
                return again
            idle = ep.orders_tried or not self.cheapest_rule() or ep.time_left() < CLOSING + 60 or ep.shop.done
            if idle or self.product_unfound():
                return False
            opens = 2 - bool(ep.shop.batch_tool())
            spare = ep.moves_left()
            due = ep.brief.order_due
            ready = list(filter(self.walkable_now, self.cg_products()))
            roomy = spare > due() + 2 + CEIL_CALLS + opens
            if not roomy:
                return self.reserve_walk(spare, opens)
            if ready:
                pages = self.value_pages_read or {}
                ready.sort(key=lambda k: (bool(bargain.walk_pick(k)), pages.get(k, 0), k if k is not None else -1))
                return bargain.run_walk([], due() + 2, ready[:1])
            return True if self.seed_price_filter() else self.reserve_walk(spare, opens)
        except Exception as err:
            ep.log.append({"kind": "walk_error", "error": f"{type(err).__name__}: {err}"[:300]})
            return False

    def product_unfound(self):
        if not self.wants.items():
            return False
        try:
            read = {label.strip().lower() for label in self.known.shelves_read()}
            if any(not home or not home & read for home in self.ep.shop.home_shelves()):
                return True
            version = str(self.wants.version)
            live = [mark for mark in self.ep.passed if mark.rpartition("|")[2] == version]
            return bool(live) and self.ep.brief.unverified()
        except Exception:
            return False

    def walkable_now(self, k):
        if not (self.ep.bargain.walk_pick(k) or self.has_amount_bound(k)):
            return False
        return self.can_list_below(k) and self.aim_price(k) is not None

    def can_list_below(self, k):
        if k is None and self.wants.items():
            return False
        if not self.cheapest_rule() or k in self.walked_products or self.price_spec() is None:
            return False
        if self.ep.lookout.page_arg() is None:
            return False
        return bool(self.shelf_spelling(self.ep.bargain.walk_home(k)))

    def has_amount_bound(self, k):
        if k is None or not self.wants.items():
            return False
        product_of = self.ep.talk.product_of
        for req in self.wants.props_left():
            head, _, value = str(req).partition(":")
            if product_of(head) == k and re.search(r"\d", value) and re.search(CH_BOUND, value.lower()):
                return True
        return False

    def aim_price(self, k):
        bargain = self.ep.bargain
        pick = bargain.walk_pick(k)
        held = self.put_aims.get(k)
        costs = [pick[1]] if pick else []
        if is_number(held):
            costs.append(float(held))
        if not costs and self.has_amount_bound(k):
            home = bargain.walk_home(k)
            seen = [cost for cost, _ in self.cheapest_unread(self.rows_on(home), k)] + self.read_on_home(home)
            if seen:
                costs = [min(seen)]
        return min(costs) if costs else None

    def read_on_home(self, home):
        ep, known = self.ep, self.known
        counsel, shop = ep.counsel, ep.shop
        prices = []
        for ident in known.read:
            if counsel.shelf_known(ident) not in home or not is_number(known.price(ident)):
                continue
            if ident not in shop.dead_lines() and ident not in ep.refused_lines and counsel.last_stock(ident) is not False:
                prices.append(float(known.price(ident)))
        return prices

    def rows_on(self, home):
        lookout, ids = self.ep.lookout, self.known.ids
        found = set()

        def collect(fields):
            ident = lookout.row_mark(fields)
            if ident in ids and lookout.shelf_of(fields) in home:
                found.add(ident)
        for record in self.known.items:
            walk_dicts(record, collect)
        return found

    def reserve_walk(self, spare, opens):
        bargain = self.ep.bargain
        due = self.ep.brief.order_due
        plan = bargain.walk_plan(WALK_TOTAL)
        if not plan:
            return False
        pages = sum(n for _, _, n in plan)
        pages += sum(1 for k, _, _ in plan if bargain.walk_pick(k))
        if spare > due() + 2 + pages + opens:
            return False
        plan = bargain.walk_plan(min(WALK_TOTAL, spare - due() - 1 - opens))
        return bargain.run_walk(plan, due()) if plan else False

    def seed_price_filter(self):
        ep = self.ep
        try:
            if self.price_spec() is not None or self.price_mode or not self.cheapest_rule():
                return False
            paging = ep.lookout.page_arg()
            several = bool(self.wants.items())
            for k in self.cg_products():
                if k in self.seeded_products or (k is None and several):
                    continue
                outcome = self.seed_one(k, paging)
                if outcome is not None:
                    return outcome
            return False
        except Exception as err:
            ep.log.append({"kind": "ceil_seed_error", "error": f"{type(err).__name__}: {err}"[:300]})
            return False

    def seed_one(self, k, paging):
        ep = self.ep
        bargain = ep.bargain
        pick = bargain.walk_pick(k)
        label = self.shelf_spelling(bargain.walk_home(k))
        if not (pick and label):
            return None
        self.seeded_products.add(k)
        args = self.seed_listing_args(label, self.just_below(pick[1]))
        if not args:
            return False
        signature = self.seed_signature(args, label, paging)
        stripped = dict((name, value) for name, value in args.items() if not paging or name != paging[0])
        key = json.dumps(stripped, sort_keys=True, default=str)
        if signature in self.broken_modes or key in (self.totals_seen or {}):
            return None
        bargain.walk_call(args, f"Cheapest wanted: listing '{label}' priced under the pick ({pick[1]:g}) to test whether this "
                                "store's price filter holds.")
        ep.log.append({"kind": "ceil_seed", "product": k, "args": args, "learned": self.price_spec() is not None})
        if self.tool_is_last() and self.price_spec() is None:
            self.append_to_tool(ep.status_line())
        return True

    def seed_listing_args(self, label, bound):
        ep = self.ep
        tool = ep.role("search")
        shape = ep.shapes.get(tool or "") or {}
        props = shape["properties"] if isinstance(shape.get("properties"), dict) else {}
        paging = ep.lookout.page_arg()
        if not (props and paging and is_number(bound)):
            return None
        described = next((str(tdef["function"].get("description") or "") for tdef in ep.toolset
                          if tdef["function"]["name"] == tool), "")
        token = ep.lookout.token_key()
        shelf_key, shelf_val = self.shelf_argument(props, label, (token, paging[0]))
        bound_key = self.price_argument(props, paging[0])
        if not (shelf_key and bound_key):
            return None
        whole = props[bound_key].get("type") == "integer" or float(bound).is_integer()
        args = {shelf_key: shelf_val, bound_key: int(bound) if whole else bound}
        modes = self.paired_modes(props, shelf_key, list(map(str.lower, re.split(r"[.;]", described))))
        if modes is None:
            return None
        args.update(modes)
        if not all(key in args for key in shape.get("required") or []):
            return None
        fixed, faults = ArgCheck.execute(args, shape)
        return None if faults or not isinstance(fixed, dict) else fixed

    @staticmethod
    def shelf_argument(props, label, skip):
        wanted = label.strip().lower()
        for key, spec in props.items():
            if not isinstance(spec, dict) or key in skip or not re.search(r"categor|shelf|department|aisle", str(key).lower()):
                continue
            options = spec.get("enum")
            if isinstance(options, list):
                for option in options:
                    if isinstance(option, str) and option.strip().lower() == wanted:
                        return key, option
            elif spec.get("type") == "string":
                return key, label
        return None, None

    @staticmethod
    def price_argument(props, page_key):
        for key, spec in props.items():
            if not isinstance(spec, dict) or spec.get("type") not in ("integer", "number") or spec.get("enum") or key == page_key:
                continue
            lowered = str(key).lower()
            if re.search(r"ceil|max|price|cap$|budget|upper|most", lowered) and not re.search(r"page|first|offset|skip|count|size|num", lowered):
                return key
        return None

    @staticmethod
    def paired_modes(props, shelf_key, sentences):
        shelf_pattern = r"\b" + re.escape(shelf_key.lower()) + r"\b"
        chosen = {}
        for key, spec in props.items():
            if key == shelf_key or not isinstance(spec, dict):
                continue
            options = spec.get("enum")
            if not isinstance(options, list) or not 2 <= len(options) <= 4:
                continue
            match = None
            for option in options:
                option_pattern = r"\b" + re.escape(str(option).lower()) + r"\b"
                if any(re.search(option_pattern, part) and re.search(shelf_pattern, part) for part in sentences):
                    match = option
                    break
            if match is None:
                return None
            chosen[key] = match
        return chosen

    def seed_signature(self, args, label, paging):
        mode = dict(self.ep.lookout.search_keys(args)[1])
        page_key = paging[0] if paging else None
        fixed_modes = sorted((m, v) for m, v in mode.items() if not (isinstance(args.get(m), str) and args[m] == label))
        number_key = ""
        for name in args:
            if is_number(args[name]) and name != page_key:
                number_key = name
                break
        return json.dumps([fixed_modes, number_key], default=str)

    def just_below(self, price):
        if not is_number(price):
            return None
        whole_only = (self.price_spec() or {}).get("int")
        value = float(price)
        if value.is_integer():
            lower = int(value) - 1
        elif whole_only:
            lower = int(value)
        else:
            lower = round(value - 0.01, 2)
        return lower if lower > 0 else None

    def rewalk_dead_aim(self):
        ep = self.ep
        if not self.cheapest_rule() or ep.shop.done or ep.time_left() < CLOSING + 60 or not ep.role("search"):
            return None
        aimed = self.aimed_at or {}
        gone = ep.shop.dead_lines() | ep.refused_lines
        opens = 2 - bool(ep.shop.batch_tool())
        for k, (aim, ident) in list(aimed.items()):
            if not ident or ident not in gone:
                continue
            if self.price_spec() is None or ep.lookout.page_arg() is None:
                continue
            pick = ep.bargain.walk_pick(k)
            if not pick or pick[1] <= aim + 1e-9 or ep.moves_left() < FIRST_ORDER_BY + 3 + opens:
                continue
            del aimed[k]
            (self.put_aims or {}).pop(k, None)
            ep.log.append({"kind": "ceil_rearm", "product": k, "dead": ident, "aim": pick[1]})
            return ep.bargain.run_walk([], FIRST_ORDER_BY, [k], repair=True)
        return None

    def walk_hold(self, call_args):
        ep = self.ep
        try:
            blocked = not self.cheapest_rule() or self.queued_walk or ep.time_left() < CLOSING + 60
            if blocked or (ep.refusal_n and ep.moves_left() <= 15):
                return ""
            ids = list(ep.basket.units) if call_args is None else \
                [ident for ident, count in ep.basket.lines(call_args) if count is None or count > 0]
            product_index = ep.bargain.product_index
            only = set(product_index(ident) for ident in ids if ident)
            floor = 4 if ep.orders_tried != 0 else FIRST_ORDER_BY + 1
            half = (ep.moves_left() - floor - 1) // 2
            spend = half - (1 if ep.shop.batch_tool() else 2)
            ceil, room = self.hold_aims(ids, only, spend)
            plan = ep.bargain.walk_plan(min(WALK_TOTAL, room), only)
            if not (plan or ceil):
                return ""
            self.queued_walk = (plan, max(floor, ep.moves_left() - half - 1), ceil)
            return self.hold_text(plan, ceil)
        except Exception as err:
            ep.log.append({"kind": "walk_hold_error", "error": f"{type(err).__name__}: {err}"[:300]})
            return ""

    def hold_aims(self, ids, only, room):
        product_index = self.ep.bargain.product_index
        ceil = []
        for k in sorted(only, key=lambda x: -1 if x is None else x):
            if room < 2 or not self.can_list_below(k):
                continue
            ceil.append(k)
            room -= CEIL_CALLS
            prices = [self.known.price(ident) for ident in ids if ident and product_index(ident) == k]
            costs = [float(price) for price in prices if is_number(price)]
            if not costs:
                continue
            held = self.put_aims.get(k)
            self.put_aims[k] = min(costs + [float(held)]) if is_number(held) else min(costs)
        return ceil, room

    def hold_text(self, plan, ceil):
        book = self.ep.page_book
        free_text = self.ep.lookout.free_text
        steps = []
        queries = [" ".join(free_text(book[key]["args"])) for _, key, _ in plan if key in book]
        if queries:
            steps.append("reads the remaining pages of " + ", ".join("'" + q + "'" for q in queries))
        if ceil:
            steps.append("uses the store's price filter to list the category's items below the pick's price")
        return ("Not run yet (no step used). The cheapest is wanted, yet this store orders search results by fixed id "
                "order, not by price, so some cheaper fit may lie on pages nobody read: the code " + " and ".join(steps)
                + (" now, then reads the lowest-priced unread rows below the pick. Then issue the call again, or the "
                   "same call with a cheaper fit."))

    def ceiling_walk(self, k, floor, repair=False):
        ep = self.ep
        bargain = ep.bargain
        spec, paging = self.price_spec(), ep.lookout.page_arg()
        home = bargain.walk_home(k)
        label = self.shelf_spelling(home)
        aim = self.aim_price(k)
        if not (spec and paging and label) or aim is None:
            return None
        self.walked_products.add(k)
        pick_now = bargain.walk_pick(k)
        self.aimed_at[k] = (float(aim), pick_now[0] if pick_now else "")
        bound = self.just_below(aim)
        if bound is None:
            return None
        state = {"floor": self.walk_floor(floor, repair), "depth": self.reachable_depth(), "keys": [], "calls": 0, "top": None,
                 "complete": False, "ran": False, "last": bound, "bound": bound, "spec": spec, "paging": paging, "home": home}
        what = "the cheapest row seen"
        if pick_now or is_number((self.put_aims or {}).get(k)):
            what = "the pick"
        say = f"Cheapest wanted: listing '{label}' priced under {what} ({aim:g}) through the store's price filter."
        rounds = CEIL_CALLS + (CEIL_MORE if self.burst_room()[1] > 8 + CEIL_MORE else 0)
        while state["calls"] < rounds and self.may_page(state["floor"]):
            if not self.ceiling_round(state, label, say):
                break
        if not state["ran"]:
            return None
        return self.walk_report(k, state["keys"], home, aim, label, state["last"], state["complete"], what, state["calls"])

    def walk_report(self, k, keys, home, aim, label, last, complete, what, calls):
        book = self.ep.page_book
        seen = set()
        for key in keys:
            seen |= set(self.on_home(book.get(key, {}), home))
        unread = [ident for _, ident in self.cheapest_unread(seen, k, aim)]
        valued = self.value_led_rows(k)
        ordered = [ident for ident in unread if ident in valued]
        others = [ident for ident in unread if ident not in valued]
        texty = list(filter(lambda ident: self.row_writes_value(ident, k), others))
        ordered += texty
        if not self.value_search_complete(k):
            ordered += [ident for ident in others if ident not in texty]
        self.ep.log.append({"kind": "ceiling_walk", "product": k, "label": label, "aim": aim, "calls": calls, "complete": complete,
                            "rows": len(seen), "ranked": ordered[:CEIL_OPENS]})
        return self.walk_summary(label, aim, last, complete, what, seen, unread), ordered[:CEIL_OPENS]

    def value_led_rows(self, k):
        if not self.all_words():
            return set()
        bargain = self.ep.bargain
        home = bargain.walk_home(k)
        found = set()
        for entry in self.ep.page_book.values():
            if entry.get("id_order") is not True:
                continue
            if not self.cg_listing(entry["args"]) and bargain.walk_carries(entry["args"], k):
                found.update(self.on_home(entry, home))
        return found

    def value_search_complete(self, k):
        try:
            if not self.all_words():
                return False
            home = self.ep.bargain.walk_home(k)
            single = set()
            for n, req in self.own_requirements(k):
                options = [p for p in re.split(r"[,;/]|\bor\b|\beither\b|\bnor\b", self.plain_value(req)) if p.strip()]
                if len(options) == 1:
                    single.add(n)
            for entry in self.ep.page_book.values():
                if self.finished_value_walk(entry, home, k, single):
                    return True
            return False
        except Exception:
            return False

    def own_requirements(self, k):
        several = bool(self.wants.items()) and k is not None
        product_of = self.ep.talk.product_of
        return [(n, req) for n, req in enumerate(self.wants.props_left())
                if not several or product_of(req.partition(":")[0]) in (k, None)]

    def finished_value_walk(self, entry, home, k, single):
        if entry.get("id_order") is not True or self.cg_listing(entry["args"]) or not entry.get("first"):
            return False
        if self.ep.lookout.next_page(entry) is not None:
            return False
        if len(entry.get("row_shelf") or {}) < entry["total"]:
            return False
        return bool(self.on_home(entry, home) and self.ep.bargain.walk_carries(entry["args"], k) & single)

    @staticmethod
    def plain_value(req):
        return re.sub(r"\((?:stated|revealed[^)]*)\)", "", req.partition(":")[2]).lower()

    @staticmethod
    def walk_summary(label, aim, last, complete, what, seen, unread):
        if not complete:
            return f"the price filter on '{label}' below {aim:g}: {len(seen)} of the cheaper listings seen ({len(unread)} unread, in stock and within the budget)"
        if not seen:
            return f"the price filter on '{label}' below {aim:g}: no listing of it costs less than {what}"
        return (f"the price filter on '{label}': every listing priced up to {last:g} seen ({len(seen)}, {len(unread)} of them unread, "
                "in stock and within the budget)")

    def row_writes_value(self, ident, k):
        stem = self.ep.lookout.stem
        wanted, figures = self.wanted_words(k)
        if not (wanted or figures):
            return False
        body = " ".join(self.texts_of(ident))
        for figure, unit in figures:
            if re.search(r"(?<![\d.,])" + re.escape(figure) + r"\s*-?\s*" + re.escape(stem(unit)), body):
                return True
        have = set(map(stem, re.findall(r"[a-z0-9]{3,}", body)))
        return any(set(words) <= have for words in wanted)

    def wanted_words(self, k):
        bargain, stem = self.ep.bargain, self.ep.lookout.stem
        wanted, figures = [], []
        values = [self.plain_value(req) for _, req in self.own_requirements(k)]
        for value in filter(lambda text: not ref_text(text), values):
            for part in re.split(r"[,;/]|\bor\b|\beither\b|\bnor\b", value):
                figure = bargain.fig_amount(part)
                if figure:
                    figures.append(figure)
                    continue
                if bargain.amount_part(part):
                    continue
                stems = list(map(stem, re.findall(r"[a-z0-9]{3,}", part)))
                if stems:
                    wanted.append(stems)
        return wanted, figures

    def texts_of(self, ident):
        texts = []

        def gather(fields):
            if any(isinstance(value, str) and value == ident and "id" in str(key).lower() for key, value in fields.items()):
                texts.extend(str(value).lower() for value in fields.values() if isinstance(value, str))
        for record in self.known.items:
            walk_dicts(record, gather)
        return texts

    def walk_floor(self, floor, repair):
        if repair:
            return floor + 2
        if self.ep.orders_tried:
            return floor
        return max(floor, self.ep.brief.order_due() + 2)

    def ceiling_round(self, state, label, say):
        book = self.ep.page_book
        spec, bound = state["spec"], state["bound"]
        args = self.filter_args(spec, label, bound)
        key = json.dumps(args, sort_keys=True, default=str)
        state["calls"] += 1
        stored = key in book and bool(book[key].get("first")) and key in (self.totals_seen or {})
        if not stored and not self.ep.bargain.walk_call(args, say):
            return False
        state["ran"], state["last"] = True, bound
        state["keys"].append(key)
        entry = book.get(key)
        if entry and any(is_number(self.known.price(ident)) and self.known.price(ident) > bound
                         for ident in entry.get("row_shelf") or {}):
            self.broken_modes.add(spec["sig"])
        total = None if self.price_spec() is None else self.totals_seen.get(key)
        if total is None or self.price_spec() is None:
            return False
        if total and entry:
            return self.narrow_from_listing(state, entry, args, total)
        if self.cheaper_in_stock(state["home"], bound):
            self.broken_modes.add(spec["sig"])
            return False
        top = state["top"]
        if top is None:
            state["complete"] = True
            return False
        middle = self.midpoint(bound, top)
        if bound < middle < top:
            state["bound"] = middle
            return True
        return False

    @staticmethod
    def filter_args(spec, label, bound):
        limit = int(bound) if spec.get("int") or float(bound).is_integer() else bound
        return {**spec["tmpl"], spec["shelf"]: label, spec["bound"]: limit}

    def cheaper_in_stock(self, home, bound):
        lookout, ids = self.ep.lookout, self.known.ids
        state = {"hit": False}

        def check(fields):
            if state["hit"] or lookout.shelf_of(fields) not in home:
                return
            if any(isinstance(v, str) and v in ids and "id" in str(key).lower() for key, v in fields.items()):
                price = self.row_price(fields)
                state["hit"] = price is not None and price <= bound and lookout.stock_state(fields) is True
        for record in self.known.items:
            walk_dicts(record, check)
        return state["hit"]

    def midpoint(self, a, b):
        whole = (self.price_spec() or {}).get("int") or (float(a).is_integer() and float(b).is_integer())
        if whole:
            return (int(a) + int(b)) // 2
        return round((float(a) + float(b)) / 2, 2)

    def narrow_from_listing(self, state, entry, args, total):
        bound, depth = state["bound"], state["depth"]
        pages = -(-total // max(1, entry["size"]))
        if (depth is None or total <= depth) and pages - len(entry["read"]) <= CEIL_CALLS - state["calls"]:
            state["calls"], state["complete"] = self.finish_listing(entry, args, state["paging"], state["calls"], state["floor"])
            return False
        state["top"] = bound
        lowered = self.lower_limit(entry, bound, total, state["calls"])
        if lowered < bound:
            state["bound"] = lowered
            return True
        return False

    def lower_limit(self, entry, bound, total, calls):
        prices = []
        for ident in entry.get("row_shelf") or {}:
            cost = self.known.price(ident)
            if is_number(cost):
                prices.append(float(cost))
        low = min(prices) if prices else 0.0
        fill = max(1, CEIL_CALLS - calls) * max(1, entry["size"])
        spread_point = low + (bound - low) * 1.8 * fill / max(total, 1)
        return min(self.midpoint(low, bound), self.midpoint(low, spread_point))

    def finish_listing(self, entry, args, paging, calls, floor):
        lookout = self.ep.lookout
        while lookout.next_page(entry) is not None and calls < CEIL_CALLS and self.may_page(floor):
            calls += 1
            after = lookout.next_page(entry)
            if not self.ep.bargain.walk_call({**args, paging[0]: after}, "I read the next page of the price-filtered listing."):
                break
            if lookout.next_page(entry) == after:
                break
        return calls, lookout.next_page(entry) is None

    def ceiling_sweep(self, tried):
        lookout = self.ep.lookout
        spec, paging = self.price_spec(), lookout.page_arg()
        if not spec or not paging or self.wants.items():
            return None, False
        rows = lookout.shelf_rows()
        shelves = lookout.shelf_labels()
        label = self.shelf_spelling(shelves)
        if not rows or not label:
            return None, False
        bound = self.just_below(min(row[0] for row in rows.values()))
        if bound is None:
            return None, True
        args = self.filter_args(spec, label, bound)
        key = json.dumps(args, sort_keys=True, default=str)
        return (args, False) if key not in tried else self._sweep_onward(args, key, paging, tried, shelves, bound)

    def _sweep_onward(self, args, key, paging, tried, shelves, bound):
        entry = self.ep.page_book.get(key)
        if not entry:
            return None, self.totals_seen.get(key) == 0 and not self.cheaper_in_stock(set(shelves), bound)
        after = self.ep.lookout.next_page(entry)
        if after is None:
            return None, len(entry.get("row_shelf") or {}) >= entry["total"]
        more = {**args, paging[0]: after}
        return (more, False) if json.dumps(more, sort_keys=True, default=str) not in tried else (None, False)

    def value_guard(self, call_name, call_args):
        return self.ep.guarded("value_guard_error", "", self._value_guard_note, call_name, call_args)

    def _value_guard_note(self, call_name, call_args):
        ep = self.ep
        if call_name != ep.role("details") or not self.cheapest_rule() or not self.all_words() or ep.brief.in_reserve():
            return ""
        ident = self.guarded_listing(call_args)
        if ident is None:
            return ""
        k = ep.bargain.product_index(ident)
        mark = (k, self.wants.version)
        if (k is None and self.wants.items()) or mark in self.guard_marks:
            return ""
        valued = self.value_led_rows(k)
        if not valued or ident in valued:
            return ""
        pick = ep.bargain.walk_pick(k)
        own_price = self.known.price(ident)
        costs = [pick[1]] if pick else []
        if is_number(own_price):
            costs.append(float(own_price))
        cheaper = self.cheapest_unread(valued, k, min(costs))[:6] if costs else []
        if cheaper:
            self.guard_marks.add(mark)
            return self.guard_text(ident, pick, cheaper)
        return ""

    def guarded_listing(self, call_args):
        ep = self.ep
        named = [v for v in call_args.values() if isinstance(v, str) and v in self.known.ids]
        if len(named) != 1:
            return None
        ident = named[0]
        if ident in self.known.read or ident in ep.basket.units:
            return None
        if ident in ep.brief.named_open() or ep.refs or self.wants.reading.get("references"):
            return None
        return ident

    @staticmethod
    def guard_text(ident, pick, cheaper):
        listed = ", ".join(f"{i} ({c:g})" for c, i in cheaper)
        than = " than it and the pick" if pick else " than it"
        return ("Not run (no step used). Cheapest is wanted, and all words are required by this store's keyword search, so rows shown by its searches led by a required value carry those words somewhere in their records; " f"{ident} is not among "
                "them, while these rows of theirs cost less" + than + " and are unread: "
                + listed + ". Read them first, cheapest first; send this call again if you still want it.")

    def title_line(self, k, rec, template):
        stem = self.ep.lookout.stem
        avoid = self._title_avoid(k)
        avoided = {stem(word) for word in avoid}

        def fresh(word):
            return word not in avoid and stem(word) not in avoided

        tokens = re.findall(r"[a-z0-9]+", self.ep.shop.title_of(rec).lower())
        picked = None
        for at, token in enumerate(tokens):
            if re.fullmatch(r"[a-z]{3,}", token) and fresh(token):
                picked = at
                break
        if picked is None:
            return None
        text = tokens[picked]
        if picked + 1 < len(tokens) and fresh(tokens[picked + 1]):
            text += " " + tokens[picked + 1]
        if self.ep.bargain.ran_query(text):
            return None
        query = dict(template[0])
        query[template[1]] = text
        return query

    def _title_avoid(self, k):
        words = set(LINK_WORDS)
        for label in self.ep.bargain.walk_home(k):
            words.update(re.findall(r"[a-z]+", label))
        for req in self.wants.props_left():
            words.update(re.findall(r"[a-z0-9]+", req.lower()))
        products = self.wants.items() or [{"product": str(self.wants.reading.get("product") or "")}]
        for entry in products:
            words.update(re.findall(r"[a-z]+", str(entry.get("product") or "").lower()))
        return words


class ReadOrders:
    def __init__(self):
        self.open_queue = self.open_floor = self.open_tool = None
        self.open_say = self.open_purpose = ""
        self.open_recheck = False
        self.put_queue = self.put_tool = None


class Briefing:
    def __init__(self, ep):
        self.ep = ep
        self.read_content = self.read_at = None
        self.steps_given = None
        self.first_reads_done = None
        self.swap_done = None
        self.menu_memo = {}
        self.hint_alts = {}
        self.twin_asks = set()
        self.late_ids = set()
        self.late_q = self.backup_n = 0

    def read_request(self):
        ep = self.ep
        tool_lines = [f"- {fn['name']}: {fn.get('description', '')} | arguments: {as_text(fn.get('parameters') or {})}"
                      for fn in (entry["function"] for entry in ep.toolset)]
        request_part = "REQUEST:\n" + ep.request
        opening = ep.pview.get("opening")
        if opening is not None:
            request_part += "\n\nOPENING OBSERVATION:\n" + as_text(opening)
        self.read_content = [
            {"type": "text", "text": READ_REQUEST + "\n\nTOOLS:\n" + "\n".join(tool_lines), "cache_control": {"type": "ephemeral"}},
            {"type": "text", "text": request_part},
        ]
        reading = self.consult_readers(self.read_content)
        if not isinstance(reading.get("roles"), dict):
            reading = self.fallback_reading(reading)
        self.shielded(lambda: self.audit_reading(reading), "check_reading_error")
        self.use_reading(reading)

    def consult_readers(self, parts):
        ep = self.ep
        seats = READER + LISTENER[:1]
        reading = {}
        for round_no in range(3):
            for seat in seats:
                if round_no and self.readers_exhausted(seats):
                    return reading
                reply = ep.counsel.converse([{"role": "user", "content": parts}], [seat], 3000)
                reading = json_object((reply or {}).get("content") or "") or {}
                if isinstance(reading.get("roles"), dict):
                    return reading
                ep.log.append({"kind": "setup_retry", "model": seat})
        return reading

    def readers_exhausted(self, seats):
        counsel = self.ep.counsel
        if counsel.misses == 0 or counsel.call_fail == "refused" or counsel.silent():
            return True
        return not counsel.has_seat(seats) or self.ep.time_left() < CLOSING + 60

    def shielded(self, action, kind):
        try:
            action()
        except Exception as err:
            self.ep.log.append({"kind": kind, "error": f"{type(err).__name__}: {err}"[:300]})

    def use_reading(self, reading, again=False):
        ep = self.ep
        wants = ep.wants
        if not again:
            self.steps_given = ep.step_cap
        self.take_roles(reading, again)
        allowance = reading.get("message_budget")
        if is_number(allowance) and allowance >= 0:
            ep.msg_cap = int(allowance)
        wants.reading = reading
        try:
            extra = len(wants.items()) - 1
            ep.counsel.scale = 1.0 + 0.3 * min(2, max(0, extra))
        except Exception:
            ep.counsel.scale = 1.0
        self.sort_requirements(reading)
        self.shielded(lambda: self.move_story_values(reading), "story_values_error")
        self.shielded(ep.pointers.ref_register, "ref_register_error")
        if not again or wants.version <= 0:
            wants.spend_cap = reading.get("price_cap")
        if reading and ep.role("message") and wants.spend_cap is None:
            reading["ask_first"] = (["the most the shopper wants to spend, since the request states no price limit (skip if the same thing is listed again here)"]
                                    + [str(item) for item in reading.get("ask_first") or [] if item])
        self.settle_steps(reading, again)

    def take_roles(self, reading, again):
        ep = self.ep
        previous = ep.role("cart")
        offered = reading.get("roles")
        if not isinstance(offered, dict):
            offered = {}
        ep.role_map = {}
        for name, tool in offered.items():
            if isinstance(tool, str) and tool in ep.shapes:
                ep.role_map[name] = tool
        if again and ep.basket.used:
            if previous:
                ep.role_map["cart"] = previous
        else:
            ep.basket = Basket(ep.shapes.get(ep.role("cart") or "", {}))
        ep.wants.sweepable = bool(ep.role("search")) and ep.lookout.page_arg() is not None

    def sort_requirements(self, reading):
        wants = self.ep.wants
        stated = [str(req) for req in reading.get("requirements") or [] if req]

        def well_formed(req):
            head, colon, value = req.partition(":")
            return bool(colon and head.strip() and value.strip())
        wants.written = list(filter(well_formed, stated))
        loose = [req for req in stated if req not in wants.written]
        if loose:
            reading["soft_wants"] = [str(wish) for wish in reading.get("soft_wants") or [] if wish] + loose
            reading["requirements"] = list(wants.written)

    def settle_steps(self, reading, again):
        ep = self.ep
        if again and self.steps_given is not None:
            ep.step_cap = self.steps_given
        allowed = reading.get("step_budget")
        if is_number(allowed) and 0 < allowed < ep.step_cap:
            ep.step_cap = int(allowed)
        if again:
            ep.wants.version += 1

    def move_story_values(self, reading):
        hints = list(map(one_prop_hint, self.flat_hints(reading.get("hints"))))
        from_reader = hints[:]
        hints.extend(self.drop_story_guesses(reading))
        try:
            lifted = self.promote_hints(reading, from_reader)
        except Exception as err:
            self.ep.log.append({"kind": "written_hint_error", "error": f"{type(err).__name__}: {err}"[:300]})
        else:
            hints = [h for h in hints if h not in lifted]
        reading["hints"] = hints[:12]

    def drop_story_guesses(self, reading):
        ep = self.ep
        written = ep.wants.written
        listed = reading.get("references")
        refs = list(map(str, listed)) if isinstance(listed, list) else []
        guesses = [req for req in written if story_guess(req, ep.request, refs)]
        if len(written) < 2 or not guesses or 2 * len(guesses) > len(written):
            return []
        kept = [req for req in written if req not in guesses]
        ep.wants.written = kept
        reading["requirements"] = list(kept)
        self._strip_cores(reading, {self.core_of(req) for req in guesses})
        ep.log.append({"kind": "story_values", "moved": guesses})
        return [self._story_ask(req) for req in guesses]

    def _strip_cores(self, reading, gone):
        for row in reading.get("items") or []:
            if isinstance(row, dict) and isinstance(row.get("requirements"), list):
                row["requirements"] = [r for r in row["requirements"] if self.core_of(r) not in gone]

    @staticmethod
    def _story_ask(req):
        prop, _, value = req.partition(":")
        return f"{prop.strip()}: the story suggests {value.strip()} (not stated as a value; ask)"

    @staticmethod
    def core_of(req):
        head, _, value = str(req).partition(":")
        return re.split(r"\s*\(", head)[0].strip().lower(), value.strip().lower()

    def promote_hints(self, reading, hints):
        rows = [r for r in reading.get("items") or [] if isinstance(r, dict)]
        return [hint for hint in hints if self._lift_hint(hint, rows, reading)]

    def _lift_hint(self, hint, rows, reading):
        head, sep, detail = str(hint).partition(":")
        prop = self.bare_name(head)
        if not (sep and prop) or junk_key(prop):
            return False
        word, name = self.first_lead(detail, self.hint_names(head, rows, reading))
        if not word:
            return False
        mine = kind_words(name)
        if self.hint_collides(prop, word, mine):
            return False
        requirement = f"{head.strip()}: {word}"
        wants = self.ep.wants
        wants.written.append(requirement)
        reading["requirements"] = list(wants.written)
        for row in rows:
            if isinstance(row.get("requirements"), list) and (len(rows) == 1 or kind_words(row.get("product") or "") & mine):
                row["requirements"].append(requirement)
                break
        self.ep.log.append({"kind": "written_hint", "hint": str(hint)[:200], "requirement": requirement})
        return True

    @staticmethod
    def bare_name(prop):
        return re.split(r"\s*\(", str(prop))[0].strip()

    def first_lead(self, detail, names):
        for name in names:
            word = lead_value(detail, self.ep.request, name)
            if word:
                return word, name
        return "", ""

    def hint_names(self, head, rows, reading):
        tags = re.findall(r"\(([^)]*)\)", head)
        if tags:
            return [tags[-1]]
        return [str(r.get("product") or "") for r in rows] or [str(reading.get("product") or "")]

    def hint_collides(self, prop, word, mine):
        bounded = r"\b" + re.escape(word) + r"\b"

        def same_product(head):
            tags = re.findall(r"\(([^)]*)\)", head)
            return not (tags and mine) or bool(kind_words(tags[-1]) & mine)
        heads = (req.partition(":") for req in self.ep.wants.written)
        return any(same_product(head) and (same_prop(head, prop) or re.search(bounded, value.lower()))
                   for head, _, value in heads)

    @staticmethod
    def flat_hints(raw):
        if isinstance(raw, dict):
            pairs = [(key, val) for key, vals in raw.items() for val in (vals if isinstance(vals, list) else [vals])]
            raw = [f"{val}" if ":" in str(val) else f"{key}: {val}" for key, val in pairs if val]
        if not isinstance(raw, (list, tuple)):
            return []
        return [str(h).strip() for h in raw or [] if isinstance(h, (str, int, float)) and str(h).strip()]

    def fallback_reading(self, partial):
        ep = self.ep
        try:
            reading = dict(ep.closer.code_reading())
            reading.update((key, val) for key, val in partial.items() if key != "roles")
        except Exception as err:
            ep.log.append({"kind": "code_reading_error", "error": f"{type(err).__name__}: {err}"[:300]})
            reading = {}
        self.read_at = monotonic()
        ep.log.append({"kind": "code_reading", "roles": reading.get("roles"), "product": reading.get("product"),
                       "price_cap": reading.get("price_cap"), "items": len(reading.get("items") or [])})
        return reading

    def audit_reading(self, reading):
        ep = self.ep
        said = ep.request
        self.pin_figures(reading, said)
        steps = self.quoted_steps(reading, said)
        rule_texts = [str(rule) for rule in reading.get("rules") or [] if rule]
        if rule_texts and not steps:
            steps = self.rule_steps_by_llm(rule_texts)
        step_words = [set(plain_words(step["quote"])) for step in steps]

        def doubles_a_step(item):
            own = set(plain_words(item))
            floor = 0.6 * max(1, len(own))
            return any(words and len(words & own) >= floor for words in step_words)
        reading["ask_first"] = [item for item in reading.get("ask_first") or [] if not doubles_a_step(item)]
        reading["procedures"] = steps
        ep.procedures = steps
        ep.no_fit = self.quoted_no_fit(reading, said)
        ep.log.append({"kind": "no_fit", "rule": ep.no_fit})
        ep.pairs = self.fit_pairs(reading, said)
        ep.log.append({"kind": "pairs", "pairs": ep.pairs})
        ep.conds = self.cond_rules(reading, said)
        ep.log.append({"kind": "conds", "conds": ep.conds})
        ep.log.append({"kind": "procedures", "steps": steps})

    def pin_figures(self, reading, said):
        figures = reading.get("figures")
        if not isinstance(figures, dict):
            figures = {}
        for field in ("price_cap", "message_budget", "step_budget"):
            claimed = reading.get(field)
            if not is_number(claimed) or float(claimed) in written_figures(said):
                continue
            quote = str(figures.get(field) or "")
            hits = []
            if matching_sentence(quote, said):
                hits = [number for number in written_figures(quote) if number in written_figures(said)]
            kept = None
            if len(hits) == 1:
                kept = int(hits[0]) if hits[0] == int(hits[0]) else hits[0]
            reading[field] = kept
            self.ep.log.append({"kind": "read_check", "field": field, "read": claimed, "kept": kept})

    def quoted_no_fit(self, reading, said):
        rule = reading.get("no_fit")
        if not isinstance(rule, dict) or rule.get("closest") is not True:
            return None
        parts = [matching_sentence(part, said) for part in request_sentences(str(rule.get("quote") or ""))]
        sentence = " ".join(dict.fromkeys(parts)) if parts and all(parts) else ""
        if not sentence:
            self.ep.log.append({"kind": "read_check", "field": "no_fit", "dropped": str(rule.get("quote"))[:200]})
            return None
        slack = rule.get("slack")
        slack = float(slack) if is_number(slack) and 0 < float(slack) < 1 else 0.0
        if slack and round(slack * 100, 6) not in written_figures(sentence):
            slack = 0.0
        counts = " ".join(map(str, rule.get("counts") or [])).lower()
        choose = str(rule.get("choose") or "").strip()
        if choose and not matching_sentence(choose, said, 0.8):
            choose = ""
        return {"quote": sentence, "slack": slack, "stock": "stock" in counts, "budget": bool(re.search(r"budget|price|cost|spend", counts)),
                "choose": choose}

    def quoted_steps(self, reading, said):
        found = []
        for raw in reading.get("procedures") or []:
            step = {"quote": raw} if isinstance(raw, str) else raw
            if not isinstance(step, dict):
                continue
            sentence = matching_sentence(str(step.get("quote") or ""), said)
            if sentence:
                found.append({"quote": sentence, "needs_yes": step.get("needs_yes") is not False,
                              "final_cart": step.get("final_cart") is not False})
            else:
                self.ep.log.append({"kind": "read_check", "field": "procedure", "dropped": str(step.get("quote"))[:200]})
        return found

    @staticmethod
    def item_names(reading):
        products = {}
        for row in reading.get("items") or []:
            if isinstance(row, dict):
                products[len(products)] = " ".join(plain_words(row.get("product")))
        return products

    @staticmethod
    def item_spot(products, name):
        joined = " ".join(plain_words(name))
        spots = set()
        for n, known in products.items():
            if known and joined and (joined == known or joined in known or known in joined):
                spots.add(n)
        return spots.pop() if len(spots) == 1 else None

    def cond_rules(self, reading, said):
        products = self.item_names(reading)
        rules = []
        for entry in reading.get("if_then") or []:
            if not isinstance(entry, dict) or not matching_sentence(str(entry.get("quote") or ""), said):
                continue
            when = entry.get("when") if isinstance(entry.get("when"), dict) else {}
            dep, anchor = self.item_spot(products, entry.get("product")), self.item_spot(products, when.get("product"))
            needs = [str(v).strip() for v in entry.get("values") or [] if str(v).strip()]
            keys = [str(v).strip() for v in when.get("values") or [] if str(v).strip()]
            prop, aprop = str(entry.get("property") or "").strip(), str(when.get("property") or "").strip()
            if None in (dep, anchor) or dep == anchor or not (needs and keys and prop and aprop):
                continue
            rules.append({"dep": dep, "prop": prop, "needs": needs, "anchor": anchor, "aprop": aprop, "keys": keys,
                          "unless": entry.get("unless") is True, "quote": matching_sentence(str(entry["quote"]), said)})
        return rules

    def fit_pairs(self, reading, said):
        products = self.item_names(reading)

        def place(name):
            return self.item_spot(products, name)
        pairs = []
        quoted = (entry for entry in reading.get("fit_with") or [] if isinstance(entry, dict))
        for entry in quoted:
            if not matching_sentence(str(entry.get("quote") or ""), said):
                continue
            pair = (place(entry.get("product")), place(entry.get("with")))
            if None in pair or pair[0] == pair[1] or pair in pairs:
                continue
            pairs.append(pair)
        return pairs

    def rule_steps_by_llm(self, rule_texts):
        ep = self.ep
        numbered = "\n".join(f"{n}. {rule}" for n, rule in enumerate(rule_texts))
        reply = ep.counsel.converse([{"role": "user", "content": STEP_CHECK + "\n\nRULES:\n" + numbered}], LISTENER, 300)
        parsed = json_object((reply or {}).get("content") or "") or {}
        valid = [n for n in parsed.get("steps") or [] if isinstance(n, int) and 0 <= n < len(rule_texts)]
        sentences = filter(None, (matching_sentence(rule_texts[n], ep.request) for n in valid))
        steps = [{"quote": sentence, "needs_yes": True, "final_cart": True} for sentence in sentences]
        ep.log.append({"kind": "rule_steps", "rules": len(rule_texts), "steps": steps})
        return steps

    def hint_mark(self, hint):
        def marked():
            prop = self.hint_head(hint).strip()
            return f"{hint} [no key]" if self.plain_hint(self.slot_for(prop), prop, hinted=True) else str(hint)
        return self.quietly(str(hint), marked)

    @staticmethod
    def quietly(default, work, *args):
        try:
            return work(*args)
        except Exception:
            return default

    def plain_hint(self, k, prop, hinted=False):
        def judge():
            name = self.bare_name(prop)
            if not name:
                return False
            if not hinted and not self.is_hinted(k, name):
                return False
            if self.menu_line(k, name) or self.reads_for(k) < 2:
                return False
            for line in self.menu_now(k).splitlines():
                if same_prop(self.line_key(line), name):
                    return False
            return True
        return self.quietly(False, judge)

    def menu_line(self, k, prop):
        def lookup():
            memo = self.menu_memo.get(k)
            if not memo or memo[0] != len(self.ep.known.items):
                memo = (None, self.fresh_menu(k))
                self.menu_memo[k] = (len(self.ep.known.items), memo[1])
            for line in memo[1].splitlines():
                if strict_prop(self.line_key(line), prop):
                    return line
            return ""
        return self.quietly("", lookup)

    def fresh_menu(self, k):
        known, shop = self.ep.known, self.ep.shop
        if not self.multi_product():
            own = shop.own_shelf()
            return (known.prop_menu(own) if own else known.prop_menu())[0]
        homes = shop.home_shelves()
        home = set() if k is None or k >= len(homes) else homes[k]
        first = next((name for name in known.shelves_read() if name.strip().lower() in home), None)
        return "" if first is None else known.prop_menu(first)[0]

    def multi_product(self):
        return bool(self.ep.wants.items())

    @staticmethod
    def line_key(line):
        return line.strip("- ").split(" (in ")[0].strip()

    def is_hinted(self, k, prop):
        name = str(prop).split("(")[0].strip()
        loose = k is None or not self.multi_product()
        heads = (self.hint_head(hint) for hint in self.hints())
        return any(strict_prop(head, name) and (loose or self.on_slot(head, k)) for head in heads)

    @staticmethod
    def hint_head(hint):
        return str(hint).split(":", 1)[0]

    def hints(self):
        return self.ep.wants.reading.get("hints") or []

    def on_slot(self, text, k):
        return self.ep.talk.product_of(text) in (k, None)

    def reads_for(self, k):
        def count():
            known, shop = self.ep.known, self.ep.shop
            if not self.multi_product():
                own = shop.own_shelf()
                return known.rich_reads(own) if own else known.rich_reads()
            homes = shop.home_shelves()
            home = homes[k] if k is not None and k < len(homes) else set()
            total = 0
            for name in home:
                total += known.rich_reads(name)
            return total
        return self.quietly(0, count)

    def menu_now(self, k):
        return (self.menu_memo.get(k) or (0, ""))[1]

    def slot_for(self, text):
        return self.ep.talk.product_of(text) if self.multi_product() else None

    @staticmethod
    def big_words(name):
        return {w for w in prop_words(name) if len(w) >= 3 and w not in LINK_WORDS}

    def distinct_ask(self, prop):
        def frame():
            target = self.twin_slot(prop)
            if target is None:
                return None
            slot, mark = target
            multi = self.multi_product()
            product = self.product_label(slot, "item") if multi else str(self.ep.wants.reading.get("product") or "item")
            field = self.bare_name(prop)
            text = self.choice_question(slot, field, product)
            if not text:
                field = self.twin_field(slot, prop)
                if not field:
                    return None
                text = self.choice_question(slot, field, product) or f"Which {field} would you like for the {product}?"
            self.twin_asks.add(mark)
            return (f"{field} ({product})" if multi else field), text
        return self.quietly(None, frame)

    def twin_slot(self, prop):
        if not self.multi_product():
            return None if -1 in self.twin_asks else (None, -1)
        slot = self.ep.talk.product_of(prop)
        if slot is None or slot in self.twin_asks:
            return None
        return slot, slot

    def choice_question(self, k, prop, product):
        def compose():
            line = self.menu_line(k, prop)
            if "):" not in line:
                return ""
            elsewhere = set()
            for other in filter(lambda row: "):" in row and row.strip() != line.strip(), self.menu_now(k).splitlines()):
                elsewhere.update(self.line_values(other))
            choices, flats = [], []
            for value in (raw.strip() for raw in line.split("):", 1)[1].split("|")):
                if not self.offerable(value):
                    continue
                flat = self.squash(value)
                if flat in elsewhere or flat in flats:
                    continue
                choices.append(value)
                flats.append(flat)
            *leading, last = choices[:3] or [""]
            if not leading:
                return ""
            return f"For the {product}, would you like {', '.join(leading)} or {last}?"
        return self.quietly("", compose)

    @staticmethod
    def squash(value):
        return " ".join(re.findall(r"[a-z0-9]+", value.lower()))

    @classmethod
    def line_values(cls, line):
        return [cls.squash(v) for v in line.split("):", 1)[1].split("|")]

    @staticmethod
    def offerable(value):
        if not value or len(value) > 30 or re.search(r"[,;/]", value):
            return False
        return bool(re.search(r"[a-z]", value.lower()))

    def product_label(self, k, fallback=""):
        wants = self.ep.wants
        if k is None or not self.multi_product():
            return str(wants.reading.get("product") or fallback)
        return wants.items()[k]["product"]

    def twin_field(self, k, prop):
        def hunt():
            self.menu_line(k, prop)
            menu = self.menu_now(k)
            own_kind = kind_tokens(self.product_label(k))
            story = self.story_for(k)
            spent = self.ep.talk.spent_asks
            prop_set = set(prop_words(prop))
            keys = [self.line_key(line) for line in menu.splitlines()]
            for key in keys:
                usable = key and not junk_key(key) and key.lower() not in spent
                if not usable or not self.twin_of(k, key, prop) or not self.plan_open(k, key) or self.nests(key, prop):
                    continue
                novel = [w for w in prop_words(key)
                         if len(w) >= 3 and w not in LINK_WORDS and w not in prop_set and not kind_in(w, own_kind)]
                if any(re.search(r"\b" + re.escape(w), story) for w in novel) or self.hint_mapped(k, key):
                    return key
            return ""
        return self.quietly("", hunt)

    def story_for(self, k):
        multi = self.multi_product()
        texts = [self.ep.request.lower()]
        for hint in self.hints():
            if multi and not self.on_slot(self.hint_head(hint), k):
                continue
            texts.append(str(hint).lower())
        return " ".join(texts)

    @staticmethod
    def nests(a, b):
        outer, inner = Briefing.big_words(a), Briefing.big_words(b)
        if outer and inner:
            return outer <= inner or inner <= outer
        return False

    def hint_mapped(self, k, prop):
        def search():
            name = self.bare_name(prop)
            needed = [w for w in prop_words(name) if len(w) >= 3 and w not in LINK_WORDS]
            narrow = self.multi_product() and k is not None
            for hint in self.hints():
                head, _, detail = str(hint).partition(":")
                if narrow and not self.on_slot(head, k):
                    continue
                if any(n and strict_prop(n, name) for n in [self.bare_name(head)] + self.alts_in(detail)):
                    return True
                if needed and all(re.search(r"\b" + re.escape(w), str(hint).lower()) for w in needed):
                    return True
            return False
        return self.quietly(False, search)

    @staticmethod
    def alts_in(detail):
        found = []
        for group in re.findall(r"\(or ([^()]+)\)", str(detail)):
            for piece in re.split(r",|\bor\b|/", group):
                if piece.strip():
                    found.append(piece.strip())
        return found

    def twin_of(self, k, a, b):
        if self.quietly(None, strict_prop, a, b):
            return False
        return self.quietly(False, lambda: self.twin_names(a, b) or self.same_field(k, a, b))

    @staticmethod
    def twin_names(a, b):
        if strict_prop(a, b):
            return False

        def near(x, y):
            return x == y or (min(len(x), len(y)) >= 4 and (x.startswith(y) or y.startswith(x)))
        words_a = [w for w in prop_words(a) if len(w) >= 3 and w not in LINK_WORDS]
        words_b = [w for w in prop_words(b) if len(w) >= 3 and w not in LINK_WORDS]
        heads_a = words_a[:-1] if len(words_a) > 1 else words_a
        heads_b = words_b[:-1] if len(words_b) > 1 else words_b
        if any(near(x, y) for x in heads_a for y in words_b):
            return True
        return any(near(x, y) for x in words_a for y in heads_b)

    def same_field(self, k, a, b):
        if strict_prop(a, b):
            return True

        def overlap():
            if not same_prop(a, b):
                return False
            return bool(self.field_values(k, a) & self.field_values(k, b))
        return self.quietly(False, overlap)

    def field_values(self, k, prop):
        line = self.menu_line(k, prop)
        return set(self.line_values(line)) - {""} if "):" in line else set()

    def plan_open(self, k, prop):
        talk = self.ep.talk
        try:
            hinted = self.is_hinted(k, prop)
            return talk.open_prop(k, prop, self.menu_line(k, prop) if hinted else "", by_samples=hinted)
        except Exception:
            return talk.open_prop(k, prop)

    def collide_alt(self, prop):
        def pick():
            target = self.twin_slot(prop)
            key = self.alt_field(target[0], prop) if target is not None else ""
            if not key:
                return ""
            self.twin_asks.add(target[1])
            return self.with_product(key, target[0])
        return self.quietly("", pick)

    def with_product(self, key, slot):
        if self.multi_product():
            return f"{key} ({self.ep.wants.items()[slot]['product']})"
        return key

    def alt_field(self, k, prop):
        def hunt():
            name = self.bare_name(prop)
            if not name:
                return ""
            multi = self.multi_product()
            for hint in self.hints():
                tagged_head, _, detail = str(hint).partition(":")
                head = self.bare_name(tagged_head)
                if not head or (multi and not self.on_slot(tagged_head, k)):
                    continue
                if not (self.same_field(k, name, head) or self.nests(name, head)):
                    continue
                spent = self.ep.talk.spent_asks
                key = next(filter(None, (self.usable_alt(k, alt, head, name, spent) for alt in self.alts_in(detail))), "")
                if key:
                    return key
            return ""
        return self.quietly("", hunt)

    def usable_alt(self, k, alt, head, name, spent):
        if junk_key(alt) or strict_prop(alt, head) or strict_prop(alt, name) or self.nests(alt, name):
            return ""
        line = self.menu_line(k, alt)
        key = line and self.line_key(line)
        blocked = not key or strict_prop(key, name) or key.lower() in spent
        if blocked or self.asked_for(k, key) or not self.plan_open(k, key):
            return ""
        return key

    def asked_for(self, k, prop):
        def scan():
            multi = self.multi_product()
            name = re.split(r"\s*\(", str(prop or ""))[0].strip()
            if not name:
                return False
            for line in self.ep.questions_log:
                label = re.sub(r"\s*\[[^\]]*\]", "", line.split(":", 1)[0].strip("- "))
                quoted = self.quoted_ask(line)
                parts = filter(None, (p.strip() for p in re.split(r";|\band\b", label)))
                for part in parts:
                    if multi and k is not None and self.log_slot(part, quoted) != k:
                        continue
                    if strict_prop(self.bare_name(part), name):
                        return True
            return False
        return self.quietly(False, scan)

    @staticmethod
    def quoted_ask(line):
        return re.search(r'asked "(.*?)" -> ', line)

    def log_slot(self, part, quoted):
        talk = self.ep.talk
        slot = talk.product_of(part)
        if slot is None and quoted:
            slot = talk.product_of(quoted.group(1))
        return slot

    def holds_plain(self, text, plain_q):
        stem = self.ep.lookout.stem
        skip = {"which", "would", "you", "like", "for", "the"}
        have = set(map(stem, re.findall(r"[a-z0-9]+", str(text).lower())))
        need = []
        for word in re.findall(r"[a-z0-9]+", str(plain_q).lower()):
            if word in skip or word in LINK_WORDS:
                continue
            need.append(stem(word))
        return bool(need) and set(need) <= have

    def alt_untried(self, k):
        def check():
            for tagged_head in map(self.hint_head, self.hints()):
                if not self.on_slot(tagged_head, k):
                    continue
                head = self.bare_name(tagged_head)
                if head and self.asked_for(k, head) and self.alt_field(k, head):
                    return True
            return False
        return self.quietly(False, check)

    def quiet_reask(self, prop):
        def pick():
            target = self.twin_slot(prop)
            if target is None or not self.hint_mapped(target[0], prop):
                return ""
            slot, mark = target
            pending = [h for h in self.open_hints(slot) if " drew no preference)" in h]
            if pending or (self.multi_product() and self.untouched_products(slot)):
                return ""
            twin = self.twin_field(slot, prop)
            if twin:
                self.twin_asks.add(mark)
                return self.with_product(twin, slot)
            return ""
        return self.quietly("", pick)

    def open_hints(self, k=None):
        reopened, fresh = [], []
        for hint in self.hints():
            prop = self.hint_head(hint).strip()
            slot = self.slot_for(prop) if prop else None
            if not prop or (k is not None and slot not in (k, None)):
                continue
            line = self.menu_line(slot, prop)
            if self.ep.talk.open_prop(slot, prop, line, by_samples=True) and not self.asked_within(slot, prop):
                fresh.append(str(hint))
            else:
                alt = self.alt_hint(slot, str(hint), prop)
                if alt:
                    reopened.append(alt)
        return reopened + fresh

    def alt_hint(self, k, hint, head):
        def build():
            if not self.remembered_alt(k, hint, head):
                return ""
            key = self.hint_alts.get(hint) or ""
            if not key or key.lower() in self.ep.talk.spent_asks or not self.plan_open(k, key):
                return ""
            name_part, _, detail = hint.partition(":")
            tag = re.search(r"\(([^()]*)\)\s*$", name_part)
            label = key if not tag else key + " (" + tag.group(1).strip() + ")"
            detail = re.sub(r"\s*\(or [^()]+\)", "", detail).strip()
            return f"{label}: {detail} (its first property, {head}, drew no preference)"
        return self.quietly("", build)

    def remembered_alt(self, k, hint, head):
        if self.hint_alts.get(hint):
            return True
        if not self.went_quiet(k, head):
            return False
        for alt in self.alts_in(hint.partition(":")[2]):
            if junk_key(alt) or strict_prop(alt, head):
                continue
            line = self.menu_line(k, alt)
            if line:
                self.hint_alts[hint] = self.line_key(line)
                break
        return True

    def went_quiet(self, k, head):
        multi = self.multi_product()
        matches = (asked for asked in self.ep.talk.quiet_asks or []
                   if self.same_field(k, asked, head) or self.nests(asked, head))
        return any(not multi or self.on_slot(asked, k) for asked in matches)

    def asked_within(self, k, head):
        def scan():
            multi = self.multi_product()
            for line in self.ep.questions_log:
                label = line.split(":", 1)[0].strip("- ")
                if "[not answered]" in label:
                    continue
                quoted = self.quoted_ask(line)
                for part in label.split(";"):
                    wrong_slot = multi and k is not None and self.log_slot(part, quoted) not in (k, None)
                    if not wrong_slot and self.nests(part, head):
                        return True
            return False
        return self.quietly(False, scan)

    def untouched_products(self, k):
        talk = self.ep.talk
        lines = self.ep.questions_log
        reached = set(talk.product_of(line.split(":", 1)[0]) for line in lines)
        reached.update(talk.product_of(q.group(1)) for q in map(self.quoted_ask, lines) if q)
        return any(j != k and j not in reached for j in range(len(self.ep.wants.items())))

    @staticmethod
    def menu_trim(menu, reads):
        if not menu or reads < 4:
            return menu
        busy = [line for line in menu.splitlines() if not re.search(r"\(in [01] listings?\)", line)]
        return "\n".join(busy) if busy else menu

    def first_reads(self):
        ep = self.ep
        try:
            if self.first_reads_done or ep.shop.batch_tool() or not ep.role("details"):
                return False
            self.first_reads_done = True
            queue = self.menu_read_queue()
            if not queue:
                return False
            orders = ep.reads
            orders.open_queue, orders.open_recheck = queue, True
            orders.open_say = "Before asking anything, I open a few listings per requested product to see what fields they have."
            orders.open_purpose = "menu"
            ep.log.append({"kind": "first_reads", "ids": queue})
            return True
        except Exception as err:
            ep.log.append({"kind": "first_reads_error", "error": f"{type(err).__name__}: {err}"[:300]})
            return False

    def menu_read_queue(self):
        ep = self.ep
        products = ep.wants.items()
        homes = ep.shop.home_shelves()
        counts = [row["quantity"] for row in products] or [ep.wants.quantity()]
        room = ep.moves_left() - self.order_due() - 2 * max(0, ep.ask_limit() - ep.msgs_sent) - 8
        plans, taken = [], []
        k = 0
        while k < (len(products) or 1):
            home = homes[k] if k < len(homes) else set()
            have = self.reads_for(k if products else None)
            if home and have < 2 and room > 0:
                picks = ep.shop.menu_rows(home, 2 - have, taken, counts[min(k, len(counts) - 1)])
                taken += picks
                if picks:
                    plans.append((have, k, picks))
            k += 1
        plans.sort()
        firsts = [picks[0] for _, _, picks in plans]
        seconds = [picks[1] for _, _, picks in plans if len(picks) > 1]
        return (firsts + seconds)[: max(0, room)]

    def order_due(self):
        cap = self.ep.step_cap
        floor = max(FIRST_ORDER_BY, min(FIRST_ORDER_DUE, cap // 4 + 1))
        due = max(floor, min(16, cap // 3))
        if self.eventful():
            due = max(due, self.busy_due())
        return max(floor, min(due, cap // 2))

    def eventful(self):
        if self.ep.event_seen:
            return True
        dead = self.ep.shop.dead
        if not dead:
            return False
        return any(line in dead for line in self.ep.basket.units)

    def busy_due(self):
        try:
            return 8 + max(1, len(self.ep.wants.items()))
        except Exception:
            return 9

    def settles(self, req, samples=""):
        prop, _, value = str(req).partition(":")
        low = value.lower()
        if ref_text(low) or re.search(r"as shown|same as|the page|sold by|redirect|refer", low):
            return True
        if not positive_value(value, prop) or self._story_only(req):
            return False
        if samples and not re.search(r"\d", value):
            words = [w for w in re.findall(r"[a-z]{3,}", low) if w not in LINK_WORDS]
            return not words or self.sample_echo(words, samples)
        return True

    def _story_only(self, req):
        wants = self.ep.wants
        listed = wants.reading.get("references")
        refs = list(map(str, listed)) if isinstance(listed, list) else []
        refs += wants.resolved or []
        heard = self.ep.request + " " + " ".join(wants.exchanges)
        return bool(story_guess(req, heard, refs))

    @staticmethod
    def sample_echo(words, samples):
        pool = set(re.findall(r"[a-z]+", samples))
        for word in words:
            stem = word if len(word) <= 4 else word[:-1]
            if re.search(r"\b" + re.escape(stem), samples):
                return True
            if len(word) >= 4 and any(edits_within(s, word, 1) for s in pool if len(s) >= 4):
                return True
        return False

    def offered(self):
        ep = self.ep
        tools = ep.toolset
        names = self.reserved_tools(ep.moves_left())
        if names is not None and self.narrowing():
            tools = [entry for entry in ep.toolset if entry["function"]["name"] in names] or tools
        messenger = ep.role("message")
        if ep.asking and messenger:
            tools = [entry for entry in tools if entry["function"]["name"] != messenger]
        return tools

    def reserved_tools(self, spare):
        ep = self.ep
        role = ep.role
        if spare <= 3:
            return {role("order"), role("decline")} | ({role("cart")} if spare > 1 else set())
        if not ((spare <= self.order_due() and ep.orders_tried == 0) or (spare <= FIRST_ORDER_BY and ep.refusal_n > 0)):
            return None
        names = {role("order"), role("decline"), role("cart"), role("cart_view"), role("view") or role("details"),
                 ep.shop.stock_tool()}
        wants_details = bool(ep.shop.swap_open())
        if ep.late_n < 3:
            loose = self.unverified() and spare > 5
            if loose or (self.named_open() and spare >= min(FIRST_ORDER_BY + 2, self.order_due())):
                wants_details = True
            if spare > 4 and (self.late_widening() or ep.counsel.named_pages or (self.unverified() and spare > 5)):
                names.add(role("search"))
        backed = spare >= 7 and self.cheaper_backups()
        if wants_details or backed:
            names.add(role("details"))
        return names

    def cheaper_backups(self):
        ep = self.ep
        if self.backup_n >= 2 or ep.wants.priority() != "cheapest":
            return []
        if not (ep.event_seen or ep.shop.dead_lines() or self.named_open()):
            return []
        found = []
        try:
            for k in ep.scout.cg_products():
                found.extend(self.below_fit(ep.shop.ed_pool(k)))
        except Exception as err:
            ep.log.append({"kind": "backup_error", "error": f"{type(err).__name__}: {err}"[:300]})
            return []
        return [ident for _, ident in sorted(found)][:6]

    def named_open(self):
        known = self.ep.known
        return {i for i in self.ep.counsel.named if i in known.ids} - known.read - self.ep.refused_lines

    @staticmethod
    def below_fit(pool):
        fitting = [e["price"] for e in pool if e["verified"] and e["fit"] and e["price"] is not None]
        if not fitting:
            return []
        ceiling = min(fitting)
        flags = ("read", "gone", "over", "pointer", "ruled", "clash")
        return [(e["price"], e["id"]) for e in pool
                if not any(e[flag] for flag in flags) and e["price"] is not None and e["price"] < ceiling]

    def unverified(self):
        ep = self.ep
        dead = ep.shop.dead_lines()
        stamp_now = str(ep.wants.version)
        live = []
        for key in ep.passed:
            thing, _, stamp = key.rpartition("|")
            if stamp != stamp_now or thing in ep.refused_lines or thing in dead:
                continue
            live.append(thing)
        if not live:
            return True
        return self.multi_product() and self.quietly(False, self.homes_unchecked, live)

    def homes_unchecked(self, live):
        shelves = set(map(self.ep.counsel.shelf_known, live))
        shelves.discard("")
        if not shelves:
            return False
        return any(home and not home & shelves for home in self.ep.shop.home_shelves())

    def late_widening(self):
        wants = self.ep.wants
        if wants.widened_at != wants.version:
            return False
        return self.ep.shop.tick - wants.widened_tick <= self.order_due()

    def narrowing(self):
        return bool(self.ep.role("order") and self.ep.role("cart"))

    def begin_batch(self):
        self.late_q, self.late_ids = 0, set()

    def late_hold(self, call_name, call_args):
        ep = self.ep
        details, search = ep.role("details"), ep.role("search")
        if call_name not in (details, search):
            if details and call_name in ep.shop.lookup_tools():
                return self.late_hold(details, call_args)
            return ""
        if not self.in_reserve():
            return ""
        named = {val for val in call_args.values() if isinstance(val, str)}
        is_details = call_name == details
        if is_details and named & (set(ep.basket.units) | {key.rpartition("|")[0] for key in ep.passed}):
            return ""
        spare = ep.moves_left() - self.late_q
        if self.late_pass(call_name, call_args, named, spare, is_details):
            self.late_q += 1
            self.late_ids.update(named)
            return ""
        extra = self.backup_ids() if is_details and spare - 1 >= 6 and self.backup_n < 2 else []
        note = ("Not run (no step used): every step left is held for placing the order. Still allowed: opening the "
                "line in the cart, any listing named by a check, or (while no fit is verified) up to 3 reads; put "
                "your best verified fit in the cart, re-check its stock and price, and order.")
        if extra:
            note += " Unread cheaper rows that could back the pick are still readable: " + ", ".join(extra) + "."
        return note

    def late_pass(self, call_name, call_args, named, spare, is_details):
        if self.ep.late_n + self.late_q < 3 and self.late_ok(call_name, call_args, named, spare):
            return True
        if not is_details:
            return False
        if spare - 1 >= 6 and self.backup_n < 2 and named & set(self.backup_ids()):
            self.backup_n += 1
            return True
        swap = self.ep.shop.swap_open()
        if swap and swap in named and spare > 2:
            self.swap_done = True
            return True
        return False

    def late_ok(self, call_name, call_args, named, spare):
        ep = self.ep
        if call_name == ep.role("details"):
            return (self.unverified() and spare > 5) or (
                bool(named & self.named_open()) and spare >= min(FIRST_ORDER_BY + 2, self.order_due()))
        if spare <= 4:
            return False
        if self.late_widening() or json.dumps(call_args, sort_keys=True, default=str) in ep.counsel.named_pages:
            return True
        return self.unverified() and spare > 5

    def backup_ids(self):
        return [i for i in self.cheaper_backups() if i not in self.late_ids][:2 - self.backup_n]

    def in_reserve(self):
        ep = self.ep
        spare = ep.moves_left()
        if not self.narrowing() or spare <= 3:
            return False
        return (ep.orders_tried == 0 and spare <= self.order_due()) or (ep.refusal_n > 0 and spare <= FIRST_ORDER_BY)


class Counsel(Assessor):
    def __init__(self, ep):
        self.llm = ProxyClient(timeout=70, max_retries=1, retry_delay=1)
        self.spend = 0.0
        self.scale = 1.0
        self.model_resume = 0.0
        self.last_reply = ep.t0
        self.misses = 0
        self.no_model = False
        self.mute = False
        self.banned = set()
        self.tool_banned = set()
        self.loud = set()
        self.quiet = False
        self.spare_models = []
        self.call_models = set()
        self.call_fail = ""
        Assessor.__init__(self, ep)

    def has_seat(self, models, tools=False):
        if self.mute or self.no_model:
            return False
        return any(name not in self.banned and not (tools and name in self.tool_banned)
                   for name in list(models) + self.spare_models)

    def ask_json(self, prompt, cap, models):
        answer = self.converse([{"role": "user", "content": prompt}], models, cap)
        return json_object((answer or {}).get("content") or "")

    def converse(self, msgs, models, cap, toolset=None):
        self.call_fail = ""
        self.call_models = set()
        if self.no_model or self.mute:
            return None
        extra = [name for name in self.spare_models if name not in models]
        ladder = list(models) + extra
        attempt = {"seed": "", "kinds": set(), "refused": set(), "ladder": ladder, "lone": self._seat_single(ladder)}
        for at, model in enumerate(ladder):
            if self.no_model:
                break
            skip = model in self.banned or model in attempt["refused"] or bool(toolset and model in self.tool_banned)
            if skip or ("sonnet" in model and self._seat_too_costly(at, models, msgs, cap)):
                continue
            finished, reply = self._seat_try(model, msgs, models, cap, toolset, attempt)
            if finished:
                return reply
        return self._seat_close(attempt["kinds"])

    def _seat_close(self, kinds):
        if "empty" in kinds:
            self.call_fail = "empty"
        else:
            self.call_fail = "refused" if kinds == {"hard"} else ""
        return None

    def _seat_single(self, ladder):
        return len(set(ladder) - self.banned) <= 1

    def _seat_try(self, model, msgs, models, cap, toolset, attempt):
        kinds = attempt["kinds"]
        plain = False
        for _ in range(1 if self.silent() else MODEL_TRIES):
            left = self.ep.time_left()
            if left < LAST_CALL - 10:
                return True, self._seat_close(kinds)
            self.llm.timeout = int(max(8, min(30 if self.silent() else 75, left - LAST_CALL + 15)))
            body = self._seat_body(model, msgs, cap, toolset, plain)
            self.call_models.add(model)
            began = monotonic()
            reply = self.ep.post(self.llm, "/inference/chat/completions", body)
            if isinstance(reply.data, dict) and reply.data.get("choices"):
                message = self._seat_answer(reply.data, model, kinds)
                return message is not None, message
            action, status = self._seat_fault(reply, model, models, toolset, attempt, plain)
            if action == "plain":
                plain = True
                if "reasoning" in body:
                    self.loud.add(model)
            elif action != "soft":
                return action
            elif monotonic() - began > 20:
                return (True, self._seat_close(kinds)) if self._seat_stalled(attempt) else (False, None)
            else:
                outcome = self._seat_wait(msgs, model, status, attempt)
                if outcome is not None:
                    return outcome
        return False, None

    def _seat_body(self, model, msgs, cap, toolset, plain):
        shaped = self._seat_flatten(msgs) if plain else self._seat_marked(msgs, model)
        body = {"model": model, "messages": shaped, "temperature": 0, "max_tokens": cap}
        if toolset:
            body.update(tools=toolset, tool_choice="auto")
        elif self.quiet and not plain and model not in self.loud and ("haiku" in model or model.startswith("google/")):
            body["reasoning"] = {"enabled": False} if model.startswith("anthropic/") else {"effort": "none"}
        return body

    @staticmethod
    def _seat_flatten(msgs):
        out = []
        for msg in msgs:
            line = dict(msg)
            content = line.get("content")
            if isinstance(content, list):
                pieces = [str(bit.get("text") or "") for bit in content if isinstance(bit, dict)]
                line["content"] = "\n\n".join(pieces)
            out.append(line)
        return out

    @staticmethod
    def _seat_marked(msgs, model):
        if not model.startswith("anthropic/") or len(msgs) < 3:
            return msgs
        copies = [dict(msg) for msg in msgs]
        for at in (0, len(copies) - 1):
            words = copies[at].get("content")
            if isinstance(words, str) and words:
                copies[at]["content"] = [{"type": "text", "text": words, "cache_control": {"type": "ephemeral"}}]
        return copies

    def _seat_fault(self, reply, model, models, toolset, attempt, plain):
        kinds = attempt["kinds"]
        self.misses += 1
        self.ep.log.append({"kind": "inference_failure", "model": model,
                            "error": str(reply.error)[:300] if reply.error else "empty"})
        fault = reply.error or {}
        status = fault.get("status")
        raw = str(fault.get("body") or "")
        if status == 402 or (status == 403 and re.search(r"key limit|insufficient|credit", raw.lower())):
            self.no_model = True
            return (True, self._seat_close(kinds)), status
        if status in (401, 403, 404):
            self._seat_ban(model, status, raw, toolset, models, attempt)
            return (False, None), status
        resend = status in (400, 413, 422)
        if resend and not plain:
            kinds.add("hard")
            return "plain", status
        if resend or (isinstance(status, int) and 400 <= status < 500 and status not in (408, 425, 429)):
            kinds.add("hard")
            attempt["refused"].add(model)
            return (False, None), status
        kinds.add("soft")
        return "soft", status

    def _seat_ban(self, model, status, raw, toolset, models, attempt):
        low = raw.lower()
        attempt["kinds"].add("hard")
        attempt["refused"].add(model)
        if status == 404 and toolset and "tool" in low:
            self.tool_banned.add(model)
        elif status != 403 or not re.search(r"moderat|flagged", low):
            self.banned.add(model)
        if status != 403:
            return
        listed = re.search(r'allowed_models"?\s*:\s*\[([^\]]*)', raw)
        if listed is None or self.spare_models:
            return
        names = re.findall(r'"([^"\s]{3,100})"', listed.group(1))
        self.spare_models = [name for name in names if name not in self.banned and name not in models][:2]
        attempt["ladder"] += [name for name in self.spare_models if name not in attempt["ladder"]]
        attempt["lone"] = self._seat_single(attempt["ladder"])

    def silent(self, at=None):
        if self.no_model:
            return True
        if self.misses > 0:
            when = monotonic() if at is None else at
            return when - self.last_reply > DOWN_AFTER
        return False

    def _seat_answer(self, data, model, kinds):
        self.last_reply = monotonic()
        self.misses = 0
        cost = (data.get("usage") or {}).get("cost")
        if is_number(cost):
            self.spend += float(cost)
        message = data["choices"][0].get("message") or {}
        if not (message.get("content") or message.get("tool_calls")):
            kinds.add("empty")
            self.ep.log.append({"kind": "inference_failure", "model": model, "error": "empty"})
            return None
        return message

    def _seat_wait(self, msgs, model, status, attempt):
        if not attempt["seed"]:
            attempt["seed"] = json.dumps(msgs[-1:], default=str)[-600:]
        delay = self.ep.backoff(attempt["seed"] + model, min(self.misses - 1, 6), status)
        if self.ep.time_left() - delay < LAST_CALL - 10:
            return True, self._seat_close(attempt["kinds"])
        self.model_resume = monotonic() + delay
        if self._seat_stalled(attempt, self.model_resume):
            return True, self._seat_close(attempt["kinds"])
        return None

    def _seat_stalled(self, attempt, at=None):
        return self.silent(at) and (attempt["lone"] or len(self.call_models) >= 2)

    def _seat_too_costly(self, at, models, msgs, cap):
        has_cheaper = at + 1 < len(models) and "sonnet" not in models[at + 1]
        estimate = sum(len(json.dumps(msg, default=str)) for msg in msgs) / 2.3 * 2e-6 + min(cap, 1500) * 1e-5
        if self.over_budget():
            return True
        if not has_cheaper:
            return False
        return self.spend >= SOFT_SPEND * self.scale or self.spend + estimate > SPEND_CAP * self.scale

    def over_budget(self):
        return self.spend >= SPEND_CAP * self.scale


class Shopfront(ShelfKeeper):
    def __init__(self, ep, bind):
        self.bind = dict(bind)
        self.world = ProxyClient(timeout=95, max_retries=1, retry_delay=1)
        self.tick = 0
        self.done = False
        self.broke = False
        ShelfKeeper.__init__(self, ep)

    def _post_settled(self, data, count):
        step = data.get("step")
        self.tick = step if isinstance(step, int) else self.tick + count
        if data.get("finished"):
            self.done = True
        return data

    def _step_conflict(self, raw, body, reused):
        if "finished" in raw:
            self.done = True
            return None
        expected = re.search(r"expected (\d+)", raw)
        if expected is not None:
            self.tick = int(expected.group(1)) - 1
            body["step"] = self.tick + 1
            return reused
        if reused >= 2 or "reused" not in raw:
            return None
        body["step"] += 1
        return reused + 1

    def perform(self, acts):
        single = len(acts) == 1
        body = dict(self.bind)
        body["step"] = self.tick + 1
        body["action" if single else "actions"] = acts[0] if single else acts
        retry = {"tries": 0, "reused": 0}
        for _ in range(3 + WORLD_TRIES):
            left = self.ep.time_left()
            if left < 2:
                break
            self.world.timeout = int(max(3, min(95, left - 1)))
            reply = self.ep.post(self.world, "/environment/call", body)
            if isinstance(reply.data, dict):
                return self._post_settled(reply.data, len(acts))
            delay = self._post_trouble(reply, body, retry)
            if delay is None or (delay >= 0 and self.ep.time_left() - delay < 2):
                break
            if delay >= 0:
                sleep(delay)
        return None

    def _post_trouble(self, reply, body, retry):
        fault = reply.error or {}
        code = fault.get("status")
        raw = str(fault.get("body") or "")
        self.ep.log.append({"kind": "env_failure", "status": code, "body": raw[:300]})
        if code == 402:
            self.broke = True
            return None
        if code == 409:
            retry["reused"] = self._step_conflict(raw, body, retry["reused"])
            return None if retry["reused"] is None else -1.0
        tries = retry["tries"]
        if code in (403, 422) or tries >= WORLD_TRIES - 1 or (tries >= 1 and code in (400, 401, 404)):
            return None
        retry["tries"] = tries + 1
        return self.ep.backoff(json.dumps(body, default=str)[-600:], tries, code)


class Episode:
    def __init__(self, pdata):
        env = pdata["environment"]
        self.t0 = monotonic()
        self.log = []
        self.last_post = self.calm_after = 0.0
        view = self.pview = env["policy_view"]
        self.counsel = Counsel(self)
        self.shop = Shopfront(self, env["binding"])
        self.request = str(view.get("query") or "")
        self._take_tools(view.get("tools") or [])
        self.step_cap = int(view.get("max_steps") or 30)
        self.shop.done = bool(view.get("finished"))
        self.known = Knowledge()
        self.wants = Wants(self)
        self.role_map = {}
        self.basket = Basket({})
        self.open_ledgers()
        LITERAL_VALUES.clear()
        self.reads = ReadOrders()
        for name, owner in (("brief", Briefing), ("talk", Talk), ("closer", Closer), ("lookout", Lookout),
                            ("bargain", Bargains), ("pointers", Pointers), ("scout", Scout)):
            setattr(self, name, owner(self))
        self.wants.ref_hook = self.pointers.ref_fix
        self.wants.field_hook = self.counsel.store_fields
        self.phase = "intake"
        self.rounds = self.idle = self.failures = self.seq_no = self.missed = 0
        self.last_ok = 0.0

    def _take_tools(self, offered):
        self.toolset = []
        self.shapes = {}
        for tool in offered:
            spec = tool.get("function") if isinstance(tool, dict) else None
            if not isinstance(spec, dict):
                continue
            self.toolset.append(tool)
            self.shapes[spec["name"]] = spec.get("parameters") or {"type": "object", "properties": {}}

    def open_ledgers(self):
        self.procedures = []
        self.no_fit = None
        self.pairs = []
        self.conds = []
        self.pair_told = {}
        self.confirmed_cart = None
        self.go_hold_mark = None
        self.go_verdict = ""
        self.word_kind = ""
        self.msg_cap = None
        self.asking = True
        self.batch_spec_known = False
        self.confirming = self.order_vetted = self.decline_vetted = self.audit_done = False
        self.msgs_sent = self.confirms = self.orders_tried = self.refusal_n = self.decline_refusals = 0
        self.asks_refused = 0
        self.clock_tried = False
        self.decline_said = {}
        self.closest_note = ""
        self.shortfalls = {}
        self.stuck = self.absorb_n = self.cart_seq = self.late_n = self.lead_words = 0
        for name in ("since_cart", "event_seen", "covering", "last_kind", "best_held", "early_audit", "recheck_mark",
                     "ref_decline_held", "slack_done", "decline_fill", "decline_kept", "decline_opens", "batch_cart",
                     "batch_proven", "batch_spec", "lead_ctx"):
            setattr(self, name, None)
        for name in ("rank_checked", "checked_adds", "passed", "refused_lines", "warned", "view_told", "near_fits"):
            setattr(self, name, set())
        for name in ("rulings", "refusals_seen", "pass_price", "page_book", "cheaper_told", "dead_told", "pointer_told",
                     "line_read", "line_seq", "puts", "lead_bounds"):
            setattr(self, name, {})
        for name in ("questions_log", "last_calls", "batch_puts", "ref_hits", "chat", "refs", "word_seen", "lead_seen"):
            setattr(self, name, [])

    def post(self, client, path, body):
        gap = SLOW_GAP if monotonic() < self.calm_after else PACE_GAP
        wait = self.last_post + gap - monotonic()
        if "/inference/" in path:
            wait = max(wait, self.counsel.model_resume - monotonic())
        wait = min(wait, self.time_left() - 1)
        if wait > 0:
            sleep(wait)
        self.last_post = monotonic()
        reply = client.post_verbose(path, body)
        if reply.data is None and (reply.error or {}).get("status") in (429, 503):
            self.calm_after = monotonic() + SLOW_FOR
        return reply

    def time_left(self):
        return DEADLINE - (monotonic() - self.t0)

    def backoff(self, seed, tries, status):
        jitter = (len(seed) * 7 + sum(ord(ch) for ch in seed[-64:]) + 31 * tries + 13 * self.shop.tick) % 101 / 100.0
        start = 2.0 if status == 429 else 1.0
        return min(8.0, start * 2 ** tries) * (0.75 + 0.5 * jitter)

    def execute(self):
        if self.shop.done:
            return self.log
        handlers = {"intake": self.on_intake, "round": self.on_round, "reply": self.on_reply,
                    "reads": self.on_reads, "sweep": self.on_sweep, "values": self.on_values,
                    "question": self.on_question, "hop": self.on_hop, "walk": self.on_walk,
                    "actor": self.on_actor, "closing": self.on_closing}
        while self.phase != "over":
            self.phase = handlers[self.phase]()
        return self.log

    def on_intake(self):
        self.open_chat()
        self.last_ok = monotonic()
        return "round"

    def open_chat(self):
        self.brief.read_request()
        self.log.append({"kind": "plan", "plan": self.wants.reading, "roles": self.role_map})
        opening = self.pview.get("opening")
        parts = ["SHOPPER REQUEST:\n" + self.request]
        if opening is not None:
            parts.append("OPENING OBSERVATION:\n" + as_text(opening))
            self.known.add(opening)
        summary = self.wants.overview(self.step_cap, self.msg_cap)
        if summary:
            parts.append(summary)
        parts.append(self.status_line())
        self.chat = [{"role": "system", "content": ACT_RULES}, {"role": "user", "content": "\n\n".join(parts)}]

    def status_line(self):
        spare = self.moves_left()
        text = self.status_head(spare)
        names = [tool["function"]["name"] for tool in self.brief.offered()]
        if len(names) < len(self.toolset):
            planned = "; questions to the shopper are planned and sent for you" if self.asking else ""
            text += "\n[tools usable now] " + ", ".join(names) + " (any other tool call is refused without using a step" + planned + ")."
        if self.basket.used:
            text += "\n" + self.basket.summary()
        return text + self.pair_text() + self.cond_text() + self.go_ahead_text() + self.status_tail(spare)

    @property
    def route(self):
        count = len(self.wants.items())
        return "single" if count <= 1 else "pair" if count == 2 else "basket"

    def moves_left(self):
        return self.step_cap - self.shop.tick

    def status_head(self, spare):
        counter = f"messages sent {self.msgs_sent}" + ("" if self.msg_cap is None else f" of {self.msg_cap}")
        bits = [f"[status] steps used {self.shop.tick} of {self.step_cap}, {spare} left", counter]
        clock = self.clock_text()
        if clock:
            bits.append(clock)
        needs_now = self.wants.text_body() + self.pointers.ref_line() + self.scout.product_steps()
        return "; ".join(bits) + ".\n[requirements now] " + needs_now

    def clock_text(self):
        left = max(0.0, self.time_left() - LAST_CALL)
        if left > (DEADLINE - LAST_CALL) / 2:
            return ""
        whole = int(left // CLOCK_STEP) * CLOCK_STEP
        return f"about {whole} s left" if whole > 0 else f"under {CLOCK_STEP} s left"

    def status_tail(self, spare):
        several = bool(self.wants.items())
        if spare <= 2:
            return (" LAST STEPS: when the cart has already a candidate for "
                    + ("every product asked for" if several else "the item") + ", buy it; otherwise add your strongest candidate (a verified fit, else the lowest-priced in-stock "
                                                                               "listing inside the budget that fails no requirement) and buy. Decline only when the records show "
                                                                               "that nothing can fit.")
        if spare <= 5 or self.time_left() < CLOSING:
            return (" FINISH NOW: set the cart to " + ("one line per product asked for" if several else "one line")
                    + " (a verified fit, else the lowest-priced in-stock listing inside the budget that fails no "
                      "requirement; drop each sold-out line), read its stock again, then buy. Decline only when the records "
                      "show that nothing can fit.")
        if self.counsel.over_budget():
            return (" MODEL BUDGET USED: stop exploring; in the coming few steps, put the best candidate for "
                    + ("every requested product (a line apiece, at its quantity)" if several else "the item asked for")
                    + " into the cart (a verified fit if one exists, or else the closest fit on record), read its stock and "
                      "price once more, and buy. A decline is correct only if the records prove that "
                    + ("some product has" if several else "the store has") + " no listing that can meet its requirements.")
        if self.orders_tried or not self.narrowing_now():
            return ""
        due = self.brief.order_due()
        if spare > due + 7:
            return ""
        if spare > due:
            return (f" Only a few steps remain: in the next two or so steps, put your best verified fit in the cart and buy; searching and detail reads end once {due} steps remain, and a refused order needs about 4 steps to repair.")
        return " Searching is over (reads are limited to the cart's own line, listings named by a check, or, while a product has no verified fit, 3 more reads or searches at most): put your top verified fit in the cart, read its stock and price again, and buy; a refused order still needs about 4 steps to repair."

    def narrowing_now(self):
        return self.brief.narrowing()

    def go_ahead_text(self):
        if not self.procedures or not self.role("message"):
            return ""
        if self.go_verdict != "yes" or self.confirmed_cart != self.cart_mark():
            return ("\n[go-ahead] Without the shopper's okay on the complete cart nothing gets bought; the code handles "
                    "this part: once your cart has the last basket, use the order tool, and the code then shows the "
                    "basket to the shopper via the spare message, reads the answer and orders only after that. Never ask "
                    "for the yes yourself, and never decline or empty the cart for want of it.")
        return ("\n[go-ahead] This basket was shown to the shopper, who left it as it is; that is the approval the "
                "request wants. Order at once, and do not change the cart unless a line stopped fitting.")

    def role(self, call_name):
        return self.role_map.get(call_name)

    def cart_mark(self):
        held = [(line, units) for line, units in self.basket.units.items() if units]
        return tuple(sorted(held))

    def cond_text(self):
        names = [str(row["product"]) for row in self.wants.items()]
        lines = []
        for rule in self.conds:
            if max(rule["dep"], rule["anchor"]) >= len(names):
                continue
            link = "unless" if rule["unless"] else "if"
            lines.append(f"the {names[rule['dep']]} needs {rule['prop']} {' or '.join(rule['needs'])} {link} the "
                         f"{names[rule['anchor']]} has {rule['aprop']} {' or '.join(rule['keys'])}")
        if not lines:
            return ""
        return "\n[linked rule] " + "; ".join(lines) + ". Read the deciding field on the item picked, then choose the other to suit."

    def pair_text(self):
        if not self.pairs:
            return ""
        names = [str(row["product"]) for row in self.wants.items()]
        count = len(names)
        rules = [f"the {names[dep]} has to work with the {names[anchor]}" for dep, anchor in self.pairs
                 if dep < count and anchor < count]
        if not rules:
            return ""
        anchors = dict.fromkeys(names[anchor] for _, anchor in self.pairs if anchor < count)
        return ("\n[fit] " + "; ".join(rules) + ". Pick the " + "/".join(anchors)
                + " first: its page states its own model name and number, and the page of a matching item names that "
                  "model as one it works with (a search for the model finds them).")

    def on_round(self):
        if self.rounds >= TURN_LIMIT:
            return "closing"
        self.rounds += 1
        if self.turn_blocked():
            return "closing"
        self.closer.catch_up()
        return "reply"

    def turn_blocked(self):
        shop = self.shop
        if shop.done or shop.broke or self.moves_left() <= 0:
            return True
        return self.time_left() < LAST_CALL

    def on_reply(self):
        due = self.wants.reply_due
        if due:
            self.wants.reply_due = ""
            if self.talk.reply_back(due):
                return "round"
        return "reads"

    def on_reads(self):
        return "round" if self.counsel.open_named() else "sweep"

    def on_sweep(self):
        if self.asking and not self.talk.empty_plans:
            return "values"
        return "round" if self.lookout.sweep_step() else "values"

    def on_values(self):
        return "round" if self.scout.value_pages() or self.scout.follow_pages() else "question"

    def on_question(self):
        if self.asking and self.talk.pose_planned():
            return "round"
        return "hop"

    def on_hop(self):
        return "round" if self.pointers.ref_step() else "walk"

    def on_walk(self):
        return "round" if self.scout.walk_step() else "actor"

    def on_actor(self):
        return "round" if self.actor_turn() else "closing"

    def actor_turn(self):
        seats = ACTOR[1:] + ACTOR[:1] if self.stuck >= STUCK_LIMIT else ACTOR
        reply = self.counsel.converse(self.chat, seats, 4000, self.toolset)
        if reply is None:
            return self.actor_missed(seats)
        self.last_ok = monotonic()
        self.missed = 0
        calls, self.seq_no = self.tool_calls(reply, self.seq_no)
        said = reply.get("content") or ""
        if not isinstance(said, str):
            said = as_text(said)
        binding = {self.role("cart"), self.role("order"), self.role("decline")} - {None}
        wants_bind = any(call["function"]["name"] in binding for call in calls)
        if self.asking and wants_bind and self.talk.pose_planned(urge=True):
            return True
        if calls:
            self.idle = 0
            self.log.append({"kind": "model", "content": said[:2000], "calls": [call["function"] for call in calls]})
            self.chat.append({"role": "assistant", "content": said or None, "tool_calls": calls})
            tick_before = self.shop.tick
            self.talk.post_replies(self.dispatch(calls))
            self.stuck = 0 if self.shop.tick != tick_before else self.stuck + 1
            return True
        self.idle += 1
        self.chat.append({"role": "assistant", "content": said or "(no tool call)"})
        self.chat.append({"role": "user", "content": "Each answer has to use at least one of the offered tools. Continue.\n"
                          + self.status_line()})
        return self.idle < IDLE_LIMIT

    def tool_calls(self, msg, seq_no):
        made = []
        for call in msg.get("tool_calls") or []:
            if isinstance(call, dict):
                seq_no += 1
                fn = call.get("function") or {}
                given = fn.get("arguments")
                text = given if isinstance(given, str) else json.dumps(given or {})
                made.append({"id": call.get("id") or f"call_{seq_no}", "type": "function",
                             "function": {"name": fn.get("name"), "arguments": text}})
        return made, seq_no

    def actor_missed(self, seats):
        counsel = self.counsel
        self.missed += 1
        if not counsel.has_seat(seats, True):
            return False
        quiet_long = monotonic() - self.last_ok > DOWN_AFTER
        if quiet_long and (self.missed >= 2 or (counsel.silent() and len(counsel.call_models) >= 2)):
            return False
        if counsel.call_fail in ("empty", "refused"):
            self.failures += 1
            return self.failures < 2
        return True

    def dispatch(self, proposals, scripted=False):
        self.brief.begin_batch()
        allowed = {entry["function"]["name"] for entry in self.brief.offered()}
        self.note_batch_puts(proposals)
        replies, queue = self._screen_calls(proposals, allowed, scripted)
        closing = self.finishers()
        if queue and queue[0][1] in closing:
            first_id, tool, args = queue[0]
            held, args = self.screen_finish(first_id, tool, args)
            if held:
                return replies + [(first_id, held)]
            queue[0] = (first_id, tool, args)
        elif queue and closing:
            keep = max(0, self.moves_left() - 1)
            replies += [(cid, "Not run (no step used): only one step remains, and it must be the order (or else the decline).")
                        for cid, _, _ in queue[keep:]]
            del queue[keep:]
        for start in range(0, len(queue), 16):
            chunk = queue[start:start + 16]
            if self.shop.done or self.shop.broke or self.moves_left() <= 0:
                replies += [(cid, "Not run: the episode has ended or no step is left.") for cid, _, _ in chunk]
            else:
                replies += self.send_chunk(chunk[:self.moves_left()])
        rank = {entry["id"]: n for n, entry in enumerate(proposals)}
        return sorted(replies, key=lambda pair: rank.get(pair[0], 0))

    def note_batch_puts(self, proposals):
        cart = self.role("cart")
        named = [(entry.get("function") or {}) for entry in proposals if isinstance(entry, dict)]
        self.batch_puts = []
        self.batch_cart = any(fn.get("name") == cart for fn in named)
        for entry in proposals:
            fn = entry.get("function") or {}
            if fn.get("name") != cart:
                continue
            given = fn.get("arguments")
            if not isinstance(given, dict):
                given = json_object(given) if isinstance(given, str) and given.strip() else {}
            if not isinstance(given, dict) or not self.basket.is_put(given):
                continue
            target = self.basket.aim(given)
            if target:
                self.batch_puts.append(target)

    def _screen_calls(self, proposals, allowed, scripted):
        replies = []
        queue = []
        messages = 0
        cart_stopped = False
        for entry in proposals:
            if cart_stopped and entry["function"]["name"] == self.role("cart"):
                replies.append((entry["id"], "Not run (no step used): an earlier cart edit in this reply was held back; read why, then send the "
                                             "cart changes again."))
                continue
            refusal, tool, args = self.screen(entry, allowed, scripted, len(proposals), messages)
            if refusal:
                cart_stopped = cart_stopped or (tool == self.role("cart") and isinstance(args, dict) and bool(args)
                                                and not self.basket.is_put(args))
                replies.append((entry["id"], refusal))
            else:
                messages += tool == self.role("message")
                queue.append((entry["id"], tool, args))
        return replies, queue

    def screen(self, proposal, permitted, scripted, batch, queued_messages):
        spec = proposal.get("function") or {}
        tool, raw = spec.get("name"), spec.get("arguments")
        early = self._entry_refusal(tool, raw, permitted, scripted)
        if early:
            return early, tool, {}
        args = self._parsed_args(raw)
        if args is None:
            return "Not run (no step used): the arguments did not form one JSON object.", tool, {}
        args, faults = ArgCheck.execute(args, self.shapes[tool])
        if faults:
            return "Not run (no step used): invalid arguments: " + "; ".join(faults[:6]), tool, args
        checks = (lambda: self.batch_rules(tool, args, scripted, batch, queued_messages),
                  lambda: self._repeat_note(tool, args),
                  lambda: "" if scripted else self.brief.late_hold(tool, args) or self.scout.value_guard(tool, args),
                  lambda: self.screen_cart(args, tool + as_text(args)) if tool == self.role("cart") else "")
        for check in checks:
            note = check()
            if note:
                return note, tool, args
        return None, tool, args

    def _entry_refusal(self, tool, raw, permitted, scripted):
        if tool not in self.shapes:
            return f"Not run (no step used): '{tool}' isn't a tool here. Use one of these: {', '.join(sorted(permitted))}."
        if scripted or tool in permitted:
            return None
        if self.asking and tool == self.role("message"):
            self.talk.want_ask(raw)
            self.asks_refused += 1
            onward = " or ".join(filter(None, (self.role("details"), self.role("search"))))
            count = (f" This is refused call {self.asks_refused} to {tool}; each one costs a turn and moves nothing."
                     if self.asks_refused >= 3 else "")
            return (f"Not run (no step used): {tool} is not yours to call now. The code frames and sends each shopper question "
                    "itself when its plan is ready (the one you drafted is noted), and the reply comes back to you here."
                    + (f" Meanwhile go on with {onward}: read more in-stock listings of the requested category." if onward else "")
                    + count)
        return ("Not run (no step used): not enough steps remain for it at this point; tools still allowed: "
                + ", ".join(sorted(permitted)) + ".")

    def _parsed_args(self, raw):
        if isinstance(raw, dict):
            return raw
        return json_object(raw) if (raw or "").strip() else {}

    def batch_rules(self, tool, call_args, scripted, batch, queued_messages):
        if batch > 1 and tool in self.finishers():
            return ("Not run (no step used): an order or a decline goes alone, sent only after you have seen what the "
                    "other calls returned.")
        if tool == self.role("message") and self.msg_cap is not None:
            kept = 0 if self.confirming else self.held_back()
            if self.msgs_sent + queued_messages >= self.msg_cap - kept:
                return "Not run (no step used): no message remains in the budget. Do not text the shopper again."
        if scripted or tool != self.role("cart_view") or tool == self.role("cart") or self.batch_cart:
            return ""
        if not self.basket.units:
            return ""
        shop = self.shop
        version = shop.cart_ver if shop.cart_ver is not None else -2
        fresh = self.since_cart if self.since_cart is not None else 1
        if fresh or shop.cart_shown != version or shop.cart_shown in self.view_told:
            return ""
        self.view_told.add(shop.cart_shown)
        return ("Not run (no step used): the latest cart answer shows the basket already - " + self.basket.summary()
                + " Viewing the cart tells nothing new; to check a line's stock, open its record with "
                + (shop.stock_tool() or "the tool that shows stock") + ".")

    def finishers(self):
        return {self.role("order"), self.role("decline")} - {None}

    def held_back(self):
        if not self.procedures or self.confirms:
            return 0
        return min(len(self.procedures), self._message_room())

    def _message_room(self):
        return ASK_DEFAULT if self.msg_cap is None else self.msg_cap

    def _repeat_note(self, tool, args):
        if tool in self.finishers() or tool in (self.role("message"), self.role("cart")):
            return ""
        if tool + as_text(args) not in self.last_calls:
            return ""
        return ("Not run (no step used): the same call just ran and nothing has moved since; what it returned stands "
                "above. Use other arguments or do something else.")

    def screen_cart(self, call_args, sig):
        cart = self.basket
        thing = cart.aim(call_args)
        adding = cart.is_put(call_args)
        pointer = self.pointers.ref_pointer_note(thing) if adding and thing else ""
        if pointer:
            seen = self.pointer_told.get(thing, 0)
            if seen < 2:
                self.pointer_told[thing] = seen + 1
                return f"Not run (no step used): {pointer}. Select another listing."
        note = self.passed_line_hold(call_args, thing, sig)
        if note:
            return note
        if not (adding and thing):
            return ""
        if cart.held(thing) > 0 and cart.adds is not False and sig not in self.warned:
            self.warned.add(sig)
            return (f"Not run (no step used): {thing} x{int(cart.held(thing))} sits in the cart already, and held lines get extra units from this cart tool, so it would increase the count. To order it, let the cart be; to change the units, add or remove only the difference. If adding is meant, send the same call again.")
        if not pointer:
            note = self.scout.walk_hold(call_args)
            if note:
                return note
        return self.check_new_lines(call_args, thing) or self.rank_lines(call_args) or self.unit_rules(call_args, thing, sig)

    def passed_line_hold(self, call_args, thing, sig):
        cart, wants = self.basket, self.wants
        mark = f"{sig}#{self.puts.get(thing, 0)}"
        own_price = self.known.price(thing) if thing else None
        cheapest = wants.priority() == "cheapest"
        swap = cheapest and (self._cheaper_in_batch(thing, own_price) or bool(self.counsel.cheaper_fit(thing)))
        moved = thing in self.pass_price and self.pass_price[thing] != own_price
        if swap or moved or not thing or cart.is_put(call_args):
            return ""
        if f"{thing}|{wants.version}" not in self.passed:
            return ""
        units = cart.count_arg(call_args)
        if units is not None and 0 < units < cart.held(thing):
            return ""
        if mark in self.warned or self.refusals_seen.get(thing, self.refusal_n) != self.refusal_n:
            return ""
        self.warned.add(mark)
        least = " It is also the cheapest verified fit." if wants.priority() == "cheapest" and not self.counsel.cheaper_fit(thing) else ""
        return (f"Not run (no step used): an independent check showed that {thing} meets each requirement now in force, reading every property with the value rules (when a field holds the value itself, only the store's words count); just the category label decides which kind of item it is, whatever the object the title or description mentions.{least} Since then nothing has changed. If a refusal or change excludes it, if a field of it clearly fails a requirement, or if you swap it for a cheaper listing, repeat the same call.")

    def _cheaper_in_batch(self, thing, own_price):
        for other in self.batch_puts:
            if other == thing or not is_number(own_price):
                continue
            price = self.known.price(other)
            if is_number(price) and self.known.price(other) < own_price:
                return True
        return False

    def check_new_lines(self, call_args, thing):
        version = self.wants.version
        lines = [line for line, count in self.basket.lines(call_args)[1:] if count is None or count > 0]
        lines.append(thing)
        for line in lines:
            key = f"{line}|{version}"
            if key in self.checked_adds or self.moves_left() <= 3:
                continue
            self.checked_adds.add(key)
            verdict = self.counsel.check_item(line, "ADD TO THE CART")
            if verdict:
                return "Not run (no step used). " + verdict + " If you're certain that it fits, send this call again."
        return ""

    def rank_lines(self, call_args):
        wanted = [line for line, count in self.basket.lines(call_args) if count is None or count > 0]
        if self.moves_left() > 3:
            for line in wanted:
                better = self.counsel.rank_check(line)
                if not better:
                    continue
                cheaper = self.counsel.rank_verify(line)
                note = better.replace("before ordering", "before putting it in the cart")
                note = note.replace("order the current pick", "send the same call again")
                if cheaper:
                    note += (f" An independent item check found that {cheaper} meets each requirement in force now while costing under {line}: put {cheaper} in its place.")
                return "Not run (no step used). " + note
        if self.wants.priority() != "cheapest":
            return ""
        told = self.cheaper_told
        for line in wanted:
            cheaper = self.counsel.cheaper_fit(line)
            key = ("cart", line, cheaper)
            if not cheaper or cheaper in self.basket.units or told.get(key):
                continue
            told[key] = 1
            return (f"Not run (no step used). The independent item check shows that {cheaper} satisfies every requirement in force now, and costs under {line}. The request wants the cheapest fit, and the check's verdict beats your own: put {cheaper} in instead. If a record shows {cheaper} failing a requirement now in force, send "
                    "the same call again.")
        return ""

    def unit_rules(self, call_args, thing, sig):
        cart, wants = self.basket, self.wants
        wanted = wants.quantity()
        asked = cart.count_arg(call_args)
        aim = wanted if cart.adds is False else wanted - cart.held(thing)
        if wants.items():
            counts, flat = wants.counts(), cart.list_mode(call_args) is None
            odd, short = [], []
            for line, count in cart.lines(call_args):
                if count is None or count <= 0:
                    continue
                slot = self.pair_slot(line)
                if slot is None or slot >= len(counts):
                    if int(count) not in counts:
                        odd.append((line, int(count)))
                    continue
                need = counts[slot] - (cart.held(line) if flat and cart.adds is not False else 0)
                key = cart.count_key(call_args) if flat else None
                figure = f" (the code reads this call's '{key}' figure as the unit count)" if key and key != cart.count_seen else ""
                if abs(count - need) > 1e-9:
                    short.append(f"{line} gets {int(count)} unit(s) here{figure}, but its product ({wants.items()[slot]['product']}) needs "
                                 f"{int(counts[slot])} and the line holds {int(cart.held(line))}")
            if sig in self.warned or not (odd or short):
                return ""
            self.warned.add(sig)
            if short:
                read = (f" The cart replies so far count this tool's '{cart.count_seen}' figure as units put in, whatever its "
                        "name says." if flat and cart.count_seen and cart.count_seen == cart.count_key(call_args) else "")
                return ("Not run (no step used): " + "; ".join(short) + "." + read + " Send the units that bring the line to its "
                        "product's quantity. If you are sure, send the same call again.")
            return (f"Not run (no step used): the requested products need {counts} units (a line apiece), but this call requests {odd}. If you are sure, send the same call again.")
        if asked is None or sig in self.warned or abs(asked - aim) <= 1e-9:
            return ""
        self.warned.add(sig)
        return (f"Not run (no step used): the request seeks a quantity of {wanted} from one listing, yet this call orders "
                f"{int(asked)}. If you are sure, send the same call again.")

    def screen_finish(self, cid, call_name, call_args):
        kind = self.call_kind(call_name, call_args)
        shop = self.shop
        plans = {"order": (self.sold_line_gate, self.basket_gate, self.pair_gate, self.cond_gate, self.unshown_need_gate, self.field_clash_gate, self.shelf_word_gate, self.cheaper_pick_gate, self.walk_gate,
                           self.stale_gate, self.vet_gate, self.go_ahead_gate),
                 "decline": (self.refused_decline_gate, self.closest_gate, self.go_ahead_decline_gate, shop.near_hold, self.kept_line_gate, self.slack_gate, self.fill_gate,
                             self.named_gate, self.pages_gate, self.audit_again_gate, self.first_audit_gate, self.fit_gate,
                             self.best_gate, self.soft_ref_gate, shop.decline_repair, shop.near_check,
                             self.unshown_order_gate, self.clock_order_gate)}
        if kind not in plans:
            return None, call_args
        if kind == "decline":
            self.decline_said = dict(call_args or {})
        held = next(filter(None, (gate() for gate in plans[kind])), None)
        if held:
            return held, call_args
        return None, call_args if kind == "order" else self.closer.scrub_decline(call_args, call_name)

    def call_kind(self, call_name, call_args):
        order = self.role("order")
        decline = self.role("decline")
        if not call_name or call_name not in (order, decline):
            return None
        if order != decline:
            return "decline" if call_name != order else "order"
        short = [text.lower() for text in (call_args or {}).values() if isinstance(text, str) and len(text) <= 40]
        for text in short:
            if re.search(DECLINE_WORDS, text):
                return "decline"
            if re.search(ORDER_WORDS, text):
                return "order"
        return "order"

    def sold_line_gate(self):
        if self.moves_left() <= 2:
            return ""
        shop = self.shop
        gone = sorted(set(self.basket.units) & shop.dead_lines())
        if not gone:
            return ""
        key = (tuple(gone), self.cart_seq)
        times = self.dead_told.get(key, 0)
        if times <= 1:
            self.dead_told[key] = times + 1
            return "Not run yet (no step used). " + shop.dead_note(gone)
        ready = times == 2 and self.role("cart") and self.moves_left() >= 6 and self.time_left() > CLOSING
        if not ready:
            return ""
        self.dead_told[key] = 3
        was_mute = self.counsel.mute
        keep = set(self.basket.units) - set(gone)
        done = self.guarded("dead_swap_error", False, self.closer.code_cart, set(gone) | shop.dead_lines(), keep)
        self.counsel.mute = was_mute
        self.log.append({"kind": "dead_swap", "lines": gone, "done": done, "cart": dict(self.basket.units)})
        if not done or not self.basket.units or set(self.basket.units) & shop.dead_lines():
            return ""
        return ("Not run yet (no step used). So far the cart had " + ", ".join(gone) + ", now sold out; the code took it out, set a candidate in its spot, left the remaining lines alone "
                                                                                       "and read every line's stock: "
                + self.basket.summary() + " Compare this cart against the request, then order it.")

    def guarded(self, kind, fallback, work, *args):
        try:
            return work(*args)
        except Exception as err:
            self.log.append({"kind": kind, "error": f"{type(err).__name__}: {err}"[:300]})
            return fallback

    def basket_gate(self):
        faults = self.basket_short()
        if not faults:
            return ""
        names = [str(row["product"]) for row in self.wants.items()]
        taken = {self.pair_slot(line) for line, units in self.basket.units.items() if units}
        missing = [name for at, name in enumerate(names) if at not in taken]
        extra = ""
        if missing and len(missing) < len(names):
            extra = " No line yet for: " + ", ".join(missing) + "."
        self.log.append({"kind": "basket_hold", "faults": faults, "missing": missing})
        listed = "; ".join(faults) + "." + extra
        head = "Lines are missing from the cart: " if missing or len(taken) < len(names) else "The cart lines do not match the request: "
        if self.moves_left() > 3:
            return ("Not run yet (no step used). " + head + listed
                    + " For each product lacking a line, find a fitting listing (a line apiece, at its quantity), then buy; "
                      "decline only when some product has no listing that can meet its requirements.")
        return ("Not run (no step used). Ordering without a product's line fails every time: " + listed
                + " Given the few steps, add the absent line at once if a fitting listing is known; otherwise decline.")

    def basket_short(self):
        counts = self.wants.counts()
        if len(counts) <= 1:
            return []
        if not self.basket.units:
            return ["the cart is empty"]
        return self.basket.basket_faults(counts)

    def pair_slot(self, line):
        label = str(self.known.label_of.get(line) or "").strip().lower()
        hits = [at for at, home in enumerate(self.shop.home_shelves()) if label and label in home]
        if len(hits) != 1:
            return None
        return hits[0]

    def pair_gate(self):
        if not self.pairs or self.moves_left() <= 3 or self.time_left() <= CHECK_FLOOR:
            return ""
        slots = {}
        for line, units in self.basket.units.items():
            at = self.pair_slot(line) if units else None
            if at is not None:
                slots.setdefault(at, []).append(line)
        texts = self.pair_cards()
        names = [str(row["product"]) for row in self.wants.items()]
        dead = self.shop.dead_lines()
        for dep, anchor, base, line, marks in self._pair_misses(slots, texts):
            key = (line, base, self.cart_seq)
            if self.pair_told.get(key, 0) < 1:
                self.pair_told[key] = 1
                return self._pair_hold_note(names, dep, anchor, base, line, marks, self._pair_named(texts, line, marks, dead))
        return ""

    @staticmethod
    def value_holds(shown, wanted):
        have = set(plain_words(shown))
        words = plain_words(wanted)
        return bool(words) and all(word in have for word in words)

    def cond_gate(self):
        if not self.conds or self.moves_left() <= 3 or self.time_left() <= CHECK_FLOOR:
            return ""
        slots = {}
        for line, units in self.basket.units.items():
            at = self.pair_slot(line) if units else None
            if at is not None:
                slots.setdefault(at, []).append(line)
        cards, _ = self.known.merge_cards()
        names = [str(row["product"]) for row in self.wants.items()]
        for rule in self.conds:
            for base in slots.get(rule["anchor"], []):
                shown = Pointers.field_value(cards.get(base) or {}, rule["aprop"])
                if not shown:
                    continue
                hit = any(self.value_holds(shown, key) for key in rule["keys"])
                if hit == rule["unless"]:
                    continue
                for line in slots.get(rule["dep"], []):
                    own = Pointers.field_value(cards.get(line) or {}, rule["prop"])
                    if own and any(self.value_holds(own, need) for need in rule["needs"]):
                        continue
                    key = ("cond", line, base, self.cart_seq)
                    if self.pair_told.get(key, 0) >= 1:
                        continue
                    self.pair_told[key] = 1
                    return self._cond_hold_note(rule, names, base, shown, line, own, cards)
        return ""

    def option_holds(self, shown, wanted):
        options = [bit for bit in re.split(r",|/|\bor\b", str(wanted).lower()) if bit.strip()]
        return any(self.value_holds(shown, option) for option in options or [wanted])

    def unshown_need_gate(self):
        if self.route == "basket" or self.moves_left() <= 3 or self.time_left() <= CHECK_FLOOR:
            return ""
        several = bool(self.wants.items())
        lines = [line for line, units in self.basket.units.items() if units] if several else [self.counsel.chosen()]
        cards, _ = self.known.merge_cards()
        dead = self.shop.dead_lines()
        for line in filter(None, lines):
            card = cards.get(line)
            slot = self.pair_slot(line) if several else 0
            if not card or slot is None:
                continue
            label = self.known.label_of.get(line)
            for prop, value in self.closer.code_values(slot):
                if Pointers.field_value(card, prop) or self.option_holds(as_text(card), value):
                    continue
                carriers = [(key, Pointers.field_value(other, prop)) for key, other in cards.items()
                            if key != line and self.known.label_of.get(key) == label]
                carriers = [(key, shown) for key, shown in carriers if shown]
                key = ("unshown", line, prop.lower(), self.cart_seq)
                if len(carriers) < 2 or self.pair_told.get(key, 0) >= 1:
                    continue
                self.pair_told[key] = 1
                return self._unshown_note(line, prop, value, carriers, dead)
        return ""

    def field_clash_gate(self):
        if self.route == "basket" or self.moves_left() <= 3 or self.time_left() <= CHECK_FLOOR:
            return ""
        several = bool(self.wants.items())
        lines = [line for line, units in self.basket.units.items() if units] if several else [self.counsel.chosen()]
        table = self.closer.code_rows()
        dead = self.shop.dead_lines()
        for line in filter(None, lines):
            row = table.get(line)
            slot = self.pair_slot(line) if several else 0
            if not row or slot is None:
                continue
            label = self.known.label_of.get(line)
            for prop, value in self.closer.code_values(slot):
                plain = str(value).lower()
                if re.search(r"\d", plain) or re.search(r"\b(?:" + VALUE_LINKS + r")\b", plain) or re.search(EXCLUDE_WORDS, plain):
                    continue
                stated = [text for key, text in row["fields"].items() if same_prop(key, prop) and str(text).strip()]
                key = ("clash", line, prop.lower(), self.cart_seq)
                if not stated or Closer.code_shows(row, prop, value, set(), True) or self.pair_told.get(key, 0) >= 1:
                    continue
                self.pair_told[key] = 1
                return self._clash_note(line, prop, value, stated[0], label, table, dead)
        return ""

    def _clash_note(self, line, prop, value, stated, label, table, dead):
        def cost(ident):
            price = self.known.price(ident)
            return price if is_number(price) else 1e18
        named = sorted((ident for ident, other in table.items()
                        if ident != line and ident not in dead and self.known.label_of.get(ident) == label
                        and other.get("stock") is not False and Closer.code_shows(other, prop, value, set(), True)), key=cost)
        self.log.append({"kind": "clash_hold", "line": line, "prop": prop, "value": value, "stated": stated, "named": named[:6]})
        head = (f"Not run yet (no step used). The shopper needs {prop} {value}, but {line}'s own {prop} field reads "
                f"\"{stated}\": the field named for the property decides, so a value elsewhere in the record does not count. ")
        if named:
            return head + f"Swap {line} for the cheapest of {', '.join(named[:6])} that meets every other requirement, then order."
        return head + (f"Search for listings whose {prop} field shows {value}, read them and swap {line} for one. Decline only if "
                       f"that search finds none; if you are sure {line} has it, send the order again.")

    def shelf_word_gate(self):
        if self.route != "single" or self.moves_left() <= 4 or self.time_left() <= CHECK_FLOOR or not self.role("search"):
            return ""
        product = str(self.wants.reading.get("product") or "").strip()
        words = set(kind_tokens(product))
        line = self.counsel.chosen() or next((held for held, units in self.basket.units.items() if units), None)
        label = self.known.label_of.get(line) if line else None
        key = ("shelf", line, self.cart_seq)
        if not words or not label or kind_score(words, set(kind_tokens(label)), set()) > 0 or self.pair_told.get(key):
            return ""
        home = {str(name).strip().lower() for name in (self.shop.home_shelves() or [set()])[0]}
        if home and str(label).strip().lower() not in home:
            return self._shelf_home_note(line, label, product, home, ("shelf_home", line, self.cart_seq))
        searched = [set(kind_tokens(" ".join(self.lookout.free_text(entry.get("args") or {}))))
                    for entry in self.page_book.values() if isinstance(entry, dict)]
        if any(words <= done for done in searched):
            return ""
        self.pair_told[key] = 1
        self.log.append({"kind": "shelf_hold", "line": line, "label": label, "product": product})
        return (f"Not run yet (no step used). The request asks for {product}, but {line} sits in the category '{label}', whose "
                f"name shares no word with it, and no search has used the full name yet. Search '{product}' once to see whether "
                f"the store files it under a category of its own; if a listing there meets every requirement, swap to it, "
                f"otherwise send the order again.")

    def _shelf_home_note(self, line, label, product, home, key):
        if self.pair_told.get(key):
            return ""
        self.pair_told[key] = 1
        cap, units = self.wants.spend_cap, self.wants.counts()[0]
        table = self.closer.code_rows()
        rows = sorted((row.get("price") if is_number(row.get("price")) else 10 ** 9, ident) for ident, row in table.items()
                      if str(self.known.label_of.get(ident) or "").strip().lower() in home and row.get("stock") is not False
                      and not (is_number(cap) and is_number(row.get("price")) and row["price"] * units > float(cap)))
        named = [ident for _, ident in rows][:6]
        self.log.append({"kind": "shelf_hold", "line": line, "label": label, "home": sorted(home), "named": named})
        where = "', '".join(sorted(home))
        return (f"Not run yet (no step used). The request asks for {product}, and the store has a category named for it ('{where}'), "
                f"but {line} sits in '{label}'. Open the in-budget listings of '{where}'" + (f" ({', '.join(named)})" if named else "")
                + f", and swap {line} for one that meets every requirement; send the order again only if none does.")

    def _unshown_note(self, line, prop, value, carriers, dead):
        def cost(key):
            price = self.known.price(key)
            return price if is_number(price) else 1e18
        named = sorted((key for key, shown in carriers if key not in dead and self.option_holds(shown, value)), key=cost)
        sample = ", ".join(f"{key}: {shown}" for key, shown in carriers[:2])
        self.log.append({"kind": "unshown_hold", "line": line, "prop": prop, "value": value, "named": named[:6]})
        head = (f"Not run yet (no step used). The shopper needs {prop} {value}, but {line} shows no {prop} at all, while other "
                f"listings of its category state it ({sample}). ")
        last = (f" Decline only if a search of the category then finds no listing whose {prop} shows {value}; if you are sure "
                f"{line} has it, send the order again.")
        if named:
            return head + f"Swap {line} for the cheapest of {', '.join(named[:6])} that meets every other requirement, then order."
        return head + f"Search for listings whose {prop} is {value}, read them and swap {line} for one." + last

    def _cond_hold_note(self, rule, names, base, shown, line, own, cards):
        dep_name = names[rule["dep"]] if rule["dep"] < len(names) else "item"
        anchor_name = names[rule["anchor"]] if rule["anchor"] < len(names) else "item"
        dead = self.shop.dead_lines()
        label = self.known.label_of.get(line)

        def cost(key):
            price = self.known.price(key)
            return price if is_number(price) else 1e18
        named = sorted((key for key, card in cards.items() if key not in dead and key != line and self.known.label_of.get(key) == label
                        and any(self.value_holds(Pointers.field_value(card, rule["prop"]), need) for need in rule["needs"])), key=cost)
        wanted = " or ".join(rule["needs"])
        self.log.append({"kind": "cond_hold", "line": line, "anchor": base, "shown": shown, "own": own, "named": named[:6]})
        head = (f"Not run yet (no step used). The request says: \"{rule['quote']}\" The {anchor_name} picked ({base}) shows "
                f"{rule['aprop']} {shown}, so the {dep_name} needs {rule['prop']} {wanted}; {line} shows "
                f"{(rule['prop'] + ' ' + own) if own else 'no ' + rule['prop'] + ' field'}. ")
        if named:
            return head + f"Swap {line} for the cheapest of {', '.join(named[:6])} that meets every other requirement, then order."
        return head + (f"Search for {dep_name} listings with {rule['prop']} {wanted}, read them and swap {line} for one that "
                       f"meets every other requirement, or pick a {anchor_name} that lifts this rule.")

    def _pair_misses(self, slots, texts):
        cards, _ = self.known.merge_cards()
        for dep, anchor in self.pairs:
            for base in slots.get(anchor, []):
                marks = self.pair_marks(base, texts)
                for line in slots.get(dep, []) if marks else []:
                    if not self.fit_shown(line, marks, texts, cards):
                        yield dep, anchor, base, line, marks

    @staticmethod
    def fit_shown(key, marks, texts, cards):
        card = cards.get(key) or {}
        named = " ".join(str(v) for k, v in card.items() if isinstance(v, str) and re.search(r"name|title", str(k).lower()))
        own = id_marks(named)
        shapes = {re.sub(r"\d", "9", mark) for mark in own}
        for piece in re.split(r"(?<=[.;!?])\s+|\n", texts.get(key, "")):
            if not any(mark in piece for mark in marks):
                continue
            others = id_marks(piece) - set(marks) - own
            if not any(re.sub(r"\d", "9", other) in shapes for other in others):
                return True
        return False

    def _pair_hold_note(self, names, dep, anchor, base, line, marks, named):
        dep_name = names[dep] if dep < len(names) else "item"
        anchor_name = names[anchor] if anchor < len(names) else "item"
        ordered = sorted(marks)
        self.log.append({"kind": "pair_hold", "line": line, "anchor": base, "marks": ordered, "named": named[:6]})
        head = (f"Not run yet (no step used). The {dep_name} has to work with the {anchor_name} picked: {base}'s own page "
                f"carries {', '.join(ordered[:2])}, and the page of {line} does not name it. ")
        if named:
            return head + (f"Read listings of the {dep_name} that name it: {', '.join(named[:6])}. Swap {line} for the cheapest one of them meeting each other requirement, then order.")
        return head + (f"Search the store for '{ordered[0]}' (search matches listing text), read the {dep_name} rows it "
                       f"shows, and swap {line} for one whose page names it.")

    def pair_cards(self):
        cards, _ = self.known.merge_cards()
        return {key: as_text(card) for key, card in cards.items()}

    def pair_marks(self, line, texts):
        labels = self.known.label_of
        own_label = labels.get(line)
        same_shelf = [text for key, text in texts.items() if labels.get(key) == own_label]
        found = set()
        for mark in id_marks(texts.get(line, "")):
            carriers = sum(1 for text in same_shelf if mark in text)
            if len(same_shelf) < 4 or carriers * 2 < len(same_shelf):
                found.add(mark)
        return found

    def _pair_named(self, texts, line, marks, dead):
        labels = self.known.label_of
        label = labels.get(line)
        cards, _ = self.known.merge_cards()
        picks = [key for key in texts
                 if key not in dead and key != line and labels.get(key) == label and self.fit_shown(key, marks, texts, cards)]

        def cost(key):
            price = self.known.price(key)
            return price if is_number(price) else 1e18
        return sorted(picks, key=cost)

    def cheaper_pick_gate(self):
        if self.moves_left() <= 3:
            return ""
        counsel = self.counsel
        line = counsel.chosen()
        rival = counsel.cheaper_fit(line) if line else ""
        if not rival:
            others = [held for held in self.basket.units if held != line]
            for other in others:
                rival = counsel.cheaper_fit(other)
                if rival:
                    line = other
                    break
        if not rival:
            return ""
        pair = (line, rival)
        times = self.cheaper_told.get(pair, 0)
        if times >= 2:
            return ""
        self.cheaper_told[pair] = times + 1
        return (f"Not run yet (no step used). The independent item check shows that {rival} satisfies each requirement, and it costs under {line}. The request wants the cheapest fit, and the check's verdict beats yours: swap "
                f"{line} out of your cart, put {rival} in, re-read its stock and price, and order it.")

    def walk_gate(self):
        return self.scout.walk_hold(None) if self.basket.units else ""

    def stale_gate(self):
        tool = self.shop.stock_tool()
        if not tool:
            return ""
        if self.moves_left() <= len(self.basket.units) + 2 or self.time_left() <= CLOSING:
            return ""
        unread = self.shop.stale_lines()
        key = (tuple(sorted(self.basket.units)), self.cart_seq)
        if not unread or key == self.recheck_mark:
            return ""
        self.recheck_mark = key
        first = unread[:4]
        queue = self.reads
        queue.open_queue = first
        queue.open_recheck = True
        queue.open_purpose = "cart"
        queue.open_tool = tool
        return ("Not run yet (no step used). Stock and prices can change at any moment, and since the latest cart "
                "change nobody has read these lines: " + ", ".join(first) + f". The code reads their stock for you now ({tool}, the tool that reports stock); then buy if each one is in stock still and the total fits the budget, else replace the line.")

    def vet_gate(self):
        if self.order_vetted or self.moves_left() <= 2 or self.time_left() <= CLOSING:
            return ""
        self.order_vetted = True
        counsel, wants = self.counsel, self.wants
        problems = self.basket.faults(wants.quantity(), wants.counts())
        if problems and self.moves_left() > 3:
            return "Not run yet (no step used). By count the cart isn't complete: " + "; ".join(problems) + ". Fix it, then order."
        several = bool(wants.items())
        if several:
            lines = list(self.basket.units)
        else:
            lines = [counsel.chosen()] if counsel.chosen() else []
        for line in lines:
            failure = counsel.check_item(line, "ORDER")
            if failure:
                return "Not run yet (no step used). " + failure
        for line in lines:
            ranking = counsel.rank_check(line)
            if ranking:
                return "Not run yet (no step used). " + ranking + self._cheaper_hint(line)
        scope = "for each requested product, at its quantity" if several else "the picked item in the required quantity"
        approval = " (when you call the order tool, the code asks the shopper to accept the cart)" if self.procedures else ""
        return ("Not run yet (no step used). Before the order, make sure: (1) the cart holds exactly one line " + scope + ", nothing besides; (2) since the newest cart change you read this item's stock and price again "
                                                                                                                          "through a tool showing both, it can be bought, and its price times the quantity remains within the "
                                                                                                                          "budget; (3) the item's own properties meet each written requirement, each need the shopper revealed "
                                                                                                                          "and each change they announced; (4) if the cheapest or best was asked for, no listing you saw fits "
                                                                                                                          "better; (5) every procedural rule of the request was followed" + approval + ". Verify whatever is unchecked, and when everything holds, call the order tool again.")

    def _cheaper_hint(self, line):
        rival = self.counsel.rank_verify(line)
        if not rival:
            return ""
        return (f" An independent item check found that {rival} meets each requirement in force now while costing under {line}: take {line} out, put {rival} in, read its stock and price anew, and order.")

    def go_ahead_gate(self):
        outcome = self.send_go_ahead()
        if not outcome:
            return ""
        note, said, verdict = outcome
        if verdict == "yes":
            return ""
        return ("Not run yet (one step used to message the shopper). The request asks for the shopper's approval of the final basket, so it went to them: \"" f"{note}\" They replied: \"{said}\" Follow their answer; any changed basket is put to them again before the order.")

    def send_go_ahead(self, room=3):
        mark = self.cart_mark() if self.procedures and self.role("message") else None
        if not mark or mark == self.confirmed_cart:
            return None
        if self.msg_cap is not None and self.msgs_sent >= self.msg_cap:
            return None
        if self.moves_left() < room or self.time_left() < LAST_CALL:
            return None
        note = self.basket_note()
        call = self.talk.message_call(note, f"goahead_{self.shop.tick}")
        if not call:
            return None
        earlier = self._needs_snapshot()
        answers = self._confirm_with(call)
        self.confirms += 1
        self.confirmed_cart = mark
        said = answers[0][1].split("\n[status]")[0] if answers else ""
        verdict = self.go_ahead(note, said)
        if earlier != self._needs_snapshot():
            verdict = "change"
        self.go_verdict = verdict
        self.log.append({"kind": "go_ahead", "note": note, "reply": said[:400], "verdict": verdict})
        return note, said, verdict

    def _needs_snapshot(self):
        return self.wants.text_body(), self.wants.spend_cap

    def _confirm_with(self, call):
        self.confirming = True
        try:
            return self.dispatch([call], scripted=True)
        finally:
            self.confirming = False

    def basket_note(self):
        rows = self.closer.code_rows()
        parts = []
        total = 0.0
        for line, units in sorted(self.basket.units.items()):
            if units:
                row = rows.get(line) or {}
                cost = row.get("price")
                total += units * (cost or 0.0)
                each = f" at {cost:g} each" if is_number(cost) else ""
                parts.append(f"{units:g} x {str(row.get('title') or line)[:90]} ({line})" + each)
        ending = ""
        if total:
            cap = self.wants.spend_cap
            within = f", within your limit of {cap:g}" if is_number(cap) and total <= cap else ""
            ending = f" Total {total:g}" + within + "."
        return "Final basket: " + "; ".join(parts) + "." + ending + " May I place this order? Please answer yes or no."

    def go_ahead(self, note, said):
        text = GO_AHEAD + "\n\nTHE ASSISTANT WROTE:\n" + note + "\n\nTHE SHOPPER REPLIED:\n" + said
        reply = self.counsel.converse([{"role": "user", "content": text}], LISTENER, 200)
        parsed = json_object((reply or {}).get("content") or "") or {}
        word = str(parsed.get("answer") or "").lower()
        return word if word in ("yes", "no", "change") else "yes"

    def refused_decline_gate(self):
        shop = self.shop
        if not self.decline_refusals or not self.role("order") or shop.done or shop.broke or self.moves_left() < 2:
            return ""
        alive = [line for line in self.basket.units if line not in shop.dead_lines()]
        named = [] if alive or self.wants.items() else self.named_fallback()
        quiet_before = self.counsel.mute
        try:
            if named and self.closer.code_put([(named[0], self.wants.counts()[0])]) != []:
                named = []
            if not alive and not named:
                return ""
            self.log.append({"kind": "refused_decline_order", "cart": dict(self.basket.units), "named": named[:1]})
            self.closer.finish_now()
        except Exception as err:
            self.log.append({"kind": "refused_decline_error", "error": f"{type(err).__name__}: {err}"[:300]})
            return ""
        finally:
            self.counsel.mute = quiet_before
        return ("Not run (no step used). The store already turned a decline down, which means something here can be bought, "
                "so the code ordered the closest fit instead of declining again: " + self.basket.summary())

    def unshown_order_gate(self):
        shop, wants = self.shop, self.wants
        if wants.items() or not self.role("order") or not self.role("cart") or shop.done or shop.broke or self.moves_left() < 3:
            return ""
        version = str(wants.version)
        alive = [line for line in self.basket.units if line not in shop.dead_lines()]
        for ident in list(dict.fromkeys(alive + self.named_fallback()))[:2]:
            if self.time_left() > CLOSING + 20 and f"{ident}|{version}" not in self.near_fits | self.passed:
                self.counsel.check_item(ident, "ORDER")
        table = self.closer.code_rows()
        cap, units = wants.spend_cap, wants.counts()[0]
        picks = []
        for stamp in self.near_fits | {x for x in self.passed if x.rpartition("|")[0] in alive}:
            ident, _, ver = stamp.rpartition("|")
            row = table.get(ident)
            if ver != version or not row or ident in shop.dead_lines() or row.get("stock") is False:
                continue
            if is_number(cap) and (not is_number(row.get("price")) or row["price"] * units > float(cap)):
                continue
            picks.append((row.get("price") if is_number(row.get("price")) else 10 ** 9, ident))
        if not picks:
            return ""
        held = [pick for pick in picks if pick[1] in self.basket.units]
        ident = min(held or picks)[1]
        quiet_before = self.counsel.mute
        try:
            if self.closer.code_put([(ident, units)]) != []:
                return ""
            self.log.append({"kind": "unshown_order", "item": ident, "near": sorted(self.near_fits)})
            self.closer.finish_now()
        except Exception as err:
            self.log.append({"kind": "unshown_order_error", "error": f"{type(err).__name__}: {err}"[:300]})
            return ""
        finally:
            self.counsel.mute = quiet_before
        return ("Not run (no step used). No record contradicts " + ident + " on any requirement; it only leaves some unstated, "
                "and declining loses for sure, so the code ordered it: " + self.basket.summary())

    def clock_order_gate(self):
        shop = self.shop
        if self.route != "basket" or self.clock_tried or not self.role("order") or not self.role("cart") or shop.done or shop.broke:
            return ""
        if self.decline_refusals or not CLOCK_FLOOR <= self.time_left() <= CLOSING + 30:
            return ""
        if self.moves_left() < len(self.wants.items()) + 2:
            return ""
        self.clock_tried = True
        quiet_before = self.counsel.mute
        try:
            picks = self.closer.code_picks(set(shop.dead_lines()))
            fit = self.closer.code_fit or {}
            if len(picks) < len(self.wants.items()) or not all(fit.get(ident) for ident, _ in picks):
                self.log.append({"kind": "clock_order", "picks": picks, "done": False, "why": "no fit for every product"})
                return ""
            if self.closer.code_put(picks) != [] or self.basket_short():
                self.log.append({"kind": "clock_order", "picks": picks, "done": False})
                return ""
            self.log.append({"kind": "clock_order", "picks": picks, "done": True, "cart": dict(self.basket.units)})
            self.closer.finish_now()
        except Exception as err:
            self.log.append({"kind": "clock_order_error", "error": f"{type(err).__name__}: {err}"[:300]})
            return ""
        finally:
            self.counsel.mute = quiet_before
        return ("Not run (no step used). Time is nearly up, and the records show a fit for every product, so the code put one "
                "line per product in the cart and ordered instead of declining: " + self.basket.summary())

    def closest_gate(self):
        shop, wants, rule = self.shop, self.wants, self.no_fit
        if not rule or wants.items() or not self.role("order") or not self.role("cart") or shop.done or shop.broke:
            return ""
        if self.moves_left() < 3:
            return ""
        version = str(wants.version)
        table = self.closer.code_rows()
        cap, units = wants.spend_cap, wants.counts()[0]
        home = set((shop.home_shelves() or [set()])[0]) or {shop.own_shelf()}
        alive = [line for line in self.basket.units if line not in shop.dead_lines()]
        pool = list(dict.fromkeys(alive + self.named_fallback() + list(self.counsel.audit_fits or [])
                                  + [x for x in self.known.read if (table.get(x) or {}).get("shelf") in home]))
        fresh = [x for x in pool if x in table and f"{x}|{version}" not in self.passed and f"{x}|{version}" not in self.shortfalls]
        for ident in fresh[:3]:
            if self.time_left() > CLOSING + 20:
                self.counsel.check_item(ident, "ORDER")
        scored = []
        for ident in pool:
            row, stamp = table.get(ident), f"{ident}|{version}"
            if not row or ident in shop.dead_lines() or not (stamp in self.passed or stamp in self.shortfalls):
                continue
            short = [] if stamp in self.passed else list(self.shortfalls[stamp])
            price = row.get("price") if is_number(row.get("price")) else None
            if row.get("stock") is False:
                if not rule["stock"]:
                    continue
                short.append("not in stock")
            if is_number(cap) and price is not None and price * units > float(cap):
                if not rule["budget"]:
                    continue
                short.append("over the budget")
            scored.append((len(short), price if price is not None else 10 ** 9, ident, short))
        if not scored:
            return ""
        fewest = min(entry[0] for entry in scored)
        nearest = sorted(entry for entry in scored if entry[0] == fewest)
        window = nearest[0][1] * (1 + rule["slack"])
        inside = ([entry for entry in nearest if entry[1] <= window] if rule["slack"] or not rule.get("choose") else nearest) or nearest[:1]
        held = [entry for entry in inside if entry[2] in alive]
        _, _, ident, short = (held or inside)[0]
        if rule.get("choose") and len(inside) > 1:
            picked = self.nearest_by_rule(rule, inside)
            ident, short = next(((e[2], e[3]) for e in inside if e[2] == picked), (ident, short))
        quiet_before = self.counsel.mute
        try:
            if self.closer.code_put([(ident, units)]) != []:
                return ""
            title = (table.get(ident) or {}).get("title") or ident
            self.closest_note = ("No listing meets every requirement, so as the request asks, this orders the one falling short on the "
                                 "fewest: " + f"{title} ({ident})" + (", short on: " + "; ".join(short)[:300] if short else "") + ".")
            self.log.append({"kind": "closest_order", "item": ident, "short": short, "scored": scored[:8]})
            self.closer.finish_now()
        except Exception as err:
            self.log.append({"kind": "closest_order_error", "error": f"{type(err).__name__}: {err}"[:300]})
            return ""
        finally:
            self.counsel.mute = quiet_before
        return "Not run (no step used). The request asks for the nearest listing when nothing fits, so the code ordered it: " + self.basket.summary()

    def nearest_by_rule(self, rule, inside):
        lines = []
        for _, price, ident, short in inside[:6]:
            record = json.dumps(self.known.about(ident), default=str)[:1500]
            lines.append(f"- {ident} (price {price:g}; falls short on: {'; '.join(short) or 'nothing'})\n{record}")
        prompt = (NEAREST_PICK + "\n\nTHE REQUEST'S RULE:\n" + rule["quote"] + "\nPICKING PART:\n" + rule["choose"]
                  + "\n\nLISTINGS:\n" + "\n".join(lines))
        answer = self.counsel.ask_json(prompt, 600, CHECKER) or {}
        picked = str(answer.get("pick") or "")
        self.log.append({"kind": "nearest_pick", "choose": rule["choose"][:200], "pick": picked, "among": [e[2] for e in inside]})
        return picked if picked in {e[2] for e in inside} else ""

    def named_fallback(self):
        shop, table = self.shop, self.closer.code_rows()
        cap = self.wants.spend_cap
        units = self.wants.counts()[0]
        out = []
        for value in self.decline_said.values():
            for bit in re.split(r"[\s.,;:!?()\[\]{}<>\"'`]+", str(value)):
                row = table.get(bit)
                if not row or bit in out or bit in shop.dead_lines() or row.get("stock") is False:
                    continue
                if is_number(cap) and is_number(row.get("price")) and row["price"] * units > float(cap):
                    continue
                out.append(bit)
        return out

    def go_ahead_decline_gate(self):
        held = self.cart_mark()
        mark = (held, self.wants.version)
        if not self.procedures or self.go_verdict == "no" or not held:
            return ""
        if self.go_hold_mark == mark or self.moves_left() <= 1:
            return ""
        self.go_hold_mark = mark
        lines = ", ".join(f"{line} x{units:g}" for line, units in held)
        return ("Not run yet (no step used). Approval from the shopper isn't lacking: once you call the order tool, "
                "the code puts the cart before the shopper in the message saved for it, and buys only if they accept "
                "it. In the cart now: "
                + lines + ". Order with this cart; a decline fits only if some line clearly breaks a requirement in force and "
                          "nothing in the records can replace it.")

    def kept_line_gate(self):
        gone = self.shop.dead_lines
        version = self.wants.version
        good = [line for line in self.basket.units
                if f"{line}|{version}" in self.passed and line not in self.refused_lines and line not in gone()]
        if not good and self.route == "single" and not self.decline_kept and self.moves_left() > 3:
            out = self._passed_out(version, gone())
            if out:
                self.decline_kept = True
                self.log.append({"kind": "kept_out", "item": out})
                return (f"Not run yet (no step used). The independent check judged {out} to satisfy each requirement in force, "
                        f"yet it is no longer in the cart. Put it back at the requested quantity, read its stock and price again, "
                        f"and buy it. Decline only if its own record clearly shows it failing a requirement now in force.")
        if not good or self.moves_left() <= 1 or self.decline_kept:
            return ""
        self.decline_kept = True
        each = " with each product's line in your cart" if self.wants.items() else ""
        return ("Not run yet (no step used). The cart holds " + ", ".join(good) + ", which the independent check judged to satisfy each requirement in force. Declining throws it away: "
                                                                                  "read its stock and price again, and buy it" + each
                + ". Decline only if a change since then clearly rules it out now.")

    def _passed_out(self, version, gone):
        table = self.closer.code_rows()
        cap, units = self.wants.spend_cap, self.wants.counts()[0]
        picks = []
        for stamp in self.passed:
            ident, _, ver = stamp.rpartition("|")
            row = table.get(ident)
            if ver != str(version) or not row or ident in gone or ident in self.refused_lines or row.get("stock") is False:
                continue
            price = row.get("price")
            if is_number(cap) and (not is_number(price) or price * units > float(cap)):
                continue
            picks.append((price if is_number(price) else 10 ** 9, ident))
        return min(picks)[1] if picks else ""

    def slack_gate(self):
        shop = self.shop
        rows = []
        if not self.slack_done and (self.role("details") or shop.batch_tool()):
            rows = shop.slack_rows()
        batch = shop.batch_tool()
        if not rows or not self._decline_room(rows, batch, 4):
            return ""
        self.slack_done = True
        self.reads.open_queue = rows
        self.reads.open_say = "Before declining, I read cheaper rows that could let all products fit the budget at once."
        total = shop.slack_total
        cost = f"{total:g}" if is_number(total) else "more than the budget"
        return ("Not run yet (no step used). Each product has a fitting listing, but together the cheapest fits cost " + cost
                + f", above the budget of {self.wants.spend_cap}. Nobody opened the rows priced within what the budget leaves each product after the other products' cheapest fits, so the code opens those records now: " + ", ".join(rows)
                + ". Judge each against its own product's requirements with the value rules (when a field holds the "
                  "value itself, only the store's words there count); as soon as fits inside the budget cover every "
                  "product at once, add a line for every product to the cart and order; decline only if no such set "
                  "exists.")

    def _decline_room(self, rows, batch, extra):
        return self.moves_left() > self.read_calls(rows, batch) + extra and self.time_left() > CLOSING + 40

    @staticmethod
    def read_calls(rows, batch):
        if not batch or len(rows) < 2:
            return len(rows)
        return -(-len(rows) // batch[2])

    def fill_gate(self):
        shop = self.shop
        batch = shop.batch_tool()
        rows = []
        if not self.decline_fill and (self.role("details") or batch):
            rows = shop.lacking_rows()
        if not rows or not self._decline_room(rows, batch, 3):
            return ""
        self.decline_fill = True
        self.reads.open_queue = rows
        self.reads.open_say = "Ahead of any decline, I read unread listings for products still without a verified fit."
        return ("Not run yet (no step used). Several requested products lack a verified fit as yet, and in-budget "
                "listings of their categories were never opened, so before any decline the code opens their records "
                "right away: "
                + ", ".join(rows) + ". Judge each against its own product's requirements with the value rules (when a field holds the "
                                    "value itself, only the store's words there count); when every product has a listing that fits, add a "
                                    "line for every product to the cart and order; decline only if a product truly has none.")

    def named_gate(self):
        batch = self.shop.batch_tool()
        rows = self.counsel.named_unread()[: batch[2] if batch else 3]
        if not rows or self.decline_opens or not (self.role("details") or batch):
            return ""
        if not self._decline_room(rows, batch, 3):
            return ""
        self.decline_opens = True
        self.reads.open_queue = rows
        return ("Not run yet (no step used). The listings named by the checks were never opened, so ahead of any "
                "decline the code fetches their records: " + ", ".join(rows) + ". Hold every one up to the requirements using the value rules (the field named for the property "
                                                                               "decides, and if the store puts the value itself in it, only its words count; without such a field, "
                                                                               "another field, the options list or the title may carry the value), buy the one that fits, and "
                                                                               "decline only when every one fails.")

    def pages_gate(self):
        return self._held_note(self.bargain.decline_reads())

    @staticmethod
    def _held_note(note):
        return "Not run yet (no step used). " + note if note else ""

    def audit_again_gate(self):
        counsel = self.counsel
        seen = counsel.audit_reads
        more_read = len(self.known.read) > (10 ** 9 if seen is None else seen) and counsel.audits < 2
        if not more_read or not self.decline_vetted or self.moves_left() <= 3 or self.time_left() <= CLOSING + 30:
            return ""
        return self._held_note(counsel.review_decline())

    def first_audit_gate(self):
        if self.decline_vetted or self.moves_left() <= 3 or self.time_left() <= CLOSING + 30:
            return ""
        self.decline_vetted = True
        audit = self.counsel.review_decline()
        return self._held_note(audit) or (
            "Not run yet (no step used). Only decline when sure: your searches took different forms (new words, "
            "filters, pages), each candidate's own properties were inspected, and each candidate misses a written "
            "requirement, a disclosed need, the budget or the stock. Any key spelling of a field counts for its "
            "property. Buy whatever fits; when nothing fits, call the decline tool again.")

    def fit_gate(self):
        return self._held_note(self.bargain.fit_hold())

    def best_gate(self):
        bargain, wants = self.bargain, self.wants
        picks = bargain.unrefuted() if not self.best_held and self.moves_left() > 4 else []
        rounds = 0
        while rounds < 3:
            rounds += 1
            open_ones = [line for line in picks if f"{line}|{wants.version}" not in self.passed]
            if not open_ones or self.time_left() < CLOSING + 30:
                break
            self.counsel.check_item(open_ones[0], "ORDER")
            picks = bargain.unrefuted()
        if not picks or self.moves_left() <= len(picks) + 3 or self.time_left() <= CLOSING + 20:
            return ""
        self.best_held = True
        several = bool(wants.items())
        holds = "a candidate for every requested product" if several else "a candidate"
        put = "one line of each product, at its quantity," if several else "it"
        return ("Not run yet (no step used). Declining is right only if no listing can satisfy the requirements, yet "
                "the records hold " + holds + " showing stock, within the budget and ruled out by no check: " + ", ".join(picks) + ". When no record shows any of them breaking a requirement in force now, add " + put + " in the cart, read stock and price anew, and buy; decline only if each candidate plainly fails.")

    def soft_ref_gate(self):
        loose = [ref for ref in self.refs or [] if ref.get("relation") == "differ"
                 and ref.get("status") in ("open", "stuck") and self.pointers.ref_live(ref)]
        if not loose or self.moves_left() <= 3 or self.ref_decline_held:
            return ""
        self.ref_decline_held = True
        names = ", ".join(dict.fromkeys(ref["head"] for ref in loose))
        each = ", one line per product" if self.wants.items() else ""
        return ("Not run yet (no step used). " + names + " depends on another listing which the code failed to identify, so its value is unknown: it neither "
                                                         "rules a listing out nor supports a decline. Add the listing meeting every other requirement (in "
                                                         "stock, inside the budget) to the cart" + each + ", check its stock and price anew, and order; decline only if no listing satisfies the other "
                                                                                                          "requirements.")

    def send_chunk(self, slab):
        actions = [{"name": tool, "args": args} for _, tool, args in slab]
        first_step = self.shop.tick + 1
        reply = self.shop.perform(actions)
        self.log.append({"kind": "env", "step": first_step, "actions": actions, "reply": reply})
        if reply is None:
            return [(cid, "The call failed (network/server). Try it again.") for cid, _, _ in slab]
        seen = reply.get("observations")
        if not isinstance(seen, list):
            seen = [reply.get("observation")]
        last_at = len(seen) - 1
        out = []
        for at, (cid, tool, args) in enumerate(slab):
            if at > last_at:
                out.append((cid, "Not run: the episode was over before this call."))
            else:
                out.append((cid, self.absorb(tool, args, seen[at], reply, at == last_at)))
        return out

    def absorb(self, call_name, call_args, obs, resp, last):
        shown = obs
        obs = lensed(obs)
        self.book_call(call_name, call_args, obs)
        gone = self.guarded("caps_error", [], self.shop.learn_caps, call_name, call_args, obs)
        if self.basket.units:
            self.note_cart_reads(call_name, call_args, obs)
        self.count_late_read(call_name, call_args, obs)
        notes = ["[stock] " + self.shop.dead_note(gone)] if gone else []
        pointer = self.guarded("notes_remark_error", None, self.pointers.notes_remark, call_name, call_args)
        if pointer:
            notes.append("[notes] " + pointer)
        self._cart_notes(call_name, call_args, obs, notes)
        refused = self.lookout.refused(call_name, obs, resp if last else {"finished": False})
        if refused and self.last_kind == "order":
            self.cart_seq = self.absorb_n
        notes.append(self.lookout.notes_for(call_name, obs, refused))
        searching = call_name == self.role("search")
        if searching:
            notes.append(self.scout.search_note(call_args, obs))
            self.counsel.named_pages.discard(json.dumps(call_args, sort_keys=True, default=str))
            notes.append(self.guarded("merge_joined_error", None, self.shop.merge_joined, obs))
        if last and self.early_audit_due(call_name):
            self.early_audit = True
            hint = self.counsel.review_decline(early=True)
            notes.append(hint and "No verified fit found yet. " + hint)
        self._shopper_notes(call_name, call_args, obs, refused, notes)
        text = as_text(clipped(shown, SEARCH_TEXT if searching else RECORD_TEXT))
        if last and resp.get("finished"):
            text += " [the episode is finished]"
        joined = " ".join(filter(None, notes))
        return f"{text}\n[note] {joined}" if joined else text

    def book_call(self, tool, call_args, obs):
        cart, message = self.role("cart"), self.role("message")
        if tool == message:
            self.msgs_sent += 1
        kind = self.call_kind(tool, call_args)
        self.last_kind = kind
        if kind == "order":
            self.orders_tried += 1
        resets = tool in (cart, message) or tool in self.finishers()
        self.last_calls = [] if resets else (self.last_calls + [tool + as_text(call_args)])[-2:]
        self.known.add(obs)
        self.absorb_n += 1
        if tool == cart or self.covering:
            return
        self.since_cart = 1 if self.since_cart is None else self.since_cart + 1

    def count_late_read(self, tool, call_args, obs):
        details, search = self.role("details"), self.role("search")
        failed = isinstance(obs, dict) and obs.get("error")
        if tool == details and not failed:
            known = self.known
            known.read.update(value for value in call_args.values() if isinstance(value, str) and value in known.ids)
        via_lookup = bool(details) and tool not in (details, search) and tool in self.shop.lookup_tools()
        asked = {value for value in call_args.values() if isinstance(value, str)}
        early = self.moves_left() >= self.brief.order_due()
        if early or self.covering or (tool not in (details, search) and not via_lookup):
            return
        in_cart = set(self.basket.units)
        if tool == details and asked & in_cart:
            return
        if via_lookup and asked & (in_cart | {key.rpartition("|")[0] for key in self.passed}):
            return
        self.late_n += 1

    def _cart_notes(self, call_name, call_args, obs, notes):
        if call_name == self.role("cart"):
            self.cart_remarks(call_args, obs, notes)
        if call_name == self.role("cart_view"):
            self.basket.viewed(obs)
            notes.append(self.basket.summary())

    def cart_remarks(self, call_args, obs, remarks):
        basket, shop = self.basket, self.shop
        self.order_vetted = False
        before = dict(basket.units)
        remarks.append(basket.booked(call_args, obs))
        self.guarded("after_put_error", None, self._after_cart_change, before, call_args, obs)
        line = basket.aim(call_args)
        if not line or not basket.is_put(call_args):
            return
        self.refusals_seen[line] = self.refusal_n
        self.puts[line] = self.puts.get(line, 0) + 1
        count = basket.count_arg(call_args)
        vetted = f"{line}|{self.wants.version}" in self.passed
        if not vetted or (count is not None and count <= 0) or basket.held(line) <= 0 or line in shop.dead_lines():
            return
        remarks.append(f"An independent check judged {line} meets each requirement in force now, judging each property with the value rules (when the field carries the value itself, only this store's words there count); keep it unless a later change rules it out.")

    def _after_cart_change(self, before, call_args, obs):
        self.shop.cart_moved(before)
        self.shop.mark_fresh(obs)
        self.shop.after_put(call_args, obs)

    def _shopper_notes(self, call_name, call_args, obs, refused, notes):
        asked = None
        if call_name == self.role("message"):
            asked = " ".join(str(value) for value in call_args.values() if isinstance(value, str))
        heard = self.scout.shopper_lines(call_name, obs, refused)
        if heard:
            notes.append(self.wants.hear(heard, asked))
            self.pointers.ref_take()

    def note_cart_reads(self, tool, call_args, obs):
        shop = self.shop
        held = self.basket.units
        by_stock = bool(shop.stock_seen)
        requested = shop.asked_ids(call_args)
        lookup_only = not (shop.rows_live and not shop.rows_lag) and tool not in (self.role("cart"), self.role("cart_view"))
        touched = set()

        def held_values(record):
            return [value for value in record.values() if isinstance(value, str) and value in held]

        def counts(record):
            if not by_stock:
                return any(isinstance(value, dict) for value in record.values())
            if self.lookout.stock_state(record) is None:
                return False
            named = {value for key, value in record.items() if isinstance(value, str) and value in held and "id" in str(key).lower()}
            return not lookup_only or bool(named & requested)

        def look(record):
            has_id = any(isinstance(value, str) and value in held and "id" in str(key).lower() for key, value in record.items())
            if has_id and counts(record):
                touched.update(held_values(record))

        walk_dicts(obs, look)
        for line in touched:
            self.line_read[line] = self.shop.tick
            self.line_seq[line] = self.absorb_n

    def early_audit_due(self, tool):
        if self.early_audit or tool in (self.role("cart"), self.role("message")) or tool in self.finishers():
            return False
        left = self.moves_left()
        if left <= FIRST_ORDER_BY or left > self.step_cap // 3 + 2 or self.basket.units:
            return False
        version = str(self.wants.version)
        return all(key.rpartition("|")[2] != version for key in self.passed)

    def on_closing(self):
        if not (self.shop.done or self.shop.broke):
            self.closer.finish_now()
        self.log.append({"kind": "end", "finished": self.shop.done, "steps": self.shop.tick,
                         "cost": round(self.counsel.spend, 5), "seconds": round(monotonic() - self.t0, 1)})
        return "over"

    def ask_limit(self):
        return max(0, self._message_room() - self.held_back())



def agent_main(pdata: dict[str, Any]) -> list[dict[str, Any]]:
    run = Episode(pdata)
    try:
        return run.execute()
    except Exception as err:
        run.log.append({"kind": "crash", "error": f"{type(err).__name__}: {err}"[:500]})
        try:
            if not run.shop.done and not run.shop.broke:
                run.closer.finish_now()
        except Exception as later:
            run.log.append({"kind": "crash", "error": f"{type(later).__name__}: {later}"[:500]})
        return run.log


__all__ = ["agent_main"]
