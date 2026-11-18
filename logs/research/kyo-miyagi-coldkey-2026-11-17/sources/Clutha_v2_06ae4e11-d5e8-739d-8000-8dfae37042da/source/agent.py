from __future__ import annotations

import json
import re
import threading
from contextlib import contextmanager
from typing import Any

from src.agent.proxy_client import ProxyClient


class _Tally:
    """A counted (never read) clock for the errand: a wait advances it by its own duration, and an open request
    advances it a quarter second per quarter second it remains pending."""
    SECTION = 0.25

    def __init__(self) -> None:
        self.seconds = 0.0

    def now(self) -> float:
        return self.seconds

    def breather(self, longness: float) -> None:
        if longness > 0:
            threading.Event().wait(longness)
            self.seconds += longness

    @contextmanager
    def outstanding(self):
        done = threading.Event()

        def count() -> None:
            while not done.wait(self.SECTION):
                self.seconds += self.SECTION

        laborer = threading.Thread(target=count, daemon=True)
        laborer.start()
        try:
            yield
        finally:
            done.set()
            laborer.join()


_TALLY = _Tally()

# model slots: acting model, single request parse, item/rank checks, decline audit, shopper-line reading, question planning
OPERATOR = ["anthropic/claude-haiku-5.5", "openai/gpt-6-luna", "anthropic/claude-haiku-5.5"]
BROWSER = ["anthropic/claude-sonnet-5.5", "openai/gpt-6-luna"]
VALIDATOR = ["anthropic/claude-haiku-5.5", "anthropic/claude-sonnet-5.5"]
EXAMINER = ["anthropic/claude-haiku-5.5", "anthropic/claude-sonnet-5.5"]
MONITOR = ["anthropic/claude-haiku-5.5", "anthropic/claude-sonnet-5.5"]
SCHEDULER = ["anthropic/claude-haiku-5.5", "anthropic/claude-sonnet-5.5"]
# own question engine: a single ranking call per episode, plus the flag that falls back to the planner above
SOLICIT_SPOT = ["anthropic/claude-haiku-5.5", "google/gemini-3.8-flash"]
MY_SOLICIT = True
SOLICIT_PRIMARY_SELECTS = 2
SOLICIT_PANEL = 24
SOLICIT_SPECIMENS = 3
SOLICIT_GRADINGS = 2
SOLICIT_SYMBOLS = 70

# time (seconds)
EXPIRY = 265.0
SEALING = 70.0
DEEPER_FOLIOS = 4             # pages a best/cheapest product's widest search may still get before its first cart put
ULTIMATE_TRIGGER = 28.0
TICKER_PHASE = 30
HX_ALIAS_LEAST = 4
# how much of each observation the acting model gets to see
GLANCE_SYMBOLS = 16000
HUNT_PROSE = 240
TRANSCRIPT_PROSE = 700
# LLM spend in USD
LENIENT_DISBURSE = 0.03
DISBURSE_TOP = 0.045
STINT_CONSTRAINT = 80
# request pacing and retries
SPEED_SHORTFALL = 0.15
GRADUAL_SHORTFALL = 1.0
GRADUAL_TOWARDS = 30.0
DOWNWARD_AFTERWARD = 45.0
ORACLE_TRIALS = 2
REALM_TRIALS = 5
# limits on what the code is allowed to do by itself
ROUTINE_TERMINATES = 4
ROUTINE_FOLIOS = 3
ROUTINE_BROWSES = 8
SPLICERS = r"\s+and\s+|\s*&\s*|\s*/\s*|\s+\+\s+"
DEPOSIT_VOCABLES = {"add", "put", "hold", "set", "insert", "place", "keep", "include", "in", "plus", "increase", "append"}
JETTISON_VOCABLES = {"remove", "drop", "release", "delete", "take", "out", "clear", "discard", "minus", "decrease", "subtract", "off", "unhold"}
FLAW_SELECTORS = r"error|fault|refus|reject|denied|invalid|fail"
SOLICIT_BASELINE = 3
PRIMARY_CHECKOUT_THROUGH = 7
PRIMARY_CHECKOUT_EXPECTED = 10
DORMANT_CONSTRAINT = 3
JAMMED_CONSTRAINT = 4
COMB_BROWSES = 8
ROUTE_FOLIOS = 6
ROUTE_OVERALL = 8
MEASUREMENT_FOLIOS = 4
MEASUREMENT_OVERALL = 8
MEASUREMENT_OVERALL_APIECE = 5
CAPTION_TRIALS = 4
CAPTION_ACCESSES = 3
ROUNDUP_TRIGGERS = 4
ROUNDUP_ACCESSES = 3
ANCHOR_VOCABLES = {"and", "or", "the", "for", "with", "of", "a", "an", "in", "on", "to", "by", "from", "at", "as", "is", "are", "be",
              "this", "that", "it", "its", "your", "our", "you", "all", "any", "not", "no"}
SOLICIT_AFTERWARD_COLLECTIONS = 2
SOLICIT_LATITUDE_PHASES = 3
SOLICIT_AFTERWARD_PHASES = 5

OPERATE_POLICIES = """Goal: you help one shopper in a simulated shop and may act only by calling the tools you are given; your job is to close the errand with everything right. The grade is all or nothing (1 or 0). A 1 needs every point below at once: the ordered item meets each written requirement of the shopper, each hidden need the shopper discloses when asked, and each change announced during the episode; the quantity and the budget are right; when the request asks for the cheapest (or the best) item, that is the one picked; and each rule the request lays down about how to proceed is obeyed. If no item in the shop meets all of it, the right ending is a decline carried out precisely as the request describes.

1. Understand the shop's mechanics.
- Every tool call uses up one step, failed and invalid calls included, and the number of steps is capped. Reasoning costs nothing.
- One reply may contain several tool calls if none of them depends on what another returns (for example opening the details of a few listings); they are executed one after another. An order or a decline is always sent by itself.
- If the details tool takes a LIST of listing ids, a single call returns the complete records of all of them for ONE step (up to its item limit): whenever you open two or more listings, use it that way.
- The names of tools, arguments and result fields differ between shops. Study the description and JSON schema of each tool and use exactly the names and allowed values written there. Work out from the first results which tool reveals prices, stock levels and detailed properties.
- At the start, find out which category labels this shop gives to listings of the wanted kind. When several neighbouring labels exist, pick the one that matches the request's wording taken as a whole, and do your searching and buying inside it.
- Search engines differ from shop to shop; deduce how this one works from what it first returns. Rows may come ordered by relevance or by a fixed id. A keyword search may require every word (adding a word makes the total smaller) or accept any word (adding a word makes the total larger, and listings hitting more words rise to the top). Possibly only a few rows or pages are readable. A filter may only work in one mode (keyword mode disregarding the category or price filter, filter mode disregarding the words). Compare any reported total with the rows you can see: if it is bigger, the fitting or cheapest listing may sit on a later page or past the visible rows. When every word is required, tighten the search with more exact words (or with category and price filters and a lower price ceiling) until the total fits inside the readable pages. Keep queries short (one to three words), putting the most telling words first: some shops read just the first word or two and drop the remainder, so a long query may in effect search only its first words (put a longer query beside a shorter one with the same start and compare totals and rows; the notes explain what the comparison means). When every word is required, a wanted value placed before the item kind ("blue chair", value first) returns exactly the listings that carry that value. When any word is enough, everyday words (the item kind, a unit) match most of the shelf and drown out the rest: searching the telling word or figure of a wanted value alone ("herringbone", "30m", "0.4") returns only listings that carry it, so search it on its own and also in the other forms listings write it in (for instance "3 atm" or "30 m"), go through every page of that search whenever the total fits within them, and combine the item kind with values only to push the closest matches upward. Keyword search may also hit category names and property values. After each search, notes summarise what its totals show about this shop's search.

2. Uncover the shopper's hidden needs.
- The shopper has needs that were never written down and reveals one only if a message asks about exactly that property. The FIRST thing a message asks gets the most reliable answer, so ask about a single property per message (two at the most, the more likely one first): a short, simple, friendly question naming the property (one or two example values are allowed). Leave out instructions, lists, labels, positions and rules.
- Answers such as "anything is fine", "no preference" or "don't mind" mean that property stays open. Any preference or limit the shopper gives is a hard requirement from that moment, kept in the shopper's exact words, figures included.
- Your planned questions go out automatically once two or three listings of the wanted kind have had their details opened, so open some early. Answers show up in the conversation with their meaning spelled out; a disclosed need is as binding as a written one: run new searches that include it together with the written values.
- If you ask on your own: begin by opening a few (two or three) listings' details, then ask about what the story hints at (the recipient, the look, the place or manner of use), then about listing properties that vary among the candidates. Do not ask about anything the request already settles, and do not exceed the message budget.

3. Obey the procedural rules.
- The request may say how to go about the job: how many listings to look at or compare, what to check, what the shopper must be told, what any remark has to include, how to decline. These rules are graded just like requirements; follow every one of them literally and completely. Being told to "look at" or "open" N listings means N listings that fit fully, each of them opened with the details tool.
- If the shopper asks you something (for instance to confirm a change), the answer is sent for you while messages are left; whatever was changed holds regardless.
- Wishes alone ("if possible", "ideally", "would be nice") do not count as requirements; they merely decide between items that already meet everything, and they never beat a cheapest rule.
- If the request leaves open a point that only the shopper can decide (it sets no price ceiling, or it names two wishes that may be impossible to satisfy together and need ranking by the shopper), ask before choosing.

4. Track changes as they arrive.
- The shopper may change things: a new budget (up or down), a requirement removed, replaced or added, a price rise, an item going out of stock. A single line can carry several changes, and such lines may turn up inside any tool result (as an extra field of sentences) or as the answer to an order attempt. Every change is binding from that point: review the pick and the backups again, and if the cheapest or best is wanted, choose afresh from the new set of acceptable items.
- When an order is REFUSED the episode goes on and the cart stays as it was. Read the refusal, apply all the changes it contains, remove from the cart anything that no longer fits (an item that sold out, for instance), add the next checked fit, check its stock and price once more, and place the order again.
- Stock and prices can also change without notice, so check both again right before every order attempt; if an item's current price multiplied by the quantity exceeds the current budget, it no longer fits.

5. Deal with several products.
- Some requests ask for several distinct products, every one having its own quantity, its own requirements and its own hidden needs. Cover all of them: search every category, confirm a fit for each product, and buy them together in one order holding one line per product at its quantity. If a cart tool accepts a list of lines, it may replace the whole cart with that list, so every such call must carry the lines of all products. Cheapest or best is judged separately for each product.
- A requirement may state its value by reference to a different page or listing (e.g. "the colour of the jumpsuit sold by X"): locate that listing first (search for whatever singles it out, e.g. the shop or brand together with its kind), open its record, take the value from the named field, and only then search for the wanted product with that value.

6. Watch the cart.
- Certain cart tools ADD units to a line that is already there instead of setting the count: adding 1 more of a line you hold gives 2. Do not add a line the cart already contains; to change the count, remove units or add just the difference. After every cart call a note reports what the code believes the cart holds.

7. Assess every listing (this is what the grade depends on).
- For each requirement the grader reads the listing field that holds that property and compares the words of the value as the shop writes them in that field. A field counts whenever its name stands for the property, however it is spelled: separators, a prefix, a shorter or longer name, or a well-known other name ("color family" meaning a colour, "dial size" meaning a watch dial size, "neckline" meaning a collar type, "fit" meaning a pants fit type). If one field carries the property's exact name, that field decides, as does a field carrying that name with one extra word before it (clothing material for material, pants length for length): once either of the two writes the value, the requirement is met. Names that differ only in their separators (size_class, size class) are the same field: the requirement is met if either writes the value. If no such field exists, the other-named field decides; only when no field at all names the property can the value be read from the title or the options list. Descriptions never serve as evidence.
- Values are matched word by word in the shop's own wording; letter case, separator characters, singular/plural and variant spellings are ignored. A narrower value that includes the wanted one meets it ("forest green" covers green, "rtx 3060 12gb" covers rtx 3060, "80% cotton" covers cotton). A single entry in a list is enough, even when the shopper wants only that value: a field that offers several variants, one of them the wanted value, meets it. Shades, synonyms, related words and abbreviations (ivory or cream for off-white, "solid" for plain, "embroidered" for embroidery, "win11" for windows 11, "tech wood" for technical wood) only count as long as none of the category's listings uses the value itself in that same field: when other listings give "matte" as their finish, a "satin" finish does not meet matte. A value joined to another with a plus ("wool + nylon") meets neither part if other listings show the other part by itself ("nylon") in that field. Lines marked FIELD EVIDENCE, when present, are computed from this shop's records and say which case applies. A rival value of that property fails (rosewood is not maple, quartz is not mechanical), as does a merely similar-looking word. A value claiming to cover all values of the property ("all seasons", "suitable for all occasions", "universal") in fact names none and fails a wanted value; a list that also names the wanted value still meets it. A field holding another value does not meet the wanted one merely because the two often come together or one suggests the other (a "pullover" style is not "zip-up", "slip-on" is not "lace-up"); what qualifies is only a narrower value containing it or a shade normally sold under the wanted colour.
- A field about some other property cannot stand for the required one (length is not style, lining material is not material, strap length is not strap type): where the category's listings have the property's own field, such a field never replaces it.
- Only what the shopper wrote or disclosed counts. Never exclude a listing for something the shopper did not ask for: an age group, a gender, a size, an occasion, a condition, a quality or a use inferred from the description or the story. The story is background; if it looks relevant, ask the shopper about that property instead of guessing. Requirements come from what the shopper replies, never from what you asked.
- Numbers are compared as quantities after unit conversion (45 mm equals 4.5 cm); limits are inclusive; a range passes only if all of it respects the limit. If no field names the property and the title therefore serves as evidence, an exact wanted value is met when the title shows the value's words and figures in the same order, possibly with extra detail after them (9' x 12' 4" shows 9' x 12', but 9' x 13' does not).
- Title figures like "117 x 228 cm" or "55 x 94 Inch" never tell you which number is which dimension: never reject a listing, or skip opening it, on their account; check the fields in its details.
- Respect exclusions (such as "leave out anything with X") and closed lists (such as "A or B and nothing else").
- The category alone decides the wanted kind of product: to the grader, a listing in the wanted category IS that kind even if its title or description speaks of some other object (an accessory, a refill, a set, a container, a case), so never skip, reject or rank it lower because of how it looks; judge only requirements, price and stock. No property needs to repeat the category name. When the shop labels its categories, the item has to sit in the one whose label fits the request's wording as a whole (a request for "A or B" belongs in a category "A & B" rather than in "B", its narrower neighbour). When the request wants goods in stock, the item must show stock, and price times quantity may not go over the budget (equal is fine). Prices are expressed in the budget's unit except where the record states another (with a budget of 1000 US cents, a price of 45 is 45 cents); do not rescale either.
- A decline is seldom right. Even if no listing meets the literal wording, a listing that meets each requirement once these rules are applied (thanks to a narrower value, a field under a different name, or a shade or synonym not contradicted by any category listing that writes the value itself) is the fit; a decline is justified only after the search has genuinely produced nothing passing every requirement under these rules.

8. Choose the cheapest or the best when the request says so.
- For the cheapest, the ordered item has to be the lowest-priced of ALL the items meeting every requirement; finding one fit does not end the matter. If the shop's keyword search demands all words and also looks at property values, then searching value plus kind brings up the candidates that carry the value: read all of its pages and open them by price. A category listing with a price limit reveals the prices of the whole category, though not the values: use it to see which cheaper rows exist (lower the limit until the total fits), then check those rows' properties. Its rows come in a fixed id order, not by price, so never page through a long price-limited listing: when it reports more matches than one or two pages show, cut the limit (roughly halve it, or set it just below the cheapest fit you know) and list again, until every match fits on the pages you can read; the cheapest rows of the category then appear, and you raise the limit step by step only if none of them fits. Search each wanted value with the item kind as a short query, value first ("blue chair", "30m watch"), and read every page when its total fits the readable pages (otherwise add a second wanted value or a narrower kind word); repeat with other names for the kind ("bench", "stool") and with the value alone. Where a value is impossible to search for (a figure the shop strips out), also search the product line of each fit or near-fit you found, as its maker plus one telling title word (two words), all pages: each size and colour of a line is a listing of its own with the same property values, and some may cost less. Sort the visible candidates by price on your own and open them in turn starting with the cheapest; the first whose property fields meet everything, after every cheaper one was opened and rejected, is the answer. Keep the next fits as backups.
- For the best by some measure (most, largest, longest, highest...), go through ALL fitting items the same way, compare them on the measure and take the extreme one. An item lacking any value for the measure is out of the running.

9. End cleanly.
- Keep a ranked list of checked fits including at least one backup: the first choice may sell out at checkout.
- Before every order attempt: the cart holds exactly one line per wanted product (one product, one line), each the chosen item at the required quantity, and nothing else; after your last cart change you have re-read the chosen item's stock and price and found it available and within the budget.
- Decline only when a thorough search turns up nothing meeting every requirement, and follow the request's decline rule exactly. When substitutes are forbidden, the remark that goes with the decline must not name, describe or hint at any listing: no ids, no titles, no prices.
- Plan your steps: make the first order attempt while at least seven steps are left, so a refusal can still be handled. When 7 steps remain, only the cart, stock-check and finishing tools can still be used. Make the decision once; do not keep switching the cart back and forth.

Return: each reply contains one or more tool calls, plus a short text giving the current requirements, the checked fits with their price and stock, and the next action."""

BROWSE_APPEAL = """Goal: study the request and the tool list given below and answer with exactly ONE compact JSON object (written without indentation), with nothing around it.
1. The object has these keys:

"product": the kind of product asked for, using the request's own words (when there are several products, name all of them joined with "and").
"items": one entry for each DIFFERENT product to be bought, in order: {"product": its kind in the request's phrasing, "quantity": the number of units of one listing of it (read as for "quantity" below), "requirements": its firm requirements (read as for "requirements" below)}. A single product gives a single entry. A name joining two related kinds with "and" or "or" under one count and one set of requirements (a store category such as "lamps and lamp shades") is one product with one entry.
"quantity": the whole number of units of one listing to buy; 1 unless the request asks for more. Things sold as a single piece (a set, a pair of shoes) count as 1. Only a clear wish for several units increases it ("order 2 packs", "I need three of them"); how many people will use or share the item, or how many old ones existed, is part of the story, so the count remains 1. A count written in front of a product's name ("4 mugs", "two packs of tea") is that product's quantity even when the request adds that each product takes one line, or one line apiece: such a remark counts lines (one per product), never units on a line.
"requirements": a list of strings, one per firm requirement on the product (when there are several products: all products' requirements, each written "property (product): constraint"), each written "property: constraint" and holding precisely the shopper's value, limit, exclusion or closed list. Keep out price, stock, quantity, budgets and mere wishes, and also any restatement of the product kind or category: words belonging to the category's name (a season, an occasion or an audience it contains) are taken care of by the category and form no separate requirement. A requirement links a property of the product to the value the request demands ("with a mechanical movement", "the pattern has to be herringbone", "at least 4 drawers", "the occasion has to be party"). Story never counts as a requirement: whom the item is for, an event at which it is worn or used ("for the staff party", "to wear at his wedding"), where, when or how it is used, how many people share it, or that it replaces an old one ("a new watch" signals a replacement and sets no condition). An audience, use or occasion becomes a requirement only when the request makes it a property of the item itself ("a blazer made for business", "the occasion has to be business"). A requirement's value is made of words or figures that the request itself writes down as the item's value; a quality, reason or wish drawn from the story (natural and breathable, slim enough for a case, works with what I already have, covers the whole drop) is just a hint. When a value comes from another listing ("the finish shown on the page of X's kettles with a glass lid"), that requirement keeps the pointing words as its constraint; the details naming which listing is meant (its seller, its kind, "with a glass lid") describe that listing and are never requirements of the product itself.
"price_cap": the highest total price, given as the exact number the request writes and in the request's own unit (no conversion: "3050 US cents" gives 3050), or null when the request gives none.
"currency": the unit the request uses for prices, or null.
"in_stock_only": true when only items that are on hand or in stock are acceptable.
"priority": "cheapest" only when the request plainly wants the fitting item with the lowest price; "best" only when it plainly wants the best by some measure, which includes a rule that among the fitting items the one with the highest or lowest value of a property wins ("the lightest one wins", "go with the most capacity"); "none" otherwise. Comments that money is tight or that every penny matters are not a request for the cheapest; they only explain the cap.
"priority_detail": the measure if "priority" is "best", naming the property and the direction ("lowest weight", "highest capacity"); else "".
"message_budget": the number of messages the shopper allows you, as an integer, or null when not stated.
"step_budget": the number of moves or steps the request allows, as an integer, or null when not stated.
"figures": {"price_cap": the request's words setting the price limit, "message_budget": the words setting the message limit, "step_budget": the words setting the step limit}, each copied verbatim, "" when the request sets none.
"fit_with": every product the request says must work with, fit or suit another product chosen within this order, as {"product": the product that must fit, as named in "items", "with": the product it must fit, as named in "items", "quote": the request's sentence saying so, copied verbatim}. Empty when none.
"procedures": every step the request requires the assistant to take with the shopper that calls for a separate message and is no question about the products (approval of the basket before the order, a report before acting), as {"quote": the request's sentence, copied verbatim, "needs_yes": true if the shopper must agree before the assistant acts, "final_cart": true if it concerns the completed basket or cart}. Questions about the goods, about whom they are for, or about anything else that is useful to know never form such a step, even when the request welcomes questions; neither do rules on choosing, on what to buy or on delivery. Empty when none.
"rules": each procedural rule the request sets for the assistant (how many listings to go through, what to check or report, what to include or avoid, how to decline), copied verbatim. What the shopper will do personally belongs to the story.
"soft_wants": things the shopper would like without strictly needing them (preferably, ideally, if possible, would be nice). Never place them in "requirements".
"ask_first": things only the shopper can answer that must be asked before choosing: the budget if no limit is given, which of two wishes counts more when the request allows that both may not be possible (in that case put both wishes in soft_wants and keep them out of requirements), or a wish that, according to the request, has to be cleared with the shopper before choosing (approval of the basket or the order goes under "procedures", not here). Empty when none.
"hints": details of the story that point at a property of a product without stating its value (its use or what it is used with, the place, the user, or what it must fit, match or withstand), each written "property (product): a few words on the detail" ("property: detail" if just one product is wanted), the property named as listings would most likely name it. When a detail could concern more than one property, put the likeliest first and the second choice after a slash ("colour / style (chair): has to go with a room that has one painted wall"): whatever the item must suit, match or go with (a room, a wall, an outfit, things used alongside it) and whatever concerns its look points first at its colour, next at its pattern or style. Skip any detail that merely justifies a requirement already given (the reason the stated size or material is needed). Weather, travel, a job, a timetable, or a place where the item is neither worn nor used is only background and never a hint; likewise a purpose that every listing of that kind fulfils (warmth for a coat, storage for a dresser). Empty when none.
Every sentence of the request must end up somewhere: as plain story, a hint, a budget, a rule, a soft want, the priority, or a requirement. When unsure whether a sentence is a rule, put it in "rules".
"references": requirements for which the request supplies the value solely by referring to ANOTHER page or listing (examples: "the colour of the jumpsuit X sells", "within 5 of the figure Y's page gives", "a colour other than the one on Z's page"), each written "property (product): as shown on <the request's name for that listing>" (or "property (product): different from the one shown on <the request's name for that listing>" when the value must differ from that listing's); put each of them into "requirements" too, in the same form. Empty when none.
"reference_targets": one object per entry of "references", describing the listing referred to: {"property": the property in question, "product": the product in question, "seller": the shop or brand that the request gives for that listing, copied verbatim, "kind": the kind of product that listing is, "details": [every other value the request gives for that listing, short and in the request's wording ("blue", "4gb", "net weight 167.262186 g")], "relation": "same", "differ" (NOT the value that listing has: other than, different from, set apart from), "within", "at most" or "at least" (how this value relates to that listing's), "tolerance": the figure when "within", else null, "unit": the unit, else ""}. Empty when none.
"decline_no_names": true when the request's decline rule forbids offering, naming or suggesting alternatives or substitutes; false when it wants something named; null when the request sets no rule of that sort.
"roles": an object giving, for every role, the exact name of a tool from the list that plays it, or null when no tool fits: "search", "details", "view", "message", "cart", "cart_view", "order", "decline". "message" sends text to the shopper; "cart" adds things to the cart or removes them; "order" purchases or checks out; "decline" ends the errand without buying.
Return: only that JSON object.
"""

ADVANCE_ONWARD = """Goal: before ordering, a shopping assistant showed the shopper the final basket. Read what the shopper answered and give back exactly ONE JSON object with nothing else: {"answer": "change"} if the answer wants something in the basket altered (a different product, quantity or property, or a new limit on price); {"answer": "no"} if the answer clearly turns down the order or tells the assistant not to buy; {"answer": "yes"} in all other cases, because consent, thanks, an acknowledgement or a comment requesting no change all leave the basket as it is."""

PHASE_VALIDATE = """Goal: below are rules that a shopper's request set for a shopping assistant. Pick out the rules that oblige the assistant to send the shopper a separate message which is no question about the products, for example asking the shopper to approve the basket before the order. Your answer is exactly ONE JSON object with no other text: {"steps": [the numbers of the rules you picked]}, or an empty list if there are none."""

SOLICIT_GRADE = """# Shopping assistant - current step: decide which properties are worth asking the customer about
- The customer holds back one or two wishes about a product and only mentions them when a question names exactly that property.
- For every product (P1, P2 ...) you receive the properties found on the shop's listings for it, how many listings have each, and a few example values.
# Rules
- Order up to eight properties per product, putting the one most likely to hide an unstated wish at the top.
- A property already decided by the request hides no unstated wish: omit it. Omit identifiers, titles and prices as well.
- Use what the background suggests without stating it: the item's user, the place or manner of use, the look or feel it should have, the fault the previous one had.
- A property pointed at by such a detail is listed in "story" and also in "ranked". When there is no such detail, "story" is left empty.
- Write property names exactly as given; two spellings of the same property count as one.
# Answer
- ONE JSON object and nothing more: {"P1": {"ranked": ["property", ...], "story": ["property", ...]}, "P2": {...}}"""

PERCEIVE = """Goal: on behalf of a shopping assistant, figure out what the shopper's most recent words mean.
1. Output ONE JSON object without any other text, using these keys:
"meaning": one short sentence stating what the shopper intends.
"needs": new firm requirements on the product, each written "property: constraint" with exactly the stated value, limit, exclusion or closed list. When a comparison phrase (at least, nothing less than, falls short of) is followed by words instead of a figure, those words are the value itself and are copied as the shopper wrote them: "falls short of not available" turns into "property: not available" (a field may actually contain "none" or "not available"). When the shopper's line names the property itself ("the strap material has to be leather"), write the need under that name, as the line words it, not under the name the assistant's question used: the shopper's name is the store's own field.
"no_preference": properties the shopper said they are indifferent about.
"dropped": earlier requirements the shopper has given up.
"new_budget": the new maximum total price, as a number expressed in the request budget's unit (if the line's figure already uses that unit, take it: for a budget held in cents, "13075 cents" gives 13075), or null.
"unavailable": true when the shopper says an item can no longer be bought (sold out, out of stock, gone, button disabled); otherwise false.
"meta_only": true when the line just wants the assistant to put fewer questions at once (the first one now and the rest later, less in one go, one question at a time) and gives no value: nothing has been answered yet, so needs, no_preference and dropped take nothing from it, and no value is named by its figurative wording; otherwise false. If the line only says the shopper did not understand or follow the question, or asks for it again in a shorter or simpler form, it is not meta_only.
"requirements_now": every firm requirement on the product that holds right now, each "property: constraint": start from REQUIREMENTS SO FAR, keep each entry the shopper has neither dropped nor replaced, keep only the latest value of a replaced one, and add every new need the shopper's lines express. Anything from the request's story missing from REQUIREMENTS SO FAR stays out; requirements only come from the shopper's lines. Omit price, stock, quantity and product category.
"reply": when the shopper puts a question to the assistant or wants something confirmed, a brief plain-worded answer to return that restates the change itself (for example confirming the new value); else "".
"references": for each need in "needs" whose value the shopper gives only by referring to a different page or listing, the listing referred to: {"property": the property in question, "seller": the shop or brand that the line gives for that listing, copied verbatim, "kind": the kind of product that listing is, "details": [every other value the line gives for that listing, short and in the line's wording], "relation": "same", "differ" (NOT the value that listing has: different from, other than, set apart from), "within", "at least" or "at most", "tolerance": "within" takes its figure here, anything else null, "unit": the unit, else ""}; [] when there is none.
2. One line can contain several changes; record every one. A need concerning another property, even a closely related one (strap width versus width, seat height versus overall height), adds a requirement and replaces nothing: an earlier requirement is only dropped or replaced if the line says that about exactly that property. A replacement appears twice: the earlier value in dropped, the new value in needs. When the shopper tells which of two wishes is more important, the favoured wish turns into a need. A budget stated by the shopper goes into new_budget. Mere wishes ("would be nice", "ideally") are not needs.
3. Believe the shopper and read them charitably: plain, folksy, figurative or roundabout wording that names a value expresses a firm wish for that value; a value is excluded only when the shopper clearly says to avoid it. Words that just stress how firmly a value is held (only, strictly, nothing but, locked to, stays, sticking with, nothing short of, holding out for) are not part of the value: write only the value's words ("color: white", "material: jute"), never "<value> only"; add an exclusion only if the shopper also explicitly rules out blends or other values. A figure with a comparison (at most, at least, no more than, under, over) is a limit, unit included. Any new budget replaces the old one, higher or lower, even when presented as good news.
4. When the shopper only says they don't mind, that anything is fine, or that they have no preference, the property that was asked about goes into no_preference (only the FIRST property if the message asked about two). When the answer only says the question was too much for the shopper ("that was rather dense", "could you put it another way", "you've lost me"), this shopper means no preference if the question mentioned which product the property belongs to: list that property in no_preference as well; when several products are wanted and the question left the product unnamed, the property remains open and no_preference stays empty.
5. When there are several products, write each need as "property (product): constraint", taking the product from the line (a line about "toe style of the women's boots" gives "toe style (women's boots): ...").
6. Requirements come solely from the shopper's words, never from the assistant's own question, and example values the question mentioned are not needs. A "don't mind" releases just the property that the assistant's message asked about; earlier needs are never removed or altered by it, and after a message that asked nothing (it merely reported something) such an answer has no effect. When a need is set by pointing at another listing, its value is the line's own pointing phrase ("finish: unlike the one shown on shop Y's kettle page"); any values the line attaches to that other listing describe it rather than the product, so they count as neither needs, drops nor freed properties.
Return: only the JSON object."""

BROWSE_POLICIES = """1. Check each requirement in the listing field that holds its property, comparing the words of the value as this shop writes them there. A field counts whenever its name stands for that property, however it is spelled (separators, a prefix, a shorter or longer name, or a well-known other name, e.g. "fit" meaning a pants fit type, "neckline" meaning a collar type, "color family" meaning a colour). A field with the property's exact name is decisive, and so is a field with that name preceded by one extra word (clothing material for material, pants length for length): once either of the two writes the value, the requirement is met. Two field names differing only in separators (size_class, size class) are a single field, and the requirement is met when either writes the value. If not, the other-named field decides; the title or the options list may supply the value only when no field names the property. Descriptions are never counted.
2. Match values word by word in the shop's own wording; letter case, separator characters, singular/plural and variant spellings make no difference. A narrower value that includes the wanted one meets it ("80% cotton" covers cotton, "forest green" covers green, "rtx 3060 12gb" covers rtx 3060); a single entry in a list is enough, even when the shopper wants only that value: a field that offers several variants, one of them the wanted value, meets it. Shades, synonyms, related words and abbreviations ("tech wood" for technical wood, ivory or cream for off-white, "embroidered" for embroidery, "solid" for plain, "win11" for windows 11) only count as long as none of the category's listings uses the value itself in that same field: when other listings give "matte" as their finish, a "satin" finish does not meet matte. A value joined to another with a plus ("wool + nylon") meets neither part if other listings show the other part by itself ("nylon") in that field. Lines marked FIELD EVIDENCE, when present, are computed from this shop's records and say which case applies. A rival value of that property fails (rosewood is not maple, quartz is not mechanical), as does a merely similar-looking word. A value claiming to cover all values of the property ("suitable for all occasions", "universal", "all seasons") in fact names none and fails a wanted value; a list that also names the wanted value still meets it. A field holding another value does not meet the wanted one merely because the two often come together or one suggests the other (a "pullover" style is not "zip-up", "slip-on" is not "lace-up"); what qualifies is only a narrower value containing it or a shade normally sold under the wanted colour.
3. A field on some other property never counts (length is not style, lining material is not material, strap length is not strap type): when the category's listings carry the property's own field, it cannot be replaced by such a field.
4. Numbers are compared as quantities after unit conversion (45 mm equals 4.5 cm); limits are inclusive; a range passes only if all of it respects the limit. If no field names the property and the title therefore serves as evidence, an exact wanted value is met when the title shows the value's words and figures in the same order, possibly with extra detail after them (9' x 12' 4" shows 9' x 12', but 9' x 13' does not).
5. Prices are expressed in the unit of the request's budget except where the record states another currency or unit (with a budget of 1000 US cents, a price of 45 is 45 cents and fits easily); do not rescale the price or the budget.
6. Only the listed requirements count: never fail a listing for an age group, a gender, a size, an occasion, a condition, a season, a quality or a use that no requirement names.
7. The category label alone decides the wanted kind: a listing in the wanted category is that kind even if its title names another object (an accessory, a refill, a set, a container, a case); its requirements, price and stock are all that count."""

VALIDATE_ARTICLE = """Goal: you are the final gate a chosen item must pass before the shopping assistant commits to it, and the errand only scores 1 if every requirement below is met. Decide from the records shown and from nothing else, step by step:
""" + BROWSE_POLICIES + """
8. The category itself satisfies words that form part of its own name.
9. Closed lists and exclusions apply.
10. The item has to be of the wanted kind; when listings have a category label, the correct one fits the whole wording of the request rather than an adjacent or narrower label.
11. Price multiplied by quantity must not go over the budget (equal is acceptable). An item the records show as out of stock fails.
12. A plain wish ("ideally", "would be nice") flagged as not required is not a requirement.
13. If records disagree (on price or stock, for instance), the latest one counts.
14. Use fits=false only when a listed requirement is clearly failed by the item's fields under these rules (a FIELD EVIDENCE line stating the item does NOT meet a requirement counts as such a failure), or is stated by no field and, where no field for it exists, by neither the options list nor the title; a missing price or stock record is never a reason to fail an item.
Return: exactly ONE JSON object, no other text: {"item": the item's id, "fits": true or false, "problems": [brief strings, one per requirement that is unmet or not shown]}."""

GRADE_VALIDATE = """Goal: an item is about to be ordered by a shopping assistant for a request asking for the PRIORITY; look for any listing that might beat it. Below you get the requirements, the chosen item and the records of the listings seen so far.
1. Use the reading rules that follow (price times quantity inside the budget; not shown as out of stock).
2. List every listing that could satisfy all requirements AND do better than the chosen item on the priority (for the cheapest, a lower price; for the best, a better value of the measure).
3. Leave out listings whose records show a failed requirement, no stock or a price over budget. The numbers in a ROW's title (55 x 94, say) leave open which dimension each one measures, so they never exclude a ROW. A word in a ROW's title that seems to contradict a requirement (a model name carrying a different network, size or colour) does not exclude it either: the record's own field decides, and titles often name a model line rather than the item's value.
READING RULES:
""" + BROWSE_POLICIES + """
Return: exactly ONE JSON object, no other text: {"better": [up to 4 ids, best first], "reason": "one sentence"}."""

SOLICIT_STRATEGY = """Goal: pick the questions a shopping assistant will ask a shopper. The shopper keeps needs secret, each one a value of a single product property, and lets one out only if a message targets that very property; the first thing a message asks gets the most reliable answer, and questions about anything else only earn a polite "don't mind". Choose no more than QUESTION_COUNT different properties, the most promising first:
1. Begin with whatever the section MUST BE LEARNED FROM THE SHOPPER below lists that is still unasked, and put it simply. For the budget, ask for the most they would spend. When the shopper must choose between two wishes, mention both values along with their properties, for example "If one listing can't have both, would you rather have the <value> <property> or the <value> <property>?".
2. Next, properties that actually appear in the listing details below. Read the story first (STORY HINTS below, when given, list its details and the property each concerns; they come first): the hidden need is usually the property that the story's concrete details are about without ever stating its value. Anything the user is going to press or touch, examine up close or in pictures, hear, compare it with or fit it into, read on its tag, or the way it was produced, singles out that part or side of the item and so its property: labels, tags and small print suggest the material, fabric or ingredients; looks and photos suggest the colour, pattern or visible part; a body part or space it has to fit suggests that part's size or shape; a gift for another person, or for someone barely known, suggests the intended wearer or user (gender, age group); the feel it should have against skin, hand or foot suggests fabric or material first; the activity, moment or place in which it gets worn or used (a gym class, the office, outdoors) suggests its type or style, that is the property whose values name such uses; anything fastened or adjusted repeatedly suggests its mechanism or closure; the part that does the work inside suggests its core component. A job, a timetable, the weather, or any place where the item is not worn or used is mere background. The first question goes to the property the story's details point at, called by the name the listings use. The intended user is a candidate like the others: ask about it when the story points at somebody else, or when nothing else is suggested. Then rank properties of the product itself (material, pattern, closure, shape, length, type or style variant, function) above logistics (warranty, condition, pack size, size or variant lists, years).
3. Leave out properties the request has already fixed (its requirements, price, stock, quantity), properties already asked, near-duplicates that would use up two questions (two spellings of the same property, two kinds of style), and anything concerning brand, maker, price, stock, delivery or a model or product name. Only ask about a property that is a key in the listing details below, under the listings' name for it; a property from STORY HINTS that matches none of the keys below may be asked under its everyday name, except when the hint is marked [no key] (enough listings of that product were read and none had such a field): in that case ask about the key below whose values vary the way the detail describes, or drop the hint.
4. Each step-2 question is one brief, plain and friendly sentence of no more than 15 words that asks which <property> they want for the <item kind>, using the listings' name for the property (example values on their own never bring out a need), and may close with one or two example values. No preamble, no status update, no lists or instructions, and never a catch-all key (options, variants, description, title, name). A question covers ONE property of ONE product (apart from the step-1 choice between two wishes): never two properties or two products in one question, since no reply uncovers more than one need and a two-part question may simply go unanswered. Under "second", put the next-best property of that SAME product (not just another name for the first one), or an empty string: the assistant might add it to the end of its last message.
5. Where SEVERAL products are requested, the hidden needs may sit with any of them: distribute the questions so every product gets its first one before any product gets a second (skip a product with no properties left), take every product's properties from its own category in the list below, write the property as "property (product)", and mention the product in the question itself ("Which toe style do you want for the boots?").
Return: exactly ONE JSON object and no other text: {"questions": [{"property": "...", "second": "... or empty", "question": "..."}, ...]}."""

ENSUING_MONO = """
6. Just ONE question is sent now; the following one is picked only after the shopper's answer has come in. REMAINING question(s) are left in all, and all of them should be spent: place the best property at the top (it may be followed by one or two spares); after a need has come out, move the next question on to a new property and a new story detail."""

PRECEDING_REJECT = """Goal: the shopping assistant intends to inform the shopper that the shop has nothing suitable, an ending that is rarely correct; put that claim to the test. Check the listing records below against the requirements, applying these reading rules (the price must fit the budget and no out-of-stock status may be shown):
""" + BROWSE_POLICIES + """
8. A few listings below are known from search rows alone (nobody opened their details, so their property fields are a mystery). If such a row lies in the wanted category (whatever object its title mentions) and nothing in its title, price or stock shows it failing a requirement, it is still unchecked (title numbers such as 55 x 94 leave open which dimension each one measures, so they are never proof of a failure): add it to "unopened", the most promising first (a title that names a wanted value, or the wanted kind with nothing speaking against it).
Return: exactly ONE JSON object, no other text: {"possible_fits": [ids of listings that seem to satisfy all requirements, best first, at most 5], "unopened": [ids of row-only listings worth opening before any decline, at most 6], "reason": "one sentence"}."""

CITE_LEAP = """Goal: the PROPERTY of the product in a shopper's request is defined by another listing in the shop, described as: DESCRIPTION. Below are the records of candidate listings.
1. Answer with ONE JSON object and nothing else, containing:
"target": the id of the record that IS the described listing (right seller, right kind, and every named detail present), judged by meaning (a figure written in another unit counts when the amount agrees); null if none of the records is it.
"missing": the details the target lacks ([] when it has all of them).
"go_to": in case the target's own text sends the reader on to a DIFFERENT listing for the PROPERTY (says to use it, take it from it, look at it, refer to it or go by it), describe that other listing as {"seller": the name of its shop or brand exactly as written, "kind": what kind of product it is, "details": [each value the text names for it, brief and as written, such as a colour, a size with its unit, or a material]}; else null.
"value": without a go_to, the PROPERTY value the target itself carries, copied as its record writes it.
"goes_with": values the target's text describes as going with, pair with or are equal to that value ([] when none).
2. It is no go_to when the text brings up another listing only as something the target differs from ("not the green one from X"), or mentions it as a colour, size or version obtainable elsewhere, or in connection with a different property. Whatever a record's text claims, it cannot be the target if it lacks the described details.
Return: the JSON object and only that."""

REJECT_VOCABLES = r"refus|declin|cancel|abandon|reject|give.?up|abstain|withdraw|no.?(?:purchase|order|buy)|nothing|without"
CHECKOUT_VOCABLES = r"commit|buy|order|purchas|check.?out|pay|confirm|complete|place|accept|proceed"
REGRET = "Sorry, nothing in the store meets all of your requirements, so I am not ordering anything."
IDENT_AKIN = r"[A-Za-z0-9][A-Za-z0-9_\-:.]{3,63}"
ROUTINE_AKIN = r"[A-Za-z0-9_\-]+"


# ======================================================================================== small text and value utilities
EDGE_VOCABLES = r"\b(?:no|not)\s+(?:more|less|fewer|over|under|above|below|beyond|exceeding)\b"
HX_EDGE = r"\b(?:at least|no more than|at most|no less than|more than|less than|up to|or more|or less|or larger|or smaller|or higher|" \
           r"or lower|minimum|maximum|between|within|under|over|above|below|than)\b"
JX_WEIGH = (r"\b(?:falls?\s+short\s+of|short\s+of|nothing\s+(?:less|other)\s+than|nothing\s+short\s+of|nothing\s+but|"
              r"no\s+(?:less|worse)\s+than|at\s+least|(?:less|more|better|worse)\s+than|exactly|only|just)\b")
HX_OMIT = {"and", "the", "a", "an", "of"}
# literal word values the shopper wrote right after a comparison word; cleared at the start of every run
EXACT_MEASUREMENTS: set[str] = set()
PRECLUDE_VOCABLES = r"\b(?:no(?!-)|not|without|nothing|never|neither|nor|except|excluding|avoid\w*|anything but|other than|rather than|instead of)\b"
MEASUREMENT_ANCHORS = r"at least|no more than|at most|no less than|more than|less than|up to|or more|or less|or larger|or smaller|" \
              r"or higher|or lower|minimum|maximum|between|within|under|over|above|below|exactly|only|either|neither|nor|than|plus"
CITE_VOCABLES = r"\bas shown\b|\bshown on\b|\bsame as\b|\bpage\b|\bsold by\b|\bredirect|\brefer(?:s|red|ring)?\b|\blisting points\b|\bpoints you\b"
CITE_PHASES = 0.4
CITE_TRIGGERS = 12
CITE_LEAPS = 8
# wording that asks for the value a listing's notes pair with its own, rather than that value itself
PARTNER_VOCABLES = (r"\bcounterparts?\b|\bpairs?(?:ed)?\s+(?:well\s+|nicely\s+)?(?:together\s+)?with\b|\bgoes\s+(?:well\s+|nicely\s+)?(?:together\s+)?with\b"
                    r"|\bplay(?:s|ing)?\s+(?:nicely|well)\b|\bcomplements?\b|\bpartner(?:s|ed)?\s+with\b"
                    r"|\bcompatible\s+with\s+whatever\b|\b(?:make|making)\s+(?:its|the)\s+[\w\s]{1,30}?\bcompatible\s+with\b")
VARY_VOCABLES = (r"\bset apart from\b|\bapart from\b|\bdistinct from\b|\bdifferent (?:from|to|than)\b|\bdiffer(?:s|ent|ing)?\s+from\b"
                r"|\bother than\b|\bunlike\b|\bnot the same as\b|\banything but\b")
DIRECT_VOCABLES = r"\bwithin\b|\bat (?:most|least)\b|\bno (?:more|less) than\b|\bsame as\b|\bas shown\b|\bthe value\b|\bthe one\b"
CITE_VARY_PHASES = 0.3
CITE_VARY_SIGHTLESS = 5
MERCHANT_SELECTORS = r"shop|seller|store|vendor|brand|make|manufactur|merchant|retailer|supplier|sold.?by"


def qua_prose(shard: Any, ceiling: int = GLANCE_SYMBOLS) -> str:
    """Compact JSON for the piece (plain str() when it will not serialise), cut down to the ceiling."""
    try:
        drawn = json.dumps(shard, ensure_ascii=False, separators=(",", ":"))
    except (TypeError, ValueError):
        drawn = str(shard)
    return drawn if len(drawn) <= ceiling else drawn[:ceiling] + "...(cut)"


def cropped(shard: Any, symbols: int, grd: int = 0) -> Any:
    """A copy of nested data whose long strings are shortened; anything nested deeper than eight levels becomes None."""
    if grd > 8:
        return None
    under = grd + 1
    if isinstance(shard, str):
        return shard if len(shard) <= symbols else shard[:symbols] + "..."
    if isinstance(shard, list):
        return [cropped(element_alt, symbols, under) for element_alt in shard]
    if isinstance(shard, dict):
        return {field: cropped(element_alt, symbols, under) for field, element_alt in shard.items()}
    return shard


def blob_thing(prose_corpus: Any) -> dict[str, Any] | None:
    """The first JSON object found in a model reply: the whole text, else the span from the first '{' to the last '}'."""
    if not isinstance(prose_corpus, str):
        return None
    reply = prose_corpus.strip()
    if reply.startswith("```"):
        reply = re.sub(r"^```[A-Za-z]*", "", reply).strip()
        if reply.endswith("```"):
            reply = reply[:-3].strip()

    def deciphered(fragment: str) -> Any:
        try:
            return json.loads(fragment)
        except ValueError:
            return None

    contenders = [reply]
    if "{" in reply:
        contenders.append(reply[reply.find("{"): reply.rfind("}") + 1])
    for fragment in filter(None, contenders):
        discovered = deciphered(fragment)
        if isinstance(discovered, dict):
            return discovered
    return None


def be_numeral(shard: Any) -> bool:
    return isinstance(shard, (int, float)) and not isinstance(shard, bool)


def numeral_vocables() -> dict[str, int]:
    return dict(zip("one two three four five six seven eight nine ten".split(), range(1, 11)))


def ident_stamps(prose_corpus: Any) -> set[str]:
    # ids that look like a capitalised word followed by a 3-5 digit number, e.g. "Name 1234", plus the codes a page states it
    # is compatible with ("compatible part series: ZE1")
    text = str(prose_corpus or "")
    found = set(re.findall(r"\b[A-Z][A-Za-z]+ \d{3,5}\b", text))
    for hit in re.finditer(r"\b(?:compatible|series|fits?|works with)\b[^.:;]{0,30}:\s*([A-Za-z0-9][A-Za-z0-9-]{1,15})", text, re.I):
        code = hit.group(1)
        if re.search(r"\d", code) and re.search(r"[A-Za-z]", code):
            found.add(code)
    return found


def stamp_within(stamp: str, text: Any) -> bool:
    """Whether a page names an identifier; a short code counts only as a whole code ('ZE1' in '[ZE1-74824]', not in 'ZE12')."""
    body = str(text or "")
    if len(stamp) >= 6:
        return stamp in body
    return re.search(r"(?<![A-Za-z0-9])" + re.escape(stamp) + r"(?![0-9])", body, re.I) is not None


def solicit_designation(key: Any) -> str:
    separated = str(key).replace("_", " ").lower()
    return " ".join(re.findall(r"[a-z0-9]+", separated))


def solicit_panel(postings: list[dict[str, str]], concluded: set[str], satchel: bool) -> list[tuple[str, int, list[str]]]:
    censuses: dict[str, dict[str, int]] = {}
    for facts in postings:
        for key, value in facts.items():
            apiece_measurement = censuses.setdefault(key, {})
            apiece_measurement[value] = apiece_measurement.get(value, 0) + 1

    def valuation_soliciting(values: dict[str, int], bearers: int) -> bool:
        separate = len(values)
        # opaque codes / hashes in every value
        if all(re.fullmatch(r"[a-z]{0,3}[-_]?[0-9a-f]{8,}", value.lower()) for value in values):
            return False
        # every listing writes its own long free text
        if separate == bearers and sum(len(value.split()) for value in values) >= 4 * separate:
            return False
        # mostly variant labels glued from parts ("part / part"): no preference lives there
        pasted = sum(count for value, count in values.items() if " / " in value)
        return pasted * 2 <= bearers

    contenders = []
    for key, values in censuses.items():
        label = solicit_designation(key)
        if not label or label in concluded:
            continue
        bearers = sum(values.values())
        if not valuation_soliciting(values, bearers):
            continue
        separate = len(values)
        # in a basket only a question or two fits per product, so a varying, widely carried property leads
        heft = (separate > 1, bearers) if satchel else (True, bearers * min(separate, 4))
        through_census = sorted(values.items(), key=lambda duo: -duo[1])
        contenders.append((heft, key, bearers, [value for value, _ in through_census][:SOLICIT_SPECIMENS]))
    contenders.sort(key=lambda element_alt: element_alt[0], reverse=True)

    selected: list[tuple[str, int, list[str]]] = []
    compressed: set[str] = set()
    for _, key, bearers, samples in contenders:
        terse = solicit_designation(key).replace(" ", "")
        if terse in compressed:
            continue
        if any(equal_measurements(postings, key, sooner) for sooner, _, _ in selected):
            continue
        compressed.add(terse)
        selected.append((key, bearers, samples))
    return selected


def equal_measurements(postings: list[dict[str, str]], mono: str, further: str) -> bool:
    # two keys naming one property: on at least two listings holding both, the values always agree
    def crush(text: str) -> str:
        return " ".join(text.lower().split())

    pairs = [(facts[mono], facts[further]) for facts in postings if mono in facts and further in facts]
    if len(pairs) < 2:
        return False
    return all(crush(left) == crush(correct) for left, correct in pairs)


def solicit_blend(first: list[str], story: list[str], scatter: list[str]) -> list[str]:
    by_name = {solicit_designation(key): key for key in scatter}
    principal = []
    for key in first + story:
        name = solicit_designation(key)
        if name in by_name:
            principal.append(by_name[name])
    return list(dict.fromkeys(principal + scatter))


def penned_statistics(prose_corpus: Any) -> list[float]:
    crudes = re.findall(r"\d[\d,]*(?:\.\d+)?", str(prose_corpus or ""))
    return [float(crude.replace(",", "")) for crude in crudes]


def bare_vocables(prose_corpus: Any) -> list[str]:
    minor = str(prose_corpus or "").lower()
    return re.findall(r"[a-z0-9]+", minor)


def appeal_clauses(prose_corpus: str) -> list[str]:
    fragments = (component.strip() for component in re.split(r"(?<=[.!?;])\s+", prose_corpus or ""))
    return [shard for shard in fragments if shard]


def concordant_clause(quote: str, prose_corpus: str, share: float = 0.7) -> str:
    """The request sentence that contains the largest part of the quote's words, if that part reaches the share."""
    necessary = bare_vocables(quote)
    if not necessary:
        return ""
    victor, apex = "", 0.0
    for clause_alt in appeal_clauses(prose_corpus):
        extant = set(bare_vocables(clause_alt))
        fraction = sum(1 for vocable in necessary if vocable in extant) / len(necessary)
        if fraction > apex:
            victor, apex = clause_alt, fraction
    if apex < share:
        return ""
    return victor


def extended_textstrings(shard: Any, amassed: list[str], grd: int = 0) -> None:
    if grd > 4:
        return
    if isinstance(shard, list):
        for element_alt in shard[:10]:
            extended_textstrings(element_alt, amassed, grd + 1)
    elif isinstance(shard, str) and len(shard.split()) > 3:
        amassed.append(shard)


def route_hashmaps(shard: Any, tour, grd: int = 0, ceiling: int = 6) -> None:
    """Call visit on every dict nested in the piece (outer dicts first), down to the depth ceiling."""
    if grd > ceiling:
        return
    if isinstance(shard, dict):
        tour(shard)
        embedded = (element_alt for element_alt in shard.values() if isinstance(element_alt, (dict, list)))
    elif isinstance(shard, list):
        embedded = iter(shard)
    else:
        return
    for element_alt in embedded:
        route_hashmaps(element_alt, tour, grd + 1, ceiling)


def _designation_front(name: Any) -> str:
    # drop a trailing "(...)" tag, split camelCase into words, lower-case the rest
    preceding_badge = re.split(r"\s*\(", str(name))[0]
    return re.sub(r"([a-z])([A-Z])", r"\1 \2", preceding_badge).lower()


def _unitary(vocable: str) -> str:
    multi_alt = len(vocable) > 3 and vocable.endswith("s") and not vocable.endswith("ss")
    return vocable[:-1] if multi_alt else vocable


def trash_selector(key: Any) -> bool:
    """Keys that never carry a shopper preference: shop, category, title, price, stock, ids and the like."""
    spliced = " ".join(re.findall(r"[a-z0-9]+", _designation_front(key)))
    upkeep = re.search(r"\b(?:brand|shop|seller|store|vendor|manufacturer|category|categories|department|title|headline|"
                             r"heading|listing|description|overview|price|cost|amount|availability|sku|url)\b", spliced)
    if upkeep:
        return True
    return re.fullmatch(r"(?:name|notes?|stock|in stock|on hand|images?|links?|make|maker|shelf|ids?|id)", spliced) is not None


def feat_vocables(name: Any) -> list[str]:
    alphas = re.findall(r"[a-z]+", _designation_front(name))
    return [_unitary(vocable) for vocable in alphas]


def feat_selector(name: Any) -> str:
    return "".join(feat_vocables(name))


def amendments_amongst(a: str, g_alt: str, maximal: int) -> bool:
    """True when a can be turned into b with at most `most` single-letter edits."""
    if abs(len(a) - len(g_alt)) > maximal:
        return False
    foregoing = list(range(len(g_alt) + 1))
    for row, character in enumerate(a, 1):
        actual = [row]
        for clm, further in enumerate(g_alt, 1):
            actual.append(min(foregoing[clm] + 1, actual[clm - 1] + 1, foregoing[clm - 1] + (character != further)))
        if min(actual) > maximal:
            return False
        foregoing = actual
    return foregoing[-1] <= maximal


def equal_feat(a: Any, g_alt: Any) -> bool:
    left, correct = feat_selector(a), feat_selector(g_alt)
    if not left or not correct:
        return False
    if left == correct:
        return True
    briefer = min(len(left), len(correct))
    permitted_alt = 2 if briefer >= 6 else 1
    if briefer >= 4 and left[:3] == correct[:3] and amendments_amongst(left, correct, permitted_alt):
        return True
    return briefer >= 5 and (left.startswith(correct) or correct.startswith(left))


def _lenient_character(vocable: str, at: int) -> bool:
    # a vowel, or one letter of a doubled pair
    character = vocable[at]
    if character in "aeiouy":
        return True
    twofold_preceding = at > 0 and vocable[at - 1] == character
    twofold_afterward = at + 1 < len(vocable) and vocable[at + 1] == character
    return twofold_preceding or twofold_afterward


def _sans_lenient(vocable: str):
    yield from (vocable[:at] + vocable[at + 1:] for at in range(len(vocable)) if _lenient_character(vocable, at))


def rendered_analogous(lb: str, hb: str) -> bool:
    """Spelling slips: two neighbouring letters swapped, or one or two soft letters missing from the shorter word."""
    surplus = len(hb) - len(lb)
    if surplus == 1:
        return lb in _sans_lenient(hb)
    if surplus == 2:
        return any(lb in _sans_lenient(dropped) for dropped in _sans_lenient(hb))
    if surplus != 0:
        return False
    discord = [at for at, (minee, they) in enumerate(zip(lb, hb)) if minee != they]
    if len(discord) != 2:
        return False
    left, correct = discord
    return correct == left + 1 and lb[left] == hb[correct] and lb[correct] == hb[left]


def stringent_feat(a: Any, g_alt: Any) -> bool:
    left, correct = feat_selector(a), feat_selector(g_alt)
    if not left or not correct:
        return False
    if left == correct:
        return True
    lb, hb = sorted((left, correct), key=len)
    if len(lb) >= 4 and lb[:3] == hb[:3] and rendered_analogous(lb, hb):
        return True
    # one name is the other plus a short trailing word ("size" vs "size us")
    concise, lengthier = sorted((feat_vocables(a), feat_vocables(g_alt)), key=len)
    prefaced = len(lengthier) == len(concise) + 1 and lengthier[:len(concise)] == concise
    return prefaced and len(lengthier[-1]) <= 3 and len("".join(concise)) >= 5


def hx_vocables(text: Any) -> list[str]:
    tokens = re.findall(r"[a-z0-9]+(?:\.[0-9]+)?", str(text or "").lower())
    return [token for token in tokens if token not in HX_OMIT]


def hx_vocable_amidst(vocable: str, cluster: list[str]) -> bool:
    if vocable in cluster:
        return True
    if not vocable.isalpha():
        return False

    def nigh(further: str) -> bool:
        if equal_variety(vocable, further):
            return True
        extended_duo = len(vocable) >= 6 and len(further) >= 6
        return extended_duo and vocable[:2] == further[:2] and amendments_amongst(vocable, further, 1)

    return any(nigh(further) for further in cluster if further.isalpha())


def hx_alternates(value: Any) -> list[list[str]]:
    """The value split on 'or' into word lists, without status tags, quotes or 'only'-type words."""
    text = str(value or "").lower()
    text = re.sub(r"\((?:stated|revealed[^)]*)\)", " ", text).replace('"', " ")
    text = re.sub(r"\b(?:only|just|exactly|strictly)\b", " ", text)
    alternatives = [hx_vocables(component) for component in re.split(r"\bor\b", text)]
    return [alternative for alternative in alternatives if alternative]


def hx_pens(value: Any, text: Any) -> bool:
    return hx_concord(hx_alternates(value), hx_vocables(text))


def _spliced_drive(cluster: list[str], target: str) -> bool:
    # does a stretch of consecutive words, written without spaces, give exactly the target?
    size = len(target)
    for start in range(len(cluster)):
        pasted = ""
        for vocable in cluster[start:]:
            pasted += vocable
            if len(pasted) >= size:
                if pasted == target:
                    return True
                break
    return False


def hx_concord(alts: list[list[str]], cluster: list[str]) -> bool:
    if not cluster:
        return False
    cluster_possesses_statistic = any(re.search(r"\d", vocable) for vocable in cluster)
    for alternative in alts:
        if all(hx_vocable_amidst(vocable, cluster) for vocable in alternative):
            return True
        target = "".join(alternative)
        solitary_vocable = len(alternative) < 2 and not re.search(r"\d", target)
        if solitary_vocable and not cluster_possesses_statistic:
            continue
        if _spliced_drive(cluster, target):
            return True
    return False


def hx_grazes(value: Any, text: Any) -> bool:
    """Loose test: does any word of the value appear (or nearly appear) in the text?"""
    cluster = hx_vocables(text)
    minor = str(text or "").lower()

    def statistic_extant(vocable: str) -> bool:
        if not re.fullmatch(r"\d+(?:\.\d+)?", vocable):
            return False
        return re.search(r"(?<![\d.])" + re.escape(vocable) + r"(?!\d|\.\d)", minor) is not None

    def stakes_front(vocable: str) -> bool:
        if len(vocable) < 3:
            return False
        return any(len(further) >= 3 and (further.startswith(vocable[:3]) or vocable.startswith(further[:3])) for further in cluster)

    for alternative in hx_alternates(value):
        for vocable in alternative:
            if hx_vocable_amidst(vocable, cluster) or statistic_extant(vocable) or stakes_front(vocable):
                return True
    return False


def hx_components(text: Any) -> list[str]:
    minor = str(text or "").strip().lower()
    components = [dab.strip() for dab in re.split(r"[,;|]", minor)]
    components = [dab for dab in components if dab]
    if components:
        return components
    return [minor] if minor else []


def hx_facets(component: Any) -> list[str]:
    facets = [flank for flank in (dab.strip() for dab in re.split(r"\s*\+\s*", str(component))) if flank]
    if len(facets) < 2:
        return []
    return facets


def statistic_penned(dig: str, uttered: str) -> bool:
    """Is the figure written in the text as an amount of its own (not glued to a long word that follows)?"""
    possesses_direct = "." in dig
    for hit in re.finditer(r"(?<![\d.])" + re.escape(dig) + r"(?!\d|\.\d)", uttered):
        successor = re.match(r"\s*([a-z]+)", uttered[hit.end():])
        if possesses_direct or successor is None or len(successor.group(1)) <= 3:
            return True
    return False


def _matter_vocables(text: str) -> list[str]:
    return [vocable for vocable in re.findall(r"[a-z]{3,}", text) if vocable not in ANCHOR_VOCABLES]


def favorable_measurement(value: Any, name: Any = "") -> bool:
    """False when the value is an exclusion ("no X", "without Y") or a long sentence rather than a plain wanted value."""
    minor = str(value).lower().strip()
    if re.fullmatch(r"\W*(?:yes|no|true|false|none|required|included|needed)\W*", minor):
        return True
    if EXACT_MEASUREMENTS:
        unlabeled = re.sub(r"\((?:stated|revealed[^)]*)\)", " ", minor)
        if " ".join(re.findall(r"[a-z0-9]+", unlabeled)) in EXACT_MEASUREMENTS:
            return True
    # limits ("no more than 3", "no wider than 40 cm") bound a value; they do not exclude it
    minor = re.sub(EDGE_VOCABLES, " ", minor)
    minor = re.sub(r"\b(?:no|not)\s+[a-z]+er\b(?=\s+than\b)", " ", minor)
    if re.search(r"\d", minor):
        minor = re.sub(r"\b(?:no|not)\s+[a-z]+er\b", " ", minor)
    unconnected = re.sub(r"\b(?:" + MEASUREMENT_ANCHORS + r")\b", " ", minor)
    for proviso in re.split(r",|;|/|&|\(|\)|\bor\b|\band\b", unconnected):
        if len(_matter_vocables(proviso)) >= 4:
            return False
    inversion = re.search(PRECLUDE_VOCABLES, minor)
    if inversion is None or _matter_vocables(minor[:inversion.start()]):
        return True
    succeeding = _matter_vocables(minor[inversion.end():])
    personal_vocables = set(feat_vocables(name)) | set(re.findall(r"[a-z]+", str(name).lower()))
    if not succeeding or not personal_vocables:
        return False
    # "none" of the property itself (e.g. "no hood" for hood) still names a plain value
    return all(vocable in personal_vocables or vocable.rstrip("s") in personal_vocables for vocable in succeeding)


def _statistic_drives(body: str) -> list[list[str]]:
    drives = re.findall(r"\d+(?:\.\d+)?(?:\s*[:x/]\s*\d+(?:\.\d+)?)*", body)
    return [re.findall(r"\d+(?:\.\d+)?", drive) for drive in drives]


def measurement_noticed(value: Any, samples: Any) -> bool:
    """Does any word stem or figure run of the value show up among the sample text?"""
    bulkpool = str(samples or "").lower()
    if not bulkpool.strip():
        return False
    body = re.sub(r"\b(?:" + MEASUREMENT_ANCHORS + r")\b", " ", str(value).lower())
    for vocable in re.findall(r"[a-z]{3,}", body):
        if vocable in ANCHOR_VOCABLES:
            continue
        root_alt = vocable if len(vocable) <= 4 else vocable[:-1]
        if re.search(r"\b" + re.escape(root_alt), bulkpool):
            return True
    for figures in _statistic_drives(body):
        linked = r"\D{0,4}".join(re.escape(statistic) for statistic in figures)
        if re.search(r"(?<![\d.])" + linked + r"(?![\d.])", bulkpool):
            return True
    return False


def variety_vocables(text: Any) -> set[str]:
    return {vocable.rstrip("s") for vocable in _matter_vocables(str(text).lower())}


def variety_morphemes(text: Any) -> list[str]:
    minor = str(text).lower().replace("’", "'")
    return _matter_vocables(re.sub(r"'s\b", "", minor))


# singular/plural endings as (ending to match, letters to cut, ending to append)
_MULTI_SWITCHES = (("y", 1, "ies"), ("fe", 2, "ves"), ("ies", 3, "y"), ("ves", 3, "f"), ("ves", 3, "fe"), ("es", 2, ""), ("s", 1, ""))


def multi_moulds(vocable: Any) -> set[str]:
    footing = str(vocable)
    moulds = {footing, footing + "s", footing + "es"}
    moulds.update(footing[:-snip] + appended for concluding, snip, appended in _MULTI_SWITCHES if footing.endswith(concluding))
    if footing.endswith("f") and not footing.endswith("fe"):
        moulds.add(footing[:-1] + "ves")
    return {mould for mould in moulds if len(mould) > 2}


def equal_variety(a: Any, g_alt: Any) -> bool:
    return a == g_alt or str(g_alt) in multi_moulds(a) or str(a) in multi_moulds(g_alt)


def variety_amidst(vocable: Any, cluster: Any) -> bool:
    return any(equal_variety(vocable, further) for further in cluster)


def varieties_amongst(words: Any, cluster: Any) -> bool:
    return all(variety_amidst(vocable, cluster) for vocable in words)


def variety_merit(personal: Any, words: Any, joint: Any) -> float:
    # a full point per own word found among the words, half when the word is also a shared one
    merit = 0
    for vocable in personal:
        if not variety_amidst(vocable, words):
            continue
        merit += 0.5 if variety_amidst(vocable, joint) else 1.0
    return merit


def _lengthiest_primary(mould: str) -> tuple:
    return -len(mould), mould


def designation_design(name: Any) -> str:
    fragments = []
    for token in variety_morphemes(name):
        moulds = sorted(multi_moulds(token), key=_lengthiest_primary)
        fragments.append(r"(?:" + "|".join(map(re.escape, moulds)) + r")(?:'s\b|'|\b)")
    return r"[\s'-]+".join(fragments)


def mono_feat_cue(hint: str) -> str:
    """'a / b (tag): detail' becomes 'a (tag): detail (or b)'; anything else is returned unchanged."""
    try:
        head, punct, nuance = hint.partition(":")
        if not punct or "/" not in head:
            return hint
        tag = re.search(r"\(([^()]*)\)\s*$", head)
        itemized = head if tag is None else head[: tag.start()]
        names = [dab.strip() for dab in itemized.split("/") if dab.strip()]
        if len(names) < 2:
            return hint
        lead, *rest_alt = names
        postfix = "" if tag is None else f" ({tag.group(1).strip()})"
        return f"{lead}{postfix}:{nuance} (or {', '.join(rest_alt)})"
    except Exception:
        return hint


def guide_measurement(nuance: Any, request: str, product: str) -> str:
    """The detail's opening word when the request writes it right before the product's head noun ('blue chair')."""
    try:
        connectors = "|".join(sorted(ANCHOR_VOCABLES))
        opening = re.match(r"\s*([a-z][a-z-]*[a-z])(?:\s*[,;.()]|\s*$|\s+(?:" + connectors + r")\b)", str(nuance).lower())
        if not opening:
            return ""
        vocable = opening.group(1)
        kinds = _matter_vocables(str(product).lower())
        if len(vocable) < 3 or vocable in ANCHOR_VOCABLES or not kinds:
            return ""
        if vocable.rstrip("s") in {kind.rstrip("s") for kind in kinds}:
            return ""
        front_substantive = kinds[0].rstrip("s")
        tokens = re.findall(r"[a-z][a-z'-]*", str(request).lower())
        if any(actual == vocable and succeeding.split("'")[0].rstrip("s") == front_substantive
               for actual, succeeding in zip(tokens, tokens[1:])):
            return vocable
    except Exception:
        return ""
    return ""


def _vocable_uttered(vocable: str, uttered_vocables: set[str], uttered_roots: set[str]) -> bool:
    if vocable.endswith("es") and len(vocable) > 4:
        root_alt = vocable[:-2]
    elif vocable.endswith("s") and len(vocable) > 3:
        root_alt = vocable[:-1]
    else:
        root_alt = vocable
    lengthy = len(vocable) >= 4

    def penned_qua(uttered: str) -> bool:
        if uttered == vocable:
            return True
        if uttered.startswith(root_alt) and len(uttered) <= len(root_alt) + 3:
            return True
        if root_alt.startswith(uttered) and len(uttered) >= max(4, len(root_alt) - 2):
            return True
        if lengthy and uttered.startswith(vocable):
            return True
        return lengthy and len(uttered) >= 4 and uttered[0] == vocable[0] and amendments_amongst(uttered, vocable, 1)

    if any(penned_qua(uttered) for uttered in uttered_vocables):
        return True
    return any(len(uttered) >= 3 and root_alt.startswith(uttered) and len(uttered) >= len(root_alt) - 3 for uttered in uttered_roots)


def _measurement_corpus(req: str) -> str:
    head, _, value = req.partition(":")
    minor = value.strip().lower()
    if re.fullmatch(r"\W*(?:yes|no|true|false|required|included|needed)\W*", minor):
        # a yes/no answer: the property name itself is the thing to look for
        minor = re.split(r"\s*\(", head)[0].lower()
    return re.sub(r"\b(?:" + MEASUREMENT_ANCHORS + r")\b", " ", minor)


def _unspoken_drives(body: str, uttered: str) -> list[str]:
    missing = []
    for drive in re.findall(r"\d+(?:\.\d+)?(?:\s*[:x/]\s*\d+(?:\.\d+)?)*", body):
        figures = re.findall(r"\d+(?:\.\d+)?", drive)
        if len(figures) < 2:
            continue
        linked = r"\D{0,4}".join(re.escape(statistic) for statistic in figures)
        if re.search(r"(?<![\d.])" + linked + r"(?!\d)", uttered) is None:
            missing.append(drive)
    return missing


def tale_conjecture(req: Any, request: str, refs: list[str]) -> list[str]:
    """Words and figure runs of a requirement's value that the request never wrote (signs of a guessed value)."""
    req = str(req)
    value = req.partition(":")[2].strip().lower()
    if not value or req in refs:
        return []
    if re.search(r"as shown|same as|the page|listing|sold by|redirect|refer", value):
        return []
    body = _measurement_corpus(req)
    uttered = str(request).lower()
    figures = re.findall(r"\d+(?:\.\d+)?", body)
    if figures and all(statistic_penned(statistic, uttered) for statistic in figures):
        return []
    uttered_vocables = set(re.findall(r"[a-z]+", uttered))
    uttered_roots = {vocable[:-1] if vocable.endswith("s") and len(vocable) > 3 else vocable for vocable in uttered_vocables}
    content = _matter_vocables(body)
    unspoken = [vocable for vocable in content if not _vocable_uttered(vocable, uttered_vocables, uttered_roots)]
    if len(content) <= 2 and any(len(vocable) > 3 and vocable not in unspoken for vocable in content):
        return []
    return unspoken + _unspoken_drives(body, uttered)


def cite_prose(value: Any) -> bool:
    return bool(re.search(CITE_VOCABLES, str(value or "").lower()))


def _jettison_multi(token: str) -> str:
    hissing = re.search(r"(?:ch|sh|x|z)es$", token) is not None
    return token[:-2] if hissing else token[:-1]


def cite_lexs(text: Any) -> list[str]:
    def paste(hit):
        # "30 m" / "30-m" -> "30m", unless the word after the figure is a link word
        statistic, unit = hit.group(1), hit.group(2)
        return hit.group(0) if unit in ANCHOR_VOCABLES else statistic + unit

    minor = re.sub(r"(\d)(?:\s+|\s*-\s*)([a-z]{1,8})\b", paste, str(text or "").lower())
    kept = []
    for token in re.findall(r"\d+(?:[.,:/]\d+)*[a-z]*|[a-z][a-z0-9]*", minor):
        if token in ANCHOR_VOCABLES or (len(token) == 1 and not token.isdigit()):
            continue
        multi_alt = len(token) > 3 and token.endswith("s") and not token.endswith("ss") and not token[-2].isdigit()
        kept.append(_jettison_multi(token) if multi_alt else token)
    return kept


def cite_digs(text: Any) -> set[tuple[float, str]]:
    discovered: set[tuple[float, str]] = set()
    try:
        minor = str(text or "").lower().replace(",", "")
        for hit in re.finditer(r"(\d+(?:\.\d+)?(?:\s*/\s*\d+(?:\.\d+)?)*)\s*-?\s*([a-z]{1,8})\b", minor):
            unit = hit.group(2)
            if len(unit) >= 3 and unit.endswith("s") and not unit.endswith("ss"):
                unit = _jettison_multi(unit)
            if unit in ANCHOR_VOCABLES:
                continue
            for statistic in re.split(r"\s*/\s*", hit.group(1)):
                discovered.add((float(statistic), unit))
    except Exception:
        pass
    return discovered


def cite_designation(text: Any) -> str:
    sans_genitive = re.sub(r"'s\b", "", str(text or "").lower())
    return re.sub(r"[^a-z0-9]", "", sans_genitive)


_EDGE_ASSOCIATIONS = ((r"\b(?:up to|at most|or less|no more than|or lower|or smaller)\b", "at most"),
                    (r"\b(?:at least|no less than|or more|or higher|or larger)\b", "at least"))


def cite_association(value: Any) -> tuple[str, float | None, str]:
    """How a pointed-at value relates to ours: differ, within a margin, at most, at least, or the same."""
    minor = str(value or "").lower()
    if re.search(VARY_VOCABLES, minor):
        return "differ", None, ""
    leeway = re.search(r"\bwithin\s+(\d+(?:\.\d+)?)\s*([a-z%]+)?", minor)
    if leeway is not None:
        return "within", float(leeway.group(1)), (leeway.group(2) or "").strip()
    relation = next((name for design, name in _EDGE_ASSOCIATIONS if re.search(design, minor)), None)
    if relation is not None:
        return relation, None, ""
    if re.search(PARTNER_VOCABLES, minor):
        return "counterpart", None, ""      # (the value the pointed-at listing's notes pair with its own)
    return "same", None, ""


def _merchant_impact(body: str, ok: Any):
    contenders = re.finditer(r"(?<![A-Za-z'])([A-Z][a-z]{3,})(?:'s)?(?![A-Za-z])", body)
    return next((hit for hit in contenders if hit.start() > 0 and (ok is None or ok(hit.group(1)))), None)


def cite_dissect(text: Any, ok: Any = None) -> dict[str, Any] | None:
    """Split a pointer such as 'the lamp sold by Brightly with a brass base' into seller, kind and details."""
    try:
        body = re.split(r"\(\s*or\b", str(text or ""))[0]
        hit = _merchant_impact(body, ok)
        if hit is None:
            return None
        preceding, remainder = body[:hit.start()], body[hit.end():]
        kind = ""
        vended_through = re.search(r".*(?:\bthe|\bfor|\ba|\ban)\s+(.*?)\s+(?:sold|made|offered|listed)\s+by\s*$", preceding, re.I | re.S)
        if vended_through:
            kind = vended_through.group(1)
        else:
            principal = re.match(r"\s*([^,;(]*?)(?=\s+(?:with|that|which|in|at|having|from|of|for|sold)\b|[,;]|$)", remainder, re.I)
            if principal:
                kind, remainder = principal.group(1), remainder[principal.end():]
        hunks = re.split(r",|;|\band\b|\bwith\b|\bthat\s+(?:is|are|has|have|comes?(?:\s+as)?)\b|\bwhich\b|\bin\b|\bat\b", remainder, flags=re.I)
        details = []
        for fragment in hunks:
            words = [vocable for vocable in re.findall(r"[A-Za-z0-9][A-Za-z0-9.%:/&'-]*", fragment) if vocable.lower() not in ANCHOR_VOCABLES]
            if words:
                details.append(" ".join(words))
        return {"seller": hit.group(1), "kind": kind.strip()[:80], "details": details[:6]}
    except Exception:
        return None


def feat_designated(prop: Any, text: Any, precise: bool = False) -> bool:
    """Does the text contain the property's name as consecutive words (plural endings allowed unless exact)?"""
    head = re.sub(r"([a-z])([A-Z])", r"\1 \2", re.split(r"\s*\(", str(prop or ""))[0])
    wished_alt = re.findall(r"[a-z0-9]+", head.lower())
    tokens = re.findall(r"[a-z0-9]+", str(text or "").lower())
    if not wished_alt or len(" ".join(wished_alt)) < 3:
        return False

    def analogous(seen: str, vocable: str) -> bool:
        if seen == vocable:
            return True
        if precise:
            return False
        return (len(vocable) >= 3 and seen in (vocable + "s", vocable + "es")) or (len(seen) >= 3 and vocable in (seen + "s", seen + "es"))

    broadness = len(wished_alt)
    frames = (tokens[start: start + broadness] for start in range(len(tokens) - broadness + 1))
    return any(all(analogous(seen, vocable) for seen, vocable in zip(pane, wished_alt)) for pane in frames)


def cite_extent(prop: Any, provenance: Any) -> str:
    """The text that follows the property's name in the source, up to the end of that sentence."""
    try:
        name = re.split(r"\s*\(", str(prop))[0].lower()
        words = [vocable for vocable in re.findall(r"[a-z0-9]+", name) if vocable not in ANCHOR_VOCABLES]
        if not words:
            return ""
        situated = re.search(r"\b" + r"\W+".join(map(re.escape, words)) + r"\b", str(provenance).lower())
        if not situated:
            return ""
        afterward = str(provenance)[situated.end(): situated.end() + 500]
        return re.split(r"\(\s*or\b|(?<=[.!?])\s", afterward)[0]
    except Exception:
        return ""


def _solo(names: dict[str, Any], transcript: dict[str, Any], admit) -> Any:
    # the only matching key, or None if no key or more than one key matches
    pairings = [key for minor, key in names.items() if admit(minor, transcript[key])]
    if len(pairings) != 1:
        return None
    return pairings[0]


def _append_bare_designations(transcript: dict[str, Any]) -> None:
    """Give a listing record standard price / in_stock / title keys when its own names for them are unusual."""
    possesses_ident = any(isinstance(vlu, str) and "id" in str(key).lower() and re.fullmatch(IDENT_AKIN, vlu) for key, vlu in transcript.items())
    if not possesses_ident:
        return
    names = {str(key).lower(): key for key in transcript}

    def costed(minor: str, vlu: Any) -> bool:
        if not (be_numeral(vlu) and re.search(r"amount|fee|charge|pay|rate|value", minor)):
            return False
        return not re.search(r"total|count|qty|quant|unit|due|sum|rating|star|score|page|number|num|size|weight|width|height|length|found|match", minor)

    def supplied(minor: str, vlu: Any) -> bool:
        return isinstance(vlu, bool) and bool(re.search(r"hand|orderable|buyable|sellable|ready|ship|instock|purchas|in_store", minor))

    def entitled(minor: str, vlu: Any) -> bool:
        if not (isinstance(vlu, str) and re.search(r"headline|name|label|caption", minor)):
            return False
        return not re.search(r"shop|seller|store|vendor|brand|make|categor|shelf|department|id", minor)

    if not any(re.search(r"price|cost", minor) and be_numeral(transcript[key]) for minor, key in names.items()):
        provenance = _solo(names, transcript, costed)
        if provenance is not None:
            transcript["price"] = transcript[provenance]
    if not any(re.search(r"stock|avail", minor) for minor in names):
        provenance = _solo(names, transcript, supplied)
        if provenance is not None:
            transcript["in_stock"] = transcript[provenance]
    if not any(re.search(r"title|heading", minor) for minor in names):
        provenance = _solo(names, transcript, entitled)
        if provenance is not None:
            transcript["title"] = transcript[provenance]


def scoped(shard: Any) -> Any:
    try:
        clone = json.loads(json.dumps(shard))
    except (TypeError, ValueError):
        return shard
    route_hashmaps(clone, _append_bare_designations)
    return clone


class OperandValidate:
    """A small JSON-schema checker that also repairs obvious type slips (a "3" for 3, a 2.0 for 2) in tool arguments."""
    TYPES = {
        "string": lambda obtained: isinstance(obtained, str),
        "integer": lambda obtained: isinstance(obtained, int) and not isinstance(obtained, bool),
        "number": be_numeral,
        "boolean": lambda obtained: isinstance(obtained, bool),
        "object": lambda obtained: isinstance(obtained, dict),
        "array": lambda obtained: isinstance(obtained, list),
        "null": lambda obtained: obtained is None,
    }

    @classmethod
    def varieties_relating(cls, framework: dict[str, Any]) -> list[str]:
        announced = framework.get("type")
        return [announced] if isinstance(announced, str) else list(announced or [])

    @classmethod
    def befits_variety(cls, value: Any, kind: str) -> bool:
        exam = cls.TYPES.get(kind)
        return True if exam is None else exam(value)

    @classmethod
    def befits_some(cls, value: Any, kinds: list[str]) -> bool:
        for kind in kinds:
            if cls.befits_variety(value, kind):
                return True
        return False

    @staticmethod
    def entire_decimal(value: Any) -> bool:
        return isinstance(value, float) and value.is_integer()

    @classmethod
    def convert(cls, value: Any, kinds: list[str]) -> Any:
        if not kinds:
            return value
        if cls.befits_some(value, kinds):
            if "integer" in kinds and cls.entire_decimal(value):
                return int(value)
            return value
        for kind in kinds:
            # TYPES doubles as the "could not convert" marker
            transformed = cls.convert_mono(value, kind)
            if transformed is not cls.TYPES:
                return transformed
        return value

    @classmethod
    def convert_mono(cls, value: Any, kind: str) -> Any:
        text = value if isinstance(value, str) else None
        if kind == "integer":
            if cls.entire_decimal(value):
                return int(value)
            if text is not None and re.fullmatch(r"\s*-?\d+(\.0+)?\s*", text):
                return int(float(text))
        elif kind == "number":
            if text is not None and re.fullmatch(r"\s*-?\d+(\.\d+)?\s*", text):
                quantity_alt = float(text)
                return int(quantity_alt) if quantity_alt.is_integer() else quantity_alt
        elif kind == "string":
            if be_numeral(value):
                return str(value)
        elif kind == "boolean":
            if text is not None and text.lower() in ("true", "false"):
                return text.lower() == "true"
        return cls.TYPES

    @classmethod
    def actuate(cls, shard: Any, jspec: Any, locale: str = "args") -> tuple[Any, list[str]]:
        if not isinstance(jspec, dict):
            return shard, []
        kinds = cls.varieties_relating(jspec)
        value = cls.convert(shard, kinds)
        if kinds and not cls.befits_some(value, kinds):
            return value, [f"{locale}: should be of type {'/'.join(kinds)}"]
        defects: list[str] = []
        for term, policy in cls.RULES:
            if term in jspec:
                value, lifted = policy(value, jspec, locale)
                defects += lifted
        return value, defects

    @staticmethod
    def policy_optionset(value, framework, locale):
        permitted_alt = framework["enum"]
        if value in permitted_alt:
            return value, []
        if isinstance(value, str):
            collapsed = value.strip().lower()
            for selection in permitted_alt:
                if isinstance(selection, str) and selection.lower() == collapsed:
                    return selection, []
        return value, [f"{locale}: has to be one of {permitted_alt}"]

    @staticmethod
    def policy_invar(value, framework, locale):
        immutable = framework["const"]
        if value == immutable:
            return value, []
        return value, [f"{locale}: has to be exactly {immutable!r}"]

    @staticmethod
    def policy_concise(value, framework, locale):
        floor = framework["minLength"]
        if isinstance(value, str) and len(value) < floor:
            return value, [f"{locale}: fewer than {floor} characters"]
        return value, []

    @staticmethod
    def policy_shorten(value, framework, locale):
        return (value[: framework["maxLength"]] if isinstance(value, str) else value), []

    @staticmethod
    def constraint_policy(term, splits, phrasing):
        def validate(value, framework, locale):
            border = framework[term]
            if be_numeral(value) and be_numeral(border) and splits(value, border):
                return value, [f"{locale}: {phrasing} {border}"]
            return value, []
        return validate

    @staticmethod
    def policy_scant(value, framework, locale):
        floor = framework["minItems"]
        if not isinstance(value, list) or len(value) >= floor:
            return value, []
        return value, [f"{locale}: requires {floor} items or more"]

    @staticmethod
    def policy_numerous(value, framework, locale):
        ceiling = framework["maxItems"]
        if not isinstance(value, list) or len(value) <= ceiling:
            return value, []
        return value, [f"{locale}: takes {ceiling} items at most"]

    @classmethod
    def policy_articles(cls, value, framework, locale):
        member_framework = framework["items"]
        if not isinstance(value, list) or not isinstance(member_framework, dict):
            return value, []
        mended, defects = [], []
        for index, member in enumerate(value):
            immutable, lifted = cls.actuate(member, member_framework, f"{locale}[{index}]")
            mended.append(immutable)
            defects += lifted
        return mended, defects

    @classmethod
    def policy_columns(cls, value, framework, locale):
        if not isinstance(value, dict):
            return value, []
        announced = framework.get("properties") or {}
        defects = [f"{locale}.{field}: missing but required" for field in framework.get("required") or [] if field not in value]
        if framework.get("additionalProperties") is False:
            defects += [f"{locale}.{field}: is not an accepted argument" for field in value if field not in announced]
        mended = {}
        for field, member in value.items():
            if field in announced:
                member, lifted = cls.actuate(member, announced[field], f"{locale}.{field}")
                defects += lifted
            mended[field] = member
        return mended, defects

    @classmethod
    def policy_thing(cls, value, framework, locale):
        if not isinstance(value, dict):
            return value, []
        return cls.policy_columns(value, framework, locale)


OperandValidate.RULES = [
    ("enum", OperandValidate.policy_optionset),
    ("const", OperandValidate.policy_invar),
    ("minLength", OperandValidate.policy_concise),
    ("maxLength", OperandValidate.policy_shorten),
    ("minimum", OperandValidate.constraint_policy("minimum", lambda obtained, border: obtained < border, "is below the minimum")),
    ("maximum", OperandValidate.constraint_policy("maximum", lambda obtained, border: obtained > border, "is above the maximum")),
    ("exclusiveMinimum", OperandValidate.constraint_policy("exclusiveMinimum", lambda obtained, border: obtained <= border, "has to be greater than")),
    ("exclusiveMaximum", OperandValidate.constraint_policy("exclusiveMaximum", lambda obtained, border: obtained >= border, "has to be less than")),
    ("minItems", OperandValidate.policy_scant),
    ("maxItems", OperandValidate.policy_numerous),
    ("items", OperandValidate.policy_articles),
    ("type", OperandValidate.policy_thing),
]


class Channel:
    """Every outgoing request (model inference and store calls): pacing, retries, fallbacks, spend and the step counter."""

    def __init__(self, fasten: dict[str, Any]):
        self.s0 = _TALLY.now()
        self.fasten = dict(fasten)
        self.oracle_lineup = ProxyClient(timeout=70, max_retries=1, retry_delay=1)
        self.realm = ProxyClient(timeout=95, max_retries=1, retry_delay=1)
        # store progress
        self.beat = 0
        self.done = False
        self.snapped = False
        self.diary: list[dict[str, Any]] = []
        # pacing
        self.ultimate_deliver = 0.0
        self.gradual_upto = 0.0
        self.oracle_carryon = 0.0
        # model health and spend
        self.disburse = 0.0
        self.factor = 1.0
        self.ultimate_rejoinder = self.s0
        self.misses = 0
        self.nil_oracle = False
        self.prohibited: set[str] = set()
        self.utility_prohibited: set[str] = set()
        self.extra_oracles: list[str] = []
        self.trigger_oracles: set[str] = set()
        self.muffle = False
        self.trigger_falter = ""

    def extra_alt(self) -> float:
        utilized = _TALLY.now() - self.s0
        return EXPIRY - utilized

    def disbursed_outward(self) -> bool:
        return self.disburse >= DISBURSE_TOP * self.factor

    def deliver(self, patron: Any, pathway: str, body: dict[str, Any]) -> Any:
        now = _TALLY.now()
        shortfall = GRADUAL_SHORTFALL if now < self.gradual_upto else SPEED_SHORTFALL
        hold = self.ultimate_deliver + shortfall - now
        if "/inference/" in pathway:
            hold = max(hold, self.oracle_carryon - now)
        hold = min(hold, self.extra_alt() - 1)
        if hold > 0:
            _TALLY.breather(hold)
        self.ultimate_deliver = _TALLY.now()
        with _TALLY.outstanding():
            reply = patron.post_verbose(pathway, body)
        limited = reply.data is None and (reply.error or {}).get("status") in (429, 503)
        if limited:
            self.gradual_upto = _TALLY.now() + GRADUAL_TOWARDS
        return reply

    def breather(self, seed: str, tries: int, status: Any) -> float:
        # deterministic jitter between 0.75x and 1.25x of an exponential backoff capped at 8 s
        mingle = (len(seed) * 7 + sum(ord(hx) for hx in seed[-64:]) + 31 * tries + 13 * self.beat) % 101 / 100.0
        footing = 2.0 if status == 429 else 1.0
        return min(8.0, footing * 2 ** tries) * (0.75 + 0.5 * mingle)

    def downward(self, at: float | None = None) -> bool:
        if self.nil_oracle:
            return True
        if self.misses <= 0:
            return False
        once = _TALLY.now() if at is None else at
        return once - self.ultimate_rejoinder > DOWNWARD_AFTERWARD

    def workable(self, oracle_lineup: list[str], tools: bool = False) -> bool:
        if self.nil_oracle or self.muffle:
            return False
        contenders = list(oracle_lineup) + self.extra_oracles
        return any(name not in self.prohibited and not (tools and name in self.utility_prohibited) for name in contenders)

    @staticmethod
    def bare_alt(txts: list[dict[str, Any]]) -> list[dict[str, Any]]:
        def even(message: dict[str, Any]) -> dict[str, Any]:
            clone = dict(message)
            content = clone.get("content")
            if isinstance(content, list):
                clone["content"] = "\n\n".join(str(segment.get("text") or "") for segment in content if isinstance(segment, dict))
            return clone
        return [even(message) for message in txts]

    def solitary_oracle(self, ladder: list[str]) -> bool:
        workable = {name for name in ladder if name not in self.prohibited}
        return len(workable) <= 1

    def ode_overly_costly(self, ofs: int, oracle_lineup: list[str], txts: list[dict[str, Any]], morpheme_top: int) -> bool:
        if self.disbursed_outward():
            return True
        possesses_thriftier_ensuing = ofs + 1 < len(oracle_lineup) and "sonnet" not in oracle_lineup[ofs + 1]
        if not possesses_thriftier_ensuing:
            return False
        approx = sum(len(json.dumps(mz, default=str)) for mz in txts) / 2.3 * 2e-6 + min(morpheme_top, 1500) * 1e-5
        return self.disburse >= LENIENT_DISBURSE * self.factor or self.disburse + approx > DISBURSE_TOP * self.factor

    def hunttext_oracle(self, txts: list[dict[str, Any]], oracle_lineup: list[str], morpheme_top: int,
            utilitykit: list[dict[str, Any]] | None = None) -> dict[str, Any] | None:
        self.trigger_falter = ""
        self.trigger_oracles = set()
        if self.nil_oracle or self.muffle:
            return None
        # NB: drop_model may extend this very list while it is being walked
        ladder = list(oracle_lineup) + [name for name in self.extra_oracles if name not in oracle_lineup]
        tryout = {"seed": "", "kinds": set(), "refused": set(), "ladder": ladder, "lone": self.solitary_oracle(ladder)}
        for ofs, llm in enumerate(ladder):
            if self.nil_oracle:
                break
            omitted = llm in self.prohibited or llm in tryout["refused"] or (utilitykit and llm in self.utility_prohibited)
            if omitted or ("sonnet" in llm and self.ode_overly_costly(ofs, oracle_lineup, txts, morpheme_top)):
                continue
            concluded, upshot = self.attempt_oracle(llm, txts, oracle_lineup, morpheme_top, utilitykit, tryout)
            if concluded:
                return upshot
        return self.faltered(tryout["kinds"])

    def grab_rejoinder(self, data: dict[str, Any], llm: str, kinds: set[str]) -> dict[str, Any] | None:
        self.ultimate_rejoinder, self.misses = _TALLY.now(), 0
        cost = (data.get("usage") or {}).get("cost")
        if be_numeral(cost):
            self.disburse += float(cost)
        message = data["choices"][0].get("message") or {}
        if not (message.get("content") or message.get("tool_calls")):
            kinds.add("empty")
            self.diary.append({"kind": "inference_failure", "model": llm, "error": "empty"})
            return None
        return message

    def jettison_oracle(self, llm: str, status: Any, crude: str, utilitykit: Any, oracle_lineup: list[str], tryout: dict[str, Any]) -> None:
        text = crude.lower()
        tryout["kinds"].add("hard")
        tryout["refused"].add(llm)
        if status == 404 and utilitykit and "tool" in text:
            self.utility_prohibited.add(llm)
        elif status != 403 or not re.search(r"moderat|flagged", text):
            self.prohibited.add(llm)
        if status != 403 or self.extra_oracles:
            return
        # a 403 may list the models this key may use: keep two of them as a fallback
        permitted_alt = re.search(r'allowed_models"?\s*:\s*\[([^\]]*)', crude)
        if not permitted_alt:
            return
        cited = re.findall(r'"([^"\s]{3,100})"', permitted_alt.group(1))
        self.extra_oracles = [name for name in cited if name not in self.prohibited and name not in oracle_lineup][:2]
        tryout["ladder"] += [name for name in self.extra_oracles if name not in tryout["ladder"]]
        tryout["lone"] = self.solitary_oracle(tryout["ladder"])

    def attempt_oracle(self, llm: str, txts: list[dict[str, Any]], oracle_lineup: list[str], morpheme_top: int, utilitykit: Any,
                  tryout: dict[str, Any]) -> tuple[bool, Any]:
        """One model's attempts. Returns (True, result) when the whole query is settled, (False, None) to move on."""
        kinds = tryout["kinds"]
        unnest = False

        def provide_aloft_cond_stagnant() -> bool:
            return tryout["lone"] or len(self.trigger_oracles) >= 2

        for _ in range(1 if self.downward() else ORACLE_TRIALS):
            extra_alt = self.extra_alt()
            if extra_alt < ULTIMATE_TRIGGER - 10:
                return True, self.faltered(kinds)
            self.oracle_lineup.timeout = int(max(8, min(30 if self.downward() else 75, extra_alt - ULTIMATE_TRIGGER + 15)))
            cargo: dict[str, Any] = {"model": llm, "messages": self.bare_alt(txts) if unnest else self.buffered(txts, llm),
                                       "temperature": 0, "max_tokens": morpheme_top}
            if utilitykit:
                cargo["tools"] = utilitykit
                cargo["tool_choice"] = "auto"
            self.trigger_oracles.add(llm)
            delivered_upon = _TALLY.now()
            answer = self.deliver(self.oracle_lineup, "/inference/chat/completions", cargo)
            data = answer.data
            if isinstance(data, dict) and data.get("choices"):
                message = self.grab_rejoinder(data, llm, kinds)
                return (False, None) if message is None else (True, message)
            self.misses += 1
            self.diary.append({"kind": "inference_failure", "model": llm,
                             "error": str(answer.error)[:300] if answer.error else "empty"})
            error = answer.error or {}
            status, crude = error.get("status"), str(error.get("body") or "")
            outward_relating_balance = status == 402 or (status == 403 and re.search(r"key limit|insufficient|credit", crude.lower()))
            if outward_relating_balance:
                self.nil_oracle = True
                return True, self.faltered(kinds)
            if status in (401, 403, 404):
                self.jettison_oracle(llm, status, crude, utilitykit, oracle_lineup, tryout)
                return False, None
            if status in (400, 413, 422):
                # the request itself was rejected: retry once with plain-text messages, then give up on this model
                kinds.add("hard")
                if not unnest:
                    unnest = True
                    continue
                tryout["refused"].add(llm)
                return False, None
            if isinstance(status, int) and 400 <= status < 500 and status not in (408, 425, 429):
                kinds.add("hard")
                tryout["refused"].add(llm)
                return False, None
            kinds.add("soft")
            if _TALLY.now() - delivered_upon > 20:
                if self.downward() and provide_aloft_cond_stagnant():
                    return True, self.faltered(kinds)
                return False, None
            if not tryout["seed"]:
                tryout["seed"] = json.dumps(txts[-1:], default=str)[-600:]
            wait = self.breather(tryout["seed"] + llm, min(self.misses - 1, 6), status)
            if self.extra_alt() - wait < ULTIMATE_TRIGGER - 10:
                return True, self.faltered(kinds)
            self.oracle_carryon = _TALLY.now() + wait
            if self.downward(self.oracle_carryon) and provide_aloft_cond_stagnant():
                return True, self.faltered(kinds)
        return False, None

    def faltered(self, kinds: set[str]) -> None:
        if "empty" in kinds:
            reason = "empty"
        elif kinds == {"hard"}:
            reason = "refused"
        else:
            reason = ""
        self.trigger_falter = reason
        return None

    @staticmethod
    def buffered(txts: list[dict[str, Any]], llm: str) -> list[dict[str, Any]]:
        # mark the first and the last message as prompt-cache breakpoints for Anthropic models
        if len(txts) <= 2 or not llm.startswith("anthropic/"):
            return txts
        denoted = [dict(message) for message in txts]
        for locus in (0, len(denoted) - 1):
            text = denoted[locus].get("content")
            if isinstance(text, str) and text:
                denoted[locus]["content"] = [{"type": "text", "text": text, "cache_control": {"type": "ephemeral"}}]
        return denoted

    def canvass(self, solicit_prose: str, morpheme_top: int, oracle_lineup: list[str]) -> dict[str, Any] | None:
        reply = self.hunttext_oracle([{"role": "user", "content": solicit_prose}], oracle_lineup, morpheme_top) or {}
        return blob_thing(reply.get("content") or "")

    def arrived(self, data: dict[str, Any], count: int) -> dict[str, Any]:
        relayed = data.get("step")
        self.beat = relayed if isinstance(relayed, int) else self.beat + count
        if data.get("finished"):
            self.done = True
        return data

    def phase_collision(self, crude: str, envt_corpus: dict[str, Any], reused: int) -> int | None:
        """React to a 409 step conflict; returns the new reuse count, or None when the call should stop."""
        if "finished" in crude:
            self.done = True
            return None
        anticipated = re.search(r"expected (\d+)", crude)
        if anticipated:
            self.beat = int(anticipated.group(1)) - 1
            envt_corpus["step"] = self.beat + 1
            return reused
        if "reused" not in crude or reused >= 2:
            return None
        envt_corpus["step"] += 1
        return reused + 1

    def operate(self, deeds: list[dict[str, Any]]) -> dict[str, Any] | None:
        envt_corpus: dict[str, Any] = dict(self.fasten)
        envt_corpus["step"] = self.beat + 1
        if len(deeds) == 1:
            envt_corpus["action"] = deeds[0]
        else:
            envt_corpus["actions"] = deeds
        tries = reused = 0
        for _ in range(REALM_TRIALS + 3):
            extra_alt = self.extra_alt()
            if extra_alt < 2:
                return None
            self.realm.timeout = int(max(3, min(95, extra_alt - 1)))
            answer = self.deliver(self.realm, "/environment/call", envt_corpus)
            if isinstance(answer.data, dict):
                return self.arrived(answer.data, len(deeds))
            error = answer.error or {}
            routine, crude = error.get("status"), str(error.get("body") or "")
            self.diary.append({"kind": "env_failure", "status": routine, "body": crude[:300]})
            if routine == 402:
                self.snapped = True
                return None
            if routine == 409:
                reused = self.phase_collision(crude, envt_corpus, reused)
                if reused is None:
                    return None
                continue
            futile = routine in (403, 422) or (routine in (400, 401, 404) and tries >= 1)
            if futile or tries >= REALM_TRIALS - 1:
                return None
            wait = self.breather(json.dumps(envt_corpus, default=str)[-600:], tries, routine)
            tries += 1
            if self.extra_alt() - wait < 2:
                return None
            _TALLY.breather(wait)
        return None


class Transcripts:
    """Everything the store has returned, with the listing ids, the ids opened in full and their category labels."""

    def __init__(self):
        self.items: list[Any] = []
        self.ids: set[str] = set()
        self.read: set[str] = set()
        self.caption_relating: dict[str, str] = {}

    @staticmethod
    def personal_caption(z: dict[str, Any]) -> str:
        for key, vlu in z.items():
            if not isinstance(vlu, str):
                continue
            text = vlu.strip()
            if not 0 < len(text) <= 60:
                continue
            minor = str(key).lower()
            if "id" not in minor and re.search(r"categor|shelf|department|aisle", minor):
                return text
        return ""

    @staticmethod
    def personal_ident(z: dict[str, Any]) -> str | None:
        for key, vlu in z.items():
            if isinstance(vlu, str):
                minor = str(key).lower()
                if minor in ("id", "_id") or (minor.endswith("id") and re.fullmatch(IDENT_AKIN, vlu)):
                    return vlu
        return None

    @staticmethod
    def cost_column(z: dict[str, Any]) -> float | None:
        return next((float(vlu) for key, vlu in z.items() if be_numeral(vlu) and re.search(r"price|cost", str(key).lower())), None)

    def browse_ident(self, z: dict[str, Any]) -> str | None:
        for key, vlu in z.items():
            if isinstance(vlu, str) and "id" in str(key).lower() and vlu in self.read:
                return vlu
        return None

    def comment_transcript(self, z: dict[str, Any]) -> None:
        self.ids.update(vlu for key, vlu in z.items()
                        if isinstance(vlu, str) and "id" in str(key).lower() and re.fullmatch(IDENT_AKIN, vlu))
        lbl = self.personal_ident(z)
        if not lbl:
            return

        def thoroughgoing(internal: Any) -> bool:
            return isinstance(internal, dict) and sum(1 for vz in internal.values() if isinstance(vz, (str, int, float))) >= 2

        # a nested map with real facts means this record is a full (opened) listing
        if any(thoroughgoing(internal) for internal in z.values()):
            self.read.add(lbl)
        label = self.personal_caption(z)
        if label:
            self.caption_relating.setdefault(lbl, label)

    def add(self, obv: Any) -> None:
        self.items.append(obv)
        route_hashmaps(obv, self.comment_transcript)

    def price(self, entity: str) -> float | None:
        newest: list[float] = []

        def peek(z: dict[str, Any]) -> None:
            designations_itr = any(isinstance(vlu, str) and vlu == entity and "id" in str(key).lower() for key, vlu in z.items())
            if not designations_itr:
                return
            statistic = self.cost_column(z)
            if statistic is not None:
                newest.append(statistic)

        for transcript in self.items:
            route_hashmaps(transcript, peek)
        return newest[-1] if newest else None

    def collect_regarding(self, shard: Any, entity: str, discovered: list[str], depth: int) -> None:
        if depth > 6 or len(discovered) >= 8:
            return
        if isinstance(shard, list):
            kids = shard
        elif isinstance(shard, dict):
            if any(isinstance(vlu, str) and vlu == entity for vlu in shard.values()):
                text = qua_prose(cropped(shard, 400))[:2500]
                if text not in discovered:
                    discovered.append(text)
                return
            kids = list(shard.values())
        else:
            return
        for kid in kids:
            self.collect_regarding(kid, entity, discovered, depth + 1)

    def regarding(self, entity: str, hold: int = 4) -> list[str]:
        discovered: list[str] = []
        for transcript in self.items:
            self.collect_regarding(transcript, entity, discovered, 0)
        return discovered[-hold:]

    def blend_tiles(self) -> tuple[dict[str, dict[str, Any]], list[str]]:
        """One merged card per known listing id (rows and detail records folded together), in first-seen order."""
        blended: dict[str, dict[str, Any]] = {}
        order: list[str] = []

        def collapse(z: dict[str, Any]) -> None:
            lbl = None
            for key, vlu in z.items():
                if isinstance(vlu, str) and "id" in str(key).lower() and re.fullmatch(IDENT_AKIN, vlu) and vlu in self.ids:
                    lbl = vlu
                    break
            if not lbl:
                return
            possesses_atlas = any(isinstance(vlu, dict) and len(vlu) >= 2 for vlu in z.values())
            if not possesses_atlas and len(z) < 4:
                return
            tile = blended.setdefault(lbl, {})
            if lbl not in order:
                order.append(lbl)
            tile.update(z)
            if possesses_atlas:
                self.read.add(lbl)

        for transcript in self.items:
            route_hashmaps(transcript, collapse)
        return blended, order

    def tiles(self, glyph_top: int = 30000, thriftier_versus: str = "", omit: set[str] | None = None, first: Any = ()) -> list[str]:
        blended, order = self.blend_tiles()

        def cost(lbl: str) -> float | None:
            return self.cost_column(blended.get(lbl) or {})

        ceiling = cost(thriftier_versus) if thriftier_versus else None
        if omit:
            order = [lbl for lbl in order if lbl == thriftier_versus or lbl not in omit]
        if ceiling is not None:
            order = [lbl for lbl in order if lbl == thriftier_versus or cost(lbl) is None or cost(lbl) < ceiling]
        principal = set(first or ())
        placement = {lbl: locus for locus, lbl in enumerate(order)}

        def vended_outward(lbl: str) -> bool:
            return any(Spectator.selector_supply(key, vlu) is False for key, vlu in blended[lbl].items())

        ranked = sorted(order, key=lambda lbl: (lbl not in principal, lbl not in self.read, placement[lbl]))
        proses = {lbl: ("DETAILS " if lbl in self.read else "ROW ") + qua_prose(cropped(blended[lbl], 160)) for lbl in ranked}
        if sum(len(text) for text in proses.values()) > glyph_top:
            # too much to show: listings known to be out of stock give way first
            ranked = sorted(order, key=lambda lbl: (lbl not in principal, lbl not in self.read, vended_outward(lbl), placement[lbl]))
        exhibited, utilized = [], 0
        for lbl in ranked:
            text = proses[lbl]
            if utilized + len(text) > glyph_top:
                continue
            exhibited.append(text)
            utilized += len(text)
        return exhibited

    def caption_towards(self, z: dict[str, Any]) -> str:
        label = self.personal_caption(z)
        if label:
            return label
        known = self.caption_relating or {}
        return next((known[vlu] for key, vlu in z.items() if isinstance(vlu, str) and vlu in known and "id" in str(key).lower()), "")

    def elaborate_browses(self, shelf: str = "") -> int:
        """How many opened listings (optionally on the given shelf) show a reasonably full set of facts."""
        elaborate: set[str] = set()
        wished_alt = shelf.strip().lower()

        def brimming_sufficient(z: dict[str, Any]) -> bool:
            assocs = [vlu for vlu in z.values() if isinstance(vlu, dict)]
            if assocs:
                return any(len(vlu) >= 3 for vlu in assocs)
            return sum(1 for vlu in z.values() if isinstance(vlu, str) and 0 < len(vlu) <= 70) >= 5

        def referee(z: dict[str, Any]) -> None:
            lbl = self.browse_ident(z)
            if not lbl or (shelf and self.caption_towards(z).strip().lower() != wished_alt):
                return
            if brimming_sufficient(z):
                elaborate.add(lbl)

        for transcript in self.items:
            route_hashmaps(transcript, referee)
        return len(elaborate)

    def aisles_browse(self) -> list[str]:
        censuses: dict[str, int] = {}

        def reckoning(z: dict[str, Any]) -> None:
            if not self.browse_ident(z):
                return
            label = self.caption_towards(z)
            if not label or not any(isinstance(vlu, dict) for vlu in z.values()):
                return
            censuses[label] = censuses.get(label, 0) + 1

        for transcript in self.items:
            route_hashmaps(transcript, reckoning)
        return sorted(censuses, key=lambda label: -censuses[label])

    def panel_skim(self, shard: Any, shelf: str, interior: bool, depth: int, drain) -> None:
        """Feed sink every dict on the shelf: a dict on the shelf and everything nested inside it."""
        if depth > 6:
            return
        if isinstance(shard, list):
            for kid in shard:
                self.panel_skim(kid, shelf, interior, depth + 1, drain)
            return
        if not isinstance(shard, dict):
            return
        interior = interior or not shelf or self.onto_aisle(shard, shelf)
        if interior:
            drain(shard)
        for kid in shard.values():
            if isinstance(kid, (dict, list)):
                self.panel_skim(kid, shelf, interior, depth + 1, drain)

    def onto_aisle(self, z: dict[str, Any], shelf: str) -> bool:
        target = shelf.lower()
        for key, vlu in z.items():
            if isinstance(vlu, str) and vlu.strip().lower() == target and re.search(r"categor|shelf|department|aisle", str(key).lower()):
                return True
        return self.caption_towards(z).strip().lower() == shelf.strip().lower()

    def datum_collections(self, shelf: str = "") -> list[dict[str, str]]:
        amassed: list[dict[str, str]] = []

        def take(z: dict[str, Any]) -> None:
            facts = {str(key): vlu for key, vlu in z.items() if isinstance(vlu, str) and 0 < len(vlu) <= 70 and not trash_selector(key)}
            if len(facts) >= 5 and facts not in amassed:
                amassed.append(facts)

        for transcript in self.items:
            self.panel_skim(transcript, shelf, False, 0, take)
        return amassed

    def feat_panel(self, shelf: str = "") -> tuple[str, int]:
        """A listing of the property names seen in the records, with how many listings carry each and sample values."""
        overalls: dict[str, int] = {}
        datarow_overalls: dict[str, int] = {}
        samples: dict[str, list[str]] = {}
        noticed_collections = 0
        charted_collections = 0

        def take(z: dict[str, Any]) -> None:
            nonlocal noticed_collections, charted_collections
            concise = [(key, vlu) for key, vlu in z.items() if isinstance(vlu, str) and 0 < len(vlu) <= 70 and not trash_selector(key)]
            scant = len(concise) < 5
            # a flat search row (no nested map) still counts on a named shelf when it has two short facts
            if scant and not (shelf and len(concise) >= 2 and not any(isinstance(vz, dict) for vz in z.values())):
                return
            noticed_collections += 1
            datarow_akin = scant and any(isinstance(vlu, str) and "id" in str(key).lower() for key, vlu in z.items())
            if not datarow_akin:
                charted_collections += 1
            for key, vlu in concise:
                if re.search(r"\d", vlu) and re.fullmatch(ROUTINE_AKIN, vlu) and ("-" in vlu or "_" in vlu):
                    continue
                name = re.sub(r"[_.]+", " ", str(key)).strip().lower()
                overalls[name] = overalls.get(name, 0) + 1
                if datarow_akin:
                    datarow_overalls[name] = datarow_overalls.get(name, 0) + 1
                tub = samples.setdefault(name, [])
                reduced = vlu.lower()
                if len(tub) < 5 and reduced not in tub:
                    tub.append(reduced)

        for transcript in self.items:
            self.panel_skim(transcript, shelf, False, 0, take)
        if charted_collections:
            # once full records are known, names that appear only in search rows are dropped
            for name in [name for name in overalls if datarow_overalls.get(name, 0) >= overalls[name]]:
                del overalls[name]
        best = sorted(overalls, key=lambda name: -overalls[name])[:28]
        lines = [f"- {name} (in {overalls[name]} listings): " + " | ".join(samples.get(name, [])) for name in best]
        return "\n".join(lines), noticed_collections

    def caption_orderlines(self) -> str:
        """Short repeated label values per key, e.g. '- colour: red (x3) | blue (x2)', at most twelve keys."""
        seen: dict[str, dict[str, int]] = {}

        def count(z: dict[str, Any]) -> None:
            if len(z) < 4:
                return
            for key, vlu in z.items():
                if not isinstance(vlu, str) or not 0 < len(vlu) <= 40:
                    continue
                if re.search(r"\d", vlu) and re.fullmatch(ROUTINE_AKIN, vlu):
                    continue
                apiece_measurement = seen.setdefault(str(key), {})
                apiece_measurement[vlu] = apiece_measurement.get(vlu, 0) + 1

        for transcript in self.items:
            route_hashmaps(transcript, count)
        lines = []
        for key, apiece_measurement in seen.items():
            if sum(apiece_measurement.values()) < 4 or not 2 <= len(apiece_measurement) <= 8 or max(apiece_measurement.values()) < 2:
                continue
            sequenced = sorted(apiece_measurement.items(), key=lambda duo: -duo[1])
            lines.append(f"- {key}: " + " | ".join(f"{vlu} (x{instances})" for vlu, instances in sequenced))
        return "\n".join(lines[:12])

class Necessities:
    def __init__(self, channel: Channel, request: str):
        self.channel = channel
        self.request = request
        # what the request reading said, and the requirement lists that grow out of the conversation
        self.browsing: dict[str, Any] = {}
        self.written: list[str] = []
        self.uncovered: list[str] = []
        self.indifferent: list[str] = []
        self.let_go: list[str] = []
        self.presently_dems: list[str] | None = None
        self.resolved: list[str] = []
        self.crisp_necessities: list[str] | None = None
        self.disburse_top: Any = None
        self.edition = 0
        # conversation bookkeeping
        self.snagged: set[str] = set()
        self.interchanges: list[str] = []
        self.rejoinder_expected = ""
        self.aux_ultimate: str | None = None
        self.pending: list[Any] | None = None
        self.awaiting_upon = 0.0
        self.pegged_perceived = 0
        self.cite_tidings: list[dict[str, Any]] = []
        # when (in versions and steps) a drop last widened the acceptable set
        self.expanded_upon = -1
        self.expanded_beat = -10 ** 6
        self.combable = False
        # hooks the run plugs in later
        self.column_callback = None
        self.cite_callback = None

    @staticmethod
    def entire_census(value: Any) -> bool:
        """True for a real positive integer (a bool does not count)."""
        if isinstance(value, bool) or not isinstance(value, int):
            return False
        return value >= 1

    def items(self) -> list[dict[str, Any]]:
        """The separate products of a multi-product request; empty when only one product is wanted."""
        def qua_orderline(element_alt: Any) -> dict[str, Any] | None:
            if not isinstance(element_alt, dict):
                return None
            label = str(element_alt.get("product") or "").strip()
            if not label:
                return None
            wished_alt = element_alt.get("quantity")
            dems = element_alt.get("requirements")
            if not isinstance(dems, list):
                dems = []
            return {"product": label,
                    "quantity": wished_alt if self.entire_census(wished_alt) else 1,
                    "requirements": [str(rq) for rq in dems if rq and ":" in str(rq)]}

        lines = [line for line in map(qua_orderline, self.browsing.get("items") or []) if line is not None]
        return lines if len(lines) >= 2 else []

    def quantity(self) -> int:
        if self.items():
            return 1
        wished_alt = self.browsing.get("quantity")
        return wished_alt if self.entire_census(wished_alt) else 1

    def censuses(self) -> list[int]:
        return [line["quantity"] for line in self.items()] or [self.quantity()]

    def articles_prose(self) -> str:
        lines = self.items()
        if not lines:
            return ""
        muster = " | ".join(f"{n}) {line['product']} x{line['quantity']}" for n, line in enumerate(lines, 1))
        return ("SEVERAL PRODUCTS ARE REQUESTED; the order holds a line for each product, at that product's quantity, and "
                "nothing else; each listing is judged solely by the requirements of its own product (its category tells "
                "which), and the cheapest or best rule applies within each product separately: " + muster)

    def priority(self) -> str:
        mode = self.browsing.get("priority")
        if mode in ("cheapest", "best"):
            return mode
        return "none"

    def feats_residual(self) -> list[str]:
        """Requirements still in force that concern a property (price, budget, quantity and stock are left out)."""
        if self.presently_dems is not None:
            footing = self.presently_dems
        else:
            footing = [rq for rq in self.written + self.uncovered if rq not in self.let_go]
        non_characteristic = r"\bprice\b|budget|quantity|\bstock\b|\bcents?\b"
        return [rq for rq in footing if rq not in self.let_go and not re.search(non_characteristic, rq.lower())]

    def prose_corpus(self) -> str:
        # NOTE: this exact wording feeds the grounding test in hear(), so it stays word for word.
        if self.presently_dems is not None:
            itemized = list(self.presently_dems)
        else:
            itemized = [f"{rq} (stated)" for rq in self.written if rq not in self.let_go]\
                + [f"{rq} (revealed by the shopper)" for rq in self.uncovered if rq not in self.let_go]
        if itemized:
            body = "; ".join(itemized)
        elif not self.let_go:
            body = "see the request"
        else:
            body = "no property requirement remains: every in-stock listing in the requested category qualifies"
        if self.disburse_top is not None:
            body += f"; total price at most {self.disburse_top} {self.browsing.get('currency') or ''}".rstrip()
        if self.items():
            body += "; " + self.articles_prose()
        else:
            body += f"; quantity {self.quantity()}"
        if self.let_go:
            body += "; no longer required: " + "; ".join(self.let_go)
        if self.indifferent:
            body += "; shopper is indifferent to: " + ", ".join(self.indifferent)
        return body

    def modifications(self) -> str:
        if not self.let_go:
            return ""
        return ("\n\nCHANGES SINCE THE REQUEST (they override its wording): the shopper has DROPPED " + "; ".join(self.let_go)
                + ". Consider it gone: no listing fails, ranks higher or is favoured on its account.")

    def outline(self, phase_census: int, txts: int | None) -> str:
        obtained = self.browsing
        if not obtained:
            return ""

        def several(preface: str, key: str) -> list[str]:
            return [preface + str(element_alt) for element_alt in obtained.get(key) or []]

        out = ["First reading of the request (check it against the request text itself):"]
        if obtained.get("product"):
            out.append(f"- product: {obtained.get('product')}")
        if self.items():
            out += [f"- product {n}: {line['product']}, quantity {line['quantity']} (one cart line)"
                    for n, line in enumerate(self.items(), 1)]
            out.append("- these lines alone make up the order, one per product; whenever the cart tool takes a list of lines, "
                       "each cart call sends the whole set")
        else:
            out.append(f"- quantity: {self.quantity()}")
        out += several("- stated requirement: ", "requirements")
        out += several("- value given by pointing at another listing (open that listing first, then search with its value): ",
                       "references")
        if obtained.get("price_cap") is not None:
            out.append(f"- price cap (total): {obtained.get('price_cap')} {obtained.get('currency') or ''}".rstrip())
        if obtained.get("in_stock_only"):
            out.append("- in-stock (on hand) items only")
        if self.priority() != "none":
            nuance = f" ({obtained.get('priority_detail')})" if obtained.get("priority_detail") else ""
            out.append(f"- out of every item that fits, take the {self.priority()}{nuance}")
        else:
            out.append("- the request names no cheapest/best rule: any item meeting every requirement within the budget is "
                       "correct, so spend no steps hunting for the lowest price (unless the request itself asks for it)")
        out += several("- a wish, NOT a requirement (it only settles ties): ", "soft_wants")
        out += several("- to find out from the shopper before choosing: ", "ask_first")
        out += several("- rule: ", "rules")
        if txts is not None:
            out.append(f"- shopper messages permitted: {txts}")
        out.append(f"- steps allowed: {phase_census}")
        return "\n".join(out)

    @staticmethod
    def orderline_designations(needs: list[Any], txl: list[str]) -> dict[str, str]:
        """Re-spell a need's property with the words the shopper used when they only share its stems."""
        uttered = re.findall(r"[a-z]+", " ".join(txl).lower())
        uttered_prose = " ".join(uttered)
        out: dict[str, str] = {}
        for element_alt in needs:
            if not isinstance(element_alt, str) or ":" not in element_alt:
                continue
            feat_component, _, remainder = element_alt.partition(":")
            postfix = feat_component[feat_component.find("("):].strip() if "(" in feat_component else ""
            trunks = re.findall(r"[a-z]+", re.split(r"\s*\(", feat_component)[0].lower())
            if not trunks or min(map(len, trunks)) < 3:
                continue
            if re.search(r"\b" + r"\s+".join(trunks) + r"\b", uttered_prose):
                continue  # the shopper already used the very words
            extent = len(trunks)
            hit = next((uttered[ii: ii + extent] for ii in range(len(uttered) - extent + 1)
                        if uttered[ii: ii + extent] != trunks
                        and all(wq.startswith(rootnode) for rootnode, wq in zip(trunks, uttered[ii: ii + extent]))), None)
            if hit is not None:
                out[element_alt] = " ".join(hit) + (" " + postfix if postfix else "") + ":" + remainder
        return out

    @staticmethod
    def badge_lineup(text: str) -> list[str]:
        return re.findall(r"\(([^)]*)\)", text)

    def jettison_designated(self, aged_necessity: str, txl: list[str]) -> bool:
        try:
            return self.jettison_permitted(aged_necessity, txl)
        except Exception:
            return True

    def jettison_permitted(self, aged_necessity: str, txl: list[str]) -> bool:
        """Accept a drop only when the shopper's words actually name the property being let go."""
        name = str(aged_necessity).split(":", 1)[0]
        tag = self.badge_lineup(name)

        def equal_niche(further: str) -> bool:
            further_front = further.split(":", 1)[0]
            return equal_feat(further_front, name) and (not tag or self.badge_lineup(further_front) in ([], tag))

        hits = [rq for rq in self.written if equal_niche(rq)]
        if not hits:
            newest = [n for n in self.crisp_necessities or [] if isinstance(n, str) and ":" in n]
            hits = [rq for rq in self.uncovered if equal_niche(rq) and rq not in self.written and rq not in newest]
            if not hits:
                return True
            if any(equal_niche(n) and n != aged_necessity for n in newest):
                return True
        words = re.findall(r"[a-z]+", " ".join(txl).lower())
        shingles = set()
        for broadness in (1, 2, 3):
            for at in range(len(words) - broadness + 1):
                shingles.add(" ".join(words[at: at + broadness]))
        shingles = {g for g in shingles if len(g) >= 3 and g not in ANCHOR_VOCABLES}
        for rq in hits:
            if any(equal_feat(rq.split(":", 1)[0], g) for g in shingles):
                return True
        if not self.column_callback:
            return False
        return self.jettisoned_through_column(hits, name, shingles)

    def jettisoned_through_column(self, concordant: list[str], name: str, expressions: set[str]) -> bool:
        """A drop may name the store's field for the property instead of the property itself."""
        def further_feats() -> list[str]:
            return [rq.split(":", 1)[0] for rq in self.written + self.uncovered
                    if isinstance(rq, str) and ":" in rq and not equal_feat(rq.split(":", 1)[0], name)]

        for req in concordant:
            head, _, crude = req.partition(":")
            value = re.sub(r"\((?:stated|revealed[^)]*)\)", " ", crude)
            if cite_prose(value) or not hx_alternates(value):
                continue
            fields = self.column_callback(head) or {}
            phrase = self.locution_designating_column(expressions, further_feats(), fields, value)
            if phrase is None:
                continue
            self.channel.diary.append({"kind": "drop_by_field", "need": req[:200], "phrase": phrase})
            return True
        return False

    @staticmethod
    def locution_designating_column(expressions: set[str], competitors: list[str], fields: dict[str, Any], value: str) -> str | None:
        for phrase in expressions:
            if any(stringent_feat(further, phrase) for further in competitors):
                continue
            if any(stringent_feat(key, phrase) and measurement_noticed(value, " | ".join(vlus)) for key, vlus in fields.items()):
                return phrase
        return None

    @staticmethod
    def bare_vocables(text: str) -> str:
        return " ".join(re.findall(r"[a-z0-9]+", text))

    def exact_measurements(self, answer: dict[str, Any]) -> None:
        """Where a need's value hides behind comparison wording, use the store's literal field value instead."""
        if not self.column_callback:
            return
        even = self.bare_vocables

        def warehouse_orthography(head: str, naked: str) -> str | None:
            prop = re.split(r"\s*\(", head)[0].strip()
            for key, vlus in (self.column_callback(head) or {}).items():
                if not stringent_feat(key, prop):
                    continue
                same = next((vz for vz in vlus if even(str(vz).lower()) == naked), None)
                if same is not None:
                    return same
            return None

        switches: dict[str, str] = {}
        for need in answer.get("needs") or []:
            if not isinstance(need, str) or ":" not in need:
                continue
            head, _, value = need.partition(":")
            if re.search(r"\d", value) or cite_prose(value):
                continue
            naked = even(re.sub(JX_WEIGH, " ", value.lower()))
            if not naked:
                continue
            if naked == even(value.lower()) and favorable_measurement(value, head):
                continue
            discovered = warehouse_orthography(head, naked)
            if discovered is None:
                continue
            EXACT_MEASUREMENTS.add(naked)
            switches[need] = f"{head.strip()}: {str(discovered).strip().lower()}"
        if not switches:
            return
        self.switch_necessities(answer, switches)
        self.channel.diary.append({"kind": "literal_value", "needs": list(switches.values())[:4]})

    @staticmethod
    def switch_necessities(answer: dict[str, Any], switches: dict[str, str]) -> None:
        def exchanged(sqn: list[Any]) -> list[Any]:
            return [switches.get(xq, xq) if isinstance(xq, str) else xq for xq in sqn]

        answer["needs"] = exchanged(answer.get("needs") or [])
        if isinstance(answer.get("requirements_now"), list):
            answer["requirements_now"] = exchanged(answer["requirements_now"])

    def pegged(self, feat_designation: str, qstr: str | None) -> bool:
        """True when a stated (not story-guessed) requirement already fixes this property."""
        try:
            provided = self.browsing.get("references")
            arrows = [str(rq) for rq in provided or []] if isinstance(provided, list) else []
            arrows += list(self.resolved or [])
            solicited_badge = self.badge_lineup(str(feat_designation))
            solicited_selector = feat_selector(feat_designation)
            for req in self.written:
                if req in self.let_go or ":" not in req:
                    continue
                head, _, value = req.partition(":")
                personal_badge = self.badge_lineup(head)
                if solicited_badge and personal_badge and not variety_vocables(solicited_badge[-1]) & variety_vocables(personal_badge[-1]):
                    continue
                if not favorable_measurement(value, head) or tale_conjecture(req, self.request, arrows):
                    continue
                personal_selector = feat_selector(head)
                if equal_feat(head, feat_designation) or (len(personal_selector) >= 4 and solicited_selector.endswith(personal_selector)):
                    return True
            return False
        except Exception:
            return False

    @staticmethod
    def rated(needs: Any, txl: list[str]) -> bool:
        """Did the shopper's lines spell out both a property and its value from one of these needs?"""
        if not isinstance(needs, list):
            return False
        uttered = " ".join(str(xq) for xq in txl).lower()
        for need in needs:
            head, _, value = str(need).partition(":")
            selector_dabs = [g_alt for g_alt in feat_vocables(head) if len(g_alt) >= 3]
            vlu_dabs = [g_alt for g_alt in re.findall(r"[a-z0-9]+", value.lower()) if g_alt not in ANCHOR_VOCABLES]
            if selector_dabs and vlu_dabs and all(g_alt in uttered for g_alt in selector_dabs + vlu_dabs):
                return True
        return False

    def monitor_directive(self, crisp: list[str], qstr: str | None, thus_distant: str) -> str:
        hunks = [PERCEIVE, "\n\nTHE SHOPPER'S REQUEST (background only):\n", self.request, "\n\nREQUIREMENTS SO FAR:\n", thus_distant]
        if self.interchanges:
            hunks.append("\n\nPAST EXCHANGES WITH THE SHOPPER (earliest first):\n" + "\n".join(self.interchanges[-12:]))
        if qstr:
            hunks.append("\n\nTHE ASSISTANT ASKED:\n" + qstr)
        hunks.append("\n\nTHE SHOPPER NOW SAID:\n" + "\n".join(crisp))
        return "".join(hunks)

    def maintain_unperceived(self, crisp: list[str], qstr: str | None) -> None:
        """No model could read the lines: forget them for now and queue them for a later try."""
        self.snagged.difference_update(crisp)
        self.interchanges.pop()
        self.pending = [*(self.pending or []), (list(crisp), qstr)]
        self.awaiting_upon = _TALLY.now()

    def solely_aux(self, answer: dict[str, Any], crisp: list[str], qstr: str | None) -> dict[str, Any]:
        """A reply that only asks for fewer questions at once carries no requirement: blank it out."""
        try:
            aux = answer.get("meta_only") is True and qstr and not self.rated(answer.get("needs"), crisp)
            if not aux:
                return answer
            void = {"needs": [], "no_preference": [], "dropped": [], "references": [], "requirements_now": None,
                     "new_budget": None, "unavailable": False, "reply": ""}
            answer = {**answer, **void}
            self.aux_ultimate = qstr
            self.channel.diary.append({"kind": "meta_only", "line": " / ".join(crisp)[:200]})
        except Exception:
            pass
        return answer

    def preserve_orderline_designations(self, answer: dict[str, Any], crisp: list[str]) -> None:
        try:
            retitles = self.orderline_designations(answer.get("needs") or [], crisp)
        except Exception:
            return
        if retitles:
            self.switch_necessities(answer, retitles)
            self.channel.diary.append({"kind": "need_renamed", "names": list(retitles.values())[:4]})

    def veiled_orderlines(self, answer: dict[str, Any], crisp: list[str], qstr: str | None) -> list[str]:
        if not self.cite_callback:
            return crisp
        try:
            veiled = self.cite_callback(answer, crisp, qstr)
        except Exception as failure:
            self.channel.diary.append({"kind": "ref_line_error", "error": f"{type(failure).__name__}: {failure}"[:300]})
            return crisp
        return veiled or crisp

    def comment_directed(self, answer: dict[str, Any], crisp: list[str]) -> None:
        """Queue each newly revealed need whose value points at another listing."""
        try:
            provided = answer.get("references")
            infos = [rq for rq in provided or [] if isinstance(rq, dict)] if isinstance(provided, list) else []
            for need in answer.get("needs") or []:
                if not isinstance(need, str) or need not in self.uncovered:
                    continue
                pointvals = cite_prose(need.partition(":")[2]) or need in (answer.get("ref_needs") or [])
                if pointvals:
                    self.cite_tidings = [*(self.cite_tidings or []), {"need": need, "infos": infos, "line": " ".join(crisp)}]
        except Exception:
            pass

    def unbound_feats(self, answer: dict[str, Any], qstr: str | None) -> None:
        """Record the properties the shopper does not mind about, unless a live requirement already fixes them."""
        provenance = self.presently_dems if self.presently_dems is not None else self.written + self.uncovered
        immutable = {rq.split(":", 1)[0].strip().lower() for rq in provenance
                 if isinstance(rq, str) and ":" in rq and rq not in self.let_go}
        for feat_designation in answer.get("no_preference") or []:
            if not isinstance(feat_designation, str) or not feat_designation or feat_designation in self.indifferent:
                continue
            if feat_designation.strip().lower() in immutable:
                continue
            if not self.pegged(feat_designation, qstr):
                self.indifferent.append(feat_designation)
            else:
                self.pegged_perceived += 1

    def grab_jettisons(self, answer: dict[str, Any], veiled: list[str], observations_alt: list[str]) -> bool:
        """Apply the drops the shopper's words support; True when one of them widens a cheapest/best search."""
        expanded = False
        allowance_maneuvered = be_numeral(answer.get("new_budget"))
        cash = r"\b(?:prices?|budget|costs?|spend\w*|cents?|dollars?)\b"
        for aged_necessity in answer.get("dropped") or []:
            if not isinstance(aged_necessity, str) or not aged_necessity or aged_necessity in self.let_go:
                continue
            if allowance_maneuvered and re.search(cash, aged_necessity.lower()):
                continue  # the budget change already covers it
            if self.jettison_designated(aged_necessity, veiled):
                self.let_go.append(aged_necessity)
                observations_alt.append(f"dropped: {aged_necessity}")
                if not expanded:
                    expanded = self.priority() != "none"
            else:
                self.channel.diary.append({"kind": "drop_kept", "need": aged_necessity[:200]})
        return expanded

    def reinstate_mislaid(self, veiled: list[str]) -> None:
        """Put back revealed and stated needs the model's current list forgot (unless they were let go)."""
        for need in self.uncovered:
            key = need.split(":", 1)[0].strip().lower()
            if not key:
                continue
            known = any(key in rq.lower() for rq in self.presently_dems) or any(key in aged.lower() for aged in self.let_go)
            if not known:
                self.presently_dems.append(need)
        for need in self.written:
            try:
                self.reinstate_declared(need, veiled)
            except Exception:
                pass

    def reinstate_declared(self, need: str, veiled: list[str]) -> None:
        if need in self.let_go or need in self.presently_dems or self.jettison_designated(need, veiled):
            return
        name = need.split(":", 1)[0]
        tag = self.badge_lineup(name)

        def equal_niche(further: str) -> bool:
            further_front = further.split(":", 1)[0]
            return equal_feat(name, further_front) and self.badge_lineup(further_front) == tag

        if any(equal_niche(aged) for aged in self.let_go):
            return
        if any(equal_niche(rq) for rq in self.presently_dems if isinstance(rq, str)):
            return
        self.presently_dems.append(need)

    def expanded_observation(self) -> str:
        if self.feats_residual():
            return ("a requirement the shopper let go WIDENS the field: rows that failed on that single point pass now, and rows "
                    "that no search has shown yet may come in cheaper than the current pick; search the kind again without it, "
                    "using short queries of each remaining required value together with the kind word, value first, reading "
                    "every page, and pick the cheapest of everything that is acceptable now")
        opening = ("no property is required any longer: any in-stock row of the requested category passes, whatever "
                   "its sort (accessories, parts and sets on that shelf as well), and the lowest price on the whole shelf "
                   "wins. Rows come in relevance or id order, never by price")
        if self.priority() == "cheapest" and self.combable:
            return (opening + ", so the following shelf slices are fetched on your behalf (the listing pages, the label's "
                    "words, the object words in titles); if no '[shelf sweep done]' list follows, cover the shelf yourself "
                    "with short queries, the first page of each - the category label, every kind word of the label on its "
                    "own, every item name that keeps recurring in the titles seen - and the category listing's pages. Then "
                    "make the lowest-priced in-stock row of the requested category within the budget the cart's only line, "
                    "check its stock and price again, and order it")
        return (opening + ": cover the shelf with short queries, the first page of each - the category label, every kind "
                "word of the label on its own, every item name that keeps recurring in the titles seen - and the category "
                "listing's pages; after that, order the cheapest in-stock offer")

    def perceive(self, txl: list[str], qstr: str | None) -> str:
        """Read new shopper lines through the listener model and fold what they say into the requirements."""
        crisp = [line for line in txl if line not in self.snagged]
        if not crisp:
            return ""
        self.snagged.update(crisp)
        if self.channel.extra_alt() < SEALING:
            return ""
        thus_distant = self.prose_corpus()
        directive = self.monitor_directive(crisp, qstr, thus_distant)
        guide_amidst = f"asked: {qstr} -> " if qstr else ""
        self.interchanges.append(guide_amidst + "shopper: " + " / ".join(crisp))
        answer = self.channel.canvass(directive, 2500, MONITOR)
        if not answer:
            if self.channel.muffle or self.channel.misses > 0:
                self.maintain_unperceived(crisp, qstr)
            return ""
        answer = self.solely_aux(answer, crisp, qstr)
        self.preserve_orderline_designations(answer, crisp)
        veiled = self.veiled_orderlines(answer, crisp, qstr)
        ctxt = " ".join([thus_distant, " ".join(self.interchanges), qstr or ""]).lower()

        def attested(req: str) -> bool:
            # a requirement counts only if one of its longer words was actually part of the talk so far
            lengthier = [wq for wq in re.findall(r"[a-z]+", req.lower()) if len(wq) >= 3]
            return not lengthier or any(wq in ctxt for wq in lengthier)

        try:
            self.exact_measurements(answer)
        except Exception as failure:
            self.channel.diary.append({"kind": "literal_value_error", "error": f"{type(failure).__name__}: {failure}"[:300]})
        uttered_necessities = answer.get("needs") or []
        self.crisp_necessities = [n for n in uttered_necessities if isinstance(n, str)]
        actual = answer.get("requirements_now")
        if isinstance(actual, list) and all(isinstance(rq, str) for rq in actual):
            self.presently_dems = [rq for rq in actual if rq.strip() and attested(rq)]
        observations_alt = []
        for need in uttered_necessities:
            if not isinstance(need, str) or not need or need in self.uncovered or not attested(need):
                continue
            self.uncovered.append(need)
            observations_alt.append(f"new requirement: {need}")
        self.comment_directed(answer, crisp)
        self.unbound_feats(answer, qstr)
        expanded = self.grab_jettisons(answer, veiled, observations_alt)
        budget = answer.get("new_budget")
        if be_numeral(budget) and budget > 0 and budget != self.disburse_top:
            self.disburse_top = budget
            observations_alt.append(f"new budget: {budget}")
        if answer.get("unavailable"):
            observations_alt.append("an item is no longer available; remove it from the cart and do not order it")
        if observations_alt:
            self.edition += 1
            if expanded:
                self.expanded_upon, self.expanded_beat = self.edition, self.channel.beat
        if self.presently_dems is not None:
            self.reinstate_mislaid(veiled)
        if expanded:
            observations_alt.append(self.expanded_observation())
        self.channel.diary.append({"kind": "heard", "question": qstr is not None, "needs": len(answer.get("needs") or []),
                              "free": len(answer.get("no_preference") or []), "dropped": len(answer.get("dropped") or []),
                              "budget": be_numeral(budget)})
        solicited_reverse = qstr is None and not answer.get("no_preference") and any("?" in line for line in crisp)
        self.rejoinder_expected = str(answer.get("reply") or "").strip() if solicited_reverse else ""
        rear = " (" + "; ".join(observations_alt) + ")" if observations_alt else ""
        return "Meaning: " + str(answer.get("meaning") or "") + rear


class BuggyTome:
    def __init__(self, that_utility: str | None, jspec: dict[str, Any]):
        self.that_utility = that_utility
        self.feat_briefs = (jspec or {}).get("properties") or {}
        # how a put call looks, and whether this store adds to a held line or sets it
        self.deposit_mould: str | None = None
        self.appends: bool | None = None
        self.utilized = False
        # the code's own count of the cart, and the add whose effect is still unconfirmed
        self.metrics: dict[str, float] = {}
        self.pending: dict[str, float] = {}
        self.modified_upon = 0
        self.glanced_upon = -1
        self.ultimate_glance: Any = None

    def mould(self, trigger_operands: dict[str, Any]) -> str:
        """The enum-valued arguments of a cart call (the operation it performs), as text."""
        def be_selection(key: str) -> bool:
            brief = self.feat_briefs.get(key)
            return isinstance(brief, dict) and "enum" in brief

        return qua_prose({key: vlu for key, vlu in trigger_operands.items() if be_selection(key)})

    def orderline_brief(self) -> tuple[str, str, str] | None:
        """(array argument, id field, count field) when the cart tool takes a list of lines."""
        def entered(fields: dict[str, Any], *kinds: str) -> list[str]:
            return [name for name, sub in fields.items() if isinstance(sub, dict) and sub.get("type") in kinds]

        for key, brief in self.feat_briefs.items():
            if not isinstance(brief, dict) or brief.get("type") != "array":
                continue
            internal = brief.get("items")
            fields = internal.get("properties") if isinstance(internal, dict) else None
            if not isinstance(fields, dict):
                continue
            proses = [name for name in entered(fields, "string") if "enum" not in fields[name]]
            ids = [name for name in proses if "id" in name.lower()] or proses
            quantities = entered(fields, "integer", "number")
            if ids and quantities:
                return key, ids[0], quantities[0]
        return None

    def itemized_orderlines(self, trigger_operands: dict[str, Any]) -> list[tuple[str, float | None]] | None:
        brief = self.orderline_brief()
        if not brief or not isinstance(trigger_operands.get(brief[0]), list):
            return None
        vector_selector, ident_selector, census_selector = brief
        out = []
        for row in trigger_operands[vector_selector]:
            if not isinstance(row, dict):
                continue
            lbl = row.get(ident_selector)
            if not isinstance(lbl, str) or not lbl.strip():
                continue
            quantity_alt = row.get(census_selector)
            out.append((lbl.strip(), float(row[census_selector]) if be_numeral(quantity_alt) else None))
        return out

    def lines(self, trigger_operands: dict[str, Any]) -> list[tuple[str, float | None]]:
        itemized = self.itemized_orderlines(trigger_operands)
        if itemized is not None:
            return itemized
        entity = self.objective_even(trigger_operands)
        return [(entity, self.census_operand(trigger_operands))] if entity else []

    @staticmethod
    def primary_deposit(discovered: list[tuple[str, float | None]]) -> tuple[str, float | None] | None:
        """The first line that adds something (falling back to the first line at all)."""
        if not discovered:
            return None
        return next((line for line in discovered if line[1] is None or line[1] > 0), discovered[0])

    def aim(self, trigger_operands: dict[str, Any]) -> str | None:
        line = self.primary_deposit(self.lines(trigger_operands))
        if not line:
            return None
        return line[0]

    def objective_even(self, trigger_operands: dict[str, Any]) -> str | None:
        """The first free-text, single-token argument of a flat cart call (taken as the item id)."""
        for key, shard in trigger_operands.items():
            brief = self.feat_briefs.get(key)
            if isinstance(brief, dict) and "enum" in brief:
                continue
            if isinstance(shard, str) and shard.strip() and " " not in shard.strip():
                return shard
        return None

    def census_operand(self, trigger_operands: dict[str, Any]) -> float | None:
        if self.orderline_brief() and isinstance(trigger_operands.get(self.orderline_brief()[0]), list):
            line = self.primary_deposit(self.lines(trigger_operands))
            return line[1] if line else None
        for key, brief in self.feat_briefs.items():
            numerical = isinstance(brief, dict) and "enum" not in brief and brief.get("type") in ("integer", "number")
            if not numerical:
                continue
            shard = trigger_operands.get(key, brief.get("default"))
            if be_numeral(shard):
                return float(shard)
        return None

    def be_deposit(self, trigger_operands: dict[str, Any]) -> bool:
        if self.deposit_mould is None:
            return True
        return self.deposit_mould == self.mould(trigger_operands)

    def maintained(self, entity: str) -> float:
        return self.metrics.get(entity, 0.0)

    def conclude_glance(self, obv: Any, entire: bool = True) -> bool:
        seen = self.glance_metrics(obv)
        if seen is None:
            return False
        if entire or len(seen) != 1:
            self.glean(seen)
            return True
        # a reply to an edit that names a single line tells nothing about the rest (stores answer line by line)
        rest_alt = {entity: amt for entity, amt in self.metrics.items() if entity not in seen}
        self.glean(seen)
        self.metrics.update(rest_alt)
        return True

    def tome_lineup(self, trigger_operands: dict[str, Any], obv: Any) -> str:
        """A list-of-lines cart call replaces the whole cart: count it afresh."""
        self.appends = False
        reckoning: dict[str, float] = {}
        for entity, count in self.lines(trigger_operands):
            if count is None:
                reckoning[entity] = reckoning.get(entity, 0.0) + 1.0
            elif count > 0:
                reckoning[entity] = reckoning.get(entity, 0.0) + count
        self.metrics = reckoning
        self.conclude_glance(obv)
        return self.abstract()

    def reserved_alt(self, trigger_operands: dict[str, Any], obv: Any) -> str:
        """Update the counted cart after a cart call and return the count as a note."""
        self.utilized = True
        if isinstance(obv, dict) and obv.get("error"):
            return ""
        if self.deposit_mould is None:
            self.deposit_mould = self.mould(trigger_operands)
        brief = self.orderline_brief()
        if brief and isinstance(trigger_operands.get(self.orderline_brief()[0]), list):
            return self.tome_lineup(trigger_operands, obv)
        entity = self.aim(trigger_operands)
        if not entity:
            return ""
        count = self.census_operand(trigger_operands)
        if count is None:
            count = 1.0
        preceding = self.maintained(entity)
        if not self.be_deposit(trigger_operands):
            afterward = 0.0 if count <= 0 else max(0.0, preceding - count)
        elif preceding > 0 and self.appends is not False:
            afterward = preceding + count
            self.pending = {entity: count}
        else:
            afterward = 0.0 if count <= 0 else count
        self.metrics[entity] = afterward
        if self.metrics.get(entity, 0) <= 0:
            self.metrics.pop(entity, None)
        if self.conclude_glance(obv, entire=False):
            return self.abstract()
        if (count <= 0 or not self.be_deposit(trigger_operands)) and self.exhibits_vacant(obv, entity):
            self.metrics.pop(entity, None)
            self.pending.pop(entity, None)
        return self.abstract()

    def exhibits_vacant(self, obv: Any, entity: str) -> bool:
        """Does the reply describe an empty cart (empty lists, zero totals, no other held id)?"""
        if not isinstance(obv, dict) or entity not in self.metrics:
            return False
        catalogs = [vlu for vlu in obv.values() if isinstance(vlu, list)]
        if not catalogs or any(catalogs):
            return False
        figures = [vlu for vlu in obv.values() if isinstance(vlu, (int, float)) and not isinstance(vlu, bool)]
        if not figures or any(vlu != 0 for vlu in figures):
            return False
        text = qua_prose(obv)
        return all(further not in text for further in self.metrics if further != entity)

    def glance_metrics(self, obv: Any) -> dict[str, float] | None:
        """Read id -> units from any cart-like rows in a reply; None when nothing usable shows."""
        enumerated: dict[str, float] = {}
        known = set(self.metrics) | set(self.pending)
        volume_selector = r"qty|quant|count|units|number|total|held|ordered"
        cash_selector = r"price|cost|due|sum|fee|charge"

        def take(row: dict[str, Any]) -> None:
            lbls = [vlu for vlu in row.values()
                      if isinstance(vlu, str) and (vlu in known or re.fullmatch(IDENT_AKIN, vlu)) and " " not in vlu]
            figures = [(key, vlu) for key, vlu in row.items()
                       if isinstance(vlu, int) and not isinstance(vlu, bool) and 0 < vlu < 1000 and str(key) != "price"]
            if len(figures) > 1:
                tapered = [(key, vlu) for key, vlu in figures
                            if re.search(volume_selector, str(key).lower()) and not re.search(cash_selector, str(key).lower())]
                if len(tapered) == 1:
                    figures = tapered
            if not lbls or len(figures) != 1:
                return
            lbl = next((ii for ii in lbls if ii in known), lbls[0])
            enumerated[lbl] = enumerated.get(lbl, 0) + figures[0][1]

        route_hashmaps(obv, take)
        if not enumerated:
            return None
        if known and not (set(enumerated) & known):
            return None
        return enumerated

    def glean(self, seen: dict[str, float]) -> None:
        """Take a shown cart as truth, and learn from it whether puts add to a held line or set it."""
        for entity, anticipated in self.pending.items():
            if entity not in seen:
                continue
            exhibited = seen[entity]
            if abs(exhibited - anticipated) < 1e-9 and abs(self.metrics.get(entity, 0) - anticipated) > 1e-9:
                self.appends = False
            elif exhibited > anticipated:
                self.appends = True
        self.pending = {}
        self.metrics = {entity: amt for entity, amt in seen.items() if amt > 0}

    def glanced(self, obv: Any, beat: int) -> None:
        self.ultimate_glance, self.glanced_upon = obv, beat
        if not self.conclude_glance(obv) and isinstance(obv, (dict, list)) and not self.idents_amidst(obv):
            self.metrics = {}

    def idents_amidst(self, obv: Any) -> bool:
        text = qua_prose(obv)
        return any(entity in text for entity in self.metrics)

    @staticmethod
    def metric_prose(amt: float) -> Any:
        if float(amt).is_integer():
            return int(amt)
        return amt

    def abstract(self) -> str:
        if not self.metrics:
            return "[cart as counted] empty."
        return "[cart as counted] " + ", ".join(f"{entity} x{self.metric_prose(amt)}" for entity, amt in self.metrics.items()) + "."

    def faults(self, necessity_k: int, censuses: list[int] | None = None) -> list[str]:
        """What is wrong with the counted cart for a request needing need_n units (or several products)."""
        if not self.metrics:
            return []
        if censuses and len(censuses) > 1:
            return self.satchel_flaws(censuses)
        if len(self.metrics) > 1:
            listing = ", ".join(f"{ii} x{int(amt)}" for ii, amt in self.metrics.items())
            return ["the cart has more than one line (" + listing + "); remove every line except the chosen one"]
        out = []
        for entity, amt in self.metrics.items():
            shortfall = amt - necessity_k
            if abs(shortfall) <= 1e-9:
                continue
            patch = "take out" if shortfall > 0 else "put in"
            out.append(f"the line {entity} has {int(amt)} units while the request needs {necessity_k}; {patch} {int(abs(shortfall))} "
                       "with the cart tool")
        return out

    def satchel_flaws(self, censuses: list[int]) -> list[str]:
        if len(self.metrics) != len(censuses):
            listing = ", ".join(f"{ii} x{int(w_alt)}" for ii, w_alt in self.metrics.items())
            return [f"the cart has {len(self.metrics)} line(s) ({listing}) while the request wants {len(censuses)} products, "
                    "one line apiece; every product needs its own line and nothing else belongs there"]
        maintained = sorted(int(w_alt) for w_alt in self.metrics.values())
        wished_alt = sorted(censuses)
        if maintained == wished_alt:
            return []
        return [f"the lines hold {maintained} units while the products need {wished_alt} (a line for each product, at that product's quantity)"]


class Referee:
    def __init__(self, drive):
        self.drive = drive
        # decline audits
        self.inspections = 0
        self.inspection_befits = None
        self.inspection_idents = None
        self.inspection_browses = None
        # caches for the code-side field evidence
        self.hx_datarows_jotting = None
        self.hx_glance_jotting = {}
        # listings the checks pointed at, and what the last scripted open brought in
        self.named = set()
        self.designated_lineup = []
        self.designated_folios = set()
        self.grade_superior = None
        self.ultimate_browses = None

    def column_designations(self) -> str:
        menu, _ = self.drive.stash.feat_panel()
        if not menu:
            return ""
        return ("\n\nFIELD NAMES SEEN ON THIS STORE'S LISTINGS (names differing only in separators are one name; the number "
                "after each says on how many read listings it occurs):\n" + menu)

    def _non_satisfied(self, entity: str, why: str) -> str:
        return f"According to an independent check, {entity} does NOT meet every requirement - {why}. Pick a different listing."

    def _directed_transcripts(self, refs: list[str], entity: str) -> str:
        """Records of up to three read listings whose text shares a word with the referenced values."""
        stash = self.drive.stash
        tips = set()
        for cite in refs:
            tips |= set(re.findall(r"[a-z]{4,}", cite.lower().split(":", 1)[-1]))
        selected = []
        for lbl in sorted(stash.read):
            if len(selected) >= 3:
                break
            if lbl == entity:
                continue
            text = " ".join(stash.regarding(lbl, 1)).lower()
            if any(cc in text for cc in tips):
                selected.append(lbl)
        if not selected:
            return ""
        transcripts = [line for lbl in selected for line in stash.regarding(lbl, 1)]
        return ("\n\nRECORDS OF READ LISTINGS THE REQUEST POINTS TO (a requirement that refers elsewhere takes its value "
                "from the named field of the listing it means):\n" + "\n".join(transcripts))

    def _personal_comments_prose(self, personal) -> str:
        def care(req, prop, delivered) -> str:
            return (f"one line of its notes may send {prop} elsewhere: \"{delivered[:240]}\". If it does so (telling the reader "
                    f"to take {prop} from, refer to, go by or look at another listing), grading takes {prop} from that "
                    f"listing, and this item's own {prop} field stops deciding '{req}': it holds only when the other "
                    "listing's record shows it. A listing named merely as a version sold elsewhere, or as what this item "
                    "is not, changes nothing.")

        return "\n\nNOTE (what the code read in this item's notes): " + " ".join(care(*element_alt) for element_alt in personal)

    def _decree_selector(self, seal: str, refs, monitored, second: str, personal) -> str:
        """Cache key for a verdict: it changes whenever anything the verdict rests on may have changed."""
        key = seal
        if refs or any(rq.get("status") != "resolved" for rq in monitored):
            key += "|" + str(len(self.drive.stash.read))
        if second:
            key += "|second"
        if personal:
            key += f"|own{len(personal)}"
        return key

    def _policy_outward(self, jotting: str, entity: str, verdict: str, element_alt: dict[str, Any]) -> str:
        self.drive.rulings[jotting] = verdict
        self.drive.diary.append({"kind": "verify", "item": entity, "result": element_alt})
        return verdict

    def _validate_directive(self, entity, objective_prose, second, personal, refs, monitored, past, testimony) -> str:
        drive = self.drive
        needs = drive.needs
        wagon = drive.wagon
        out = [VALIDATE_ARTICLE, "\n\nTHE SHOPPER'S REQUEST:\n", drive.request, needs.modifications(),
               "\n\nREQUIREMENTS NOW (as stated, revealed or changed so far):\n", needs.prose_corpus()]
        if needs.items():
            amidst_buggy = ""
            if wagon.maintained(entity) > 0:
                amidst_buggy = f" It sits in the cart with {int(wagon.maintained(entity))} units."
            out.append("\n\nThis item is one of several products requested, and its category tells which: judge it only "
                       "against that product's own requirements and quantity, leaving aside what the other products need."
                       + amidst_buggy)
        out.append(drive.citecounter.cite_comment(entity) or "")
        unrecorded = drive.citecounter.cite_unrecorded(refs, monitored)
        if unrecorded:
            out.append(self._directed_transcripts(unrecorded, entity))
        out.append(second)
        if personal:
            out.append(self._personal_comments_prose(personal))
        out.append(f"\n\nTHE ITEM TO {objective_prose}: {entity}\nSTORE RECORDS OF THIS ITEM (earliest first):\n"
                   + "\n".join(past))
        if wagon.ultimate_glance is not None:
            out.append("\n\nLATEST CART VIEW:\n" + qua_prose(wagon.ultimate_glance)[:3000])
        out.append(self.column_designations())
        if testimony["lines"]:
            out.append("\n\nFIELD EVIDENCE (worked out by code from the records of this item's category: how the shop "
                       "spells each required value, and what this item shows):\n" + "\n".join(testimony["lines"]))
        captions = drive.stash.caption_orderlines()
        if captions:
            out.append("\n\nLABELS RECURRING ON THIS STORE'S LISTINGS (with counts; the item's category label must fit "
                       "the request's wording as a whole, never a neighbouring label):\n" + captions)
        return "".join(out)

    def validate_article(self, entity: str, objective_prose: str, second: str = "") -> str:
        """Independent fit check of one listing; returns a warning for the agent, or "" when it passes or cannot run."""
        drive = self.drive
        if drive.channel.extra_alt() < SEALING + 20:
            return ""
        seal = f"{entity}|{drive.needs.edition}"
        if seal in drive.shelf.nigh_sound:
            return ""
        provided = drive.needs.browsing.get("references")
        refs = [str(rq) for rq in provided or [] if rq] if isinstance(provided, list) else []
        monitored = list(drive.refs or [])
        personal = drive.citecounter.personal_diverts(entity)
        jotting = self._decree_selector(seal, refs, monitored, second, personal)
        if jotting in drive.rulings:
            return drive.rulings[jotting]
        pointer = drive.citecounter.cite_arrow_comment(entity)
        if pointer:
            return self._policy_outward(jotting, entity, self._non_satisfied(entity, pointer),
                                  {"item": entity, "fits": False, "problems": ["reference chain"]})
        past = drive.stash.regarding(entity)
        if not past:
            return ""
        testimony = self.column_glance(entity)
        if testimony["fails"]:
            apex = testimony["fails"][:3]
            return self._policy_outward(jotting, entity, self._non_satisfied(entity, "; ".join(apex)),
                                  {"item": entity, "fits": False, "problems": apex, "by": "field evidence"})
        answer = drive.channel.canvass(self._validate_directive(entity, objective_prose, second, personal, refs, monitored, past, testimony),
                                  3000, VALIDATOR)
        drive.diary.append({"kind": "verify", "item": entity, "result": answer})
        if answer and answer.get("fits") is False:
            problems = [str(pz) for pz in answer.get("problems") or []][:5]
            verdict = (f"According to an independent check, {entity} does NOT meet every requirement - " + "; ".join(problems)
                       + ". Unless you are sure the check got it wrong, pick another fully verified fit.")
            drive.rulings[jotting] = verdict
            return verdict
        drive.rulings[jotting] = ""
        if answer:
            drive.passed.add(seal)
            drive.pass_price[entity] = drive.stash.price(entity)
        return ""

    def hx_datarows(self) -> dict[str, dict[str, Any]]:
        """Code-parsed rows of every record, cached until the archive grows."""
        buffered = self.hx_datarows_jotting
        if buffered and buffered[0] == len(self.drive.stash.items):
            return buffered[1]
        rows = self.drive.sealer.routine_datarows()
        self.hx_datarows_jotting = (len(self.drive.stash.items), rows)
        return rows

    def warehouse_columns(self, label: Any = "") -> dict[str, list[str]]:
        """field name -> every value read on the listings of the relevant shelf (or all shelves)."""
        drive = self.drive
        try:
            rows = self.hx_datarows()
            houses = drive.shelf.hub_aisles()
            if drive.needs.items():
                ofs = drive.speak.commodity_relating(str(label))
                shelves = set(houses[ofs]) if ofs is not None and ofs < len(houses) else set()
            else:
                minee = drive.shelf.personal_aisle()
                if minee:
                    shelves = {minee}
                elif houses:
                    shelves = set(houses[0])
                else:
                    shelves = set()
            harvested: dict[str, list[str]] = {}
            for row in rows.values():
                if not row["fields"] or (shelves and row["shelf"] not in shelves):
                    continue
                for name, vlu in row["fields"].items():
                    if trash_selector(name) or not str(vlu).strip():
                        continue
                    harvested.setdefault(str(name), []).append(str(vlu))
            return harvested
        except Exception:
            return {}

    def hx_diverted(self, entity: str, prop: str) -> bool:
        """Do this listing's own notes possibly send the property elsewhere?"""
        try:
            for obtained in self.drive.citecounter.personal_diverts(entity) or []:
                crowns = list(obtained if isinstance(obtained, (list, tuple)) else [obtained])[:2]
                if any(equal_feat(str(f_alt).partition(":")[0], prop) for f_alt in crowns):
                    return True
            return False
        except Exception:
            return True

    def column_glance(self, entity: str) -> dict[str, list[str]]:
        """Cached field evidence for a listing, softened by any meaning check that confirmed a value."""
        drive = self.drive
        needs = drive.needs
        jotting = self.hx_glance_jotting
        key = (entity, len(drive.stash.items), needs.edition, "|".join(needs.feats_residual()))
        if key not in jotting:
            try:
                jotting[key] = self.column_data(entity)
            except Exception as failure:
                drive.diary.append({"kind": "field_facts_error", "error": f"{type(failure).__name__}: {failure}"[:300]})
                jotting[key] = {"lines": [], "rules": [], "fails": []}
        view = jotting[key]
        affirmed = (drive.shelf.nigh_uttered or {}).get(f"{entity}|{needs.edition}")
        if not affirmed:
            return view
        try:
            return self._affirmed_glance(view, affirmed)
        except Exception:
            return view

    @staticmethod
    def _affirmed_glance(view: dict[str, list[str]], affirmed) -> dict[str, list[str]]:
        def pairings(field: str, prop: str) -> bool:
            return equal_feat(field, prop) or stringent_feat(field, prop)

        def glossed(line: str) -> str:
            head = re.match(r"- (.+?) = '", line)
            if not head:
                return line
            hit = next(((pz, vz, s) for pz, vz, s in affirmed if pairings(head.group(1), pz)), None)
            if not hit:
                return line
            return line + (f" A meaning check found that its '{hit[2][:80]}' says '{hit[1][:60]}' in other "
                           "words, so it meets it.")

        props = [pz for pz, _, _ in affirmed]
        kept = [f for f in view["fails"] if not any(pairings(f.split(":", 1)[0].strip(), pz) for pz in props)]
        return {"lines": [glossed(line) for line in view["lines"]], "rules": list(view["rules"]), "fails": kept}

    @staticmethod
    def _crop(text: Any, n: int = 70) -> str:
        even = " ".join(str(text).split())
        return even if len(even) <= n else even[: n - 3] + "..."

    def column_data(self, entity: str) -> dict[str, list[str]]:
        """Compare the listing's fields with how the rest of its category writes each required value."""
        result: dict[str, list[str]] = {"lines": [], "rules": [], "fails": []}
        rows = self.hx_datarows()
        minee = rows.get(entity)
        if not minee or not minee["fields"]:
            return result
        several = bool(self.drive.needs.items())
        shelf = minee["shelf"]
        if several and not shelf:
            return result
        slot = None
        if several:
            slot = next((n for n, hub in enumerate(self.drive.shelf.hub_aisles()) if shelf in hub), None)
        cohort = {ii: rq for ii, rq in rows.items() if rq["fields"] and (not shelf or rq["shelf"] == shelf)}
        if len(cohort) < 3:
            return result
        convey: dict[str, int] = {}  # field name -> how many peers fill it
        grid: dict[str, list[tuple[str, str, list[str]]]] = {}  # field name -> (id, value piece, its words)
        for lbl, row in cohort.items():
            for name, vlu in row["fields"].items():
                if trash_selector(name) or not str(vlu).strip():
                    continue
                convey[name] = convey.get(name, 0) + 1
                fragments = hx_components(vlu)
                if len(fragments) > 1:
                    fragments += [str(vlu).strip().lower()]
                grid.setdefault(name, []).extend((lbl, shard, hx_vocables(shard)) for shard in fragments)
        ctxt = (rows, minee, shelf, several, slot, cohort, convey, grid)
        for req in self.drive.needs.feats_residual()[:12]:
            self._datum_towards(entity, req, ctxt, result)
        return result

    def _edge_datum(self, prop, value, head, ctxt, result) -> None:
        """A numeric limit the store keeps in a field this listing lacks cannot be shown met."""
        rows, minee, shelf, several, slot, cohort, convey, grid = ctxt
        crop = self._crop
        personal_niche = self.drive.speak.commodity_relating(head) if several else None
        relatives = {wq for wq in set(feat_vocables(prop)) if len(wq) >= 3 and wq not in ANCHOR_VOCABLES}
        named = [name for name in convey if convey[name] >= 2 and stringent_feat(name, prop)]
        if not prop or not relatives or not named:
            return
        if several and personal_niche is not None and personal_niche != slot:
            return
        if any(str(vz).strip() and (stringent_feat(name, prop) or set(feat_vocables(name)) & relatives)
               for name, vz in minee["fields"].items()):
            return
        itemized = " / ".join(f"'{xq}'" for xq in named[:3])
        retaining = sum(1 for rq in cohort.values() if any(str(rq['fields'].get(xq, '')).strip() for xq in named))
        result["lines"].append(f"- {prop} = '{crop(value)}': the shop keeps it under {itemized}, present on {retaining} of "
                               f"the {len(cohort)} listings read in this category. This listing lacks that field, so the limit is "
                               "not proven and the item fails it: it does NOT meet it (a figure from another field does not "
                               "count instead).")
        result["fails"].append(f"{prop}: it has no {named[0]} field although this store's listings of the category do, so "
                               f"'{crop(value)}' remains unproven")

    @staticmethod
    def _dissect_prerequisite(req):
        """(head, property, value) of a requirement, or None for references and non-comparable values."""
        head, _, crude = str(req).partition(":")
        value = " ".join(re.sub(r"\((?:stated|revealed[^)]*)\)", " ", crude).split())
        prop = re.split(r"\s*\(", head)[0].strip()
        if not prop or not value or cite_prose(value):
            return None
        if re.search(r"as shown|same as|the page|sold by|redirect|refer", value.lower()):
            return None
        return head, prop, value

    def _column_designations_towards(self, datum, ctxt):
        """The store fields that carry the property (by name, or else by the values they write), heaviest first."""
        convey, grid = ctxt[6], ctxt[7]
        prop, alts = datum["prop"], datum["alts"]

        def through_heft(names):
            return sorted(names, key=lambda xq: -convey[xq])

        named = through_heft(n for n in convey if stringent_feat(n, prop))
        datum["by_name"] = bool(named)
        if named and not datum["positive"]:
            even = " ".join(re.findall(r"[a-z0-9]+", datum["value"].lower()))
            if any(" ".join(words) == even for n in named for _, _, words in grid.get(n, [])):
                datum["positive"] = True
                EXACT_MEASUREMENTS.add(even)
        if named:
            return named
        if not datum["positive"] or (ctxt[3] and datum["slot"] is None):
            return None
        authoring = [n for n in convey if convey[n] >= 2 and any(hx_concord(alts, words) for _, _, words in grid.get(n, []))]
        if not authoring:
            return None
        fullest = max(authoring, key=lambda xq: convey[xq])
        return through_heft(n for n in convey if stringent_feat(n, fullest))

    def _policy_prose(self, datum) -> str:
        crop = self._crop
        lead, value = datum["lead"], datum["value"]
        if not datum["positive"]:
            return lead
        if datum["writers"]:
            surplus = ""
            if not datum["by_name"]:
                surplus = (" As no field bears the property's name, a title that spells the value out is enough too, unless "
                         "a field of its own sharing the property's name says otherwise.")
            return (lead + " Only a value spelling those very words passes here (containment and a single list element "
                    "count; synonyms and related words do not)." + surplus)
        satisfied = lead + " Values met so far: " + " | ".join(datum["samples"])
        if datum["naming"]:
            return (satisfied + f"; '{crop(value)}' appears in the title or text of listings in this category "
                    f"({', '.join(datum['naming'])}), so the shop uses that very word: other wording counts only once such "
                    "a listing's record shows this field writes it differently.")
        if datum["on"] < HX_ALIAS_LEAST:
            return (satisfied + f"; none spells it out, but this field occurs on too few read listings to tell whether the shop "
                    f"would write '{crop(value)}' literally: a value with the same meaning is not shown to meet it yet.")
        return satisfied + "; none spells it out, so a value with the same meaning may meet it."

    def _datum_towards(self, entity, req, ctxt, result) -> None:
        """Add the FIELD EVIDENCE rule and line (and maybe a hard fail) for one requirement."""
        rows, minee, shelf, several, slot, cohort, convey, grid = ctxt
        crop = self._crop
        dissected = self._dissect_prerequisite(req)
        if dissected is None:
            return
        head, prop, value = dissected
        minor = value.lower()
        if re.search(r"\d", value) and re.search(HX_EDGE, minor):
            self._edge_datum(prop, value, head, ctxt, result)
            return
        if re.fullmatch(r"\s*\d+(?:\.\d+)?\s+[a-z]+\s*", minor):
            return
        datum = {"prop": prop, "value": value, "slot": self.drive.speak.commodity_relating(head) if several else None}
        if several and datum["slot"] is not None and datum["slot"] != slot:
            return
        datum["positive"] = favorable_measurement(value, head)
        if not datum["positive"] and not re.search(PRECLUDE_VOCABLES, minor):
            return
        alts = datum["alts"] = hx_alternates(value)
        if not alts:
            return
        named = self._column_designations_towards(datum, ctxt)
        if named is None:
            return
        datum["named"] = named

        def populates(row) -> bool:
            return any(str(row["fields"].get(n, "")).strip() for n in named)

        datum["on"] = sum(1 for rq in cohort.values() if populates(rq))
        rest_alt = [(lbl, words) for n in named for lbl, _, words in grid.get(n, []) if lbl != entity]
        datum["writers"] = sorted({lbl for lbl, words in rest_alt if hx_concord(alts, words)})
        datum["alone"] = {" ".join(words) for _, words in rest_alt}
        datum["samples"] = list(dict.fromkeys(crop(rq["fields"][n], 40) for rq in cohort.values() for n in named
                                             if str(rq["fields"].get(n, "")).strip()))[:6]
        rear = f"{len(datum['writers'])} other listing(s) write '{crop(value)}' itself in it."
        if not datum["positive"]:
            naked = " ".join(re.sub(PRECLUDE_VOCABLES, " ", minor).split())
            naked_alternates = hx_alternates(naked)
            if naked_alternates:
                maintained = {lbl for lbl, words in rest_alt if hx_concord(naked_alternates, words)}
                rear = (f"{len(maintained)} other listing(s) put '{crop(naked)}' there, which is the excluded value (a listing "
                        "writing it does not meet it).")
        lead = f"- {prop} = '{crop(value)}': the store records it under " + " / ".join(f"'{xq}'" for xq in named[:3])
        if not datum["by_name"]:
            lead += f" (no field is called '{prop}'; this one's values carry the value)"
        datum["lead"] = lead + f", present on {datum['on']} of the {len(cohort)} listings read in this category; " + rear
        kinds = variety_vocables(value)
        datum["naming"] = []
        if kinds and not datum["writers"]:
            datum["naming"] = [ii for ii, rq in rows.items() if ii != entity and (not shelf or rq["shelf"] == shelf)
                              and kinds <= rq["words"]][:3]
        result["rules"].append(self._policy_prose(datum))
        self._datum_orderline(entity, prop, value, datum["positive"], datum["by_name"], named, datum["writers"], datum["alone"],
                        datum["samples"], datum["naming"], datum["on"], datum["lead"], ctxt, result)

    def _eligible_columns(self, fields, named, prop, value):
        """Fields whose name is the property's name with one extra word in front."""
        footing = feat_vocables(prop)
        if not footing:
            return []
        out = []
        for name, vz in fields.items():
            if name in named or not str(vz).strip() or trash_selector(name):
                continue
            words = feat_vocables(name)
            if len(words) == len(footing) + 1 and words[1:] == footing:
                out.append((name, str(vz)))
        return out

    def _absent_column_orderline(self, lead, prop, value, fields, named, qlt_comment, headline_states) -> str:
        crop = self._crop
        if qlt_comment:
            return lead + " That field is missing from this listing." + qlt_comment
        if headline_states:
            return (lead + " No such field here, and no field bears the property's name, but its title writes "
                    f"'{crop(value)}': it meets it.")
        concise = {wq for wq in feat_vocables(prop) if len(wq) >= 3}
        pertinent = [(name, vz) for name, vz in fields.items() if name not in named and str(vz).strip() and not trash_selector(name)
                   and (set(feat_vocables(name)) & concise or hx_pens(value, vz))]
        pertinent = sorted(pertinent, key=lambda duo: not hx_pens(value, duo[1]))[:4]
        itemized = ""
        if pertinent:
            itemized = ": fields that might hold the property: " + "; ".join(f"{name}: '{crop(vz, 60)}'" for name, vz in pertinent)
        return (lead + " No such field here" + itemized + ". Another label for this SAME property is acceptable; a field of "
                "some other property (a height for a type, a frame material for the material) does not stand in.")

    def _datum_orderline(self, entity, prop, value, positive, by_name, named, writers, alone, samples, naming, on, lead, ctxt,
                   result) -> None:
        """The FIELD EVIDENCE line on what this listing itself shows for one requirement."""
        crop = self._crop
        fields = ctxt[1]["fields"]
        personal = [(name, str(fields[name])) for name in named if str(fields.get(name, "")).strip()]
        exhibited = "; ".join(f"{name}: '{crop(vz, 90)}'" for name, vz in personal)
        eligible_alt = self._eligible_columns(fields, named, prop, value)
        qlt_pens = [duo for duo in eligible_alt if hx_pens(value, duo[1])]
        qlt_nigh = bool(qlt_pens) or any(hx_grazes(value, vz) for _, vz in eligible_alt)
        qlt_comment = ""
        if qlt_pens:
            lengthier_designation, lengthier_measurement = qlt_pens[0]
            qlt_comment = (f" Its {lengthier_designation} '{crop(lengthier_measurement, 60)}' holds it (a longer key naming this same "
                         "property: one extra word ahead of the property's name still names that property, only more "
                         "fully), so it meets it.")
        lines = result["lines"]
        if self.hx_diverted(entity, prop):
            lines.append(lead + " Notes on this listing may point to another listing for this property, so its own "
                         "field might not decide.")
            return
        if not positive:
            lines.append(lead + " Here: " + (exhibited or "no such field") + " (an exclusion, so the listing must not carry "
                         "the excluded value).")
            return
        relatives_vocables = {wq for wq in feat_vocables(prop) if len(wq) >= 3 and wq not in ANCHOR_VOCABLES}
        relatives = [name for name, vz in fields.items() if str(vz).strip() and not trash_selector(name) and set(feat_vocables(name)) & relatives_vocables
               and not hx_pens(value, vz)]
        headline_states = not by_name and not relatives and hx_pens(value, ctxt[1].get("title") or "")
        if not personal:
            lines.append(self._absent_column_orderline(lead, prop, value, fields, named, qlt_comment, headline_states))
            return
        alts = hx_alternates(value)

        def possesses_measurement(component: str) -> bool:
            return hx_concord(alts, hx_vocables(component))

        fragments = [pz for _, vz in personal for pz in hx_components(vz)] + [vz.strip().lower() for _, vz in personal if len(hx_components(vz)) > 1]

        def bumped(component: str) -> list[str]:
            # sides of a joined value that other listings write on their own in this field
            facets = hx_facets(component)
            if not facets or not any(possesses_measurement(flank) for flank in facets):
                return []
            return [flank for flank in facets if not possesses_measurement(flank) and " ".join(hx_vocables(flank)) in alone]

        unions = [(pz, bumped(pz)) for pz in fragments if possesses_measurement(pz)]
        tidy = [pz for pz, surplus in unions if not surplus]
        spliced = [(pz, surplus) for pz, surplus in unions if surplus]
        line, falter = self._personal_column_decision(lead, exhibited, prop, value, named, personal, writers, samples, naming, on, by_name,
                                             tidy, spliced, qlt_comment, qlt_nigh, headline_states)
        lines.append(line)
        if falter:
            result["fails"].append(falter)

    def _personal_column_decision(self, lead, exhibited, prop, value, named, personal, writers, samples, naming, on, by_name, tidy, spliced,
                           qlt_comment, qlt_nigh, headline_states):
        """(evidence line, hard-fail text or "") for a listing that does fill the property's field."""
        crop = self._crop
        herein = lead + f" Here: {exhibited}"
        if tidy:
            return herein + f" - it writes '{crop(value)}': it meets it.", ""
        if qlt_comment:
            return herein + "." + qlt_comment, ""
        if headline_states:
            return (herein + " - as no field bears the property's name, the title decides, and its title writes "
                    f"'{crop(value)}': it meets it."), ""
        if spliced:
            component, rest_alt = spliced[0]
            line = (herein + f" - '{crop(component)}' joins '{crop(value)}' with '{rest_alt[0]}', a value other listings carry on "
                    "its own in this field, so it does NOT meet it.")
            falter = ""
            if by_name and not qlt_nigh:
                falter = (f"{prop}: its {named[0]} is '{crop(component)}', which joins {crop(value)} with {rest_alt[0]} (a "
                        f"value this store also writes on its own there), so it is not {crop(value)}")
            return line, falter
        if writers:
            line = (herein + f" - '{crop(value)}' is absent, although other listings of this category write it in this "
                    "field: so it fails (different wording, a synonym or a related word does not count where the store "
                    "writes the value itself).")
            falter = ""
            if by_name and not qlt_nigh and not any(hx_grazes(value, vz) for _, vz in personal):
                falter = (f"{prop}: its {named[0]} is '{crop(personal[0][1], 60)}', whereas other listings of this store carry "
                        f"'{crop(value)}' literally in that field")
            return line, falter
        noticed_vlus = " | ".join(samples)
        if naming:
            return (herein + f" - '{crop(value)}' is not written. No read listing puts it in this field yet (values: "
                    + noticed_vlus + f"), but titles or texts of this category's listings use '{crop(value)}' "
                    f"({', '.join(naming)}), so the shop has its own word for it: this listing's different value is "
                    "therefore not shown to meet it."), ""
        if on < HX_ALIAS_LEAST:
            return (herein + f". No read listing of this category puts '{crop(value)}' into this field (seen values: "
                    + noticed_vlus + f"); with just {on} read listing(s) having the field, there is no telling yet whether "
                    f"'{crop(value)}' would be written literally, so a value with the same meaning is not shown to meet "
                    "it yet."), ""
        return (herein + f". No read listing of this category puts '{crop(value)}' into this field (seen values: "
                + noticed_vlus + "), so a value with the same meaning may meet it."), ""

    def grade_validate(self, entity: str) -> str:
        """Ask a ranker whether any seen listing could beat the pick on the cheapest/best rule."""
        drive = self.drive
        needs = drive.needs
        priority = needs.priority()
        seal = f"{entity}|{needs.edition}|{drive.refusal_n}"
        if priority == "none" or seal in drive.rank_checked:
            return ""
        if drive.channel.extra_alt() < SEALING + 30 or drive.maneuvers_residual() < 4:
            return ""
        drive.rank_checked.add(seal)
        cheapest = priority == "cheapest"
        if cheapest:
            which = "cheapest (lowest price)"
        else:
            nuance = needs.browsing.get("priority_detail") or ""
            which = f"best ({nuance})" if nuance else "best"
        components = [GRADE_VALIDATE.replace("PRIORITY", which), "\n\nTHE SHOPPER'S REQUEST:\n", drive.request, needs.modifications(),
                 "\n\nREQUIREMENTS NOW:\n", needs.prose_corpus(), drive.citecounter.cite_comment()]
        if needs.items():
            components.append("\n\nSeveral products are wanted, so the chosen item competes only with listings of the SAME "
                         "product (category); a listing for another product can never beat it.")
        tiles = drive.stash.tiles(20000, entity if cheapest else "", drive.citecounter.cite_alien(), {entity})
        under = "; the cards cover only listings cheaper than the chosen one" if cheapest else ""
        components.append(f"\n\nTHE CHOSEN ITEM: {entity}\n\nLISTINGS SEEN (one card per listing; DETAILS = its full record was opened, ROW = "
                     "seen only as a search row, its fields unread: a ROW in the requested category (whatever object its "
                     "title names) that beats the chosen item and is not excluded by its price or stock MAY fit and "
                     "gets named" + under + "):\n" + "\n".join(tiles))
        components.append(self.column_designations())
        rules = self.column_glance(entity)["rules"]
        if rules:
            components.append("\n\nFIELD EVIDENCE (worked out by code from the records of the chosen item's category: how the "
                         "shop spells each required value):\n" + "\n".join(rules))
        answer = drive.channel.canvass("".join(components), 2500, VALIDATOR)
        drive.diary.append({"kind": "rank", "result": answer})
        gone = drive.shelf.inert_orderlines()
        better = []
        for xq in (answer or {}).get("better") or []:
            if not isinstance(xq, (str, int)):
                continue
            xq = str(xq)
            if xq == entity or xq in drive.refused_lines or xq in gone:
                continue
            better.append(xq)
        better = [xq for xq in better[:4] if not self.column_glance(xq)["fails"]]
        self.grade_superior = list(better)
        self.named.update(better)
        if not better:
            return ""
        reason = f" ({answer.get('reason')})" if answer.get("reason") else ""
        return (f"A ranker thinks these listings might also fit and beat {entity} on the {which}: " + ", ".join(better)
                + reason + ". Look each one over (details, stock, price) before ordering; if all of them fail, order the current pick.")

    def _unbrowsed_impacts(self, cap, amt, shelves, impact_aisle) -> list[str]:
        """Unopened search hits, best scored first, on the right shelves and inside the budget."""
        drive = self.drive
        stash = drive.stash
        merits = drive.encompass.impact_merit

        def admissible(lbl: str) -> bool:
            if lbl in stash.read:
                return False
            label = impact_aisle.get(lbl)
            if shelves and label and label not in shelves:
                return False
            return not (be_numeral(cap) and be_numeral(stash.price(lbl)) and stash.price(lbl) * amt > cap)

        return [lbl for lbl in sorted(drive.encompass.every_impacts, key=lambda ii: -merits.get(ii, 0)) if admissible(lbl)]

    def _inspection_directive(self, tiles, crisp) -> str:
        drive = self.drive
        out = [PRECEDING_REJECT, "\n\nTHE SHOPPER'S REQUEST:\n", drive.request, drive.needs.modifications(),
               "\n\nREQUIREMENTS NOW:\n", drive.needs.prose_corpus(), drive.citecounter.cite_comment(),
               "\n\nLISTINGS SEEN (one card per listing; DETAILS = its full record was opened, ROW = seen only as a search row):\n",
               "\n".join(tiles)]
        if crisp:
            out.append("\n\nOPENED SINCE THE PREVIOUS CHECK (their DETAILS cards above are new: weigh them before the "
                       "rest, field by field, by the reading rules): " + ", ".join(crisp))
        out.append(self.column_designations())
        return "".join(out)

    def _inspection_sequels(self, unopened: list[str]):
        """Gather what the agent should still open or search before a decline: rows, result pages, joined queries."""
        drive = self.drive
        stash = drive.stash
        shelves = drive.observe.aisle_captions()
        cap, amt = drive.needs.disburse_top, drive.needs.quantity()
        hits = self._unbrowsed_impacts(cap, amt, shelves, drive.observe.impact_aisle)
        try:
            cushion = [ii for ii in drive.shelf.cushion_datarows() if ii not in stash.read]
        except Exception:
            cushion = []
        blended = cushion + hits[:4] + [ii for ii in unopened if ii not in hits]
        unopened = list(dict.fromkeys(blended))[:6]
        spacious = drive.maneuvers_residual() > 5
        pages = drive.observe.access_folios(2) if spacious else []
        if pages:
            unopened = unopened[:4]
            self.designated_folios.update(json.dumps(pz, sort_keys=True, default=str) for pz in pages)
        try:
            combis = drive.walk.pooled_hunttexts() if drive.maneuvers_residual() > 5 else []
        except Exception as failure:
            drive.diary.append({"kind": "combo_error", "error": f"{type(failure).__name__}: {failure}"[:300]})
            combis = []
        if combis:
            self.designated_folios.update(json.dumps(pz, sort_keys=True, default=str) for pz in combis)
        return unopened, pages, combis

    def _inspection_comment(self, answer, fits, unopened, pages, combis) -> str:
        search = self.drive.role('search')

        def calls(discovered) -> str:
            return "; ".join(f"{search} {json.dumps(pz, ensure_ascii=False)}" for pz in discovered)

        notes = []
        if fits:
            reason = f" ({answer.get('reason')})" if answer.get("reason") else ""
            notes.append("A separate review suggests these listings might meet every requirement: " + ", ".join(fits)
                         + reason + ".")
        if pages:
            notes.append("Some searches for required values matched more listings than were read so far; fits may be "
                         "waiting on pages not read yet: " + calls(pages) + ". Work through those pages first (each step "
                         "brings a full page of rows) and open the in-budget rows of the requested kind they show.")
        if combis:
            notes.append("Searches combining the required values have not been tried yet (where the store requires every "
                         "word, each returns only the listings holding all of them): " + calls(combis) + ". Run them "
                         "(the search tool still allows them) and open the in-budget rows of the requested kind they show.")
        if unopened:
            notes.append("Listings of the requested kind seen only as search rows and never opened, so their fields are still "
                         "unchecked: " + ", ".join(unopened) + ". Read their details first (several may go in one reply).")
        return (" ".join(notes) + " Confirm them (details, stock, price) and buy the one that fits; decline only when every one of "
                "them really fails.")

    def evaluation_reject(self, early: bool = False) -> str:
        """Before a decline: have an auditor look for possible fits and list what is still unopened."""
        drive = self.drive
        stash = drive.stash
        constraint = self.inspection_browses if self.inspection_browses is not None else 10 ** 9
        might_recur = len(stash.read) > constraint and self.inspections < 2
        if drive.audit_done and not might_recur:
            return ""
        if drive.channel.extra_alt() < SEALING + 30:
            return ""
        drive.audit_done = True
        if not early:
            self.inspections += 1
        preceding = self.inspection_idents
        omit = () if preceding is None else set(stash.read) - set(preceding or ())
        tiles = stash.tiles(16000, "", drive.citecounter.cite_alien(), omit)
        self.inspection_browses = len(stash.read)
        if not tiles:
            return ""
        crisp = sorted(stash.read - preceding) if preceding is not None else []
        self.inspection_idents = set(stash.read)
        answer = drive.channel.canvass(self._inspection_directive(tiles, crisp), 6000, EXAMINER)
        drive.diary.append({"kind": "audit", "result": answer})
        reply = answer or {}

        def idents_beneath(key: str) -> list[str]:
            return [str(xq) for xq in reply.get(key) or [] if isinstance(xq, (str, int))]

        fits = [xq for xq in idents_beneath("possible_fits")[:5] if not self.column_glance(xq)["fails"]]
        self.inspection_befits = fits
        proposed = [xq for xq in idents_beneath("unopened") if xq not in stash.read and xq in stash.ids][:6]
        unopened, pages, combis = self._inspection_sequels(proposed)
        self.named.update(fits + unopened)
        self.designated_lineup += [ii for ii in fits + unopened if ii not in self.designated_lineup]
        if fits or unopened or pages or combis:
            return self._inspection_comment(answer, fits, unopened, pages, combis)
        return ""

    def designated_unbrowsed(self) -> list[str]:
        """Unopened listings worth opening: scored hits first, then those the checks named."""
        drive = self.drive
        stash = drive.stash
        cap, amt = drive.needs.disburse_top, drive.needs.quantity()
        shelves, impact_aisle = drive.observe.aisle_captions(), drive.observe.impact_aisle
        merits = drive.encompass.impact_merit
        ranked = sorted(drive.encompass.every_impacts, key=lambda ii: -merits.get(ii, 0))

        def valuation(lbl: str) -> bool:
            price = stash.price(lbl)
            if lbl in stash.read or lbl in drive.refused_lines or lbl not in stash.ids:
                return False
            if be_numeral(cap) and be_numeral(price) and price * amt > cap:
                return False
            return not (shelves and impact_aisle.get(lbl) and impact_aisle[lbl] not in shelves)

        return [lbl for lbl in dict.fromkeys(ranked + list(self.designated_lineup)) if valuation(lbl)]

    def nuances_trigger(self, lbl: str) -> dict[str, Any] | None:
        """Arguments for opening ident: copy an earlier details call, else fill the tool's single id slot."""
        drive = self.drive
        tool = drive.role("details")
        known = drive.stash.ids

        def be_recognized(vz: Any) -> bool:
            return isinstance(vz, str) and vz in known

        for element_alt in reversed(drive.diary):
            if element_alt.get("kind") != "env":
                continue
            for operate in element_alt.get("actions") or []:
                if operate.get("name") != tool:
                    continue
                args = operate.get("args")
                if isinstance(args, dict) and any(be_recognized(vz) for vz in args.values()):
                    return {kx: (lbl if be_recognized(vz) else vz) for kx, vz in args.items()}
        framework = drive.contours.get(tool or "") or {}
        props = framework.get("properties") if isinstance(framework.get("properties"), dict) else {}
        proses = [kx for kx, pz in props.items() if isinstance(pz, dict) and pz.get("type") == "string" and not pz.get("enum")]
        required = framework.get("required") or []
        slot = next((kx for kx in proses if kx in required), None)
        if not slot and len(proses) == 1:
            slot = proses[0]
        if not slot:
            return None
        return {slot: lbl}

    def access_browses(self, waitlist: list[str], utter: str, lot: tuple[str, str, int] | None, browser: str | None = None) -> list[tuple[str, str]]:
        """Run scripted detail reads of the queued ids (batched when a batch tool exists) and return the replies."""
        drive = self.drive
        if lot:
            tool, operand, maximal = lot
            calls = [{operand: waitlist[at: at + maximal]} for at in range(0, len(waitlist), maximal)]
        else:
            tool = browser or drive.role("details")
            calls = []
            if tool:
                constructed_every = (drive.shelf.fetch_trigger(lbl, tool) for lbl in waitlist)
                calls = [constructed for constructed in constructed_every if constructed]
        beat = drive.channel.beat
        pitches = [{"id": f"open_{n}_{beat}", "type": "function", "function": {"name": tool, "arguments": json.dumps(args)}}
                     for n, args in enumerate(calls)]
        if not pitches:
            return []
        drive.chat.append({"role": "assistant", "content": utter, "tool_calls": pitches})
        start = len(drive.stash.items)
        drive.covering = True
        try:
            rejoinders = drive.forward(pitches, automated=True)
        finally:
            drive.covering = False
        self.ultimate_browses = drive.stash.items[start:]
        return rejoinders

    def _grab_waitlist(self):
        """Pop the queued opens; failing that, the stock re-check queued after a cart put."""
        drive = self.drive
        browses = drive.browses
        waitlist, browses.access_waitlist = browses.access_waitlist, None
        if waitlist or not browses.deposit_waitlist or drive.channel.done or drive.channel.snapped:
            return waitlist, False
        waitlist, browses.deposit_waitlist = browses.deposit_waitlist, None
        tool, browses.deposit_utility = browses.deposit_utility, None
        browses.access_revalidate, browses.access_intention, browses.access_utility = True, "cart", tool
        browses.access_utter = "Re-checking stock on the line just added, as an item can sell out the moment it reaches the cart."
        return waitlist, True

    def _stockpile_shorten(self, waitlist: list[str], purpose: str) -> list[str]:
        """Shrink the queue when opening it all would eat the steps kept for carting and ordering."""
        drive = self.drive
        try:
            keep = drive.admission.checkout_expected() if drive.orders_tried == 0 else PRIMARY_CHECKOUT_THROUGH
            room = drive.maneuvers_residual() - keep - 1
            itemized = drive.shelf.lot_utility() and len(waitlist) >= 2
            necessary = 1 if itemized else len(waitlist)
            if room < necessary and (drive.wagon.metrics or not drive.admission.unvalidated()):
                bundled = drive.shelf.lot_utility() and len(waitlist) >= 2
                snip = [] if bundled else waitlist[: max(0, room)]
                drive.diary.append({"kind": "reserve_skip", "purpose": purpose, "ids": waitlist[:6], "kept": snip})
                return snip
        except Exception as failure:
            drive.diary.append({"kind": "reserve_skip_error", "error": f"{type(failure).__name__}: {failure}"[:300]})
        return waitlist

    def access_designated(self) -> bool:
        """Open the queued listings (scripted), then attach a decline review to the last reply when useful."""
        drive = self.drive
        channel = drive.channel
        waitlist, fromsrc_deposit = self._grab_waitlist()
        if not waitlist or channel.done or channel.snapped:
            return False
        waitlist = list(dict.fromkeys(str(ii) for ii in waitlist))
        browses = drive.browses
        revalidate, browses.access_revalidate = browses.access_revalidate, False
        purpose = str(browses.access_intention or "")
        browses.access_intention = ""
        floor, browses.access_bottom = browses.access_bottom, None
        browser, browses.access_utility = browses.access_utility, None
        utter, browses.access_utter = browses.access_utter, ""
        if not utter:
            if revalidate:
                utter = "A last look at the cart lines before the order."
            else:
                utter = "The checks named some listings, so they get opened before any decline."
        if purpose in ("menu", "quota", "walk") and drive.role("order") and drive.role("cart") and not channel.muffle:
            waitlist = self._stockpile_shorten(waitlist, purpose)
            if not waitlist:
                return False
        if purpose == "cart":
            obsolete = set(drive.shelf.obsolete_orderlines())
            waitlist = [ii for ii in waitlist if ii in obsolete or (not fromsrc_deposit and ii not in drive.wagon.metrics)]
            if not waitlist:
                return False
        lot = drive.shelf.lot_utility() if len(waitlist) >= 2 else None
        if lot and purpose == "cart" and not drive.batch_proven:
            lot = None
        if lot and browser and not ((drive.shelf.utility_tops or {}).get(lot[0]) or {}).get("stock"):
            lot = None
        rejoinders = self.access_browses(waitlist, utter, lot, browser)
        if lot and rejoinders and not drive.batch_proven:
            if drive.shelf.lot_sound(self.ultimate_browses or [], waitlist):
                drive.batch_proven = True
            else:
                # the batch tool did not deliver usable records: forget it and open a few one by one
                drive.batch_spec = None
                drive.lot_brief_recognized = True
                drive.diary.append({"kind": "batch_dropped", "tool": lot[0]})
                maximal = {"menu": 0, "quota": 2, "walk": 2}.get(purpose, 3)
                if be_numeral(floor):
                    keep = int(floor)
                elif purpose in ("menu", "quota"):
                    keep = drive.admission.checkout_expected()
                else:
                    keep = 3
                count = min(maximal, len(waitlist), drive.maneuvers_residual() - keep - 1)
                if count > 0 and not channel.done and (browser or drive.role("details")):
                    drive.speak.subsequent_rejoinders(rejoinders)
                    rejoinders = self.access_browses(waitlist[:count], utter, None, browser)
                    if not rejoinders:
                        return True
        if not rejoinders:
            return False
        note = self.evaluation_reject() if channel.extra_alt() > SEALING + 30 and not revalidate else ""
        if note:
            cuid, body = rejoinders[-1]
            rejoinders[-1] = (cuid, body + "\n[check after opening] " + note)
        drive.speak.subsequent_rejoinders(rejoinders)
        return True

    @staticmethod
    def hashmaps_amidst(fragments) -> list[dict[str, Any]]:
        discovered: list[dict[str, Any]] = []
        for shard in fragments:
            route_hashmaps(shard, discovered.append)
        return discovered

    def ultimate_supply(self, lbl: str) -> bool | None:
        """The newest known stock state of a listing, from every record that names it."""
        observe = self.drive.observe
        situation = None
        for z in self.hashmaps_amidst(self.drive.stash.items):
            if observe.datarow_stamp(z) != lbl:
                continue
            obtained = observe.supply_situation(z)
            if obtained is not None:
                situation = obtained
        return situation

    def _grade_contender_sound(self, xq: str, edition: str, cap) -> bool:
        drive = self.drive
        if xq not in drive.stash.read or f"{xq}|{edition}" in drive.passed or xq in drive.refused_lines:
            return False
        if xq in drive.shelf.inert_orderlines() or self.ultimate_supply(xq) is False or self.column_glance(xq)["fails"]:
            return False
        ofs = drive.walk.commodity_offset(xq) if drive.needs.items() else None
        if ofs is not None and ofs < len(drive.needs.items()):
            amt = drive.needs.items()[ofs]["quantity"]
        else:
            amt = drive.needs.quantity()
        price = drive.stash.price(xq)
        if not be_numeral(price):
            return False
        if be_numeral(cap) and be_numeral(amt) and price * amt > cap:
            return False
        return True

    def grade_authenticate(self, entity: str) -> str:
        """Run the full check on up to two of the ranker's cheaper candidates, then report a cheaper verified fit."""
        drive = self.drive
        try:
            if drive.needs.priority() != "cheapest":
                return ""
            edition = str(drive.needs.edition)
            cap = drive.needs.disburse_top
            done = 0
            for xq in self.grade_superior or []:
                if done >= 2 or drive.channel.extra_alt() < SEALING + 30:
                    break
                if not self._grade_contender_sound(xq, edition, cap):
                    continue
                done += 1
                self.validate_article(xq, "ORDER")
            return self.thriftier_befit(entity)
        except Exception as failure:
            drive.diary.append({"kind": "rank_verify_error", "error": f"{type(failure).__name__}: {failure}"[:300]})
            return ""

    def thriftier_befit(self, entity: str) -> str:
        """The cheapest listing that passed the check at this version and undercuts thing ("" if none)."""
        drive = self.drive
        if drive.needs.priority() != "cheapest" or not entity:
            return ""
        ceiling = drive.stash.price(entity)
        if ceiling is None:
            return ""
        label = self.aisle_recognized(entity)
        gone = drive.shelf.inert_orderlines()
        edition = str(drive.needs.edition)
        optimal_ident, optimal_expense = "", None
        for key in drive.passed:
            lbl, _, ver = key.rpartition("|")
            if ver != edition or lbl == entity or lbl in drive.refused_lines or lbl in gone:
                continue
            if drive.needs.items() and (not label or self.aisle_recognized(lbl) != label):
                continue
            cost = drive.stash.price(lbl)
            if cost is None or cost >= ceiling:
                continue
            if optimal_expense is not None and cost >= optimal_expense:
                continue
            if self.column_glance(lbl)["fails"]:
                continue
            optimal_ident, optimal_expense = lbl, cost
        return optimal_ident

    def aisle_recognized(self, entity: str) -> str:
        """The shelf label of a listing, from records naming it by an id field, else from the archive's labels."""
        observe = self.drive.observe
        label = ""
        for z in self.hashmaps_amidst(self.drive.stash.items):
            designations_itr = next((key for key, vlu in z.items()
                             if isinstance(vlu, str) and vlu == entity and "id" in str(key).lower()), None)
            if designations_itr is not None:
                label = observe.aisle_relating(z) or label
        if label:
            return label
        return str((self.drive.stash.caption_relating or {}).get(entity) or "").strip().lower()

    def preferred(self) -> str | None:
        metrics = self.drive.wagon.metrics
        if not metrics:
            return None
        return list(metrics)[-1]


class Directory:
    def __init__(self, drive):
        self.drive = drive
        # cart bookkeeping
        self.buggy_rev = None
        self.buggy_exhibited = -1
        self.primary_deposits = set()
        # stock knowledge, per listing and per source
        self.supply_noticed = None
        self.supply_through = {}
        self.dead = set()
        self.inert_through = {}
        self.datarows_ongoing = None
        self.datarows_delay = None
        # what each tool's replies were seen to carry
        self.utility_tops = {}
        self.utility_selectors = {}
        self.comment_selectors = {}
        self.prose_selectors = {}
        self.browse_alongside = {}
        self.datarow_selectors = set()
        self.datarow_prose = {}
        self.function_conjecture = None
        # categories, joined labels and pooled candidates
        self.hub_jotting = None
        self.spliced_noticed = {}
        self.dx_jotting = {}
        # sold-out replacement hints
        self.switch_folio = None
        self.switch_unbrowsed = None
        self.mend_tested = set()
        # wording-review state
        self.nigh_buggy = None
        self.nigh_accomplished = None
        self.nigh_sound = set()
        self.nigh_uttered = {}
        # pre-reading and budget slack
        self.prior_selected = None
        self.prior_browse_accomplished = None
        self.prior_bouts = 0
        self.cushion_overall = None

    def _captions_noticed(self) -> set[str]:
        stash, observe = self.drive.stash, self.drive.observe

        def cits_recognized_ident(rcd):
            return any(isinstance(vlu, str) and vlu in stash.ids and "id" in str(field).lower()
                       for field, vlu in rcd.items())
        shelves: set[str] = set()
        for rcd in Referee.hashmaps_amidst(stash.items):
            shelf = observe.aisle_relating(rcd)
            if shelf and cits_recognized_ident(rcd):
                shelves.add(shelf)
        return shelves

    @staticmethod
    def _optimal_captions(kinds: list[set], captions: set[str]) -> list[set[str]]:
        # For each product, the shelf labels whose kind words score highest against it (ties kept together).
        sequenced = sorted(captions)
        result = []
        for ind, minee in enumerate(kinds):
            competitors = set()
            for further_ind, further in enumerate(kinds):
                if further_ind != ind:
                    competitors |= further
            frontrunners, pinnacle = set(), 0.0
            for shelf in sequenced:
                pointvals = variety_merit(minee, set(variety_morphemes(shelf)), competitors)
                if pointvals > pinnacle:
                    frontrunners, pinnacle = {shelf}, pointvals
                elif pointvals > 0 and pointvals == pinnacle:
                    frontrunners.add(shelf)
            result.append(frontrunners)
        return result

    def hub_aisles(self) -> list[set[str]]:
        drive = self.drive
        jotting = self.hub_jotting
        if jotting and jotting[0] == len(drive.stash.items):
            return [set(group) for group in jotting[1]]
        try:
            wished_alt = drive.needs.items() or [{"product": str(drive.needs.browsing.get("product") or "")}]
            noticed_captions = self._captions_noticed()
            houses = self._optimal_captions([set(variety_morphemes(element_alt["product"])) for element_alt in wished_alt], noticed_captions)
        except Exception:
            houses = [set() for _ in (drive.needs.items() or [0])]
        self.hub_jotting = (len(drive.stash.items), [set(group) for group in houses])
        return houses

    def personal_aisle(self) -> str:
        # Single-product requests only: the home category, when the shelves read so far are not all inside it.
        try:
            if self.drive.needs.items():
                return ""
            hub = self.hub_aisles()[0]
            toured = [name.strip().lower() for name in self.drive.stash.aisles_browse()]
            if not hub or all(name in hub for name in toured):
                return ""
            hit = next((name for name in toured if name in hub), None)
            return sorted(hub)[0] if hit is None else hit
        except Exception:
            return ""

    @staticmethod
    def _caption_amidst_checkout(text: str, label: str):
        components = [dab for dab in re.split(SPLICERS, str(label).lower()) if dab.strip()]
        if len(components) != 2:
            return None
        first, second = designation_design(components[0]), designation_design(components[1])
        if not (first and second):
            return None
        return re.search(r"\b" + first + r"\s*(?:\band\b|&|\bor\b|/|\+)\s*" + second, text)

    @staticmethod
    def _lax_splice(text: str, xq: str, y: str, a: str, g_alt: str):
        # x ... y joined by a short connective, with neither a nor b mentioned anywhere else in the text.
        qx, qy = designation_design(xq), designation_design(y)
        if not (qx and qy):
            return None
        for mz in re.finditer(r"\b(?:" + qx + r")(?P<mid>[^.;:!?]{0,40}?)(?:" + qy + r")", text):
            amid = mz.group("mid")
            if len(re.findall(r"[a-z0-9]+", amid)) > 3:
                continue
            if not re.search(r"\band\b|&|\bor\b|,|/|\+", amid) or re.search(r"\d|\bx\b", amid):
                continue
            otherwhere = text[:mz.start()] + " | " + text[mz.end():]
            if all(re.search(r"\b" + designation_design(vocable), otherwhere) for vocable in (a, g_alt)):
                continue
            return mz
        return None

    def spliced_locution(self, a: str, g_alt: str, label: str) -> tuple[str, bool] | None:
        text = re.sub(r"\s+", " ", self.drive.request.lower().replace("’", "'"))
        precise = self._caption_amidst_checkout(text, label)
        if precise:
            return precise.group(0).strip(), True
        morphemes_u, morphemes_g = variety_morphemes(a), variety_morphemes(g_alt)
        if not morphemes_u or not morphemes_g or equal_variety(morphemes_u[-1], morphemes_g[-1]):
            return None
        for first, second in ((a, g_alt), (g_alt, a)):
            mz = self._lax_splice(text, first, second, a, g_alt)
            if mz:
                return mz.group(0).strip(), False
        return None

    def commodity_inquiry(self, line: str, rows: list[dict[str, Any]]) -> bool:
        # Was this logged question aimed at one of the products (by its tag or by naming it)?
        dissected = re.match(r'-\s*(.*?): asked "(.*?)" -> ', str(line), re.S)
        label, question = (dissected.group(1), dissected.group(2)) if dissected else (str(line), str(line))
        datarow_varieties = [variety_morphemes(element_alt["product"]) for element_alt in rows]
        for tag in re.findall(r"\(([^()]*)\)", label):
            badge_vocables = variety_morphemes(tag)
            if badge_vocables and any(variety_amidst(vocable, group) for vocable in badge_vocables for group in datarow_varieties):
                return True
        solicited = question.lower().replace("’", "'")
        for element_alt in rows:
            ptn = designation_design(element_alt["product"])
            if ptn and re.search(r"\b" + ptn, solicited):
                return True
        return False

    @staticmethod
    def spliced_collision(a: dict[str, Any], g_alt: dict[str, Any]) -> bool:
        def feat_measurements(row: dict[str, Any]) -> list[tuple[str, str]]:
            out = []
            for need in row.get("requirements") or []:
                name, _, vlu = str(need).partition(":")
                out.append((re.split(r"\s*\(", name)[0].strip(), " ".join(re.findall(r"[a-z0-9]+", vlu.lower()))))
            return out
        they = feat_measurements(g_alt)
        return any(equal_feat(pz_u, pz_g) and vz_u != vz_g for pz_u, vz_u in feat_measurements(a) for pz_g, vz_g in they)

    def _comment_spliced_captions(self, obv: Any) -> None:
        def transports_ident(rcd):
            return any(isinstance(vlu, str) and "id" in str(field).lower() and re.fullmatch(IDENT_AKIN, vlu)
                       for field, vlu in rcd.items())

        def genre_prose(rcd):
            for field, vlu in rcd.items():
                if not isinstance(vlu, str) or not 0 < len(vlu.strip()) <= 60:
                    continue
                minor = str(field).lower()
                if re.search(r"categor|shelf|department|aisle", minor) and "id" not in minor:
                    return vlu.strip()
            return ""
        for rcd in Referee.hashmaps_amidst([obv]):
            if not transports_ident(rcd):
                continue
            name = genre_prose(rcd)
            if name and len([dab for dab in re.split(SPLICERS, name.lower()) if dab.strip()]) == 2:
                self.spliced_noticed.setdefault(name.lower(), name)

    @staticmethod
    def _facets_divide(tn, tk, facets) -> bool:
        surplus_ii = [wq for wq in tn if not variety_amidst(wq, tk)]
        surplus_s = [wq for wq in tk if not variety_amidst(wq, tn)]

        def restricted(words, interior, exterior):
            return all(variety_amidst(wq, interior) and not variety_amidst(wq, exterior) for wq in words)
        for left, correct in ((facets[0], facets[1]), (facets[1], facets[0])):
            if (restricted(surplus_ii, left, correct) and restricted(surplus_s, correct, left)
                    and any(variety_amidst(wq, left) for wq in tn) and any(variety_amidst(wq, correct) for wq in tk)):
                return True
        return False

    def _spliced_duo(self, rows):
        # First (by sorted label) joined category that two same-quantity products split between them.
        seen = self.spliced_noticed
        pairs = [(ii, s_alt) for ii in range(len(rows)) for s_alt in range(ii + 1, len(rows))]
        for key in sorted(seen):
            facets = [variety_morphemes(dab) for dab in re.split(SPLICERS, key) if dab.strip()]
            if len(facets) != 2 or not all(facets):
                continue
            dual = facets[0] + facets[1]
            for ii, s_alt in pairs:
                first, second = rows[ii], rows[s_alt]
                tn, tk = variety_morphemes(first["product"]), variety_morphemes(second["product"])
                if not (tn and tk):
                    continue
                if varieties_amongst(tn, tk) and varieties_amongst(tk, tn):
                    continue
                if not varieties_amongst(tn + tk, dual) or first["quantity"] != second["quantity"]:
                    continue
                if not self._facets_divide(tn, tk, facets) or self.spliced_collision(first, second):
                    continue
                phrase = self.spliced_locution(first["product"], second["product"], seen[key])
                if phrase:
                    return ii, s_alt, seen[key], phrase[0], phrase[1]
        return None

    def merge_joined(self, obv: Any) -> str:
        drive = self.drive
        self._comment_spliced_captions(obv)
        uttered: list[str] = []
        for _ in range(2):
            rows = drive.needs.items()
            halt_alt = (len(rows) < 2 or not self.spliced_noticed or drive.wagon.utilized or drive.wagon.metrics
                    or drive.needs.uncovered or drive.needs.indifferent
                    or any(self.commodity_inquiry(element_alt, rows) for element_alt in drive.questions_log))
            if halt_alt:
                break
            duo = self._spliced_duo(rows)
            if not duo:
                break
            uttered.append(self.blend_articles(*duo))
        return " ".join(filter(None, uttered))

    def _blended_prerequisites(self, first: dict[str, Any], second: dict[str, Any]) -> list[str]:
        out: list[str] = []
        selectors: set[str] = set()
        for need in [*first["requirements"], *second["requirements"]]:
            name, _, remainder = str(need).partition(":")
            tidy = re.sub(r"\s*\([^()]*\)", "", name).strip() + ":" + remainder
            minor = tidy.lower()
            if minor not in selectors:
                selectors.add(minor)
                out.append(tidy)
        return out

    def _jettison_commodity_stashes(self) -> None:
        drive = self.drive
        drive.walk.routed = set()
        drive.speak.ration_browses = {}
        drive.admission.panel_jotting = {}
        self.hub_jotting = None

    def blend_articles(self, ii: int, s_alt: int, label: str, phrase: str, amidst_checkout: bool = True) -> str:
        drive = self.drive
        needs = drive.needs
        tome = needs.browsing
        products = needs.items()
        itemized = [element_alt for element_alt in tome.get("items") or []
                  if isinstance(element_alt, dict) and str(element_alt.get("product") or "").strip()]
        if len(itemized) != len(products) or not (0 <= ii < s_alt < len(products)):
            return ""
        duo_varieties = variety_morphemes(products[ii]["product"]) + variety_morphemes(products[s_alt]["product"])
        caption_varieties = variety_morphemes(label)
        remainder_varieties = [variety_morphemes(element_alt["product"]) for at, element_alt in enumerate(products) if at not in (ii, s_alt)]
        unspent = len(products) - 1
        novel_badge = phrase if unspent >= 2 else ""

        def pertains(tag: Any) -> bool:
            # The tag names one of the merged pair (within the label) and no surviving product.
            words = variety_morphemes(tag)
            if not words or not varieties_amongst(words, caption_varieties):
                return False
            if not any(variety_amidst(wq, duo_varieties) for wq in words):
                return False
            return not any(varieties_amongst(words, further) and varieties_amongst(further, words) for further in remainder_varieties)

        def switch(mz: Any) -> str:
            if not pertains(mz.group(1)):
                return mz.group(0)
            return f" ({novel_badge})" if novel_badge else ""

        def relabel(text: Any) -> Any:
            if not isinstance(text, str):
                return text
            head, punct, rear = text.partition(":")
            return re.sub(r"\s*\(([^()]*)\)", switch, head) + punct + rear

        def relabel_lineup(values: list[Any]) -> list[Any]:
            sole: list[Any] = []
            for value in map(relabel, values):
                if value not in sole:
                    sole.append(value)
            return sole
        spliced = {"product": phrase, "quantity": products[ii]["quantity"],
                  "requirements": self._blended_prerequisites(products[ii], products[s_alt])}
        remnants = [spliced if at == ii else element_alt for at, element_alt in enumerate(itemized) if at != s_alt]
        for trait in ("written", "uncovered", "let_go", "indifferent"):
            setattr(needs, trait, relabel_lineup(getattr(needs, trait)))
        if needs.presently_dems is not None:
            needs.presently_dems = relabel_lineup(needs.presently_dems)
        needs.resolved = list(map(relabel, needs.resolved or []))
        for field in ("requirements", "hints", "references", "soft_wants"):
            if isinstance(tome.get(field), list):
                tome[field] = relabel_lineup(tome[field])
        goals = tome.get("reference_targets")
        if isinstance(goals, list):
            for target in goals:
                if isinstance(target, dict) and pertains(str(target.get("product") or "")):
                    target["product"] = phrase
        for cite in drive.refs or []:
            for field in ("req", "head", "now"):
                if isinstance(cite.get(field), str):
                    cite[field] = relabel(cite[field])
            if pertains(str(cite.get("tag") or "")):
                cite["tag"] = novel_badge
        speak = drive.speak
        if isinstance(speak.awaiting_runnersup, list):
            speak.awaiting_runnersup = list(map(relabel, speak.awaiting_runnersup))
        if isinstance(speak.access_duo, list):
            speak.access_duo = list(map(relabel, speak.access_duo))
        if unspent >= 2:
            tome["items"] = remnants
            tome["product"] = " and ".join(str(element_alt.get("product")) for element_alt in remnants)
        else:
            tome.pop("items", None)
            tome["product"] = phrase
            tome["quantity"] = products[ii]["quantity"]
        needs.edition += 1
        self._jettison_commodity_stashes()
        drive.channel.factor = 1.0 + 0.3 * min(2, max(0, len(needs.items()) - 1))
        drive.diary.append({"kind": "merge_joined", "label": label, "products": [products[ii]["product"], products[s_alt]["product"]],
                        "into": phrase, "label_words": amidst_checkout, "products_left": max(1, len(needs.items()))})
        orderlines_component = "needs just one line for it" if unspent >= 2 else "needs ONE line"
        return (f"This store files '{label}' as a SINGLE category, and the request asks for one product from it ('{phrase}'); "
                "the order therefore " + orderlines_component + " rather than a separate line for every kind the category name "
                "combines. Current requirements: " + needs.prose_corpus())

    def fetch_selector(self, tool: str | None) -> str | None:
        # The single free-text argument a by-id lookup tool takes, if its schema has exactly that shape.
        brief = self.drive.contours.get(tool or "") or {}
        props = brief.get("properties")
        if not isinstance(props, dict):
            props = {}
        req = brief.get("required")
        shall = [name for name in req or [] if name in props] if isinstance(req, list) else []
        textstrings = [name for name, pz in props.items() if isinstance(pz, dict) and pz.get("type") == "string" and not pz.get("enum")]
        key = next((name for name in textstrings if name in shall), None)
        if not key and len(textstrings) == 1:
            key = textstrings[0]
        if not key or any(name != key for name in shall):
            return None
        return key

    def fetch_utilities(self) -> list[str]:
        drive = self.drive
        occupied = {drive.role(rq) for rq in ("search", "message", "cart", "cart_view", "order", "decline")}
        occupied.discard(None)
        return [name for name in drive.contours if name not in occupied and self.fetch_selector(name)]

    def fetch_trigger(self, lbl: str, tool: str | None) -> dict[str, Any] | None:
        drive = self.drive
        if not tool:
            return None
        if tool == drive.role("details"):
            return drive.referee.nuances_trigger(lbl)
        known = drive.stash.ids

        def be_recognized(vlu):
            return isinstance(vlu, str) and vlu in known
        # Reuse the shape of the latest call to this tool that named a known listing, swapping in the new id.
        for element_alt in reversed(drive.diary):
            if element_alt.get("kind") != "env":
                continue
            for operate in element_alt.get("actions") or []:
                if operate.get("name") != tool:
                    continue
                args = operate.get("args")
                if isinstance(args, dict) and any(be_recognized(vlu) for vlu in args.values()):
                    return {name: (lbl if be_recognized(vlu) else vlu) for name, vlu in args.items()}
        key = self.fetch_selector(tool)
        return {key: lbl} if key else None

    @staticmethod
    def rcd_ident(z: dict[str, Any]) -> str | None:
        # Prefer a field called exactly "id"; otherwise accept any field whose name ends in "id".
        for stringent in (True, False):
            for field, vlu in z.items():
                minor = str(field).lower()
                if not (minor == "id" if stringent else minor.endswith("id")):
                    continue
                if isinstance(vlu, str) and re.fullmatch(IDENT_AKIN, vlu):
                    return vlu
        return None

    @staticmethod
    def solicited_idents(trigger_operands: Any) -> set[str]:
        args = trigger_operands if isinstance(trigger_operands, dict) else {}

        def glances_akin_ident(vlu):
            return isinstance(vlu, str) and re.fullmatch(IDENT_AKIN, vlu)
        out = {vlu for vlu in args.values() if glances_akin_ident(vlu)}
        for vlu in args.values():
            if isinstance(vlu, list):
                out.update(xq for xq in vlu if glances_akin_ident(xq))
        return out

    def _tome_supply(self, lbl: str, situation: bool, provenance: str, era: int, freshly: list[str]) -> None:
        self.supply_noticed = True
        wake = self.supply_through.setdefault(lbl, {})
        duplicate = wake.get("row" if provenance == "lookup" else "lookup")
        if duplicate and duplicate[0] == era and duplicate[1] != situation:
            self.datarows_delay = True
        antecedent_datarow = wake.get("row")
        if provenance == "row" and antecedent_datarow and antecedent_datarow[1] != situation:
            self.datarows_ongoing = True
        wake[provenance] = (era, situation)
        if situation is False:
            if lbl not in self.dead and lbl in self.drive.wagon.metrics and lbl not in freshly:
                freshly.append(lbl)
            self.dead.add(lbl)
            self.inert_through[lbl] = provenance
            return
        # A row listing may not revive a line that a direct lookup declared gone.
        if provenance == "lookup" or self.inert_through.get(lbl) != "lookup":
            self.dead.discard(lbl)

    @staticmethod
    def _possesses_characteristic_atlas(transcript: dict[str, Any]) -> bool:
        return any(isinstance(vlu, dict) and sum(isinstance(xq, (str, int, float)) for xq in vlu.values()) >= 2
                   for vlu in transcript.values())

    def glean_tops(self, trigger_designation: str, trigger_operands: Any, obv: Any) -> list[str]:
        drive = self.drive
        solicited = self.solicited_idents(trigger_operands)
        tool = str(trigger_designation)
        reckoning = self.utility_tops.setdefault(tool, {"calls": 0, "id": 0, "stock": 0, "props": 0, "notes": 0, "price": 0})
        reckoning["calls"] += 1
        flags = dict.fromkeys(("id", "stock", "props", "notes", "price"), False)
        exhibited_selectors = self.utility_selectors.setdefault(tool, set())
        extended_selectors = self.comment_selectors.setdefault(tool, set())
        buggy_flank = {drive.role("cart"), drive.role("cart_view")} - {None}
        era = drive.cart_seq
        freshly: list[str] = []
        observe = drive.observe
        transcripts = [] if isinstance(obv, dict) and obv.get("error") else Referee.hashmaps_amidst([obv])
        for rcd in transcripts:
            lbl = self.rcd_ident(rcd)
            if not lbl:
                continue
            situation = observe.supply_situation(rcd)
            straight = lbl in solicited or trigger_designation in buggy_flank
            if situation is not None:
                self._tome_supply(lbl, situation, "lookup" if straight else "row", era, freshly)
            if lbl not in solicited:
                # A listing row the call did not ask about: remember its plain fields for later comparison.
                if trigger_designation not in buggy_flank:
                    self.datarow_selectors.update(str(field) for field in rcd)
                    self.datarow_prose.setdefault(lbl, {}).update({str(field): vlu for field, vlu in rcd.items() if isinstance(vlu, str)})
                continue
            flags["id"] = True
            self.browse_alongside.setdefault(tool, set()).add(lbl)
            self.prose_selectors.setdefault(tool, set()).update(str(field) for field, vlu in rcd.items() if vlu is None or isinstance(vlu, str))
            if situation is not None:
                flags["stock"] = True
            if not flags["props"]:
                flags["props"] = self._possesses_characteristic_atlas(rcd)
            if not flags["price"]:
                flags["price"] = any(be_numeral(vlu) and re.search(r"price|cost", str(field).lower()) for field, vlu in rcd.items())
            exhibited_selectors.update(str(field) for field in rcd)
            sooner = self.datarow_prose.get(lbl)
            for field, vlu in rcd.items():
                if not isinstance(vlu, str) or len(vlu.split()) < 8:
                    continue
                original = str(field) not in self.datarow_selectors if sooner is None else sooner.get(str(field)) != vlu
                if original:
                    extended_selectors.add(str(field))
                    flags["notes"] = True
        for name, toggle in flags.items():
            reckoning[name] += int(toggle)
        return freshly

    def supply_utility(self) -> str | None:
        drive = self.drive
        tops = self.utility_tops or {}
        workable = self.fetch_utilities()

        def top_relating(name, field):
            return (tops.get(name or "") or {}).get(field, 0)
        relaying = [name for name in workable if top_relating(name, "stock") > 0]
        if relaying:
            return max(relaying, key=lambda name: (tops[name]["stock"], name == drive.role("view"), name == drive.role("details"), name))
        if self.function_conjecture is None:
            try:
                self.function_conjecture = drive.sealer.routine_functions()
            except Exception:
                self.function_conjecture = {}
        conjecture = self.function_conjecture
        for name in (drive.role("view"), conjecture.get("view"), drive.role("details")):
            if not name or name not in workable:
                continue
            # Skip a tool that answered by id at least twice without ever showing stock.
            if top_relating(name, "id") >= 2 and top_relating(name, "stock") == 0:
                continue
            return name
        occupied = {conjecture.get(rq) for rq in ("cart", "cart_view", "order", "decline", "message")} - {None}
        return next((name for name in workable if name not in occupied and not top_relating(name, "id")), None)

    def comments_utility(self) -> str | None:
        drive = self.drive
        tops = self.utility_tops or {}
        details = drive.role("details")
        footing_selectors = set((self.utility_selectors or {}).get(details or "", set()))
        notes = self.comment_selectors or {}
        surplus = [name for name in self.fetch_utilities() if name != details and set(notes.get(name) or set()) - footing_selectors]
        if not surplus:
            return None
        return max(surplus, key=lambda name: ((tops.get(name) or {}).get("notes", 0), name))

    def inert_orderlines(self) -> set[str]:
        return set(self.dead or ())

    def obsolete_orderlines(self) -> list[str]:
        drive = self.drive
        srs, starting = drive.line_seq or {}, drive.cart_seq
        return [line for line in drive.wagon.metrics if srs.get(line, -1) < starting]

    def _cost_either_huge(self, lbl: str) -> float:
        cited = self.drive.stash.price(lbl)
        return float(cited) if be_numeral(cited) else float("inf")

    def _authenticated_switch(self, line, gone, edition, label, several, cap, amt, remainder):
        drive = self.drive
        referee = drive.referee
        omit = {line, *gone, *drive.refused_lines, *drive.wagon.metrics}
        discovered = []
        for seal in drive.passed:
            lbl, _, ver = seal.rpartition("|")
            if ver != edition or lbl in omit or referee.column_glance(lbl)["fails"]:
                continue
            shelf = referee.aisle_recognized(lbl)
            if label:
                displaced = shelf != label if several else shelf not in ("", label)
                if displaced:
                    continue
            price = self._cost_either_huge(lbl)
            if be_numeral(cap) and price != float("inf") and price * amt + remainder > cap:
                continue
            discovered.append(lbl)
        return discovered

    def _switch_datarows(self, line, gone, edition, label, slot, cap, amt, remainder):
        drive = self.drive
        sealer = drive.sealer
        wished_alt = [(prop, value) for prop, value in sealer.routine_measurements(slot) if self.dx_bare(value)]
        adjudged_outward = {seal.split("|")[0] for seal, verdict in drive.rulings.items()
                     if verdict and seal.split("|")[1:2] == [edition]}
        arrows = drive.citecounter.cite_arrows()
        omit = {line, *gone, *drive.refused_lines, *drive.wagon.metrics}
        inf = float("inf")

        def browse_plus_robust(lbl, row):
            if drive.referee.column_glance(lbl)["fails"]:
                return False
            if any(sealer.routine_collision(row["fields"], prop, value) for prop, value in wished_alt):
                return False
            return all(sealer.routine_exhibits(row, prop, value, set(), True) for prop, value in wished_alt)
        ranked = []
        for lbl, row in sealer.routine_datarows().items():
            price = row["price"] if be_numeral(row["price"]) else self._cost_either_huge(lbl)
            if lbl in omit or row["stock"] is not True:
                continue
            if (label and row["shelf"] != label) or price == inf:
                continue
            if be_numeral(cap) and price * amt + remainder > cap:
                continue
            if lbl in adjudged_outward or lbl in arrows or drive.citecounter.cite_precluded(row["fields"], slot):
                continue
            opened = lbl in drive.stash.read
            if opened and not browse_plus_robust(lbl, row):
                continue
            ranked.append((0 if opened else 1, price, lbl))
        return sorted(ranked)

    def dead_swap(self, line: str) -> tuple[str, str]:
        drive = self.drive
        gone = self.inert_orderlines()
        edition = str(drive.needs.edition)
        label = drive.referee.aisle_recognized(line)
        several = len(drive.needs.items()) >= 2
        cap = drive.needs.disburse_top
        amt = max(1.0, drive.wagon.maintained(line))
        cost = self._cost_either_huge
        remainder = sum(cost(further) * drive.wagon.maintained(further) for further in drive.wagon.metrics
                   if further != line and cost(further) != float("inf"))
        verified = self._authenticated_switch(line, gone, edition, label, several, cap, amt, remainder)
        if verified:
            return min(verified, key=lambda lbl: (cost(lbl), lbl)), ""
        if several and not label:
            return "", ""
        slot = None
        if several:
            slot = next((n for n, hub in enumerate(self.hub_aisles()) if label and label in hub), None)
        ranked = self._switch_datarows(line, gone, edition, label, slot, cap, amt, remainder)
        unbrowsed = [element_alt for element_alt in ranked if element_alt[0] == 1]
        thriftiest_unbrowsed = min(unbrowsed, key=lambda element_alt: (element_alt[1], element_alt[2])) if unbrowsed else None
        # When the cheapest is wanted, an unread row that undercuts every read one goes first.
        if (thriftiest_unbrowsed and drive.needs.priority() == "cheapest"
                and all(thriftiest_unbrowsed[1] < price for rank, price, _ in ranked if rank == 0)):
            ranked.remove(thriftiest_unbrowsed)
            ranked.insert(0, thriftiest_unbrowsed)
        apex = ranked[:3]
        self.switch_unbrowsed = [lbl for rank, _, lbl in apex if rank == 1]
        self.switch_folio = None
        if not any(rank == 0 for rank, _, _ in ranked):
            try:
                self.switch_folio = self.inert_folio(slot)
            except Exception:
                self.switch_folio = None

        def depict(rank, price, lbl):
            exhibited = int(price) if float(price).is_integer() else price
            surplus = ", already read: every stated value appears in its record" if rank == 0 else ""
            return f"{lbl} ({exhibited}{surplus})"
        return "", ", ".join(depict(*element_alt) for element_alt in apex)

    def inert_folio(self, kx: int | None) -> dict[str, Any] | None:
        drive = self.drive
        folio_selector = drive.observe.folio_operand()
        if not folio_selector:
            return None
        traverser, observe = drive.walk, drive.observe
        access_routes = [wq for wq in drive.page_book.values()
                      if traverser.route_transports(wq["args"], kx) and observe.ensuing_folio(wq) is not None]
        if not access_routes:
            return None
        best = max(access_routes, key=lambda wq: (len(traverser.route_transports(wq["args"], kx)), -len(wq["read"])))
        return {**best["args"], folio_selector[0]: observe.ensuing_folio(best)}

    def switch_access(self) -> str:
        drive = self.drive
        if drive.admission.switch_accomplished:
            return ""
        gone = self.inert_orderlines()
        return next((lbl for lbl in self.switch_unbrowsed or []
                     if lbl not in drive.stash.read and lbl not in gone and lbl not in drive.wagon.metrics), "")

    def inert_comment(self, lines: list[str]) -> str:
        drive = self.drive
        drive.event_seen = True
        notes = []
        for line in lines:
            self.switch_folio, self.switch_unbrowsed = None, []
            select, rows = self.dead_swap(line)
            lead = (f"{line} is SOLD OUT according to its newest record, and any order that still holds it will fail. "
                    "Remove it from the cart")
            if select:
                cheapest = ", and it is the cheapest such listing" if drive.needs.priority() == "cheapest" else ""
                notes.append(f"{lead}, then put {select} in (the item check confirmed it meets every requirement currently in "
                             f"force{cheapest}), check its stock and price once more, then place the order.")
            elif rows:
                notes.append(f"{lead}; no other listing for that product has been verified, so turn to the cheapest in-stock "
                             f"rows of its category that fit the budget ({rows}; rows already read whose record lacks or "
                             "contradicts a stated requirement were left out, and an unread one needs its details opened "
                             "first), add the first one that fits, check its stock and price once more, then place the order.")
            else:
                notes.append(f"{lead} and look for another listing that fits.")
            if self.switch_folio and not select:
                notes.append("More rows of the same kind are on the next unread page of the search that leads with the stated "
                             "value: " + json.dumps(self.switch_folio, default=str) + ".")
        return " ".join(notes)

    def denial_rebrowse(self, obv: Any = None) -> str:
        try:
            heard, gone = self.denial_departed(obv)
        except Exception as failure:
            heard, gone = "", ""
            self.drive.diary.append({"kind": "refusal_gone_error", "error": f"{type(failure).__name__}: {failure}"[:300]})
        return heard + (gone or self.denial_waitlist())

    def denial_departed(self, obv: Any) -> tuple[str, str]:
        drive = self.drive
        wagon = drive.wagon
        maintained = [lbl for lbl in wagon.metrics if wagon.maintained(lbl) > 0]
        if obv is None or len(maintained) != 1 or drive.channel.muffle or drive.covering or drive.last_kind != "order":
            return "", ""
        proses = drive.encompass.buyer_orderlines(drive.role("order") or "", obv, True)
        if not proses:
            return "", ""
        needs = drive.needs

        def situation():
            return (needs.edition, len(needs.let_go), len(needs.uncovered), str(needs.disburse_top), "|".join(needs.feats_residual()))
        preceding = situation()
        note = needs.perceive(proses, None)
        afterward = situation()
        heard = (" " + note) if note else ""
        # Only treat the line as gone when hearing the refusal bumped the version and nothing else.
        if not (note and afterward[0] != preceding[0] and afterward[1:] == preceding[1:]):
            return heard, ""
        line = maintained[0]
        self.dead.add(line)
        self.inert_through[line] = "lookup"
        drive.diary.append({"kind": "refusal_dead", "line": line})
        return heard, (" The refusal says an item is no longer available, and the cart held just one line, so the code "
                       "counts " + line + " as sold out (without re-reading it): " + self.inert_comment([line]))

    def denial_waitlist(self) -> str:
        drive = self.drive
        tool = self.supply_utility()
        lines = list(drive.wagon.metrics)[:4]
        browses = drive.browses
        if not lines or not tool or drive.channel.muffle or drive.covering or browses.access_waitlist:
            return ""
        if drive.maneuvers_residual() < len(lines) + 3 or drive.channel.extra_alt() < SEALING:
            return ""
        browses.access_waitlist = lines
        browses.access_revalidate = True
        browses.access_intention = "cart"
        browses.access_utility = tool
        browses.access_utter = "The order was refused, so the cart lines get their stock checked again (one may have sold out)."
        return " Reading the stock of the cart lines for you now (" + tool + ")."

    def routine_referees_every(self) -> bool:
        # Can the code itself settle every open requirement (plain values plus resolved "differ" references)?
        drive = self.drive
        items = drive.needs.items()
        openings = range(len(items)) if items else [None]
        values = [value for kx in openings for _, value in drive.sealer.routine_measurements(kx)]
        bare_alt = [value for value in values if self.dx_bare(value)]
        concluded = self.vary_accomplished(None)
        judged = sum(1 for req in drive.needs.feats_residual() if any(drive.citecounter.cite_concord(cite, req) for cite in concluded))
        return len(bare_alt) + judged >= len(list(drive.needs.feats_residual()))

    def vary_accomplished(self, kx: int | None) -> list[dict[str, Any]]:
        drive = self.drive
        bureau = drive.citecounter
        rows, left = drive.needs.items(), drive.needs.feats_residual()

        def workable(cite):
            if cite.get("relation") != "differ" or cite.get("status") != "resolved":
                return False
            if not str(cite.get("value") or "").strip() or not bureau.cite_ongoing(cite):
                return False
            if not any(bureau.cite_concord(cite, req) for req in left):
                return False
            if rows and kx is not None and cite.get("tag") and drive.speak.commodity_relating(f"x ({cite['tag']})") not in (kx, None):
                return False
            return True
        return [cite for cite in drive.refs or [] if workable(cite)]

    def _mend_befits(self, picks, personal: bool, edition: str) -> bool:
        drive = self.drive
        fit = drive.sealer.routine_befit or {}
        if not picks:
            return False

        def reliable(lbl):
            return f"{lbl}|{edition}" in drive.passed or (personal and lbl in drive.stash.read and fit.get(lbl))
        return all(reliable(lbl) for lbl, _ in picks)

    def _mend_selects(self, several: bool, gone: set, personal: bool, edition: str):
        drive = self.drive
        sealer = drive.sealer
        evade = set(gone)
        cluster: dict[str, dict[str, Any]] = {}
        if several:
            cluster = {y_alt["id"]: y_alt for kx in range(len(drive.needs.items())) for y_alt in self.dx_cluster(kx)}
            evade |= {lbl for lbl, y_alt in cluster.items() if y_alt["clash"]}

        def unstable(y_alt):
            return not y_alt["revealed_ok"] or y_alt["misses"]
        picks = sealer.routine_selects(evade)
        probes = bouts = 0
        while several and picks and probes < 3 and bouts < 30 and drive.channel.extra_alt() > SEALING + 30:
            unvetted = [lbl for lbl, _ in picks if not self._mend_befits([(lbl, 1)], personal, edition)]
            if not unvetted:
                break
            naked = [lbl for lbl in unvetted if not cluster.get(lbl) or unstable(cluster[lbl])]
            if naked:
                # Drop the weak picks (and every unread, unpassed shaky listing) and pick again.
                evade.update(naked)
                evade.update(lbl for lbl, y_alt in cluster.items() if not y_alt["read"] and f"{lbl}|{edition}" not in drive.passed
                             and unstable(y_alt))
                bouts += 1
            else:
                probes += 1
                drive.referee.validate_article(unvetted[0], "ORDER")
            picks = sealer.routine_selects(evade)
        return picks

    def _mend_pane(self):
        drive = self.drive
        channel = drive.channel
        several = len(drive.needs.items()) >= 2
        early = several and drive.refusal_n == 0 and drive.maneuvers_residual() > drive.admission.checkout_expected()
        if not early and drive.refusal_n <= 0 and drive.maneuvers_residual() > drive.admission.checkout_expected():
            return None
        if (not drive.role("cart") or not drive.role("order") or drive.maneuvers_residual() < 3 or channel.extra_alt() <= SEALING
                or channel.muffle or drive.covering or channel.done):
            return None
        stamp_alt: Any = ("early", drive.refusal_n) if early else drive.refusal_n
        if stamp_alt in self.mend_tested:
            return None
        self.mend_tested.add(stamp_alt)
        return several

    def _attestation_allows_reject(self) -> bool:
        drive = self.drive
        attestation = []
        try:
            attestation = self.dx_attestation()
        except Exception as failure:
            drive.diary.append({"kind": "decline_proof_error", "error": f"{type(failure).__name__}: {failure}"[:300]})
        if attestation:
            drive.diary.append({"kind": "decline_proof", "products": attestation})
        return bool(attestation)

    def decline_repair(self) -> str:
        drive = self.drive
        several = self._mend_pane()
        if several is None or (several and self._attestation_allows_reject()):
            return ""
        gone = self.inert_orderlines()
        obsolete = set(self.obsolete_orderlines())
        living = {line for line in drive.wagon.metrics if line not in gone and line not in obsolete}
        conserved = set(drive.refused_lines)
        edition = str(drive.needs.edition)
        personal = self.routine_referees_every()
        done = False
        try:
            drive.refused_lines.difference_update(living)
            picks = self._mend_selects(several, gone, personal, edition)
            if not self._mend_befits(picks, personal, edition):
                drive.diary.append({"kind": "decline_repair", "picks": picks, "done": False})
            elif drive.maneuvers_residual() < len(picks) + 2:
                drive.diary.append({"kind": "decline_repair", "picks": picks, "done": False, "why": "steps"})
            else:
                done = self._mend_deposit(picks)
                drive.diary.append({"kind": "decline_repair", "picks": picks, "done": done, "cart": dict(drive.wagon.metrics)})
        except Exception as failure:
            done = False
            drive.diary.append({"kind": "decline_repair_error", "error": f"{type(failure).__name__}: {failure}"[:300]})
        finally:
            if not done:
                drive.refused_lines.clear()
                drive.refused_lines.update(conserved)
        if not done:
            return ""
        several = bool(drive.needs.items())
        return ("Not run yet (no step used). A decline is right only when no listing can work, but the records already hold "
                "a fitting candidate" + (" for each product requested" if several else "") + " (it passed the item check, "
                "or its opened record shows every stated value), so the code put " + ("them" if several else "it")
                + " into the cart and checked stock: " + drive.wagon.abstract() + " Place the order for this cart now; "
                "decline only if a record you have read shows one of these lines breaking a requirement currently in force.")

    def _mend_deposit(self, picks) -> bool:
        drive = self.drive
        channel = drive.channel
        antecedent = channel.muffle
        channel.muffle = True
        try:
            overlooked = drive.sealer.routine_deposit(picks)
            ok = overlooked == [] and all(lbl in drive.wagon.metrics for lbl, _ in picks)
            preferred = [lbl for lbl, _ in picks]
            obsolete_presently = [line for line in self.obsolete_orderlines() if line in preferred]
            if ok and obsolete_presently and drive.maneuvers_residual() >= len(obsolete_presently) + 2:
                _, vended = drive.sealer.routine_revalidate(obsolete_presently)
                ok = not vended
        finally:
            channel.muffle = antecedent
        return ok and bool(drive.wagon.metrics) and not set(drive.wagon.metrics) & self.inert_orderlines()

    @staticmethod
    def dx_bare(value: str) -> bool:
        # True when the value is a single exact one: no linking words, ranges, comparisons or approximations.
        cautious = r"\b(?:" + MEASUREMENT_ANCHORS + r")\b|\d\s*(?:-|to)\s*\d|[+<>]|\b(?:about|around|approx\w*|roughly)\b"
        return re.search(cautious, str(value).lower()) is None

    @staticmethod
    def dx_vocable(value: str) -> bool:
        minor = str(value).lower()
        if re.search(r"\d", minor) or not re.search(r"[a-z]{3,}", minor):
            return False
        return Directory.dx_bare(minor) and not re.search(PRECLUDE_VOCABLES, minor) and not cite_prose(minor)

    def dx_dems(self, kx: int | None) -> list[tuple[str, str, bool]]:
        drive = self.drive
        needs = drive.needs
        manifold = needs.items()
        out = []
        for req in needs.feats_residual():
            head, _, value = str(req).partition(":")
            if manifold and drive.speak.commodity_relating(head) not in (kx, None):
                continue
            prop = re.split(r"\s*\(", head)[0].strip()
            naked = re.sub(r"\((?:stated|revealed[^)]*)\)", "", value).strip()
            out.append((prop, naked, req not in needs.written))
        return out

    def _cluster_element(self, lbl, row, env_alt) -> dict[str, Any]:
        drive = self.drive
        sealer = drive.sealer
        wished_alt, combis, exhibited_necessities, cites_outward, dead, adjudged_idents, cleared_idents, cap, amt, slot = env_alt
        opened = lbl in drive.stash.read
        cited = row["price"] if row["price"] is not None else drive.stash.price(lbl)
        price = float(cited) if be_numeral(cited) else None
        gone = lbl in dead or row["stock"] is False or lbl in drive.refused_lines
        over = cap is not None and price is not None and price * amt > cap + 1e-9
        pointer = lbl in cites_outward or drive.citecounter.cite_precluded(row["fields"], slot)
        fields = row["fields"]

        def measurements_towards(prop):
            return [vlu for key, vlu in fields.items() if equal_feat(key, prop)]

        def shows(prop, value):
            return sealer.routine_exhibits(row, prop, value, set(), True)
        # A clash is a field that contradicts the wanted value and nowhere repeats it.
        clash = [prop for prop, value in wished_alt
                 if sealer.routine_collision(fields, prop, value) and not any(sealer.routine_noticed(value, vlu) for vlu in measurements_towards(prop))]
        for prop, components in combis:
            extant = measurements_towards(prop)
            if extant and not all(any(sealer.routine_noticed(component, vlu) for vlu in extant) for component in components):
                clash.append(prop)
        misses = [(prop, value) for prop, value in wished_alt if not shows(prop, value)]
        ruled = lbl in adjudged_idents or bool(opened and drive.referee.column_glance(lbl)["fails"])
        verified = lbl in cleared_idents and not gone
        fit = not (ruled or pointer or gone or over) and (verified or (opened and not misses and not clash))
        element_alt = {"id": lbl, "price": price, "read": opened, "verified": verified, "fit": fit, "gone": gone,
                 "over": over, "pointer": pointer, "ruled": ruled, "clash": clash, "misses": misses, "row": row}
        element_alt["out"] = bool(gone or over or pointer or ruled or clash)
        element_alt["revealed_ok"] = all(shows(prop, value) for prop, value in exhibited_necessities)
        return element_alt

    def _cluster_hub(self, kx, several) -> set:
        houses = self.hub_aisles()
        if several:
            return set(houses[kx]) if kx is not None and kx < len(houses) else set()
        return set(self.drive.observe.aisle_captions()) or (set(houses[0]) if houses else set())

    def dx_cluster(self, kx: int | None) -> list[dict[str, Any]]:
        drive = self.drive
        needs = drive.needs
        manifold = needs.items()
        edition = str(needs.edition)
        key = (kx, len(drive.stash.items), len(drive.stash.read), edition, "|".join(needs.feats_residual()), len(drive.passed),
               sum(1 for verdict in drive.rulings.values() if verdict), tuple(sorted(self.inert_orderlines())),
               tuple(sorted(drive.refused_lines)), str(needs.disburse_top), len(drive.refs or []))
        jotting = self.dx_jotting
        if key in jotting:
            return jotting[key]
        if len(jotting) > 64:
            jotting.clear()
        hub = self._cluster_hub(kx, manifold)
        amalgamated: list[dict[str, Any]] = []
        if hub:
            censuses = needs.censuses()
            if manifold and kx is not None and kx < len(censuses):
                amt = censuses[kx]
            else:
                amt = censuses[0] if censuses else 1
            adjudged_idents = {seal.split("|")[0] for seal, verdict in drive.rulings.items()
                         if verdict and seal.split("|")[1:2] == [edition]}
            cleared_idents = {seal.rpartition("|")[0] for seal in drive.passed if seal.rpartition("|")[2] == edition}
            cap = float(needs.disburse_top) if be_numeral(needs.disburse_top) else None
            slot = kx if manifold else None
            sealer = drive.sealer
            wished_alt = [(prop, value) for prop, value in sealer.routine_measurements(slot) if self.dx_bare(value)]
            combis = []
            for prop, value in sealer.routine_measurements(slot):
                if "+" not in value:
                    continue
                components = [dab.strip() for dab in value.split("+") if dab.strip()]
                if len(components) >= 2 and all(re.search(r"[a-z]{2,}", dab.lower()) for dab in components):
                    combis.append((prop, components))
            exhibited_necessities = [(prop, value) for prop, value, exhibited in self.dx_dems(slot) if exhibited]
            env_alt = (wished_alt, combis, exhibited_necessities, drive.citecounter.cite_arrows(), self.inert_orderlines(), adjudged_idents, cleared_idents, cap, amt, slot)
            amalgamated = [self._cluster_element(lbl, row, env_alt) for lbl, row in sealer.routine_datarows().items() if row["shelf"] in hub]
        jotting[key] = amalgamated
        return amalgamated

    def dx_attestation(self) -> list[int]:
        # Products whose every read listing is ruled out.
        def every_outward(kx):
            opened = [y_alt for y_alt in self.dx_cluster(kx) if y_alt["read"]]
            return bool(opened) and all(y_alt["out"] for y_alt in opened)
        return [kx for kx in range(len(self.drive.needs.items())) if every_outward(kx)]

    def _guide_merits(self, kx: int, hits: set, tome: list) -> dict[str, int]:
        drive = self.drive
        cohorts = [{tkn for tkn in re.findall(r"[a-z0-9]{2,}", value.lower()) if tkn not in ANCHOR_VOCABLES}
                  for _, value, _ in self.dx_dems(kx)]
        merits: dict[str, int] = dict.fromkeys(hits, 1)
        for search in tome:
            words = [tkn for text in drive.observe.unbound_prose(search["args"]) for tkn in re.findall(r"[a-z0-9]{2,}", text.lower())]
            if drive.guide_vocables:
                words = words[: drive.guide_vocables]
            fore = set(words)
            conveyed = sum(1 for group in cohorts if group & fore)
            if not conveyed:
                continue
            for lbl in search.get("row_shelf") or {}:
                name = str(lbl)
                merits[name] = max(merits.get(name, 0), conveyed)
        return merits

    def _thriftiest_bottoms(self, count: int, apiece_metric):
        bottoms = []
        for kx in range(count):
            suitable = [y_alt["price"] for y_alt in self.dx_cluster(kx) if y_alt["fit"] and y_alt["price"] is not None]
            if not suitable:
                return None
            bottoms.append(min(suitable) * apiece_metric(kx))
        return bottoms

    def cushion_datarows(self) -> list[str]:
        drive = self.drive
        wished_alt = drive.needs.items()
        cap = drive.needs.disburse_top
        if len(wished_alt) < 2 or not be_numeral(cap):
            return []
        try:
            censuses = drive.needs.censuses()

            def apiece_metric(kx: int):
                return censuses[kx] if kx < len(censuses) else 1
            bottoms = self._thriftiest_bottoms(len(wished_alt), apiece_metric)
            if bottoms is None or sum(bottoms) <= float(cap) + 1e-9:
                return []
            tome = list(drive.page_book.values())
            hits = set(drive.encompass.every_impacts or [])

            def waitlist_towards(kx):
                # Unread candidates for product k cheap enough to fit beside the other products' floors.
                room = (float(cap) - (sum(bottoms) - bottoms[kx])) / max(1, apiece_metric(kx))
                merits = self._guide_merits(kx, hits, tome)
                ranked = []
                for y_alt in self.dx_cluster(kx):
                    if y_alt["read"] or y_alt["gone"] or y_alt["pointer"] or y_alt["ruled"] or y_alt["clash"] or y_alt["price"] is None:
                        continue
                    if 0 < room and y_alt["price"] <= room + 1e-9:
                        ranked.append((-merits.get(y_alt["id"], 0), y_alt["price"], y_alt["id"]))
                return [lbl for _, _, lbl in sorted(ranked)]
            backlogs = [waitlist_towards(kx) for kx in range(len(wished_alt))]
            preferred: list[str] = []
            while len(preferred) < 3 and any(backlogs):
                for waitlist in backlogs:
                    if waitlist and len(preferred) < 3:
                        preferred.append(waitlist.pop(0))
            if preferred:
                self.cushion_overall = sum(bottoms)
            return preferred
        except Exception as failure:
            drive.diary.append({"kind": "slack_error", "error": f"{type(failure).__name__}: {failure}"[:300]})
            return []

    def authenticate_troubles(self, lbl: str) -> list[str] | None:
        # Problems from the newest verify entry for this listing, only when it said the listing does not fit.
        for element_alt in reversed(self.drive.diary):
            if not isinstance(element_alt, dict) or element_alt.get("kind") != "verify" or element_alt.get("item") != lbl:
                continue
            upshot = element_alt.get("result")
            if not isinstance(upshot, dict) or upshot.get("fits") is not False:
                return None
            return [str(pz) for pz in upshot.get("problems") or [] if str(pz).strip()]
        return None

    def dx_tidings(self, kx: int | None) -> bool:
        drive = self.drive
        if drive.refusal_n > 0:
            return True
        rows = drive.needs.items()
        return any(not rows or drive.speak.commodity_relating(str(req).partition(":")[0]) in (kx, None)
                   for req in [*drive.needs.uncovered, *drive.needs.let_go])

    def nigh_maintain(self) -> str:
        drive = self.drive
        try:
            intel = self.nigh_buggy
            if (not intel or intel.get("ver") != drive.needs.edition or intel.get("refusals") != drive.refusal_n
                    or not drive.role("order")):
                return ""
            lines = list(intel.get("lines") or [])
            if not lines:
                return ""
            if any(lbl not in drive.wagon.metrics or lbl in self.inert_orderlines() or lbl in drive.refused_lines
                   for lbl in lines):
                return ""
            channel = drive.channel
            if drive.maneuvers_residual() < 1 or channel.done or channel.snapped or channel.extra_alt() < 2:
                return ""
            self.nigh_buggy = None
            drive.diary.append({"kind": "near_order", "lines": lines})
            antecedent = channel.muffle
            try:
                drive.sealer.terminate_presently()
            finally:
                channel.muffle = antecedent
            return ("Not run (no step used). An earlier wording review established that the cart's lines ("
                    + ", ".join(lines) + ") state the requested values in other words, and nothing has changed since, so "
                    "the code placed the order for the cart rather than declining.")
        except Exception as failure:
            drive.diary.append({"kind": "near_hold_error", "error": f"{type(failure).__name__}: {failure}"[:300]})
            return ""

    def near_check(self) -> str:
        drive = self.drive
        channel = drive.channel
        if self.nigh_accomplished or not drive.role("cart") or not drive.role("order") or drive.maneuvers_residual() < 4:
            return ""
        if channel.extra_alt() <= SEALING + 30 or channel.muffle or drive.covering or channel.done or channel.disbursed_outward() or channel.nil_oracle:
            return ""
        try:
            return self.near_basket()
        except Exception as failure:
            drive.diary.append({"kind": "near_error", "error": f"{type(failure).__name__}: {failure}"[:300]})
            return ""

    def _couched_orderlines(self, element_alt, dems):
        # (prop, value, field text) for every requirement the row misses only in wording; None if any miss is real.
        sealer = self.drive.sealer
        row = element_alt["row"]
        out = []
        for prop, value, disclosed in dems:
            if sealer.routine_exhibits(row, prop, value, set(), True):
                continue
            written = [str(vlu).strip() for key, vlu in row["fields"].items() if equal_feat(key, prop) and str(vlu).strip()]
            if disclosed or not written or not self.dx_vocable(value) or any(re.search(r"\d", text) for text in written):
                return None
            out.append((prop, value, "; ".join(dict.fromkeys(written))[:120]))
        return out

    def _nigh_contenders(self, cluster, dems):
        referee = self.drive.referee
        discovered = []
        for element_alt in cluster:
            if not element_alt["read"] or element_alt["gone"] or element_alt["over"] or element_alt["pointer"] or element_alt["price"] is None:
                continue
            lines = self._couched_orderlines(element_alt, dems)
            faltered_feats, problems = set(), []
            if element_alt["ruled"]:
                faltered_feats = {str(xq).split(":", 1)[0].strip() for xq in referee.column_glance(element_alt["id"])["fails"]}
                problems = self.authenticate_troubles(element_alt["id"]) or []
            if not lines:
                continue
            props = [prop for prop, _, _ in lines]
            # Every failed field must be one of the reworded properties ...
            if any(not any(equal_feat(faltered, prop) for prop in props) for faltered in faltered_feats):
                continue
            # ... and every reported problem must mention one of their words.
            marks = {tkn for prop, value, _ in lines for tkn in feat_vocables(prop) + re.findall(r"[a-z0-9]+", value.lower())
                     if len(tkn) >= 3 and tkn not in ANCHOR_VOCABLES}
            if all(any(re.search(r"\b" + re.escape(tkn), problem.lower()) for tkn in marks) for problem in problems):
                discovered.append((len(lines), element_alt["price"], element_alt["id"], lines))
        return discovered

    def _nigh_directive(self, even) -> str:
        listing = "\n".join(f"{n}. {lbl}: {prop} = '{value[:60]}' | the listing's own field reads: {states}"
                            for n, (_, lbl, prop, value, states) in enumerate(even, 1))
        return ("Task: some listings do not match a requested value word for word, which would make the shopping assistant "
                "decline them; decide whether each one states that value in other words. THE SHOPPER'S REQUEST (background "
                "only):\n" + self.drive.request[:1500] + "\n\nEach line gives a listing, a requested property with its "
                "requested value, and what that listing's own field says for it.\n" + listing + "\n\n1. Judge each line "
                "separately.\n2. Answer true only when the field expresses the requested value in different words (a "
                "synonym, a variant spelling or phrasing, or a more specific form of the same thing); every other value of "
                "the property counts as false.\n3. If unsure, answer false.\n"
                "Return: exactly ONE JSON object and nothing else: {\"lines\": [{\"n\": 1, \"same\": true}]}")

    def _nigh_clusters(self, openings, metrics, satchel):
        clusters: dict[int, list] = {}
        keep = 6 if len(openings) == 1 else 3
        for n, kx in enumerate(openings):
            logs = self.dx_cluster(kx)
            cheapest = min(((y_alt["price"], y_alt["id"]) for y_alt in logs if y_alt["fit"] and y_alt["price"] is not None), default=None)
            if cheapest is not None:
                satchel[n] = (cheapest[1], cheapest[0], metrics(n))
                continue
            if not self.dx_tidings(kx):
                return None
            alternatives = self._nigh_contenders(logs, self.dx_dems(kx))
            if not alternatives:
                return None
            clusters[n] = sorted(alternatives)[:keep]
        return clusters

    def _nigh_accepted(self, answer) -> set[int]:
        if not isinstance(answer, dict) or not isinstance(answer.get("lines"), list):
            return set()
        return {int(verdict["n"]) - 1 for verdict in answer["lines"]
                if isinstance(verdict, dict) and verdict.get("same") is True and be_numeral(verdict.get("n"))}

    def near_basket(self) -> str:
        drive = self.drive
        needs = drive.needs
        wished_alt = needs.items()
        openings: list[int | None] = list(range(len(wished_alt))) if wished_alt else [None]
        censuses = needs.censuses()

        def metrics(n: int):
            return censuses[n] if n < len(censuses) else 1
        cap = float(needs.disburse_top) if be_numeral(needs.disburse_top) else None
        satchel: dict[int, tuple[str, float, int]] = {}
        clusters = self._nigh_clusters(openings, metrics, satchel)
        if not clusters:
            return ""

        def total(surplus: float = 0.0) -> float:
            return sum(price * numerous for _, price, numerous in satchel.values()) + surplus

        def above_top(quantity_alt):
            return cap is not None and quantity_alt > cap + 1e-9
        if above_top(total(sum(min(alternative[1] for alternative in clusters[n]) * metrics(n) for n in clusters))):
            return ""
        even = [(n, lbl, prop, value, states)
                for n in sorted(clusters) for _, _, lbl, lines in clusters[n] for prop, value, states in lines][:12]
        self.nigh_accomplished = True
        answer = drive.channel.canvass(self._nigh_directive(even), 600, VALIDATOR)
        accepted_alt = self._nigh_accepted(answer)
        drive.diary.append({"kind": "near_check", "lines": [f"{ii}:{pz}={vz}|{s}"[:120] for _, ii, pz, vz, s in even], "same": sorted(accepted_alt)})
        # Per product, take the first candidate whose every reworded line the review accepted.
        for n in sorted(clusters):
            for _, price, lbl, lines in clusters[n]:
                loci = [at for at, row in enumerate(even) if row[0] == n and row[1] == lbl]
                if len(loci) == len(lines) and all(at in accepted_alt for at in loci):
                    satchel[n] = (lbl, price, metrics(n))
                    break
            if n not in satchel:
                return ""
        if above_top(total()):
            return ""
        picks = [(satchel[n][0], satchel[n][2]) for n in range(len(openings))]
        if drive.maneuvers_residual() < len(picks) + 2:
            return ""
        edition = str(needs.edition)
        for n in clusters:
            preferred = satchel[n][0]
            seal = f"{preferred}|{edition}"
            for jotting in [mz for mz in drive.rulings if mz == seal or mz.startswith(seal + "|")]:
                drive.rulings[jotting] = ""
            drive.passed.add(seal)
            self.nigh_sound.add(seal)
            drive.pass_price[preferred] = drive.stash.price(preferred)
        done = self._mend_deposit(picks)
        drive.diary.append({"kind": "near_basket", "picks": picks, "done": done, "cart": dict(drive.wagon.metrics)})
        if not done:
            return ""
        self.nigh_buggy = {"ver": needs.edition, "refusals": drive.refusal_n, "lines": [lbl for lbl, _ in picks]}
        kept = [(lbl, prop, value, states) for n, lbl, prop, value, states in even if lbl == satchel.get(n, ("",))[0]]
        for lbl, prop, value, states in kept:
            self.nigh_uttered.setdefault(f"{lbl}|{edition}", []).append((prop, value, states))
        uttered = "; ".join(f"{lbl} gives {prop} as '{states}' where '{value[:60]}' was asked" for lbl, prop, value, states in kept)
        return ("Not run yet (no step used). A wording review found that " + uttered + " (the same value, just worded "
                "differently), and each listing involved meets every need the shopper revealed, is in stock and fits the "
                "budget, so the code put the basket into the cart and checked its stock: " + drive.wagon.abstract()
                + " Place the order for this cart now; decline only if a record you have read shows one of these lines "
                "breaking a requirement currently in force.")

    def buggy_maneuvered(self, preceding: dict[str, float]) -> None:
        drive = self.drive
        self.buggy_rev = 1 if self.buggy_rev is None else self.buggy_rev + 1
        drive.starting_buggy = 0
        now = drive.wagon.metrics
        # A brand-new line restarts the stock clock; a line whose count grew loses its fresh-read mark.
        if any(line not in preceding for line in now):
            drive.cart_seq = drive.absorb_n
        for line, count in now.items():
            if line in preceding and count > preceding[line] + 1e-9:
                drive.line_seq.pop(line, None)

    def afterward_deposit(self, trigger_operands: dict[str, Any], obv: Any) -> None:
        drive = self.drive
        wagon = drive.wagon
        if wagon.glance_metrics(obv) is not None:
            self.buggy_exhibited = self.buggy_rev if self.buggy_rev is not None else 0
        crisp = [line for line, count in wagon.lines(trigger_operands)
                 if (count is None or count > 0) and line in wagon.metrics and line not in self.primary_deposits]
        if not crisp or not wagon.be_deposit(trigger_operands) or drive.sealer.routine_flaw(obv):
            return
        self.primary_deposits.update(crisp)
        obsolete = set(self.obsolete_orderlines())
        crisp = [line for line in crisp if line in obsolete]
        tool = self.supply_utility()
        if not crisp or not tool or drive.channel.muffle or drive.covering:
            return
        if drive.maneuvers_residual() < 4 or drive.channel.extra_alt() < SEALING:
            return
        # Queue a stock read for lines just added for the first time (at most four waiting).
        browses = drive.browses
        tarrying = list(browses.deposit_waitlist or [])
        browses.deposit_waitlist = (tarrying + [line for line in crisp if line not in tarrying])[:4]
        browses.deposit_utility = tool
        drive.diary.append({"kind": "after_put_read", "lines": list(browses.deposit_waitlist), "tool": tool})

    def stamp_crisp(self, obv: Any) -> None:
        drive = self.drive
        if not self.supply_noticed:
            return
        for rcd in Referee.hashmaps_amidst([obv]):
            lbl = self.rcd_ident(rcd)
            if lbl in drive.wagon.metrics and drive.observe.supply_situation(rcd) is not None:
                drive.line_seq[lbl] = drive.absorb_n

    def buggy_departed(self, lines: list[str]) -> set[str]:
        drive = self.drive
        gone = set(lines) & self.inert_orderlines()
        unbrowsed = [line for line in self.obsolete_orderlines() if line in lines and line not in gone]
        if unbrowsed and self.supply_utility() and drive.maneuvers_residual() >= len(unbrowsed) + 2:
            _, vended = drive.sealer.routine_revalidate(unbrowsed)
            gone |= set(vended) | (set(lines) & self.inert_orderlines())
        return gone

    def _modifying_utilities(self) -> set:
        # Tools that change state (cart, order, removal ...) by role or by what their name/description says.
        drive = self.drive
        engaged = {drive.role(rq) for rq in ("search", "message", "cart", "cart_view", "order", "decline")} - {None}
        voiced: dict[str, str] = {}
        for tdecl in drive.utilitykit or []:
            cb = tdecl.get("function") if isinstance(tdecl, dict) else None
            if isinstance(cb, dict) and isinstance(cb.get("name"), str):
                name = cb["name"]
                words = re.sub(r"([a-z])([A-Z])", r"\1 \2", name).replace("_", " ")
                voiced[name] = words + " " + str(cb.get("description") or "")
        hazardous = r"\b(?:remov|delet|cancel|clear|discard|releas|refund|purchas|checkout|wish|basket|cart|bag)"
        engaged.update(name for name in list(drive.contours) if re.search(hazardous, voiced.get(name, name).lower()))
        return engaged

    @staticmethod
    def _lineup_parameter(framework):
        be_atlas = isinstance(framework, dict)
        props = framework.get("properties") if be_atlas else None
        if not isinstance(props, dict):
            props = {}

        def strval_vector(brief):
            if not isinstance(brief, dict) or brief.get("type") != "array":
                return False
            internal = brief.get("items")
            return isinstance(internal, dict) and internal.get("type") == "string" and not internal.get("enum")
        vectors = [key for key, brief in props.items() if strval_vector(brief)]
        if len(vectors) != 1:
            return None
        key = vectors[0]
        required = framework.get("required") if be_atlas and isinstance(framework.get("required"), list) else []
        if any(name not in vectors for name in required):
            return None
        maximal = props[key].get("maxItems")
        if not be_numeral(maximal) or maximal < 2:
            return None
        return key, int(min(maximal, 10))

    def _locate_lot_brief(self):
        engaged = self._modifying_utilities()
        for name, framework in self.drive.contours.items():
            if name in engaged:
                continue
            brief = self._lineup_parameter(framework)
            if brief:
                return (name, *brief)
        return None

    def lot_utility(self) -> tuple[str, str, int] | None:
        drive = self.drive
        if not drive.lot_brief_recognized:
            try:
                brief = self._locate_lot_brief()
            except Exception:
                brief = None
            drive.batch_spec, drive.lot_brief_recognized = brief, True
        return drive.batch_spec

    @staticmethod
    def lot_sound(discovered: list[Any], ids: list[str]) -> bool:
        # Did a batch read return full records for at least half of the asked ids?
        wished_alt = set(ids)
        obtained: set[str] = set()

        def primitives(assoc):
            return sum(1 for xq in assoc.values() if isinstance(xq, (str, int, float)))
        try:
            for rcd in Referee.hashmaps_amidst(discovered):
                values = list(rcd.values())
                if any(isinstance(vlu, dict) for vlu in values):
                    brimming = any(isinstance(vlu, dict) and primitives(vlu) >= 2 for vlu in values)
                else:
                    brimming = sum(1 for vlu in values if isinstance(vlu, str) and 0 < len(vlu) <= 70) >= 5
                if brimming:
                    obtained.update(vlu for vlu in values if isinstance(vlu, str) and vlu in wished_alt)
        except Exception:
            return False
        return bool(obtained) and 2 * len(obtained) >= len(wished_alt)

    @staticmethod
    def headline_relating(row: dict[str, Any]) -> str:
        if isinstance(row.get("title"), str):
            return row["title"]
        return next((vlu for key, vlu in row.items()
                     if isinstance(vlu, str) and re.search(r"title|heading|headline", str(key).lower())), "")

    def _guided_primary(self, hub, maximal, words, known) -> list[str]:
        drive = self.drive
        needs = drive.needs
        manifold = bool(needs.items())
        slot = next((n for n, shelf in enumerate(self.hub_aisles()) if shelf == hub), None) if manifold else None
        values = []
        for req in needs.feats_residual():
            head, _, rear = req.partition(":")
            if manifold and drive.speak.commodity_relating(head) not in (slot, None):
                continue
            values.append(re.sub(r"\((?:stated|revealed[^)]*)\)", "", rear))
        values = [vlu for vlu in values if vlu.strip() and favorable_measurement(vlu)
                  and not re.search(r"as shown|same as|the page|sold by|redirect|refer", vlu.lower())]
        if not (values and maximal > 0 and (not manifold or slot is not None)):
            return []
        # Gather the property texts of read listings on this shelf; stop if one already shows every value.
        observe = drive.observe
        facts: dict[str, list[str]] = {}
        for rcd in Referee.hashmaps_amidst(drive.stash.items):
            lbl = observe.datarow_stamp(rcd)
            if lbl not in drive.stash.read or observe.aisle_relating(rcd) not in hub:
                continue
            for vlu in rcd.values():
                if isinstance(vlu, dict):
                    facts.setdefault(lbl, []).extend(str(xq) for xq in vlu.values() if isinstance(xq, (str, int, float)))
        if any(all(measurement_noticed(vlu, " | ".join(proses)) for vlu in values) for proses in facts.values()):
            return []
        traverser = drive.walk
        guided = sorted((wq for wq in drive.page_book.values() if traverser.route_transports(wq["args"], slot)),
                     key=lambda wq: -len(traverser.route_transports(wq["args"], slot)))
        for walk in guided:
            first = next((lbl for lbl in (walk.get("row_shelf") or {}) if lbl in words), None)
            if first:
                known.append(words[first])
                return [first]
        return []

    def panel_datarows(self, hub: set[str], maximal: int, occupied: list[str], amt: int = 1) -> list[str]:
        drive = self.drive
        stash, observe = drive.stash, drive.observe
        cap = drive.needs.disburse_top

        def headline_vocables(text: str) -> set[str]:
            return set(re.findall(r"[a-z0-9]{3,}", text.lower()))
        crisp: list[str] = []
        names: dict[str, str] = {}
        gone: set[str] = set()
        browse_collections: list[set[str]] = []
        for row in Referee.hashmaps_amidst(stash.items):
            stamp_alt = observe.datarow_stamp(row)
            if stamp_alt not in stash.ids or observe.aisle_relating(row) not in hub:
                continue
            if stamp_alt in stash.read:
                # an already opened listing of this shelf: its title words steer later picks toward variety
                browse_collections.append(headline_vocables(self.headline_relating(row)))
                continue
            if observe.vended_outward(row):
                gone.add(stamp_alt)
            if stamp_alt not in names:
                crisp.append(stamp_alt)
                names[stamp_alt] = self.headline_relating(row)
        heft = drive.encompass.impact_merit
        hits = sorted(drive.encompass.every_impacts, key=lambda stamp_alt: -heft.get(stamp_alt, 0))
        cluster = [mz for mz in hits if mz in names]
        cluster += [mz for mz in crisp if mz not in hits]

        def amongst_top(stamp_alt: str) -> bool:
            if not be_numeral(cap):
                return True
            price = stash.price(stamp_alt)
            return not (be_numeral(price) and price * amt > cap)
        cluster = [mz for mz in cluster
                if mz not in gone and mz not in occupied and mz not in drive.refused_lines and amongst_top(mz)]
        sacks = {mz: headline_vocables(names.get(mz, "")) for mz in cluster}
        guided = drive.encompass.datarow_merit
        try:
            picks = self._guided_primary(hub, maximal, sacks, browse_collections)
        except Exception:
            picks = []

        def posture(stamp_alt: str):
            sack = sacks[stamp_alt]
            proximity = max((len(sack & further) / max(1, len(sack | further)) for further in browse_collections), default=0.0)
            if stamp_alt in hits:
                stratum = 0
            elif guided.get(stamp_alt):
                stratum = 1
            else:
                stratum = 2
            return stratum, proximity, cluster.index(stamp_alt)
        while len(picks) < maximal:
            pane = [mz for mz in cluster if mz not in picks][:16]
            if not pane:
                break
            victor = min(pane, key=posture)
            picks.append(victor)
            browse_collections.append(sacks[victor])
        return picks

    def _waitlist_panel(self, picks: list[str], utter: str) -> None:
        browses = self.drive.browses
        browses.access_waitlist, browses.access_revalidate = picks, True
        browses.access_utter, browses.access_intention = utter, "menu"

    def _elaborate_census(self, hub) -> int:
        if hub:
            stash = self.drive.stash
            return sum(stash.elaborate_browses(label) for label in hub)
        return 4

    def _latter_bout(self, houses, lot, amt_towards) -> bool:
        self.prior_bouts = 2
        channels: list[list[str]] = []
        asserted: list[str] = []
        for kx, hub in enumerate(houses):
            possess = self._elaborate_census(hub)
            if possess >= 3:
                continue
            surplus = self.panel_datarows(hub, 4 - possess, asserted + list(self.prior_selected or []), amt_towards(kx))
            asserted += surplus
            channels.append(surplus)
        # one listing from each product in turn, until every lane runs dry
        blended: list[str] = []
        for depth in range(max((len(corridor) for corridor in channels), default=0)):
            blended.extend(corridor[depth] for corridor in channels if depth < len(corridor))
        blended = blended[: lot[2]]
        if not blended:
            return False
        self._waitlist_panel(blended, "Reading a few extra listings first, since what was opened for some products showed only a handful of fields.")
        return True

    def _primary_bout(self, houses, lot, amt_towards) -> bool:
        drive = self.drive
        naked: list[str] = []
        concise: list[tuple[int, int]] = []
        for kx, hub in enumerate(houses):
            possess = self._elaborate_census(hub)
            if possess == 0:
                wished_alt = 2 if drive.needs.items() else 4
                naked += self.panel_datarows(hub, wished_alt, naked, amt_towards(kx))
            elif possess < 4:
                concise.append((kx, 4 - possess))
        waitlist = list(naked)
        for kx, missing in concise:
            waitlist += self.panel_datarows(houses[kx], missing, waitlist, amt_towards(kx))
        broadness = lot[2]
        twofold = len(naked) > broadness and drive.maneuvers_residual() > 25
        waitlist = waitlist[: broadness * (2 if twofold else 1)]
        if not waitlist:
            return False
        self.prior_browse_accomplished = True
        self.prior_bouts = 1
        self.prior_selected = list(waitlist)
        self._waitlist_panel(waitlist, "Before asking anything, a few listings of each requested product get opened so their fields are known.")
        return True

    def prior_browse(self) -> bool:
        drive = self.drive
        try:
            lot = self.lot_utility()
            second = self.prior_bouts == 1
            if not lot:
                return False
            if self.prior_browse_accomplished and not second:
                return False
            if len(drive.questions_log) > 1:
                return False
            if drive.maneuvers_residual() <= (25 if second else 20) or drive.channel.extra_alt() < SEALING + 60:
                return False
            houses = self.hub_aisles()
            apiece_article = [row["quantity"] for row in drive.needs.items()] or [drive.needs.quantity()]

            def amt_towards(slot: int):
                return apiece_article[min(slot, len(apiece_article) - 1)]
            step = self._latter_bout if second else self._primary_bout
            return step(houses, lot, amt_towards)
        except Exception as failure:
            drive.diary.append({"kind": "pre_read_error", "error": f"{type(failure).__name__}: {failure}"[:300]})
            return False

    def _encompassed_captions(self) -> set[str]:
        drive = self.drive
        referee = drive.referee
        actual = str(drive.needs.edition)
        wiped = {lbl for lbl, _, tag in (seal.rpartition("|") for seal in drive.passed) if tag == actual}
        wiped = wiped - drive.refused_lines - self.inert_orderlines()
        captions = {referee.aisle_recognized(lbl) for lbl in wiped | set(referee.inspection_befits or [])}
        captions.discard("")
        return captions

    def deficient_datarows(self) -> list[str]:
        drive = self.drive
        stash = drive.stash
        try:
            wished_alt = drive.needs.items()
            if not wished_alt:
                return []
            houses = self.hub_aisles()
            encompassed = self._encompassed_captions()
            access_openings = [kx for kx, hub in enumerate(houses) if hub and not hub & encompassed]
            if not access_openings:
                return []
            lot = self.lot_utility()
            constraint = lot[2] if lot else 3
            cap = drive.needs.disburse_top
            observe = drive.observe
            # newest records first, so each listing keeps the shelf label it was last seen under
            spied: list[str] = []
            caption_relating: dict[str, str] = {}
            gone: set[str] = set()
            for row in Referee.hashmaps_amidst(reversed(stash.items)):
                lbl, label = observe.datarow_stamp(row), observe.aisle_relating(row)
                if lbl not in stash.ids or not label:
                    continue
                spied.append(lbl)
                caption_relating.setdefault(lbl, label)
                if observe.vended_outward(row):
                    gone.add(lbl)
            heft = drive.encompass.impact_merit
            hits = sorted(drive.encompass.every_impacts, key=lambda lbl: -heft.get(lbl, 0))
            channels: dict[int, list[str]] = {kx: [] for kx in access_openings}
            for lbl in dict.fromkeys([ii for ii in hits if ii in caption_relating] + spied):
                if lbl in stash.read or lbl in drive.refused_lines or lbl in gone:
                    continue
                slot = next((kx for kx in access_openings if caption_relating.get(lbl) in houses[kx]), None)
                if slot is None:
                    continue
                price = stash.price(lbl)
                if be_numeral(cap) and be_numeral(price) and price * wished_alt[slot]["quantity"] > cap:
                    continue
                channels[slot].append(lbl)
            blended: list[str] = []
            for depth in range(max(len(corridor) for corridor in channels.values())):
                blended.extend(channels[kx][depth] for kx in access_openings if depth < len(channels[kx]))
            return blended[: max(0, constraint)]
        except Exception as failure:
            drive.diary.append({"kind": "lacking_error", "error": f"{type(failure).__name__}: {failure}"[:300]})
            return []

class Colloquy:
    def __init__(self, drive):
        self.drive = drive
        # when and where the last question went out
        self.solicited_upon_collections = None
        self.solicited_upon_beat = 0
        self.soliciting_caption = None
        # planner bookkeeping
        self.vacant_strategies = 0
        self.strategy_tarry = None
        self.commodity_strategies = 0
        # pairs and the halves still owed
        self.nil_duos = None
        self.access_duo = []
        self.awaiting_runnersup = []
        self.awaiting_prose = {}
        # per-product quota state
        self.ration_browses = {}
        self.ration_tarry = False
        # history of asked properties
        self.still_solicits = []
        self.resolicited = set()
        self.disbursed_solicits = set()
        self.wished_solicits = None

    def prose_operand(self, that_utility: str) -> str | None:
        framework = self.drive.contours.get(that_utility or "") or {}
        fields = framework.get("properties") or {}
        shall = list(framework.get("required") or [])

        def maintains_prose(brief: Any) -> bool:
            if not isinstance(brief, dict) or "enum" in brief:
                return False
            kind = brief.get("type")
            return kind == "string" or "string" in (kind or [])
        verbal = [field for field, brief in fields.items() if maintains_prose(brief)]
        preferred = next((field for field in shall if field in verbal), None)
        if not preferred and len(verbal) == 1:
            preferred = verbal[0]
        if not preferred:
            return None
        # every required argument has to be that one text field, or the call cannot be built from text alone
        return None if any(field != preferred for field in shall) else preferred

    def utterance_trigger(self, prose_corpus: str, cuid: str) -> dict[str, Any] | None:
        drive = self.drive
        tool = drive.role("message")
        field = self.prose_operand(tool) if tool else None
        if not field:
            return None
        args, faults = OperandValidate.actuate({field: prose_corpus}, drive.contours[tool])
        if faults:
            return None
        return {"id": cuid, "type": "function", "function": {"name": tool, "arguments": json.dumps(args)}}

    def commodity_relating(self, text: str) -> int | None:
        rows = self.drive.needs.items()
        if not rows:
            return None
        kinds = [set(variety_morphemes(row["product"])) for row in rows]
        competitors = [set().union(*(kinds[s_alt] for s_alt in range(len(kinds)) if s_alt != ofs)) for ofs in range(len(kinds))]
        entire = str(text)
        # a bracketed product tag is tried before the full text
        for shard in re.findall(r"\(([^)]*)\)", entire)[:1] + [entire]:
            words = set(variety_morphemes(shard))
            marks = [variety_merit(personal, words, competitors[ofs]) for ofs, personal in enumerate(kinds)]
            best = max(marks)
            if best > 0 and marks.count(best) == 1:
                return marks.index(best)
        return None

    def strategy_commodity(self, qrow: dict[str, Any]) -> int | None:
        for field in ("property", "question"):
            slot = self.commodity_relating(str(qrow.get(field) or ""))
            if slot is not None:
                return slot
        return None

    def _reckoning_solicited(self, count: int):
        retorted = [0] * count
        submitted = [0] * count
        for element_alt in self.drive.questions_log:
            head = element_alt.split(":", 1)[0]
            slot = self.commodity_relating(head)
            if slot is None:
                uttered = re.search(r'asked "(.*?)" -> ', element_alt)
                if uttered:
                    slot = self.commodity_relating(uttered.group(1))
            if slot is None:
                continue
            submitted[slot] += 1
            if "[not answered]" not in head:
                retorted[slot] += 1
        return retorted, submitted

    def _cued_commodities(self, count: int, responded: list[int]) -> set:
        admission = self.drive.admission
        try:
            admission.access_cues()
        except Exception:
            pass
        try:
            by_name: set[int] = set()
            anonymous = [0] * count
            for hint in admission.access_cues():
                head = str(hint).split(":", 1)[0]
                slot = self.commodity_relating(head)
                if slot is None:
                    continue
                if admission.bare_cue(slot, head.strip(), cued=True):
                    anonymous[slot] += 1
                else:
                    by_name.add(slot)
            return {s_alt for s_alt in range(count) if s_alt in by_name or anonymous[s_alt] > responded[s_alt]}
        except Exception:
            return set()

    def _latter_unaccompanied(self, automated, expected_alt, rows):
        for plan in automated:
            executor_aloft = str(plan.get("second") or "").strip()
            slot = self.commodity_relating(executor_aloft) if executor_aloft else None
            if slot not in expected_alt:
                continue
            tag = re.search(r"\(([^)]+)\)", executor_aloft)
            product = tag.group(1).strip() if tag else rows[slot]["product"]
            return {"property": executor_aloft, "question": f"Which {executor_aloft.split('(')[0].strip()} would you like for the {product}?"}
        return None

    def _tiers(self, count, responded, delivered, disclosed, cued, alternate_residual) -> dict[int, tuple]:
        unperceived = [s_alt for s_alt in range(count) if not responded[s_alt]]

        def degree(kx: int) -> tuple:
            maintains_reverse = 1 if disclosed[kx] and any(s_alt != kx for s_alt in unperceived) else 0
            disregarded = 0 if kx in alternate_residual else delivered[kx] - responded[kx]
            return (maintains_reverse, 0 if kx in cued else 1, responded[kx] + disclosed[kx], disregarded, disclosed[kx])
        return {kx: degree(kx) for kx in range(count)}

    def _ration_towards(self, expected_alt, automated, cued, responded, ok):
        admission = self.drive.admission
        for slot in sorted(expected_alt, key=lambda s: (s not in cued, s)):
            try:
                select = self.ration_select(slot, responded[slot] == 0, ok)
            except Exception:
                select = None
            if select and ok is not None and (not ok(select.get("property")) or admission.solicited_towards(slot, select.get("property"))):
                select = None
            if select:
                return [select] + automated
            if self.ration_tarry:
                return automated
        return None

    def ration_primary(self, automated: list[dict[str, Any]], ok: Any = None) -> list[dict[str, Any]]:
        drive = self.drive
        rows = drive.needs.items()
        if not rows:
            return automated
        count = len(rows)
        responded, delivered = self._reckoning_solicited(count)
        if max(responded) == 0 and automated and self.strategy_commodity(automated[0]) is None:
            return automated
        disclosed = [0] * count
        for need in drive.needs.uncovered:
            slot = self.commodity_relating(str(need).split(":", 1)[0])
            if slot is not None:
                disclosed[slot] += 1
        cued = self._cued_commodities(count, responded)
        alternate_residual = {s_alt for s_alt in range(count) if delivered[s_alt] > responded[s_alt] and drive.admission.alternate_untested(s_alt)}
        grid = self._tiers(count, responded, delivered, disclosed, cued, alternate_residual)
        # products sharing the most urgent level are served first
        for degree in sorted(set(grid.values())):
            expected_alt = [kx for kx in range(count) if grid[kx] == degree]
            locus = next((n for n, plan in enumerate(automated) if self.strategy_commodity(plan) in expected_alt), None)
            if locus is not None:
                return [automated[locus]] + automated[:locus] + automated[locus + 1:]
            alone = self._latter_unaccompanied(automated, expected_alt, rows)
            if alone:
                return [alone] + automated
            rearranged = self._ration_towards(expected_alt, automated, cued, responded, ok)
            if rearranged is not None:
                return rearranged
        return automated

    def _ration_access_datarows(self, kx: int, hub, product: str) -> None:
        drive = self.drive
        shelf = drive.shelf
        lot = shelf.lot_utility()
        if hub:
            unbrowsed = shelf.panel_datarows(hub, min(4, lot[2]) if lot else 2, [], drive.needs.items()[kx]["quantity"])
        else:
            unbrowsed = []
        if not unbrowsed or not (drive.role("details") or lot):
            return
        if self.ration_browses.get(kx) or drive.maneuvers_residual() <= 12:
            return
        self.ration_browses[kx] = True
        browses = drive.browses
        browses.access_waitlist = unbrowsed
        browses.access_revalidate = True
        browses.access_utter = f"First a look at a few {product} listings to learn their fields; the question about it follows."
        browses.access_intention = "quota"
        self.ration_tarry = True

    def _browse_hub(self, kx: int, hub) -> list[str]:
        stash = self.drive.stash
        discovered = [shelf for shelf in stash.aisles_browse() if shelf.strip().lower() in hub]
        if discovered or not hub or not self.ration_browses.get(kx):
            return discovered
        return [shelf for shelf in sorted(hub) if stash.feat_panel(shelf)[0]][:1]

    def _panel_inquiry(self, kx, product, menu, failsafe, fine):
        admission = self.drive.admission

        def workable(prop: str) -> bool:
            return fine(f"{prop} ({product})")
        try:
            hints = admission.access_cues(kx)
        except Exception:
            hints = []
        for hint in hints:
            prop = re.split(r"\s*\(", hint.split(":", 1)[0])[0].strip()
            if prop and not trash_selector(prop) and not admission.bare_cue(kx, prop, cued=True) and workable(prop):
                return prop
        if not failsafe:
            return None
        # a plain pass over the menu, then one that also lets the sample values decide
        for ponder in (False, True):
            for element_alt in menu.splitlines():
                prop = element_alt.strip("- ").split(" (in ")[0].strip()
                if prop and self.access_feat(kx, prop, element_alt, through_specimens=ponder) and workable(prop):
                    return prop
        return None

    def ration_select(self, kx: int, failsafe: bool = True, ok: Any = None) -> dict[str, Any] | None:
        drive = self.drive
        stash, admission = drive.stash, drive.admission
        product = drive.needs.items()[kx]["product"]
        houses = drive.shelf.hub_aisles()
        hub = houses[kx] if kx < len(houses) else set()
        captions = self._browse_hub(kx, hub)
        if not captions:
            self._ration_access_datarows(kx, hub, product)
            return None
        menu = admission.panel_shorten(stash.feat_panel(captions[0])[0], stash.elaborate_browses(captions[0]))

        def fine(label: Any) -> bool:
            if ok is None:
                return True
            return bool(ok(label)) and not admission.solicited_towards(kx, label)
        scheduled = self.product_plan(kx, menu)
        if scheduled and fine(scheduled.get("property")):
            return scheduled
        prop = self._panel_inquiry(kx, product, menu, failsafe, fine)
        if prop is None:
            return None
        return {"property": f"{prop} ({product})", "question": f"Which {prop} would you like for the {product}?"}

    def product_plan(self, kx: int, menu: str) -> dict[str, Any] | None:
        try:
            return self.commodity_strategy_trigger(kx, menu)
        except Exception:
            return None

    def commodity_strategy_trigger(self, kx: int, menu: str) -> dict[str, Any] | None:
        drive = self.drive
        admission = drive.admission
        if not menu or self.commodity_strategies >= 1 or drive.channel.extra_alt() < SEALING + 40:
            return None
        self.commodity_strategies += 1
        product = drive.needs.items()[kx]["product"]
        noisy = product.upper()
        left = max(1, drive.solicit_constraint() - drive.txts_delivered)
        components = [SOLICIT_STRATEGY.replace("QUESTION_COUNT", str(min(3, left))),
                 "\n\nTHE SHOPPER'S REQUEST:\n" + drive.request,
                 "\n\nREQUIREMENTS SO FAR:\n" + drive.needs.prose_corpus()]
        if drive.questions_log:
            components.append("\n\nALREADY ASKED (each with the shopper's answer):\n" + "\n".join(drive.questions_log))
        components.append(f"\n\nFOR THIS ROUND, EVERY QUESTION IS ABOUT THE {noisy}: each of the other products has had a question "
                     f"already, this one has had none. Pick only properties of the {product}, from its list below, "
                     "starting with the one the story's details point to.")
        try:
            hints = admission.access_cues(kx)
        except Exception:
            hints = []
        if hints:
            components.append("\n\nSTORY HINTS (details from the request's story, each paired with the property it most likely "
                         "points to, likeliest first; these are not requirements):\n"
                         + "\n".join(f"- {admission.cue_stamp(f_alt)}" for f_alt in hints[:6]))
        components.append(self.strategy_comments(kx))
        components.append(f"\n\nPROPERTY KEYS FOUND IN THE DETAILS OF THE {noisy} LISTINGS READ SO FAR (property name, number of listings "
                     "carrying it, sample values):\n" + menu)
        answer = drive.channel.canvass("".join(components), 1500, SCHEDULER[:1])
        drive.diary.append({"kind": "product_plan", "product": product, "result": answer})
        return self._strategy_selection(kx, product, answer)

    def _strategy_selection(self, kx, product, answer):
        admission = self.drive.admission
        personal_variety = variety_vocables(product)

        def solicit(prop: str) -> str:
            return f"Which {prop} would you like for the {product}?"
        for plan in (answer or {}).get("questions") or []:
            if not isinstance(plan, dict):
                continue
            lead = str(plan.get("property") or "").split("(")[0].strip()
            standby = str(plan.get("second") or "").split("(")[0].strip()
            if re.search(r"none|^-$|n/a", standby.lower()):
                standby = ""
            couched = str(plan.get("question") or "").strip()
            if admission.bare_cue(kx, lead):
                lead = ""
            if standby and admission.bare_cue(kx, standby):
                standby = ""
            if lead and admission.strategy_access(kx, lead):
                # the planner's own wording survives only when it names the product
                designations_itr = bool(couched) and bool(personal_variety & variety_vocables(couched))
                duo = ""
                if designations_itr and standby and not stringent_feat(lead, standby) and admission.strategy_access(kx, standby):
                    duo = standby
                return {"property": f"{lead} ({product})", "second": f"{duo} ({product})" if duo else "",
                        "question": couched if designations_itr else solicit(lead)}
            if standby and admission.strategy_access(kx, standby):
                return {"property": f"{standby} ({product})", "question": solicit(standby)}
        return None

    def _concluded_through(self, req: str, name: str, slot, samples: str, through_specimens: bool) -> bool:
        admission = self.drive.admission
        head, _, value = str(req).partition(":")
        if stringent_feat(head, name):
            return bool(admission.concludes(str(req), samples if through_specimens else ""))
        if equal_feat(head, name):
            if not favorable_measurement(value, head):
                return False
            cluster = samples or " | ".join(admission.column_measurements(slot, name))
            return bool(measurement_noticed(value, cluster))
        # a requirement on another key still settles this one when its words all show up in the sample values
        words = [wq for wq in re.findall(r"[a-z]{3,}", value.lower()) if wq not in ANCHOR_VOCABLES]
        if not samples or not words or re.search(r"\d", value) or not favorable_measurement(value, head):
            return False
        return all(re.search(r"\b" + re.escape(wq), samples) for wq in words)

    def access_feat(self, kx: int | None, prop: str, line: str = "", through_specimens: bool = False) -> bool:
        drive = self.drive
        admission = drive.admission
        try:
            name = str(prop).split("(")[0].strip()
            if not name or trash_selector(name):
                return False
            several = bool(drive.needs.items())

            def worries(text: str, keyed_solely: bool = False) -> bool:
                if kx is None or not several:
                    return True
                slot = self.commodity_relating(text)
                return slot == kx or (slot is None and not keyed_solely)
            for element_alt in drive.questions_log:
                head = element_alt.split(":", 1)[0].strip("- ")
                if "[not answered]" in head:
                    continue
                if any(worries(component) and admission.equal_column(kx, component, name) for component in head.split(";")):
                    return False
            samples = line.split("):", 1)[1].lower() if "):" in line else ""
            for req in drive.needs.feats_residual():
                if worries(str(req).partition(":")[0]) and self._concluded_through(req, name, kx, samples, through_specimens):
                    return False
            for loose in drive.needs.indifferent:
                if worries(str(loose), keyed_solely=True) and admission.equal_column(kx, str(loose), name):
                    return False
            return True
        except Exception:
            return True

    def strategy_comments(self, kx: int | None) -> str:
        drive = self.drive
        try:
            several = bool(drive.needs.items())
            solicited = {component.strip().lower() for element_alt in drive.questions_log
                     for component in element_alt.split(":", 1)[0].strip("- ").split(";")}

            def towards_cur(text: str) -> bool:
                return kx is None or not several or self.commodity_relating(text) in (kx, None)
            yet_access = []
            for prop in self.access_duo or []:
                slot = self.commodity_relating(prop) if several else None
                if kx is not None and several and slot not in (kx, None):
                    continue
                if prop.lower() not in solicited and drive.admission.strategy_access(slot, prop):
                    yet_access.append(prop)
            hoped = [wq for wq in self.wished_solicits or [] if towards_cur(wq)]
            partitions = []
            if yet_access:
                partitions.append("\n\nOPEN FROM LAST PAIR (the second half of a planned pair left without an answer, because "
                                "the shopper answered only the first or only the first was sent; each is still open and was "
                                "the planner's next pick):\n" + "\n".join(f"- {pz}" for pz in yet_access[-3:]))
            if hoped:
                partitions.append("\n\nQUESTIONS THE ASSISTANT WANTED TO ASK (noted while reading listing records; if one names "
                                "a still-open property that also appears as a key below, ask that key first, under its own "
                                "name):\n" + "\n".join(f"- {wq}" for wq in hoped[-3:]))
            return "".join(partitions)
        except Exception:
            return ""

    def wish_solicit(self, crude_operands: Any) -> None:
        try:
            if isinstance(crude_operands, dict):
                args = crude_operands
            elif isinstance(crude_operands, str) and crude_operands.strip():
                args = blob_thing(crude_operands)
            else:
                args = {}
            uttered = [str(vz).strip() for vz in (args or {}).values() if isinstance(vz, str) and vz.strip()]
            text = " ".join(uttered)[:300]
            if not text:
                return
            reduced = text.lower()
            rest_alt = [wq for wq in self.wished_solicits or [] if wq.lower() != reduced]
            self.wished_solicits = (rest_alt + [text])[-3:]
        except Exception:
            pass

    def jettison_wished(self, label: str) -> None:
        root_alt = self.drive.observe.root_alt
        try:
            scrubbed = str(label).replace("[not answered]", "")
            props = [re.split(r"\s*\(", shard)[0] for shard in re.split(r";|\band\b", scrubbed)]

            def responded_through_caption(text: str) -> bool:
                words = {root_alt(wq) for wq in re.findall(r"[a-z]+", text.lower())}
                return any(feat_vocables(pz) and all(wq in words or root_alt(wq) in words for wq in feat_vocables(pz)) for pz in props)
            self.wished_solicits = [text for text in self.wished_solicits or [] if not responded_through_caption(text)]
        except Exception:
            pass

    @staticmethod
    def _unitary(vocable: str) -> str:
        if len(vocable) > 3 and vocable.endswith("s"):
            return vocable[:-1]
        return vocable

    def _personal_vocables(self, prop: str) -> set[str]:
        needs = self.drive.needs
        personal = set(feat_vocables(prop)) | {"which", "would", "you", "like"}
        rows = needs.items() or [{"product": str(needs.browsing.get("product") or "")}]
        for row in rows:
            for token in re.findall(r"[a-z0-9]+", str(row.get("product") or "").lower()):
                personal.update((token, self._unitary(token)))
        return personal

    def designations_further(self, question: str, prop: str, menu: str, second: str = "") -> bool:
        try:
            voiced = " ".join(re.findall(r"[a-z0-9]+", str(question).lower()))
            heard = set(voiced.split())
            heard |= {tz[:-1] for tz in heard if len(tz) > 3 and tz.endswith("s") and not tz.endswith("ss")}
            personal = self._personal_vocables(prop)

            def references(field: str) -> bool:
                kernel = [tz for tz in feat_vocables(field) if len(tz) >= 3 and tz not in ANCHOR_VOCABLES]
                if not kernel or not all(tz in heard for tz in kernel):
                    return False
                return any(tz not in personal for tz in kernel)

            def apostrophes_specimen(element_alt: str) -> bool:
                itemized = element_alt.split("):", 1)[1] if "):" in element_alt else ""
                for exemplar in itemized.split("|"):
                    tokens = re.findall(r"[a-z0-9]+", exemplar.lower())
                    alien = [tz for tz in tokens if tz not in personal and tz not in ANCHOR_VOCABLES]
                    if not alien or len("".join(tokens)) < 3:
                        continue
                    if re.search(r"(?<![a-z0-9])" + re.escape(" ".join(tokens)) + r"(?![a-z0-9])", voiced):
                        return True
                return False
            if second and not stringent_feat(second, prop) and references(second):
                return True
            for element_alt in str(menu or "").splitlines():
                field = element_alt.strip().lstrip("-").strip().split(" (in ")[0].strip()
                if not field or field.endswith(":") or stringent_feat(field, prop) or trash_selector(field):
                    continue
                if references(field) or apostrophes_specimen(element_alt):
                    return True
            return False
        except Exception:
            return False

    def duo_befits(self, first: str, second: str) -> bool:
        drive = self.drive
        admission = drive.admission
        try:
            several = bool(drive.needs.items())
            slot = self.commodity_relating(first) if several else None
            if several and (slot is None or self.commodity_relating(second) != slot):
                return False
            if stringent_feat(first, second) or equal_feat(first, second) or admission.equal_column(slot, first, second):
                return False

            def content(prop: str) -> set[str]:
                return {wq for wq in feat_vocables(prop) if len(wq) >= 3 and wq not in ANCHOR_VOCABLES}
            if (content(first) & content(second)) or (admission.column_measurements(slot, first) & admission.column_measurements(slot, second)):
                return False
            # a still-open twin of the second key would compete with it
            menu = (admission.panel_jotting.get(slot) or (0, ""))[1]
            for element_alt in menu.splitlines():
                key = element_alt.strip("- ").split(" (in ")[0].strip()
                if not key or stringent_feat(key, first) or stringent_feat(key, second):
                    continue
                if admission.duplicate_relating(slot, key, second) and admission.strategy_access(slot, key):
                    return False
            return True
        except Exception:
            return False

    def waitlist_latter(self, prop: str) -> None:
        if not prop:
            return
        waitlist = self.awaiting_runnersup
        if prop.lower() not in {maintained.lower() for maintained in waitlist}:
            waitlist.append(prop)

    def _commodity_badge(self, label: str, failsafe: Any = None) -> str:
        tag = re.search(r"\(([^)]+)\)", label)
        if tag:
            return tag.group(1).strip()
        if failsafe is None:
            return str(self.drive.needs.browsing.get("product") or "item")
        return failsafe

    @staticmethod
    def _bare_solicit(prop: str, product: str) -> str:
        naked = prop.split('(')[0].strip()
        return f"Which {naked} would you like for the {product}?"

    def _overly_opening(self, collections: int, urgent: bool, impulse: bool, ceiling: int) -> bool | None:
        drive = self.drive
        admission = drive.admission
        beat = drive.channel.beat
        if collections < SOLICIT_AFTERWARD_COLLECTIONS and not urgent and not impulse:
            # too little read so far: hold questions for the first steps, a little longer while steps are plentiful
            if beat < SOLICIT_AFTERWARD_PHASES:
                return True
            if beat < SOLICIT_AFTERWARD_PHASES + SOLICIT_LATITUDE_PHASES and drive.maneuvers_residual() > 20:
                return True
        if drive.maneuvers_residual() <= max(10, admission.checkout_expected()) or drive.channel.extra_alt() < SEALING + 40:
            drive.soliciting = False
            return True
        if self.strategy_tarry is not None and collections <= self.strategy_tarry and not impulse:
            return True
        ultimate = self.solicited_upon_collections
        if ultimate is None or collections > ultimate or impulse or urgent:
            return False
        if drive.maneuvers_residual() <= admission.checkout_expected() + 2 * max(0, ceiling - drive.txts_delivered) + 14:
            return False
        return drive.channel.beat - self.solicited_upon_beat < 4 and not drive.needs.items()

    def _genre_panels(self, feat_lineup: str, exhibited: str):
        drive = self.drive
        stash, admission = drive.stash, drive.admission
        captions = stash.aisles_browse()
        houses = drive.shelf.hub_aisles()
        universally_hub = set().union(*houses)
        personal = [label for label in captions if label.strip().lower() in universally_hub]
        if all(houses):
            order = personal or captions
        else:
            order = personal + [label for label in captions if label not in personal]
        itemized = []
        for label in order[:4]:
            menu = stash.feat_panel(label)[0]
            if menu:
                itemized.append((label, menu))
        entire = "\n".join(f"CATEGORY {label}:\n{menu}" for label, menu in itemized) or feat_lineup
        pruned = "\n".join(f"CATEGORY {label}:\n{admission.panel_shorten(menu, stash.elaborate_browses(label))}"
                            for label, menu in itemized) or exhibited
        return entire, pruned

    def _strategy_directive(self, unspent: int, feat_lineup: str, collections: int):
        drive = self.drive
        needs, admission = drive.needs, drive.admission
        components = [SOLICIT_STRATEGY.replace("QUESTION_COUNT", str(unspent)),
                 ENSUING_MONO.replace("REMAINING", str(unspent)),
                 "\n\nTHE SHOPPER'S REQUEST:\n" + drive.request,
                 "\n\nREQUIREMENTS SO FAR:\n" + needs.prose_corpus()]
        if drive.questions_log:
            components.append("\n\nALREADY ASKED (each with the shopper's answer):\n" + "\n".join(drive.questions_log))
        shall = needs.browsing.get("ask_first")
        if shall:
            components.append("\n\nMUST BE LEARNED FROM THE SHOPPER: " + "; ".join(str(item) for item in shall))
        components.append(self.strategy_comments(None))
        try:
            hints = admission.access_cues()
        except Exception:
            hints = []
        if hints:
            components.append("\n\nSTORY HINTS (details from the request's story, each paired with the property it most likely "
                         "points to; none is a requirement, though the hidden need is usually among them):\n"
                         + "\n".join(f"- {admission.cue_stamp(f_alt)}" for f_alt in hints[:8]))
        exhibited = admission.panel_shorten(feat_lineup, collections)
        rows = needs.items()
        if rows:
            listing = " | ".join(f"{n}) {row['product']} x{row['quantity']}" for n, row in enumerate(needs.items(), start=1))
            components.append("\n\nTHE REQUESTED PRODUCTS: " + listing)
            feat_lineup, exhibited = self._genre_panels(feat_lineup, exhibited)
        components.append("\n\nPROPERTIES IN THE LISTINGS' DETAILS SEEN SO FAR (property name, how many listings carry it, "
                     "sample values):\n" + (exhibited or "(nothing read yet)"))
        return "".join(components), feat_lineup

    def _scrutinize_strategy(self, answer, crisp) -> list[dict[str, Any]]:
        drive = self.drive
        several = bool(drive.needs.items())
        kept = []
        for plan in (answer or {}).get("questions") or []:
            if not isinstance(plan, dict) or not str(plan.get("question") or "").strip():
                continue
            standby = str(plan.get("second") or "").strip()
            if re.search(r"none|^-$|n/a", standby.lower()):
                standby = ""
            if not crisp(plan.get("property")):
                # the lead property is spent: its runner-up may still stand alone
                if standby and crisp(standby):
                    kept.append({"property": standby, "question": self._bare_solicit(standby, self._commodity_badge(standby))})
                continue
            element_alt = dict(plan, second=standby if standby and crisp(standby) else "")
            if standby and drive.admission.bare_cue(self.commodity_relating(standby) if several else None, standby):
                lead = str(plan.get("property") or "").strip()
                element_alt["question"] = self._bare_solicit(lead, self._commodity_badge(lead))
            kept.append(element_alt)
        return kept

    def _designated_commodity_inquiry(self, entity):
        prop = str(entity.get("property") or "")
        product = self._commodity_badge(prop, "")
        words = re.findall(r"[a-z]{3,}", product.lower())
        if not product or not words:
            return entity
        couched = str(entity.get("question") or "").lower()
        if any(wq.rstrip("s") in couched for wq in words):
            return entity
        return dict(entity, question=self._bare_solicit(prop, product))

    def _alongside_latter(self, label, second, unspent):
        drive = self.drive
        lead = label[0]
        slot = self.commodity_relating(lead) if drive.needs.items() else None
        seal = f"ask_{drive.txts_delivered + 1}_{drive.channel.beat}"
        product = self._commodity_badge(lead)
        if unspent == 1 and not self.nil_duos and self.duo_befits(lead, second) and drive.admission.cue_charted(slot, lead):
            # the last message left: both properties go out in one sentence
            dual = (f"Which {lead.split('(')[0].strip()} would you like for the {product}, "
                    f"and which {second.split('(')[0].strip()}?")
            return self.utterance_trigger(dual, seal), (lead + "; " + second, dual), ""
        solo = self._bare_solicit(lead, product)
        return self.utterance_trigger(solo, seal), (lead, solo), second

    def _suggest(self, automated, pending, feat_lineup, unspent):
        drive = self.drive
        needs = drive.needs
        suggestion, label, divide_latter = None, None, ""
        stringent_commence = needs.browsing.get("ask_first") and not drive.questions_log
        for entity in automated[:2]:
            if needs.items():
                entity = self._designated_commodity_inquiry(entity)
            if not pending and not stringent_commence and self.designations_further(
                    str(entity.get("question") or ""), str(entity.get("property") or ""), feat_lineup,
                    str(entity.get("second") or "")):
                # the wording leaks another property or a sample value: fall back to the plain form
                prop = str(entity.get("property") or "")
                entity = dict(entity, question=self._bare_solicit(prop, self._commodity_badge(prop)))
            couched = str(entity.get("question")).strip()
            suggestion = self.utterance_trigger(couched, f"ask_{drive.txts_delivered + 1}_{drive.channel.beat}")
            if not suggestion:
                continue
            label = (str(entity.get("property")), couched)
            standby = str(entity.get("second") or "").strip()
            if not standby or standby.lower() == label[0].lower() or re.search(r"none|^-$|n/a", standby.lower()):
                break
            suggestion, spliced, residue = self._alongside_latter(label, standby, unspent)
            if not suggestion:
                continue
            label, divide_latter = spliced, residue
            break
        return suggestion, label, divide_latter

    def _panel_foundation(self):
        stash = self.drive.stash
        failsafe = stash.feat_panel()[0], stash.elaborate_browses()
        personal = self.drive.shelf.personal_aisle()
        if personal:
            return stash.feat_panel(personal)[0], stash.elaborate_browses(personal)
        return failsafe

    def _scheduled_response(self, directive, pending, collections):
        if pending:
            persisted = (self.awaiting_prose or {}).pop(pending.lower(), "")
            couched = persisted or self._bare_solicit(pending, self._commodity_badge(pending))
            return {"questions": [{"property": pending, "question": couched}]}
        if MY_SOLICIT:
            we = self.our_plan()
            if we is False:
                return False
            if we.get("questions"):
                return we
        channel = self.drive.channel
        answer = channel.canvass(directive, 2500, SCHEDULER)
        if answer is None and channel.misses > 0:
            self.strategy_tarry = collections
            return False
        return answer

    def our_plan(self):
        drive = self.drive
        needs = drive.needs
        stash, admission = drive.stash, drive.admission
        names = [str(row["product"]) for row in needs.items()] or [str(needs.browsing.get("product") or "item")]
        satchel = len(names) > 1
        houses = drive.shelf.hub_aisles() if satchel else []
        menus = []
        for slot in range(len(names)):
            if satchel:
                shelves = sorted(houses[slot] if slot < len(houses) else [])
                facts = [discovered for label in shelves for discovered in stash.datum_collections(label)]
            else:
                hub = (drive.shelf.hub_aisles() or [set()])[0]
                if not hub and drive.maneuvers_residual() > admission.checkout_expected() + 12:
                    # nothing of the product's own kind read yet: questions drawn from other shelves would name none of its wishes
                    drive.diary.append({"kind": "our_plan", "wait": "no home shelf yet"})
                    return False
                if hub:
                    facts = [discovered for label in sorted(hub) for discovered in stash.datum_collections(label)]
                else:
                    facts = stash.datum_collections(drive.shelf.personal_aisle())
            crowns = (line.partition(":")[0] for line in needs.written)
            concluded = {solicit_designation(head.split("(")[0]) for head in crowns if not satchel or self.commodity_relating(head) == slot}
            menus.append(solicit_panel(facts, concluded, satchel)[:SOLICIT_PANEL])
        if satchel and not all(menus):
            # some product has no listing read yet: wait for one while steps allow, otherwise hand the turn to the planner
            empty = [names[slot] for slot, menu in enumerate(menus) if not menu]
            tarrying = drive.maneuvers_residual() > admission.checkout_expected() + 12
            drive.diary.append({"kind": "our_plan", "empty": empty, "wait": tarrying})
            return False if tarrying else {}
        grades = self.solicit_grades(names, menus)
        backlogs = []
        for slot, menu in enumerate(menus):
            select = grades.get(f"P{slot + 1}") or {}
            if isinstance(select, dict):
                ranked = [str(key) for key in select.get("ranked") or []]
                story = [str(key) for key in select.get("story") or []]
            else:
                ranked, story = [], []
            backlogs.append(solicit_blend(ranked[:SOLICIT_PRIMARY_SELECTS], story, [key for key, _, _ in menu]))
        # every round opens with the product asked about least so far, answered or not
        delivered = self._reckoning_solicited(len(names))[1] if satchel else [0]
        stints = sorted(range(len(backlogs)), key=lambda slot: (delivered[slot] if slot < len(delivered) else 0, slot))
        questions = []
        for locus in range(max((len(waitlist) for waitlist in backlogs), default=0)):
            for slot in stints:
                waitlist = backlogs[slot]
                if locus >= len(waitlist):
                    continue
                key = waitlist[locus]
                prop = solicit_designation(key)
                samples = next((discovered for itemized, _, discovered in menus[slot] if itemized == key), [])
                questions.append({"property": f"{prop} ({names[slot]})" if satchel else prop,
                                  "question": self.my_inquiry(prop, names[slot], samples, satchel)})
        drive.diary.append({"kind": "our_plan", "menus": [[key for key, _, _ in menu][:8] for menu in menus],
                        "first": [w_alt["property"] for w_alt in questions[:4]]})
        return {"questions": questions}

    def solicit_grades(self, names, menus):
        stamp_alt = tuple(tuple(key for key, _, _ in menu[:12]) for menu in menus)
        jotting = getattr(self, "rank_memo", None)
        recycle = bool(jotting) and (jotting[0] == stamp_alt or jotting[2] >= SOLICIT_GRADINGS)
        if recycle or not any(menus):
            return jotting[1] if jotting else {}
        partitions = []
        for n, (name, menu) in enumerate(zip(names, menus), start=1):
            rows = "\n".join(f"- {key} ({bearers} listings carry it): {', '.join(samples)}" for key, bearers, samples in menu)
            partitions.append(f"## P{n}: {name}\n" + rows)
        directive = SOLICIT_GRADE + "\n\n# The request\n" + self.drive.request + "\n\n# The products\n" + "\n\n".join(partitions)
        answer = self.drive.channel.canvass(directive, 1500, SOLICIT_SPOT) or {}
        self.rank_memo = (stamp_alt, answer, (jotting[2] if jotting else 0) + 1)
        return answer

    @staticmethod
    def my_inquiry(prop: str, product: str, samples: list[str], satchel: bool) -> str:
        # shorter questions get answered more often, so sample values are added only while the text stays short
        bare_alt = f"What {prop} do you need for the {product}?"
        if satchel:
            return bare_alt
        for maximal in (2, 1):
            exemplars = [value for value in samples if len(value) <= 24][:maximal]
            rear = f", such as {' or '.join(exemplars)}" if exemplars else ""
            text = f"What {prop} do you need for the {product}{rear}?"
            if len(text) <= SOLICIT_SYMBOLS:
                return text
        return bare_alt

    def _nil_suggestion(self, collections: int, impulse: bool) -> bool:
        drive = self.drive
        if collections < SOLICIT_AFTERWARD_COLLECTIONS and drive.maneuvers_residual() > 12 and not impulse:
            self.strategy_tarry = collections
            return False
        self.vacant_strategies += 1
        if drive.maneuvers_residual() > 12 and self.vacant_strategies <= 1:
            self.strategy_tarry = collections
        else:
            drive.soliciting = False
        return False

    def _deliver_inquiry(self, suggestion, label):
        drive = self.drive
        needs = drive.needs
        drive.chat.append({"role": "assistant", "content": f"Next step: asking the shopper about {label[0]}.",
                         "tool_calls": [suggestion]})
        censuses = len(needs.uncovered), len(needs.indifferent) + needs.pegged_perceived
        self.soliciting_caption = label[0]
        try:
            rejoinders = drive.forward([suggestion], automated=True)
        finally:
            self.soliciting_caption = ""
        self.subsequent_rejoinders(rejoinders)
        return rejoinders, censuses

    def broach_scheduled(self, impulse: bool = False) -> bool:
        drive = self.drive
        needs, admission = drive.needs, drive.admission
        ceiling = drive.solicit_constraint()
        if not drive.role("message") or drive.txts_delivered >= ceiling:
            drive.soliciting = False
            return False
        feat_lineup, collections = self._panel_foundation()
        # before any listing is read, a question names no listed property and so draws out no hidden need: only the budget is worth it
        urgent = bool(needs.browsing.get("ask_first")) and not drive.questions_log and not be_numeral(needs.disburse_top)
        if self._overly_opening(collections, urgent, impulse, ceiling):
            return False
        if drive.shelf.prior_browse():
            return True
        if not drive.questions_log and not drive.txts_delivered and not urgent and not impulse and admission.first_reads():
            return True
        unspent = ceiling - drive.txts_delivered
        directive, feat_lineup = self._strategy_directive(unspent, feat_lineup, collections)
        solicited_feats = {component.strip().lower() for element_alt in drive.questions_log
                       for component in element_alt.split(":", 1)[0].strip("- ").split(";")}

        def crisp(prop: Any) -> bool:
            text = str(prop or "").strip()
            minor = text.lower()
            if not text or minor in solicited_feats or trash_selector(text) or minor in self.disbursed_solicits:
                return False
            slot = self.commodity_relating(text) if needs.items() else None
            return not admission.bare_cue(slot, text) and admission.strategy_access(slot, text)
        self.awaiting_runnersup = [pz for pz in self.awaiting_runnersup or [] if crisp(pz)]
        pending = self.awaiting_runnersup[0] if self.awaiting_runnersup else ""
        answer = self._scheduled_response(directive, pending, collections)
        if answer is False:
            return False
        automated = self._scrutinize_strategy(answer, crisp)
        if needs.items() and not pending:
            automated = self.ration_primary(automated, crisp)
            lingered, self.ration_tarry = self.ration_tarry, False
            if lingered:
                return True
        suggestion, label, divide_latter = self._suggest(automated, pending, feat_lineup, unspent)
        if not suggestion:
            return self._nil_suggestion(collections, impulse)
        self.solicited_upon_collections, self.solicited_upon_beat = collections, drive.channel.beat
        rejoinders, (discovered_preceding, released_preceding) = self._deliver_inquiry(suggestion, label)
        solicited_caption = label[0]
        lead = solicited_caption.split(";")[0].strip()
        if pending and lead.lower() == pending.lower():
            self.awaiting_runnersup = [pz for pz in self.awaiting_runnersup or [] if pz.lower() != pending.lower()]
        perceived_aux, needs.aux_ultimate = needs.aux_ultimate, None
        label, note = self._browse_rejoinder(label, perceived_aux, discovered_preceding, released_preceding, divide_latter)
        if perceived_aux is None and len(needs.uncovered) == discovered_preceding:
            # nothing revealed: remember the quiet question and line up its story twin when it has one
            self.still_solicits.append(lead)
            duplicate = "" if ";" in solicited_caption else admission.still_resolicit(solicited_caption)
            if duplicate:
                self._deposit_primary(duplicate)
                note += f" [{duplicate}: the story twin of this one, asked next]"
        if rejoinders:
            body = rejoinders[0][1]
            drive.questions_log.append(f"- {label[0]}: asked \"{label[1]}\" -> {body.split(chr(10) + '[status]')[0][:400]}{note}")
        self.jettison_wished(solicited_caption)
        drive.soliciting = drive.txts_delivered < ceiling
        return True

    def _deposit_primary(self, prop: str) -> None:
        minor = prop.lower()
        self.awaiting_runnersup = [prop] + [pz for pz in self.awaiting_runnersup or [] if pz.lower() != minor]

    def _preserve_access(self, prop: str) -> None:
        if prop.lower() in {maintained.lower() for maintained in self.access_duo}:
            return
        self.access_duo.append(prop)

    def _mono_upon_u_moment(self, label, divide_latter):
        admission = self.drive.admission

        def bare_alt(text: Any) -> str:
            return " ".join(re.findall(r"[a-z0-9]+", str(text).lower()))
        self.nil_duos = True
        components = [pz.strip() for pz in label[0].split(";") if pz.strip()] or [label[0]]
        first = components[0]
        key = first.lower()
        bare_w = self._bare_solicit(first, self._commodity_badge(first))
        alone = len(components) == 1
        alternate = admission.intersect_alternate(first) if alone else ""
        if alternate:
            # the story hint offers another property: that one goes next instead
            self.resolicited.add(key)
            self.disbursed_solicits.add(key)
            self._deposit_primary(alternate)
            note = f" [{first}: no answer; next comes {alternate}, the alternative from its story hint]"
        elif key not in self.resolicited and bare_alt(bare_w) != bare_alt(label[1]) and not (
                alone and admission.maintains_bare(label[1], bare_w)):
            self.resolicited.add(key)
            self._deposit_primary(first)
            note = f" [{first}: no answer; asked again on its own next]"
        else:
            rerun = admission.separate_solicit(first) if key not in self.resolicited else None
            if not rerun:
                self.disbursed_solicits.add(key)
                note = f" [{first}: no answer; it will not be asked again]"
            else:
                aka, couched = rerun[0], rerun[1]
                self.resolicited.update({key, aka.lower()})
                self.awaiting_prose[aka.lower()] = couched
                self._deposit_primary(aka)
                equal_designation = re.split(r"\s*\(", aka)[0].strip().lower() == re.split(r"\s*\(", first)[0].strip().lower()
                if not equal_designation:
                    self.disbursed_solicits.add(key)
                note = f" [{first}: no answer; asked again next " + ("through its values]" if equal_designation else f"as {aka}]")
        residues = components[1:] + ([divide_latter] if divide_latter else [])
        for further in residues:
            self._preserve_access(further)
        return (" and ".join(components) + " [not answered] [one at a time]", label[1]), note

    def _browse_rejoinder(self, label, perceived_aux, discovered_preceding, released_preceding, divide_latter):
        needs = self.drive.needs

        def bare_alt(text: Any) -> str:
            return " ".join(re.findall(r"[a-z0-9]+", str(text).lower()))

        def designated_amidst_novel(prop: str) -> bool:
            # did a need found since the question carry one of the property's longer words?
            discovered_prose = " ".join(needs.uncovered[discovered_preceding:]).lower()
            return any(wq in discovered_prose for wq in re.findall(r"[a-z]{4,}", prop.split("(")[0].lower()))
        if perceived_aux is not None and bare_alt(label[1]) and bare_alt(label[1]) in bare_alt(perceived_aux):
            return self._mono_upon_u_moment(label, divide_latter)
        if ";" in label[0]:
            lead = label[0].split(";")[0].strip()
            second = label[0].split(";", 1)[1].strip()
            if len(needs.uncovered) > discovered_preceding:
                if designated_amidst_novel(second):
                    return (lead, label[1]), ""
                self.waitlist_latter(second)
                return (lead, label[1]), f" [{second}: no answer, still open]"
            released_presently = len(needs.indifferent) + needs.pegged_perceived
            if released_presently > released_preceding:
                # the "don't mind" belongs to the first half only; the second stays open
                bare_latter = second.split("(")[0].strip().lower()
                needs.indifferent = [pz for pz in needs.indifferent if pz.strip().lower() not in (second.lower(), bare_latter)]
                self._preserve_access(second)
                return (lead, label[1]), f" [{second}: no answer, still open]"
            if released_presently == released_preceding:
                self.nil_duos = True
                for component in label[0].split(";"):
                    if component.strip():
                        self._preserve_access(component.strip())
                return (label[0].replace(";", " and") + " [not answered]", label[1]), ""
        if divide_latter and not designated_amidst_novel(divide_latter):
            self._preserve_access(divide_latter)
        return label, ""

    def rejoinder_reverse(self, prose_corpus: str) -> bool:
        drive = self.drive
        channel = drive.channel
        if not drive.role("message") or drive.txts_delivered >= drive.solicit_constraint() or drive.maneuvers_residual() <= 4 or channel.extra_alt() < SEALING:
            return False
        # while still asking, the last message stays reserved for a question unless time is short
        if drive.soliciting and drive.txts_delivered >= drive.solicit_constraint() - 1 and drive.maneuvers_residual() > 10 and channel.extra_alt() >= SEALING + 40:
            return False
        suggestion = self.utterance_trigger(prose_corpus, f"reply_{channel.beat}")
        if not suggestion:
            return False
        drive.chat.append({"role": "assistant", "content": "Answering the shopper's question.", "tool_calls": [suggestion]})
        self.subsequent_rejoinders(drive.forward([suggestion], automated=True))
        return True

    def subsequent_rejoinders(self, reactions: list[tuple[str, str]]) -> None:
        drive = self.drive
        terminal = len(reactions) - 1
        for n, (cuid, body) in enumerate(reactions):
            content = body + "\n" + drive.standing_orderline() if n == terminal else body
            drive.chat.append({"role": "tool", "tool_call_id": cuid, "content": content})


class RoutineSealer:
    def __init__(self, drive):
        self.drive = drive
        self.rebrowse = None
        self.tidings_browse = set()
        self.routine_hunted = None
        self.routine_additional = None
        self.routine_impacts = {}
        self.routine_befit = {}

    # ---- arguments for the closing tools, built from their schemas ----
    def framework_operands(self, tool: str, text: str, kind: str = "") -> dict[str, Any]:
        contour = self.drive.contours.get(tool) or {}
        props = contour.get("properties") or {}
        signal = REJECT_VOCABLES if kind == "decline" else CHECKOUT_VOCABLES if kind == "order" else ""

        def cropped(brief: dict[str, Any]) -> str:
            return text[: brief.get("maxLength", len(text))]

        out: dict[str, Any] = {}
        for name in contour.get("required") or []:
            brief = props.get(name) or {}
            announced = brief.get("type")
            if brief.get("enum"):
                suitable = [sel for sel in brief["enum"] if signal and isinstance(sel, str) and re.search(signal, sel.lower())]
                out[name] = (suitable or brief["enum"])[0]
            elif announced == "string" or "string" in (announced or []):
                out[name] = cropped(brief)
            elif announced == "boolean":
                out[name] = True
            elif announced in ("integer", "number"):
                out[name] = max(1, brief.get("minimum", 1))
        if len(props) == 1:
            (solely, brief), = props.items()
            bare_prose = isinstance(brief, dict) and brief.get("type") == "string" and "enum" not in brief
            if bare_prose and solely not in out:
                out[solely] = cropped(brief)
        return out

    def sanitize_reject(self, trigger_operands: dict[str, Any], trigger_designation: str) -> dict[str, Any]:
        configuring = self.drive.needs.browsing.get("decline_no_names")
        if configuring is False:
            return trigger_operands
        ids = self.drive.stash.ids

        def designations_uno_article(crude: Any) -> bool:
            if not isinstance(crude, str):
                return False
            return any(dab and dab in ids for dab in re.split(r"[\s.,;:!?()\[\]{}<>\"'`]+", crude))

        if not configuring and not any(designations_uno_article(crude) for crude in trigger_operands.values()):
            return trigger_operands
        props = (self.drive.contours.get(trigger_designation) or {}).get("properties") or {}

        def emptied(name: str, crude: Any) -> Any:
            brief = props.get(name) or {}
            if not isinstance(crude, str) or "enum" in brief:
                return crude
            return REGRET[: brief.get("maxLength", len(REGRET))]

        return {name: emptied(name, crude) for name, crude in trigger_operands.items()}

    # ---- fallback plan written by plain code when no model managed to read the request ----
    def code_reading(self) -> dict[str, Any]:
        request = self.drive.request
        minor = request.lower()
        plan: dict[str, Any] = {"code_reading": True, "requirements": [], "items": [], "quantity": 1, "rules": [],
                                "soft_wants": [], "ask_first": [], "hints": [], "references": [], "priority_detail": ""}
        plan["price_cap"], plan["currency"] = self.allowance_amidst(request)
        plan["in_stock_only"] = re.search(r"\bin[- ]stock\b|\bon hand\b|\bon the shelf\b|\bsold[- ]out\b|\bback[- ]?order|"
                                          r"\bout of stock\b|\bavailable now\b", minor) is not None
        desires_inexpensive = re.search(r"\bcheapest\b|\blowest[- ](?:price|priced|cost)\b|\bleast expensive\b|"
                                r"\blowest\b[^.]{0,20}\bprice\b|\bsmallest (?:total|price)\b", minor)
        plan["priority"] = "none" if desires_inexpensive is None else "cheapest"
        txt_top = self.constraint_amidst(minor, r"messages?|questions?|notes?|repl(?:y|ies)|texts?")
        if txt_top is not None:
            plan["message_budget"] = txt_top
        maneuver_top = self.constraint_amidst(minor, r"steps?|moves?|actions?|(?:tool )?calls?|turns?")
        if maneuver_top is not None and maneuver_top >= PRIMARY_CHECKOUT_THROUGH + 3:
            plan["step_budget"] = maneuver_top
        plan["roles"] = utility_functions = self.routine_functions()
        seeker = utility_functions.get("search")
        itemized = self.enumerated_commodities(seeker)
        if len(itemized) >= 2:
            plan["items"] = [{key: mono[key] for key in ("product", "quantity", "requirements")} for mono in itemized]
            plan["product"] = " and ".join(mono["product"] for mono in itemized)
            plan["requirements"] = [need for mono in itemized for need in mono["requirements"]]
            return plan
        if itemized:
            plan["product"], plan["quantity"] = itemized[0]["label"], itemized[0]["quantity"]
        else:
            plan["product"] = self.nearest_caption(seeker, variety_vocables(request))
            plan["quantity"] = self.metrics_amidst(minor)
        plan["requirements"] = [f"{prop}: {wished_alt}" for prop, wished_alt in self.declared_duos(minor)]
        return plan

    @staticmethod
    def allowance_amidst(text: str) -> tuple[Any, Any]:
        best: tuple[int, float] = (0, 0.0)
        result: tuple[Any, Any] = (None, None)

        def metric_afterward(remainder: str) -> list[str]:
            seized = re.match(r"\s*((?:[a-z][a-z.]*\s*){0,2})", remainder)
            kept = []
            for token in (seized.group(1).split() if seized else []):
                kernel = token.strip(".")
                if kernel not in ANCHOR_VOCABLES and not re.fullmatch(r"or|and|not|tops|max\w*|only|just|is|was", kernel):
                    kept.append(kernel)
            return kept

        for proviso in re.split(r"(?<=[.!?;])\s+", str(text).lower()):
            entire_checkout = 1 if re.search(r"\b(?:order|total|whole|lot|sum|bill|altogether|overall|everything)\b", proviso) else 0
            for discovered in re.finditer(r"([$€£]\s*)?(\d[\d,]*(?:\.\d+)?)", proviso):
                remainder = proviso[discovered.end():]
                if re.match(r"\s*(?:x\b|×|%|:\s*\d|-\s*\d|/\s*\d|(?:[a-z-]+\s+){0,2}?(?:messages?|questions?|notes?|repl\w*|texts?|"
                            r"steps?|moves?|actions?|(?:tool )?calls?|turns?|units?|items?|pieces?|packs?|pairs?|lots?|sets?|of\b|times?)\b)", remainder):
                    continue
                currency = discovered.group(1)
                if not currency and re.search(r"[\d.,:/-]\s*$|\bx\s*$", proviso[:discovered.start()]):
                    continue
                preceding = proviso[max(0, discovered.start() - 60): discovered.start()]
                heft = entire_checkout
                if currency or re.search(r"\b(?:spend\w*|budget\w*|afford\w*|pay\w*|price[ds]?|costs?|cash|money)\b(?:\W+[\w']+){0,4}\W*$", preceding):
                    heft += 3
                if re.search(r"\b(?:under|below|no more than|not more than|up to|within|at most|max\w*|less than|cheaper than|"
                             r"not (?:more|over|above)|limit\w*|reach\w*|go to|to|at|cap\w*|past|exceed\w*|beyond|go over)\W+(?:[\w']+\W+)?$", preceding):
                    heft += 2
                if re.match(r"\s*(?:[a-z.()]+\s*){0,3}?(?:or (?:lower|less|under|below|cheaper)|at (?:the (?:very )?)?most|max\w*|tops\b|"
                            r",?\s*(?:and )?not (?:one|a) \w+ (?:over|more|above))", remainder):
                    heft += 2
                statistic = float(discovered.group(2).replace(",", ""))
                if heft < 2 or (heft, statistic) <= best:
                    continue
                best = (heft, statistic)
                label = " ".join(metric_afterward(remainder)) or (currency or "").strip() or None
                result = (int(statistic) if statistic.is_integer() else statistic, label)
        return result

    @staticmethod
    def constraint_amidst(reduced: str, substantives: str) -> int | None:
        rendered_alt = {**numeral_vocables(), "twelve": 12, "fifteen": 15, "twenty": 20}
        design = (r"\b(\d+|one|two|three|four|five|six|seven|eight|nine|ten|twelve|fifteen|twenty)\s+(?:[a-z-]+\s+){0,2}?(?:"
                   + substantives + r")\b")
        for discovered in re.finditer(design, reduced):
            preceding = reduced[max(0, discovered.start() - 60): discovered.start()]
            afterward = reduced[discovered.end(): discovered.end() + 40]
            clamped_preceding = re.search(r"\b(?:no more than|not more than|up to|at most|only|max\w*|use|using|take|send|allow\w*|receive|get|total of|"
                                      r"limit\w*|inside|within|under|fewer than|less than|you've|you have|have|keep\w*|stick to|go past|past|"
                                      r"exceed\w*|beyond)\W+(?:[\w']+\W+){0,3}$", preceding)
            clamped_afterward = re.match(r"\W*(?:[\w']+\W+){0,3}?(?:in total|total|no more|at most|max\w*|tops|or fewer|or less|allowed|limit)\b", afterward)
            if clamped_preceding or clamped_afterward:
                token = discovered.group(1)
                return int(token) if token.isdigit() else rendered_alt[token]
        return None

    @staticmethod
    def metrics_amidst(reduced: str) -> int:
        discovered = re.search(r"\b(\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s+(?:units?|of (?:them|those|these)|copies)\b", reduced)
        if not discovered:
            return 1
        token = discovered.group(1)
        quantity_alt = int(token) if token.isdigit() else numeral_vocables()[token]
        return quantity_alt if 1 <= quantity_alt <= 99 else 1

    @staticmethod
    def declared_duos(elongate: str) -> list[tuple[str, str]]:
        connecting = r"(?:is|are|has to be|have to be|must be|should be|needs to be|need to be|can be)"
        cease = r"(?=\s*(?:[,.;:!?()]|$)|\s+(?:so|since|as|because|for|to|which|that|nothing|otherwise|but|when|while|if|then|though|please|and (?:the|its|their|a|an|one|two|three|\d))\b)"
        design = (r"(?:\b(?:where|whose)\s+(?:the\s+|its\s+|their\s+)?|(?:\b(?:and|with|but)\s+|[,;:]\s*|^|[.!?]\s+)(?:the|its|their)\s+)"
                   r"([a-z][a-z'&/ -]{1,32}?)\s+" + connecting + r"\s+(?:either\s+|just\s+)?([^,.;:!?()]{1,60}?)" + cease)

        def workable(prop: str, wished_alt: str, succeeding: str) -> bool:
            if not wished_alt or len(wished_alt.split()) > 4 or len(prop.split()) > 4:
                return False
            if cite_prose(wished_alt):
                return False
            if cite_prose(succeeding) and re.search(r"\b(?:page|listing|shown|sold by)\b", succeeding):
                return False
            if re.search(r"\b(?:i|you|we|it|he|she|they|my|your|our|there|this|that|which|what|who|me|us|them|one|ones|price|budget|cost|total|order|stock)\b", prop):
                return False
            return not re.match(r"(?:a|an|the|my|your|our|his|her|their|me|us|for|to|in|on|at|of|see|go|get|be|have|only|too|very|so|not|no|what|how|it|all|"
                                r"mostly|always|never|just|really|still|also|from|about|like|one|ones)\b", wished_alt)

        discovered_duos: list[tuple[str, str]] = []
        for discovered in re.finditer(design, elongate):
            prop, wished_alt = discovered.group(1).strip(), discovered.group(2).strip()
            if workable(prop, wished_alt, elongate[discovered.end(): discovered.end() + 60]):
                discovered_duos.append((prop, wished_alt))
        return discovered_duos

    def hunt_captions(self, tool: str | None) -> list:
        props = (self.drive.contours.get(tool or "") or {}).get("properties")
        if not isinstance(props, dict):
            return []
        choices_alt = [brief["enum"] for brief in props.values()
                 if isinstance(brief, dict) and isinstance(brief.get("enum"), list) and len(brief["enum"]) >= 4]
        return max(choices_alt, key=len) if choices_alt else []

    def enumerated_commodities(self, tool: str | None) -> list[dict[str, Any]]:
        minor = self.drive.request.lower()
        loci = [(discovered.group(0).rstrip("s"), discovered.start(), discovered.end())
                 for discovered in re.finditer(r"[a-z]{3,}", minor) if discovered.group(0) not in ANCHOR_VOCABLES]
        trunks = [locus[0] for locus in loci]
        rendered_alt = numeral_vocables()

        def primary_enumerated(caption_trunks: list[str]) -> tuple[int, int, str] | None:
            extent = len(caption_trunks)
            for ofs in range(len(trunks) - extent + 1):
                if trunks[ofs: ofs + extent] != caption_trunks:
                    continue
                start, end = loci[ofs][1], loci[ofs + extent - 1][2]
                if len(re.findall(r"[a-z0-9]+", minor[start:end])) > 2 * extent:
                    continue
                census_impact = re.search(r"\b(\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s*(?:x\s+|×\s*)?(?:(?:of|[a-z'&-]{3,})\s+){0,4}$",
                                      minor[max(0, start - 56): start])
                if census_impact:
                    return start, end, census_impact.group(1)
            return None

        situated: dict[str, tuple[int, int, int]] = {}
        for label in (str(crude) for crude in self.hunt_captions(tool)):
            caption_trunks = [component.rstrip("s") for component in re.findall(r"[a-z]{3,}", label.lower()) if component not in ANCHOR_VOCABLES]
            if not caption_trunks or label in situated:
                continue
            locus = primary_enumerated(caption_trunks)
            if locus is None:
                continue
            start, end, token = locus
            quantity_alt = int(token) if token.isdigit() else rendered_alt[token]
            if 1 <= quantity_alt <= 99:
                situated[label] = (start, end, quantity_alt)

        sequenced = sorted(situated.items(), key=lambda item: item[1][0])
        out = []
        for ofs, (label, (_, end, quantity_alt)) in enumerate(sequenced):
            if ofs + 1 < len(sequenced):
                rear = minor[end: sequenced[ofs + 1][1][0]]
            else:
                rear = re.split(r"(?<=[.!?])\s", minor[end: len(minor)])[0]
            reduced_caption = label.lower()
            out.append({"product": reduced_caption, "label": label, "quantity": quantity_alt,
                        "requirements": [f"{prop} ({reduced_caption}): {wished_alt}" for prop, wished_alt in self.declared_duos(rear)]})
        return out

    def routine_functions(self) -> dict[str, str]:
        snippets: dict[str, str] = {}
        for element_alt in self.drive.utilitykit:
            proc = element_alt["function"]
            legible = re.sub(r"([a-z])([A-Z])", r"\1 \2", proc["name"]).replace("_", " ").lower()
            snippets[proc["name"]] = legible + " | " + str(proc.get("description") or "").lower()
        contours = self.drive.contours

        def feats_relating(name: str) -> dict[str, Any]:
            obtained = (contours.get(name) or {}).get("properties")
            return obtained if isinstance(obtained, dict) else {}

        def mandatory_relating(name: str) -> list[str]:
            obtained = (contours.get(name) or {}).get("required")
            return [key for key in obtained if key in feats_relating(name)] if isinstance(obtained, list) else []

        def prose_feats(name: str) -> list[str]:
            return [key for key, brief in feats_relating(name).items()
                    if isinstance(brief, dict) and brief.get("type") == "string" and not brief.get("enum")]

        def tails_dual_avenues(name: str, key: str) -> bool:
            brief = feats_relating(name)[key]
            sels = [str(sel).lower() for sel in (brief.get("enum") or [])] if isinstance(brief, dict) else []
            return any(re.search(CHECKOUT_VOCABLES, sel) for sel in sels) and any(re.search(REJECT_VOCABLES, sel) for sel in sels)

        discovered: dict[str, str] = {}
        # one tool whose required choice covers both ordering and declining serves both endings
        for name in snippets:
            if any(tails_dual_avenues(name, key) for key in mandatory_relating(name)):
                discovered.setdefault("order", name)
                discovered.setdefault("decline", name)
        # otherwise the tool's own name decides
        for name in snippets:
            if name in discovered.values():
                continue
            banner = snippets[name].split(" | ")[0]
            if "decline" not in discovered and re.search(REJECT_VOCABLES, banner):
                discovered["decline"] = name
            elif "order" not in discovered and re.search(CHECKOUT_VOCABLES, banner):
                discovered["order"] = name
        # argument-free tools that call themselves final
        for name in snippets:
            if name in discovered.values() or mandatory_relating(name):
                continue
            snippet = snippets[name]
            be_terminal = re.search(r"\bfinal\b", snippet)
            if "decline" not in discovered and be_terminal and re.search(REJECT_VOCABLES, snippet):
                discovered["decline"] = name
            elif "order" not in discovered and be_terminal and re.search(CHECKOUT_VOCABLES, snippet):
                discovered["order"] = name
        # cart tools: one that changes it, one that just shows it
        for name in snippets:
            if name in discovered.values() or not re.search(r"\b(?:cart|basket|bag|trolley)\b", snippets[name]):
                continue
            possesses_vector = any(isinstance(brief, dict) and brief.get("type") == "array" for brief in feats_relating(name).values())
            discovered.setdefault("cart" if mandatory_relating(name) or possesses_vector else "cart_view", name)
        buyer = r"\b(?:shoppers?|customers?|buyers?|clients?|users?)\b"
        speaking = r"\b(?:send\w*|repl\w*|notes?|tell\w*|messag\w*|chat\w*|writ\w*|talk\w*)\b"

        def accords(role: str, signal: str, mono_prose: bool, name: str) -> bool:
            snippet = snippets[name]
            if name in discovered.values() or not re.search(signal, snippet):
                return False
            if role == "search" and re.search(buyer, snippet) and re.search(speaking, snippet):
                return False
            if role == "message" and not re.search(buyer, snippet) and not re.search(r"\b(?:messag\w*|chat\w*)\b", snippet):
                return False
            if mono_prose:
                necessary = mandatory_relating(name)
                return len(necessary) == 1 and necessary[0] in prose_feats(name)
            return bool(prose_feats(name)) or any(isinstance(brief, dict) and brief.get("enum") for brief in feats_relating(name).values())

        for role, signal, mono_prose in (("search", r"\b(?:search\w*|find\w*|look\w*|locat\w*|hunt\w*|brows\w*|quer\w*)\b", False),
                                    ("message", speaking, True),
                                    ("details", r"\b(?:details?|records?|spec\w*|propert\w*|attribut\w*|facts?|full)\b", True),
                                    ("view", r"\b(?:offer|price|stock|availab\w*)\b", True)):
            if role in discovered:
                continue
            for name in snippets:
                if accords(role, signal, mono_prose, name):
                    discovered[role] = name
                    break
        return discovered

    def nearest_caption(self, tool: str | None, wished_alt: set[str]) -> str:
        optimal_caption, optimal_grade = "", (0, 0.0)
        for label in self.hunt_captions(tool):
            words = variety_vocables(label)
            intersection = len(words & wished_alt)
            if intersection:
                rank = (intersection, intersection / max(1, len(words)))
                if rank > optimal_grade:
                    optimal_caption, optimal_grade = str(label), rank
        return optimal_caption

    def routine_measurements(self, kx: int | None) -> list[tuple[str, str]]:
        needs = self.drive.needs
        manifold = needs.items()
        out = []
        for need in needs.feats_residual():
            prop, _, crude = str(need).partition(":")
            if manifold and self.drive.speak.commodity_relating(prop) not in (kx, None):
                continue
            bare_alt = re.sub(r"\((?:stated|revealed[^)]*)\)", "", crude).strip()
            if not bare_alt or cite_prose(bare_alt):
                continue
            concise_plus_favorable = len(re.findall(r"[a-z0-9]+", bare_alt.lower())) <= 5 and not re.search(PRECLUDE_VOCABLES, bare_alt.lower())
            if concise_plus_favorable or favorable_measurement(crude, prop):
                out.append((re.split(r"\s*\(", prop)[0].strip(), bare_alt))
        return out

    def routine_hunttexts(self, kx: int | None) -> list[dict[str, Any]]:
        drive = self.drive
        needs = drive.needs
        seeker = drive.role("search")
        contour = drive.contours.get(seeker or "") or {}
        props = contour.get("properties") if isinstance(contour.get("properties"), dict) else {}
        required = [key for key in contour.get("required") or [] if key in props]
        manifold = needs.items()
        utilize_niche = manifold and kx is not None and kx < len(manifold)
        wished_commodity = str(manifold[kx]["product"]) if utilize_niche else str(needs.browsing.get("product") or "")
        label = self.nearest_caption(seeker, variety_vocables(wished_commodity) or variety_vocables(drive.request))
        description = next((str(element_alt["function"].get("description") or "") for element_alt in drive.utilitykit
                            if element_alt["function"]["name"] == seeker), "")
        clauses = [shard.lower() for shard in re.split(r"[.;]", description)]
        folio_selector = drive.observe.morpheme_selector()
        proses = [key for key, brief in props.items()
                 if isinstance(brief, dict) and brief.get("type") == "string" and not brief.get("enum") and key != folio_selector]
        hunttext_selector = next((key for key in proses if re.search(r"^q$|query|text|keyword|needle|term|word|phrase|string|search", key.lower())),
                         proses[0] if proses else None)
        choices_alt = {key: brief["enum"] for key, brief in props.items()
                 if isinstance(brief, dict) and isinstance(brief.get("enum"), list) and brief["enum"]}
        aisle_selector = max((key for key in choices_alt if len(choices_alt[key]) >= 4), key=lambda key: len(choices_alt[key]), default=None)
        little_choices = [key for key in choices_alt if key != aisle_selector and len(choices_alt[key]) <= 4]
        manner_selector = next((key for key in little_choices if key in required), little_choices[0] if little_choices else None)

        def uttered_jointly(first: str, second: str, clause_alt: str) -> bool:
            return (re.search(r"\b" + re.escape(first.lower()) + r"\b", clause_alt) is not None
                    and re.search(r"\b" + re.escape(second.lower()) + r"\b", clause_alt) is not None)

        def manner_towards(operand: str, grab_ultimate: bool) -> Any:
            sels = choices_alt.get(manner_selector or "", [])
            for sel in sels:
                if any(uttered_jointly(str(sel), operand, clause_alt) for clause_alt in clauses):
                    return sel
            return sels[-1] if grab_ultimate else sels[0]

        def cost_akin(key: str) -> bool:
            minor = key.lower()
            if re.search(r"ceil|max|price|cap$|budget|upper|most", minor):
                return True
            return any(minor in clause_alt and re.search(r"\bpric|\bcost", clause_alt) for clause_alt in clauses)

        top_selector = next((key for key, brief in props.items()
                        if isinstance(brief, dict) and brief.get("type") in ("integer", "number") and not brief.get("enum")
                        and cost_akin(key) and not re.search(r"page|first|offset|skip|count|size|num", key.lower())), None)
        cap = needs.disburse_top
        apiece_niche = needs.censuses()
        metrics = (apiece_niche[kx] if kx is not None and kx < len(apiece_niche) else needs.quantity()) or 1
        top_operand = {}
        if top_selector and be_numeral(cap) and cap > 0:
            severally = int(cap // metrics) if props[top_selector].get("type") == "integer" else round(cap / metrics, 2)
            top_operand = {top_selector: severally}
        mx_sz = props.get(hunttext_selector, {}).get("maxLength") if hunttext_selector and isinstance(props.get(hunttext_selector), dict) else None
        designation_vocables = [component for component in re.findall(r"[a-z0-9]+", wished_commodity.lower()) if len(component) > 2 and component not in ANCHOR_VOCABLES]
        substantive = designation_vocables[-1] if designation_vocables else ""
        precise = [bare_alt for _, bare_alt in self.routine_measurements(kx)
                 if not re.search(r"\bor\b|/|,|\b(?:" + MEASUREMENT_ANCHORS + r")\b", bare_alt.lower())]
        proses_unto_attempt: list[str] = []
        guide_vocables = [component for component in re.findall(r"[a-z0-9]+", precise[0].lower()) if component not in ANCHOR_VOCABLES] if precise else []
        if guide_vocables:
            proses_unto_attempt.append(" ".join(dict.fromkeys(guide_vocables + [substantive])))
            if len(guide_vocables) > 1:
                proses_unto_attempt.append(" ".join(dict.fromkeys([max(guide_vocables, key=len), substantive])))
        if not proses_unto_attempt:
            if designation_vocables:
                proses_unto_attempt.append(" ".join(dict.fromkeys(designation_vocables[:1] + designation_vocables[-1:])))
            else:
                proses_unto_attempt.append(" ".join(sorted(variety_vocables(label))[:2]))

        def abbreviated(phrase: str) -> str:
            while " " in phrase and len(phrase) > mx_sz:
                phrase = phrase.rsplit(" ", 1)[0]
            return phrase[: int(mx_sz)]

        if be_numeral(mx_sz):
            proses_unto_attempt = [abbreviated(phrase) for phrase in proses_unto_attempt]

        def manner_component(key: str, grab_ultimate: bool) -> dict[str, Any]:
            if not manner_selector:
                return {}
            return {manner_selector: manner_towards(key, grab_ultimate)}

        sketches: list[dict[str, Any]] = []
        if hunttext_selector and proses_unto_attempt[0]:
            sketches.append({hunttext_selector: proses_unto_attempt[0], **manner_component(hunttext_selector, False), **top_operand})
        if label and aisle_selector:
            sketches.append({aisle_selector: label, **manner_component(aisle_selector, True), **top_operand})
        if hunttext_selector and len(proses_unto_attempt) > 1:
            sketches.append({hunttext_selector: proses_unto_attempt[1], **manner_component(hunttext_selector, False), **top_operand})

        def populate_mandatory(sketch: dict[str, Any]) -> None:
            for key in required:
                if key in sketch:
                    continue
                brief = props[key] if isinstance(props[key], dict) else {}
                if brief.get("enum"):
                    sketch[key] = brief["enum"][0]
                elif brief.get("type") in ("integer", "number"):
                    sketch[key] = brief.get("default", brief.get("minimum", 1))
                elif brief.get("type") == "string":
                    sketch[key] = proses_unto_attempt[0] or label

        legit: list[dict[str, Any]] = []
        for sketch in sketches:
            populate_mandatory(sketch)
            immutable, faults = OperandValidate.actuate(sketch, contour)
            if faults and top_operand:
                immutable, faults = OperandValidate.actuate({key: vlu for key, vlu in sketch.items() if key not in top_operand}, contour)
            if faults or immutable in legit:
                continue
            legit.append(immutable)
        return legit

    def routine_datarows(self) -> dict[str, dict[str, Any]]:
        observe = self.drive.observe
        ids = self.drive.stash.ids
        rows: dict[str, dict[str, Any]] = {}

        def headline_amidst(transcript: dict[str, Any]) -> str:
            for key, vlu in transcript.items():
                minor = str(key).lower()
                if isinstance(vlu, str) and re.search(r"title|name", minor) and "id" not in minor:
                    return str(vlu)
            return ""

        def cost_amidst(transcript: dict[str, Any]) -> float | None:
            for key, vlu in transcript.items():
                minor = str(key).lower()
                if be_numeral(vlu) and not isinstance(vlu, bool) and re.search(r"price|cost|amount", minor)\
                        and not re.search(r"count|qty|quant|total|units|line|due|payable|sum", minor):
                    return float(vlu)
            return None

        def primitive(vlu: Any) -> bool:
            return isinstance(vlu, (str, int, float)) and not isinstance(vlu, bool)

        def ingest(transcript: dict[str, Any]) -> None:
            lbl = observe.datarow_stamp(transcript)
            if lbl not in ids:
                return
            row = rows.setdefault(lbl, {"shelf": "", "title": "", "price": None, "stock": None, "words": set(), "fields": {}})
            row["shelf"] = observe.aisle_relating(transcript) or row["shelf"]
            row["title"] = headline_amidst(transcript) or row["title"]
            itemized_cost = cost_amidst(transcript)
            if itemized_cost is not None:
                row["price"] = itemized_cost
            for key, vlu in transcript.items():
                supply_situation = observe.selector_supply(key, vlu)
                if supply_situation is not None:
                    row["stock"] = supply_situation
            embedded = [vlu for vlu in transcript.values() if isinstance(vlu, dict)]
            fragments = [str(vlu) for vlu in transcript.values() if primitive(vlu)]
            for internal in embedded:
                fragments.extend(str(xq) for xq in internal.values() if isinstance(xq, (str, int, float)))
            row["words"] |= variety_vocables(" ".join(fragments))
            for internal in embedded:
                row["fields"].update({str(key): str(xq) for key, xq in internal.items() if primitive(xq)})

        for transcript in self.drive.stash.items:
            route_hashmaps(transcript, ingest)
        return rows

    @staticmethod
    def routine_exhibits(row: dict[str, Any], prop: str, value: str, impact_vocables: set[str], read: bool) -> bool:
        concordant = [exhibited for key, exhibited in row["fields"].items() if equal_feat(key, prop)]
        if concordant:
            return any(RoutineSealer.routine_noticed(value, exhibited) for exhibited in concordant)
        if RoutineSealer.routine_noticed(value, row.get("title") or ""):
            return True
        if read:
            return False
        wished_alt = variety_vocables(value)
        return bool(wished_alt) and wished_alt <= impact_vocables

    @staticmethod
    def routine_noticed(value: str, text: str) -> bool:
        target = str(text or "").lower()
        if not target.strip():
            return False

        def extant(alternative: str) -> bool:
            words = [wq for wq in re.findall(r"[a-z]{2,}", alternative) if wq not in ANCHOR_VOCABLES]
            numerals = re.findall(r"\d+(?:\.\d+)?", alternative)
            if not words and not numerals:
                return False
            if not all(re.search(r"\b" + re.escape(wq[:-1] if len(wq) > 4 else wq), target) for wq in words):
                return False
            return all(re.search(r"(?<![\d.])" + re.escape(z) + r"(?![\d.])", target) for z in numerals)

        detached = re.sub(r"\b(?:" + MEASUREMENT_ANCHORS + r")\b", " ", str(value).lower())
        return any(extant(alternative) for alternative in re.split(r"\s*(?:\bor\b|/|,|;)\s*", detached))

    @staticmethod
    def testimony_grade(row: dict[str, Any], prop: str, value: str) -> tuple[int, int]:
        admitted = set()
        for alternative in re.split(r"\s*(?:\bor\b|/)\s*", value.lower()):
            alternative_vocables = hx_vocables(alternative)
            if alternative_vocables:
                admitted.add(" ".join(alternative_vocables))
        best = (2, 3)
        for key, exhibited in row["fields"].items():
            if not equal_feat(key, prop) or not RoutineSealer.routine_noticed(value, exhibited):
                continue
            if " ".join(hx_vocables(exhibited)) in admitted:
                proximity = 0
            elif any(" ".join(hx_vocables(component)) in admitted for component in hx_components(exhibited)):
                proximity = 1
            else:
                proximity = 3
            precise_selector = 0 if feat_selector(key) == feat_selector(prop) else 1
            best = min(best, (precise_selector, proximity))
        if best == (2, 3) and RoutineSealer.routine_noticed(value, row.get("title") or ""):
            return (2, 2)
        return best

    def inspection_hush(self, ongoing: set[str]) -> set[str]:
        drive = self.drive
        referee = drive.referee
        validated = referee.inspection_idents
        if drive.audit_done:
            browses_thereafter = -1 if referee.inspection_browses is None else referee.inspection_browses
            if validated is None or len(drive.stash.read) <= browses_thereafter:
                validated = set(drive.stash.read)
        ultimate_inspection = next((element_alt for element_alt in reversed(drive.diary) if isinstance(element_alt, dict) and element_alt.get("kind") == "audit"), None)
        if ultimate_inspection is None or not isinstance(ultimate_inspection.get("result"), dict):
            return set()
        return set(validated or set()) - set(referee.inspection_befits or []) - ongoing

    def routine_selects(self, evade: set[str] | None = None, keep: set[str] | None = None) -> list[tuple[str, int]]:
        drive = self.drive
        needs, stash = drive.needs, drive.stash
        manifold = needs.items()
        apiece_niche = needs.censuses()
        if manifold:
            shelves = drive.shelf.hub_aisles()
        else:
            shelves = [set(drive.observe.aisle_captions()) or (drive.shelf.hub_aisles() or [set()])[0]]
        rows = self.routine_datarows()
        edition = str(needs.edition)
        declined = {key.split("|")[0] for key, verdict in drive.rulings.items() if verdict and key.split("|")[1:2] == [edition]}
        ongoing = {key.rpartition("|")[0] for key in drive.passed if key.rpartition("|")[2] == edition}
        vetted = self.inspection_hush(ongoing)
        faith_personal = bool(vetted) and drive.shelf.routine_referees_every()
        cap = float(needs.disburse_top) if be_numeral(needs.disburse_top) else None
        solicited_vocables = variety_vocables(drive.request)
        hits = self.routine_impacts or {}
        cheapest = needs.priority() == "cheapest"
        evade = set(evade or ())
        keep = set(keep or ())
        evade |= drive.shelf.inert_orderlines()
        evade |= drive.citecounter.cite_arrows()

        def metrics_towards(niche_nil: int) -> int:
            return apiece_niche[niche_nil] if niche_nil < len(apiece_niche) else 1

        def retains_asunder(fields: dict[str, str], prop: str, prohibited: list[str]) -> bool:
            return any(equal_feat(key, prop) and str(exhibited).strip() and not any(hx_pens(vz, exhibited) for vz in prohibited)
                       for key, exhibited in fields.items())

        ordinal_clusters: list[list[tuple]] = []
        for niche_nil, shelf in enumerate(shelves):
            metrics = metrics_towards(niche_nil)
            slot = niche_nil if manifold else None
            kind = variety_vocables(manifold[niche_nil]["product"] if manifold else needs.browsing.get("product") or "")
            wished_vocables: set[str] = set()
            solid: list[tuple[str, str]] = []
            for prop, bare_alt in self.routine_measurements(slot):
                wished_vocables |= variety_vocables(bare_alt)
                loose = re.search(r"\b(?:" + MEASUREMENT_ANCHORS + r")\b|\d\s*(?:-|to)\s*\d|[+<>]|\b(?:about|around|approx\w*|roughly)\b", bare_alt.lower())
                if not loose:
                    solid.append((prop, bare_alt))
            asunder = [(cite["prop"], [str(vz) for vz in [cite.get("value")] + list(cite.get("goes_with") or []) if str(vz or "").strip()])
                     for cite in drive.shelf.vary_accomplished(slot)]
            cluster = []
            for lbl, row in rows.items():
                if lbl in declined or lbl in drive.refused_lines or lbl in evade or row["stock"] is False:
                    continue
                if shelf and row["shelf"] not in shelf:
                    continue
                if not shelf and not kind & row["words"]:
                    continue
                if any(self.routine_collision(row["fields"], prop, bare_alt) for prop, bare_alt in solid):
                    continue
                if drive.citecounter.cite_precluded(row["fields"], slot):
                    continue
                cost = stash.price(lbl) if row["price"] is None else row["price"]
                if cap is not None and be_numeral(cost) and cost * metrics > cap:
                    continue
                ever_browse = lbl in stash.read
                noticed_vocables = hits.get(lbl) or set()
                fits = all(self.routine_exhibits(row, prop, bare_alt, noticed_vocables, ever_browse) for prop, bare_alt in solid)
                fits = fits and all(retains_asunder(row["fields"], prop, prohibited) for prop, prohibited in asunder)
                grades = [self.testimony_grade(row, prop, bare_alt) for prop, bare_alt in solid] if ever_browse else []
                if lbl in vetted and not (faith_personal and fits and (solid or asunder) and all(max(rq) <= 1 for rq in grades)):
                    continue
                words = row["words"]
                merit = (3 * len(wished_vocables & words) + 2 * len(kind & words) + len(solicited_vocables & words)
                         + 4 * len(wished_vocables & noticed_vocables))
                price = cost if be_numeral(cost) else float("inf")
                testimony = (sum(rq[0] for rq in grades), sum(rq[1] for rq in grades)) if grades else (9, 9)
                cluster.append((0 if lbl in keep else 1, 0 if lbl in ongoing else 1, not fits, cap is not None and not be_numeral(cost),
                             price if cheapest else 0.0, 0 if ever_browse else 1, row["stock"] is not True,
                             (testimony, -merit), price, lbl))
            if not cluster:
                return []
            ordinal_clusters.append(sorted(cluster))

        def select_alongside(checkout_selector) -> tuple[list[tuple[str, int]], list[tuple], float] | None:
            preferred_idents: list[tuple[str, int]] = []
            preferred_datarows = []
            disburse = 0.0
            for niche_nil, cluster in enumerate(ordinal_clusters):
                occupied = None
                for element_alt in sorted(cluster, key=checkout_selector):
                    lbl = element_alt[-1]
                    if any(lbl == obtained for obtained, _ in preferred_idents):
                        continue
                    if lbl in stash.read and drive.referee.column_glance(lbl)["fails"]:
                        continue
                    occupied = element_alt
                    break
                if occupied is None:
                    return None
                metrics = metrics_towards(niche_nil)
                preferred_idents.append((occupied[-1], metrics))
                preferred_datarows.append(occupied)
                disburse += (0.0 if occupied[8] == float("inf") else occupied[8]) * metrics
            return preferred_idents, preferred_datarows, disburse

        for checkout_selector in ((lambda y_alt: y_alt), (lambda y_alt: (y_alt[2], y_alt[3], y_alt[8], y_alt[-1]))):
            upshot = select_alongside(checkout_selector)
            if upshot is None:
                return []
            preferred_idents, preferred_datarows, disburse = upshot
            if cap is None or disburse <= cap:
                self.routine_befit = {element_alt[-1]: not element_alt[2] for element_alt in preferred_datarows}
                return preferred_idents
        return []

    @staticmethod
    def routine_collision(fields: dict[str, str], prop: str, value: str) -> bool:
        concordant = [exhibited for key, exhibited in fields.items() if equal_feat(key, prop)]
        return bool(concordant) and not any(measurement_noticed(value, exhibited) for exhibited in concordant)

    @staticmethod
    def selection_vocables(value: Any) -> set[str]:
        divide_humps = re.sub(r"([a-z])([A-Z])", r"\1 \2", str(value))
        return set(re.findall(r"[a-z]+", divide_humps.lower()))

    def buggy_mould(self) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], str | None, list[str]]:
        drive = self.drive
        contour = drive.contours.get(drive.role("cart") or "") or {}
        props = contour.get("properties") if isinstance(contour.get("properties"), dict) else {}
        sooner = blob_thing(drive.wagon.deposit_mould) if drive.wagon.deposit_mould else None
        append_operands: dict[str, Any] = {}
        eliminate_operands: dict[str, Any] = {}
        for key in contour.get("required") or []:
            brief = props.get(key) if isinstance(props.get(key), dict) else {}
            sels = brief.get("enum") if isinstance(brief.get("enum"), list) else []
            if not sels:
                continue
            appending = [sel for sel in sels if self.selection_vocables(sel) & DEPOSIT_VOCABLES and not self.selection_vocables(sel) & JETTISON_VOCABLES]
            eliminating = [sel for sel in sels if self.selection_vocables(sel) & JETTISON_VOCABLES]
            if isinstance(sooner, dict) and sooner.get(key) in sels:
                deposit_selection = sooner[key]
            elif appending:
                deposit_selection = appending[0]
            else:
                deposit_selection = sels[0]
            append_operands[key] = deposit_selection
            rest_alt = [sel for sel in sels if sel != deposit_selection]
            failsafe = rest_alt[0] if len(sels) == 2 else None
            eliminate_operands[key] = next((sel for sel in eliminating if sel != deposit_selection), failsafe)
        proses = [key for key, brief in props.items() if isinstance(brief, dict) and brief.get("type") == "string" and not brief.get("enum")]
        ident_selector = next((key for key in proses if re.search(r"id|item|product|listing|sku|target|which|number", key.lower())),
                      proses[0] if proses else None)
        census_selectors = [key for key, brief in props.items()
                      if isinstance(brief, dict) and brief.get("type") in ("integer", "number") and not brief.get("enum")]
        return contour, append_operands, eliminate_operands, ident_selector, census_selectors

    def buggy_operands(self, picks: list[tuple[str, int]]) -> list[dict[str, Any]]:
        contour, append_operands, _, ident_selector, census_selectors = self.buggy_mould()
        orderline_brief = self.drive.wagon.orderline_brief()
        if orderline_brief:
            lineup_selector, datarow_ident, datarow_metrics = orderline_brief
            sketches = [{**append_operands, lineup_selector: [{datarow_ident: lbl, datarow_metrics: metrics} for lbl, metrics in picks]}]
        elif ident_selector:
            sketches = []
            for lbl, metrics in picks:
                sketch = {**append_operands, ident_selector: lbl}
                if census_selectors:
                    sketch[census_selectors[0]] = metrics
                sketches.append(sketch)
        else:
            return []
        legit = []
        for sketch in sketches:
            immutable, faults = OperandValidate.actuate(sketch, contour)
            if faults:
                return []
            legit.append(immutable)
        return legit

    def buggy_jettisons(self, lines: list[str]) -> list[dict[str, Any]] | None:
        wagon = self.drive.wagon
        if not lines or wagon.orderline_brief():
            return []
        contour, append_operands, eliminate_operands, ident_selector, census_selectors = self.buggy_mould()
        props = contour.get("properties") if isinstance(contour.get("properties"), dict) else {}
        if not ident_selector:
            return None
        # the way to take a line out does not depend on the line, so settle it once
        if append_operands:
            if any(eliminate_operands.get(key) is None for key in append_operands):
                return None
        else:
            if not census_selectors:
                return None
            floor = (props.get(census_selectors[0]) or {}).get("minimum")
            if be_numeral(floor) and floor > 0:
                return None
        legit = []
        for lbl in lines:
            if append_operands:
                sketch = {**eliminate_operands, ident_selector: lbl}
                if census_selectors:
                    sketch[census_selectors[0]] = max(1, int(wagon.maintained(lbl) or 1))
            else:
                sketch = {ident_selector: lbl, census_selectors[0]: 0}
            immutable, faults = OperandValidate.actuate(sketch, contour)
            if faults:
                return None
            legit.append(immutable)
        return legit

    def newest_rejoinders(self, count: int) -> list[Any]:
        ultimate_envt = next((element_alt for element_alt in reversed(self.drive.diary) if isinstance(element_alt, dict) and element_alt.get("kind") == "env"), None)
        reply = (ultimate_envt or {}).get("reply")
        if not isinstance(reply, dict):
            return []
        numerous = reply.get("observations")
        witnessed = numerous if isinstance(numerous, list) else [reply.get("observation")]
        return list(witnessed)[:count]

    @staticmethod
    def routine_flaw(obv: Any) -> bool:
        detected = [False]

        def examine(transcript: dict[str, Any]) -> None:
            if any(re.search(FLAW_SELECTORS, str(key).lower()) and vlu not in (None, False, "", [], {}) for key, vlu in transcript.items()):
                detected[0] = True

        route_hashmaps(obv, examine, 0, 2)
        return detected[0]

    def tome_impacts(self, calls: list[dict[str, Any]], rejoinders: list[Any], prose_columns: list[str]) -> None:
        drive = self.drive
        journal = self.routine_impacts
        for trigger_operands, obv in zip(calls, rejoinders):
            entered = " ".join(str(trigger_operands.get(key) or "") for key in prose_columns)
            if not entered.strip():
                continue
            hunttext_vocables = variety_vocables(entered)

            def balance(transcript: dict[str, Any], hunttext_vocables: set[str] = hunttext_vocables) -> None:
                lbl = drive.observe.datarow_stamp(transcript)
                if lbl in drive.stash.ids:
                    journal.setdefault(lbl, set()).update(hunttext_vocables)

            route_hashmaps(obv, balance)

    def routine_hunt(self, products: int, additional: bool = False) -> None:
        drive = self.drive
        seeker = drive.role("search")
        props = (drive.contours.get(seeker or "") or {}).get("properties") or {}
        prose_selectors = [key for key, brief in props.items()
                     if isinstance(brief, dict) and brief.get("type") == "string" and not brief.get("enum")]
        maintained_reverse = 2 + 2 * products
        if additional:
            obeyed = list(self.routine_additional or [])
            lot = obeyed[: max(0, drive.maneuvers_residual() - maintained_reverse - 2 * products)]
            bouts = 1
        else:
            apiece_commodity = [self.routine_hunttexts(kx if drive.needs.items() else None) for kx in range(products)]
            lot = []
            # interleave: every product's first query, then every product's second, and so on
            for depth in range(max((len(queries) for queries in apiece_commodity), default=0)):
                for queries in apiece_commodity:
                    if depth < len(queries) and queries[depth] not in lot:
                        lot.append(queries[depth])
            lot = lot[: max(0, drive.maneuvers_residual() - maintained_reverse)]
            obeyed = [queries[0] for queries in apiece_commodity if queries]
            bouts = 1 + ROUTINE_FOLIOS + (2 if drive.needs.priority() == "cheapest" else 0)
        if not lot:
            return
        self.routine_hunted = True
        for bout_nil in range(bouts):
            drive.deliver_fragment([(f"code_search_{n}_{drive.channel.beat}", seeker, trigger_operands) for n, trigger_operands in enumerate(lot)])
            rejoinders = self.newest_rejoinders(len(lot))
            if drive.vocable_variety != "any":
                self.tome_impacts(lot, rejoinders, prose_selectors)
            ensuing_folios = []
            for trigger_operands, obv in zip(lot, rejoinders):
                if trigger_operands in obeyed:
                    folio_trigger = self.ensuing_folio_trigger(trigger_operands, obv)
                    if folio_trigger:
                        ensuing_folios.append(folio_trigger)
            self.routine_additional = ensuing_folios
            if bout_nil == bouts - 1:
                break
            picks = self.routine_selects()
            befit_atlas = self.routine_befit or {}
            if drive.needs.priority() != "cheapest" and picks and all(befit_atlas.get(lbl) for lbl, _ in picks):
                break
            obeyed = ensuing_folios
            lot = ensuing_folios[: max(0, drive.maneuvers_residual() - maintained_reverse - 3 * products)]
            if not lot:
                break

    def ensuing_folio_trigger(self, trigger_operands: dict[str, Any], obv: Any) -> dict[str, Any] | None:
        scrolling = self.drive.observe.folio_operand()
        if not scrolling:
            return None
        key, apiece_folio, minimal, tallest = scrolling
        postings: list[dict[str, Any]] = []

        def amass(transcript: dict[str, Any]) -> None:
            if any(isinstance(vlu, list) and vlu and all(isinstance(xq, dict) for xq in vlu) for vlu in transcript.values()):
                postings.append(transcript)

        route_hashmaps(obv, amass, 0, 3)
        if not postings:
            return None
        page = postings[0]
        onto_folio = max(len(vlu) for vlu in page.values() if isinstance(vlu, list))
        carets = [vlu for name, vlu in page.items() if re.search(r"next|onward|cursor|bookmark|token|continu|more", str(name).lower())]
        overall_discovered = next((vlu for name, vlu in page.items() if be_numeral(vlu) and not isinstance(vlu, bool)
                            and re.search(r"count|total|matched|found|hits", str(name).lower())), None)
        if carets and all(cc in (None, False, "", 0) for cc in carets):
            return None
        if apiece_folio == "token":
            caret = next((cc for cc in carets if isinstance(cc, str) and cc.strip()), None)
            if not caret:
                return None
            fwd = {**trigger_operands, key: caret}
        else:
            herein = trigger_operands.get(key, minimal)
            if not be_numeral(herein):
                herein = minimal
            phase_unto = herein + 1 if apiece_folio else herein + onto_folio
            if tallest is not None and phase_unto > tallest:
                return None
            if not carets:
                encompassed = (herein - minimal + 1) * onto_folio if apiece_folio else herein + onto_folio
                if overall_discovered is None or encompassed >= overall_discovered:
                    return None
            fwd = {**trigger_operands, key: int(phase_unto) if float(phase_unto).is_integer() else phase_unto}
        immutable, faults = OperandValidate.actuate(fwd, self.drive.contours.get(self.drive.role("search") or "") or {})
        return immutable if not faults else None

    def browse_selects(self, picks: list[tuple[str, int]], evade: set[str], keep: set[str] | None) -> list[tuple[str, int]]:
        drive = self.drive
        for _ in range(ROUTINE_BROWSES):
            unbrowsed = [lbl for lbl, _ in picks if lbl not in drive.stash.read]
            mass = drive.shelf.lot_utility()
            amidst_mass = bool(mass) and len(unbrowsed) >= 2
            if not unbrowsed or not (mass or drive.role("details")):
                break
            cost = 1 if amidst_mass else len(unbrowsed)
            if drive.maneuvers_residual() < cost + 2 * len(picks) + 3 or drive.channel.extra_alt() <= 8:
                break
            if amidst_mass:
                fragment = [(f"code_read_{drive.channel.beat}", mass[0], {mass[1]: unbrowsed[: mass[2]]})]
            else:
                browser = drive.role("details")
                fragment = [(f"code_read_{n}_{drive.channel.beat}", browser, trigger_operands)
                         for n, trigger_operands in enumerate(drive.referee.nuances_trigger(lbl) for lbl in unbrowsed) if trigger_operands]
            if not fragment:
                break
            drive.deliver_fragment(fragment)
            crisp = self.routine_selects(evade, keep)
            if crisp == picks:
                break
            picks = crisp
        return picks

    def routine_deposit(self, picks: list[tuple[str, int]]) -> list[str] | None:
        drive = self.drive
        wagon = drive.wagon
        buggy_utility = drive.role("cart")
        wished_alt = {lbl for lbl, _ in picks}
        orderline_brief = wagon.orderline_brief()
        surplus = [line for line in wagon.metrics if line not in wished_alt]
        if orderline_brief:
            modifications = list(picks)
        else:
            modifications = [(lbl, metrics) for lbl, metrics in picks if abs(wagon.maintained(lbl) - metrics) > 1e-9]
        jettisons = self.buggy_jettisons(surplus)
        puts = self.buggy_operands(modifications) if modifications else []
        if jettisons is None or (modifications and not puts):
            return None
        calls = jettisons + puts
        if not calls:
            return []
        if drive.maneuvers_residual() < len(calls) + 1:
            return None
        drive.deliver_fragment([(f"code_cart_{n}_{drive.channel.beat}", buggy_utility, trigger_operands) for n, trigger_operands in enumerate(calls)])
        rejoinders = self.newest_rejoinders(len(calls))
        mislaid: set[str] = set()
        buggy_noticed = None
        for trigger_operands, obv in zip(calls, rejoinders):
            exhibited_metrics = wagon.glance_metrics(obv)
            if exhibited_metrics is not None:
                buggy_noticed = exhibited_metrics
            if self.routine_flaw(obv):
                amidst_rejoinder = [lbl for lbl in wished_alt if lbl in qua_prose(obv)]
                amidst_trigger = [line for line, _ in wagon.lines(trigger_operands) if line in wished_alt]
                mislaid.update(amidst_rejoinder or amidst_trigger or (list(wished_alt) if orderline_brief else []))
        if len(rejoinders) < len(calls):
            return None
        watcher = drive.role("cart_view")
        if buggy_noticed is None and not mislaid and watcher and drive.maneuvers_residual() >= 3:
            drive.deliver_fragment([(f"code_view_{drive.channel.beat}", drive.role("cart_view"), {})])
            exhibited = self.newest_rejoinders(1)
            buggy_noticed = wagon.glance_metrics(exhibited[0]) if exhibited else None
            if exhibited and buggy_noticed is None:
                mislaid.update(lbl for lbl in wished_alt if lbl not in wagon.metrics)
        if buggy_noticed is not None:
            mislaid.update(lbl for lbl in wished_alt if lbl not in buggy_noticed)
        for lbl in mislaid:
            wagon.metrics.pop(lbl, None)
        return sorted(mislaid)

    def routine_revalidate(self, lines: list[str]) -> tuple[bool, set[str]]:
        drive = self.drive
        validator = drive.shelf.supply_utility() or drive.role("view") or drive.role("details")
        nada = (False, set())
        if not lines or not validator:
            return nada
        if drive.maneuvers_residual() < len(lines) + 2 or drive.channel.extra_alt() < 8:
            return nada
        fragment = []
        for n, lbl in enumerate(lines):
            trigger_operands = drive.shelf.fetch_trigger(lbl, validator)
            if trigger_operands:
                fragment.append((f"code_check_{n}_{drive.channel.beat}", validator, trigger_operands))
        if not fragment:
            return nada
        drive.deliver_fragment(fragment)
        rows = self.routine_datarows()
        gone = {lbl for lbl in lines if (rows.get(lbl) or {}).get("stock") is False}
        return True, gone

    def browse_tidings(self, lines: list[str]) -> None:
        drive = self.drive
        tarrying = [line for group, _ in drive.needs.pending or [] for line in group]
        for line in [*lines, *tarrying]:
            if line in self.tidings_browse:
                continue
            self.tidings_browse.add(line)
            statistic, _ = self.allowance_amidst(line)
            if be_numeral(statistic) and statistic > 0:
                drive.needs.disburse_top = statistic
                drive.diary.append({"kind": "code_news", "budget": statistic})

    def routine_buggy(self, evade: set[str] | None = None, keep: set[str] | None = None) -> bool:
        drive = self.drive
        channel = drive.channel
        buggy_utility, seeker = drive.role("cart"), drive.role("search")
        if not buggy_utility or channel.done or channel.snapped:
            return False
        if drive.maneuvers_residual() < 2 or channel.extra_alt() < 4:
            return False
        channel.muffle = True
        try:
            return self.mart_bouts(set(evade or ()), keep, seeker)
        except Exception as failure:
            drive.diary.append({"kind": "code_finish_error", "error": f"{type(failure).__name__}: {failure}"[:300]})
            return False

    def mart_bouts(self, evade: set[str], keep: set[str] | None, seeker: str | None) -> bool:
        drive = self.drive
        products = max(1, len(drive.needs.items()))
        for _ in range(4):
            picks = self.routine_selects(evade, keep)
            feeble = not picks or not all(self.routine_befit.get(lbl) for lbl, _ in picks)
            if seeker and not self.routine_hunted and drive.maneuvers_residual() >= 4 and feeble:
                self.routine_hunt(products)
                picks = self.routine_selects(evade, keep)
            elif seeker and self.routine_additional and drive.maneuvers_residual() >= 6 + 2 * products and (
                    feeble or drive.needs.priority() == "cheapest"):
                self.routine_hunt(products, True)
                picks = self.routine_selects(evade, keep)
            picks = self.browse_selects(picks, evade, keep)
            if not picks:
                drive.diary.append({"kind": "code_finish", "picks": [], "result": "nothing to order"})
                return False
            poor = self.routine_deposit(picks)
            if poor is None:
                drive.diary.append({"kind": "code_finish", "picks": picks, "result": "no cart call"})
                return False
            if not poor:
                gone = set()
                if drive.maneuvers_residual() >= len(picks) + 3:
                    _, gone = self.routine_revalidate([lbl for lbl, _ in picks])
                poor = sorted(gone)
            if poor:
                evade |= set(poor)
                drive.diary.append({"kind": "code_swap", "lines": poor})
                continue
            drive.diary.append({"kind": "code_finish", "picks": picks, "cart": dict(drive.wagon.metrics)})
            return bool(drive.wagon.metrics)
        return False

    def checkout_maneuver(self, tool: str) -> dict[str, Any]:
        return {"name": tool, "args": self.framework_operands(tool, "Order the cart.", "order")}

    def reject_maneuver(self, tool: str) -> dict[str, Any]:
        return {"name": tool, "args": self.framework_operands(tool, REGRET, "decline")}

    def halt_terminating(self) -> bool:
        channel = self.drive.channel
        if channel.done or channel.snapped:
            return True
        return self.drive.maneuvers_residual() <= 0 or channel.extra_alt() < 2

    def terminate_presently(self) -> None:
        if self.halt_terminating():
            return
        drive = self.drive
        channel, wagon = drive.channel, drive.wagon
        checkout_utility, reject_utility = drive.role("order"), drive.role("decline")
        evade: set[str] = set()
        keep: set[str] | None = None
        maneuver: dict[str, Any] | None = None
        kind = None

        def jettison_vended_orderlines() -> set[str]:
            nonlocal evade, keep
            try:
                channel.muffle = True
                possessed = list(wagon.metrics)
                gone = drive.shelf.buggy_departed(possessed)
                if gone:
                    evade |= gone
                    keep = set(possessed) - gone
                    drive.diary.append({"kind": "code_dead", "lines": sorted(gone)})
                return gone
            except Exception as failure:
                drive.diary.append({"kind": "cart_gone_error", "error": f"{type(failure).__name__}: {failure}"[:300]})
                return set()

        for stint in range(ROUTINE_TERMINATES):
            if self.halt_terminating():
                return
            if maneuver is None:
                primary_alongside_buggy = checkout_utility and stint == 0
                gone = jettison_vended_orderlines() if primary_alongside_buggy and wagon.metrics else set()
                qua_be = (primary_alongside_buggy and not gone and wagon.utilized
                         and (wagon.metrics or wagon.glance_metrics(wagon.ultimate_glance) is None))
                if qua_be and drive.satchel_concise():
                    # an incomplete basket is never ordered as it stands: code tops it up around the lines kept, or declines
                    qua_be = False
                    keep = set(wagon.metrics) if keep is None else keep
                    drive.diary.append({"kind": "code_partial", "faults": drive.satchel_concise()})
                # (an order needs lines in the cart: when the code's own shopping left the cart empty, the finish is a decline,
                # never a repeated order of nothing)
                if qua_be or (checkout_utility and self.routine_buggy(evade, keep) and wagon.metrics):
                    maneuver = self.checkout_maneuver(checkout_utility)
                elif checkout_utility and stint > 0 and self.afresh_sound(evade):
                    maneuver = self.checkout_maneuver(checkout_utility)
                elif reject_utility:
                    maneuver = self.reject_maneuver(reject_utility)
                else:
                    return
            if drive.trigger_variety(maneuver["name"], maneuver["args"]) == "order":
                # budget nearly spent: send the go-ahead message first while one message and one step remain
                drive.deliver_advance_onward(room=2)
            reply = channel.operate([maneuver])
            drive.diary.append({"kind": "forced_finish", "action": maneuver, "reply": reply})
            kind = drive.trigger_variety(maneuver["name"], maneuver["args"])
            if not reply or reply.get("finished") or channel.done or stint == ROUTINE_TERMINATES - 1 or drive.maneuvers_residual() <= 1:
                break
            channel.muffle = True
            lines = list(wagon.metrics)
            obv = reply.get("observation")
            drive.ingest(maneuver["name"], maneuver["args"], obv, reply, True)
            self.browse_tidings(drive.encompass.buyer_orderlines(maneuver["name"], obv, True))
            maneuver = None
            if kind != "order" or not lines:
                continue
            reverified, gone = self.routine_revalidate(lines)
            if not reverified:
                continue
            drive.refused_lines.difference_update(set(lines) - gone)
            rows = self.routine_datarows()
            invoice = sum(((rows.get(line) or {}).get("price") or 0.0) * wagon.maintained(line) for line in lines)
            constraint = drive.needs.disburse_top
            overly_costly = be_numeral(constraint) and invoice > constraint
            if overly_costly:
                evade, keep = set(gone), None
            elif gone:
                evade, keep = set(gone), set(lines) - gone
            else:
                maneuver = self.checkout_maneuver(checkout_utility)
        if channel.done or not reply or reply.get("finished") or not reject_utility or kind == "decline" or drive.maneuvers_residual() <= 0:
            return
        reattempt = False
        try:
            reattempt = bool(checkout_utility) and kind == "order" and self.afresh_sound(evade)
        except Exception as failure:
            drive.diary.append({"kind": "again_error", "error": f"{type(failure).__name__}: {failure}"[:300]})
        maneuver = self.checkout_maneuver(checkout_utility) if reattempt else self.reject_maneuver(reject_utility)
        drive.diary.append({"kind": "forced_finish", "action": maneuver, "reply": channel.operate([maneuver])})

    def afresh_sound(self, evade: set[str]) -> bool:
        amidst_buggy = set(self.drive.wagon.metrics)
        precluded = set(evade) | self.drive.shelf.inert_orderlines()
        if amidst_buggy and not amidst_buggy & precluded:
            return bool(self.routine_selects(precluded))
        return False

    def snag_aloft(self) -> None:
        drive = self.drive
        channel, needs = drive.channel, drive.needs
        try:
            if channel.muffle or channel.extra_alt() < SEALING:
                return
            notes = []
            browse_upon = drive.admission.browse_upon
            failsafe_strategy = needs.browsing.get("code_reading") and not self.rebrowse and browse_upon is not None
            if failsafe_strategy and channel.ultimate_rejoinder > browse_upon and channel.extra_alt() > SEALING + 60:
                self.rebrowse = True
                answer = channel.hunttext_oracle([{"role": "user", "content": drive.admission.browse_matter}], BROWSER, 3000)
                crisp = blob_thing((answer or {}).get("content") or "") or {}
                workable = isinstance(crisp.get("roles"), dict)
                drive.diary.append({"kind": "re_read", "ok": workable})
                if workable:
                    drive.admission.utilize_browsing(crisp, True)
                    notes.append("[plan updated] The language models answer again, so the request was read a second time; "
                                 "this reading takes the place of the earlier one:\n" + needs.outline(drive.phase_top, drive.txt_top))
            tarrying = needs.pending or []
            if tarrying and channel.ultimate_rejoinder > needs.awaiting_upon and channel.extra_alt() >= SEALING:
                needs.pending = []
                for group, question in tarrying[-3:]:
                    heard = needs.perceive(group, question)
                    if heard:
                        notes.append("[earlier shopper lines, now read] " + heard)
            if not notes:
                return
            text = "\n".join(notes) + "\n" + drive.standing_orderline()
            ultimate = drive.chat[-1] if drive.chat else None
            if ultimate and ultimate.get("role") in ("user", "tool") and isinstance(ultimate.get("content"), str):
                ultimate["content"] += "\n" + text
            else:
                drive.chat.append({"role": "user", "content": text})
        except Exception as failure:
            drive.diary.append({"kind": "catch_up_error", "error": f"{type(failure).__name__}: {failure}"[:300]})

class Spectator:
    """Reads store replies: refusals, shopper pushes, shelf labels, paging and search-word behaviour."""

    def __init__(self, drive):
        self.drive = drive
        self.denial_proses = None
        self.directed_jotting = None
        self.impact_aisle = {}
        self.combed = 0
        self.comb_stint = 0
        self.comb_tested = set()

    def refused(self, trigger_designation: str, obv: Any, ans: dict[str, Any]) -> bool:
        qua_atlas = isinstance(obv, dict)
        if qua_atlas and "refusal" in obv:
            return True
        # only a non-final order call can be refused without an explicit refusal key
        if trigger_designation != self.drive.role("order") or ans.get("finished"):
            return False
        if qua_atlas and obv.get("error") in ("invalid_args", "unknown_tool"):
            return False
        return True

    @staticmethod
    def verbose_proses(obv: Any) -> list[str]:
        """Every string of four or more words nested in obs (depth-limited, first 20 list entries)."""
        proses: list[str] = []

        def tour(vertex: Any, degree: int = 0) -> None:
            if degree > 6:
                return
            if isinstance(vertex, str):
                if len(vertex.split()) >= 4:
                    proses.append(vertex)
            elif isinstance(vertex, dict):
                for kid in vertex.values():
                    tour(kid, degree + 1)
            elif isinstance(vertex, list):
                for kid in vertex[:20]:
                    tour(kid, degree + 1)
        tour(obv)
        return proses

    def checkout_denial_comment(self, obv: Any) -> str:
        drive = self.drive
        drive.refusal_n += 1
        drive.order_vetted = False
        drive.event_seen = True
        surplus = ""
        try:
            surplus = drive.shelf.denial_rebrowse(obv)
        except Exception as exn:
            drive.diary.append({"kind": "refusal_reread_error", "error": f"{type(exn).__name__}: {exn}"[:300]})
        drive.refused_lines.update(drive.wagon.metrics)
        self.denial_proses = {" ".join(str(text).lower().split()) for text in self.verbose_proses(obv)}
        return ("ORDER REFUSED: the episode continues and the cart is unchanged. Make every change the refusal asks for, "
                "fix the cart (remove any line that is no longer acceptable, a sold-out item for example), check stock and "
                "price of the replacement again, then place the order once more." + surplus)

    def buyer_propel_comment(self, obv: dict[str, Any]) -> str | None:
        def crush(text: Any) -> str:
            return " ".join(str(text).lower().split())
        shoved: list[str] = []
        for element_alt in obv.values():
            if not (isinstance(element_alt, list) and element_alt):
                continue
            if all(isinstance(shard, str) for shard in element_alt):
                extended_textstrings(element_alt, shoved)
        if not shoved:
            return None
        preceding = self.denial_proses or set()
        self.denial_proses = None
        if preceding and all(any(crush(shard) in aged for aged in preceding) for shard in shoved):
            return ("This shopper text repeats what the refused finish already carried (dealt with above); it says nothing "
                    "new about this result.")
        self.drive.event_seen = True
        return "This result arrives with a shopper message (a change). Act on it: " + " | ".join(shoved[:4])

    def comments_towards(self, trigger_designation: str, obv: Any, refused: bool) -> str:
        drive = self.drive
        components = []
        if isinstance(obv, dict) and obv.get("error") == "budget_exceeded":
            if trigger_designation == drive.role("message") or drive.txt_top is not None:
                if drive.txt_top is None:
                    drive.txt_top = drive.txts_delivered
                else:
                    drive.txt_top = min(drive.txt_top, drive.txts_delivered)
            components.append("This budget is used up; stop calling this tool.")
        if refused and trigger_designation == drive.role("decline") and drive.last_kind == "decline":
            drive.audit_done = False
            drive.decline_vetted = False
            drive.event_seen = True
            components.append("DECLINE REFUSED: the episode continues because the store thinks something can still be bought. "
                         "Follow the refusal, take one more look at the closest candidates (their own properties, stock and "
                         "price measured against the current requirements) and order the one that fits; decline again only "
                         "when every one of them really fails.")
        elif refused:
            components.append(self.checkout_denial_comment(obv))
        elif isinstance(obv, dict) and len(obv) >= 2 and trigger_designation != drive.role("message"):
            shoved = self.buyer_propel_comment(obv)
            if shoved:
                components.append(shoved)
        return " ".join(components)

    @staticmethod
    def aisle_relating(row: dict[str, Any]) -> str:
        for key, value in row.items():
            if not isinstance(value, str):
                continue
            text = value.strip()
            if not 0 < len(text) <= 60:
                continue
            reduced = str(key).lower()
            if "id" not in reduced and re.search(r"categor|shelf|department|aisle", reduced):
                return text.lower()
        return ""

    @staticmethod
    def selector_supply(key: Any, vlu: Any) -> bool | None:
        """True / False for in stock / not, from one field; None when the field says nothing about stock."""
        reduced = str(key).lower()
        if vlu is None or not re.search(r"stock|avail|on.?hand", reduced):
            return None
        empty = vlu is False or (be_numeral(vlu) and vlu <= 0)
        gone = False
        if re.search(r"stock|avail", reduced):
            if re.search(r"out|sold", reduced):
                gone = vlu is True
            elif empty:
                gone = True
            elif isinstance(vlu, str):
                gone = bool(re.search(r"\b(?:out|sold|none|unavailable|not|no|false)\b", vlu.lower()))
        if empty and re.search(r"on.?hand", reduced):
            gone = True
        if isinstance(vlu, str) and re.search(r"back.?order|pre.?order|discontinu", vlu.lower()):
            gone = True
        return not gone

    @staticmethod
    def supply_situation(row: dict[str, Any]) -> bool | None:
        rulings_alt = [toggle for toggle in (Spectator.selector_supply(key, vlu) for key, vlu in row.items()) if toggle is not None]
        return all(rulings_alt) if rulings_alt else None

    @staticmethod
    def vended_outward(row: dict[str, Any]) -> bool:
        return any(Spectator.selector_supply(key, vlu) is False for key, vlu in row.items())

    def browse_caption_reckoning(self) -> dict[str, int]:
        """How many distinct read items sit on each shelf label (foreign references left out)."""
        stash = self.drive.stash
        apiece_caption: dict[str, int] = {}
        noticed_idents: set[str] = set()
        alien = self.drive.citecounter.cite_alien()

        def reckoning(transcript: dict[str, Any]) -> None:
            lbl = next((vlu for key, vlu in transcript.items()
                          if isinstance(vlu, str) and "id" in str(key).lower() and vlu in stash.read), None)
            if not lbl or lbl in noticed_idents or lbl in alien:
                return
            label = self.aisle_relating(transcript)
            if not label:
                return
            noticed_idents.add(lbl)
            apiece_caption[label] = apiece_caption.get(label, 0) + 1
        for item in stash.items:
            route_hashmaps(item, reckoning)
        return apiece_caption

    def aisle_captions(self) -> set[str]:
        censuses = self.browse_caption_reckoning()
        pinnacle = max(censuses.values(), default=0)
        frontrunners = {label for label, n in censuses.items() if n == pinnacle}
        if not censuses or not self.drive.needs.items():
            return frontrunners
        try:
            houses = self.drive.shelf.hub_aisles()
            selected: set[str] = set()
            for hub in houses:
                extant = [label for label in hub if label in censuses]
                if extant:
                    best = max(censuses[label] for label in extant)
                    selected |= {label for label in extant if censuses[label] == best}
                else:
                    selected |= set(hub)
            if not all(houses):
                # some product has no known shelf yet: top up with the busiest labels
                for label in sorted(censuses, key=lambda s: -censuses[s]):
                    if len(selected) >= len(houses):
                        break
                    selected.add(label)
            return selected or frontrunners
        except Exception:
            return frontrunners

    @staticmethod
    def datarow_stamp(row: dict[str, Any]) -> str:
        for precise in (True, False):
            for key, vlu in row.items():
                if isinstance(vlu, bool) or not isinstance(vlu, (str, int)):
                    continue
                name = str(key).lower()
                if (name == "id") if precise else name.endswith("id"):
                    return str(vlu)
        return json.dumps(row, sort_keys=True, default=str)

    def hunt_selectors(self, trigger_operands: dict[str, Any]) -> tuple[list[tuple], tuple, bool, set]:
        """Split search arguments into free-text word tuples, enum modes, a first-page flag and the paging fields."""
        drive = self.drive
        contour = drive.contours.get(drive.role("search") or "")
        props = contour.get("properties") if isinstance(contour, dict) else None
        if not isinstance(props, dict):
            props = {}

        def brief(field: str) -> dict[str, Any]:
            element_alt = props.get(field)
            return element_alt if isinstance(element_alt, dict) else {}
        proses: list[tuple] = []
        manners: list[tuple] = []
        scrolling: set = set()
        primary_folio = True
        token = self.morpheme_selector()
        for field, value in trigger_operands.items():
            if isinstance(value, str):
                if field == token:
                    scrolling.add(field)
                    if value.strip():
                        primary_folio = False
                elif brief(field).get("enum"):
                    manners.append((field, value))
                else:
                    fragments = (dab.strip("\"'()[]{},.;:!?|/") for dab in value.lower().split())
                    words = tuple(dict.fromkeys(dab for dab in fragments if re.search(r"[a-z0-9]", dab)))
                    if words and all(re.fullmatch(r"[a-z0-9]{2,}", dab) for dab in words):
                        proses.append((field, words))
                continue
            name = str(field).lower()
            if not be_numeral(value) or not re.search(r"page|offset|first|start|skip|from|cursor", name):
                continue
            if re.search(r"size|per_?page|limit|price|cost", name):
                continue
            scrolling.add(field)
            ceilings = brief(field)
            floor = ceilings.get("minimum", ceilings.get("default"))
            if not be_numeral(floor):
                floor = 1 if "page" in name else 0
            if not value <= floor:
                primary_folio = False
        return proses, tuple(sorted(manners)), primary_folio, scrolling

    @staticmethod
    def outcome_overall(obv: dict[str, Any], rows: list) -> int:
        censuses = [vlu for vlu in obv.values() if isinstance(vlu, int) and not isinstance(vlu, bool)]
        return max(censuses + [len(rows)])

    def vocable_policy(self, trigger_operands: dict[str, Any], obv: dict[str, Any], rows: list) -> str:
        """Learn whether the store's keyword search needs all words or any word, from pairs of first-page searches."""
        drive = self.drive
        proses, mode, primary_folio, scrolling = self.hunt_selectors(trigger_operands)
        if not primary_folio:
            return drive.vocable_variety
        total = self.outcome_overall(obv, rows)

        def widens(lengthier: tuple, briefer: tuple) -> bool:
            return len(lengthier) > len(briefer) and lengthier[: len(briefer)] == briefer
        for field, words in proses:
            ctxt = (field, mode)
            rest_alt = tuple(sorted((kx, str(vz)) for kx, vz in trigger_operands.items() if kx != field and kx not in scrolling))
            subgroup_cue = False
            for bfr_env, bfr_rest, bfr_vocables, bfr_overall in drive.vocable_noticed:
                if bfr_env != ctxt or bfr_rest != rest_alt or bfr_overall == total:
                    continue
                if widens(words, bfr_vocables):
                    drive.vocable_variety = "any" if total > bfr_overall else "all"
                elif widens(bfr_vocables, words):
                    drive.vocable_variety = "any" if total < bfr_overall else "all"
                else:
                    depth = drive.guide_vocables or None
                    minee, they = set(words[:depth]), set(bfr_vocables[:depth])
                    if (minee > they and total < bfr_overall) or (they > minee and bfr_overall < total):
                        subgroup_cue = True
            if subgroup_cue and not drive.vocable_variety:
                drive.vocable_variety = "all"
            drive.vocable_noticed.append((ctxt, rest_alt, words, total))
        return drive.vocable_variety

    def guide_policy(self, trigger_operands: dict[str, Any], obv: dict[str, Any], rows: list) -> int:
        """Learn how many leading query words the store actually uses."""
        drive = self.drive
        proses, mode, primary_folio, scrolling = self.hunt_selectors(trigger_operands)
        if not primary_folio:
            return drive.guide_vocables
        total = self.outcome_overall(obv, rows)
        marks = tuple(self.datarow_stamp(row) for row in rows)
        for field, words in proses:
            ctxt = (field, mode)
            rest_alt = tuple(sorted((kx, str(vz)) for kx, vz in trigger_operands.items() if kx != field and kx not in scrolling))
            edges = drive.guide_edges.setdefault(ctxt, [-1, set()])
            for bfr_env, bfr_rest, bfr_vocables, bfr_overall, bfr_stamps in drive.guide_noticed:
                if bfr_env != ctxt or bfr_rest != rest_alt or bfr_vocables == words:
                    continue
                briefer = min(len(words), len(bfr_vocables))
                joint = next((ii for ii in range(briefer) if words[ii] != bfr_vocables[ii]), briefer)
                if total != bfr_overall or marks != bfr_stamps:
                    edges[0] = max(edges[0], joint)
                elif joint and joint == briefer and len(set(marks)) >= 3:
                    edges[1].add(joint)
            drive.guide_noticed.append((ctxt, rest_alt, words, total, marks))
        contenders = []
        for ctxt, (demonstrated, unbrowsed) in drive.guide_edges.items():
            taller = [n for n in unbrowsed if n > demonstrated]
            if taller:
                contenders.append((min(taller), ctxt))
        concluded = [(n, ctxt) for n, ctxt in contenders if n == drive.guide_edges[ctxt][0] + 1]
        drive.guide_vocables, drive.guide_env = min(concluded) if concluded else (0, None)
        return drive.guide_vocables

    def hunt_columns(self) -> dict[str, Any]:
        drive = self.drive
        contour = drive.contours.get(drive.role("search") or "") or {}
        props = contour.get("properties")
        return props if isinstance(props, dict) else {}

    def morpheme_selector(self) -> str | None:
        for field, brief in self.hunt_columns().items():
            if not isinstance(brief, dict) or brief.get("enum") or brief.get("type") != "string":
                continue
            if re.search(r"bookmark|cursor|token|resume|continu|after|next|page|offset|start", str(field).lower()):
                return field
        return None

    def folio_operand(self) -> tuple[str, Any, Any, float | None] | None:
        """(field, whole-page flag or "token", first value, highest allowed) for the search tool's paging argument."""
        for field, brief in self.hunt_columns().items():
            if not isinstance(brief, dict) or brief.get("type") not in ("integer", "number"):
                continue
            name = str(field).lower()
            if re.search(r"size|per_?page|limit|price|cost", name) or not re.search(r"page|offset|first|start|skip|from|cursor", name):
                continue
            entire = "page" in name and not re.search(r"offset|skip|first|start|from", name)
            start = brief.get("minimum", brief.get("default"))
            if not be_numeral(start):
                start = 1 if entire else 0
            apex = brief.get("maximum")
            return field, entire, start, (apex if be_numeral(apex) else None)
        token = self.morpheme_selector()
        return (token, "token", "", None) if token else None

    @staticmethod
    def root_alt(vocable: str) -> str:
        multi_alt = len(vocable) > 3 and vocable.endswith("s") and not vocable.endswith("ss")
        return vocable[:-1] if multi_alt else vocable

    def aisle_datarows(self) -> dict[str, tuple[float, str, str]]:
        """id -> (price, title, maker) for in-stock items on the wanted shelves that fit the budget."""
        drive = self.drive
        stash = drive.stash
        captions, budget, metrics = self.aisle_captions(), drive.needs.disburse_top, drive.needs.quantity()
        discovered: dict[str, tuple[float, str, str]] = {}
        if not captions:
            return discovered

        def primary_relating(transcript: dict[str, Any], admit, failsafe: Any = None) -> Any:
            return next((vlu for key, vlu in transcript.items() if admit(str(key).lower(), vlu)), failsafe)

        def amass(transcript: dict[str, Any]) -> None:
            lbl = primary_relating(transcript, lambda name, vlu: isinstance(vlu, str) and "id" in name and vlu in stash.ids)
            if not lbl or lbl in drive.refused_lines or self.aisle_relating(transcript) not in captions:
                return
            if self.vended_outward(transcript):
                discovered.pop(lbl, None)
                return
            price = primary_relating(transcript, lambda name, vlu: be_numeral(vlu) and not isinstance(vlu, bool)
                             and re.search(r"price|cost", name))
            if price is None:
                return
            title = transcript.get("title")
            if not (isinstance(title, str) and 0 < len(title) <= 300):
                title = primary_relating(transcript, lambda name, vlu: isinstance(vlu, str) and 0 < len(vlu) <= 300
                                 and re.search(r"title|heading|headline|name", name)
                                 and not re.search(r"categor|shelf|brand|make|manufactur|vendor|shop|seller|store|id", name), "")
            creator = " ".join(vlu for key, vlu in transcript.items()
                             if isinstance(vlu, str) and re.search(r"brand|make|manufactur|vendor", str(key).lower()))
            discovered[lbl] = (float(price), title, creator)
        for item in stash.items:
            route_hashmaps(item, amass)
        for lbl, (price, title, creator) in list(discovered.items()):
            crisp = stash.price(lbl)
            if crisp is not None:
                price = crisp
            if be_numeral(budget) and price * metrics > budget:
                discovered.pop(lbl)
            else:
                discovered[lbl] = (float(price), title, creator)
        return discovered

    def caption_hunttexts(self) -> list[str]:
        queries: list[str] = []
        for label in sorted(self.aisle_captions()):
            drives: list[list[str]] = [[]]
            for crude in label.lower().split():
                vocable = crude.strip(".,;:!?'\"()[]")
                if vocable not in ANCHOR_VOCABLES and re.fullmatch(r"[a-z0-9][a-z0-9'-]*", vocable):
                    drives[-1].append(vocable)
                elif drives[-1]:
                    drives.append([])
            cohorts = [fragment for fragment in drives if fragment]
            words = [wq for fragment in cohorts for wq in fragment]
            crowns = [fragment[-1] for fragment in cohorts]
            if len(words) >= 2:
                queries.append(" ".join(words[:2]))
            queries += crowns
            queries += [wq for wq in words if wq not in crowns]
        return list(dict.fromkeys(queries))

    def variety_hunttexts(self, pairs: bool) -> list[str]:
        """Words (or word pairs) shared by two or more shelf titles, cheapest first."""
        caption_roots = {self.root_alt(wq) for label in self.aisle_captions() for wq in re.findall(r"[a-z0-9]+", label)}
        through_headline: dict[str, tuple[float, str]] = {}
        for price, title, creator in self.aisle_datarows().values():
            key = title.lower().strip()
            if key:
                antecedent = through_headline.get(key, (price, ""))[0]
                through_headline[key] = (min(price, antecedent), creator.lower())
        hits: dict[str, int] = {}
        cheapest: dict[str, float] = {}
        rendered_alt: dict[str, str] = {}
        for title, (price, creator) in through_headline.items():
            creator_vocables = set(re.findall(r"[a-z]+", creator))
            words = re.findall(r"[a-z]+", title)
            fine = [len(wq) >= 3 and wq not in ANCHOR_VOCABLES and wq not in creator_vocables for wq in words]
            if pairs:
                shingles = [(a, g_alt) for a, g_alt, sound_u, sound_g in zip(words, words[1:], fine, fine[1:]) if sound_u and sound_g]
            else:
                shingles = [(wq,) for wq, ok in zip(words, fine) if ok]
            done: set[str] = set()
            for shingle in shingles:
                roots_alt = [self.root_alt(wq) for wq in shingle]
                spliced = " ".join(roots_alt)
                if spliced in done or all(s in caption_roots for s in roots_alt):
                    continue
                done.add(spliced)
                hits[spliced] = hits.get(spliced, 0) + 1
                cheapest[spliced] = min(cheapest.get(spliced, price), price)
                rendered_alt.setdefault(spliced, " ".join(shingle))
        usual = sorted((g for g in hits if hits[g] >= 2), key=lambda g: (cheapest[g], -hits[g], g))
        return [rendered_alt[g] for g in usual]

    def comb_ensuing(self, stint: int) -> tuple[dict[str, Any] | None, bool]:
        """Next shelf-sweep search call and whether it continues a listing's paging."""
        drive = self.drive
        scrolling = self.folio_operand()
        captions = self.aisle_captions()
        tome = drive.page_book
        if not scrolling or not captions or not tome:
            return None, False
        folio_column, first = scrolling[0], scrolling[2]
        tested = self.comb_tested

        def seal(trigger_operands: dict[str, Any]) -> str:
            return json.dumps(trigger_operands, sort_keys=True, default=str)

        def fingerprint(text: str) -> tuple:
            return tuple(self.root_alt(wq) for wq in re.findall(r"[a-z0-9]{2,}", text.lower())[:2])
        access_catalogs = [wq for wq in tome.values()
                      if wq.get("id_order") and wq["shelves"] and wq["shelves"] <= captions and self.ensuing_folio(wq) is not None]
        listing = max(access_catalogs, key=lambda wq: wq["total"], default=None)
        ranked = ([wq for wq in tome.values() if wq.get("id_order") is False and self.unbound_prose(wq["args"])]
                  or [wq for wq in tome.values() if wq.get("id_order") is None and self.unbound_prose(wq["args"])])
        stencil, prose_column = None, None
        if ranked:
            fields = self.hunt_columns() or {}
            oracle_operands = ranked[-1]["args"]

            def immutable(key: str, vlu: Any) -> bool:
                if not isinstance(vlu, str) or (fields.get(key) or {}).get("enum"):
                    return True
                return vlu.strip().lower() in captions or key == folio_column
            free = [key for key, vlu in oracle_operands.items() if not immutable(key, vlu)]
            if drive.guide_env and drive.guide_env[0] in free:
                prose_column = drive.guide_env[0]
            else:
                prose_column = max(free, key=lambda key: len(oracle_operands[key].split()), default=None)
            if prose_column:
                stencil = {key: vlu for key, vlu in oracle_operands.items() if key != prose_column and immutable(key, vlu)}
        encompassed = {fingerprint(" ".join(self.unbound_prose(wq["args"]))) for wq in tome.values()
                   if first in wq["read"] and wq.get("id_order") is not True}

        def triggers_towards(proses: list[str]) -> list[dict[str, Any]]:
            constructed = []
            for text in proses:
                sig = fingerprint(text)
                if stencil is None or not sig or sig in encompassed:
                    continue
                trigger_operands = {**stencil, prose_column: text}
                if seal(trigger_operands) not in tested:
                    constructed.append(trigger_operands)
            return constructed
        provenances = [
            lambda: [{**listing["args"], folio_column: self.ensuing_folio(listing)}] if listing else [],
            lambda: triggers_towards(self.variety_hunttexts(True)[:6]),
            lambda: triggers_towards(self.variety_hunttexts(False)[:6]),
            lambda: triggers_towards(self.caption_hunttexts()),
        ]
        summit_trigger, finished = drive.encompass.summit_comb(tested)
        if summit_trigger is not None:
            return summit_trigger, True
        if finished:
            return None, False
        # the listing page always goes first; the three query sources rotate with the turn
        for select in [0] + [1 + (stint + displacement) % 3 for displacement in range(3)]:
            for trigger_operands in provenances[select]():
                if seal(trigger_operands) not in tested:
                    return trigger_operands, select == 0
        return None, False

    def comb_space(self, stockpile: int) -> bool:
        drive = self.drive
        if drive.maneuvers_residual() <= stockpile:
            return False
        return drive.channel.extra_alt() >= SEALING + 40

    def comb_phase(self) -> bool:
        drive = self.drive
        needs = drive.needs
        if needs.priority() != "cheapest" or not needs.let_go or needs.feats_residual() or not drive.role("search") or drive.channel.done:
            return False
        stockpile = 4 if drive.wagon.metrics else 3
        if drive.orders_tried == 0:
            stockpile = max(stockpile + 3, PRIMARY_CHECKOUT_THROUGH)
        if self.combed >= COMB_BROWSES or not self.comb_space(stockpile):
            return False
        trigger_operands, fromsrc_offering = self.comb_ensuing(self.comb_stint)
        if trigger_operands is None:
            return False
        self.combed += 1
        if not fromsrc_offering:
            self.comb_stint += 1
        self.comb_tested.add(json.dumps(trigger_operands, sort_keys=True, default=str))
        suggestion = {"id": f"sweep_{self.combed}_{drive.channel.beat}", "type": "function",
                    "function": {"name": drive.role("search"), "arguments": json.dumps(trigger_operands)}}
        drive.chat.append({"role": "assistant", "content": "No property requirement is left, so the cheapest in-stock listing "
                         "on the shelf is the winner: I read one more slice of the shelf before I choose.",
                         "tool_calls": [suggestion]})
        drive.covering = True
        try:
            rejoinders = drive.forward([suggestion], automated=True)
        finally:
            drive.covering = False
        additional = (self.combed < COMB_BROWSES and self.comb_space(stockpile)
                and self.comb_ensuing(self.comb_stint)[0] is not None)
        if rejoinders and not additional:
            cheapest = sorted(self.aisle_datarows().items(), key=lambda duo: (duo[1][0], duo[0]))[:6]
            if cheapest:
                cuid, text = rejoinders[-1]
                itemized = "; ".join(f"{lbl} '{title[:80]}' {price:g}" for lbl, (price, title, _) in cheapest)
                rejoinders[-1] = (cuid, text + "\n[shelf sweep done] Cheapest in-stock rows of the requested category within "
                               "the budget, collected across every page seen: " + itemized + ". Pick the cheapest one unless "
                               "a change excludes it: make it the cart's only line, check its stock and price again, and "
                               "order it.")
        drive.speak.subsequent_rejoinders(rejoinders)
        return True

    def unbound_prose(self, trigger_operands: dict[str, Any]) -> list[str]:
        fields = self.hunt_columns()
        token = self.morpheme_selector()

        def be_unbound(key: str, vlu: Any) -> bool:
            return isinstance(vlu, str) and not (fields.get(key) or {}).get("enum") and key != token
        return [vlu for key, vlu in trigger_operands.items() if be_unbound(key, vlu)]

    @staticmethod
    def ensuing_morpheme(obv: Any) -> str | None:
        if not isinstance(obv, dict):
            return None
        return next((vlu for key, vlu in obv.items()
                     if isinstance(vlu, str) and vlu.strip()
                     and re.search(r"next|onward|cursor|bookmark|token|continu|more", str(key).lower())), None)

    @staticmethod
    def organic_selector(stamp_alt: str) -> list:
        return [int(shard) if shard.isdigit() else shard for shard in re.split(r"(\d+)", stamp_alt)]

    def tome_folio(self, trigger_operands: dict[str, Any], rows: list, total: int, obv: Any = None) -> None:
        """Record one result page of a search in the page book (pages read, size, total, shelves, id order)."""
        scrolling = self.folio_operand()
        if not scrolling or not rows:
            return
        field, mode, first, _ = scrolling
        locus = trigger_operands.get(field, first)
        if mode == "token":
            locus = locus.strip() if isinstance(locus, str) else ""
        elif not be_numeral(locus):
            return
        remainder = {key: vlu for key, vlu in trigger_operands.items() if key != field}
        element_alt = self.drive.page_book.setdefault(json.dumps(remainder, sort_keys=True, default=str),
                                              {"args": remainder, "read": set(), "size": 0, "total": 0, "shelves": set()})
        element_alt["read"].add(locus)
        if mode == "token":
            element_alt.setdefault("tokens", {})[locus] = self.ensuing_morpheme(obv)
        transcripts = [row for row in rows if isinstance(row, dict)]
        element_alt["size"] = max(element_alt["size"], len(rows))
        element_alt["total"] = max(element_alt["total"], total)
        element_alt["shelves"] |= {self.aisle_relating(row) for row in transcripts} - {""}
        through_datarow = element_alt.setdefault("row_shelf", {})
        for row in transcripts:
            through_datarow.setdefault(self.datarow_stamp(row), self.aisle_relating(row))
        if locus == first:
            element_alt["first"] = tuple(self.datarow_stamp(row) for row in transcripts)
            if total <= len(rows):
                element_alt["every"] = transcripts
        marks = [self.datarow_stamp(row) for row in transcripts]
        if len(marks) >= 5:
            rising = marks == sorted(marks) or marks == sorted(marks, key=self.organic_selector)
            element_alt["id_order"] = element_alt.get("id_order", True) is not False and rising

    def ensuing_folio(self, walk: dict[str, Any]) -> Any:
        field, phase_manner, first, tallest = self.folio_operand()
        if phase_manner == "token":
            return next((token for token in (walk.get("tokens") or {}).values()
                         if isinstance(token, str) and token not in walk["read"]), None)
        start = min([first] + [vz for vz in walk["read"] if be_numeral(vz)])
        size = max(1, walk["size"])
        gait = 1 if phase_manner else size
        for n in range(1, 400):
            if n * size >= walk["total"]:
                return None
            contender = start + n * gait
            if tallest is not None and contender > tallest:
                return None
            if contender not in walk["read"]:
                return contender
        return None

    @staticmethod
    def shut_vocables(a: str, g_alt: str) -> bool:
        if min(len(a), len(g_alt)) < 3:
            return False
        return a.startswith(g_alt) or g_alt.startswith(a)

    def measurement_merit(self, trigger_operands: dict[str, Any]) -> int:
        """How many open required values the query's (leading) words touch."""
        drive = self.drive
        cohorts = []
        for need in drive.needs.feats_residual():
            value = need.partition(":")[2]
            if not cite_prose(value):
                cohorts.append(re.findall(r"[a-z0-9]{3,}", value.lower()))
        entered = [wq for text in self.unbound_prose(trigger_operands) for wq in re.findall(r"[a-z0-9]{2,}", text.lower())]
        if drive.guide_vocables:
            entered = entered[: drive.guide_vocables]
        return sum(1 for group in cohorts if any(self.shut_vocables(tz, vz) for tz in entered for vz in group))

    def datarow_impacts(self, rows: list, words: list[str] | None = None) -> list[str]:
        """Unread, in-stock, in-budget rows on the wanted shelves; also notes rows a referenced seller points to."""
        drive = self.drive
        stash = drive.stash
        budget, metrics = drive.needs.disburse_top, drive.needs.quantity()
        captions = self.aisle_captions()
        hits = []
        named = set()
        if drive.needs.browsing.get("references"):
            named = {wq for text in words or [] for wq in re.findall(r"[a-z0-9]+", str(text).lower())}
        arrows = drive.ref_hits

        def merchant_collections(row: dict[str, Any]) -> list[set]:
            out = []
            for key, vlu in row.items():
                reduced = str(key).lower()
                if not (isinstance(vlu, str) and vlu.strip()) or "id" in reduced:
                    continue
                if re.search(r"shop|seller|store|vendor|brand|make|manufactur", reduced):
                    out.append(set(re.findall(r"[a-z0-9]+", vlu.lower())))
            return out
        for row in rows:
            if not isinstance(row, dict):
                continue
            lbl = self.datarow_stamp(row)
            producers = merchant_collections(row)
            unbrowsed = lbl in stash.ids and lbl not in stash.read
            if named and unbrowsed and any(mz and mz <= named and mz <= self.directed_vocables() for mz in producers):
                if lbl not in arrows:
                    arrows.append(lbl)
            if not unbrowsed:
                continue
            price = next((vlu for key, vlu in row.items() if be_numeral(vlu) and re.search(r"price|cost", str(key).lower())), None)
            label = self.aisle_relating(row)
            if be_numeral(budget) and be_numeral(price) and price * metrics > budget:
                continue
            if self.vended_outward(row) or (captions and label and label not in captions):
                continue
            hits.append(lbl)
            self.impact_aisle[lbl] = label
        return hits

    def directed_vocables(self) -> set[str]:
        """Words of the shopper's references plus body words of read records (seller names removed); cached per archive size."""
        drive = self.drive
        stash = drive.stash
        buffered = self.directed_jotting
        if buffered and buffered[0] == len(stash.items):
            return buffered[1]
        words: set[str] = set()
        try:
            for cite in drive.needs.browsing.get("references") or []:
                words |= set(re.findall(r"[a-z0-9]+", str(cite).lower()))

            def ingest(transcript: dict[str, Any]) -> None:
                if self.datarow_stamp(transcript) not in stash.read:
                    return
                merchants: set[str] = set()
                bodytexts: list[str] = []
                for key, vlu in transcript.items():
                    if not isinstance(vlu, str):
                        continue
                    reduced = str(key).lower()
                    be_merchant = (re.search(r"shop|seller|store|vendor|brand|make|manufactur", reduced) and "id" not in reduced
                                 and len(vlu) <= 40
                                 and not re.search(r"note|text|desc|comment|detail|info|message|section", reduced))
                    if be_merchant:
                        merchants |= set(re.findall(r"[a-z0-9]+", vlu.lower()))
                    elif not re.search(r"title|heading|headline|name|label|caption", reduced):
                        bodytexts.append(vlu.lower())
                for body in bodytexts:
                    words.update(set(re.findall(r"[a-z0-9]+", body)) - merchants)
            for item in stash.items:
                route_hashmaps(item, ingest)
        except Exception:
            pass
        self.directed_jotting = (len(stash.items), words)
        return words

    def access_folios(self, maximal: int = 2) -> list[dict[str, Any]]:
        scrolling = self.folio_operand()
        if not scrolling:
            return []
        captions = self.aisle_captions()
        graded = []
        for walk in self.drive.page_book.values():
            forthcoming = self.ensuing_folio(walk)
            if forthcoming is None:
                continue
            if captions and walk["shelves"] and not walk["shelves"] & captions:
                continue
            merit = self.measurement_merit(walk["args"])
            if not merit:
                continue
            graded.append((-merit, walk["total"], {**walk["args"], scrolling[0]: forthcoming}))
        graded.sort(key=lambda element_alt: (element_alt[0], element_alt[1]))
        return [trigger_operands for _, _, trigger_operands in graded[:maximal]]

    @staticmethod
    def investigate_nigh(a: str, g_alt: str) -> bool:
        if not any(hx.isdigit() for hx in a + g_alt):
            return min(len(a), len(g_alt)) >= 3 and (a.startswith(g_alt) or g_alt.startswith(a))
        if a.lstrip("0") == g_alt.lstrip("0"):
            return True
        return a.isdigit() and g_alt.startswith(a) and not g_alt[len(a):len(a) + 1].isdigit()

    def measurement_investigate(self, crude: list[str]) -> tuple[str, str]:
        """When the query leads with a property NAME, suggest a search that leads with its value instead."""
        drive = self.drive
        if drive.vocable_variety == "any":
            return "", ""
        depth = drive.guide_vocables or 2
        nigh = self.investigate_nigh

        def head(text: str) -> list[str]:
            return [wq for wq in re.findall(r"[a-z0-9]+", text.lower()) if len(wq) >= 2][:depth]
        lead = head(" ".join(crude))
        product = " ".join(re.findall(r"[a-z0-9]+", str(drive.needs.browsing.get("product") or "").lower())[:3])
        wished_alt = []
        for need in drive.needs.feats_residual():
            name, punct, value = need.partition(":")
            if punct and not cite_prose(value):
                bare_alt = re.split(r"[,;]", re.sub(r"\((?:stated|revealed[^)]*)\)", "", value))[0].strip().lower()
                wished_alt.append((name.strip(), bare_alt))
        known = [wq for _, value in wished_alt for wq in re.findall(r"[a-z0-9]{2,}", value)] + product.split()
        if not lead or any(nigh(wq, kx) for wq in lead for kx in known):
            return "", ""
        for name, value in wished_alt:
            if not value:
                continue
            if not any(nigh(wq, n) for wq in lead for n in re.findall(r"[a-z0-9]{2,}", name.lower())):
                continue
            cipher = re.search(r"\d", value)
            kernel = value[cipher.start():] if cipher else value
            investigate = f"{kernel.strip(' .;')} {product}".strip()
            investigate_front = head(investigate)
            for walk in drive.page_book.values():
                drove = head(" ".join(self.unbound_prose(walk["args"])))
                if len(drove) == len(investigate_front) and all(nigh(xq, y) for xq, y in zip(investigate_front, drove)):
                    return "", ""
            return name, investigate
        return "", ""


class InexpensiveRoute:
    """Cheapest-first coverage: walks id-ordered result pages, flags trap words, prepares decline-time reads."""

    def __init__(self, drive):
        self.drive = drive
        self.routed = set()
        self.befit_maintained = None
        self.reject_browse_accomplished = None
        self.reject_waitlist = None

    def drove_hunttext(self, text: str) -> bool:
        drive = self.drive
        observe = drive.observe
        broadness = drive.guide_vocables or 3

        def key(vocables_prose: str) -> list[str]:
            return sorted(observe.root_alt(wq) for wq in re.findall(r"[a-z0-9]+", vocables_prose)[:broadness])
        target = key(str(text).lower())
        return any(key(" ".join(observe.unbound_prose(walk["args"])).lower()) == target for walk in drive.page_book.values())

    def hunt_stencil(self) -> tuple[dict[str, Any], str] | None:
        """Most recent plain search with exactly one free-text field: (other args, text field)."""
        drive = self.drive
        for walk in reversed(list(drive.page_book.values())):
            trigger_operands = walk["args"]
            if drive.encompass.gx_offering(trigger_operands):
                continue
            proses = drive.observe.unbound_prose(trigger_operands)
            if len(proses) != 1:
                continue
            openings = [key for key, vlu in trigger_operands.items() if isinstance(vlu, str) and vlu in proses and vlu.strip()]
            if len(openings) != 1:
                continue
            return {key: vlu for key, vlu in trigger_operands.items() if key != openings[0]}, openings[0]
        return None

    @staticmethod
    def pitfall_nigh(a: str, g_alt: str) -> bool:
        if any(hx.isdigit() for hx in a + g_alt):
            return a.lstrip("0") == g_alt.lstrip("0")
        if min(len(a), len(g_alt)) < 3:
            return False
        return a.startswith(g_alt) or g_alt.startswith(a)

    @staticmethod
    def morpheme_components(token: str) -> list[str]:
        return [shard for shard in re.findall(r"[a-z0-9]+", token) if len(shard) >= 2]

    def vocable_pitfall(self, crude: list[str]) -> tuple[str, str]:
        try:
            return self.locate_pitfall(crude)
        except Exception as exn:
            self.drive.diary.append({"kind": "trap_error", "error": f"{type(exn).__name__}: {exn}"[:300]})
            return "", ""

    def locate_pitfall(self, crude: list[str]) -> tuple[str, str]:
        """Spot a query word taken from a property name or a dropped requirement; offer a cleaner query."""
        drive = self.drive
        needs = drive.needs
        nigh, components = self.pitfall_nigh, self.morpheme_components
        if drive.vocable_variety == "any":
            return "", ""
        tokens = [shard.strip("\"'()[]{},.;!?|").lower() for shard in " ".join(crude).split()]
        tokens = [shard for shard in tokens if re.search(r"[a-z0-9]", shard)]
        broadness = drive.guide_vocables or (max(3, len(tokens)) if drive.vocable_variety == "all" else 3)
        tokens = tokens[:broadness]
        access_necessities = []
        for need in needs.feats_residual():
            head, punct, value = need.partition(":")
            if punct and not cite_prose(value):
                naked = re.sub(r"\((?:stated|revealed[^)]*)\)", "", value).lower()
                access_necessities.append((re.split(r"\s*\(", head)[0].lower(), head, naked))
        measurement_vocables = {wq for _, _, naked in access_necessities for wq in re.findall(r"[a-z0-9]{2,}", naked)}
        products = [element_alt["product"] for element_alt in needs.items()] or [str(needs.browsing.get("product") or "")]
        kinds = set(re.findall(r"[a-z0-9]{3,}", " ".join(products + sorted(drive.observe.aisle_captions())).lower())) - ANCHOR_VOCABLES
        feat_designations = {wq for name, _, _ in access_necessities for wq in re.findall(r"[a-z]{3,}", name)} - ANCHOR_VOCABLES
        dropped = {wq for aged in needs.let_go for wq in re.findall(r"[a-z]{3,}", aged.partition(":")[2].lower())} - ANCHOR_VOCABLES
        secure = measurement_vocables | kinds

        def grazes(fragments: list[str], cluster) -> bool:
            return any(nigh(pz, wq) for pz in fragments for wq in cluster)
        noted_alt: list[str] = []
        reason = ""
        for token in tokens:
            fragments = components(token)
            if not fragments or grazes(fragments, secure):
                continue
            if grazes(fragments, feat_designations):
                noted_alt.append(token)
                reason = reason or "name"
            elif grazes(fragments, dropped):
                noted_alt.append(token)
                reason = reason or "dropped"
        if not noted_alt:
            return "", ""
        kept = [tz for tz in tokens if tz not in noted_alt]
        conveyed = [(tag, naked) for _, tag, naked in access_necessities
                   if any(nigh(pz, wq) for tz in kept for pz in components(tz) for wq in re.findall(r"[a-z0-9]{2,}", naked))]
        if not conveyed:
            return "", ""
        if not any(nigh(pz, kx) for tz in kept for pz in components(tz) for kx in kinds):
            slot = drive.speak.commodity_relating(conveyed[0][0]) if needs.items() else None
            name = products[slot] if slot is not None and slot < len(products) else products[0]
            kept += [wq for wq in re.findall(r"[a-z]{3,}", name.lower()) if wq not in ANCHOR_VOCABLES][-1:]
        query = " ".join(kept[:broadness])
        if not query or self.drove_hunttext(query):
            return "", ""
        vocable = noted_alt[0]
        if reason == "name":
            note = (f"The query word '{vocable}' is taken from the NAME of a required property, not from a required value. "
                    "Because the store matches every word, it keeps only listings whose fields or title repeat that name and "
                    "loses the ones whose field shows only the value.")
        else:
            note = (f"The query word '{vocable}' belongs only to a requirement the shopper DROPPED, so the search still keeps "
                    "just the listings that carry it, although they no longer have to.")
        return note + f" Search '{query}' as well and read all of its pages.", query

    def pooled_hunttexts(self) -> list[dict[str, Any]]:
        """Up to two searches that join two short required values (plus the product word) in one query."""
        drive = self.drive
        needs = drive.needs
        stencil = self.hunt_stencil()
        if not stencil:
            return []
        several = needs.items()
        broadness = drive.guide_vocables or 3
        edition = str(needs.edition)
        ongoing = {key.rpartition("|")[0] for key in drive.passed if key.rpartition("|")[2] == edition} - drive.refused_lines
        encompassed = {drive.referee.aisle_recognized(lbl) for lbl in ongoing} - {""}
        houses = drive.shelf.hub_aisles() if several else []
        constructed: list[dict[str, Any]] = []
        for kx in (range(len(several)) if several else [None]):
            if several:
                if kx >= len(houses) or not houses[kx] or houses[kx] & encompassed:
                    continue
            elif not drive.admission.unvalidated():
                continue
            cohorts: list[list[str]] = []
            for need in needs.feats_residual():
                head, _, value = need.partition(":")
                if several and drive.speak.commodity_relating(head) not in (kx, None):
                    continue
                value = re.sub(r"\((?:stated|revealed[^)]*)\)", "", value).strip().lower()
                if not value:
                    continue
                if re.search(r"\d|as shown|same as|page|redirect|refer|\bnot\b|\bno\b|without|except|anything|other than", value):
                    continue
                primary_alternative = re.split(r"[,;/]|\bor\b|\band\b", value)[0]
                words = [wq for wq in re.findall(r"[a-z]{2,}", primary_alternative) if wq not in ANCHOR_VOCABLES]
                if 1 <= len(words) <= 2:
                    cohorts.append(words)
            if len(cohorts) < 2:
                continue
            query: list[str] = []
            for words in cohorts:
                if len(query) + len(words) <= broadness:
                    query += words
            product = several[kx]["product"] if several else str(needs.browsing.get("product") or "")
            kind = [wq for wq in re.findall(r"[a-z]{3,}", product.lower()) if wq not in ANCHOR_VOCABLES][-1:]
            if kind and len(query) < broadness and kind[0] not in query:
                query += kind
            text = " ".join(query)
            if len(query) >= 2 and not self.drove_hunttext(text):
                constructed.append({**stencil[0], stencil[1]: text})
        return constructed[:2]

    def mediate(self, entity: str, decree: str) -> str:
        """Second item check when the decline audit and an earlier check disagree about the same item."""
        drive = self.drive
        try:
            reason = next((str((element_alt.get("result") or {}).get("reason") or "") for element_alt in reversed(drive.diary)
                           if isinstance(element_alt, dict) and element_alt.get("kind") == "audit" and isinstance(element_alt.get("result"), dict)),
                          "")
            inspection_component = f" ({reason[:400]})" if reason else ""
            surplus = ("\n\nTWO READINGS OF THIS ITEM DISAGREE. Goal: decide which reading of the same records is correct. "
                     "1. The decline audit found that it meets every requirement" + inspection_component
                     + "; an earlier check said: " + decree[:500]
                     + "\n2. Decide from the records: a field that states the required value in other words, or a shade, "
                     "qualifier or spelling of that same named value, meets the requirement; a field that states a different "
                     "value fails it.")
            verdict = drive.referee.validate_article(entity, "ORDER", second=surplus)
            if verdict:
                return verdict
            key = f"{entity}|{drive.needs.edition}"
            obsolete = [jotting for jotting in drive.rulings if jotting == key or jotting.startswith(key + "|")]
            for jotting in obsolete:
                drive.rulings[jotting] = ""
            return ""
        except Exception as exn:
            drive.diary.append({"kind": "arbitrate_error", "error": f"{type(exn).__name__}: {exn}"[:300]})
            return decree

    def undisputed(self) -> list[str]:
        try:
            return self.reject_contenders()
        except Exception as exn:
            self.drive.diary.append({"kind": "unrefuted_error", "error": f"{type(exn).__name__}: {exn}"[:300]})
            return []

    def reject_contenders(self) -> list[str]:
        """Cheapest read, unrefuted, in-budget item per product that no check or audit has turned down yet."""
        drive = self.drive
        needs, stash, referee = drive.needs, drive.stash, drive.referee
        several = needs.items()
        houses = drive.shelf.hub_aisles() if several else [set(drive.observe.aisle_captions())]
        if not houses or not all(houses):
            return []
        edition = str(needs.edition)
        faltered = {key.split("|")[0] for key, verdict in drive.rulings.items() if verdict and key.split("|")[1:2] == [edition]}
        ongoing = {key.rpartition("|")[0] for key in drive.passed if key.rpartition("|")[2] == edition}
        assessed = referee.inspection_idents
        if drive.audit_done:
            constraint = -1 if referee.inspection_browses is None else referee.inspection_browses
            if assessed is None or len(stash.read) <= constraint:
                assessed = set(stash.read)
        vetted = set(assessed or set()) - set(referee.inspection_befits or [])
        vended: dict[str, bool] = {}

        def comment_supply(transcript: dict[str, Any]) -> None:
            lbl = drive.observe.datarow_stamp(transcript)
            if lbl not in stash.ids:
                return
            if any(re.search(r"stock|avail", str(key).lower()) for key in transcript):
                vended[lbl] = drive.observe.vended_outward(transcript)
        for item in stash.items:
            route_hashmaps(item, comment_supply)
        budget = needs.disburse_top
        preferred: list[str] = []
        disbursed = 0.0
        arrows = drive.citecounter.cite_arrows()

        def precluded(lbl: str, hub: set) -> bool:
            if lbl in faltered or lbl in vetted or lbl in drive.refused_lines or vended.get(lbl):
                return True
            return lbl in preferred or lbl in arrows or referee.aisle_recognized(lbl) not in hub
        for kx, hub in enumerate(houses):
            metrics = several[kx]["quantity"] if several else needs.quantity()
            cluster = []
            for lbl in sorted(stash.read):
                if precluded(lbl, hub):
                    continue
                price = stash.price(lbl)
                if not be_numeral(price):
                    continue
                if be_numeral(budget) and price * metrics > budget:
                    continue
                cluster.append((lbl not in ongoing, price, lbl))
            if not cluster:
                return []
            _, price, lbl = min(cluster)
            preferred.append(lbl)
            disbursed += price * metrics
        if be_numeral(budget) and disbursed > budget:
            return []
        return preferred

    def befit_maintain(self) -> str:
        try:
            return self.maintain_towards_befit()
        except Exception as exn:
            self.drive.diary.append({"kind": "fit_hold_error", "error": f"{type(exn).__name__}: {exn}"[:300]})
            return ""

    def maintain_towards_befit(self) -> str:
        """Once, before a decline with an empty cart: re-check the audit's possible fits and hold the decline if one passes."""
        drive = self.drive
        needs, stash, channel = drive.needs, drive.stash, drive.channel
        if self.befit_maintained or drive.wagon.metrics or drive.maneuvers_residual() <= 4 or channel.extra_alt() < SEALING + 30:
            return ""
        contenders = [lbl for lbl in drive.referee.inspection_befits or [] if lbl in stash.ids and lbl not in drive.refused_lines]
        if not contenders:
            return ""
        self.befit_maintained = True
        fine = []
        for entity in contenders[:2]:
            if entity not in stash.read:
                continue
            verdict = drive.referee.validate_article(entity, "ORDER")
            if verdict and channel.extra_alt() > SEALING + 30:
                verdict = self.mediate(entity, verdict)
            if verdict or f"{entity}|{needs.edition}" not in drive.passed:
                continue
            fine.append(entity)
            if not needs.items():
                break
        if not fine:
            return ""
        if needs.items():
            edition = str(needs.edition)
            ongoing = {key.rpartition("|")[0] for key in drive.passed if key.rpartition("|")[2] == edition} - drive.refused_lines
            encompassed = {drive.referee.aisle_recognized(lbl) for lbl in ongoing} - {""}
            if any(hub and not hub & encompassed for hub in drive.shelf.hub_aisles()):
                return ""
        if len(fine) == 1:
            it, its, predicate = "it", "its", "it meets"
        else:
            it, its, predicate = "them", "their", "they meet"
        target = "a verified line for each product" if needs.items() else fine[0]
        return ("The decline audit flagged " + ", ".join(fine) + " as a possible fit, and the independent item check went "
                "through " + its + " full record and found " + predicate + " all requirements currently in force. Look at " + it
                + " before declining: if it fits, put " + target + " in the cart, check its stock and price again, and "
                "order; decline should you find that it breaks a requirement in force.")

    def ident_warehouse(self) -> bool:
        flags = [walk.get("id_order") for walk in self.drive.page_book.values() if "id_order" in walk]
        if False in flags:
            return False
        return flags.count(True) >= 2

    def commodity_offset(self, entity: str) -> int | None:
        drive = self.drive
        if not drive.needs.items():
            return None
        label = drive.referee.aisle_recognized(entity)
        if not label:
            return None
        return next((kx for kx, hub in enumerate(drive.shelf.hub_aisles()) if label in hub), None)

    def route_hub(self, kx: int | None) -> set[str]:
        houses = self.drive.shelf.hub_aisles()
        slot = kx or 0
        hub = set(houses[slot]) if houses and slot < len(houses) else set()
        if hub or kx is not None:
            return hub
        return set(self.drive.observe.aisle_captions())

    @staticmethod
    def dig_quantity(text: Any) -> tuple[str, str] | None:
        reduced = str(text).lower()
        if re.search(HX_EDGE, reduced):
            return None
        hit = re.match(r"\s*(\d+)\s+([a-z]+)", reduced)
        return (hit.group(1), hit.group(2)) if hit else None

    @staticmethod
    def quantity_component(text: Any) -> bool:
        reduced = str(text).lower()
        if not re.search(r"\d", reduced):
            return False
        if re.search(HX_EDGE, reduced):
            return True
        return bool(re.match(r"\s*\d+(?:[.,]\d+)?\s+[a-z]", reduced))

    def principal_vocables(self, trigger_operands: dict[str, Any], design: str) -> list[str]:
        drive = self.drive
        words = [wq for text in drive.observe.unbound_prose(trigger_operands) for wq in re.findall(design, text.lower())]
        if drive.guide_vocables:
            return words[: drive.guide_vocables]
        return words

    def route_transports(self, args: dict[str, Any], kx: int | None, alts: bool = False) -> set[Any]:
        """Which open requirements (or requirement alternatives, with alts) the search words carry."""
        drive = self.drive
        nigh = drive.observe.shut_vocables
        apiece_commodity = bool(drive.needs.items()) and kx is not None
        words = self.principal_vocables(args, r"[a-z0-9]{2,}")
        tokens = self.principal_vocables(args, r"[a-z0-9]+")
        conveyed: set[Any] = set()
        for n, need in enumerate(drive.needs.feats_residual()):
            if apiece_commodity and drive.speak.commodity_relating(need.partition(":")[0]) not in (kx, None):
                continue
            value = re.sub(r"\((?:stated|revealed[^)]*)\)", "", need.partition(":")[2]).lower()
            for a, component in enumerate(re.split(r"[,;/]|\bor\b|\beither\b|\bnor\b", value)):
                tag = (n, a) if alts else n
                quantity_alt = self.dig_quantity(component)
                if quantity_alt:
                    number, unit = quantity_alt
                    if number in tokens and any(tz == unit or nigh(tz, unit) for tz in tokens):
                        conveyed.add(tag)
                elif not self.quantity_component(component):
                    if any(nigh(wq, vz) for wq in words for vz in re.findall(r"[a-z0-9]{3,}", component)):
                        conveyed.add(tag)
        return conveyed

    def route_merit(self, args: dict[str, Any], kx: int | None) -> int:
        return len(self.route_transports(args, kx))

    def route_alternatives(self, kx: int | None) -> list[tuple[int, str, dict[str, Any]]]:
        """Unfinished id-ordered walks for product k worth paging on: (pages left, book key, walk), best first."""
        drive = self.drive
        observe = drive.observe
        hub = self.route_hub(kx)
        if not hub:
            return []
        depth = drive.guide_vocables or None

        def lead(walk: dict[str, Any]) -> set[str]:
            entered = " ".join(observe.unbound_prose(walk["args"])).lower()
            return {observe.root_alt(wq) for wq in re.findall(r"[a-z0-9]+", entered)[:depth]}
        caption_roots = {observe.root_alt(wq) for label in hub for wq in re.findall(r"[a-z0-9]{2,}", label)}
        apiece_commodity = bool(drive.needs.items()) and kx is not None
        measurement_roots = set()
        for need in drive.needs.feats_residual():
            if apiece_commodity and drive.speak.commodity_relating(need.partition(":")[0]) not in (kx, None):
                continue
            measurement_roots |= {observe.root_alt(wq) for wq in re.findall(r"[a-z0-9]{2,}", need.partition(":")[2].lower())}
        permitted_alt = measurement_roots | caption_roots
        tome = drive.page_book

        def workable(walk: dict[str, Any]) -> bool:
            return walk.get("id_order") is True and bool(walk.get("first"))
        finished = []
        for walk in tome.values():
            if not workable(walk) or observe.ensuing_folio(walk) is not None:
                continue
            if not lead(walk) or len(walk.get("row_shelf") or {}) < walk["total"]:
                continue
            finished.append((lead(walk), self.route_transports(walk["args"], kx, alts=True), drive.encompass.mv_manner(walk["args"])))
        best: dict[tuple, tuple[int, str, dict[str, Any]]] = {}
        for key, walk in tome.items():
            if not workable(walk) or observe.ensuing_folio(walk) is None:
                continue
            datarow_captions = walk.get("row_shelf") or {}
            if not self.route_transports(walk["args"], kx) or not any(label in hub for label in datarow_captions.values()):
                continue
            words = lead(walk)
            alts = self.route_transports(walk["args"], kx, alts=True)
            mode = drive.encompass.mv_manner(walk["args"])
            if any(accomplished_manner == mode and (accomplished_vocables <= words or (accomplished_vocables <= permitted_alt and alts <= accomplished_alternates))
                   for accomplished_vocables, accomplished_alternates, accomplished_manner in finished):
                continue
            pages = -(-walk["total"] // max(1, walk["size"])) - len(walk["read"])
            if not 0 < pages <= 2 * ROUTE_FOLIOS:
                continue
            duplicate = (walk["total"], walk["first"])
            if duplicate not in best or pages < best[duplicate][0]:
                best[duplicate] = (pages, key, walk)
        alternatives = list(best.values())
        if drive.encompass.every_vocables():
            def embedded(pz: tuple, oq: tuple) -> bool:
                minee, further = lead(pz[2]), lead(oq[2])
                if minee < further and not (further - minee) <= permitted_alt and pz[0] <= ROUTE_FOLIOS:
                    return True
                return further < minee and (minee - further) <= permitted_alt
            alternatives = [oq for oq in alternatives if not any(embedded(pz, oq) for pz in alternatives if pz is not oq)]
        return sorted(alternatives, key=lambda pz: (pz[0], -self.route_merit(pz[2]["args"], kx), pz[1]))

    def route_strategy(self, budget: int, solely: set | None = None) -> list[tuple[int | None, str, int]]:
        """Share a page budget over the products' best walks: [(product, book key, pages)]."""
        drive = self.drive
        if budget <= 0 or drive.needs.priority() != "cheapest" or not drive.role("search"):
            return []
        if not drive.observe.folio_operand() or not self.ident_warehouse():
            return []
        wished_alt = []
        for kx in (list(range(len(drive.needs.items()))) or [None]):
            if kx in self.routed or (solely is not None and kx not in solely):
                continue
            alternatives = self.route_alternatives(kx)
            if alternatives:
                pages, key, _ = alternatives[0]
                wished_alt.append((pages, -1 if kx is None else kx, kx, key))
        wished_alt.sort(key=lambda element_alt: (element_alt[0], element_alt[1]))
        plan = []
        room = budget
        for pages, _, kx, key in wished_alt:
            take = min(pages, ROUTE_FOLIOS, room)
            if take <= 0:
                break
            plan.append((kx, key, take))
            room -= take
        return plan

    def route_select(self, kx: int | None) -> tuple[str, float] | None:
        """Cheapest current pick for product k: a cart line, else a passed item."""
        drive = self.drive
        vended = drive.shelf.inert_orderlines()

        def we(lbl: str) -> bool:
            return kx is None or self.commodity_offset(lbl) == kx
        alternatives = [lbl for lbl in drive.wagon.metrics if we(lbl) and lbl not in vended]
        if not alternatives:
            edition = str(drive.needs.edition)
            passed = [key.rpartition("|")[0] for key in drive.passed if key.rpartition("|")[2] == edition]
            alternatives = [lbl for lbl in passed if lbl not in drive.refused_lines and lbl not in vended and we(lbl)]
        costed = []
        for lbl in alternatives:
            price = drive.stash.price(lbl)
            if be_numeral(price):
                costed.append((lbl, float(price)))
        return min(costed, key=lambda duo: (duo[1], duo[0])) if costed else None

    def orderline_hunttext(self, kx: int | None) -> dict[str, Any] | None:
        """Search for the pick's product line: its maker plus the first distinctive title word."""
        drive = self.drive
        needs = drive.needs
        select = self.route_select(kx)
        stencil = self.hunt_stencil()
        if not select or not stencil:
            return None
        transcript: dict[str, Any] = {}

        def recall(contender: dict[str, Any]) -> None:
            nonlocal transcript
            if drive.observe.datarow_stamp(contender) == select[0] and drive.shelf.headline_relating(contender):
                transcript = contender
        for item in drive.stash.items:
            route_hashmaps(item, recall)
        creator = next((vlu for key, vlu in transcript.items()
                      if isinstance(vlu, str) and vlu.strip() and re.search(r"brand|make|manufactur", str(key).lower())
                      and "id" not in str(key).lower()), "")
        creator_vocables = re.findall(r"[a-z0-9]+", creator.lower())
        if not creator_vocables:
            return drive.encompass.headline_orderline(kx, transcript, stencil)
        omit = set(creator_vocables) | ANCHOR_VOCABLES | {wq for label in self.route_hub(kx) for wq in re.findall(r"[a-z]+", label)}
        omit |= {wq for need in needs.feats_residual() for wq in re.findall(r"[a-z]+", need.lower())}
        products = needs.items() or [{"product": str(needs.browsing.get("product") or "")}]
        omit |= {wq for element_alt in products for wq in re.findall(r"[a-z]+", element_alt["product"].lower())}
        headline_vocables = re.findall(r"[a-z][a-z'-]{3,}", drive.shelf.headline_relating(transcript).lower())
        marque = creator_vocables[0]
        if marque in headline_vocables:
            headline_vocables = headline_vocables[headline_vocables.index(marque) + 1:]
        orderline_vocables = [wq for wq in headline_vocables if wq.strip("'-") not in omit and "'" not in wq]
        if not orderline_vocables:
            return None
        text = f"{marque} {orderline_vocables[0]}"
        if self.drove_hunttext(text):
            return None
        return {**stencil[0], stencil[1]: text}

    def route_datarows(self, kx: int | None, selectors: list[str], maximal: int) -> list[str]:
        """Unread in-stock in-budget rows the walks showed for product k that undercut the current pick."""
        drive = self.drive
        needs, stash = drive.needs, drive.stash
        hub = self.route_hub(kx)
        tome = drive.page_book
        apex = max([len(self.route_transports(wq["args"], kx)) for wq in tome.values() if wq.get("id_order") is True] or [0])
        exhibited: set[str] = set()
        for key, walk in tome.items():
            conveyed = self.route_transports(walk["args"], kx) if walk.get("id_order") is True else set()
            if (conveyed and len(conveyed) == apex) or key in selectors:
                exhibited |= {lbl for lbl, label in (walk.get("row_shelf") or {}).items() if label in hub}
        select = self.route_select(kx)
        if not select:
            browse_rates = sorted((float(stash.price(lbl)), lbl) for lbl in exhibited
                                 if lbl in stash.read and be_numeral(stash.price(lbl)))
            select = (browse_rates[0][1], browse_rates[0][0]) if browse_rates else None
        several = needs.items()
        metrics = several[kx]["quantity"] if kx is not None and kx < len(several) else needs.quantity()
        budget = needs.disburse_top
        vended: dict[str, bool] = {}

        def comment_supply(transcript: dict[str, Any]) -> None:
            if any(isinstance(vlu, str) and vlu in exhibited and "id" in str(key).lower() for key, vlu in transcript.items()):
                vended[drive.observe.datarow_stamp(transcript)] = drive.observe.vended_outward(transcript)
        for item in stash.items:
            route_hashmaps(item, comment_supply)
        thriftier = []
        for lbl in exhibited:
            price = stash.price(lbl)
            if lbl in stash.read or lbl in drive.refused_lines or vended.get(lbl) or not be_numeral(price):
                continue
            if be_numeral(budget) and price * metrics > budget:
                continue
            if select and price >= select[1]:
                continue
            thriftier.append((price, lbl))
        thriftier.sort()
        return [lbl for _, lbl in thriftier[:maximal]]

    def folio_distill(self, obv: Any) -> str:
        drive = self.drive
        observe = drive.observe
        rows: list[dict[str, Any]] = []

        def collect_alt(transcript: dict[str, Any]) -> None:
            if len(transcript) >= 4 and any(isinstance(vlu, str) and vlu in drive.stash.ids and "id" in str(key).lower()
                                        for key, vlu in transcript.items()):
                rows.append(transcript)
        route_hashmaps(obv, collect_alt, ceiling=3)

        def depict(row: dict[str, Any]) -> str:
            price = next((vlu for key, vlu in row.items() if be_numeral(vlu) and re.search(r"price|cost", str(key).lower())), None)
            exhibited = "?" if price is None else price
            toggle = " SOLD OUT" if observe.vended_outward(row) else ""
            return f"{observe.datarow_stamp(row)} {exhibited}{toggle} '{drive.shelf.headline_relating(row)[:70]}' [{observe.aisle_relating(row)}]"
        lines = [depict(row) for row in rows[:12]]
        return "Rows on this page (id, price, title, category): " + ("; ".join(lines) if lines else "none")

    def route_trigger(self, args: dict[str, Any], utter: str) -> bool:
        """Run one scripted search; the tool reply shown to the model is a compact digest of any new rows."""
        drive = self.drive
        stash = drive.stash
        suggestion = {"id": f"walk_{len(drive.chat)}_{drive.channel.beat}", "type": "function",
                    "function": {"name": drive.role("search"), "arguments": json.dumps(args)}}
        drive.chat.append({"role": "assistant", "content": utter, "tool_calls": [suggestion]})
        preceding = len(stash.items)
        drive.covering = True
        try:
            rejoinders = drive.forward([suggestion], automated=True)
        finally:
            drive.covering = False
        if not rejoinders:
            drive.chat.append({"role": "tool", "tool_call_id": suggestion["id"], "content": "Not run."})
            return False
        cuid, text = rejoinders[0]
        increased = len(stash.items) > preceding
        if not increased:
            drive.chat.append({"role": "tool", "tool_call_id": cuid, "content": text})
            return False
        body = self.folio_distill(stash.items[-1])
        indicator = "\n[note] "
        note = text.split(indicator, 1)[1] if indicator in text else ""
        if note and re.search(r"shopper|Meaning:|REFUSED", note):
            body += indicator + note
        drive.chat.append({"role": "tool", "tool_call_id": cuid, "content": body})
        return True

    def route_access(self) -> bool:
        channel = self.drive.channel
        if channel.done:
            return False
        return channel.extra_alt() >= SEALING + 30

    @staticmethod
    def interlace(clusters: list[list[str]], maximal: int) -> list[str]:
        """Round-robin over the pools (consuming them), skipping repeats, up to most entries."""
        blended: list[str] = []
        while len(blended) < maximal and any(clusters):
            for cluster in clusters:
                while cluster and cluster[0] in blended:
                    cluster.pop(0)
                if cluster and len(blended) < maximal:
                    blended.append(cluster.pop(0))
        return blended

    def shorten_waitlist(self, waitlist: list[str], lot: Any, floor: int) -> None:
        def cost() -> int:
            return 1 if lot and len(waitlist) >= 2 else len(waitlist)
        while waitlist and cost() >= self.drive.maneuvers_residual() - floor:
            waitlist.pop()

    def browse_folios(self, plan: list[tuple[int | None, str, int]], field: str, floor: int, bulletin: list[str]) -> bool:
        drive = self.drive
        channel = drive.channel
        tome = drive.page_book
        advanced = False

        def obstructed() -> bool:
            return drive.maneuvers_residual() <= floor + 1 or channel.done or channel.snapped or channel.extra_alt() < SEALING + 30
        for kx, route_selector, count in plan:
            self.routed.add(kx)
            walk = tome.get(route_selector)
            if not walk:
                continue
            query = " ".join(drive.observe.unbound_prose(walk["args"]))
            folios_browse = 0
            while folios_browse < count:
                forthcoming = drive.observe.ensuing_folio(walk)
                if forthcoming is None or obstructed():
                    break
                uttered = ("The cheapest is wanted, and this store orders search results by a fixed id rather than by price, "
                        f"so I read the next page of '{query}'.")
                if not self.route_trigger({**walk["args"], field: forthcoming}, uttered):
                    break
                folios_browse += 1
                advanced = True
            seen = len(walk.get("row_shelf") or {})
            bulletin.append(f"'{query}': read {folios_browse} more page(s), {seen} of its {walk['total']} matches seen")
        return advanced

    def browse_orderlines(self, plan: list[tuple[int | None, str, int]], field: str, floor: int, bulletin: list[str],
                   lines: dict[Any, str]) -> bool:
        drive = self.drive
        channel = drive.channel
        observe = drive.observe
        advanced = False

        def might_folio(line: dict[str, Any] | None, surplus: int) -> bool:
            return bool(line and surplus < 2 and line["total"] <= 3 * max(1, line["size"]) and observe.ensuing_folio(line) is not None
                        and drive.maneuvers_residual() > floor + 3 and not channel.done and channel.extra_alt() > SEALING + 30)
        for kx, _, _ in plan:
            if drive.maneuvers_residual() <= floor + 2 or not self.route_access():
                break
            trigger_operands = self.orderline_hunttext(kx)
            if not trigger_operands:
                continue
            uttered = ("I also search the pick's product line (its maker plus the word naming the line): other sizes and "
                    "colours in a line are listed separately and may cost less.")
            if not self.route_trigger(trigger_operands, uttered):
                continue
            advanced = True
            lines[kx] = json.dumps({key: vlu for key, vlu in trigger_operands.items() if key != field}, sort_keys=True, default=str)
            line = drive.page_book.get(lines[kx])
            surplus = 0
            while might_folio(line, surplus):
                if not self.route_trigger({**line["args"], field: observe.ensuing_folio(line)}, "Turning to the following page of the line search."):
                    break
                surplus += 1
            bulletin.append(f"the pick's line '{' '.join(observe.unbound_prose(trigger_operands))}'")
        return advanced

    def drive_route(self, plan: list[tuple[int | None, str, int]], floor: int, roundup: list | None = None, mend: bool = False) -> bool:
        """Read planned walk pages, line searches and ceiling walks, then queue the cheapest unread rows they showed."""
        drive = self.drive
        scrolling = drive.observe.folio_operand()
        if not (plan or roundup) or not scrolling or not drive.role("search"):
            return False
        field = scrolling[0]
        bulletin: list[str] = []
        lines: dict[Any, str] = {}
        drove = self.browse_folios(plan, field, floor, bulletin)
        if self.browse_orderlines(plan, field, floor, bulletin, lines):
            drove = True
        summit_datarows: dict[Any, list[str]] = {}
        for kx in dict.fromkeys(roundup or []):
            keep = floor if mend else max(floor, drive.admission.checkout_expected() + 2)
            if drive.maneuvers_residual() <= keep + 1 or not self.route_access():
                break
            upshot = drive.encompass.ceiling_walk(kx, floor, mend)
            if upshot:
                drove = True
                bulletin.append(upshot[0])
                summit_datarows[kx] = list(upshot[1])
        if not drove:
            return False
        lot = drive.shelf.lot_utility()
        if lot:
            maximal = lot[2]
        else:
            maximal = ROUNDUP_ACCESSES if summit_datarows else 2
        scheduled = [kx for kx, _, _ in plan]
        clusters = [self.route_datarows(kx, [lines.get(kx, "")], maximal) for kx in scheduled] + list(summit_datarows.values())
        waitlist = self.interlace(clusters, maximal)
        self.shorten_waitlist(waitlist, lot, floor)
        houses = set().union(*(self.route_hub(kx) for kx in scheduled + list(summit_datarows)))
        drive.rank_checked = {stamp_alt for stamp_alt in drive.rank_checked if drive.referee.aisle_recognized(stamp_alt.split("|")[0]) not in houses}
        drive.order_vetted = False
        if waitlist:
            itemized = ", ".join(f"{lbl} ({drive.stash.price(lbl):g})" for lbl in waitlist)
            midst = ("Next I open the cheapest unread in-stock rows within the budget that these searches showed below "
                      "each pick's price: " + itemized + ". ")
        else:
            midst = "These searches showed no unread in-stock row within the budget below the pick's price. "
        abstract = ("[cheapest walk] " + "; ".join(bulletin) + ". " + midst
                   + "For every product, choose the lowest-priced listing whose own fields meet every requirement in "
                   "force; the rank check runs again on the cart call and on the order.")
        if waitlist:
            browses = drive.browses
            browses.access_waitlist = waitlist
            browses.access_revalidate = True
            browses.access_utter = "The cheapest is wanted: I open the cheapest unread rows the walk turned up."
            browses.access_intention, browses.access_bottom = "walk", floor
            drive.referee.access_designated()
        if drive.chat and drive.chat[-1].get("role") == "tool":
            rear = "" if waitlist else "\n" + drive.standing_orderline()
            drive.chat[-1]["content"] = str(drive.chat[-1]["content"]) + "\n" + abstract + rear
        drive.diary.append({"kind": "walk", "plan": [[kx, n] for kx, _, n in plan], "ceiling": list(summit_datarows), "opened": waitlist})
        return True

    def reject_befits(self) -> set[str]:
        drive = self.drive
        edition = str(drive.needs.edition)
        ongoing = {key.rpartition("|")[0] for key in drive.passed if key.rpartition("|")[2] == edition}
        faltered = {key.split("|")[0] for key, verdict in drive.rulings.items() if verdict and key.split("|")[1:2] == [edition]}
        named = set(drive.referee.inspection_befits or []) - faltered
        return (ongoing | named) - drive.refused_lines

    def reject_deficient(self) -> list[int | None]:
        """Products (index, or None for a single product) with no known fit yet."""
        drive = self.drive
        encompassed = {drive.referee.aisle_recognized(lbl) for lbl in self.reject_befits()} - {""}
        if drive.needs.items():
            return [kx for kx, hub in enumerate(drive.shelf.hub_aisles()) if hub and not hub & encompassed]
        return [] if set(drive.observe.aisle_captions()) & encompassed else [None]

    def reject_routes(self) -> list[str]:
        access_necessities = self.drive.needs.feats_residual()

        def couched(trigger_operands: dict[str, Any], kx: int | None) -> int:
            return sum(1 for n in self.route_transports(trigger_operands, kx)
                       if n < len(access_necessities)
                       and not re.search(r"as shown|same as|the page|sold by|redirect|refer", access_necessities[n].lower()))
        preferred: list[str] = []
        for kx in self.reject_deficient():
            alternatives = [sel for sel in self.route_alternatives(kx) if couched(sel[2]["args"], kx)]
            alternatives.sort(key=lambda pz: (-couched(pz[2]["args"], kx), -self.route_merit(pz[2]["args"], kx), pz[0], pz[1]))
            crisp = next((key for _, key, _ in alternatives if key not in preferred), None)
            if crisp is not None:
                preferred.append(crisp)
        return preferred[:2]

    def datarow_clusters(self, houses: list[tuple[set[str], int]], ids: set[str] | None, under: list[float] | None = None) -> list[list[str]]:
        """Per home: unread, unrefused, in-stock, in-budget ids (optionally limited to ids / below a price), cheapest first."""
        drive = self.drive
        stash, observe = drive.stash, drive.observe
        budget = drive.needs.disburse_top
        captions: dict[str, str] = {}
        vended: dict[str, bool] = {}

        def note(transcript: dict[str, Any]) -> None:
            lbl = observe.datarow_stamp(transcript)
            if lbl not in stash.ids or not observe.aisle_relating(transcript):
                return
            captions[lbl] = observe.aisle_relating(transcript)
            if any(re.search(r"stock|avail", str(key).lower()) for key in transcript):
                vended[lbl] = observe.vended_outward(transcript)
        for item in stash.items:
            route_hashmaps(item, note)

        def admissible(lbl: str, label: str, hub: set, metrics: int, n: int, price: Any) -> bool:
            if label not in hub or lbl in stash.read or lbl in drive.refused_lines or vended.get(lbl):
                return False
            if (ids is not None and lbl not in ids) or not be_numeral(price):
                return False
            if be_numeral(budget) and price * metrics > budget:
                return False
            return under is None or price < under[n]
        clusters = []
        for n, (hub, metrics) in enumerate(houses):
            ranked = []
            for lbl, label in captions.items():
                price = stash.price(lbl)
                if admissible(lbl, label, hub, metrics, n, price):
                    ranked.append((float(price), lbl))
            ranked.sort()
            clusters.append([lbl for _, lbl in ranked])
        return clusters

    def above_allowance_datarows(self) -> list[list[str]]:
        """When the cheapest known fits of all products together exceed the budget: cheaper unread rows per product."""
        drive = self.drive
        try:
            several, budget = drive.needs.items(), drive.needs.disburse_top
            if not several or drive.needs.priority() == "cheapest" or not be_numeral(budget):
                return []
            houses = drive.shelf.hub_aisles()
            fits = self.reject_befits()
            bottoms: list[float] = []
            stakes: list[float] = []
            total = 0.0
            for kx, hub in enumerate(houses):
                rates = [float(drive.stash.price(lbl)) for lbl in fits
                          if drive.referee.aisle_recognized(lbl) in hub and be_numeral(drive.stash.price(lbl))]
                if not hub or not rates:
                    return []
                minimal = min(rates)
                share = minimal * several[kx]["quantity"]
                bottoms.append(minimal)
                stakes.append(share)
                total += share
            if total <= budget:
                return []
            clusters = self.datarow_clusters([(hub, several[kx]["quantity"]) for kx, hub in enumerate(houses)], None, bottoms)
            largest_primary = sorted(range(len(clusters)), key=lambda kx: -stakes[kx])
            clusters = [clusters[kx] for kx in largest_primary]
            return clusters if any(clusters) else []
        except Exception as exn:
            drive.diary.append({"kind": "over_budget_error", "error": f"{type(exn).__name__}: {exn}"[:300]})
            return []

    def decline_reads(self) -> str:
        try:
            return self.waitlist_reject_browses()
        except Exception as exn:
            self.drive.diary.append({"kind": "decline_reads_error", "error": f"{type(exn).__name__}: {exn}"[:300]})
            return ""

    def waitlist_reject_browses(self) -> str:
        """Once, before a decline: plan extra page reads and cheaper-row opens, and explain them to the model."""
        drive = self.drive
        channel = drive.channel
        lot = drive.shelf.lot_utility()
        if self.reject_browse_accomplished or channel.extra_alt() < SEALING + 60 or not (drive.role("details") or lot) or drive.wagon.metrics:
            return ""
        opens = 1 if lot else 2
        room = min(3, drive.maneuvers_residual() - opens - 6)
        selectors = []
        if room > 0 and drive.observe.folio_operand() and drive.role("search") and self.ident_warehouse():
            selectors = self.reject_routes()
        thriftier = self.above_allowance_datarows() if drive.maneuvers_residual() > opens + 6 else []
        own_unread = []
        if not selectors and not thriftier and drive.maneuvers_residual() > opens + 6:
            # (rows of the wanted products' own categories that searches showed but nobody opened: a required value often
            # sits in a field their titles do not show)
            several = drive.needs.items()
            houses = [(self.route_hub(kx), several[kx]["quantity"] if kx is not None and kx < len(several) else drive.needs.quantity())
                      for kx in self.reject_deficient()]
            own_unread = [cluster for cluster in self.datarow_clusters(houses, None) if cluster] if houses else []
        if not selectors and not thriftier and not own_unread:
            return ""
        self.reject_browse_accomplished = True
        self.reject_own = bool(own_unread)
        self.reject_waitlist = (selectors, room, 6)
        components = []
        if selectors:
            named = "; ".join(f"'{' '.join(drive.observe.unbound_prose(drive.page_book[key]['args']))}'" for key in selectors)
            components.append("searches carrying required values matched more listings than were read (" + named + "), and "
                         "this store returns results in a fixed id order, not by fit or price, so fits are as likely on "
                         f"unread pages as on read ones: the code will read as many as {room} of those pages for you and open "
                         "the cheapest in-stock rows within the budget that they show")
        if thriftier:
            components.append("taken together, the listings found for all products exceed the budget: the cheapest unread "
                         "in-stock rows priced below each product's pick are opened for you as well")
        if own_unread:
            components.append("search rows of the requested category that nobody opened may hold a required value in a field "
                         "their titles do not show: the cheapest in-stock ones within the budget are opened for you")
        target = "a line for each product, at that product's quantity" if drive.needs.items() else "the item"
        return ("Before a decline, " + "; ".join(components) + ". Judge those records against the requirements now in force by "
                "meaning; buy what fits (" + target + "), and decline only if every one of them fails.")

    def drive_reject_browses(self) -> bool:
        drive = self.drive
        channel = drive.channel
        selectors, maximal_folios, floor = self.reject_waitlist
        self.reject_waitlist = None
        scrolling = drive.observe.folio_operand()
        tome = drive.page_book
        lot = drive.shelf.lot_utility()
        opens = 1 if lot else 2
        read = 0
        routed: list[str] = []

        def able_browse() -> bool:
            return (read < maximal_folios and drive.maneuvers_residual() > floor + opens and not channel.done and not channel.snapped
                    and channel.extra_alt() > SEALING + 30)
        for key in (selectors if scrolling else []):
            walk = tome.get(key)
            if not walk:
                continue
            routed.append(key)
            query = " ".join(drive.observe.unbound_prose(walk["args"]))
            while able_browse():
                forthcoming = drive.observe.ensuing_folio(walk)
                if forthcoming is None:
                    break
                uttered = (f"Before any decline, I go through the next page of '{query}': this store lists results in a fixed id "
                        "order, so fits may be on pages not yet read.")
                if not self.route_trigger({**walk["args"], scrolling[0]: forthcoming}, uttered):
                    break
                read += 1
        seen: set[str] = set()
        for key in routed:
            seen |= set((tome.get(key) or {}).get("row_shelf") or {})
        several = drive.needs.items()
        houses = []
        for kx in self.reject_deficient():
            metrics = several[kx]["quantity"] if kx is not None and kx < len(several) else drive.needs.quantity()
            houses.append((self.route_hub(kx), metrics))
        clusters = (self.datarow_clusters(houses, seen) if routed else []) + self.above_allowance_datarows()
        if getattr(self, "reject_own", False):
            clusters += self.datarow_clusters(houses, None)
            self.reject_own = False
        maximal = lot[2] if lot else 2
        waitlist = self.interlace(clusters, maximal)
        self.shorten_waitlist(waitlist, lot, floor)
        drive.diary.append({"kind": "decline_reads", "pages": read, "opened": waitlist})
        if waitlist and not channel.done:
            browses = drive.browses
            browses.access_waitlist = waitlist
            browses.access_utter = "Before declining, I open the cheapest unread in-budget rows that those pages and picks show."
            browses.access_intention, browses.access_bottom = "decline", floor
            drive.referee.access_designated()
        elif drive.chat and drive.chat[-1].get("role") == "tool":
            drive.chat[-1]["content"] = (str(drive.chat[-1]["content"]) + "\nThose pages showed no unread in-stock row within "
                                       "the budget.\n" + drive.standing_orderline())
        return read > 0 or bool(waitlist)


class BuyerRejoinderPhase:
    def undertake(self, drive) -> bool:
        expected_alt = drive.needs.rejoinder_expected
        if expected_alt:
            drive.needs.rejoinder_expected = ""
            return drive.speak.rejoinder_reverse(expected_alt)
        return False


class WaitlistedBrowsesPhase:
    def undertake(self, drive) -> bool:
        return drive.referee.access_designated()


class AisleCombPhase:
    def undertake(self, drive) -> bool:
        maintained_towards_inquiries = drive.soliciting and not drive.speak.vacant_strategies
        return False if maintained_towards_inquiries else drive.observe.comb_phase()


class MeasurementFoliosPhase:
    def undertake(self, drive) -> bool:
        return drive.encompass.value_pages()


class ScheduledInquiryPhase:
    def undertake(self, drive) -> bool:
        if not drive.soliciting:
            return False
        return drive.speak.broach_scheduled()


class CitationLeapPhase:
    def undertake(self, drive) -> bool:
        return drive.citecounter.cite_phase()


class InexpensiveRoutePhase:
    def undertake(self, drive) -> bool:
        return drive.encompass.route_phase()


class OperatorStintPhase:
    def __init__(self):
        self.sqn_nil = 0
        self.dormant = 0
        self.lapses = 0
        self.overlooked = 0
        self.ultimate_sound = _TALLY.now()

    def undertake(self, drive) -> bool:
        channel = drive.channel
        cycled = drive.stuck >= JAMMED_CONSTRAINT
        oracles = OPERATOR[1:] + OPERATOR[:1] if cycled else OPERATOR
        reply = channel.hunttext_oracle(drive.chat, oracles, 2500, drive.utilitykit)
        if reply is None:
            return self.afterward_overlook(drive, oracles)
        self.ultimate_sound = _TALLY.now()
        self.overlooked = 0
        calls, self.sqn_nil = drive.tool_calls(reply, self.sqn_nil)
        uttered = reply.get("content") or ""
        if not isinstance(uttered, str):
            uttered = qua_prose(uttered)
        finalizing = {drive.role(component) for component in ("cart", "order", "decline")} - {None}
        if drive.soliciting and any(trigger["function"]["name"] in finalizing for trigger in calls):
            if drive.speak.broach_scheduled(impulse=True):
                return True
        if not calls:
            self.dormant += 1
            drive.chat.append({"role": "assistant", "content": uttered or "(no tool call)"})
            prod = "Each reply must use at least one of the tools on offer. Carry on.\n" + drive.standing_orderline()
            drive.chat.append({"role": "user", "content": prod})
            return self.dormant < DORMANT_CONSTRAINT
        self.dormant = 0
        drive.diary.append({"kind": "model", "content": uttered[:2000], "calls": [trigger["function"] for trigger in calls]})
        drive.chat.append({"role": "assistant", "content": uttered or None, "tool_calls": calls})
        beat_were = channel.beat
        drive.speak.subsequent_rejoinders(drive.forward(calls))
        drive.stuck = 0 if channel.beat != beat_were else drive.stuck + 1
        return True

    def afterward_overlook(self, drive, oracles: list[str]) -> bool:
        channel = drive.channel
        self.overlooked += 1
        if not channel.workable(oracles, True):
            return False
        wordless_overly_extended = _TALLY.now() - self.ultimate_sound > DOWNWARD_AFTERWARD
        if wordless_overly_extended and (self.overlooked >= 2 or (channel.downward() and len(channel.trigger_oracles) >= 2)):
            return False
        if channel.trigger_falter not in ("empty", "refused"):
            return True
        self.lapses += 1
        return self.lapses < 2


class DirectingOrderline:
    def __init__(self, text, phrase, start, end, dissected):
        self.interior = text[start:end]
        self.exterior = text[:start] + " " + text[end:]
        self.text = text
        self.dissected = dissected
        self.phrase = phrase

    def interior_solely(self, prop):
        return bool(prop) and feat_designated(prop, self.interior) and not feat_designated(prop, self.exterior)

    def depicts_further(self, req, name):
        if self.interior_solely(name(req)):
            return True
        rear_vocables = set(cite_lexs(str(req).partition(":")[2]))
        if not rear_vocables:
            return False
        return rear_vocables.issubset(cite_lexs(self.interior)) and rear_vocables.isdisjoint(cite_lexs(self.exterior))


class CitationDesk:
    def __init__(self, drive):
        self.drive = drive
        self.leap_triggers = 0
        self.phases_disbursed = 0
        self.divert_notified = set()
        self.contender_idents = set()
        self.offset_jotting = None
        self.offset_noticed_upon = None
        self.merchant_column_jotting = None
        self.column_designations_jotting = None
        self.awaiting_notifies = None
        self.tarry_journaled = None

    # -- convenience accessors
    @property
    def needs(self):
        return self.drive.needs

    @property
    def stash(self):
        return self.drive.stash

    def access_cites(self):
        return [cite for cite in self.drive.refs or [] if cite.get("status") == "open"]

    @staticmethod
    def front_feat(head):
        return re.split(r"\s*\(", str(head))[0]

    def commodity_varieties(self):
        logs = self.needs.items() or [{"product": str(self.needs.browsing.get("product") or "")}]
        discovered = set()
        for element_alt in logs:
            discovered.update(variety_vocables(element_alt["product"]))
        return discovered

    # -- registering the references to follow
    def cite_enlist(self):
        browsing = self.needs.browsing
        goals = browsing.get("reference_targets")
        infos = [tz for tz in goals if isinstance(tz, dict)] if isinstance(goals, list) else []
        itemized_cites = browsing.get("references")
        itemized = set(map(str, itemized_cites)) if isinstance(itemized_cites, list) else set()
        for req in list(self.needs.written):
            text = str(req)
            value = text.partition(":")[2]
            if cite_prose(value) or (text in itemized and re.search(VARY_VOCABLES, value.lower())):
                self.cite_append(text, infos, self.drive.request, "request")

    def cite_unrecorded(self, refs, monitored):
        def obeyed(req):
            return any(self.cite_concord(known, req) and (known.get("status") == "resolved" or known.get("chain"))
                       for known in monitored)
        try:
            return [req for req in refs if not obeyed(req)]
        except Exception:
            return refs

    def cite_grab(self):
        try:
            for done in self.drive.refs:
                if done["status"] == "resolved" and any(
                        self.cite_concord(done, req) and req != done["now"] for req in self.needs.feats_residual()):
                    self.switch_amidst(done, done["now"])
            inbound = list(self.needs.cite_tidings or [])
            self.needs.cite_tidings = []
            for tidings in inbound:
                need = str(tidings.get("need") or "")
                infos = list(tidings.get("infos") or [])
                line = str(tidings.get("line") or "")
                transcript = self.cite_append(need, infos, line, "reply")
                if transcript:
                    self.lineup_citation(transcript["req"])
        except Exception as failure:
            self.drive.diary.append({"kind": "ref_take_error", "error": f"{type(failure).__name__}: {failure}"[:300]})

    def lineup_citation(self, req):
        browsing = self.needs.browsing
        known = browsing.get("references")
        if not isinstance(known, list):
            known = []
            browsing["references"] = known
        if req not in known:
            known.append(req)

    def intel_towards(self, infos, prop, tag):
        for optn in infos:
            if not equal_feat(str(optn.get("property") or ""), prop):
                continue
            if not tag or not optn.get("product") or not self.needs.items():
                return optn
            if variety_vocables(optn.get("product")) & variety_vocables(tag):
                return optn
        return None

    def previously_access(self, req, prop, tag):
        def clash(known):
            if known["req"] == req:
                return True
            return equal_feat(known["prop"], prop) and variety_vocables(known["tag"]) == variety_vocables(tag)
        return any(clash(known) for known in self.drive.refs)

    def association_relating(self, value, intel, prop, provenance, target):
        relation, tolerance, unit = cite_association(value)
        if relation in ("same", "differ") and re.search(PARTNER_VOCABLES, cite_extent(prop, provenance).lower()):
            # (the request's own words ask for the counterpart: a paraphrase of the requirement must not turn it into
            # an exclusion or a plain copy)
            return "counterpart", None, ""
        notified = str(intel.get("relation") or "").lower() if isinstance(intel, dict) else ""
        if relation == "same" and notified == "within" and be_numeral(intel.get("tolerance")):
            relation, tolerance, unit = "within", float(intel["tolerance"]), str(intel.get("unit") or "")
        if relation != "same":
            return relation, tolerance, unit
        # differ wording counts only when it sits between the property and the seller's name
        extent = cite_extent(prop, provenance).lower()
        seller = str(target.get("seller") or "")
        snip = extent.find(seller.lower()) if seller.strip() else -1
        overt = re.search(r"differ|not the same|exclu|apart|other than|unlike", notified)
        if overt or (snip > 0 and re.search(VARY_VOCABLES, extent[:snip])):
            return "differ", None, ""
        return relation, tolerance, unit

    def cite_append(self, req, infos, provenance, origin):
        text = str(req)
        head, _, value = text.partition(":")
        prop = self.front_feat(head).strip()
        tagnames = re.findall(r"\(([^)]*)\)", head)
        tag = tagnames[-1].strip() if tagnames else ""
        if not prop or trash_selector(prop) or self.previously_access(req, prop, tag):
            return None
        intel = self.intel_towards(infos, prop, tag)
        target = self.cite_destination(intel, value, provenance, prop)
        if not target:
            self.drive.diary.append({"kind": "ref_skip", "req": text[:200]})
            return None
        relation, tolerance, unit = self.association_relating(value, intel, prop, provenance, target)
        transcript = {"req": text, "head": head.strip(), "prop": prop, "tag": tag, "relation": relation, "tol": tolerance,
                  "unit": unit, "target": target, "first": dict(target), "chain": [], "seen": {self.destination_selector(target)},
                  "judged": set(), "queries": [], "pages": 0, "hops": 0, "tries": 0, "fails": 0, "status": "open",
                  "value": "", "goes_with": [], "now": "", "origin": origin, "unindexed": False}
        self.drive.refs.append(transcript)
        self.drive.diary.append({"kind": "ref_open", "req": text[:200], "target": target, "relation": relation, "origin": origin})
        return transcript

    def cite_destination(self, intel, value, provenance, prop):
        seller = kind = ""
        details = []
        if isinstance(intel, dict):
            seller = str(intel.get("seller") or "").strip()
            kind = str(intel.get("kind") or "").strip()
            details = self.tidy_textstrings(intel.get("details"))

        def credible(name):
            return self.cite_merchant_sound(name, provenance)

        dissected = None
        for text in (cite_extent(prop, provenance), value):
            obtained = cite_dissect(text, credible)
            if obtained and credible(obtained["seller"]):
                dissected = obtained
                break
        if credible(seller):
            if not details and dissected and cite_designation(dissected["seller"]) == cite_designation(seller):
                details = dissected["details"]
        elif dissected:
            seller = dissected["seller"]
            kind = kind or dissected["kind"]
            details = details or dissected["details"]
        else:
            return None
        provenance_vocables = set(cite_lexs(provenance))
        kept = [z for z in dict.fromkeys(details) if provenance_vocables & set(cite_lexs(z))][:6]
        if dissected:
            kind = kind or dissected["kind"]
        return {"seller": seller, "kind": kind[:80], "details": kept}

    def cite_merchant_sound(self, seller, provenance):
        name = str(seller or "").strip()
        if len(re.sub(r"[^A-Za-z]", "", name)) < 3:
            return False
        named = re.search(r"(?<![a-z0-9])" + re.escape(name.lower()) + r"(?:'s)?(?![a-z0-9])", str(provenance or "").lower())
        if not named:
            return False
        kinds = self.commodity_varieties()
        hunt_feats = (self.drive.contours.get(self.drive.role("search") or "") or {}).get("properties") or {}
        for brief in hunt_feats.values():
            alternatives = brief.get("enum") if isinstance(brief, dict) else None
            if isinstance(alternatives, list):
                for alternative in alternatives:
                    kinds.update(variety_vocables(alternative))
        personal = variety_vocables(name)
        return bool(personal) and not personal <= kinds

    @staticmethod
    def destination_selector(target):
        words = set().union(*(cite_lexs(z) for z in target.get("details") or []))
        return cite_designation(target.get("seller") or ""), tuple(sorted(words))

    @staticmethod
    def depict_destination(target):
        which = target.get("kind") or "listing"
        surplus = target.get("details")
        rear = " (" + "; ".join(surplus or []) + ")" if surplus else ""
        return f"the {which} sold by {target.get('seller')}{rear}"

    def depict(self, cite):
        return self.depict_destination(cite["target"])

    # -- index of the listings met so far
    def offering_offset(self):
        transcripts = self.stash.items
        jotting = self.offset_jotting
        if jotting and jotting[0] == len(transcripts):
            return jotting[1]
        blended, ultimate_upon = {}, {}
        datarow_stamp = self.drive.observe.datarow_stamp
        persisted = self.stash.ids

        def possesses_atlas(fields):
            return any(isinstance(vz, dict) and len(vz) >= 2 for vz in fields.values())

        for at, transcript in enumerate(transcripts):
            def ingest(fields, at=at):
                lbl = datarow_stamp(fields)
                if lbl not in persisted:
                    return
                # a short record is still kept when it carries a property map
                if len(fields) < 4 and not possesses_atlas(fields):
                    return
                blended.setdefault(lbl, {}).update(fields)
                ultimate_upon[lbl] = at
            route_hashmaps(transcript, ingest)
        self.offset_jotting = (len(transcripts), blended)
        self.offset_noticed_upon = ultimate_upon
        return blended

    def merchants_relating(self, row):
        learned = self.merchant_columns()
        discovered = set()
        for key, vlu in row.items():
            if not isinstance(vlu, str) or not vlu.strip() or len(vlu) > 60:
                continue
            minor = str(key).lower()
            merchant_selector = re.search(MERCHANT_SELECTORS, minor) or str(key) in learned
            if not merchant_selector or "id" in minor:
                continue
            if re.search(r"note|text|desc|comment|detail|info|message|section", minor):
                continue
            discovered.add(cite_designation(vlu))
        discovered.discard("")
        return discovered

    def uppercased_designations(self, rows):
        discovered = set()

        def purge(spell):
            for first in range(len(spell)):
                for ultimate in range(first + 1, min(len(spell), first + 3) + 1):
                    discovered.add(cite_designation(" ".join(spell[first:ultimate])))

        for row in rows.values():
            for clause_alt in re.split(r"(?<=[.!?;])\s+", self.comment_prose(row, dscr=False)):
                spell = []
                for vocable in re.findall(r"[A-Za-z][A-Za-z'-]*", clause_alt)[1:] + [""]:
                    if vocable[:1].isupper() and re.search(r"[a-z]", vocable):
                        spell.append(vocable)
                    else:
                        purge(spell)
                        spell = []
        return discovered

    def pursued_merchants(self):
        names = {cite_designation(flank.get("seller"))
                 for cite in self.drive.refs or []
                 for flank in (cite.get("target") or {}, cite.get("first") or {})
                 if isinstance(flank, dict)}
        return names - {""}

    def merchant_columns(self):
        try:
            names = self.pursued_merchants()
            seal = (len(self.stash.items), tuple(sorted(names)))
            jotting = self.merchant_column_jotting
            if jotting and jotting[0] == seal:
                return jotting[1]
            rows = self.offering_offset()
            lexicon = set()
            for row in rows.values():
                for vlu in row.values():
                    if isinstance(vlu, dict):
                        lexicon.update(cite_designation(xq) for xq in vlu.values() if isinstance(xq, str))
            noted = {wq for wq in self.uppercased_designations(rows) if len(wq) >= 4} - lexicon
            omit = (r"title|heading|name|label|caption|categor|shelf|department|aisle|note|text|desc|comment|detail|info|"
                    r"message|section|url|image|summary|overview")

            def designations_merchant(vlu):
                if cite_designation(vlu) in names:
                    return True
                return vlu.strip()[:1].isupper() and len(vlu.split()) <= 3 and cite_designation(vlu) in noted

            fields = set()
            for row in rows.values():
                for key, vlu in row.items():
                    minor = str(key).lower()
                    if not isinstance(vlu, str) or not 0 < len(vlu) <= 60 or "id" in minor or re.search(omit, minor):
                        continue
                    if designations_merchant(vlu):
                        fields.add(str(key))
            self.merchant_column_jotting = (seal, fields)
            return fields
        except Exception:
            return set()

    @staticmethod
    def comment_prose(row, dscr=True):
        lines, digests = [], []
        for key, vlu in row.items():
            minor = str(key).lower()
            if not isinstance(vlu, str) or len(vlu.split()) < 3 or minor.endswith("id"):
                continue
            if re.search(r"title|heading|headline|name|label|caption|categor|shelf|department|url|image", minor):
                continue
            (digests if re.search(r"desc|overview|summary|about", minor) else lines).append(vlu)
        if lines:
            return "\n".join(lines)
        return "\n".join(digests) if dscr else ""

    def morpheme_components(self, row):
        def primitive(xq):
            return isinstance(xq, (str, int, float)) and not isinstance(xq, bool)

        inverted = r"\b(?:not|no|never|without|nor)\b(?:\W+[A-Za-z0-9.:/&'-]+){1,3}"
        notes = re.sub(inverted, " ", self.comment_prose(row, dscr=False), flags=re.I)
        crowns, vlus = [], [self.drive.shelf.headline_relating(row), notes]
        assocs = [vz for vz in row.values() if isinstance(vz, dict)]
        if not assocs:
            omit = r"id$|title|shop|seller|brand|price|amount|desc|overview|note"
            assocs = [{kx: vz for kx, vz in row.items() if primitive(vz) and not re.search(omit, str(kx).lower())}]
        for fields in assocs:
            for key, vlu in fields.items():
                crowns.append(re.sub(r"([a-z])([A-Z])", r"\1 \2", str(key)).replace("_", " "))
                if primitive(vlu):
                    vlus.append(str(vlu))
        return set(cite_lexs(" ".join(crowns))), set(cite_lexs(" | ".join(vlus)))

    def personal_statistics_prose(self, row):
        fragments = [self.drive.shelf.headline_relating(row)]
        for vlu in row.values():
            if isinstance(vlu, dict):
                fragments.extend(str(xq) for xq in vlu.values() if isinstance(xq, (str, int, float)))
        return " | ".join(fragments)

    def might_exhibit(self, row, nuance, components=None):
        names, values = components or self.morpheme_components(row)
        wished_alt = {tz for tz in cite_lexs(nuance) if len(tz) >= 3 and not re.search(r"\d", tz)} - names
        if wished_alt and wished_alt.isdisjoint(values):
            return False
        figures = cite_digs(nuance)
        if not figures:
            return True
        written = cite_digs(self.personal_statistics_prose(row))
        for statistic, unit in figures:
            equal_metric = [f for f, uq in written if uq == unit]
            if equal_metric and not any(abs(f - statistic) < 1e-9 for f in equal_metric):
                return False
        return True

    @staticmethod
    def column_measurement(row, prop):
        def workable(xq):
            return isinstance(xq, (str, int, float)) and not isinstance(xq, bool) and str(xq).strip()

        assocs = [vlu for vlu in row.values() if isinstance(vlu, dict)]
        for comparator in (stringent_feat, equal_feat):
            for fields in assocs:
                for key, xq in fields.items():
                    if workable(xq) and comparator(key, prop):
                        return str(xq).strip()
        return ""

    def recall_idents(self, ids):
        self.contender_idents = set(self.contender_idents or set()).union(ids)

    def cite_alien(self):
        try:
            ids = self.contender_idents or set()
            if not ids:
                return set()
            shelves = self.drive.shelf.hub_aisles()
            if not shelves or not all(shelves):
                return set()
            houses = set().union(*shelves)
            rows = self.offering_offset()
            amidst_buggy = set(self.drive.wagon.metrics)
            aisle_relating = self.drive.observe.aisle_relating
            afield = set()
            for lbl in ids:
                if lbl in amidst_buggy:
                    continue
                label = aisle_relating(rows.get(lbl) or {})
                if label and label not in houses:
                    afield.add(lbl)
            return afield
        except Exception:
            return set()

    def contenders(self, cite):
        target = cite["target"]
        wished_alt = cite_designation(target["seller"])
        kinds = variety_vocables(target["kind"])
        rows = self.offering_offset()
        noticed_upon = self.offset_noticed_upon or {}
        headline_relating, aisle_relating = self.drive.shelf.headline_relating, self.drive.observe.aisle_relating

        def snippet(row):
            return " ".join(str(vz) for kx, vz in row.items()
                            if isinstance(vz, str) and re.search(r"desc|overview|summary", str(kx).lower()))

        graded = []
        for lbl, row in rows.items():
            if wished_alt not in self.merchants_relating(row):
                continue
            straw = set(cite_lexs(headline_relating(row) + " " + snippet(row)))
            exhibited = 0
            for nuance in target["details"]:
                lexs = set(cite_lexs(nuance))
                if lexs and lexs <= straw:
                    exhibited += 1
            perk = 2 if kinds & variety_vocables(aisle_relating(row)) else 0
            graded.append((-perk - exhibited, -noticed_upon.get(lbl, 0), lbl, exhibited))
        graded.sort()
        cite["shows"] = {lbl: exhibited for _, _, lbl, exhibited in graded}
        return [lbl for _, _, lbl, _ in graded]

    # -- working out how the store's search is called
    @staticmethod
    def disregards_vocables(seen):
        for at, (vocables_u, datarows_u) in enumerate(seen):
            if len(datarows_u) < 2:
                continue
            if any(datarows_u == datarows_g and not (vocables_u & vocables_g) for vocables_g, datarows_g in seen[at + 1:]):
                return True
        return False

    def hunt_manners(self, necessary, captions):
        observe = self.drive.observe
        cohorts = {}
        for at, page in enumerate(self.drive.page_book.values()):
            args = page.get("args") if isinstance(page.get("args"), dict) else {}
            proses = observe.unbound_prose(args)
            prose_selectors = [kx for kx, vz in args.items() if isinstance(vz, str) and vz in proses and vz.strip()]
            if len(proses) != 1 or len(prose_selectors) != 1:
                continue
            hunttext_selector = prose_selectors[0]
            remainder = {kx: vz for kx, vz in args.items()
                    if kx != hunttext_selector and (kx in necessary or not (isinstance(vz, str) and vz.strip().lower() in captions))}
            slot = json.dumps(remainder, sort_keys=True, default=str)
            group = cohorts.setdefault(slot, {"tpl": (remainder, hunttext_selector), "seen": [], "at": at})
            roots_alt = {observe.root_alt(wq) for wq in re.findall(r"[a-z0-9]+", proses[0].lower())}
            group["seen"].append((roots_alt, tuple(page.get("first") or ())))
        ranked, sightless = [], set()
        for slot, group in cohorts.items():
            seen = group["seen"]
            if self.disregards_vocables(seen):
                sightless.add(slot)
                continue
            diversity = len({tuple(sorted(words)) for words, _ in seen})
            ranked.append((-diversity, group["at"], group["tpl"]))
        ranked.sort(key=lambda item: (item[0], item[1]))
        return [tpl for _, _, tpl in ranked], sightless

    def journaled_stencil(self, necessary, captions, sightless, scrolling):
        search = self.drive.role("search")
        unbound_prose = self.drive.observe.unbound_prose
        for element_alt in reversed(self.drive.diary):
            if element_alt.get("kind") != "env":
                continue
            for operate in reversed(element_alt.get("actions") or []):
                if operate.get("name") != search:
                    continue
                args = operate.get("args")
                if not isinstance(args, dict):
                    continue
                proses = [kx for kx, vz in args.items() if isinstance(vz, str) and vz.strip() and vz in unbound_prose(args)]
                if len(proses) != 1:
                    continue
                hunttext_selector = proses[0]
                rest_alt = {kx: vz for kx, vz in args.items() if kx != hunttext_selector and not (scrolling and kx == scrolling[0])}
                kept = {kx: vz for kx, vz in rest_alt.items()
                        if kx in necessary or not (isinstance(vz, str) and vz.strip().lower() in captions)}
                if json.dumps(kept, sort_keys=True, default=str) not in sightless:
                    return rest_alt, hunttext_selector
        return None

    def hunt_stencil(self):
        contour = self.drive.contours.get(self.drive.role("search") or "") or {}
        required = contour.get("required")
        necessary = set(required or []) if isinstance(required, list) else set()
        aisle_relating = self.drive.observe.aisle_relating
        captions = {aisle_relating(rq) for rq in self.offering_offset().values()} - {""}
        manners, sightless = self.hunt_manners(necessary, captions)
        scrolling = self.drive.observe.folio_operand()
        stencil = manners[0] if manners else self.journaled_stencil(necessary, captions, sightless, scrolling)
        if stencil:
            rest_alt, key = stencil
            return {kx: vz for kx, vz in rest_alt.items()
                    if kx in necessary or not (isinstance(vz, str) and vz.strip().lower() in captions)}, key
        props = contour.get("properties") if isinstance(contour.get("properties"), dict) else {}
        token = self.drive.observe.morpheme_selector()
        proses = [kx for kx, pz in props.items()
                 if isinstance(pz, dict) and pz.get("type") == "string" and not pz.get("enum") and kx != token]
        if len(proses) == 1 and not necessary - {proses[0]}:
            return {}, proses[0]
        return None

    def recognized_vocables(self):
        def reduced_vocables(text):
            return re.findall(r"[a-z]+", text.lower())

        known = set()
        for row in self.offering_offset().values():
            for vlu in row.values():
                if isinstance(vlu, dict):
                    for key in vlu:
                        known.update(reduced_vocables(re.sub(r"([a-z])([A-Z])", r"\1 \2", str(key))))
        for req in self.needs.written + self.needs.uncovered:
            known.update(reduced_vocables(self.front_feat(str(req).partition(":")[0])))
        return known

    def nuance_vocables(self, cite):
        known = self.recognized_vocables()
        target = cite["target"]
        details = target["details"]
        merchant_vocables = set(re.findall(r"[a-z0-9]+", str(target["seller"]).lower()))
        mingled, quantities, bare_alt = [], [], []
        for nuance in details:
            bfr_entire = ""
            for entire in str(nuance).lower().split():
                succeeds_numeral = re.fullmatch(r"\d+(?:[.,]\d+)?", bfr_entire) is not None
                bfr_entire = entire
                naked = entire.strip(".,;:!?()[]{}\"'")
                # a unit word straight after a figure is left out
                for vocable in re.findall(r"[a-z0-9]+", entire):
                    possesses_cipher = re.search(r"\d", vocable)
                    if vocable in ANCHOR_VOCABLES or vocable in merchant_vocables or (succeeds_numeral and not possesses_cipher):
                        continue
                    if possesses_cipher and re.search(r"[a-z]", vocable):
                        mingled.append(vocable)
                    elif vocable.isdigit():
                        if len(vocable) >= 2:
                            quantities.append(vocable)
                    elif len(vocable) >= 3 and vocable == naked and vocable not in known and vocable.rstrip("s") not in known:
                        bare_alt.append(vocable)
        bare_alt.sort(key=len, reverse=True)
        principal = self.measurement_vocables_primary(cite, details, bare_alt)
        bare_alt = [wq for wq in principal if wq in bare_alt] + [wq for wq in bare_alt if wq not in principal]
        return list(dict.fromkeys(mingled + quantities + bare_alt))

    def measurement_vocables_primary(self, cite, details, bare_alt):
        # in a detail shaped like "<field> <value>", the value word is searched first
        column_designations = self.column_designations()
        split = [[wq for wq in re.findall(r"[a-z0-9]+", str(z).lower()) if wq not in ANCHOR_VOCABLES] for z in details]
        every_duos = cite.get("hops", 0) == 0 and bool(split) and all(len(components) >= 2 for components in split)

        def captioned(components):
            return any(" ".join(components[:s_alt]) in column_designations for s_alt in range(1, len(components)))

        return [components[-1] for components in split
                if len(components) >= 2 and components[-1] in bare_alt and (every_duos or captioned(components))]

    def column_designations(self):
        seal = len(self.stash.items)
        jotting = self.column_designations_jotting
        if jotting and jotting[0] == seal:
            return jotting[1]
        discovered = set()

        def collect_alt(fields):
            for vlu in fields.values():
                if not isinstance(vlu, dict):
                    continue
                if sum(1 for xq in vlu.values() if isinstance(xq, (str, int, float))) < 2:
                    continue
                for key in vlu:
                    separated = " ".join(re.findall(r"[a-z0-9]+", re.sub(r"([a-z])([A-Z])", r"\1 \2", str(key)).lower()))
                    if separated:
                        discovered.add(separated)

        for transcript in self.stash.items:
            route_hashmaps(transcript, collect_alt)
        self.column_designations_jotting = (seal, discovered)
        return discovered

    def scheduled_hunttexts(self, cite):
        broadness = max(2, self.drive.guide_vocables or 3)
        target = cite["target"]
        seller = " ".join(re.findall(r"[A-Za-z0-9]+", re.sub(r"'s\b", "", str(target["seller"]))))
        words = self.nuance_vocables(cite)
        kind = [wq for wq in re.findall(r"[a-z]{3,}", str(target["kind"]).lower()) if wq not in ANCHOR_VOCABLES][-1:]
        if cite.get("unindexed"):
            strategies = [" ".join((words[:1] + kind)[:broadness])] if words or kind else []
        elif words and len(seller.split()) < broadness:
            strategies = [seller + " " + words[0], seller]
        else:
            strategies = [seller]
        tested = cite["queries"]
        return [plan for plan in strategies if plan.strip() and plan.lower() not in tested]

    def ensuing_merchant_folio(self, cite):
        observe = self.drive.observe
        scrolling = observe.folio_operand()
        if not scrolling or cite["pages"] >= 3:
            return None
        wished_alt = cite_designation(cite["target"]["seller"])
        if not wished_alt:
            return None
        for page in reversed(list(self.drive.page_book.values())):
            if wished_alt not in cite_designation(" ".join(observe.unbound_prose(page["args"]))):
                continue
            succeeding = observe.ensuing_folio(page)
            if succeeding is not None:
                return {**page["args"], scrolling[0]: succeeding}
        return None

    # -- how many steps the chase may take
    def cite_space(self):
        pending = self.access_cites()
        every_vary = bool(pending) and all(rq.get("relation") == "differ" for rq in pending)
        if every_vary and not any(rq.get("chain") for rq in pending) and self.phases_disbursed >= CITE_VARY_SIGHTLESS:
            return False
        cap = self.drive.phase_top
        floor = self.phase_bottom()
        constraint = max(6, int(CITE_VARY_PHASES * cap)) if every_vary else max(8, int(CITE_PHASES * cap))
        channel = self.drive.channel
        if self.phases_disbursed >= constraint or channel.extra_alt() <= SEALING + 60 or channel.disbursed_outward() or channel.done or channel.snapped:
            return False
        if self.tarries_towards_rest(floor):
            return False
        return self.drive.maneuvers_residual() > floor or self.tardy_leap_space()

    def phase_bottom(self):
        cap = self.drive.phase_top
        stockpile = max(PRIMARY_CHECKOUT_THROUGH, min(PRIMARY_CHECKOUT_EXPECTED, cap // 4 + 1)) + 10
        return max(stockpile, self.drive.admission.checkout_expected() + 7)

    def commodity_towards(self, cite):
        if self.needs.items() and cite.get("tag"):
            return self.drive.speak.commodity_relating(f"x ({cite['tag']})")
        return None

    def tarries_towards_rest(self, floor=None):
        rows = self.needs.items()
        pending = self.access_cites()
        if not rows or not pending:
            return False
        if any(rq.get("relation") != "differ" or not rq.get("chain") for rq in pending):
            return False
        if floor is None:
            floor = self.phase_bottom()
        if self.drive.maneuvers_residual() <= floor + 4:
            return False
        try:
            holders = {self.commodity_towards(rq) for rq in pending}
            lacking = [kx for kx in range(len(rows)) if not any(y_alt["fit"] for y_alt in self.drive.shelf.dx_cluster(kx))]
        except Exception as failure:
            self.drive.diary.append({"kind": "ref_wait_error", "error": f"{type(failure).__name__}: {failure}"[:300]})
            return False
        designated_holders = holders - {None}
        rest_alt = [kx for kx in lacking if kx not in designated_holders]
        tarrying = len(rest_alt) >= (2 if None in holders else 1)
        if tarrying and self.tarry_journaled != tuple(rest_alt):
            self.tarry_journaled = tuple(rest_alt)
            self.drive.diary.append({"kind": "ref_wait", "lacking": rest_alt})
        return tarrying

    def tardy_leap_space(self):
        if self.drive.maneuvers_residual() <= self.drive.admission.checkout_expected() + 2:
            return False
        disbursed = self.phases_disbursed
        for cite in self.drive.refs or []:
            target = cite.get("target") or {}
            named = (str(target.get("seller") or "").strip() and str(target.get("kind") or "").strip()
                     and target.get("details"))
            if not named or cite.get("status") != "open" or cite.get("hops", 0) < 1:
                continue
            started = cite.get("late_from")
            if started is not None:
                if disbursed - started < 3:
                    return True
                continue
            if cite.get("queries") or not self.rest_authenticated(cite):
                continue
            cite["late_from"] = disbursed
            self.drive.diary.append({"kind": "ref_late", "prop": cite.get("prop"), "to": str(target.get("seller"))[:80]})
            return True
        return False

    def rest_authenticated(self, cite):
        rows = self.needs.items()
        if not rows:
            return True
        try:
            naked = [kx for kx in range(len(rows))
                    if not any(y_alt["verified"] and y_alt["fit"] for y_alt in self.drive.shelf.dx_cluster(kx))]
        except Exception:
            return False
        personal = self.commodity_towards(cite)
        return len(naked) <= 1 if personal is None else all(kx == personal for kx in naked)

    def leap_allowance_sound(self):
        channel = self.drive.channel
        if self.leap_triggers >= CITE_TRIGGERS or channel.extra_alt() <= SEALING + 40 or channel.disbursed_outward():
            return False
        # hop checks end once 80% of the spending cap is gone
        return channel.disburse < 0.8 * DISBURSE_TOP * channel.factor

    def cite_concord(self, cite, req):
        text = str(req)
        if text in (cite["req"], cite.get("now")):
            return True
        head, _, value = text.partition(":")
        directing = cite_prose(value) or (cite.get("relation") == "differ" and re.search(VARY_VOCABLES, value.lower()))
        if not directing or not equal_feat(self.front_feat(head), cite["prop"]):
            return False
        tagnames = re.findall(r"\(([^)]*)\)", head)
        if not tagnames or not cite["tag"]:
            return True
        return bool(variety_vocables(tagnames[-1]) & variety_vocables(cite["tag"]))

    def cite_ongoing(self, cite):
        def onto_feat(req):
            return equal_feat(self.front_feat(str(req).partition(":")[0]), cite["prop"])
        if any(self.cite_concord(cite, req) or onto_feat(req) for req in self.needs.feats_residual()):
            return True
        return not any(onto_feat(req) for req in self.needs.let_go)

    # -- a single resolver turn
    def cite_phase(self):
        channel = self.drive.channel
        if not self.drive.refs or channel.done or channel.snapped:
            return False
        try:
            return self.decider_stint()
        except Exception as failure:
            self.drive.diary.append({"kind": "ref_error", "error": f"{type(failure).__name__}: {failure}"[:300]})
            for cite in self.drive.refs:
                if cite["status"] == "open":
                    cite["status"], cite["why"] = "stuck", "the code's reference resolver broke down"
            return False

    def ongoing_access(self):
        living = []
        for cite in self.drive.refs:
            if cite["status"] != "open":
                continue
            if self.cite_ongoing(cite):
                living.append(cite)
                continue
            cite["status"] = "dropped"
            self.drive.diary.append({"kind": "ref_dropped", "req": cite["req"][:200]})
        return living

    @staticmethod
    def browse_waitlist(clusters, maximal):
        waitlist = []
        while len(waitlist) < maximal and any(clusters):
            for cluster in clusters:
                while cluster and cluster[0] in waitlist:
                    del cluster[0]
                if cluster and len(waitlist) < maximal:
                    waitlist.append(cluster.pop(0))
        return waitlist

    def browse_clusters(self, pending, maximal):
        previously = self.stash.read
        clusters = []
        for cite in pending:
            cluster = [ii for ii in self.contenders(cite) if ii not in previously and ii not in cite["judged"]]
            shows = cite.get("shows") or {}
            # open them once a detail search ran, when only a few are left, or when the top title shows a detail
            primed = (bool(cite["queries"]) or not self.scheduled_hunttexts(cite) or len(cluster) <= maximal
                     or any(shows.get(ii) for ii in cluster[:1]))
            clusters.append(cluster if primed else [])
        return clusters

    def hunt_ensuing(self, deprived, pending, stencil, performed):
        order = deprived or sorted(pending, key=lambda rq: (rq["hops"], len(rq["queries"])))
        for cite in order:
            strategies = self.scheduled_hunttexts(cite)
            if strategies:
                query = strategies[0]
                cite["queries"].append(query.lower())
                if self.drive.walk.drove_hunttext(query):
                    return True
                return self.hunt_destination(cite, {**stencil[0], stencil[1]: query}) or performed
            additional = self.ensuing_merchant_folio(cite)
            if additional:
                cite["pages"] += 1
                return self.hunt_destination(cite, additional) or performed
        for cite in pending:
            unrated = [ii for ii in self.contenders(cite) if ii not in cite["judged"]]
            if unrated or self.scheduled_hunttexts(cite) or self.ensuing_merchant_folio(cite):
                continue
            self.provide_aloft(cite, "the store's search never turned up the listing it points at")
            performed = True
        return performed

    def decider_stint(self):
        pending = self.ongoing_access()
        if not pending:
            return False
        beat0 = self.drive.channel.beat
        performed = False
        for cite in sorted(pending, key=lambda rq: rq["hops"]):
            performed = self.referee_browse(cite) or performed
        if self.drive.channel.beat > beat0:
            return True
        pending = [rq for rq in pending if rq["status"] == "open"]
        if not pending or not self.cite_space():
            return performed
        lot = self.drive.shelf.lot_utility()
        maximal = lot[2] if lot else 2
        clusters = self.browse_clusters(pending, maximal)
        stencil = self.hunt_stencil()
        empty = [rq for rq, cluster in zip(pending, clusters) if not cluster and self.scheduled_hunttexts(rq)] if stencil else []
        waitlist = self.browse_waitlist(clusters, maximal)
        deprived = empty if waitlist and len(waitlist) < maximal else []
        if waitlist and not deprived:
            return self.browse_contenders(waitlist, pending) or performed
        if stencil:
            return self.hunt_ensuing(deprived, pending, stencil, performed)
        return performed

    def browse_contenders(self, waitlist, pending):
        drive = self.drive
        lot = drive.shelf.lot_utility() if len(waitlist) >= 2 else None
        beat0 = drive.channel.beat
        crowns = ", ".join(dict.fromkeys(rq["head"] for rq in pending))
        utter = f"The request takes {crowns} from another listing: opening the records that could be that listing."
        rejoinders = drive.referee.access_browses(waitlist, utter, lot)
        dropped = False
        if lot and rejoinders and not drive.batch_proven:
            if drive.shelf.lot_sound(drive.referee.ultimate_browses or [], waitlist):
                drive.batch_proven = True
            else:
                drive.batch_spec, drive.lot_brief_recognized, dropped = None, True, True
                drive.diary.append({"kind": "batch_dropped", "tool": lot[0]})
        if rejoinders:
            drive.speak.subsequent_rejoinders(rejoinders)
        self.phases_disbursed += drive.channel.beat - beat0
        self.recall_idents(waitlist)
        drive.diary.append({"kind": "ref_read", "ids": waitlist})
        if not dropped:
            unbrowsed = [ii for ii in waitlist if ii not in self.stash.read]
            for cite in pending:
                cite["judged"].update(unbrowsed)
        return drive.channel.beat > beat0

    def hunt_destination(self, cite, args):
        drive = self.drive
        beat0 = drive.channel.beat
        utter = f"The request takes {cite['head']} from another listing ({self.depict(cite)}): looking that listing up."
        obtained = drive.walk.route_trigger(args, utter)
        self.phases_disbursed += drive.channel.beat - beat0
        rows = []
        if obtained:
            datarow_stamp, persisted = drive.observe.datarow_stamp, self.stash.ids

            def amass(fields):
                if datarow_stamp(fields) in persisted and len(fields) >= 4:
                    rows.append(fields)
            route_hashmaps(self.stash.items[-1], amass, ceiling=3)
        wished_alt = cite_designation(cite["target"]["seller"])
        query = " ".join(drive.observe.unbound_prose(args))
        if rows and not any(wished_alt in self.merchants_relating(rq) for rq in rows) and cite_designation(query) == wished_alt:
            cite["unindexed"] = True
        drive.diary.append({"kind": "ref_search", "prop": cite["prop"], "query": query[:120], "rows": len(rows)})
        return drive.channel.beat > beat0

    # -- weighing the records that were read
    def referee_browse(self, cite):
        target = cite["target"]
        read = self.stash.read
        optns = [ii for ii in self.contenders(cite) if ii in read and ii not in cite["judged"]]
        if not optns:
            return False
        rows = self.offering_offset()
        through_merchant = [ii for ii in self.contenders(cite) if ii in read]
        components = {ii: self.morpheme_components(rows.get(ii) or {}) for ii in set(optns) | set(through_merchant)}
        loose = cite.setdefault("loose", set())
        details = target["details"]

        def shows(lbl, nuance):
            return self.might_exhibit(rows.get(lbl) or {}, nuance, components[lbl])

        if len(through_merchant) >= 4:
            # a worded detail missing from all of 4+ records is beyond what code can check
            loose.update(z for z in details if not any(shows(ii, z) for ii in through_merchant))

        def vocables_sound(lbl):
            return all(shows(lbl, z) for z in details if z not in loose)

        def encompassed(lbl):
            names, values = components[lbl]
            cluster = names | values
            return sum(1 for z in details if set(cite_lexs(z)) and set(cite_lexs(z)) <= cluster)

        brimming = {lbl: encompassed(lbl) for lbl in optns}
        ranked = sorted(optns, key=lambda ii: (not vocables_sound(ii), -brimming[ii], optns.index(ii)))
        if not vocables_sound(ranked[0]) and self.additional_unto_browse(cite):
            return False
        suitable = [ii for ii in ranked if vocables_sound(ii)]
        select = suitable[:4] or ranked[:4]
        precise = [ii for ii in select if details and brimming[ii] == len(details)]
        solicited = precise if len(precise) == 1 else (suitable[:2] or select[:1])
        verdict = self.conclude_through_comments(cite, select, self.comments_unbrowsed(cite, solicited))
        if verdict is not None:
            return verdict
        if len(precise) == 1 and self.bare_transcript(cite, precise[0]):
            cite["judged"].update(select)
            self.recall_idents(select)
            self.decide(cite, precise[0], self.column_measurement(rows.get(precise[0]) or {}, cite["prop"]), [])
            return True
        if not self.leap_allowance_sound():
            cite["judged"].update(select)
            self.provide_aloft(cite, "the hop checks are used up")
            return True
        self.implement_leap(cite, select, self.solicit_leap(cite, select, ""))
        if cite["status"] == "open" and cite["target"] is target:
            cite["judged"].update(ii for ii in optns if not vocables_sound(ii))
        return True

    def additional_unto_browse(self, cite):
        if not self.cite_space() and not self.tarries_towards_rest():
            return False
        read = self.stash.read
        if any(ii not in read and ii not in cite["judged"] for ii in self.contenders(cite)):
            return True
        return bool(self.scheduled_hunttexts(cite) or self.ensuing_merchant_folio(cite))

    def conclude_through_comments(self, cite, select, missing):
        if not missing:
            return None
        if self.cite_space() and self.browse_comments(cite, missing):
            return True
        if self.tarries_towards_rest():
            return False
        if self.cite_space() or cite.get("relation") != "differ":
            return None
        # notes still unread may send the reader elsewhere, so the exclusion cannot be settled
        cite["judged"].update(select)
        self.provide_aloft(cite, "the steps left were too few to read the notes of the listing it points at")
        return True

    def column_morphemes(self, row, names):
        fragments = [self.drive.shelf.headline_relating(row), self.drive.observe.aisle_relating(row)]
        for vlu in row.values():
            if isinstance(vlu, dict):
                fragments.extend(str(xq) for xq in vlu.values() if isinstance(xq, (str, int, float)))
        return names | set(cite_lexs(" | ".join(fragments)))

    def bare_transcript(self, cite, lbl):
        rows = self.offering_offset()
        row = rows.get(lbl) or {}
        value = self.column_measurement(row, cite["prop"])
        notes = self.comment_prose(row, dscr=False)
        if not value or self.designations_characteristic(cite["prop"], notes):
            return False
        if set(cite_lexs(value)) & set(cite_lexs(notes)):
            return False
        personal = self.merchants_relating(row)
        otherwhere = {s for further in rows.values() for s in self.merchants_relating(further)} - personal
        names, _ = self.morpheme_components(row)
        fields = self.column_morphemes(row, names)
        for clause_alt in re.split(r"(?<=[.!?;])\s+", notes):
            for vocable in re.findall(r"[A-Za-z][A-Za-z'-]*", clause_alt)[1:]:
                minor = cite_designation(vocable)
                errant = vocable[:1].isupper() and len(minor) >= 4 and minor not in personal and minor not in fields
                if minor in otherwhere or errant:
                    return False
        return True

    def comments_browser(self):
        shelf = self.drive.shelf
        learned = shelf.comments_utility()
        if learned:
            return learned
        details = self.drive.role("details")
        if (shelf.comment_selectors or {}).get(details or ""):
            return None
        utility_selectors = shelf.utility_selectors or {}
        prose_selectors = shelf.prose_selectors or {}
        encompassed = set(utility_selectors.get(details or "") or set()) | set(shelf.datarow_selectors or set())
        tops = shelf.utility_tops or {}
        rest_alt = [tz for tz in shelf.fetch_utilities() if tz != details]
        surplus = [tz for tz in rest_alt if (set(utility_selectors.get(tz) or set()) - encompassed) & set(prose_selectors.get(tz) or set())]
        if surplus:
            return max(surplus, key=lambda tz: ((tops.get(tz) or {}).get("calls", 0), tz))
        reserves = (self.drive.role("view"), shelf.supply_utility())
        return next((tz for tz in reserves if tz in rest_alt and not (tops.get(tz) or {}).get("id", 0)), None)

    def comments_unbrowsed(self, cite, ids):
        tool = self.comments_browser()
        if not tool or not ids:
            return []
        done = set((self.drive.shelf.browse_alongside or {}).get(tool) or set())
        tested = cite.setdefault("noted", set())
        return [ii for ii in dict.fromkeys(ids) if ii not in done and ii not in tested][:2]

    def browse_comments(self, cite, need):
        tool = self.comments_browser()
        if not tool or not need:
            return False
        drive = self.drive
        cite.setdefault("noted", set()).update(need)
        beat0 = drive.channel.beat
        utter = (f"The request takes {cite['head']} from another listing: reading the notes ({tool}) of the record that "
               "could be it, since a note can pass the reader on to yet another listing.")
        rejoinders = drive.referee.access_browses(need, utter, None, tool)
        if rejoinders:
            drive.speak.subsequent_rejoinders(rejoinders)
        self.phases_disbursed += drive.channel.beat - beat0
        self.recall_idents(need)
        drive.diary.append({"kind": "ref_notes", "ids": need, "tool": tool})
        return drive.channel.beat > beat0

    # -- differ references checked against listings
    def cite_arrows(self):
        directed = set()
        for cite in self.drive.refs or []:
            if cite.get("relation") != "differ" or cite.get("status") not in ("open", "stuck", "resolved"):
                continue
            if self.cite_ongoing(cite):
                directed.update(ii for ii in cite.get("chain") or [] if self.equal_commodity(cite, ii))
        return directed

    def equal_commodity(self, cite, lbl):
        if not self.needs.items():
            return True
        shelf = self.drive.referee.aisle_recognized(lbl)
        if not shelf:
            return True
        commodity_relating = self.drive.speak.commodity_relating
        kx = commodity_relating(f"x ({shelf})")
        if kx is None or not cite.get("tag"):
            return True
        return kx == commodity_relating(f"x ({cite['tag']})")

    def cite_arrow_comment(self, entity):
        for cite in self.drive.refs or []:
            if cite.get("relation") != "differ" or entity not in (cite.get("chain") or []):
                continue
            if not self.cite_ongoing(cite) or not self.equal_commodity(cite, entity):
                continue
            known = f" ({cite['value']})" if cite.get("status") == "resolved" and cite.get("value") else ""
            prop = cite["prop"]
            return (f"{entity} is the listing the shopper's reference for {cite['head']} points at, or one its notes lead "
                    f"through, so its {prop} is exactly the value{known} that the shopper's {prop} has to differ from: "
                    "it cannot satisfy that requirement")
        return ""

    def cite_precluded(self, fields, kx):
        several = self.needs.items()
        for cite in self.drive.refs or []:
            if cite.get("relation") != "differ" or cite.get("status") != "resolved" or not self.cite_ongoing(cite):
                continue
            if several and cite.get("tag") and self.drive.speak.commodity_relating(f"x ({cite['tag']})") not in (kx, None):
                continue
            forbidden = {" ".join(cite_lexs(vz)) for vz in [cite.get("value")] + list(cite.get("goes_with") or [])
                      if str(vz or "").strip()}
            if any(equal_feat(key, cite["prop"]) and " ".join(cite_lexs(vlu)) in forbidden
                   for key, vlu in (fields or {}).items()):
                return True
        return False

    def divert_amidst(self, clause_alt, prop, personal, otherwhere, fields):
        words = list(re.finditer(r"[A-Za-z][A-Za-z'-]*", clause_alt))
        # the kind word in phrasings like 'the lamp by Name' is blanked out
        emptied = re.sub(r"[A-Za-z0-9]+(?=\s+by\s+[A-Z])", lambda hit: " " * len(hit.group(0)), clause_alt)
        first = next((mz.start() for mz in words if feat_designated(prop, emptied[:mz.end()])), None)
        if first is None:
            return False
        refusals = ("not", "nor", "never", "no", "neither")
        for at, mz in enumerate(words[1:], 1):
            if mz.start() <= first:
                continue
            vocable = mz.group(0)
            minor = cite_designation(vocable)
            if not minor:
                continue
            alien = minor in otherwhere or (vocable[:1].isupper() and len(minor) >= 4 and minor not in personal
                                           and not set(cite_lexs(vocable)) & fields and minor not in fields)
            refused_alt = any(wq.group(0).lower() in refusals for wq in words[max(0, at - 5): at])
            if alien and not refused_alt:
                return True
        return False

    def personal_diverts(self, entity):
        try:
            rows = self.offering_offset()
            row = rows.get(entity) or {}
            notes = self.comment_prose(row, dscr=False)
            if not notes:
                return []
            personal = self.merchants_relating(row)
            otherwhere = {s for further in rows.values() for s in self.merchants_relating(further)} - personal
            names, _ = self.morpheme_components(row)
            fields = self.column_morphemes(row, names)
            several = self.needs.items()
            aisle_relating, commodity_relating = self.drive.observe.aisle_relating, self.drive.speak.commodity_relating
            kx = commodity_relating(f"x ({aisle_relating(row)})") if several and aisle_relating(row) else None
            clauses = re.split(r"(?<=[.!?;])\s+", notes)
            discovered = []
            for req in self.needs.feats_residual():
                head = str(req).partition(":")[0]
                if several and kx is not None and commodity_relating(head) not in (kx, None):
                    continue
                prop = self.front_feat(head).strip()
                if not prop:
                    continue
                hit = next((s for s in clauses if self.divert_amidst(s, prop, personal, otherwhere, fields)), None)
                if hit is not None:
                    discovered.append((str(req), prop, hit.strip()))
            return discovered
        except Exception:
            return []

    def comments_observation(self, trigger_designation, trigger_operands):
        drive = self.drive
        still = (drive.role("search"), drive.role("cart"), drive.role("cart_view"), drive.role("message"))
        if trigger_designation in still or trigger_designation in drive.terminators():
            return ""
        edition = str(self.needs.edition)
        observations_alt = []
        for lbl in sorted(drive.shelf.solicited_idents(trigger_operands)):
            key = f"{lbl}|{edition}"
            maintained = lbl in drive.wagon.metrics
            if not maintained and key not in drive.passed:
                continue
            if key in self.divert_notified:
                continue
            discovered = self.personal_diverts(lbl)
            if not discovered:
                continue
            self.divert_notified.add(key)
            drive.passed.discard(key)
            drive.diary.append({"kind": "own_redirect", "id": lbl, "props": [pz for _, pz, _ in discovered]})
            which = "; ".join(f"{pz} (\"{s[:160]}\")" for _, pz, s in discovered)
            cited = "; ".join(f"'{rq}'" for rq, _, _ in discovered)
            them = "them" if len(discovered) > 1 else "it"
            rear = (". If the notes do send it there, remove it from the cart and add a different verified fit."
                    if maintained else ".")
            observations_alt.append(f"The notes of {lbl} may send {which} to another listing: in that case its own field does not "
                           f"count for {cited} (the grader takes that other listing's value), so it satisfies {them} only "
                           "if that listing's record shows it; the item is checked again before any order" + rear)
        return " ".join(observations_alt)

    # -- lines in which the shopper points at another listing
    def directing_locution(self, line):
        text = str(line or "")
        if not cite_prose(text):
            return None

        def credible(name):
            return self.cite_merchant_sound(name, text)

        dissected = cite_dissect(text, credible)
        if not dissected or not credible(dissected["seller"]):
            return None
        hit = re.search(r"(?<![A-Za-z'])" + re.escape(dissected["seller"]) + r"(?:'s)?(?![A-Za-z])", text)
        if not hit:
            return None
        preceding = text[:hit.start()]
        pauses = [preceding.rfind(cc) for cc in ",;!?"] + [mz.start() for mz in re.finditer(r"[.](?:\s|$)", preceding)]
        snip = max(pauses)
        proviso = preceding[snip + 1:].lower()
        marks = [mz.start() for design in (VARY_VOCABLES, CITE_VOCABLES, DIRECT_VOCABLES) for mz in re.finditer(design, proviso)]
        if not marks:
            return None
        start = snip + 1 + min(marks)
        afterward = text[hit.end():]
        parked = re.search(r"\(\s*or\b[^)]*\)", afterward)
        halt_alt = re.search(r"[.!?;](?:\s|$)", afterward)
        if parked and (not halt_alt or parked.start() < halt_alt.start()):
            attain = parked.end()
        elif halt_alt:
            attain = halt_alt.start()
        else:
            attain = len(afterward)
        end = hit.end() + attain
        phrase = text[start:end].strip(" ,;")
        if not phrase:
            return None
        return phrase, start, end, dissected

    def inquiry_vocables(self, qstr, exterior):
        if not qstr:
            return []
        logs = self.needs.items() or [{"product": str(self.needs.browsing.get("product") or "")}]
        kinds = {wq for element_alt in logs for wq in re.findall(r"[a-z]+", str(element_alt["product"]).lower())}
        tokens = re.findall(r"[a-z0-9]+", str(qstr).lower())

        def workable(wq):
            return not (wq in ANCHOR_VOCABLES or len(wq) < 3 or wq in kinds or wq.rstrip("s") in kinds)

        drives = []
        for size in (3, 2, 1):
            for at in range(len(tokens) - size + 1):
                shard = tokens[at: at + size]
                if not all(workable(wq) for wq in shard):
                    continue
                phrase = " ".join(shard)
                if feat_designated(phrase, exterior):
                    drives.append(phrase)
        return drives

    def cite_patch(self, answer, lines, qstr):
        veiled = []
        for line in lines:
            obtained = self.directing_locution(line)
            if obtained:
                phrase, start, end, dissected = obtained
                shard = DirectingOrderline(line, phrase, start, end, dissected)
                veiled.append(shard.exterior)
                self.situate_directing_necessity(answer, shard, qstr)
            else:
                veiled.append(line)
        return veiled

    def necessity_designation(self, req):
        return self.front_feat(str(req).partition(":")[0]).strip()

    def orderline_characteristic(self, shard, needs, refs, qstr):
        name = self.necessity_designation
        exterior = shard.exterior
        solicited = [name(a) for a in str(self.drive.speak.soliciting_caption or "").split(";") if name(a)]

        def pointvals(need):
            rear = need.partition(":")[2]
            return cite_prose(rear) or re.search(VARY_VOCABLES, rear.lower())

        directing = [name(n) for n in needs if pointvals(n)]
        strata = [[str(rq.get("property") or "") for rq in refs], directing, [name(n) for n in needs], solicited,
                 sorted(self.column_designations()) + [name(rq) for rq in self.needs.written], self.inquiry_vocables(qstr, exterior)]

        def rank(pz):
            return feat_designated(pz, exterior, precise=True), len(re.findall(r"[a-z0-9]+", pz.lower())), len(pz)

        for stratum in strata:
            named = [pz for pz in stratum if pz and feat_designated(pz, exterior)]
            if named:
                return max(named, key=rank)
        failsafe = strata[0] + strata[1] + (solicited if len(solicited) == 1 else [])
        return next((pz for pz in failsafe if pz and not shard.interior_solely(pz)), "")

    def orderline_badge(self, shard, needs, prop, qstr, several):
        commodity_relating = self.drive.speak.commodity_relating
        kx = commodity_relating(shard.exterior)
        if kx is None:
            tagnames = [tz for n in needs if equal_feat(self.necessity_designation(n), prop)
                    for tz in re.findall(r"\(([^)]*)\)", n.partition(":")[0])]
            if tagnames:
                kx = commodity_relating(f"x ({tagnames[-1]})")
        if kx is None and qstr:
            kx = commodity_relating(qstr)
        return "" if kx is None else several[kx]["product"]

    def preserved_directing(self, minee, relation, seller):
        for need in minee:
            value = need.partition(":")[2]
            directing = cite_prose(value) or re.search(VARY_VOCABLES, value.lower())
            if directing and cite_association(value)[0] == relation and seller in cite_designation(value):
                return need
        return None

    def situate_directing_necessity(self, answer, shard, qstr):
        def itemized(key, admit):
            crude = answer.get(key)
            return [xq for xq in crude if admit(xq)] if isinstance(crude, list) else []

        needs = itemized("needs", lambda n: isinstance(n, str) and ":" in n)
        refs = itemized("references", lambda rq: isinstance(rq, dict))
        name = self.necessity_designation
        prop = self.orderline_characteristic(shard, needs, refs, qstr)
        if not prop:
            self.drive.diary.append({"kind": "ref_line_unplaced", "line": shard.text[:200]})
            return
        several = self.needs.items()
        tag = self.orderline_badge(shard, needs, prop, qstr, several) if several else ""

        def equal_article(req):
            tagnames = re.findall(r"\(([^)]*)\)", str(req).partition(":")[0])
            return not several or not tag or not tagnames or bool(variety_vocables(tagnames[-1]) & variety_vocables(tag))

        minee = [n for n in needs if equal_feat(name(n), prop) and equal_article(n)]
        relation, tolerance, unit = cite_association(shard.phrase)
        seller = cite_designation(shard.dissected["seller"])
        keep = self.preserved_directing(minee, relation, seller)
        if minee:
            head = minee[0].partition(":")[0].strip()
        else:
            head = f"{prop} ({tag})" if tag else prop
        if several and tag and not re.search(r"\(([^)]*)\)", head):
            head = f"{head} ({tag})"
        need = keep or f"{head}: {shard.phrase}"
        details = [n for n in needs if n not in minee and shard.depicts_further(n, name)]
        locus = needs.index(minee[0]) if minee else len(needs)
        remainder = [n for n in needs if n not in minee and n not in details]
        answer["needs"] = remainder[:locus] + [need] + remainder[locus:]
        self.shorten_response(answer, shard, prop, need, details, equal_article)
        self.blend_element(refs, prop, shard, relation, tolerance, unit, seller)
        answer["references"] = refs
        answer["ref_needs"] = list(answer.get("ref_needs") or []) + [need]
        self.drive.diary.append({"kind": "ref_line", "prop": prop, "tag": tag, "relation": relation, "need": need[:200],
                             "kept": bool(keep), "details_out": [z[:80] for z in details]})

    def shorten_response(self, answer, shard, prop, need, details, equal_article):
        name = self.necessity_designation
        actual = answer.get("requirements_now")
        if isinstance(actual, list):
            written = self.needs.written

            def replaced(req):
                if req in written:
                    return False
                if req in details or shard.depicts_further(req, name):
                    return True
                return equal_feat(name(req), prop) and equal_article(req)

            kept = [req for req in actual if isinstance(req, str) and not replaced(req)]
            if need not in kept:
                kept.append(need)
            answer["requirements_now"] = kept
        dropped = answer.get("dropped")
        if isinstance(dropped, list):
            answer["dropped"] = [z for z in dropped if not (isinstance(z, str) and (
                shard.interior_solely(name(z)) or shard.depicts_further(z, name)))]
        access_selection = answer.get("no_preference")
        if isinstance(access_selection, list):
            answer["no_preference"] = [pz for pz in access_selection if not (isinstance(pz, str) and (
                shard.interior_solely(name(pz)) or equal_feat(name(pz), prop)))]

    def blend_element(self, refs, prop, shard, relation, tolerance, unit, seller):
        dissected = shard.dissected
        element_alt = next((rq for rq in refs if equal_feat(str(rq.get("property") or ""), prop)), None)
        if element_alt is None:
            element_alt = {"property": prop}
            refs.append(element_alt)
        equal_merchant = self.cite_merchant_sound(element_alt.get("seller"), shard.text) and cite_designation(element_alt.get("seller")) == seller
        if not equal_merchant:
            element_alt.update(seller=dissected["seller"], kind=dissected["kind"], details=list(dissected["details"]))
        elif not element_alt.get("details"):
            element_alt["details"] = list(dissected["details"])
        element_alt.setdefault("kind", dissected["kind"])
        if relation != "same" or not element_alt.get("relation"):
            element_alt.update(relation=relation, tolerance=tolerance, unit=unit)

    @staticmethod
    def designations_characteristic(prop, text):
        words = [wq for wq in feat_vocables(prop) if len(wq) >= 3]
        if not words:
            return True
        tokens = set(re.findall(r"[a-z]+", str(text).lower()))

        def similar(tkn, vocable):
            return tkn == vocable or (len(vocable) >= 4 and len(tkn) >= 4 and tkn[:4] == vocable[:4])

        return any(similar(tkn, vocable) for tkn in tokens for vocable in words)

    # -- the hop call
    def transcript_tile(self, lbl):
        row = self.offering_offset().get(lbl) or {}
        facts = {}
        for vlu in row.values():
            if not isinstance(vlu, dict):
                continue
            for key, xq in list(vlu.items())[:40]:
                if isinstance(xq, (str, int, float)) and not isinstance(xq, bool):
                    facts[str(key)] = str(xq)[:90]
        learned = self.merchant_columns()
        merchants = set()
        for key, vlu in row.items():
            minor = str(key).lower()
            if not isinstance(vlu, str) or not vlu.strip() or len(vlu) > 60 or "id" in minor:
                continue
            merchant_selector = re.search(MERCHANT_SELECTORS, minor) or str(key) in learned
            if merchant_selector and not re.search(r"note|text|desc|comment|detail|info|message|section", minor):
                merchants.add(str(vlu))
        tile = {"id": lbl, "sold by": " / ".join(sorted(merchants)), "category": self.drive.observe.aisle_relating(row),
                "title": self.drive.shelf.headline_relating(row)[:160], "facts": facts, "notes": self.comment_prose(row)[:900]}
        return qua_prose(tile, 5000)

    def solicit_leap(self, cite, select, reason):
        # one checker call over the candidate records; None when the hop budget is spent
        if not self.leap_allowance_sound():
            return None
        self.leap_triggers += 1
        populated = CITE_LEAP.replace("PROPERTY", cite["prop"]).replace("DESCRIPTION", self.depict(cite))
        tiles = "\n".join(self.transcript_tile(lbl) for lbl in select)
        text = populated + "\n\nRECORDS:\n" + tiles
        if reason:
            text += "\n\nYOUR LAST ANSWER COULD NOT BE USED: " + reason + " Answer once more, working from the records."
        reply = self.drive.channel.canvass(text, 3000, VALIDATOR[:1])
        self.drive.diary.append({"kind": "ref_hop", "prop": cite["prop"], "ids": select, "answer": reply})
        return reply

    def implement_leap(self, cite, select, answer, afresh=True):
        if not isinstance(answer, dict):
            cite["fails"] += 1
            if cite["fails"] >= 2:
                cite["judged"].update(select)
                self.provide_aloft(cite, "the hop check returned nothing usable")
            return
        named = answer.get("target")
        primitive = isinstance(named, (str, int)) and not isinstance(named, bool)
        target = str(named).strip() if primitive else ""
        holes = answer.get("missing")
        missing = [str(g) for g in holes or [] if str(g).strip()] if isinstance(holes, list) else []
        arrived = target in select
        if missing or not arrived:
            cite["judged"].update(select)
            self.recall_idents(select)
            if arrived:
                self.drive.diary.append({"kind": "ref_partial", "prop": cite["prop"], "target": target, "missing": missing[:4]})
            return
        locale, problem, advance, value, pairs = self.validate_leap(cite, target, answer, select)
        if problem and afresh and self.leap_allowance_sound():
            reattempt = self.solicit_leap(cite, select, problem)
            self.implement_leap(cite, select, reattempt, afresh=False)
            return
        self.recall_idents(select)
        if problem:
            self.drive.diary.append({"kind": "ref_invalid", "prop": cite["prop"], "target": target, "problem": problem[:300]})
            if locale == "target":
                cite["judged"].add(target)
                return
            cite["judged"].update(select)
            cite["chain"].append(target)
            self.provide_aloft(cite, problem)
            return
        cite["judged"].update(select)
        if target in cite["chain"]:
            self.provide_aloft(cite, "its notes point back to a listing that was already followed")
            return
        if advance:
            self.trail_alt(cite, target, advance)
        else:
            self.decide(cite, target, value, pairs)

    @staticmethod
    def tidy_textstrings(crude):
        def workable(item):
            return isinstance(item, (str, int, float)) and not isinstance(item, bool) and str(item).strip()
        return [str(item).strip() for item in (crude if isinstance(crude, list) else []) if workable(item)]

    def validate_advance_unto(self, cite, target, advance, row, notes, comment_morphemes):
        def spurn(problem):
            return "answer", problem, None, "", []

        seller = str(advance["seller"]).strip()
        design = r"(?<![a-z0-9])" + re.escape(seller.lower()) + r"(?:'s)?(?![a-z0-9])"
        if not re.search(design, notes.lower()):
            return spurn(f"{target}'s notes never mention the go_to seller '{seller}'.")
        clauses = re.split(r"(?<=[.!?])\s+", notes)
        onto_subject = any(re.search(design, line.lower()) and self.designations_characteristic(cite["prop"], line) for line in clauses)
        if not onto_subject:
            return spurn(f"the note that names {seller} concerns a different property, not {cite['prop']}.")

        def supported(nuance):
            lexs = set(cite_lexs(nuance))
            return bool(lexs) and 2 * len(lexs & comment_morphemes) >= len(lexs)
        details = [z for z in self.tidy_textstrings(advance.get("details")) if supported(z)]
        succeeding = {"seller": seller, "kind": str(advance.get("kind") or "").strip()[:80], "details": details[:6]}
        pointvals_hub = cite_designation(seller) in self.merchants_relating(row)
        if pointvals_hub and self.destination_selector(succeeding) == self.destination_selector(cite["target"]):
            return spurn(f"the go_to points at {target} itself.")
        return "", "", succeeding, "", []

    def nuances_exhibited(self, cite, lbl):
        names, values = self.morpheme_components(self.offering_offset().get(lbl) or {})
        displayed = names | values
        count = 0
        for nuance in cite["target"]["details"]:
            lexs = set(cite_lexs(nuance))
            if lexs and lexs <= displayed:
                count += 1
        return count

    def validate_leap(self, cite, target, answer, select=None):
        def incorrect_offering(problem):
            return "target", problem, None, "", []

        row = self.offering_offset().get(target) or {}
        names, values = self.morpheme_components(row)
        for nuance in cite["target"]["details"]:
            if nuance in (cite.get("loose") or set()):
                continue
            if not self.might_exhibit(row, nuance, (names, values)):
                return incorrect_offering(f"{target} does not show '{nuance}', so it is not the listing that was described.")

        def exhibited(lbl):
            return self.nuances_exhibited(cite, lbl)
        best = max(select or [target], key=exhibited)
        if exhibited(best) > exhibited(target):
            return incorrect_offering(f"{best} shows more of the named details than {target} does.")
        notes = self.comment_prose(row)
        comment_morphemes = set(cite_lexs(notes))
        advance = answer.get("go_to")
        if isinstance(advance, dict) and str(advance.get("seller") or "").strip():
            return self.validate_advance_unto(cite, target, advance, row, notes, comment_morphemes)
        value = str(answer.get("value") or "").strip()
        measurement_morphemes = set(cite_lexs(value))
        if not (value and measurement_morphemes and measurement_morphemes <= values | names):
            return "answer", f"{target}'s record does not contain the value '{value}'.", None, "", []

        def ally(text):
            lexs = set(cite_lexs(text))
            return bool(lexs) and lexs <= comment_morphemes and lexs != measurement_morphemes
        pairs = [text for text in dict.fromkeys(self.tidy_textstrings(answer.get("goes_with"))) if ally(text)]
        return "", "", None, value, pairs[:3]

    # -- what happens once a hop is judged
    def trail_alt(self, cite, target, advance):
        cite["chain"].append(target)
        key = self.destination_selector(advance)
        if key in cite["seen"]:
            self.provide_aloft(cite, "its notes point back to a listing that was already followed")
            return
        if cite["hops"] + 1 >= CITE_LEAPS:
            self.provide_aloft(cite, f"over {CITE_LEAPS} listings in a row pass the reader on to yet another one")
            return
        cite["seen"].add(key)
        cite["hops"] += 1
        cite.update(target=advance, judged=set(), queries=[], pages=0, tries=0, fails=0, unindexed=False, loose=set())
        self.drive.diary.append({"kind": "ref_go", "prop": cite["prop"], "from": target, "to": advance})
        self.notify_operator(f"{target} is the listing named for {cite['head']}; its notes send the reader on to "
                        f"{self.depict(cite)}, and the code is following that trail.")

    @staticmethod
    def decided_measurement(cite):
        value = str(cite["value"])
        pairs = list(cite.get("goes_with") or [])
        relation = cite["relation"]
        if relation == "within" and be_numeral(cite.get("tol")):
            hit = re.search(r"(-?\d+(?:\.\d+)?)\s*([a-z%]*)", value.lower().replace(",", ""))
            if hit:
                center, scatter = float(hit.group(1)), float(cite["tol"])
                unit = hit.group(2) or str(cite.get("unit") or "")
                minor, tall = round(center - scatter, 6), round(center + scatter, 6)
                return f"{minor:g} to {tall:g} {unit}".strip() + f" (within {scatter:g} of {value})"
        if relation in ("at most", "at least"):
            return f"{relation} {value}"
        alternatives = [value] + [pz for pz in pairs if pz.lower() != value.lower()]
        if relation == "counterpart":
            partners = alternatives[1:]
            return " or ".join(partners) + f" (the counterpart of {value})" if partners else value
        if relation == "differ":
            return " and ".join("not " + alternative for alternative in alternatives)
        return " or ".join(alternatives)

    def decide(self, cite, target, value, pairs):
        if not str(value or "").strip():
            self.provide_aloft(cite, f"{target} gives no value for it")
            return
        cite["chain"].append(target)
        cite.update(status="resolved", value=str(value).strip(), goes_with=list(pairs))
        concluded = f"{cite['head']}: {self.decided_measurement(cite)}"
        cite["now"] = concluded
        self.switch_amidst(cite, concluded)
        self.needs.resolved.append(concluded)
        self.needs.edition += 1
        self.drive.diary.append({"kind": "ref_resolved", "req": cite["req"][:200], "now": concluded, "chain": list(cite["chain"])})
        chain = cite["chain"]
        throughvia = f", reached from {chain[0]} by following the listings' notes" if len(chain) > 1 else ""
        spliced = " and ".join(pairs)
        if cite["relation"] == "differ":
            surplus = f"; its notes pair {spliced} with it" if pairs else ""
            surplus += ". The shopper's value MUST be DIFFERENT from that one: any other value satisfies it, that one does not"
        elif pairs:
            surplus = f"; its notes say {cite['value']} goes together with {spliced}, so any of them satisfies it"
        else:
            surplus = ""
        self.notify_operator(f"RESOLVED: {cite['head']} is set by another listing, and {target} shows {cite['value']}{throughvia}{surplus}. "
                        f"From now on the requirement reads: {concluded}.")

    def switch_amidst(self, cite, novel_dem):
        needs = self.needs
        seller = cite_designation(cite["first"]["seller"])

        def encompassed(req):
            if self.cite_concord(cite, req):
                return True
            head, _, value = str(req).partition(":")
            enclosed = re.findall(r"\(([^)]*)\)", head)
            if enclosed and cite["tag"] and not variety_vocables(enclosed[-1]) & variety_vocables(cite["tag"]):
                return False
            return bool(seller) and seller in cite_designation(value) and equal_feat(self.front_feat(head), cite["prop"])

        def exchanged(dems):
            return [novel_dem if encompassed(req) else req for req in dems]

        handled = False
        for trait in ("written", "uncovered"):
            aged = list(getattr(needs, trait))
            novel_alt = exchanged(aged)
            handled = handled or novel_alt != aged
            setattr(needs, trait, list(dict.fromkeys(novel_alt)))
        if needs.presently_dems is not None:
            actual = exchanged(needs.presently_dems)
            if novel_dem not in actual:
                actual.append(novel_dem)
            needs.presently_dems = list(dict.fromkeys(actual))
        if not handled:
            needs.uncovered.append(novel_dem)
        if isinstance(needs.browsing.get("requirements"), list):
            needs.browsing["requirements"] = list(needs.written)
        concluded_measurement = novel_dem.partition(":")[2].strip()
        logs = [y_alt for y_alt in needs.browsing.get("items") or [] if isinstance(y_alt, dict) and isinstance(y_alt.get("requirements"), list)]
        several = len(logs) >= 2
        for element_alt in logs:
            if several and cite["tag"] and not (variety_vocables(element_alt.get("product") or "") & variety_vocables(cite["tag"])):
                continue
            element_alt["requirements"] = [f"{str(req).partition(':')[0].strip()}: {concluded_measurement}" if encompassed(req) else req
                                     for req in element_alt["requirements"]]

    def provide_aloft(self, cite, why):
        if cite["status"] != "open":
            return
        cite["status"], cite["why"] = "stuck", why
        self.drive.diary.append({"kind": "ref_stuck", "req": cite["req"][:200], "why": why[:300], "chain": list(cite["chain"])})
        self.notify_operator(f"NOT RESOLVED: {cite['head']} ({why}). Its value remains unknown: when the notes of a listing you have read "
                        "show where the chain ends, use that value; otherwise do not decline or reject any listing because of it.")

    def notify_operator(self, text):
        chat = self.drive.chat
        rear = chat[-1] if chat else {}
        addable = rear.get("role") in ("tool", "user") and isinstance(rear.get("content"), str)
        if chat and addable and len(chat) > 2:
            rear["content"] += "\n[reference] " + text
            return
        self.awaiting_notifies = list(self.awaiting_notifies or []) + [text]

    def ref_line(self):
        try:
            components = []
            for cite in self.drive.refs or []:
                situation = cite["status"]
                if situation == "resolved":
                    differ = "; it must differ from that" if cite.get("relation") == "differ" else ""
                    components.append(f"{cite['now']} (as read on {cite['chain'][-1]}{differ})")
                elif situation == "open":
                    components.append(f"{cite['head']}: still being traced by the code (currently: {self.depict(cite)})")
                elif situation == "stuck":
                    components.append(f"{cite['head']}: unresolved, value unknown (do not decline or reject a listing because of it)")
            waitlisted, self.awaiting_notifies = self.awaiting_notifies, None
            text = "\n[references] " + "; ".join(components) if components else ""
            for notified in waitlisted or []:
                text += "\n[reference] " + notified
            return text
        except Exception:
            return ""

    @staticmethod
    def decided_orderline(cite):
        chain = cite["chain"]
        pairs = cite["goes_with"]
        throughvia = f" (reached from {chain[0]} by following the listings' notes)" if len(chain) > 1 else ""
        spliced = " and ".join(pairs)
        if cite.get("relation") == "differ":
            surplus = f"; its notes pair {spliced} with it" if pairs else ""
            return (f"- {cite['now']}: the shopper's {cite['prop']} must DIFFER from the one {chain[-1]} shows "
                    f"({cite['value']}){throughvia}{surplus}: every other value satisfies it, that value does not; the listings along "
                    f"its chain ({', '.join(chain)}) carry it.")
        if cite.get("relation") == "counterpart" and pairs:
            return (f"- {cite['now']}: the shopper wants the COUNTERPART of the value {chain[-1]} shows ({cite['value']}){throughvia}; "
                    f"its notes pair {cite['value']} with {spliced}, so {spliced} is required and {cite['value']} itself does not satisfy it.")
        surplus = f"; its notes say {cite['value']} goes together with {spliced}, so any of them satisfies it" if pairs else ""
        return f"- {cite['now']}: the value {chain[-1]} shows{throughvia}{surplus}."

    def cite_comment(self, entity=""):
        try:
            lines, tails = [], []
            for cite in self.drive.refs or []:
                situation = cite["status"]
                if situation == "resolved":
                    lines.append(self.decided_orderline(cite))
                    continue
                if situation not in ("open", "stuck"):
                    continue
                chain = cite["chain"]
                locale = ("the chain so far: " + " -> ".join(chain)) if chain else "the listing it points to was not found"
                lines.append(f"- {cite['head']}: the code was UNABLE TO RESOLVE it ({locale}). When a record below is that listing, or "
                             "the last one along its chain, and states its own value for it without sending the reader further, that "
                             "value is the requirement; otherwise its value is unknown, so never fail an item on that alone.")
                tails.extend(lbl for lbl in chain[-2:] if lbl != entity)
            if not lines:
                return ""
            text = ("\n\nREQUIREMENTS DEFINED BY POINTING AT ANOTHER LISTING (the code followed that listing's notes through the "
                    "store's records):\n" + "\n".join(lines))
            transcripts = [tile for lbl in dict.fromkeys(reversed(tails)) for tile in self.stash.regarding(lbl, 1)]
            if transcripts:
                text += "\nRECORDS OF THE LAST LISTINGS ALONG ITS CHAIN:\n" + "\n".join(transcripts[:2])
            return text
        except Exception:
            return ""


class Breadth:
    def __init__(self, drive):
        self.drive = drive
        self.every_impacts = []
        self.impact_merit = {}
        self.datarow_merit = {}
        self.targeted_upon = {}
        self.deposit_objectives = {}
        self.damaged_manners = set()
        self.offering_selectors = set()
        self.cost_manner = None
        self.origin_selectors = set()
        self.originated_commodities = set()
        self.overalls_noticed = {}
        self.routed_commodities = set()
        self.reverse_impacts_accomplished = None
        self.concluded_measurements = {}
        self.caption_tabulated = None
        self.assembled_collections = {}
        self.caption_endeavors = {}
        self.designation_cues = 0
        self.deepness_top = None
        self.folio_fingerprints = {}
        self.concise_cue_provided = None
        self.pitfall_cues = 0
        self.rated_routes = set()
        self.measurement_folios_browse = {}
        self.rated_route_selectors = {}
        self.accesses_through_commodity = {}
        self.tarrying_purge = {}
        self.shield_stamps = set()
        self.waitlisted_route = None

    @property
    def needs(self):
        return self.drive.needs

    @property
    def stash(self):
        return self.drive.stash

    def thriftiest_policy(self):
        return self.needs.priority() == "cheapest"

    def notify_utility(self, text):
        ultimate = self.drive.chat[-1]
        ultimate["content"] = str(ultimate["content"]) + "\n" + text

    def ultimate_be_utility(self):
        chat = self.drive.chat
        if not chat:
            return False
        return chat[-1].get("role") == "tool"

    @staticmethod
    def onto_hub(element_alt, hub):
        shelves = element_alt.get("row_shelf") or {}
        return [lbl for lbl, shelf in shelves.items() if shelf in hub]

    def costed_lineup(self, ids):
        price = self.stash.price
        return ", ".join(f"{lbl} ({price(lbl):g})" for lbl in ids)

    def might_folio(self, floor):
        drive = self.drive
        if drive.maneuvers_residual() <= floor + 1:
            return False
        channel = drive.channel
        if channel.done or channel.snapped:
            return False
        return channel.extra_alt() >= SEALING + 30

    # -- working out the store's price filter and how deep its listings reach
    def comment_deepness_top(self, trigger_operands, scrolling, hashmaps, total, obv):
        def sequel(key):
            return re.search(r"next|more|onward|continu|cursor", str(key).lower())

        def depleted(toggle):
            return toggle is None or toggle is False or toggle == ""

        at = trigger_operands.get(scrolling[0], scrolling[2])
        flags = [vlu for key, vlu in obv.items() if sequel(key) and not isinstance(vlu, (list, dict))]
        size = max(len(hashmaps), self.broadest_folio())
        if not be_numeral(at) or not flags or len(hashmaps) < size or not all(depleted(f) for f in flags):
            return
        attain = int((at - scrolling[2] + 1) * size)
        if total > attain:
            self.deepness_top = min(self.deepness_top or attain, attain)

    @staticmethod
    def constraint_parameter(trigger_operands, folio_selector, scrolling_selectors):
        def edge_akin(key, vlu):
            if not be_numeral(vlu) or isinstance(vlu, bool) or key == folio_selector or key in scrolling_selectors or not vlu > 1:
                return False
            return not re.search(r"page|size|count|result|min|low|from|start|offset", str(key).lower())
        discovered = [key for key, vlu in trigger_operands.items() if edge_akin(key, vlu)]
        return discovered[0] if len(discovered) == 1 else None

    @staticmethod
    def creator_unbound(footing, hashmaps, edge_selector, aisle_selector, immutable):
        def onto_universally_datarow(text):
            wish = text.strip().lower()

            def datarow_possesses(row):
                return any(isinstance(box, str) and box.strip().lower() == wish for name, box in row.items()
                           if not re.search(r"categor|shelf|department|aisle", str(name).lower()))
            return all(datarow_possesses(row) for row in hashmaps)

        def dropped(key, vlu):
            if key in (edge_selector, aisle_selector):
                return True
            return isinstance(vlu, str) and bool(vlu.strip()) and key not in immutable and onto_universally_datarow(vlu)
        return {key: vlu for key, vlu in footing.items() if not dropped(key, vlu)}

    def glean_winnow(self, trigger_operands, rows, total, obv):
        observe = self.drive.observe
        scrolling = observe.folio_operand()
        folio_selector = scrolling[0] if scrolling else None
        footing = {key: vlu for key, vlu in trigger_operands.items() if key != folio_selector}
        self.overalls_noticed[json.dumps(footing, sort_keys=True, default=str)] = int(total)
        hashmaps = [row for row in rows if isinstance(row, dict)]
        if scrolling and scrolling[1] is True and hashmaps and isinstance(obv, dict):
            self.comment_deepness_top(trigger_operands, scrolling, hashmaps, total, obv)
        _, mode, _, scrolling_selectors = observe.hunt_selectors(trigger_operands)
        edge_selector = self.constraint_parameter(trigger_operands, folio_selector, scrolling_selectors)
        if edge_selector is None or not hashmaps:
            return
        constraint = float(trigger_operands[edge_selector])
        captions = {observe.aisle_relating(row) for row in hashmaps}

        def designations_aisle(key, vlu):
            return isinstance(vlu, str) and key != folio_selector and bool(vlu.strip()) and vlu.strip().lower() in captions
        aisle_selectors = [key for key, vlu in trigger_operands.items() if designations_aisle(key, vlu)]
        aisle_selector = aisle_selectors[0] if len(aisle_selectors) == 1 else None
        immutable = {key: vlu for key, vlu in mode if key != aisle_selector}
        sig = json.dumps([sorted(immutable.items()), edge_selector], default=str)
        rates = [self.datarow_cost(row) for row in hashmaps]
        if any(pz is not None and pz > constraint + 1e-9 for pz in rates):
            # a row priced over the limit: this argument does not filter by price
            self.damaged_manners.add(sig)
            return
        if sig in self.damaged_manners or aisle_selector is None or len(captions) != 1 or "" in captions or None in rates:
            return
        stencil = self.creator_unbound(footing, hashmaps, edge_selector, aisle_selector, immutable)
        words = sum(len(vlu.split()) for key, vlu in stencil.items() if isinstance(vlu, str) and key not in immutable)
        props = (self.drive.contours.get(self.drive.role("search") or "") or {}).get("properties") or {}
        maintained = self.cost_manner
        if maintained and maintained["sig"] not in self.damaged_manners and maintained["words"] < words:
            return
        edge_brief = (props.get(edge_selector) or {}) if isinstance(props, dict) else {}
        self.cost_manner = dict(sig=sig, tmpl=stencil, mode=immutable, shelf=aisle_selector, bound=edge_selector, words=words,
                               int=edge_brief.get("type") == "integer")

    @staticmethod
    def datarow_cost(row):
        return next((float(vlu) for key, vlu in row.items()
                     if be_numeral(vlu) and not isinstance(vlu, bool) and re.search(r"price|cost", str(key).lower())), None)

    def cost_brief(self):
        brief = self.cost_manner
        return brief if brief and brief["sig"] not in self.damaged_manners else None

    def gx_offering(self, args):
        brief = self.cost_brief()
        if not (brief and isinstance(args, dict)):
            return False
        mode = brief["mode"]
        if mode:
            return all(args.get(key) == vlu for key, vlu in mode.items())
        return brief["bound"] in args

    def broadest_folio(self):
        return max((element_alt.get("size", 0) for element_alt in self.drive.page_book.values()), default=0)

    def attainable_deepness(self):
        scrolling = self.drive.observe.folio_operand()
        size = self.broadest_folio()
        if not scrolling or size <= 0 or scrolling[1] == "token" or scrolling[3] is None:
            return self.deepness_top
        first, ultimate = scrolling[2], scrolling[3]
        if scrolling[1]:
            return int((ultimate - first + 1) * size)
        return int(ultimate - first + size)

    def aisle_orthography(self, captions):
        if not captions:
            return ""
        known = self.stash.ids
        reckoning = {}

        def rendered_aisle(fields):
            for key, vlu in fields.items():
                name = str(key).lower()
                if isinstance(vlu, str) and 0 < len(vlu.strip()) <= 60 and re.search(r"categor|shelf|department|aisle", name)\
                        and "id" not in name:
                    return vlu.strip()
            return ""

        def count(fields):
            if not any(isinstance(vlu, str) and vlu in known and "id" in str(key).lower() for key, vlu in fields.items()):
                return
            rendered_alt = rendered_aisle(fields)
            if rendered_alt and rendered_alt.lower() in captions:
                reckoning[rendered_alt] = reckoning.get(rendered_alt, 0) + 1
        for transcript in self.stash.items:
            route_hashmaps(transcript, count)
        return max(sorted(reckoning), key=lambda orthography: reckoning[orthography], default="")

    def merely_under(self, price):
        if not be_numeral(price):
            return None
        brief = self.cost_brief() or {}
        vlu = float(price)
        if vlu.is_integer():
            lower = int(vlu) - 1
        elif brief.get("int"):
            lower = int(vlu)
        else:
            lower = round(vlu - 0.01, 2)
        return lower if lower > 0 else None

    def median(self, a, g_alt):
        brief = self.cost_brief() or {}
        if brief.get("int") or (float(a).is_integer() and float(g_alt).is_integer()):
            return (int(a) + int(g_alt)) // 2
        return round((float(a) + float(g_alt)) / 2, 2)

    @staticmethod
    def winnow_operands(brief, label, bound):
        entire = brief.get("int") or float(bound).is_integer()
        return {**brief["tmpl"], brief["shelf"]: label, brief["bound"]: int(bound) if entire else bound}

    def gx_commodities(self):
        count = len(self.needs.items())
        return list(range(count)) if count else [None]

    def able_lineup_under(self, kx):
        if kx is None and self.needs.items():
            return False
        if not self.thriftiest_policy() or kx in self.routed_commodities or self.cost_brief() is None:
            return False
        if self.drive.observe.folio_operand() is None:
            return False
        return bool(self.aisle_orthography(self.drive.walk.route_hub(kx)))

    def browse_onto_hub(self, hub):
        drive, stash = self.drive, self.stash
        referee = drive.referee

        def censuses(lbl):
            if referee.aisle_recognized(lbl) not in hub or not be_numeral(stash.price(lbl)):
                return False
            gone = lbl in drive.shelf.inert_orderlines() or lbl in drive.refused_lines or referee.ultimate_supply(lbl) is False
            return not gone
        return [float(stash.price(lbl)) for lbl in stash.read if censuses(lbl)]

    def objective_cost(self, kx):
        walk = self.drive.walk
        select = walk.route_select(kx)
        maintained = self.deposit_objectives.get(kx)
        expenses = []
        if select:
            expenses.append(select[1])
        if be_numeral(maintained):
            expenses.append(float(maintained))
        if not expenses and self.possesses_quantity_edge(kx):
            hub = walk.route_hub(kx)
            seen = [cost for cost, _ in self.thriftiest_unbrowsed(self.datarows_onto(hub), kx)]
            seen += self.browse_onto_hub(hub)
            if seen:
                expenses = [min(seen)]
        return min(expenses) if expenses else None

    def possesses_quantity_edge(self, kx):
        if kx is None or not self.needs.items():
            return False
        commodity_relating = self.drive.speak.commodity_relating

        def constrained(req):
            head, _, value = str(req).partition(":")
            return commodity_relating(head) == kx and re.search(r"\d", value) and re.search(HX_EDGE, value.lower())
        return any(constrained(req) for req in self.needs.feats_residual())

    def datarows_onto(self, hub):
        observe, known = self.drive.observe, self.stash.ids
        discovered = set()

        def note(fields):
            lbl = observe.datarow_stamp(fields)
            if lbl in known and observe.aisle_relating(fields) in hub:
                discovered.add(lbl)
        for transcript in self.stash.items:
            route_hashmaps(transcript, note)
        return discovered

    def thriftier_amidst_supply(self, hub, bound):
        observe, known = self.drive.observe, self.stash.ids
        situation = {"hit": False}

        def investigate(fields):
            if situation["hit"] or observe.aisle_relating(fields) not in hub:
                return
            if not any(isinstance(vlu, str) and vlu in known and "id" in str(key).lower() for key, vlu in fields.items()):
                return
            price = self.datarow_cost(fields)
            situation["hit"] = price is not None and price <= bound and observe.supply_situation(fields) is True
        for transcript in self.stash.items:
            route_hashmaps(transcript, investigate)
        return situation["hit"]

    def vended_outward_stamps(self, ids):
        observe = self.drive.observe
        marks = {}

        def note(fields):
            if not any(isinstance(vlu, str) and vlu in ids and "id" in str(key).lower() for key, vlu in fields.items()):
                return
            in_stock = observe.supply_situation(fields)
            if in_stock is not None:
                marks[observe.datarow_stamp(fields)] = not in_stock
        for transcript in self.stash.items:
            route_hashmaps(transcript, note)
        return marks

    def thriftiest_unbrowsed(self, ids, kx, under=None):
        cluster = set(ids)
        if not cluster:
            return []
        rows = self.needs.items()
        amt = rows[kx]["quantity"] if kx is not None and kx < len(rows) else self.needs.quantity()
        budget = self.needs.disburse_top
        vended_outward = self.vended_outward_stamps(cluster)
        discovered = []
        for lbl in cluster:
            price = self.stash.price(lbl)
            if lbl in self.stash.read or lbl in self.drive.refused_lines or vended_outward.get(lbl) or not be_numeral(price):
                continue
            above_allowance = be_numeral(budget) and be_numeral(amt) and price * amt > budget
            overly_costly = under is not None and price >= under
            if not above_allowance and not overly_costly:
                discovered.append((float(price), lbl))
        return sorted(discovered)

    def every_vocables(self):
        kind = self.drive.vocable_variety
        if kind == "all":
            return True
        return bool(self.drive.guide_vocables) and kind != "any"

    def measurement_guided_datarows(self, kx):
        if not self.every_vocables():
            return set()
        walk = self.drive.walk
        hub = walk.route_hub(kx)
        discovered = set()
        for element_alt in self.drive.page_book.values():
            if element_alt.get("id_order") is not True or self.gx_offering(element_alt["args"]):
                continue
            if walk.route_transports(element_alt["args"], kx):
                discovered.update(self.onto_hub(element_alt, hub))
        return discovered

    def personal_prerequisites(self, kx):
        several = bool(self.needs.items()) and kx is not None
        commodity_relating = self.drive.speak.commodity_relating
        return [(n, req) for n, req in enumerate(self.needs.feats_residual())
                if not several or commodity_relating(req.partition(":")[0]) in (kx, None)]

    @staticmethod
    def bare_measurement(req):
        return re.sub(r"\((?:stated|revealed[^)]*)\)", "", req.partition(":")[2]).lower()

    def terminated_measurement_route(self, element_alt, hub, kx, solo):
        if element_alt.get("id_order") is not True or self.gx_offering(element_alt["args"]) or not element_alt.get("first"):
            return False
        unbrowsed_residual = self.drive.observe.ensuing_folio(element_alt) is not None or len(element_alt.get("row_shelf") or {}) < element_alt["total"]
        if unbrowsed_residual:
            return False
        return bool(self.onto_hub(element_alt, hub) and self.drive.walk.route_transports(element_alt["args"], kx) & solo)

    def measurement_hunt_exhaustive(self, kx):
        try:
            if not self.every_vocables():
                return False
            hub = self.drive.walk.route_hub(kx)

            def mono_alternative(req):
                alternatives = [pz for pz in re.split(r"[,;/]|\bor\b|\beither\b|\bnor\b", self.bare_measurement(req)) if pz.strip()]
                return len(alternatives) == 1
            solo = {n for n, req in self.personal_prerequisites(kx) if mono_alternative(req)}
            return any(self.terminated_measurement_route(element_alt, hub, kx, solo) for element_alt in self.drive.page_book.values())
        except Exception:
            return False

    def wished_vocables(self, kx):
        walk, root_alt = self.drive.walk, self.drive.observe.root_alt
        wished_alt, figures = [], []
        for _, req in self.personal_prerequisites(kx):
            value = self.bare_measurement(req)
            if cite_prose(value):
                continue
            for component in re.split(r"[,;/]|\bor\b|\beither\b|\bnor\b", value):
                statistic = walk.dig_quantity(component)
                if statistic:
                    figures.append(statistic)
                    continue
                if walk.quantity_component(component):
                    continue
                words = [root_alt(wq) for wq in re.findall(r"[a-z0-9]{3,}", component)]
                if words:
                    wished_alt.append(words)
        return wished_alt, figures

    def datarow_pens_measurement(self, lbl, kx):
        root_alt = self.drive.observe.root_alt
        wished_alt, figures = self.wished_vocables(kx)
        if not (wished_alt or figures):
            return False
        proses = []

        def collect_alt(fields):
            if any(isinstance(vlu, str) and vlu == lbl and "id" in str(key).lower() for key, vlu in fields.items()):
                proses.extend(str(vlu).lower() for vlu in fields.values() if isinstance(vlu, str))
        for transcript in self.stash.items:
            route_hashmaps(transcript, collect_alt)
        body = " ".join(proses)
        possess = {root_alt(wq) for wq in re.findall(r"[a-z0-9]{3,}", body)}
        if any(re.search(r"(?<![\d.,])" + re.escape(statistic) + r"\s*-?\s*" + re.escape(root_alt(unit)), body) for statistic, unit in figures):
            return True
        return any(all(wq in possess for wq in words) for words in wished_alt)

    # -- searches that join a required value with the category label
    def flurry_space(self):
        drive = self.drive
        left = 0
        if drive.soliciting and drive.role("message"):
            left = max(0, drive.solicit_constraint() - drive.txts_delivered)
        return left, drive.maneuvers_residual() - drive.admission.checkout_expected() - 2 * left

    def alternative_vocables(self, shard, personal_designation):
        walk = self.drive.walk
        statistic = walk.dig_quantity(shard)
        if re.search(PRECLUDE_VOCABLES, shard) or (walk.quantity_component(shard) and not statistic):
            return None
        text = re.sub(r"\b(?:" + MEASUREMENT_ANCHORS + r")\b", " ", re.sub(r"'s\b", "", shard))
        # a number written with separators is dropped; the figure's own number and unit survive the filters
        words = []
        for vocable in re.findall(r"\d+(?:[.,]\d+)+|[a-z0-9]+", text):
            if len(vocable) < 2 and not (statistic and vocable == statistic[0]):
                continue
            if re.search(r"[.,]", vocable) or vocable in ANCHOR_VOCABLES:
                continue
            if vocable in personal_designation and not (statistic and vocable == statistic[1]):
                continue
            words.append(vocable)
        return words

    def solid_measurements(self, kx):
        def fragments_relating(minor):
            if re.search(r"\bor\b|\beither\b|/", minor):
                return [pz for pz in re.split(r"[,;/]|\bor\b|\beither\b", minor) if pz.strip()][:3]
            return [re.split(r"[,;&+]|\band\b", minor)[0]]

        values = []
        for head, value in self.drive.sealer.routine_measurements(kx):
            minor = re.sub(r"\([^)]*\)", " ", str(value).lower().replace("’", "'")).strip()
            if not minor or cite_prose(minor):
                continue
            personal_designation = set(re.findall(r"[a-z0-9]+", str(head).lower()))
            alternatives = []
            for shard in fragments_relating(minor):
                words = self.alternative_vocables(shard, personal_designation)
                if words and words not in alternatives:
                    alternatives.append(words)
            if alternatives and alternatives not in values:
                values.append(alternatives)
        return values

    def headline_roots_onto(self, kx):
        observe = self.drive.observe
        hub = self.drive.walk.route_hub(kx)
        legends = []

        def note(fields):
            if observe.aisle_relating(fields) not in hub or observe.datarow_stamp(fields) not in self.stash.ids:
                return
            title = self.drive.shelf.headline_relating(fields).lower()
            legends.append({observe.root_alt(wq) for wq in re.findall(r"[a-z0-9]+", title)})
        for transcript in self.stash.items:
            route_hashmaps(transcript, note)
        return legends

    def label_words(self, kx, label):
        scrubbed = re.sub(r"'s\b", "", label.lower().replace("’", "'"))
        words = list(dict.fromkeys(wq for wq in re.findall(r"[a-z0-9]+", scrubbed) if len(wq) >= 2 and wq not in ANCHOR_VOCABLES))
        if self.caption_tabulated is not False:
            return words
        root_alt = self.drive.observe.root_alt
        legends = self.headline_roots_onto(kx)
        if legends:
            # label words that half or more of the shelf's titles also carry
            usual = [wq for wq in words if 2 * sum(1 for tz in legends if root_alt(wq) in tz) >= len(legends)]
            if usual:
                return usual
        rows = self.needs.items()
        if rows and kx is not None and kx < len(rows):
            product = str(rows[kx]["product"])
        else:
            product = str(self.needs.browsing.get("product") or "")
        kind = [wq for wq in re.findall(r"[a-z0-9]+", product.lower()) if len(wq) > 2 and wq not in ANCHOR_VOCABLES]
        return kind[-1:]

    def hunttext_prose(self, components, caption_stipulations, top_sz):
        root_alt = self.drive.observe.root_alt
        lead = self.drive.guide_vocables
        roots_alt, selected = set(), []
        for vocable in [wq for component in components for wq in component] + caption_stipulations:
            rootnode = root_alt(vocable)
            if rootnode in roots_alt:
                continue
            roots_alt.add(rootnode)
            selected.append(vocable)
        text = " ".join(selected[:lead] if lead else selected)
        if not be_numeral(top_sz):
            return text
        while len(text) > top_sz and " " in text:
            text = text.rsplit(" ", 1)[0]
        return text[: int(top_sz)]

    def assembled_hunttexts(self, kx, label, top_sz):
        values = self.solid_measurements(kx)
        if not values:
            return [], []
        caption_stipulations = self.label_words(kx, label)
        universally_alternative = self.thriftiest_policy()

        def alternatives(value):
            return value if universally_alternative else value[:1]

        def separate(moulds):
            return list(dict.fromkeys(mould for mould in moulds if mould))

        def prose_relating(components):
            return self.hunttext_prose(components, caption_stipulations, top_sz)
        solo = [prose_relating([alternative]) for alternative in alternatives(values[0])]
        if len(values) < 2:
            return separate(solo), []
        scatter = next((n for n, value in enumerate(values) if len(alternatives(value)) > 1), 0)
        pooled_alt = [prose_relating([alternative if n == scatter else value[0] for n, value in enumerate(values)])
                    for alternative in alternatives(values[scatter])]
        return separate(pooled_alt), separate(solo)

    def ensuing_mould(self, first, second, prose_selector, mode):
        for moulds in (first, second):
            if not moulds:
                continue
            overalls = [self.delivered_overall(mould, prose_selector, mode) for mould in moulds]
            if None in overalls:
                return moulds[overalls.index(None)]
            if any(overalls):
                return ""
        return None if (first or second) else ""

    def mv_manner(self, args):
        props = (self.drive.contours.get(self.drive.role("search") or "") or {}).get("properties")
        if not isinstance(props, dict):
            props = {}

        def little_optionset(brief):
            return isinstance(brief, dict) and isinstance(brief.get("enum"), list) and len(brief["enum"]) <= 4
        return tuple(sorted((key, vlu) for key, vlu in args.items() if isinstance(vlu, str) and little_optionset(props.get(key))))

    def vocable_collection(self, text):
        root_alt = self.drive.observe.root_alt
        tokens = re.findall(r"[a-z0-9]+", str(text).lower())
        return frozenset(root_alt(wq) for wq in tokens[: self.drive.guide_vocables or None])

    def delivered_overall(self, text, prose_selector, mode):
        wished_alt = self.vocable_collection(text)

        def deciphered(key):
            try:
                return json.loads(key)
            except Exception:
                return None
        for key, total in (self.overalls_noticed or {}).items():
            args = deciphered(key)
            if not isinstance(args, dict) or not isinstance(args.get(prose_selector), str):
                continue
            if not all(args.get(name) == vlu for name, vlu in mode.items()):
                continue
            if self.vocable_collection(args[prose_selector]) == wished_alt:
                return int(total)
        return None

    def comment_caption_tabulating(self, label):
        if self.caption_tabulated is not None or not self.stash.items:
            return
        observe = self.drive.observe
        shelf = label.strip().lower()
        caption_roots = {observe.root_alt(wq) for wq in re.findall(r"[a-z0-9]+", re.sub(r"'s\b", "", label.lower()))
                       if len(wq) >= 2 and wq not in ANCHOR_VOCABLES}
        seen = {"hits": 0}

        def count(fields):
            if observe.datarow_stamp(fields) not in self.stash.ids or observe.aisle_relating(fields) != shelf:
                return
            otherwhere = set()
            for key, vlu in fields.items():
                if isinstance(vlu, str) and not re.search(r"categor|shelf|department|aisle", str(key).lower()):
                    otherwhere.update(observe.root_alt(wq) for wq in re.findall(r"[a-z0-9]+", vlu.lower()))
            # a label word found nowhere but in the category field means the label itself matched
            seen["hits"] += bool(caption_roots - otherwhere)
        route_hashmaps(self.stash.items[-1], count, ceiling=3)
        if seen["hits"] >= 2:
            self.caption_tabulated = True

    def shorten_unto_phases(self, waitlist, lot, floor):
        def cost():
            return 1 if lot and len(waitlist) >= 2 else len(waitlist)
        while waitlist and cost() >= self.drive.maneuvers_residual() - floor:
            waitlist.pop()
        return waitlist

    def inexpensive_waitlist(self, kx, ids, floor, maximal):
        cheapest = self.thriftiest_policy()
        select = self.drive.walk.route_select(kx) if cheapest else None
        lot = self.drive.shelf.lot_utility()
        opened = self.accesses_through_commodity
        room = ROUNDUP_ACCESSES - opened.get(kx, 0) if cheapest else maximal
        take = max(0, min(lot[2] if lot else maximal, room))
        ranked = self.thriftiest_unbrowsed(ids, kx, select[1] if select else None)
        waitlist = self.shorten_unto_phases([lbl for _, lbl in ranked][:take], lot, floor)
        if cheapest:
            opened[kx] = opened.get(kx, 0) + len(waitlist)
        return waitlist

    def initiate_browses(self, waitlist, houses, floor, utter):
        drive = self.drive
        aisle_recognized = drive.referee.aisle_recognized
        drive.rank_checked = {stamp_alt for stamp_alt in drive.rank_checked if aisle_recognized(stamp_alt.split("|")[0]) not in houses}
        drive.order_vetted = False
        browses = drive.browses
        browses.access_waitlist = waitlist
        browses.access_revalidate = True
        browses.access_utter = utter
        browses.access_intention = "walk"
        browses.access_bottom = floor
        drive.referee.access_designated()

    def caption_stencil(self, kx):
        unbound_prose = self.drive.observe.unbound_prose
        for trigger in self.drive.sealer.routine_hunttexts(kx):
            proses = unbound_prose(trigger)
            selectors = [key for key, vlu in trigger.items() if isinstance(vlu, str) and vlu.strip() and vlu in proses]
            if len(selectors) == 1 and not self.gx_offering(trigger):
                return trigger, selectors[0]
        return None, ""

    def awaiting_mould(self, kx, label, hub, top_sz, prose_selector, mode):
        def construct():
            first, second = self.assembled_hunttexts(kx, label, top_sz)
            self.assembled_collections.setdefault(str(kx), set()).update(self.vocable_collection(mould) for mould in first + second)
            return first, second
        first, second = construct()
        agenda = self.ensuing_mould(first, second, prose_selector, mode)
        if agenda is not None:
            return agenda, first, second
        walk = self.drive.walk
        conveyed = any(walk.route_transports(wq["args"], kx) and self.onto_hub(wq, hub) for wq in self.drive.page_book.values())
        if not conveyed or self.caption_tabulated is not None:
            return "", first, second
        # the label is not a search term here: title words (or the product's kind word) take its place
        self.caption_tabulated = False
        first, second = construct()
        return self.ensuing_mould(first, second, prose_selector, mode) or "", first, second

    def measurement_caption_hunt(self):
        _, room = self.flurry_space()
        if room <= 4:
            return False
        drive = self.drive
        walk = drive.walk
        contour = drive.contours.get(drive.role("search") or "") or {}
        props = contour.get("properties") if isinstance(contour.get("properties"), dict) else {}
        scrolling = drive.observe.folio_operand()
        products = self.gx_commodities()

        def readied(kx):
            # the checked search to send for product k; None when there is none (a dead end is remembered as settled)
            if self.caption_endeavors.get(kx, 0) >= CAPTION_TRIALS:
                return None
            sig = json.dumps(self.solid_measurements(kx))
            concluded = self.concluded_measurements.setdefault(str(kx), [])
            if sig in concluded:
                return None
            hub = walk.route_hub(kx)
            label = self.aisle_orthography(hub)
            if not label:
                return None
            stencil, prose_selector = self.caption_stencil(kx)
            if stencil:
                mode = {key: vlu for key, vlu in stencil.items() if isinstance(props.get(key), dict) and props[key].get("enum")}
                # strip it to a plain keyword search that matches what the actor would send
                required = contour.get("required") or []
                pruned = {key: vlu for key, vlu in stencil.items() if key == prose_selector or key in mode or key in required}
                prose_brief = props.get(prose_selector)
                top_sz = prose_brief.get("maxLength") if isinstance(prose_brief, dict) else None
                agenda, first, second = self.awaiting_mould(kx, label, hub, top_sz, prose_selector, mode)
                if agenda:
                    args, faults = OperandValidate.actuate({**pruned, prose_selector: agenda}, contour)
                    if not faults and isinstance(args, dict):
                        return args, agenda, prose_selector, mode, label, hub, first or second
            concluded.append(sig)
            return None

        for kx in products:
            primed = readied(kx)
            if primed is None:
                continue
            args, agenda, prose_selector, mode, label, hub, moulds = primed
            self.caption_endeavors[kx] = self.caption_endeavors.get(kx, 0) + 1
            multi_alt = "s" if len(self.solid_measurements(kx)) > 1 else ""
            walk.route_trigger(args, f"Searching for the required value{multi_alt} together with the category name spelled as this "
                                 f"store's rows spell it ('{agenda}'): the keyword search matches listing values, titles and "
                                 "categories, so the result lists the category's listings that carry the value.")
            self.afterward_caption_hunt(kx, args, agenda, prose_selector, mode, label, hub, moulds, scrolling, len(products) > 1)
            return True
        return False

    def afterward_caption_hunt(self, kx, args, agenda, prose_selector, mode, label, hub, moulds, scrolling, several):
        drive = self.drive
        total = self.delivered_overall(agenda, prose_selector, mode)
        self.comment_caption_tabulating(label)
        key = json.dumps({name: vlu for name, vlu in args.items() if not scrolling or name != scrolling[0]}, sort_keys=True, default=str)
        element_alt = drive.page_book.get(key)
        every = bool(element_alt) and len(element_alt.get("row_shelf") or {}) >= element_alt["total"] and bool(self.onto_hub(element_alt, hub))
        floor = drive.admission.checkout_expected() + 2
        waitlist = []
        if total is None:
            note = ""
        elif total:
            note = f"[value search] '{agenda}': {total} match(es)" + (" - all of them are shown here." if every else ".")
        else:
            note = f"[value search] '{agenda}': nothing matched."
        cheapest = self.thriftiest_policy()
        if every and cheapest and scrolling and element_alt.get("id_order") is not False:
            self.rated_routes.add(key)
            self.rated_route_selectors.setdefault(str(kx), []).append(key)
            if several:
                self.tarrying_purge[kx] = True
            else:
                waitlist = self.inexpensive_waitlist(kx, self.onto_hub(element_alt, hub), floor, ROUNDUP_ACCESSES)
        elif every and not cheapest and agenda in moulds:
            waitlist = self.inexpensive_waitlist(kx, self.onto_hub(element_alt, hub), floor, CAPTION_ACCESSES)
        if waitlist:
            under = " below the pick" if drive.walk.route_select(kx) and self.thriftiest_policy() else ""
            note += (" Its unread in-stock rows within the budget in the requested category" + under + " are opened next, "
                     "cheapest first: " + self.costed_lineup(waitlist) + ".")
        if note and self.ultimate_be_utility():
            self.notify_utility(note + ("" if waitlist else "\n" + drive.standing_orderline()))
        drive.diary.append({"kind": "label_search", "product": kx, "query": agenda, "total": total, "every": every, "opened": waitlist})
        if waitlist:
            self.initiate_browses(waitlist, hub, floor, "Opening that search's cheapest unread rows: every listing in the category "
                                                  "carrying the required values is among its matches.")

    def purge_clusters(self, tarrying, per):
        walk = self.drive.walk
        tome = self.drive.page_book
        selectors = self.rated_route_selectors or {}

        def cluster_towards(kx):
            hub = walk.route_hub(kx)
            itemized = set()
            for key in selectors.get(str(kx), []):
                if key in tome and walk.route_transports(tome[key]["args"], kx):
                    itemized.update(self.onto_hub(tome[key], hub))
            select = walk.route_select(kx)
            room = max(0, min(per, ROUNDUP_ACCESSES - self.accesses_through_commodity.get(kx, 0)))
            ranked = self.thriftiest_unbrowsed(itemized, kx, select[1] if select else None)
            return [lbl for _, lbl in ranked][:room]
        return [cluster_towards(kx) for kx in tarrying]

    @staticmethod
    def grab_stints(tarrying, clusters, maximal):
        # round-robin over the products' pools, never taking a row twice
        holder, waitlist = {}, []
        while len(waitlist) < maximal and any(clusters):
            for kx, cluster in zip(tarrying, clusters):
                while cluster and cluster[0] in waitlist:
                    cluster.pop(0)
                if not cluster or len(waitlist) >= maximal:
                    continue
                holder[cluster[0]] = kx
                waitlist.append(cluster.pop(0))
        return holder, waitlist

    def purge_tarrying(self, floor):
        tarrying = [kx for kx, on in self.tarrying_purge.items() if on]
        self.tarrying_purge.clear()
        if not tarrying:
            return False
        drive = self.drive
        lot = drive.shelf.lot_utility()
        per = 2 if drive.maneuvers_residual() - floor - 1 >= 2 * len(tarrying) else 1
        clusters = self.purge_clusters(tarrying, per)
        holder, waitlist = self.grab_stints(tarrying, clusters, lot[2] if lot else per * len(tarrying))
        if not self.shorten_unto_phases(waitlist, lot, floor):
            return False
        opened = self.accesses_through_commodity
        for lbl in waitlist:
            ofwhich = holder[lbl]
            opened[ofwhich] = opened.get(ofwhich, 0) + 1
        houses = set().union(*(drive.walk.route_hub(kx) for kx in tarrying))
        self.initiate_browses(waitlist, houses, floor, "Cheapest wanted: opening the cheapest unread rows listed by each product's value search.")
        abstract = ("[value searches read] Opened, below each pick, the cheapest unread in-stock rows within the budget listed by "
                   "each product's value search: " + self.costed_lineup(waitlist) + ". For each product, choose the cheapest "
                   "listing whose own fields meet every requirement in force; picking a pricier fit requires every cheaper "
                   "candidate to be opened and rejected first.")
        if self.ultimate_be_utility():
            self.notify_utility(abstract)
        drive.diary.append({"kind": "value_flush", "products": tarrying, "opened": waitlist})
        return True

    def commodity_phases(self):
        try:
            rows = self.needs.items()
            if len(rows) < 2 or not self.thriftiest_policy():
                return ""
            read = self.measurement_folios_browse or {}
            components, missing = [], ""
            for kx, row in enumerate(rows):
                select = self.drive.walk.route_select(kx)
                components.append(f"{row['product']}: {read.get(kx, 0)} value page(s) read, pick {'yes' if select else 'no'}")
                if not (missing or select):
                    missing = str(row["product"])
            rear = f"; next: {missing}" if missing else ""
            return "\n[product steps] " + "; ".join(components) + rear + "."
        except Exception:
            return ""

    # -- paging value-led searches through to their last page
    def terminated_hunts(self, kx):
        observe, walk = self.drive.observe, self.drive.walk

        def done(wq):
            if wq.get("id_order") is not True or not wq.get("first") or observe.ensuing_folio(wq) is not None:
                return False
            return not self.gx_offering(wq["args"]) and len(wq.get("row_shelf") or {}) >= wq["total"]
        return [(self.vocable_collection(" ".join(observe.unbound_prose(wq["args"]))), walk.route_transports(wq["args"], kx, alts=True), self.mv_manner(wq["args"]))
                for wq in self.drive.page_book.values() if done(wq)]

    def commodity_stipulations(self, kx, hub):
        root_alt = self.drive.observe.root_alt
        commodity_relating = self.drive.speak.commodity_relating
        several = bool(self.needs.items()) and kx is not None
        stipulations = set()
        for req in self.needs.feats_residual():
            head, _, value = req.partition(":")
            if several and commodity_relating(head) not in (kx, None):
                continue
            stipulations.update(root_alt(wq) for wq in re.findall(r"[a-z0-9]{2,}", value.lower()))
        label = {root_alt(wq) for shelf in hub for wq in re.findall(r"[a-z0-9]{2,}", shelf)}
        return stipulations | label

    def access_measurement_route(self, key, element_alt, kx, hub):
        if key in self.rated_routes or element_alt.get("id_order") is not True or not element_alt.get("first"):
            return False
        if self.drive.observe.ensuing_folio(element_alt) is None or self.gx_offering(element_alt["args"]):
            return False
        return bool(self.drive.walk.route_transports(element_alt["args"], kx)) and bool(self.onto_hub(element_alt, hub))

    def folio_contenders(self, kx, n, hub, plan):
        drive = self.drive
        observe, walk = drive.observe, drive.walk
        utilized, opened = self.measurement_folios_browse, self.accesses_through_commodity
        known, depth = plan["known"], plan["depth"]
        minee = (self.assembled_collections or {}).get(str(kx)) or set()
        stipulations = self.commodity_stipulations(kx, hub)
        finished = self.terminated_hunts(kx)

        def encompassed(words, pairs, methodq):
            # a finished search that showed all its matches already holds whatever this one could add
            return any(mode == methodq and (seen <= words or (seen <= stipulations and pairs <= obtained)) for seen, obtained, mode in finished)

        discovered = []
        for key, element_alt in drive.page_book.items():
            if not self.access_measurement_route(key, element_alt, kx, hub):
                continue
            words = self.vocable_collection(" ".join(observe.unbound_prose(element_alt["args"])))
            assembled = words in minee
            overly_thorough = depth is not None and element_alt["total"] > depth
            if known:
                if overly_thorough and not assembled:
                    continue
            elif depth is None or overly_thorough:
                continue
            pairs = walk.route_transports(element_alt["args"], kx, alts=True)
            methodq = self.mv_manner(element_alt["args"])
            if known and encompassed(words, pairs, methodq):
                continue
            attain = element_alt["total"] if depth is None else min(element_alt["total"], depth)
            pages = -(-attain // max(1, element_alt["size"])) - len(element_alt["read"])
            permitted_alt = min(plan["per"] - utilized.get(kx, 0), plan["cap"] - sum(utilized.values()))
            if plan["several"]:
                stockpile = 0 if opened.get(kx) else plan["opens"]
                permitted_alt = min(permitted_alt, plan["share"] - utilized.get(kx, 0) - opened.get(kx, 0) - stockpile)
            if pages <= permitted_alt:
                take = pages
            else:
                take = permitted_alt if assembled else 0
            if take <= 0 or drive.maneuvers_residual() - take - plan["opens"] <= plan["floor"]:
                continue
            if (take > 2 or plan["several"]) and plan["left"] and plan["room"] > take + plan["opens"] + 4:
                plan["pending"] = True
                continue
            discovered.append((take, -walk.route_merit(element_alt["args"], kx), key, words))
        if known:
            # between two nested searches keep the one whose extra words are not value or label words
            def overshadowed(minee_vocables, further_vocables):
                if further_vocables < minee_vocables and not (minee_vocables - further_vocables) <= stipulations:
                    return True
                return minee_vocables < further_vocables and (further_vocables - minee_vocables) <= stipulations
            discovered = [cc for cc in discovered if not any(overshadowed(cc[3], oq[3]) for oq in discovered)]
        return [(utilized.get(kx, 0), take, merit, n, key, kx) for take, merit, key, _ in discovered]

    def scrolling_strategy(self, products, several):
        drive = self.drive
        depth = self.attainable_deepness()
        size = max(1, self.broadest_folio())
        left, room = self.flurry_space()
        walk = drive.walk
        unbrowsed_houses = sum(1 for kx in products if walk.route_hub(kx) and not self.measurement_folios_browse.get(kx))
        per = max(MEASUREMENT_FOLIOS, -(-depth // size) - 1) if depth else MEASUREMENT_FOLIOS
        cap = max(MEASUREMENT_OVERALL, MEASUREMENT_OVERALL_APIECE * len(products))
        opens = 1 if drive.shelf.lot_utility() else (2 if several else ROUNDUP_ACCESSES)
        return {"depth": depth, "per": per, "several": several, "cap": cap, "opens": opens,
                "floor": drive.admission.checkout_expected() + 2, "known": self.every_vocables(), "left": left, "room": room,
                "share": max(4, room // max(1, unbrowsed_houses)), "pending": False}

    def value_pages(self):
        drive = self.drive
        try:
            diagnosing = any(cite.get("status") == "open" for cite in drive.refs or [])
            if diagnosing and drive.citecounter.cite_space():
                # a reference is still being traced and has room: leave the steps to it
                return False
            channel = drive.channel
            obstructed = drive.orders_tried or drive.refusal_n or channel.done or channel.snapped
            if obstructed or channel.extra_alt() < SEALING + 60 or not drive.role("search"):
                return False
            if self.measurement_caption_hunt():
                return True
            if not self.thriftiest_policy() or drive.vocable_variety == "any" or not drive.observe.folio_operand():
                return False
            products = self.gx_commodities()
            several = len(products) > 1
            plan = self.scrolling_strategy(products, several)
            alternatives = []
            for n, kx in enumerate(products):
                hub = drive.walk.route_hub(kx)
                if hub:
                    alternatives.extend(self.folio_contenders(kx, n, hub, plan))
            if alternatives:
                _, take, _, _, key, kx = min(alternatives, key=lambda alternative: alternative[:5])
                self.rated_routes.add(key)
                return self.browse_measurement_route(kx, key, plan["floor"], take, several)
            tarrying = self.tarrying_purge
            if not tarrying or plan["pending"]:
                return False
            if plan["left"] and plan["room"] > plan["opens"] * len(tarrying) + 4:
                return False
            return self.purge_tarrying(plan["floor"])
        except Exception as failure:
            drive.diary.append({"kind": "value_pages_error", "error": f"{type(failure).__name__}: {failure}"[:300]})
            return False

    def browse_folios(self, kx, element_alt, scrolling, maximal, floor, utter):
        observe, walk = self.drive.observe, self.drive.walk
        retrieved, essayed = 0, False
        while retrieved < maximal and observe.ensuing_folio(element_alt) is not None and self.might_folio(floor):
            essayed = True
            self.measurement_folios_browse[kx] = self.measurement_folios_browse.get(kx, 0) + 1
            succeeding = observe.ensuing_folio(element_alt)
            if not walk.route_trigger({**element_alt["args"], scrolling[0]: succeeding}, utter):
                break
            retrieved += 1
            if observe.ensuing_folio(element_alt) == succeeding:
                # the cursor did not move on: stop rather than loop
                break
        return retrieved, essayed

    def browse_measurement_route(self, kx, key, floor, maximal=MEASUREMENT_FOLIOS, defer=False):
        drive = self.drive
        observe = drive.observe
        scrolling = observe.folio_operand()
        element_alt = drive.page_book.get(key)
        if not scrolling or not element_alt:
            return False
        query = " ".join(observe.unbound_prose(element_alt["args"]))
        known = self.every_vocables()
        if known:
            utter = ("Cheapest wanted, and this store's keyword search requires every word while listing results in a fixed id "
                   f"order rather than by price: going on to the next page of '{query}' so that every listing with these words gets seen.")
        else:
            utter = ("Cheapest wanted, and this store lists search results in a fixed id order rather than by price: going on to the "
                   f"next page of '{query}' so that every listing it matched gets seen.")
        obtained, tested = self.browse_folios(kx, element_alt, scrolling, maximal, floor, utter)
        if not obtained:
            return tested
        self.rated_route_selectors.setdefault(str(kx), []).append(key)
        seen = len(element_alt.get("row_shelf") or {})
        upon_terminus = observe.ensuing_folio(element_alt) is None
        complete = upon_terminus and seen >= element_alt["total"]
        hub = drive.walk.route_hub(kx)
        if complete:
            attain = " - all listings containing these words are included" if known else " - all listings it matched are included"
        elif upon_terminus:
            attain = " - the rest is out of reach, so a cheaper fit may sit beyond the reachable rows"
        else:
            attain = ""
        head = f"[value search read] '{query}': {seen} of its {element_alt['total']} matches seen" + attain + ". "
        sealing = ("Select the lowest-priced listing whose own fields satisfy every requirement in force; picking a pricier fit "
                   "requires every cheaper candidate to be opened and rejected first.")
        waitlist = []
        if defer:
            self.tarrying_purge[kx] = True
            midst = "Once each product's value search has been read, its cheapest unread in-stock rows within the budget get opened. "
        else:
            select = drive.walk.route_select(kx)
            waitlist = self.inexpensive_waitlist(kx, self.onto_hub(element_alt, hub), floor, ROUNDUP_ACCESSES)
            if waitlist:
                under = " below the pick" if select else ""
                midst = ("Its cheapest unread in-stock rows within the budget in the requested category" + under
                          + " are opened next: " + self.costed_lineup(waitlist) + ". ")
            else:
                midst = "None of them is an unread in-stock row within the budget in the requested category. "
        abstract = head + midst + sealing
        if waitlist:
            self.initiate_browses(waitlist, hub, floor, "Cheapest wanted: opening that search's cheapest unread rows.")
        if self.ultimate_be_utility():
            self.notify_utility(abstract + ("" if waitlist else "\n" + drive.standing_orderline()))
        drive.diary.append({"kind": "value_pages", "product": kx, "query": query, "pages": obtained, "complete": complete, "opened": waitlist,
                        "defer": defer})
        return True

    # -- walking the price filter down below a pick
    def route_bottom(self, floor, mend):
        if mend:
            return floor + 2
        if self.drive.orders_tried:
            return floor
        return max(floor, self.drive.admission.checkout_expected() + 2)

    def terminate_offering(self, element_alt, args, scrolling, calls, floor):
        observe = self.drive.observe
        while observe.ensuing_folio(element_alt) is not None and calls < ROUNDUP_TRIGGERS and self.might_folio(floor):
            calls += 1
            succeeding = observe.ensuing_folio(element_alt)
            moved = self.drive.walk.route_trigger({**args, scrolling[0]: succeeding}, "Moving on to the following page of that price-filter listing.")
            if not moved or observe.ensuing_folio(element_alt) == succeeding:
                break
        return calls, observe.ensuing_folio(element_alt) is None

    def smaller_constraint(self, element_alt, bound, total, calls):
        expenses = (self.stash.price(lbl) for lbl in (element_alt.get("row_shelf") or {}))
        rates = [float(cc) for cc in expenses if be_numeral(cc)]
        minor = min(rates) if rates else 0.0
        # go halfway down, or lower still when the total spread across the price range fits in the calls left
        populate = max(1, ROUNDUP_TRIGGERS - calls) * max(1, element_alt["size"])
        resized = minor + (bound - minor) * 1.8 * populate / max(total, 1)
        return min(self.median(minor, bound), self.median(minor, resized))

    def ceiling_walk(self, kx, floor, mend=False):
        drive = self.drive
        observe, walk = drive.observe, drive.walk
        brief, scrolling = self.cost_brief(), observe.folio_operand()
        hub = walk.route_hub(kx)
        label = self.aisle_orthography(hub)
        aim = self.objective_cost(kx)
        if not brief or not scrolling or not label or aim is None:
            return None
        self.routed_commodities.add(kx)
        select_presently = walk.route_select(kx)
        self.targeted_upon[kx] = (float(aim), select_presently[0] if select_presently else "")
        bound = self.merely_under(aim)
        if bound is None:
            return None
        floor = self.route_bottom(floor, mend)
        depth = self.attainable_deepness()
        tome = drive.page_book
        selectors = []
        calls, apex, complete, drove, ultimate = 0, None, False, False, bound
        if select_presently or be_numeral((self.deposit_objectives or {}).get(kx)):
            which = "the pick"
        else:
            which = "the cheapest row seen"
        utter = f"Cheapest wanted: listing '{label}' priced below {which} ({aim:g}) with the store's price filter."

        def costed_above(element_alt, constraint):
            return any(be_numeral(self.stash.price(ii)) and self.stash.price(ii) > constraint for ii in (element_alt.get("row_shelf") or {}))
        while calls < ROUNDUP_TRIGGERS and self.might_folio(floor):
            args = self.winnow_operands(brief, label, bound)
            key = json.dumps(args, sort_keys=True, default=str)
            self.offering_selectors.add(key)
            calls += 1
            buffered = key in tome and bool(tome[key].get("first")) and key in (self.overalls_noticed or {})
            if not (buffered or walk.route_trigger(args, utter)):
                break
            drove, ultimate = True, bound
            selectors.append(key)
            element_alt = tome.get(key)
            if element_alt and costed_above(element_alt, bound):
                self.damaged_manners.add(brief["sig"])
            if self.cost_brief() is None:
                break
            total = self.overalls_noticed.get(key)
            if total is None:
                break
            if total and element_alt:
                pages = -(-total // max(1, element_alt["size"]))
                fits = depth is None or total <= depth
                if fits and pages - len(element_alt["read"]) <= ROUNDUP_TRIGGERS - calls:
                    calls, complete = self.terminate_offering(element_alt, args, scrolling, calls, floor)
                    break
                # too many rows under this limit: remember it as the ceiling and try a lower one
                apex = bound
                reduced = self.smaller_constraint(element_alt, bound, total, calls)
                workable = reduced < bound
            elif self.thriftier_amidst_supply(hub, bound):
                self.damaged_manners.add(brief["sig"])
                break
            elif apex is None:
                complete = True
                break
            else:
                # nothing under this limit: search back up between it and the last crowded one
                reduced = self.median(bound, apex)
                workable = bound < reduced < apex
            if not workable:
                break
            bound = reduced
        if not drove:
            return None
        return self.route_bulletin(kx, selectors, hub, aim, label, ultimate, complete, which, calls)

    def route_bulletin(self, kx, selectors, hub, aim, label, ultimate, complete, which, calls):
        tome = self.drive.page_book
        ids = set().union(*(self.onto_hub(tome.get(key, {}), hub) for key in selectors))
        optns = [lbl for _, lbl in self.thriftiest_unbrowsed(ids, kx, aim)]
        rated = self.measurement_guided_datarows(kx)
        # rows led by a value search first, then rows whose own text carries the value, then the rest
        lead, remainder = [], []
        for lbl in optns:
            (lead if lbl in rated else remainder).append(lbl)
        prosy = [lbl for lbl in remainder if self.datarow_pens_measurement(lbl, kx)]
        naked = [] if self.measurement_hunt_exhaustive(kx) else [lbl for lbl in remainder if lbl not in prosy]
        ranked = lead + prosy + naked
        if not complete:
            bulletin = (f"price filter on '{label}' under {aim:g}: {len(ids)} cheaper listings seen ({len(optns)} of them unread, "
                      "in stock and inside the budget)")
        elif ids:
            bulletin = (f"price filter on '{label}': all listings costing at most {ultimate:g} seen ({len(ids)}; {len(optns)} "
                      "unread, in stock and inside the budget)")
        else:
            bulletin = f"price filter on '{label}' under {aim:g}: none of its listings is cheaper than {which}"
        self.drive.diary.append({"kind": "ceiling_walk", "product": kx, "label": label, "aim": aim, "calls": calls, "complete": complete,
                             "rows": len(ids), "ranked": ranked[:ROUNDUP_ACCESSES]})
        return bulletin, ranked[:ROUNDUP_ACCESSES]

    def summit_comb(self, tested):
        observe = self.drive.observe
        brief, scrolling = self.cost_brief(), observe.folio_operand()
        if not brief or not scrolling or self.needs.items():
            return None, False
        rows = observe.aisle_datarows()
        shelves = observe.aisle_captions()
        label = self.aisle_orthography(shelves)
        if not rows or not label:
            return None, False
        bound = self.merely_under(min(cost for cost, _, _ in rows.values()))
        if bound is None:
            return None, True

        def serialized(args):
            return json.dumps(args, sort_keys=True, default=str)
        args = self.winnow_operands(brief, label, bound)
        key = serialized(args)
        if key not in tested:
            self.offering_selectors.add(key)
            return args, False
        element_alt = self.drive.page_book.get(key)
        if not element_alt:
            return None, self.overalls_noticed.get(key) == 0 and not self.thriftier_amidst_supply(set(shelves), bound)
        succeeding = observe.ensuing_folio(element_alt)
        if succeeding is None:
            return None, len(element_alt.get("row_shelf") or {}) >= element_alt["total"]
        additional = {**args, scrolling[0]: succeeding}
        return (None, False) if serialized(additional) in tested else (additional, False)

    def shielded_offering(self, trigger_operands):
        drive = self.drive
        named = [vlu for vlu in trigger_operands.values() if isinstance(vlu, str) and vlu in self.stash.ids]
        if len(named) != 1:
            return None
        lbl = named[0]
        if lbl in self.stash.read or lbl in drive.wagon.metrics:
            return None
        if lbl in drive.admission.designated_access() or drive.refs or self.needs.browsing.get("references"):
            return None
        return lbl

    def measurement_shield(self, trigger_designation, trigger_operands):
        drive = self.drive
        try:
            conditional = (trigger_designation == drive.role("details") and self.thriftiest_policy() and self.every_vocables()
                     and not drive.admission.amidst_stockpile())
            lbl = self.shielded_offering(trigger_operands) if conditional else None
            if lbl is None:
                return ""
            kx = drive.walk.commodity_offset(lbl)
            stamp_alt = (kx, self.needs.edition)
            if (kx is None and self.needs.items()) or stamp_alt in self.shield_stamps:
                return ""
            rated = self.measurement_guided_datarows(kx)
            if not rated or lbl in rated:
                return ""
            select = drive.walk.route_select(kx)
            personal_cost = self.stash.price(lbl)
            expenses = [select[1]] if select else []
            if be_numeral(personal_cost):
                expenses.append(float(personal_cost))
            if not expenses:
                return ""
            rows = self.thriftiest_unbrowsed(rated, kx, min(expenses))[:6]
            if not rows:
                return ""
            self.shield_stamps.add(stamp_alt)
            versus = " than both it and the pick" if select else " than it"
            itemized = ", ".join(f"{ii} ({cc:g})" for cc, ii in rows)
            return ("Not run (no step used). The cheapest is wanted, and this store's keyword search requires every word, so the "
                    f"rows shown by its searches led by a required value carry those words somewhere in their records; {lbl} is "
                    "not one of them, while these rows of theirs cost less" + versus + " and are still unread: " + itemized
                    + ". Open those first, lowest price first; repeat this call afterwards if you still want it.")
        except Exception as failure:
            drive.diary.append({"kind": "value_guard_error", "error": f"{type(failure).__name__}: {failure}"[:300]})
            return ""

    def headline_orderline(self, kx, rcd, stencil):
        drive = self.drive
        root_alt = drive.observe.root_alt
        prohibited = set(ANCHOR_VOCABLES)
        for aisle_caption in drive.walk.route_hub(kx):
            prohibited.update(re.findall(r"[a-z]+", aisle_caption))
        for wished_alt in self.needs.feats_residual():
            prohibited.update(re.findall(r"[a-z0-9]+", wished_alt.lower()))
        logs = self.needs.items() or [{"product": str(self.needs.browsing.get("product") or "")}]
        for element_alt in logs:
            prohibited.update(re.findall(r"[a-z]+", str(element_alt.get("product") or "").lower()))
        prohibited_roots = {root_alt(vocable) for vocable in prohibited}
        words = re.findall(r"[a-z0-9]+", drive.shelf.headline_relating(rcd).lower())

        def workable(vocable):
            return vocable not in prohibited and root_alt(vocable) not in prohibited_roots
        # the first fresh title word (plus its fresh neighbour) becomes the query, unless it already ran
        for ofs, vocable in enumerate(words):
            if not (re.fullmatch(r"[a-z]{3,}", vocable) and workable(vocable)):
                continue
            fwd = words[ofs + 1] if ofs + 1 < len(words) else ""
            phrase = f"{vocable} {fwd}" if fwd and workable(fwd) else vocable
            if drive.walk.drove_hunttext(phrase):
                return None
            return {**stencil[0], stencil[1]: phrase}
        return None

    def reroute_inert_objective(self):
        drive = self.drive
        if not self.thriftiest_policy() or drive.channel.done or drive.channel.extra_alt() < SEALING + 60 or not drive.role("search"):
            return None
        objectives = self.targeted_upon or {}
        mislaid = drive.shelf.inert_orderlines() | drive.refused_lines
        access_expense = 1 if drive.shelf.lot_utility() else 2
        for kx, (aim, lbl) in list(objectives.items()):
            if not lbl or lbl not in mislaid:
                continue
            if self.cost_brief() is None or drive.observe.folio_operand() is None:
                continue
            select = drive.walk.route_select(kx)
            if not select or select[1] <= aim + 1e-9:
                continue
            if drive.maneuvers_residual() < PRIMARY_CHECKOUT_THROUGH + 3 + access_expense:
                continue
            # the listing the price aim was set from is gone: drop the aim and walk that product again
            objectives.pop(kx, None)
            (self.deposit_objectives or {}).pop(kx, None)
            drive.diary.append({"kind": "ceil_rearm", "product": kx, "dead": lbl, "aim": select[1]})
            return drive.walk.drive_route([], PRIMARY_CHECKOUT_THROUGH, [kx], mend=True)
        return None

    def stockpile_route(self, extra_alt, opens):
        walk = self.drive.walk
        checkout_expected = self.drive.admission.checkout_expected
        plan = walk.route_strategy(ROUTE_OVERALL)
        if not plan:
            return False
        cost = sum(n for _, _, n in plan) + sum(1 for kx, _, _ in plan if walk.route_select(kx))
        if extra_alt > checkout_expected() + 2 + cost + opens:
            return False
        pruned = walk.route_strategy(min(ROUTE_OVERALL, extra_alt - checkout_expected() - 1 - opens))
        return walk.drive_route(pruned, checkout_expected()) if pruned else False

    def route_phase(self):
        drive = self.drive
        walk = drive.walk
        try:
            if walk.reject_waitlist:
                return walk.drive_reject_browses()
            if drive.__dict__.get("deeper_queue") or self.deeper_ready():
                return self.deeper_pages()
            pending = self.waitlisted_route
            self.waitlisted_route = None
            if pending:
                return walk.drive_route(*pending)
            reloaded = self.reroute_inert_objective()
            if reloaded is not None:
                return reloaded
            if drive.orders_tried or not self.thriftiest_policy() or drive.channel.extra_alt() < SEALING + 60 or drive.channel.done\
                    or self.commodity_unlocated():
                return False
            access_expense = 1 if drive.shelf.lot_utility() else 2
            extra_alt = drive.maneuvers_residual()
            checkout_expected = drive.admission.checkout_expected

            def traversable(kx):
                if not (walk.route_select(kx) or self.possesses_quantity_edge(kx)):
                    return False
                return self.able_lineup_under(kx) and self.objective_cost(kx) is not None
            primed = [kx for kx in self.gx_commodities() if traversable(kx)]
            if extra_alt > checkout_expected() + 2 + ROUNDUP_TRIGGERS + access_expense:
                if primed:
                    # products still without a pick lead, then those with the fewest value pages read
                    folios_browse = self.measurement_folios_browse or {}
                    primed.sort(key=lambda kx: (bool(walk.route_select(kx)), folios_browse.get(kx, 0), -1 if kx is None else kx))
                    return walk.drive_route([], checkout_expected() + 2, primed[:1])
                if self.origin_cost_winnow():
                    return True
            return self.stockpile_route(extra_alt, access_expense)
        except Exception as failure:
            drive.diary.append({"kind": "walk_error", "error": f"{type(failure).__name__}: {failure}"[:300]})
            return False

    @staticmethod
    def aisle_parameter(props, label, omit):
        for key, brief in props.items():
            if not isinstance(brief, dict) or key in omit:
                continue
            if not re.search(r"categor|shelf|department|aisle", str(key).lower()):
                continue
            choices = brief.get("enum")
            if isinstance(choices, list):
                target = label.strip().lower()
                match = next((cc for cc in choices if isinstance(cc, str) and cc.strip().lower() == target), None)
                if match is not None:
                    return key, match
            elif brief.get("type") == "string":
                return key, label
        return None, None

    @staticmethod
    def cost_parameter(props, folio_selector):
        for key, brief in props.items():
            if not isinstance(brief, dict) or key == folio_selector:
                continue
            if brief.get("type") not in ("integer", "number") or brief.get("enum"):
                continue
            name = str(key).lower()
            glances_akin_top = re.search(r"ceil|max|price|cap$|budget|upper|most", name)
            if glances_akin_top and not re.search(r"page|first|offset|skip|count|size|num", name):
                return key
        return None

    @staticmethod
    def coupled_manners(props, aisle_selector, clauses):
        aisle_vocable = r"\b" + re.escape(aisle_selector.lower()) + r"\b"

        def notified_alongside_aisle(alternative):
            personal = r"\b" + re.escape(str(alternative).lower()) + r"\b"
            return any(re.search(personal, component) and re.search(aisle_vocable, component) for component in clauses)
        selected = {}
        for key, brief in props.items():
            if key == aisle_selector or not isinstance(brief, dict):
                continue
            alternatives = brief.get("enum")
            if not isinstance(alternatives, list) or not 2 <= len(alternatives) <= 4:
                continue
            # a small mode switch must be described together with the shelf argument, or the seed is off
            coupled = next((alternative for alternative in alternatives if notified_alongside_aisle(alternative)), None)
            if coupled is None:
                return None
            selected[key] = coupled
        return selected

    def origin_offering_operands(self, label, bound):
        drive = self.drive
        tool = drive.role("search")
        contour = drive.contours.get(tool or "") or {}
        props = contour.get("properties") if isinstance(contour.get("properties"), dict) else {}
        scrolling = drive.observe.folio_operand()
        if not props or not scrolling or not be_numeral(bound):
            return None
        snippet = next((str(element_alt["function"].get("description") or "") for element_alt in drive.utilitykit
                      if element_alt["function"]["name"] == tool), "")
        clauses = [shard.lower() for shard in re.split(r"[.;]", snippet)]
        token = drive.observe.morpheme_selector()
        aisle_selector, aisle_vlu = self.aisle_parameter(props, label, (token, scrolling[0]))
        edge_selector = self.cost_parameter(props, scrolling[0])
        if not (aisle_selector and edge_selector):
            return None
        wholenum = props[edge_selector].get("type") == "integer" or float(bound).is_integer()
        trigger = {aisle_selector: aisle_vlu, edge_selector: int(bound) if wholenum else bound}
        manners = self.coupled_manners(props, aisle_selector, clauses)
        if manners is None:
            return None
        trigger.update(manners)
        if any(key not in trigger for key in contour.get("required") or []):
            return None
        immutable, faults = OperandValidate.actuate(trigger, contour)
        if faults or not isinstance(immutable, dict):
            return None
        return immutable

    def origin_fingerprint(self, args, label, scrolling):
        manners = dict(self.drive.observe.hunt_selectors(args)[1])
        folio_selector = scrolling[0] if scrolling else None
        concluded = sorted((name, vlu) for name, vlu in manners.items()
                         if not (isinstance(args.get(name), str) and args[name] == label))
        numerical = next((name for name in args if be_numeral(args[name]) and name != folio_selector), "")
        return json.dumps([concluded, numerical], default=str)

    def origin_cost_winnow(self):
        drive = self.drive
        try:
            if self.cost_brief() is not None or self.cost_manner or not self.thriftiest_policy():
                return False
            scrolling = drive.observe.folio_operand()
            walk = drive.walk
            for kx in self.gx_commodities():
                if kx in self.originated_commodities or (self.needs.items() and kx is None):
                    continue
                select = walk.route_select(kx)
                label = self.aisle_orthography(walk.route_hub(kx))
                if not (select and label):
                    continue
                self.originated_commodities.add(kx)
                args = self.origin_offering_operands(label, self.merely_under(select[1]))
                if not args:
                    return False
                sig = self.origin_fingerprint(args, label, scrolling)
                unpaginated = {name: vlu for name, vlu in args.items() if not scrolling or name != scrolling[0]}
                key = json.dumps(unpaginated, sort_keys=True, default=str)
                if sig in self.damaged_manners or key in (self.overalls_noticed or {}):
                    continue
                self.origin_selectors.add(key)
                walk.route_trigger(args, f"Cheapest wanted: listing '{label}' below the pick's price ({select[1]:g}) to find out "
                                     "whether this store's price filter really applies.")
                drive.diary.append({"kind": "ceil_seed", "product": kx, "args": args, "learned": self.cost_brief() is not None})
                if self.ultimate_be_utility() and self.cost_brief() is None:
                    self.notify_utility(drive.standing_orderline())
                return True
            return False
        except Exception as failure:
            drive.diary.append({"kind": "ceil_seed_error", "error": f"{type(failure).__name__}: {failure}"[:300]})
            return False

    def commodity_unlocated(self):
        if not self.needs.items():
            return False
        try:
            opened = {name.strip().lower() for name in self.stash.aisles_browse()}
            if any(not hub or not hub & opened for hub in self.drive.shelf.hub_aisles()):
                return True
            edition = str(self.needs.edition)
            actual = any(stamp_alt.rpartition("|")[2] == edition for stamp_alt in self.drive.passed)
            return actual and self.drive.admission.unvalidated()
        except Exception:
            return False

    def maintain_objectives(self, ids, solely, room):
        walk = self.drive.walk
        itemized = []
        for kx in sorted(solely, key=lambda xq: -1 if xq is None else xq):
            if room < 2 or not self.able_lineup_under(kx):
                continue
            itemized.append(kx)
            room -= ROUNDUP_TRIGGERS
            rates = [float(pz) for pz in (self.stash.price(ii) for ii in ids if ii and walk.commodity_offset(ii) == kx) if be_numeral(pz)]
            if not rates:
                continue
            sooner = self.deposit_objectives.get(kx)
            if be_numeral(sooner):
                rates.append(float(sooner))
            self.deposit_objectives[kx] = min(rates)
        return itemized, room

    def deeper_pick(self, pids):
        """Deeper paging: the first of `pids` (product slots, None for a lone product) whose widest search - the one listing
        most rows of its category - still has unread pages, with how many the steps allow (up to DEEPER_FOLIOS, keeping the
        order's reserve, two steps per unsent message and a margin free). Products already paged this way are skipped."""
        drive = self.drive
        observe = drive.observe
        done = drive.__dict__.setdefault("deeper_done", set())
        cap = SOLICIT_BASELINE if drive.txt_top is None else drive.txt_top
        unsent = max(0, cap - drive.txts_delivered)
        room = drive.maneuvers_residual() - drive.admission.checkout_expected() - 2 * unsent - 6
        for pid in pids:
            if pid in done:
                continue
            hub = drive.walk.route_hub(pid)
            best_key, best_rows = None, 0
            for key, walk in (drive.page_book or {}).items():
                if not walk.get("first") or observe.ensuing_folio(walk) is None:
                    continue
                own = sum(1 for shelf in (walk.get("row_shelf") or {}).values() if not hub or shelf in hub)
                if own > best_rows:
                    best_key, best_rows = key, own
            if best_key is None or best_rows < 3:
                continue
            walk = drive.page_book[best_key]
            unread = -(-max(0, walk["total"] - len(walk.get("row_shelf") or {})) // max(1, walk["size"]))
            take = min(DEEPER_FOLIOS, unread, room)
            if take > 0:
                return pid, best_key, take
        return None

    def deeper_blocked(self):
        drive = self.drive
        # (best-by-a-measure only: a cheapest rule already has this store's cheapest walk, and paging twice starves the order)
        if drive.needs.priority() != "best" or drive.orders_tried or drive.refusal_n:
            return True
        if not drive.observe.folio_operand() or drive.__dict__.get("deeper_queue"):
            return True
        if drive.channel.extra_alt() < SEALING + 60:
            return True
        # an open pointer chain needs the steps first, and its value may change what fits
        return any(cite.get("status") == "open" for cite in drive.refs or [])

    def deeper_ready(self):
        """With nothing in the cart yet: queue the deeper pages while the steps still allow them, a couple of moves before
        they no longer would, so a late first cart put does not lose them."""
        drive = self.drive
        try:
            if self.deeper_blocked() or drive.wagon.metrics:
                return False
            cap = SOLICIT_BASELINE if drive.txt_top is None else drive.txt_top
            unsent = max(0, cap - drive.txts_delivered)
            room = drive.maneuvers_residual() - drive.admission.checkout_expected() - 2 * unsent - 6
            if room > DEEPER_FOLIOS + 2:
                return False
            products = list(range(len(drive.needs.items()))) or [None]
            picked = self.deeper_pick(products)
            if not picked:
                return False
            drive.__dict__.setdefault("deeper_done", set()).add(picked[0])
            drive.deeper_queue = picked
            drive.diary.append({"kind": "deeper_due", "product": picked[0], "pages": picked[2]})
            return True
        except Exception as exc:
            drive.diary.append({"kind": "deeper_ready_error", "error": f"{type(exc).__name__}: {exc}"[:300]})
            return False

    def deeper_keep(self, trigger_operands):
        """Before a product's first cart put under a best or cheapest rule: when its widest search still has unread pages
        the steps allow, hold the put once and queue those pages - a winner on the measure can sit on any page of a store
        that lists rows in a fixed order, and the rank check only weighs rows that were seen."""
        drive = self.drive
        try:
            if self.deeper_blocked():
                return ""
            hubs = drive.shelf.hub_aisles()
            several = bool(drive.needs.items())
            pids = []
            for ident, count in drive.wagon.lines(trigger_operands):
                if count is not None and count <= 0:
                    continue
                if not several:
                    pids = [None]
                    break
                label = drive.stash.caption_relating.get(ident)
                pid = next((n for n, hub in enumerate(hubs) if label and label in hub), None)
                if pid is not None and pid not in pids:
                    pids.append(pid)
            picked = self.deeper_pick(pids)
            if not picked:
                return ""
            pid, key, take = picked
            drive.__dict__.setdefault("deeper_done", set()).add(pid)
            drive.deeper_queue = picked
            walk = drive.page_book[key]
            query = " ".join(drive.observe.unbound_prose(walk["args"]))
            wanted = "the lowest price" if drive.needs.priority() == "cheapest" else (
                drive.needs.browsing.get("priority_detail") or "the measure asked for")
            drive.diary.append({"kind": "deeper_hold", "product": pid, "query": query, "pages": take})
            return (f"Not run (no step used). The shopper wants the winner on {wanted}, and the search '{query}' still has "
                    f"unread pages ({walk['total']} matches, listed in a fixed order rather than ranked): the code reads up to "
                    f"{take} more of them now. Then compare every fitting listing seen on that measure and put the winner in the cart.")
        except Exception as exc:
            drive.diary.append({"kind": "deeper_keep_error", "error": f"{type(exc).__name__}: {exc}"[:300]})
            return ""

    def deeper_pages(self):
        """Read the queued deeper pages (scripted, one page per step), keeping the order's reserve free."""
        drive = self.drive
        queued = drive.__dict__.pop("deeper_queue", None)
        if not queued:
            return False
        pid, key, take = queued
        operand = drive.observe.folio_operand()
        walk = (drive.page_book or {}).get(key)
        if not (operand and walk):
            return False
        query = " ".join(drive.observe.unbound_prose(walk["args"]))
        say = f"Reading the next page of '{query}', so the winner on the measure is chosen from every row it lists."
        keep = drive.admission.checkout_expected() + 2
        read = 0
        while (read < take and drive.observe.ensuing_folio(walk) is not None and drive.maneuvers_residual() > keep + 1
               and not drive.channel.done and drive.channel.extra_alt() >= SEALING + 30):
            nxt = drive.observe.ensuing_folio(walk)
            if not drive.walk.route_trigger({**walk["args"], operand[0]: nxt}, say):
                break
            read += 1
            if drive.observe.ensuing_folio(walk) == nxt:
                break
        seen = len(walk.get("row_shelf") or {})
        recap = (f"[pages read] '{query}': {seen} of {walk['total']} matches now seen. Compare every fitting listing seen on the "
                 "shopper's measure (open the strongest candidates' details first) and put the winner in the cart.")
        if drive.chat and drive.chat[-1].get("role") == "tool":
            drive.chat[-1]["content"] = str(drive.chat[-1]["content"]) + "\n" + recap + "\n" + drive.standing_orderline()
        drive.diary.append({"kind": "deeper_pages", "product": pid, "query": query, "pages": read})
        return read > 0

    def route_maintain(self, trigger_operands):
        drive = self.drive
        try:
            if not self.thriftiest_policy() or self.waitlisted_route or drive.channel.extra_alt() < SEALING + 60:
                return ""
            if drive.refusal_n and drive.maneuvers_residual() <= 15:
                return ""
            walk = drive.walk
            if trigger_operands is None:
                ids = list(drive.wagon.metrics)
            else:
                ids = [ii for ii, count in drive.wagon.lines(trigger_operands) if count is None or count > 0]
            solely = {walk.commodity_offset(ii) for ii in ids if ii}
            floor = PRIMARY_CHECKOUT_THROUGH + 1 if drive.orders_tried == 0 else 4
            # page reads, the priced line search and the opens split half of what is spare
            semi = (drive.maneuvers_residual() - floor - 1) // 2
            roundup, room = self.maintain_objectives(ids, solely, semi - (1 if drive.shelf.lot_utility() else 2))
            plan = walk.route_strategy(min(ROUTE_OVERALL, room), solely)
            if not (plan or roundup):
                return ""
            self.waitlisted_route = (plan, max(floor, drive.maneuvers_residual() - semi - 1), roundup)
            tome = drive.page_book
            unbound_prose = drive.observe.unbound_prose
            queries = [" ".join(unbound_prose(tome[key]["args"])) for _, key, _ in plan if key in tome]
            tasks = []
            if queries:
                tasks.append("reads the rest of the pages of " + ", ".join(f"'{w_alt}'" for w_alt in queries))
            if roundup:
                tasks.append("uses the store's price filter to list the category's listings cheaper than the pick")
            return ("Not run yet (no step used). The cheapest is wanted, but this store returns search results in a fixed id "
                    "order rather than by price, so a cheaper fit may be on a page nobody read: the code now "
                    + " and ".join(tasks) + " for you, then opens the cheapest unread rows under the pick. After that, send the "
                    "call again, or send it with a cheaper fit.")
        except Exception as failure:
            drive.diary.append({"kind": "walk_hold_error", "error": f"{type(failure).__name__}: {failure}"[:300]})
            return ""

    # -- notes added to a search result
    @staticmethod
    def outcome_datarows(shard):
        grids = [vlu for vlu in shard.values() if isinstance(vlu, list) and all(isinstance(rq, dict) for rq in vlu)]
        if not grids:
            return None
        return next((vlu for vlu in grids if vlu), grids[0])

    def unbox_datarows(self, obv):
        rows = self.outcome_datarows(obv)
        if rows is not None:
            return obv, rows
        # some stores wrap the rows in an inner dict next to a message
        for vlu in obv.values():
            if isinstance(vlu, dict) and self.outcome_datarows(vlu) is not None:
                return vlu, self.outcome_datarows(vlu) or []
        return None, None

    def merit_datarows(self, trigger_operands, rows):
        observe = self.drive.observe
        merit = observe.measurement_merit(trigger_operands) if rows else 0
        if not merit:
            return
        for row in rows:
            if not isinstance(row, dict):
                continue
            stamp_alt = observe.datarow_stamp(row)
            self.datarow_merit[stamp_alt] = max(self.datarow_merit.get(stamp_alt, 0), merit)

    def hunt_comment(self, trigger_operands, obv):
        if not isinstance(obv, dict):
            return ""
        obv, rows = self.unbox_datarows(obv)
        if obv is None:
            return ""
        drive = self.drive
        observe = drive.observe
        overalls = [vlu for vlu in obv.values() if isinstance(vlu, int) and not isinstance(vlu, bool) and vlu > len(rows)]
        correlated = max(overalls or [len(rows)])
        observe.tome_folio(trigger_operands, rows, correlated, obv)
        try:
            self.glean_winnow(trigger_operands, rows, correlated, obv)
        except Exception as failure:
            drive.diary.append({"kind": "ceil_learn_error", "error": f"{type(failure).__name__}: {failure}"[:300]})
        self.merit_datarows(trigger_operands, rows)
        notes = []
        kind = observe.vocable_policy(trigger_operands, obv, rows)
        lead = observe.guide_policy(trigger_operands, obv, rows)
        if lead:
            notes.append(self.guide_observation(trigger_operands, lead))
        proses, mode, first, scrolling_selectors = observe.hunt_selectors(trigger_operands)
        manner_selectors = dict(mode)
        crude = [vlu for key, vlu in trigger_operands.items() if isinstance(vlu, str) and key not in manner_selectors and key not in scrolling_selectors]
        if first and crude:
            self.hunttext_observations(trigger_operands, rows, crude, overalls, mode, scrolling_selectors, proses, kind, notes)
        if rows:
            if overalls:
                notes.append(self.unnoticed_observation(kind, overalls, rows))
            disregarded = self.above_constraint_observation(trigger_operands, rows)
            if disregarded:
                notes.append(disregarded)
        return " ".join(notes)

    def above_constraint_observation(self, trigger_operands, rows):
        scrolling = self.drive.observe.hunt_selectors(trigger_operands)[3]
        tops = [vlu for key, vlu in trigger_operands.items()
                if be_numeral(vlu) and vlu > 1 and key not in scrolling
                and not re.search(r"page|size|count|result|min|low|from|start", str(key).lower())]
        if not tops:
            return ""
        cap = max(tops)

        def atop(row):
            return any(be_numeral(vlu) and vlu > cap and "price" in str(key).lower() for key, vlu in row.items())
        costlier = [row for row in rows if atop(row)]
        if not costlier:
            return ""
        return (f"{len(costlier)} of these rows cost more than the limit {cap} this call set, so this search mode skipped "
                "that filter (and possibly others too, the category among them). Switch to the mode whose filters work, "
                "or check every row yourself.")

    def guide_observation(self, trigger_operands, lead):
        field, env_manner = self.drive.guide_env[0], self.drive.guide_env[1]
        proses, mode, _, _ = self.drive.observe.hunt_selectors(trigger_operands)
        words = ()
        if mode == env_manner:
            words = next((wq for key, wq in proses if key == field), ())
        enumerated = f" (of this query it kept only: {' '.join(words[:lead])})" if len(words) > lead else ""
        return (f"Only the FIRST {lead} word(s) of a query count in this store's keyword search ({field}); the rest is "
                f"dropped{enumerated}. Lead with the {lead} most distinctive words (a required value and the item kind, the "
                "value first), and look up other values in separate short queries.")

    def cue_observations(self, rows, crude, observations_alt):
        drive = self.drive
        # some stores skip one-letter query words entirely
        fragments = [pz for text in crude for pz in re.findall(r"[a-z0-9]+", text.lower())]
        if fragments and all(len(pz) == 1 for pz in fragments) and not rows and not self.concise_cue_provided:
            self.concise_cue_provided = True
            observations_alt.append("No results: this store's search skips one-character words, so a figure such as 0.4 or 5 x 7 "
                           "searches nothing (or only the other words). Search a value written as a word, or the unit word "
                           "plus the item kind, and read every page.")
        prop, investigate = drive.observe.measurement_investigate(crude)

        def possesses_space():
            return drive.maneuvers_residual() > drive.admission.checkout_expected() + 3
        if investigate:
            provided = self.designation_cues
            if provided >= 2 or not possesses_space():
                return
            self.designation_cues = provided + 1
            if provided:
                opening = f"Once more the query starts with a property's name ({prop})."
            else:
                opening = (f"This query starts with a property's NAME ({prop}) and holds no required value. Keyword search "
                           "matches listing values, titles and categories, and finds a field name only where a value or "
                           f"title repeats it, so listings whose {prop} field holds only the value are missed.")
            observations_alt.append(opening + f" Put the value first and the item kind after it: '{investigate}' (a figure the store cannot "
                           "read falls away and leaves its unit word), and read all of its pages.")
        elif self.pitfall_cues < 2 and possesses_space():
            pitfall, _ = drive.walk.vocable_pitfall(crude)
            if pitfall:
                self.pitfall_cues += 1
                observations_alt.append(pitfall)

    def hunttext_observations(self, trigger_operands, rows, crude, overalls, mode, scrolling_selectors, proses, kind, observations_alt):
        drive = self.drive
        self.cue_observations(rows, crude, observations_alt)
        marks = tuple(drive.observe.datarow_stamp(rq) for rq in rows)
        manner_selectors = dict(mode)
        remainder = tuple(sorted((name, str(vlu)) for name, vlu in trigger_operands.items()
                            if name not in scrolling_selectors and not (isinstance(vlu, str) and name not in manner_selectors)))
        sig = (mode, remainder, max(overalls or [len(rows)]), marks)
        query = " ".join(crude)
        sooner = self.folio_fingerprints.get(sig)
        if len(set(marks)) >= 3 and sig in self.folio_fingerprints and sooner != query:
            observations_alt.append(f"Your earlier search '{sooner}' came back with the same total and rows: the store treats both "
                           "the same, so spelling it differently shows nothing new. Read its next page, or change the words "
                           "it starts with.")
        self.folio_fingerprints.setdefault(sig, query)
        # when an all-words store shows its whole result, every row carries the query's words
        universally_vocable = bool(kind == "all" or drive.guide_vocables)
        notified = bool(universally_vocable and not overalls and rows)
        hits = []
        if notified or (kind != "any" and not overalls and rows):
            hits = self.tome_impacts(trigger_operands, rows, crude, notified, proses, observations_alt)
        if universally_vocable and not self.reverse_impacts_accomplished:
            self.reverse_impacts_accomplished = True
            self.sooner_impacts(trigger_operands, scrolling_selectors, hits, observations_alt)

    def tome_impacts(self, trigger_operands, rows, crude, notified, proses, observations_alt):
        drive = self.drive
        observe = drive.observe
        conserved = None if notified else list(drive.ref_hits or [])
        drive.ref_hits = []
        hits = observe.datarow_impacts(rows, crude)
        merit = observe.measurement_merit(trigger_operands)
        self.impact_merit.update({ii: max(self.impact_merit.get(ii, 0), merit) for ii in hits})
        self.every_impacts += [ii for ii in hits if ii not in self.every_impacts]
        if conserved is not None:
            drive.ref_hits = conserved
        if notified and hits and len(proses) and len(rows) <= 12:
            observations_alt.append("Every word is required here and every match is shown: each row holds the words the store read "
                           "somewhere in its record, often in fields its title leaves out. Open the ones within budget before "
                           "searching further: " + ", ".join(hits[:6]) + ".")
        if not notified:
            return hits
        directed, drive.ref_hits = drive.ref_hits, []
        arrows = [ii for ii in directed if ii not in hits][:6]
        if arrows:
            observations_alt.append("A value in the request refers to another listing, and these rows come from the shop or brand "
                           "this search names: open them regardless of price, stock or category (a listing that supplies a "
                           "value is read, not bought), and do what their notes say: " + ", ".join(arrows) + ".")
        return hits

    def sooner_impacts(self, trigger_operands, scrolling_selectors, hits, observations_alt):
        observe = self.drive.observe
        herein = json.dumps({name: vlu for name, vlu in trigger_operands.items() if name not in scrolling_selectors},
                          sort_keys=True, default=str)
        discovered = []
        for route_selector, element_alt in self.drive.page_book.items():
            if route_selector == herein or not element_alt.get("every") or not observe.measurement_merit(element_alt["args"]):
                continue
            crisp = [ii for ii in observe.datarow_impacts(element_alt["every"]) if ii not in discovered and ii not in hits]
            self.impact_merit.update({ii: max(self.impact_merit.get(ii, 0), observe.measurement_merit(element_alt["args"])) for ii in crisp})
            discovered += crisp
        self.every_impacts += [ii for ii in discovered if ii not in self.every_impacts]
        if discovered:
            observations_alt.append("This store's keyword search needs every word, so earlier searches that showed their whole result "
                           "listed only rows holding all of their words, often in fields the titles leave out. Open the ones "
                           "within budget that are still unread before searching further: " + ", ".join(discovered[:6]) + ".")

    @staticmethod
    def unnoticed_observation(kind, overalls, rows):
        if kind == "any":
            methodq = ("this store's keyword search accepts ANY of the words and ranks listings matching more of them higher, so "
                   "extra words widen it: a required value's own distinctive word or figure on its own lists just the "
                   "listings that carry it (often few enough to read every page and compare prices); the item kind plus the "
                   "values puts the closest matches first; use the category mode (or a price filter, if one holds) for the rest.")
        elif kind == "all":
            methodq = ("this store's keyword search needs EVERY word, so more exact words narrow it, and so do the category and "
                   "price filters with a lower price limit.")
        else:
            methodq = ("narrow it further for the rest: if every word is needed, add exact words; if any word matches (a longer "
                   "query that matches more listings shows this), put the required values into the query so fitting listings "
                   "come first; a price filter that holds helps as well.")
        return (f"{max(overalls)} listings matched this search but this page shows just {len(rows)}, and only a few pages can "
                "be read, so the listings that matter may be among those not shown. First read the further pages of this "
                "same search (as many as can be read); if the total exceeds those pages, " + methodq)

    def buyer_orderlines(self, trigger_designation, obv, refused):
        discovered = []
        if trigger_designation == self.drive.role("message"):
            if isinstance(obv, str):
                return [obv]
            if isinstance(obv, dict) and "error" not in obv:
                extended_textstrings([vlu for vlu in obv.values() if isinstance(vlu, (str, list))], discovered)
            return discovered
        if not isinstance(obv, dict):
            return discovered
        for shard in obv.values():
            if isinstance(shard, list) and shard and all(isinstance(vlu, str) for vlu in shard):
                extended_textstrings(shard, discovered)
            elif refused and isinstance(shard, str) and len(shard.split()) >= 4:
                discovered.append(shard)
        return discovered


class BrowseWaitlist:
    # listings the code itself will open next, and the words that announce it
    def __init__(self):
        self.access_waitlist = None
        self.access_utter = ""
        self.access_revalidate = False
        self.access_intention = ""
        self.access_bottom = None
        self.access_utility = None
        self.deposit_waitlist = None
        self.deposit_utility = None


class Admission:
    def __init__(self, drive):
        self.drive = drive
        self.browse_matter = None
        self.browse_upon = None
        self.phases_provided = None
        self.panel_jotting = {}
        self.cue_alternates = {}
        self.duplicate_solicits = set()
        self.primary_browses_accomplished = None
        self.tardy_w = 0
        self.tardy_idents = set()
        self.standby_k = 0
        self.switch_accomplished = None

    # understanding the request

    def browse_appeal(self):
        drive = self.drive

        def utility_orderline(element_alt):
            brief = element_alt["function"]
            return f"- {brief['name']}: {brief.get('description', '')} | arguments: {qua_prose(brief.get('parameters') or {})}"
        directory = "\n".join(utility_orderline(element_alt) for element_alt in drive.utilitykit)
        rear = "REQUEST:\n" + drive.request
        opening = drive.plook.get("opening")
        if opening is not None:
            rear += "\n\nOPENING OBSERVATION:\n" + qua_prose(opening)
        content = [
            {"type": "text", "text": BROWSE_APPEAL + "\n\nTOOLS:\n" + directory, "cache_control": {"type": "ephemeral"}},
            {"type": "text", "text": rear},
        ]
        self.browse_matter = content
        browsing = self.solicit_browsers(content)
        if not isinstance(browsing.get("roles"), dict):
            browsing = self.browsing_through_routine(browsing)
        self.shielded(lambda: self.validate_browsing(browsing), "check_reading_error")
        self.utilize_browsing(browsing)

    def validate_browsing(self, browsing):
        drive = self.drive
        text = drive.request
        amidst_prose = penned_statistics(text)
        figures = browsing.get("figures") if isinstance(browsing.get("figures"), dict) else {}
        for key in ("price_cap", "message_budget", "step_budget"):
            value = browsing.get(key)
            if not be_numeral(value) or float(value) in amidst_prose:
                continue
            # a limit is kept only when the request writes it; the quoted words may hold the true figure instead
            quote = str(figures.get(key) or "")
            discovered = []
            if concordant_clause(quote, text):
                discovered = [number for number in penned_statistics(quote) if number in amidst_prose]
            kept = None
            if len(discovered) == 1:
                solely = discovered[0]
                kept = int(solely) if solely == int(solely) else solely
            browsing[key] = kept
            drive.diary.append({"kind": "read_check", "field": key, "read": value, "kept": kept})
        steps = []
        for step in browsing.get("procedures") or []:
            if isinstance(step, str):
                step = {"quote": step}
            if not isinstance(step, dict):
                continue
            clause_alt = concordant_clause(str(step.get("quote") or ""), text)
            if clause_alt:
                steps.append({"quote": clause_alt, "needs_yes": step.get("needs_yes") is not False,
                              "final_cart": step.get("final_cart") is not False})
            else:
                drive.diary.append({"kind": "read_check", "field": "procedure", "dropped": str(step.get("quote"))[:200]})
        rules = [str(policy) for policy in browsing.get("rules") or [] if policy]
        if rules and not steps:
            steps = self.phases_amidst_policies(rules)
        phase_vocables = [set(bare_vocables(step["quote"])) for step in steps]

        def reverbs_phase(item):
            for words in phase_vocables:
                if not words:
                    continue
                personal = set(bare_vocables(item))
                if len(words & personal) >= 0.6 * max(1, len(personal)):
                    return True
            return False
        # a go-ahead that shows up both as a step and as an early question is asked only as the step, when it is due
        browsing["ask_first"] = [item for item in browsing.get("ask_first") or [] if not reverbs_phase(item)]
        browsing["procedures"] = steps
        drive.procedures = steps
        names = [" ".join(bare_vocables(row.get("product"))) for row in browsing.get("items") or [] if isinstance(row, dict)]

        def niche_relating(name):
            words = " ".join(bare_vocables(name))
            loci = [locus for locus, item in enumerate(names) if item and (item == words or item in words or words in item)]
            return loci[0] if len(loci) == 1 else None
        pairs = []
        for duo in browsing.get("fit_with") or []:
            if not isinstance(duo, dict) or not concordant_clause(str(duo.get("quote") or ""), text):
                continue
            dpd, anchor = niche_relating(duo.get("product")), niche_relating(duo.get("with"))
            if dpd is None or anchor is None or dpd == anchor or (dpd, anchor) in pairs:
                continue
            pairs.append((dpd, anchor))
        drive.pairs = pairs
        drive.diary.append({"kind": "pairs", "pairs": pairs})
        drive.diary.append({"kind": "procedures", "steps": steps})

    def phases_amidst_policies(self, rules):
        drive = self.drive
        tallied = "\n".join(f"{locus}. {policy}" for locus, policy in enumerate(rules))
        answer = drive.channel.hunttext_oracle([{"role": "user", "content": PHASE_VALIDATE + "\n\nRULES:\n" + tallied}], MONITOR, 300)
        preferred = (blob_thing((answer or {}).get("content") or "") or {}).get("steps") or []
        steps = []
        for locus in preferred:
            if not (isinstance(locus, int) and 0 <= locus < len(rules)):
                continue
            clause_alt = concordant_clause(rules[locus], drive.request)
            if clause_alt:
                steps.append({"quote": clause_alt, "needs_yes": True, "final_cart": True})
        drive.diary.append({"kind": "rule_steps", "rules": len(rules), "steps": steps})
        return steps

    def solicit_browsers(self, components):
        channel = self.drive.channel
        ladder = BROWSER + MONITOR[:1]
        browsing = {}
        for stint, model in enumerate(ladder * 3):
            # after one full pass, stop retrying once no reader can still answer
            if stint >= len(ladder) and self.browsers_departed(ladder):
                break
            answer = channel.hunttext_oracle([{"role": "user", "content": components}], [model], 3000)
            browsing = blob_thing((answer or {}).get("content") or "") or {}
            if isinstance(browsing.get("roles"), dict):
                break
            channel.diary.append({"kind": "setup_retry", "model": model})
        return browsing

    def browsers_departed(self, ladder):
        channel = self.drive.channel
        if channel.misses == 0 or channel.trigger_falter == "refused" or channel.downward():
            return True
        return not channel.workable(ladder) or channel.extra_alt() < SEALING + 60

    def browsing_through_routine(self, incomplete):
        channel = self.drive.channel
        try:
            remainder = {key: vlu for key, vlu in incomplete.items() if key != "roles"}
            browsing = {**self.drive.sealer.code_reading(), **remainder}
        except Exception as failure:
            channel.diary.append({"kind": "code_reading_error", "error": f"{type(failure).__name__}: {failure}"[:300]})
            browsing = {}
        self.browse_upon = _TALLY.now()
        channel.diary.append({"kind": "code_reading", "roles": browsing.get("roles"), "product": browsing.get("product"),
                         "price_cap": browsing.get("price_cap"), "items": len(browsing.get("items") or [])})
        return browsing

    def utilize_browsing(self, browsing, afresh=False):
        drive = self.drive
        needs, channel = drive.needs, drive.channel
        if not afresh:
            self.phases_provided = drive.phase_top
        antecedent_buggy = drive.role("cart")
        roles = browsing.get("roles")
        if not isinstance(roles, dict):
            roles = {}
        drive.function_atlas = {name: tool for name, tool in roles.items() if isinstance(tool, str) and tool in drive.contours}
        if afresh and drive.wagon.utilized:
            # a cart already in use stays bound to the tool that holds it
            if antecedent_buggy:
                drive.function_atlas["cart"] = antecedent_buggy
        else:
            drive.wagon = BuggyTome(drive.role("cart"), drive.contours.get(drive.role("cart") or "", {}))
        needs.combable = bool(drive.role("search")) and drive.observe.folio_operand() is not None
        budget = browsing.get("message_budget")
        if be_numeral(budget) and budget >= 0:
            drive.txt_top = int(budget)
        needs.browsing = browsing
        try:
            channel.factor = 1.0 + 0.3 * min(2, max(0, len(needs.items()) - 1))
        except Exception:
            channel.factor = 1.0
        self.divide_prerequisites(browsing)
        self.shielded(lambda: self.story_values(browsing), "story_values_error")
        self.shielded(drive.citecounter.cite_enlist, "ref_register_error")
        if not afresh or needs.edition <= 0:
            needs.disburse_top = browsing.get("price_cap")
        if browsing and needs.disburse_top is None and drive.role("message"):
            afterwards = [str(item) for item in browsing.get("ask_first") or [] if item]
            browsing["ask_first"] = ["the highest amount the shopper is willing to spend, as the request gives no price limit (skip it if this same point appears again here)"] + afterwards
        if afresh and self.phases_provided is not None:
            drive.phase_top = self.phases_provided
        permitted_alt = browsing.get("step_budget")
        if be_numeral(permitted_alt) and 0 < permitted_alt < drive.phase_top:
            drive.phase_top = int(permitted_alt)
        if afresh:
            needs.edition += 1

    def divide_prerequisites(self, browsing):
        needs = self.drive.needs
        named = [str(req) for req in browsing.get("requirements") or [] if req]

        def nicely_fashioned(req):
            head, punct, value = req.partition(":")
            return bool(punct and head.strip() and value.strip())
        needs.written = [req for req in named if nicely_fashioned(req)]
        errant = [req for req in named if req not in needs.written]
        if errant:
            # anything without a "property: value" shape is only a wish
            browsing["soft_wants"] = [str(wq) for wq in browsing.get("soft_wants") or [] if wq] + errant
            browsing["requirements"] = list(needs.written)

    def shielded(self, action, kind):
        try:
            action()
        except Exception as failure:
            self.drive.channel.diary.append({"kind": kind, "error": f"{type(failure).__name__}: {failure}"[:300]})

    @staticmethod
    def cue_lineup(crude):
        if isinstance(crude, dict):
            even = []
            for key, vlus in crude.items():
                for vlu in (vlus if isinstance(vlus, list) else [vlus]):
                    if not vlu:
                        continue
                    even.append(f"{vlu}" if ":" in str(vlu) else f"{key}: {vlu}")
            crude = even
        if not isinstance(crude, (list, tuple)):
            return []
        return [str(f_alt).strip() for f_alt in crude if isinstance(f_alt, (str, int, float)) and str(f_alt).strip()]

    @staticmethod
    def dem_kernel(req):
        head, _, value = str(req).partition(":")
        return re.split(r"\s*\(", head)[0].strip().lower(), value.strip().lower()

    def story_values(self, browsing):
        drive = self.drive
        needs = drive.needs
        hints = [mono_feat_cue(f_alt) for f_alt in self.cue_lineup(browsing.get("hints"))]
        fromsrc_browser = list(hints)
        provided = browsing.get("references")
        refs = [str(rq) for rq in provided or []] if isinstance(provided, list) else []
        inferred = [req for req in needs.written if tale_conjecture(req, drive.request, refs)]
        # values only the story implies turn into hints, unless that would empty most of the requirements
        if inferred and len(needs.written) >= 2 and 2 * len(inferred) <= len(needs.written):
            needs.written = [req for req in needs.written if req not in inferred]
            browsing["requirements"] = list(needs.written)
            kernels = {self.dem_kernel(req) for req in inferred}
            for row in browsing.get("items") or []:
                if isinstance(row, dict) and isinstance(row.get("requirements"), list):
                    row["requirements"] = [rq for rq in row["requirements"] if self.dem_kernel(rq) not in kernels]
            for req in inferred:
                prop, _, value = req.partition(":")
                hints.append(f"{prop.strip()}: the story suggests {value.strip()} (not stated as a value; ask)")
            drive.channel.diary.append({"kind": "story_values", "moved": inferred})
        try:
            elevated = self.penned_cues(browsing, fromsrc_browser)
            hints = [f_alt for f_alt in hints if f_alt not in elevated]
        except Exception as failure:
            drive.channel.diary.append({"kind": "written_hint_error", "error": f"{type(failure).__name__}: {failure}"[:300]})
        browsing["hints"] = hints[:12]

    def cue_vocable(self, nuance, names):
        discovered, ultimate = "", ""
        for ultimate in names:
            discovered = guide_measurement(nuance, self.drive.request, ultimate)
            if discovered:
                break
        return discovered, ultimate

    def collisions(self, prop, vocable, minee):
        constrained = r"\b" + re.escape(vocable) + r"\b"
        for req in self.drive.needs.written:
            rcap, _, rhs = req.partition(":")
            tagnames = re.findall(r"\(([^)]*)\)", rcap)
            if tagnames and minee and not (variety_vocables(tagnames[-1]) & minee):
                continue  # a requirement of some other product
            if equal_feat(rcap, prop) or re.search(constrained, rhs.lower()):
                return True
        return False

    def penned_cues(self, browsing, hints):
        drive = self.drive
        needs = drive.needs
        rows = [rq for rq in browsing.get("items") or [] if isinstance(rq, dict)]
        elevated = []
        for hint in hints:
            head, delim, nuance = str(hint).partition(":")
            prop = re.split(r"\s*\(", head)[0].strip()
            if not delim or not prop or trash_selector(prop):
                continue
            tagnames = re.findall(r"\(([^)]*)\)", head)
            names = [tagnames[-1]] if tagnames else ([str(rq.get("product") or "") for rq in rows] or [str(browsing.get("product") or "")])
            vocable, name = self.cue_vocable(nuance, names)
            if not vocable:
                continue
            minee = variety_vocables(name)
            if self.collisions(prop, vocable, minee):
                continue
            # the request itself writes this value: it becomes a requirement
            req = f"{head.strip()}: {vocable}"
            needs.written.append(req)
            browsing["requirements"] = list(needs.written)

            def retainsown(row):
                return isinstance(row.get("requirements"), list) and (len(rows) == 1 or variety_vocables(row.get("product") or "") & minee)
            target = next((row for row in rows if retainsown(row)), None)
            if target is not None:
                target["requirements"].append(req)
            elevated.append(hint)
            drive.channel.diary.append({"kind": "written_hint", "hint": str(hint)[:200], "requirement": req})
        return elevated

    # story hints and the menu of askable properties

    def several(self):
        return bool(self.drive.needs.items())

    def niche_relating(self, text):
        if not self.several():
            return None
        return self.drive.speak.commodity_relating(text)

    def access_cues(self, kx=None):
        speak = self.drive.speak
        unsolicited, exchanged = [], []
        for hint in self.drive.needs.browsing.get("hints") or []:
            prop = str(hint).split(":", 1)[0].strip()
            if not prop:
                continue
            slot = self.niche_relating(prop)
            if kx is not None and slot not in (kx, None):
                continue
            personal = slot if self.several() else None
            if speak.access_feat(personal, prop, self.feat_orderline(slot, prop), through_specimens=True) and not self.solicited_interior(personal, prop):
                unsolicited.append(str(hint))
                continue
            # the hint's first property got nowhere: offer its named alternative instead
            alternate = self.cue_alternate(personal, str(hint), prop)
            if alternate:
                exchanged.append(alternate)
        return exchanged + unsolicited

    def construct_panel(self, kx):
        drive = self.drive
        stash, shelf = drive.stash, drive.shelf
        if not self.several():
            personal = shelf.personal_aisle()
            return (stash.feat_panel(personal) if personal else stash.feat_panel())[0]
        houses = shelf.hub_aisles()
        hub = houses[kx] if kx is not None and kx < len(houses) else set()
        concordant = [name for name in stash.aisles_browse() if name.strip().lower() in hub]
        return stash.feat_panel(concordant[0])[0] if concordant else ""

    def feat_orderline(self, kx, prop):
        stash = self.drive.stash
        try:
            size = len(stash.items)
            jotting = self.panel_jotting.get(kx)
            if jotting and jotting[0] == size:
                menu = jotting[1]
            else:
                menu = self.construct_panel(kx)
                self.panel_jotting[kx] = (len(stash.items), menu)
            return next((line for line in menu.splitlines() if stringent_feat(self.panel_selector(line), prop)), "")
        except Exception:
            return ""

    @staticmethod
    def panel_selector(line):
        return line.strip("- ").split(" (in ")[0].strip()

    def buffered_panel(self, kx):
        jotting = self.panel_jotting.get(kx) or (0, "")
        return jotting[1]

    @staticmethod
    def bare_vocables(value):
        return " ".join(re.findall(r"[a-z0-9]+", value.lower()))

    def column_measurements(self, kx, prop):
        line = self.feat_orderline(kx, prop)
        if "):" not in line:
            return set()
        values = line.split("):", 1)[1].split("|")
        return {self.bare_vocables(vz) for vz in values} - {""}

    def equal_column(self, kx, a, g_alt):
        if stringent_feat(a, g_alt):
            return True
        try:
            if not equal_feat(a, g_alt):
                return False
            # two spellings of one field share at least one listed value
            return bool(self.column_measurements(kx, a) & self.column_measurements(kx, g_alt))
        except Exception:
            return False

    def cued_feat(self, kx, prop):
        name = str(prop).split("(")[0].strip()
        for hint in self.drive.needs.browsing.get("hints") or []:
            head = str(hint).split(":", 1)[0]
            if not stringent_feat(head, name):
                continue
            if kx is None or not self.several():
                return True
            if self.drive.speak.commodity_relating(head) in (kx, None):
                return True
        return False

    def commodity_browses(self, kx):
        drive = self.drive
        stash = drive.stash
        try:
            if self.several():
                houses = drive.shelf.hub_aisles()
                hub = houses[kx] if kx is not None and kx < len(houses) else set()
                return sum(stash.elaborate_browses(name) for name in hub) if hub else 0
            personal = drive.shelf.personal_aisle()
            return stash.elaborate_browses(personal) if personal else stash.elaborate_browses()
        except Exception:
            return 0

    def bare_cue(self, kx, prop, cued=False):
        try:
            name = re.split(r"\s*\(", str(prop))[0].strip()
            if not name or not (cued or self.cued_feat(kx, name)):
                return False
            if self.feat_orderline(kx, name) or self.commodity_browses(kx) < 2:
                return False
            # enough listings read and still no menu key like it: the hint's property has no field here
            return not any(equal_feat(self.panel_selector(line), name) for line in self.buffered_panel(kx).splitlines())
        except Exception:
            return False

    def cue_stamp(self, hint):
        try:
            prop = str(hint).split(":", 1)[0].strip()
            if self.bare_cue(self.niche_relating(prop), prop, cued=True):
                return f"{hint} [no key]"
            return str(hint)
        except Exception:
            return str(hint)

    @staticmethod
    def cue_alternates_relating(nuance):
        alts = []
        for group in re.findall(r"\(or ([^()]+)\)", str(nuance)):
            fragments = (shard.strip() for shard in re.split(r",|\bor\b|/", group))
            alts.extend(shard for shard in fragments if shard)
        return alts

    @staticmethod
    def extended_vocables(name):
        return {vocable for vocable in feat_vocables(name) if len(vocable) >= 3 and vocable not in ANCHOR_VOCABLES}

    @staticmethod
    def designation_interior(a, g_alt):
        left = Admission.extended_vocables(a)
        correct = Admission.extended_vocables(g_alt)
        return bool(left and correct) and (left <= correct or correct <= left)

    @staticmethod
    def solicited_prose(line):
        return re.search(r'asked "(.*?)" -> ', line)

    def solicited_interior(self, kx, head):
        speak = self.drive.speak
        try:
            several = self.several()
            for line in self.drive.questions_log:
                label = line.split(":", 1)[0].strip("- ")
                if "[not answered]" in label:
                    continue
                uttered = self.solicited_prose(line)
                for component in label.split(";"):
                    slot = speak.commodity_relating(component) if several else None
                    if several and slot is None and uttered:
                        slot = speak.commodity_relating(uttered.group(1))
                    if several and kx is not None and slot not in (kx, None):
                        continue
                    if self.designation_interior(component, head):
                        return True
        except Exception:
            return False
        return False

    def still_onto(self, kx, head):
        several = self.several()
        speak = self.drive.speak
        for solicited in speak.still_solicits or []:
            if self.equal_column(kx, solicited, head) or self.designation_interior(solicited, head):
                if not several or speak.commodity_relating(solicited) in (kx, None):
                    return True
        return False

    def cue_alternate(self, kx, hint, head):
        try:
            jotting = self.cue_alternates
            if not jotting.get(hint):
                if not self.still_onto(kx, head):
                    return ""
                # remember the first alternative the menu really carries
                for alternate in self.cue_alternates_relating(hint.partition(":")[2]):
                    if trash_selector(alternate) or stringent_feat(alternate, head):
                        continue
                    line = self.feat_orderline(kx, alternate)
                    if line:
                        jotting[hint] = self.panel_selector(line)
                        break
            key = jotting.get(hint) or ""
            if not key or key.lower() in self.drive.speak.disbursed_solicits or not self.strategy_access(kx, key):
                return ""
            fore, _, remainder = hint.partition(":")
            tag = re.search(r"\(([^()]*)\)\s*$", fore)
            nuance = re.sub(r"\s*\(or [^()]+\)", "", remainder).strip()
            label = f"{key} ({tag.group(1).strip()})" if tag else key
            return f"{label}: {nuance} (its first property, {head}, drew no preference)"
        except Exception:
            return ""

    def cue_charted(self, kx, prop):
        try:
            name = re.split(r"\s*\(", str(prop))[0].strip()
            words = [wq for wq in feat_vocables(name) if len(wq) >= 3 and wq not in ANCHOR_VOCABLES]
            several = self.several()
            speak = self.drive.speak
            for hint in self.drive.needs.browsing.get("hints") or []:
                head, _, nuance = str(hint).partition(":")
                if several and kx is not None and speak.commodity_relating(head) not in (kx, None):
                    continue
                names = [re.split(r"\s*\(", head)[0].strip()] + self.cue_alternates_relating(nuance)
                if any(n and stringent_feat(n, name) for n in names):
                    return True
                reduced = str(hint).lower()
                if words and all(re.search(r"\b" + re.escape(wq), reduced) for wq in words):
                    return True
        except Exception:
            return False
        return False

    @staticmethod
    def designation_duplicate(a, g_alt):
        if stringent_feat(a, g_alt):
            return False

        def vocable_collections(name):
            brimming = [wq for wq in feat_vocables(name) if len(wq) >= 3 and wq not in ANCHOR_VOCABLES]
            return brimming, (brimming[:-1] if len(brimming) > 1 else brimming)

        def nigh(xq, y):
            return xq == y or (min(len(xq), len(y)) >= 4 and (xq.startswith(y) or y.startswith(xq)))
        brimming_u, front_u = vocable_collections(a)
        brimming_g, front_g = vocable_collections(g_alt)
        # a leading word of either name close to any word of the other
        if any(nigh(xq, y) for xq in front_u for y in brimming_g):
            return True
        return any(nigh(xq, y) for xq in brimming_u for y in front_g)

    def duplicate_relating(self, kx, a, g_alt):
        try:
            if stringent_feat(a, g_alt):
                return False
            return self.designation_duplicate(a, g_alt) or self.equal_column(kx, a, g_alt)
        except Exception:
            return False

    def tale_prose(self, kx):
        drive = self.drive
        several = self.several()
        proses = [drive.request.lower()]
        for hint in drive.needs.browsing.get("hints") or []:
            if several and drive.speak.commodity_relating(str(hint).split(":", 1)[0]) not in (kx, None):
                continue
            proses.append(str(hint).lower())
        return " ".join(proses)

    def commodity_designation(self, kx, failsafe=""):
        needs = self.drive.needs
        if kx is not None and self.several():
            return needs.items()[kx]["product"]
        return str(needs.browsing.get("product") or failsafe)

    def duplicate_selector(self, kx, prop):
        try:
            self.feat_orderline(kx, prop)
            menu = self.buffered_panel(kx)
            personal = variety_morphemes(self.commodity_designation(kx))
            story = self.tale_prose(kx)
            disbursed = self.drive.speak.disbursed_solicits
            minee = set(feat_vocables(prop))

            def novel_vocables(key):
                return [wq for wq in feat_vocables(key)
                        if len(wq) >= 3 and wq not in ANCHOR_VOCABLES and wq not in minee and not variety_amidst(wq, personal)]
            for line in menu.splitlines():
                key = self.panel_selector(line)
                if not key or trash_selector(key) or key.lower() in disbursed:
                    continue
                if not self.duplicate_relating(kx, key, prop) or not self.strategy_access(kx, key) or self.designation_interior(key, prop):
                    continue
                # the near-twin key must be backed by the story or a hint
                if any(re.search(r"\b" + re.escape(wq), story) for wq in novel_vocables(key)) or self.cue_charted(kx, key):
                    return key
        except Exception:
            return ""
        return ""

    def measurements_mould(self, kx, prop, product):
        try:
            line = self.feat_orderline(kx, prop)
            if "):" not in line:
                return ""
            personal_orderline = line.strip()
            otherwhere = set()
            for further in self.buffered_panel(kx).splitlines():
                if "):" in further and further.strip() != personal_orderline:
                    otherwhere.update(self.bare_vocables(vz) for vz in further.split("):", 1)[1].split("|"))
            picks = []
            for crude in line.split("):", 1)[1].split("|"):
                value = crude.strip()
                if not value or len(value) > 30 or not re.search(r"[a-z]", value.lower()) or re.search(r"[,;/]", value):
                    continue
                bare_alt = self.bare_vocables(value)
                if bare_alt in otherwhere or bare_alt in [self.bare_vocables(pz) for pz in picks]:
                    continue
                picks.append(value)
            picks = picks[:3]
            if len(picks) < 2:
                return ""
            return f"For the {product}, would you like {', '.join(picks[:-1])} or {picks[-1]}?"
        except Exception:
            return ""

    def duplicate_destination(self, prop):
        if not self.several():
            return (None, -1) if -1 not in self.duplicate_solicits else None
        slot = self.drive.speak.commodity_relating(prop)
        if slot is None or slot in self.duplicate_solicits:
            return None
        return slot, slot

    def captioned(self, key, slot):
        if self.several():
            return f"{key} ({self.drive.needs.items()[slot]['product']})"
        return key

    def separate_solicit(self, prop):
        try:
            target = self.duplicate_destination(prop)
            if target is None:
                return None
            slot, stamp_alt = target
            several = self.several()
            product = self.commodity_designation(slot, "item") if several else str(self.drive.needs.browsing.get("product") or "item")
            field = re.split(r"\s*\(", str(prop))[0].strip()
            text = self.measurements_mould(slot, field, product)
            if not text:
                duplicate = self.duplicate_selector(slot, prop)
                if duplicate:
                    field = duplicate
                    text = self.measurements_mould(slot, duplicate, product) or f"Which {duplicate} would you like for the {product}?"
            if not text:
                return None
            self.duplicate_solicits.add(stamp_alt)
            return (f"{field} ({product})" if self.several() else field), text
        except Exception:
            return None

    def solicited_towards(self, kx, prop):
        speak = self.drive.speak
        try:
            several = self.several()
            name = re.split(r"\s*\(", str(prop or ""))[0].strip()
            if not name:
                return False
            for line in self.drive.questions_log:
                label = re.sub(r"\s*\[[^\]]*\]", "", line.split(":", 1)[0].strip("- "))
                uttered = self.solicited_prose(line)
                for component in (pz.strip() for pz in re.split(r";|\band\b", label)):
                    if not component:
                        continue
                    slot = speak.commodity_relating(component) if several else None
                    if several and slot is None and uttered:
                        slot = speak.commodity_relating(uttered.group(1))
                    if several and kx is not None and slot != kx:
                        continue
                    if stringent_feat(re.split(r"\s*\(", component)[0].strip(), name):
                        return True
        except Exception:
            return False
        return False

    def alternate_workable(self, kx, alternate, head, name, disbursed):
        if trash_selector(alternate) or stringent_feat(alternate, head) or stringent_feat(alternate, name) or self.designation_interior(alternate, name):
            return ""
        line = self.feat_orderline(kx, alternate)
        key = self.panel_selector(line) if line else ""
        if not key or stringent_feat(key, name) or key.lower() in disbursed:
            return ""
        return "" if self.solicited_towards(kx, key) or not self.strategy_access(kx, key) else key

    def alternate_selector(self, kx, prop):
        try:
            name = re.split(r"\s*\(", str(prop))[0].strip()
            if not name:
                return ""
            several = self.several()
            speak = self.drive.speak
            disbursed = speak.disbursed_solicits
            for hint in self.drive.needs.browsing.get("hints") or []:
                brimming_front, _, nuance = str(hint).partition(":")
                head = re.split(r"\s*\(", brimming_front)[0].strip()
                if not head or (several and speak.commodity_relating(brimming_front) not in (kx, None)):
                    continue
                if not (self.equal_column(kx, name, head) or self.designation_interior(name, head)):
                    continue
                for alternate in self.cue_alternates_relating(nuance):
                    key = self.alternate_workable(kx, alternate, head, name, disbursed)
                    if key:
                        return key
        except Exception:
            return ""
        return ""

    def intersect_alternate(self, prop):
        try:
            target = self.duplicate_destination(prop)
            if target is None:
                return ""
            slot, stamp_alt = target
            key = self.alternate_selector(slot, prop)
            if not key:
                return ""
            self.duplicate_solicits.add(stamp_alt)
            return self.captioned(key, slot)
        except Exception:
            return ""

    def maintains_bare(self, text, bare_w):
        root_alt = self.drive.observe.root_alt
        padding = {"which", "would", "you", "like", "for", "the"}
        extant = {root_alt(wq) for wq in re.findall(r"[a-z0-9]+", str(text).lower())}
        wished_alt = [root_alt(wq) for wq in re.findall(r"[a-z0-9]+", str(bare_w).lower()) if wq not in padding and wq not in ANCHOR_VOCABLES]
        return bool(wished_alt) and all(wq in extant for wq in wished_alt)

    def alternate_untested(self, kx):
        speak = self.drive.speak
        try:
            for hint in self.drive.needs.browsing.get("hints") or []:
                brimming_front = str(hint).split(":", 1)[0]
                if speak.commodity_relating(brimming_front) not in (kx, None):
                    continue
                head = re.split(r"\s*\(", brimming_front)[0].strip()
                if head and self.solicited_towards(kx, head) and self.alternate_selector(kx, head):
                    return True
            return False
        except Exception:
            return False

    def rest_unsolicited(self, kx):
        speak = self.drive.speak
        diary = self.drive.questions_log
        attained = {speak.commodity_relating(line.split(":", 1)[0]) for line in diary}
        for line in diary:
            uttered = self.solicited_prose(line)
            if uttered:
                attained.add(speak.commodity_relating(uttered.group(1)))
        return any(s_alt != kx and s_alt not in attained for s_alt in range(len(self.drive.needs.items())))

    def still_resolicit(self, prop):
        try:
            target = self.duplicate_destination(prop)
            if target is None:
                return ""
            slot, stamp_alt = target
            if not self.cue_charted(slot, prop):
                return ""
            # a reopened hint or a product nobody asked about yet goes first
            if any(" drew no preference)" in f_alt for f_alt in self.access_cues(slot)):
                return ""
            if self.several() and self.rest_unsolicited(slot):
                return ""
            duplicate = self.duplicate_selector(slot, prop)
            if not duplicate:
                return ""
            self.duplicate_solicits.add(stamp_alt)
            return self.captioned(duplicate, slot)
        except Exception:
            return ""

    @staticmethod
    def panel_shorten(menu, browses):
        if not menu or browses < 4:
            return menu
        # with enough listings read, keys seen in at most one listing are noise
        kept = [line for line in menu.splitlines() if not re.search(r"\(in [01] listings?\)", line)]
        return "\n".join(kept) if kept else menu

    def first_reads(self):
        drive = self.drive
        try:
            if self.primary_browses_accomplished or drive.shelf.lot_utility() or not drive.role("details"):
                return False
            self.primary_browses_accomplished = True
            rows = drive.needs.items()
            houses = drive.shelf.hub_aisles()
            censuses = [row["quantity"] for row in rows] or [drive.needs.quantity()]
            room = drive.maneuvers_residual() - self.checkout_expected() - 2 * max(0, drive.solicit_constraint() - drive.txts_delivered) - 8
            wished_alt, occupied = [], []
            for kx in range(len(rows) or 1):
                hub = houses[kx] if kx < len(houses) else set()
                possess = self.commodity_browses(kx if rows else None)
                if not hub or possess >= 2 or room <= 0:
                    continue
                picks = drive.shelf.panel_datarows(hub, 2 - possess, occupied, censuses[min(kx, len(censuses) - 1)])
                occupied += picks
                if picks:
                    wished_alt.append((possess, kx, picks))
            ranked = sorted(wished_alt)
            # round robin: every product's first listing, then every product's second
            waitlist = [picks[n] for n in range(2) for _, _, picks in ranked if n < len(picks)]
            waitlist = waitlist[: max(0, room)]
            if not waitlist:
                return False
            browses = drive.browses
            browses.access_waitlist = waitlist
            browses.access_revalidate = True
            browses.access_utter = "Before asking anything, I open a few listings of every requested product to see which fields they have."
            browses.access_intention = "menu"
            drive.diary.append({"kind": "first_reads", "ids": waitlist})
            return True
        except Exception as failure:
            drive.diary.append({"kind": "first_reads_error", "error": f"{type(failure).__name__}: {failure}"[:300]})
            return False

    def strategy_access(self, kx, prop):
        speak = self.drive.speak
        try:
            cued = self.cued_feat(kx, prop)
            return speak.access_feat(kx, prop, self.feat_orderline(kx, prop) if cued else "", through_specimens=cued)
        except Exception:
            return speak.access_feat(kx, prop)

    def concludes(self, req, samples=""):
        drive = self.drive
        head, _, value = str(req).partition(":")
        minor = value.lower()
        if re.search(r"as shown|same as|the page|sold by|redirect|refer", minor) or cite_prose(minor):
            return True
        if not favorable_measurement(value, head):
            return False
        provided = drive.needs.browsing.get("references")
        refs = [str(rq) for rq in provided or []] if isinstance(provided, list) else []
        refs += list(drive.needs.resolved or [])
        if tale_conjecture(req, drive.request + " " + " ".join(drive.needs.interchanges), refs):
            return False
        if not samples or re.search(r"\d", value):
            return True
        words = [wq for wq in re.findall(r"[a-z]{3,}", minor) if wq not in ANCHOR_VOCABLES]
        if not words:
            return True
        # a value word must show up among the sample values (allowing a plural or a one-letter slip)
        cluster = set(re.findall(r"[a-z]+", samples))
        for vocable in words:
            root_component = vocable[:-1] if len(vocable) > 4 else vocable
            if re.search(r"\b" + re.escape(root_component), samples):
                return True
            if len(vocable) >= 4 and any(len(s) >= 4 and amendments_amongst(s, vocable, 1) for s in cluster):
                return True
        return False

    # which tools the actor is offered

    def lively(self):
        drive = self.drive
        if drive.event_seen:
            return True
        dead = drive.shelf.dead
        return bool(dead and set(drive.wagon.metrics) & dead)

    def checkout_expected(self):
        cap = self.drive.phase_top
        floor = max(PRIMARY_CHECKOUT_THROUGH, min(PRIMARY_CHECKOUT_EXPECTED, cap // 4 + 1))
        expected_alt = max(floor, min(16, cap // 3))
        if self.lively():
            try:
                surplus = 8 + max(1, len(self.drive.needs.items()))
            except Exception:
                surplus = 9
            expected_alt = max(expected_alt, surplus)
        return max(floor, min(expected_alt, cap // 2))

    def shrinking(self):
        role = self.drive.role
        return bool(role("order") and role("cart"))

    def stockpile_designations(self, extra_alt):
        drive = self.drive
        role = drive.role
        order, decline, cart = role("order"), role("decline"), role("cart")
        if extra_alt <= 1:
            return {order, decline}
        if extra_alt <= 3:
            return {order, decline, cart}
        primary_stockpile = extra_alt <= self.checkout_expected() and drive.orders_tried == 0
        if not primary_stockpile and not (extra_alt <= PRIMARY_CHECKOUT_THROUGH and drive.refusal_n > 0):
            return None
        names = {order, decline, cart, role("cart_view"), role("view") or role("details"), drive.shelf.supply_utility()}
        if drive.shelf.switch_access():
            names.add(role("details"))
        if drive.late_n < 3:
            # a few late reads stay possible while nothing is verified or a check named a listing
            if (self.unvalidated() and extra_alt > 5) or (self.designated_access() and extra_alt >= min(PRIMARY_CHECKOUT_THROUGH + 2, self.checkout_expected())):
                names.add(role("details"))
            if (self.expanded_tardy() or drive.referee.designated_folios or (self.unvalidated() and extra_alt > 5)) and extra_alt > 4:
                names.add(role("search"))
        if extra_alt >= 7 and self.standby_datarows():
            names.add(role("details"))
        return names

    def tendered(self):
        drive = self.drive
        names = self.stockpile_designations(drive.maneuvers_residual())
        tools = drive.utilitykit
        if names is not None and drive.role("order") and drive.role("cart"):
            tools = [element_alt for element_alt in drive.utilitykit if element_alt["function"]["name"] in names] or tools
        courier = drive.role("message")
        if drive.soliciting and courier:
            tools = [element_alt for element_alt in tools if element_alt["function"]["name"] != courier]
        return tools

    def amidst_stockpile(self):
        drive = self.drive
        extra_alt = drive.maneuvers_residual()
        if not (drive.role("order") and drive.role("cart")) or extra_alt <= 3:
            return False
        if drive.orders_tried == 0 and extra_alt <= self.checkout_expected():
            return True
        return extra_alt <= PRIMARY_CHECKOUT_THROUGH and drive.refusal_n > 0

    def unvalidated(self):
        drive = self.drive
        dead = drive.shelf.inert_orderlines()
        edition = str(drive.needs.edition)
        ongoing = []
        for key in drive.passed:
            line, _, seal = key.rpartition("|")
            if seal == edition and line not in drive.refused_lines and line not in dead:
                ongoing.append(line)
        if not ongoing:
            return True
        if not self.several():
            return False
        try:
            # with several products, one whose shelves hold none of the verified lines is still unverified
            shelves = {drive.referee.aisle_recognized(ii) for ii in ongoing} - {""}
            return bool(shelves) and any(hub and not hub & shelves for hub in drive.shelf.hub_aisles())
        except Exception:
            return False

    def designated_access(self):
        drive = self.drive
        itemized = {ii for ii in drive.referee.named if ii in drive.stash.ids}
        return itemized - drive.stash.read - drive.refused_lines

    def expanded_tardy(self):
        drive = self.drive
        needs = drive.needs
        if needs.expanded_upon != needs.edition:
            return False
        return drive.channel.beat - needs.expanded_beat <= self.checkout_expected()

    def start_lot(self):
        self.tardy_w = 0
        self.tardy_idents = set()

    def extra_standbys(self):
        left = 2 - self.standby_k
        crisp = [ii for ii in self.standby_datarows() if ii not in self.tardy_idents]
        return crisp[:left]

    def tardy_permitted(self, trigger_designation, trigger_operands, named, extra_alt):
        drive = self.drive
        if trigger_designation == drive.role("details"):
            if self.unvalidated() and extra_alt > 5:
                return True
            return bool(named & self.designated_access()) and extra_alt >= min(PRIMARY_CHECKOUT_THROUGH + 2, self.checkout_expected())
        expanded = self.expanded_tardy() or json.dumps(trigger_operands, sort_keys=True, default=str) in drive.referee.designated_folios
        return (expanded or (self.unvalidated() and extra_alt > 5)) and extra_alt > 4

    def tardy_maintain(self, trigger_designation, trigger_operands):
        drive = self.drive
        details, search = drive.role("details"), drive.role("search")
        if trigger_designation not in (details, search):
            # another lookup tool is judged like the details tool
            if trigger_designation in drive.shelf.fetch_utilities() and details:
                return self.tardy_maintain(details, trigger_operands)
            return ""
        if not self.amidst_stockpile():
            return ""
        named = {vlu for vlu in trigger_operands.values() if isinstance(vlu, str)}
        if trigger_designation == details:
            personal = set(drive.wagon.metrics) | {key.rpartition("|")[0] for key in drive.passed}
            if named & personal:
                return ""
        waitlisted = self.tardy_w
        extra_alt = drive.maneuvers_residual() - waitlisted
        permitted_alt = drive.late_n + waitlisted < 3 and self.tardy_permitted(trigger_designation, trigger_operands, named, extra_alt)
        standby_space = trigger_designation == details and extra_alt - 1 >= 6
        if not permitted_alt and standby_space and self.standby_k < 2 and named & set(self.extra_standbys()):
            permitted_alt = True
            self.standby_k += 1
        if not permitted_alt and trigger_designation == details:
            opened = drive.shelf.switch_access()
            if opened and opened in named and extra_alt > 2:
                permitted_alt = True
                self.switch_accomplished = True
        if permitted_alt:
            self.tardy_w = waitlisted + 1
            self.tardy_idents.update(named)
            return ""
        standbys = self.extra_standbys() if standby_space and self.standby_k < 2 else []
        note = ("Not run (no step used): the steps left are kept for the order. Details can only be read for the cart's "
                "line, a listing a check named, or (while nothing is verified) at most 3 listings overall: add your best "
                "verified fit to the cart, check its stock and price once more, and order.")
        if standbys:
            note += " Cheaper unread rows that could back up the pick may still be read: " + ", ".join(standbys) + "."
        return note

    def standby_datarows(self):
        drive = self.drive
        if drive.needs.priority() != "cheapest" or self.standby_k >= 2:
            return []
        if not (drive.event_seen or drive.shelf.inert_orderlines() or self.designated_access()):
            return []
        discovered = []
        try:
            for kx in drive.encompass.gx_commodities():
                cluster = drive.shelf.dx_cluster(kx)
                befit_rates = [y_alt["price"] for y_alt in cluster if y_alt["verified"] and y_alt["fit"] and y_alt["price"] is not None]
                if not befit_rates:
                    continue
                ceiling = min(befit_rates)
                for element_alt in cluster:
                    if element_alt["read"] or element_alt["gone"] or element_alt["over"] or element_alt["pointer"] or element_alt["ruled"] or element_alt["clash"]:
                        continue
                    if element_alt["price"] is not None and element_alt["price"] < ceiling:
                        discovered.append((element_alt["price"], element_alt["id"]))
        except Exception as failure:
            drive.diary.append({"kind": "backup_error", "error": f"{type(failure).__name__}: {failure}"[:300]})
            return []
        return [ii for _, ii in sorted(discovered)][:6]


class ShoppingRun:
    def __init__(self, pdata):
        env = pdata["environment"]
        self.channel = Channel(env["binding"])
        self.plook = env["policy_view"]
        self.request = str(self.plook.get("query") or "")
        self.utilitykit = [tdecl for tdecl in (self.plook.get("tools") or [])
                        if isinstance(tdecl, dict) and isinstance(tdecl.get("function"), dict)]
        self.contours = {}
        for tdecl in self.utilitykit:
            cb = tdecl["function"]
            self.contours[cb["name"]] = cb.get("parameters") or {"type": "object", "properties": {}}
        self.phase_top = int(self.plook.get("max_steps") or 30)
        self.channel.done = bool(self.plook.get("finished"))
        self.stash = Transcripts()
        self.needs = Necessities(self.channel, self.request)
        self.function_atlas = {}
        self.wagon = BuggyTome(None, {})
        self.access_logbooks()
        EXACT_MEASUREMENTS.clear()
        self.browses = BrowseWaitlist()
        # the helper desks all hold a back-reference to this run; they are built in a fixed order
        self.admission = Admission(self)
        self.referee = Referee(self)
        self.shelf = Directory(self)
        self.speak = Colloquy(self)
        self.sealer = RoutineSealer(self)
        self.observe = Spectator(self)
        self.walk = InexpensiveRoute(self)
        self.citecounter = CitationDesk(self)
        self.encompass = Breadth(self)
        self.needs.cite_callback = self.citecounter.cite_patch
        self.needs.column_callback = self.referee.warehouse_columns

    def access_logbooks(self):
        # message and order bookkeeping
        self.txts_delivered = 0
        self.txt_top = None
        self.procedures: list[dict[str, Any]] = []
        self.affirms = 0
        self.affirmed_buggy: Any = None
        self.affirming = False
        self.advance_decision = ""
        self.advance_maintain_stamp: Any = None
        self.pairs: list[tuple[int, int]] = []
        self.duo_notified: dict[Any, int] = {}
        for tallier in ("orders_tried", "refusal_n", "stuck", "absorb_n", "cart_seq", "late_n"):
            setattr(self, tallier, 0)
        self.starting_buggy = None
        self.soliciting = True
        for toggle in ("order_vetted", "decline_vetted", "audit_done"):
            setattr(self, toggle, False)
        # one-shot markers, unset until the matching hold or check has fired
        for indicator in ("event_seen", "covering", "last_kind", "best_held", "early_audit", "recheck_mark",
                       "ref_decline_held", "slack_done", "decline_fill", "decline_kept", "decline_opens",
                       "batch_cart", "batch_proven", "batch_spec"):
            setattr(self, indicator, None)
        self.lot_brief_recognized = False
        # memory of checks made, refusals met and notes already given
        for sack in ("rank_checked", "checked_adds", "passed", "refused_lines", "warned", "view_told"):
            setattr(self, sack, set())
        for tome in ("rulings", "refusals_seen", "pass_price", "page_book", "cheaper_told", "dead_told",
                     "pointer_told", "line_read", "line_seq", "puts"):
            setattr(self, tome, {})
        for sqn in ("questions_log", "last_calls", "batch_puts", "ref_hits", "chat", "refs"):
            setattr(self, sqn, [])
        # what has been learnt about the way this store's keyword search treats a query
        self.vocable_noticed = []
        self.vocable_variety = ""
        self.guide_noticed = []
        self.guide_edges = {}
        self.guide_vocables = 0
        self.guide_env = None

    @property
    def diary(self):
        return self.channel.diary

    def maneuvers_residual(self):
        return self.phase_top - self.channel.beat

    def role(self, trigger_designation):
        return self.function_atlas.get(trigger_designation)

    def solicit_constraint(self):
        stipend = SOLICIT_BASELINE if self.txt_top is None else self.txt_top
        return max(0, stipend - self.maintained_reverse())

    def maintained_reverse(self):
        # while a stated procedure is still owed, one message per such step is kept out of the question budget
        if self.procedures and not self.affirms:
            stipend = SOLICIT_BASELINE if self.txt_top is None else self.txt_top
            return min(len(self.procedures), stipend)
        return 0

    def buggy_stamp(self):
        return tuple(sorted(duo for duo in self.wagon.metrics.items() if duo[1]))

    def satchel_comment(self):
        rows = self.sealer.routine_datarows()
        logs = []
        total = 0.0
        for lbl, count in sorted(self.wagon.metrics.items()):
            if not count:
                continue
            intel = rows.get(lbl) or {}
            severally = intel.get("price")
            total += (severally or 0.0) * count
            name = str(intel.get("title") or lbl)[:90]
            costed = f" at {severally:g} each" if be_numeral(severally) else ""
            logs.append(f"{count:g} x {name} ({lbl}){costed}")
        cap = self.needs.disburse_top
        concluding = ""
        if total:
            within = f", within your limit of {cap:g}" if be_numeral(cap) and total <= cap else ""
            concluding = f" Total {total:g}{within}."
        return "Final basket: " + "; ".join(logs) + "." + concluding + " May I place this order? Please answer yes or no."

    def go_ahead(self, note, uttered):
        text = ADVANCE_ONWARD + "\n\nTHE ASSISTANT WROTE:\n" + note + "\n\nTHE SHOPPER REPLIED:\n" + uttered
        reply = self.channel.hunttext_oracle([{"role": "user", "content": text}], MONITOR, 200)
        dissected = blob_thing((reply or {}).get("content") or "") or {}
        vocable = str(dissected.get("answer") or "").lower()
        if vocable in ("yes", "no", "change"):
            return vocable
        return "yes"

    def deliver_advance_onward(self, room=3):
        if not (self.procedures and self.role("message") and self.buggy_stamp()):
            return None
        actual = self.buggy_stamp()
        nil_utterance_residual = self.txt_top is not None and self.txts_delivered >= self.txt_top
        if actual == self.affirmed_buggy or nil_utterance_residual:
            return None
        if self.maneuvers_residual() < room:
            return None
        # (with little time left the owed total still goes out - it is a step, not a model call - and only the reading of
        # the shopper's answer is skipped; below a few seconds nothing can be sent safely)
        hurried = self.channel.extra_alt() < ULTIMATE_TRIGGER
        if hurried and self.channel.extra_alt() < 6:
            return None
        note = self.satchel_comment()
        trigger = self.speak.utterance_trigger(note, f"goahead_{self.channel.beat}")
        if not trigger:
            return None

        def capture():
            return self.needs.prose_corpus(), self.needs.disburse_top

        sooner = capture()
        self.affirming = True
        try:
            rejoinders = self.forward([trigger], automated=True)
        finally:
            self.affirming = False
        self.affirms += 1
        self.affirmed_buggy = actual
        uttered = rejoinders[0][1].split("\n[status]")[0] if rejoinders else ""
        verdict = "yes" if hurried else self.go_ahead(note, uttered)
        if capture() != sooner:
            verdict = "change"
        self.advance_decision = verdict
        self.diary.append({"kind": "go_ahead", "note": note, "reply": uttered[:400], "verdict": verdict})
        return note, uttered, verdict

    def duo_tiles(self):
        tiles, _remainder = self.stash.blend_tiles()
        return {lbl: qua_prose(tile) for lbl, tile in tiles.items()}

    def duo_niche(self, line):
        name = str(self.stash.caption_relating.get(line) or "").strip().lower()
        hits = [index for index, hub in enumerate(self.shelf.hub_aisles()) if name and name in hub]
        if len(hits) == 1:
            return hits[0]
        return None

    def duo_stamps(self, line, proses):
        # identifiers of the anchor item itself (its model, other names it is sold under) that its page shows and its shelf
        # does not share widely: once four or more pages of the shelf are known, a mark found on half of them or more is
        # treated as generic and dropped
        label = self.stash.caption_relating.get(line)
        cohort = [text for lbl, text in proses.items() if self.stash.caption_relating.get(lbl) == label]

        def universal(stamp_alt):
            return len(cohort) >= 4 and 2 * sum(1 for text in cohort if stamp_within(stamp_alt, text)) >= len(cohort)

        return {stamp_alt for stamp_alt in ident_stamps(proses.get(line, "")) if not universal(stamp_alt)}

    def duo_checkpoint(self):
        if not self.pairs or self.maneuvers_residual() <= 3 or self.channel.extra_alt() <= SEALING:
            return ""
        niche_orderlines: dict[int, list[str]] = {}
        for line in [lbl for lbl, count in self.wagon.metrics.items() if count]:
            locale = self.duo_niche(line)
            if locale is not None:
                niche_orderlines.setdefault(locale, []).append(line)
        proses = self.duo_tiles()
        products = [str(row["product"]) for row in self.needs.items()]
        dead = self.shelf.inert_orderlines()

        def commodity_designation(index):
            return products[index] if index < len(products) else "item"

        def cost_selector(lbl):
            return self.stash.price(lbl) if be_numeral(self.stash.price(lbl)) else 1e18

        for dpd, anchor in self.pairs:
            for footing in niche_orderlines.get(anchor, []):
                marks = self.duo_stamps(footing, proses)
                if not marks:
                    continue
                for line in niche_orderlines.get(dpd, []):
                    page = proses.get(line, "")
                    if any(stamp_within(stamp_alt, page) for stamp_alt in marks):
                        continue
                    key = (line, footing, self.cart_seq)
                    if self.duo_notified.get(key, 0) >= 1:
                        continue
                    self.duo_notified[key] = 1
                    aisle_caption = self.stash.caption_relating.get(line)
                    named = sorted((lbl for lbl, body in proses.items()
                                    if lbl not in dead and lbl != line and self.stash.caption_relating.get(lbl) == aisle_caption
                                    and any(stamp_within(stamp_alt, body) for stamp_alt in marks)), key=cost_selector)
                    dpd_designation, pivot_designation = commodity_designation(dpd), commodity_designation(anchor)
                    sequenced = sorted(marks)
                    self.diary.append({"kind": "pair_hold", "line": line, "anchor": footing, "marks": sequenced, "named": named[:6]})
                    head = (f"Not run yet (no step used). The {dpd_designation} must be compatible with the {pivot_designation} you chose: the "
                            f"page of {footing} gives {', '.join(sequenced[:2])} as its own, and the page of {line} never mentions it. ")
                    if named:
                        return head + (f"These {dpd_designation} listings do mention it, so open them: {', '.join(named[:6])}. Replace "
                                       f"{line} with the cheapest one that also meets every other requirement, then order.")
                    return head + (f"Run a store search for '{sequenced[0]}' (the search looks through listing text), open the "
                                   f"{dpd_designation} rows it brings up, and replace {line} with one whose page mentions it.")
        return ""

    def satchel_concise(self):
        censuses = self.needs.censuses()
        if len(censuses) < 2:
            return []
        if not self.wagon.metrics:
            return ["the cart is empty"]
        return self.wagon.satchel_flaws(censuses)

    def satchel_checkpoint(self):
        # each product must have a line of its own; an order without one always fails, so this check runs on every attempt
        faults = self.satchel_concise()
        if not faults:
            return ""
        products = [str(row["product"]) for row in self.needs.items()]
        encompassed = {self.duo_niche(line) for line, count in self.wagon.metrics.items() if count}
        missing = [name for index, name in enumerate(products) if index not in encompassed]
        lacking = ""
        if missing and len(missing) < len(products):
            lacking = " Still without a line: " + ", ".join(missing) + "."
        self.diary.append({"kind": "basket_hold", "faults": faults, "missing": missing})
        itemized = "; ".join(faults) + "." + lacking
        if self.maneuvers_residual() <= 3:
            return ("Not run (no step used). An order lacking some product's line always fails: " + itemized
                    + " With the steps that remain, add the missing line now if a fitting listing is already known; if not, "
                    "decline.")
        return ("Not run yet (no step used). The cart does not hold the whole basket yet: " + itemized
                + " Find a fitting listing for every product that is missing and add it (a line each, at its quantity), then "
                "order; decline only when some product has no listing able to meet its requirements.")

    def duo_prose(self):
        if not self.pairs:
            return ""
        products = [str(row["product"]) for row in self.needs.items()]
        count = len(products)
        ties = "; ".join(f"the {products[dpd]} must suit the {products[anchor]}" for dpd, anchor in self.pairs
                          if dpd < count and anchor < count)
        if not ties:
            return ""
        pivots = "/".join(dict.fromkeys(products[anchor] for _, anchor in self.pairs if anchor < count))
        return ("\n[fit] " + ties + ". Choose the " + pivots + " first: its own page gives its model name and number, and "
                "the page of a compatible item names that model in its list of supported ones (searching for the model "
                "turns them up).")

    def advance_onward_prose(self):
        # the actor is never the one to ask for the go-ahead: while a stated step is pending, every status line says so
        if not self.procedures or not self.role("message"):
            return ""
        if self.advance_decision == "yes" and self.affirmed_buggy == self.buggy_stamp():
            return ("\n[go-ahead] This basket has been shown to the shopper, who let it stand, which is the go-ahead the "
                    "request requires. Call the order tool now, and leave the cart unchanged unless a line no longer fits.")
        return ("\n[go-ahead] The finished cart needs the shopper's approval before it is bought, and the code takes care of "
                "that: as soon as the cart holds the final basket, call the order tool; the code then sends the basket to the "
                "shopper in the message reserved for it, reads the answer, and only then places the order. Never request the "
                "yes yourself, and never decline or clear the cart because it is still pending.")

    def advance_onward_reject_checkpoint(self):
        key = (self.buggy_stamp(), self.needs.edition)
        maintained_orderlines = key[0]
        if (not self.procedures or self.advance_decision == "no" or not maintained_orderlines or self.advance_maintain_stamp == key
                or self.maneuvers_residual() <= 1):
            return ""
        self.advance_maintain_stamp = key
        exhibited = ", ".join(f"{line} x{count:g}" for line, count in maintained_orderlines)
        return ("Not run yet (no step used). Nothing is lacking for the shopper's approval: when you call the order tool, the "
                "code presents the cart to the shopper in the message reserved for it and orders only if they let it stand. "
                "In the cart now: " + exhibited + ". Call the order tool with this cart; decline only if a line clearly fails a "
                "requirement currently in force and no record offers a replacement.")

    def advance_onward_checkpoint(self):
        delivered = self.deliver_advance_onward()
        if not delivered or delivered[2] == "yes":
            return ""
        note, uttered = delivered[0], delivered[1]
        return ("Not run yet (one step went on a message to the shopper). The request requires the shopper's go-ahead on the "
                f"final basket, so it was shown to them: \"{note}\" Their answer: \"{uttered}\" Act on that answer; if the "
                "basket changes, it is shown to them again before ordering.")

    def trigger_variety(self, trigger_designation, trigger_operands):
        order, decline = self.role("order"), self.role("decline")
        if not trigger_designation or trigger_designation not in (order, decline):
            return None
        if order != decline:
            return "decline" if trigger_designation != order else "order"
        concise_proses = (shard.lower() for shard in (trigger_operands or {}).values() if isinstance(shard, str) and len(shard) <= 40)
        for minor in concise_proses:
            if re.search(REJECT_VOCABLES, minor):
                return "decline"
            if re.search(CHECKOUT_VOCABLES, minor):
                return "order"
        return "order"

    def terminators(self):
        return {self.role("order"), self.role("decline")} - {None}

    def ticker_prose(self):
        left = max(0.0, self.channel.extra_alt() - ULTIMATE_TRIGGER)
        if left > (EXPIRY - ULTIMATE_TRIGGER) / 2:
            return ""
        entire = int(left // TICKER_PHASE) * TICKER_PHASE
        if entire > 0:
            return f"about {entire} s left"
        return f"under {TICKER_PHASE} s left"

    def standing_front(self, extra_alt):
        delivered = f"messages sent {self.txts_delivered}" + ("" if self.txt_top is None else f" of {self.txt_top}")
        fragments = [f"[status] steps used {self.channel.beat} of {self.phase_top}, {extra_alt} left", delivered]
        ticker = self.ticker_prose()
        if ticker:
            fragments.append(ticker)
        head = "; ".join(fragments) + ".\n[requirements now] "
        return head + self.needs.prose_corpus() + self.citecounter.ref_line() + self.encompass.commodity_phases()

    def standing_rear(self, extra_alt):
        several = bool(self.needs.items())
        if extra_alt <= 2:
            target = "every product asked for" if several else "the item"
            return (" LAST STEPS: when the cart already has a candidate for " + target + ", order it; otherwise add the best "
                    "candidate (a verified fit, or if there is none, the cheapest in-stock listing inside the budget that fails "
                    "no requirement) and order. Decline only if the records prove that nothing can fit.")
        if extra_alt <= 5 or self.channel.extra_alt() < SEALING:
            contour = "a line for each requested product" if several else "one line"
            return (" FINISH NOW: make the cart hold " + contour + " (a verified fit, or if there is none, the cheapest in-stock "
                    "listing inside the budget that fails no requirement; take out any sold-out line), check its stock once "
                    "more, then order. Decline only if the records prove that nothing can fit.")
        if self.channel.disbursed_outward():
            whomever = "every product asked for (a line each, at its quantity)" if several else "the item asked for"
            locale = "some product has" if several else "the store has"
            return (" MODEL BUDGET USED: stop exploring; within the next few steps put the best candidate for " + whomever
                    + " into the cart (a verified fit if one exists, otherwise the closest fit in the records), check its stock "
                    "and price again, and order. A decline is only right when the records prove that " + locale
                    + " no listing able to meet its requirements.")
        if self.orders_tried != 0 or not self.shrinking_presently():
            return ""
        expected_alt = self.admission.checkout_expected()
        if expected_alt < extra_alt <= expected_alt + 7:
            return (f" Steps are running low: within the next couple of steps put your best verified fit into the cart and "
                    f"order; searching and reading details stop once {expected_alt} steps remain, and repairing a refused order takes "
                    f"about 4 steps.")
        if extra_alt > expected_alt:
            return ""
        return (" Searching is over (details only for the cart's line, for listings a check named, or, while some product "
                "still lacks a verified fit, at most 3 further reads or searches altogether): put your best verified fit into the "
                "cart, check its stock and price again, and order; repairing a refused order still takes about 4 steps.")

    def shrinking_presently(self):
        return self.admission.shrinking()

    def standing_orderline(self):
        extra_alt = self.maneuvers_residual()
        text = self.standing_front(extra_alt)
        workable = [tdecl["function"]["name"] for tdecl in self.admission.tendered()]
        if len(workable) < len(self.utilitykit):
            scheduled = "; the code plans your questions to the shopper and sends them" if self.soliciting else ""
            text += ("\n[tools usable now] " + ", ".join(workable) + " (a call to any other tool is refused and costs no step"
                     + scheduled + ").")
        if self.wagon.utilized:
            text += "\n" + self.wagon.abstract()
        return text + self.duo_prose() + self.advance_onward_prose() + self.standing_rear(extra_alt)

    # vetting one batch of proposed calls
    def display(self, suggestion: dict[str, Any], sanctioned: set[str], automated: bool, lot: int, waitlisted_missives: int) -> tuple[str | None, str, dict[str, Any]]:
        cb = suggestion.get("function") or {}
        tool, crude = cb.get("name"), cb.get("arguments")
        if tool not in self.contours:
            return (f"Not run (no step used): there is no tool called '{tool}' here. Choose one of: "
                    f"{', '.join(sorted(sanctioned))}."), tool, {}
        if not automated and tool not in sanctioned:
            if tool == self.role("message") and self.soliciting:
                self.speak.wish_solicit(crude)
                return ("Not run (no step used): questions to the shopper are planned and sent by the code once you have "
                        "opened the details of a few listings."), tool, {}
            return ("Not run (no step used): not enough steps remain for that now; the tools you can still use are: "
                    + ", ".join(sorted(sanctioned)) + "."), tool, {}
        if isinstance(crude, dict):
            trigger_operands = crude
        elif (crude or "").strip():
            trigger_operands = blob_thing(crude)
        else:
            trigger_operands = {}
        if trigger_operands is None:
            return "Not run (no step used): the arguments were not a single JSON object.", tool, {}
        trigger_operands, faults = OperandValidate.actuate(trigger_operands, self.contours[tool])
        if faults:
            return "Not run (no step used): the arguments are not valid: " + "; ".join(faults[:6]), tool, trigger_operands
        refusal = self.lot_policies(tool, trigger_operands, automated, lot, waitlisted_missives)
        if refusal:
            return refusal, tool, trigger_operands
        fingerprint = tool + qua_prose(trigger_operands)
        bare_utility = tool not in self.terminators() and tool not in (self.role("message"), self.role("cart"))
        if bare_utility and fingerprint in self.last_calls:
            return ("Not run (no step used): you just made this very call and nothing has changed since; its result is "
                    "shown above. Try different arguments or another move."), tool, trigger_operands
        if not automated:
            maintained = self.admission.tardy_maintain(tool, trigger_operands) or self.encompass.measurement_shield(tool, trigger_operands)
            if maintained:
                return maintained, tool, trigger_operands
        if not automated:
            capped = self.capped_paging_hold(tool, trigger_operands)
            if capped:
                return capped, tool, trigger_operands
        if tool == self.role("cart"):
            refusal = self.display_buggy(trigger_operands, fingerprint)
            if not refusal and not automated:
                refusal = self.encompass.deeper_keep(trigger_operands)
            if refusal:
                return refusal, tool, trigger_operands
        return None, tool, trigger_operands

    def capped_paging_hold(self, tool: str, trigger_operands: dict[str, Any]) -> str:
        """Under a cheapest rule: a call that pages on through a listing limited by a price cap is held once per listing - its
        rows come in a fixed id order, so later pages are no cheaper than earlier ones, while a lower cap brings the cheapest
        rows of the whole category onto the first page."""
        try:
            if self.needs.priority() != "cheapest" or not isinstance(trigger_operands, dict):
                return ""
            cap_key = next((key for key, vlu in trigger_operands.items() if be_numeral(vlu) and not isinstance(vlu, bool)
                            and re.search(r"cap|max|price|cost|budget|ceiling|upper", str(key), re.I)
                            and not re.search(r"size|count|limit|page|per", str(key), re.I)), None)
            page_key = next((key for key, vlu in trigger_operands.items() if vlu not in (None, "", 0) and not be_numeral(vlu)
                             and re.search(r"next|token|cursor|continue|page|offset", str(key), re.I)), None)
            if cap_key is None or page_key is None:
                return ""
            cap = float(trigger_operands[cap_key])
            listing = json.dumps({key: vlu for key, vlu in trigger_operands.items() if key != page_key}, sort_keys=True, default=str)
            told = self.__dict__.setdefault("capped_told", set())
            if listing in told or cap < 4:
                return ""
            told.add(listing)
            lower = max(1, int(cap // 2))
            self.diary.append({"kind": "capped_paging_hold", "tool": tool, "cap": cap, "suggest": lower})
            return (f"Not run (no step used). The cheapest fit is wanted, and this listing (cap {cap:g}) still has further pages; "
                    "its rows come in a fixed id order, not by price, so paging on is a slow way to find the cheapest. List it "
                    f"again with a lower cap (about {lower}, or just below the cheapest fit you know) and no page token: the "
                    "cheapest rows of the category show up first. Raise the cap step by step only if nothing there fits; send "
                    "this same paging call again if you really need this page.")
        except Exception as exc:
            self.diary.append({"kind": "capped_paging_error", "error": f"{type(exc).__name__}: {exc}"[:300]})
            return ""

    def lot_policies(self, tool: str, trigger_operands: dict[str, Any], automated: bool, lot: int, waitlisted_missives: int) -> str:
        if tool in self.terminators() and lot > 1:
            return ("Not run (no step used): an order or a decline has to travel alone, and only once you have read what "
                    "the other calls returned.")
        if tool == self.role("message") and self.txt_top is not None:
            earmarked = 0 if self.affirming else self.maintained_reverse()
            if self.txts_delivered + waitlisted_missives >= self.txt_top - earmarked:
                return "Not run (no step used): the message budget is used up. Send the shopper nothing further."
        if automated or tool != self.role("cart_view") or tool == self.role("cart") or self.batch_cart:
            return ""
        if not self.wagon.metrics:
            return ""
        shelf = self.shelf
        edition = -2 if shelf.buggy_rev is None else shelf.buggy_rev
        starting = 1 if self.starting_buggy is None else self.starting_buggy
        if shelf.buggy_exhibited != edition or starting or shelf.buggy_exhibited in self.view_told:
            return ""
        self.view_told.add(shelf.buggy_exhibited)
        return ("Not run (no step used): the latest cart result already shows the basket - " + self.wagon.abstract()
                + " Viewing the cart again tells you nothing new; to check a line's stock once more, open its record with "
                + (shelf.supply_utility() or "whichever tool reports stock") + ".")

    def display_buggy(self, trigger_operands: dict[str, Any], sig: str) -> str:
        wagon = self.wagon
        entity = wagon.aim(trigger_operands)
        inserting = wagon.be_deposit(trigger_operands)
        direct_alt = self.citecounter.cite_arrow_comment(entity) if entity and inserting else ""
        if direct_alt and self.pointer_told.get(entity, 0) < 2:
            self.pointer_told[entity] = self.pointer_told.get(entity, 0) + 1
            return f"Not run (no step used): {direct_alt}. Pick a different listing."
        maintained_comment = self.cleared_orderline_maintain(trigger_operands, entity, sig)
        if maintained_comment:
            return maintained_comment
        if not (inserting and entity):
            return ""
        if wagon.maintained(entity) > 0 and wagon.appends is not False and sig not in self.warned:
            self.warned.add(sig)
            return (f"Not run (no step used): the cart already has {entity} x{int(wagon.maintained(entity))}, and this cart tool "
                    "adds to lines it holds, so the call would push the count up. To order it, leave the cart as it is; to "
                    "change its units, take away or add only the difference. If you really want to add, send the same call "
                    "once more.")
        if not direct_alt:
            walk = self.encompass.route_maintain(trigger_operands)
            if walk:
                return walk
        return (self.validate_novel_orderlines(trigger_operands, entity) or self.grade_orderlines(trigger_operands)
                or self.metric_policies(trigger_operands, entity, sig))

    def cleared_orderline_maintain(self, trigger_operands: dict[str, Any], entity: str | None, sig: str) -> str:
        wagon, needs, stash = self.wagon, self.needs, self.stash
        stamp_alt = f"{sig}#{self.puts.get(entity, 0)}"
        personal_cost = stash.price(entity) if entity else None
        thriftier_deposit = any(further != entity and be_numeral(personal_cost) and be_numeral(stash.price(further))
                          and stash.price(further) < personal_cost for further in self.batch_puts)
        switch = needs.priority() == "cheapest" and (thriftier_deposit or bool(self.referee.thriftier_befit(entity)))
        moved = entity in self.pass_price and self.pass_price[entity] != personal_cost
        if not entity or wagon.be_deposit(trigger_operands):
            return ""
        if f"{entity}|{needs.edition}" not in self.passed or switch or moved:
            return ""
        count = wagon.census_operand(trigger_operands)
        if count is not None and 0 < count < wagon.maintained(entity):
            return ""
        if self.refusal_n != self.refusals_seen.get(entity, self.refusal_n) or stamp_alt in self.warned:
            return ""
        self.warned.add(stamp_alt)
        fewest = ""
        if needs.priority() == "cheapest" and not self.referee.thriftier_befit(entity):
            fewest = " It is also the cheapest verified fit."
        return (f"Not run (no step used): an independent check found that {entity} meets every requirement in force, reading "
                "each property by the value rules (where the store writes the value itself in a field, only those words "
                "count); what kind of item it is follows from the category label alone, whatever object its title or "
                f"description mentions.{fewest} Nothing has changed since. Send the same call again only if a change or a "
                "refusal rules it out, if one of its fields clearly fails a requirement, or if you are replacing it with a "
                "cheaper listing.")

    def validate_novel_orderlines(self, trigger_operands: dict[str, Any], entity: str) -> str:
        edition = self.needs.edition
        surplus = [lbl for lbl, count in self.wagon.lines(trigger_operands)[1:] if count is None or count > 0]
        for lbl in surplus + [entity]:
            key = f"{lbl}|{edition}"
            if self.maneuvers_residual() <= 3 or key in self.checked_adds:
                continue
            self.checked_adds.add(key)
            verdict = self.referee.validate_article(lbl, "ADD TO THE CART")
            if verdict:
                return "Not run (no step used). " + verdict + " Should you be sure it fits, repeat this exact call."
        return ""

    def grade_orderlines(self, trigger_operands: dict[str, Any]) -> str:
        lines = self.wagon.lines(trigger_operands)

        def dropped(count):
            return count is not None and count <= 0

        if self.maneuvers_residual() > 3:
            for orderline_ident, count in lines:
                if dropped(count):
                    continue
                better = self.referee.grade_validate(orderline_ident)
                if not better:
                    continue
                thriftier = self.referee.grade_authenticate(orderline_ident)
                text = better.replace("before ordering", "before you add it to the cart").replace(
                    "order the current pick", "repeat this call unchanged")
                if thriftier:
                    text += (f" An independent item check found {thriftier} meeting every requirement now in force at a "
                             f"lower price than {orderline_ident}: use {thriftier} instead.")
                return "Not run (no step used). " + text
        if self.needs.priority() != "cheapest":
            return ""
        notified = self.cheaper_told
        for orderline_ident, count in lines:
            if dropped(count):
                continue
            thriftier = self.referee.thriftier_befit(orderline_ident)
            key = ("cart", orderline_ident, thriftier)
            if not thriftier or thriftier in self.wagon.metrics or notified.get(key):
                continue
            notified[key] = 1
            return (f"Not run (no step used). The independent item check found {thriftier} meeting every requirement now in "
                    f"force at a lower price than {orderline_ident}. The request wants the cheapest fit, and the check's reading "
                    f"takes precedence over yours: add {thriftier} instead. Should a record show {thriftier} failing a requirement "
                    "now in force, repeat the same call.")
        return ""

    def metric_policies(self, trigger_operands: dict[str, Any], entity: str, sig: str) -> str:
        wagon, needs = self.wagon, self.needs
        wished_alt = needs.quantity()
        solicited = wagon.census_operand(trigger_operands)
        aim = wished_alt - wagon.maintained(entity) if wagon.appends is not False else wished_alt
        if needs.items():
            unusual = [(lbl, int(count)) for lbl, count in wagon.lines(trigger_operands)
                   if count is not None and count > 0 and int(count) not in needs.censuses()]
            if not unusual or sig in self.warned:
                return ""
            self.warned.add(sig)
            return (f"Not run (no step used): the products requested call for {needs.censuses()} units (a line each), but this "
                    f"call asks for {unusual}. If you are certain, send the same call again.")
        if solicited is None or abs(solicited - aim) <= 1e-9 or sig in self.warned:
            return ""
        self.warned.add(sig)
        return (f"Not run (no step used): the request calls for {wished_alt} units of one listing, yet this call asks for "
                f"{int(solicited)}. If you are certain, send the same call again.")

    def display_terminate(self, cuid: str, trigger_designation: str, trigger_operands: dict[str, Any]) -> tuple[str | None, dict[str, Any]]:
        def primary_maintain(guards):
            for checkpoint in guards:
                note = checkpoint()
                if note:
                    return note
            return None

        kind = self.trigger_variety(trigger_designation, trigger_operands)
        if kind == "order":
            return primary_maintain((self.vended_orderline_checkpoint, self.satchel_checkpoint, self.duo_checkpoint, self.thriftier_select_checkpoint,
                               self.route_checkpoint, self.obsolete_checkpoint, self.scrutinize_checkpoint, self.advance_onward_checkpoint)), trigger_operands
        if kind != "decline":
            return None, trigger_operands
        maintained = primary_maintain((self.advance_onward_reject_checkpoint, self.shelf.nigh_maintain, self.preserved_orderline_checkpoint, self.cushion_checkpoint,
                           self.populate_checkpoint, self.designated_checkpoint, self.folios_checkpoint, self.inspection_afresh_checkpoint, self.primary_inspection_checkpoint,
                           self.befit_checkpoint, self.optimal_checkpoint, self.lenient_cite_checkpoint, self.shelf.decline_repair,
                           self.shelf.near_check))
        if maintained:
            return maintained, trigger_operands
        return None, self.sealer.sanitize_reject(trigger_operands, trigger_designation)

    def vended_orderline_checkpoint(self) -> str:
        if self.maneuvers_residual() <= 2:
            return ""
        shelf = self.shelf
        dead = sorted(set(self.wagon.metrics) & shelf.inert_orderlines())
        key = (tuple(dead), self.cart_seq)
        notified = self.dead_told
        if not dead:
            return ""
        instances = notified.get(key, 0)
        if instances < 2:
            notified[key] = instances + 1
            return "Not run yet (no step used). " + shelf.inert_comment(dead)
        if instances != 2 or not self.role("cart") or self.maneuvers_residual() < 6 or self.channel.extra_alt() <= SEALING:
            return ""
        notified[key] = 3
        were_hushed = self.channel.muffle
        try:
            exchanged = self.sealer.routine_buggy(set(dead) | shelf.inert_orderlines(), set(self.wagon.metrics) - set(dead))
        except Exception as exn:
            exchanged = False
            self.diary.append({"kind": "dead_swap_error", "error": f"{type(exn).__name__}: {exn}"[:300]})
        finally:
            self.channel.muffle = were_hushed
        self.diary.append({"kind": "dead_swap", "lines": dead, "done": exchanged, "cart": dict(self.wagon.metrics)})
        if not exchanged or not self.wagon.metrics or set(self.wagon.metrics) & shelf.inert_orderlines():
            return ""
        return ("Not run yet (no step used). The cart still had " + ", ".join(dead) + ", which has sold out, so the code took "
                "it out, put a candidate in its place, left the other lines alone, and read the lines' stock: "
                + self.wagon.abstract() + " Check this cart against the request, then order it.")

    def thriftier_select_checkpoint(self) -> str:
        if self.maneuvers_residual() <= 3:
            return ""
        referee = self.referee
        entity = referee.preferred()
        thriftier = referee.thriftier_befit(entity) if entity else ""
        if not thriftier:
            for line in [lbl for lbl in self.wagon.metrics if lbl != entity]:
                thriftier = referee.thriftier_befit(line)
                if thriftier:
                    entity = line
                    break
        notified = self.cheaper_told
        key = (entity, thriftier)
        if not thriftier or notified.get(key, 0) >= 2:
            return ""
        notified[key] = notified.get(key, 0) + 1
        return (f"Not run yet (no step used). The independent item check found {thriftier} meeting every requirement at a lower "
                f"price than {entity}. The request wants the cheapest fit, and the check's reading takes precedence over yours: "
                f"remove {entity} from the cart, add {thriftier}, read its stock and price again, and order it.")

    def route_checkpoint(self) -> str:
        if self.wagon.metrics:
            return self.encompass.route_maintain(None)
        return ""

    def obsolete_checkpoint(self) -> str:
        browser = self.shelf.supply_utility()
        if not browser or self.maneuvers_residual() <= len(self.wagon.metrics) + 2 or self.channel.extra_alt() <= SEALING:
            return ""
        obsolete = self.shelf.obsolete_orderlines()
        key = (tuple(sorted(self.wagon.metrics)), self.cart_seq)
        if not obsolete or self.recheck_mark == key:
            return ""
        self.recheck_mark = key
        first = obsolete[:4]
        browses = self.browses
        browses.access_waitlist = first
        browses.access_revalidate = True
        browses.access_intention = "cart"
        browses.access_utility = browser
        return ("Not run yet (no step used). Stock and prices can change silently, and nobody has read these cart lines since "
                "the cart last changed: " + ", ".join(first) + f". The code now reads their stock for you ({browser}, the tool "
                "that reports stock); afterwards order if every one is still in stock and the total stays within the budget, "
                "and otherwise replace that line.")

    def scrutinize_checkpoint(self) -> str:
        if self.order_vetted or self.maneuvers_residual() <= 2 or self.channel.extra_alt() <= SEALING:
            return ""
        self.order_vetted = True
        referee, needs = self.referee, self.needs
        faults = self.wagon.faults(needs.quantity(), needs.censuses())
        if faults and self.maneuvers_residual() > 3:
            return ("Not run yet (no step used). By the code's count the cart is not ready yet: " + "; ".join(faults)
                    + ". Put it right, then order.")
        several = needs.items()
        if several:
            lines = list(self.wagon.metrics)
        else:
            lines = [referee.preferred()] if referee.preferred() else []
        for entity in lines:
            verdict = referee.validate_article(entity, "ORDER")
            if verdict:
                return "Not run yet (no step used). " + verdict
        for entity in lines:
            better = referee.grade_validate(entity)
            if not better:
                continue
            thriftier = referee.grade_authenticate(entity)
            surplus = ""
            if thriftier:
                surplus = (f" An independent item check found {thriftier} meeting every requirement now in force at a lower "
                         f"price than {entity}: remove {entity}, add {thriftier}, read its stock and price afresh, and order.")
            return "Not run yet (no step used). " + better + surplus
        purview = "for each product asked for, each at its quantity" if several else "holding the chosen item at the required quantity"
        consent = " (when you call the order tool, the code asks the shopper to approve the cart)" if self.procedures else ""
        return ("Not run yet (no step used). Before ordering, make sure that: (1) the cart has exactly one line " + purview
                + " and no other lines; (2) since the last cart change you have read this item's stock and price again with a "
                "tool that reports them, it is available, and price times quantity stays within the current budget; (3) the "
                "item's own properties meet every stated requirement, every need the shopper revealed and every announced "
                "change; (4) if the cheapest or best was asked for, nothing you found fits better; (5) every procedural rule of "
                "the request has been followed" + consent + ". Check whatever is still unchecked now; if all of it holds, "
                "call the order tool again.")

    def preserved_orderline_checkpoint(self) -> str:
        dead = self.shelf.inert_orderlines
        edition = self.needs.edition
        verified = [lbl for lbl in self.wagon.metrics
                    if f"{lbl}|{edition}" in self.passed and lbl not in self.refused_lines and lbl not in dead()]
        if not verified or self.maneuvers_residual() <= 1 or self.decline_kept:
            return ""
        self.decline_kept = True
        every = " together with each product's line in your cart" if self.needs.items() else ""
        return ("Not run yet (no step used). The cart has " + ", ".join(verified) + ", which the independent check judged to "
                "satisfy all requirements currently in force. Declining would throw that away: read its stock and price again and "
                "order it" + every + ". Decline only if a later change now plainly excludes it.")

    @staticmethod
    def browse_triggers(rows: list[str], lot: Any) -> int:
        count = len(rows)
        if not lot or count < 2:
            return count
        return -(-count // lot[2])

    def cushion_checkpoint(self) -> str:
        shelf = self.shelf
        cushion = shelf.cushion_datarows() if not self.slack_done and (self.role("details") or shelf.lot_utility()) else []
        scatter = shelf.lot_utility()
        if not cushion or self.maneuvers_residual() <= self.browse_triggers(cushion, scatter) + 4 or self.channel.extra_alt() <= SEALING + 40:
            return ""
        self.slack_done = True
        self.browses.access_waitlist = cushion
        self.browses.access_utter = "Ahead of any decline, I open cheaper rows that could bring all products inside the budget together."
        total = shelf.cushion_overall
        exhibited = f"{total:g}" if be_numeral(total) else "more than the budget"
        return ("Not run yet (no step used). Every product has a fitting listing, but together their cheapest fits cost "
                + exhibited + f", over the budget of {self.needs.disburse_top}. Rows priced inside what the budget leaves each "
                "product once the other products' cheapest fits are paid for were never opened, so the code reads their "
                "records now: " + ", ".join(cushion) + ". Test each against its own product's requirements by the value rules "
                "(where the store writes the value itself in a field, only those words count); as soon as fits for all "
                "products together stay within the budget, put each product's line in the cart and order; decline only if no "
                "such combination exists.")

    def populate_checkpoint(self) -> str:
        shelf = self.shelf
        lot = shelf.lot_utility()
        populate = shelf.deficient_datarows() if not self.decline_fill and (self.role("details") or lot) else []
        if not populate or self.maneuvers_residual() <= self.browse_triggers(populate, lot) + 3 or self.channel.extra_alt() <= SEALING + 40:
            return ""
        self.decline_fill = True
        self.browses.access_waitlist = populate
        self.browses.access_utter = "Ahead of any decline, I open unread listings for the products still lacking a verified fit."
        return ("Not run yet (no step used). Some requested products have no verified fit yet, and in-budget listings of their "
                "categories were never opened, so before any decline the code reads their records now: " + ", ".join(populate)
                + ". Test each against its own product's requirements by the value rules (where the store writes the value "
                "itself in a field, only those words count); once each product has a fitting listing, put each product's "
                "line in the cart and order; decline only if some product really has none.")

    def designated_checkpoint(self) -> str:
        lot = self.shelf.lot_utility()
        waitlist = self.referee.designated_unbrowsed()[: lot[2] if lot else 3]
        if not waitlist or self.decline_opens or not (self.role("details") or lot):
            return ""
        if self.maneuvers_residual() <= self.browse_triggers(waitlist, lot) + 3 or self.channel.extra_alt() <= SEALING + 40:
            return ""
        self.decline_opens = True
        self.browses.access_waitlist = waitlist
        return ("Not run yet (no step used). The listings the checks pointed to were never opened, so before any decline the "
                "code fetches their records now: " + ", ".join(waitlist) + ". Hold each one against the requirements by the "
                "value rules (the field named for the property decides, and where the store writes the value itself there, "
                "only those words count; without such a field, another field, the options list or the title may give the "
                "value), buy one that fits, and decline only if they all fail.")

    def folios_checkpoint(self) -> str:
        note = self.walk.decline_reads()
        if not note:
            return ""
        return "Not run yet (no step used). " + note

    def befit_checkpoint(self) -> str:
        note = self.walk.befit_maintain()
        if not note:
            return ""
        return "Not run yet (no step used). " + note

    def inspection_afresh_checkpoint(self) -> str:
        referee = self.referee
        constraint = referee.inspection_browses if referee.inspection_browses is not None else 10 ** 9
        crisp_browses = len(self.stash.read) > constraint and referee.inspections < 2
        if not (crisp_browses and self.decline_vetted) or self.maneuvers_residual() <= 3 or self.channel.extra_alt() <= SEALING + 30:
            return ""
        note = referee.evaluation_reject()
        return "Not run yet (no step used). " + note if note else ""

    def primary_inspection_checkpoint(self) -> str:
        if self.decline_vetted or self.maneuvers_residual() <= 3 or self.channel.extra_alt() <= SEALING + 30:
            return ""
        self.decline_vetted = True
        note = self.referee.evaluation_reject()
        if note:
            return "Not run yet (no step used). " + note
        return ("Not run yet (no step used). Make sure before declining: you varied your searches (different words, filters, "
                "pages), you looked at each candidate's own properties, and every candidate fails a stated requirement, a "
                "revealed need, the budget or stock. A property's field counts however its key is spelled. If something fits, "
                "buy it instead; if nothing does, call the decline tool once more.")

    def optimal_checkpoint(self) -> str:
        walk, needs = self.walk, self.needs
        best = walk.undisputed() if not self.best_held and self.maneuvers_residual() > 4 else []
        for _ in range(3):
            unvetted = [lbl for lbl in best if f"{lbl}|{needs.edition}" not in self.passed]
            if not unvetted or self.channel.extra_alt() < SEALING + 30:
                break
            self.referee.validate_article(unvetted[0], "ORDER")
            best = walk.undisputed()
        if not best or self.maneuvers_residual() <= len(best) + 3 or self.channel.extra_alt() <= SEALING + 20:
            return ""
        self.best_held = True
        several = bool(needs.items())
        retaining = "a candidate for each requested product" if several else "a candidate"
        depositing = "a line for each product, at its own quantity," if several else "it"
        return ("Not run yet (no step used). Declining is only proper when no listing can meet the requirements, and the "
                "records contain " + retaining + " that shows stock, stays within the budget and has not been ruled out by "
                "any check: " + ", ".join(best) + ". Unless some record shows one of them breaking a requirement now in force, "
                "put " + depositing + " in the cart, check stock and price once more, and order; decline only if every candidate "
                "clearly fails.")

    def lenient_cite_checkpoint(self) -> str:
        soft = [cite for cite in self.refs or []
                if cite.get("relation") == "differ" and cite.get("status") in ("open", "stuck") and self.citecounter.cite_ongoing(cite)]
        if not soft or self.maneuvers_residual() <= 3 or self.ref_decline_held:
            return ""
        self.ref_decline_held = True
        crowns = ", ".join(dict.fromkeys(cite["head"] for cite in soft))
        apiece_commodity = ", a line for each product" if self.needs.items() else ""
        return ("Not run yet (no step used). " + crowns + " depends on another listing that the code could not identify, so "
                "its value is unknown: it can neither rule a listing out nor justify declining. Put the listing that meets "
                "all the other requirements (in stock, inside the budget) in the cart" + apiece_commodity + ", check its stock and "
                "price once more, and order; decline only if no listing meets the other requirements.")

    def comment_lot_deposits(self, pitches: list[dict[str, Any]]) -> None:
        cart = self.role("cart")
        self.batch_puts = []
        self.batch_cart = any((item.get("function") or {}).get("name") == cart for item in pitches if isinstance(item, dict))

        def dissected(crude):
            if isinstance(crude, dict):
                return crude
            return blob_thing(crude) if isinstance(crude, str) and crude.strip() else {}

        for suggestion in pitches:
            cb = suggestion.get("function") or {}
            if cb.get("name") != cart:
                continue
            deposit_operands = dissected(cb.get("arguments"))
            if isinstance(deposit_operands, dict) and self.wagon.be_deposit(deposit_operands) and self.wagon.aim(deposit_operands):
                self.batch_puts.append(self.wagon.aim(deposit_operands))

    def forward(self, pitches: list[dict[str, Any]], automated: bool = False) -> list[tuple[str, str]]:
        retorts: list[tuple[str, str]] = []
        self.admission.start_lot()
        sanctioned = {tdecl["function"]["name"] for tdecl in self.admission.tendered()}
        outbound: list[tuple[str, str, dict[str, Any]]] = []
        waitlisted = 0
        buggy_obstructed = False
        self.comment_lot_deposits(pitches)
        for suggestion in pitches:
            if buggy_obstructed and suggestion["function"]["name"] == self.role("cart"):
                retorts.append((suggestion["id"], "Not run (no step used): an earlier cart change in this reply was held back; "
                                                "read the reason, then send the cart changes again."))
                continue
            refusal, tool, trigger_operands = self.display(suggestion, sanctioned, automated, len(pitches), waitlisted)
            if refusal:
                if not buggy_obstructed:
                    buggy_obstructed = (tool == self.role("cart") and isinstance(trigger_operands, dict) and bool(trigger_operands)
                                    and not self.wagon.be_deposit(trigger_operands))
                retorts.append((suggestion["id"], refusal))
                continue
            if tool == self.role("message"):
                waitlisted += 1
            outbound.append((suggestion["id"], tool, trigger_operands))
        if outbound and outbound[0][1] in self.terminators():
            cuid, tool, trigger_operands = outbound[0]
            maintained, trigger_operands = self.display_terminate(cuid, tool, trigger_operands)
            if maintained:
                retorts.append((cuid, maintained))
                return retorts
            outbound[0] = (cuid, tool, trigger_operands)
        if outbound and outbound[0][1] not in self.terminators() and self.terminators():
            room = max(0, self.maneuvers_residual() - 1)
            retorts.extend((cuid, "Not run (no step used): only one step remains, and it is kept for the order (or the "
                                 "decline).") for cuid, _, _ in outbound[room:])
            outbound = outbound[:room]
        for start in range(0, len(outbound), 16):
            plank = outbound[start: start + 16]
            if self.channel.done or self.channel.snapped or self.maneuvers_residual() <= 0:
                retorts.extend((cuid, "Not run: the episode is over or no step remains.") for cuid, _, _ in plank)
                continue
            retorts.extend(self.deliver_fragment(plank[: self.maneuvers_residual()]))
        checkout_relating = {trigger["id"]: index for index, trigger in enumerate(pitches)}
        retorts.sort(key=lambda answer: checkout_relating.get(answer[0], 0))
        return retorts

    def deliver_fragment(self, plank: list[tuple[str, str, dict[str, Any]]]) -> list[tuple[str, str]]:
        actions = [{"name": tool, "args": trigger_operands} for _, tool, trigger_operands in plank]
        phase_preceding = self.channel.beat
        reply = self.channel.operate(actions)
        self.diary.append({"kind": "env", "step": phase_preceding + 1, "actions": actions, "reply": reply})
        if reply is None:
            return [(cuid, "The call did not go through (network or server trouble). Send it again.") for cuid, _, _ in plank]
        observations = reply.get("observations")
        if not isinstance(observations, list):
            observations = [reply.get("observation")]
        terminal = len(observations) - 1
        proses = []
        for index, (cuid, tool, trigger_operands) in enumerate(plank):
            if index > terminal:
                proses.append((cuid, "Not run: the episode had ended before this call."))
            else:
                proses.append((cuid, self.ingest(tool, trigger_operands, observations[index], reply, index == terminal)))
        return proses

    def tome_trigger(self, tool: str, trigger_operands: dict[str, Any], obv: Any) -> None:
        utterance_utility, buggy_utility = self.role("message"), self.role("cart")
        if tool == utterance_utility:
            self.txts_delivered += 1
        self.last_kind = self.trigger_variety(tool, trigger_operands)
        if self.last_kind == "order":
            self.orders_tried += 1
        if tool in (buggy_utility, utterance_utility) or tool in self.terminators():
            self.last_calls = []
        else:
            self.last_calls = (self.last_calls + [tool + qua_prose(trigger_operands)])[-2:]
        self.stash.add(obv)
        self.absorb_n += 1
        if tool != buggy_utility and not self.covering:
            self.starting_buggy = 1 if self.starting_buggy is None else self.starting_buggy + 1

    def comment_buggy_browses(self, tool: str, trigger_operands: dict[str, Any], obv: Any) -> None:
        shelf = self.shelf
        maintained = self.wagon.metrics
        supply_policy = bool(shelf.supply_noticed)
        solicited = shelf.solicited_idents(trigger_operands)
        datarows_reliable = bool(shelf.datarows_ongoing) and not shelf.datarows_delay
        retrievals_solely = not datarows_reliable and tool not in (self.role("cart"), self.role("cart_view"))
        crisp: set[str] = set()

        def locus(transcript: dict[str, Any]) -> None:
            idents_herein = {vlu for key, vlu in transcript.items() if isinstance(vlu, str) and vlu in maintained and "id" in str(key).lower()}
            if not idents_herein:
                return
            if supply_policy:
                if self.observe.supply_situation(transcript) is None or (retrievals_solely and not idents_herein & solicited):
                    return
            elif not any(isinstance(vlu, dict) for vlu in transcript.values()):
                return
            crisp.update(vlu for vlu in transcript.values() if isinstance(vlu, str) and vlu in maintained)

        route_hashmaps(obv, locus)
        for orderline_ident in crisp:
            self.line_read[orderline_ident] = self.channel.beat
            self.line_seq[orderline_ident] = self.absorb_n

    def census_tardy_browse(self, tool: str, trigger_operands: dict[str, Any], obv: Any) -> None:
        details, search = self.role("details"), self.role("search")
        if tool == details and not (isinstance(obv, dict) and obv.get("error")):
            self.stash.read.update(vlu for vlu in trigger_operands.values() if isinstance(vlu, str) and vlu in self.stash.ids)
        peeked = bool(details) and tool not in (details, search) and tool in self.shelf.fetch_utilities()
        named = {vlu for vlu in trigger_operands.values() if isinstance(vlu, str)}
        if self.maneuvers_residual() >= self.admission.checkout_expected() or self.covering:
            return
        if tool not in (details, search) and not peeked:
            return
        maintained = set(self.wagon.metrics)
        if tool == details and named & maintained:
            return
        if peeked and named & (maintained | {key.rpartition("|")[0] for key in self.passed}):
            return
        self.late_n += 1

    def buggy_observations(self, trigger_operands: dict[str, Any], obv: Any, observations_alt: list[str]) -> None:
        wagon, shelf = self.wagon, self.shelf
        self.order_vetted = False
        wagon.modified_upon = self.channel.beat
        maintained_preceding = dict(wagon.metrics)
        observations_alt.append(wagon.reserved_alt(trigger_operands, obv))
        try:
            shelf.buggy_maneuvered(maintained_preceding)
            shelf.stamp_crisp(obv)
            shelf.afterward_deposit(trigger_operands, obv)
        except Exception as exn:
            self.diary.append({"kind": "after_put_error", "error": f"{type(exn).__name__}: {exn}"[:300]})
        item = wagon.aim(trigger_operands)
        inserting = bool(item) and wagon.be_deposit(trigger_operands)
        if inserting:
            self.refusals_seen[item] = self.refusal_n
            self.puts[item] = self.puts.get(item, 0) + 1
        metrics = wagon.census_operand(trigger_operands)
        if not inserting or f"{item}|{self.needs.edition}" not in self.passed:
            return
        if (metrics is None or metrics > 0) and wagon.maintained(item) > 0 and item not in shelf.inert_orderlines():
            observations_alt.append(f"An independent check found that {item} meets every requirement now in force, reading each "
                           "property by the value rules (where this store writes the value itself in a field, only those "
                           "words count); keep it unless a later change excludes it.")

    def ingest(self, trigger_designation: str, trigger_operands: dict[str, Any], obv: Any, ans: dict[str, Any], ultimate: bool) -> str:
        exhibited, obv = obv, scoped(obv)
        buggy_utility, utterance_utility = self.role("cart"), self.role("message")

        def journaled(kind, exn):
            self.diary.append({"kind": kind, "error": f"{type(exn).__name__}: {exn}"[:300]})

        self.tome_trigger(trigger_designation, trigger_operands, obv)
        try:
            vended_presently = self.shelf.glean_tops(trigger_designation, trigger_operands, obv)
        except Exception as exn:
            vended_presently = []
            journaled("caps_error", exn)
        if self.wagon.metrics:
            self.comment_buggy_browses(trigger_designation, trigger_operands, obv)
        self.census_tardy_browse(trigger_designation, trigger_operands, obv)
        observations_alt = []
        if vended_presently:
            observations_alt.append("[stock] " + self.shelf.inert_comment(vended_presently))
        try:
            diverted = self.citecounter.comments_observation(trigger_designation, trigger_operands)
            if diverted:
                observations_alt.append("[notes] " + diverted)
        except Exception as exn:
            journaled("notes_remark_error", exn)
        if trigger_designation == buggy_utility:
            self.buggy_observations(trigger_operands, obv, observations_alt)
        if trigger_designation == self.role("cart_view"):
            self.wagon.glanced(obv, self.channel.beat)
            observations_alt.append(self.wagon.abstract())
        refused = self.observe.refused(trigger_designation, obv, ans if ultimate else {"finished": False})
        if refused and self.last_kind == "order":
            self.cart_seq = self.absorb_n
        observations_alt.append(self.observe.comments_towards(trigger_designation, obv, refused))
        if trigger_designation == self.role("search"):
            observations_alt.append(self.encompass.hunt_comment(trigger_operands, obv))
            self.referee.designated_folios.discard(json.dumps(trigger_operands, sort_keys=True, default=str))
            try:
                observations_alt.append(self.shelf.merge_joined(obv))
            except Exception as exn:
                journaled("merge_joined_error", exn)
        if ultimate and self.opening_inspection_expected(trigger_designation):
            self.early_audit = True
            onward = self.referee.evaluation_reject(early=True)
            if onward:
                observations_alt.append("So far no fit has been verified. " + onward)
        question = None
        if trigger_designation == utterance_utility:
            question = " ".join(str(vlu) for vlu in trigger_operands.values() if isinstance(vlu, str))
        snagged = self.encompass.buyer_orderlines(trigger_designation, obv, refused)
        if snagged:
            observations_alt.append(self.needs.perceive(snagged, question))
            self.citecounter.cite_grab()
        text = qua_prose(cropped(exhibited, HUNT_PROSE if trigger_designation == self.role("search") else TRANSCRIPT_PROSE))
        if ultimate and ans.get("finished"):
            text += " [the episode has finished]"
        spliced = " ".join(observation for observation in observations_alt if observation)
        if not spliced:
            return text
        return text + "\n[note] " + spliced

    def opening_inspection_expected(self, tool: str) -> bool:
        if self.early_audit or tool in (self.role("cart"), self.role("message")) or tool in self.terminators():
            return False
        if self.wagon.metrics or not PRIMARY_CHECKOUT_THROUGH < self.maneuvers_residual() <= self.phase_top // 3 + 2:
            return False
        edition = str(self.needs.edition)
        return all(key.rpartition("|")[2] != edition for key in self.passed)

    # the main loop of an episode
    def tool_calls(self, txt: dict[str, Any], sqn_nil: int) -> tuple[list[dict[str, Any]], int]:
        pitches = []
        for trigger in txt.get("tool_calls") or []:
            if not isinstance(trigger, dict):
                continue
            sqn_nil += 1
            cb = trigger.get("function") or {}
            crude = cb.get("arguments")
            pitches.append({"id": trigger.get("id") or f"call_{sqn_nil}", "type": "function",
                              "function": {"name": cb.get("name"),
                                           "arguments": crude if isinstance(crude, str) else json.dumps(crude or {})}})
        return pitches, sqn_nil

    def access_parley(self) -> None:
        self.admission.browse_appeal()
        self.diary.append({"kind": "plan", "plan": self.needs.browsing, "roles": self.function_atlas})
        opening = self.plook.get("opening")
        partitions = ["SHOPPER REQUEST:\n" + self.request]
        if opening is not None:
            partitions.append("OPENING OBSERVATION:\n" + qua_prose(opening))
            self.stash.add(opening)
        outline = self.needs.outline(self.phase_top, self.txt_top)
        if outline:
            partitions.append(outline)
        partitions.append(self.standing_orderline())
        self.chat = [{"role": "system", "content": OPERATE_POLICIES}, {"role": "user", "content": "\n\n".join(partitions)}]

    def stint_obstructed(self) -> bool:
        channel = self.channel
        if channel.done or channel.snapped or self.maneuvers_residual() <= 0:
            return True
        return channel.extra_alt() < ULTIMATE_TRIGGER

    def actuate(self) -> list[dict[str, Any]]:
        if self.channel.done:
            return self.diary
        self.access_parley()
        automated = (BuyerRejoinderPhase(), WaitlistedBrowsesPhase(), AisleCombPhase(), MeasurementFoliosPhase(),
                    ScheduledInquiryPhase(), CitationLeapPhase(), InexpensiveRoutePhase())
        operator = OperatorStintPhase()
        stints = 0
        while stints < STINT_CONSTRAINT and not self.stint_obstructed():
            stints += 1
            self.sealer.snag_aloft()
            if any(step.undertake(self) for step in automated):
                continue
            if not operator.undertake(self):
                break
        if not (self.channel.done or self.channel.snapped):
            self.sealer.terminate_presently()
        self.diary.append({"kind": "end", "finished": self.channel.done, "steps": self.channel.beat,
                         "cost": round(self.channel.disburse, 5), "seconds": round(_TALLY.now() - self.channel.s0, 1)})
        return self.diary


def agent_main(pdata: dict[str, Any]) -> list[dict[str, Any]]:
    drive = ShoppingRun(pdata)
    try:
        return drive.actuate()
    except Exception as exn:
        drive.diary.append({"kind": "crash", "error": f"{type(exn).__name__}: {exn}"[:500]})
        try:
            if not (drive.channel.done or drive.channel.snapped):
                drive.sealer.terminate_presently()
        except Exception as second:
            drive.diary.append({"kind": "crash", "error": f"{type(second).__name__}: {second}"[:500]})
        return drive.diary


__all__ = ["agent_main"]
