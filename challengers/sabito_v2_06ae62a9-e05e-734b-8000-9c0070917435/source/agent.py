"""Sabito v7.1 - a shopping agent for the ORO benchmark (Bittensor SN15), built by make_sabito_v7_1.py from v7.0
(make_sabito_v7_0.py, d8ee64b3); v7.1 restates every prompt text in our own words with the same meaning.
v7.2 (make_sabito_v7_2.py, from v7.1 df5a72bc): finishing-enum kinds (W1), endearments dropped before the
LISTENER (E1), list caps read from tool descriptions (W8).
v7.3 (make_sabito_v7_3.py, from v7.2 as built above): no catch-all fields asked (Q1), no twin of a settled
property asked (Q2), hop targets must show each detail themselves (H), next page of the full search read
before a cheapest order (C), compatibility read only from the part's own sentences (P).
v7.4 (make_sabito_v7_4.py, from v7.3 as built above): list-valued menu drops kept per product, a fit
sentence dropped only when another item's code is its subject, short words kept in property names.
v7.5 (make_sabito_v7_5.py, from v7.4 as built above): partner pointer relation, open-before-order gate,
no decline while a verified basket stands.
v7.6 (make_sabito_v7_6.py, from v7.5 as built above): a batch call counts only the full records it returned,
and an unknown price under a spending cap does not hold a decline.
Base: Hawthorn v1 (public agent version 06ae2427, sha fa9b6c4c), the race-9 winner. The logic is the base's, except that
one loop assigns its two attributes directly and one prompt example value is shorter; the documentation is our own.
Entry point: agent_main(pdata). One actor model proposes tool calls; scripted steps run before each actor turn; a one-time
request reader, item and rank checkers, a decline audit and the shopper-line seats are separate model calls."""
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
ASK_RANKINGS = 2
ASK_CHARS = 70

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
ASK_GRACE_STEPS = 3
ASK_AFTER_STEPS = 5

ACT_RULES = """Goal: you assist a single shopper in a simulated store and may act only via the tools provided; finish the errand right in every respect. The score is 1 or 0, and a 1 needs all of the following: the ordered item meets every requirement the shopper wrote, every hidden need the shopper discloses when asked, and every change announced during the errand; the quantity and the budget are right; whenever the request asks for the cheapest (or best) item, that is the item chosen; and each procedural rule in the request is obeyed. If no item in the store meets everything, the right finish is a decline carried out precisely as the request lays down.

1. Know how the store behaves.
- Every tool call, failed or invalid ones too, uses up one of a limited number of steps. Thinking costs nothing.
- One reply may carry several tool calls when none relies on another's result (opening a few listings' details, for instance); they run in order. Orders and declines always go out alone.
- If a details tool takes a LIST of listing ids, it returns the full records of all of them for ONE step (up to its item limit): use it any time you open two or more listings.
- Names of tools, arguments and results vary between stores. Study every tool's description and JSON schema, and use exactly the names and allowed values given there. Let the first results teach you what each tool reveals (price, stock, detailed properties).
- Find out early which category labels this store gives to listings of the requested kind. Of neighbouring labels, the right one is whichever matches the wording of the request taken as a whole; do your searching and buying inside it.
- Search works differently in each store; deduce this store's search habits from its first results. Rows can come best match first or in a fixed id order; a keyword search may demand every word (an added word lowers the total) or accept any word (an added word raises the total, and listings matching more words rank higher); perhaps only a few rows or pages are readable; a filter may operate in one mode only (keyword mode disregards the category or price filter, filter mode disregards the words). Set a reported total against the rows you can see: if it is larger, the fitting or cheapest listing may sit on a later page or past the visible ones. If every word is required, narrow down using more exact words (or by applying the category and price filters with a reduced price limit) until the total fits within the visible pages. Keep queries short (one to three words) and put the most distinctive words first: some stores read just the first word or two and drop the rest, so a long query may in effect search only its first words (compare a longer query's total and rows with those of a shorter one that starts the same way; the notes explain what that reveals). If every word is required, a required value placed before the item kind ("blue chair", value leading) returns exactly the listings that hold that value. If a single word is enough to match, common words (the item kind, a unit) pull in most of the shelf and bury the rest: a required value's distinctive word or figure alone ("herringbone", "30m", "0.4") returns only listings that hold it, so search it alone and in the other spellings listings use ("30 m", "3 atm"), and when its total fits the pages, read all of that search's pages; combine the item kind with values only to raise the closest matches to the top. Keyword searches can also hit category names and property values. After each search, notes summarise what the totals show about how this store searches.

2. Learn the shopper's hidden needs.
- Beyond what is written, the shopper has needs and reveals one only if a message asks about precisely that property. The FIRST thing a message asks gets the most reliable answer, so cover one property per message, at most two with the more likely one first: a short, plain, friendly question naming the property (one or two example values are fine). Leave out instructions, lists, labels, positions and rules.
- An answer such as "don't mind", "no preference" or "anything is fine" leaves the property open. Any preference or limit stated by the shopper becomes a hard requirement from then on, kept in its exact words, numbers included.
- Planned questions go out for you once two or three listings of the kind requested have their details open, so open those early. The answers appear in the conversation together with their meaning; a revealed need is as binding as a stated one: run the search again with it next to the stated values.
- If you ask on your own: first open two or three listings' details, then ask about what the story hints at (who it is for, the look it should have, where or how it gets used), and last about listing properties that vary across candidates. Never ask what the request already settles, and never exceed the budget of messages.

3. Keep the rules of procedure.
- A request may lay down how you work: the number of listings to view or compare, what to check, what to say to the shopper, what a remark has to include, how to decline. These rules are graded just like requirements; follow every one to the letter and completely. "Look at" or "open" N listings means N fully fitting listings, each opened with the details tool.
- When the shopper asks you something (to confirm a change, say), it gets answered for you as long as messages remain; the change itself still binds.
- Plain wishes ("ideally", "would be nice", "if possible") do not count as requirements: they decide only between items that already meet everything, and a cheapest rule always beats them.
- If the request leaves open a point that only the shopper can decide (no price limit, or two wishes that perhaps cannot both be met, so the shopper must rank them), ask before choosing.

4. Follow changes as they come.
- Expect changes from the shopper: a new budget (raised or lowered), a requirement removed, replaced or added, a price rise, an item going out of stock. A single line can carry several changes, and such lines may turn up in any tool result (an extra field with sentences) or be returned with an order attempt. Every change binds from that moment: check the pick and the backups again, and if the cheapest or best is wanted, pick again from the items now acceptable.
- A REFUSED order leaves the episode running and the cart unchanged. Read the refusal, apply each change it contains, remove from the cart anything that no longer fits (a sold-out item, for example), put in the next verified fit, recheck its stock and price, and order again.
- Stock and prices may also shift unannounced, so recheck both right before every order attempt; an item no longer fits once its current price times quantity exceeds the current budget.

5. Handle several products.
- A request may ask for several different products, each having its own quantity, requirements and hidden needs. Cover them all: search every category, verify a fit for each, and buy them together in one order with one line per product at its quantity. A cart tool that takes a list of lines may replace the whole cart with that list, so every such call must contain all products' lines. Cheapest or best is settled product by product.
- A requirement may state its value by reference to another listing or page ("the colour of the jumpsuit sold by X"): locate that listing first (searching for what identifies it, for example its shop or brand with its kind), open its record, read the value off the named field, then search the requested product using that value.

6. Mind the cart.
- Certain cart tools ADD units to a line that is already held instead of setting the count: putting 1 more of a held line in gives 2. Never add a line the cart already has; to change units, add or remove the difference. After every cart call a note follows, giving the cart as the code counts it.

7. Judge each listing (this decides the score).
- For each requirement, the grader checks the listing field holding its property, matching the value's own words as this store writes them in that field. A field counts whenever its name expresses the property, however it is spelled: separators, a prefix, a shorter or longer name, or a well-known alternative name ("neckline" for a collar type, "color family" for a colour, "fit" for a pants fit type, "dial size" for a watch dial size). When a field carries the exact name of the property, that field decides, and so does a field with that name plus one extra word before it (pants length for length, clothing material for material): as soon as one of the two writes the value, the requirement is met. Names that differ only in separators (size class, size_class) belong to a single field: the requirement is met if either writes the value. Lacking such fields, the alternative field decides; if no field at all names the property, the title or the options list may supply the value. Descriptions never count as evidence.
- Values are matched word by word in the store's own writing: differences of case, separators, singular versus plural or spelling do not matter; a narrower value containing the required one meets it ("rtx 3060 12gb" for rtx 3060, "80% cotton" for cotton, "forest green" for green); a single entry of a list is enough, even when the shopper wants that value and no other: a field that offers several variants, one of them this value, satisfies it. Synonyms, abbreviations, related words or shades ("solid" for plain, "win11" for windows 11, ivory or cream for off-white, "tech wood" for technical wood, "embroidered" for embroidery) meet a value only as long as no category listing has the value itself written in that field: when other listings give "matte" as their finish, a "satin" finish does not meet matte. A value joined to another with a plus ("wool + nylon") meets neither part if elsewhere the field holds the other part on its own ("nylon"). FIELD EVIDENCE lines, when present, are derived from this store's records and show which case applies. A different value of that property is a failure (quartz is not mechanical, rosewood is not maple), as is a word that only looks like it. A value that claims to stand for all values of the property ("universal", "all seasons", "suitable for all occasions") in fact names no value, so it fails a required value; but a list also naming the required value still meets it. A field that names a different value does not meet the required one merely because the two often go together or one suggests its partner ("slip-on" is not "lace-up", a "pullover" style is not "zip-up"); the only exceptions are a shade commonly marketed as the required colour and a narrower value containing it.
- A field about a different property is not the property of the requirement (a strap length is not a strap type, a lining material is not the material, a length is not a style): where the category's listings have the property's own field, it may never take its place.
- Only what the shopper stated or revealed counts. Never exclude a listing because of something the shopper never requested: an age group, gender, size, occasion, condition, quality or use inferred from the description or the story. The story is context; if it seems to matter, ask the shopper about that property instead of guessing. Requirements arise from the shopper's replies, never from the questions you ask.
- Numbers are compared as quantities after converting units (4.5 cm is 45 mm); a limit is inclusive; a range passes only when all of it meets the limit. When no field names the property and the title is therefore the evidence, an exact required value is met if the title shows its words and figures in the same order, even when more detail follows (9' x 13' does not show 9' x 12', while 9' x 12' 4" does).
- Figures in a title ("55 x 94 Inch", "117 x 228 cm") do not say which dimension each one is: never reject a listing, or skip opening it, because of them; check the fields its details show.
- Closed lists ("A or B and nothing else") and exclusions ("leave out anything with X") have to be honoured.
- Only the listing's category decides whether it is the requested kind of product: to the grader, a listing filed under the requested category IS that kind, even if its title or description names a different object (a case, a container, a set, a refill, an accessory), therefore never skip, reject or rank it lower for how it looks; weigh nothing but the requirements, price and stock. The category name need not be restated by any property. If the store labels its categories, the item has to sit in the one whose label fits the request's wording as a whole (a request for "A or B" fits a category "A & B" rather than the narrower neighbour "B"). If the request wants goods in stock, the item must show stock, and its price times quantity must stay within the budget (equal passes). A price uses the budget's unit unless the record names another (45 against a budget of 1000 US cents is 45 cents); do not rescale either figure.
- A decline is seldom right. When nothing meets the strict wording but under these rules some listing fulfils all requirements (a narrower value, a field with another name, or a synonym or shade not contradicted by any category listing that writes the value itself), that listing counts as the fit; decline only when, under these rules, the search really found nothing that meets every requirement.

8. Pick the cheapest or the best when asked.
- When the cheapest is wanted, the ordered item has to be the lowest-priced of ALL items that meet every requirement; finding one fitting item does not decide it. If the keyword search wants every word and also matches property values, the value-and-kind search yields the candidates that hold the value: work through all of its pages and open them in order of price; a category listing with a price limit shows prices for the entire category, though not the rows' values: use it to find which cheaper rows exist (lower the limit until the total fits), then inspect the properties of those rows. Search every required value with the item kind as a short query, value leading ("blue chair", "30m watch"), and read every page of it if the readable pages can hold its total (otherwise add one more required value, or a narrower word for the kind); do the same with other names for the kind ("stool", "bench") and with the bare value; where some value is unsearchable (a figure that the store strips out), also search the product line of each fit or near-fit you found, as two words, its maker plus one distinctive title word, all pages: other sizes and colours in a line are separate listings that share its property values and may cost less; sort the candidates you can see by price yourself and open their details beginning with the cheapest; the first whose property fields meet everything, once every cheaper one has been opened and rejected, is the answer. Keep the next fits in reserve as backups.
- For the best by some measure (most, largest, longest, highest...), likewise compare ALL fitting items by that measure and pick the extreme; an item lacking a value for the measure cannot be the winner.

9. Finish cleanly.
- Keep a ranked list of verified fits with at least one backup, as the top pick might be sold out at checkout.
- Before every order attempt: the cart holds exactly one line per requested product (one line for one product), each the chosen item at the required quantity, and nothing else; after your last cart change you have re-read the chosen item's stock and price and found it available and within the budget.
- Decline only once a thorough search has found nothing meeting every requirement, and follow the request's decline rule to the letter. When substitutes must not be offered, the decline remark does not name, describe or point to any listing: no ids, titles or prices.
- Plan your steps: make the first order attempt while at least seven steps are left, so a refusal can still be handled. From 7 remaining steps on, only the cart, stock-check and finishing tools can be used. Decide once; do not switch the cart back and forth.

Return: each reply makes at least one tool call, plus brief text giving the requirements now, each verified fit with its price and stock, and the next move."""

READ_REQUEST = """Goal: study the request and the tools listed below and reply with exactly ONE JSON object, compact (no indentation), and nothing else.
1. Use these keys:

"product": the kind of product wanted, in the request's words (several products: each of them, joined by "and").
"items": one entry for each DIFFERENT product to buy, in order: {"product": its kind in the request's phrasing, "quantity": the number of units of one listing of it (read as for "quantity" below), "requirements": its firm requirements (read as for "requirements" below)}. A single product gives one entry.
"quantity": how many units of one listing to buy, as an integer; 1 unless the request asks for more. Something sold as a single piece (a pair of shoes, a set) is 1. The count goes up only on an explicit wish for several units ("I need three of them", "order 2 packs"); how many people are to share or use it, or the number of old ones there were, is part of the story and leaves the count at 1.
"requirements": a list of strings, one per firm requirement on the product (for several products: the requirements of all of them, each written "property (product): constraint"), each written "property: constraint" and carrying precisely the value, limit, exclusion or closed list that the shopper states. Leave out price, stock, quantity, budgets and plain wishes, and any restatement of the product kind or category: words of the category's own name (such as a season, occasion or audience in it) are covered by the category and form no separate requirement. A requirement joins a property of the product to the value the request insists on ("at least 4 drawers", "the pattern has to be herringbone", "the occasion has to be party", "with a mechanical movement"). The story never makes a requirement: whom it is for, an event at which it is worn or used ("to wear at his wedding", "for the staff party"), where, when or how it is used, how many people share it, or the fact that it replaces an old one ("a new watch" signals a replacement, not a condition). An occasion, a use or an audience is a requirement only when the request makes it a property of the item itself ("the occasion has to be business", "a blazer made for business"). A requirement's value is made of words or figures that the request itself sets down as the item's value; a quality, reason or wish that comes from the story (slim enough for a case, natural and breathable, covers the whole drop, works with what I already have) is only a hint.
"price_cap": the highest total price, given as the exact number the request writes and in the request's own unit (no conversion: "3050 US cents" gives 3050), or null when the request states none.
"currency": the unit the request's prices are in, or null.
"in_stock_only": true when only items in stock or on hand are acceptable.
"priority": "cheapest" only when the request plainly wants the cheapest or lowest-priced item that fits; "best" only when it plainly wants the best by some measure; "none" otherwise. Saying that money is tight or that every penny counts is no request for the cheapest; such remarks only explain the cap.
"priority_detail": the measure if "priority" is "best"; else "".
"message_budget": how many messages the shopper allows you, as an integer, or null when not stated.
"step_budget": how many moves or steps the request allows, as an integer, or null when not stated.
"figures": {"price_cap": the request's words setting the price limit, "message_budget": the words setting the message limit, "step_budget": the words setting the step limit}, each one copied exactly, "" where the request gives no such limit.
"fit_with": each product that the request says must work with, fit or suit another product of the same order, as {"product": the product that must fit, worded as in "items", "with": the product it must fit, worded as in "items", "quote": the request's sentence saying so, copied exactly}. Empty when there is none.
"procedures": every step the request has the assistant take with the shopper that calls for a separate message and does not ask about the products (a go-ahead on the basket before ordering, a report before acting), in the form {"quote": the request's sentence for it, copied exactly, "needs_yes": true when the assistant may act only after the shopper agrees, "final_cart": true when it concerns the finished basket or cart}. Questions about the products, about whom they are for or about anything else worth knowing never count as such a step, even if the request welcomes questions; neither do rules about choosing, about what to buy or about delivery. Empty when there is none.
"rules": each procedural rule the request sets for the assistant (how many listings to look at, what to check or report, the way to decline, what to avoid or include), copied verbatim. Whatever the shopper will do in person is story.
"soft_wants": what the shopper would like without strictly needing it (ideally, preferably, would be nice, if possible). These must never show up in "requirements".
"ask_first": what only the shopper can tell, to be put to them before the choice: the budget if no limit is given, which of two wishes counts for more when the request admits both might not be possible (in that case both wishes belong in soft_wants, not in requirements), or a wish that the request wants cleared with the shopper before any choice (a go-ahead on the basket or on the order goes under "procedures", not here). Empty when there is none.
"hints": details of the story that point at a product property while leaving its value unstated (what it is used for or with, where, by whom, what it must fit, match or withstand), each written "property (product): the detail in a few words" ("property: detail" when just one product is wanted), with the property named as listings would most probably name it. When a detail might concern more than one property, put the most likely first and the next one after a slash ("colour / style (chair): must suit a room with one painted wall"): whatever the item must suit, match or blend with (an outfit, a wall, a room, things used alongside it) and whatever concerns its appearance points first to its colour, next to its pattern or style. Skip a detail that only explains a requirement already given (the reason the stated material or size is required). A job, the weather, travel, a schedule, or any place in which the item is neither worn nor used is just background and never a hint; so is a purpose served by every listing of that kind (storage for a dresser, warmth for a coat). If none, leave it empty.
Every sentence of the request gets a place: as a requirement, the priority, a soft want, a rule, a budget, a hint, or plain story. When unsure whether a sentence is a rule, put it under "rules".
"references": requirements whose value the request gives only by pointing at ANOTHER listing or page ("a colour other than the one on Z's page", "the colour of the jumpsuit X sells", "within 5 of the figure Y's page gives"), each written "property (product): as shown on <how the request names that listing>" (or "property (product): different from the one shown on <how the request names that listing>" when the value must differ from that listing's); put each of them in "requirements" as well, in the same form. Empty when there is none.
"reference_targets": an object per entry in "references" that describes the listing pointed at: {"property": the property in question, "product": the product in question, "seller": the shop or brand that the request gives for that listing, copied exactly, "kind": the kind of product that listing is, "details": [every further value the request states for that listing, short and in the request's own words ("4gb", "blue", "net weight 167.26 g")], "relation": "same", "differ" (NOT the value of that listing: other than, different from, set apart from), "within", "at most" or "at least" (how this value relates to that listing's), "tolerance": the figure for "within", else null, "unit": the unit, else ""}. Empty when there is none.
"decline_no_names": true when the request's decline rule forbids offering, naming or suggesting alternatives or substitutes; false when it wants something named; null when it has no rule of that kind.
"roles": an object giving, for each role, an exact tool name taken from the list, or null when no tool suits it: "search", "details", "view", "message", "cart", "cart_view", "order", "decline". "message" writes to the shopper; "cart" adds things to the cart or removes them from it; "order" buys or checks out; "decline" ends the errand without a purchase.
Return: only that JSON object.
"""

GO_AHEAD = """Goal: before ordering, a shopping assistant presented the final basket to the shopper. Read what the shopper replied and return exactly ONE JSON object, nothing else: {"answer": "change"} if the reply wants something in the basket altered (a different product, quantity or property, or a new price limit); {"answer": "no"} if the reply clearly turns the order down or says not to buy; {"answer": "yes"} in all other cases, because agreement, thanks, an acknowledgement or a comment requesting no change all leave the basket as it is."""

STEP_CHECK = """Goal: the rules below come from a shopper's request to a shopping assistant. Mark the ones that require the assistant to send the shopper a separate message that is no question about the products, e.g. getting the shopper's go-ahead for the basket ahead of ordering. Answer with exactly ONE JSON object, nothing more: {"steps": [the numbers of those rules]}, or an empty list if no rule does."""

ASK_RANK = """# Shopping assistant - this step: choose which properties to ask the customer about
- The customer hides one or two wishes about a product, revealing each only if a question names exactly its property.
- For every product (P1, P2 ...) you get the properties found on the shop's listings for it, each with the count of listings carrying it and a few sample values.
# Rules
- Rank at most eight properties per product, starting with the likeliest to hide an unstated wish.
- A property the request has settled already hides no unstated wish: omit it. Omit identifiers, titles and prices as well.
- Use what the background suggests without saying so: who will use the item, where or how it gets used, how it ought to look or feel, what failed with the old one.
- Any property that such a detail points to goes in "story" and also in "ranked". Without any such detail, "story" is left empty.
- Write property names exactly as listed; one property spelled two ways counts once.
# Answer
- ONE JSON object, nothing more: {"P1": {"ranked": ["property", ...], "story": ["property", ...]}, "P2": {...}}"""

HEAR = """Goal: figure out, for a shopping assistant, what the shopper's most recent words mean.
1. Output ONE JSON object and no other text, using these keys:
"meaning": one short sentence stating what the shopper intends.
"needs": new firm requirements on the product, each written "property: constraint" carrying precisely the value, limit, exclusion or closed list that was stated. After a phrase of comparison (falls short of, nothing less than, at least), words given in place of a figure are themselves the value and get copied in the shopper's wording: "falls short of not available" becomes "property: not available" (a field can truly hold "not available" or "none").
"no_preference": the properties the shopper said they are indifferent to.
"dropped": earlier requirements the shopper has given up.
"new_budget": the new highest total price, as a number expressed in the unit the request's budget uses (use the line's figure if it is in that unit: "13075 cents" gives 13075 for a budget held in cents), or null.
"unavailable": true when the shopper says an item can no longer be bought (sold out, gone, out of stock, button disabled); else false.
"meta_only": true when the line only wants the assistant to ask fewer things at once (one question at a time, less in one go, the first one now and the rest later) and gives no value: nothing is answered so far, hence needs, no_preference and dropped take nothing from it, and no value is named by its figurative wording; else false. A line saying only that the question was not followed or not understood, or asking for it again in shorter or simpler form, is not meta_only.
"requirements_now": the complete present list of the product's firm requirements, each "property: constraint": start from REQUIREMENTS SO FAR, keep each entry the shopper has neither dropped nor replaced, keep just the latest value of a replaced one, and add every new need expressed in the shopper's lines. Add nothing from the request's story that is missing from REQUIREMENTS SO FAR; requirements come solely from the shopper's lines. Omit price, stock, quantity and product category.
"reply": when the shopper puts a question to the assistant or asks for a confirmation, a brief plain answer to send back that conveys the change in substance (for instance confirming the new value); otherwise "".
"references": for each need in "needs" whose value the shopper conveys only through a pointer to another listing or page, that listing: {"property": the property in question, "seller": the shop or brand that the line gives for that listing, copied exactly, "kind": the kind of product that listing is, "details": [every further value the line states for that listing, short and in the line's own words], "relation": "same", "differ" (NOT the value of that listing: other than, different from, set apart from), "within", "at most" or "at least", "tolerance": the figure for "within", else null, "unit": the unit, else ""}; [] when there is none.
2. One line can carry several changes; list each of them. A need about a different property, even a closely related one (seat height versus overall height, strap width versus width), counts as an additional requirement that replaces nothing: an earlier requirement counts as dropped or replaced only if the line says so about that exact property. A replacement appears twice: the old requirement in dropped, the new one in needs. If the shopper names which of two wishes is more important, the preferred one becomes a need. Any budget the shopper gives goes in new_budget. Plain wishes ("ideally", "would be nice") are no needs.
3. Believe the shopper and read them in good faith: homely, figurative or indirect wording that names a value signals a firm wish for that value; a value is excluded only when the shopper clearly says to avoid it. Words that just stress the firmness of a value (strictly, only, locked to, nothing but, sticking with, stays, holding out for, nothing short of) are not part of the value: write just the value words ("material: jute", "color: white"), never "<value> only"; add an exclusion only if the shopper also explicitly rules out blends or other values. A figure joined to a comparison word (under, over, no more than, at least, at most) makes a limit, unit included. A new budget replaces the old one, whether higher or lower, even when presented as good news.
4. When the shopper just says they don't mind, they have no preference, or that anything will do, the property asked about goes in no_preference (if two properties were asked about in the message, only the FIRST). When the reply says only that the question was beyond the shopper ("you've lost me", "could you put it another way", "that was rather dense"), the shopper shows no preference, provided the question named the product that owns the property: add that property to no_preference as well; when several products are requested and the question did not specify which one, the property stays open (no_preference stays empty).
5. When there are several products, write each need as "property (product): constraint", taking the product from the line (a line about "toe style of the women's boots" gives "toe style (women's boots): ...").
6. The assistant's own question never turns into a requirement; only the shopper's own words do, and sample values named in the question are not needs. A "don't mind" frees just the property the assistant's message was about; it never deletes or changes a need given earlier, and when the message asked nothing (it merely reported something), a reply like that alters nothing. When a need is given by pointing at another listing, its value is the pointing wording of the line itself ("finish: unlike the one shown on Y's kettle page"); values the line gives for the other listing describe that listing rather than the product, so they are not needs, drops or freed properties.
Return: nothing but the JSON object."""

READ_RULES = """1. Check every requirement in the listing field holding its property, matching the value's own words in the spelling this store uses there. A field qualifies whenever its name expresses the property, whatever the spelling (separators, a prefix, a name shortened or lengthened, or a common alternative name, e.g. "neckline" for a collar type, "color family" for a colour, "fit" for a pants fit type). A field carrying exactly the property's name decides it, as does one carrying that name with one extra word before it (pants length for length, clothing material for material): once either of them writes the value, the requirement is met. Field names differing only in separators (size class, size_class) count as a single field, so the requirement is met if either one writes the value. Failing that, the alternative field decides it; the options list or the title may show the value only when no field names the property. Descriptions never count.
2. Values are compared word for word in the store's own writing: case, separators, singular, plural and spelling variants make no difference; a narrower value containing the required one meets it ("rtx 3060 12gb" for rtx 3060, "80% cotton" for cotton, "forest green" for green); a single entry of a list is enough, even if the shopper insists on that value only: a field offering several variants passes if it is one of them. Synonyms, abbreviations, related words and shades ("solid" for plain, "win11" for windows 11, ivory or cream for off-white, "tech wood" for technical wood, "embroidered" for embroidery) meet a value only so long as the category has no listing that writes the value itself in that field: when other listings give "matte" as their finish, a "satin" finish does not meet matte. A value joined to another with a plus ("wool + nylon") meets neither part if elsewhere that field carries the other part on its own ("nylon"). FIELD EVIDENCE lines, when present, come from this store's records and show which case applies. Another value of the same property is a failure (quartz is not mechanical, rosewood is not maple), as is a word that merely looks similar. A value that claims to span all values of the property ("universal", "all seasons", "suitable for all occasions") names none of them and so fails a required value; a list naming the required value too still meets it. A field naming a different value does not meet the required one merely because the two often go together or one suggests the other ("slip-on" is not "lace-up", a "pullover" style is not "zip-up"); only a shade usually marketed under the required colour, or a narrower value that contains it, qualifies.
3. Fields about another property never count (a strap length is not a strap type, a lining material is not the material, a length is not a style): such a field cannot stand in for a property that has its own field on the category's listings.
4. Numbers are compared as amounts after unit conversion (4.5 cm is 45 mm); the edge of a limit is inside it; a range passes only when all of it meets the limit. If the title serves as evidence because the property has no field, an exact required value is met when the title shows its words and figures in the same order, even with more detail after them (9' x 13' does not show 9' x 12', but 9' x 12' 4" does).
5. Unless the record names another currency or unit, prices are in the same unit as the request's budget (45 against a budget of 1000 US cents is 45 cents, well inside it); do not rescale the price or the budget.
6. Only the listed requirements count: never fail a listing over an age group, gender, size, occasion, condition, season, quality or use not named in the requirements.
7. The category label by itself settles whether a listing is of the requested kind: one in the requested category counts as that kind, even if its title names another thing (a case, a container, a set, a refill, an accessory); judge it solely on the requirements, price and stock."""

CHECK_ITEM = """Goal: be the final check prior to a shopping assistant committing to an item; the errand is scored 1 only when every requirement below is met by the item. Judge ONLY by the records shown, with these steps:
""" + READ_RULES + """
8. The category meets any words that form part of its own name.
9. Closed lists and exclusions must hold.
10. The item must belong to the requested kind; if listings show a category label, the correct one is the label matching the whole of the request's wording, not a narrower or adjacent one.
11. Price times quantity must not exceed the budget (being equal is fine). An item the records show as out of stock fails.
12. A mere wish ("ideally", "would be nice") flagged as not required is not a requirement.
13. If records disagree (on price or stock, for example), go by the most recent one.
14. Set fits=false only when the item's fields clearly fail a listed requirement under these rules (a FIELD EVIDENCE line stating the item does NOT meet a requirement counts as such a failure), or when neither a field nor (if it has no field) the options list or the title states it; never fail an item over a missing stock or price record.
Return: ONE JSON object, nothing more: {"item": the item's id, "fits": true or false, "problems": [short strings, each naming a requirement not met or not shown]}."""

RANK_CHECK = """Goal: a shopping assistant will soon order an item for a request wanting the PRIORITY; look for any listing able to beat it. The requirements, the item chosen and the listing records seen are given further down.
1. Use the reading rules below (price times quantity inside the budget; not shown as out of stock).
2. Name any listing that may meet every requirement AND beat the chosen item on the priority (for the cheapest, a lower price; for the best, a better value of the measure).
3. Leave out listings shown by the records to fail a requirement, be out of stock or exceed the budget. A ROW's title figures (like 55 x 94) do not reveal which dimension each stands for, so they never rule out a ROW.
READING RULES:
""" + READ_RULES + """
Return: ONE JSON object, nothing more: {"better": [ids, best first, at most 4], "reason": "one sentence"}."""

ASK_PLAN = """Goal: pick the questions a shopping assistant asks a shopper. The shopper keeps needs hidden, each one a value of a single product property, and reveals one only when a message asks about exactly that property; what a message asks first gets the most reliable answer, while questions on anything else just draw a polite "don't mind". Choose no more than QUESTION_COUNT distinct properties, most promising first:
1. Begin with whatever is listed under MUST BE LEARNED FROM THE SHOPPER below and not asked yet, and word it plainly. For the budget, ask for the top amount they are prepared to pay. For a choice between two wishes, give both values along with their properties, e.g. "If one listing can't have both, would you rather have the <value> <property> or the <value> <property>?".
2. Next come properties that really appear in the listings' details below. Read the story first (STORY HINTS below, when present, give its details with the property each one concerns; they lead): the hidden need is usually the property that the story's concrete details concern without ever stating its value. Whatever the user will touch or press, examine up close or in pictures, hear, measure it against or fit it to, or read on its label, and how it was made, points to the matching part or aspect of the item, and so to its property: tags, labels and small print hint at the material, fabric or ingredients; photos and looks hint at the colour, the pattern or the visible part; a space or a body part that it must fit hints at the shape or size of that part; a present for someone else, or for a person barely known, hints at the person it is made for (gender, age group); the way it should feel on the hand, skin or foot hints first at the material or fabric; where, when or in what activity people wear or use it (outdoors, the office, a gym class) hints at its style or type, i.e. the property with values naming uses like these; something that is fastened or adjusted repeatedly hints at its closure or mechanism; the part doing the work inside it hints at its core component. Background covers weather, a job, a schedule, or any place that is not where the item gets worn or used. Put the first question on the property that the story's details point to, under the listings' own name for it. Who the item is made for is one candidate among the rest: ask about it when the story points at another person, or when nothing else is suggested. Then put properties describing the item itself (material, pattern, closure, shape, length, type or style variant, function) ahead of logistics (warranty, condition, pack size, variant or size lists, years).
3. Leave out what the request already settles (its requirements, the price, stock, quantity), properties asked before, near-duplicates that would take up two questions (two kinds of style, two spellings of one property), as well as anything about price, stock, delivery, brand, maker or a product or model name. Ask only about a property that the listings' details below carry as a key, under the listings' own name for it; if a STORY HINTS property matches no key below, it may be asked under its plain name, unless the hint is marked [no key] (enough listings of that product were read without such a field): in that case ask about the key below whose values vary the way the detail describes, or else drop it.
4. Every step-2 question is a single short, plain, friendly sentence of no more than 15 words that asks which <property> they would like for the <item kind>, calling the property what the listings call it (sample values by themselves never reveal a need), and it may end with one or two sample values. Leave out any introduction, progress report, instructions or lists, and never use a catch-all key such as options, variants, a description, a title or a name. Each question aims at ONE property of ONE product (except the step-1 choice between two wishes): never two properties in a question nor properties of two products, because a reply reveals at most one need and a question with two subjects may be left unanswered. Under "second", put the runner-up property of the SAME product (never merely another name for the first), or leave it empty: the assistant may add it to its final message.
5. When SEVERAL products are wanted, any of them may hold the hidden needs: spread the questions so that each product gets one before any gets a second (setting aside a product with no property left), for each product use the properties listed below for its own category, write the property as "property (product)", and name the product in the question ("Which toe style would you like for the boots?").
Return: ONE JSON object and nothing more: {"questions": [{"property": "...", "second": "... or empty", "question": "..."}, ...]}."""

NEXT_ONE = """
6. Only ONE question is sent now; the next is picked after the shopper's reply has been read. REMAINING question(s) are left in total and each should be used: put the most promising single property first (one or two spares may come after it); once a need is revealed, the next question turns to another property and story detail."""

BEFORE_DECLINE = """Goal: a shopping assistant intends to tell the shopper that no item in the store fits, which is rarely the correct ending; test that claim. Compare the listing records below with the requirements using these reading rules (the price has to stay within the budget, and the records must not show the item out of stock):
""" + READ_RULES + """
8. Some listings below are known from search rows alone (their details were never read, so their property fields are unknown). Any such row in the requested category (whatever object the title names) is still unchecked unless its title, price or stock shows it failing a requirement (title figures such as 55 x 94 do not reveal which dimension each is, hence they never show a row failing): list it under "unopened", most likely first (one whose title names a required value, or one of the requested kind with nothing counting against it).
Return: ONE JSON object, nothing more: {"possible_fits": [ids of listings that seem to meet every requirement, best first, at most 5], "unopened": [ids of rows-only listings to open before declining, at most 6], "reason": "one sentence"}."""

REF_HOP = """Goal: a shopper's request takes the PROPERTY of its product from a different store listing, described as: DESCRIPTION. Each record below comes from a listing that could be that one.
1. Answer with ONE JSON object and nothing else, containing:
"target": the id of the record that IS the described listing: sold by that seller, of that kind, and showing every named detail, judged on meaning (a figure in another unit counts when the amount matches); null if none of the records is.
"missing": the details the target lacks ([] if it has all of them).
"go_to": if, for the PROPERTY, the target's own text refers the reader to a DIFFERENT listing (take it from, use, refer to, look at or go by another listing), that listing as {"seller": its shop or brand name as written, "kind": its kind of product, "details": [each value the text gives for it, brief and verbatim: a colour, a size with its unit, a material]}; null otherwise.
"value": absent a go_to, the PROPERTY's value on the target itself, copied the way its record gives it.
"goes_with": values that, according to the target's text, go with, pair with or equal that value (else []).
2. A listing that the text names only as what the target is not ("not the green one from X"), or as a version, size or colour sold elsewhere, gives no go_to; neither does a listing named for another property. A record missing the described details is never the target, no matter what its text claims.
Return: the JSON object."""

DECLINE_WORDS = r"refus|declin|cancel|abandon|reject|give.?up|abstain|withdraw|no.?(?:purchase|order|buy)|nothing|without"
ORDER_WORDS = r"commit|buy|order|purchas|check.?out|pay|confirm|complete|place|accept|proceed"


def finish_choice_kinds(choices: Any, leftover: bool = True) -> dict[Any, str]:
 """Sort the enum values of a finishing argument into "order" and "decline" (v7.2 W1).
    A decline word or a bare none / skip / no / stop / abort / nothing is a decline, an order word an order. With
    leftover (only for a tool already known to finish), one value left beside order values is the decline."""
 kinds: dict[Any, str] = {}
 for choice in choices or []:
  if not isinstance(choice, str):
   continue
  low = choice.lower().strip()
  if re.search(DECLINE_WORDS, low) or re.fullmatch(r"none|skip|no|stop|abort|nothing", low):
   kinds[choice] = "decline"
  elif re.search(ORDER_WORDS, low):
   kinds[choice] = "order"
 rest = [c for c in choices or [] if isinstance(c, str) and c not in kinds]
 if leftover and len(rest) == 1 and "order" in kinds.values() and "decline" not in kinds.values():
  kinds[rest[0]] = "decline"
 return kinds


ENDEARMENTS = ("pet|love|dear|darling|sweetie|sweetheart|sweetpea|hun|hon|petal|poppet|mate|pal|buddy|chuck|ducky|duckie|"
      "my dear|my love|my friend")


def drop_endearments(line: Any) -> str:
 """Remove the terms of endearment a shopper uses to address the assistant from one line (v7.2 E1).
    Only a lowercase one set off by a comma before punctuation or the line end, or a capitalised one opening the
    line, goes; one followed by "or" stays, as it may be a value in a list."""
 text = re.sub(r",\s*(?:%s)\s*(?=[,.;:!?)]|$)(?!,?\s*or\b)" % ENDEARMENTS, "", str(line))
 return re.sub(r"^(\s*)(?:%s),\s*" % "|".join(w.capitalize() for w in ENDEARMENTS.split("|")), r"\1", text)


NUMBER_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
    "ten": 10}
LIST_NOUNS = r"(?:of\s+)?(?:the\s+)?(?:ids?|listings?|items?|products?|records?|entries|pages?)\b"
LIST_COUNT = r"(\d{1,2}|one|two|three|four|five|six|seven|eight|nine|ten)"


def listed_cap(text: Any) -> int | None:
 """The smallest list size a tool description allows (v7.2 W8): "up to N", "at most N", "no more than N",
    "a maximum of N", or "A or B" / "A-B" / "A to B" followed by ids, listings, items, products, records or pages."""
 low = str(text or "").lower()
 caps = []
 for pattern in (r"\b(?:up to|at most|no more than|a maximum of|maximum of)\s+" + LIST_COUNT + r"\s+" + LIST_NOUNS,
     r"\b" + LIST_COUNT + r"\s*(?:or|-|to)\s*" + LIST_COUNT + r"\s+" + LIST_NOUNS):
  for found in re.finditer(pattern, low):
   count = found.groups()[-1]
   caps.append(int(count) if count.isdigit() else NUMBER_WORDS[count])
 caps = [c for c in caps if c >= 1]
 return min(caps) if caps else None


def capped_shape(shape: Any, cap: int | None) -> Any:
 """A copy of a tool's parameter schema whose single string-list argument takes at most cap items (v7.2 W8).
    The schema is returned unchanged without a cap or without exactly one such argument; a smaller maxItems stays."""
 props = shape.get("properties") if isinstance(shape, dict) else None
 if not cap or not isinstance(props, dict):
  return shape
 lists = [key for key, spec in props.items() if isinstance(spec, dict) and spec.get("type") == "array"
    and isinstance(spec.get("items"), dict) and spec["items"].get("type") == "string"]
 if len(lists) != 1:
  return shape
 spec = dict(props[lists[0]])
 most = spec.get("maxItems")
 spec["maxItems"] = min(cap, int(most)) if is_number(most) else cap
 return {**shape, "properties": {**props, lists[0]: spec}}
SORRY = "Sorry, nothing in the store meets all your requirements, so I will not order anything."
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


def as_text(piece: Any, ceiling: int = VIEW_CHARS) -> str:
 """Render any value as compact JSON text for prompts and logs, keeping non-ASCII characters.
    Falls back to str() when the value cannot be serialised; output longer than ceiling
    (VIEW_CHARS by default) is truncated and tagged as cut."""
 try:
  shown = json.dumps(piece, ensure_ascii=False, separators=(",", ":"))
 except (TypeError, ValueError):
  shown = str(piece)
 if len(shown) > ceiling:
  return shown[:ceiling] + "...(cut)"
 return shown


def clipped(piece: Any, chars: int, lvl: int = 0) -> Any:
 """Return a copy of a nested dict/list structure with every string longer than chars truncated.
    Truncated strings end in an ellipsis; anything nested deeper than eight levels becomes None."""
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
 """Extract a JSON object from an LLM reply text, or None when no dict can be parsed.
    Strips a leading code fence, then tries the whole text and the span from the first opening
    brace to the last closing brace; lists and scalars are rejected."""
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
 """Tell whether a value is an int or float, treating booleans as not numeric.
    Used throughout to read prices, counts, budgets and usage costs safely."""
 if isinstance(piece, bool):
  return False
 return isinstance(piece, (int, float))


def number_words() -> dict[str, int]:
 """Map the English words one through ten to their integer values.
    A fresh dict is built per call; used when reading unit counts written out in words."""
 return {word: spot for spot, word in enumerate("one two three four five six seven eight nine ten".split(), 1)}


def id_marks(text_body: Any) -> set[str]:
 """Find identifier-like marks in text: a capitalised word followed by a 3-5 digit number.
    Returns the set of such strings; the run uses it to spot marks that single out one listing."""
 return set(re.findall(r"\b[A-Z][A-Za-z]+ \d{3,5}\b", str(text_body or "")))


def ask_name(key: Any) -> str:
 """Normalise a field key into lowercase words: underscores become spaces, other symbols vanish.
    Used to compare property names when building and merging the question menus."""
 return " ".join(re.findall(r"[a-z0-9]+", str(key).replace("_", " ").lower()))


def ask_menu(listings: list[dict[str, str]], settled: set[str], basket: bool, skipped: list | None = None,
    product: Any = "") -> list[tuple[str, int, list[str]]]:
 """Build the menu of listing properties worth asking the shopper about for one product.
    Counts values per key over read fact sets, dropping settled names, hash-like ids, free text,
    slash-joined variant values and keys duplicating another; returns (key, carriers, samples).
    Basket products favour keys that vary; single products rank by carriers x distinct (max 4).
    v7.3: catch-all keys (catch_all_key) and keys whose ask_twin set is a twin (twin_match, v7.4) of a settled name's
    or a kept key's are left out too, each noted in skipped as (key, reason)."""
 seen: dict[str, dict[str, int]] = {}
 for facts in listings:
  for key, value in facts.items():
   seen.setdefault(key, {})
   seen[key][value] = seen[key].get(value, 0) + 1
 ranked = []
 stated = {twin for twin in (ask_twin(name, product) for name in settled) if twin}
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
  if sum(count for value, count in values.items() if " / " in value) * 2 > carriers:
   continue
  why = catch_all_key(key, [value for value, count in values.items() for _ in range(count)])
  if not why and any(twin_match(ask_twin(key, product), one) for one in stated):
   why = "stated"
  if why:
   if skipped is not None:
    skipped.append((key, why))
   continue
  weight = (distinct > 1, carriers) if basket else (True, carriers * min(distinct, 4))
  samples = [value for value, _ in sorted(values.items(), key=lambda pair: -pair[1])][:ASK_SAMPLES]
  ranked.append((weight, key, carriers, samples))
 ranked.sort(key=lambda entry: entry[0], reverse=True)
 unique: list[tuple[str, int, list[str]]] = []
 names: set[str] = set()
 twins: dict[frozenset, str] = {}
 for _, key, carriers, samples in ranked:
  if ask_name(key).replace(" ", "") in names or any(same_values(listings, key, kept) for kept, _, _ in unique):
   continue
  twin = ask_twin(key, product)
  kin = next((kept for seen, kept in twins.items() if twin_match(twin, seen)), None)
  if kin is not None:
   if skipped is not None:
    skipped.append((key, "twin of " + kin))
   continue
  twins[twin] = key
  names.add(ask_name(key).replace(" ", ""))
  unique.append((key, carriers, samples))
 return unique


def same_values(listings: list[dict[str, str]], one: str, other: str) -> bool:
 """Tell whether two keys look like one property recorded twice on the listings.
    True when at least two listings carry both and every pair of values agrees, ignoring case
    and spacing; ask_menu uses it to drop the duplicate key."""
 shared = [(facts[one], facts[other]) for facts in listings if one in facts and other in facts]
 return len(shared) >= 2 and all(" ".join(a.lower().split()) == " ".join(b.lower().split()) for a, b in shared)


def ask_merge(first: list[str], story: list[str], spread: list[str]) -> list[str]:
 """Order the question queue for one product from the ranking reply and the menu keys.
    The ranker's first picks and story properties come first, mapped back onto menu keys by
    normalised name; the remaining menu keys follow, without repeats."""
 known = {ask_name(key): key for key in spread}
 order = [known[ask_name(key)] for key in first + story if ask_name(key) in known]
 return list(dict.fromkeys(order + spread))


CATCH_ALL = {"features", "options", "specifications", "specs", "details", "highlights", "certifications", "credentials",
    "ingredients", "included", "includes", "accessories", "contents"}
ASK_STOP = {"of", "the", "a", "an", "for", "and", "or", "in", "on", "to", "with", "per", "by", "its", "your"}


def catch_all_key(key: Any, values: Any = None) -> str:
 """Tell why a field is a catch-all that no question should name (v7.3 Q1); "" when it is none.
    "name": a word of the field name is a generic plural container (CATCH_ALL). "list": of the values given, two or
    more exist and more than half hold three or more items split by commas or semicolons."""
 words = re.findall(r"[a-z]+", re.sub(r"([a-z])([A-Z])", r"\1 \2", str(key)).lower())
 if CATCH_ALL & set(words):
  return "name"
 listed = [str(value) for value in values or []]
 lists = sum(1 for value in listed if len([part for part in re.split(r"[,;]", value) if part.strip()]) >= 3)
 return "list" if len(listed) >= 2 and 2 * lists > len(listed) else ""


def _twin_singular(word: str) -> str:
 """Singular form for comparing property names: a final ies becomes y, a final s goes (not after s).
    Words of three letters or fewer stay as they are."""
 if len(word) > 4 and word.endswith("ies"):
  return word[:-3] + "y"
 if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
  return word[:-1]
 return word


def ask_twin(name: Any, product: Any = "") -> frozenset:
 """Word set of a property name for telling spellings of one property apart from other properties (v7.3 Q2).
    Takes the name before any product tag, lowercase and singular, without stop words; the product's own name words
    go too unless nothing else is left. Short words and words holding a digit stay (v7.4: "ac" and "dc" matter);
    twin_match decides which differences count."""
 words = [_twin_singular(word) for word in re.findall(r"[a-z0-9]+", _name_head(name).replace("_", " "))]
 words = [word for word in words if word not in ASK_STOP]
 own = {_twin_singular(word) for word in re.findall(r"[a-z]+", str(product or "").lower())}
 kept = [word for word in words if word not in own]
 return frozenset(kept or words)


def twin_match(one: Any, other: Any) -> bool:
 """Tell whether two ask_twin word sets name one property (v7.4 Q2).
    Yes when both are non-empty and equal, or when one holds the other and every extra word is short and code-like:
    one or two letters, or holding a digit ("fa pattern" and "pattern", "country origin hb" and "country of origin",
    "output voltage v" and "output voltage"). Two different qualifiers stay apart ("ac voltage" and "dc voltage")."""
 if not one or not other:
  return False
 if one == other:
  return True
 if not (one < other or other < one):
  return False
 return all(len(word) <= 2 or re.search(r"\d", word) for word in one ^ other)


FIT_CODE = r"\b(part|item|sku|model|ref|article|product)\b\s*(?:no\.?|number|#)?\s*[:#]?\s*([a-z]{0,6}-?\d[a-z0-9-]{2,})"


FIT_WORDS = r"\b(?:fit|fits|fitted|fitting|work|works|compatible|suit|suits|goes with|pairs with)\b"
FIT_SUBJECT = r"\b[a-z]{1,6}-?\d{3,}[a-z0-9-]*\b"


def fit_statement(card: Any) -> str:
 """Text of a listing card without the sentences that speak for some other item (v7.3 P, v7.4 subject rule).
    The codes its title (or name) field labels, as in "part NE8641", are the listing's own. A sentence is left out
    when a code (FIT_SUBJECT: letters, then three or more digits) that is not its own stands before the sentence's
    first fit word, so that code is the subject ("Part NE8784 also fits: Lubel 852."); a code after the fit word is
    the object of this listing's claim and stays ("Fits host part NE8784 (Lubel 852)."). Sentences without a fit word
    stay, and without a labelled code in the title the whole card text (as_text) is kept."""
 text = as_text(card)
 title = ""
 if isinstance(card, dict):
  title = next((str(value) for key, value in card.items() if isinstance(value, str)
       and re.search(r"title|heading|headline|name", str(key).lower())
       and not re.search(r"brand|shop|seller|store|vendor|maker|manufactur", str(key).lower())), "")
 own = re.findall(FIT_CODE, title, flags=re.I)
 if not own:
  return text
 codes = {code.lower() for _, code in own}
 kept = []
 for piece in re.split(r'(?<!\\)"|(?<=[.!?;])\s+|\\n', text):
  claim = re.search(FIT_WORDS, piece, flags=re.I)
  subject = piece[:claim.start()] if claim else ""
  if not any(code.lower() not in codes for code in re.findall(FIT_SUBJECT, subject, flags=re.I)):
   kept.append(piece)
 return " ".join(kept)


def written_figures(text_body: Any) -> list[float]:
 """List every number written in a text as floats, with thousands commas removed.
    Used to confirm that limits read by the READER seat are figures the request actually writes."""
 return [float(raw.replace(",", "")) for raw in re.findall(r"\d[\d,]*(?:\.\d+)?", str(text_body or ""))]


def plain_words(text_body: Any) -> list[str]:
 """Split text into lowercase alphanumeric tokens, dropping punctuation and spacing.
    A list is returned (order and repeats kept); empty or None input gives an empty list."""
 return re.findall(r"[a-z0-9]+", str(text_body or "").lower())


def request_sentences(text_body: str) -> list[str]:
 """Split the request into trimmed sentences after full stops, question marks, exclamation marks
    and semicolons. Empty pieces are dropped; matching_sentence searches these."""
 return [part.strip() for part in re.split(r"(?<=[.!?;])\s+", text_body or "") if part.strip()]


def matching_sentence(quote: str, text_body: str, share: float = 0.7) -> str:
 """Find the request sentence that contains the largest share of a quote's words.
    Returns it when that share reaches the threshold (0.7 by default), else an empty string;
    used to verify READER quotes for procedures, fit pairs and limits against the request."""
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
 """Collect strings of four or more words from a string or nested lists into collected.
    Visits at most ten items per list, down to nesting level four; dicts are not entered. Used to
    pull shopper lines out of message-tool and other tool results."""
 if lvl <= 4:
  if isinstance(piece, str):
   if len(piece.split()) > 3:
    collected.append(piece)
  elif isinstance(piece, list):
   nested = lvl + 1
   for inner in piece[:10]:
    long_strings(inner, collected, nested)


def walk_dicts(piece: Any, visit, lvl: int = 0, ceiling: int = 6) -> None:
 """Visit every dict in a nested structure depth first, calling visit on each one.
    Lists are entered item by item; recursion stops past ceiling levels (6 by default)."""
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
 """Reduce a property name to its lowercase head: the part before any parenthesised product
    tag, with camelCase boundaries turned into spaces."""
 head = re.split(r"\s*\(", str(name))[0]
 return re.sub(r"([a-z])([A-Z])", r"\1 \2", head).lower()


def _singular(word: str) -> str:
 """Drop a final s from a word longer than three letters unless it ends in ss.
    A cheap singular form used when comparing property name words."""
 if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
  return word[:-1]
 return word


def junk_key(key: Any) -> bool:
 """Tell whether a field name carries no shopper property worth matching or asking about.
    True for brand, shop, category, title, description, price, availability, sku and url style
    names, and for bare names such as name, notes, stock, images, links or id."""
 words = " ".join(re.findall(r"[a-z0-9]+", _name_head(key)))
 if re.search(r"\b(?:brand|shop|seller|store|vendor|manufacturer|category|categories|department|title|headline|"
     r"heading|listing|description|overview|price|cost|amount|availability|sku|url)\b", words):
  return True
 return bool(re.fullmatch(r"(?:name|notes?|stock|in stock|on hand|images?|links?|make|maker|shelf|ids?|id)", words))


def prop_words(name: Any) -> list[str]:
 """Return the alphabetic words of a property name head, each reduced to a singular form.
    Digits and the product tag are ignored."""
 return [_singular(word) for word in re.findall(r"[a-z]+", _name_head(name))]


def prop_key(name: Any) -> str:
 """Join a property name's singular words into one compact key with no spaces.
    Used by same_prop, strict_prop and pinned to compare property names across spellings."""
 return "".join(prop_words(name))


def edits_within(a: str, b: str, most: int) -> bool:
 """Tell whether two strings are within a given number of single-character edits.
    Computes Levenshtein distance row by row, exiting early when the length gap or a whole row
    already exceeds the limit."""
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
 """Loosely decide whether two property names mean the same property.
    Matches equal compact keys, keys of 4+ letters sharing the first three with one or two
    edits, or, when both keys have five or more letters, one key being a prefix of the other."""
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
 """Tell whether the letter at a position can be dropped in a spelling variant.
    Vowels, y, and either letter of a doubled pair count."""
 if word[at] in "aeiouy":
  return True
 if at > 0 and word[at - 1] == word[at]:
  return True
 return at + 1 < len(word) and word[at + 1] == word[at]


def _without_soft(word: str):
 """Yield each variant of a word with exactly one droppable letter removed.
    Feeds spelled_alike when comparing words that differ by a vowel or a doubled letter."""
 for at in range(len(word)):
  if _soft_letter(word, at):
   yield word[:at] + word[at + 1:]


def spelled_alike(lo: str, hi: str) -> bool:
 """Tell whether a longer word is a spelling variant of a shorter one.
    Accepts one swap of two adjacent letters at equal length, or one or two extra vowels or
    doubled letters (colour against color); any other length gap fails."""
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
 """Decide more strictly than same_prop whether two property names are the same property.
    Equal compact keys, a spelling variant with the same first three letters, or one name equal
    to the other plus one trailing word of up to three letters (the shorter having 5+ letters)."""
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
 """Tokenise text into lowercase words and numbers (decimals kept) for field-evidence checks.
    The filler words and, the, a, an and of are removed."""
 found = re.findall(r"[a-z0-9]+(?:\.[0-9]+)?", str(text or "").lower())
 return [word for word in found if word not in CH_SKIP]


def ch_word_in(word: str, pool: list[str]) -> bool:
 """Tell whether a word occurs in a token pool, allowing singular and plural forms.
    Words of six or more letters also match a pool word with the same first two letters and
    one edit; words containing digits must match exactly."""
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
 """Split a requirement value into its alternatives on the word or, each as a ch_words list.
    Stated or revealed tags, quote marks and only, just, exactly, strictly are removed first;
    empty alternatives are dropped."""
 marked = str(value or "").lower()
 bare = re.sub(r"\((?:stated|revealed[^)]*)\)", " ", marked).replace('"', " ")
 bare = re.sub(r"\b(?:only|just|exactly|strictly)\b", " ", bare)
 split = (ch_words(part) for part in re.split(r"\bor\b", bare))
 return list(filter(None, split))


def ch_writes(value: Any, text: Any) -> bool:
 """Tell whether a text writes a requirement value, checking each of its or-alternatives.
    Used by the field-evidence code to see whether a listing field or title shows the value."""
 options = ch_alts(value)
 return ch_match(options, ch_words(text))


def _joined_run(pool: list[str], target: str) -> bool:
 """Tell whether consecutive words of a pool, concatenated without spaces, spell the target.
    Catches values written as one word in one place and as several in another."""
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
 """Tell whether any alternative is found in a token pool: all its words by ch_word_in, or as
    consecutive pool words joined together. For a lone word without digits the joined form is
    tried only when the pool holds a token containing a digit."""
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
 """Looser than ch_writes: tell whether a text comes near any word of a requirement value.
    A word counts if found by ch_word_in, written as the same standalone figure, or sharing a
    three-letter start with a pool word; used to tell near misses from absent values."""
 pool = ch_words(text)
 low = str(text or "").lower()

 def touches(word: str) -> bool:
  """Per-word test for ch_touches against the enclosing text and its token pool.
        Words under three letters only count on an exact or figure match."""
  if ch_word_in(word, pool):
   return True
  if re.fullmatch(r"\d+(?:\.\d+)?", word) and re.search(r"(?<![\d.])" + re.escape(word) + r"(?!\d|\.\d)", low):
   return True
  if len(word) < 3:
   return False
  return any(len(other) >= 3 and (other.startswith(word[:3]) or word.startswith(other[:3])) for other in pool)

 return any(touches(word) for option in ch_alts(value) for word in option)


def ch_parts(text: Any) -> list[str]:
 """Split a field value into lowercase parts on commas, semicolons and vertical bars.
    Blank parts are dropped; when none remain the whole trimmed text is the only part."""
 low = str(text or "").strip().lower()
 pieces = [bit.strip() for bit in re.split(r"[,;|]", low)]
 kept = [bit for bit in pieces if bit]
 if kept:
  return kept
 return [low] if low else []


def ch_sides(part: Any) -> list[str]:
 """Split a value part on plus signs into its sides, such as a two-material combination.
    Returns the sides only when there are at least two, otherwise an empty list."""
 sides = [side.strip() for side in re.split(r"\s*\+\s*", str(part))]
 sides = [side for side in sides if side]
 return sides if len(sides) > 1 else []


def figure_written(fig: str, said: str) -> bool:
 """Tell whether a figure appears as a standalone number in the request text.
    A decimal figure counts on any hit; a whole number counts only when not followed by a word
    of four or more letters. Used by story_guess to accept figures the request wrote."""
 decimal = "." in fig
 for hit in re.finditer(r"(?<![\d.])" + re.escape(fig) + r"(?!\d|\.\d)", said):
  if decimal:
   return True
  unit = re.match(r"\s*([a-z]+)", said[hit.end():])
  if unit is None or len(unit.group(1)) <= 3:
   return True
 return False


def _content_words(text: str) -> list[str]:
 """Return the words of three or more letters in lowercase text that are not linking words.
    Linking words come from LINK_WORDS; the caller lowercases the text."""
 return [word for word in re.findall(r"[a-z]{3,}", text) if word not in LINK_WORDS]


def positive_value(value: Any, name: Any = "") -> bool:
 """Tell whether a requirement value asks for something present rather than an exclusion.
    Yes/no style values and LITERAL_VALUES entries pass; bound phrases are ignored; a clause of
    four content words fails; a negation fails unless content words precede it or it negates
    only the property's own name."""
 low = str(value).lower().strip()
 if re.fullmatch(r"\W*(?:yes|no|true|false|none|required|included|needed)\W*", low):
  return True
 if LITERAL_VALUES:
  plain = " ".join(re.findall(r"[a-z0-9]+", re.sub(r"\((?:stated|revealed[^)]*)\)", " ", low)))
  if plain in LITERAL_VALUES:
   return True
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
 """Find number runs joined by colon, x or slash (16:9, 55 x 94) and split each into figures.
    Returns one list of figure strings per run, single numbers included."""
 runs = re.findall(r"\d+(?:\.\d+)?(?:\s*[:x/]\s*\d+(?:\.\d+)?)*", body)
 return [re.findall(r"\d+(?:\.\d+)?", run) for run in runs]


def value_seen(value: Any, samples: Any) -> bool:
 """Tell whether a requirement value shows up in a text of sample store values.
    True if a content-word stem starts a word in the samples, or a figure run appears with the
    same figures in order; comparison words are stripped from the value first."""
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
 """Return the set of content-word stems (trailing s removed) of a product or label text.
    Used to compare the product tags of requirements."""
 stems = set()
 for word in _content_words(str(text).lower()):
  stems.add(word.rstrip("s"))
 return stems


def kind_tokens(text: Any) -> list[str]:
 """Return the content words of a product or label name, possessive endings removed.
    Curly apostrophes are normalised first; feeds name_pattern and the shelf label scoring."""
 low = re.sub(r"'s\b", "", str(text).lower().replace("’", "'"))
 return _content_words(low)


_PLURAL_SWAPS = (("y", 1, "ies"), ("fe", 2, "ves"), ("ies", 3, "y"), ("ves", 3, "f"), ("ves", 3, "fe"), ("es", 2, ""), ("s", 1, ""))


def plural_forms(word: Any) -> set[str]:
 """Return a word together with its likely plural and singular variants.
    Adds s and es, swaps y/ies and f or fe/ves, strips es or s, and keeps forms of 3+ letters."""
 base = str(word)
 forms = {base, base + "s", base + "es"}
 for ending, cut, added in _PLURAL_SWAPS:
  if base.endswith(ending):
   forms.add(base[:-cut] + added)
 if base.endswith("f") and not base.endswith("fe"):
  forms.add(base[:-1] + "ves")
 return set(form for form in forms if len(form) > 2)


def same_kind(a: Any, b: Any) -> bool:
 """Tell whether two words are equal or one is a singular or plural form of the other.
    Built on plural_forms; the base of kind_in, kinds_within and ch_word_in."""
 if a == b:
  return True
 return str(b) in plural_forms(a) or str(a) in plural_forms(b)


def kind_in(word: Any, pool: Any) -> bool:
 """Tell whether any word in a pool is the same kind word as the given word.
    Plural and singular variants count as equal through same_kind."""
 for other in pool:
  if same_kind(word, other):
   return True
 return False


def kinds_within(words: Any, pool: Any) -> bool:
 """Tell whether every word of one collection has a same-kind match in another.
    Callers test both directions to see whether two names hold the same kind words."""
 for word in words:
  if not kind_in(word, pool):
   return False
 return True


def kind_score(own: Any, words: Any, shared: Any) -> float:
 """Score how well a category label's words cover one product's kind words.
    Each own word found in the label scores 1, or 0.5 when other products of the basket share
    it; used to pick the home shelf of each product."""
 total = 0
 for word in own:
  if kind_in(word, words):
   total += 0.5 if kind_in(word, shared) else 1.0
 return total


def _longest_first(form: str) -> tuple:
 """Sort key placing longer strings first, ties broken alphabetically.
    Lets name_pattern put longer word forms ahead in its regex alternations."""
 return -len(form), form


def name_pattern(name: Any) -> str:
 """Build a regex that finds a product or label name in text in any plural spelling.
    Each content word matches any of its forms, optionally possessive, and words may be
    separated by spaces, apostrophes or hyphens; returns an empty string for no words."""
 def word_pattern(token: str) -> str:
  """Regex for one name word: any of its plural forms, longest first, then a possessive or a
        word boundary."""
  choices = "|".join(re.escape(form) for form in sorted(plural_forms(token), key=_longest_first))
  return r"(?:" + choices + r")(?:'s\b|'|\b)"
 return r"[\s'-]+".join(word_pattern(token) for token in kind_tokens(name))


def one_prop_hint(hint: str) -> str:
 """Rewrite a story hint that names several properties joined by slashes to lead with one.
    The first property keeps the product tag and detail and the rest move into a trailing
    or-list; other hints, and any that fail to parse, come back unchanged."""
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
 """Return the opening word of a hint detail when the request writes it right before the
    product's head kind word (wool before socks), otherwise an empty string.
    Linking words, very short words and words of the product name itself are never returned."""
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
 """Tell whether a value word, or its stem without s or es, appears among the request words.
    Accepts prefix matches either way within a few letters, and one edit with the same first
    letter for words of four or more letters."""
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
 """Return the value side of a property: value requirement, lowercased, minus comparison words.
    For yes/no style values the property name stands in, since that is what the request wrote."""
 head, _, value = req.partition(":")
 low = value.strip().lower()
 if re.fullmatch(r"\W*(?:yes|no|true|false|required|included|needed)\W*", low):
  low = re.split(r"\s*\(", head)[0].lower()
 return re.sub(r"\b(?:" + VALUE_LINKS + r")\b", " ", low)


def _unsaid_runs(body: str, said: str) -> list[str]:
 """Return the multi-figure runs (55 x 94, 16:9) in a value that the request never writes in
    that order; single figures are ignored."""
 runs = []
 for run in re.findall(r"\d+(?:\.\d+)?(?:\s*[:x/]\s*\d+(?:\.\d+)?)*", body):
  figs = re.findall(r"\d+(?:\.\d+)?", run)
  if len(figs) > 1 and re.search(r"(?<![\d.])" + r"\D{0,4}".join(re.escape(f) for f in figs) + r"(?!\d)", said) is None:
   runs.append(run)
 return runs


def story_guess(req: Any, request: str, refs: list[str]) -> list[str]:
 """List the parts of a requirement value that the request text never writes.
    Empty for pointing values, listed references, values whose figures are all written, and
    short values with a written word; a non-empty result marks a value guessed from the story."""
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
 """Tell whether a value uses wording that points at another listing or page.
    Matches REF_WORDS (same as, sold by, shown on, refers and the like) in lowercase."""
 return re.search(REF_WORDS, str(value or "").lower()) is not None


def _drop_plural(token: str) -> str:
 """Strip a plural ending from a token: es after ch, sh, x or z, otherwise the last letter.
    Callers only pass tokens that already look plural."""
 if re.search(r"(?:ch|sh|x|z)es$", token):
  return token[:-2]
 return token[:-1]


def ref_toks(text: Any) -> list[str]:
 """Tokenise text for comparing reference descriptions with listing text.
    Numbers are glued to a following unit word (12 gb becomes 12gb), linking words and single
    letters are dropped, and plural endings are trimmed."""
 def glue(hit):
  """Regex replacement that joins a number to the short word after it, dropping the space or dash.
        The match is kept as written when that word is a linking word."""
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
 """Extract (figure, unit) pairs from text such as 12 gb or 10/12 mm.
    Units are singularised, linking words are not taken as units, and slash-joined figures
    each get the unit; returns whatever was gathered if parsing fails."""
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
 """Reduce a seller or brand name to lowercase letters and digits, dropping a possessive s.
    Used to compare seller names from the request with those on listings."""
 low = re.sub(r"'s\b", "", str(text or "").lower())
 return re.sub(r"[^a-z0-9]", "", low)


_BOUND_RELATIONS = ((r"\b(?:up to|at most|or less|no more than|or lower|or smaller)\b", "at most"),
     (r"\b(?:at least|no less than|or more|or higher|or larger)\b", "at least"))


def ref_relation(value: Any) -> tuple[str, float | None, str]:
 """Classify how a pointing requirement relates to the value it borrows.
    Returns (relation, tolerance, unit): differ, within with its amount and unit, at most,
    at least, or same when no relation wording is found."""
 low = str(value or "").lower()
 if re.search(DIFFER_WORDS, low):
  return "differ", None, ""
 within = re.search(r"\bwithin\s+(\d+(?:\.\d+)?)\s*([a-z%]+)?", low)
 if within is not None:
  unit = within.group(2) or ""
  return "within", float(within.group(1)), unit.strip()
 for pattern, relation in _BOUND_RELATIONS:
  if re.search(pattern, low):
   return relation, None, ""
 return "same", None, ""


PARTNER_WORDS = (r"\bcounterpart\b|\bcompatible with\b|\bgo(?:es)? (?:together )?with\b|\bpair(?:s|ed)? (?:up )?with\b"
     r"|\bpartner(?:s|ed)? with\b|\bplay(?:s|ing)? nicely together\b")
SAME_WORDS = r"\bnowt but\b|\bnothing but\b|\bsame as\b|\bas shown\b|\bas given\b|\bas stated\b|\bmatch(?:es|ed|ing)?\b|\bidentical to\b"
BOUND_TALK = r"\b(?:less|more|fewer|greater|smaller|larger|longer|shorter|higher|lower|least|most|up to|within)\b"


def pointer_relation(text: Any) -> str:
 """The relation that the pointing words nearest the pointed-to listing state (v7.5 a): "partner" (PARTNER_WORDS:
    counterpart, compatible with, goes with, pairs with ...), "differ" (DIFFER_WORDS, "not", "never") or "same"
    (SAME_WORDS: nowt but, same as, as shown, matching ...). The phrase that ends last wins, a differ phrase winning a
    tie ("not the same as"). "" when the text states none, or speaks of a bound (less, more, at least, within ...)."""
 low = str(text or "").lower()
 if re.search(BOUND_TALK, low):
  return ""
 found = []
 for relation, pattern, rank in (("differ", DIFFER_WORDS + r"|\bnot\b|\bnever\b", 2), ("partner", PARTNER_WORDS, 1),
         ("same", SAME_WORDS, 0)):
  found += [(hit.end(), rank, relation) for hit in re.finditer(pattern, low)]
 return max(found)[2] if found else ""


def _seller_hit(body: str, ok: Any):
 """Find the first capitalised word of four or more letters, not at the very start of the
    text, that the ok filter accepts; returns the regex match or None. Used by ref_parse."""
 for hit in re.finditer(r"(?<![A-Za-z'])([A-Z][a-z]{3,})(?:'s)?(?![A-Za-z])", body):
  if hit.start() > 0 and (ok is None or ok(hit.group(1))):
   return hit
 return None


def ref_parse(text: Any, ok: Any = None) -> dict[str, Any] | None:
 """Parse a description of a pointed-to listing into seller, kind and details.
    The seller is the first accepted capitalised name; the kind comes from the words before
    sold by or right after the name; up to six detail phrases follow. Text after an or-bracket
    is ignored; returns None when no seller is found."""
 try:
  body = re.split(r"\(\s*or\b", str(text or ""))[0]
  hit = _seller_hit(body, ok)
  if hit is None:
   return None
  before, rest = body[:hit.start()], body[hit.end():]
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
   kept = [word for word in re.findall(r"[A-Za-z0-9][A-Za-z0-9.%:/&'-]*", chunk) if word.lower() not in LINK_WORDS]
   if kept:
    details.append(" ".join(kept))
  return {"seller": hit.group(1), "kind": kind.strip()[:80], "details": details[:6]}
 except Exception:
  return None


def prop_named(prop: Any, text: Any, exact: bool = False) -> bool:
 """Tell whether a text names a property: its words appear as a consecutive run of tokens.
    The product tag is dropped and camelCase split; s and es plurals match unless exact is set."""
 head = re.sub(r"([a-z])([A-Z])", r"\1 \2", re.split(r"\s*\(", str(prop or ""))[0])
 wanted = re.findall(r"[a-z0-9]+", head.lower())
 tokens = re.findall(r"[a-z0-9]+", str(text or "").lower())
 if not wanted or len(" ".join(wanted)) < 3:
  return False

 def alike(seen: str, word: str) -> bool:
  """Compare one text token with one property word, allowing s or es plurals unless exact.
        Plural tolerance applies only to words of three or more letters."""
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
 """Return the source text that follows a property's words, up to the end of that sentence or
    an or-bracket and at most 500 characters; empty when the property is not found."""
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
 """Pick a record key passing the accept test only when exactly one key passes; else None.
    Used by _add_plain_names so that an ambiguous record gains no alias field."""
 fitting = [key for low, key in names.items() if accept(low, record[key])]
 return fitting[0] if len(fitting) == 1 else None


def _add_plain_names(record: dict[str, Any]) -> None:
 """Add plain price, in_stock and title aliases to a listing record that uses other names.
    Only for records with an id-like value; each alias is copied from a single unambiguous field
    (an amount or fee, an availability boolean, a headline or name) when the plain one is absent."""
 if not any(isinstance(val, str) and "id" in str(key).lower() and re.fullmatch(ID_LIKE, val) for key, val in record.items()):
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
 """Return a deep copy of a world observation with plain-name aliases added to every record.
    Applies _add_plain_names through walk_dicts; values that cannot pass through JSON come back
    as given. Every tool result is passed through it before the run absorbs it."""
 try:
  piece = json.loads(json.dumps(piece))
 except (TypeError, ValueError):
  return piece
 walk_dicts(piece, _add_plain_names)
 return piece


class ArgCheck:
 """Validate and repair tool-call arguments against a tool's JSON schema before sending.
    Coerces simple types (whole floats, numeric or true/false strings), fixes enum case, trims
    maxLength strings, and reports missing, unknown or out-of-range fields as problem strings.
    Rules are keyed by schema keyword in ArgCheck.RULES and recurse into arrays and objects."""
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
  """Return the schema's declared type names as a list.
        A single type string becomes a one-item list; a missing type gives an empty list."""
  declared = schema.get("type")
  if isinstance(declared, str):
   return [declared]
  return list(declared or [])

 @classmethod
 def fits_kind(cls, value: Any, kind: str) -> bool:
  """Tell whether a value matches one JSON schema type name.
        Unknown type names always pass; integer excludes booleans."""
  probe = cls.TYPES.get(kind)
  if probe is None:
   return True
  return probe(value)

 @classmethod
 def fits_any(cls, value: Any, kinds: list[str]) -> bool:
  """Tell whether a value matches at least one of the given schema type names.
        An empty list of types never matches."""
  return any(cls.fits_kind(value, kind) for kind in kinds)

 @staticmethod
 def whole_float(value: Any) -> bool:
  """Tell whether a value is a float holding a whole number, such as 2.0.
        Such values are turned into ints for integer-typed fields."""
  return isinstance(value, float) and value.is_integer()

 @classmethod
 def coerce(cls, value: Any, kinds: list[str]) -> Any:
  """Convert a value toward the declared schema types where a safe conversion exists.
        A fitting value is kept (a whole float becomes int for integer fields); otherwise each
        type's conversion is tried in order, and the value is returned unchanged if none applies."""
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
  """Try one type conversion: whole floats or integer strings to integer, numeric strings to
        number, numbers to string, true/false strings to boolean.
        Returns the TYPES table itself as a sentinel when no conversion applies."""
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
  """Check and repair one value against a schema; the entry point used for tool-call arguments.
        Coerces types, reports a type mismatch at once, then applies each RULES entry whose keyword
        the schema holds; returns (fixed value, problems) with paths such as args.items[0]."""
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
  """Enforce an enum: an exact choice passes, a case-insensitive string match is replaced by
        the listed choice, and anything else yields a must-be-one-of problem."""
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
  """Enforce a const keyword: the value must equal the schema constant, else one problem.
        The value itself is never changed."""
  if value == schema["const"]:
   return value, []
  return value, [f"{where}: must equal {schema['const']!r}"]

 @staticmethod
 def rule_short(value, schema, where):
  """Enforce minLength on strings, reporting a string shorter than the minimum.
        Non-string values pass untouched."""
  least = schema["minLength"]
  if not isinstance(value, str) or len(value) >= least:
   return value, []
  return value, [f"{where}: shorter than {least} characters"]

 @staticmethod
 def rule_trim(value, schema, where):
  """Apply maxLength by cutting an over-long string to the limit instead of reporting it.
        Never yields a problem."""
  if isinstance(value, str):
   value = value[: schema["maxLength"]]
  return value, []

 @staticmethod
 def limit_rule(keyword, breaks, wording):
  """Build a rule function for one numeric bound keyword (minimum, maximum or an exclusive one).
        breaks decides when the bound is violated and wording phrases the problem; see RULES."""
  def check(value, schema, where):
   """Report the value when both it and the schema edge are numbers and the bound is broken.
            The value is returned unchanged."""
   edge = schema[keyword]
   broken = is_number(value) and is_number(edge) and breaks(value, edge)
   return value, ([f"{where}: {wording} {edge}"] if broken else [])
  return check

 @staticmethod
 def rule_few(value, schema, where):
  """Enforce minItems: a list with fewer items than required yields a problem.
        Non-list values pass; the list is never padded."""
  least = schema["minItems"]
  if isinstance(value, list) and len(value) < least:
   return value, [f"{where}: needs at least {least} items"]
  return value, []

 @staticmethod
 def rule_many(value, schema, where):
  """Enforce maxItems: a list longer than allowed yields a problem.
        The list is not shortened."""
  most = schema["maxItems"]
  if isinstance(value, list) and len(value) > most:
   return value, [f"{where}: allows at most {most} items"]
  return value, []

 @classmethod
 def rule_items(cls, value, schema, where):
  """Validate each element of a list against the items schema by calling execute recursively.
        Returns the list of repaired elements and all element problems with indexed paths."""
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
  """Validate a dict against an object schema's properties.
        Reports missing required fields and, when additionalProperties is false, unknown ones;
        declared fields are repaired recursively and unknown fields are kept as they are."""
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
  """Run the object field checks for any schema with a type keyword when the value is a dict.
        Hooked to the type keyword in RULES so that nested objects are always checked."""
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
 """Transport for one episode: LLM calls and world-environment calls through ProxyClient.
    Tracks the deadline clock, LLM spend against the cap (scaled up for multi-product baskets),
    request pacing and backoff, banned and spare models, and the world step counter.
    Flags done, broke (out of credit), no_model and mute stop further work; failures are logged."""
 def __init__(self, bind: dict[str, Any]):
  """Set up the two proxy clients, the clock start, spend and step counters, and model state.
        bind is the environment binding copied into every world call body; the inference client
        is kept as model_list and the environment client as world."""
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
  """Return the seconds left before the episode deadline (DEADLINE after construction).
        Used to size timeouts and to stop work near the end."""
  return DEADLINE - (monotonic() - self.t0)

 def spent_out(self) -> bool:
  """Tell whether LLM spend has reached SPEND_CAP times the current scale.
        Once true, sonnet_too_dear skips every Sonnet model."""
  return self.spend >= SPEND_CAP * self.scale

 def send(self, client: Any, path: str, body: dict[str, Any]) -> Any:
  """Post one request body to a proxy path, keeping a minimum gap between requests.
        The gap widens for SLOW_FOR seconds after a 429 or 503, inference calls also wait out
        model_resume, and no wait runs past the deadline. Returns the raw proxy reply."""
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
  """Compute a retry wait in seconds: exponential from 1 s (2 s on a 429), capped at 8 s.
        A deterministic jitter of 0.75 to 1.25 taken from the seed text, tries and tick spreads
        retries without randomness."""
  spread = (len(seed) * 7 + sum(ord(ch) for ch in seed[-64:]) + 31 * tries + 13 * self.tick) % 101 / 100.0
  first = 2.0 if status == 429 else 1.0
  return min(8.0, first * 2 ** tries) * (0.75 + 0.5 * spread)

 def down(self, at: float | None = None) -> bool:
  """Tell whether the model service looks down.
        True when no model is usable at all, or when failures are outstanding and no reply has
        arrived for DOWN_AFTER seconds before the given moment (now by default)."""
  if self.no_model:
   return True
  if self.misses <= 0:
   return False
  moment = monotonic() if at is None else at
  return moment - self.last_reply > DOWN_AFTER

 def usable(self, model_list: list[str], tools: bool = False) -> bool:
  """Tell whether any model of a seat (spares included) is still callable.
        False when muted or out of models; with tools set, models banned for tool use are skipped."""
  if self.no_model or self.mute:
   return False
  for name in list(model_list) + self.spare_models:
   if name in self.banned or (tools and name in self.tool_banned):
    continue
   return True
  return False

 @staticmethod
 def plain(msgs: list[dict[str, Any]]) -> list[dict[str, Any]]:
  """Flatten messages whose content is a list of parts into one text string each.
        Used to resend a request without cache-control blocks after a 400, 413 or 422."""
  flat = []
  for original in msgs:
   copy = dict(original)
   parts = copy.get("content")
   if isinstance(parts, list):
    copy["content"] = "\n\n".join(str(part.get("text") or "") for part in parts if isinstance(part, dict))
   flat.append(copy)
  return flat

 def lone_model(self, ladder: list[str]) -> bool:
  """Tell whether at most one unbanned model remains in a ladder.
        With a lone model, try_model gives up sooner when the service looks down."""
  return len({name for name in ladder if name not in self.banned}) <= 1

 def sonnet_too_dear(self, pos: int, model_list: list[str], msgs: list[dict[str, Any]], token_cap: int) -> bool:
  """Decide whether to skip a Sonnet model at this ladder position to save spend.
        Always skip once spend is at the cap; otherwise skip only if a non-Sonnet model follows and
        spend is past SOFT_SPEND or the estimated cost of this call would pass the cap."""
  cheaper_next = pos + 1 < len(model_list) and "sonnet" not in model_list[pos + 1]
  guess = sum(len(json.dumps(m, default=str)) for m in msgs) / 2.3 * 2e-6 + min(token_cap, 1500) * 1e-5
  scale = self.scale
  if self.spent_out():
   return True
  return bool(cheaper_next and (self.spend >= SOFT_SPEND * scale or self.spend + guess > SPEND_CAP * scale))

 def query_model(self, msgs: list[dict[str, Any]], model_list: list[str], token_cap: int,
   toolset: list[dict[str, Any]] | None = None) -> dict[str, Any] | None:
  """Run one chat completion down a seat's model ladder (the seat list plus spare models).
        Skips banned, refused and tool-banned models, and Sonnet when too dear; returns the reply
        message dict, or None with call_fail set. Every LLM seat goes through it."""
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
  """Accept a completion: note the reply time, clear misses and add the reported cost to spend.
        Returns the message when it holds content or tool calls; an empty one is logged and noted."""
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
  """Retire a model after a 401, 403 or 404, for this call and usually for the episode.
        A 404 about tools bans it only for tool calls and a moderation 403 is no ban; a 403 listing
        allowed models adopts up to two of them as spares, once, and extends the current ladder."""
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
  """Call one model, retrying soft failures; returns (settled, message) for query_model.
        Sets the timeout from time left, adds tools when given, and resends flattened after a size
        or format error; 402 or credit errors stop all model use, other 4xx refuse the model.
        Soft errors back off via model_resume and give up early when the service looks down."""
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
  """Record why a model call came back empty-handed and return None.
        call_fail becomes empty for blank replies, refused when only hard errors occurred, else blank."""
  if "empty" in kinds:
   self.call_fail = "empty"
  elif kinds == {"hard"}:
   self.call_fail = "refused"
  else:
   self.call_fail = ""
  return None

 @staticmethod
 def cached(msgs: list[dict[str, Any]], mdl: str) -> list[dict[str, Any]]:
  """Mark the first and last messages for prompt caching when calling an Anthropic model.
        Their string content is wrapped as a text part with ephemeral cache_control; conversations
        of two messages or fewer and other providers are sent as they are."""
  if len(msgs) <= 2 or not mdl.startswith("anthropic/"):
   return msgs
  marked = [dict(m) for m in msgs]
  for spot in (0, len(marked) - 1):
   text = marked[spot].get("content")
   if text and isinstance(text, str):
    marked[spot]["content"] = [{"type": "text", "text": text, "cache_control": {"type": "ephemeral"}}]
  return marked

 def consult(self, ask_text: str, token_cap: int, model_list: list[str]) -> dict[str, Any] | None:
  """Send one prompt as a single user message to a seat and parse a JSON object from the reply.
        Returns None when the call fails or the reply holds no object; used by the JSON seats."""
  reply = self.query_model([{"role": "user", "content": ask_text}], model_list, token_cap)
  return json_object((reply or {}).get("content") or "")

 def landed(self, data: dict[str, Any], count: int) -> dict[str, Any]:
  """Take a successful world reply: adopt its step number (or advance by the number of actions)
        and set done when the reply says the episode has finished. Returns the reply."""
  step = data.get("step")
  self.tick = step if isinstance(step, int) else self.tick + count
  if data.get("finished"):
   self.done = True
  return data

 def step_clash(self, raw: str, env_body: dict[str, Any], reused: int) -> int | None:
  """Recover from a 409 step conflict on a world call by fixing the step number.
        A finished episode sets done; an expected step resyncs tick and body; a reused step is
        bumped up to twice. Returns the new reuse count to retry with, or None to give up."""
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
  """Send one action, or several as an action list, to the world at the next step number.
        Retries with backoff up to WORLD_TRIES; 402 sets broke, 409 goes through step_clash, and
        403/422 or repeated 400/401/404 end the attempt. Returns the reply dict or None."""
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
 """Archive of every world observation seen in the episode, with indexes over its records.
    Keeps all id-like values, the ids whose full details were read, and the category label per
    id; builds listing cards, field-name menus and label tallies for the checker prompts and
    for the question planner."""
 def __init__(self):
  """Start an empty archive: observations, seen ids, read ids and the id-to-label map.
        One instance lives on the run as archive."""
  self.items: list[Any] = []
  self.ids: set[str] = set()
  self.read: set[str] = set()
  self.label_of: dict[str, str] = {}

 @staticmethod
 def own_label(d: dict[str, Any]) -> str:
  """Return a record's own category label: a short string under a category, shelf, department
        or aisle key that is not an id key; empty string if none."""
  for key, val in d.items():
   if not isinstance(val, str) or not 0 < len(val.strip()) <= 60:
    continue
   low = str(key).lower()
   if re.search(r"categor|shelf|department|aisle", low) and "id" not in low:
    return val.strip()
  return ""

 @staticmethod
 def own_id(d: dict[str, Any]) -> str | None:
  """Return a record's own identifier: the value of an id or _id key, or of a key ending in id
        whose value looks like an identifier; None when the record has none."""
  for key, val in d.items():
   if not isinstance(val, str):
    continue
   low = str(key).lower()
   if low in ("id", "_id") or (low.endswith("id") and re.fullmatch(ID_LIKE, val)):
    return val
  return None

 @staticmethod
 def price_field(d: dict[str, Any]) -> float | None:
  """Return the first numeric value under a key mentioning price or cost, as a float.
        None when the record shows no such field."""
  for key, val in d.items():
   if is_number(val) and not isinstance(val, bool) and re.search(r"price|cost", str(key).lower()):
    return float(val)
  return None

 def read_id(self, d: dict[str, Any]) -> str | None:
  """Return an id value in the record that belongs to an already-read listing, else None.
        Any key containing id is considered."""
  return next((val for key, val in d.items() if isinstance(val, str) and "id" in str(key).lower() and val in self.read), None)

 def note_record(self, d: dict[str, Any]) -> None:
  """Index one record: collect its id-like values and, when it holds a nested map with two or
        more plain values, mark its own id as read (details opened).
        The first category label seen for the id is kept."""
  for key, val in d.items():
   if isinstance(val, str) and "id" in str(key).lower() and re.fullmatch(ID_LIKE, val):
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
  """Store a world observation and index every record inside it via note_record.
        The run calls it for each tool result and for the opening observation."""
  self.items.append(obs)
  walk_dicts(obs, self.note_record)

 def price(self, thing: str) -> float | None:
  """Return the most recent price recorded for a listing id across all stored observations.
        None when no record carrying that id under an id key shows a price."""
  seen: list[float] = []

  def look(d: dict[str, Any]) -> None:
   """Collect the price of a record that carries the wanted id under an id-named key.
            Visitor for walk_dicts; appends to the enclosing list."""
   if any(isinstance(val, str) and val == thing and "id" in str(key).lower() for key, val in d.items()):
    figure = self.price_field(d)
    if figure is not None:
     seen.append(figure)
  for record in self.items:
   walk_dicts(record, look)
  return seen[-1] if seen else None

 def gather_about(self, piece: Any, thing: str, found: list[str], depth: int) -> None:
  """Recursively collect compact texts of records that hold the given string as a value.
        Each hit is clipped (strings to 400 chars, text to 2500) and kept once; stops at depth 6
        or after eight hits, and does not search inside a matching record."""
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
  """Return the latest record texts that mention a listing id, at most hold (default 4).
        Feeds the item check and the reference checks with the raw records of one listing."""
  found: list[str] = []
  for record in self.items:
   self.gather_about(record, thing, found, 0)
  return found[-hold:]

 def merge_cards(self) -> tuple[dict[str, dict[str, Any]], list[str]]:
  """Merge every record of each known listing id into one card, in first-seen order.
        Only records with four or more fields or a nested map count; ids with a nested map are
        marked read. Returns (cards by id, id order)."""
  merged: dict[str, dict[str, Any]] = {}
  order: list[str] = []

  def fold(d: dict[str, Any]) -> None:
   """Fold one record into the card of the known id it carries, if it is substantial enough.
            Later records overwrite earlier fields of the same card."""
   ident = next((val for key, val in d.items() if isinstance(val, str) and "id" in str(key).lower()
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
  """Render listing cards for checker prompts within a character budget.
        Each card is DETAILS or ROW plus the merged record; ids in first lead, then read ones.
        cheaper_than keeps only listings priced below that item (or unpriced) and skip drops ids;
        when over budget, sold-out cards sink before the cards are packed greedily."""
  merged, order = self.merge_cards()

  def cost(ident: str) -> float | None:
   """Price of one merged card, or None when it shows none.
            Used for the cheaper_than filter."""
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
    """Tell whether any field of a merged card reads as out of stock by Observer.key_stock.
                Used only to push such cards back when the budget is tight."""
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
  """Return a record's category label: its own, or the one learnt earlier for an id it carries.
        Empty string when neither is known."""
  label = self.own_label(d)
  if label:
   return label
  known = self.label_of or {}
  for key, val in d.items():
   if isinstance(val, str) and val in known and "id" in str(key).lower():
    return known[val]
  return ""

 def rich_reads(self, shelf: str = "") -> int:
  """Count read listings whose records look complete, optionally on one category label.
        A record counts with a nested map of three or more fields or, without maps, five or more
        short strings."""
  rich: set[str] = set()
  wanted = shelf.strip().lower()

  def judge(d: dict[str, Any]) -> None:
   """Add a record's read id to the set when it is on the wanted shelf and looks complete.
            Records of other labels are skipped when a shelf is given."""
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
  """Return the category labels of read listings, most often seen first.
        Only records carrying a nested map (a details view) are counted."""
  counts: dict[str, int] = {}

  def tally(d: dict[str, Any]) -> None:
   """Count one read record that carries a nested map under its category label.
            Records without a known label are ignored."""
   if not self.read_id(d):
    return
   label = self.label_for(d)
   if label and any(isinstance(val, dict) for val in d.values()):
    counts[label] = counts.get(label, 0) + 1
  for record in self.items:
   walk_dicts(record, tally)
  return sorted(counts, key=lambda label: -counts[label])

 def menu_scan(self, piece: Any, shelf: str, inside: bool, depth: int, sink) -> None:
  """Walk an observation and pass the records on a given shelf to sink.
        Once a record is on the shelf, everything nested in it is passed too; with no shelf every
        record goes. Stops at depth 6."""
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
  """Tell whether a record belongs to a category label, by a category-like field or label_for.
        The comparison ignores case."""
  for key, val in d.items():
   if isinstance(val, str) and val.strip().lower() == shelf.lower() and re.search(r"categor|shelf|department|aisle", str(key).lower()):
    return True
  return self.label_for(d).strip().lower() == shelf.strip().lower()

 def fact_sets(self, shelf: str = "") -> list[dict[str, str]]:
  """Return the distinct property maps of listings on a shelf: short string fields per record.
        A record counts with five or more short (up to 70 chars) non-junk string fields; the
        result feeds ask_menu when our question planner builds its menu."""
  sets: list[dict[str, str]] = []

  def take(d: dict[str, Any]) -> None:
   """Keep one record's short non-junk string fields when there are five or more and the map
            is new. Called by menu_scan for every record on the shelf."""
   short = {str(key): val for key, val in d.items() if isinstance(val, str) and 0 < len(val) <= 70 and not junk_key(key)}
   if len(short) >= 5 and short not in sets:
    sets.append(short)

  for record in self.items:
   self.menu_scan(record, shelf, False, 0, take)
  return sets

 def prop_menu(self, shelf: str = "") -> tuple[str, int]:
  """Tally field names across listing records on a shelf, with up to five sample values each.
        Search rows count only when a shelf is given; once full records exist, names seen only on
        rows are dropped. Returns the top 28 as bullet lines with counts, plus the records counted."""
  totals: dict[str, int] = {}
  row_totals: dict[str, int] = {}
  samples: dict[str, list[str]] = {}
  tally = {"sets": 0, "mapped": 0}

  def take(d: dict[str, Any]) -> None:
   """Count one record's short non-junk string fields, tallying rows (few fields plus an id) apart.
            Values that look like codes, with digits and dashes or underscores, are skipped."""
   short = [(key, val) for key, val in d.items() if isinstance(val, str) and 0 < len(val) <= 70 and not junk_key(key)]
   if len(short) < 5:
    flat_row = shelf and len(short) >= 2 and not any(isinstance(v, dict) for v in d.values())
    if not flat_row:
     return
   tally["sets"] += 1
   row_like = len(short) < 5 and any(isinstance(val, str) and "id" in str(key).lower() for key, val in d.items())
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
  """List recurring short labels on the store's records for the item check prompt.
        Keeps keys with at least four values, two to eight distinct and one repeated; returns up to
        twelve lines of key: value (xN) | ... text."""
  seen: dict[str, dict[str, int]] = {}

  def count(d: dict[str, Any]) -> None:
   """Count each short string value per key on records of four or more fields.
            Values over 40 characters and digit-bearing codes are skipped."""
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
 """The shopper's requirements as first read from the request and as changed during the talk.
    Holds the READER's reading, stated (written) and revealed (uncovered) needs, dropped ones,
    indifferent properties, the LISTENER's current list and the budget, and renders them for
    prompts. hear() runs the LISTENER seat on new shopper lines; version counts the changes."""
 def __init__(self, wire: Wire, request: str):
  """Start with an empty reading and no changes, bound to the wire and the request text.
        field_hook and ref_hook are attached later by the run (store field values, pointing fix)."""
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
  """Tell whether a value is a whole count of at least one (an int, not a bool).
        Used to accept quantities from the reading."""
  return isinstance(value, int) and not isinstance(value, bool) and value >= 1

 def items(self) -> list[dict[str, Any]]:
  """Return the per-product entries of a multi-product request from the reading.
        Each has product, quantity (1 unless a whole count) and the requirements holding a colon;
        the list is empty when fewer than two products are read."""
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
  """Return the unit count for a single-product request, 1 when unread or for a basket.
        Basket quantities come from items() instead."""
  if self.items():
   return 1
  asked = self.reading.get("quantity")
  return asked if self.whole_count(asked) else 1

 def counts(self) -> list[int]:
  """Return the quantity of each product, or a one-item list for a single product.
        Used to check cart lines against what the order needs."""
  per_product = [entry["quantity"] for entry in self.items()]
  return per_product if per_product else [self.quantity()]

 def items_text(self) -> str:
  """Return the multi-product instruction paragraph that lists each product with its quantity.
        Empty for a single-product request; text_body appends it."""
  products = self.items()
  if not products:
   return ""
  numbered = " | ".join(f"{pos}) {entry['product']} x{entry['quantity']}" for pos, entry in enumerate(products, 1))
  return ("THE REQUEST ASKS FOR SEVERAL PRODUCTS; one line per product goes in the order, at that product's quantity, and "
    "no more; each listing is judged only by the requirements of its own product (the category tells which), "
    "and the cheapest or best rule applies within each product: " + numbered)

 def priority(self) -> str:
  """Return the request's selection rule: cheapest, best or none.
        Any other value read from the request counts as none."""
  rule = self.reading.get("priority")
  return rule if rule in ("cheapest", "best") else "none"

 def props_left(self) -> list[str]:
  """Return the property requirements still in force, without price, budget, quantity or stock.
        Uses the LISTENER's full list when one exists, else stated plus revealed needs, minus drops."""
  if self.now_reqs is None:
   pool = [req for req in self.written + self.uncovered if req not in self.let_go]
  else:
   pool = self.now_reqs
  return [req for req in pool if req not in self.let_go
    and not re.search(r"\bprice\b|budget|quantity|\bstock\b|\bcents?\b", req.lower())]

 def text_body(self) -> str:
  """Render the current requirements as one line of text for prompts.
        Stated and revealed needs are tagged (or the LISTENER list is used); the budget, products
        or quantity, dropped needs and indifferent properties are appended."""
  if self.now_reqs is None:
   shown = [f"{req} (stated)" for req in self.written if req not in self.let_go]
   shown += [f"{req} (revealed by the shopper)" for req in self.uncovered if req not in self.let_go]
  else:
   shown = list(self.now_reqs)
  if shown:
   text = "; ".join(shown)
  elif self.let_go:
   text = "no property requirement is left: every in-stock listing in the requested category fits"
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
  """Return a prompt section telling checkers which requirements the shopper dropped.
        Empty when nothing was dropped; added to the item, rank and decline check prompts."""
  if not self.let_go:
   return ""
  return ("\n\nCHANGES SINCE THE REQUEST (they override its wording): the shopper DROPPED " + "; ".join(self.let_go)
    + ". Count it as gone: it makes no listing fail, rank or win preference.")

 def overview(self, step_count: int, msgs: int | None) -> str:
  """Render the initial request reading as a bullet list for the actor.
        Covers product(s) and quantities, requirements, pointing values, price cap, stock rule,
        priority, soft wants, ask-first items, rules and the message and step limits."""
  got = self.reading
  if not got:
   return ""
  out = ["First reading of the request (check it against the request's text):"]
  if got.get("product"):
   out.append(f"- product: {got.get('product')}")
  products = self.items()
  if not products:
   out.append(f"- quantity: {self.quantity()}")
  else:
   for pos, entry in enumerate(self.items(), 1):
    out.append(f"- product {pos}: {entry['product']}, quantity {entry['quantity']} (one cart line)")
   out.append("- the order is just these lines, one per product; when the cart tool takes a list of lines, each cart call sends the whole set")
  out.extend(f"- stated requirement: {req}" for req in got.get("requirements") or [])
  out.extend(f"- value defined by referring to another listing (open it first, then search by its value): {req}" for req in got.get("references") or [])
  if got.get("price_cap") is not None:
   out.append(f"- price cap (total): {got.get('price_cap')} {got.get('currency') or ''}".rstrip())
  if got.get("in_stock_only"):
   out.append("- only items in stock / on hand")
  if self.priority() == "none":
   out.append("- the request gives no cheapest/best rule: any item that meets every requirement within the budget is right, so spend no steps hunting for the lowest price (unless the request text says otherwise)")
  else:
   detail = f" ({got.get('priority_detail')})" if got.get("priority_detail") else ""
   out.append(f"- among all items that fit, pick the {self.priority()}{detail}")
  out.extend(f"- a wish, NOT a requirement (just a tie-breaker): {want}" for want in got.get("soft_wants") or [])
  out.extend(f"- find out from the shopper before choosing: {item}" for item in got.get("ask_first") or [])
  out.extend(f"- rule: {req}" for req in got.get("rules") or [])
  if msgs is not None:
   out.append(f"- messages to the shopper allowed: {msgs}")
  out.append(f"- steps allowed: {step_count}")
  return "\n".join(out)

 @staticmethod
 def line_names(needs: list[Any], lns: list[str]) -> dict[str, str]:
  """Map needs to versions that use the shopper's own wording of the property name.
        Applies when the name is not written as such but a run of words in the lines begins with
        each of its words; the product tag and value are kept. Returns {old need: renamed need}."""
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
  """Return the texts inside all parentheses of a string, such as product tags.
        Used to compare which product a requirement belongs to."""
  return re.findall(r"\(([^)]*)\)", text)

 def drop_named(self, old_need: str, lns: list[str]) -> bool:
  """Safe wrapper for drop_allowed: any error counts as the drop being allowed.
        Called for LISTENER drops and before restoring stated needs."""
  try:
   return self.drop_allowed(old_need, lns)
  except Exception:
   return True

 def drop_allowed(self, old_need: str, lns: list[str]) -> bool:
  """Decide whether a requirement the LISTENER reports dropped is backed by the shopper's lines.
        Allowed when no stated or revealed need holds that slot or a fresh need replaces it, when a
        one-to-three word phrase of the lines names the property, or when a phrase names a store
        field holding the value (dropped_by_field)."""
  name = str(old_need).split(":", 1)[0]
  tag = self.tag_list(name)

  def same_slot(other: str) -> bool:
   """Tell whether a requirement is on the dropped need's property with a compatible product tag.
            When the dropped need has a tag, the other must carry the same tag or none."""
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
  """Tell whether the shopper dropped a requirement by naming the store field that holds its value.
        Checks concrete, non-pointing values against field values from field_hook through
        phrase_naming_field, and logs the phrase on success."""
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
  """Return a phrase from the lines that names a store field whose values show the given value.
        Phrases naming another held property are skipped; None when no phrase qualifies."""
  for phrase in phrases:
   if any(strict_prop(rival, phrase) for rival in rivals):
    continue
   for key, vals in fields.items():
    if strict_prop(key, phrase) and value_seen(value, " | ".join(vals)):
     return phrase
  return None

 @staticmethod
 def plain_words(text: str) -> str:
  """Join the alphanumeric tokens of already-lowercased text with single spaces.
        Used to compare shopper values with store values word for word."""
  return " ".join(re.findall(r"[a-z0-9]+", text))

 def literal_values(self, answer: dict[str, Any]) -> None:
  """Rewrite LISTENER needs whose word value sits behind a comparison phrase (nothing but X).
        If the bare words equal a value of the matching store field (via field_hook), the need becomes
        property: store value and the words join LITERAL_VALUES so that positive_value accepts them.
        Values with digits, pointing values and plain positive values are left alone."""
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
  """Replace entries of a LISTENER answer's needs and requirements_now by a mapping, in place.
        Entries not in the mapping are kept."""
  answer["needs"] = [swaps.get(n, n) if isinstance(n, str) else n for n in answer.get("needs") or []]
  if isinstance(answer.get("requirements_now"), list):
   answer["requirements_now"] = [swaps.get(r, r) if isinstance(r, str) else r for r in answer["requirements_now"]]

 def pinned(self, prop_name: str, qtext: str | None) -> bool:
  """Tell whether a stated requirement already fixes the property the shopper called indifferent.
        Needs a positive value the request itself wrote (no story guess or reference), the same
        property (or an asked name ending in its name) and a compatible product tag; errors: False."""
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
  """Tell whether the shopper's lines spell out one of the given needs in full.
        True when all property words of three or more letters and all value words are in the lines."""
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
  """Build the LISTENER prompt: the HEAR instructions, the request, requirements so far, up to
        twelve earlier exchanges, the question asked (if any) and the shopper's new lines."""
  parts = [HEAR, "\n\nTHE SHOPPER'S REQUEST (for context):\n", self.request, "\n\nREQUIREMENTS SO FAR:\n", so_far]
  if self.exchanges:
   parts += ["\n\nEARLIER EXCHANGES WITH THE SHOPPER (earliest first):\n", "\n".join(self.exchanges[-12:])]
  if qtext:
   parts += ["\n\nTHE ASSISTANT ASKED:\n", qtext]
  parts += ["\n\nTHE SHOPPER NOW SAID:\n", "\n".join(fresh)]
  return "".join(parts)

 def hold_unheard(self, fresh: list[str], qtext: str | None) -> None:
  """Put shopper lines back for later when the LISTENER could not be reached.
        Un-marks them as caught, removes the exchange just logged, and queues (lines, question) in
        pending with a timestamp so that they are heard once a model answers again."""
  self.caught.difference_update(fresh)
  self.exchanges.pop()
  self.pending = list(self.pending or []) + [(list(fresh), qtext)]
  self.pending_at = monotonic()

 def only_meta(self, answer: dict[str, Any], fresh: list[str], qtext: str | None) -> dict[str, Any]:
  """Neutralise a LISTENER answer flagged meta_only (the shopper only asked for fewer questions).
        Applies when a question was asked and the lines give no value: every change is cleared and
        the question is remembered in meta_last."""
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
  """Rename the answer's needs to the shopper's own property wording using line_names.
        The renames go into needs and requirements_now and are logged."""
  try:
   renames = self.line_names(answer.get("needs") or [], fresh)
  except Exception:
   renames = {}
  if not renames:
   return
  self.swap_needs(answer, renames)
  self.wire.log.append({"kind": "need_renamed", "names": list(renames.values())[:4]})

 def masked_lines(self, answer: dict[str, Any], fresh: list[str], qtext: str | None) -> list[str]:
  """Return the shopper lines with pointing phrases cut out by ref_hook.
        Falls back to the raw lines when no hook is set, it returns nothing, or it raises."""
  if not self.ref_hook:
   return fresh
  try:
   return self.ref_hook(answer, fresh, qtext) or fresh
  except Exception as err:
   self.wire.log.append({"kind": "ref_line_error", "error": f"{type(err).__name__}: {err}"[:300]})
   return fresh

 def note_pointed(self, answer: dict[str, Any], fresh: list[str]) -> None:
  """Queue newly revealed needs whose value points at another listing for the reference desk.
        Each goes to ref_news with the answer's reference info and the shopper's line."""
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
  """Record the properties the shopper does not mind about in indifferent.
        Skips properties already listed, still held as requirements, or pinned by a stated
        requirement (those are counted in pinned_heard instead)."""
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
  """Move LISTENER-reported drops into let_go when the masked lines back them (drop_named).
        Price or budget drops are ignored when a new budget arrived; returns True when something
        was dropped under a cheapest or best priority, meaning the set of fits widened."""
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
  """Repair the LISTENER's full requirement list once it has been taken over.
        Revealed needs whose property is missing and not dropped are added back, and stated ones
        go through restore_stated."""
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
  """Add a stated requirement back to the current list unless the shopper dropped it.
        Skipped when it is let go, already listed, backed as a drop by the lines, or its property
        (with the same product tag) was dropped or is already present in another form."""
  if need in self.let_go or need in self.now_reqs or self.drop_named(need, masked):
   return
  name = need.split(":", 1)[0]
  tag = self.tag_list(name)

  def same_slot(other: str) -> bool:
   """Tell whether a requirement is on the same property with exactly the same product tags.
            Used to spot a dropped or already listed form of the stated need."""
   head = other.split(":", 1)[0]
   return same_prop(name, head) and self.tag_list(head) == tag

  if any(same_slot(old) for old in self.let_go):
   return
  if not any(same_slot(req) for req in self.now_reqs if isinstance(req, str)):
   self.now_reqs.append(need)

 def widened_remark(self) -> str:
  """Return the instruction given to the actor after a drop widens a cheapest or best choice.
        With properties left it asks for a fresh search without the dropped one; with none left it
        asks for a full shelf sweep for the lowest price, naming the automatic sweep when sweepable."""
  if self.props_left():
   return ("a requirement the shopper gave up OPENS the field: rows that failed only on that point pass now, and rows "
     "that no search has surfaced yet may be cheaper than the current pick; search the kind again without it, with short "
     "queries made of each still-required value plus the kind word (value first), read "
     "every page, and pick the cheapest of all that is acceptable now")
  if self.priority() == "cheapest" and self.sweepable:
   return ("no property is required any longer: every in-stock row in the requested category qualifies, "
     "whatever it is (accessories, parts and sets on that shelf included), and the lowest price on the whole shelf "
     "wins. Rows come in relevance or id order, never sorted by price, so further shelf slices are fetched on "
     "your behalf (the listing pages, the words of the label, the object words in titles); should no "
     "'[shelf sweep done]' list follow, sweep the shelf yourself with short queries, reading page one of "
     "each - the category label, every kind word of the label on its own, every item name repeated in the "
     "titles seen - plus the category listing's pages. Then put the lowest-priced in-stock row of the "
     "requested category within budget in the cart as its only line, re-check its stock and price, then order it")
  return ("no property is required any longer: every in-stock row in the requested category qualifies, "
    "whatever it is (accessories, parts and sets on that shelf included), and the lowest price on the whole shelf "
    "wins. Rows come in relevance or id order, never sorted by price: sweep the shelf with short queries, reading page one "
    "of each - the category label, every kind word of the label on its own, every item name repeated in the "
    "titles seen - plus the category listing's pages; then order the cheapest in-stock row")

 def hear(self, lns: list[str], qtext: str | None) -> str:
  """Read new shopper lines with the LISTENER seat and fold the changes into the requirements.
        Skips lines already caught and does nothing with under CLOSING seconds left; applies new
        needs, indifference, drops, a new budget and sell-outs, bumps version on changes and sets
        reply_due for an unprompted question. Returns a Meaning: summary line, or empty."""
  fresh = [line for line in lns if line not in self.caught]
  if not fresh:
   return ""
  self.caught.update(fresh)
  if self.wire.spare() < CLOSING:
   return ""
  so_far = self.text_body()
  plain = [drop_endearments(line) for line in fresh]
  prompt = self.listener_prompt(plain, qtext, so_far)
  self.exchanges.append((f"asked: {qtext} -> " if qtext else "") + "shopper: " + " / ".join(plain))
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
   """Tell whether a requirement shares a word of three or more letters with the conversation.
            That text holds the requirements so far, the exchanges (new lines included) and the question."""
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
   remarks.append("an item went unavailable; remove it from the cart and don't order it")
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
 """The cart as counted by the agent from cart-tool calls and the store's replies.
    Learns the tool's argument shape (a list of lines or a flat id and count), whether a put
    adds to a held line or sets it, and which enum form means put; keeps units per listing id
    and the last cart view, and reports lines or units that do not match the order."""
 def __init__(self, the_tool: str | None, jschema: dict[str, Any]):
  """Bind to the cart tool's name and its JSON schema properties, with nothing counted yet.
        The run builds an empty one at start and rebuilds it once the tool roles are read."""
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
  """Return compact JSON of a call's enum-valued arguments, such as an action mode.
        Compared with the first put's form to tell puts from removals."""
  chosen = {}
  for key, val in call_args.items():
   spec = self.prop_specs.get(key)
   if isinstance(spec, dict) and "enum" in spec:
    chosen[key] = val
  return as_text(chosen)

 def line_spec(self) -> tuple[str, str, str] | None:
  """Find the cart argument that takes a list of lines and the id and count fields of a line.
        Returns (array key, id field, count field), preferring string fields named like id; None
        when the schema has no such list."""
  for key, spec in self.prop_specs.items():
   if not (isinstance(spec, dict) and spec.get("type") == "array"):
    continue
   inner = spec.get("items")
   fields = inner.get("properties") if isinstance(inner, dict) else None
   if not isinstance(fields, dict):
    continue
   texts = [name for name, sub in fields.items() if isinstance(sub, dict) and sub.get("type") == "string" and "enum" not in sub]
   id_fields = [name for name in texts if "id" in name.lower()] or texts
   numbers = [name for name, sub in fields.items() if isinstance(sub, dict) and sub.get("type") in ("integer", "number")]
   if id_fields and numbers:
    return key, id_fields[0], numbers[0]
  return None

 def listed_lines(self, call_args: dict[str, Any]) -> list[tuple[str, float | None]] | None:
  """Read (id, count) pairs from a call's list-of-lines argument; count is None when not numeric.
        Returns None when the tool takes no line list or the call does not use it."""
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
  """Return the (id, count) lines of a cart call, from its line list or its flat arguments.
        Empty when no listing id can be found."""
  listed = self.listed_lines(call_args)
  if listed is not None:
   return listed
  thing = self.aim_flat(call_args)
  if not thing:
   return []
  return [(thing, self.count_arg(call_args))]

 @staticmethod
 def first_put(found: list[tuple[str, float | None]]) -> tuple[str, float | None] | None:
  """Return the first line that puts units in (count missing or positive), else the first line.
        None for an empty list."""
  if not found:
   return None
  putting = [line for line in found if line[1] is None or line[1] > 0]
  return (putting or found)[0]

 def aim(self, call_args: dict[str, Any]) -> str | None:
  """Return the listing id of a cart call's first putting line, or None.
        Tells which item a cart call is about."""
  line = self.first_put(self.lines(call_args))
  return line[0] if line else None

 def aim_flat(self, call_args: dict[str, Any]) -> str | None:
  """Return the first string argument without spaces that is not an enum field: the listing id.
        None when no argument qualifies."""
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
  """Return the unit count of a cart call as a float, or None.
        Line-list calls use the first putting line; flat calls the first numeric non-enum argument,
        falling back to the schema default."""
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
  """Tell whether a cart call puts units in: its enum arguments match the first call's form.
        Every call counts as a put until a form is recorded."""
  return self.put_form is None or self.put_form == self.form(call_args)

 def held(self, thing: str) -> float:
  """Return the units counted for a listing id, 0 when it is not in the cart.
        Reads the counted state only, not the store."""
  return self.units.get(thing, 0.0)

 def settle_view(self, obs: Any, whole: bool = True) -> bool:
  """Adopt the unit counts a reply shows, returning False when it shows none.
        For a partial reply (whole False) naming one line, the other lines keep their counts."""
  seen = self.view_units(obs)
  if seen is None:
   return False
  if not whole and len(seen) == 1:
   kept = {thing: qty for thing, qty in self.units.items() if thing not in seen}
   self.learn(seen)
   self.units.update(kept)
   return True
  self.learn(seen)
  return True

 def book_list(self, call_args: dict[str, Any], obs: Any) -> str:
  """Count a line-list cart call as setting the whole cart to the lines sent.
        Marks the tool as setting rather than adding, then corrects from the reply; returns summary."""
  self.adds = False
  self.units = {}
  for thing, count in self.lines(call_args):
   if count is not None and count <= 0:
    continue
   self.units[thing] = self.units.get(thing, 0.0) + (1.0 if count is None else count)
  self.settle_view(obs)
  return self.summary()

 def booked(self, call_args: dict[str, Any], obs: Any) -> str:
  """Update the counted cart after a cart call and return the summary line.
        Ignores error replies and records the first put form; removals subtract, puts on a held line
        add unless the tool is known to set (pending keeps the check), then the reply corrects it.
        The run calls it for every cart-tool result."""
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
  """Tell whether a reply shows an empty cart for a line: only empty lists and zero numbers, and
        no other counted id in its text. Used when a removal reply has no readable counts."""
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
  """Read unit counts per listing id from a cart reply.
        A record counts when it has an id-like value and exactly one small int (a quantity-named one
        when several); returns None when nothing is read or no known id appears."""
  found: dict[str, float] = {}
  known = set(self.units) | set(self.pending)

  def read_row(d: dict[str, Any]) -> None:
   """Read one record as an (id, units) line when it holds an id and a single count field.
            A field named price is never the count; counts for the same id are summed."""
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
  """Adopt shown unit counts and learn from a pending put whether the tool adds or sets.
        Shown equal to the count sent while a different total was assumed means it sets; more means
        it adds. Clears pending and keeps only positive counts."""
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
  """Record a cart-view result with its step and adopt the counts it shows.
        When it shows none and names no counted id, the cart is taken as empty."""
  self.last_view = obs
  self.viewed_at = tick
  if self.settle_view(obs):
   return
  if isinstance(obs, (dict, list)) and not self.ids_in(obs):
   self.units = {}

 def ids_in(self, obs: Any) -> bool:
  """Tell whether any counted listing id appears in an observation's text.
        Used by viewed before it empties the count."""
  text = as_text(obs)
  return any(thing in text for thing in self.units)

 @staticmethod
 def unit_text(qty: float) -> Any:
  """Format a unit count as an int when whole, else leave it as a float.
        Used for the summary text."""
  return int(qty) if float(qty).is_integer() else qty

 def summary(self) -> str:
  """Return the bracketed cart-as-counted line listing each id with its units, or empty.
        Added to results the actor sees after cart calls and cart views."""
  if not self.units:
   return "[cart as counted] empty."
  held = ", ".join(f"{thing} x{self.unit_text(qty)}" for thing, qty in self.units.items())
  return f"[cart as counted] {held}."

 def faults(self, need_n: int, counts: list[int] | None = None) -> list[str]:
  """List what is wrong with the counted cart against the needed quantity, for the order gate.
        One product: more than one line, or a line with the wrong units and how many to change;
        several products go to basket_faults. An empty cart gives no faults."""
  if not self.units:
   return []
  if counts and len(counts) > 1:
   return self.basket_faults(counts)
  found = []
  if len(self.units) > 1:
   listing = ", ".join(f"{i} x{int(qty)}" for i, qty in self.units.items())
   found.append("the cart holds more than one line (" + listing + "); remove all lines except the chosen one")
   return found
  for thing, qty in self.units.items():
   if abs(qty - need_n) <= 1e-9:
    continue
   change = "take out" if qty > need_n else "put in"
   found.append(f"line {thing} has {int(qty)} units while the request needs {need_n}; {change} {int(abs(qty - need_n))} using the cart tool")
  return found

 def basket_faults(self, counts: list[int]) -> list[str]:
  """List basket problems: the number of lines differs from the number of products, or the sorted
        unit counts differ from the products' quantities. Empty when they match."""
  held = sorted(int(q) for q in self.units.values())
  if len(self.units) != len(counts):
   listing = ", ".join(f"{i} x{int(q)}" for i, q in self.units.items())
   return [f"the cart has {len(self.units)} line(s) ({listing}), yet "
     f"the request wants {len(counts)} products, one line apiece; each needs a line of its own and nothing else"]
  if held != sorted(counts):
   return [f"the lines carry {held} units, yet the products need {sorted(counts)} (each product one line at its quantity)"]
  return []


class Judge:
 """Independent checks the run applies before it commits to an item or gives up on the errand.
    Runs the per-item LLM check (CHECKER seat), builds code-derived field evidence from read
    listings, asks for cheaper or better rivals when a priority applies, and audits a decline
    (AUDITOR seat). It also opens the listings those checks name through scripted read calls.
    Verdicts are memoised in run.rulings; passes go to run.passed as id|needs-version."""
 def __init__(self, run):
  """Keeps the run and sets up the checker's memo state.
        Covers the last audit (fits, ids seen, read count, audit count), the row and field-view caches,
        records from the last scripted read, listings and pages named by checks, and rank_better."""
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
  """Renders the store's field-name menu from archive.prop_menu as a prompt section.
        Each name carries the number of read listings showing it; empty string when there is no menu.
        Appended to the item check, rank check and decline audit prompts."""
  menu, _ = self.run.archive.prop_menu()
  if menu:
   return ("\n\nFIELD NAMES SEEN ON THIS STORE'S LISTINGS (spellings differing only in separators are one name; "
     "each number shows on how many read listings it occurs):\n" + menu)
  return ""

 def _not_met(self, thing: str, why: str) -> str:
  """Formats the failing verdict handed back to the actor for one item.
        Takes the item id and the reason text and closes by telling the actor to pick another listing."""
  return f"An independent check states {thing} does NOT meet every requirement: {why}. Pick another listing."

 def _pointed_records(self, refs: list[str], thing: str) -> str:
  """Looks for already-read listings that the reference requirements of the request may target.
        Takes 4+ letter words from each reference's value part, scans other read listings for them and
        returns the latest record of up to three matches as a prompt section, or an empty string."""
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
  return ("\n\nRECORDS OF READ LISTINGS THAT THE REQUEST POINTS TO (if a requirement refers to another listing, its value "
    "is taken from the named field of that listing):\n" + "\n".join(line for ident in hits for line in archive.about(ident, 1)))

 def _own_notes_text(self, own) -> str:
  """Writes the prompt note warning that the item's own notes may send a property elsewhere.
        Takes (requirement, property, sentence) triples from refdesk.own_redirects and explains that
        the record of that other listing would then settle the requirement, not this item's field."""
  pieces = []
  for req, prop, sent in own:
   pieces.append(f"its notes contain a line that may send {prop} elsewhere: \"{sent[:240]}\". If it does (telling the reader to "
        f"refer to, go by, look to or take {prop} from another listing), grading takes {prop} from "
        f"that listing, and this item's own {prop} field stops deciding '{req}': it is met only when the "
        "other listing's record shows it. A listing named only as a version sold somewhere else, or as something "
        "this item is not, changes nothing.")
  return "\n\nNOTE (code's reading of the notes on this item): " + " ".join(pieces)

 def _ruling_key(self, stamp: str, refs, tracked, second: str, own) -> str:
  """Builds the run.rulings memo key for one item check.
        Starts from the item|needs-version stamp and adds the read-listing count while references are
        in play or unresolved, a marker for re-check text, and the number of own-note redirects."""
  parts = [stamp]
  if refs or any(r.get("status") != "resolved" for r in tracked):
   parts.append(str(len(self.run.archive.read)))
  if second:
   parts.append("second")
  if own:
   parts.append(f"own{len(own)}")
  return "|".join(parts)

 def _rule_out(self, memo: str, thing: str, verdict: str, entry: dict[str, Any]) -> str:
  """Stores a failing verdict under its memo key in run.rulings and logs it as a verify entry.
        Used for failures code decides alone (reference chain, field evidence); returns the verdict."""
  self.run.rulings[memo] = verdict
  self.run.log.append({"kind": "verify", "item": thing, "result": entry})
  return verdict

 def _check_prompt(self, thing, aim_text, second, own, refs, tracked, history, evidence) -> str:
  """Assembles the CHECK_ITEM prompt for one listing.
        Adds the request, changes and current requirements, a per-product note with cart units, the
        reference notes and pointed records, re-check text, the item's record history, the latest
        cart view, field names, code field evidence lines and the recurring category labels."""
  run = self.run
  needs = run.needs
  sections = [CHECK_ITEM, "\n\nTHE SHOPPER'S REQUEST:\n", run.request, needs.changes(),
     "\n\nREQUIREMENTS NOW (stated, revealed, changed):\n", needs.text_body()]
  if needs.items():
   held = run.trolley.held(thing)
   cart_note = f" It is in the cart with {int(run.trolley.held(thing))} units." if held > 0 else ""
   sections.append("\n\nSeveral products are requested and this item is one of them (its category tells which): judge it only "
       "by that product's own requirements and quantity, disregarding what the other products require."
       + cart_note)
  sections.append(run.refdesk.ref_note(thing) or "")
  refs = run.refdesk.ref_untracked(refs, tracked)
  if refs:
   sections.append(self._pointed_records(refs, thing))
  sections.append(second)
  if own:
   sections.append(self._own_notes_text(own))
  sections.append(f"\n\nTHE ITEM TO {aim_text}: {thing}\nRECORDS OF THIS ITEM SEEN IN THE STORE (earliest first):\n"
      + "\n".join(history))
  if run.trolley.last_view is not None:
   sections.append("\n\nLATEST CART VIEW:\n" + as_text(run.trolley.last_view)[:3000])
  sections.append(self.field_names())
  if evidence["lines"]:
   sections.append("\n\nFIELD EVIDENCE (computed by code over the records of this item's category: how the shop "
       "writes each required value, and what this item shows):\n" + "\n".join(evidence["lines"]))
  labels = run.archive.label_lines()
  if labels:
   sections.append("\n\nLABELS RECURRING ON THIS STORE'S LISTINGS (with counts; the item's category label must match "
       "the request's wording as a whole, never a neighbouring label):\n" + labels)
  return "".join(sections)

 def check_item(self, thing: str, aim_text: str, second: str = "") -> str:
  """Vets one listing against all current requirements; returns a failing verdict or empty.
        Skipped when time is short or the near-wording check accepted it; results are memoised.
        Fails without a model call on a differ-reference pointer or field-evidence fails, else asks
        the CHECKER seat; a pass adds id|version to run.passed and stores the price in pass_price."""
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
  memo = self._ruling_key(stamp, refs, tracked, second, own)
  if memo in run.rulings:
   return run.rulings[memo]
  pointer = run.refdesk.ref_pointer_note(thing)
  if pointer:
   return self._rule_out(memo, thing, self._not_met(thing, pointer),
          {"item": thing, "fits": False, "problems": ["reference chain"]})
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
   verdict = (f"An independent check states {thing} does NOT meet every requirement: " + "; ".join(problems)
        + ". Pick a different fully verified fit unless you are sure the check is mistaken.")
   run.rulings[memo] = verdict
   return verdict
  run.rulings[memo] = ""
  if answer:
   run.passed.add(stamp)
   run.pass_price[thing] = run.archive.price(thing)
  return ""

 def ch_rows(self) -> dict[str, dict[str, Any]]:
  """Returns the CodeCloser.code_rows table of known listings, cached until the archive grows.
        Each row holds the listing's category, title, price, stock, words and property fields."""
  count = len(self.run.archive.items)
  cached = self.ch_rows_memo
  if cached and cached[0] == count:
   return cached[1]
  rows = self.run.closer.code_rows()
  self.ch_rows_memo = (len(self.run.archive.items), rows)
  return rows

 def store_fields(self, label: Any = "") -> dict[str, list[str]]:
  """Collects field name -> values from read listings in a requirement's home category.
        Installed as needs.field_hook; label is a requirement head that picks the product when several
        are requested, else the own shelf or the first home label set is used. Returns {} on error."""
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
  """Tells whether the item's notes may redirect the given property to another listing.
        Compares the requirement and property of each refdesk.own_redirects entry with same_prop;
        answers True when the lookup raises, to stay on the safe side."""
  try:
   for got in self.run.refdesk.own_redirects(thing) or []:
    heads = (got if isinstance(got, (list, tuple)) else [got])[:2]
    if any(same_prop(str(h).partition(":")[0], prop) for h in heads):
     return True
   return False
  except Exception:
   return True

 def field_view(self, thing: str) -> dict[str, list[str]]:
  """Returns the code field evidence (lines, rules, fails) for one listing, memoised.
        The cache key holds the archive size, needs version and remaining requirements; errors are
        logged and give an empty view. Near-wording confirmations for the item are applied on top."""
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
  """Applies near-wording confirmations to a field-evidence view and returns a new view.
        Evidence lines whose field matches a confirmed property get a note that the value is met in
        other words, and fails on those properties are dropped."""
  names = [p for p, _, _ in confirmed]

  def matches(field: str, prop: str) -> bool:
   """Treats a store field and a requirement property as one when same_prop or strict_prop agrees.
            Pairs evidence lines and fails with the confirmed properties."""
   return same_prop(field, prop) or strict_prop(field, prop)
  lines = []
  for line in view["lines"]:
   head = re.match(r"- (.+?) = '", line)
   hit = None
   if head:
    hit = next(((p, v, s) for p, v, s in confirmed if matches(head.group(1), p)), None)
   if hit:
    line += (f" A meaning check found its '{hit[2][:80]}' to be '{hit[1][:60]}' (one value, worded differently): it "
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
  """Collapses whitespace in any value and shortens it to n characters (70 by default).
        Overlong text keeps n-3 characters plus three dots; used for values quoted in evidence."""
  text = " ".join(str(text).split())
  if len(text) <= n:
   return text
  return text[: n - 3] + "..."

 def field_facts(self, thing: str) -> dict[str, list[str]]:
  """Builds the field evidence for one listing by comparing it with read peers in its category.
        Needs the item's fields, its category when several products exist, and at least three peers;
        tallies field names and value parts across peers, then runs _fact_for on up to 12 requirements.
        Returns lines for the item check, rules for the rank check and fails that rule the item out."""
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
  """Handles a numeric limit requirement in the field evidence.
        When peers carry a field strictly named for the property on 2+ listings but this item has no
        field naming or sharing a word with it, adds a line and a fail saying the limit is unproven.
        Requirements of a different product slot are skipped."""
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
  result["lines"].append(f"- {prop} = '{clip(value)}': the shop keeps it under {listed}, present on {holding} of "
          f"{len(peers)} read listings in this category. This item lacks that field, so the limit stays unproven, "
          "which fails the item: it does NOT meet it (a figure in some other field does not replace it).")
  result["fails"].append(f"{prop}: it lacks the {named[0]} field that this store's category listings carry, so "
          f"'{clip(value)}' is not proven met")

 @staticmethod
 def _parse_requirement(req):
  """Splits a requirement string into (head, property, value) with stated/revealed tags removed.
        Returns None when the property or value is empty, or when the value refers to some other
        listing or page rather than naming a value."""
  head, _, raw = str(req).partition(":")
  value = " ".join(re.sub(r"\((?:stated|revealed[^)]*)\)", " ", raw).split())
  prop = re.split(r"\s*\(", head)[0].strip()
  if not prop or not value or ref_text(value):
   return None
  if re.search(r"as shown|same as|the page|sold by|redirect|refer", value.lower()):
   return None
  return head, prop, value

 def _field_names_for(self, fact, context):
  """Picks the store field names that carry a requirement's property, most common first.
        Prefers fields strictly named for the property (sets by_name); a value first judged negative
        that such a field holds word for word becomes positive and joins LITERAL_VALUES. Otherwise
        falls back to fields whose values match the value; None when no field can be tied to it."""
  carry, table = context[6], context[7]
  prop, alts = fact["prop"], fact["alts"]

  def heaviest_first(names):
   """Sorts field names by how many peer listings carry them, highest count first.
            Reads the carry tally from the enclosing context and returns a list."""
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
  """Writes the rule line telling the rank check how this store spells one required value.
        Exclusions get only the lead. Otherwise it says whether only literal words pass, titles use the
        word, too few listings (under CH_SYNONYM_MIN) rule out a synonym for now, or a synonym may
        count, quoting the sample values seen."""
  clip = self._clip
  lead, value = fact["lead"], fact["value"]
  if not fact["positive"]:
   return lead
  seen_vals = " | ".join(fact["samples"])
  if fact["writers"]:
   extra = "" if fact["by_name"] else (" As no field bears the property's name, a title writing out the value also suffices, "
            "unless the item's own field sharing the property's name says something else.")
   return (lead + " Here only a value writing those exact words passes (containing them or one list element counts; "
     "a synonym or related word does not)." + extra)
  if fact["naming"]:
   return (lead + " Values met so far: " + seen_vals + f"; '{clip(value)}' appears in the titles or texts of "
     f"listings in this category ({', '.join(fact['naming'])}), so the shop uses this exact word: different "
     "wording counts only after such a listing's record shows this field spells it differently.")
  if fact["on"] < CH_SYNONYM_MIN:
   return (lead + " Values met so far: " + seen_vals + f"; none writes it out, yet this field is present "
     f"on too few read listings to tell whether the shop would write '{clip(value)}' word for word: a same-meaning value "
     "is not shown to meet it yet.")
  return lead + " Values met so far: " + seen_vals + "; none of them writes it out, so a same-meaning value may meet it."

 def _fact_for(self, thing, req, context, result) -> None:
  """Builds the evidence for one requirement of the item and adds it to the result.
        Numeric limits go to _bound_fact; bare figure-plus-unit values, other products' needs and values
        neither positive nor exclusions are skipped. Gathers field names, listings writing the value,
        samples and title mentions, adds a rule line, then calls _fact_line for the item's verdict."""
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
  tail = f"{len(fact['writers'])} other listing(s) put '{clip(value)}' itself in it."
  if not fact["positive"]:
   bare = " ".join(re.sub(EXCLUDE_WORDS, " ", value.lower()).split())
   bare_alts = ch_alts(bare)
   held = {ident for ident, words in others if bare_alts and ch_match(bare_alts, words)}
   if bare_alts:
    tail = (f"{len(held)} other listing(s) write '{clip(bare)}' there, the value that is excluded (any listing writing "
      "it does not meet it).")
  lead = f"- {prop} = '{clip(value)}': the store's field for it is " + " / ".join(f"'{x}'" for x in named[:3])
  if not fact["by_name"]:
   lead += f" (no field is named '{prop}'; this one's values hold the value)"
  fact["lead"] = lead + f", seen on {fact['on']} of the {len(peers)} category listings read; " + tail
  kinds = kind_words(value)
  fact["naming"] = []
  if kinds and not fact["writers"]:
   fact["naming"] = [i for i, r in rows.items() if i != thing and (not shelf or r["shelf"] == shelf)
         and kinds <= r["words"]][:3]
  result["rules"].append(self._rule_text(fact))
  self._fact_line(thing, prop, value, fact["positive"], fact["by_name"], named, fact["writers"], fact["alone"],
      fact["samples"], fact["naming"], fact["on"], fact["lead"], context, result)

 def _qualified_fields(self, fields, named, prop, value):
  """Finds the item's fields whose name is the property's name plus one extra leading word.
        Skips already matched names, empty values and junk keys; returns (name, value) pairs.
        The value argument is not used."""
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
  """Writes the evidence line for an item lacking the field this store uses for the property.
        Credits a longer-named field that carries the value, or the title when the property has no
        field of its own name; otherwise lists up to four related fields, value writers first."""
  clip = self._clip
  if qual_note:
   return lead + " This listing has no such field." + qual_note
  if title_says:
   return (lead + f" The field is missing here and none bears the property's name, but the title "
     f"writes '{clip(value)}': it meets it.")
  short = {w for w in prop_words(prop) if len(w) >= 3}
  related = [(name, v) for name, v in fields.items() if name not in named and str(v).strip() and not junk_key(name)
       and (set(prop_words(name)) & short or ch_writes(value, v))]
  related = sorted(related, key=lambda pair: not ch_writes(value, pair[1]))[:4]
  listed = ""
  if related:
   listed = ": candidate fields for the property: " + "; ".join(f"{name}: '{clip(v, 60)}'" for name, v in related)
  return (lead + " No such field here" + listed + ". Another label for this SAME property is fine; a field "
    "belonging to a different property (a height for a type, a frame material for the material) cannot replace it.")

 def _fact_line(self, thing, prop, value, positive, by_name, named, writers, alone, samples, naming, on, lead, context,
       result) -> None:
  """Decides how the item's own fields stand on one requirement; records the line and any fail.
        Covers notes that redirect the property, exclusions, a missing field, and matching parts that
        join the value with another value peers write alone; _own_field_verdict gives the verdict."""
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
   qual_note = (f" Its {first_name} '{clip(first_value, 60)}' holds it (a longer key for exactly this "
       "property: a single extra word before the property's name still names the same property, just under a fuller "
       "name), so it meets it.")
  lines = result["lines"]
  if self.ch_redirected(thing, prop):
   lines.append(lead + " This listing's notes may refer to another listing for this property, so the field it "
       "has might not settle it.")
   return
  if not positive:
   lines.append(lead + " Here: " + (shown or "no such field") + " (this is an exclusion: the listing must not "
       "hold the excluded value).")
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
   """Returns the other sides of a combined field value that peers write alone in this field.
            Empty unless one side of the part matches the required value."""
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
  """Chooses the verdict line, plus a hard fail when warranted, for a field the item does carry.
        Met: a clean match, a longer-named field or the title deciding. Not met: the value joined with
        another option, or peers writing the value literally (hard fail only for property-named
        fields). Else not yet shown or possibly met by synonym; returns (line, fail)."""
  clip = self._clip
  seen_vals = " | ".join(samples)
  if clean:
   return lead + f" Here: {shown} - it writes '{clip(value)}': it meets it.", ""
  if qual_note:
   return lead + f" Here: {shown}." + qual_note, ""
  if title_says:
   return (lead + f" Here: {shown} - with no field bearing the property's name, the title settles it, and the title writes "
     f"'{clip(value)}': it meets it."), ""
  if joined:
   part, others = joined[0]
   line = (lead + f" Here: {shown} - '{clip(part)}' joins '{clip(value)}' to '{others[0]}', which "
     "other listings hold on its own in that field: it does NOT meet it.")
   fail = ""
   if by_name and not qual_near:
    fail = (f"{prop}: its {named[0]} reads '{clip(part)}', combining {clip(value)} with {others[0]} (which "
      f"this store also writes on its own there), so it is not {clip(value)}")
   return line, fail
  if writers:
   line = (lead + f" Here: {shown} - '{clip(value)}' is absent, while other listings in "
     "this category put it in this field: so it fails (different wording, a synonym or a related word is not "
     "enough when the store writes the value as such).")
   fail = ""
   if by_name and not qual_near and not any(ch_touches(value, v) for _, v in own):
    fail = (f"{prop}: its {named[0]} reads '{clip(own[0][1], 60)}', yet on other listings this store puts "
      f"'{clip(value)}' itself in that field")
   return line, fail
  if naming:
   return (lead + f" Here: {shown} - '{clip(value)}' does not appear. No read listing has put it in "
     "this field yet (values: " + seen_vals + f"), yet the category's listing titles or texts use "
     f"'{clip(value)}' ({', '.join(naming)}), so the shop has its own word for it: "
     "this listing's differing value is not shown to meet it."), ""
  if on < CH_SYNONYM_MIN:
   return (lead + f" Here: {shown}. No read listing of this category has '{clip(value)}' in "
     "this field (values: " + seen_vals + f"); since only {on} read listing(s) carry it, it is too early to "
     f"tell whether '{clip(value)}' would be written word for word, so a same-meaning value is "
     "not shown to meet it yet."), ""
  return (lead + f" Here: {shown}. No read listing of this category has '{clip(value)}' in "
    "this field (values: " + seen_vals + "), hence a same-meaning value may meet it."), ""

 def rank_check(self, thing: str) -> str:
  """Asks the CHECKER seat which seen listings could outdo the item about to be ordered on priority.
        Once per item, needs version and refusal count, for cheapest or best, with time and 4+ steps;
        for cheapest only cheaper cards are shown. Keeps up to 4 ids that are not refused, sold out or
        failing field evidence in rank_better and named; returns advice text or an empty string."""
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
   prompt += ("\n\nAs several products are requested, the chosen item is weighed only against listings of the SAME product "
        "(category); no listing for a different product can ever beat it.")
  cheapest = priority == "cheapest"
  cards = run.archive.cards(20000, thing if cheapest else "", run.refdesk.ref_foreign(), {thing})
  prompt += (f"\n\nTHE CHOSEN ITEM: {thing}\n\nLISTINGS SEEN (a card per listing; DETAILS = full record opened, ROW = seen only "
       "in search rows, its fields unknown: a ROW of the requested category (whatever object its title names) "
       "beating the chosen item and not excluded by its title, price or stock MAY fit and gets named"
       + ("; shown are only listings cheaper than the chosen one" if cheapest else "") + "):\n" + "\n".join(cards))
  prompt += self.field_names()
  rules = self.field_view(thing)["rules"]
  if rules:
   prompt += ("\n\nFIELD EVIDENCE (computed by code over the records of the chosen item's category: how the shop "
        "writes each required value):\n" + "\n".join(rules))
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
  return (f"A ranker thinks these listings may fit as well and beat {thing} on the {what}: " + ", ".join(better) + reason
    + ". Look into them (details, stock, price) before ordering; if all fail, order the current pick.")

 def _unread_hits(self, cap, qty, shelves, hit_shelf) -> list[str]:
  """Lists unread search hits in hit-score order, filtered to the home labels and the budget.
        A hit is dropped when its known category is not a home label or price times qty exceeds cap."""
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
  """Assembles the BEFORE_DECLINE audit prompt from the request, changes and requirements.
        Adds the reference note, the listing cards, the ids opened since the previous audit (to be
        assessed first) and the field-name menu."""
  run = self.run
  sections = [BEFORE_DECLINE, "\n\nTHE SHOPPER'S REQUEST:\n", run.request, run.needs.changes(),
     "\n\nREQUIREMENTS NOW:\n", run.needs.text_body(), run.refdesk.ref_note(),
     "\n\nLISTINGS SEEN (a card per listing; DETAILS = full record opened, ROW = seen only in search rows):\n",
     "\n".join(cards)]
  if fresh:
   sections.append("\n\nOPENED SINCE THE PREVIOUS CHECK (their DETAILS cards above are new: judge them "
       "ahead of the others, field by field, by the reading rules): " + ", ".join(fresh))
  sections.append(self.field_names())
  return "".join(sections)

 def _audit_followups(self, unopened: list[str]):
  """Gathers what the decline audit should send the actor to open next.
        Merges budget-slack rows, top unread search hits and the audit's own unopened picks (max 6);
        with more than 5 steps left adds unread pages of value searches and untried combined searches,
        recording both in named_pages. Returns (unopened, pages, combos)."""
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
  """Writes the actor remark that follows a decline audit.
        Names possible fits with the audit's reason, unread result pages and combined searches as
        ready search calls, and unopened rows, ending with the order to verify before deciding."""
  search = self.run.role('search')

  def calls(found) -> str:
   """Formats search argument sets as calls of the store's search tool, joined by semicolons.
            Each set appears as the tool name followed by its JSON arguments."""
   return "; ".join(f"{search} {json.dumps(p, ensure_ascii=False)}" for p in found)
  parts = []
  if fits:
   reason = f" ({answer.get('reason')})" if answer.get("reason") else ""
   parts.append("A separate review finds that these listings may meet every requirement: " + ", ".join(fits) + reason + ".")
  if pages:
   parts.append("Some searches with required values matched more listings than have been read so far; fits may be sitting on "
       "unread pages: " + calls(pages) + ". Read those pages first (one step returns a whole page of "
       "rows) and open the rows of the requested kind within budget that they list.")
  if combos:
   parts.append("Searches that combine the required values are still untried (in a store that needs every word, each gives back "
       "only the listings holding all of them): " + calls(combos) + ". Run them (the search tool still allows them) "
       "and open the rows of the requested kind within budget that they list.")
  if unopened:
   parts.append("Listings of the requested kind seen only in search rows and never opened, so their fields are still "
       "unchecked: " + ", ".join(unopened) + ". First open their details (several per reply).")
  return " ".join(parts) + " Check them (details, stock, price), buy the one that fits, and decline only if all truly fail."

 def review_decline(self, early: bool = False) -> str:
  """Audits a possible decline with the AUDITOR seat over the cards of every listing seen.
        Runs once, or again (up to two non-early audits) after more reads, newly read cards first.
        Keeps fits free of field-evidence fails plus unopened ids and follow-ups; returns the note or
        an empty string. Called after scripted reads, by decline gates and as an early audit."""
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
  """Lists unread listings to open before a decline: scored search hits, then audit-named ids.
        Drops refused or unknown ids, those over budget at the current quantity, and those whose
        known category is not a home label."""
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
  """Builds the arguments that open one listing with the details tool.
        Reuses the latest logged details call that carried a known id, swapping in the new id; else
        uses the schema's required free-text argument or its only one. None if neither exists."""
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
  """Opens a queue of listings through scripted tool calls the actor did not choose.
        With a batch spec the ids go in list calls of at most its size, else one lookup call per id
        via the reader or details tool. Posts them as an assistant turn, dispatches with run.covering
        set, keeps the new records in last_reads and returns the (call id, text) replies."""
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
  """Pops the pending read queue from run.reads, falling back to the post-add stock queue.
        The fallback sets a cart purpose, recheck on, the chosen stock tool and a stock-check line.
        Returns (queue, from_put)."""
  run = self.run
  reads = run.reads
  queue, reads.open_queue = reads.open_queue, None
  from_put = False
  if not queue and reads.put_queue and not run.wire.done and not run.wire.broke:
   queue, reads.put_queue = reads.put_queue, None
   from_put = True
   tool, reads.put_tool = reads.put_tool, None
   reads.open_recheck, reads.open_purpose, reads.open_tool = True, "cart", tool
   reads.open_say = "I re-check the stock of the line I just put in: an item can sell out right as it enters the cart."
  return queue, from_put

 def _reserve_trim(self, queue: list[str], purpose: str) -> list[str]:
  """Shortens a code-chosen read queue so it does not eat the steps kept for ordering.
        The reserve is intake.order_due before the first order and FIRST_ORDER_BY after; trims only
        when the cart holds units or a verified fit exists, logging reserve_skip. Errors keep it whole."""
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
  """Runs the queued scripted reads (menus, quotas, walks, cart rechecks, named listings).
        Takes the queue and its settings from run.reads, trims for the step reserve, drops fresh cart
        lines from cart rechecks and prefers a batch tool; a batch failing batch_ok gives way to single
        reads. Appends a decline-audit note when due, posts replies; True when anything was sent."""
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
   say = "A last look at the cart lines before ordering." if recheck else "Listings named by the checks get opened before any decline."
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
  """Collects every dict nested inside the given pieces, using walk_dicts.
        Used to scan archived tool results for listing records."""
  found: list[dict[str, Any]] = []
  for piece in pieces:
   walk_dicts(piece, found.append)
  return found

 def last_stock(self, ident: str) -> bool | None:
  """Returns the latest stock state shown for a listing across all archived records.
        True or False from Observer.stock_state; None when no record of it showed stock."""
  watch = self.run.watch
  state = None
  for d in self.dicts_in(self.run.archive.items):
   if watch.row_mark(d) == ident:
    got = watch.stock_state(d)
    state = state if got is None else got
  return state

 def _rank_candidate_ok(self, x: str, version: str, cap) -> bool:
  """Tells whether a rank-check suggestion deserves an item check now.
        It must be read, not yet passed at this version, not refused or sold out, free of field fails,
        and priced so that price times its product's quantity stays within the cap."""
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
  """Item-checks up to two rank-check suggestions, then returns a cheaper verified fit if any.
        Cheapest priority only and while time remains; errors are logged and give an empty string."""
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
  """Finds the cheapest listing that passed the item check and costs less than the given item.
        Cheapest priority only; a candidate must have passed at this version, not be refused or sold
        out, share the item's category when several products exist, and have no field fails."""
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
  """Returns a listing's category label as seen in the records that carry its id.
        Falls back to the archive's stored label, lowercased, or an empty string."""
  watch = self.run.watch
  label = ""
  for d in self.dicts_in(self.run.archive.items):
   for key, val in d.items():
    if isinstance(val, str) and val == thing and "id" in str(key).lower():
     label = watch.shelf_of(d) or label
     break
  if label:
   return label
  return str((self.run.archive.label_of or {}).get(thing) or "").strip().lower()

 def chosen(self) -> str | None:
  """Returns the id of the most recently added cart line, or None when the cart is empty.
        Reads run.trolley.units in insertion order."""
  units = self.run.trolley.units
  return list(units)[-1] if units else None


class Catalogue:
 """Store layout, tool abilities and stock bookkeeping, plus the code's own cart repairs.
    Works out each product's home category labels and merges products sold under one joined label.
    Learns which tools look up listings, show stock or notes, or take id lists, and tracks sold-out
    and stale cart lines. Offers swaps for sold-out lines, loads a cart instead of a decline, runs
    the near-wording check, and picks listings to pre-read before questions go out."""
 def __init__(self, run):
  """Holds the run and initialises the catalogue's caches and bookkeeping.
        Covers sold-out sets, stock history, tool capability tallies and key sets, the home-label memo,
        joined labels, near-check state, pre-read rounds, repair marks and swap suggestions."""
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
  """Collects category labels from archived records that carry a known listing id.
        Labels come from Observer.shelf_of; records lacking an id key with a known id are skipped."""
  archive = self.run.archive
  watch = self.run.watch
  labels: set[str] = set()
  for record in Judge.dicts_in(archive.items):
   label = watch.shelf_of(record)
   if label and any(isinstance(v, str) and v in archive.ids and "id" in str(key).lower() for key, v in record.items()):
    labels.add(label)
  return labels

 @staticmethod
 def _best_labels(kinds: list[set], labels: set[str]) -> list[set[str]]:
  """Picks, for each product's kind tokens, the category labels with the top kind score.
        Words of the other products count as shared when scoring; ties keep every top label, and a
        product with no positive score gets an empty set."""
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
  """Returns the set of home category labels for each requested product (one set if single).
        Scores product kind tokens against the labels seen; cached until the archive grows, with empty
        sets on error."""
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
  """Picks the one home label to keep to when read listings also span other categories.
        Single product only; empty when there is no home set or all read labels are home labels, else
        the home label among those read, or the first home label in sorted order."""
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
  """Searches text for a two-part label written with its halves in order and a joiner between.
        Splits the label on JOINERS; returns the regex match, or None without two usable halves."""
  halves = [p for p in re.split(JOINERS, str(label).lower()) if p.strip()]
  if len(halves) != 2:
   return None
  left, right = name_pattern(halves[0]), name_pattern(halves[1])
  if not left or not right:
   return None
  return re.search(r"\b" + left + r"\s*(?:\band\b|&|\bor\b|/|\+)\s*" + right, text)

 @staticmethod
 def _loose_join(text: str, x: str, y: str, a: str, b: str):
  """Finds a short phrase in the request naming x then y with a gap of three words at most.
        The gap must hold a joiner (and, or, &, /, +, comma) and no digit or bare x; skipped when both
        products a and b are also named elsewhere in the text. Returns the match or None."""
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
  """Finds where the request joins two products into one phrase, for a joined-category merge.
        Returns (phrase, True) when the label's halves appear in order, (phrase, False) for a loose
        join of the two product names, or None, also when both end in the same kind word."""
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
  """Tells whether a logged question line was aimed at one of the requested products.
        True when a bracketed tag in its label matches a product's kind or the question names one;
        merge_joined stops merging once such a question exists."""
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
  """Tells whether two product entries set different values for the same property.
        Values are compared as lowercase letter and digit words; a clash blocks merging them."""
  def pairs(row: dict[str, Any]) -> list[tuple[str, str]]:
   """Lists (property, normalised value) for each requirement of one product entry.
            The property loses any bracketed product tag; the value keeps lowercase letters and digits."""
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
  """Records two-part category labels (split by JOINERS) found on id-bearing records in a result.
        Stored in joined_seen under the lowercase label, keeping the original spelling."""
  seen = self.joined_seen
  for d in Judge.dicts_in([obs]):
   if not any(isinstance(v, str) and "id" in str(k).lower() and re.fullmatch(ID_LIKE, v) for k, v in d.items()):
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
  """Tells whether two products fall on opposite halves of a joined label.
        Words unique to each product must sit only on its own half and each product must touch its
        half; both orderings of the halves are tried."""
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
  """Finds two requested products that one seen joined label covers together.
        They must differ in kind, fit inside the label's words, share a quantity, split across its
        halves, not clash on a property and be joined in the request text. Returns (i, j, label,
        phrase, in_order) or None."""
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
  """Merges requested products that the store sells under one joined category; runs after searches.
        Notes joined labels from the result, then merges up to two pairs, but only before any cart use,
        revealed or waived needs, or product-specific question. Returns the merge remarks."""
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
  """Combines two products' requirements into one list without product tags.
        Strips bracketed tags from each head and drops case-insensitive duplicates, keeping order."""
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
  """Clears caches that depend on the product list, after products are merged.
        Resets the home-label memo, the intake menu memo, dialogue quota reads and walked searches."""
  run = self.run
  self.home_memo = None
  run.intake.menu_memo = {}
  run.talk.quota_reads = {}
  run.walk.walked = set()

 def merge_items(self, i: int, j: int, label: str, phrase: str, in_order: bool = True) -> str:
  """Merges products i and j into one product named by the request's joining phrase.
        Keeps i's quantity and both products' requirements, retags product tags in needs, the request
        reading, references and dialogue question state, bumps needs.version, clears caches, rescales the
        spend cap. Returns a remark with the new requirements, or empty if the merge cannot apply."""
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
   """Tells whether a product tag refers to one of the two products being merged.
            Its kind tokens must sit within the label, touch the pair and not equal another product's kind."""
   tokens = kind_tokens(tag)
   if not tokens or not kinds_within(tokens, label_kinds):
    return False
   if not any(kind_in(t, pair_kinds) for t in tokens):
    return False
   return all(not (kinds_within(tokens, o) and kinds_within(o, tokens)) for o in rest_kinds)

  def swap(hit: Any) -> str:
   """Regex callback that rewrites a bracketed product tag belonging to the merged pair.
            It becomes the new tag when two or more products remain, else it is removed; others stay."""
   if belongs(hit.group(1)):
    return f" ({new_tag})" if new_tag else ""
   return hit.group(0)

  def retag(text: Any) -> Any:
   """Rewrites the product tags found in the head of a requirement-like string, before its colon.
            Values that are not strings come back unchanged."""
   if isinstance(text, str):
    head, colon, tail = text.partition(":")
    return re.sub(r"\s*\(([^()]*)\)", swap, head) + colon + tail
   return text

  def retag_list(values: list[Any]) -> list[Any]:
   """Retags every value of a list and drops duplicates that retagging creates.
            Keeps the original order."""
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
  return (f"In this store '{label}' is a SINGLE category and the request seeks one product in it ('{phrase}'), so the order "
    + tail + ", not a line per kind that its name combines. Requirements now: " + needs.text_body())

 def lookup_key(self, tool: str | None) -> str | None:
  """Finds the single free-text argument through which a tool takes a listing id.
        Uses the required non-enum string, or the only one; None when other arguments are required."""
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
  """Lists tools that can look up one listing by id through a lookup_key argument.
        Excludes the tools holding the order, decline, search, message, cart and cart-view roles."""
  run = self.run
  busy = {run.role(r) for r in ("search", "message", "cart", "cart_view", "order", "decline")} - {None}
  return [name for name in run.shapes if name not in busy and self.lookup_key(name)]

 def lookup_call(self, ident: str, tool: str | None) -> dict[str, Any] | None:
  """Builds the arguments to look up one listing with a given tool.
        The details tool goes through Judge.details_call; others reuse the latest logged call of that
        tool with a known id, else {key: id} from lookup_key. None without a tool or key."""
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
  """Returns a record's listing id: a key named exactly id first, then any key ending in id.
        The value must be an id-like string; None otherwise."""
  for exact in (True, False):
   for key, val in d.items():
    name = str(key).lower()
    named = name == "id" if exact else name.endswith("id")
    if named and isinstance(val, str) and re.fullmatch(ID_LIKE, val):
     return val
  return None

 @staticmethod
 def asked_ids(call_args: Any) -> set[str]:
  """Collects the id-like strings a tool call passed, directly or inside list arguments.
        Tells records the call looked up apart from rows it merely listed."""
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
  """Records one stock observation of a listing and updates the sold-out set.
        Sources are lookup or row, stamped with the cart epoch: disagreement in one epoch sets rows_lag,
        a changed row state sets rows_live. Sold out adds to dead (newly dead cart lines go in newly);
        in stock clears it, unless a lookup set it and this is only a row."""
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
  """Tells whether a record nests a dict with at least two scalar values (a property map).
        Shows that a lookup tool returns a listing's full properties."""
  for v in record.values():
   if isinstance(v, dict) and sum(1 for x in v.values() if isinstance(x, (str, int, float))) >= 2:
    return True
  return False

 def learn_caps(self, call_name: str, call_args: Any, obs: Any) -> list[str]:
  """Learns from one tool result what the tool shows, and books every stock state seen in it.
        Tallies per tool whether looked-up records gave the id, stock, a property map, a price or long
        notes absent from search rows; other rows feed row_keys and row_text. Runs on every result
        and returns the cart lines it newly found sold out."""
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
  """Picks the lookup tool used to read a listing's stock.
        Prefers the tool whose looked-up records most often showed stock; else view, a guessed view or
        details unless it read 2+ ids with no stock, else an untried tool not guessed to be a cart,
        order, decline or message tool. None if nothing fits."""
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
  """Picks a lookup tool, other than details, whose records carry long text keys details lacks.
        The one with the most such note sightings wins; None when there is none."""
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
  """Returns a copy of the set of listings currently believed sold out.
        The set is kept by _book_stock and by refusal handling."""
  return set(self.dead or set())

 def stale_lines(self) -> list[str]:
  """Lists cart lines whose stock was not read since the current cart epoch.
        Compares each line's run.line_seq mark with run.cart_seq."""
  run = self.run
  seqs = run.line_seq or {}
  mark = run.cart_seq
  return [i for i in run.trolley.units if seqs.get(i, -1) < mark]

 def _price_or_inf(self, ident: str) -> float:
  """Returns a listing's latest archived price as a float, or infinity when unknown.
        Lets unknown prices sort last and stay out of budget sums."""
  price = self.run.archive.price(ident)
  return float(price) if is_number(price) else float("inf")

 def _verified_swap(self, line, gone, version, label, several, cap, qty, rest):
  """Lists replacements for a sold-out line that already passed the item check at this version.
        Excludes dead, refused and in-cart listings and the line itself, field-evidence fails, other
        categories, and any pick that with the rest of the cart would exceed the cap."""
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
  """Ranks in-stock rows from the line's category as replacements when no verified swap exists.
        Applies the budget, failed rulings and reference exclusions; read rows also need no field fails
        or clashes and must show every plain value. Returns (unread flag, price, id), read first."""
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
  """Proposes a replacement for a sold-out cart line.
        Returns (id, empty) when a listing passed the item check, else (empty, text naming up to three
        ranked rows); for cheapest, a cheaper unread row moves to the front. Sets swap_unread, and
        swap_page when no read row qualifies."""
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
   shown.append(f"{ident} ({figure}{', read: its record has each stated value' if r == 0 else ''})")
  return "", ", ".join(shown)

 def dead_page(self, k: int | None) -> dict[str, Any] | None:
  """Returns next-page search arguments for the open walk carrying most required values of product k.
        Ties go to the walk with fewer pages read; None without a paging argument or an open walk."""
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
  """Returns the first suggested swap row that is still unread, not sold out and not in the cart.
        Empty once intake.swap_done is set or when no such row is left."""
  run = self.run
  if run.intake.swap_done:
   return ""
  gone = self.dead_lines()
  for ident in self.swap_unread or []:
   if ident not in run.archive.read and ident not in gone and ident not in run.trolley.units:
    return ident
  return ""

 def dead_note(self, lines: list[str]) -> str:
  """Writes the remark for cart lines that turned sold out, a passage per line.
        Points to a verified swap, else to the cheapest ranked rows, else to finding another listing,
        plus the next unread page when known. Sets run.event_seen."""
  run = self.run
  parts = []
  run.event_seen = True
  for line in lines:
   self.swap_page, self.swap_unread = None, []
   pick, rows = self.dead_swap(line)
   head = f"{line} is SOLD OUT (per its latest record), so an order containing it fails. Remove it from the cart"
   if pick:
    least = ", the cheapest such listing" if run.needs.priority() == "cheapest" else ""
    parts.append(f"{head}, then add {pick} (per the item check it meets every requirement now in force{least}), "
        "check its stock and price again, then order.")
   elif rows:
    parts.append(f"{head}; that product has no other verified match, so turn to the cheapest in-stock rows of its "
        f"category within the budget ({rows}; read rows whose record lacks or violates a stated "
        "requirement were left out, and an unread one needs its details opened first), add the first that fits, check "
        "its stock and price again, then order.")
   else:
    parts.append(f"{head} and find another listing that fits.")
   if self.swap_page and not pick:
    parts.append("More rows of this kind wait on the following unread page of the search led by the stated value, "
        + json.dumps(self.swap_page, default=str) + ".")
  return " ".join(parts)

 def refusal_reread(self, obs: Any = None) -> str:
  """Reacts to a refused order: infers the lone cart line is gone, or queues a stock re-read.
        Returns what the shopper's text changed plus the sold-out note or the re-read remark."""
  heard = ""
  try:
   heard, gone = self.refusal_gone(obs)
  except Exception as err:
   gone = ""
   self.run.log.append({"kind": "refusal_gone_error", "error": f"{type(err).__name__}: {err}"[:300]})
  return heard + (gone if gone else self.refusal_queue())

 def refusal_gone(self, obs: Any) -> tuple[str, str]:
  """Infers from a refusal's shopper text that the lone cart line is no longer available.
        Only for a real order refusal with one held line: the text goes to needs.hear and, if only the
        version moved, the line is marked dead by lookup. Returns (heard note, sold-out note)."""
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
   """Captures the needs version, dropped and revealed counts, cap and remaining requirements.
            Compared before and after the refusal text is heard."""
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
  return heard, (" The refusal says an item is not available any more, and the cart held a single line, so the code counts " + line
        + " as sold out (no re-read): " + self.dead_note([line]))

 def refusal_queue(self) -> str:
  """Queues a stock re-read of up to four cart lines after a refused order.
        Needs a stock tool, no pending queue, enough steps and time; sets run.reads for a cart recheck
        and returns a short remark naming the tool."""
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
  reads.open_say = "The order was refused, so I check the cart lines' stock again (one may be gone)."
  return " Stock of the cart lines is read for you now (" + tool + ")."

 def code_judges_all(self) -> bool:
  """Tells whether code alone can judge every remaining requirement.
        Counts plain values from closer.code_values for every product plus requirements matched by a
        resolved differ reference, and compares that with the requirements left."""
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
  """Lists resolved, live differ references with a value that match a remaining requirement.
        With several products and k given, references tagged for another product are skipped."""
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
  """Tells whether every repair pick passed the item check at this version.
        When code may judge alone, a read pick that closer.code_fit marks fit counts too."""
  run = self.run
  fit = run.closer.code_fit or {}
  if not picks:
   return False
  return all(f"{i}|{version}" in run.passed or (own and i in run.archive.read and fit.get(i)) for i, _ in picks)

 def _repair_picks(self, several: bool, gone: set, own: bool, version: str):
  """Gets repair picks from closer.code_picks, avoiding sold-out lines.
        With several products it also avoids clashing rows, drops picks whose revealed needs are not
        shown or that miss values, and item-checks up to three others. Returns (id, units) pairs."""
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
  """Decides whether a decline repair may run now; returns the several-products flag or None.
        Opens after a refusal, once the first-order point is reached, or early with several products;
        needs cart and order tools, 3+ steps and time, and runs once per refusal count."""
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
  """Tells whether some product has every read candidate ruled out, so a decline may stand.
        Uses ed_proof and logs the products found or the error."""
  run = self.run
  try:
   proof = self.ed_proof()
  except Exception as err:
   proof = []
   run.log.append({"kind": "decline_proof_error", "error": f"{type(err).__name__}: {err}"[:300]})
  if proof:
   run.log.append({"kind": "decline_proof", "products": proof})
  return bool(proof)

 def decline_repair(self) -> str:
  """Decline gate that loads a cart of code picks instead of letting the actor decline.
        Live cart lines are briefly removed from refused_lines so they can be picked again (restored on
        failure). When the picks pass the checks and steps allow, puts them in muted and returns an
        order-now remark; otherwise an empty string."""
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
  return ("Not run yet (no step used). A decline is right only if no listing can work, but the records show a fitting candidate "
    + ("for every product asked for" if several else "") + " (cleared by the item check, or opened and found to hold every stated "
    "value in its record), so the code put " + ("them" if several else "it") + " in the cart and checked stock: "
    + run.trolley.summary() + " Order this cart now. Only decline if a record you read shows a requirement "
    "now in force that one of these lines fails.")

 def _repair_put(self, picks) -> bool:
  """Puts repair picks in the cart with the wire muted, then re-reads stock of stale picked lines.
        True only if every pick is in the cart, none turned sold out and no dead line remains."""
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
  """Tells whether a requirement value is plain, with no comparison or range in it.
        Rejects limit phrases, numeric ranges, plus or angle signs and approximate wording."""
  return not re.search(r"\b(?:" + VALUE_LINKS + r")\b|\d\s*(?:-|to)\s*\d|[+<>]|\b(?:about|around|approx\w*|roughly)\b",
        str(value).lower())

 @staticmethod
 def ed_word(value: str) -> bool:
  """Tells whether a value is plain words only: letters, no digits, no exclusion, no reference.
        Only such values can go to the near-wording check."""
  text = str(value).lower()
  if not re.search(r"[a-z]{3,}", text) or re.search(r"\d", text):
   return False
  return Catalogue.ed_plain(text) and not re.search(EXCLUDE_WORDS, text) and not ref_text(text)

 def ed_reqs(self, k: int | None) -> list[tuple[str, str, bool]]:
  """Lists (property, value, revealed) for each remaining requirement of product k.
        Tags are stripped; revealed is True for requirements not in the written request."""
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
  """Builds the status record of one listing in a product's candidate pool.
        Notes price, read, passed, gone, over budget, reference pointer, ruled out, value clashes and
        misses, then derives fit, out and whether every revealed need shows."""
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
   """Returns the row's field values whose field name matches the property.
            Matching uses same_prop."""
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
  """Returns the home labels for product k's candidate pool.
        Single product: the observer's labels or the first home set; several: home set k or empty."""
  homes = self.home_shelves()
  if not several:
   return set(self.run.watch.shelf_labels()) or (set(homes[0]) if homes else set())
  if k is not None and k < len(homes):
   return set(homes[k])
  return set()

 def ed_pool(self, k: int | None) -> list[dict[str, Any]]:
  """Builds and memoises candidate-pool entries for product k from code rows on its home labels.
        The key covers archive size, reads, version, requirements, passes, rulings, dead, refused, cap
        and references; plain values, joined values and revealed needs feed _pool_entry."""
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

 def ed_proof(self) -> list[int]:
  """Lists products whose read candidates are all out (gone, over budget, pointer, ruled, clash).
        A product with no read candidate is not listed."""
  proven = []
  for k in range(len(self.run.needs.items())):
   seen = [e for e in self.ed_pool(k) if e["read"]]
   if seen and all(e["out"] for e in seen):
    proven.append(k)
  return proven

 def _lead_scores(self, k: int, hits: set, book: list) -> dict[str, int]:
  """Scores listings by how many of product k's values led the searches that showed them.
        Search hits start at 1; each walk counts values whose words are among its leading query words."""
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
  """Returns the cheapest fitting price times units for each product, or None if one has no fit.
        Shows whether the cheapest fits together break the budget."""
  floors = []
  for k in range(count):
   prices = [e["price"] for e in self.ed_pool(k) if e["fit"] and e["price"] is not None]
   if not prices:
    return None
   floors.append(min(prices) * per_unit(k))
  return floors

 def slack_rows(self) -> list[str]:
  """Finds unread rows that could make the whole multi-product basket affordable.
        Several products and a cap only; when cheapest fits exceed the cap, keeps rows priced within
        the room left after other products' floors, ranks by lead score then price, and takes up to
        three round-robin. Sets slack_total."""
  run = self.run
  wanted = run.needs.items()
  budget = run.needs.spend_cap
  if len(wanted) < 2 or not is_number(budget):
   return []
  try:
   counts = run.needs.counts()

   def per_unit(k: int):
    """Returns product k's quantity, or 1 when the counts list is shorter.
                Turns unit prices into line totals."""
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
  """Returns the problem list of a listing's latest item check when that check failed it.
        None when the latest check passed or there was none."""
  for entry in reversed(self.run.log):
   if not isinstance(entry, dict) or entry.get("kind") != "verify" or entry.get("item") != ident:
    continue
   res = entry.get("result")
   if isinstance(res, dict) and res.get("fits") is False:
    return [str(p) for p in res.get("problems") or [] if str(p).strip()]
   return None
  return None

 def ed_news(self, k: int | None) -> bool:
  """Tells whether anything new came up since the request was read.
        True after any order refusal, or once a revealed or dropped need applies to product k."""
  run = self.run
  if run.refusal_n > 0:
   return True
  rows = run.needs.items()
  for req in list(run.needs.uncovered) + list(run.needs.let_go):
   if not rows or run.talk.product_of(str(req).partition(":")[0]) in (k, None):
    return True
  return False

 def near_hold(self) -> str:
  """Decline gate that orders the near-wording basket if it is still intact.
        Needs the same needs version and refusal count, with every line still held and not dead or
        refused; calls closer.finish_now and returns a remark."""
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
   return ("Not run (no step used). Earlier a wording review found the cart's lines (" + ", ".join(lines) + ") to express the "
     "requested values in other words; nothing changed since, so the code placed the order rather than declining.")
  except Exception as err:
   run.log.append({"kind": "near_hold_error", "error": f"{type(err).__name__}: {err}"[:300]})
   return ""

 def near_check(self) -> str:
  """Decline gate that runs the near-wording basket check once, when conditions allow.
        Needs cart and order tools, 4+ steps, time, a model and spend left; errors are logged."""
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
  """Lists the requirements an entry misses only in wording, with the field text it shows instead.
        None when a miss is a revealed need, has no field, is not a plain word or the field has a figure."""
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
  """Selects read, available, in-budget listings whose misses are only wording differences.
        For ruled-out ones every failed field and check problem must concern those same properties.
        Returns (miss count, price, id, lines) tuples."""
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
  """Writes the short prompt asking whether each field gives the requested value in other words.
        Lists each line as listing, property, value and field text, with the request for context."""
  listing = "\n".join(f"{i + 1}. {ident}: {prop} = '{value[:60]}' | the listing's field says: {says}"
       for i, (_, ident, prop, value, says) in enumerate(flat))
  return ("Goal: decide whether listings missing a requested value literally (a miss that would make the shopping "
    "assistant decline) still state it differently. THE SHOPPER'S REQUEST (context):\n"
    + self.run.request[:1500] + "\n\nLines: a listing, a requested property plus the value asked for, and what the "
    "listing's own field gives for it.\n" + listing + "\n\n1. Take the lines one at a time.\n2. Mark a line true only if the "
    "field states the requested value in other words (a synonym, a variant spelling or phrasing, or a more "
    "specific form of the same thing); a value of the property that differs is false.\n3. If unsure, answer false.\n"
    "Return: ONE JSON object, nothing else: {\"lines\": [{\"n\": 1, \"same\": true}]}")

 def _near_pools(self, slots, units, basket):
  """Puts each product's cheapest fit in the basket and gathers near candidates for the rest.
        A product without a fit needs ed_news and candidates (6 for one product, else 3); None if not."""
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
  """Reads the zero-based line numbers the model marked as the same value.
        Malformed answers give an empty set."""
  agreed: set[int] = set()
  if not isinstance(answer, dict) or not isinstance(answer.get("lines"), list):
   return agreed
  for verdict in answer["lines"]:
   if isinstance(verdict, dict) and verdict.get("same") is True and is_number(verdict.get("n")):
    agreed.add(int(verdict["n"]) - 1)
  return agreed

 def near_basket(self) -> str:
  """Builds a basket where wording-only misses are accepted by a CHECKER verdict, and loads it.
        Every product needs a fit or an agreed candidate within budget; marks them in run.passed,
        near_ok and near_said so later checks accept them, and fills the cart muted.
        Returns an order-now remark or an empty string."""
  run = self.run
  needs = run.needs
  wanted = needs.items()
  slots: list[int | None] = list(range(len(wanted))) if wanted else [None]
  counts = needs.counts()

  def units(n: int):
   """Returns the quantity for basket slot n from needs.counts.
            Defaults to 1 past the end of the list."""
   return counts[n] if n < len(counts) else 1
  budget = float(needs.spend_cap) if is_number(needs.spend_cap) else None
  basket: dict[int, tuple[str, float, int]] = {}
  pools = self._near_pools(slots, units, basket)
  if not pools:
   return ""

  def total(extra: float = 0.0) -> float:
   """Sums price times units over the basket so far, plus an extra amount.
            Tests the cheapest possible basket and the final one against the budget."""
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
  return ("Not run yet (no step used). Per a wording review, " + said + " - the same value, put another way - and each "
    "listing involved meets every need the shopper revealed, is in stock and within budget, so the code added the basket "
    "to the cart and read its stock: " + run.trolley.summary() + " Order this cart now. Only decline if a record "
    "you read shows a requirement now in force that one of these lines fails.")

 def cart_moved(self, before: dict[str, float]) -> None:
  """Updates cart epochs after a cart call, given the units held before it.
        Bumps cart_ver and resets since_cart; a new line starts a new stock epoch, and lines whose
        units rose lose their fresh-stock mark."""
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
  """After a cart call, queues a stock read of lines put in for the first time.
        Notes when the result shows the cart; only successful puts of stale lines count, up to four,
        with a stock tool, steps and time, and not while muted or covering."""
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
  """Marks cart lines whose stock this result shows as freshly read.
        Sets run.line_seq to the current absorb count; does nothing until any stock was seen."""
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
  """Returns which of the given cart lines are sold out, re-reading stale ones first when possible.
        Used by the closer before it finishes an order."""
  run = self.run
  gone = set(lines) & self.dead_lines()
  stale = [i for i in self.stale_lines() if i in lines and i not in gone]
  if stale and self.stock_tool() and run.moves_left() >= len(stale) + 2:
   _, sold = run.closer.code_recheck(stale)
   gone |= set(sold) | (set(lines) & self.dead_lines())
  return gone

 def _changing_tools(self) -> set:
  """Lists tools never used for scripted reads: the main role tools plus ones that change state.
        A tool counts as changing when its name or description mentions removal, purchase, cart etc."""
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
  """Finds a tool's single list-of-ids argument and how many items it takes (at most 10).
        Needs one non-enum string-array property, no other required argument and maxItems of 2+."""
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
  """Finds the first tool that does not change state and accepts a list of ids.
        Returns (tool, argument, max items) or None."""
  busy = self._changing_tools()
  for name, schema in self.run.shapes.items():
   if name in busy:
    continue
   found = self._list_argument(schema)
   if found:
    return name, found[0], found[1]
  return None

 def batch_tool(self) -> tuple[str, str, int] | None:
  """Returns the batch-read spec (tool, list argument, max items), working it out only once.
        Stored on run.batch_spec; open_named may later clear it."""
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
  """Tells whether a batch read returned full records for at least half of the requested ids.
        A record counts as full with a property map or five short text fields."""
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
  """Returns a row's title text, from a title key or any key mentioning title or heading.
        Empty string when none is found."""
  if isinstance(row.get("title"), str):
   return row["title"]
  for key, v in row.items():
   if isinstance(v, str) and re.search(r"title|heading|headline", str(key).lower()):
    return v
  return ""

 def _led_first(self, home, most, words, known) -> list[str]:
  """Picks the first unread row of the best value-led search when no read listing shows every value.
        Prefers the walk carrying most required values of this home; adds the row's title words to known."""
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
  """Chooses up to most unread listings on the home labels for scripted reading.
        Skips sold-out, taken, refused and over-budget rows; may start with a value-led row, then favours
        search hits, scored rows and titles least like those already seen."""
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
   """Tells whether a row may be offered: not sold, taken or refused, and within budget at qty.
            Rows with unknown prices pass."""
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
   """Sort key for menu picks: hit tier, title overlap with titles seen, then pool order.
            Lower values are picked first."""
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
  """Sets the given picks as the pending read queue with a menu purpose.
        Also turns recheck on and stores the line to say."""
  reads = self.run.reads
  reads.open_queue = picks
  reads.open_recheck = True
  reads.open_say = say
  reads.open_purpose = "menu"

 def _rich_count(self, home) -> int:
  """Counts listings read with full records on the given home labels.
        Returns 4 when no home label is known."""
  if not home:
   return 4
  return sum(self.run.archive.rich_reads(s) for s in home)

 def _second_round(self, homes, batch, qty_for) -> bool:
  """Queues the second pre-read round for products with fewer than three full reads.
        Picks new rows to reach four per product, interleaves the products, caps at the batch size."""
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
  self._queue_menu(order, "First a few more listings: for some products the ones opened so far had only a few fields.")
  return True

 def _first_round(self, homes, batch, qty_for) -> bool:
  """Queues the first pre-read round so each product has listings with known fields.
        Products with no full read get 2 rows (4 for one product), thin ones are topped up to 4; two
        batches are allowed when many rows are needed and more than 25 steps remain."""
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
  self._queue_menu(order, "I open a few listings of each requested product to learn their fields before any question is sent.")
  return True

 def pre_read(self) -> bool:
  """Opens a few listings per product before questions go out, when a batch tool exists.
        Runs the first round, or a second when the first left products thin; needs one question at
        most so far, enough steps and time. True when reads were queued."""
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
    """Returns the quantity of product k, using the last one when k runs past the list.
                Passed to menu_rows for its budget test."""
    return quantities[min(k, len(quantities) - 1)]
   if again:
    return self._second_round(homes, batch, qty_for)
   return self._first_round(homes, batch, qty_for)
  except Exception as err:
   run.log.append({"kind": "pre_read_error", "error": f"{type(err).__name__}: {err}"[:300]})
   return False

 def _covered_labels(self) -> set[str]:
  """Returns the category labels of listings verified at this version or named by the last audit.
        Refused and sold-out verified listings are left out; audit fits are taken as they are."""
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
  """Lists unread rows for products still lacking a verified fit, before a decline.
        Several products only; takes rows on those products' home labels, in budget and not sold,
        hits first, round-robin up to the batch size (3 without a batch tool)."""
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
 """Shopper conversation: plans the questions to the shopper and sends them for the actor.
    Keeps the asking state across turns: pending re-asks and second halves, open pair halves,
    questions the actor wanted to send, per-product quota reads and properties given up on.
    Questions come from the ASK_SEAT ranking (our_plan) or the PLANNER seat; with several products
    it maps text to a product slot and shares the questions out between products.
    Built once per ShoppingRun as run.talk."""
 def __init__(self, run):
  """Bind the run and reset the asking state.
        Tracks when the last question went out (detail-set count and tick), the label in flight, empty
        plans, pair handling, pending seconds and their texts, quota reads and the wanted/spent asks."""
  self.run = run
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
  self.v73_logged = set()
  self.v73_listy = set()

 def v73_note(self, kind: str, prop: Any, why: Any, where: str) -> None:
  """Log a v7.3 question skip (kind v73_q1_skip or v73_q2_skip) once per kind, property, reason and place.
        where is "menu" for ask_menu drops and "plan" for planned questions held back before sending."""
  mark = (kind, str(prop).lower(), str(why).lower(), where)
  if mark in self.v73_logged:
   return
  self.v73_logged.add(mark)
  self.run.log.append({"kind": kind, "prop": str(prop)[:80], "why": str(why)[:80], "at": where})

 def v73_closed(self, slot: int | None, prop: Any) -> bool:
  """Tell whether a planned property must not be asked (v7.3); a first finding is logged by v73_note.
        Q1: a catch-all field by name, or one ask_menu dropped as a list for the same product (v7.4: v73_listy holds
        (slot, name), slot 0 for a single product). Q2: its ask_twin set is a twin (twin_match) of that of a question
        already answered for its product (not marked [not answered]), of a stated, revealed or resolved requirement
        that settles it (Intake.settles), or of a property the shopper does not mind."""
  try:
   return self._v73_closed(slot, prop)
  except Exception as err:
   self.run.log.append({"kind": "v73_q_error", "error": f"{type(err).__name__}: {err}"[:300]})
   return False

 def _v73_closed(self, slot: int | None, prop: Any) -> bool:
  """The checks of v73_closed, run inside its error guard; True holds the property back."""
  run = self.run
  needs = run.needs
  name = str(prop).split("(")[0].strip()
  if not name:
   return False
  rows = needs.items()
  listed = (slot if rows else 0, ask_name(name)) in self.v73_listy
  if catch_all_key(name) or listed:
   self.v73_note("v73_q1_skip", prop, catch_all_key(name) or "list", "plan")
   return True
  if rows:
   product = str(rows[slot]["product"]) if slot is not None and slot < len(rows) else ""
  else:
   product = str(needs.reading.get("product") or "item")
  mine = ask_twin(name, product)
  if not mine:
   return False

  def concerns(text: str, tagged_only: bool) -> bool:
   """Whether a label or requirement head is about this product: always without several products or a slot;
            an untagged head counts unless tagged_only is set."""
   if slot is None or not rows:
    return True
   owner = self.product_of(text)
   return owner == slot or (owner is None and not tagged_only)
  heads = []
  for logged in run.questions_log:
   head = logged.split(":", 1)[0].strip("- ")
   if "[not answered]" not in head:
    heads += [(part.strip(), False) for part in head.split(";")]
  for req in needs.written + needs.uncovered + list(needs.now_reqs or []) + list(needs.resolved or []):
   if run.intake.settles(str(req)):
    heads.append((str(req).partition(":")[0].strip(), False))
  heads += [(str(free).strip(), True) for free in needs.indifferent]
  for head, tagged_only in heads:
   if head and concerns(head, tagged_only) and twin_match(ask_twin(head, product), mine):
    self.v73_note("v73_q2_skip", prop, head, "plan")
    return True
  return False

 def text_arg(self, the_tool: str) -> str | None:
  """Find the one free-text string argument of a tool's schema, or None.
        Enum fields are ignored; the slot is a required string field, or the only string field.
        Returns None when the tool has any other required field, so a bare text call cannot be built."""
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
  """Build a tool call that sends text_body through the message-role tool, or None.
        The text goes into the slot found by text_arg and is checked with ArgCheck.execute; any fault
        gives None. cid becomes the call id. Used for questions, replies and the basket go-ahead."""
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
  """Map a piece of text to the index of the product it is about, in multi-product requests.
        Scores product kind words against the first parenthesised part, then the whole text; words
        shared with other products count half. None for a single product, a tie or no match."""
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
  """Product index of a planned question item: from its property label first, then its question.
        None when neither names a product (always None for a single-product request)."""
  slot = self.product_of(str(qitem.get("property") or ""))
  if slot is None:
   slot = self.product_of(str(qitem.get("question") or ""))
  return slot

 def _tally_asked(self, count: int):
  """Count questions per product from run.questions_log: sent, and answered (no [not answered]).
        A log line is tied to a product by its label, else by the quoted question text.
        Returns the (answered, sent) lists, each of length count."""
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
  """Products that still have open story hints worth asking about.
        A product qualifies with a hint whose property is a key in its listings' menu, or with more
        keyless hints than answered questions. Returns a set of indices; empty on any error."""
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
  """Turn the second property of a scripted pair into a standalone question for a due product.
        Takes the first scripted item whose second belongs to a product in due and words a plain
        question for it from the label's product tag. Returns the new item, or None."""
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
  """Build a per-product priority tuple that decides which product is asked next (lowest first).
        Orders by: held back when it has revealed needs while another got no answer, no open story
        hint, answers plus revealed needs, unanswered sends (zero with an untried hint alternative)."""
  never_answered = [j for j in range(count) if not answered[j]]
  table = {}
  for k in range(count):
   waits = 1 if revealed[k] and any(j != k for j in never_answered) else 0
   silent = 0 if k in alt_left else sent[k] - answered[k]
   table[k] = (waits, 0 if k in hinted else 1, answered[k] + revealed[k], silent, revealed[k])
  return table

 def _quota_for(self, due, scripted, hinted, answered, ok):
  """Find a quota question for one of the due products, hinted products first.
        Calls quota_pick per product and drops picks that ok rejects or that were already asked.
        Returns the pick put before scripted, scripted alone when a quota read was queued, or None."""
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
  """Reorder planned questions so the product most in need of a question is asked next.
        Multi-product only: tallies answers, revealed needs and story hints per product, then level by
        level moves a matching item first, splits off a pair's second, or builds a quota pick.
        ok is the freshness filter from pose_planned; returns the reordered list."""
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
  """Queue some unread listings of product k for opening before asking about it.
        Puts up to four rows (two without a batch tool) from its home categories into run.reads with
        purpose quota and sets quota_wait; once per product, and only with more than 12 moves left."""
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
  reads.open_say = f"I open a few {product} listings to learn their fields; the question on it follows later."
  reads.open_purpose = "quota"
  self.quota_wait = True

 def _read_home(self, k: int, home) -> list[str]:
  """Category labels already read that belong to product k's home categories.
        When none are read but quota reads were already queued for k, falls back to the first home
        label whose property menu is non-empty."""
  archive = self.run.archive
  labels = [label for label in archive.shelves_read() if label.strip().lower() in home]
  if labels or not home or not self.quota_reads.get(k):
   return labels
  return [label for label in sorted(home) if archive.prop_menu(label)[0]][:1]

 def _menu_question(self, k, product, menu, fallback, good):
  """Choose a property name to ask product k about, from its story hints or its property menu.
        An open story hint whose property has a menu key wins; with fallback, the first menu entry
        still open is taken, a second pass judging openness by sample values. good vets each one."""
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
  """Build one question for product k from the listings read in its home category.
        Returns None, possibly after queueing reads via _quota_open_rows, when none is read yet.
        Tries the product's PLANNER plan (once per run), else _menu_question; the item is labelled
        'property (product)' and worded as the plain question."""
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
   """Accept a property label when no ok filter is given, or when ok passes it
            and the property has not been asked for this product yet."""
   return ok is None or (bool(ok(label)) and not intake.asked_for(k, label))
  planned = self.product_plan(k, menu)
  if planned and good(planned.get("property")):
   return planned
  prop = self._menu_question(k, product, menu, fallback, good)
  if prop is None:
   return None
  return {"property": f"{prop} ({product})", "question": f"Which {prop} would you like for the {product}?"}

 def product_plan(self, k: int, menu: str) -> dict[str, Any] | None:
  """Run product_plan_call for product k, returning None on any error.
        menu is the trimmed property menu of the product's home category."""
  try:
   return self.product_plan_call(k, menu)
  except Exception:
   return None

 def product_plan_call(self, k: int, menu: str) -> dict[str, Any] | None:
  """Ask the PLANNER seat (first model only) for questions about one product of a basket.
        At most one such call per run, and only with time beyond CLOSING + 40 s. The prompt adds the
        request, requirements, asked log, story hints, plan_notes and the product's property menu.
        Logs the answer as product_plan and returns the item picked by _plan_choice."""
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
   prompt += "\n\nALREADY ASKED (and the shopper's answer):\n" + "\n".join(run.questions_log)
  prompt += (f"\n\nTHIS ROUND EVERY QUESTION CONCERNS THE {product.upper()}: every other product has had its questions "
       f"already, this one none yet. Pick {product} properties only, from its list below, "
       "starting with the ones the story's details point to.")
  try:
   hints = intake.open_hints(k)
  except Exception:
   hints = []
  if hints:
   prompt += ("\n\nSTORY HINTS (pieces of the request's story, each paired with the property it most likely refers to, likeliest "
        "first; none is a requirement):\n" + "\n".join(f"- {intake.hint_mark(h)}" for h in hints[:6]))
  prompt += self.plan_notes(k)
  prompt += (f"\n\nPROPERTIES IN THE DETAILS OF THE {product.upper()} LISTINGS SEEN SO FAR (name, how many listings, "
       "sample values):\n" + menu)
  answer = run.wire.consult(prompt, 1500, PLANNER[:1])
  run.log.append({"kind": "product_plan", "product": product, "result": answer})
  return self._plan_choice(k, product, answer)

 def _plan_choice(self, k, product, answer):
  """Pick the first usable question from a planner answer for product k.
        Keyless hint properties are dropped and the property must still be open; the planner wording
        is kept only if it names the product, else a plain question. A distinct open second is paired
        only then; a lone open second can stand in for the first."""
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
  """Decide whether a known requirement already settles the property name being considered.
        Same property: intake.settles decides; a near-same one: settled when its positive value shows
        in samples or seen field values; otherwise every word of a non-numeric value must be in samples."""
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
  """Tell whether prop is still open to ask about for product k (None: any product).
        Closed when it is a junk key, was answered in the question log, is settled by a remaining
        requirement (_settled_by), or the shopper said no preference. Errors count as open."""
  run = self.run
  intake = run.intake
  try:
   name = str(prop).split("(")[0].strip()
   if not name or junk_key(name):
    return False
   several = bool(run.needs.items())

   def concerns(text: str, tagged_only: bool = False) -> bool:
    """Whether a logged label or requirement text is about product k.
                Always true without several products; untagged text counts unless tagged_only is set."""
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
  """Extra planner prompt text about leftover asks, filtered to product k when given.
        Lists up to three unasked open halves of earlier pairs (open_pair) and up to three questions
        the actor tried to send itself (wanted_asks). Empty string on error or when nothing applies."""
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
    text += ("\n\nOPEN FROM LAST PAIR (the second halves of planned pairs left unanswered, because the shopper answered "
       "only the first or only the first was sent; they are still open, and each was what the planner "
       "would ask next):\n" + "\n".join(f"- {p}" for p in opens[-3:]))
   if wanted:
    text += ("\n\nQUESTIONS THE ASSISTANT WANTED TO ASK (noted after reading listing records; if one names an open property "
       "that is also a key below, put that key first and ask it under its own name):\n"
       + "\n".join(f"- {w}" for w in wanted[-3:]))
   return text
  except Exception:
   return ""

 def want_ask(self, raw_args: Any) -> None:
  """Remember the text of a message the actor tried to send while the code owns the asking.
        raw_args are the blocked call's arguments (dict or JSON text); string values are joined, cut
        to 300 chars and kept among the last three distinct wanted_asks. Called from ShoppingRun.screen."""
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
  """Forget wanted asks that the question just sent has covered.
        label holds the asked property names (split on ';' and 'and'); a wanted ask is dropped when its
        words, compared by stem, include every word of one of those properties."""
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
  """Cut a trailing 's' from a word longer than three letters; other words come back unchanged.
        A rough singular form used by _own_words for product name tokens."""
  return word[:-1] if len(word) > 3 and word.endswith("s") else word

 def _own_words(self, prop: str) -> set[str]:
  """Word set a question about prop may hold without straying to another property.
        Holds the property's words, the template words of a plain question and every product name
        token in plain and singular form."""
  needs = self.run.needs
  mine = set(prop_words(prop)) | {"which", "would", "you", "like"}
  for row in needs.items() or [{"product": str(needs.reading.get("product") or "")}]:
   for token in re.findall(r"[a-z0-9]+", str(row.get("product") or "").lower()):
    mine.add(token)
    mine.add(self._singular(token))
  return mine

 def names_other(self, question: str, prop: str, menu: str, second: str = "") -> bool:
  """Tell whether a question text strays beyond the asked property.
        True when it names another menu field (or the paired second) with words foreign to the asked
        property and product, or quotes a sample value of another field. False on error."""
  try:
   spoken = " ".join(re.findall(r"[a-z0-9]+", str(question).lower()))
   heard = set(spoken.split())
   for token in list(heard):
    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
     heard.add(token[:-1])
   mine = self._own_words(prop)

   def mentions(field: str) -> bool:
    """Whether every core word of field occurs in the question while at least one of them
                is foreign to the asked property and product."""
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

 def pair_fits(self, first: str, second: str) -> bool:
  """Tell whether two properties may share one question message.
        They must belong to the same product and differ in name, words and seen field values, and no
        other open menu key may be a twin of the second. False on any error."""
  run = self.run
  intake = run.intake
  try:
   several = bool(run.needs.items())
   slot = self.product_of(first) if several else None
   if several and (slot is None or self.product_of(second) != slot):
    return False
   if strict_prop(first, second) or same_prop(first, second) or intake.same_field(slot, first, second):
    return False
   left = {w for w in prop_words(first) if len(w) >= 3 and w not in LINK_WORDS}
   right = {w for w in prop_words(second) if len(w) >= 3 and w not in LINK_WORDS}
   if (left & right) or (intake.field_values(slot, first) & intake.field_values(slot, second)):
    return False
   menu = (intake.menu_memo.get(slot) or (0, ""))[1]
   for line in menu.splitlines():
    key = line.strip("- ").split(" (in ")[0].strip()
    if not key or strict_prop(key, first) or strict_prop(key, second):
     continue
    if intake.twin_of(slot, key, second) and intake.plan_open(slot, key):
     return False
   return True
  except Exception:
   return False

 def queue_second(self, prop: str) -> None:
  """Add a property to pending_seconds unless it is already there (case-insensitive).
        Used for the unaddressed second half of a paired question, so a later round asks it alone."""
  queue = self.pending_seconds
  if prop and prop.lower() not in [p.lower() for p in queue]:
   queue.append(prop)

 def _product_tag(self, label: str, fallback: Any = None) -> str:
  """Product name a property label carries in parentheses.
        Without a tag, returns fallback when given, else the request's product, else 'item'."""
  tied = re.search(r"\(([^)]+)\)", label)
  if tied:
   return tied.group(1).strip()
  if fallback is not None:
   return fallback
  return str(self.run.needs.reading.get("product") or "item")

 @staticmethod
 def _plain_ask(prop: str, product: str) -> str:
  """Word the standard one-property question 'Which <prop> would you like for the <product>?'.
        Any parenthesised product tag is cut from prop first."""
  return f"Which {prop.split('(')[0].strip()} would you like for the {product}?"

 def _too_early(self, sets: int, pressing: bool, urge: bool, ceiling: int) -> bool | None:
  """Decide whether the next planned question should wait (True) or may go now (False).
        Waits while few detail sets are read in the first steps, after an empty or failed plan until
        more sets arrive, and for a few ticks after an ask with no new reads in single-product runs.
        Stops asking for good (run.asking False) when moves or time run low."""
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
  """Property menus per category for a multi-product plan prompt.
        Takes up to four read category labels, the products' home categories first, and returns the
        full and the trimmed menu text, each headed CATEGORY <label>; falls back to the given menus."""
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
  """Assemble the PLANNER prompt for one planned question.
        Adds the question counts, the one-at-a-time rule, request, requirements, asked log, ask_first
        items, plan_notes, up to eight story hints and the property menus (per category for baskets).
        Returns the prompt and the property list it used."""
  run = self.run
  intake = run.intake
  needs = run.needs
  parts = [ASK_PLAN.replace("QUESTION_COUNT", str(remaining)), NEXT_ONE.replace("REMAINING", str(remaining)),
     "\n\nTHE SHOPPER'S REQUEST:\n" + run.request, "\n\nREQUIREMENTS SO FAR:\n" + needs.text_body()]
  if run.questions_log:
   parts.append("\n\nALREADY ASKED (and the shopper's answer):\n" + "\n".join(run.questions_log))
  must = needs.reading.get("ask_first")
  if must:
   parts.append("\n\nMUST BE LEARNED FROM THE SHOPPER: " + "; ".join(str(h) for h in must or []))
  parts.append(self.plan_notes(None))
  try:
   hints = intake.open_hints()
  except Exception:
   hints = []
  if hints:
   parts.append("\n\nSTORY HINTS (pieces of the request's story, each paired with the property it likely refers to; none is a "
       "requirement, but the hidden need is usually among them):\n"
       + "\n".join(f"- {intake.hint_mark(h)}" for h in hints[:8]))
  shown = intake.menu_trim(prop_list, sets)
  if needs.items():
   parts.append("\n\nTHE PRODUCTS ASKED FOR: " + " | ".join(f"{n + 1}) {row['product']} x{row['quantity']}"
                 for n, row in enumerate(needs.items())))
   prop_list, shown = self._category_menus(prop_list, shown)
  parts.append("\n\nPROPERTIES IN THE LISTINGS' DETAILS SEEN SO FAR (name, how many listings, sample values):\n"
      + (shown or "(none seen yet)"))
  return "".join(parts), prop_list

 def _vet_plan(self, answer, fresh) -> list[dict[str, Any]]:
  """Filter planned questions through fresh and clean up their second halves.
        Empty or stale items are dropped and placeholder seconds cleared; a keyless-hint second makes
        the question plain; a stale first with a fresh second is replaced by the second alone."""
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
  """Make sure a basket question names its product.
        When the question text holds no word of the label's product tag it is replaced with the plain
        question; otherwise the item comes back unchanged."""
  prop = str(thing.get("property") or "")
  product = self._product_tag(prop, "")
  words = re.findall(r"[a-z]{3,}", product.lower())
  if not product or not words:
   return thing
  asked = str(thing.get("question") or "").lower()
  if any(w.rstrip("s") in asked for w in words):
   return thing
  return dict(thing, question=self._plain_ask(prop, product))

 def _with_second(self, label, second, remaining):
  """Build the message call for a planned item that carries a second property.
        With one message left, pairs allowed, pair_fits and a story hint behind the first, both go in
        one question; otherwise only the first is asked and the second is handed back to keep open.
        Returns (call, label, second left over)."""
  run = self.run
  slot = self.product_of(label[0]) if run.needs.items() else None
  stamp = f"ask_{run.msgs_sent + 1}_{run.wire.tick}"
  product = self._product_tag(label[0])
  if remaining == 1 and not self.no_pairs and self.pair_fits(label[0], second) and run.intake.hint_mapped(slot, label[0]):
   pair = (f"Which {label[0].split('(')[0].strip()} would you prefer for the {product}, "
     f"and which {second.split('(')[0].strip()}?")
   return self.message_call(pair, stamp), (label[0] + "; " + second, pair), ""
  single = self._plain_ask(label[0], product)
  return self.message_call(single, stamp), (label[0], single), second

 def _propose(self, scripted, pending, prop_list, remaining):
  """Turn the first usable of the top two planned items into a message call.
        Basket questions get their product named; questions naming other properties become plain (not
        for a pending re-ask or a must-ask opener); seconds go through _with_second.
        Returns (call or None, (property, question) label, second left to keep open)."""
  run = self.run
  needs = run.needs
  proposal, label, split_second = None, None, ""
  strict_start = needs.reading.get("ask_first") and not run.questions_log
  for thing in scripted[:2]:
   if needs.items():
    thing = self._named_product_question(thing)
   if not pending and not strict_start and self.names_other(
     str(thing.get("question") or ""), str(thing.get("property") or ""), prop_list, str(thing.get("second") or "")):
    prop = str(thing.get("property") or "")
    thing = dict(thing, question=self._plain_ask(prop, self._product_tag(prop)))
   proposal = self.message_call(str(thing.get("question")).strip(), f"ask_{run.msgs_sent + 1}_{run.wire.tick}")
   if not proposal:
    continue
   label = (str(thing.get("property")), str(thing.get("question")).strip())
   second = str(thing.get("second") or "").strip()
   if not second or second.lower() == label[0].lower() or re.search(r"none|^-$|n/a", second.lower()):
    break
   proposal, candidate, split = self._with_second(label, second, remaining)
   if not proposal:
    continue
   label, split_second = candidate, split
   break
  return proposal, label, split_second

 def _menu_basis(self):
  """Property menu text and count of read detail sets that question planning works from.
        Uses the request's own category when one is known, else the whole archive."""
  archive = self.run.archive
  whole_menu, whole_sets = archive.prop_menu()[0], archive.rich_reads()
  own = self.run.shelf.own_shelf()
  if not own:
   return whole_menu, whole_sets
  return archive.prop_menu(own)[0], archive.rich_reads(own)

 def _planned_answer(self, prompt, pending, sets):
  """Get this round's question plan in the planner's answer shape.
        A pending property is asked alone (with its stored text if any); else our_plan when OUR_ASK is
        on; else the PLANNER seat. False means wait (our_plan waiting, or a failed call sets plan_wait)."""
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
  """Build the question plan in code, with one ranking call, instead of the PLANNER seat.
        Per product, ask_menu turns fact sets of listings in its home categories into keys, minus those
        the written requirements name; ask_ranks orders them and products take turns, least asked first.
        Returns False to wait for reads, {} to hand over to the planner, else {'questions': [...]}."""
  run = self.run
  needs = run.needs
  rows = needs.items()
  names = [str(row["product"]) for row in rows] or [str(needs.reading.get("product") or "item")]
  basket = len(names) > 1
  homes = run.shelf.home_shelves() if basket else []
  menus = []
  for slot, name in enumerate(names):
   if basket:
    facts = [found for label in sorted(homes[slot] if slot < len(homes) else []) for found in run.archive.fact_sets(label)]
   else:
    home = (run.shelf.home_shelves() or [set()])[0]
    if not home and run.moves_left() > run.intake.order_due() + 12:
     run.log.append({"kind": "our_plan", "wait": "no home shelf yet"})
     return False
    facts = ([found for label in sorted(home) for found in run.archive.fact_sets(label)] if home
       else run.archive.fact_sets(run.shelf.own_shelf()))
   settled = set()
   for line in needs.written:
    head = line.partition(":")[0]
    if not basket or self.product_of(head) == slot:
     settled.add(ask_name(head.split("(")[0]))
   skipped: list[tuple[str, str]] = []
   menus.append(ask_menu(facts, settled, basket, skipped, names[slot])[:ASK_MENU])
   for key, why in skipped:
    if why == "list":
     self.v73_listy.add((slot, ask_name(key)))
    self.v73_note("v73_q1_skip" if why in ("name", "list") else "v73_q2_skip",
         f"{ask_name(key)} ({names[slot]})" if basket else ask_name(key), why, "menu")
  if basket and not all(menus):
   empty = [names[slot] for slot, menu in enumerate(menus) if not menu]
   waiting = run.moves_left() > run.intake.order_due() + 12
   run.log.append({"kind": "our_plan", "empty": empty, "wait": waiting})
   return False if waiting else {}
  ranks = self.ask_ranks(names, menus)
  queues = []
  for slot, menu in enumerate(menus):
   pick = ranks.get(f"P{slot + 1}") or {}
   ranked = [str(key) for key in pick.get("ranked") or []] if isinstance(pick, dict) else []
   story = [str(key) for key in pick.get("story") or []] if isinstance(pick, dict) else []
   queues.append(ask_merge(ranked[:ASK_FIRST_PICKS], story, [key for key, _, _ in menu]))
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
    questions.append({"property": label, "question": self.our_question(ask_name(key), names[slot], samples, basket)})
  run.log.append({"kind": "our_plan", "menus": [[key for key, _, _ in menu][:8] for menu in menus], "first": [q["property"] for q in questions[:4]]})
  return {"questions": questions}

 def ask_ranks(self, names, menus):
  """Ask the ASK_SEAT model which properties of each product to ask about, best first.
        Sends ASK_RANK with the request and one P1, P2 ... block of keys, carrier counts and samples.
        The answer is memoised and reused while menus are unchanged or once ASK_RANKINGS calls are made."""
  mark = tuple(tuple(key for key, _, _ in menu[:12]) for menu in menus)
  memo = getattr(self, "rank_memo", None)
  if memo and (memo[0] == mark or memo[2] >= ASK_RANKINGS) or not any(menus):
   return memo[1] if memo else {}
  blocks = []
  for slot, (name, menu) in enumerate(zip(names, menus)):
   lines = [f"- {key} ({carriers} listings): {', '.join(samples)}" for key, carriers, samples in menu]
   blocks.append(f"## P{slot + 1}: {name}\n" + "\n".join(lines))
  prompt = ASK_RANK + "\n\n# Request\n" + self.run.request + "\n\n# Products\n" + "\n\n".join(blocks)
  answer = self.run.wire.consult(prompt, 1500, ASK_SEAT) or {}
  self.rank_memo = (mark, answer, (memo[2] if memo else 0) + 1)
  return answer

 @staticmethod
 def our_question(prop: str, product: str, samples: list[str], basket: bool) -> str:
  """Word a short question about one property for our_plan.
        Basket products get the bare form; a single product gets up to two short sample values as
        examples while the question stays within ASK_CHARS characters."""
  bare = f"What {prop} do you need for the {product}?"
  if basket:
   return bare
  for count in (2, 1):
   shown = [value for value in samples if len(value) <= 24][:count]
   worded = f"What {prop} do you need for the {product}" + (f", such as {' or '.join(shown)}" if shown else "") + "?"
   if len(worded) <= ASK_CHARS:
    return worded
  return bare

 def _no_proposal(self, sets: int, urge: bool) -> bool:
  """Handle a round where no question could be built; always returns False.
        With few detail sets and moves left it waits for more reads (plan_wait); otherwise it counts
        an empty plan, waits once more if moves allow, and then stops asking (run.asking False)."""
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
  """Send one planned question through run.dispatch as a scripted call and post the reply.
        Notes the question in the chat and sets asking_label while it is in flight. Returns the replies
        with the revealed-need and no-preference counts taken before sending."""
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
  """Plan and send one question to the shopper, or start listing reads first; True if it acted.
        Gates on the message limit and timing, may queue first reads, builds the plan (pending re-ask,
        our_plan or PLANNER), sends it and logs the reply in run.questions_log. Run by
        PlannedQuestionStep, and by ActorTurnStep with urge=True before a cart/order/decline move."""
  run = self.run
  needs = run.needs
  intake = run.intake
  ceiling = run.ask_limit()
  if not run.role("message") or run.msgs_sent >= ceiling:
   run.asking = False
   return False
  prop_list, sets = self._menu_basis()
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
   """True when a planned property may still be asked: not empty, not in the question log, not a
            junk key, not given up on, not a keyless story hint, still open for its product, and not held
            back by v73_closed (a catch-all field, or a twin of a settled property)."""
   prop = str(prop or "").strip()
   lowered = prop.lower()
   if not prop or lowered in asked_props or junk_key(prop) or lowered in self.spent_asks:
    return False
   slot = self.product_of(prop) if needs.items() else None
   return not intake.plain_hint(slot, prop) and intake.plan_open(slot, prop) and not self.v73_closed(slot, prop)
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
  proposal, label, split_second = self._propose(scripted, pending, prop_list, remaining)
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
   twin = "" if ";" in asked_label else intake.quiet_reask(asked_label)
   if twin:
    self._put_first(twin)
    note += f" [{twin}: its story twin, asked next]"
  if replies:
   body = replies[0][1]
   run.questions_log.append(f"- {label[0]}: asked \"{label[1]}\" -> {body.split(chr(10) + '[status]')[0][:400]}{note}")
  self.drop_wanted(asked_label)
  run.asking = run.msgs_sent < ceiling
  return True

 def _put_first(self, prop: str) -> None:
  """Put prop at the head of pending_seconds, dropping any other copy of it.
        The next pose_planned round then asks it alone before any planned question."""
  self.pending_seconds = [prop] + [p for p in self.pending_seconds or [] if p.lower() != prop.lower()]

 def _keep_open(self, prop: str) -> None:
  """Add a property to open_pair unless it is already listed (case-insensitive).
        plan_notes later shows these unanswered pair halves to the planner."""
  if prop.lower() not in [p.lower() for p in self.open_pair]:
   self.open_pair.append(prop)

 def _one_at_a_time(self, label, split_second):
  """React to a shopper reply that only asked for one question at a time.
        Sets no_pairs and re-plans the first property: its story-hint alternative, a plain re-ask, a
        wording from intake.distinct_ask, or giving up (spent_asks); other halves stay open.
        Returns the label marked not answered and a note for the question log."""
  intake = self.run.intake

  def plain(text: Any) -> str:
   """Lowercase a text to its alphanumeric words joined by single spaces.
            Used to compare the plain re-ask with the question that was sent."""
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
   note = f" [{first}: not answered, story-hint alternative {alt} is asked next]"
  elif first.lower() not in self.reasked and plain(plain_q) != plain(label[1]) and not (alone and intake.holds_plain(label[1], plain_q)):
   self.reasked.add(first.lower())
   self._put_first(first)
   note = f" [{first}: not answered, re-asked alone next]"
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
  """Work out what the shopper's reply to a sent question settled and fix up its label.
        A one-at-a-time reply goes to _one_at_a_time; for a pair, an unaddressed second is queued or
        kept open and a no-preference is credited to the first only; a pair with no effect ends pairing.
        Returns (label, note) for run.questions_log."""
  needs = self.run.needs

  def plain(text: Any) -> str:
   """Lowercase a text to its alphanumeric words joined by single spaces.
            Used to test whether the sent question is the one the meta-only reply answered."""
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
  """Answer a question the shopper put to the assistant with a message, when allowed.
        Skips without a message role, at the message limit, with 4 moves or less, or under CLOSING
        time; keeps the last message for planned questions while asking. Run by ShopperReplyStep."""
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
  """Append tool results to run.chat as tool messages keyed by call id.
        The last one gets run.status_line() appended. Used after scripted and actor calls alike."""
  run = self.run
  last = len(responses) - 1
  for idx, (cid, body) in enumerate(responses):
   if idx == last:
    body += "\n" + run.status_line()
   run.chat.append({"role": "tool", "tool_call_id": cid, "content": body})


class CodeCloser:
 """Code-only fallbacks that work without LLM help; built once per ShoppingRun as run.closer.
    Fills order and decline arguments, reads the request by regex when no reader model answered,
    builds search calls, picks listings from archived records, fills the cart and forces a finish
    when the turn loop ends. catch_up feeds late model answers back in each turn."""
 def __init__(self, run):
  """Bind the run and reset the fallback state.
        code_fit maps picked ids to whether every judged value showed; code_hits maps ids to words of
        searches that returned them; code_more holds next-page calls; the rest guard repeats."""
  self.run = run
  self.code_fit = {}
  self.code_hits = {}
  self.code_more = None
  self.code_searched = None
  self.news_read = set()
  self.reread = None

 def schema_args(self, tool: str, text: str, kind: str = "") -> dict[str, Any]:
  """Fill the required arguments of a finishing tool from its schema.
        Enums take the first choice matching ORDER_WORDS or DECLINE_WORDS (by kind), else the first;
        strings take text cut to maxLength, booleans True, numbers max(1, minimum). A lone optional
        free-text field also gets the text."""
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
    kinds = finish_choice_kinds(spec["enum"])
    matching = [choice for choice in spec["enum"] if kind and kinds.get(choice) == kind]
    matching = matching or [choice for choice in spec["enum"]
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
  """Replace free-text decline arguments with the fixed SORRY text when names must not appear.
        Applies when the request bans naming substitutes, or when that rule is unknown and some text
        mentions a known listing id. Enum values are kept. Applied by ShoppingRun.screen_finish."""
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

 def code_reading(self) -> dict[str, Any]:
  """Read the request with regexes alone, for when the reader models gave no usable reading.
        Fills budget and currency, stock and cheapest cues, message and step limits, tool roles and the
        product(s) with quantities and stated requirements; flagged code_reading. Used by
        Intake.reading_by_code."""
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
  """Find the total price limit stated in a text; returns (amount, unit) or (None, None).
        Each number is scored on a currency sign, spending or limit words before it, limit words after
        it and order-total words in its sentence; counts of messages, steps or units are skipped.
        The unit is the words after the number, else the currency sign."""
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
  """Find a stated count limit for the given nouns (messages or steps, say) in lowered text.
        The number (digits or a word up to twenty) needs a limiting cue just before or after it.
        Returns the first such count as an int, or None."""
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
  """Read how many units the request wants from phrases like 'three units' or '2 copies'.
        Returns 1 when none is found or the count falls outside 1 to 99."""
  numbers = number_words()
  hit = re.search(r"\b(\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s+(?:units?|of (?:them|those|these)|copies)\b", lowered)
  if hit is None:
   return 1
  word = hit.group(1)
  count = int(word) if word.isdigit() else numbers[word]
  return count if 1 <= count <= 99 else 1

 @staticmethod
 def stated_pairs(stretch: str) -> list[tuple[str, str]]:
  """Pull (property, value) pairs from phrases like 'the colour is red' or 'whose size must be M'.
        Skips values pointing at another listing, long names or values, names holding pronouns or
        price words, and values that open with a function word."""
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
  """Category labels offered by the search tool: its longest enum list of four or more options.
        Returns [] when the tool has none."""
  fields = (self.run.shapes.get(tool or "") or {}).get("properties")
  choice_lists = []
  for spec in (fields.values() if isinstance(fields, dict) else []):
   if isinstance(spec, dict) and isinstance(spec.get("enum"), list) and len(spec["enum"]) >= 4:
    choice_lists.append(spec["enum"])
  return max(choice_lists, key=len) if choice_lists else []

 def counted_products(self, tool: str | None) -> list[dict[str, Any]]:
  """Find counted products in the request by matching search category labels in its text.
        A label counts only when a number from 1 to 99 shortly precedes it; each product gets its
        label, quantity and stated_pairs requirements from its text up to the next one, in order."""
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
  """Guess which tool fills each role from tool names, descriptions and schemas.
        Order and decline come from an enum offering both, from name words or 'final' descriptions;
        cart tools split into cart and cart_view by whether they take arguments; search, message,
        details and view follow word cues and argument shape. Returns role -> tool name."""
  described: dict[str, str] = {}
  for tdef in self.run.toolset:
   fn = tdef["function"]
   spaced = re.sub(r"([a-z])([A-Z])", r"\1 \2", fn["name"]).replace("_", " ").lower()
   described[fn["name"]] = spaced + " | " + str(fn.get("description") or "").lower()
  shapes = self.run.shapes

  def fields_of(tool: str) -> dict[str, Any]:
   """Properties map of a tool's argument schema.
            Returns {} when the schema has no dict of properties."""
   found = (shapes.get(tool) or {}).get("properties")
   return found if isinstance(found, dict) else {}

  def required_of(tool: str) -> list[str]:
   """Required argument names of a tool that also appear among its properties.
            Returns [] when the schema lists none."""
   found = (shapes.get(tool) or {}).get("required")
   if not isinstance(found, list):
    return []
   return [field for field in found if field in fields_of(tool)]

  def free_texts(tool: str) -> list[str]:
   """Free-text argument names of a tool: string fields without an enum.
            Used to test the argument shape of search, message, details and view candidates."""
   return [field for field, spec in fields_of(tool).items()
     if isinstance(spec, dict) and spec.get("type") == "string" and not spec.get("enum")]

  roles: dict[str, str] = {}
  for tool in described:
   for field in required_of(tool):
    spec = fields_of(tool)[field]
    choices = [str(c).lower() for c in (spec.get("enum") or [])] if isinstance(spec, dict) else []
    both = {"order", "decline"} <= set(finish_choice_kinds(choices, leftover=False).values())
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
  """Search category label sharing the most kind words with wanted.
        Ties go to the label with the larger share of its own words matched; '' when none overlap."""
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
  """Simple (property, value) pairs from the remaining requirements, for product k in baskets.
        Strips source tags and skips references; keeps positive values or short ones without exclusion
        words. Feeds code-made search queries, picks and line repairs."""
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
  """Build up to three validated search argument sets for product k without the LLM.
        A keyword query from the first firm requirement value plus the product's kind word, a category
        filter with the closest label, and a shorter keyword variant; a per-unit price ceiling and mode
        choices are added where the schema has them. Each set passes ArgCheck.execute."""
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
   """Pick the mode enum option that a description clause names together with the given argument.
            Falls back to the last option when use_last is set, else the first."""
   options = choices.get(mode_field or "", [])
   for option in options:
    for clause in clauses:
     if re.search(r"\b" + re.escape(str(option).lower()) + r"\b", clause) and re.search(r"\b" + re.escape(arg.lower()) + r"\b", clause):
      return option
   return options[-1] if use_last else options[0]

  def about_price(field: str) -> bool:
   """Whether a numeric search field looks like a price ceiling, by its name or by a description
            clause that names it next to price or cost."""
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
   """Mode argument for a draft query, chosen by mode_value, or {} when the tool has no mode field.
            use_last makes the last option the fallback."""
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
  """Table of every known listing seen in archived records, keyed by listing id.
        Each row merges category, title, price, stock state, the kind words of all its text and the
        fields of nested records. Read by code_picks, stock rechecks and basket notes."""
  watch = self.run.watch
  known = self.run.archive.ids
  table: dict[str, dict[str, Any]] = {}

  def note(record: dict[str, Any]) -> None:
   """Merge one record into its listing's row: category, title, first price-like number, stock
            state, text words and nested field values. Records with unknown ids are ignored."""
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
  """Whether a table row shows that prop has value.
        Fields of that property decide when present; else the title; for unread rows also when all
        the value's kind words were among the search words that returned the listing."""
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
  """Whether text holds one alternative of value with all its words and numbers.
        Alternatives split on or, slash, comma and semicolon; limit phrases are dropped first and
        longer words match by stem."""
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
  """Rank how directly a row's fields show a value; lower is better.
        First part: 0 for a field named like the property, 1 for a similar name, 2 for title or none;
        second: 0 exact words, 1 a listed part matches, 3 merely contained. (2, 2) means title only."""
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
  """Listings the latest decline audit saw but did not name as possible fits.
        Live (checker-passed) ids are left out; empty when the last audit got no answer. code_picks
        skips these unless its own evidence is strong."""
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
  """Choose one listing per product from archived records without the LLM; returns (id, units).
        Skips checker-failed, refused, avoided, sold-out, off-category, clashing, reference-excluded
        and over-budget rows, then ranks by keep/passed, fit, price (cheapest), read state, evidence.
        Retries price-first if over budget; sets code_fit; [] when a product has no candidate."""
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
  """Whether a listing's own fields for prop contradict value.
        True only when such fields exist and none holds a word or figure of the value."""
  own_field = [text for key, text in fields.items() if same_prop(key, prop)]
  if not own_field:
   return False
  return not any(value_seen(value, text) for text in own_field)

 @staticmethod
 def choice_words(value: Any) -> set[str]:
  """Lowercase word set of an enum choice, with camelCase split apart.
        Used to spot the put and drop actions in a cart schema."""
  return set(re.findall(r"[a-z]+", re.sub(r"([a-z])([A-Z])", r"\1 \2", str(value)).lower()))

 def cart_form(self) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], str | None, list[str]]:
  """Read the cart tool schema into what code-made cart calls need.
        Returns the schema, enum values for putting and dropping (reusing the first put form the cart
        took), the listing id field and the numeric unit fields."""
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
  """Build validated cart calls that put the picked lines in with their units.
        One list-style call when the cart takes a line array, else one call per pick; [] when no id
        field exists or any call fails ArgCheck.execute."""
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
  """Build validated cart calls that remove the given lines.
        Uses the drop action with the held units, or sets units to 0 when the cart has no action field.
        [] for list-style carts or no lines; None when removal cannot be expressed or fails validation."""
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
  """Observations of the newest env entry in run.log, at most count of them.
        Returns [] when that entry holds no reply dict."""
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
  """Whether a tool observation reports an error.
        True when any record within two levels has an error-like key with a non-empty value."""
  flagged = []

  def look(record: dict[str, Any]) -> None:
   """Flag a record that carries an error-like key (FAULT_KEYS) with a non-empty value.
            Stops after the first such key in the record."""
   for key, value in record.items():
    if re.search(FAULT_KEYS, str(key).lower()) and value not in (None, False, "", [], {}):
     flagged.append(record)
     return
  walk_dicts(obs, look, 0, 2)
  return bool(flagged)

 def book_hits(self, calls: list[dict[str, Any]], replies: list[Any], text_fields: list[str]) -> None:
  """Record, for each code search, which listings its typed query words returned.
        Adds the query's kind words to code_hits for every known listing id in the reply; code_shows
        uses them as evidence for unread rows."""
  run = self.run
  hits = self.code_hits
  for call_args, obs in zip(calls, replies):
   typed = " ".join(str(call_args.get(field) or "") for field in text_fields)
   if not typed.strip():
    continue
   words = kind_words(typed)

   def mark(record: dict[str, Any]) -> None:
    """Add the call's query words to code_hits under the record's listing id, if the id is known.
                Applied to every record in the search reply."""
    ident = run.watch.row_mark(record)
    if ident in run.archive.ids:
     hits.setdefault(ident, set()).update(words)
   walk_dicts(obs, mark)

 def code_search(self, products: int, more: bool = False) -> None:
  """Run code-made search rounds for every product, keeping a reserve of moves.
        First use sends the interleaved code_queries and pages on from each product's lead query (more
        rounds for cheapest), stopping once all picks fit; more=True sends the stored next-page calls
        once. Records query hits unless the store's search matches any word."""
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
  """Build the arguments for the next results page of a search call, or None.
        Uses the store's paging argument (token, page number or offset) and the reply's markers or
        total count to tell whether more rows exist; the result passes ArgCheck.execute."""
  paging = self.run.watch.page_arg()
  if not paging:
   return None
  field, per_page, lowest, highest = paging
  pages: list[dict[str, Any]] = []

  def gather(record: dict[str, Any]) -> None:
   """Collect a record that holds a non-empty list of dicts.
            Such a record is taken as the page of result rows."""
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
  """Open the details of picked listings not read yet, re-picking after each read.
        Uses the batch tool for two or more, else the details tool; up to CODE_READS rounds, stopping
        when picks stop changing or moves or time run short. Returns the final picks."""
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
  """Make the cart hold exactly the picks, using code-made cart calls.
        Removes surplus lines and adds or resizes the rest in one chunk, then checks faults and the
        cart view. Returns picked ids that did not land (dropped from trolley.units), [] on success,
        or None when the calls cannot be built or sent."""
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
  """Re-read the given lines with the stock, view or details tool to refresh price and stock.
        Returns (whether calls were sent, ids now shown out of stock)."""
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
  """Take a new budget from shopper lines by regex during the forced finish.
        Reads the given lines and held pending lines once each; a positive amount from budget_in
        becomes needs.spend_cap."""
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
  """Fill the cart by code alone; True when the cart ends up holding lines.
        Needs a cart tool, moves and time; turns off model calls (wire.mute) and runs shop_rounds.
        Used by finish_now and by the sold-line gate to swap out dead lines."""
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
  """Up to four pick-and-cart rounds for code_cart.
        Each round picks, searches or pages on when picks are weak, opens unread picks, puts them in
        the cart and rechecks stock; failed or sold lines join avoid for the next round."""
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
  """Order move for the given tool, with arguments from schema_args in order mode.
        Text slots carry a short order note."""
  return {"name": tool, "args": self.schema_args(tool, "Order the cart.", "order")}

 def decline_move(self, tool: str) -> dict[str, Any]:
  """Decline move for the given tool, with arguments from schema_args in decline mode.
        Text slots carry the fixed SORRY message."""
  return {"name": tool, "args": self.schema_args(tool, SORRY, "decline")}

 def stop_finishing(self) -> bool:
  """Whether the forced finish must stop now.
        True once the episode is done, the wire broke, no moves remain or under 2 s of time is left."""
  wire = self.run.wire
  return wire.done or wire.broke or self.run.moves_left() <= 0 or wire.spare() < 2

 def finish_now(self) -> None:
  """Force an ending when the turn loop stops without one; also used after a crash.
        Up to CODE_FINISHES moves: a ready, complete cart (or one filled by code_cart) is ordered, with
        the go-ahead first when due, else it declines. After a refused order it rereads budget news,
        stock and prices, then reorders or swaps lines; a last order retry or decline closes the run."""
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
     cart_ready = False
     keep = set(trolley.units) if keep is None else keep
     run.log.append({"kind": "code_partial", "faults": run.basket_short()})
    if cart_ready or (order_tool and self.code_cart(avoid, keep)):
     move = self.order_move(order_tool)
    elif order_tool and turn > 0 and self.again_ok(avoid):
     move = self.order_move(order_tool)
    elif decline_tool:
     move = self.decline_move(decline_tool)
    else:
     return
   if run.call_kind(move["name"], move["args"]) == "order":
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
  """Whether the held cart may be ordered again after a refusal.
        Needs a non-empty cart with no avoided or dead line, and code_picks still finding a basket
        without them."""
  held = set(self.run.trolley.units)
  ruled_out = set(avoid) | self.run.shelf.dead_lines()
  if not held or held & ruled_out:
   return False
  return bool(self.code_picks(ruled_out))

 def catch_up(self) -> None:
  """Fold late model answers back in at the start of each turn when models are reachable.
        Re-reads the request once with the READER seat if the first reading came from code, and
        passes held shopper lines to needs.hear; notes are appended to the latest chat message."""
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
     notes.append("[plan updated] The models answer again and have re-read the request; this reading replaces the first:\n"
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
 """Reads world tool results for the run and turns them into store knowledge and actor remarks.
    Detects refused orders and declines, shopper text pushed with results, stock fields and
    category labels; books search pages per listing and infers how keyword search treats words
    (any vs all, first N words counted) and how it pages; also drives the cheapest shelf sweep.
    One instance lives on the run as run.watch."""
 def __init__(self, run):
  """Binds the observer to the run and resets its per-episode memory.
        hit_shelf maps hit ids to labels, pointed_memo caches pointed_words, refusal_texts holds the
        last refusal's wording, and sweep_tried, sweep_turn and swept track the shelf sweep."""
  self.run = run
  self.hit_shelf = {}
  self.pointed_memo = None
  self.refusal_texts = None
  self.sweep_tried = set()
  self.sweep_turn = 0
  self.swept = 0

 def refused(self, call_name: str, obs: Any, resp: dict[str, Any]) -> bool:
  """Decides whether a tool result counts as a refusal by the store.
        A dict with a 'refusal' key always does; an order-role call that did not finish the episode
        does too, unless it failed with invalid_args or unknown_tool. Called from result handling."""
  if isinstance(obs, dict) and "refusal" in obs:
   return True
  if call_name != self.run.role("order") or resp.get("finished"):
   return False
  return not (isinstance(obs, dict) and obs.get("error") in ("invalid_args", "unknown_tool"))

 @staticmethod
 def wordy_texts(obs: Any) -> list[str]:
  """Collects every string of four or more words found inside a nested observation.
        Walks dict values and the first 20 items of each list, down to depth 6, keeping found order."""
  found: list[str] = []

  def collect(piece: Any, depth: int = 0) -> None:
   """Recursive visitor for wordy_texts: descends into dict values and list heads and appends each
            string of at least four words to the outer list; stops below depth 6."""
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
  """Records a refused order on the run and returns the remark for the actor.
        Bumps refusal_n, clears order_vetted, sets event_seen, marks every cart line as refused, saves
        the refusal's wordy texts for shopper_push_note, and adds the hint from shelf.refusal_reread."""
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
  return ("ORDER REFUSED; the episode continues and the cart is unchanged. Apply each change the refusal names, "
    "fix the cart (remove any line that is no longer acceptable, a sold-out item for instance), check the new pick's stock and "
    "price again, and send the order once more." + hint)

 def shopper_push_note(self, obs: dict[str, Any]) -> str | None:
  """Builds a remark when a result carries shopper text in a list-of-strings field, else None.
        If every pushed line repeats the wording of the refusal just seen, says it adds nothing new;
        otherwise sets event_seen and asks the actor to apply up to four lines. Clears refusal_texts."""
  pushed: list[str] = []
  for value in obs.values():
   if isinstance(value, list) and value and all(isinstance(x, str) for x in value):
    long_strings(value, pushed)
  if not pushed:
   return None
  earlier = self.refusal_texts or set()
  self.refusal_texts = None
  if earlier and all(any(" ".join(str(s).lower().split()) in text for text in earlier) for s in pushed):
   return ("This shopper text repeats what the refused finish already brought (dealt with above); it says "
     "nothing about this result.")
  self.run.event_seen = True
  return "This result comes with a shopper message (a change). Apply it: " + " | ".join(pushed[:4])

 def notes_for(self, call_name: str, obs: Any, refused: bool) -> str:
  """Assembles the extra remarks attached to one tool result and updates run flags.
        budget_exceeded warns off the tool and can lower run.msg_cap to messages sent; a refused decline
        resets audit_done and decline_vetted; a refused order goes to order_refusal_note; other dict
        results with 2+ keys (not from the message tool) are checked by shopper_push_note."""
  run = self.run
  remarks = []
  if isinstance(obs, dict) and obs.get("error") == "budget_exceeded":
   if call_name == run.role("message") or run.msg_cap is not None:
    run.msg_cap = run.msgs_sent if run.msg_cap is None else min(run.msg_cap, run.msgs_sent)
   remarks.append("This budget is spent; do not use this tool again.")
  if refused and call_name == run.role("decline") and run.last_kind == "decline":
   run.audit_done = False
   run.decline_vetted = False
   run.event_seen = True
   remarks.append("DECLINE REFUSED; the episode continues, as the store thinks something can still be bought. "
         "Follow the refusal, re-examine the closest candidates (own properties, stock "
         "and price measured against the requirements now), and buy the one that fits; decline again only when "
         "every one of them truly fails.")
  elif refused:
   remarks.append(self.order_refusal_note(obs))
  elif isinstance(obs, dict) and len(obs) >= 2 and call_name != run.role("message"):
   pushed = self.shopper_push_note(obs)
   if pushed:
    remarks.append(pushed)
  return " ".join(remarks)

 @staticmethod
 def shelf_of(row: dict[str, Any]) -> str:
  """Returns a record's category label, lowercased, or an empty string.
        Uses the first non-empty string of up to 60 chars under a key naming a category, shelf,
        department or aisle, skipping keys that contain 'id'."""
  for key, value in row.items():
   if not isinstance(value, str) or not 0 < len(value.strip()) <= 60:
    continue
   name = str(key).lower()
   if re.search(r"categor|shelf|department|aisle", name) and "id" not in name:
    return value.strip().lower()
  return ""

 @staticmethod
 def key_stock(key: Any, val: Any) -> bool | None:
  """Reads one field as a stock verdict: True in stock, False out, None if not about stock.
        Treats true out/sold flags, false or non-positive stock and on-hand values, negative stock
        words, and backorder, preorder or discontinued text as out of stock."""
  name = str(key).lower()
  if val is None or not re.search(r"stock|avail|on.?hand", name):
   return None
  out = False
  if re.search(r"stock|avail", name):
   if re.search(r"out|sold", name):
    out = val is True
   elif val is False or (is_number(val) and val <= 0):
    out = True
   elif isinstance(val, str):
    out = bool(re.search(r"\b(?:out|sold|none|unavailable|not|no|false)\b", val.lower()))
  if re.search(r"on.?hand", name) and (val is False or (is_number(val) and val <= 0)):
   out = True
  if isinstance(val, str) and re.search(r"back.?order|pre.?order|discontinu", val.lower()):
   out = True
  return not out

 @staticmethod
 def stock_state(row: dict[str, Any]) -> bool | None:
  """Combines key_stock over every field of a row.
        Returns None when no field speaks about stock, else True only if all stock fields say in stock."""
  says = [Observer.key_stock(key, val) for key, val in row.items()]
  says = [s for s in says if s is not None]
  if not says:
   return None
  return all(says)

 @staticmethod
 def sold_out(row: dict[str, Any]) -> bool:
  """True when any field of the row reads as out of stock by key_stock.
        Rows with no stock field count as not sold out."""
  for key, val in row.items():
   if Observer.key_stock(key, val) is False:
    return True
  return False

 def read_label_tally(self) -> dict[str, int]:
  """Counts, per category label, the distinct read listings seen in the archived records.
        Only ids in archive.read count, once each, skipping ids that ReferenceDesk.ref_foreign flags."""
  archive = self.run.archive
  tally: dict[str, int] = {}
  counted: set[str] = set()
  foreign = self.run.refdesk.ref_foreign()

  def count(record: dict[str, Any]) -> None:
   """Visitor for read_label_tally: finds the record's read id, skips counted or foreign ids, and
            adds one to the tally of the record's category label."""
   ident = next((value for key, value in record.items()
        if isinstance(value, str) and "id" in str(key).lower() and value in archive.read), None)
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
  """Picks the category labels taken as the request's own shelf.
        One product: the labels with the most read listings. Several products: for each home shelf from
        shelf.home_shelves its most-read known label (or the whole home if none was read), topped up
        with frequent labels; falls back to the most-read labels on any error."""
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
  """Returns a row's identifier as a string.
        Prefers a key named exactly 'id', then any key ending in 'id' (string or int, not bool);
        with neither, falls back to the whole row as sorted JSON."""
  for whole_name in (True, False):
   for key, val in row.items():
    name = str(key).lower()
    matches = name == "id" if whole_name else name.endswith("id")
    if matches and isinstance(val, (str, int)) and not isinstance(val, bool):
     return str(val)
  return json.dumps(row, sort_keys=True, default=str)

 def search_keys(self, call_args: dict[str, Any]) -> tuple[list[tuple], tuple, bool, set]:
  """Splits a search call's arguments by role using the search tool's schema.
        Returns (texts, modes, first_page, paging): (field, unique words) for free text made only of
        2+ char alphanumerics, sorted enum (mode) pairs, whether the call asks for the first page or an
        empty cursor, and the paging field names (cursor, page or offset; not size, limit or price)."""
  run = self.run
  schema = run.shapes.get(run.role("search") or "")
  fields = schema.get("properties") if isinstance(schema, dict) else None
  if not isinstance(fields, dict):
   fields = {}

  def spec_of(field: str) -> dict[str, Any]:
   """Returns the schema entry of one search field, or an empty dict when missing or not a dict.
            Lets search_keys read a field's enum, minimum and default."""
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
  """Estimates a search result's total match count.
        Takes the largest plain int at the observation's top level, or the row count if that is larger."""
  return max([v for v in obs.values() if isinstance(v, int) and not isinstance(v, bool)] + [len(rows)])

 def word_rule(self, call_args: dict[str, Any], obs: dict[str, Any], rows: list) -> str:
  """Infers whether the store's keyword search needs all query words or any of them.
        Compares first-page totals with earlier searches of the same field, mode and other args where
        one word list extends the other; sets run.word_kind to 'all' or 'any' and logs run.word_seen.
        Returns run.word_kind; called from Coverage.search_note on each search."""
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
  """Infers whether the keyword search only counts the first N words of a query.
        Against earlier first-page queries in the same context, differing totals or ids prove a lower
        bound, while identical results past a shared prefix suggest a cut-off. Sets run.lead_words and
        run.lead_ctx (field, mode) from the tightest claim; returns lead_words, 0 when unknown."""
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
  """Returns the properties dict of the search tool's parameter schema, or an empty dict.
        Finds the tool through run.role('search') in run.shapes."""
  found = (self.run.shapes.get(self.run.role("search") or "") or {}).get("properties")
  return found if isinstance(found, dict) else {}

 def token_key(self) -> str | None:
  """Finds the search field that carries a paging cursor or bookmark.
        Returns the first non-enum string field whose name looks like a cursor, token, next, page or
        offset marker, or None."""
  for field, spec in self.search_fields().items():
   if not isinstance(spec, dict) or spec.get("type") != "string" or spec.get("enum"):
    continue
   if re.search(r"bookmark|cursor|token|resume|continu|after|next|page|offset|start", str(field).lower()):
    return field
  return None

 def page_arg(self) -> tuple[str, Any, Any, float | None] | None:
  """Describes how the search tool pages its results, or returns None if it cannot page.
        Prefers a numeric page/offset field: (field, whole_pages, first value, maximum or None);
        otherwise falls back to the cursor field as (field, 'token', '', None)."""
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
  """Drops a trailing plural 's' from words longer than three letters, leaving '-ss' endings alone.
        Used to compare query, label and title words loosely."""
  if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
   return word[:-1]
  return word

 def shelf_rows(self) -> dict[str, tuple[float, str, str]]:
  """Collects the priced, in-stock archived listings on the request's shelf labels.
        Returns id -> (price, title, maker); skips refused lines, removes rows later seen sold out, uses
        the latest archived price and drops rows whose price times quantity exceeds the budget."""
  run = self.run
  archive = run.archive
  labels, budget, units = self.shelf_labels(), run.needs.spend_cap, run.needs.quantity()
  rows: dict[str, tuple[float, str, str]] = {}
  if not labels:
   return rows

  def take(record: dict[str, Any]) -> None:
   """Visitor for shelf_rows: for a record with a known id on the shelf labels, removes it if sold
            out, else stores its price, title (title field or a name-like key) and brand or maker text."""
   ident = next((value for key, value in record.items()
        if isinstance(value, str) and "id" in str(key).lower() and value in archive.ids), None)
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
  """Builds short search texts from the words of the shelf labels.
        Splits each label at link words and punctuation; offers the first two words together, each
        group's last word, then the remaining words; returns them deduplicated in order."""
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
  """Mines recurring title words (pairs False) or adjacent word pairs (pairs True) from shelf_rows.
        Skips short, link and maker words and grams made only of label words; keeps grams found in two
        or more titles, ordered by lowest price then frequency. Feeds the shelf sweep queries."""
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
  """Chooses the next search call for the cheapest shelf sweep; returns (call_args, from_listing).
        A price-filter call from Coverage.ceiling_sweep comes first; then the next page of the largest
        id-ordered shelf listing, then title-pair, title-word and label queries (rotated by turn) on
        the latest free-text search; skips calls already tried or whose lead words were read."""
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
    """Tells whether an argument of the template search stays in the sweep call.
                Keeps non-strings, enum fields, values equal to a shelf label, and the page field."""
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
   """Turns one query text into sweep call args on the template, or returns None.
            Rejects texts when there is no template or no lead words, when a first page with the same lead
            words was already read, or when the call was already tried."""
   signature = tuple(self.stem(w) for w in re.findall(r"[a-z0-9]{2,}", text.lower())[:2])
   if template is None or not signature or signature in covered:
    return None
   call_args = {**template, text_field: text}
   if json.dumps(call_args, sort_keys=True, default=str) in tried:
    return None
   return call_args

  def from_texts(texts: list[str]) -> list[dict[str, Any]]:
   """Maps a list of query texts through as_call and keeps only the usable call args.
            Serves the title and label sources of sweep_next."""
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
  """True while more than reserve steps remain and at least CLOSING + 40 seconds are left.
        Guards each extra shelf sweep read."""
  run = self.run
  return run.moves_left() > reserve and run.wire.spare() >= CLOSING + 40

 def sweep_step(self) -> bool:
  """Runs one scripted shelf-sweep search when the cheapest is wanted and no property need is left.
        Needs a dropped requirement, a search tool and a step reserve (larger before the first order),
        up to SWEEP_READS reads. Dispatches the sweep_next call; once no further read fits, appends the
        six cheapest shelf rows to the reply. Returns True if a call ran (used by ShelfSweepStep)."""
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
  run.chat.append({"role": "assistant", "content": "No property requirement is left, so the shelf's cheapest in-stock listing "
       "wins: I read another slice of the shelf before I choose.", "tool_calls": [proposal]})
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
    replies[-1] = (cid, text + "\n[shelf sweep done] The cheapest in-stock rows of the requested category within "
          "the budget, collected across every page seen: " + listed + ". Choose the cheapest unless a change excludes "
          "it: make it the cart's sole line, check its stock and price again, then order it.")
  run.talk.post_replies(replies)
  return True

 def free_text(self, call_args: dict[str, Any]) -> list[str]:
  """Returns the free-text values of a search call: string args that are neither enum fields nor
        the paging cursor field."""
  fields = self.search_fields()
  token = self.token_key()
  return [value for key, value in call_args.items()
    if isinstance(value, str) and not (fields.get(key) or {}).get("enum") and key != token]

 @staticmethod
 def next_token(obs: Any) -> str | None:
  """Pulls the next-page cursor out of a search result.
        Returns the first non-empty string under a key like next, cursor, bookmark, token or more."""
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
  """Sort key that splits an id into digit and text runs so that numbers compare numerically.
        Used by book_page to test whether rows come in id order."""
  return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", mark)]

 def book_page(self, call_args: dict[str, Any], rows: list, total: int, obs: Any = None) -> None:
  """Records one search result page in run.page_book, keyed by the call args minus the paging field.
        Tracks pages read, cursors, page size, total, shelf labels, row id -> label, first-page ids and
        all rows when one page holds them; id_order stays True only while pages of 5+ rows are id-sorted.
        Called from Coverage.search_note."""
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
  if len(marks) < 5:
   return
  ascending = marks == sorted(marks) or marks == sorted(marks, key=self.natural_key)
  walk["id_order"] = walk.get("id_order", True) is not False and ascending

 def next_page(self, walk: dict[str, Any]) -> Any:
  """Returns the next unread page position of a booked listing, or None when it is exhausted.
        Cursor stores give an unread next token; numeric stores step from the lowest read start by one
        page or one page size, stopping at the total or the schema maximum."""
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
  """Loose word match: both words have three or more letters and one is a prefix of the other.
        Used by value_score and walk_carries to match query words to requirement values."""
  return min(len(a), len(b)) >= 3 and (a.startswith(b) or b.startswith(a))

 def value_score(self, call_args: dict[str, Any]) -> int:
  """Counts the open requirement values that a search query carries.
        Matches each non-reference value's 3+ char words against the query's free-text words (cut to
        run.lead_words when known) with close_words; returns how many values were hit."""
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
  """Picks from result rows the ids of unread, in-budget, in-stock listings on the shelf labels.
        Stores each hit's label in hit_shelf. With references in the request, unread rows whose seller
        or brand words are named by the query and appear in pointed_words go to run.ref_hits.
        Called by Coverage.book_hits and Coverage.earlier_hits."""
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
  """Builds the vocabulary that may point at another seller: words of the request's references and
        of the body fields of read records, minus each record's own seller words.
        Memoized on the archive size; an error keeps the words gathered so far."""
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
    """Visitor for pointed_words: for a read record, sets aside short seller or brand fields and adds
                the words of its other non-title text fields, minus that seller's words."""
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
  """Returns up to most next-page search calls for booked listings that carry required values.
        Skips exhausted listings and those on other shelves; ranks by value_score, then smaller total.
        Used by the decline audit follow-ups."""
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
  """Token match for value_probe: tokens with digits match up to leading zeros, or when a pure
        number prefixes the other without running into more digits; words need a 3+ letter prefix."""
  if any(c.isdigit() for c in a + b):
   if a.lstrip("0") == b.lstrip("0"):
    return True
   return a.isdigit() and b.startswith(a) and not b[len(a):len(a) + 1].isdigit()
  return min(len(a), len(b)) >= 3 and (a.startswith(b) or b.startswith(a))

 def value_probe(self, raw: list[str]) -> tuple[str, str]:
  """Spots a query led by a property's name instead of its value; returns (property, probe query).
        Gives up in any-word stores, when the lead already holds a required value or product word, or
        when a booked search already used the probe's lead words. The probe is the value (from its first
        digit) plus the product words. Used by Coverage.hint_remarks."""
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
 """Cheapest-price coverage for stores whose search returns rows in a fixed id order.
    Plans and reads extra pages of searches that carry required values, searches the pick's
    product line, and queues unread rows priced under the current pick for opening.
    Also prepares pre-decline work (audit fits, extra page reads, unrefuted candidates) and query
    hints (word traps, combined queries). Lives on the run as run.walk."""
 def __init__(self, run):
  """Binds the walker to the run and resets its state.
        decline_queue holds planned pre-decline reads, decline_read_done and fit_held make those notes
        one-shot, and walked records which products were already walked."""
  self.run = run
  self.decline_queue = None
  self.decline_read_done = None
  self.fit_held = None
  self.walked = set()

 def ran_query(self, text: str) -> bool:
  """Tells whether a booked search already used the same leading words as text.
        Compares sorted stems of the first lead_words (default 3) words, so word order is ignored."""
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
  """Finds a booked search to reuse as a template for new queries.
        Takes the latest listing that is not a price-filter listing and has exactly one free-text
        field; returns (the other args, that field's name) or None."""
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
  """Token match for find_trap: tokens with digits must be equal up to leading zeros; words need
        three or more letters and a shared prefix."""
  if any(c.isdigit() for c in a + b):
   return a.lstrip("0") == b.lstrip("0")
  return min(len(a), len(b)) >= 3 and (a.startswith(b) or b.startswith(a))

 @staticmethod
 def token_parts(token: str) -> list[str]:
  """Splits a token into its alphanumeric runs of two or more characters.
        Used by find_trap to compare query tokens with requirement words."""
  return [p for p in re.findall(r"[a-z0-9]+", token) if len(p) >= 2]

 def word_trap(self, raw: list[str]) -> tuple[str, str]:
  """Error-safe wrapper around find_trap; logs a trap_error entry and returns empty strings.
        Called from Coverage.hint_remarks after a first-page search."""
  run = self.run
  try:
   return self.find_trap(raw)
  except Exception as err:
   run.log.append({"kind": "trap_error", "error": f"{type(err).__name__}: {err}"[:300]})
   return "", ""

 def find_trap(self, raw: list[str]) -> tuple[str, str]:
  """Flags query words that wrongly narrow an all-words search; returns (remark, new query).
        A word is flagged when it comes from a required property's name or only from a dropped
        requirement, not from a value or the item kind. The new query keeps the other words (adding
        the product kind if missing); nothing is returned in any-word stores or if it was already run."""
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
   note = (f"This query's word '{word}' is drawn from a required property's NAME and not from a required value. Since "
     "the store needs all words, it keeps only listings that repeat that name in their fields or title, missing "
     "those whose field holds only the value.")
  else:
   note = (f"The query word '{word}' comes only from a requirement the shopper DROPPED, so it still limits results to "
     "listings carrying it, though that is no longer needed.")
  return note + f" Search '{query}' too, and read all its pages.", query

 def combined_queries(self) -> list[dict[str, Any]]:
  """Builds up to two searches that join short values of two or more open requirements.
        Adds the product kind word, stays within lead_words (default 3) and skips queries already run.
        Per product only where no passed line covers its shelf; for one product only while no passed
        item is live. Used by the decline audit follow-ups."""
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
  """Gets a second item check when the decline audit and an earlier check disagree on an item.
        Re-runs judge.check_item with both readings (the last audit reason and the earlier ruling);
        returns the new verdict, or clears the item's memoized rulings and returns '' when it passes.
        On error it logs and keeps the original ruling."""
  run = self.run
  try:
   reason = ""
   for entry in reversed(run.log):
    if isinstance(entry, dict) and entry.get("kind") == "audit" and isinstance(entry.get("result"), dict):
     reason = str((entry.get("result") or {}).get("reason") or "")
     break
   extra = ("\n\nCONFLICTING READINGS OF THIS ITEM. Goal: decide which reading of these same records is correct. 1. The "
      "decline audit found that it meets every requirement" + (f" ({reason[:400]})" if reason else "")
      + "; an earlier check reported: " + ruling[:500]
      + "\n2. Decide by the records: a field with the required value in other wording, or a shade, qualifier or "
      "spelling of that same named value, satisfies the requirement; a field with another value fails it.")
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
  """Error-safe wrapper around decline_candidates; logs and returns an empty list on failure.
        Used by the run's best_gate before a decline."""
  run = self.run
  try:
   return self.decline_candidates()
  except Exception as err:
   run.log.append({"kind": "unrefuted_error", "error": f"{type(err).__name__}: {err}"[:300]})
   return []

 def decline_candidates(self) -> list[str]:
  """Finds, before a decline, one candidate per product that no check has ruled out.
        For each home shelf takes the cheapest read listing in stock and in budget that is not failed,
        refused, a reference pointer or passed over by the audit, preferring items that passed checks.
        Returns [] unless every product gets one and their total fits the budget."""
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
   """Visitor for decline_candidates: records, by id, whether each archived record that has a stock
            field is sold out."""
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
  """Error-safe wrapper around hold_for_fit; logs and returns an empty string on failure.
        Called by the run's fit_gate before a decline."""
  run = self.run
  try:
   return self.hold_for_fit()
  except Exception as err:
   run.log.append({"kind": "fit_hold_error", "error": f"{type(err).__name__}: {err}"[:300]})
   return ""

 def hold_for_fit(self) -> str:
  """Once per run, with an empty cart and time left, re-checks up to two fits the decline audit named.
        Uses judge.check_item and arbitrate and keeps the items that pass; returns a remark asking the
        actor to buy them before declining, or '' (several products: only if every shelf is covered)."""
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
  return ("The decline audit named " + ", ".join(good) + " as possibly fitting, and the independent item check read "
    + ("its" if one else "their") + " full record and found " + ("it meets" if one else "they meet")
    + " each requirement now in force. Review " + ("it" if one else "them") + " before declining: if it fits, put "
    + target + " in the cart, check its stock and price again, and order; if you find it fails a "
    "requirement in force, decline.")

 def id_store(self) -> bool:
  """True when the store's search sorts rows by id: no booked listing was seen out of id order and
        at least two were seen in it."""
  flags = [walk.get("id_order") for walk in self.run.page_book.values() if "id_order" in walk]
  return False not in flags and flags.count(True) >= 2

 def product_index(self, thing: str) -> int | None:
  """Maps an item to the index of the product whose home shelf holds the item's category.
        Returns None for single-product requests or when the item's label is unknown or unmatched."""
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
  """Returns the category labels that count as product k's shelf, from shelf.home_shelves.
        For k None an empty home falls back to Observer.shelf_labels; an unknown product index gives
        an empty set."""
  homes = self.run.shelf.home_shelves()
  slot = k if k is not None else 0
  home = set(homes[slot]) if homes and (k or 0) < len(homes) else set()
  if home:
   return home
  return set(self.run.watch.shelf_labels()) if k is None else set()

 @staticmethod
 def fig_amount(text: Any) -> tuple[str, str] | None:
  """Parses a requirement part that starts with a number followed by a word.
        Returns (number, word), or None when the text holds a comparison bound or lacks that start."""
  lowered = str(text).lower()
  if re.search(CH_BOUND, lowered):
   return None
  hit = re.match(r"\s*(\d+)\s+([a-z]+)", lowered)
  if hit is None:
   return None
  return hit.group(1), hit.group(2)

 @staticmethod
 def amount_part(text: Any) -> bool:
  """True when a requirement part is an amount: it has a digit and either a comparison bound word
        or starts with a number followed by a word."""
  lowered = str(text).lower()
  if not re.search(r"\d", lowered):
   return False
  return bool(re.search(CH_BOUND, lowered) or re.match(r"\s*\d+(?:[.,]\d+)?\s+[a-z]", lowered))

 def leading_words(self, call_args: dict[str, Any], pattern: str) -> list[str]:
  """Returns the words matching pattern in a search call's free text, cut to the first lead_words
        when the store's lead limit is known."""
  run = self.run
  found = [w for text in run.watch.free_text(call_args) for w in re.findall(pattern, text.lower())]
  return found[: run.lead_words] if run.lead_words else found

 def walk_carries(self, args: dict[str, Any], k: int | None, alts: bool = False) -> set[Any]:
  """Lists which open requirements a search's leading words carry.
        Number-plus-word parts need both tokens; other amounts are skipped; plain parts match by
        close_words. Returns requirement indexes, or (index, alternative) pairs when alts is True;
        with several products only product k's requirements count."""
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
  """Number of open requirements the search carries for product k, from walk_carries.
        Used to rank walk options."""
  return len(self.walk_carries(args, k))

 def walk_options(self, k: int | None) -> list[tuple[int, str, dict[str, Any]]]:
  """Lists listings worth reading further for product k as (pages left, book key, walk).
        Takes id-ordered listings with unread pages that carry a requirement and show rows on k's shelf;
        drops ones shadowed by a fully read listing, too long, or duplicates; in all-words stores also
        drops nested queries. Sorted by pages left, then walk_score."""
  run = self.run
  watch = run.watch
  home = self.walk_home(k)
  if not home:
   return []
  depth = run.lead_words or None

  def lead(walk: dict[str, Any]) -> set[str]:
   """Returns the stemmed leading words (up to lead_words) of a booked listing's query text.
            Used to compare queries for shadowing and nesting."""
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
    """Tells whether option o is redundant beside option p in an all-words store.
                True if p's words are a strict subset of o's, o adds non-value words and p is short enough,
                or if p strictly extends o's words with only value and label words."""
    mine, other = lead(p[2]), lead(o[2])
    if mine < other and not (other - mine) <= allowed and p[0] <= WALK_PAGES:
     return True
    return other < mine and (mine - other) <= allowed
   options = [o for o in options if not any(nested(p, o) for p in options if p is not o)]
  return sorted(options, key=lambda p: (p[0], -self.walk_score(p[2]["args"], k), p[1]))

 def walk_plan(self, budget: int, only: set | None = None) -> list[tuple[int | None, str, int]]:
  """Plans the cheapest walk within a page budget as a list of (product k, book key, pages).
        Only for cheapest-priority requests in id-ordered stores that page; takes each unwalked
        product's best walk option (optionally only those in only), fewest pages first, WALK_PAGES max."""
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
  """Returns the current cheapest pick for product k as (id, price), or None.
        Looks at live cart lines first, else items that passed the item check for this needs version;
        skips dead and refused lines."""
  run = self.run
  sold = run.shelf.dead_lines()

  def mine(ident: str) -> bool:
   """Tells whether an item belongs to product k, using product_index.
            Every item does when k is None."""
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
  """Builds a search for the pick's product line: the maker's first word plus the first distinctive
        title word after it (not a shelf, requirement or product word).
        Without a maker it defers to Coverage.title_line; returns None if there is no pick or template,
        or the query was already run."""
  run = self.run
  needs = run.needs
  pick = self.walk_pick(k)
  template = self.search_template()
  if not pick or not template:
   return None
  record: dict[str, Any] = {}

  def find(candidate: dict[str, Any]) -> None:
   """Visitor for line_query: keeps the archived record whose id is the pick and that has a title.
            The last match found wins."""
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
  """Lists unread rows worth opening for product k, cheapest first, at most most.
        Takes rows on k's shelf from the best-carrying id-ordered listings or the given book keys; keeps
        in-stock, in-budget, unrefused rows priced below the pick (or the cheapest read row)."""
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
   """Visitor for walk_rows: records the sold-out state of archived records whose id field is among
            the shown rows."""
   if any(isinstance(v, str) and v in shown and "id" in str(key).lower() for key, v in record.items()):
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
  """Condenses a search observation into one line of up to 12 rows: id, price, sold-out flag, title
        and category. Replaces the raw page text for scripted walk reads."""
  run = self.run
  watch = run.watch
  rows: list[dict[str, Any]] = []

  def gather(record: dict[str, Any]) -> None:
   """Visitor for page_digest: keeps records with four or more fields and an id field holding a
            known archive id."""
   if len(record) < 4:
    return
   if any(isinstance(v, str) and v in run.archive.ids and "id" in str(key).lower() for key, v in record.items()):
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
  """Runs one scripted search call with the given args, with say as the assistant text.
        Dispatches with run.covering set and writes the tool reply itself: a page_digest when a record
        was archived, plus any shopper or refusal note. Returns True if the archive grew."""
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
  """True while the episode is not finished and at least CLOSING + 30 seconds remain.
        Gates the line searches and ceiling walks of run_walk."""
  wire = self.run.wire
  return not wire.done and wire.spare() >= CLOSING + 30

 @staticmethod
 def interleave(pools: list[list[str]], most: int) -> list[str]:
  """Merges id pools round-robin into one queue of at most most ids, skipping duplicates.
        Consumes the pools in place."""
  queue: list[str] = []
  while len(queue) < most and any(pools):
   for pool in pools:
    while pool and pool[0] in queue:
     pool.pop(0)
    if pool and len(queue) < most:
     queue.append(pool.pop(0))
  return queue

 def trim_queue(self, queue: list[str], batch: Any, floor: int) -> None:
  """Drops ids from the end of an open queue until the reads it needs fit in the steps left above
        floor; with a batch tool, two or more ids take one call."""
  while queue and (1 if batch and len(queue) >= 2 else len(queue)) >= self.run.moves_left() - floor:
   queue.pop()

 def read_pages(self, plan: list[tuple[int | None, str, int]], field: str, floor: int, report: list[str]) -> bool:
  """Reads the planned extra pages of each walk listing through walk_call.
        Marks each product as walked, stops at the step floor, the time limit or a failed call, and adds
        a progress line per listing to report. Returns True if any page was read."""
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
    said = ("The cheapest is wanted, but this store lists search results by fixed id, not price: I go on to the "
      f"next page of '{query}'.")
    if not self.walk_call({**walk["args"], field: following}, said):
     break
    read += 1
    progressed = True
   report.append(f"'{query}': read {read} more page(s), seen {len(walk.get('row_shelf') or {})} of its {walk['total']} matches")
  return progressed

 def read_lines(self, plan: list[tuple[int | None, str, int]], field: str, floor: int, report: list[str],
       lines: dict[Any, str]) -> bool:
  """Searches each planned product's pick line (line_query), plus up to two pages when it is short.
        Stores the line's book key in lines[k] and a note in report; returns True if any line ran."""
  run = self.run
  wire = run.wire
  progressed = False
  for k, _, _ in plan:
   if run.moves_left() <= floor + 2 or not self.walk_open():
    break
   call_args = self.line_query(k)
   if not call_args:
    continue
   said = ("I search the pick's product line as well (its maker plus the word naming the line): a line's other sizes and "
     "colours are listings of their own and may be cheaper.")
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
  """Drives one cheapest walk: planned page reads, line searches, and Coverage.ceiling_walk for ceil.
        Then queues the cheapest unread rows below each pick for opening (purpose 'walk') through
        judge.open_named, resets rank checks for those shelves and order_vetted, appends a summary to
        the last tool reply and logs it. Returns True if anything ran. Called from Coverage."""
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
   middle = ("Next I open the cheapest of the unread in-stock rows in budget that these searches showed under each "
       "pick's price: " + listed + ". ")
  else:
   middle = "No unread in-stock row in budget and under the pick's price turned up. "
  summary = ("[cheapest walk] " + "; ".join(report) + ". " + middle
       + "For each product, choose the lowest-priced listing whose own fields meet all requirements in "
       "force; the rank check is repeated on the cart call and on the order.")
  if queue:
   reads = run.reads
   reads.open_queue = queue
   reads.open_recheck = True
   reads.open_say = "Since the cheapest is wanted, I open the walk's cheapest unread rows."
   reads.open_purpose, reads.open_floor = "walk", floor
   run.judge.open_named()
  if run.chat and run.chat[-1].get("role") == "tool":
   tail = "" if queue else "\n" + run.status_line()
   run.chat[-1]["content"] = str(run.chat[-1]["content"]) + "\n" + summary + tail
  run.log.append({"kind": "walk", "plan": [[k, n] for k, _, n in plan], "ceiling": list(ceiling_rows), "opened": queue})
  return True

 def decline_fits(self) -> set[str]:
  """Returns the items that count as fits before a decline: those that passed the item check for
        this needs version plus audit-named fits not ruled out, minus refused lines."""
  run = self.run
  version = str(run.needs.version)
  live = {k.rpartition("|")[0] for k in run.passed if k.rpartition("|")[2] == version}
  failed = {k.split("|")[0] for k, v in run.rulings.items() if v and k.split("|")[1:2] == [version]}
  named = set(run.judge.audit_fits or []) - failed
  return (live | named) - run.refused_lines

 def decline_lacking(self) -> list[int | None]:
  """Lists the products with no decline fit on their shelf: product indexes, or [None] for a single
        product whose shelf labels hold no fit."""
  run = self.run
  covered = {run.judge.shelf_known(i) for i in self.decline_fits()} - {""}
  if run.needs.items():
   return [k for k, home in enumerate(run.shelf.home_shelves()) if home and not home & covered]
  if set(run.watch.shelf_labels()) & covered:
   return []
  return [None]

 def decline_walks(self) -> list[str]:
  """Picks up to two listing keys to read further before a decline, one per product lacking a fit.
        Prefers walk options carrying the most literal requirement values, then walk_score and pages."""
  needs_now = self.run.needs.props_left()

  def worded(call_args: dict[str, Any], k: int | None) -> int:
   """Counts the requirements a search carries whose value is literal, not a reference or redirect
            phrase. Used to rank decline walks."""
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
  """Builds one pool of unread archived ids per (home shelf, units) pair, cheapest first.
        Keeps in-stock, unrefused, in-budget rows; optionally only ids in ids and priced below below[n]."""
  run = self.run
  archive, watch = run.archive, run.watch
  budget = run.needs.spend_cap
  labels: dict[str, str] = {}
  sold: dict[str, bool] = {}

  def note(record: dict[str, Any]) -> None:
   """Visitor for row_pools: maps each archived id to its category label and, when the record has a
            stock field, to its sold-out state."""
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
  """For multi-product requests with a budget and no cheapest rule, finds cheaper unread rows when
        the cheapest fits of all products together go over the budget.
        Returns per-product pools below each fit's price, largest share first; [] otherwise or on error."""
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
  """Error-safe wrapper around queue_decline_reads; logs and returns an empty string on failure.
        Called by the run's pages_gate before a decline."""
  run = self.run
  try:
   return self.queue_decline_reads()
  except Exception as err:
   run.log.append({"kind": "decline_reads_error", "error": f"{type(err).__name__}: {err}"[:300]})
   return ""

 def queue_decline_reads(self) -> str:
  """Once per run, with an empty cart and time left, plans extra reads before a decline.
        Stores decline_walks keys, a page room of at most 3 and a step floor of 6 in decline_queue;
        returns a remark on those reads and on cheaper over-budget rows, or '' when neither applies."""
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
   parts.append("searches with required values found more matches than were read (" + named + "), and this "
       "store lists results in a fixed id order, not by fit or price, so fits are as likely to be on "
       f"unread pages as on the read ones: the code will now read as many as {room} of those pages for you and open the "
       "cheapest in-stock rows within budget that they show")
  if cheaper:
   parts.append("combined, the listings found for all products exceed the budget: so the cheapest unread "
       "in-stock rows costing less than each product's pick are opened for you as well")
  target = "one line per product, at its quantity" if run.needs.items() else "the item"
  return ("Before a decline, " + "; ".join(parts) + ". Judge those records by meaning against the requirements now in "
    "force; buy what fits (" + target + "), and decline only if each one fails.")

 def run_decline_reads(self) -> bool:
  """Carries out the queued pre-decline reads; called from Coverage.walk_step.
        Reads next pages of each queued listing, then queues the cheapest unread in-budget rows on
        lacking shelves plus over-budget pools for judge.open_named, or notes that none were seen.
        Returns True if a page was read or rows were queued."""
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
    said = (f"Before declining I go to the next page of '{query}': results come in a fixed id order here, "
      "so unread pages may hold fits.")
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
   reads.open_say = "Before a decline I open the cheapest unread in-budget rows shown by those pages and picks."
   reads.open_purpose, reads.open_floor = "decline", floor
   run.judge.open_named()
  elif run.chat and run.chat[-1].get("role") == "tool":
   run.chat[-1]["content"] = (str(run.chat[-1]["content"]) + "\nNo unread in-stock row in budget was on "
            "these pages.\n" + run.status_line())
  return read > 0 or bool(queue)


class ShopperReplyStep:
 """Scripted turn step that answers a question the shopper asked back.
    One of the steps ShoppingRun.execute tries in order before the actor model; the first step
    that returns True takes the turn."""
 def perform(self, run) -> bool:
  """Sends the pending needs.reply_due text through talk.reply_back after clearing it.
        Returns False when no reply is due."""
  pending = run.needs.reply_due
  if not pending:
   return False
  run.needs.reply_due = ""
  return run.talk.reply_back(pending)


class QueuedReadsStep:
 """Scripted turn step that opens listings queued for reading.
    Tried second in ShoppingRun.execute; the queue is filled elsewhere, for example by run_walk
    and run_decline_reads."""
 def perform(self, run) -> bool:
  """Delegates to judge.open_named; True when it opened queued listings this turn.
        False when nothing is queued or the episode is over."""
  return run.judge.open_named()


class ShelfSweepStep:
 """Scripted turn step for the cheapest shelf sweep (Observer.sweep_step).
    Stands aside while run.asking is on and no question plan has come back empty yet, so planned
    shopper questions go first."""
 def perform(self, run) -> bool:
  """Runs run.watch.sweep_step unless asking is on with no empty question plan so far.
        Returns True if a sweep search ran."""
  if run.asking and not run.talk.empty_plans:
   return False
  return run.watch.sweep_step()


class ValuePagesStep:
 """Scripted turn step that reads search pages carrying required values.
    Delegates to Coverage.value_pages, which works before any order or refusal and needs a search
    tool and time to spare."""
 def perform(self, run) -> bool:
  """Returns run.cover.value_pages(); True when that call used the turn.
        Tried fourth among the scripted steps."""
  return run.cover.value_pages()


class PlannedQuestionStep:
 """Scripted turn step that sends a planned question to the shopper.
    Active only while run.asking is set; talk.pose_planned turns asking off once the message limit
    is reached or no useful question can be planned."""
 def perform(self, run) -> bool:
  """Calls talk.pose_planned when run.asking is set; True when it used the turn.
        Returns False straight away while asking is off."""
  return bool(run.asking) and run.talk.pose_planned()


class ReferenceHopStep:
 """Scripted turn step that follows references to other listings.
    Delegates to ReferenceDesk.ref_step, which works through the reference chains in run.refs.
    Tried sixth among the scripted steps."""
 def perform(self, run) -> bool:
  """Returns run.refdesk.ref_step(); False when there are no references or the episode is over.
        A resolver failure inside it marks open references as stuck."""
  return run.refdesk.ref_step()


class CheapWalkStep:
 """Scripted turn step for the cheapest walk and the pre-decline reads.
    Delegates to Coverage.walk_step, which runs queued decline reads, queued or repair walks,
    ceiling walks and reserve walks through CheapWalk. Last scripted step before the actor."""
 def perform(self, run) -> bool:
  """Returns run.cover.walk_step(); True when a walk or decline read used the turn.
        Errors inside walk_step are logged there and give False."""
  return run.cover.walk_step()


class ActorTurnStep:
 """The actor model's turn: asks the ACTOR seat for tool calls and runs them.
    Used by ShoppingRun.execute when no scripted step took the turn; False ends the loop.
    Keeps idle (turns without calls), failures (empty or refused model replies), seq_no (call ids),
    last_ok (time of the last good reply) and missed (consecutive misses)."""
 def __init__(self):
  """Starts the counters at zero and last_ok at the current monotonic time.
        The same instance serves every actor turn of one episode."""
  self.idle = 0
  self.failures = 0
  self.seq_no = 0
  self.last_ok = monotonic()
  self.missed = 0

 def perform(self, run) -> bool:
  """Queries the ACTOR models (rotated once run.stuck reaches STUCK_LIMIT) with the chat and tools.
        A cart, order or decline call may first yield to a planned question; a reply with no calls gets
        a nudge and counts toward IDLE_LIMIT. Otherwise dispatches the calls, posts the replies and
        bumps run.stuck when the step counter did not move. A missing reply goes to after_miss."""
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
   run.chat.append({"role": "user", "content": "Each reply must call one or more of the offered tools. Continue.\n"
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
  """Decides whether the loop goes on after the actor model gave no reply.
        Stops when no model is usable; when DOWN_AFTER seconds passed since the last good reply and
        either two misses in a row or the wire is down after two models; or after two empty or refused
        failures. Otherwise returns True so the loop retries."""
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
 """One shopper answer line split around a phrase that points at another listing or seller.
    inside is the pointing span, outside the text around it, parsed the parsed reference.
    Built by ReferenceDesk.ref_fix, which keeps the outside part as the shopper's line and places
    the pointing need separately."""
 def __init__(self, text, phrase, start, end, parsed):
  """Stores the line, the pointing phrase and its parse, and splits the text at start and end
        into inside (the span) and outside (the text around it)."""
  self.text = text
  self.phrase = phrase
  self.parsed = parsed
  self.outside = text[:start] + " " + text[end:]
  self.inside = text[start:end]

 def inside_only(self, prop):
  """True when property prop is named inside the pointing span but nowhere outside it.
        An empty prop gives False; matching uses prop_named."""
  if not prop:
   return False
  return prop_named(prop, self.inside) and not prop_named(prop, self.outside)

 def describes_other(self, req, name):
  """Tells whether requirement req is about the pointed-at listing rather than the shopper's own.
        True if its property name (via name) shows only inside the span, or all its value tokens appear
        inside and none outside."""
  if self.inside_only(name(req)):
   return True
  words = set(ref_toks(str(req).partition(":")[2]))
  if not words:
   return False
  return words <= set(ref_toks(self.inside)) and not words & set(ref_toks(self.outside))


class ReferenceDesk:
 """Chases requirements whose value is borrowed from another listing named by seller and kind.
    Each chase is a dict in run.refs (target seller/kind/details, relation, chain of listing ids,
    status open, resolved, stuck or dropped). It searches for the target, opens candidates, reads
    notes, asks the CHECKER hop prompt for the value or a go_to, then rewrites the requirement.
    Also parses shopper pointing lines (Needs.ref_hook) and feeds reference text to checker prompts.
    Created by ShoppingRun as run.refdesk; ReferenceHopStep drives it through ref_step."""
 def __init__(self, run):
  """Keeps the run and starts empty chase state: hop LLM call count, steps spent, candidate ids,
        redirect remarks already told, queued actor notices and memos for the listing index and fields."""
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
  self.index_row_at = None
  self.v73_rows_logged = set()

 @property
 def needs(self):
  """Shortcut to the run's Needs object (requirements, reading of the request, versions).
        Read-only property; every reference rewrite goes through it."""
  return self.run.needs

 @property
 def archive(self):
  """Shortcut to the run's Records archive of tool results.
        Used for the merged listing index, the set of ids whose details were read, and record lookups."""
  return self.run.archive

 def open_refs(self):
  """Lists the entries of run.refs whose status is still open.
        Used by the step-budget checks to see which chases are pending."""
  return [entry for entry in self.run.refs or [] if entry.get("status") == "open"]

 @staticmethod
 def head_prop(head):
  """Strips the product tag from a requirement head: returns the text before the first bracket.
        So 'colour (boots)' gives 'colour'."""
  return re.split(r"\s*\(", str(head))[0]

 def product_kinds(self):
  """Collects the kind words of every requested product, or of the single product in the reading.
        ref_seller_ok uses them so a product word is not mistaken for a seller name."""
  kinds = set()
  for entry in self.needs.items() or [{"product": str(self.needs.reading.get("product") or "")}]:
   kinds |= kind_words(entry["product"])
  return kinds

 def ref_register(self):
  """Opens a chase for each written request requirement whose value points at another listing.
        A value counts when it has reference wording, or is listed under the reader's references and
        uses differ wording. Uses the reader's reference_targets; called by Intake after the request read."""
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
  """Filters reference requirement strings down to those no tracked chase has resolved or chained.
        Judge's item check calls it to decide which pointed records to show; on error returns refs as is."""
  def followed(req):
   """True when some tracked chase matches the requirement and is resolved or already has a chain.
            Closure over the tracked list passed to ref_untracked."""
   for entry in tracked:
    if self.ref_match(entry, req) and (entry.get("status") == "resolved" or entry.get("chain")):
     return True
   return False
  try:
   return [req for req in refs if not followed(req)]
  except Exception:
   return refs

 def ref_take(self):
  """Runs after shopper lines are heard: re-swaps a resolved value where a current requirement still
        matches the chase in another wording, then drains needs.ref_news and opens a chase (origin
        'reply') for each new pointing need, listing it under the reading's references.
        Errors are logged as ref_take_error."""
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
  """Appends a requirement to needs.reading['references'] unless it is already there.
        Creates the list when the reading has none or holds a non-list value."""
  reading = self.needs.reading
  known = reading.get("references")
  if not isinstance(known, list):
   known = reading["references"] = []
  if req not in known:
   known.append(req)

 def info_for(self, infos, prop, tag):
  """Picks the reader's target description whose property matches the given property.
        With several products the entry's product must share kind words with the tag; returns None
        when nothing matches."""
  for cand in infos:
   if not same_prop(str(cand.get("property") or ""), prop):
    continue
   if not tag or not cand.get("product") or not self.needs.items() or kind_words(cand.get("product")) & kind_words(tag):
    return cand
  return None

 def already_open(self, req, prop, tag):
  """True when run.refs already tracks this requirement, or the same property for the same product.
        Prevents ref_add from opening a duplicate chase."""
  for known in self.run.refs:
   if known["req"] == req:
    return True
   if same_prop(known["prop"], prop) and kind_words(known["tag"]) == kind_words(tag):
    return True
  return False

 def relation_of(self, value, info, prop, source, target):
  """Decides how the borrowed value applies: same, within (tolerance, unit), at most, at least, differ, partner.
        Takes base_relation's reading; when that is same or differ, the pointing words nearest the pointed-to seller
        in the source (or the value, when the property is not found in the source) decide instead if they state a
        relation (pointer_relation, v7.5 a): partner is logged v75_partner, a same/differ swap v75_relation."""
  relation, tolerance, unit = self.base_relation(value, info, prop, source, target)
  if relation not in ("same", "differ"):
   return relation, tolerance, unit
  span = ref_span(prop, source).lower()
  seller_text = str(target.get("seller") or "")
  cut = span.find(seller_text.lower()) if seller_text.strip() else -1
  lead = span[:cut] if cut > 0 else span
  said = pointer_relation(lead if lead.strip() else value)
  if not said or said == relation:
   return relation, tolerance, unit
  self.run.log.append({"kind": "v75_partner" if said == "partner" else "v75_relation", "prop": str(prop)[:60],
        "was": relation, "now": said, "words": str(lead.strip() or value)[:120]})
  return said, None, ""

 def base_relation(self, value, info, prop, source, target):
  """The v7.4 reading of a borrowed value's relation: same, within (tolerance, unit), at most, at least, differ.
        Starts from the value's wording, takes a within tolerance from the reader's info, and turns 'same'
        into 'differ' when the info, or source words between the property and the seller name, say so."""
  relation, tolerance, unit = ref_relation(value)
  stated = isinstance(info, dict)
  if relation == "same" and stated and str(info.get("relation") or "").lower() == "within" and is_number(info.get("tolerance")):
   relation, tolerance, unit = "within", float(info["tolerance"]), str(info.get("unit") or "")
  told = str(info.get("relation") or "").lower() if stated else ""
  span = ref_span(prop, source).lower()
  seller_text = str(target.get("seller") or "")
  cut = span.find(seller_text.lower()) if seller_text.strip() else -1
  if relation != "same":
   return relation, tolerance, unit
  if re.search(r"differ|not the same|exclu|apart|other than|unlike", told) or (cut > 0 and re.search(DIFFER_WORDS, span[:cut])):
   return "differ", None, ""
  return relation, tolerance, unit

 def ref_add(self, req, infos, source, origin):
  """Opens a chase for one requirement string: splits head, property and product tag, finds the
        target, sets the relation and appends a fresh record (status open) to run.refs.
        Returns the record, or None for junk properties, duplicates or no usable target (logs ref_skip)."""
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
  """Builds the target {seller, kind, details} from the reader's info or by parsing the source text
        (request or shopper line) after the property name, or the value. The seller must pass
        ref_seller_ok; at most 6 details are kept, each sharing a word with the source text.
        Returns None when no acceptable seller is found."""
  seller, kind, details = "", "", []
  if isinstance(info, dict):
   seller = str(info.get("seller") or "").strip()
   kind = str(info.get("kind") or "").strip()
   details = self.clean_strings(info.get("details"))

  def seller_ok(name):
   """Checks a candidate seller name against the source text via ref_seller_ok.
            Passed to ref_parse as its acceptance test."""
   return self.ref_seller_ok(name, source)
  parsed = None
  for text in (ref_span(prop, source), value):
   got = ref_parse(text, seller_ok)
   if got and seller_ok(got["seller"]):
    parsed = got
    break
  if self.ref_seller_ok(seller, source):
   if not details and parsed and ref_name(parsed["seller"]) == ref_name(seller):
    details = parsed["details"]
  elif parsed:
   seller, kind, details = parsed["seller"], kind or parsed["kind"], details or parsed["details"]
  else:
   return None
  written = set(ref_toks(source))
  kept = [d for d in dict.fromkeys(details) if set(ref_toks(d)) & written][:6]
  shown_kind = (kind or parsed["kind"]) if parsed else kind
  return {"seller": seller, "kind": shown_kind[:80], "details": kept}

 def ref_seller_ok(self, seller, source):
  """Accepts a seller name of 3+ letters that appears as a whole word in the source text and whose
        kind words are not all product words (requested products or search-tool enum options)."""
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
  """Identity key of a target: normalised seller name plus the sorted tokens of its details.
        Lets the chase notice when a go_to returns to a target it has visited before."""
  tokens = set()
  for detail in target.get("details") or []:
   tokens.update(ref_toks(detail))
  return ref_name(target.get("seller") or ""), tuple(sorted(tokens))

 @staticmethod
 def describe_target(target):
  """Phrases a target as 'the <kind> sold by <seller>' with its details in brackets.
        Used in actor notices, search narration and the hop prompt."""
  text = f"the {target.get('kind') or 'listing'} sold by {target.get('seller')}"
  if target.get("details"):
   text += " (" + "; ".join(target.get("details") or []) + ")"
  return text

 def describe(self, ref):
  """Describes the chase's current target with describe_target.
        Changes as the chase follows go_to hops."""
  return self.describe_target(ref["target"])

 def listing_index(self):
  """Merges every archived dict that carries a known listing id into one row per id.
        Memoised on the archive length; also records the newest record position per id in index_seen_at."""
  items = self.archive.items
  if self.index_memo and self.index_memo[0] == len(items):
   return self.index_memo[1]
  merged, newest, rank = {}, {}, {}
  mark_of = self.run.watch.row_mark
  known = self.archive.ids
  for position, record in enumerate(items):
   count = [0]

   def take(fields, position=position, count=count):
    """walk_dicts visitor: merges a dict into its id's row when the id is archived and the dict has
                4+ fields or holds a property map of 2+ entries; notes its record position and its order there."""
    ident = mark_of(fields)
    if ident not in known:
     return
    if len(fields) < 4 and not any(isinstance(v, dict) and len(v) >= 2 for v in fields.values()):
     return
    merged.setdefault(ident, {}).update(fields)
    newest[ident] = position
    rank[ident] = count[0]
    count[0] += 1
   walk_dicts(record, take)
  self.index_memo = (len(items), merged)
  self.index_seen_at = newest
  self.index_row_at = rank
  return merged

 def sellers_of(self, row):
  """Returns the normalised seller names a listing row shows.
        Reads short string fields whose key looks like a shop/brand key or was learned by seller_fields,
        skipping id keys and note-like keys."""
  learned = self.seller_fields()
  names = set()
  for key, val in row.items():
   if not isinstance(val, str) or not val.strip() or len(val) > 60:
    continue
   low = str(key).lower()
   if not (re.search(SELLER_KEYS, low) or str(key) in learned) or "id" in low:
    continue
   if re.search(r"note|text|desc|comment|detail|info|message|section", low):
    continue
   names.add(ref_name(val))
  return names - {""}

 def capitalised_names(self, rows):
  """Collects every 1-3 word span of consecutive capitalised words in listings' notes, skipping each
        sentence's first word.
        Returns them normalised; seller_fields uses them as possible seller names."""
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
  """Returns the normalised seller names of every chase's current and first target.
        Feeds seller_fields and its memo stamp."""
  names = set()
  for entry in self.run.refs or []:
   for side in (entry.get("target") or {}, entry.get("first") or {}):
    if isinstance(side, dict):
     names.add(ref_name(side.get("seller")))
  return names - {""}

 def seller_fields(self):
  """Learns which row keys hold seller names in this store: short non-title, non-note string fields
        whose value is a chased seller, or a short capitalised value also seen capitalised in notes.
        Memoised on archive length and chased sellers; returns an empty set on error."""
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
     if not isinstance(val, str) or not 0 < len(val) <= 60 or "id" in low or re.search(skip, low):
      continue
     if ref_name(val) in names or (val.strip()[:1].isupper() and len(val.split()) <= 3 and ref_name(val) in noted):
      fields.add(str(key))
   self.seller_field_memo = (stamp, fields)
   return fields
  except Exception:
   return set()

 @staticmethod
 def note_text(row, desc=True):
  """Joins a row's free-text fields of 3+ words, skipping ids, titles, names, labels, categories, urls.
        Note fields come first; description or overview fields are used only when there are no notes
        and desc is True."""
  notes, overview = [], []
  for key, val in row.items():
   low = str(key).lower()
   if not isinstance(val, str) or len(val.split()) < 3 or low == "id" or low.endswith("id"):
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
  """Splits a row into (field-name tokens, value tokens) for detail matching.
        Names come from property-map keys (or flat scalar fields when there is no map); values come from
        the title, the notes with negated phrases removed, and the field values."""
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
  """Joins a row's title and its property-map values into one text.
        may_show reads the listing's own figures and units from it."""
  pieces = [self.run.shelf.title_of(row)]
  for val in row.values():
   if isinstance(val, dict):
    pieces += [str(x) for x in val.values() if isinstance(x, (str, int, float))]
  return " | ".join(pieces)

 def may_show(self, row, detail, parts=None):
  """Tells whether a row could show one target detail.
        At least one of its non-numeric words that is not a field name must be among the row's values,
        and every figure must match one the row writes in the same unit, if it writes that unit at all."""
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

 def v73_unshown(self, ref, ident):
  """The first target detail that listing ident does not show in its own data (v7.3 H), or "".
        Checks the details of up to five tokens that are not set loose the way may_show does, but only against the
        title, property values of fields that are no catch-all (catch_all_key) and note sentences naming no seller
        except the listing's own; figures against the title and those property values."""
  rows = self.listing_index()
  row = rows.get(ident) or {}
  names, _ = self.token_parts(row)
  own = self.sellers_of(row)
  others = set()
  for other in rows.values():
   others |= self.sellers_of(other)
  others = {seller for seller in others - own if len(seller) >= 4}
  figures_from = [self.run.shelf.title_of(row)]
  for found in (val for val in row.values() if isinstance(val, dict)):
   figures_from += [str(x) for key, x in found.items()
        if isinstance(x, (str, int, float)) and not isinstance(x, bool) and not catch_all_key(key)]
  words_from = list(figures_from)
  if not any(isinstance(val, dict) for val in row.values()):
   words_from += [str(val) for key, val in row.items() if isinstance(val, (str, int, float)) and not isinstance(val, bool)
         and not re.search(r"id$|title|shop|seller|brand|price|amount|desc|overview|note", str(key).lower())
         and not catch_all_key(key)]
  notes = re.sub(r"\b(?:not|no|never|without|nor)\b(?:\W+[A-Za-z0-9.:/&'-]+){1,3}", " ", self.note_text(row, desc=False), flags=re.I)
  words_from += [part for part in re.split(r"(?<=[.!?;])\s+|\n", notes) if not any(seller in ref_name(part) for seller in others)]
  words = set(ref_toks(" | ".join(words_from)))
  written = ref_figs(" | ".join(figures_from))
  for detail in ref["target"]["details"]:
   if detail in (ref.get("loose") or set()) or len(ref_toks(detail)) > 5:
    continue
   wanted = {t for t in ref_toks(detail) if len(t) >= 3 and not re.search(r"\d", t)} - names
   if wanted and not wanted & words:
    return detail
   for figure, unit in ref_figs(detail):
    same_unit = [f for f, u in written if u == unit]
    if same_unit and not any(abs(f - figure) < 1e-9 for f in same_unit):
     return detail
  return ""

 @staticmethod
 def field_value(row, prop):
  """Returns the first property-map value whose key names the property, else an empty string.
        Tries strict_prop matching before the looser same_prop."""
  for matcher in (strict_prop, same_prop):
   for val in row.values():
    if not isinstance(val, dict):
     continue
    for key, x in val.items():
     if isinstance(x, (str, int, float)) and not isinstance(x, bool) and str(x).strip() and matcher(key, prop):
      return str(x).strip()
  return ""

 def remember_ids(self, ids):
  """Adds listing ids to candidate_ids, the set opened or judged while chasing references.
        ref_foreign later screens this set."""
  self.candidate_ids = set(self.candidate_ids or set()) | set(ids)

 def ref_foreign(self):
  """Returns chase candidate ids, not in the cart, whose category label lies outside every product's
        home shelves. Empty when home shelves are not all known. Callers leave these listings out of
        record cards and category label tallies."""
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
  """Ranks indexed listings sold by the target's seller as possible targets.
        Order: category matching the target kind and details shown in title or description, then the
        most recently seen, then the row order of that record (v7.3: the store's order, top first, before the id).
        Stores the per-id count of details shown in ref['shows']; logs v73_h_rows when the row order changes the
        first four from the id order."""
  target = ref["target"]
  wanted = ref_name(target["seller"])
  kinds = kind_words(target["kind"])
  rows = self.listing_index()
  seen_at = self.index_seen_at or {}
  row_at = self.index_row_at or {}
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
   scored.append((-on_kind - shown, -seen_at.get(ident, 0), row_at.get(ident, 0), ident, shown))
  scored.sort()
  ref["shows"] = {ident: shown for _, _, _, ident, shown in scored}
  order = [entry[3] for entry in scored]
  by_id = [entry[3] for entry in sorted(scored, key=lambda entry: (entry[0], entry[1], entry[3]))]
  mark = (ref["prop"], tuple(order[:4]))
  if order[:4] != by_id[:4] and mark not in self.v73_rows_logged:
   self.v73_rows_logged.add(mark)
   self.run.log.append({"kind": "v73_h_rows", "prop": ref["prop"], "rows": order[:4], "by_id": by_id[:4]})
  return order

 @staticmethod
 def ignores_words(seen):
  """True when two searches with no word in common returned the same first-page ids (2+ rows).
        Marks a search setup whose typed text has no effect."""
  for i, (words_a, rows_a) in enumerate(seen):
   for words_b, rows_b in seen[i + 1:]:
    if len(rows_a) >= 2 and rows_a == rows_b and not (words_a & words_b):
     return True
  return False

 def search_modes(self, needed, labels):
  """Groups past searches in page_book by their non-text arguments, ignoring optional category values.
        Returns templates (fixed args, text key) ranked by query variety then first use, plus the group
        keys whose searches ignore the typed words."""
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
  """Falls back to the run log: the newest search action with exactly one free-text argument.
        Returns (other args without the paging key, text key), skipping text-blind setups; None if none."""
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
  """Chooses the argument template for a target search: (fixed args, free-text key) or None.
        Prefers search_modes, then logged_template, and drops optional args holding a category label.
        Without history it uses the schema's single plain string property if nothing else is required."""
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
  """Collects lower-case words of indexed property-map field names and of requirement properties.
        detail_words drops these so a query uses value words, not property names."""
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
  """Turns the target's details into search words: letter-digit codes, then 2+ digit numbers, then
        plain words of 3+ letters that are not field or property names (longest first, value words
        promoted). Link words, seller words and unit words after a number are left out."""
  known = self.known_words()
  details = ref["target"]["details"]
  seller_words = set(re.findall(r"[a-z0-9]+", str(ref["target"]["seller"]).lower()))
  mixed, numbers, plain = [], [], []
  for detail in details:
   previous = ""
   for whole in str(detail).lower().split():
    after_figure = bool(re.fullmatch(r"\d+(?:[.,]\d+)?", previous))
    previous = whole
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
  """Finds value words of details shaped like '<field name> <value>' so detail_words can lead with them.
        Before any hop, a detail list made only of multi-word details is all treated that way."""
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
  """Returns the set of spaced, lower-case field names of property maps across all archived records.
        Memoised on the archive length; used by query building and pointing-line placement."""
  stamp = len(self.archive.items)
  if self.field_names_memo and self.field_names_memo[0] == stamp:
   return self.field_names_memo[1]
  names = set()

  def collect(fields):
   """walk_dicts visitor: adds the keys of every nested map holding 2+ scalar values.
            camelCase keys are split into words before they are stored."""
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
  """Plans search texts for a target: '<seller> <first detail word>' when the seller name is shorter
        than the lead width, then '<seller>'. If the search does not index sellers (unindexed), the first
        detail word plus the kind word instead. Queries this chase already ran are left out."""
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
  """Builds the next-page arguments of the newest past search whose text holds the seller name.
        Returns None without a paging argument, after 3 extra pages for this chase, or with no next page."""
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

 def ref_room(self):
  """Decides whether the chase may spend another world step.
        Caps steps at a REF_STEPS share of the step cap (REF_DIFFER_STEPS, or REF_DIFFER_BLIND before any
        chain, for differ-only chases), stops near the time close or spend cap, pauses while other
        products wait, then needs moves above step_floor or a late hop window."""
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
  """Returns the moves the chase must leave untouched for the rest of the errand.
        The larger of the first-order reserve plus 10 and the intake's order_due plus 7."""
  reserve = max(FIRST_ORDER_BY, min(FIRST_ORDER_DUE, self.run.step_cap // 4 + 1)) + 10
  return max(reserve, self.run.intake.order_due() + 7)

 def product_for(self, ref):
  """Maps a chase's product tag to its product index via talk.product_of.
        None for a single-product request or a chase without a tag."""
  if not self.needs.items() or not ref.get("tag"):
   return None
  return self.run.talk.product_of(f"x ({ref['tag']})")

 def waits_for_others(self, floor=None):
  """Pauses the chase in multi-product requests when every open chase is a differ with a chain and
        products owning no chase still lack any fit, while more than floor+4 moves remain.
        Logs ref_wait when the list of waiting products changes."""
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
  """Grants a 3-step window past the floor for an open chase that already hopped once, names seller,
        kind and details, has run no query for its current target, and whose other products are verified.
        Never when moves left are within order_due plus 2; stores late_from and logs ref_late."""
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
  """True when every product other than the chase's own has a verified fit in its pool.
        Single product: True. Chase with no known product: at most one product may lack one."""
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
  """Checks whether another hop LLM call is allowed: fewer than REF_CALLS so far, time left above
        CLOSING plus 40, and spend below 0.8 of the scaled SPEND_CAP."""
  wire = self.run.wire
  if self.hop_calls >= REF_CALLS or wire.spare() <= CLOSING + 40 or wire.spent_out():
   return False
  return wire.spend < 0.8 * SPEND_CAP * wire.scale

 def ref_match(self, ref, req):
  """Tells whether a requirement string belongs to a chase.
        Matches its original or resolved text, or a pointing (or, for differ chases, differ-worded) value
        on the same property whose product tag overlaps the chase's tag."""
  if str(req) in (ref["req"], ref.get("now")):
   return True
  head, _, value = str(req).partition(":")
  if ref_text(value):
   pointing = True
  else:
   pointing = ref.get("relation") == "differ" and re.search(DIFFER_WORDS, value.lower())
  if not pointing or not same_prop(self.head_prop(head), ref["prop"]):
   return False
  bracketed = re.findall(r"\(([^)]*)\)", head)
  if not bracketed or not ref["tag"]:
   return True
  return bool(kind_words(bracketed[-1]) & kind_words(ref["tag"]))

 def ref_live(self, ref):
  """True while the chase still matters: a current requirement matches it or has its property,
        or the shopper has not let go of a requirement on that property."""
  def same(req):
   """True when a requirement's property equals the chase's property, ignoring the product tag.
            Used for both the current and the let-go requirement lists."""
   return same_prop(self.head_prop(str(req).partition(":")[0]), ref["prop"])
  if any(self.ref_match(ref, req) or same(req) for req in self.needs.props_left()):
   return True
  return not any(same(req) for req in self.needs.let_go)

 def ref_step(self):
  """Runs one resolver turn for ReferenceHopStep; False when there are no chases or the run is over.
        On an exception it logs ref_error and marks every open chase stuck."""
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
  """Marks open chases that are no longer live as dropped (logged ref_dropped).
        Returns the chases still open."""
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
  """Takes ids round-robin from per-chase pools, skipping repeats, until `most` are queued.
        Pops from the pools it is given."""
  queue = []
  while len(queue) < most and any(pools):
   for pool in pools:
    while pool and pool[0] in queue:
     pool.pop(0)
    if pool and len(queue) < most:
     queue.append(pool.pop(0))
  return queue

 def read_pools(self, pending, most):
  """Builds per-chase pools of unread, unjudged candidate ids.
        A pool is held empty until the chase has run a query, has no query left, fits in one batch, or its
        top candidate shows a detail."""
  read = self.archive.read
  pools = []
  for ref in pending:
   pool = [i for i in self.candidates(ref) if i not in read and i not in ref["judged"]]
   shows = ref.get("shows") or {}
   ready = bool(ref["queries"]) or not self.planned_queries(ref) or len(pool) <= most or any(shows.get(i) for i in pool[:1])
   pools.append(pool if ready else [])
  return pools

 def search_next(self, needy, pending, template, acted):
  """Runs the next target search: the first planned query of a needy (or least advanced) chase, else
        its seller's next page. A query the walker already ran is marked used and no search is sent.
        Chases with nothing left to read or search are given up. Returns whether anything acted."""
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
   self.give_up(ref, "the store's search never turned up the listing it refers to")
   acted = True
  return acted

 def resolver_turn(self):
  """One resolver turn: judge read candidates of each live chase; if that used no world step, open
        queued candidates, or search via the template when nothing is queued or a short queue leaves
        some chase with only searches left. Returns whether it acted."""
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
  """Opens queued candidate listings through Judge.open_reads (batch tool for 2+) and posts the replies.
        An unproven batch tool is proven or dropped by its result. Counts steps and remembers the ids;
        ids still unread are marked judged unless the batch was dropped.
        Returns whether a world step was spent."""
  run = self.run
  batch = run.shelf.batch_tool() if len(queue) >= 2 else None
  start = run.wire.tick
  heads = ", ".join(dict.fromkeys(r["head"] for r in pending))
  replies = run.judge.open_reads(queue, f"The request borrows {heads} from another listing, so I open the ones that could be it.", batch)
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
  """Runs one target search through the cheap walker's walk_call with a narration, counting steps.
        If rows came back with none from the seller and the query was just the seller name, the chase is
        marked unindexed. Logs ref_search; returns whether a step was spent."""
  run = self.run
  start = run.wire.tick
  got = run.walk.walk_call(args, f"The request borrows {ref['head']} from another listing ({self.describe(ref)}): I search for it.")
  self.steps_spent += run.wire.tick - start
  rows = []
  if got:
   mark_of, known = run.watch.row_mark, self.archive.ids

   def keep(fields):
    """walk_dicts visitor over the newest archive record: keeps result rows with a known id and 4+ fields.
                The kept rows decide whether the search returned anything from the seller."""
    if mark_of(fields) in known and len(fields) >= 4:
     rows.append(fields)
   walk_dicts(self.archive.items[-1], keep, ceiling=3)
  wanted = ref_name(ref["target"]["seller"])
  text = " ".join(run.watch.free_text(args))
  if rows and not any(wanted in self.sellers_of(r) for r in rows) and ref_name(text) == wanted:
   ref["unindexed"] = True
  run.log.append({"kind": "ref_search", "prop": ref["prop"], "query": text[:120], "rows": len(rows)})
  return run.wire.tick > start

 def judge_read(self, ref):
  """Judges a chase's opened candidates: ranks them by details shown, may wait for more reads or notes,
        resolves a single exact plain record by code, else asks the hop LLM and applies the answer.
        Once 4+ seller listings are read, a detail none of them shows is set loose.
        Returns whether it acted."""
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
   """Runs may_show for one candidate and detail, using its precomputed token parts.
            Closure inside judge_read."""
   return self.may_show(rows.get(ident) or {}, detail, parts[ident])
  if len(by_seller) >= 4:
   loose |= {d for d in details if not any(shows(i, d) for i in by_seller)}

  def words_ok(ident):
   """True when a candidate may show every target detail not marked loose.
            Primary key for ranking candidates in judge_read."""
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
   unshown = self.v73_unshown(ref, exact[0])
   if unshown:
    ref["judged"].add(exact[0])
    self.remember_ids(pick)
    self.run.log.append({"kind": "v73_h_reject", "prop": ref["prop"], "target": exact[0], "detail": unshown[:80], "by": "code"})
    return True
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
  """True when the chase can still find more: step room (or a wait) and unread candidates, planned
        queries or a next seller page."""
  if not (self.ref_room() or self.waits_for_others()):
   return False
  read = self.archive.read
  if [i for i in self.candidates(ref) if i not in read and i not in ref["judged"]]:
   return True
  return bool(self.planned_queries(ref) or self.next_seller_page(ref))

 def settle_by_notes(self, ref, pick, missing):
  """Handles picked listings whose notes are unread: reads them when there is room (True), pauses on
        a wait (False), gives up a differ chase when out of room (True). None lets judging continue."""
  if not missing:
   return None
  if self.ref_room() and self.read_notes(ref, missing):
   return True
  if self.waits_for_others():
   return False
  if not self.ref_room() and ref.get("relation") == "differ":
   ref["judged"].update(pick)
   self.give_up(ref, "the steps left were too few to read the notes of the listing it refers to")
   return True
  return None

 def field_tokens(self, row, names):
  """Returns the field-name tokens plus tokens of the title, category label and property-map values.
        Used to tell capitalised words of the listing's own fields from names of other sellers."""
  pieces = [self.run.shelf.title_of(row), self.run.watch.shelf_of(row)]
  for val in row.values():
   if isinstance(val, dict):
    pieces += [str(x) for x in val.values() if isinstance(x, (str, int, float))]
  return names | set(ref_toks(" | ".join(pieces)))

 def plain_record(self, ref, ident):
  """Tells whether a candidate can be resolved by code: it has a field value for the property, and its
        notes neither name the property, repeat the value, mention another seller seen, nor carry an
        unknown capitalised word."""
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
  """Picks the tool that shows listing notes: the learned notes tool; none when details already
        shows notes; else another lookup tool returning extra text keys (most used), else the view or
        stock tool as a last choice."""
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
  """Returns up to 2 of the ids whose notes neither the notes tool has read nor this chase has tried.
        Empty when there is no notes tool."""
  tool = self.notes_reader()
  if not tool or not ids:
   return []
  seen = set((self.run.shelf.read_with or {}).get(tool) or set())
  tried = ref.setdefault("noted", set())
  return [i for i in dict.fromkeys(ids) if i not in seen and i not in tried][:2]

 def read_notes(self, ref, need):
  """Opens the listed ids with the notes tool through Judge.open_reads and posts the replies.
        Marks them noted, counts steps, remembers ids, logs ref_notes; returns whether a step was spent."""
  tool = self.notes_reader()
  if not tool or not need:
   return False
  run = self.run
  ref.setdefault("noted", set()).update(need)
  start = run.wire.tick
  replies = run.judge.open_reads(need, f"The request borrows {ref['head']} from another listing: as a note may redirect the reader, I read "
            f"the notes of the listing that might be it ({tool}).", None, tool)
  if replies:
   run.talk.post_replies(replies)
  self.steps_spent += run.wire.tick - start
  self.remember_ids(need)
  run.log.append({"kind": "ref_notes", "ids": need, "tool": tool})
  return run.wire.tick > start

 def ref_pointers(self):
  """Collects the chain ids of live differ chases (open, stuck or resolved) for the same product.
        Ranking, closing and cheap-walk code avoid these listings, as they hold the value to differ from."""
  pointed = set()
  for ref in self.run.refs or []:
   if ref.get("relation") != "differ" or ref.get("status") not in ("open", "stuck", "resolved") or not self.ref_live(ref):
    continue
   pointed.update(i for i in ref.get("chain") or [] if self.same_product(ref, i))
  return pointed

 def same_product(self, ref, ident):
  """Tells whether a listing's category maps to the chase's product.
        True when the request has one product, or the listing's product or the chase's tag is unknown."""
  if not self.needs.items():
   return True
  shelf = self.run.judge.shelf_known(ident)
  product_of = self.run.talk.product_of
  k = product_of(f"x ({shelf})") if shelf else None
  if k is None or not ref.get("tag"):
   return True
  return k == product_of(f"x ({ref['tag']})")

 def ref_pointer_note(self, thing):
  """Explains, for a listing in a live differ chase's chain, why it cannot meet that requirement.
        Returns an empty string otherwise. Used by the item check and the cart screen."""
  for ref in self.run.refs or []:
   if ref.get("relation") != "differ" or thing not in (ref.get("chain") or []):
    continue
   if not self.ref_live(ref) or not self.same_product(ref, thing):
    continue
   what = f" ({ref['value']})" if ref.get("status") == "resolved" and ref.get("value") else ""
   return (f"{thing} is the target of the shopper's reference for {ref['head']}, or a listing its notes route "
     f"through, so its {ref['prop']} is the exact value{what} the shopper's {ref['prop']} has to differ from: it "
     "can never meet that requirement")
  return ""

 def ref_excluded(self, fields, k):
  """True when a listing's fields carry the value (or a paired value) a resolved live differ chase
        excludes, for product k. Used by ranking and closing code to skip such listings."""
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
  """Tells whether a note sentence names the property and later mentions another seller.
        A seller is a known foreign name or an unknown capitalised word of 4+ letters; a negation in
        the five words before it cancels the hit."""
  words = list(re.finditer(r"[A-Za-z][A-Za-z'-]*", sentence))
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
  """Scans a listing's notes for sentences that send a current requirement's property elsewhere.
        Returns (requirement, property, sentence) per hit, only for the listing's product.
        Used by Judge (item check, field checks) and notes_remark; returns [] on error."""
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

 def notes_remark(self, call_name, call_args):
  """After a lookup call, warns the actor once per id and version when a listing in the cart or
        passed by the item check has notes that redirect a property, clearing its pass so it is checked
        again. Returns the remark text, or '' for search, cart, message and finish calls."""
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
   self.redirect_told.add(key)
   run.passed.discard(key)
   run.log.append({"kind": "own_redirect", "id": ident, "props": [p for _, p, _ in found]})
   what = "; ".join(f"{p} (\"{s[:160]}\")" for _, p, s in found)
   quoted = "; ".join(f"'{r}'" for r, _, _ in found)
   tail = ". If they do, remove it from the cart and add a different verified fit." if ident in run.trolley.units else "."
   remarks.append(f"{ident}'s notes may refer {what} to another listing: if so, its own field is not what counts for "
         + quoted + " (the grader takes that other listing's value), so it satisfies "
         + ("them" if len(found) > 1 else "it") + " only if the record of that listing shows it; the item check is repeated "
         "before the order" + tail)
  return " ".join(remarks)

 def pointing_phrase(self, line):
  """Cuts the pointing phrase out of a shopper line: from the first differ, reference or pointing word
        in the clause before the seller name to the end of that sentence or its '(or ...)' aside.
        Returns (phrase, start, end, parsed target) or None."""
  text = str(line or "")
  if not ref_text(text):
   return None
  parsed = ref_parse(text, lambda name: self.ref_seller_ok(name, text))
  if not parsed or not self.ref_seller_ok(parsed["seller"], text):
   return None
  hit = re.search(r"(?<![A-Za-z'])" + re.escape(parsed["seller"]) + r"(?:'s)?(?![A-Za-z])", text)
  if not hit:
   return None
  before = text[:hit.start()]
  stops = [before.rfind(c) for c in (",", ";", "!", "?")] + [m.start() for m in re.finditer(r"[.](?:\s|$)", before)]
  cut = max(stops + [-1])
  clause = before[cut + 1:].lower()
  marks = [m.start() for pattern in (DIFFER_WORDS, REF_WORDS, POINT_WORDS) for m in re.finditer(pattern, clause)]
  if not marks:
   return None
  start = cut + 1 + min(marks)
  after = text[hit.end():]
  aside = re.search(r"\(\s*or\b[^)]*\)", after)
  stop = re.search(r"[.!?;](?:\s|$)", after)
  if aside and (not stop or aside.start() < stop.start()):
   end = hit.end() + aside.end()
  elif stop:
   end = hit.end() + stop.start()
  else:
   end = hit.end() + len(after)
  phrase = text[start:end].strip(" ,;")
  return (phrase, start, end, parsed) if phrase else None

 def question_words(self, qtext, outside):
  """Lists 1-3 word runs of the assistant's question that the line, outside the phrase, names.
        Link words, short words and product kind words are skipped."""
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
  """Needs.ref_hook: for each shopper line with a pointing phrase, places the pointing need into the
        listener's answer. Returns the lines with those phrases cut out."""
  masked = []
  for line in lines:
   got = self.pointing_phrase(line)
   if not got:
    masked.append(line)
    continue
   phrase, start, end, parsed = got
   piece = PointingLine(line, phrase, start, end, parsed)
   masked.append(piece.outside)
   self.place_pointing_need(answer, piece, qtext)
  return masked

 def need_name(self, req):
  """Returns a requirement string's property name without its product tag.
        Shared name function for the pointing-line helpers."""
  return self.head_prop(str(req).partition(":")[0]).strip()

 def line_property(self, piece, needs, refs, qtext):
  """Works out which property a pointing line sets. Tries, in order: the answer's references, its
        pointing needs, all needs, the label being asked, known field names, words of the question;
        the first tier named outside the phrase wins. Else a reference, pointing need or single asked
        label that the phrase itself does not name."""
  name = self.need_name
  outside = piece.outside
  asked = [name(a) for a in str(self.run.talk.asking_label or "").split(";") if name(a)]
  pointing = [name(n) for n in needs if ref_text(n.partition(":")[2]) or re.search(DIFFER_WORDS, n.partition(":")[2].lower())]
  tiers = [[str(r.get("property") or "") for r in refs], pointing, [name(n) for n in needs], asked,
     sorted(self.field_names()) + [name(r) for r in self.needs.written], self.question_words(qtext, outside)]
  for tier in tiers:
   named = [p for p in tier if p and prop_named(p, outside)]
   if named:
    return max(named, key=lambda p: (prop_named(p, outside, exact=True), len(re.findall(r"[a-z0-9]+", p.lower())), len(p)))
  fallback = tiers[0] + tiers[1] + (asked if len(asked) == 1 else [])
  return next((p for p in fallback if p and not piece.inside_only(p)), "")

 def line_tag(self, piece, needs, prop, qtext, several):
  """Finds the product name a pointing line is about: from the line outside the phrase, the tags of
        needs on that property, or the question. Returns an empty string when none maps."""
  product_of = self.run.talk.product_of
  k = product_of(piece.outside)
  if k is None:
   tags = [t for n in needs if same_prop(self.need_name(n), prop) for t in re.findall(r"\(([^)]*)\)", n.partition(":")[0])]
   k = product_of(f"x ({tags[-1]})") if tags else None
  if k is None and qtext:
   k = product_of(qtext)
  return several[k]["product"] if k is not None else ""

 def kept_pointing(self, mine, relation, seller):
  """Returns a need from the answer that already points with the same relation at the same seller.
        None when no such need exists, so a new one gets built."""
  for n in mine:
   value = n.partition(":")[2]
   if not (ref_text(value) or re.search(DIFFER_WORDS, value.lower())):
    continue
   if ref_relation(value)[0] == relation and seller in ref_name(value):
    return n
  return None

 def place_pointing_need(self, answer, piece, qtext):
  """Rewrites the listener answer for one pointing line: one 'head: phrase' need replaces the needs on
        that property, needs that only describe the target are removed, the reference entry is merged.
        Also records the need under ref_needs and logs ref_line."""
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
   """True when a requirement's product tag overlaps the line's tag, or tags do not apply.
            Keeps the rewrite inside one product of a multi-product request."""
   tags = re.findall(r"\(([^)]*)\)", str(req).partition(":")[0])
   return not several or not tag or not tags or bool(kind_words(tags[-1]) & kind_words(tag))
  mine = [n for n in needs if same_prop(name(n), prop) and same_item(n)]
  relation, tolerance, unit = ref_relation(piece.phrase)
  seller = ref_name(piece.parsed["seller"])
  keep = self.kept_pointing(mine, relation, seller)
  head = mine[0].partition(":")[0].strip() if mine else (f"{prop} ({tag})" if tag else prop)
  if several and tag and not re.findall(r"\(([^)]*)\)", head):
   head = f"{head} ({tag})"
  need = keep or f"{head}: {piece.phrase}"
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
  """Cleans the rest of the answer after a pointing need is placed: requirements_now loses unwritten
        entries on that property or about the target and gains the need; dropped and no_preference
        lose entries the phrase covers."""
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
  """Creates or updates the answer's reference entry for the property from the parsed line.
        Seller, kind and details are replaced if the entry names another seller; relation, tolerance and
        unit are set unless the line's relation is plain 'same' and one exists."""
  parsed = piece.parsed
  entry = next((r for r in refs if same_prop(str(r.get("property") or ""), prop)), None)
  if entry is None:
   entry = {"property": prop}
   refs.append(entry)
  if not self.ref_seller_ok(entry.get("seller"), piece.text) or ref_name(entry.get("seller")) != seller:
   entry.update(seller=parsed["seller"], kind=parsed["kind"], details=list(parsed["details"]))
  elif not entry.get("details"):
   entry["details"] = list(parsed["details"])
  entry.setdefault("kind", parsed["kind"])
  if relation != "same" or not entry.get("relation"):
   entry.update(relation=relation, tolerance=tolerance, unit=unit)

 @staticmethod
 def names_property(prop, text):
  """True when the text holds a 3+ letter word of the property name, or one sharing its first 4.
        Also True when the property has no such word."""
  words = [w for w in prop_words(prop) if len(w) >= 3]
  if not words:
   return True
  tokens = set(re.findall(r"[a-z]+", str(text).lower()))
  for tok in tokens:
   for word in words:
    if tok == word or (len(word) >= 4 and len(tok) >= 4 and tok[:4] == word[:4]):
     return True
  return False

 def record_card(self, ident):
  """Formats a listing for the hop prompt: id, sellers, category, title, up to 40 facts per property
        map and the notes, as text capped at 5000 characters."""
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
   if not isinstance(val, str) or not val.strip() or len(val) > 60 or "id" in low:
    continue
   if (re.search(SELLER_KEYS, low) or str(key) in learned) and not re.search(r"note|text|desc|comment|detail|info|message|section", low):
    sellers.add(str(val))
  card = {"id": ident, "sold by": " / ".join(sorted(sellers)), "category": self.run.watch.shelf_of(row),
    "title": self.run.shelf.title_of(row)[:160], "facts": facts, "notes": self.note_text(row)[:900]}
  return as_text(card, 5000)

 def ask_hop(self, ref, pick, reason):
  """Asks the first CHECKER model the REF_HOP question about the picked listings' record cards.
        Adds the earlier answer's fault on a retry. Counts hop_calls, logs ref_hop; returns the JSON reply,
        or None when the hop budget is spent."""
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
  """Acts on a hop answer: a missing answer twice gives up; a target outside the pick or with missing
        details marks the pick judged, and so does (for the target only, with no retry call) a target that
        v73_unshown finds lacking a detail. check_hop validates the rest with one retry; a remaining fault
        rejects the target or gives up. Else it gives up on a loop, follows a go_to, or resolves."""
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
  unshown = self.v73_unshown(ref, target)
  if unshown:
   ref["judged"].add(target)
   self.remember_ids(pick)
   self.run.log.append({"kind": "v73_h_reject", "prop": ref["prop"], "target": target, "detail": unshown[:80], "by": "hop"})
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
   self.give_up(ref, "its notes loop back to a listing followed before")
  elif go:
   self.follow(ref, target, go)
  else:
   self.resolve(ref, target, value, pairs)

 @staticmethod
 def clean_strings(raw):
  """Returns the non-empty, stripped string forms of a list's scalar items; booleans are dropped.
        Anything that is not a list gives an empty list."""
  items = raw if isinstance(raw, list) else []
  return [str(x).strip() for x in items if isinstance(x, (str, int, float)) and not isinstance(x, bool) and str(x).strip()]

 def check_go_to(self, ref, target, go, row, notes, note_tokens):
  """Validates a go_to: its seller must be written in the notes, in a sentence naming the property,
        and it must not be the target itself. Keeps go_to details with at least half their words in notes.
        Returns (where, problem, next target, value, pairs)."""
  seller = str(go["seller"]).strip()
  pattern = r"(?<![a-z0-9])" + re.escape(seller.lower()) + r"(?:'s)?(?![a-z0-9])"
  if not re.search(pattern, notes.lower()):
   return "answer", f"the go_to seller '{seller}' does not appear in {target}'s notes.", None, "", []
  sentences = re.split(r"(?<=[.!?])\s+", notes)
  if not any(re.search(pattern, s.lower()) and self.names_property(ref["prop"], s) for s in sentences):
   return "answer", f"the note naming {seller} covers another property, not {ref['prop']}.", None, "", []
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
  """Counts the target details whose tokens all appear among a listing's field names and values.
        check_hop compares picked listings with it."""
  names, values = self.token_parts(self.listing_index().get(ident) or {})
  return sum(1 for d in ref["target"]["details"] if set(ref_toks(d)) and set(ref_toks(d)) <= names | values)

 def check_hop(self, ref, target, answer, pick=None):
  """Checks a hop answer against the target's record: each counted detail must be possibly shown and no
        other pick may show more; a go_to goes to check_go_to; else the value must be in the record and
        paired values in the notes. Returns (where, problem, go_to target, value, pairs)."""
  row = self.listing_index().get(target) or {}
  names, values = self.token_parts(row)
  for d in ref["target"]["details"]:
   if d in (ref.get("loose") or set()):
    continue
   if not self.may_show(row, d, (names, values)):
    return "target", f"{target} lacks '{d}', so it cannot be the listing described.", None, "", []
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
   return "answer", f"the value '{value}' does not appear in {target}'s record.", None, "", []
  pairs = []
  for p in dict.fromkeys(self.clean_strings(answer.get("goes_with"))):
   toks = set(ref_toks(p))
   if toks and toks <= note_tokens and toks != value_tokens:
    pairs.append(p)
  return "", "", None, value, pairs[:3]

 def follow(self, ref, target, go):
  """Moves a chase to the go_to listing: adds the current target to the chain and resets search state.
        Gives up on a loop or when REF_HOPS would be reached; logs ref_go and tells the actor."""
  ref["chain"].append(target)
  key = self.target_key(go)
  if key in ref["seen"]:
   self.give_up(ref, "its notes loop back to a listing followed before")
   return
  if ref["hops"] + 1 >= REF_HOPS:
   self.give_up(ref, f"more than {REF_HOPS} listings in succession redirect the reader")
   return
  ref["seen"].add(key)
  ref["hops"] += 1
  ref.update(target=go, judged=set(), queries=[], pages=0, tries=0, fails=0, unindexed=False, loose=set())
  self.run.log.append({"kind": "ref_go", "prop": ref["prop"], "from": target, "to": go})
  self.tell_actor(f"{target} is the listing found for {ref['head']}; its notes redirect the reader to {self.describe(ref)}, "
      "which the code follows.")

 @staticmethod
 def resolved_value(ref):
  """Writes the requirement value for a resolved chase by relation: a range for within, a bound for
        at most/at least, 'not X' parts joined by 'and' for differ, the paired values alone joined by 'or' for
        partner (v7.5), else the value and pairs joined by 'or'."""
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
  if relation == "differ":
   return " and ".join(f"not {v}" for v in [value] + others)
  if relation == "partner" and others:
   return " or ".join(others)
  return " or ".join([value] + others)

 def resolve(self, ref, target, value, pairs):
  """Closes a chase with a value: marks it resolved, swaps the new requirement into Needs, adds it to
        needs.resolved and bumps the needs version, then logs and tells the actor.
        Gives up instead when the value is empty."""
  if not str(value or "").strip():
   self.give_up(ref, f"{target} shows no value for it")
   return
  if ref["relation"] == "partner" and not [p for p in pairs if str(p).lower() != str(value).strip().lower()]:
   self.run.log.append({"kind": "v75_partner_none", "prop": ref["prop"], "target": target, "value": str(value)[:60]})
   self.give_up(ref, f"{target} shows {value} but names no value that goes with it, and the shopper wants that partner")
   return
  ref["chain"].append(target)
  ref.update(status="resolved", value=str(value).strip(), goes_with=list(pairs))
  new_req = f"{ref['head']}: {self.resolved_value(ref)}"
  ref["now"] = new_req
  self.swap_in(ref, new_req)
  self.needs.resolved.append(new_req)
  self.needs.version += 1
  self.run.log.append({"kind": "ref_resolved", "req": ref["req"][:200], "now": new_req, "chain": list(ref["chain"])})
  chain = ref["chain"]
  via = f", traced from {chain[0]} along the listings' notes" if len(chain) > 1 else ""
  joined = " and ".join(pairs)
  if ref["relation"] == "differ":
   pair = (f"; its notes pair {joined} with it" if pairs else "") + (
    ". The shopper's value MUST DIFFER from this one: any other value meets it, but that one does not")
  elif ref["relation"] == "partner":
   pair = (f"; its notes pair it with {joined}. The shopper wants that partner: {joined} meets it, while "
     f"{ref['value']} itself does not")
  else:
   pair = f"; its notes put {ref['value']} together with {joined}, so each of those meets it" if pairs else ""
  self.tell_actor(f"RESOLVED: {ref['head']} is set by another listing; {target} shows {ref['value']}{via}{pair}. The "
      f"requirement now reads: {new_req}.")

 def swap_in(self, ref, new_req):
  """Replaces the chase's pointing requirement with the resolved one in written, uncovered and now
        lists, the reading's requirements and the matching product entries. Appends it to uncovered
        when nothing was replaced."""
  needs = self.needs
  seller = ref_name(ref["first"]["seller"])

  def mine(req):
   """True when a requirement belongs to the chase: ref_match, or same property and a value naming the
            first target's seller, with a compatible product tag."""
   if self.ref_match(ref, req):
    return True
   head, _, value = str(req).partition(":")
   bracketed = re.findall(r"\(([^)]*)\)", head)
   if bracketed and ref["tag"] and not kind_words(bracketed[-1]) & kind_words(ref["tag"]):
    return False
   return bool(seller) and seller in ref_name(value) and same_prop(self.head_prop(head), ref["prop"])
  changed = False
  before = list(needs.written)
  after = [new_req if mine(r) else r for r in before]
  if after != before:
   changed = True
  needs.written = list(dict.fromkeys(after))
  before = list(needs.uncovered)
  after = [new_req if mine(r) else r for r in before]
  if after != before:
   changed = True
  needs.uncovered = list(dict.fromkeys(after))
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
  """Marks an open chase stuck with its reason, logs ref_stuck and tells the actor the value is unknown.
        No effect on chases that are not open."""
  if ref["status"] != "open":
   return
  ref["status"], ref["why"] = "stuck", why
  self.run.log.append({"kind": "ref_stuck", "req": ref["req"][:200], "why": why[:300], "chain": list(ref["chain"])})
  self.tell_actor(f"NOT RESOLVED: {ref['head']} ({why}). The value stays unknown: if notes of a listing you read show where "
      "it ends, take that value; else it never justifies declining or rejecting a listing.")

 def tell_actor(self, text):
  """Appends a reference notice to the latest tool or user message in the actor chat.
        When that is not possible the notice is queued in pending_tells for ref_line."""
  chat = self.run.chat
  if chat and chat[-1].get("role") in ("tool", "user") and isinstance(chat[-1].get("content"), str) and len(chat) > 2:
   chat[-1]["content"] += "\n[reference] " + text
  else:
   self.pending_tells = list(self.pending_tells or []) + [text]

 def ref_line(self):
  """Builds the reference part of the actor's status line: one summary per chase by status, then any
        queued notices (cleared). Called by the run's status head; returns '' on error."""
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
     parts.append(f"{ref['head']}: unresolved, value unknown (never reject a listing or decline over it)")
   told, self.pending_tells = self.pending_tells, None
   text = "\n[references] " + "; ".join(parts) if parts else ""
   if told:
    text += "".join("\n[reference] " + t for t in told)
   return text
  except Exception:
   return ""

 @staticmethod
 def resolved_line(ref):
  """Formats one bullet for a resolved chase for ref_note: the new requirement, the listing it came
        from, the route through notes and paired values; differ chases also name their chain."""
  chain = ref["chain"]
  via = f" (traced from {chain[0]} along the listings' notes)" if len(chain) > 1 else ""
  joined = " and ".join(ref["goes_with"])
  if ref.get("relation") != "differ":
   pair = f"; its notes put {ref['value']} together with {joined}, so each of those meets it" if ref["goes_with"] else ""
   return f"- {ref['now']}: the value shown on {chain[-1]}{via}{pair}."
  pair = f"; its notes pair {joined} with it" if ref["goes_with"] else ""
  return (f"- {ref['now']}: the shopper's {ref['prop']} must DIFFER from the value shown on {chain[-1]} "
    f"({ref['value']}){via}{pair}: every other value meets it, but not that one; its chain's listings "
    f"({', '.join(chain)}) carry it.")

 def ref_note(self, thing=""):
  """Builds the reference section for checker prompts (item check, rank check, decline audit).
        Resolved chases get resolved_line; unresolved ones their chain so far and a caution. Adds up to two
        records of the last chain listings, leaving out the item being checked."""
  try:
   lines, ends = [], []
   for ref in self.run.refs or []:
    status = ref["status"]
    if status == "resolved":
     lines.append(self.resolved_line(ref))
    elif status in ("open", "stuck"):
     chain = ref["chain"]
     where = "its chain so far: " + " -> ".join(chain) if chain else "the listing it points at was not found"
     lines.append(f"- {ref['head']}: the code could NOT RESOLVE it (" + where + "). Should a record below be that listing (or the final one its chain reached) and give its own value for it "
         "rather than passing the reader on, the requirement is that value; otherwise its value is unknown and no item "
         "may fail on it alone.")
     ends += [i for i in chain[-2:] if i != thing]
   if not lines:
    return ""
   text = ("\n\nREQUIREMENTS SET BY POINTING AT ANOTHER LISTING (traced by the code via that listing's notes in the store's "
     "records):\n" + "\n".join(lines))
   records = [r for i in dict.fromkeys(reversed(ends)) for r in self.archive.about(i, 1)]
   if records:
    text += "\nRECORDS OF THE LAST LISTINGS ITS CHAIN REACHED:\n" + "\n".join(records[:2])
   return text
  except Exception:
   return ""


class Coverage:
 """Search-coverage engine for cheapest-fit requests, held by the ShoppingRun as run.cover.
    Learns whether the search tool's price-limit argument really filters, walks price-filtered and
    value-led listings below the current pick, and queues the cheapest unread rows for opening.
    Also annotates every search result for the actor (word rules, unseen totals, hits) and pulls
    shopper text out of tool results. All of its memory lasts for one episode."""
 def __init__(self, run):
  """Binds the engine to its ShoppingRun and starts every per-episode ledger empty.
        Covers search hits and scores, the learned price mode and broken modes, seen totals, walked and
        seeded products, label-search attempts, value-page and open counters, hint counters, queued walk."""
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
  self.v73_page = None
  self.v73_paged = False

 def v73_full_search(self):
  """Book entry of the newest search whose typed words carry every known requirement (v7.3 C), or None.
        A requirement counts as carried when a word of its value (two or more letters or digits, no linking word)
        is among the search words, both compared by stem; requirements with no such word are not counted, and
        with none left there is no such search."""
  run = self.run
  stem = run.watch.stem
  wanted = []
  for req in self.needs.props_left():
   words = {stem(w) for w in re.findall(r"[a-z0-9]+", str(req).partition(":")[2].lower()) if len(w) >= 2 and w not in LINK_WORDS}
   if words:
    wanted.append(words)
  if not wanted:
   return None
  for entry in reversed(list(run.page_book.values())):
   said = {stem(w) for w in re.findall(r"[a-z0-9]+", " ".join(run.watch.free_text(entry.get("args") or {})).lower())}
   if said and entry.get("read") and all(words & said for words in wanted):
    return entry
  return None

 def v73_page_hold(self):
  """Order hold for a cheapest single-product errand (v7.3 C): when the newest search carrying every known
        requirement (v73_full_search) has a next page never read, queue that page for walk_step and return a no-step
        note. Once per errand, with more than FIRST_ORDER_BY steps and CLOSING + 60 seconds left; else ""."""
  run = self.run
  try:
   if self.v73_paged or self.v73_page or not self.cheapest_rule() or self.needs.items() or not run.role("search"):
    return ""
   if run.moves_left() <= FIRST_ORDER_BY or run.wire.spare() < CLOSING + 60:
    return ""
   paging = run.watch.page_arg()
   entry = self.v73_full_search() if paging else None
   following = run.watch.next_page(entry) if entry else None
   if following is None:
    return ""
   self.v73_paged = True
   self.v73_page = {**entry["args"], paging[0]: following}
   query = " ".join(run.watch.free_text(entry["args"]))
   run.log.append({"kind": "v73_c_page", "query": query[:120], "page": following})
   return ("Not run yet (no step used). The cheapest fit is wanted, and the search '" + query + "' carries every known "
     "requirement, but its next page was never read: the code reads that page for you now. Then order, or "
     "check a cheaper fit it shows and order that one.")
  except Exception as err:
   run.log.append({"kind": "v73_c_error", "error": f"{type(err).__name__}: {err}"[:300]})
   return ""

 @property
 def needs(self):
  """Shortcut property to the run's Needs object (requirements, items, priority, budget).
        Nothing is cached on the Coverage instance."""
  return self.run.needs

 @property
 def archive(self):
  """Shortcut property to the run's Records archive of observations and read listings.
        Nearly every lookup in this class walks archive.items or checks archive.ids and archive.read."""
  return self.run.archive

 def cheapest_rule(self):
  """Tells whether the request's priority is the cheapest fitting item.
        True only when Needs.priority() is exactly 'cheapest'; most walks in this class are gated on it."""
  return self.needs.priority() == "cheapest"

 def tell_tool(self, text):
  """Appends a text note on a new line to the content of the newest chat message.
        Callers check last_is_tool first, so the note lands on the latest tool result the actor reads."""
  chat = self.run.chat
  chat[-1]["content"] = str(chat[-1]["content"]) + "\n" + text

 def last_is_tool(self):
  """Reports whether the run's chat is non-empty and its newest message has the tool role.
        Guards tell_tool so code notes are only attached to a tool result."""
  chat = self.run.chat
  return bool(chat) and chat[-1].get("role") == "tool"

 @staticmethod
 def on_home(entry, home):
  """Lists the row ids of a page_book entry whose recorded category label is in the home set.
        Reads the entry's row_shelf map (id to lowercased label); an entry without one gives []."""
  return [i for i, s in (entry.get("row_shelf") or {}).items() if s in home]

 def priced_list(self, ids):
  """Formats listing ids as a comma-separated 'id (price)' string for notes to the actor.
        Prices come from the archive; used when announcing which rows the code opens next."""
  return ", ".join(f"{i} ({self.archive.price(i):g})" for i in ids)

 def may_page(self, floor):
  """Decides whether one more scripted search page may be spent.
        Needs more than floor + 1 moves left, a wire that is neither done nor broken, and at least
        CLOSING + 30 seconds of spare time."""
  run = self.run
  wire = run.wire
  return run.moves_left() > floor + 1 and not wire.done and not wire.broke and wire.spare() >= CLOSING + 30

 def note_depth_cap(self, call_args, paging, dicts, total, obs):
  """Detects a store that stops paging before its reported total and records how deep it goes.
        On a full whole-page result whose next/more markers are all empty while the total is larger,
        sets depth_cap to the rows reachable so far, keeping the smallest value seen."""
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
  """Finds the one numeric search argument that can act as an upper price limit.
        Skips the page key, paging keys, booleans, values of 1 or less and names hinting at page, size,
        count, minimum or offset; returns the key only if exactly one candidate is left, else None."""
  found = []
  for k, v in call_args.items():
   if not is_number(v) or isinstance(v, bool) or k == page_key or k in paging_keys or not v > 1:
    continue
   if not re.search(r"page|size|count|result|min|low|from|start|offset", str(k).lower()):
    found.append(k)
  return found[0] if len(found) == 1 else None

 @staticmethod
 def maker_free(base, dicts, bound_key, shelf_key, fixed):
  """Builds the reusable argument template of a working price-filter search.
        Drops the bound and shelf keys, and any free-text value that every returned row repeats exactly
        in a non-category field (a brand, say), unless the key is one of the fixed mode keys."""
  def carried_by_all(val):
   """Tests whether every row holds the given text, case-insensitively, as one of its field values.
            Category, shelf, department and aisle fields do not count as holding it."""
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
  """Learns from one search result whether the search tool's price-limit argument filters.
        Records the total per argument set and the paging depth cap; a row priced over the limit marks the
        mode broken. A clean single-label result sets price_mode (template, shelf key, bound key, int),
        replacing a working mode only when the new template has no more free-text words."""
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
  """Returns as a float the first numeric, non-boolean field of a row whose key names price or cost.
        None when the row carries no such field."""
  for key, val in row.items():
   if is_number(val) and not isinstance(val, bool) and re.search(r"price|cost", str(key).lower()):
    return float(val)
  return None

 def price_spec(self):
  """Returns the learned price-filter mode, or None when none is learned or it was marked broken.
        The dict holds sig, tmpl, mode, shelf, bound, words and int (whether the bound must be an integer)."""
  spec = self.price_mode
  if not spec or spec["sig"] in self.broken_modes:
   return None
  return spec

 def cg_listing(self, args):
  """Tells whether search arguments are a call in the learned price-filter mode.
        With fixed mode keys, each must equal its learned value; without any, the bound key must be present.
        False when no working price mode is known."""
  spec = self.price_spec()
  if not spec or not isinstance(args, dict):
   return False
  if not spec["mode"]:
   return spec["bound"] in args
  return all(args.get(k) == v for k, v in spec["mode"].items())

 def widest_page(self):
  """Returns the largest page size recorded across the run's page_book entries.
        Zero when no search page has been booked yet; serves as the page-size estimate."""
  sizes = [entry.get("size", 0) for entry in self.run.page_book.values()]
  return max(sizes or [0])

 def reachable_depth(self):
  """Estimates how many result rows the search tool can reach by paging.
        Uses the paging argument's maximum with the widest page size (page count times size, or offset
        span plus one page); falls back to depth_cap for token paging, no maximum or no size yet."""
  paging = self.run.watch.page_arg()
  size = self.widest_page()
  if not paging or size <= 0 or paging[1] == "token" or paging[3] is None:
   return self.depth_cap
  last, first = paging[3], paging[2]
  if paging[1]:
   return int((last - first + 1) * size)
  return int(last - first + size)

 def shelf_spelling(self, labels):
  """Finds how the store itself spells a category, given a set of lowercased labels.
        Counts category-field values of archived dicts that carry a known listing id and match a label;
        returns the most frequent spelling (ties go to the alphabetically first), or ''."""
  if not labels:
   return ""
  counts = {}
  known = self.archive.ids

  def tally(fields):
   """Counts one dict's category spelling when the dict holds a known listing id in an id field.
            Takes its first short category/shelf value from a non-id key and tallies it if the lowercase form
            is one of the wanted labels."""
   if not any(isinstance(v, str) and v in known and "id" in str(key).lower() for key, v in fields.items()):
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
  """Gives the largest price limit strictly under a price, for a below-the-pick filter call.
        A whole price drops by one, a fractional price under an integer-only bound is floored, and any
        other price drops one cent. None for a non-number or a result of zero or less."""
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
  """Returns the value halfway between two prices, used to bisect the price-filter bound.
        Integer division when the bound is integer-only or both ends are whole; else rounded to cents."""
  spec = self.price_spec() or {}
  whole = spec.get("int") or (float(a).is_integer() and float(b).is_integer())
  return (int(a) + int(b)) // 2 if whole else round((float(a) + float(b)) / 2, 2)

 @staticmethod
 def filter_args(spec, label, bound):
  """Builds the arguments of one price-filtered category listing from a learned spec.
        Fills the spec's template with the shelf label and the bound, cast to int when the bound is
        integer-only or whole."""
  limit = int(bound) if spec.get("int") or float(bound).is_integer() else bound
  return {**spec["tmpl"], spec["shelf"]: label, spec["bound"]: limit}

 def cg_products(self):
  """Returns the product slots to work through: 0..n-1 for a multi-item request.
        A single-product request gives [None], the slot that stands for the whole request."""
  return list(range(len(self.needs.items()))) or [None]

 def can_list_below(self, k):
  """Decides whether a price-filter walk below the pick is possible for product k.
        Needs the cheapest rule, a working price mode, a paging argument, a store spelling for k's home
        shelf, and k not walked yet; slot None is refused on multi-item requests."""
  if self.needs.items() and k is None:
   return False
  if not self.cheapest_rule() or k in self.walked_products or self.price_spec() is None:
   return False
  return self.run.watch.page_arg() is not None and bool(self.shelf_spelling(self.run.walk.walk_home(k)))

 def read_on_home(self, home):
  """Collects the prices of read listings on the given home shelves that are still usable.
        Skips unpriced listings, dead or refused lines, and ones whose last stock reading was out."""
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
  """Picks the price a below-the-pick walk for product k has to undercut.
        The lowest of the pick's price and any held cart aim; with neither, and a numeric bound among k's
        requirements, the cheapest unread candidate row or read listing on its home shelf. None if none."""
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
  """Tells whether product k has a remaining requirement holding a number and a comparison word.
        Only for multi-item requests and a real slot; comparison words come from the CH_BOUND pattern."""
  if not self.needs.items() or k is None:
   return False
  product_of = self.run.talk.product_of
  for req in self.needs.props_left():
   head, _, value = str(req).partition(":")
   if product_of(head) == k and re.search(r"\d", value) and re.search(CH_BOUND, value.lower()):
    return True
  return False

 def rows_on(self, home):
  """Collects ids of known listings that appear in archived records with a home-shelf label.
        Walks every nested dict of every archived observation."""
  watch, known = self.run.watch, self.archive.ids
  found = set()

  def collect(fields):
   """Adds one nested dict's row id to the result set when the archive knows that id
            and the dict's category label is one of the home shelves."""
   ident = watch.row_mark(fields)
   if ident in known and watch.shelf_of(fields) in home:
    found.add(ident)
  for record in self.archive.items:
   walk_dicts(record, collect)
  return found

 def cheaper_in_stock(self, home, bound):
  """Checks whether any archived row on the home shelves costs at most the bound and is in stock.
        The price-filter walks use it to catch a filter that returned nothing yet hid such a row."""
  watch, known = self.run.watch, self.archive.ids
  hit = False

  def check(fields):
   """Sets the hit flag for one nested dict on a home shelf that carries a known id field.
            The hit needs a price at or below the bound and a stock state that reads as fully in stock."""
   nonlocal hit
   if hit or watch.shelf_of(fields) not in home:
    return
   if not any(isinstance(v, str) and v in known and "id" in str(key).lower() for key, v in fields.items()):
    return
   price = self.row_price(fields)
   hit = price is not None and price <= bound and watch.stock_state(fields) is True
  for record in self.archive.items:
   walk_dicts(record, check)
  return hit

 def sold_out_marks(self, ids):
  """Maps the given ids to whether archived records show them out of stock.
        Only dicts naming one of the ids in an id field and giving a readable stock state count; a later
        record overwrites an earlier one."""
  watch = self.run.watch
  sold_out = {}

  def note_stock(fields):
   """Notes the stock state of one nested dict that names a wanted id in an id field.
            Stores, under the dict's row id, whether it reads as out of stock; no stock field means no entry."""
   if not any(isinstance(v, str) and v in ids and "id" in str(key).lower() for key, v in fields.items()):
    return
   state = watch.stock_state(fields)
   if state is not None:
    sold_out[watch.row_mark(fields)] = not state
  for record in self.archive.items:
   walk_dicts(record, note_stock)
  return sold_out

 def cheapest_unread(self, ids, k, below=None):
  """Lists unread candidate rows among ids as (price, id) pairs, cheapest first.
        Drops read, refused, sold-out and unpriced ids, any whose price times product k's quantity exceeds
        the budget, and, when below is given, any priced at or above it."""
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
  """Tells whether this store's keyword search is known to need every query word.
        True for word_kind 'all', or when a lead-word count is learned and the kind is not 'any'."""
  kind = self.run.word_kind
  return kind == "all" or (bool(self.run.lead_words) and kind != "any")

 def value_led_rows(self, k):
  """Collects home-shelf rows of id-ordered searches whose query carries one of product k's values.
        Only in all-words stores; price-filter listings are left out. Empty otherwise."""
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
  """Lists (index, text) pairs of the remaining requirements that belong to product k.
        On multi-item requests a requirement tied to another product is skipped; untied ones stay."""
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
  """Strips a requirement down to its lowercased value text after the first colon.
        Parenthesised '(stated)' and '(revealed ...)' tags are removed."""
  return re.sub(r"\((?:stated|revealed[^)]*)\)", "", req.partition(":")[2]).lower()

 def finished_value_walk(self, entry, home, k, single):
  """Tells whether one booked search is a fully read id-ordered value search for product k.
        Every page must be read and every match seen, it must show home-shelf rows, and its query must carry
        a requirement index from single. Price-filter listings never count."""
  if entry.get("id_order") is not True or self.cg_listing(entry["args"]) or not entry.get("first"):
   return False
  if self.run.watch.next_page(entry) is not None or len(entry.get("row_shelf") or {}) < entry["total"]:
   return False
  return bool(self.on_home(entry, home) and self.run.walk.walk_carries(entry["args"], k) & single)

 def value_search_complete(self, k):
  """Tells whether a finished value search already covers product k in an all-words store.
        Collects k's requirements whose value has a single option and asks finished_value_walk about each
        booked search. Any error reads as False."""
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
  """Splits product k's requirement values into word groups and leading figure/unit pairs.
        Values that point at another listing are skipped; each option gives a (number, unit) figure or a
        list of stemmed 3+ letter words, while comparison amounts give neither. Returns (groups, figures)."""
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
  """Tells whether a listing's archived text carries one of product k's required values.
        Pools the string fields of every dict naming the id, then looks for a figure followed by its unit,
        or all stemmed words of one option. False when k has no usable values."""
  stem = self.run.watch.stem
  wanted, figures = self.wanted_words(k)
  if not wanted and not figures:
   return False
  texts = []

  def gather(fields):
   """Adds the lowercased string fields of a nested dict to the text pool
            when that dict names the listing id in an id field."""
   if any(isinstance(v, str) and v == ident and "id" in str(key).lower() for key, v in fields.items()):
    texts.extend(str(v).lower() for v in fields.values() if isinstance(v, str))
  for record in self.archive.items:
   walk_dicts(record, gather)
  have = {stem(w) for w in re.findall(r"[a-z0-9]{3,}", " ".join(texts))}
  body = " ".join(texts)
  for figure, unit in figures:
   if re.search(r"(?<![\d.,])" + re.escape(figure) + r"\s*-?\s*" + re.escape(stem(unit)), body):
    return True
  return any(all(w in have for w in words) for words in wanted)

 def burst_room(self):
  """Estimates the questions and steps left before a scripted burst of searches.
        Returns (questions still allowed, moves left minus the order reserve and two steps per question);
        questions count only while asking is on and a message tool exists."""
  run = self.run
  left = max(0, run.ask_limit() - run.msgs_sent) if run.asking and run.role("message") else 0
  return left, run.moves_left() - run.intake.order_due() - 2 * left

 def option_words(self, piece, own_name):
  """Turns one requirement option into query words for a composed value search.
        None for exclusions and for amount phrases without a leading figure; drops comparison and link
        words, possessive s, decimals, and words of the property's own name except a figure's unit."""
  figure = self.run.walk.fig_amount(piece)
  if re.search(EXCLUDE_WORDS, piece) or (self.run.walk.amount_part(piece) and not figure):
   return None
  piece = re.sub(r"\b(?:" + VALUE_LINKS + r")\b", " ", re.sub(r"'s\b", "", piece))
  words = []
  for w in re.findall(r"\d+(?:[.,]\d+)+|[a-z0-9]+", piece):
   long_enough = len(w) >= 2 or (figure and w == figure[0])
   if not long_enough or re.search(r"[.,]", w) or w in LINK_WORDS:
    continue
   if w in own_name and not (figure and w == figure[1]):
    continue
   words.append(w)
  return words

 def firm_values(self, k):
  """Lists product k's firm requirement values as option word lists for composed searches.
        Reads closer.code_values and skips pointer values; an either/or value gives up to three options,
        any other keeps its first and-part. Duplicate options and values are dropped."""
  values = []
  for head, value in self.run.closer.code_values(k):
   low = re.sub(r"\([^)]*\)", " ", str(value).lower().replace("’", "'")).strip()
   if not low or ref_text(low):
    continue
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
   if options and options not in values:
    values.append(options)
  return values

 def title_stems_on(self, k):
  """Collects the stemmed title words of every archived row on product k's home shelves.
        Returns one set per row occurrence, so a row met in two records appears twice."""
  watch = self.run.watch
  home = self.run.walk.walk_home(k)
  titles = []

  def collect(fields):
   """Adds the stemmed word set of a nested dict's title when the dict sits on a home shelf
            and carries a row id the archive knows."""
   if watch.shelf_of(fields) in home and watch.row_mark(fields) in self.archive.ids:
    titles.append({watch.stem(w) for w in re.findall(r"[a-z0-9]+", self.run.shelf.title_of(fields).lower())})
  for record in self.archive.items:
   walk_dicts(record, collect)
  return titles

 def label_words(self, k, label):
  """Chooses the category words added to a composed value search for product k.
        Normally every label word of 2+ characters minus link words. When the label is known not to be
        indexed: words found in at least half of the shelf's titles, else the last word of the kind."""
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
  """Joins value word groups and label words into one query, skipping repeated stems.
        Keeps only the first lead_words words when the store reads that many, then trims whole words
        to fit the text field's maxLength."""
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
  """Builds composed value-plus-category queries for product k as (combined, single) lists.
        Single forms use the first value; combined forms use every value and vary the first value that has
        several options. All options are used under the cheapest rule, else only the first. With one value
        only the single forms come back, as the first list."""
  values = self.firm_values(k)
  if not values:
   return [], []
  label_terms = self.label_words(k, label)
  every_option = self.cheapest_rule()

  def options(v):
   """Selects which options of one value to try: all of them when the cheapest is wanted,
            otherwise only the first."""
   return v if every_option else v[:1]

  def unique(forms):
   """Drops empty strings and repeats from a list of query forms,
            keeping the order in which forms first appear."""
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
  """Picks the next composed query to send, or reports that the forms are settled.
        Goes through the first list, then the second: returns an unsent form, or '' once a list's sent
        forms found matches; None when every form was sent and found nothing; '' when both are empty."""
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
  """Extracts the small-enum mode arguments of a search call as a sorted tuple of pairs.
        Only string arguments whose schema enum has at most four options count; lets searches run in the
        same mode be compared."""
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
  """Turns text into a frozenset of stemmed words, cut to the store's lead-word count.
        Used to tell whether two queries read the same to the store."""
  stem = self.run.watch.stem
  return frozenset(stem(w) for w in re.findall(r"[a-z0-9]+", str(text).lower())[: self.run.lead_words or None])

 def sent_total(self, text, text_key, mode):
  """Looks up the result total recorded for a query already sent in the same mode.
        Matches on the stemmed word set of the text field and on every mode value; None when not sent."""
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
  """Decides once whether keyword search matches category label words, from the newest record.
        Counts rows on exactly that label whose other fields lack some label word; two or more such rows
        set label_indexed to True. Called after each composed value search."""
  if self.label_indexed is not None or not self.archive.items:
   return
  watch = self.run.watch
  label_stems = {watch.stem(w) for w in re.findall(r"[a-z0-9]+", re.sub(r"'s\b", "", label.lower())) if len(w) >= 2 and w not in LINK_WORDS}
  hits = 0

  def count(fields):
   """Counts one known row on the searched label whose non-category fields
            lack at least one of the label's stemmed words."""
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
  """Drops ids from the end of a queue until opening them fits the moves left above the floor.
        With a batch details tool, two or more ids cost one step; returns the trimmed queue."""
  while queue and (1 if batch and len(queue) >= 2 else len(queue)) >= self.run.moves_left() - floor:
   queue.pop()
  return queue

 def cheap_queue(self, k, ids, floor, most):
  """Chooses which unread rows to open for product k, cheapest first.
        Under the cheapest rule only rows below the pick, within k's CEIL_OPENS allowance (counted here);
        otherwise up to most. Also capped by the batch tool's limit and by the steps left."""
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
  """Hands the given ids to run.reads and triggers their opening right away.
        Clears rank checks on the given shelves and the order vetting flag, fills run.reads with the
        queue, purpose 'walk', floor and announcement, then calls judge.open_named."""
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
  """Finds a code-built search call for product k with exactly one free-text argument.
        Returns (call, text key) from closer.code_queries, skipping price-filter calls; (None, '') if none."""
  free_text = self.run.watch.free_text
  for call in self.run.closer.code_queries(k):
   keys = [key for key, val in call.items() if isinstance(val, str) and val.strip() and val in free_text(call)]
   if len(keys) == 1 and not self.cg_listing(call):
    return call, keys[0]
  return None, ""

 def pending_form(self, k, label, home, cap_len, text_key, mode):
  """Works out the next composed value search for product k and records its word sets.
        If every form was sent and found nothing while a value-carrying search showed home rows and
        indexing is unknown, marks the label not indexed and rebuilds the forms.
        Returns (query or '', first forms, second forms)."""
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
  self.label_indexed = False
  first, second = self.composed_queries(k, label, cap_len)
  composed.update(self.word_set(q) for q in first + second)
  return self.next_form(first, second, text_key, mode) or "", first, second

 def value_label_search(self):
  """Sends one scripted keyword search of required values plus the store's category spelling.
        Needs more than four steps of room; takes the first product with label tries left and unsettled
        values, builds a bare query from a code template, checks it with ArgCheck, sends it by walk_call
        and passes the result to after_label_search. True when a search was sent."""
  _, room = self.burst_room()
  if room <= 4:
   return False
  run = self.run
  walk = run.walk
  shape = run.shapes.get(run.role("search") or "") or {}
  props = shape.get("properties") if isinstance(shape.get("properties"), dict) else {}
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
   template, text_key = self.label_template(k)
   if not template:
    settled.append(sig)
    continue
   mode = {key: val for key, val in template.items() if isinstance(props.get(key), dict) and props[key].get("enum")}
   required = shape.get("required") or []
   template = {key: val for key, val in template.items() if key == text_key or key in mode or key in required}
   cap_len = props[text_key].get("maxLength") if isinstance(props.get(text_key), dict) else None
   todo, first, second = self.pending_form(k, label, home, cap_len, text_key, mode)
   args, faults = ArgCheck.execute({**template, text_key: todo}, shape) if todo else (None, True)
   if faults or not isinstance(args, dict):
    settled.append(sig)
    continue
   self.label_attempts[k] = self.label_attempts.get(k, 0) + 1
   plural = "s" if len(self.firm_values(k)) > 1 else ""
   walk.walk_call(args, f"Searching the category name in this store's spelling plus the required value{plural} "
         f"('{todo}'): since keyword search covers listing values, titles and categories, this "
         "lists the category's listings that carry the value.")
   self.after_label_search(k, args, todo, text_key, mode, label, home, first or second, paging, len(products) > 1)
   return True
  return False

 def after_label_search(self, k, args, todo, text_key, mode, label, home, forms, paging, several):
  """Reports a composed value search's total on the latest tool result and queues rows to open.
        When every match is shown: under the cheapest rule records it as a value walk (multi-item waits
        for flush_waiting, else its cheapest rows are opened); otherwise opens up to LABEL_OPENS rows.
        Logs a label_search entry."""
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
   note += (" Its unread in-stock, in-budget rows on the requested category" + below + ", cheapest first, are opened next: "
      + self.priced_list(queue) + ".")
  if note and self.last_is_tool():
   self.tell_tool(note + ("" if queue else "\n" + run.status_line()))
  run.log.append({"kind": "label_search", "product": k, "query": todo, "total": total, "every": every, "opened": queue})
  if queue:
   self.launch_reads(queue, home, floor, "Opening that search's cheapest unread rows, since its matches include every listing in the category "
              "that carries the required values.")

 def flush_pools(self, waiting, per):
  """Builds one pool per waiting product of rows to open after the value searches are read.
        Each pool takes unread rows priced under the pick, lowest first, from that product's value walks,
        capped by per and by the product's remaining CEIL_OPENS allowance."""
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
  """Merges per-product pools round-robin into one queue of at most most ids.
        Returns (owner map from id to product, queue); ids already queued are skipped."""
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
  """Opens the rows that multi-item value searches set aside, once the searches are read.
        Takes one or two rows per waiting product in turns, fitted to the batch size and steps, counts them
        against each product's open allowance, launches the reads and notes it. True when queued."""
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
  self.launch_reads(queue, homes, floor, "Cheapest wanted: for each product, opening the cheapest unread rows its value search showed.")
  summary = ("[value searches read] Opened, below every pick, the lowest-priced unread rows in stock and in budget that "
       "each product's value search showed: " + self.priced_list(queue) + ". Per product, "
       "pick the cheapest listing whose own fields satisfy every requirement now in force; a pricier fit is only allowed "
       "once each cheaper candidate has been opened and rejected.")
  if self.last_is_tool():
   self.tell_tool(summary)
  run.log.append({"kind": "value_flush", "products": waiting, "opened": queue})
  return True

 def product_steps(self):
  """Builds a status suffix with each product's value pages read and whether it has a pick.
        Only for multi-item cheapest requests; names the first product without a pick as next. Appended
        to the requirements part of ShoppingRun.status_head; '' otherwise or on error."""
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

 def finished_searches(self, k):
  """Lists booked id-ordered searches read to the end with every match seen.
        Each entry is (stemmed query words, requirement-option marks it carries for k, enum mode);
        price-filter listings are left out."""
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
  """Collects stemmed words of product k's requirement values and of its home shelf labels.
        Used to judge whether a query's extra words are only value or category words."""
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
  """Tells whether a booked search is a value walk for product k that still has pages to read.
        It must be id-ordered, not walked yet, have its first page, not be a price-filter listing, carry
        one of k's values and show rows on k's home shelf."""
  if key in self.valued_walks or entry.get("id_order") is not True or not entry.get("first"):
   return False
  if self.run.watch.next_page(entry) is None or self.cg_listing(entry["args"]):
   return False
  return bool(self.run.walk.walk_carries(entry["args"], k)) and bool(self.on_home(entry, home))

 def page_candidates(self, k, n, home, plan):
  """Lists product k's unfinished value searches worth paging, as sortable candidate tuples.
        Skips ones past the reachable depth, covered by a finished search, over the page allowances, or
        held while questions remain (setting plan['pending']); prunes nested pairs in all-words stores.
        Tuples are (pages used, pages to read, -score, n, key, k)."""
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
   cands = [c for c in cands if not any((o[3] < c[3] and not (c[3] - o[3]) <= terms)
             or (c[3] < o[3] and (o[3] - c[3]) <= terms) for o in cands)]
  return [(used.get(k, 0), take, score, n, key, k) for take, score, key, _ in cands]

 def paging_plan(self, products, several):
  """Builds the shared budget for paging value searches across products.
        Holds the reachable depth, per-product and total page caps, opens per product, the order floor,
        whether all words are needed, questions left, spare room and a per-product share of it."""
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
  """Scripted step that widens coverage for the cheapest fit through value searches.
        Skipped while an open reference has step room, after any order or refusal, or when time is short;
        tries value_label_search, then pages the best candidate search, else flushes waiting products.
        Run by ValuePagesStep; True when it used a step."""
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
  """Reads further pages of one booked search through walk_call, up to most pages.
        Counts each attempt in value_pages_read[k]; stops when may_page fails, a call adds no record or
        the next page does not move. Returns (pages read, whether any was tried)."""
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
  """Pages through one id-ordered value search for product k and reports what it covered.
        Says whether every match was seen or the rest is unreachable, then either defers the opening to
        flush_waiting or opens its lowest-priced unread rows under the pick. Logs a value_pages entry."""
  run = self.run
  watch = run.watch
  paging = watch.page_arg()
  entry = run.page_book.get(key)
  if not paging or not entry:
   return False
  query = " ".join(watch.free_text(entry["args"]))
  known = self.all_words()
  if known:
   say = ("Cheapest wanted; this store's keyword search requires all words and returns results in a fixed id order rather "
       f"than by price, so I read the next page of '{query}' to see every listing that carries these words.")
  else:
   say = ("Cheapest wanted; this store returns search results in a fixed id order rather than by price, so I read the next "
       f"page of '{query}' to see every listing it matched.")
  got, tried = self.read_pages(k, entry, paging, most, floor, say)
  if not got:
   return tried
  self.valued_walk_keys.setdefault(str(k), []).append(key)
  seen = len(entry.get("row_shelf") or {})
  complete = watch.next_page(entry) is None and seen >= entry["total"]
  home = run.walk.walk_home(k)
  if complete:
   reach = " - they include every listing with these words" if known else " - they include every listing it matched"
  elif watch.next_page(entry) is None:
   reach = " - the rest is out of reach, so a cheaper fit may sit beyond the readable rows"
  else:
   reach = ""
  head = f"[value search read] '{query}': seen {seen} of its {entry['total']} matches" + reach + ". "
  closing = ("Pick the cheapest listing whose own fields satisfy every requirement now in force; a pricier fit is only allowed "
       "once each cheaper candidate has been opened and rejected.")
  queue = []
  if defer:
   self.waiting_flush[k] = True
   middle = "After all products' value searches are read, each one's cheapest unread in-stock, in-budget rows are opened. "
  else:
   pick = run.walk.walk_pick(k)
   queue = self.cheap_queue(k, self.on_home(entry, home), floor, CEIL_OPENS)
   if queue:
    middle = ("Its cheapest unread in-stock, in-budget rows on the requested category" + (" below the pick" if pick else "")
        + " are opened next: " + self.priced_list(queue) + ". ")
   else:
    middle = "Among them, no unread in-stock row within budget remains on the requested category. "
  summary = head + middle + closing
  if queue:
   self.launch_reads(queue, home, floor, "Cheapest wanted: I open that search's cheapest unread rows.")
  if self.last_is_tool():
   self.tell_tool(summary + ("" if queue else "\n" + run.status_line()))
  run.log.append({"kind": "value_pages", "product": k, "query": query, "pages": got, "complete": complete, "opened": queue,
      "defer": defer})
  return True

 def walk_floor(self, floor, repair):
  """Sets the step floor a price-filter walk must stay above.
        A repair walk adds two steps; before any order the floor rises to the order reserve plus two."""
  if repair:
   return floor + 2
  if not self.run.orders_tried:
   return max(floor, self.run.intake.order_due() + 2)
  return floor

 def finish_listing(self, entry, args, paging, calls, floor):
  """Pages one price-filter listing to its end while the call count stays under CEIL_CALLS.
        Stops early when may_page fails or a page adds nothing. Returns (calls used, no next page left)."""
  watch = self.run.watch
  while watch.next_page(entry) is not None and calls < CEIL_CALLS and self.may_page(floor):
   calls += 1
   following = watch.next_page(entry)
   if not self.run.walk.walk_call({**args, paging[0]: following}, "Reading that price-filter listing's next page.") \
     or watch.next_page(entry) == following:
    break
  return calls, watch.next_page(entry) is None

 def lower_limit(self, entry, bound, total, calls):
  """Chooses a lower price bound when a price-filter listing has more matches than can be read.
        Takes the smaller of the midpoint between the cheapest row seen and the bound, and a cut sized
        from how many rows the remaining calls can read against the total."""
  prices = [float(c) for c in (self.archive.price(i) for i in (entry.get("row_shelf") or {})) if is_number(c)]
  low = min(prices) if prices else 0.0
  fill = max(1, CEIL_CALLS - calls) * max(1, entry["size"])
  return min(self.midpoint(low, bound), self.midpoint(low, low + (bound - low) * 1.8 * fill / max(total, 1)))

 def ceiling_walk(self, k, floor, repair=False):
  """Walks the store's price filter for product k below its aim price, looking for cheaper fits.
        Starts just under the aim, lowers the bound while a listing holds more than the calls can read,
        bisects back up after an empty one, and reads a small one out; rows over the bound mark the mode
        broken. Called by CheapWalk.run_walk; returns walk_report's (text, ids) or None."""
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
  say = f"Cheapest wanted: using the store's price filter to list '{label}' priced below {what} ({aim:g})."
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
  """Summarises a price-filter walk for product k and ranks the rows to open.
        Candidates are unread, in-stock, in-budget rows below the aim: value-led rows first, then rows whose
        text carries a required value, then the rest (dropped once a value search is complete).
        Returns (report text, up to CEIL_OPENS ids) and logs a ceiling_walk entry."""
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
   report = f"the price filter on '{label}' under {aim:g}: nothing in it is cheaper than {what}"
  elif complete:
   report = (f"the price filter on '{label}': seen every listing priced up to {last:g} ({len(ids)}, of which {len(cands)} are unread, "
       "in stock and within the budget)")
  else:
   report = f"the price filter on '{label}' under {aim:g}: seen {len(ids)} of the cheaper listings ({len(cands)} unread, in stock, within budget)"
  self.run.log.append({"kind": "ceiling_walk", "product": k, "label": label, "aim": aim, "calls": calls, "complete": complete,
        "rows": len(ids), "ranked": ranked[:CEIL_OPENS]})
  return report, ranked[:CEIL_OPENS]

 def ceiling_sweep(self, tried):
  """Proposes the next price-filter listing for Observer.sweep_next on a single-product request.
        Returns (call or None, finished): first the shelf listed just below its cheapest known row, then
        that listing's next pages; finished once it is fully read or came back empty with nothing cheaper."""
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
  """Returns the single unread, uncarted listing id named in a details call, or None.
        Also None when the code's checks named that id or any reference is in play."""
  run = self.run
  named = [v for v in call_args.values() if isinstance(v, str) and v in self.archive.ids]
  if len(named) != 1 or named[0] in self.archive.read or named[0] in run.trolley.units:
   return None
  ident = named[0]
  if ident in run.intake.named_open() or run.refs or self.needs.reading.get("references"):
   return None
  return ident

 def value_guard(self, call_name, call_args):
  """Holds back an actor details call on a listing outside the value-led rows.
        In all-words cheapest stores, when unread value-led rows cost less than both the target and pick,
        returns a no-step note naming up to six of them, once per product and needs version; else ''.
        Called from ShoppingRun.screen for unscripted calls; skipped in the order reserve."""
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
   return ("Not run (no step used). Since the cheapest is requested and this store's keyword search requires all words, "
     f"the rows shown by its searches led by a required value contain those words in some field; {ident} is not among "
     "them, whereas these rows from those searches cost less" + (" than it and the pick" if pick else " than it") + " and are unread: "
     + ", ".join(f"{i} ({c:g})" for c, i in rows) + ". Open these first, cheapest first; if you still want this call, "
     "send it again.")
  except Exception as err:
   run.log.append({"kind": "value_guard_error", "error": f"{type(err).__name__}: {err}"[:300]})
   return ""

 def title_line(self, k, rec, template):
  """Builds a search from a pick's title words that are not requirement, label or kind words.
        Used by CheapWalk.line_query when the record names no brand: the first free word of 3+ letters
        plus the next word if free. None when that query already ran or no word qualifies."""
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
   """Tells whether a title word may go into the line query:
            the word and its stem must both be absent from the avoided words."""
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
  """Re-runs a price-filter walk when the listing a walk aimed below has died.
        For a product whose aimed listing is now sold out or refused and whose pick costs more than the
        old aim, clears that product's aims and calls run_walk in repair mode. Returns its result, or None."""
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
  """Runs a shortened keyword-page walk when steps are too tight for the full plan.
        Returns False when the full walk_plan (its pages plus one per product with a pick) fits the steps;
        otherwise re-plans within the steps left after the order reserve and opens, and runs it."""
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
  """Scripted step for cheapest requests that looks below the current pick.
        In order: decline reads, a queued walk, a dead-aim rewalk; then, before any order, a price-filter
        walk for the readiest product, a seed test of the price filter, or reserve_walk.
        Run by CheapWalkStep; True when it used a step."""
  run = self.run
  walk = run.walk
  try:
   if self.v73_page:
    args, self.v73_page = self.v73_page, None
    walk.walk_call(args, "Cheapest wanted: reading the next page of the search that carries every known requirement.")
    return True
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
  """Finds the search argument that takes a category label, and the value to send in it.
        An enum entry equal to the label (case-insensitive) is used as spelled; a plain string field takes
        the label. Keys in skip are ignored; (None, None) when nothing fits."""
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
  """Finds the numeric search argument that looks like an upper price bound.
        Matches names such as max, price, budget or upper; skips enums, the page key and names about
        page, offset, count, size or number."""
  for key, spec in props.items():
   if not isinstance(spec, dict) or spec.get("type") not in ("integer", "number") or spec.get("enum") or key == page_key:
    continue
   if re.search(r"ceil|max|price|cap$|budget|upper|most", str(key).lower()) \
     and not re.search(r"page|first|offset|skip|count|size|num", str(key).lower()):
    return key
  return None

 @staticmethod
 def paired_modes(props, shelf_key, sentences):
  """Chooses values for small enum arguments that the tool description pairs with the shelf key.
        For each enum of two to four options, takes the first option named in a sentence that also names
        the shelf key; None if any such enum has no paired option."""
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
  """Builds a first price-filter call from the search schema and description alone.
        Combines the shelf argument, the price bound and description-paired modes; returns the
        ArgCheck-checked arguments, or None when a part or a required field is missing."""
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
  """Builds the broken-mode signature of seeded price-filter arguments.
        JSON of the enum mode values other than the label plus the first numeric non-page key,
        in the same shape as the signatures learn_filter marks broken."""
  mode = dict(self.run.watch.search_keys(args)[1])
  page_key = paging[0] if paging else None
  fixed_modes = sorted((m, v) for m, v in mode.items() if not (isinstance(args.get(m), str) and args[m] == label))
  number_key = next((a for a in args if is_number(args[a]) and a != page_key), "")
  return json.dumps([fixed_modes, number_key], default=str)

 def seed_price_filter(self):
  """Tests the price filter once per product by listing its category priced just under the pick.
        Runs only under the cheapest rule while no price mode is known; skips seeds whose mode is broken
        or whose arguments were already sent. search_note learns from the result. True if sent."""
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
    walk.walk_call(args, f"Cheapest wanted: testing whether this store's price filter holds by listing '{label}' "
          f"below the pick ({pick[1]:g}).")
    run.log.append({"kind": "ceil_seed", "product": k, "args": args, "learned": self.price_spec() is not None})
    if self.last_is_tool() and self.price_spec() is None:
     self.tell_tool(run.status_line())
    return True
   return False
  except Exception as err:
   run.log.append({"kind": "ceil_seed_error", "error": f"{type(err).__name__}: {err}"[:300]})
   return False

 def product_unfound(self):
  """Tells whether a multi-item request still has a product to find before walking below picks.
        True when some product's home shelf has no read listing, or when fits passed for the current
        needs but intake still reports a product unverified."""
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
  """Picks the products in a cart call that can get a price-filter walk below the pick.
        Each one costs CEIL_CALLS of the room and needs at least 2 left; the lowest carted price becomes
        that product's aim in put_aims. Returns (products to walk, room left)."""
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
  """Holds a cart call so the code can first look for cheaper fits below the pick.
        Plans page reads and price-filter walks within half the spare steps, stores them as queued_walk
        for walk_step, and returns a no-step note; '' when nothing is planned. Called from screen_cart
        with the call's arguments and from walk_gate with None (the whole cart)."""
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
    parts.append("uses the store's price filter to list the category's listings priced below the pick")
   return ("Not run yet (no step used). Since the cheapest is requested and this store returns search results by a fixed id "
     "order, not by price, a cheaper fit could be on pages not yet read: the code " + " and ".join(parts)
     + " for you now, then opens the lowest-priced unread rows below the pick. After that, repeat the call, or make "
     "it with a cheaper fit.")
  except Exception as err:
   run.log.append({"kind": "walk_hold_error", "error": f"{type(err).__name__}: {err}"[:300]})
   return ""

 @staticmethod
 def result_rows(piece):
  """Finds the list of row dicts in a result dict: the first non-empty one, else an empty one.
        None when no value of the dict is a list of dicts."""
  lists = [val for val in piece.values() if isinstance(val, list) and all(isinstance(r, dict) for r in val)]
  return next((val for val in lists if val), lists[0] if lists else None)

 def unwrap_rows(self, obs):
  """Locates the rows of a search result, looking one level into a nested dict if needed.
        Returns (dict holding the rows, rows), or (None, None) when no row list exists."""
  rows = self.result_rows(obs)
  if rows is not None:
   return obs, rows
  inner = next((val for val in obs.values() if isinstance(val, dict) and self.result_rows(val) is not None), None)
  if inner is None:
   return None, None
  return inner, self.result_rows(inner) or []

 def score_rows(self, call_args, rows):
  """Records for each returned row the highest value score among the queries that showed it.
        The score counts requirement values the query's words carry; stored in row_score by row id."""
  watch = self.run.watch
  score = watch.value_score(call_args) if rows else 0
  if not score:
   return
  for row in rows:
   if isinstance(row, dict):
    mark = watch.row_mark(row)
    self.row_score[mark] = max(self.row_score.get(mark, 0), score)

 def search_note(self, call_args, obs):
  """Annotates a search result for the actor and updates the search bookkeeping.
        Books the page, learns the price filter, scores rows, reads the word and lead-word rules, and adds
        remarks on re-spelled queries, hits, unseen totals and ignored price limits. Called from
        ShoppingRun.absorb for every search-tool result."""
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
  """Warns when rows come back priced above the numeric limit the call set.
        Takes the largest non-paging number above 1 as the limit; '' when there is none or no row is over."""
  paging = self.run.watch.search_keys(call_args)[3]
  limits = [val for key, val in call_args.items() if is_number(val) and val > 1 and key not in paging
      and not re.search(r"page|size|count|result|min|low|from|start", str(key).lower())]
  if not limits:
   return ""
  cap = max(limits)
  over = [r for r in rows if any(is_number(val) and val > cap and "price" in str(key).lower() for key, val in r.items())]
  if not over:
   return ""
  return (f"{len(over)} of these rows cost more than the {cap} limit this call set, so the search mode disregarded that "
    "filter (and may disregard others, the category for one). Use a mode whose filters hold, or check each row yourself.")

 def lead_remark(self, call_args, lead):
  """Tells the actor that the keyword search reads only the first few words of a query.
        Shows which words the current query kept when it had more, and how to order query words."""
  context = self.run.lead_ctx
  texts, mode, _, _ = self.run.watch.search_keys(call_args)
  words = next((w for key, w in texts if key == context[0]), ()) if mode == context[1] else ()
  counted = f" (this query counted only: {' '.join(words[:lead])})" if len(words) > lead else ""
  return (f"This store's keyword search ({context[0]}) only uses the FIRST {lead} word(s) in a query, ignoring "
    f"what follows{counted}: lead with the {lead} most distinctive words (a required value plus the item kind, "
    "value first), and look up other values in short queries of their own.")

 def hint_remarks(self, rows, raw, remarks):
  """Adds query-wording hints to a search's remarks.
        Warns once when only one-character words found nothing, at most twice when a query leads with a
        property name (offering a value-first query), else at most twice about a word trap.
        No property or trap hint near the order reserve."""
  run = self.run
  pieces = [p for t in raw for p in re.findall(r"[a-z0-9]+", t.lower())]
  if pieces and all(len(p) == 1 for p in pieces) and not rows and not self.short_hint_given:
   self.short_hint_given = True
   remarks.append("No results: this store's search ignores one-character words, so figures such as 0.4 or 5 x 7 find nothing "
         "(or match only their other words). Search a value written as a word, or the unit word plus the item kind, and "
         "read all of its pages.")
  prop, probe = run.watch.value_probe(raw)
  hints = self.name_hints
  if probe and hints < 2 and run.moves_left() > run.intake.order_due() + 3:
   self.name_hints = hints + 1
   if hints:
    opening = f"Again led by a property's name ({prop})."
   else:
    opening = (f"The query starts with a property's NAME ({prop}) and holds no required value. Keyword search covers listing "
         "values, titles and categories, and a field's name only if a value or title restates it, so "
         f"it misses listings whose {prop} field contains only the value.")
   remarks.append(opening + f" Lead with the value, then the item kind: '{probe}' (figures the store cannot read are dropped, "
         "keeping their unit word), and read all of its pages.")
  elif not probe and self.trap_hints < 2 and run.moves_left() > run.intake.order_due() + 3:
   trap, _ = run.walk.word_trap(raw)
   if trap:
    self.trap_hints += 1
    remarks.append(trap)

 def query_remarks(self, call_args, rows, raw, totals, mode, paging_keys, texts, kind, remarks):
  """Adds remarks for a first-page keyword search with free text.
        Flags a re-spelling that returned the same rows as an earlier query, books row hits when the
        result is fully shown, and once per run collects hits from earlier fully shown searches."""
  run = self.run
  self.hint_remarks(rows, raw, remarks)
  marks = tuple(run.watch.row_mark(r) for r in rows)
  others = tuple(sorted((k2, str(v2)) for k2, v2 in call_args.items()
         if k2 not in paging_keys and not (isinstance(v2, str) and k2 not in dict(mode))))
  sig = (mode, others, max(totals or [len(rows)]), marks)
  query = " ".join(raw)
  earlier = self.page_signatures.get(sig)
  if len(set(marks)) >= 3 and sig in self.page_signatures and earlier != query:
   remarks.append(f"Your earlier search '{earlier}' had the same total and rows: the store treats both as one query, and a new "
         "spelling reveals nothing. Read its next page, or alter its first words.")
  self.page_signatures.setdefault(sig, query)
  hits = []
  told = bool((kind == "all" or run.lead_words) and not totals and rows)
  if told or (kind != "any" and not totals and rows):
   hits = self.book_hits(call_args, rows, raw, told, texts, remarks)
  if (kind == "all" or run.lead_words) and not self.back_hits_done:
   self.back_hits_done = True
   self.earlier_hits(call_args, paging_keys, hits, remarks)

 def book_hits(self, call_args, rows, raw, told, texts, remarks):
  """Records the rows of a fully shown search as hits, scored by the query's value score.
        In all-words stores with at most 12 rows, names up to six hits to open, plus rows of a seller the
        request points at; otherwise pointer rows found here are discarded. Returns the hits."""
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
   remarks.append("Every word is required and every match is listed here: each row holds, somewhere in its record, the words "
         "the store read, often in fields missing from its title. Before searching further, open the in-budget ones: " + ", ".join(hits[:6]) + ".")
  pointers = []
  if told:
   popped, run.ref_hits = run.ref_hits, []
   pointers = [i for i in popped if i not in hits][:6]
  if pointers:
   remarks.append("A value in the request refers to another listing, and these rows are sold by the shop or brand this "
         "search names: open them regardless of price, stock or category (a listing that supplies a value is read "
         "and not bought), then do what its notes say: " + ", ".join(pointers) + ".")
  return hits

 def earlier_hits(self, call_args, paging_keys, hits, remarks):
  """Adds hits from earlier fully shown searches whose query carried a required value.
        Scores them, appends them to all_hits and names up to six to open; runs once, when the store is
        first known to need every word."""
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
   remarks.append("This store's keyword search requires every word, so earlier searches shown in full listed only rows that "
         "contain every one of their words, often in fields missing from the titles. Before searching further, open the "
         "unread in-budget ones: " + ", ".join(back[:6]) + ".")

 @staticmethod
 def unseen_remark(kind, totals, rows):
  """Tells the actor that the search matched more listings than this page shows.
        Advises reading further pages first and how to narrow the search, worded for any-word, all-word
        or still unknown keyword behaviour."""
  if kind == "any":
   how = ("this store's keyword search takes ANY of the words and puts listings matching more of them higher, so "
       "adding words broadens it: the distinctive word or figure of a required value, searched alone, lists only the "
       "listings that carry it (often few enough to read all pages and compare prices); the item kind plus the "
       "values puts the closest matches on top; for the rest, rely on the category mode (or a price filter that holds).")
  elif kind == "all":
   how = ("this store's keyword search needs EVERY word, so adding exact words narrows it; the category and price filters "
       "with a lowered price limit do too.")
  else:
   how = ("then narrow it for the rest: if every word is required, add exact words; if any word matches (which a longer "
       "query matching more listings reveals), include the required values in the query to make fitting listings come first; "
       "a price filter that holds helps as well.")
  return (f"This search matched {max(totals)} listings, but this page shows just {len(rows)} and only a few pages are "
    "readable, so the listings that matter could be among those not yet seen. First read the next pages of the same "
    "search (as many as are readable); if the total exceeds those pages, " + how)

 def shopper_lines(self, call_name, obs, refused):
  """Pulls the shopper's words out of a tool result so the run can interpret them.
        For the message tool: the reply string, or long strings from a reply without an error. Elsewhere:
        long strings in string lists, plus any string of 4+ words when the call was refused."""
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
 """Per-run slots for listing reads the code has scheduled, with the text that introduces them.
    The open_* slots (queue, say, recheck, purpose, floor, tool) are filled by walks and checks and
    taken by Judge.open_named, which clears them. put_queue and put_tool keep just-added cart lines
    waiting for a stock re-read and the tool to read them with. One instance per run."""
 def __init__(self):
  """Sets every slot empty: no listings to open or re-read after a put, no tools,
        no announcement, purpose or step floor, and the recheck flag off."""
  self.open_queue = None
  self.open_say = ""
  self.open_recheck = False
  self.open_purpose = ""
  self.open_floor = None
  self.open_tool = None
  self.put_queue = None
  self.put_tool = None


class Intake:
 """Reads the shopper request once at the start and turns the reader's JSON into run state.
    Also holds the hint and question-menu helpers Dialogue uses to decide what to ask next, and the
    reserve-window rules that shrink the actor's tool list as the first order falls due.
    Keeps small memos: per-product field menus, hint alternatives, products already re-asked through
    a twin field, and per-batch counts of late detail and search reads."""
 def __init__(self, run):
  """Binds the owning ShoppingRun and sets every memo and counter to its empty start value.
        read_content and read_at are kept so CodeCloser can redo the reading later; steps_given keeps
        the original step cap across re-readings."""
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


 def read_request(self):
  """Builds the reader prompt (READ_REQUEST, one line per tool with description and arguments, then
        the request and any opening observation) and gets a reading from the models via ask_readers.
        Falls back to reading_by_code when no "roles" map comes back, then runs check_reading (guarded)
        and use_reading. Called once by ShoppingRun.open_chat."""
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
  """Cross-checks the reader's JSON against the literal request text and fixes it in place.
        price_cap, message_budget and step_budget stay only if written in the request, else one written
        figure from the quoted sentence or None; procedure steps need a matching request sentence.
        With no steps, rules go to steps_in_rules; sets run.procedures and run.pairs (fit_with indexes)."""
  run = self.run
  text = run.request
  figures = reading.get("figures") if isinstance(reading.get("figures"), dict) else {}
  for key in ("price_cap", "message_budget", "step_budget"):
   value = reading.get(key)
   if not is_number(value):
    continue
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
  reading["ask_first"] = [item for item in reading.get("ask_first") or [] if not any(
   words and len(words & set(plain_words(item))) >= 0.6 * max(1, len(set(plain_words(item)))) for words in quotes)]
  reading["procedures"] = steps
  run.procedures = steps
  names = [" ".join(plain_words(row.get("product"))) for row in reading.get("items") or [] if isinstance(row, dict)]

  def slot_of(name):
   """Maps a product name to the index of the single reading item whose plain-word name equals,
            contains or is contained in it; None when no item or several items match."""
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
  """Asks the LISTENER seat (STEP_CHECK prompt over the numbered rules) which rules call for a
        message of the assistant's own, for example getting approval before the order.
        Each picked index whose rule matches a request sentence becomes a procedure step with needs_yes
        and final_cart set; logged as rule_steps."""
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
  """Asks the READER models and then the first LISTENER model, one model per call, for up to three
        rounds, and stops at the first parsed reply that has a "roles" map.
        From the second round on it stops early once readers_gone says retrying is pointless.
        Returns the last parsed object, which may be empty."""
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
  """Tells whether another pass over the reader models is not worth it: the last model call did
        answer (no misses), it was refused, the models are down or unusable, or too little time is left."""
  wire = self.run.wire
  return (wire.misses == 0 or wire.call_fail == "refused" or wire.down()
    or not wire.usable(ladder) or wire.spare() < CLOSING + 60)

 def reading_by_code(self, partial):
  """Builds a reading from CodeCloser.code_reading when no model gave a usable one; non-roles keys
        of the partial model reading override the code's values, and an error yields an empty reading.
        Stamps read_at, which lets CodeCloser.catch_up try a model re-read later, and logs a summary."""
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
  """Applies a reading to the run; again=True marks the re-read from CodeCloser.catch_up.
        Maps roles to real tool names, rebuilds the CartBook unless a used cart is kept, sets msg_cap,
        needs.reading, the spend scale and price cap; runs split_requirements, story_values, ref_register.
        Asks for a budget first when no cap is written, applies step_budget; a re-read bumps the version."""
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
   reading["ask_first"] = ["the most the shopper would pay, as the request names no price limit (skip this when the list already repeats that point)"] + ask_more
  if again:
   run.step_cap = run.step_cap if self.steps_given is None else self.steps_given
  allowed = reading.get("step_budget")
  if is_number(allowed) and 0 < allowed < run.step_cap:
   run.step_cap = int(allowed)
  if again:
   needs.version += 1

 def split_requirements(self, reading):
  """Keeps only requirements shaped "property: value" in needs.written and in the reading.
        Malformed ones are not lost: they are appended to soft_wants instead."""
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
  """Runs a callable and swallows any exception, logging it to the wire log under the given kind.
        Keeps optional setup steps (story values, reference register) from aborting the reading."""
  try:
   action()
  except Exception as err:
   self.run.wire.log.append({"kind": kind, "error": f"{type(err).__name__}: {err}"[:300]})

 @staticmethod
 def hint_list(raw):
  """Normalises the reader's hints into a list of non-empty strings.
        A dict of property to value(s) is flattened to "property: value" lines (a value that already has
        a colon is kept as is); input that is neither a list nor a dict gives an empty list."""
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
  """Splits a requirement string into (property, value), both lowercased, with any "(product)" tag
        cut from the property; used to match requirements across the per-item rows."""
  head, _, value = str(req).partition(":")
  return re.split(r"\s*\(", head)[0].strip().lower(), value.strip().lower()

 def story_values(self, reading):
  """Moves requirements whose value the request never writes (story_guess) out of needs.written and
        the item rows into hints that say the story only suggests the value; this happens only with two
        or more requirements and when at most half of them are guessed.
        Then written_hints may promote reader hints the other way; the hint list is capped at 12."""
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
    hints.append(f"{prop.strip()}: the story hints at {value.strip()} (no value stated; ask)")
   run.wire.log.append({"kind": "story_values", "moved": guessed})
  try:
   promoted = self.written_hints(reading, reader_hints)
   hints = [h for h in hints if h not in promoted]
  except Exception as err:
   run.wire.log.append({"kind": "written_hint_error", "error": f"{type(err).__name__}: {err}"[:300]})
  reading["hints"] = hints[:12]

 def hint_word(self, detail, names):
  """Finds the hint's leading value word that the request writes right before a product's kind word
        (lead_value), trying each product name in turn; returns (word, name), word empty if none fits."""
  word, name = "", ""
  for name in names:
   word = lead_value(detail, self.run.request, name)
   if word:
    break
  return word, name

 def clashes(self, prop, word, mine):
  """True when a written requirement for the same product already covers this property or already
        has the word in its value; requirements tagged for another product (no shared kind word) are
        skipped. A clash stops written_hints from promoting the hint."""
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
  """Promotes reader hints whose value the request states as a modifier of the product word into firm
        "property: word" requirements, unless clashes() finds the property or word already covered.
        Each one is appended to needs.written, the reading and its item row; returns the promoted hints."""
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


 def several(self):
  """True when the reading lists two or more products; Needs.items is non-empty only in that case.
        Most helpers below key their per-product logic on it."""
  return bool(self.run.needs.items())

 def slot_of(self, text):
  """Product index a property text belongs to (Dialogue.product_of) in multi-product runs.
        Always None for a single product."""
  return self.run.talk.product_of(text) if self.several() else None

 def open_hints(self, k=None):
  """Lists the reader hints still worth a question, optionally only those of product k.
        A hint stays if its property is still open (Dialogue.open_prop with menu samples) and no answered
        question covered it; otherwise hint_alt may offer it again under an alternative field.
        Reworded hints come first, then the fresh ones."""
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
  """Builds the field menu text (Records.prop_menu) for product k from the category labels read.
        Single product: its own label when known, else all records; several products: the first read
        label among product k's home labels, or "" when none of them was read yet."""
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
  """Returns the menu line of product k whose field name strictly matches prop, or "".
        The menu is memoised per product and rebuilt when the record count changes; errors give ""."""
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
  """Field name of one menu line: the text with the leading dash and the "(in N listings)" count cut.
        Static; used wherever menu lines are compared with property names."""
  return line.strip("- ").split(" (in ")[0].strip()

 def cached_menu(self, k):
  """Last memoised menu text for product k, without rebuilding it.
        Returns "" when no menu was built for that product yet."""
  return (self.menu_memo.get(k) or (0, ""))[1]

 @staticmethod
 def plain_words(value):
  """Lowercases a value and keeps only its runs of letters and digits, joined by single spaces.
        Static; used to compare menu sample values loosely."""
  return " ".join(re.findall(r"[a-z0-9]+", value.lower()))

 def field_values(self, k, prop):
  """Set of sample values, normalised by plain_words, shown on prop's menu line for product k.
        Empty when the property has no menu line."""
  line = self.prop_line(k, prop)
  if "):" not in line:
   return set()
  return {self.plain_words(v) for v in line.split("):", 1)[1].split("|")} - {""}

 def same_field(self, k, a, b):
  """True when two property names denote the same field: a strict name match, or loosely similar
        names (same_prop) whose menu sample values overlap for product k. False on error."""
  if strict_prop(a, b):
   return True
  try:
   if not same_prop(a, b):
    return False
   return bool(self.field_values(k, a) & self.field_values(k, b))
  except Exception:
   return False

 def hinted_prop(self, k, prop):
  """True when a reader hint's property strictly matches prop and, in multi-product runs, that hint
        belongs to product k or to no product in particular."""
  name = str(prop).split("(")[0].strip()
  for hint in self.run.needs.reading.get("hints") or []:
   head = str(hint).split(":", 1)[0]
   if not strict_prop(head, name):
    continue
   if k is None or not self.several() or self.run.talk.product_of(head) in (k, None):
    return True
  return False

 def product_reads(self, k):
  """Counts listings with rich records read (Records.rich_reads) for product k: on its own label for a
        single product, summed over product k's home labels otherwise; 0 on error or with no home label."""
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
  """True when a hinted property seems to have no field on this store's listings: the product has two
        or more rich reads, prop has no menu line, and no menu key is even loosely the same name.
        hint_mark tags such hints, and Dialogue does not plan questions on them."""
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
  """Returns the hint with a "[no key]" tag when plain_hint finds no store field for its property, or
        the hint unchanged. Used when hints are listed in the question-planning prompts."""
  try:
   prop = str(hint).split(":", 1)[0].strip()
   marked = self.plain_hint(self.slot_of(prop), prop, hinted=True)
   return f"{hint} [no key]" if marked else str(hint)
  except Exception:
   return str(hint)

 @staticmethod
 def hint_alts_of(detail):
  """Collects the alternative field names a hint lists in "(or ...)" groups, split on commas, "or"
        and slashes. Such groups come from one_prop_hint rewriting hint heads like "a/b"."""
  found = []
  for group in re.findall(r"\(or ([^()]+)\)", str(detail)):
   found.extend(piece.strip() for piece in re.split(r",|\bor\b|/", group) if piece.strip())
  return found

 @staticmethod
 def long_words(name):
  """Set of the words of a property name that have three or more letters and are not link words.
        Static; the basis for the name containment and twin tests."""
  return {w for w in prop_words(name) if len(w) >= 3 and w not in LINK_WORDS}

 @staticmethod
 def name_inside(a, b):
  """True when one name's long-word set is a subset of the other's.
        False when either name has no long words."""
  left, right = Intake.long_words(a), Intake.long_words(b)
  if not left or not right:
   return False
  return left <= right or right <= left

 @staticmethod
 def asked_text(line):
  """Regex match on a questions_log line; group 1 is the question text that was actually sent.
        None when the line holds no quoted question."""
  return re.search(r'asked "(.*?)" -> ', line)

 def asked_inside(self, k, head):
  """True when an answered questions_log entry asked about a property whose name contains head or is
        contained in it, for product k; the product comes from the label part, else from the question text.
        Entries marked as not answered are skipped."""
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
  """True when a question already sent on head (same field, or name containment) drew a reply that
        revealed nothing (Dialogue.quiet_asks), for product k in multi-product runs."""
  several = self.several()
  for asked in self.run.talk.quiet_asks or []:
   if not (self.same_field(k, asked, head) or self.name_inside(asked, head)):
    continue
   if not several or self.run.talk.product_of(asked) in (k, None):
    return True
  return False

 def hint_alt(self, k, hint, head):
  """Rewords a hint whose first property drew no preference so that it targets an alternative field.
        Takes the first "(or ...)" name that has a menu line (memoised in hint_alts) and returns the hint
        relabelled with it, or "" when there is none, it was spent, or it is no longer open to ask."""
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
   return f"{label}: {detail} (the first property, {head}, drew no preference)"
  except Exception:
   return ""

 def hint_mapped(self, k, prop):
  """True when some hint for product k points at prop: its head or one of its listed alternatives
        strictly matches prop, or the hint text contains every long word of prop."""
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
  """True when two different property names share a word and at least one side of that pair is a
        leading word (not the last word) of its name; words match when equal or one prefixes the other.
        Names that are strictly the same property are never twins."""
  if strict_prop(a, b):
   return False

  def split(name):
   """Returns the long words of a name and its leading words: all but the last, or the only word.
            name_twin needs a leading word on at least one side of a match."""
   every = [w for w in prop_words(name) if len(w) >= 3 and w not in LINK_WORDS]
   leading = every[:-1] if len(every) > 1 else every
   return every, leading

  def close(x, y):
   """True when two words are equal, or both have four or more letters and one starts the other.
            Short words must match exactly."""
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
  """True when two names that are not the same property are still related: name_twin, or same_field
        for product k. False on error."""
  try:
   if strict_prop(a, b):
    return False
   return self.name_twin(a, b) or self.same_field(k, a, b)
  except Exception:
   return False

 def story_text(self, k):
  """Lowercased request text joined with the hints of product k (every hint for a single product).
        twin_key searches it for the extra words of a candidate field."""
  run = self.run
  several = self.several()
  pieces = [run.request.lower()]
  for hint in run.needs.reading.get("hints") or []:
   if not several or run.talk.product_of(str(hint).split(":", 1)[0]) in (k, None):
    pieces.append(str(hint).lower())
  return " ".join(pieces)

 def product_name(self, k, fallback=""):
  """Name of product k in multi-product runs; otherwise the reading's product, or the fallback when
        the reading names none."""
  needs = self.run.needs
  if self.several() and k is not None:
   return needs.items()[k]["product"]
  return str(needs.reading.get("product") or fallback)

 def twin_key(self, k, prop):
  """Finds another menu field of product k to ask in place of prop: a twin_of field that is not spent,
        still open, not containing or contained in prop, whose own extra words appear in the story or which
        a hint points at. Returns its menu key or ""."""
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
  """Builds a choice question for prop from two or three of its menu sample values (short, with
        letters, no separators, not shared with other fields' samples); "" when fewer than two qualify."""
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
  """Resolves the product slot of prop for a re-ask, plus its twin_asks mark (-1 for one product).
        Returns None when the product cannot be told or that product was already re-asked once."""
  multi = self.several()
  slot = self.run.talk.product_of(prop) if multi else None
  if multi and slot is None:
   return None
  mark = slot if multi else -1
  if mark in self.twin_asks:
   return None
  return slot, mark

 def labelled(self, key, slot):
  """Adds the product name in parentheses to a field key in multi-product runs.
        Returns the key unchanged for a single product."""
  if not self.several():
   return key
  return f"{key} ({self.run.needs.items()[slot]['product']})"

 def distinct_ask(self, prop):
  """Builds a different re-ask for an unanswered property: a values_form question on it, or else on
        a twin_key field (a plain which-question when it has no usable values).
        Returns (label, question) and marks the product in twin_asks, or None. Used by Dialogue."""
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
  """True when a questions_log entry, answered or not and with its tags stripped, has a label part
        (split on ";" and "and") that strictly names prop for product k."""
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
  """Returns the menu key for alternative name alt when it is a real field of product k that differs
        from head and name, is not spent, was not asked yet and is still open to ask; "" otherwise."""
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
  """Scans hints whose property is prop (same field, or name containment) for product k and returns
        the first listed alternative that alt_usable accepts; "" when none does."""
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
  """For a lone unanswered question, picks a hint alternative field (alt_key) to ask next instead.
        Marks the product in twin_asks and returns the labelled key, or "". Called by Dialogue."""
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
  """True when every content word of the plain question (stemmed with Observer.stem, filler and link
        words left out) also occurs in text, i.e. the question sent already asked it plainly."""
  stem = self.run.watch.stem
  filler = {"which", "would", "you", "like", "for", "the"}
  present = {stem(w) for w in re.findall(r"[a-z0-9]+", str(text).lower())}
  wanted = [stem(w) for w in re.findall(r"[a-z0-9]+", str(plain_q).lower()) if w not in filler and w not in LINK_WORDS]
  return bool(wanted) and all(w in present for w in wanted)

 def alt_untried(self, k):
  """True when some hint for product k was already asked and still has a usable alternative field
        (alt_key) left to try. Dialogue uses it when ranking products for the next question."""
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
  """True when some product other than k has not been the subject of any questions_log entry yet;
        the product is read from each entry's label and from its question text."""
  talk = self.run.talk
  log = self.run.questions_log
  reached = {talk.product_of(line.split(":", 1)[0]) for line in log}
  for line in log:
   said = self.asked_text(line)
   if said:
    reached.add(talk.product_of(said.group(1)))
  return any(j != k and j not in reached for j in range(len(self.run.needs.items())))

 def quiet_reask(self, prop):
  """After a question on prop revealed nothing, offers a story twin field (twin_key) to ask next,
        labelled with its product. Only when a hint maps to prop, no reworded hint is pending and, with
        several products, every other product was asked already; marks twin_asks. Called by Dialogue."""
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
  """Once four or more rich records were read, drops menu lines for fields seen in at most one
        listing; keeps the menu whole if trimming would empty it."""
  if reads < 4 or not menu:
   return menu
  kept = [line for line in menu.splitlines() if not re.search(r"\(in [01] listings?\)", line)]
  return "\n".join(kept) if kept else menu

 def first_reads(self):
  """Queues, once per run, a few detail reads before any question so the field menus can be built.
        Only with a details tool and no batch one: up to two listings per product short of two rich reads
        (Catalogue.menu_rows), interleaved and fitted to the steps not kept for the order and questions.
        Sets run.reads with purpose "menu" and returns True when something was queued."""
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
   reads.open_say = "Before asking anything, I open a few listings of each requested product to learn their fields."
   reads.open_purpose = "menu"
   run.log.append({"kind": "first_reads", "ids": queue})
   return True
  except Exception as err:
   run.log.append({"kind": "first_reads_error", "error": f"{type(err).__name__}: {err}"[:300]})
   return False

 def plan_open(self, k, prop):
  """Whether prop is still open to ask for product k, via Dialogue.open_prop; for a hinted property
        it also passes the menu line so its sample values can settle the question."""
  talk = self.run.talk
  try:
   hinted = self.hinted_prop(k, prop)
   line = self.prop_line(k, prop) if hinted else ""
   return talk.open_prop(k, prop, line, by_samples=hinted)
  except Exception:
   return talk.open_prop(k, prop)

 def settles(self, req, samples=""):
  """Decides whether a known requirement already settles a property when planning questions.
        References always do and non-positive values never; values not written in the request or the
        exchanges (story_guess) do not; numbers, or any value when no samples are given, do.
        Otherwise a value word must appear in the menu samples, allowing one edit. Used by _settled_by."""
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


 def eventful(self):
  """True once the store produced an event such as a refusal or sell-out, or a cart line is known
        to be sold out; order_due then keeps more steps in reserve."""
  run = self.run
  if run.event_seen:
   return True
  dead = run.shelf.dead
  return bool(dead and set(run.trolley.units) & dead)

 def order_due(self):
  """Steps left at which the first order should be under way, derived from the step cap: at least
        FIRST_ORDER_BY, about a third of the cap (at most 16), capped at half the cap but not below that
        floor, and raised to 8 plus the product count after a store event. Gates compare moves_left() to it."""
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
  """True when the store has both an order and a cart tool, the precondition for narrowing the
        actor's tools in the reserve window and for the related status hints."""
  return bool(self.run.role("order") and self.run.role("cart"))

 def reserve_names(self, spare):
  """Tool names allowed with spare steps left, or None while no narrowing applies yet.
        One step: order and decline; two or three: plus cart. In the late window (first order due, or a
        refusal with FIRST_ORDER_BY or fewer left): cart, cart view, details or view and the stock tool,
        plus details or search only for limited late reads, the sell-out swap or cheaper backup rows."""
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
  """Tool specs the actor may use now: the reserve_names subset when the store has order and cart
        tools (the whole list if that subset is empty), minus the message tool while the code asks.
        dispatch permits only these, and status_line lists them when the list is narrowed."""
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
  """True when the store has order and cart tools, more than three steps remain, and either no order
        was tried and order_due() or fewer remain, or a refusal came with FIRST_ORDER_BY or fewer left."""
  run = self.run
  spare = run.moves_left()
  if not (run.role("order") and run.role("cart")) or spare <= 3:
   return False
  if spare <= self.order_due() and run.orders_tried == 0:
   return True
  return spare <= FIRST_ORDER_BY and run.refusal_n > 0

 def unverified(self):
  """True when no listing has passed the item check under the current needs version (refused and
        sold-out lines aside); with several products also when some product's home labels hold no
        passed listing."""
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
  """Listings a check named (Judge.named) that are in the records but not yet read or refused.
        Returned as a set; non-empty means late detail reads may still be allowed."""
  run = self.run
  seen = {i for i in run.judge.named if i in run.archive.ids}
  return seen - run.archive.read - run.refused_lines

 def widened_late(self):
  """True when the needs were widened in the current needs version and that happened within
        order_due() steps of the current world step."""
  run = self.run
  needs = run.needs
  return needs.widened_at == needs.version and run.wire.tick - needs.widened_tick <= self.order_due()

 def begin_batch(self):
  """Resets the per-batch count and ids of late reads already let through.
        dispatch calls it before screening each batch."""
  self.late_q = 0
  self.late_ids = set()

 def spare_backups(self):
  """Backup row ids (backup_rows) not yet let through in this batch, limited to the backup reads
        still left (two per run in all)."""
  left = 2 - self.backup_n
  return [i for i in self.backup_rows() if i not in self.late_ids][:left]

 def late_allowed(self, call_name, call_args, named, spare):
  """Decides whether a late details or search call fits the reserve rules.
        Details: while nothing is verified with more than five steps, or when it opens a check-named
        listing with enough steps; search: after a recent widening, on a page an audit pointed to,
        or while no fit is verified yet, with more than four steps."""
  run = self.run
  if call_name == run.role("details"):
   if self.unverified() and spare > 5:
    return True
   return bool(named & self.named_open()) and spare >= min(FIRST_ORDER_BY + 2, self.order_due())
  widened = self.widened_late() or json.dumps(call_args, sort_keys=True, default=str) in run.judge.named_pages
  return (widened or (self.unverified() and spare > 5)) and spare > 4

 def late_hold(self, call_name, call_args):
  """Returns a hold note for a details or search call in the reserve window, or "" to let it run.
        Other lookup tools count as details. Cart lines, passed listings, up to three late reads, two
        cheaper backup rows and the sell-out swap read pass; anything else is held with a note that
        keeps the steps for the order and may list backup rows. Called from screen for actor calls."""
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
  note = ("Not run (no step used): the steps left are kept for the order. Detail reads are limited to the cart's line, "
    "a listing named by a check, or (while nothing is verified) at most 3 listings in total: place your best "
    "verified fit in the cart, check its stock and price again, and order.")
  if extra:
   note += " Cheaper unread rows that could back up the pick are still readable: " + ", ".join(extra) + "."
  return note

 def backup_rows(self):
  """For cheapest requests after a store event, a sold-out line or an open check-named listing, lists
        up to six rows priced below each product's cheapest verified fit that are not read, gone, over
        budget, pointers or ruled out, cheapest first; empty once two backup reads were used."""
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
 """One episode of the agent: the request, the tool specs, the step cap and every helper component.
    Owns the Wire, the Records archive, Needs, the CartBook, the role map and the ledgers the helpers
    share. Screens each proposed tool call through argument, batch, cart and order or decline gates,
    sends the rest to the world and turns the results into text for the actor.
    execute() runs the turn loop of scripted code steps and actor turns."""
 def __init__(self, pdata):
  """Builds the run from the environment payload: binding, policy view, request (query), tools and
        their parameter schemas, max_steps (30 when absent) and the finished flag.
        Creates Records, Needs, an empty CartBook, the ledgers and all helpers, clears LITERAL_VALUES,
        and hooks Needs to ReferenceDesk.ref_fix and Judge.store_fields."""
  environment = pdata["environment"]
  self.wire = Wire(environment["binding"])
  self.pview = environment["policy_view"]
  self.request = str(self.pview.get("query") or "")
  self.toolset = [entry for entry in self.pview.get("tools") or []
      if isinstance(entry, dict) and isinstance(entry.get("function"), dict)]
  self.shapes = {}
  for entry in self.toolset:
   spec = entry["function"]
   self.shapes[spec["name"]] = capped_shape(spec.get("parameters") or {"type": "object", "properties": {}},
              listed_cap(spec.get("description")))
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
  """Sets every per-episode counter, flag and record to its start value: message and order counts,
        go-ahead state, fit pairs, one-shot gate flags, checked, passed and refused sets, cart read
        sequences, the questions log, the chat and the search-reading state. Called from __init__."""
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
  self.batch_cart = None
  self.batch_proven = None
  self.batch_spec = None
  self.batch_spec_known = False
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
  self.v75_opened = set()
  self.v75_open_marks = set()
  self.v75_fit_marks = set()
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
  self.word_seen = []
  self.word_kind = ""
  self.lead_seen = []
  self.lead_bounds = {}
  self.lead_words = 0
  self.lead_ctx = None

 @property
 def log(self):
  """The episode's event log list, kept on the Wire and shared by every component.
        Read-only property; callers append to the list it returns."""
  return self.wire.log

 def moves_left(self):
  """Steps still available: the step cap minus the world step counter (Wire.tick).
        The cap itself may have been lowered by a step budget from the request reading."""
  return self.step_cap - self.wire.tick

 def role(self, call_name):
  """Looks up the store tool bound to a role name (search, details, cart, order, decline, message,
        cart_view and so on) in role_map; None when the reading found no such tool."""
  return self.role_map.get(call_name)

 def ask_limit(self):
  """Messages the question planner may spend: the message budget (ASK_DEFAULT when the request sets
        none) minus the ones held back for go-ahead steps, never below zero."""
  return max(0, (ASK_DEFAULT if self.msg_cap is None else self.msg_cap) - self.held_back())

 def held_back(self):
  """Messages kept out of question planning for the request's procedure steps: one per step, capped
        by the budget; zero when there are no steps or a go-ahead was already sent."""
  if self.confirms or not self.procedures:
   return 0
  return min(len(self.procedures), ASK_DEFAULT if self.msg_cap is None else self.msg_cap)

 def cart_mark(self):
  """Hashable snapshot of the cart: sorted (line, units) pairs for lines with nonzero units.
        Used to tell whether the basket changed since a go-ahead or a hold."""
  return tuple(sorted((line, units) for line, units in self.trolley.units.items() if units))

 def basket_note(self):
  """Writes the go-ahead message: each cart line with units, title, id and unit price (from
        CodeCloser.code_rows), the total and whether it is within the price cap, then a yes/no question."""
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
  return "Final basket: " + "; ".join(lines) + "." + tail + " Shall I place this order? Please reply yes or no."

 def go_ahead(self, note, said):
  """Classifies the shopper's reply to the basket note with the LISTENER seat (GO_AHEAD prompt).
        Returns "yes", "no" or "change", and "yes" when the answer is missing or unexpected."""
  prompt = GO_AHEAD + "\n\nTHE ASSISTANT WROTE:\n" + note + "\n\nTHE SHOPPER REPLIED:\n" + said
  answer = self.wire.query_model([{"role": "user", "content": prompt}], LISTENER, 200)
  verdict = str((json_object((answer or {}).get("content") or "") or {}).get("answer") or "").lower()
  return verdict if verdict in ("yes", "no", "change") else "yes"

 def send_go_ahead(self, room=3):
  """Sends the basket note as a scripted message when the request has procedure steps, a message tool
        exists, the cart is non-empty and changed since the last go-ahead, and budget, steps and time allow.
        Classifies the reply (any change to requirements or price cap forces "change"), records confirms
        and go_verdict, and returns (note, reply, verdict), or None when nothing was sent."""
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
  """Text of every merged listing card (Records.merge_cards), keyed by listing id.
        pair_gate searches these texts for the anchor's model marks."""
  merged, _ = self.archive.merge_cards()
  return {ident: as_text(card) for ident, card in merged.items()}

 def pair_slot(self, line):
  """Product index whose home labels include the cart line's category label.
        None unless exactly one product matches."""
  label = str(self.archive.label_of.get(line) or "").strip().lower()
  homes = self.shelf.home_shelves()
  slots = [spot for spot, home in enumerate(homes) if label and label in home]
  return slots[0] if len(slots) == 1 else None

 def pair_marks(self, line, texts):
  """Model-like marks (id_marks) on an anchor line's card that point at that item alone.
        When at least four cards carry the same label, marks that appear on half of those cards or more
        are left out as too common."""
  label = self.archive.label_of.get(line)
  shelf = [body for ident, body in texts.items() if self.archive.label_of.get(ident) == label]
  return {mark for mark in id_marks(texts.get(line, ""))
    if len(shelf) < 4 or 2 * sum(1 for body in shelf if mark in body) < len(shelf)}

 def pair_gate(self):
  """Order gate for fit-with pairs (a product that has to work with another): holds the order when a
        dependent cart line's card names none of the anchor line's marks in its own sentences (fit_statement, v7.3: a
        mark only in a sentence about another item does not count; logs v73_p_foreign), once per line, anchor and cart
        state. Points to same-label listings that name a mark (cheapest first) or to a search for it.
        Skipped with three or fewer steps or under CLOSING seconds; logs pair_hold."""
  if not self.pairs or self.moves_left() <= 3 or self.wire.spare() <= CLOSING:
   return ""
  lines = [line for line, units in self.trolley.units.items() if units]
  by_slot: dict[int, list[str]] = {}
  for line in lines:
   spot = self.pair_slot(line)
   if spot is not None:
    by_slot.setdefault(spot, []).append(line)
  cards = self.archive.merge_cards()[0]
  texts = {ident: as_text(card) for ident, card in cards.items()}
  owns: dict[str, str] = {}

  def own(ident: str) -> str:
   """The card text of ident that speaks for that listing itself (fit_statement), kept per gate call."""
   if ident not in owns:
    owns[ident] = fit_statement(cards.get(ident) or {})
   return owns[ident]
  products = [str(row["product"]) for row in self.needs.items()]
  dead = self.shelf.dead_lines()
  for dep, anchor in self.pairs:
   for base in by_slot.get(anchor, []):
    marks = self.pair_marks(base, texts)
    if not marks:
     continue
    for line in by_slot.get(dep, []):
     if any(mark in own(line) for mark in marks):
      continue
     told = (line, base, self.cart_seq)
     if self.pair_told.get(told, 0) >= 1:
      continue
     self.pair_told[told] = 1
     foreign = any(mark in texts.get(line, "") for mark in marks)
     if foreign:
      self.log.append({"kind": "v73_p_foreign", "line": line, "anchor": base, "marks": sorted(marks)})
     label = self.archive.label_of.get(line)
     named = sorted((ident for ident, body in texts.items() if ident not in dead and ident != line
         and self.archive.label_of.get(ident) == label and any(mark in own(ident) for mark in marks)),
           key=lambda ident: self.archive.price(ident) if is_number(self.archive.price(ident)) else 1e18)
     dep_name = products[dep] if dep < len(products) else "item"
     anchor_name = products[anchor] if anchor < len(products) else "item"
     shown = ", ".join(sorted(marks)[:2])
     self.log.append({"kind": "pair_hold", "line": line, "anchor": base, "marks": sorted(marks), "named": named[:6]})
     return (f"Not run yet (no step used). The {dep_name} must work with the chosen {anchor_name}: the page of {base} itself "
       f"shows {shown}, but {line}'s page "
       f"{'names it only in a sentence about another item' if foreign else 'does not mention it'}. "
       + (f"Already-read {dep_name} listings naming it: {', '.join(named[:6])}. Replace {line} with the cheapest "
          "of these that meets every other requirement, then order." if named else
          f"Run '{sorted(marks)[0]}' through store search (it matches listing text), open the {dep_name} rows it lists, "
          f"and replace {line} with one whose page names it."))
  return ""

 def basket_short(self):
  """Basket faults for multi-product requests (CartBook.basket_faults against the needed counts), or
        an empty-cart fault; always an empty list for a single product."""
  counts = self.needs.counts()
  if len(counts) < 2:
   return []
  return self.trolley.basket_faults(counts) if self.trolley.units else ["the cart is empty"]

 def basket_gate(self):
  """Order gate that refuses any order while the multi-product basket is incomplete, naming the
        products with no line yet. With three or fewer steps it says to add the line or decline.
        Runs on every order attempt; logs basket_hold."""
  faults = self.basket_short()
  if not faults:
   return ""
  products = [str(row["product"]) for row in self.needs.items()]
  slots = {self.pair_slot(line) for line, units in self.trolley.units.items() if units}
  missing = [name for spot, name in enumerate(products) if spot not in slots]
  named = (" No line yet for: " + ", ".join(missing) + ".") if missing and len(missing) < len(products) else ""
  self.log.append({"kind": "basket_hold", "faults": faults, "missing": missing})
  if self.moves_left() <= 3:
   return ("Not run (no step used). An order lacking a product's line always fails: " + "; ".join(faults) + "." + named
     + " With the remaining steps, add the missing line now if you know a fitting listing; if not, decline.")
  return ("Not run yet (no step used). The cart's basket is incomplete: " + "; ".join(faults) + "." + named
    + " Find and add a fitting listing for every missing product (one line apiece, at its quantity), then order; "
    "decline only when some product has no listing able to meet its requirements.")

 def pair_text(self):
  """Status-line fragment that states the fit-with pairs and tells the actor to choose the anchor
        product first and match its model on the dependent listing's page; "" with no pairs."""
  if not self.pairs:
   return ""
  products = [str(row["product"]) for row in self.needs.items()]
  pairs = "; ".join(f"the {products[dep]} has to work with the {products[anchor]}" for dep, anchor in self.pairs
        if dep < len(products) and anchor < len(products))
  return ("\n[fit] " + pairs + ". Pick the " + "/".join(dict.fromkeys(products[a] for _, a in self.pairs if a < len(products)))
    + " first: its page states its own model name and number, which a fitting item's page lists among the models it "
    "works with (searching for that model finds them).") if pairs else ""

 def go_ahead_text(self):
  """Status-line fragment for requests with procedure steps and a message tool: either the shopper
        already let the current cart stand, or the code will ask for approval once the order tool is
        called, so the actor must neither ask for it nor decline over it."""
  if not self.procedures or not self.role("message"):
   return ""
  if self.go_verdict == "yes" and self.confirmed_cart == self.cart_mark():
   return ("\n[go-ahead] The shopper saw this basket and left it as it is, which is the go-ahead the request requires. "
     "Call the order tool now; leave the cart unchanged unless some line no longer fits.")
  return ("\n[go-ahead] The finished cart needs the shopper's approval before it is bought; the code does that step: "
    "when the final basket is in the cart, call the order tool; then the code first shows the shopper the basket using the "
    "message kept for it, reads the answer, and only then orders. Do not request the yes yourself; its absence is "
    "never a reason to decline or to empty the cart.")

 def go_ahead_decline_gate(self):
  """Decline gate for go-ahead requests: holds a decline once per cart and needs version while the cart
        has lines, the shopper has not said no and more than one step remains; points to the order tool."""
  mark = (self.cart_mark(), self.needs.version)
  if not self.procedures or self.go_verdict == "no" or not mark[0] or self.go_hold_mark == mark or self.moves_left() <= 1:
   return ""
  self.go_hold_mark = mark
  return ("Not run yet (no step used). No approval from the shopper is lacking: once you call the order tool, the code shows "
    "the shopper the cart, in the message kept for it, and buys only if they accept it as is. In the cart: "
    + ", ".join(f"{line} x{units:g}" for line, units in mark[0]) + ". Call the order tool on it; decline only if a line clearly fails a requirement now in force and the records "
    "offer nothing to replace it.")

 def go_ahead_gate(self):
  """Last order gate: sends the go-ahead message (send_go_ahead) when it is due.
        Lets the order pass when nothing was sent or the reply was yes; otherwise holds it with a note
        quoting the basket note and the shopper's reply."""
  sent = self.send_go_ahead()
  if not sent or sent[2] == "yes":
   return ""
  note, said, _ = sent
  return ("Not run yet (one step used to message the shopper). Since the request asks for the shopper's go-ahead on the final "
    f"basket, they were shown: \"{note}\" They replied: \"{said}\" Follow their answer; any changed basket goes to them again before the order.")

 def call_kind(self, call_name, call_args):
  """Classifies a call as "order", "decline" or None (not a finishing tool). When one tool serves
        both, short string arguments decide by DECLINE_WORDS, then ORDER_WORDS, defaulting to order."""
  order, decline = self.role("order"), self.role("decline")
  if not call_name or call_name not in (order, decline):
   return None
  if order != decline:
   return "order" if call_name == order else "decline"
  fields = (self.shapes.get(call_name) or {}).get("properties") or {}
  for field, piece in (call_args or {}).items():
   spec = fields.get(field) if isinstance(fields, dict) else None
   if isinstance(spec, dict) and spec.get("enum") and isinstance(piece, str):
    kind = finish_choice_kinds(spec["enum"]).get(piece)
    if kind:
     return kind
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
  """Set of tool names bound to the order and decline roles, leaving out roles with no tool.
        These calls must travel alone and pass the screen_finish gates."""
  return {self.role("order"), self.role("decline")} - {None}

 def clock_text(self):
  """Status fragment with the seconds left before the LAST_CALL reserve, rounded down to CLOCK_STEP.
        Returns "" until half of that time has been used."""
  left = max(0.0, self.wire.spare() - LAST_CALL)
  if left > (DEADLINE - LAST_CALL) / 2:
   return ""
  rounded = int(left // CLOCK_STEP) * CLOCK_STEP
  if rounded <= 0:
   return f"under {CLOCK_STEP} s left"
  return f"about {rounded} s left"

 def status_head(self, spare):
  """First part of the status line: steps used and left, messages sent (of the cap), the clock, and
        the current requirements text with the reference line and per-product step notes."""
  sent = f"messages sent {self.msgs_sent}"
  if self.msg_cap is not None:
   sent += f" of {self.msg_cap}"
  pieces = [f"[status] steps used {self.wire.tick} of {self.step_cap}, {spare} left", sent]
  clock = self.clock_text()
  if clock:
   pieces.append(clock)
  return "; ".join(pieces) + ".\n[requirements now] " + self.needs.text_body() + self.refdesk.ref_line() + self.cover.product_steps()

 def status_tail(self, spare):
  """Closing urge of the status line, chosen by pressure: two or fewer steps, five or fewer steps or
        under CLOSING seconds, model spend used up, or (before any order, with narrowing) near or past
        order_due(); "" otherwise."""
  several = bool(self.needs.items())
  if spare <= 2:
   target = "every product asked for" if several else "the item"
   return (" LAST STEPS: when the cart already has a candidate for " + target + ", order it"
     "; if not, put in the best candidate (a verified fit, or if there is none, the cheapest in-stock "
     "listing within budget that fails no requirement) and order. Decline only if the records show that "
     "nothing can fit.")
  if spare <= 5 or self.wire.spare() < CLOSING:
   lines = "one line per product asked for" if several else "one line"
   return (" FINISH NOW: set the cart to " + lines
     + " (a verified fit, or if there is none, the cheapest in-stock listing within budget that fails no "
     "requirement; drop any sold-out line), check its stock again, then order. Decline only if the records show "
     "that nothing can fit.")
  if self.wire.spent_out():
   return (" MODEL BUDGET USED: stop exploring; within the next few steps place the best candidate for "
     + ("each product asked for (one line apiece, at its quantity)" if several else "the item asked for")
     + " in the cart (a verified fit if one exists, otherwise the closest fit in the records), re-check its "
     "stock and price, and order. A decline is correct only if the records show that "
     + ("some product has" if several else "the store has") + " no listing able to meet its requirements.")
  if self.orders_tried != 0 or not self.narrowing_now():
   return ""
  due = self.intake.order_due()
  if due < spare <= due + 7:
   return (f" Steps are running low: within the next two or so steps, get your best verified fit into the cart and order; searching and detail reading stop once {due} steps are left, and fixing a refused order takes about 4 steps.")
  if spare <= due:
   return " Searching is over (details only for the cart's line, listings named by a check, or, while a product lacks a verified fit, at most 3 further reads or searches in total): get your best verified fit into the cart, check its stock and price again, and order; a refused order still takes about 4 steps to repair."
  return ""

 def narrowing_now(self):
  """Same as Intake.narrowing: whether the store has both order and cart tools.
        status_tail uses it to decide on the reserve-window hints."""
  return self.intake.narrowing()

 def status_line(self):
  """Full status text shown to the actor: status_head, the usable tools when offered() is narrowed,
        the cart summary once the cart was used, pair_text, go_ahead_text and status_tail."""
  spare = self.moves_left()
  text = self.status_head(spare)
  usable = [entry["function"]["name"] for entry in self.intake.offered()]
  if len(usable) < len(self.toolset):
   text += "\n[tools usable now] " + ", ".join(usable) + " (calls to any other tool are refused and use no step" + (
    "; questions for the shopper are chosen and sent for you" if self.asking else "") + ")."
  if self.trolley.used:
   text += "\n" + self.trolley.summary()
  return text + self.pair_text() + self.go_ahead_text() + self.status_tail(spare)

 def screen(self, proposal: dict[str, Any], permitted: set[str], scripted: bool, batch: int, queued_messages: int) -> tuple[str | None, str, dict[str, Any]]:
  """Vets one proposed call; returns (refusal text or None, tool, arguments).
        Refuses unknown or non-permitted tools (a blocked shopper question is kept via want_ask), bad
        JSON or schema faults (ArgCheck.execute), batch_rules, repeats of a recent lookup, late holds and
        the value guard (actor calls only), then passes cart calls through screen_cart."""
  fn = proposal.get("function") or {}
  tool, raw = fn.get("name"), fn.get("arguments")
  if tool not in self.shapes:
   return f"Not run (no step used): there is no '{tool}' tool here. Choose from: {', '.join(sorted(permitted))}.", tool, {}
  if not scripted and tool not in permitted:
   if tool == self.role("message") and self.asking:
    self.talk.want_ask(raw)
    return ("Not run (no step used): after you read some listings' details, the code plans questions for the shopper "
      "and sends them for you."), tool, {}
   return ("Not run (no step used): not enough steps remain for it; the tools you can still use now are: "
     + ", ".join(sorted(permitted)) + "."), tool, {}
  if isinstance(raw, dict):
   call_args = raw
  else:
   call_args = json_object(raw) if (raw or "").strip() else {}
  if call_args is None:
   return "Not run (no step used): the arguments were not a single JSON object.", tool, {}
  call_args, faults = ArgCheck.execute(call_args, self.shapes[tool])
  if faults:
   return "Not run (no step used): invalid arguments: " + "; ".join(faults[:6]), tool, call_args
  refusal = self.batch_rules(tool, call_args, scripted, batch, queued_messages)
  if refusal:
   return refusal, tool, call_args
  signature = tool + as_text(call_args)
  if tool not in self.finishers() and tool not in (self.role("message"), self.role("cart")) and signature in self.last_calls:
   return ("Not run (no step used): you just made this same call and nothing has changed since; see its result "
     "above. Try other arguments or another action."), tool, call_args
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
  """Batch-level refusals: an order or decline sent with other calls, a message past the budget (with
        the go-ahead messages held back unless confirming), and a cart view that would only repeat the
        newest cart result (once per cart version, with nothing run since and no cart call in the batch)."""
  if tool in self.finishers() and batch > 1:
   return ("Not run (no step used): an order or a decline goes in a call of its own, only after you have read the other "
     "calls' results.")
  if tool == self.role("message") and self.msg_cap is not None and self.msgs_sent + queued_messages >= self.msg_cap - (0 if self.confirming else self.held_back()):
   return "Not run (no step used): the message budget is spent. Do not write to the shopper again."
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
  return ("Not run (no step used): the latest cart result already shows the basket - " + self.trolley.summary()
    + " A cart view adds nothing at all; for a line's current stock, open its record with "
    + (shelf.stock_tool() or "the tool that shows stock") + ".")

 def screen_cart(self, call_args: dict[str, Any], sig: str) -> str:
  """Checks a cart call in order: a put of a listing the reference desk flags as a pointer (up to
        twice), passed_line_hold, a put of a held line on a cart that adds units (once), the cheap-walk
        hold, check_new_lines, rank_lines and unit_rules. Returns the first refusal text or ""."""
  trolley, needs = self.trolley, self.needs
  thing = trolley.aim(call_args)
  putting = trolley.is_put(call_args)
  point = self.refdesk.ref_pointer_note(thing) if thing and putting else ""
  if point and self.pointer_told.get(thing, 0) < 2:
   self.pointer_told[thing] = self.pointer_told.get(thing, 0) + 1
   return f"Not run (no step used): {point}. Pick another listing."
  held_note = self.passed_line_hold(call_args, thing, sig)
  if held_note:
   return held_note
  if not putting or not thing:
   return ""
  if trolley.held(thing) > 0 and trolley.adds is not False and sig not in self.warned:
   self.warned.add(sig)
   return (f"Not run (no step used): the cart already holds {thing} x{int(trolley.held(thing))}, and this cart tool adds units "
     "to lines already held, so this call would increase the count. To order it, do not touch the cart; to change the "
     "units, add or remove only the difference. If you mean to add, send the same call again.")
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
  """Holds, once, a cart call that would remove or reset a line that passed the item check under the
        current needs version. Lets it through when a cheaper put or fit exists for a cheapest request,
        the price moved, units are only lowered, or a refusal came after the line was put in."""
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
  return (f"Not run (no step used): per an independent check, {thing} meets every requirement in force, reading "
    "each property by the value rules (when the store writes the value itself in a field, only that field's words count); "
    "what kind of item it is gets settled by the category label alone, whatever object its title or description "
    f"mentions.{least} Nothing has changed since. If a change or a refusal rules it out, if one of its fields clearly "
    "fails a requirement, or if you are swapping it for a cheaper listing, send the same call again.")

 def check_new_lines(self, call_args: dict[str, Any], thing: str) -> str:
  """Runs Judge.check_item (CHECKER seat) on each listing a cart call puts in, once per listing and
        needs version and only with more than three steps left.
        Returns the first adverse verdict as a refusal that the actor may override by resending."""
  version = self.needs.version
  others = [other for other, count in self.trolley.lines(call_args)[1:] if count is None or count > 0]
  for other in others + [thing]:
   mark = f"{other}|{version}"
   if self.moves_left() <= 3 or mark in self.checked_adds:
    continue
   self.checked_adds.add(mark)
   verdict = self.judge.check_item(other, "ADD TO THE CART")
   if verdict:
    return "Not run (no step used). " + verdict + " If you are sure it fits, send the call once more."
  return ""

 def rank_lines(self, call_args: dict[str, Any]) -> str:
  """For each line a cart call puts in, with more than three steps, asks Judge.rank_check whether a
        better listing may exist and Judge.rank_verify for a cheaper verified one, reworded for the cart.
        For cheapest requests it also refuses once per line and cheaper passed fit (Judge.cheaper_fit)."""
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
     text += (f" An independent item check judged that {cheaper} meets every requirement now in force and is cheaper "
        f"than {line_id}: put {cheaper} in instead.")
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
    return (f"Not run (no step used). Per the independent item check, {cheaper} satisfies every requirement now in force "
      f"and is cheaper than {line_id}. The cheapest fit is wanted, and the check's reading outranks yours: put "
      f"{cheaper} in instead. Should a record show {cheaper} failing a requirement now in force, repeat the same call.")
  return ""

 def unit_rules(self, call_args: dict[str, Any], thing: str, sig: str) -> str:
  """Quantity check on a cart call, once per call signature. Several products: any line count that is
        not among the needed counts; one product: units that differ from what is still needed, taking an
        adding cart's held units into account."""
  trolley, needs = self.trolley, self.needs
  wanted = needs.quantity()
  asked = trolley.count_arg(call_args)
  aim = wanted - trolley.held(thing) if trolley.adds is not False else wanted
  if needs.items():
   odd = [(i, int(c)) for i, c in trolley.lines(call_args) if c is not None and c > 0 and int(c) not in needs.counts()]
   if not odd or sig in self.warned:
    return ""
   self.warned.add(sig)
   return (f"Not run (no step used): the requested products call for {needs.counts()} units (one line each), but this call "
     f"requests {odd}. If you are certain, send the same call again.")
  if asked is None or abs(asked - aim) <= 1e-9 or sig in self.warned:
   return ""
  self.warned.add(sig)
  return (f"Not run (no step used): the request asks for {wanted} units of a single listing, while this call wants "
    f"{int(asked)}. If you are certain, send the same call again.")

 def screen_finish(self, cid: str, call_name: str, call_args: dict[str, Any]) -> tuple[str | None, dict[str, Any]]:
  """Runs the order gates (sold line, basket, pair, cheaper pick, walk, stale, vet, go-ahead) or the
        decline gates (go-ahead, near hold, kept line, slack, fill, named, pages, audits, fit, best, soft
        reference, repair, near check) and returns the first hold with the arguments.
        A decline that passes gets its arguments cleaned by CodeCloser.scrub_decline."""
  kind = self.call_kind(call_name, call_args)
  if kind == "order":
   for gate in (self.sold_line_gate, self.basket_gate, self.pair_gate, self.cheaper_pick_gate, self.walk_gate, self.v73_page_gate, self.stale_gate, self.v75_open_gate, self.vet_gate,
       self.go_ahead_gate):
    held = gate()
    if held:
     return held, call_args
   return None, call_args
  if kind != "decline":
   return None, call_args
  gates = (self.go_ahead_decline_gate, self.shelf.near_hold, self.kept_line_gate, self.v75_fit_gate, self.slack_gate, self.fill_gate, self.named_gate,
     self.pages_gate, self.audit_again_gate, self.first_audit_gate, self.fit_gate, self.best_gate,
     self.soft_ref_gate, self.shelf.decline_repair, self.shelf.near_check)
  for gate in gates:
   held = gate()
   if held:
    return held, call_args
  return None, self.closer.scrub_decline(call_args, call_name)

 def sold_line_gate(self) -> str:
  """Order gate for sold-out cart lines: the first two holds per sold set and cart state show
        Catalogue.dead_note. A third attempt, with enough steps and time, lets CodeCloser.code_cart swap
        the sold lines while keeping the rest and reports the new cart; logs dead_swap."""
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
  return ("Not run yet (no step used). Your cart still had " + ", ".join(dead) + ", now sold out, so the code "
    "took it out, put a candidate in its place, left the other lines as they were, and read the lines' stock: "
    + self.trolley.summary() + " Verify the cart against the request, then order.")

 def cheaper_pick_gate(self) -> str:
  """Order gate for cheapest requests: when Judge.cheaper_fit knows a passed listing cheaper than the
        chosen line (or another cart line), holds the order up to twice per pair and asks for the swap."""
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
  return (f"Not run yet (no step used). Per the independent item check, {cheaper} meets every requirement and is "
    f"cheaper than {thing}. Since the cheapest fit is wanted and the check's reading counts over yours, remove "
    f"{thing} from your cart, add {cheaper}, read its stock and price again, and order it.")

 def walk_gate(self) -> str:
  """Order gate that defers to Coverage.walk_hold for the whole cart, so a cheapest walk still due
        can run before the order; "" for an empty cart."""
  if not self.trolley.units:
   return ""
  return self.cover.walk_hold(None)

 def v73_page_gate(self) -> str:
  """Order gate that defers to Coverage.v73_page_hold (v7.3 C), so the unread next page of the search that
        carries every known requirement is read before a cheapest order; "" for an empty cart."""
  if not self.trolley.units:
   return ""
  return self.cover.v73_page_hold()

 def v75_note_opened(self, tool: str, call_args: dict[str, Any], obs: Any) -> None:
  """Remember the listings a call opened as a full page (v7.5 b, v7.6 F2): each id of a batch-tool call whose
        result shows a full record for it (a property map with two or more plain values, or five short text fields),
        even when that tool also holds the details role; every id named in any other details-role call without error."""
  try:
   if isinstance(obs, dict) and (obs.get("error") or obs.get("call_problem")):
    return
   named = set()
   for value in (call_args or {}).values():
    if isinstance(value, str):
     named.add(value)
    elif isinstance(value, list):
     named.update(item for item in value if isinstance(item, str))
   named &= set(self.archive.ids)
   if not named:
    return
   batch = self.shelf.batch_tool()
   if batch and tool == batch[0]:
    self.v75_opened |= {ident for ident in named if self.shelf.batch_ok([obs], [ident])}
   elif tool == self.role("details"):
    self.v75_opened |= named
  except Exception as err:
   self.log.append({"kind": "v75_open_error", "error": f"{type(err).__name__}: {err}"[:300]})

 def v75_unopened(self) -> list[str]:
  """Cart lines and reference-chain pages (run.refs chains) that no full-page call has opened yet (v7.5 b).
        Only ids the archive knows count."""
  lines = [line for line, units in self.trolley.units.items() if units]
  pages = [page for ref in self.refs or [] for page in ref.get("chain") or []]
  known = self.archive.ids
  return [ident for ident in dict.fromkeys(lines + pages) if ident in known and ident not in self.v75_opened]

 def v75_open_gate(self) -> str:
  """Order gate (v7.5 b): when a cart line or a reference-chain page was never opened as a full page, holds the
        order at no step and queues those ids for the code to open with the details role (batch when proven).
        Once per missing set and cart state; only when the reads, the order and, while it is still due, the
        go-ahead message (three moves) fit in the steps left (else logs v75_open_short and lets the order pass)."""
  try:
   missing = self.v75_unopened()
   batch = self.shelf.batch_tool() if self.batch_proven else None
   if not missing or not (self.role("details") or batch):
    return ""
   mark = (tuple(missing), self.cart_seq)
   if mark in self.v75_open_marks:
    return ""
   self.v75_open_marks.add(mark)
   due = 3 if self.procedures and self.role("message") and self.cart_mark() != self.confirmed_cart else 1
   reads = self.read_calls(missing, batch)
   if self.moves_left() < reads + due + 1 or self.wire.spare() <= CLOSING + 30:
    self.log.append({"kind": "v75_open_short", "ids": missing[:6], "moves": self.moves_left(), "reads": reads})
    return ""
   queue = self.reads
   queue.open_queue = list(missing)
   queue.open_recheck = True
   queue.open_purpose = "opened"
   queue.open_tool = None
   queue.open_say = "Before the order, the code opens with the full-page tool each cart line and reference page not opened that way yet."
   self.log.append({"kind": "v75_open_hold", "ids": missing[:6]})
   return ("Not run yet (no step used). Before an order, every listing in the cart and every page a reference led to "
     "must have been opened with the full-page details tool; not yet: " + ", ".join(missing[:6]) + ". The code "
     "opens them for you now; then order.")
  except Exception as err:
   self.log.append({"kind": "v75_open_error", "error": f"{type(err).__name__}: {err}"[:300]})
   return ""

 def v75_fit_gate(self) -> str:
  """Decline gate (v7.5 c): holds a decline, naming the lines, while a verified basket stands: every cart line
        passed the item check for the requirements now in force, none is refused or sold out, the basket is complete
        and its known total is within the cap. Once per (lines, requirements version), and not for a state
        kept_line_gate already held; logs v75_fit_stands."""
  if self.moves_left() <= 1:
   return ""
  version = self.needs.version
  dead = self.shelf.dead_lines()
  lines = [line for line, units in self.trolley.units.items() if units]
  verified = [line for line in lines if f"{line}|{version}" in self.passed and line not in self.refused_lines
     and line not in dead]
  if not lines or len(verified) != len(lines) or self.basket_short():
   return ""
  cap = self.needs.spend_cap
  prices = [self.archive.price(line) for line in lines]
  if is_number(cap):
   if not all(is_number(price) for price in prices):
    return ""
   if sum(price * self.trolley.units[line] for price, line in zip(prices, lines)) > cap:
    return ""
  mark = (tuple(sorted(verified)), version)
  if mark in self.v75_fit_marks:
   return ""
  self.v75_fit_marks.add(mark)
  self.log.append({"kind": "v75_fit_stands", "lines": verified, "version": version})
  every = " with each product's line in it" if self.needs.items() else ""
  return ("Not run yet (no step used). A fit stands: " + ", ".join(verified) + " passed the independent check against "
    "every requirement now in force, and nothing since (a refusal, a sell-out, the budget, a new need) has ruled "
    "it out. Read its stock and price again and order the cart" + every + "; decline only if a change now clearly "
    "excludes it.")

 def stale_gate(self) -> str:
  """Order gate for stock freshness: when some cart lines were not read after the latest cart edit and a
        stock-showing tool exists, holds once per cart state and queues up to four of those lines on
        run.reads for the code to re-read with that tool (purpose "cart")."""
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
  return ("Not run yet (no step used). Prices and stock can change silently, and these cart lines have not been read "
    "since the cart last changed: " + ", ".join(stale[:4]) + f". The code now reads their stock for you ({reader}, the tool "
    "that shows stock); next, order if each is still in stock and the total is within budget, otherwise replace the line.")

 def vet_gate(self) -> str:
  """One-time order vet, reset by cart changes and refusals: cart count faults first, then
        Judge.check_item on the lines (or the chosen one), then rank_check and rank_verify.
        When all of that finds nothing, it still holds once with a final checklist for the actor."""
  if self.order_vetted or self.moves_left() <= 2 or self.wire.spare() <= CLOSING:
   return ""
  self.order_vetted = True
  judge, needs = self.judge, self.needs
  faults = self.trolley.faults(needs.quantity(), needs.counts())
  if faults and self.moves_left() > 3:
   return "Not run yet (no step used). By count, the cart is not ready: " + "; ".join(faults) + ". Fix it, then order."
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
    extra = (f" An independent item check judged that {cheaper} meets every requirement now in force and is cheaper "
       f"than {thing}: remove {thing}, add {cheaper}, read its stock and price again, and order.")
   return "Not run yet (no step used). " + better + extra
  scope = ("for each product asked for, at its quantity" if needs.items()
     else "of the chosen item at the needed quantity")
  return ("Not run yet (no step used). Before ordering, confirm that (1) the cart holds exactly one line " + scope + ", with "
    "no other lines; (2) since the last cart change you have read this item's stock and price again with a tool that shows "
    "them, it is in stock, and price times quantity is within the current budget; (3) the properties of the item itself meet "
    "every stated requirement, every need the shopper revealed and every change announced; (4) if the cheapest or best "
    "was asked for, nothing you found is a better fit; (5) every procedural rule of the request was "
    "followed" + (" (calling the order tool makes the code ask the shopper to approve the cart)" if self.procedures else "")
    + ". Verify whatever is still unchecked now; if all of it holds, call the order tool again.")

 def kept_line_gate(self) -> str:
  """Decline gate: once, when the cart already holds lines that passed the item check this version
        (not refused, not sold out), holds the decline and tells the actor to re-check and order them."""
  dead = self.shelf.dead_lines
  version = self.needs.version
  verified = [i for i in self.trolley.units
     if f"{i}|{version}" in self.passed and i not in self.refused_lines and i not in dead()]
  if not verified or self.moves_left() <= 1 or self.decline_kept:
   return ""
  self.decline_kept = True
  self.v75_fit_marks.add((tuple(sorted(verified)), version))
  every = " with each product's line in your cart" if self.needs.items() else ""
  return ("Not run yet (no step used). Your cart has " + ", ".join(verified) + ", which the independent check judged to "
    "satisfy every requirement now in force. A decline wastes that: read its stock and price again, then order it" + every
    + ". Decline only if some later change now clearly excludes it.")

 @staticmethod
 def read_calls(rows: list[str], batch: Any) -> int:
  """Steps needed to open rows: with a batch details tool and two or more rows, the row count divided
        by its per-call limit, rounded up; otherwise one step per row."""
  if batch and len(rows) >= 2:
   return -(-len(rows) // batch[2])
  return len(rows)

 def slack_gate(self) -> str:
  """Decline gate for multi-product budgets: once, when the cheapest fits together exceed the cap and
        Catalogue.slack_rows finds unopened rows that might close the gap, queues them for code reads
        and explains the shortfall; needs enough steps for the reads and spare time."""
  shelf = self.shelf
  slack = []
  if not self.slack_done and (self.role("details") or shelf.batch_tool()):
   slack = shelf.slack_rows()
  spread = shelf.batch_tool()
  if not slack or self.moves_left() <= self.read_calls(slack, spread) + 4 or self.wire.spare() <= CLOSING + 40:
   return ""
  self.slack_done = True
  self.reads.open_queue = slack
  self.reads.open_say = "Before any decline, I open cheaper rows that could let every product fit the budget at once."
  total = shelf.slack_total
  shown = f"{total:g}" if is_number(total) else "more than the budget"
  return ("Not run yet (no step used). Every product has a fitting listing, but the cheapest fits add up to " + shown
    + f", over the budget of {self.needs.spend_cap}. Nobody opened the rows priced within what the budget leaves a product "
    "once the other products' cheapest fits are paid, so the code reads their records now: " + ", ".join(slack)
    + ". Test each against its own product's requirements by the value rules (when the store writes the value "
    "itself in a field, only that field's words count); as soon as fits within budget cover every product at once, "
    "add each product's line to the cart and order; decline only when no such combination exists.")

 def fill_gate(self) -> str:
  """Decline gate: once, queues unread in-budget listings of products with no verified fit yet
        (Catalogue.lacking_rows) for code reads before a decline may go through."""
  shelf = self.shelf
  batch = shelf.batch_tool()
  fill = shelf.lacking_rows() if not self.decline_fill and (self.role("details") or batch) else []
  if not fill or self.moves_left() <= self.read_calls(fill, batch) + 3 or self.wire.spare() <= CLOSING + 40:
   return ""
  self.decline_fill = True
  self.reads.open_queue = fill
  self.reads.open_say = "Before any decline, I open unread listings of the products still lacking a verified fit."
  return ("Not run yet (no step used). Some requested products have no verified fit yet, and in-budget listings from their "
    "categories were never opened, so the code is reading their records now, ahead of any decline: "
    + ", ".join(fill) + ". Test each against its own product's requirements by the value rules (when the store writes the value "
    "itself in a field, only that field's words count); once each product has a fitting listing, add every "
    "product's line to the cart and order; decline only when some product really has none.")

 def named_gate(self) -> str:
  """Decline gate: once, queues listings the checks named but nobody opened (Judge.named_unread, up
        to the batch limit or three) for code reads, and asks the actor to weigh them before declining."""
  batch = self.shelf.batch_tool()
  queue = self.judge.named_unread()[: batch[2] if batch else 3]
  if not queue or self.decline_opens or not (self.role("details") or batch):
   return ""
  if self.moves_left() <= self.read_calls(queue, batch) + 3 or self.wire.spare() <= CLOSING + 40:
   return ""
  self.decline_opens = True
  self.reads.open_queue = queue
  return ("Not run yet (no step used). The listings the checks named were never opened, so the code fetches their records "
    "now, ahead of any decline: " + ", ".join(queue) + ". Judge each by the value rules against the requirements "
    "(the field carrying the property's name decides, and when the store writes the value itself there, only that field's "
    "words count; lacking such a field, the title, the options list or another field can hold the value), buy a fitting "
    "one, and decline only if all of them fail.")

 def pages_gate(self) -> str:
  """Decline gate around CheapWalk.decline_reads, which queues unread search pages and cheaper rows
        for code reads before a decline; holds with its note when it has one."""
  note = self.walk.decline_reads()
  return "Not run yet (no step used). " + note if note else ""

 def fit_gate(self) -> str:
  """Decline gate around CheapWalk.fit_hold: when the decline audit's possible fits pass the item
        check, the decline is held with a note that points at them."""
  note = self.walk.fit_hold()
  return "Not run yet (no step used). " + note if note else ""

 def audit_again_gate(self) -> str:
  """Decline gate that reruns the decline audit (Judge.review_decline, AUDITOR seat) after the first
        one, when more listings were read since and fewer than two audits ran; holds with its note."""
  judge = self.judge
  limit = 10 ** 9 if judge.audit_reads is None else judge.audit_reads
  fresh_reads = len(self.archive.read) > limit and judge.audits < 2
  if not fresh_reads or not self.decline_vetted or self.moves_left() <= 3 or self.wire.spare() <= CLOSING + 30:
   return ""
  note = judge.review_decline()
  return "Not run yet (no step used). " + note if note else ""

 def first_audit_gate(self) -> str:
  """Decline gate for the first decline: sets decline_vetted and runs the decline audit (AUDITOR seat).
        Holds with the audit's note, or with a generic be-certain note when the audit raises nothing."""
  if self.decline_vetted or self.moves_left() <= 3 or self.wire.spare() <= CLOSING + 30:
   return ""
  self.decline_vetted = True
  note = self.judge.review_decline()
  if note:
   return "Not run yet (no step used). " + note
  return ("Not run yet (no step used). Before declining, be sure: you varied your searches (other words, filters, pages), "
    "you checked each candidate's own properties, and every candidate fails on budget, stock, a stated requirement "
    "or a revealed need. A property's field counts whatever its key's spelling. If something fits, buy it instead; "
    "if nothing does, call the decline tool again.")

 def best_gate(self) -> str:
  """Decline gate: takes CheapWalk.unrefuted candidates (in stock, in budget, ruled out by no check),
        item-checks up to three unchecked ones, then once holds the decline naming those that remain."""
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
  return ("Not run yet (no step used). Declining is right only if no listing can satisfy the requirements, yet the "
    "records contain " + ("a candidate for every product asked for" if several else "a candidate") + " with "
    "stock, within the budget and excluded by no check: " + ", ".join(best) + ". Unless a record shows that one of "
    "them fails a requirement now in force, put " + ("one line per product, each at its quantity," if several
    else "it") + " in the cart, read stock and price again, and order; decline only if every candidate clearly fails.")

 def soft_ref_gate(self) -> str:
  """Decline gate: once, when a "differ" reference requirement is still open or stuck, explains that
        its value is unknown and cannot justify a decline, and asks for an order on the other needs."""
  soft = [r for r in self.refs or [] if r.get("relation") == "differ" and r.get("status") in ("open", "stuck")
    and self.refdesk.ref_live(r)]
  if not soft or self.moves_left() <= 3 or self.ref_decline_held:
   return ""
  self.ref_decline_held = True
  heads = ", ".join(dict.fromkeys(r["head"] for r in soft))
  per_product = ", one line per product" if self.needs.items() else ""
  return ("Not run yet (no step used). " + heads + " depends on another listing that the code could not identify, so its value "
    "is unknown: it neither excludes a listing nor justifies a decline. Put the listing that meets every other "
    "requirement (in stock, within budget) into the cart" + per_product + ", check its stock and price once more, and "
    "order; decline only if no listing meets the other requirements.")

 def note_batch_puts(self, proposals: list[dict[str, Any]]) -> None:
  """Pre-scans a batch: records whether it holds any cart call (batch_cart) and the listing ids being
        put in (batch_puts), which passed_line_hold and batch_rules consult."""
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
  """Runs one batch of proposed calls and returns (call id, result text) pairs in proposal order.
        Screens each call against offered() (a refused removal blocks later cart calls), gates a leading
        order or decline via screen_finish, keeps the last step for a finisher, sends in chunks of 16.
        scripted=True marks code-made calls, which skip the permission check and late holds."""
  answers: list[tuple[str, str]] = []
  self.intake.begin_batch()
  permitted = {tdef["function"]["name"] for tdef in self.intake.offered()}
  outgoing: list[tuple[str, str, dict[str, Any]]] = []
  messages_queued = 0
  cart_blocked = False
  self.note_batch_puts(proposals)
  for proposal in proposals:
   if cart_blocked and proposal["function"]["name"] == self.role("cart"):
    answers.append((proposal["id"], "Not run (no step used): an earlier cart change in this reply did not go through; "
            "see why, then resend the cart changes."))
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
    answers.append((cid, "Not run (no step used): a single step remains, and it must go to the order (or else the decline)."))
   outgoing = outgoing[:room]
  for start in range(0, len(outgoing), 16):
   slab = outgoing[start: start + 16]
   if self.wire.done or self.wire.broke or self.moves_left() <= 0:
    answers.extend((cid, "Not run: the episode is over or no steps remain.") for cid, _, _ in slab)
    continue
   answers.extend(self.send_chunk(slab[: self.moves_left()]))
  position = {c["id"]: i for i, c in enumerate(proposals)}
  answers.sort(key=lambda answer: position.get(answer[0], 0))
  return answers

 def send_chunk(self, slab: list[tuple[str, str, dict[str, Any]]]) -> list[tuple[str, str]]:
  """Sends a group of screened calls as one world action (Wire.act), logs it, and turns each returned
        observation into actor text via absorb; gives retry or not-run text when replies are missing."""
  actions = [{"name": tool, "args": call_args} for _, tool, call_args in slab]
  step_before = self.wire.tick
  reply = self.wire.act(actions)
  self.log.append({"kind": "env", "step": step_before + 1, "actions": actions, "reply": reply})
  if reply is None:
   return [(cid, "The call failed (network/server); retry it.") for cid, _, _ in slab]
  if isinstance(reply.get("observations"), list):
   observations = reply.get("observations")
  else:
   observations = [reply.get("observation")]
  texts = []
  for index, (cid, tool, call_args) in enumerate(slab):
   if index >= len(observations):
    texts.append((cid, "Not run: the episode ended before this call."))
    continue
   texts.append((cid, self.absorb(tool, call_args, observations[index], reply, index == len(observations) - 1)))
  return texts

 def book_call(self, tool: str, call_args: dict[str, Any], obs: Any) -> None:
  """Bookkeeping for each world result: counts messages and order attempts, sets last_kind, keeps the
        last two lookup signatures (cleared by cart, message and finishing calls), adds the observation
        to the archive and counts calls since the last cart change."""
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
  """Marks cart lines whose records appear in an observation as freshly read (line_read, line_seq),
        which stale_gate relies on. Where the store shows stock, the record must carry a stock state
        (and, unless search rows are trusted, be an id the call asked for); else nested fields suffice."""
  shelf = self.shelf
  held = self.trolley.units
  stock_rule = bool(shelf.stock_seen)
  asked = shelf.asked_ids(call_args)
  rows_trusted = bool(shelf.rows_live) and not shelf.rows_lag
  lookups_only = not rows_trusted and tool not in (self.role("cart"), self.role("cart_view"))
  fresh: set[str] = set()

  def spot(record: dict[str, Any]) -> None:
   """walk_dicts visitor: when a record names a cart line in an id field and meets the stock rule,
            adds every cart line it mentions to the fresh set."""
   named = {v for k, v in record.items() if isinstance(v, str) and v in held and "id" in str(k).lower()}
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
  """Adds opened ids to archive.read after a successful details call. Once fewer than order_due()
        steps remain, counts a details, search or other lookup call toward late_n, unless the code is
        covering or the call only reads cart lines (or, for other lookups, passed lines)."""
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
  """After a cart call: clears order_vetted, stamps the change, appends CartBook.booked text and
        updates the Catalogue cart state. A put records refusals_seen and the put count, and a passed
        line put in gets a remark that the item check approved it."""
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
   remarks.append(f"An independent check judged {item} meets every requirement now in force, with each property read by the "
         "value rules (when this store writes the value itself in a field, only that field's words count); leave it in "
         "unless a later change excludes it.")

 def absorb(self, call_name: str, call_args: dict[str, Any], obs: Any, resp: dict[str, Any], last: bool) -> str:
  """Turns one observation into the text the actor reads, updating state on the way: bookkeeping,
        stock learning with sell-out notes, reference and cart remarks, refusal detection, observer and
        search notes, an early decline audit, and shopper lines heard by Needs.hear.
        Returns the clipped observation (shorter for searches) with a note tail."""
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
  self.v75_note_opened(call_name, call_args, obs)
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
  """True once per run after a lookup call (not cart, message or finishing) when steps left are above
        FIRST_ORDER_BY but within a third of the cap plus two, the cart is empty and nothing passed the
        item check under the current needs version."""
  if self.early_audit or tool in (self.role("cart"), self.role("message")) or tool in self.finishers():
   return False
  if not FIRST_ORDER_BY < self.moves_left() <= self.step_cap // 3 + 2 or self.trolley.units:
   return False
  version = str(self.needs.version)
  return not any(key.rpartition("|")[2] == version for key in self.passed)

 def tool_calls(self, msg: dict[str, Any], seq_no: int) -> tuple[list[dict[str, Any]], int]:
  """Converts a model reply's tool_calls into proposal dicts with ids (call_N when missing) and
        string arguments; returns them with the updated sequence number."""
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
  """Reads the request (Intake.read_request), logs the plan, and builds the actor chat: ACT_RULES as
        the system text, then the request, any opening observation (also added to the archive), the
        needs overview and the status line."""
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
  """True when no further turn should run: the episode finished, the wire is broke (payment refused),
        no steps are left, or fewer than LAST_CALL seconds remain."""
  wire = self.wire
  if wire.done or wire.broke or self.moves_left() <= 0:
   return True
  return wire.spare() < LAST_CALL

 def execute(self) -> list[dict[str, Any]]:
  """Runs the episode: open_chat, then up to TURN_LIMIT turns, each after CodeCloser.catch_up, taking
        the first scripted step that acts (shopper reply, queued reads, sweep, value pages, planned
        question, reference hop, cheap walk) or else an actor turn, which can end the loop.
        Finishes via CodeCloser.finish_now if still open, logs an end record and returns the log."""
  if self.wire.done:
   return self.log
  self.open_chat()
  scripted = (ShopperReplyStep(), QueuedReadsStep(), ShelfSweepStep(), ValuePagesStep(),
     PlannedQuestionStep(), ReferenceHopStep(), CheapWalkStep())
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
 """Entry point: builds a ShoppingRun from the payload and returns the log from execute().
    On a crash it logs the error, still tries CodeCloser.finish_now, and returns the log."""
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
